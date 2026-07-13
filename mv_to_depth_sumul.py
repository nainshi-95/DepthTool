#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Core patch for depth RDO:
- no-signal reuse from downsampled depth buffer
- predictor-only mode
- predictor+ABC residual mode
- direct ABC mode
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def ceil_log2(v: int) -> int:
    return 0 if v <= 1 else int(math.ceil(math.log2(v)))


def signed_to_code_num(v: int) -> int:
    if v == 0:
        return 0
    return 2 * v - 1 if v > 0 else -2 * v


def exp_golomb_len_unsigned(v: int) -> int:
    if v < 0:
        raise ValueError("negative ue(v)")
    return 2 * int(math.floor(math.log2(v + 1))) + 1


def exp_golomb_len_signed(v: int) -> int:
    return exp_golomb_len_unsigned(signed_to_code_num(int(v)))


def qp_scale(qp: int, qp_ref: int = 32) -> float:
    return 2.0 ** ((float(qp) - float(qp_ref)) / 6.0)


@dataclass(frozen=True)
class QSteps:
    qa: float
    qb: float
    qc: float


def qsteps_from_qp(qp: int, qa0: float, qb0: float, qc0: float, qp_ref: int = 32) -> QSteps:
    s = qp_scale(qp, qp_ref)
    return QSteps(qa0 * s, qb0 * s, qc0 * s)


def quantize(v: float, step: float) -> int:
    return int(np.rint(float(v) / float(step)))


def dequantize(q: int, step: float) -> float:
    return float(q) * float(step)


@dataclass
class Plane:
    a: float
    b: float
    c: float


@dataclass
class Candidate:
    name: str
    depth: np.ndarray
    valid: np.ndarray


@dataclass
class RDOResult:
    mode: str
    candidate_name: str
    candidate_idx: int
    residual_present: bool
    qda: int
    qdb: int
    qdc: int
    bits: float
    distortion: float
    cost: float
    recon: np.ndarray
    plane: Optional[Plane]


def local_grid(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(w, dtype=np.float64) - (w - 1) * 0.5
    ys = np.arange(h, dtype=np.float64) - (h - 1) * 0.5
    return np.meshgrid(xs, ys)


def fit_inverse_depth_plane(depth: np.ndarray, depth_eps: float, valid_mask=None) -> Optional[Plane]:
    z = np.asarray(depth, dtype=np.float64)
    h, w = z.shape
    xx, yy = local_grid(w, h)
    valid = np.isfinite(z) & (z >= depth_eps)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    n = int(np.count_nonzero(valid))
    if n < 3:
        return None
    A = np.stack([xx[valid], yy[valid], np.ones(n)], axis=1)
    b = 1.0 / z[valid]
    try:
        coeff, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.isfinite(coeff).all():
        return None
    return Plane(*map(float, coeff))


def fit_constant_inverse_depth(depth: np.ndarray, depth_eps: float, valid_mask=None) -> Optional[Plane]:
    z = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(z) & (z >= depth_eps)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return None
    c = float(np.mean(1.0 / z[valid]))
    return Plane(0.0, 0.0, c) if np.isfinite(c) and c > 0 else None


def render_plane(p: Plane, w: int, h: int, depth_eps: float, max_value: float):
    xx, yy = local_grid(w, h)
    inv = p.a * xx + p.b * yy + p.c
    valid = np.isfinite(inv) & (inv > 0.0)
    inv = np.clip(inv, 1.0 / max(max_value, 1.0), 1.0 / max(depth_eps, 1e-12))
    out = np.zeros((h, w), dtype=np.float64)
    out[valid] = 1.0 / inv[valid]
    return out, valid


def block_sse(gt: np.ndarray, recon: np.ndarray, valid=None) -> float:
    d = np.asarray(gt, dtype=np.float64) - np.asarray(recon, dtype=np.float64)
    m = np.isfinite(d)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not np.any(m):
        return float("inf")
    return float(np.sum(d[m] * d[m]))


class DepthReuseBuffer:
    def __init__(self, frame_w: int, frame_h: int, scale: int, depth_eps: float):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.scale = scale
        self.depth_eps = depth_eps
        self.w = (frame_w + scale - 1) // scale
        self.h = (frame_h + scale - 1) // scale
        self.depth = np.zeros((self.h, self.w), dtype=np.float64)
        self.valid = np.zeros((self.h, self.w), dtype=bool)

    def cell_rect(self, x: int, y: int, w: int, h: int):
        x0 = max(0, x // self.scale)
        y0 = max(0, y // self.scale)
        x1 = min(self.w, (x + w + self.scale - 1) // self.scale)
        y1 = min(self.h, (y + h + self.scale - 1) // self.scale)
        return x0, y0, x1, y1

    def can_reuse(self, x: int, y: int, w: int, h: int) -> bool:
        x0, y0, x1, y1 = self.cell_rect(x, y, w, h)
        return x1 > x0 and y1 > y0 and bool(np.all(self.valid[y0:y1, x0:x1]))

    def reconstruct(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0, y0, x1, y1 = self.cell_rect(x, y, w, h)
        cells = self.depth[y0:y1, x0:x1]
        if cells.size == 0 or not np.all(self.valid[y0:y1, x0:x1]):
            raise RuntimeError("invalid no-signal reuse")
        return cv2.resize(cells.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    def commit(self, x: int, y: int, recon: np.ndarray) -> None:
        h, w = recon.shape
        x0, y0, x1, y1 = self.cell_rect(x, y, w, h)
        tw, th = x1 - x0, y1 - y0
        if tw <= 0 or th <= 0:
            return
        down = cv2.resize(recon.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
        good = np.isfinite(down) & (down >= self.depth_eps)
        self.depth[y0:y1, x0:x1][good] = down[good]
        self.valid[y0:y1, x0:x1][good] = True


def candidate_idx_bits(idx: int, n: int, coding: str = "truncated_unary") -> float:
    if n <= 1:
        return 0.0
    if coding == "fixed":
        return float(ceil_log2(n))
    if coding == "truncated_unary":
        return float(idx + (0 if idx == n - 1 else 1))
    raise ValueError(coding)


def mode_bits(mode: str) -> float:
    # direct_flag + residual_flag model
    if mode == "buffer_reuse":
        return 0.0
    if mode == "direct":
        return 1.0
    if mode in ("predictor_only", "predictor_residual"):
        return 2.0
    raise ValueError(mode)


def eval_predictor_only(gt: np.ndarray, cand: Candidate, idx: int, n: int, lam: float, coding: str) -> RDOResult:
    d = block_sse(gt, cand.depth, cand.valid)
    b = mode_bits("predictor_only") + candidate_idx_bits(idx, n, coding)
    return RDOResult("predictor_only", cand.name, idx, False, 0, 0, 0, b, d, d + lam * b, cand.depth.copy(), fit_inverse_depth_plane(cand.depth, 1e-12, cand.valid))


def eval_predictor_residual(gt: np.ndarray, gt_plane: Plane, cand: Candidate, idx: int, n: int, q: QSteps, lam: float, depth_eps: float, max_value: float, coding: str) -> Optional[RDOResult]:
    pp = fit_inverse_depth_plane(cand.depth, depth_eps, cand.valid)
    if pp is None:
        pp = fit_constant_inverse_depth(cand.depth, depth_eps, cand.valid)
    if pp is None:
        return None
    qda = quantize(gt_plane.a - pp.a, q.qa)
    qdb = quantize(gt_plane.b - pp.b, q.qb)
    qdc = quantize(gt_plane.c - pp.c, q.qc)
    rp = Plane(pp.a + dequantize(qda, q.qa), pp.b + dequantize(qdb, q.qb), pp.c + dequantize(qdc, q.qc))
    recon, valid = render_plane(rp, gt.shape[1], gt.shape[0], depth_eps, max_value)
    d = block_sse(gt, recon, valid)
    b = mode_bits("predictor_residual") + candidate_idx_bits(idx, n, coding) + exp_golomb_len_signed(qda) + exp_golomb_len_signed(qdb) + exp_golomb_len_signed(qdc)
    return RDOResult("predictor_residual", cand.name, idx, True, qda, qdb, qdc, b, d, d + lam * b, recon, rp)


def eval_direct(gt: np.ndarray, gt_plane: Plane, q: QSteps, lam: float, depth_eps: float, max_value: float) -> RDOResult:
    qa = quantize(gt_plane.a, q.qa)
    qb = quantize(gt_plane.b, q.qb)
    qc = quantize(gt_plane.c, q.qc)
    rp = Plane(dequantize(qa, q.qa), dequantize(qb, q.qb), dequantize(qc, q.qc))
    recon, valid = render_plane(rp, gt.shape[1], gt.shape[0], depth_eps, max_value)
    d = block_sse(gt, recon, valid)
    b = mode_bits("direct") + exp_golomb_len_signed(qa) + exp_golomb_len_signed(qb) + exp_golomb_len_signed(qc)
    return RDOResult("direct", "none", -1, True, qa, qb, qc, b, d, d + lam * b, recon, rp)


def evaluate_block(gt: np.ndarray, candidates: Sequence[Candidate], q: QSteps, lam: float, depth_eps: float, max_value: float, coding: str = "truncated_unary") -> RDOResult:
    gt_plane = fit_inverse_depth_plane(gt, depth_eps)
    if gt_plane is None:
        gt_plane = fit_constant_inverse_depth(gt, depth_eps)
    if gt_plane is None:
        zero = np.zeros_like(gt)
        d = block_sse(gt, zero)
        return RDOResult("direct", "none", -1, False, 0, 0, 0, 1.0, d, d + lam, zero, None)

    results: List[RDOResult] = [eval_direct(gt, gt_plane, q, lam, depth_eps, max_value)]
    n = len(candidates)
    for idx, cand in enumerate(candidates):
        results.append(eval_predictor_only(gt, cand, idx, n, lam, coding))
        r = eval_predictor_residual(gt, gt_plane, cand, idx, n, q, lam, depth_eps, max_value, coding)
        if r is not None:
            results.append(r)
    return min(results, key=lambda r: r.cost)


def process_block_with_reuse_buffer(
    gt_block: np.ndarray,
    x: int,
    y: int,
    candidates: Sequence[Candidate],
    reuse_buffer: DepthReuseBuffer,
    qsteps: QSteps,
    lambda_rd: float,
    depth_eps: float,
    max_value: float,
    candidate_coding: str = "truncated_unary",
) -> RDOResult:
    h, w = gt_block.shape

    # Decoder-deterministic no-signal path.
    if reuse_buffer.can_reuse(x, y, w, h):
        recon = reuse_buffer.reconstruct(x, y, w, h)
        valid = np.isfinite(recon) & (recon >= depth_eps)
        d = block_sse(gt_block, recon, valid)
        return RDOResult(
            mode="buffer_reuse",
            candidate_name="depth_buffer",
            candidate_idx=-1,
            residual_present=False,
            qda=0,
            qdb=0,
            qdc=0,
            bits=0.0,
            distortion=d,
            cost=d,
            recon=recon,
            plane=fit_inverse_depth_plane(recon, depth_eps, valid),
        )

    result = evaluate_block(
        gt=gt_block,
        candidates=candidates,
        q=qsteps,
        lam=lambda_rd,
        depth_eps=depth_eps,
        max_value=max_value,
        coding=candidate_coding,
    )

    # Commit only after winner is fixed. Encoder/decoder update identically.
    reuse_buffer.commit(x, y, result.recon)
    return result


# Integration example inside the raster block loop:
#
# reuse_buffer = DepthReuseBuffer(width, height, scale=args.depth_buffer_scale,
#                                 depth_eps=args.depth_eps)
# qsteps = qsteps_from_qp(args.qp, args.qa_base, args.qb_base,
#                         args.qc_base, args.qp_ref)
#
# for by in range(0, height, block_size):
#     for bx in range(0, width, block_size):
#         gt_block = gt_depth[by:by+bh, bx:bx+bw]
#         candidates = make_your_candidates(...)
#         best = process_block_with_reuse_buffer(
#             gt_block, bx, by, candidates, reuse_buffer, qsteps,
#             args.lambda_rd, args.depth_eps, args.max_value,
#             args.candidate_coding)
#         recon[by:by+bh, bx:bx+bw] = best.recon
#
# Required argparse additions:
#   --depth-buffer-scale 4
#   --qp 47
#   --qp-ref 32
#   --qa-base 1e-6
#   --qb-base 1e-6
#   --qc-base 1e-4
#   --candidate-coding {fixed,truncated_unary}
