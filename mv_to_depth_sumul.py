#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-block depth predictor candidate RDO simulator.

Inputs
------
- GT depth YUV420p10le: Y is depth code, z = Y * depth_scale_real.
- camParam JSONL.
- Optional MV CSV: poc,x,y,w,h,list,ref_poc,mv_x,mv_y.

For every fixed block, the encoder fits the GT inverse-depth plane
  1/z = a*(x-cx) + b*(y-cy) + c
and evaluates these predictors without block split:
  direct_zero, left, top, top_left, spatial_all,
  fw_ref_<POC>, fw_average.

Forward-warp candidates are built once per reference/current pair in a
low-resolution buffer. Reference depth is always GT depth, by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Camera:
    poc: int
    K: np.ndarray
    W2C: np.ndarray
    C2W: np.ndarray
    z_sign: float


@dataclass
class Plane:
    a: float
    b: float
    c: float
    cx: float
    cy: float


@dataclass
class MVRecord:
    poc: int
    x: int
    y: int
    w: int
    h: int
    list_id: str
    ref_poc: int
    mv_x: float
    mv_y: float


@dataclass
class CandidateResult:
    name: str
    pred: Plane
    recon: Plane
    q: Tuple[int, int, int]
    candidate_bits: int
    residual_bits: int
    bits: int
    sse: float
    cost: float
    recon_y: np.ndarray


def ceil_log2(v: int) -> int:
    return 0 if v <= 1 else int(math.ceil(math.log2(v)))


def signed_to_code_num(v: int) -> int:
    return 0 if v == 0 else (2 * v - 1 if v > 0 else -2 * v)


def se_bits(v: int) -> int:
    u = signed_to_code_num(int(v))
    return 2 * int(math.floor(math.log2(u + 1))) + 1


def frame_size(w: int, h: int) -> int:
    return w * h * 3


def count_frames(path: str, w: int, h: int) -> int:
    fs = frame_size(w, h)
    size = os.path.getsize(path)
    if size % fs:
        print(f"[WARN] trailing bytes ignored: {size % fs}")
    return size // fs


def read_y(fp, idx: int, w: int, h: int) -> np.ndarray:
    fp.seek(idx * frame_size(w, h))
    raw = fp.read(w * h * 2)
    if len(raw) != w * h * 2:
        raise EOFError(f"Cannot read frame {idx}")
    return np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float64)


def write_depth_frame(fp, y: np.ndarray, w: int, h: int) -> None:
    y16 = np.clip(np.rint(y), 0, 1023).astype("<u2")
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    fp.write(np.ascontiguousarray(y16).tobytes())
    fp.write(uv.tobytes())
    fp.write(uv.tobytes())


def rt4(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def load_cameras(path: str) -> Tuple[Dict[str, Any], Dict[int, Camera]]:
    header = None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") in ("header", "intrinsic"):
                header = o
            elif "poc" in o:
                records.append(o)
    if header is None or not records:
        raise RuntimeError("Invalid camera JSONL")

    records.sort(key=lambda x: int(x["poc"]))
    intr = header["intrinsic"]
    base = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]], np.float64)
    fixed = header.get("intrinsic_mode") == "rap_fixed" or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    z_sign = 1.0 if float(intr.get("z_sign", 1.0)) >= 0 else -1.0
    pose_mode = str(header.get("pose_mode", "current_to_previous"))

    cur = base.copy()
    prev_w2c = np.eye(4, dtype=np.float64)
    cams: Dict[int, Camera] = {}
    for order, rec in enumerate(records):
        poc = int(rec["poc"])
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), np.float64)
        cur = base.copy() if fixed else cur + delta
        K = np.array([[cur[0], 0, cur[2]], [0, cur[1], cur[3]], [0, 0, 1]], np.float64)
        Trec = rt4(rec["rvec"], rec["tvec"])
        if pose_mode == "current_to_previous":
            W2C = np.eye(4, dtype=np.float64) if order == 0 else np.linalg.inv(Trec) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            W2C = Trec
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")
        C2W = np.linalg.inv(W2C)
        cams[poc] = Camera(poc, K, W2C, C2W, z_sign)
        prev_w2c = W2C
    return header, cams


def depth_scale_real(header: Dict[str, Any]) -> float:
    if "depth_scale_precision" in header:
        p = float(header["depth_scale_precision"])
        if p <= 0:
            raise ValueError("depth_scale_precision must be positive")
        return float(header["depth_scale"]) / p
    return float(header.get("depth_scale_real", header["depth_scale"]))


def scaled_camera(cam: Camera, sx: float, sy: float) -> Camera:
    K = cam.K.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return Camera(cam.poc, K, cam.W2C.copy(), cam.C2W.copy(), cam.z_sign)


def load_mv_csv(path: str, nframes: int, bs: int):
    rows = [[] for _ in range(nframes)]
    refs = [{} for _ in range(nframes)]
    if not path:
        return rows, refs
    req = {"poc", "x", "y", "w", "h", "list", "ref_poc", "mv_x", "mv_y"}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or req - set(rd.fieldnames):
            raise RuntimeError(f"Bad MV CSV header; missing={sorted(req - set(rd.fieldnames or []))}")
        for line_no, r in enumerate(rd, 2):
            try:
                o = MVRecord(int(r["poc"]), int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), str(r["list"]), int(r["ref_poc"]), float(r["mv_x"]), float(r["mv_y"]))
            except Exception as e:
                raise RuntimeError(f"Bad MV CSV row {line_no}: {r}") from e
            if 0 <= o.poc < nframes:
                rows[o.poc].append(o)
                key = (o.x // bs, o.y // bs)
                a = refs[o.poc].setdefault(key, [])
                if o.ref_poc not in a:
                    a.append(o.ref_poc)
    return rows, refs


class GridCache:
    def __init__(self):
        self.cache = {}

    def get(self, w: int, h: int):
        key = (w, h)
        if key not in self.cache:
            xx, yy = np.meshgrid(np.arange(w, dtype=np.float64) - (w - 1) / 2.0,
                                 np.arange(h, dtype=np.float64) - (h - 1) / 2.0)
            self.cache[key] = (xx, yy)
        return self.cache[key]


def fit_plane(z: np.ndarray, cx: float, cy: float, grid: GridCache, min_depth: float, mask=None) -> Optional[Plane]:
    z = np.asarray(z, np.float64)
    h, w = z.shape
    xx, yy = grid.get(w, h)
    valid = np.isfinite(z) & (z > min_depth)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return None
    inv = 1.0 / z[valid]
    if n >= 3:
        A = np.stack([xx[valid], yy[valid], np.ones(n)], axis=1)
        try:
            c, _, rank, _ = np.linalg.lstsq(A, inv, rcond=None)
            if rank >= 3 and np.isfinite(c).all():
                return Plane(float(c[0]), float(c[1]), float(c[2]), cx, cy)
        except np.linalg.LinAlgError:
            pass
    return Plane(0.0, 0.0, float(np.mean(inv)), cx, cy)


def recenter(p: Plane, cx: float, cy: float) -> Plane:
    return Plane(p.a, p.b, p.c + p.a * (cx - p.cx) + p.b * (cy - p.cy), cx, cy)


def average_planes(ps: Sequence[Plane], cx: float, cy: float) -> Optional[Plane]:
    if not ps:
        return None
    q = [recenter(p, cx, cy) for p in ps]
    return Plane(float(np.mean([p.a for p in q])), float(np.mean([p.b for p in q])), float(np.mean([p.c for p in q])), cx, cy)


def render_plane(p: Plane, w: int, h: int, grid: GridCache, min_depth: float, max_depth: float):
    xx, yy = grid.get(w, h)
    inv = p.a * xx + p.b * yy + p.c
    valid = np.isfinite(inv) & (inv > 1.0 / max_depth)
    z = np.zeros((h, w), np.float64)
    z[valid] = 1.0 / inv[valid]
    valid &= (z >= min_depth) & (z <= max_depth)
    z[~valid] = 0.0
    return z, valid


def resize_depth_valid(z: np.ndarray, ow: int, oh: int):
    valid = np.isfinite(z) & (z > 0)
    num = cv2.resize(np.where(valid, z, 0).astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float64)
    den = cv2.resize(valid.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float64)
    out = np.zeros((oh, ow), np.float64)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out, ok


def forward_warp_lowres(zref: np.ndarray, cref: Camera, ccur: Camera, ow: int, oh: int):
    sh, sw = zref.shape
    sx, sy = ow / sw, oh / sh
    z, valid = resize_depth_valid(zref, ow, oh)
    cr = scaled_camera(cref, sx, sy)
    cc = scaled_camera(ccur, sx, sy)
    x, y = np.meshgrid(np.arange(ow, dtype=np.float64), np.arange(oh, dtype=np.float64))
    rays = np.stack([(x - cr.K[0, 2]) / cr.K[0, 0], (y - cr.K[1, 2]) / cr.K[1, 1], np.full_like(x, cr.z_sign)], axis=-1)
    Xr = rays * z[..., None]
    M = cc.W2C @ cr.C2W
    Xc = Xr @ M[:3, :3].T + M[:3, 3]
    d = cc.z_sign * Xc[..., 2]
    front = valid & np.isfinite(d) & (d > 1e-10)
    safe = np.where(front, d, 1.0)
    u = cc.K[0, 0] * Xc[..., 0] / safe + cc.K[0, 2]
    v = cc.K[1, 1] * Xc[..., 1] / safe + cc.K[1, 2]
    good = front & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= ow - 1) & (v >= 0) & (v <= oh - 1)
    zbuf = np.full(ow * oh, np.inf, np.float64)
    if np.any(good):
        uu, vv, dd = u[good], v[good], d[good]
        x0, y0 = np.floor(uu).astype(np.int64), np.floor(vv).astype(np.int64)
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < ow) & (yi >= 0) & (yi < oh)
                if np.any(ok):
                    np.minimum.at(zbuf, yi[ok] * ow + xi[ok], dd[ok])
    out = zbuf.reshape(oh, ow)
    mask = np.isfinite(out)
    out[~mask] = 0.0
    return out, mask


def extract_fw_block(z: np.ndarray, m: np.ndarray, bx: int, by: int, bw: int, bh: int, fw: int, fh: int):
    h, w = z.shape
    x0 = max(0, min(int(math.floor(bx * w / fw)), w - 1))
    y0 = max(0, min(int(math.floor(by * h / fh)), h - 1))
    x1 = max(x0 + 1, min(int(math.ceil((bx + bw) * w / fw)), w))
    y1 = max(y0 + 1, min(int(math.ceil((by + bh) * h / fh)), h))
    zs, ms = z[y0:y1, x0:x1], m[y0:y1, x0:x1].astype(np.float32)
    num = cv2.resize((zs * ms).astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    den = cv2.resize(ms, (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    out = np.zeros((bh, bw), np.float64)
    ok = den > 0.25
    out[ok] = num[ok] / np.maximum(den[ok], 1e-8)
    return out, ok


def combine_fw(blocks):
    if not blocks:
        return None
    total = np.zeros_like(blocks[0][0])
    count = np.zeros_like(total)
    for z, m in blocks:
        total[m] += z[m]
        count[m] += 1
    valid = count > 0
    out = np.zeros_like(total)
    out[valid] = total[valid] / count[valid]
    return out, valid



@dataclass
class RDResult:
    mode: str
    name: str
    pred: Plane
    recon: Plane
    q: Tuple[int, int, int]
    candidate_idx: int
    mode_bits: float
    candidate_bits: float
    residual_bits: float
    bits: float
    sse: float
    cost: float
    recon_y: np.ndarray
    residual_present: bool
    buffer_no_signal: bool


def truncated_unary_bits(idx: int, n: int) -> int:
    if n <= 1:
        return 0
    return idx if idx == n - 1 else idx + 1


def candidate_idx_bits(idx: int, n: int, coding: str) -> int:
    if n <= 1:
        return 0
    if coding == "fixed":
        return ceil_log2(n)
    if coding == "truncated_unary":
        return truncated_unary_bits(idx, n)
    raise ValueError(coding)


def syntax_mode_bits(mode: str) -> int:
    if mode == "direct":
        return 1
    if mode in ("predictor_only", "predictor_residual"):
        return 2
    if mode == "buffer_reuse":
        return 0
    raise ValueError(mode)


class DepthReuseBuffer:
    def __init__(self, fw: int, fh: int, scale: int, min_depth: float):
        self.fw = fw
        self.fh = fh
        self.scale = scale
        self.min_depth = min_depth
        self.w = int(math.ceil(fw / scale))
        self.h = int(math.ceil(fh / scale))
        self.depth = np.zeros((self.h, self.w), np.float64)
        self.valid = np.zeros((self.h, self.w), bool)

    def rect(self, x: int, y: int, w: int, h: int):
        x0 = max(0, x // self.scale)
        y0 = max(0, y // self.scale)
        x1 = min(self.w, int(math.ceil((x + w) / self.scale)))
        y1 = min(self.h, int(math.ceil((y + h) / self.scale)))
        return x0, y0, x1, y1

    def can_reuse(self, x: int, y: int, w: int, h: int) -> bool:
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        return x1 > x0 and y1 > y0 and bool(np.all(self.valid[y0:y1, x0:x1]))

    def reconstruct(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        if not np.all(self.valid[y0:y1, x0:x1]):
            raise RuntimeError("invalid depth-buffer reuse")
        return cv2.resize(self.depth[y0:y1, x0:x1].astype(np.float32),
                          (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    def commit(self, x: int, y: int, z: np.ndarray):
        h, w = z.shape
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        tw, th = x1 - x0, y1 - y0
        if tw <= 0 or th <= 0:
            return
        valid = np.isfinite(z) & (z > self.min_depth)
        num = cv2.resize(np.where(valid, z, 0).astype(np.float32),
                         (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
        den = cv2.resize(valid.astype(np.float32),
                         (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
        ok = den > 1e-6
        val = np.zeros((th, tw), np.float64)
        val[ok] = num[ok] / den[ok]
        self.depth[y0:y1, x0:x1][ok] = val[ok]
        self.valid[y0:y1, x0:x1][ok] = True


def plane_to_recon_y(p: Plane, gt_y: np.ndarray, scale: float, grid: GridCache,
                     min_depth: float, max_depth: float):
    z, valid = render_plane(p, gt_y.shape[1], gt_y.shape[0], grid, min_depth, max_depth)
    ry = np.zeros_like(z)
    ry[valid] = z[valid] / scale
    return np.clip(np.rint(ry), 0, 1023), z


def make_direct(actual: Plane, gt_y: np.ndarray, scale: float, qs, lam: float,
                grid: GridCache, min_depth: float, max_depth: float) -> RDResult:
    qs = np.asarray(qs, np.float64)
    q = np.rint(np.array([actual.a, actual.b, actual.c]) / qs).astype(np.int64)
    d = q * qs
    rec = Plane(float(d[0]), float(d[1]), float(d[2]), actual.cx, actual.cy)
    ry, _ = plane_to_recon_y(rec, gt_y, scale, grid, min_depth, max_depth)
    sse = float(np.sum((gt_y - ry) ** 2))
    mb = float(syntax_mode_bits("direct"))
    rb = float(sum(se_bits(int(v)) for v in q))
    bits = mb + rb
    return RDResult("direct", "direct", Plane(0,0,0,actual.cx,actual.cy), rec,
                    tuple(map(int,q)), -1, mb, 0.0, rb, bits, sse,
                    sse + lam * bits, ry, True, False)


def make_predictor_only(name: str, pred: Plane, idx: int, n: int, gt_y: np.ndarray,
                        scale: float, coding: str, lam: float, grid: GridCache,
                        min_depth: float, max_depth: float) -> RDResult:
    ry, _ = plane_to_recon_y(pred, gt_y, scale, grid, min_depth, max_depth)
    sse = float(np.sum((gt_y - ry) ** 2))
    mb = float(syntax_mode_bits("predictor_only"))
    cb = float(candidate_idx_bits(idx, n, coding))
    bits = mb + cb
    return RDResult("predictor_only", name, pred, pred, (0,0,0), idx,
                    mb, cb, 0.0, bits, sse, sse + lam * bits, ry, False, False)


def make_predictor_residual(name: str, pred: Plane, actual: Plane, idx: int, n: int,
                            gt_y: np.ndarray, scale: float, coding: str, qs, lam: float,
                            grid: GridCache, min_depth: float, max_depth: float) -> RDResult:
    qs = np.asarray(qs, np.float64)
    q = np.rint(np.array([actual.a-pred.a, actual.b-pred.b, actual.c-pred.c]) / qs).astype(np.int64)
    d = q * qs
    rec = Plane(pred.a+float(d[0]), pred.b+float(d[1]), pred.c+float(d[2]), actual.cx, actual.cy)
    ry, _ = plane_to_recon_y(rec, gt_y, scale, grid, min_depth, max_depth)
    sse = float(np.sum((gt_y - ry) ** 2))
    mb = float(syntax_mode_bits("predictor_residual"))
    cb = float(candidate_idx_bits(idx, n, coding))
    rb = float(sum(se_bits(int(v)) for v in q))
    bits = mb + cb + rb
    return RDResult("predictor_residual", name, pred, rec, tuple(map(int,q)), idx,
                    mb, cb, rb, bits, sse, sse + lam * bits, ry, True, False)


def make_buffer_reuse(gt_y: np.ndarray, z: np.ndarray, scale: float, cx: float, cy: float,
                      grid: GridCache, min_depth: float) -> RDResult:
    valid = np.isfinite(z) & (z > min_depth)
    ry = np.zeros_like(z)
    ry[valid] = z[valid] / scale
    ry = np.clip(np.rint(ry), 0, 1023)
    sse = float(np.sum((gt_y - ry) ** 2))
    p = fit_plane(z, cx, cy, grid, min_depth, valid) or Plane(0,0,0,cx,cy)
    return RDResult("buffer_reuse", "depth_buffer", p, p, (0,0,0), -1,
                    0.0, 0.0, 0.0, 0.0, sse, sse, ry, False, True)


def qp_steps(args):
    s = 2.0 ** ((args.qp - args.qp_ref) / 6.0)
    return args.qa_base * s, args.qb_base * s, args.qc_base * s


def derive_lambda(args):
    return args.lambda_rd if args.lambda_rd is not None else args.lambda_scale * 2.0 ** ((args.qp - 12.0) / 3.0)


def parse_args():
    p = argparse.ArgumentParser(description="GT depth fixed-block candidate RDO simulator with predictor-only and implicit depth-buffer reuse")
    p.add_argument("--gt-depth-yuv", required=True)
    p.add_argument("--camera-param", required=True)
    p.add_argument("--mv-csv", default="")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--forward-downsample", type=int, default=4)
    p.add_argument("--depth-buffer-downsample", type=int, default=4)
    p.add_argument("--max-forward-refs", type=int, default=2)
    p.add_argument("--default-ref-offset", type=int, default=1)
    p.add_argument("--min-forward-valid-ratio", type=float, default=0.25)
    p.add_argument("--candidate-idx-coding", choices=["fixed","truncated_unary"], default="truncated_unary")
    p.add_argument("--disable-buffer-reuse", action="store_true")
    p.add_argument("--disable-predictor-only", action="store_true")
    p.add_argument("--disable-predictor-residual", action="store_true")
    p.add_argument("--disable-direct", action="store_true")
    p.add_argument("--qp", type=int, default=37)
    p.add_argument("--qp-ref", type=int, default=37)
    p.add_argument("--qa-base", type=float, default=1e-6)
    p.add_argument("--qb-base", type=float, default=1e-6)
    p.add_argument("--qc-base", type=float, default=1e-4)
    p.add_argument("--lambda-rd", type=float, default=None)
    p.add_argument("--lambda-scale", type=float, default=0.57)
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--max-depth", type=float, default=1e9)
    p.add_argument("--out-recon-yuv", required=True)
    p.add_argument("--out-block-csv", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError("Invalid YUV420 resolution")
    if min(a.block_size, a.forward_downsample, a.depth_buffer_downsample) <= 0:
        raise ValueError("Invalid block/downsample size")
    if min(a.qa_base, a.qb_base, a.qc_base) <= 0:
        raise ValueError("qsteps must be positive")
    if a.disable_direct and a.disable_predictor_only and a.disable_predictor_residual:
        raise ValueError("All explicit coding modes are disabled")

    outs = [Path(a.out_recon_yuv), Path(a.out_block_csv), Path(a.out_summary_json)]
    for p in outs:
        if p.exists():
            if a.overwrite: p.unlink()
            else: raise RuntimeError(f"Output exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)

    header, cams = load_cameras(a.camera_param)
    scale = depth_scale_real(header)
    total = count_frames(a.gt_depth_yuv, a.width, a.height)
    start = a.start_frame
    end = total if a.num_frames == 0 else min(total, start + a.num_frames)
    if start < 0 or start >= end: raise ValueError("Invalid frame range")

    rows, block_refs = load_mv_csv(a.mv_csv, total, a.block_size)
    qa, qb, qc = qp_steps(a); lam = derive_lambda(a)
    lw = int(math.ceil(a.width / a.forward_downsample)); lh = int(math.ceil(a.height / a.forward_downsample))
    dbw = int(math.ceil(a.width / a.depth_buffer_downsample)); dbh = int(math.ceil(a.height / a.depth_buffer_downsample))
    grid = GridCache()

    print(f"block={a.block_size}, no split, forward={lw}x{lh}, reuse={dbw}x{dbh}, QP={a.qp}")
    print(f"qsteps={qa:.6g},{qb:.6g},{qc:.6g}, lambda={lam:.6g}, depth_scale={scale:.12g}")

    fields = ["poc","x","y","w","h","mode","candidate_count","candidate_idx","candidate","mode_bits","candidate_bits","residual_bits","total_bits","residual_present","buffer_no_signal","q_a","q_b","q_c","pred_a","pred_b","pred_c","gt_a","gt_b","gt_c","recon_a","recon_b","recon_c","sse","cost","forward_refs"]
    frames=[]; global_modes={}; global_cands={}; total_bits=0.0; total_sse=0.0

    with open(a.gt_depth_yuv,"rb") as fp, open(a.out_recon_yuv,"wb") as outfp, open(a.out_block_csv,"w",newline="",encoding="utf-8") as cfp:
        wr=csv.DictWriter(cfp,fieldnames=fields); wr.writeheader(); gt_cache={}
        def gt(poc):
            if poc not in gt_cache: gt_cache[poc]=read_y(fp,poc,a.width,a.height)
            return gt_cache[poc]

        for ord_idx,poc in enumerate(range(start,end)):
            if poc not in cams: raise KeyError(f"No camera for POC {poc}")
            gy=gt(poc); gz=gy*scale; recon=np.zeros_like(gy)
            frame_refs=[]
            for r in rows[poc]:
                if r.ref_poc!=poc and r.ref_poc in cams and r.ref_poc not in frame_refs: frame_refs.append(r.ref_poc)
            if not frame_refs and poc-a.default_ref_offset in cams: frame_refs=[poc-a.default_ref_offset]
            frame_refs=frame_refs[:a.max_forward_refs]
            fw_cache={ref:forward_warp_lowres(gt(ref)*scale,cams[ref],cams[poc],lw,lh) for ref in frame_refs if 0<=ref<total}
            reuse=DepthReuseBuffer(a.width,a.height,a.depth_buffer_downsample,a.min_depth)
            rec_planes={}; fbits=0.0; fsse=0.0; fmodes={}; fcands={}; nblocks=0
            gw=int(math.ceil(a.width/a.block_size)); gh=int(math.ceil(a.height/a.block_size))

            for byi in range(gh):
                by=byi*a.block_size; bh=min(a.block_size,a.height-by)
                for bxi in range(gw):
                    bx=bxi*a.block_size; bw=min(a.block_size,a.width-bx); nblocks+=1
                    cx=bx+(bw-1)/2; cy=by+(bh-1)/2
                    gyb=gy[by:by+bh,bx:bx+bw]; gzb=gz[by:by+bh,bx:bx+bw]
                    actual=fit_plane(gzb,cx,cy,grid,a.min_depth) or Plane(0,0,1/a.max_depth,cx,cy)
                    cands=[]; refs=[]

                    if not a.disable_buffer_reuse and reuse.can_reuse(bx,by,bw,bh):
                        best=make_buffer_reuse(gyb,reuse.reconstruct(bx,by,bw,bh),scale,cx,cy,grid,a.min_depth)
                    else:
                        spatial=[]
                        for name,key in (("left",(bxi-1,byi)),("top",(bxi,byi-1)),("top_left",(bxi-1,byi-1))):
                            if key in rec_planes:
                                p=recenter(rec_planes[key],cx,cy); cands.append((name,p)); spatial.append(p)
                        if len(spatial)>=2:
                            p=average_planes(spatial,cx,cy)
                            if p is not None: cands.append(("spatial_all",p))

                        for ref in block_refs[poc].get((bxi,byi),[]):
                            if ref!=poc and ref in cams and ref not in refs: refs.append(ref)
                        for ref in frame_refs:
                            if ref not in refs: refs.append(ref)
                        refs=refs[:a.max_forward_refs]; fw_blocks=[]
                        for ref in refs:
                            if ref not in fw_cache and 0<=ref<total and ref in cams:
                                fw_cache[ref]=forward_warp_lowres(gt(ref)*scale,cams[ref],cams[poc],lw,lh)
                            if ref not in fw_cache: continue
                            zb,mb=extract_fw_block(*fw_cache[ref],bx,by,bw,bh,a.width,a.height)
                            if np.mean(mb)<a.min_forward_valid_ratio: continue
                            p=fit_plane(zb,cx,cy,grid,a.min_depth,mb)
                            if p is not None: cands.append((f"fw_ref_{ref}",p)); fw_blocks.append((zb,mb))
                        if len(fw_blocks)>=2:
                            comb=combine_fw(fw_blocks)
                            if comb is not None and np.mean(comb[1])>=a.min_forward_valid_ratio:
                                p=fit_plane(comb[0],cx,cy,grid,a.min_depth,comb[1])
                                if p is not None: cands.append(("fw_average",p))

                        results=[]
                        if not a.disable_direct: results.append(make_direct(actual,gyb,scale,(qa,qb,qc),lam,grid,a.min_depth,a.max_depth))
                        for idx,(name,pred) in enumerate(cands):
                            if not a.disable_predictor_only:
                                results.append(make_predictor_only(name,pred,idx,len(cands),gyb,scale,a.candidate_idx_coding,lam,grid,a.min_depth,a.max_depth))
                            if not a.disable_predictor_residual:
                                results.append(make_predictor_residual(name,pred,actual,idx,len(cands),gyb,scale,a.candidate_idx_coding,(qa,qb,qc),lam,grid,a.min_depth,a.max_depth))
                        if not results: raise RuntimeError(f"No RDO mode at POC={poc}, block=({bx},{by})")
                        best=min(results,key=lambda r:r.cost)
                        reuse.commit(bx,by,best.recon_y*scale)

                    recon[by:by+bh,bx:bx+bw]=best.recon_y; rec_planes[(bxi,byi)]=best.recon
                    fbits+=best.bits; fsse+=best.sse
                    fmodes[best.mode]=fmodes.get(best.mode,0)+1; fcands[best.name]=fcands.get(best.name,0)+1
                    global_modes[best.mode]=global_modes.get(best.mode,0)+1; global_cands[best.name]=global_cands.get(best.name,0)+1
                    wr.writerow({"poc":poc,"x":bx,"y":by,"w":bw,"h":bh,"mode":best.mode,"candidate_count":len(cands),"candidate_idx":best.candidate_idx,"candidate":best.name,"mode_bits":best.mode_bits,"candidate_bits":best.candidate_bits,"residual_bits":best.residual_bits,"total_bits":best.bits,"residual_present":int(best.residual_present),"buffer_no_signal":int(best.buffer_no_signal),"q_a":best.q[0],"q_b":best.q[1],"q_c":best.q[2],"pred_a":best.pred.a,"pred_b":best.pred.b,"pred_c":best.pred.c,"gt_a":actual.a,"gt_b":actual.b,"gt_c":actual.c,"recon_a":best.recon.a,"recon_b":best.recon.b,"recon_c":best.recon.c,"sse":best.sse,"cost":best.cost,"forward_refs":"|".join(map(str,refs))})

            write_depth_frame(outfp,recon,a.width,a.height)
            mse=fsse/(a.width*a.height); psnr=float("inf") if mse==0 else 10*math.log10(1023**2/mse)
            frames.append({"poc":poc,"num_blocks":nblocks,"frame_bits":fbits,"frame_bpp":fbits/(a.width*a.height),"frame_sse":fsse,"frame_mse":mse,"frame_psnr":psnr,"frame_reference_pocs":frame_refs,"forward_buffers_built":sorted(fw_cache.keys()),"mode_selection_counts":fmodes,"candidate_selection_counts":fcands,"buffer_reuse_ratio":fmodes.get("buffer_reuse",0)/nblocks,"predictor_only_ratio":fmodes.get("predictor_only",0)/nblocks,"predictor_residual_ratio":fmodes.get("predictor_residual",0)/nblocks,"direct_ratio":fmodes.get("direct",0)/nblocks})
            total_bits+=fbits; total_sse+=fsse
            ratio=(ord_idx+1)/(end-start); width=30; fill=int(round(width*ratio))
            print(f"\r[{'#'*fill}{'-'*(width-fill)}] {ord_idx+1}/{end-start} POC={poc} bits={fbits:.1f} reuse={fmodes.get('buffer_reuse',0)/nblocks:.3f} predOnly={fmodes.get('predictor_only',0)/nblocks:.3f} PSNR={psnr:.3f}",end="",flush=True)
    print()

    pixels=a.width*a.height*(end-start); mse=total_sse/pixels; psnr=float("inf") if mse==0 else 10*math.log10(1023**2/mse)
    summary={"gt_depth_yuv":a.gt_depth_yuv,"camera_param":a.camera_param,"mv_csv":a.mv_csv,"width":a.width,"height":a.height,"start_frame":start,"num_frames":end-start,"block_size":a.block_size,"block_split":False,"forward_downsample":a.forward_downsample,"forward_buffer_size":[lw,lh],"depth_buffer_downsample":a.depth_buffer_downsample,"depth_buffer_size":[dbw,dbh],"depth_buffer_rule":"all covered cells valid -> implicit zero-bit reuse; otherwise run explicit RDO and commit winner","reference_depth_source":"GT depth frame","depth_scale_real":scale,"plane_equation":"1/z = a*(x-cx)+b*(y-cy)+c","qp":a.qp,"qp_ref":a.qp_ref,"quantization":{"qa":qa,"qb":qb,"qc":qc,"formula":"base*2^((QP-QP_ref)/6)"},"lambda_rd":lam,"rate_model":{"buffer_reuse":"0 bits","direct":"1 mode bit + se(q_a,q_b,q_c)","predictor_only":"2 mode bits + candidate_idx","predictor_residual":"2 mode bits + candidate_idx + se(q_a,q_b,q_c)","candidate_idx_coding":a.candidate_idx_coding},"candidate_types":["left","top","top_left","spatial_all","fw_ref_<POC>","fw_average"],"mode_types":["buffer_reuse","predictor_only","predictor_residual","direct"],"total_bits":total_bits,"overall_bpp":total_bits/pixels,"overall_sse":total_sse,"overall_mse":mse,"overall_psnr":psnr,"mode_selection_counts":global_modes,"candidate_selection_counts":global_cands,"frames":frames,"out_recon_yuv":a.out_recon_yuv,"out_block_csv":a.out_block_csv}
    with open(a.out_summary_json,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f"Recon YUV : {a.out_recon_yuv}\nBlock CSV : {a.out_block_csv}\nSummary   : {a.out_summary_json}\nTotal bits: {total_bits:.3f}\nBPP       : {summary['overall_bpp']:.9f}\nPSNR      : {psnr:.6f} dB")


if __name__ == "__main__":
    main()
