#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-resolution inverse-depth plane coding simulator with:

  * selectable depth downsampling at 1 / --depth-scale in each dimension
  * optional zero-bit/zero-depth I pictures selected by POC
  * hierarchical RA or sequential coding order
  * five-point temporal plane resampling for L0/L1
  * projection-domain luma SATD RDO
  * cached separable Hadamard transforms by evaluated block size
  * depth-context adaptive leaf/BH/BV/QT split-type signaling
  * display-order full-resolution reconstructed-depth YUV output
  * L0/L1 camera-depth video prediction output

Depth processing
----------------
If --depth-scale 4 is used, the input depth is area-downsampled to W/4 x H/4.
All depth-plane fitting, partitioning, candidate construction, and stored plane
state operate at that reduced resolution. For output and projection-domain RDO,
the reconstructed depth is enlarged to the original resolution using
--depth-upsample (nearest or bilinear).

Temporal camera-plane candidate
-------------------------------
For each current low-resolution block, five target samples are used:
four pixel-cell corners and the block center. At each sample position, every
projected reference leaf covering that point is tested, and the nearest visible
transformed 3D plane supplies the sample depth. The valid samples are then
refitted into one current-view inverse-depth plane. This is done independently
for L0 and L1. A bidirectional average candidate is also fitted from the valid
per-point L0/L1 depths.

RDO distortion
--------------
For an inter picture, each reconstructed candidate depth block is enlarged to
the original video resolution. Current pixels are backward-projected into the
actual L0/L1 reference pictures and sampled bilinearly. The candidate
distortion is the cached-Hadamard SATD between this predictor and the original
target Y block. Out-of-picture samples use reflected coordinates. If every projection in a
block is geometrically invalid, RDO falls back to the least-bit candidate.

For a non-zero-coded anchor picture with no temporal reference, depth-domain
SSE is used because projection-domain prediction is undefined. POCs selected
by --zero-depth-frames bypass RDO entirely, reconstruct all-zero depth, and use
zero depth bits.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ============================================================
# Data classes
# ============================================================

@dataclass
class Plane:
    # invY(x,y) = a * (x-cx) + b * (y-cy) + c
    a: float
    b: float
    c: float
    cx: float
    cy: float


@dataclass
class ModeResult:
    mode: str
    candidate_name: str
    plane: Plane
    recon_block: np.ndarray
    bits: float
    distortion: float
    cost: float
    q_values: Tuple[int, ...]


@dataclass
class LeafRecord:
    x: int
    y: int
    w: int
    h: int
    plane: Plane


@dataclass
class CSNode:
    x: int
    y: int
    w: int
    h: int
    depth: int
    parent: Optional["CSNode"] = None
    split: str = "leaf"
    children: List["CSNode"] = field(default_factory=list)
    best: Optional[ModeResult] = None
    actual: Optional[Plane] = None
    avail_modes: List[str] = field(default_factory=list)
    avail_cands: List[str] = field(default_factory=list)
    bits: float = 0.0
    distortion: float = 0.0
    cost: float = 0.0
    split_bits: float = 0.0
    avail_splits: List[str] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.split == "leaf"


@dataclass
class ProjectionRDOContext:
    prediction_type: str
    cur_y: np.ndarray
    cam_cur_full: Dict[str, Any]
    depth_scale: int
    max_value: int
    invalid_fill: str
    upsample_mode: str
    l0_y: Optional[np.ndarray] = None
    cam_l0_full: Optional[Dict[str, Any]] = None
    l1_y: Optional[np.ndarray] = None
    cam_l1_full: Optional[Dict[str, Any]] = None


@dataclass
class PlaneWarpContext:
    l0_store: List[LeafRecord]
    cam_l0_low: Dict[str, Any]
    cam_cur_low: Dict[str, Any]
    frame_w_low: int
    frame_h_low: int
    l1_store: Optional[List[LeafRecord]] = None
    cam_l1_low: Optional[Dict[str, Any]] = None
    projected_leaf_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# ============================================================
# Probability models / bit estimates
# ============================================================

class AdaptiveProbTable:
    def __init__(
        self,
        symbols: Sequence[Any],
        update_rate: float = 0.05,
        p_min: float = 0.02,
        p_max: float = 0.95,
        name: str = "",
    ):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.name = name
        if self.n <= 0:
            raise ValueError("AdaptiveProbTable requires symbols")
        if self.p_min * self.n > 1.0:
            raise ValueError(f"{name}: p_min is too large")
        if self.p_max * self.n < 1.0:
            raise ValueError(f"{name}: p_max is too small")
        self.probs = {s: 1.0 / self.n for s in self.symbols}
        self._project()

    def bits(self, symbol: Any, available_symbols: Optional[Sequence[Any]] = None) -> float:
        if symbol not in self.probs:
            raise KeyError(f"unknown symbol {symbol!r} in {self.name}")
        if available_symbols is None:
            return -math.log2(max(self.probs[symbol], 1e-12))
        av = [s for s in available_symbols if s in self.probs]
        if symbol not in av:
            raise KeyError(f"{symbol!r} is not available in {self.name}")
        if len(av) <= 1:
            return 0.0
        norm = sum(self.probs[s] for s in av)
        p = self.probs[symbol] / norm if norm > 0.0 else 1.0 / len(av)
        return -math.log2(max(p, 1e-12))

    def update(self, selected: Any) -> None:
        if selected not in self.probs:
            raise KeyError(f"unknown selected symbol {selected!r}")
        psel = min(self.p_max, 1.0 - (self.n - 1) * self.p_min)
        others = [s for s in self.symbols if s != selected]
        target = {selected: psel}
        for s in others:
            target[s] = (1.0 - psel) / len(others)
        lr = self.update_rate
        for s in self.symbols:
            self.probs[s] = (1.0 - lr) * self.probs[s] + lr * target[s]
        self._project()

    def _project(self) -> None:
        for s in self.symbols:
            self.probs[s] = min(max(self.probs[s], self.p_min), self.p_max)
        for _ in range(64):
            diff = 1.0 - sum(self.probs.values())
            if abs(diff) < 1e-12:
                break
            if diff > 0.0:
                adjustable = [s for s in self.symbols if self.probs[s] < self.p_max - 1e-12]
            else:
                adjustable = [s for s in self.symbols if self.probs[s] > self.p_min + 1e-12]
            if not adjustable:
                break
            add = diff / len(adjustable)
            for s in adjustable:
                self.probs[s] = min(max(self.probs[s] + add, self.p_min), self.p_max)

    def snapshot(self, prefix: str) -> Dict[str, float]:
        return {f"{prefix}_{s}_prob": self.probs[s] for s in self.symbols}


class BinaryAdaptiveProb:
    def __init__(
        self,
        init_p1: float = 0.5,
        update_rate: float = 0.05,
        p_min: float = 0.02,
        p_max: float = 0.98,
    ):
        self.p1 = float(init_p1)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self._clip()

    def _clip(self) -> None:
        self.p1 = min(max(self.p1, self.p_min), self.p_max)

    def bits(self, b: int) -> float:
        if b not in (0, 1):
            raise ValueError("binary symbol must be 0 or 1")
        p = self.p1 if b else 1.0 - self.p1
        return -math.log2(max(p, 1e-12))

    def update(self, b: int) -> None:
        target = self.p_max if b else self.p_min
        self.p1 = (1.0 - self.update_rate) * self.p1 + self.update_rate * target
        self._clip()


def unary_candidate_bits(idx: int, n: int, ctx: Sequence[BinaryAdaptiveProb]) -> float:
    if n <= 1:
        return 0.0
    bits = sum(ctx[i].bits(0) for i in range(idx))
    if idx != n - 1:
        bits += ctx[idx].bits(1)
    return bits


def unary_candidate_update(idx: int, n: int, ctx: Sequence[BinaryAdaptiveProb]) -> None:
    if n <= 1:
        return
    for i in range(idx):
        ctx[i].update(0)
    if idx != n - 1:
        ctx[idx].update(1)


def split_type_bits(
    adaptive: Optional[Dict[str, Any]],
    depth: int,
    split: str,
    available_splits: Sequence[str],
) -> float:
    """Estimate the syntax cost of one partition decision.

    A separate probability table is maintained for each partition depth.
    Probabilities are renormalized over only the split types available for the
    current node, so impossible QT/BH/BV symbols consume no probability mass.
    """
    if split not in available_splits:
        raise ValueError(f"split {split!r} is not available: {available_splits}")
    if len(available_splits) <= 1:
        return 0.0
    if adaptive is not None and "split_type" in adaptive:
        contexts = adaptive["split_type"]
        ctx = contexts[min(depth, len(contexts) - 1)]
        return ctx.bits(split, available_splits)
    return float(ceil_log2(len(available_splits)))


def split_type_update(
    adaptive: Optional[Dict[str, Any]],
    node: CSNode,
) -> None:
    if (
        adaptive is None
        or "split_type" not in adaptive
        or len(node.avail_splits) <= 1
    ):
        return
    contexts = adaptive["split_type"]
    contexts[min(node.depth, len(contexts) - 1)].update(node.split)


def available_split_types(
    depth: int,
    w: int,
    h: int,
    max_qt_depth: int,
) -> List[str]:
    out = ["leaf"]
    if h >= 2 and h % 2 == 0:
        out.append("bh")
    if w >= 2 and w % 2 == 0:
        out.append("bv")
    if (
        depth < max_qt_depth
        and w >= 2
        and h >= 2
        and w % 2 == 0
        and h % 2 == 0
    ):
        out.append("qt")
    return out

def ceil_log2(x: int) -> int:
    return 0 if x <= 1 else int(math.ceil(math.log2(x)))


def exp_golomb_len_unsigned(u: int) -> int:
    if u < 0:
        raise ValueError("unsigned Exp-Golomb input is negative")
    return 2 * int(math.floor(math.log2(u + 1))) + 1


def signed_to_code_num(v: int) -> int:
    if v == 0:
        return 0
    return 2 * v - 1 if v > 0 else -2 * v


def exp_golomb_len_signed(v: int) -> int:
    return exp_golomb_len_unsigned(signed_to_code_num(v))


def quantize(x: float, q: float) -> int:
    return int(np.rint(x / q))


def dequantize(v: int, q: float) -> float:
    return float(v) * q


def adaptive_signed_residual_bits(q: int, model: AdaptiveProbTable, abs_max: int) -> float:
    a = abs(q)
    if a <= abs_max:
        bits = model.bits(a)
    else:
        bits = model.bits("esc") + exp_golomb_len_unsigned(a - abs_max - 1)
    if a > 0:
        bits += 1.0
    return bits


def adaptive_signed_residual_update(q: int, model: AdaptiveProbTable, abs_max: int) -> None:
    model.update(abs(q) if abs(q) <= abs_max else "esc")


def create_adaptive_models(args: argparse.Namespace) -> Dict[str, Any]:
    candidate_symbols = [
        "plane_warp_avg",
        "plane_warp_l0",
        "plane_warp_l1",
        "left",
        "top",
        "top_left",
        "top_right",
        "avg_left_top",
    ]
    models: Dict[str, Any] = {
        "mode": AdaptiveProbTable(
            ["direct", "copy", "delta"],
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="mode",
        ),
        "candidate": AdaptiveProbTable(
            candidate_symbols,
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="candidate",
        ),
        "delta_abs_max": args.delta_abs_max,
        "split_type": [
            AdaptiveProbTable(
                ["leaf", "bh", "bv", "qt"],
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"split_type_depth{depth}",
            )
            for depth in range(args.max_qt_depth + 1)
        ],
    }
    if args.copy_candidate_unary:
        models["copy_candidate_unary"] = [
            BinaryAdaptiveProb(
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
            )
            for _ in range(args.max_candidates)
        ]
    if args.delta_residual_adaptive:
        symbols = list(range(args.delta_abs_max + 1)) + ["esc"]
        for k in "abc":
            models[f"delta_res_abs_{k}"] = AdaptiveProbTable(
                symbols,
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"delta_res_abs_{k}",
            )
    return models


# ============================================================
# Grid, depth scaling, plane fitting
# ============================================================

class GridCache:
    def __init__(self):
        self.cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (w, h)
        if key not in self.cache:
            xs = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
            ys = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
            xx, yy = np.meshgrid(xs, ys)
            a = np.stack(
                [xx.reshape(-1), yy.reshape(-1), np.ones(w * h, dtype=np.float64)],
                axis=1,
            )
            self.cache[key] = (xx, yy, np.linalg.pinv(a))
        return self.cache[key]


def downsample_depth_integer(img: np.ndarray, scale: int, mode: str) -> np.ndarray:
    if scale == 1:
        return np.asarray(img, dtype=np.float64).copy()
    h, w = img.shape
    if h % scale or w % scale:
        raise ValueError(
            f"input resolution {w}x{h} must be divisible by depth scale {scale}"
        )

    src = np.asarray(img, dtype=np.float64)
    blocks = src.reshape(h // scale, scale, w // scale, scale)

    if mode == "max":
        return blocks.max(axis=(1, 3))
    if mode == "min":
        return blocks.min(axis=(1, 3))
    if mode == "mean":
        return blocks.mean(axis=(1, 3))
    if mode == "nearest":
        # Sample the pixel nearest to the center of each scale x scale cell.
        offset = (scale - 1) // 2
        return src[offset::scale, offset::scale].copy()

    raise ValueError(f"unsupported downsample mode: {mode}")


def upsample_depth_integer(img: np.ndarray, scale: int, mode: str) -> np.ndarray:
    src = np.asarray(img, dtype=np.float64)
    if scale == 1:
        return src.copy()

    if mode == "nearest":
        return np.repeat(np.repeat(src, scale, axis=0), scale, axis=1)

    if mode == "bilinear":
        in_h, in_w = src.shape
        out_h, out_w = in_h * scale, in_w * scale

        # Pixel-center aligned bilinear resize. Border samples are replicated.
        x = (np.arange(out_w, dtype=np.float64) + 0.5) / scale - 0.5
        y = (np.arange(out_h, dtype=np.float64) + 0.5) / scale - 0.5
        x = np.clip(x, 0.0, max(in_w - 1, 0))
        y = np.clip(y, 0.0, max(in_h - 1, 0))

        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, in_w - 1)
        y1 = np.minimum(y0 + 1, in_h - 1)
        dx = x - x0
        dy = y - y0

        row0 = (1.0 - dx)[None, :] * src[:, x0] + dx[None, :] * src[:, x1]
        return (1.0 - dy)[:, None] * row0[y0, :] + dy[:, None] * row0[y1, :]

    raise ValueError(f"unsupported upsample mode: {mode}")


def fit_inv_depth_plane_from_depth_block(
    block_y: np.ndarray,
    pinv: np.ndarray,
    cx: float,
    cy: float,
    args: argparse.Namespace,
) -> Plane:
    y = np.clip(np.asarray(block_y, dtype=np.float64), args.depth_eps, args.max_value)
    inv = 1.0 / y
    if args.c_only_plane:
        return Plane(0.0, 0.0, float(np.mean(inv)), cx, cy)
    a, b, c = (pinv @ inv.reshape(-1)).tolist()
    return Plane(float(a), float(b), float(c), cx, cy)


def fit_inv_plane_from_samples(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    depth_y: np.ndarray,
    valid: np.ndarray,
    cx: float,
    cy: float,
    args: argparse.Namespace,
) -> Optional[Plane]:
    valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(sample_x)
        & np.isfinite(sample_y)
        & np.isfinite(depth_y)
        & (depth_y >= args.depth_eps)
        & (depth_y <= args.max_value)
    )
    count = int(np.count_nonzero(valid))
    if args.c_only_plane:
        if count < 1:
            return None
        inv = 1.0 / np.clip(depth_y[valid], args.depth_eps, args.max_value)
        return Plane(0.0, 0.0, float(np.median(inv)), cx, cy)
    if count < 3:
        return None
    a = np.stack(
        [
            sample_x[valid] - cx,
            sample_y[valid] - cy,
            np.ones(count, dtype=np.float64),
        ],
        axis=1,
    )
    inv = 1.0 / np.clip(depth_y[valid], args.depth_eps, args.max_value)
    try:
        coeff, _, rank, _ = np.linalg.lstsq(a, inv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.isfinite(coeff).all():
        return None
    pred = a @ coeff
    rmse = math.sqrt(float(np.mean((pred - inv) ** 2)))
    if args.temporal_sample_max_inv_rmse > 0.0 and rmse > args.temporal_sample_max_inv_rmse:
        return None
    return Plane(float(coeff[0]), float(coeff[1]), float(coeff[2]), cx, cy)


def plane_to_center(p: Plane, cx: float, cy: float) -> Plane:
    return Plane(
        p.a,
        p.b,
        p.c + p.a * (cx - p.cx) + p.b * (cy - p.cy),
        cx,
        cy,
    )


def eval_inv_plane_value(p: Plane, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return p.a * (gx - p.cx) + p.b * (gy - p.cy) + p.c


def inv_plane_to_depth_value(
    p: Plane,
    gx: np.ndarray,
    gy: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    inv = eval_inv_plane_value(p, gx, gy)
    inv_min = 1.0 / max(float(args.max_value), 1.0)
    inv_max = 1.0 / max(float(args.depth_eps), 1e-12)
    inv = np.clip(inv, inv_min, inv_max)
    return np.clip(1.0 / inv, 0.0, args.max_value)


def render_inv_depth_plane(
    p: Plane,
    xx: np.ndarray,
    yy: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    inv = p.a * xx + p.b * yy + p.c
    inv_min = 1.0 / max(float(args.max_value), 1.0)
    inv_max = 1.0 / max(float(args.depth_eps), 1e-12)
    inv = np.clip(inv, inv_min, inv_max)
    return np.clip(np.rint(1.0 / inv), 0, args.max_value).astype(np.float64)


def block_sse(orig: np.ndarray, recon: np.ndarray) -> float:
    d = np.asarray(orig, dtype=np.float64) - np.asarray(recon, dtype=np.float64)
    return float(np.sum(d * d))


# ============================================================
# Spatial neighbors
# ============================================================

def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def best_left(store: Sequence[LeafRecord], x: int, y: int, w: int, h: int) -> Optional[LeafRecord]:
    best = None
    best_overlap = 0
    for r in store:
        if r.x + r.w == x:
            o = overlap(r.y, r.y + r.h, y, y + h)
            if o > best_overlap:
                best, best_overlap = r, o
    return best


def best_top(store: Sequence[LeafRecord], x: int, y: int, w: int, h: int) -> Optional[LeafRecord]:
    best = None
    best_overlap = 0
    for r in store:
        if r.y + r.h == y:
            o = overlap(r.x, r.x + r.w, x, x + w)
            if o > best_overlap:
                best, best_overlap = r, o
    return best


def top_left(store: Sequence[LeafRecord], x: int, y: int) -> Optional[LeafRecord]:
    for r in store:
        if r.x + r.w == x and r.y + r.h == y:
            return r
    return None


def top_right(store: Sequence[LeafRecord], x: int, y: int, w: int) -> Optional[LeafRecord]:
    for r in store:
        if r.x == x + w and r.y + r.h == y:
            return r
    return None


# ============================================================
# Camera JSONL v2
# ============================================================

def rodrigues_to_matrix(rvec: Sequence[float]) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        x, y, z = r
        k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + k
    axis = r / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rt_to_4x4(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rodrigues_to_matrix(rvec)
    out[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def intrinsic_vec_to_matrix(v: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in v]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"invalid intrinsic {v}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_camera_json(path: str) -> Dict[str, Any]:
    header = None
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"camera JSONL parse error at line {line_no}: {exc}") from exc
            if obj.get("type") == "header":
                if header is not None:
                    raise ValueError("multiple camera headers")
                header = obj
            else:
                frames.append(obj)
    if header is None or not frames:
        raise ValueError("camera JSONL requires one header and frame records")
    for key in ("width", "height", "depth_scale", "depth_scale_precision", "intrinsic", "pose_mode"):
        if key not in header:
            raise KeyError(f"camera header missing {key}")
    return {"header": header, "frames": frames}


def get_depth_scale_real_from_header(header: Dict[str, Any]) -> float:
    precision = float(header["depth_scale_precision"])
    if precision <= 0.0:
        raise ValueError("depth_scale_precision must be positive")
    scale = float(header["depth_scale"]) / precision
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid real depth scale {scale}")
    return scale


def build_camera_lookup(camera_json: Dict[str, Any]) -> Dict[str, Any]:
    header = camera_json["header"]
    records = sorted(camera_json["frames"], key=lambda r: int(r["poc"]))
    pose_mode = str(header["pose_mode"])
    intr0 = header["intrinsic"]
    base_intr = np.array(
        [intr0["fx"], intr0["fy"], intr0["cx"], intr0["cy"]],
        dtype=np.float64,
    )
    z_sign = 1.0 if float(intr0.get("z_sign", 1.0)) > 0.0 else -1.0
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )
    depth_scale_real = get_depth_scale_real_from_header(header)

    by_poc: Dict[int, Dict[str, Any]] = {}
    by_frame_idx: Dict[int, Dict[str, Any]] = {}
    cur_intr = base_intr.copy()
    prev_w2c = np.eye(4, dtype=np.float64)

    for order, rec in enumerate(records):
        poc = int(rec["poc"])
        frame_idx = int(rec.get("frame_idx", poc))
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64)
        cur_intr = base_intr.copy() if fixed_intrinsic else cur_intr + delta
        k = intrinsic_vec_to_matrix(cur_intr)
        t_rec = rt_to_4x4(rec["rvec"], rec["tvec"])

        if pose_mode == "current_to_previous":
            if order == 0:
                w2c = np.eye(4, dtype=np.float64)
            else:
                w2c = np.linalg.inv(t_rec) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            w2c = t_rec
        else:
            raise ValueError(f"unsupported pose_mode {pose_mode}")

        c2w = np.linalg.inv(w2c)
        cam = {
            "poc": poc,
            "frame_idx": frame_idx,
            "K": k,
            "W2C": w2c,
            "C2W": c2w,
            "z_sign": z_sign,
            "depth_scale_real": depth_scale_real,
        }
        by_poc[poc] = cam
        by_frame_idx.setdefault(frame_idx, cam)
        prev_w2c = w2c

    return {"header": header, "by_poc": by_poc, "by_frame_idx": by_frame_idx}


def get_camera(lookup: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    if frame_idx in lookup["by_poc"]:
        return lookup["by_poc"][frame_idx]
    if frame_idx in lookup["by_frame_idx"]:
        return lookup["by_frame_idx"][frame_idx]
    raise KeyError(f"camera for frame/POC {frame_idx} not found")


def scale_camera_intrinsic(cam: Dict[str, Any], scale: int) -> Dict[str, Any]:
    out = dict(cam)
    k = np.asarray(cam["K"], dtype=np.float64).copy()
    k[0, 0] /= scale
    k[1, 1] /= scale
    k[0, 2] /= scale
    k[1, 2] /= scale
    out["K"] = k
    return out


def get_depth_scale_real(cam: Dict[str, Any]) -> float:
    return float(cam["depth_scale_real"])


# ============================================================
# Camera geometry / five-point temporal resampling
# ============================================================

def pixel_rays_camera(u: np.ndarray, v: np.ndarray, cam: Dict[str, Any]) -> np.ndarray:
    k = np.asarray(cam["K"], dtype=np.float64)
    z_sign = float(cam["z_sign"])
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rx = (u - k[0, 2]) / k[0, 0]
    ry = (v - k[1, 2]) / k[1, 1]
    rz = np.full_like(rx, z_sign)
    return np.stack([rx, ry, rz], axis=-1)


def project_camera_points(points_cam: np.ndarray, cam: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points_cam, dtype=np.float64)
    k = np.asarray(cam["K"], dtype=np.float64)
    z_sign = float(cam["z_sign"])
    depth = z_sign * p[..., 2]
    front = np.isfinite(depth) & (depth > 1e-12)
    safe = np.where(front, depth, 1.0)
    u = k[0, 0] * (p[..., 0] / safe) + k[0, 2]
    v = k[1, 1] * (p[..., 1] / safe) + k[1, 2]
    return u, v, depth, front


def transform_camera_points(points_src: np.ndarray, cam_src: Dict[str, Any], cam_tgt: Dict[str, Any]) -> np.ndarray:
    p = np.asarray(points_src, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(cam_tgt["W2C"]) @ np.asarray(cam_src["C2W"])
    ph = np.concatenate([p, np.ones((p.shape[0], 1), dtype=np.float64)], axis=1)
    return (ph @ m.T)[:, :3]


def fit_3d_plane(points: np.ndarray) -> Optional[np.ndarray]:
    if points.shape[0] < 3:
        return None
    center = np.mean(points, axis=0)
    q = points - center
    try:
        _, s, vh = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(s) < 2 or s[1] < 1e-9:
        return None
    n = vh[-1]
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return None
    n = n / norm
    return np.array([n[0], n[1], n[2], -float(np.dot(n, center))], dtype=np.float64)


def image_inv_plane_to_3d_plane(
    leaf: LeafRecord,
    cam: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    ns = max(2, int(args.plane_warp_samples))
    xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)
    ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    depth_y = inv_plane_to_depth_value(leaf.plane, uu, vv, args)
    depth_real = depth_y * get_depth_scale_real(cam)
    rays = pixel_rays_camera(uu, vv, cam)
    points = rays.reshape(-1, 3) * depth_real.reshape(-1, 1)
    valid = np.isfinite(points).all(axis=1) & (depth_real.reshape(-1) > 0.0)
    return fit_3d_plane(points[valid])


def transform_plane_src_to_tgt(
    plane_src: np.ndarray,
    cam_src: Dict[str, Any],
    cam_tgt: Dict[str, Any],
) -> Optional[np.ndarray]:
    m = np.asarray(cam_tgt["W2C"]) @ np.asarray(cam_src["C2W"])
    try:
        plane_tgt = np.linalg.inv(m).T @ plane_src
    except np.linalg.LinAlgError:
        return None
    norm = float(np.linalg.norm(plane_tgt[:3]))
    if norm < 1e-12:
        return None
    return plane_tgt / norm


def project_reference_leaf_polygon(
    leaf: LeafRecord,
    cam_ref: Dict[str, Any],
    cam_cur: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    us = np.array(
        [leaf.x - 0.5, leaf.x + leaf.w - 0.5, leaf.x + leaf.w - 0.5, leaf.x - 0.5],
        dtype=np.float64,
    )
    vs = np.array(
        [leaf.y - 0.5, leaf.y - 0.5, leaf.y + leaf.h - 0.5, leaf.y + leaf.h - 0.5],
        dtype=np.float64,
    )
    depth_y = inv_plane_to_depth_value(leaf.plane, us, vs, args)
    points_ref = pixel_rays_camera(us, vs, cam_ref) * (
        depth_y * get_depth_scale_real(cam_ref)
    )[:, None]
    points_cur = transform_camera_points(points_ref, cam_ref, cam_cur)
    pu, pv, _, front = project_camera_points(points_cur, cam_cur)
    if not np.all(front & np.isfinite(pu) & np.isfinite(pv)):
        return None
    poly = np.stack([pu, pv], axis=1)
    if polygon_area(poly) <= 1e-8:
        return None
    return poly


def polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    # Boundary-inclusive ray casting.
    p = np.asarray(poly, dtype=np.float64)
    inside = False
    n = p.shape[0]
    eps = 1e-9
    for i in range(n):
        a = p[i]
        b = p[(i + 1) % n]
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        if abs(cross) <= eps and min(a[0], b[0]) - eps <= x <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= y <= max(a[1], b[1]) + eps:
            return True
        if (a[1] > y) != (b[1] > y):
            xin = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x <= xin + eps:
                inside = not inside
    return inside


def plane_depth_at_pixel(
    plane3d_cur: np.ndarray,
    cam_cur: Dict[str, Any],
    x: float,
    y: float,
) -> float:
    ray = pixel_rays_camera(np.array([x]), np.array([y]), cam_cur)[0]
    denom = float(np.dot(plane3d_cur[:3], ray))
    if abs(denom) < 1e-12:
        return float("inf")
    d = -float(plane3d_cur[3]) / denom
    return d if np.isfinite(d) and d > 0.0 else float("inf")


def build_projected_leaf_cache(
    ref_store: Sequence[LeafRecord],
    cam_ref: Dict[str, Any],
    cam_cur: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    records = []
    for leaf in ref_store:
        plane_ref = image_inv_plane_to_3d_plane(leaf, cam_ref, args)
        if plane_ref is None:
            continue
        plane_cur = transform_plane_src_to_tgt(plane_ref, cam_ref, cam_cur)
        poly = project_reference_leaf_polygon(leaf, cam_ref, cam_cur, args)
        if plane_cur is None or poly is None:
            continue
        records.append(
            {
                "leaf": leaf,
                "plane3d_cur": plane_cur,
                "polygon": poly,
                "bbox": (
                    float(np.min(poly[:, 0])),
                    float(np.min(poly[:, 1])),
                    float(np.max(poly[:, 0])),
                    float(np.max(poly[:, 1])),
                ),
            }
        )
    return records


def get_projected_leaf_cache(
    ctx: PlaneWarpContext,
    side: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if side in ctx.projected_leaf_cache:
        return ctx.projected_leaf_cache[side]
    if side == "l0":
        store, cam_ref = ctx.l0_store, ctx.cam_l0_low
    elif side == "l1":
        store, cam_ref = ctx.l1_store, ctx.cam_l1_low
    else:
        raise ValueError(f"bad side {side}")
    records = [] if not store or cam_ref is None else build_projected_leaf_cache(
        store, cam_ref, ctx.cam_cur_low, args
    )
    ctx.projected_leaf_cache[side] = records
    return records


def block_five_sample_points(x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    # Pixel-center locations near the four corners plus center.
    xs = np.array(
        [
            float(x),
            float(x + w - 1),
            float(x),
            float(x + w - 1),
            x + (w - 1) / 2.0,
        ],
        dtype=np.float64,
    )
    ys = np.array(
        [
            float(y),
            float(y),
            float(y + h - 1),
            float(y + h - 1),
            y + (h - 1) / 2.0,
        ],
        dtype=np.float64,
    )
    return xs, ys


def sample_visible_depths_from_projected_leaves(
    records: Sequence[Dict[str, Any]],
    cam_cur: Dict[str, Any],
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_y = np.full(sample_x.shape, np.nan, dtype=np.float64)
    valid = np.zeros(sample_x.shape, dtype=bool)
    depth_scale_real = get_depth_scale_real(cam_cur)

    for i, (sx, sy) in enumerate(zip(sample_x, sample_y)):
        nearest = float("inf")
        for rec in records:
            bx0, by0, bx1, by1 = rec["bbox"]
            if sx < bx0 - 1e-9 or sx > bx1 + 1e-9 or sy < by0 - 1e-9 or sy > by1 + 1e-9:
                continue
            if not point_in_polygon(float(sx), float(sy), rec["polygon"]):
                continue
            d = plane_depth_at_pixel(rec["plane3d_cur"], cam_cur, float(sx), float(sy))
            if d < nearest:
                nearest = d
        if np.isfinite(nearest):
            sample = nearest / depth_scale_real
            if args.depth_eps <= sample <= args.max_value:
                depth_y[i] = sample
                valid[i] = True
    return depth_y, valid


def make_five_point_temporal_candidates(
    ctx: Optional[PlaneWarpContext],
    x: int,
    y: int,
    w: int,
    h: int,
    cx: float,
    cy: float,
    args: argparse.Namespace,
) -> List[Tuple[str, Plane]]:
    if ctx is None:
        return []

    sx, sy = block_five_sample_points(x, y, w, h)
    rec0 = get_projected_leaf_cache(ctx, "l0", args)
    d0, v0 = sample_visible_depths_from_projected_leaves(
        rec0, ctx.cam_cur_low, sx, sy, args
    )
    p0 = fit_inv_plane_from_samples(sx, sy, d0, v0, cx, cy, args)

    p1 = None
    d1 = np.full_like(d0, np.nan)
    v1 = np.zeros_like(v0)
    if ctx.l1_store is not None and ctx.cam_l1_low is not None:
        rec1 = get_projected_leaf_cache(ctx, "l1", args)
        d1, v1 = sample_visible_depths_from_projected_leaves(
            rec1, ctx.cam_cur_low, sx, sy, args
        )
        p1 = fit_inv_plane_from_samples(sx, sy, d1, v1, cx, cy, args)

    out: List[Tuple[str, Plane]] = []
    if p0 is not None and p1 is not None:
        valid_union = v0 | v1
        if not np.any(valid_union):
            return []
        both = v0 & v1
        only0 = v0 & ~v1
        only1 = v1 & ~v0
        vavg = v0 | v1
        davg = np.full_like(d0, np.nan)
        davg[only0] = d0[only0]
        davg[only1] = d1[only1]
        davg[both] = 0.5 * (d0[both] + d1[both])
        pavg = fit_inv_plane_from_samples(sx, sy, davg, vavg, cx, cy, args)
        if pavg is not None:
            out.append(("plane_warp_avg", pavg))
    if p0 is not None:
        out.append(("plane_warp_l0", p0))
    if p1 is not None:
        out.append(("plane_warp_l1", p1))
    return out


def make_candidates(
    store: Sequence[LeafRecord],
    x: int,
    y: int,
    w: int,
    h: int,
    cx: float,
    cy: float,
    args: argparse.Namespace,
    plane_warp_ctx: Optional[PlaneWarpContext],
) -> List[Tuple[str, Plane]]:
    candidates: List[Tuple[str, Plane]] = []
    converted: Dict[str, Plane] = {}

    if args.plane_warp_candidate:
        for name, p in make_five_point_temporal_candidates(
            plane_warp_ctx, x, y, w, h, cx, cy, args
        ):
            candidates.append((name, p))
            converted[name] = p

    spatial = [
        ("left", best_left(store, x, y, w, h)),
        ("top", best_top(store, x, y, w, h)),
        ("top_left", top_left(store, x, y)),
        ("top_right", top_right(store, x, y, w)),
    ]
    for name, rec in spatial:
        if rec is not None:
            p = plane_to_center(rec.plane, cx, cy)
            candidates.append((name, p))
            converted[name] = p

    if "left" in converted and "top" in converted:
        l, t = converted["left"], converted["top"]
        candidates.append(
            (
                "avg_left_top",
                Plane(
                    0.5 * (l.a + t.a),
                    0.5 * (l.b + t.b),
                    0.5 * (l.c + t.c),
                    cx,
                    cy,
                ),
            )
        )

    return candidates[: args.max_candidates]


# ============================================================
# Cached Hadamard SATD
# ============================================================

class HadamardCache:
    def __init__(self):
        self.cache_1d: Dict[int, np.ndarray] = {}

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    def matrix(self, n: int) -> np.ndarray:
        n2 = self._next_pow2(n)
        if n2 not in self.cache_1d:
            h = np.array([[1.0]], dtype=np.float64)
            while h.shape[0] < n2:
                h = np.block([[h, h], [h, -h]])
            self.cache_1d[n2] = h
        return self.cache_1d[n2]

    def satd(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, dtype=np.float64)
        if r.size == 0:
            return 0.0
        h, w = r.shape
        hh = self.matrix(h)
        hw = self.matrix(w)
        padded = np.zeros((hh.shape[0], hw.shape[0]), dtype=np.float64)
        padded[:h, :w] = r
        coeff = hh @ padded @ hw.T
        return float(np.sum(np.abs(coeff)) / math.sqrt(hh.shape[0] * hw.shape[0]))


# ============================================================
# Projection-domain candidate distortion
# ============================================================

def relative_camera_transform(
    cam_source: Dict[str, Any],
    cam_target: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(cam_target["W2C"]) @ np.asarray(cam_source["C2W"])
    return m[:3, :3], m[:3, 3]


def make_backward_map_for_region(
    depth_y_region: np.ndarray,
    x0: int,
    y0: int,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth_y_region.shape
    xs, ys = np.meshgrid(
        np.arange(x0, x0 + w, dtype=np.float64),
        np.arange(y0, y0 + h, dtype=np.float64),
    )
    rays = pixel_rays_camera(xs, ys, cam_cur)
    depth_real = np.asarray(depth_y_region, dtype=np.float64) * get_depth_scale_real(cam_cur)
    points_cur = rays * depth_real[..., None]

    r, t = relative_camera_transform(cam_cur, cam_ref)
    points_ref = points_cur @ r.T + t.reshape(1, 1, 3)
    map_x, map_y, _, front = project_camera_points(points_ref, cam_ref)

    # Coordinates outside the reference picture are intentionally kept valid.
    # bilinear_sample_region() reflects them back into the reference picture.
    valid = (
        front
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(depth_real)
        & (depth_real > 0.0)
    )
    return map_x, map_y, valid


def reflect_coordinates(coord: np.ndarray, size: int) -> np.ndarray:
    """Reflect coordinates like symmetric image padding, excluding edge repeat."""
    c = np.asarray(coord, dtype=np.float64)
    if size <= 1:
        return np.zeros_like(c)
    period = 2.0 * (size - 1)
    m = np.mod(c, period)
    return np.where(m <= size - 1, m, period - m)


def bilinear_sample_region(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    fill: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    valid2 = np.asarray(valid, dtype=bool) & finite

    # Out-of-picture coordinates are reflected into the reference picture.
    mx = reflect_coordinates(np.where(finite, map_x, 0.0), w)
    my = reflect_coordinates(np.where(finite, map_y, 0.0), h)

    x0 = np.floor(mx).astype(np.int64)
    y0 = np.floor(my).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx, dy = mx - x0, my - y0

    out = (
        (1.0 - dx) * (1.0 - dy) * img[y0, x0]
        + dx * (1.0 - dy) * img[y0, x1]
        + (1.0 - dx) * dy * img[y1, x0]
        + dx * dy * img[y1, x1]
    )
    return np.where(valid2, out, fill), valid2


def reference_fill_block(
    ref_y: np.ndarray,
    x0: int,
    y0: int,
    h: int,
    w: int,
    mode: str,
    max_value: int,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((h, w), dtype=np.float64)
    if mode == "neutral":
        return np.full((h, w), max_value // 2, dtype=np.float64)
    return np.asarray(ref_y[y0 : y0 + h, x0 : x0 + w], dtype=np.float64)


def candidate_projection_satd(
    recon_depth_low: np.ndarray,
    x_low: int,
    y_low: int,
    ctx: ProjectionRDOContext,
    hadamard: HadamardCache,
) -> Optional[float]:
    scale = ctx.depth_scale
    depth_full = upsample_depth_integer(
        recon_depth_low, scale, ctx.upsample_mode
    )
    x0 = x_low * scale
    y0 = y_low * scale
    full_h, full_w = ctx.cur_y.shape
    h = min(depth_full.shape[0], full_h - y0)
    w = min(depth_full.shape[1], full_w - x0)
    if h <= 0 or w <= 0:
        return 0.0
    depth_full = depth_full[:h, :w]
    target = np.asarray(ctx.cur_y[y0 : y0 + h, x0 : x0 + w], dtype=np.float64)

    def warp_one(ref_y: np.ndarray, cam_ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        mx, my, valid = make_backward_map_for_region(
            depth_full, x0, y0, ctx.cam_cur_full, cam_ref
        )
        fill = reference_fill_block(ref_y, x0, y0, h, w, ctx.invalid_fill, ctx.max_value)
        return bilinear_sample_region(ref_y, mx, my, valid, fill)

    if ctx.prediction_type == "P":
        if ctx.l0_y is None or ctx.cam_l0_full is None:
            raise RuntimeError("P RDO context is missing L0")
        pred, valid0 = warp_one(ctx.l0_y, ctx.cam_l0_full)
        if not np.any(valid0):
            return None
    elif ctx.prediction_type == "B":
        if (
            ctx.l0_y is None
            or ctx.l1_y is None
            or ctx.cam_l0_full is None
            or ctx.cam_l1_full is None
        ):
            raise RuntimeError("B RDO context is missing L0/L1")
        p0, v0 = warp_one(ctx.l0_y, ctx.cam_l0_full)
        p1, v1 = warp_one(ctx.l1_y, ctx.cam_l1_full)
        both = v0 & v1
        only0 = v0 & ~v1
        only1 = v1 & ~v0
        if ctx.invalid_fill == "zero":
            pred = np.zeros_like(target)
        elif ctx.invalid_fill == "neutral":
            pred = np.full_like(target, ctx.max_value // 2)
        else:
            pred = 0.5 * (
                ctx.l0_y[y0 : y0 + h, x0 : x0 + w]
                + ctx.l1_y[y0 : y0 + h, x0 : x0 + w]
            )
        pred[only0] = p0[only0]
        pred[only1] = p1[only1]
        pred[both] = 0.5 * (p0[both] + p1[both])
    else:
        raise ValueError("projection SATD is defined only for P/B pictures")

    return hadamard.satd(target - pred)


def candidate_distortion(
    original_depth_block: np.ndarray,
    recon_depth_block: np.ndarray,
    x: int,
    y: int,
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> float:
    if rdo_ctx is None:
        return block_sse(original_depth_block, recon_depth_block)
    satd = candidate_projection_satd(recon_depth_block, x, y, rdo_ctx, hadamard)
    # If every projected sample is geometrically invalid (for example, all
    # points are behind the reference camera), use zero distortion. RDO then
    # falls back to the candidate requiring the fewest estimated bits.
    return 0.0 if satd is None else satd


# ============================================================
# Mode evaluation
# ============================================================

def eval_direct(
    block: np.ndarray,
    actual: Plane,
    xx: np.ndarray,
    yy: np.ndarray,
    x: int,
    y: int,
    args: argparse.Namespace,
    adaptive: Optional[Dict[str, Any]],
    avail_modes: Sequence[str],
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> ModeResult:
    qa = 0 if args.c_only_plane else quantize(actual.a, args.qa)
    qb = 0 if args.c_only_plane else quantize(actual.b, args.qb)
    qc = quantize(actual.c, args.qc)
    p = Plane(
        0.0 if args.c_only_plane else dequantize(qa, args.qa),
        0.0 if args.c_only_plane else dequantize(qb, args.qb),
        dequantize(qc, args.qc),
        actual.cx,
        actual.cy,
    )
    recon = render_inv_depth_plane(p, xx, yy, args)
    distortion = candidate_distortion(block, recon, x, y, rdo_ctx, hadamard)
    bits = adaptive["mode"].bits("direct", avail_modes) if adaptive else float(args.mode_bits)
    if not args.c_only_plane:
        bits += exp_golomb_len_signed(qa) + exp_golomb_len_signed(qb)
    bits += exp_golomb_len_signed(qc)
    return ModeResult(
        "direct", "none", p, recon, bits, distortion,
        distortion + args.lambda_rd * bits, (qa, qb, qc)
    )


def eval_copy(
    block: np.ndarray,
    candidates: Sequence[Tuple[str, Plane]],
    xx: np.ndarray,
    yy: np.ndarray,
    x: int,
    y: int,
    args: argparse.Namespace,
    adaptive: Optional[Dict[str, Any]],
    avail_modes: Sequence[str],
    avail_cands: Sequence[str],
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> List[ModeResult]:
    out = []
    for i, (name, p) in enumerate(candidates):
        recon = render_inv_depth_plane(p, xx, yy, args)
        distortion = candidate_distortion(block, recon, x, y, rdo_ctx, hadamard)
        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(candidates)))
        else:
            bits = adaptive["mode"].bits("copy", avail_modes)
            if "copy_candidate_unary" in adaptive:
                bits += unary_candidate_bits(i, len(candidates), adaptive["copy_candidate_unary"])
            else:
                bits += adaptive["candidate"].bits(name, avail_cands)
        out.append(
            ModeResult(
                "copy", name, p, recon, bits, distortion,
                distortion + args.lambda_rd * bits, ()
            )
        )
    return out


def eval_delta(
    block: np.ndarray,
    actual: Plane,
    candidates: Sequence[Tuple[str, Plane]],
    xx: np.ndarray,
    yy: np.ndarray,
    x: int,
    y: int,
    args: argparse.Namespace,
    adaptive: Optional[Dict[str, Any]],
    avail_modes: Sequence[str],
    avail_cands: Sequence[str],
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> List[ModeResult]:
    out = []
    for name, pred in candidates:
        qda = 0 if args.c_only_plane else quantize(actual.a - pred.a, args.qa)
        qdb = 0 if args.c_only_plane else quantize(actual.b - pred.b, args.qb)
        qdc = quantize(actual.c - pred.c, args.qc)
        p = Plane(
            0.0 if args.c_only_plane else pred.a + dequantize(qda, args.qa),
            0.0 if args.c_only_plane else pred.b + dequantize(qdb, args.qb),
            pred.c + dequantize(qdc, args.qc),
            actual.cx,
            actual.cy,
        )
        recon = render_inv_depth_plane(p, xx, yy, args)
        distortion = candidate_distortion(block, recon, x, y, rdo_ctx, hadamard)

        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(candidates)))
        else:
            bits = adaptive["mode"].bits("delta", avail_modes)
            bits += adaptive["candidate"].bits(name, avail_cands)

        if adaptive is not None and "delta_res_abs_a" in adaptive:
            if not args.c_only_plane:
                bits += adaptive_signed_residual_bits(qda, adaptive["delta_res_abs_a"], adaptive["delta_abs_max"])
                bits += adaptive_signed_residual_bits(qdb, adaptive["delta_res_abs_b"], adaptive["delta_abs_max"])
            bits += adaptive_signed_residual_bits(qdc, adaptive["delta_res_abs_c"], adaptive["delta_abs_max"])
        else:
            if not args.c_only_plane:
                bits += exp_golomb_len_signed(qda) + exp_golomb_len_signed(qdb)
            bits += exp_golomb_len_signed(qdc)

        out.append(
            ModeResult(
                "delta", name, p, recon, bits, distortion,
                distortion + args.lambda_rd * bits, (qda, qdb, qdc)
            )
        )
    return out


def eval_leaf(
    padded: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    depth: int,
    parent: Optional[CSNode],
    args: argparse.Namespace,
    grid: GridCache,
    store: Sequence[LeafRecord],
    adaptive: Optional[Dict[str, Any]],
    plane_warp_ctx: Optional[PlaneWarpContext],
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> CSNode:
    block = padded[y : y + h, x : x + w]
    cx = x + (w - 1) / 2.0
    cy = y + (h - 1) / 2.0
    xx, yy, pinv = grid.get(w, h)
    actual = fit_inv_depth_plane_from_depth_block(block, pinv, cx, cy, args)
    candidates = make_candidates(store, x, y, w, h, cx, cy, args, plane_warp_ctx)
    avail_cands = [name for name, _ in candidates]
    avail_modes = ["direct", "copy", "delta"] if candidates else ["direct"]

    modes = [
        eval_direct(
            block, actual, xx, yy, x, y, args, adaptive, avail_modes,
            rdo_ctx, hadamard
        )
    ]
    if candidates:
        modes += eval_copy(
            block, candidates, xx, yy, x, y, args, adaptive, avail_modes,
            avail_cands, rdo_ctx, hadamard
        )
        modes += eval_delta(
            block, actual, candidates, xx, yy, x, y, args, adaptive, avail_modes,
            avail_cands, rdo_ctx, hadamard
        )

    best = min(modes, key=lambda r: r.cost)
    return CSNode(
        x=x, y=y, w=w, h=h, depth=depth, parent=parent,
        split="leaf", best=best, actual=actual,
        avail_modes=avail_modes, avail_cands=avail_cands,
        bits=best.bits, distortion=best.distortion, cost=best.cost
    )


# ============================================================
# Recursive partition coding
# ============================================================

def add_leaves_to_store(node: CSNode, store: List[LeafRecord]) -> None:
    if node.is_leaf():
        store.append(LeafRecord(node.x, node.y, node.w, node.h, node.best.plane))
        return
    for child in node.children:
        add_leaves_to_store(child, store)


def finalize_partition_node(
    node: CSNode,
    split_bits: float,
    available_splits: Sequence[str],
    args: argparse.Namespace,
) -> CSNode:
    """Attach partition syntax cost after leaf/children distortion is known."""
    node.split_bits = float(split_bits)
    node.avail_splits = list(available_splits)
    node.bits += node.split_bits
    node.cost += args.lambda_rd * node.split_bits
    return node


def make_parent_node(
    x: int,
    y: int,
    w: int,
    h: int,
    depth: int,
    parent: Optional[CSNode],
    split: str,
    children: Sequence[CSNode],
    split_bits: float,
    available_splits: Sequence[str],
    args: argparse.Namespace,
) -> CSNode:
    node = CSNode(
        x=x, y=y, w=w, h=h, depth=depth, parent=parent,
        split=split, children=list(children),
        avail_splits=list(available_splits),
    )
    for child in node.children:
        child.parent = node
    node.bits = sum(child.bits for child in node.children)
    node.distortion = sum(child.distortion for child in node.children)
    node.cost = sum(child.cost for child in node.children)
    return finalize_partition_node(node, split_bits, available_splits, args)


def binary_split_specs(
    split: str, x: int, y: int, w: int, h: int
) -> List[Tuple[int, int, int, int]]:
    if split == "bh":
        h0 = h // 2
        return [(x, y, w, h0), (x, y + h0, w, h - h0)]
    if split == "bv":
        w0 = w // 2
        return [(x, y, w0, h), (x + w0, y, w - w0, h)]
    raise ValueError(f"not a binary split: {split}")


def qt_split_specs(x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    w0, h0 = w // 2, h // 2
    return [
        (x, y, w0, h0),
        (x + w0, y, w - w0, h0),
        (x, y + h0, w0, h - h0),
        (x + w0, y + h0, w - w0, h - h0),
    ]


def encode_node(
    padded: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    depth: int,
    parent: Optional[CSNode],
    args: argparse.Namespace,
    grid: GridCache,
    store: Sequence[LeafRecord],
    adaptive: Optional[Dict[str, Any]],
    plane_warp_ctx: Optional[PlaneWarpContext],
    rdo_ctx: Optional[ProjectionRDOContext],
    hadamard: HadamardCache,
) -> CSNode:
    available = available_split_types(depth, w, h, args.max_qt_depth)
    options: List[CSNode] = []

    leaf = eval_leaf(
        padded, x, y, w, h, depth, parent, args, grid, store,
        adaptive, plane_warp_ctx, rdo_ctx, hadamard,
    )
    options.append(
        finalize_partition_node(
            leaf,
            split_type_bits(adaptive, depth, "leaf", available),
            available,
            args,
        )
    )

    # Binary partitions use two terminal children, matching the original
    # simulator. The common evaluation path removes duplicated BH/BV code.
    for split in ("bh", "bv"):
        if split not in available:
            continue
        local_store = list(store)
        children: List[CSNode] = []
        for child_x, child_y, child_w, child_h in binary_split_specs(
            split, x, y, w, h
        ):
            child = eval_leaf(
                padded, child_x, child_y, child_w, child_h, depth + 1, None,
                args, grid, local_store, adaptive, plane_warp_ctx, rdo_ctx,
                hadamard,
            )
            children.append(child)
            add_leaves_to_store(child, local_store)
        options.append(
            make_parent_node(
                x, y, w, h, depth, parent, split, children,
                split_type_bits(adaptive, depth, split, available),
                available, args,
            )
        )

    if "qt" in available:
        local_store = list(store)
        children = []
        for child_x, child_y, child_w, child_h in qt_split_specs(x, y, w, h):
            child = encode_node(
                padded, child_x, child_y, child_w, child_h, depth + 1, None,
                args, grid, local_store, adaptive, plane_warp_ctx, rdo_ctx,
                hadamard,
            )
            children.append(child)
            add_leaves_to_store(child, local_store)
        options.append(
            make_parent_node(
                x, y, w, h, depth, parent, "qt", children,
                split_type_bits(adaptive, depth, "qt", available),
                available, args,
            )
        )

    best = min(options, key=lambda node: node.cost)
    best.parent = parent
    return best

def commit_node(
    node: CSNode,
    store: List[LeafRecord],
    adaptive: Optional[Dict[str, Any]],
    writer: Optional[csv.DictWriter],
    frame_idx: int,
    args: argparse.Namespace,
) -> None:
    split_type_update(adaptive, node)
    if not node.is_leaf():
        for child in node.children:
            commit_node(child, store, adaptive, writer, frame_idx, args)
        return

    b = node.best
    if adaptive is not None:
        if len(node.avail_modes) > 1:
            adaptive["mode"].update(b.mode)
        if b.mode == "copy" and len(node.avail_cands) > 1:
            if "copy_candidate_unary" in adaptive:
                unary_candidate_update(
                    node.avail_cands.index(b.candidate_name),
                    len(node.avail_cands),
                    adaptive["copy_candidate_unary"],
                )
            else:
                adaptive["candidate"].update(b.candidate_name)
        elif b.mode == "delta" and len(node.avail_cands) > 1:
            adaptive["candidate"].update(b.candidate_name)

        if b.mode == "delta" and "delta_res_abs_a" in adaptive:
            residual_items = (
                [(b.q_values[2], "c")]
                if args.c_only_plane
                else list(zip(b.q_values, "abc"))
            )
            for q, k in residual_items:
                adaptive_signed_residual_update(
                    q, adaptive[f"delta_res_abs_{k}"], adaptive["delta_abs_max"]
                )

    store.append(LeafRecord(node.x, node.y, node.w, node.h, b.plane))

    if writer is not None:
        q = list(b.q_values) + ["", "", ""]
        writer.writerow(
            {
                "frame": frame_idx,
                "bx": node.x,
                "by": node.y,
                "block_w": node.w,
                "block_h": node.h,
                "qt_depth": node.depth,
                "split_type": node.split,
                "mode": b.mode,
                "candidate": b.candidate_name,
                "bits": node.bits,
                "split_bits": node.split_bits,
                "distortion": node.distortion,
                "cost": node.cost,
                "q0": q[0],
                "q1": q[1],
                "q2": q[2],
                "actual_inv_a": node.actual.a,
                "actual_inv_b": node.actual.b,
                "actual_inv_c": node.actual.c,
                "recon_inv_a": b.plane.a,
                "recon_inv_b": b.plane.b,
                "recon_inv_c": b.plane.c,
            }
        )


def paint(node: CSNode, recon: np.ndarray) -> None:
    if node.is_leaf():
        recon[node.y : node.y + node.h, node.x : node.x + node.w] = node.best.recon_block
        return
    for child in node.children:
        paint(child, recon)


def collect(node: CSNode, stats: Dict[str, Any]) -> None:
    stats["split_bits"] += node.split_bits
    if node.split == "qt":
        stats["qt_nodes"] += 1
    elif node.split == "bh":
        stats["bin_h_nodes"] += 1
    elif node.split == "bv":
        stats["bin_v_nodes"] += 1

    if node.is_leaf():
        b = node.best
        stats["leaf_blocks"] += 1
        stats[f"{b.mode}_blocks"] += 1
        key = f"candidate_{b.candidate_name}_count"
        stats[key] = stats.get(key, 0) + 1
        if b.mode == "delta":
            stats["delta_mode_count"] += 1
            if b.q_values == (0, 0, 0):
                stats["zero_delta_blocks"] += 1
        return
    for child in node.children:
        collect(child, stats)


# ============================================================
# Frame simulation
# ============================================================

def pad_to_block_multiple(img: np.ndarray, block_size: int) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    return img.copy(), h + ph, w + pw


def compute_metrics(
    orig: np.ndarray,
    recon: np.ndarray,
    maxv: float,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    d = np.asarray(orig, dtype=np.float64) - np.asarray(recon, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return {k: float("nan") for k in ("mae", "mse", "rmse", "psnr", "max_error")}
        d = d[mask]
    mse = float(np.mean(d * d))
    return {
        "mae": float(np.mean(np.abs(d))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": float("inf") if mse == 0.0 else 10.0 * math.log10(maxv * maxv / mse),
        "max_error": float(np.max(np.abs(d))),
    }


def simulate_zero_depth_frame(
    depth_low: np.ndarray,
    frame_idx: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any], List[LeafRecord]]:
    recon = np.zeros_like(depth_low, dtype=np.float64)
    m = compute_metrics(depth_low, recon, args.max_value)
    summary = {
        "frame": frame_idx,
        "width_low": depth_low.shape[1],
        "height_low": depth_low.shape[0],
        "zero_depth_picture": 1,
        "depth_bits": 0.0,
        "depth_bpp_low": 0.0,
        "depth_bpp_full": 0.0,
        "rdo_distortion": 0.0,
        "rdo_distortion_type": "bypass_zero_depth",
        "depth_mae": m["mae"],
        "depth_mse": m["mse"],
        "depth_rmse": m["rmse"],
        "depth_psnr": m["psnr"],
        "depth_max_error": m["max_error"],
        "leaf_blocks": 0,
        "qt_nodes": 0,
        "bin_h_nodes": 0,
        "bin_v_nodes": 0,
        "split_bits": 0.0,
        "direct_blocks": 0,
        "copy_blocks": 0,
        "delta_blocks": 0,
        "direct_ratio": 0.0,
        "copy_ratio": 0.0,
        "delta_ratio": 0.0,
        "zero_delta_blocks": 0,
        "zero_delta_ratio_in_delta": 0.0,
    }
    return recon, summary, []


def simulate_one_depth_frame(
    depth_low: np.ndarray,
    frame_idx: int,
    args: argparse.Namespace,
    grid: GridCache,
    hadamard: HadamardCache,
    writer: Optional[csv.DictWriter],
    adaptive: Optional[Dict[str, Any]],
    plane_warp_ctx: Optional[PlaneWarpContext],
    rdo_ctx: Optional[ProjectionRDOContext],
) -> Tuple[np.ndarray, Dict[str, Any], List[LeafRecord]]:
    h, w = depth_low.shape
    padded, hp, wp = pad_to_block_multiple(depth_low, args.block_size)
    recon = np.zeros_like(padded, dtype=np.float64)
    store: List[LeafRecord] = []
    total_bits = 0.0
    total_distortion = 0.0
    root_count = 0
    stats: Dict[str, Any] = {
        "leaf_blocks": 0,
        "qt_nodes": 0,
        "bin_h_nodes": 0,
        "bin_v_nodes": 0,
        "split_bits": 0.0,
        "direct_blocks": 0,
        "copy_blocks": 0,
        "delta_blocks": 0,
        "zero_delta_blocks": 0,
        "delta_mode_count": 0,
    }

    for y in range(0, hp, args.block_size):
        for x in range(0, wp, args.block_size):
            root_count += 1
            root = encode_node(
                padded, x, y, args.block_size, args.block_size, 0, None,
                args, grid, store, adaptive, plane_warp_ctx, rdo_ctx, hadamard
            )
            commit_node(root, store, adaptive, writer, frame_idx, args)
            paint(root, recon)
            collect(root, stats)
            total_bits += root.bits
            total_distortion += root.distortion

    rec = recon[:h, :w]
    m = compute_metrics(depth_low, rec, args.max_value)
    leaves = max(int(stats["leaf_blocks"]), 1)
    summary: Dict[str, Any] = {
        "frame": frame_idx,
        "width_low": w,
        "height_low": h,
        "padded_width_low": wp,
        "padded_height_low": hp,
        "zero_depth_picture": 0,
        "root_block_size_low": args.block_size,
        "max_qt_depth": args.max_qt_depth,
        "num_roots": root_count,
        "leaf_blocks": int(stats["leaf_blocks"]),
        "qt_nodes": int(stats["qt_nodes"]),
        "bin_h_nodes": int(stats["bin_h_nodes"]),
        "bin_v_nodes": int(stats["bin_v_nodes"]),
        "split_bits": float(stats["split_bits"]),
        "depth_bits": total_bits,
        "depth_bpp_low": total_bits / (h * w),
        "depth_bpp_full": total_bits / (args.width * args.height),
        "rdo_distortion": total_distortion,
        "rdo_distortion_type": "projection_y_satd" if rdo_ctx is not None else "depth_sse_anchor_fallback",
        "depth_mae": m["mae"],
        "depth_mse": m["mse"],
        "depth_rmse": m["rmse"],
        "depth_psnr": m["psnr"],
        "depth_max_error": m["max_error"],
        "direct_blocks": int(stats["direct_blocks"]),
        "copy_blocks": int(stats["copy_blocks"]),
        "delta_blocks": int(stats["delta_blocks"]),
        "direct_ratio": stats["direct_blocks"] / leaves,
        "copy_ratio": stats["copy_blocks"] / leaves,
        "delta_ratio": stats["delta_blocks"] / leaves,
        "zero_delta_blocks": int(stats["zero_delta_blocks"]),
        "zero_delta_ratio_in_delta": (
            stats["zero_delta_blocks"] / stats["delta_mode_count"]
            if stats["delta_mode_count"] else 0.0
        ),
    }
    for k, v in stats.items():
        if k.startswith("candidate_"):
            summary[k] = int(v)
    if adaptive is not None:
        summary.update(adaptive["mode"].snapshot("final_mode"))
        summary.update(adaptive["candidate"].snapshot("final_candidate"))
        for depth, model in enumerate(adaptive.get("split_type", [])):
            summary.update(model.snapshot(f"final_split_depth{depth}"))
    return rec, summary, store


# ============================================================
# Full-frame video prediction for output/statistics
# ============================================================

def make_backward_map_full(
    depth_y_cur: np.ndarray,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth_y_cur.shape
    return make_backward_map_for_region(depth_y_cur, 0, 0, cam_cur, cam_ref)


def downsample_map_for_chroma(
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map_x.shape
    hc, wc = h // 2, w // 2
    mx = map_x[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    my = map_y[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    mv = valid[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) >= 0.5
    return mx, my, mv


def single_reference_warp(
    ref_yuv: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rec_depth_full: np.ndarray,
    cam_ref: Dict[str, Any],
    cam_cur: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    ref_y, ref_u, ref_v = ref_yuv
    map_x, map_y, valid_y_map = make_backward_map_full(rec_depth_full, cam_cur, cam_ref)

    if args.invalid_fill == "zero":
        fill_y = np.zeros_like(ref_y)
    elif args.invalid_fill == "neutral":
        fill_y = np.full_like(ref_y, args.max_value // 2)
    else:
        fill_y = ref_y
    pred_y, valid_y = bilinear_sample_region(ref_y, map_x, map_y, valid_y_map, fill_y)

    mx_c, my_c, valid_c = downsample_map_for_chroma(map_x, map_y, valid_y)
    if args.invalid_fill == "zero":
        fill_u = np.zeros_like(ref_u)
        fill_v = np.zeros_like(ref_v)
    elif args.invalid_fill == "neutral":
        fill_u = np.full_like(ref_u, min(512, args.max_value))
        fill_v = np.full_like(ref_v, min(512, args.max_value))
    else:
        fill_u, fill_v = ref_u, ref_v
    pred_u, valid_u = bilinear_sample_region(ref_u, mx_c, my_c, valid_c, fill_u)
    pred_v, valid_v = bilinear_sample_region(ref_v, mx_c, my_c, valid_c, fill_v)

    return {
        "pred": (pred_y, pred_u, pred_v),
        "valid_y": valid_y,
        "valid_u": valid_u,
        "valid_v": valid_v,
    }


def prediction_metrics(
    cur_yuv: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_yuv: Tuple[np.ndarray, np.ndarray, np.ndarray],
    valid_y: np.ndarray,
    valid_u: np.ndarray,
    valid_v: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    cur_y, cur_u, cur_v = cur_yuv
    pred_y, pred_u, pred_v = pred_yuv
    my = compute_metrics(cur_y, pred_y, args.max_value)
    myv = compute_metrics(cur_y, pred_y, args.max_value, valid_y)
    mu = compute_metrics(cur_u, pred_u, args.max_value)
    mv = compute_metrics(cur_v, pred_v, args.max_value)
    return {
        "warp_valid_y_ratio": float(np.mean(valid_y)),
        "warp_valid_uv_ratio": float(np.mean(valid_u & valid_v)),
        "warp_y_psnr": my["psnr"],
        "warp_y_mae": my["mae"],
        "warp_y_mse": my["mse"],
        "warp_y_psnr_valid": myv["psnr"],
        "warp_y_mae_valid": myv["mae"],
        "warp_u_psnr": mu["psnr"],
        "warp_v_psnr": mv["psnr"],
    }


def make_video_prediction(
    prediction_type: str,
    cur_yuv: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rec_depth_full: np.ndarray,
    args: argparse.Namespace,
    cam_cur: Dict[str, Any],
    l0_yuv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    cam_l0: Optional[Dict[str, Any]] = None,
    l1_yuv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    cam_l1: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]]:
    if prediction_type == "I":
        return cur_yuv, {
            "warp_valid_y_ratio": 0.0,
            "warp_valid_uv_ratio": 0.0,
            "warp_y_psnr": float("inf"),
            "warp_y_mae": 0.0,
            "warp_y_mse": 0.0,
            "warp_y_psnr_valid": float("inf"),
            "warp_y_mae_valid": 0.0,
            "warp_u_psnr": float("inf"),
            "warp_v_psnr": float("inf"),
        }

    if l0_yuv is None or cam_l0 is None:
        raise RuntimeError("inter prediction requires L0")
    w0 = single_reference_warp(l0_yuv, rec_depth_full, cam_l0, cam_cur, args)

    if prediction_type == "P":
        return w0["pred"], prediction_metrics(
            cur_yuv, w0["pred"], w0["valid_y"], w0["valid_u"], w0["valid_v"], args
        )

    if l1_yuv is None or cam_l1 is None:
        raise RuntimeError("B prediction requires L1")
    w1 = single_reference_warp(l1_yuv, rec_depth_full, cam_l1, cam_cur, args)

    p0y, p0u, p0v = w0["pred"]
    p1y, p1u, p1v = w1["pred"]
    valid_y = w0["valid_y"] | w1["valid_y"]
    valid_u = w0["valid_u"] | w1["valid_u"]
    valid_v = w0["valid_v"] | w1["valid_v"]

    def blend(a0, v0, a1, v1, fallback):
        both = v0 & v1
        only0 = v0 & ~v1
        only1 = v1 & ~v0
        out = np.asarray(fallback, dtype=np.float64).copy()
        out[only0] = a0[only0]
        out[only1] = a1[only1]
        out[both] = 0.5 * (a0[both] + a1[both])
        return out

    if args.invalid_fill == "zero":
        fy = np.zeros_like(p0y)
        fu = np.zeros_like(p0u)
        fv = np.zeros_like(p0v)
    elif args.invalid_fill == "neutral":
        fy = np.full_like(p0y, args.max_value // 2)
        fu = np.full_like(p0u, min(512, args.max_value))
        fv = np.full_like(p0v, min(512, args.max_value))
    else:
        fy = 0.5 * (l0_yuv[0] + l1_yuv[0])
        fu = 0.5 * (l0_yuv[1] + l1_yuv[1])
        fv = 0.5 * (l0_yuv[2] + l1_yuv[2])

    pred = (
        blend(p0y, w0["valid_y"], p1y, w1["valid_y"], fy),
        blend(p0u, w0["valid_u"], p1u, w1["valid_u"], fu),
        blend(p0v, w0["valid_v"], p1v, w1["valid_v"], fv),
    )
    return pred, prediction_metrics(cur_yuv, pred, valid_y, valid_u, valid_v, args)


# ============================================================
# YUV I/O
# ============================================================

def frame_size_yuv420p10le(w: int, h: int) -> int:
    return w * h * 3


def count_frames(path: str, w: int, h: int) -> int:
    fs = frame_size_yuv420p10le(w, h)
    size = os.path.getsize(path)
    if size % fs:
        print(f"[WARN] trailing bytes ignored: {path}, trailing={size % fs}")
    return size // fs


def read_yuv420p10le_frame(
    fp,
    idx: int,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp.seek(idx * frame_size_yuv420p10le(w, h))
    y_raw = fp.read(w * h * 2)
    cw, ch = w // 2, h // 2
    u_raw = fp.read(cw * ch * 2)
    v_raw = fp.read(cw * ch * 2)
    if len(y_raw) != w * h * 2 or len(u_raw) != cw * ch * 2 or len(v_raw) != cw * ch * 2:
        raise EOFError(f"failed to read frame {idx}")
    y = np.frombuffer(y_raw, dtype="<u2").reshape(h, w).astype(np.float64)
    u = np.frombuffer(u_raw, dtype="<u2").reshape(ch, cw).astype(np.float64)
    v = np.frombuffer(v_raw, dtype="<u2").reshape(ch, cw).astype(np.float64)
    return y, u, v


def write_yuv420p10le_frame(fp, y: np.ndarray, u: np.ndarray, v: np.ndarray, maxv: int) -> None:
    fp.write(np.clip(np.rint(y), 0, maxv).astype("<u2").tobytes())
    fp.write(np.clip(np.rint(u), 0, maxv).astype("<u2").tobytes())
    fp.write(np.clip(np.rint(v), 0, maxv).astype("<u2").tobytes())


def write_yuv420p10le_frame_at(
    fp,
    output_idx: int,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: int,
    h: int,
    maxv: int,
) -> None:
    fp.seek(output_idx * frame_size_yuv420p10le(w, h))
    write_yuv420p10le_frame(fp, y, u, v, maxv)


def write_depth_yuv_at(
    fp,
    output_idx: int,
    depth_full: np.ndarray,
    w: int,
    h: int,
    maxv: int,
) -> None:
    uv = np.full((h // 2, w // 2), min(512, maxv), dtype=np.float64)
    write_yuv420p10le_frame_at(fp, output_idx, depth_full, uv, uv, w, h, maxv)


# ============================================================
# Coding plan / I-frame parsing
# ============================================================

def parse_frame_set(spec: str) -> Set[int]:
    out: Set[int] = set()
    if not spec.strip():
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_text, b_text = token.split("-", 1)
            a, b = int(a_text), int(b_text)
            lo, hi = min(a, b), max(a, b)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return out


def append_ra_midpoints(
    lo: int,
    hi: int,
    layer: int,
    order: List[int],
    plan_by_frame: Dict[int, Dict[str, Any]],
) -> None:
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    if mid <= lo or mid >= hi:
        return
    order.append(mid)
    plan_by_frame[mid] = {
        "reference_l0": lo,
        "reference_l1": hi,
        "prediction_type": "B",
        "temporal_layer": layer,
    }
    append_ra_midpoints(lo, mid, layer + 1, order, plan_by_frame)
    append_ra_midpoints(mid, hi, layer + 1, order, plan_by_frame)


def build_frame_coding_plan(
    start: int,
    end: int,
    coding_order: str,
    ra_gop_size: int,
    ref_offset: int,
) -> List[Dict[str, Any]]:
    if coding_order == "sequential":
        out = []
        for coding_idx, fi in enumerate(range(start, end)):
            ref = fi - ref_offset
            if ref < start:
                ref = None
            out.append(
                {
                    "frame": fi,
                    "reference_l0": ref,
                    "reference_l1": None,
                    "prediction_type": "I" if ref is None else "P",
                    "temporal_layer": 0,
                    "coding_order_idx": coding_idx,
                    "display_order_idx": fi - start,
                }
            )
        return out

    order = [start]
    plan_by_frame = {
        start: {
            "reference_l0": None,
            "reference_l1": None,
            "prediction_type": "I",
            "temporal_layer": 0,
        }
    }
    gop_start = start
    last = end - 1
    while gop_start < last:
        gop_end = min(gop_start + ra_gop_size, last)
        if gop_end not in plan_by_frame:
            order.append(gop_end)
            plan_by_frame[gop_end] = {
                "reference_l0": None,
                "reference_l1": None,
                "prediction_type": "I",
                "temporal_layer": 0,
            }
        append_ra_midpoints(gop_start, gop_end, 1, order, plan_by_frame)
        gop_start = gop_end

    if sorted(order) != list(range(start, end)):
        raise RuntimeError("RA coding plan did not emit every frame exactly once")
    out = []
    for coding_idx, fi in enumerate(order):
        rec = dict(plan_by_frame[fi])
        rec.update(
            {
                "frame": fi,
                "coding_order_idx": coding_idx,
                "display_order_idx": fi - start,
            }
        )
        out.append(rec)
    return out


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-resolution inverse-depth plane coding with projection-SATD RDO"
    )
    p.add_argument("--input-depth", required=True)
    p.add_argument("--input-video", default="")
    p.add_argument("--camera-param", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)

    p.add_argument("--coding-order", choices=["ra", "sequential"], default="ra")
    p.add_argument("--ra-gop-size", type=int, default=32)
    p.add_argument("--ref-offset", type=int, default=1)

    p.add_argument(
        "--depth-scale",
        type=int,
        default=1,
        help="process depth at width/scale x height/scale",
    )
    p.add_argument(
        "--depth-downsample",
        choices=["max", "mean", "min", "nearest"],
        default="max",
        help="depth reduction method inside each scale x scale cell",
    )
    p.add_argument(
        "--depth-upsample",
        choices=["nearest", "bilinear"],
        default="nearest",
        help="depth enlargement method used for projection RDO and output",
    )
    p.add_argument(
        "--zero-depth-frames",
        default="",
        help="comma/range POC list coded as all-zero depth with zero bits, e.g. 0,32,64-96",
    )

    p.add_argument("--block-size", type=int, default=128, help="low-resolution root block size")
    p.add_argument("--max-qt-depth", type=int, default=0)
    p.add_argument("--lambda-rd", type=float, default=0.0)
    p.add_argument("--qa", type=float, default=1e-6)
    p.add_argument("--qb", type=float, default=1e-6)
    p.add_argument("--qc", type=float, default=1e-4)
    p.add_argument("--c-only-plane", action="store_true")

    p.add_argument("--mode-bits", type=int, default=2)
    p.add_argument("--max-value", type=int, default=1023)
    p.add_argument("--depth-eps", type=float, default=1.0)

    p.add_argument("--plane-warp-candidate", dest="plane_warp_candidate", action="store_true")
    p.add_argument("--no-plane-warp-candidate", dest="plane_warp_candidate", action="store_false")
    p.set_defaults(plane_warp_candidate=True)
    p.add_argument("--plane-warp-samples", type=int, default=5)
    p.add_argument(
        "--temporal-sample-max-inv-rmse",
        type=float,
        default=0.0,
        help="reject five-point candidate if inverse-depth fit RMSE exceeds this; 0 disables",
    )

    p.add_argument("--adaptive-prob", action="store_true")
    p.add_argument("--copy-candidate-unary", action="store_true")
    p.add_argument(
        "--qt-split-adaptive",
        action="store_true",
        help="deprecated compatibility option; split-type adaptation is enabled by --adaptive-prob",
    )
    p.add_argument("--delta-residual-adaptive", action="store_true")
    p.add_argument("--delta-abs-max", type=int, default=7)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--prob-lr", type=float, default=0.05)
    p.add_argument("--prob-min", type=float, default=0.02)
    p.add_argument("--prob-max", type=float, default=0.95)
    p.add_argument("--prob-reset", choices=["frame", "sequence"], default="frame")

    p.add_argument("--invalid-fill", choices=["prev_same", "zero", "neutral"], default="prev_same")

    p.add_argument("--out-csv", default="projection_satd_depth_stats.csv")
    p.add_argument("--out-json", default="projection_satd_depth_summary.json")
    p.add_argument("--out-depth-recon-yuv", default="recon_depth_fullres.yuv")
    p.add_argument("--out-pred-yuv", default="projection_pred.yuv")
    p.add_argument("--out-block-csv", default="")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width and height")
    if args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    if args.width % args.depth_scale or args.height % args.depth_scale:
        raise ValueError("width and height must be divisible by --depth-scale")
    if args.block_size <= 0 or args.max_qt_depth < 0:
        raise ValueError("invalid block configuration")
    if min(args.qa, args.qb, args.qc) <= 0.0:
        raise ValueError("qsteps must be positive")
    if args.ra_gop_size <= 0 or args.ref_offset <= 0:
        raise ValueError("invalid coding-order configuration")
    if args.coding_order == "ra" and args.ra_gop_size & (args.ra_gop_size - 1):
        raise ValueError("--ra-gop-size must be a power of two")
    if args.copy_candidate_unary and not args.adaptive_prob:
        raise ValueError("--copy-candidate-unary requires --adaptive-prob")
    if args.delta_residual_adaptive and not args.adaptive_prob:
        raise ValueError("--delta-residual-adaptive requires --adaptive-prob")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    validate_args(args)
    zero_depth_frames = parse_frame_set(args.zero_depth_frames)

    video_path = args.input_video if args.input_video else args.input_depth
    total = min(
        count_frames(args.input_depth, args.width, args.height),
        count_frames(video_path, args.width, args.height),
    )
    if total <= 0:
        raise ValueError("no complete frames found")
    end = total if args.num_frames == 0 else min(total, args.start_frame + args.num_frames)
    if args.start_frame < 0 or args.start_frame >= end:
        raise ValueError("invalid frame range")

    camera_json = load_camera_json(args.camera_param)
    header = camera_json["header"]
    if int(header["width"]) != args.width or int(header["height"]) != args.height:
        raise ValueError("camera/input resolution mismatch")
    camera_lookup = build_camera_lookup(camera_json)

    coding_plan = build_frame_coding_plan(
        args.start_frame, end, args.coding_order, args.ra_gop_size, args.ref_offset
    )
    grid = GridCache()
    hadamard = HadamardCache()
    seq_adaptive = (
        create_adaptive_models(args)
        if args.adaptive_prob and args.prob_reset == "sequence"
        else None
    )

    frame_store: Dict[int, List[LeafRecord]] = {}
    summaries: List[Dict[str, Any]] = []

    depth_out = open(args.out_depth_recon_yuv, "wb+") if args.out_depth_recon_yuv else None
    pred_out = open(args.out_pred_yuv, "wb+") if args.out_pred_yuv else None

    block_fp = None
    block_writer = None
    if args.out_block_csv:
        block_fp = open(args.out_block_csv, "w", newline="", encoding="utf-8")
        fields = [
            "frame", "bx", "by", "block_w", "block_h", "qt_depth",
            "split_type", "mode", "candidate", "bits", "split_bits",
            "distortion", "cost", "q0", "q1", "q2",
            "actual_inv_a", "actual_inv_b", "actual_inv_c",
            "recon_inv_a", "recon_inv_b", "recon_inv_c",
        ]
        block_writer = csv.DictWriter(block_fp, fieldnames=fields)
        block_writer.writeheader()

    try:
        with open(args.input_depth, "rb") as depth_fp, open(video_path, "rb") as video_fp:
            for item in coding_plan:
                fi = int(item["frame"])
                ref_l0 = item["reference_l0"]
                ref_l1 = item["reference_l1"]
                prediction_type = str(item["prediction_type"])
                display_idx = int(item["display_order_idx"])
                coding_idx = int(item["coding_order_idx"])

                adaptive = (
                    create_adaptive_models(args)
                    if args.adaptive_prob and args.prob_reset == "frame"
                    else seq_adaptive
                )

                depth_full_gt = read_yuv420p10le_frame(
                    depth_fp, fi, args.width, args.height
                )[0]
                depth_low_gt = downsample_depth_integer(
                    depth_full_gt, args.depth_scale, args.depth_downsample
                )
                cur_video = read_yuv420p10le_frame(
                    video_fp, fi, args.width, args.height
                )
                cam_cur_full = get_camera(camera_lookup, fi)
                cam_cur_full = dict(cam_cur_full, width=args.width, height=args.height)
                cam_cur_low = scale_camera_intrinsic(cam_cur_full, args.depth_scale)

                l0_video = None
                l1_video = None
                cam_l0_full = None
                cam_l1_full = None

                if ref_l0 is not None:
                    l0_video = read_yuv420p10le_frame(
                        video_fp, ref_l0, args.width, args.height
                    )
                    cam_l0_full = dict(
                        get_camera(camera_lookup, ref_l0),
                        width=args.width,
                        height=args.height,
                    )
                if ref_l1 is not None:
                    l1_video = read_yuv420p10le_frame(
                        video_fp, ref_l1, args.width, args.height
                    )
                    cam_l1_full = dict(
                        get_camera(camera_lookup, ref_l1),
                        width=args.width,
                        height=args.height,
                    )

                if fi in zero_depth_frames:
                    rec_low, summary, cur_store = simulate_zero_depth_frame(
                        depth_low_gt, fi, args
                    )
                else:
                    plane_warp_ctx = None
                    if args.plane_warp_candidate and prediction_type in ("P", "B"):
                        l0_store = frame_store.get(ref_l0, []) if ref_l0 is not None else []
                        l1_store = frame_store.get(ref_l1, []) if ref_l1 is not None else None
                        plane_warp_ctx = PlaneWarpContext(
                            l0_store=l0_store,
                            cam_l0_low=scale_camera_intrinsic(cam_l0_full, args.depth_scale),
                            cam_cur_low=cam_cur_low,
                            frame_w_low=args.width // args.depth_scale,
                            frame_h_low=args.height // args.depth_scale,
                            l1_store=l1_store,
                            cam_l1_low=(
                                scale_camera_intrinsic(cam_l1_full, args.depth_scale)
                                if cam_l1_full is not None else None
                            ),
                        )

                    rdo_ctx = None
                    if prediction_type in ("P", "B"):
                        rdo_ctx = ProjectionRDOContext(
                            prediction_type=prediction_type,
                            cur_y=cur_video[0],
                            cam_cur_full=cam_cur_full,
                            depth_scale=args.depth_scale,
                            max_value=args.max_value,
                            invalid_fill=args.invalid_fill,
                            upsample_mode=args.depth_upsample,
                            l0_y=None if l0_video is None else l0_video[0],
                            cam_l0_full=cam_l0_full,
                            l1_y=None if l1_video is None else l1_video[0],
                            cam_l1_full=cam_l1_full,
                        )

                    rec_low, summary, cur_store = simulate_one_depth_frame(
                        depth_low_gt, fi, args, grid, hadamard, block_writer,
                        adaptive, plane_warp_ctx, rdo_ctx
                    )

                frame_store[fi] = cur_store
                rec_full = upsample_depth_integer(
                    rec_low, args.depth_scale, args.depth_upsample
                )
                rec_full = rec_full[: args.height, : args.width]

                if depth_out is not None:
                    write_depth_yuv_at(
                        depth_out, display_idx, rec_full,
                        args.width, args.height, args.max_value
                    )

                pred_video, warp_stats = make_video_prediction(
                    prediction_type,
                    cur_video,
                    rec_full,
                    args,
                    cam_cur_full,
                    l0_yuv=l0_video,
                    cam_l0=cam_l0_full,
                    l1_yuv=l1_video,
                    cam_l1=cam_l1_full,
                )
                if pred_out is not None:
                    write_yuv420p10le_frame_at(
                        pred_out, display_idx,
                        pred_video[0], pred_video[1], pred_video[2],
                        args.width, args.height, args.max_value
                    )

                summary.update(warp_stats)
                summary.update(
                    {
                        "coding_order_idx": coding_idx,
                        "display_order_idx": display_idx,
                        "prediction_type": prediction_type,
                        "reference_l0": -1 if ref_l0 is None else int(ref_l0),
                        "reference_l1": -1 if ref_l1 is None else int(ref_l1),
                        "temporal_layer": int(item["temporal_layer"]),
                        "depth_scale_factor": args.depth_scale,
                        "full_width": args.width,
                        "full_height": args.height,
                        "zero_depth_requested": int(fi in zero_depth_frames),
                    }
                )
                summaries.append(summary)

                print(
                    f"CO={coding_idx:3d} POC={fi:3d} {prediction_type} "
                    f"L0/L1={ref_l0}/{ref_l1} "
                    f"bits={summary['depth_bits']:.1f} "
                    f"RDO={summary['rdo_distortion']:.1f} "
                    f"depthPSNR={summary['depth_psnr']:.3f} "
                    f"warpY={summary['warp_y_psnr']:.3f}"
                )
    finally:
        if depth_out is not None:
            depth_out.close()
        if pred_out is not None:
            pred_out.close()
        if block_fp is not None:
            block_fp.close()

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fields = sorted(set().union(*(s.keys() for s in summaries)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    average: Dict[str, float] = {}
    for key in sorted(set().union(*(s.keys() for s in summaries))):
        vals = []
        for s in summaries:
            try:
                v = float(s[key])
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        if vals:
            average[key] = float(np.mean(vals))

    total_bits = float(sum(s["depth_bits"] for s in summaries))
    overall = {
        **vars(args),
        "zero_depth_frame_set": sorted(zero_depth_frames),
        "num_processed_frames": len(summaries),
        "coding_plan": coding_plan,
        "total_depth_bits": total_bits,
        "overall_depth_bpp_full": total_bits / (
            args.width * args.height * len(summaries)
        ),
        "rdo_distortion": "projection_domain_luma_satd_for_inter",
        "temporal_candidate": "five_target_points_nearest_visible_projected_leaf_refit",
        "depth_processing_resolution": [
            args.width // args.depth_scale,
            args.height // args.depth_scale,
        ],
        "depth_output_resolution": [args.width, args.height],
        "depth_downsampling": args.depth_downsample,
        "depth_upsampling": args.depth_upsample,
        "out_of_picture_sampling": "reflect",
        "all_projection_invalid_fallback": "minimum_estimated_bits",
        "split_signaling": "depth_context_adaptive_leaf_bh_bv_qt",
        "average": average,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Frame CSV       : {args.out_csv}")
    print(f"Summary         : {args.out_json}")
    print(f"Recon depth YUV : {args.out_depth_recon_yuv}")
    print(f"Projection pred : {args.out_pred_yuv}")
    if args.out_block_csv:
        print(f"Block CSV       : {args.out_block_csv}")
    print(f"Overall bpp     : {overall['overall_depth_bpp_full']:.8f}")


if __name__ == "__main__":
    main()

