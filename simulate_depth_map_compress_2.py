#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse-depth plane compression simulation with hierarchical RA coding,
VGGT/canonical camparam_v2 JSONL, analytic temporal plane transformation, and
bidirectional video prediction.

Important behavior:
  1) POC 0 and each RA endpoint (for example POC 32) are coded as anchors.
  2) From POC 16 onward, L0/L1 reconstructed leaf planes are converted to
     camera-space 3D planes, transformed analytically into the current camera,
     and rendered directly over the current block. No full-frame forward
     splatting or z-buffer hole filling is used for depth-plane candidates.
  3) The following candidates are independently tested by copy/delta RDO:
       plane_warp_avg, plane_warp_l0, plane_warp_l1,
       temporal_avg, temporal_l0, temporal_l1.
  4) Video prediction separately uses the reconstructed current depth to
     backward-project current pixels into L0/L1 reference pictures.
  5) Projection follows the VGGT pixel-coordinate convention directly, and
     fixed-point depth is decoded as:
       depth_linear = depth_y * depth_scale / depth_scale_precision
     JSONL tvec values are already in the same real-depth unit.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Data classes
# ============================================================

@dataclass
class Plane:
    # Inverse-depth plane in the stored depth-sample domain:
    #   invY(x,y) = a * (x - cx) + b * (y - cy) + c
    # Real camera depth is reconstructed separately using depth_scale_real.
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
    sse: float
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
    split: str = "leaf"  # leaf, qt, bh, bv
    children: List["CSNode"] = field(default_factory=list)

    best: Optional[ModeResult] = None
    actual: Optional[Plane] = None
    avail_modes: List[str] = field(default_factory=list)
    avail_cands: List[str] = field(default_factory=list)

    bits: float = 0.0
    sse: float = 0.0
    cost: float = 0.0
    split_bits: float = 0.0
    qt_flag_present: bool = False

    def is_leaf(self):
        return self.split == "leaf"


@dataclass
class PlaneWarpContext:
    # Reconstructed reference-plane stores and cameras used for analytic
    # per-block 3D plane transformation. L1 is absent for sequential P frames.
    l0_store: List[LeafRecord]
    cam_l0: Dict[str, Any]
    cam_cur: Dict[str, Any]
    frame_w: int
    frame_h: int
    l1_store: Optional[List[LeafRecord]] = None
    cam_l1: Optional[Dict[str, Any]] = None
    source_type: str = ""


# ============================================================
# Adaptive probability model
# ============================================================

class AdaptiveProbTable:
    def __init__(
        self,
        symbols,
        init_probs=None,
        update_rate=0.05,
        p_min=0.02,
        p_max=0.95,
        name="",
    ):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.name = name

        if self.n <= 0:
            raise ValueError("AdaptiveProbTable needs symbols")
        if self.p_min * self.n > 1.0:
            raise ValueError(f"{name}: p_min too large")
        if self.p_max * self.n < 1.0:
            raise ValueError(f"{name}: p_max too small")

        if init_probs is None:
            self.probs = {s: 1.0 / self.n for s in self.symbols}
        else:
            total = sum(float(init_probs.get(s, 0.0)) for s in self.symbols)
            if total <= 0:
                raise ValueError("init_probs sum must be positive")
            self.probs = {
                s: float(init_probs.get(s, 0.0)) / total for s in self.symbols
            }

        self._project()

    def bits(self, symbol, available_symbols=None):
        if symbol not in self.probs:
            raise KeyError(f"unknown symbol {symbol} in {self.name}")

        if available_symbols is None:
            return -math.log2(max(self.probs[symbol], 1e-12))

        av = [s for s in available_symbols if s in self.probs]

        if symbol not in av:
            raise KeyError(f"{symbol} not available in {self.name}")

        if len(av) <= 1:
            return 0.0

        norm = sum(self.probs[s] for s in av)
        p = self.probs[symbol] / norm if norm > 0 else 1.0 / len(av)
        return -math.log2(max(p, 1e-12))

    def update(self, selected):
        if selected not in self.probs:
            raise KeyError(f"unknown selected {selected} in {self.name}")

        psel = min(self.p_max, 1.0 - (self.n - 1) * self.p_min)
        others = [s for s in self.symbols if s != selected]
        target = {selected: psel}

        for s in others:
            target[s] = (1.0 - psel) / len(others) if others else 0.0

        lr = self.update_rate

        for s in self.symbols:
            self.probs[s] = (1.0 - lr) * self.probs[s] + lr * target[s]

        self._project()

    def _project(self):
        for s in self.symbols:
            self.probs[s] = min(max(self.probs[s], self.p_min), self.p_max)

        for _ in range(64):
            diff = 1.0 - sum(self.probs.values())

            if abs(diff) < 1e-12:
                break

            if diff > 0:
                adj = [
                    s for s in self.symbols
                    if self.probs[s] < self.p_max - 1e-12
                ]
            else:
                adj = [
                    s for s in self.symbols
                    if self.probs[s] > self.p_min + 1e-12
                ]

            if not adj:
                break

            add = diff / len(adj)

            for s in adj:
                self.probs[s] = min(max(self.probs[s] + add, self.p_min), self.p_max)

    def snapshot(self, prefix):
        return {f"{prefix}_{s}_prob": self.probs[s] for s in self.symbols}


class BinaryAdaptiveProb:
    def __init__(
        self,
        init_p1=0.5,
        update_rate=0.05,
        p_min=0.02,
        p_max=0.98,
        name="",
    ):
        self.p1 = float(init_p1)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.name = name
        self._clip()

    def _clip(self):
        self.p1 = min(max(self.p1, self.p_min), self.p_max)

    def bits(self, b):
        if b not in (0, 1):
            raise ValueError("bin must be 0/1")

        p = self.p1 if b else 1.0 - self.p1
        return -math.log2(max(p, 1e-12))

    def update(self, b):
        if b not in (0, 1):
            raise ValueError("bin must be 0/1")

        target = self.p_max if b else self.p_min
        self.p1 = (1.0 - self.update_rate) * self.p1 + self.update_rate * target
        self._clip()


def unary_candidate_bits(idx, n, ctx, truncated=True):
    if n <= 1:
        return 0.0

    if idx < 0 or idx >= n or n > len(ctx):
        raise ValueError("bad unary candidate")

    bits = sum(ctx[i].bits(0) for i in range(idx))

    if not (truncated and idx == n - 1):
        bits += ctx[idx].bits(1)

    return bits


def unary_candidate_update(idx, n, ctx, truncated=True):
    if n <= 1:
        return

    if idx < 0 or idx >= n or n > len(ctx):
        raise ValueError("bad unary candidate")

    for i in range(idx):
        ctx[i].update(0)

    if not (truncated and idx == n - 1):
        ctx[idx].update(1)


def qt_split_flag_bits(adaptive, depth, flag):
    if adaptive is not None and "qt_split" in adaptive and depth < len(adaptive["qt_split"]):
        return adaptive["qt_split"][depth].bits(flag)

    return 1.0


def qt_split_flag_update(adaptive, node):
    if adaptive is None or "qt_split" not in adaptive:
        return

    if not node.qt_flag_present:
        return

    if node.depth >= len(adaptive["qt_split"]):
        return

    flag = 1 if node.split == "qt" else 0
    adaptive["qt_split"][node.depth].update(flag)


def ceil_log2(x):
    return 0 if x <= 1 else int(math.ceil(math.log2(x)))


def exp_golomb_len_unsigned(u):
    if u < 0:
        raise ValueError("ue input negative")

    return 2 * int(math.floor(math.log2(u + 1))) + 1


def signed_to_code_num(v):
    if v == 0:
        return 0

    return 2 * v - 1 if v > 0 else -2 * v


def exp_golomb_len_signed(v):
    return exp_golomb_len_unsigned(signed_to_code_num(v))


def quantize(x, q):
    return int(np.rint(x / q))


def dequantize(v, q):
    return float(v) * q


def adaptive_signed_residual_bits(q, model, abs_max):
    a = abs(q)

    if a <= abs_max:
        bits = model.bits(a)
    else:
        bits = model.bits("esc")
        bits += exp_golomb_len_unsigned(a - (abs_max + 1))

    if a > 0:
        bits += 1.0

    return bits


def adaptive_signed_residual_update(q, model, abs_max):
    model.update(abs(q) if abs(q) <= abs_max else "esc")


def create_adaptive_models(args):
    # Analytic camera-plane candidates and same-position temporal candidates
    # are kept separate for L0, L1, and their bidirectional average.
    cand_symbols = [
        "plane_warp_avg",
        "plane_warp_l0",
        "plane_warp_l1",
        "temporal_avg",
        "temporal_l0",
        "temporal_l1",
        "left",
        "top",
        "top_left",
        "top_right",
        "avg_left_top",
    ]

    models = {
        "mode": AdaptiveProbTable(
            ["direct", "copy", "delta"],
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="mode",
        ),
        "candidate": AdaptiveProbTable(
            cand_symbols,
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="candidate",
        ),
        "delta_abs_max": args.delta_abs_max,
    }

    if args.copy_candidate_unary:
        models["copy_candidate_unary"] = [
            BinaryAdaptiveProb(
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"copy_ctx{i}",
            )
            for i in range(args.max_candidates)
        ]

    if args.qt_split_adaptive:
        models["qt_split"] = [
            BinaryAdaptiveProb(
                init_p1=0.5,
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"qt_split_depth{i}",
            )
            for i in range(args.max_qt_depth)
        ]

    if args.delta_residual_adaptive:
        syms = list(range(args.delta_abs_max + 1)) + ["esc"]

        for k in "abc":
            models[f"delta_res_abs_{k}"] = AdaptiveProbTable(
                syms,
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"delta_res_abs_{k}",
            )

    return models


# ============================================================
# Inverse-depth plane fitting/rendering
# ============================================================

class GridCache:
    def __init__(self):
        self.cache = {}

    def get(self, w, h):
        key = (w, h)

        if key not in self.cache:
            xs = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
            ys = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
            xx, yy = np.meshgrid(xs, ys)

            A = np.stack(
                [xx.reshape(-1), yy.reshape(-1), np.ones(w * h)],
                axis=1,
            )

            self.cache[key] = (xx, yy, np.linalg.pinv(A))

        return self.cache[key]


def fit_inv_depth_plane_from_depth_block(block_y, pinv, cx, cy, args):
    y = np.clip(block_y.astype(np.float64), args.depth_eps, args.max_value)
    inv = 1.0 / y
    a, b, c = (pinv @ inv.reshape(-1)).tolist()
    return Plane(a, b, c, cx, cy)


def fit_inv_depth_plane_from_depth_block_masked(
    block_y,
    valid_mask,
    xx,
    yy,
    cx,
    cy,
    args,
):
    """Fit an inverse-depth plane using only projected valid samples."""
    block_y = np.asarray(block_y, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    valid &= np.isfinite(block_y)
    valid &= block_y >= args.depth_eps
    valid &= block_y <= args.max_value

    if np.count_nonzero(valid) < 3:
        return None

    y = np.clip(block_y[valid], args.depth_eps, args.max_value)
    inv = 1.0 / y
    A = np.stack(
        [
            np.asarray(xx, dtype=np.float64)[valid],
            np.asarray(yy, dtype=np.float64)[valid],
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        ],
        axis=1,
    )

    try:
        coeff, _, rank, _ = np.linalg.lstsq(A, inv, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3 or not np.isfinite(coeff).all():
        return None

    return Plane(float(coeff[0]), float(coeff[1]), float(coeff[2]), cx, cy)


def plane_to_center(p, cx, cy):
    return Plane(
        p.a,
        p.b,
        p.c + p.a * (cx - p.cx) + p.b * (cy - p.cy),
        cx,
        cy,
    )


def eval_inv_plane_value(p, gx, gy):
    return p.a * (gx - p.cx) + p.b * (gy - p.cy) + p.c


def inv_plane_to_depth_value(p, gx, gy, args):
    inv = eval_inv_plane_value(p, gx, gy)
    inv_min = 1.0 / max(float(args.max_value), 1.0)
    inv_max = 1.0 / max(float(args.depth_eps), 1e-12)
    inv = np.clip(inv, inv_min, inv_max)
    return np.clip(1.0 / inv, 0.0, args.max_value)


def render_inv_depth_plane(p, xx, yy, args):
    inv = p.a * xx + p.b * yy + p.c
    inv_min = 1.0 / max(float(args.max_value), 1.0)
    inv_max = 1.0 / max(float(args.depth_eps), 1e-12)
    inv = np.clip(inv, inv_min, inv_max)
    y = 1.0 / inv
    return np.clip(np.rint(y), 0, args.max_value).astype(np.float64)


def block_sse(orig, recon):
    d = orig.astype(np.float64) - recon.astype(np.float64)
    return float(np.sum(d * d))


# ============================================================
# Spatial candidates / previous-frame leaf lookup
# ============================================================

def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def best_left(store, x, y, w, h):
    best = None
    bo = 0

    for r in store:
        if r.x + r.w == x:
            o = overlap(r.y, r.y + r.h, y, y + h)

            if o > bo:
                best = r
                bo = o

    return best


def best_top(store, x, y, w, h):
    best = None
    bo = 0

    for r in store:
        if r.y + r.h == y:
            o = overlap(r.x, r.x + r.w, x, x + w)

            if o > bo:
                best = r
                bo = o

    return best


def top_left(store, x, y):
    for r in store:
        if r.x + r.w == x and r.y + r.h == y:
            return r

    return None


def top_right(store, x, y, w):
    for r in store:
        if r.x == x + w and r.y + r.h == y:
            return r

    return None


def leaf_covering_point(store, cx, cy):
    """Find a previous-frame leaf only for the camera plane-warp candidate."""
    if not store:
        return None

    for r in store:
        if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
            return r

    return None


# ============================================================
# Camera JSONL v2 / pose reconstruction
# ============================================================

def rodrigues_to_matrix(rvec):
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))

    if theta < 1e-12:
        x, y, z = r
        k = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + k

    axis = r / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * k
        + (1.0 - math.cos(theta)) * (k @ k)
    )


def rt_to_4x4(rvec, tvec):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rodrigues_to_matrix(rvec)
    t[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return t


def intrinsic_vec_to_matrix(v):
    fx, fy, cx, cy = [float(x) for x in v]

    if not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError(f"non-finite intrinsic: {v}")

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"fx/fy must be positive: fx={fx}, fy={fy}")

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_camera_json(path):
    """Read camparam_v2_vggt_or_canonical JSONL."""
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
                raise RuntimeError(
                    f"Failed to parse camera JSONL at line {line_no}: {path}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Camera JSONL line {line_no} must be an object, got {type(obj)}"
                )

            if obj.get("type") == "header":
                if header is not None:
                    raise ValueError(f"Multiple camera headers found: {path}")
                header = obj
            else:
                frames.append(obj)

    if header is None:
        raise ValueError(f"Camera JSONL header not found: {path}")

    required_header = [
        "width",
        "height",
        "depth_scale",
        "depth_scale_precision",
        "intrinsic",
        "pose_mode",
    ]

    for k in required_header:
        if k not in header:
            raise KeyError(f"Camera JSONL header is missing '{k}': {path}")

    if not frames:
        raise ValueError(f"Camera JSONL has no frame records: {path}")

    required_frame = ["poc", "rvec", "tvec", "intrinsic_delta"]

    for i, rec in enumerate(frames):
        for k in required_frame:
            if k not in rec:
                raise KeyError(f"Camera frame record {i} is missing '{k}': {path}")

    return {"header": header, "frames": frames}


def get_depth_scale_real_from_header(header):
    """Decode the fixed-point depth scale stored in the JSONL header."""
    scale_int = float(header["depth_scale"])
    precision = float(header["depth_scale_precision"])

    if precision <= 0.0:
        raise ValueError("depth_scale_precision must be positive")

    # Critical conversion:
    #   depth_real = depth_y * (depth_scale / depth_scale_precision)
    scale_real = scale_int / precision

    if not np.isfinite(scale_real) or scale_real <= 0.0:
        raise ValueError(
            f"invalid real depth scale: {scale_int} / {precision} = {scale_real}"
        )

    stored = header.get("depth_scale_real")

    if stored is not None:
        stored = float(stored)
        tol = max(1e-12, abs(scale_real) * 1e-7)

        if not math.isclose(stored, scale_real, rel_tol=1e-7, abs_tol=tol):
            print(
                "[WARN] header depth_scale_real differs from "
                "depth_scale/depth_scale_precision; using integer ratio: "
                f"stored={stored}, derived={scale_real}"
            )

    return scale_real


def build_camera_lookup(camera_json):
    header = camera_json["header"]
    frame_records = sorted(camera_json["frames"], key=lambda r: int(r["poc"]))

    pocs = [int(r["poc"]) for r in frame_records]

    if len(set(pocs)) != len(pocs):
        raise ValueError("duplicate POC in camera JSONL")

    pose_mode = str(header["pose_mode"])

    if pose_mode == "current_to_previous":
        expected = list(range(len(frame_records)))

        if pocs != expected:
            raise ValueError(
                "current_to_previous camera JSONL requires consecutive local POCs "
                f"0..N-1, got first values={pocs[:8]}"
            )

    intr0 = header["intrinsic"]
    base_intr = np.array(
        [
            float(intr0["fx"]),
            float(intr0["fy"]),
            float(intr0["cx"]),
            float(intr0["cy"]),
        ],
        dtype=np.float64,
    )
    z_sign = float(intr0.get("z_sign", 1.0))

    if z_sign == 0.0 or not np.isfinite(z_sign):
        raise ValueError(f"invalid z_sign: {z_sign}")

    z_sign = 1.0 if z_sign > 0.0 else -1.0
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )
    depth_scale_real = get_depth_scale_real_from_header(header)

    by_poc = {}
    by_frame_idx = {}
    ordered = []

    cur_intr = base_intr.copy()
    prev_w2c = np.eye(4, dtype=np.float64)

    for order, rec in enumerate(frame_records):
        poc = int(rec["poc"])
        frame_idx = int(rec.get("frame_idx", poc))

        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64)

        if delta.shape != (4,):
            raise ValueError(
                f"POC {poc}: intrinsic_delta must have 4 values, got {delta.shape}"
            )

        if fixed_intrinsic:
            cur_intr = base_intr.copy()
        else:
            # Header intrinsic is frame-0 K. Frame 0 normally carries zero delta;
            # each later record carries the delta from the previous frame.
            cur_intr = cur_intr + delta

        k = intrinsic_vec_to_matrix(cur_intr)
        t_rec = rt_to_4x4(rec["rvec"], rec["tvec"])

        if pose_mode == "current_to_previous":
            if order == 0:
                w2c = np.eye(4, dtype=np.float64)
            else:
                # JSONL: X_prev = T_prev_from_cur * X_cur
                # Hence: W2C_cur = inv(T_prev_from_cur) * W2C_prev
                try:
                    w2c = np.linalg.inv(t_rec) @ prev_w2c
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f"POC {poc}: singular current_to_previous pose") from exc

        elif pose_mode in ("gop_local", "absolute"):
            # gop_local: X_i = T_i * X_0, and X_0 is the local world frame.
            # absolute : T_i is already camera_from_world.
            w2c = t_rec

        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")

        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"POC {poc}: singular W2C") from exc

        cam = {
            "poc": poc,
            "frame_idx": frame_idx,
            "K": k,
            "W2C": w2c,
            "C2W": c2w,
            "z_sign": z_sign,
            "depth_scale_real": depth_scale_real,
            "pose_mode": pose_mode,
        }

        by_poc[poc] = cam
        by_frame_idx.setdefault(frame_idx, cam)
        ordered.append(cam)
        prev_w2c = w2c

    declared_count = header.get("frame_count")

    if declared_count is not None and int(declared_count) != len(ordered):
        raise ValueError(
            f"camera frame_count mismatch: header={declared_count}, records={len(ordered)}"
        )

    return {
        "header": header,
        "by_poc": by_poc,
        "by_frame_idx": by_frame_idx,
        "ordered": ordered,
    }


def get_camera(lookup, frame_idx):
    # Generated depth YUVs are normally RAP-local; local POC is primary.
    if frame_idx in lookup["by_poc"]:
        return lookup["by_poc"][frame_idx]

    # Fallback supports callers indexing with original/global frame_idx.
    if frame_idx in lookup["by_frame_idx"]:
        return lookup["by_frame_idx"][frame_idx]

    raise KeyError(f"camera for frame/POC {frame_idx} not found")


def camera_has_required_mats(cam):
    try:
        k = np.asarray(cam["K"], dtype=np.float64)
        w2c = np.asarray(cam["W2C"], dtype=np.float64)
        c2w = np.asarray(cam["C2W"], dtype=np.float64)
        scale = float(cam["depth_scale_real"])
        z_sign = float(cam["z_sign"])

        return (
            k.shape == (3, 3)
            and w2c.shape == (4, 4)
            and c2w.shape == (4, 4)
            and np.isfinite(k).all()
            and np.isfinite(w2c).all()
            and np.isfinite(c2w).all()
            and np.isfinite(scale)
            and scale > 0.0
            and z_sign in (-1.0, 1.0)
        )
    except Exception:
        return False


def get_depth_scale_real(cam):
    scale = float(cam["depth_scale_real"])

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid camera depth scale: {scale}")

    return scale


# ============================================================
# Camera geometry: K-based pinhole model
# ============================================================

def pixel_rays_camera(u, v, cam):
    """Return rays with signed camera-Z equal to z_sign.

    Thus X_cam = ray * depth_real, where depth_real is positive.
    """
    k = np.asarray(cam["K"], dtype=np.float64)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    z_sign = float(cam["z_sign"])

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    rx = (u - cx) / fx
    ry = (v - cy) / fy
    rz = np.full_like(rx, z_sign, dtype=np.float64)

    return np.stack([rx, ry, rz], axis=-1)


def project_camera_points(points_cam, cam):
    """Project camera-space points using K and the configured z_sign."""
    p = np.asarray(points_cam, dtype=np.float64)
    k = np.asarray(cam["K"], dtype=np.float64)
    z_sign = float(cam["z_sign"])

    depth = z_sign * p[..., 2]
    front = np.isfinite(depth) & (depth > 1e-12)
    safe_depth = np.where(front, depth, 1.0)

    u = k[0, 0] * (p[..., 0] / safe_depth) + k[0, 2]
    v = k[1, 1] * (p[..., 1] / safe_depth) + k[1, 2]

    return u, v, depth, front


def fit_3d_plane(points):
    if points.shape[0] < 3:
        return None

    c = np.mean(points, axis=0)
    q = points - c

    try:
        _, s, vh = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if len(s) < 2 or s[1] < 1e-9:
        return None

    n = vh[-1]
    norm = np.linalg.norm(n)

    if norm < 1e-12:
        return None

    n = n / norm
    d = -float(np.dot(n, c))

    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def transform_plane_src_to_tgt(plane_src, cam_src, cam_tgt):
    c2w_src = np.asarray(cam_src["C2W"], dtype=np.float64)
    w2c_tgt = np.asarray(cam_tgt["W2C"], dtype=np.float64)
    m = w2c_tgt @ c2w_src

    try:
        plane_tgt = np.linalg.inv(m).T @ plane_src
    except np.linalg.LinAlgError:
        return None

    n = plane_tgt[:3]
    norm = np.linalg.norm(n)

    if norm < 1e-12:
        return None

    return plane_tgt / norm


def image_inv_plane_to_3d_plane(leaf, cam, frame_w, frame_h, args):
    del frame_w, frame_h

    depth_scale_real = get_depth_scale_real(cam)
    ns = max(2, int(args.plane_warp_samples))

    xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)
    ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)

    depth_y = inv_plane_to_depth_value(leaf.plane, uu, vv, args)
    depth_real = depth_y * depth_scale_real

    rays = pixel_rays_camera(uu, vv, cam)
    pts = rays.reshape(-1, 3) * depth_real.reshape(-1, 1)

    valid = np.isfinite(pts).all(axis=1) & (depth_real.reshape(-1) > 0)
    pts = pts[valid]

    if pts.shape[0] < 3:
        return None

    return fit_3d_plane(pts)


def render_3d_plane_to_depth_block(
    plane_cam,
    cam_cur,
    x,
    y,
    w,
    h,
    frame_w,
    frame_h,
    args,
):
    del frame_w, frame_h

    depth_scale_real = get_depth_scale_real(cam_cur)

    gx = np.arange(x, x + w, dtype=np.float64)
    gy = np.arange(y, y + h, dtype=np.float64)
    uu, vv = np.meshgrid(gx, gy)

    rays = pixel_rays_camera(uu, vv, cam_cur)

    n = plane_cam[:3]
    d = plane_cam[3]
    denom = np.sum(rays * n.reshape(1, 1, 3), axis=-1)

    valid = np.abs(denom) > 1e-12
    depth_real = np.full((h, w), np.nan, dtype=np.float64)
    depth_real[valid] = -d / denom[valid]

    valid = valid & np.isfinite(depth_real) & (depth_real > 0)
    valid_ratio = float(np.mean(valid))

    if valid_ratio < args.plane_warp_min_valid_ratio:
        return None

    # Convert real camera depth back to stored 10-bit depth samples.
    depth_y = depth_real / depth_scale_real
    depth_y = np.clip(depth_y, args.depth_eps, args.max_value)

    if not np.all(valid):
        med = np.median(depth_y[valid]) if np.any(valid) else args.max_value
        depth_y[~valid] = med

    return depth_y


def make_analytic_plane_warp_candidate(
    ref_store,
    cam_ref,
    cam_cur,
    x,
    y,
    w,
    h,
    cx,
    cy,
    frame_w,
    frame_h,
    args,
    grid,
):
    """Transform one reconstructed reference leaf plane into current view.

    This intentionally follows the successful LD path:
      reference inverse-depth leaf plane
        -> sampled camera-space 3D points
        -> fitted homogeneous 3D plane
        -> analytic source-to-current plane transform
        -> dense current-block plane rendering
        -> current inverse-depth plane fitting

    It does not forward-splat a full reconstructed depth frame.
    """
    if not ref_store or cam_ref is None or cam_cur is None:
        return None

    leaf = leaf_covering_point(ref_store, cx, cy)
    if leaf is None:
        return None

    plane3d_ref = image_inv_plane_to_3d_plane(
        leaf,
        cam_ref,
        frame_w,
        frame_h,
        args,
    )
    if plane3d_ref is None:
        return None

    plane3d_cur = transform_plane_src_to_tgt(
        plane3d_ref,
        cam_ref,
        cam_cur,
    )
    if plane3d_cur is None:
        return None

    depth_block = render_3d_plane_to_depth_block(
        plane3d_cur,
        cam_cur,
        x,
        y,
        w,
        h,
        frame_w,
        frame_h,
        args,
    )
    if depth_block is None:
        return None

    _, _, pinv = grid.get(w, h)
    plane = fit_inv_depth_plane_from_depth_block(
        depth_block,
        pinv,
        cx,
        cy,
        args,
    )
    return plane, depth_block


def make_plane_warp_candidates(ctx, x, y, w, h, cx, cy, args, grid):
    if ctx is None:
        return []

    l0 = make_analytic_plane_warp_candidate(
        ctx.l0_store,
        ctx.cam_l0,
        ctx.cam_cur,
        x,
        y,
        w,
        h,
        cx,
        cy,
        ctx.frame_w,
        ctx.frame_h,
        args,
        grid,
    )

    l1 = None
    if ctx.l1_store is not None and ctx.cam_l1 is not None:
        l1 = make_analytic_plane_warp_candidate(
            ctx.l1_store,
            ctx.cam_l1,
            ctx.cam_cur,
            x,
            y,
            w,
            h,
            cx,
            cy,
            ctx.frame_w,
            ctx.frame_h,
            args,
            grid,
        )

    out = []

    # Put the bidirectional average first because max_candidates may truncate
    # the tail of the list. Both input blocks are already rendered in the
    # current camera and share the current frame's fixed-point depth scale.
    if l0 is not None and l1 is not None:
        avg_depth = 0.5 * (l0[1] + l1[1])
        _, _, pinv = grid.get(w, h)
        avg_plane = fit_inv_depth_plane_from_depth_block(
            avg_depth,
            pinv,
            cx,
            cy,
            args,
        )
        out.append(("plane_warp_avg", avg_plane))

    if l0 is not None:
        out.append(("plane_warp_l0", l0[0]))

    if l1 is not None:
        out.append(("plane_warp_l1", l1[0]))

    return out


def make_same_position_temporal_candidates(ctx, cx, cy):
    if ctx is None:
        return []

    p0 = None
    p1 = None

    r0 = leaf_covering_point(ctx.l0_store, cx, cy)
    if r0 is not None:
        p0 = plane_to_center(r0.plane, cx, cy)

    if ctx.l1_store is not None:
        r1 = leaf_covering_point(ctx.l1_store, cx, cy)
        if r1 is not None:
            p1 = plane_to_center(r1.plane, cx, cy)

    out = []

    if p0 is not None and p1 is not None:
        out.append(
            (
                "temporal_avg",
                Plane(
                    0.5 * (p0.a + p1.a),
                    0.5 * (p0.b + p1.b),
                    0.5 * (p0.c + p1.c),
                    cx,
                    cy,
                ),
            )
        )

    if p0 is not None:
        out.append(("temporal_l0", p0))

    if p1 is not None:
        out.append(("temporal_l1", p1))

    return out


def make_candidates(
    store,
    temporal_store,
    x,
    y,
    w,
    h,
    cx,
    cy,
    max_cands,
    plane_warp_ctx,
    args,
    grid,
):
    del temporal_store  # references are carried explicitly in plane_warp_ctx

    cand = []
    conv = {}

    if args.plane_warp_candidate and plane_warp_ctx is not None:
        for name, p in make_plane_warp_candidates(
            plane_warp_ctx,
            x,
            y,
            w,
            h,
            cx,
            cy,
            args,
            grid,
        ):
            conv[name] = p
            cand.append((name, p))

    if args.same_position_temporal_candidate and plane_warp_ctx is not None:
        for name, p in make_same_position_temporal_candidates(
            plane_warp_ctx,
            cx,
            cy,
        ):
            conv[name] = p
            cand.append((name, p))

    items = [
        ("left", best_left(store, x, y, w, h)),
        ("top", best_top(store, x, y, w, h)),
        ("top_left", top_left(store, x, y)),
        ("top_right", top_right(store, x, y, w)),
    ]

    for name, r in items:
        if r is not None:
            p = plane_to_center(r.plane, cx, cy)
            conv[name] = p
            cand.append((name, p))

    if "left" in conv and "top" in conv:
        l = conv["left"]
        t = conv["top"]
        cand.append(
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

    return cand[:max_cands]


# ============================================================
# Mode evaluation
# ============================================================

def eval_direct(block, actual, xx, yy, args, adaptive, avail_modes):
    qa = quantize(actual.a, args.qa)
    qb = quantize(actual.b, args.qb)
    qc = quantize(actual.c, args.qc)

    p = Plane(
        dequantize(qa, args.qa),
        dequantize(qb, args.qb),
        dequantize(qc, args.qc),
        actual.cx,
        actual.cy,
    )

    recon = render_inv_depth_plane(p, xx, yy, args)
    sse = block_sse(block, recon)

    if adaptive is not None:
        bits = adaptive["mode"].bits("direct", avail_modes)
    else:
        bits = float(args.mode_bits)

    bits += exp_golomb_len_signed(qa)
    bits += exp_golomb_len_signed(qb)
    bits += exp_golomb_len_signed(qc)

    return ModeResult(
        "direct",
        "none",
        p,
        recon,
        bits,
        sse,
        sse + args.lambda_rd * bits,
        (qa, qb, qc),
    )


def eval_copy(block, cands, xx, yy, args, adaptive, avail_modes, avail_cands):
    out = []

    for i, (name, p) in enumerate(cands):
        recon = render_inv_depth_plane(p, xx, yy, args)
        sse = block_sse(block, recon)

        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(cands)))
        else:
            bits = adaptive["mode"].bits("copy", avail_modes)

            if "copy_candidate_unary" in adaptive:
                bits += unary_candidate_bits(
                    i,
                    len(cands),
                    adaptive["copy_candidate_unary"],
                )
            else:
                bits += adaptive["candidate"].bits(name, avail_cands)

        out.append(
            ModeResult(
                "copy",
                name,
                p,
                recon,
                bits,
                sse,
                sse + args.lambda_rd * bits,
                (),
            )
        )

    return out


def eval_delta(block, actual, cands, xx, yy, args, adaptive, avail_modes, avail_cands):
    out = []

    for name, pred in cands:
        qda = quantize(actual.a - pred.a, args.qa)
        qdb = quantize(actual.b - pred.b, args.qb)
        qdc = quantize(actual.c - pred.c, args.qc)

        p = Plane(
            pred.a + dequantize(qda, args.qa),
            pred.b + dequantize(qdb, args.qb),
            pred.c + dequantize(qdc, args.qc),
            actual.cx,
            actual.cy,
        )

        recon = render_inv_depth_plane(p, xx, yy, args)
        sse = block_sse(block, recon)

        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(cands)))
        else:
            bits = adaptive["mode"].bits("delta", avail_modes)
            bits += adaptive["candidate"].bits(name, avail_cands)

        if adaptive is not None and "delta_res_abs_a" in adaptive:
            bits += adaptive_signed_residual_bits(
                qda,
                adaptive["delta_res_abs_a"],
                adaptive["delta_abs_max"],
            )
            bits += adaptive_signed_residual_bits(
                qdb,
                adaptive["delta_res_abs_b"],
                adaptive["delta_abs_max"],
            )
            bits += adaptive_signed_residual_bits(
                qdc,
                adaptive["delta_res_abs_c"],
                adaptive["delta_abs_max"],
            )
        else:
            bits += exp_golomb_len_signed(qda)
            bits += exp_golomb_len_signed(qdb)
            bits += exp_golomb_len_signed(qdc)

        out.append(
            ModeResult(
                "delta",
                name,
                p,
                recon,
                bits,
                sse,
                sse + args.lambda_rd * bits,
                (qda, qdb, qdc),
            )
        )

    return out


def eval_leaf(
    padded,
    x,
    y,
    w,
    h,
    depth,
    parent,
    args,
    grid,
    store,
    prev_store,
    adaptive,
    plane_warp_ctx,
):
    block = padded[y : y + h, x : x + w]

    cx = x + (w - 1) / 2.0
    cy = y + (h - 1) / 2.0

    xx, yy, pinv = grid.get(w, h)
    actual = fit_inv_depth_plane_from_depth_block(block, pinv, cx, cy, args)

    cands = make_candidates(
        store=store,
        temporal_store=prev_store,
        x=x,
        y=y,
        w=w,
        h=h,
        cx=cx,
        cy=cy,
        max_cands=args.max_candidates,
        plane_warp_ctx=plane_warp_ctx,
        args=args,
        grid=grid,
    )

    avail_cands = [n for n, _ in cands]
    avail_modes = ["direct", "copy", "delta"] if cands else ["direct"]

    modes = [eval_direct(block, actual, xx, yy, args, adaptive, avail_modes)]

    if cands:
        modes += eval_copy(block, cands, xx, yy, args, adaptive, avail_modes, avail_cands)
        modes += eval_delta(block, actual, cands, xx, yy, args, adaptive, avail_modes, avail_cands)

    best = min(modes, key=lambda r: r.cost)

    return CSNode(
        x=x,
        y=y,
        w=w,
        h=h,
        depth=depth,
        parent=parent,
        split="leaf",
        best=best,
        actual=actual,
        avail_modes=avail_modes,
        avail_cands=avail_cands,
        bits=best.bits,
        sse=best.sse,
        cost=best.cost,
    )


# ============================================================
# Recursive block coding
# ============================================================

def add_leaves_to_store(node, store):
    if node.is_leaf():
        store.append(LeafRecord(node.x, node.y, node.w, node.h, node.best.plane))
        return

    for c in node.children:
        add_leaves_to_store(c, store)


def parent_node(x, y, w, h, depth, parent, split, split_bits, children, args, qt_flag_present):
    n = CSNode(
        x=x,
        y=y,
        w=w,
        h=h,
        depth=depth,
        parent=parent,
        split=split,
        children=children,
        split_bits=split_bits,
        qt_flag_present=qt_flag_present,
    )

    for c in children:
        c.parent = n

    n.bits = split_bits + sum(c.bits for c in children)
    n.sse = sum(c.sse for c in children)
    n.cost = args.lambda_rd * split_bits + sum(c.cost for c in children)

    return n


def encode_node(
    padded,
    x,
    y,
    w,
    h,
    depth,
    parent,
    args,
    grid,
    store,
    prev_store,
    adaptive,
    plane_warp_ctx,
):
    qt_ok = (
        depth < args.max_qt_depth
        and w >= 2
        and h >= 2
        and w % 2 == 0
        and h % 2 == 0
    )

    bh_ok = h >= 2 and h % 2 == 0
    bv_ok = w >= 2 and w % 2 == 0
    extra_ok = bh_ok or bv_ok

    cand = []

    leaf = eval_leaf(
        padded,
        x,
        y,
        w,
        h,
        depth,
        parent,
        args,
        grid,
        store,
        prev_store,
        adaptive,
        plane_warp_ctx,
    )

    leaf.qt_flag_present = qt_ok
    leaf.split_bits = 0.0

    if qt_ok:
        leaf.split_bits += qt_split_flag_bits(adaptive, depth, 0)

    if extra_ok:
        leaf.split_bits += 1.0

    leaf.bits += leaf.split_bits
    leaf.cost += args.lambda_rd * leaf.split_bits
    cand.append(leaf)

    if bh_ok:
        st = list(store)
        h0 = h // 2

        c0 = eval_leaf(
            padded,
            x,
            y,
            w,
            h0,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
            plane_warp_ctx,
        )
        add_leaves_to_store(c0, st)

        c1 = eval_leaf(
            padded,
            x,
            y + h0,
            w,
            h - h0,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
            plane_warp_ctx,
        )

        split_bits = 0.0

        if qt_ok:
            split_bits += qt_split_flag_bits(adaptive, depth, 0)

        split_bits += 1.0
        split_bits += 1.0

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "bh",
                split_bits,
                [c0, c1],
                args,
                qt_flag_present=qt_ok,
            )
        )

    if bv_ok:
        st = list(store)
        w0 = w // 2

        c0 = eval_leaf(
            padded,
            x,
            y,
            w0,
            h,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
            plane_warp_ctx,
        )
        add_leaves_to_store(c0, st)

        c1 = eval_leaf(
            padded,
            x + w0,
            y,
            w - w0,
            h,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
            plane_warp_ctx,
        )

        split_bits = 0.0

        if qt_ok:
            split_bits += qt_split_flag_bits(adaptive, depth, 0)

        split_bits += 1.0
        split_bits += 1.0

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "bv",
                split_bits,
                [c0, c1],
                args,
                qt_flag_present=qt_ok,
            )
        )

    if qt_ok:
        st = list(store)
        w0 = w // 2
        h0 = h // 2

        specs = [
            (x, y, w0, h0),
            (x + w0, y, w - w0, h0),
            (x, y + h0, w0, h - h0),
            (x + w0, y + h0, w - w0, h - h0),
        ]

        children = []

        for child_x, child_y, child_w, child_h in specs:
            c = encode_node(
                padded,
                child_x,
                child_y,
                child_w,
                child_h,
                depth + 1,
                None,
                args,
                grid,
                st,
                prev_store,
                adaptive,
                plane_warp_ctx,
            )

            children.append(c)
            add_leaves_to_store(c, st)

        split_bits = qt_split_flag_bits(adaptive, depth, 1)

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "qt",
                split_bits,
                children,
                args,
                qt_flag_present=True,
            )
        )

    best = min(cand, key=lambda n: n.cost)
    best.parent = parent

    return best


def commit_node(node, store, adaptive, writer, frame_idx):
    qt_split_flag_update(adaptive, node)

    if not node.is_leaf():
        for c in node.children:
            commit_node(c, store, adaptive, writer, frame_idx)
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
            for q, k in zip(b.q_values, "abc"):
                adaptive_signed_residual_update(
                    q,
                    adaptive[f"delta_res_abs_{k}"],
                    adaptive["delta_abs_max"],
                )

    store.append(LeafRecord(node.x, node.y, node.w, node.h, b.plane))

    if writer:
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
                "sse": node.sse,
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


def paint(node, recon):
    if node.is_leaf():
        recon[node.y : node.y + node.h, node.x : node.x + node.w] = node.best.recon_block
        return

    for c in node.children:
        paint(c, recon)


def collect(node, s):
    s["split_bits"] += node.split_bits

    if node.split == "qt":
        s["qt_nodes"] += 1
    elif node.split == "bh":
        s["bin_h_nodes"] += 1
    elif node.split == "bv":
        s["bin_v_nodes"] += 1

    if node.is_leaf():
        b = node.best
        s["leaf_blocks"] += 1
        s[f"{b.mode}_blocks"] += 1
        s[f"candidate_{b.candidate_name}_count"] = s.get(
            f"candidate_{b.candidate_name}_count",
            0,
        ) + 1

        if b.mode == "delta":
            s["delta_mode_count"] += 1

            if b.q_values == (0, 0, 0):
                s["zero_delta_blocks"] += 1

        return

    for c in node.children:
        collect(c, s)


# ============================================================
# Metrics / padding
# ============================================================

def pad_to_block_multiple(img, bs):
    h, w = img.shape
    ph = (bs - h % bs) % bs
    pw = (bs - w % bs) % bs

    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")

    return img.copy(), h + ph, w + pw


def compute_metrics(orig, recon, maxv, mask=None):
    d = orig.astype(np.float64) - recon.astype(np.float64)

    if mask is not None:
        mask = mask.astype(bool)
        if not np.any(mask):
            return {
                "mae": float("nan"),
                "mse": float("nan"),
                "rmse": float("nan"),
                "psnr": float("nan"),
                "max_error": float("nan"),
            }
        d = d[mask]

    mse = float(np.mean(d * d))

    return {
        "mae": float(np.mean(np.abs(d))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": float("inf") if mse == 0 else 10.0 * math.log10(maxv * maxv / mse),
        "max_error": float(np.max(np.abs(d))),
    }


# ============================================================
# Backward warping of previous GT video frame
# ============================================================

def bilinear_sample(img, map_x, map_y, valid, fill):
    h, w = img.shape

    mx = np.where(np.isfinite(map_x), map_x, 0.0)
    my = np.where(np.isfinite(map_y), map_y, 0.0)

    x0 = np.floor(mx).astype(np.int64)
    y0 = np.floor(my).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid2 = (
        valid
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < w)
        & (y1 < h)
    )

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    dx = mx - x0
    dy = my - y0

    v00 = img[y0c, x0c]
    v01 = img[y0c, x1c]
    v10 = img[y1c, x0c]
    v11 = img[y1c, x1c]

    out = (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v01
        + (1.0 - dx) * dy * v10
        + dx * dy * v11
    )

    out = np.where(valid2, out, fill)

    return np.clip(np.rint(out), 0, np.iinfo(np.uint16).max).astype(np.float64), valid2


def relative_camera_transform(cam_source, cam_target):
    """Return target_from_source in camera coordinates.

    For current_to_previous JSONL, the adjacent transforms were accumulated
    when the camera lookup was built. This relative transform also supports
    arbitrary RA references such as current 16 -> reference 0 or 32.
    """
    w2c_target = np.asarray(cam_target["W2C"], dtype=np.float64)
    c2w_source = np.asarray(cam_source["C2W"], dtype=np.float64)
    m = w2c_target @ c2w_source
    return m[:3, :3], m[:3, 3]


def make_projection_precompute_dual(width, height, cam_source, cam_target):
    """VGGT pixel-coordinate precompute.

    ``source`` is unprojected with its own intrinsic and ``target`` is
    projected with its own intrinsic. There is no NDC/projection matrix.
    """
    ks = np.asarray(cam_source["K"], dtype=np.float64)
    kt = np.asarray(cam_target["K"], dtype=np.float64)

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )

    return {
        "width": int(width),
        "height": int(height),
        "x_norm": (xs - float(ks[0, 2])) / float(ks[0, 0]),
        "y_norm": (ys - float(ks[1, 2])) / float(ks[1, 1]),
        "fx_target": float(kt[0, 0]),
        "fy_target": float(kt[1, 1]),
        "cx_target": float(kt[0, 2]),
        "cy_target": float(kt[1, 2]),
        "z_sign": float(cam_source["z_sign"]),
    }


def backward_map_fast_pixel_coord_dual(depth_linear, precomp, r, t):
    """Exact pixel-coordinate mapping used by the supplied VGGT script."""
    z = np.asarray(depth_linear, dtype=np.float64)
    x_norm = precomp["x_norm"]
    y_norm = precomp["y_norm"]
    z_sign = float(precomp["z_sign"])

    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    kx = r[0, 0] * x_norm + r[0, 1] * y_norm + r[0, 2] * z_sign
    ky = r[1, 0] * x_norm + r[1, 1] * y_norm + r[1, 2] * z_sign
    kz = r[2, 0] * x_norm + r[2, 1] * y_norm + r[2, 2] * z_sign

    xp = z * kx + float(t[0])
    yp = z * ky + float(t[1])
    zp = z * kz + float(t[2])

    denom = np.maximum(np.abs(zp), 1e-8)
    map_x = precomp["fx_target"] * (xp / denom) + precomp["cx_target"]
    map_y = precomp["fy_target"] * (yp / denom) + precomp["cy_target"]

    width = int(precomp["width"])
    height = int(precomp["height"])
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(z)
        & np.isfinite(zp)
        & (zp * z_sign > 0.0)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
        & (z > 0.0)
    )

    map_x = map_x.astype(np.float64)
    map_y = map_y.astype(np.float64)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid, zp


def make_backward_map_cur_to_prev(depth_y_cur, cam_cur, cam_prev, width, height):
    """Current-depth backward map to an arbitrary RA reference camera."""
    depth_linear = (
        np.asarray(depth_y_cur, dtype=np.float64)
        * get_depth_scale_real(cam_cur)
    )
    precomp = make_projection_precompute_dual(
        width,
        height,
        cam_source=cam_cur,
        cam_target=cam_prev,
    )
    r, t = relative_camera_transform(cam_cur, cam_prev)
    map_x, map_y, valid, _ = backward_map_fast_pixel_coord_dual(
        depth_linear,
        precomp,
        r,
        t,
    )
    return map_x, map_y, valid


def forward_project_recon_depth_to_target(
    ref_depth_y,
    cam_ref,
    cam_target,
    width,
    height,
):
    """Forward-project a decoded reference depth frame into the target view.

    A z-buffer keeps the nearest projected surface. The resulting depth is
    returned in the target frame's stored depth-sample domain.
    """
    depth_linear = (
        np.asarray(ref_depth_y, dtype=np.float64)
        * get_depth_scale_real(cam_ref)
    )
    precomp = make_projection_precompute_dual(
        width,
        height,
        cam_source=cam_ref,
        cam_target=cam_target,
    )
    r, t = relative_camera_transform(cam_ref, cam_target)
    map_x, map_y, valid, z_target_signed = backward_map_fast_pixel_coord_dual(
        depth_linear,
        precomp,
        r,
        t,
    )

    # Linear depth follows the supplied projection convention: abs(camera Z).
    z_target = np.abs(z_target_signed)
    valid &= np.isfinite(z_target) & (z_target > 0.0)

    zbuf = np.full(height * width, np.inf, dtype=np.float64)
    if np.any(valid):
        mx = map_x[valid]
        my = map_y[valid]
        zz = z_target[valid]
        x0 = np.floor(mx).astype(np.int64)
        y0 = np.floor(my).astype(np.int64)

        # Bilinear footprint with a nearest-depth z-buffer. The interpolation
        # weights are intentionally not used for depth: each source surface
        # contributes its actual transformed camera depth to covered samples.
        for dy in (0, 1):
            for dx in (0, 1):
                xi = x0 + dx
                yi = y0 + dy
                ok = (
                    (xi >= 0)
                    & (xi < width)
                    & (yi >= 0)
                    & (yi < height)
                )
                if np.any(ok):
                    flat = yi[ok] * width + xi[ok]
                    np.minimum.at(zbuf, flat, zz[ok])

    depth_target_linear = zbuf.reshape(height, width)
    target_valid = np.isfinite(depth_target_linear)
    depth_target_y = np.zeros((height, width), dtype=np.float64)
    depth_target_y[target_valid] = (
        depth_target_linear[target_valid]
        / get_depth_scale_real(cam_target)
    )
    depth_target_y[target_valid] = np.clip(
        depth_target_y[target_valid],
        0.0,
        1023.0,
    )
    return depth_target_y, target_valid


def combine_projected_depths(pred0, valid0, pred1=None, valid1=None):
    """Combine one or two projected reference depths without using GT depth."""
    pred0 = np.asarray(pred0, dtype=np.float64)
    valid0 = np.asarray(valid0, dtype=bool)

    if pred1 is None or valid1 is None:
        return pred0.copy(), valid0.copy(), np.zeros_like(valid0)

    pred1 = np.asarray(pred1, dtype=np.float64)
    valid1 = np.asarray(valid1, dtype=bool)
    both = valid0 & valid1
    only0 = valid0 & ~valid1
    only1 = valid1 & ~valid0

    out = np.zeros_like(pred0, dtype=np.float64)
    out[only0] = pred0[only0]
    out[only1] = pred1[only1]
    out[both] = 0.5 * (pred0[both] + pred1[both])
    return out, valid0 | valid1, both


def downsample_map_for_chroma(map_x, map_y, valid):
    h, w = map_x.shape
    hc = h // 2
    wc = w // 2

    mx = map_x[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    my = map_y[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    mv = valid[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) >= 0.5

    return mx, my, mv


def _single_reference_warp(ref_yuv, rec_depth_y, cam_ref, cam_cur, args):
    """Warp one already-coded reference view into the current view."""
    ref_y, ref_u, ref_v = ref_yuv
    h, w = ref_y.shape

    map_x, map_y, valid_y_map = make_backward_map_cur_to_prev(
        rec_depth_y,
        cam_cur,
        cam_ref,
        w,
        h,
    )

    if args.invalid_fill == "zero":
        fill_y = np.zeros_like(ref_y)
    elif args.invalid_fill == "neutral":
        fill_y = np.full_like(ref_y, args.max_value // 2)
    else:
        # Same-position sample from this reference. It is used only where the
        # geometric map is invalid; valid samples always come from bilinear warp.
        fill_y = ref_y

    pred_y, valid_y = bilinear_sample(ref_y, map_x, map_y, valid_y_map, fill_y)

    mx_c, my_c, valid_c_map = downsample_map_for_chroma(map_x, map_y, valid_y)

    if args.invalid_fill == "zero":
        fill_u = np.zeros_like(ref_u)
        fill_v = np.zeros_like(ref_v)
    elif args.invalid_fill == "neutral":
        neutral = min(512, args.max_value)
        fill_u = np.full_like(ref_u, neutral)
        fill_v = np.full_like(ref_v, neutral)
    else:
        fill_u = ref_u
        fill_v = ref_v

    pred_u, valid_u = bilinear_sample(ref_u, mx_c, my_c, valid_c_map, fill_u)
    pred_v, valid_v = bilinear_sample(ref_v, mx_c, my_c, valid_c_map, fill_v)

    return {
        "pred": (pred_y, pred_u, pred_v),
        "valid_y": valid_y,
        "valid_u": valid_u,
        "valid_v": valid_v,
    }


def _prediction_metrics(cur_yuv, pred_yuv, valid_y, valid_u, valid_v, args):
    cur_y, cur_u, cur_v = cur_yuv
    pred_y, pred_u, pred_v = pred_yuv

    my = compute_metrics(cur_y, pred_y, args.max_value)
    my_valid = compute_metrics(cur_y, pred_y, args.max_value, mask=valid_y)
    mu = compute_metrics(cur_u, pred_u, args.max_value)
    mv = compute_metrics(cur_v, pred_v, args.max_value)

    return {
        "warp_valid_y_ratio": float(np.mean(valid_y)),
        "warp_valid_uv_ratio": float(np.mean(valid_u & valid_v)),
        "warp_y_psnr": my["psnr"],
        "warp_y_mae": my["mae"],
        "warp_y_mse": my["mse"],
        "warp_y_psnr_valid": my_valid["psnr"],
        "warp_y_mae_valid": my_valid["mae"],
        "warp_u_psnr": mu["psnr"],
        "warp_v_psnr": mv["psnr"],
    }


def _blend_two_predictions(pred0, valid0, pred1, valid1, fill, max_value):
    """Blend two arrays without allowing invalid samples into the average."""
    both = valid0 & valid1
    only0 = valid0 & ~valid1
    only1 = valid1 & ~valid0

    out = np.asarray(fill, dtype=np.float64).copy()
    out[only0] = pred0[only0]
    out[only1] = pred1[only1]
    out[both] = 0.5 * (pred0[both] + pred1[both])

    return np.clip(np.rint(out), 0, max_value).astype(np.float64), both


def backward_warp_prev_yuv_to_cur(prev_yuv, cur_yuv, rec_depth_y, cam_prev, cam_cur, args):
    """Single-reference camera/depth predictor, used by sequential mode."""
    warped = _single_reference_warp(prev_yuv, rec_depth_y, cam_prev, cam_cur, args)
    stats = _prediction_metrics(
        cur_yuv,
        warped["pred"],
        warped["valid_y"],
        warped["valid_u"],
        warped["valid_v"],
        args,
    )
    stats.update(
        {
            "prediction_type": "P",
            "warp_l0_valid_y_ratio": float(np.mean(warped["valid_y"])),
            "warp_l1_valid_y_ratio": 0.0,
            "warp_both_valid_y_ratio": 0.0,
        }
    )
    return warped["pred"], stats


def bidirectional_warp_yuv_to_cur(
    l0_yuv,
    l1_yuv,
    cur_yuv,
    rec_depth_y,
    cam_l0,
    cam_l1,
    cam_cur,
    args,
):
    """Create an RA B-frame predictor from two camera/depth warps.

    Combination rule per sample:
      - both valid: arithmetic mean of L0 and L1
      - only one valid: use the valid prediction
      - neither valid: apply --invalid-fill
    """
    w0 = _single_reference_warp(l0_yuv, rec_depth_y, cam_l0, cam_cur, args)
    w1 = _single_reference_warp(l1_yuv, rec_depth_y, cam_l1, cam_cur, args)

    l0_y, l0_u, l0_v = l0_yuv
    l1_y, l1_u, l1_v = l1_yuv

    if args.invalid_fill == "zero":
        fill_y = np.zeros_like(l0_y)
        fill_u = np.zeros_like(l0_u)
        fill_v = np.zeros_like(l0_v)
    elif args.invalid_fill == "neutral":
        fill_y = np.full_like(l0_y, args.max_value // 2)
        neutral = min(512, args.max_value)
        fill_u = np.full_like(l0_u, neutral)
        fill_v = np.full_like(l0_v, neutral)
    else:
        # Same-position bi-prediction is a neutral fallback when neither
        # camera projection reaches the current sample.
        fill_y = 0.5 * (l0_y + l1_y)
        fill_u = 0.5 * (l0_u + l1_u)
        fill_v = 0.5 * (l0_v + l1_v)

    p0_y, p0_u, p0_v = w0["pred"]
    p1_y, p1_u, p1_v = w1["pred"]

    pred_y, both_y = _blend_two_predictions(
        p0_y, w0["valid_y"], p1_y, w1["valid_y"], fill_y, args.max_value
    )
    pred_u, both_u = _blend_two_predictions(
        p0_u, w0["valid_u"], p1_u, w1["valid_u"], fill_u, args.max_value
    )
    pred_v, both_v = _blend_two_predictions(
        p0_v, w0["valid_v"], p1_v, w1["valid_v"], fill_v, args.max_value
    )

    valid_y = w0["valid_y"] | w1["valid_y"]
    valid_u = w0["valid_u"] | w1["valid_u"]
    valid_v = w0["valid_v"] | w1["valid_v"]
    pred = (pred_y, pred_u, pred_v)

    stats = _prediction_metrics(cur_yuv, pred, valid_y, valid_u, valid_v, args)

    l0_valid_metrics = compute_metrics(
        cur_yuv[0], p0_y, args.max_value, mask=w0["valid_y"]
    )
    l1_valid_metrics = compute_metrics(
        cur_yuv[0], p1_y, args.max_value, mask=w1["valid_y"]
    )

    stats.update(
        {
            "prediction_type": "B",
            "warp_l0_valid_y_ratio": float(np.mean(w0["valid_y"])),
            "warp_l1_valid_y_ratio": float(np.mean(w1["valid_y"])),
            "warp_both_valid_y_ratio": float(np.mean(both_y)),
            "warp_both_valid_uv_ratio": float(np.mean(both_u & both_v)),
            "warp_l0_y_psnr_valid": l0_valid_metrics["psnr"],
            "warp_l1_y_psnr_valid": l1_valid_metrics["psnr"],
        }
    )

    return pred, stats


def intra_prediction_stats():
    """Statistics for RA anchor frames whose output predictor is the source frame."""
    return {
        "prediction_type": "I",
        "warp_valid_y_ratio": 0.0,
        "warp_valid_uv_ratio": 0.0,
        "warp_l0_valid_y_ratio": 0.0,
        "warp_l1_valid_y_ratio": 0.0,
        "warp_both_valid_y_ratio": 0.0,
        "warp_both_valid_uv_ratio": 0.0,
        "warp_y_psnr": float("inf"),
        "warp_y_mae": 0.0,
        "warp_y_mse": 0.0,
        "warp_y_psnr_valid": float("inf"),
        "warp_y_mae_valid": 0.0,
        "warp_u_psnr": float("inf"),
        "warp_v_psnr": float("inf"),
        "warp_l0_y_psnr_valid": float("nan"),
        "warp_l1_y_psnr_valid": float("nan"),
    }


# ============================================================
# Frame simulation
# ============================================================

def simulate_one_depth_frame(
    depth,
    frame_idx,
    args,
    grid,
    prev_store=None,
    writer=None,
    adaptive=None,
    plane_warp_ctx=None,
):
    h, w = depth.shape
    padded, hp, wp = pad_to_block_multiple(depth, args.block_size)

    recon = np.zeros_like(padded, dtype=np.float64)
    store = []
    prev_store = prev_store or []

    root_count = 0
    total_bits = 0.0
    total_sse = 0.0

    st = {
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
                padded,
                x,
                y,
                args.block_size,
                args.block_size,
                0,
                None,
                args,
                grid,
                store,
                prev_store,
                adaptive,
                plane_warp_ctx,
            )

            commit_node(root, store, adaptive, writer, frame_idx)
            paint(root, recon)
            collect(root, st)

            total_bits += root.bits
            total_sse += root.sse

    rec = recon[:h, :w]
    m = compute_metrics(depth, rec, args.max_value)

    leaves = max(int(st["leaf_blocks"]), 1)

    summary = {
        "frame": frame_idx,
        "width": w,
        "height": h,
        "padded_width": wp,
        "padded_height": hp,
        "root_block_size": args.block_size,
        "max_qt_depth": args.max_qt_depth,
        "num_roots": root_count,
        "leaf_blocks": int(st["leaf_blocks"]),
        "qt_nodes": int(st["qt_nodes"]),
        "bin_h_nodes": int(st["bin_h_nodes"]),
        "bin_v_nodes": int(st["bin_v_nodes"]),
        "split_bits": float(st["split_bits"]),
        "depth_bits": total_bits,
        "depth_bpp": total_bits / (h * w),
        "depth_sse": total_sse,
        "depth_mae": m["mae"],
        "depth_mse": m["mse"],
        "depth_rmse": m["rmse"],
        "depth_psnr": m["psnr"],
        "depth_max_error": m["max_error"],
        "direct_blocks": int(st["direct_blocks"]),
        "copy_blocks": int(st["copy_blocks"]),
        "delta_blocks": int(st["delta_blocks"]),
        "direct_ratio": st["direct_blocks"] / leaves,
        "copy_ratio": st["copy_blocks"] / leaves,
        "delta_ratio": st["delta_blocks"] / leaves,
        "zero_delta_blocks": int(st["zero_delta_blocks"]),
        "zero_delta_ratio_in_delta": (
            st["zero_delta_blocks"] / st["delta_mode_count"]
            if st["delta_mode_count"]
            else 0.0
        ),
    }

    for k, v in st.items():
        if k.startswith("candidate_"):
            summary[k.replace("-", "_")] = int(v)

    if adaptive is not None:
        summary.update(adaptive["mode"].snapshot("final_mode"))
        summary.update(adaptive["candidate"].snapshot("final_candidate"))

        if "copy_candidate_unary" in adaptive:
            for i, c in enumerate(adaptive["copy_candidate_unary"]):
                summary[f"final_copy_unary_ctx{i}_p1"] = c.p1

        if "qt_split" in adaptive:
            for i, c in enumerate(adaptive["qt_split"]):
                summary[f"final_qt_split_depth{i}_p1"] = c.p1

        if "delta_res_abs_a" in adaptive:
            for k in "abc":
                summary.update(
                    adaptive[f"delta_res_abs_{k}"].snapshot(f"final_delta_abs_{k}")
                )

    return rec, summary, store


# ============================================================
# YUV420p10le IO
# ============================================================

def frame_size_yuv420p10le(w, h):
    return w * h * 3


def count_frames(path, w, h):
    fs = frame_size_yuv420p10le(w, h)
    size = os.path.getsize(path)
    n = size // fs
    trailing = size % fs

    if trailing:
        print(f"[WARN] trailing bytes ignored: {path}, trailing={trailing}")

    return n


def read_yuv420p10le_frame(fp, idx, w, h):
    fs = frame_size_yuv420p10le(w, h)
    fp.seek(idx * fs)

    y_raw = fp.read(w * h * 2)
    if len(y_raw) != w * h * 2:
        raise EOFError(f"Failed to read Y frame {idx}")

    cw = w // 2
    ch = h // 2

    u_raw = fp.read(cw * ch * 2)
    v_raw = fp.read(cw * ch * 2)

    if len(u_raw) != cw * ch * 2 or len(v_raw) != cw * ch * 2:
        raise EOFError(f"Failed to read UV frame {idx}")

    y = np.frombuffer(y_raw, dtype="<u2").reshape(h, w).astype(np.float64)
    u = np.frombuffer(u_raw, dtype="<u2").reshape(ch, cw).astype(np.float64)
    v = np.frombuffer(v_raw, dtype="<u2").reshape(ch, cw).astype(np.float64)

    return y, u, v


def read_yuv420p10le_y_frame(fp, idx, w, h):
    y, _, _ = read_yuv420p10le_frame(fp, idx, w, h)
    return y


def write_yuv420p10le_frame(fp, y, u, v, maxv):
    y16 = np.clip(np.rint(y), 0, maxv).astype("<u2")
    u16 = np.clip(np.rint(u), 0, maxv).astype("<u2")
    v16 = np.clip(np.rint(v), 0, maxv).astype("<u2")

    fp.write(y16.tobytes())
    fp.write(u16.tobytes())
    fp.write(v16.tobytes())


def write_depth_as_yuv420p10le(fp, y, w, h, maxv):
    uv = np.full((h // 2, w // 2), min(512, maxv), dtype=np.float64)
    write_yuv420p10le_frame(fp, y, uv, uv, maxv)


def write_yuv420p10le_frame_at(fp, output_idx, y, u, v, w, h, maxv):
    """Write one frame at a display-order slot while coding in RA order."""
    if output_idx < 0:
        raise ValueError("output_idx must be non-negative")
    fp.seek(output_idx * frame_size_yuv420p10le(w, h))
    write_yuv420p10le_frame(fp, y, u, v, maxv)


def write_depth_as_yuv420p10le_at(fp, output_idx, y, w, h, maxv):
    uv = np.full((h // 2, w // 2), min(512, maxv), dtype=np.float64)
    write_yuv420p10le_frame_at(fp, output_idx, y, uv, uv, w, h, maxv)


# ============================================================
# Frame coding order / RA hierarchy
# ============================================================

def _append_ra_midpoints(lo, hi, layer, order, plan_by_frame, coded_rank):
    """Depth-first hierarchical B-frame order for one RA interval."""
    if hi - lo <= 1:
        return

    mid = (lo + hi) // 2
    if mid <= lo or mid >= hi:
        return

    # Both endpoints are already coded and become L0/L1 references.
    # depth_reference is retained only as a backward-compatible summary field;
    # analytic temporal coding below uses both endpoints independently.
    depth_ref = lo if coded_rank[lo] > coded_rank[hi] else hi

    order.append(mid)
    plan_by_frame[mid] = {
        "reference_l0": lo,
        "reference_l1": hi,
        "depth_reference": depth_ref,
        "prediction_type": "B",
        "temporal_layer": layer,
    }
    coded_rank[mid] = len(order) - 1

    _append_ra_midpoints(lo, mid, layer + 1, order, plan_by_frame, coded_rank)
    _append_ra_midpoints(mid, hi, layer + 1, order, plan_by_frame, coded_rank)


def build_frame_coding_plan(start, end, coding_order, ra_gop_size, ref_offset):
    """Return coding-order records for display frames in [start, end).

    RA GOP size 32 starts as:
      0(I), 32(I), 16(B:0/32), 8(B:0/16), 4(B:0/8), ...

    Output YUV files are still written in display order.
    """
    if start < 0 or end <= start:
        return []

    if coding_order == "sequential":
        plan = []
        for coding_idx, fi in enumerate(range(start, end)):
            ref = fi - ref_offset
            if ref < start:
                ref = None
            pred_type = "I" if ref is None else "P"
            plan.append(
                {
                    "frame": fi,
                    "reference_l0": ref,
                    "reference_l1": None,
                    "depth_reference": ref,
                    "prediction_type": pred_type,
                    "temporal_layer": 0,
                    "coding_order_idx": coding_idx,
                    "display_order_idx": fi - start,
                }
            )
        return plan

    order = [start]
    plan_by_frame = {
        start: {
            "reference_l0": None,
            "reference_l1": None,
            "depth_reference": None,
            "prediction_type": "I",
            "temporal_layer": 0,
        }
    }
    coded_rank = {start: 0}

    gop_start = start
    last_frame = end - 1

    while gop_start < last_frame:
        gop_end = min(gop_start + ra_gop_size, last_frame)

        # Each RA endpoint is coded as an independent anchor/I picture.
        if gop_end not in coded_rank:
            order.append(gop_end)
            plan_by_frame[gop_end] = {
                "reference_l0": None,
                "reference_l1": None,
                "depth_reference": None,
                "prediction_type": "I",
                "temporal_layer": 0,
            }
            coded_rank[gop_end] = len(order) - 1

        _append_ra_midpoints(
            gop_start,
            gop_end,
            1,
            order,
            plan_by_frame,
            coded_rank,
        )
        gop_start = gop_end

    expected = list(range(start, end))
    if sorted(order) != expected or len(order) != len(expected):
        raise RuntimeError(
            "Internal RA order error: requested frames were not emitted exactly once"
        )

    plan = []
    for coding_idx, fi in enumerate(order):
        rec = dict(plan_by_frame[fi])
        rec.update(
            {
                "frame": fi,
                "coding_order_idx": coding_idx,
                "display_order_idx": fi - start,
            }
        )
        plan.append(rec)

    return plan


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inverse-depth plane compression simulation + "
            "camera plane candidate + RA bidirectional projection predictor "
            "using camparam_v2 JSONL."
        )
    )

    p.add_argument("--input-depth", required=True, help="depth YUV420p10le sequence")
    p.add_argument(
        "--input-video",
        default="",
        help="GT video YUV420p10le sequence to backward warp. If empty, input-depth is used.",
    )
    p.add_argument("--camera-param", required=True, help="camparam_v2 JSONL")

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument(
        "--coding-order",
        choices=["ra", "sequential"],
        default="ra",
        help="Frame processing order. Default: hierarchical random access.",
    )
    p.add_argument(
        "--ra-gop-size",
        type=int,
        default=32,
        help="RA anchor interval. 32 gives 0(I),32(I),16(B),8(B),...",
    )
    p.add_argument(
        "--ref-offset",
        type=int,
        default=1,
        help="Reference offset used only with --coding-order sequential.",
    )

    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-qt-depth", type=int, default=0)

    p.add_argument("--lambda-rd", type=float, default=0.0)

    # Plane is 1/Y, so qsteps are much smaller than linear-depth qsteps.
    p.add_argument("--qa", type=float, default=1e-6)
    p.add_argument("--qb", type=float, default=1e-6)
    p.add_argument("--qc", type=float, default=1e-4)

    p.add_argument("--mode-bits", type=int, default=2)
    p.add_argument("--max-value", type=int, default=1023)
    p.add_argument("--depth-eps", type=float, default=1.0)

    # Camera-projected reconstructed-depth candidate. Enabled by default so
    # RA B pictures actually use temporal depth prediction from POC 16 onward.
    p.add_argument(
        "--plane-warp-candidate",
        dest="plane_warp_candidate",
        action="store_true",
        help="enable analytic L0/L1/average camera-plane candidates (default)",
    )
    p.add_argument(
        "--no-plane-warp-candidate",
        dest="plane_warp_candidate",
        action="store_false",
        help="disable analytic camera-plane candidates",
    )
    p.set_defaults(plane_warp_candidate=True)

    p.add_argument(
        "--same-position-temporal-candidate",
        dest="same_position_temporal_candidate",
        action="store_true",
        help=(
            "enable same-position L0/L1/average reconstructed-plane "
            "candidates (default)"
        ),
    )
    p.add_argument(
        "--no-same-position-temporal-candidate",
        dest="same_position_temporal_candidate",
        action="store_false",
        help="disable the same-position RA temporal plane candidate",
    )
    p.set_defaults(same_position_temporal_candidate=True)

    p.add_argument("--plane-warp-samples", type=int, default=5)
    p.add_argument("--plane-warp-min-valid-ratio", type=float, default=0.5)

    p.add_argument("--adaptive-prob", action="store_true")
    p.add_argument("--copy-candidate-unary", action="store_true")
    p.add_argument("--qt-split-adaptive", action="store_true")

    p.add_argument("--delta-residual-adaptive", action="store_true")
    p.add_argument("--delta-abs-max", type=int, default=7)

    p.add_argument("--max-candidates", type=int, default=8)

    p.add_argument("--prob-lr", type=float, default=0.05)
    p.add_argument("--prob-min", type=float, default=0.02)
    p.add_argument("--prob-max", type=float, default=0.95)
    p.add_argument("--prob-reset", choices=["frame", "sequence"], default="frame")

    p.add_argument(
        "--invalid-fill",
        choices=["prev_same", "zero", "neutral"],
        default="prev_same",
        help="fill strategy used only where no temporal warp is valid",
    )

    p.add_argument("--out-csv", default="inv_depth_plane_backward_stats.csv")
    p.add_argument("--out-json", default="inv_depth_plane_backward_summary.json")
    p.add_argument("--out-depth-recon-yuv", default="recon_inv_plane_depth.yuv")
    p.add_argument("--out-pred-yuv", default="backward_pred.yuv")
    p.add_argument("--out-block-csv", default="")

    return p.parse_args()


def validate_args(args):
    if args.block_size <= 0 or args.max_qt_depth < 0:
        raise ValueError("bad block/split size")

    if min(args.qa, args.qb, args.qc) <= 0:
        raise ValueError("qstep must be positive")

    if args.width % 2 or args.height % 2:
        raise ValueError("yuv420p10le requires even width/height")

    if args.copy_candidate_unary and not args.adaptive_prob:
        raise ValueError("--copy-candidate-unary requires --adaptive-prob")

    if args.qt_split_adaptive and not args.adaptive_prob:
        raise ValueError("--qt-split-adaptive requires --adaptive-prob")

    if args.delta_residual_adaptive and not args.adaptive_prob:
        raise ValueError("--delta-residual-adaptive requires --adaptive-prob")

    if args.max_candidates <= 0 or args.delta_abs_max < 0:
        raise ValueError("bad candidate/residual setting")

    if not (0 <= args.prob_lr <= 1):
        raise ValueError("bad probability setting")

    if args.prob_min < 0 or args.prob_max <= 0 or args.prob_min >= args.prob_max:
        raise ValueError("bad probability setting")

    if args.ref_offset <= 0:
        raise ValueError("--ref-offset must be positive")

    if args.ra_gop_size <= 0:
        raise ValueError("--ra-gop-size must be positive")

    if args.coding_order == "ra" and (args.ra_gop_size & (args.ra_gop_size - 1)) != 0:
        raise ValueError("--ra-gop-size must be a power of two in RA mode")

    if args.depth_eps <= 0:
        raise ValueError("--depth-eps must be positive")

    if args.plane_warp_samples < 2:
        raise ValueError("--plane-warp-samples must be >= 2")

    if not (0.0 <= args.plane_warp_min_valid_ratio <= 1.0):
        raise ValueError("--plane-warp-min-valid-ratio must be in [0,1]")


def ensure_camera_or_raise(camera_lookup, frame_idx, label):
    cam = get_camera(camera_lookup, frame_idx)

    if not camera_has_required_mats(cam):
        raise ValueError(f"{label} camera frame {frame_idx} is invalid")

    return cam


def main():
    args = parse_args()
    validate_args(args)

    video_path = args.input_video if args.input_video else args.input_depth

    depth_total = count_frames(args.input_depth, args.width, args.height)
    video_total = count_frames(video_path, args.width, args.height)
    total = min(depth_total, video_total)

    if total <= 0:
        raise ValueError("no complete frames found")

    if args.start_frame < 0 or args.start_frame >= total:
        raise ValueError("bad frame range")

    end = total if args.num_frames == 0 else min(total, args.start_frame + args.num_frames)

    coding_plan = build_frame_coding_plan(
        start=args.start_frame,
        end=end,
        coding_order=args.coding_order,
        ra_gop_size=args.ra_gop_size,
        ref_offset=args.ref_offset,
    )

    if not coding_plan:
        raise ValueError("empty coding plan")

    cam_json = load_camera_json(args.camera_param)
    cam_header = cam_json["header"]

    if int(cam_header["width"]) != args.width or int(cam_header["height"]) != args.height:
        raise ValueError(
            "camera/depth resolution mismatch: "
            f"camera={cam_header['width']}x{cam_header['height']}, "
            f"input={args.width}x{args.height}"
        )

    camera_lookup = build_camera_lookup(cam_json)
    depth_scale_real = get_depth_scale_real_from_header(cam_header)

    print(
        "Camera depth scale real: "
        f"{depth_scale_real:.12g} "
        "(depth_scale / depth_scale_precision)"
    )
    print(f"Camera pose mode       : {cam_header['pose_mode']}")
    print(f"Coding order           : {args.coding_order}")
    if args.coding_order == "ra":
        print(f"RA GOP size            : {args.ra_gop_size}")
    preview = ", ".join(str(x["frame"]) for x in coding_plan[:16])
    if len(coding_plan) > 16:
        preview += ", ..."
    print(f"Coding POC preview     : {preview}")

    grid = GridCache()

    seq_adapt = (
        create_adaptive_models(args)
        if args.adaptive_prob and args.prob_reset == "sequence"
        else None
    )

    summaries = []
    frame_store = {}
    frame_recon_depth = {}

    depth_recon_fp = open(args.out_depth_recon_yuv, "wb+") if args.out_depth_recon_yuv else None
    pred_fp = open(args.out_pred_yuv, "wb+") if args.out_pred_yuv else None

    block_fp = None
    writer = None

    if args.out_block_csv:
        block_fp = open(args.out_block_csv, "w", newline="")

        fields = [
            "frame",
            "bx",
            "by",
            "block_w",
            "block_h",
            "qt_depth",
            "split_type",
            "mode",
            "candidate",
            "bits",
            "split_bits",
            "sse",
            "cost",
            "q0",
            "q1",
            "q2",
            "actual_inv_a",
            "actual_inv_b",
            "actual_inv_c",
            "recon_inv_a",
            "recon_inv_b",
            "recon_inv_c",
        ]

        writer = csv.DictWriter(block_fp, fieldnames=fields)
        writer.writeheader()

    try:
        with open(args.input_depth, "rb") as depth_fp, open(video_path, "rb") as video_fp:
            for item in coding_plan:
                fi = int(item["frame"])
                ref_l0 = item["reference_l0"]
                ref_l1 = item["reference_l1"]
                depth_ref_idx = item["depth_reference"]
                prediction_type = str(item["prediction_type"])
                coding_idx = int(item["coding_order_idx"])
                display_idx = int(item["display_order_idx"])
                temporal_layer = int(item["temporal_layer"])

                if args.adaptive_prob and args.prob_reset == "frame":
                    adaptive = create_adaptive_models(args)
                else:
                    adaptive = seq_adapt

                depth_y = read_yuv420p10le_y_frame(depth_fp, fi, args.width, args.height)
                cur_video = read_yuv420p10le_frame(video_fp, fi, args.width, args.height)

                cam_cur = ensure_camera_or_raise(camera_lookup, fi, "current")

                # --------------------------------------------------------
                # RA temporal depth candidates.
                #
                # Both same-position temporal candidates and camera-plane
                # candidates use reconstructed leaf planes from the actual RA
                # L0/L1 references. Camera candidates are transformed
                # analytically per block, matching the successful LD path.
                # --------------------------------------------------------
                plane_warp_ctx = None
                depth_pred_stats = {
                    "depth_pred_valid_ratio": 0.0,
                    "depth_pred_l0_valid_ratio": 0.0,
                    "depth_pred_l1_valid_ratio": 0.0,
                    "depth_pred_both_valid_ratio": 0.0,
                }

                temporal_candidate_enabled = (
                    args.plane_warp_candidate
                    or args.same_position_temporal_candidate
                )

                if temporal_candidate_enabled and prediction_type in ("P", "B"):
                    if ref_l0 is None or ref_l0 not in frame_store:
                        raise RuntimeError(
                            f"POC {fi}: reconstructed L0 plane store {ref_l0} "
                            "is unavailable"
                        )

                    cam_l0_depth = ensure_camera_or_raise(
                        camera_lookup,
                        ref_l0,
                        "depth-L0-reference",
                    )

                    l1_store = None
                    cam_l1_depth = None
                    if prediction_type == "B":
                        if ref_l1 is None or ref_l1 not in frame_store:
                            raise RuntimeError(
                                f"POC {fi}: reconstructed L1 plane store {ref_l1} "
                                "is unavailable"
                            )
                        l1_store = frame_store[ref_l1]
                        cam_l1_depth = ensure_camera_or_raise(
                            camera_lookup,
                            ref_l1,
                            "depth-L1-reference",
                        )

                    plane_warp_ctx = PlaneWarpContext(
                        l0_store=frame_store[ref_l0],
                        cam_l0=cam_l0_depth,
                        cam_cur=cam_cur,
                        frame_w=args.width,
                        frame_h=args.height,
                        l1_store=l1_store,
                        cam_l1=cam_l1_depth,
                        source_type=(
                            "analytic_plane_bi"
                            if prediction_type == "B"
                            else "analytic_plane_l0"
                        ),
                    )

                    depth_pred_stats = {
                        "depth_pred_valid_ratio": 1.0,
                        "depth_pred_l0_valid_ratio": 1.0,
                        "depth_pred_l1_valid_ratio": (
                            1.0 if prediction_type == "B" else 0.0
                        ),
                        "depth_pred_both_valid_ratio": (
                            1.0 if prediction_type == "B" else 0.0
                        ),
                    }

                rec_depth_y, sm, cur_store = simulate_one_depth_frame(
                    depth_y,
                    fi,
                    args,
                    grid,
                    prev_store=None,
                    writer=writer,
                    adaptive=adaptive,
                    plane_warp_ctx=plane_warp_ctx,
                )

                frame_store[fi] = cur_store
                frame_recon_depth[fi] = np.asarray(rec_depth_y, dtype=np.float64).copy()
                sm.update(depth_pred_stats)

                if depth_recon_fp:
                    write_depth_as_yuv420p10le_at(
                        depth_recon_fp,
                        display_idx,
                        rec_depth_y,
                        args.width,
                        args.height,
                        args.max_value,
                    )

                if prediction_type == "B":
                    if ref_l0 is None or ref_l1 is None:
                        raise RuntimeError(
                            f"POC {fi}: B prediction requires both L0 and L1"
                        )

                    l0_video = read_yuv420p10le_frame(
                        video_fp, ref_l0, args.width, args.height
                    )
                    l1_video = read_yuv420p10le_frame(
                        video_fp, ref_l1, args.width, args.height
                    )
                    cam_l0 = ensure_camera_or_raise(
                        camera_lookup, ref_l0, "warp-L0-reference"
                    )
                    cam_l1 = ensure_camera_or_raise(
                        camera_lookup, ref_l1, "warp-L1-reference"
                    )

                    pred_video, warp_stats = bidirectional_warp_yuv_to_cur(
                        l0_video,
                        l1_video,
                        cur_video,
                        rec_depth_y,
                        cam_l0,
                        cam_l1,
                        cam_cur,
                        args,
                    )

                elif prediction_type == "P":
                    if ref_l0 is None:
                        raise RuntimeError(f"POC {fi}: P prediction requires L0")

                    l0_video = read_yuv420p10le_frame(
                        video_fp, ref_l0, args.width, args.height
                    )
                    cam_l0 = ensure_camera_or_raise(
                        camera_lookup, ref_l0, "warp-L0-reference"
                    )

                    pred_video, warp_stats = backward_warp_prev_yuv_to_cur(
                        l0_video,
                        cur_video,
                        rec_depth_y,
                        cam_l0,
                        cam_cur,
                        args,
                    )

                else:
                    # RA endpoints (e.g. POC 0 and 32) are I/anchor frames.
                    pred_video = cur_video
                    warp_stats = intra_prediction_stats()

                if pred_fp:
                    write_yuv420p10le_frame_at(
                        pred_fp,
                        display_idx,
                        pred_video[0],
                        pred_video[1],
                        pred_video[2],
                        args.width,
                        args.height,
                        args.max_value,
                    )

                sm.update(warp_stats)
                sm["depth_scale_real"] = depth_scale_real
                sm["camera_pose_mode"] = cam_header["pose_mode"]
                sm["coding_order"] = args.coding_order
                sm["coding_order_idx"] = coding_idx
                sm["display_order_idx"] = display_idx
                sm["prediction_type"] = prediction_type
                sm["reference_l0"] = -1 if ref_l0 is None else int(ref_l0)
                sm["reference_l1"] = -1 if ref_l1 is None else int(ref_l1)
                sm["plane_reference_frame"] = (
                    -1 if ref_l0 is None else int(ref_l0)
                )
                sm["plane_reference_l0"] = -1 if ref_l0 is None else int(ref_l0)
                sm["plane_reference_l1"] = -1 if ref_l1 is None else int(ref_l1)
                sm["depth_prediction_type"] = (
                    "none" if plane_warp_ctx is None else plane_warp_ctx.source_type
                )
                sm["same_position_temporal_enabled"] = int(
                    args.same_position_temporal_candidate
                )
                sm["same_position_temporal_count"] = (
                    0
                    if plane_warp_ctx is None
                    else 2 if plane_warp_ctx.l1_store is not None else 1
                )
                # Backward-compatible single-reference column.
                sm["reference_frame"] = -1 if ref_l0 is None else int(ref_l0)
                sm["temporal_layer"] = temporal_layer

                summaries.append(sm)

                l0_text = "-" if ref_l0 is None else str(ref_l0)
                l1_text = "-" if ref_l1 is None else str(ref_l1)
                print(
                    f"CO={coding_idx:4d} | "
                    f"POC={fi:4d} | "
                    f"Type={prediction_type} | "
                    f"L0/L1={l0_text:>4s}/{l1_text:<4s} | "
                    f"TL={temporal_layer:2d} | "
                    f"depthBits={sm['depth_bits']:.1f} | "
                    f"depthPSNR={sm['depth_psnr']:.3f} | "
                    f"warpYPSNR={sm['warp_y_psnr']:.3f} | "
                    f"warpYPSNRValid={sm['warp_y_psnr_valid']:.3f} | "
                    f"valid={sm['warp_valid_y_ratio']:.3f} | "
                    f"depthPredValid={sm['depth_pred_valid_ratio']:.3f} | "
                    f"leaf={sm['leaf_blocks']} | "
                    f"QT={sm['qt_nodes']} | "
                    f"BH/BV={sm['bin_h_nodes']}/{sm['bin_v_nodes']} | "
                    f"D/C/Δ={sm['direct_ratio']:.3f}/"
                    f"{sm['copy_ratio']:.3f}/"
                    f"{sm['delta_ratio']:.3f}"
                )

    finally:
        if depth_recon_fp:
            depth_recon_fp.close()

        if pred_fp:
            pred_fp.close()

        if block_fp:
            block_fp.close()

    if not summaries:
        raise RuntimeError("No frames processed")

    with open(args.out_csv, "w", newline="") as f:
        fields = sorted(set().union(*(s.keys() for s in summaries)))
        w_csv = csv.DictWriter(f, fieldnames=fields)
        w_csv.writeheader()
        w_csv.writerows(summaries)

    avg = {}

    for k in sorted(set().union(*(s.keys() for s in summaries))):
        vals = []

        for s in summaries:
            try:
                v = float(s[k])

                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass

        if vals:
            avg[k] = float(np.mean(vals))

    total_depth_bits = float(sum(s["depth_bits"] for s in summaries))
    total_pixels = float(args.width * args.height * len(summaries))

    overall = {
        **vars(args),
        "input_video_resolved": video_path,
        "camera_format": cam_header.get("format"),
        "camera_pose_mode": cam_header["pose_mode"],
        "projection_mode": "vggt_fast_pixel_coordinate_dual_intrinsic_no_ndc",
        "ra_depth_prediction": (
            "analytic_transform_reconstructed_L0_L1_leaf_planes"
            "+analytic_l0_l1_avg_plane_transform"
            "+same_position_l0_l1_avg_plane_candidates"
        ),
        "depth_scale": cam_header["depth_scale"],
        "depth_scale_precision": cam_header["depth_scale_precision"],
        "depth_scale_real_used": depth_scale_real,
        "num_processed_frames": len(summaries),
        "coding_plan": coding_plan,
        "coding_poc_order": [int(x["frame"]) for x in coding_plan],
        "total_depth_bits": total_depth_bits,
        "overall_depth_bpp": total_depth_bits / total_pixels,
        "average": avg,
        "frame_csv": args.out_csv,
        "depth_recon_yuv": args.out_depth_recon_yuv,
        "pred_yuv": args.out_pred_yuv,
        "block_csv": args.out_block_csv,
        "output_yuv_order": "display_order",
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print()
    print("Done.")
    print(f"Frame CSV       : {args.out_csv}")
    print(f"Summary         : {args.out_json}")
    print(f"Recon depth YUV : {args.out_depth_recon_yuv}")
    print(f"RA inter pred   : {args.out_pred_yuv}")

    if args.out_block_csv:
        print(f"Block CSV       : {args.out_block_csv}")

    print(f"Overall depth bpp: {overall['overall_depth_bpp']:.6f}")


if __name__ == "__main__":
    main()
