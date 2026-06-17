#!/usr/bin/env python3
# depth_inv_plane_backward_warp_sim.py

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
    # Inverse-depth plane:
    #   invY(x,y) = a * (x - cx) + b * (y - cy) + c
    # where Y is stored depth sample, e.g. Z_real = Y * near.
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
    prev_store: List[LeafRecord]
    cam_prev: Dict[str, Any]
    cam_cur: Dict[str, Any]
    frame_w: int
    frame_h: int


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
    cand_symbols = [
        "plane_warp",
        "temporal",
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
# Spatial / temporal candidates
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


def temporal_center(prev_store, cx, cy):
    if not prev_store:
        return None

    for r in prev_store:
        if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
            return r

    return None


# ============================================================
# Camera JSON / JSONL / matrices
# ============================================================

def load_camera_json(path):
    """
    Supports:
      1. JSON object
      2. JSON array
      3. JSONL, one camera frame per line

    Matches the previously verified forward-warp parser style.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_err:
        entries = []

        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as jsonl_err:
                raise RuntimeError(
                    f"Failed to parse camera parameter file as JSON or JSONL: {path}\n"
                    f"JSON error={json_err}\n"
                    f"JSONL error at line {line_no}: {jsonl_err}"
                ) from jsonl_err

            entries.append(obj)

        if not entries:
            raise RuntimeError(f"Camera parameter file is empty or invalid: {path}")

        return entries


def has_camera_matrices(obj):
    if not isinstance(obj, dict):
        return False

    keys = set(obj.keys())

    inv_keys = {
        "InvProjectionMatrix",
        "invProjectionMatrix",
        "InverseProjectionMatrix",
    }
    proj_keys = {
        "ProjectionMatrix",
        "projectionMatrix",
    }
    w2c_keys = {
        "WorldToCameraMatrix",
        "worldToCameraMatrix",
        "ViewMatrix",
    }
    c2w_keys = {
        "CameraToWorldMatrix",
        "cameraToWorldMatrix",
        "CameraToWorldMarix",
        "cameraToWorldMarix",
        "InvViewMatrix",
    }

    return (
        bool(keys & inv_keys)
        and bool(keys & proj_keys)
        and bool(keys & w2c_keys)
        and bool(keys & c2w_keys)
    )


def camera_frame_poc(entry, fallback_idx):
    for key in ["frames", "frame", "Frame", "frameIdx", "frame_idx", "poc", "POC"]:
        if isinstance(entry, dict) and key in entry:
            try:
                return int(entry[key])
            except Exception:
                pass

    return int(fallback_idx)


def extract_camera_entries(cams_json):
    entries = []

    if isinstance(cams_json, list):
        entries = cams_json

    elif isinstance(cams_json, dict):
        for k in [
            "frames",
            "Frames",
            "cameras",
            "Cameras",
            "cameraFrames",
            "CameraFrames",
            "camera_params",
            "cameraParams",
        ]:
            if k in cams_json and isinstance(cams_json[k], list):
                entries = cams_json[k]
                break

        if not entries and has_camera_matrices(cams_json):
            entries = [cams_json]

        if not entries:
            numeric_items = []
            for k, v in cams_json.items():
                if isinstance(v, dict) and has_camera_matrices(v):
                    try:
                        numeric_items.append((int(k), v))
                    except Exception:
                        pass

            if numeric_items:
                numeric_items.sort(key=lambda x: x[0])
                entries = [v for _, v in numeric_items]

    if not entries:
        raise RuntimeError("Cannot parse camera parameter structure")

    return entries


def build_camera_lookup(cams_json):
    entries = extract_camera_entries(cams_json)

    lookup = {}

    for i, entry in enumerate(entries):
        poc = camera_frame_poc(entry, i)

        lookup[poc] = entry

        # fallback by list index
        if i not in lookup:
            lookup[i] = entry

    return lookup


def get_camera(camera_lookup, frame_idx):
    if frame_idx not in camera_lookup:
        raise KeyError(f"Camera for frame {frame_idx} not found")
    return camera_lookup[frame_idx]


def get_value_by_alias(entry, aliases, required=True):
    for key in aliases:
        if isinstance(entry, dict) and key in entry:
            return entry[key]

    if required:
        raise KeyError(f"Missing key. aliases={aliases}")

    return None


def get_near_clip(entry):
    val = get_value_by_alias(
        entry,
        ["nearClipPlane", "NearClipPlane", "near_clip_plane", "near", "Near"],
        required=False,
    )

    if val is None:
        return 1.0

    return float(val)


def get_matrix(frame_entry, logical_name_or_aliases):
    """
    Important:
      If matrix is stored as dict e00~e33, return mat.T.

    This matches the previously verified forward-warp code:
      # ì¤ì: dict e00~e33 matrixë transposeí´ì ì¬ì©
      return mat.T

    Accepts either:
      get_matrix(cam, "ProjectionMatrix")
      get_matrix(cam, ["ProjectionMatrix", "projectionMatrix"])
    """
    alias_map = {
        "InvProjectionMatrix": [
            "InvProjectionMatrix",
            "invProjectionMatrix",
            "InverseProjectionMatrix",
        ],
        "ProjectionMatrix": [
            "ProjectionMatrix",
            "projectionMatrix",
        ],
        "WorldToCameraMatrix": [
            "WorldToCameraMatrix",
            "worldToCameraMatrix",
            "ViewMatrix",
        ],
        "CameraToWorldMatrix": [
            "CameraToWorldMatrix",
            "cameraToWorldMatrix",
            "CameraToWorldMarix",
            "cameraToWorldMarix",
            "InvViewMatrix",
        ],
    }

    if isinstance(logical_name_or_aliases, str):
        aliases = alias_map.get(logical_name_or_aliases, [logical_name_or_aliases])
    else:
        aliases = list(logical_name_or_aliases)

    obj = get_value_by_alias(frame_entry, aliases, required=True)

    if isinstance(obj, dict):
        mat = np.zeros((4, 4), dtype=np.float64)

        for r in range(4):
            for c in range(4):
                key = f"e{r}{c}"
                if key not in obj:
                    raise KeyError(f"Missing matrix key {key} for aliases={aliases}")
                mat[r, c] = float(obj[key])

        return mat.T

    mat = np.array(obj, dtype=np.float64)

    if mat.shape == (16,):
        mat = mat.reshape(4, 4)

    if mat.shape != (4, 4):
        raise ValueError(f"matrix shape is {mat.shape}, expected 4x4. aliases={aliases}")

    return mat


def camera_has_required_mats(cam):
    try:
        get_matrix(cam, "InvProjectionMatrix")
        get_matrix(cam, "ProjectionMatrix")
        get_matrix(cam, "WorldToCameraMatrix")
        get_matrix(cam, "CameraToWorldMatrix")
        return True
    except Exception:
        return False


# ============================================================
# Camera geometry
# ============================================================

def pixel_rays_camera(u, v, width, height, inv_proj):
    x_ndc = ((u + 0.5) / float(width)) * 2.0 - 1.0
    y_ndc = 1.0 - ((v + 0.5) / float(height)) * 2.0

    ones = np.ones_like(x_ndc, dtype=np.float64)
    p_ndc = np.stack([x_ndc, y_ndc, ones, ones], axis=-1)

    p_view_h = p_ndc @ inv_proj.T
    w = p_view_h[..., 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)

    p_view = p_view_h[..., :3] / w
    z = p_view[..., 2:3]
    z_abs = np.maximum(np.abs(z), 1e-12)

    # Same convention as verified forward warp:
    # p_view_scaled = p_view / abs(p_view.z) * depth
    return p_view / z_abs


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
    c2w_src = get_matrix(cam_src, "CameraToWorldMatrix")
    w2c_tgt = get_matrix(cam_tgt, "WorldToCameraMatrix")

    # Column-vector transform equivalent to:
    # p_tgt = W2C_tgt @ C2W_src @ p_src
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
    inv_proj = get_matrix(cam, "InvProjectionMatrix")
    near = get_near_clip(cam)

    ns = max(2, int(args.plane_warp_samples))

    xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)
    ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)

    depth_y = inv_plane_to_depth_value(leaf.plane, uu, vv, args)
    linear_z = depth_y * near

    rays = pixel_rays_camera(uu, vv, frame_w, frame_h, inv_proj)
    pts = rays.reshape(-1, 3) * linear_z.reshape(-1, 1)

    valid = np.isfinite(pts).all(axis=1) & (linear_z.reshape(-1) > 0)

    pts = pts[valid]

    if pts.shape[0] < 3:
        return None

    return fit_3d_plane(pts)


def render_3d_plane_to_depth_block(plane_cam, cam_cur, x, y, w, h, frame_w, frame_h, args):
    inv_proj = get_matrix(cam_cur, "InvProjectionMatrix")
    near = get_near_clip(cam_cur)

    gx = np.arange(x, x + w, dtype=np.float64)
    gy = np.arange(y, y + h, dtype=np.float64)
    uu, vv = np.meshgrid(gx, gy)

    rays = pixel_rays_camera(uu, vv, frame_w, frame_h, inv_proj)

    n = plane_cam[:3]
    d = plane_cam[3]

    denom = (
        n[0] * rays[..., 0]
        + n[1] * rays[..., 1]
        + n[2] * rays[..., 2]
    )

    valid = np.abs(denom) > 1e-12
    scale = np.full((h, w), np.nan, dtype=np.float64)
    scale[valid] = -d / denom[valid]

    # rays have abs(z)==1, so scale is linear depth.
    valid = valid & np.isfinite(scale) & (scale > 0)

    valid_ratio = float(np.mean(valid))

    if valid_ratio < args.plane_warp_min_valid_ratio:
        return None

    depth_y = scale / max(near, 1e-12)
    depth_y = np.clip(depth_y, args.depth_eps, args.max_value)

    if not np.all(valid):
        med = np.median(depth_y[valid]) if np.any(valid) else args.max_value
        depth_y[~valid] = med

    return depth_y


def make_plane_warp_candidate(ctx, x, y, w, h, cx, cy, args, grid):
    if ctx is None:
        return None

    r = temporal_center(ctx.prev_store, cx, cy)

    if r is None:
        return None

    plane3d_src = image_inv_plane_to_3d_plane(
        r,
        ctx.cam_prev,
        ctx.frame_w,
        ctx.frame_h,
        args,
    )

    if plane3d_src is None:
        return None

    plane3d_cur = transform_plane_src_to_tgt(
        plane3d_src,
        ctx.cam_prev,
        ctx.cam_cur,
    )

    if plane3d_cur is None:
        return None

    depth_block = render_3d_plane_to_depth_block(
        plane3d_cur,
        ctx.cam_cur,
        x,
        y,
        w,
        h,
        ctx.frame_w,
        ctx.frame_h,
        args,
    )

    if depth_block is None:
        return None

    _, _, pinv = grid.get(w, h)
    return fit_inv_depth_plane_from_depth_block(depth_block, pinv, cx, cy, args)


def make_candidates(
    store,
    prev_store,
    x,
    y,
    w,
    h,
    cx,
    cy,
    max_cands,
    use_temporal,
    plane_warp_ctx,
    args,
    grid,
):
    cand = []
    conv = {}

    if args.plane_warp_candidate and plane_warp_ctx is not None:
        p = make_plane_warp_candidate(
            plane_warp_ctx,
            x,
            y,
            w,
            h,
            cx,
            cy,
            args,
            grid,
        )

        if p is not None:
            conv["plane_warp"] = p
            cand.append(("plane_warp", p))

    if use_temporal:
        r = temporal_center(prev_store, cx, cy)

        if r is not None:
            p = plane_to_center(r.plane, cx, cy)
            conv["temporal"] = p
            cand.append(("temporal", p))

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
        prev_store=prev_store,
        x=x,
        y=y,
        w=w,
        h=h,
        cx=cx,
        cy=cy,
        max_cands=args.max_candidates,
        use_temporal=args.temporal_candidate,
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

        for cx, cy, cw, ch in specs:
            c = encode_node(
                padded,
                cx,
                cy,
                cw,
                ch,
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

    safe_x = np.where(np.isfinite(map_x), map_x, 0.0)
    safe_y = np.where(np.isfinite(map_y), map_y, 0.0)

    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid2 = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < w)
        & (y1 < h)
    )

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    dx = safe_x - x0
    dy = safe_y - y0

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


def make_backward_map_cur_to_prev(depth_y_cur, cam_cur, cam_prev, width, height):
    inv_proj_cur = get_matrix(cam_cur, "InvProjectionMatrix")
    c2w_cur = get_matrix(cam_cur, "CameraToWorldMatrix")
    w2c_prev = get_matrix(cam_prev, "WorldToCameraMatrix")
    proj_prev = get_matrix(cam_prev, "ProjectionMatrix")

    near_cur = get_near_clip(cam_cur)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )

    rays = pixel_rays_camera(xx, yy, width, height, inv_proj_cur)
    linear_z = depth_y_cur.astype(np.float64) * near_cur
    p_cur = rays * linear_z[..., None]

    ones = np.ones((height, width, 1), dtype=np.float64)
    p_cur_h = np.concatenate([p_cur, ones], axis=-1)

    p_world = p_cur_h @ c2w_cur.T
    p_prev = p_world @ w2c_prev.T
    clip = p_prev @ proj_prev.T

    cw = clip[..., 3]
    valid = np.abs(cw) > 1e-12

    ndc_x = np.zeros_like(cw)
    ndc_y = np.zeros_like(cw)

    ndc_x[valid] = clip[..., 0][valid] / cw[valid]
    ndc_y[valid] = clip[..., 1][valid] / cw[valid]

    map_x = (ndc_x + 1.0) * 0.5 * width - 0.5
    map_y = (1.0 - ndc_y) * 0.5 * height - 0.5

    valid = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(linear_z)
        & (linear_z > 0)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y <= height - 1.0)
    )

    return map_x, map_y, valid


def downsample_map_for_chroma(map_x, map_y, valid):
    h, w = map_x.shape
    hc = h // 2
    wc = w // 2

    mx = map_x[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    my = map_y[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    mv = valid[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) >= 0.5

    return mx, my, mv


def backward_warp_prev_yuv_to_cur(prev_yuv, cur_yuv, rec_depth_y, cam_prev, cam_cur, args):
    prev_y, prev_u, prev_v = prev_yuv
    cur_y, cur_u, cur_v = cur_yuv

    h, w = cur_y.shape
    map_x, map_y, valid_y = make_backward_map_cur_to_prev(
        rec_depth_y,
        cam_cur,
        cam_prev,
        w,
        h,
    )

    if args.invalid_fill == "zero":
        fill_y = np.zeros_like(cur_y)
    elif args.invalid_fill == "neutral":
        fill_y = np.full_like(cur_y, args.max_value // 2)
    else:
        fill_y = prev_y

    pred_y, valid_y = bilinear_sample(prev_y, map_x, map_y, valid_y, fill_y)

    mx_c, my_c, valid_c = downsample_map_for_chroma(map_x, map_y, valid_y)

    if args.invalid_fill == "zero":
        fill_u = np.zeros_like(cur_u)
        fill_v = np.zeros_like(cur_v)
    elif args.invalid_fill == "neutral":
        fill_u = np.full_like(cur_u, min(512, args.max_value))
        fill_v = np.full_like(cur_v, min(512, args.max_value))
    else:
        fill_u = prev_u
        fill_v = prev_v

    pred_u, valid_u = bilinear_sample(prev_u, mx_c, my_c, valid_c, fill_u)
    pred_v, valid_v = bilinear_sample(prev_v, mx_c, my_c, valid_c, fill_v)

    my = compute_metrics(cur_y, pred_y, args.max_value)
    my_valid = compute_metrics(cur_y, pred_y, args.max_value, mask=valid_y)

    mu = compute_metrics(cur_u, pred_u, args.max_value)
    mv = compute_metrics(cur_v, pred_v, args.max_value)

    stats = {
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

    return (pred_y, pred_u, pred_v), stats


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
    size = os.path.getsize(path)
    frame_size = frame_size_yuv420p10le(w, h)
    trailing = size % frame_size
    if trailing:
        print(f"[WARN] trailing bytes ignored: {path}, trailing={trailing}")
    return size // frame_size


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


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inverse-depth plane compression simulation + "
            "camera plane candidate + backward projection predictor."
        )
    )

    p.add_argument("--input-depth", required=True, help="depth YUV420p10le sequence")
    p.add_argument(
        "--input-video",
        default="",
        help="GT video YUV420p10le sequence to backward warp. If empty, input-depth is used.",
    )
    p.add_argument("--camera-param", required=True, help="camera JSON/JSONL/TXT")

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--ref-offset", type=int, default=1)

    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-qt-depth", type=int, default=0)

    p.add_argument("--lambda-rd", type=float, default=0.0)

    # Since plane is now 1/Y, qsteps must be much smaller than linear-depth qsteps.
    p.add_argument("--qa", type=float, default=1e-6)
    p.add_argument("--qb", type=float, default=1e-6)
    p.add_argument("--qc", type=float, default=1e-4)

    p.add_argument("--mode-bits", type=int, default=2)
    p.add_argument("--max-value", type=int, default=1023)
    p.add_argument("--depth-eps", type=float, default=1.0)

    p.add_argument("--temporal-candidate", action="store_true")
    p.add_argument("--plane-warp-candidate", action="store_true")
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
        help="fill strategy for out-of-view pixels in backward warped predictor",
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

    if args.depth_eps <= 0:
        raise ValueError("--depth-eps must be positive")

    if args.plane_warp_samples < 2:
        raise ValueError("--plane-warp-samples must be >= 2")


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

    cam_json = load_camera_json(args.camera_param)
    camera_lookup = build_camera_lookup(cam_json)

    grid = GridCache()

    seq_adapt = (
        create_adaptive_models(args)
        if args.adaptive_prob and args.prob_reset == "sequence"
        else None
    )

    summaries = []
    prev_store = None
    prev_processed_frame_idx = None

    depth_recon_fp = open(args.out_depth_recon_yuv, "wb") if args.out_depth_recon_yuv else None
    pred_fp = open(args.out_pred_yuv, "wb") if args.out_pred_yuv else None

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
            for fi in range(args.start_frame, end):
                if args.adaptive_prob and args.prob_reset == "frame":
                    adaptive = create_adaptive_models(args)
                else:
                    adaptive = seq_adapt

                depth_y = read_yuv420p10le_y_frame(depth_fp, fi, args.width, args.height)
                cur_video = read_yuv420p10le_frame(video_fp, fi, args.width, args.height)

                cam_cur = get_camera(camera_lookup, fi)

                if not camera_has_required_mats(cam_cur):
                    raise ValueError(f"camera frame {fi} does not have required matrices")

                plane_warp_ctx = None

                if (
                    args.plane_warp_candidate
                    and prev_store is not None
                    and prev_processed_frame_idx is not None
                ):
                    cam_prev_for_depth = get_camera(camera_lookup, prev_processed_frame_idx)

                    plane_warp_ctx = PlaneWarpContext(
                        prev_store=prev_store,
                        cam_prev=cam_prev_for_depth,
                        cam_cur=cam_cur,
                        frame_w=args.width,
                        frame_h=args.height,
                    )

                rec_depth_y, sm, cur_store = simulate_one_depth_frame(
                    depth_y,
                    fi,
                    args,
                    grid,
                    prev_store=prev_store,
                    writer=writer,
                    adaptive=adaptive,
                    plane_warp_ctx=plane_warp_ctx,
                )

                if depth_recon_fp:
                    write_depth_as_yuv420p10le(
                        depth_recon_fp,
                        rec_depth_y,
                        args.width,
                        args.height,
                        args.max_value,
                    )

                ref_idx = fi - args.ref_offset

                if ref_idx >= 0:
                    prev_video = read_yuv420p10le_frame(video_fp, ref_idx, args.width, args.height)
                    cam_prev_for_warp = get_camera(camera_lookup, ref_idx)

                    pred_video, warp_stats = backward_warp_prev_yuv_to_cur(
                        prev_video,
                        cur_video,
                        rec_depth_y,
                        cam_prev_for_warp,
                        cam_cur,
                        args,
                    )
                else:
                    pred_video = cur_video
                    warp_stats = {
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

                if pred_fp:
                    write_yuv420p10le_frame(
                        pred_fp,
                        pred_video[0],
                        pred_video[1],
                        pred_video[2],
                        args.max_value,
                    )

                sm.update(warp_stats)

                prev_store = cur_store
                prev_processed_frame_idx = fi
                summaries.append(sm)

                print(
                    f"Frame {fi:4d} | "
                    f"depthBits={sm['depth_bits']:.1f} | "
                    f"depthPSNR={sm['depth_psnr']:.3f} | "
                    f"warpYPSNR={sm['warp_y_psnr']:.3f} | "
                    f"valid={sm['warp_valid_y_ratio']:.3f} | "
                    f"leaf={sm['leaf_blocks']} | "
                    f"QT={sm['qt_nodes']} | "
                    f"BH/BV={sm['bin_h_nodes']}/{sm['bin_v_nodes']} | "
                    f"D/C/Î={sm['direct_ratio']:.3f}/"
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)

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
        "num_processed_frames": len(summaries),
        "total_depth_bits": total_depth_bits,
        "overall_depth_bpp": total_depth_bits / total_pixels,
        "average": avg,
        "frame_csv": args.out_csv,
        "depth_recon_yuv": args.out_depth_recon_yuv,
        "pred_yuv": args.out_pred_yuv,
        "block_csv": args.out_block_csv,
    }

    with open(args.out_json, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("Done.")
    print(f"Frame CSV       : {args.out_csv}")
    print(f"Summary         : {args.out_json}")
    print(f"Recon depth YUV : {args.out_depth_recon_yuv}")
    print(f"Backward pred   : {args.out_pred_yuv}")

    if args.out_block_csv:
        print(f"Block CSV       : {args.out_block_csv}")

    print(f"Overall depth bpp: {overall['overall_depth_bpp']:.6f}")


if __name__ == "__main__":
    main()
