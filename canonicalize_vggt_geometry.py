#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonicalize VGGT-Omega geometry before codec input.

Input:
  - VGGT-Omega NPZ produced by the user's runner:
      depth_original          : [N,H,W] float32
      extrinsic               : [N,3,4] camera_from_world [R|t]
      intrinsic_original      : [N,3,3]
      frame_indices optional  : [N]
  - Optional camera JSONL produced by the same runner, used only for metadata/validation.

Output:
  - canonical NPZ:
      depth_canonical         : [N,H,W] float32
      K_fixed                 : [3,3]
      rel_R_current_to_prev   : [N,3,3], poc0 identity
      rel_t_current_to_prev   : [N,3],   poc0 zero
      rvec_current_to_prev    : [N,3],   poc0 zero
      plus debug arrays
  - codec-friendly camera JSONL:
      header fixed intrinsic only, no intrinsic_delta
      frame lines with rvec/tvec in current_to_previous mode
  - depth YUV420p10le:
      linear depth in Y plane using fixed-point depth_scale
      U/V neutral 512
  - manifest JSON

Design:
  1. Pick one fixed K per RAP, by median fx/fy and fixed/median cx/cy.
  2. Convert original VGGT per-frame K into exact fixed-K affine pair transforms:
       H_i = inv(K_fixed) @ K_i
       A_i = H_ref @ R_rel @ inv(H_cur)
       b_i = H_ref @ t_rel
     This is exact if affine camera transforms are allowed.
  3. Project A_i to closest rotation by SVD.
  4. Smooth rotation and translation over the whole RAP with polynomial fitting.
  5. Given chosen K_fixed/R'/t', solve a per-pixel depth target that reproduces
     the exact affine/VGGT target rays as closely as possible.
  6. Fit final depth in a video-friendly way: none/direct/global/block/invblock.

Notes:
  - This is an encoder-side pre-preprocessing/canonicalization step. It can use
    all frames in a RAP at once. Decoder only sees the final camera/depth.
  - z_sign is default +1, matching OpenCV-style VGGT camera coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np


DepthFitMode = Literal["none", "target", "global", "block", "invblock"]
CenterMode = Literal["median", "image-center", "first"]
FitKind = Literal["none", "poly"]


# ============================================================
# Small utilities
# ============================================================

def sanitize_windows_filename_component(name: str, replacement: str = "_") -> str:
    """Sanitize one filename component for Windows. Does not sanitize full paths."""
    name = unicodedata.normalize("NFC", str(name))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, name)
    if replacement:
        name = re.sub(re.escape(replacement) + r"+", replacement, name)
    name = name.rstrip(" .")
    if not name:
        name = "unnamed"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.split(".")[0].upper() in reserved:
        name = "_" + name
    return name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def finite_positive_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def safe_percentile(x: np.ndarray, p: float, default: float = 1.0) -> float:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return default
    out = float(np.percentile(vals, p))
    if not np.isfinite(out):
        return default
    return out


def json_safe_float(x: Any) -> Any:
    if x is None:
        return None
    x = float(x)
    if math.isinf(x):
        return "inf"
    if math.isnan(x):
        return None
    return x


def as_float_list(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a).reshape(-1)]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)



# ============================================================
# Camera helpers
# ============================================================

def split_extrinsic(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape != (3, 4):
        raise ValueError(f"extrinsic must be [3,4], got {E.shape}")
    R = E[:, :3]
    t = E[:, 3]
    return R, t


def relative_current_to_previous(E_cur: np.ndarray, E_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input extrinsics are camera_from_world:
      X_cam = R_cw * X_world + t_cw

    Return relative transform current -> previous:
      X_prev = R_rel * X_cur + t_rel

    Derivation:
      X_world = R_cur.T * (X_cur - t_cur)
      X_prev  = R_prev * X_world + t_prev
              = R_prev * R_cur.T * X_cur + (t_prev - R_prev * R_cur.T * t_cur)
    """
    R_cur, t_cur = split_extrinsic(E_cur)
    R_prev, t_prev = split_extrinsic(E_prev)
    R_rel = R_prev @ R_cur.T
    t_rel = t_prev - R_rel @ t_cur
    return R_rel, t_rel


def make_fixed_intrinsic(Ks: np.ndarray, width: int, height: int, center_mode: CenterMode) -> np.ndarray:
    Ks = np.asarray(Ks, dtype=np.float64)
    fx0 = float(np.median(Ks[:, 0, 0]))
    fy0 = float(np.median(Ks[:, 1, 1]))

    if center_mode == "median":
        cx0 = float(np.median(Ks[:, 0, 2]))
        cy0 = float(np.median(Ks[:, 1, 2]))
    elif center_mode == "image-center":
        cx0 = float(width) / 2.0
        cy0 = float(height) / 2.0
    elif center_mode == "first":
        cx0 = float(Ks[0, 0, 2])
        cy0 = float(Ks[0, 1, 2])
    else:
        raise ValueError(center_mode)

    K0 = np.eye(3, dtype=np.float64)
    K0[0, 0] = fx0
    K0[1, 1] = fy0
    K0[0, 2] = cx0
    K0[1, 2] = cy0
    return K0


def closest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3).astype(np.float64)


def R_from_rvec(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def poly_smooth_sequence(values: np.ndarray, degree: int, keep_first: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Batch polynomial smoothing over frame index.
    values: [N,D]
    Uses normalized x in [-1,1] for conditioning.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    N, D = values.shape
    if degree < 0 or N <= 1:
        return values.copy(), {"kind": "none", "degree": None, "coefficients": None}

    deg = min(int(degree), max(N - 1, 0))
    if deg == 0:
        fit = np.repeat(np.mean(values, axis=0, keepdims=True), N, axis=0)
    else:
        x = np.linspace(-1.0, 1.0, N, dtype=np.float64)
        X = np.vander(x, N=deg + 1, increasing=True)  # [N,deg+1]
        coef, _, _, _ = np.linalg.lstsq(X, values, rcond=None)  # [deg+1,D]
        fit = X @ coef

    if keep_first:
        fit[0] = values[0]

    # Recompute coefficients for metadata if deg set.
    x = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    X = np.vander(x, N=deg + 1, increasing=True)
    coef, _, _, _ = np.linalg.lstsq(X, fit, rcond=None)

    return fit.astype(np.float64), {
        "kind": "poly",
        "degree": int(deg),
        "x_domain": "normalized_frame_index_-1_to_1",
        "coefficients_increasing_power": coef.astype(float).tolist(),
    }


# ============================================================
# Geometry grids and projection
# ============================================================

def make_rays(K: np.ndarray, width: int, height: int, z_sign: float = 1.0) -> np.ndarray:
    """Return [H,W,3] camera rays [x_norm, y_norm, z_sign]."""
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    rays = np.empty((height, width, 3), dtype=np.float64)
    rays[..., 0] = (xs - cx) / fx
    rays[..., 1] = (ys - cy) / fy
    rays[..., 2] = float(z_sign)
    return rays


def apply_affine_to_depth_rays(depth: np.ndarray, rays: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """X_ref = A * (depth * ray_cur) + b. Returns [H,W,3]."""
    X = depth[..., None].astype(np.float64) * rays
    Xr = np.einsum("ij,hwj->hwi", A, X) + b.reshape(1, 1, 3)
    return Xr


def normalized_ray_from_points(X: np.ndarray, z_sign: float = 1.0, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D camera points to normalized image rays [x/z, y/z, z_sign].
    Returns rays and validity mask.
    """
    Z = X[..., 2]
    valid = np.isfinite(X).all(axis=-1) & (Z * z_sign > eps)
    denom = np.where(np.abs(Z) > eps, Z, np.where(Z >= 0, eps, -eps))
    s = np.empty_like(X, dtype=np.float64)
    s[..., 0] = X[..., 0] / denom
    s[..., 1] = X[..., 1] / denom
    s[..., 2] = float(z_sign)
    return s, valid


def map_from_normalized_ray(s: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    map_x = fx * s[..., 0] + cx
    map_y = fy * s[..., 1] + cy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def solve_depth_for_target_ray(
    rays_cur: np.ndarray,
    target_rays_ref: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    valid_target: np.ndarray,
    z_min: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve z from:
      R * (z * r) + t = lambda * s

    Using:
      z * (s x Rr) = - (s x t)
      z = - dot(a,b) / dot(a,a)
    where a=s x Rr, b=s x t.
    """
    Rr = np.einsum("ij,hwj->hwi", R.astype(np.float64), rays_cur.astype(np.float64))
    s = target_rays_ref.astype(np.float64)
    t_b = np.asarray(t, dtype=np.float64).reshape(1, 1, 3)

    a = np.cross(s, Rr)
    b = np.cross(s, np.broadcast_to(t_b, s.shape))
    denom = np.sum(a * a, axis=-1)
    numer = np.sum(a * b, axis=-1)
    z = -numer / np.maximum(denom, 1e-30)

    valid = (
        valid_target
        & np.isfinite(z)
        & (z > z_min)
        & np.isfinite(denom)
        & (denom > 1e-20)
    )
    z = z.astype(np.float32)
    return z, valid


# ============================================================
# Depth fitting modes
# ============================================================

def fit_global_scale_bias(depth_src: np.ndarray, depth_tgt: np.ndarray, valid: np.ndarray, reg_identity: float) -> tuple[np.ndarray, dict[str, float]]:
    z = depth_src.astype(np.float64)
    y = depth_tgt.astype(np.float64)
    m = valid & np.isfinite(z) & np.isfinite(y) & (z > 0) & (y > 0)
    if np.count_nonzero(m) < 16:
        return depth_src.astype(np.float32), {"a": 1.0, "b": 0.0, "valid_count": int(np.count_nonzero(m))}

    X0 = z[m].reshape(-1)
    Y0 = y[m].reshape(-1)
    A = np.stack([X0, np.ones_like(X0)], axis=1)

    if reg_identity > 0:
        # Penalize a->1, b->0.
        A_reg = np.array([[math.sqrt(reg_identity), 0.0], [0.0, math.sqrt(reg_identity)]], dtype=np.float64)
        Y_reg = np.array([math.sqrt(reg_identity), 0.0], dtype=np.float64)
        A = np.vstack([A, A_reg])
        Y0 = np.concatenate([Y0, Y_reg])

    coef, _, _, _ = np.linalg.lstsq(A, Y0, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    out = a * z + b
    out = np.where(np.isfinite(out) & (out > 0), out, z)
    return out.astype(np.float32), {"a": a, "b": b, "valid_count": int(np.count_nonzero(m))}


def fit_block_scale_bias(
    depth_src: np.ndarray,
    depth_tgt: np.ndarray,
    valid: np.ndarray,
    block_size: int,
    reg_identity: float,
    inverse: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Fit y ~= a*x + b per block.
    If inverse=True, fit in inverse-depth domain and convert back.
    """
    h, w = depth_src.shape
    out = depth_src.astype(np.float64).copy()
    stats: list[dict[str, Any]] = []

    if inverse:
        src_domain = np.zeros_like(out)
        tgt_domain = np.zeros_like(out)
        src_valid = depth_src > 1e-12
        tgt_valid = depth_tgt > 1e-12
        src_domain[src_valid] = 1.0 / np.maximum(depth_src[src_valid], 1e-12)
        tgt_domain[tgt_valid] = 1.0 / np.maximum(depth_tgt[tgt_valid], 1e-12)
        base_valid = valid & src_valid & tgt_valid & np.isfinite(src_domain) & np.isfinite(tgt_domain)
    else:
        src_domain = depth_src.astype(np.float64)
        tgt_domain = depth_tgt.astype(np.float64)
        base_valid = valid & np.isfinite(src_domain) & np.isfinite(tgt_domain) & (src_domain > 0) & (tgt_domain > 0)

    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        for x0 in range(0, w, block_size):
            x1 = min(x0 + block_size, w)
            m = base_valid[y0:y1, x0:x1]
            cnt = int(np.count_nonzero(m))
            if cnt < 8:
                stats.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "a": 1.0, "b": 0.0, "valid_count": cnt})
                continue

            X0 = src_domain[y0:y1, x0:x1][m].reshape(-1)
            Y0 = tgt_domain[y0:y1, x0:x1][m].reshape(-1)
            A = np.stack([X0, np.ones_like(X0)], axis=1)

            if reg_identity > 0:
                A_reg = np.array([[math.sqrt(reg_identity), 0.0], [0.0, math.sqrt(reg_identity)]], dtype=np.float64)
                Y_reg = np.array([math.sqrt(reg_identity), 0.0], dtype=np.float64)
                A = np.vstack([A, A_reg])
                Y0 = np.concatenate([Y0, Y_reg])

            coef, _, _, _ = np.linalg.lstsq(A, Y0, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            pred_domain = a * src_domain[y0:y1, x0:x1] + b

            if inverse:
                pred = np.where(pred_domain > 1e-12, 1.0 / np.maximum(pred_domain, 1e-12), out[y0:y1, x0:x1])
            else:
                pred = pred_domain

            old = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = np.where(np.isfinite(pred) & (pred > 0), pred, old)
            stats.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "a": a, "b": b, "valid_count": cnt})

    return out.astype(np.float32), stats


# ============================================================
# Depth YUV writer
# ============================================================

def choose_depth_scale_fixed_point(depth: np.ndarray, percentile: float, precision: int, bit_depth: int) -> dict[str, Any]:
    max_code = (1 << bit_depth) - 1
    m = finite_positive_mask(depth)
    if not np.any(m):
        scale_real = 1.0 / max_code
    else:
        ref = safe_percentile(depth[m], percentile, default=float(np.max(depth[m])))
        ref = max(ref, 1e-12)
        scale_real = ref / float(max_code)
    scale_int = max(1, int(round(scale_real * precision)))
    scale_real_q = scale_int / float(precision)
    return {
        "depth_scale": int(scale_int),
        "depth_scale_precision": int(precision),
        "depth_scale_real": float(scale_real_q),
        "depth_scale_percentile": float(percentile),
        "depth_bit_depth": int(bit_depth),
        "max_code": int(max_code),
    }


def write_depth_yuv420p10le_linear(path: Path, depth: np.ndarray, scale_meta: dict[str, Any]) -> dict[str, Any]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"depth must be [N,H,W], got {depth.shape}")
    n, h, w = depth.shape
    if w % 2 or h % 2:
        raise ValueError("YUV420 output requires even width and height")
    max_code = int(scale_meta["max_code"])
    scale = float(scale_meta["depth_scale_real"])
    neutral = np.uint16(512)
    clipped_total = 0
    ensure_parent(path)
    with open(path, "wb") as f:
        for i in range(n):
            y = np.round(depth[i].astype(np.float64) / scale)
            clipped = (y < 0) | (y > max_code) | ~np.isfinite(y)
            clipped_total += int(np.count_nonzero(clipped))
            y = np.nan_to_num(y, nan=0.0, posinf=max_code, neginf=0.0)
            y = np.clip(y, 0, max_code).astype("<u2")
            uv = np.full((h // 2, w // 2), neutral, dtype="<u2")
            f.write(y.tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())
    return {
        **scale_meta,
        "depth_yuv": str(path),
        "depth_yuv_format": "yuv420p10le",
        "depth_yuv_semantics": "Y stores linear depth code = round(depth / depth_scale_real); U/V are neutral 512",
        "clipped_samples_total": int(clipped_total),
        "total_samples": int(n * h * w),
    }


# ============================================================
# Bit estimation helpers
# ============================================================

def signed_to_code_num(x: int) -> int:
    if x == 0:
        return 0
    if x > 0:
        return 2 * x - 1
    return -2 * x


def ue_exp_golomb_bits(code_num: int) -> int:
    if code_num < 0:
        raise ValueError(code_num)
    k = (code_num + 1).bit_length() - 1
    return 2 * k + 1


def signed_exp_golomb_bits_for_q(q: np.ndarray) -> tuple[np.ndarray, int]:
    q = np.asarray(q, dtype=np.int64)
    bits = np.zeros_like(q, dtype=np.int64)
    it = np.nditer(q, flags=["multi_index"])
    for val in it:
        bits[it.multi_index] = ue_exp_golomb_bits(signed_to_code_num(int(val)))
    return bits, int(np.sum(bits))


def estimate_param_bits_zero_predictor(rvecs: np.ndarray, tvecs: np.ndarray, r_step: float, t_step: float) -> dict[str, Any]:
    """Rough signed Exp-Golomb estimate for poc>=1 with zero predictor."""
    if len(rvecs) <= 1:
        return {}
    q_r = np.round(rvecs[1:] / r_step).astype(np.int64)
    q_t = np.round(tvecs[1:] / t_step).astype(np.int64)
    b_r, sum_r = signed_exp_golomb_bits_for_q(q_r)
    b_t, sum_t = signed_exp_golomb_bits_for_q(q_t)
    n = len(rvecs) - 1
    return {
        "estimator": "signed_exp_golomb_zero_predictor_rough",
        "r_step": float(r_step),
        "t_step": float(t_step),
        "coded_frames": int(n),
        "rotation_total_bits": int(sum_r),
        "translation_total_bits": int(sum_t),
        "rotation_avg_bits_frame": float(sum_r / max(n, 1)),
        "translation_avg_bits_frame": float(sum_t / max(n, 1)),
        "rotation_avg_bits_each": np.mean(b_r, axis=0).astype(float).tolist(),
        "translation_avg_bits_each": np.mean(b_t, axis=0).astype(float).tolist(),
    }


# ============================================================
# Main canonicalization
# ============================================================

@dataclass
class CanonicalizationConfig:
    fixed_center_mode: str
    rot_fit_degree: int
    trans_fit_degree: int
    depth_fit_mode: str
    depth_block_size: int
    depth_fit_reg_identity: float
    z_sign: float
    depth_scale_precision: int
    depth_scale_percentile: float
    bit_est_r_step: float
    bit_est_t_step: float
    progress_every: int
    save_debug_arrays: bool
    compressed_npz: bool
    epe_stride: int


def load_camera_jsonl_optional(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def canonicalize(
    npz_path: Path,
    camera_jsonl_path: Path | None,
    out_prefix: Path,
    width: int | None,
    height: int | None,
    cfg: CanonicalizationConfig,
    overwrite: bool,
) -> None:
    log(f"Start canonicalization: {npz_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    log("Loading NPZ metadata/arrays...")
    data = np.load(npz_path, allow_pickle=True)
    required = ["depth_original", "extrinsic", "intrinsic_original"]
    for k in required:
        if k not in data:
            raise KeyError(f"NPZ missing required key: {k}")

    depth = data["depth_original"].astype(np.float32)
    E_abs = data["extrinsic"].astype(np.float64)
    K_orig = data["intrinsic_original"].astype(np.float64)
    log(f"Loaded arrays: depth={depth.shape}, extrinsic={E_abs.shape}, intrinsic={K_orig.shape}")

    if depth.ndim != 3:
        raise ValueError(f"depth_original must be [N,H,W], got {depth.shape}")
    n, h, w = depth.shape
    if E_abs.shape[:2] != (n, 3) or E_abs.shape[2] != 4:
        raise ValueError(f"extrinsic shape mismatch: {E_abs.shape}, N={n}")
    if K_orig.shape != (n, 3, 3):
        raise ValueError(f"intrinsic_original shape mismatch: {K_orig.shape}, N={n}")

    if width is not None and width != w:
        raise ValueError(f"--width {width} does not match depth width {w}")
    if height is not None and height != h:
        raise ValueError(f"--height {height} does not match depth height {h}")

    frame_indices = data["frame_indices"].astype(np.int32) if "frame_indices" in data else np.arange(n, dtype=np.int32)
    log("Loading optional camera JSONL...")
    camera_records = load_camera_jsonl_optional(camera_jsonl_path)

    base = sanitize_windows_filename_component(out_prefix.name)
    out_prefix = out_prefix.with_name(base)

    out_npz = out_prefix.with_name(out_prefix.name + "_canonical_geometry.npz")
    out_jsonl = out_prefix.with_name(out_prefix.name + "_camParam_canonical.jsonl")
    out_depth_yuv = out_prefix.with_name(out_prefix.name + "_depth_canonical_linear_yuv420p10le.yuv")
    out_manifest = out_prefix.with_name(out_prefix.name + "_canonical_manifest.json")

    for p in [out_npz, out_jsonl, out_depth_yuv, out_manifest]:
        if p.exists():
            if overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --overwrite.")
        ensure_parent(p)

    log("Building fixed intrinsic and exact fixed-K affine transforms...")
    K_fixed = make_fixed_intrinsic(K_orig, w, h, center_mode=cfg.fixed_center_mode)  # type: ignore[arg-type]
    inv_K_fixed = np.linalg.inv(K_fixed)
    H = np.stack([inv_K_fixed @ K_orig[i] for i in range(n)], axis=0)  # raw-K cam coord -> fixed-K cam coord
    H_inv = np.stack([np.linalg.inv(H[i]) for i in range(n)], axis=0)

    # Pair transforms poc i -> i-1. poc0 is identity/zero.
    R_rel = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    t_rel = np.zeros((n, 3), dtype=np.float64)
    A_exact = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    b_exact = np.zeros((n, 3), dtype=np.float64)
    R0 = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)

    for i in range(1, n):
        if cfg.progress_every > 0 and (i == 1 or i == n - 1 or i % cfg.progress_every == 0):
            log(f"Affine transform stage: pair {i}/{n-1}")
        R_i, t_i = relative_current_to_previous(E_abs[i], E_abs[i - 1])
        R_rel[i] = R_i
        t_rel[i] = t_i
        A_exact[i] = H[i - 1] @ R_i @ H_inv[i]
        b_exact[i] = H[i - 1] @ t_i
        R0[i] = closest_rotation(A_exact[i])

    log("Projecting exact affine matrices to closest rotations...")
    rvec0 = np.stack([rvec_from_R(R0[i]) for i in range(n)], axis=0)
    t0 = b_exact.copy()

    # Smooth/fit over the whole RAP. Keep poc0 identity/zero.
    log(f"Smoothing camera trajectory: rot_degree={cfg.rot_fit_degree}, trans_degree={cfg.trans_fit_degree}")
    rvec_fit, rot_fit_meta = poly_smooth_sequence(rvec0, degree=cfg.rot_fit_degree, keep_first=True)
    t_fit, trans_fit_meta = poly_smooth_sequence(t0, degree=cfg.trans_fit_degree, keep_first=True)
    rvec_fit[0] = 0.0
    t_fit[0] = 0.0
    R_fit = np.stack([R_from_rvec(rvec_fit[i]) for i in range(n)], axis=0)

    log("Creating fixed-K ray grid...")
    rays = make_rays(K_fixed, w, h, z_sign=cfg.z_sign)

    depth_target = depth.copy().astype(np.float32)
    depth_target_valid = np.ones((n, h, w), dtype=bool)
    depth_canon = depth.copy().astype(np.float32)
    depth_fit_stats: list[Any] = [{"poc": 0, "mode": "copy_first"}]

    log(f"Depth target + fitting stage started: mode={cfg.depth_fit_mode}, frames={n-1}")
    # For each pair, generate exact target rays using A_exact/b_exact, then solve depth for chosen R_fit/t_fit.
    for i in range(1, n):
        if cfg.progress_every > 0 and (i == 1 or i == n - 1 or i % cfg.progress_every == 0):
            log(f"Depth target/fitting: frame {i}/{n-1}, mode={cfg.depth_fit_mode}")
        X_ref_exact = apply_affine_to_depth_rays(depth[i], rays, A_exact[i], b_exact[i])
        s_ref, valid_ref = normalized_ray_from_points(X_ref_exact, z_sign=cfg.z_sign)
        z_tgt, valid_z = solve_depth_for_target_ray(
            rays_cur=rays,
            target_rays_ref=s_ref,
            R=R_fit[i],
            t=t_fit[i],
            valid_target=valid_ref,
        )
        depth_target[i] = z_tgt
        depth_target_valid[i] = valid_z

        if cfg.depth_fit_mode == "none":
            depth_canon[i] = depth[i]
            depth_fit_stats.append({"poc": int(i), "mode": "none", "valid_count": int(np.count_nonzero(valid_z))})
        elif cfg.depth_fit_mode == "target":
            # Upper bound; may be less video-friendly.
            out = np.where(valid_z, z_tgt, depth[i])
            depth_canon[i] = out.astype(np.float32)
            depth_fit_stats.append({"poc": int(i), "mode": "target", "valid_count": int(np.count_nonzero(valid_z))})
        elif cfg.depth_fit_mode == "global":
            out, stat = fit_global_scale_bias(depth[i], z_tgt, valid_z, cfg.depth_fit_reg_identity)
            depth_canon[i] = out
            depth_fit_stats.append({"poc": int(i), "mode": "global", **stat})
        elif cfg.depth_fit_mode == "block":
            out, stats = fit_block_scale_bias(
                depth[i], z_tgt, valid_z,
                block_size=cfg.depth_block_size,
                reg_identity=cfg.depth_fit_reg_identity,
                inverse=False,
            )
            depth_canon[i] = out
            valid_counts = [s["valid_count"] for s in stats]
            depth_fit_stats.append({
                "poc": int(i),
                "mode": "block",
                "block_size": int(cfg.depth_block_size),
                "num_blocks": int(len(stats)),
                "mean_valid_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
            })
        elif cfg.depth_fit_mode == "invblock":
            out, stats = fit_block_scale_bias(
                depth[i], z_tgt, valid_z,
                block_size=cfg.depth_block_size,
                reg_identity=cfg.depth_fit_reg_identity,
                inverse=True,
            )
            depth_canon[i] = out
            valid_counts = [s["valid_count"] for s in stats]
            depth_fit_stats.append({
                "poc": int(i),
                "mode": "invblock",
                "block_size": int(cfg.depth_block_size),
                "num_blocks": int(len(stats)),
                "mean_valid_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
            })
        else:
            raise ValueError(cfg.depth_fit_mode)

    # Compute simple map EPE against exact affine target for reporting.
    log(f"EPE reporting stage started: stride={cfg.epe_stride}")
    epe_stats = []
    epe_stride = max(1, int(cfg.epe_stride))
    rays_epe = rays[::epe_stride, ::epe_stride]
    for i in range(1, n):
        if cfg.progress_every > 0 and (i == 1 or i == n - 1 or i % cfg.progress_every == 0):
            log(f"EPE report: frame {i}/{n-1}")
        depth_i_epe = depth[i][::epe_stride, ::epe_stride]
        depth_canon_i_epe = depth_canon[i][::epe_stride, ::epe_stride]
        X_ref_exact = apply_affine_to_depth_rays(depth_i_epe, rays_epe, A_exact[i], b_exact[i])
        s_exact, valid_exact = normalized_ray_from_points(X_ref_exact, z_sign=cfg.z_sign)
        mx_exact, my_exact = map_from_normalized_ray(s_exact, K_fixed)

        X_ref_new = apply_affine_to_depth_rays(depth_canon_i_epe, rays_epe, R_fit[i], t_fit[i])
        s_new, valid_new = normalized_ray_from_points(X_ref_new, z_sign=cfg.z_sign)
        mx_new, my_new = map_from_normalized_ray(s_new, K_fixed)
        valid = valid_exact & valid_new & np.isfinite(mx_exact) & np.isfinite(mx_new) & np.isfinite(my_exact) & np.isfinite(my_new)
        valid &= (mx_exact >= 0) & (mx_exact <= w - 1) & (my_exact >= 0) & (my_exact <= h - 1)
        if np.count_nonzero(valid) == 0:
            epe_stats.append({"poc": int(i), "valid_count": 0, "mean_epe": None, "p95_epe": None})
            continue
        epe = np.sqrt((mx_new[valid] - mx_exact[valid]) ** 2 + (my_new[valid] - my_exact[valid]) ** 2)
        epe_stats.append({
            "poc": int(i),
            "valid_count": int(np.count_nonzero(valid)),
            "mean_epe": float(np.mean(epe)),
            "p50_epe": float(np.percentile(epe, 50)),
            "p95_epe": float(np.percentile(epe, 95)),
        })

    log("Choosing depth scale and writing canonical depth YUV...")
    scale_meta = choose_depth_scale_fixed_point(
        depth_canon,
        percentile=cfg.depth_scale_percentile,
        precision=cfg.depth_scale_precision,
        bit_depth=10,
    )
    depth_yuv_meta = write_depth_yuv420p10le_linear(out_depth_yuv, depth_canon, scale_meta)

    log("Estimating rough camera parameter bits...")
    bit_est = estimate_param_bits_zero_predictor(
        rvec_fit,
        t_fit / float(depth_yuv_meta["depth_scale_real"]),
        r_step=cfg.bit_est_r_step,
        t_step=cfg.bit_est_t_step,
    )

    log("Writing canonical camera JSONL...")
    # Save camera JSONL.
    with open(out_jsonl, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": "canonical_camparam_v1",
            "source_npz": os.path.abspath(npz_path),
            "source_camera_jsonl": os.path.abspath(camera_jsonl_path) if camera_jsonl_path else None,
            "frame_count": int(n),
            "frame_indices": frame_indices.astype(int).tolist(),
            "source_size": {"width": int(w), "height": int(h)},
            "pose_mode": "current_to_previous",
            "camera_convention": {
                "coordinate": "OpenCV camera coordinates",
                "transform": "X_prev = R * X_cur + t",
                "poc0": "identity rotation and zero translation",
            },
            "intrinsic_mode": "rap_fixed",
            "intrinsic": {
                "fx": float(K_fixed[0, 0]),
                "fy": float(K_fixed[1, 1]),
                "cx": float(K_fixed[0, 2]),
                "cy": float(K_fixed[1, 2]),
                "z_sign": float(cfg.z_sign),
            },
            "intrinsic_delta_order": [],
            "intrinsic_delta_bits_per_frame": 0,
            "depth_output": depth_yuv_meta,
            "canonicalization": {
                "summary": "Fixed intrinsic + smooth rigid current_to_previous pose + depth refitting before codec input",
                "fixed_center_mode": cfg.fixed_center_mode,
                "exact_fixedK_affine": "H_i=inv(K_fixed)@K_i; A_i=H_prev@R_rel@inv(H_i); b_i=H_prev@t_rel",
                "rotation_projection": "closest_rotation_by_svd(A_i)",
                "rotation_fit": rot_fit_meta,
                "translation_fit": trans_fit_meta,
                "depth_fit_mode": cfg.depth_fit_mode,
                "depth_block_size": cfg.depth_block_size,
                "depth_fit_reg_identity": cfg.depth_fit_reg_identity,
            },
            "rough_bit_estimate": bit_est,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in range(n):
            rec = {
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec": as_float_list(rvec_fit[i]),
                "tvec": as_float_list(t_fit[i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save NPZ.
    log(f"Saving canonical NPZ... compressed={cfg.compressed_npz}, debug_arrays={cfg.save_debug_arrays}")
    npz_payload = dict(
        frame_indices=frame_indices.astype(np.int32),
        depth_canonical=depth_canon.astype(np.float32),
        K_original=K_orig.astype(np.float32),
        K_fixed=K_fixed.astype(np.float32),
        extrinsic_original=E_abs.astype(np.float32),
        rel_R_original_current_to_previous=R_rel.astype(np.float32),
        rel_t_original_current_to_previous=t_rel.astype(np.float32),
        fixedK_affine_A_exact=A_exact.astype(np.float32),
        fixedK_affine_b_exact=b_exact.astype(np.float32),
        rel_R_canonical_current_to_previous=R_fit.astype(np.float32),
        rel_t_canonical_current_to_previous=t_fit.astype(np.float32),
        rvec_canonical_current_to_previous=rvec_fit.astype(np.float32),
        config_json=np.asarray(json.dumps(asdict(cfg), ensure_ascii=False), dtype=object),
        depth_fit_stats_json=np.asarray(json.dumps(depth_fit_stats, ensure_ascii=False), dtype=object),
        epe_stats_json=np.asarray(json.dumps(epe_stats, ensure_ascii=False), dtype=object),
    )
    if cfg.save_debug_arrays:
        npz_payload.update(
            depth_original=depth.astype(np.float32),
            depth_target=depth_target.astype(np.float32),
            depth_target_valid=depth_target_valid.astype(np.bool_),
        )
    if cfg.compressed_npz:
        np.savez_compressed(out_npz, **npz_payload)
    else:
        np.savez(out_npz, **npz_payload)
    log("Canonical NPZ saved.")

    avg_epe = [x["mean_epe"] for x in epe_stats if x.get("mean_epe") is not None]
    p95_epe = [x["p95_epe"] for x in epe_stats if x.get("p95_epe") is not None]
    manifest = {
        "source_npz": os.path.abspath(npz_path),
        "source_camera_jsonl": os.path.abspath(camera_jsonl_path) if camera_jsonl_path else None,
        "outputs": {
            "canonical_npz": os.path.abspath(out_npz),
            "canonical_camera_jsonl": os.path.abspath(out_jsonl),
            "canonical_depth_yuv": os.path.abspath(out_depth_yuv),
        },
        "frame_count": int(n),
        "size": {"width": int(w), "height": int(h)},
        "K_fixed": K_fixed.astype(float).tolist(),
        "depth_yuv": depth_yuv_meta,
        "config": asdict(cfg),
        "epe_summary_vs_exact_vggt_affine_target": {
            "mean_of_mean_epe": float(np.mean(avg_epe)) if avg_epe else None,
            "mean_of_p95_epe": float(np.mean(p95_epe)) if p95_epe else None,
            "per_pair": epe_stats,
        },
        "rough_bit_estimate": bit_est,
        "notes": [
            "This is pre-codec canonicalization, not a decoder-side causal predictor.",
            "Intrinsic is fixed for the RAP. No per-frame dfx/dfy is written.",
            "Depth is rewritten so that simplified camera can approximate the original VGGT projection behavior.",
            "Final RD gain must be evaluated after encoding canonical depth with the target depth/video codec.",
        ],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print("Canonicalization done")
    print("============================================================")
    print(f"input npz             : {npz_path}")
    print(f"frames                : {n}")
    print(f"size                  : {w}x{h}")
    print(f"fixed K               : fx={K_fixed[0,0]:.6f}, fy={K_fixed[1,1]:.6f}, cx={K_fixed[0,2]:.6f}, cy={K_fixed[1,2]:.6f}")
    print(f"depth fit mode        : {cfg.depth_fit_mode}")
    print(f"rot fit degree        : {cfg.rot_fit_degree}")
    print(f"trans fit degree      : {cfg.trans_fit_degree}")
    if avg_epe:
        print(f"mean EPE vs target    : {np.mean(avg_epe):.6f} px")
        print(f"mean p95 EPE          : {np.mean(p95_epe):.6f} px")
    print("------------------------------------------------------------")
    print(f"canonical npz         : {out_npz}")
    print(f"camera jsonl          : {out_jsonl}")
    print(f"depth yuv             : {out_depth_yuv}")
    print(f"manifest              : {out_manifest}")
    print("============================================================")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonicalize VGGT-Omega geometry into codec-friendly fixed-K/smooth-pose/adjusted-depth representation."
    )
    p.add_argument("--npz", required=True, help="VGGT-Omega output NPZ")
    p.add_argument("--camera-jsonl", default=None, help="Optional VGGT-Omega camera JSONL for metadata/validation")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--width", type=int, default=None, help="Optional validation width")
    p.add_argument("--height", type=int, default=None, help="Optional validation height")

    p.add_argument("--fixed-center-mode", choices=["median", "image-center", "first"], default="image-center")
    p.add_argument("--z-sign", type=float, default=1.0)

    p.add_argument("--rot-fit-degree", type=int, default=1, help="Polynomial degree over whole RAP for rvec. Use -1 for no smoothing.")
    p.add_argument("--trans-fit-degree", type=int, default=3, help="Polynomial degree over whole RAP for tvec. Use -1 for no smoothing.")

    p.add_argument(
        "--depth-fit-mode",
        choices=["none", "target", "global", "block", "invblock"],
        default="invblock",
        help=(
            "How to rewrite depth for the simplified camera. "
            "none keeps original depth; target uses closed-form z_target directly; "
            "global fits z'=a*z+b per frame; block fits per-block linear depth; "
            "invblock fits per-block inverse-depth."
        ),
    )
    p.add_argument("--depth-block-size", type=int, default=64)
    p.add_argument("--depth-fit-reg-identity", type=float, default=1e-3,
                   help="Regularization toward a=1,b=0 for global/block depth fitting.")

    p.add_argument("--depth-scale-precision", type=int, default=100000)
    p.add_argument("--depth-scale-percentile", type=float, default=99.9)

    p.add_argument("--bit-est-r-step", type=float, default=2.0 ** -12)
    p.add_argument("--bit-est-t-step", type=float, default=2.0 ** -10)

    p.add_argument("--progress-every", type=int, default=1,
                   help="Print progress every N frame pairs. Use 0 to disable per-frame progress.")
    p.add_argument("--epe-stride", type=int, default=4,
                   help="Downsample stride for EPE reporting. 1=full resolution, 4 is much faster.")
    p.add_argument("--save-debug-arrays", action="store_true",
                   help="Also save depth_original/depth_target/depth_target_valid in NPZ. Large and slow; off by default.")
    p.add_argument("--compressed-npz", action="store_true",
                   help="Use np.savez_compressed. Smaller but can be much slower; off by default.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CanonicalizationConfig(
        fixed_center_mode=args.fixed_center_mode,
        rot_fit_degree=int(args.rot_fit_degree),
        trans_fit_degree=int(args.trans_fit_degree),
        depth_fit_mode=args.depth_fit_mode,
        depth_block_size=int(args.depth_block_size),
        depth_fit_reg_identity=float(args.depth_fit_reg_identity),
        z_sign=float(args.z_sign),
        depth_scale_precision=int(args.depth_scale_precision),
        depth_scale_percentile=float(args.depth_scale_percentile),
        bit_est_r_step=float(args.bit_est_r_step),
        bit_est_t_step=float(args.bit_est_t_step),
        progress_every=int(args.progress_every),
        save_debug_arrays=bool(args.save_debug_arrays),
        compressed_npz=bool(args.compressed_npz),
        epe_stride=int(args.epe_stride),
    )
    canonicalize(
        npz_path=Path(args.npz),
        camera_jsonl_path=Path(args.camera_jsonl) if args.camera_jsonl else None,
        out_prefix=Path(args.out_prefix),
        width=args.width,
        height=args.height,
        cfg=cfg,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
