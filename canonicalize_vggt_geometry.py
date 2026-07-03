#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize fixed-K current-to-previous R/t for VGGT-Omega outputs.

Purpose
-------
This is a minimal pre-codec geometry canonicalization experiment.
It removes per-frame intrinsic signaling by fixing one K per RAP, then optimizes
only a rigid current-to-previous R/t for each frame pair so that the fixed-K
projection with the original depth matches the original VGGT projection as well
as possible.

Unlike the larger canonicalization script, this version intentionally removes:
  - camera trajectory smoothing
  - polynomial fitting
  - depth target solving
  - depth block/global refitting

Inputs
------
  NPZ from the VGGT-Omega YUV runner, containing:
    depth_original     : [N,H,W] float32
    extrinsic          : [N,3,4] camera_from_world [R|t]
    intrinsic_original : [N,3,3]
    frame_indices      : optional [N]

Outputs
-------
  <out-prefix>_fixedK_rtopt.npz
  <out-prefix>_camParam_fixedK_rtopt.jsonl
  <out-prefix>_fixedK_rtopt_manifest.json

Optional:
  <out-prefix>_depth_original_linear_yuv420p10le.yuv

Conventions
-----------
  Original extrinsic: X_cam = R_cw * X_world + t_cw
  Output pose mode  : current_to_previous, X_prev = R * X_cur + t
  z_sign            : +1 by default, OpenCV/VGGT-style camera coordinates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception as exc:  # pragma: no cover
    least_squares = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


CenterMode = Literal["median", "image-center", "first"]


# ============================================================
# Utilities
# ============================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sanitize_windows_filename_component(name: str, replacement: str = "_") -> str:
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


# ============================================================
# Camera helpers
# ============================================================

def split_extrinsic(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape != (3, 4):
        raise ValueError(f"extrinsic must be [3,4], got {E.shape}")
    return E[:, :3], E[:, 3]


def relative_current_to_previous(E_cur: np.ndarray, E_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input extrinsics are camera_from_world:
      X_cam = R_cw * X_world + t_cw

    Return relative transform current -> previous:
      X_prev = R_rel * X_cur + t_rel
    """
    R_cur, t_cur = split_extrinsic(E_cur)
    R_prev, t_prev = split_extrinsic(E_prev)
    R_rel = R_prev @ R_cur.T
    t_rel = t_prev - R_rel @ t_cur
    return R_rel.astype(np.float64), t_rel.astype(np.float64)


def make_fixed_intrinsic(Ks: np.ndarray, width: int, height: int, center_mode: CenterMode) -> np.ndarray:
    Ks = np.asarray(Ks, dtype=np.float64)
    K0 = np.eye(3, dtype=np.float64)
    K0[0, 0] = float(np.median(Ks[:, 0, 0]))
    K0[1, 1] = float(np.median(Ks[:, 1, 1]))

    if center_mode == "median":
        K0[0, 2] = float(np.median(Ks[:, 0, 2]))
        K0[1, 2] = float(np.median(Ks[:, 1, 2]))
    elif center_mode == "image-center":
        K0[0, 2] = float(width) / 2.0
        K0[1, 2] = float(height) / 2.0
    elif center_mode == "first":
        K0[0, 2] = float(Ks[0, 0, 2])
        K0[1, 2] = float(Ks[0, 1, 2])
    else:
        raise ValueError(center_mode)
    return K0


def closest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R.astype(np.float64)


def rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3).astype(np.float64)


def R_from_rvec(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


# ============================================================
# Projection / EPE
# ============================================================

def make_rays(K: np.ndarray, width: int, height: int, z_sign: float = 1.0) -> np.ndarray:
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


def sample_grid(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, height, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, width, max(1, int(stride)), dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def apply_transform_points(depth_flat: np.ndarray, rays_flat: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    X = depth_flat[:, None].astype(np.float64) * rays_flat.astype(np.float64)
    return X @ A.T + b.reshape(1, 3)


def project_points_to_map(X: np.ndarray, K: np.ndarray, z_sign: float, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = X[:, 2]
    valid = np.isfinite(X).all(axis=1) & (Z * z_sign > eps)
    denom = np.where(np.abs(Z) > eps, Z, np.where(Z >= 0, eps, -eps))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    mx = fx * (X[:, 0] / denom) + cx
    my = fy * (X[:, 1] / denom) + cy
    return mx.astype(np.float64), my.astype(np.float64), valid


def make_pair_samples(
    depth: np.ndarray,
    rays: np.ndarray,
    A_exact: np.ndarray,
    b_exact: np.ndarray,
    K_fixed: np.ndarray,
    z_sign: float,
    stride: int,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    h, w = depth.shape
    yy, xx = sample_grid(h, w, stride)
    depth_flat = depth[yy, xx].astype(np.float64)
    rays_flat = rays[yy, xx].astype(np.float64)

    X_exact = apply_transform_points(depth_flat, rays_flat, A_exact, b_exact)
    target_x, target_y, valid = project_points_to_map(X_exact, K_fixed, z_sign=z_sign)
    valid &= finite_positive_mask(depth_flat)
    valid &= np.isfinite(target_x) & np.isfinite(target_y)
    valid &= (target_x >= 0.0) & (target_x <= w - 1) & (target_y >= 0.0) & (target_y <= h - 1)

    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise RuntimeError("No valid projection samples for this pair")
    if max_samples > 0 and idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()

    return {
        "depth": depth_flat[idx],
        "rays": rays_flat[idx],
        "target_x": target_x[idx],
        "target_y": target_y[idx],
    }


def residual_fixedK_rt(params: np.ndarray, samples: dict[str, np.ndarray], K_fixed: np.ndarray, z_sign: float, residual_scale: float) -> np.ndarray:
    rvec = params[:3]
    t = params[3:6]
    R = R_from_rvec(rvec)
    X = apply_transform_points(samples["depth"], samples["rays"], R, t)
    mx, my, valid = project_points_to_map(X, K_fixed, z_sign=z_sign)

    # Invalid samples get a bounded but strong penalty. Normally the robust loss
    # prevents a few bad samples from dominating.
    dx = mx - samples["target_x"]
    dy = my - samples["target_y"]
    bad = ~valid | ~np.isfinite(dx) | ~np.isfinite(dy)
    if np.any(bad):
        dx = dx.copy()
        dy = dy.copy()
        dx[bad] = 1000.0
        dy[bad] = 1000.0
    return np.concatenate([dx, dy], axis=0) / float(residual_scale)


def optimize_pair_rt(
    samples: dict[str, np.ndarray],
    K_fixed: np.ndarray,
    rvec_init: np.ndarray,
    t_init: np.ndarray,
    z_sign: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    residual_scale: float,
) -> dict[str, Any]:
    if least_squares is None:
        raise ImportError(
            "scipy is required for R/t optimization. Install with: pip install scipy"
        ) from SCIPY_IMPORT_ERROR

    x0 = np.concatenate([np.asarray(rvec_init, dtype=np.float64).reshape(3), np.asarray(t_init, dtype=np.float64).reshape(3)])
    res0 = residual_fixedK_rt(x0, samples, K_fixed, z_sign, residual_scale=residual_scale)
    mean0 = float(np.mean(np.sqrt(res0[: len(res0)//2] ** 2 + res0[len(res0)//2:] ** 2)) * residual_scale)

    opt = least_squares(
        residual_fixedK_rt,
        x0,
        args=(samples, K_fixed, z_sign, residual_scale),
        method="trf",
        loss=loss,
        f_scale=float(f_scale) / float(residual_scale),
        x_scale="jac",
        max_nfev=int(max_nfev),
        verbose=0,
    )

    res1 = residual_fixedK_rt(opt.x, samples, K_fixed, z_sign, residual_scale=residual_scale)
    mean1 = float(np.mean(np.sqrt(res1[: len(res1)//2] ** 2 + res1[len(res1)//2:] ** 2)) * residual_scale)

    return {
        "rvec": opt.x[:3].astype(np.float64),
        "R": R_from_rvec(opt.x[:3]),
        "t": opt.x[3:6].astype(np.float64),
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "nfev": int(opt.nfev),
        "cost": float(opt.cost),
        "sample_mean_epe_init": mean0,
        "sample_mean_epe_opt": mean1,
    }


def evaluate_epe_pair(
    depth: np.ndarray,
    rays: np.ndarray,
    A_exact: np.ndarray,
    b_exact: np.ndarray,
    R_new: np.ndarray,
    t_new: np.ndarray,
    K_fixed: np.ndarray,
    z_sign: float,
    stride: int,
) -> dict[str, Any]:
    h, w = depth.shape
    yy, xx = sample_grid(h, w, stride)
    depth_flat = depth[yy, xx].astype(np.float64)
    rays_flat = rays[yy, xx].astype(np.float64)

    X_exact = apply_transform_points(depth_flat, rays_flat, A_exact, b_exact)
    mx_exact, my_exact, valid_exact = project_points_to_map(X_exact, K_fixed, z_sign=z_sign)

    X_new = apply_transform_points(depth_flat, rays_flat, R_new, t_new)
    mx_new, my_new, valid_new = project_points_to_map(X_new, K_fixed, z_sign=z_sign)

    valid = valid_exact & valid_new & finite_positive_mask(depth_flat)
    valid &= np.isfinite(mx_exact) & np.isfinite(my_exact) & np.isfinite(mx_new) & np.isfinite(my_new)
    valid &= (mx_exact >= 0.0) & (mx_exact <= w - 1) & (my_exact >= 0.0) & (my_exact <= h - 1)
    valid &= (mx_new >= 0.0) & (mx_new <= w - 1) & (my_new >= 0.0) & (my_new <= h - 1)
    cnt = int(np.count_nonzero(valid))
    if cnt == 0:
        return {"valid_count": 0, "mean_epe": None, "p50_epe": None, "p95_epe": None, "p99_epe": None}
    epe = np.sqrt((mx_new[valid] - mx_exact[valid]) ** 2 + (my_new[valid] - my_exact[valid]) ** 2)
    return {
        "valid_count": cnt,
        "mean_epe": float(np.mean(epe)),
        "p50_epe": float(np.percentile(epe, 50)),
        "p95_epe": float(np.percentile(epe, 95)),
        "p99_epe": float(np.percentile(epe, 99)),
    }


# ============================================================
# Optional depth YUV writer
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
        "depth_yuv_semantics": "Y stores original linear depth code = round(depth / depth_scale_real); U/V neutral 512",
        "clipped_samples_total": int(clipped_total),
        "total_samples": int(n * h * w),
    }


# ============================================================
# Rough bit estimate
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


def estimate_bits_zero_predictor(rvecs: np.ndarray, tvecs: np.ndarray, r_step: float, t_step: float) -> dict[str, Any]:
    if len(rvecs) <= 1:
        return {}
    q_r = np.round(rvecs[1:] / r_step).astype(np.int64)
    q_t = np.round(tvecs[1:] / t_step).astype(np.int64)
    b_r, sum_r = signed_exp_golomb_bits_for_q(q_r)
    b_t, sum_t = signed_exp_golomb_bits_for_q(q_t)
    n = len(rvecs) - 1
    return {
        "estimator": "signed_exp_golomb_zero_predictor_rough",
        "coded_frames": int(n),
        "r_step": float(r_step),
        "t_step": float(t_step),
        "rotation_total_bits": int(sum_r),
        "translation_total_bits": int(sum_t),
        "rotation_avg_bits_frame": float(sum_r / max(n, 1)),
        "translation_avg_bits_frame": float(sum_t / max(n, 1)),
        "rotation_avg_bits_each": np.mean(b_r, axis=0).astype(float).tolist(),
        "translation_avg_bits_each": np.mean(b_t, axis=0).astype(float).tolist(),
    }


# ============================================================
# Main processing
# ============================================================

def load_camera_jsonl_optional(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def run(args: argparse.Namespace) -> None:
    npz_path = Path(args.npz)
    out_prefix = Path(args.out_prefix)
    camera_jsonl_path = Path(args.camera_jsonl) if args.camera_jsonl else None

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    base = sanitize_windows_filename_component(out_prefix.name)
    out_prefix = out_prefix.with_name(base)
    out_npz = out_prefix.with_name(out_prefix.name + "_fixedK_rtopt.npz")
    out_jsonl = out_prefix.with_name(out_prefix.name + "_camParam_fixedK_rtopt.jsonl")
    out_manifest = out_prefix.with_name(out_prefix.name + "_fixedK_rtopt_manifest.json")
    out_depth_yuv = out_prefix.with_name(out_prefix.name + "_depth_original_linear_yuv420p10le.yuv")

    for p in [out_npz, out_jsonl, out_manifest] + ([out_depth_yuv] if args.write_depth_yuv else []):
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --overwrite.")
        ensure_parent(p)

    log(f"Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for key in ["depth_original", "extrinsic", "intrinsic_original"]:
        if key not in data:
            raise KeyError(f"NPZ missing required key: {key}")

    depth = data["depth_original"].astype(np.float32)
    E_abs = data["extrinsic"].astype(np.float64)
    K_orig = data["intrinsic_original"].astype(np.float64)
    frame_indices = data["frame_indices"].astype(np.int32) if "frame_indices" in data else np.arange(depth.shape[0], dtype=np.int32)

    if depth.ndim != 3:
        raise ValueError(f"depth_original must be [N,H,W], got {depth.shape}")
    n, h, w = depth.shape
    if E_abs.shape != (n, 3, 4):
        raise ValueError(f"extrinsic shape mismatch: {E_abs.shape}")
    if K_orig.shape != (n, 3, 3):
        raise ValueError(f"intrinsic_original shape mismatch: {K_orig.shape}")
    if args.width is not None and int(args.width) != w:
        raise ValueError(f"--width {args.width} does not match depth width {w}")
    if args.height is not None and int(args.height) != h:
        raise ValueError(f"--height {args.height} does not match depth height {h}")

    log(f"Loaded: frames={n}, size={w}x{h}")
    log("Loading optional camera JSONL metadata...")
    _camera_records = load_camera_jsonl_optional(camera_jsonl_path)

    log("Building fixed K and exact fixed-K affine targets...")
    K_fixed = make_fixed_intrinsic(K_orig, w, h, args.fixed_center_mode)
    inv_K_fixed = np.linalg.inv(K_fixed)
    H = np.stack([inv_K_fixed @ K_orig[i] for i in range(n)], axis=0)
    H_inv = np.stack([np.linalg.inv(H[i]) for i in range(n)], axis=0)

    R_rel = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    t_rel = np.zeros((n, 3), dtype=np.float64)
    A_exact = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    b_exact = np.zeros((n, 3), dtype=np.float64)
    R_init = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    rvec_init = np.zeros((n, 3), dtype=np.float64)
    t_init = np.zeros((n, 3), dtype=np.float64)

    for i in range(1, n):
        R_i, t_i = relative_current_to_previous(E_abs[i], E_abs[i - 1])
        R_rel[i] = R_i
        t_rel[i] = t_i
        A_exact[i] = H[i - 1] @ R_i @ H_inv[i]
        b_exact[i] = H[i - 1] @ t_i
        R_init[i] = closest_rotation(A_exact[i])
        rvec_init[i] = rvec_from_R(R_init[i])
        t_init[i] = b_exact[i]

    log("Creating fixed-K ray grid...")
    rays = make_rays(K_fixed, w, h, z_sign=args.z_sign)

    R_opt = R_init.copy()
    rvec_opt = rvec_init.copy()
    t_opt = t_init.copy()
    opt_stats: list[dict[str, Any]] = [{"poc": 0, "mode": "copy_first"}]
    init_epe_stats: list[dict[str, Any]] = []
    opt_epe_stats: list[dict[str, Any]] = []

    rng = np.random.default_rng(int(args.seed))

    log(
        "Optimizing fixed-K R/t per pair: "
        f"opt_stride={args.opt_stride}, max_samples={args.max_samples}, "
        f"loss={args.loss}, max_nfev={args.max_nfev}"
    )

    for i in range(1, n):
        if args.progress_every > 0 and (i == 1 or i == n - 1 or i % args.progress_every == 0):
            log(f"R/t optimization: pair {i}/{n-1}")

        samples = make_pair_samples(
            depth=depth[i],
            rays=rays,
            A_exact=A_exact[i],
            b_exact=b_exact[i],
            K_fixed=K_fixed,
            z_sign=args.z_sign,
            stride=args.opt_stride,
            max_samples=args.max_samples,
            rng=rng,
        )
        result = optimize_pair_rt(
            samples=samples,
            K_fixed=K_fixed,
            rvec_init=rvec_init[i],
            t_init=t_init[i],
            z_sign=args.z_sign,
            loss=args.loss,
            f_scale=args.f_scale,
            max_nfev=args.max_nfev,
            residual_scale=args.residual_scale,
        )
        rvec_opt[i] = result["rvec"]
        R_opt[i] = result["R"]
        t_opt[i] = result["t"]

        stat = {
            "poc": int(i),
            "sample_count": int(samples["depth"].shape[0]),
            "success": result["success"],
            "status": result["status"],
            "message": result["message"],
            "nfev": result["nfev"],
            "cost": result["cost"],
            "sample_mean_epe_init": result["sample_mean_epe_init"],
            "sample_mean_epe_opt": result["sample_mean_epe_opt"],
        }
        opt_stats.append(stat)
        if args.progress_every > 0:
            log(
                f"  pair {i}: sample mean EPE {stat['sample_mean_epe_init']:.4f} -> "
                f"{stat['sample_mean_epe_opt']:.4f} px, nfev={stat['nfev']}, success={stat['success']}"
            )

    log(f"Evaluating EPE at stride={args.epe_stride}...")
    for i in range(1, n):
        if args.progress_every > 0 and (i == 1 or i == n - 1 or i % args.progress_every == 0):
            log(f"EPE evaluation: pair {i}/{n-1}")
        init_stat = evaluate_epe_pair(depth[i], rays, A_exact[i], b_exact[i], R_init[i], t_init[i], K_fixed, args.z_sign, args.epe_stride)
        opt_stat = evaluate_epe_pair(depth[i], rays, A_exact[i], b_exact[i], R_opt[i], t_opt[i], K_fixed, args.z_sign, args.epe_stride)
        init_stat = {"poc": int(i), **init_stat}
        opt_stat = {"poc": int(i), **opt_stat}
        init_epe_stats.append(init_stat)
        opt_epe_stats.append(opt_stat)

    avg_init = [x["mean_epe"] for x in init_epe_stats if x.get("mean_epe") is not None]
    p95_init = [x["p95_epe"] for x in init_epe_stats if x.get("p95_epe") is not None]
    avg_opt = [x["mean_epe"] for x in opt_epe_stats if x.get("mean_epe") is not None]
    p95_opt = [x["p95_epe"] for x in opt_epe_stats if x.get("p95_epe") is not None]

    depth_yuv_meta = None
    if args.write_depth_yuv:
        log("Writing original linear depth YUV420p10le...")
        scale_meta = choose_depth_scale_fixed_point(depth, args.depth_scale_percentile, args.depth_scale_precision, 10)
        depth_yuv_meta = write_depth_yuv420p10le_linear(out_depth_yuv, depth, scale_meta)

    log("Estimating rough camera bits...")
    # The bit estimate uses depth scale if available; otherwise use a robust sequence-level scale.
    if depth_yuv_meta is not None:
        depth_scale_real = float(depth_yuv_meta["depth_scale_real"])
    else:
        # For rough normalized t estimate only. This does not affect saved tvec.
        positive = depth[finite_positive_mask(depth)]
        depth_scale_real = float(np.percentile(positive, args.depth_scale_percentile) / 1023.0) if positive.size else 1.0
        depth_scale_real = max(depth_scale_real, 1e-12)
    bit_est = estimate_bits_zero_predictor(
        rvec_opt,
        t_opt / depth_scale_real,
        r_step=args.bit_est_r_step,
        t_step=args.bit_est_t_step,
    )

    log("Writing fixed-K R/t camera JSONL...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": "fixedK_rtopt_camparam_v1",
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
                "z_sign": float(args.z_sign),
            },
            "intrinsic_delta_order": [],
            "intrinsic_delta_bits_per_frame": 0,
            "depth_output": depth_yuv_meta,
            "optimization": {
                "summary": "Fixed RAP K; original depth unchanged; per-pair rigid R/t optimized to match original VGGT fixed-K exact affine projection.",
                "target": "exact fixed-K affine projection induced by original per-frame VGGT K/R/t",
                "optimize": "rvec and tvec per current_to_previous frame pair",
                "opt_stride": int(args.opt_stride),
                "max_samples": int(args.max_samples),
                "loss": args.loss,
                "f_scale_px": float(args.f_scale),
                "max_nfev": int(args.max_nfev),
                "fixed_center_mode": args.fixed_center_mode,
            },
            "rough_bit_estimate": bit_est,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in range(n):
            f.write(json.dumps({
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec": as_float_list(rvec_opt[i]),
                "tvec": as_float_list(t_opt[i]),
            }, ensure_ascii=False) + "\n")

    log("Saving NPZ...")
    payload = dict(
        frame_indices=frame_indices.astype(np.int32),
        K_original=K_orig.astype(np.float32),
        K_fixed=K_fixed.astype(np.float32),
        extrinsic_original=E_abs.astype(np.float32),
        rel_R_original_current_to_previous=R_rel.astype(np.float32),
        rel_t_original_current_to_previous=t_rel.astype(np.float32),
        fixedK_affine_A_exact=A_exact.astype(np.float32),
        fixedK_affine_b_exact=b_exact.astype(np.float32),
        rel_R_closest_current_to_previous=R_init.astype(np.float32),
        rel_t_closest_current_to_previous=t_init.astype(np.float32),
        rvec_closest_current_to_previous=rvec_init.astype(np.float32),
        rel_R_optimized_current_to_previous=R_opt.astype(np.float32),
        rel_t_optimized_current_to_previous=t_opt.astype(np.float32),
        rvec_optimized_current_to_previous=rvec_opt.astype(np.float32),
        opt_stats_json=np.asarray(json.dumps(opt_stats, ensure_ascii=False), dtype=object),
        init_epe_stats_json=np.asarray(json.dumps(init_epe_stats, ensure_ascii=False), dtype=object),
        opt_epe_stats_json=np.asarray(json.dumps(opt_epe_stats, ensure_ascii=False), dtype=object),
    )
    if args.save_depth_in_npz:
        payload["depth_original"] = depth.astype(np.float32)
    if args.compressed_npz:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)

    manifest = {
        "source_npz": os.path.abspath(npz_path),
        "source_camera_jsonl": os.path.abspath(camera_jsonl_path) if camera_jsonl_path else None,
        "outputs": {
            "npz": os.path.abspath(out_npz),
            "camera_jsonl": os.path.abspath(out_jsonl),
            "depth_yuv": os.path.abspath(out_depth_yuv) if args.write_depth_yuv else None,
            "manifest": os.path.abspath(out_manifest),
        },
        "frame_count": int(n),
        "size": {"width": int(w), "height": int(h)},
        "K_fixed": K_fixed.astype(float).tolist(),
        "config": vars(args),
        "depth_yuv": depth_yuv_meta,
        "epe_summary_vs_exact_fixedK_affine_target": {
            "closest_rotation": {
                "mean_of_mean_epe": float(np.mean(avg_init)) if avg_init else None,
                "mean_of_p95_epe": float(np.mean(p95_init)) if p95_init else None,
                "per_pair": init_epe_stats,
            },
            "optimized_rt": {
                "mean_of_mean_epe": float(np.mean(avg_opt)) if avg_opt else None,
                "mean_of_p95_epe": float(np.mean(p95_opt)) if p95_opt else None,
                "per_pair": opt_epe_stats,
            },
        },
        "optimization_stats": opt_stats,
        "rough_bit_estimate": bit_est,
        "notes": [
            "This script intentionally does not smooth camera trajectories and does not rewrite depth.",
            "It tests the isolated gain from replacing per-frame intrinsic with one fixed RAP K and optimizing per-pair rigid R/t.",
            "If EPE is low enough, K signaling can be removed while preserving most of the original VGGT projection behavior.",
        ],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print("Fixed-K R/t optimization done")
    print("============================================================")
    print(f"frames                  : {n}")
    print(f"size                    : {w}x{h}")
    print(f"fixed K                 : fx={K_fixed[0,0]:.6f}, fy={K_fixed[1,1]:.6f}, cx={K_fixed[0,2]:.6f}, cy={K_fixed[1,2]:.6f}")
    if avg_init:
        print(f"closest mean EPE        : {np.mean(avg_init):.6f} px")
        print(f"closest mean p95 EPE    : {np.mean(p95_init):.6f} px")
    if avg_opt:
        print(f"optimized mean EPE      : {np.mean(avg_opt):.6f} px")
        print(f"optimized mean p95 EPE  : {np.mean(p95_opt):.6f} px")
    print("------------------------------------------------------------")
    print(f"camera jsonl            : {out_jsonl}")
    print(f"npz                     : {out_npz}")
    if args.write_depth_yuv:
        print(f"depth yuv               : {out_depth_yuv}")
    print(f"manifest                : {out_manifest}")
    print("============================================================")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize fixed-K rigid R/t per VGGT frame pair, with no depth rewrite and no trajectory smoothing."
    )
    p.add_argument("--npz", required=True, help="VGGT-Omega output NPZ")
    p.add_argument("--camera-jsonl", default=None, help="Optional VGGT-Omega camera JSONL for metadata")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--width", type=int, default=None, help="Optional validation width")
    p.add_argument("--height", type=int, default=None, help="Optional validation height")

    p.add_argument("--fixed-center-mode", choices=["median", "image-center", "first"], default="image-center")
    p.add_argument("--z-sign", type=float, default=1.0)

    p.add_argument("--opt-stride", type=int, default=8, help="Sampling stride for optimization. 8 is fast; 4 is more accurate.")
    p.add_argument("--max-samples", type=int, default=60000, help="Max valid samples per pair for optimization. 0 means all sampled points.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    p.add_argument("--f-scale", type=float, default=1.0, help="Robust loss transition scale in pixels.")
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--residual-scale", type=float, default=1.0, help="Internal residual scaling; keep 1.0 normally.")

    p.add_argument("--epe-stride", type=int, default=4, help="Stride for final EPE reporting. 1=full res.")
    p.add_argument("--progress-every", type=int, default=1)

    p.add_argument("--write-depth-yuv", action="store_true", help="Also write original depth as linear yuv420p10le using fixed-point depth_scale.")
    p.add_argument("--depth-scale-precision", type=int, default=100000)
    p.add_argument("--depth-scale-percentile", type=float, default=99.9)
    p.add_argument("--bit-est-r-step", type=float, default=2.0 ** -12)
    p.add_argument("--bit-est-t-step", type=float, default=2.0 ** -10)

    p.add_argument("--save-depth-in-npz", action="store_true", help="Save depth_original into output NPZ. Off by default to keep file small.")
    p.add_argument("--compressed-npz", action="store_true", help="Use np.savez_compressed. Smaller but slower.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()















--opt-stride 4 \
--max-samples 120000 \
--max-nfev 120






