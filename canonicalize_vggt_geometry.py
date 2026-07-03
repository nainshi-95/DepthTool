#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternating optimization of one fixed RAP intrinsic K and per-pair current->previous R/t
for VGGT-Omega geometry, without depth rewriting.

Goal
----
Remove per-frame intrinsic signaling. Instead, send one fixed K per RAP and optimize
only per-frame-pair rigid R/t so that

  project(K_fixed, R_i, t_i, depth_i, K_fixed)

matches the original VGGT projection map produced by

  project(K_{i-1}, R_rel_i, t_rel_i, depth_i, K_i).

This is intentionally narrow:
  - no trajectory smoothing
  - no depth target solve
  - no depth rewrite
  - no block/global fitting

It tests how much of the per-frame K variation can be absorbed into one RAP K plus
optimized rigid R/t.
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
KTuneMode = Literal["none", "fx-fy", "fx-fy-cx-cy"]


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


def as_float_list(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a).reshape(-1)]


def finite_positive_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


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


def make_initial_fixed_intrinsic(Ks: np.ndarray, width: int, height: int, center_mode: CenterMode) -> np.ndarray:
    Ks = np.asarray(Ks, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(np.median(Ks[:, 0, 0]))
    K[1, 1] = float(np.median(Ks[:, 1, 1]))
    if center_mode == "median":
        K[0, 2] = float(np.median(Ks[:, 0, 2]))
        K[1, 2] = float(np.median(Ks[:, 1, 2]))
    elif center_mode == "image-center":
        K[0, 2] = float(width) / 2.0
        K[1, 2] = float(height) / 2.0
    elif center_mode == "first":
        K[0, 2] = float(Ks[0, 0, 2])
        K[1, 2] = float(Ks[0, 1, 2])
    else:
        raise ValueError(center_mode)
    return K


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


def rays_from_pixels(K: np.ndarray, x: np.ndarray, y: np.ndarray, z_sign: float) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    out = np.empty((x.size, 3), dtype=np.float64)
    out[:, 0] = (x.astype(np.float64) - cx) / fx
    out[:, 1] = (y.astype(np.float64) - cy) / fy
    out[:, 2] = float(z_sign)
    return out


def project_points_to_map(X: np.ndarray, K: np.ndarray, z_sign: float, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = X[:, 2]
    valid = np.isfinite(X).all(axis=1) & (Z * z_sign > eps)
    denom = np.where(np.abs(Z) > eps, Z, np.where(Z >= 0, eps, -eps))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    mx = fx * (X[:, 0] / denom) + cx
    my = fy * (X[:, 1] / denom) + cy
    return mx.astype(np.float64), my.astype(np.float64), valid


def apply_rt(depth_flat: np.ndarray, rays_flat: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    X = depth_flat[:, None].astype(np.float64) * rays_flat.astype(np.float64)
    return X @ R.T + np.asarray(t, dtype=np.float64).reshape(1, 3)


def sample_grid(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, height, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, width, max(1, int(stride)), dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


# ============================================================
# Fixed-K initialization and samples
# ============================================================

def init_rt_from_fixedK(K_fixed: np.ndarray, K_orig: np.ndarray, R_rel: np.ndarray, t_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pair i->i-1, form the exact fixed-K affine transform and project it
    to a rigid initial R/t.

      H_i = inv(K_fixed) @ K_i
      A_i = H_{i-1} @ R_rel_i @ inv(H_i)
      b_i = H_{i-1} @ t_rel_i
    """
    n = K_orig.shape[0]
    invK = np.linalg.inv(K_fixed)
    H = np.stack([invK @ K_orig[i] for i in range(n)], axis=0)
    H_inv = np.stack([np.linalg.inv(H[i]) for i in range(n)], axis=0)

    A_exact = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    b_exact = np.zeros((n, 3), dtype=np.float64)
    R_init = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    t_init = np.zeros((n, 3), dtype=np.float64)
    rvec_init = np.zeros((n, 3), dtype=np.float64)

    for i in range(1, n):
        A_exact[i] = H[i - 1] @ R_rel[i] @ H_inv[i]
        b_exact[i] = H[i - 1] @ t_rel[i]
        R_init[i] = closest_rotation(A_exact[i])
        t_init[i] = b_exact[i]
        rvec_init[i] = rvec_from_R(R_init[i])
    return R_init, rvec_init, t_init, A_exact


def make_pair_samples_original_target(
    depth_i: np.ndarray,
    K_cur: np.ndarray,
    K_prev: np.ndarray,
    R_rel_i: np.ndarray,
    t_rel_i: np.ndarray,
    z_sign: float,
    stride: int,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Create sample points and original VGGT pixel target map for one pair."""
    h, w = depth_i.shape
    yy, xx = sample_grid(h, w, stride)
    depth_flat = depth_i[yy, xx].astype(np.float64)
    rays_orig = rays_from_pixels(K_cur, xx.astype(np.float64), yy.astype(np.float64), z_sign)
    X_prev = apply_rt(depth_flat, rays_orig, R_rel_i, t_rel_i)
    tx, ty, valid = project_points_to_map(X_prev, K_prev, z_sign)
    valid &= finite_positive_mask(depth_flat)
    valid &= np.isfinite(tx) & np.isfinite(ty)
    valid &= (tx >= 0.0) & (tx <= w - 1) & (ty >= 0.0) & (ty <= h - 1)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise RuntimeError("No valid projection samples for this pair")
    if max_samples > 0 and idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
    return {
        "x": xx[idx].astype(np.float64),
        "y": yy[idx].astype(np.float64),
        "depth": depth_flat[idx].astype(np.float64),
        "target_x": tx[idx].astype(np.float64),
        "target_y": ty[idx].astype(np.float64),
    }


def subset_samples(samples: dict[str, np.ndarray], max_samples: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = samples["depth"].shape[0]
    if max_samples <= 0 or n <= max_samples:
        return samples
    idx = rng.choice(np.arange(n), size=max_samples, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in samples.items()}


# ============================================================
# Residuals and optimization
# ============================================================

def predict_map_fixedK(samples: dict[str, np.ndarray], K_fixed: np.ndarray, R: np.ndarray, t: np.ndarray, z_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = rays_from_pixels(K_fixed, samples["x"], samples["y"], z_sign)
    X = apply_rt(samples["depth"], rays, R, t)
    return project_points_to_map(X, K_fixed, z_sign)


def residual_rt(params: np.ndarray, samples: dict[str, np.ndarray], K_fixed: np.ndarray, z_sign: float, residual_scale: float) -> np.ndarray:
    rvec = params[:3]
    t = params[3:6]
    R = R_from_rvec(rvec)
    mx, my, valid = predict_map_fixedK(samples, K_fixed, R, t, z_sign)
    dx = mx - samples["target_x"]
    dy = my - samples["target_y"]
    bad = ~valid | ~np.isfinite(dx) | ~np.isfinite(dy)
    if np.any(bad):
        dx = dx.copy(); dy = dy.copy()
        dx[bad] = 1000.0
        dy[bad] = 1000.0
    return np.concatenate([dx, dy], axis=0) / float(residual_scale)


def optimize_one_rt(
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
        raise ImportError("scipy is required: pip install scipy") from SCIPY_IMPORT_ERROR
    x0 = np.concatenate([np.asarray(rvec_init, dtype=np.float64).reshape(3), np.asarray(t_init, dtype=np.float64).reshape(3)])
    res0 = residual_rt(x0, samples, K_fixed, z_sign, residual_scale)
    half = res0.size // 2
    mean0 = float(np.mean(np.sqrt(res0[:half] ** 2 + res0[half:] ** 2)) * residual_scale)
    opt = least_squares(
        residual_rt,
        x0,
        args=(samples, K_fixed, z_sign, residual_scale),
        method="trf",
        loss=loss,
        f_scale=float(f_scale) / float(residual_scale),
        x_scale="jac",
        max_nfev=int(max_nfev),
        verbose=0,
    )
    res1 = residual_rt(opt.x, samples, K_fixed, z_sign, residual_scale)
    half = res1.size // 2
    mean1 = float(np.mean(np.sqrt(res1[:half] ** 2 + res1[half:] ** 2)) * residual_scale)
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


def k_to_params(K: np.ndarray, K_ref: np.ndarray, mode: KTuneMode) -> np.ndarray:
    if mode == "none":
        return np.zeros(0, dtype=np.float64)
    p = [math.log(float(K[0, 0]) / float(K_ref[0, 0])), math.log(float(K[1, 1]) / float(K_ref[1, 1]))]
    if mode == "fx-fy-cx-cy":
        p += [float(K[0, 2]) - float(K_ref[0, 2]), float(K[1, 2]) - float(K_ref[1, 2])]
    return np.asarray(p, dtype=np.float64)


def params_to_k(params: np.ndarray, K_ref: np.ndarray, mode: KTuneMode) -> np.ndarray:
    K = np.asarray(K_ref, dtype=np.float64).copy()
    if mode == "none":
        return K
    params = np.asarray(params, dtype=np.float64)
    K[0, 0] = float(K_ref[0, 0]) * math.exp(float(params[0]))
    K[1, 1] = float(K_ref[1, 1]) * math.exp(float(params[1]))
    if mode == "fx-fy-cx-cy":
        K[0, 2] = float(K_ref[0, 2]) + float(params[2])
        K[1, 2] = float(K_ref[1, 2]) + float(params[3])
    return K


def residual_k_global(
    params: np.ndarray,
    samples_per_pair: list[dict[str, np.ndarray]],
    R_all: np.ndarray,
    t_all: np.ndarray,
    K_ref: np.ndarray,
    mode: KTuneMode,
    z_sign: float,
    residual_scale: float,
    logf_prior_sigma: float,
    center_prior_sigma: float,
    k_prior_weight: float,
) -> np.ndarray:
    K = params_to_k(params, K_ref, mode)
    out = []
    # Pairs list starts at poc 1; R_all/t_all include poc 0.
    for pair_idx, s in enumerate(samples_per_pair, start=1):
        mx, my, valid = predict_map_fixedK(s, K, R_all[pair_idx], t_all[pair_idx], z_sign)
        dx = mx - s["target_x"]
        dy = my - s["target_y"]
        bad = ~valid | ~np.isfinite(dx) | ~np.isfinite(dy)
        if np.any(bad):
            dx = dx.copy(); dy = dy.copy()
            dx[bad] = 1000.0
            dy[bad] = 1000.0
        out.append(dx / float(residual_scale))
        out.append(dy / float(residual_scale))
    if k_prior_weight > 0 and params.size > 0:
        pri = []
        pri.append(params[0] / max(logf_prior_sigma, 1e-12))
        pri.append(params[1] / max(logf_prior_sigma, 1e-12))
        if mode == "fx-fy-cx-cy":
            pri.append(params[2] / max(center_prior_sigma, 1e-12))
            pri.append(params[3] / max(center_prior_sigma, 1e-12))
        out.append(math.sqrt(float(k_prior_weight)) * np.asarray(pri, dtype=np.float64))
    return np.concatenate(out, axis=0)


def optimize_k_global(
    K_current: np.ndarray,
    K_ref: np.ndarray,
    mode: KTuneMode,
    samples_per_pair: list[dict[str, np.ndarray]],
    R_all: np.ndarray,
    t_all: np.ndarray,
    z_sign: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    residual_scale: float,
    logf_bound: float,
    center_bound: float,
    logf_prior_sigma: float,
    center_prior_sigma: float,
    k_prior_weight: float,
) -> dict[str, Any]:
    if mode == "none":
        return {"K": K_current.copy(), "params": np.zeros(0), "success": True, "nfev": 0, "cost": 0.0, "message": "k tuning disabled"}
    if least_squares is None:
        raise ImportError("scipy is required: pip install scipy") from SCIPY_IMPORT_ERROR
    x0 = k_to_params(K_current, K_ref, mode)
    if mode == "fx-fy":
        lb = np.array([-abs(logf_bound), -abs(logf_bound)], dtype=np.float64)
        ub = np.array([ abs(logf_bound),  abs(logf_bound)], dtype=np.float64)
    elif mode == "fx-fy-cx-cy":
        lb = np.array([-abs(logf_bound), -abs(logf_bound), -abs(center_bound), -abs(center_bound)], dtype=np.float64)
        ub = np.array([ abs(logf_bound),  abs(logf_bound),  abs(center_bound),  abs(center_bound)], dtype=np.float64)
    else:
        raise ValueError(mode)
    x0 = np.clip(x0, lb, ub)
    opt = least_squares(
        residual_k_global,
        x0,
        args=(samples_per_pair, R_all, t_all, K_ref, mode, z_sign, residual_scale, logf_prior_sigma, center_prior_sigma, k_prior_weight),
        method="trf",
        loss=loss,
        f_scale=float(f_scale) / float(residual_scale),
        x_scale="jac",
        bounds=(lb, ub),
        max_nfev=int(max_nfev),
        verbose=0,
    )
    return {
        "K": params_to_k(opt.x, K_ref, mode),
        "params": opt.x.astype(np.float64),
        "success": bool(opt.success),
        "status": int(opt.status),
        "message": str(opt.message),
        "nfev": int(opt.nfev),
        "cost": float(opt.cost),
    }


def evaluate_epe_pair(samples: dict[str, np.ndarray], K_fixed: np.ndarray, R: np.ndarray, t: np.ndarray, z_sign: float) -> dict[str, Any]:
    mx, my, valid = predict_map_fixedK(samples, K_fixed, R, t, z_sign)
    valid &= np.isfinite(mx) & np.isfinite(my)
    valid &= np.isfinite(samples["target_x"]) & np.isfinite(samples["target_y"])
    cnt = int(np.count_nonzero(valid))
    if cnt == 0:
        return {"valid_count": 0, "mean_epe": None, "p50_epe": None, "p95_epe": None, "p99_epe": None}
    epe = np.sqrt((mx[valid] - samples["target_x"][valid]) ** 2 + (my[valid] - samples["target_y"][valid]) ** 2)
    return {
        "valid_count": cnt,
        "mean_epe": float(np.mean(epe)),
        "p50_epe": float(np.percentile(epe, 50)),
        "p95_epe": float(np.percentile(epe, 95)),
        "p99_epe": float(np.percentile(epe, 99)),
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
    k = (code_num + 1).bit_length() - 1
    return 2 * k + 1


def signed_exp_golomb_bits_for_q(q: np.ndarray) -> tuple[np.ndarray, int]:
    q = np.asarray(q, dtype=np.int64)
    bits = np.zeros_like(q, dtype=np.int64)
    it = np.nditer(q, flags=["multi_index"])
    for val in it:
        bits[it.multi_index] = ue_exp_golomb_bits(signed_to_code_num(int(val)))
    return bits, int(np.sum(bits))


def estimate_bits_zero_predictor(rvecs: np.ndarray, tvecs: np.ndarray, r_step: float, t_step: float, depth_for_scale: np.ndarray, depth_scale_percentile: float) -> dict[str, Any]:
    if len(rvecs) <= 1:
        return {}
    pos = depth_for_scale[finite_positive_mask(depth_for_scale)]
    depth_scale_real = float(np.percentile(pos, depth_scale_percentile) / 1023.0) if pos.size else 1.0
    depth_scale_real = max(depth_scale_real, 1e-12)
    q_r = np.round(rvecs[1:] / r_step).astype(np.int64)
    q_t = np.round((tvecs[1:] / depth_scale_real) / t_step).astype(np.int64)
    b_r, sum_r = signed_exp_golomb_bits_for_q(q_r)
    b_t, sum_t = signed_exp_golomb_bits_for_q(q_t)
    n = len(rvecs) - 1
    return {
        "estimator": "signed_exp_golomb_zero_predictor_rough",
        "coded_frames": int(n),
        "r_step": float(r_step),
        "t_step": float(t_step),
        "depth_scale_real_for_t_norm": float(depth_scale_real),
        "rotation_total_bits": int(sum_r),
        "translation_total_bits": int(sum_t),
        "rotation_avg_bits_frame": float(sum_r / max(n, 1)),
        "translation_avg_bits_frame": float(sum_t / max(n, 1)),
        "rotation_avg_bits_each": np.mean(b_r, axis=0).astype(float).tolist(),
        "translation_avg_bits_each": np.mean(b_t, axis=0).astype(float).tolist(),
    }


# ============================================================
# Main
# ============================================================

def summarize_epe(stats: list[dict[str, Any]]) -> dict[str, Any]:
    means = [s["mean_epe"] for s in stats if s.get("mean_epe") is not None]
    p95s = [s["p95_epe"] for s in stats if s.get("p95_epe") is not None]
    p99s = [s["p99_epe"] for s in stats if s.get("p99_epe") is not None]
    return {
        "mean_of_mean_epe": float(np.mean(means)) if means else None,
        "mean_of_p95_epe": float(np.mean(p95s)) if p95s else None,
        "mean_of_p99_epe": float(np.mean(p99s)) if p99s else None,
        "per_pair": stats,
    }


def run(args: argparse.Namespace) -> None:
    if least_squares is None:
        raise ImportError("scipy is required: pip install scipy") from SCIPY_IMPORT_ERROR

    npz_path = Path(args.npz)
    camera_jsonl_path = Path(args.camera_jsonl) if args.camera_jsonl else None
    out_prefix = Path(args.out_prefix)
    base = sanitize_windows_filename_component(out_prefix.name)
    out_prefix = out_prefix.with_name(base)
    out_npz = out_prefix.with_name(out_prefix.name + "_fixedK_alt_rtopt.npz")
    out_jsonl = out_prefix.with_name(out_prefix.name + "_camParam_fixedK_alt_rtopt.jsonl")
    out_manifest = out_prefix.with_name(out_prefix.name + "_fixedK_alt_rtopt_manifest.json")

    for p in [out_npz, out_jsonl, out_manifest]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --overwrite.")
        ensure_parent(p)

    log(f"Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for k in ["depth_original", "extrinsic", "intrinsic_original"]:
        if k not in data:
            raise KeyError(f"NPZ missing required key: {k}")
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

    R_rel = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    t_rel = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n):
        R_rel[i], t_rel[i] = relative_current_to_previous(E_abs[i], E_abs[i - 1])

    rng = np.random.default_rng(int(args.seed))
    log("Building original VGGT pixel target samples...")
    # Build full optimization sample sets.
    samples_rt: list[dict[str, np.ndarray]] = []
    samples_k: list[dict[str, np.ndarray]] = []
    for i in range(1, n):
        if args.progress_every > 0 and (i == 1 or i == n - 1 or i % args.progress_every == 0):
            log(f"Target samples: pair {i}/{n-1}")
        s = make_pair_samples_original_target(
            depth_i=depth[i],
            K_cur=K_orig[i],
            K_prev=K_orig[i - 1],
            R_rel_i=R_rel[i],
            t_rel_i=t_rel[i],
            z_sign=float(args.z_sign),
            stride=int(args.opt_stride),
            max_samples=max(int(args.max_samples), int(args.k_max_samples)),
            rng=rng,
        )
        samples_rt.append(subset_samples(s, int(args.max_samples), rng))
        samples_k.append(subset_samples(s, int(args.k_max_samples), rng))

    # Separate EPE samples so reporting is stable and independent of opt sampling.
    log("Building EPE reporting samples...")
    samples_epe: list[dict[str, np.ndarray]] = []
    for i in range(1, n):
        samples_epe.append(make_pair_samples_original_target(
            depth_i=depth[i],
            K_cur=K_orig[i],
            K_prev=K_orig[i - 1],
            R_rel_i=R_rel[i],
            t_rel_i=t_rel[i],
            z_sign=float(args.z_sign),
            stride=int(args.epe_stride),
            max_samples=0,
            rng=rng,
        ))

    K_ref = make_initial_fixed_intrinsic(K_orig, w, h, args.fixed_center_mode)
    K_cur = K_ref.copy()
    log(f"Initial K: fx={K_cur[0,0]:.6f}, fy={K_cur[1,1]:.6f}, cx={K_cur[0,2]:.6f}, cy={K_cur[1,2]:.6f}")

    R_opt, rvec_opt, t_opt, _A0 = init_rt_from_fixedK(K_cur, K_orig, R_rel, t_rel)

    iteration_stats: list[dict[str, Any]] = []

    for it in range(int(args.alt_iters) + 1):
        log(f"========== Alternation iteration {it}/{args.alt_iters} ==========")
        if it > 0 or args.optimize_rt_first:
            log("Optimizing per-pair R/t with current K...")
            for i in range(1, n):
                if args.progress_every > 0 and (i == 1 or i == n - 1 or i % args.progress_every == 0):
                    log(f"R/t opt: iter {it}, pair {i}/{n-1}")
                res = optimize_one_rt(
                    samples=samples_rt[i - 1],
                    K_fixed=K_cur,
                    rvec_init=rvec_opt[i],
                    t_init=t_opt[i],
                    z_sign=float(args.z_sign),
                    loss=args.loss,
                    f_scale=float(args.f_scale),
                    max_nfev=int(args.max_nfev),
                    residual_scale=float(args.residual_scale),
                )
                R_opt[i] = res["R"]
                rvec_opt[i] = res["rvec"]
                t_opt[i] = res["t"]
                if args.progress_every > 0:
                    log(f"  pair {i}: sample mean EPE {res['sample_mean_epe_init']:.4f} -> {res['sample_mean_epe_opt']:.4f}, nfev={res['nfev']}")

        # Report after current R/t, before K update.
        report_stats = [evaluate_epe_pair(samples_epe[i - 1], K_cur, R_opt[i], t_opt[i], float(args.z_sign)) for i in range(1, n)]
        report_summary = summarize_epe(report_stats)
        log(f"Before K update: mean={report_summary['mean_of_mean_epe']:.6f}, p95={report_summary['mean_of_p95_epe']:.6f}")

        k_stat: dict[str, Any] | None = None
        if it < int(args.alt_iters) and args.k_tune_mode != "none":
            log(f"Optimizing global fixed K: mode={args.k_tune_mode}")
            k_res = optimize_k_global(
                K_current=K_cur,
                K_ref=K_ref,
                mode=args.k_tune_mode,
                samples_per_pair=samples_k,
                R_all=R_opt,
                t_all=t_opt,
                z_sign=float(args.z_sign),
                loss=args.loss,
                f_scale=float(args.f_scale),
                max_nfev=int(args.k_max_nfev),
                residual_scale=float(args.residual_scale),
                logf_bound=float(args.k_logf_bound),
                center_bound=float(args.k_center_bound),
                logf_prior_sigma=float(args.k_logf_prior_sigma),
                center_prior_sigma=float(args.k_center_prior_sigma),
                k_prior_weight=float(args.k_prior_weight),
            )
            K_prev = K_cur.copy()
            K_cur = k_res["K"]
            k_stat = {
                "success": k_res["success"],
                "status": k_res.get("status"),
                "message": k_res["message"],
                "nfev": k_res["nfev"],
                "cost": k_res["cost"],
                "K_before": K_prev.astype(float).tolist(),
                "K_after": K_cur.astype(float).tolist(),
                "delta_fx": float(K_cur[0, 0] - K_prev[0, 0]),
                "delta_fy": float(K_cur[1, 1] - K_prev[1, 1]),
                "delta_cx": float(K_cur[0, 2] - K_prev[0, 2]),
                "delta_cy": float(K_cur[1, 2] - K_prev[1, 2]),
            }
            log(
                f"K update: fx {K_prev[0,0]:.6f}->{K_cur[0,0]:.6f}, "
                f"fy {K_prev[1,1]:.6f}->{K_cur[1,1]:.6f}, "
                f"cx {K_prev[0,2]:.6f}->{K_cur[0,2]:.6f}, "
                f"cy {K_prev[1,2]:.6f}->{K_cur[1,2]:.6f}"
            )

        iteration_stats.append({
            "iteration": int(it),
            "K": K_cur.astype(float).tolist(),
            "epe_before_k_update": report_summary,
            "k_update": k_stat,
        })

    log("Final R/t optimization after last K update...")
    for i in range(1, n):
        if args.progress_every > 0 and (i == 1 or i == n - 1 or i % args.progress_every == 0):
            log(f"Final R/t opt: pair {i}/{n-1}")
        res = optimize_one_rt(
            samples=samples_rt[i - 1],
            K_fixed=K_cur,
            rvec_init=rvec_opt[i],
            t_init=t_opt[i],
            z_sign=float(args.z_sign),
            loss=args.loss,
            f_scale=float(args.f_scale),
            max_nfev=int(args.max_nfev),
            residual_scale=float(args.residual_scale),
        )
        R_opt[i] = res["R"]
        rvec_opt[i] = res["rvec"]
        t_opt[i] = res["t"]

    final_epe_stats = [evaluate_epe_pair(samples_epe[i - 1], K_cur, R_opt[i], t_opt[i], float(args.z_sign)) for i in range(1, n)]
    final_summary = summarize_epe(final_epe_stats)
    bit_est = estimate_bits_zero_predictor(
        rvec_opt,
        t_opt,
        r_step=float(args.bit_est_r_step),
        t_step=float(args.bit_est_t_step),
        depth_for_scale=depth,
        depth_scale_percentile=float(args.depth_scale_percentile),
    )

    log("Writing camera JSONL...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": "fixedK_alternating_rtopt_camparam_v1",
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
            "intrinsic_mode": "rap_fixed_optimized",
            "intrinsic": {
                "fx": float(K_cur[0, 0]),
                "fy": float(K_cur[1, 1]),
                "cx": float(K_cur[0, 2]),
                "cy": float(K_cur[1, 2]),
                "z_sign": float(args.z_sign),
            },
            "intrinsic_delta_order": [],
            "intrinsic_delta_bits_per_frame": 0,
            "optimization": {
                "summary": "One RAP-level fixed K is globally tuned, and per-pair rigid R/t is optimized. Depth is unchanged.",
                "target": "original VGGT pixel projection using per-frame K_i/K_{i-1} and original relative R/t",
                "k_tune_mode": args.k_tune_mode,
                "alt_iters": int(args.alt_iters),
                "opt_stride": int(args.opt_stride),
                "max_samples": int(args.max_samples),
                "k_max_samples": int(args.k_max_samples),
                "loss": args.loss,
                "f_scale_px": float(args.f_scale),
                "max_nfev_rt": int(args.max_nfev),
                "max_nfev_k": int(args.k_max_nfev),
                "fixed_center_mode_initial": args.fixed_center_mode,
            },
            "epe_summary_vs_original_vggt_pixel_target": final_summary,
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
        K_initial=K_ref.astype(np.float32),
        K_fixed_optimized=K_cur.astype(np.float32),
        extrinsic_original=E_abs.astype(np.float32),
        rel_R_original_current_to_previous=R_rel.astype(np.float32),
        rel_t_original_current_to_previous=t_rel.astype(np.float32),
        rel_R_optimized_current_to_previous=R_opt.astype(np.float32),
        rel_t_optimized_current_to_previous=t_opt.astype(np.float32),
        rvec_optimized_current_to_previous=rvec_opt.astype(np.float32),
        final_epe_stats_json=np.asarray(json.dumps(final_epe_stats, ensure_ascii=False), dtype=object),
        iteration_stats_json=np.asarray(json.dumps(iteration_stats, ensure_ascii=False), dtype=object),
        config_json=np.asarray(json.dumps(vars(args), ensure_ascii=False), dtype=object),
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
            "manifest": os.path.abspath(out_manifest),
        },
        "frame_count": int(n),
        "size": {"width": int(w), "height": int(h)},
        "K_initial": K_ref.astype(float).tolist(),
        "K_fixed_optimized": K_cur.astype(float).tolist(),
        "config": vars(args),
        "final_epe_summary_vs_original_vggt_pixel_target": final_summary,
        "iteration_stats": iteration_stats,
        "rough_bit_estimate": bit_est,
        "notes": [
            "Depth is unchanged. This isolates the effect of removing per-frame K signaling.",
            "Only one K is written in the JSONL header, so intrinsic_delta_bits_per_frame is zero.",
            "K tuning is global over the RAP; R/t optimization is per current-to-previous pair.",
        ],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print("Alternating fixed-K + R/t optimization done")
    print("============================================================")
    print(f"frames                    : {n}")
    print(f"size                      : {w}x{h}")
    print(f"initial K                 : fx={K_ref[0,0]:.6f}, fy={K_ref[1,1]:.6f}, cx={K_ref[0,2]:.6f}, cy={K_ref[1,2]:.6f}")
    print(f"optimized K               : fx={K_cur[0,0]:.6f}, fy={K_cur[1,1]:.6f}, cx={K_cur[0,2]:.6f}, cy={K_cur[1,2]:.6f}")
    print(f"final optimized mean EPE  : {final_summary['mean_of_mean_epe']:.6f} px")
    print(f"final optimized p95 EPE   : {final_summary['mean_of_p95_epe']:.6f} px")
    print("------------------------------------------------------------")
    print(f"camera jsonl              : {out_jsonl}")
    print(f"npz                       : {out_npz}")
    print(f"manifest                  : {out_manifest}")
    print("============================================================")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alternating optimize one RAP fixed K and per-pair R/t for VGGT-Omega outputs.")
    p.add_argument("--npz", required=True, help="VGGT-Omega output NPZ")
    p.add_argument("--camera-jsonl", default=None, help="Optional source camera JSONL for metadata")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)

    p.add_argument("--fixed-center-mode", choices=["median", "image-center", "first"], default="image-center")
    p.add_argument("--k-tune-mode", choices=["none", "fx-fy", "fx-fy-cx-cy"], default="fx-fy")
    p.add_argument("--alt-iters", type=int, default=2, help="Number of K update iterations. A final R/t optimization is always run after the last K update.")
    p.add_argument("--optimize-rt-first", action="store_true", default=True, help="Run R/t optimization before the first K update. Default true.")

    p.add_argument("--z-sign", type=float, default=1.0)
    p.add_argument("--opt-stride", type=int, default=8, help="Sampling stride for R/t optimization.")
    p.add_argument("--max-samples", type=int, default=60000, help="Max samples per pair for R/t optimization. 0 means all sampled points.")
    p.add_argument("--k-max-samples", type=int, default=12000, help="Max samples per pair for global K optimization.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    p.add_argument("--f-scale", type=float, default=1.0, help="Robust loss transition scale in pixels.")
    p.add_argument("--max-nfev", type=int, default=80, help="Max function evaluations for each per-pair R/t optimization.")
    p.add_argument("--k-max-nfev", type=int, default=60, help="Max function evaluations for global K optimization.")
    p.add_argument("--residual-scale", type=float, default=1.0)

    p.add_argument("--k-logf-bound", type=float, default=0.15, help="Bound on log(f/f_initial). 0.15 is about +/-16%.")
    p.add_argument("--k-center-bound", type=float, default=32.0, help="Bound on cx/cy delta in pixels if center tuning is enabled.")
    p.add_argument("--k-logf-prior-sigma", type=float, default=0.08)
    p.add_argument("--k-center-prior-sigma", type=float, default=16.0)
    p.add_argument("--k-prior-weight", type=float, default=0.05, help="Small regularization toward initial K. Use 0 to disable.")

    p.add_argument("--epe-stride", type=int, default=4, help="Stride for final EPE reporting.")
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--bit-est-r-step", type=float, default=2.0 ** -12)
    p.add_argument("--bit-est-t-step", type=float, default=2.0 ** -10)
    p.add_argument("--depth-scale-percentile", type=float, default=99.9)
    p.add_argument("--save-depth-in-npz", action="store_true")
    p.add_argument("--compressed-npz", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()













python -u optimize_fixedK_rt_alternating_vggt.py \
  --npz out/test_rap0_vggt_omega_outputs.npz \
  --camera-jsonl out/test_rap0_camera.jsonl \
  --out-prefix out/test_rap0_fixedK_alt \
  --width 1920 \
  --height 1080 \
  --k-tune-mode fx-fy \
  --alt-iters 2 \
  --opt-stride 8 \
  --max-samples 60000 \
  --k-max-samples 12000 \
  --loss soft_l1 \
  --f-scale 1.0 \
  --max-nfev 80 \
  --k-max-nfev 60 \
  --epe-stride 4 \
  --progress-every 1 \
  --overwrite





