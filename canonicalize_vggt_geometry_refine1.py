#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gop_camera_refine_rf_from_structure_h.py

Second-stage refinement for the output of:
  optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py

Expected first-stage outputs from that script:
  <prefix>_fixedK_gop_nn_geometry.npz
    - K_fixed                         [3,3]
    - rvec_abs_final                  [N,3]  camera_from_world / W2C
    - tvec_abs_final                  [N,3]  camera_from_world / W2C
    - depth_canonical                 [N,H,W] linear depth
    - frame_indices                   [N]
    - pairs_json                      optional JSON list of PairSpec dicts

  <prefix>_fixedK_gop_nn_cam.jsonl    optional, used only to copy metadata
  <prefix>_fixedK_gop_nn_depth_linear_yuv420p10le.yuv  unchanged depth output

What this script does:
  1) Use fixed depth_canonical and final W2C poses from the first-stage NPZ.
  2) For each target/ref pair, render the current camera projection.
  3) Estimate a pair-wise global residual transform using OpenCV ECC on Scharr
     structure images. This is preprocessing-only and may use OpenCV.
  4) Build pseudo-GT correspondences:
        q_gt(x,y) = base_cam_map(x,y) + alpha * clamp(H(x,y)-x,y)
  5) Fit only GOP focal and frame-wise W2C R|t to q_gt, with depth fixed.
     R and focal are the main variables; t is strongly regularized/tiny.

Pose convention in this script:
  Absolute pose is camera_from_world / W2C:
      X_cam_i = R_i X_world + t_i

  Relative target camera -> reference camera:
      R_rel = R_ref R_tar^T
      t_rel = t_ref - R_rel t_tar
      X_ref = R_rel X_tar + t_rel

Coordinate convention:
  target pixel -> reference pixel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("This script requires PyTorch.") from exc


# ============================================================
# Basic I/O
# ============================================================

def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def frame_size_yuv420(w: int, h: int, bitdepth: int) -> int:
    bps = 1 if bitdepth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def read_y_frame(path: str | Path, w: int, h: int, bitdepth: int, idx: int) -> np.ndarray:
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    fs = frame_size_yuv420(w, h, bitdepth)
    y_samples = w * h
    with open(path, "rb") as f:
        f.seek(int(idx) * fs)
        y = np.fromfile(f, dtype=dtype, count=y_samples)
    if y.size != y_samples:
        raise RuntimeError(f"Cannot read Y frame idx={idx} from {path}")
    return y.reshape(h, w)


def write_yuv420_y_only(path: str | Path, y: np.ndarray, bitdepth: int) -> None:
    h, w = y.shape
    with open(path, "wb") as f:
        if bitdepth <= 8:
            yy = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            uv = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        else:
            yy = np.clip(np.rint(y), 0, (1 << bitdepth) - 1).astype("<u2")
            uv = np.full((h // 2, w // 2), 1 << (bitdepth - 1), dtype="<u2")
        f.write(yy.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())


def save_gray_png(path: str | Path, y: np.ndarray, bitdepth: int) -> None:
    if bitdepth <= 8:
        out = np.clip(y, 0, 255).astype(np.uint8)
    else:
        out = np.clip(y.astype(np.float32) / float(1 << (bitdepth - 8)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def calc_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, bitdepth: int) -> dict[str, Any]:
    valid = valid.astype(bool)
    valid_ratio = float(np.mean(valid))
    if not np.any(valid):
        return {"valid_ratio": valid_ratio, "mae": None, "mse": None, "psnr": None}
    diff = target_y.astype(np.float32)[valid] - pred_y.astype(np.float32)[valid]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    maxv = float((1 << bitdepth) - 1)
    psnr = 999.0 if mse <= 1e-12 else float(10.0 * np.log10((maxv * maxv) / mse))
    return {"valid_ratio": valid_ratio, "mae": mae, "mse": mse, "psnr": psnr}


def _json_from_np_object(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return _json_from_np_object(x.item())
        return [_json_from_np_object(v) for v in x.tolist()]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, str):
        return json.loads(x)
    return x


def load_first_jsonl_object(path: str | Path) -> Optional[dict[str, Any]]:
    if path is None or not str(path):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict):
                return obj
    return None


# ============================================================
# First-stage fixedK_gop_nn output loading
# ============================================================

def _npz_key(data: np.lib.npyio.NpzFile, candidates: list[str], required: bool = True) -> Optional[str]:
    for k in candidates:
        if k in data.files:
            return k
    if required:
        raise KeyError(f"NPZ missing any of keys: {candidates}. Available keys: {data.files}")
    return None


def load_fixedk_stage1_npz(npz_path: str | Path) -> dict[str, Any]:
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)

    k_key = _npz_key(data, ["K_fixed", "K_refined", "K"])
    r_key = _npz_key(data, ["rvec_abs_final", "rvec_abs_stage4_smooth", "rvec_abs_stage3_joint", "rvec_abs_stage2_t_nn", "rvec_abs_stage1_rt"])
    t_key = _npz_key(data, ["tvec_abs_final", "tvec_abs_stage4_smooth", "tvec_abs_stage3_joint", "tvec_abs_stage2_t_nn", "tvec_abs_stage1_rt"])
    d_key = _npz_key(data, ["depth_canonical", "depth_original"])

    K = np.asarray(data[k_key], dtype=np.float64).reshape(3, 3)
    rvecs = np.asarray(data[r_key], dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(data[t_key], dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(data[d_key], dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"depth must be [N,H,W], got {depth.shape}")
    n, h, w = depth.shape
    if rvecs.shape[0] != n or tvecs.shape[0] != n:
        raise ValueError(f"Pose count mismatch: depth N={n}, r={rvecs.shape}, t={tvecs.shape}")

    if "frame_indices" in data.files:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.shape[0] != n:
            raise ValueError(f"frame_indices count mismatch: {frame_indices.shape[0]} vs N={n}")
    else:
        frame_indices = np.arange(n, dtype=np.int32)

    pairs = None
    if "pairs_json" in data.files:
        try:
            pairs_obj = _json_from_np_object(data["pairs_json"])
            if isinstance(pairs_obj, list):
                pairs = []
                for p in pairs_obj:
                    if isinstance(p, dict) and "target" in p and "ref" in p:
                        pairs.append((int(p["target"]), int(p["ref"]), float(p.get("weight", 1.0)), str(p.get("kind", "stage1"))))
        except Exception:
            pairs = None

    return {
        "npz_path": str(path),
        "K": K,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "depth": depth,
        "frame_indices": frame_indices,
        "pairs": pairs,
        "source_keys": {"K": k_key, "rvecs": r_key, "tvecs": t_key, "depth": d_key},
    }


def parse_pairs(s: str, default_weight: float = 1.0) -> list[tuple[int, int, float, str]]:
    out: list[tuple[int, int, float, str]] = []
    if not s or not s.strip():
        return out
    for tok in re.split(r"[,;\s]+", s.strip()):
        if not tok:
            continue
        tok = tok.replace("->", ":")
        parts = tok.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid pair token '{tok}'. Use target:ref[:weight].")
        t = int(parts[0])
        r = int(parts[1])
        w = float(parts[2]) if len(parts) == 3 else float(default_weight)
        out.append((t, r, w, "cli"))
    return out


def generate_dyadic_pairs(n: int, bidirectional: bool = True, weight: float = 1.0) -> list[tuple[int, int, float, str]]:
    acc: dict[tuple[int, int], tuple[int, int, float, str]] = {}

    def add(t: int, r: int, w: float, kind: str) -> None:
        if t == r or not (0 <= t < n and 0 <= r < n):
            return
        key = (int(t), int(r))
        if key in acc:
            old = acc[key]
            acc[key] = (old[0], old[1], old[2] + float(w), old[3] + "+" + kind)
        else:
            acc[key] = (key[0], key[1], float(w), kind)

    def rec(a: int, b: int, level: int) -> None:
        if b <= a + 1:
            return
        m = (a + b) // 2
        ww = float(weight) / math.sqrt(level + 1.0)
        add(m, a, ww, f"dyadic_L{level}")
        add(m, b, ww, f"dyadic_L{level}")
        if bidirectional:
            add(a, m, ww, f"dyadic_rev_L{level}")
            add(b, m, ww, f"dyadic_rev_L{level}")
        rec(a, m, level + 1)
        rec(m, b, level + 1)

    rec(0, n - 1, 0)
    return sorted(acc.values(), key=lambda x: (abs(x[0] - x[1]), x[0], x[1]))


def build_pair_list(args: argparse.Namespace, stage: dict[str, Any]) -> list[tuple[int, int, float, str]]:
    n = int(stage["depth"].shape[0])
    if args.pairs.strip():
        pairs = parse_pairs(args.pairs)
    elif args.pair_source == "npz" and stage.get("pairs"):
        pairs = list(stage["pairs"])
    elif args.pair_source in ("dyadic", "npz"):
        pairs = generate_dyadic_pairs(n, bidirectional=not args.no_bidirectional_pairs, weight=args.pair_weight)
    elif args.pair_source == "all":
        pairs = [(t, r, args.pair_weight, "all") for t in range(n) for r in range(n) if t != r]
    else:
        raise ValueError(args.pair_source)

    checked = []
    seen = set()
    for t, r, w, kind in pairs:
        if not (0 <= int(t) < n and 0 <= int(r) < n):
            raise ValueError(f"Pair out of range for N={n}: {t}->{r}")
        key = (int(t), int(r))
        if key in seen:
            continue
        seen.add(key)
        checked.append((int(t), int(r), float(w), str(kind)))
    if args.max_pairs > 0:
        checked = checked[: int(args.max_pairs)]
    if not checked:
        raise RuntimeError("No pairs selected.")
    return checked


# ============================================================
# Geometry / projection, W2C convention
# ============================================================

def rodrigues_np(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def all_rotation_matrices_np(rvecs: np.ndarray) -> np.ndarray:
    return np.stack([rodrigues_np(r) for r in np.asarray(rvecs)], axis=0)


def torch_rodrigues(rvecs: torch.Tensor) -> torch.Tensor:
    dtype = rvecs.dtype
    device = rvecs.device
    n = rvecs.shape[0]
    x, y, z = rvecs[:, 0], rvecs[:, 1], rvecs[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    theta2 = torch.sum(rvecs * rvecs, dim=-1)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-30))
    small = theta2 < 1e-12
    A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta)
    B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2)
    I = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
    return I + A[:, None, None] * K + B[:, None, None] * (K @ K)


def camera_map_w2c_np(
    target: int,
    ref: int,
    width: int,
    height: int,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    depth_img: np.ndarray,
    z_sign: float,
    z_min: float,
    row_batch: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rs = all_rotation_matrices_np(rvecs)
    R_tar = Rs[int(target)]
    R_ref = Rs[int(ref)]
    t_tar = tvecs[int(target)]
    t_ref = tvecs[int(ref)]

    R_rel = R_ref @ R_tar.T
    t_rel = t_ref - R_rel @ t_tar

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)
    xs_full = np.arange(width, dtype=np.float64)

    for y0 in range(0, height, max(1, int(row_batch))):
        y1 = min(height, y0 + int(row_batch))
        ys = np.arange(y0, y1, dtype=np.float64)
        xs, yy = np.meshgrid(xs_full, ys)
        ray_x = (xs - cx) / fx
        ray_y = (yy - cy) / fy
        rays = np.stack([ray_x.reshape(-1), ray_y.reshape(-1), np.full((y1 - y0) * width, float(z_sign), dtype=np.float64)], axis=1)
        dep = depth_img[y0:y1, :].reshape(-1).astype(np.float64)
        X_tar = dep[:, None] * rays
        X_ref = X_tar @ R_rel.T + t_rel[None, :]
        z = X_ref[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, np.where(z >= 0, 1e-9, -1e-9))
        mx = fx * (X_ref[:, 0] / z_safe) + cx
        my = fy * (X_ref[:, 1] / z_safe) + cy
        valid = (
            np.isfinite(mx) & np.isfinite(my) & np.isfinite(dep)
            & (dep > 0.0)
            & (z * float(z_sign) > float(z_min))
            & (mx >= 0.0) & (mx <= width - 1.0)
            & (my >= 0.0) & (my <= height - 1.0)
        )
        map_x[y0:y1, :] = mx.reshape(y1 - y0, width).astype(np.float32)
        map_y[y0:y1, :] = my.reshape(y1 - y0, width).astype(np.float32)
        valid_all[y0:y1, :] = valid.reshape(y1 - y0, width)

    map_x[~valid_all] = -1.0
    map_y[~valid_all] = -1.0
    return map_x, map_y, valid_all


def remap_y(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# Structure ECC residual transform
# ============================================================

def normalize_for_ecc(img: np.ndarray, bitdepth: int) -> np.ndarray:
    return np.clip(img.astype(np.float32) / float((1 << bitdepth) - 1), 0.0, 1.0)


def make_structure_image(img: np.ndarray, bitdepth: int, mode: str, log_gain: float, pre_blur: int) -> np.ndarray:
    y = normalize_for_ecc(img, bitdepth)
    if int(pre_blur) > 0:
        k = 2 * int(pre_blur) + 1
        y = cv2.GaussianBlur(y, (k, k), 0)
    gx = cv2.Scharr(y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(y, cv2.CV_32F, 0, 1)
    if mode == "scharr_mag":
        s = np.sqrt(gx * gx + gy * gy)
    elif mode == "scharr_l1":
        s = np.abs(gx) + np.abs(gy)
    elif mode == "scharr_x":
        s = np.abs(gx)
    elif mode == "scharr_y":
        s = np.abs(gy)
    elif mode == "scharr_x_weighted":
        s = 0.75 * np.abs(gx) + 0.25 * np.abs(gy)
    else:
        raise ValueError(mode)
    if float(log_gain) > 0:
        s = np.log1p(float(log_gain) * s)
    m = float(np.max(s))
    if m > 1e-8:
        s = s / m
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def make_valid_mask_u8(valid: np.ndarray, erode: int = 2) -> np.ndarray:
    mask = (valid.astype(np.uint8) * 255)
    if int(erode) > 0:
        k = 2 * int(erode) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def make_structure_mask_u8(structure: np.ndarray, base_mask_u8: np.ndarray, keep_percent: float, dilate: int) -> tuple[np.ndarray, dict[str, Any]]:
    base = base_mask_u8 > 0
    vals = structure[base]
    stats: dict[str, Any] = {
        "base_count": int(np.count_nonzero(base)),
        "keep_percent": float(keep_percent),
        "threshold": None,
        "structure_count": 0,
        "final_count": 0,
    }
    if vals.size < 100:
        return base_mask_u8.copy(), stats
    percentile = 100.0 - float(np.clip(keep_percent, 0.1, 100.0))
    thr = float(np.percentile(vals, percentile))
    mask = base & (structure >= thr)
    stats["threshold"] = thr
    stats["structure_count"] = int(np.count_nonzero(mask))
    mask_u8 = mask.astype(np.uint8) * 255
    if int(dilate) > 0:
        k = 2 * int(dilate) + 1
        mask_u8 = cv2.dilate(mask_u8, np.ones((k, k), dtype=np.uint8), iterations=1)
        mask_u8 = np.where(base, mask_u8, 0).astype(np.uint8)
    stats["final_count"] = int(np.count_nonzero(mask_u8))
    return mask_u8, stats


def identity_transform(cp_num: int) -> np.ndarray:
    return np.eye(3, dtype=np.float32) if int(cp_num) == 4 else np.eye(2, 3, dtype=np.float32)


def warp_ecc_input_to_template_domain(inp: np.ndarray, M: np.ndarray, cp_num: int) -> np.ndarray:
    h, w = inp.shape
    if int(cp_num) == 4:
        return cv2.warpPerspective(
            inp.astype(np.float32), np.asarray(M, dtype=np.float32), (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ).astype(np.float32)
    return cv2.warpAffine(
        inp.astype(np.float32), np.asarray(M, dtype=np.float32), (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ).astype(np.float32)


def run_find_transform_ecc_with_init(
    template: np.ndarray,
    inp: np.ndarray,
    mask_u8: np.ndarray,
    cp_num: int,
    init_matrix: np.ndarray,
    max_iters: int,
    eps: float,
    gauss_filt_size: int,
) -> tuple[np.ndarray, float]:
    cp_num = int(cp_num)
    if cp_num == 4:
        motion_type = cv2.MOTION_HOMOGRAPHY
        warp = np.asarray(init_matrix, dtype=np.float32).reshape(3, 3).copy()
    else:
        motion_type = cv2.MOTION_AFFINE
        warp = np.asarray(init_matrix, dtype=np.float32).reshape(2, 3).copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    cc, warp = cv2.findTransformECC(
        templateImage=template.astype(np.float32),
        inputImage=inp.astype(np.float32),
        warpMatrix=warp,
        motionType=motion_type,
        criteria=criteria,
        inputMask=mask_u8.astype(np.uint8),
        gaussFiltSize=int(gauss_filt_size),
    )
    return warp.astype(np.float32), float(cc)


def refine_mask_by_structure_residual(
    template: np.ndarray,
    inp: np.ndarray,
    M: np.ndarray,
    current_mask_u8: np.ndarray,
    base_mask_u8: np.ndarray,
    cp_num: int,
    keep_percent: float,
    min_mask_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = current_mask_u8 > 0
    base = base_mask_u8 > 0
    use = current & base
    stats: dict[str, Any] = {
        "input_count": int(np.count_nonzero(use)),
        "output_count": int(np.count_nonzero(use)),
        "keep_percent": float(keep_percent),
        "threshold": None,
        "mean_residual": None,
        "median_residual": None,
        "p90_residual": None,
    }
    if np.count_nonzero(use) < int(min_mask_count):
        return current_mask_u8.copy(), stats
    warped = warp_ecc_input_to_template_domain(inp, M, cp_num=cp_num)
    residual = np.abs(template.astype(np.float32) - warped.astype(np.float32))
    vals = residual[use]
    if vals.size < int(min_mask_count):
        return current_mask_u8.copy(), stats
    keep_percent = max(0.1, min(100.0, float(keep_percent)))
    thr = float(np.percentile(vals, keep_percent))
    new_mask = use & (residual <= thr)
    if np.count_nonzero(new_mask) < int(min_mask_count):
        return current_mask_u8.copy(), stats
    stats.update({
        "output_count": int(np.count_nonzero(new_mask)),
        "threshold": thr,
        "mean_residual": float(np.mean(vals)),
        "median_residual": float(np.median(vals)),
        "p90_residual": float(np.percentile(vals, 90.0)),
    })
    return (new_mask.astype(np.uint8) * 255), stats


def estimate_pair_structure_ecc_stable(
    target_y: np.ndarray,
    cam_warp_y: np.ndarray,
    valid_mask_u8: np.ndarray,
    bitdepth: int,
    cp_num: int,
    structure_mode: str,
    keep_percent: float,
    mask_dilate: int,
    log_gain: float,
    pre_blur: int,
    ecc_iters: int,
    ecc_eps: float,
    ecc_gauss: int,
    ecc_rounds: int,
    residual_keep_percent: float,
    min_mask_count: int,
) -> tuple[np.ndarray, bool, Optional[float], np.ndarray, dict[str, Any], np.ndarray]:
    template = make_structure_image(target_y, bitdepth, structure_mode, log_gain, pre_blur)
    inp = make_structure_image(cam_warp_y, bitdepth, structure_mode, log_gain, pre_blur)
    mask_u8, mask_stats = make_structure_mask_u8(template, valid_mask_u8, keep_percent, mask_dilate)

    stats: dict[str, Any] = {
        "structure_mode": structure_mode,
        "keep_percent": float(keep_percent),
        "mask_dilate": int(mask_dilate),
        "mask": mask_stats,
        "rounds": [],
    }
    if np.count_nonzero(mask_u8) < int(min_mask_count):
        return identity_transform(cp_num), False, None, mask_u8, stats, template

    M = identity_transform(cp_num)
    score: Optional[float] = None
    success = False
    base_mask_u8 = mask_u8.copy()
    for r in range(max(1, int(ecc_rounds))):
        round_stats: dict[str, Any] = {"round": int(r), "mask_count_before": int(np.count_nonzero(mask_u8)), "success": False}
        try:
            M, score = run_find_transform_ecc_with_init(
                template=template,
                inp=inp,
                mask_u8=mask_u8,
                cp_num=cp_num,
                init_matrix=M,
                max_iters=ecc_iters,
                eps=ecc_eps,
                gauss_filt_size=ecc_gauss,
            )
            success = True
            round_stats["success"] = True
            round_stats["ecc_cc"] = float(score)
        except cv2.error as exc:
            round_stats["cv2_error"] = str(exc)
            stats["rounds"].append(round_stats)
            break

        if r < max(1, int(ecc_rounds)) - 1:
            mask_u8, res_stats = refine_mask_by_structure_residual(
                template=template,
                inp=inp,
                M=M,
                current_mask_u8=mask_u8,
                base_mask_u8=base_mask_u8,
                cp_num=cp_num,
                keep_percent=residual_keep_percent,
                min_mask_count=min_mask_count,
            )
            round_stats["residual_refine"] = res_stats
            round_stats["mask_count_after"] = int(np.count_nonzero(mask_u8))
        stats["rounds"].append(round_stats)

    stats["final_mask_count"] = int(np.count_nonzero(mask_u8))
    stats["final_ecc_cc"] = None if score is None else float(score)
    stats["success"] = bool(success)
    if not success:
        return identity_transform(cp_num), False, None, mask_u8, stats, template
    return M.astype(np.float32), True, score, mask_u8, stats, template


def apply_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)
    if M.shape == (2, 3):
        return (homo @ M.T).astype(np.float32)
    if M.shape == (3, 3):
        q = homo @ M.T
        den = q[:, 2:3]
        den = np.where(np.abs(den) < 1e-8, 1e-8, den)
        return (q[:, :2] / den).astype(np.float32)
    raise ValueError(f"bad transform shape: {M.shape}")


def transform_cp_bias(M: np.ndarray, w: int, h: int, cp_num: int) -> np.ndarray:
    if int(cp_num) == 4:
        src = np.asarray([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    elif int(cp_num) == 3:
        src = np.asarray([[0, 0], [w, 0], [0, h]], dtype=np.float32)
    else:
        src = np.asarray([[0, 0], [w, 0]], dtype=np.float32)
    return apply_transform_points(M, src) - src


# ============================================================
# Observation extraction
# ============================================================

def yuv_frame_index_for_poc(poc: int, frame_indices: np.ndarray, args: argparse.Namespace) -> int:
    if args.frame_index_mode == "frame_indices":
        return int(args.seq_start) + int(frame_indices[int(poc)])
    return int(args.seq_start) + int(poc)


def collect_pair_observations(
    pair: tuple[int, int, float, str],
    seq_yuv: str,
    width: int,
    height: int,
    bitdepth: int,
    frame_indices: np.ndarray,
    K_base: np.ndarray,
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    depth: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    pair_out_dir: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target, ref, pair_weight, kind = pair
    target = int(target)
    ref = int(ref)
    tar_yuv_idx = yuv_frame_index_for_poc(target, frame_indices, args)
    ref_yuv_idx = yuv_frame_index_for_poc(ref, frame_indices, args)
    target_y = read_y_frame(seq_yuv, width, height, bitdepth, tar_yuv_idx)
    ref_y = read_y_frame(seq_yuv, width, height, bitdepth, ref_yuv_idx)
    depth_img = np.asarray(depth[target], dtype=np.float32)

    map_x, map_y, valid = camera_map_w2c_np(
        target=target,
        ref=ref,
        width=width,
        height=height,
        K=K_base,
        rvecs=rvecs_base,
        tvecs=tvecs_base,
        depth_img=depth_img,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.render_row_batch,
    )
    cam_warp = remap_y(ref_y, map_x, map_y)
    base_mask_u8 = make_valid_mask_u8(valid, erode=args.ecc_valid_erode)
    M, success, cc, ecc_mask_u8, ecc_stats, structure = estimate_pair_structure_ecc_stable(
        target_y=target_y,
        cam_warp_y=cam_warp,
        valid_mask_u8=base_mask_u8,
        bitdepth=bitdepth,
        cp_num=args.ecc_cp_num,
        structure_mode=args.structure_mode,
        keep_percent=args.structure_keep_percent,
        mask_dilate=args.structure_mask_dilate,
        log_gain=args.structure_log_gain,
        pre_blur=args.structure_pre_blur,
        ecc_iters=args.ecc_iters,
        ecc_eps=args.ecc_eps,
        ecc_gauss=args.ecc_gauss,
        ecc_rounds=args.structure_ecc_rounds,
        residual_keep_percent=args.structure_residual_keep_percent,
        min_mask_count=args.ecc_min_mask_count,
    )

    ys, xs = np.where(ecc_mask_u8 > 0)
    n_mask = int(xs.size)
    cp_bias = transform_cp_bias(M, width, height, args.ecc_cp_num).astype(np.float32)

    if (not success) or n_mask < args.min_obs_per_pair:
        info = {
            "target": target,
            "ref": ref,
            "target_yuv_idx": int(tar_yuv_idx),
            "ref_yuv_idx": int(ref_yuv_idx),
            "pair_weight": float(pair_weight),
            "kind": kind,
            "success": bool(success),
            "ecc_cc": None if cc is None else float(cc),
            "num_mask_pixels": n_mask,
            "num_observations": 0,
            "cp_bias_raw": cp_bias.astype(float).tolist(),
            "ecc_stats": ecc_stats,
        }
        return {"target": np.empty(0, np.int32)}, info

    if args.max_obs_per_pair > 0 and xs.size > args.max_obs_per_pair:
        sel = rng.choice(xs.size, size=int(args.max_obs_per_pair), replace=False)
        xs = xs[sel]
        ys = ys[sel]

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dst = apply_transform_points(M, pts)
    bias = dst - pts
    bias *= float(args.ecc_alpha)
    if float(args.ecc_bias_max_abs) > 0.0:
        bias = np.clip(bias, -float(args.ecc_bias_max_abs), float(args.ecc_bias_max_abs))

    qx = map_x[ys, xs].astype(np.float32) + bias[:, 0]
    qy = map_y[ys, xs].astype(np.float32) + bias[:, 1]
    dep = depth_img[ys, xs].astype(np.float32)
    ok = (
        np.isfinite(qx) & np.isfinite(qy) & np.isfinite(dep)
        & (dep > 0.0)
        & (map_x[ys, xs] >= 0.0) & (map_y[ys, xs] >= 0.0)
        & (qx >= 0.0) & (qx <= width - 1.0)
        & (qy >= 0.0) & (qy <= height - 1.0)
    )

    xs = xs[ok]
    ys = ys[ok]
    qx = qx[ok]
    qy = qy[ok]
    dep = dep[ok]
    structure_w = 0.25 + structure[ys, xs].astype(np.float32)
    cc_w = 1.0 if cc is None or not np.isfinite(cc) else float(max(0.05, min(2.0, cc + 1.0)))
    weights = structure_w * cc_w * float(pair_weight)

    obs = {
        "target": np.full(xs.shape[0], target, dtype=np.int32),
        "ref": np.full(xs.shape[0], ref, dtype=np.int32),
        "px": xs.astype(np.float32),
        "py": ys.astype(np.float32),
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "depth": dep.astype(np.float32),
        "weight": weights.astype(np.float32),
    }

    cost_cam = calc_cost(target_y, cam_warp, valid, bitdepth)
    if pair_out_dir is not None:
        ensure_dir(pair_out_dir)
        tag = f"t{target:03d}_r{ref:03d}"
        if not args.no_pair_debug_yuv:
            write_yuv420_y_only(os.path.join(pair_out_dir, f"cam_base_{tag}.yuv"), cam_warp, bitdepth)
        save_gray_png(os.path.join(pair_out_dir, f"cam_base_{tag}.png"), cam_warp, bitdepth)
        mask_vis = np.where(ecc_mask_u8 > 0, (1 << bitdepth) - 1, 0).astype(np.float32)
        save_gray_png(os.path.join(pair_out_dir, f"ecc_mask_{tag}.png"), mask_vis, bitdepth)

    info = {
        "target": target,
        "ref": ref,
        "target_yuv_idx": int(tar_yuv_idx),
        "ref_yuv_idx": int(ref_yuv_idx),
        "pair_weight": float(pair_weight),
        "kind": kind,
        "success": bool(success),
        "ecc_cc": None if cc is None else float(cc),
        "motion_type": "homography" if int(args.ecc_cp_num) == 4 else "affine",
        "num_mask_pixels": n_mask,
        "num_observations": int(xs.shape[0]),
        "cp_bias_raw": cp_bias.astype(float).tolist(),
        "ecc_alpha": float(args.ecc_alpha),
        "ecc_bias_max_abs": float(args.ecc_bias_max_abs),
        "matrix": np.asarray(M, dtype=float).tolist(),
        "base_cam_cost": cost_cam,
        "ecc_stats": ecc_stats,
    }
    return obs, info


def concat_observations(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ["target", "ref", "px", "py", "qx", "qy", "depth", "weight"]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [o[k] for o in obs_list if k in o and o[k].size > 0]
        if not vals:
            raise RuntimeError(f"No observations for key {k}")
        out[k] = np.concatenate(vals, axis=0)
    return out


# ============================================================
# Fitting: W2C R + GOP focal + tiny t, fixed depth
# ============================================================

def choose_batch_indices(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    if batch <= 0 or batch >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=int(batch), replace=False).astype(np.int64)


def robust_loss_from_err2(err2: torch.Tensor, loss_name: str, f_scale: float) -> torch.Tensor:
    f = float(max(f_scale, 1e-6))
    if loss_name == "linear":
        return err2
    if loss_name == "huber":
        err = torch.sqrt(err2.clamp_min(1e-12))
        return torch.where(err <= f, 0.5 * err * err, f * (err - 0.5 * f))
    if loss_name == "cauchy":
        return (f * f) * torch.log1p(err2 / (f * f))
    return 2.0 * (f * f) * (torch.sqrt(1.0 + err2 / (f * f)) - 1.0)


def fit_rf_tiny_t_w2c(
    observations: dict[str, np.ndarray],
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    K_base: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.torch_float64 else torch.float32
    rng = np.random.default_rng(int(args.seed))
    n_obs = int(observations["px"].shape[0])
    n_frames = int(rvecs_base.shape[0])

    target = torch.tensor(observations["target"], device=device, dtype=torch.long)
    ref = torch.tensor(observations["ref"], device=device, dtype=torch.long)
    px = torch.tensor(observations["px"], device=device, dtype=dtype)
    py = torch.tensor(observations["py"], device=device, dtype=dtype)
    qx = torch.tensor(observations["qx"], device=device, dtype=dtype)
    qy = torch.tensor(observations["qy"], device=device, dtype=dtype)
    depth = torch.tensor(observations["depth"], device=device, dtype=dtype).clamp_min(1e-12)

    w_np = observations["weight"].astype(np.float64)
    good = np.isfinite(w_np) & (w_np > 0)
    med = float(np.median(w_np[good])) if np.any(good) else 1.0
    w_np = np.clip(w_np / max(med, 1e-12), 1e-4, 100.0).astype(np.float32)
    weight = torch.tensor(w_np, device=device, dtype=dtype)

    r_base = torch.tensor(rvecs_base, device=device, dtype=dtype)
    t_base = torch.tensor(tvecs_base, device=device, dtype=dtype)
    r_delta = torch.nn.Parameter(torch.zeros_like(r_base))
    t_delta = torch.nn.Parameter(torch.zeros_like(t_base))
    anchor = int(args.anchor_poc)
    if anchor < 0:
        anchor = 0
    if not (0 <= anchor < n_frames):
        raise ValueError(f"--anchor-poc {anchor} out of range N={n_frames}")

    f_base_x = float(K_base[0, 0])
    f_base_y = float(K_base[1, 1])
    if args.f_init == "geom":
        f0 = math.sqrt(max(f_base_x * f_base_y, 1e-12))
    elif args.f_init == "fx":
        f0 = f_base_x
    elif args.f_init == "fy":
        f0 = f_base_y
    else:
        f0 = 0.5 * (f_base_x + f_base_y)

    focal_mode = str(args.focal_mode)
    log_f_delta: Optional[torch.nn.Parameter]
    if focal_mode == "single":
        log_f_delta = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        params = [
            {"params": [r_delta], "lr": float(args.lr_rot)},
            {"params": [t_delta], "lr": float(args.lr_trans)},
            {"params": [log_f_delta], "lr": float(args.lr_focal)},
        ]
    elif focal_mode == "separate":
        log_f_delta = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        params = [
            {"params": [r_delta], "lr": float(args.lr_rot)},
            {"params": [t_delta], "lr": float(args.lr_trans)},
            {"params": [log_f_delta], "lr": float(args.lr_focal)},
        ]
    elif focal_mode == "fixed":
        log_f_delta = None
        params = [
            {"params": [r_delta], "lr": float(args.lr_rot)},
            {"params": [t_delta], "lr": float(args.lr_trans)},
        ]
    else:
        raise ValueError(focal_mode)

    if args.freeze_t:
        t_delta.requires_grad_(False)
        params = [{"params": [r_delta], "lr": float(args.lr_rot)}]
        if log_f_delta is not None:
            params.append({"params": [log_f_delta], "lr": float(args.lr_focal)})
    if args.freeze_r:
        r_delta.requires_grad_(False)
        params = [g for g in params if g["params"][0] is not r_delta]
    if not params:
        raise RuntimeError("No trainable parameters: check --freeze-r/--freeze-t/--focal-mode fixed.")

    opt = torch.optim.Adam(params)
    cx = torch.tensor(float(K_base[0, 2]), device=device, dtype=dtype)
    cy = torch.tensor(float(K_base[1, 2]), device=device, dtype=dtype)

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor] = 0.0
        td[anchor] = 0.0
        r_cur = r_base + rd
        t_cur = t_base + td
        if focal_mode == "fixed":
            fx = torch.tensor(f_base_x, device=device, dtype=dtype)
            fy = torch.tensor(f_base_y, device=device, dtype=dtype)
        elif focal_mode == "single":
            assert log_f_delta is not None
            f = torch.tensor(float(f0), device=device, dtype=dtype) * torch.exp(log_f_delta[0])
            fx = f
            fy = f
        else:
            assert log_f_delta is not None
            fx = torch.tensor(f_base_x, device=device, dtype=dtype) * torch.exp(log_f_delta[0])
            fy = torch.tensor(f_base_y, device=device, dtype=dtype) * torch.exp(log_f_delta[1])
        return r_cur, t_cur, fx, fy

    def project_indices(idx_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        r_cur, t_cur, fx, fy = current_params()
        R = torch_rodrigues(r_cur)
        tc = target[idx]
        rc = ref[idx]
        R_rel = torch.bmm(R[rc], R[tc].transpose(1, 2))
        t_rel = t_cur[rc] - torch.bmm(R_rel, t_cur[tc].unsqueeze(-1)).squeeze(-1)
        ray_x = (px[idx] - cx) / fx
        ray_y = (py[idx] - cy) / fy
        ray_z = torch.full_like(ray_x, float(args.z_sign))
        X_tar = depth[idx, None] * torch.stack([ray_x, ray_y, ray_z], dim=1)
        X_ref = torch.bmm(R_rel, X_tar.unsqueeze(-1)).squeeze(-1) + t_rel
        z = X_ref[:, 2]
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        denom = torch.where(torch.abs(z) > eps, z, torch.where(z >= 0, eps, -eps))
        u = fx * (X_ref[:, 0] / denom) + cx
        v = fy * (X_ref[:, 1] / denom) + cy
        return u, v, z

    def regularization() -> torch.Tensor:
        reg = torch.zeros((), dtype=dtype, device=device)
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor] = 0.0
        td[anchor] = 0.0
        if args.rot_delta_prior_weight > 0:
            reg = reg + float(args.rot_delta_prior_weight) * torch.mean(rd * rd)
        if args.trans_delta_prior_weight > 0 and not args.freeze_t:
            reg = reg + float(args.trans_delta_prior_weight) * torch.mean(td * td)
        if args.pose_delta_smooth_weight > 0 and n_frames >= 2:
            reg = reg + float(args.pose_delta_smooth_weight) * (torch.mean((rd[1:] - rd[:-1]) ** 2) + torch.mean((td[1:] - td[:-1]) ** 2))
        if log_f_delta is not None and args.f_prior_weight > 0:
            reg = reg + float(args.f_prior_weight) * torch.mean(log_f_delta * log_f_delta)
        return reg

    def loss_for_batch(idx_np: np.ndarray) -> torch.Tensor:
        u, v, z = project_indices(idx_np)
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        dx = u - qx[idx]
        dy = v - qy[idx]
        err2 = dx * dx + dy * dy
        pix = robust_loss_from_err2(err2, args.robust_loss, args.robust_f_scale)
        if args.z_min > 0:
            zbad = torch.relu(float(args.z_min) - z * float(args.z_sign))
            pix = pix + float(args.z_penalty) * zbad * zbad
        ww = weight[idx]
        return torch.sum(ww * pix) / (torch.sum(ww) + 1e-12) + regularization()

    @torch.no_grad()
    def eval_all(batch: int) -> np.ndarray:
        out = np.full(n_obs, np.inf, dtype=np.float64)
        for s in range(0, n_obs, int(batch)):
            e = min(n_obs, s + int(batch))
            idx_np = np.arange(s, e, dtype=np.int64)
            u, v, z = project_indices(idx_np)
            idx = torch.tensor(idx_np, device=device, dtype=torch.long)
            err = torch.sqrt((u - qx[idx]) ** 2 + (v - qy[idx]) ** 2).detach().cpu().numpy()
            zz = z.detach().cpu().numpy()
            err[zz * float(args.z_sign) <= args.z_min] = np.inf
            out[s:e] = err
        return out

    report: dict[str, Any] = {
        "device": str(device),
        "dtype": str(dtype),
        "num_observations": int(n_obs),
        "anchor_poc": int(anchor),
        "focal_mode": focal_mode,
        "f_init": args.f_init,
        "f0": float(f0),
        "iterations": [],
    }

    for step in range(int(args.steps)):
        idx = choose_batch_indices(n_obs, int(args.batch_size), rng)
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(idx)
        loss.backward()
        if args.grad_clip > 0:
            train_params = [p for g in params for p in g["params"] if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(train_params, float(args.grad_clip))
        opt.step()
        with torch.no_grad():
            r_delta[anchor].zero_()
            t_delta[anchor].zero_()
            if args.max_trans_delta > 0 and not args.freeze_t:
                t_delta.clamp_(-float(args.max_trans_delta), float(args.max_trans_delta))
            if log_f_delta is not None:
                log_f_delta.clamp_(-float(args.f_log_max_delta), float(args.f_log_max_delta))

        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            err = eval_all(batch=int(args.eval_batch_size))
            finite = np.isfinite(err)
            stat = {
                "count": int(np.count_nonzero(finite)),
                "mean": float(np.mean(err[finite])) if np.any(finite) else None,
                "median": float(np.median(err[finite])) if np.any(finite) else None,
                "p90": float(np.percentile(err[finite], 90)) if np.any(finite) else None,
                "p95": float(np.percentile(err[finite], 95)) if np.any(finite) else None,
            }
            _, _, fx_cur, fy_cur = current_params()
            info = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "residual_px": stat,
                "fx": float(fx_cur.detach().cpu()),
                "fy": float(fy_cur.detach().cpu()),
                "max_abs_r_delta": float(torch.max(torch.abs(r_delta)).detach().cpu()),
                "max_abs_t_delta": float(torch.max(torch.abs(t_delta)).detach().cpu()),
            }
            report["iterations"].append(info)
            print("[RF REFINE]")
            print(json.dumps(info, indent=2))

    final_err = eval_all(batch=int(args.eval_batch_size))
    finite = np.isfinite(final_err)
    report["final_residual_px"] = {
        "count": int(np.count_nonzero(finite)),
        "mean": float(np.mean(final_err[finite])) if np.any(finite) else None,
        "median": float(np.median(final_err[finite])) if np.any(finite) else None,
        "p90": float(np.percentile(final_err[finite], 90)) if np.any(finite) else None,
        "p95": float(np.percentile(final_err[finite], 95)) if np.any(finite) else None,
    }

    with torch.no_grad():
        r_final, t_final, fx_final, fy_final = current_params()
        K_final = np.array([
            [float(fx_final.detach().cpu()), 0.0, float(K_base[0, 2])],
            [0.0, float(fy_final.detach().cpu()), float(K_base[1, 2])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    return K_final, r_final.detach().cpu().numpy().astype(np.float64), t_final.detach().cpu().numpy().astype(np.float64), report


# ============================================================
# Rendering and final outputs
# ============================================================

def render_refined_pairs(
    pairs: list[tuple[int, int, float, str]],
    seq_yuv: str,
    width: int,
    height: int,
    bitdepth: int,
    frame_indices: np.ndarray,
    K_base: np.ndarray,
    r_base: np.ndarray,
    t_base: np.ndarray,
    K_final: np.ndarray,
    r_final: np.ndarray,
    t_final: np.ndarray,
    depth: np.ndarray,
    out_dir: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    costs = []
    for target, ref, pair_weight, kind in pairs:
        tar_yuv_idx = yuv_frame_index_for_poc(target, frame_indices, args)
        ref_yuv_idx = yuv_frame_index_for_poc(ref, frame_indices, args)
        target_y = read_y_frame(seq_yuv, width, height, bitdepth, tar_yuv_idx)
        ref_y = read_y_frame(seq_yuv, width, height, bitdepth, ref_yuv_idx)
        depth_img = depth[int(target)]
        bx, by, bvalid = camera_map_w2c_np(target, ref, width, height, K_base, r_base, t_base, depth_img, args.z_sign, args.z_min, args.render_row_batch)
        fx, fy, fvalid = camera_map_w2c_np(target, ref, width, height, K_final, r_final, t_final, depth_img, args.z_sign, args.z_min, args.render_row_batch)
        pred_base = remap_y(ref_y, bx, by)
        pred_final = remap_y(ref_y, fx, fy)
        cost_base = calc_cost(target_y, pred_base, bvalid, bitdepth)
        cost_final = calc_cost(target_y, pred_final, fvalid, bitdepth)
        tag = f"t{target:03d}_r{ref:03d}"
        if not args.no_render_yuv:
            write_yuv420_y_only(os.path.join(out_dir, f"pred_refined_{tag}.yuv"), pred_final, bitdepth)
        save_gray_png(os.path.join(out_dir, f"pred_refined_{tag}.png"), pred_final, bitdepth)
        costs.append({
            "target": int(target),
            "ref": int(ref),
            "target_yuv_idx": int(tar_yuv_idx),
            "ref_yuv_idx": int(ref_yuv_idx),
            "pair_weight": float(pair_weight),
            "kind": kind,
            "base_cost": cost_base,
            "refined_cost": cost_final,
            "psnr_gain_vs_base": None if (cost_base["psnr"] is None or cost_final["psnr"] is None) else float(cost_final["psnr"] - cost_base["psnr"]),
        })
        print("[PAIR RENDER COST]")
        print(json.dumps(costs[-1], indent=2))
    return costs


def write_refined_camera_jsonl(
    path: str | Path,
    source_npz: str,
    source_camera_jsonl: Optional[str],
    frame_indices: np.ndarray,
    K_final: np.ndarray,
    r_final: np.ndarray,
    t_final: np.ndarray,
    z_sign: float,
    copied_header: Optional[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    R_all = all_rotation_matrices_np(r_final)
    depth_output = None
    if copied_header is not None:
        depth_output = copied_header.get("depth_output") or copied_header.get("depth_yuv")
    header = {
        "type": "header",
        "format": "fixedK_gop_nn_rf_refine_v1",
        "source_npz": os.path.abspath(source_npz),
        "source_camera_jsonl": os.path.abspath(source_camera_jsonl) if source_camera_jsonl else None,
        "frame_count": int(len(frame_indices)),
        "frame_indices": frame_indices.astype(int).tolist(),
        "intrinsic_mode": "rap_fixed_rf_refined",
        "intrinsic": {
            "fx": float(K_final[0, 0]),
            "fy": float(K_final[1, 1]),
            "cx": float(K_final[0, 2]),
            "cy": float(K_final[1, 2]),
            "z_sign": float(z_sign),
        },
        "intrinsic_delta_order": [],
        "intrinsic_delta_bits_per_frame": 0,
        "pose_storage": {
            "absolute_pose": "camera_from_world / W2C in fixed-K canonical camera coordinates",
            "relative_pair_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target; X_ref=R_rel*X_target+t_rel",
            "adjacent_current_to_previous_fields": "also written for compatibility",
        },
        "depth_output": depth_output,
        "refinement": {
            "description": "R/focal/tiny-t refinement from structure-ECC pair homography pseudo-GT; depth unchanged.",
            "options": {k: v for k, v in vars(args).items() if k not in ("input",)},
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in range(len(frame_indices)):
            rec: dict[str, Any] = {
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec_abs": r_final[i].astype(float).tolist(),
                "tvec_abs": t_final[i].astype(float).tolist(),
                "extrinsic_abs": np.concatenate([R_all[i], t_final[i].reshape(3, 1)], axis=1).astype(float).tolist(),
            }
            if i == 0:
                rec["rvec_current_to_previous"] = [0.0, 0.0, 0.0]
                rec["tvec_current_to_previous"] = [0.0, 0.0, 0.0]
            else:
                R_rel = R_all[i - 1] @ R_all[i].T
                t_rel = t_final[i - 1] - R_rel @ t_final[i]
                rv, _ = cv2.Rodrigues(R_rel.astype(np.float64))
                rec["rvec_current_to_previous"] = rv.reshape(3).astype(float).tolist()
                rec["tvec_current_to_previous"] = t_rel.astype(float).tolist()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_minimal_npz(path: str | Path, stage: dict[str, Any], K_final: np.ndarray, r_final: np.ndarray, t_final: np.ndarray, args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "frame_indices": stage["frame_indices"].astype(np.int32),
        "K_base": stage["K"].astype(np.float32),
        "K_refined": K_final.astype(np.float32),
        "K_fixed": K_final.astype(np.float32),
        "rvec_abs_base": stage["rvecs"].astype(np.float32),
        "tvec_abs_base": stage["tvecs"].astype(np.float32),
        "rvec_abs_refined": r_final.astype(np.float32),
        "tvec_abs_refined": t_final.astype(np.float32),
        "rvec_abs_final": r_final.astype(np.float32),
        "tvec_abs_final": t_final.astype(np.float32),
        "source_stage1_npz": np.asarray(stage["npz_path"], dtype=object),
    }
    if args.save_depth_in_npz:
        payload["depth_canonical"] = stage["depth"].astype(np.float32)
    np.savez(path, **payload)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Source YUV420 sequence")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)
    ap.add_argument("--stage1-npz", "--geometry-npz", required=True, help="*_fixedK_gop_nn_geometry.npz from optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py")
    ap.add_argument("--stage1-camera-jsonl", default="", help="Optional *_fixedK_gop_nn_cam.jsonl, only for metadata copy")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pairs", default="", help="Optional pair list: target:ref[:weight], e.g. 8:0:2,8:16:2")
    ap.add_argument("--pair-source", choices=["npz", "dyadic", "all"], default="npz", help="Default uses pairs_json from NPZ, fallback to dyadic if absent")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--no-bidirectional-pairs", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    # YUV frame indexing.
    ap.add_argument("--seq-start", type=int, default=0)
    ap.add_argument("--frame-index-mode", choices=["local", "frame_indices"], default="local", help="local: YUV idx=seq_start+poc. frame_indices: YUV idx=seq_start+frame_indices[poc]")

    # ECC pair residual extraction.
    ap.add_argument("--ecc-cp-num", type=int, choices=[3, 4], default=4, help="4=homography, 3=affine")
    ap.add_argument("--structure-mode", choices=["scharr_mag", "scharr_l1", "scharr_x", "scharr_y", "scharr_x_weighted"], default="scharr_mag")
    ap.add_argument("--structure-keep-percent", type=float, default=35.0)
    ap.add_argument("--structure-mask-dilate", type=int, default=1)
    ap.add_argument("--structure-log-gain", type=float, default=20.0)
    ap.add_argument("--structure-pre-blur", type=int, default=0)
    ap.add_argument("--structure-ecc-rounds", type=int, default=2)
    ap.add_argument("--structure-residual-keep-percent", type=float, default=80.0)
    ap.add_argument("--ecc-valid-erode", type=int, default=2)
    ap.add_argument("--ecc-iters", type=int, default=80)
    ap.add_argument("--ecc-eps", type=float, default=1e-5)
    ap.add_argument("--ecc-gauss", type=int, default=5)
    ap.add_argument("--ecc-min-mask-count", type=int, default=100)
    ap.add_argument("--ecc-alpha", type=float, default=1.0, help="Pseudo-GT damping: q_gt = base_map + alpha * H_bias")
    ap.add_argument("--ecc-bias-max-abs", type=float, default=0.0, help="Clamp H bias per sampled pixel. <=0 disables")
    ap.add_argument("--max-obs-per-pair", type=int, default=25000)
    ap.add_argument("--min-obs-per-pair", type=int, default=500)
    ap.add_argument("--no-pair-debug-yuv", action="store_true")

    # Fitting.
    ap.add_argument("--device", default="auto")
    ap.add_argument("--torch-float64", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-batch-size", type=int, default=262144)
    ap.add_argument("--lr-rot", type=float, default=5e-4)
    ap.add_argument("--lr-trans", type=float, default=5e-5)
    ap.add_argument("--lr-focal", type=float, default=2e-4)
    ap.add_argument("--focal-mode", choices=["single", "separate", "fixed"], default="single", help="single => fx=fy=f, separate => fx/fy free, fixed => K fixed")
    ap.add_argument("--f-init", choices=["avg", "geom", "fx", "fy"], default="avg")
    ap.add_argument("--f-log-max-delta", type=float, default=0.05)
    ap.add_argument("--f-prior-weight", type=float, default=10.0)
    ap.add_argument("--rot-delta-prior-weight", type=float, default=1e-3)
    ap.add_argument("--trans-delta-prior-weight", type=float, default=100.0)
    ap.add_argument("--pose-delta-smooth-weight", type=float, default=1e-3)
    ap.add_argument("--max-trans-delta", type=float, default=0.0)
    ap.add_argument("--anchor-poc", type=int, default=0)
    ap.add_argument("--freeze-t", action="store_true")
    ap.add_argument("--freeze-r", action="store_true")
    ap.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
    ap.add_argument("--robust-f-scale", type=float, default=2.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--z-min", type=float, default=1e-4)
    ap.add_argument("--z-penalty", type=float, default=100.0)
    ap.add_argument("--render-row-batch", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--no-render-yuv", action="store_true")
    ap.add_argument("--save-depth-in-npz", action="store_true")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    if args.ecc_gauss <= 0 or args.ecc_gauss % 2 == 0:
        raise ValueError("--ecc-gauss must be a positive odd integer")
    if args.structure_keep_percent <= 0 or args.structure_keep_percent > 100:
        raise ValueError("--structure-keep-percent must be in (0,100]")
    if args.structure_ecc_rounds < 1:
        raise ValueError("--structure-ecc-rounds must be >=1")
    if args.structure_residual_keep_percent <= 0 or args.structure_residual_keep_percent > 100:
        raise ValueError("--structure-residual-keep-percent must be in (0,100]")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")

    ensure_dir(args.output_dir)
    out_json = Path(args.output_dir) / "gop_camera_refine_rf_result.json"
    out_jsonl = Path(args.output_dir) / "gop_camera_refine_rf_cam.jsonl"
    out_npz = Path(args.output_dir) / "gop_camera_refine_rf_geometry.npz"
    for p in [out_json, out_jsonl, out_npz]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --overwrite.")

    stage = load_fixedk_stage1_npz(args.stage1_npz)
    K_base = stage["K"]
    r_base = stage["rvecs"]
    t_base = stage["tvecs"]
    depth = stage["depth"]
    frame_indices = stage["frame_indices"]
    n, h, w = depth.shape
    if int(args.width) != w or int(args.height) != h:
        raise ValueError(f"Size mismatch: args={args.width}x{args.height}, NPZ depth={w}x{h}")

    copied_header = load_first_jsonl_object(args.stage1_camera_jsonl) if args.stage1_camera_jsonl else None
    pairs = build_pair_list(args, stage)

    print("[INFO] Loaded fixedK_gop_nn geometry NPZ")
    print(f"  npz         : {args.stage1_npz}")
    print(f"  source keys : {stage['source_keys']}")
    print(f"  frames      : {n}")
    print(f"  size        : {w}x{h}")
    print("[INFO] K_base:")
    print(K_base)
    print(f"[INFO] frame_index_mode={args.frame_index_mode}, seq_start={args.seq_start}")
    print(f"[INFO] pairs={len(pairs)}")
    for p in pairs[:80]:
        print(f"  target={p[0]} ref={p[1]} weight={p[2]:.4g} kind={p[3]}")
    if len(pairs) > 80:
        print(f"  ... {len(pairs)-80} more")

    rng = np.random.default_rng(int(args.seed))
    pair_info: list[dict[str, Any]] = []
    obs_list: list[dict[str, np.ndarray]] = []
    pair_extract_dir = os.path.join(args.output_dir, "pair_ecc")
    ensure_dir(pair_extract_dir)

    for target, ref, weight, kind in pairs:
        print(f"[PAIR ECC] target={target}, ref={ref}, weight={weight:.4g}, kind={kind}")
        obs, info = collect_pair_observations(
            pair=(target, ref, weight, kind),
            seq_yuv=args.input,
            width=w,
            height=h,
            bitdepth=args.bitdepth,
            frame_indices=frame_indices,
            K_base=K_base,
            rvecs_base=r_base,
            tvecs_base=t_base,
            depth=depth,
            args=args,
            rng=rng,
            pair_out_dir=pair_extract_dir,
        )
        pair_info.append(info)
        if obs.get("target", np.empty(0)).size > 0:
            obs_list.append(obs)
        print(json.dumps({
            "target": info["target"],
            "ref": info["ref"],
            "success": info["success"],
            "ecc_cc": info["ecc_cc"],
            "num_observations": info["num_observations"],
            "cp_bias_raw": info["cp_bias_raw"],
        }, indent=2))

    if not obs_list:
        raise RuntimeError("No valid pair ECC observations were generated.")
    observations = concat_observations(obs_list)
    print(f"[INFO] total observations = {observations['px'].shape[0]}")

    K_final, r_final, t_final, fit_report = fit_rf_tiny_t_w2c(
        observations=observations,
        rvecs_base=r_base,
        tvecs_base=t_base,
        K_base=K_base,
        args=args,
    )

    render_costs: list[dict[str, Any]] = []
    if not args.skip_render:
        render_costs = render_refined_pairs(
            pairs=pairs,
            seq_yuv=args.input,
            width=w,
            height=h,
            bitdepth=args.bitdepth,
            frame_indices=frame_indices,
            K_base=K_base,
            r_base=r_base,
            t_base=t_base,
            K_final=K_final,
            r_final=r_final,
            t_final=t_final,
            depth=depth,
            out_dir=os.path.join(args.output_dir, "refined_pairs"),
            args=args,
        )

    pose_json = []
    for i in range(n):
        pose_json.append({
            "poc": int(i),
            "frame_idx": int(frame_indices[i]),
            "is_anchor": bool(i == int(args.anchor_poc)),
            "rvec_base": r_base[i].astype(float).tolist(),
            "t_base": t_base[i].astype(float).tolist(),
            "rvec_refined": r_final[i].astype(float).tolist(),
            "t_refined": t_final[i].astype(float).tolist(),
            "rvec_delta": (r_final[i] - r_base[i]).astype(float).tolist(),
            "t_delta": (t_final[i] - t_base[i]).astype(float).tolist(),
            "R_refined": rodrigues_np(r_final[i]).astype(float).tolist(),
        })

    result = {
        "input": args.input,
        "width": int(w),
        "height": int(h),
        "bitdepth": int(args.bitdepth),
        "stage1_npz": str(args.stage1_npz),
        "stage1_camera_jsonl": str(args.stage1_camera_jsonl) if args.stage1_camera_jsonl else None,
        "stage1_source_keys": stage["source_keys"],
        "frame_indices": frame_indices.astype(int).tolist(),
        "method": {
            "description": "Second-stage fixed-depth fitting from pair-wise structure-ECC homography/affine residual transforms. Fits GOP focal and frame-wise W2C R|t; t is strongly regularized.",
            "depth": "fixed depth_canonical from first-stage NPZ",
            "supervision": "q_gt = base_camera_map + ECC_transform_bias(x,y)",
            "pose_convention": "camera_from_world / W2C: X_cam=R X_world+t",
            "relative_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target",
            "coordinate": "target pixel -> ref pixel",
        },
        "options": vars(args),
        "K_base": K_base.astype(float).tolist(),
        "K_refined": K_final.astype(float).tolist(),
        "focal_delta": {
            "fx_base": float(K_base[0, 0]),
            "fy_base": float(K_base[1, 1]),
            "fx_refined": float(K_final[0, 0]),
            "fy_refined": float(K_final[1, 1]),
            "fx_ratio": float(K_final[0, 0] / K_base[0, 0]),
            "fy_ratio": float(K_final[1, 1] / K_base[1, 1]),
            "fxfy_base_ratio": float(K_base[0, 0] / K_base[1, 1]),
            "fxfy_refined_ratio": float(K_final[0, 0] / K_final[1, 1]),
        },
        "pairs": pair_info,
        "fit_report": fit_report,
        "poses": pose_json,
        "render_costs": render_costs,
        "outputs": {
            "pair_ecc_dir": pair_extract_dir,
            "refined_pair_dir": os.path.join(args.output_dir, "refined_pairs"),
            "result_json": str(out_json),
            "camera_jsonl": str(out_jsonl),
            "geometry_npz": str(out_npz),
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    write_refined_camera_jsonl(
        path=out_jsonl,
        source_npz=str(args.stage1_npz),
        source_camera_jsonl=str(args.stage1_camera_jsonl) if args.stage1_camera_jsonl else None,
        frame_indices=frame_indices,
        K_final=K_final,
        r_final=r_final,
        t_final=t_final,
        z_sign=args.z_sign,
        copied_header=copied_header,
        args=args,
    )
    write_minimal_npz(out_npz, stage, K_final, r_final, t_final, args)

    print("[DONE]")
    print(f"  result JSON  : {out_json}")
    print(f"  camera JSONL : {out_jsonl}")
    print(f"  geometry NPZ : {out_npz}")
    print("  K_base:")
    print(K_base)
    print("  K_refined:")
    print(K_final)


if __name__ == "__main__":
    main()
