#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp a reference YUV frame to a target frame using VGGT-Omega depth + camera output,
then refine the target depth by using screen-space residual homographies estimated
between the initial camera/depth warp result and the target frame.

Pipeline:
  1) Load VGGT-Omega target depth and fixed camera parameters.
  2) Compute initial backward projection map:
       target pixel (x,y) + target depth -> reference pixel (u,v)
  3) Render initial warped reference image.
  4) For each H-block, estimate a residual homography in target-screen coordinates:
       target image coords -> initial-warped image coords
     using ORB matching inside the valid projection region. Optionally use a small
     translation sweep fallback when local ORB is weak.
  5) Compose the residual screen homography with the initial projection map:
       desired_ref_xy(x,y) = initial_map( H_residual_block(x,y) )
  6) Fit block-wise inverse-depth residual planes with fixed cameras:
       inv_z'(x,y) = inv_z(x,y) * (1 + max_rel * tanh(a*xn + b*yn + c))
     so that fixed_camera_project(x,y,z') ~= desired_ref_xy(x,y).
  7) Warp reference frame again using the fitted depth.

This script is intended for experiments, not bitstream integration.

Example:
  python warp_vggt_omega_yuv_residual_homography_depthfit.py \
    --yuv input.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p10le \
    --npz out/test_vggt_omega_outputs.npz \
    --camera-jsonl out/test_camera.jsonl \
    --ref-idx 0 --tar-idx 7 \
    --output-prefix out/rhfit_t007_r000 \
    --h-block-size 64 \
    --fit-mode plane --fit-block-size 64 --fit-sample-stride 4 \
    --fit-iters 300 --fit-lr 0.03 --max-rel-inv-correction 0.30 \
    --write-before-fit --write-residual-h-warp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required: pip install torch") from exc

PixFmt = Literal["yuv420p", "yuv420p10le"]
DepthMode = Literal["linear", "inverse"]
FitMode = Literal["none", "bias", "plane"]
InvalidFill = Literal["black", "copy_target", "neutral"]


# ============================================================
# Data
# ============================================================

@dataclass
class ResidualHBlock:
    block_x: int
    block_y: int
    block_w: int
    block_h: int
    H: np.ndarray
    source: str
    accepted: bool
    match_count: int
    inlier_count: int
    base_cost: float
    candidate_cost: float
    chosen_cost: float
    valid_ratio: float
    max_corner_disp: float
    reason: str


# ============================================================
# YUV I/O
# ============================================================

def normalize_pix_fmt(s: str) -> PixFmt:
    s = s.lower().replace("-", "").replace("_", "")
    aliases = {
        "420p": "yuv420p",
        "yuv420p": "yuv420p",
        "i420": "yuv420p",
        "420p8": "yuv420p",
        "yuv420p8": "yuv420p",
        "420p10le": "yuv420p10le",
        "yuv420p10le": "yuv420p10le",
        "i010": "yuv420p10le",
    }
    if s not in aliases:
        raise ValueError(f"Unsupported pix-fmt: {s}. Use yuv420p or yuv420p10le.")
    return aliases[s]  # type: ignore[return-value]


def frame_size_bytes(width: int, height: int, pix_fmt: PixFmt) -> int:
    if width % 2 or height % 2:
        raise ValueError("YUV420 requires even width and height.")
    samples = width * height + 2 * ((width // 2) * (height // 2))
    return samples if pix_fmt == "yuv420p" else samples * 2


def read_yuv420_frame(
    path: str,
    frame_idx: int,
    width: int,
    height: int,
    pix_fmt: PixFmt,
    tenbit_shift_right: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = frame_size_bytes(width, height, pix_fmt)
    with open(path, "rb") as f:
        f.seek(frame_idx * fs)
        raw = f.read(fs)
    if len(raw) != fs:
        raise EOFError(f"Cannot read frame {frame_idx} from {path}: expected {fs} bytes, got {len(raw)}")

    y_n = width * height
    uv_n = (width // 2) * (height // 2)

    if pix_fmt == "yuv420p":
        arr = np.frombuffer(raw, dtype=np.uint8)
        y = arr[:y_n].reshape(height, width).copy()
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).copy()
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).copy()
    else:
        arr = np.frombuffer(raw, dtype="<u2")
        if tenbit_shift_right > 0:
            arr = arr >> tenbit_shift_right
        y = arr[:y_n].reshape(height, width).copy()
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).copy()
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).copy()
    return y, u, v


def write_yuv420_frame(path: str, y: np.ndarray, u: np.ndarray, v: np.ndarray, pix_fmt: PixFmt, bit_depth: int) -> None:
    maxv = (1 << bit_depth) - 1
    y = np.clip(np.rint(y), 0, maxv)
    u = np.clip(np.rint(u), 0, maxv)
    v = np.clip(np.rint(v), 0, maxv)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        if pix_fmt == "yuv420p":
            f.write(y.astype(np.uint8).tobytes())
            f.write(u.astype(np.uint8).tobytes())
            f.write(v.astype(np.uint8).tobytes())
        else:
            f.write(y.astype("<u2").tobytes())
            f.write(u.astype("<u2").tobytes())
            f.write(v.astype("<u2").tobytes())


def write_mask_yuv420p(path: str, mask: np.ndarray) -> None:
    h, w = mask.shape
    y = mask.astype(np.uint8) * 255
    u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes())


def write_inverse_depth_yuv420p10le(path: str, depth: np.ndarray) -> dict:
    h, w = depth.shape
    inv = np.zeros_like(depth, dtype=np.float32)
    good = np.isfinite(depth) & (depth > 1e-12)
    inv[good] = 1.0 / depth[good]
    valid_inv = inv[np.isfinite(inv) & (inv > 0)]
    if valid_inv.size == 0:
        qmin, qmax = 0.0, 1.0
    else:
        qmin = float(np.percentile(valid_inv, 0.1))
        qmax = float(np.percentile(valid_inv, 99.9))
        if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
            qmin = float(np.min(valid_inv))
            qmax = float(np.max(valid_inv))
        if qmax <= qmin:
            qmax = qmin + 1e-6
    y = np.clip((inv - qmin) / (qmax - qmin), 0.0, 1.0) * 1023.0
    u = np.full((h // 2, w // 2), 512, dtype=np.float32)
    v = np.full((h // 2, w // 2), 512, dtype=np.float32)
    write_yuv420_frame(path, y, u, v, "yuv420p10le", 10)
    return {"depth_quant_mode": "inverse", "quant_min": qmin, "quant_max": qmax, "pix_fmt": "yuv420p10le"}


def to_8bit(y: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        return np.clip(y, 0, 255).astype(np.uint8)
    return np.clip(y.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)


# ============================================================
# VGGT output loading
# ============================================================

def load_camera_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "frame_idx" not in rec:
                raise ValueError(f"camera jsonl line {line_no} has no frame_idx")
            records.append(rec)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_depth_from_npz(npz_path: str, tar_idx: int) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    if "frame_indices" not in z or "depth_original" not in z:
        raise ValueError("NPZ must contain frame_indices and depth_original")
    frame_indices = z["frame_indices"].astype(np.int64)
    matches = np.where(frame_indices == tar_idx)[0]
    if len(matches) != 1:
        raise ValueError(f"target frame {tar_idx} not found uniquely in {npz_path}; matches={matches.tolist()}")
    return z["depth_original"][int(matches[0])].astype(np.float32)


def load_quantized_depth_yuv(
    depth_yuv: str,
    depth_frame_pos: int,
    width: int,
    height: int,
    depth_pix_fmt: PixFmt,
    depth_meta: dict,
) -> np.ndarray:
    y, _, _ = read_yuv420_frame(depth_yuv, depth_frame_pos, width, height, depth_pix_fmt)
    bit_depth = 8 if depth_pix_fmt == "yuv420p" else 10
    max_code = float((1 << bit_depth) - 1)
    qmin = float(depth_meta["quant_min"])
    qmax = float(depth_meta["quant_max"])
    mode: DepthMode = depth_meta.get("depth_quant_mode", "inverse")

    q = y.astype(np.float32) / max_code
    qsrc = qmin + q * (qmax - qmin)
    if mode == "linear":
        depth = qsrc
    elif mode == "inverse":
        eps = max(1e-12, abs(qmax - qmin) * 1e-9)
        depth = 1.0 / np.maximum(qsrc, eps)
    else:
        raise ValueError(f"Unsupported depth_quant_mode: {mode}")
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def as_k3(k: list | np.ndarray) -> np.ndarray:
    K = np.asarray(k, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected K shape 3x3, got {K.shape}")
    return K


def as_rt34(e: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(e, dtype=np.float64)
    if E.shape == (3, 4):
        R = E[:, :3]
        t = E[:, 3]
    elif E.shape == (4, 4):
        R = E[:3, :3]
        t = E[:3, 3]
    else:
        raise ValueError(f"Expected extrinsic shape 3x4 or 4x4, got {E.shape}")
    return R, t


# ============================================================
# Projection / warping
# ============================================================

def make_backward_map(
    depth_tar: np.ndarray,
    K_ref: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    K_tar: np.ndarray,
    R_tar: np.ndarray,
    t_tar: np.ndarray,
    min_depth: float = 1e-8,
    chunk_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    h, w = depth_tar.shape
    inv_K_tar = np.linalg.inv(K_tar)
    Rt_tar = R_tar.T
    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    total_z_valid = 0
    total_in_front_ref = 0

    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        ys, xs = np.mgrid[y0:y1, 0:w]
        z = depth_tar[y0:y1].astype(np.float64)
        depth_ok = np.isfinite(z) & (z > min_depth)
        total_z_valid += int(depth_ok.sum())

        pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(z)], axis=0).reshape(3, -1)
        rays_tar = inv_K_tar @ pix
        x_tar = rays_tar * z.reshape(-1)[None, :]
        x_world = Rt_tar @ (x_tar - t_tar.reshape(3, 1))
        x_ref = R_ref @ x_world + t_ref.reshape(3, 1)
        zr = x_ref[2]
        in_front = zr > min_depth
        total_in_front_ref += int((in_front & depth_ok.reshape(-1)).sum())

        proj = K_ref @ x_ref
        denom = np.where(np.abs(proj[2]) > min_depth, proj[2], min_depth)
        xr = proj[0] / denom
        yr = proj[1] / denom
        inside = (xr >= 0.0) & (xr <= w - 1.0) & (yr >= 0.0) & (yr <= h - 1.0)
        ok = depth_ok.reshape(-1) & in_front & inside & np.isfinite(xr) & np.isfinite(yr)

        map_x[y0:y1].reshape(-1)[ok] = xr[ok].astype(np.float32)
        map_y[y0:y1].reshape(-1)[ok] = yr[ok].astype(np.float32)
        valid[y0:y1].reshape(-1)[ok] = True

    stats = {
        "pixels": int(h * w),
        "target_depth_valid": int(total_z_valid),
        "target_depth_valid_ratio": float(total_z_valid / max(h * w, 1)),
        "in_front_of_ref_camera": int(total_in_front_ref),
        "projection_inside_ref": int(valid.sum()),
        "projection_inside_ref_ratio": float(valid.mean()),
    }
    return map_x, map_y, valid, stats


def remap_plane(plane: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray, interp: int, border: int, fill: float) -> np.ndarray:
    out = cv2.remap(
        plane.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interp,
        borderMode=border,
        borderValue=float(fill),
    )
    out[~valid] = fill
    return out


def chroma_maps_from_luma(map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map_x.shape
    cw, ch = w // 2, h // 2
    cmx = cv2.resize(map_x, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cmy = cv2.resize(map_y, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cvalid = cv2.resize(valid.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA) > 0.999
    cmx[~cvalid] = -1.0
    cmy[~cvalid] = -1.0
    return cmx.astype(np.float32), cmy.astype(np.float32), cvalid


def y_mae_psnr(pred_y: np.ndarray, target_y: np.ndarray, valid: np.ndarray, bit_depth: int) -> tuple[Optional[float], Optional[float]]:
    if int(valid.sum()) == 0:
        return None, None
    diff = pred_y.astype(np.float64)[valid] - target_y.astype(np.float64)[valid]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        psnr = float("inf")
    else:
        maxv = float((1 << bit_depth) - 1)
        psnr = float(10.0 * np.log10((maxv * maxv) / mse))
    return mae, psnr


def warp_yuv_from_map(
    ref_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    tar_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    bit_depth: int,
    interp_name: str,
    border_name: str,
    invalid_fill: InvalidFill,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Optional[float], Optional[float]]:
    ref_y, ref_u, ref_v = ref_yuv
    tar_y, tar_u, tar_v = tar_yuv
    neutral = 128 if bit_depth == 8 else 512
    interp = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[interp_name]
    border = cv2.BORDER_REPLICATE if border_name == "replicate" else cv2.BORDER_CONSTANT
    y_fill = 0.0 if invalid_fill in ["black", "copy_target"] else float(neutral)
    uv_fill = float(neutral)

    raw_y = remap_plane(ref_y, map_x, map_y, valid, interp, border, y_fill)
    cmx, cmy, cvalid = chroma_maps_from_luma(map_x, map_y, valid)
    raw_u = remap_plane(ref_u, cmx, cmy, cvalid, interp, border, uv_fill)
    raw_v = remap_plane(ref_v, cmx, cmy, cvalid, interp, border, uv_fill)

    wy, wu, wv = raw_y, raw_u, raw_v
    if invalid_fill == "copy_target":
        wy = raw_y.copy(); wu = raw_u.copy(); wv = raw_v.copy()
        wy[~valid] = tar_y.astype(np.float32)[~valid]
        wu[~cvalid] = tar_u.astype(np.float32)[~cvalid]
        wv[~cvalid] = tar_v.astype(np.float32)[~cvalid]

    mae, psnr = y_mae_psnr(raw_y, tar_y, valid, bit_depth)
    return (wy, wu, wv), (raw_y, raw_u, raw_v), cvalid, mae, psnr


def warp_yuv_with_depth(
    depth_tar: np.ndarray,
    ref_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    tar_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    cameras: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bit_depth: int,
    interp_name: str,
    border_name: str,
    invalid_fill: InvalidFill,
    min_depth: float,
    chunk_rows: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict, Optional[float], Optional[float], np.ndarray, np.ndarray]:
    K_ref, R_ref, t_ref, K_tar, R_tar, t_tar = cameras
    map_x, map_y, valid, stats = make_backward_map(depth_tar, K_ref, R_ref, t_ref, K_tar, R_tar, t_tar, min_depth, chunk_rows)
    warped, _raw, _cvalid, mae, psnr = warp_yuv_from_map(ref_yuv, tar_yuv, map_x, map_y, valid, bit_depth, interp_name, border_name, invalid_fill)
    return warped, valid, stats, mae, psnr, map_x, map_y


# ============================================================
# Residual homography estimation
# ============================================================

def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H = normalize_homography(H)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    q = ph @ H.T
    z = q[:, 2:3] + 1e-12
    return (q[:, :2] / z).astype(np.float32)


def homography_maps_for_roi(H: np.ndarray, x0: int, y0: int, bw: int, bh: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = normalize_homography(H)
    xs, ys = np.meshgrid(
        np.arange(x0, x0 + bw, dtype=np.float32),
        np.arange(y0, y0 + bh, dtype=np.float32),
    )
    denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    good_denom = np.abs(denom) > 1e-9
    denom = denom + 1e-12
    hx = (H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / denom
    hy = (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / denom
    valid = good_denom & np.isfinite(hx) & np.isfinite(hy) & (hx >= 0) & (hx <= width - 1) & (hy >= 0) & (hy <= height - 1)
    return hx.astype(np.float32), hy.astype(np.float32), valid


def sample_map_at_screen(map_img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    return cv2.remap(map_img.astype(np.float32), sx.astype(np.float32), sy.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)


def sample_valid_at_screen(valid_img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    v = cv2.remap(valid_img.astype(np.uint8), sx.astype(np.float32), sy.astype(np.float32), interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return v > 0


def compose_initial_map_with_screen_H(
    H: np.ndarray,
    x0: int,
    y0: int,
    bw: int,
    bh: int,
    init_map_x: np.ndarray,
    init_map_y: np.ndarray,
    init_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = init_map_x.shape
    sx, sy, screen_valid = homography_maps_for_roi(H, x0, y0, bw, bh, w, h)
    v0 = sample_valid_at_screen(init_valid, sx, sy)
    qx = sample_map_at_screen(init_map_x, sx, sy)
    qy = sample_map_at_screen(init_map_y, sx, sy)
    valid = screen_valid & v0 & np.isfinite(qx) & np.isfinite(qy) & (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1)
    qx[~valid] = -1.0
    qy[~valid] = -1.0
    return qx.astype(np.float32), qy.astype(np.float32), valid


def block_cost_from_desired_map(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    valid: np.ndarray,
    x0: int,
    y0: int,
    min_valid_ratio: float,
) -> tuple[float, float]:
    valid_ratio = float(np.mean(valid)) if valid.size else 0.0
    if valid_ratio < min_valid_ratio or not np.any(valid):
        return float("inf"), valid_ratio
    pred = cv2.remap(ref_y.astype(np.float32), qx, qy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tgt = target_y[y0:y0 + qx.shape[0], x0:x0 + qx.shape[1]].astype(np.float32)
    cost = float(np.mean(np.abs(pred[valid] - tgt[valid])))
    return cost, valid_ratio


def max_corner_displacement(H: np.ndarray, x0: int, y0: int, bw: int, bh: int) -> float:
    pts = np.array([[x0, y0], [x0 + bw - 1, y0], [x0 + bw - 1, y0 + bh - 1], [x0, y0 + bh - 1]], dtype=np.float32)
    q = apply_homography_points(H, pts)
    return float(np.max(np.sqrt(np.sum((q - pts) ** 2, axis=1))))


def fit_residual_H_orb_for_block(
    target_y: np.ndarray,
    init_warp_y: np.ndarray,
    valid: np.ndarray,
    x0: int,
    y0: int,
    bw: int,
    bh: int,
    bit_depth: int,
    max_features: int,
    ratio: float,
    ransac_thresh: float,
    min_matches: int,
    min_inliers: int,
    use_clahe: bool,
) -> tuple[Optional[np.ndarray], int, int, str]:
    tgt_patch = to_8bit(target_y[y0:y0 + bh, x0:x0 + bw], bit_depth)
    wrp_patch = to_8bit(init_warp_y[y0:y0 + bh, x0:x0 + bw], bit_depth)
    mask = (valid[y0:y0 + bh, x0:x0 + bw].astype(np.uint8) * 255)
    if int(np.count_nonzero(mask)) < max(16, min_matches):
        return None, 0, 0, "not_enough_valid_pixels"

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        tgt_patch = clahe.apply(tgt_patch)
        wrp_patch = clahe.apply(wrp_patch)

    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=4, edgeThreshold=5, patchSize=15, fastThreshold=7)
    kp_t, des_t = orb.detectAndCompute(tgt_patch, mask)
    kp_w, des_w = orb.detectAndCompute(wrp_patch, mask)
    if des_t is None or des_w is None or len(kp_t) < min_matches or len(kp_w) < min_matches:
        return None, 0, 0, "not_enough_keypoints"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_t, des_w, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None, len(good), 0, "not_enough_matches"

    pts_t = np.float32([[kp_t[m.queryIdx].pt[0] + x0, kp_t[m.queryIdx].pt[1] + y0] for m in good])
    pts_w = np.float32([[kp_w[m.trainIdx].pt[0] + x0, kp_w[m.trainIdx].pt[1] + y0] for m in good])
    try:
        H, inlier_mask = cv2.findHomography(pts_t, pts_w, cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=1000, confidence=0.99)
    except cv2.error as exc:
        return None, len(good), 0, f"cv2_error:{exc}"
    if H is None or inlier_mask is None:
        return None, len(good), 0, "findHomography_failed"
    inliers = int(np.count_nonzero(inlier_mask.reshape(-1)))
    if inliers < min_inliers:
        return None, len(good), inliers, "not_enough_inliers"
    return normalize_homography(H), len(good), inliers, "ok"


def translation_H(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], dtype=np.float64)


def find_best_translation_H(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    init_map_x: np.ndarray,
    init_map_y: np.ndarray,
    init_valid: np.ndarray,
    x0: int,
    y0: int,
    bw: int,
    bh: int,
    min_valid_ratio: float,
    max_shift: int,
    step: int,
) -> tuple[np.ndarray, float, float]:
    best_H = np.eye(3, dtype=np.float64)
    qx0, qy0, v0 = compose_initial_map_with_screen_H(best_H, x0, y0, bw, bh, init_map_x, init_map_y, init_valid)
    best_cost, best_vr = block_cost_from_desired_map(target_y, ref_y, qx0, qy0, v0, x0, y0, min_valid_ratio)
    if max_shift <= 0:
        return best_H, best_cost, best_vr
    step = max(1, int(step))
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            if dx == 0 and dy == 0:
                continue
            H = translation_H(dx, dy)
            qx, qy, vv = compose_initial_map_with_screen_H(H, x0, y0, bw, bh, init_map_x, init_map_y, init_valid)
            cost, vr = block_cost_from_desired_map(target_y, ref_y, qx, qy, vv, x0, y0, min_valid_ratio)
            if cost < best_cost:
                best_H, best_cost, best_vr = H, cost, vr
    return best_H, best_cost, best_vr


def estimate_residual_homographies(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    init_warp_y: np.ndarray,
    init_map_x: np.ndarray,
    init_map_y: np.ndarray,
    init_valid: np.ndarray,
    bit_depth: int,
    block_size: int,
    min_valid_ratio: float,
    min_gain: float,
    max_corner_disp: float,
    max_features: int,
    match_ratio: float,
    ransac_thresh: float,
    min_matches: int,
    min_inliers: int,
    use_clahe: bool,
    enable_translation_sweep: bool,
    translation_max_shift: int,
    translation_step: int,
) -> tuple[list[ResidualHBlock], np.ndarray, np.ndarray, np.ndarray, dict]:
    h, w = target_y.shape
    records: list[ResidualHBlock] = []
    desired_map_x = np.full((h, w), -1.0, dtype=np.float32)
    desired_map_y = np.full((h, w), -1.0, dtype=np.float32)
    desired_valid = np.zeros((h, w), dtype=bool)
    accepted = 0
    orb_accepted = 0
    trans_accepted = 0
    inherited = 0

    for y0 in range(0, h, block_size):
        bh = min(block_size, h - y0)
        for x0 in range(0, w, block_size):
            bw = min(block_size, w - x0)
            I = np.eye(3, dtype=np.float64)
            qx_base, qy_base, v_base = compose_initial_map_with_screen_H(I, x0, y0, bw, bh, init_map_x, init_map_y, init_valid)
            base_cost, base_vr = block_cost_from_desired_map(target_y, ref_y, qx_base, qy_base, v_base, x0, y0, min_valid_ratio)

            chosen_H = I
            chosen_cost = base_cost
            chosen_vr = base_vr
            candidate_cost = float("inf")
            match_count = 0
            inlier_count = 0
            source = "identity"
            reason = "identity_fallback"
            accepted_this = False
            max_disp = 0.0

            H_orb, match_count, inlier_count, orb_reason = fit_residual_H_orb_for_block(
                target_y=target_y,
                init_warp_y=init_warp_y,
                valid=init_valid,
                x0=x0,
                y0=y0,
                bw=bw,
                bh=bh,
                bit_depth=bit_depth,
                max_features=max_features,
                ratio=match_ratio,
                ransac_thresh=ransac_thresh,
                min_matches=min_matches,
                min_inliers=min_inliers,
                use_clahe=use_clahe,
            )
            if H_orb is not None:
                disp = max_corner_displacement(H_orb, x0, y0, bw, bh)
                qx_c, qy_c, v_c = compose_initial_map_with_screen_H(H_orb, x0, y0, bw, bh, init_map_x, init_map_y, init_valid)
                cost_c, vr_c = block_cost_from_desired_map(target_y, ref_y, qx_c, qy_c, v_c, x0, y0, min_valid_ratio)
                candidate_cost = cost_c
                if np.isfinite(cost_c) and cost_c <= base_cost - min_gain and disp <= max_corner_disp:
                    chosen_H = H_orb
                    chosen_cost = cost_c
                    chosen_vr = vr_c
                    max_disp = disp
                    source = "orb_homography"
                    reason = "orb_accepted"
                    accepted_this = True
                    orb_accepted += 1
                else:
                    reason = f"orb_rejected:{orb_reason},cost_or_disp"
            else:
                reason = f"orb_failed:{orb_reason}"

            if (not accepted_this) and enable_translation_sweep:
                H_t, cost_t, vr_t = find_best_translation_H(
                    target_y, ref_y, init_map_x, init_map_y, init_valid,
                    x0, y0, bw, bh, min_valid_ratio, translation_max_shift, translation_step,
                )
                disp_t = max_corner_displacement(H_t, x0, y0, bw, bh)
                if np.isfinite(cost_t) and cost_t <= base_cost - min_gain and disp_t <= max_corner_disp:
                    chosen_H = H_t
                    chosen_cost = cost_t
                    chosen_vr = vr_t
                    max_disp = disp_t
                    source = "translation_sweep"
                    reason = "translation_accepted"
                    accepted_this = True
                    trans_accepted += 1

            qx_ch, qy_ch, v_ch = compose_initial_map_with_screen_H(chosen_H, x0, y0, bw, bh, init_map_x, init_map_y, init_valid)
            desired_map_x[y0:y0 + bh, x0:x0 + bw] = qx_ch
            desired_map_y[y0:y0 + bh, x0:x0 + bw] = qy_ch
            desired_valid[y0:y0 + bh, x0:x0 + bw] = v_ch

            if accepted_this:
                accepted += 1
            else:
                inherited += 1
                max_disp = 0.0

            records.append(ResidualHBlock(
                block_x=x0, block_y=y0, block_w=bw, block_h=bh,
                H=normalize_homography(chosen_H),
                source=source,
                accepted=accepted_this,
                match_count=int(match_count),
                inlier_count=int(inlier_count),
                base_cost=float(base_cost) if np.isfinite(base_cost) else None,
                candidate_cost=float(candidate_cost) if np.isfinite(candidate_cost) else None,
                chosen_cost=float(chosen_cost) if np.isfinite(chosen_cost) else None,
                valid_ratio=float(chosen_vr),
                max_corner_disp=float(max_disp),
                reason=reason,
            ))

    summary = {
        "block_size": int(block_size),
        "num_blocks": int(len(records)),
        "accepted": int(accepted),
        "orb_accepted": int(orb_accepted),
        "translation_accepted": int(trans_accepted),
        "inherited_identity": int(inherited),
        "desired_valid_ratio": float(np.mean(desired_valid)),
    }
    return records, desired_map_x, desired_map_y, desired_valid, summary


# ============================================================
# Torch fitting to desired ref XY map
# ============================================================

def torch_camera_project_xy(
    depth: torch.Tensor,
    rays_tar: torch.Tensor,
    K_ref: torch.Tensor,
    R_ref: torch.Tensor,
    t_ref: torch.Tensor,
    R_tar_t: torch.Tensor,
    t_tar: torch.Tensor,
    min_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_tar = rays_tar * depth.unsqueeze(0)
    x_world = R_tar_t @ (x_tar - t_tar.reshape(3, 1))
    x_ref = R_ref @ x_world + t_ref.reshape(3, 1)
    zr = x_ref[2]
    proj = K_ref @ x_ref
    denom = torch.clamp(proj[2], min=float(min_depth))
    xr = proj[0] / denom
    yr = proj[1] / denom
    return xr, yr, zr


def smoothness_loss(params: torch.Tensor, nby: int, nbx: int) -> torch.Tensor:
    p = params.reshape(nby, nbx, 3)
    loss = p.new_tensor(0.0)
    cnt = 0
    if nbx > 1:
        loss = loss + (p[:, 1:, :] - p[:, :-1, :]).pow(2).mean(); cnt += 1
    if nby > 1:
        loss = loss + (p[1:, :, :] - p[:-1, :, :]).pow(2).mean(); cnt += 1
    return loss if cnt == 0 else loss / cnt


def fit_inverse_depth_planes_to_xy_torch(
    depth0_np: np.ndarray,
    desired_map_x: np.ndarray,
    desired_map_y: np.ndarray,
    desired_valid: np.ndarray,
    init_map_x: np.ndarray,
    init_map_y: np.ndarray,
    K_ref_np: np.ndarray,
    R_ref_np: np.ndarray,
    t_ref_np: np.ndarray,
    K_tar_np: np.ndarray,
    R_tar_np: np.ndarray,
    t_tar_np: np.ndarray,
    fit_mode: FitMode,
    block_size: int,
    sample_stride: int,
    iters: int,
    lr: float,
    max_rel_inv_correction: float,
    reg_lambda: float,
    smooth_lambda: float,
    min_depth: float,
    device_name: str,
    loss_type: str,
    target_blend: float,
    print_every: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if fit_mode == "none":
        return depth0_np.astype(np.float32), np.zeros((0, 3), dtype=np.float32), {"fit_mode": "none"}
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device_name} was requested, but CUDA is not available")
    if not (0.0 <= target_blend <= 1.0):
        raise ValueError("--target-blend must be in [0,1]")

    h, w = depth0_np.shape
    nbx = (w + block_size - 1) // block_size
    nby = (h + block_size - 1) // block_size
    num_blocks = nbx * nby
    device = torch.device(device_name)
    dtype = torch.float32

    ys_np = np.arange(0, h, sample_stride, dtype=np.float32)
    xs_np = np.arange(0, w, sample_stride, dtype=np.float32)
    gy_np, gx_np = np.meshgrid(ys_np, xs_np, indexing="ij")
    xs_flat = gx_np.reshape(-1)
    ys_flat = gy_np.reshape(-1)
    xi = np.clip(np.rint(xs_flat).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(ys_flat).astype(np.int64), 0, h - 1)

    dvalid = desired_valid[yi, xi]
    depth_s = depth0_np[yi, xi].astype(np.float32)
    depth_ok = np.isfinite(depth_s) & (depth_s > min_depth)
    sample_ok = dvalid & depth_ok
    if int(np.count_nonzero(sample_ok)) < 16:
        raise RuntimeError("Too few valid samples for depth fitting. Check residual H valid region / camera projection.")

    qx_des = desired_map_x[yi, xi].astype(np.float32)
    qy_des = desired_map_y[yi, xi].astype(np.float32)
    qx_init = init_map_x[yi, xi].astype(np.float32)
    qy_init = init_map_y[yi, xi].astype(np.float32)
    qx_tgt = (1.0 - target_blend) * qx_init + target_blend * qx_des
    qy_tgt = (1.0 - target_blend) * qy_init + target_blend * qy_des

    valid_idx_np = np.flatnonzero(sample_ok).astype(np.int64)
    xs_flat = xs_flat[valid_idx_np]
    ys_flat = ys_flat[valid_idx_np]
    xi = xi[valid_idx_np]
    yi = yi[valid_idx_np]
    depth_s = depth_s[valid_idx_np]
    qx_tgt = qx_tgt[valid_idx_np]
    qy_tgt = qy_tgt[valid_idx_np]

    bx_np = xi // block_size
    by_np = yi // block_size
    bid_np = by_np * nbx + bx_np
    x0_np = bx_np * block_size
    y0_np = by_np * block_size
    x1_np = np.minimum(x0_np + block_size, w)
    y1_np = np.minimum(y0_np + block_size, h)
    cx_np = (x0_np + x1_np - 1) * 0.5
    cy_np = (y0_np + y1_np - 1) * 0.5
    xn_np = (xs_flat - cx_np.astype(np.float32)) / float(block_size)
    yn_np = (ys_flat - cy_np.astype(np.float32)) / float(block_size)
    inv0_np = 1.0 / np.maximum(depth_s, min_depth)

    xs = torch.from_numpy(xs_flat).to(device=device, dtype=dtype)
    ys = torch.from_numpy(ys_flat).to(device=device, dtype=dtype)
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=0)
    inv_K_tar = torch.linalg.inv(torch.as_tensor(K_tar_np, device=device, dtype=dtype))
    rays_tar = inv_K_tar @ pix

    K_ref = torch.as_tensor(K_ref_np, device=device, dtype=dtype)
    R_ref = torch.as_tensor(R_ref_np, device=device, dtype=dtype)
    t_ref = torch.as_tensor(t_ref_np, device=device, dtype=dtype)
    R_tar_t = torch.as_tensor(R_tar_np.T, device=device, dtype=dtype)
    t_tar = torch.as_tensor(t_tar_np, device=device, dtype=dtype)

    block_id = torch.from_numpy(bid_np).to(device=device, dtype=torch.long)
    xn = torch.from_numpy(xn_np.astype(np.float32)).to(device=device, dtype=dtype)
    yn = torch.from_numpy(yn_np.astype(np.float32)).to(device=device, dtype=dtype)
    inv0 = torch.from_numpy(inv0_np.astype(np.float32)).to(device=device, dtype=dtype)
    qx = torch.from_numpy(qx_tgt.astype(np.float32)).to(device=device, dtype=dtype)
    qy = torch.from_numpy(qy_tgt.astype(np.float32)).to(device=device, dtype=dtype)

    params = torch.nn.Parameter(torch.zeros((num_blocks, 3), device=device, dtype=dtype))
    opt = torch.optim.Adam([params], lr=lr)

    def compute_loss_metrics() -> tuple[torch.Tensor, dict]:
        p = params[block_id]
        if fit_mode == "bias":
            plane = p[:, 2]
        else:
            plane = p[:, 0] * xn + p[:, 1] * yn + p[:, 2]
        rel = float(max_rel_inv_correction) * torch.tanh(plane)
        inv_corr = inv0 * (1.0 + rel)
        depth = 1.0 / torch.clamp(inv_corr, min=float(1.0 / 1e12))
        xr, yr, zr = torch_camera_project_xy(depth, rays_tar, K_ref, R_ref, t_ref, R_tar_t, t_tar, min_depth)
        inside = (zr > float(min_depth)) & (xr >= 0.0) & (xr <= w - 1.0) & (yr >= 0.0) & (yr <= h - 1.0) & torch.isfinite(xr) & torch.isfinite(yr)
        dx = xr - qx
        dy = yr - qy
        err2 = dx * dx + dy * dy
        if inside.any():
            e2 = err2[inside]
            if loss_type == "l2":
                reproj = e2.mean()
            elif loss_type == "l1":
                reproj = torch.sqrt(e2.clamp_min(1e-12)).mean()
            else:
                reproj = torch.sqrt(e2 + 1e-4).mean()
            mae_px = torch.sqrt(e2.clamp_min(1e-12)).mean()
        else:
            reproj = err2.mean() * 0.0 + 1.0
            mae_px = err2.mean() * 0.0 + 1.0
        reg = params.pow(2).mean()
        smooth = smoothness_loss(params, nby, nbx)
        total = reproj + float(reg_lambda) * reg + float(smooth_lambda) * smooth
        return total, {
            "reproj_loss": float(reproj.detach().cpu()),
            "mae_px": float(mae_px.detach().cpu()),
            "inside_ratio": float(inside.detach().float().mean().cpu()),
            "reg_loss": float(reg.detach().cpu()),
            "smooth_loss": float(smooth.detach().cpu()),
        }

    with torch.no_grad():
        initial_loss, initial_metrics = compute_loss_metrics()
        initial_total = float(initial_loss.detach().cpu())

    final_metrics = initial_metrics
    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, metrics = compute_loss_metrics()
        loss.backward()
        opt.step()
        final_metrics = metrics
        if print_every > 0 and (it == 1 or it % print_every == 0 or it == iters):
            print(
                f"fit iter {it:04d}/{iters}: "
                f"loss={float(loss.detach().cpu()):.6f}, "
                f"mae_px={metrics['mae_px']:.4f}, "
                f"inside={metrics['inside_ratio']:.4f}"
            )

    with torch.no_grad():
        final_loss, final_metrics = compute_loss_metrics()
        final_total = float(final_loss.detach().cpu())
        params_np = params.detach().cpu().numpy().astype(np.float32)

    corrected = apply_inverse_depth_params_fullres(depth0_np, params_np, w, h, block_size, fit_mode, max_rel_inv_correction, min_depth, device_name)

    stats = {
        "fit_mode": fit_mode,
        "formula": "inv_z_prime = inv_z * (1 + max_rel * tanh(a*xn + b*yn + c))",
        "block_size": int(block_size),
        "num_blocks_x": int(nbx),
        "num_blocks_y": int(nby),
        "num_blocks": int(num_blocks),
        "sample_stride": int(sample_stride),
        "num_samples_total": int(desired_valid[::sample_stride, ::sample_stride].size),
        "num_samples_used": int(valid_idx_np.size),
        "target_blend": float(target_blend),
        "iters": int(iters),
        "lr": float(lr),
        "max_rel_inv_correction": float(max_rel_inv_correction),
        "reg_lambda": float(reg_lambda),
        "smooth_lambda": float(smooth_lambda),
        "loss_type": loss_type,
        "device": str(device),
        "initial_total_loss": initial_total,
        "final_total_loss": final_total,
        "initial_mae_px": initial_metrics["mae_px"],
        "final_mae_px": final_metrics["mae_px"],
        "initial_inside_ratio": initial_metrics["inside_ratio"],
        "final_inside_ratio": final_metrics["inside_ratio"],
    }
    return corrected.astype(np.float32), params_np, stats


def apply_inverse_depth_params_fullres(
    depth0_np: np.ndarray,
    params_np: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    fit_mode: FitMode,
    max_rel_inv_correction: float,
    min_depth: float,
    device_name: str,
    chunk_rows: int = 256,
) -> np.ndarray:
    if fit_mode == "none" or params_np.size == 0:
        return depth0_np.astype(np.float32).copy()
    device = torch.device(device_name)
    dtype = torch.float32
    h, w = height, width
    nbx = (w + block_size - 1) // block_size
    params = torch.from_numpy(params_np).to(device=device, dtype=dtype)
    depth0 = torch.from_numpy(depth0_np.astype(np.float32)).to(device=device, dtype=dtype)
    out = torch.empty_like(depth0)

    for yy0 in range(0, h, chunk_rows):
        yy1 = min(yy0 + chunk_rows, h)
        yy, xx = torch.meshgrid(
            torch.arange(yy0, yy1, device=device, dtype=dtype),
            torch.arange(0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        xi = xx.to(torch.long)
        yi = yy.to(torch.long)
        bx = torch.div(xi, block_size, rounding_mode="floor")
        by = torch.div(yi, block_size, rounding_mode="floor")
        bid = by * nbx + bx
        x0 = bx * block_size
        y0 = by * block_size
        x1 = torch.clamp(x0 + block_size, max=w)
        y1 = torch.clamp(y0 + block_size, max=h)
        cx = (x0 + x1 - 1).to(dtype) * 0.5
        cy = (y0 + y1 - 1).to(dtype) * 0.5
        xn = (xx - cx) / float(block_size)
        yn = (yy - cy) / float(block_size)
        p = params[bid.reshape(-1)].reshape(yy1 - yy0, w, 3)
        if fit_mode == "bias":
            plane = p[..., 2]
        else:
            plane = p[..., 0] * xn + p[..., 1] * yn + p[..., 2]
        rel = float(max_rel_inv_correction) * torch.tanh(plane)
        d0 = depth0[yy0:yy1]
        inv0 = 1.0 / torch.clamp(d0, min=float(min_depth))
        inv_corr = inv0 * (1.0 + rel)
        dc = 1.0 / torch.clamp(inv_corr, min=float(1.0 / 1e12))
        dc = torch.where(torch.isfinite(d0) & (d0 > float(min_depth)), dc, d0)
        out[yy0:yy1] = dc
    return out.detach().cpu().numpy().astype(np.float32)


def write_fit_params_csv(path: str, params: np.ndarray, width: int, height: int, block_size: int) -> None:
    if params.size == 0:
        return
    nbx = (width + block_size - 1) // block_size
    nby = (height + block_size - 1) // block_size
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "block_x", "block_y", "x0", "y0", "w", "h", "a", "b", "c"])
        for by in range(nby):
            for bx in range(nbx):
                bid = by * nbx + bx
                x0 = bx * block_size
                y0 = by * block_size
                bw = min(block_size, width - x0)
                bh = min(block_size, height - y0)
                a, b, c = params[bid].tolist()
                writer.writerow([bid, bx, by, x0, y0, bw, bh, a, b, c])


def save_diff_png(path: str, target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, bit_depth: int) -> None:
    diff = np.abs(target_y.astype(np.float32) - pred_y.astype(np.float32))
    if np.any(valid):
        scale = float(np.percentile(diff[valid], 99))
    else:
        scale = float(np.percentile(diff, 99))
    scale = max(scale, 1.0)
    diff8 = np.clip(diff / scale * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(diff8, cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)
    cv2.imwrite(path, color)


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT depth/camera warp + residual screen homography guided depth fitting")
    p.add_argument("--yuv", required=True, help="Original source YUV sequence")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    p.add_argument("--ref-idx", type=int, required=True)
    p.add_argument("--tar-idx", type=int, required=True)
    p.add_argument("--camera-jsonl", required=True)
    p.add_argument("--depth-yuv", default=None)
    p.add_argument("--depth-pix-fmt", default="yuv420p10le")
    p.add_argument("--npz", default=None, help="Use raw float depth from *_vggt_omega_outputs.npz")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--target-output", default=None)

    p.add_argument("--tenbit-shift-right", type=int, default=0)
    p.add_argument("--interp", choices=["linear", "nearest", "cubic"], default="linear")
    p.add_argument("--border", choices=["constant", "replicate"], default="constant")
    p.add_argument("--invalid-fill", choices=["black", "neutral", "copy_target"], default="black")
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--no-write-mask", action="store_true")
    p.add_argument("--write-before-fit", action="store_true")
    p.add_argument("--write-residual-h-warp", action="store_true")
    p.add_argument("--no-write-fitted-depth", action="store_true")

    # Residual H estimation.
    p.add_argument("--h-block-size", type=int, default=64)
    p.add_argument("--h-min-valid-ratio", type=float, default=0.50)
    p.add_argument("--h-min-gain", type=float, default=0.0)
    p.add_argument("--h-max-corner-disp", type=float, default=24.0)
    p.add_argument("--h-max-features", type=int, default=500)
    p.add_argument("--h-match-ratio", type=float, default=0.75)
    p.add_argument("--h-ransac-thresh", type=float, default=2.0)
    p.add_argument("--h-min-matches", type=int, default=8)
    p.add_argument("--h-min-inliers", type=int, default=6)
    p.add_argument("--h-no-clahe", action="store_true")
    p.add_argument("--no-h-translation-sweep", action="store_true")
    p.add_argument("--h-translation-max-shift", type=int, default=4)
    p.add_argument("--h-translation-step", type=int, default=1)

    # Depth fitting.
    p.add_argument("--fit-mode", choices=["none", "bias", "plane"], default="plane")
    p.add_argument("--fit-device", default="cuda")
    p.add_argument("--fit-block-size", type=int, default=64)
    p.add_argument("--fit-sample-stride", type=int, default=4)
    p.add_argument("--fit-iters", type=int, default=300)
    p.add_argument("--fit-lr", type=float, default=0.03)
    p.add_argument("--fit-loss", choices=["charbonnier", "l1", "l2"], default="charbonnier")
    p.add_argument("--max-rel-inv-correction", type=float, default=0.30)
    p.add_argument("--fit-reg-lambda", type=float, default=1e-4)
    p.add_argument("--fit-smooth-lambda", type=float, default=1e-3)
    p.add_argument("--target-blend", type=float, default=1.0, help="0=initial cam map, 1=residual-H composed map")
    p.add_argument("--fit-print-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pix_fmt: PixFmt = normalize_pix_fmt(args.pix_fmt)
    depth_pix_fmt: PixFmt = normalize_pix_fmt(args.depth_pix_fmt)
    bit_depth = 8 if pix_fmt == "yuv420p" else 10
    write_mask = not args.no_write_mask
    write_fitted_depth = not args.no_write_fitted_depth

    if args.npz is None and args.depth_yuv is None:
        raise ValueError("Provide either --depth-yuv or --npz")
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    records = load_camera_jsonl(args.camera_jsonl)
    pos_by_frame = {int(r["frame_idx"]): i for i, r in enumerate(records)}
    if args.ref_idx not in pos_by_frame:
        raise ValueError(f"ref_idx {args.ref_idx} not found in camera JSONL")
    if args.tar_idx not in pos_by_frame:
        raise ValueError(f"tar_idx {args.tar_idx} not found in camera JSONL")
    ref_pos = pos_by_frame[args.ref_idx]
    tar_pos = pos_by_frame[args.tar_idx]
    ref_rec = records[ref_pos]
    tar_rec = records[tar_pos]

    K_ref = as_k3(ref_rec["intrinsic_original"])
    K_tar = as_k3(tar_rec["intrinsic_original"])
    R_ref, t_ref = as_rt34(ref_rec["extrinsic"])
    R_tar, t_tar = as_rt34(tar_rec["extrinsic"])
    cameras = (K_ref, R_ref, t_ref, K_tar, R_tar, t_tar)

    ref_yuv = read_yuv420_frame(args.yuv, args.ref_idx, args.width, args.height, pix_fmt, args.tenbit_shift_right)
    tar_yuv = read_yuv420_frame(args.yuv, args.tar_idx, args.width, args.height, pix_fmt, args.tenbit_shift_right)
    ref_y, ref_u, ref_v = ref_yuv
    tar_y, tar_u, tar_v = tar_yuv

    out_target = args.target_output or (args.output_prefix + "_target.yuv")
    write_yuv420_frame(out_target, tar_y, tar_u, tar_v, pix_fmt, bit_depth)

    if args.npz:
        depth0 = load_depth_from_npz(args.npz, args.tar_idx)
        depth_source = args.npz
    else:
        depth_meta = tar_rec.get("depth_output", {})
        for key in ["quant_min", "quant_max", "depth_quant_mode"]:
            if key not in depth_meta:
                raise ValueError(f"camera JSONL target record has no depth_output.{key}; use --npz instead")
        depth0 = load_quantized_depth_yuv(args.depth_yuv, tar_pos, args.width, args.height, depth_pix_fmt, depth_meta)
        depth_source = args.depth_yuv
    if depth0.shape != (args.height, args.width):
        raise ValueError(f"depth shape {depth0.shape} != {(args.height, args.width)}")

    # Initial camera/depth projection and warp.
    init_warp, init_valid, init_stats, init_mae, init_psnr, init_map_x, init_map_y = warp_yuv_with_depth(
        depth_tar=depth0,
        ref_yuv=ref_yuv,
        tar_yuv=tar_yuv,
        cameras=cameras,
        bit_depth=bit_depth,
        interp_name=args.interp,
        border_name=args.border,
        invalid_fill=args.invalid_fill,
        min_depth=args.min_depth,
        chunk_rows=args.chunk_rows,
    )
    init_warp_y = init_warp[0]

    out_before = None
    if args.write_before_fit:
        out_before = args.output_prefix + "_warped_before_fit.yuv"
        write_yuv420_frame(out_before, init_warp[0], init_warp[1], init_warp[2], pix_fmt, bit_depth)
        print(f"before-fit warped yuv: {out_before}")

    # Estimate local residual screen homographies and compose with initial projection map.
    print("[1/3] Estimating residual screen homographies from initial warp vs target...")
    h_records, desired_map_x, desired_map_y, desired_valid, h_summary = estimate_residual_homographies(
        target_y=tar_y,
        ref_y=ref_y,
        init_warp_y=init_warp_y,
        init_map_x=init_map_x,
        init_map_y=init_map_y,
        init_valid=init_valid,
        bit_depth=bit_depth,
        block_size=args.h_block_size,
        min_valid_ratio=args.h_min_valid_ratio,
        min_gain=args.h_min_gain,
        max_corner_disp=args.h_max_corner_disp,
        max_features=args.h_max_features,
        match_ratio=args.h_match_ratio,
        ransac_thresh=args.h_ransac_thresh,
        min_matches=args.h_min_matches,
        min_inliers=args.h_min_inliers,
        use_clahe=not args.h_no_clahe,
        enable_translation_sweep=not args.no_h_translation_sweep,
        translation_max_shift=args.h_translation_max_shift,
        translation_step=args.h_translation_step,
    )
    print(json.dumps(h_summary, indent=2))

    residual_h_mae = residual_h_psnr = None
    out_residual_h_warp = None
    if args.write_residual_h_warp:
        residual_h_warp, _raw, _cvalid, residual_h_mae, residual_h_psnr = warp_yuv_from_map(
            ref_yuv, tar_yuv, desired_map_x, desired_map_y, desired_valid,
            bit_depth, args.interp, args.border, args.invalid_fill,
        )
        out_residual_h_warp = args.output_prefix + "_residual_h_composed_warp.yuv"
        write_yuv420_frame(out_residual_h_warp, residual_h_warp[0], residual_h_warp[1], residual_h_warp[2], pix_fmt, bit_depth)
        print(f"residual-H composed warped yuv: {out_residual_h_warp}")

    # Fit depth to desired ref XY map.
    print("[2/3] Fitting block-wise inverse-depth residual planes to residual-H composed map...")
    fitted_depth, fit_params, fit_stats = fit_inverse_depth_planes_to_xy_torch(
        depth0_np=depth0,
        desired_map_x=desired_map_x,
        desired_map_y=desired_map_y,
        desired_valid=desired_valid,
        init_map_x=init_map_x,
        init_map_y=init_map_y,
        K_ref_np=K_ref,
        R_ref_np=R_ref,
        t_ref_np=t_ref,
        K_tar_np=K_tar,
        R_tar_np=R_tar,
        t_tar_np=t_tar,
        fit_mode=args.fit_mode,
        block_size=args.fit_block_size,
        sample_stride=args.fit_sample_stride,
        iters=args.fit_iters,
        lr=args.fit_lr,
        max_rel_inv_correction=args.max_rel_inv_correction,
        reg_lambda=args.fit_reg_lambda,
        smooth_lambda=args.fit_smooth_lambda,
        min_depth=args.min_depth,
        device_name=args.fit_device,
        loss_type=args.fit_loss,
        target_blend=args.target_blend,
        print_every=args.fit_print_every,
    )

    out_fit_csv = args.output_prefix + "_fit_params.csv"
    if args.fit_mode != "none":
        write_fit_params_csv(out_fit_csv, fit_params, args.width, args.height, args.fit_block_size)

    fitted_depth_meta = None
    out_fitted_depth = None
    if write_fitted_depth:
        out_fitted_depth = args.output_prefix + "_fitted_depth_inverse_yuv420p10le.yuv"
        fitted_depth_meta = write_inverse_depth_yuv420p10le(out_fitted_depth, fitted_depth)

    # Final warp.
    print("[3/3] Rendering final warp with fitted depth...")
    final_warp, final_valid, final_stats, final_mae, final_psnr, final_map_x, final_map_y = warp_yuv_with_depth(
        depth_tar=fitted_depth,
        ref_yuv=ref_yuv,
        tar_yuv=tar_yuv,
        cameras=cameras,
        bit_depth=bit_depth,
        interp_name=args.interp,
        border_name=args.border,
        invalid_fill=args.invalid_fill,
        min_depth=args.min_depth,
        chunk_rows=args.chunk_rows,
    )
    out_warp = args.output_prefix + "_warped.yuv"
    write_yuv420_frame(out_warp, final_warp[0], final_warp[1], final_warp[2], pix_fmt, bit_depth)

    out_mask = args.output_prefix + "_valid_mask_yuv420p.yuv"
    if write_mask:
        write_mask_yuv420p(out_mask, final_valid)

    out_diff = args.output_prefix + "_diff.png"
    save_diff_png(out_diff, tar_y, final_warp[0], final_valid, bit_depth)

    # Save residual-H JSON.
    out_h_json = args.output_prefix + "_residual_homography.json"
    h_json_records = []
    for r in h_records:
        h_json_records.append({
            "block_x": int(r.block_x),
            "block_y": int(r.block_y),
            "block_w": int(r.block_w),
            "block_h": int(r.block_h),
            "H_target_to_initial_warp_screen": r.H.tolist(),
            "source": r.source,
            "accepted": bool(r.accepted),
            "match_count": int(r.match_count),
            "inlier_count": int(r.inlier_count),
            "base_cost": r.base_cost,
            "candidate_cost": r.candidate_cost,
            "chosen_cost": r.chosen_cost,
            "valid_ratio": float(r.valid_ratio),
            "max_corner_disp": float(r.max_corner_disp),
            "reason": r.reason,
        })
    with open(out_h_json, "w", encoding="utf-8") as f:
        json.dump({"summary": h_summary, "blocks": h_json_records}, f, indent=2)

    stats = dict(final_stats)
    stats.update({
        "source_yuv": os.path.abspath(args.yuv),
        "depth_source": os.path.abspath(depth_source),
        "camera_jsonl": os.path.abspath(args.camera_jsonl),
        "ref_idx": int(args.ref_idx),
        "tar_idx": int(args.tar_idx),
        "ref_camera_jsonl_position": int(ref_pos),
        "tar_camera_jsonl_position": int(tar_pos),
        "width": int(args.width),
        "height": int(args.height),
        "pix_fmt": pix_fmt,
        "depth0_min": float(np.nanmin(depth0)),
        "depth0_max": float(np.nanmax(depth0)),
        "depth0_mean": float(np.nanmean(depth0)),
        "fitted_depth_min": float(np.nanmin(fitted_depth)),
        "fitted_depth_max": float(np.nanmax(fitted_depth)),
        "fitted_depth_mean": float(np.nanmean(fitted_depth)),
        "before_fit_warp_y_mae_valid": init_mae,
        "before_fit_warp_y_psnr_valid": init_psnr,
        "residual_h_composed_warp_y_mae_valid": residual_h_mae,
        "residual_h_composed_warp_y_psnr_valid": residual_h_psnr,
        "after_fit_warp_y_mae_valid": final_mae,
        "after_fit_warp_y_psnr_valid": final_psnr,
        "initial_projection_stats": init_stats,
        "residual_homography_summary": h_summary,
        "fit_stats": fit_stats,
        "fitted_depth_output_meta": fitted_depth_meta,
        "invalid_fill": args.invalid_fill,
        "output_warped_yuv": os.path.abspath(out_warp),
        "output_target_yuv": os.path.abspath(out_target),
        "output_before_fit_yuv": os.path.abspath(out_before) if out_before else None,
        "output_residual_h_composed_warp_yuv": os.path.abspath(out_residual_h_warp) if out_residual_h_warp else None,
        "output_valid_mask_yuv420p": os.path.abspath(out_mask) if write_mask else None,
        "output_fitted_depth_yuv": os.path.abspath(out_fitted_depth) if out_fitted_depth else None,
        "output_fit_params_csv": os.path.abspath(out_fit_csv) if args.fit_mode != "none" else None,
        "output_residual_homography_json": os.path.abspath(out_h_json),
        "output_diff_png": os.path.abspath(out_diff),
        "note": "Camera fixed. Residual H is estimated between initial cam/depth warp and target in target-screen coordinates, then composed with initial projection map. Depth is fitted to that composed ref-XY target.",
    })
    out_stats = args.output_prefix + "_map_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done")
    print(f"  warped yuv        : {out_warp}")
    print(f"  target yuv        : {out_target}")
    if out_before:
        print(f"  before-fit yuv    : {out_before}")
    if out_residual_h_warp:
        print(f"  residual-H warp   : {out_residual_h_warp}")
    if write_mask:
        print(f"  valid mask        : {out_mask}")
    if out_fitted_depth:
        print(f"  fitted depth yuv  : {out_fitted_depth}")
    if args.fit_mode != "none":
        print(f"  fit params csv    : {out_fit_csv}")
    print(f"  residual H json   : {out_h_json}")
    print(f"  stats             : {out_stats}")
    print(f"  diff png          : {out_diff}")
    print(f"  valid ratio       : {stats['projection_inside_ref_ratio']:.6f}")
    if init_mae is not None:
        print(f"  before Y MAE(valid): {init_mae:.6f}")
    if residual_h_mae is not None:
        print(f"  res-H  Y MAE(valid): {residual_h_mae:.6f}")
    if final_mae is not None:
        print(f"  after  Y MAE(valid): {final_mae:.6f}")


if __name__ == "__main__":
    main()
