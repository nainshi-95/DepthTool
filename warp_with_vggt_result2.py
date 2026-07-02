#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp a reference YUV frame to a target frame using VGGT-Omega depth + camera output,
with optional block-wise inverse-depth residual plane fitting on CUDA/PyTorch.

Main idea:
  For each target pixel in a block:
    inv_z'(x,y) = inv_z(x,y) * (1 + max_rel * tanh(a*xn + b*yn + c))

  a,b,c are learnable parameters per block.
  Camera parameters are fixed.
  The fitting minimizes photometric error between:
    bilinear_sample(ref_y, projected_ref_xy(inv_z')) and target_y.

Default fitting:
  --fit-mode plane
  --fit-block-size 32
  --fit-sample-stride 4

Inputs from run_vggt_omega_yuv.py:
  - original YUV sequence
  - *_depth_inverse_yuv420p10le.yuv, or *_vggt_omega_outputs.npz
  - *_camera.jsonl

Outputs:
  <prefix>_warped.yuv                         warped ref image using fitted depth
  <prefix>_target.yuv                         original target frame
  <prefix>_valid_mask_yuv420p.yuv             valid projection mask after fitting
  <prefix>_fitted_depth_inverse_yuv420p10le.yuv  fitted target depth as inverse-depth YUV
  <prefix>_fit_params.csv                     block-wise a,b,c parameters
  <prefix>_map_stats.json                     statistics

Example:
  python warp_vggt_omega_yuv_fit_depth.py \
    --yuv input.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p10le \
    --depth-yuv out/test_depth_inverse_yuv420p10le.yuv \
    --camera-jsonl out/test_camera.jsonl \
    --ref-idx 0 --tar-idx 7 \
    --output-prefix out/warp_ref000_to_tar007_fit \
    --fit-mode plane --fit-block-size 32 --fit-sample-stride 4 --fit-iters 200 --fit-lr 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Literal

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError("PyTorch is required for depth fitting: pip install torch") from exc

PixFmt = Literal["yuv420p", "yuv420p10le"]
DepthMode = Literal["linear", "inverse"]
FitMode = Literal["none", "bias", "plane"]
InvalidFill = Literal["black", "copy_target", "neutral"]


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


def write_yuv420_frame(
    path: str,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pix_fmt: PixFmt,
    bit_depth: int,
) -> None:
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
    """Write a single-frame inverse-depth visualization/transport YUV420p10le."""
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
    return {
        "depth_quant_mode": "inverse",
        "quant_min": qmin,
        "quant_max": qmax,
        "pix_fmt": "yuv420p10le",
    }


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
# Numpy projection + YUV warping
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
    """For each target pixel, compute corresponding reference pixel coordinate."""
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

        ones = np.ones_like(z)
        pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), ones], axis=0).reshape(3, -1)
        rays_tar = inv_K_tar @ pix
        z_flat = z.reshape(-1)
        x_tar = rays_tar * z_flat[None, :]

        # camera_from_world: X_cam = R * X_world + t
        # world_from_target: X_world = R_tar.T * (X_tar - t_tar)
        x_world = Rt_tar @ (x_tar - t_tar.reshape(3, 1))
        x_ref = R_ref @ x_world + t_ref.reshape(3, 1)

        zr = x_ref[2]
        in_front = zr > min_depth
        total_in_front_ref += int((in_front & depth_ok.reshape(-1)).sum())

        proj = K_ref @ x_ref
        xr = proj[0] / np.maximum(proj[2], min_depth)
        yr = proj[1] / np.maximum(proj[2], min_depth)

        inside = (xr >= 0.0) & (xr <= w - 1.0) & (yr >= 0.0) & (yr <= h - 1.0)
        ok = depth_ok.reshape(-1) & in_front & inside & np.isfinite(xr) & np.isfinite(yr)

        mx = map_x[y0:y1].reshape(-1)
        my = map_y[y0:y1].reshape(-1)
        vv = valid[y0:y1].reshape(-1)
        mx[ok] = xr[ok].astype(np.float32)
        my[ok] = yr[ok].astype(np.float32)
        vv[ok] = True

    stats = {
        "pixels": int(h * w),
        "target_depth_valid": int(total_z_valid),
        "target_depth_valid_ratio": float(total_z_valid / max(h * w, 1)),
        "in_front_of_ref_camera": int(total_in_front_ref),
        "projection_inside_ref": int(valid.sum()),
        "projection_inside_ref_ratio": float(valid.mean()),
    }
    return map_x, map_y, valid, stats


def remap_plane(
    plane: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: float,
) -> np.ndarray:
    remapped = cv2.remap(
        plane.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=float(border_value),
    )
    remapped[~valid] = border_value
    return remapped


def chroma_maps_from_luma(
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map_x.shape
    cw, ch = w // 2, h // 2
    cmx = cv2.resize(map_x, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cmy = cv2.resize(map_y, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cvalid_f = cv2.resize(valid.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    cvalid = cvalid_f > 0.999
    cmx[~cvalid] = -1.0
    cmy[~cvalid] = -1.0
    return cmx.astype(np.float32), cmy.astype(np.float32), cvalid


def fill_invalid_with_target(
    warped: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    valid_luma: np.ndarray,
    valid_chroma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wy, wu, wv = warped
    ty, tu, tv = target
    out_y = wy.copy()
    out_u = wu.copy()
    out_v = wv.copy()
    out_y[~valid_luma] = ty.astype(np.float32)[~valid_luma]
    out_u[~valid_chroma] = tu.astype(np.float32)[~valid_chroma]
    out_v[~valid_chroma] = tv.astype(np.float32)[~valid_chroma]
    return out_y, out_u, out_v


def y_mae_psnr(
    pred_y: np.ndarray,
    target_y: np.ndarray,
    valid: np.ndarray,
    bit_depth: int,
) -> tuple[float | None, float | None]:
    if valid.sum() == 0:
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
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict, float | None, float | None]:
    K_ref, R_ref, t_ref, K_tar, R_tar, t_tar = cameras
    ref_y, ref_u, ref_v = ref_yuv
    tar_y, tar_u, tar_v = tar_yuv
    neutral = 128 if bit_depth == 8 else 512

    map_x, map_y, valid, stats = make_backward_map(
        depth_tar=depth_tar,
        K_ref=K_ref,
        R_ref=R_ref,
        t_ref=t_ref,
        K_tar=K_tar,
        R_tar=R_tar,
        t_tar=t_tar,
        min_depth=min_depth,
        chunk_rows=chunk_rows,
    )

    if interp_name == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp_name == "cubic":
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    border_mode = cv2.BORDER_REPLICATE if border_name == "replicate" else cv2.BORDER_CONSTANT
    y_fill = 0.0 if invalid_fill in ["black", "copy_target"] else float(neutral)
    uv_fill = float(neutral)

    raw_wy = remap_plane(ref_y, map_x, map_y, valid, interp, border_mode, y_fill)
    cmx, cmy, cvalid = chroma_maps_from_luma(map_x, map_y, valid)
    raw_wu = remap_plane(ref_u, cmx, cmy, cvalid, interp, border_mode, uv_fill)
    raw_wv = remap_plane(ref_v, cmx, cmy, cvalid, interp, border_mode, uv_fill)

    wy, wu, wv = raw_wy, raw_wu, raw_wv
    if invalid_fill == "copy_target":
        wy, wu, wv = fill_invalid_with_target((raw_wy, raw_wu, raw_wv), (tar_y, tar_u, tar_v), valid, cvalid)

    raw_mae, raw_psnr = y_mae_psnr(raw_wy, tar_y, valid, bit_depth)
    return (wy, wu, wv), valid, stats, raw_mae, raw_psnr


# ============================================================
# Torch block-wise inverse-depth residual plane fitting
# ============================================================

def _torch_camera_project_xy(
    depth: torch.Tensor,
    pix: torch.Tensor,
    rays_tar: torch.Tensor,
    K_ref: torch.Tensor,
    R_ref: torch.Tensor,
    t_ref: torch.Tensor,
    R_tar_t: torch.Tensor,
    t_tar: torch.Tensor,
    min_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project target pixels with target depth into reference pixel coordinates."""
    # x_tar = invK_tar @ pix * depth; rays_tar is precomputed invK_tar @ pix.
    x_tar = rays_tar * depth.unsqueeze(0)
    x_world = R_tar_t @ (x_tar - t_tar.reshape(3, 1))
    x_ref = R_ref @ x_world + t_ref.reshape(3, 1)
    zr = x_ref[2]
    proj = K_ref @ x_ref
    denom = torch.clamp(proj[2], min=float(min_depth))
    xr = proj[0] / denom
    yr = proj[1] / denom
    return xr, yr, zr


def _smoothness_loss(params: torch.Tensor, nby: int, nbx: int) -> torch.Tensor:
    p = params.reshape(nby, nbx, 3)
    loss = p.new_tensor(0.0)
    cnt = 0
    if nbx > 1:
        loss = loss + (p[:, 1:, :] - p[:, :-1, :]).pow(2).mean()
        cnt += 1
    if nby > 1:
        loss = loss + (p[1:, :, :] - p[:-1, :, :]).pow(2).mean()
        cnt += 1
    if cnt == 0:
        return loss
    return loss / cnt


def fit_inverse_depth_planes_torch(
    depth0_np: np.ndarray,
    ref_y_np: np.ndarray,
    tar_y_np: np.ndarray,
    K_ref_np: np.ndarray,
    R_ref_np: np.ndarray,
    t_ref_np: np.ndarray,
    K_tar_np: np.ndarray,
    R_tar_np: np.ndarray,
    t_tar_np: np.ndarray,
    bit_depth: int,
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
    print_every: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit per-block inverse-depth correction using target/ref luma photometric loss.

    Correction parameterization:
      plane = c                      for fit_mode='bias'
      plane = a*xn + b*yn + c         for fit_mode='plane'
      inv_z' = inv_z * (1 + max_rel * tanh(plane))

    Returns corrected_depth, params_np [num_blocks,3], fit_stats.
    """
    if fit_mode == "none":
        return depth0_np.astype(np.float32), np.zeros((0, 3), dtype=np.float32), {"fit_mode": "none"}
    if block_size <= 0:
        raise ValueError("--fit-block-size must be positive")
    if sample_stride <= 0:
        raise ValueError("--fit-sample-stride must be positive")
    if iters <= 0:
        raise ValueError("--fit-iters must be positive when fitting is enabled")
    if max_rel_inv_correction <= 0.0 or max_rel_inv_correction >= 0.99:
        raise ValueError("--max-rel-inv-correction should be in (0, 0.99)")

    h, w = depth0_np.shape
    nbx = (w + block_size - 1) // block_size
    nby = (h + block_size - 1) // block_size
    num_blocks = nbx * nby

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--fit-device cuda was requested, but CUDA is not available")
    device = torch.device(device_name)
    dtype = torch.float32

    maxv = float((1 << bit_depth) - 1)

    # Sparse sample grid.
    ys_np = np.arange(0, h, sample_stride, dtype=np.float32)
    xs_np = np.arange(0, w, sample_stride, dtype=np.float32)
    grid_y_np, grid_x_np = np.meshgrid(ys_np, xs_np, indexing="ij")
    xs_flat_np = grid_x_np.reshape(-1)
    ys_flat_np = grid_y_np.reshape(-1)
    xi_np = np.clip(np.rint(xs_flat_np).astype(np.int64), 0, w - 1)
    yi_np = np.clip(np.rint(ys_flat_np).astype(np.int64), 0, h - 1)
    n_samples = int(xs_flat_np.size)

    bx_np = xi_np // block_size
    by_np = yi_np // block_size
    block_id_np = by_np * nbx + bx_np

    # Normalize local coordinates around each actual block center.
    x0_np = bx_np * block_size
    y0_np = by_np * block_size
    x1_np = np.minimum(x0_np + block_size, w)
    y1_np = np.minimum(y0_np + block_size, h)
    cx_np = (x0_np + x1_np - 1) * 0.5
    cy_np = (y0_np + y1_np - 1) * 0.5
    xn_np = (xs_flat_np - cx_np.astype(np.float32)) / float(block_size)
    yn_np = (ys_flat_np - cy_np.astype(np.float32)) / float(block_size)

    depth_s_np = depth0_np[yi_np, xi_np].astype(np.float32)
    depth_ok_np = np.isfinite(depth_s_np) & (depth_s_np > min_depth)
    inv0_np = np.zeros_like(depth_s_np, dtype=np.float32)
    inv0_np[depth_ok_np] = 1.0 / np.maximum(depth_s_np[depth_ok_np], min_depth)
    target_np = tar_y_np[yi_np, xi_np].astype(np.float32) / maxv

    xs = torch.from_numpy(xs_flat_np).to(device=device, dtype=dtype)
    ys = torch.from_numpy(ys_flat_np).to(device=device, dtype=dtype)
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0)

    inv_K_tar = torch.linalg.inv(torch.as_tensor(K_tar_np, device=device, dtype=dtype))
    rays_tar = inv_K_tar @ pix

    K_ref = torch.as_tensor(K_ref_np, device=device, dtype=dtype)
    R_ref = torch.as_tensor(R_ref_np, device=device, dtype=dtype)
    t_ref = torch.as_tensor(t_ref_np, device=device, dtype=dtype)
    R_tar_t = torch.as_tensor(R_tar_np.T, device=device, dtype=dtype)
    t_tar = torch.as_tensor(t_tar_np, device=device, dtype=dtype)

    block_id = torch.from_numpy(block_id_np).to(device=device, dtype=torch.long)
    xn = torch.from_numpy(xn_np).to(device=device, dtype=dtype)
    yn = torch.from_numpy(yn_np).to(device=device, dtype=dtype)
    inv0 = torch.from_numpy(inv0_np).to(device=device, dtype=dtype)
    depth_ok = torch.from_numpy(depth_ok_np).to(device=device, dtype=torch.bool)
    target = torch.from_numpy(target_np).to(device=device, dtype=dtype)

    ref_img = torch.from_numpy(ref_y_np.astype(np.float32) / maxv).to(device=device, dtype=dtype)
    ref_img = ref_img.reshape(1, 1, h, w)

    params = torch.nn.Parameter(torch.zeros((num_blocks, 3), device=device, dtype=dtype))
    opt = torch.optim.Adam([params], lr=lr)

    def compute_loss_and_metrics() -> tuple[torch.Tensor, dict]:
        p = params[block_id]
        if fit_mode == "bias":
            plane = p[:, 2]
        else:
            plane = p[:, 0] * xn + p[:, 1] * yn + p[:, 2]

        rel = float(max_rel_inv_correction) * torch.tanh(plane)
        inv_corr = inv0 * (1.0 + rel)
        depth = 1.0 / torch.clamp(inv_corr, min=float(1.0 / 1e12))

        xr, yr, zr = _torch_camera_project_xy(
            depth=depth,
            pix=pix,
            rays_tar=rays_tar,
            K_ref=K_ref,
            R_ref=R_ref,
            t_ref=t_ref,
            R_tar_t=R_tar_t,
            t_tar=t_tar,
            min_depth=min_depth,
        )

        inside = (xr >= 0.0) & (xr <= (w - 1.0)) & (yr >= 0.0) & (yr <= (h - 1.0))
        valid = depth_ok & (zr > float(min_depth)) & inside & torch.isfinite(xr) & torch.isfinite(yr)

        gx = 2.0 * xr / max(float(w - 1), 1.0) - 1.0
        gy = 2.0 * yr / max(float(h - 1), 1.0) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(1, n_samples, 1, 2)
        sampled = F.grid_sample(
            ref_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(-1)

        if valid.any():
            diff = sampled[valid] - target[valid]
            if loss_type == "l2":
                photo_loss = (diff * diff).mean()
            elif loss_type == "l1":
                photo_loss = diff.abs().mean()
            else:
                photo_loss = torch.sqrt(diff * diff + 1e-6).mean()
            mae = diff.abs().mean()
        else:
            # Keep graph connected.
            photo_loss = sampled.mean() * 0.0 + 1.0
            mae = sampled.mean() * 0.0 + 1.0

        reg_loss = params.pow(2).mean()
        smooth_loss = _smoothness_loss(params, nby, nbx)
        total_loss = photo_loss + float(reg_lambda) * reg_loss + float(smooth_lambda) * smooth_loss
        metrics = {
            "photo_loss": float(photo_loss.detach().cpu()),
            "mae_norm": float(mae.detach().cpu()),
            "valid_samples": int(valid.detach().sum().cpu()),
            "valid_sample_ratio": float(valid.detach().float().mean().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "smooth_loss": float(smooth_loss.detach().cpu()),
        }
        return total_loss, metrics

    with torch.no_grad():
        initial_loss, initial_metrics = compute_loss_and_metrics()
        initial_total = float(initial_loss.detach().cpu())

    if initial_metrics["valid_samples"] <= 0:
        raise RuntimeError("No valid samples before fitting. Check camera/depth scale/convention.")

    final_metrics = initial_metrics
    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, metrics = compute_loss_and_metrics()
        loss.backward()
        opt.step()
        final_metrics = metrics
        if print_every > 0 and (it == 1 or it % print_every == 0 or it == iters):
            print(
                f"fit iter {it:04d}/{iters}: "
                f"loss={float(loss.detach().cpu()):.8f}, "
                f"photo={metrics['photo_loss']:.8f}, "
                f"mae_norm={metrics['mae_norm']:.8f}, "
                f"valid={metrics['valid_sample_ratio']:.4f}"
            )

    with torch.no_grad():
        final_loss, final_metrics = compute_loss_and_metrics()
        final_total = float(final_loss.detach().cpu())
        params_np = params.detach().cpu().numpy().astype(np.float32)

    # Apply the fitted correction to the full-resolution depth map.
    corrected = apply_inverse_depth_params_fullres(
        depth0_np=depth0_np,
        params_np=params_np,
        width=w,
        height=h,
        block_size=block_size,
        fit_mode=fit_mode,
        max_rel_inv_correction=max_rel_inv_correction,
        min_depth=min_depth,
        device_name=device_name,
        chunk_rows=256,
    )

    stats = {
        "fit_mode": fit_mode,
        "fit_param_type": "bounded_relative_inverse_depth_residual_plane",
        "formula": "inv_z_prime = inv_z * (1 + max_rel * tanh(a*xn + b*yn + c))",
        "block_size": int(block_size),
        "num_blocks_x": int(nbx),
        "num_blocks_y": int(nby),
        "num_blocks": int(num_blocks),
        "sample_stride": int(sample_stride),
        "num_samples": int(n_samples),
        "iters": int(iters),
        "lr": float(lr),
        "max_rel_inv_correction": float(max_rel_inv_correction),
        "reg_lambda": float(reg_lambda),
        "smooth_lambda": float(smooth_lambda),
        "loss_type": loss_type,
        "device": str(device),
        "initial_total_loss": initial_total,
        "final_total_loss": final_total,
        "initial_photo_loss": initial_metrics["photo_loss"],
        "final_photo_loss": final_metrics["photo_loss"],
        "initial_mae_norm": initial_metrics["mae_norm"],
        "final_mae_norm": final_metrics["mae_norm"],
        "initial_valid_sample_ratio": initial_metrics["valid_sample_ratio"],
        "final_valid_sample_ratio": final_metrics["valid_sample_ratio"],
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

    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=dtype),
            torch.arange(0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        xi = xx.to(torch.long)
        yi = yy.to(torch.long)
        bx = torch.div(xi, block_size, rounding_mode="floor")
        by = torch.div(yi, block_size, rounding_mode="floor")
        bid = by * nbx + bx

        x0 = bx * block_size
        y0b = by * block_size
        x1b = torch.clamp(x0 + block_size, max=w)
        y1b = torch.clamp(y0b + block_size, max=h)
        cx = (x0 + x1b - 1).to(dtype) * 0.5
        cy = (y0b + y1b - 1).to(dtype) * 0.5
        xn = (xx - cx) / float(block_size)
        yn = (yy - cy) / float(block_size)

        p = params[bid.reshape(-1)].reshape(y1 - y0, w, 3)
        if fit_mode == "bias":
            plane = p[..., 2]
        else:
            plane = p[..., 0] * xn + p[..., 1] * yn + p[..., 2]
        rel = float(max_rel_inv_correction) * torch.tanh(plane)

        d0 = depth0[y0:y1]
        inv0 = 1.0 / torch.clamp(d0, min=float(min_depth))
        inv_corr = inv0 * (1.0 + rel)
        d_corr = 1.0 / torch.clamp(inv_corr, min=float(1.0 / 1e12))
        d_corr = torch.where(torch.isfinite(d0) & (d0 > float(min_depth)), d_corr, d0)
        out[y0:y1] = d_corr

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


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warp ref YUV to target using VGGT-Omega depth/camera with block-wise depth fitting")
    p.add_argument("--yuv", required=True, help="Original source YUV sequence")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    p.add_argument("--ref-idx", type=int, required=True, help="absolute frame index of reference frame in original YUV")
    p.add_argument("--tar-idx", type=int, required=True, help="absolute frame index of target frame in original YUV")
    p.add_argument("--camera-jsonl", required=True, help="*_camera.jsonl from run_vggt_omega_yuv.py")
    p.add_argument("--depth-yuv", default=None, help="quantized depth YUV from run_vggt_omega_yuv.py")
    p.add_argument("--depth-pix-fmt", default="yuv420p10le", help="depth YUV pix fmt; default yuv420p10le")
    p.add_argument("--npz", default=None, help="optional *_vggt_omega_outputs.npz. If set, uses raw float depth instead of quantized depth YUV")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--target-output", default=None, help="Optional explicit output path for original target frame YUV")

    p.add_argument("--tenbit-shift-right", type=int, default=0,
                   help="Use 0 for normal yuv420p10le. Use 6 only if samples are MSB-aligned in uint16.")
    p.add_argument("--interp", choices=["linear", "nearest", "cubic"], default="linear")
    p.add_argument("--border", choices=["constant", "replicate"], default="constant")
    p.add_argument("--invalid-fill", choices=["black", "neutral", "copy_target"], default="black",
                   help="How to fill pixels with no valid projection. copy_target fills holes with target.")
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--no-write-mask", action="store_true", help="Disable valid mask YUV output")

    # Depth fitting options.
    p.add_argument("--fit-mode", choices=["none", "bias", "plane"], default="plane",
                   help="none: original behavior, bias: c only, plane: a*xn+b*yn+c. Default: plane")
    p.add_argument("--fit-device", default="cuda", help="cuda or cpu. Default: cuda")
    p.add_argument("--fit-block-size", type=int, default=32)
    p.add_argument("--fit-sample-stride", type=int, default=4,
                   help="Use every N pixels during fitting. 4 means 64 samples per 32x32 block.")
    p.add_argument("--fit-iters", type=int, default=200)
    p.add_argument("--fit-lr", type=float, default=0.05)
    p.add_argument("--fit-loss", choices=["charbonnier", "l1", "l2"], default="charbonnier")
    p.add_argument("--max-rel-inv-correction", type=float, default=0.50,
                   help="Bound for relative inverse-depth correction. 0.5 means inv_z can change by about +/-50%.")
    p.add_argument("--fit-reg-lambda", type=float, default=1e-4)
    p.add_argument("--fit-smooth-lambda", type=float, default=1e-3,
                   help="Neighbor block parameter smoothness regularization.")
    p.add_argument("--fit-print-every", type=int, default=25)
    p.add_argument("--write-before-fit", action="store_true",
                   help="Also write warped result before depth fitting for comparison.")
    p.add_argument("--no-write-fitted-depth", action="store_true",
                   help="Disable fitted inverse-depth YUV output")
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
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    records = load_camera_jsonl(args.camera_jsonl)
    pos_by_frame = {int(r["frame_idx"]): i for i, r in enumerate(records)}
    if args.ref_idx not in pos_by_frame:
        raise ValueError(f"ref_idx {args.ref_idx} not found in camera JSONL. Available: {sorted(pos_by_frame)[:10]}...")
    if args.tar_idx not in pos_by_frame:
        raise ValueError(f"tar_idx {args.tar_idx} not found in camera JSONL. Available: {sorted(pos_by_frame)[:10]}...")

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

    # Save original target immediately.
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
        depth0 = load_quantized_depth_yuv(
            args.depth_yuv,
            tar_pos,
            args.width,
            args.height,
            depth_pix_fmt,
            depth_meta,
        )
        depth_source = args.depth_yuv

    if depth0.shape != (args.height, args.width):
        raise ValueError(f"depth shape {depth0.shape} != {(args.height, args.width)}")

    # Optional before-fit warp.
    before_stats = None
    before_mae = before_psnr = None
    if args.write_before_fit:
        before_warp, before_valid, before_stats, before_mae, before_psnr = warp_yuv_with_depth(
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
        out_before = args.output_prefix + "_warped_before_fit.yuv"
        write_yuv420_frame(out_before, before_warp[0], before_warp[1], before_warp[2], pix_fmt, bit_depth)
        print(f"before-fit warped yuv: {out_before}")

    # Fit depth.
    fitted_depth, fit_params, fit_stats = fit_inverse_depth_planes_torch(
        depth0_np=depth0,
        ref_y_np=ref_y,
        tar_y_np=tar_y,
        K_ref_np=K_ref,
        R_ref_np=R_ref,
        t_ref_np=t_ref,
        K_tar_np=K_tar,
        R_tar_np=R_tar,
        t_tar_np=t_tar,
        bit_depth=bit_depth,
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

    # Warp with fitted depth.
    warped, valid, stats, raw_mae, raw_psnr = warp_yuv_with_depth(
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

    out_yuv = args.output_prefix + "_warped.yuv"
    write_yuv420_frame(out_yuv, warped[0], warped[1], warped[2], pix_fmt, bit_depth)

    out_mask = args.output_prefix + "_valid_mask_yuv420p.yuv"
    if write_mask:
        write_mask_yuv420p(out_mask, valid)

    stats.update(
        {
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
            "before_fit_warp_y_mae_valid": before_mae,
            "before_fit_warp_y_psnr_valid": before_psnr,
            "after_fit_warp_y_mae_valid": raw_mae,
            "after_fit_warp_y_psnr_valid": raw_psnr,
            "invalid_fill": args.invalid_fill,
            "fit_stats": fit_stats,
            "fitted_depth_output_meta": fitted_depth_meta,
            "output_warped_yuv": os.path.abspath(out_yuv),
            "output_target_yuv": os.path.abspath(out_target),
            "output_valid_mask_yuv420p": os.path.abspath(out_mask) if write_mask else None,
            "output_fitted_depth_yuv": os.path.abspath(out_fitted_depth) if out_fitted_depth else None,
            "output_fit_params_csv": os.path.abspath(out_fit_csv) if args.fit_mode != "none" else None,
            "note": "Camera fixed. Depth fitting uses block-wise bounded relative inverse-depth residual plane and luma photometric loss.",
        }
    )
    if before_stats is not None:
        stats["before_fit_projection_stats"] = before_stats

    out_stats = args.output_prefix + "_map_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done")
    print(f"  warped yuv       : {out_yuv}")
    print(f"  target yuv       : {out_target}")
    if write_mask:
        print(f"  valid mask       : {out_mask}")
    if out_fitted_depth:
        print(f"  fitted depth yuv : {out_fitted_depth}")
    if args.fit_mode != "none":
        print(f"  fit params csv   : {out_fit_csv}")
    print(f"  stats            : {out_stats}")
    print(f"  valid ratio      : {stats['projection_inside_ref_ratio']:.6f}")
    if before_mae is not None:
        print(f"  before Y MAE(valid): {before_mae:.6f}")
    if raw_mae is not None:
        print(f"  after  Y MAE(valid): {raw_mae:.6f}")
    if before_psnr is not None:
        print(f"  before Y PSNR(valid): {before_psnr:.6f} dB")
    if raw_psnr is not None:
        print(f"  after  Y PSNR(valid): {raw_psnr:.6f} dB")


if __name__ == "__main__":
    main()
