#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp a reference YUV frame to a target frame using VGGT-Omega depth + fixed camera,
then optionally refine the target depth with block-homography-guided inverse-depth
plane fitting.

Main fitting objective:
  For sampled target pixels (x, y) inside each homography block:

    p_depth_ref(x, y; depth') = project_to_ref_by_fixed_camera(x, y, depth')
    p_H_ref(x, y)             = H_block @ [x, y, 1]

    loss = robust( p_depth_ref - p_H_ref )

Depth correction per block:
  inv_z'(x, y) = inv_z(x, y) * (1 + max_rel * tanh(a*xn + b*yn + c))

where a,b,c are learnable parameters per final homography block.
Camera parameters are fixed.

Inputs:
  - original source YUV sequence
  - VGGT-Omega camera JSONL from run_vggt_omega_yuv.py
  - VGGT-Omega depth: either depth YUV or raw NPZ
  - block homography result JSON from hierarchical_block_homography.py

Example:
  python warp_vggt_omega_yuv_fit_depth_homography.py \
    --yuv input.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p10le \
    --depth-yuv out/test_depth_inverse_yuv420p10le.yuv \
    --camera-jsonl out/test_camera.jsonl \
    --homography-json h_out/result.json \
    --ref-idx 0 --tar-idx 7 \
    --output-prefix out/warp_ref000_to_tar007_hfit \
    --fit-mode plane \
    --fit-sample-stride 4 \
    --fit-iters 300 \
    --fit-lr 0.03 \
    --max-rel-inv-correction 0.30 \
    --write-before-fit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Literal, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required: pip install torch") from exc

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
    depth = z["depth_original"][int(matches[0])].astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth_original for frame {tar_idx} must be 2D after squeeze, got {depth.shape}")
    return depth


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
# Homography result loading
# ============================================================

def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography_np(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = normalize_homography(H)
    denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    valid = np.abs(denom) > 1e-9
    denom_safe = denom + 1e-12
    qx = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom_safe
    qy = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom_safe
    return qx.astype(np.float32), qy.astype(np.float32), valid


def load_homography_result(path: str) -> tuple[int, list[list[dict]], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "levels_blocks" in data:
        records = data["levels_blocks"][-1]
        block_size = int(data["final"]["block_size"])
    elif "final_records" in data:
        records = data["final_records"]
        block_size = int(data["final_block_size"])
    else:
        raise ValueError("Homography JSON must contain levels_blocks or final_records")

    if not records or not records[0]:
        raise ValueError("Homography JSON has empty final block records")
    return block_size, records, data


def parse_sources(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


def build_homography_training_samples(
    records: list[list[dict]],
    width: int,
    height: int,
    block_size: int,
    depth0_np: np.ndarray,
    sample_stride: int,
    use_sources: set[str],
    min_valid_ratio: float,
    max_chosen_cost: float,
    max_reproj_mae: float,
    min_inliers: int,
    min_depth: float,
    cost_weight_power: float,
) -> tuple[dict[str, np.ndarray], dict]:
    xs_all = []
    ys_all = []
    qx_all = []
    qy_all = []
    bid_all = []
    xn_all = []
    yn_all = []
    w_all = []
    block_used = []
    block_info = []

    nby = len(records)
    nbx = len(records[0])
    num_blocks = nby * nbx
    used_mask = np.zeros(num_blocks, dtype=bool)

    rejected = {
        "source": 0,
        "valid_ratio": 0,
        "chosen_cost": 0,
        "reproj_mae": 0,
        "inliers": 0,
        "no_valid_samples": 0,
    }

    for by, row in enumerate(records):
        for bx, rec in enumerate(row):
            bid = by * nbx + bx
            source = str(rec.get("source", ""))
            if source not in use_sources:
                rejected["source"] += 1
                continue

            vr = float(rec.get("valid_ratio", 0.0) or 0.0)
            if vr < min_valid_ratio:
                rejected["valid_ratio"] += 1
                continue

            chosen_cost = rec.get("chosen_cost", None)
            chosen_cost_f = float(chosen_cost) if chosen_cost is not None and np.isfinite(chosen_cost) else np.inf
            if max_chosen_cost > 0.0 and (not np.isfinite(chosen_cost_f) or chosen_cost_f > max_chosen_cost):
                rejected["chosen_cost"] += 1
                continue

            reproj_mae = rec.get("reproj_mae", None)
            reproj_mae_f = float(reproj_mae) if reproj_mae is not None and np.isfinite(reproj_mae) else np.inf
            if max_reproj_mae > 0.0 and (not np.isfinite(reproj_mae_f) or reproj_mae_f > max_reproj_mae):
                rejected["reproj_mae"] += 1
                continue

            inliers = int(rec.get("inlier_count", 0) or 0)
            if min_inliers > 0 and source == "local_fit" and inliers < min_inliers:
                rejected["inliers"] += 1
                continue

            x0 = int(rec.get("block_x", rec.get("out_x", bx * block_size)))
            y0 = int(rec.get("block_y", rec.get("out_y", by * block_size)))
            bw = int(rec.get("block_w", rec.get("out_w", min(block_size, width - x0))))
            bh = int(rec.get("block_h", rec.get("out_h", min(block_size, height - y0))))
            x1 = min(width, x0 + bw)
            y1 = min(height, y0 + bh)
            if x1 <= x0 or y1 <= y0:
                rejected["no_valid_samples"] += 1
                continue

            # Include endpoints lightly by using the output block's actual area.
            ys = np.arange(y0, y1, sample_stride, dtype=np.float32)
            xs = np.arange(x0, x1, sample_stride, dtype=np.float32)
            if ys.size == 0 or xs.size == 0:
                rejected["no_valid_samples"] += 1
                continue
            gy, gx = np.meshgrid(ys, xs, indexing="ij")
            x_flat = gx.reshape(-1)
            y_flat = gy.reshape(-1)

            H = normalize_homography(np.asarray(rec["H"], dtype=np.float64).reshape(3, 3))
            qx, qy, h_ok = apply_homography_np(H, x_flat, y_flat)
            inside = h_ok & (qx >= 0.0) & (qx <= width - 1.0) & (qy >= 0.0) & (qy <= height - 1.0)

            xi = np.clip(np.rint(x_flat).astype(np.int64), 0, width - 1)
            yi = np.clip(np.rint(y_flat).astype(np.int64), 0, height - 1)
            d = depth0_np[yi, xi]
            d_ok = np.isfinite(d) & (d > min_depth)
            keep = inside & d_ok
            if not np.any(keep):
                rejected["no_valid_samples"] += 1
                continue

            x_flat = x_flat[keep]
            y_flat = y_flat[keep]
            qx = qx[keep]
            qy = qy[keep]

            cx = 0.5 * float(x0 + x1 - 1)
            cy = 0.5 * float(y0 + y1 - 1)
            denom = float(max(block_size, 1))
            xn = (x_flat - cx) / denom
            yn = (y_flat - cy) / denom

            # Reliability weight: valid_ratio and chosen_cost. Keep it mild.
            cw = 1.0
            if np.isfinite(chosen_cost_f) and chosen_cost_f > 0.0 and cost_weight_power > 0.0:
                cw = 1.0 / (chosen_cost_f ** cost_weight_power)
            src_w = 1.0
            if source == "parent_inherit":
                src_w = 0.5
            elif source == "root_fallback":
                src_w = 0.25
            weight = float(np.clip(vr, 0.05, 1.0) * cw * src_w)

            xs_all.append(x_flat.astype(np.float32))
            ys_all.append(y_flat.astype(np.float32))
            qx_all.append(qx.astype(np.float32))
            qy_all.append(qy.astype(np.float32))
            bid_all.append(np.full(x_flat.shape, bid, dtype=np.int64))
            xn_all.append(xn.astype(np.float32))
            yn_all.append(yn.astype(np.float32))
            w_all.append(np.full(x_flat.shape, weight, dtype=np.float32))
            block_used.append(bid)
            used_mask[bid] = True
            block_info.append(
                {
                    "block_id": int(bid),
                    "block_x": int(bx),
                    "block_y": int(by),
                    "x0": int(x0),
                    "y0": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                    "source": source,
                    "valid_ratio": float(vr),
                    "chosen_cost": float(chosen_cost_f) if np.isfinite(chosen_cost_f) else None,
                    "reproj_mae": float(reproj_mae_f) if np.isfinite(reproj_mae_f) else None,
                    "samples": int(x_flat.size),
                    "weight": float(weight),
                }
            )

    if not xs_all:
        raise RuntimeError("No homography-guided training samples. Loosen source/valid/cost thresholds.")

    samples = {
        "x": np.concatenate(xs_all).astype(np.float32),
        "y": np.concatenate(ys_all).astype(np.float32),
        "qx": np.concatenate(qx_all).astype(np.float32),
        "qy": np.concatenate(qy_all).astype(np.float32),
        "block_id": np.concatenate(bid_all).astype(np.int64),
        "xn": np.concatenate(xn_all).astype(np.float32),
        "yn": np.concatenate(yn_all).astype(np.float32),
        "weight": np.concatenate(w_all).astype(np.float32),
    }
    w = samples["weight"]
    finite_w = np.isfinite(w) & (w > 0)
    if np.any(finite_w):
        med = float(np.median(w[finite_w]))
        if med > 1e-12:
            samples["weight"] = np.clip(w / med, 1e-4, 100.0).astype(np.float32)

    summary = {
        "homography_block_size": int(block_size),
        "num_blocks_x": int(nbx),
        "num_blocks_y": int(nby),
        "num_blocks": int(num_blocks),
        "num_used_blocks": int(np.count_nonzero(used_mask)),
        "num_samples": int(samples["x"].size),
        "rejected_blocks": rejected,
        "used_blocks": block_info,
    }
    return samples, summary


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


def chroma_maps_from_luma(map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def y_mae_psnr(pred_y: np.ndarray, target_y: np.ndarray, valid: np.ndarray, bit_depth: int) -> tuple[Optional[float], Optional[float]]:
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
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict, Optional[float], Optional[float]]:
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
# Torch homography-guided inverse-depth plane fitting
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
    x_tar = rays_tar * depth.unsqueeze(0)
    x_world = R_tar_t @ (x_tar - t_tar.reshape(3, 1))
    x_ref = R_ref @ x_world + t_ref.reshape(3, 1)
    zr = x_ref[2]
    proj = K_ref @ x_ref
    denom = torch.where(torch.abs(proj[2]) > 1e-9, proj[2], torch.full_like(proj[2], 1e-9))
    xr = proj[0] / denom
    yr = proj[1] / denom
    return xr, yr, zr


def _smoothness_loss(params: torch.Tensor, used_mask: torch.Tensor, nby: int, nbx: int) -> torch.Tensor:
    p = params.reshape(nby, nbx, 3)
    m = used_mask.reshape(nby, nbx)
    loss = p.new_tensor(0.0)
    cnt = p.new_tensor(0.0)
    if nbx > 1:
        keep = (m[:, 1:] & m[:, :-1]).unsqueeze(-1)
        if keep.any():
            d = p[:, 1:, :] - p[:, :-1, :]
            loss = loss + (d[keep.expand_as(d)] ** 2).mean()
            cnt = cnt + 1.0
    if nby > 1:
        keep = (m[1:, :] & m[:-1, :]).unsqueeze(-1)
        if keep.any():
            d = p[1:, :, :] - p[:-1, :, :]
            loss = loss + (d[keep.expand_as(d)] ** 2).mean()
            cnt = cnt + 1.0
    return loss / torch.clamp(cnt, min=1.0)


def robust_xy_loss(err2: torch.Tensor, loss_type: str, f_scale: float) -> torch.Tensor:
    if loss_type == "l2":
        return err2
    err = torch.sqrt(torch.clamp(err2, min=1e-12))
    if loss_type == "l1":
        return err
    if loss_type == "huber":
        d = float(max(f_scale, 1e-6))
        return torch.where(err <= d, 0.5 * err * err, d * (err - 0.5 * d))
    if loss_type == "cauchy":
        f = float(max(f_scale, 1e-6))
        return (f * f) * torch.log1p(err2 / (f * f))
    # soft_l1
    f = float(max(f_scale, 1e-6))
    return 2.0 * (f * f) * (torch.sqrt(1.0 + err2 / (f * f)) - 1.0)


def fit_inverse_depth_planes_to_homography_torch(
    depth0_np: np.ndarray,
    K_ref_np: np.ndarray,
    R_ref_np: np.ndarray,
    t_ref_np: np.ndarray,
    K_tar_np: np.ndarray,
    R_tar_np: np.ndarray,
    t_tar_np: np.ndarray,
    homography_records: list[list[dict]],
    homography_block_size: int,
    samples_np: dict[str, np.ndarray],
    fit_mode: FitMode,
    iters: int,
    lr: float,
    max_rel_inv_correction: float,
    reg_lambda: float,
    smooth_lambda: float,
    min_depth: float,
    device_name: str,
    loss_type: str,
    robust_f_scale: float,
    print_every: int,
    grad_clip: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if fit_mode == "none":
        return depth0_np.astype(np.float32), np.zeros((0, 3), dtype=np.float32), {"fit_mode": "none"}
    if iters <= 0:
        raise ValueError("--fit-iters must be positive when fitting is enabled")
    if not (0.0 < max_rel_inv_correction < 0.99):
        raise ValueError("--max-rel-inv-correction should be in (0, 0.99)")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--fit-device cuda was requested, but CUDA is not available")

    h, w = depth0_np.shape
    nby = len(homography_records)
    nbx = len(homography_records[0])
    num_blocks = nby * nbx
    if homography_block_size <= 0:
        raise ValueError("Invalid homography block size")

    device = torch.device(device_name)
    dtype = torch.float32

    x_np = samples_np["x"]
    y_np = samples_np["y"]
    qx_np = samples_np["qx"]
    qy_np = samples_np["qy"]
    bid_np = samples_np["block_id"]
    xn_np = samples_np["xn"]
    yn_np = samples_np["yn"]
    weight_np = samples_np["weight"]
    n_samples = int(x_np.size)

    xi_np = np.clip(np.rint(x_np).astype(np.int64), 0, w - 1)
    yi_np = np.clip(np.rint(y_np).astype(np.int64), 0, h - 1)
    depth_s_np = depth0_np[yi_np, xi_np].astype(np.float32)
    depth_ok_np = np.isfinite(depth_s_np) & (depth_s_np > min_depth)
    inv0_np = np.zeros_like(depth_s_np, dtype=np.float32)
    inv0_np[depth_ok_np] = 1.0 / np.maximum(depth_s_np[depth_ok_np], min_depth)

    xs = torch.from_numpy(x_np).to(device=device, dtype=dtype)
    ys = torch.from_numpy(y_np).to(device=device, dtype=dtype)
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0)

    inv_K_tar = torch.linalg.inv(torch.as_tensor(K_tar_np, device=device, dtype=dtype))
    rays_tar = inv_K_tar @ pix

    K_ref = torch.as_tensor(K_ref_np, device=device, dtype=dtype)
    R_ref = torch.as_tensor(R_ref_np, device=device, dtype=dtype)
    t_ref = torch.as_tensor(t_ref_np, device=device, dtype=dtype)
    R_tar_t = torch.as_tensor(R_tar_np.T, device=device, dtype=dtype)
    t_tar = torch.as_tensor(t_tar_np, device=device, dtype=dtype)

    block_id = torch.from_numpy(bid_np).to(device=device, dtype=torch.long)
    xn = torch.from_numpy(xn_np).to(device=device, dtype=dtype)
    yn = torch.from_numpy(yn_np).to(device=device, dtype=dtype)
    inv0 = torch.from_numpy(inv0_np).to(device=device, dtype=dtype)
    depth_ok = torch.from_numpy(depth_ok_np).to(device=device, dtype=torch.bool)
    qx = torch.from_numpy(qx_np).to(device=device, dtype=dtype)
    qy = torch.from_numpy(qy_np).to(device=device, dtype=dtype)
    weight = torch.from_numpy(np.clip(weight_np, 1e-4, 100.0)).to(device=device, dtype=dtype)

    used_mask_np = np.zeros(num_blocks, dtype=bool)
    used_mask_np[np.unique(bid_np)] = True
    used_mask = torch.from_numpy(used_mask_np).to(device=device, dtype=torch.bool)

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

        valid = depth_ok & (zr > float(min_depth)) & torch.isfinite(xr) & torch.isfinite(yr)
        err2 = (xr - qx) ** 2 + (yr - qy) ** 2
        pix_loss = robust_xy_loss(err2, loss_type=loss_type, f_scale=robust_f_scale)

        if valid.any():
            wv = weight[valid]
            photo = torch.sum(wv * pix_loss[valid]) / (torch.sum(wv) + 1e-12)
            e = torch.sqrt(torch.clamp(err2[valid], min=1e-12))
            mae_px = torch.sum(wv * e) / (torch.sum(wv) + 1e-12)
            median_px = torch.median(e)
            valid_ratio = valid.float().mean()
        else:
            photo = pix_loss.mean() * 0.0 + 1.0
            mae_px = pix_loss.mean() * 0.0 + 1.0
            median_px = pix_loss.mean() * 0.0 + 1.0
            valid_ratio = valid.float().mean()

        if used_mask.any():
            reg_loss = params[used_mask].pow(2).mean()
        else:
            reg_loss = params.pow(2).mean()
        smooth_loss = _smoothness_loss(params, used_mask, nby, nbx)
        total = photo + float(reg_lambda) * reg_loss + float(smooth_lambda) * smooth_loss
        metrics = {
            "geom_loss": float(photo.detach().cpu()),
            "mae_px": float(mae_px.detach().cpu()),
            "median_px": float(median_px.detach().cpu()),
            "valid_samples": int(valid.detach().sum().cpu()),
            "valid_sample_ratio": float(valid_ratio.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "smooth_loss": float(smooth_loss.detach().cpu()),
        }
        return total, metrics

    with torch.no_grad():
        initial_loss, initial_metrics = compute_loss_and_metrics()
        initial_total = float(initial_loss.detach().cpu())
    if initial_metrics["valid_samples"] <= 0:
        raise RuntimeError("No valid samples before fitting. Check camera/depth/homography convention.")

    final_metrics = initial_metrics
    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, metrics = compute_loss_and_metrics()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([params], float(grad_clip))
        opt.step()
        final_metrics = metrics
        if print_every > 0 and (it == 1 or it % print_every == 0 or it == iters):
            print(
                f"fit iter {it:04d}/{iters}: "
                f"loss={float(loss.detach().cpu()):.6f}, "
                f"geom={metrics['geom_loss']:.6f}, "
                f"mae_px={metrics['mae_px']:.4f}, "
                f"median_px={metrics['median_px']:.4f}, "
                f"valid={metrics['valid_sample_ratio']:.4f}"
            )

    with torch.no_grad():
        final_loss, final_metrics = compute_loss_and_metrics()
        final_total = float(final_loss.detach().cpu())
        params_np = params.detach().cpu().numpy().astype(np.float32)

    corrected = apply_inverse_depth_params_fullres(
        depth0_np=depth0_np,
        params_np=params_np,
        width=w,
        height=h,
        block_size=homography_block_size,
        fit_mode=fit_mode,
        max_rel_inv_correction=max_rel_inv_correction,
        min_depth=min_depth,
        device_name=device_name,
        chunk_rows=256,
    )

    stats = {
        "fit_mode": fit_mode,
        "fit_target": "block_homography_ref_xy",
        "fit_param_type": "bounded_relative_inverse_depth_residual_plane",
        "formula": "inv_z_prime = inv_z * (1 + max_rel * tanh(a*xn + b*yn + c))",
        "homography_block_size": int(homography_block_size),
        "num_blocks_x": int(nbx),
        "num_blocks_y": int(nby),
        "num_blocks": int(num_blocks),
        "num_used_blocks": int(np.count_nonzero(used_mask_np)),
        "num_samples": int(n_samples),
        "iters": int(iters),
        "lr": float(lr),
        "max_rel_inv_correction": float(max_rel_inv_correction),
        "reg_lambda": float(reg_lambda),
        "smooth_lambda": float(smooth_lambda),
        "loss_type": loss_type,
        "robust_f_scale": float(robust_f_scale),
        "device": str(device),
        "initial_total_loss": initial_total,
        "final_total_loss": final_total,
        "initial_geom_loss": initial_metrics["geom_loss"],
        "final_geom_loss": final_metrics["geom_loss"],
        "initial_mae_px": initial_metrics["mae_px"],
        "final_mae_px": final_metrics["mae_px"],
        "initial_median_px": initial_metrics["median_px"],
        "final_median_px": final_metrics["median_px"],
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

        x0b = bx * block_size
        y0b = by * block_size
        x1b = torch.clamp(x0b + block_size, max=w)
        y1b = torch.clamp(y0b + block_size, max=h)
        cx = (x0b + x1b - 1).to(dtype) * 0.5
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


def write_fit_params_csv(
    path: str,
    params: np.ndarray,
    records: list[list[dict]],
    width: int,
    height: int,
    block_size: int,
) -> None:
    if params.size == 0:
        return
    nby = len(records)
    nbx = len(records[0])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_id", "block_x", "block_y", "x0", "y0", "w", "h",
            "source", "valid_ratio", "chosen_cost", "reproj_mae",
            "a", "b", "c",
        ])
        for by in range(nby):
            for bx in range(nbx):
                bid = by * nbx + bx
                rec = records[by][bx]
                x0 = int(rec.get("block_x", rec.get("out_x", bx * block_size)))
                y0 = int(rec.get("block_y", rec.get("out_y", by * block_size)))
                bw = int(rec.get("block_w", rec.get("out_w", min(block_size, width - x0))))
                bh = int(rec.get("block_h", rec.get("out_h", min(block_size, height - y0))))
                a, b, c = params[bid].tolist()
                writer.writerow([
                    bid, bx, by, x0, y0, bw, bh,
                    rec.get("source", ""),
                    rec.get("valid_ratio", None),
                    rec.get("chosen_cost", None),
                    rec.get("reproj_mae", None),
                    a, b, c,
                ])


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warp ref YUV to target using VGGT-Omega depth/camera with homography-guided block-wise depth fitting")
    p.add_argument("--yuv", required=True, help="Original source YUV sequence")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    p.add_argument("--ref-idx", type=int, required=True, help="absolute frame index of reference frame in original YUV")
    p.add_argument("--tar-idx", type=int, required=True, help="absolute frame index of target frame in original YUV")
    p.add_argument("--camera-jsonl", required=True, help="*_camera.jsonl from run_vggt_omega_yuv.py")
    p.add_argument("--homography-json", required=True, help="result.json from hierarchical_block_homography.py")
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

    # Homography sample selection.
    p.add_argument("--h-use-sources", default="local_fit",
                   help="Comma-separated homography sources to use. Default: local_fit. Example: local_fit,parent_inherit")
    p.add_argument("--h-min-valid-ratio", type=float, default=0.70)
    p.add_argument("--h-max-chosen-cost", type=float, default=-1.0,
                   help="Disable if <= 0. Otherwise reject blocks with chosen_cost above this.")
    p.add_argument("--h-max-reproj-mae", type=float, default=-1.0,
                   help="Disable if <= 0. Otherwise reject blocks with reproj_mae above this.")
    p.add_argument("--h-min-inliers", type=int, default=8)
    p.add_argument("--h-cost-weight-power", type=float, default=0.25,
                   help="Weight multiplier roughly 1/chosen_cost^power. Use 0 to disable.")

    # Depth fitting options.
    p.add_argument("--fit-mode", choices=["none", "bias", "plane"], default="plane",
                   help="none: no fit, bias: c only, plane: a*xn+b*yn+c. Default: plane")
    p.add_argument("--fit-device", default="cuda", help="cuda or cpu. Default: cuda")
    p.add_argument("--fit-sample-stride", type=int, default=4,
                   help="Use every N pixels inside each homography block during fitting.")
    p.add_argument("--fit-iters", type=int, default=300)
    p.add_argument("--fit-lr", type=float, default=0.03)
    p.add_argument("--fit-loss", choices=["soft_l1", "huber", "cauchy", "l1", "l2"], default="soft_l1")
    p.add_argument("--robust-f-scale", type=float, default=2.0,
                   help="Pixel unit robust loss scale for homography reprojection loss.")
    p.add_argument("--max-rel-inv-correction", type=float, default=0.30,
                   help="Bound for relative inverse-depth correction. 0.3 means inv_z can change by about +/-30%.")
    p.add_argument("--fit-reg-lambda", type=float, default=1e-4)
    p.add_argument("--fit-smooth-lambda", type=float, default=1e-3,
                   help="Neighbor block parameter smoothness regularization.")
    p.add_argument("--fit-print-every", type=int, default=25)
    p.add_argument("--fit-grad-clip", type=float, default=10.0)
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
    if args.fit_sample_stride <= 0:
        raise ValueError("--fit-sample-stride must be positive")

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

    h_block_size, h_records, h_data = load_homography_result(args.homography_json)
    if int(h_data.get("target_idx", args.tar_idx)) != args.tar_idx:
        print(f"[WARN] homography target_idx={h_data.get('target_idx')} but --tar-idx={args.tar_idx}")
    if int(h_data.get("ref_idx", args.ref_idx)) != args.ref_idx:
        print(f"[WARN] homography ref_idx={h_data.get('ref_idx')} but --ref-idx={args.ref_idx}")

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

    h_samples, h_sample_stats = build_homography_training_samples(
        records=h_records,
        width=args.width,
        height=args.height,
        block_size=h_block_size,
        depth0_np=depth0,
        sample_stride=args.fit_sample_stride,
        use_sources=parse_sources(args.h_use_sources),
        min_valid_ratio=args.h_min_valid_ratio,
        max_chosen_cost=args.h_max_chosen_cost,
        max_reproj_mae=args.h_max_reproj_mae,
        min_inliers=args.h_min_inliers,
        min_depth=args.min_depth,
        cost_weight_power=args.h_cost_weight_power,
    )
    print("[HOMOGRAPHY SAMPLES]")
    print(json.dumps({k: v for k, v in h_sample_stats.items() if k != "used_blocks"}, indent=2))

    fitted_depth, fit_params, fit_stats = fit_inverse_depth_planes_to_homography_torch(
        depth0_np=depth0,
        K_ref_np=K_ref,
        R_ref_np=R_ref,
        t_ref_np=t_ref,
        K_tar_np=K_tar,
        R_tar_np=R_tar,
        t_tar_np=t_tar,
        homography_records=h_records,
        homography_block_size=h_block_size,
        samples_np=h_samples,
        fit_mode=args.fit_mode,
        iters=args.fit_iters,
        lr=args.fit_lr,
        max_rel_inv_correction=args.max_rel_inv_correction,
        reg_lambda=args.fit_reg_lambda,
        smooth_lambda=args.fit_smooth_lambda,
        min_depth=args.min_depth,
        device_name=args.fit_device,
        loss_type=args.fit_loss,
        robust_f_scale=args.robust_f_scale,
        print_every=args.fit_print_every,
        grad_clip=args.fit_grad_clip,
    )

    out_fit_csv = args.output_prefix + "_fit_params.csv"
    if args.fit_mode != "none":
        write_fit_params_csv(out_fit_csv, fit_params, h_records, args.width, args.height, h_block_size)

    fitted_depth_meta = None
    out_fitted_depth = None
    if write_fitted_depth:
        out_fitted_depth = args.output_prefix + "_fitted_depth_inverse_yuv420p10le.yuv"
        fitted_depth_meta = write_inverse_depth_yuv420p10le(out_fitted_depth, fitted_depth)

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
            "homography_json": os.path.abspath(args.homography_json),
            "ref_idx": int(args.ref_idx),
            "tar_idx": int(args.tar_idx),
            "ref_camera_jsonl_position": int(ref_pos),
            "tar_camera_jsonl_position": int(tar_pos),
            "width": int(args.width),
            "height": int(args.height),
            "pix_fmt": pix_fmt,
            "homography_block_size": int(h_block_size),
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
            "homography_sample_stats": h_sample_stats,
            "fit_stats": fit_stats,
            "fitted_depth_output_meta": fitted_depth_meta,
            "output_warped_yuv": os.path.abspath(out_yuv),
            "output_target_yuv": os.path.abspath(out_target),
            "output_valid_mask_yuv420p": os.path.abspath(out_mask) if write_mask else None,
            "output_fitted_depth_yuv": os.path.abspath(out_fitted_depth) if out_fitted_depth else None,
            "output_fit_params_csv": os.path.abspath(out_fit_csv) if args.fit_mode != "none" else None,
            "note": "Camera fixed. Depth fitting uses block-wise bounded relative inverse-depth plane guided by block homography ref coordinates.",
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
    print(f"  H-fit initial MAE(px): {fit_stats.get('initial_mae_px')}")
    print(f"  H-fit final   MAE(px): {fit_stats.get('final_mae_px')}")


if __name__ == "__main__":
    main()
