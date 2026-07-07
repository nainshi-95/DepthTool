#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-stage GOP camera refinement from structure-ECC pair residuals.

Purpose
-------
Use the camera/depth output from gop_camera_like_torch_lowres_rho.py as a
starting point, then run one more stage:

  1) For each target/ref pair, render the current camera-like projection using
     fixed rho/depth from the first stage.
  2) Estimate a pair-wise global residual transform with OpenCV ECC on Scharr
     structure images.  This is the preprocessing-only, encoder-side step.
  3) Convert the residual transform into pseudo-GT final correspondences:
        q_gt(x,y) = cam_map_base(x,y) + (H_or_A(x,y) - (x,y))
  4) Fit only GOP-level focal and frame-wise pose to q_gt while keeping depth
     fixed.  Translation is allowed, but strongly regularized.

This does NOT optimize depth/rho.  It is intended to test whether the stable
structure homography/affine residual can be absorbed into R + GOP focal, with
only tiny t corrections.

Pose convention is inherited from the first-stage script:
  R_i, t_i maps camera_i coordinates -> GOP-local world coordinates.
  X_w = R_t X_t + t_t
  X_r = R_r^T (X_w - t_r)

Coordinate convention:
  target pixel -> reference pixel
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("This script requires PyTorch.") from exc


# ============================================================
# Basic I/O
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def frame_size_yuv420(w: int, h: int, bitdepth: int) -> int:
    bps = 1 if bitdepth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def read_y_frame(path: str, w: int, h: int, bitdepth: int, idx: int) -> np.ndarray:
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    fs = frame_size_yuv420(w, h, bitdepth)
    y_samples = w * h
    with open(path, "rb") as f:
        f.seek(idx * fs)
        y = np.fromfile(f, dtype=dtype, count=y_samples)
    if y.size != y_samples:
        raise RuntimeError(f"Cannot read Y frame idx={idx} from {path}")
    return y.reshape(h, w)


def write_yuv420_y_only(path: str, y: np.ndarray, bitdepth: int):
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


def save_gray_png(path: str, y: np.ndarray, bitdepth: int):
    if bitdepth <= 8:
        out = np.clip(y, 0, 255).astype(np.uint8)
    else:
        out = np.clip(y.astype(np.float32) / float(1 << (bitdepth - 8)), 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def calc_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, bitdepth: int) -> Dict:
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


# ============================================================
# Result JSON / rho loading
# ============================================================

def load_stage1_result(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path_maybe(base_json: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    base_dir = os.path.dirname(os.path.abspath(base_json))
    cand = os.path.join(base_dir, p)
    if os.path.exists(cand):
        return cand
    return p


def load_poses_from_result(result: Dict) -> Tuple[List[int], Dict[int, int], np.ndarray, np.ndarray, int]:
    poses = result.get("poses", [])
    if not poses:
        raise RuntimeError("No poses[] found in first-stage result JSON")

    frames = [int(p["frame_idx"]) for p in poses]
    frames = sorted(frames)
    frame_to_cid = {f: i for i, f in enumerate(frames)}
    rvecs = np.zeros((len(frames), 3), dtype=np.float64)
    tvecs = np.zeros((len(frames), 3), dtype=np.float64)
    anchor_frame = frames[0]

    for p in poses:
        f = int(p["frame_idx"])
        cid = frame_to_cid[f]
        rvecs[cid] = np.asarray(p.get("rvec", [0, 0, 0]), dtype=np.float64).reshape(3)
        tvecs[cid] = np.asarray(p.get("t", [0, 0, 0]), dtype=np.float64).reshape(3)
        if bool(p.get("is_anchor", False)):
            anchor_frame = f

    return frames, frame_to_cid, rvecs, tvecs, anchor_frame


def get_used_pairs(result: Dict) -> List[Tuple[int, int]]:
    gop = result.get("gop", {})
    pairs = gop.get("used_pairs") or gop.get("requested_pairs")
    if not pairs:
        raise RuntimeError("No used_pairs/requested_pairs found in result JSON. Pass --pairs.")
    return [(int(a), int(b)) for a, b in pairs]


def parse_pairs(s: str) -> List[Tuple[int, int]]:
    out = []
    if not s.strip():
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "->" in tok:
            a, b = tok.split("->")
        elif ":" in tok:
            a, b = tok.split(":")
        else:
            raise ValueError(f"Invalid pair token: {tok}")
        out.append((int(a), int(b)))
    return out


def load_rho_map(rho_dir: str, target_idx: int) -> np.ndarray:
    path = os.path.join(rho_dir, f"rho_t{int(target_idx):03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"rho map not found: {path}")
    return np.load(path).astype(np.float32)


# ============================================================
# Geometry / camera projection
# ============================================================

def rodrigues_np(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def all_rotation_matrices_np(rvecs: np.ndarray) -> np.ndarray:
    Rs = np.zeros((rvecs.shape[0], 3, 3), dtype=np.float64)
    for i in range(rvecs.shape[0]):
        Rs[i] = rodrigues_np(rvecs[i])
    return Rs


def torch_rodrigues(rvecs: "torch.Tensor") -> "torch.Tensor":
    device = rvecs.device
    dtype = rvecs.dtype
    n = rvecs.shape[0]
    theta = torch.linalg.norm(rvecs, dim=1, keepdim=True).clamp_min(1e-12)
    k = rvecs / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = torch.zeros_like(kx)
    Kmat = torch.stack([
        torch.stack([z, -kz, ky], dim=1),
        torch.stack([kz, z, -kx], dim=1),
        torch.stack([-ky, kx, z], dim=1),
    ], dim=1)
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
    st = torch.sin(theta).view(n, 1, 1)
    ct = torch.cos(theta).view(n, 1, 1)
    R = I + st * Kmat + (1.0 - ct) * torch.bmm(Kmat, Kmat)
    small = (theta.view(-1) < 1e-7).view(n, 1, 1)
    return torch.where(small, I, R)


def camera_map_for_pair_np(
    target_idx: int,
    ref_idx: int,
    width: int,
    height: int,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    frame_to_cid: Dict[int, int],
    rho_img: np.ndarray,
    z_min: float,
    row_batch: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tcid = frame_to_cid[int(target_idx)]
    rcid = frame_to_cid[int(ref_idx)]
    Rs = all_rotation_matrices_np(rvecs)
    R_t, R_r = Rs[tcid], Rs[rcid]
    t_t, t_r = tvecs[tcid], tvecs[rcid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

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
        rays = np.stack([ray_x.reshape(-1), ray_y.reshape(-1), np.ones((y1 - y0) * width)], axis=1)
        rho = np.clip(rho_img[y0:y1, :].reshape(-1).astype(np.float64), 1e-12, np.inf)
        X_t = rays / rho[:, None]
        X_w = X_t @ R_t.T + t_t[None, :]
        X_r = (X_w - t_r[None, :]) @ R_r
        z = X_r[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
        mx = fx * (X_r[:, 0] / z_safe) + cx
        my = fy * (X_r[:, 1] / z_safe) + cy
        valid = (
            (z > float(z_min)) & np.isfinite(mx) & np.isfinite(my)
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


def make_structure_mask_u8(structure: np.ndarray, base_mask_u8: np.ndarray, keep_percent: float, dilate: int) -> Tuple[np.ndarray, Dict]:
    base = base_mask_u8 > 0
    vals = structure[base]
    stats = {
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


def estimate_pair_structure_ecc(
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
) -> Tuple[np.ndarray, bool, Optional[float], np.ndarray, Dict, np.ndarray]:
    template = make_structure_image(target_y, bitdepth, structure_mode, log_gain, pre_blur)
    inp = make_structure_image(cam_warp_y, bitdepth, structure_mode, log_gain, pre_blur)
    mask_u8, mask_stats = make_structure_mask_u8(template, valid_mask_u8, keep_percent, mask_dilate)
    stats = {
        "structure_mode": structure_mode,
        "keep_percent": float(keep_percent),
        "mask_dilate": int(mask_dilate),
        "mask": mask_stats,
    }
    if np.count_nonzero(mask_u8) < 100:
        return identity_transform(cp_num), False, None, mask_u8, stats, template

    motion_type = cv2.MOTION_HOMOGRAPHY if int(cp_num) == 4 else cv2.MOTION_AFFINE
    M = identity_transform(cp_num)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(ecc_iters), float(ecc_eps))
    try:
        cc, M = cv2.findTransformECC(
            templateImage=template.astype(np.float32),
            inputImage=inp.astype(np.float32),
            warpMatrix=M,
            motionType=motion_type,
            criteria=criteria,
            inputMask=mask_u8.astype(np.uint8),
            gaussFiltSize=int(ecc_gauss),
        )
        return M.astype(np.float32), True, float(cc), mask_u8, stats, template
    except cv2.error as exc:
        stats["cv2_error"] = str(exc)
        return identity_transform(cp_num), False, None, mask_u8, stats, template


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
# Observation generation from pair-wise ECC residuals
# ============================================================

def collect_pair_observations(
    pair: Tuple[int, int],
    seq_yuv: str,
    width: int,
    height: int,
    bitdepth: int,
    K_base: np.ndarray,
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    frame_to_cid: Dict[int, int],
    rho_dir: str,
    args,
    rng: np.random.Generator,
    pair_out_dir: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    target_idx, ref_idx = int(pair[0]), int(pair[1])
    target_y = read_y_frame(seq_yuv, width, height, bitdepth, target_idx)
    ref_y = read_y_frame(seq_yuv, width, height, bitdepth, ref_idx)
    rho_img = load_rho_map(rho_dir, target_idx)
    if rho_img.shape != (height, width):
        raise RuntimeError(f"rho_t{target_idx:03d}.npy has shape {rho_img.shape}, expected {(height,width)}")

    map_x, map_y, valid = camera_map_for_pair_np(
        target_idx, ref_idx, width, height, K_base,
        rvecs_base, tvecs_base, frame_to_cid, rho_img,
        z_min=args.z_min, row_batch=args.render_row_batch,
    )
    cam_warp = remap_y(ref_y, map_x, map_y)
    base_mask_u8 = make_valid_mask_u8(valid, erode=args.ecc_valid_erode)
    M, success, cc, ecc_mask_u8, ecc_stats, structure = estimate_pair_structure_ecc(
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
    )

    ys, xs = np.where(ecc_mask_u8 > 0)
    n0 = int(xs.size)
    if (not success) or n0 < args.min_obs_per_pair:
        info = {
            "target_idx": target_idx,
            "ref_idx": ref_idx,
            "success": bool(success),
            "ecc_cc": cc,
            "num_mask_pixels": n0,
            "num_observations": 0,
            "cp_bias": transform_cp_bias(M, width, height, args.ecc_cp_num).astype(float).tolist(),
            "ecc_stats": ecc_stats,
        }
        return {"target": np.empty(0, np.int32)}, info

    if args.max_obs_per_pair > 0 and xs.size > args.max_obs_per_pair:
        sel = rng.choice(xs.size, size=args.max_obs_per_pair, replace=False)
        xs = xs[sel]
        ys = ys[sel]

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dst = apply_transform_points(M, pts)
    bias = dst - pts

    qx = map_x[ys, xs].astype(np.float32) + bias[:, 0]
    qy = map_y[ys, xs].astype(np.float32) + bias[:, 1]
    rho = rho_img[ys, xs].astype(np.float32)

    ok = (
        np.isfinite(qx) & np.isfinite(qy) & np.isfinite(rho)
        & (rho > 0.0)
        & (map_x[ys, xs] >= 0.0) & (map_y[ys, xs] >= 0.0)
        & (qx >= 0.0) & (qx <= width - 1.0)
        & (qy >= 0.0) & (qy <= height - 1.0)
    )

    xs = xs[ok]
    ys = ys[ok]
    qx = qx[ok]
    qy = qy[ok]
    rho = rho[ok]
    w = (0.25 + structure[ys, xs].astype(np.float32))
    if cc is not None and np.isfinite(cc):
        w = w * float(max(0.05, min(2.0, cc + 1.0)))

    obs = {
        "target": np.full(xs.shape[0], target_idx, dtype=np.int32),
        "ref": np.full(xs.shape[0], ref_idx, dtype=np.int32),
        "px": xs.astype(np.float32),
        "py": ys.astype(np.float32),
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "rho": rho.astype(np.float32),
        "weight": w.astype(np.float32),
    }

    cost_cam = calc_cost(target_y, cam_warp, valid, bitdepth)

    if pair_out_dir is not None:
        ensure_dir(pair_out_dir)
        write_yuv420_y_only(os.path.join(pair_out_dir, f"cam_base_t{target_idx:03d}_r{ref_idx:03d}.yuv"), cam_warp, bitdepth)
        save_gray_png(os.path.join(pair_out_dir, f"cam_base_t{target_idx:03d}_r{ref_idx:03d}.png"), cam_warp, bitdepth)
        mask_vis = np.where(ecc_mask_u8 > 0, (1 << bitdepth) - 1, 0).astype(np.float32)
        save_gray_png(os.path.join(pair_out_dir, f"ecc_mask_t{target_idx:03d}_r{ref_idx:03d}.png"), mask_vis, bitdepth)

    info = {
        "target_idx": target_idx,
        "ref_idx": ref_idx,
        "success": bool(success),
        "ecc_cc": None if cc is None else float(cc),
        "motion_type": "homography" if int(args.ecc_cp_num) == 4 else "affine",
        "num_mask_pixels": n0,
        "num_observations": int(xs.shape[0]),
        "cp_bias": transform_cp_bias(M, width, height, args.ecc_cp_num).astype(float).tolist(),
        "matrix": np.asarray(M, dtype=float).tolist(),
        "base_cam_cost": cost_cam,
        "ecc_stats": ecc_stats,
    }
    return obs, info


def concat_observations(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ["target", "ref", "px", "py", "qx", "qy", "rho", "weight"]
    out = {}
    for k in keys:
        vals = [o[k] for o in obs_list if k in o and o[k].size > 0]
        if not vals:
            raise RuntimeError(f"No observations for key {k}")
        out[k] = np.concatenate(vals, axis=0)
    return out


# ============================================================
# Second-stage fitting: R + focal + tiny t, fixed rho
# ============================================================

def choose_batch_indices(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    if batch <= 0 or batch >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=batch, replace=False).astype(np.int64)


def robust_loss_from_err2(err2: "torch.Tensor", loss_name: str, f_scale: float) -> "torch.Tensor":
    f = float(max(f_scale, 1e-6))
    if loss_name == "linear":
        return err2
    if loss_name == "huber":
        err = torch.sqrt(err2.clamp_min(1e-12))
        return torch.where(err <= f, 0.5 * err * err, f * (err - 0.5 * f))
    if loss_name == "cauchy":
        return (f * f) * torch.log1p(err2 / (f * f))
    return 2.0 * (f * f) * (torch.sqrt(1.0 + err2 / (f * f)) - 1.0)


def fit_rf_tiny_t(
    observations: Dict[str, np.ndarray],
    frames: List[int],
    frame_to_cid: Dict[int, int],
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    K_base: np.ndarray,
    anchor_frame: int,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.torch_float64 else torch.float32
    rng = np.random.default_rng(int(args.seed))

    n = int(observations["px"].shape[0])
    target_cid_np = np.asarray([frame_to_cid[int(f)] for f in observations["target"]], dtype=np.int64)
    ref_cid_np = np.asarray([frame_to_cid[int(f)] for f in observations["ref"]], dtype=np.int64)

    target_cid = torch.tensor(target_cid_np, device=device, dtype=torch.long)
    ref_cid = torch.tensor(ref_cid_np, device=device, dtype=torch.long)
    px = torch.tensor(observations["px"], device=device, dtype=dtype)
    py = torch.tensor(observations["py"], device=device, dtype=dtype)
    qx = torch.tensor(observations["qx"], device=device, dtype=dtype)
    qy = torch.tensor(observations["qy"], device=device, dtype=dtype)
    rho = torch.tensor(observations["rho"], device=device, dtype=dtype).clamp_min(1e-12)

    w_np = observations["weight"].astype(np.float64)
    good = np.isfinite(w_np) & (w_np > 0)
    med = float(np.median(w_np[good])) if np.any(good) else 1.0
    w_np = np.clip(w_np / max(med, 1e-12), 1e-4, 100.0).astype(np.float32)
    weight = torch.tensor(w_np, device=device, dtype=dtype)

    r_base = torch.tensor(rvecs_base, device=device, dtype=dtype)
    t_base = torch.tensor(tvecs_base, device=device, dtype=dtype)
    r_delta = torch.nn.Parameter(torch.zeros_like(r_base))
    t_delta = torch.nn.Parameter(torch.zeros_like(t_base))

    anchor_cid = frame_to_cid[int(anchor_frame)]
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
        params = [{"params": [r_delta], "lr": float(args.lr_rot)}]
        if focal_mode != "fixed":
            params.append({"params": [log_f_delta], "lr": float(args.lr_focal)})
        t_delta.requires_grad_(False)

    if args.freeze_r:
        r_delta.requires_grad_(False)

    opt = torch.optim.Adam(params)
    cx = torch.tensor(float(K_base[0, 2]), device=device, dtype=dtype)
    cy = torch.tensor(float(K_base[1, 2]), device=device, dtype=dtype)

    def current_params():
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor_cid] = 0.0
        td[anchor_cid] = 0.0
        r = r_base + rd
        t = t_base + td
        if focal_mode == "single":
            lfd = torch.clamp(log_f_delta[0], -float(args.f_log_max_delta), float(args.f_log_max_delta))
            fx = torch.tensor(float(f0), device=device, dtype=dtype) * torch.exp(lfd)
            fy = fx
        elif focal_mode == "separate":
            lfdx = torch.clamp(log_f_delta[0], -float(args.f_log_max_delta), float(args.f_log_max_delta))
            lfdy = torch.clamp(log_f_delta[1], -float(args.f_log_max_delta), float(args.f_log_max_delta))
            fx = torch.tensor(float(f_base_x), device=device, dtype=dtype) * torch.exp(lfdx)
            fy = torch.tensor(float(f_base_y), device=device, dtype=dtype) * torch.exp(lfdy)
        else:
            fx = torch.tensor(float(f_base_x), device=device, dtype=dtype)
            fy = torch.tensor(float(f_base_y), device=device, dtype=dtype)
        return r, t, fx, fy

    def project_indices(idx_np: np.ndarray):
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        r, t, fx, fy = current_params()
        R = torch_rodrigues(r)
        tc = target_cid[idx]
        rc = ref_cid[idx]
        ray_x = (px[idx] - cx) / fx
        ray_y = (py[idx] - cy) / fy
        rays = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=1)
        X_t = rays / rho[idx, None]
        X_w = torch.bmm(R[tc], X_t.unsqueeze(-1)).squeeze(-1) + t[tc]
        X_rel = X_w - t[rc]
        X_r = torch.bmm(R[rc].transpose(1, 2), X_rel.unsqueeze(-1)).squeeze(-1)
        z = X_r[:, 2]
        z_safe = torch.where(torch.abs(z) > 1e-9, z, torch.full_like(z, 1e-9))
        u = fx * (X_r[:, 0] / z_safe) + cx
        v = fy * (X_r[:, 1] / z_safe) + cy
        return u, v, z

    def regularization():
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor_cid] = 0.0
        td[anchor_cid] = 0.0
        loss = torch.zeros((), device=device, dtype=dtype)
        if args.rot_delta_prior_weight > 0:
            loss = loss + float(args.rot_delta_prior_weight) * torch.mean(rd * rd)
        if args.trans_delta_prior_weight > 0 and not args.freeze_t:
            loss = loss + float(args.trans_delta_prior_weight) * torch.mean(td * td)
        if args.pose_delta_smooth_weight > 0:
            # Smooth deltas over sorted frame order.
            if rd.shape[0] > 1:
                loss = loss + float(args.pose_delta_smooth_weight) * torch.mean((rd[1:] - rd[:-1]) ** 2)
                if not args.freeze_t:
                    loss = loss + float(args.pose_delta_smooth_weight) * torch.mean((td[1:] - td[:-1]) ** 2)
        if focal_mode != "fixed" and args.f_prior_weight > 0:
            loss = loss + float(args.f_prior_weight) * torch.mean(log_f_delta * log_f_delta)
        return loss

    def loss_for_batch(idx_np: np.ndarray):
        u, v, z = project_indices(idx_np)
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        dx = u - qx[idx]
        dy = v - qy[idx]
        err2 = dx * dx + dy * dy
        pix = robust_loss_from_err2(err2, args.robust_loss, args.robust_f_scale)
        if args.z_min > 0:
            zbad = torch.relu(float(args.z_min) - z)
            pix = pix + float(args.z_penalty) * zbad * zbad
        ww = weight[idx]
        return torch.sum(ww * pix) / (torch.sum(ww) + 1e-12) + regularization()

    @torch.no_grad()
    def eval_all(batch: int = 262144) -> np.ndarray:
        out = np.full(n, np.inf, dtype=np.float64)
        for s in range(0, n, batch):
            e = min(n, s + batch)
            idx_np = np.arange(s, e, dtype=np.int64)
            u, v, z = project_indices(idx_np)
            idx = torch.tensor(idx_np, device=device, dtype=torch.long)
            err = torch.sqrt((u - qx[idx]) ** 2 + (v - qy[idx]) ** 2).detach().cpu().numpy()
            zz = z.detach().cpu().numpy()
            err[zz <= args.z_min] = np.inf
            out[s:e] = err
        return out

    report = {
        "device": str(device),
        "dtype": str(dtype),
        "num_observations": int(n),
        "anchor_frame": int(anchor_frame),
        "focal_mode": focal_mode,
        "f_init": args.f_init,
        "f0": float(f0),
        "iterations": [],
    }

    for step in range(int(args.steps)):
        idx = choose_batch_indices(n, int(args.batch_size), rng)
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(idx)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"] if p.requires_grad], float(args.grad_clip))
        opt.step()
        with torch.no_grad():
            r_delta[anchor_cid].zero_()
            t_delta[anchor_cid].zero_()
            if args.max_trans_delta > 0 and not args.freeze_t:
                t_delta.clamp_(-float(args.max_trans_delta), float(args.max_trans_delta))
            if log_f_delta is not None:
                log_f_delta.clamp_(-float(args.f_log_max_delta), float(args.f_log_max_delta))

        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            err = eval_all(batch=int(args.eval_batch_size))
            finite = np.isfinite(err)
            if np.any(finite):
                stat = {
                    "count": int(np.count_nonzero(finite)),
                    "mean": float(np.mean(err[finite])),
                    "median": float(np.median(err[finite])),
                    "p90": float(np.percentile(err[finite], 90)),
                    "p95": float(np.percentile(err[finite], 95)),
                }
            else:
                stat = {"count": 0}
            r_cur, t_cur, fx_cur, fy_cur = current_params()
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
        return (
            K_final,
            r_final.detach().cpu().numpy().astype(np.float64),
            t_final.detach().cpu().numpy().astype(np.float64),
            report,
        )


# ============================================================
# Rendering final refined pairs
# ============================================================

def render_refined_pairs(
    pairs: List[Tuple[int, int]],
    seq_yuv: str,
    width: int,
    height: int,
    bitdepth: int,
    K_base: np.ndarray,
    r_base: np.ndarray,
    t_base: np.ndarray,
    K_final: np.ndarray,
    r_final: np.ndarray,
    t_final: np.ndarray,
    frame_to_cid: Dict[int, int],
    rho_dir: str,
    out_dir: str,
    args,
) -> List[Dict]:
    ensure_dir(out_dir)
    costs = []
    for target_idx, ref_idx in pairs:
        target_y = read_y_frame(seq_yuv, width, height, bitdepth, target_idx)
        ref_y = read_y_frame(seq_yuv, width, height, bitdepth, ref_idx)
        rho_img = load_rho_map(rho_dir, target_idx)
        bx, by, bvalid = camera_map_for_pair_np(target_idx, ref_idx, width, height, K_base, r_base, t_base, frame_to_cid, rho_img, args.z_min, args.render_row_batch)
        fx, fy, fvalid = camera_map_for_pair_np(target_idx, ref_idx, width, height, K_final, r_final, t_final, frame_to_cid, rho_img, args.z_min, args.render_row_batch)
        pred_base = remap_y(ref_y, bx, by)
        pred_final = remap_y(ref_y, fx, fy)
        cost_base = calc_cost(target_y, pred_base, bvalid, bitdepth)
        cost_final = calc_cost(target_y, pred_final, fvalid, bitdepth)
        tag = f"t{target_idx:03d}_r{ref_idx:03d}"
        if not args.no_render_yuv:
            write_yuv420_y_only(os.path.join(out_dir, f"pred_refined_{tag}.yuv"), pred_final, bitdepth)
            save_gray_png(os.path.join(out_dir, f"pred_refined_{tag}.png"), pred_final, bitdepth)
        costs.append({
            "target_idx": int(target_idx),
            "ref_idx": int(ref_idx),
            "base_cost": cost_base,
            "refined_cost": cost_final,
            "psnr_gain_vs_base": None if (cost_base["psnr"] is None or cost_final["psnr"] is None) else float(cost_final["psnr"] - cost_base["psnr"]),
        })
        print("[PAIR RENDER COST]")
        print(json.dumps(costs[-1], indent=2))
    return costs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Source YUV420 sequence")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)
    ap.add_argument("--stage1-result-json", required=True, help="gop_camera_like_result.json from first stage")
    ap.add_argument("--rho-map-dir", default="", help="Override rho_maps dir. Default: read from stage1 JSON outputs.rho_map_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pairs", default="", help="Optional pair list like 8:0,8:16. Default: used_pairs from stage1 JSON")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    # ECC pair residual extraction.
    ap.add_argument("--ecc-cp-num", type=int, choices=[3, 4], default=4, help="4=homography, 3=affine")
    ap.add_argument("--structure-mode", choices=["scharr_mag", "scharr_l1", "scharr_x", "scharr_y", "scharr_x_weighted"], default="scharr_mag")
    ap.add_argument("--structure-keep-percent", type=float, default=35.0)
    ap.add_argument("--structure-mask-dilate", type=int, default=1)
    ap.add_argument("--structure-log-gain", type=float, default=20.0)
    ap.add_argument("--structure-pre-blur", type=int, default=0)
    ap.add_argument("--ecc-valid-erode", type=int, default=2)
    ap.add_argument("--ecc-iters", type=int, default=80)
    ap.add_argument("--ecc-eps", type=float, default=1e-5)
    ap.add_argument("--ecc-gauss", type=int, default=5)
    ap.add_argument("--max-obs-per-pair", type=int, default=25000)
    ap.add_argument("--min-obs-per-pair", type=int, default=500)

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
    ap.add_argument("--f-log-max-delta", type=float, default=0.05, help="Clamp log focal delta, 0.05 ~= +-5%")
    ap.add_argument("--f-prior-weight", type=float, default=10.0)
    ap.add_argument("--rot-delta-prior-weight", type=float, default=1e-3)
    ap.add_argument("--trans-delta-prior-weight", type=float, default=100.0, help="Large value keeps t correction tiny")
    ap.add_argument("--pose-delta-smooth-weight", type=float, default=1e-3)
    ap.add_argument("--max-trans-delta", type=float, default=0.0, help="Optional component-wise clamp on t delta. <=0 disables")
    ap.add_argument("--freeze-t", action="store_true")
    ap.add_argument("--freeze-r", action="store_true")
    ap.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
    ap.add_argument("--robust-f-scale", type=float, default=2.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--z-min", type=float, default=1e-4)
    ap.add_argument("--z-penalty", type=float, default=100.0)
    ap.add_argument("--render-row-batch", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--no-render-yuv", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.output_dir)
    rng = np.random.default_rng(int(args.seed))

    result = load_stage1_result(args.stage1_result_json)
    K_base = np.asarray(result["K"], dtype=np.float64).reshape(3, 3)
    frames, frame_to_cid, r_base, t_base, anchor_frame = load_poses_from_result(result)

    if args.rho_map_dir:
        rho_dir = args.rho_map_dir
    else:
        rho_dir = result.get("outputs", {}).get("rho_map_dir", "")
        rho_dir = resolve_path_maybe(args.stage1_result_json, rho_dir)
    if not rho_dir or not os.path.isdir(rho_dir):
        raise RuntimeError(f"Invalid rho map dir: {rho_dir}")

    pairs = parse_pairs(args.pairs) if args.pairs.strip() else get_used_pairs(result)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    print("[INFO] Stage-1 K:")
    print(K_base)
    print(f"[INFO] anchor_frame={anchor_frame}")
    print(f"[INFO] rho_dir={rho_dir}")
    print("[INFO] pairs:")
    for p in pairs:
        print(f"  target={p[0]} ref={p[1]}")

    pair_info = []
    obs_list = []
    pair_extract_dir = os.path.join(args.output_dir, "pair_ecc")
    ensure_dir(pair_extract_dir)
    for target_idx, ref_idx in pairs:
        print(f"[PAIR ECC] target={target_idx}, ref={ref_idx}")
        obs, info = collect_pair_observations(
            pair=(target_idx, ref_idx),
            seq_yuv=args.input,
            width=args.width,
            height=args.height,
            bitdepth=args.bitdepth,
            K_base=K_base,
            rvecs_base=r_base,
            tvecs_base=t_base,
            frame_to_cid=frame_to_cid,
            rho_dir=rho_dir,
            args=args,
            rng=rng,
            pair_out_dir=pair_extract_dir,
        )
        pair_info.append(info)
        if obs.get("target", np.empty(0)).size > 0:
            obs_list.append(obs)
        print(json.dumps({k: info[k] for k in ["target_idx", "ref_idx", "success", "ecc_cc", "num_observations", "cp_bias"]}, indent=2))

    if not obs_list:
        raise RuntimeError("No valid pair ECC observations were generated.")
    observations = concat_observations(obs_list)
    print(f"[INFO] total observations = {observations['px'].shape[0]}")

    K_final, r_final, t_final, fit_report = fit_rf_tiny_t(
        observations=observations,
        frames=frames,
        frame_to_cid=frame_to_cid,
        rvecs_base=r_base,
        tvecs_base=t_base,
        K_base=K_base,
        anchor_frame=anchor_frame,
        args=args,
    )

    render_costs = []
    if not args.skip_render:
        render_costs = render_refined_pairs(
            pairs=pairs,
            seq_yuv=args.input,
            width=args.width,
            height=args.height,
            bitdepth=args.bitdepth,
            K_base=K_base,
            r_base=r_base,
            t_base=t_base,
            K_final=K_final,
            r_final=r_final,
            t_final=t_final,
            frame_to_cid=frame_to_cid,
            rho_dir=rho_dir,
            out_dir=os.path.join(args.output_dir, "refined_pairs"),
            args=args,
        )

    pose_json = []
    for f in frames:
        cid = frame_to_cid[f]
        pose_json.append({
            "frame_idx": int(f),
            "compact_id": int(cid),
            "is_anchor": bool(f == anchor_frame),
            "rvec_base": r_base[cid].astype(float).tolist(),
            "t_base": t_base[cid].astype(float).tolist(),
            "rvec_refined": r_final[cid].astype(float).tolist(),
            "t_refined": t_final[cid].astype(float).tolist(),
            "rvec_delta": (r_final[cid] - r_base[cid]).astype(float).tolist(),
            "t_delta": (t_final[cid] - t_base[cid]).astype(float).tolist(),
            "R_refined": rodrigues_np(r_final[cid]).astype(float).tolist(),
        })

    out = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "stage1_result_json": args.stage1_result_json,
        "rho_map_dir": rho_dir,
        "method": {
            "description": "Second-stage fixed-depth fitting from pair-wise structure-ECC residual transforms. Fits GOP focal and frame-wise R|t; t is strongly regularized.",
            "depth": "fixed rho maps from first stage",
            "supervision": "q_gt = base_camera_map + ECC_transform_bias(x,y)",
            "pose_convention": "R_i,t_i maps camera_i coordinates -> GOP-local world coordinates",
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
        },
        "pairs": pair_info,
        "fit_report": fit_report,
        "poses": pose_json,
        "render_costs": render_costs,
        "outputs": {
            "pair_ecc_dir": pair_extract_dir,
            "refined_pair_dir": os.path.join(args.output_dir, "refined_pairs"),
            "result_json": os.path.join(args.output_dir, "gop_camera_refine_rf_result.json"),
        },
    }

    out_path = os.path.join(args.output_dir, "gop_camera_refine_rf_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print("[DONE]")
    print(f"  result JSON: {out_path}")
    print("  K_refined:")
    print(K_final)


if __name__ == "__main__":
    main()
