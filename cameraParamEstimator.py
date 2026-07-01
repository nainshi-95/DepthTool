#!/usr/bin/env python3
# hierarchical_block_homography.py
#
# Hierarchical block-wise homography fitting experiment.
#
# Coordinate convention:
#   target pixel x_t -> ref pixel x_r
#
# For each level:
#   block_size = start_block_size / 2^level
#
# For each block:
#   1. collect feature matches whose target-side point lies in block + margin
#   2. fit local homography with RANSAC
#   3. compare candidate H against parent/fallback H by photometric MAE
#   4. accept candidate if reliable; otherwise inherit parent H
#
# Outputs:
#   target_pair.yuv
#   ref_pair.yuv
#   pred_levelXX_blockYYY.yuv
#   pred_hier_block_homography.yuv
#   result.json
#
# Example:
#   python hierarchical_block_homography.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --start-block-size 256 \
#     --levels 3 \
#     --block-margin 32 \
#     --max-features 60000 \
#     --match-ratio 0.70 \
#     --ransac-thresh 2.0 \
#     --min-matches 12 \
#     --min-inliers 8 \
#     --root-fallback global_translation \
#     --output-dir hier_homo_t1_r0

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Data
# ============================================================

@dataclass
class FrameY:
    y: np.ndarray
    width: int
    height: int


@dataclass
class MatchResult:
    pts_target: np.ndarray
    pts_ref: np.ndarray
    keypoints_target: list
    keypoints_ref: list
    good_matches: list


@dataclass
class BlockHResult:
    H: np.ndarray
    source: str
    ok: bool
    match_count: int
    inlier_count: int
    reproj_mae: float
    candidate_cost: float
    fallback_cost: float
    chosen_cost: float
    valid_ratio: float
    reason: str


# ============================================================
# Basic I/O
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def yuv420_frame_size_bytes(width: int, height: int, bitdepth: int) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    samples = width * height + 2 * ((width // 2) * (height // 2))

    if bitdepth == 8:
        return samples
    if bitdepth == 10:
        return samples * 2

    raise ValueError("Only bitdepth 8 and 10 are supported.")


def read_y_frame(path: str, width: int, height: int, bitdepth: int, frame_idx: int) -> FrameY:
    frame_size = yuv420_frame_size_bytes(width, height, bitdepth)
    y_samples = width * height
    offset = frame_idx * frame_size

    file_size = os.path.getsize(path)
    if offset + frame_size > file_size:
        raise ValueError(
            f"frame_idx={frame_idx} is out of range. "
            f"Need {offset + frame_size} bytes, file size={file_size}."
        )

    with open(path, "rb") as f:
        f.seek(offset)

        if bitdepth == 8:
            y = np.fromfile(f, dtype=np.uint8, count=y_samples).reshape(height, width)
        else:
            y = np.fromfile(f, dtype="<u2", count=y_samples).reshape(height, width)
            y = np.clip(y, 0, 1023).astype(np.uint16)

    return FrameY(y=y, width=width, height=height)


def write_single_yuv420(path: str, y: np.ndarray, width: int, height: int, bitdepth: int):
    y = np.asarray(y)[:height, :width]

    with open(path, "wb") as f:
        if bitdepth == 8:
            y_out = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        else:
            y_out = np.clip(np.rint(y), 0, 1023).astype("<u2")
            uv = np.full((height // 2, width // 2), 512, dtype="<u2")

        f.write(y_out.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())


def to_8bit(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return np.clip(y, 0, 255).astype(np.uint8)

    return np.clip(y.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)


def to_8bit_feature(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return y.astype(np.uint8)

    return (np.clip(y, 0, 1023).astype(np.uint16) >> 2).astype(np.uint8)


def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


# ============================================================
# Feature matching
# ============================================================

def detect_and_match_orb(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    max_features: int,
    ratio: float,
    use_clahe: bool,
) -> MatchResult:
    target_8 = to_8bit_feature(target_y, bitdepth)
    ref_8 = to_8bit_feature(ref_y, bitdepth)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_8 = clahe.apply(target_8)
        ref_8 = clahe.apply(ref_8)

    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=10,
    )

    kp_t, des_t = orb.detectAndCompute(target_8, None)
    kp_r, des_r = orb.detectAndCompute(ref_8, None)

    if des_t is None or des_r is None:
        raise RuntimeError("ORB failed to extract descriptors.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_t, des_r, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError(f"Not enough good matches: {len(good)}")

    pts_target = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_ref = np.float32([kp_r[m.trainIdx].pt for m in good])

    return MatchResult(
        pts_target=pts_target,
        pts_ref=pts_ref,
        keypoints_target=kp_t,
        keypoints_ref=kp_r,
        good_matches=good,
    )


def save_match_vis(
    out_path: str,
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    match_result: MatchResult,
    max_draw: int = 300,
):
    target_8 = to_8bit(target_y, bitdepth)
    ref_8 = to_8bit(ref_y, bitdepth)

    vis = cv2.drawMatches(
        target_8,
        match_result.keypoints_target,
        ref_8,
        match_result.keypoints_ref,
        match_result.good_matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(out_path, vis)


# ============================================================
# Homography utilities
# ============================================================

def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H = normalize_homography(H)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)

    q = ph @ H.T
    z = q[:, 2:3] + 1e-12

    out = q[:, :2] / z
    return out.astype(np.float32)


def homography_maps_for_roi(
    H: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = normalize_homography(H)

    xs, ys = np.meshgrid(
        np.arange(bx, bx + bw, dtype=np.float32),
        np.arange(by, by + bh, dtype=np.float32),
    )

    denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    valid_denom = np.abs(denom) > 1e-9
    denom_safe = denom + 1e-12

    map_x = (H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / denom_safe
    map_y = (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / denom_safe

    valid = (
        valid_denom
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def remap_ref(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def estimate_global_translation_H(pts_t: np.ndarray, pts_r: np.ndarray) -> np.ndarray:
    d = pts_r - pts_t
    dx, dy = np.median(d, axis=0)

    return np.array(
        [
            [1.0, 0.0, float(dx)],
            [0.0, 1.0, float(dy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_global_homography_H(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    ransac_thresh: float,
) -> Optional[np.ndarray]:
    if len(pts_t) < 8:
        return None

    try:
        H, mask = cv2.findHomography(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=5000,
            confidence=0.995,
        )
    except cv2.error:
        return None

    if H is None:
        return None

    return normalize_homography(H)


def fit_local_homography(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    ransac_thresh: float,
    min_matches: int,
    min_inliers: int,
) -> Tuple[Optional[np.ndarray], int, float, str]:
    if len(pts_t) < min_matches:
        return None, 0, float("inf"), "not_enough_matches"

    try:
        H, mask = cv2.findHomography(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=2000,
            confidence=0.99,
        )
    except cv2.error as e:
        return None, 0, float("inf"), f"cv2_error:{str(e)}"

    if H is None or mask is None:
        return None, 0, float("inf"), "findHomography_failed"

    H = normalize_homography(H)

    mask = mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(mask))

    if inlier_count < min_inliers:
        return None, inlier_count, float("inf"), "not_enough_inliers"

    pred = apply_homography_points(H, pts_t)
    err = np.sqrt(np.sum((pred - pts_r) ** 2, axis=1))

    if np.any(mask):
        reproj_mae = float(np.mean(err[mask]))
    else:
        reproj_mae = float(np.mean(err))

    return H, inlier_count, reproj_mae, "ok"


def eval_block_photometric_cost(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    H: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    min_valid_ratio: float,
) -> Tuple[float, float]:
    height, width = target_y.shape

    map_x, map_y, valid = homography_maps_for_roi(
        H=H,
        bx=bx,
        by=by,
        bw=bw,
        bh=bh,
        width=width,
        height=height,
    )

    valid_ratio = float(np.mean(valid))

    if valid_ratio < min_valid_ratio or not np.any(valid):
        return float("inf"), valid_ratio

    pred = remap_ref(ref_y, map_x, map_y)
    tgt = target_y[by:by + bh, bx:bx + bw].astype(np.float32)

    cost = float(np.mean(np.abs(tgt[valid] - pred[valid])))
    return cost, valid_ratio


# ============================================================
# Hierarchical fitting
# ============================================================

def collect_matches_for_block(
    pts_t_all: np.ndarray,
    pts_r_all: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    width: int,
    height: int,
    margin: int,
    parent_H: Optional[np.ndarray],
    parent_match_gate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = max(0, bx - margin)
    y0 = max(0, by - margin)
    x1 = min(width, bx + bw + margin)
    y1 = min(height, by + bh + margin)

    mask = (
        (pts_t_all[:, 0] >= x0)
        & (pts_t_all[:, 0] < x1)
        & (pts_t_all[:, 1] >= y0)
        & (pts_t_all[:, 1] < y1)
    )

    pts_t = pts_t_all[mask]
    pts_r = pts_r_all[mask]

    if parent_H is not None and parent_match_gate > 0.0 and len(pts_t) > 0:
        pred_parent = apply_homography_points(parent_H, pts_t)
        err = np.sqrt(np.sum((pred_parent - pts_r) ** 2, axis=1))
        keep = err <= parent_match_gate

        # If parent gate removes too much, do not apply it.
        if np.count_nonzero(keep) >= max(8, len(pts_t) // 4):
            pts_t = pts_t[keep]
            pts_r = pts_r[keep]

    return pts_t, pts_r


def parent_H_for_block(
    parent_H_grid: Optional[np.ndarray],
    parent_block_size: Optional[int],
    bx: int,
    by: int,
) -> Optional[np.ndarray]:
    if parent_H_grid is None or parent_block_size is None:
        return None

    py = by // parent_block_size
    px = bx // parent_block_size

    py = min(parent_H_grid.shape[0] - 1, max(0, py))
    px = min(parent_H_grid.shape[1] - 1, max(0, px))

    return parent_H_grid[py, px].reshape(3, 3).copy()


def render_prediction_from_H_grid(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    H_grid: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_y.shape
    pred = np.zeros((height, width), dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        bh = y1 - by

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            bw = x1 - bx

            H = H_grid[by_idx, bx_idx].reshape(3, 3)

            map_x, map_y, valid = homography_maps_for_roi(
                H=H,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                width=width,
                height=height,
            )

            pred_roi = remap_ref(ref_y, map_x, map_y)

            pred[by:y1, bx:x1] = pred_roi
            valid_all[by:y1, bx:x1] = valid

    return pred, valid_all


def run_one_level(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    pts_t_all: np.ndarray,
    pts_r_all: np.ndarray,
    block_size: int,
    margin: int,
    parent_H_grid: Optional[np.ndarray],
    parent_block_size: Optional[int],
    root_fallback_H: np.ndarray,
    args,
) -> Tuple[np.ndarray, List[List[Dict]], Dict[str, np.ndarray]]:
    height, width = target_y.shape
    ny, nx = block_grid_shape(width, height, block_size)

    H_grid = np.zeros((ny, nx, 3, 3), dtype=np.float64)

    source_grid = np.zeros((ny, nx), dtype=np.float64)
    match_count_grid = np.zeros((ny, nx), dtype=np.float64)
    inlier_count_grid = np.zeros((ny, nx), dtype=np.float64)
    reproj_mae_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    candidate_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    fallback_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    chosen_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)

    # source id:
    # 0 root_fallback
    # 1 parent_inherit
    # 2 local_fit
    # 3 local_fit_rejected_cost
    source_id = {
        "root_fallback": 0,
        "parent_inherit": 1,
        "local_fit": 2,
        "local_fit_rejected_cost": 3,
    }

    level_records: List[List[Dict]] = []

    accepted = 0
    inherited = 0
    rejected = 0

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        bh = y1 - by
        row_records = []

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            bw = x1 - bx

            parent_H = parent_H_for_block(parent_H_grid, parent_block_size, bx, by)

            if parent_H is None:
                fallback_H = root_fallback_H.copy()
                fallback_source = "root_fallback"
            else:
                fallback_H = parent_H.copy()
                fallback_source = "parent_inherit"

            fallback_cost, fallback_valid_ratio = eval_block_photometric_cost(
                target_y=target_y,
                ref_y=ref_y,
                H=fallback_H,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                min_valid_ratio=args.min_block_valid_ratio,
            )

            pts_t, pts_r = collect_matches_for_block(
                pts_t_all=pts_t_all,
                pts_r_all=pts_r_all,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                width=width,
                height=height,
                margin=margin,
                parent_H=fallback_H,
                parent_match_gate=args.parent_match_gate,
            )

            H_candidate, inlier_count, reproj_mae, fit_reason = fit_local_homography(
                pts_t=pts_t,
                pts_r=pts_r,
                ransac_thresh=args.ransac_thresh,
                min_matches=args.min_matches,
                min_inliers=args.min_inliers,
            )

            chosen_H = fallback_H
            chosen_source = fallback_source
            candidate_cost = float("inf")
            chosen_cost = fallback_cost
            chosen_valid_ratio = fallback_valid_ratio
            ok = False
            reason = fit_reason

            if H_candidate is not None:
                candidate_cost, candidate_valid_ratio = eval_block_photometric_cost(
                    target_y=target_y,
                    ref_y=ref_y,
                    H=H_candidate,
                    bx=bx,
                    by=by,
                    bw=bw,
                    bh=bh,
                    min_valid_ratio=args.min_block_valid_ratio,
                )

                accept_by_cost = True

                if not args.disable_cost_gate:
                    # Accept local H if it is better, or not much worse when fallback is invalid.
                    if np.isfinite(fallback_cost):
                        accept_by_cost = candidate_cost <= fallback_cost - args.min_gain
                    else:
                        accept_by_cost = np.isfinite(candidate_cost)

                if np.isfinite(candidate_cost) and accept_by_cost:
                    chosen_H = H_candidate
                    chosen_source = "local_fit"
                    chosen_cost = candidate_cost
                    chosen_valid_ratio = candidate_valid_ratio
                    ok = True
                    reason = "local_fit_accepted"
                    accepted += 1
                else:
                    chosen_H = fallback_H
                    chosen_source = "local_fit_rejected_cost"
                    chosen_cost = fallback_cost
                    chosen_valid_ratio = fallback_valid_ratio
                    ok = False
                    reason = "local_fit_rejected_by_cost"
                    rejected += 1
            else:
                inherited += 1

            H_grid[by_idx, bx_idx] = normalize_homography(chosen_H)
            source_grid[by_idx, bx_idx] = source_id.get(chosen_source, 0)
            match_count_grid[by_idx, bx_idx] = len(pts_t)
            inlier_count_grid[by_idx, bx_idx] = inlier_count
            reproj_mae_grid[by_idx, bx_idx] = reproj_mae
            candidate_cost_grid[by_idx, bx_idx] = candidate_cost
            fallback_cost_grid[by_idx, bx_idx] = fallback_cost
            chosen_cost_grid[by_idx, bx_idx] = chosen_cost
            valid_ratio_grid[by_idx, bx_idx] = chosen_valid_ratio

            row_records.append(
                {
                    "block_x": int(bx),
                    "block_y": int(by),
                    "block_w": int(bw),
                    "block_h": int(bh),
                    "source": chosen_source,
                    "reason": reason,
                    "match_count": int(len(pts_t)),
                    "inlier_count": int(inlier_count),
                    "reproj_mae": float(reproj_mae) if np.isfinite(reproj_mae) else None,
                    "candidate_cost": float(candidate_cost) if np.isfinite(candidate_cost) else None,
                    "fallback_cost": float(fallback_cost) if np.isfinite(fallback_cost) else None,
                    "chosen_cost": float(chosen_cost) if np.isfinite(chosen_cost) else None,
                    "valid_ratio": float(chosen_valid_ratio),
                    "H": H_grid[by_idx, bx_idx].tolist(),
                }
            )

        level_records.append(row_records)

    aux = {
        "source_grid": source_grid,
        "match_count_grid": match_count_grid,
        "inlier_count_grid": inlier_count_grid,
        "reproj_mae_grid": reproj_mae_grid,
        "candidate_cost_grid": candidate_cost_grid,
        "fallback_cost_grid": fallback_cost_grid,
        "chosen_cost_grid": chosen_cost_grid,
        "valid_ratio_grid": valid_ratio_grid,
        "accepted": np.array([accepted], dtype=np.float64),
        "inherited": np.array([inherited], dtype=np.float64),
        "rejected": np.array([rejected], dtype=np.float64),
    }

    return H_grid, level_records, aux


# ============================================================
# Cost / visualization
# ============================================================

def calc_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, bitdepth: int) -> dict:
    valid_ratio = float(np.mean(valid))

    if not np.any(valid):
        return {
            "valid_ratio": valid_ratio,
            "mae": float("inf"),
            "mse": float("inf"),
            "psnr": None,
        }

    diff = target_y.astype(np.float32)[valid] - pred_y.astype(np.float32)[valid]

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    maxv = 255.0 if bitdepth == 8 else 1023.0
    psnr = 999.0 if mse <= 1e-12 else float(10.0 * np.log10((maxv * maxv) / mse))

    return {
        "valid_ratio": valid_ratio,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }


def save_gray_png(path: str, y: np.ndarray, bitdepth: int):
    cv2.imwrite(path, to_8bit(y, bitdepth))


def expand_grid_to_image(grid: np.ndarray, width: int, height: int, block_size: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            out[by:y1, bx:x1] = float(grid[by_idx, bx_idx])

    return out


def save_scalar_map_png(path: str, grid: np.ndarray, width: int, height: int, block_size: int):
    img = expand_grid_to_image(grid, width, height, block_size)

    finite = np.isfinite(img)

    if not np.any(finite):
        out = np.zeros_like(img, dtype=np.uint8)
    else:
        vals = img[finite]
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))

        if abs(hi - lo) < 1e-12:
            out = np.full_like(img, 128, dtype=np.uint8)
        else:
            out = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, color)


def save_diff_png(path: str, target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray):
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)
    parser.add_argument("--target-idx", type=int, required=True)
    parser.add_argument("--ref-idx", type=int, required=True)

    parser.add_argument("--start-block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--min-block-size", type=int, default=32)
    parser.add_argument("--block-margin", type=int, default=32)

    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--no-clahe", action="store_true")

    parser.add_argument("--ransac-thresh", type=float, default=2.0)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--parent-match-gate", type=float, default=30.0)

    parser.add_argument(
        "--root-fallback",
        choices=["identity", "global_translation", "global_homography"],
        default="global_translation",
    )

    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--disable-cost-gate", action="store_true")

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    width = args.width
    height = args.height

    print(f"[INFO] target={args.target_idx}, ref={args.ref_idx}")
    print(f"[INFO] size={width}x{height}, bitdepth={args.bitdepth}")
    print(f"[INFO] start_block={args.start_block_size}, levels={args.levels}, min_block={args.min_block_size}")

    target = read_y_frame(args.input, width, height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, width, height, args.bitdepth, args.ref_idx)

    # ------------------------------------------------------------
    # Feature matching.
    # ------------------------------------------------------------

    match = detect_and_match_orb(
        target_y=target.y,
        ref_y=ref.y,
        bitdepth=args.bitdepth,
        max_features=args.max_features,
        ratio=args.match_ratio,
        use_clahe=not args.no_clahe,
    )

    print(f"[INFO] good matches = {len(match.good_matches)}")

    save_match_vis(
        os.path.join(args.output_dir, "match_vis.png"),
        target.y,
        ref.y,
        args.bitdepth,
        match,
    )

    # ------------------------------------------------------------
    # Root fallback H.
    # ------------------------------------------------------------

    H_identity = np.eye(3, dtype=np.float64)
    H_global_translation = estimate_global_translation_H(match.pts_target, match.pts_ref)
    H_global_homography = estimate_global_homography_H(
        match.pts_target,
        match.pts_ref,
        args.ransac_thresh,
    )

    if args.root_fallback == "identity":
        root_H = H_identity
    elif args.root_fallback == "global_translation":
        root_H = H_global_translation
    elif args.root_fallback == "global_homography":
        if H_global_homography is None:
            print("[WARN] global_homography fallback failed. Using global_translation.")
            root_H = H_global_translation
        else:
            root_H = H_global_homography
    else:
        raise ValueError(args.root_fallback)

    print("[INFO] root fallback H:")
    print(root_H)

    # ------------------------------------------------------------
    # Hierarchical levels.
    # ------------------------------------------------------------

    all_level_records = []
    all_level_costs = []
    all_level_aux_summary = []

    parent_H_grid = None
    parent_block_size = None

    final_pred = None
    final_valid = None
    final_H_grid = None
    final_block_size = None

    block_size = args.start_block_size

    for level in range(args.levels):
        if block_size < args.min_block_size:
            print(f"[INFO] stop: block_size={block_size} < min_block_size={args.min_block_size}")
            break

        print(f"[LEVEL {level}] block_size={block_size}")

        H_grid, level_records, aux = run_one_level(
            target_y=target.y,
            ref_y=ref.y,
            pts_t_all=match.pts_target,
            pts_r_all=match.pts_ref,
            block_size=block_size,
            margin=args.block_margin,
            parent_H_grid=parent_H_grid,
            parent_block_size=parent_block_size,
            root_fallback_H=root_H,
            args=args,
        )

        pred, valid = render_prediction_from_H_grid(
            target_y=target.y,
            ref_y=ref.y,
            H_grid=H_grid,
            block_size=block_size,
        )

        cost = calc_cost(target.y, pred, valid, args.bitdepth)

        level_tag = f"level{level:02d}_block{block_size}"

        yuv_path = os.path.join(args.output_dir, f"pred_{level_tag}.yuv")
        png_path = os.path.join(args.output_dir, f"pred_{level_tag}.png")
        diff_path = os.path.join(args.output_dir, f"diff_{level_tag}.png")

        write_single_yuv420(yuv_path, pred, width, height, args.bitdepth)
        save_gray_png(png_path, pred, args.bitdepth)
        save_diff_png(diff_path, target.y, pred, valid)

        save_scalar_map_png(
            os.path.join(args.output_dir, f"source_map_{level_tag}.png"),
            aux["source_grid"],
            width,
            height,
            block_size,
        )

        save_scalar_map_png(
            os.path.join(args.output_dir, f"match_count_{level_tag}.png"),
            aux["match_count_grid"],
            width,
            height,
            block_size,
        )

        save_scalar_map_png(
            os.path.join(args.output_dir, f"inlier_count_{level_tag}.png"),
            aux["inlier_count_grid"],
            width,
            height,
            block_size,
        )

        save_scalar_map_png(
            os.path.join(args.output_dir, f"chosen_cost_{level_tag}.png"),
            aux["chosen_cost_grid"],
            width,
            height,
            block_size,
        )

        save_scalar_map_png(
            os.path.join(args.output_dir, f"valid_ratio_{level_tag}.png"),
            aux["valid_ratio_grid"],
            width,
            height,
            block_size,
        )

        accepted = int(aux["accepted"][0])
        inherited = int(aux["inherited"][0])
        rejected = int(aux["rejected"][0])

        summary = {
            "level": int(level),
            "block_size": int(block_size),
            "num_blocks": int(H_grid.shape[0] * H_grid.shape[1]),
            "accepted_local_fit": accepted,
            "inherited_no_fit": inherited,
            "rejected_by_cost": rejected,
            "cost": cost,
            "output_yuv": yuv_path,
        }

        print("[LEVEL SUMMARY]")
        print(json.dumps(summary, indent=2))

        all_level_records.append(level_records)
        all_level_costs.append(summary)
        all_level_aux_summary.append(
            {
                "source_grid": aux["source_grid"].tolist(),
                "match_count_grid": aux["match_count_grid"].tolist(),
                "inlier_count_grid": aux["inlier_count_grid"].tolist(),
                "reproj_mae_grid": np.where(np.isfinite(aux["reproj_mae_grid"]), aux["reproj_mae_grid"], -1).tolist(),
                "candidate_cost_grid": np.where(np.isfinite(aux["candidate_cost_grid"]), aux["candidate_cost_grid"], -1).tolist(),
                "fallback_cost_grid": np.where(np.isfinite(aux["fallback_cost_grid"]), aux["fallback_cost_grid"], -1).tolist(),
                "chosen_cost_grid": np.where(np.isfinite(aux["chosen_cost_grid"]), aux["chosen_cost_grid"], -1).tolist(),
                "valid_ratio_grid": aux["valid_ratio_grid"].tolist(),
            }
        )

        final_pred = pred
        final_valid = valid
        final_H_grid = H_grid
        final_block_size = block_size

        parent_H_grid = H_grid
        parent_block_size = block_size
        block_size = block_size // 2

    if final_pred is None or final_valid is None:
        raise RuntimeError("No level was processed.")

    # ------------------------------------------------------------
    # Save final aliases.
    # ------------------------------------------------------------

    target_path = os.path.join(args.output_dir, "target_pair.yuv")
    ref_path = os.path.join(args.output_dir, "ref_pair.yuv")
    final_path = os.path.join(args.output_dir, "pred_hier_block_homography.yuv")

    write_single_yuv420(target_path, target.y, width, height, args.bitdepth)
    write_single_yuv420(ref_path, ref.y, width, height, args.bitdepth)
    write_single_yuv420(final_path, final_pred, width, height, args.bitdepth)

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_hier_block_homography.png"), final_pred, args.bitdepth)
    save_diff_png(os.path.join(args.output_dir, "diff_hier_block_homography.png"), target.y, final_pred, final_valid)

    final_cost = calc_cost(target.y, final_pred, final_valid, args.bitdepth)

    # ------------------------------------------------------------
    # JSON.
    # ------------------------------------------------------------

    result = {
        "input": args.input,
        "width": int(width),
        "height": int(height),
        "bitdepth": int(args.bitdepth),
        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "method": {
            "description": "hierarchical block-wise homography fitting",
            "coordinate": "target pixel -> ref pixel",
            "start_block_size": int(args.start_block_size),
            "levels": int(args.levels),
            "min_block_size": int(args.min_block_size),
            "block_margin": int(args.block_margin),
            "root_fallback": args.root_fallback,
            "parent_inheritance": True,
            "cost_gate": bool(not args.disable_cost_gate),
        },

        "feature_matching": {
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "clahe": bool(not args.no_clahe),
            "num_good_matches": int(len(match.good_matches)),
        },

        "fitting": {
            "ransac_thresh": float(args.ransac_thresh),
            "min_matches": int(args.min_matches),
            "min_inliers": int(args.min_inliers),
            "parent_match_gate": float(args.parent_match_gate),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
            "min_gain": float(args.min_gain),
        },

        "root_H": {
            "identity": H_identity.tolist(),
            "global_translation": H_global_translation.tolist(),
            "global_homography": H_global_homography.tolist() if H_global_homography is not None else None,
            "selected": root_H.tolist(),
        },

        "levels_summary": all_level_costs,
        "levels_aux_grids": all_level_aux_summary,
        "levels_blocks": all_level_records,

        "final": {
            "block_size": int(final_block_size),
            "cost": final_cost,
            "output_yuv": final_path,
        },

        "outputs": {
            "target_yuv": target_path,
            "ref_yuv": ref_path,
            "final_yuv": final_path,
        },

        "interpretation": [
            "source_map value 2 means local homography accepted.",
            "source_map value 1 means parent homography inherited.",
            "source_map value 3 means local homography was fitted but rejected by photometric cost.",
            "If many blocks are inherited, local feature fitting is not stable at that block size.",
            "If smaller levels improve MAE but look blocky, add smoothness/gain threshold or use larger block size.",
            "This code is diagnostic only; it does not yet compress block homographies into global parameters + scalar c.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[DONE]")
    print(f"  result JSON: {json_path}")
    print(f"  target YUV : {target_path}")
    print(f"  ref YUV    : {ref_path}")
    print(f"  final YUV  : {final_path}")
    print("[FINAL COST]")
    print(json.dumps(final_cost, indent=2))


if __name__ == "__main__":
    main()
