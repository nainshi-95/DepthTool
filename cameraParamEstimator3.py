#!/usr/bin/env python3
# gop_camera_like_from_block_homography.py
#
# GOP-level camera-like fitting from hierarchical block homography pseudo-GT.
#
# Coordinate convention:
#   target pixel x_t -> ref pixel x_r
#
# Pipeline:
#   1. For selected GOP reference pairs, run hierarchical block-wise homography.
#   2. Convert final block H into control-point correspondences:
#        p_target -> q_ref = H_b p_target
#   3. Use those CP correspondences as weighted pseudo-GT observations.
#   4. Fit GOP-shared camera-like variables:
#        fixed K
#        frame-wise pose R_i, t_i
#        target-frame/block-wise constant inverse depth rho_{i,b}
#      using robust alternating optimization:
#        depth update -> pose least-squares -> residual trimming
#   5. Warp ref frames using fitted camera-like model and save YUV.
#
# Note:
#   This is not intended to recover true physical 3D.
#   It is a GOP-local, view-consistent low-dimensional motion model.
#
# Example:
#   python gop_camera_like_from_block_homography.py \
#     --input input.yuv \
#     --width 1920 --height 1080 --bitdepth 10 \
#     --gop-start 0 --gop-size 33 \
#     --pair-mode dyadic \
#     --output-dir gop0_camlike
#
# Default block hierarchy:
#   512 -> 256 -> 128 -> 64

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as e:
    raise ImportError("This script requires scipy. Install with: pip install scipy") from e


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
class Tile:
    out_x0: int
    out_y0: int
    out_x1: int
    out_y1: int

    fit_x0: int
    fit_y0: int
    fit_x1: int
    fit_y1: int

    out_w: int
    out_h: int
    fit_w: int
    fit_h: int


@dataclass
class PairHResult:
    target_idx: int
    ref_idx: int
    final_block_size: int
    final_H_grid: np.ndarray
    final_records: List[List[Dict]]
    direct_cost: Dict
    direct_yuv_path: str


@dataclass
class ObservationSet:
    target_frame: np.ndarray
    ref_frame: np.ndarray
    depth_index: np.ndarray
    pair_index: np.ndarray
    px: np.ndarray
    py: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    ray_x: np.ndarray
    ray_y: np.ndarray
    weight: np.ndarray
    block_source: np.ndarray
    block_cost: np.ndarray
    meta: List[Dict]


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


# ============================================================
# Block / tile
# ============================================================

def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def make_tiles(
    width: int,
    height: int,
    block_size: int,
    edge_anchored_fit: bool,
) -> List[List[Tile]]:
    tiles: List[List[Tile]] = []

    for out_y0 in range(0, height, block_size):
        out_y1 = min(out_y0 + block_size, height)
        row: List[Tile] = []

        if edge_anchored_fit:
            fit_y1 = out_y1
            fit_y0 = fit_y1 - block_size
            if fit_y0 < 0:
                fit_y0 = 0
                fit_y1 = min(block_size, height)
        else:
            fit_y0 = out_y0
            fit_y1 = out_y1

        for out_x0 in range(0, width, block_size):
            out_x1 = min(out_x0 + block_size, width)

            if edge_anchored_fit:
                fit_x1 = out_x1
                fit_x0 = fit_x1 - block_size
                if fit_x0 < 0:
                    fit_x0 = 0
                    fit_x1 = min(block_size, width)
            else:
                fit_x0 = out_x0
                fit_x1 = out_x1

            row.append(
                Tile(
                    out_x0=int(out_x0),
                    out_y0=int(out_y0),
                    out_x1=int(out_x1),
                    out_y1=int(out_y1),
                    fit_x0=int(fit_x0),
                    fit_y0=int(fit_y0),
                    fit_x1=int(fit_x1),
                    fit_y1=int(fit_y1),
                    out_w=int(out_x1 - out_x0),
                    out_h=int(out_y1 - out_y0),
                    fit_w=int(fit_x1 - fit_x0),
                    fit_h=int(fit_y1 - fit_y0),
                )
            )

        tiles.append(row)

    return tiles


# ============================================================
# Feature matching / homography
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
        H, _ = cv2.findHomography(
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

        if np.count_nonzero(keep) >= max(8, len(pts_t) // 4):
            pts_t = pts_t[keep]
            pts_r = pts_r[keep]

    return pts_t, pts_r


def parent_H_for_block(
    parent_H_grid: Optional[np.ndarray],
    parent_block_size: Optional[int],
    out_x0: int,
    out_y0: int,
) -> Optional[np.ndarray]:
    if parent_H_grid is None or parent_block_size is None:
        return None

    py = out_y0 // parent_block_size
    px = out_x0 // parent_block_size

    py = min(parent_H_grid.shape[0] - 1, max(0, py))
    px = min(parent_H_grid.shape[1] - 1, max(0, px))

    return parent_H_grid[py, px].reshape(3, 3).copy()


def render_prediction_from_H_grid(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    H_grid: np.ndarray,
    block_size: int,
    edge_anchored_fit: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_y.shape
    pred = np.zeros((height, width), dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    tiles = make_tiles(width, height, block_size, edge_anchored_fit)

    for by_idx, row in enumerate(tiles):
        for bx_idx, tile in enumerate(row):
            H = H_grid[by_idx, bx_idx].reshape(3, 3)

            map_x, map_y, valid = homography_maps_for_roi(
                H=H,
                bx=tile.out_x0,
                by=tile.out_y0,
                bw=tile.out_w,
                bh=tile.out_h,
                width=width,
                height=height,
            )

            pred_roi = remap_ref(ref_y, map_x, map_y)
            pred[tile.out_y0:tile.out_y1, tile.out_x0:tile.out_x1] = pred_roi
            valid_all[tile.out_y0:tile.out_y1, tile.out_x0:tile.out_x1] = valid

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
    edge_anchored_fit = not args.disable_edge_anchored_fit
    tiles = make_tiles(width, height, block_size, edge_anchored_fit)

    H_grid = np.zeros((ny, nx, 3, 3), dtype=np.float64)

    source_grid = np.zeros((ny, nx), dtype=np.float64)
    match_count_grid = np.zeros((ny, nx), dtype=np.float64)
    inlier_count_grid = np.zeros((ny, nx), dtype=np.float64)
    reproj_mae_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    candidate_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    fallback_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    chosen_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)

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

    for by_idx, row in enumerate(tiles):
        row_records = []

        for bx_idx, tile in enumerate(row):
            parent_H = parent_H_for_block(
                parent_H_grid,
                parent_block_size,
                tile.out_x0,
                tile.out_y0,
            )

            if parent_H is None:
                fallback_H = root_fallback_H.copy()
                fallback_source = "root_fallback"
            else:
                fallback_H = parent_H.copy()
                fallback_source = "parent_inherit"

            fallback_cost, fallback_valid_ratio = eval_block_photometric_cost(
                target_y,
                ref_y,
                fallback_H,
                tile.out_x0,
                tile.out_y0,
                tile.out_w,
                tile.out_h,
                args.min_block_valid_ratio,
            )

            pts_t, pts_r = collect_matches_for_block(
                pts_t_all,
                pts_r_all,
                tile.fit_x0,
                tile.fit_y0,
                tile.fit_w,
                tile.fit_h,
                width,
                height,
                margin,
                fallback_H,
                args.parent_match_gate,
            )

            H_candidate, inlier_count, reproj_mae, fit_reason = fit_local_homography(
                pts_t,
                pts_r,
                args.ransac_thresh,
                args.min_matches,
                args.min_inliers,
            )

            chosen_H = fallback_H
            chosen_source = fallback_source
            candidate_cost = float("inf")
            chosen_cost = fallback_cost
            chosen_valid_ratio = fallback_valid_ratio
            reason = fit_reason

            if H_candidate is not None:
                candidate_cost, candidate_valid_ratio = eval_block_photometric_cost(
                    target_y,
                    ref_y,
                    H_candidate,
                    tile.out_x0,
                    tile.out_y0,
                    tile.out_w,
                    tile.out_h,
                    args.min_block_valid_ratio,
                )

                if not args.disable_cost_gate:
                    if np.isfinite(fallback_cost):
                        accept_by_cost = candidate_cost <= fallback_cost - args.min_gain
                    else:
                        accept_by_cost = np.isfinite(candidate_cost)
                else:
                    accept_by_cost = True

                if np.isfinite(candidate_cost) and accept_by_cost:
                    chosen_H = H_candidate
                    chosen_source = "local_fit"
                    chosen_cost = candidate_cost
                    chosen_valid_ratio = candidate_valid_ratio
                    reason = "local_fit_accepted"
                    accepted += 1
                else:
                    chosen_H = fallback_H
                    chosen_source = "local_fit_rejected_cost"
                    chosen_cost = fallback_cost
                    chosen_valid_ratio = fallback_valid_ratio
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
                    "tile_index_x": int(bx_idx),
                    "tile_index_y": int(by_idx),
                    "out_x": int(tile.out_x0),
                    "out_y": int(tile.out_y0),
                    "out_w": int(tile.out_w),
                    "out_h": int(tile.out_h),
                    "fit_x": int(tile.fit_x0),
                    "fit_y": int(tile.fit_y0),
                    "fit_w": int(tile.fit_w),
                    "fit_h": int(tile.fit_h),
                    "block_x": int(tile.out_x0),
                    "block_y": int(tile.out_y0),
                    "block_w": int(tile.out_w),
                    "block_h": int(tile.out_h),
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


def run_hierarchical_pair(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    target_idx: int,
    ref_idx: int,
    bitdepth: int,
    args,
    pair_out_dir: str,
) -> PairHResult:
    ensure_dir(pair_out_dir)

    width = args.width
    height = args.height
    edge_anchored_fit = not args.disable_edge_anchored_fit

    match = detect_and_match_orb(
        target_y=target_y,
        ref_y=ref_y,
        bitdepth=bitdepth,
        max_features=args.max_features,
        ratio=args.match_ratio,
        use_clahe=not args.no_clahe,
    )

    save_match_vis(
        os.path.join(pair_out_dir, "match_vis.png"),
        target_y,
        ref_y,
        bitdepth,
        match,
    )

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
        root_H = H_global_homography if H_global_homography is not None else H_global_translation
    else:
        raise ValueError(args.root_fallback)

    all_level_records = []
    all_level_costs = []
    all_level_aux_summary = []

    parent_H_grid = None
    parent_block_size = None

    final_pred = None
    final_valid = None
    final_H_grid = None
    final_records = None
    final_block_size = None

    block_size = args.start_block_size

    for level in range(args.levels):
        if block_size < args.min_block_size:
            break

        print(f"[PAIR t{target_idx:03d}->r{ref_idx:03d}] level={level}, block={block_size}")

        H_grid, level_records, aux = run_one_level(
            target_y=target_y,
            ref_y=ref_y,
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
            target_y=target_y,
            ref_y=ref_y,
            H_grid=H_grid,
            block_size=block_size,
            edge_anchored_fit=edge_anchored_fit,
        )

        cost = calc_cost(target_y, pred, valid, bitdepth)
        level_tag = f"level{level:02d}_block{block_size}"

        write_single_yuv420(
            os.path.join(pair_out_dir, f"pred_hier_{level_tag}.yuv"),
            pred,
            width,
            height,
            bitdepth,
        )

        save_gray_png(
            os.path.join(pair_out_dir, f"pred_hier_{level_tag}.png"),
            pred,
            bitdepth,
        )

        save_diff_png(
            os.path.join(pair_out_dir, f"diff_hier_{level_tag}.png"),
            target_y,
            pred,
            valid,
        )

        save_scalar_map_png(
            os.path.join(pair_out_dir, f"source_map_{level_tag}.png"),
            aux["source_grid"],
            width,
            height,
            block_size,
        )

        all_level_records.append(level_records)
        all_level_costs.append(
            {
                "level": int(level),
                "block_size": int(block_size),
                "cost": cost,
                "accepted": int(aux["accepted"][0]),
                "inherited": int(aux["inherited"][0]),
                "rejected": int(aux["rejected"][0]),
            }
        )

        all_level_aux_summary.append(
            {
                "source_grid": aux["source_grid"].tolist(),
                "match_count_grid": aux["match_count_grid"].tolist(),
                "inlier_count_grid": aux["inlier_count_grid"].tolist(),
                "chosen_cost_grid": np.where(
                    np.isfinite(aux["chosen_cost_grid"]),
                    aux["chosen_cost_grid"],
                    -1,
                ).tolist(),
                "valid_ratio_grid": aux["valid_ratio_grid"].tolist(),
            }
        )

        final_pred = pred
        final_valid = valid
        final_H_grid = H_grid
        final_records = level_records
        final_block_size = block_size

        parent_H_grid = H_grid
        parent_block_size = block_size
        block_size //= 2

    if final_pred is None:
        raise RuntimeError("No level was processed for pair.")

    direct_yuv_path = os.path.join(pair_out_dir, "pred_hier_final.yuv")
    write_single_yuv420(direct_yuv_path, final_pred, width, height, bitdepth)
    save_gray_png(os.path.join(pair_out_dir, "pred_hier_final.png"), final_pred, bitdepth)
    save_diff_png(os.path.join(pair_out_dir, "diff_hier_final.png"), target_y, final_pred, final_valid)

    direct_cost = calc_cost(target_y, final_pred, final_valid, bitdepth)

    pair_json = {
        "target_idx": int(target_idx),
        "ref_idx": int(ref_idx),
        "final_block_size": int(final_block_size),
        "direct_cost": direct_cost,
        "levels_summary": all_level_costs,
        "levels_aux_grids": all_level_aux_summary,
        "final_records": final_records,
        "root_H": {
            "identity": H_identity.tolist(),
            "global_translation": H_global_translation.tolist(),
            "global_homography": H_global_homography.tolist() if H_global_homography is not None else None,
            "selected": root_H.tolist(),
        },
    }

    with open(os.path.join(pair_out_dir, "pair_homography_result.json"), "w", encoding="utf-8") as f:
        json.dump(pair_json, f, indent=2)

    return PairHResult(
        target_idx=target_idx,
        ref_idx=ref_idx,
        final_block_size=int(final_block_size),
        final_H_grid=final_H_grid,
        final_records=final_records,
        direct_cost=direct_cost,
        direct_yuv_path=direct_yuv_path,
    )


# ============================================================
# Pair generation
# ============================================================

def parse_pairs_string(s: str) -> List[Tuple[int, int]]:
    pairs = []
    if not s.strip():
        return pairs

    for token in s.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            a, b = token.split(":")
        elif "->" in token:
            a, b = token.split("->")
        else:
            raise ValueError(f"Invalid pair token: {token}")

        pairs.append((int(a), int(b)))

    return pairs


def read_pairs_file(path: str) -> List[Tuple[int, int]]:
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            line = line.replace(",", " ").replace(":", " ").replace("->", " ")
            parts = line.split()

            if len(parts) < 2:
                continue

            pairs.append((int(parts[0]), int(parts[1])))

    return pairs


def generate_dyadic_pairs(gop_start: int, gop_size: int) -> List[Tuple[int, int]]:
    start = gop_start
    end = gop_start + gop_size - 1

    pairs = []

    def rec(a: int, b: int):
        if b <= a + 1:
            return

        m = (a + b) // 2
        pairs.append((m, a))
        pairs.append((m, b))

        rec(a, m)
        rec(m, b)

    rec(start, end)

    # remove duplicates while preserving order
    seen = set()
    out = []

    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return out


def generate_all_ordered_pairs(gop_start: int, gop_size: int) -> List[Tuple[int, int]]:
    frames = list(range(gop_start, gop_start + gop_size))
    pairs = []

    for t in frames:
        for r in frames:
            if t != r:
                pairs.append((t, r))

    return pairs


def get_gop_pairs(args) -> List[Tuple[int, int]]:
    if args.pairs:
        return parse_pairs_string(args.pairs)

    if args.pairs_file:
        return read_pairs_file(args.pairs_file)

    if args.pair_mode == "dyadic":
        return generate_dyadic_pairs(args.gop_start, args.gop_size)

    if args.pair_mode == "all":
        return generate_all_ordered_pairs(args.gop_start, args.gop_size)

    raise ValueError(args.pair_mode)


# ============================================================
# CP pseudo-observation generation
# ============================================================

def cp_points_for_tile(tile: Tile, pattern: str, inset: float) -> np.ndarray:
    x0 = tile.out_x0 + inset
    y0 = tile.out_y0 + inset
    x1 = tile.out_x1 - 1 - inset
    y1 = tile.out_y1 - 1 - inset

    if x1 < x0:
        x0 = tile.out_x0
        x1 = tile.out_x1 - 1

    if y1 < y0:
        y0 = tile.out_y0
        y1 = tile.out_y1 - 1

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    if pattern == "center":
        return np.array([[cx, cy]], dtype=np.float32)

    if pattern == "center4":
        return np.array(
            [
                [cx, cy],
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )

    if pattern == "grid3":
        xs = [x0, cx, x1]
        ys = [y0, cy, y1]
        pts = []
        for yy in ys:
            for xx in xs:
                pts.append([xx, yy])
        return np.array(pts, dtype=np.float32)

    raise ValueError(pattern)


def build_observations_from_pairs(
    pair_results: List[PairHResult],
    width: int,
    height: int,
    K: np.ndarray,
    args,
) -> Tuple[ObservationSet, Dict[Tuple[int, int, int], int], List[Dict], List[Tuple[int, int]]]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    target_frame = []
    ref_frame = []
    depth_index = []
    pair_index = []
    px = []
    py = []
    qx = []
    qy = []
    ray_x = []
    ray_y = []
    weight = []
    block_source = []
    block_cost = []
    meta = []

    depth_key_to_index: Dict[Tuple[int, int, int], int] = {}
    depth_meta: List[Dict] = []
    pairs_meta: List[Tuple[int, int]] = []

    for pidx, pr in enumerate(pair_results):
        pairs_meta.append((pr.target_idx, pr.ref_idx))

        final_block_size = pr.final_block_size
        tiles = make_tiles(
            width=width,
            height=height,
            block_size=final_block_size,
            edge_anchored_fit=not args.disable_edge_anchored_fit,
        )

        for by_idx, row in enumerate(tiles):
            for bx_idx, tile in enumerate(row):
                rec = pr.final_records[by_idx][bx_idx]

                source = rec["source"]

                if args.obs_exclude_rejected and source == "local_fit_rejected_cost":
                    continue

                chosen_cost = rec.get("chosen_cost", None)
                valid_ratio = rec.get("valid_ratio", 0.0)
                inlier_count = rec.get("inlier_count", 0)

                if chosen_cost is None or not np.isfinite(chosen_cost):
                    continue

                if valid_ratio < args.obs_min_valid_ratio:
                    continue

                if inlier_count < args.obs_min_inliers and source == "local_fit":
                    continue

                H = np.asarray(rec["H"], dtype=np.float64).reshape(3, 3)
                cps = cp_points_for_tile(tile, args.cp_pattern, args.cp_inset)
                qs = apply_homography_points(H, cps)

                dkey = (int(pr.target_idx), int(by_idx), int(bx_idx))
                if dkey not in depth_key_to_index:
                    depth_key_to_index[dkey] = len(depth_meta)
                    depth_meta.append(
                        {
                            "target_idx": int(pr.target_idx),
                            "block_y_idx": int(by_idx),
                            "block_x_idx": int(bx_idx),
                            "block_x": int(tile.out_x0),
                            "block_y": int(tile.out_y0),
                            "block_w": int(tile.out_w),
                            "block_h": int(tile.out_h),
                            "block_size": int(final_block_size),
                        }
                    )

                didx = depth_key_to_index[dkey]

                if source == "local_fit":
                    src_w = 1.0
                elif source == "parent_inherit":
                    src_w = 0.75
                elif source == "root_fallback":
                    src_w = 0.50
                else:
                    src_w = 0.50

                cost_w = 1.0 / np.sqrt(max(float(chosen_cost), 1.0))
                src_weight = src_w * cost_w

                for k in range(cps.shape[0]):
                    x, y = float(cps[k, 0]), float(cps[k, 1])
                    u, v = float(qs[k, 0]), float(qs[k, 1])

                    if not (0.0 <= u <= width - 1.0 and 0.0 <= v <= height - 1.0):
                        continue

                    target_frame.append(int(pr.target_idx))
                    ref_frame.append(int(pr.ref_idx))
                    depth_index.append(int(didx))
                    pair_index.append(int(pidx))
                    px.append(x)
                    py.append(y)
                    qx.append(u)
                    qy.append(v)
                    ray_x.append((x - cx) / fx)
                    ray_y.append((y - cy) / fy)
                    weight.append(float(src_weight))
                    block_source.append(source)
                    block_cost.append(float(chosen_cost))
                    meta.append(
                        {
                            "pair_index": int(pidx),
                            "target_idx": int(pr.target_idx),
                            "ref_idx": int(pr.ref_idx),
                            "block_y_idx": int(by_idx),
                            "block_x_idx": int(bx_idx),
                            "source": source,
                            "chosen_cost": float(chosen_cost),
                        }
                    )

    obs = ObservationSet(
        target_frame=np.asarray(target_frame, dtype=np.int32),
        ref_frame=np.asarray(ref_frame, dtype=np.int32),
        depth_index=np.asarray(depth_index, dtype=np.int32),
        pair_index=np.asarray(pair_index, dtype=np.int32),
        px=np.asarray(px, dtype=np.float64),
        py=np.asarray(py, dtype=np.float64),
        qx=np.asarray(qx, dtype=np.float64),
        qy=np.asarray(qy, dtype=np.float64),
        ray_x=np.asarray(ray_x, dtype=np.float64),
        ray_y=np.asarray(ray_y, dtype=np.float64),
        weight=np.asarray(weight, dtype=np.float64),
        block_source=np.asarray(block_source, dtype=object),
        block_cost=np.asarray(block_cost, dtype=np.float64),
        meta=meta,
    )

    return obs, depth_key_to_index, depth_meta, pairs_meta


# ============================================================
# Camera-like model
# ============================================================

def make_fixed_K(width: int, height: int, args) -> np.ndarray:
    if args.fx > 0:
        fx = float(args.fx)
    else:
        fx = float(max(width, height) * args.f_scale)

    if args.fy > 0:
        fy = float(args.fy)
    else:
        fy = float(max(width, height) * args.f_scale)

    if args.cx >= 0:
        cx = float(args.cx)
    else:
        cx = 0.5 * float(width - 1)

    if args.cy >= 0:
        cy = float(args.cy)
    else:
        cy = 0.5 * float(height - 1)

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def build_frame_index(frames:
