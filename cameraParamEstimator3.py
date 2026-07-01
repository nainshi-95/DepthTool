#!/usr/bin/env python3
# gop_camera_like_torch_lowres_rho.py
#
# GOP-level camera-like fitting from hierarchical block homography pseudo-GT.
#
# Updates in this version:
#   1) --reuse-pair-results: skip pair-wise homography and load existing
#      pair_tXXX_rYYY/pair_homography_result.json or result.json.
#   2) Alternating fitting order changed to:
#        fixed rho -> pose R|t update
#        fixed pose -> block-wise rho update
#      repeated.
#   3) Block-level trimming is based on fitted camera-like residual:
#        after each fitting round, remove worst X% pair-blocks and refit.
#      This is much less dependent on hand-written validity heuristics.
#   4) Memory reduction:
#        - optional active observation cap per iteration
#        - sparse Jacobian for depth update
#        - row-batched rendering
#
# Coordinate convention:
#   target pixel x_t -> ref pixel x_r
#
# Pose convention:
#   R_i, t_i maps camera_i coordinates -> GOP-local world coordinates.
#   anchor frame is fixed as R=I, t=0.
#
# Point model:
#   ray_t = [(x-cx)/fx, (y-cy)/fy, 1]
#   X_t   = ray_t / rho[target_frame, block]
#   X_w   = R_t X_t + t_t
#   X_r   = R_r^T (X_w - t_r)

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy import sparse
except ImportError:
    least_squares = None
    sparse = None

try:
    import torch
except ImportError as e:
    raise ImportError("This script requires PyTorch. Install a CUDA build if you want GPU acceleration.") from e


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
    final_H_grid: Optional[np.ndarray]
    final_records: List[List[Dict]]
    direct_cost: Dict
    direct_yuv_path: str


@dataclass
class ObservationSet:
    target_frame: np.ndarray
    ref_frame: np.ndarray
    depth_index: np.ndarray
    pair_index: np.ndarray
    pair_block_index: np.ndarray
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


@dataclass
class PreparedObs:
    target_cid: np.ndarray
    ref_cid: np.ndarray
    depth_index: np.ndarray
    pair_block_index: np.ndarray
    px: np.ndarray
    py: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    rays: np.ndarray
    sqrt_weight: np.ndarray


@dataclass
class CameraLikeFitResult:
    K: np.ndarray
    frames: List[int]
    frame_to_index: Dict[int, int]
    anchor_frame: int
    anchor_cid: int
    rvecs: np.ndarray
    tvecs: np.ndarray
    log_rho: np.ndarray
    rho: np.ndarray
    pair_block_valid: np.ndarray
    depth_valid: np.ndarray
    residual_px: np.ndarray
    depth_neighbors: List[Tuple[int, int]]
    temporal_pairs: List[Tuple[int, int]]
    rho_grid_frames: List[int]
    rho_grids: Optional[np.ndarray]
    pair_block_weight: Optional[np.ndarray]
    summary: Dict


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


def make_tiles(width: int, height: int, block_size: int, edge_anchored_fit: bool) -> List[List[Tile]]:
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
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_global_homography_H(pts_t: np.ndarray, pts_r: np.ndarray, ransac_thresh: float) -> Optional[np.ndarray]:
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
    reproj_mae = float(np.mean(err[mask])) if np.any(mask) else float(np.mean(err))

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

    save_match_vis(os.path.join(pair_out_dir, "match_vis.png"), target_y, ref_y, bitdepth, match)

    H_identity = np.eye(3, dtype=np.float64)
    H_global_translation = estimate_global_translation_H(match.pts_target, match.pts_ref)
    H_global_homography = estimate_global_homography_H(match.pts_target, match.pts_ref, args.ransac_thresh)

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

        pred, valid = render_prediction_from_H_grid(target_y, ref_y, H_grid, block_size, edge_anchored_fit)
        cost = calc_cost(target_y, pred, valid, bitdepth)
        level_tag = f"level{level:02d}_block{block_size}"

        if not args.no_pair_yuv_png:
            write_single_yuv420(os.path.join(pair_out_dir, f"pred_hier_{level_tag}.yuv"), pred, width, height, bitdepth)
            save_gray_png(os.path.join(pair_out_dir, f"pred_hier_{level_tag}.png"), pred, bitdepth)
            save_diff_png(os.path.join(pair_out_dir, f"diff_hier_{level_tag}.png"), target_y, pred, valid)
            save_scalar_map_png(os.path.join(pair_out_dir, f"source_map_{level_tag}.png"), aux["source_grid"], width, height, block_size)

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
                "chosen_cost_grid": np.where(np.isfinite(aux["chosen_cost_grid"]), aux["chosen_cost_grid"], -1).tolist(),
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
    if not args.no_pair_yuv_png:
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
# Pair loading / generation
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

    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def generate_all_ordered_pairs(gop_start: int, gop_size: int) -> List[Tuple[int, int]]:
    frames = list(range(gop_start, gop_start + gop_size))
    return [(t, r) for t in frames for r in frames if t != r]


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


def find_pair_json_path(base_dir: str, target_idx: int, ref_idx: int) -> Optional[str]:
    pair_dir = os.path.join(base_dir, f"pair_t{target_idx:03d}_r{ref_idx:03d}")
    candidates = [
        os.path.join(pair_dir, "pair_homography_result.json"),
        os.path.join(pair_dir, "result.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_pair_result_from_json(path: str) -> PairHResult:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_idx = int(data["target_idx"])
    ref_idx = int(data["ref_idx"])

    if "final_records" in data:
        final_records = data["final_records"]
        final_block_size = int(data["final_block_size"])
        direct_cost = data.get("direct_cost", {})
        direct_yuv_path = data.get("direct_yuv_path", "")
    else:
        # Compatibility with original hierarchical_block_homography.py result.json.
        levels_blocks = data.get("levels_blocks", None)
        if not levels_blocks:
            raise ValueError(f"Cannot find levels_blocks/final_records in {path}")
        final_records = levels_blocks[-1]
        final_block_size = int(data["final"]["block_size"])
        direct_cost = data["final"].get("cost", {})
        direct_yuv_path = data["final"].get("output_yuv", "")

    return PairHResult(
        target_idx=target_idx,
        ref_idx=ref_idx,
        final_block_size=final_block_size,
        final_H_grid=None,
        final_records=final_records,
        direct_cost=direct_cost,
        direct_yuv_path=direct_yuv_path,
    )


def load_existing_pair_results(args, pairs: List[Tuple[int, int]]) -> Tuple[List[PairHResult], List[Dict]]:
    base_dir = args.pair_results_dir if args.pair_results_dir else args.output_dir
    pair_results = []
    failed = []

    for target_idx, ref_idx in pairs:
        path = find_pair_json_path(base_dir, target_idx, ref_idx)
        if path is None:
            msg = f"missing pair result for target={target_idx}, ref={ref_idx} under {base_dir}"
            failed.append({"target_idx": int(target_idx), "ref_idx": int(ref_idx), "error": msg})
            print(f"[WARN] {msg}")
            if not args.skip_failed_pairs:
                raise FileNotFoundError(msg)
            continue

        try:
            pr = load_pair_result_from_json(path)
            pair_results.append(pr)
            print(f"[REUSE] loaded {path}")
        except Exception as e:
            failed.append({"target_idx": int(target_idx), "ref_idx": int(ref_idx), "error": str(e)})
            print(f"[WARN] failed to load {path}: {e}")
            if not args.skip_failed_pairs:
                raise

    return pair_results, failed


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
        return np.array([[cx, cy], [x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    if pattern == "grid3":
        xs = [x0, cx, x1]
        ys = [y0, cy, y1]
        return np.array([[xx, yy] for yy in ys for xx in xs], dtype=np.float32)

    raise ValueError(pattern)


def build_observations_from_pairs(
    pair_results: List[PairHResult],
    width: int,
    height: int,
    K: np.ndarray,
    args,
) -> Tuple[ObservationSet, Dict[Tuple[int, int, int], int], List[Dict], List[Tuple[int, int]], List[Dict]]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    target_frame = []
    ref_frame = []
    depth_index = []
    pair_index = []
    pair_block_index = []
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
    pair_block_meta: List[Dict] = []

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

                pbidx = len(pair_block_meta)
                pair_block_meta.append(
                    {
                        "pair_block_index": int(pbidx),
                        "pair_index": int(pidx),
                        "target_idx": int(pr.target_idx),
                        "ref_idx": int(pr.ref_idx),
                        "depth_index": int(didx),
                        "block_y_idx": int(by_idx),
                        "block_x_idx": int(bx_idx),
                        "block_x": int(tile.out_x0),
                        "block_y": int(tile.out_y0),
                        "block_w": int(tile.out_w),
                        "block_h": int(tile.out_h),
                        "block_size": int(final_block_size),
                        "source": source,
                        "chosen_cost": float(chosen_cost),
                        "valid_ratio": float(valid_ratio),
                    }
                )

                if source == "local_fit":
                    src_w = 1.0
                elif source == "parent_inherit":
                    src_w = 0.75
                elif source == "root_fallback":
                    src_w = 0.50
                else:
                    src_w = 0.50

                cost_w = 1.0 / np.sqrt(max(float(chosen_cost), 1.0))
                # valid_ratio is a good soft reliability cue for this pair/block.
                # Use JSON scalar values, not the saved PNG visualization.
                vr = float(np.clip(valid_ratio, 0.0, 1.0))
                vmin = float(getattr(args, "valid_ratio_soft_min", 0.20))
                gamma = float(getattr(args, "valid_ratio_weight_gamma", 2.0))
                if vr <= vmin:
                    valid_w = 0.05
                else:
                    valid_w = ((vr - vmin) / max(1.0 - vmin, 1e-6)) ** gamma
                    valid_w = float(np.clip(valid_w, 0.05, 1.0))
                src_weight = src_w * cost_w * valid_w

                for k in range(cps.shape[0]):
                    x, y = float(cps[k, 0]), float(cps[k, 1])
                    u, v = float(qs[k, 0]), float(qs[k, 1])

                    if not (0.0 <= u <= width - 1.0 and 0.0 <= v <= height - 1.0):
                        continue

                    target_frame.append(int(pr.target_idx))
                    ref_frame.append(int(pr.ref_idx))
                    depth_index.append(int(didx))
                    pair_index.append(int(pidx))
                    pair_block_index.append(int(pbidx))
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
                            "pair_block_index": int(pbidx),
                            "target_idx": int(pr.target_idx),
                            "ref_idx": int(pr.ref_idx),
                            "depth_index": int(didx),
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
        pair_block_index=np.asarray(pair_block_index, dtype=np.int32),
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

    return obs, depth_key_to_index, depth_meta, pairs_meta, pair_block_meta


# ============================================================
# Camera-like model
# ============================================================

def make_fixed_K(width: int, height: int, args) -> np.ndarray:
    fx = float(args.fx) if args.fx > 0 else float(max(width, height) * args.f_scale)
    fy = float(args.fy) if args.fy > 0 else float(max(width, height) * args.f_scale)
    cx = float(args.cx) if args.cx >= 0 else 0.5 * float(width - 1)
    cy = float(args.cy) if args.cy >= 0 else 0.5 * float(height - 1)

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def build_frame_index(frames: List[int]) -> Tuple[Dict[int, int], List[int]]:
    frames_sorted = sorted(set(int(f) for f in frames))
    frame_to_index = {f: i for i, f in enumerate(frames_sorted)}
    return frame_to_index, frames_sorted


def prepare_observations(obs: ObservationSet, frame_to_index: Dict[int, int]) -> PreparedObs:
    target_cid = np.asarray([frame_to_index[int(f)] for f in obs.target_frame], dtype=np.int32)
    ref_cid = np.asarray([frame_to_index[int(f)] for f in obs.ref_frame], dtype=np.int32)

    rays = np.stack(
        [obs.ray_x.astype(np.float64), obs.ray_y.astype(np.float64), np.ones_like(obs.ray_x, dtype=np.float64)],
        axis=1,
    )

    w = obs.weight.astype(np.float64).copy()
    finite = np.isfinite(w) & (w > 0.0)
    if np.any(finite):
        med = float(np.median(w[finite]))
        if med > 1e-12:
            w = w / med
    w = np.clip(w, 1e-4, 100.0)

    return PreparedObs(
        target_cid=target_cid,
        ref_cid=ref_cid,
        depth_index=obs.depth_index.astype(np.int32),
        pair_block_index=obs.pair_block_index.astype(np.int32),
        px=obs.px.astype(np.float64),
        py=obs.py.astype(np.float64),
        qx=obs.qx.astype(np.float64),
        qy=obs.qy.astype(np.float64),
        rays=rays,
        sqrt_weight=np.sqrt(w),
    )


def build_depth_neighbors(depth_meta: List[Dict]) -> List[Tuple[int, int]]:
    key_to_idx = {}
    for i, m in enumerate(depth_meta):
        key = (int(m["target_idx"]), int(m["block_y_idx"]), int(m["block_x_idx"]))
        key_to_idx[key] = i

    neighbors = []
    for i, m in enumerate(depth_meta):
        t = int(m["target_idx"])
        by = int(m["block_y_idx"])
        bx = int(m["block_x_idx"])
        for nb in [(t, by, bx + 1), (t, by + 1, bx)]:
            j = key_to_idx.get(nb, None)
            if j is not None:
                neighbors.append((i, j))
    return neighbors


def build_temporal_pairs(frames: List[int], frame_to_index: Dict[int, int]) -> List[Tuple[int, int]]:
    return [(frame_to_index[a], frame_to_index[b]) for a, b in zip(frames[:-1], frames[1:])]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float(np.median(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0

    v = values[valid]
    w = weights[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * float(cw[-1])
    return float(v[np.searchsorted(cw, cutoff)])


def initialize_camera_like(
    prep: PreparedObs,
    frames: List[int],
    frame_to_index: Dict[int, int],
    anchor_cid: int,
    K: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = len(frames)
    num_depths = int(np.max(prep.depth_index)) + 1

    rvecs = np.zeros((num_frames, 3), dtype=np.float64)
    tvecs = np.zeros((num_frames, 3), dtype=np.float64)

    init_rho = float(np.clip(args.init_rho, args.min_rho, args.max_rho))
    log_rho = np.full(num_depths, np.log(init_rho), dtype=np.float64)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    opt_cids = [cid for cid in range(num_frames) if cid != anchor_cid]
    cid_to_var = {cid: i for i, cid in enumerate(opt_cids)}

    pair_keys = sorted(set(zip(prep.target_cid.tolist(), prep.ref_cid.tolist())))
    rows_x, rhs_x, w_x = [], [], []
    rows_y, rhs_y, w_y = [], [], []

    for tcid, rcid in pair_keys:
        mask = (prep.target_cid == tcid) & (prep.ref_cid == rcid)
        if np.count_nonzero(mask) < 1:
            continue

        dx = weighted_median(prep.qx[mask] - prep.px[mask], prep.sqrt_weight[mask] ** 2)
        dy = weighted_median(prep.qy[mask] - prep.py[mask], prep.sqrt_weight[mask] ** 2)
        ww = float(np.sqrt(np.count_nonzero(mask)))

        bx = dx / max(fx * init_rho, 1e-12)
        by = dy / max(fy * init_rho, 1e-12)

        row = np.zeros(len(opt_cids), dtype=np.float64)
        if tcid != anchor_cid:
            row[cid_to_var[tcid]] += 1.0
        if rcid != anchor_cid:
            row[cid_to_var[rcid]] -= 1.0

        if np.any(np.abs(row) > 0):
            rows_x.append(row.copy())
            rhs_x.append(bx)
            w_x.append(ww)
            rows_y.append(row.copy())
            rhs_y.append(by)
            w_y.append(ww)

    if rows_x:
        A = np.vstack(rows_x)
        b = np.asarray(rhs_x, dtype=np.float64)
        w = np.asarray(w_x, dtype=np.float64)
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        for cid, vi in cid_to_var.items():
            tvecs[cid, 0] = sol[vi]

    if rows_y:
        A = np.vstack(rows_y)
        b = np.asarray(rhs_y, dtype=np.float64)
        w = np.asarray(w_y, dtype=np.float64)
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        for cid, vi in cid_to_var.items():
            tvecs[cid, 1] = sol[vi]

    return rvecs, tvecs, log_rho


def poses_to_vec(rvecs: np.ndarray, tvecs: np.ndarray, opt_frame_cids: List[int]) -> np.ndarray:
    vals = []
    for cid in opt_frame_cids:
        vals.extend(rvecs[cid].tolist())
        vals.extend(tvecs[cid].tolist())
    return np.asarray(vals, dtype=np.float64)


def vec_to_poses(x: np.ndarray, base_rvecs: np.ndarray, base_tvecs: np.ndarray, opt_frame_cids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    rvecs = base_rvecs.copy()
    tvecs = base_tvecs.copy()
    k = 0
    for cid in opt_frame_cids:
        rvecs[cid] = x[k:k + 3]
        k += 3
        tvecs[cid] = x[k:k + 3]
        k += 3
    return rvecs, tvecs


def all_rotation_matrices(rvecs: np.ndarray) -> np.ndarray:
    Rs = np.zeros((rvecs.shape[0], 3, 3), dtype=np.float64)
    for i in range(rvecs.shape[0]):
        Rs[i] = rodrigues(rvecs[i])
    return Rs


def project_prepared_observations(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    log_rho: np.ndarray,
    min_rho: float,
    max_rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = obs_indices
    Rs = all_rotation_matrices(rvecs)

    tcid = prep.target_cid[idx]
    rcid = prep.ref_cid[idx]
    didx = prep.depth_index[idx]

    log_rho_safe = np.clip(log_rho, np.log(min_rho), np.log(max_rho))
    rho = np.exp(log_rho_safe[didx])
    X_t = prep.rays[idx] / rho[:, None]

    X_w = np.einsum("nij,nj->ni", Rs[tcid], X_t) + tvecs[tcid]
    X_r = np.einsum("nji,nj->ni", Rs[rcid], X_w - tvecs[rcid])

    z = X_r[:, 2]
    z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * (X_r[:, 0] / z_safe) + cx
    v = fy * (X_r[:, 1] / z_safe) + cy
    return u, v, z


def camera_residual_vector(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    log_rho: np.ndarray,
    args,
    depth_neighbors: Optional[List[Tuple[int, int]]] = None,
    temporal_pairs: Optional[List[Tuple[int, int]]] = None,
    opt_frame_cids: Optional[List[int]] = None,
    include_depth_prior: bool = True,
    include_pose_prior: bool = True,
    active_depth_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    u, v, z = project_prepared_observations(
        prep=prep,
        obs_indices=obs_indices,
        K=K,
        rvecs=rvecs,
        tvecs=tvecs,
        log_rho=log_rho,
        min_rho=args.min_rho,
        max_rho=args.max_rho,
    )

    sw = prep.sqrt_weight[obs_indices]
    rx = (u - prep.qx[obs_indices]) * sw
    ry = (v - prep.qy[obs_indices]) * sw

    bad_z = z <= args.z_min
    if np.any(bad_z):
        penalty = args.z_penalty * (args.z_min - z[bad_z])
        rx[bad_z] += penalty
        ry[bad_z] += penalty

    if args.residual_clip_px > 0.0:
        clip = float(args.residual_clip_px)
        rx = np.clip(rx, -clip, clip)
        ry = np.clip(ry, -clip, clip)

    residuals = [rx, ry]

    if include_depth_prior and args.depth_prior_weight > 0.0:
        log_init = np.log(float(np.clip(args.init_rho, args.min_rho, args.max_rho)))
        w = np.sqrt(float(args.depth_prior_weight))
        if active_depth_mask is None:
            residuals.append(w * (log_rho - log_init))
        else:
            ids = np.flatnonzero(active_depth_mask)
            residuals.append(w * (log_rho[ids] - log_init))

    if include_depth_prior and depth_neighbors is not None and len(depth_neighbors) > 0 and args.depth_smooth_weight > 0.0:
        a = np.asarray([p[0] for p in depth_neighbors], dtype=np.int32)
        b = np.asarray([p[1] for p in depth_neighbors], dtype=np.int32)
        if active_depth_mask is not None:
            keep = active_depth_mask[a] & active_depth_mask[b]
            a = a[keep]
            b = b[keep]
        if a.size > 0:
            w = np.sqrt(float(args.depth_smooth_weight))
            residuals.append(w * (log_rho[a] - log_rho[b]))

    if include_pose_prior and opt_frame_cids is not None and args.rot_prior_weight > 0.0:
        w = np.sqrt(float(args.rot_prior_weight))
        residuals.append(w * rvecs[opt_frame_cids].reshape(-1))

    if include_pose_prior and opt_frame_cids is not None and args.trans_prior_weight > 0.0:
        w = np.sqrt(float(args.trans_prior_weight))
        residuals.append(w * tvecs[opt_frame_cids].reshape(-1))

    if include_pose_prior and temporal_pairs is not None and len(temporal_pairs) > 0 and args.pose_smooth_weight > 0.0:
        a = np.asarray([p[0] for p in temporal_pairs], dtype=np.int32)
        b = np.asarray([p[1] for p in temporal_pairs], dtype=np.int32)
        w = np.sqrt(float(args.pose_smooth_weight))
        residuals.append(w * (rvecs[a] - rvecs[b]).reshape(-1))
        residuals.append(w * (tvecs[a] - tvecs[b]).reshape(-1))

    return np.concatenate([r.reshape(-1) for r in residuals]).astype(np.float64)


def compute_pixel_residuals_batched(
    prep: PreparedObs,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    log_rho: np.ndarray,
    args,
    batch: int = 200000,
) -> np.ndarray:
    n = prep.qx.shape[0]
    err = np.full(n, np.inf, dtype=np.float64)

    for s in range(0, n, batch):
        e = min(n, s + batch)
        idx = np.arange(s, e, dtype=np.int32)
        u, v, z = project_prepared_observations(
            prep=prep,
            obs_indices=idx,
            K=K,
            rvecs=rvecs,
            tvecs=tvecs,
            log_rho=log_rho,
            min_rho=args.min_rho,
            max_rho=args.max_rho,
        )
        ee = np.sqrt((u - prep.qx[idx]) ** 2 + (v - prep.qy[idx]) ** 2)
        ee[~np.isfinite(ee)] = np.inf
        ee[z <= args.z_min] = np.inf
        err[idx] = ee

    return err


def summarize_residuals(err: np.ndarray, obs_active_mask: np.ndarray) -> Dict:
    finite = np.isfinite(err)
    active = finite & obs_active_mask

    def stats(mask: np.ndarray) -> Dict:
        if not np.any(mask):
            return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
        vals = err[mask]
        return {
            "count": int(vals.shape[0]),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(np.max(vals)),
        }

    return {"all": stats(finite), "active": stats(active), "active_ratio": float(np.mean(obs_active_mask))}


def choose_active_indices(obs_active: np.ndarray, max_count: int) -> np.ndarray:
    idx = np.flatnonzero(obs_active).astype(np.int32)
    if max_count > 0 and idx.shape[0] > max_count:
        sel = np.linspace(0, idx.shape[0] - 1, max_count).round().astype(np.int64)
        idx = idx[sel]
    return idx


def build_depth_jac_sparsity(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    num_depths: int,
    depth_neighbors: List[Tuple[int, int]],
    args,
    active_depth_mask: Optional[np.ndarray],
) -> sparse.csr_matrix:
    n_obs = int(obs_indices.shape[0])
    rows = []
    cols = []

    # Pixel residual rows.
    didx = prep.depth_index[obs_indices]
    r0 = np.arange(n_obs, dtype=np.int64) * 2
    rows.extend(r0.tolist())
    cols.extend(didx.tolist())
    rows.extend((r0 + 1).tolist())
    cols.extend(didx.tolist())

    row_base = 2 * n_obs

    if args.depth_prior_weight > 0.0:
        if active_depth_mask is None:
            ids = np.arange(num_depths, dtype=np.int64)
        else:
            ids = np.flatnonzero(active_depth_mask).astype(np.int64)
        rows.extend((row_base + np.arange(ids.size, dtype=np.int64)).tolist())
        cols.extend(ids.tolist())
        row_base += ids.size

    if len(depth_neighbors) > 0 and args.depth_smooth_weight > 0.0:
        for a, b in depth_neighbors:
            if active_depth_mask is not None and not (active_depth_mask[a] and active_depth_mask[b]):
                continue
            rows.append(row_base)
            cols.append(int(a))
            rows.append(row_base)
            cols.append(int(b))
            row_base += 1

    data = np.ones(len(rows), dtype=np.float64)
    return sparse.coo_matrix((data, (rows, cols)), shape=(row_base, num_depths)).tocsr()


def get_obs_active_mask(prep: PreparedObs, pair_block_valid: np.ndarray, depth_valid: np.ndarray) -> np.ndarray:
    return pair_block_valid[prep.pair_block_index] & depth_valid[prep.depth_index]


def block_stats_from_residuals(
    err: np.ndarray,
    block_index: np.ndarray,
    block_valid: np.ndarray,
    obs_active: np.ndarray,
    num_blocks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.full(num_blocks, np.inf, dtype=np.float64)
    mean = np.full(num_blocks, np.inf, dtype=np.float64)
    count = np.zeros(num_blocks, dtype=np.int32)

    active = obs_active & np.isfinite(err)
    if not np.any(active):
        return median, mean, count

    for b in np.unique(block_index[active]):
        b = int(b)
        if b < 0 or b >= num_blocks or not block_valid[b]:
            continue
        vals = err[active & (block_index == b)]
        if vals.size == 0:
            continue
        median[b] = float(np.median(vals))
        mean[b] = float(np.mean(vals))
        count[b] = int(vals.size)

    return median, mean, count


def trim_worst_pair_blocks(
    err: np.ndarray,
    prep: PreparedObs,
    pair_block_valid: np.ndarray,
    depth_valid: np.ndarray,
    args,
    it: int,
) -> Tuple[np.ndarray, Dict]:
    if args.disable_block_trim or it < args.block_trim_start_iter or args.block_trim_fraction <= 0.0:
        return pair_block_valid, {"removed": 0, "reason": "disabled_or_not_started"}

    obs_active = get_obs_active_mask(prep, pair_block_valid, depth_valid)
    num_blocks = pair_block_valid.shape[0]
    med, mean, count = block_stats_from_residuals(err, prep.pair_block_index, pair_block_valid, obs_active, num_blocks)

    candidates = np.flatnonzero(pair_block_valid & np.isfinite(med) & (count >= args.block_trim_min_obs))
    if candidates.size == 0:
        return pair_block_valid, {"removed": 0, "reason": "no_candidates"}

    current_valid = int(np.count_nonzero(pair_block_valid))
    min_keep = max(1, int(np.ceil(num_blocks * args.min_pair_block_valid_ratio)))
    max_removable = max(0, current_valid - min_keep)
    if max_removable <= 0:
        return pair_block_valid, {"removed": 0, "reason": "min_valid_ratio_reached"}

    remove_n = int(np.ceil(candidates.size * args.block_trim_fraction))
    remove_n = max(1, remove_n)
    remove_n = min(remove_n, max_removable)

    order = candidates[np.argsort(med[candidates])[::-1]]
    remove = order[:remove_n]

    out = pair_block_valid.copy()
    out[remove] = False

    return out, {
        "removed": int(remove.size),
        "candidate_blocks": int(candidates.size),
        "valid_before": int(current_valid),
        "valid_after": int(np.count_nonzero(out)),
        "worst_removed_median_px": float(np.max(med[remove])) if remove.size > 0 else None,
        "best_removed_median_px": float(np.min(med[remove])) if remove.size > 0 else None,
    }


def trim_worst_depth_blocks(
    err: np.ndarray,
    prep: PreparedObs,
    pair_block_valid: np.ndarray,
    depth_valid: np.ndarray,
    args,
    it: int,
) -> Tuple[np.ndarray, Dict]:
    if args.depth_block_trim_fraction <= 0.0 or it < args.depth_block_trim_start_iter:
        return depth_valid, {"removed": 0, "reason": "disabled_or_not_started"}

    obs_active = get_obs_active_mask(prep, pair_block_valid, depth_valid)
    num_depths = depth_valid.shape[0]
    med, mean, count = block_stats_from_residuals(err, prep.depth_index, depth_valid, obs_active, num_depths)

    candidates = np.flatnonzero(depth_valid & np.isfinite(med) & (count >= args.depth_block_trim_min_obs))
    if candidates.size == 0:
        return depth_valid, {"removed": 0, "reason": "no_candidates"}

    current_valid = int(np.count_nonzero(depth_valid))
    min_keep = max(1, int(np.ceil(num_depths * args.min_depth_block_valid_ratio)))
    max_removable = max(0, current_valid - min_keep)
    if max_removable <= 0:
        return depth_valid, {"removed": 0, "reason": "min_valid_ratio_reached"}

    remove_n = int(np.ceil(candidates.size * args.depth_block_trim_fraction))
    remove_n = max(1, remove_n)
    remove_n = min(remove_n, max_removable)

    order = candidates[np.argsort(med[candidates])[::-1]]
    remove = order[:remove_n]

    out = depth_valid.copy()
    out[remove] = False

    return out, {
        "removed": int(remove.size),
        "candidate_blocks": int(candidates.size),
        "valid_before": int(current_valid),
        "valid_after": int(np.count_nonzero(out)),
        "worst_removed_median_px": float(np.max(med[remove])) if remove.size > 0 else None,
        "best_removed_median_px": float(np.min(med[remove])) if remove.size > 0 else None,
    }



def torch_rodrigues(rvecs: "torch.Tensor") -> "torch.Tensor":
    """Batched Rodrigues formula. rvecs: [F,3] -> R: [F,3,3]."""
    device = rvecs.device
    dtype = rvecs.dtype
    n = rvecs.shape[0]
    theta = torch.linalg.norm(rvecs, dim=1, keepdim=True).clamp_min(1e-12)
    k = rvecs / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = torch.zeros_like(kx)
    Kmat = torch.stack(
        [
            torch.stack([z, -kz, ky], dim=1),
            torch.stack([kz, z, -kx], dim=1),
            torch.stack([-ky, kx, z], dim=1),
        ],
        dim=1,
    )
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, 3, 3)
    st = torch.sin(theta).view(n, 1, 1)
    ct = torch.cos(theta).view(n, 1, 1)
    R = I + st * Kmat + (1.0 - ct) * torch.bmm(Kmat, Kmat)
    small = (theta.view(-1) < 1e-7).view(n, 1, 1)
    return torch.where(small, I, R)


def sample_log_rho_grid(
    log_rho_grid: "torch.Tensor",
    rho_frame_index: "torch.Tensor",
    px: "torch.Tensor",
    py: "torch.Tensor",
    width: int,
    height: int,
) -> "torch.Tensor":
    """Manual bilinear sampling from per-target-frame low-res log-rho grids."""
    # log_rho_grid: [T, Hc, Wc]
    gh = int(log_rho_grid.shape[1])
    gw = int(log_rho_grid.shape[2])
    x = px / max(float(width - 1), 1.0) * float(gw - 1)
    y = py / max(float(height - 1), 1.0) * float(gh - 1)

    x0 = torch.floor(x).long().clamp(0, gw - 1)
    y0 = torch.floor(y).long().clamp(0, gh - 1)
    x1 = (x0 + 1).clamp(0, gw - 1)
    y1 = (y0 + 1).clamp(0, gh - 1)

    wx = (x - x0.to(x.dtype)).clamp(0.0, 1.0)
    wy = (y - y0.to(y.dtype)).clamp(0.0, 1.0)

    f = rho_frame_index.long()
    v00 = log_rho_grid[f, y0, x0]
    v10 = log_rho_grid[f, y0, x1]
    v01 = log_rho_grid[f, y1, x0]
    v11 = log_rho_grid[f, y1, x1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def soft_l1_from_err2(err2: "torch.Tensor", f_scale: float) -> "torch.Tensor":
    f = float(max(f_scale, 1e-6))
    return 2.0 * (f * f) * (torch.sqrt(1.0 + err2 / (f * f)) - 1.0)


def huber_from_err(err: "torch.Tensor", delta: float) -> "torch.Tensor":
    d = float(max(delta, 1e-6))
    return torch.where(err <= d, 0.5 * err * err, d * (err - 0.5 * d))


def robust_reproj_loss(err2: "torch.Tensor", args) -> "torch.Tensor":
    err = torch.sqrt(err2.clamp_min(1e-12))
    loss_name = getattr(args, "torch_robust_loss", getattr(args, "robust_loss", "soft_l1"))
    f = float(getattr(args, "robust_f_scale", 2.0))
    if loss_name == "linear":
        return err2
    if loss_name == "huber":
        return huber_from_err(err, f)
    if loss_name == "cauchy":
        return (f * f) * torch.log1p(err2 / (f * f))
    # default soft_l1
    return soft_l1_from_err2(err2, f)


def make_device(args) -> "torch.device":
    req = str(getattr(args, "device", "auto"))
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def choose_training_indices(active: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(active).astype(np.int64)
    if max_count > 0 and idx.size > max_count:
        idx = rng.choice(idx, size=max_count, replace=False)
    return idx.astype(np.int64)


def summarize_np_error(err: np.ndarray, active: np.ndarray) -> Dict:
    finite = np.isfinite(err)
    mask = finite & active
    def stats(m):
        if not np.any(m):
            return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
        v = err[m]
        return {
            "count": int(v.size),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
        }
    return {"all": stats(finite), "active": stats(mask), "active_ratio": float(np.mean(active))}


def pair_block_stats(err: np.ndarray, pbidx: np.ndarray, active_obs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = int(weights.shape[0])
    med = np.full(n, np.inf, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int32)
    valid = active_obs & np.isfinite(err)
    for b in np.unique(pbidx[valid]):
        b = int(b)
        vals = err[valid & (pbidx == b)]
        if vals.size > 0:
            med[b] = float(np.median(vals))
            cnt[b] = int(vals.size)
    return med, cnt


def depth_block_stats(err: np.ndarray, didx: np.ndarray, active_obs: np.ndarray, n_depths: int) -> Tuple[np.ndarray, np.ndarray]:
    med = np.full(n_depths, np.inf, dtype=np.float64)
    cnt = np.zeros(n_depths, dtype=np.int32)
    valid = active_obs & np.isfinite(err)
    for d in np.unique(didx[valid]):
        d = int(d)
        vals = err[valid & (didx == d)]
        if vals.size > 0:
            med[d] = float(np.median(vals))
            cnt[d] = int(vals.size)
    return med, cnt


def fit_camera_like_model(
    obs: ObservationSet,
    depth_meta: List[Dict],
    pairs_meta: List[Tuple[int, int]],
    pair_block_meta: List[Dict],
    K: np.ndarray,
    args,
) -> CameraLikeFitResult:
    """GPU-friendly explicit parameter fitting.

    Learnable parameters:
      - frame-wise rvec/tvec
      - target-frame-wise low-resolution log-rho grid

    Supervision:
      - homography control point correspondences p_target -> q_ref

    Outlier handling:
      - valid_ratio/chosen_cost/source are used as initial soft weights
      - during training, worst residual pair-blocks are progressively downweighted
    """
    if obs.px.shape[0] < 16:
        raise RuntimeError(f"Too few pseudo observations: {obs.px.shape[0]}")

    device = make_device(args)
    dtype = torch.float32 if not getattr(args, "torch_float64", False) else torch.float64
    print(f"[TORCH] device={device}, dtype={dtype}")

    frames = sorted(set(obs.target_frame.astype(int).tolist()) | set(obs.ref_frame.astype(int).tolist()))
    frame_to_index, frames = build_frame_index(frames)
    if args.anchor_frame >= 0:
        if args.anchor_frame not in frame_to_index:
            raise ValueError(f"anchor_frame={args.anchor_frame} is not in GOP pair frames.")
        anchor_frame = int(args.anchor_frame)
    else:
        anchor_frame = int(frames[0])
    anchor_cid = frame_to_index[anchor_frame]

    prep = prepare_observations(obs, frame_to_index)
    init_rvecs, init_tvecs, _ = initialize_camera_like(prep, frames, frame_to_index, anchor_cid, K, args)

    # Target frames that own a depth/rho field.
    rho_grid_frames = sorted(set(int(m["target_idx"]) for m in depth_meta))
    rho_frame_to_gid = {f: i for i, f in enumerate(rho_grid_frames)}
    obs_rho_gid = np.asarray([rho_frame_to_gid[int(f)] for f in obs.target_frame], dtype=np.int64)

    # Low-resolution rho grid resolution.
    if args.rho_grid_w > 0 and args.rho_grid_h > 0:
        rho_grid_w = int(args.rho_grid_w)
        rho_grid_h = int(args.rho_grid_h)
    else:
        stride = max(1, int(args.rho_grid_stride))
        rho_grid_w = max(2, int(np.ceil(args.width / stride)))
        rho_grid_h = max(2, int(np.ceil(args.height / stride)))

    print(f"[TORCH] rho grid: frames={len(rho_grid_frames)}, size={rho_grid_w}x{rho_grid_h}")

    init_log_rho = float(np.log(np.clip(args.init_rho, args.min_rho, args.max_rho)))
    rvecs_param = torch.nn.Parameter(torch.tensor(init_rvecs, device=device, dtype=dtype))
    tvecs_param = torch.nn.Parameter(torch.tensor(init_tvecs, device=device, dtype=dtype))
    log_rho_grid = torch.nn.Parameter(
        torch.full((len(rho_grid_frames), rho_grid_h, rho_grid_w), init_log_rho, device=device, dtype=dtype)
    )

    # Constant tensors.
    target_cid = torch.tensor(prep.target_cid, device=device, dtype=torch.long)
    ref_cid = torch.tensor(prep.ref_cid, device=device, dtype=torch.long)
    obs_rho_gid_t = torch.tensor(obs_rho_gid, device=device, dtype=torch.long)
    pair_block_index_t = torch.tensor(obs.pair_block_index, device=device, dtype=torch.long)
    depth_index_t = torch.tensor(obs.depth_index, device=device, dtype=torch.long)
    px_t = torch.tensor(obs.px, device=device, dtype=dtype)
    py_t = torch.tensor(obs.py, device=device, dtype=dtype)
    qx_t = torch.tensor(obs.qx, device=device, dtype=dtype)
    qy_t = torch.tensor(obs.qy, device=device, dtype=dtype)
    rays_t = torch.tensor(prep.rays, device=device, dtype=dtype)

    base_w = prep.sqrt_weight.astype(np.float64) ** 2
    base_w = base_w / max(float(np.median(base_w[np.isfinite(base_w) & (base_w > 0)])), 1e-12)
    base_w = np.clip(base_w, 1e-4, 100.0).astype(np.float32)
    base_w_t = torch.tensor(base_w, device=device, dtype=dtype)

    K_t = torch.tensor(K, device=device, dtype=dtype)
    fx = K_t[0, 0]
    fy = K_t[1, 1]
    cx = K_t[0, 2]
    cy = K_t[1, 2]

    pair_block_weight = np.ones(len(pair_block_meta), dtype=np.float32)
    depth_valid = np.ones(len(depth_meta), dtype=bool)
    rng = np.random.default_rng(int(getattr(args, "seed", 1234)))

    optimizer = torch.optim.Adam(
        [
            {"params": [rvecs_param, tvecs_param], "lr": float(args.lr_pose)},
            {"params": [log_rho_grid], "lr": float(args.lr_depth)},
        ]
    )

    temporal_pairs = build_temporal_pairs(frames, frame_to_index)
    temporal_pairs_t = torch.tensor(temporal_pairs, device=device, dtype=torch.long) if temporal_pairs else None

    def current_pose():
        r = rvecs_param.clone()
        t = tvecs_param.clone()
        r[anchor_cid] = 0.0
        t[anchor_cid] = 0.0
        return r, t

    def project_indices(idx_t: "torch.Tensor", detach_depth: bool = False):
        r, t = current_pose()
        R = torch_rodrigues(r)
        tc = target_cid[idx_t]
        rc = ref_cid[idx_t]
        rg = obs_rho_gid_t[idx_t]
        grid = log_rho_grid.detach() if detach_depth else log_rho_grid
        log_rho = sample_log_rho_grid(grid, rg, px_t[idx_t], py_t[idx_t], args.width, args.height)
        log_rho = log_rho.clamp(float(np.log(args.min_rho)), float(np.log(args.max_rho)))
        rho = torch.exp(log_rho)
        X_t = rays_t[idx_t] / rho[:, None]
        X_w = torch.bmm(R[tc], X_t.unsqueeze(-1)).squeeze(-1) + t[tc]
        X_rel = X_w - t[rc]
        X_r = torch.bmm(R[rc].transpose(1, 2), X_rel.unsqueeze(-1)).squeeze(-1)
        z = X_r[:, 2]
        z_safe = torch.where(torch.abs(z) > 1e-9, z, torch.full_like(z, 1e-9))
        u = fx * (X_r[:, 0] / z_safe) + cx
        v = fy * (X_r[:, 1] / z_safe) + cy
        return u, v, z, rho

    def regularization_loss(depth_active: bool):
        r, t = current_pose()
        loss = torch.zeros((), device=device, dtype=dtype)
        if depth_active and not args.freeze_depth:
            if args.depth_tv_weight > 0.0:
                dx = log_rho_grid[:, :, 1:] - log_rho_grid[:, :, :-1]
                dy = log_rho_grid[:, 1:, :] - log_rho_grid[:, :-1, :]
                loss = loss + float(args.depth_tv_weight) * (dx.abs().mean() + dy.abs().mean())
            if args.depth_prior_weight > 0.0:
                loss = loss + float(args.depth_prior_weight) * torch.mean((log_rho_grid - init_log_rho) ** 2)
        if args.rot_prior_weight > 0.0:
            loss = loss + float(args.rot_prior_weight) * torch.mean(r ** 2)
        if args.trans_prior_weight > 0.0:
            loss = loss + float(args.trans_prior_weight) * torch.mean(t ** 2)
        if temporal_pairs_t is not None and args.pose_smooth_weight > 0.0:
            a = temporal_pairs_t[:, 0]
            b = temporal_pairs_t[:, 1]
            loss = loss + float(args.pose_smooth_weight) * (torch.mean((r[a] - r[b]) ** 2) + torch.mean((t[a] - t[b]) ** 2))
        return loss

    def loss_for_indices(idx_np: np.ndarray, step: int):
        idx_t = torch.tensor(idx_np, device=device, dtype=torch.long)
        depth_active = (step >= args.pose_only_steps) and (not args.freeze_depth)
        u, v, z, _rho = project_indices(idx_t, detach_depth=not depth_active)
        dx = u - qx_t[idx_t]
        dy = v - qy_t[idx_t]
        err2 = dx * dx + dy * dy
        pix_loss = robust_reproj_loss(err2, args)

        pbw = torch.tensor(pair_block_weight, device=device, dtype=dtype)[pair_block_index_t[idx_t]]
        dw_np = depth_valid.astype(np.float32)
        dw = torch.tensor(dw_np, device=device, dtype=dtype)[depth_index_t[idx_t]]
        w = base_w_t[idx_t] * pbw * dw
        if args.z_min > 0.0:
            z_bad = torch.relu(float(args.z_min) - z)
            pix_loss = pix_loss + float(args.z_penalty) * z_bad * z_bad
        loss = torch.sum(w * pix_loss) / (torch.sum(w) + 1e-12)
        loss = loss + regularization_loss(depth_active)
        return loss

    @torch.no_grad()
    def compute_all_residuals_np() -> np.ndarray:
        bs = int(max(1, args.torch_eval_batch_size))
        out = np.full(obs.qx.shape[0], np.inf, dtype=np.float64)
        for st in range(0, obs.qx.shape[0], bs):
            en = min(obs.qx.shape[0], st + bs)
            idx_t = torch.arange(st, en, device=device, dtype=torch.long)
            u, v, z, _ = project_indices(idx_t, detach_depth=False)
            err = torch.sqrt((u - qx_t[idx_t]) ** 2 + (v - qy_t[idx_t]) ** 2).detach().cpu().numpy()
            zz = z.detach().cpu().numpy()
            err[~np.isfinite(err)] = np.inf
            err[zz <= args.z_min] = np.inf
            out[st:en] = err
        return out

    def active_obs_mask_np() -> np.ndarray:
        return (pair_block_weight[obs.pair_block_index] > float(args.block_weight_min) + 1e-8) & depth_valid[obs.depth_index]

    def update_pair_block_weights(step: int, err: np.ndarray) -> Dict:
        if args.disable_block_trim or args.block_trim_fraction <= 0.0 or step < args.block_trim_start_step:
            return {"updated": False, "reason": "disabled_or_not_started"}
        active = active_obs_mask_np()
        med, cnt = pair_block_stats(err, obs.pair_block_index, active, pair_block_weight)
        candidates = np.flatnonzero((pair_block_weight > float(args.block_weight_min) + 1e-8) & np.isfinite(med) & (cnt >= args.block_trim_min_obs))
        if candidates.size == 0:
            return {"updated": False, "reason": "no_candidates"}
        current_valid = int(np.count_nonzero(pair_block_weight > float(args.block_weight_min) + 1e-8))
        min_keep = max(1, int(np.ceil(len(pair_block_weight) * float(args.min_pair_block_valid_ratio))))
        max_change = max(0, current_valid - min_keep)
        if max_change <= 0:
            return {"updated": False, "reason": "min_valid_ratio_reached"}
        n = int(np.ceil(candidates.size * float(args.block_trim_fraction)))
        n = max(1, min(n, max_change))
        order = candidates[np.argsort(med[candidates])[::-1]]
        bad = order[:n]
        before = pair_block_weight[bad].copy()
        pair_block_weight[bad] = np.maximum(pair_block_weight[bad] * float(args.block_downweight_factor), float(args.block_weight_min))
        return {
            "updated": True,
            "downweighted": int(bad.size),
            "valid_before": current_valid,
            "valid_after": int(np.count_nonzero(pair_block_weight > float(args.block_weight_min) + 1e-8)),
            "worst_median_px": float(np.max(med[bad])) if bad.size else None,
            "best_median_px": float(np.min(med[bad])) if bad.size else None,
            "old_weight_mean": float(np.mean(before)) if bad.size else None,
            "new_weight_mean": float(np.mean(pair_block_weight[bad])) if bad.size else None,
        }

    def update_depth_valid(step: int, err: np.ndarray) -> Dict:
        if args.depth_block_trim_fraction <= 0.0 or step < args.depth_block_trim_start_step:
            return {"updated": False, "reason": "disabled_or_not_started"}
        active = active_obs_mask_np()
        med, cnt = depth_block_stats(err, obs.depth_index, active, len(depth_meta))
        candidates = np.flatnonzero(depth_valid & np.isfinite(med) & (cnt >= args.depth_block_trim_min_obs))
        if candidates.size == 0:
            return {"updated": False, "reason": "no_candidates"}
        current_valid = int(np.count_nonzero(depth_valid))
        min_keep = max(1, int(np.ceil(len(depth_valid) * float(args.min_depth_block_valid_ratio))))
        max_remove = max(0, current_valid - min_keep)
        if max_remove <= 0:
            return {"updated": False, "reason": "min_valid_ratio_reached"}
        n = int(np.ceil(candidates.size * float(args.depth_block_trim_fraction)))
        n = max(1, min(n, max_remove))
        bad = candidates[np.argsort(med[candidates])[::-1]][:n]
        depth_valid[bad] = False
        return {
            "updated": True,
            "removed": int(bad.size),
            "valid_before": current_valid,
            "valid_after": int(np.count_nonzero(depth_valid)),
            "worst_median_px": float(np.max(med[bad])) if bad.size else None,
            "best_median_px": float(np.min(med[bad])) if bad.size else None,
        }

    report = {
        "backend": "torch_explicit_parameter_fitting",
        "device": str(device),
        "dtype": str(dtype),
        "num_observations": int(obs.qx.shape[0]),
        "num_pair_blocks": int(len(pair_block_meta)),
        "num_depth_blocks": int(len(depth_meta)),
        "num_frames": int(len(frames)),
        "anchor_frame": int(anchor_frame),
        "rho_grid_frames": [int(x) for x in rho_grid_frames],
        "rho_grid_shape": [int(rho_grid_h), int(rho_grid_w)],
        "iterations": [],
    }

    steps = int(args.torch_steps)
    log_every = max(1, int(args.log_every))
    update_every = max(1, int(args.block_weight_update_every))
    batch_size = int(max(1, args.torch_batch_size))

    for step in range(steps):
        active = active_obs_mask_np()
        idx = choose_training_indices(active, batch_size, rng)
        if idx.size < 8:
            raise RuntimeError("Too few active observations during torch fitting.")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_indices(idx, step)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([rvecs_param, tvecs_param, log_rho_grid], float(args.grad_clip))
        optimizer.step()
        with torch.no_grad():
            log_rho_grid.clamp_(float(np.log(args.min_rho)), float(np.log(args.max_rho)))
            rvecs_param[anchor_cid].zero_()
            tvecs_param[anchor_cid].zero_()

        if (step % log_every == 0) or (step == steps - 1):
            err = compute_all_residuals_np()
            summary = summarize_np_error(err, active_obs_mask_np())
            info = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "residual_summary": summary,
                "pair_block_weight_mean": float(np.mean(pair_block_weight)),
                "pair_block_weight_min": float(np.min(pair_block_weight)),
                "depth_valid_ratio": float(np.mean(depth_valid)),
            }
            report["iterations"].append(info)
            print("[TORCH FIT]")
            print(json.dumps(info, indent=2))

        if step > 0 and (step % update_every == 0):
            err = compute_all_residuals_np()
            pb_info = update_pair_block_weights(step, err)
            db_info = update_depth_valid(step, err)
            if pb_info.get("updated") or db_info.get("updated"):
                print("[TORCH BLOCK WEIGHT UPDATE]")
                print(json.dumps({"step": int(step), "pair_block": pb_info, "depth_block": db_info}, indent=2))

    final_err = compute_all_residuals_np()
    final_active = active_obs_mask_np()
    report["final_residual_summary"] = summarize_np_error(final_err, final_active)

    with torch.no_grad():
        r_final, t_final = current_pose()
        r_np = r_final.detach().cpu().numpy().astype(np.float64)
        t_np = t_final.detach().cpu().numpy().astype(np.float64)
        rho_grids_np = torch.exp(log_rho_grid.detach()).cpu().numpy().astype(np.float32)

        # Sample learned low-res rho at each original block center for compatibility with existing output JSON/render helpers.
        centers_x = np.asarray([float(m["block_x"]) + 0.5 * float(m["block_w"] - 1) for m in depth_meta], dtype=np.float32)
        centers_y = np.asarray([float(m["block_y"]) + 0.5 * float(m["block_h"] - 1) for m in depth_meta], dtype=np.float32)
        depth_rho_gid = np.asarray([rho_frame_to_gid[int(m["target_idx"])] for m in depth_meta], dtype=np.int64)
        cx_t = torch.tensor(centers_x, device=device, dtype=dtype)
        cy_t = torch.tensor(centers_y, device=device, dtype=dtype)
        dg_t = torch.tensor(depth_rho_gid, device=device, dtype=torch.long)
        log_depth = sample_log_rho_grid(log_rho_grid, dg_t, cx_t, cy_t, args.width, args.height)
        rho_per_depth = torch.exp(log_depth.clamp(float(np.log(args.min_rho)), float(np.log(args.max_rho)))).detach().cpu().numpy().astype(np.float64)

    report["rho_summary"] = {
        "sampled_depth_min": float(np.min(rho_per_depth)) if rho_per_depth.size else None,
        "sampled_depth_median": float(np.median(rho_per_depth)) if rho_per_depth.size else None,
        "sampled_depth_mean": float(np.mean(rho_per_depth)) if rho_per_depth.size else None,
        "sampled_depth_max": float(np.max(rho_per_depth)) if rho_per_depth.size else None,
        "grid_min": float(np.min(rho_grids_np)) if rho_grids_np.size else None,
        "grid_median": float(np.median(rho_grids_np)) if rho_grids_np.size else None,
        "grid_max": float(np.max(rho_grids_np)) if rho_grids_np.size else None,
    }

    return CameraLikeFitResult(
        K=K,
        frames=frames,
        frame_to_index=frame_to_index,
        anchor_frame=anchor_frame,
        anchor_cid=anchor_cid,
        rvecs=r_np,
        tvecs=t_np,
        log_rho=np.log(np.clip(rho_per_depth, args.min_rho, args.max_rho)),
        rho=rho_per_depth,
        pair_block_valid=pair_block_weight > float(args.block_weight_min) + 1e-8,
        depth_valid=depth_valid,
        residual_px=final_err,
        depth_neighbors=build_depth_neighbors(depth_meta),
        temporal_pairs=temporal_pairs,
        rho_grid_frames=[int(x) for x in rho_grid_frames],
        rho_grids=rho_grids_np,
        pair_block_weight=pair_block_weight.astype(np.float32),
        summary=report,
    )


# ============================================================
# Rho map / trim mask / rendering
# ============================================================

def neighbor_fill_rho_grid(grid: np.ndarray, valid: np.ndarray, fallback: float, max_iters: int = 2048) -> np.ndarray:
    out = grid.copy()
    val = valid.copy()
    h, w = out.shape

    if not np.any(val):
        out[:, :] = fallback
        return out

    for _ in range(max_iters):
        if np.all(val):
            break
        changed = False
        new_out = out.copy()
        new_val = val.copy()

        ys, xs = np.where(~val)
        for y, x in zip(ys.tolist(), xs.tolist()):
            vals = []
            for yy, xx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= yy < h and 0 <= xx < w and val[yy, xx]:
                    vals.append(float(out[yy, xx]))
            if vals:
                new_out[y, x] = float(np.mean(vals))
                new_val[y, x] = True
                changed = True
        out = new_out
        val = new_val
        if not changed:
            break

    out[~val] = fallback
    return out


def build_rho_image_for_target(
    target_idx: int,
    width: int,
    height: int,
    depth_meta: List[Dict],
    rho: np.ndarray,
    depth_valid: Optional[np.ndarray] = None,
    fill_invalid: bool = True,
    rho_grid_frames: Optional[List[int]] = None,
    rho_grids: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Prefer the learned low-res continuous rho grid when available.
    if rho_grid_frames is not None and rho_grids is not None and int(target_idx) in set(int(x) for x in rho_grid_frames):
        gi = [int(x) for x in rho_grid_frames].index(int(target_idx))
        grid = np.asarray(rho_grids[gi], dtype=np.float32)
        img = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(img, 1e-12, np.inf).astype(np.float32)

    # Fill at block-grid level first, then expand to image. This prevents a rejected block
    # from simply receiving an arbitrary median without considering neighbors.
    metas = [m for m in depth_meta if int(m["target_idx"]) == int(target_idx)]
    if not metas:
        fill = float(np.median(rho)) if rho.size > 0 else 1.0
        return np.full((height, width), fill, dtype=np.float32)

    max_by = max(int(m["block_y_idx"]) for m in metas)
    max_bx = max(int(m["block_x_idx"]) for m in metas)
    grid = np.full((max_by + 1, max_bx + 1), np.nan, dtype=np.float64)
    valid = np.zeros_like(grid, dtype=bool)

    for i, m in enumerate(depth_meta):
        if int(m["target_idx"]) != int(target_idx):
            continue
        by = int(m["block_y_idx"])
        bx = int(m["block_x_idx"])
        is_valid = True if depth_valid is None else bool(depth_valid[i])
        if is_valid:
            grid[by, bx] = float(rho[i])
            valid[by, bx] = True

    if np.any(valid):
        fallback = float(np.median(grid[valid]))
    else:
        if depth_valid is not None and np.any(depth_valid):
            fallback = float(np.median(rho[depth_valid]))
        else:
            fallback = float(np.median(rho)) if rho.size > 0 else 1.0

    if fill_invalid:
        grid_filled = neighbor_fill_rho_grid(grid, valid, fallback)
    else:
        grid_filled = grid.copy()
        grid_filled[~valid] = fallback

    rho_img = np.full((height, width), fallback, dtype=np.float32)
    for m in metas:
        by = int(m["block_y_idx"])
        bx = int(m["block_x_idx"])
        x0 = int(m["block_x"])
        y0 = int(m["block_y"])
        x1 = min(width, x0 + int(m["block_w"]))
        y1 = min(height, y0 + int(m["block_h"]))
        rho_img[y0:y1, x0:x1] = float(grid_filled[by, bx])

    rho_img = np.clip(rho_img, 1e-12, np.inf)
    return rho_img


def save_rho_maps(output_dir: str, width: int, height: int, depth_meta: List[Dict], fit: CameraLikeFitResult):
    rho_dir = os.path.join(output_dir, "rho_maps")
    ensure_dir(rho_dir)
    target_frames = sorted(set(int(m["target_idx"]) for m in depth_meta))

    for t in target_frames:
        rho_img = build_rho_image_for_target(t, width, height, depth_meta, fit.rho, fit.depth_valid, fill_invalid=True, rho_grid_frames=fit.rho_grid_frames, rho_grids=fit.rho_grids)
        np.save(os.path.join(rho_dir, f"rho_t{t:03d}.npy"), rho_img.astype(np.float32))

        vals = rho_img[np.isfinite(rho_img)]
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        if abs(hi - lo) < 1e-12:
            out = np.full((height, width), 128, dtype=np.uint8)
        else:
            out = np.clip((rho_img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(rho_dir, f"rho_t{t:03d}.png"), color)


def save_trim_masks(
    output_dir: str,
    width: int,
    height: int,
    depth_meta: List[Dict],
    pair_block_meta: List[Dict],
    fit: CameraLikeFitResult,
):
    mask_dir = os.path.join(output_dir, "trim_masks")
    ensure_dir(mask_dir)

    # Depth block valid mask per target frame. 255 means valid, 0 means rejected.
    for t in sorted(set(int(m["target_idx"]) for m in depth_meta)):
        img = np.zeros((height, width), dtype=np.uint8)
        for i, m in enumerate(depth_meta):
            if int(m["target_idx"]) != t:
                continue
            x0 = int(m["block_x"])
            y0 = int(m["block_y"])
            x1 = min(width, x0 + int(m["block_w"]))
            y1 = min(height, y0 + int(m["block_h"]))
            img[y0:y1, x0:x1] = 255 if fit.depth_valid[i] else 0
        cv2.imwrite(os.path.join(mask_dir, f"depth_block_valid_t{t:03d}.png"), img)

    # Pair-block valid mask per target-ref pair. 255 means valid, 0 means rejected.
    pair_keys = sorted(set((int(m["target_idx"]), int(m["ref_idx"])) for m in pair_block_meta))
    for t, r in pair_keys:
        img = np.zeros((height, width), dtype=np.uint8)
        for i, m in enumerate(pair_block_meta):
            if int(m["target_idx"]) != t or int(m["ref_idx"]) != r:
                continue
            x0 = int(m["block_x"])
            y0 = int(m["block_y"])
            x1 = min(width, x0 + int(m["block_w"]))
            y1 = min(height, y0 + int(m["block_h"]))
            img[y0:y1, x0:x1] = 255 if fit.pair_block_valid[i] else 0
        cv2.imwrite(os.path.join(mask_dir, f"pair_block_valid_t{t:03d}_r{r:03d}.png"), img)


def camera_like_maps_for_pair_batched(
    target_idx: int,
    ref_idx: int,
    width: int,
    height: int,
    fit: CameraLikeFitResult,
    depth_meta: List[Dict],
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_idx not in fit.frame_to_index:
        raise ValueError(f"target_idx={target_idx} not present in fitted frame set.")
    if ref_idx not in fit.frame_to_index:
        raise ValueError(f"ref_idx={ref_idx} not present in fitted frame set.")

    tcid = fit.frame_to_index[int(target_idx)]
    rcid = fit.frame_to_index[int(ref_idx)]

    K = fit.K
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    rho_img = build_rho_image_for_target(target_idx, width, height, depth_meta, fit.rho, fit.depth_valid, fill_invalid=True, rho_grid_frames=fit.rho_grid_frames, rho_grids=fit.rho_grids).astype(np.float64)

    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    Rs = all_rotation_matrices(fit.rvecs)
    R_t = Rs[tcid]
    R_r = Rs[rcid]
    t_t = fit.tvecs[tcid]
    t_r = fit.tvecs[rcid]

    row_batch = max(1, int(args.render_row_batch))
    xs_full = np.arange(width, dtype=np.float64)

    for y0 in range(0, height, row_batch):
        y1 = min(height, y0 + row_batch)
        ys = np.arange(y0, y1, dtype=np.float64)
        xs, yy = np.meshgrid(xs_full, ys)

        ray_x = (xs - cx) / fx
        ray_y = (yy - cy) / fy
        rays = np.stack([ray_x.reshape(-1), ray_y.reshape(-1), np.ones((y1 - y0) * width, dtype=np.float64)], axis=1)
        rho_flat = rho_img[y0:y1, :].reshape(-1)
        X_t = rays / rho_flat[:, None]

        X_w = X_t @ R_t.T + t_t[None, :]
        X_r = (X_w - t_r[None, :]) @ R_r

        z = X_r[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
        mx = fx * (X_r[:, 0] / z_safe) + cx
        my = fy * (X_r[:, 1] / z_safe) + cy

        valid = (
            (z > args.z_min)
            & np.isfinite(mx)
            & np.isfinite(my)
            & (mx >= 0.0)
            & (mx <= width - 1.0)
            & (my >= 0.0)
            & (my <= height - 1.0)
        )

        map_x[y0:y1, :] = mx.reshape(y1 - y0, width).astype(np.float32)
        map_y[y0:y1, :] = my.reshape(y1 - y0, width).astype(np.float32)
        valid_all[y0:y1, :] = valid.reshape(y1 - y0, width)

    return map_x, map_y, valid_all


def render_camera_like_pair(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    target_idx: int,
    ref_idx: int,
    fit: CameraLikeFitResult,
    depth_meta: List[Dict],
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    map_x, map_y, valid = camera_like_maps_for_pair_batched(target_idx, ref_idx, args.width, args.height, fit, depth_meta, args)
    pred = remap_ref(ref_y, map_x, map_y)
    return pred, valid


# ============================================================
# GOP main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)
    parser.add_argument("--gop-start", type=int, required=True)
    parser.add_argument("--gop-size", type=int, required=True)

    parser.add_argument("--pair-mode", choices=["dyadic", "all"], default="dyadic")
    parser.add_argument("--pairs", type=str, default="")
    parser.add_argument("--pairs-file", type=str, default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--skip-failed-pairs", action="store_true")
    parser.add_argument("--output-dir", required=True)

    # Reuse existing pair results.
    parser.add_argument("--reuse-pair-results", action="store_true")
    parser.add_argument("--pair-results-dir", type=str, default="")

    # Hierarchical block homography options. Defaults match the supplied reference code.
    parser.add_argument("--start-block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--min-block-size", type=int, default=32)
    parser.add_argument("--block-margin", type=int, default=32)
    parser.add_argument("--disable-edge-anchored-fit", action="store_true")
    parser.add_argument("--no-pair-yuv-png", action="store_true")

    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--ransac-thresh", type=float, default=2.0)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--parent-match-gate", type=float, default=30.0)
    parser.add_argument("--root-fallback", choices=["identity", "global_translation", "global_homography"], default="global_translation")
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--disable-cost-gate", action="store_true")

    # CP pseudo observation options.
    parser.add_argument("--cp-pattern", choices=["center", "center4", "grid3"], default="center")
    parser.add_argument("--cp-inset", type=float, default=2.0)
    parser.add_argument("--obs-exclude-rejected", action="store_true")
    parser.add_argument("--obs-min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--obs-min-inliers", type=int, default=8)

    # Fixed deterministic K.
    parser.add_argument("--fx", type=float, default=-1.0)
    parser.add_argument("--fy", type=float, default=-1.0)
    parser.add_argument("--cx", type=float, default=-1.0)
    parser.add_argument("--cy", type=float, default=-1.0)
    parser.add_argument("--f-scale", type=float, default=1.0)

    # Camera-like fitting options.
    parser.add_argument("--anchor-frame", type=int, default=-1)
    parser.add_argument("--init-rho", type=float, default=1.0)
    parser.add_argument("--min-rho", type=float, default=0.5)
    parser.add_argument("--max-rho", type=float, default=2.0)
    parser.add_argument("--freeze-depth", action="store_true")

    # PyTorch explicit parameter fitting options.
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--torch-float64", action="store_true")
    parser.add_argument("--torch-steps", type=int, default=2000)
    parser.add_argument("--torch-batch-size", type=int, default=65536)
    parser.add_argument("--torch-eval-batch-size", type=int, default=262144)
    parser.add_argument("--pose-only-steps", type=int, default=300)
    parser.add_argument("--lr-pose", type=float, default=1e-3)
    parser.add_argument("--lr-depth", type=float, default=5e-3)
    parser.add_argument("--rho-grid-stride", type=int, default=128)
    parser.add_argument("--rho-grid-w", type=int, default=0)
    parser.add_argument("--rho-grid-h", type=int, default=0)
    parser.add_argument("--depth-tv-weight", type=float, default=0.1)
    parser.add_argument("--torch-robust-loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
    parser.add_argument("--block-weight-update-every", type=int, default=200)
    parser.add_argument("--block-trim-start-step", type=int, default=500)
    parser.add_argument("--block-downweight-factor", type=float, default=0.2)
    parser.add_argument("--block-weight-min", type=float, default=0.02)
    parser.add_argument("--valid-ratio-weight-gamma", type=float, default=2.0)
    parser.add_argument("--valid-ratio-soft-min", type=float, default=0.20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--alt-iters", type=int, default=8)
    parser.add_argument("--pose-only-iters-before-depth", type=int, default=1)
    parser.add_argument("--depth-max-nfev", type=int, default=40)
    parser.add_argument("--pose-max-nfev", type=int, default=60)
    parser.add_argument("--max-active-observations-per-iter", type=int, default=120000)

    parser.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--robust-f-scale", type=float, default=2.0)

    # Residual-based block trimming.
    parser.add_argument("--disable-block-trim", action="store_true")
    parser.add_argument("--block-trim-fraction", type=float, default=0.05)
    parser.add_argument("--block-trim-start-iter", type=int, default=2)
    parser.add_argument("--block-trim-min-obs", type=int, default=1)
    parser.add_argument("--min-pair-block-valid-ratio", type=float, default=0.70)

    parser.add_argument("--depth-block-trim-fraction", type=float, default=0.0)
    parser.add_argument("--depth-block-trim-start-iter", type=int, default=3)
    parser.add_argument("--depth-block-trim-min-obs", type=int, default=2)
    parser.add_argument("--min-depth-block-valid-ratio", type=float, default=0.80)

    parser.add_argument("--depth-prior-weight", type=float, default=0.1)
    parser.add_argument("--depth-smooth-weight", type=float, default=1.0)
    parser.add_argument("--rot-prior-weight", type=float, default=1e-5)
    parser.add_argument("--trans-prior-weight", type=float, default=1e-5)
    parser.add_argument("--pose-smooth-weight", type=float, default=1e-3)

    parser.add_argument("--z-min", type=float, default=1e-4)
    parser.add_argument("--z-penalty", type=float, default=100.0)
    parser.add_argument("--residual-clip-px", type=float, default=200.0)
    parser.add_argument("--residual-batch-size", type=int, default=200000)
    parser.add_argument("--render-row-batch", type=int, default=64)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--verbose-opt", action="store_true")

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    pairs = get_gop_pairs(args)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise RuntimeError("No GOP pairs were generated.")

    print("[INFO] GOP pairs:")
    for p in pairs:
        print(f"  target={p[0]} ref={p[1]}")

    K = make_fixed_K(args.width, args.height, args)
    print("[INFO] fixed K:")
    print(K)

    frame_cache: Dict[int, np.ndarray] = {}

    def get_frame_y(frame_idx: int) -> np.ndarray:
        if frame_idx not in frame_cache:
            frame_cache[frame_idx] = read_y_frame(args.input, args.width, args.height, args.bitdepth, frame_idx).y
        return frame_cache[frame_idx]

    # ------------------------------------------------------------
    # 1) Pair-wise hierarchical block homography or reuse.
    # ------------------------------------------------------------
    failed_pairs = []

    if args.reuse_pair_results:
        pair_results, failed_pairs = load_existing_pair_results(args, pairs)
    else:
        pair_results: List[PairHResult] = []
        for target_idx, ref_idx in pairs:
            pair_dir = os.path.join(args.output_dir, f"pair_t{target_idx:03d}_r{ref_idx:03d}")
            print(f"[PAIR] target={target_idx}, ref={ref_idx}")
            try:
                target_y = get_frame_y(target_idx)
                ref_y = get_frame_y(ref_idx)
                pr = run_hierarchical_pair(target_y, ref_y, target_idx, ref_idx, args.bitdepth, args, pair_dir)
                pair_results.append(pr)
                print("[PAIR DIRECT COST]")
                print(json.dumps(pr.direct_cost, indent=2))
            except Exception as e:
                failed_pairs.append({"target_idx": int(target_idx), "ref_idx": int(ref_idx), "error": str(e)})
                print(f"[WARN] pair failed: target={target_idx}, ref={ref_idx}, error={e}")
                if not args.skip_failed_pairs:
                    raise

    if not pair_results:
        raise RuntimeError("No pair results are available. No camera-like fitting is possible.")

    # ------------------------------------------------------------
    # 2) Block-H -> CP pseudo observations.
    # ------------------------------------------------------------
    obs, depth_key_to_index, depth_meta, pairs_meta, pair_block_meta = build_observations_from_pairs(
        pair_results=pair_results,
        width=args.width,
        height=args.height,
        K=K,
        args=args,
    )

    print(f"[INFO] pseudo observations = {obs.px.shape[0]}")
    print(f"[INFO] pair blocks         = {len(pair_block_meta)}")
    print(f"[INFO] depth variables     = {len(depth_meta)}")

    if obs.px.shape[0] < 16:
        raise RuntimeError("Too few pseudo observations after filtering.")

    # ------------------------------------------------------------
    # 3) GOP-shared camera-like fitting.
    # ------------------------------------------------------------
    fit = fit_camera_like_model(obs, depth_meta, pairs_meta, pair_block_meta, K, args)

    save_rho_maps(args.output_dir, args.width, args.height, depth_meta, fit)
    save_trim_masks(args.output_dir, args.width, args.height, depth_meta, pair_block_meta, fit)

    # ------------------------------------------------------------
    # 4) Render fitted camera-like prediction for each pair.
    # ------------------------------------------------------------
    camlike_costs = []
    camlike_dir = os.path.join(args.output_dir, "camlike_pairs")
    ensure_dir(camlike_dir)

    if not args.skip_render:
        for pr in pair_results:
            target_y = get_frame_y(pr.target_idx)
            ref_y = get_frame_y(pr.ref_idx)
            pred, valid = render_camera_like_pair(target_y, ref_y, pr.target_idx, pr.ref_idx, fit, depth_meta, args)
            cost = calc_cost(target_y, pred, valid, args.bitdepth)
            tag = f"t{pr.target_idx:03d}_r{pr.ref_idx:03d}"
            yuv_path = os.path.join(camlike_dir, f"pred_camlike_{tag}.yuv")
            png_path = os.path.join(camlike_dir, f"pred_camlike_{tag}.png")
            diff_path = os.path.join(camlike_dir, f"diff_camlike_{tag}.png")
            write_single_yuv420(yuv_path, pred, args.width, args.height, args.bitdepth)
            save_gray_png(png_path, pred, args.bitdepth)
            save_diff_png(diff_path, target_y, pred, valid)
            camlike_costs.append(
                {
                    "target_idx": int(pr.target_idx),
                    "ref_idx": int(pr.ref_idx),
                    "direct_block_homography_cost": pr.direct_cost,
                    "camlike_cost": cost,
                    "pred_yuv": yuv_path,
                }
            )
            print("[CAMLIKE PAIR COST]")
            print(json.dumps(camlike_costs[-1], indent=2))

    # ------------------------------------------------------------
    # 5) Save final JSON.
    # ------------------------------------------------------------
    pose_json = []
    for f in fit.frames:
        cid = fit.frame_to_index[f]
        pose_json.append(
            {
                "frame_idx": int(f),
                "compact_id": int(cid),
                "is_anchor": bool(cid == fit.anchor_cid),
                "rvec": fit.rvecs[cid].tolist(),
                "R": rodrigues(fit.rvecs[cid]).tolist(),
                "t": fit.tvecs[cid].tolist(),
            }
        )

    depth_json = []
    for i, m in enumerate(depth_meta):
        mm = dict(m)
        mm["depth_index"] = int(i)
        mm["rho"] = float(fit.rho[i])
        mm["log_rho"] = float(fit.log_rho[i])
        mm["valid"] = bool(fit.depth_valid[i])
        depth_json.append(mm)

    pair_block_json = []
    for i, m in enumerate(pair_block_meta):
        mm = dict(m)
        mm["valid"] = bool(fit.pair_block_valid[i])
        pair_block_json.append(mm)

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "gop": {
            "gop_start": int(args.gop_start),
            "gop_size": int(args.gop_size),
            "pair_mode": args.pair_mode,
            "requested_pairs": [(int(a), int(b)) for a, b in pairs],
            "used_pairs": [(int(pr.target_idx), int(pr.ref_idx)) for pr in pair_results],
            "failed_pairs": failed_pairs,
            "reuse_pair_results": bool(args.reuse_pair_results),
            "pair_results_dir": args.pair_results_dir if args.pair_results_dir else args.output_dir,
        },
        "method": {
            "description": (
                "GOP-shared camera-like fitting from block homography pseudo-GT. "
                "K is fixed. Frame-wise R,t and target-frame/block-wise scalar inverse depth rho are fitted. "
                "Fitting alternates pose update with fixed rho, then depth update with fixed pose. "
                "Block trimming removes worst residual pair-blocks after fitting rounds."
            ),
            "coordinate": "target pixel -> ref pixel",
            "pose_convention": "R_i,t_i maps camera_i coordinates to GOP-local world coordinates.",
            "point_model": "X_target_cam = [(x-cx)/fx, (y-cy)/fy, 1] / rho[target_frame, block]",
        },
        "K": fit.K.tolist(),
        "homography_options": {
            "start_block_size": int(args.start_block_size),
            "levels": int(args.levels),
            "min_block_size": int(args.min_block_size),
            "block_margin": int(args.block_margin),
            "edge_anchored_fit": bool(not args.disable_edge_anchored_fit),
            "root_fallback": args.root_fallback,
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "clahe": bool(not args.no_clahe),
            "ransac_thresh": float(args.ransac_thresh),
            "min_matches": int(args.min_matches),
            "min_inliers": int(args.min_inliers),
            "parent_match_gate": float(args.parent_match_gate),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
            "min_gain": float(args.min_gain),
            "cost_gate": bool(not args.disable_cost_gate),
        },
        "observation_options": {
            "cp_pattern": args.cp_pattern,
            "cp_inset": float(args.cp_inset),
            "obs_exclude_rejected": bool(args.obs_exclude_rejected),
            "obs_min_valid_ratio": float(args.obs_min_valid_ratio),
            "obs_min_inliers": int(args.obs_min_inliers),
            "num_observations": int(obs.px.shape[0]),
            "num_pair_blocks": int(len(pair_block_meta)),
            "num_depth_variables": int(len(depth_meta)),
        },
        "fit_options": {
            "anchor_frame": int(fit.anchor_frame),
            "init_rho": float(args.init_rho),
            "min_rho": float(args.min_rho),
            "max_rho": float(args.max_rho),
            "freeze_depth": bool(args.freeze_depth),
            "alt_iters": int(args.alt_iters),
            "pose_only_iters_before_depth": int(args.pose_only_iters_before_depth),
            "pose_max_nfev": int(args.pose_max_nfev),
            "depth_max_nfev": int(args.depth_max_nfev),
            "max_active_observations_per_iter": int(args.max_active_observations_per_iter),
            "robust_loss": args.robust_loss,
            "robust_f_scale": float(args.robust_f_scale),
            "block_trim_fraction": float(args.block_trim_fraction),
            "block_trim_start_iter": int(args.block_trim_start_iter),
            "min_pair_block_valid_ratio": float(args.min_pair_block_valid_ratio),
            "depth_block_trim_fraction": float(args.depth_block_trim_fraction),
            "depth_block_trim_start_iter": int(args.depth_block_trim_start_iter),
            "min_depth_block_valid_ratio": float(args.min_depth_block_valid_ratio),
            "depth_prior_weight": float(args.depth_prior_weight),
            "depth_smooth_weight": float(args.depth_smooth_weight),
            "pose_smooth_weight": float(args.pose_smooth_weight),
        },
        "fit_summary": fit.summary,
        "poses": pose_json,
        "depth_blocks": depth_json,
        "pair_blocks": pair_block_json,
        "pair_costs": camlike_costs,
        "outputs": {
            "pair_homography_dirs": [os.path.join(args.output_dir, f"pair_t{pr.target_idx:03d}_r{pr.ref_idx:03d}") for pr in pair_results],
            "camlike_pair_dir": camlike_dir,
            "rho_map_dir": os.path.join(args.output_dir, "rho_maps"),
            "trim_mask_dir": os.path.join(args.output_dir, "trim_masks"),
        },
        "torch_fit": {
            "rho_grid_frames": fit.rho_grid_frames,
            "rho_grid_shape": list(fit.rho_grids.shape) if fit.rho_grids is not None else None,
            "pair_block_weight_min": float(np.min(fit.pair_block_weight)) if fit.pair_block_weight is not None else None,
            "pair_block_weight_mean": float(np.mean(fit.pair_block_weight)) if fit.pair_block_weight is not None else None,
        },
    }

    result_path = os.path.join(args.output_dir, "gop_camera_like_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[DONE]")
    print(f"  result JSON: {result_path}")
    print(f"  camlike dir: {camlike_dir}")
    print(f"  rho maps   : {os.path.join(args.output_dir, 'rho_maps')}")
    print(f"  trim masks : {os.path.join(args.output_dir, 'trim_masks')}")


if __name__ == "__main__":
    main()



















python gop_camera_like_torch_lowres_rho.py \
  --input input.yuv \
  --width 1920 --height 1080 --bitdepth 10 \
  --gop-start 0 --gop-size 10 \
  --pair-mode dyadic \
  --output-dir gop0_9_torchfit \
  --reuse-pair-results \
  --pair-results-dir 기존_pair_output_dir \
  --obs-exclude-rejected \
  --cp-pattern center4 \
  --device cuda \
  --torch-steps 2000 \
  --pose-only-steps 300 \
  --rho-grid-stride 128 \
  --lr-pose 1e-3 \
  --lr-depth 5e-3 \
  --depth-tv-weight 0.1 \
  --depth-prior-weight 0.1 \
  --valid-ratio-weight-gamma 2.0 \
  --block-weight-update-every 200 \
  --block-trim-start-step 500 \
  --block-trim-fraction 0.05 \
  --block-downweight-factor 0.2
