#!/usr/bin/env python3
# hierarchical_block_homography.py
#
# Hierarchical block-wise homography fitting experiment.
#
# Coordinate convention:
#   target pixel x_t -> ref pixel x_r
#
# Main features:
#   1. Edge-anchored fitting:
#      For right/bottom partial blocks, the output region remains non-overlapping,
#      but the fitting region is shifted inward so that local homography is
#      estimated from a full block_size x block_size window whenever possible.
#
#   2. Root-level CP sweep refinement:
#      At the largest block level, refine the RANSAC homography by moving
#      4 destination control points with coordinate-descent photometric search.
#
#   3. Strong parent propagation:
#      Lower levels try local homography, but accept it only if it improves
#      over inherited parent H by enough absolute/relative MAE gain.
#
# Outputs:
#   target_pair.yuv
#   ref_pair.yuv
#   pred_levelXX_blockYYY.yuv
#   pred_hier_block_homography.yuv
#   result.json

import argparse
import json
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
class Tile:
    # Non-overlap output region.
    out_x0: int
    out_y0: int
    out_x1: int
    out_y1: int

    # Fitting region. For edge blocks this may be shifted inward.
    fit_x0: int
    fit_y0: int
    fit_x1: int
    fit_y1: int

    out_w: int
    out_h: int
    fit_w: int
    fit_h: int


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


def make_tiles(
    width: int,
    height: int,
    block_size: int,
    edge_anchored_fit: bool,
) -> List[List[Tile]]:
    """
    Create non-overlap output tiles and fitting tiles.

    If edge_anchored_fit is True:
        For partial right/bottom edge blocks, the fitting region is shifted
        inward so that fit_w/fit_h becomes block_size whenever possible.

    Example:
        width=1856, block_size=256
        last output tile: out_x0=1792, out_x1=1856, out_w=64
        last fitting tile: fit_x0=1600, fit_x1=1856, fit_w=256
    """
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

            tile = Tile(
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

            row.append(tile)

        tiles.append(row)

    return tiles


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


def eval_H_on_tile_region(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    H: np.ndarray,
    tile: Tile,
    region: str,
    min_valid_ratio: float,
) -> Tuple[float, float]:
    if region == "out":
        return eval_block_photometric_cost(
            target_y, ref_y, H,
            tile.out_x0, tile.out_y0, tile.out_w, tile.out_h,
            min_valid_ratio,
        )

    if region == "fit":
        return eval_block_photometric_cost(
            target_y, ref_y, H,
            tile.fit_x0, tile.fit_y0, tile.fit_w, tile.fit_h,
            min_valid_ratio,
        )

    if region == "both":
        cost_fit, vr_fit = eval_block_photometric_cost(
            target_y, ref_y, H,
            tile.fit_x0, tile.fit_y0, tile.fit_w, tile.fit_h,
            min_valid_ratio,
        )
        cost_out, vr_out = eval_block_photometric_cost(
            target_y, ref_y, H,
            tile.out_x0, tile.out_y0, tile.out_w, tile.out_h,
            min_valid_ratio,
        )

        vals = []
        if np.isfinite(cost_fit):
            vals.append(cost_fit)
        if np.isfinite(cost_out):
            vals.append(cost_out)

        if not vals:
            return float("inf"), min(vr_fit, vr_out)

        return float(np.mean(vals)), min(vr_fit, vr_out)

    raise ValueError(f"Unknown CP sweep cost region: {region}")


# ============================================================
# CP sweep refinement
# ============================================================

def region_corners(x0: int, y0: int, w: int, h: int) -> np.ndarray:
    x1 = x0 + w - 1
    y1 = y0 + h - 1

    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.float32,
    )


def H_from_control_points(src_cps: np.ndarray, dst_cps: np.ndarray) -> Optional[np.ndarray]:
    try:
        H = cv2.getPerspectiveTransform(
            src_cps.astype(np.float32),
            dst_cps.astype(np.float32),
        )
    except cv2.error:
        return None

    if H is None:
        return None

    return normalize_homography(H)


def refine_homography_cp_sweep(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    H_init: np.ndarray,
    tile: Tile,
    args,
) -> Tuple[np.ndarray, Dict]:
    """
    Coordinate-descent CP sweep.

    Source CPs:
        4 corners of fitting tile.

    Destination CPs:
        H_init(source CPs), then each CP is locally perturbed.

    Cost:
        MAE over args.cp_sweep_cost_region.
        Default: fit region.
    """
    if args.disable_cp_sweep:
        return H_init, {
            "enabled": False,
            "reason": "disabled",
            "initial_cost": None,
            "final_cost": None,
            "improvement": 0.0,
            "num_evals": 0,
        }

    if args.cp_sweep_steps <= 0:
        return H_init, {
            "enabled": False,
            "reason": "steps_le_0",
            "initial_cost": None,
            "final_cost": None,
            "improvement": 0.0,
            "num_evals": 0,
        }

    src_cps = region_corners(tile.fit_x0, tile.fit_y0, tile.fit_w, tile.fit_h)
    dst_cps = apply_homography_points(H_init, src_cps).astype(np.float32)

    H_best = H_from_control_points(src_cps, dst_cps)
    if H_best is None:
        return H_init, {
            "enabled": True,
            "reason": "initial_cp_to_H_failed",
            "initial_cost": None,
            "final_cost": None,
            "improvement": 0.0,
            "num_evals": 0,
        }

    best_cost, best_vr = eval_H_on_tile_region(
        target_y=target_y,
        ref_y=ref_y,
        H=H_best,
        tile=tile,
        region=args.cp_sweep_cost_region,
        min_valid_ratio=args.min_block_valid_ratio,
    )

    initial_cost = best_cost
    num_evals = 1

    if not np.isfinite(best_cost):
        return H_init, {
            "enabled": True,
            "reason": "initial_cost_invalid",
            "initial_cost": None,
            "final_cost": None,
            "improvement": 0.0,
            "num_evals": num_evals,
        }

    current_cps = dst_cps.copy()

    step = float(args.cp_sweep_initial_step)

    for step_idx in range(args.cp_sweep_steps):
        for _pass in range(args.cp_sweep_passes):
            any_improved = False

            for cp_idx in range(4):
                local_best_cost = best_cost
                local_best_cps = current_cps.copy()
                local_best_H = H_best

                # 8-neighborhood + stay.
                for dy in (-step, 0.0, step):
                    for dx in (-step, 0.0, step):
                        if dx == 0.0 and dy == 0.0:
                            continue

                        trial_cps = current_cps.copy()
                        trial_cps[cp_idx, 0] += dx
                        trial_cps[cp_idx, 1] += dy

                        H_trial = H_from_control_points(src_cps, trial_cps)
                        if H_trial is None:
                            continue

                        cost_trial, vr_trial = eval_H_on_tile_region(
                            target_y=target_y,
                            ref_y=ref_y,
                            H=H_trial,
                            tile=tile,
                            region=args.cp_sweep_cost_region,
                            min_valid_ratio=args.min_block_valid_ratio,
                        )
                        num_evals += 1

                        if not np.isfinite(cost_trial):
                            continue

                        if cost_trial + 1e-9 < local_best_cost:
                            local_best_cost = cost_trial
                            local_best_cps = trial_cps
                            local_best_H = H_trial

                if local_best_cost + 1e-9 < best_cost:
                    best_cost = local_best_cost
                    current_cps = local_best_cps
                    H_best = local_best_H
                    any_improved = True

            if not any_improved:
                break

        step *= float(args.cp_sweep_step_decay)

    improvement = float(initial_cost - best_cost)

    # Safety: keep initial H if CP sweep improved too little or got worse.
    if improvement < float(args.cp_sweep_min_gain):
        return H_init, {
            "enabled": True,
            "accepted": False,
            "reason": "cp_sweep_gain_too_small",
            "initial_cost": float(initial_cost),
            "final_cost": float(best_cost),
            "improvement": improvement,
            "num_evals": int(num_evals),
            "cost_region": args.cp_sweep_cost_region,
            "initial_cps": dst_cps.tolist(),
            "final_cps": current_cps.tolist(),
        }

    return H_best, {
        "enabled": True,
        "accepted": True,
        "reason": "cp_sweep_accepted",
        "initial_cost": float(initial_cost),
        "final_cost": float(best_cost),
        "improvement": improvement,
        "num_evals": int(num_evals),
        "cost_region": args.cp_sweep_cost_region,
        "initial_cps": dst_cps.tolist(),
        "final_cps": current_cps.tolist(),
    }


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

    tiles = make_tiles(
        width=width,
        height=height,
        block_size=block_size,
        edge_anchored_fit=edge_anchored_fit,
    )

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


def candidate_accept_by_cost(
    candidate_cost: float,
    fallback_cost: float,
    level: int,
    args,
) -> Tuple[bool, Dict]:
    if args.disable_cost_gate:
        return np.isfinite(candidate_cost), {
            "mode": "disabled",
            "abs_gain": None,
            "rel_gain": None,
            "abs_ok": True,
            "rel_ok": True,
        }

    if not np.isfinite(candidate_cost):
        return False, {
            "mode": "invalid_candidate",
            "abs_gain": None,
            "rel_gain": None,
            "abs_ok": False,
            "rel_ok": False,
        }

    if not np.isfinite(fallback_cost):
        return True, {
            "mode": "fallback_invalid",
            "abs_gain": None,
            "rel_gain": None,
            "abs_ok": True,
            "rel_ok": True,
        }

    abs_gain = float(fallback_cost - candidate_cost)
    rel_gain = float(abs_gain / max(abs(fallback_cost), 1e-9))

    # Root is allowed to be more permissive.
    min_rel_gain = args.root_min_rel_gain if level == 0 else args.child_min_rel_gain

    abs_ok = abs_gain >= float(args.min_gain)
    rel_ok = rel_gain >= float(min_rel_gain)

    if args.gain_gate_mode == "absolute":
        accepted = abs_ok
    elif args.gain_gate_mode == "relative":
        accepted = rel_ok
    elif args.gain_gate_mode == "either":
        accepted = abs_ok or rel_ok
    elif args.gain_gate_mode == "both":
        accepted = abs_ok and rel_ok
    else:
        raise ValueError(f"Unknown gain_gate_mode: {args.gain_gate_mode}")

    return accepted, {
        "mode": args.gain_gate_mode,
        "abs_gain": abs_gain,
        "rel_gain": rel_gain,
        "min_abs_gain": float(args.min_gain),
        "min_rel_gain": float(min_rel_gain),
        "abs_ok": bool(abs_ok),
        "rel_ok": bool(rel_ok),
    }


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
    level: int,
    args,
) -> Tuple[np.ndarray, List[List[Dict]], Dict[str, np.ndarray]]:
    height, width = target_y.shape
    ny, nx = block_grid_shape(width, height, block_size)

    edge_anchored_fit = not args.disable_edge_anchored_fit

    tiles = make_tiles(
        width=width,
        height=height,
        block_size=block_size,
        edge_anchored_fit=edge_anchored_fit,
    )

    H_grid = np.zeros((ny, nx, 3, 3), dtype=np.float64)

    source_grid = np.zeros((ny, nx), dtype=np.float64)
    match_count_grid = np.zeros((ny, nx), dtype=np.float64)
    inlier_count_grid = np.zeros((ny, nx), dtype=np.float64)
    reproj_mae_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    candidate_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    fallback_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    chosen_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)
    cp_sweep_gain_grid = np.zeros((ny, nx), dtype=np.float64)

    # source id:
    # 0 root_fallback
    # 1 parent_inherit
    # 2 local_fit
    # 3 local_fit_rejected_cost
    # 4 local_fit_cp_sweep
    source_id = {
        "root_fallback": 0,
        "parent_inherit": 1,
        "local_fit": 2,
        "local_fit_rejected_cost": 3,
        "local_fit_cp_sweep": 4,
    }

    level_records: List[List[Dict]] = []

    accepted = 0
    accepted_cp_sweep = 0
    inherited = 0
    rejected = 0

    do_cp_sweep_this_level = (
        (not args.disable_cp_sweep)
        and level < args.cp_sweep_levels
    )

    for by_idx, row in enumerate(tiles):
        row_records = []

        for bx_idx, tile in enumerate(row):
            parent_H = parent_H_for_block(
                parent_H_grid=parent_H_grid,
                parent_block_size=parent_block_size,
                out_x0=tile.out_x0,
                out_y0=tile.out_y0,
            )

            if parent_H is None:
                fallback_H = root_fallback_H.copy()
                fallback_source = "root_fallback"
            else:
                fallback_H = parent_H.copy()
                fallback_source = "parent_inherit"

            # Cost is evaluated only on the actual non-overlap output tile.
            fallback_cost, fallback_valid_ratio = eval_block_photometric_cost(
                target_y=target_y,
                ref_y=ref_y,
                H=fallback_H,
                bx=tile.out_x0,
                by=tile.out_y0,
                bw=tile.out_w,
                bh=tile.out_h,
                min_valid_ratio=args.min_block_valid_ratio,
            )

            # Matches are collected from the fitting tile.
            pts_t, pts_r = collect_matches_for_block(
                pts_t_all=pts_t_all,
                pts_r_all=pts_r_all,
                bx=tile.fit_x0,
                by=tile.fit_y0,
                bw=tile.fit_w,
                bh=tile.fit_h,
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

            cp_sweep_meta = {
                "enabled": False,
                "reason": "not_attempted",
                "improvement": 0.0,
                "num_evals": 0,
            }

            chosen_H = fallback_H
            chosen_source = fallback_source
            candidate_cost = float("inf")
            chosen_cost = fallback_cost
            chosen_valid_ratio = fallback_valid_ratio
            reason = fit_reason
            gate_meta = {}

            if H_candidate is not None:
                if do_cp_sweep_this_level:
                    H_refined, cp_sweep_meta = refine_homography_cp_sweep(
                        target_y=target_y,
                        ref_y=ref_y,
                        H_init=H_candidate,
                        tile=tile,
                        args=args,
                    )

                    H_candidate = H_refined

                # Candidate is judged only on the output tile.
                candidate_cost, candidate_valid_ratio = eval_block_photometric_cost(
                    target_y=target_y,
                    ref_y=ref_y,
                    H=H_candidate,
                    bx=tile.out_x0,
                    by=tile.out_y0,
                    bw=tile.out_w,
                    bh=tile.out_h,
                    min_valid_ratio=args.min_block_valid_ratio,
                )

                accept_by_cost, gate_meta = candidate_accept_by_cost(
                    candidate_cost=candidate_cost,
                    fallback_cost=fallback_cost,
                    level=level,
                    args=args,
                )

                if np.isfinite(candidate_cost) and accept_by_cost:
                    chosen_H = H_candidate

                    if cp_sweep_meta.get("accepted", False):
                        chosen_source = "local_fit_cp_sweep"
                        accepted_cp_sweep += 1
                    else:
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
            cp_sweep_gain_grid[by_idx, bx_idx] = float(cp_sweep_meta.get("improvement", 0.0))

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

                    # Keep old names for compatibility.
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
                    "gain_gate": gate_meta,
                    "cp_sweep": cp_sweep_meta,
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
        "cp_sweep_gain_grid": cp_sweep_gain_grid,
        "accepted": np.array([accepted], dtype=np.float64),
        "accepted_cp_sweep": np.array([accepted_cp_sweep], dtype=np.float64),
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

    parser.add_argument(
        "--disable-edge-anchored-fit",
        action="store_true",
        help=(
            "Disable edge-anchored fitting. If disabled, right/bottom partial blocks "
            "use only their partial output area for fitting, matching the old behavior."
        ),
    )

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

    # Cost gate.
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--root-min-rel-gain", type=float, default=0.0)
    parser.add_argument("--child-min-rel-gain", type=float, default=0.05)
    parser.add_argument(
        "--gain-gate-mode",
        choices=["absolute", "relative", "either", "both"],
        default="both",
        help=(
            "How to accept local H over parent/fallback. "
            "For child levels, default requires both abs gain and relative gain. "
            "If min-gain is 0, child-min-rel-gain dominates."
        ),
    )
    parser.add_argument("--disable-cost-gate", action="store_true")

    # CP sweep.
    parser.add_argument("--disable-cp-sweep", action="store_true")
    parser.add_argument(
        "--cp-sweep-levels",
        type=int,
        default=1,
        help="Number of top levels where CP sweep is enabled. Default 1 means root level only.",
    )
    parser.add_argument(
        "--cp-sweep-cost-region",
        choices=["fit", "out", "both"],
        default="fit",
        help="Photometric region used during CP sweep.",
    )
    parser.add_argument("--cp-sweep-steps", type=int, default=3)
    parser.add_argument("--cp-sweep-passes", type=int, default=1)
    parser.add_argument("--cp-sweep-initial-step", type=float, default=2.0)
    parser.add_argument("--cp-sweep-step-decay", type=float, default=0.5)
    parser.add_argument("--cp-sweep-min-gain", type=float, default=0.0)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    width = args.width
    height = args.height
    edge_anchored_fit = not args.disable_edge_anchored_fit

    print(f"[INFO] target={args.target_idx}, ref={args.ref_idx}")
    print(f"[INFO] size={width}x{height}, bitdepth={args.bitdepth}")
    print(f"[INFO] start_block={args.start_block_size}, levels={args.levels}, min_block={args.min_block_size}")
    print(f"[INFO] edge_anchored_fit={edge_anchored_fit}")
    print(f"[INFO] cp_sweep_enabled={not args.disable_cp_sweep}, cp_sweep_levels={args.cp_sweep_levels}")
    print(f"[INFO] gain_gate_mode={args.gain_gate_mode}, child_min_rel_gain={args.child_min_rel_gain}")

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
            level=level,
            args=args,
        )

        pred, valid = render_prediction_from_H_grid(
            target_y=target.y,
            ref_y=ref.y,
            H_grid=H_grid,
            block_size=block_size,
            edge_anchored_fit=edge_anchored_fit,
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

        save_scalar_map_png(
            os.path.join(args.output_dir, f"cp_sweep_gain_{level_tag}.png"),
            aux["cp_sweep_gain_grid"],
            width,
            height,
            block_size,
        )

        accepted = int(aux["accepted"][0])
        accepted_cp_sweep = int(aux["accepted_cp_sweep"][0])
        inherited = int(aux["inherited"][0])
        rejected = int(aux["rejected"][0])

        summary = {
            "level": int(level),
            "block_size": int(block_size),
            "num_blocks": int(H_grid.shape[0] * H_grid.shape[1]),
            "accepted_local_fit": accepted,
            "accepted_cp_sweep": accepted_cp_sweep,
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
                "cp_sweep_gain_grid": aux["cp_sweep_gain_grid"].tolist(),
            }
        )

        final_pred = pred
        final_valid = valid
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
            "description": "hierarchical block-wise homography fitting with root CP sweep and strong parent propagation",
            "coordinate": "target pixel -> ref pixel",
            "start_block_size": int(args.start_block_size),
            "levels": int(args.levels),
            "min_block_size": int(args.min_block_size),
            "block_margin": int(args.block_margin),
            "edge_anchored_fit": bool(edge_anchored_fit),
            "root_fallback": args.root_fallback,
            "parent_inheritance": True,
            "cost_gate": bool(not args.disable_cost_gate),
            "fit_output_region_separation": (
                "H is fitted from fit tile; photometric accept/reject and final rendering use output tile only."
            ),
            "root_cp_sweep": {
                "enabled": bool(not args.disable_cp_sweep),
                "cp_sweep_levels": int(args.cp_sweep_levels),
                "cost_region": args.cp_sweep_cost_region,
                "steps": int(args.cp_sweep_steps),
                "passes": int(args.cp_sweep_passes),
                "initial_step": float(args.cp_sweep_initial_step),
                "step_decay": float(args.cp_sweep_step_decay),
                "min_gain": float(args.cp_sweep_min_gain),
            },
            "gain_gate": {
                "mode": args.gain_gate_mode,
                "min_abs_gain": float(args.min_gain),
                "root_min_rel_gain": float(args.root_min_rel_gain),
                "child_min_rel_gain": float(args.child_min_rel_gain),
            },
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
            "root_min_rel_gain": float(args.root_min_rel_gain),
            "child_min_rel_gain": float(args.child_min_rel_gain),
            "gain_gate_mode": args.gain_gate_mode,
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
            "source_map value 4 means local homography accepted after CP sweep refinement.",
            "source_map value 1 means parent homography inherited.",
            "source_map value 3 means local homography was fitted but rejected by cost gate.",
            "Root level CP sweep refines RANSAC H by photometric control-point coordinate descent.",
            "Child levels skip CP sweep by default and accept local H only when it significantly beats parent H.",
            "For edge blocks, fit_x/fit_y/fit_w/fit_h may differ from output block region.",
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
