#!/usr/bin/env python3
# homography_block_residual_basis.py
#
# Image-space factorized warp:
#
#   1. Estimate global homography H_g from target -> ref.
#   2. Around H_g, estimate block-wise residual MV by block matching.
#   3. Fit a smooth global residual vector field B(x) from reliable block residual MVs.
#   4. For each block, find scalar c_b.
#   5. Final predictor:
#
#        p_ref(x, b) = H_g(x) + c_b * B(x)
#
# This is not physical R|t + depth.
# But it has a similar coding/tool structure:
#
#   global info + block-wise constant scalar.
#
# Outputs:
#   - pred_homography.yuv
#   - pred_block_residual_mv.yuv       : upper-bound block residual MV predictor
#   - pred_basis_scalar.yuv            : H + global basis * block scalar
#   - c_map.png
#   - residual_mv_map.png
#   - basis_field.png
#   - result.json
#
# Example:
#   python homography_block_residual_basis.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --block-size 32 \
#     --res-search-range 12 \
#     --res-search-step 1 \
#     --basis-degree 2 \
#     --c-samples 9 \
#     --c-refine-range 2.0 \
#     --max-features 30000 \
#     --match-ratio 0.65 \
#     --ransac-thresh 0.75 \
#     --h-refine-iters 100 \
#     --h-refine-scale 0.5 \
#     --output-dir h_basis_t1_r0

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
class BlockResidualResult:
    dx_grid: np.ndarray
    dy_grid: np.ndarray
    cost_grid: np.ndarray
    second_cost_grid: np.ndarray
    gap_grid: np.ndarray
    valid_ratio_grid: np.ndarray
    texture_grid: np.ndarray
    reliable_grid: np.ndarray
    pred_block_mv: np.ndarray
    valid_block_mv: np.ndarray


@dataclass
class BasisFitResult:
    coef_x: np.ndarray
    coef_y: np.ndarray
    norm_scale: float
    basis_x: np.ndarray
    basis_y: np.ndarray
    basis_x_block: np.ndarray
    basis_y_block: np.ndarray
    fit_mask: np.ndarray
    fit_weights: np.ndarray
    fit_error_grid: np.ndarray
    num_fit_blocks: int


@dataclass
class ScalarPredictResult:
    c_grid: np.ndarray
    cost_grid: np.ndarray
    valid_ratio_grid: np.ndarray
    pred: np.ndarray
    valid: np.ndarray


# ============================================================
# Basic utilities
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

    raise ValueError("Only 8-bit and 10-bit are supported.")


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


def to_float_ecc(y: np.ndarray, bitdepth: int) -> np.ndarray:
    maxv = 255.0 if bitdepth == 8 else 1023.0
    return np.clip(y.astype(np.float32) / maxv, 0.0, 1.0).astype(np.float32)


def resize_frame(y: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-12:
        return y.copy()

    h, w = y.shape
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))

    return cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def block_sum_grid(arr: np.ndarray, block_size: int) -> np.ndarray:
    h, w = arr.shape
    ny, nx = block_grid_shape(w, h, block_size)

    integ = cv2.integral(arr.astype(np.float64))
    out = np.zeros((ny, nx), dtype=np.float64)

    for by_idx, by in enumerate(range(0, h, block_size)):
        y1 = min(by + block_size, h)

        for bx_idx, bx in enumerate(range(0, w, block_size)):
            x1 = min(bx + block_size, w)
            s = (
                integ[y1, x1]
                - integ[by, x1]
                - integ[y1, bx]
                + integ[by, bx]
            )
            out[by_idx, bx_idx] = s

    return out


def block_mean_grid(arr: np.ndarray, block_size: int) -> np.ndarray:
    h, w = arr.shape
    sums = block_sum_grid(arr, block_size)
    ny, nx = sums.shape

    out = np.zeros_like(sums, dtype=np.float64)

    for by_idx, by in enumerate(range(0, h, block_size)):
        y1 = min(by + block_size, h)

        for bx_idx, bx in enumerate(range(0, w, block_size)):
            x1 = min(bx + block_size, w)
            area = max(1, (y1 - by) * (x1 - bx))
            out[by_idx, bx_idx] = sums[by_idx, bx_idx] / area

    return out


def calc_block_texture(target_y: np.ndarray, block_size: int) -> np.ndarray:
    target_f = target_y.astype(np.float32)
    mean = block_mean_grid(target_f, block_size)
    mean2 = block_mean_grid(target_f * target_f, block_size)

    var = np.maximum(0.0, mean2 - mean * mean)
    return np.sqrt(var)


# ============================================================
# Feature matching / Homography / ECC
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


def estimate_initial_homography(
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    ransac_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    H, mask = cv2.findHomography(
        pts_target,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
    )

    if H is None:
        raise RuntimeError("cv2.findHomography failed.")

    return normalize_homography(H), mask


def save_match_vis(
    out_path: str,
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    match_result: MatchResult,
    inlier_mask: Optional[np.ndarray],
    max_draw: int = 200,
):
    target_8 = to_8bit(target_y, bitdepth)
    ref_8 = to_8bit(ref_y, bitdepth)

    matches = match_result.good_matches

    if inlier_mask is not None:
        mask = np.asarray(inlier_mask).reshape(-1) != 0
        matches = [m for m, ok in zip(matches, mask) if ok]

    matches = matches[:max_draw]

    vis = cv2.drawMatches(
        target_8,
        match_result.keypoints_target,
        ref_8,
        match_result.keypoints_ref,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(out_path, vis)


def scale_homography_for_resized_image(H_full: np.ndarray, scale: float) -> np.ndarray:
    S = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_homography(S @ H_full @ np.linalg.inv(S))


def unscale_homography_to_full_image(H_scaled: np.ndarray, scale: float) -> np.ndarray:
    S = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_homography(np.linalg.inv(S) @ H_scaled @ S)


def preprocess_ecc(img: np.ndarray, blur_ksize: int) -> np.ndarray:
    out = img.astype(np.float32)

    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)

    return out.astype(np.float32)


def refine_homography_ecc(
    H_init_full: np.ndarray,
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    iters: int,
    eps: float,
    scale: float,
    blur_ksize: int,
) -> Tuple[np.ndarray, dict]:
    meta = {
        "enabled": bool(iters > 0),
        "success": False,
        "cc": None,
        "iters": int(iters),
        "eps": float(eps),
        "scale": float(scale),
        "blur_ksize": int(blur_ksize),
        "error": None,
    }

    H_init_full = normalize_homography(H_init_full)

    if iters <= 0:
        return H_init_full, meta

    if scale <= 0.0 or scale > 1.0:
        raise ValueError("--h-refine-scale must be in (0, 1].")

    target_f = resize_frame(to_float_ecc(target_y, bitdepth), scale)
    ref_f = resize_frame(to_float_ecc(ref_y, bitdepth), scale)

    target_f = preprocess_ecc(target_f, blur_ksize)
    ref_f = preprocess_ecc(ref_f, blur_ksize)

    H_scaled = scale_homography_for_resized_image(H_init_full, scale).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(iters),
        float(eps),
    )

    try:
        try:
            cc, warp_refined = cv2.findTransformECC(
                target_f,
                ref_f,
                H_scaled,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
                None,
                5,
            )
        except TypeError:
            cc, warp_refined = cv2.findTransformECC(
                target_f,
                ref_f,
                H_scaled,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
            )

        H_refined = unscale_homography_to_full_image(warp_refined.astype(np.float64), scale)
        meta["success"] = True
        meta["cc"] = float(cc)

        return H_refined, meta

    except cv2.error as e:
        meta["error"] = str(e)
        return H_init_full, meta


# ============================================================
# Homography mapping
# ============================================================

def homography_maps(H: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = normalize_homography(H)

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
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


# ============================================================
# Block residual MV search around homography
# ============================================================

def candidate_shifts(search_range: float, search_step: float) -> List[Tuple[float, float]]:
    vals = np.arange(-search_range, search_range + 1e-9, search_step, dtype=np.float32)
    shifts = []

    for dy in vals:
        for dx in vals:
            shifts.append((float(dx), float(dy)))

    return shifts


def estimate_block_residual_mv(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    map_x_H: np.ndarray,
    map_y_H: np.ndarray,
    valid_H: np.ndarray,
    block_size: int,
    search_range: float,
    search_step: float,
    min_block_valid_ratio: float,
    min_texture: float,
    min_gap: float,
) -> BlockResidualResult:
    h, w = target_y.shape
    ny, nx = block_grid_shape(w, h, block_size)

    target_f = target_y.astype(np.float32)

    texture_grid = calc_block_texture(target_y, block_size)

    best_cost = np.full((ny, nx), np.inf, dtype=np.float64)
    second_cost = np.full((ny, nx), np.inf, dtype=np.float64)
    best_dx = np.zeros((ny, nx), dtype=np.float64)
    best_dy = np.zeros((ny, nx), dtype=np.float64)
    best_valid_ratio = np.zeros((ny, nx), dtype=np.float64)

    shifts = candidate_shifts(search_range, search_step)
    print(f"[INFO] residual MV candidate shifts = {len(shifts)}")

    for idx, (dx, dy) in enumerate(shifts):
        if idx % max(1, len(shifts) // 10) == 0:
            print(f"[BM] shift {idx + 1}/{len(shifts)} dx={dx} dy={dy}")

        map_x = map_x_H + dx
        map_y = map_y_H + dy

        valid = (
            valid_H
            & (map_x >= 0.0)
            & (map_x <= w - 1.0)
            & (map_y >= 0.0)
            & (map_y <= h - 1.0)
        )

        pred = remap_ref(ref_y, map_x, map_y)
        diff = np.abs(target_f - pred)

        diff_sum = block_sum_grid(diff * valid.astype(np.float32), block_size)
        valid_sum = block_sum_grid(valid.astype(np.float32), block_size)

        for by_idx, by in enumerate(range(0, h, block_size)):
            y1 = min(by + block_size, h)

            for bx_idx, bx in enumerate(range(0, w, block_size)):
                x1 = min(bx + block_size, w)
                area = max(1, (y1 - by) * (x1 - bx))

                valid_ratio = valid_sum[by_idx, bx_idx] / area

                if valid_ratio < min_block_valid_ratio:
                    continue

                cost = diff_sum[by_idx, bx_idx] / max(valid_sum[by_idx, bx_idx], 1.0)

                if cost < best_cost[by_idx, bx_idx]:
                    second_cost[by_idx, bx_idx] = best_cost[by_idx, bx_idx]
                    best_cost[by_idx, bx_idx] = cost
                    best_dx[by_idx, bx_idx] = dx
                    best_dy[by_idx, bx_idx] = dy
                    best_valid_ratio[by_idx, bx_idx] = valid_ratio
                elif cost < second_cost[by_idx, bx_idx]:
                    second_cost[by_idx, bx_idx] = cost

    gap = second_cost - best_cost

    reliable = (
        np.isfinite(best_cost)
        & (best_valid_ratio >= min_block_valid_ratio)
        & (texture_grid >= min_texture)
        & (gap >= min_gap)
    )

    pred_block = np.zeros_like(target_f, dtype=np.float32)
    valid_block = np.zeros_like(target_y, dtype=bool)

    for by_idx, by in enumerate(range(0, h, block_size)):
        y1 = min(by + block_size, h)

        for bx_idx, bx in enumerate(range(0, w, block_size)):
            x1 = min(bx + block_size, w)

            dx = best_dx[by_idx, bx_idx]
            dy = best_dy[by_idx, bx_idx]

            mx = map_x_H[by:y1, bx:x1] + dx
            my = map_y_H[by:y1, bx:x1] + dy

            v = (
                valid_H[by:y1, bx:x1]
                & (mx >= 0.0)
                & (mx <= w - 1.0)
                & (my >= 0.0)
                & (my <= h - 1.0)
            )

            pred_roi = remap_ref(ref_y, mx.astype(np.float32), my.astype(np.float32))

            pred_block[by:y1, bx:x1] = pred_roi
            valid_block[by:y1, bx:x1] = v

    return BlockResidualResult(
        dx_grid=best_dx,
        dy_grid=best_dy,
        cost_grid=best_cost,
        second_cost_grid=second_cost,
        gap_grid=gap,
        valid_ratio_grid=best_valid_ratio,
        texture_grid=texture_grid,
        reliable_grid=reliable,
        pred_block_mv=pred_block,
        valid_block_mv=valid_block,
    )


# ============================================================
# Polynomial basis fitting
# ============================================================

def normalized_xy_pixel_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    if width > 1:
        xn = (xs / (width - 1.0)) * 2.0 - 1.0
    else:
        xn = np.zeros_like(xs)

    if height > 1:
        yn = (ys / (height - 1.0)) * 2.0 - 1.0
    else:
        yn = np.zeros_like(ys)

    return xn.astype(np.float32), yn.astype(np.float32)


def block_center_normalized_grid(width: int, height: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = block_grid_shape(width, height, block_size)

    cx = np.zeros((ny, nx), dtype=np.float64)
    cy = np.zeros((ny, nx), dtype=np.float64)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        yy = 0.5 * (by + y1 - 1.0)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            xx = 0.5 * (bx + x1 - 1.0)

            cx[by_idx, bx_idx] = (xx / max(width - 1.0, 1.0)) * 2.0 - 1.0
            cy[by_idx, bx_idx] = (yy / max(height - 1.0, 1.0)) * 2.0 - 1.0

    return cx, cy


def poly_features(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)

    if degree == 0:
        return np.stack([np.ones_like(x)], axis=-1)

    if degree == 1:
        return np.stack(
            [
                np.ones_like(x),
                x,
                y,
            ],
            axis=-1,
        )

    if degree == 2:
        return np.stack(
            [
                np.ones_like(x),
                x,
                y,
                x * x,
                x * y,
                y * y,
            ],
            axis=-1,
        )

    raise ValueError("--basis-degree must be 0, 1, or 2.")


def weighted_lstsq(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.maximum(w.astype(np.float64), 1e-12)
    sw = np.sqrt(w)

    Aw = A * sw[:, None]
    bw = b * sw

    coef, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    return coef.astype(np.float64)


def fit_polynomial_residual_basis(
    dx_grid: np.ndarray,
    dy_grid: np.ndarray,
    reliable_grid: np.ndarray,
    cost_grid: np.ndarray,
    gap_grid: np.ndarray,
    texture_grid: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    degree: int,
    robust_iters: int,
    min_fit_blocks: int,
) -> BasisFitResult:
    ny, nx = dx_grid.shape

    cx, cy = block_center_normalized_grid(width, height, block_size)
    A_all = poly_features(cx.reshape(-1), cy.reshape(-1), degree).astype(np.float64)

    dx_all = dx_grid.reshape(-1).astype(np.float64)
    dy_all = dy_grid.reshape(-1).astype(np.float64)
    reliable = reliable_grid.reshape(-1).astype(bool)

    finite = np.isfinite(dx_all) & np.isfinite(dy_all) & np.isfinite(cost_grid.reshape(-1))
    mask = reliable & finite

    if np.count_nonzero(mask) < min_fit_blocks:
        print("[WARN] Not enough reliable blocks for polynomial fit. Falling back to all finite blocks.")
        mask = finite

    if np.count_nonzero(mask) < max(3, min_fit_blocks):
        print("[WARN] Still not enough blocks. Using mean residual fallback.")

        mean_dx = float(np.mean(dx_all[finite])) if np.any(finite) else 0.0
        mean_dy = float(np.mean(dy_all[finite])) if np.any(finite) else 0.0

        ncoef = poly_features(np.array([0.0]), np.array([0.0]), degree).shape[-1]
        coef_x = np.zeros(ncoef, dtype=np.float64)
        coef_y = np.zeros(ncoef, dtype=np.float64)
        coef_x[0] = mean_dx
        coef_y[0] = mean_dy

        fit_mask = finite.reshape(ny, nx)
        weights = fit_mask.astype(np.float64)

    else:
        A = A_all[mask]
        bx = dx_all[mask]
        by = dy_all[mask]

        # Base reliability weights.
        gap = gap_grid.reshape(-1)[mask].astype(np.float64)
        tex = texture_grid.reshape(-1)[mask].astype(np.float64)
        cost = cost_grid.reshape(-1)[mask].astype(np.float64)

        w = np.ones_like(bx, dtype=np.float64)
        w *= np.clip(gap / (np.median(gap) + 1e-6), 0.25, 4.0)
        w *= np.clip(tex / (np.median(tex) + 1e-6), 0.25, 4.0)
        w *= np.clip((np.median(cost) + 1e-6) / (cost + 1e-6), 0.25, 4.0)

        coef_x = None
        coef_y = None

        for it in range(max(1, robust_iters)):
            coef_x = weighted_lstsq(A, bx, w)
            coef_y = weighted_lstsq(A, by, w)

            pred_x = A @ coef_x
            pred_y = A @ coef_y
            err = np.sqrt((bx - pred_x) ** 2 + (by - pred_y) ** 2)

            med = np.median(err)
            sigma = 1.4826 * np.median(np.abs(err - med)) + 1e-6
            huber = np.minimum(1.0, (2.5 * sigma) / (err + 1e-6))
            w = w * huber

        fit_mask = mask.reshape(ny, nx)
        weights = np.zeros(ny * nx, dtype=np.float64)
        weights[mask] = w
        weights = weights.reshape(ny, nx)

    # Evaluate basis on pixels.
    xn, yn = normalized_xy_pixel_grid(width, height)
    A_pix = poly_features(xn, yn, degree)
    basis_x = np.tensordot(A_pix, coef_x, axes=([-1], [0])).astype(np.float32)
    basis_y = np.tensordot(A_pix, coef_y, axes=([-1], [0])).astype(np.float32)

    # Evaluate basis at block centers.
    A_blk = poly_features(cx, cy, degree)
    basis_x_blk = np.tensordot(A_blk, coef_x, axes=([-1], [0])).astype(np.float64)
    basis_y_blk = np.tensordot(A_blk, coef_y, axes=([-1], [0])).astype(np.float64)

    # Normalize basis RMS to 1 pixel, so c_b roughly means residual strength in pixels.
    mag2 = basis_x_blk * basis_x_blk + basis_y_blk * basis_y_blk
    if np.any(fit_mask):
        rms = math.sqrt(float(np.mean(mag2[fit_mask])))
    else:
        rms = math.sqrt(float(np.mean(mag2)))

    norm_scale = max(rms, 1e-6)

    basis_x /= norm_scale
    basis_y /= norm_scale
    basis_x_blk /= norm_scale
    basis_y_blk /= norm_scale
    coef_x = coef_x / norm_scale
    coef_y = coef_y / norm_scale

    # Fit error grid after normalization, with scalar projection allowed.
    fit_error = np.full((ny, nx), np.inf, dtype=np.float64)

    denom = basis_x_blk * basis_x_blk + basis_y_blk * basis_y_blk + 1e-12
    c_ls = (dx_grid * basis_x_blk + dy_grid * basis_y_blk) / denom

    pred_dx = c_ls * basis_x_blk
    pred_dy = c_ls * basis_y_blk

    err_grid = np.sqrt((dx_grid - pred_dx) ** 2 + (dy_grid - pred_dy) ** 2)
    fit_error[np.isfinite(err_grid)] = err_grid[np.isfinite(err_grid)]

    return BasisFitResult(
        coef_x=coef_x,
        coef_y=coef_y,
        norm_scale=float(norm_scale),
        basis_x=basis_x.astype(np.float32),
        basis_y=basis_y.astype(np.float32),
        basis_x_block=basis_x_blk.astype(np.float64),
        basis_y_block=basis_y_blk.astype(np.float64),
        fit_mask=fit_mask,
        fit_weights=weights.astype(np.float64),
        fit_error_grid=fit_error,
        num_fit_blocks=int(np.count_nonzero(fit_mask)),
    )


# ============================================================
# Scalar c search
# ============================================================

def estimate_initial_c_grid_from_observed_residual(
    dx_grid: np.ndarray,
    dy_grid: np.ndarray,
    basis_x_blk: np.ndarray,
    basis_y_blk: np.ndarray,
    c_min: float,
    c_max: float,
) -> np.ndarray:
    denom = basis_x_blk * basis_x_blk + basis_y_blk * basis_y_blk + 1e-12
    c = (dx_grid * basis_x_blk + dy_grid * basis_y_blk) / denom
    return np.clip(c, c_min, c_max).astype(np.float64)


def c_candidates(center: float, refine_range: float, samples: int, c_min: float, c_max: float) -> np.ndarray:
    if samples <= 1 or refine_range <= 1e-12:
        return np.array([np.clip(center, c_min, c_max)], dtype=np.float64)

    lo = max(c_min, center - refine_range)
    hi = min(c_max, center + refine_range)

    if abs(hi - lo) < 1e-12:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)

    return np.linspace(lo, hi, samples, dtype=np.float64)


def predict_with_basis_scalar(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    map_x_H: np.ndarray,
    map_y_H: np.ndarray,
    valid_H: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    c_center_grid: np.ndarray,
    block_size: int,
    c_refine_range: float,
    c_samples: int,
    c_min: float,
    c_max: float,
    min_block_valid_ratio: float,
) -> ScalarPredictResult:
    h, w = target_y.shape
    ny, nx = block_grid_shape(w, h, block_size)

    target_f = target_y.astype(np.float32)

    c_grid = np.zeros((ny, nx), dtype=np.float64)
    cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)

    pred_out = np.zeros((h, w), dtype=np.float32)
    valid_out = np.zeros((h, w), dtype=bool)

    for by_idx, by in enumerate(range(0, h, block_size)):
        y1 = min(by + block_size, h)

        for bx_idx, bx in enumerate(range(0, w, block_size)):
            x1 = min(bx + block_size, w)

            center = float(c_center_grid[by_idx, bx_idx])
            cand_list = c_candidates(center, c_refine_range, c_samples, c_min, c_max)

            best_cost = float("inf")
            best_c = center
            best_pred = None
            best_valid = None
            best_valid_ratio = 0.0

            target_roi = target_f[by:y1, bx:x1]

            Hx = map_x_H[by:y1, bx:x1]
            Hy = map_y_H[by:y1, bx:x1]
            Vh = valid_H[by:y1, bx:x1]
            Bx = basis_x[by:y1, bx:x1]
            By = basis_y[by:y1, bx:x1]

            for c in cand_list:
                mx = Hx + float(c) * Bx
                my = Hy + float(c) * By

                valid = (
                    Vh
                    & (mx >= 0.0)
                    & (mx <= w - 1.0)
                    & (my >= 0.0)
                    & (my <= h - 1.0)
                )

                valid_ratio = float(np.mean(valid))

                if valid_ratio < min_block_valid_ratio or not np.any(valid):
                    continue

                pred_roi = remap_ref(ref_y, mx.astype(np.float32), my.astype(np.float32))
                cost = float(np.mean(np.abs(target_roi[valid] - pred_roi[valid])))

                if cost < best_cost:
                    best_cost = cost
                    best_c = float(c)
                    best_pred = pred_roi
                    best_valid = valid
                    best_valid_ratio = valid_ratio

            c_grid[by_idx, bx_idx] = best_c
            cost_grid[by_idx, bx_idx] = best_cost
            valid_ratio_grid[by_idx, bx_idx] = best_valid_ratio

            if best_pred is not None and best_valid is not None:
                pred_out[by:y1, bx:x1] = best_pred
                valid_out[by:y1, bx:x1] = best_valid

    return ScalarPredictResult(
        c_grid=c_grid,
        cost_grid=cost_grid,
        valid_ratio_grid=valid_ratio_grid,
        pred=pred_out,
        valid=valid_out,
    )


# ============================================================
# Cost / Visualization
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
    ny, nx = grid.shape
    out = np.zeros((height, width), dtype=np.float32)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            out[by:y1, bx:x1] = float(grid[by_idx, bx_idx])

    return out


def save_scalar_map_png(path: str, grid: np.ndarray, width: int, height: int, block_size: int):
    img = expand_grid_to_image(grid, width, height, block_size)

    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))

    if abs(vmax - vmin) < 1e-12:
        out = np.full_like(img, 128, dtype=np.uint8)
    else:
        out = np.clip((img - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, color)


def save_vector_field_png(
    path: str,
    vx_grid: np.ndarray,
    vy_grid: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    scale: float = 4.0,
):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    mag = np.sqrt(vx_grid * vx_grid + vy_grid * vy_grid)
    save_scalar_map_png(path + ".mag.png", mag, width, height, block_size)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        cy = int(round(0.5 * (by + y1 - 1)))

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            cx = int(round(0.5 * (bx + x1 - 1)))

            dx = float(vx_grid[by_idx, bx_idx])
            dy = float(vy_grid[by_idx, bx_idx])

            p0 = (cx, cy)
            p1 = (int(round(cx + scale * dx)), int(round(cy + scale * dy)))

            cv2.arrowedLine(canvas, p0, p1, (0, 255, 0), 1, tipLength=0.3)

    cv2.imwrite(path, canvas)


def save_basis_field_png(path: str, basis_x: np.ndarray, basis_y: np.ndarray, stride: int = 64, scale: float = 12.0):
    h, w = basis_x.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    mag = np.sqrt(basis_x * basis_x + basis_y * basis_y)
    mag_norm = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    canvas = cv2.applyColorMap(mag_norm, cv2.COLORMAP_TURBO)

    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            dx = float(basis_x[y, x])
            dy = float(basis_y[y, x])

            p0 = (x, y)
            p1 = (int(round(x + scale * dx)), int(round(y + scale * dy)))

            cv2.arrowedLine(canvas, p0, p1, (255, 255, 255), 1, tipLength=0.3)

    cv2.imwrite(path, canvas)


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

    # Homography estimation.
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--match-ratio", type=float, default=0.65)
    parser.add_argument("--ransac-thresh", type=float, default=0.75)
    parser.add_argument("--no-clahe", action="store_true")

    # ECC refinement.
    parser.add_argument("--h-refine-iters", type=int, default=100)
    parser.add_argument("--h-refine-eps", type=float, default=1e-7)
    parser.add_argument("--h-refine-scale", type=float, default=0.5)
    parser.add_argument("--h-refine-blur", type=int, default=5)

    # Block residual MV search.
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--res-search-range", type=float, default=12.0)
    parser.add_argument("--res-search-step", type=float, default=1.0)

    # Reliability filtering.
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-texture", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=0.0)

    # Basis.
    parser.add_argument("--basis-degree", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--robust-fit-iters", type=int, default=3)
    parser.add_argument("--min-fit-blocks", type=int, default=20)

    # Block scalar c.
    parser.add_argument("--c-min", type=float, default=-32.0)
    parser.add_argument("--c-max", type=float, default=32.0)
    parser.add_argument("--c-samples", type=int, default=9)
    parser.add_argument("--c-refine-range", type=float, default=2.0)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print(f"[INFO] target_idx={args.target_idx}, ref_idx={args.ref_idx}")
    print(f"[INFO] resolution={args.width}x{args.height}, bitdepth={args.bitdepth}")
    print(f"[INFO] block_size={args.block_size}")
    print(f"[INFO] residual search range=±{args.res_search_range}, step={args.res_search_step}")

    target = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.ref_idx)

    # ------------------------------------------------------------
    # Global homography.
    # ------------------------------------------------------------

    match_result = detect_and_match_orb(
        target_y=target.y,
        ref_y=ref.y,
        bitdepth=args.bitdepth,
        max_features=args.max_features,
        ratio=args.match_ratio,
        use_clahe=not args.no_clahe,
    )

    print(f"[INFO] good matches = {len(match_result.good_matches)}")

    H_init, H_mask = estimate_initial_homography(
        pts_target=match_result.pts_target,
        pts_ref=match_result.pts_ref,
        ransac_thresh=args.ransac_thresh,
    )

    H_inliers = int(np.count_nonzero(H_mask)) if H_mask is not None else 0

    print(f"[INFO] H inliers = {H_inliers}")
    print("[INFO] Initial H:")
    print(H_init)

    save_match_vis(
        os.path.join(args.output_dir, "match_vis_homography_inliers.png"),
        target.y,
        ref.y,
        args.bitdepth,
        match_result,
        H_mask,
    )

    H_refined, ecc_meta = refine_homography_ecc(
        H_init_full=H_init,
        target_y=target.y,
        ref_y=ref.y,
        bitdepth=args.bitdepth,
        iters=args.h_refine_iters,
        eps=args.h_refine_eps,
        scale=args.h_refine_scale,
        blur_ksize=args.h_refine_blur,
    )

    print("[INFO] ECC meta:")
    print(json.dumps(ecc_meta, indent=2))
    print("[INFO] Refined H:")
    print(H_refined)

    map_x_H, map_y_H, valid_H = homography_maps(H_refined, args.width, args.height)

    pred_H = remap_ref(ref.y, map_x_H, map_y_H)
    cost_H = calc_cost(target.y, pred_H, valid_H, args.bitdepth)

    print("[INFO] Homography cost:")
    print(json.dumps(cost_H, indent=2))

    # ------------------------------------------------------------
    # Block residual MV search around H.
    # ------------------------------------------------------------

    block_res = estimate_block_residual_mv(
        target_y=target.y,
        ref_y=ref.y,
        map_x_H=map_x_H,
        map_y_H=map_y_H,
        valid_H=valid_H,
        block_size=args.block_size,
        search_range=args.res_search_range,
        search_step=args.res_search_step,
        min_block_valid_ratio=args.min_block_valid_ratio,
        min_texture=args.min_texture,
        min_gap=args.min_gap,
    )

    cost_block_mv = calc_cost(target.y, block_res.pred_block_mv, block_res.valid_block_mv, args.bitdepth)

    reliable_count = int(np.count_nonzero(block_res.reliable_grid))
    total_blocks = int(block_res.reliable_grid.size)

    print(f"[INFO] reliable block residuals = {reliable_count}/{total_blocks}")
    print("[INFO] Block residual MV upper-bound cost:")
    print(json.dumps(cost_block_mv, indent=2))

    # ------------------------------------------------------------
    # Fit global residual basis B(x).
    # ------------------------------------------------------------

    basis = fit_polynomial_residual_basis(
        dx_grid=block_res.dx_grid,
        dy_grid=block_res.dy_grid,
        reliable_grid=block_res.reliable_grid,
        cost_grid=block_res.cost_grid,
        gap_grid=block_res.gap_grid,
        texture_grid=block_res.texture_grid,
        width=args.width,
        height=args.height,
        block_size=args.block_size,
        degree=args.basis_degree,
        robust_iters=args.robust_fit_iters,
        min_fit_blocks=args.min_fit_blocks,
    )

    print(f"[INFO] basis fit blocks = {basis.num_fit_blocks}")
    print(f"[INFO] basis norm_scale = {basis.norm_scale}")

    # Initial c from projection of observed block residual onto basis.
    c_init = estimate_initial_c_grid_from_observed_residual(
        dx_grid=block_res.dx_grid,
        dy_grid=block_res.dy_grid,
        basis_x_blk=basis.basis_x_block,
        basis_y_blk=basis.basis_y_block,
        c_min=args.c_min,
        c_max=args.c_max,
    )

    # Refine c by actual block remap cost.
    scalar_pred = predict_with_basis_scalar(
        target_y=target.y,
        ref_y=ref.y,
        map_x_H=map_x_H,
        map_y_H=map_y_H,
        valid_H=valid_H,
        basis_x=basis.basis_x,
        basis_y=basis.basis_y,
        c_center_grid=c_init,
        block_size=args.block_size,
        c_refine_range=args.c_refine_range,
        c_samples=args.c_samples,
        c_min=args.c_min,
        c_max=args.c_max,
        min_block_valid_ratio=args.min_block_valid_ratio,
    )

    cost_scalar = calc_cost(target.y, scalar_pred.pred, scalar_pred.valid, args.bitdepth)

    print("[INFO] Basis scalar predictor cost:")
    print(json.dumps(cost_scalar, indent=2))

    # ------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------

    paths = {
        "target_yuv": os.path.join(args.output_dir, "target_pair.yuv"),
        "ref_yuv": os.path.join(args.output_dir, "ref_pair.yuv"),
        "pred_homography_yuv": os.path.join(args.output_dir, "pred_homography.yuv"),
        "pred_block_residual_mv_yuv": os.path.join(args.output_dir, "pred_block_residual_mv.yuv"),
        "pred_basis_scalar_yuv": os.path.join(args.output_dir, "pred_basis_scalar.yuv"),
    }

    write_single_yuv420(paths["target_yuv"], target.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["ref_yuv"], ref.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_homography_yuv"], pred_H, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_block_residual_mv_yuv"], block_res.pred_block_mv, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_basis_scalar_yuv"], scalar_pred.pred, args.width, args.height, args.bitdepth)

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_homography.png"), pred_H, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_block_residual_mv.png"), block_res.pred_block_mv, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_basis_scalar.png"), scalar_pred.pred, args.bitdepth)

    save_diff_png(os.path.join(args.output_dir, "diff_homography.png"), target.y, pred_H, valid_H)
    save_diff_png(os.path.join(args.output_dir, "diff_block_residual_mv.png"), target.y, block_res.pred_block_mv, block_res.valid_block_mv)
    save_diff_png(os.path.join(args.output_dir, "diff_basis_scalar.png"), target.y, scalar_pred.pred, scalar_pred.valid)

    save_scalar_map_png(
        os.path.join(args.output_dir, "c_map.png"),
        scalar_pred.c_grid,
        args.width,
        args.height,
        args.block_size,
    )

    save_scalar_map_png(
        os.path.join(args.output_dir, "block_residual_cost.png"),
        block_res.cost_grid,
        args.width,
        args.height,
        args.block_size,
    )

    save_scalar_map_png(
        os.path.join(args.output_dir, "block_residual_gap.png"),
        block_res.gap_grid,
        args.width,
        args.height,
        args.block_size,
    )

    save_vector_field_png(
        os.path.join(args.output_dir, "residual_mv_field.png"),
        block_res.dx_grid,
        block_res.dy_grid,
        args.width,
        args.height,
        args.block_size,
        scale=4.0,
    )

    save_basis_field_png(
        os.path.join(args.output_dir, "basis_field.png"),
        basis.basis_x,
        basis.basis_y,
        stride=max(16, args.block_size * 2),
        scale=16.0,
    )

    # ------------------------------------------------------------
    # JSON.
    # ------------------------------------------------------------

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "model": {
            "description": "image-space factorized warp",
            "formula": "p_ref(x,b) = H_g(x) + c_b * B(x)",
            "global": "H_g and polynomial residual basis B(x)",
            "block": "scalar c_b",
            "note": "This is not physical R|t/depth, but syntax/structure is similar: global info + block constant.",
        },

        "homography": {
            "H_initial_target_to_ref": H_init.tolist(),
            "H_refined_target_to_ref": H_refined.tolist(),
            "num_good_matches": int(len(match_result.good_matches)),
            "num_H_inliers": int(H_inliers),
            "ecc": ecc_meta,
        },

        "block_residual_search": {
            "block_size": int(args.block_size),
            "res_search_range": float(args.res_search_range),
            "res_search_step": float(args.res_search_step),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
            "min_texture": float(args.min_texture),
            "min_gap": float(args.min_gap),
            "reliable_blocks": reliable_count,
            "total_blocks": total_blocks,
            "dx_grid": block_res.dx_grid.tolist(),
            "dy_grid": block_res.dy_grid.tolist(),
            "cost_grid": block_res.cost_grid.tolist(),
            "gap_grid": block_res.gap_grid.tolist(),
            "valid_ratio_grid": block_res.valid_ratio_grid.tolist(),
            "texture_grid": block_res.texture_grid.tolist(),
            "reliable_grid": block_res.reliable_grid.astype(int).tolist(),
        },

        "basis": {
            "degree": int(args.basis_degree),
            "robust_fit_iters": int(args.robust_fit_iters),
            "num_fit_blocks": int(basis.num_fit_blocks),
            "norm_scale": float(basis.norm_scale),
            "coef_x": basis.coef_x.tolist(),
            "coef_y": basis.coef_y.tolist(),
            "fit_error_grid": basis.fit_error_grid.tolist(),
        },

        "scalar_c": {
            "c_min": float(args.c_min),
            "c_max": float(args.c_max),
            "c_samples": int(args.c_samples),
            "c_refine_range": float(args.c_refine_range),
            "c_grid": scalar_pred.c_grid.tolist(),
            "cost_grid": scalar_pred.cost_grid.tolist(),
            "valid_ratio_grid": scalar_pred.valid_ratio_grid.tolist(),
        },

        "costs": {
            "homography": cost_H,
            "block_residual_mv_upper_bound": cost_block_mv,
            "basis_scalar": cost_scalar,
        },

        "outputs": paths,

        "png_outputs": {
            "match_vis": os.path.join(args.output_dir, "match_vis_homography_inliers.png"),
            "target": os.path.join(args.output_dir, "target.png"),
            "ref": os.path.join(args.output_dir, "ref.png"),
            "pred_homography": os.path.join(args.output_dir, "pred_homography.png"),
            "pred_block_residual_mv": os.path.join(args.output_dir, "pred_block_residual_mv.png"),
            "pred_basis_scalar": os.path.join(args.output_dir, "pred_basis_scalar.png"),
            "diff_homography": os.path.join(args.output_dir, "diff_homography.png"),
            "diff_block_residual_mv": os.path.join(args.output_dir, "diff_block_residual_mv.png"),
            "diff_basis_scalar": os.path.join(args.output_dir, "diff_basis_scalar.png"),
            "c_map": os.path.join(args.output_dir, "c_map.png"),
            "residual_mv_field": os.path.join(args.output_dir, "residual_mv_field.png"),
            "residual_mv_magnitude": os.path.join(args.output_dir, "residual_mv_field.png.mag.png"),
            "basis_field": os.path.join(args.output_dir, "basis_field.png"),
        },

        "interpretation": [
            "pred_homography.yuv: baseline global homography predictor.",
            "pred_block_residual_mv.yuv: upper bound using independent block residual MV around H.",
            "pred_basis_scalar.yuv: actual proposed factorized predictor, H + c_b * B(x).",
            "If block_residual_mv is much better but basis_scalar is not, one basis is insufficient; try degree=2 or later multi-basis.",
            "If basis_scalar is close to block_residual_mv, global basis + block scalar is promising.",
            "If homography is already best, residual MV may be fitting noise or moving objects.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] result JSON: {json_path}")
    print(f"[DONE] target YUV: {paths['target_yuv']}")
    print(f"[DONE] ref YUV: {paths['ref_yuv']}")
    print(f"[DONE] homography YUV: {paths['pred_homography_yuv']}")
    print(f"[DONE] block residual MV YUV: {paths['pred_block_residual_mv_yuv']}")
    print(f"[DONE] basis scalar YUV: {paths['pred_basis_scalar_yuv']}")


if __name__ == "__main__":
    main()
