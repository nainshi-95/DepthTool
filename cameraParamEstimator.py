#!/usr/bin/env python3
# pair_camera_coarse_depth_search.py
#
# Pairwise camera-parameter-first coarse-to-fine search.
#
# Goal:
#   Find camera parameters that produce globally natural alignment.
#   Depth C is treated as a coarse nuisance/block variable, not as an accurate depth map.
#
# Camera model:
#   ray_t = K^-1 * p_target
#   X_ref = R_ref_target * ray_t + c_b * t_ref_target
#   p_ref = K * X_ref
#
# Search objective:
#   Score(R, t, f) = sum_b min_{c_b in coarse C candidates} D_b(R, t, f, c_b)
#
# Important:
#   - R/f/t are refined more carefully.
#   - C is kept very coarse and large-block-based.
#   - This avoids scaling full homography flow, so C does not scale rotation.
#
# Example:
#   python pair_camera_coarse_depth_search.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --search-scale 0.25 \
#     --search-iters 4 \
#     --search-block-size 128 \
#     --final-block-size 64 \
#     --f-rel-list 0.8,1.0,1.2,1.5 \
#     --r-step-deg 1.0 \
#     --r-grid 3 \
#     --c-list 0,0.02,0.08 \
#     --max-features 20000 \
#     --match-ratio 0.70 \
#     --ransac-thresh 1.0 \
#     --h-refine-iters 100 \
#     --h-refine-scale 0.5 \
#     --output-dir cam_search_t1_r0

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
class PoseCandidateResult:
    score: float
    mae: float
    valid_ratio: float
    R: np.ndarray
    t: np.ndarray
    f_full: float
    c_block_grid: np.ndarray
    c_block_cost_grid: np.ndarray
    pred: Optional[np.ndarray]
    valid: Optional[np.ndarray]


# ============================================================
# I/O utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def yuv420_frame_size_bytes(width: int, height: int, bitdepth: int) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    samples = width * height + 2 * ((width // 2) * (height // 2))

    if bitdepth == 8:
        return samples
    if bitdepth == 10:
        return samples * 2

    raise ValueError("Only 8-bit and 10-bit YUV420 are supported.")


def read_y_frame(path: str, width: int, height: int, bitdepth: int, frame_idx: int) -> FrameY:
    frame_size = yuv420_frame_size_bytes(width, height, bitdepth)
    y_samples = width * height
    offset = frame_idx * frame_size

    file_size = os.path.getsize(path)
    if offset + frame_size > file_size:
        raise ValueError(
            f"frame_idx={frame_idx} is out of range. "
            f"Need {offset + frame_size} bytes, file size is {file_size}."
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


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


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
# Camera model
# ============================================================

def build_K(width: int, height: int, f: float) -> np.ndarray:
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5

    return np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    return R.astype(np.float64)


def rotation_from_homography(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    R_approx = np.linalg.inv(K) @ H @ K

    det = np.linalg.det(R_approx)
    if abs(det) > 1e-12:
        scale = np.cbrt(abs(det))
        R_approx = R_approx / scale

    return project_to_so3(R_approx)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_delta_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    # yaw: Y-axis, pitch: X-axis, roll: Z-axis
    return rot_z(roll_deg) @ rot_y(yaw_deg) @ rot_x(pitch_deg)


def normalize_t(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(t))

    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)

    return t / n


def estimate_essential_pose(
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    meta = {
        "success": False,
        "inliers": 0,
        "error": None,
    }

    try:
        E, mask = cv2.findEssentialMat(
            pts_target,
            pts_ref,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh,
        )

        if E is None:
            meta["error"] = "findEssentialMat returned None"
            return None, None, meta

        _, R, t, pose_mask = cv2.recoverPose(E, pts_target, pts_ref, K)

        meta["success"] = True
        meta["inliers"] = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0

        return R.astype(np.float64), normalize_t(t.reshape(3)), meta

    except cv2.error as e:
        meta["error"] = str(e)
        return None, None, meta


def unique_t_candidates(cands: List[np.ndarray], eps: float = 1e-5) -> List[np.ndarray]:
    out = []

    for t in cands:
        t = normalize_t(t)

        duplicate = False
        for u in out:
            if np.linalg.norm(t - u) < eps:
                duplicate = True
                break

        if not duplicate:
            out.append(t)

    return out


def build_initial_t_candidates(
    essential_t: Optional[np.ndarray],
    include_axes: bool,
    include_zero: bool,
) -> List[np.ndarray]:
    cands = []

    if essential_t is not None:
        cands.append(essential_t)
        cands.append(-essential_t)

    if include_axes:
        cands += [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

    if include_zero:
        cands.append(np.zeros(3, dtype=np.float64))

    return unique_t_candidates(cands)


def perturb_t_candidates(t_center: np.ndarray, step_deg: float, grid: int) -> List[np.ndarray]:
    t_center = normalize_t(t_center)

    if step_deg <= 0.0 or np.linalg.norm(t_center) < 1e-12:
        return [t_center]

    offsets = np.linspace(-step_deg, step_deg, grid)

    cands = []
    for ax in offsets:
        for ay in offsets:
            for az in offsets:
                R_delta = make_delta_R(yaw_deg=ay, pitch_deg=ax, roll_deg=az)
                cands.append(normalize_t(R_delta @ t_center))

    return unique_t_candidates(cands)


def make_rays(width: int, height: int, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    rx = (xs.astype(np.float32) - cx) / fx
    ry = (ys.astype(np.float32) - cy) / fy
    rz = np.ones_like(rx, dtype=np.float32)

    return rx, ry, rz


def camera_maps_from_rays(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    c: float,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx, ry, rz = rays

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    t = np.asarray(t, dtype=np.float64).reshape(3)

    X = (
        R[0, 0] * rx
        + R[0, 1] * ry
        + R[0, 2] * rz
        + float(c) * t[0]
    )
    Y = (
        R[1, 0] * rx
        + R[1, 1] * ry
        + R[1, 2] * rz
        + float(c) * t[1]
    )
    Z = (
        R[2, 0] * rx
        + R[2, 1] * ry
        + R[2, 2] * rz
        + float(c) * t[2]
    )

    valid_z = Z > 1e-6
    Z_safe = Z + 1e-12

    map_x = fx * (X / Z_safe) + cx
    map_y = fy * (Y / Z_safe) + cy

    h, w = rx.shape

    valid = (
        valid_z
        & (map_x >= 0.0)
        & (map_x <= w - 1.0)
        & (map_y >= 0.0)
        & (map_y <= h - 1.0)
    )

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def remap_full(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# Scoring
# ============================================================

def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def calc_pose_score(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    K: np.ndarray,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    f_full: float,
    c_list: List[float],
    block_size: int,
    min_block_valid_ratio: float,
    c_smooth_lambda: float,
    return_pred: bool = False,
) -> PoseCandidateResult:
    h, w = target_y.shape
    target_f = target_y.astype(np.float32)

    ny, nx = block_grid_shape(w, h, block_size)

    best_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    best_c_grid = np.zeros((ny, nx), dtype=np.float64)
    best_valid_grid = np.zeros((ny, nx), dtype=np.float64)

    out_pred = np.zeros((h, w), dtype=np.float32) if return_pred else None
    out_valid = np.zeros((h, w), dtype=bool) if return_pred else None

    for c in c_list:
        map_x, map_y, valid = camera_maps_from_rays(K, R, t, c, rays)
        pred = remap_full(ref_y, map_x, map_y)

        diff = np.abs(target_f - pred)

        for by_idx, by in enumerate(range(0, h, block_size)):
            y1 = min(by + block_size, h)

            for bx_idx, bx in enumerate(range(0, w, block_size)):
                x1 = min(bx + block_size, w)

                v = valid[by:y1, bx:x1]
                valid_ratio = float(np.mean(v))

                if valid_ratio < min_block_valid_ratio or not np.any(v):
                    cost = np.inf
                else:
                    cost = float(np.mean(diff[by:y1, bx:x1][v]))

                if cost < best_cost_grid[by_idx, bx_idx]:
                    best_cost_grid[by_idx, bx_idx] = cost
                    best_c_grid[by_idx, bx_idx] = float(c)
                    best_valid_grid[by_idx, bx_idx] = valid_ratio

                    if return_pred:
                        out_pred[by:y1, bx:x1] = pred[by:y1, bx:x1]
                        out_valid[by:y1, bx:x1] = v

    finite = np.isfinite(best_cost_grid)

    if not np.any(finite):
        score = float("inf")
        mae = float("inf")
        valid_ratio_total = 0.0
    else:
        mae = float(np.mean(best_cost_grid[finite]))
        valid_ratio_total = float(np.mean(best_valid_grid[finite]))
        score = mae

        if c_smooth_lambda > 0.0:
            smooth_terms = []

            if best_c_grid.shape[1] > 1:
                smooth_terms.append(np.abs(np.diff(best_c_grid, axis=1)))

            if best_c_grid.shape[0] > 1:
                smooth_terms.append(np.abs(np.diff(best_c_grid, axis=0)))

            if smooth_terms:
                smooth = float(np.mean([np.mean(x) for x in smooth_terms]))
                score += float(c_smooth_lambda) * smooth

    return PoseCandidateResult(
        score=float(score),
        mae=float(mae),
        valid_ratio=float(valid_ratio_total),
        R=R.astype(np.float64),
        t=normalize_t(t),
        f_full=float(f_full),
        c_block_grid=best_c_grid,
        c_block_cost_grid=best_cost_grid,
        pred=out_pred,
        valid=out_valid,
    )


def make_rotation_offsets(step_deg: float, grid: int) -> np.ndarray:
    if grid <= 1 or step_deg <= 0.0:
        return np.array([0.0], dtype=np.float64)

    return np.linspace(-step_deg, step_deg, grid, dtype=np.float64)


def candidate_f_values_iter0(width: int, f_rel_list: List[float]) -> List[float]:
    return [float(fr * width) for fr in f_rel_list]


def candidate_f_values_refine(f_center: float, f_step_rel: float, grid: int) -> List[float]:
    if grid <= 1 or f_step_rel <= 0.0:
        return [float(f_center)]

    offsets = np.linspace(-f_step_rel, f_step_rel, grid, dtype=np.float64)
    vals = [float(f_center * (1.0 + off)) for off in offsets]
    vals = [max(1.0, v) for v in vals]
    return vals


# ============================================================
# Search
# ============================================================

def run_camera_search(
    target_s: np.ndarray,
    ref_s: np.ndarray,
    full_width: int,
    full_height: int,
    scale: float,
    H_refined_full: np.ndarray,
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    args,
) -> Tuple[PoseCandidateResult, List[dict]]:
    search_h, search_w = target_s.shape

    c_list = parse_float_list(args.c_list)
    f_rel_list = parse_float_list(args.f_rel_list)

    # Essential pose from center/default focal.
    f0_full = float(args.essential_f_rel * full_width)
    K0_full = build_K(full_width, full_height, f0_full)

    R_E, t_E, essential_meta = estimate_essential_pose(
        pts_target=pts_target,
        pts_ref=pts_ref,
        K=K0_full,
        ransac_thresh=args.essential_thresh,
    )

    print("[INFO] Essential meta:")
    print(json.dumps(essential_meta, indent=2))

    fixed_t_candidates = build_initial_t_candidates(
        essential_t=t_E,
        include_axes=args.include_axis_t,
        include_zero=args.include_zero_t,
    )

    if not fixed_t_candidates:
        fixed_t_candidates = [np.zeros(3, dtype=np.float64)]

    best: Optional[PoseCandidateResult] = None
    history = []

    r_step = float(args.r_step_deg)
    f_step_rel = float(args.f_step_rel)
    t_step = float(args.t_step_deg)

    for it in range(args.search_iters):
        print(f"[SEARCH] iter={it}")

        if it == 0 or best is None:
            f_values_full = candidate_f_values_iter0(full_width, f_rel_list)
            t_candidates = fixed_t_candidates
        else:
            f_values_full = candidate_f_values_refine(best.f_full, f_step_rel, args.f_grid)

            if args.refine_t and t_step > 0.0:
                t_candidates = perturb_t_candidates(best.t, t_step, args.t_grid)
                # Keep zero fallback if enabled.
                if args.include_zero_t:
                    t_candidates.append(np.zeros(3, dtype=np.float64))
                t_candidates = unique_t_candidates(t_candidates)
            else:
                t_candidates = [best.t]

        r_offsets = make_rotation_offsets(r_step, args.r_grid)

        iter_best: Optional[PoseCandidateResult] = None
        eval_count = 0

        for f_full in f_values_full:
            f_search = float(f_full * scale)
            K_search = build_K(search_w, search_h, f_search)
            rays = make_rays(search_w, search_h, K_search)

            # For iteration 0, derive an R center from H for each focal candidate.
            # Later, refine around the previous best R.
            R_centers = []

            if it == 0 or best is None:
                K_full = build_K(full_width, full_height, f_full)
                R_H = rotation_from_homography(H_refined_full, K_full)
                R_centers.append(R_H)

                if R_E is not None:
                    R_centers.append(R_E)

                R_centers = [project_to_so3(R) for R in R_centers]
            else:
                R_centers.append(best.R)

            # Remove duplicate R centers roughly by matrix distance.
            unique_R_centers = []
            for R in R_centers:
                duplicate = False
                for U in unique_R_centers:
                    if np.linalg.norm(R - U) < 1e-5:
                        duplicate = True
                        break
                if not duplicate:
                    unique_R_centers.append(R)

            for R_center in unique_R_centers:
                for yaw in r_offsets:
                    for pitch in r_offsets:
                        for roll in r_offsets:
                            R = make_delta_R(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll) @ R_center
                            R = project_to_so3(R)

                            for t in t_candidates:
                                res = calc_pose_score(
                                    target_y=target_s,
                                    ref_y=ref_s,
                                    K=K_search,
                                    rays=rays,
                                    R=R,
                                    t=t,
                                    f_full=f_full,
                                    c_list=c_list,
                                    block_size=max(4, int(round(args.search_block_size * scale))),
                                    min_block_valid_ratio=args.min_block_valid_ratio,
                                    c_smooth_lambda=args.c_smooth_lambda,
                                    return_pred=False,
                                )

                                eval_count += 1

                                if iter_best is None or res.score < iter_best.score:
                                    iter_best = res

        if iter_best is None:
            raise RuntimeError("Search produced no valid candidate.")

        best = iter_best

        iter_record = {
            "iter": int(it),
            "eval_count": int(eval_count),
            "best_score": float(best.score),
            "best_mae": float(best.mae),
            "best_valid_ratio": float(best.valid_ratio),
            "best_f_full": float(best.f_full),
            "best_f_rel_to_width": float(best.f_full / full_width),
            "best_t": best.t.tolist(),
            "best_R": best.R.tolist(),
            "r_step_deg": float(r_step),
            "f_step_rel": float(f_step_rel),
            "t_step_deg": float(t_step),
            "num_t_candidates": int(len(t_candidates)),
            "num_f_candidates": int(len(f_values_full)),
        }

        history.append(iter_record)

        print(json.dumps(iter_record, indent=2))

        r_step *= float(args.r_shrink)
        f_step_rel *= float(args.f_shrink)
        t_step *= float(args.t_shrink)

    return best, history


# ============================================================
# Final output
# ============================================================

def calc_basic_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray) -> dict:
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

    maxv = 255.0 if target_y.dtype == np.uint8 else 1023.0
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


def save_c_map_png(path: str, c_block_grid: np.ndarray, out_w: int, out_h: int, block_size: int):
    ny, nx = c_block_grid.shape
    c_img = np.zeros((out_h, out_w), dtype=np.float32)

    for by_idx in range(ny):
        by = by_idx * block_size
        y1 = min(by + block_size, out_h)

        for bx_idx in range(nx):
            bx = bx_idx * block_size
            x1 = min(bx + block_size, out_w)
            c_img[by:y1, bx:x1] = float(c_block_grid[by_idx, bx_idx])

    c_min = float(np.min(c_img))
    c_max = float(np.max(c_img))

    if abs(c_max - c_min) < 1e-12:
        c8 = np.full_like(c_img, 128, dtype=np.uint8)
    else:
        c8 = np.clip((c_img - c_min) / (c_max - c_min) * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(c8, cv2.COLORMAP_TURBO)
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

    # Matching / H init
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--no-clahe", action="store_true")

    # ECC H refinement
    parser.add_argument("--h-refine-iters", type=int, default=100)
    parser.add_argument("--h-refine-eps", type=float, default=1e-7)
    parser.add_argument("--h-refine-scale", type=float, default=0.5)
    parser.add_argument("--h-refine-blur", type=int, default=5)

    # Search
    parser.add_argument("--search-scale", type=float, default=0.25)
    parser.add_argument("--search-iters", type=int, default=4)

    parser.add_argument("--search-block-size", type=int, default=128)
    parser.add_argument("--final-block-size", type=int, default=64)

    parser.add_argument("--f-rel-list", default="0.8,1.0,1.2,1.5")
    parser.add_argument("--f-grid", type=int, default=3)
    parser.add_argument("--f-step-rel", type=float, default=0.10)
    parser.add_argument("--f-shrink", type=float, default=0.5)

    parser.add_argument("--r-step-deg", type=float, default=1.0)
    parser.add_argument("--r-grid", type=int, default=3)
    parser.add_argument("--r-shrink", type=float, default=0.5)

    parser.add_argument("--essential-f-rel", type=float, default=1.0)
    parser.add_argument("--essential-thresh", type=float, default=1.0)
    parser.add_argument("--include-axis-t", action="store_true")
    parser.add_argument("--include-zero-t", action="store_true", default=True)

    parser.add_argument("--refine-t", action="store_true")
    parser.add_argument("--t-step-deg", type=float, default=2.0)
    parser.add_argument("--t-grid", type=int, default=3)
    parser.add_argument("--t-shrink", type=float, default=0.5)

    parser.add_argument("--c-list", default="0,0.02,0.08")
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)
    parser.add_argument("--c-smooth-lambda", type=float, default=0.0)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.search_scale <= 0.0 or args.search_scale > 1.0:
        raise ValueError("--search-scale must be in (0, 1].")

    print(f"[INFO] target_idx={args.target_idx}, ref_idx={args.ref_idx}")
    print(f"[INFO] resolution={args.width}x{args.height}, bitdepth={args.bitdepth}")
    print(f"[INFO] search_scale={args.search_scale}, search_iters={args.search_iters}")

    target = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.ref_idx)

    # ------------------------------------------------------------
    # Match + H init + ECC refine.
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

    # ------------------------------------------------------------
    # Search images.
    # ------------------------------------------------------------

    target_s = resize_frame(target.y, args.search_scale)
    ref_s = resize_frame(ref.y, args.search_scale)

    # ------------------------------------------------------------
    # Camera-first search.
    # ------------------------------------------------------------

    best_search, search_history = run_camera_search(
        target_s=target_s,
        ref_s=ref_s,
        full_width=args.width,
        full_height=args.height,
        scale=args.search_scale,
        H_refined_full=H_refined,
        pts_target=match_result.pts_target,
        pts_ref=match_result.pts_ref,
        args=args,
    )

    print("[INFO] Best search result:")
    print(
        json.dumps(
            {
                "score": best_search.score,
                "mae": best_search.mae,
                "valid_ratio": best_search.valid_ratio,
                "f_full": best_search.f_full,
                "f_rel_to_width": best_search.f_full / args.width,
                "t": best_search.t.tolist(),
                "R": best_search.R.tolist(),
            },
            indent=2,
        )
    )

    # ------------------------------------------------------------
    # Final full-res evaluation/output.
    # ------------------------------------------------------------

    K_full = build_K(args.width, args.height, best_search.f_full)
    rays_full = make_rays(args.width, args.height, K_full)

    c_list = parse_float_list(args.c_list)

    # R-only / c=0 baseline.
    map_x0, map_y0, valid0 = camera_maps_from_rays(
        K_full,
        best_search.R,
        best_search.t,
        0.0,
        rays_full,
    )
    pred_c0 = remap_full(ref.y, map_x0, map_y0)
    cost_c0 = calc_basic_cost(target.y, pred_c0, valid0)

    # Final block-wise coarse C.
    final_res = calc_pose_score(
        target_y=target.y,
        ref_y=ref.y,
        K=K_full,
        rays=rays_full,
        R=best_search.R,
        t=best_search.t,
        f_full=best_search.f_full,
        c_list=c_list,
        block_size=args.final_block_size,
        min_block_valid_ratio=args.min_block_valid_ratio,
        c_smooth_lambda=args.c_smooth_lambda,
        return_pred=True,
    )

    cost_block = calc_basic_cost(target.y, final_res.pred, final_res.valid)

    print("[INFO] Final c=0 cost:")
    print(json.dumps(cost_c0, indent=2))
    print("[INFO] Final block-C cost:")
    print(json.dumps(cost_block, indent=2))

    # ------------------------------------------------------------
    # Save files.
    # ------------------------------------------------------------

    paths = {
        "target_yuv": os.path.join(args.output_dir, "target_pair.yuv"),
        "ref_yuv": os.path.join(args.output_dir, "ref_pair.yuv"),
        "pred_c0_yuv": os.path.join(args.output_dir, "pred_camera_c0.yuv"),
        "pred_block_c_yuv": os.path.join(args.output_dir, f"pred_camera_block{args.final_block_size}C.yuv"),
    }

    write_single_yuv420(paths["target_yuv"], target.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["ref_yuv"], ref.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_c0_yuv"], pred_c0, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_block_c_yuv"], final_res.pred, args.width, args.height, args.bitdepth)

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_camera_c0.png"), pred_c0, args.bitdepth)
    save_gray_png(
        os.path.join(args.output_dir, f"pred_camera_block{args.final_block_size}C.png"),
        final_res.pred,
        args.bitdepth,
    )

    save_diff_png(os.path.join(args.output_dir, "diff_camera_c0.png"), target.y, pred_c0, valid0)
    save_diff_png(
        os.path.join(args.output_dir, f"diff_camera_block{args.final_block_size}C.png"),
        target.y,
        final_res.pred,
        final_res.valid,
    )

    save_c_map_png(
        os.path.join(args.output_dir, f"c_map_block{args.final_block_size}.png"),
        final_res.c_block_grid,
        args.width,
        args.height,
        args.final_block_size,
    )

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "format": "yuv420p" if args.bitdepth == 8 else "yuv420p10le",

        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "model": "p_ref = project(K * (R * ray_target + c_block * t))",
        "search_objective": "Score(R,t,f)=sum_blocks min_c block_MAE",

        "matching": {
            "num_good_matches": int(len(match_result.good_matches)),
            "homography_inliers": int(H_inliers),
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "ransac_thresh": float(args.ransac_thresh),
            "clahe": bool(not args.no_clahe),
        },

        "homography": {
            "H_initial": H_init.tolist(),
            "H_refined": H_refined.tolist(),
            "ecc": ecc_meta,
        },

        "best_camera": {
            "f_full": float(best_search.f_full),
            "f_rel_to_width": float(best_search.f_full / args.width),
            "K": K_full.tolist(),
            "R_ref_target": best_search.R.tolist(),
            "t_ref_target_unit": best_search.t.tolist(),
            "search_score": float(best_search.score),
            "search_mae": float(best_search.mae),
            "search_valid_ratio": float(best_search.valid_ratio),
        },

        "search_args": {
            "search_scale": float(args.search_scale),
            "search_iters": int(args.search_iters),
            "search_block_size": int(args.search_block_size),
            "final_block_size": int(args.final_block_size),
            "f_rel_list": parse_float_list(args.f_rel_list),
            "f_grid": int(args.f_grid),
            "f_step_rel": float(args.f_step_rel),
            "f_shrink": float(args.f_shrink),
            "r_step_deg": float(args.r_step_deg),
            "r_grid": int(args.r_grid),
            "r_shrink": float(args.r_shrink),
            "refine_t": bool(args.refine_t),
            "t_step_deg": float(args.t_step_deg),
            "t_grid": int(args.t_grid),
            "t_shrink": float(args.t_shrink),
            "c_list": c_list,
            "c_smooth_lambda": float(args.c_smooth_lambda),
        },

        "search_history": search_history,

        "final_costs": {
            "camera_c0": cost_c0,
            "camera_block_c": cost_block,
        },

        "final_c_block_grid": final_res.c_block_grid.tolist(),
        "final_c_block_cost_grid": final_res.c_block_cost_grid.tolist(),

        "outputs": paths,

        "png_outputs": {
            "match_vis": os.path.join(args.output_dir, "match_vis_homography_inliers.png"),
            "target_png": os.path.join(args.output_dir, "target.png"),
            "ref_png": os.path.join(args.output_dir, "ref.png"),
            "pred_c0_png": os.path.join(args.output_dir, "pred_camera_c0.png"),
            "pred_block_c_png": os.path.join(args.output_dir, f"pred_camera_block{args.final_block_size}C.png"),
            "diff_c0_png": os.path.join(args.output_dir, "diff_camera_c0.png"),
            "diff_block_c_png": os.path.join(args.output_dir, f"diff_camera_block{args.final_block_size}C.png"),
            "c_map_png": os.path.join(args.output_dir, f"c_map_block{args.final_block_size}.png"),
        },

        "notes": [
            "Depth C is intentionally coarse and used only as a nuisance variable during camera search.",
            "R/f/t receive more search budget than C.",
            "C is applied only to the translation/parallax term, not to the rotation term.",
            "pred_camera_c0.yuv shows the selected R-only camera warp.",
            "pred_camera_block*C.yuv shows the selected R/t camera warp with block-wise coarse C.",
            "If pred_camera_c0 is already good, the scene is rotation-dominant.",
            "If block C improves significantly, translation/depth parallax is useful.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] result JSON: {json_path}")
    print(f"[DONE] target YUV: {paths['target_yuv']}")
    print(f"[DONE] c0 YUV: {paths['pred_c0_yuv']}")
    print(f"[DONE] block C YUV: {paths['pred_block_c_yuv']}")


if __name__ == "__main__":
    main()
