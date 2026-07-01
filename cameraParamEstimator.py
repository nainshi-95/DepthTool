#!/usr/bin/env python3
# pair_camera_depth_iterative_search.py
#
# Pairwise iterative camera parameter + coarse block-depth search.
#
# Main idea:
#   Camera parameter search is more important than accurate depth.
#   Depth C is used as a coarse nuisance variable to help camera parameter selection.
#
# Camera model:
#   p_t    = [x, y, 1]^T
#   ray_t  = K^-1 p_t
#   X_ref  = R_ref_target * ray_t + c_b * t_ref_target
#   p_ref  = K * X_ref
#
# Search objective:
#   Score(R, t, f)
#     = sum_b min_{c_b in local C candidates}
#         MAE(target_b, remap(ref, R, t, f, c_b))
#
# Iterative refinement:
#   - Camera params:
#       R/f/t search ranges shrink every iteration.
#   - Depth map:
#       Each block keeps its own c_center.
#       After each iteration, c_center_map is updated from the best camera candidate.
#       c_half_range shrinks every iteration.
#
# Initial pose candidates:
#   - Homography decomposition candidates: cv2.decomposeHomographyMat(H, K)
#   - Rotation-only candidate from R ~= K^-1 H K
#   - Essential matrix candidates
#   - Optional axis t candidates and zero-t candidate
#
# Output:
#   target_pair.yuv
#   ref_pair.yuv
#   pred_camera_c0.yuv
#   pred_camera_blockC.yuv
#   c_map png
#   result.json
#
# Example:
#   python pair_camera_depth_iterative_search.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --search-scale 0.25 \
#     --search-iters 5 \
#     --search-block-size 128 \
#     --final-block-size 64 \
#     --f-rel-list 0.7,0.9,1.1,1.4 \
#     --f-grid 3 \
#     --f-step-rel 0.12 \
#     --f-shrink 0.5 \
#     --r-step-deg 1.5 \
#     --r-grid 3 \
#     --r-shrink 0.5 \
#     --c-min 0.0 \
#     --c-max 0.02 \
#     --c-samples 3 \
#     --c-shrink 0.5 \
#     --final-c-iters 3 \
#     --max-features 20000 \
#     --match-ratio 0.70 \
#     --ransac-thresh 1.0 \
#     --h-refine-iters 100 \
#     --h-refine-scale 0.5 \
#     --include-axis-t \
#     --output-dir iter_cam_t1_r0

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
class SeedPose:
    R: np.ndarray
    t: np.ndarray
    name: str


@dataclass
class EvalResult:
    score: float
    mae: float
    valid_block_ratio: float
    valid_pixel_ratio_mean: float
    R: np.ndarray
    t: np.ndarray
    f_full: float
    c_grid: np.ndarray
    cost_grid: np.ndarray
    valid_grid: np.ndarray
    pred: Optional[np.ndarray]
    valid: Optional[np.ndarray]


# ============================================================
# Basic utilities
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

    raise ValueError("Only bitdepth 8 and 10 are supported.")


def read_y_frame(path: str, width: int, height: int, bitdepth: int, frame_idx: int) -> FrameY:
    frame_size = yuv420_frame_size_bytes(width, height, bitdepth)
    y_samples = width * height
    offset = frame_idx * frame_size

    file_size = os.path.getsize(path)
    if offset + frame_size > file_size:
        raise ValueError(
            f"Frame index {frame_idx} is out of range. "
            f"Need byte offset {offset + frame_size}, file size is {file_size}."
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
# Camera model utilities
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
        R_approx = R_approx / np.cbrt(abs(det))

    return project_to_so3(R_approx)


def normalize_t(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(t))

    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)

    return t / n


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


def matrix_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B))


def unique_seed_poses(seeds: List[SeedPose], r_eps: float = 1e-5, t_eps: float = 1e-5) -> List[SeedPose]:
    out = []

    for s in seeds:
        dup = False
        for u in out:
            if matrix_distance(s.R, u.R) < r_eps and np.linalg.norm(s.t - u.t) < t_eps:
                dup = True
                break

        if not dup:
            out.append(s)

    return out


def estimate_essential_pose(
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    K: np.ndarray,
    thresh: float,
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
            threshold=thresh,
        )

        if E is None:
            meta["error"] = "findEssentialMat returned None"
            return None, None, meta

        _, R, t, pose_mask = cv2.recoverPose(E, pts_target, pts_ref, K)

        meta["success"] = True
        meta["inliers"] = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0

        return project_to_so3(R), normalize_t(t.reshape(3)), meta

    except cv2.error as e:
        meta["error"] = str(e)
        return None, None, meta


def decompose_homography_candidates(H: np.ndarray, K: np.ndarray) -> List[SeedPose]:
    seeds = []

    try:
        num, Rs, ts, normals = cv2.decomposeHomographyMat(
            H.astype(np.float64),
            K.astype(np.float64),
        )

        for i in range(num):
            R = project_to_so3(Rs[i])
            t = normalize_t(ts[i].reshape(3))

            if np.linalg.norm(t) > 1e-12:
                seeds.append(SeedPose(R=R, t=t, name=f"H_decomp_{i}_plus"))
                seeds.append(SeedPose(R=R, t=-t, name=f"H_decomp_{i}_minus"))
            else:
                seeds.append(SeedPose(R=R, t=t, name=f"H_decomp_{i}_zero"))

    except cv2.error:
        pass

    return seeds


def build_axis_t_candidates(include_zero: bool) -> List[np.ndarray]:
    out = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ]

    if include_zero:
        out.append(np.zeros(3, dtype=np.float64))

    return out


def perturb_t_candidates(t_center: np.ndarray, step_deg: float, grid: int, include_zero: bool) -> List[np.ndarray]:
    t_center = normalize_t(t_center)

    if np.linalg.norm(t_center) < 1e-12 or step_deg <= 0.0 or grid <= 1:
        cands = [t_center]
        if include_zero:
            cands.append(np.zeros(3, dtype=np.float64))
        return unique_t_vectors(cands)

    offsets = np.linspace(-step_deg, step_deg, grid, dtype=np.float64)

    cands = []
    for yaw in offsets:
        for pitch in offsets:
            for roll in offsets:
                R_delta = make_delta_R(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
                cands.append(normalize_t(R_delta @ t_center))

    if include_zero:
        cands.append(np.zeros(3, dtype=np.float64))

    return unique_t_vectors(cands)


def unique_t_vectors(cands: List[np.ndarray], eps: float = 1e-5) -> List[np.ndarray]:
    out = []

    for t in cands:
        t = normalize_t(t)

        dup = False
        for u in out:
            if np.linalg.norm(t - u) < eps:
                dup = True
                break

        if not dup:
            out.append(t)

    return out


def make_rays(width: int, height: int, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    rx = (xs - cx) / fx
    ry = (ys - cy) / fy
    rz = np.ones_like(rx, dtype=np.float32)

    return rx, ry, rz


def camera_maps_roi(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    c: float,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    bx: int,
    by: int,
    bw: int,
    bh: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx, ry, rz = rays

    rr_x = rx[by:by + bh, bx:bx + bw]
    rr_y = ry[by:by + bh, bx:bx + bw]
    rr_z = rz[by:by + bh, bx:bx + bw]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    t = np.asarray(t, dtype=np.float64).reshape(3)

    X = (
        R[0, 0] * rr_x
        + R[0, 1] * rr_y
        + R[0, 2] * rr_z
        + float(c) * t[0]
    )
    Y = (
        R[1, 0] * rr_x
        + R[1, 1] * rr_y
        + R[1, 2] * rr_z
        + float(c) * t[1]
    )
    Z = (
        R[2, 0] * rr_x
        + R[2, 1] * rr_y
        + R[2, 2] * rr_z
        + float(c) * t[2]
    )

    valid_z = Z > 1e-6
    Z_safe = Z + 1e-12

    map_x = fx * (X / Z_safe) + cx
    map_y = fy * (Y / Z_safe) + cy

    h_full, w_full = rx.shape

    valid = (
        valid_z
        & (map_x >= 0.0)
        & (map_x <= w_full - 1.0)
        & (map_y >= 0.0)
        & (map_y <= h_full - 1.0)
    )

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def camera_maps_full(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    c: float,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = rays[0].shape
    return camera_maps_roi(K, R, t, c, rays, 0, 0, w, h)


def remap_roi(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# Depth C candidate logic
# ============================================================

def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def make_initial_c_center_grid(ny: int, nx: int, c_min: float, c_max: float) -> np.ndarray:
    return np.full((ny, nx), 0.5 * (c_min + c_max), dtype=np.float64)


def c_candidates_for_block(
    center: float,
    half_range: float,
    c_min: float,
    c_max: float,
    samples: int,
) -> np.ndarray:
    if samples <= 1 or half_range <= 1e-15:
        return np.array([np.clip(center, c_min, c_max)], dtype=np.float64)

    lo = max(c_min, float(center) - float(half_range))
    hi = min(c_max, float(center) + float(half_range))

    if hi < lo:
        lo, hi = hi, lo

    if abs(hi - lo) < 1e-15:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)

    return np.linspace(lo, hi, samples, dtype=np.float64)


def smoothness_penalty(c_grid: np.ndarray) -> float:
    vals = []

    if c_grid.shape[1] > 1:
        vals.append(float(np.mean(np.abs(np.diff(c_grid, axis=1)))))

    if c_grid.shape[0] > 1:
        vals.append(float(np.mean(np.abs(np.diff(c_grid, axis=0)))))

    if not vals:
        return 0.0

    return float(np.mean(vals))


def resample_c_grid_nearest(src: np.ndarray, dst_ny: int, dst_nx: int) -> np.ndarray:
    src_ny, src_nx = src.shape
    out = np.zeros((dst_ny, dst_nx), dtype=np.float64)

    for y in range(dst_ny):
        sy = min(src_ny - 1, int(round((y + 0.5) / dst_ny * src_ny - 0.5)))

        for x in range(dst_nx):
            sx = min(src_nx - 1, int(round((x + 0.5) / dst_nx * src_nx - 0.5)))
            out[y, x] = float(src[sy, sx])

    return out


# ============================================================
# Evaluation
# ============================================================

def evaluate_camera_with_iterative_depth(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    K: np.ndarray,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    f_full: float,
    block_size: int,
    c_center_grid: np.ndarray,
    c_half_range: float,
    c_min: float,
    c_max: float,
    c_samples: int,
    min_block_valid_ratio: float,
    min_valid_blocks_ratio: float,
    c_smooth_lambda: float,
    return_pred: bool,
) -> EvalResult:
    h, w = target_y.shape
    target_f = target_y.astype(np.float32)

    ny, nx = block_grid_shape(w, h, block_size)

    if c_center_grid.shape != (ny, nx):
        raise ValueError(f"c_center_grid shape mismatch: expected {(ny, nx)}, got {c_center_grid.shape}")

    c_grid = np.zeros((ny, nx), dtype=np.float64)
    cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_grid = np.zeros((ny, nx), dtype=np.float64)

    pred_out = np.zeros((h, w), dtype=np.float32) if return_pred else None
    valid_out = np.zeros((h, w), dtype=bool) if return_pred else None

    valid_blocks = 0
    cost_sum = 0.0

    for by_idx, by in enumerate(range(0, h, block_size)):
        bh = min(block_size, h - by)

        for bx_idx, bx in enumerate(range(0, w, block_size)):
            bw = min(block_size, w - bx)

            center = float(c_center_grid[by_idx, bx_idx])
            cands = c_candidates_for_block(
                center=center,
                half_range=c_half_range,
                c_min=c_min,
                c_max=c_max,
                samples=c_samples,
            )

            target_roi = target_f[by:by + bh, bx:bx + bw]

            best_cost = float("inf")
            best_c = float(center)
            best_valid_ratio = 0.0
            best_pred = None
            best_valid = None

            for c in cands:
                map_x, map_y, valid = camera_maps_roi(
                    K=K,
                    R=R,
                    t=t,
                    c=float(c),
                    rays=rays,
                    bx=bx,
                    by=by,
                    bw=bw,
                    bh=bh,
                )

                valid_ratio = float(np.mean(valid))

                if valid_ratio < min_block_valid_ratio or not np.any(valid):
                    continue

                pred_roi = remap_roi(ref_y, map_x, map_y)
                diff = target_roi[valid] - pred_roi[valid]
                cost = float(np.mean(np.abs(diff)))

                if cost < best_cost:
                    best_cost = cost
                    best_c = float(c)
                    best_valid_ratio = valid_ratio

                    if return_pred:
                        best_pred = pred_roi
                        best_valid = valid

            c_grid[by_idx, bx_idx] = best_c
            cost_grid[by_idx, bx_idx] = best_cost
            valid_grid[by_idx, bx_idx] = best_valid_ratio

            if np.isfinite(best_cost):
                valid_blocks += 1
                cost_sum += best_cost

                if return_pred and best_pred is not None and best_valid is not None:
                    pred_out[by:by + bh, bx:bx + bw] = best_pred
                    valid_out[by:by + bh, bx:bx + bw] = best_valid

    total_blocks = ny * nx
    valid_block_ratio = float(valid_blocks / max(total_blocks, 1))

    if valid_blocks <= 0 or valid_block_ratio < min_valid_blocks_ratio:
        mae = float("inf")
        score = float("inf")
        valid_pixel_ratio_mean = 0.0
    else:
        mae = float(cost_sum / valid_blocks)
        valid_pixel_ratio_mean = float(np.mean(valid_grid[np.isfinite(cost_grid)]))
        score = mae

        if c_smooth_lambda > 0.0:
            score += float(c_smooth_lambda) * smoothness_penalty(c_grid)

    return EvalResult(
        score=float(score),
        mae=float(mae),
        valid_block_ratio=valid_block_ratio,
        valid_pixel_ratio_mean=float(valid_pixel_ratio_mean),
        R=R.astype(np.float64),
        t=normalize_t(t),
        f_full=float(f_full),
        c_grid=c_grid,
        cost_grid=cost_grid,
        valid_grid=valid_grid,
        pred=pred_out,
        valid=valid_out,
    )


def calc_full_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray) -> dict:
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


# ============================================================
# Candidate generation
# ============================================================

def make_rotation_offsets(step_deg: float, grid: int) -> np.ndarray:
    if grid <= 1 or step_deg <= 0.0:
        return np.array([0.0], dtype=np.float64)

    return np.linspace(-step_deg, step_deg, grid, dtype=np.float64)


def initial_f_values(width: int, f_rel_list: List[float]) -> List[float]:
    return [float(fr * width) for fr in f_rel_list]


def refined_f_values(f_center: float, step_rel: float, grid: int) -> List[float]:
    if grid <= 1 or step_rel <= 0.0:
        return [float(f_center)]

    offs = np.linspace(-step_rel, step_rel, grid, dtype=np.float64)
    return [max(1.0, float(f_center * (1.0 + o))) for o in offs]


def build_initial_seeds_for_f(
    H_refined: np.ndarray,
    K_full: np.ndarray,
    R_E: Optional[np.ndarray],
    t_E: Optional[np.ndarray],
    include_axis_t: bool,
    include_zero_t: bool,
) -> List[SeedPose]:
    seeds = []

    # 1. Homography decomposition: strong translation-scene initial candidates.
    seeds.extend(decompose_homography_candidates(H_refined, K_full))

    # 2. Rotation-only interpretation of H.
    R_H = rotation_from_homography(H_refined, K_full)

    base_ts = []

    if t_E is not None:
        base_ts.append(t_E)
        base_ts.append(-t_E)

    if include_axis_t:
        base_ts.extend(build_axis_t_candidates(include_zero=include_zero_t))
    elif include_zero_t:
        base_ts.append(np.zeros(3, dtype=np.float64))

    base_ts = unique_t_vectors(base_ts)

    for i, t in enumerate(base_ts):
        seeds.append(SeedPose(R=R_H, t=t, name=f"R_from_H_t_{i}"))

    # 3. Essential R/t candidates.
    if R_E is not None and t_E is not None:
        seeds.append(SeedPose(R=R_E, t=t_E, name="Essential_plus"))
        seeds.append(SeedPose(R=R_E, t=-t_E, name="Essential_minus"))

    # 4. Identity fallback.
    if include_zero_t:
        seeds.append(SeedPose(R=np.eye(3, dtype=np.float64), t=np.zeros(3, dtype=np.float64), name="Identity_zero"))

    return unique_seed_poses(seeds)


# ============================================================
# Iterative camera + depth search
# ============================================================

def run_iterative_search(
    target_s: np.ndarray,
    ref_s: np.ndarray,
    full_width: int,
    full_height: int,
    scale: float,
    H_refined: np.ndarray,
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    args,
) -> Tuple[EvalResult, List[dict]]:
    search_h, search_w = target_s.shape
    search_block_size = max(4, int(round(args.search_block_size * scale)))

    ny, nx = block_grid_shape(search_w, search_h, search_block_size)

    c_center_grid = make_initial_c_center_grid(
        ny=ny,
        nx=nx,
        c_min=args.c_min,
        c_max=args.c_max,
    )

    c_half_range = 0.5 * (args.c_max - args.c_min)

    f_rel_list = parse_float_list(args.f_rel_list)

    K_E = build_K(full_width, full_height, args.essential_f_rel * full_width)
    R_E, t_E, essential_meta = estimate_essential_pose(
        pts_target=pts_target,
        pts_ref=pts_ref,
        K=K_E,
        thresh=args.essential_thresh,
    )

    print("[INFO] Essential meta:")
    print(json.dumps(essential_meta, indent=2))

    best: Optional[EvalResult] = None
    history = []

    r_step = float(args.r_step_deg)
    f_step_rel = float(args.f_step_rel)
    t_step = float(args.t_step_deg)

    for it in range(args.search_iters):
        print(f"[SEARCH] iteration {it}")

        if it == 0 or best is None:
            f_values = initial_f_values(full_width, f_rel_list)
        else:
            f_values = refined_f_values(best.f_full, f_step_rel, args.f_grid)

        r_offsets = make_rotation_offsets(r_step, args.r_grid)

        iter_best: Optional[EvalResult] = None
        eval_count = 0
        seed_count_total = 0

        for f_full in f_values:
            K_full = build_K(full_width, full_height, f_full)
            f_search = f_full * scale
            K_search = build_K(search_w, search_h, f_search)
            rays_search = make_rays(search_w, search_h, K_search)

            if it == 0 or best is None:
                seeds = build_initial_seeds_for_f(
                    H_refined=H_refined,
                    K_full=K_full,
                    R_E=R_E,
                    t_E=t_E,
                    include_axis_t=args.include_axis_t,
                    include_zero_t=not args.no_zero_t,
                )
            else:
                if args.refine_t:
                    t_list = perturb_t_candidates(
                        t_center=best.t,
                        step_deg=t_step,
                        grid=args.t_grid,
                        include_zero=not args.no_zero_t,
                    )
                else:
                    t_list = [best.t]

                seeds = [
                    SeedPose(R=best.R, t=t, name=f"refine_t_{i}")
                    for i, t in enumerate(t_list)
                ]

            seed_count_total += len(seeds)

            for seed in seeds:
                for yaw in r_offsets:
                    for pitch in r_offsets:
                        for roll in r_offsets:
                            R = make_delta_R(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll) @ seed.R
                            R = project_to_so3(R)

                            res = evaluate_camera_with_iterative_depth(
                                target_y=target_s,
                                ref_y=ref_s,
                                K=K_search,
                                rays=rays_search,
                                R=R,
                                t=seed.t,
                                f_full=f_full,
                                block_size=search_block_size,
                                c_center_grid=c_center_grid,
                                c_half_range=c_half_range,
                                c_min=args.c_min,
                                c_max=args.c_max,
                                c_samples=args.c_samples,
                                min_block_valid_ratio=args.min_block_valid_ratio,
                                min_valid_blocks_ratio=args.min_valid_blocks_ratio,
                                c_smooth_lambda=args.c_smooth_lambda,
                                return_pred=False,
                            )

                            eval_count += 1

                            if iter_best is None or res.score < iter_best.score:
                                iter_best = res

        if iter_best is None or not np.isfinite(iter_best.score):
            raise RuntimeError(
                "No valid camera candidate found. "
                "Try smaller c range, larger search range, lower min-valid thresholds, or include more t candidates."
            )

        best = iter_best

        # Depth-map iteration: update c center map from the best candidate.
        c_center_grid = best.c_grid.copy()
        c_half_range *= float(args.c_shrink)

        record = {
            "iter": int(it),
            "eval_count": int(eval_count),
            "seed_count_total_over_f": int(seed_count_total),
            "best_score": float(best.score),
            "best_mae": float(best.mae),
            "best_valid_block_ratio": float(best.valid_block_ratio),
            "best_valid_pixel_ratio_mean": float(best.valid_pixel_ratio_mean),
            "best_f_full": float(best.f_full),
            "best_f_rel_to_width": float(best.f_full / full_width),
            "best_t": best.t.tolist(),
            "best_R": best.R.tolist(),
            "c_half_range_after_update": float(c_half_range),
            "c_grid_mean": float(np.mean(best.c_grid)),
            "c_grid_min": float(np.min(best.c_grid)),
            "c_grid_max": float(np.max(best.c_grid)),
            "r_step_deg": float(r_step),
            "f_step_rel": float(f_step_rel),
            "t_step_deg": float(t_step),
        }

        history.append(record)
        print(json.dumps(record, indent=2))

        # Camera-param range shrink.
        r_step *= float(args.r_shrink)
        f_step_rel *= float(args.f_shrink)
        t_step *= float(args.t_shrink)

    return best, history


# ============================================================
# Final C refinement with fixed camera
# ============================================================

def refine_final_c_map(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    K: np.ndarray,
    rays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    f_full: float,
    block_size: int,
    initial_c_grid: np.ndarray,
    initial_half_range: float,
    args,
) -> EvalResult:
    h, w = target_y.shape
    ny, nx = block_grid_shape(w, h, block_size)

    c_center = resample_c_grid_nearest(initial_c_grid, ny, nx)
    c_half = float(initial_half_range)

    best = None

    for it in range(args.final_c_iters):
        res = evaluate_camera_with_iterative_depth(
            target_y=target_y,
            ref_y=ref_y,
            K=K,
            rays=rays,
            R=R,
            t=t,
            f_full=f_full,
            block_size=block_size,
            c_center_grid=c_center,
            c_half_range=c_half,
            c_min=args.c_min,
            c_max=args.c_max,
            c_samples=args.c_samples,
            min_block_valid_ratio=args.min_block_valid_ratio,
            min_valid_blocks_ratio=args.min_valid_blocks_ratio,
            c_smooth_lambda=args.c_smooth_lambda,
            return_pred=(it == args.final_c_iters - 1),
        )

        best = res
        c_center = res.c_grid.copy()
        c_half *= float(args.c_shrink)

        print(
            "[FINAL-C]",
            json.dumps(
                {
                    "iter": it,
                    "score": res.score,
                    "mae": res.mae,
                    "valid_block_ratio": res.valid_block_ratio,
                    "c_half_after_update": c_half,
                    "c_mean": float(np.mean(res.c_grid)),
                    "c_min": float(np.min(res.c_grid)),
                    "c_max": float(np.max(res.c_grid)),
                },
                indent=2,
            ),
        )

    if best is None:
        raise RuntimeError("Final C refinement failed.")

    return best


# ============================================================
# Visualization
# ============================================================

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


def expand_c_grid_to_image(c_grid: np.ndarray, width: int, height: int, block_size: int) -> np.ndarray:
    ny, nx = c_grid.shape
    out = np.zeros((height, width), dtype=np.float32)

    for by_idx in range(ny):
        by = by_idx * block_size
        y1 = min(by + block_size, height)

        for bx_idx in range(nx):
            bx = bx_idx * block_size
            x1 = min(bx + block_size, width)
            out[by:y1, bx:x1] = float(c_grid[by_idx, bx_idx])

    return out


def save_c_map_png(path: str, c_grid: np.ndarray, width: int, height: int, block_size: int):
    c_img = expand_c_grid_to_image(c_grid, width, height, block_size)

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

    # Matching
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--no-clahe", action="store_true")

    # Homography ECC
    parser.add_argument("--h-refine-iters", type=int, default=100)
    parser.add_argument("--h-refine-eps", type=float, default=1e-7)
    parser.add_argument("--h-refine-scale", type=float, default=0.5)
    parser.add_argument("--h-refine-blur", type=int, default=5)

    # Iterative search
    parser.add_argument("--search-scale", type=float, default=0.25)
    parser.add_argument("--search-iters", type=int, default=5)
    parser.add_argument("--search-block-size", type=int, default=128)
    parser.add_argument("--final-block-size", type=int, default=64)

    # Focal search
    parser.add_argument("--f-rel-list", default="0.7,0.9,1.1,1.4")
    parser.add_argument("--f-grid", type=int, default=3)
    parser.add_argument("--f-step-rel", type=float, default=0.12)
    parser.add_argument("--f-shrink", type=float, default=0.5)

    # Rotation search
    parser.add_argument("--r-step-deg", type=float, default=1.5)
    parser.add_argument("--r-grid", type=int, default=3)
    parser.add_argument("--r-shrink", type=float, default=0.5)

    # Translation candidates
    parser.add_argument("--essential-f-rel", type=float, default=1.0)
    parser.add_argument("--essential-thresh", type=float, default=1.0)
    parser.add_argument("--include-axis-t", action="store_true")
    parser.add_argument("--no-zero-t", action="store_true")
    parser.add_argument("--refine-t", action="store_true")
    parser.add_argument("--t-step-deg", type=float, default=2.0)
    parser.add_argument("--t-grid", type=int, default=3)
    parser.add_argument("--t-shrink", type=float, default=0.5)

    # Depth C search
    parser.add_argument("--c-min", type=float, default=0.0)
    parser.add_argument("--c-max", type=float, default=0.02)
    parser.add_argument("--c-samples", type=int, default=3)
    parser.add_argument("--c-shrink", type=float, default=0.5)
    parser.add_argument("--final-c-iters", type=int, default=3)

    # Validity / regularization
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.30)
    parser.add_argument("--min-valid-blocks-ratio", type=float, default=0.30)
    parser.add_argument("--c-smooth-lambda", type=float, default=0.0)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.search_scale <= 0.0 or args.search_scale > 1.0:
        raise ValueError("--search-scale must be in (0, 1].")

    if args.c_max < args.c_min:
        raise ValueError("--c-max must be >= --c-min.")

    print(f"[INFO] target_idx={args.target_idx}, ref_idx={args.ref_idx}")
    print(f"[INFO] resolution={args.width}x{args.height}, bitdepth={args.bitdepth}")
    print(f"[INFO] search_scale={args.search_scale}, search_iters={args.search_iters}")
    print(f"[INFO] c range=[{args.c_min}, {args.c_max}], c_samples={args.c_samples}, c_shrink={args.c_shrink}")

    target = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.ref_idx)

    # ------------------------------------------------------------
    # Feature matching + H + ECC.
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
        match_result.pts_target,
        match_result.pts_ref,
        args.ransac_thresh,
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
    # Search image.
    # ------------------------------------------------------------

    target_s = resize_frame(target.y, args.search_scale)
    ref_s = resize_frame(ref.y, args.search_scale)

    # ------------------------------------------------------------
    # Iterative camera + depth search.
    # ------------------------------------------------------------

    best_search, search_history = run_iterative_search(
        target_s=target_s,
        ref_s=ref_s,
        full_width=args.width,
        full_height=args.height,
        scale=args.search_scale,
        H_refined=H_refined,
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
                "valid_block_ratio": best_search.valid_block_ratio,
                "valid_pixel_ratio_mean": best_search.valid_pixel_ratio_mean,
                "f_full": best_search.f_full,
                "f_rel_to_width": best_search.f_full / args.width,
                "t": best_search.t.tolist(),
                "R": best_search.R.tolist(),
                "c_grid_mean": float(np.mean(best_search.c_grid)),
                "c_grid_min": float(np.min(best_search.c_grid)),
                "c_grid_max": float(np.max(best_search.c_grid)),
            },
            indent=2,
        )
    )

    # ------------------------------------------------------------
    # Final full-res prediction.
    # ------------------------------------------------------------

    K_full = build_K(args.width, args.height, best_search.f_full)
    rays_full = make_rays(args.width, args.height, K_full)

    # c=0 baseline.
    map_x0, map_y0, valid0 = camera_maps_full(
        K=K_full,
        R=best_search.R,
        t=best_search.t,
        c=0.0,
        rays=rays_full,
    )

    pred_c0 = remap_roi(ref.y, map_x0, map_y0)
    cost_c0 = calc_full_cost(target.y, pred_c0, valid0)

    # Final c half range starts from the current remaining search range.
    final_initial_half = 0.5 * (args.c_max - args.c_min) * (args.c_shrink ** max(args.search_iters, 1))

    final_res = refine_final_c_map(
        target_y=target.y,
        ref_y=ref.y,
        K=K_full,
        rays=rays_full,
        R=best_search.R,
        t=best_search.t,
        f_full=best_search.f_full,
        block_size=args.final_block_size,
        initial_c_grid=best_search.c_grid,
        initial_half_range=final_initial_half,
        args=args,
    )

    cost_block = calc_full_cost(target.y, final_res.pred, final_res.valid)

    print("[INFO] Final c=0 cost:")
    print(json.dumps(cost_c0, indent=2))
    print("[INFO] Final block-C cost:")
    print(json.dumps(cost_block, indent=2))

    # ------------------------------------------------------------
    # Save output.
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
        final_res.c_grid,
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
        "iteration": {
            "camera": "R/f/t search ranges shrink every iteration",
            "depth": "block c_center_grid is updated from the best candidate each iteration, and c_half_range shrinks",
        },

        "matching": {
            "num_good_matches": int(len(match_result.good_matches)),
            "homography_inliers": int(H_inliers),
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "ransac_thresh": float(args.ransac_thresh),
            "clahe": bool(not args.no_clahe),
        },

        "homography": {
            "H_initial_target_to_ref": H_init.tolist(),
            "H_refined_target_to_ref": H_refined.tolist(),
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
            "search_valid_block_ratio": float(best_search.valid_block_ratio),
            "search_valid_pixel_ratio_mean": float(best_search.valid_pixel_ratio_mean),
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
            "c_min": float(args.c_min),
            "c_max": float(args.c_max),
            "c_samples": int(args.c_samples),
            "c_shrink": float(args.c_shrink),
            "final_c_iters": int(args.final_c_iters),
            "c_smooth_lambda": float(args.c_smooth_lambda),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
            "min_valid_blocks_ratio": float(args.min_valid_blocks_ratio),
        },

        "search_history": search_history,

        "final_costs": {
            "camera_c0": cost_c0,
            "camera_block_c": cost_block,
        },

        "final_c_block_grid": final_res.c_grid.tolist(),
        "final_c_block_cost_grid": final_res.cost_grid.tolist(),
        "final_c_valid_grid": final_res.valid_grid.tolist(),

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
            "Homography is used only to generate initial R/t candidates.",
            "Homography decomposition candidates are included for translation-dominant scenes.",
            "C is applied only to the translation/parallax term, not to rotation.",
            "Depth C map is intentionally coarse and iteratively refined.",
            "If output is mostly black, reduce c_max, include zero t, loosen valid thresholds, or widen focal/R search.",
            "pred_camera_c0.yuv is rotation-only selected camera warp.",
            "pred_camera_block*C.yuv is selected camera warp with block-wise coarse C.",
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
