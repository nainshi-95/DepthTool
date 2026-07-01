#!/usr/bin/env python3
# local_block_feature_fit.py
#
# Local block feature fitting experiment.
#
# Core idea:
#   1. Extract global feature matches between target and ref.
#   2. Assign matches to target-side blocks.
#   3. Fit local motion model per block:
#        - translation
#        - partial affine
#        - affine
#        - optional homography
#   4. Use local block models as motion observations.
#   5. Aggregate observations into:
#        - global affine model
#        - radial FOE + block scalar c model
#
# Coordinate convention:
#   target pixel x_t -> ref pixel x_r
#
# Local predictor:
#   x_r = M_b(x_t)
#
# Radial compressed model:
#   x_r = x_t + c_b * [x_t - foe_x, y_t - foe_y]
#
# Outputs:
#   - pred_local_block_model.yuv
#   - pred_global_affine_from_blocks.yuv
#   - pred_radial_scalar_from_blocks.yuv
#   - block MV / reliability / model type maps
#   - result.json
#
# Example:
#   python local_block_feature_fit.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --block-size 128 \
#     --block-margin 32 \
#     --fit-model affine \
#     --max-features 50000 \
#     --match-ratio 0.70 \
#     --ransac-thresh 2.0 \
#     --min-matches-affine 8 \
#     --output-dir local_fit_t1_r0

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
class LocalModel:
    ok: bool
    model_type: str
    param: Optional[np.ndarray]
    inlier_count: int
    match_count: int
    reproj_mae: float
    center_mv: Tuple[float, float]
    valid_ratio: float
    fallback: bool
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

    raise ValueError("Only 8-bit and 10-bit YUV420 are supported.")


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
# Model fitting
# ============================================================

def apply_model_points(model_type: str, param: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    if model_type == "translation":
        dx, dy = float(param[0]), float(param[1])
        return pts + np.array([dx, dy], dtype=np.float32)

    if model_type in ("partial_affine", "affine"):
        M = param.reshape(2, 3).astype(np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        ph = np.concatenate([pts.astype(np.float64), ones], axis=1)
        out = ph @ M.T
        return out.astype(np.float32)

    if model_type == "homography":
        H = param.reshape(3, 3).astype(np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        ph = np.concatenate([pts.astype(np.float64), ones], axis=1)
        q = ph @ H.T
        z = q[:, 2:3] + 1e-12
        out = q[:, :2] / z
        return out.astype(np.float32)

    raise ValueError(f"Unknown model_type: {model_type}")


def model_reproj_stats(
    model_type: str,
    param: np.ndarray,
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    thresh: float,
) -> Tuple[int, float, np.ndarray]:
    if len(pts_t) == 0:
        return 0, float("inf"), np.zeros((0,), dtype=bool)

    pred = apply_model_points(model_type, param, pts_t)
    err = np.sqrt(np.sum((pred - pts_r) ** 2, axis=1))
    inliers = err <= thresh

    if np.any(inliers):
        mae = float(np.mean(err[inliers]))
    else:
        mae = float(np.mean(err))

    return int(np.count_nonzero(inliers)), mae, inliers


def fit_translation(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    thresh: float,
    min_matches: int,
) -> Optional[LocalModel]:
    n = len(pts_t)
    if n < min_matches:
        return None

    d = pts_r - pts_t
    dx, dy = np.median(d, axis=0)

    param = np.array([dx, dy], dtype=np.float64)

    inlier_count, mae, _ = model_reproj_stats("translation", param, pts_t, pts_r, thresh)

    return LocalModel(
        ok=inlier_count >= min_matches,
        model_type="translation",
        param=param,
        inlier_count=inlier_count,
        match_count=n,
        reproj_mae=mae,
        center_mv=(0.0, 0.0),
        valid_ratio=0.0,
        fallback=False,
        reason="ok" if inlier_count >= min_matches else "not_enough_translation_inliers",
    )


def fit_partial_affine(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    thresh: float,
    min_matches: int,
) -> Optional[LocalModel]:
    n = len(pts_t)
    if n < min_matches:
        return None

    try:
        M, inlier_mask = cv2.estimateAffinePartial2D(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    except cv2.error:
        return None

    if M is None:
        return None

    param = M.astype(np.float64).reshape(-1)
    inlier_count, mae, _ = model_reproj_stats("partial_affine", param, pts_t, pts_r, thresh)

    return LocalModel(
        ok=inlier_count >= min_matches,
        model_type="partial_affine",
        param=param,
        inlier_count=inlier_count,
        match_count=n,
        reproj_mae=mae,
        center_mv=(0.0, 0.0),
        valid_ratio=0.0,
        fallback=False,
        reason="ok" if inlier_count >= min_matches else "not_enough_partial_affine_inliers",
    )


def fit_affine(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    thresh: float,
    min_matches: int,
) -> Optional[LocalModel]:
    n = len(pts_t)
    if n < min_matches:
        return None

    try:
        M, inlier_mask = cv2.estimateAffine2D(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    except cv2.error:
        return None

    if M is None:
        return None

    param = M.astype(np.float64).reshape(-1)
    inlier_count, mae, _ = model_reproj_stats("affine", param, pts_t, pts_r, thresh)

    return LocalModel(
        ok=inlier_count >= min_matches,
        model_type="affine",
        param=param,
        inlier_count=inlier_count,
        match_count=n,
        reproj_mae=mae,
        center_mv=(0.0, 0.0),
        valid_ratio=0.0,
        fallback=False,
        reason="ok" if inlier_count >= min_matches else "not_enough_affine_inliers",
    )


def fit_homography(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    thresh: float,
    min_matches: int,
) -> Optional[LocalModel]:
    n = len(pts_t)
    if n < min_matches:
        return None

    try:
        H, mask = cv2.findHomography(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=thresh,
            maxIters=2000,
            confidence=0.99,
        )
    except cv2.error:
        return None

    if H is None:
        return None

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    param = H.astype(np.float64).reshape(-1)
    inlier_count, mae, _ = model_reproj_stats("homography", param, pts_t, pts_r, thresh)

    return LocalModel(
        ok=inlier_count >= min_matches,
        model_type="homography",
        param=param,
        inlier_count=inlier_count,
        match_count=n,
        reproj_mae=mae,
        center_mv=(0.0, 0.0),
        valid_ratio=0.0,
        fallback=False,
        reason="ok" if inlier_count >= min_matches else "not_enough_homography_inliers",
    )


def choose_local_model(
    pts_t: np.ndarray,
    pts_r: np.ndarray,
    fit_model: str,
    allow_homography_in_auto: bool,
    ransac_thresh: float,
    min_matches_translation: int,
    min_matches_affine: int,
    min_matches_homography: int,
) -> Optional[LocalModel]:
    candidates: List[LocalModel] = []

    tr = fit_translation(pts_t, pts_r, ransac_thresh, min_matches_translation)
    if tr is not None and tr.ok:
        candidates.append(tr)

    if fit_model in ("partial_affine", "auto"):
        pa = fit_partial_affine(pts_t, pts_r, ransac_thresh, min_matches_affine)
        if pa is not None and pa.ok:
            candidates.append(pa)

    if fit_model in ("affine", "auto"):
        af = fit_affine(pts_t, pts_r, ransac_thresh, min_matches_affine)
        if af is not None and af.ok:
            candidates.append(af)

    if fit_model == "homography" or (fit_model == "auto" and allow_homography_in_auto):
        hg = fit_homography(pts_t, pts_r, ransac_thresh, min_matches_homography)
        if hg is not None and hg.ok:
            candidates.append(hg)

    if fit_model == "translation":
        return tr if tr is not None and tr.ok else None

    if fit_model == "partial_affine":
        pa = fit_partial_affine(pts_t, pts_r, ransac_thresh, min_matches_affine)
        return pa if pa is not None and pa.ok else tr

    if fit_model == "affine":
        af = fit_affine(pts_t, pts_r, ransac_thresh, min_matches_affine)
        return af if af is not None and af.ok else tr

    if fit_model == "homography":
        hg = fit_homography(pts_t, pts_r, ransac_thresh, min_matches_homography)
        if hg is not None and hg.ok:
            return hg
        af = fit_affine(pts_t, pts_r, ransac_thresh, min_matches_affine)
        if af is not None and af.ok:
            return af
        return tr

    if not candidates:
        return None

    # Conservative model selection.
    # Penalize high-DOF models so they are selected only when they really improve reprojection error.
    penalty = {
        "translation": 0.75,
        "partial_affine": 1.00,
        "affine": 1.25,
        "homography": 2.50,
    }

    def score(m: LocalModel):
        inlier_ratio = m.inlier_count / max(m.match_count, 1)
        return m.reproj_mae + penalty.get(m.model_type, 1.0) - 0.5 * inlier_ratio

    candidates.sort(key=score)
    return candidates[0]


# ============================================================
# Map generation and prediction
# ============================================================

def model_maps_for_roi(
    model_type: str,
    param: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(bx, bx + bw, dtype=np.float32),
        np.arange(by, by + bh, dtype=np.float32),
    )

    pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    out = apply_model_points(model_type, param, pts)

    mx = out[:, 0].reshape(bh, bw).astype(np.float32)
    my = out[:, 1].reshape(bh, bw).astype(np.float32)

    valid = (
        (mx >= 0.0)
        & (mx <= width - 1.0)
        & (my >= 0.0)
        & (my <= height - 1.0)
    )

    return mx, my, valid


def remap_ref(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


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


# ============================================================
# Aggregation models
# ============================================================

def fit_global_translation_from_matches(pts_t: np.ndarray, pts_r: np.ndarray) -> np.ndarray:
    d = pts_r - pts_t
    dx, dy = np.median(d, axis=0)
    return np.array([dx, dy], dtype=np.float64)


def fit_global_affine_from_block_mvs(
    centers: np.ndarray,
    mvs: np.ndarray,
    weights: np.ndarray,
    ransac_thresh: float,
) -> Optional[np.ndarray]:
    if len(centers) < 6:
        return None

    pts_t = centers.astype(np.float32)
    pts_r = (centers + mvs).astype(np.float32)

    try:
        M, mask = cv2.estimateAffine2D(
            pts_t,
            pts_r,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=3000,
            confidence=0.99,
            refineIters=10,
        )
    except cv2.error:
        return None

    if M is None:
        return None

    return M.astype(np.float64).reshape(-1)


def fit_foe_from_block_mvs(
    centers: np.ndarray,
    mvs: np.ndarray,
    weights: np.ndarray,
    width: int,
    height: int,
    min_mv_mag: float,
) -> Tuple[np.ndarray, dict]:
    rows = []
    rhs = []
    ws = []

    for p, v, w in zip(centers, mvs, weights):
        dx, dy = float(v[0]), float(v[1])
        mag = math.sqrt(dx * dx + dy * dy)

        if mag < min_mv_mag:
            continue

        # FOE lies on the line passing through p with direction v.
        # normal n = [-dy, dx]
        # n dot foe = n dot p
        n = np.array([-dy, dx], dtype=np.float64)
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-9:
            continue

        n = n / n_norm

        rows.append(n)
        rhs.append(float(n @ p))
        ws.append(max(float(w), 1e-6))

    meta = {
        "num_equations": len(rows),
        "success": False,
        "fallback": False,
    }

    if len(rows) < 4:
        meta["fallback"] = True
        foe = np.array([(width - 1.0) * 0.5, (height - 1.0) * 0.5], dtype=np.float64)
        return foe, meta

    A = np.stack(rows, axis=0)
    b = np.array(rhs, dtype=np.float64)
    w = np.array(ws, dtype=np.float64)
    sw = np.sqrt(w)

    Aw = A * sw[:, None]
    bw = b * sw

    try:
        foe, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    except np.linalg.LinAlgError:
        meta["fallback"] = True
        foe = np.array([(width - 1.0) * 0.5, (height - 1.0) * 0.5], dtype=np.float64)
        return foe, meta

    meta["success"] = True
    meta["foe_x"] = float(foe[0])
    meta["foe_y"] = float(foe[1])

    return foe.astype(np.float64), meta


def make_full_affine_prediction(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    width: int,
    height: int,
    affine_param: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    out = apply_model_points("affine", affine_param, pts)

    mx = out[:, 0].reshape(height, width).astype(np.float32)
    my = out[:, 1].reshape(height, width).astype(np.float32)

    valid = (
        (mx >= 0.0)
        & (mx <= width - 1.0)
        & (my >= 0.0)
        & (my <= height - 1.0)
    )

    pred = remap_ref(ref_y, mx, my)
    return pred, valid


def make_radial_scalar_prediction(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    foe: np.ndarray,
    center_mv_grid_x: np.ndarray,
    center_mv_grid_y: np.ndarray,
    reliable_grid: np.ndarray,
    c_clip: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = block_grid_shape(width, height, block_size)

    c_grid = np.zeros((ny, nx), dtype=np.float64)

    # Estimate c_b from observed block center MV:
    #   mv ≈ c * (center - foe)
    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        cy = 0.5 * (by + y1 - 1.0)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            cx = 0.5 * (bx + x1 - 1.0)

            r = np.array([cx - foe[0], cy - foe[1]], dtype=np.float64)
            mv = np.array([center_mv_grid_x[by_idx, bx_idx], center_mv_grid_y[by_idx, bx_idx]], dtype=np.float64)

            denom = float(r @ r) + 1e-12
            c = float((mv @ r) / denom)

            c_grid[by_idx, bx_idx] = np.clip(c, -c_clip, c_clip)

    # Fill unreliable blocks with median reliable c.
    if np.any(reliable_grid):
        med_c = float(np.median(c_grid[reliable_grid]))
    else:
        med_c = 0.0

    c_grid[~reliable_grid] = med_c

    pred = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)

            c = float(c_grid[by_idx, bx_idx])

            xs, ys = np.meshgrid(
                np.arange(bx, x1, dtype=np.float32),
                np.arange(by, y1, dtype=np.float32),
            )

            mx = xs + c * (xs - float(foe[0]))
            my = ys + c * (ys - float(foe[1]))

            v = (
                (mx >= 0.0)
                & (mx <= width - 1.0)
                & (my >= 0.0)
                & (my <= height - 1.0)
            )

            pred_roi = remap_ref(ref_y, mx.astype(np.float32), my.astype(np.float32))

            pred[by:y1, bx:x1] = pred_roi
            valid[by:y1, bx:x1] = v

    return pred, valid, c_grid


# ============================================================
# Visualization
# ============================================================

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


def save_mv_field_png(
    path: str,
    mvx: np.ndarray,
    mvy: np.ndarray,
    reliable: np.ndarray,
    width: int,
    height: int,
    block_size: int,
    scale: float = 4.0,
):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    mag = np.sqrt(mvx * mvx + mvy * mvy)
    save_scalar_map_png(path + ".mag.png", mag, width, height, block_size)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        cy = int(round(0.5 * (by + y1 - 1)))

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            cx = int(round(0.5 * (bx + x1 - 1)))

            dx = float(mvx[by_idx, bx_idx])
            dy = float(mvy[by_idx, bx_idx])

            color = (0, 255, 0) if reliable[by_idx, bx_idx] else (0, 0, 255)

            p0 = (cx, cy)
            p1 = (int(round(cx + scale * dx)), int(round(cy + scale * dy)))

            cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.3)

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

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-margin", type=int, default=32)

    parser.add_argument(
        "--fit-model",
        choices=["translation", "partial_affine", "affine", "homography", "auto"],
        default="affine",
    )
    parser.add_argument("--auto-allow-homography", action="store_true")

    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--no-clahe", action="store_true")

    parser.add_argument("--ransac-thresh", type=float, default=2.0)
    parser.add_argument("--min-matches-translation", type=int, default=3)
    parser.add_argument("--min-matches-affine", type=int, default=8)
    parser.add_argument("--min-matches-homography", type=int, default=16)

    parser.add_argument(
        "--fallback",
        choices=["none", "global_translation"],
        default="global_translation",
    )

    parser.add_argument("--radial-c-clip", type=float, default=0.20)
    parser.add_argument("--foe-min-mv-mag", type=float, default=0.25)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    width = args.width
    height = args.height
    block_size = args.block_size

    print(f"[INFO] target={args.target_idx}, ref={args.ref_idx}")
    print(f"[INFO] size={width}x{height}, bitdepth={args.bitdepth}")
    print(f"[INFO] block={block_size}, margin={args.block_margin}, fit_model={args.fit_model}")

    target = read_y_frame(args.input, width, height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, width, height, args.bitdepth, args.ref_idx)

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

    global_tr = fit_global_translation_from_matches(match.pts_target, match.pts_ref)
    print(f"[INFO] global median translation fallback = dx={global_tr[0]:.3f}, dy={global_tr[1]:.3f}")

    ny, nx = block_grid_shape(width, height, block_size)

    pred_local = np.zeros((height, width), dtype=np.float32)
    valid_local = np.zeros((height, width), dtype=bool)

    model_type_grid = np.zeros((ny, nx), dtype=np.float64)
    match_count_grid = np.zeros((ny, nx), dtype=np.float64)
    inlier_count_grid = np.zeros((ny, nx), dtype=np.float64)
    reproj_mae_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)
    mvx_grid = np.zeros((ny, nx), dtype=np.float64)
    mvy_grid = np.zeros((ny, nx), dtype=np.float64)
    reliable_grid = np.zeros((ny, nx), dtype=bool)

    type_to_id = {
        "invalid": 0,
        "fallback_translation": 1,
        "translation": 2,
        "partial_affine": 3,
        "affine": 4,
        "homography": 5,
    }

    block_models: List[List[Dict]] = []

    pts_t_all = match.pts_target
    pts_r_all = match.pts_ref

    for by_idx, by in enumerate(range(0, height, block_size)):
        row_records = []
        y1 = min(by + block_size, height)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)

            mx0 = max(0, bx - args.block_margin)
            mx1 = min(width, x1 + args.block_margin)
            my0 = max(0, by - args.block_margin)
            my1 = min(height, y1 + args.block_margin)

            # Assign matches by target-side position.
            mask = (
                (pts_t_all[:, 0] >= mx0)
                & (pts_t_all[:, 0] < mx1)
                & (pts_t_all[:, 1] >= my0)
                & (pts_t_all[:, 1] < my1)
            )

            pts_t = pts_t_all[mask]
            pts_r = pts_r_all[mask]

            model = choose_local_model(
                pts_t=pts_t,
                pts_r=pts_r,
                fit_model=args.fit_model,
                allow_homography_in_auto=args.auto_allow_homography,
                ransac_thresh=args.ransac_thresh,
                min_matches_translation=args.min_matches_translation,
                min_matches_affine=args.min_matches_affine,
                min_matches_homography=args.min_matches_homography,
            )

            if model is None or not model.ok:
                if args.fallback == "global_translation":
                    model = LocalModel(
                        ok=True,
                        model_type="translation",
                        param=global_tr.copy(),
                        inlier_count=0,
                        match_count=len(pts_t),
                        reproj_mae=float("inf"),
                        center_mv=(float(global_tr[0]), float(global_tr[1])),
                        valid_ratio=0.0,
                        fallback=True,
                        reason="fallback_global_translation",
                    )
                else:
                    model = LocalModel(
                        ok=False,
                        model_type="invalid",
                        param=None,
                        inlier_count=0,
                        match_count=len(pts_t),
                        reproj_mae=float("inf"),
                        center_mv=(0.0, 0.0),
                        valid_ratio=0.0,
                        fallback=False,
                        reason="no_valid_model",
                    )

            if model.ok and model.param is not None and model.model_type != "invalid":
                bw = x1 - bx
                bh = y1 - by

                map_x, map_y, valid = model_maps_for_roi(
                    model.model_type,
                    model.param,
                    bx,
                    by,
                    bw,
                    bh,
                    width,
                    height,
                )

                pred_roi = remap_ref(ref.y, map_x, map_y)

                pred_local[by:y1, bx:x1] = pred_roi
                valid_local[by:y1, bx:x1] = valid

                valid_ratio = float(np.mean(valid))

                cx = 0.5 * (bx + x1 - 1.0)
                cy = 0.5 * (by + y1 - 1.0)
                center_out = apply_model_points(
                    model.model_type,
                    model.param,
                    np.array([[cx, cy]], dtype=np.float32),
                )[0]

                dx = float(center_out[0] - cx)
                dy = float(center_out[1] - cy)

                model.center_mv = (dx, dy)
                model.valid_ratio = valid_ratio

                is_reliable = (
                    (not model.fallback)
                    and model.inlier_count >= max(args.min_matches_translation, 3)
                    and np.isfinite(model.reproj_mae)
                    and valid_ratio > 0.5
                )
            else:
                valid_ratio = 0.0
                dx = 0.0
                dy = 0.0
                is_reliable = False

            model_id = type_to_id["fallback_translation"] if model.fallback else type_to_id.get(model.model_type, 0)

            model_type_grid[by_idx, bx_idx] = model_id
            match_count_grid[by_idx, bx_idx] = model.match_count
            inlier_count_grid[by_idx, bx_idx] = model.inlier_count
            reproj_mae_grid[by_idx, bx_idx] = model.reproj_mae
            valid_ratio_grid[by_idx, bx_idx] = model.valid_ratio
            mvx_grid[by_idx, bx_idx] = model.center_mv[0]
            mvy_grid[by_idx, bx_idx] = model.center_mv[1]
            reliable_grid[by_idx, bx_idx] = is_reliable

            row_records.append(
                {
                    "block_x": int(bx),
                    "block_y": int(by),
                    "model_type": model.model_type,
                    "fallback": bool(model.fallback),
                    "reason": model.reason,
                    "match_count": int(model.match_count),
                    "inlier_count": int(model.inlier_count),
                    "reproj_mae": float(model.reproj_mae) if np.isfinite(model.reproj_mae) else None,
                    "valid_ratio": float(model.valid_ratio),
                    "center_mv": [float(model.center_mv[0]), float(model.center_mv[1])],
                    "param": model.param.tolist() if model.param is not None else None,
                    "reliable": bool(is_reliable),
                }
            )

        block_models.append(row_records)

    cost_local = calc_cost(target.y, pred_local, valid_local, args.bitdepth)

    reliable_count = int(np.count_nonzero(reliable_grid))
    total_blocks = int(reliable_grid.size)

    print(f"[INFO] reliable blocks = {reliable_count}/{total_blocks}")
    print("[INFO] local block model cost:")
    print(json.dumps(cost_local, indent=2))

    # ------------------------------------------------------------
    # Aggregate local observations.
    # ------------------------------------------------------------

    centers = []
    mvs = []
    weights = []

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        cy = 0.5 * (by + y1 - 1.0)

        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            cx = 0.5 * (bx + x1 - 1.0)

            if not reliable_grid[by_idx, bx_idx]:
                continue

            centers.append([cx, cy])
            mvs.append([mvx_grid[by_idx, bx_idx], mvy_grid[by_idx, bx_idx]])

            err = reproj_mae_grid[by_idx, bx_idx]
            inl = inlier_count_grid[by_idx, bx_idx]
            wgt = max(1.0, float(inl)) / max(1.0, float(err))
            weights.append(wgt)

    centers = np.asarray(centers, dtype=np.float64)
    mvs = np.asarray(mvs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Global affine from block observations.
    affine_from_blocks = fit_global_affine_from_block_mvs(
        centers,
        mvs,
        weights,
        ransac_thresh=max(2.0, args.ransac_thresh * 2.0),
    )

    if affine_from_blocks is not None:
        pred_affine, valid_affine = make_full_affine_prediction(
            target.y,
            ref.y,
            width,
            height,
            affine_from_blocks,
        )
        cost_affine = calc_cost(target.y, pred_affine, valid_affine, args.bitdepth)
    else:
        pred_affine = np.zeros_like(target.y, dtype=np.float32)
        valid_affine = np.zeros_like(target.y, dtype=bool)
        cost_affine = {
            "valid_ratio": 0.0,
            "mae": float("inf"),
            "mse": float("inf"),
            "psnr": None,
        }

    print("[INFO] global affine from blocks cost:")
    print(json.dumps(cost_affine, indent=2))

    # FOE/radial scalar from block observations.
    foe, foe_meta = fit_foe_from_block_mvs(
        centers,
        mvs,
        weights,
        width=width,
        height=height,
        min_mv_mag=args.foe_min_mv_mag,
    )

    pred_radial, valid_radial, c_radial_grid = make_radial_scalar_prediction(
        target_y=target.y,
        ref_y=ref.y,
        width=width,
        height=height,
        block_size=block_size,
        foe=foe,
        center_mv_grid_x=mvx_grid,
        center_mv_grid_y=mvy_grid,
        reliable_grid=reliable_grid,
        c_clip=args.radial_c_clip,
    )

    cost_radial = calc_cost(target.y, pred_radial, valid_radial, args.bitdepth)

    print("[INFO] FOE/radial meta:")
    print(json.dumps(foe_meta, indent=2))
    print("[INFO] radial scalar from blocks cost:")
    print(json.dumps(cost_radial, indent=2))

    # ------------------------------------------------------------
    # Save output.
    # ------------------------------------------------------------

    paths = {
        "target_yuv": os.path.join(args.output_dir, "target_pair.yuv"),
        "ref_yuv": os.path.join(args.output_dir, "ref_pair.yuv"),
        "pred_local_block_model_yuv": os.path.join(args.output_dir, "pred_local_block_model.yuv"),
        "pred_global_affine_from_blocks_yuv": os.path.join(args.output_dir, "pred_global_affine_from_blocks.yuv"),
        "pred_radial_scalar_from_blocks_yuv": os.path.join(args.output_dir, "pred_radial_scalar_from_blocks.yuv"),
    }

    write_single_yuv420(paths["target_yuv"], target.y, width, height, args.bitdepth)
    write_single_yuv420(paths["ref_yuv"], ref.y, width, height, args.bitdepth)
    write_single_yuv420(paths["pred_local_block_model_yuv"], pred_local, width, height, args.bitdepth)
    write_single_yuv420(paths["pred_global_affine_from_blocks_yuv"], pred_affine, width, height, args.bitdepth)
    write_single_yuv420(paths["pred_radial_scalar_from_blocks_yuv"], pred_radial, width, height, args.bitdepth)

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_local_block_model.png"), pred_local, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_global_affine_from_blocks.png"), pred_affine, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_radial_scalar_from_blocks.png"), pred_radial, args.bitdepth)

    save_diff_png(os.path.join(args.output_dir, "diff_local_block_model.png"), target.y, pred_local, valid_local)
    save_diff_png(os.path.join(args.output_dir, "diff_global_affine_from_blocks.png"), target.y, pred_affine, valid_affine)
    save_diff_png(os.path.join(args.output_dir, "diff_radial_scalar_from_blocks.png"), target.y, pred_radial, valid_radial)

    save_scalar_map_png(os.path.join(args.output_dir, "model_type_map.png"), model_type_grid, width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "match_count_map.png"), match_count_grid, width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "inlier_count_map.png"), inlier_count_grid, width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "reproj_mae_map.png"), reproj_mae_grid, width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "valid_ratio_map.png"), valid_ratio_grid, width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "reliable_map.png"), reliable_grid.astype(np.float64), width, height, block_size)
    save_scalar_map_png(os.path.join(args.output_dir, "radial_c_map.png"), c_radial_grid, width, height, block_size)

    save_mv_field_png(
        os.path.join(args.output_dir, "local_center_mv_field.png"),
        mvx_grid,
        mvy_grid,
        reliable_grid,
        width,
        height,
        block_size,
        scale=4.0,
    )

    result = {
        "input": args.input,
        "width": int(width),
        "height": int(height),
        "bitdepth": int(args.bitdepth),
        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "method": {
            "description": "local block feature fitting, then aggregation",
            "local_model": args.fit_model,
            "block_size": int(block_size),
            "block_margin": int(args.block_margin),
            "coordinate": "target pixel -> ref pixel",
        },

        "feature_matching": {
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "clahe": bool(not args.no_clahe),
            "num_good_matches": int(len(match.good_matches)),
        },

        "fitting_args": {
            "ransac_thresh": float(args.ransac_thresh),
            "min_matches_translation": int(args.min_matches_translation),
            "min_matches_affine": int(args.min_matches_affine),
            "min_matches_homography": int(args.min_matches_homography),
            "fallback": args.fallback,
            "auto_allow_homography": bool(args.auto_allow_homography),
        },

        "summary": {
            "total_blocks": total_blocks,
            "reliable_blocks": reliable_count,
            "reliable_ratio": float(reliable_count / max(total_blocks, 1)),
            "global_translation_fallback": global_tr.tolist(),
            "foe": foe.tolist(),
            "foe_meta": foe_meta,
            "global_affine_from_blocks": affine_from_blocks.tolist() if affine_from_blocks is not None else None,
        },

        "costs": {
            "local_block_model": cost_local,
            "global_affine_from_blocks": cost_affine,
            "radial_scalar_from_blocks": cost_radial,
        },

        "grids": {
            "model_type_id": model_type_grid.tolist(),
            "model_type_id_legend": {
                "0": "invalid",
                "1": "fallback_translation",
                "2": "translation",
                "3": "partial_affine",
                "4": "affine",
                "5": "homography",
            },
            "match_count": match_count_grid.tolist(),
            "inlier_count": inlier_count_grid.tolist(),
            "reproj_mae": np.where(np.isfinite(reproj_mae_grid), reproj_mae_grid, -1).tolist(),
            "valid_ratio": valid_ratio_grid.tolist(),
            "mvx": mvx_grid.tolist(),
            "mvy": mvy_grid.tolist(),
            "reliable": reliable_grid.astype(int).tolist(),
            "radial_c": c_radial_grid.tolist(),
        },

        "block_models": block_models,

        "outputs": paths,

        "png_outputs": {
            "match_vis": os.path.join(args.output_dir, "match_vis.png"),
            "local_center_mv_field": os.path.join(args.output_dir, "local_center_mv_field.png"),
            "local_center_mv_magnitude": os.path.join(args.output_dir, "local_center_mv_field.png.mag.png"),
            "model_type_map": os.path.join(args.output_dir, "model_type_map.png"),
            "match_count_map": os.path.join(args.output_dir, "match_count_map.png"),
            "inlier_count_map": os.path.join(args.output_dir, "inlier_count_map.png"),
            "reproj_mae_map": os.path.join(args.output_dir, "reproj_mae_map.png"),
            "reliable_map": os.path.join(args.output_dir, "reliable_map.png"),
            "radial_c_map": os.path.join(args.output_dir, "radial_c_map.png"),
        },

        "interpretation": [
            "pred_local_block_model is an upper-bound-like local fitting predictor. It may be blocky.",
            "local_center_mv_field shows observed local motion from block feature fitting.",
            "pred_global_affine_from_blocks tests whether local observations agree with one global affine model.",
            "pred_radial_scalar_from_blocks tests whether local observations can be compressed into FOE/radial global info + block scalar c.",
            "If local block model is good but radial/global aggregation is poor, the scene needs richer basis or multiple modes.",
            "If radial scalar is good for forward/backward camera motion, the global+block-c syntax is promising.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] result JSON: {json_path}")
    print(f"[DONE] local block model YUV: {paths['pred_local_block_model_yuv']}")
    print(f"[DONE] global affine YUV: {paths['pred_global_affine_from_blocks_yuv']}")
    print(f"[DONE] radial scalar YUV: {paths['pred_radial_scalar_from_blocks_yuv']}")


if __name__ == "__main__":
    main()
