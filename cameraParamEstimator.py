#!/usr/bin/env python3
# pair_homography_c_strength_refine.py
#
# Pairwise homography-flow + block-wise C strength matching.
#
# Model:
#   p_hom(x)  = H * x
#   flow(x)   = p_hom(x) - x
#   p_pred(x) = x + c_block * flow(x)
#
# Meaning:
#   c = 0 : identity / no warp
#   c = 1 : homography
#   c > 1 : stronger than homography along the same flow direction
#   c < 1 : weaker or opposite direction
#
# Coarse-to-fine C search:
#   Start from [c_min, c_max].
#   Sample c_samples candidates.
#   Pick best c for each block.
#   Narrow range around the best c.
#   Repeat c_search_iters times.
#
# Example:
#   python pair_homography_c_strength_refine.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --block-size 64 \
#     --c-min 0.0 \
#     --c-max 2.0 \
#     --c-samples 9 \
#     --c-search-iters 4 \
#     --output-dir pair_homo_c_t1_r0

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

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


# ============================================================
# I/O
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


# ============================================================
# Feature matching + homography
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

    if len(kp_t) < 8 or len(kp_r) < 8:
        raise RuntimeError(f"Not enough keypoints: target={len(kp_t)}, ref={len(kp_r)}")

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


def estimate_homography(
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

    H = H.astype(np.float64)

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H, mask


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


# ============================================================
# Homography flow
# ============================================================

def make_identity_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return xs, ys


def homography_map(H: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = make_identity_grid(width, height)

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    denom_safe = denom + 1e-12

    map_x = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom_safe
    map_y = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom_safe

    valid = (
        (np.abs(denom) > 1e-9)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )

    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def build_homography_flow(H: np.ndarray, width: int, height: int):
    xs, ys = make_identity_grid(width, height)
    hom_x, hom_y, hom_valid = homography_map(H, width, height)

    flow_x = hom_x - xs
    flow_y = hom_y - ys

    return xs, ys, flow_x, flow_y, hom_x, hom_y, hom_valid


def remap_full(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def remap_roi(
    ref_y: np.ndarray,
    map_x_roi: np.ndarray,
    map_y_roi: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x_roi.astype(np.float32),
        map_y_roi.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# Cost
# ============================================================

def calc_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, min_valid_ratio: float):
    valid_ratio = float(np.mean(valid))

    if valid_ratio < min_valid_ratio or not np.any(valid):
        return {
            "valid_ratio": valid_ratio,
            "mae": float("inf"),
            "mse": float("inf"),
            "psnr": None,
        }

    diff = target_y.astype(np.float32)[valid] - pred_y.astype(np.float32)[valid]

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    if mse <= 1e-12:
        psnr = 999.0
    else:
        maxv = 255.0 if target_y.dtype == np.uint8 else 1023.0
        psnr = float(10.0 * np.log10((maxv * maxv) / mse))

    return {
        "valid_ratio": valid_ratio,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }


def valid_from_map(map_x: np.ndarray, map_y: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )


# ============================================================
# Coarse-to-fine C search
# ============================================================

def update_range_around_best(
    candidates: np.ndarray,
    best_idx: int,
    c_global_min: float,
    c_global_max: float,
    min_width: float,
) -> Tuple[float, float]:
    n = len(candidates)

    if n <= 1:
        c = float(candidates[best_idx])
        return c, c

    best_c = float(candidates[best_idx])

    if best_idx == 0:
        step = float(candidates[1] - candidates[0])
    elif best_idx == n - 1:
        step = float(candidates[-1] - candidates[-2])
    else:
        step_left = float(candidates[best_idx] - candidates[best_idx - 1])
        step_right = float(candidates[best_idx + 1] - candidates[best_idx])
        step = max(step_left, step_right)

    new_lo = best_c - step
    new_hi = best_c + step

    if new_hi - new_lo < min_width:
        mid = 0.5 * (new_lo + new_hi)
        new_lo = mid - 0.5 * min_width
        new_hi = mid + 0.5 * min_width

    new_lo = max(c_global_min, new_lo)
    new_hi = min(c_global_max, new_hi)

    if new_hi < new_lo:
        new_lo, new_hi = new_hi, new_lo

    return new_lo, new_hi


def search_best_c_for_block(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    c_min: float,
    c_max: float,
    c_samples: int,
    c_search_iters: int,
    min_block_valid_ratio: float,
    min_range_width: float,
) -> dict:
    height, width = target_y.shape

    x1 = bx + bw
    y1 = by + bh

    target_roi = target_y[by:y1, bx:x1].astype(np.float32)
    xs_roi = xs[by:y1, bx:x1]
    ys_roi = ys[by:y1, bx:x1]
    flow_x_roi = flow_x[by:y1, bx:x1]
    flow_y_roi = flow_y[by:y1, bx:x1]

    lo = float(c_min)
    hi = float(c_max)

    history = []

    final_best = {
        "c": 0.0,
        "cost": float("inf"),
        "valid_ratio": 0.0,
        "second_cost": float("inf"),
        "confidence": 0.0,
    }

    for it in range(c_search_iters):
        if c_samples <= 1 or abs(hi - lo) < 1e-12:
            candidates = np.array([0.5 * (lo + hi)], dtype=np.float64)
        else:
            candidates = np.linspace(lo, hi, c_samples, dtype=np.float64)

        costs = []
        valid_ratios = []

        for c in candidates:
            map_x = xs_roi + float(c) * flow_x_roi
            map_y = ys_roi + float(c) * flow_y_roi

            valid = valid_from_map(map_x, map_y, width, height)
            valid_ratio = float(np.mean(valid))
            valid_ratios.append(valid_ratio)

            if valid_ratio < min_block_valid_ratio or not np.any(valid):
                costs.append(float("inf"))
                continue

            pred = remap_roi(ref_y, map_x, map_y)
            diff = target_roi[valid] - pred[valid]
            cost = float(np.mean(np.abs(diff)))
            costs.append(cost)

        costs_np = np.asarray(costs, dtype=np.float64)
        order = np.argsort(costs_np)

        best_idx = int(order[0])
        best_c = float(candidates[best_idx])
        best_cost = float(costs_np[best_idx])
        best_valid_ratio = float(valid_ratios[best_idx])

        if len(order) > 1:
            second_cost = float(costs_np[int(order[1])])
        else:
            second_cost = float("inf")

        confidence = 0.0
        if np.isfinite(best_cost) and np.isfinite(second_cost):
            confidence = max(0.0, second_cost - best_cost)

        history.append(
            {
                "iter": int(it),
                "range": [float(lo), float(hi)],
                "best_c": best_c,
                "best_cost_mae": best_cost,
                "second_best_cost_mae": second_cost,
                "confidence_second_minus_best": float(confidence),
            }
        )

        final_best = {
            "c": best_c,
            "cost": best_cost,
            "valid_ratio": best_valid_ratio,
            "second_cost": second_cost,
            "confidence": confidence,
        }

        lo, hi = update_range_around_best(
            candidates=candidates,
            best_idx=best_idx,
            c_global_min=c_min,
            c_global_max=c_max,
            min_width=min_range_width,
        )

    best_c = final_best["c"]

    map_x = xs_roi + best_c * flow_x_roi
    map_y = ys_roi + best_c * flow_y_roi
    valid = valid_from_map(map_x, map_y, width, height)
    pred = remap_roi(ref_y, map_x, map_y)

    return {
        "best_c": float(best_c),
        "best_cost_mae": float(final_best["cost"]),
        "second_best_cost_mae": float(final_best["second_cost"]),
        "confidence_second_minus_best": float(final_best["confidence"]),
        "valid_ratio": float(np.mean(valid)),
        "pred": pred,
        "valid": valid,
        "history": history,
    }


def blockwise_c_refine(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    block_size: int,
    c_min: float,
    c_max: float,
    c_samples: int,
    c_search_iters: int,
    min_block_valid_ratio: float,
    min_range_width: float,
):
    height, width = target_y.shape

    out_pred = np.zeros((height, width), dtype=np.float32)
    out_valid = np.zeros((height, width), dtype=bool)
    c_map = np.zeros((height, width), dtype=np.float32)
    conf_map = np.zeros((height, width), dtype=np.float32)

    block_records = []
    valid_blocks = 0
    total_blocks = 0

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            bw = min(block_size, width - bx)
            bh = min(block_size, height - by)

            res = search_best_c_for_block(
                target_y=target_y,
                ref_y=ref_y,
                xs=xs,
                ys=ys,
                flow_x=flow_x,
                flow_y=flow_y,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                c_min=c_min,
                c_max=c_max,
                c_samples=c_samples,
                c_search_iters=c_search_iters,
                min_block_valid_ratio=min_block_valid_ratio,
                min_range_width=min_range_width,
            )

            x1 = bx + bw
            y1 = by + bh

            out_pred[by:y1, bx:x1] = res["pred"]
            out_valid[by:y1, bx:x1] = res["valid"]
            c_map[by:y1, bx:x1] = res["best_c"]
            conf_map[by:y1, bx:x1] = res["confidence_second_minus_best"]

            if np.isfinite(res["best_cost_mae"]):
                valid_blocks += 1

            total_blocks += 1

            block_records.append(
                {
                    "bx": int(bx),
                    "by": int(by),
                    "w": int(bw),
                    "h": int(bh),
                    "best_c": float(res["best_c"]),
                    "best_cost_mae": float(res["best_cost_mae"]),
                    "second_best_cost_mae": float(res["second_best_cost_mae"]),
                    "confidence_second_minus_best": float(res["confidence_second_minus_best"]),
                    "valid_ratio": float(res["valid_ratio"]),
                    "history": res["history"],
                }
            )

    cost = calc_cost(target_y, out_pred, out_valid, min_valid_ratio=0.0)

    summary = {
        "block_size": int(block_size),
        "num_blocks": int(total_blocks),
        "valid_blocks": int(valid_blocks),
        "valid_block_ratio": float(valid_blocks / max(total_blocks, 1)),
        "valid_ratio": float(cost["valid_ratio"]),
        "mae": float(cost["mae"]),
        "mse": float(cost["mse"]),
        "psnr": cost["psnr"],
        "c_mean": float(np.mean(c_map)),
        "c_median": float(np.median(c_map)),
        "c_min": float(np.min(c_map)),
        "c_max": float(np.max(c_map)),
        "confidence_mean": float(np.mean(conf_map)),
        "confidence_median": float(np.median(conf_map)),
    }

    return out_pred, out_valid, c_map, conf_map, {
        "summary": summary,
        "blocks": block_records,
    }


def framewise_c_refine(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    c_min: float,
    c_max: float,
    c_samples: int,
    c_search_iters: int,
    min_valid_ratio: float,
    min_range_width: float,
):
    height, width = target_y.shape

    # Treat the whole picture as one block.
    res = search_best_c_for_block(
        target_y=target_y,
        ref_y=ref_y,
        xs=xs,
        ys=ys,
        flow_x=flow_x,
        flow_y=flow_y,
        bx=0,
        by=0,
        bw=width,
        bh=height,
        c_min=c_min,
        c_max=c_max,
        c_samples=c_samples,
        c_search_iters=c_search_iters,
        min_block_valid_ratio=min_valid_ratio,
        min_range_width=min_range_width,
    )

    pred = res["pred"]
    valid = res["valid"]

    cost = calc_cost(target_y, pred, valid, min_valid_ratio=0.0)

    meta = {
        "best_c": float(res["best_c"]),
        "best_cost_mae": float(res["best_cost_mae"]),
        "second_best_cost_mae": float(res["second_best_cost_mae"]),
        "confidence_second_minus_best": float(res["confidence_second_minus_best"]),
        "valid_ratio": float(cost["valid_ratio"]),
        "mae": float(cost["mae"]),
        "mse": float(cost["mse"]),
        "psnr": cost["psnr"],
        "history": res["history"],
    }

    return pred, valid, meta


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


def save_c_map_png(path: str, c_map: np.ndarray):
    c = c_map.astype(np.float32)

    c_min = float(np.min(c))
    c_max = float(np.max(c))

    if abs(c_max - c_min) < 1e-12:
        c8 = np.full_like(c, 128, dtype=np.uint8)
    else:
        c8 = np.clip((c - c_min) / (c_max - c_min) * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(c8, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, color)


def save_conf_png(path: str, conf_map: np.ndarray):
    conf = conf_map.astype(np.float32)
    p99 = float(np.percentile(conf, 99))
    p99 = max(p99, 1e-6)

    conf8 = np.clip(conf / p99 * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(conf8, cv2.COLORMAP_VIRIDIS)
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

    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--match-ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=2.0)
    parser.add_argument("--no-clahe", action="store_true")

    parser.add_argument("--block-size", type=int, default=64)

    parser.add_argument("--c-min", type=float, default=0.0)
    parser.add_argument("--c-max", type=float, default=2.0)
    parser.add_argument("--c-samples", type=int, default=9)
    parser.add_argument("--c-search-iters", type=int, default=4)
    parser.add_argument(
        "--min-range-width",
        type=float,
        default=1e-4,
        help="Minimum C search range width during refinement.",
    )

    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.c_max < args.c_min:
        raise ValueError("--c-max must be >= --c-min")

    print(f"[INFO] target_idx={args.target_idx}, ref_idx={args.ref_idx}")
    print(f"[INFO] resolution={args.width}x{args.height}, bitdepth={args.bitdepth}")
    print(f"[INFO] block_size={args.block_size}")
    print(f"[INFO] c range=[{args.c_min}, {args.c_max}], samples={args.c_samples}, iters={args.c_search_iters}")

    target = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.target_idx)
    ref = read_y_frame(args.input, args.width, args.height, args.bitdepth, args.ref_idx)

    # ------------------------------------------------------------
    # Match + estimate homography.
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

    H, h_mask = estimate_homography(
        pts_target=match_result.pts_target,
        pts_ref=match_result.pts_ref,
        ransac_thresh=args.ransac_thresh,
    )

    h_inliers = int(np.count_nonzero(h_mask)) if h_mask is not None else 0

    print(f"[INFO] homography inliers = {h_inliers}")
    print("[INFO] H target->ref:")
    print(H)

    save_match_vis(
        os.path.join(args.output_dir, "match_vis_homography_inliers.png"),
        target.y,
        ref.y,
        args.bitdepth,
        match_result,
        h_mask,
    )

    # ------------------------------------------------------------
    # Build homography flow.
    # ------------------------------------------------------------

    xs, ys, flow_x, flow_y, hom_x, hom_y, hom_valid = build_homography_flow(
        H=H,
        width=args.width,
        height=args.height,
    )

    pred_hom = remap_full(ref.y, hom_x, hom_y)
    hom_cost = calc_cost(target.y, pred_hom, hom_valid, args.min_valid_ratio)

    print(f"[INFO] homography c=1 cost: {hom_cost}")

    # Identity/no warp baseline.
    pred_identity = remap_full(ref.y, xs, ys)
    identity_valid = valid_from_map(xs, ys, args.width, args.height)
    identity_cost = calc_cost(target.y, pred_identity, identity_valid, args.min_valid_ratio)

    print(f"[INFO] identity c=0 cost: {identity_cost}")

    # ------------------------------------------------------------
    # Frame-level C coarse-to-fine.
    # ------------------------------------------------------------

    pred_frame_c, valid_frame_c, frame_c_meta = framewise_c_refine(
        target_y=target.y,
        ref_y=ref.y,
        xs=xs,
        ys=ys,
        flow_x=flow_x,
        flow_y=flow_y,
        c_min=args.c_min,
        c_max=args.c_max,
        c_samples=args.c_samples,
        c_search_iters=args.c_search_iters,
        min_valid_ratio=args.min_valid_ratio,
        min_range_width=args.min_range_width,
    )

    print("[INFO] frame-level C refine:")
    print(json.dumps(frame_c_meta, indent=2))

    # ------------------------------------------------------------
    # Block-level C coarse-to-fine.
    # ------------------------------------------------------------

    pred_block_c, valid_block_c, c_map, conf_map, block_meta = blockwise_c_refine(
        target_y=target.y,
        ref_y=ref.y,
        xs=xs,
        ys=ys,
        flow_x=flow_x,
        flow_y=flow_y,
        block_size=args.block_size,
        c_min=args.c_min,
        c_max=args.c_max,
        c_samples=args.c_samples,
        c_search_iters=args.c_search_iters,
        min_block_valid_ratio=args.min_block_valid_ratio,
        min_range_width=args.min_range_width,
    )

    print("[INFO] block-level C summary:")
    print(json.dumps(block_meta["summary"], indent=2))

    # ------------------------------------------------------------
    # Save YUV outputs.
    # ------------------------------------------------------------

    paths = {
        "target_yuv": os.path.join(args.output_dir, "target_pair.yuv"),
        "ref_yuv": os.path.join(args.output_dir, "ref_pair.yuv"),
        "pred_identity_yuv": os.path.join(args.output_dir, "pred_identity_c0.yuv"),
        "pred_homography_yuv": os.path.join(args.output_dir, "pred_homography_c1.yuv"),
        "pred_frame_c_yuv": os.path.join(args.output_dir, "pred_frameC_refined.yuv"),
        "pred_block_c_yuv": os.path.join(args.output_dir, f"pred_block{args.block_size}C_refined.yuv"),
    }

    write_single_yuv420(paths["target_yuv"], target.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["ref_yuv"], ref.y, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_identity_yuv"], pred_identity, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_homography_yuv"], pred_hom, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_frame_c_yuv"], pred_frame_c, args.width, args.height, args.bitdepth)
    write_single_yuv420(paths["pred_block_c_yuv"], pred_block_c, args.width, args.height, args.bitdepth)

    # ------------------------------------------------------------
    # Save PNG outputs.
    # ------------------------------------------------------------

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_identity_c0.png"), pred_identity, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_homography_c1.png"), pred_hom, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_frameC_refined.png"), pred_frame_c, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, f"pred_block{args.block_size}C_refined.png"), pred_block_c, args.bitdepth)

    save_diff_png(os.path.join(args.output_dir, "diff_identity_c0.png"), target.y, pred_identity, identity_valid)
    save_diff_png(os.path.join(args.output_dir, "diff_homography_c1.png"), target.y, pred_hom, hom_valid)
    save_diff_png(os.path.join(args.output_dir, "diff_frameC_refined.png"), target.y, pred_frame_c, valid_frame_c)
    save_diff_png(os.path.join(args.output_dir, f"diff_block{args.block_size}C_refined.png"), target.y, pred_block_c, valid_block_c)

    save_c_map_png(os.path.join(args.output_dir, f"c_map_block{args.block_size}.png"), c_map)
    save_conf_png(os.path.join(args.output_dir, f"c_confidence_block{args.block_size}.png"), conf_map)

    # ------------------------------------------------------------
    # Save JSON.
    # ------------------------------------------------------------

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "format": "yuv420p" if args.bitdepth == 8 else "yuv420p10le",

        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "model": "p_pred = p_identity + c_block * (p_homography - p_identity)",
        "c_meaning": {
            "c=0": "identity / no warp",
            "c=1": "homography",
            "c<1": "weaker than homography",
            "c>1": "stronger than homography along the same flow direction",
        },

        "matching": {
            "num_good_matches": int(len(match_result.good_matches)),
            "homography_inliers": int(h_inliers),
            "ransac_thresh": float(args.ransac_thresh),
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "clahe": bool(not args.no_clahe),
        },

        "homography_H_target_to_ref": H.tolist(),

        "search": {
            "block_size": int(args.block_size),
            "c_min": float(args.c_min),
            "c_max": float(args.c_max),
            "c_samples": int(args.c_samples),
            "c_search_iters": int(args.c_search_iters),
            "min_range_width": float(args.min_range_width),
            "min_valid_ratio": float(args.min_valid_ratio),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
        },

        "costs": {
            "identity_c0": identity_cost,
            "homography_c1": hom_cost,
            "frame_c_refined": frame_c_meta,
            "block_c_refined_summary": block_meta["summary"],
        },

        "outputs": paths,

        "png_outputs": {
            "match_vis": os.path.join(args.output_dir, "match_vis_homography_inliers.png"),
            "target_png": os.path.join(args.output_dir, "target.png"),
            "ref_png": os.path.join(args.output_dir, "ref.png"),
            "pred_identity_png": os.path.join(args.output_dir, "pred_identity_c0.png"),
            "pred_homography_png": os.path.join(args.output_dir, "pred_homography_c1.png"),
            "pred_frame_c_png": os.path.join(args.output_dir, "pred_frameC_refined.png"),
            "pred_block_c_png": os.path.join(args.output_dir, f"pred_block{args.block_size}C_refined.png"),
            "diff_identity_png": os.path.join(args.output_dir, "diff_identity_c0.png"),
            "diff_homography_png": os.path.join(args.output_dir, "diff_homography_c1.png"),
            "diff_frame_c_png": os.path.join(args.output_dir, "diff_frameC_refined.png"),
            "diff_block_c_png": os.path.join(args.output_dir, f"diff_block{args.block_size}C_refined.png"),
            "c_map_png": os.path.join(args.output_dir, f"c_map_block{args.block_size}.png"),
            "c_confidence_png": os.path.join(args.output_dir, f"c_confidence_block{args.block_size}.png"),
        },

        "block_records": block_meta["blocks"],

        "notes": [
            "Affine is not used in this version.",
            "The base direction field is the dense flow from identity to homography.",
            "Block-wise C is found by coarse-to-fine search.",
            "c_search_iters controls how many times the best C neighborhood is refined.",
            "If most blocks select c near 1, the homography itself is already a good global predictor.",
            "If blocks select different C values, the homography flow direction is useful but local strength differs spatially.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] result JSON: {json_path}")
    print(f"[DONE] target YUV: {paths['target_yuv']}")
    print(f"[DONE] homography YUV: {paths['pred_homography_yuv']}")
    print(f"[DONE] frame refined C YUV: {paths['pred_frame_c_yuv']}")
    print(f"[DONE] block refined C YUV: {paths['pred_block_c_yuv']}")


if __name__ == "__main__":
    main()
