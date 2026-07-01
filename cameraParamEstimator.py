#!/usr/bin/env python3
# pair_affine_c_strength_match.py
#
# Pairwise affine + block-wise C strength matching.
#
# Input:
#   YUV420 8-bit or 10-bit little-endian
#
# For a given target_idx and ref_idx:
#   1. ORB feature matching
#   2. Estimate full affine transform:
#        p_ref_aff = A * p_target
#   3. Estimate homography:
#        p_ref_hom ~ H * p_target
#   4. Build residual direction field:
#        d(x) = p_ref_hom(x) - p_ref_aff(x)
#   5. Block-wise C sweep:
#        p_ref_pred(x) = p_ref_aff(x) + c_b * d(x)
#   6. Dump:
#        target_pair.yuv
#        pred_affine.yuv
#        pred_homography.yuv
#        pred_blockC.yuv
#        c_map.png
#        diff_*.png
#        match_vis.png
#        result.json
#
# Example:
#   python pair_affine_c_strength_match.py \
#     --input input.yuv \
#     --width 1920 \
#     --height 1080 \
#     --bitdepth 10 \
#     --target-idx 1 \
#     --ref-idx 0 \
#     --block-size 64 \
#     --output-dir pair_vis_t1_r0

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    num_matches: int
    keypoints_target: list
    keypoints_ref: list
    good_matches: list


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def get_yuv420_frame_size_bytes(width: int, height: int, bitdepth: int) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    num_samples = width * height + 2 * ((width // 2) * (height // 2))

    if bitdepth == 8:
        return num_samples

    if bitdepth == 10:
        return num_samples * 2

    raise ValueError("Only bitdepth 8 and 10 are supported.")


def read_y_frame(
    path: str,
    width: int,
    height: int,
    bitdepth: int,
    frame_idx: int,
) -> FrameY:
    frame_size = get_yuv420_frame_size_bytes(width, height, bitdepth)
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
            y = np.fromfile(f, dtype=np.uint8, count=y_samples)
            y = y.reshape(height, width)
        else:
            y = np.fromfile(f, dtype="<u2", count=y_samples)
            y = y.reshape(height, width)
            y = np.clip(y, 0, 1023).astype(np.uint16)

    return FrameY(y=y, width=width, height=height)


def write_yuv420_frame_from_y(
    f,
    y: np.ndarray,
    width: int,
    height: int,
    bitdepth: int,
):
    y_crop = np.asarray(y)[:height, :width]

    if bitdepth == 8:
        y_out = np.clip(np.rint(y_crop), 0, 255).astype(np.uint8)
        uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)

        f.write(y_out.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())
        return

    if bitdepth == 10:
        y_out = np.clip(np.rint(y_crop), 0, 1023).astype("<u2")
        uv = np.full((height // 2, width // 2), 512, dtype="<u2")

        f.write(y_out.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())
        return

    raise ValueError("Only bitdepth 8 and 10 are supported.")


def write_single_frame_yuv420(
    path: str,
    y: np.ndarray,
    width: int,
    height: int,
    bitdepth: int,
):
    with open(path, "wb") as f:
        write_yuv420_frame_from_y(f, y, width, height, bitdepth)


def y_to_8bit(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return np.clip(y, 0, 255).astype(np.uint8)

    return np.clip(np.asarray(y) / 4.0, 0, 255).astype(np.uint8)


def y_to_8bit_for_feature(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return y.astype(np.uint8)

    return (y.astype(np.uint16) >> 2).astype(np.uint8)


# ============================================================
# Matching
# ============================================================

def detect_and_match_orb(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    max_features: int,
    ratio: float,
) -> MatchResult:
    target_8 = y_to_8bit_for_feature(target_y, bitdepth)
    ref_8 = y_to_8bit_for_feature(ref_y, bitdepth)

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
        raise RuntimeError("ORB descriptor extraction failed.")

    if len(kp_t) < 8 or len(kp_r) < 8:
        raise RuntimeError("Not enough ORB keypoints.")

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
        num_matches=len(good),
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
    affine_inlier_mask: Optional[np.ndarray],
    max_draw: int = 200,
):
    target_8 = y_to_8bit(target_y, bitdepth)
    ref_8 = y_to_8bit(ref_y, bitdepth)

    matches = match_result.good_matches

    if affine_inlier_mask is not None:
        mask = np.asarray(affine_inlier_mask).reshape(-1) != 0
        inlier_matches = [m for m, ok in zip(matches, mask) if ok]
    else:
        inlier_matches = matches

    inlier_matches = inlier_matches[:max_draw]

    vis = cv2.drawMatches(
        target_8,
        match_result.keypoints_target,
        ref_8,
        match_result.keypoints_ref,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(out_path, vis)


# ============================================================
# Transform estimation
# ============================================================

def estimate_affine_and_homography(
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    ransac_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A, mask_aff = cv2.estimateAffine2D(
        pts_target,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )

    if A is None:
        raise RuntimeError("cv2.estimateAffine2D failed.")

    H, mask_h = cv2.findHomography(
        pts_target,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
    )

    if H is None:
        raise RuntimeError("cv2.findHomography failed.")

    return A.astype(np.float64), H.astype(np.float64), mask_aff, mask_h


# ============================================================
# Dense mapping
# ============================================================

def make_grid(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    return xs, ys


def affine_maps(
    A: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = make_grid(width, height)

    x_ref = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    y_ref = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]

    valid = (
        (x_ref >= 0.0)
        & (x_ref <= width - 1.0)
        & (y_ref >= 0.0)
        & (y_ref <= height - 1.0)
    )

    return x_ref.astype(np.float32), y_ref.astype(np.float32), valid


def homography_maps(
    H: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = make_grid(width, height)

    denom = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    denom_safe = denom + 1e-12

    x_ref = (H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / denom_safe
    y_ref = (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / denom_safe

    valid = (
        (np.abs(denom) > 1e-9)
        & (x_ref >= 0.0)
        & (x_ref <= width - 1.0)
        & (y_ref >= 0.0)
        & (y_ref <= height - 1.0)
    )

    return x_ref.astype(np.float32), y_ref.astype(np.float32), valid


def remap_with_maps(
    ref_y: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    pred = cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return pred


def calc_cost(
    target_y: np.ndarray,
    pred_y: np.ndarray,
    valid: np.ndarray,
    min_valid_ratio: float,
) -> dict:
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


# ============================================================
# C strength model
# ============================================================

def build_c_maps_and_candidates(
    A: np.ndarray,
    H: np.ndarray,
    width: int,
    height: int,
    c_list: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    aff_x, aff_y, valid_aff = affine_maps(A, width, height)
    hom_x, hom_y, valid_hom = homography_maps(H, width, height)

    dx = hom_x.astype(np.float32) - aff_x.astype(np.float32)
    dy = hom_y.astype(np.float32) - aff_y.astype(np.float32)

    candidates = []

    for c in c_list:
        map_x = aff_x + float(c) * dx
        map_y = aff_y + float(c) * dy

        valid = (
            valid_aff
            & valid_hom
            & (map_x >= 0.0)
            & (map_x <= width - 1.0)
            & (map_y >= 0.0)
            & (map_y <= height - 1.0)
        )

        candidates.append(
            {
                "c": float(c),
                "map_x": map_x.astype(np.float32),
                "map_y": map_y.astype(np.float32),
                "valid": valid,
            }
        )

    return aff_x, aff_y, dx, dy, candidates


def frame_best_c(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    candidates: List[dict],
    min_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    best = None

    for cand in candidates:
        pred = remap_with_maps(ref_y, cand["map_x"], cand["map_y"])
        cost = calc_cost(target_y, pred, cand["valid"], min_valid_ratio)

        meta = {
            "c": float(cand["c"]),
            **cost,
        }

        if best is None or meta["mae"] < best["meta"]["mae"]:
            best = {
                "pred": pred,
                "valid": cand["valid"],
                "meta": meta,
            }

    if best is None:
        raise RuntimeError("No frame C candidate evaluated.")

    return best["pred"], best["valid"], best["meta"]


def block_best_c(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    candidates: List[dict],
    block_size: int,
    min_block_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    height, width = target_y.shape
    target_f = target_y.astype(np.float32)

    cand_preds = []
    for cand in candidates:
        pred = remap_with_maps(ref_y, cand["map_x"], cand["map_y"])
        cand_preds.append(pred)

    out_pred = np.zeros((height, width), dtype=np.float32)
    out_valid = np.zeros((height, width), dtype=bool)
    c_map = np.zeros((height, width), dtype=np.float32)
    confidence_map = np.zeros((height, width), dtype=np.float32)

    block_records = []

    total_blocks = 0
    valid_blocks = 0

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            y1 = min(by + block_size, height)
            x1 = min(bx + block_size, width)

            target_blk = target_f[by:y1, bx:x1]

            costs = []

            for idx, cand in enumerate(candidates):
                valid_blk = cand["valid"][by:y1, bx:x1]
                valid_ratio = float(np.mean(valid_blk))

                if valid_ratio < min_block_valid_ratio or not np.any(valid_blk):
                    cost = float("inf")
                else:
                    pred_blk = cand_preds[idx][by:y1, bx:x1]
                    diff = target_blk[valid_blk] - pred_blk[valid_blk]
                    cost = float(np.mean(np.abs(diff)))

                costs.append(cost)

            order = np.argsort(np.asarray(costs, dtype=np.float64))
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else best_idx

            best_cost = float(costs[best_idx])
            second_cost = float(costs[second_idx])

            if not np.isfinite(best_cost):
                best_idx = 0
                best_cost = float("inf")
                second_cost = float("inf")
                confidence = 0.0
            else:
                valid_blocks += 1
                if np.isfinite(second_cost):
                    confidence = max(0.0, second_cost - best_cost)
                else:
                    confidence = 0.0

            best_cand = candidates[best_idx]
            best_pred = cand_preds[best_idx]
            best_valid = best_cand["valid"]

            c_val = float(best_cand["c"])

            out_pred[by:y1, bx:x1] = best_pred[by:y1, bx:x1]
            out_valid[by:y1, bx:x1] = best_valid[by:y1, bx:x1]
            c_map[by:y1, bx:x1] = c_val
            confidence_map[by:y1, bx:x1] = confidence

            block_records.append(
                {
                    "bx": int(bx),
                    "by": int(by),
                    "w": int(x1 - bx),
                    "h": int(y1 - by),
                    "best_c": c_val,
                    "best_cost_mae": best_cost,
                    "second_best_cost_mae": second_cost,
                    "confidence_second_minus_best": float(confidence),
                }
            )

            total_blocks += 1

    cost = calc_cost(target_y, out_pred, out_valid, min_valid_ratio=0.0)

    summary = {
        "block_size": int(block_size),
        "num_blocks": int(total_blocks),
        "valid_block_ratio": float(valid_blocks / max(total_blocks, 1)),
        "valid_ratio": cost["valid_ratio"],
        "mae": cost["mae"],
        "mse": cost["mse"],
        "psnr": cost["psnr"],
        "c_mean": float(np.mean(c_map)),
        "c_median": float(np.median(c_map)),
        "c_min": float(np.min(c_map)),
        "c_max": float(np.max(c_map)),
        "confidence_mean": float(np.mean(confidence_map)),
        "confidence_median": float(np.median(confidence_map)),
    }

    return out_pred, out_valid, c_map, confidence_map, {
        "summary": summary,
        "blocks": block_records,
    }


# ============================================================
# Visualization
# ============================================================

def save_diff_png(
    path: str,
    target_y: np.ndarray,
    pred_y: np.ndarray,
    valid: np.ndarray,
    bitdepth: int,
):
    diff = np.abs(target_y.astype(np.float32) - pred_y.astype(np.float32))

    if np.any(valid):
        scale = float(np.percentile(diff[valid], 99))
    else:
        scale = float(np.percentile(diff, 99))

    scale = max(scale, 1.0)

    diff8 = np.clip(diff / scale * 255.0, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff8, cv2.COLORMAP_JET)

    invalid = ~valid
    diff_color[invalid] = (0, 0, 0)

    cv2.imwrite(path, diff_color)


def save_c_map_png(path: str, c_map: np.ndarray):
    c = c_map.astype(np.float32)

    c_min = float(np.min(c))
    c_max = float(np.max(c))

    if abs(c_max - c_min) < 1e-12:
        c_norm = np.full_like(c, 128, dtype=np.uint8)
    else:
        c_norm = np.clip((c - c_min) / (c_max - c_min) * 255.0, 0, 255).astype(np.uint8)

    c_color = cv2.applyColorMap(c_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, c_color)


def save_confidence_png(path: str, conf_map: np.ndarray):
    conf = conf_map.astype(np.float32)

    p99 = float(np.percentile(conf, 99))
    p99 = max(p99, 1e-6)

    conf8 = np.clip(conf / p99 * 255.0, 0, 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf8, cv2.COLORMAP_VIRIDIS)

    cv2.imwrite(path, conf_color)


def save_gray_png(path: str, y: np.ndarray, bitdepth: int):
    cv2.imwrite(path, y_to_8bit(y, bitdepth))


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

    parser.add_argument("--block-size", type=int, default=64)

    parser.add_argument(
        "--c-list",
        default="-1.0,-0.5,0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="C strength candidates. c=0 affine only, c=1 homography.",
    )

    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    c_list = parse_float_list(args.c_list)

    target = read_y_frame(
        args.input,
        args.width,
        args.height,
        args.bitdepth,
        args.target_idx,
    )

    ref = read_y_frame(
        args.input,
        args.width,
        args.height,
        args.bitdepth,
        args.ref_idx,
    )

    print(f"[INFO] target_idx={args.target_idx}, ref_idx={args.ref_idx}")
    print(f"[INFO] resolution={args.width}x{args.height}, bitdepth={args.bitdepth}")
    print(f"[INFO] block_size={args.block_size}")
    print(f"[INFO] c_list={c_list}")

    # ------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------

    match_result = detect_and_match_orb(
        target.y,
        ref.y,
        bitdepth=args.bitdepth,
        max_features=args.max_features,
        ratio=args.match_ratio,
    )

    print(f"[INFO] good matches = {match_result.num_matches}")

    # ------------------------------------------------------------
    # Estimate affine and homography.
    # ------------------------------------------------------------

    A, H, mask_aff, mask_h = estimate_affine_and_homography(
        match_result.pts_target,
        match_result.pts_ref,
        ransac_thresh=args.ransac_thresh,
    )

    aff_inliers = int(np.count_nonzero(mask_aff)) if mask_aff is not None else 0
    hom_inliers = int(np.count_nonzero(mask_h)) if mask_h is not None else 0

    print(f"[INFO] affine inliers = {aff_inliers}")
    print(f"[INFO] homography inliers = {hom_inliers}")
    print("[INFO] affine A:")
    print(A)
    print("[INFO] homography H:")
    print(H)

    save_match_vis(
        os.path.join(args.output_dir, "match_vis_affine_inliers.png"),
        target.y,
        ref.y,
        args.bitdepth,
        match_result,
        mask_aff,
    )

    # ------------------------------------------------------------
    # Build maps and candidates.
    # ------------------------------------------------------------

    aff_x, aff_y, dx, dy, candidates = build_c_maps_and_candidates(
        A=A,
        H=H,
        width=args.width,
        height=args.height,
        c_list=c_list,
    )

    pred_aff = remap_with_maps(ref.y, aff_x, aff_y)
    valid_aff = (
        (aff_x >= 0.0)
        & (aff_x <= args.width - 1.0)
        & (aff_y >= 0.0)
        & (aff_y <= args.height - 1.0)
    )

    hom_x, hom_y, valid_hom = homography_maps(H, args.width, args.height)
    pred_hom = remap_with_maps(ref.y, hom_x, hom_y)

    aff_cost = calc_cost(target.y, pred_aff, valid_aff, args.min_valid_ratio)
    hom_cost = calc_cost(target.y, pred_hom, valid_hom, args.min_valid_ratio)

    print(f"[INFO] affine cost: {aff_cost}")
    print(f"[INFO] homography cost: {hom_cost}")

    # ------------------------------------------------------------
    # Frame-level best C.
    # ------------------------------------------------------------

    pred_frame_c, valid_frame_c, frame_c_meta = frame_best_c(
        target.y,
        ref.y,
        candidates,
        min_valid_ratio=args.min_valid_ratio,
    )

    print(f"[INFO] frame best C: {frame_c_meta}")

    # ------------------------------------------------------------
    # Block-level best C.
    # ------------------------------------------------------------

    pred_block_c, valid_block_c, c_map, conf_map, block_c_meta = block_best_c(
        target.y,
        ref.y,
        candidates,
        block_size=args.block_size,
        min_block_valid_ratio=args.min_block_valid_ratio,
    )

    print("[INFO] block C summary:")
    print(json.dumps(block_c_meta["summary"], indent=2))

    # ------------------------------------------------------------
    # Save YUV.
    # ------------------------------------------------------------

    paths = {
        "target_yuv": os.path.join(args.output_dir, "target_pair.yuv"),
        "ref_yuv": os.path.join(args.output_dir, "ref_pair.yuv"),
        "pred_affine_yuv": os.path.join(args.output_dir, "pred_affine.yuv"),
        "pred_homography_yuv": os.path.join(args.output_dir, "pred_homography.yuv"),
        "pred_frame_c_yuv": os.path.join(args.output_dir, "pred_frameC.yuv"),
        "pred_block_c_yuv": os.path.join(args.output_dir, f"pred_block{args.block_size}C.yuv"),
    }

    write_single_frame_yuv420(
        paths["target_yuv"],
        target.y,
        args.width,
        args.height,
        args.bitdepth,
    )

    write_single_frame_yuv420(
        paths["ref_yuv"],
        ref.y,
        args.width,
        args.height,
        args.bitdepth,
    )

    write_single_frame_yuv420(
        paths["pred_affine_yuv"],
        pred_aff,
        args.width,
        args.height,
        args.bitdepth,
    )

    write_single_frame_yuv420(
        paths["pred_homography_yuv"],
        pred_hom,
        args.width,
        args.height,
        args.bitdepth,
    )

    write_single_frame_yuv420(
        paths["pred_frame_c_yuv"],
        pred_frame_c,
        args.width,
        args.height,
        args.bitdepth,
    )

    write_single_frame_yuv420(
        paths["pred_block_c_yuv"],
        pred_block_c,
        args.width,
        args.height,
        args.bitdepth,
    )

    # ------------------------------------------------------------
    # Save PNG helpers.
    # ------------------------------------------------------------

    save_gray_png(os.path.join(args.output_dir, "target.png"), target.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "ref.png"), ref.y, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_affine.png"), pred_aff, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_homography.png"), pred_hom, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, "pred_frameC.png"), pred_frame_c, args.bitdepth)
    save_gray_png(os.path.join(args.output_dir, f"pred_block{args.block_size}C.png"), pred_block_c, args.bitdepth)

    save_diff_png(
        os.path.join(args.output_dir, "diff_affine.png"),
        target.y,
        pred_aff,
        valid_aff,
        args.bitdepth,
    )

    save_diff_png(
        os.path.join(args.output_dir, "diff_homography.png"),
        target.y,
        pred_hom,
        valid_hom,
        args.bitdepth,
    )

    save_diff_png(
        os.path.join(args.output_dir, "diff_frameC.png"),
        target.y,
        pred_frame_c,
        valid_frame_c,
        args.bitdepth,
    )

    save_diff_png(
        os.path.join(args.output_dir, f"diff_block{args.block_size}C.png"),
        target.y,
        pred_block_c,
        valid_block_c,
        args.bitdepth,
    )

    save_c_map_png(os.path.join(args.output_dir, f"c_map_block{args.block_size}.png"), c_map)
    save_confidence_png(os.path.join(args.output_dir, f"c_confidence_block{args.block_size}.png"), conf_map)

    # ------------------------------------------------------------
    # Export JSON.
    # ------------------------------------------------------------

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "format": "yuv420p" if args.bitdepth == 8 else "yuv420p10le",

        "target_idx": int(args.target_idx),
        "ref_idx": int(args.ref_idx),

        "model": "p_pred = p_affine + c_block * (p_homography - p_affine)",

        "matching": {
            "num_good_matches": int(match_result.num_matches),
            "affine_inliers": int(aff_inliers),
            "homography_inliers": int(hom_inliers),
            "ransac_thresh": float(args.ransac_thresh),
        },

        "affine_A_2x3": A.tolist(),
        "homography_H_3x3": H.tolist(),

        "c_list": c_list,
        "block_size": int(args.block_size),

        "costs": {
            "affine": aff_cost,
            "homography": hom_cost,
            "frame_best_c": frame_c_meta,
            "block_best_c_summary": block_c_meta["summary"],
        },

        "outputs": paths,

        "png_outputs": {
            "match_vis": os.path.join(args.output_dir, "match_vis_affine_inliers.png"),
            "target_png": os.path.join(args.output_dir, "target.png"),
            "ref_png": os.path.join(args.output_dir, "ref.png"),
            "pred_affine_png": os.path.join(args.output_dir, "pred_affine.png"),
            "pred_homography_png": os.path.join(args.output_dir, "pred_homography.png"),
            "pred_frame_c_png": os.path.join(args.output_dir, "pred_frameC.png"),
            "pred_block_c_png": os.path.join(args.output_dir, f"pred_block{args.block_size}C.png"),
            "diff_affine_png": os.path.join(args.output_dir, "diff_affine.png"),
            "diff_homography_png": os.path.join(args.output_dir, "diff_homography.png"),
            "diff_frame_c_png": os.path.join(args.output_dir, "diff_frameC.png"),
            "diff_block_c_png": os.path.join(args.output_dir, f"diff_block{args.block_size}C.png"),
            "c_map_png": os.path.join(args.output_dir, f"c_map_block{args.block_size}.png"),
            "c_confidence_png": os.path.join(args.output_dir, f"c_confidence_block{args.block_size}.png"),
        },

        "block_records": block_c_meta["blocks"],

        "notes": [
            "This script estimates a 6-parameter affine base transform and a homography transform from target to ref.",
            "C is not physical inverse depth here. C is a residual strength parameter.",
            "c=0 means affine-only prediction.",
            "c=1 means homography prediction.",
            "block-wise C selects the best residual strength per block by valid-pixel MAE.",
            "confidence_second_minus_best indicates how clearly a block preferred its chosen C.",
            "Use pred_block*C.yuv and target_pair.yuv in the YUV viewer for direct comparison.",
        ],
    }

    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] wrote result JSON: {json_path}")
    print(f"[DONE] wrote target YUV: {paths['target_yuv']}")
    print(f"[DONE] wrote affine YUV: {paths['pred_affine_yuv']}")
    print(f"[DONE] wrote homography YUV: {paths['pred_homography_yuv']}")
    print(f"[DONE] wrote frame C YUV: {paths['pred_frame_c_yuv']}")
    print(f"[DONE] wrote block C YUV: {paths['pred_block_c_yuv']}")


if __name__ == "__main__":
    main()
