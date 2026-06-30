#!/usr/bin/env python3
# estimate_rt_from_yuv420.py
#
# Estimate rough relative pose R,t between target and reference frames
# from a YUV420 file.
#
# Direction:
#   target camera coord -> reference camera coord
#
#   X_ref = R_ref_target * X_target + t_ref_target
#
# This is the direction normally needed for backward projection:
#   target pixel -> target camera ray -> reference camera -> reference pixel
#
# Example:
#   python estimate_rt_from_yuv420.py \
#     --input input.yuv \
#     --width 1920 --height 1080 \
#     --bitdepth 10 \
#     --target-idx 16 \
#     --ref-idx 0 \
#     --output rt_16_to_0.json

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass
class FrameY:
    y: np.ndarray
    y_padded: np.ndarray
    width: int
    height: int
    padded_width: int
    padded_height: int


def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def get_yuv420_frame_size_bytes(width: int, height: int, bitdepth: int) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    num_samples = width * height + 2 * ((width // 2) * (height // 2))

    if bitdepth == 8:
        return num_samples
    elif bitdepth == 10:
        # yuv420p10le: each sample stored in uint16 little-endian
        return num_samples * 2
    else:
        raise ValueError("Only bitdepth 8 or 10 is supported.")


def read_y_frame(
    path: str,
    width: int,
    height: int,
    bitdepth: int,
    frame_idx: int,
    pad_multiple: int = 8,
    pad_mode: str = "edge",
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

            # Keep original 10-bit values in y.
            # Feature extraction later converts to 8-bit.
            y = np.clip(y, 0, 1023).astype(np.uint16)

    padded_width = ceil_to_multiple(width, pad_multiple)
    padded_height = ceil_to_multiple(height, pad_multiple)

    pad_right = padded_width - width
    pad_bottom = padded_height - height

    if pad_right > 0 or pad_bottom > 0:
        y_padded = np.pad(
            y,
            ((0, pad_bottom), (0, pad_right)),
            mode=pad_mode,
        )
    else:
        y_padded = y.copy()

    return FrameY(
        y=y,
        y_padded=y_padded,
        width=width,
        height=height,
        padded_width=padded_width,
        padded_height=padded_height,
    )


def y_to_8bit_for_features(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return y.astype(np.uint8)

    # 10-bit to 8-bit.
    # yuv420p10le normally stores 10-bit value in low bits.
    y8 = (y.astype(np.uint16) >> 2).astype(np.uint8)
    return y8


def build_default_K(width: int, height: int) -> np.ndarray:
    """
    Rough intrinsic if no K is given.

    This is only a rough assumption.
    If real fx/fy/cx/cy are known, pass them from CLI.
    """
    f = float(max(width, height))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    K = np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def build_K_from_args(args) -> np.ndarray:
    if args.fx is None or args.fy is None:
        return build_default_K(args.width, args.height)

    cx = args.cx if args.cx is not None else (args.width - 1) * 0.5
    cy = args.cy if args.cy is not None else (args.height - 1) * 0.5

    K = np.array(
        [
            [float(args.fx), 0.0, float(cx)],
            [0.0, float(args.fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def detect_and_match_orb(
    img_target_8: np.ndarray,
    img_ref_8: np.ndarray,
    max_features: int = 8000,
    ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return matched points:
      pts_target: Nx2
      pts_ref:    Nx2
    """

    # CLAHE helps with low-contrast Y images.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_t = clahe.apply(img_target_8)
    img_r = clahe.apply(img_ref_8)

    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=10,
    )

    kp_t, des_t = orb.detectAndCompute(img_t, None)
    kp_r, des_r = orb.detectAndCompute(img_r, None)

    if des_t is None or des_r is None or len(kp_t) < 8 or len(kp_r) < 8:
        raise RuntimeError("Not enough ORB features detected.")

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

    return pts_target, pts_ref, len(good)


def choose_best_pose_from_E(
    E: np.ndarray,
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    OpenCV can return multiple 3x3 essential matrices stacked vertically.
    Try each and select the one with the largest recoverPose inlier count.

    Direction:
      recoverPose(E, pts_target, pts_ref, K)
      gives R,t such that target camera -> ref camera.
    """

    if E is None:
        raise RuntimeError("findEssentialMat failed: E is None.")

    if E.shape == (3, 3):
        candidates = [E]
    else:
        if E.shape[1] != 3 or E.shape[0] % 3 != 0:
            raise RuntimeError(f"Unexpected E shape: {E.shape}")
        candidates = [E[i : i + 3, :] for i in range(0, E.shape[0], 3)]

    best = None

    for Ei in candidates:
        try:
            inlier_count, R, t, pose_mask = cv2.recoverPose(
                Ei,
                pts_target,
                pts_ref,
                K,
            )
        except cv2.error:
            continue

        if best is None or inlier_count > best[0]:
            best = (inlier_count, R, t, pose_mask)

    if best is None:
        raise RuntimeError("recoverPose failed for all E candidates.")

    inlier_count, R, t, pose_mask = best

    t = t.reshape(3).astype(np.float64)
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-12:
        t = t / t_norm

    return R.astype(np.float64), t, pose_mask, int(inlier_count)


def estimate_rt_target_to_ref(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    K: np.ndarray,
    max_features: int,
    ransac_threshold: float,
    ransac_prob: float,
) -> dict:
    target_8 = y_to_8bit_for_features(target_y, bitdepth)
    ref_8 = y_to_8bit_for_features(ref_y, bitdepth)

    pts_target, pts_ref, num_matches = detect_and_match_orb(
        target_8,
        ref_8,
        max_features=max_features,
    )

    E, inlier_mask_E = cv2.findEssentialMat(
        pts_target,
        pts_ref,
        K,
        method=cv2.RANSAC,
        prob=ransac_prob,
        threshold=ransac_threshold,
    )

    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed.")

    R, t, pose_mask, pose_inliers = choose_best_pose_from_E(
        E,
        pts_target,
        pts_ref,
        K,
    )

    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.reshape(3)

    # Also provide inverse direction just in case.
    # If X_ref = R * X_target + t,
    # then X_target = R.T * X_ref - R.T * t
    R_target_ref = R.T
    t_target_ref = -R.T @ t

    return {
        "num_matches": int(num_matches),
        "essential_inliers": int(inlier_mask_E.sum()) if inlier_mask_E is not None else None,
        "pose_inliers": int(pose_inliers),

        "R_ref_target": R.tolist(),
        "t_ref_target_unit": t.tolist(),
        "rvec_ref_target": rvec.tolist(),

        "R_target_ref": R_target_ref.tolist(),
        "t_target_ref_unit": t_target_ref.tolist(),

        "note": (
            "R_ref_target/t_ref_target maps target camera coordinates to "
            "reference camera coordinates. Translation scale is arbitrary and "
            "normalized to unit length."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input YUV420 file")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)

    parser.add_argument("--target-idx", type=int, required=True)
    parser.add_argument("--ref-idx", type=int, required=True)

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--pad-multiple", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--ransac-prob", type=float, default=0.999)

    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    K = build_K_from_args(args)

    target = read_y_frame(
        args.input,
        args.width,
        args.height,
        args.bitdepth,
        args.target_idx,
        pad_multiple=args.pad_multiple,
    )

    ref = read_y_frame(
        args.input,
        args.width,
        args.height,
        args.bitdepth,
        args.ref_idx,
        pad_multiple=args.pad_multiple,
    )

    result = estimate_rt_target_to_ref(
        target_y=target.y,
        ref_y=ref.y,
        bitdepth=args.bitdepth,
        K=K,
        max_features=args.max_features,
        ransac_threshold=args.ransac_threshold,
        ransac_prob=args.ransac_prob,
    )

    result.update(
        {
            "input": args.input,
            "width": args.width,
            "height": args.height,
            "bitdepth": args.bitdepth,
            "target_idx": args.target_idx,
            "ref_idx": args.ref_idx,
            "padded_width": target.padded_width,
            "padded_height": target.padded_height,
            "pad_multiple": args.pad_multiple,
            "K": K.tolist(),
        }
    )

    print(json.dumps(result, indent=2))

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
