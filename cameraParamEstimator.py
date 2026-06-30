#!/usr/bin/env python3
# estimate_gop_rt_from_yuv420.py
#
# Estimate rough GOP-consistent relative camera poses from a YUV420 file.
#
# Main idea:
#   1. Estimate pairwise R,t between many frame pairs using ORB + Essential matrix.
#   2. Build a pose graph over frames in the GOP.
#   3. Anchor one frame as identity.
#   4. Estimate global poses T_anchor_frame for all frames.
#   5. Export consistent R_ref_target, t_ref_target for any target/ref pair.
#
# Convention:
#   Pairwise transform:
#     X_ref = R_ref_target * X_target + t_ref_target
#
#   Global pose:
#     X_anchor = R_anchor_frame * X_frame + t_anchor_frame
#
#   Then:
#     T_ref_target = inverse(T_anchor_ref) * T_anchor_target
#
# Translation scale note:
#   Monocular Essential matrix gives translation direction only.
#   All exported t vectors are normalized to unit length.
#   Later, block-wise inverse-depth scalar c_b can absorb translation scale:
#     p_ref ~ K * ( R * K^-1 * p_target + c_b * t )
#
# Example:
#   python estimate_gop_rt_from_yuv420.py \
#     --input input.yuv \
#     --width 1920 --height 1080 \
#     --bitdepth 10 \
#     --gop-start 0 --gop-size 33 \
#     --edge-mode hierarchical \
#     --output gop_rt.json

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np


# ============================================================
# Data classes
# ============================================================

@dataclass
class FrameY:
    y: np.ndarray
    y_padded: np.ndarray
    width: int
    height: int
    padded_width: int
    padded_height: int


@dataclass
class PoseEdge:
    target_idx: int
    ref_idx: int

    # Maps target camera coordinates to ref camera coordinates:
    #   X_ref = R_ref_target * X_target + t_ref_target
    R_ref_target: np.ndarray
    t_ref_target: np.ndarray

    num_matches: int
    essential_inliers: int
    pose_inliers: int


# ============================================================
# Basic utilities
# ============================================================

def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def project_rotation_to_so3(R: np.ndarray) -> np.ndarray:
    """Numerically project a near-rotation matrix to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def invert_rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If:
      X_b = R_b_a * X_a + t_b_a
    return inverse:
      X_a = R_a_b * X_b + t_a_b
    """
    R_inv = R.T
    t_inv = -R_inv @ t.reshape(3)
    return R_inv, t_inv


def compose_rt(
    R_b_a: np.ndarray,
    t_b_a: np.ndarray,
    R_a_c: np.ndarray,
    t_a_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose:
      X_b = R_b_a * X_a + t_b_a
      X_a = R_a_c * X_c + t_a_c

    Result:
      X_b = R_b_c * X_c + t_b_c
    """
    R_b_c = R_b_a @ R_a_c
    t_b_c = R_b_a @ t_a_c.reshape(3) + t_b_a.reshape(3)
    return project_rotation_to_so3(R_b_c), t_b_c


def rt_to_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def rt_to_json(R: np.ndarray, t: np.ndarray) -> dict:
    return {
        "R": np.asarray(R, dtype=np.float64).tolist(),
        "t": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
    }


# ============================================================
# YUV reading
# ============================================================

def get_yuv420_frame_size_bytes(width: int, height: int, bitdepth: int) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    num_samples = width * height + 2 * ((width // 2) * (height // 2))

    if bitdepth == 8:
        return num_samples
    if bitdepth == 10:
        return num_samples * 2

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

    # yuv420p10le: usually 10-bit value in low 10 bits.
    return (y.astype(np.uint16) >> 2).astype(np.uint8)


# ============================================================
# Intrinsic K
# ============================================================

def build_default_K(width: int, height: int) -> np.ndarray:
    """
    Rough default intrinsic.
    Real fx/fy/cx/cy should be used when available.
    """
    f = float(max(width, height))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    return np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_K_from_args(args) -> np.ndarray:
    if args.fx is None or args.fy is None:
        return build_default_K(args.width, args.height)

    cx = args.cx if args.cx is not None else (args.width - 1) * 0.5
    cy = args.cy if args.cy is not None else (args.height - 1) * 0.5

    return np.array(
        [
            [float(args.fx), 0.0, float(cx)],
            [0.0, float(args.fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# Feature matching and pairwise pose estimation
# ============================================================

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
    OpenCV may return multiple 3x3 Essential matrices stacked vertically.
    Try all and select the one with the largest recoverPose inlier count.

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
    t = normalize_vec(t)

    return project_rotation_to_so3(R.astype(np.float64)), t, pose_mask, int(inlier_count)


def estimate_pair_edge(
    target_idx: int,
    ref_idx: int,
    target_y: np.ndarray,
    ref_y: np.ndarray,
    bitdepth: int,
    K: np.ndarray,
    max_features: int,
    ransac_threshold: float,
    ransac_prob: float,
    ratio: float,
) -> PoseEdge:
    """
    Estimate:
      X_ref = R_ref_target * X_target + t_ref_target
    """

    target_8 = y_to_8bit_for_features(target_y, bitdepth)
    ref_8 = y_to_8bit_for_features(ref_y, bitdepth)

    pts_target, pts_ref, num_matches = detect_and_match_orb(
        target_8,
        ref_8,
        max_features=max_features,
        ratio=ratio,
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

    essential_inliers = int(np.count_nonzero(inlier_mask_E)) if inlier_mask_E is not None else 0

    return PoseEdge(
        target_idx=target_idx,
        ref_idx=ref_idx,
        R_ref_target=R,
        t_ref_target=t,
        num_matches=int(num_matches),
        essential_inliers=essential_inliers,
        pose_inliers=int(pose_inliers),
    )


# ============================================================
# Edge generation
# ============================================================

def parse_frames(args) -> List[int]:
    if args.frames is not None:
        frames = [int(x.strip()) for x in args.frames.split(",") if x.strip() != ""]
        frames = sorted(set(frames))
        if len(frames) < 2:
            raise ValueError("--frames must contain at least two frame indices.")
        return frames

    if args.gop_start is None or args.gop_size is None:
        raise ValueError("Use either --frames or both --gop-start and --gop-size.")

    return list(range(args.gop_start, args.gop_start + args.gop_size))


def generate_edge_pairs(
    frames: List[int],
    edge_mode: str,
    max_edge_dist: int,
) -> List[Tuple[int, int]]:
    """
    Return list of (target_idx, ref_idx).

    Only one direction is estimated per unordered pair.
    The inverse direction is derived analytically later.
    """

    frames = sorted(frames)
    pairs: Set[Tuple[int, int]] = set()

    if edge_mode == "adjacent":
        for i in range(1, len(frames)):
            target = frames[i]
            ref = frames[i - 1]
            if abs(target - ref) <= max_edge_dist:
                pairs.add((target, ref))

    elif edge_mode == "all":
        for i in range(len(frames)):
            for j in range(i):
                target = frames[i]
                ref = frames[j]
                if abs(target - ref) <= max_edge_dist:
                    pairs.add((target, ref))

    elif edge_mode == "hierarchical":
        def rec(lo: int, hi: int):
            if hi - lo <= 1:
                return

            mid = (lo + hi) // 2

            left = frames[lo]
            center = frames[mid]
            right = frames[hi]

            if abs(center - left) <= max_edge_dist:
                pairs.add((center, left))
            if abs(center - right) <= max_edge_dist:
                pairs.add((center, right))

            rec(lo, mid)
            rec(mid, hi)

        rec(0, len(frames) - 1)

        # Add adjacent edges as stabilizers.
        for i in range(1, len(frames)):
            target = frames[i]
            ref = frames[i - 1]
            if abs(target - ref) <= max_edge_dist:
                pairs.add((target, ref))

    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    return sorted(pairs, key=lambda x: (abs(x[0] - x[1]), x[1], x[0]))


# ============================================================
# Rotation averaging
# ============================================================

def average_rotations(rotations: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Small-spread weighted SO(3) averaging.

    Uses the strongest rotation as local reference R0.
    Each rotation R_i is represented as:
      R_i = Exp(delta_i) * R0
    Then average delta_i in tangent space.
    """

    if len(rotations) == 0:
        return np.eye(3, dtype=np.float64)

    if len(rotations) == 1:
        return project_rotation_to_so3(rotations[0])

    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = np.maximum(weights_np, 1e-6)

    ref_idx = int(np.argmax(weights_np))
    R0 = rotations[ref_idx]

    weighted_delta = np.zeros(3, dtype=np.float64)
    wsum = float(np.sum(weights_np))

    for R, w in zip(rotations, weights_np):
        delta_R = R @ R0.T
        delta_R = project_rotation_to_so3(delta_R)
        rvec, _ = cv2.Rodrigues(delta_R)
        weighted_delta += float(w) * rvec.reshape(3)

    weighted_delta /= max(wsum, 1e-12)

    dR, _ = cv2.Rodrigues(weighted_delta)
    R_avg = dR @ R0

    return project_rotation_to_so3(R_avg)


# ============================================================
# Pose graph
# ============================================================

def build_directed_transform_graph(
    edges: List[PoseEdge],
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]]:
    """
    directed[(src, dst)] = (R_dst_src, t_dst_src, weight)
    meaning:
      X_dst = R_dst_src * X_src + t_dst_src
    """

    directed = {}

    for e in edges:
        w = float(max(e.pose_inliers, 1))

        # target -> ref
        directed[(e.target_idx, e.ref_idx)] = (
            e.R_ref_target,
            e.t_ref_target,
            w,
        )

        # ref -> target
        R_target_ref, t_target_ref = invert_rt(e.R_ref_target, e.t_ref_target)
        directed[(e.ref_idx, e.target_idx)] = (
            R_target_ref,
            normalize_vec(t_target_ref),
            w,
        )

    return directed


def initialize_global_poses_bfs(
    frames: List[int],
    directed: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]],
    anchor_idx: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Global pose:
      X_anchor = R_anchor_frame * X_frame + t_anchor_frame

    Anchor:
      T_anchor_anchor = identity
    """

    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        anchor_idx: (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    }

    # Sort directed edges by confidence, high first.
    directed_items = sorted(
        directed.items(),
        key=lambda kv: kv[1][2],
        reverse=True,
    )

    changed = True
    while changed:
        changed = False

        for (src, dst), (R_dst_src, t_dst_src, _w) in directed_items:
            # If dst pose is known, src pose can be predicted:
            #   P_src = P_dst ◦ T_dst_src
            if dst in poses and src not in poses:
                R_anchor_dst, t_anchor_dst = poses[dst]

                R_anchor_src, t_anchor_src = compose_rt(
                    R_anchor_dst,
                    t_anchor_dst,
                    R_dst_src,
                    t_dst_src,
                )

                poses[src] = (R_anchor_src, t_anchor_src)
                changed = True

    return poses


def refine_global_poses(
    frames: List[int],
    directed: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    anchor_idx: int,
    num_iters: int,
    temporal_smooth: float = 0.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Iteratively update each pose from all neighbor constraints.

    For frame i and neighbor j:
      directed[(i,j)] = T_j_i
      P_i_pred = P_j ◦ T_j_i

    Then average all P_i_pred.
    """

    frames_set = set(frames)

    neighbors: Dict[int, List[int]] = {f: [] for f in frames}
    for (src, dst) in directed.keys():
        if src in frames_set and dst in frames_set:
            neighbors[src].append(dst)

    for _ in range(num_iters):
        new_poses = dict(poses)

        for i in frames:
            if i == anchor_idx:
                new_poses[i] = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
                continue

            if i not in poses:
                continue

            pred_Rs: List[np.ndarray] = []
            pred_ts: List[np.ndarray] = []
            weights: List[float] = []

            for j in neighbors.get(i, []):
                if j not in poses:
                    continue

                # We need transform from i -> j.
                key = (i, j)
                if key not in directed:
                    continue

                R_j_i, t_j_i, w = directed[key]
                R_anchor_j, t_anchor_j = poses[j]

                R_anchor_i_pred, t_anchor_i_pred = compose_rt(
                    R_anchor_j,
                    t_anchor_j,
                    R_j_i,
                    t_j_i,
                )

                pred_Rs.append(R_anchor_i_pred)
                pred_ts.append(t_anchor_i_pred)
                weights.append(w)

            if len(pred_Rs) == 0:
                continue

            R_avg = average_rotations(pred_Rs, weights)

            w_np = np.asarray(weights, dtype=np.float64)
            w_np = np.maximum(w_np, 1e-6)
            t_stack = np.stack(pred_ts, axis=0)
            t_avg = np.sum(t_stack * w_np[:, None], axis=0) / np.sum(w_np)

            # Optional mild temporal smoothing on translation only.
            # This is intentionally weak; translation scale is arbitrary.
            if temporal_smooth > 0.0:
                temporal_preds = []
                for j in (i - 1, i + 1):
                    if j in poses:
                        temporal_preds.append(poses[j][1])
                if len(temporal_preds) > 0:
                    t_temporal = np.mean(np.stack(temporal_preds, axis=0), axis=0)
                    alpha = float(np.clip(temporal_smooth, 0.0, 1.0))
                    t_avg = (1.0 - alpha) * t_avg + alpha * t_temporal

            new_poses[i] = (R_avg, t_avg)

        poses = new_poses

    # Re-anchor exactly.
    if anchor_idx in poses:
        R_anchor, t_anchor = poses[anchor_idx]
        R_inv, t_inv = invert_rt(R_anchor, t_anchor)

        reanchored = {}
        for i, (R_ai, t_ai) in poses.items():
            R_new, t_new = compose_rt(R_inv, t_inv, R_ai, t_ai)
            reanchored[i] = (R_new, t_new)

        poses = reanchored
        poses[anchor_idx] = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    return poses


def relative_from_global_poses(
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    target_idx: int,
    ref_idx: int,
    normalize_translation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given global poses:
      X_anchor = P_frame * X_frame

    Return:
      X_ref = T_ref_target * X_target
    """

    if target_idx not in poses:
        raise KeyError(f"target_idx {target_idx} has no global pose.")
    if ref_idx not in poses:
        raise KeyError(f"ref_idx {ref_idx} has no global pose.")

    R_anchor_target, t_anchor_target = poses[target_idx]
    R_anchor_ref, t_anchor_ref = poses[ref_idx]

    R_ref_anchor, t_ref_anchor = invert_rt(R_anchor_ref, t_anchor_ref)

    R_ref_target, t_ref_target = compose_rt(
        R_ref_anchor,
        t_ref_anchor,
        R_anchor_target,
        t_anchor_target,
    )

    if normalize_translation:
        t_ref_target = normalize_vec(t_ref_target)

    return R_ref_target, t_ref_target


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input YUV420 file")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)

    parser.add_argument("--frames", default=None,
                        help="Comma-separated frame indices, e.g. 0,1,2,4,8,16,32")
    parser.add_argument("--gop-start", type=int, default=None)
    parser.add_argument("--gop-size", type=int, default=None)

    parser.add_argument("--anchor-idx", type=int, default=None)

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--pad-multiple", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--match-ratio", type=float, default=0.75)

    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--ransac-prob", type=float, default=0.999)

    parser.add_argument("--edge-mode", choices=["adjacent", "hierarchical", "all"], default="hierarchical")
    parser.add_argument("--max-edge-dist", type=int, default=32)

    parser.add_argument("--min-pose-inliers", type=int, default=30)
    parser.add_argument("--pose-iters", type=int, default=5)
    parser.add_argument("--temporal-smooth", type=float, default=0.0)

    parser.add_argument("--export-pairs", choices=["none", "edges", "all"], default="all")

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    frames = parse_frames(args)
    anchor_idx = args.anchor_idx if args.anchor_idx is not None else frames[0]

    if anchor_idx not in frames:
        raise ValueError(f"anchor_idx {anchor_idx} is not in frames: {frames}")

    K = build_K_from_args(args)

    padded_width = ceil_to_multiple(args.width, args.pad_multiple)
    padded_height = ceil_to_multiple(args.height, args.pad_multiple)

    print(f"[INFO] frames = {frames}")
    print(f"[INFO] anchor_idx = {anchor_idx}")
    print(f"[INFO] padded = {padded_width}x{padded_height}")
    print(f"[INFO] K =\n{K}")

    # Lazy frame cache.
    frame_cache: Dict[int, FrameY] = {}

    def get_frame(idx: int) -> FrameY:
        if idx not in frame_cache:
            frame_cache[idx] = read_y_frame(
                args.input,
                args.width,
                args.height,
                args.bitdepth,
                idx,
                pad_multiple=args.pad_multiple,
            )
        return frame_cache[idx]

    edge_pairs = generate_edge_pairs(
        frames=frames,
        edge_mode=args.edge_mode,
        max_edge_dist=args.max_edge_dist,
    )

    print(f"[INFO] estimating {len(edge_pairs)} pairwise edges...")

    edges: List[PoseEdge] = []
    failed_edges = []

    for n, (target_idx, ref_idx) in enumerate(edge_pairs):
        print(f"[EDGE {n+1:4d}/{len(edge_pairs):4d}] target={target_idx}, ref={ref_idx}")

        try:
            target = get_frame(target_idx)
            ref = get_frame(ref_idx)

            edge = estimate_pair_edge(
                target_idx=target_idx,
                ref_idx=ref_idx,
                target_y=target.y,
                ref_y=ref.y,
                bitdepth=args.bitdepth,
                K=K,
                max_features=args.max_features,
                ransac_threshold=args.ransac_threshold,
                ransac_prob=args.ransac_prob,
                ratio=args.match_ratio,
            )

            if edge.pose_inliers < args.min_pose_inliers:
                print(
                    f"  [SKIP] pose_inliers={edge.pose_inliers} "
                    f"< min_pose_inliers={args.min_pose_inliers}"
                )
                failed_edges.append({
                    "target_idx": target_idx,
                    "ref_idx": ref_idx,
                    "reason": "too_few_pose_inliers",
                    "pose_inliers": edge.pose_inliers,
                })
                continue

            print(
                f"  [OK] matches={edge.num_matches}, "
                f"E_inliers={edge.essential_inliers}, "
                f"pose_inliers={edge.pose_inliers}"
            )

            edges.append(edge)

        except Exception as e:
            print(f"  [FAIL] {e}")
            failed_edges.append({
                "target_idx": target_idx,
                "ref_idx": ref_idx,
                "reason": str(e),
            })

    if len(edges) == 0:
        raise RuntimeError("No valid pairwise pose edges were estimated.")

    print(f"[INFO] valid edges = {len(edges)}")

    directed = build_directed_transform_graph(edges)

    poses = initialize_global_poses_bfs(
        frames=frames,
        directed=directed,
        anchor_idx=anchor_idx,
    )

    missing = [f for f in frames if f not in poses]
    if len(missing) > 0:
        print(f"[WARN] Some frames are disconnected from the pose graph: {missing}")

    poses = refine_global_poses(
        frames=frames,
        directed=directed,
        poses=poses,
        anchor_idx=anchor_idx,
        num_iters=args.pose_iters,
        temporal_smooth=args.temporal_smooth,
    )

    # Export direct pairwise edges.
    direct_edges_json = []
    for e in edges:
        rvec, _ = cv2.Rodrigues(e.R_ref_target)
        direct_edges_json.append({
            "target_idx": e.target_idx,
            "ref_idx": e.ref_idx,
            "num_matches": e.num_matches,
            "essential_inliers": e.essential_inliers,
            "pose_inliers": e.pose_inliers,
            "R_ref_target": e.R_ref_target.tolist(),
            "t_ref_target_unit": normalize_vec(e.t_ref_target).tolist(),
            "rvec_ref_target": rvec.reshape(3).tolist(),
        })

    # Export global poses.
    global_poses_json = {}
    for f in frames:
        if f not in poses:
            continue

        R_anchor_frame, t_anchor_frame = poses[f]
        rvec, _ = cv2.Rodrigues(R_anchor_frame)

        global_poses_json[str(f)] = {
            "R_anchor_frame": R_anchor_frame.tolist(),
            "t_anchor_frame_arbitrary_scale": t_anchor_frame.reshape(3).tolist(),
            "rvec_anchor_frame": rvec.reshape(3).tolist(),
        }

    # Export consistent pair transforms.
    pair_transforms_json = []

    if args.export_pairs != "none":
        if args.export_pairs == "edges":
            export_pair_list = [(e.target_idx, e.ref_idx) for e in edges]
        else:
            export_pair_list = []
            for target_idx in frames:
                for ref_idx in frames:
                    if target_idx == ref_idx:
                        continue
                    if target_idx in poses and ref_idx in poses:
                        export_pair_list.append((target_idx, ref_idx))

        for target_idx, ref_idx in export_pair_list:
            if target_idx not in poses or ref_idx not in poses:
                continue

            R_ref_target, t_ref_target = relative_from_global_poses(
                poses=poses,
                target_idx=target_idx,
                ref_idx=ref_idx,
                normalize_translation=True,
            )

            rvec, _ = cv2.Rodrigues(R_ref_target)

            pair_transforms_json.append({
                "target_idx": target_idx,
                "ref_idx": ref_idx,
                "R_ref_target": R_ref_target.tolist(),
                "t_ref_target_unit": t_ref_target.reshape(3).tolist(),
                "rvec_ref_target": rvec.reshape(3).tolist(),
                "source": "gop_consistent_pose_graph",
            })

    result = {
        "input": args.input,
        "width": args.width,
        "height": args.height,
        "bitdepth": args.bitdepth,
        "frames": frames,
        "anchor_idx": anchor_idx,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "pad_multiple": args.pad_multiple,
        "block_size_ready": 8,

        "K": K.tolist(),

        "edge_mode": args.edge_mode,
        "max_edge_dist": args.max_edge_dist,
        "min_pose_inliers": args.min_pose_inliers,
        "pose_iters": args.pose_iters,
        "temporal_smooth": args.temporal_smooth,

        "direct_pairwise_edges": direct_edges_json,
        "failed_edges": failed_edges,
        "global_poses_anchor_from_frame": global_poses_json,
        "consistent_pair_transforms": pair_transforms_json,

        "notes": [
            "Global pose convention: X_anchor = R_anchor_frame * X_frame + t_anchor_frame.",
            "Pair transform convention: X_ref = R_ref_target * X_target + t_ref_target.",
            "All pairwise and exported translations are normalized or arbitrary scale because monocular Essential matrix does not recover metric translation scale.",
            "Use R_ref_target and t_ref_target_unit for later backward projection with block-wise inverse-depth scalar c_b.",
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] wrote {args.output}")


if __name__ == "__main__":
    main()
