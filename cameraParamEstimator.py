#!/usr/bin/env python3
# estimate_gop_rt_and_visualize_projection.py
#
# GOP-consistent rough camera pose estimation + backward remap visualization.
#
# Pose convention:
#   Pair transform:
#     X_ref = R_ref_target * X_target + t_ref_target
#
#   Global pose:
#     X_anchor = R_anchor_frame * X_frame + t_anchor_frame
#
#   Pair from global:
#     T_ref_target = inverse(T_anchor_ref) * T_anchor_target
#
# Backward remap convention:
#   For each target pixel p_target:
#     ray_target = K^-1 p_target
#     X_ref      = R_ref_target * ray_target + c * t_ref_target
#     p_ref      = K * X_ref
#     pred_target(p_target) = ref(p_ref)
#
# Visualization rule requested:
#   For each target frame, use only the closer endpoint ref:
#     ref = start_ref_idx or end_ref_idx
#   Usually start_ref_idx=0 and end_ref_idx=32.
#
# Example:
#   python estimate_gop_rt_and_visualize_projection.py \
#     --input input.yuv \
#     --width 1920 --height 1080 \
#     --bitdepth 10 \
#     --gop-start 0 --gop-size 33 \
#     --anchor-idx 0 \
#     --start-ref-idx 0 --end-ref-idx 32 \
#     --edge-mode all \
#     --max-edge-dist 32 \
#     --visualize-targets all \
#     --output-json gop_rt_vis.json \
#     --output-dir vis_out

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

    # X_ref = R_ref_target * X_target + t_ref_target
    R_ref_target: np.ndarray
    t_ref_target: np.ndarray

    num_matches: int
    essential_inliers: int
    pose_inliers: int

    sampson_error_mean: float
    sampson_error_median: float
    sampson_error_p90: float


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
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def invert_rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    t_inv = -R_inv @ np.asarray(t, dtype=np.float64).reshape(3)
    return R_inv, t_inv


def compose_rt(
    R_b_a: np.ndarray,
    t_b_a: np.ndarray,
    R_a_c: np.ndarray,
    t_a_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_b = R_b_a * X_a + t_b_a
    X_a = R_a_c * X_c + t_a_c

    Result:
      X_b = R_b_c * X_c + t_b_c
    """
    R_b_c = R_b_a @ R_a_c
    t_b_c = R_b_a @ np.asarray(t_a_c).reshape(3) + np.asarray(t_b_a).reshape(3)
    return project_rotation_to_so3(R_b_c), t_b_c


def skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = np.asarray(t, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -tz,  ty],
            [tz,   0.0, -tx],
            [-ty,  tx,   0.0],
        ],
        dtype=np.float64,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


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
    return (y.astype(np.uint16) >> 2).astype(np.uint8)


def to_vis8(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return np.clip(y, 0, 255).astype(np.uint8)
    return np.clip(np.asarray(y) / 4.0, 0, 255).astype(np.uint8)


# ============================================================
# Intrinsic K
# ============================================================

def build_default_K(width: int, height: int) -> np.ndarray:
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


def compute_sampson_error_pixels(
    pts_target: np.ndarray,
    pts_ref: np.ndarray,
    K: np.ndarray,
    R_ref_target: np.ndarray,
    t_ref_target: np.ndarray,
) -> np.ndarray:
    t_ref_target = normalize_vec(t_ref_target)

    E = skew(t_ref_target) @ R_ref_target
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv

    ones = np.ones((pts_target.shape[0], 1), dtype=np.float64)

    x1 = np.concatenate([pts_target.astype(np.float64), ones], axis=1)
    x2 = np.concatenate([pts_ref.astype(np.float64), ones], axis=1)

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T

    numerator = np.sum(x2 * Fx1, axis=1) ** 2
    denominator = (
        Fx1[:, 0] ** 2
        + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2
        + Ftx2[:, 1] ** 2
        + 1e-12
    )

    return numerator / denominator


def summarize_sampson_error(
    sampson_error: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    e = np.asarray(sampson_error, dtype=np.float64).reshape(-1)

    if mask is not None:
        m = np.asarray(mask).reshape(-1) != 0
        if m.shape[0] == e.shape[0] and np.count_nonzero(m) > 0:
            e = e[m]

    if e.size == 0:
        return float("nan"), float("nan"), float("nan")

    return (
        float(np.mean(e)),
        float(np.median(e)),
        float(np.percentile(e, 90)),
    )


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

    R, t, _pose_mask, pose_inliers = choose_best_pose_from_E(
        E,
        pts_target,
        pts_ref,
        K,
    )

    essential_inliers = int(np.count_nonzero(inlier_mask_E)) if inlier_mask_E is not None else 0

    sampson_error = compute_sampson_error_pixels(
        pts_target=pts_target,
        pts_ref=pts_ref,
        K=K,
        R_ref_target=R,
        t_ref_target=t,
    )

    sampson_mean, sampson_median, sampson_p90 = summarize_sampson_error(
        sampson_error,
        mask=inlier_mask_E,
    )

    return PoseEdge(
        target_idx=target_idx,
        ref_idx=ref_idx,
        R_ref_target=R,
        t_ref_target=t,
        num_matches=int(num_matches),
        essential_inliers=essential_inliers,
        pose_inliers=int(pose_inliers),
        sampson_error_mean=sampson_mean,
        sampson_error_median=sampson_median,
        sampson_error_p90=sampson_p90,
    )


# ============================================================
# Edge generation
# ============================================================

def parse_frames(args) -> List[int]:
    if args.frames is not None:
        frames = parse_int_list(args.frames)
        frames = sorted(set(frames))
        if len(frames) < 2:
            raise ValueError("--frames must contain at least two frame indices.")
        return frames

    if args.gop_start is None or args.gop_size is None:
        raise ValueError("Use either --frames or both --gop-start and --gop-size.")

    return list(range(args.gop_start, args.gop_start + args.gop_size))


def edge_dist_ok(target: int, ref: int, max_edge_dist: int) -> bool:
    if max_edge_dist <= 0:
        return True
    return abs(target - ref) <= max_edge_dist


def generate_edge_pairs(
    frames: List[int],
    edge_mode: str,
    max_edge_dist: int,
) -> List[Tuple[int, int]]:
    frames = sorted(frames)
    pairs: Set[Tuple[int, int]] = set()

    if edge_mode == "adjacent":
        for i in range(1, len(frames)):
            target = frames[i]
            ref = frames[i - 1]
            if edge_dist_ok(target, ref, max_edge_dist):
                pairs.add((target, ref))

    elif edge_mode == "all":
        for i in range(len(frames)):
            for j in range(i):
                target = frames[i]
                ref = frames[j]
                if edge_dist_ok(target, ref, max_edge_dist):
                    pairs.add((target, ref))

    elif edge_mode == "anchor_all":
        anchor = frames[0]
        for f in frames:
            if f == anchor:
                continue
            if edge_dist_ok(f, anchor, max_edge_dist):
                pairs.add((f, anchor))

    elif edge_mode == "hierarchical":
        def rec(lo: int, hi: int):
            if hi - lo <= 1:
                return

            mid = (lo + hi) // 2

            left = frames[lo]
            center = frames[mid]
            right = frames[hi]

            if edge_dist_ok(center, left, max_edge_dist):
                pairs.add((center, left))
            if edge_dist_ok(center, right, max_edge_dist):
                pairs.add((center, right))

            rec(lo, mid)
            rec(mid, hi)

        rec(0, len(frames) - 1)

        for i in range(1, len(frames)):
            target = frames[i]
            ref = frames[i - 1]
            if edge_dist_ok(target, ref, max_edge_dist):
                pairs.add((target, ref))

    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    return sorted(pairs, key=lambda x: (abs(x[0] - x[1]), x[1], x[0]))


# ============================================================
# Pose graph
# ============================================================

def average_rotations(rotations: List[np.ndarray], weights: List[float]) -> np.ndarray:
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


def build_directed_transform_graph(
    edges: List[PoseEdge],
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]]:
    directed = {}

    for e in edges:
        # Basic confidence weight.
        # Down-weight geometrically suspicious edges by Sampson error.
        sampson_weight = 1.0 / (1.0 + max(e.sampson_error_median, 0.0))
        w = float(max(e.pose_inliers, 1)) * sampson_weight

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
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        anchor_idx: (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    }

    directed_items = sorted(
        directed.items(),
        key=lambda kv: kv[1][2],
        reverse=True,
    )

    changed = True
    while changed:
        changed = False

        for (src, dst), (R_dst_src, t_dst_src, _w) in directed_items:
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
    frames_set = set(frames)

    neighbors: Dict[int, List[int]] = {f: [] for f in frames}
    for (src, dst) in directed.keys():
        if src in frames_set and dst in frames_set:
            neighbors[src].append(dst)

    for _ in range(num_iters):
        new_poses = dict(poses)

        for i in frames:
            if i == anchor_idx:
                new_poses[i] = (
                    np.eye(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                )
                continue

            if i not in poses:
                continue

            pred_Rs: List[np.ndarray] = []
            pred_ts: List[np.ndarray] = []
            weights: List[float] = []

            for j in neighbors.get(i, []):
                if j not in poses:
                    continue

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
        poses[anchor_idx] = (
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

    return poses


def relative_from_global_poses(
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    target_idx: int,
    ref_idx: int,
    normalize_translation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
# Projection / backward remap
# ============================================================

def choose_endpoint_ref(
    target_idx: int,
    start_ref_idx: int,
    end_ref_idx: int,
    tie_ref: str,
) -> int:
    d_start = abs(target_idx - start_ref_idx)
    d_end = abs(target_idx - end_ref_idx)

    if d_start < d_end:
        return start_ref_idx
    if d_end < d_start:
        return end_ref_idx

    if tie_ref == "start":
        return start_ref_idx
    if tie_ref == "end":
        return end_ref_idx

    raise ValueError(f"Unknown tie_ref: {tie_ref}")


def make_projection_maps(
    height: int,
    width: int,
    K: np.ndarray,
    R_ref_target: np.ndarray,
    t_ref_target: np.ndarray,
    c_eff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build maps for:
      target pixel -> ref pixel

    X_ref = R * ray_target + c_eff * t
    p_ref = K * X_ref
    """

    Kinv = np.linalg.inv(K)

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )

    ones = np.ones_like(xs)
    p = np.stack([xs.reshape(-1), ys.reshape(-1), ones.reshape(-1)], axis=0)

    rays = Kinv @ p

    t = normalize_vec(t_ref_target).reshape(3, 1)

    X_ref = R_ref_target @ rays + float(c_eff) * t

    z = X_ref[2, :]
    valid_z = np.abs(z) > 1e-9

    q = K @ X_ref

    x_ref = q[0, :] / (q[2, :] + 1e-12)
    y_ref = q[1, :] / (q[2, :] + 1e-12)

    map_x = x_ref.reshape(height, width).astype(np.float32)
    map_y = y_ref.reshape(height, width).astype(np.float32)

    valid = (
        valid_z.reshape(height, width)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )

    return map_x, map_y, valid


def build_effective_c_candidates(
    c_candidates: List[float],
    t_signs: List[int],
) -> List[Tuple[float, float, int]]:
    """
    Return list of:
      (c_eff, c_abs, sign)

    c_eff = sign * c_abs
    """
    out = []
    seen = set()

    for c in c_candidates:
        c_abs = abs(float(c))
        for s in t_signs:
            sign = 1 if int(s) >= 0 else -1
            c_eff = sign * c_abs

            key = round(c_eff, 12)
            if key in seen:
                continue

            seen.add(key)
            out.append((c_eff, c_abs, sign))

    out = sorted(out, key=lambda x: abs(x[0]))
    return out


def remap_ref_for_candidate(
    ref_y: np.ndarray,
    K: np.ndarray,
    R_ref_target: np.ndarray,
    t_ref_target: np.ndarray,
    c_eff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = ref_y.shape

    map_x, map_y, valid = make_projection_maps(
        height=h,
        width=w,
        K=K,
        R_ref_target=R_ref_target,
        t_ref_target=t_ref_target,
        c_eff=c_eff,
    )

    pred = cv2.remap(
        ref_y.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return pred, valid


def evaluate_frame_level_candidates(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    K: np.ndarray,
    R_ref_target: np.ndarray,
    t_ref_target: np.ndarray,
    candidates: List[Tuple[float, float, int]],
    min_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    best = None

    target_f = target_y.astype(np.float32)

    for c_eff, c_abs, sign in candidates:
        pred, valid = remap_ref_for_candidate(
            ref_y=ref_y,
            K=K,
            R_ref_target=R_ref_target,
            t_ref_target=t_ref_target,
            c_eff=c_eff,
        )

        valid_ratio = float(np.mean(valid))
        if valid_ratio < min_valid_ratio:
            cost = float("inf")
        else:
            diff = np.abs(target_f - pred)
            cost = float(np.mean(diff[valid]))

        if best is None or cost < best["mae_valid"]:
            best = {
                "c_eff": float(c_eff),
                "c_abs": float(c_abs),
                "t_sign": int(sign),
                "mae_valid": cost,
                "valid_ratio": valid_ratio,
                "pred": pred,
                "valid": valid,
            }

    if best is None:
        raise RuntimeError("No frame-level candidate evaluated.")

    meta = dict(best)
    pred = meta.pop("pred")
    valid = meta.pop("valid")

    return pred, valid, meta


def blockwise_best_c_remap(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    K: np.ndarray,
    R_ref_target: np.ndarray,
    t_ref_target: np.ndarray,
    candidates: List[Tuple[float, float, int]],
    block_size: int,
    min_block_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    For each block, choose best c_eff candidate by valid-pixel MAE.
    """

    h, w = target_y.shape
    target_f = target_y.astype(np.float32)

    candidate_preds = []
    candidate_valids = []
    candidate_meta = []

    for c_eff, c_abs, sign in candidates:
        pred, valid = remap_ref_for_candidate(
            ref_y=ref_y,
            K=K,
            R_ref_target=R_ref_target,
            t_ref_target=t_ref_target,
            c_eff=c_eff,
        )
        candidate_preds.append(pred)
        candidate_valids.append(valid)
        candidate_meta.append((c_eff, c_abs, sign))

    out_pred = np.zeros((h, w), dtype=np.float32)
    out_valid = np.zeros((h, w), dtype=bool)
    c_eff_map = np.zeros((h, w), dtype=np.float32)
    sign_map = np.zeros((h, w), dtype=np.int8)
    cost_map = np.full((h // block_size, w // block_size), np.inf, dtype=np.float32)

    block_count = 0
    valid_block_count = 0
    sign_pos_count = 0
    sign_neg_count = 0

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            y1 = min(by + block_size, h)
            x1 = min(bx + block_size, w)

            target_blk = target_f[by:y1, bx:x1]

            best_idx = -1
            best_cost = float("inf")
            best_valid_ratio = 0.0

            for idx, (pred, valid) in enumerate(zip(candidate_preds, candidate_valids)):
                valid_blk = valid[by:y1, bx:x1]
                valid_ratio = float(np.mean(valid_blk))

                if valid_ratio < min_block_valid_ratio:
                    continue

                pred_blk = pred[by:y1, bx:x1]
                diff = np.abs(target_blk - pred_blk)

                cost = float(np.mean(diff[valid_blk]))

                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                    best_valid_ratio = valid_ratio

            block_count += 1

            if best_idx < 0:
                # Fallback to c_eff=0 candidate if everything is invalid.
                best_idx = 0
                best_cost = float("inf")

            c_eff, _c_abs, sign = candidate_meta[best_idx]
            pred = candidate_preds[best_idx]
            valid = candidate_valids[best_idx]

            out_pred[by:y1, bx:x1] = pred[by:y1, bx:x1]
            out_valid[by:y1, bx:x1] = valid[by:y1, bx:x1]
            c_eff_map[by:y1, bx:x1] = float(c_eff)
            sign_map[by:y1, bx:x1] = int(sign)

            cost_map[by // block_size, bx // block_size] = best_cost

            if np.isfinite(best_cost):
                valid_block_count += 1

            if sign >= 0:
                sign_pos_count += 1
            else:
                sign_neg_count += 1

    valid_ratio = float(np.mean(out_valid))

    if np.any(out_valid):
        mae_valid = float(np.mean(np.abs(target_f[out_valid] - out_pred[out_valid])))
        mse_valid = float(np.mean((target_f[out_valid] - out_pred[out_valid]) ** 2))
    else:
        mae_valid = float("inf")
        mse_valid = float("inf")

    summary = {
        "block_size": int(block_size),
        "num_candidates": int(len(candidates)),
        "valid_ratio": valid_ratio,
        "mae_valid": mae_valid,
        "mse_valid": mse_valid,
        "valid_block_ratio": float(valid_block_count / max(block_count, 1)),
        "sign_pos_block_ratio": float(sign_pos_count / max(block_count, 1)),
        "sign_neg_block_ratio": float(sign_neg_count / max(block_count, 1)),
        "c_eff_mean": float(np.mean(c_eff_map)),
        "c_eff_median": float(np.median(c_eff_map)),
        "c_eff_min": float(np.min(c_eff_map)),
        "c_eff_max": float(np.max(c_eff_map)),
    }

    return out_pred, out_valid, c_eff_map, sign_map, summary


# ============================================================
# Visualization saving
# ============================================================

def save_projection_visuals(
    out_dir: str,
    prefix: str,
    target_y: np.ndarray,
    ref_y: np.ndarray,
    pred_y: np.ndarray,
    valid: np.ndarray,
    c_eff_map: Optional[np.ndarray],
    sign_map: Optional[np.ndarray],
    bitdepth: int,
    crop_h: int,
    crop_w: int,
):
    ensure_dir(out_dir)

    target_crop = target_y[:crop_h, :crop_w]
    ref_crop = ref_y[:crop_h, :crop_w]
    pred_crop = pred_y[:crop_h, :crop_w]
    valid_crop = valid[:crop_h, :crop_w]

    target8 = to_vis8(target_crop, bitdepth)
    ref8 = to_vis8(ref_crop, bitdepth)
    pred8 = to_vis8(pred_crop, bitdepth)

    diff = np.abs(target_crop.astype(np.float32) - pred_crop.astype(np.float32))
    diff_vis_scale = np.percentile(diff[valid_crop], 99) if np.any(valid_crop) else np.percentile(diff, 99)
    diff_vis_scale = max(float(diff_vis_scale), 1.0)
    diff8 = np.clip(diff / diff_vis_scale * 255.0, 0, 255).astype(np.uint8)

    valid8 = (valid_crop.astype(np.uint8) * 255)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_target.png"), target8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_ref_raw.png"), ref8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_pred.png"), pred8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_diff.png"), diff8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_valid_mask.png"), valid8)

    target_bgr = cv2.cvtColor(target8, cv2.COLOR_GRAY2BGR)
    ref_bgr = cv2.cvtColor(ref8, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred8, cv2.COLOR_GRAY2BGR)
    diff_bgr = cv2.applyColorMap(diff8, cv2.COLORMAP_JET)
    valid_bgr = cv2.cvtColor(valid8, cv2.COLOR_GRAY2BGR)

    panel = np.concatenate([target_bgr, ref_bgr, pred_bgr, diff_bgr, valid_bgr], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_panel_target_ref_pred_diff_valid.png"), panel)

    if c_eff_map is not None:
        c_crop = c_eff_map[:crop_h, :crop_w].astype(np.float32)

        c_abs_max = max(float(np.percentile(np.abs(c_crop), 99)), 1e-6)
        c_norm = np.clip((c_crop / c_abs_max) * 127.0 + 128.0, 0, 255).astype(np.uint8)
        c_color = cv2.applyColorMap(c_norm, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_c_eff_map.png"), c_color)

    if sign_map is not None:
        s_crop = sign_map[:crop_h, :crop_w]
        sign_vis = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        sign_vis[s_crop > 0] = (0, 255, 0)
        sign_vis[s_crop < 0] = (0, 0, 255)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_sign_map.png"), sign_vis)


# ============================================================
# Evaluation helpers
# ============================================================

def rotation_angle_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = R_a @ R_b.T
    R_delta = project_rotation_to_so3(R_delta)

    cos_theta = (np.trace(R_delta) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_theta)))


def translation_direction_error_deg(t_a: np.ndarray, t_b: np.ndarray) -> float:
    ta = normalize_vec(t_a)
    tb = normalize_vec(t_b)

    if np.linalg.norm(ta) < 1e-12 or np.linalg.norm(tb) < 1e-12:
        return float("nan")

    dot = float(np.dot(ta, tb))
    dot = float(np.clip(dot, -1.0, 1.0))

    return float(np.degrees(np.arccos(dot)))


def evaluate_pose_graph_edges(
    edges: List[PoseEdge],
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> List[dict]:
    stats = []

    for e in edges:
        if e.target_idx not in poses or e.ref_idx not in poses:
            continue

        R_graph, t_graph = relative_from_global_poses(
            poses=poses,
            target_idx=e.target_idx,
            ref_idx=e.ref_idx,
            normalize_translation=True,
        )

        rot_err = rotation_angle_error_deg(R_graph, e.R_ref_target)
        trans_err = translation_direction_error_deg(t_graph, e.t_ref_target)

        stats.append(
            {
                "target_idx": int(e.target_idx),
                "ref_idx": int(e.ref_idx),

                "num_matches": int(e.num_matches),
                "essential_inliers": int(e.essential_inliers),
                "pose_inliers": int(e.pose_inliers),

                "sampson_error_mean": float(e.sampson_error_mean),
                "sampson_error_median": float(e.sampson_error_median),
                "sampson_error_p90": float(e.sampson_error_p90),

                "rotation_error_deg": float(rot_err),
                "translation_dir_error_deg": float(trans_err),
            }
        )

    return stats


def summarize_pose_graph_residuals(residuals: List[dict]) -> dict:
    if len(residuals) == 0:
        return {"num_edges": 0}

    def arr(key):
        x = np.array([r[key] for r in residuals], dtype=np.float64)
        return x[np.isfinite(x)]

    def summary(x):
        if x.size == 0:
            return None, None, None
        return float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 90))

    rot_mean, rot_med, rot_p90 = summary(arr("rotation_error_deg"))
    tr_mean, tr_med, tr_p90 = summary(arr("translation_dir_error_deg"))
    sm_mean, sm_med, sm_p90 = summary(arr("sampson_error_median"))

    return {
        "num_edges": int(len(residuals)),
        "rotation_error_deg_mean": rot_mean,
        "rotation_error_deg_median": rot_med,
        "rotation_error_deg_p90": rot_p90,
        "translation_dir_error_deg_mean": tr_mean,
        "translation_dir_error_deg_median": tr_med,
        "translation_dir_error_deg_p90": tr_p90,
        "sampson_error_median_mean": sm_mean,
        "sampson_error_median_median": sm_med,
        "sampson_error_median_p90": sm_p90,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)

    parser.add_argument("--frames", default=None)
    parser.add_argument("--gop-start", type=int, default=None)
    parser.add_argument("--gop-size", type=int, default=None)

    parser.add_argument("--anchor-idx", type=int, default=None)
    parser.add_argument("--start-ref-idx", type=int, default=None)
    parser.add_argument("--end-ref-idx", type=int, default=None)
    parser.add_argument("--tie-ref", choices=["start", "end"], default="start")

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--pad-multiple", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--match-ratio", type=float, default=0.75)

    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--ransac-prob", type=float, default=0.999)

    parser.add_argument(
        "--edge-mode",
        choices=["adjacent", "hierarchical", "all", "anchor_all"],
        default="all",
    )
    parser.add_argument("--max-edge-dist", type=int, default=32)

    parser.add_argument("--min-pose-inliers", type=int, default=30)
    parser.add_argument("--pose-iters", type=int, default=10)
    parser.add_argument("--temporal-smooth", type=float, default=0.0)

    parser.add_argument("--export-pairs", choices=["none", "edges", "all"], default="all")

    parser.add_argument(
        "--visualize-targets",
        default="all",
        help='all or comma list, e.g. "1,8,16,24,31"',
    )
    parser.add_argument(
        "--skip-endpoints",
        action="store_true",
        help="Skip start/end reference frames when visualize-targets=all.",
    )

    parser.add_argument(
        "--c-list",
        default="0,0.0025,0.005,0.01,0.02,0.04,0.08,0.16",
        help="Positive c candidates. Actual effective c can be sign*c if --t-signs includes -1.",
    )
    parser.add_argument(
        "--t-signs",
        default="1,-1",
        help='Translation signs to test. Use "1" for physical-only direction, "1,-1" for diagnostic.',
    )

    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)

    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    frames = parse_frames(args)

    anchor_idx = args.anchor_idx if args.anchor_idx is not None else frames[0]
    start_ref_idx = args.start_ref_idx if args.start_ref_idx is not None else frames[0]
    end_ref_idx = args.end_ref_idx if args.end_ref_idx is not None else frames[-1]

    if anchor_idx not in frames:
        raise ValueError(f"anchor_idx {anchor_idx} is not in frames.")
    if start_ref_idx not in frames:
        raise ValueError(f"start_ref_idx {start_ref_idx} is not in frames.")
    if end_ref_idx not in frames:
        raise ValueError(f"end_ref_idx {end_ref_idx} is not in frames.")

    K = build_K_from_args(args)

    print(f"[INFO] frames = {frames}")
    print(f"[INFO] anchor_idx = {anchor_idx}")
    print(f"[INFO] visualization endpoints = {start_ref_idx}, {end_ref_idx}")
    print(f"[INFO] tie_ref = {args.tie_ref}")
    print(f"[INFO] edge_mode = {args.edge_mode}")
    print(f"[INFO] K =\n{K}")

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

    # ------------------------------------------------------------
    # 1. Estimate pairwise edges.
    # ------------------------------------------------------------

    edge_pairs = generate_edge_pairs(
        frames=frames,
        edge_mode=args.edge_mode,
        max_edge_dist=args.max_edge_dist,
    )

    print(f"[INFO] estimating {len(edge_pairs)} pairwise edges...")

    edges: List[PoseEdge] = []
    failed_edges = []

    for n, (target_idx, ref_idx) in enumerate(edge_pairs):
        print(f"[EDGE {n + 1:4d}/{len(edge_pairs):4d}] target={target_idx}, ref={ref_idx}")

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
                failed_edges.append(
                    {
                        "target_idx": int(target_idx),
                        "ref_idx": int(ref_idx),
                        "reason": "too_few_pose_inliers",
                        "num_matches": int(edge.num_matches),
                        "essential_inliers": int(edge.essential_inliers),
                        "pose_inliers": int(edge.pose_inliers),
                        "sampson_error_median": float(edge.sampson_error_median),
                    }
                )
                continue

            print(
                f"  [OK] matches={edge.num_matches}, "
                f"E_inliers={edge.essential_inliers}, "
                f"pose_inliers={edge.pose_inliers}, "
                f"sampson_med={edge.sampson_error_median:.4f}"
            )

            edges.append(edge)

        except Exception as e:
            print(f"  [FAIL] {e}")
            failed_edges.append(
                {
                    "target_idx": int(target_idx),
                    "ref_idx": int(ref_idx),
                    "reason": str(e),
                }
            )

    if len(edges) == 0:
        raise RuntimeError("No valid pairwise pose edges were estimated.")

    print(f"[INFO] valid edges = {len(edges)}")

    # ------------------------------------------------------------
    # 2. Build GOP-consistent global poses.
    # ------------------------------------------------------------

    directed = build_directed_transform_graph(edges)

    poses = initialize_global_poses_bfs(
        frames=frames,
        directed=directed,
        anchor_idx=anchor_idx,
    )

    missing = [f for f in frames if f not in poses]
    if len(missing) > 0:
        print(f"[WARN] disconnected frames: {missing}")

    poses = refine_global_poses(
        frames=frames,
        directed=directed,
        poses=poses,
        anchor_idx=anchor_idx,
        num_iters=args.pose_iters,
        temporal_smooth=args.temporal_smooth,
    )

    pose_graph_residuals = evaluate_pose_graph_edges(edges, poses)
    pose_graph_residual_summary = summarize_pose_graph_residuals(pose_graph_residuals)

    print("[INFO] pose graph residual summary:")
    print(json.dumps(pose_graph_residual_summary, indent=2))

    # ------------------------------------------------------------
    # 3. Visualization targets.
    # ------------------------------------------------------------

    if args.visualize_targets == "all":
        visualize_targets = list(frames)
        if args.skip_endpoints:
            visualize_targets = [
                f for f in visualize_targets
                if f not in (start_ref_idx, end_ref_idx)
            ]
    else:
        visualize_targets = parse_int_list(args.visualize_targets)

    c_candidates = parse_float_list(args.c_list)
    t_signs = parse_int_list(args.t_signs)
    candidates = build_effective_c_candidates(c_candidates, t_signs)

    print(f"[INFO] c candidates = {c_candidates}")
    print(f"[INFO] t signs = {t_signs}")
    print(f"[INFO] effective candidates = {[x[0] for x in candidates]}")
    print(f"[INFO] visualize_targets = {visualize_targets}")

    projection_results = []

    for target_idx in visualize_targets:
        if target_idx not in poses:
            print(f"[VIS SKIP] target={target_idx}: no pose")
            continue

        ref_idx = choose_endpoint_ref(
            target_idx=target_idx,
            start_ref_idx=start_ref_idx,
            end_ref_idx=end_ref_idx,
            tie_ref=args.tie_ref,
        )

        if ref_idx == target_idx:
            print(f"[VIS SKIP] target={target_idx}: chosen ref is itself")
            continue

        if ref_idx not in poses:
            print(f"[VIS SKIP] target={target_idx}: ref={ref_idx} has no pose")
            continue

        print(f"[VIS] target={target_idx}, chosen endpoint ref={ref_idx}")

        target = get_frame(target_idx)
        ref = get_frame(ref_idx)

        R_ref_target, t_ref_target = relative_from_global_poses(
            poses=poses,
            target_idx=target_idx,
            ref_idx=ref_idx,
            normalize_translation=True,
        )

        # Frame-level single c visualization.
        frame_pred, frame_valid, frame_meta = evaluate_frame_level_candidates(
            target_y=target.y_padded,
            ref_y=ref.y_padded,
            K=K,
            R_ref_target=R_ref_target,
            t_ref_target=t_ref_target,
            candidates=candidates,
            min_valid_ratio=args.min_valid_ratio,
        )

        frame_prefix = f"target{target_idx:04d}_ref{ref_idx:04d}_frameC"
        save_projection_visuals(
            out_dir=args.output_dir,
            prefix=frame_prefix,
            target_y=target.y_padded,
            ref_y=ref.y_padded,
            pred_y=frame_pred,
            valid=frame_valid,
            c_eff_map=None,
            sign_map=None,
            bitdepth=args.bitdepth,
            crop_h=target.height,
            crop_w=target.width,
        )

        # 8x8 block-wise c visualization.
        block_pred, block_valid, c_eff_map, sign_map, block_meta = blockwise_best_c_remap(
            target_y=target.y_padded,
            ref_y=ref.y_padded,
            K=K,
            R_ref_target=R_ref_target,
            t_ref_target=t_ref_target,
            candidates=candidates,
            block_size=args.block_size,
            min_block_valid_ratio=args.min_block_valid_ratio,
        )

        block_prefix = f"target{target_idx:04d}_ref{ref_idx:04d}_block{args.block_size}C"
        save_projection_visuals(
            out_dir=args.output_dir,
            prefix=block_prefix,
            target_y=target.y_padded,
            ref_y=ref.y_padded,
            pred_y=block_pred,
            valid=block_valid,
            c_eff_map=c_eff_map,
            sign_map=sign_map,
            bitdepth=args.bitdepth,
            crop_h=target.height,
            crop_w=target.width,
        )

        projection_results.append(
            {
                "target_idx": int(target_idx),
                "chosen_ref_idx": int(ref_idx),
                "selection_rule": "closer_of_start_or_end_ref",
                "start_ref_idx": int(start_ref_idx),
                "end_ref_idx": int(end_ref_idx),
                "tie_ref": args.tie_ref,

                "R_ref_target": R_ref_target.tolist(),
                "t_ref_target_unit": t_ref_target.reshape(3).tolist(),

                "frame_level": frame_meta,
                "blockwise": block_meta,

                "visual_files": {
                    "frame_level_prefix": frame_prefix,
                    "blockwise_prefix": block_prefix,
                },
            }
        )

    # ------------------------------------------------------------
    # 4. Export JSON.
    # ------------------------------------------------------------

    direct_edges_json = []
    for e in edges:
        rvec, _ = cv2.Rodrigues(e.R_ref_target)
        direct_edges_json.append(
            {
                "target_idx": int(e.target_idx),
                "ref_idx": int(e.ref_idx),
                "num_matches": int(e.num_matches),
                "essential_inliers": int(e.essential_inliers),
                "pose_inliers": int(e.pose_inliers),
                "sampson_error_mean": float(e.sampson_error_mean),
                "sampson_error_median": float(e.sampson_error_median),
                "sampson_error_p90": float(e.sampson_error_p90),
                "R_ref_target": e.R_ref_target.tolist(),
                "t_ref_target_unit": normalize_vec(e.t_ref_target).tolist(),
                "rvec_ref_target_rad": rvec.reshape(3).tolist(),
            }
        )

    global_poses_json = {}
    for f in frames:
        if f not in poses:
            continue

        R_anchor_frame, t_anchor_frame = poses[f]
        rvec, _ = cv2.Rodrigues(R_anchor_frame)

        global_poses_json[str(f)] = {
            "R_anchor_frame": R_anchor_frame.tolist(),
            "t_anchor_frame_arbitrary_scale": np.asarray(t_anchor_frame).reshape(3).tolist(),
            "rvec_anchor_frame_rad": rvec.reshape(3).tolist(),
        }

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
            R_ref_target, t_ref_target = relative_from_global_poses(
                poses=poses,
                target_idx=target_idx,
                ref_idx=ref_idx,
                normalize_translation=True,
            )
            rvec, _ = cv2.Rodrigues(R_ref_target)

            pair_transforms_json.append(
                {
                    "target_idx": int(target_idx),
                    "ref_idx": int(ref_idx),
                    "R_ref_target": R_ref_target.tolist(),
                    "t_ref_target_unit": t_ref_target.reshape(3).tolist(),
                    "rvec_ref_target_rad": rvec.reshape(3).tolist(),
                    "source": "gop_consistent_pose_graph",
                    "formula": "T_ref_target = inverse(T_anchor_ref) * T_anchor_target",
                }
            )

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),

        "frames": frames,
        "anchor_idx": int(anchor_idx),
        "start_ref_idx": int(start_ref_idx),
        "end_ref_idx": int(end_ref_idx),
        "tie_ref": args.tie_ref,

        "K": K.tolist(),

        "edge_mode": args.edge_mode,
        "max_edge_dist": int(args.max_edge_dist),
        "min_pose_inliers": int(args.min_pose_inliers),
        "pose_iters": int(args.pose_iters),
        "temporal_smooth": float(args.temporal_smooth),

        "c_candidates_input": c_candidates,
        "t_signs_input": t_signs,
        "effective_c_candidates": [
            {"c_eff": float(c_eff), "c_abs": float(c_abs), "t_sign": int(sign)}
            for c_eff, c_abs, sign in candidates
        ],

        "direct_pairwise_edges": direct_edges_json,
        "failed_edges": failed_edges,

        "pose_graph_residual_summary": pose_graph_residual_summary,
        "pose_graph_residuals": pose_graph_residuals,

        "global_poses_anchor_from_frame": global_poses_json,
        "consistent_pair_transforms": pair_transforms_json,

        "projection_visualization_results": projection_results,

        "notes": [
            "Visualization uses only the closer endpoint reference: start_ref_idx or end_ref_idx.",
            "For target exactly in the middle, tie_ref decides which endpoint is used.",
            "Backward remap is target pixel -> reference pixel, then sample reference.",
            "Frame-level visualization uses one c for the whole frame.",
            "Blockwise visualization chooses best c per 8x8 block by valid-pixel MAE.",
            "If --t-signs includes -1, negative effective c is tested as a diagnostic for translation sign instability.",
            "For physical inverse-depth-only behavior, use --t-signs 1.",
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] wrote JSON: {args.output_json}")
    print(f"[DONE] wrote visuals to: {args.output_dir}")


if __name__ == "__main__":
    main()
