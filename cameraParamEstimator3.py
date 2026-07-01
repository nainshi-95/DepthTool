#!/usr/bin/env python3
# gop_camera_like_block_trimmed.py
#
# GOP-level camera-like fitting from hierarchical block homography pseudo-GT.
#
# Main goals:
#   1) Keep the pair-wise hierarchical block homography behavior aligned with
#      hierarchical_block_homography.py defaults and logic.
#   2) Fit GOP-local camera-like variables:
#        fixed deterministic K
#        frame-wise camera-to-world pose R_i, t_i
#        target-frame/block-wise constant inverse depth rho_{i,b}
#   3) Reduce memory use:
#        sequential pair processing
#        lightweight observation accumulation
#        sparse Jacobian least_squares
#        chunked residual/stat evaluation
#        chunked full-frame rendering
#   4) Strengthen outlier rejection:
#        pair-block quality gates
#        block-level residual trimming
#        target-depth-block consistency trimming
#        neighbor fill for missing / rejected rho blocks
#
# Coordinate convention:
#   pair homography: target pixel x_t -> ref pixel x_r
#   camera model   : target pixel + target block rho -> ref pixel
#
# Pose convention:
#   R_i, t_i maps camera_i coordinates -> GOP-local world coordinates.
#   Anchor frame has R=I, t=0.
#
# Example:
#   python gop_camera_like_block_trimmed.py \
#     --input input.yuv \
#     --width 1920 --height 1080 --bitdepth 10 \
#     --gop-start 0 --gop-size 33 \
#     --pair-mode dyadic \
#     --output-dir gop0_camlike_trimmed \
#     --obs-exclude-rejected \
#     --block-trim-percentile 80 \
#     --block-trim-max-median-px 12 \
#     --render-row-batch 64

import argparse
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix, csr_matrix
except ImportError as e:
    raise ImportError("This script requires scipy. Install with: pip install scipy") from e


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


@dataclass
class PairHResult:
    target_idx: int
    ref_idx: int
    final_block_size: int
    final_records: List[List[Dict]]
    direct_cost: Dict
    direct_yuv_path: str


@dataclass
class PairRenderItem:
    target_idx: int
    ref_idx: int
    direct_cost: Dict
    direct_yuv_path: str


@dataclass
class ObservationSet:
    target_frame: np.ndarray
    ref_frame: np.ndarray
    depth_index: np.ndarray
    pair_index: np.ndarray
    pair_block_uid: np.ndarray
    px: np.ndarray
    py: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    ray_x: np.ndarray
    ray_y: np.ndarray
    weight: np.ndarray
    block_cost: np.ndarray
    block_source_id: np.ndarray


@dataclass
class PreparedObs:
    target_cid: np.ndarray
    ref_cid: np.ndarray
    depth_index: np.ndarray
    pair_index: np.ndarray
    pair_block_uid: np.ndarray
    px: np.ndarray
    py: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    rays: np.ndarray
    sqrt_weight: np.ndarray


@dataclass
class CameraLikeFitResult:
    K: np.ndarray
    frames: List[int]
    frame_to_index: Dict[int, int]
    anchor_frame: int
    anchor_cid: int
    rvecs: np.ndarray
    tvecs: np.ndarray
    log_rho: np.ndarray
    rho: np.ndarray
    active_obs_mask: np.ndarray
    active_pair_block_mask: np.ndarray
    active_depth_mask: np.ndarray
    residual_px: np.ndarray
    depth_neighbors: List[Tuple[int, int]]
    temporal_pairs: List[Tuple[int, int]]
    summary: Dict


class FrameCache:
    def __init__(self, path: str, width: int, height: int, bitdepth: int, max_cached: int):
        self.path = path
        self.width = width
        self.height = height
        self.bitdepth = bitdepth
        self.max_cached = max(1, int(max_cached))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx in self.cache:
            y = self.cache.pop(frame_idx)
            self.cache[frame_idx] = y
            return y

        y = read_y_frame(self.path, self.width, self.height, self.bitdepth, frame_idx).y
        self.cache[frame_idx] = y

        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)

        return y


class ObservationBuilder:
    def __init__(self, K: np.ndarray, width: int, height: int, args):
        self.K = K
        self.width = width
        self.height = height
        self.args = args

        self.target_frame = []
        self.ref_frame = []
        self.depth_index = []
        self.pair_index = []
        self.pair_block_uid = []
        self.px = []
        self.py = []
        self.qx = []
        self.qy = []
        self.ray_x = []
        self.ray_y = []
        self.weight = []
        self.block_cost = []
        self.block_source_id = []

        self.depth_key_to_index: Dict[Tuple[int, int, int], int] = {}
        self.depth_meta: List[Dict] = []
        self.pairs_meta: List[Tuple[int, int]] = []
        self.pair_block_meta: List[Dict] = []

    def _source_id(self, source: str) -> int:
        if source == "local_fit":
            return 2
        if source == "parent_inherit":
            return 1
        if source == "root_fallback":
            return 0
        if source == "local_fit_rejected_cost":
            return 3
        return -1

    def add_pair(self, pr: PairHResult):
        pair_index = len(self.pairs_meta)
        self.pairs_meta.append((int(pr.target_idx), int(pr.ref_idx)))

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])

        tiles = make_tiles(
            width=self.width,
            height=self.height,
            block_size=pr.final_block_size,
            edge_anchored_fit=not self.args.disable_edge_anchored_fit,
        )

        for by_idx, row in enumerate(tiles):
            for bx_idx, tile in enumerate(row):
                rec = pr.final_records[by_idx][bx_idx]
                source = str(rec.get("source", ""))
                sid = self._source_id(source)

                if self.args.obs_exclude_rejected and source == "local_fit_rejected_cost":
                    continue

                if self.args.obs_local_only and source != "local_fit":
                    continue

                chosen_cost = rec.get("chosen_cost", None)
                valid_ratio = float(rec.get("valid_ratio", 0.0))
                inlier_count = int(rec.get("inlier_count", 0))
                match_count = int(rec.get("match_count", 0))
                reproj_mae = rec.get("reproj_mae", None)

                if chosen_cost is None or not np.isfinite(float(chosen_cost)):
                    continue

                chosen_cost = float(chosen_cost)

                if valid_ratio < self.args.obs_min_valid_ratio:
                    continue

                if match_count < self.args.obs_min_matches:
                    continue

                if source == "local_fit" and inlier_count < self.args.obs_min_inliers:
                    continue

                if self.args.obs_max_cost > 0.0 and chosen_cost > self.args.obs_max_cost:
                    continue

                if reproj_mae is not None and self.args.obs_max_reproj_mae > 0.0:
                    if float(reproj_mae) > self.args.obs_max_reproj_mae:
                        continue

                H = np.asarray(rec["H"], dtype=np.float64).reshape(3, 3)
                cps = cp_points_for_tile(tile, self.args.cp_pattern, self.args.cp_inset)
                qs = apply_homography_points(H, cps)

                dkey = (int(pr.target_idx), int(by_idx), int(bx_idx))
                if dkey not in self.depth_key_to_index:
                    self.depth_key_to_index[dkey] = len(self.depth_meta)
                    self.depth_meta.append(
                        {
                            "target_idx": int(pr.target_idx),
                            "block_y_idx": int(by_idx),
                            "block_x_idx": int(bx_idx),
                            "block_x": int(tile.out_x0),
                            "block_y": int(tile.out_y0),
                            "block_w": int(tile.out_w),
                            "block_h": int(tile.out_h),
                            "block_size": int(pr.final_block_size),
                        }
                    )

                didx = self.depth_key_to_index[dkey]
                pb_uid = len(self.pair_block_meta)
                self.pair_block_meta.append(
                    {
                        "pair_block_uid": int(pb_uid),
                        "pair_index": int(pair_index),
                        "target_idx": int(pr.target_idx),
                        "ref_idx": int(pr.ref_idx),
                        "depth_index": int(didx),
                        "block_y_idx": int(by_idx),
                        "block_x_idx": int(bx_idx),
                        "source": source,
                        "source_id": int(sid),
                        "chosen_cost": float(chosen_cost),
                        "valid_ratio": float(valid_ratio),
                        "inlier_count": int(inlier_count),
                        "match_count": int(match_count),
                        "reproj_mae": float(reproj_mae) if reproj_mae is not None else None,
                    }
                )

                if source == "local_fit":
                    src_w = 1.0
                elif source == "parent_inherit":
                    src_w = 0.50
                elif source == "root_fallback":
                    src_w = 0.30
                else:
                    src_w = 0.20

                cost_w = 1.0 / math.sqrt(max(chosen_cost, 1.0))
                inlier_w = math.sqrt(max(inlier_count, 1)) / 8.0
                src_weight = src_w * cost_w * min(inlier_w, 2.0)

                for k in range(cps.shape[0]):
                    x, y = float(cps[k, 0]), float(cps[k, 1])
                    u, v = float(qs[k, 0]), float(qs[k, 1])

                    if not (0.0 <= u <= self.width - 1.0 and 0.0 <= v <= self.height - 1.0):
                        continue

                    self.target_frame.append(int(pr.target_idx))
                    self.ref_frame.append(int(pr.ref_idx))
                    self.depth_index.append(int(didx))
                    self.pair_index.append(int(pair_index))
                    self.pair_block_uid.append(int(pb_uid))
                    self.px.append(x)
                    self.py.append(y)
                    self.qx.append(u)
                    self.qy.append(v)
                    self.ray_x.append((x - cx) / fx)
                    self.ray_y.append((y - cy) / fy)
                    self.weight.append(float(src_weight))
                    self.block_cost.append(float(chosen_cost))
                    self.block_source_id.append(int(sid))

    def build(self, max_observations: int, seed: int) -> ObservationSet:
        n = len(self.px)
        if n == 0:
            raise RuntimeError("No pseudo observations were collected.")

        idx = np.arange(n, dtype=np.int64)

        if max_observations > 0 and n > max_observations:
            # Stratified-ish downsample: keep at least one observation from as many pair-blocks as possible.
            rng = np.random.default_rng(seed)
            pb = np.asarray(self.pair_block_uid, dtype=np.int64)
            unique_pb = np.unique(pb)

            chosen = []
            for uid in unique_pb:
                inds = np.flatnonzero(pb == uid)
                if inds.size > 0:
                    chosen.append(int(rng.choice(inds)))

            chosen = np.asarray(chosen, dtype=np.int64)

            if chosen.size < max_observations:
                remain_mask = np.ones(n, dtype=bool)
                remain_mask[chosen] = False
                remain = idx[remain_mask]
                add_n = min(max_observations - chosen.size, remain.size)
                if add_n > 0:
                    extra = rng.choice(remain, size=add_n, replace=False)
                    chosen = np.concatenate([chosen, extra])

            if chosen.size > max_observations:
                chosen = rng.choice(chosen, size=max_observations, replace=False)

            chosen.sort()
            idx = chosen.astype(np.int64)
            print(f"[INFO] downsample observations: {n} -> {idx.size}")

        def arr(lst, dtype):
            return np.asarray(lst, dtype=dtype)[idx]

        return ObservationSet(
            target_frame=arr(self.target_frame, np.int32),
            ref_frame=arr(self.ref_frame, np.int32),
            depth_index=arr(self.depth_index, np.int32),
            pair_index=arr(self.pair_index, np.int32),
            pair_block_uid=arr(self.pair_block_uid, np.int32),
            px=arr(self.px, np.float64),
            py=arr(self.py, np.float64),
            qx=arr(self.qx, np.float64),
            qy=arr(self.qy, np.float64),
            ray_x=arr(self.ray_x, np.float64),
            ray_y=arr(self.ray_y, np.float64),
            weight=arr(self.weight, np.float64),
            block_cost=arr(self.block_cost, np.float64),
            block_source_id=arr(self.block_source_id, np.int32),
        )


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
    offset = int(frame_idx) * frame_size

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


def write_y_plane_raw(path: str, y: np.ndarray, bitdepth: int):
    with open(path, "wb") as f:
        if bitdepth == 8:
            f.write(np.clip(np.rint(y), 0, 255).astype(np.uint8).tobytes())
        else:
            f.write(np.clip(np.rint(y), 0, 1023).astype("<u2").tobytes())


def to_8bit(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return np.clip(y, 0, 255).astype(np.uint8)
    return np.clip(y.astype(np.float32) / 4.0, 0, 255).astype(np.uint8)


def to_8bit_feature(y: np.ndarray, bitdepth: int) -> np.ndarray:
    if bitdepth == 8:
        return y.astype(np.uint8)
    return (np.clip(y, 0, 1023).astype(np.uint16) >> 2).astype(np.uint8)


# ============================================================
# Block / tile
# ============================================================

def block_grid_shape(width: int, height: int, block_size: int) -> Tuple[int, int]:
    nx = (width + block_size - 1) // block_size
    ny = (height + block_size - 1) // block_size
    return ny, nx


def normalize_homography(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def make_tiles(width: int, height: int, block_size: int, edge_anchored_fit: bool) -> List[List[Tile]]:
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

            row.append(
                Tile(
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
            )

        tiles.append(row)

    return tiles


# ============================================================
# Feature matching / homography
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


def save_match_vis(out_path: str, target_y: np.ndarray, ref_y: np.ndarray, bitdepth: int, match_result: MatchResult, max_draw: int = 300):
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


def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H = normalize_homography(H)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)

    q = ph @ H.T
    z = q[:, 2:3] + 1e-12
    out = q[:, :2] / z

    return out.astype(np.float32)


def homography_maps_for_roi(H: np.ndarray, bx: int, by: int, bw: int, bh: int, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def estimate_global_homography_H(pts_t: np.ndarray, pts_r: np.ndarray, ransac_thresh: float) -> Optional[np.ndarray]:
    if len(pts_t) < 8:
        return None

    try:
        H, _ = cv2.findHomography(
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


def fit_local_homography(pts_t: np.ndarray, pts_r: np.ndarray, ransac_thresh: float, min_matches: int, min_inliers: int) -> Tuple[Optional[np.ndarray], int, float, str]:
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

    reproj_mae = float(np.mean(err[mask])) if np.any(mask) else float(np.mean(err))
    return H, inlier_count, reproj_mae, "ok"


def eval_block_photometric_cost(target_y: np.ndarray, ref_y: np.ndarray, H: np.ndarray, bx: int, by: int, bw: int, bh: int, min_valid_ratio: float) -> Tuple[float, float]:
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

        # Keep original behavior: if parent gate removes too much, do not apply it.
        if np.count_nonzero(keep) >= max(8, len(pts_t) // 4):
            pts_t = pts_t[keep]
            pts_r = pts_r[keep]

    return pts_t, pts_r


def parent_H_for_block(parent_H_grid: Optional[np.ndarray], parent_block_size: Optional[int], out_x0: int, out_y0: int) -> Optional[np.ndarray]:
    if parent_H_grid is None or parent_block_size is None:
        return None

    py = out_y0 // parent_block_size
    px = out_x0 // parent_block_size

    py = min(parent_H_grid.shape[0] - 1, max(0, py))
    px = min(parent_H_grid.shape[1] - 1, max(0, px))

    return parent_H_grid[py, px].reshape(3, 3).copy()


def render_prediction_from_H_grid(target_y: np.ndarray, ref_y: np.ndarray, H_grid: np.ndarray, block_size: int, edge_anchored_fit: bool) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_y.shape
    pred = np.zeros((height, width), dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    tiles = make_tiles(width, height, block_size, edge_anchored_fit)

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
    args,
) -> Tuple[np.ndarray, List[List[Dict]], Dict[str, np.ndarray]]:
    height, width = target_y.shape
    ny, nx = block_grid_shape(width, height, block_size)
    edge_anchored_fit = not args.disable_edge_anchored_fit
    tiles = make_tiles(width, height, block_size, edge_anchored_fit)

    H_grid = np.zeros((ny, nx, 3, 3), dtype=np.float64)

    source_grid = np.zeros((ny, nx), dtype=np.float64)
    match_count_grid = np.zeros((ny, nx), dtype=np.float64)
    inlier_count_grid = np.zeros((ny, nx), dtype=np.float64)
    reproj_mae_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    candidate_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    fallback_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    chosen_cost_grid = np.full((ny, nx), np.inf, dtype=np.float64)
    valid_ratio_grid = np.zeros((ny, nx), dtype=np.float64)

    source_id = {
        "root_fallback": 0,
        "parent_inherit": 1,
        "local_fit": 2,
        "local_fit_rejected_cost": 3,
    }

    level_records: List[List[Dict]] = []
    accepted = 0
    inherited = 0
    rejected = 0

    for by_idx, row in enumerate(tiles):
        row_records = []

        for bx_idx, tile in enumerate(row):
            parent_H = parent_H_for_block(parent_H_grid, parent_block_size, tile.out_x0, tile.out_y0)

            if parent_H is None:
                fallback_H = root_fallback_H.copy()
                fallback_source = "root_fallback"
            else:
                fallback_H = parent_H.copy()
                fallback_source = "parent_inherit"

            fallback_cost, fallback_valid_ratio = eval_block_photometric_cost(
                target_y,
                ref_y,
                fallback_H,
                tile.out_x0,
                tile.out_y0,
                tile.out_w,
                tile.out_h,
                args.min_block_valid_ratio,
            )

            pts_t, pts_r = collect_matches_for_block(
                pts_t_all,
                pts_r_all,
                tile.fit_x0,
                tile.fit_y0,
                tile.fit_w,
                tile.fit_h,
                width,
                height,
                margin,
                fallback_H,
                args.parent_match_gate,
            )

            H_candidate, inlier_count, reproj_mae, fit_reason = fit_local_homography(
                pts_t,
                pts_r,
                args.ransac_thresh,
                args.min_matches,
                args.min_inliers,
            )

            chosen_H = fallback_H
            chosen_source = fallback_source
            candidate_cost = float("inf")
            chosen_cost = fallback_cost
            chosen_valid_ratio = fallback_valid_ratio
            reason = fit_reason

            if H_candidate is not None:
                candidate_cost, candidate_valid_ratio = eval_block_photometric_cost(
                    target_y,
                    ref_y,
                    H_candidate,
                    tile.out_x0,
                    tile.out_y0,
                    tile.out_w,
                    tile.out_h,
                    args.min_block_valid_ratio,
                )

                if not args.disable_cost_gate:
                    if np.isfinite(fallback_cost):
                        accept_by_cost = candidate_cost <= fallback_cost - args.min_gain
                    else:
                        accept_by_cost = np.isfinite(candidate_cost)
                else:
                    accept_by_cost = True

                if np.isfinite(candidate_cost) and accept_by_cost:
                    chosen_H = H_candidate
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
        "accepted": np.array([accepted], dtype=np.float64),
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


def save_bool_grid_png(path: str, valid_grid: np.ndarray, width: int, height: int, block_size: int):
    img = expand_grid_to_image(valid_grid.astype(np.float32), width, height, block_size)
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


# ============================================================
# Pair generation
# ============================================================

def parse_pairs_string(s: str) -> List[Tuple[int, int]]:
    pairs = []
    if not s.strip():
        return pairs

    for token in s.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            a, b = token.split(":")
        elif "->" in token:
            a, b = token.split("->")
        else:
            raise ValueError(f"Invalid pair token: {token}")

        pairs.append((int(a), int(b)))

    return pairs


def read_pairs_file(path: str) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ").replace(":", " ").replace("->", " ")
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def generate_dyadic_pairs(gop_start: int, gop_size: int) -> List[Tuple[int, int]]:
    start = gop_start
    end = gop_start + gop_size - 1
    pairs = []

    def rec(a: int, b: int):
        if b <= a + 1:
            return
        m = (a + b) // 2
        pairs.append((m, a))
        pairs.append((m, b))
        rec(a, m)
        rec(m, b)

    rec(start, end)

    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def generate_all_ordered_pairs(gop_start: int, gop_size: int) -> List[Tuple[int, int]]:
    frames = list(range(gop_start, gop_start + gop_size))
    pairs = []
    for t in frames:
        for r in frames:
            if t != r:
                pairs.append((t, r))
    return pairs


def generate_window_pairs(gop_start: int, gop_size: int, window: int) -> List[Tuple[int, int]]:
    frames = list(range(gop_start, gop_start + gop_size))
    pairs = []
    for t in frames:
        for r in frames:
            if t != r and abs(t - r) <= window:
                pairs.append((t, r))
    return pairs


def get_gop_pairs(args) -> List[Tuple[int, int]]:
    if args.pairs:
        pairs = parse_pairs_string(args.pairs)
    elif args.pairs_file:
        pairs = read_pairs_file(args.pairs_file)
    elif args.pair_mode == "dyadic":
        pairs = generate_dyadic_pairs(args.gop_start, args.gop_size)
    elif args.pair_mode == "all":
        pairs = generate_all_ordered_pairs(args.gop_start, args.gop_size)
    elif args.pair_mode == "window":
        pairs = generate_window_pairs(args.gop_start, args.gop_size, args.pair_window)
    else:
        raise ValueError(args.pair_mode)

    if args.max_pair_distance > 0:
        pairs = [(t, r) for t, r in pairs if abs(t - r) <= args.max_pair_distance]

    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    return pairs


# ============================================================
# Pair-wise homography pipeline, aligned with original script
# ============================================================

def run_hierarchical_pair(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    target_idx: int,
    ref_idx: int,
    bitdepth: int,
    args,
    pair_out_dir: str,
) -> PairHResult:
    ensure_dir(pair_out_dir)

    width = args.width
    height = args.height
    edge_anchored_fit = not args.disable_edge_anchored_fit

    match = detect_and_match_orb(
        target_y=target_y,
        ref_y=ref_y,
        bitdepth=bitdepth,
        max_features=args.max_features,
        ratio=args.match_ratio,
        use_clahe=not args.no_clahe,
    )

    if args.save_pair_debug:
        save_match_vis(os.path.join(pair_out_dir, "match_vis.png"), target_y, ref_y, bitdepth, match)

    H_identity = np.eye(3, dtype=np.float64)
    H_global_translation = estimate_global_translation_H(match.pts_target, match.pts_ref)
    H_global_homography = estimate_global_homography_H(match.pts_target, match.pts_ref, args.ransac_thresh)

    if args.root_fallback == "identity":
        root_H = H_identity
    elif args.root_fallback == "global_translation":
        root_H = H_global_translation
    elif args.root_fallback == "global_homography":
        root_H = H_global_homography if H_global_homography is not None else H_global_translation
    else:
        raise ValueError(args.root_fallback)

    all_level_records = []
    all_level_costs = []
    all_level_aux_summary = []

    parent_H_grid = None
    parent_block_size = None

    final_pred = None
    final_valid = None
    final_H_grid = None
    final_records = None
    final_block_size = None

    block_size = args.start_block_size

    for level in range(args.levels):
        if block_size < args.min_block_size:
            break

        print(f"[PAIR t{target_idx:03d}->r{ref_idx:03d}] level={level}, block={block_size}")

        H_grid, level_records, aux = run_one_level(
            target_y=target_y,
            ref_y=ref_y,
            pts_t_all=match.pts_target,
            pts_r_all=match.pts_ref,
            block_size=block_size,
            margin=args.block_margin,
            parent_H_grid=parent_H_grid,
            parent_block_size=parent_block_size,
            root_fallback_H=root_H,
            args=args,
        )

        pred, valid = render_prediction_from_H_grid(target_y, ref_y, H_grid, block_size, edge_anchored_fit)
        cost = calc_cost(target_y, pred, valid, bitdepth)
        level_tag = f"level{level:02d}_block{block_size}"

        if args.save_pair_debug:
            write_single_yuv420(os.path.join(pair_out_dir, f"pred_{level_tag}.yuv"), pred, width, height, bitdepth)
            save_gray_png(os.path.join(pair_out_dir, f"pred_{level_tag}.png"), pred, bitdepth)
            save_diff_png(os.path.join(pair_out_dir, f"diff_{level_tag}.png"), target_y, pred, valid)
            save_scalar_map_png(os.path.join(pair_out_dir, f"source_map_{level_tag}.png"), aux["source_grid"], width, height, block_size)
            save_scalar_map_png(os.path.join(pair_out_dir, f"match_count_{level_tag}.png"), aux["match_count_grid"], width, height, block_size)
            save_scalar_map_png(os.path.join(pair_out_dir, f"inlier_count_{level_tag}.png"), aux["inlier_count_grid"], width, height, block_size)
            save_scalar_map_png(os.path.join(pair_out_dir, f"chosen_cost_{level_tag}.png"), aux["chosen_cost_grid"], width, height, block_size)
            save_scalar_map_png(os.path.join(pair_out_dir, f"valid_ratio_{level_tag}.png"), aux["valid_ratio_grid"], width, height, block_size)

        all_level_records.append(level_records)
        all_level_costs.append(
            {
                "level": int(level),
                "block_size": int(block_size),
                "num_blocks": int(H_grid.shape[0] * H_grid.shape[1]),
                "accepted_local_fit": int(aux["accepted"][0]),
                "inherited_no_fit": int(aux["inherited"][0]),
                "rejected_by_cost": int(aux["rejected"][0]),
                "cost": cost,
            }
        )
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
            }
        )

        final_pred = pred
        final_valid = valid
        final_H_grid = H_grid
        final_records = level_records
        final_block_size = block_size

        parent_H_grid = H_grid
        parent_block_size = block_size
        block_size = block_size // 2

    if final_pred is None or final_valid is None:
        raise RuntimeError("No level was processed for pair.")

    # Original-style final aliases per pair.
    target_path = os.path.join(pair_out_dir, "target_pair.yuv")
    ref_path = os.path.join(pair_out_dir, "ref_pair.yuv")
    direct_yuv_path = os.path.join(pair_out_dir, "pred_hier_block_homography.yuv")

    if args.save_pair_debug or args.save_pair_final_yuv:
        write_single_yuv420(target_path, target_y, width, height, bitdepth)
        write_single_yuv420(ref_path, ref_y, width, height, bitdepth)
        write_single_yuv420(direct_yuv_path, final_pred, width, height, bitdepth)

    if args.save_pair_debug:
        save_gray_png(os.path.join(pair_out_dir, "target.png"), target_y, bitdepth)
        save_gray_png(os.path.join(pair_out_dir, "ref.png"), ref_y, bitdepth)
        save_gray_png(os.path.join(pair_out_dir, "pred_hier_block_homography.png"), final_pred, bitdepth)
        save_diff_png(os.path.join(pair_out_dir, "diff_hier_block_homography.png"), target_y, final_pred, final_valid)

    direct_cost = calc_cost(target_y, final_pred, final_valid, bitdepth)

    result = {
        "input": args.input,
        "width": int(width),
        "height": int(height),
        "bitdepth": int(bitdepth),
        "target_idx": int(target_idx),
        "ref_idx": int(ref_idx),
        "method": {
            "description": "hierarchical block-wise homography fitting",
            "coordinate": "target pixel -> ref pixel",
            "start_block_size": int(args.start_block_size),
            "levels": int(args.levels),
            "min_block_size": int(args.min_block_size),
            "block_margin": int(args.block_margin),
            "edge_anchored_fit": bool(edge_anchored_fit),
            "root_fallback": args.root_fallback,
            "parent_inheritance": True,
            "cost_gate": bool(not args.disable_cost_gate),
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
        },
        "root_H": {
            "identity": H_identity.tolist(),
            "global_translation": H_global_translation.tolist(),
            "global_homography": H_global_homography.tolist() if H_global_homography is not None else None,
            "selected": root_H.tolist(),
        },
        "levels_summary": all_level_costs,
        "levels_aux_grids": all_level_aux_summary if args.save_json_aux_grids else [],
        "levels_blocks": all_level_records if args.save_json_all_levels else [],
        "final_records": final_records,
        "final": {
            "block_size": int(final_block_size),
            "cost": direct_cost,
            "output_yuv": direct_yuv_path,
        },
    }

    with open(os.path.join(pair_out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return PairHResult(
        target_idx=int(target_idx),
        ref_idx=int(ref_idx),
        final_block_size=int(final_block_size),
        final_records=final_records,
        direct_cost=direct_cost,
        direct_yuv_path=direct_yuv_path,
    )


# ============================================================
# CP pseudo-observation generation
# ============================================================

def cp_points_for_tile(tile: Tile, pattern: str, inset: float) -> np.ndarray:
    x0 = tile.out_x0 + inset
    y0 = tile.out_y0 + inset
    x1 = tile.out_x1 - 1 - inset
    y1 = tile.out_y1 - 1 - inset

    if x1 < x0:
        x0 = tile.out_x0
        x1 = tile.out_x1 - 1

    if y1 < y0:
        y0 = tile.out_y0
        y1 = tile.out_y1 - 1

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    if pattern == "center":
        return np.array([[cx, cy]], dtype=np.float32)

    if pattern == "center4":
        return np.array(
            [
                [cx, cy],
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )

    if pattern == "grid3":
        xs = [x0, cx, x1]
        ys = [y0, cy, y1]
        pts = []
        for yy in ys:
            for xx in xs:
                pts.append([xx, yy])
        return np.array(pts, dtype=np.float32)

    raise ValueError(pattern)


# ============================================================
# Camera-like model
# ============================================================

def make_fixed_K(width: int, height: int, args) -> np.ndarray:
    if args.fx > 0:
        fx = float(args.fx)
    else:
        fx = float(max(width, height) * args.f_scale)

    if args.fy > 0:
        fy = float(args.fy)
    else:
        fy = float(max(width, height) * args.f_scale)

    if args.cx >= 0:
        cx = float(args.cx)
    else:
        cx = 0.5 * float(width - 1)

    if args.cy >= 0:
        cy = float(args.cy)
    else:
        cy = 0.5 * float(height - 1)

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def build_frame_index(frames: List[int]) -> Tuple[Dict[int, int], List[int]]:
    frames_sorted = sorted(set(int(f) for f in frames))
    frame_to_index = {f: i for i, f in enumerate(frames_sorted)}
    return frame_to_index, frames_sorted


def prepare_observations(obs: ObservationSet, frame_to_index: Dict[int, int]) -> PreparedObs:
    target_cid = np.asarray([frame_to_index[int(f)] for f in obs.target_frame], dtype=np.int32)
    ref_cid = np.asarray([frame_to_index[int(f)] for f in obs.ref_frame], dtype=np.int32)

    rays = np.stack(
        [
            obs.ray_x.astype(np.float64),
            obs.ray_y.astype(np.float64),
            np.ones_like(obs.ray_x, dtype=np.float64),
        ],
        axis=1,
    )

    w = obs.weight.astype(np.float64).copy()
    finite = np.isfinite(w) & (w > 0.0)
    if np.any(finite):
        med = float(np.median(w[finite]))
        if med > 1e-12:
            w = w / med
    w = np.clip(w, 1e-4, 100.0)

    return PreparedObs(
        target_cid=target_cid,
        ref_cid=ref_cid,
        depth_index=obs.depth_index.astype(np.int32),
        pair_index=obs.pair_index.astype(np.int32),
        pair_block_uid=obs.pair_block_uid.astype(np.int32),
        px=obs.px.astype(np.float64),
        py=obs.py.astype(np.float64),
        qx=obs.qx.astype(np.float64),
        qy=obs.qy.astype(np.float64),
        rays=rays,
        sqrt_weight=np.sqrt(w),
    )


def build_depth_neighbors(depth_meta: List[Dict]) -> List[Tuple[int, int]]:
    key_to_idx = {}
    for i, m in enumerate(depth_meta):
        key = (int(m["target_idx"]), int(m["block_y_idx"]), int(m["block_x_idx"]))
        key_to_idx[key] = i

    neighbors = []
    for i, m in enumerate(depth_meta):
        t = int(m["target_idx"])
        by = int(m["block_y_idx"])
        bx = int(m["block_x_idx"])
        for nb in [(t, by, bx + 1), (t, by + 1, bx)]:
            j = key_to_idx.get(nb, None)
            if j is not None:
                neighbors.append((i, j))
    return neighbors


def build_temporal_pairs(frames: List[int], frame_to_index: Dict[int, int]) -> List[Tuple[int, int]]:
    pairs = []
    for a, b in zip(frames[:-1], frames[1:]):
        pairs.append((frame_to_index[a], frame_to_index[b]))
    return pairs


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)

    if not np.any(valid):
        return float(np.median(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0

    v = values[valid]
    w = weights[valid]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * float(cw[-1])
    return float(v[np.searchsorted(cw, cutoff)])


def initialize_camera_like(prep: PreparedObs, frames: List[int], anchor_cid: int, K: np.ndarray, args, num_depths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = len(frames)
    rvecs = np.zeros((num_frames, 3), dtype=np.float64)
    tvecs = np.zeros((num_frames, 3), dtype=np.float64)

    init_rho = float(np.clip(args.init_rho, args.min_rho, args.max_rho))
    log_rho = np.full(num_depths, np.log(init_rho), dtype=np.float64)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    opt_cids = [cid for cid in range(num_frames) if cid != anchor_cid]
    cid_to_var = {cid: i for i, cid in enumerate(opt_cids)}

    pair_keys = sorted(set(zip(prep.target_cid.tolist(), prep.ref_cid.tolist())))
    rows_x, rhs_x, w_x = [], [], []
    rows_y, rhs_y, w_y = [], [], []

    for tcid, rcid in pair_keys:
        mask = (prep.target_cid == tcid) & (prep.ref_cid == rcid)
        if np.count_nonzero(mask) < 1:
            continue

        dx = weighted_median(prep.qx[mask] - prep.px[mask], prep.sqrt_weight[mask] ** 2)
        dy = weighted_median(prep.qy[mask] - prep.py[mask], prep.sqrt_weight[mask] ** 2)
        ww = float(np.sqrt(np.count_nonzero(mask)))

        bx = dx / max(fx * init_rho, 1e-12)
        by = dy / max(fy * init_rho, 1e-12)

        row = np.zeros(len(opt_cids), dtype=np.float64)
        if tcid != anchor_cid:
            row[cid_to_var[tcid]] += 1.0
        if rcid != anchor_cid:
            row[cid_to_var[rcid]] -= 1.0

        if np.any(np.abs(row) > 0):
            rows_x.append(row.copy())
            rhs_x.append(bx)
            w_x.append(ww)
            rows_y.append(row.copy())
            rhs_y.append(by)
            w_y.append(ww)

    if rows_x:
        A = np.vstack(rows_x)
        b = np.asarray(rhs_x, dtype=np.float64)
        w = np.asarray(w_x, dtype=np.float64)
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        for cid, vi in cid_to_var.items():
            tvecs[cid, 0] = sol[vi]

    if rows_y:
        A = np.vstack(rows_y)
        b = np.asarray(rhs_y, dtype=np.float64)
        w = np.asarray(w_y, dtype=np.float64)
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        for cid, vi in cid_to_var.items():
            tvecs[cid, 1] = sol[vi]

    return rvecs, tvecs, log_rho


def poses_to_vec(rvecs: np.ndarray, tvecs: np.ndarray, opt_frame_cids: List[int]) -> np.ndarray:
    vals = []
    for cid in opt_frame_cids:
        vals.extend(rvecs[cid].tolist())
        vals.extend(tvecs[cid].tolist())
    return np.asarray(vals, dtype=np.float64)


def vec_to_poses(x: np.ndarray, base_rvecs: np.ndarray, base_tvecs: np.ndarray, opt_frame_cids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    rvecs = base_rvecs.copy()
    tvecs = base_tvecs.copy()
    k = 0
    for cid in opt_frame_cids:
        rvecs[cid] = x[k:k + 3]
        k += 3
        tvecs[cid] = x[k:k + 3]
        k += 3
    return rvecs, tvecs


def all_rotation_matrices(rvecs: np.ndarray) -> np.ndarray:
    Rs = np.zeros((rvecs.shape[0], 3, 3), dtype=np.float64)
    for i in range(rvecs.shape[0]):
        Rs[i] = rodrigues(rvecs[i])
    return Rs


def project_prepared_observations(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    log_rho: np.ndarray,
    min_rho: float,
    max_rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = obs_indices
    Rs = all_rotation_matrices(rvecs)

    tcid = prep.target_cid[idx]
    rcid = prep.ref_cid[idx]
    didx = prep.depth_index[idx]

    log_rho_safe = np.clip(log_rho, np.log(min_rho), np.log(max_rho))
    rho = np.exp(log_rho_safe[didx])

    X_t = prep.rays[idx] / rho[:, None]

    X_w = np.einsum("nij,nj->ni", Rs[tcid], X_t) + tvecs[tcid]
    X_r = np.einsum("nji,nj->ni", Rs[rcid], X_w - tvecs[rcid])

    z = X_r[:, 2]
    z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * (X_r[:, 0] / z_safe) + cx
    v = fy * (X_r[:, 1] / z_safe) + cy
    return u, v, z


def camera_residual_vector(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    log_rho: np.ndarray,
    args,
    depth_neighbors: Optional[List[Tuple[int, int]]] = None,
    temporal_pairs: Optional[List[Tuple[int, int]]] = None,
    opt_frame_cids: Optional[List[int]] = None,
    include_depth_prior: bool = True,
    include_pose_prior: bool = True,
) -> np.ndarray:
    # Chunk projection inside residual to avoid large temporary arrays.
    idx_all = np.asarray(obs_indices, dtype=np.int32)
    chunks = []
    bs = max(1024, int(args.residual_eval_batch))

    for s in range(0, idx_all.shape[0], bs):
        idx = idx_all[s:s + bs]
        u, v, z = project_prepared_observations(
            prep,
            idx,
            K,
            rvecs,
            tvecs,
            log_rho,
            args.min_rho,
            args.max_rho,
        )
        sw = prep.sqrt_weight[idx]
        rx = (u - prep.qx[idx]) * sw
        ry = (v - prep.qy[idx]) * sw

        bad_z = z <= args.z_min
        if np.any(bad_z):
            penalty = args.z_penalty * (args.z_min - z[bad_z])
            rx[bad_z] += penalty
            ry[bad_z] += penalty

        if args.residual_clip_px > 0.0:
            clip = float(args.residual_clip_px)
            rx = np.clip(rx, -clip, clip)
            ry = np.clip(ry, -clip, clip)

        chunks.append(rx)
        chunks.append(ry)

    residuals = chunks

    if include_depth_prior and args.depth_prior_weight > 0.0:
        log_init = np.log(float(np.clip(args.init_rho, args.min_rho, args.max_rho)))
        w = np.sqrt(float(args.depth_prior_weight))
        residuals.append(w * (log_rho - log_init))

    if include_depth_prior and depth_neighbors and args.depth_smooth_weight > 0.0:
        a = np.asarray([p[0] for p in depth_neighbors], dtype=np.int32)
        b = np.asarray([p[1] for p in depth_neighbors], dtype=np.int32)
        w = np.sqrt(float(args.depth_smooth_weight))
        residuals.append(w * (log_rho[a] - log_rho[b]))

    if include_pose_prior and opt_frame_cids is not None and args.rot_prior_weight > 0.0:
        w = np.sqrt(float(args.rot_prior_weight))
        residuals.append(w * rvecs[opt_frame_cids].reshape(-1))

    if include_pose_prior and opt_frame_cids is not None and args.trans_prior_weight > 0.0:
        w = np.sqrt(float(args.trans_prior_weight))
        residuals.append(w * tvecs[opt_frame_cids].reshape(-1))

    if include_pose_prior and temporal_pairs and args.pose_smooth_weight > 0.0:
        a = np.asarray([p[0] for p in temporal_pairs], dtype=np.int32)
        b = np.asarray([p[1] for p in temporal_pairs], dtype=np.int32)
        w = np.sqrt(float(args.pose_smooth_weight))
        residuals.append(w * (rvecs[a] - rvecs[b]).reshape(-1))
        residuals.append(w * (tvecs[a] - tvecs[b]).reshape(-1))

    return np.concatenate([r.reshape(-1) for r in residuals]).astype(np.float64)


def compute_pixel_residuals_batched(prep: PreparedObs, K: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, log_rho: np.ndarray, args) -> np.ndarray:
    n = prep.qx.shape[0]
    err = np.full(n, np.inf, dtype=np.float64)
    bs = max(1024, int(args.residual_eval_batch))

    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n), dtype=np.int32)
        u, v, z = project_prepared_observations(prep, idx, K, rvecs, tvecs, log_rho, args.min_rho, args.max_rho)
        e = np.sqrt((u - prep.qx[idx]) ** 2 + (v - prep.qy[idx]) ** 2)
        e[~np.isfinite(e)] = np.inf
        e[z <= args.z_min] = np.inf
        err[idx] = e

    return err


def aggregate_by_uid(values: np.ndarray, uid: np.ndarray, num_uid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = np.full(num_uid, np.inf, dtype=np.float64)
    mean = np.full(num_uid, np.inf, dtype=np.float64)
    mx = np.full(num_uid, np.inf, dtype=np.float64)
    count = np.zeros(num_uid, dtype=np.int32)

    order = np.argsort(uid)
    uid_s = uid[order]
    val_s = values[order]

    start = 0
    while start < uid_s.shape[0]:
        u = int(uid_s[start])
        end = start + 1
        while end < uid_s.shape[0] and uid_s[end] == u:
            end += 1
        vals = val_s[start:end]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0 and 0 <= u < num_uid:
            med[u] = float(np.median(vals))
            mean[u] = float(np.mean(vals))
            mx[u] = float(np.max(vals))
            count[u] = int(vals.size)
        start = end

    return med, mean, mx, count


def make_block_trim_masks(
    prep: PreparedObs,
    err: np.ndarray,
    num_pair_blocks: int,
    num_depths: int,
    prev_pair_block_mask: np.ndarray,
    prev_depth_mask: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    finite_obs = np.isfinite(err)

    # Pair-block residual stats.
    pb_med, pb_mean, pb_max, pb_count = aggregate_by_uid(err, prep.pair_block_uid, num_pair_blocks)
    db_med, db_mean, db_max, db_count = aggregate_by_uid(err, prep.depth_index, num_depths)

    pair_block_mask = prev_pair_block_mask.copy()
    depth_mask = prev_depth_mask.copy()

    valid_pb = np.isfinite(pb_med) & (pb_count >= args.block_trim_min_obs)
    valid_db = np.isfinite(db_med) & (db_count >= args.depth_block_trim_min_obs)

    # Pair-block threshold: absolute + percentile.
    pb_thresholds = []
    if args.block_trim_percentile > 0.0 and np.any(valid_pb & pair_block_mask):
        pb_thresholds.append(float(np.percentile(pb_med[valid_pb & pair_block_mask], args.block_trim_percentile)))
    if args.block_trim_max_median_px > 0.0:
        pb_thresholds.append(float(args.block_trim_max_median_px))
    pb_th = min(pb_thresholds) if pb_thresholds else np.inf

    # Depth-block consistency threshold.
    db_thresholds = []
    if args.depth_block_trim_percentile > 0.0 and np.any(valid_db & depth_mask):
        db_thresholds.append(float(np.percentile(db_med[valid_db & depth_mask], args.depth_block_trim_percentile)))
    if args.depth_block_trim_max_median_px > 0.0:
        db_thresholds.append(float(args.depth_block_trim_max_median_px))
    db_th = min(db_thresholds) if db_thresholds else np.inf

    # Hard reject pair-blocks.
    reject_pb = np.zeros(num_pair_blocks, dtype=bool)
    reject_pb |= ~valid_pb
    reject_pb |= pb_med > pb_th
    if args.block_trim_max_max_px > 0.0:
        reject_pb |= pb_max > float(args.block_trim_max_max_px)

    # Hard reject target-depth blocks if globally inconsistent across refs.
    reject_db = np.zeros(num_depths, dtype=bool)
    reject_db |= ~valid_db
    reject_db |= db_med > db_th
    if args.depth_block_trim_max_max_px > 0.0:
        reject_db |= db_max > float(args.depth_block_trim_max_max_px)

    pair_block_mask &= ~reject_pb
    depth_mask &= ~reject_db

    # Keep enough active blocks.
    min_pb_keep = max(8, int(math.ceil(num_pair_blocks * float(args.min_active_block_ratio))))
    if np.count_nonzero(pair_block_mask) < min_pb_keep:
        candidates = np.flatnonzero(np.isfinite(pb_med))
        order = candidates[np.argsort(pb_med[candidates])]
        pair_block_mask[:] = False
        pair_block_mask[order[:min(min_pb_keep, order.size)]] = True

    min_db_keep = max(4, int(math.ceil(num_depths * float(args.min_active_depth_ratio))))
    if np.count_nonzero(depth_mask) < min_db_keep:
        candidates = np.flatnonzero(np.isfinite(db_med))
        order = candidates[np.argsort(db_med[candidates])]
        depth_mask[:] = False
        depth_mask[order[:min(min_db_keep, order.size)]] = True

    obs_mask = finite_obs & pair_block_mask[prep.pair_block_uid] & depth_mask[prep.depth_index]

    summary = {
        "pair_block_threshold_median_px": float(pb_th) if np.isfinite(pb_th) else None,
        "depth_block_threshold_median_px": float(db_th) if np.isfinite(db_th) else None,
        "active_obs": int(np.count_nonzero(obs_mask)),
        "active_obs_ratio": float(np.mean(obs_mask)),
        "active_pair_blocks": int(np.count_nonzero(pair_block_mask)),
        "num_pair_blocks": int(num_pair_blocks),
        "active_depth_blocks": int(np.count_nonzero(depth_mask)),
        "num_depth_blocks": int(num_depths),
        "pair_block_residual_median_px": residual_stats(pb_med[np.isfinite(pb_med) & prev_pair_block_mask]),
        "depth_block_residual_median_px": residual_stats(db_med[np.isfinite(db_med) & prev_depth_mask]),
    }

    return obs_mask, pair_block_mask, depth_mask, summary


def residual_stats(vals: np.ndarray) -> Dict:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def build_depth_jac_sparsity(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    num_depths: int,
    depth_neighbors: List[Tuple[int, int]],
    include_prior: bool,
    args,
) -> csr_matrix:
    m_obs = int(obs_indices.shape[0]) * 2
    m = m_obs
    if include_prior and args.depth_prior_weight > 0.0:
        m += num_depths
    if include_prior and depth_neighbors and args.depth_smooth_weight > 0.0:
        m += len(depth_neighbors)

    S = lil_matrix((m, num_depths), dtype=np.int8)
    didx = prep.depth_index[obs_indices]
    for i, d in enumerate(didx):
        S[2 * i, int(d)] = 1
        S[2 * i + 1, int(d)] = 1

    row = m_obs
    if include_prior and args.depth_prior_weight > 0.0:
        for d in range(num_depths):
            S[row + d, d] = 1
        row += num_depths

    if include_prior and depth_neighbors and args.depth_smooth_weight > 0.0:
        for k, (a, b) in enumerate(depth_neighbors):
            S[row + k, int(a)] = 1
            S[row + k, int(b)] = 1

    return S.tocsr()


def build_pose_jac_sparsity(
    prep: PreparedObs,
    obs_indices: np.ndarray,
    num_frames: int,
    opt_frame_cids: List[int],
    temporal_pairs: List[Tuple[int, int]],
    include_prior: bool,
    args,
) -> csr_matrix:
    pose_dim = 6 * len(opt_frame_cids)
    cid_to_base = {cid: 6 * i for i, cid in enumerate(opt_frame_cids)}

    m_obs = int(obs_indices.shape[0]) * 2
    m = m_obs
    if include_prior and args.rot_prior_weight > 0.0:
        m += 3 * len(opt_frame_cids)
    if include_prior and args.trans_prior_weight > 0.0:
        m += 3 * len(opt_frame_cids)
    if include_prior and temporal_pairs and args.pose_smooth_weight > 0.0:
        m += 6 * len(temporal_pairs)

    S = lil_matrix((m, pose_dim), dtype=np.int8)
    tcid = prep.target_cid[obs_indices]
    rcid = prep.ref_cid[obs_indices]

    for i, (t, r) in enumerate(zip(tcid, rcid)):
        rows = [2 * i, 2 * i + 1]
        for cid in [int(t), int(r)]:
            if cid in cid_to_base:
                base = cid_to_base[cid]
                for rr in rows:
                    for c in range(base, base + 6):
                        S[rr, c] = 1

    row = m_obs
    if include_prior and args.rot_prior_weight > 0.0:
        for i in range(len(opt_frame_cids)):
            base = 6 * i
            for k in range(3):
                S[row, base + k] = 1
                row += 1

    if include_prior and args.trans_prior_weight > 0.0:
        for i in range(len(opt_frame_cids)):
            base = 6 * i
            for k in range(3):
                S[row, base + 3 + k] = 1
                row += 1

    if include_prior and temporal_pairs and args.pose_smooth_weight > 0.0:
        for a, b in temporal_pairs:
            for part in range(2):  # r then t
                for k in range(3):
                    ca = cid_to_base.get(int(a), None)
                    cb = cid_to_base.get(int(b), None)
                    off = part * 3 + k
                    if ca is not None:
                        S[row, ca + off] = 1
                    if cb is not None:
                        S[row, cb + off] = 1
                    row += 1

    return S.tocsr()


def fit_camera_like_model(
    obs: ObservationSet,
    depth_meta: List[Dict],
    pair_block_meta: List[Dict],
    pairs_meta: List[Tuple[int, int]],
    K: np.ndarray,
    args,
) -> CameraLikeFitResult:
    if obs.px.shape[0] < 8:
        raise RuntimeError(f"Too few pseudo observations: {obs.px.shape[0]}")

    frames = sorted(set(obs.target_frame.astype(int).tolist()) | set(obs.ref_frame.astype(int).tolist()))
    frame_to_index, frames = build_frame_index(frames)

    if args.anchor_frame >= 0:
        if args.anchor_frame not in frame_to_index:
            raise ValueError(f"anchor_frame={args.anchor_frame} is not in GOP pair frames.")
        anchor_frame = int(args.anchor_frame)
    else:
        anchor_frame = int(frames[0])

    anchor_cid = frame_to_index[anchor_frame]
    opt_frame_cids = [cid for cid in range(len(frames)) if cid != anchor_cid]

    prep = prepare_observations(obs, frame_to_index)
    num_depths = len(depth_meta)
    num_pair_blocks = len(pair_block_meta)

    rvecs, tvecs, log_rho = initialize_camera_like(prep, frames, anchor_cid, K, args, num_depths)
    depth_neighbors = build_depth_neighbors(depth_meta)
    temporal_pairs = build_temporal_pairs(frames, frame_to_index)

    active_obs_mask = np.ones(prep.qx.shape[0], dtype=bool)
    active_pair_block_mask = np.ones(num_pair_blocks, dtype=bool)
    active_depth_mask = np.ones(num_depths, dtype=bool)

    lower_log_rho = np.full(num_depths, np.log(args.min_rho), dtype=np.float64)
    upper_log_rho = np.full(num_depths, np.log(args.max_rho), dtype=np.float64)

    report = {
        "num_observations": int(prep.qx.shape[0]),
        "num_pair_blocks": int(num_pair_blocks),
        "num_depth_blocks": int(num_depths),
        "num_frames": int(len(frames)),
        "anchor_frame": int(anchor_frame),
        "iterations": [],
    }

    for it in range(args.alt_iters):
        active_idx = np.flatnonzero(active_obs_mask).astype(np.int32)
        if active_idx.shape[0] < 8:
            raise RuntimeError("Too few active observations after block trimming.")

        if args.max_active_observations_per_iter > 0 and active_idx.shape[0] > args.max_active_observations_per_iter:
            rng = np.random.default_rng(args.random_seed + it)
            active_idx = np.sort(rng.choice(active_idx, size=args.max_active_observations_per_iter, replace=False)).astype(np.int32)
            print(f"[INFO] iteration {it}: active observation subsample -> {active_idx.shape[0]}")

        # -------------------------------
        # 1) Depth update with fixed pose.
        # -------------------------------
        depth_sparsity = None
        if args.use_sparse_jacobian:
            depth_sparsity = build_depth_jac_sparsity(prep, active_idx, num_depths, depth_neighbors, True, args)

        def depth_fun(x_log_rho):
            return camera_residual_vector(
                prep=prep,
                obs_indices=active_idx,
                K=K,
                rvecs=rvecs,
                tvecs=tvecs,
                log_rho=x_log_rho,
                args=args,
                depth_neighbors=depth_neighbors,
                temporal_pairs=temporal_pairs,
                opt_frame_cids=opt_frame_cids,
                include_depth_prior=True,
                include_pose_prior=False,
            )

        depth_res = least_squares(
            depth_fun,
            log_rho,
            bounds=(lower_log_rho, upper_log_rho),
            loss=args.robust_loss,
            f_scale=args.robust_f_scale,
            max_nfev=args.depth_max_nfev,
            jac_sparsity=depth_sparsity,
            tr_solver="lsmr" if depth_sparsity is not None else None,
            verbose=2 if args.verbose_opt else 0,
        )
        log_rho = depth_res.x.astype(np.float64)

        # -------------------------------
        # 2) Pose update with fixed depth.
        # -------------------------------
        x_pose0 = poses_to_vec(rvecs, tvecs, opt_frame_cids)
        pose_sparsity = None
        if args.use_sparse_jacobian:
            pose_sparsity = build_pose_jac_sparsity(prep, active_idx, len(frames), opt_frame_cids, temporal_pairs, True, args)

        def pose_fun(x_pose):
            rr, tt = vec_to_poses(x_pose, rvecs, tvecs, opt_frame_cids)
            rr[anchor_cid] = 0.0
            tt[anchor_cid] = 0.0
            return camera_residual_vector(
                prep=prep,
                obs_indices=active_idx,
                K=K,
                rvecs=rr,
                tvecs=tt,
                log_rho=log_rho,
                args=args,
                depth_neighbors=depth_neighbors,
                temporal_pairs=temporal_pairs,
                opt_frame_cids=opt_frame_cids,
                include_depth_prior=False,
                include_pose_prior=True,
            )

        pose_res = least_squares(
            pose_fun,
            x_pose0,
            loss=args.robust_loss,
            f_scale=args.robust_f_scale,
            max_nfev=args.pose_max_nfev,
            jac_sparsity=pose_sparsity,
            tr_solver="lsmr" if pose_sparsity is not None else None,
            verbose=2 if args.verbose_opt else 0,
        )
        rvecs, tvecs = vec_to_poses(pose_res.x, rvecs, tvecs, opt_frame_cids)
        rvecs[anchor_cid] = 0.0
        tvecs[anchor_cid] = 0.0

        # -------------------------------
        # 3) Strong block-level trimming.
        # -------------------------------
        err = compute_pixel_residuals_batched(prep, K, rvecs, tvecs, log_rho, args)
        active_obs_mask, active_pair_block_mask, active_depth_mask, trim_summary = make_block_trim_masks(
            prep=prep,
            err=err,
            num_pair_blocks=num_pair_blocks,
            num_depths=num_depths,
            prev_pair_block_mask=active_pair_block_mask,
            prev_depth_mask=active_depth_mask,
            args=args,
        )

        iter_info = {
            "iter": int(it),
            "depth_cost": float(depth_res.cost),
            "depth_nfev": int(depth_res.nfev),
            "pose_cost": float(pose_res.cost),
            "pose_nfev": int(pose_res.nfev),
            "trim_summary": trim_summary,
        }
        report["iterations"].append(iter_info)
        print("[CAMLIKE ITER]")
        print(json.dumps(iter_info, indent=2))

    final_err = compute_pixel_residuals_batched(prep, K, rvecs, tvecs, log_rho, args)
    active_obs_mask, active_pair_block_mask, active_depth_mask, final_trim = make_block_trim_masks(
        prep=prep,
        err=final_err,
        num_pair_blocks=num_pair_blocks,
        num_depths=num_depths,
        prev_pair_block_mask=active_pair_block_mask,
        prev_depth_mask=active_depth_mask,
        args=args,
    )

    rho = np.exp(np.clip(log_rho, np.log(args.min_rho), np.log(args.max_rho)))

    report["final_trim_summary"] = final_trim
    report["final_observation_residual_px"] = residual_stats(final_err[active_obs_mask])
    report["rho_summary_all"] = residual_stats(rho)
    report["rho_summary_active_depth"] = residual_stats(rho[active_depth_mask])

    return CameraLikeFitResult(
        K=K,
        frames=frames,
        frame_to_index=frame_to_index,
        anchor_frame=anchor_frame,
        anchor_cid=anchor_cid,
        rvecs=rvecs,
        tvecs=tvecs,
        log_rho=log_rho,
        rho=rho,
        active_obs_mask=active_obs_mask,
        active_pair_block_mask=active_pair_block_mask,
        active_depth_mask=active_depth_mask,
        residual_px=final_err,
        depth_neighbors=depth_neighbors,
        temporal_pairs=temporal_pairs,
        summary=report,
    )


# ============================================================
# Rho fill and rendering
# ============================================================

def build_target_grid_info(target_idx: int, depth_meta: List[Dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    ms = [m for m in depth_meta if int(m["target_idx"]) == int(target_idx)]
    if not ms:
        return None, None, None
    block_size = int(ms[0]["block_size"])
    ny = max(int(m["block_y_idx"]) for m in ms) + 1
    nx = max(int(m["block_x_idx"]) for m in ms) + 1
    return block_size, ny, nx


def fill_grid_by_neighbors(grid: np.ndarray, valid: np.ndarray, global_fill: float, max_passes: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    g = grid.copy().astype(np.float64)
    v = valid.copy().astype(bool)

    if not np.any(v):
        g[:, :] = global_fill
        v[:, :] = True
        return g, v

    for _ in range(max_passes):
        if np.all(v):
            break
        new_g = g.copy()
        new_v = v.copy()
        changed = False
        ny, nx = g.shape
        for y in range(ny):
            for x in range(nx):
                if v[y, x]:
                    continue
                vals = []
                for yy, xx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                    if 0 <= yy < ny and 0 <= xx < nx and v[yy, xx]:
                        vals.append(g[yy, xx])
                if vals:
                    new_g[y, x] = float(np.mean(vals))
                    new_v[y, x] = True
                    changed = True
        g, v = new_g, new_v
        if not changed:
            break

    g[~v] = global_fill
    v[:, :] = True
    return g, v


def make_rho_grid_for_target(target_idx: int, depth_meta: List[Dict], fit: CameraLikeFitResult, args) -> Tuple[np.ndarray, np.ndarray, int]:
    block_size, ny, nx = build_target_grid_info(target_idx, depth_meta)
    if block_size is None:
        raise ValueError(f"No depth grid for target_idx={target_idx}")

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = np.zeros((ny, nx), dtype=bool)

    active_rho_vals = fit.rho[fit.active_depth_mask]
    if active_rho_vals.size > 0:
        global_fill = float(np.median(active_rho_vals))
    else:
        global_fill = float(np.median(fit.rho))

    for i, m in enumerate(depth_meta):
        if int(m["target_idx"]) != int(target_idx):
            continue
        by = int(m["block_y_idx"])
        bx = int(m["block_x_idx"])
        if fit.active_depth_mask[i]:
            grid[by, bx] = float(fit.rho[i])
            valid[by, bx] = True

    if args.rho_fill_mode == "median":
        grid[~valid] = global_fill
        valid[:, :] = True
    elif args.rho_fill_mode == "neighbor":
        grid, valid = fill_grid_by_neighbors(grid, valid, global_fill, max_passes=args.rho_fill_max_passes)
    else:
        raise ValueError(args.rho_fill_mode)

    grid = np.clip(grid, args.min_rho, args.max_rho)
    return grid, valid, int(block_size)


def build_rho_image_for_target(target_idx: int, width: int, height: int, depth_meta: List[Dict], fit: CameraLikeFitResult, args) -> np.ndarray:
    grid, _, block_size = make_rho_grid_for_target(target_idx, depth_meta, fit, args)
    img = np.full((height, width), float(np.median(grid)), dtype=np.float32)

    for by_idx, by in enumerate(range(0, height, block_size)):
        y1 = min(by + block_size, height)
        for bx_idx, bx in enumerate(range(0, width, block_size)):
            x1 = min(bx + block_size, width)
            if by_idx < grid.shape[0] and bx_idx < grid.shape[1]:
                img[by:y1, bx:x1] = float(grid[by_idx, bx_idx])

    return img


def save_rho_maps(output_dir: str, width: int, height: int, depth_meta: List[Dict], fit: CameraLikeFitResult, args):
    rho_dir = os.path.join(output_dir, "rho_maps")
    ensure_dir(rho_dir)
    target_frames = sorted(set(int(m["target_idx"]) for m in depth_meta))

    for t in target_frames:
        rho_img = build_rho_image_for_target(t, width, height, depth_meta, fit, args)
        vals = rho_img[np.isfinite(rho_img)]
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))

        if abs(hi - lo) < 1e-12:
            out = np.full((height, width), 128, dtype=np.uint8)
        else:
            out = np.clip((rho_img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(rho_dir, f"rho_t{t:03d}.png"), color)

        np.save(os.path.join(rho_dir, f"rho_t{t:03d}.npy"), rho_img.astype(np.float32))

        if args.save_rho_yuv:
            # Visualization-like 10/8-bit normalized rho, not physical depth coding.
            maxv = 255.0 if args.bitdepth == 8 else 1023.0
            norm = out.astype(np.float32) / 255.0 * maxv
            write_single_yuv420(os.path.join(rho_dir, f"rho_vis_t{t:03d}.yuv"), norm, width, height, args.bitdepth)


def camera_like_maps_for_rows(
    target_idx: int,
    ref_idx: int,
    y0: int,
    y1: int,
    width: int,
    height: int,
    fit: CameraLikeFitResult,
    rho_img: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_idx not in fit.frame_to_index:
        raise ValueError(f"target_idx={target_idx} not present in fitted frame set.")
    if ref_idx not in fit.frame_to_index:
        raise ValueError(f"ref_idx={ref_idx} not present in fitted frame set.")

    tcid = fit.frame_to_index[int(target_idx)]
    rcid = fit.frame_to_index[int(ref_idx)]

    K = fit.K
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(y0, y1, dtype=np.float64),
    )

    ray_x = (xs - cx) / fx
    ray_y = (ys - cy) / fy
    rho_flat = rho_img[y0:y1, :].reshape(-1).astype(np.float64)

    rays = np.stack([ray_x.reshape(-1), ray_y.reshape(-1), np.ones(rho_flat.shape[0], dtype=np.float64)], axis=1)
    X_t = rays / np.clip(rho_flat[:, None], args.min_rho, args.max_rho)

    Rs = all_rotation_matrices(fit.rvecs)
    R_t = Rs[tcid]
    R_r = Rs[rcid]
    t_t = fit.tvecs[tcid]
    t_r = fit.tvecs[rcid]

    X_w = X_t @ R_t.T + t_t[None, :]
    X_r = (X_w - t_r[None, :]) @ R_r

    z = X_r[:, 2]
    z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)

    map_x = fx * (X_r[:, 0] / z_safe) + cx
    map_y = fy * (X_r[:, 1] / z_safe) + cy

    valid = (
        (z > args.z_min)
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )

    return (
        map_x.reshape(y1 - y0, width).astype(np.float32),
        map_y.reshape(y1 - y0, width).astype(np.float32),
        valid.reshape(y1 - y0, width),
    )


def render_camera_like_pair_chunked(
    target_y: np.ndarray,
    ref_y: np.ndarray,
    target_idx: int,
    ref_idx: int,
    fit: CameraLikeFitResult,
    depth_meta: List[Dict],
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_y.shape
    pred = np.zeros((height, width), dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    rho_img = build_rho_image_for_target(target_idx, width, height, depth_meta, fit, args)
    row_batch = max(1, int(args.render_row_batch))

    for y0 in range(0, height, row_batch):
        y1 = min(height, y0 + row_batch)
        map_x, map_y, valid = camera_like_maps_for_rows(
            target_idx=target_idx,
            ref_idx=ref_idx,
            y0=y0,
            y1=y1,
            width=width,
            height=height,
            fit=fit,
            rho_img=rho_img,
            args=args,
        )
        pred_rows = remap_ref(ref_y, map_x, map_y)
        pred[y0:y1, :] = pred_rows
        valid_all[y0:y1, :] = valid

    return pred, valid_all


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--bitdepth", type=int, choices=[8, 10], required=True)

    parser.add_argument("--gop-start", type=int, required=True)
    parser.add_argument("--gop-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--pair-mode", choices=["dyadic", "all", "window"], default="dyadic")
    parser.add_argument("--pair-window", type=int, default=4)
    parser.add_argument("--pairs", type=str, default="")
    parser.add_argument("--pairs-file", type=str, default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-pair-distance", type=int, default=0)
    parser.add_argument("--skip-failed-pairs", action="store_true")

    # Match original hierarchical_block_homography.py defaults.
    parser.add_argument("--start-block-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--min-block-size", type=int, default=32)
    parser.add_argument("--block-margin", type=int, default=32)
    parser.add_argument("--disable-edge-anchored-fit", action="store_true")

    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--match-ratio", type=float, default=0.70)
    parser.add_argument("--no-clahe", action="store_true")

    parser.add_argument("--ransac-thresh", type=float, default=2.0)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--parent-match-gate", type=float, default=30.0)
    parser.add_argument("--root-fallback", choices=["identity", "global_translation", "global_homography"], default="global_translation")
    parser.add_argument("--min-block-valid-ratio", type=float, default=0.50)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--disable-cost-gate", action="store_true")

    # Pair saving / memory options.
    parser.add_argument("--save-pair-debug", action="store_true", help="Save original-style per-level PNG/YUV/debug maps. Slower and larger on disk.")
    parser.add_argument("--save-pair-final-yuv", action="store_true", help="Save target/ref/final pair YUV aliases per pair.")
    parser.add_argument("--save-json-all-levels", action="store_true", help="Save all levels_blocks to pair JSON. Default saves only final_records to reduce JSON size.")
    parser.add_argument("--save-json-aux-grids", action="store_true")
    parser.add_argument("--max-cached-frames", type=int, default=4)

    # Pseudo observation options.
    parser.add_argument("--cp-pattern", choices=["center", "center4", "grid3"], default="center4")
    parser.add_argument("--cp-inset", type=float, default=2.0)
    parser.add_argument("--obs-exclude-rejected", action="store_true")
    parser.add_argument("--obs-local-only", action="store_true")
    parser.add_argument("--obs-min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--obs-min-inliers", type=int, default=8)
    parser.add_argument("--obs-min-matches", type=int, default=0)
    parser.add_argument("--obs-max-cost", type=float, default=0.0)
    parser.add_argument("--obs-max-reproj-mae", type=float, default=0.0)
    parser.add_argument("--max-observations", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=1)

    # Fixed K.
    parser.add_argument("--fx", type=float, default=-1.0)
    parser.add_argument("--fy", type=float, default=-1.0)
    parser.add_argument("--cx", type=float, default=-1.0)
    parser.add_argument("--cy", type=float, default=-1.0)
    parser.add_argument("--f-scale", type=float, default=1.0)

    # Camera-like fitting.
    parser.add_argument("--anchor-frame", type=int, default=-1)
    parser.add_argument("--init-rho", type=float, default=1.0)
    parser.add_argument("--min-rho", type=float, default=0.05)
    parser.add_argument("--max-rho", type=float, default=20.0)

    parser.add_argument("--alt-iters", type=int, default=8)
    parser.add_argument("--depth-max-nfev", type=int, default=30)
    parser.add_argument("--pose-max-nfev", type=int, default=30)
    parser.add_argument("--max-active-observations-per-iter", type=int, default=120000)

    parser.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--robust-f-scale", type=float, default=2.0)
    parser.add_argument("--use-sparse-jacobian", action="store_true", default=True)
    parser.add_argument("--disable-sparse-jacobian", dest="use_sparse_jacobian", action="store_false")

    # Strong block-level trimming.
    parser.add_argument("--block-trim-percentile", type=float, default=80.0)
    parser.add_argument("--block-trim-max-median-px", type=float, default=15.0)
    parser.add_argument("--block-trim-max-max-px", type=float, default=60.0)
    parser.add_argument("--block-trim-min-obs", type=int, default=1)
    parser.add_argument("--depth-block-trim-percentile", type=float, default=85.0)
    parser.add_argument("--depth-block-trim-max-median-px", type=float, default=20.0)
    parser.add_argument("--depth-block-trim-max-max-px", type=float, default=80.0)
    parser.add_argument("--depth-block-trim-min-obs", type=int, default=1)
    parser.add_argument("--min-active-block-ratio", type=float, default=0.20)
    parser.add_argument("--min-active-depth-ratio", type=float, default=0.20)

    # Regularization.
    parser.add_argument("--depth-prior-weight", type=float, default=1e-3)
    parser.add_argument("--depth-smooth-weight", type=float, default=1e-2)
    parser.add_argument("--rot-prior-weight", type=float, default=1e-5)
    parser.add_argument("--trans-prior-weight", type=float, default=1e-5)
    parser.add_argument("--pose-smooth-weight", type=float, default=1e-4)

    parser.add_argument("--z-min", type=float, default=1e-4)
    parser.add_argument("--z-penalty", type=float, default=100.0)
    parser.add_argument("--residual-clip-px", type=float, default=200.0)
    parser.add_argument("--residual-eval-batch", type=int, default=20000)

    # Rendering / rho fill.
    parser.add_argument("--render-row-batch", type=int, default=64)
    parser.add_argument("--rho-fill-mode", choices=["neighbor", "median"], default="neighbor")
    parser.add_argument("--rho-fill-max-passes", type=int, default=256)
    parser.add_argument("--save-rho-yuv", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help="Fit and save JSON/rho maps only; skip camlike full-frame YUV rendering.")

    parser.add_argument("--verbose-opt", action="store_true")
    parser.add_argument("--cv-seed", type=int, default=-1)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.cv_seed >= 0:
        cv2.setRNGSeed(int(args.cv_seed))

    pairs = get_gop_pairs(args)
    if not pairs:
        raise RuntimeError("No GOP pairs were generated.")

    print("[INFO] GOP pairs:")
    for p in pairs:
        print(f"  target={p[0]} ref={p[1]}")

    K = make_fixed_K(args.width, args.height, args)
    print("[INFO] fixed K:")
    print(K)

    frame_cache = FrameCache(args.input, args.width, args.height, args.bitdepth, args.max_cached_frames)
    obs_builder = ObservationBuilder(K=K, width=args.width, height=args.height, args=args)

    pair_render_items: List[PairRenderItem] = []
    failed_pairs = []

    # ------------------------------------------------------------
    # 1) Sequential pair-wise hierarchical block homography.
    #    This avoids holding all pair H grids in memory at once.
    # ------------------------------------------------------------
    for pair_i, (target_idx, ref_idx) in enumerate(pairs):
        pair_dir = os.path.join(args.output_dir, f"pair_t{target_idx:03d}_r{ref_idx:03d}")
        print(f"[PAIR {pair_i + 1}/{len(pairs)}] target={target_idx}, ref={ref_idx}")

        try:
            target_y = frame_cache.get(target_idx)
            ref_y = frame_cache.get(ref_idx)
            pr = run_hierarchical_pair(
                target_y=target_y,
                ref_y=ref_y,
                target_idx=target_idx,
                ref_idx=ref_idx,
                bitdepth=args.bitdepth,
                args=args,
                pair_out_dir=pair_dir,
            )

            obs_builder.add_pair(pr)
            pair_render_items.append(
                PairRenderItem(
                    target_idx=int(pr.target_idx),
                    ref_idx=int(pr.ref_idx),
                    direct_cost=pr.direct_cost,
                    direct_yuv_path=pr.direct_yuv_path,
                )
            )

            print("[PAIR DIRECT COST]")
            print(json.dumps(pr.direct_cost, indent=2))

            # Explicitly drop large records after observations were collected.
            del pr

        except Exception as e:
            failed_pairs.append({"target_idx": int(target_idx), "ref_idx": int(ref_idx), "error": str(e)})
            print(f"[WARN] pair failed: target={target_idx}, ref={ref_idx}, error={e}")
            if not args.skip_failed_pairs:
                raise

    if not pair_render_items:
        raise RuntimeError("All pairs failed. No fitting possible.")

    obs = obs_builder.build(max_observations=args.max_observations, seed=args.random_seed)
    depth_meta = obs_builder.depth_meta
    pair_block_meta = obs_builder.pair_block_meta
    pairs_meta = obs_builder.pairs_meta

    print(f"[INFO] pseudo observations = {obs.px.shape[0]}")
    print(f"[INFO] pair blocks         = {len(pair_block_meta)}")
    print(f"[INFO] depth variables     = {len(depth_meta)}")

    # ------------------------------------------------------------
    # 2) GOP-shared camera-like fitting with block-level trimming.
    # ------------------------------------------------------------
    fit = fit_camera_like_model(
        obs=obs,
        depth_meta=depth_meta,
        pair_block_meta=pair_block_meta,
        pairs_meta=pairs_meta,
        K=K,
        args=args,
    )

    # ------------------------------------------------------------
    # 3) Save rho maps with neighbor fill for rejected/missing blocks.
    # ------------------------------------------------------------
    save_rho_maps(args.output_dir, args.width, args.height, depth_meta, fit, args)

    # ------------------------------------------------------------
    # 4) Render fitted camera-like predictions pair by pair, row-chunked.
    # ------------------------------------------------------------
    camlike_dir = os.path.join(args.output_dir, "camlike_pairs")
    ensure_dir(camlike_dir)
    camlike_costs = []

    if not args.skip_render:
        for item_i, item in enumerate(pair_render_items):
            print(f"[RENDER {item_i + 1}/{len(pair_render_items)}] target={item.target_idx}, ref={item.ref_idx}")
            target_y = frame_cache.get(item.target_idx)
            ref_y = frame_cache.get(item.ref_idx)

            try:
                pred, valid = render_camera_like_pair_chunked(
                    target_y=target_y,
                    ref_y=ref_y,
                    target_idx=item.target_idx,
                    ref_idx=item.ref_idx,
                    fit=fit,
                    depth_meta=depth_meta,
                    args=args,
                )
                cost = calc_cost(target_y, pred, valid, args.bitdepth)
                tag = f"t{item.target_idx:03d}_r{item.ref_idx:03d}"
                yuv_path = os.path.join(camlike_dir, f"pred_camlike_{tag}.yuv")
                png_path = os.path.join(camlike_dir, f"pred_camlike_{tag}.png")
                diff_path = os.path.join(camlike_dir, f"diff_camlike_{tag}.png")

                write_single_yuv420(yuv_path, pred, args.width, args.height, args.bitdepth)
                save_gray_png(png_path, pred, args.bitdepth)
                save_diff_png(diff_path, target_y, pred, valid)

                camlike_costs.append(
                    {
                        "target_idx": int(item.target_idx),
                        "ref_idx": int(item.ref_idx),
                        "direct_block_homography_cost": item.direct_cost,
                        "camlike_cost": cost,
                        "pred_yuv": yuv_path,
                    }
                )
            except Exception as e:
                camlike_costs.append(
                    {
                        "target_idx": int(item.target_idx),
                        "ref_idx": int(item.ref_idx),
                        "direct_block_homography_cost": item.direct_cost,
                        "camlike_error": str(e),
                    }
                )
                print(f"[WARN] render failed: target={item.target_idx}, ref={item.ref_idx}, error={e}")
    else:
        print("[INFO] skip render enabled.")

    # ------------------------------------------------------------
    # 5) Save block masks and final JSON.
    # ------------------------------------------------------------
    masks_dir = os.path.join(args.output_dir, "trim_masks")
    ensure_dir(masks_dir)

    # Save active depth grid per target.
    target_frames_with_depth = sorted(set(int(m["target_idx"]) for m in depth_meta))
    for t in target_frames_with_depth:
        block_size, ny, nx = build_target_grid_info(t, depth_meta)
        if block_size is None:
            continue
        grid = np.zeros((ny, nx), dtype=np.float32)
        for i, m in enumerate(depth_meta):
            if int(m["target_idx"]) != t:
                continue
            grid[int(m["block_y_idx"]), int(m["block_x_idx"])] = 1.0 if fit.active_depth_mask[i] else 0.0
        save_bool_grid_png(os.path.join(masks_dir, f"active_depth_t{t:03d}.png"), grid, args.width, args.height, block_size)

    pose_json = []
    for f in fit.frames:
        cid = fit.frame_to_index[f]
        pose_json.append(
            {
                "frame_idx": int(f),
                "compact_id": int(cid),
                "is_anchor": bool(cid == fit.anchor_cid),
                "rvec": fit.rvecs[cid].tolist(),
                "R": rodrigues(fit.rvecs[cid]).tolist(),
                "t": fit.tvecs[cid].tolist(),
            }
        )

    depth_json = []
    for i, m in enumerate(depth_meta):
        mm = dict(m)
        mm["depth_index"] = int(i)
        mm["rho"] = float(fit.rho[i])
        mm["log_rho"] = float(fit.log_rho[i])
        mm["active_depth_block"] = bool(fit.active_depth_mask[i])
        depth_json.append(mm)

    pair_block_json = []
    for i, m in enumerate(pair_block_meta):
        mm = dict(m)
        mm["active_pair_block"] = bool(fit.active_pair_block_mask[i]) if i < fit.active_pair_block_mask.shape[0] else False
        pair_block_json.append(mm)

    result = {
        "input": args.input,
        "width": int(args.width),
        "height": int(args.height),
        "bitdepth": int(args.bitdepth),
        "gop": {
            "gop_start": int(args.gop_start),
            "gop_size": int(args.gop_size),
            "pair_mode": args.pair_mode,
            "requested_pairs": [(int(a), int(b)) for a, b in pairs],
            "used_pairs": [(int(p.target_idx), int(p.ref_idx)) for p in pair_render_items],
            "failed_pairs": failed_pairs,
        },
        "method": {
            "description": "GOP-shared camera-like fitting from hierarchical block homography pseudo-GT with strong block-level trimming.",
            "coordinate": "target pixel -> ref pixel",
            "pose_convention": "R_i,t_i maps camera_i coordinates to GOP-local world coordinates. Anchor frame has R=I,t=0.",
            "consistency": "Same frame pose shared by all pairs; same target-frame/block rho shared across all references.",
            "memory_reduction": [
                "Sequential pair processing",
                "Observation cap/downsample",
                "Sparse Jacobian least_squares with LSMR",
                "Batched residual evaluation",
                "Row-chunked full-frame rendering",
            ],
            "outlier_rejection": [
                "Pair homography RANSAC/cost gate",
                "Observation quality gate",
                "Pair-block residual median/max trim",
                "Target-depth-block consistency residual trim",
                "Neighbor fill for missing or rejected rho blocks",
            ],
        },
        "K": fit.K.tolist(),
        "homography_options": {
            "start_block_size": int(args.start_block_size),
            "levels": int(args.levels),
            "min_block_size": int(args.min_block_size),
            "block_margin": int(args.block_margin),
            "edge_anchored_fit": bool(not args.disable_edge_anchored_fit),
            "root_fallback": args.root_fallback,
            "max_features": int(args.max_features),
            "match_ratio": float(args.match_ratio),
            "clahe": bool(not args.no_clahe),
            "ransac_thresh": float(args.ransac_thresh),
            "min_matches": int(args.min_matches),
            "min_inliers": int(args.min_inliers),
            "parent_match_gate": float(args.parent_match_gate),
            "min_block_valid_ratio": float(args.min_block_valid_ratio),
            "min_gain": float(args.min_gain),
            "cost_gate": bool(not args.disable_cost_gate),
        },
        "observation_options": {
            "cp_pattern": args.cp_pattern,
            "cp_inset": float(args.cp_inset),
            "obs_exclude_rejected": bool(args.obs_exclude_rejected),
            "obs_local_only": bool(args.obs_local_only),
            "max_observations": int(args.max_observations),
            "num_observations_used": int(obs.px.shape[0]),
            "num_pair_blocks": int(len(pair_block_meta)),
            "num_depth_variables": int(len(depth_meta)),
        },
        "fit_options": {
            "anchor_frame": int(fit.anchor_frame),
            "init_rho": float(args.init_rho),
            "min_rho": float(args.min_rho),
            "max_rho": float(args.max_rho),
            "alt_iters": int(args.alt_iters),
            "depth_max_nfev": int(args.depth_max_nfev),
            "pose_max_nfev": int(args.pose_max_nfev),
            "use_sparse_jacobian": bool(args.use_sparse_jacobian),
            "max_active_observations_per_iter": int(args.max_active_observations_per_iter),
            "block_trim_percentile": float(args.block_trim_percentile),
            "block_trim_max_median_px": float(args.block_trim_max_median_px),
            "depth_block_trim_percentile": float(args.depth_block_trim_percentile),
            "depth_block_trim_max_median_px": float(args.depth_block_trim_max_median_px),
            "rho_fill_mode": args.rho_fill_mode,
        },
        "fit_summary": fit.summary,
        "poses": pose_json,
        "depth_blocks": depth_json,
        "pair_blocks": pair_block_json,
        "pair_costs": camlike_costs,
        "outputs": {
            "result_json": os.path.join(args.output_dir, "gop_camera_like_result.json"),
            "rho_map_dir": os.path.join(args.output_dir, "rho_maps"),
            "trim_masks_dir": masks_dir,
            "camlike_pair_dir": camlike_dir,
        },
    }

    result_path = os.path.join(args.output_dir, "gop_camera_like_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[DONE]")
    print(f"  result JSON: {result_path}")
    print(f"  rho maps   : {os.path.join(args.output_dir, 'rho_maps')}")
    print(f"  trim masks : {masks_dir}")
    if not args.skip_render:
        print(f"  camlike dir: {camlike_dir}")


if __name__ == "__main__":
    main()
