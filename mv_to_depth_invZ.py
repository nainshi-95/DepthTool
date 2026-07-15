#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MV + camera parameter -> causal c-only sparse depth YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Model:
  rho(x,y) = 1 / z(x,y) = c

Behavior:
  1) Each MV observation is converted to a closed-form depth estimate.
  2) The current fit-block never uses its own MV observations.
  3) Multiple already-decoded causal neighboring blocks are searched:
       - same row: left blocks only
       - previous rows: left / center / right blocks
     within --neighbor-radius blocks.
  4) Their inverse-depth observations are fused into one robust constant c.
     The base weight combines:
       - MV reprojection confidence
       - spatial distance from the target block
       - source-block balancing
  5) Huber IRLS suppresses inconsistent MV/depth outliers.
  6) The entire target block is filled with the constant depth z = 1 / c.
  7) Blocks with too few observations, too few source blocks, or unstable
     inverse-depth samples remain zero.

Camera JSONL format:
  camparam_v2_vggt_or_canonical
  pose_mode: current_to_previous, gop_local, or absolute

Depth output:
  N x H x W YUV420p10le.
  Invalid depth remains zero.
  UV planes are fixed to 512.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Camera
# ============================================================

def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))

    if theta < 1e-12:
        x, y, z = r
        K = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + K

    axis = r / theta
    x, y, z = axis
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * K
        + (1.0 - math.cos(theta)) * (K @ K)
    )


def rt_to_4x4(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rodrigues_to_matrix(rvec)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def intrinsic_vec_to_matrix(v: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in v]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid intrinsic: {v}")

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_camera_jsonl(path: str) -> Dict[str, Any]:
    header = None
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
            elif "poc" in obj:
                records.append(obj)

    if header is None:
        raise RuntimeError(f"Camera header not found: {path}")
    if not records:
        raise RuntimeError(f"No camera records: {path}")

    return {"header": header, "records": records}


def build_camera_lookup(camera_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    header = camera_json["header"]
    records = sorted(camera_json["records"], key=lambda x: int(x["poc"]))

    pose_mode = str(header["pose_mode"])
    intr0 = header["intrinsic"]
    base_intr = np.array(
        [
            float(intr0["fx"]),
            float(intr0["fy"]),
            float(intr0["cx"]),
            float(intr0["cy"]),
        ],
        dtype=np.float64,
    )

    z_sign = 1.0 if float(intr0.get("z_sign", 1.0)) >= 0.0 else -1.0
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )

    cur_intr = base_intr.copy()
    prev_w2c = np.eye(4, dtype=np.float64)
    lookup: Dict[int, Dict[str, Any]] = {}

    for order, rec in enumerate(records):
        poc = int(rec["poc"])
        delta = np.asarray(
            rec.get("intrinsic_delta", [0, 0, 0, 0]),
            dtype=np.float64,
        )

        if fixed_intrinsic:
            cur_intr = base_intr.copy()
        else:
            cur_intr = cur_intr + delta

        K = intrinsic_vec_to_matrix(cur_intr)
        T_rec = rt_to_4x4(rec["rvec"], rec["tvec"])

        if pose_mode == "current_to_previous":
            if order == 0:
                W2C = np.eye(4, dtype=np.float64)
            else:
                # X_prev = T_prev_from_cur * X_cur
                # W2C_cur = inv(T_prev_from_cur) * W2C_prev
                W2C = np.linalg.inv(T_rec) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            W2C = T_rec
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")

        lookup[poc] = {
            "poc": poc,
            "K": K,
            "W2C": W2C,
            "C2W": np.linalg.inv(W2C),
            "z_sign": z_sign,
        }
        prev_w2c = W2C

    return lookup


# ============================================================
# MV observation -> depth
# ============================================================

@dataclass
class MVObservation:
    poc: int
    x: int
    y: int
    w: int
    h: int
    list_id: str
    ref_poc: int
    mv_x: float
    mv_y: float


@dataclass
class DepthObservation:
    poc: int
    x: float
    y: float
    depth: float
    reproj_error: float
    ref_poc: int
    list_id: str


def parse_mv_csv(path: str, num_frames: int) -> List[List[MVObservation]]:
    by_frame: List[List[MVObservation]] = [[] for _ in range(num_frames)]

    required = {
        "poc", "x", "y", "w", "h",
        "list", "ref_poc", "mv_x", "mv_y",
    }

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV header not found")

        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Missing CSV columns: {sorted(missing)}")

        for row_no, row in enumerate(reader, 2):
            try:
                poc = int(row["poc"])
                if not (0 <= poc < num_frames):
                    continue

                by_frame[poc].append(
                    MVObservation(
                        poc=poc,
                        x=int(row["x"]),
                        y=int(row["y"]),
                        w=int(row["w"]),
                        h=int(row["h"]),
                        list_id=str(row["list"]),
                        ref_poc=int(row["ref_poc"]),
                        mv_x=float(row["mv_x"]),
                        mv_y=float(row["mv_y"]),
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"Bad CSV row {row_no}: {row}") from exc

    return by_frame


def pixel_ray(u: float, v: float, cam: Dict[str, Any]) -> np.ndarray:
    K = cam["K"]
    z_sign = float(cam["z_sign"])

    return np.array(
        [
            (u - K[0, 2]) / K[0, 0],
            (v - K[1, 2]) / K[1, 1],
            z_sign,
        ],
        dtype=np.float64,
    )


def project_point(X: np.ndarray, cam: Dict[str, Any]) -> Optional[np.ndarray]:
    K = cam["K"]
    z_sign = float(cam["z_sign"])
    depth = z_sign * float(X[2])

    if not np.isfinite(depth) or depth <= 1e-10:
        return None

    return np.array(
        [
            K[0, 0] * float(X[0]) / depth + K[0, 2],
            K[1, 1] * float(X[1]) / depth + K[1, 2],
        ],
        dtype=np.float64,
    )


def relative_transform(
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    # X_ref = R * X_cur + t
    M = np.asarray(cam_ref["W2C"]) @ np.asarray(cam_cur["C2W"])
    return M[:3, :3], M[:3, 3]


def solve_depth_closed_form(
    u: float,
    v: float,
    mv_x: float,
    mv_y: float,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
    min_parallax: float,
    max_reproj_error: float,
) -> Optional[Tuple[float, float]]:
    """
    Current pixel p=(u,v), reference match p'=(u+mv_x,v+mv_y).

    X_cur(z) = z * ray_cur
    X_ref(z) = R * X_cur(z) + t = z*q + t

    The two reference projection equations are linear in z.
    """
    ur = u + mv_x
    vr = v + mv_y

    ray = pixel_ray(u, v, cam_cur)
    R, t = relative_transform(cam_cur, cam_ref)
    q = R @ ray

    K = cam_ref["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z_sign_ref = float(cam_ref["z_sign"])

    du = ur - cx
    dv = vr - cy

    Au = du * z_sign_ref * q[2] - fx * q[0]
    Bu = fx * t[0] - du * z_sign_ref * t[2]

    Av = dv * z_sign_ref * q[2] - fy * q[1]
    Bv = fy * t[1] - dv * z_sign_ref * t[2]

    A = np.array([Au, Av], dtype=np.float64)
    B = np.array([Bu, Bv], dtype=np.float64)

    denom = float(np.dot(A, A))
    if not np.isfinite(denom) or denom < min_parallax * min_parallax:
        return None

    depth = float(np.dot(A, B) / denom)
    if not np.isfinite(depth) or depth <= 0.0:
        return None

    X_ref = depth * q + t
    pred = project_point(X_ref, cam_ref)
    if pred is None:
        return None

    reproj_error = float(
        np.linalg.norm(pred - np.array([ur, vr], dtype=np.float64))
    )
    if not np.isfinite(reproj_error) or reproj_error > max_reproj_error:
        return None

    return depth, reproj_error


# ============================================================
# Robust c-only inverse-depth fitting
# ============================================================

def huber_weights(residual_normalized: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual_normalized)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    cutoff = 0.5 * float(cumulative[-1])
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(v[min(idx, len(v) - 1)])


def effective_sample_size(weights: np.ndarray) -> float:
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights * weights))
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def fit_inv_depth_constant_cpu(
    depths: np.ndarray,
    reproj_errors: np.ndarray,
    spatial_distances: np.ndarray,
    source_sizes: np.ndarray,
    irls_iters: int,
    huber_delta: float,
    distance_decay: float,
    source_balance_power: float,
    min_effective_points: float,
    max_relative_invd_mad: float,
) -> Optional[float]:
    """
    Robustly estimate a single inverse-depth constant:

        c = argmin sum_i w_i * rho(1/z_i - c)

    Base weight:
        reprojection confidence
        * spatial-distance decay
        * source-block balancing
    """
    if depths.size == 0:
        return None

    invz = 1.0 / depths
    finite = (
        np.isfinite(invz)
        & (invz > 0.0)
        & np.isfinite(reproj_errors)
        & np.isfinite(spatial_distances)
        & np.isfinite(source_sizes)
        & (source_sizes > 0.0)
    )
    if not np.any(finite):
        return None

    invz = invz[finite]
    errs = reproj_errors[finite]
    dists = spatial_distances[finite]
    src_sizes = source_sizes[finite]

    reproj_w = 1.0 / np.maximum(1.0 + errs * errs, 1e-12)
    spatial_w = np.exp(-max(distance_decay, 0.0) * dists)
    balance_w = 1.0 / np.power(
        np.maximum(src_sizes, 1.0),
        max(source_balance_power, 0.0),
    )

    base_w = reproj_w * spatial_w * balance_w
    valid_w = np.isfinite(base_w) & (base_w > 0.0)
    if not np.any(valid_w):
        return None

    invz = invz[valid_w]
    base_w = base_w[valid_w]

    if effective_sample_size(base_w) < min_effective_points:
        return None

    # Weighted median is a stable initialization under foreground/background
    # mixtures and isolated erroneous MVs.
    c = weighted_median(invz, base_w)

    final_w = base_w.copy()
    for _ in range(max(1, irls_iters)):
        residual = invz - c
        residual_med = weighted_median(residual, base_w)
        abs_dev = np.abs(residual - residual_med)
        mad = weighted_median(abs_dev, base_w)
        scale = max(1.4826 * mad, 1e-10)

        robust_w = huber_weights(residual / scale, huber_delta)
        final_w = base_w * robust_w

        denom = float(np.sum(final_w))
        if not np.isfinite(denom) or denom <= 1e-15:
            return None

        c_new = float(np.sum(final_w * invz) / denom)
        if not np.isfinite(c_new) or c_new <= 0.0:
            return None

        if abs(c_new - c) <= 1e-10 * max(abs(c), 1.0):
            c = c_new
            break
        c = c_new

    if effective_sample_size(final_w) < min_effective_points:
        return None

    # Optional multimodal/unstable-sample rejection.
    # 0 disables this check.
    if max_relative_invd_mad > 0.0:
        final_abs_dev = np.abs(invz - c)
        final_mad = weighted_median(final_abs_dev, final_w)
        relative_mad = final_mad / max(abs(c), 1e-12)
        if relative_mad > max_relative_invd_mad:
            return None

    return c


# ============================================================
# Main reconstruction
# ============================================================

def make_depth_observations(
    mv_rows: List[MVObservation],
    cameras: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    min_parallax: float,
    max_reproj_error: float,
) -> List[DepthObservation]:
    out: List[DepthObservation] = []

    for row in mv_rows:
        if row.poc not in cameras or row.ref_poc not in cameras:
            continue

        cx = row.x + (row.w - 1) * 0.5
        cy = row.y + (row.h - 1) * 0.5

        if not (0.0 <= cx < width and 0.0 <= cy < height):
            continue

        solved = solve_depth_closed_form(
            u=cx,
            v=cy,
            mv_x=row.mv_x,
            mv_y=row.mv_y,
            cam_cur=cameras[row.poc],
            cam_ref=cameras[row.ref_poc],
            min_parallax=min_parallax,
            max_reproj_error=max_reproj_error,
        )
        if solved is None:
            continue

        depth, err = solved
        if not (min_depth <= depth <= max_depth):
            continue

        out.append(
            DepthObservation(
                poc=row.poc,
                x=cx,
                y=cy,
                depth=depth,
                reproj_error=err,
                ref_poc=row.ref_poc,
                list_id=row.list_id,
            )
        )

    return out


def causal_neighbor_offsets(radius: int) -> List[Tuple[int, int]]:
    """
    Raster-causal block offsets.

    Same row:
      only blocks to the left.

    Previous rows:
      blocks from left through right are already decoded and may be used.
    """
    offsets: List[Tuple[int, int]] = []

    for dy in range(-radius, 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx >= 0:
                continue
            if dx == 0 and dy == 0:
                continue
            if max(abs(dx), abs(dy)) > radius:
                continue
            offsets.append((dx, dy))

    # Prefer nearby blocks first. This also makes --max-points deterministic.
    offsets.sort(key=lambda p: (math.hypot(p[0], p[1]), abs(p[1]), abs(p[0])))
    return offsets


def build_block_fit_jobs(
    observations: List[DepthObservation],
    width: int,
    height: int,
    fit_block: int,
    neighbor_radius: int,
    min_points: int,
    min_source_blocks: int,
    max_points: int,
) -> List[Dict[str, Any]]:
    """
    Build c-only predictor jobs from multiple causal neighboring fit-blocks.

    The current block is always excluded.
    """
    if not observations:
        return []

    block_obs: Dict[Tuple[int, int], List[DepthObservation]] = {}
    for o in observations:
        gx = int(o.x) // fit_block
        gy = int(o.y) // fit_block
        block_obs.setdefault((gx, gy), []).append(o)

    offsets = causal_neighbor_offsets(neighbor_radius)
    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block
    jobs: List[Dict[str, Any]] = []

    for gy in range(grid_h):
        by = gy * fit_block
        bh = min(fit_block, height - by)

        for gx in range(grid_w):
            bx = gx * fit_block
            bw = min(fit_block, width - bx)

            selected_obs: List[DepthObservation] = []
            selected_dist: List[float] = []
            selected_source_size: List[float] = []
            used_source_blocks = 0

            for dx, dy in offsets:
                sx = gx + dx
                sy = gy + dy
                if not (0 <= sx < grid_w and 0 <= sy < grid_h):
                    continue

                vals = block_obs.get((sx, sy), [])
                if not vals:
                    continue

                used_source_blocks += 1
                block_distance = math.hypot(dx, dy)
                source_size = float(len(vals))

                for o in vals:
                    selected_obs.append(o)
                    selected_dist.append(block_distance)
                    selected_source_size.append(source_size)

            if len(selected_obs) < min_points:
                continue
            if used_source_blocks < min_source_blocks:
                continue

            # Keep the most reliable/nearby points when the search region is
            # dense. Current-block observations are still never included.
            if max_points > 0 and len(selected_obs) > max_points:
                ranking = np.asarray(
                    [
                        (
                            selected_dist[i],
                            selected_obs[i].reproj_error,
                            abs(selected_obs[i].x - (bx + (bw - 1) * 0.5))
                            + abs(selected_obs[i].y - (by + (bh - 1) * 0.5)),
                        )
                        for i in range(len(selected_obs))
                    ],
                    dtype=[
                        ("dist", np.float64),
                        ("err", np.float64),
                        ("pixel_dist", np.float64),
                    ],
                )
                keep = np.argsort(
                    ranking,
                    order=("dist", "err", "pixel_dist"),
                )[:max_points]

                selected_obs = [selected_obs[int(i)] for i in keep]
                selected_dist = [selected_dist[int(i)] for i in keep]
                selected_source_size = [
                    selected_source_size[int(i)] for i in keep
                ]

            jobs.append(
                {
                    "bx": bx,
                    "by": by,
                    "bw": bw,
                    "bh": bh,
                    "depths": np.asarray(
                        [o.depth for o in selected_obs],
                        dtype=np.float64,
                    ),
                    "errors": np.asarray(
                        [o.reproj_error for o in selected_obs],
                        dtype=np.float64,
                    ),
                    "spatial_distances": np.asarray(
                        selected_dist,
                        dtype=np.float64,
                    ),
                    "source_sizes": np.asarray(
                        selected_source_size,
                        dtype=np.float64,
                    ),
                    "num_source_blocks": used_source_blocks,
                    "num_l0": sum(
                        str(o.list_id).upper() in ("0", "L0", "LIST_0")
                        for o in selected_obs
                    ),
                    "num_l1": sum(
                        str(o.list_id).upper() in ("1", "L1", "LIST_1")
                        for o in selected_obs
                    ),
                }
            )

    return jobs


def render_constant_jobs(
    jobs: List[Dict[str, Any]],
    constants: List[Optional[float]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)

    for job, c in zip(jobs, constants):
        if c is None or not np.isfinite(c) or c <= 0.0:
            continue

        z = 1.0 / float(c)
        if not np.isfinite(z) or not (min_depth <= z <= max_depth):
            continue

        bx, by = int(job["bx"]), int(job["by"])
        bw, bh = int(job["bw"]), int(job["bh"])

        depth[by:by + bh, bx:bx + bw] = z
        valid[by:by + bh, bx:bx + bw] = True

    return depth, valid


def write_depth_yuv420p10le(
    output_path: str,
    depth_frames: List[np.ndarray],
    depth_scale_real: float,
    max_code: int = 1023,
) -> None:
    if depth_scale_real <= 0.0:
        raise ValueError("depth_scale_real must be positive")
    if not depth_frames:
        raise ValueError("No depth frames")

    h, w = depth_frames[0].shape
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")

    with open(output_path, "wb") as f:
        for depth in depth_frames:
            y = np.zeros((h, w), dtype=np.float64)
            valid = np.isfinite(depth) & (depth > 0.0)
            y[valid] = np.rint(depth[valid] / depth_scale_real)
            y = np.clip(y, 0, max_code).astype("<u2")

            f.write(np.ascontiguousarray(y).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())


def print_progress(
    frame_idx: int,
    num_frames: int,
    valid_obs: int,
    valid_ratio: float,
) -> None:
    done = frame_idx + 1
    ratio = done / max(num_frames, 1)
    width = 32
    n = int(round(ratio * width))
    bar = "#" * n + "-" * (width - n)

    print(
        f"\r[{bar}] {done:3d}/{num_frames:3d} "
        f"obs={valid_obs:7d} valid={valid_ratio:7.3%}",
        end="",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Predict each fit-block with a robust constant inverse depth c, "
            "using multiple causal neighboring-block MV observations."
        )
    )
    ap.add_argument("--mv-csv", required=True)
    ap.add_argument("--camera-param", required=True)
    ap.add_argument("--out-yuv", required=True)

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--num-frames", type=int, default=33)

    ap.add_argument(
        "--fit-block",
        type=int,
        default=16,
        help="Output constant-depth block size. Default: 16.",
    )
    ap.add_argument(
        "--neighbor-radius",
        type=int,
        default=2,
        help=(
            "Causal neighboring fit-block search radius. "
            "2 uses nearby and slightly separated blocks. Default: 2."
        ),
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum MV-derived depth observations per target block.",
    )
    ap.add_argument(
        "--min-source-blocks",
        type=int,
        default=2,
        help="Minimum distinct causal source blocks. Default: 2.",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=64,
        help=(
            "Maximum observations fused per target block; 0 means unlimited. "
            "Nearest and lowest-error observations are retained first."
        ),
    )

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument(
        "--min-parallax",
        type=float,
        default=1e-6,
        help="Reject nearly depth-unobservable constraints.",
    )
    ap.add_argument("--max-reproj-error", type=float, default=1.5)

    ap.add_argument("--irls-iters", type=int, default=4)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument(
        "--distance-decay",
        type=float,
        default=0.7,
        help=(
            "Weight multiplier exp(-value * block_distance). "
            "Larger values trust nearer blocks more."
        ),
    )
    ap.add_argument(
        "--source-balance-power",
        type=float,
        default=0.5,
        help=(
            "Each point is divided by source_block_point_count^power. "
            "0 disables balancing; 1 gives equal total weight per source block."
        ),
    )
    ap.add_argument(
        "--min-effective-points",
        type=float,
        default=2.5,
        help="Minimum Kish effective sample size after weighting.",
    )
    ap.add_argument(
        "--max-relative-invd-mad",
        type=float,
        default=0.0,
        help=(
            "Reject unstable samples when weighted MAD(inv-depth)/c exceeds "
            "this value. 0 disables the rejection. Try 0.25~0.5."
        ),
    )

    ap.add_argument(
        "--depth-scale-real",
        type=float,
        default=None,
        help=(
            "Override output depth scale. Default: camera header "
            "depth_scale/depth_scale_precision."
        ),
    )

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.neighbor_radius < 1:
        raise ValueError("--neighbor-radius must be >= 1")
    if args.min_points < 1:
        raise ValueError("--min-points must be >= 1")
    if args.min_source_blocks < 1:
        raise ValueError("--min-source-blocks must be >= 1")
    if args.max_points > 0 and args.max_points < args.min_points:
        raise ValueError("--max-points must be 0 or >= --min-points")
    if args.min_effective_points <= 0.0:
        raise ValueError("--min-effective-points must be positive")

    camera_json = load_camera_jsonl(args.camera_param)
    cameras = build_camera_lookup(camera_json)

    header = camera_json["header"]
    if args.depth_scale_real is None:
        precision = float(header.get("depth_scale_precision", 1.0))
        if precision <= 0.0:
            raise ValueError("Invalid depth_scale_precision")
        depth_scale_real = float(header["depth_scale"]) / precision
    else:
        depth_scale_real = float(args.depth_scale_real)

    mv_by_frame = parse_mv_csv(args.mv_csv, args.num_frames)

    print("model            : inverse-depth constant rho(x,y)=c")
    print(f"fit block        : {args.fit_block}x{args.fit_block}")
    print(f"neighbor radius  : {args.neighbor_radius} blocks")
    print("predictor source : raster-causal neighboring blocks")
    print("current block MV : disabled")
    print(f"min points       : {args.min_points}")
    print(f"min source blocks: {args.min_source_blocks}")
    print(f"max points       : {args.max_points or 'unlimited'}")
    print(f"depth scale real : {depth_scale_real:.12g}")

    depth_frames: List[np.ndarray] = []
    frame_stats: List[Dict[str, Any]] = []

    for poc in range(args.num_frames):
        observations = make_depth_observations(
            mv_rows=mv_by_frame[poc],
            cameras=cameras,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_parallax=args.min_parallax,
            max_reproj_error=args.max_reproj_error,
        )

        jobs = build_block_fit_jobs(
            observations=observations,
            width=args.width,
            height=args.height,
            fit_block=args.fit_block,
            neighbor_radius=args.neighbor_radius,
            min_points=args.min_points,
            min_source_blocks=args.min_source_blocks,
            max_points=args.max_points,
        )

        constants: List[Optional[float]] = []
        for job in jobs:
            constants.append(
                fit_inv_depth_constant_cpu(
                    depths=job["depths"],
                    reproj_errors=job["errors"],
                    spatial_distances=job["spatial_distances"],
                    source_sizes=job["source_sizes"],
                    irls_iters=args.irls_iters,
                    huber_delta=args.huber_delta,
                    distance_decay=args.distance_decay,
                    source_balance_power=args.source_balance_power,
                    min_effective_points=args.min_effective_points,
                    max_relative_invd_mad=args.max_relative_invd_mad,
                )
            )

        depth, valid = render_constant_jobs(
            jobs=jobs,
            constants=constants,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

        depth_frames.append(depth)
        valid_ratio = float(np.mean(valid))
        successful = sum(c is not None for c in constants)

        used_points = [len(job["depths"]) for job in jobs]
        used_sources = [job["num_source_blocks"] for job in jobs]

        frame_stats.append(
            {
                "poc": poc,
                "mv_rows": len(mv_by_frame[poc]),
                "valid_depth_observations": len(observations),
                "fit_jobs": len(jobs),
                "successful_constants": successful,
                "mean_points_per_job": (
                    float(np.mean(used_points)) if used_points else 0.0
                ),
                "mean_source_blocks_per_job": (
                    float(np.mean(used_sources)) if used_sources else 0.0
                ),
                "valid_pixel_ratio": valid_ratio,
            }
        )
        print_progress(poc, args.num_frames, len(observations), valid_ratio)

    print()

    write_depth_yuv420p10le(
        output_path=args.out_yuv,
        depth_frames=depth_frames,
        depth_scale_real=depth_scale_real,
    )

    stats_path = str(Path(args.out_yuv).with_suffix(".stats.json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mv_csv": args.mv_csv,
                "camera_param": args.camera_param,
                "out_yuv": args.out_yuv,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "model": "inverse_depth_constant",
                "equation": "rho(x,y)=c, z=1/c",
                "fit_block": args.fit_block,
                "neighbor_radius": args.neighbor_radius,
                "causal_rule": {
                    "same_row": "left_only",
                    "previous_rows": "left_center_right",
                    "current_block_mv_used": False,
                },
                "min_points": args.min_points,
                "min_source_blocks": args.min_source_blocks,
                "max_points": args.max_points,
                "irls_iters": args.irls_iters,
                "huber_delta": args.huber_delta,
                "distance_decay": args.distance_decay,
                "source_balance_power": args.source_balance_power,
                "min_effective_points": args.min_effective_points,
                "max_relative_invd_mad": args.max_relative_invd_mad,
                "depth_scale_real": depth_scale_real,
                "frames": frame_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Depth YUV : {args.out_yuv}")
    print(f"Stats     : {stats_path}")


if __name__ == "__main__":
    main()
