#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MV + camera parameter -> causal candidate-based dense depth YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Main behavior:
  1) Convert each MV observation into a closed-form depth observation.
  2) The current fit-block does not use its own MV observations.
  3) Build multiple inverse-depth candidates from causal neighboring blocks:
       - left + top + top-left
       - neighbor pairs
       - individual neighbors
       - same-reference subsets
       - wider causal neighborhood
       - copied/recentered previously selected planes
       - constant inverse-depth fallback
  4) Select the candidate using:
       - fit error on causal MV-derived observations
       - continuity with already reconstructed left/top boundaries
       - raw invalid-pixel penalty
       - excessive plane-slope penalty
  5) Optionally retain moderately inaccurate MV observations with reduced weight.
  6) Optionally clip rendered inverse depth into the valid range so that a valid
     selected candidate fills the whole block instead of leaving partial holes.
  7) Last-resort fallback can be strict causal, frame median, or zero.

The default settings remain causal except for reading the frame's MV observations
when building the observation bins. The current block's own observations are never
used as a predictor source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64)

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

@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DepthObservation:
    poc: int
    x: float
    y: float
    depth: float
    reproj_error: float
    ref_poc: int
    list_id: str

    @property
    def inv_depth(self) -> float:
        return 1.0 / self.depth


@dataclass
class InvDepthModel:
    a: float
    b: float
    c: float
    cx: float
    cy: float
    kind: str
    source: str
    point_count: int

    def evaluate(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        return self.a * (np.asarray(x) - self.cx) + self.b * (np.asarray(y) - self.cy) + self.c


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
) -> Optional[Tuple[float, float]]:
    """Return positive depth and reprojection error.

    Reprojection error is returned instead of being rejected here. This lets the
    caller keep moderately inaccurate observations with a small fitting weight.
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

    reproj_error = float(np.linalg.norm(pred - np.array([ur, vr], dtype=np.float64)))
    if not np.isfinite(reproj_error):
        return None

    return depth, reproj_error


def make_depth_observations(
    mv_rows: List[MVObservation],
    cameras: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    min_parallax: float,
    hard_max_reproj_error: float,
) -> Tuple[List[DepthObservation], Dict[str, int]]:
    out: List[DepthObservation] = []
    stats = {
        "missing_camera": 0,
        "center_outside": 0,
        "solve_failed": 0,
        "reprojection_hard_reject": 0,
        "depth_range_reject": 0,
        "accepted": 0,
    }

    for row in mv_rows:
        if row.poc not in cameras or row.ref_poc not in cameras:
            stats["missing_camera"] += 1
            continue

        cx = row.x + (row.w - 1) * 0.5
        cy = row.y + (row.h - 1) * 0.5
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            stats["center_outside"] += 1
            continue

        solved = solve_depth_closed_form(
            u=cx,
            v=cy,
            mv_x=row.mv_x,
            mv_y=row.mv_y,
            cam_cur=cameras[row.poc],
            cam_ref=cameras[row.ref_poc],
            min_parallax=min_parallax,
        )
        if solved is None:
            stats["solve_failed"] += 1
            continue

        depth, err = solved
        if err > hard_max_reproj_error:
            stats["reprojection_hard_reject"] += 1
            continue
        if depth < min_depth or depth > max_depth:
            stats["depth_range_reject"] += 1
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
        stats["accepted"] += 1

    return out, stats


# ============================================================
# Robust inverse-depth fitting
# ============================================================

def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def observation_base_weights(errors: np.ndarray, soft_reproj_error: float) -> np.ndarray:
    scale = max(float(soft_reproj_error), 1e-6)
    r = errors / scale
    return 1.0 / (1.0 + r * r)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("weighted_median requires at least one value")
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.median(values))
    idx = int(np.searchsorted(np.cumsum(w), 0.5 * total, side="left"))
    return float(v[min(idx, len(v) - 1)])


def fit_inv_depth_plane_cpu(
    observations: Sequence[DepthObservation],
    cx: float,
    cy: float,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    soft_reproj_error: float,
    source: str,
) -> Optional[InvDepthModel]:
    if len(observations) < 3:
        return None

    xs = np.asarray([o.x for o in observations], dtype=np.float64)
    ys = np.asarray([o.y for o in observations], dtype=np.float64)
    invz = np.asarray([o.inv_depth for o in observations], dtype=np.float64)
    errs = np.asarray([o.reproj_error for o in observations], dtype=np.float64)
    A = np.stack([xs - cx, ys - cy, np.ones_like(xs)], axis=1)

    base_w = observation_base_weights(errs, soft_reproj_error)
    weights = base_w.copy()
    coeff: Optional[np.ndarray] = None

    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-12))
        Aw = A * sw[:, None]
        bw = invz * sw
        normal = Aw.T @ Aw
        eig = np.linalg.eigvalsh(normal)
        if eig[-1] <= 1e-15 or eig[0] / eig[-1] < min_condition:
            return None
        try:
            coeff = np.linalg.solve(normal, Aw.T @ bw)
        except np.linalg.LinAlgError:
            return None

        residual = invz - A @ coeff
        med = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - med)))
        scale = max(scale, 1e-10)
        weights = base_w * huber_weights(residual / scale, huber_delta)

    if coeff is None or not np.isfinite(coeff).all():
        return None

    return InvDepthModel(
        a=float(coeff[0]),
        b=float(coeff[1]),
        c=float(coeff[2]),
        cx=cx,
        cy=cy,
        kind="plane",
        source=source,
        point_count=len(observations),
    )


def fit_inv_depth_axis_cpu(
    observations: Sequence[DepthObservation],
    cx: float,
    cy: float,
    axis: str,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    soft_reproj_error: float,
    source: str,
) -> Optional[InvDepthModel]:
    if len(observations) < 2:
        return None

    coords = np.asarray(
        [o.x - cx if axis == "x" else o.y - cy for o in observations],
        dtype=np.float64,
    )
    invz = np.asarray([o.inv_depth for o in observations], dtype=np.float64)
    errs = np.asarray([o.reproj_error for o in observations], dtype=np.float64)
    A = np.stack([coords, np.ones_like(coords)], axis=1)

    base_w = observation_base_weights(errs, soft_reproj_error)
    weights = base_w.copy()
    coeff: Optional[np.ndarray] = None

    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-12))
        Aw = A * sw[:, None]
        bw = invz * sw
        normal = Aw.T @ Aw
        eig = np.linalg.eigvalsh(normal)
        if eig[-1] <= 1e-15 or eig[0] / eig[-1] < min_condition:
            return None
        try:
            coeff = np.linalg.solve(normal, Aw.T @ bw)
        except np.linalg.LinAlgError:
            return None

        residual = invz - A @ coeff
        med = float(np.median(residual))
        scale = max(1.4826 * float(np.median(np.abs(residual - med))), 1e-10)
        weights = base_w * huber_weights(residual / scale, huber_delta)

    if coeff is None or not np.isfinite(coeff).all():
        return None

    if axis == "x":
        a, b = float(coeff[0]), 0.0
    else:
        a, b = 0.0, float(coeff[0])

    return InvDepthModel(
        a=a,
        b=b,
        c=float(coeff[1]),
        cx=cx,
        cy=cy,
        kind=f"line_{axis}",
        source=source,
        point_count=len(observations),
    )


def fit_inv_depth_constant(
    observations: Sequence[DepthObservation],
    cx: float,
    cy: float,
    soft_reproj_error: float,
    source: str,
) -> Optional[InvDepthModel]:
    if not observations:
        return None
    invz = np.asarray([o.inv_depth for o in observations], dtype=np.float64)
    errs = np.asarray([o.reproj_error for o in observations], dtype=np.float64)
    weights = observation_base_weights(errs, soft_reproj_error)
    c = weighted_median(invz, weights)
    if not np.isfinite(c) or c <= 0.0:
        return None
    return InvDepthModel(
        a=0.0,
        b=0.0,
        c=c,
        cx=cx,
        cy=cy,
        kind="constant",
        source=source,
        point_count=len(observations),
    )


def recenter_model(
    model: InvDepthModel,
    cx: float,
    cy: float,
    source: str,
) -> InvDepthModel:
    c_new = (
        model.a * (cx - model.cx)
        + model.b * (cy - model.cy)
        + model.c
    )
    return InvDepthModel(
        a=model.a,
        b=model.b,
        c=float(c_new),
        cx=cx,
        cy=cy,
        kind="copy_" + model.kind,
        source=source,
        point_count=model.point_count,
    )


# ============================================================
# Candidate creation and selection
# ============================================================

def unique_indices(indices: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(i) for i in indices))


def build_block_observation_bins(
    observations: Sequence[DepthObservation],
    fit_block: int,
) -> Dict[Tuple[int, int], List[int]]:
    bins: Dict[Tuple[int, int], List[int]] = {}
    for idx, obs in enumerate(observations):
        key = (int(obs.x) // fit_block, int(obs.y) // fit_block)
        bins.setdefault(key, []).append(idx)
    return bins


def causal_neighbor_keys(gx: int, gy: int, ring: int) -> List[Tuple[int, int]]:
    """Causal neighbors up to a Chebyshev ring.

    Includes blocks on previous rows and blocks to the left on the current row.
    Never includes the current or future raster blocks.
    """
    keys: List[Tuple[int, int]] = []
    for dy in range(-ring, 1):
        for dx in range(-ring, ring + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = gx + dx, gy + dy
            if nx < 0 or ny < 0:
                continue
            if dy == 0 and dx >= 0:
                continue
            if max(abs(dx), abs(dy)) <= ring:
                keys.append((nx, ny))
    return keys


def immediate_neighbor_groups(
    gx: int,
    gy: int,
    bins: Dict[Tuple[int, int], List[int]],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    if gx > 0:
        groups["left"] = list(bins.get((gx - 1, gy), []))
    if gy > 0:
        groups["top"] = list(bins.get((gx, gy - 1), []))
    if gx > 0 and gy > 0:
        groups["top_left"] = list(bins.get((gx - 1, gy - 1), []))
    return groups


def build_observation_subsets(
    observations: Sequence[DepthObservation],
    bins: Dict[Tuple[int, int], List[int]],
    gx: int,
    gy: int,
    max_neighbor_rings: int,
    add_reference_candidates: bool,
) -> Tuple[List[Tuple[str, List[int]]], List[int], List[int]]:
    groups = immediate_neighbor_groups(gx, gy, bins)
    left = groups.get("left", [])
    top = groups.get("top", [])
    tl = groups.get("top_left", [])

    subsets: List[Tuple[str, List[int]]] = []

    def add(name: str, values: Iterable[int]) -> None:
        ids = unique_indices(values)
        if ids:
            subsets.append((name, ids))

    add("all_immediate", left + top + tl)
    add("left_top", left + top)
    add("left_top_left", left + tl)
    add("top_top_left", top + tl)
    add("left", left)
    add("top", top)
    add("top_left", tl)

    expanded: List[int] = []
    for ring in range(2, max(2, max_neighbor_rings + 1)):
        ring_ids: List[int] = []
        for key in causal_neighbor_keys(gx, gy, ring):
            ring_ids.extend(bins.get(key, []))
        ring_ids = unique_indices(ring_ids)
        add(f"causal_ring_{ring}", ring_ids)
        expanded.extend(ring_ids)

    validation_ids = unique_indices(left + top + tl)
    if not validation_ids:
        validation_ids = unique_indices(expanded)

    if add_reference_candidates:
        base_ids = unique_indices(left + top + tl + expanded)
        by_ref: Dict[Tuple[str, int], List[int]] = {}
        by_ref_any_list: Dict[int, List[int]] = {}
        for idx in base_ids:
            obs = observations[idx]
            by_ref.setdefault((obs.list_id, obs.ref_poc), []).append(idx)
            by_ref_any_list.setdefault(obs.ref_poc, []).append(idx)
        for (list_id, ref_poc), ids in by_ref.items():
            add(f"ref_{list_id}_{ref_poc}", ids)
        for ref_poc, ids in by_ref_any_list.items():
            add(f"ref_any_{ref_poc}", ids)

    # Remove duplicate subsets with different names but exactly the same points.
    dedup: List[Tuple[str, List[int]]] = []
    seen: set[Tuple[int, ...]] = set()
    for name, ids in subsets:
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((name, ids))

    return dedup, validation_ids, unique_indices(expanded)


def model_raw_block(
    model: InvDepthModel,
    bx: int,
    by: int,
    bw: int,
    bh: int,
) -> np.ndarray:
    xs = np.arange(bx, bx + bw, dtype=np.float64)
    ys = np.arange(by, by + bh, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return model.evaluate(xx, yy)


def weighted_relative_observation_error(
    model: InvDepthModel,
    observations: Sequence[DepthObservation],
    soft_reproj_error: float,
) -> float:
    if not observations:
        return 0.0
    xs = np.asarray([o.x for o in observations], dtype=np.float64)
    ys = np.asarray([o.y for o in observations], dtype=np.float64)
    target = np.asarray([o.inv_depth for o in observations], dtype=np.float64)
    errs = np.asarray([o.reproj_error for o in observations], dtype=np.float64)
    pred = model.evaluate(xs, ys)
    rel = np.abs(pred - target) / np.maximum(np.abs(target), 1e-12)
    rel = np.minimum(rel, 10.0)
    weights = observation_base_weights(errs, soft_reproj_error)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(rel))
    return float(np.sum(weights * rel) / total)


def boundary_relative_error(
    raw_invz: np.ndarray,
    bx: int,
    by: int,
    recon_invz: np.ndarray,
    recon_valid: np.ndarray,
) -> float:
    errors: List[np.ndarray] = []

    if bx > 0:
        mask = recon_valid[by:by + raw_invz.shape[0], bx - 1]
        if np.any(mask):
            ref = recon_invz[by:by + raw_invz.shape[0], bx - 1][mask]
            pred = raw_invz[:, 0][mask]
            errors.append(np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-12))

    if by > 0:
        mask = recon_valid[by - 1, bx:bx + raw_invz.shape[1]]
        if np.any(mask):
            ref = recon_invz[by - 1, bx:bx + raw_invz.shape[1]][mask]
            pred = raw_invz[0, :][mask]
            errors.append(np.abs(pred - ref) / np.maximum(np.abs(ref), 1e-12))

    if not errors:
        return 0.0
    joined = np.concatenate(errors)
    joined = joined[np.isfinite(joined)]
    if joined.size == 0:
        return 0.0
    return float(np.median(np.minimum(joined, 10.0)))


def candidate_score(
    model: InvDepthModel,
    validation_observations: Sequence[DepthObservation],
    bx: int,
    by: int,
    bw: int,
    bh: int,
    recon_invz: np.ndarray,
    recon_valid: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
    soft_reproj_error: float,
    data_weight: float,
    boundary_weight: float,
    invalid_weight: float,
    slope_weight: float,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    raw = model_raw_block(model, bx, by, bw, bh)
    min_invz = 1.0 / max_depth
    max_invz = 1.0 / min_depth
    raw_valid = np.isfinite(raw) & (raw >= min_invz) & (raw <= max_invz)
    invalid_ratio = 1.0 - float(np.mean(raw_valid))

    data_err = weighted_relative_observation_error(
        model,
        validation_observations,
        soft_reproj_error,
    )
    boundary_err = boundary_relative_error(raw, bx, by, recon_invz, recon_valid)

    center_scale = max(abs(model.c), min_invz, 1e-12)
    relative_span = (
        abs(model.a) * max(bw - 1, 1)
        + abs(model.b) * max(bh - 1, 1)
    ) / center_scale
    absolute_slope_excess = max(0.0, abs(model.a) - max_plane_slope) + max(
        0.0, abs(model.b) - max_plane_slope
    )
    slope_penalty = min(relative_span, 20.0) + 100.0 * absolute_slope_excess

    score = (
        data_weight * data_err
        + boundary_weight * boundary_err
        + invalid_weight * invalid_ratio
        + slope_weight * slope_penalty
    )

    details = {
        "data": data_err,
        "boundary": boundary_err,
        "invalid": invalid_ratio,
        "slope": slope_penalty,
    }
    return float(score), details, raw


def generate_models_for_subset(
    observations: Sequence[DepthObservation],
    cx: float,
    cy: float,
    source: str,
    min_points: int,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    soft_reproj_error: float,
) -> List[InvDepthModel]:
    models: List[InvDepthModel] = []

    if len(observations) >= max(3, min_points):
        plane = fit_inv_depth_plane_cpu(
            observations,
            cx,
            cy,
            irls_iters,
            huber_delta,
            min_condition,
            soft_reproj_error,
            source,
        )
        if plane is not None:
            models.append(plane)

    # Degenerate sample geometry often rejects a 2-D plane. Axis models and a
    # constant model preserve coverage without forcing an unstable full plane.
    if len(observations) >= 2:
        line_x = fit_inv_depth_axis_cpu(
            observations,
            cx,
            cy,
            "x",
            irls_iters,
            huber_delta,
            min_condition,
            soft_reproj_error,
            source,
        )
        line_y = fit_inv_depth_axis_cpu(
            observations,
            cx,
            cy,
            "y",
            irls_iters,
            huber_delta,
            min_condition,
            soft_reproj_error,
            source,
        )
        if line_x is not None:
            models.append(line_x)
        if line_y is not None:
            models.append(line_y)

    constant = fit_inv_depth_constant(
        observations,
        cx,
        cy,
        soft_reproj_error,
        source,
    )
    if constant is not None:
        models.append(constant)

    return models


def reconstruct_frame_candidate_based(
    observations: List[DepthObservation],
    width: int,
    height: int,
    fit_block: int,
    min_points: int,
    max_neighbor_rings: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    soft_reproj_error: float,
    add_reference_candidates: bool,
    clip_render_depth: bool,
    last_resort: str,
    data_weight: float,
    boundary_weight: float,
    invalid_weight: float,
    slope_weight: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    depth = np.zeros((height, width), dtype=np.float64)
    invz_map = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)

    bins = build_block_observation_bins(observations, fit_block)
    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block
    selected_models: Dict[Tuple[int, int], InvDepthModel] = {}

    frame_median_model: Optional[InvDepthModel] = None
    if observations:
        frame_median_model = fit_inv_depth_constant(
            observations,
            0.0,
            0.0,
            soft_reproj_error,
            "frame_median",
        )

    stats: Dict[str, Any] = {
        "total_blocks": grid_w * grid_h,
        "blocks_with_observation_candidate": 0,
        "blocks_using_copy_candidate": 0,
        "blocks_using_frame_median": 0,
        "blocks_without_candidate": 0,
        "raw_partial_or_invalid_blocks": 0,
        "pixels_recovered_by_clipping": 0,
        "candidate_counts": {},
        "selected_kind_counts": {},
        "selected_source_counts": {},
    }

    for gy in range(grid_h):
        by = gy * fit_block
        bh = min(fit_block, height - by)
        cy = by + (bh - 1) * 0.5

        for gx in range(grid_w):
            bx = gx * fit_block
            bw = min(fit_block, width - bx)
            cx = bx + (bw - 1) * 0.5

            subset_specs, validation_ids, expanded_ids = build_observation_subsets(
                observations,
                bins,
                gx,
                gy,
                max_neighbor_rings,
                add_reference_candidates,
            )
            validation_obs = [observations[i] for i in validation_ids]

            candidates: List[InvDepthModel] = []
            for source_name, ids in subset_specs:
                subset_obs = [observations[i] for i in ids]
                models = generate_models_for_subset(
                    subset_obs,
                    cx,
                    cy,
                    source_name,
                    min_points,
                    irls_iters,
                    huber_delta,
                    min_condition,
                    soft_reproj_error,
                )
                candidates.extend(models)
                stats["candidate_counts"][source_name] = (
                    stats["candidate_counts"].get(source_name, 0) + len(models)
                )

            if candidates:
                stats["blocks_with_observation_candidate"] += 1

            # Previously selected causal block models are explicit candidates.
            copy_sources = [
                ("copy_left", (gx - 1, gy)),
                ("copy_top", (gx, gy - 1)),
                ("copy_top_left", (gx - 1, gy - 1)),
            ]
            copy_added = 0
            for source_name, key in copy_sources:
                if key in selected_models:
                    candidates.append(
                        recenter_model(selected_models[key], cx, cy, source_name)
                    )
                    copy_added += 1
            if copy_added:
                stats["blocks_using_copy_candidate"] += 1

            # If immediate validation points are absent, use wider causal points
            # to score candidate extrapolation. Copy candidates can still be
            # scored by boundary continuity when no observations exist at all.
            if not validation_obs and expanded_ids:
                validation_obs = [observations[i] for i in expanded_ids]

            best_model: Optional[InvDepthModel] = None
            best_raw: Optional[np.ndarray] = None
            best_score = float("inf")

            for candidate in candidates:
                score, _details, raw = candidate_score(
                    candidate,
                    validation_obs,
                    bx,
                    by,
                    bw,
                    bh,
                    invz_map,
                    valid,
                    min_depth,
                    max_depth,
                    max_plane_slope,
                    soft_reproj_error,
                    data_weight,
                    boundary_weight,
                    invalid_weight,
                    slope_weight,
                )
                if score < best_score:
                    best_score = score
                    best_model = candidate
                    best_raw = raw

            if best_model is None and last_resort == "frame_median" and frame_median_model is not None:
                best_model = recenter_model(frame_median_model, cx, cy, "frame_median")
                best_raw = model_raw_block(best_model, bx, by, bw, bh)
                stats["blocks_using_frame_median"] += 1

            if best_model is None:
                stats["blocks_without_candidate"] += 1
                continue

            assert best_raw is not None
            min_invz = 1.0 / max_depth
            max_invz = 1.0 / min_depth
            raw_valid = np.isfinite(best_raw) & (best_raw >= min_invz) & (best_raw <= max_invz)
            if not np.all(raw_valid):
                stats["raw_partial_or_invalid_blocks"] += 1

            if clip_render_depth:
                safe = np.nan_to_num(
                    best_raw,
                    nan=best_model.c,
                    posinf=max_invz,
                    neginf=min_invz,
                )
                block_invz = np.clip(safe, min_invz, max_invz)
                block_valid = np.ones((bh, bw), dtype=bool)
                stats["pixels_recovered_by_clipping"] += int(np.size(raw_valid) - np.count_nonzero(raw_valid))
            else:
                block_invz = np.zeros((bh, bw), dtype=np.float64)
                block_invz[raw_valid] = best_raw[raw_valid]
                block_valid = raw_valid

            block_depth = np.zeros((bh, bw), dtype=np.float64)
            block_depth[block_valid] = 1.0 / block_invz[block_valid]

            depth[by:by + bh, bx:bx + bw][block_valid] = block_depth[block_valid]
            invz_map[by:by + bh, bx:bx + bw][block_valid] = block_invz[block_valid]
            valid[by:by + bh, bx:bx + bw][block_valid] = True
            selected_models[(gx, gy)] = best_model

            stats["selected_kind_counts"][best_model.kind] = (
                stats["selected_kind_counts"].get(best_model.kind, 0) + 1
            )
            stats["selected_source_counts"][best_model.source] = (
                stats["selected_source_counts"].get(best_model.source, 0) + 1
            )

    stats["valid_pixel_ratio"] = float(np.mean(valid))
    stats["selected_blocks"] = len(selected_models)
    return depth, valid, stats


# ============================================================
# Output
# ============================================================

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


def print_progress(frame_idx: int, num_frames: int, valid_obs: int, valid_ratio: float) -> None:
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
            "Causal candidate-based depth prediction from left/top/top-left "
            "MV-derived depth observations with coverage fallbacks."
        )
    )
    ap.add_argument("--mv-csv", required=True)
    ap.add_argument("--camera-param", required=True)
    ap.add_argument("--out-yuv", required=True)

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--fit-block", type=int, default=16)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument(
        "--max-neighbor-rings",
        type=int,
        default=2,
        help="Use wider causal observations as additional candidates. Default: 2.",
    )

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument("--min-parallax", type=float, default=1e-6)
    ap.add_argument(
        "--soft-reproj-error",
        type=float,
        default=1.5,
        help="Confidence scale; observations above this are down-weighted, not rejected.",
    )
    ap.add_argument(
        "--hard-max-reproj-error",
        type=float,
        default=8.0,
        help="Only observations above this reprojection error are rejected.",
    )

    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-condition", type=float, default=1e-8)
    ap.add_argument("--max-plane-slope", type=float, default=1.0)

    ap.add_argument(
        "--reference-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add per-reference and per-(list,reference) observation candidates.",
    )
    ap.add_argument(
        "--clip-render-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip candidate inverse depth to the legal range instead of leaving pixel holes.",
    )
    ap.add_argument(
        "--last-resort",
        choices=("causal", "frame_median", "zero"),
        default="causal",
        help=(
            "causal: observation/copy candidates only; frame_median: fill every "
            "remaining block with frame median; zero: same output behavior as strict failure."
        ),
    )

    ap.add_argument("--score-data-weight", type=float, default=1.0)
    ap.add_argument("--score-boundary-weight", type=float, default=0.35)
    ap.add_argument("--score-invalid-weight", type=float, default=2.0)
    ap.add_argument("--score-slope-weight", type=float, default=0.02)

    ap.add_argument(
        "--depth-scale-real",
        type=float,
        default=None,
        help="Override camera header depth scale.",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help=(
            "Retained for command compatibility. Candidate selection is raster-causal "
            "and currently runs on CPU."
        ),
    )

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.min_points < 1:
        raise ValueError("--min-points must be >= 1")
    if args.max_neighbor_rings < 1:
        raise ValueError("--max-neighbor-rings must be >= 1")
    if args.soft_reproj_error <= 0.0 or args.hard_max_reproj_error <= 0.0:
        raise ValueError("Reprojection thresholds must be positive")

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

    print("device           : cpu (raster-causal candidate selection)")
    print(f"fit block        : {args.fit_block}x{args.fit_block}")
    print("predictor source : causal neighbors; current block MV disabled")
    print(f"neighbor rings   : {args.max_neighbor_rings}")
    print(f"soft/hard reproj : {args.soft_reproj_error} / {args.hard_max_reproj_error}")
    print(f"last resort      : {args.last_resort}")
    print(f"depth scale real : {depth_scale_real:.12g}")

    depth_frames: List[np.ndarray] = []
    frame_stats: List[Dict[str, Any]] = []

    for poc in range(args.num_frames):
        observations, observation_stats = make_depth_observations(
            mv_rows=mv_by_frame[poc],
            cameras=cameras,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_parallax=args.min_parallax,
            hard_max_reproj_error=args.hard_max_reproj_error,
        )

        depth, valid, reconstruction_stats = reconstruct_frame_candidate_based(
            observations=observations,
            width=args.width,
            height=args.height,
            fit_block=args.fit_block,
            min_points=args.min_points,
            max_neighbor_rings=args.max_neighbor_rings,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_plane_slope=args.max_plane_slope,
            irls_iters=args.irls_iters,
            huber_delta=args.huber_delta,
            min_condition=args.min_condition,
            soft_reproj_error=args.soft_reproj_error,
            add_reference_candidates=args.reference_candidates,
            clip_render_depth=args.clip_render_depth,
            last_resort=args.last_resort,
            data_weight=args.score_data_weight,
            boundary_weight=args.score_boundary_weight,
            invalid_weight=args.score_invalid_weight,
            slope_weight=args.score_slope_weight,
        )

        depth_frames.append(depth)
        valid_ratio = float(np.mean(valid))
        frame_stats.append(
            {
                "poc": poc,
                "mv_rows": len(mv_by_frame[poc]),
                "valid_depth_observations": len(observations),
                "observation_rejections": observation_stats,
                "reconstruction": reconstruction_stats,
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
                "fit_block": args.fit_block,
                "current_block_mv_used": False,
                "max_neighbor_rings": args.max_neighbor_rings,
                "reference_candidates": args.reference_candidates,
                "clip_render_depth": args.clip_render_depth,
                "last_resort": args.last_resort,
                "soft_reproj_error": args.soft_reproj_error,
                "hard_max_reproj_error": args.hard_max_reproj_error,
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
