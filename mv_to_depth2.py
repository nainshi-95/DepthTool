#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal MV + camera parameter -> propagated depth YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Core behavior:
  1) Generate a local depth predictor for the current POC from only the
     left/top/top-left fit-block MV-derived observations.
  2) Process pictures in decoder order, not display POC order.
  3) Forward-project depth maps of already decoded pictures into the current
     picture. A future picture is never used.
  4) Select/fuse local and propagated candidates pixel by pixel using
     inverse-depth agreement and confidence.
  5) Register the fused current depth in a causal depth bank so that good depth
     continues to propagate to pictures decoded later.

Default RA decoder order for 33 pictures:
  0, 32, 16, 8, 24, 4, 12, 20, 28, ...

Depth output is written in display POC order even though reconstruction is
performed in decoder order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


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

        C2W = np.linalg.inv(W2C)
        lookup[poc] = {
            "poc": poc,
            "K": K,
            "W2C": W2C,
            "C2W": C2W,
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


@dataclass
class DepthState:
    poc: int
    decode_rank: int
    depth: np.ndarray
    confidence: np.ndarray
    valid_ratio: float
    mean_confidence: float
    quality_score: float
    local_valid_ratio: float
    propagated_valid_ratio: float


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
    max_reproj_error: float,
) -> Optional[Tuple[float, float]]:
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
    if not np.isfinite(reproj_error) or reproj_error > max_reproj_error:
        return None

    return depth, reproj_error


# ============================================================
# Robust inverse-depth plane fitting
# ============================================================

def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def fit_inv_depth_plane_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    reproj_errors: np.ndarray,
    cx: float,
    cy: float,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
) -> Optional[np.ndarray]:
    if depths.size < 3:
        return None

    invz = 1.0 / depths
    A = np.stack([xs - cx, ys - cy, np.ones_like(xs)], axis=1)
    weights = 1.0 / np.maximum(1.0 + reproj_errors * reproj_errors, 1e-6)

    coeff = None
    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-10))
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
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), 1e-8)
        robust = huber_weights(residual / scale, huber_delta)
        weights = robust / np.maximum(1.0 + reproj_errors * reproj_errors, 1e-6)

    if coeff is None or not np.isfinite(coeff).all():
        return None
    return coeff.astype(np.float64)


def fit_inv_depth_planes_gpu(
    blocks: List[Dict[str, Any]],
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    device: str,
) -> List[Optional[np.ndarray]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if not blocks:
        return []

    dev = torch.device(device)
    B = len(blocks)
    max_n = max(len(b["depths"]) for b in blocks)

    A = torch.zeros((B, max_n, 3), dtype=torch.float64, device=dev)
    y = torch.zeros((B, max_n), dtype=torch.float64, device=dev)
    base_w = torch.zeros((B, max_n), dtype=torch.float64, device=dev)
    mask = torch.zeros((B, max_n), dtype=torch.bool, device=dev)

    for i, b in enumerate(blocks):
        n = len(b["depths"])
        xs = torch.as_tensor(b["xs"], dtype=torch.float64, device=dev)
        ys = torch.as_tensor(b["ys"], dtype=torch.float64, device=dev)
        depths = torch.as_tensor(b["depths"], dtype=torch.float64, device=dev)
        errs = torch.as_tensor(b["errors"], dtype=torch.float64, device=dev)
        cx = float(b["cx"])
        cy = float(b["cy"])

        A[i, :n, 0] = xs - cx
        A[i, :n, 1] = ys - cy
        A[i, :n, 2] = 1.0
        y[i, :n] = 1.0 / depths
        base_w[i, :n] = 1.0 / (1.0 + errs * errs)
        mask[i, :n] = True

    weights = base_w.clone()
    coeff = torch.zeros((B, 3), dtype=torch.float64, device=dev)
    valid_batch = torch.ones(B, dtype=torch.bool, device=dev)
    eye = torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0)

    for _ in range(max(1, irls_iters)):
        W = weights * mask.to(torch.float64)
        AtW = A.transpose(1, 2) * W.unsqueeze(1)
        normal = AtW @ A
        rhs = (AtW @ y.unsqueeze(-1)).squeeze(-1)

        eig = torch.linalg.eigvalsh(normal)
        stable = (eig[:, -1] > 1e-15) & (
            eig[:, 0] / torch.clamp(eig[:, -1], min=1e-15) >= min_condition
        )
        valid_batch &= stable

        normal_safe = normal + (~stable).to(torch.float64).view(-1, 1, 1) * eye
        coeff = torch.linalg.solve(normal_safe, rhs.unsqueeze(-1)).squeeze(-1)

        residual = y - torch.sum(A * coeff[:, None, :], dim=2)
        residual_masked = torch.where(mask, residual, torch.nan)
        med = torch.nanmedian(residual_masked, dim=1).values
        mad = torch.nanmedian(
            torch.abs(residual_masked - med[:, None]), dim=1
        ).values
        scale = torch.clamp(1.4826 * mad, min=1e-8)

        r = torch.abs(residual) / scale[:, None]
        robust = torch.where(
            r <= huber_delta,
            torch.ones_like(r),
            huber_delta / torch.clamp(r, min=1e-12),
        )
        weights = base_w * robust * mask.to(torch.float64)

    coeff_np = coeff.detach().cpu().numpy()
    valid_np = valid_batch.detach().cpu().numpy()

    out: List[Optional[np.ndarray]] = []
    for i in range(B):
        if valid_np[i] and np.isfinite(coeff_np[i]).all():
            out.append(coeff_np[i].astype(np.float64))
        else:
            out.append(None)
    return out


# ============================================================
# Local current-frame predictor
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
        if depth < min_depth or depth > max_depth:
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


def build_block_fit_jobs(
    observations: List[DepthObservation],
    width: int,
    height: int,
    fit_block: int,
    neighborhood: int,
    min_points: int,
) -> List[Dict[str, Any]]:
    del neighborhood
    if not observations:
        return []

    block_obs: Dict[Tuple[int, int], List[DepthObservation]] = {}
    for o in observations:
        gx = int(o.x) // fit_block
        gy = int(o.y) // fit_block
        block_obs.setdefault((gx, gy), []).append(o)

    jobs: List[Dict[str, Any]] = []
    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block

    for gy in range(grid_h):
        by = gy * fit_block
        bh = min(fit_block, height - by)
        cy = by + (bh - 1) * 0.5

        for gx in range(grid_w):
            bx = gx * fit_block
            bw = min(fit_block, width - bx)
            cx = bx + (bw - 1) * 0.5

            source_keys: List[Tuple[int, int]] = []
            if gx > 0:
                source_keys.append((gx - 1, gy))
            if gy > 0:
                source_keys.append((gx, gy - 1))
            if gx > 0 and gy > 0:
                source_keys.append((gx - 1, gy - 1))

            selected: List[DepthObservation] = []
            for key in source_keys:
                selected.extend(block_obs.get(key, []))

            if len(selected) < min_points:
                continue

            jobs.append(
                {
                    "bx": bx,
                    "by": by,
                    "bw": bw,
                    "bh": bh,
                    "cx": cx,
                    "cy": cy,
                    "xs": np.asarray([o.x for o in selected], dtype=np.float64),
                    "ys": np.asarray([o.y for o in selected], dtype=np.float64),
                    "depths": np.asarray([o.depth for o in selected], dtype=np.float64),
                    "errors": np.asarray(
                        [o.reproj_error for o in selected], dtype=np.float64
                    ),
                }
            )

    return jobs


def render_jobs(
    jobs: List[Dict[str, Any]],
    coeffs: List[Optional[np.ndarray]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((height, width), dtype=np.float32)

    for job, coeff in zip(jobs, coeffs):
        if coeff is None:
            continue

        a, b, c = [float(v) for v in coeff]
        if abs(a) > max_plane_slope or abs(b) > max_plane_slope or c <= 0.0:
            continue

        bx, by = int(job["bx"]), int(job["by"])
        bw, bh = int(job["bw"]), int(job["bh"])
        cx, cy = float(job["cx"]), float(job["cy"])

        gx = np.arange(bx, bx + bw, dtype=np.float64)
        gy = np.arange(by, by + bh, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)

        invz = a * (xx - cx) + b * (yy - cy) + c
        block_valid = np.isfinite(invz) & (invz > 1.0 / max_depth)
        z = np.zeros_like(invz)
        z[block_valid] = 1.0 / invz[block_valid]
        block_valid &= (z >= min_depth) & (z <= max_depth)

        if not np.any(block_valid):
            continue

        obs_invz = 1.0 / np.asarray(job["depths"], dtype=np.float64)
        obs_pred = (
            a * (np.asarray(job["xs"]) - cx)
            + b * (np.asarray(job["ys"]) - cy)
            + c
        )
        rel_residual = np.abs(obs_pred - obs_invz) / np.maximum(obs_invz, 1e-12)
        fit_error = float(np.median(rel_residual))
        reproj = float(np.median(np.asarray(job["errors"], dtype=np.float64)))
        point_factor = min(1.0, len(job["depths"]) / max(2.0 * min_points, 1.0))
        fit_conf = math.exp(-4.0 * fit_error)
        reproj_conf = 1.0 / (1.0 + reproj * reproj)
        slope_conf = 1.0 / (1.0 + abs(a) + abs(b))
        block_conf = float(np.clip(point_factor * fit_conf * reproj_conf * slope_conf, 0.02, 1.0))

        dst = depth[by:by + bh, bx:bx + bw]
        vm = valid[by:by + bh, bx:bx + bw]
        cm = confidence[by:by + bh, bx:bx + bw]
        dst[block_valid] = z[block_valid]
        vm[block_valid] = True
        cm[block_valid] = block_conf

    return depth, valid, confidence


# ============================================================
# Decoder order
# ============================================================

def build_ra_decode_order(num_frames: int) -> List[int]:
    """Breadth-first hierarchical RA order: 0, N-1, midpoint levels."""
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [0]

    order = [0, num_frames - 1]
    queue: List[Tuple[int, int]] = [(0, num_frames - 1)]
    seen = set(order)

    while queue:
        next_queue: List[Tuple[int, int]] = []
        for left, right in queue:
            if right - left <= 1:
                continue
            mid = (left + right) // 2
            if mid not in seen:
                order.append(mid)
                seen.add(mid)
            if mid - left > 1:
                next_queue.append((left, mid))
            if right - mid > 1:
                next_queue.append((mid, right))
        queue = next_queue

    for poc in range(num_frames):
        if poc not in seen:
            order.append(poc)
    return order


def parse_decode_order(spec: str, num_frames: int) -> List[int]:
    value = spec.strip().lower()
    if value in ("auto", "ra"):
        order = build_ra_decode_order(num_frames)
    elif value in ("display", "poc"):
        order = list(range(num_frames))
    else:
        try:
            order = [int(x.strip()) for x in spec.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError("--decode-order must be ra, display, or comma-separated POCs") from exc

    if len(order) != num_frames or set(order) != set(range(num_frames)):
        raise ValueError(
            f"Decode order must contain each POC 0..{num_frames - 1} exactly once: {order}"
        )
    return order


# ============================================================
# Causal depth propagation
# ============================================================

def select_propagation_sources(
    states: Dict[int, DepthState],
    target_poc: int,
    target_mv_rows: Sequence[MVObservation],
    max_sources: int,
    min_source_quality: float,
    poc_distance_scale: float,
) -> List[DepthState]:
    if max_sources <= 0 or not states:
        return []

    referenced = {int(r.ref_poc) for r in target_mv_rows}
    candidates = [s for s in states.values() if s.quality_score >= min_source_quality]

    def rank(s: DepthState) -> Tuple[int, float]:
        is_direct_ref = 1 if s.poc in referenced else 0
        distance_factor = 1.0 + abs(s.poc - target_poc) / max(poc_distance_scale, 1e-6)
        score = s.quality_score / distance_factor
        return is_direct_ref, score

    candidates.sort(key=rank, reverse=True)
    return candidates[:max_sources]


def _update_zbuffer_chunk(
    depth_buffer: np.ndarray,
    conf_buffer: np.ndarray,
    src_buffer: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    conf: np.ndarray,
    source_id: int,
) -> None:
    h, w = depth_buffer.shape
    inside = (
        np.isfinite(z)
        & np.isfinite(conf)
        & (conf > 0.0)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    if not np.any(inside):
        return

    x = x[inside].astype(np.int64, copy=False)
    y = y[inside].astype(np.int64, copy=False)
    z = z[inside]
    conf = conf[inside]
    idx = y * w + x

    # Sort by destination index, then nearest depth. Keep one sample per pixel.
    order = np.lexsort((z, idx))
    idx_s = idx[order]
    first = np.empty(idx_s.size, dtype=bool)
    first[0] = True
    first[1:] = idx_s[1:] != idx_s[:-1]
    chosen = order[first]

    idx_c = idx[chosen]
    z_c = z[chosen]
    c_c = conf[chosen]

    flat_d = depth_buffer.reshape(-1)
    flat_c = conf_buffer.reshape(-1)
    flat_s = src_buffer.reshape(-1)
    replace = z_c < flat_d[idx_c]
    if np.any(replace):
        dst_idx = idx_c[replace]
        flat_d[dst_idx] = z_c[replace]
        flat_c[dst_idx] = c_c[replace]
        flat_s[dst_idx] = source_id


def forward_warp_depth(
    source: DepthState,
    cam_src: Dict[str, Any],
    cam_dst: Dict[str, Any],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    splat_radius: int,
    propagation_conf_decay: float,
    chunk_pixels: int,
    source_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_depth = source.depth
    src_conf = source.confidence
    valid = (
        np.isfinite(src_depth)
        & (src_depth >= min_depth)
        & (src_depth <= max_depth)
        & np.isfinite(src_conf)
        & (src_conf > 0.0)
    )

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return (
            np.zeros((height, width), dtype=np.float64),
            np.zeros((height, width), dtype=np.float32),
            np.full((height, width), -1, dtype=np.int16),
        )

    zs = src_depth[ys, xs].astype(np.float64, copy=False)
    cs = src_conf[ys, xs].astype(np.float64, copy=False) * propagation_conf_decay

    dst_depth = np.full((height, width), np.inf, dtype=np.float64)
    dst_conf = np.zeros((height, width), dtype=np.float32)
    dst_src = np.full((height, width), -1, dtype=np.int16)

    Ksrc = np.asarray(cam_src["K"], dtype=np.float64)
    Kdst = np.asarray(cam_dst["K"], dtype=np.float64)
    zsign_src = float(cam_src["z_sign"])
    zsign_dst = float(cam_dst["z_sign"])
    M = np.asarray(cam_dst["W2C"], dtype=np.float64) @ np.asarray(
        cam_src["C2W"], dtype=np.float64
    )
    R = M[:3, :3]
    t = M[:3, 3]

    offsets = [
        (dx, dy)
        for dy in range(-splat_radius, splat_radius + 1)
        for dx in range(-splat_radius, splat_radius + 1)
    ]

    for start in range(0, xs.size, chunk_pixels):
        end = min(start + chunk_pixels, xs.size)
        u = xs[start:end].astype(np.float64)
        v = ys[start:end].astype(np.float64)
        z = zs[start:end]
        c = cs[start:end]

        X = np.empty((3, end - start), dtype=np.float64)
        X[0] = z * (u - Ksrc[0, 2]) / Ksrc[0, 0]
        X[1] = z * (v - Ksrc[1, 2]) / Ksrc[1, 1]
        X[2] = z * zsign_src

        Xd = R @ X + t[:, None]
        zd = zsign_dst * Xd[2]
        front = np.isfinite(zd) & (zd >= min_depth) & (zd <= max_depth)
        if not np.any(front):
            continue

        ud = Kdst[0, 0] * Xd[0, front] / zd[front] + Kdst[0, 2]
        vd = Kdst[1, 1] * Xd[1, front] / zd[front] + Kdst[1, 2]
        zd = zd[front]
        c = c[front]

        base_x = np.rint(ud).astype(np.int64)
        base_y = np.rint(vd).astype(np.int64)

        for dx, dy in offsets:
            # Slightly reduce confidence for neighboring splat pixels.
            splat_penalty = 1.0 / (1.0 + 0.35 * (abs(dx) + abs(dy)))
            _update_zbuffer_chunk(
                dst_depth,
                dst_conf,
                dst_src,
                base_x + dx,
                base_y + dy,
                zd,
                c * splat_penalty,
                source_id,
            )

    invalid = ~np.isfinite(dst_depth)
    dst_depth[invalid] = 0.0
    dst_conf[invalid] = 0.0
    dst_src[invalid] = -1
    return dst_depth, dst_conf, dst_src


def fuse_depth_candidates(
    local_depth: np.ndarray,
    local_conf: np.ndarray,
    propagated: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    min_depth: float,
    max_depth: float,
    log_depth_threshold: float,
    local_confidence_boost: float,
    single_candidate_penalty: float,
    minimum_output_confidence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    depth_candidates = [local_depth]
    conf_candidates = [np.clip(local_conf * local_confidence_boost, 0.0, 1.0)]
    labels = [0]  # 0=local; positive values are propagation source IDs + 1.

    for depth, conf, source_label in propagated:
        depth_candidates.append(depth)
        conf_candidates.append(conf)
        labels.append(source_label + 1)

    D = np.stack(depth_candidates, axis=0).astype(np.float64, copy=False)
    C = np.stack(conf_candidates, axis=0).astype(np.float64, copy=False)
    valid = (
        np.isfinite(D)
        & (D >= min_depth)
        & (D <= max_depth)
        & np.isfinite(C)
        & (C > 0.0)
    )
    C = np.where(valid, C, 0.0)

    n_valid = np.sum(valid, axis=0)
    logD = np.zeros_like(D)
    logD[valid] = np.log(D[valid])

    support = C.copy()
    for i in range(D.shape[0]):
        if not np.any(valid[i]):
            continue
        for j in range(D.shape[0]):
            if i == j:
                continue
            pair = valid[i] & valid[j]
            if not np.any(pair):
                continue
            diff = np.abs(logD[i] - logD[j])
            agreement = np.exp(-np.square(diff / max(log_depth_threshold, 1e-8)))
            support[i] += np.where(pair, C[j] * agreement, 0.0)

    best_idx = np.argmax(support, axis=0)
    best_log = np.take_along_axis(logD, best_idx[None, ...], axis=0)[0]
    best_valid = n_valid > 0

    agree = valid & (
        np.abs(logD - best_log[None, ...]) <= max(log_depth_threshold, 1e-8)
    )
    weights = C * agree
    weight_sum = np.sum(weights, axis=0)

    invD = np.zeros_like(D)
    invD[valid] = 1.0 / D[valid]
    fused_inv = np.sum(weights * invD, axis=0) / np.maximum(weight_sum, 1e-12)

    out_depth = np.zeros(local_depth.shape, dtype=np.float64)
    usable = best_valid & np.isfinite(fused_inv) & (fused_inv > 0.0)
    out_depth[usable] = 1.0 / fused_inv[usable]

    agreeing_count = np.sum(agree, axis=0)
    max_conf = np.max(np.where(agree, C, 0.0), axis=0)
    mean_conf = weight_sum / np.maximum(agreeing_count, 1)
    out_conf = np.clip(0.55 * max_conf + 0.45 * mean_conf, 0.0, 1.0)
    single = agreeing_count <= 1
    out_conf[single] *= single_candidate_penalty

    usable &= out_conf >= minimum_output_confidence
    out_depth[~usable] = 0.0
    out_conf[~usable] = 0.0

    label_array = np.asarray(labels, dtype=np.int16)
    selected_label = label_array[best_idx]
    selected_label[~usable] = -1

    stats = {
        "fused_valid_pixels": int(np.count_nonzero(usable)),
        "multi_supported_pixels": int(np.count_nonzero(usable & (agreeing_count >= 2))),
        "single_supported_pixels": int(np.count_nonzero(usable & (agreeing_count == 1))),
        "selected_local_pixels": int(np.count_nonzero(usable & (selected_label == 0))),
        "selected_propagated_pixels": int(np.count_nonzero(usable & (selected_label > 0))),
    }
    return out_depth, out_conf.astype(np.float32), selected_label, stats


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


def write_confidence_yuv420p10le(
    output_path: str,
    confidence_frames: List[np.ndarray],
) -> None:
    h, w = confidence_frames[0].shape
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    with open(output_path, "wb") as f:
        for conf in confidence_frames:
            y = np.rint(np.clip(conf, 0.0, 1.0) * 1023.0).astype("<u2")
            f.write(np.ascontiguousarray(y).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())


def print_progress(
    decode_rank: int,
    num_frames: int,
    poc: int,
    valid_obs: int,
    local_ratio: float,
    final_ratio: float,
    num_sources: int,
) -> None:
    done = decode_rank + 1
    ratio = done / max(num_frames, 1)
    width = 28
    n = int(round(ratio * width))
    bar = "#" * n + "-" * (width - n)
    print(
        f"\r[{bar}] dec={done:3d}/{num_frames:3d} poc={poc:3d} "
        f"obs={valid_obs:6d} src={num_sources:2d} "
        f"local={local_ratio:7.3%} final={final_ratio:7.3%}",
        end="",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Causally propagate already decoded depth maps in decoder order and "
            "fuse them with the current left/top/top-left MV depth predictor."
        )
    )
    ap.add_argument("--mv-csv", required=True)
    ap.add_argument("--camera-param", required=True)
    ap.add_argument("--out-yuv", required=True)

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--fit-block", type=int, default=16)
    ap.add_argument("--neighborhood", type=int, default=0)
    ap.add_argument("--min-points", type=int, default=4)

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument("--min-parallax", type=float, default=1e-6)
    ap.add_argument("--max-reproj-error", type=float, default=1.5)

    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-condition", type=float, default=1e-8)
    ap.add_argument("--max-plane-slope", type=float, default=1.0)

    ap.add_argument(
        "--decode-order",
        default="ra",
        help=(
            "ra/auto for hierarchical RA order, display for 0..N-1, or an "
            "explicit comma-separated POC list."
        ),
    )
    ap.add_argument("--max-propagation-sources", type=int, default=4)
    ap.add_argument("--min-source-quality", type=float, default=0.01)
    ap.add_argument("--source-poc-distance-scale", type=float, default=16.0)
    ap.add_argument("--propagation-splat-radius", type=int, default=1)
    ap.add_argument("--propagation-chunk-pixels", type=int, default=262144)
    ap.add_argument(
        "--propagation-half-life",
        type=float,
        default=8.0,
        help="Confidence half-life measured in decoder-order picture distance.",
    )
    ap.add_argument(
        "--propagation-poc-half-life",
        type=float,
        default=32.0,
        help="Additional confidence half-life measured in absolute POC distance.",
    )
    ap.add_argument(
        "--depth-consistency-ratio",
        type=float,
        default=1.12,
        help="Candidates within this multiplicative depth ratio support fusion.",
    )
    ap.add_argument("--local-confidence-boost", type=float, default=1.10)
    ap.add_argument("--single-candidate-penalty", type=float, default=0.90)
    ap.add_argument("--minimum-output-confidence", type=float, default=0.01)

    ap.add_argument(
        "--depth-scale-real",
        type=float,
        default=None,
        help=(
            "Override output depth scale. Default: camera header "
            "depth_scale/depth_scale_precision."
        ),
    )
    ap.add_argument("--out-confidence-yuv", default=None)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu-batch-blocks", type=int, default=4096)

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.min_points < 3:
        raise ValueError("--min-points must be >= 3")
    if args.max_propagation_sources < 0:
        raise ValueError("--max-propagation-sources must be >= 0")
    if args.propagation_splat_radius < 0:
        raise ValueError("--propagation-splat-radius must be >= 0")
    if args.depth_consistency_ratio <= 1.0:
        raise ValueError("--depth-consistency-ratio must be > 1")

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
    decode_order = parse_decode_order(args.decode_order, args.num_frames)

    missing_cameras = [p for p in decode_order if p not in cameras]
    if missing_cameras:
        raise RuntimeError(f"Missing camera records for POCs: {missing_cameras}")

    if args.device == "auto":
        use_cuda = torch is not None and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
    else:
        device = args.device
        use_cuda = device.startswith("cuda")

    if use_cuda and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable")

    print(f"device                  : {device}")
    print(f"fit block               : {args.fit_block}x{args.fit_block}")
    print(f"decode order            : {decode_order}")
    print("current block MV        : disabled")
    print("propagation causality   : decoded pictures only")
    print(f"max propagation sources : {args.max_propagation_sources}")
    print(f"depth scale real        : {depth_scale_real:.12g}")

    depth_frames: List[Optional[np.ndarray]] = [None] * args.num_frames
    confidence_frames: List[Optional[np.ndarray]] = [None] * args.num_frames
    state_bank: Dict[int, DepthState] = {}
    frame_stats_by_poc: Dict[int, Dict[str, Any]] = {}

    log_depth_threshold = math.log(args.depth_consistency_ratio)

    for decode_rank, poc in enumerate(decode_order):
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
            neighborhood=args.neighborhood,
            min_points=args.min_points,
        )

        coeffs: List[Optional[np.ndarray]] = []
        if use_cuda and jobs:
            for start in range(0, len(jobs), args.gpu_batch_blocks):
                batch = jobs[start:start + args.gpu_batch_blocks]
                coeffs.extend(
                    fit_inv_depth_planes_gpu(
                        blocks=batch,
                        irls_iters=args.irls_iters,
                        huber_delta=args.huber_delta,
                        min_condition=args.min_condition,
                        device=device,
                    )
                )
        else:
            for job in jobs:
                coeffs.append(
                    fit_inv_depth_plane_cpu(
                        xs=job["xs"],
                        ys=job["ys"],
                        depths=job["depths"],
                        reproj_errors=job["errors"],
                        cx=job["cx"],
                        cy=job["cy"],
                        irls_iters=args.irls_iters,
                        huber_delta=args.huber_delta,
                        min_condition=args.min_condition,
                    )
                )

        local_depth, local_valid, local_conf = render_jobs(
            jobs=jobs,
            coeffs=coeffs,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_plane_slope=args.max_plane_slope,
            min_points=args.min_points,
        )
        local_ratio = float(np.mean(local_valid))

        selected_sources = select_propagation_sources(
            states=state_bank,
            target_poc=poc,
            target_mv_rows=mv_by_frame[poc],
            max_sources=args.max_propagation_sources,
            min_source_quality=args.min_source_quality,
            poc_distance_scale=args.source_poc_distance_scale,
        )

        propagated_candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []
        source_stats: List[Dict[str, Any]] = []

        for source_index, source in enumerate(selected_sources):
            decode_age = max(1, decode_rank - source.decode_rank)
            poc_distance = abs(poc - source.poc)
            decode_decay = math.exp(
                -math.log(2.0) * decode_age / max(args.propagation_half_life, 1e-6)
            )
            poc_decay = math.exp(
                -math.log(2.0) * poc_distance
                / max(args.propagation_poc_half_life, 1e-6)
            )
            decay = float(decode_decay * poc_decay)

            warp_depth, warp_conf, _ = forward_warp_depth(
                source=source,
                cam_src=cameras[source.poc],
                cam_dst=cameras[poc],
                width=args.width,
                height=args.height,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                splat_radius=args.propagation_splat_radius,
                propagation_conf_decay=decay,
                chunk_pixels=args.propagation_chunk_pixels,
                source_id=source_index,
            )
            warp_valid = warp_depth > 0.0
            propagated_candidates.append((warp_depth, warp_conf, source_index))
            source_stats.append(
                {
                    "source_poc": source.poc,
                    "source_decode_rank": source.decode_rank,
                    "source_quality": source.quality_score,
                    "decode_age": decode_age,
                    "poc_distance": poc_distance,
                    "confidence_decay": decay,
                    "warped_valid_ratio": float(np.mean(warp_valid)),
                    "warped_mean_confidence": (
                        float(np.mean(warp_conf[warp_valid]))
                        if np.any(warp_valid)
                        else 0.0
                    ),
                }
            )

        final_depth, final_conf, selected_label, fusion_stats = fuse_depth_candidates(
            local_depth=local_depth,
            local_conf=local_conf,
            propagated=propagated_candidates,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            log_depth_threshold=log_depth_threshold,
            local_confidence_boost=args.local_confidence_boost,
            single_candidate_penalty=args.single_candidate_penalty,
            minimum_output_confidence=args.minimum_output_confidence,
        )

        final_valid = final_depth > 0.0
        final_ratio = float(np.mean(final_valid))
        mean_conf = float(np.mean(final_conf[final_valid])) if np.any(final_valid) else 0.0
        quality_score = final_ratio * mean_conf
        propagated_selected = final_valid & (selected_label > 0)
        propagated_valid_ratio = float(np.mean(propagated_selected))

        state = DepthState(
            poc=poc,
            decode_rank=decode_rank,
            depth=final_depth,
            confidence=final_conf,
            valid_ratio=final_ratio,
            mean_confidence=mean_conf,
            quality_score=quality_score,
            local_valid_ratio=local_ratio,
            propagated_valid_ratio=propagated_valid_ratio,
        )
        state_bank[poc] = state
        depth_frames[poc] = final_depth
        confidence_frames[poc] = final_conf

        frame_stats_by_poc[poc] = {
            "poc": poc,
            "decode_rank": decode_rank,
            "mv_rows": len(mv_by_frame[poc]),
            "valid_depth_observations": len(observations),
            "fit_jobs": len(jobs),
            "successful_planes": sum(c is not None for c in coeffs),
            "local_valid_pixel_ratio": local_ratio,
            "final_valid_pixel_ratio": final_ratio,
            "mean_output_confidence": mean_conf,
            "quality_score": quality_score,
            "selected_source_pocs": [s.poc for s in selected_sources],
            "propagation_sources": source_stats,
            "fusion": fusion_stats,
        }

        print_progress(
            decode_rank=decode_rank,
            num_frames=args.num_frames,
            poc=poc,
            valid_obs=len(observations),
            local_ratio=local_ratio,
            final_ratio=final_ratio,
            num_sources=len(selected_sources),
        )

    print()

    zero_depth = np.zeros((args.height, args.width), dtype=np.float64)
    zero_conf = np.zeros((args.height, args.width), dtype=np.float32)
    final_depth_frames = [d if d is not None else zero_depth for d in depth_frames]
    final_conf_frames = [c if c is not None else zero_conf for c in confidence_frames]

    write_depth_yuv420p10le(
        output_path=args.out_yuv,
        depth_frames=final_depth_frames,
        depth_scale_real=depth_scale_real,
    )

    if args.out_confidence_yuv:
        write_confidence_yuv420p10le(
            output_path=args.out_confidence_yuv,
            confidence_frames=final_conf_frames,
        )

    stats_path = str(Path(args.out_yuv).with_suffix(".stats.json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mv_csv": args.mv_csv,
                "camera_param": args.camera_param,
                "out_yuv": args.out_yuv,
                "out_confidence_yuv": args.out_confidence_yuv,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "fit_block": args.fit_block,
                "decode_order": decode_order,
                "predictor_neighbors": ["left", "top", "top_left"],
                "current_block_mv_used": False,
                "propagation_causality": "already-decoded-depth-only",
                "max_propagation_sources": args.max_propagation_sources,
                "depth_consistency_ratio": args.depth_consistency_ratio,
                "depth_scale_real": depth_scale_real,
                "device": device,
                "frames_decode_order": [frame_stats_by_poc[p] for p in decode_order],
                "frames_poc_order": [frame_stats_by_poc[p] for p in range(args.num_frames)],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Depth YUV      : {args.out_yuv}")
    if args.out_confidence_yuv:
        print(f"Confidence YUV : {args.out_confidence_yuv}")
    print(f"Stats          : {stats_path}")


if __name__ == "__main__":
    main()
