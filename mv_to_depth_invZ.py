#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MV + camera parameter -> causal inverse-depth-plane predictor YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Main behavior
-------------
1) Each MV observation is converted directly to inverse depth rho=1/z.
   The solver does not estimate z first, which is more stable for small motion.
2) True depth observability is measured from the translation-dependent term.
   Low-observability samples are rejected or down-weighted.
3) Rotation-compensated parallax and reprojection error are recorded.
4) L0/L1 observations at the same spatial position are checked for inverse-depth
   consistency. Strongly inconsistent bi-pred observations are rejected.
5) The current fit-block never uses its own MV observations. Only the left, top,
   and top-left fit-blocks are used.
6) Neighbor source groups are deterministically filtered for mutual consistency
   using decoder-available geometry only; original-picture distortion is never used.
7) A robust inverse-depth plane
       rho(x,y) = a*(x-cx) + b*(y-cy) + c
   is fitted with normalized coordinates and IRLS.
8) Observability is included in the fitting weights.
9) If a full plane is geometrically unstable, an optional C-only fallback
   (a=b=0, robust c) is used. If even that is unreliable, the block remains zero.
10) Optional CUDA acceleration is used for batched final plane fitting.

Camera JSONL format:
  camparam_v2_vggt_or_canonical
  pose_mode: current_to_previous, gop_local, or absolute

Depth output:
  H x W YUV420p10le for each frame.
  Invalid depth remains zero. UV planes are fixed to 512.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}") from exc

            if obj.get("type") in ("header", "intrinsic"):
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

    pose_mode = str(header.get("pose_mode", "current_to_previous"))
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
            rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64
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
# MV observation -> inverse depth
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
class InvDepthSolveResult:
    inv_depth: float
    depth: float
    reproj_error: float
    observability: float
    parallax_px: float


@dataclass(frozen=True)
class InvDepthObservation:
    poc: int
    x: float
    y: float
    block_x: int
    block_y: int
    block_w: int
    block_h: int
    inv_depth: float
    reproj_error: float
    observability: float
    parallax_px: float
    ref_poc: int
    list_id: str

    @property
    def spatial_key(self) -> Tuple[int, int, int, int, int]:
        return (self.poc, self.block_x, self.block_y, self.block_w, self.block_h)


def canonical_list_id(value: str) -> str:
    s = str(value).strip().upper().replace("_", "")
    if s in ("0", "L0", "LIST0", "REFPICLIST0"):
        return "L0"
    if s in ("1", "L1", "LIST1", "REFPICLIST1"):
        return "L1"
    return s


def parse_mv_csv(path: str, num_frames: int) -> List[List[MVObservation]]:
    by_frame: List[List[MVObservation]] = [[] for _ in range(num_frames)]
    required = {
        "poc",
        "x",
        "y",
        "w",
        "h",
        "list",
        "ref_poc",
        "mv_x",
        "mv_y",
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

                obs = MVObservation(
                    poc=poc,
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                    list_id=canonical_list_id(row["list"]),
                    ref_poc=int(row["ref_poc"]),
                    mv_x=float(row["mv_x"]),
                    mv_y=float(row["mv_y"]),
                )
                by_frame[poc].append(obs)
            except Exception as exc:
                raise RuntimeError(f"Bad CSV row {row_no}: {row}") from exc

    return by_frame


def pixel_ray(u: float, v: float, cam: Dict[str, Any]) -> np.ndarray:
    K = np.asarray(cam["K"], dtype=np.float64)
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
    K = np.asarray(cam["K"], dtype=np.float64)
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
    cam_cur: Dict[str, Any], cam_ref: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    # X_ref = R * X_cur + t
    M = np.asarray(cam_ref["W2C"], dtype=np.float64) @ np.asarray(
        cam_cur["C2W"], dtype=np.float64
    )
    return M[:3, :3], M[:3, 3]


def solve_inv_depth_closed_form(
    u: float,
    v: float,
    mv_x: float,
    mv_y: float,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
    min_depth: float,
    max_depth: float,
    min_observability: float,
    min_parallax_px: float,
    max_reproj_error: float,
) -> Optional[InvDepthSolveResult]:
    """Solve inverse depth directly from one current/reference correspondence.

    Current pixel p=(u,v), reference match p'=(u+mv_x,v+mv_y).

      X_cur(z) = z * ray_cur
      X_ref(z) = z*q + t

    The cross-multiplied projection constraints have the form

      A*z = B.

    With rho=1/z this becomes

      A = B*rho,

    so the least-squares inverse-depth estimate is

      rho = (B^T A) / (B^T B).

    B^T B is translation-dependent. It is therefore used as an observability
    measure rather than A^T A, which can be small simply because the true
    inverse depth is close to zero.
    """
    ur = float(u + mv_x)
    vr = float(v + mv_y)

    ray = pixel_ray(u, v, cam_cur)
    R, t = relative_transform(cam_cur, cam_ref)
    q = R @ ray

    K = np.asarray(cam_ref["K"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z_sign_ref = float(cam_ref["z_sign"])

    du = ur - cx
    dv = vr - cy

    A = np.array(
        [
            du * z_sign_ref * q[2] - fx * q[0],
            dv * z_sign_ref * q[2] - fy * q[1],
        ],
        dtype=np.float64,
    )
    B = np.array(
        [
            fx * t[0] - du * z_sign_ref * t[2],
            fy * t[1] - dv * z_sign_ref * t[2],
        ],
        dtype=np.float64,
    )

    observability = float(B @ B)
    if (
        not np.isfinite(observability)
        or observability < float(min_observability) ** 2
    ):
        return None

    inv_depth = float((B @ A) / observability)
    min_inv_depth = 1.0 / float(max_depth)
    max_inv_depth = 1.0 / float(min_depth)
    if (
        not np.isfinite(inv_depth)
        or inv_depth < min_inv_depth
        or inv_depth > max_inv_depth
    ):
        return None

    depth = 1.0 / inv_depth

    # Rotation-only projection. The difference from the observed match is the
    # translation-induced image parallax that carries depth information.
    rotation_only = project_point(q, cam_ref)
    if rotation_only is None:
        return None
    observed = np.array([ur, vr], dtype=np.float64)
    parallax_px = float(np.linalg.norm(observed - rotation_only))
    if not np.isfinite(parallax_px) or parallax_px < min_parallax_px:
        return None

    X_ref = depth * q + t
    pred = project_point(X_ref, cam_ref)
    if pred is None:
        return None

    reproj_error = float(np.linalg.norm(pred - observed))
    if not np.isfinite(reproj_error) or reproj_error > max_reproj_error:
        return None

    return InvDepthSolveResult(
        inv_depth=inv_depth,
        depth=depth,
        reproj_error=reproj_error,
        observability=observability,
        parallax_px=parallax_px,
    )


def confidence_weights(
    observabilities: np.ndarray,
    reproj_errors: np.ndarray,
    observability_weight_cap: float,
) -> np.ndarray:
    """Dimensionless per-job confidence weights.

    Observability is normalized by its positive median so that camera units do
    not directly set the fitting scale. High-observability samples are capped
    to prevent one reference baseline from completely dominating the fit.
    """
    obs = np.asarray(observabilities, dtype=np.float64)
    err = np.asarray(reproj_errors, dtype=np.float64)
    positive = obs[np.isfinite(obs) & (obs > 0.0)]
    if positive.size == 0:
        return np.zeros_like(obs)

    obs_ref = max(float(np.median(positive)), 1e-20)
    obs_weight = np.clip(obs / obs_ref, 0.0, observability_weight_cap)
    return obs_weight / np.maximum(1.0 + err * err, 1e-12)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(ok):
        return None
    sw = float(np.sum(weights[ok]))
    if sw <= 1e-20:
        return None
    return float(np.sum(values[ok] * weights[ok]) / sw)


def apply_bipred_consistency_filter(
    observations: List[InvDepthObservation],
    enabled: bool,
    rel_threshold: float,
    abs_threshold: float,
) -> Tuple[List[InvDepthObservation], Dict[str, int]]:
    stats = {
        "bipred_checked_locations": 0,
        "bipred_rejected_locations": 0,
        "bipred_rejected_observations": 0,
    }
    if not enabled or not observations:
        return observations, stats

    groups: Dict[
        Tuple[int, int, int, int, int], List[InvDepthObservation]
    ] = {}
    for obs in observations:
        groups.setdefault(obs.spatial_key, []).append(obs)

    rejected_keys = set()
    for key, vals in groups.items():
        l0 = [o for o in vals if canonical_list_id(o.list_id) == "L0"]
        l1 = [o for o in vals if canonical_list_id(o.list_id) == "L1"]
        if not l0 or not l1:
            continue

        stats["bipred_checked_locations"] += 1

        def list_estimate(items: Sequence[InvDepthObservation]) -> Optional[float]:
            inv = np.asarray([o.inv_depth for o in items], dtype=np.float64)
            obs = np.asarray([o.observability for o in items], dtype=np.float64)
            err = np.asarray([o.reproj_error for o in items], dtype=np.float64)
            # Raw observability is meaningful here because alternatives from the
            # same list/location are competing measurements of the same rho.
            w = obs / np.maximum(1.0 + err * err, 1e-12)
            return weighted_mean(inv, w)

        rho0 = list_estimate(l0)
        rho1 = list_estimate(l1)
        if rho0 is None or rho1 is None:
            continue

        tolerance = abs_threshold + rel_threshold * max(
            abs(rho0), abs(rho1), 1e-20
        )
        if abs(rho0 - rho1) > tolerance:
            rejected_keys.add(key)
            stats["bipred_rejected_locations"] += 1
            stats["bipred_rejected_observations"] += len(vals)

    if not rejected_keys:
        return observations, stats

    filtered = [o for o in observations if o.spatial_key not in rejected_keys]
    return filtered, stats


def make_inv_depth_observations(
    mv_rows: List[MVObservation],
    cameras: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    min_observability: float,
    min_parallax_px: float,
    max_reproj_error: float,
    bipred_consistency: bool,
    bipred_rel_threshold: float,
    bipred_abs_threshold: float,
) -> Tuple[List[InvDepthObservation], Dict[str, int]]:
    raw: List[InvDepthObservation] = []
    stats = {
        "mv_rows": len(mv_rows),
        "missing_camera": 0,
        "out_of_bounds": 0,
        "solve_rejected": 0,
        "accepted_before_bipred": 0,
        "accepted": 0,
    }

    for row in mv_rows:
        if row.poc not in cameras or row.ref_poc not in cameras:
            stats["missing_camera"] += 1
            continue

        cx = row.x + (row.w - 1) * 0.5
        cy = row.y + (row.h - 1) * 0.5
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            stats["out_of_bounds"] += 1
            continue

        solved = solve_inv_depth_closed_form(
            u=cx,
            v=cy,
            mv_x=row.mv_x,
            mv_y=row.mv_y,
            cam_cur=cameras[row.poc],
            cam_ref=cameras[row.ref_poc],
            min_depth=min_depth,
            max_depth=max_depth,
            min_observability=min_observability,
            min_parallax_px=min_parallax_px,
            max_reproj_error=max_reproj_error,
        )
        if solved is None:
            stats["solve_rejected"] += 1
            continue

        raw.append(
            InvDepthObservation(
                poc=row.poc,
                x=cx,
                y=cy,
                block_x=row.x,
                block_y=row.y,
                block_w=row.w,
                block_h=row.h,
                inv_depth=solved.inv_depth,
                reproj_error=solved.reproj_error,
                observability=solved.observability,
                parallax_px=solved.parallax_px,
                ref_poc=row.ref_poc,
                list_id=canonical_list_id(row.list_id),
            )
        )

    stats["accepted_before_bipred"] = len(raw)
    filtered, bipred_stats = apply_bipred_consistency_filter(
        raw,
        enabled=bipred_consistency,
        rel_threshold=bipred_rel_threshold,
        abs_threshold=bipred_abs_threshold,
    )
    stats.update(bipred_stats)
    stats["accepted"] = len(filtered)
    return filtered, stats


# ============================================================
# Robust inverse-depth plane fitting
# ============================================================


@dataclass
class PlaneFitResult:
    coeff: np.ndarray  # [a, b, c], a/b are per-pixel slopes
    mode: str  # "full" or "c_only"
    condition_ratio: float
    effective_weight: float


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def robust_c_only_fit(
    values: np.ndarray,
    base_weights: np.ndarray,
    irls_iters: int,
    huber_delta: float,
) -> Optional[Tuple[float, float]]:
    y = np.asarray(values, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    c = weighted_mean(y, base)
    if c is None:
        return None

    weights = base.copy()
    for _ in range(max(1, irls_iters)):
        c_new = weighted_mean(y, weights)
        if c_new is None:
            return None
        c = c_new
        residual = y - c
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), 1e-10)
        weights = base * huber_weights(residual / scale, huber_delta)

    effective_weight = float(np.sum(weights[np.isfinite(weights)]))
    if not np.isfinite(c) or effective_weight <= 1e-12:
        return None
    return float(c), effective_weight


def fit_inv_depth_plane_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    inv_depths: np.ndarray,
    reproj_errors: np.ndarray,
    observabilities: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    allow_full: bool,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    observability_weight_cap: float,
    c_only_fallback: bool,
) -> Optional[PlaneFitResult]:
    if inv_depths.size < 3:
        return None

    xn = (np.asarray(xs, dtype=np.float64) - cx) / sx
    yn = (np.asarray(ys, dtype=np.float64) - cy) / sy
    y = np.asarray(inv_depths, dtype=np.float64)
    A = np.stack([xn, yn, np.ones_like(xn)], axis=1)

    base_weights = confidence_weights(
        observabilities,
        reproj_errors,
        observability_weight_cap,
    )

    last_condition = 0.0
    if allow_full:
        weights = base_weights.copy()
        coeff_norm: Optional[np.ndarray] = None
        stable = True

        for _ in range(max(1, irls_iters)):
            sw = np.sqrt(np.maximum(weights, 1e-12))
            Aw = A * sw[:, None]
            bw = y * sw

            normal = Aw.T @ Aw
            eig = np.linalg.eigvalsh(normal)
            if eig[-1] <= 1e-15:
                stable = False
                break
            last_condition = float(eig[0] / eig[-1])
            if last_condition < min_condition:
                stable = False
                break

            try:
                coeff_norm = np.linalg.solve(normal, Aw.T @ bw)
            except np.linalg.LinAlgError:
                stable = False
                break

            residual = y - A @ coeff_norm
            scale = 1.4826 * np.median(
                np.abs(residual - np.median(residual))
            )
            scale = max(float(scale), 1e-10)
            robust = huber_weights(residual / scale, huber_delta)
            weights = base_weights * robust

        if (
            stable
            and coeff_norm is not None
            and np.isfinite(coeff_norm).all()
        ):
            coeff_pixel = np.array(
                [coeff_norm[0] / sx, coeff_norm[1] / sy, coeff_norm[2]],
                dtype=np.float64,
            )
            return PlaneFitResult(
                coeff=coeff_pixel,
                mode="full",
                condition_ratio=last_condition,
                effective_weight=float(np.sum(weights)),
            )

    if not c_only_fallback:
        return None

    cfit = robust_c_only_fit(
        values=y,
        base_weights=base_weights,
        irls_iters=irls_iters,
        huber_delta=huber_delta,
    )
    if cfit is None:
        return None

    c, effective_weight = cfit
    return PlaneFitResult(
        coeff=np.array([0.0, 0.0, c], dtype=np.float64),
        mode="c_only",
        condition_ratio=last_condition,
        effective_weight=effective_weight,
    )


def fit_inv_depth_planes_gpu(
    blocks: List[Dict[str, Any]],
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    observability_weight_cap: float,
    c_only_fallback: bool,
    device: str,
) -> List[Optional[PlaneFitResult]]:
    """Batched padded IRLS with normalized coordinates and C-only fallback."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if not blocks:
        return []

    dev = torch.device(device)
    batch_size = len(blocks)
    max_n = max(len(b["inv_depths"]) for b in blocks)

    A = torch.zeros((batch_size, max_n, 3), dtype=torch.float64, device=dev)
    y = torch.zeros((batch_size, max_n), dtype=torch.float64, device=dev)
    obs = torch.zeros((batch_size, max_n), dtype=torch.float64, device=dev)
    err = torch.zeros((batch_size, max_n), dtype=torch.float64, device=dev)
    mask = torch.zeros((batch_size, max_n), dtype=torch.bool, device=dev)
    sx_tensor = torch.ones(batch_size, dtype=torch.float64, device=dev)
    sy_tensor = torch.ones(batch_size, dtype=torch.float64, device=dev)
    allow_full = torch.zeros(batch_size, dtype=torch.bool, device=dev)

    for i, block in enumerate(blocks):
        n = len(block["inv_depths"])
        xs = torch.as_tensor(block["xs"], dtype=torch.float64, device=dev)
        ys = torch.as_tensor(block["ys"], dtype=torch.float64, device=dev)
        sx = float(block["sx"])
        sy = float(block["sy"])

        A[i, :n, 0] = (xs - float(block["cx"])) / sx
        A[i, :n, 1] = (ys - float(block["cy"])) / sy
        A[i, :n, 2] = 1.0
        y[i, :n] = torch.as_tensor(
            block["inv_depths"], dtype=torch.float64, device=dev
        )
        obs[i, :n] = torch.as_tensor(
            block["observabilities"], dtype=torch.float64, device=dev
        )
        err[i, :n] = torch.as_tensor(
            block["errors"], dtype=torch.float64, device=dev
        )
        mask[i, :n] = True
        sx_tensor[i] = sx
        sy_tensor[i] = sy
        allow_full[i] = bool(block["allow_full"])

    obs_masked = torch.where(mask & (obs > 0.0), obs, torch.nan)
    obs_ref = torch.nanmedian(obs_masked, dim=1).values
    obs_ref = torch.clamp(obs_ref, min=1e-20)
    obs_weight = torch.clamp(
        obs / obs_ref[:, None],
        min=0.0,
        max=observability_weight_cap,
    )
    base_w = (
        obs_weight
        / torch.clamp(1.0 + err * err, min=1e-12)
        * mask.to(torch.float64)
    )

    # Full-plane fit.
    weights = base_w.clone()
    coeff_norm = torch.zeros((batch_size, 3), dtype=torch.float64, device=dev)
    full_valid = allow_full.clone()
    condition_ratio = torch.zeros(batch_size, dtype=torch.float64, device=dev)
    eye = torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0)

    for _ in range(max(1, irls_iters)):
        AtW = A.transpose(1, 2) * weights.unsqueeze(1)
        normal = AtW @ A
        rhs = (AtW @ y.unsqueeze(-1)).squeeze(-1)

        eig = torch.linalg.eigvalsh(normal)
        ratio = eig[:, 0] / torch.clamp(eig[:, -1], min=1e-15)
        stable = (eig[:, -1] > 1e-15) & (ratio >= min_condition)
        full_valid &= stable
        condition_ratio = ratio

        normal_safe = normal + (~stable).to(torch.float64).view(-1, 1, 1) * eye
        coeff_norm = torch.linalg.solve(
            normal_safe, rhs.unsqueeze(-1)
        ).squeeze(-1)

        residual = y - torch.sum(A * coeff_norm[:, None, :], dim=2)
        residual_masked = torch.where(mask, residual, torch.nan)
        med = torch.nanmedian(residual_masked, dim=1).values
        mad = torch.nanmedian(
            torch.abs(residual_masked - med[:, None]), dim=1
        ).values
        scale = torch.clamp(1.4826 * mad, min=1e-10)
        r = torch.abs(residual) / scale[:, None]
        robust = torch.where(
            r <= huber_delta,
            torch.ones_like(r),
            huber_delta / torch.clamp(r, min=1e-12),
        )
        weights = base_w * robust * mask.to(torch.float64)

    full_valid &= torch.isfinite(coeff_norm).all(dim=1)

    # C-only fit is computed for every block so it can be used as a fallback.
    c_weights = base_w.clone()
    c = torch.sum(c_weights * y, dim=1) / torch.clamp(
        torch.sum(c_weights, dim=1), min=1e-20
    )
    for _ in range(max(1, irls_iters)):
        c = torch.sum(c_weights * y, dim=1) / torch.clamp(
            torch.sum(c_weights, dim=1), min=1e-20
        )
        residual = y - c[:, None]
        residual_masked = torch.where(mask, residual, torch.nan)
        med = torch.nanmedian(residual_masked, dim=1).values
        mad = torch.nanmedian(
            torch.abs(residual_masked - med[:, None]), dim=1
        ).values
        scale = torch.clamp(1.4826 * mad, min=1e-10)
        r = torch.abs(residual) / scale[:, None]
        robust = torch.where(
            r <= huber_delta,
            torch.ones_like(r),
            huber_delta / torch.clamp(r, min=1e-12),
        )
        c_weights = base_w * robust * mask.to(torch.float64)

    c_effective = torch.sum(c_weights, dim=1)
    c_valid = (
        torch.isfinite(c)
        & (c_effective > 1e-12)
        & (c > 0.0)
    )

    coeff_np = coeff_norm.detach().cpu().numpy()
    full_valid_np = full_valid.detach().cpu().numpy()
    ratio_np = condition_ratio.detach().cpu().numpy()
    full_weight_np = torch.sum(weights, dim=1).detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()
    c_valid_np = c_valid.detach().cpu().numpy()
    c_weight_np = c_effective.detach().cpu().numpy()
    sx_np = sx_tensor.detach().cpu().numpy()
    sy_np = sy_tensor.detach().cpu().numpy()

    out: List[Optional[PlaneFitResult]] = []
    for i in range(batch_size):
        if full_valid_np[i]:
            coeff_pixel = np.array(
                [
                    coeff_np[i, 0] / sx_np[i],
                    coeff_np[i, 1] / sy_np[i],
                    coeff_np[i, 2],
                ],
                dtype=np.float64,
            )
            out.append(
                PlaneFitResult(
                    coeff=coeff_pixel,
                    mode="full",
                    condition_ratio=float(ratio_np[i]),
                    effective_weight=float(full_weight_np[i]),
                )
            )
        elif c_only_fallback and c_valid_np[i]:
            out.append(
                PlaneFitResult(
                    coeff=np.array([0.0, 0.0, c_np[i]], dtype=np.float64),
                    mode="c_only",
                    condition_ratio=float(ratio_np[i]),
                    effective_weight=float(c_weight_np[i]),
                )
            )
        else:
            out.append(None)

    return out


# ============================================================
# Causal source selection and job construction
# ============================================================


def estimate_group_center_inv_depth(
    observations: Sequence[InvDepthObservation],
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    min_condition: float,
    observability_weight_cap: float,
    irls_iters: int,
    huber_delta: float,
) -> Optional[Tuple[float, float]]:
    """Estimate a source group's inverse depth at the current block center.

    This is used only for deterministic source-group consistency filtering.
    It never uses original-picture distortion.
    """
    if len(observations) < 3:
        return None

    xs = np.asarray([o.x for o in observations], dtype=np.float64)
    ys = np.asarray([o.y for o in observations], dtype=np.float64)
    inv = np.asarray([o.inv_depth for o in observations], dtype=np.float64)
    err = np.asarray([o.reproj_error for o in observations], dtype=np.float64)
    obs = np.asarray([o.observability for o in observations], dtype=np.float64)

    A = np.stack(
        [(xs - cx) / sx, (ys - cy) / sy, np.ones_like(xs)], axis=1
    )
    base = confidence_weights(obs, err, observability_weight_cap)
    weights = base.copy()
    coeff = None
    condition = 0.0

    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-12))
        Aw = A * sw[:, None]
        bw = inv * sw
        normal = Aw.T @ Aw
        eig = np.linalg.eigvalsh(normal)
        if eig[-1] <= 1e-15:
            return None
        condition = float(eig[0] / eig[-1])
        if condition < min_condition:
            return None
        try:
            coeff = np.linalg.solve(normal, Aw.T @ bw)
        except np.linalg.LinAlgError:
            return None

        residual = inv - A @ coeff
        scale = 1.4826 * np.median(
            np.abs(residual - np.median(residual))
        )
        scale = max(float(scale), 1e-10)
        weights = base * huber_weights(residual / scale, huber_delta)

    if coeff is None or not np.isfinite(coeff).all():
        return None
    # Because the design matrix is centered at the current block, coeff[2]
    # directly predicts inverse depth at that center.
    reliability = float(np.sum(weights)) * max(condition, 1e-20)
    return float(coeff[2]), reliability


def select_consistent_source_groups(
    groups: Dict[str, List[InvDepthObservation]],
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    enabled: bool,
    rel_threshold: float,
    abs_threshold: float,
    min_group_points: int,
    min_condition: float,
    observability_weight_cap: float,
    irls_iters: int,
    huber_delta: float,
) -> Tuple[List[str], Dict[str, Any]]:
    order = ["left", "top", "top_left"]
    nonempty = [name for name in order if groups.get(name)]
    info: Dict[str, Any] = {
        "available_groups": nonempty,
        "selected_groups": nonempty,
        "group_center_inv_depth": {},
    }
    if not enabled or len(nonempty) <= 1:
        return nonempty, info

    summaries: Dict[str, Tuple[float, float]] = {}
    for name in nonempty:
        vals = groups[name]
        if len(vals) < min_group_points:
            continue
        estimate = estimate_group_center_inv_depth(
            vals,
            cx=cx,
            cy=cy,
            sx=sx,
            sy=sy,
            min_condition=min_condition,
            observability_weight_cap=observability_weight_cap,
            irls_iters=irls_iters,
            huber_delta=huber_delta,
        )
        if estimate is not None:
            summaries[name] = estimate
            info["group_center_inv_depth"][name] = float(estimate[0])

    if not summaries:
        return nonempty, info
    if len(summaries) == 1:
        selected = [next(iter(summaries.keys()))]
        info["selected_groups"] = selected
        return selected, info

    best_cluster: Optional[List[str]] = None
    best_score = -1.0
    best_anchor_idx = len(order)

    for anchor_idx, anchor in enumerate(order):
        if anchor not in summaries:
            continue
        rho_anchor = summaries[anchor][0]
        cluster: List[str] = []
        score = 0.0
        for name in order:
            if name not in summaries:
                continue
            rho, reliability = summaries[name]
            tolerance = abs_threshold + rel_threshold * max(
                abs(rho_anchor), abs(rho), 1e-20
            )
            if abs(rho - rho_anchor) <= tolerance:
                cluster.append(name)
                score += reliability

        # Fixed source order is the deterministic tie-breaker.
        if (
            score > best_score + 1e-20
            or (
                abs(score - best_score) <= 1e-20
                and anchor_idx < best_anchor_idx
            )
        ):
            best_cluster = cluster
            best_score = score
            best_anchor_idx = anchor_idx

    selected = best_cluster if best_cluster else [next(iter(summaries.keys()))]
    info["selected_groups"] = selected
    return selected, info


def build_block_fit_jobs(
    observations: List[InvDepthObservation],
    width: int,
    height: int,
    fit_block: int,
    neighborhood: int,
    min_points: int,
    min_normalized_span: float,
    source_consistency: bool,
    source_consistency_rel_threshold: float,
    source_consistency_abs_threshold: float,
    source_consistency_min_points: int,
    min_condition: float,
    observability_weight_cap: float,
    irls_iters: int,
    huber_delta: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build causal jobs from left/top/top-left fit-blocks only."""
    del neighborhood

    stats = {
        "target_blocks": 0,
        "blocks_without_neighbor_observations": 0,
        "blocks_below_min_points": 0,
        "source_consistency_applied": 0,
        "source_groups_dropped": 0,
        "full_plane_geometry_allowed": 0,
        "full_plane_geometry_disallowed": 0,
        "fit_jobs": 0,
    }
    if not observations:
        return [], stats

    block_obs: Dict[Tuple[int, int], List[InvDepthObservation]] = {}
    for obs in observations:
        gx = int(obs.x) // fit_block
        gy = int(obs.y) // fit_block
        block_obs.setdefault((gx, gy), []).append(obs)

    jobs: List[Dict[str, Any]] = []
    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block

    for gy in range(grid_h):
        by = gy * fit_block
        bh = min(fit_block, height - by)
        cy = by + (bh - 1) * 0.5

        for gx in range(grid_w):
            stats["target_blocks"] += 1
            bx = gx * fit_block
            bw = min(fit_block, width - bx)
            cx = bx + (bw - 1) * 0.5
            sx = max(bw * 0.5, 1.0)
            sy = max(bh * 0.5, 1.0)

            groups: Dict[str, List[InvDepthObservation]] = {
                "left": block_obs.get((gx - 1, gy), []) if gx > 0 else [],
                "top": block_obs.get((gx, gy - 1), []) if gy > 0 else [],
                "top_left": (
                    block_obs.get((gx - 1, gy - 1), [])
                    if gx > 0 and gy > 0
                    else []
                ),
            }

            available_groups = [name for name, vals in groups.items() if vals]
            if not available_groups:
                stats["blocks_without_neighbor_observations"] += 1
                continue

            selected_groups, source_info = select_consistent_source_groups(
                groups=groups,
                cx=cx,
                cy=cy,
                sx=sx,
                sy=sy,
                enabled=source_consistency,
                rel_threshold=source_consistency_rel_threshold,
                abs_threshold=source_consistency_abs_threshold,
                min_group_points=source_consistency_min_points,
                min_condition=min_condition,
                observability_weight_cap=observability_weight_cap,
                irls_iters=irls_iters,
                huber_delta=huber_delta,
            )

            if source_consistency and len(available_groups) > 1:
                stats["source_consistency_applied"] += 1
            stats["source_groups_dropped"] += max(
                0, len(available_groups) - len(selected_groups)
            )

            selected: List[InvDepthObservation] = []
            for name in selected_groups:
                selected.extend(groups[name])

            if len(selected) < min_points:
                stats["blocks_below_min_points"] += 1
                continue

            xs = np.asarray([o.x for o in selected], dtype=np.float64)
            ys = np.asarray([o.y for o in selected], dtype=np.float64)
            inv_depths = np.asarray(
                [o.inv_depth for o in selected], dtype=np.float64
            )
            errors = np.asarray(
                [o.reproj_error for o in selected], dtype=np.float64
            )
            observabilities = np.asarray(
                [o.observability for o in selected], dtype=np.float64
            )
            parallaxes = np.asarray(
                [o.parallax_px for o in selected], dtype=np.float64
            )

            xn = (xs - cx) / sx
            yn = (ys - cy) / sy
            x_span = float(np.ptp(xn)) if xn.size else 0.0
            y_span = float(np.ptp(yn)) if yn.size else 0.0
            allow_full = (
                x_span >= min_normalized_span
                and y_span >= min_normalized_span
            )
            if allow_full:
                stats["full_plane_geometry_allowed"] += 1
            else:
                stats["full_plane_geometry_disallowed"] += 1

            jobs.append(
                {
                    "bx": bx,
                    "by": by,
                    "bw": bw,
                    "bh": bh,
                    "cx": cx,
                    "cy": cy,
                    "sx": sx,
                    "sy": sy,
                    "xs": xs,
                    "ys": ys,
                    "inv_depths": inv_depths,
                    "errors": errors,
                    "observabilities": observabilities,
                    "parallaxes": parallaxes,
                    "allow_full": allow_full,
                    "selected_source_groups": selected_groups,
                    "source_info": source_info,
                    "source_left_points": len(groups["left"]),
                    "source_top_points": len(groups["top"]),
                    "source_top_left_points": len(groups["top_left"]),
                    "selected_points": len(selected),
                    "x_span_normalized": x_span,
                    "y_span_normalized": y_span,
                }
            )

    stats["fit_jobs"] = len(jobs)
    return jobs, stats


# ============================================================
# Rendering and output
# ============================================================


def render_jobs(
    jobs: List[Dict[str, Any]],
    fits: List[Optional[PlaneFitResult]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    depth = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)
    stats = {
        "full_planes": 0,
        "c_only_planes": 0,
        "failed_planes": 0,
        "slope_rejected_planes": 0,
        "nonpositive_center_rejected_planes": 0,
    }

    for job, fit in zip(jobs, fits):
        if fit is None:
            stats["failed_planes"] += 1
            continue

        a, b, c = [float(v) for v in fit.coeff]
        if abs(a) > max_plane_slope or abs(b) > max_plane_slope:
            stats["slope_rejected_planes"] += 1
            continue
        if not np.isfinite(c) or c <= 0.0:
            stats["nonpositive_center_rejected_planes"] += 1
            continue

        if fit.mode == "full":
            stats["full_planes"] += 1
        else:
            stats["c_only_planes"] += 1

        bx, by = int(job["bx"]), int(job["by"])
        bw, bh = int(job["bw"]), int(job["bh"])
        cx, cy = float(job["cx"]), float(job["cy"])

        gx = np.arange(bx, bx + bw, dtype=np.float64)
        gy = np.arange(by, by + bh, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)

        inv_depth = a * (xx - cx) + b * (yy - cy) + c
        block_valid = (
            np.isfinite(inv_depth)
            & (inv_depth >= 1.0 / max_depth)
            & (inv_depth <= 1.0 / min_depth)
        )

        z = np.zeros_like(inv_depth)
        z[block_valid] = 1.0 / inv_depth[block_valid]
        block_valid &= (z >= min_depth) & (z <= max_depth)

        dst = depth[by : by + bh, bx : bx + bw]
        vm = valid[by : by + bh, bx : bx + bw]
        dst[block_valid] = z[block_valid]
        vm[block_valid] = True

    return depth, valid, stats


def write_depth_yuv420p10le(
    output_path: str,
    depth_frames: List[np.ndarray],
    depth_scale_real: float,
    max_code: int = 1023,
) -> None:
    if depth_scale_real <= 0.0:
        raise ValueError("depth_scale_real must be positive")
    if not depth_frames:
        raise ValueError("No depth frames to write")

    h, w = depth_frames[0].shape
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")

    with open(output_path, "wb") as f:
        for depth in depth_frames:
            if depth.shape != (h, w):
                raise ValueError("Inconsistent depth frame size")
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
    full_planes: int,
    c_only_planes: int,
) -> None:
    done = frame_idx + 1
    ratio = done / max(num_frames, 1)
    width = 32
    n = int(round(ratio * width))
    bar = "#" * n + "-" * (width - n)
    print(
        f"\r[{bar}] {done:3d}/{num_frames:3d} "
        f"obs={valid_obs:7d} valid={valid_ratio:7.3%} "
        f"full={full_planes:6d} cOnly={c_only_planes:6d}",
        end="",
        flush=True,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Predict each fit-block inverse-depth plane using only causal "
            "left/top/top-left MV observations."
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
        help="Output inverse-depth plane block size. Default: 16.",
    )
    ap.add_argument(
        "--neighborhood",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. Only left/top/top-left "
            "fit-blocks are used and this value is ignored."
        ),
    )
    ap.add_argument("--min-points", type=int, default=4)

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument(
        "--min-observability",
        type=float,
        default=1e-6,
        help=(
            "Minimum norm of the translation-dependent inverse-depth "
            "constraint B. B^T B below this squared threshold is rejected."
        ),
    )
    ap.add_argument(
        "--min-parallax",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --min-observability, retained for existing "
            "command lines."
        ),
    )
    ap.add_argument(
        "--min-parallax-px",
        type=float,
        default=0.0,
        help=(
            "Optional minimum rotation-compensated image parallax in pixels. "
            "Keep at 0 to retain valid far-depth observations."
        ),
    )
    ap.add_argument("--max-reproj-error", type=float, default=1.5)
    ap.add_argument(
        "--observability-weight-cap",
        type=float,
        default=8.0,
        help="Maximum normalized observability contribution to IRLS weight.",
    )

    ap.add_argument(
        "--disable-bipred-consistency",
        action="store_true",
        help="Do not reject inconsistent L0/L1 inverse-depth observations.",
    )
    ap.add_argument(
        "--bipred-inv-depth-rel-threshold",
        type=float,
        default=0.35,
    )
    ap.add_argument(
        "--bipred-inv-depth-abs-threshold",
        type=float,
        default=1e-8,
    )

    ap.add_argument(
        "--disable-source-consistency",
        action="store_true",
        help=(
            "Do not filter mutually inconsistent left/top/top-left source "
            "groups before the final fit."
        ),
    )
    ap.add_argument(
        "--source-consistency-rel-threshold",
        type=float,
        default=0.50,
    )
    ap.add_argument(
        "--source-consistency-abs-threshold",
        type=float,
        default=1e-8,
    )
    ap.add_argument(
        "--source-consistency-min-points",
        type=int,
        default=3,
    )

    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-condition", type=float, default=1e-8)
    ap.add_argument(
        "--min-normalized-span",
        type=float,
        default=0.25,
        help=(
            "Minimum x and y sample span after normalization by half block "
            "size. Otherwise only the C-only fallback is allowed."
        ),
    )
    ap.add_argument(
        "--disable-c-only-fallback",
        action="store_true",
        help="Leave unstable blocks invalid instead of fitting a=b=0 and c only.",
    )
    ap.add_argument(
        "--max-plane-slope",
        type=float,
        default=1.0,
        help="Maximum absolute inverse-depth slope per pixel.",
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

    ap.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    ap.add_argument(
        "--gpu-batch-blocks",
        type=int,
        default=4096,
        help="Number of fitting blocks processed per CUDA batch.",
    )

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.min_points < 3:
        raise ValueError("--min-points must be >= 3")
    if args.source_consistency_min_points < 3:
        raise ValueError("--source-consistency-min-points must be >= 3")
    if args.min_depth <= 0.0 or args.max_depth <= args.min_depth:
        raise ValueError("Invalid depth range")
    if args.min_observability <= 0.0:
        raise ValueError("--min-observability must be positive")
    if args.min_parallax is not None and args.min_parallax <= 0.0:
        raise ValueError("--min-parallax must be positive when provided")
    if args.min_parallax_px < 0.0 or args.max_reproj_error < 0.0:
        raise ValueError("Invalid parallax/reprojection threshold")
    if args.observability_weight_cap <= 0.0:
        raise ValueError("--observability-weight-cap must be positive")
    if args.irls_iters <= 0 or args.huber_delta <= 0.0:
        raise ValueError("Invalid IRLS settings")
    if not (0.0 < args.min_condition < 1.0):
        raise ValueError("--min-condition must be in (0,1)")
    if args.min_normalized_span < 0.0:
        raise ValueError("--min-normalized-span must be non-negative")
    if args.gpu_batch_blocks <= 0:
        raise ValueError("--gpu-batch-blocks must be positive")

    min_observability = (
        float(args.min_parallax)
        if args.min_parallax is not None
        else float(args.min_observability)
    )

    camera_json = load_camera_jsonl(args.camera_param)
    cameras = build_camera_lookup(camera_json)

    header = camera_json["header"]
    if args.depth_scale_real is None:
        precision = float(header.get("depth_scale_precision", 1.0))
        if precision <= 0.0:
            raise ValueError("Invalid depth_scale_precision")
        if "depth_scale" in header:
            depth_scale_real = float(header["depth_scale"]) / precision
        elif "depth_scale_real" in header:
            depth_scale_real = float(header["depth_scale_real"])
        else:
            raise KeyError("Camera header has no depth_scale/depth_scale_real")
    else:
        depth_scale_real = float(args.depth_scale_real)

    mv_by_frame = parse_mv_csv(args.mv_csv, args.num_frames)

    if args.device == "auto":
        use_cuda = torch is not None and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
    else:
        device = args.device
        use_cuda = device.startswith("cuda")

    if use_cuda and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable")

    print(f"device                 : {device}")
    print(f"fit block              : {args.fit_block}x{args.fit_block}")
    print("predictor source       : left + top + top-left fit-blocks only")
    print("current block MV       : disabled")
    print("MV solve parameter     : direct inverse depth")
    print(f"min observability      : {min_observability:.12g}")
    print(f"min parallax px        : {args.min_parallax_px:.6g}")
    print(f"bi-pred consistency    : {not args.disable_bipred_consistency}")
    print(f"source consistency     : {not args.disable_source_consistency}")
    print(f"C-only fallback        : {not args.disable_c_only_fallback}")
    print(f"depth scale real       : {depth_scale_real:.12g}")

    depth_frames: List[np.ndarray] = []
    frame_stats: List[Dict[str, Any]] = []

    for poc in range(args.num_frames):
        observations, observation_stats = make_inv_depth_observations(
            mv_rows=mv_by_frame[poc],
            cameras=cameras,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_observability=min_observability,
            min_parallax_px=args.min_parallax_px,
            max_reproj_error=args.max_reproj_error,
            bipred_consistency=not args.disable_bipred_consistency,
            bipred_rel_threshold=args.bipred_inv_depth_rel_threshold,
            bipred_abs_threshold=args.bipred_inv_depth_abs_threshold,
        )

        jobs, job_stats = build_block_fit_jobs(
            observations=observations,
            width=args.width,
            height=args.height,
            fit_block=args.fit_block,
            neighborhood=args.neighborhood,
            min_points=args.min_points,
            min_normalized_span=args.min_normalized_span,
            source_consistency=not args.disable_source_consistency,
            source_consistency_rel_threshold=(
                args.source_consistency_rel_threshold
            ),
            source_consistency_abs_threshold=(
                args.source_consistency_abs_threshold
            ),
            source_consistency_min_points=args.source_consistency_min_points,
            min_condition=args.min_condition,
            observability_weight_cap=args.observability_weight_cap,
            irls_iters=args.irls_iters,
            huber_delta=args.huber_delta,
        )

        fits: List[Optional[PlaneFitResult]] = []
        if use_cuda and jobs:
            for start in range(0, len(jobs), args.gpu_batch_blocks):
                batch = jobs[start : start + args.gpu_batch_blocks]
                fits.extend(
                    fit_inv_depth_planes_gpu(
                        blocks=batch,
                        irls_iters=args.irls_iters,
                        huber_delta=args.huber_delta,
                        min_condition=args.min_condition,
                        observability_weight_cap=(
                            args.observability_weight_cap
                        ),
                        c_only_fallback=not args.disable_c_only_fallback,
                        device=device,
                    )
                )
        else:
            for job in jobs:
                fits.append(
                    fit_inv_depth_plane_cpu(
                        xs=job["xs"],
                        ys=job["ys"],
                        inv_depths=job["inv_depths"],
                        reproj_errors=job["errors"],
                        observabilities=job["observabilities"],
                        cx=job["cx"],
                        cy=job["cy"],
                        sx=job["sx"],
                        sy=job["sy"],
                        allow_full=job["allow_full"],
                        irls_iters=args.irls_iters,
                        huber_delta=args.huber_delta,
                        min_condition=args.min_condition,
                        observability_weight_cap=(
                            args.observability_weight_cap
                        ),
                        c_only_fallback=not args.disable_c_only_fallback,
                    )
                )

        depth, valid, render_stats = render_jobs(
            jobs=jobs,
            fits=fits,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_plane_slope=args.max_plane_slope,
        )

        depth_frames.append(depth)
        valid_ratio = float(np.mean(valid))
        obs_values = np.asarray(
            [o.observability for o in observations], dtype=np.float64
        )
        parallax_values = np.asarray(
            [o.parallax_px for o in observations], dtype=np.float64
        )
        reproj_values = np.asarray(
            [o.reproj_error for o in observations], dtype=np.float64
        )

        frame_stats.append(
            {
                "poc": poc,
                "observation_stats": observation_stats,
                "job_stats": job_stats,
                "render_stats": render_stats,
                "valid_pixel_ratio": valid_ratio,
                "observability_median": (
                    float(np.median(obs_values)) if obs_values.size else None
                ),
                "parallax_px_median": (
                    float(np.median(parallax_values))
                    if parallax_values.size
                    else None
                ),
                "reprojection_error_median": (
                    float(np.median(reproj_values))
                    if reproj_values.size
                    else None
                ),
            }
        )

        print_progress(
            frame_idx=poc,
            num_frames=args.num_frames,
            valid_obs=len(observations),
            valid_ratio=valid_ratio,
            full_planes=render_stats["full_planes"],
            c_only_planes=render_stats["c_only_planes"],
        )

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
                "version": "2026-07-15-direct-invdepth-observability-irls",
                "mv_csv": args.mv_csv,
                "camera_param": args.camera_param,
                "out_yuv": args.out_yuv,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "fit_block": args.fit_block,
                "predictor_neighbors": ["left", "top", "top_left"],
                "current_block_mv_used": False,
                "neighborhood_legacy_ignored": args.neighborhood,
                "inverse_depth_solver": "rho=(B^T A)/(B^T B)",
                "min_observability": min_observability,
                "min_parallax_px": args.min_parallax_px,
                "max_reproj_error": args.max_reproj_error,
                "observability_weight_cap": args.observability_weight_cap,
                "bipred_consistency_enabled": (
                    not args.disable_bipred_consistency
                ),
                "bipred_inv_depth_rel_threshold": (
                    args.bipred_inv_depth_rel_threshold
                ),
                "bipred_inv_depth_abs_threshold": (
                    args.bipred_inv_depth_abs_threshold
                ),
                "source_consistency_enabled": (
                    not args.disable_source_consistency
                ),
                "source_consistency_rel_threshold": (
                    args.source_consistency_rel_threshold
                ),
                "source_consistency_abs_threshold": (
                    args.source_consistency_abs_threshold
                ),
                "min_normalized_span": args.min_normalized_span,
                "c_only_fallback": not args.disable_c_only_fallback,
                "min_points": args.min_points,
                "irls_iters": args.irls_iters,
                "huber_delta": args.huber_delta,
                "min_condition": args.min_condition,
                "device": device,
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
