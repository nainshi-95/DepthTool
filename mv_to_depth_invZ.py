#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neighbor affine-CP MV + camera parameter -> inverse-depth-plane depth YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Predictor behavior:
  1) MV rows are binned into same-sized fit-blocks.
  2) For every source fit-block and every independent (list, ref_poc) group,
     a 6-parameter affine MV field is robustly fitted and represented by three
     control points:

       CP0: source top-left
       CP1: source top-right
       CP2: source bottom-left

     mv(u,v) = CP0
             + ((u-source_x)/source_w) * (CP1-CP0)
             + ((v-source_y)/source_h) * (CP2-CP0)

  3) The current fit-block never uses its own MV rows. Only affine models from
     the left, top, and top-left same-sized fit-blocks are extrapolated into the
     current fit-block.
  4) At each sampled position in the current block, inverse depth rho=1/z is
     solved directly from the predicted affine MV and the camera parameters.
     It is not obtained by solving z first and then inverting it.
  5) All valid affine-derived rho constraints are robustly fused into

       rho(x,y) = a * (x-cx) + b * (y-cy) + c

     using IRLS. L0/L1 and different reference POCs remain independent
     constraints.
  6) The fitted inverse-depth plane is rendered as z=1/rho only at the final
     output stage. Invalid or unstable blocks remain zero.
  7) Optional CUDA acceleration is used for batched inverse-depth plane fitting.
     Affine MV fitting is performed on CPU because each source block is small.

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
# MV / affine / inverse-depth data structures
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

    @property
    def center_x(self) -> float:
        return self.x + (self.w - 1) * 0.5

    @property
    def center_y(self) -> float:
        return self.y + (self.h - 1) * 0.5


@dataclass
class AffineCPModel:
    poc: int
    source_gx: int
    source_gy: int
    source_x: int
    source_y: int
    source_w: int
    source_h: int
    list_id: str
    ref_poc: int
    cp0: np.ndarray  # top-left, shape (2,)
    cp1: np.ndarray  # top-right, shape (2,)
    cp2: np.ndarray  # bottom-left, shape (2,)
    num_points: int
    fit_rmse: float

    def predict_mv(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrapolate the source-block affine MV field at arbitrary positions."""
        alpha = (u - float(self.source_x)) / float(self.source_w)
        beta = (v - float(self.source_y)) / float(self.source_h)

        d1 = self.cp1 - self.cp0
        d2 = self.cp2 - self.cp0
        mv_x = self.cp0[0] + alpha * d1[0] + beta * d2[0]
        mv_y = self.cp0[1] + alpha * d1[1] + beta * d2[1]
        return mv_x, mv_y


@dataclass
class InvDepthObservation:
    x: float
    y: float
    inv_depth: float
    reproj_error: float
    affine_rmse: float
    ref_poc: int
    list_id: str
    source_name: str

    @property
    def base_weight(self) -> float:
        # Both terms are measured in pixels. The product prevents a poor affine
        # fit from dominating merely because its camera reprojection is small.
        w_reproj = 1.0 / (1.0 + self.reproj_error * self.reproj_error)
        w_affine = 1.0 / (1.0 + self.affine_rmse * self.affine_rmse)
        return w_reproj * w_affine


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

                obs = MVObservation(
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
                if obs.w <= 0 or obs.h <= 0:
                    continue
                if not np.isfinite([obs.mv_x, obs.mv_y]).all():
                    continue
                by_frame[poc].append(obs)
            except Exception as exc:
                raise RuntimeError(f"Bad CSV row {row_no}: {row}") from exc

    return by_frame


# ============================================================
# Camera projection and direct inverse-depth solver
# ============================================================


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


def solve_inv_depth_closed_form(
    u: float,
    v: float,
    mv_x: float,
    mv_y: float,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
    min_observability: float,
    max_reproj_error: float,
) -> Optional[Tuple[float, float]]:
    """Solve rho=1/z directly from a current-to-reference correspondence.

    Current point:
      X_cur = z * ray = ray / rho

    Relative camera transform:
      X_ref = R * X_cur + t = q / rho + t

    After multiplying the reference projection equations by rho, each image
    coordinate gives one equation linear in rho:

      B_u * rho = A_u
      B_v * rho = A_v

    where
      A_u = (u'-cx) * s * q_z - fx * q_x
      B_u = fx * t_x - (u'-cx) * s * t_z

    and likewise for v. The two equations are solved by scalar least squares.
    """
    ur = u + mv_x
    vr = v + mv_y

    ray = pixel_ray(u, v, cam_cur)
    R, t = relative_transform(cam_cur, cam_ref)
    q = R @ ray

    K = cam_ref["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    s = float(cam_ref["z_sign"])

    du = ur - cx
    dv = vr - cy

    A_u = du * s * q[2] - fx * q[0]
    B_u = fx * t[0] - du * s * t[2]

    A_v = dv * s * q[2] - fy * q[1]
    B_v = fy * t[1] - dv * s * t[2]

    A = np.array([A_u, A_v], dtype=np.float64)
    B = np.array([B_u, B_v], dtype=np.float64)

    denom = float(np.dot(B, B))
    if not np.isfinite(denom) or denom < min_observability * min_observability:
        return None

    inv_depth = float(np.dot(B, A) / denom)
    if not np.isfinite(inv_depth) or inv_depth <= 0.0:
        return None

    # z is formed only for geometric validation; rho itself is the estimated
    # variable and is what gets passed to the plane fitting stage.
    depth = 1.0 / inv_depth
    X_ref = depth * q + t
    pred = project_point(X_ref, cam_ref)
    if pred is None:
        return None

    reproj_error = float(
        np.linalg.norm(pred - np.array([ur, vr], dtype=np.float64))
    )
    if not np.isfinite(reproj_error) or reproj_error > max_reproj_error:
        return None

    return inv_depth, reproj_error


# ============================================================
# Robust affine CP fitting in each source block
# ============================================================


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def block_geometry(
    gx: int,
    gy: int,
    fit_block: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    bx = gx * fit_block
    by = gy * fit_block
    bw = min(fit_block, width - bx)
    bh = min(fit_block, height - by)
    return bx, by, bw, bh


def fit_affine_cp_model(
    rows: List[MVObservation],
    source_gx: int,
    source_gy: int,
    source_x: int,
    source_y: int,
    source_w: int,
    source_h: int,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
    max_rmse: float,
) -> Optional[AffineCPModel]:
    if len(rows) < 3:
        return None

    xs = np.asarray([r.center_x for r in rows], dtype=np.float64)
    ys = np.asarray([r.center_y for r in rows], dtype=np.float64)
    target = np.asarray([[r.mv_x, r.mv_y] for r in rows], dtype=np.float64)

    # Normalized source-block coordinates make the fitted coefficients equal to
    # CP differences and keep the conditioning independent of block size.
    alpha = (xs - float(source_x)) / float(source_w)
    beta = (ys - float(source_y)) / float(source_h)
    A = np.stack([np.ones_like(alpha), alpha, beta], axis=1)

    weights = np.ones(len(rows), dtype=np.float64)
    coeff: Optional[np.ndarray] = None  # shape (3, 2)

    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-10))
        Aw = A * sw[:, None]
        Yw = target * sw[:, None]

        normal = Aw.T @ Aw
        eig = np.linalg.eigvalsh(normal)
        if eig[-1] <= 1e-15 or eig[0] / eig[-1] < min_condition:
            return None

        try:
            coeff = np.linalg.solve(normal, Aw.T @ Yw)
        except np.linalg.LinAlgError:
            return None

        residual_vec = target - A @ coeff
        residual_mag = np.linalg.norm(residual_vec, axis=1)
        med = float(np.median(residual_mag))
        mad = float(np.median(np.abs(residual_mag - med)))
        scale = max(1.4826 * mad, 1e-6)
        weights = huber_weights(residual_mag / scale, huber_delta)

    if coeff is None or not np.isfinite(coeff).all():
        return None

    residual_vec = target - A @ coeff
    rmse = float(np.sqrt(np.mean(np.sum(residual_vec * residual_vec, axis=1))))
    if not np.isfinite(rmse) or rmse > max_rmse:
        return None

    cp0 = coeff[0].copy()
    cp1 = coeff[0] + coeff[1]
    cp2 = coeff[0] + coeff[2]

    first = rows[0]
    return AffineCPModel(
        poc=first.poc,
        source_gx=source_gx,
        source_gy=source_gy,
        source_x=source_x,
        source_y=source_y,
        source_w=source_w,
        source_h=source_h,
        list_id=first.list_id,
        ref_poc=first.ref_poc,
        cp0=cp0.astype(np.float64),
        cp1=cp1.astype(np.float64),
        cp2=cp2.astype(np.float64),
        num_points=len(rows),
        fit_rmse=rmse,
    )


def build_affine_models_by_block(
    mv_rows: List[MVObservation],
    width: int,
    height: int,
    fit_block: int,
    min_affine_points: int,
    affine_irls_iters: int,
    affine_huber_delta: float,
    affine_min_condition: float,
    max_affine_rmse: float,
) -> Tuple[Dict[Tuple[int, int], List[AffineCPModel]], Dict[str, int]]:
    """Fit one affine model per (source block, list, reference POC)."""
    grouped: Dict[Tuple[int, int, str, int], List[MVObservation]] = {}

    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block

    for row in mv_rows:
        cx = row.center_x
        cy = row.center_y
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            continue

        gx = int(cx) // fit_block
        gy = int(cy) // fit_block
        if not (0 <= gx < grid_w and 0 <= gy < grid_h):
            continue

        grouped.setdefault((gx, gy, row.list_id, row.ref_poc), []).append(row)

    models_by_block: Dict[Tuple[int, int], List[AffineCPModel]] = {}
    attempted = 0
    rejected_too_few = 0
    rejected_fit = 0

    for (gx, gy, _list_id, _ref_poc), rows in grouped.items():
        if len(rows) < min_affine_points:
            rejected_too_few += 1
            continue

        attempted += 1
        bx, by, bw, bh = block_geometry(gx, gy, fit_block, width, height)
        model = fit_affine_cp_model(
            rows=rows,
            source_gx=gx,
            source_gy=gy,
            source_x=bx,
            source_y=by,
            source_w=bw,
            source_h=bh,
            irls_iters=affine_irls_iters,
            huber_delta=affine_huber_delta,
            min_condition=affine_min_condition,
            max_rmse=max_affine_rmse,
        )
        if model is None:
            rejected_fit += 1
            continue

        models_by_block.setdefault((gx, gy), []).append(model)

    stats = {
        "affine_groups": len(grouped),
        "affine_attempted": attempted,
        "affine_rejected_too_few": rejected_too_few,
        "affine_rejected_fit": rejected_fit,
        "affine_models": sum(len(v) for v in models_by_block.values()),
    }
    return models_by_block, stats


# ============================================================
# Current-block inverse-depth constraints from neighbor affine CPs
# ============================================================


def sample_block_centers(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return centers of step-sized cells clipped at the block boundary."""
    xs: List[float] = []
    ys: List[float] = []

    for y0 in range(by, by + bh, step):
        cell_h = min(step, by + bh - y0)
        cy = y0 + (cell_h - 1) * 0.5
        for x0 in range(bx, bx + bw, step):
            cell_w = min(step, bx + bw - x0)
            cx = x0 + (cell_w - 1) * 0.5
            xs.append(cx)
            ys.append(cy)

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def iter_same_size_neighbor_models(
    gx: int,
    gy: int,
    current_w: int,
    current_h: int,
    fit_block: int,
    width: int,
    height: int,
    models_by_block: Dict[Tuple[int, int], List[AffineCPModel]],
) -> Iterable[Tuple[str, AffineCPModel]]:
    neighbor_specs = (
        ("left", gx - 1, gy),
        ("top", gx, gy - 1),
        ("top_left", gx - 1, gy - 1),
    )

    for source_name, sx, sy in neighbor_specs:
        if sx < 0 or sy < 0:
            continue

        _bx, _by, sw, sh = block_geometry(sx, sy, fit_block, width, height)
        if sw != current_w or sh != current_h:
            continue

        for model in models_by_block.get((sx, sy), []):
            yield source_name, model


def build_block_fit_jobs_from_affine(
    poc: int,
    models_by_block: Dict[Tuple[int, int], List[AffineCPModel]],
    cameras: Dict[int, Dict[str, Any]],
    width: int,
    height: int,
    fit_block: int,
    sample_step: int,
    min_points: int,
    min_depth: float,
    max_depth: float,
    min_observability: float,
    max_reproj_error: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    jobs: List[Dict[str, Any]] = []
    grid_w = (width + fit_block - 1) // fit_block
    grid_h = (height + fit_block - 1) // fit_block

    if poc not in cameras:
        return jobs, {
            "neighbor_affine_models_used": 0,
            "inv_depth_constraints": 0,
            "blocks_with_constraints": 0,
        }

    cam_cur = cameras[poc]
    total_models_used = 0
    total_constraints = 0
    blocks_with_constraints = 0

    for gy in range(grid_h):
        for gx in range(grid_w):
            bx, by, bw, bh = block_geometry(gx, gy, fit_block, width, height)
            cx = bx + (bw - 1) * 0.5
            cy = by + (bh - 1) * 0.5

            sample_x, sample_y = sample_block_centers(
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                step=sample_step,
            )

            selected: List[InvDepthObservation] = []
            source_model_counts = {"left": 0, "top": 0, "top_left": 0}
            source_constraint_counts = {"left": 0, "top": 0, "top_left": 0}

            for source_name, model in iter_same_size_neighbor_models(
                gx=gx,
                gy=gy,
                current_w=bw,
                current_h=bh,
                fit_block=fit_block,
                width=width,
                height=height,
                models_by_block=models_by_block,
            ):
                if model.ref_poc not in cameras:
                    continue

                mv_x, mv_y = model.predict_mv(sample_x, sample_y)
                model_constraint_count = 0

                for u, v, dx, dy in zip(sample_x, sample_y, mv_x, mv_y):
                    solved = solve_inv_depth_closed_form(
                        u=float(u),
                        v=float(v),
                        mv_x=float(dx),
                        mv_y=float(dy),
                        cam_cur=cam_cur,
                        cam_ref=cameras[model.ref_poc],
                        min_observability=min_observability,
                        max_reproj_error=max_reproj_error,
                    )
                    if solved is None:
                        continue

                    inv_depth, reproj_error = solved
                    if inv_depth < 1.0 / max_depth or inv_depth > 1.0 / min_depth:
                        continue

                    selected.append(
                        InvDepthObservation(
                            x=float(u),
                            y=float(v),
                            inv_depth=inv_depth,
                            reproj_error=reproj_error,
                            affine_rmse=model.fit_rmse,
                            ref_poc=model.ref_poc,
                            list_id=model.list_id,
                            source_name=source_name,
                        )
                    )
                    model_constraint_count += 1

                if model_constraint_count > 0:
                    total_models_used += 1
                    source_model_counts[source_name] += 1
                    source_constraint_counts[source_name] += model_constraint_count

            if len(selected) < min_points:
                continue

            blocks_with_constraints += 1
            total_constraints += len(selected)

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
                    "inv_depths": np.asarray(
                        [o.inv_depth for o in selected], dtype=np.float64
                    ),
                    "base_weights": np.asarray(
                        [o.base_weight for o in selected], dtype=np.float64
                    ),
                    "source_left_models": source_model_counts["left"],
                    "source_top_models": source_model_counts["top"],
                    "source_top_left_models": source_model_counts["top_left"],
                    "source_left_constraints": source_constraint_counts["left"],
                    "source_top_constraints": source_constraint_counts["top"],
                    "source_top_left_constraints": source_constraint_counts["top_left"],
                }
            )

    stats = {
        "neighbor_affine_models_used": total_models_used,
        "inv_depth_constraints": total_constraints,
        "blocks_with_constraints": blocks_with_constraints,
    }
    return jobs, stats


# ============================================================
# Robust current-block inverse-depth plane fitting
# ============================================================


def fit_inv_depth_plane_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    inv_depths: np.ndarray,
    base_weights: np.ndarray,
    cx: float,
    cy: float,
    irls_iters: int,
    huber_delta: float,
    min_condition: float,
) -> Optional[np.ndarray]:
    if inv_depths.size < 3:
        return None

    A = np.stack([xs - cx, ys - cy, np.ones_like(xs)], axis=1)
    weights = np.asarray(base_weights, dtype=np.float64).copy()

    coeff = None
    for _ in range(max(1, irls_iters)):
        sw = np.sqrt(np.maximum(weights, 1e-10))
        Aw = A * sw[:, None]
        bw = inv_depths * sw

        normal = Aw.T @ Aw
        eig = np.linalg.eigvalsh(normal)
        if eig[-1] <= 1e-15 or eig[0] / eig[-1] < min_condition:
            return None

        try:
            coeff = np.linalg.solve(normal, Aw.T @ bw)
        except np.linalg.LinAlgError:
            return None

        residual = inv_depths - A @ coeff
        med = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - med)))
        scale = max(scale, 1e-10)
        robust = huber_weights(residual / scale, huber_delta)
        weights = base_weights * robust

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
    """Batched padded IRLS with direct inverse-depth observations."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if not blocks:
        return []

    dev = torch.device(device)
    B = len(blocks)
    max_n = max(len(b["inv_depths"]) for b in blocks)

    A = torch.zeros((B, max_n, 3), dtype=torch.float64, device=dev)
    y = torch.zeros((B, max_n), dtype=torch.float64, device=dev)
    base_w = torch.zeros((B, max_n), dtype=torch.float64, device=dev)
    mask = torch.zeros((B, max_n), dtype=torch.bool, device=dev)

    for i, b in enumerate(blocks):
        n = len(b["inv_depths"])
        xs = torch.as_tensor(b["xs"], dtype=torch.float64, device=dev)
        ys = torch.as_tensor(b["ys"], dtype=torch.float64, device=dev)
        inv_depths = torch.as_tensor(
            b["inv_depths"], dtype=torch.float64, device=dev
        )
        weights = torch.as_tensor(
            b["base_weights"], dtype=torch.float64, device=dev
        )
        cx = float(b["cx"])
        cy = float(b["cy"])

        A[i, :n, 0] = xs - cx
        A[i, :n, 1] = ys - cy
        A[i, :n, 2] = 1.0
        y[i, :n] = inv_depths
        base_w[i, :n] = weights
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
        scale = torch.clamp(1.4826 * mad, min=1e-10)

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
# Rendering / output
# ============================================================


def render_jobs(
    jobs: List[Dict[str, Any]],
    coeffs: List[Optional[np.ndarray]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)

    min_inv_depth = 1.0 / max_depth
    max_inv_depth = 1.0 / min_depth

    for job, coeff in zip(jobs, coeffs):
        if coeff is None:
            continue

        a, b, c = [float(v) for v in coeff]
        if abs(a) > max_plane_slope or abs(b) > max_plane_slope:
            continue

        bx, by = job["bx"], job["by"]
        bw, bh = job["bw"], job["bh"]
        cx, cy = job["cx"], job["cy"]

        gx = np.arange(bx, bx + bw, dtype=np.float64)
        gy = np.arange(by, by + bh, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)

        invz = a * (xx - cx) + b * (yy - cy) + c
        block_valid = (
            np.isfinite(invz)
            & (invz >= min_inv_depth)
            & (invz <= max_inv_depth)
        )

        z = np.zeros_like(invz)
        z[block_valid] = 1.0 / invz[block_valid]
        block_valid &= np.isfinite(z) & (z >= min_depth) & (z <= max_depth)

        dst = depth[by:by + bh, bx:bx + bw]
        vm = valid[by:by + bh, bx:bx + bw]
        dst[block_valid] = z[block_valid]
        vm[block_valid] = True

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
        raise ValueError("No depth frames to write")

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
    affine_models: int,
    constraints: int,
    valid_ratio: float,
) -> None:
    done = frame_idx + 1
    ratio = done / max(num_frames, 1)
    width = 32
    n = int(round(ratio * width))
    bar = "#" * n + "-" * (width - n)
    print(
        f"\r[{bar}] {done:3d}/{num_frames:3d} "
        f"affine={affine_models:6d} rho={constraints:8d} "
        f"valid={valid_ratio:7.3%}",
        end="",
        flush=True,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fit affine CPs in left/top/top-left same-sized blocks, extrapolate "
            "their MV fields into the current block, and solve inverse depth "
            "rho=1/z directly before robust inverse-depth-plane fitting."
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
        help="Source affine block and output inverse-depth-plane block size.",
    )
    ap.add_argument(
        "--sample-step",
        type=int,
        default=4,
        help="Sampling interval inside the current block for affine-MV-to-rho constraints.",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum valid affine-derived rho constraints for plane fitting.",
    )
    ap.add_argument(
        "--min-affine-points",
        type=int,
        default=4,
        help="Minimum MV samples required to fit one source-block affine model.",
    )

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument(
        "--min-observability",
        "--min-parallax",
        dest="min_observability",
        type=float,
        default=1e-6,
        help=(
            "Minimum norm of the direct inverse-depth coefficient B. "
            "--min-parallax is retained as an alias."
        ),
    )
    ap.add_argument("--max-reproj-error", type=float, default=1.5)

    ap.add_argument("--affine-irls-iters", type=int, default=3)
    ap.add_argument("--affine-huber-delta", type=float, default=1.5)
    ap.add_argument("--affine-min-condition", type=float, default=1e-8)
    ap.add_argument(
        "--max-affine-rmse",
        type=float,
        default=2.0,
        help="Reject source affine MV models whose 2D MV RMSE exceeds this value.",
    )

    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-condition", type=float, default=1e-8)
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
        help="Number of inverse-depth plane jobs processed per CUDA batch.",
    )

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.sample_step <= 0:
        raise ValueError("--sample-step must be positive")
    if args.min_points < 3:
        raise ValueError("--min-points must be >= 3")
    if args.min_affine_points < 3:
        raise ValueError("--min-affine-points must be >= 3")
    if args.min_depth <= 0.0 or args.max_depth <= args.min_depth:
        raise ValueError("Invalid depth range")

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
    print(f"affine/rho sample step : {args.sample_step}")
    print("predictor source       : left + top + top-left same-sized blocks")
    print("source representation  : per-(list,ref) 6-parameter affine CP")
    print("estimated variable     : inverse depth rho=1/z directly")
    print("current block MV       : disabled")
    print(f"depth scale real       : {depth_scale_real:.12g}")

    depth_frames: List[np.ndarray] = []
    frame_stats: List[Dict[str, Any]] = []

    for poc in range(args.num_frames):
        models_by_block, affine_stats = build_affine_models_by_block(
            mv_rows=mv_by_frame[poc],
            width=args.width,
            height=args.height,
            fit_block=args.fit_block,
            min_affine_points=args.min_affine_points,
            affine_irls_iters=args.affine_irls_iters,
            affine_huber_delta=args.affine_huber_delta,
            affine_min_condition=args.affine_min_condition,
            max_affine_rmse=args.max_affine_rmse,
        )

        jobs, predictor_stats = build_block_fit_jobs_from_affine(
            poc=poc,
            models_by_block=models_by_block,
            cameras=cameras,
            width=args.width,
            height=args.height,
            fit_block=args.fit_block,
            sample_step=args.sample_step,
            min_points=args.min_points,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_observability=args.min_observability,
            max_reproj_error=args.max_reproj_error,
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
                        inv_depths=job["inv_depths"],
                        base_weights=job["base_weights"],
                        cx=job["cx"],
                        cy=job["cy"],
                        irls_iters=args.irls_iters,
                        huber_delta=args.huber_delta,
                        min_condition=args.min_condition,
                    )
                )

        depth, valid = render_jobs(
            jobs=jobs,
            coeffs=coeffs,
            width=args.width,
            height=args.height,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_plane_slope=args.max_plane_slope,
        )

        depth_frames.append(depth)
        valid_ratio = float(np.mean(valid))
        successful_planes = sum(c is not None for c in coeffs)

        frame_stats.append(
            {
                "poc": poc,
                "mv_rows": len(mv_by_frame[poc]),
                **affine_stats,
                **predictor_stats,
                "fit_jobs": len(jobs),
                "successful_planes": successful_planes,
                "valid_pixel_ratio": valid_ratio,
            }
        )

        print_progress(
            frame_idx=poc,
            num_frames=args.num_frames,
            affine_models=affine_stats["affine_models"],
            constraints=predictor_stats["inv_depth_constraints"],
            valid_ratio=valid_ratio,
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
                "mv_csv": args.mv_csv,
                "camera_param": args.camera_param,
                "out_yuv": args.out_yuv,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "fit_block": args.fit_block,
                "sample_step": args.sample_step,
                "predictor_neighbors": ["left", "top", "top_left"],
                "same_size_neighbor_only": True,
                "current_block_mv_used": False,
                "source_motion_model": "6_parameter_affine_cp",
                "affine_grouping": ["list", "ref_poc"],
                "estimated_variable": "inverse_depth_direct",
                "min_points": args.min_points,
                "min_affine_points": args.min_affine_points,
                "max_affine_rmse": args.max_affine_rmse,
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
