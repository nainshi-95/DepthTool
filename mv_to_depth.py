#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MV + camera parameter -> smoothed sparse depth YUV420p10le

Input MV CSV columns:
  poc,x,y,w,h,list,ref_poc,mv_x,mv_y

Behavior:
  1) Each 4x4 MV observation is converted to a closed-form depth estimate.
  2) The current fit-block never uses its own MV observations. For each current
     fit-block, only MV-derived depth points from the left, top, and top-left
     fit-blocks are collected. This emulates a causal predictor available before
     decoding the current block motion.
  3) A robust inverse-depth plane
         1 / z(x,y) = a * (x-cx) + b * (y-cy) + c
     is fitted using IRLS.
  4) The fitted/extrapolated plane fills the current fitting block.
  5) For bi-prediction, L0/L1 observations are treated as independent
     constraints. They are fused by robust fitting rather than simple averaging.
  6) Blocks with too few or geometrically unstable samples remain zero.
  7) Optional CUDA acceleration is used for batched least-squares fitting.

Camera JSONL format:
  camparam_v2_vggt_or_canonical
  pose_mode: current_to_previous, gop_local, or absolute

Depth output:
  33 x H x W YUV420p10le by default.
  Invalid depth remains zero.
  UV planes are fixed to 512.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
                by_frame[poc].append(obs)
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


def relative_transform(cam_cur: Dict[str, Any], cam_ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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

    Reference projection constraints are linear in z:
      (u'-cx_ref) * (z*qz+t_z) = fx_ref * (z*qx+t_x)
      (v'-cy_ref) * (z*qz+t_z) = fy_ref * (z*qy+t_y)

    Solve two equations in least-squares closed form.
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

    # Projection uses signed positive depth = z_sign * Z.
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
# Robust 16x16 inverse-depth plane fitting
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

    # Reprojection confidence as initial weight.
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
    """
    Batched padded IRLS. Variable point counts are handled with masks.
    """
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
            torch.abs(residual_masked - med[:, None]),
            dim=1,
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
    """Build causal predictor jobs from left/top/top-left fit-blocks only.

    ``neighborhood`` is retained in the function signature and CLI for backward
    compatibility, but is intentionally ignored in this predictor mode. The
    current fit-block contributes no MV/depth observations.
    """
    del neighborhood

    if not observations:
        return []

    # Bin every valid MV-derived depth point into the fit-block that contains
    # its 4x4 center. This is substantially faster than scanning all points for
    # every target block.
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

            # Raster-causal neighboring fit-blocks. Never include (gx, gy).
            source_keys = []
            if gx > 0:
                source_keys.append((gx - 1, gy))       # left
            if gy > 0:
                source_keys.append((gx, gy - 1))       # top
            if gx > 0 and gy > 0:
                source_keys.append((gx - 1, gy - 1))   # top-left

            selected: List[DepthObservation] = []
            source_counts = {"left": 0, "top": 0, "top_left": 0}
            for key in source_keys:
                vals = block_obs.get(key, [])
                selected.extend(vals)
                if key == (gx - 1, gy):
                    source_counts["left"] += len(vals)
                elif key == (gx, gy - 1):
                    source_counts["top"] += len(vals)
                else:
                    source_counts["top_left"] += len(vals)

            if len(selected) < min_points:
                continue

            xs = np.asarray([o.x for o in selected], dtype=np.float64)
            ys = np.asarray([o.y for o in selected], dtype=np.float64)
            depths = np.asarray([o.depth for o in selected], dtype=np.float64)
            errors = np.asarray([o.reproj_error for o in selected], dtype=np.float64)

            jobs.append(
                {
                    "bx": bx,
                    "by": by,
                    "bw": bw,
                    "bh": bh,
                    "cx": cx,
                    "cy": cy,
                    "xs": xs,
                    "ys": ys,
                    "depths": depths,
                    "errors": errors,
                    "source_left_points": source_counts["left"],
                    "source_top_points": source_counts["top"],
                    "source_top_left_points": source_counts["top_left"],
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
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((height, width), dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)

    for job, coeff in zip(jobs, coeffs):
        if coeff is None:
            continue

        a, b, c = [float(v) for v in coeff]
        if abs(a) > max_plane_slope or abs(b) > max_plane_slope or c <= 0.0:
            continue

        bx, by = job["bx"], job["by"]
        bw, bh = job["bw"], job["bh"]
        cx, cy = job["cx"], job["cy"]

        gx = np.arange(bx, bx + bw, dtype=np.float64)
        gy = np.arange(by, by + bh, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)

        invz = a * (xx - cx) + b * (yy - cy) + c
        block_valid = np.isfinite(invz) & (invz > 1.0 / max_depth)
        z = np.zeros_like(invz)
        z[block_valid] = 1.0 / invz[block_valid]

        block_valid &= (z >= min_depth) & (z <= max_depth)

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
        description="Predict each fit-block depth using only left/top/top-left fit-block MV-derived depths."
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
        help="Output plane block size. Default: 16.",
    )
    ap.add_argument(
        "--neighborhood",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. In causal predictor mode, only "
            "the left/top/top-left fit-blocks are used and this value is ignored."
        ),
    )
    ap.add_argument("--min-points", type=int, default=4)

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument(
        "--min-parallax",
        type=float,
        default=1e-6,
        help="Reject nearly pure-rotation / depth-unobservable constraints.",
    )
    ap.add_argument("--max-reproj-error", type=float, default=1.5)

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
            "Override output depth scale. Default: "
            "camera header depth_scale/depth_scale_precision."
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

    print(f"device           : {device}")
    print(f"fit block        : {args.fit_block}x{args.fit_block}")
    print("predictor source : left + top + top-left fit-blocks only")
    print("current block MV : disabled")
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
        frame_stats.append(
            {
                "poc": poc,
                "mv_rows": len(mv_by_frame[poc]),
                "valid_depth_observations": len(observations),
                "fit_jobs": len(jobs),
                "successful_planes": sum(c is not None for c in coeffs),
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
                "predictor_neighbors": ["left", "top", "top_left"],
                "current_block_mv_used": False,
                "neighborhood_legacy_ignored": args.neighborhood,
                "min_points": args.min_points,
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
