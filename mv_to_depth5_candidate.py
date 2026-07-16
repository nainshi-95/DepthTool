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
  4) Reproject every candidate into already decoded depth maps, reject
     geometrically unstable candidates, and blend only the surviving
     inverse-depth cluster.
  5) Preserve valid local depth in configured anchor pictures and use
     propagated depth only to fill their holes.
  6) Generate multiple deterministic local/temporal/fused candidates.
  7) Use an encoder-side GT depth map to choose the closest candidate per
     processing block, write the oracle-selected result, and register it in the
     causal depth bank for pictures decoded later.

Default RA decoder order for 33 pictures:
  0, 32, 16, 8, 24, 4, 12, 20, 28, ...

All predictor generation, propagation, geometry checking, and fusion are
performed at a configurable reduced resolution. The final depth and confidence
maps are upsampled to the original resolution using bilinear or nearest-neighbor
interpolation and written in display POC order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None


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


def scale_camera_lookup(
    cameras: Dict[int, Dict[str, Any]],
    downsample_scale: int,
) -> Dict[int, Dict[str, Any]]:
    """Scale camera intrinsics to the reduced processing resolution.

    Pixel centers follow the common half-pixel resize convention:
      u_low = (u_full + 0.5) / scale - 0.5

    Extrinsics and metric depth are unchanged.
    """
    if downsample_scale <= 0:
        raise ValueError("downsample_scale must be positive")

    scale = float(downsample_scale)
    out: Dict[int, Dict[str, Any]] = {}
    for poc, cam in cameras.items():
        K = np.asarray(cam["K"], dtype=np.float64).copy()
        K[0, 0] /= scale
        K[1, 1] /= scale
        K[0, 2] = (K[0, 2] + 0.5) / scale - 0.5
        K[1, 2] = (K[1, 2] + 0.5) / scale - 0.5
        out[poc] = {
            "poc": int(cam["poc"]),
            "K": K,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "W2C": np.asarray(cam["W2C"], dtype=np.float64).copy(),
            "C2W": np.asarray(cam["C2W"], dtype=np.float64).copy(),
            "z_sign": float(cam["z_sign"]),
        }
    return out


def full_to_processing_coordinate(value: float, downsample_scale: int) -> float:
    return (float(value) + 0.5) / float(downsample_scale) - 0.5


@dataclass(frozen=True)
class RelativeCameraTransform:
    """Cached transform from one camera coordinate system to another.

    The scalar fields are intentionally duplicated from the matrices so the
    per-MV closed-form solver can avoid constructing tiny NumPy arrays.
    """

    src_poc: int
    dst_poc: int
    R: np.ndarray
    t: np.ndarray
    r00: float
    r01: float
    r02: float
    r10: float
    r11: float
    r12: float
    r20: float
    r21: float
    r22: float
    tx: float
    ty: float
    tz: float
    src_fx: float
    src_fy: float
    src_cx: float
    src_cy: float
    src_z_sign: float
    dst_fx: float
    dst_fy: float
    dst_cx: float
    dst_cy: float
    dst_z_sign: float


def build_relative_transform_cache(
    cameras: Dict[int, Dict[str, Any]],
) -> Dict[Tuple[int, int], RelativeCameraTransform]:
    """Build every camera-pair transform once.

    A 33-picture GOP has only 1089 ordered pairs, so precomputing them is much
    cheaper than repeating matrix multiplication for every 4x4 MV or every
    candidate/reference geometry check.
    """
    cache: Dict[Tuple[int, int], RelativeCameraTransform] = {}
    for src_poc, src in cameras.items():
        src_c2w = np.asarray(src["C2W"], dtype=np.float64)
        for dst_poc, dst in cameras.items():
            M = np.asarray(dst["W2C"], dtype=np.float64) @ src_c2w
            R = np.ascontiguousarray(M[:3, :3], dtype=np.float64)
            t = np.ascontiguousarray(M[:3, 3], dtype=np.float64)
            cache[(src_poc, dst_poc)] = RelativeCameraTransform(
                src_poc=src_poc,
                dst_poc=dst_poc,
                R=R,
                t=t,
                r00=float(R[0, 0]),
                r01=float(R[0, 1]),
                r02=float(R[0, 2]),
                r10=float(R[1, 0]),
                r11=float(R[1, 1]),
                r12=float(R[1, 2]),
                r20=float(R[2, 0]),
                r21=float(R[2, 1]),
                r22=float(R[2, 2]),
                tx=float(t[0]),
                ty=float(t[1]),
                tz=float(t[2]),
                src_fx=float(src["fx"]),
                src_fy=float(src["fy"]),
                src_cx=float(src["cx"]),
                src_cy=float(src["cy"]),
                src_z_sign=float(src["z_sign"]),
                dst_fx=float(dst["fx"]),
                dst_fy=float(dst["fy"]),
                dst_cx=float(dst["cx"]),
                dst_cy=float(dst["cy"]),
                dst_z_sign=float(dst["z_sign"]),
            )
    return cache


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
class ProcessingMVSample:
    poc: int
    x: float
    y: float
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


@dataclass
class DepthCandidate:
    name: str
    label: int
    source_poc: Optional[int]
    depth: np.ndarray
    confidence: np.ndarray


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


def solve_depth_closed_form_scalar(
    u: float,
    v: float,
    mv_x: float,
    mv_y: float,
    rel: RelativeCameraTransform,
    min_parallax: float,
    max_reproj_error: float,
) -> Optional[Tuple[float, float]]:
    """Closed-form depth solve without per-MV NumPy allocations.

    All camera-pair matrices and scalar intrinsic values are cached in
    ``RelativeCameraTransform``. This function therefore performs only scalar
    arithmetic for each sampled MV.
    """
    ur = u + mv_x
    vr = v + mv_y

    rx = (u - rel.src_cx) / rel.src_fx
    ry = (v - rel.src_cy) / rel.src_fy
    rz = rel.src_z_sign

    qx = rel.r00 * rx + rel.r01 * ry + rel.r02 * rz
    qy = rel.r10 * rx + rel.r11 * ry + rel.r12 * rz
    qz = rel.r20 * rx + rel.r21 * ry + rel.r22 * rz

    du = ur - rel.dst_cx
    dv = vr - rel.dst_cy

    au = du * rel.dst_z_sign * qz - rel.dst_fx * qx
    bu = rel.dst_fx * rel.tx - du * rel.dst_z_sign * rel.tz
    av = dv * rel.dst_z_sign * qz - rel.dst_fy * qy
    bv = rel.dst_fy * rel.ty - dv * rel.dst_z_sign * rel.tz

    denom = au * au + av * av
    min_denom = min_parallax * min_parallax
    if not math.isfinite(denom) or denom < min_denom:
        return None

    depth = (au * bu + av * bv) / denom
    if not math.isfinite(depth) or depth <= 0.0:
        return None

    x_ref = depth * qx + rel.tx
    y_ref = depth * qy + rel.ty
    z_ref = depth * qz + rel.tz
    ref_depth = rel.dst_z_sign * z_ref
    if not math.isfinite(ref_depth) or ref_depth <= 1e-10:
        return None

    pred_u = rel.dst_fx * x_ref / ref_depth + rel.dst_cx
    pred_v = rel.dst_fy * y_ref / ref_depth + rel.dst_cy
    err_u = pred_u - ur
    err_v = pred_v - vr
    reproj_error = math.hypot(err_u, err_v)
    if not math.isfinite(reproj_error) or reproj_error > max_reproj_error:
        return None

    return float(depth), float(reproj_error)


# ============================================================
# Robust inverse-depth plane fitting
# ============================================================

def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(residual)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w



def _weighted_constant_inv_depth(
    invz: np.ndarray,
    weights: np.ndarray,
) -> Optional[float]:
    sw = float(np.sum(weights))
    if not math.isfinite(sw) or sw <= 1e-12:
        return None
    c = float(np.sum(weights * invz) / sw)
    return c if math.isfinite(c) and c > 0.0 else None


def _solve_weighted_plane_normal_equation(
    dx: np.ndarray,
    dy: np.ndarray,
    invz: np.ndarray,
    weights: np.ndarray,
    determinant_threshold: float,
) -> Optional[np.ndarray]:
    """Solve a 3-parameter weighted plane without eigendecomposition."""
    sxx = float(np.sum(weights * dx * dx))
    syy = float(np.sum(weights * dy * dy))
    sxy = float(np.sum(weights * dx * dy))
    sx = float(np.sum(weights * dx))
    sy = float(np.sum(weights * dy))
    sw = float(np.sum(weights))
    sxz = float(np.sum(weights * dx * invz))
    syz = float(np.sum(weights * dy * invz))
    sz = float(np.sum(weights * invz))

    # Determinant of the symmetric normal matrix.
    det = (
        sxx * (syy * sw - sy * sy)
        - sxy * (sxy * sw - sy * sx)
        + sx * (sxy * sy - syy * sx)
    )
    scale = max(abs(sxx), abs(syy), abs(sw), 1.0)
    if not math.isfinite(det) or abs(det) <= determinant_threshold * scale * scale * scale:
        return None

    normal = np.array(
        [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, sw]],
        dtype=np.float64,
    )
    rhs = np.array([sxz, syz, sz], dtype=np.float64)
    try:
        coeff = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coeff).all() or coeff[2] <= 0.0:
        return None
    return coeff


def fit_inv_depth_plane_fast_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    reproj_errors: np.ndarray,
    cx: float,
    cy: float,
    constant_relative_threshold: float,
    refit_relative_threshold: float,
    determinant_threshold: float,
    enable_refit: bool,
) -> Optional[np.ndarray]:
    """Decoder-oriented local fit: c-only early exit, one WLS, optional refit."""
    if depths.size < 3:
        return None

    invz = 1.0 / depths
    weights = 1.0 / np.maximum(1.0 + reproj_errors * reproj_errors, 1e-6)
    c_only = _weighted_constant_inv_depth(invz, weights)
    if c_only is None:
        return None

    c_residual = np.abs(invz - c_only) / np.maximum(invz, 1e-12)
    weighted_c_error = float(np.sum(weights * c_residual) / max(np.sum(weights), 1e-12))
    if weighted_c_error <= constant_relative_threshold:
        return np.array([0.0, 0.0, c_only], dtype=np.float64)

    dx = xs - cx
    dy = ys - cy
    coeff = _solve_weighted_plane_normal_equation(
        dx, dy, invz, weights, determinant_threshold
    )
    if coeff is None:
        return np.array([0.0, 0.0, c_only], dtype=np.float64)

    if enable_refit and depths.size >= 4:
        pred = coeff[0] * dx + coeff[1] * dy + coeff[2]
        rel = np.abs(pred - invz) / np.maximum(invz, 1e-12)
        inlier = rel <= refit_relative_threshold
        if 3 <= int(np.count_nonzero(inlier)) < depths.size:
            refit = _solve_weighted_plane_normal_equation(
                dx[inlier], dy[inlier], invz[inlier], weights[inlier], determinant_threshold
            )
            if refit is not None:
                coeff = refit

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

def _select_spatially_diverse_mv_samples(
    items: List[ProcessingMVSample],
    budget: int,
    block_center_x: float,
    block_center_y: float,
) -> List[ProcessingMVSample]:
    """Select a small deterministic subset with spatial/ref-list diversity."""
    if budget <= 0 or len(items) <= budget:
        return items

    selected: List[ProcessingMVSample] = []
    remaining = list(items)

    # Preserve at least one sample from as many list/reference groups as the
    # budget permits. The representative nearest the fit-block center is less
    # likely to sit exactly on a block boundary; larger MV magnitude breaks ties.
    groups: Dict[Tuple[str, int], List[ProcessingMVSample]] = {}
    for item in remaining:
        groups.setdefault((item.list_id, item.ref_poc), []).append(item)

    group_order = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), str(kv[0][0]), int(kv[0][1])),
    )
    for _, group_items in group_order:
        if len(selected) >= budget:
            break
        chosen = min(
            group_items,
            key=lambda s: (
                (s.x - block_center_x) ** 2 + (s.y - block_center_y) ** 2,
                -(s.mv_x * s.mv_x + s.mv_y * s.mv_y),
                s.x,
                s.y,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)

    # Fill the rest by farthest-point sampling. This avoids taking four nearly
    # identical 4x4 MVs from the same corner of the fit block.
    while remaining and len(selected) < budget:
        def candidate_score(sample: ProcessingMVSample) -> Tuple[float, float, float, float]:
            min_dist = min(
                (sample.x - chosen.x) ** 2 + (sample.y - chosen.y) ** 2
                for chosen in selected
            )
            group_is_new = all(
                (sample.list_id, sample.ref_poc)
                != (chosen.list_id, chosen.ref_poc)
                for chosen in selected
            )
            motion_mag = sample.mv_x * sample.mv_x + sample.mv_y * sample.mv_y
            # Tuple max: favor spatial coverage, then unseen ref/list, then
            # larger motion/parallax proxy, with deterministic coordinate ties.
            return (
                min_dist,
                1.0 if group_is_new else 0.0,
                motion_mag,
                -(sample.x + sample.y * 1e-3),
            )

        chosen = max(remaining, key=candidate_score)
        selected.append(chosen)
        remaining.remove(chosen)

    return selected


def sample_mv_rows_for_processing(
    mv_rows: Sequence[MVObservation],
    full_width: int,
    full_height: int,
    processing_width: int,
    processing_height: int,
    downsample_scale: int,
    processing_fit_block: int,
    max_samples_per_fit_block: int,
) -> Tuple[List[ProcessingMVSample], Dict[str, int]]:
    """Reduce 4x4-span MVs before any closed-form depth solve.

    Sampling is done per reduced-resolution fit block, across all L0/L1 and
    reference POCs. ``max_samples_per_fit_block=0`` disables reduction.
    """
    scale = float(downsample_scale)
    grouped: Dict[Tuple[int, int], List[ProcessingMVSample]] = {}
    eligible = 0
    duplicate_count = 0
    seen_keys: Dict[Tuple[int, int], set[Tuple[Any, ...]]] = {}

    for row in mv_rows:
        cx_full = row.x + (row.w - 1) * 0.5
        cy_full = row.y + (row.h - 1) * 0.5
        if not (0.0 <= cx_full < full_width and 0.0 <= cy_full < full_height):
            continue

        cx = full_to_processing_coordinate(cx_full, downsample_scale)
        cy = full_to_processing_coordinate(cy_full, downsample_scale)
        if not (
            -0.5 <= cx < processing_width - 0.5
            and -0.5 <= cy < processing_height - 0.5
        ):
            continue

        gx = int(math.floor(max(cx, 0.0) / processing_fit_block))
        gy = int(math.floor(max(cy, 0.0) / processing_fit_block))
        key = (gx, gy)
        sample = ProcessingMVSample(
            poc=row.poc,
            x=cx,
            y=cy,
            list_id=row.list_id,
            ref_poc=row.ref_poc,
            mv_x=row.mv_x / scale,
            mv_y=row.mv_y / scale,
        )
        eligible += 1

        # Remove exact duplicated span records without collapsing L0/L1 or
        # different reference pictures.
        dedup_key = (
            round(cx, 6),
            round(cy, 6),
            str(row.list_id),
            int(row.ref_poc),
            round(sample.mv_x, 6),
            round(sample.mv_y, 6),
        )
        block_seen = seen_keys.setdefault(key, set())
        if dedup_key in block_seen:
            duplicate_count += 1
            continue
        block_seen.add(dedup_key)
        grouped.setdefault(key, []).append(sample)

    selected: List[ProcessingMVSample] = []
    max_before = 0
    for (gx, gy), items in grouped.items():
        max_before = max(max_before, len(items))
        bx = gx * processing_fit_block
        by = gy * processing_fit_block
        block_center_x = bx + (processing_fit_block - 1) * 0.5
        block_center_y = by + (processing_fit_block - 1) * 0.5
        selected.extend(
            _select_spatially_diverse_mv_samples(
                items=items,
                budget=max_samples_per_fit_block,
                block_center_x=block_center_x,
                block_center_y=block_center_y,
            )
        )

    stats = {
        "input_mv_rows": int(len(mv_rows)),
        "eligible_mv_rows": int(eligible),
        "duplicate_mv_rows_removed": int(duplicate_count),
        "sampled_mv_rows": int(len(selected)),
        "mv_fit_blocks": int(len(grouped)),
        "max_mv_rows_in_one_fit_block_before_sampling": int(max_before),
        "max_samples_per_fit_block": int(max_samples_per_fit_block),
    }
    return selected, stats


def make_depth_observations(
    mv_samples: Sequence[ProcessingMVSample],
    relative_cache: Dict[Tuple[int, int], RelativeCameraTransform],
    min_depth: float,
    max_depth: float,
    min_parallax: float,
    max_reproj_error: float,
) -> List[DepthObservation]:
    """Convert only sampled reduced-resolution MVs into depth observations."""
    out: List[DepthObservation] = []

    for sample in mv_samples:
        rel = relative_cache.get((sample.poc, sample.ref_poc))
        if rel is None:
            continue

        solved = solve_depth_closed_form_scalar(
            u=sample.x,
            v=sample.y,
            mv_x=sample.mv_x,
            mv_y=sample.mv_y,
            rel=rel,
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
                poc=sample.poc,
                x=sample.x,
                y=sample.y,
                depth=depth,
                reproj_error=err,
                ref_poc=sample.ref_poc,
                list_id=sample.list_id,
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



LOCAL_CANDIDATE_OFFSETS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "all": ((-1, 0), (0, -1), (-1, -1)),
    "left": ((-1, 0),),
    "top": ((0, -1),),
    "left_top": ((-1, 0), (0, -1)),
    "top_left": ((-1, -1),),
    "left2": ((-2, 0),),
    "top2": ((0, -2),),
    "left2_top2": ((-2, 0), (0, -2)),
}


def parse_local_candidate_names(spec: str) -> List[str]:
    names = [x.strip().lower() for x in spec.split(",") if x.strip()]
    if not names:
        raise ValueError("--local-candidates must contain at least one name")
    unknown = [name for name in names if name not in LOCAL_CANDIDATE_OFFSETS]
    if unknown:
        raise ValueError(
            f"Unknown local candidates: {unknown}; available: "
            f"{sorted(LOCAL_CANDIDATE_OFFSETS)}"
        )
    # Preserve user order while removing duplicates.
    return list(dict.fromkeys(names))


def build_block_fit_jobs_from_offsets(
    observations: List[DepthObservation],
    width: int,
    height: int,
    fit_block: int,
    min_points: int,
    source_offsets: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Build causal fit jobs from an explicit list of source-block offsets."""
    if not observations:
        return []

    block_obs: Dict[Tuple[int, int], List[DepthObservation]] = {}
    for obs in observations:
        gx = int(math.floor(max(obs.x, 0.0) / fit_block))
        gy = int(math.floor(max(obs.y, 0.0) / fit_block))
        block_obs.setdefault((gx, gy), []).append(obs)

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

            selected: List[DepthObservation] = []
            used_keys: List[Tuple[int, int]] = []
            for dx, dy in source_offsets:
                key = (gx + dx, gy + dy)
                if key[0] < 0 or key[1] < 0:
                    continue
                source = block_obs.get(key)
                if source:
                    selected.extend(source)
                    used_keys.append(key)

            if len(selected) < min_points:
                continue

            jobs.append({
                "bx": bx,
                "by": by,
                "bw": bw,
                "bh": bh,
                "cx": cx,
                "cy": cy,
                "source_offsets": [list(v) for v in source_offsets],
                "source_keys": [list(v) for v in used_keys],
                "xs": np.asarray([o.x for o in selected], dtype=np.float64),
                "ys": np.asarray([o.y for o in selected], dtype=np.float64),
                "depths": np.asarray([o.depth for o in selected], dtype=np.float64),
                "errors": np.asarray([o.reproj_error for o in selected], dtype=np.float64),
            })
    return jobs


def fit_and_render_local_candidate(
    observations: List[DepthObservation],
    source_offsets: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    fit_block: int,
    min_points: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
    coordinate_scale: float,
    constant_relative_threshold: float,
    refit_relative_threshold: float,
    determinant_threshold: float,
    enable_refit: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    jobs = build_block_fit_jobs_from_offsets(
        observations=observations,
        width=width,
        height=height,
        fit_block=fit_block,
        min_points=min_points,
        source_offsets=source_offsets,
    )
    coeffs: List[Optional[np.ndarray]] = []
    c_only_count = 0
    for job in jobs:
        coeff = fit_inv_depth_plane_fast_cpu(
            xs=job["xs"],
            ys=job["ys"],
            depths=job["depths"],
            reproj_errors=job["errors"],
            cx=job["cx"],
            cy=job["cy"],
            constant_relative_threshold=constant_relative_threshold,
            refit_relative_threshold=refit_relative_threshold,
            determinant_threshold=determinant_threshold,
            enable_refit=enable_refit,
        )
        coeffs.append(coeff)
        if coeff is not None and abs(float(coeff[0])) <= 1e-15 and abs(float(coeff[1])) <= 1e-15:
            c_only_count += 1

    depth, valid, confidence = render_jobs(
        jobs=jobs,
        coeffs=coeffs,
        width=width,
        height=height,
        min_depth=min_depth,
        max_depth=max_depth,
        max_plane_slope=max_plane_slope,
        min_points=min_points,
        coordinate_scale=coordinate_scale,
    )
    stats = {
        "source_offsets": [list(v) for v in source_offsets],
        "fit_jobs": len(jobs),
        "successful_planes": int(sum(c is not None for c in coeffs)),
        "constant_planes": int(c_only_count),
        "valid_pixel_ratio": float(np.mean(valid)),
    }
    return depth, valid, confidence, stats


def read_gt_depth_yuv420p10le(
    path: str,
    width: int,
    height: int,
    num_frames: int,
    depth_scale_real: float,
    processing_width: int,
    processing_height: int,
    downsample_scale: int,
    downsample_mode: str,
) -> List[np.ndarray]:
    """Read GT Y from YUV420p10le and convert it to metric processing depth."""
    if depth_scale_real <= 0.0:
        raise ValueError("GT depth scale must be positive")
    y_samples = width * height
    uv_samples = (width // 2) * (height // 2)
    frame_bytes = (y_samples + 2 * uv_samples) * 2
    file_size = Path(path).stat().st_size
    available = file_size // frame_bytes
    if available < num_frames:
        raise RuntimeError(
            f"GT depth has only {available} complete frames, requested {num_frames}: {path}"
        )

    frames: List[np.ndarray] = []
    with open(path, "rb") as f:
        for _ in range(num_frames):
            y = np.fromfile(f, dtype="<u2", count=y_samples)
            if y.size != y_samples:
                raise RuntimeError("Unexpected EOF while reading GT Y plane")
            f.seek(2 * uv_samples * 2, 1)
            full = y.reshape(height, width).astype(np.float64) * depth_scale_real
            full[y.reshape(height, width) == 0] = 0.0

            if downsample_mode == "nearest":
                low = resize_nearest_2d(full, processing_height, processing_width).astype(np.float64)
            else:
                s = downsample_scale
                reshaped = full.reshape(processing_height, s, processing_width, s)
                valid = reshaped > 0.0
                if downsample_mode == "average":
                    num = np.sum(np.where(valid, reshaped, 0.0), axis=(1, 3))
                    den = np.sum(valid, axis=(1, 3))
                    low = np.zeros((processing_height, processing_width), dtype=np.float64)
                    good = den > 0
                    low[good] = num[good] / den[good]
                elif downsample_mode == "median":
                    masked = np.where(valid, reshaped, np.nan)
                    with np.errstate(all="ignore"):
                        low = np.nanmedian(masked, axis=(1, 3))
                    low = np.where(np.isfinite(low), low, 0.0)
                else:
                    raise ValueError(f"Unsupported GT downsample mode: {downsample_mode}")
            frames.append(low)
    return frames


def _oracle_error_values(
    pred: np.ndarray,
    gt: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "mae":
        return np.abs(pred - gt)
    if metric == "mse":
        return np.square(pred - gt)
    if metric == "rel_mae":
        return np.abs(pred - gt) / np.maximum(gt, 1e-12)
    if metric == "log_mae":
        return np.abs(np.log(np.maximum(pred, 1e-12)) - np.log(np.maximum(gt, 1e-12)))
    if metric == "inv_mae":
        return np.abs(1.0 / np.maximum(pred, 1e-12) - 1.0 / np.maximum(gt, 1e-12))
    raise ValueError(f"Unsupported oracle metric: {metric}")


def select_gt_oracle_candidates(
    candidates: Sequence[DepthCandidate],
    gt_depth: np.ndarray,
    block_size: int,
    metric: str,
    minimum_coverage: float,
    missing_penalty: float,
    fallback_candidate_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Choose one complete candidate per processing block using GT distortion."""
    if not candidates:
        raise ValueError("Oracle selection requires at least one candidate")
    h, w = gt_depth.shape
    by_name = {candidate.name: i for i, candidate in enumerate(candidates)}
    fallback_idx = by_name.get(fallback_candidate_name, len(candidates) - 1)

    out_depth = np.zeros((h, w), dtype=np.float64)
    out_conf = np.zeros((h, w), dtype=np.float32)
    out_label = np.full((h, w), -1, dtype=np.int32)
    block_choice = np.full(
        ((h + block_size - 1) // block_size, (w + block_size - 1) // block_size),
        -1,
        dtype=np.int32,
    )

    counts = {candidate.name: 0 for candidate in candidates}
    pixels = {candidate.name: 0 for candidate in candidates}
    score_sum = {candidate.name: 0.0 for candidate in candidates}
    scored_blocks = 0
    fallback_blocks = 0

    for gy, by in enumerate(range(0, h, block_size)):
        bh = min(block_size, h - by)
        for gx, bx in enumerate(range(0, w, block_size)):
            bw = min(block_size, w - bx)
            gt_block = gt_depth[by:by + bh, bx:bx + bw]
            gt_valid = np.isfinite(gt_block) & (gt_block > 0.0)
            gt_count = int(np.count_nonzero(gt_valid))

            best_idx = fallback_idx
            best_score = math.inf
            if gt_count > 0:
                for idx, candidate in enumerate(candidates):
                    pred_block = candidate.depth[by:by + bh, bx:bx + bw]
                    conf_block = candidate.confidence[by:by + bh, bx:bx + bw]
                    pred_valid = (
                        gt_valid
                        & np.isfinite(pred_block)
                        & (pred_block > 0.0)
                        & np.isfinite(conf_block)
                        & (conf_block > 0.0)
                    )
                    overlap = int(np.count_nonzero(pred_valid))
                    coverage = overlap / max(gt_count, 1)
                    if overlap == 0 or coverage < minimum_coverage:
                        continue
                    errors = _oracle_error_values(pred_block[pred_valid], gt_block[pred_valid], metric)
                    score = float(np.mean(errors)) / max(coverage, 1e-12)
                    score += (1.0 - coverage) * missing_penalty
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                if math.isfinite(best_score):
                    scored_blocks += 1
                    score_sum[candidates[best_idx].name] += best_score
                else:
                    fallback_blocks += 1
            else:
                fallback_blocks += 1

            selected = candidates[best_idx]
            block_depth = selected.depth[by:by + bh, bx:bx + bw]
            block_conf = selected.confidence[by:by + bh, bx:bx + bw]
            selected_valid = np.isfinite(block_depth) & (block_depth > 0.0) & (block_conf > 0.0)

            # Preserve the selected candidate, but use the configured fallback
            # only for holes so the output remains useful as a causal state.
            fallback = candidates[fallback_idx]
            fallback_depth = fallback.depth[by:by + bh, bx:bx + bw]
            fallback_conf = fallback.confidence[by:by + bh, bx:bx + bw]
            dst_depth = out_depth[by:by + bh, bx:bx + bw]
            dst_conf = out_conf[by:by + bh, bx:bx + bw]
            dst_label = out_label[by:by + bh, bx:bx + bw]
            dst_depth[:] = block_depth
            dst_conf[:] = block_conf
            dst_label[:] = selected.label
            holes = ~selected_valid & np.isfinite(fallback_depth) & (fallback_depth > 0.0) & (fallback_conf > 0.0)
            dst_depth[holes] = fallback_depth[holes]
            dst_conf[holes] = fallback_conf[holes]
            dst_label[holes] = fallback.label
            invalid = ~(np.isfinite(dst_depth) & (dst_depth > 0.0) & (dst_conf > 0.0))
            dst_depth[invalid] = 0.0
            dst_conf[invalid] = 0.0
            dst_label[invalid] = -1

            block_choice[gy, gx] = best_idx
            counts[selected.name] += 1
            pixels[selected.name] += int(bw * bh)

    mean_scores = {
        name: (score_sum[name] / counts[name] if counts[name] > 0 else None)
        for name in counts
    }
    stats = {
        "metric": metric,
        "block_size_processing": int(block_size),
        "minimum_coverage": float(minimum_coverage),
        "missing_penalty": float(missing_penalty),
        "fallback_candidate": candidates[fallback_idx].name,
        "candidate_order": [candidate.name for candidate in candidates],
        "selected_blocks_by_candidate": counts,
        "selected_pixels_by_candidate": pixels,
        "mean_selected_score_by_candidate": mean_scores,
        "gt_scored_blocks": int(scored_blocks),
        "fallback_blocks": int(fallback_blocks),
        "total_blocks": int(block_choice.size),
    }
    return out_depth, out_conf, out_label, stats


def write_candidate_index_yuv420p10le(
    output_path: str,
    index_frames: List[np.ndarray],
    output_width: int,
    output_height: int,
) -> None:
    """Write candidate IDs as Y=ID+1 (0 remains invalid), nearest upsampled."""
    uv = np.full((output_height // 2, output_width // 2), 512, dtype="<u2")
    with open(output_path, "wb") as f:
        for low_idx in index_frames:
            full = resize_nearest_2d(low_idx, output_height, output_width).astype(np.int32)
            y = np.where(full >= 0, full + 1, 0).clip(0, 1023).astype("<u2")
            f.write(np.ascontiguousarray(y).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())


def render_jobs(
    jobs: List[Dict[str, Any]],
    coeffs: List[Optional[np.ndarray]],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_plane_slope: float,
    min_points: int,
    coordinate_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render with row/pixel increments; no per-block meshgrid allocation."""
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

        obs_invz = 1.0 / job["depths"]
        obs_pred = a * (job["xs"] - cx) + b * (job["ys"] - cy) + c
        rel_residual = np.abs(obs_pred - obs_invz) / np.maximum(obs_invz, 1e-12)
        fit_error = float(np.mean(rel_residual))
        reproj = float(np.mean(job["errors"]))
        point_factor = min(1.0, len(job["depths"]) / max(2.0 * min_points, 1.0))
        fit_conf = math.exp(-4.0 * fit_error)
        reproj_conf = 1.0 / (1.0 + reproj * reproj)
        slope_norm = (abs(a) + abs(b)) / max(float(coordinate_scale), 1e-12)
        slope_conf = 1.0 / (1.0 + slope_norm)
        block_conf = float(np.clip(
            point_factor * fit_conf * reproj_conf * slope_conf, 0.02, 1.0
        ))

        inv_row = a * (bx - cx) + b * (by - cy) + c
        for yy in range(bh):
            inv_value = inv_row
            dst_y = by + yy
            for xx in range(bw):
                dst_x = bx + xx
                if math.isfinite(inv_value) and inv_value > 1.0 / max_depth:
                    z = 1.0 / inv_value
                    if min_depth <= z <= max_depth:
                        depth[dst_y, dst_x] = z
                        valid[dst_y, dst_x] = True
                        confidence[dst_y, dst_x] = block_conf
                inv_value += a
            inv_row += b

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
    current_decode_rank: int,
    max_sources: int,
    min_source_quality: float,
    poc_distance_scale: float,
    decode_distance_scale: float,
) -> List[DepthState]:
    """Select only already-decoded depth states.

    Current-picture MV rows are deliberately not inspected here. Source
    selection therefore remains reproducible before the current block MV is
    decoded.
    """
    if max_sources <= 0 or not states:
        return []

    candidates = [s for s in states.values() if s.quality_score >= min_source_quality]

    def rank(s: DepthState) -> float:
        poc_factor = 1.0 + abs(s.poc - target_poc) / max(poc_distance_scale, 1e-6)
        decode_age = max(1, current_decode_rank - s.decode_rank)
        decode_factor = 1.0 + decode_age / max(decode_distance_scale, 1e-6)
        return s.quality_score / (poc_factor * decode_factor)

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
    """Nearest-depth reduction without sorting destination indices."""
    h, w = depth_buffer.shape
    inside = (
        np.isfinite(z) & np.isfinite(conf) & (conf > 0.0)
        & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    )
    if not np.any(inside):
        return

    x = x[inside].astype(np.int64, copy=False)
    y = y[inside].astype(np.int64, copy=False)
    z = z[inside]
    conf = conf[inside]
    idx = y * w + x

    flat_d = depth_buffer.reshape(-1)
    local_min = np.full(flat_d.size, np.inf, dtype=np.float64)
    np.minimum.at(local_min, idx, z)
    touched = np.isfinite(local_min)
    replace = touched & (local_min < flat_d)
    if not np.any(replace):
        return

    # Recover confidence for the winning depth. Exact ties use max confidence.
    winning_sample = np.isclose(z, local_min[idx], rtol=1e-10, atol=1e-12)
    local_conf = np.zeros(flat_d.size, dtype=np.float64)
    np.maximum.at(local_conf, idx[winning_sample], conf[winning_sample])

    flat_c = conf_buffer.reshape(-1)
    flat_s = src_buffer.reshape(-1)
    flat_d[replace] = local_min[replace]
    flat_c[replace] = local_conf[replace]
    flat_s[replace] = source_id


def forward_warp_depth(
    source: DepthState,
    rel: RelativeCameraTransform,
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
            np.full((height, width), -1, dtype=np.int32),
        )

    zs = src_depth[ys, xs].astype(np.float64, copy=False)
    cs = src_conf[ys, xs].astype(np.float64, copy=False) * propagation_conf_decay

    dst_depth = np.full((height, width), np.inf, dtype=np.float64)
    dst_conf = np.zeros((height, width), dtype=np.float32)
    dst_src = np.full((height, width), -1, dtype=np.int32)

    R = rel.R
    t = rel.t

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
        X[0] = z * (u - rel.src_cx) / rel.src_fx
        X[1] = z * (v - rel.src_cy) / rel.src_fy
        X[2] = z * rel.src_z_sign

        Xd = R @ X + t[:, None]
        zd = rel.dst_z_sign * Xd[2]
        front = np.isfinite(zd) & (zd >= min_depth) & (zd <= max_depth)
        if not np.any(front):
            continue

        ud = rel.dst_fx * Xd[0, front] / zd[front] + rel.dst_cx
        vd = rel.dst_fy * Xd[1, front] / zd[front] + rel.dst_cy
        zd = zd[front]
        c = c[front]

        base_x = np.rint(ud).astype(np.int64)
        base_y = np.rint(vd).astype(np.int64)

        for dx, dy in offsets:
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


def _sample_reference_consistency(
    ref_depth: np.ndarray,
    ref_conf: np.ndarray,
    base_x: np.ndarray,
    base_y: np.ndarray,
    predicted_depth: np.ndarray,
    sample_radius: int,
    min_depth: float,
    max_depth: float,
    occlusion_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the most consistent valid depth near each projected reference pixel.

    A candidate point farther than the visible reference surface is treated as
    occluded and does not vote against the candidate.
    """
    h, w = ref_depth.shape
    n = base_x.size
    best_error = np.full(n, np.inf, dtype=np.float64)
    best_conf = np.zeros(n, dtype=np.float64)

    for dy in range(-sample_radius, sample_radius + 1):
        for dx in range(-sample_radius, sample_radius + 1):
            xx = base_x + dx
            yy = base_y + dy
            inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            if not np.any(inside):
                continue

            pos = np.nonzero(inside)[0]
            rd = ref_depth[yy[pos], xx[pos]].astype(np.float64, copy=False)
            rc = ref_conf[yy[pos], xx[pos]].astype(np.float64, copy=False)
            pd = predicted_depth[pos]

            valid = (
                np.isfinite(rd)
                & (rd >= min_depth)
                & (rd <= max_depth)
                & np.isfinite(rc)
                & (rc > 0.0)
                & np.isfinite(pd)
                & (pd >= min_depth)
                & (pd <= max_depth)
            )
            if not np.any(valid):
                continue

            pos = pos[valid]
            rd = rd[valid]
            rc = rc[valid]
            pd = pd[valid]

            # If the candidate point lies behind a nearer visible surface in
            # this reference, the candidate may simply be occluded there.
            comparable = pd <= rd * occlusion_ratio
            if not np.any(comparable):
                continue

            pos = pos[comparable]
            rd = rd[comparable]
            rc = rc[comparable]
            pd = pd[comparable]
            err = np.abs(np.log(np.maximum(pd, 1e-12) / np.maximum(rd, 1e-12)))

            improve = err < best_error[pos]
            if np.any(improve):
                dst = pos[improve]
                best_error[dst] = err[improve]
                best_conf[dst] = rc[improve]

    return best_error, best_conf


def compute_geometry_check_mask(
    candidates: Sequence[DepthCandidate],
    min_depth: float,
    max_depth: float,
    agreement_log_threshold: float,
    skip_agreed_pixels: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return pixels that still require expensive geometry reprojection.

    Pixels with at least two valid candidates whose complete log-depth range is
    inside ``agreement_log_threshold`` are already mutually consistent and may
    bypass geometry checking. Single-candidate pixels remain checked.
    """
    shape = candidates[0].depth.shape
    if not skip_agreed_pixels or len(candidates) < 2:
        return np.ones(shape, dtype=bool), {
            "geometry_skip_enabled": int(skip_agreed_pixels),
            "geometry_skipped_agreed_pixels": 0,
            "geometry_check_pixels": int(shape[0] * shape[1]),
        }

    count = np.zeros(shape, dtype=np.uint8)
    min_log = np.full(shape, np.inf, dtype=np.float64)
    max_log = np.full(shape, -np.inf, dtype=np.float64)

    for candidate in candidates:
        depth = np.asarray(candidate.depth, dtype=np.float64)
        conf = np.asarray(candidate.confidence)
        valid = (
            np.isfinite(depth)
            & (depth >= min_depth)
            & (depth <= max_depth)
            & np.isfinite(conf)
            & (conf > 0.0)
        )
        if not np.any(valid):
            continue
        log_depth = np.log(depth[valid])
        count[valid] = np.minimum(count[valid].astype(np.uint16) + 1, 255).astype(np.uint8)
        min_log[valid] = np.minimum(min_log[valid], log_depth)
        max_log[valid] = np.maximum(max_log[valid], log_depth)

    mutually_agreed = (
        (count >= 2)
        & np.isfinite(min_log)
        & np.isfinite(max_log)
        & ((max_log - min_log) <= max(agreement_log_threshold, 1e-8))
    )
    check_mask = ~mutually_agreed
    return check_mask, {
        "geometry_skip_enabled": 1,
        "geometry_skipped_agreed_pixels": int(np.count_nonzero(mutually_agreed)),
        "geometry_check_pixels": int(np.count_nonzero(check_mask)),
    }


def geometry_gate_candidate_pixelwise(
    candidate: DepthCandidate,
    target_poc: int,
    reference_states: Sequence[DepthState],
    relative_cache: Dict[Tuple[int, int], RelativeCameraTransform],
    geometry_check_mask: Optional[np.ndarray],
    min_depth: float,
    max_depth: float,
    soft_log_threshold: float,
    hard_log_threshold: float,
    min_support: int,
    max_references: int,
    sample_radius: int,
    occlusion_ratio: float,
    unsupported_penalty: float,
    confidence_mix: float,
    self_reference_weight: float,
    chunk_pixels: int,
) -> Tuple[DepthCandidate, Dict[str, Any]]:
    depth = np.asarray(candidate.depth, dtype=np.float64)
    base_conf = np.asarray(candidate.confidence, dtype=np.float64)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
        & np.isfinite(base_conf)
        & (base_conf > 0.0)
    )

    if geometry_check_mask is None:
        check_valid = valid
    else:
        if geometry_check_mask.shape != depth.shape:
            raise ValueError("geometry_check_mask shape mismatch")
        check_valid = valid & geometry_check_mask

    input_valid_count = int(np.count_nonzero(valid))
    checked_count = int(np.count_nonzero(check_valid))
    skipped_count = input_valid_count - checked_count

    if input_valid_count == 0:
        return candidate, {
            "candidate": candidate.name,
            "source_poc": candidate.source_poc,
            "input_valid_pixels": 0,
            "geometry_checked_pixels": 0,
            "geometry_skipped_agreement_pixels": 0,
            "geometry_supported_pixels": 0,
            "geometry_rejected_pixels": 0,
            "output_valid_pixels": 0,
            "mean_geometry_score": 0.0,
        }

    if checked_count == 0:
        return candidate, {
            "candidate": candidate.name,
            "source_poc": candidate.source_poc,
            "reference_pocs": [],
            "input_valid_pixels": input_valid_count,
            "geometry_checked_pixels": 0,
            "geometry_skipped_agreement_pixels": skipped_count,
            "geometry_supported_pixels": 0,
            "external_geometry_supported_pixels": 0,
            "geometry_rejected_pixels": 0,
            "output_valid_pixels": input_valid_count,
            "mean_geometry_score": 1.0,
            "median_best_depth_ratio": 1.0,
        }

    refs = list(reference_states)
    if candidate.source_poc is not None:
        refs.sort(key=lambda state: 0 if state.poc == candidate.source_poc else 1)
    refs = refs[:max_references]

    h, w = depth.shape
    ys, xs = np.nonzero(check_valid)
    zs_all = depth[ys, xs].astype(np.float64, copy=False)
    checked_flat_idx = ys.astype(np.int64) * w + xs.astype(np.int64)
    n = xs.size

    # Allocate only for conflicting/single-candidate pixels that actually need
    # geometry evaluation, rather than for the full frame.
    score_sum = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    best_error = np.full(n, np.inf, dtype=np.float64)
    support_count = np.zeros(n, dtype=np.uint8)
    external_weight_sum = np.zeros(n, dtype=np.float64)
    best_external_error = np.full(n, np.inf, dtype=np.float64)
    external_support_count = np.zeros(n, dtype=np.uint8)

    for ref in refs:
        rel = relative_cache.get((target_poc, ref.poc))
        if rel is None:
            continue
        ref_weight_scale = (
            self_reference_weight
            if candidate.source_poc is not None and ref.poc == candidate.source_poc
            else 1.0
        )

        for start in range(0, n, chunk_pixels):
            end = min(start + chunk_pixels, n)
            u = xs[start:end].astype(np.float64)
            v = ys[start:end].astype(np.float64)
            z = zs_all[start:end]

            X = np.empty((3, end - start), dtype=np.float64)
            X[0] = z * (u - rel.src_cx) / rel.src_fx
            X[1] = z * (v - rel.src_cy) / rel.src_fy
            X[2] = z * rel.src_z_sign

            Xr = rel.R @ X + rel.t[:, None]
            zr = rel.dst_z_sign * Xr[2]
            front = np.isfinite(zr) & (zr >= min_depth) & (zr <= max_depth)
            if not np.any(front):
                continue

            pos = np.nonzero(front)[0]
            ur = rel.dst_fx * Xr[0, pos] / zr[pos] + rel.dst_cx
            vr = rel.dst_fy * Xr[1, pos] / zr[pos] + rel.dst_cy
            bx = np.rint(ur).astype(np.int64)
            by = np.rint(vr).astype(np.int64)

            err, ref_c = _sample_reference_consistency(
                ref_depth=ref.depth,
                ref_conf=ref.confidence,
                base_x=bx,
                base_y=by,
                predicted_depth=zr[pos],
                sample_radius=sample_radius,
                min_depth=min_depth,
                max_depth=max_depth,
                occlusion_ratio=occlusion_ratio,
            )
            comparable = np.isfinite(err) & (ref_c > 0.0)
            if not np.any(comparable):
                continue

            local_dst = start + pos[comparable]
            e = err[comparable]
            rw = ref_c[comparable] * ref_weight_scale
            soft_score = np.exp(-np.square(e / max(soft_log_threshold, 1e-8)))

            score_sum[local_dst] += rw * soft_score
            weight_sum[local_dst] += rw
            best_error[local_dst] = np.minimum(best_error[local_dst], e)
            support_count[local_dst] = np.minimum(
                support_count[local_dst].astype(np.uint16) + 1,
                255,
            ).astype(np.uint8)

            is_self_reference = (
                candidate.source_poc is not None and ref.poc == candidate.source_poc
            )
            if not is_self_reference:
                external_weight_sum[local_dst] += rw
                best_external_error[local_dst] = np.minimum(
                    best_external_error[local_dst], e
                )
                external_support_count[local_dst] = np.minimum(
                    external_support_count[local_dst].astype(np.uint16) + 1,
                    255,
                ).astype(np.uint8)

    geom_score = np.ones(n, dtype=np.float64)
    supported = weight_sum > 0.0
    geom_score[supported] = score_sum[supported] / np.maximum(
        weight_sum[supported], 1e-12
    )

    required_support = max(min_support, 1)
    enough_external = external_support_count >= required_support
    enough_any = support_count >= required_support
    chosen_error = np.where(enough_external, best_external_error, best_error)
    chosen_supported = enough_external | (~enough_external & enough_any)
    hard_bad_local = chosen_supported & (
        (chosen_error > hard_log_threshold) | (geom_score <= 1e-6)
    )

    adjusted_conf_flat = base_conf.reshape(-1).copy()
    mix = float(np.clip(confidence_mix, 0.0, 1.0))
    supported_global = checked_flat_idx[supported]
    adjusted_conf_flat[supported_global] *= (
        (1.0 - mix) + mix * np.clip(geom_score[supported], 0.0, 1.0)
    )
    unsupported_global = checked_flat_idx[~supported]
    adjusted_conf_flat[unsupported_global] *= unsupported_penalty
    hard_bad_global = checked_flat_idx[hard_bad_local]
    adjusted_conf_flat[hard_bad_global] = 0.0

    out_depth_flat = depth.reshape(-1).copy()
    out_depth_flat[adjusted_conf_flat <= 0.0] = 0.0
    out_conf = np.clip(adjusted_conf_flat, 0.0, 1.0).reshape(h, w).astype(np.float32)
    out_depth = out_depth_flat.reshape(h, w)

    output_valid = out_conf > 0.0
    mean_score = float(np.mean(geom_score[supported])) if np.any(supported) else 0.0
    finite_best = best_error[supported & np.isfinite(best_error)]
    median_best_ratio = float(np.exp(np.median(finite_best))) if finite_best.size else 0.0

    gated = DepthCandidate(
        name=candidate.name,
        label=candidate.label,
        source_poc=candidate.source_poc,
        depth=out_depth,
        confidence=out_conf,
    )
    stats = {
        "candidate": candidate.name,
        "source_poc": candidate.source_poc,
        "reference_pocs": [state.poc for state in refs],
        "input_valid_pixels": input_valid_count,
        "geometry_checked_pixels": checked_count,
        "geometry_skipped_agreement_pixels": skipped_count,
        "geometry_supported_pixels": int(np.count_nonzero(supported)),
        "external_geometry_supported_pixels": int(
            np.count_nonzero(external_weight_sum > 0.0)
        ),
        "geometry_rejected_pixels": int(np.count_nonzero(hard_bad_local)),
        "output_valid_pixels": int(np.count_nonzero(output_valid)),
        "mean_geometry_score": mean_score,
        "median_best_depth_ratio": median_best_ratio,
    }
    return gated, stats



def _geometry_representative_points_for_block(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    mode: str,
) -> List[Tuple[int, int]]:
    """Return deterministic representative pixels for one processing block."""
    cx = bx + (bw - 1) // 2
    cy = by + (bh - 1) // 2
    if mode == "center":
        return [(cx, cy)]
    if mode != "center_corners":
        raise ValueError(f"Unsupported geometry representative mode: {mode}")

    points = [
        (cx, cy),
        (bx, by),
        (bx + bw - 1, by),
        (bx, by + bh - 1),
        (bx + bw - 1, by + bh - 1),
    ]
    # Small edge blocks can collapse several points to the same coordinate.
    return list(dict.fromkeys(points))


def geometry_gate_candidate(
    candidate: DepthCandidate,
    target_poc: int,
    reference_states: Sequence[DepthState],
    relative_cache: Dict[Tuple[int, int], RelativeCameraTransform],
    geometry_check_mask: Optional[np.ndarray],
    min_depth: float,
    max_depth: float,
    soft_log_threshold: float,
    hard_log_threshold: float,
    min_support: int,
    max_references: int,
    sample_radius: int,
    occlusion_ratio: float,
    unsupported_penalty: float,
    confidence_mix: float,
    self_reference_weight: float,
    chunk_pixels: int,
    hierarchical_enabled: bool,
    block_size: int,
    representative_mode: str,
    accept_confidence_ratio: float,
    reject_fraction: float,
) -> Tuple[DepthCandidate, Dict[str, Any]]:
    """Hierarchical geometry check with representative-block early decisions.

    1. Mutually agreed pixels have already been removed by
       ``compute_geometry_check_mask``.
    2. For each remaining processing block, evaluate only its center or
       center+corners.
    3. Accept or reject a whole block when representative evidence is clear.
    4. Run the original pixelwise geometry check only for ambiguous blocks.
    """
    if not hierarchical_enabled or block_size <= 1:
        gated, stats = geometry_gate_candidate_pixelwise(
            candidate=candidate,
            target_poc=target_poc,
            reference_states=reference_states,
            relative_cache=relative_cache,
            geometry_check_mask=geometry_check_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            soft_log_threshold=soft_log_threshold,
            hard_log_threshold=hard_log_threshold,
            min_support=min_support,
            max_references=max_references,
            sample_radius=sample_radius,
            occlusion_ratio=occlusion_ratio,
            unsupported_penalty=unsupported_penalty,
            confidence_mix=confidence_mix,
            self_reference_weight=self_reference_weight,
            chunk_pixels=chunk_pixels,
        )
        stats["hierarchical_geometry_enabled"] = False
        return gated, stats

    depth = np.asarray(candidate.depth, dtype=np.float64)
    base_conf = np.asarray(candidate.confidence, dtype=np.float64)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
        & np.isfinite(base_conf)
        & (base_conf > 0.0)
    )
    if geometry_check_mask is None:
        requested = valid.copy()
    else:
        if geometry_check_mask.shape != depth.shape:
            raise ValueError("geometry_check_mask shape mismatch")
        requested = valid & geometry_check_mask

    h, w = depth.shape
    representative_mask = np.zeros((h, w), dtype=bool)
    block_records: List[Tuple[int, int, int, int, List[Tuple[int, int]]]] = []

    for by in range(0, h, block_size):
        bh = min(block_size, h - by)
        for bx in range(0, w, block_size):
            bw = min(block_size, w - bx)
            block_requested = requested[by:by + bh, bx:bx + bw]
            if not np.any(block_requested):
                continue

            points = _geometry_representative_points_for_block(
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                mode=representative_mode,
            )
            usable_points: List[Tuple[int, int]] = []
            for px, py in points:
                if requested[py, px]:
                    usable_points.append((px, py))

            # If a representative location is not itself conflicting/valid,
            # choose the nearest requested location inside the block.
            if not usable_points:
                yy, xx = np.nonzero(block_requested)
                center_x = bx + (bw - 1) * 0.5
                center_y = by + (bh - 1) * 0.5
                best = int(np.argmin(
                    np.square((xx + bx) - center_x)
                    + np.square((yy + by) - center_y)
                ))
                usable_points = [(int(xx[best] + bx), int(yy[best] + by))]

            for px, py in usable_points:
                representative_mask[py, px] = True
            block_records.append((bx, by, bw, bh, usable_points))

    representative_count = int(np.count_nonzero(representative_mask))
    requested_count = int(np.count_nonzero(requested))
    if representative_count == 0:
        stats = {
            "candidate": candidate.name,
            "source_poc": candidate.source_poc,
            "input_valid_pixels": int(np.count_nonzero(valid)),
            "geometry_checked_pixels": 0,
            "geometry_skipped_agreement_pixels": int(np.count_nonzero(valid)) - requested_count,
            "geometry_supported_pixels": 0,
            "geometry_rejected_pixels": 0,
            "output_valid_pixels": int(np.count_nonzero(valid)),
            "mean_geometry_score": 1.0,
            "hierarchical_geometry_enabled": True,
            "representative_pixels_checked": 0,
            "blocks_considered": 0,
            "blocks_accepted": 0,
            "blocks_rejected": 0,
            "blocks_pixel_fallback": 0,
            "pixel_fallback_pixels": 0,
        }
        return candidate, stats

    rep_gated, rep_stats = geometry_gate_candidate_pixelwise(
        candidate=candidate,
        target_poc=target_poc,
        reference_states=reference_states,
        relative_cache=relative_cache,
        geometry_check_mask=representative_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        soft_log_threshold=soft_log_threshold,
        hard_log_threshold=hard_log_threshold,
        min_support=min_support,
        max_references=max_references,
        sample_radius=sample_radius,
        occlusion_ratio=occlusion_ratio,
        unsupported_penalty=unsupported_penalty,
        confidence_mix=confidence_mix,
        self_reference_weight=self_reference_weight,
        chunk_pixels=chunk_pixels,
    )

    rep_conf = np.asarray(rep_gated.confidence, dtype=np.float64)
    out_depth = depth.copy()
    out_conf = base_conf.copy()
    fallback_mask = np.zeros((h, w), dtype=bool)

    accepted_blocks = 0
    rejected_blocks = 0
    fallback_blocks = 0
    accepted_pixels = 0
    rejected_pixels = 0

    accept_threshold = float(np.clip(accept_confidence_ratio, 0.0, 1.0))
    reject_threshold = float(np.clip(reject_fraction, 0.0, 1.0))

    for bx, by, bw, bh, points in block_records:
        orig_values = np.asarray([base_conf[py, px] for px, py in points], dtype=np.float64)
        gated_values = np.asarray([rep_conf[py, px] for px, py in points], dtype=np.float64)
        usable = orig_values > 0.0
        if not np.any(usable):
            continue

        orig_values = orig_values[usable]
        gated_values = gated_values[usable]
        survived = gated_values > 0.0
        rejected_ratio = 1.0 - float(np.count_nonzero(survived)) / float(survived.size)
        confidence_ratio = np.zeros_like(gated_values)
        confidence_ratio[survived] = gated_values[survived] / np.maximum(
            orig_values[survived], 1e-12
        )

        block_requested = requested[by:by + bh, bx:bx + bw]
        block_pixel_count = int(np.count_nonzero(block_requested))

        if survived.size > 0 and np.all(survived) and float(np.mean(confidence_ratio)) >= accept_threshold:
            # Representative evidence is consistently good. Apply its mean
            # attenuation to only the pixels that originally required checking.
            ratio = float(np.clip(np.mean(confidence_ratio), 0.0, 1.0))
            block_conf = out_conf[by:by + bh, bx:bx + bw]
            block_conf[block_requested] *= ratio
            accepted_blocks += 1
            accepted_pixels += block_pixel_count
        elif rejected_ratio >= reject_threshold:
            block_depth = out_depth[by:by + bh, bx:bx + bw]
            block_conf = out_conf[by:by + bh, bx:bx + bw]
            block_depth[block_requested] = 0.0
            block_conf[block_requested] = 0.0
            rejected_blocks += 1
            rejected_pixels += block_pixel_count
        else:
            fallback_mask[by:by + bh, bx:bx + bw] |= block_requested
            fallback_blocks += 1

    fallback_pixels = int(np.count_nonzero(fallback_mask))
    fallback_stats: Dict[str, Any] = {}
    if fallback_pixels > 0:
        fallback_candidate = DepthCandidate(
            name=candidate.name,
            label=candidate.label,
            source_poc=candidate.source_poc,
            depth=out_depth,
            confidence=np.clip(out_conf, 0.0, 1.0).astype(np.float32),
        )
        pixel_gated, fallback_stats = geometry_gate_candidate_pixelwise(
            candidate=fallback_candidate,
            target_poc=target_poc,
            reference_states=reference_states,
            relative_cache=relative_cache,
            geometry_check_mask=fallback_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            soft_log_threshold=soft_log_threshold,
            hard_log_threshold=hard_log_threshold,
            min_support=min_support,
            max_references=max_references,
            sample_radius=sample_radius,
            occlusion_ratio=occlusion_ratio,
            unsupported_penalty=unsupported_penalty,
            confidence_mix=confidence_mix,
            self_reference_weight=self_reference_weight,
            chunk_pixels=chunk_pixels,
        )
        out_depth[fallback_mask] = pixel_gated.depth[fallback_mask]
        out_conf[fallback_mask] = pixel_gated.confidence[fallback_mask]

    out_conf = np.clip(out_conf, 0.0, 1.0)
    out_depth[out_conf <= 0.0] = 0.0
    gated = DepthCandidate(
        name=candidate.name,
        label=candidate.label,
        source_poc=candidate.source_poc,
        depth=out_depth,
        confidence=out_conf.astype(np.float32),
    )

    stats = dict(rep_stats)
    stats.update({
        "hierarchical_geometry_enabled": True,
        "geometry_block_size": int(block_size),
        "geometry_representative_mode": representative_mode,
        "representative_pixels_checked": representative_count,
        "full_requested_geometry_pixels": requested_count,
        "blocks_considered": int(len(block_records)),
        "blocks_accepted": int(accepted_blocks),
        "blocks_rejected": int(rejected_blocks),
        "blocks_pixel_fallback": int(fallback_blocks),
        "block_accepted_pixels": int(accepted_pixels),
        "block_rejected_pixels": int(rejected_pixels),
        "pixel_fallback_pixels": fallback_pixels,
        "pixel_fallback_stats": fallback_stats,
        "output_valid_pixels": int(np.count_nonzero(out_conf > 0.0)),
    })
    return gated, stats

def fuse_depth_candidates(
    candidates: Sequence[DepthCandidate],
    min_depth: float,
    max_depth: float,
    log_depth_threshold: float,
    single_candidate_penalty: float,
    minimum_output_confidence: float,
    preserve_local_valid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if not candidates:
        raise ValueError("At least one depth candidate is required")

    D = np.stack([c.depth for c in candidates], axis=0).astype(np.float64, copy=False)
    C = np.stack([c.confidence for c in candidates], axis=0).astype(np.float64, copy=False)
    labels = np.asarray([c.label for c in candidates], dtype=np.int32)

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

    # Candidate support combines its own confidence with agreement from all
    # other surviving candidates. This performs hard selection first.
    support = C.copy()
    sigma = max(log_depth_threshold, 1e-8)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i == j:
                continue
            pair = valid[i] & valid[j]
            if not np.any(pair):
                continue
            diff = np.abs(logD[i] - logD[j])
            agreement = np.exp(-np.square(diff / sigma))
            support[i] += np.where(pair, C[j] * agreement, 0.0)

    best_idx = np.argmax(support, axis=0)
    best_log = np.take_along_axis(logD, best_idx[None, ...], axis=0)[0]
    best_valid = n_valid > 0

    # Blend only the candidates belonging to the winning inverse-depth cluster.
    agree = valid & (np.abs(logD - best_log[None, ...]) <= sigma)
    weights = C * agree
    weight_sum = np.sum(weights, axis=0)

    invD = np.zeros_like(D)
    invD[valid] = 1.0 / D[valid]
    fused_inv = np.sum(weights * invD, axis=0) / np.maximum(weight_sum, 1e-12)

    shape = candidates[0].depth.shape
    out_depth = np.zeros(shape, dtype=np.float64)
    usable = best_valid & np.isfinite(fused_inv) & (fused_inv > 0.0)
    out_depth[usable] = 1.0 / fused_inv[usable]

    agreeing_count = np.sum(agree, axis=0)
    max_conf = np.max(np.where(agree, C, 0.0), axis=0)
    mean_conf = weight_sum / np.maximum(agreeing_count, 1)
    out_conf = np.clip(0.55 * max_conf + 0.45 * mean_conf, 0.0, 1.0)
    out_conf[agreeing_count <= 1] *= single_candidate_penalty

    selected_label = labels[best_idx]

    # For good but sparse anchor pictures, preserve locally derived structure
    # exactly and use propagation only to fill local holes.
    if preserve_local_valid:
        local_valid = valid[0]
        out_depth[local_valid] = D[0][local_valid]
        out_conf[local_valid] = C[0][local_valid]
        selected_label[local_valid] = labels[0]
        usable |= local_valid

    usable &= out_conf >= minimum_output_confidence
    out_depth[~usable] = 0.0
    out_conf[~usable] = 0.0
    selected_label[~usable] = -1

    per_candidate = {
        c.name: int(np.count_nonzero(usable & (selected_label == c.label)))
        for c in candidates
    }
    stats: Dict[str, Any] = {
        "fused_valid_pixels": int(np.count_nonzero(usable)),
        "multi_supported_pixels": int(np.count_nonzero(usable & (agreeing_count >= 2))),
        "single_supported_pixels": int(np.count_nonzero(usable & (agreeing_count == 1))),
        "selected_local_pixels": int(np.count_nonzero(usable & (selected_label == 0))),
        "selected_propagated_pixels": int(np.count_nonzero(usable & (selected_label > 0))),
        "anchor_local_preserved": bool(preserve_local_valid),
        "selected_pixels_by_candidate": per_candidate,
    }
    return out_depth, out_conf.astype(np.float32), selected_label, stats



def fill_fused_depth_holes(
    depth: np.ndarray,
    confidence: np.ndarray,
    selected_label: np.ndarray,
    radius: int,
    confidence_decay: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Fill only invalid fused pixels once, choosing the highest-confidence neighbor."""
    if radius <= 0:
        return depth, confidence, selected_label, {"hole_pixels_filled": 0}

    src_depth = np.asarray(depth, dtype=np.float64)
    src_conf = np.asarray(confidence, dtype=np.float32)
    invalid = (src_depth <= 0.0) | (src_conf <= 0.0)
    if not np.any(invalid):
        return depth, confidence, selected_label, {"hole_pixels_filled": 0}

    h, w = src_depth.shape
    best_depth = np.zeros_like(src_depth)
    best_conf = np.zeros_like(src_conf)
    best_label = np.full_like(selected_label, -1)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y_dst0 = max(0, -dy)
            y_dst1 = min(h, h - dy)
            x_dst0 = max(0, -dx)
            x_dst1 = min(w, w - dx)
            if y_dst0 >= y_dst1 or x_dst0 >= x_dst1:
                continue
            y_src0, y_src1 = y_dst0 + dy, y_dst1 + dy
            x_src0, x_src1 = x_dst0 + dx, x_dst1 + dx

            candidate_conf = src_conf[y_src0:y_src1, x_src0:x_src1]
            candidate_depth = src_depth[y_src0:y_src1, x_src0:x_src1]
            dst_invalid = invalid[y_dst0:y_dst1, x_dst0:x_dst1]
            current_best = best_conf[y_dst0:y_dst1, x_dst0:x_dst1]
            choose = dst_invalid & (candidate_depth > 0.0) & (candidate_conf > current_best)
            if not np.any(choose):
                continue
            bd = best_depth[y_dst0:y_dst1, x_dst0:x_dst1]
            bc = best_conf[y_dst0:y_dst1, x_dst0:x_dst1]
            bl = best_label[y_dst0:y_dst1, x_dst0:x_dst1]
            bd[choose] = candidate_depth[choose]
            bc[choose] = candidate_conf[choose]
            bl[choose] = selected_label[y_src0:y_src1, x_src0:x_src1][choose]

    fill = invalid & (best_depth > 0.0) & (best_conf > 0.0)
    out_depth = src_depth.copy()
    out_conf = src_conf.copy()
    out_label = selected_label.copy()
    out_depth[fill] = best_depth[fill]
    out_conf[fill] = np.clip(best_conf[fill] * confidence_decay, 0.0, 1.0)
    out_label[fill] = best_label[fill]
    return out_depth, out_conf, out_label, {
        "hole_pixels_filled": int(np.count_nonzero(fill))
    }


# ============================================================
# Output / final upsampling
# ============================================================

def resize_nearest_2d(src: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    src = np.asarray(src)
    in_height, in_width = src.shape
    if (in_height, in_width) == (out_height, out_width):
        return src.copy()
    if cv2 is not None:
        return cv2.resize(
            src, (out_width, out_height), interpolation=cv2.INTER_NEAREST
        )

    y_idx = np.minimum(
        (np.arange(out_height, dtype=np.int64) * in_height) // out_height,
        in_height - 1,
    )
    x_idx = np.minimum(
        (np.arange(out_width, dtype=np.int64) * in_width) // out_width,
        in_width - 1,
    )
    return src[y_idx[:, None], x_idx[None, :]]


def resize_bilinear_2d(src: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Dependency-free bilinear resize using half-pixel center mapping."""
    src = np.asarray(src, dtype=np.float64)
    in_height, in_width = src.shape
    if (in_height, in_width) == (out_height, out_width):
        return src.copy()
    if cv2 is not None:
        return cv2.resize(
            src, (out_width, out_height), interpolation=cv2.INTER_LINEAR
        )

    src_y = (np.arange(out_height, dtype=np.float64) + 0.5) * (
        in_height / float(out_height)
    ) - 0.5
    src_x = (np.arange(out_width, dtype=np.float64) + 0.5) * (
        in_width / float(out_width)
    ) - 0.5

    y0_raw = np.floor(src_y).astype(np.int64)
    x0_raw = np.floor(src_x).astype(np.int64)
    y1_raw = y0_raw + 1
    x1_raw = x0_raw + 1

    wy = src_y - y0_raw
    wx = src_x - x0_raw

    y0 = np.clip(y0_raw, 0, in_height - 1)
    y1 = np.clip(y1_raw, 0, in_height - 1)
    x0 = np.clip(x0_raw, 0, in_width - 1)
    x1 = np.clip(x1_raw, 0, in_width - 1)

    v00 = src[y0[:, None], x0[None, :]]
    v01 = src[y0[:, None], x1[None, :]]
    v10 = src[y1[:, None], x0[None, :]]
    v11 = src[y1[:, None], x1[None, :]]

    top = v00 * (1.0 - wx[None, :]) + v01 * wx[None, :]
    bottom = v10 * (1.0 - wx[None, :]) + v11 * wx[None, :]
    return top * (1.0 - wy[:, None]) + bottom * wy[:, None]


def upsample_depth_map(
    depth: np.ndarray,
    out_height: int,
    out_width: int,
    mode: str,
    invalid_aware_bilinear: bool,
) -> np.ndarray:
    if mode == "nearest":
        return resize_nearest_2d(depth, out_height, out_width).astype(
            np.float64, copy=False
        )
    if mode != "bilinear":
        raise ValueError(f"Unsupported upsample mode: {mode}")

    depth64 = np.asarray(depth, dtype=np.float64)
    if not invalid_aware_bilinear:
        return resize_bilinear_2d(depth64, out_height, out_width)

    valid = (
        np.isfinite(depth64) & (depth64 > 0.0)
    ).astype(np.float64)
    numerator = resize_bilinear_2d(depth64 * valid, out_height, out_width)
    denominator = resize_bilinear_2d(valid, out_height, out_width)
    out = np.zeros((out_height, out_width), dtype=np.float64)
    good = denominator > 1e-8
    out[good] = numerator[good] / denominator[good]
    return out


def upsample_confidence_map(
    confidence: np.ndarray,
    out_height: int,
    out_width: int,
    mode: str,
) -> np.ndarray:
    if mode == "nearest":
        out = resize_nearest_2d(confidence, out_height, out_width)
    elif mode == "bilinear":
        out = resize_bilinear_2d(confidence, out_height, out_width)
    else:
        raise ValueError(f"Unsupported upsample mode: {mode}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def write_depth_yuv420p10le(
    output_path: str,
    depth_frames: List[np.ndarray],
    depth_scale_real: float,
    output_width: int,
    output_height: int,
    upsample_mode: str,
    invalid_aware_bilinear: bool,
    max_code: int = 1023,
) -> None:
    if depth_scale_real <= 0.0:
        raise ValueError("depth_scale_real must be positive")
    if output_width % 2 or output_height % 2:
        raise ValueError("YUV420 output requires even width/height")

    uv = np.full((output_height // 2, output_width // 2), 512, dtype="<u2")

    with open(output_path, "wb") as f:
        for low_depth in depth_frames:
            depth = upsample_depth_map(
                low_depth,
                out_height=output_height,
                out_width=output_width,
                mode=upsample_mode,
                invalid_aware_bilinear=invalid_aware_bilinear,
            )
            y = np.zeros((output_height, output_width), dtype=np.float64)
            valid = np.isfinite(depth) & (depth > 0.0)
            y[valid] = np.rint(depth[valid] / depth_scale_real)
            y = np.clip(y, 0, max_code).astype("<u2")

            f.write(np.ascontiguousarray(y).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())


def write_confidence_yuv420p10le(
    output_path: str,
    confidence_frames: List[np.ndarray],
    output_width: int,
    output_height: int,
    upsample_mode: str,
) -> None:
    if output_width % 2 or output_height % 2:
        raise ValueError("YUV420 output requires even width/height")

    uv = np.full((output_height // 2, output_width // 2), 512, dtype="<u2")
    with open(output_path, "wb") as f:
        for low_conf in confidence_frames:
            conf = upsample_confidence_map(
                low_conf,
                out_height=output_height,
                out_width=output_width,
                mode=upsample_mode,
            )
            y = np.rint(conf * 1023.0).astype("<u2")
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

def parse_poc_set(spec: str, num_frames: int) -> set[int]:
    value = spec.strip()
    if not value:
        return set()
    try:
        pocs = {int(x.strip()) for x in value.split(",") if x.strip()}
    except ValueError as exc:
        raise ValueError("POC list must be comma-separated integers") from exc
    return {p for p in pocs if 0 <= p < num_frames}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Causal decoded-depth propagation with multiple local/temporal "
            "candidates and GT-oracle block selection."
        )
    )
    ap.add_argument("--mv-csv", required=True)
    ap.add_argument("--camera-param", required=True)
    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--gt-depth-yuv", required=True, help="GT depth YUV420p10le used only for encoder-side oracle candidate selection.")
    ap.add_argument("--gt-depth-scale-real", type=float, default=None, help="GT code-to-metric scale. Default: --depth-scale-real/header scale.")
    ap.add_argument("--gt-downsample-mode", choices=("nearest", "average", "median"), default="median")
    ap.add_argument("--out-candidate-index-yuv", default=None, help="Optional YUV containing selected candidate index+1 per oracle block.")

    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument(
        "--downsample-scale",
        type=int,
        default=4,
        help=(
            "Process depth at 1/scale in each dimension. For FHD, scale=4 "
            "uses 480x270 processing and writes 1920x1080 output."
        ),
    )
    ap.add_argument(
        "--upsample-mode",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="Final reduced-depth to full-resolution interpolation mode.",
    )
    ap.add_argument(
        "--bilinear-invalid-aware",
        dest="bilinear_invalid_aware",
        action="store_true",
        default=True,
        help="Ignore zero/invalid depth samples during bilinear upsampling.",
    )
    ap.add_argument(
        "--no-bilinear-invalid-aware",
        dest="bilinear_invalid_aware",
        action="store_false",
        help="Apply ordinary bilinear interpolation including zero depth.",
    )
    ap.add_argument(
        "--fit-block",
        type=int,
        default=16,
        help="Fit-block size in original full-resolution pixels.",
    )
    ap.add_argument("--neighborhood", type=int, default=0)
    ap.add_argument(
        "--local-candidates",
        default="all,left,top,left_top,top_left,left2,top2,left2_top2",
        help="Comma-separated local candidate names. Available: all,left,top,left_top,top_left,left2,top2,left2_top2.",
    )
    ap.add_argument("--oracle-block-size", type=int, default=0, help="GT selection block size in processing pixels. 0 uses processing fit-block.")
    ap.add_argument("--oracle-metric", choices=("mae", "mse", "rel_mae", "log_mae", "inv_mae"), default="rel_mae")
    ap.add_argument("--oracle-minimum-coverage", type=float, default=0.75)
    ap.add_argument("--oracle-missing-penalty", type=float, default=1.0)
    ap.add_argument("--oracle-fallback-candidate", default="fused_final")
    ap.add_argument("--oracle-state-mode", choices=("oracle", "fused"), default="oracle", help="Depth state registered for later decoded pictures.")
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument(
        "--max-mv-samples-per-fit-block",
        type=int,
        default=4,
        help=(
            "Maximum sampled MV rows per reduced-resolution fit block before "
            "depth solving. 0 keeps all rows."
        ),
    )

    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--max-depth", type=float, default=1e6)
    ap.add_argument("--min-parallax", type=float, default=1e-6)
    ap.add_argument(
        "--max-reproj-error",
        type=float,
        default=1.5,
        help="Maximum reprojection error in original full-resolution pixels.",
    )

    ap.add_argument("--irls-iters", type=int, default=3)
    ap.add_argument("--huber-delta", type=float, default=1.5)
    ap.add_argument("--min-condition", type=float, default=1e-8)
    ap.add_argument(
        "--max-plane-slope",
        type=float,
        default=1.0,
        help="Maximum inverse-depth slope per original full-resolution pixel.",
    )

    ap.add_argument(
        "--constant-plane-relative-threshold", type=float, default=0.04,
        help="Use c-only when weighted relative inverse-depth error is below this value.",
    )
    ap.add_argument(
        "--plane-refit-relative-threshold", type=float, default=0.08,
        help="Samples above this relative inverse-depth error are removed for one optional refit.",
    )
    ap.add_argument(
        "--plane-determinant-threshold", type=float, default=1e-10,
        help="Relative determinant threshold for rejecting an unstable 3x3 normal equation.",
    )
    ap.add_argument(
        "--plane-refit", dest="plane_refit", action="store_true", default=True,
        help="Perform at most one inlier-only plane refit.",
    )
    ap.add_argument("--no-plane-refit", dest="plane_refit", action="store_false")

    ap.add_argument(
        "--decode-order",
        default="ra",
        help=(
            "ra/auto for hierarchical RA order, display for 0..N-1, or an "
            "explicit comma-separated POC list."
        ),
    )
    ap.add_argument("--max-propagation-sources", type=int, default=2)
    ap.add_argument("--min-source-quality", type=float, default=0.01)
    ap.add_argument("--source-poc-distance-scale", type=float, default=16.0)
    ap.add_argument("--source-decode-distance-scale", type=float, default=8.0)
    ap.add_argument(
        "--propagation-splat-radius",
        type=int,
        default=0,
        help="Splat radius in reduced-resolution processing pixels; decoder-oriented default is 0.",
    )
    ap.add_argument("--propagation-chunk-pixels", type=int, default=262144)
    ap.add_argument(
        "--post-fusion-hole-fill-radius", type=int, default=1,
        help="Fill invalid fused pixels once using neighbors in this processing-pixel radius.",
    )
    ap.add_argument(
        "--post-fusion-hole-fill-confidence-decay", type=float, default=0.75,
        help="Confidence multiplier for post-fusion hole-filled pixels.",
    )
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
        help="Candidates inside this multiplicative depth ratio may blend.",
    )
    ap.add_argument(
        "--geometry-consistency-ratio",
        type=float,
        default=1.15,
        help="Soft ratio used when scoring a candidate against decoded depth.",
    )
    ap.add_argument(
        "--geometry-hard-ratio",
        type=float,
        default=1.35,
        help="Reject a supported candidate if every geometry check exceeds this ratio.",
    )
    ap.add_argument("--geometry-min-support", type=int, default=1)
    ap.add_argument("--geometry-max-references", type=int, default=3)
    ap.add_argument(
        "--geometry-sample-radius",
        type=int,
        default=1,
        help="Geometry search radius in reduced-resolution processing pixels.",
    )
    ap.add_argument(
        "--geometry-occlusion-ratio",
        type=float,
        default=1.03,
        help="A candidate behind a nearer reference surface is treated as occluded.",
    )
    ap.add_argument("--geometry-unsupported-penalty", type=float, default=0.90)
    ap.add_argument("--geometry-confidence-mix", type=float, default=0.75)
    ap.add_argument("--geometry-self-reference-weight", type=float, default=0.50)
    ap.add_argument("--geometry-chunk-pixels", type=int, default=262144)
    ap.add_argument(
        "--geometry-skip-agreed-pixels",
        dest="geometry_skip_agreed_pixels",
        action="store_true",
        default=True,
        help="Skip reprojection where at least two candidates already agree.",
    )
    ap.add_argument(
        "--no-geometry-skip-agreed-pixels",
        dest="geometry_skip_agreed_pixels",
        action="store_false",
    )
    ap.add_argument(
        "--geometry-skip-ratio",
        type=float,
        default=None,
        help=(
            "Multiplicative full candidate range considered already agreed. "
            "Default: --depth-consistency-ratio."
        ),
    )
    ap.add_argument(
        "--hierarchical-geometry",
        dest="hierarchical_geometry",
        action="store_true",
        default=True,
        help="Use block representative checks and pixelwise fallback only for ambiguous blocks.",
    )
    ap.add_argument(
        "--no-hierarchical-geometry",
        dest="hierarchical_geometry",
        action="store_false",
    )
    ap.add_argument(
        "--geometry-block-size",
        type=int,
        default=0,
        help="Geometry decision block size in processing pixels. 0 uses the processing fit-block size.",
    )
    ap.add_argument(
        "--geometry-representative-mode",
        choices=("center", "center_corners"),
        default="center_corners",
        help="Representative positions used before pixelwise fallback.",
    )
    ap.add_argument(
        "--geometry-block-accept-confidence-ratio",
        type=float,
        default=0.75,
        help="Accept a block when every representative survives and mean confidence ratio is at least this value.",
    )
    ap.add_argument(
        "--geometry-block-reject-fraction",
        type=float,
        default=0.80,
        help="Reject a block when at least this fraction of representatives is rejected.",
    )

    ap.add_argument(
        "--anchor-pocs",
        default="8,16,24",
        help="Good but sparse POCs whose local valid depth is preserved exactly.",
    )
    ap.add_argument(
        "--anchor-hole-fill-only",
        dest="anchor_hole_fill_only",
        action="store_true",
        default=True,
        help="For anchor POCs, use propagation only where local depth is invalid.",
    )
    ap.add_argument(
        "--no-anchor-hole-fill-only",
        dest="anchor_hole_fill_only",
        action="store_false",
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
    ap.add_argument("--profile", action="store_true", help="Print per-stage timings.")

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0 or args.num_frames <= 0:
        raise ValueError("Invalid dimensions/frame count")
    if args.downsample_scale <= 0:
        raise ValueError("--downsample-scale must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if (
        args.width % args.downsample_scale != 0
        or args.height % args.downsample_scale != 0
    ):
        raise ValueError(
            "For exact camera/resize alignment, width and height must be "
            "divisible by --downsample-scale"
        )
    if args.fit_block <= 0:
        raise ValueError("--fit-block must be positive")
    if args.oracle_block_size < 0:
        raise ValueError("--oracle-block-size must be >= 0")
    if not (0.0 < args.oracle_minimum_coverage <= 1.0):
        raise ValueError("--oracle-minimum-coverage must be in (0,1]")
    if args.oracle_missing_penalty < 0.0:
        raise ValueError("--oracle-missing-penalty must be >= 0")
    if args.min_points < 3:
        raise ValueError("--min-points must be >= 3")
    if args.max_mv_samples_per_fit_block < 0:
        raise ValueError("--max-mv-samples-per-fit-block must be >= 0")
    if args.max_propagation_sources < 0:
        raise ValueError("--max-propagation-sources must be >= 0")
    if args.propagation_splat_radius < 0:
        raise ValueError("--propagation-splat-radius must be >= 0")
    if args.post_fusion_hole_fill_radius < 0:
        raise ValueError("--post-fusion-hole-fill-radius must be >= 0")
    if not (0.0 <= args.post_fusion_hole_fill_confidence_decay <= 1.0):
        raise ValueError("--post-fusion-hole-fill-confidence-decay must be in [0,1]")
    if args.geometry_sample_radius < 0:
        raise ValueError("--geometry-sample-radius must be >= 0")
    if args.geometry_max_references < 0:
        raise ValueError("--geometry-max-references must be >= 0")
    if args.depth_consistency_ratio <= 1.0:
        raise ValueError("--depth-consistency-ratio must be > 1")
    if args.geometry_consistency_ratio <= 1.0:
        raise ValueError("--geometry-consistency-ratio must be > 1")
    if args.geometry_hard_ratio < args.geometry_consistency_ratio:
        raise ValueError("--geometry-hard-ratio must be >= --geometry-consistency-ratio")
    if args.geometry_occlusion_ratio < 1.0:
        raise ValueError("--geometry-occlusion-ratio must be >= 1")
    if args.geometry_skip_ratio is not None and args.geometry_skip_ratio <= 1.0:
        raise ValueError("--geometry-skip-ratio must be > 1")
    if args.geometry_block_size < 0:
        raise ValueError("--geometry-block-size must be >= 0")
    if not (0.0 <= args.geometry_block_accept_confidence_ratio <= 1.0):
        raise ValueError("--geometry-block-accept-confidence-ratio must be in [0,1]")
    if not (0.0 <= args.geometry_block_reject_fraction <= 1.0):
        raise ValueError("--geometry-block-reject-fraction must be in [0,1]")

    processing_width = (args.width + args.downsample_scale - 1) // args.downsample_scale
    processing_height = (args.height + args.downsample_scale - 1) // args.downsample_scale
    processing_fit_block = max(
        1, (args.fit_block + args.downsample_scale - 1) // args.downsample_scale
    )
    geometry_block_size = (
        processing_fit_block if args.geometry_block_size == 0
        else args.geometry_block_size
    )
    oracle_block_size = (
        processing_fit_block if args.oracle_block_size == 0
        else args.oracle_block_size
    )
    local_candidate_names = parse_local_candidate_names(args.local_candidates)

    # Quantities expressed in image pixels must move to the reduced coordinate
    # system. Plane slope is inverse-depth per pixel, so it scales oppositely.
    processing_min_parallax = args.min_parallax / float(args.downsample_scale)
    processing_max_reproj_error = args.max_reproj_error / float(
        args.downsample_scale
    )
    processing_max_plane_slope = args.max_plane_slope * float(
        args.downsample_scale
    )

    camera_json = load_camera_jsonl(args.camera_param)
    full_cameras = build_camera_lookup(camera_json)
    cameras = scale_camera_lookup(full_cameras, args.downsample_scale)
    relative_cache = build_relative_transform_cache(cameras)

    header = camera_json["header"]
    if args.depth_scale_real is None:
        precision = float(header.get("depth_scale_precision", 1.0))
        if precision <= 0.0:
            raise ValueError("Invalid depth_scale_precision")
        depth_scale_real = float(header["depth_scale"]) / precision
    else:
        depth_scale_real = float(args.depth_scale_real)

    gt_depth_scale_real = (
        depth_scale_real if args.gt_depth_scale_real is None
        else float(args.gt_depth_scale_real)
    )
    gt_depth_frames = read_gt_depth_yuv420p10le(
        path=args.gt_depth_yuv,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        depth_scale_real=gt_depth_scale_real,
        processing_width=processing_width,
        processing_height=processing_height,
        downsample_scale=args.downsample_scale,
        downsample_mode=args.gt_downsample_mode,
    )

    mv_by_frame = parse_mv_csv(args.mv_csv, args.num_frames)
    decode_order = parse_decode_order(args.decode_order, args.num_frames)
    anchor_pocs = parse_poc_set(args.anchor_pocs, args.num_frames)

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

    print(f"device                    : {device}")
    print(f"full resolution           : {args.width}x{args.height}")
    print(
        f"processing resolution     : {processing_width}x{processing_height} "
        f"(1/{args.downsample_scale} each axis)"
    )
    print(
        f"fit block                 : full {args.fit_block}x{args.fit_block}, "
        f"processing {processing_fit_block}x{processing_fit_block}"
    )
    print(f"final upsample            : {args.upsample_mode}")
    print(f"resize backend            : {'opencv' if cv2 is not None else 'numpy'}")
    if args.upsample_mode == "bilinear":
        print(f"bilinear invalid aware    : {args.bilinear_invalid_aware}")
    print(f"decode order              : {decode_order}")
    print(f"anchor POCs               : {sorted(anchor_pocs)}")
    print("current block MV          : disabled")
    print("source selection MV       : current-picture MV not used")
    print("propagation causality     : decoded pictures only")
    print("candidate processing      : geometry gate -> selective blend")
    print(f"max propagation sources   : {args.max_propagation_sources}")
    print(f"propagation splat radius  : {args.propagation_splat_radius}")
    print(f"post-fusion hole fill     : radius {args.post_fusion_hole_fill_radius}")
    print(f"local fit                 : c-only early exit + 1 WLS + optional refit")
    print(f"MV samples / fit block    : {args.max_mv_samples_per_fit_block or 'all'}")
    print(f"geometry skip agreed      : {args.geometry_skip_agreed_pixels}")
    print(f"hierarchical geometry     : {args.hierarchical_geometry}")
    print(
        f"geometry block check     : {geometry_block_size}x{geometry_block_size}, "
        f"{args.geometry_representative_mode}"
    )
    print(f"relative transform cache  : {len(relative_cache)} pairs")
    print(f"depth scale real          : {depth_scale_real:.12g}")
    print(f"GT depth                  : {args.gt_depth_yuv}")
    print(f"GT downsample             : {args.gt_downsample_mode}")
    print(f"local candidate set       : {local_candidate_names}")
    print(f"oracle block/metric       : {oracle_block_size}x{oracle_block_size}, {args.oracle_metric}")
    print(f"oracle state mode         : {args.oracle_state_mode}")

    depth_frames: List[Optional[np.ndarray]] = [None] * args.num_frames
    confidence_frames: List[Optional[np.ndarray]] = [None] * args.num_frames
    candidate_index_frames: List[Optional[np.ndarray]] = [None] * args.num_frames
    state_bank: Dict[int, DepthState] = {}
    frame_stats_by_poc: Dict[int, Dict[str, Any]] = {}

    blend_log_threshold = math.log(args.depth_consistency_ratio)
    geometry_soft_log = math.log(args.geometry_consistency_ratio)
    geometry_hard_log = math.log(args.geometry_hard_ratio)
    geometry_skip_ratio = (
        args.depth_consistency_ratio
        if args.geometry_skip_ratio is None
        else args.geometry_skip_ratio
    )
    geometry_skip_log = math.log(geometry_skip_ratio)
    cumulative_timing: Dict[str, float] = {
        "mv_sampling": 0.0,
        "mv_depth": 0.0,
        "local_fit_render": 0.0,
        "warp": 0.0,
        "geometry": 0.0,
        "fusion": 0.0,
        "total": 0.0,
    }

    for decode_rank, poc in enumerate(decode_order):
        frame_start = perf_counter()

        stage_start = perf_counter()
        sampled_mvs, mv_sampling_stats = sample_mv_rows_for_processing(
            mv_rows=mv_by_frame[poc],
            full_width=args.width,
            full_height=args.height,
            processing_width=processing_width,
            processing_height=processing_height,
            downsample_scale=args.downsample_scale,
            processing_fit_block=processing_fit_block,
            max_samples_per_fit_block=args.max_mv_samples_per_fit_block,
        )
        t_mv_sampling = perf_counter() - stage_start

        stage_start = perf_counter()
        observations = make_depth_observations(
            mv_samples=sampled_mvs,
            relative_cache=relative_cache,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            min_parallax=processing_min_parallax,
            max_reproj_error=processing_max_reproj_error,
        )
        t_mv_depth = perf_counter() - stage_start

        stage_start = perf_counter()
        local_candidate_maps: Dict[str, DepthCandidate] = {}
        local_candidate_stats: Dict[str, Dict[str, Any]] = {}
        for local_index, local_name in enumerate(local_candidate_names):
            local_depth_i, local_valid_i, local_conf_i, local_stats_i = (
                fit_and_render_local_candidate(
                    observations=observations,
                    source_offsets=LOCAL_CANDIDATE_OFFSETS[local_name],
                    width=processing_width,
                    height=processing_height,
                    fit_block=processing_fit_block,
                    min_points=args.min_points,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    max_plane_slope=processing_max_plane_slope,
                    coordinate_scale=float(args.downsample_scale),
                    constant_relative_threshold=args.constant_plane_relative_threshold,
                    refit_relative_threshold=args.plane_refit_relative_threshold,
                    determinant_threshold=args.plane_determinant_threshold,
                    enable_refit=args.plane_refit,
                )
            )
            candidate = DepthCandidate(
                name=f"local_{local_name}",
                label=local_index,
                source_poc=None,
                depth=local_depth_i,
                confidence=np.clip(
                    local_conf_i * args.local_confidence_boost, 0.0, 1.0
                ).astype(np.float32),
            )
            local_candidate_maps[local_name] = candidate
            local_candidate_stats[local_name] = local_stats_i

        primary_local_name = "all" if "all" in local_candidate_maps else local_candidate_names[0]
        primary_local = local_candidate_maps[primary_local_name]
        local_depth = primary_local.depth
        local_conf = primary_local.confidence
        local_valid = local_conf > 0.0
        local_ratio = float(np.mean(local_valid))
        t_local = perf_counter() - stage_start

        stage_start = perf_counter()
        selected_sources = select_propagation_sources(
            states=state_bank,
            target_poc=poc,
            current_decode_rank=decode_rank,
            max_sources=args.max_propagation_sources,
            min_source_quality=args.min_source_quality,
            poc_distance_scale=args.source_poc_distance_scale,
            decode_distance_scale=args.source_decode_distance_scale,
        )

        raw_candidates: List[DepthCandidate] = [primary_local]
        source_stats: List[Dict[str, Any]] = []

        for source in selected_sources:
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
                rel=relative_cache[(source.poc, poc)],
                width=processing_width,
                height=processing_height,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                splat_radius=args.propagation_splat_radius,
                propagation_conf_decay=decay,
                chunk_pixels=args.propagation_chunk_pixels,
                source_id=source.poc,
            )
            warp_valid = warp_depth > 0.0
            raw_candidates.append(
                DepthCandidate(
                    name=f"propagated_poc_{source.poc}",
                    label=len(local_candidate_names) + len(raw_candidates) - 1,
                    source_poc=source.poc,
                    depth=warp_depth,
                    confidence=warp_conf,
                )
            )
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

        t_warp = perf_counter() - stage_start
        preserve_local = args.anchor_hole_fill_only and poc in anchor_pocs

        stage_start = perf_counter()
        geometry_check_mask, geometry_skip_stats = compute_geometry_check_mask(
            candidates=raw_candidates,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            agreement_log_threshold=geometry_skip_log,
            skip_agreed_pixels=args.geometry_skip_agreed_pixels,
        )

        gated_candidates: List[DepthCandidate] = []
        geometry_stats: List[Dict[str, Any]] = []
        for candidate_index, candidate in enumerate(raw_candidates):
            if preserve_local and candidate_index == 0:
                # The local anchor predictor is copied exactly, so avoid doing
                # geometry work that would be overwritten immediately.
                gated_candidates.append(candidate)
                local_valid_count = int(np.count_nonzero(candidate.confidence > 0.0))
                geometry_stats.append(
                    {
                        "candidate": candidate.name,
                        "source_poc": candidate.source_poc,
                        "reference_pocs": [],
                        "input_valid_pixels": local_valid_count,
                        "geometry_checked_pixels": 0,
                        "geometry_skipped_agreement_pixels": local_valid_count,
                        "geometry_supported_pixels": 0,
                        "external_geometry_supported_pixels": 0,
                        "geometry_rejected_pixels": 0,
                        "output_valid_pixels": local_valid_count,
                        "mean_geometry_score": 1.0,
                        "median_best_depth_ratio": 1.0,
                        "skip_reason": "anchor_local_preserved",
                    }
                )
                continue

            gated, gstats = geometry_gate_candidate(
                candidate=candidate,
                target_poc=poc,
                reference_states=selected_sources,
                relative_cache=relative_cache,
                geometry_check_mask=geometry_check_mask,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                soft_log_threshold=geometry_soft_log,
                hard_log_threshold=geometry_hard_log,
                min_support=args.geometry_min_support,
                max_references=args.geometry_max_references,
                sample_radius=args.geometry_sample_radius,
                occlusion_ratio=args.geometry_occlusion_ratio,
                unsupported_penalty=args.geometry_unsupported_penalty,
                confidence_mix=args.geometry_confidence_mix,
                self_reference_weight=args.geometry_self_reference_weight,
                chunk_pixels=args.geometry_chunk_pixels,
                hierarchical_enabled=args.hierarchical_geometry,
                block_size=geometry_block_size,
                representative_mode=args.geometry_representative_mode,
                accept_confidence_ratio=args.geometry_block_accept_confidence_ratio,
                reject_fraction=args.geometry_block_reject_fraction,
            )
            gated_candidates.append(gated)
            geometry_stats.append(gstats)

        t_geometry = perf_counter() - stage_start
        stage_start = perf_counter()
        final_depth, final_conf, selected_label, fusion_stats = fuse_depth_candidates(
            candidates=gated_candidates,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            log_depth_threshold=blend_log_threshold,
            single_candidate_penalty=args.single_candidate_penalty,
            minimum_output_confidence=args.minimum_output_confidence,
            preserve_local_valid=preserve_local,
        )

        final_depth, final_conf, selected_label, hole_fill_stats = fill_fused_depth_holes(
            depth=final_depth,
            confidence=final_conf,
            selected_label=selected_label,
            radius=args.post_fusion_hole_fill_radius,
            confidence_decay=args.post_fusion_hole_fill_confidence_decay,
        )
        fusion_stats["post_fusion_hole_fill"] = hole_fill_stats

        # Candidate pool for encoder-side GT oracle selection. Local variants
        # remain distinct; individually geometry-gated temporal candidates and
        # the existing fused result are also exposed as choices.
        oracle_candidates: List[DepthCandidate] = []
        for local_name in local_candidate_names:
            oracle_candidates.append(local_candidate_maps[local_name])
        used_names = {candidate.name for candidate in oracle_candidates}
        next_label = len(oracle_candidates)
        for gated in gated_candidates[1:]:
            if gated.name in used_names:
                continue
            oracle_candidates.append(DepthCandidate(
                name=gated.name,
                label=next_label,
                source_poc=gated.source_poc,
                depth=gated.depth,
                confidence=gated.confidence,
            ))
            used_names.add(gated.name)
            next_label += 1
        fused_candidate = DepthCandidate(
            name="fused_final",
            label=next_label,
            source_poc=None,
            depth=final_depth,
            confidence=final_conf,
        )
        oracle_candidates.append(fused_candidate)

        oracle_depth, oracle_conf, oracle_label, oracle_stats = select_gt_oracle_candidates(
            candidates=oracle_candidates,
            gt_depth=gt_depth_frames[poc],
            block_size=oracle_block_size,
            metric=args.oracle_metric,
            minimum_coverage=args.oracle_minimum_coverage,
            missing_penalty=args.oracle_missing_penalty,
            fallback_candidate_name=args.oracle_fallback_candidate,
        )

        t_fusion = perf_counter() - stage_start
        output_depth = oracle_depth
        output_conf = oracle_conf
        output_label = oracle_label
        final_valid = output_depth > 0.0
        final_ratio = float(np.mean(final_valid))
        mean_conf = float(np.mean(output_conf[final_valid])) if np.any(final_valid) else 0.0
        quality_score = final_ratio * mean_conf
        propagated_selected = final_valid & (output_label >= len(local_candidate_names))
        propagated_valid_ratio = float(np.mean(propagated_selected))

        state_depth = output_depth if args.oracle_state_mode == "oracle" else final_depth
        state_conf = output_conf if args.oracle_state_mode == "oracle" else final_conf
        state = DepthState(
            poc=poc,
            decode_rank=decode_rank,
            depth=state_depth,
            confidence=state_conf,
            valid_ratio=final_ratio,
            mean_confidence=mean_conf,
            quality_score=quality_score,
            local_valid_ratio=local_ratio,
            propagated_valid_ratio=propagated_valid_ratio,
        )
        state_bank[poc] = state
        depth_frames[poc] = output_depth
        confidence_frames[poc] = output_conf
        candidate_index_frames[poc] = output_label

        frame_stats_by_poc[poc] = {
            "poc": poc,
            "decode_rank": decode_rank,
            "is_anchor": poc in anchor_pocs,
            "anchor_local_preserved": preserve_local,
            "mv_rows": len(mv_by_frame[poc]),
            "mv_sampling": mv_sampling_stats,
            "valid_depth_observations": len(observations),
            "local_candidates": local_candidate_stats,
            "primary_local_candidate": primary_local_name,
            "local_valid_pixel_ratio": local_ratio,
            "final_valid_pixel_ratio": final_ratio,
            "mean_output_confidence": mean_conf,
            "quality_score": quality_score,
            "selected_source_pocs": [s.poc for s in selected_sources],
            "propagation_sources": source_stats,
            "geometry_skip": geometry_skip_stats,
            "geometry_candidates": geometry_stats,
            "fusion": fusion_stats,
            "oracle_selection": oracle_stats,
        }

        t_total = perf_counter() - frame_start
        timing = {
            "mv_sampling": t_mv_sampling,
            "mv_depth": t_mv_depth,
            "local_fit_render": t_local,
            "warp": t_warp,
            "geometry": t_geometry,
            "fusion": t_fusion,
            "total": t_total,
        }
        frame_stats_by_poc[poc]["timing_seconds"] = timing
        for key, value in timing.items():
            cumulative_timing[key] += value

        print_progress(
            decode_rank=decode_rank,
            num_frames=args.num_frames,
            poc=poc,
            valid_obs=len(observations),
            local_ratio=local_ratio,
            final_ratio=final_ratio,
            num_sources=len(selected_sources),
        )
        if args.profile:
            print(
                f"\n  time poc={poc:3d}: sample={t_mv_sampling:.3f}s "
                f"mv={t_mv_depth:.3f}s local={t_local:.3f}s "
                f"warp={t_warp:.3f}s geom={t_geometry:.3f}s "
                f"fuse={t_fusion:.3f}s total={t_total:.3f}s"
            )

    print()
    if args.profile:
        print("Cumulative stage time:")
        for key, value in cumulative_timing.items():
            print(f"  {key:18s}: {value:.3f}s")

    zero_depth = np.zeros(
        (processing_height, processing_width), dtype=np.float64
    )
    zero_conf = np.zeros(
        (processing_height, processing_width), dtype=np.float32
    )
    final_depth_frames = [d if d is not None else zero_depth for d in depth_frames]
    final_conf_frames = [c if c is not None else zero_conf for c in confidence_frames]
    zero_idx = np.full((processing_height, processing_width), -1, dtype=np.int32)
    final_candidate_index_frames = [
        idx if idx is not None else zero_idx for idx in candidate_index_frames
    ]

    write_depth_yuv420p10le(
        output_path=args.out_yuv,
        depth_frames=final_depth_frames,
        depth_scale_real=depth_scale_real,
        output_width=args.width,
        output_height=args.height,
        upsample_mode=args.upsample_mode,
        invalid_aware_bilinear=args.bilinear_invalid_aware,
    )

    if args.out_confidence_yuv:
        write_confidence_yuv420p10le(
            output_path=args.out_confidence_yuv,
            confidence_frames=final_conf_frames,
            output_width=args.width,
            output_height=args.height,
            upsample_mode=args.upsample_mode,
        )
    if args.out_candidate_index_yuv:
        write_candidate_index_yuv420p10le(
            output_path=args.out_candidate_index_yuv,
            index_frames=final_candidate_index_frames,
            output_width=args.width,
            output_height=args.height,
        )

    stats_path = str(Path(args.out_yuv).with_suffix(".stats.json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mv_csv": args.mv_csv,
                "camera_param": args.camera_param,
                "gt_depth_yuv": args.gt_depth_yuv,
                "gt_depth_scale_real": gt_depth_scale_real,
                "gt_downsample_mode": args.gt_downsample_mode,
                "out_yuv": args.out_yuv,
                "out_confidence_yuv": args.out_confidence_yuv,
                "out_candidate_index_yuv": args.out_candidate_index_yuv,
                "width": args.width,
                "height": args.height,
                "processing_width": processing_width,
                "processing_height": processing_height,
                "downsample_scale": args.downsample_scale,
                "upsample_mode": args.upsample_mode,
                "bilinear_invalid_aware": args.bilinear_invalid_aware,
                "resize_backend": "opencv" if cv2 is not None else "numpy",
                "num_frames": args.num_frames,
                "fit_block_full_resolution": args.fit_block,
                "fit_block_processing_resolution": processing_fit_block,
                "local_candidate_names": local_candidate_names,
                "oracle_block_size_processing": oracle_block_size,
                "oracle_metric": args.oracle_metric,
                "oracle_minimum_coverage": args.oracle_minimum_coverage,
                "oracle_missing_penalty": args.oracle_missing_penalty,
                "oracle_fallback_candidate": args.oracle_fallback_candidate,
                "oracle_state_mode": args.oracle_state_mode,
                "max_mv_samples_per_fit_block": args.max_mv_samples_per_fit_block,
                "relative_transform_cache_pairs": len(relative_cache),
                "max_reproj_error_full_pixels": args.max_reproj_error,
                "max_reproj_error_processing_pixels": processing_max_reproj_error,
                "max_plane_slope_full_pixel": args.max_plane_slope,
                "max_plane_slope_processing_pixel": processing_max_plane_slope,
                "decode_order": decode_order,
                "anchor_pocs": sorted(anchor_pocs),
                "anchor_hole_fill_only": args.anchor_hole_fill_only,
                "predictor_neighbors": ["left", "top", "top_left"],
                "current_block_mv_used": False,
                "current_picture_mv_used_for_source_selection": False,
                "propagation_causality": "already-decoded-depth-only",
                "candidate_policy": "geometry-gate-then-selective-inverse-depth-blend",
                "max_propagation_sources": args.max_propagation_sources,
                "propagation_splat_radius": args.propagation_splat_radius,
                "post_fusion_hole_fill_radius": args.post_fusion_hole_fill_radius,
                "post_fusion_hole_fill_confidence_decay": args.post_fusion_hole_fill_confidence_decay,
                "constant_plane_relative_threshold": args.constant_plane_relative_threshold,
                "plane_refit_relative_threshold": args.plane_refit_relative_threshold,
                "plane_determinant_threshold": args.plane_determinant_threshold,
                "plane_refit": args.plane_refit,
                "depth_consistency_ratio": args.depth_consistency_ratio,
                "geometry_consistency_ratio": args.geometry_consistency_ratio,
                "geometry_hard_ratio": args.geometry_hard_ratio,
                "geometry_skip_agreed_pixels": args.geometry_skip_agreed_pixels,
                "geometry_skip_ratio": geometry_skip_ratio,
                "hierarchical_geometry": args.hierarchical_geometry,
                "geometry_block_size_processing": geometry_block_size,
                "geometry_representative_mode": args.geometry_representative_mode,
                "geometry_block_accept_confidence_ratio": args.geometry_block_accept_confidence_ratio,
                "geometry_block_reject_fraction": args.geometry_block_reject_fraction,
                "depth_scale_real": depth_scale_real,
                "device": device,
                "cumulative_timing_seconds": cumulative_timing,
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
    if args.out_candidate_index_yuv:
        print(f"Candidate index: {args.out_candidate_index_yuv}")
    print(f"Stats          : {stats_path}")


if __name__ == "__main__":
    main()
