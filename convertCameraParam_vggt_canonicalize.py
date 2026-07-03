#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert either
  1) original VGGT-Omega NPZ
       depth_original, extrinsic, intrinsic_original
  2) our canonical / fixed-K optimized NPZ
       depth_canonical, K_fixed, rvec_abs_final, tvec_abs_final

to
  - depth YUV420p10le
  - camera parameter JSONL

The canonical/fixed-K version is expected to contain absolute canonical camera poses:
  rvec_abs_final, tvec_abs_final
which represent camera_from_world in the fixed-K canonical camera coordinates.
The script converts those absolute poses to current_to_previous or GOP-local rvec/tvec.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Small utilities
# ============================================================

def npz_scalar_to_str(x: Any) -> str:
    arr = np.asarray(x)
    v = arr.item() if arr.shape == () else arr.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def sanitize_windows_filename_component(name: str, replacement: str = "_") -> str:
    name = unicodedata.normalize("NFC", str(name))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, name)
    if replacement:
        name = re.sub(re.escape(replacement) + r"+", replacement, name)
    name = name.rstrip(" .")
    if not name:
        name = "unnamed"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.split(".")[0].upper() in reserved:
        name = "_" + name
    return name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Pose helpers
# ============================================================

def R_from_rvec(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3).astype(np.float64)


def extrinsic_to_4x4(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = E[:, :3]
        T[:3, 3] = E[:, 3]
        return T
    if E.shape == (4, 4):
        return E.copy()
    raise ValueError(f"Unsupported extrinsic shape: {E.shape}")


def make_extrinsics_from_abs_rvec_tvec(rvec_abs: np.ndarray, tvec_abs: np.ndarray) -> np.ndarray:
    rvec_abs = np.asarray(rvec_abs, dtype=np.float64)
    tvec_abs = np.asarray(tvec_abs, dtype=np.float64)
    if rvec_abs.ndim != 2 or rvec_abs.shape[1] != 3:
        raise ValueError(f"rvec_abs must be [N,3], got {rvec_abs.shape}")
    if tvec_abs.shape != rvec_abs.shape:
        raise ValueError(f"tvec_abs shape mismatch: {tvec_abs.shape} vs {rvec_abs.shape}")
    n = rvec_abs.shape[0]
    E = np.zeros((n, 3, 4), dtype=np.float64)
    for i in range(n):
        E[i, :, :3] = R_from_rvec(rvec_abs[i])
        E[i, :, 3] = tvec_abs[i]
    return E


def rt_cur_to_prev_from_extrinsics(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input extrinsics are camera_from_world W2C.
    Output poc i transform:
      X_prev = R_i * X_cur + t_i
    poc 0 is identity/zero.
    """
    n = extrinsics.shape[0]
    W2Cs = [extrinsic_to_4x4(extrinsics[i]) for i in range(n)]
    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)
    for i in range(1, n):
        T = W2Cs[i - 1] @ np.linalg.inv(W2Cs[i])
        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3].astype(np.float64)
        rvecs[i] = rvec_from_R(R).astype(np.float32)
        tvecs[i] = t.astype(np.float32)
    return rvecs, tvecs


def rt_gop_local_from_extrinsics(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Output poc i transform from frame-0 camera coordinates to frame-i camera coordinates:
      X_i = R_i * X_0 + t_i
    poc 0 is identity/zero.
    """
    n = extrinsics.shape[0]
    W2Cs = [extrinsic_to_4x4(extrinsics[i]) for i in range(n)]
    C2W_0 = np.linalg.inv(W2Cs[0])
    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)
    for i in range(1, n):
        T = W2Cs[i] @ C2W_0
        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3].astype(np.float64)
        rvecs[i] = rvec_from_R(R).astype(np.float32)
        tvecs[i] = t.astype(np.float32)
    return rvecs, tvecs


def rt_absolute_from_extrinsics(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = extrinsics.shape[0]
    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        T = extrinsic_to_4x4(extrinsics[i])
        rvecs[i] = rvec_from_R(T[:3, :3]).astype(np.float32)
        tvecs[i] = T[:3, 3].astype(np.float32)
    return rvecs, tvecs


# ============================================================
# NPZ loading: original VGGT + canonical optimized
# ============================================================

def cleanup_depth(depth: np.ndarray, npz_path: Path, key: str) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 3:
        raise ValueError(f"{npz_path}: {key} must be [N,H,W], got {depth.shape}")
    return depth.astype(np.float32)


def repeat_K(K: np.ndarray, n: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 3):
        return np.repeat(K[None, :, :], n, axis=0).astype(np.float32)
    if K.shape == (n, 3, 3):
        return K.astype(np.float32)
    raise ValueError(f"Unsupported K shape {K.shape}; expected [3,3] or [N,3,3]")


def find_first_key(z: np.lib.npyio.NpzFile, keys: list[str]) -> str | None:
    for k in keys:
        if k in z:
            return k
    return None


def load_geometry_npz(npz_path: Path, input_mode: str = "auto") -> dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    canonical_depth_key = find_first_key(z, [
        "depth_canonical",
        "depth_final",
        "depth_optimized",
        "depth_modified",
    ])
    canonical_pose_r_key = find_first_key(z, [
        "rvec_abs_final",
        "rvec_abs_stage3_joint",
        "rvec_abs_stage2_t_nn",
        "rvec_abs_stage1_rt",
        "rvec_abs_init",
    ])
    canonical_pose_t_key = None
    if canonical_pose_r_key is not None:
        # Use the final t if present. Some stage4 variants keep unscaled stage3 separately.
        canonical_pose_t_key = find_first_key(z, [
            "tvec_abs_final",
            "tvec_abs_stage3_joint",
            "tvec_abs_stage3_joint_unscaled",
            "tvec_abs_stage2_t_nn",
            "tvec_abs_stage1_rt",
            "tvec_abs_init",
        ])

    looks_canonical = (
        canonical_depth_key is not None
        and "K_fixed" in keys
        and canonical_pose_r_key is not None
        and canonical_pose_t_key is not None
    )
    looks_vggt = all(k in keys for k in ["depth_original", "extrinsic", "intrinsic_original"])

    if input_mode == "auto":
        mode = "canonical" if looks_canonical else "vggt" if looks_vggt else "unknown"
    else:
        mode = input_mode

    if mode == "canonical":
        if not looks_canonical:
            raise KeyError(
                f"{npz_path}: canonical mode requires depth_canonical-like key, K_fixed, "
                f"rvec_abs_* and tvec_abs_* keys. Available keys: {sorted(keys)}"
            )
        depth = cleanup_depth(z[canonical_depth_key], npz_path, canonical_depth_key)  # type: ignore[arg-type]
        n = depth.shape[0]
        intrinsic = repeat_K(z["K_fixed"], n)
        rvec_abs = np.asarray(z[canonical_pose_r_key], dtype=np.float64)  # type: ignore[arg-type]
        tvec_abs = np.asarray(z[canonical_pose_t_key], dtype=np.float64)  # type: ignore[arg-type]
        if rvec_abs.shape[0] != n or tvec_abs.shape[0] != n:
            raise ValueError(
                f"{npz_path}: pose/depth frame mismatch: depth={n}, "
                f"{canonical_pose_r_key}={rvec_abs.shape}, {canonical_pose_t_key}={tvec_abs.shape}"
            )
        extrinsic = make_extrinsics_from_abs_rvec_tvec(rvec_abs, tvec_abs).astype(np.float32)
        source_type = "canonical_fixedK_optimized"
        pose_source = {
            "depth_key": canonical_depth_key,
            "intrinsic_key": "K_fixed",
            "rvec_abs_key": canonical_pose_r_key,
            "tvec_abs_key": canonical_pose_t_key,
        }
        fixed_intrinsic = True
    elif mode == "vggt":
        for k in ["depth_original", "extrinsic", "intrinsic_original"]:
            if k not in z:
                raise KeyError(f"{npz_path}: missing key '{k}'")
        depth = cleanup_depth(z["depth_original"], npz_path, "depth_original")
        extrinsic = np.asarray(z["extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(z["intrinsic_original"], dtype=np.float32)
        if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
            extrinsic = extrinsic[0]
        if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
            intrinsic = intrinsic[0]
        source_type = "vggt_original"
        pose_source = {
            "depth_key": "depth_original",
            "intrinsic_key": "intrinsic_original",
            "extrinsic_key": "extrinsic",
        }
        fixed_intrinsic = False
    else:
        raise KeyError(
            f"{npz_path}: cannot detect supported geometry NPZ. "
            f"Expected original VGGT keys or canonical optimized keys. Available keys: {sorted(keys)}"
        )

    if depth.ndim != 3:
        raise ValueError(f"{npz_path}: depth must be [N,H,W], got {depth.shape}")
    n = depth.shape[0]

    if extrinsic.ndim != 3 or extrinsic.shape[0] != n or extrinsic.shape[1:] not in [(3, 4), (4, 4)]:
        raise ValueError(f"{npz_path}: extrinsic must be [N,3,4] or [N,4,4], got {extrinsic.shape}, N={n}")
    if intrinsic.ndim != 3 or intrinsic.shape != (n, 3, 3):
        raise ValueError(f"{npz_path}: intrinsic must be [N,3,3], got {intrinsic.shape}, N={n}")

    if "frame_indices" in z:
        frame_indices = z["frame_indices"].astype(np.int64).tolist()
    else:
        frame_indices = list(range(n))

    rap_name = npz_scalar_to_str(z["rap_name"]) if "rap_name" in z else None
    rap_index = int(np.asarray(z["rap_index"]).item()) if "rap_index" in z else None

    return {
        "source_type": source_type,
        "pose_source": pose_source,
        "depth": depth.astype(np.float32),
        "extrinsic": extrinsic.astype(np.float32),
        "intrinsic": intrinsic.astype(np.float32),
        "fixed_intrinsic": bool(fixed_intrinsic),
        "frame_indices": [int(x) for x in frame_indices],
        "rap_name": rap_name,
        "rap_index": rap_index,
        "npz_keys": sorted(keys),
    }


# ============================================================
# Naming
# ============================================================

def infer_sequence_and_rap(npz_path: Path, rap_name_from_npz=None, rap_index_from_npz=None) -> tuple[str, str]:
    stem = npz_path.stem

    suffixes = [
        "_fixedK_gop_nn_frame_scale_geometry",
        "_fixedK_gop_nn_scale_geometry",
        "_fixedK_gop_nn_geometry",
        "_fixedK_gop_geometry",
        "_canonical_geometry",
        "_vggt_omega_outputs",
        "_outputs",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True

    m = re.search(r"^(?P<seq>.+?)_(?P<rap>rap\d+)$", stem)
    if m:
        return m.group("seq"), m.group("rap")

    if rap_name_from_npz is not None and re.fullmatch(r"rap\d+", str(rap_name_from_npz)):
        rap_name = str(rap_name_from_npz)
    elif rap_index_from_npz is not None:
        rap_name = f"rap{rap_index_from_npz}"
    else:
        rap_name = "rap0"

    m2 = re.search(r"(rap\d+)$", stem)
    if m2:
        rap_name = m2.group(1)
        seq = stem[: -len(rap_name)].rstrip("_")
        if seq:
            return seq, rap_name

    return stem, rap_name


# ============================================================
# Depth fixed-point coding
# ============================================================

def choose_fixed_point_depth_scale(depth: np.ndarray, percentile: float, precision: int, max_code: int = 1023) -> dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        scale_float = 1.0
        depth_ref = float(max_code)
        policy = "fallback_no_valid_depth"
    else:
        vals = depth[valid].astype(np.float64)
        depth_ref = float(np.percentile(vals, percentile))
        if not np.isfinite(depth_ref) or depth_ref <= 0:
            depth_ref = float(np.max(vals))
        if not np.isfinite(depth_ref) or depth_ref <= 0:
            depth_ref = float(max_code)
        scale_float = depth_ref / float(max_code)
        policy = "fixed_point_round"

    scale_int = max(1, int(round(scale_float * float(precision))))
    scale_real = scale_int / float(precision)
    return {
        "depth_scale": int(scale_int),
        "depth_scale_precision": int(precision),
        "depth_scale_real": float(scale_real),
        "depth_scale_float_before_fixed_point": float(scale_float),
        "depth_ref": float(depth_ref),
        "depth_percentile": float(percentile),
        "max_code": int(max_code),
        "scale_policy": policy,
        "encode_formula": "depth_y = round(depth / (depth_scale / depth_scale_precision))",
        "decode_formula": "depth = depth_y * depth_scale / depth_scale_precision",
    }


def quantize_depth_with_fixed_point_scale(depth: np.ndarray, depth_meta: dict[str, Any]) -> np.ndarray:
    scale_real = float(depth_meta["depth_scale_real"])
    max_code = int(depth_meta["max_code"])
    if scale_real <= 0:
        raise ValueError("depth_scale_real must be positive")
    y = np.nan_to_num(depth, nan=0.0, posinf=max_code * scale_real, neginf=0.0)
    y = np.round(y / scale_real)
    return np.clip(y, 0, max_code).astype(np.dtype("<u2"))


def compute_depth_quant_stats(depth: np.ndarray, yq: np.ndarray, depth_meta: dict[str, Any]) -> dict[str, Any]:
    scale_real = float(depth_meta["depth_scale_real"])
    max_code = int(depth_meta["max_code"])
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return {"valid_count": 0, "clip_ratio": 0.0, "mae": None, "rmse": None, "max_abs_err": None}
    recon = yq.astype(np.float32) * scale_real
    err = recon[valid].astype(np.float64) - depth[valid].astype(np.float64)
    abs_err = np.abs(err)
    clip = valid & (yq >= max_code)
    return {
        "valid_count": int(np.count_nonzero(valid)),
        "clip_ratio": float(np.count_nonzero(clip) / max(np.count_nonzero(valid), 1)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs_err": float(np.max(abs_err)),
    }


def write_depth_yuv420p10le(out_path: Path, depth: np.ndarray, depth_meta: dict[str, Any]) -> dict[str, Any]:
    n, h, w = depth.shape
    if w % 2 or h % 2:
        raise ValueError(f"YUV420 requires even resolution. Got {w}x{h}")
    ensure_parent(out_path)
    uv = np.full((h // 2, w // 2), 512, dtype=np.dtype("<u2"))
    all_stats = []
    with open(out_path, "wb") as f:
        for i in range(n):
            y = quantize_depth_with_fixed_point_scale(depth[i], depth_meta)
            f.write(np.ascontiguousarray(y).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())
            all_stats.append(compute_depth_quant_stats(depth[i], y, depth_meta))

    valid_maes = [s["mae"] for s in all_stats if s["mae"] is not None]
    valid_rmses = [s["rmse"] for s in all_stats if s["rmse"] is not None]
    clip_ratios = [s["clip_ratio"] for s in all_stats]
    return {
        "frame_stats": all_stats,
        "mean_mae": float(np.mean(valid_maes)) if valid_maes else None,
        "mean_rmse": float(np.mean(valid_rmses)) if valid_rmses else None,
        "max_clip_ratio": float(np.max(clip_ratios)) if clip_ratios else 0.0,
    }


# ============================================================
# Intrinsic deltas / JSONL
# ============================================================

def intrinsic_to_dict(K: np.ndarray, z_sign: float) -> dict[str, float]:
    K = np.asarray(K, dtype=np.float64)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "z_sign": float(z_sign),
    }


def intrinsic_to_vec4(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)


def intrinsic_delta_from_previous(intrinsic: np.ndarray, force_zero: bool = False) -> np.ndarray:
    n = intrinsic.shape[0]
    if force_zero:
        return np.zeros((n, 4), dtype=np.float32)
    vecs = np.stack([intrinsic_to_vec4(intrinsic[i]) for i in range(n)], axis=0).astype(np.float64)
    deltas = np.zeros_like(vecs)
    deltas[1:] = vecs[1:] - vecs[:-1]
    return deltas.astype(np.float32)


def compute_intrinsic_delta_stats(intrinsic: np.ndarray, force_zero: bool = False) -> dict[str, Any]:
    deltas = intrinsic_delta_from_previous(intrinsic, force_zero=force_zero)
    body = deltas[1:] if deltas.shape[0] > 1 else deltas
    abs_body = np.abs(body).astype(np.float64)
    if abs_body.size == 0:
        abs_body = np.zeros((1, 4), dtype=np.float64)
    return {
        "delta_order": ["dfx", "dfy", "dcx", "dcy"],
        "max_abs_delta": {
            "dfx": float(np.max(abs_body[:, 0])),
            "dfy": float(np.max(abs_body[:, 1])),
            "dcx": float(np.max(abs_body[:, 2])),
            "dcy": float(np.max(abs_body[:, 3])),
        },
        "mean_abs_delta": {
            "dfx": float(np.mean(abs_body[:, 0])),
            "dfy": float(np.mean(abs_body[:, 1])),
            "dcx": float(np.mean(abs_body[:, 2])),
            "dcy": float(np.mean(abs_body[:, 3])),
        },
    }


def write_camparam_jsonl(
    out_path: Path,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrinsic: np.ndarray,
    depth_meta: dict[str, Any],
    quant_summary: dict[str, Any],
    z_sign: float,
    pose_mode: str,
    width: int,
    height: int,
    source_npz: Path,
    depth_yuv_path: Path,
    source_type: str,
    pose_source: dict[str, Any],
    frame_indices: list[int],
    fixed_intrinsic: bool,
) -> None:
    force_zero_intr_delta = bool(fixed_intrinsic)
    intrinsic_deltas = intrinsic_delta_from_previous(intrinsic, force_zero=force_zero_intr_delta)
    intrinsic_delta_stats = compute_intrinsic_delta_stats(intrinsic, force_zero=force_zero_intr_delta)
    ensure_parent(out_path)

    header = {
        "type": "header",
        "format": "camparam_v2_vggt_or_canonical",
        "source_geometry_type": source_type,
        "source_npz": str(source_npz.resolve()),
        "source_keys": pose_source,
        "frame_count": int(rvecs.shape[0]),
        "frame_indices": [int(x) for x in frame_indices],
        "width": int(width),
        "height": int(height),
        "bit_depth": 10,
        "depth_yuv": str(depth_yuv_path.name),
        "depth_scale": int(depth_meta["depth_scale"]),
        "depth_scale_precision": int(depth_meta["depth_scale_precision"]),
        "depth_scale_real": float(depth_meta["depth_scale_real"]),
        "depth_quant": {
            **depth_meta,
            "quant_summary": {
                "mean_mae": quant_summary["mean_mae"],
                "mean_rmse": quant_summary["mean_rmse"],
                "max_clip_ratio": quant_summary["max_clip_ratio"],
            },
        },
        "intrinsic": intrinsic_to_dict(intrinsic[0], z_sign=z_sign),
        "intrinsic_param": "fx_fy_cx_cy",
        "intrinsic_mode": "rap_fixed" if fixed_intrinsic else "first_plus_previous_delta",
        "intrinsic_delta_mode": "fixed_zero_delta" if fixed_intrinsic else "previous_frame_delta",
        "intrinsic_delta_order": ["dfx", "dfy", "dcx", "dcy"],
        "intrinsic_delta_bits_per_frame": 0 if fixed_intrinsic else None,
        "intrinsic_delta_decode": (
            "fixed intrinsic: ignore all frame intrinsic_delta values"
            if fixed_intrinsic
            else "cur_fx += dfx; cur_fy += dfy; cur_cx += dcx; cur_cy += dcy"
        ),
        "intrinsic_delta_stats": intrinsic_delta_stats,
        "camera_param": "rvec_tvec_6d",
        "pose_mode": pose_mode,
        "pose_convention": {
            "current_to_previous": "X_prev = R * X_cur + t; poc0 identity/zero",
            "gop_local": "X_i = R * X_0 + t; poc0 identity/zero",
            "absolute": "camera_from_world in source/canonical camera coordinates",
        }.get(pose_mode, pose_mode),
        "frame_line_format": {
            "fields": ["poc", "frame_idx", "rvec", "tvec", "intrinsic_delta"],
            "note": "For fixed intrinsic, intrinsic_delta is written as [0,0,0,0] for compatibility.",
        },
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(header, ensure_ascii=False) + "\n")
        for poc in range(rvecs.shape[0]):
            rec = {
                "poc": int(poc),
                "frame_idx": int(frame_indices[poc]) if poc < len(frame_indices) else int(poc),
                "rvec": [float(x) for x in rvecs[poc]],
                "tvec": [float(x) for x in tvecs[poc]],
                "intrinsic_delta": [float(x) for x in intrinsic_deltas[poc]],
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Processing
# ============================================================

def process_one_npz(npz_path: Path, args: argparse.Namespace) -> bool:
    try:
        data = load_geometry_npz(npz_path, input_mode=args.input_mode)
    except Exception as exc:
        if args.skip_unrecognized:
            print(f"[SKIP] unsupported: {npz_path} ({type(exc).__name__}: {exc})")
            return False
        raise

    depth = data["depth"]
    extrinsics = data["extrinsic"]
    intrinsic = data["intrinsic"]
    n, h, w = depth.shape

    sequence_name, rap_name = infer_sequence_and_rap(
        npz_path,
        rap_name_from_npz=data["rap_name"],
        rap_index_from_npz=data["rap_index"],
    )
    sequence_name = sanitize_windows_filename_component(sequence_name)
    rap_name = sanitize_windows_filename_component(rap_name)

    if args.output_tag == "auto":
        tag = "canonical" if data["source_type"].startswith("canonical") else ""
    else:
        tag = args.output_tag.strip("_")

    mid = f"_{rap_name}" if not tag else f"_{rap_name}_{tag}"
    out_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_cam = out_dir / f"{sequence_name}_camParam{mid}.jsonl"
    out_depth = out_dir / f"{sequence_name}_depth{mid}.yuv"

    if not args.overwrite:
        if out_cam.exists():
            print(f"[SKIP] exists: {out_cam}")
            return False
        if out_depth.exists():
            print(f"[SKIP] exists: {out_depth}")
            return False

    depth_meta = choose_fixed_point_depth_scale(
        depth=depth,
        percentile=args.depth_percentile,
        precision=args.depth_scale_precision,
        max_code=1023,
    )
    quant_summary = write_depth_yuv420p10le(out_depth, depth, depth_meta)

    if args.pose_mode == "current_to_previous":
        rvecs, tvecs = rt_cur_to_prev_from_extrinsics(extrinsics)
    elif args.pose_mode == "gop_local":
        rvecs, tvecs = rt_gop_local_from_extrinsics(extrinsics)
    elif args.pose_mode == "absolute":
        rvecs, tvecs = rt_absolute_from_extrinsics(extrinsics)
    else:
        raise ValueError(args.pose_mode)

    write_camparam_jsonl(
        out_path=out_cam,
        rvecs=rvecs,
        tvecs=tvecs,
        intrinsic=intrinsic,
        depth_meta=depth_meta,
        quant_summary=quant_summary,
        z_sign=args.z_sign,
        pose_mode=args.pose_mode,
        width=w,
        height=h,
        source_npz=npz_path,
        depth_yuv_path=out_depth,
        source_type=data["source_type"],
        pose_source=data["pose_source"],
        frame_indices=data["frame_indices"],
        fixed_intrinsic=bool(data["fixed_intrinsic"]),
    )

    valid = np.isfinite(depth) & (depth > 0)
    dmin = float(np.min(depth[valid])) if np.any(valid) else 0.0
    dmax = float(np.max(depth[valid])) if np.any(valid) else 0.0
    intr_deltas = intrinsic_delta_from_previous(intrinsic, force_zero=bool(data["fixed_intrinsic"]))
    intr_delta_abs = np.abs(intr_deltas[1:]) if intr_deltas.shape[0] > 1 else intr_deltas

    print(f"[OK] {npz_path}")
    print(f"     source type         : {data['source_type']}")
    print(f"     source keys         : {data['pose_source']}")
    print(f"     frames / size       : {n}, {w}x{h}")
    print(f"     depth range         : {dmin:.6g} ~ {dmax:.6g}")
    print(f"     depth_ref           : {depth_meta['depth_ref']:.6g} @ p{args.depth_percentile}")
    print(f"     depth_scale int     : {depth_meta['depth_scale']}")
    print(f"     depth_scale real    : {depth_meta['depth_scale_real']:.9g}")
    print(f"     quant mean MAE      : {quant_summary['mean_mae']}")
    print(f"     quant mean RMSE     : {quant_summary['mean_rmse']}")
    print(f"     max clip ratio      : {quant_summary['max_clip_ratio']:.6g}")
    if intr_delta_abs.size:
        print(
            "     intrinsic delta max : "
            f"dfx={np.max(intr_delta_abs[:, 0]):.6g}, "
            f"dfy={np.max(intr_delta_abs[:, 1]):.6g}, "
            f"dcx={np.max(intr_delta_abs[:, 2]):.6g}, "
            f"dcy={np.max(intr_delta_abs[:, 3]):.6g}"
        )
    print(f"     pose_mode           : {args.pose_mode}")
    print(f"     camParam            : {out_cam}")
    print(f"     depth yuv           : {out_depth}")
    return True


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert original VGGT NPZ or optimized fixed-K/canonical NPZ to depth YUV + camParam JSONL."
    )
    ap.add_argument("--root", required=True, help="Root folder. NPZ files under this folder are recursively converted.")
    ap.add_argument("--pattern", default="*.npz", help="Recursive NPZ glob pattern. Default: *.npz")
    ap.add_argument("--input-mode", choices=["auto", "vggt", "canonical"], default="auto")
    ap.add_argument("--output-dir", default=None, help="Optional output folder. Default: same folder as each NPZ.")
    ap.add_argument(
        "--output-tag",
        default="auto",
        help="Filename tag after rap name. 'auto' uses 'canonical' for optimized NPZ and no tag for original VGGT.",
    )
    ap.add_argument("--depth-percentile", type=float, default=99.9)
    ap.add_argument("--depth-scale-precision", type=int, default=100000)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--pose-mode", choices=["current_to_previous", "gop_local", "absolute"], default="current_to_previous")
    ap.add_argument("--skip-unrecognized", action="store_true", default=True,
                    help="Skip NPZ files that are not supported geometry files. Default: enabled.")
    ap.add_argument("--no-skip-unrecognized", dest="skip_unrecognized", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f"Not a directory: {root}")
    if not (0.0 < args.depth_percentile <= 100.0):
        raise ValueError("--depth-percentile must be in (0, 100]")
    if args.depth_scale_precision <= 0:
        raise ValueError("--depth-scale-precision must be positive")

    npz_files = sorted(root.rglob(args.pattern))
    if not npz_files:
        print(f"No NPZ files found under: {root}")
        return

    print(f"Found NPZ files       : {len(npz_files)}")
    print(f"input_mode            : {args.input_mode}")
    print(f"pose_mode             : {args.pose_mode}")
    print(f"depth_percentile      : {args.depth_percentile}")
    print(f"depth_scale_precision : {args.depth_scale_precision}")
    print("supported input       : original VGGT NPZ or optimized fixed-K/canonical NPZ")

    ok = 0
    fail = 0
    skip = 0
    for npz_path in npz_files:
        try:
            converted = process_one_npz(npz_path, args)
            if converted:
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {npz_path}")
            print(f"       {type(exc).__name__}: {exc}")

    print("Done.")
    print(f"converted: {ok}")
    print(f"skipped  : {skip}")
    print(f"failed   : {fail}")


if __name__ == "__main__":
    main()
