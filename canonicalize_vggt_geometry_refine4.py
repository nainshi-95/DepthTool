#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_affine_cp_rf_refine_canonical_outputs_pose_coding.py

Batch second-stage refinement for canonicalized fixedK GOP NN-depth outputs.

This version adds a codec-aware predictive-pose regularizer that reproduces
how the refined camera poses are intended to be stored later:

  * The GOP first frame is implicit and re-anchored to R=I, t=0.
  * For frame i > 0, absolute local rotation R_i is predicted from the
    previous quantized/reconstructed rotation.
  * The stored rotation residual is a 3-component Rodrigues vector of
        R_res_i = R_i @ R_rec_{i-1}.T
  * Absolute local W2C t_i is predicted from the previous quantized/
    reconstructed t vector, and the stored translation residual is
        t_res_i = t_i - t_rec_{i-1}
  * Both residuals are quantized in a closed loop using the configured qsteps.
  * The optimization loss includes differentiable rate proxies for the six
    stored integer residual components and optional quantization-reconstruction
    penalties.

Input per sequence, recursively found under --src-root:
  <base>_fixedK_gop_nn_geometry.npz
  <base>_fixedK_gop_nn_cam.jsonl                         optional metadata source
  <base>_fixedK_gop_nn_depth_linear_yuv420p10le.yuv       copied unchanged if found
  <base>_fixedK_gop_nn_manifest.json                      optional metadata source

Output per sequence under --dst-root, with the SAME canonical filename format:
  <base>_fixedK_gop_nn_geometry.npz
  <base>_fixedK_gop_nn_cam.jsonl
  <base>_fixedK_gop_nn_depth_linear_yuv420p10le.yuv
  <base>_fixedK_gop_nn_manifest.json

Pose convention:
  Absolute pose is camera_from_world / W2C:
      X_cam_i = R_i X_world + t_i

  Relative target camera -> reference camera:
      R_rel = R_ref @ R_target.T
      t_rel = t_ref - R_rel @ t_target
      X_ref = R_rel X_target + t_rel

Coordinate convention:
  target pixel -> reference pixel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch


GEOM_SUFFIX = "_fixedK_gop_nn_geometry.npz"
CAM_SUFFIX = "_fixedK_gop_nn_cam.jsonl"
DEPTH_SUFFIX = "_fixedK_gop_nn_depth_linear_yuv420p10le.yuv"
MANIFEST_SUFFIX = "_fixedK_gop_nn_manifest.json"


# ============================================================
# Logging / JSON helpers
# ============================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def json_safe_float(x: Any) -> Any:
    if x is None:
        return None
    x = float(x)
    if np.isnan(x):
        return None
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return json_safe_float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def npz_scalar_json(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return npz_scalar_json(x.item())
        return [npz_scalar_json(v) for v in x.tolist()]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, str):
        return json.loads(x)
    return x


def load_first_jsonl_object(path: str | Path | None) -> Optional[dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict):
                return obj
    return None


# ============================================================
# Batch discovery and canonical filenames
# ============================================================

def derive_base_from_geometry_npz(npz_path: Path) -> str:
    name = npz_path.name
    if name.endswith(GEOM_SUFFIX):
        return name[: -len(GEOM_SUFFIX)]
    return npz_path.stem


def find_geometry_npz_files(src_root: Path, pattern: str) -> list[Path]:
    return [p for p in sorted(src_root.rglob(pattern)) if p.is_file()]


def make_out_prefix(npz_path: Path, src_root: Path, dst_root: Path, layout: str) -> Path:
    base = derive_base_from_geometry_npz(npz_path)
    if layout == "preserve":
        rel_dir = npz_path.parent.relative_to(src_root)
        out_dir = dst_root / rel_dir
    elif layout == "flat":
        out_dir = dst_root
    else:
        raise ValueError(layout)
    return out_dir / base


def canonical_paths_from_prefix(prefix: Path) -> dict[str, Path]:
    base = prefix.name
    parent = prefix.parent
    return {
        "geometry_npz": parent / f"{base}{GEOM_SUFFIX}",
        "camera_jsonl": parent / f"{base}{CAM_SUFFIX}",
        "depth_yuv": parent / f"{base}{DEPTH_SUFFIX}",
        "manifest": parent / f"{base}{MANIFEST_SUFFIX}",
    }


def find_canonical_sidecars(geometry_npz: Path) -> dict[str, Optional[Path]]:
    base = derive_base_from_geometry_npz(geometry_npz)
    parent = geometry_npz.parent
    paths = {
        "camera_jsonl": parent / f"{base}{CAM_SUFFIX}",
        "depth_yuv": parent / f"{base}{DEPTH_SUFFIX}",
        "manifest": parent / f"{base}{MANIFEST_SUFFIX}",
    }
    return {k: (v if v.is_file() else None) for k, v in paths.items()}


def already_done(out_prefix: Path) -> bool:
    return canonical_paths_from_prefix(out_prefix)["manifest"].is_file()


def validate_canonical_npz(npz_path: Path) -> bool:
    required_any = {
        "K": ["K_fixed", "K_refined", "K"],
        "r": [
            "rvec_abs_final",
            "rvec_abs_refined",
            "rvec_abs_stage4_smooth",
            "rvec_abs_stage3_joint",
            "rvec_abs_stage2_t_nn",
            "rvec_abs_stage1_rt",
        ],
        "t": [
            "tvec_abs_final",
            "tvec_abs_refined",
            "tvec_abs_stage4_smooth",
            "tvec_abs_stage3_joint",
            "tvec_abs_stage2_t_nn",
            "tvec_abs_stage1_rt",
        ],
        "depth": ["depth_canonical", "depth_original"],
    }
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            files = set(data.files)
        missing = [name for name, keys in required_any.items() if not any(k in files for k in keys)]
        if missing:
            log(f"SKIP invalid canonical NPZ: {npz_path} / missing groups {missing}")
            return False
        return True
    except Exception as e:
        log(f"SKIP unreadable NPZ: {npz_path} / {e}")
        return False


# ============================================================
# YUV helpers
# ============================================================

def frame_size_yuv420(w: int, h: int, bitdepth: int) -> int:
    bps = 1 if bitdepth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def count_frames_yuv420(path: str | Path, w: int, h: int, bitdepth: int) -> int:
    fs = frame_size_yuv420(w, h, bitdepth)
    size = os.path.getsize(path)
    trailing = size % fs
    if trailing:
        log(f"[WARN] trailing bytes ignored: {path}, trailing={trailing}")
    return size // fs


def read_y_frame(path: str | Path, w: int, h: int, bitdepth: int, idx: int) -> np.ndarray:
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    fs = frame_size_yuv420(w, h, bitdepth)
    y_samples = int(w) * int(h)
    with open(path, "rb") as f:
        f.seek(int(idx) * fs)
        y = np.fromfile(f, dtype=dtype, count=y_samples)
    if y.size != y_samples:
        raise RuntimeError(f"Cannot read Y frame idx={idx} from {path}")
    return y.reshape(h, w)


def write_depth_yuv420p10le_linear(path: Path, depth: np.ndarray, scale_meta: dict[str, Any]) -> dict[str, Any]:
    n, h, w = depth.shape
    if w % 2 or h % 2:
        raise ValueError("YUV420 output requires even width/height")
    ensure_parent(path)
    max_code = int(scale_meta["max_code"])
    scale = float(scale_meta["depth_scale_real"])
    clipped_total = 0
    with open(path, "wb") as f:
        for i in range(n):
            y = np.round(depth[i].astype(np.float64) / scale)
            clipped = (y < 0) | (y > max_code) | ~np.isfinite(y)
            clipped_total += int(np.count_nonzero(clipped))
            y = np.nan_to_num(y, nan=0.0, posinf=max_code, neginf=0.0)
            y = np.clip(y, 0, max_code).astype("<u2")
            uv = np.full((h // 2, w // 2), 512, dtype="<u2")
            f.write(y.tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())
    return {
        **scale_meta,
        "depth_yuv": str(path),
        "depth_yuv_format": "yuv420p10le",
        "depth_yuv_semantics": "Y stores linear depth code = round(depth / depth_scale_real); U/V neutral 512",
        "clipped_samples_total": int(clipped_total),
        "total_samples": int(n * h * w),
    }


def choose_depth_scale_fixed_point(
    depth: np.ndarray,
    percentile: float,
    precision: int,
    bit_depth: int = 10,
) -> dict[str, Any]:
    max_code = (1 << bit_depth) - 1
    m = np.isfinite(depth) & (depth > 0)
    if not np.any(m):
        scale_real = 1.0 / max_code
    else:
        ref = float(np.percentile(depth[m], percentile))
        ref = max(ref, 1e-12)
        scale_real = ref / float(max_code)
    scale_int = max(1, int(round(scale_real * precision)))
    scale_real_q = scale_int / float(precision)
    return {
        "depth_scale": int(scale_int),
        "depth_scale_precision": int(precision),
        "depth_scale_real": float(scale_real_q),
        "depth_scale_percentile": float(percentile),
        "depth_bit_depth": int(bit_depth),
        "max_code": int(max_code),
    }


# ============================================================
# Canonical stage loading
# ============================================================

def npz_key(data: np.lib.npyio.NpzFile, candidates: list[str], required: bool = True) -> Optional[str]:
    for k in candidates:
        if k in data.files:
            return k
    if required:
        raise KeyError(f"NPZ missing any of keys: {candidates}. Available keys: {data.files}")
    return None


def load_canonical_npz(npz_path: str | Path) -> dict[str, Any]:
    path = Path(npz_path)
    data = np.load(path, allow_pickle=True)
    k_key = npz_key(data, ["K_fixed", "K_refined", "K"])
    r_key = npz_key(
        data,
        [
            "rvec_abs_final",
            "rvec_abs_refined",
            "rvec_abs_stage4_smooth",
            "rvec_abs_stage3_joint",
            "rvec_abs_stage2_t_nn",
            "rvec_abs_stage1_rt",
        ],
    )
    t_key = npz_key(
        data,
        [
            "tvec_abs_final",
            "tvec_abs_refined",
            "tvec_abs_stage4_smooth",
            "tvec_abs_stage3_joint",
            "tvec_abs_stage2_t_nn",
            "tvec_abs_stage1_rt",
        ],
    )
    d_key = npz_key(data, ["depth_canonical", "depth_original"])

    K = np.asarray(data[k_key], dtype=np.float64).reshape(3, 3)
    rvecs = np.asarray(data[r_key], dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(data[t_key], dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(data[d_key], dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"depth must be [N,H,W], got {depth.shape}")
    n, _, _ = depth.shape
    if rvecs.shape[0] != n or tvecs.shape[0] != n:
        raise ValueError(f"Pose count mismatch: depth N={n}, r={rvecs.shape}, t={tvecs.shape}")

    if "frame_indices" in data.files:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.shape[0] != n:
            raise ValueError(f"frame_indices count mismatch: {frame_indices.shape[0]} vs N={n}")
    else:
        frame_indices = np.arange(n, dtype=np.int32)

    pairs = None
    if "pairs_json" in data.files:
        try:
            obj = npz_scalar_json(data["pairs_json"])
            if isinstance(obj, list):
                pairs = []
                for p in obj:
                    if isinstance(p, dict):
                        if "target" in p and "ref" in p:
                            pairs.append(
                                (
                                    int(p["target"]),
                                    int(p["ref"]),
                                    float(p.get("weight", 1.0)),
                                    str(p.get("kind", "npz")),
                                )
                            )
                        elif "tar" in p and "ref" in p:
                            pairs.append(
                                (
                                    int(p["tar"]),
                                    int(p["ref"]),
                                    float(p.get("weight", 1.0)),
                                    str(p.get("kind", "npz")),
                                )
                            )
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        pairs.append(
                            (
                                int(p[0]),
                                int(p[1]),
                                float(p[2]) if len(p) >= 3 else 1.0,
                                "npz",
                            )
                        )
        except Exception:
            pairs = None

    return {
        "npz_path": str(path),
        "data": data,
        "K": K,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "depth": depth,
        "frame_indices": frame_indices,
        "pairs": pairs,
        "source_keys": {"K": k_key, "rvecs": r_key, "tvecs": t_key, "depth": d_key},
    }


# ============================================================
# Pair selection
# ============================================================

def parse_pairs(s: str, default_weight: float = 1.0) -> list[tuple[int, int, float, str]]:
    out: list[tuple[int, int, float, str]] = []
    if not s or not s.strip():
        return out
    for tok in re.split(r"[,;\s]+", s.strip()):
        if not tok:
            continue
        tok = tok.replace("->", ":")
        parts = tok.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid pair token '{tok}'. Use target:ref[:weight].")
        target = int(parts[0])
        ref = int(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else float(default_weight)
        out.append((target, ref, weight, "cli"))
    return out


def generate_adjacent_pairs(n: int, bidirectional: bool, weight: float) -> list[tuple[int, int, float, str]]:
    out = []
    for i in range(1, n):
        out.append((i, i - 1, float(weight), "adjacent"))
        if bidirectional:
            out.append((i - 1, i, float(weight), "adjacent_rev"))
    return out


def generate_dyadic_pairs(n: int, bidirectional: bool = True, weight: float = 1.0) -> list[tuple[int, int, float, str]]:
    acc: dict[tuple[int, int], tuple[int, int, float, str]] = {}

    def add(t: int, r: int, w: float, kind: str) -> None:
        if t == r or not (0 <= t < n and 0 <= r < n):
            return
        key = (int(t), int(r))
        if key in acc:
            old = acc[key]
            acc[key] = (old[0], old[1], old[2] + float(w), old[3] + "+" + kind)
        else:
            acc[key] = (key[0], key[1], float(w), kind)

    def rec(a: int, b: int, level: int) -> None:
        if b <= a + 1:
            return
        m = (a + b) // 2
        ww = float(weight) / math.sqrt(level + 1.0)
        add(m, a, ww, f"dyadic_L{level}")
        add(m, b, ww, f"dyadic_L{level}")
        if bidirectional:
            add(a, m, ww, f"dyadic_rev_L{level}")
            add(b, m, ww, f"dyadic_rev_L{level}")
        rec(a, m, level + 1)
        rec(m, b, level + 1)

    rec(0, n - 1, 0)
    return sorted(acc.values(), key=lambda x: (abs(x[0] - x[1]), x[0], x[1]))


def build_pair_list(args: argparse.Namespace, stage: dict[str, Any]) -> list[tuple[int, int, float, str]]:
    n = int(stage["depth"].shape[0])
    if args.pairs.strip():
        pairs = parse_pairs(args.pairs, default_weight=args.pair_weight)
    elif args.pair_source == "npz" and stage.get("pairs"):
        pairs = list(stage["pairs"])
    elif args.pair_source == "adjacent":
        pairs = generate_adjacent_pairs(n, bidirectional=not args.no_bidirectional_pairs, weight=args.pair_weight)
    elif args.pair_source in ("dyadic", "npz"):
        pairs = generate_dyadic_pairs(n, bidirectional=not args.no_bidirectional_pairs, weight=args.pair_weight)
    elif args.pair_source == "all":
        pairs = [(t, r, args.pair_weight, "all") for t in range(n) for r in range(n) if t != r]
    else:
        raise ValueError(args.pair_source)

    checked = []
    seen = set()
    for t, r, w, kind in pairs:
        if not (0 <= int(t) < n and 0 <= int(r) < n):
            raise ValueError(f"Pair out of range for N={n}: {t}->{r}")
        key = (int(t), int(r))
        if key in seen:
            continue
        seen.add(key)
        checked.append((int(t), int(r), float(w), str(kind)))
    if args.max_pairs > 0:
        checked = checked[: int(args.max_pairs)]
    if not checked:
        raise RuntimeError("No pairs selected.")
    return checked


# ============================================================
# Geometry / projection
# ============================================================

def rodrigues_np(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def all_rotation_matrices_np(rvecs: np.ndarray) -> np.ndarray:
    return np.stack([rodrigues_np(r) for r in np.asarray(rvecs)], axis=0)


def torch_rodrigues(rvecs: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues exponential map with stable zero-angle gradients."""
    dtype = rvecs.dtype
    device = rvecs.device
    n = rvecs.shape[0]
    x, y, z = rvecs[:, 0], rvecs[:, 1], rvecs[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    theta2 = torch.sum(rvecs * rvecs, dim=-1)
    small = theta2 < 1e-12

    # Never evaluate 0/0 in the non-selected torch.where branches. NaNs in a
    # non-selected branch can still poison autograd at an exact zero rvec.
    eps = torch.finfo(dtype).eps
    theta_safe = torch.sqrt(torch.clamp(theta2, min=eps))
    theta2_safe = torch.clamp(theta2, min=eps)
    A_general = torch.sin(theta_safe) / theta_safe
    B_general = (1.0 - torch.cos(theta_safe)) / theta2_safe

    A_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    B_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    A = torch.where(small, A_series, A_general)
    B = torch.where(small, B_series, B_general)

    I = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
    return I + A[:, None, None] * K + B[:, None, None] * (K @ K)


def torch_matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """
    Differentiable matrix -> Rodrigues vector conversion.

    This is robust for the small adjacent-frame residual rotations expected by
    the predictive coding model. Rotations extremely close to pi are not the
    intended operating region.
    """
    single = R.ndim == 2
    Rf = R.reshape(-1, 3, 3)
    vee = torch.stack(
        [
            Rf[:, 2, 1] - Rf[:, 1, 2],
            Rf[:, 0, 2] - Rf[:, 2, 0],
            Rf[:, 1, 0] - Rf[:, 0, 1],
        ],
        dim=-1,
    )
    sin_theta = 0.5 * torch.sqrt(
        torch.sum(vee * vee, dim=-1) + torch.finfo(Rf.dtype).eps
    )
    trace = Rf[:, 0, 0] + Rf[:, 1, 1] + Rf[:, 2, 2]
    cos_theta = torch.clamp(0.5 * (trace - 1.0), min=-1.0, max=1.0)
    theta = torch.atan2(sin_theta, cos_theta)

    small = sin_theta < 1e-6
    scale_small = 0.5 + theta * theta / 12.0
    scale_normal = theta / torch.clamp(2.0 * sin_theta, min=1e-12)
    scale = torch.where(small, scale_small, scale_normal)
    rvec = scale[:, None] * vee

    if single:
        return rvec[0]
    return rvec.reshape(*R.shape[:-2], 3)


def hard_uniform_quantize(x: torch.Tensor, qstep: float) -> torch.Tensor:
    qstep = float(qstep)
    if qstep <= 0.0:
        raise ValueError("Quantization step must be positive")
    return torch.round(x / qstep) * qstep


def ste_uniform_quantize(x: torch.Tensor, qstep: float) -> torch.Tensor:
    """Hard quantization in the forward pass, identity gradient in backward."""
    xq = hard_uniform_quantize(x, qstep)
    return x + (xq - x).detach()


def pose_symbol_rate_proxy(normalized_residual: torch.Tensor, mode: str) -> torch.Tensor:
    magnitude = torch.abs(normalized_residual)
    if mode == "log1p":
        return torch.mean(torch.log1p(magnitude) / math.log(2.0))
    if mode == "l1":
        return torch.mean(magnitude)
    if mode == "l2":
        return torch.mean(normalized_residual * normalized_residual)
    raise ValueError(f"Unsupported pose coding rate mode: {mode}")


def camera_map_w2c_np(
    target: int,
    ref: int,
    width: int,
    height: int,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    depth_img: np.ndarray,
    z_sign: float,
    z_min: float,
    row_batch: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rs = all_rotation_matrices_np(rvecs)
    R_tar = Rs[int(target)]
    R_ref = Rs[int(ref)]
    t_tar = tvecs[int(target)]
    t_ref = tvecs[int(ref)]
    R_rel = R_ref @ R_tar.T
    t_rel = t_ref - R_rel @ t_tar
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)
    xs_full = np.arange(width, dtype=np.float64)

    for y0 in range(0, height, max(1, int(row_batch))):
        y1 = min(height, y0 + int(row_batch))
        ys = np.arange(y0, y1, dtype=np.float64)
        xs, yy = np.meshgrid(xs_full, ys)
        ray_x = (xs - cx) / fx
        ray_y = (yy - cy) / fy
        rays = np.stack(
            [
                ray_x.reshape(-1),
                ray_y.reshape(-1),
                np.full((y1 - y0) * width, float(z_sign), dtype=np.float64),
            ],
            axis=1,
        )
        dep = depth_img[y0:y1, :].reshape(-1).astype(np.float64)
        X_tar = dep[:, None] * rays
        X_ref = X_tar @ R_rel.T + t_rel[None, :]
        z = X_ref[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, np.where(z >= 0, 1e-9, -1e-9))
        mx = fx * (X_ref[:, 0] / z_safe) + cx
        my = fy * (X_ref[:, 1] / z_safe) + cy
        valid = (
            np.isfinite(mx)
            & np.isfinite(my)
            & np.isfinite(dep)
            & (dep > 0.0)
            & (z * float(z_sign) > float(z_min))
            & (mx >= 0.0)
            & (mx <= width - 1.0)
            & (my >= 0.0)
            & (my <= height - 1.0)
        )
        map_x[y0:y1, :] = mx.reshape(y1 - y0, width).astype(np.float32)
        map_y[y0:y1, :] = my.reshape(y1 - y0, width).astype(np.float32)
        valid_all[y0:y1, :] = valid.reshape(y1 - y0, width)

    map_x[~valid_all] = -1.0
    map_y[~valid_all] = -1.0
    return map_x, map_y, valid_all


def remap_y(ref_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def calc_cost(target_y: np.ndarray, pred_y: np.ndarray, valid: np.ndarray, bitdepth: int) -> dict[str, Any]:
    valid = valid.astype(bool)
    if not np.any(valid):
        return {"valid_ratio": float(np.mean(valid)), "mae": None, "mse": None, "psnr": None}
    diff = target_y.astype(np.float32)[valid] - pred_y.astype(np.float32)[valid]
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    maxv = float((1 << bitdepth) - 1)
    psnr = 999.0 if mse <= 1e-12 else float(10.0 * np.log10((maxv * maxv) / mse))
    return {"valid_ratio": float(np.mean(valid)), "mae": mae, "mse": mse, "psnr": psnr}


# ============================================================
# Structure ECC residual transform
# ============================================================

def normalize_for_ecc(img: np.ndarray, bitdepth: int) -> np.ndarray:
    return np.clip(img.astype(np.float32) / float((1 << bitdepth) - 1), 0.0, 1.0).astype(np.float32)


def make_structure_image(img: np.ndarray, bitdepth: int, mode: str, log_gain: float, pre_blur: int) -> np.ndarray:
    y = normalize_for_ecc(img, bitdepth)
    if int(pre_blur) > 0:
        k = 2 * int(pre_blur) + 1
        y = cv2.GaussianBlur(y, (k, k), 0)
    gx = cv2.Scharr(y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(y, cv2.CV_32F, 0, 1)
    if mode == "scharr_mag":
        s = np.sqrt(gx * gx + gy * gy)
    elif mode == "scharr_l1":
        s = np.abs(gx) + np.abs(gy)
    elif mode == "scharr_x":
        s = np.abs(gx)
    elif mode == "scharr_y":
        s = np.abs(gy)
    elif mode == "scharr_x_weighted":
        s = 0.75 * np.abs(gx) + 0.25 * np.abs(gy)
    else:
        raise ValueError(mode)
    if float(log_gain) > 0:
        s = np.log1p(float(log_gain) * s)
    m = float(np.max(s))
    if m > 1e-8:
        s = s / m
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def make_valid_mask_u8(valid: np.ndarray, erode: int = 2) -> np.ndarray:
    mask = valid.astype(np.uint8) * 255
    if int(erode) > 0:
        k = 2 * int(erode) + 1
        mask = cv2.erode(mask, np.ones((k, k), dtype=np.uint8), iterations=1)
    return mask


def make_structure_mask_u8(
    structure: np.ndarray,
    base_mask_u8: np.ndarray,
    keep_percent: float,
    dilate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = base_mask_u8 > 0
    vals = structure[base]
    stats: dict[str, Any] = {
        "base_count": int(np.count_nonzero(base)),
        "keep_percent": float(keep_percent),
        "threshold": None,
        "structure_count": 0,
        "final_count": 0,
    }
    if vals.size < 100:
        return base_mask_u8.copy(), stats
    percentile = 100.0 - float(np.clip(keep_percent, 0.1, 100.0))
    thr = float(np.percentile(vals, percentile))
    mask = base & (structure >= thr)
    stats["threshold"] = thr
    stats["structure_count"] = int(np.count_nonzero(mask))
    mask_u8 = mask.astype(np.uint8) * 255
    if int(dilate) > 0:
        k = 2 * int(dilate) + 1
        mask_u8 = cv2.dilate(mask_u8, np.ones((k, k), dtype=np.uint8), iterations=1)
        mask_u8 = np.where(base, mask_u8, 0).astype(np.uint8)
    stats["final_count"] = int(np.count_nonzero(mask_u8))
    return mask_u8, stats


def apply_initial_static_residual_mask(
    template: np.ndarray,
    inp: np.ndarray,
    mask_u8: np.ndarray,
    keep_percent: float,
    min_mask_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    keep_percent = float(keep_percent)
    stats = {
        "enabled": keep_percent < 99.999,
        "keep_percent": keep_percent,
        "threshold": None,
        "input_count": int(np.count_nonzero(mask_u8)),
        "output_count": int(np.count_nonzero(mask_u8)),
    }
    if keep_percent >= 99.999:
        return mask_u8, stats
    use = mask_u8 > 0
    if np.count_nonzero(use) < int(min_mask_count):
        stats["reason"] = "too_few_input_pixels"
        return mask_u8, stats
    residual = np.abs(template.astype(np.float32) - inp.astype(np.float32))
    vals = residual[use]
    thr = float(np.percentile(vals, max(0.1, min(100.0, keep_percent))))
    new_mask = use & (residual <= thr)
    if np.count_nonzero(new_mask) < int(min_mask_count):
        stats["reason"] = "too_few_after_filter"
        stats["threshold"] = thr
        return mask_u8, stats
    out = new_mask.astype(np.uint8) * 255
    stats["threshold"] = thr
    stats["output_count"] = int(np.count_nonzero(out))
    return out, stats


def apply_depth_edge_rejection(
    depth_img: np.ndarray,
    mask_u8: np.ndarray,
    reject_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    stats = {
        "enabled": float(reject_percentile) < 99.999,
        "reject_percentile": float(reject_percentile),
        "threshold": None,
        "input_count": int(np.count_nonzero(mask_u8)),
        "output_count": int(np.count_nonzero(mask_u8)),
    }
    if float(reject_percentile) >= 99.999:
        return mask_u8, stats
    base = mask_u8 > 0
    if np.count_nonzero(base) < 100:
        stats["reason"] = "too_few_input_pixels"
        return mask_u8, stats
    logd = np.log(np.maximum(depth_img.astype(np.float32), 1e-12))
    gx = cv2.Scharr(logd, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(logd, cv2.CV_32F, 0, 1)
    edge = np.sqrt(gx * gx + gy * gy)
    vals = edge[base]
    thr = float(np.percentile(vals, float(reject_percentile)))
    keep = base & (edge <= thr)
    out = keep.astype(np.uint8) * 255
    stats["threshold"] = thr
    stats["output_count"] = int(np.count_nonzero(out))
    return out, stats


def identity_transform(cp_num: int) -> np.ndarray:
    return np.eye(3, dtype=np.float32) if int(cp_num) == 4 else np.eye(2, 3, dtype=np.float32)


def transform_to_full_resolution(M_s: np.ndarray, cp_num: int, scale: float) -> np.ndarray:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-9:
        return M_s.astype(np.float32)
    S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if int(cp_num) == 4:
        H_s = np.asarray(M_s, dtype=np.float32).reshape(3, 3)
    else:
        H_s = np.eye(3, dtype=np.float32)
        H_s[:2, :] = np.asarray(M_s, dtype=np.float32).reshape(2, 3)
    H = np.linalg.inv(S) @ H_s @ S
    if int(cp_num) == 4:
        den = float(H[2, 2])
        if abs(den) < 1e-12:
            den = 1e-12
        H = H / den
        return H.astype(np.float32)
    return H[:2, :].astype(np.float32)


def resize_for_ecc(img: np.ndarray, mask_u8: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-9:
        return img.astype(np.float32), mask_u8.astype(np.uint8)
    h, w = img.shape
    sw = max(8, int(round(w * scale)))
    sh = max(8, int(round(h * scale)))
    img_s = cv2.resize(img.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask_u8.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    return img_s.astype(np.float32), mask_s.astype(np.uint8)


def warp_ecc_input_to_template_domain(inp: np.ndarray, M: np.ndarray, cp_num: int) -> np.ndarray:
    h, w = inp.shape
    if int(cp_num) == 4:
        return cv2.warpPerspective(
            inp.astype(np.float32),
            np.asarray(M, dtype=np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)
    return cv2.warpAffine(
        inp.astype(np.float32),
        np.asarray(M, dtype=np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)


def run_find_transform_ecc_with_init(
    template: np.ndarray,
    inp: np.ndarray,
    mask_u8: np.ndarray,
    cp_num: int,
    init_matrix_full: np.ndarray,
    max_iters: int,
    eps: float,
    gauss_filt_size: int,
    ecc_scale: float,
) -> tuple[np.ndarray, float]:
    cp_num = int(cp_num)
    scale = float(ecc_scale)
    template_s, mask_s = resize_for_ecc(template, mask_u8, scale)
    inp_s, _ = resize_for_ecc(inp, mask_u8, scale)
    init_s = init_matrix_full.copy()
    if abs(scale - 1.0) >= 1e-9:
        S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        if cp_num == 4:
            H_full = np.asarray(init_matrix_full, dtype=np.float32).reshape(3, 3)
        else:
            H_full = np.eye(3, dtype=np.float32)
            H_full[:2, :] = np.asarray(init_matrix_full, dtype=np.float32).reshape(2, 3)
        H_s = S @ H_full @ np.linalg.inv(S)
        init_s = H_s if cp_num == 4 else H_s[:2, :]

    if cp_num == 4:
        motion_type = cv2.MOTION_HOMOGRAPHY
        warp = np.asarray(init_s, dtype=np.float32).reshape(3, 3).copy()
    else:
        motion_type = cv2.MOTION_AFFINE
        warp = np.asarray(init_s, dtype=np.float32).reshape(2, 3).copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    cc, warp_s = cv2.findTransformECC(
        templateImage=template_s.astype(np.float32),
        inputImage=inp_s.astype(np.float32),
        warpMatrix=warp,
        motionType=motion_type,
        criteria=criteria,
        inputMask=mask_s.astype(np.uint8),
        gaussFiltSize=int(gauss_filt_size),
    )
    return transform_to_full_resolution(warp_s, cp_num, scale), float(cc)


def refine_mask_by_structure_residual(
    template: np.ndarray,
    inp: np.ndarray,
    M: np.ndarray,
    current_mask_u8: np.ndarray,
    base_mask_u8: np.ndarray,
    cp_num: int,
    keep_percent: float,
    min_mask_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = current_mask_u8 > 0
    base = base_mask_u8 > 0
    use = current & base
    stats: dict[str, Any] = {
        "input_count": int(np.count_nonzero(use)),
        "output_count": int(np.count_nonzero(use)),
        "keep_percent": float(keep_percent),
        "threshold": None,
        "mean_residual": None,
        "median_residual": None,
        "p90_residual": None,
    }
    if np.count_nonzero(use) < int(min_mask_count):
        return current_mask_u8.copy(), stats
    warped = warp_ecc_input_to_template_domain(inp, M, cp_num=cp_num)
    residual = np.abs(template.astype(np.float32) - warped.astype(np.float32))
    vals = residual[use]
    if vals.size < int(min_mask_count):
        return current_mask_u8.copy(), stats
    keep_percent = max(0.1, min(100.0, float(keep_percent)))
    thr = float(np.percentile(vals, keep_percent))
    new_mask = use & (residual <= thr)
    if np.count_nonzero(new_mask) < int(min_mask_count):
        return current_mask_u8.copy(), stats
    stats.update(
        {
            "output_count": int(np.count_nonzero(new_mask)),
            "threshold": thr,
            "mean_residual": float(np.mean(vals)),
            "median_residual": float(np.median(vals)),
            "p90_residual": float(np.percentile(vals, 90.0)),
        }
    )
    return new_mask.astype(np.uint8) * 255, stats


def estimate_pair_structure_ecc_stable(
    target_y: np.ndarray,
    cam_warp_y: np.ndarray,
    valid_mask_u8: np.ndarray,
    depth_img: np.ndarray,
    bitdepth: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, Optional[float], np.ndarray, dict[str, Any], np.ndarray]:
    cp_num = int(args.ecc_cp_num)
    template = make_structure_image(
        target_y,
        bitdepth,
        args.structure_mode,
        args.structure_log_gain,
        args.structure_pre_blur,
    )
    inp = make_structure_image(
        cam_warp_y,
        bitdepth,
        args.structure_mode,
        args.structure_log_gain,
        args.structure_pre_blur,
    )
    mask_u8, mask_stats = make_structure_mask_u8(
        template,
        valid_mask_u8,
        args.structure_keep_percent,
        args.structure_mask_dilate,
    )
    mask_u8, depth_edge_stats = apply_depth_edge_rejection(
        depth_img,
        mask_u8,
        args.depth_edge_keep_percentile,
    )
    mask_u8, static_stats = apply_initial_static_residual_mask(
        template,
        inp,
        mask_u8,
        args.static_residual_keep_percent,
        args.ecc_min_mask_count,
    )

    stats: dict[str, Any] = {
        "motion_type": "homography" if cp_num == 4 else "affine",
        "ecc_cp_num": cp_num,
        "ecc_scale": float(args.ecc_scale),
        "structure_mode": args.structure_mode,
        "structure_keep_percent": float(args.structure_keep_percent),
        "mask_dilate": int(args.structure_mask_dilate),
        "structure_mask": mask_stats,
        "depth_edge_rejection": depth_edge_stats,
        "static_residual_mask": static_stats,
        "rounds": [],
    }
    if np.count_nonzero(mask_u8) < int(args.ecc_min_mask_count):
        stats["success"] = False
        stats["reason"] = "too_few_mask_pixels_before_ecc"
        return identity_transform(cp_num), False, None, mask_u8, stats, template

    M = identity_transform(cp_num)
    score: Optional[float] = None
    success = False
    base_mask_u8 = mask_u8.copy()
    rounds = max(1, int(args.structure_ecc_rounds))
    for r in range(rounds):
        round_stats: dict[str, Any] = {
            "round": int(r),
            "mask_count_before": int(np.count_nonzero(mask_u8)),
            "success": False,
        }
        try:
            M, score = run_find_transform_ecc_with_init(
                template=template,
                inp=inp,
                mask_u8=mask_u8,
                cp_num=cp_num,
                init_matrix_full=M,
                max_iters=args.ecc_iters,
                eps=args.ecc_eps,
                gauss_filt_size=args.ecc_gauss,
                ecc_scale=args.ecc_scale,
            )
            success = True
            round_stats["success"] = True
            round_stats["ecc_cc"] = float(score)
        except cv2.error as exc:
            round_stats["cv2_error"] = str(exc)
            stats["rounds"].append(round_stats)
            break

        if r < rounds - 1:
            mask_u8, res_stats = refine_mask_by_structure_residual(
                template=template,
                inp=inp,
                M=M,
                current_mask_u8=mask_u8,
                base_mask_u8=base_mask_u8,
                cp_num=cp_num,
                keep_percent=args.structure_residual_keep_percent,
                min_mask_count=args.ecc_min_mask_count,
            )
            round_stats["residual_refine"] = res_stats
            round_stats["mask_count_after"] = int(np.count_nonzero(mask_u8))
        stats["rounds"].append(round_stats)

    stats["final_mask_count"] = int(np.count_nonzero(mask_u8))
    stats["final_ecc_cc"] = None if score is None else float(score)
    stats["success"] = bool(success)
    if not success:
        return identity_transform(cp_num), False, None, mask_u8, stats, template
    return M.astype(np.float32), True, score, mask_u8, stats, template


def apply_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)
    if M.shape == (2, 3):
        return (homo @ M.T).astype(np.float32)
    if M.shape == (3, 3):
        q = homo @ M.T
        den = q[:, 2:3]
        den = np.where(np.abs(den) < 1e-8, 1e-8, den)
        return (q[:, :2] / den).astype(np.float32)
    raise ValueError(f"Bad transform shape: {M.shape}")


def transform_cp_bias(M: np.ndarray, w: int, h: int, cp_num: int) -> np.ndarray:
    if int(cp_num) == 4:
        src = np.asarray([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    elif int(cp_num) == 3:
        src = np.asarray([[0, 0], [w, 0], [0, h]], dtype=np.float32)
    else:
        raise ValueError("This batch refiner supports only 3CP affine or 4CP homography supervision.")
    return apply_transform_points(M, src) - src


# ============================================================
# Observation extraction
# ============================================================

def yuv_frame_index_for_poc(poc: int, frame_indices: np.ndarray, args: argparse.Namespace) -> int:
    if args.frame_index_mode == "frame_indices":
        return int(args.seq_start) + int(frame_indices[int(poc)])
    return int(args.seq_start) + int(poc)


def collect_pair_observations(
    pair: tuple[int, int, float, str],
    seq_yuv: str | Path,
    width: int,
    height: int,
    bitdepth: int,
    frame_indices: np.ndarray,
    K_base: np.ndarray,
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    depth: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target, ref, pair_weight, kind = pair
    target = int(target)
    ref = int(ref)
    tar_yuv_idx = yuv_frame_index_for_poc(target, frame_indices, args)
    ref_yuv_idx = yuv_frame_index_for_poc(ref, frame_indices, args)
    target_y = read_y_frame(seq_yuv, width, height, bitdepth, tar_yuv_idx)
    ref_y = read_y_frame(seq_yuv, width, height, bitdepth, ref_yuv_idx)
    depth_img = np.asarray(depth[target], dtype=np.float32)

    map_x, map_y, valid = camera_map_w2c_np(
        target=target,
        ref=ref,
        width=width,
        height=height,
        K=K_base,
        rvecs=rvecs_base,
        tvecs=tvecs_base,
        depth_img=depth_img,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.render_row_batch,
    )
    cam_warp = remap_y(ref_y, map_x, map_y)
    base_mask_u8 = make_valid_mask_u8(valid, erode=args.ecc_valid_erode)
    M, success, cc, ecc_mask_u8, ecc_stats, structure = estimate_pair_structure_ecc_stable(
        target_y=target_y,
        cam_warp_y=cam_warp,
        valid_mask_u8=base_mask_u8,
        depth_img=depth_img,
        bitdepth=bitdepth,
        args=args,
    )

    ys, xs = np.where(ecc_mask_u8 > 0)
    n_mask = int(xs.size)
    cp_bias = transform_cp_bias(M, width, height, args.ecc_cp_num).astype(np.float32)
    cost_cam = calc_cost(target_y, cam_warp, valid, bitdepth)

    if (not success) or n_mask < int(args.min_obs_per_pair):
        info = {
            "target": target,
            "ref": ref,
            "target_yuv_idx": int(tar_yuv_idx),
            "ref_yuv_idx": int(ref_yuv_idx),
            "pair_weight": float(pair_weight),
            "kind": kind,
            "success": bool(success),
            "ecc_cc": None if cc is None else float(cc),
            "num_mask_pixels": n_mask,
            "num_observations": 0,
            "cp_bias_raw": cp_bias.astype(float).tolist(),
            "base_cam_cost": cost_cam,
            "ecc_stats": ecc_stats,
        }
        return {"target": np.empty(0, np.int32)}, info

    if args.max_obs_per_pair > 0 and xs.size > int(args.max_obs_per_pair):
        sel = rng.choice(xs.size, size=int(args.max_obs_per_pair), replace=False)
        xs = xs[sel]
        ys = ys[sel]

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dst = apply_transform_points(M, pts)
    bias = (dst - pts) * float(args.ecc_alpha)
    if float(args.ecc_bias_max_abs) > 0.0:
        bias = np.clip(bias, -float(args.ecc_bias_max_abs), float(args.ecc_bias_max_abs)).astype(np.float32)

    qx = map_x[ys, xs].astype(np.float32) + bias[:, 0]
    qy = map_y[ys, xs].astype(np.float32) + bias[:, 1]
    dep = depth_img[ys, xs].astype(np.float32)
    ok = (
        np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(dep)
        & (dep > 0.0)
        & (map_x[ys, xs] >= 0.0)
        & (map_y[ys, xs] >= 0.0)
        & (qx >= 0.0)
        & (qx <= width - 1.0)
        & (qy >= 0.0)
        & (qy <= height - 1.0)
    )
    xs = xs[ok]
    ys = ys[ok]
    qx = qx[ok]
    qy = qy[ok]
    dep = dep[ok]
    if xs.size < int(args.min_obs_per_pair):
        info = {
            "target": target,
            "ref": ref,
            "target_yuv_idx": int(tar_yuv_idx),
            "ref_yuv_idx": int(ref_yuv_idx),
            "pair_weight": float(pair_weight),
            "kind": kind,
            "success": bool(success),
            "ecc_cc": None if cc is None else float(cc),
            "num_mask_pixels": n_mask,
            "num_observations": int(xs.size),
            "reason": "too_few_after_filter",
            "cp_bias_raw": cp_bias.astype(float).tolist(),
            "base_cam_cost": cost_cam,
            "ecc_stats": ecc_stats,
        }
        return {"target": np.empty(0, np.int32)}, info

    structure_w = 0.25 + structure[ys, xs].astype(np.float32)
    cc_w = 1.0 if cc is None or not np.isfinite(cc) else float(max(0.05, min(2.0, cc + 1.0)))
    weights = structure_w * cc_w * float(pair_weight)
    obs = {
        "target": np.full(xs.shape[0], target, dtype=np.int32),
        "ref": np.full(xs.shape[0], ref, dtype=np.int32),
        "px": xs.astype(np.float32),
        "py": ys.astype(np.float32),
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "depth": dep.astype(np.float32),
        "weight": weights.astype(np.float32),
    }
    info = {
        "target": target,
        "ref": ref,
        "target_yuv_idx": int(tar_yuv_idx),
        "ref_yuv_idx": int(ref_yuv_idx),
        "pair_weight": float(pair_weight),
        "kind": kind,
        "success": bool(success),
        "ecc_cc": None if cc is None else float(cc),
        "motion_type": "homography" if int(args.ecc_cp_num) == 4 else "affine",
        "num_mask_pixels": n_mask,
        "num_observations": int(xs.shape[0]),
        "cp_bias_raw": cp_bias.astype(float).tolist(),
        "ecc_alpha": float(args.ecc_alpha),
        "ecc_bias_max_abs": float(args.ecc_bias_max_abs),
        "matrix": np.asarray(M, dtype=float).tolist(),
        "base_cam_cost": cost_cam,
        "ecc_stats": ecc_stats,
    }
    return obs, info


def concat_observations(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ["target", "ref", "px", "py", "qx", "qy", "depth", "weight"]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [o[k] for o in obs_list if k in o and o[k].size > 0]
        if not vals:
            raise RuntimeError(f"No observations for key {k}")
        out[k] = np.concatenate(vals, axis=0)
    return out


# ============================================================
# Fitting: W2C R + GOP focal + tiny t, fixed depth
# ============================================================

def choose_batch_indices(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    if batch <= 0 or batch >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=int(batch), replace=False).astype(np.int64)


def robust_loss_from_err2(err2: torch.Tensor, loss_name: str, f_scale: float) -> torch.Tensor:
    f = float(max(f_scale, 1e-6))
    if loss_name == "linear":
        return err2
    if loss_name == "huber":
        err = torch.sqrt(err2.clamp_min(1e-12))
        return torch.where(err <= f, 0.5 * err * err, f * (err - 0.5 * f))
    if loss_name == "cauchy":
        return (f * f) * torch.log1p(err2 / (f * f))
    return 2.0 * (f * f) * (torch.sqrt(1.0 + err2 / (f * f)) - 1.0)


def fit_rf_tiny_t_w2c(
    observations: dict[str, np.ndarray],
    rvecs_base: np.ndarray,
    tvecs_base: np.ndarray,
    K_base: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.torch_float64 else torch.float32
    rng = np.random.default_rng(int(args.seed))
    n_obs = int(observations["px"].shape[0])
    n_frames = int(rvecs_base.shape[0])

    target = torch.tensor(observations["target"], device=device, dtype=torch.long)
    ref = torch.tensor(observations["ref"], device=device, dtype=torch.long)
    px = torch.tensor(observations["px"], device=device, dtype=dtype)
    py = torch.tensor(observations["py"], device=device, dtype=dtype)
    qx = torch.tensor(observations["qx"], device=device, dtype=dtype)
    qy = torch.tensor(observations["qy"], device=device, dtype=dtype)
    depth = torch.tensor(observations["depth"], device=device, dtype=dtype).clamp_min(1e-12)

    w_np = observations["weight"].astype(np.float64)
    good = np.isfinite(w_np) & (w_np > 0)
    med = float(np.median(w_np[good])) if np.any(good) else 1.0
    w_np = np.clip(w_np / max(med, 1e-12), 1e-4, 100.0).astype(np.float32)
    weight = torch.tensor(w_np, device=device, dtype=dtype)

    r_base = torch.tensor(rvecs_base, device=device, dtype=dtype)
    t_base = torch.tensor(tvecs_base, device=device, dtype=dtype)
    r_delta = torch.nn.Parameter(torch.zeros_like(r_base))
    t_delta = torch.nn.Parameter(torch.zeros_like(t_base))
    anchor = int(args.anchor_poc if args.anchor_poc >= 0 else 0)
    if not (0 <= anchor < n_frames):
        raise ValueError(f"--anchor-poc {anchor} out of range N={n_frames}")

    pose_code_enabled = any(
        float(v) > 0.0
        for v in [
            args.pose_code_rot_rate_weight,
            args.pose_code_trans_rate_weight,
            args.pose_code_rot_quant_weight,
            args.pose_code_trans_quant_weight,
        ]
    )
    if pose_code_enabled and anchor != 0:
        raise ValueError(
            "Predictive pose coding loss requires --anchor-poc 0 because the GOP first frame is implicit I/0."
        )

    f_base_x = float(K_base[0, 0])
    f_base_y = float(K_base[1, 1])
    if args.f_init == "geom":
        f0 = math.sqrt(max(f_base_x * f_base_y, 1e-12))
    elif args.f_init == "fx":
        f0 = f_base_x
    elif args.f_init == "fy":
        f0 = f_base_y
    else:
        f0 = 0.5 * (f_base_x + f_base_y)

    focal_mode = str(args.focal_mode)
    if focal_mode == "single":
        log_f_delta: Optional[torch.nn.Parameter] = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
    elif focal_mode == "separate":
        log_f_delta = torch.nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
    elif focal_mode == "fixed":
        log_f_delta = None
    else:
        raise ValueError(focal_mode)

    params: list[dict[str, Any]] = []
    if not args.freeze_r:
        params.append({"params": [r_delta], "lr": float(args.lr_rot)})
    else:
        r_delta.requires_grad_(False)
    if not args.freeze_t:
        params.append({"params": [t_delta], "lr": float(args.lr_trans)})
    else:
        t_delta.requires_grad_(False)
    if log_f_delta is not None:
        params.append({"params": [log_f_delta], "lr": float(args.lr_focal)})
    if not params:
        raise RuntimeError("No trainable parameters: check --freeze-r/--freeze-t/--focal-mode fixed.")

    opt = torch.optim.Adam(params)
    cx = torch.tensor(float(K_base[0, 2]), device=device, dtype=dtype)
    cy = torch.tensor(float(K_base[1, 2]), device=device, dtype=dtype)

    def current_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor] = 0.0
        td[anchor] = 0.0
        r_cur = r_base + rd
        t_cur = t_base + td
        if focal_mode == "fixed":
            fx = torch.tensor(f_base_x, device=device, dtype=dtype)
            fy = torch.tensor(f_base_y, device=device, dtype=dtype)
        elif focal_mode == "single":
            assert log_f_delta is not None
            f = torch.tensor(float(f0), device=device, dtype=dtype) * torch.exp(log_f_delta[0])
            fx = f
            fy = f
        else:
            assert log_f_delta is not None
            fx = torch.tensor(f_base_x, device=device, dtype=dtype) * torch.exp(log_f_delta[0])
            fy = torch.tensor(f_base_y, device=device, dtype=dtype) * torch.exp(log_f_delta[1])
        return r_cur, t_cur, fx, fy

    def predictive_pose_coding_terms(
        r_cur: torch.Tensor,
        t_cur: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), dtype=dtype, device=device)
        if n_frames <= 1:
            empty = torch.empty((0, 3), dtype=dtype, device=device)
            return {
                "rot_residual": empty,
                "trans_residual": empty,
                "rot_recon_error": empty,
                "trans_recon_error": empty,
                "rot_rate_proxy": zero,
                "trans_rate_proxy": zero,
                "rot_quant_mse": zero,
                "trans_quant_mse": zero,
            }

        R_abs = torch_rodrigues(r_cur)

        # Re-anchor the GOP to frame 0. This leaves every pair-wise relative pose
        # unchanged while making the first stored pose exactly I/0.
        R_anchor = R_abs[anchor]
        t_anchor = t_cur[anchor]
        R_local = torch.matmul(R_abs, R_anchor.transpose(0, 1))
        t_local = t_cur - torch.matmul(R_local, t_anchor.reshape(3, 1)).squeeze(-1)

        if args.pose_code_trans_domain == "tvec":
            trans_signal = t_local
        elif args.pose_code_trans_domain == "center":
            trans_signal = -torch.bmm(R_local.transpose(1, 2), t_local.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(args.pose_code_trans_domain)

        R_rec_prev = torch.eye(3, dtype=dtype, device=device)
        trans_rec_prev = torch.zeros(3, dtype=dtype, device=device)

        rot_residuals: list[torch.Tensor] = []
        trans_residuals: list[torch.Tensor] = []
        rot_recon_errors: list[torch.Tensor] = []
        trans_recon_errors: list[torch.Tensor] = []
        rot_symbol_quant_errors: list[torch.Tensor] = []
        trans_symbol_quant_errors: list[torch.Tensor] = []

        for i in range(1, n_frames):
            R_res = R_local[i] @ R_rec_prev.transpose(0, 1)
            r_res = torch_matrix_to_rotvec(R_res)
            r_res_hard = hard_uniform_quantize(r_res, args.pose_code_rot_qstep)
            r_res_hat = r_res + (r_res_hard - r_res).detach()
            R_res_hat = torch_rodrigues(r_res_hat.unsqueeze(0))[0]
            R_rec = R_res_hat @ R_rec_prev

            trans_res = trans_signal[i] - trans_rec_prev
            trans_res_hard = hard_uniform_quantize(trans_res, args.pose_code_trans_qstep)
            trans_res_hat = trans_res + (trans_res_hard - trans_res).detach()
            trans_rec = trans_rec_prev + trans_res_hat

            R_error = R_local[i] @ R_rec.transpose(0, 1)
            r_recon_error = torch_matrix_to_rotvec(R_error)
            trans_recon_error = trans_signal[i] - trans_rec

            rot_residuals.append(r_res)
            trans_residuals.append(trans_res)
            rot_recon_errors.append(r_recon_error)
            trans_recon_errors.append(trans_recon_error)

            # Use detached hard-bin centers so this term has a real gradient
            # that attracts each stored residual component to a quantizer bin.
            # A naive (x - STE_quantize(x)) loss would have zero gradient.
            rot_symbol_quant_errors.append(r_res - r_res_hard.detach())
            trans_symbol_quant_errors.append(trans_res - trans_res_hard.detach())

            R_rec_prev = R_rec
            trans_rec_prev = trans_rec

        rot_residual = torch.stack(rot_residuals, dim=0)
        trans_residual = torch.stack(trans_residuals, dim=0)
        rot_recon_error = torch.stack(rot_recon_errors, dim=0)
        trans_recon_error = torch.stack(trans_recon_errors, dim=0)
        rot_symbol_quant_error = torch.stack(rot_symbol_quant_errors, dim=0)
        trans_symbol_quant_error = torch.stack(trans_symbol_quant_errors, dim=0)

        rot_rate_proxy = pose_symbol_rate_proxy(
            rot_residual / float(args.pose_code_rot_qstep),
            args.pose_code_rate_mode,
        )
        trans_rate_proxy = pose_symbol_rate_proxy(
            trans_residual / float(args.pose_code_trans_qstep),
            args.pose_code_rate_mode,
        )
        # Dimensionless distance from the actual stored residual values to
        # their nearest quantizer reconstruction levels. Unlike an STE-only
        # reconstruction loss, this provides a non-zero bin-attraction gradient.
        rot_quant_mse = torch.mean((rot_symbol_quant_error / float(args.pose_code_rot_qstep)) ** 2)
        trans_quant_mse = torch.mean((trans_symbol_quant_error / float(args.pose_code_trans_qstep)) ** 2)

        return {
            "rot_residual": rot_residual,
            "trans_residual": trans_residual,
            "rot_recon_error": rot_recon_error,
            "trans_recon_error": trans_recon_error,
            "rot_rate_proxy": rot_rate_proxy,
            "trans_rate_proxy": trans_rate_proxy,
            "rot_quant_mse": rot_quant_mse,
            "trans_quant_mse": trans_quant_mse,
        }

    def project_indices(idx_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        r_cur, t_cur, fx, fy = current_params()
        R = torch_rodrigues(r_cur)
        tc = target[idx]
        rc = ref[idx]
        R_rel = torch.bmm(R[rc], R[tc].transpose(1, 2))
        t_rel = t_cur[rc] - torch.bmm(R_rel, t_cur[tc].unsqueeze(-1)).squeeze(-1)
        ray_x = (px[idx] - cx) / fx
        ray_y = (py[idx] - cy) / fy
        ray_z = torch.full_like(ray_x, float(args.z_sign))
        X_tar = depth[idx, None] * torch.stack([ray_x, ray_y, ray_z], dim=1)
        X_ref = torch.bmm(R_rel, X_tar.unsqueeze(-1)).squeeze(-1) + t_rel
        z = X_ref[:, 2]
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        denom = torch.where(torch.abs(z) > eps, z, torch.where(z >= 0, eps, -eps))
        u = fx * (X_ref[:, 0] / denom) + cx
        v = fy * (X_ref[:, 1] / denom) + cy
        return u, v, z

    def regularization() -> torch.Tensor:
        reg = torch.zeros((), dtype=dtype, device=device)
        rd = r_delta.clone()
        td = t_delta.clone()
        rd[anchor] = 0.0
        td[anchor] = 0.0

        if args.rot_delta_prior_weight > 0:
            reg = reg + float(args.rot_delta_prior_weight) * torch.mean(rd * rd)
        if args.trans_delta_prior_weight > 0 and not args.freeze_t:
            reg = reg + float(args.trans_delta_prior_weight) * torch.mean(td * td)
        if args.pose_delta_smooth_weight > 0 and n_frames >= 2:
            reg = reg + float(args.pose_delta_smooth_weight) * (
                torch.mean((rd[1:] - rd[:-1]) ** 2)
                + torch.mean((td[1:] - td[:-1]) ** 2)
            )
        if log_f_delta is not None and args.f_prior_weight > 0:
            reg = reg + float(args.f_prior_weight) * torch.mean(log_f_delta * log_f_delta)

        if pose_code_enabled and n_frames >= 2:
            r_cur, t_cur, _, _ = current_params()
            coding = predictive_pose_coding_terms(r_cur, t_cur)
            reg = reg + float(args.pose_code_rot_rate_weight) * coding["rot_rate_proxy"]
            reg = reg + float(args.pose_code_trans_rate_weight) * coding["trans_rate_proxy"]
            reg = reg + float(args.pose_code_rot_quant_weight) * coding["rot_quant_mse"]
            reg = reg + float(args.pose_code_trans_quant_weight) * coding["trans_quant_mse"]
        return reg

    def loss_for_batch(idx_np: np.ndarray) -> torch.Tensor:
        u, v, z = project_indices(idx_np)
        idx = torch.tensor(idx_np, device=device, dtype=torch.long)
        dx = u - qx[idx]
        dy = v - qy[idx]
        err2 = dx * dx + dy * dy
        pix = robust_loss_from_err2(err2, args.robust_loss, args.robust_f_scale)
        if args.z_min > 0:
            zbad = torch.relu(float(args.z_min) - z * float(args.z_sign))
            pix = pix + float(args.z_penalty) * zbad * zbad
        ww = weight[idx]
        return torch.sum(ww * pix) / (torch.sum(ww) + 1e-12) + regularization()

    @torch.no_grad()
    def eval_all(batch: int) -> dict[str, Any]:
        errs: list[np.ndarray] = []
        for s in range(0, n_obs, int(batch)):
            e = min(n_obs, s + int(batch))
            idx_np = np.arange(s, e, dtype=np.int64)
            idx = torch.tensor(idx_np, device=device, dtype=torch.long)
            u, v, z = project_indices(idx_np)
            err = torch.sqrt((u - qx[idx]) ** 2 + (v - qy[idx]) ** 2).detach().cpu().numpy()
            zz = z.detach().cpu().numpy()
            err[zz * float(args.z_sign) <= args.z_min] = np.inf
            errs.append(err)
        err_all = np.concatenate(errs, axis=0)
        finite = np.isfinite(err_all)
        return {
            "count": int(np.count_nonzero(finite)),
            "mean": float(np.mean(err_all[finite])) if np.any(finite) else None,
            "median": float(np.median(err_all[finite])) if np.any(finite) else None,
            "p90": float(np.percentile(err_all[finite], 90)) if np.any(finite) else None,
            "p95": float(np.percentile(err_all[finite], 95)) if np.any(finite) else None,
        }

    @torch.no_grad()
    def eval_pose_predictive_coding() -> dict[str, Any]:
        r_cur, t_cur, _, _ = current_params()
        coding = predictive_pose_coding_terms(r_cur, t_cur)

        rot_residual = coding["rot_residual"].detach().cpu().numpy().astype(np.float64)
        trans_residual = coding["trans_residual"].detach().cpu().numpy().astype(np.float64)
        rot_recon_error = coding["rot_recon_error"].detach().cpu().numpy().astype(np.float64)
        trans_recon_error = coding["trans_recon_error"].detach().cpu().numpy().astype(np.float64)

        rot_qindex = np.rint(rot_residual / float(args.pose_code_rot_qstep)).astype(np.int64)
        trans_qindex = np.rint(trans_residual / float(args.pose_code_trans_qstep)).astype(np.int64)

        return {
            "model": "first frame implicit I/0; closed-loop previous reconstructed absolute pose prediction",
            "coded_frame_count": int(max(0, n_frames - 1)),
            "first_frame_signaled": False,
            "rotation_representation": "Rodrigues vector of R_local_i @ R_reconstructed_previous.T",
            "translation_domain": str(args.pose_code_trans_domain),
            "rotation_qstep": float(args.pose_code_rot_qstep),
            "translation_qstep": float(args.pose_code_trans_qstep),
            "rate_mode": str(args.pose_code_rate_mode),
            "rot_rate_proxy": float(coding["rot_rate_proxy"].detach().cpu()),
            "trans_rate_proxy": float(coding["trans_rate_proxy"].detach().cpu()),
            "rot_quant_mse": float(coding["rot_quant_mse"].detach().cpu()),
            "trans_quant_mse": float(coding["trans_quant_mse"].detach().cpu()),
            "quant_loss_semantics": "dimensionless squared distance of each stored residual component to its nearest quantizer reconstruction level",
            "rot_residual_mean_abs": float(np.mean(np.abs(rot_residual))) if rot_residual.size else 0.0,
            "trans_residual_mean_abs": float(np.mean(np.abs(trans_residual))) if trans_residual.size else 0.0,
            "rot_quant_index_mean_abs": float(np.mean(np.abs(rot_qindex))) if rot_qindex.size else 0.0,
            "trans_quant_index_mean_abs": float(np.mean(np.abs(trans_qindex))) if trans_qindex.size else 0.0,
            "rot_zero_ratio": float(np.mean(rot_qindex == 0)) if rot_qindex.size else 1.0,
            "trans_zero_ratio": float(np.mean(trans_qindex == 0)) if trans_qindex.size else 1.0,
            "rot_residual_rvec": rot_residual.tolist(),
            "trans_residual": trans_residual.tolist(),
            "rot_quant_index": rot_qindex.tolist(),
            "trans_quant_index": trans_qindex.tolist(),
            "rot_recon_error_rvec": rot_recon_error.tolist(),
            "trans_recon_error": trans_recon_error.tolist(),
        }

    report: dict[str, Any] = {
        "device": str(device),
        "dtype": str(dtype),
        "num_observations": int(n_obs),
        "anchor_poc": int(anchor),
        "focal_mode": focal_mode,
        "f_init": str(args.f_init),
        "f0": float(f0),
        "pose_predictive_coding_loss_enabled": bool(pose_code_enabled),
        "pose_predictive_coding_options": {
            "rotation_qstep": float(args.pose_code_rot_qstep),
            "translation_qstep": float(args.pose_code_trans_qstep),
            "translation_domain": str(args.pose_code_trans_domain),
            "rate_mode": str(args.pose_code_rate_mode),
            "rotation_rate_weight": float(args.pose_code_rot_rate_weight),
            "translation_rate_weight": float(args.pose_code_trans_rate_weight),
            "rotation_quant_weight": float(args.pose_code_rot_quant_weight),
            "translation_quant_weight": float(args.pose_code_trans_quant_weight),
        },
        "initial_pose_predictive_coding": eval_pose_predictive_coding(),
        "iterations": [],
    }

    for step in range(int(args.steps)):
        idx_np = choose_batch_indices(n_obs, int(args.batch_size), rng)
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(idx_np)
        if not torch.isfinite(loss):
            log(
                f"RF step {step:04d}/{args.steps}: non-finite loss={float(loss.detach().cpu())}; "
                "stop fitting and keep current parameters"
            )
            break
        loss.backward()

        if args.grad_clip > 0:
            train_params = [p for g in params for p in g["params"] if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(train_params, float(args.grad_clip))
        opt.step()

        with torch.no_grad():
            r_delta[anchor].zero_()
            t_delta[anchor].zero_()
            if args.max_trans_delta > 0 and not args.freeze_t:
                t_delta.clamp_(-float(args.max_trans_delta), float(args.max_trans_delta))
            if log_f_delta is not None:
                log_f_delta.clamp_(-float(args.f_log_max_delta), float(args.f_log_max_delta))

        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            stat = eval_all(batch=int(args.eval_batch_size))
            coding_stat = eval_pose_predictive_coding()
            _, _, fx_cur, fy_cur = current_params()
            info = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "residual_px": stat,
                "fx": float(fx_cur.detach().cpu()),
                "fy": float(fy_cur.detach().cpu()),
                "max_abs_r_delta": float(torch.max(torch.abs(r_delta)).detach().cpu()),
                "max_abs_t_delta": float(torch.max(torch.abs(t_delta)).detach().cpu()),
                "pose_code_rot_rate_proxy": coding_stat["rot_rate_proxy"],
                "pose_code_trans_rate_proxy": coding_stat["trans_rate_proxy"],
                "pose_code_rot_qindex_mean_abs": coding_stat["rot_quant_index_mean_abs"],
                "pose_code_trans_qindex_mean_abs": coding_stat["trans_quant_index_mean_abs"],
                "pose_code_rot_zero_ratio": coding_stat["rot_zero_ratio"],
                "pose_code_trans_zero_ratio": coding_stat["trans_zero_ratio"],
            }
            report["iterations"].append(info)
            log(
                f"RF step {step:04d}/{args.steps}: loss={info['loss']:.6f}, "
                f"mean={stat['mean']}, p95={stat['p95']}, "
                f"fx={info['fx']:.4f}, fy={info['fy']:.4f}, "
                f"codeR={info['pose_code_rot_qindex_mean_abs']:.3f}, "
                f"codeT={info['pose_code_trans_qindex_mean_abs']:.3f}, "
                f"zeroR={info['pose_code_rot_zero_ratio']:.3f}, "
                f"zeroT={info['pose_code_trans_zero_ratio']:.3f}"
            )

    report["final_residual_px"] = eval_all(batch=int(args.eval_batch_size))
    report["final_pose_predictive_coding"] = eval_pose_predictive_coding()

    with torch.no_grad():
        r_final, t_final, fx_final, fy_final = current_params()
        K_final = np.array(
            [
                [float(fx_final.detach().cpu()), 0.0, float(K_base[0, 2])],
                [0.0, float(fy_final.detach().cpu()), float(K_base[1, 2])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return (
        K_final,
        r_final.detach().cpu().numpy().astype(np.float64),
        t_final.detach().cpu().numpy().astype(np.float64),
        report,
    )


# ============================================================
# Output writing
# ============================================================

def write_camera_jsonl_canonical(
    path: Path,
    source_npz: Path,
    source_camera_jsonl: Optional[Path],
    frame_indices: np.ndarray,
    K_final: np.ndarray,
    r_final: np.ndarray,
    t_final: np.ndarray,
    z_sign: float,
    copied_header: Optional[dict[str, Any]],
    depth_yuv_meta: Optional[dict[str, Any]],
    refine_report_summary: dict[str, Any],
) -> None:
    ensure_parent(path)
    R_all = all_rotation_matrices_np(r_final)
    copied_depth = None
    if copied_header is not None:
        copied_depth = copied_header.get("depth_output") or copied_header.get("depth_yuv")
    depth_output = depth_yuv_meta if depth_yuv_meta is not None else copied_depth
    header = {
        "type": "header",
        "format": "fixedK_gop_nn_affine_cp_rf_pose_coding_refine_v1",
        "source_npz": os.path.abspath(source_npz),
        "source_camera_jsonl": os.path.abspath(source_camera_jsonl) if source_camera_jsonl else None,
        "frame_count": int(len(frame_indices)),
        "frame_indices": frame_indices.astype(int).tolist(),
        "intrinsic_mode": "rap_fixed_affine_cp_rf_pose_coding_refined",
        "intrinsic": {
            "fx": float(K_final[0, 0]),
            "fy": float(K_final[1, 1]),
            "cx": float(K_final[0, 2]),
            "cy": float(K_final[1, 2]),
            "z_sign": float(z_sign),
        },
        "intrinsic_delta_order": [],
        "intrinsic_delta_bits_per_frame": 0,
        "pose_storage": {
            "absolute_pose": "camera_from_world / W2C in fixed-K canonical camera coordinates",
            "relative_pair_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target; X_ref=R_rel*X_target+t_rel",
            "adjacent_current_to_previous_fields": "also written for compatibility",
            "future_codec_model": "first GOP frame implicit I/0; rotation uses 3-component Rodrigues residual against previous reconstructed rotation; translation uses previous reconstructed local absolute tvec residual",
        },
        "depth_output": depth_output,
        "refinement": refine_report_summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(header), ensure_ascii=False) + "\n")
        for i in range(len(frame_indices)):
            rec: dict[str, Any] = {
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec_abs": r_final[i].astype(float).tolist(),
                "tvec_abs": t_final[i].astype(float).tolist(),
                "extrinsic_abs": np.concatenate([R_all[i], t_final[i].reshape(3, 1)], axis=1).astype(float).tolist(),
            }
            if i == 0:
                rec["rvec_current_to_previous"] = [0.0, 0.0, 0.0]
                rec["tvec_current_to_previous"] = [0.0, 0.0, 0.0]
            else:
                R_rel = R_all[i - 1] @ R_all[i].T
                t_rel = t_final[i - 1] - R_rel @ t_final[i]
                rv, _ = cv2.Rodrigues(R_rel.astype(np.float64))
                rec["rvec_current_to_previous"] = rv.reshape(3).astype(float).tolist()
                rec["tvec_current_to_previous"] = t_rel.astype(float).tolist()
            f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")


def copy_or_write_depth_yuv(
    stage: dict[str, Any],
    src_depth_yuv: Optional[Path],
    out_depth_yuv: Path,
    copied_header: Optional[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ensure_parent(out_depth_yuv)
    if src_depth_yuv is not None and src_depth_yuv.is_file():
        if src_depth_yuv.resolve() != out_depth_yuv.resolve():
            shutil.copy2(src_depth_yuv, out_depth_yuv)
        meta = None
        if copied_header is not None:
            meta = copied_header.get("depth_output") or copied_header.get("depth_yuv")
        if isinstance(meta, dict):
            out = dict(meta)
            out["depth_yuv"] = str(out_depth_yuv)
            out["copied_from"] = str(src_depth_yuv)
            return out
        return {
            "depth_yuv": str(out_depth_yuv),
            "depth_yuv_format": "yuv420p10le",
            "depth_yuv_semantics": "copied unchanged from source canonicalize output",
            "copied_from": str(src_depth_yuv),
        }
    scale_meta = choose_depth_scale_fixed_point(
        stage["depth"],
        args.depth_scale_percentile,
        args.depth_scale_precision,
        10,
    )
    return write_depth_yuv420p10le_linear(out_depth_yuv, stage["depth"], scale_meta)


def write_refined_geometry_npz(
    out_npz: Path,
    stage: dict[str, Any],
    K_final: np.ndarray,
    r_final: np.ndarray,
    t_final: np.ndarray,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ensure_parent(out_npz)
    data = stage["data"]
    payload: dict[str, Any] = {}
    for k in data.files:
        payload[k] = data[k]
    payload["K_before_affine_cp_rf"] = stage["K"].astype(np.float32)
    payload["rvec_abs_before_affine_cp_rf"] = stage["rvecs"].astype(np.float32)
    payload["tvec_abs_before_affine_cp_rf"] = stage["tvecs"].astype(np.float32)
    payload["K_fixed"] = K_final.astype(np.float32)
    payload["K_refined"] = K_final.astype(np.float32)
    payload["rvec_abs_refined"] = r_final.astype(np.float32)
    payload["tvec_abs_refined"] = t_final.astype(np.float32)
    payload["rvec_abs_final"] = r_final.astype(np.float32)
    payload["tvec_abs_final"] = t_final.astype(np.float32)
    payload["affine_cp_rf_pose_coding_refine_result_json"] = np.asarray(
        json.dumps(to_jsonable(result), ensure_ascii=False),
        dtype=object,
    )
    if args.compressed_npz:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)


def write_manifest(
    out_manifest: Path,
    source_npz: Path,
    sidecars: dict[str, Optional[Path]],
    out_paths: dict[str, Path],
    stage: dict[str, Any],
    K_final: np.ndarray,
    result: dict[str, Any],
    depth_meta: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ensure_parent(out_manifest)
    manifest = {
        "source_npz": os.path.abspath(source_npz),
        "source_camera_jsonl": os.path.abspath(sidecars["camera_jsonl"]) if sidecars.get("camera_jsonl") else None,
        "source_depth_yuv": os.path.abspath(sidecars["depth_yuv"]) if sidecars.get("depth_yuv") else None,
        "outputs": {k: os.path.abspath(v) for k, v in out_paths.items()},
        "frame_count": int(stage["depth"].shape[0]),
        "size": {"width": int(stage["depth"].shape[2]), "height": int(stage["depth"].shape[1])},
        "K_before_affine_cp_rf": stage["K"].astype(float).tolist(),
        "K_final": K_final.astype(float).tolist(),
        "depth_yuv": depth_meta,
        "affine_cp_rf_pose_coding_refine": result,
        "options": vars(args),
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=2, ensure_ascii=False)
        f.write("\n")


# ============================================================
# Sequence YUV resolution
# ============================================================

def resolve_seq_yuv_for_npz(
    geometry_npz: Path,
    src_root: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    base = derive_base_from_geometry_npz(geometry_npz)
    if args.seq_yuv:
        p = Path(args.seq_yuv).expanduser().resolve()
        return p if p.is_file() else None
    rel_key = (
        str(geometry_npz.relative_to(src_root))
        if src_root in geometry_npz.parents or geometry_npz == src_root
        else str(geometry_npz)
    )
    if args.seq_yuv_map_json:
        with open(args.seq_yuv_map_json, "r", encoding="utf-8") as f:
            mp = json.load(f)
        for k in [str(geometry_npz), rel_key, base, geometry_npz.stem]:
            if k in mp:
                p = Path(mp[k]).expanduser().resolve()
                return p if p.is_file() else None
    if args.seq_yuv_root:
        root = Path(args.seq_yuv_root).expanduser().resolve()
        candidates: list[Path] = []
        try:
            rel_dir = geometry_npz.parent.relative_to(src_root)
            candidates.extend(
                [
                    root / rel_dir / f"{base}.yuv",
                    root / rel_dir / f"{base}{args.seq_yuv_suffix}",
                ]
            )
        except Exception:
            pass
        candidates.extend([root / f"{base}.yuv", root / f"{base}{args.seq_yuv_suffix}"])
        for c in candidates:
            if c.is_file():
                return c.resolve()
        hits = sorted(root.rglob(f"*{base}*.yuv"))
        if hits:
            return hits[0].resolve()
    return None


# ============================================================
# Main per-file pipeline
# ============================================================

def run_one(
    geometry_npz: Path,
    src_root: Path,
    out_prefix: Path,
    args: argparse.Namespace,
) -> int:
    out_paths = canonical_paths_from_prefix(out_prefix)
    for p in out_paths.values():
        if p.exists():
            if args.force:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --force.")
        ensure_parent(p)

    sidecars = find_canonical_sidecars(geometry_npz)
    seq_yuv = resolve_seq_yuv_for_npz(geometry_npz, src_root, args)
    if seq_yuv is None:
        raise RuntimeError("Sequence YUV not found. Provide --seq-yuv, --seq-yuv-root, or --seq-yuv-map-json.")

    stage = load_canonical_npz(geometry_npz)
    K_base = stage["K"]
    r_base = stage["rvecs"]
    t_base = stage["tvecs"]
    depth = stage["depth"]
    frame_indices = stage["frame_indices"]
    n, h, w = depth.shape

    if args.width is not None and int(args.width) != w:
        raise ValueError(f"--width {args.width} != NPZ width {w}")
    if args.height is not None and int(args.height) != h:
        raise ValueError(f"--height {args.height} != NPZ height {h}")

    seq_count = count_frames_yuv420(seq_yuv, w, h, args.bitdepth)
    if seq_count <= 0:
        raise RuntimeError(f"Sequence YUV has no full frames: {seq_yuv}")

    copied_header = load_first_jsonl_object(sidecars.get("camera_jsonl"))
    pairs = build_pair_list(args, stage)
    rng = np.random.default_rng(int(args.seed))
    log(f"Loaded canonical NPZ: frames={n}, size={w}x{h}, pairs={len(pairs)}, seq_yuv={seq_yuv}")
    log(f"K base: fx={K_base[0,0]:.6f}, fy={K_base[1,1]:.6f}, focal_mode={args.focal_mode}")
    log(
        "Pose coding model: first frame implicit I/0, "
        f"rot_qstep={args.pose_code_rot_qstep:g}, trans_qstep={args.pose_code_trans_qstep:g}, "
        f"trans_domain={args.pose_code_trans_domain}"
    )

    pair_info: list[dict[str, Any]] = []
    obs_list: list[dict[str, np.ndarray]] = []
    for idx, pair in enumerate(pairs, start=1):
        target, ref, weight, kind = pair
        tar_yuv_idx = yuv_frame_index_for_poc(target, frame_indices, args)
        ref_yuv_idx = yuv_frame_index_for_poc(ref, frame_indices, args)
        if tar_yuv_idx < 0 or tar_yuv_idx >= seq_count or ref_yuv_idx < 0 or ref_yuv_idx >= seq_count:
            info = {
                "target": int(target),
                "ref": int(ref),
                "success": False,
                "reason": "yuv_index_out_of_range",
                "target_yuv_idx": int(tar_yuv_idx),
                "ref_yuv_idx": int(ref_yuv_idx),
            }
            pair_info.append(info)
            log(f"PAIR {idx:03d}/{len(pairs):03d} {target}->{ref}: skip yuv index out of range")
            continue

        log(f"PAIR ECC {idx:03d}/{len(pairs):03d}: target={target}, ref={ref}, weight={weight:.4g}, kind={kind}")
        obs, info = collect_pair_observations(
            pair=pair,
            seq_yuv=seq_yuv,
            width=w,
            height=h,
            bitdepth=args.bitdepth,
            frame_indices=frame_indices,
            K_base=K_base,
            rvecs_base=r_base,
            tvecs_base=t_base,
            depth=depth,
            args=args,
            rng=rng,
        )
        pair_info.append(info)
        if obs.get("target", np.empty(0)).size > 0:
            obs_list.append(obs)
        log(
            f"PAIR result {target}->{ref}: success={info.get('success')}, "
            f"cc={info.get('ecc_cc')}, obs={info.get('num_observations')}, "
            f"base_psnr={None if info.get('base_cam_cost') is None else info['base_cam_cost'].get('psnr')}"
        )

    if not obs_list:
        raise RuntimeError("No valid pair ECC observations were generated.")
    observations = concat_observations(obs_list)
    log(f"Total observations: {observations['px'].shape[0]}")

    K_final, r_final, t_final, fit_report = fit_rf_tiny_t_w2c(
        observations,
        r_base,
        t_base,
        K_base,
        args,
    )

    result = {
        "method": {
            "description": "Batch affine-CP RF refinement with codec-aware closed-loop predictive-pose residual loss.",
            "depth": "fixed depth_canonical from input canonical NPZ",
            "supervision": "q_gt = base_camera_map + alpha * ECC_affine_bias(x,y)",
            "pose_convention": "camera_from_world / W2C: X_cam=R X_world+t",
            "relative_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target",
            "coordinate": "target pixel -> ref pixel",
            "pose_coding_loss": (
                "First GOP frame is implicit I/0. Each following absolute local pose is predicted from the previous "
                "quantized reconstructed pose. Rotation stores three Rodrigues residual components; translation stores "
                "three previous-reconstructed local absolute translation residual components."
            ),
        },
        "input": {
            "geometry_npz": str(geometry_npz),
            "camera_jsonl": str(sidecars["camera_jsonl"]) if sidecars.get("camera_jsonl") else None,
            "depth_yuv": str(sidecars["depth_yuv"]) if sidecars.get("depth_yuv") else None,
            "seq_yuv": str(seq_yuv),
        },
        "size": {"width": int(w), "height": int(h)},
        "frame_indices": frame_indices.astype(int).tolist(),
        "stage_source_keys": stage["source_keys"],
        "options": vars(args),
        "K_base": K_base.astype(float).tolist(),
        "K_refined": K_final.astype(float).tolist(),
        "focal_delta": {
            "fx_base": float(K_base[0, 0]),
            "fy_base": float(K_base[1, 1]),
            "fx_refined": float(K_final[0, 0]),
            "fy_refined": float(K_final[1, 1]),
            "fx_ratio": float(K_final[0, 0] / K_base[0, 0]),
            "fy_ratio": float(K_final[1, 1] / K_base[1, 1]),
            "fxfy_base_ratio": float(K_base[0, 0] / K_base[1, 1]),
            "fxfy_refined_ratio": float(K_final[0, 0] / K_final[1, 1]),
        },
        "pairs": pair_info,
        "fit_report": fit_report,
    }

    depth_meta = copy_or_write_depth_yuv(
        stage,
        sidecars.get("depth_yuv"),
        out_paths["depth_yuv"],
        copied_header,
        args,
    )
    summary = {
        "description": "affine CP R|t/focal refinement with predictive-pose coding loss; canonical naming preserved",
        "focal_mode": args.focal_mode,
        "ecc_cp_num": int(args.ecc_cp_num),
        "pair_source": args.pair_source,
        "num_pairs": len(pairs),
        "num_valid_pair_observation_sets": len(obs_list),
        "total_observations": int(observations["px"].shape[0]),
        "K_base": K_base.astype(float).tolist(),
        "K_refined": K_final.astype(float).tolist(),
        "pose_predictive_coding": fit_report.get("final_pose_predictive_coding"),
    }
    write_camera_jsonl_canonical(
        out_paths["camera_jsonl"],
        source_npz=geometry_npz,
        source_camera_jsonl=sidecars.get("camera_jsonl"),
        frame_indices=frame_indices,
        K_final=K_final,
        r_final=r_final,
        t_final=t_final,
        z_sign=args.z_sign,
        copied_header=copied_header,
        depth_yuv_meta=depth_meta,
        refine_report_summary=summary,
    )
    result["outputs"] = {k: str(v) for k, v in out_paths.items()}
    result["depth_yuv"] = depth_meta
    write_refined_geometry_npz(out_paths["geometry_npz"], stage, K_final, r_final, t_final, result, args)
    write_manifest(out_paths["manifest"], geometry_npz, sidecars, out_paths, stage, K_final, result, depth_meta, args)

    coding_final = fit_report.get("final_pose_predictive_coding", {})
    print("============================================================")
    print("Affine-CP RF + pose-coding refine done")
    print("============================================================")
    print(f"input geometry : {geometry_npz}")
    print(f"seq yuv        : {seq_yuv}")
    print(f"frames         : {n}")
    print(f"size           : {w}x{h}")
    print(f"focal mode     : {args.focal_mode}")
    print(f"K base         : fx={K_base[0,0]:.6f}, fy={K_base[1,1]:.6f}")
    print(f"K refined      : fx={K_final[0,0]:.6f}, fy={K_final[1,1]:.6f}")
    print(f"rot zero ratio : {coding_final.get('rot_zero_ratio')}")
    print(f"trn zero ratio : {coding_final.get('trans_zero_ratio')}")
    print(f"geometry npz   : {out_paths['geometry_npz']}")
    print(f"camera jsonl   : {out_paths['camera_jsonl']}")
    print(f"depth yuv      : {out_paths['depth_yuv']}")
    print(f"manifest       : {out_paths['manifest']}")
    print("============================================================")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch affine-CP R|t/focal refinement with codec-aware predictive-pose residual loss."
    )

    # Batch I/O.
    ap.add_argument("--src-root", required=True, help="Folder containing canonicalize outputs")
    ap.add_argument("--dst-root", required=True, help="Output root. Canonical filenames are preserved under this root.")
    ap.add_argument("--pattern", default=f"*{GEOM_SUFFIX}", help=f"Input geometry NPZ pattern. Default: *{GEOM_SUFFIX}")
    ap.add_argument("--layout", choices=["preserve", "flat"], default="preserve")
    ap.add_argument("--force", action="store_true", help="Overwrite dst outputs")
    ap.add_argument("--skip-invalid", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compressed-npz", action="store_true")

    # Original sequence YUV resolution.
    ap.add_argument("--seq-yuv", default="", help="Use one source YUV for all NPZs. Good for single-sequence run.")
    ap.add_argument("--seq-yuv-root", default="", help="Root to auto-search source YUV for each NPZ.")
    ap.add_argument("--seq-yuv-map-json", default="", help="JSON dict mapping full path / relative path / base name to source YUV path.")
    ap.add_argument("--seq-yuv-suffix", default=".yuv")
    ap.add_argument("--bitdepth", type=int, choices=[8, 10], default=10)
    ap.add_argument("--width", type=int, default=None, help="Optional sanity check against NPZ depth width")
    ap.add_argument("--height", type=int, default=None, help="Optional sanity check against NPZ depth height")
    ap.add_argument("--seq-start", type=int, default=0)
    ap.add_argument(
        "--frame-index-mode",
        choices=["local", "frame_indices"],
        default="local",
        help="local: YUV idx=seq_start+poc. frame_indices: YUV idx=seq_start+frame_indices[poc]",
    )

    # Pair selection.
    ap.add_argument("--pairs", default="", help="Optional pair list: target:ref[:weight], e.g. 16:0:2,16:32:2")
    ap.add_argument("--pair-source", choices=["npz", "adjacent", "dyadic", "all"], default="npz")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--no-bidirectional-pairs", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    # ECC pair residual extraction.
    ap.add_argument("--ecc-cp-num", type=int, choices=[3, 4], default=3, help="3=affine CP, 4=homography supervision")
    ap.add_argument("--ecc-scale", type=float, default=1.0, help="Run ECC at scaled resolution, then convert transform back to full resolution.")
    ap.add_argument("--structure-mode", choices=["scharr_mag", "scharr_l1", "scharr_x", "scharr_y", "scharr_x_weighted"], default="scharr_mag")
    ap.add_argument("--structure-keep-percent", type=float, default=35.0)
    ap.add_argument("--structure-mask-dilate", type=int, default=1)
    ap.add_argument("--structure-log-gain", type=float, default=20.0)
    ap.add_argument("--structure-pre-blur", type=int, default=0)
    ap.add_argument("--structure-ecc-rounds", type=int, default=2)
    ap.add_argument("--structure-residual-keep-percent", type=float, default=80.0)
    ap.add_argument("--static-residual-keep-percent", type=float, default=100.0, help="100 disables; 70 keeps lowest 70%% residual among structure pixels.")
    ap.add_argument("--depth-edge-keep-percentile", type=float, default=100.0, help="100 disables; 90 rejects top 10%% depth-gradient pixels.")
    ap.add_argument("--ecc-valid-erode", type=int, default=2)
    ap.add_argument("--ecc-iters", type=int, default=80)
    ap.add_argument("--ecc-eps", type=float, default=1e-5)
    ap.add_argument("--ecc-gauss", type=int, default=5)
    ap.add_argument("--ecc-min-mask-count", type=int, default=100)
    ap.add_argument("--ecc-alpha", type=float, default=1.0)
    ap.add_argument("--ecc-bias-max-abs", type=float, default=16.0, help="Clamp affine/homography bias per sampled pixel. <=0 disables")
    ap.add_argument("--max-obs-per-pair", type=int, default=25000)
    ap.add_argument("--min-obs-per-pair", type=int, default=500)

    # Fitting.
    ap.add_argument("--device", default="auto")
    ap.add_argument("--torch-float64", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--eval-batch-size", type=int, default=262144)
    ap.add_argument("--lr-rot", type=float, default=5e-4)
    ap.add_argument("--lr-trans", type=float, default=5e-5)
    ap.add_argument("--lr-focal", type=float, default=2e-4)
    ap.add_argument("--focal-mode", choices=["single", "separate", "fixed"], default="single", help="single forces fx=fy=f")
    ap.add_argument("--f-init", choices=["avg", "geom", "fx", "fy"], default="avg")
    ap.add_argument("--f-log-max-delta", type=float, default=0.05)
    ap.add_argument("--f-prior-weight", type=float, default=10.0)
    ap.add_argument("--rot-delta-prior-weight", type=float, default=1e-3)
    ap.add_argument("--trans-delta-prior-weight", type=float, default=100.0)
    ap.add_argument("--pose-delta-smooth-weight", type=float, default=1e-3)
    ap.add_argument("--max-trans-delta", type=float, default=0.0)
    ap.add_argument("--anchor-poc", type=int, default=0)
    ap.add_argument("--freeze-t", action="store_true")
    ap.add_argument("--freeze-r", action="store_true")
    ap.add_argument("--robust-loss", choices=["linear", "soft_l1", "huber", "cauchy"], default="soft_l1")
    ap.add_argument("--robust-f-scale", type=float, default=2.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--z-min", type=float, default=1e-4)
    ap.add_argument("--z-penalty", type=float, default=100.0)
    ap.add_argument("--render-row-batch", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=100)

    # Codec-aware predictive pose storage model.
    ap.add_argument(
        "--pose-code-rot-qstep",
        type=float,
        default=1e-5,
        help="Future codec rotation qstep in radians for each Rodrigues residual component.",
    )
    ap.add_argument(
        "--pose-code-trans-qstep",
        type=float,
        default=1e-5,
        help="Future codec translation qstep for each predictive translation residual component.",
    )
    ap.add_argument(
        "--pose-code-trans-domain",
        choices=["tvec", "center"],
        default="tvec",
        help="tvec matches the discussed design; center codes camera-center differences instead.",
    )
    ap.add_argument(
        "--pose-code-rate-mode",
        choices=["log1p", "l1", "l2"],
        default="log1p",
        help="Differentiable residual-rate proxy. log1p roughly follows variable-length signed integer coding.",
    )
    ap.add_argument("--pose-code-rot-rate-weight", type=float, default=1e-3)
    ap.add_argument("--pose-code-trans-rate-weight", type=float, default=1e-3)
    ap.add_argument("--pose-code-rot-quant-weight", type=float, default=1e-4)
    ap.add_argument("--pose-code-trans-quant-weight", type=float, default=1e-4)

    # Depth YUV fallback only.
    ap.add_argument("--depth-scale-precision", type=int, default=100000)
    ap.add_argument("--depth-scale-percentile", type=float, default=99.9)

    args = ap.parse_args()

    if args.ecc_gauss <= 0 or args.ecc_gauss % 2 == 0:
        raise ValueError("--ecc-gauss must be a positive odd integer")
    if not (0.0 < args.structure_keep_percent <= 100.0):
        raise ValueError("--structure-keep-percent must be in (0,100]")
    if args.structure_ecc_rounds < 1:
        raise ValueError("--structure-ecc-rounds must be >=1")
    if not (0.0 < args.structure_residual_keep_percent <= 100.0):
        raise ValueError("--structure-residual-keep-percent must be in (0,100]")
    if not (0.0 < args.static_residual_keep_percent <= 100.0):
        raise ValueError("--static-residual-keep-percent must be in (0,100]")
    if not (0.0 < args.depth_edge_keep_percentile <= 100.0):
        raise ValueError("--depth-edge-keep-percentile must be in (0,100]")
    if args.ecc_scale <= 0.0 or args.ecc_scale > 1.0:
        raise ValueError("--ecc-scale must be in (0,1]")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.render_row_batch <= 0:
        raise ValueError("--render-row-batch must be positive")
    if args.pose_code_rot_qstep <= 0.0:
        raise ValueError("--pose-code-rot-qstep must be positive")
    if args.pose_code_trans_qstep <= 0.0:
        raise ValueError("--pose-code-trans-qstep must be positive")
    for name in [
        "pose_code_rot_rate_weight",
        "pose_code_trans_rate_weight",
        "pose_code_rot_quant_weight",
        "pose_code_trans_quant_weight",
    ]:
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.depth_scale_precision <= 0:
        raise ValueError("--depth-scale-precision must be positive")
    if not (0.0 < args.depth_scale_percentile <= 100.0):
        raise ValueError("--depth-scale-percentile must be in (0,100]")
    return args


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"src-root not found: {src_root}")
    ensure_dir(dst_root)

    npz_files = find_geometry_npz_files(src_root, args.pattern)
    log(f"Source root : {src_root}")
    log(f"Dest root   : {dst_root}")
    log(f"Pattern     : {args.pattern}")
    log(f"Found NPZ   : {len(npz_files)}")
    if not npz_files:
        return

    success = 0
    skipped = 0
    failed = 0
    for idx, geometry_npz in enumerate(npz_files, start=1):
        rel = geometry_npz.relative_to(src_root)
        log("=" * 72)
        log(f"[{idx}/{len(npz_files)}] {rel}")
        if args.skip_invalid and not validate_canonical_npz(geometry_npz):
            skipped += 1
            continue
        out_prefix = make_out_prefix(geometry_npz, src_root, dst_root, args.layout)
        if already_done(out_prefix) and not args.force:
            log(f"SKIP already done: {canonical_paths_from_prefix(out_prefix)['manifest']}")
            skipped += 1
            continue
        if args.dry_run:
            log(f"DRY RUN output prefix: {out_prefix}")
            skipped += 1
            continue
        try:
            ret = run_one(geometry_npz, src_root, out_prefix, args)
        except Exception as exc:
            failed += 1
            log(f"FAILED: {exc}")
            if not args.continue_on_error:
                raise
            continue
        if ret == 0:
            success += 1
            log("OK")
        else:
            failed += 1
            log(f"FAILED returncode={ret}")
            if not args.continue_on_error:
                raise RuntimeError(f"Failed on {geometry_npz}")
    log("=" * 72)
    log(f"Done. success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
