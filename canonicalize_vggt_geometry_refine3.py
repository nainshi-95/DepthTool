#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_depth_satd_refine_canonical_outputs_fast.py

Fast batch depth-only SATD refinement for canonical fixedK GOP NN-depth outputs.

Input per sequence, recursively found under --src-root:
  <base>_fixedK_gop_nn_geometry.npz
  <base>_fixedK_gop_nn_cam.jsonl                         optional metadata source
  <base>_fixedK_gop_nn_depth_linear_yuv420p10le.yuv       optional metadata/source
  <base>_fixedK_gop_nn_manifest.json                      optional metadata source

Output per sequence under --dst-root, with the SAME canonical filename format:
  <base>_fixedK_gop_nn_geometry.npz
  <base>_fixedK_gop_nn_cam.jsonl
  <base>_fixedK_gop_nn_depth_linear_yuv420p10le.yuv
  <base>_fixedK_gop_nn_manifest.json

What this fast version changes compared with the first depth-SATD script:
  1) Uses a single GT/original --yuv. There is no seq-yuv/target-yuv split.
  2) Trains at --train-scale resolution, not full 4K.
     K is scaled consistently: fx,cx by scale_x and fy,cy by scale_y.
  3) Caches all required Y frames and target depth tensors on GPU once.
  4) Computes larger block SATD, e.g. --satd-block-size 16 or 32, fully parallel on GPU.
  5) Optimizes a low-resolution inverse-depth offset grid and writes full-resolution
     canonical depth outputs by upsampling the learned delta to the original resolution.

Pose convention:
  Absolute pose is camera_from_world / W2C:
      X_cam_i = R_i X_world + t_i
  Relative target camera -> reference camera:
      R_rel = R_ref @ R_target.T
      t_rel = t_ref - R_rel @ t_target
      X_ref = R_rel X_target + t_rel

Pair convention:
  --pairs target:ref[:weight]
  Example: 16:0 means ref 0 -> target 16 projection using target depth D_16.
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

import numpy as np
import torch
import torch.nn.functional as F


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


def load_manifest(path: str | Path | None) -> Optional[dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
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
        "r": ["rvec_abs_final", "rvec_abs_refined", "rvec_abs_stage4_smooth", "rvec_abs_stage3_joint", "rvec_abs_stage2_t_nn", "rvec_abs_stage1_rt"],
        "t": ["tvec_abs_final", "tvec_abs_refined", "tvec_abs_stage4_smooth", "tvec_abs_stage3_joint", "tvec_abs_stage2_t_nn", "tvec_abs_stage1_rt"],
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
    max_code = int(scale_meta.get("max_code", 1023))
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


def choose_depth_scale_fixed_point(depth: np.ndarray, percentile: float, precision: int, bit_depth: int = 10) -> dict[str, Any]:
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


def extract_depth_scale_meta(
    stage_depth: np.ndarray,
    copied_header: Optional[dict[str, Any]],
    source_manifest: Optional[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.recompute_depth_scale:
        candidates: list[Any] = []
        if copied_header is not None:
            candidates.append(copied_header.get("depth_output"))
            candidates.append(copied_header.get("depth_yuv"))
        if source_manifest is not None:
            candidates.append(source_manifest.get("depth_yuv"))
            candidates.append(source_manifest.get("depth_output"))
        for c in candidates:
            if isinstance(c, dict) and "depth_scale_real" in c:
                meta = dict(c)
                meta.setdefault("depth_scale_precision", int(args.depth_scale_precision))
                meta.setdefault("depth_scale", int(round(float(meta["depth_scale_real"]) * int(meta["depth_scale_precision"]))))
                meta.setdefault("depth_bit_depth", 10)
                meta.setdefault("max_code", 1023)
                meta["depth_scale_real"] = float(meta["depth_scale_real"])
                return meta
            if isinstance(c, dict) and "depth_scale" in c and "depth_scale_precision" in c:
                meta = dict(c)
                meta["depth_scale"] = int(meta["depth_scale"])
                meta["depth_scale_precision"] = int(meta["depth_scale_precision"])
                meta["depth_scale_real"] = float(meta["depth_scale"]) / float(meta["depth_scale_precision"])
                meta.setdefault("depth_bit_depth", 10)
                meta.setdefault("max_code", 1023)
                return meta
    return choose_depth_scale_fixed_point(stage_depth, args.depth_scale_percentile, args.depth_scale_precision, 10)


# ============================================================
# Canonical NPZ loading
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
    r_key = npz_key(data, ["rvec_abs_final", "rvec_abs_refined", "rvec_abs_stage4_smooth", "rvec_abs_stage3_joint", "rvec_abs_stage2_t_nn", "rvec_abs_stage1_rt"])
    t_key = npz_key(data, ["tvec_abs_final", "tvec_abs_refined", "tvec_abs_stage4_smooth", "tvec_abs_stage3_joint", "tvec_abs_stage2_t_nn", "tvec_abs_stage1_rt"])
    d_key = npz_key(data, ["depth_canonical", "depth_original"])

    K = np.asarray(data[k_key], dtype=np.float64).reshape(3, 3)
    rvecs = np.asarray(data[r_key], dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(data[t_key], dtype=np.float64).reshape(-1, 3)
    depth = np.asarray(data[d_key], dtype=np.float32)

    if depth.ndim != 3:
        raise ValueError(f"depth must be [N,H,W], got {depth.shape}")
    n, h, w = depth.shape
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
                            pairs.append((int(p["target"]), int(p["ref"]), float(p.get("weight", 1.0)), str(p.get("kind", "npz"))))
                        elif "tar" in p and "ref" in p:
                            pairs.append((int(p["tar"]), int(p["ref"]), float(p.get("weight", 1.0)), str(p.get("kind", "npz"))))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        pairs.append((int(p[0]), int(p[1]), float(p[2]) if len(p) >= 3 else 1.0, "npz"))
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
            raise ValueError(f"Pair out of range for N={n}: target={t}, ref={r}")
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
# Geometry helpers
# ============================================================

def rodrigues_np(rvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + K
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def matrix_to_rodrigues_np(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    denom = 2.0 * math.sin(theta)
    if abs(denom) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return axis / denom * theta


def all_rotation_matrices_np(rvecs: np.ndarray) -> np.ndarray:
    return np.stack([rodrigues_np(r) for r in np.asarray(rvecs)], axis=0)


def torch_rodrigues(rvecs: torch.Tensor) -> torch.Tensor:
    dtype = rvecs.dtype
    device = rvecs.device
    n = rvecs.shape[0]
    x, y, z = rvecs[:, 0], rvecs[:, 1], rvecs[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    theta2 = torch.sum(rvecs * rvecs, dim=-1)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-30))
    small = theta2 < 1e-12
    A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta)
    B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2)
    I = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
    return I + A[:, None, None] * K + B[:, None, None] * (K @ K)


# ============================================================
# YUV resolve and frame indices
# ============================================================

def yuv_frame_index_for_poc(poc: int, frame_indices: np.ndarray, args: argparse.Namespace) -> int:
    if args.frame_index_mode == "frame_indices":
        return int(args.seq_start) + int(frame_indices[int(poc)])
    return int(args.seq_start) + int(poc)


def resolve_yuv_for_npz(geometry_npz: Path, src_root: Path, args: argparse.Namespace) -> Optional[Path]:
    base = derive_base_from_geometry_npz(geometry_npz)
    if args.yuv:
        p = Path(args.yuv).expanduser().resolve()
        return p if p.is_file() else None
    rel_key = str(geometry_npz.relative_to(src_root)) if src_root in geometry_npz.parents or geometry_npz == src_root else str(geometry_npz)
    if args.yuv_map_json:
        with open(args.yuv_map_json, "r", encoding="utf-8") as f:
            mp = json.load(f)
        for k in [str(geometry_npz), rel_key, base, geometry_npz.stem]:
            if k in mp:
                p = Path(mp[k]).expanduser().resolve()
                return p if p.is_file() else None
    if args.yuv_root:
        root = Path(args.yuv_root).expanduser().resolve()
        candidates: list[Path] = []
        try:
            rel_dir = geometry_npz.parent.relative_to(src_root)
            candidates.extend([root / rel_dir / f"{base}.yuv", root / rel_dir / f"{base}{args.yuv_suffix}"])
        except Exception:
            pass
        candidates.extend([root / f"{base}.yuv", root / f"{base}{args.yuv_suffix}"])
        for c in candidates:
            if c.is_file():
                return c.resolve()
        hits = sorted(root.rglob(f"*{base}*.yuv"))
        if hits:
            return hits[0].resolve()
    return None


# ============================================================
# Fast GPU training cache
# ============================================================

class FastGpuCache:
    def __init__(
        self,
        yuv_path: Path,
        width: int,
        height: int,
        bitdepth: int,
        train_w: int,
        train_h: int,
        depth_np: np.ndarray,
        used_yuv_indices: set[int],
        target_pocs: list[int],
        device: torch.device,
        dtype: torch.dtype,
        downsample_mode: str,
    ):
        self.yuv_path = Path(yuv_path)
        self.width = int(width)
        self.height = int(height)
        self.bitdepth = int(bitdepth)
        self.train_w = int(train_w)
        self.train_h = int(train_h)
        self.device = device
        self.dtype = dtype
        self.downsample_mode = str(downsample_mode)
        self.y_train: dict[int, torch.Tensor] = {}
        self.depth_train: dict[int, torch.Tensor] = {}

        maxv = float((1 << bitdepth) - 1)
        log(f"Caching {len(used_yuv_indices)} Y frames on GPU at {train_w}x{train_h}...")
        for idx in sorted(used_yuv_indices):
            y = read_y_frame(yuv_path, width, height, bitdepth, idx).astype(np.float32) / maxv
            t = torch.from_numpy(y)[None, None].to(device=device, dtype=dtype)
            t = resize_tensor_2d(t, train_h, train_w, downsample_mode)[0, 0].contiguous()
            self.y_train[int(idx)] = t

        log(f"Caching {len(target_pocs)} target depth maps on GPU at {train_w}x{train_h}...")
        for poc in sorted(target_pocs):
            d = torch.from_numpy(depth_np[int(poc)].astype(np.float32))[None, None].to(device=device, dtype=dtype)
            d = resize_tensor_2d(d, train_h, train_w, downsample_mode)[0, 0].contiguous()
            self.depth_train[int(poc)] = d


def resize_tensor_2d(x4: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    if x4.shape[-2] == h and x4.shape[-1] == w:
        return x4
    if mode == "area":
        return F.interpolate(x4, size=(h, w), mode="area")
    if mode == "bilinear":
        return F.interpolate(x4, size=(h, w), mode="bilinear", align_corners=False)
    raise ValueError(f"Unsupported resize mode: {mode}")


def scaled_K(K: np.ndarray, full_w: int, full_h: int, train_w: int, train_h: int) -> np.ndarray:
    sx = float(train_w) / float(full_w)
    sy = float(train_h) / float(full_h)
    Kt = np.asarray(K, dtype=np.float64).copy()
    Kt[0, 0] *= sx
    Kt[1, 1] *= sy
    Kt[0, 2] *= sx
    Kt[1, 2] *= sy
    return Kt


def choose_train_size(width: int, height: int, train_scale: float, satd_block_size: int) -> tuple[int, int]:
    tw = max(satd_block_size, int(round(width * float(train_scale))))
    th = max(satd_block_size, int(round(height * float(train_scale))))
    # Keep dimensions multiples of the SATD block. This avoids uneven block tails and is usually faster.
    tw = max(satd_block_size, (tw // satd_block_size) * satd_block_size)
    th = max(satd_block_size, (th // satd_block_size) * satd_block_size)
    if tw % 2:
        tw -= 1
    if th % 2:
        th -= 1
    return int(tw), int(th)


# ============================================================
# Projection and warp
# ============================================================

def make_xy_norm_precompute(width: int, height: int, K: np.ndarray, z_sign: float, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    return {
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "z_sign": float(z_sign),
        "x_norm": (xx - cx) / fx,
        "y_norm": (yy - cy) / fy,
    }


def precompute_relative_pose_torch(rvecs: np.ndarray, tvecs: np.ndarray, pairs: list[tuple[int, int, float, str]], device: torch.device, dtype: torch.dtype) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    r = torch.tensor(rvecs, device=device, dtype=dtype)
    t = torch.tensor(tvecs, device=device, dtype=dtype)
    R = torch_rodrigues(r)
    out: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for target, ref, _, _ in pairs:
        target = int(target)
        ref = int(ref)
        R_rel = R[ref] @ R[target].transpose(0, 1)
        t_rel = t[ref] - R_rel @ t[target]
        out[(target, ref)] = {"R_rel": R_rel.contiguous(), "t_rel": t_rel.contiguous()}
    return out


def backward_map_torch(depth_linear: torch.Tensor, precomp: dict[str, Any], rel: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm = precomp["x_norm"]
    y_norm = precomp["y_norm"]
    fx = float(precomp["fx"])
    fy = float(precomp["fy"])
    cx = float(precomp["cx"])
    cy = float(precomp["cy"])
    z_sign = float(precomp["z_sign"])
    width = int(precomp["width"])
    height = int(precomp["height"])
    R = rel["R_rel"]
    t = rel["t_rel"]
    z = depth_linear

    kx = R[0, 0] * x_norm + R[0, 1] * y_norm + R[0, 2] * z_sign
    ky = R[1, 0] * x_norm + R[1, 1] * y_norm + R[1, 2] * z_sign
    kz = R[2, 0] * x_norm + R[2, 1] * y_norm + R[2, 2] * z_sign

    Xp = z * kx + t[0]
    Yp = z * ky + t[1]
    Zp = z * kz + t[2]
    denom = torch.clamp(torch.abs(Zp), min=1e-8)
    map_x = fx * (Xp / denom) + cx
    map_y = fy * (Yp / denom) + cy

    valid = (
        torch.isfinite(map_x)
        & torch.isfinite(map_y)
        & torch.isfinite(z)
        & (Zp * z_sign > 0.0)
        & (z > 0.0)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    return map_x, map_y, valid


def warp_y_torch(ref_y_norm: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor, valid: torch.Tensor, target_y_norm: Optional[torch.Tensor], invalid_fill: str) -> torch.Tensor:
    h, w = ref_y_norm.shape
    gx = 2.0 * map_x / max(w - 1, 1) - 1.0
    gy = 2.0 * map_y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    src = ref_y_norm[None, None]
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    if invalid_fill == "zero":
        out = torch.where(valid, out, torch.zeros_like(out))
    elif invalid_fill == "target":
        if target_y_norm is None:
            raise ValueError("target_y_norm is required for invalid_fill=target")
        out = torch.where(valid, out, target_y_norm)
    else:
        raise ValueError(f"Unsupported invalid_fill: {invalid_fill}")
    return out


# ============================================================
# Parallel block SATD and regularizers
# ============================================================

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("SATD block size must be power of two")
    key = (int(n), str(device), dtype)
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    _HADAMARD_CACHE[key] = H.contiguous()
    return _HADAMARD_CACHE[key]


def block_satd_loss_parallel(residual: torch.Tensor, valid: Optional[torch.Tensor], block_size: int, reduction: str = "mean", normalize: str = "sqrt") -> tuple[torch.Tensor, int]:
    """
    Fully parallel GPU SATD over non-overlapping BxB blocks.

    residual: [H,W]
    valid: optional bool [H,W]. If provided, only all-valid blocks are used.
    block_size: 4/8/16/32/...
    """
    B = int(block_size)
    h, w = residual.shape
    hB = (h // B) * B
    wB = (w // B) * B
    residual = residual[:hB, :wB].contiguous()

    block_ok = None
    if valid is not None:
        valid = valid[:hB, :wB].contiguous()
        vb = valid.view(hB // B, B, wB // B, B).permute(0, 2, 1, 3)
        block_ok = torch.all(vb, dim=(2, 3))

    rb = residual.view(hB // B, B, wB // B, B).permute(0, 2, 1, 3).contiguous()  # [Bh,Bw,B,B]
    Hm = hadamard_matrix(B, residual.device, residual.dtype)
    tmp = torch.matmul(Hm, rb)
    coeff = torch.matmul(tmp, Hm.t())
    satd = coeff.abs().sum(dim=(2, 3))

    if normalize == "sqrt":
        satd = satd / math.sqrt(float(B))
    elif normalize == "n":
        satd = satd / float(B)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unsupported --satd-normalize: {normalize}")

    if block_ok is not None:
        satd = satd[block_ok]

    if satd.numel() == 0:
        return residual.new_tensor(0.0), 0
    if reduction == "sum":
        return torch.sum(satd), int(satd.numel())
    return torch.mean(satd), int(satd.numel())


def tv_l1_2d(x: torch.Tensor) -> torch.Tensor:
    loss = x.new_tensor(0.0)
    if x.shape[-1] > 1:
        loss = loss + torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
    if x.shape[-2] > 1:
        loss = loss + torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :]))
    return loss


def temp_l1(x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.new_tensor(0.0)
    diff = torch.abs(x[1:] - x[:-1])
    if active_mask is not None:
        m = (active_mask[1:] & active_mask[:-1]).float().view(-1, 1, 1, 1)
        if torch.sum(m) <= 0:
            return x.new_tensor(0.0)
        return torch.sum(diff * m) / torch.clamp(torch.sum(m) * diff.shape[1] * diff.shape[2] * diff.shape[3], min=1.0)
    return torch.mean(diff)


# ============================================================
# Low-res inverse-depth offset model
# ============================================================

class LowResDepthOffsetModel(torch.nn.Module):
    def __init__(self, n_frames: int, train_h: int, train_w: int, offset_stride: int, max_delta_rho: float, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.n_frames = int(n_frames)
        self.train_h = int(train_h)
        self.train_w = int(train_w)
        self.offset_stride = int(offset_stride)
        self.max_delta_rho = float(max_delta_rho)
        h_lr = max(1, int(math.ceil(train_h / offset_stride)))
        w_lr = max(1, int(math.ceil(train_w / offset_stride)))
        self.raw = torch.nn.Parameter(torch.zeros(self.n_frames, 1, h_lr, w_lr, device=device, dtype=dtype))

    def delta_lr_all(self) -> torch.Tensor:
        return torch.tanh(self.raw) * self.max_delta_rho

    def delta_train(self, frame_idx: int) -> torch.Tensor:
        x = self.delta_lr_all()[int(frame_idx):int(frame_idx) + 1]
        y = F.interpolate(x, size=(self.train_h, self.train_w), mode="bilinear", align_corners=False)
        return y[0, 0]

    def delta_full(self, frame_idx: int, full_h: int, full_w: int) -> torch.Tensor:
        x = self.delta_lr_all()[int(frame_idx):int(frame_idx) + 1]
        y = F.interpolate(x, size=(int(full_h), int(full_w)), mode="bilinear", align_corners=False)
        return y[0, 0]


def refined_depth_train(base_depth_train: torch.Tensor, model: LowResDepthOffsetModel, frame_idx: int, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.clamp(base_depth_train, min=float(args.depth_min))
    rho = 1.0 / d
    delta = model.delta_train(frame_idx)
    rho_refined = rho + delta
    rho_min = 1.0 / max(float(args.depth_max), 1e-8)
    rho_max = 1.0 / max(float(args.depth_min), 1e-8)
    rho_refined = torch.clamp(rho_refined, min=rho_min, max=rho_max)
    return 1.0 / rho_refined, delta


# ============================================================
# Training / evaluation
# ============================================================

def calc_psnr_np(a: np.ndarray, b: np.ndarray, bitdepth: int, mask: Optional[np.ndarray] = None) -> Optional[float]:
    d = a.astype(np.float64) - b.astype(np.float64)
    if mask is not None:
        m = mask.astype(bool)
        if not np.any(m):
            return None
        d = d[m]
    mse = float(np.mean(d * d))
    if mse <= 0:
        return 999.0
    maxv = float((1 << bitdepth) - 1)
    return 10.0 * math.log10(maxv * maxv / mse)


def compute_pair_loss_fast(
    pair: tuple[int, int, float, str],
    stage: dict[str, Any],
    gpu_cache: FastGpuCache,
    model: LowResDepthOffsetModel,
    xy_precomp: dict[str, Any],
    rel_pose: dict[tuple[int, int], dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target, ref, pair_weight, kind = pair
    tar_yuv_idx = yuv_frame_index_for_poc(target, stage["frame_indices"], args)
    ref_yuv_idx = yuv_frame_index_for_poc(ref, stage["frame_indices"], args)

    ref_y = gpu_cache.y_train[int(ref_yuv_idx)]
    tar_y = gpu_cache.y_train[int(tar_yuv_idx)]
    base_depth = gpu_cache.depth_train[int(target)]

    depth_refined, delta = refined_depth_train(base_depth, model, target, args)
    map_x, map_y, valid = backward_map_torch(depth_refined, xy_precomp, rel_pose[(target, ref)])
    pred = warp_y_torch(ref_y, map_x, map_y, valid, target_y_norm=tar_y, invalid_fill=args.invalid_fill)
    residual = tar_y - pred

    valid_for_satd = valid if args.satd_valid_only else None
    satd, satd_blocks = block_satd_loss_parallel(
        residual,
        valid_for_satd,
        block_size=args.satd_block_size,
        reduction=args.satd_reduction,
        normalize=args.satd_normalize,
    )

    delta_mag = torch.mean(torch.abs(delta))
    delta_tv = tv_l1_2d(delta[None, None])
    loss = float(pair_weight) * float(args.lambda_satd) * satd
    loss = loss + float(args.lambda_delta_mag) * delta_mag + float(args.lambda_delta_tv) * delta_tv

    with torch.no_grad():
        valid_ratio = float(torch.mean(valid.float()).detach().cpu())

    return loss, {
        "target": int(target),
        "ref": int(ref),
        "kind": str(kind),
        "weight": float(pair_weight),
        "satd": float(satd.detach().cpu()),
        "satd_blocks": int(satd_blocks),
        "delta_mag": float(delta_mag.detach().cpu()),
        "delta_tv": float(delta_tv.detach().cpu()),
        "valid_ratio": valid_ratio,
    }


def compute_total_loss_fast(
    pairs_this_step: list[tuple[int, int, float, str]],
    stage: dict[str, Any],
    gpu_cache: FastGpuCache,
    model: LowResDepthOffsetModel,
    xy_precomp: dict[str, Any],
    rel_pose: dict[tuple[int, int], dict[str, torch.Tensor]],
    active_frame_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    dtype = model.raw.dtype
    device = model.raw.device
    total = torch.zeros((), device=device, dtype=dtype)
    pair_stats = []
    for pair in pairs_this_step:
        loss, st = compute_pair_loss_fast(pair, stage, gpu_cache, model, xy_precomp, rel_pose, args)
        total = total + loss
        pair_stats.append(st)

    extra = {}
    if args.lambda_delta_temp > 0:
        temp = temp_l1(model.delta_lr_all(), active_mask=active_frame_mask)
        total = total + float(args.lambda_delta_temp) * temp
        extra["delta_temp"] = float(temp.detach().cpu())
    else:
        extra["delta_temp"] = 0.0

    if args.lambda_delta_lr_tv > 0:
        lr_tv = tv_l1_2d(model.delta_lr_all())
        total = total + float(args.lambda_delta_lr_tv) * lr_tv
        extra["delta_lr_tv"] = float(lr_tv.detach().cpu())
    else:
        extra["delta_lr_tv"] = 0.0

    return total, pair_stats, extra


@torch.no_grad()
def evaluate_pairs_fast(
    pairs: list[tuple[int, int, float, str]],
    stage: dict[str, Any],
    gpu_cache: FastGpuCache,
    model: LowResDepthOffsetModel,
    xy_precomp: dict[str, Any],
    rel_pose: dict[tuple[int, int], dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out = []
    maxv = float((1 << args.bitdepth) - 1)
    for pair in pairs:
        target, ref, pair_weight, kind = pair
        tar_yuv_idx = yuv_frame_index_for_poc(target, stage["frame_indices"], args)
        ref_yuv_idx = yuv_frame_index_for_poc(ref, stage["frame_indices"], args)
        ref_y = gpu_cache.y_train[int(ref_yuv_idx)]
        tar_y = gpu_cache.y_train[int(tar_yuv_idx)]
        base_depth = torch.clamp(gpu_cache.depth_train[int(target)], min=float(args.depth_min))

        map_x_b, map_y_b, valid_b = backward_map_torch(base_depth, xy_precomp, rel_pose[(target, ref)])
        pred_b = warp_y_torch(ref_y, map_x_b, map_y_b, valid_b, target_y_norm=tar_y, invalid_fill=args.invalid_fill)
        depth_refined, _ = refined_depth_train(base_depth, model, target, args)
        map_x_r, map_y_r, valid_r = backward_map_torch(depth_refined, xy_precomp, rel_pose[(target, ref)])
        pred_r = warp_y_torch(ref_y, map_x_r, map_y_r, valid_r, target_y_norm=tar_y, invalid_fill=args.invalid_fill)

        satd_b, nb = block_satd_loss_parallel(tar_y - pred_b, valid_b if args.satd_valid_only else None, args.satd_block_size, reduction="mean", normalize=args.satd_normalize)
        satd_r, nr = block_satd_loss_parallel(tar_y - pred_r, valid_r if args.satd_valid_only else None, args.satd_block_size, reduction="mean", normalize=args.satd_normalize)

        tar_np = (tar_y.detach().cpu().numpy() * maxv).astype(np.float32)
        pb_np = (pred_b.detach().cpu().numpy() * maxv).astype(np.float32)
        pr_np = (pred_r.detach().cpu().numpy() * maxv).astype(np.float32)
        vb_np = valid_b.detach().cpu().numpy().astype(bool)
        vr_np = valid_r.detach().cpu().numpy().astype(bool)

        out.append({
            "target": int(target),
            "ref": int(ref),
            "kind": str(kind),
            "weight": float(pair_weight),
            "train_size": {"width": int(gpu_cache.train_w), "height": int(gpu_cache.train_h)},
            "satd_block_size": int(args.satd_block_size),
            "base_satd": float(satd_b.detach().cpu()),
            "refined_satd": float(satd_r.detach().cpu()),
            "base_satd_blocks": int(nb),
            "refined_satd_blocks": int(nr),
            "base_psnr_full": json_safe_float(calc_psnr_np(pb_np, tar_np, args.bitdepth)),
            "refined_psnr_full": json_safe_float(calc_psnr_np(pr_np, tar_np, args.bitdepth)),
            "base_psnr_valid": json_safe_float(calc_psnr_np(pb_np, tar_np, args.bitdepth, mask=vb_np)),
            "refined_psnr_valid": json_safe_float(calc_psnr_np(pr_np, tar_np, args.bitdepth, mask=vr_np)),
            "base_valid_ratio": float(np.mean(vb_np)),
            "refined_valid_ratio": float(np.mean(vr_np)),
        })
    return out


def sample_pairs_for_step(pairs: list[tuple[int, int, float, str]], pairs_per_step: int, rng: np.random.Generator) -> list[tuple[int, int, float, str]]:
    if pairs_per_step <= 0 or pairs_per_step >= len(pairs):
        return pairs
    idx = rng.choice(len(pairs), size=int(pairs_per_step), replace=False)
    return [pairs[int(i)] for i in idx]


@torch.no_grad()
def make_refined_depth_np(stage: dict[str, Any], model: LowResDepthOffsetModel, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    depth = stage["depth"].astype(np.float32)
    out = depth.copy()
    target_frames = getattr(args, "_target_frames", range(depth.shape[0]))
    for fi in target_frames:
        fi = int(fi)
        d = torch.from_numpy(depth[fi].astype(np.float32)).to(device=device, dtype=dtype)
        d = torch.clamp(d, min=float(args.depth_min))
        rho = 1.0 / d
        delta_full = model.delta_full(fi, depth.shape[1], depth.shape[2])
        rho_ref = rho + delta_full
        rho_min = 1.0 / max(float(args.depth_max), 1e-8)
        rho_max = 1.0 / max(float(args.depth_min), 1e-8)
        rho_ref = torch.clamp(rho_ref, min=rho_min, max=rho_max)
        out[fi] = (1.0 / rho_ref).detach().cpu().numpy().astype(np.float32)
    return out


def derive_max_delta_rho(depth: np.ndarray, target_frames: list[int], args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.max_delta_rho > 0:
        return float(args.max_delta_rho), {"mode": "absolute", "max_delta_rho": float(args.max_delta_rho)}
    vals = []
    for fi in target_frames:
        d = np.maximum(depth[int(fi)].astype(np.float64), float(args.depth_min))
        vals.append((1.0 / d).reshape(-1))
    rho = np.concatenate(vals, axis=0)
    p5 = float(np.percentile(rho, 5.0))
    p95 = float(np.percentile(rho, 95.0))
    robust_range = max(p95 - p5, 1e-12)
    max_delta = float(args.max_delta_rho_ratio) * robust_range
    return max_delta, {
        "mode": "ratio",
        "rho_p5": p5,
        "rho_p95": p95,
        "rho_robust_range": robust_range,
        "max_delta_rho_ratio": float(args.max_delta_rho_ratio),
        "max_delta_rho": float(max_delta),
    }


# ============================================================
# Output writing
# ============================================================

def write_camera_jsonl_canonical(
    path: Path,
    source_npz: Path,
    source_camera_jsonl: Optional[Path],
    frame_indices: np.ndarray,
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    z_sign: float,
    depth_yuv_meta: dict[str, Any],
    refine_report_summary: dict[str, Any],
) -> None:
    ensure_parent(path)
    R_all = all_rotation_matrices_np(rvecs)
    header = {
        "type": "header",
        "format": "fixedK_gop_nn_depth_satd_fast_refine_v1",
        "source_npz": os.path.abspath(source_npz),
        "source_camera_jsonl": os.path.abspath(source_camera_jsonl) if source_camera_jsonl else None,
        "frame_count": int(len(frame_indices)),
        "frame_indices": frame_indices.astype(int).tolist(),
        "intrinsic_mode": "rap_fixed_depth_satd_refine_camera_unchanged",
        "intrinsic": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "z_sign": float(z_sign),
        },
        "intrinsic_delta_order": [],
        "intrinsic_delta_bits_per_frame": 0,
        "pose_storage": {
            "absolute_pose": "camera_from_world / W2C in fixed-K canonical camera coordinates",
            "relative_pair_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target; X_ref=R_rel*X_target+t_rel",
            "adjacent_current_to_previous_fields": "also written for compatibility",
            "camera_refinement": "unchanged from source canonical input",
        },
        "depth_output": depth_yuv_meta,
        "refinement": refine_report_summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(header), ensure_ascii=False) + "\n")
        for i in range(len(frame_indices)):
            rec: dict[str, Any] = {
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec_abs": rvecs[i].astype(float).tolist(),
                "tvec_abs": tvecs[i].astype(float).tolist(),
                "extrinsic_abs": np.concatenate([R_all[i], tvecs[i].reshape(3, 1)], axis=1).astype(float).tolist(),
            }
            if i == 0:
                rec["rvec_current_to_previous"] = [0.0, 0.0, 0.0]
                rec["tvec_current_to_previous"] = [0.0, 0.0, 0.0]
            else:
                R_rel = R_all[i - 1] @ R_all[i].T
                t_rel = tvecs[i - 1] - R_rel @ tvecs[i]
                rec["rvec_current_to_previous"] = matrix_to_rodrigues_np(R_rel).astype(float).tolist()
                rec["tvec_current_to_previous"] = t_rel.astype(float).tolist()
            f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")


def write_refined_geometry_npz(out_npz: Path, stage: dict[str, Any], refined_depth: np.ndarray, result: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_parent(out_npz)
    data = stage["data"]
    payload: dict[str, Any] = {}
    for k in data.files:
        payload[k] = data[k]
    payload["depth_before_satd_refine"] = stage["depth"].astype(np.float32)
    payload["depth_canonical"] = refined_depth.astype(np.float32)
    if args.update_depth_original:
        payload["depth_original"] = refined_depth.astype(np.float32)
    payload["K_fixed"] = stage["K"].astype(np.float32)
    payload["K_refined"] = stage["K"].astype(np.float32)
    payload["rvec_abs_refined"] = stage["rvecs"].astype(np.float32)
    payload["tvec_abs_refined"] = stage["tvecs"].astype(np.float32)
    payload["rvec_abs_final"] = stage["rvecs"].astype(np.float32)
    payload["tvec_abs_final"] = stage["tvecs"].astype(np.float32)
    payload["depth_satd_fast_refine_result_json"] = np.asarray(json.dumps(to_jsonable(result), ensure_ascii=False), dtype=object)
    if args.compressed_npz:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)


def write_manifest(out_manifest: Path, source_npz: Path, sidecars: dict[str, Optional[Path]], out_paths: dict[str, Path], stage: dict[str, Any], result: dict[str, Any], depth_meta: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_parent(out_manifest)
    manifest = {
        "source_npz": os.path.abspath(source_npz),
        "source_camera_jsonl": os.path.abspath(sidecars["camera_jsonl"]) if sidecars.get("camera_jsonl") else None,
        "source_depth_yuv": os.path.abspath(sidecars["depth_yuv"]) if sidecars.get("depth_yuv") else None,
        "outputs": {k: os.path.abspath(v) for k, v in out_paths.items()},
        "frame_count": int(stage["depth"].shape[0]),
        "size": {"width": int(stage["depth"].shape[2]), "height": int(stage["depth"].shape[1])},
        "K": stage["K"].astype(float).tolist(),
        "camera_status": "unchanged from source canonical input",
        "depth_yuv": depth_meta,
        "depth_satd_fast_refine": result,
        "options": vars(args),
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=2, ensure_ascii=False)
        f.write("\n")


# ============================================================
# Main per-file pipeline
# ============================================================

def run_one(geometry_npz: Path, src_root: Path, out_prefix: Path, args: argparse.Namespace) -> int:
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
    copied_header = load_first_jsonl_object(sidecars.get("camera_jsonl"))
    source_manifest = load_manifest(sidecars.get("manifest"))

    yuv = resolve_yuv_for_npz(geometry_npz, src_root, args)
    if yuv is None:
        raise RuntimeError("GT/original YUV not found. Provide --yuv, --yuv-root, or --yuv-map-json.")

    stage = load_canonical_npz(geometry_npz)
    K = stage["K"]
    rvecs = stage["rvecs"]
    tvecs = stage["tvecs"]
    depth = stage["depth"]
    frame_indices = stage["frame_indices"]
    n, h, w = depth.shape

    if args.width is not None and int(args.width) != w:
        raise ValueError(f"--width {args.width} != NPZ width {w}")
    if args.height is not None and int(args.height) != h:
        raise ValueError(f"--height {args.height} != NPZ height {h}")

    if args.depth_max <= 0.0:
        finite = depth[np.isfinite(depth)]
        args.depth_max = float(np.max(finite)) if finite.size else 1.0
        args.depth_max = max(args.depth_max, args.depth_min * 2.0)

    pairs = build_pair_list(args, stage)
    yuv_count = count_frames_yuv420(yuv, w, h, args.bitdepth)

    checked_pairs = []
    used_yuv_indices: set[int] = set()
    for target, ref, weight, kind in pairs:
        tar_idx = yuv_frame_index_for_poc(target, frame_indices, args)
        ref_idx = yuv_frame_index_for_poc(ref, frame_indices, args)
        if tar_idx < 0 or tar_idx >= yuv_count or ref_idx < 0 or ref_idx >= yuv_count:
            if args.skip_out_of_range_pairs:
                log(f"SKIP pair target={target}, ref={ref}: YUV idx out of range tar={tar_idx}/{yuv_count}, ref={ref_idx}/{yuv_count}")
                continue
            raise RuntimeError(f"Pair target={target}, ref={ref} out of YUV range: tar={tar_idx}/{yuv_count}, ref={ref_idx}/{yuv_count}")
        checked_pairs.append((target, ref, weight, kind))
        used_yuv_indices.add(int(tar_idx))
        used_yuv_indices.add(int(ref_idx))
    pairs = checked_pairs
    if not pairs:
        raise RuntimeError("No valid pairs after YUV range check.")

    target_frames = sorted(set(int(t) for t, _, _, _ in pairs))
    args._target_frames = target_frames

    # Scaled training resolution and scaled K.
    train_w, train_h = choose_train_size(w, h, args.train_scale, args.satd_block_size)
    K_train = scaled_K(K, w, h, train_w, train_h)

    log(f"Loaded canonical NPZ: frames={n}, full={w}x{h}, train={train_w}x{train_h}, pairs={len(pairs)}, target_frames={target_frames}")
    log(f"GT YUV={yuv}")
    log(f"Full K : fx={K[0,0]:.6f}, fy={K[1,1]:.6f}, cx={K[0,2]:.6f}, cy={K[1,2]:.6f}")
    log(f"Train K: fx={K_train[0,0]:.6f}, fy={K_train[1,1]:.6f}, cx={K_train[0,2]:.6f}, cy={K_train[1,2]:.6f}")
    log("Camera K/R|t will be copied unchanged in outputs.")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.torch_float64 else torch.float32
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    log(f"device={device}, dtype={dtype}")

    max_delta_rho, rho_stats = derive_max_delta_rho(depth, target_frames, args)
    log(f"max_delta_rho={max_delta_rho:.8e}, mode={rho_stats.get('mode')}")

    gpu_cache = FastGpuCache(
        yuv_path=yuv,
        width=w,
        height=h,
        bitdepth=args.bitdepth,
        train_w=train_w,
        train_h=train_h,
        depth_np=depth,
        used_yuv_indices=used_yuv_indices,
        target_pocs=target_frames,
        device=device,
        dtype=dtype,
        downsample_mode=args.train_downsample,
    )

    model = LowResDepthOffsetModel(
        n_frames=n,
        train_h=train_h,
        train_w=train_w,
        offset_stride=args.offset_stride,
        max_delta_rho=max_delta_rho,
        device=device,
        dtype=dtype,
    ).to(device)

    active_np = np.zeros(n, dtype=np.bool_)
    for fi in target_frames:
        active_np[int(fi)] = True
    active_mask = torch.from_numpy(active_np).to(device)

    def grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
        m = active_mask.to(dtype=grad.dtype).view(-1, 1, 1, 1)
        return grad * m
    model.raw.register_hook(grad_mask_hook)

    xy_precomp = make_xy_norm_precompute(train_w, train_h, K_train, args.z_sign, device, dtype)
    rel_pose = precompute_relative_pose_torch(rvecs, tvecs, pairs, device, dtype)

    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log("Initial evaluation...")
    initial_eval = evaluate_pairs_fast(pairs, stage, gpu_cache, model, xy_precomp, rel_pose, args)
    for e in initial_eval:
        log(f"INIT pair {e['target']}->{e['ref']}: SATD {e['base_satd']:.6f}->{e['refined_satd']:.6f}, PSNR full {e['base_psnr_full']}->{e['refined_psnr_full']}, valid {e['base_valid_ratio']:.4f}->{e['refined_valid_ratio']:.4f}")

    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize()

    for step in range(int(args.depth_steps)):
        pairs_step = sample_pairs_for_step(pairs, args.pairs_per_step, rng)
        opt.zero_grad(set_to_none=True)
        loss, pair_stats, extra = compute_total_loss_fast(pairs_step, stage, gpu_cache, model, xy_precomp, rel_pose, active_mask, args)
        if not torch.isfinite(loss):
            log(f"Depth step {step:04d}: non-finite loss={float(loss.detach().cpu())}; stop.")
            break
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step % max(1, int(args.log_every)) == 0 or step == int(args.depth_steps) - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            st = {"step": int(step), "loss": float(loss.detach().cpu()), "num_pairs_this_step": int(len(pairs_step)), "pair_stats": pair_stats, **extra}
            history.append(st)
            satd_avg = float(np.mean([x["satd"] for x in pair_stats])) if pair_stats else 0.0
            valid_avg = float(np.mean([x["valid_ratio"] for x in pair_stats])) if pair_stats else 0.0
            dmag_avg = float(np.mean([x["delta_mag"] for x in pair_stats])) if pair_stats else 0.0
            dtv_avg = float(np.mean([x["delta_tv"] for x in pair_stats])) if pair_stats else 0.0
            log(f"Depth step {step:04d}/{args.depth_steps}: loss={st['loss']:.6f}, pairs={len(pairs_step)}, satd={satd_avg:.6f}, valid={valid_avg:.4f}, dmag={dmag_avg:.3e}, dtv={dtv_avg:.3e}, temp={extra['delta_temp']:.3e}")

    log("Final evaluation...")
    final_eval = evaluate_pairs_fast(pairs, stage, gpu_cache, model, xy_precomp, rel_pose, args)
    for e in final_eval:
        log(f"FINAL pair {e['target']}->{e['ref']}: SATD {e['base_satd']:.6f}->{e['refined_satd']:.6f}, PSNR full {e['base_psnr_full']}->{e['refined_psnr_full']}, valid {e['base_valid_ratio']:.4f}->{e['refined_valid_ratio']:.4f}")

    log("Materializing refined full-resolution depth...")
    refined_depth = make_refined_depth_np(stage, model, args, device, dtype)
    depth_scale_meta = extract_depth_scale_meta(refined_depth, copied_header, source_manifest, args)
    depth_meta = write_depth_yuv420p10le_linear(out_paths["depth_yuv"], refined_depth, depth_scale_meta)

    delta_lr = model.delta_lr_all().detach().cpu().numpy().astype(np.float32)
    delta_stats = {
        "offset_stride_train_pixels": int(args.offset_stride),
        "delta_lr_shape": list(delta_lr.shape),
        "max_delta_rho": float(max_delta_rho),
        "delta_lr_min": float(np.min(delta_lr)),
        "delta_lr_max": float(np.max(delta_lr)),
        "delta_lr_mean_abs": float(np.mean(np.abs(delta_lr))),
        "target_frames": [int(x) for x in target_frames],
    }

    result = {
        "method": {
            "description": "Fast depth-only low-resolution inverse-depth SATD refinement. K and W2C R|t are fixed and copied unchanged.",
            "training": "Y, depth, K are downscaled to train resolution; projection/warp/SATD are evaluated at train resolution on GPU.",
            "depth_model": "rho_refined = 1/depth_canonical + bilinear_upsample(max_delta_rho*tanh(raw_delta_lr)); depth_refined=1/rho_refined",
            "loss": "multi-pair parallel block SATD plus L1/TV/temporal regularization",
            "pose_convention": "camera_from_world / W2C: X_cam=R X_world+t",
            "relative_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target",
            "camera": "unchanged",
        },
        "input": {
            "geometry_npz": str(geometry_npz),
            "camera_jsonl": str(sidecars["camera_jsonl"]) if sidecars.get("camera_jsonl") else None,
            "depth_yuv": str(sidecars["depth_yuv"]) if sidecars.get("depth_yuv") else None,
            "yuv": str(yuv),
        },
        "size": {"width": int(w), "height": int(h)},
        "train_size": {"width": int(train_w), "height": int(train_h), "scale_x": float(train_w / w), "scale_y": float(train_h / h)},
        "frame_indices": frame_indices.astype(int).tolist(),
        "stage_source_keys": stage["source_keys"],
        "options": vars(args),
        "K_full": K.astype(float).tolist(),
        "K_train": K_train.astype(float).tolist(),
        "camera_status": "unchanged from source canonical input",
        "pairs": [{"target": int(t), "ref": int(r), "weight": float(wt), "kind": str(k)} for t, r, wt, k in pairs],
        "rho_stats": rho_stats,
        "delta_stats": delta_stats,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history_tail": history[-20:],
    }

    summary = {
        "description": "fast depth SATD low-res inverse-depth refinement; canonical output naming preserved; camera unchanged",
        "pair_source": args.pair_source,
        "num_pairs": len(pairs),
        "target_frames": [int(x) for x in target_frames],
        "train_size": {"width": int(train_w), "height": int(train_h)},
        "satd_block_size": int(args.satd_block_size),
        "K": K.astype(float).tolist(),
        "camera_status": "unchanged",
        "rho_stats": rho_stats,
        "delta_stats": delta_stats,
    }

    write_camera_jsonl_canonical(out_paths["camera_jsonl"], geometry_npz, sidecars.get("camera_jsonl"), frame_indices, K, rvecs, tvecs, args.z_sign, depth_meta, summary)
    result["outputs"] = {k: str(v) for k, v in out_paths.items()}
    result["depth_yuv"] = depth_meta
    write_refined_geometry_npz(out_paths["geometry_npz"], stage, refined_depth, result, args)
    write_manifest(out_paths["manifest"], geometry_npz, sidecars, out_paths, stage, result, depth_meta, args)

    print("============================================================")
    print("Fast depth SATD refine done")
    print("============================================================")
    print(f"input geometry : {geometry_npz}")
    print(f"GT yuv         : {yuv}")
    print(f"frames         : {n}")
    print(f"full size      : {w}x{h}")
    print(f"train size     : {train_w}x{train_h}")
    print(f"camera         : unchanged")
    print(f"satd block     : {args.satd_block_size}")
    print(f"offset stride  : {args.offset_stride} train pixels")
    print(f"target frames  : {target_frames}")
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
    ap = argparse.ArgumentParser(description="Fast batch depth-only SATD refinement for canonical fixedK GOP NN-depth outputs.")

    # Batch I/O.
    ap.add_argument("--src-root", required=True, help="Folder containing canonical outputs")
    ap.add_argument("--dst-root", required=True, help="Output root. Canonical filenames are preserved under this root.")
    ap.add_argument("--pattern", default=f"*{GEOM_SUFFIX}", help=f"Input geometry NPZ pattern. Default: *{GEOM_SUFFIX}")
    ap.add_argument("--layout", choices=["preserve", "flat"], default="preserve")
    ap.add_argument("--force", action="store_true", help="Overwrite dst outputs")
    ap.add_argument("--skip-invalid", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compressed-npz", action="store_true")

    # Single GT/original YUV.
    ap.add_argument("--yuv", default="", help="GT/original YUV for all NPZs. Ref and target are both read from this YUV.")
    ap.add_argument("--yuv-root", default="", help="Root to auto-search GT/original YUV for each NPZ.")
    ap.add_argument("--yuv-map-json", default="", help="JSON mapping full path / relative path / base name to GT/original YUV path.")
    ap.add_argument("--yuv-suffix", default=".yuv")

    ap.add_argument("--bitdepth", type=int, choices=[8, 10], default=10)
    ap.add_argument("--width", type=int, default=None, help="Optional sanity check against NPZ depth width")
    ap.add_argument("--height", type=int, default=None, help="Optional sanity check against NPZ depth height")
    ap.add_argument("--seq-start", type=int, default=0)
    ap.add_argument("--frame-index-mode", choices=["local", "frame_indices"], default="local", help="local: YUV idx=seq_start+poc. frame_indices: YUV idx=seq_start+frame_indices[poc]")

    # Pair selection.
    ap.add_argument("--pairs", default="", help="Pair list target:ref[:weight], e.g. 16:0:1,16:32:1")
    ap.add_argument("--pair-source", choices=["npz", "adjacent", "dyadic", "all"], default="npz")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--no-bidirectional-pairs", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--skip-out-of-range-pairs", action="store_true")
    ap.add_argument("--pairs-per-step", type=int, default=0, help="0 means all pairs every step; otherwise random pair minibatch size.")

    # Fast training resolution.
    ap.add_argument("--train-scale", type=float, default=0.25, help="Training resolution scale. 0.25 makes 3840x2160 -> about 960x540.")
    ap.add_argument("--train-downsample", choices=["area", "bilinear"], default="area")

    # Depth SATD refine.
    ap.add_argument("--device", default="auto")
    ap.add_argument("--torch-float64", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--depth-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--log-every", type=int, default=20)

    ap.add_argument("--offset-stride", type=int, default=16, help="Low-res inverse-depth offset stride in TRAIN pixels. With train-scale 0.25 and stride 16, full-res equivalent is 64 pixels.")
    ap.add_argument("--max-delta-rho", type=float, default=0.0, help="Absolute max inverse-depth offset. If 0, derived from --max-delta-rho-ratio.")
    ap.add_argument("--max-delta-rho-ratio", type=float, default=0.01, help="Max delta rho = ratio * robust rho range of target frames.")

    ap.add_argument("--lambda-satd", type=float, default=1.0)
    ap.add_argument("--lambda-delta-mag", type=float, default=1.0)
    ap.add_argument("--lambda-delta-tv", type=float, default=5.0)
    ap.add_argument("--lambda-delta-lr-tv", type=float, default=0.0)
    ap.add_argument("--lambda-delta-temp", type=float, default=1.0)

    ap.add_argument("--satd-block-size", type=int, default=16, help="GPU-parallel SATD block size at train resolution. Must be power of two: 8/16/32 recommended.")
    ap.add_argument("--satd-valid-only", action=argparse.BooleanOptionalAction, default=True, help="Use only all-valid SATD blocks.")
    ap.add_argument("--satd-reduction", choices=["mean", "sum"], default="mean")
    ap.add_argument("--satd-normalize", choices=["sqrt", "n", "none"], default="sqrt")
    ap.add_argument("--invalid-fill", choices=["zero", "target"], default="zero")

    # Projection/depth options.
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--depth-min", type=float, default=1e-6)
    ap.add_argument("--depth-max", type=float, default=0.0, help="Clamp refined depth max. If 0, derived from source depth max.")

    # Depth YUV scale.
    ap.add_argument("--recompute-depth-scale", action="store_true", help="Ignore source depth_scale metadata and recompute from refined depth.")
    ap.add_argument("--depth-scale-precision", type=int, default=100000)
    ap.add_argument("--depth-scale-percentile", type=float, default=99.9)
    ap.add_argument("--update-depth-original", action="store_true", help="Also overwrite depth_original in output NPZ. Default only updates depth_canonical.")

    args = ap.parse_args()

    if args.offset_stride <= 0:
        raise ValueError("--offset-stride must be positive")
    if args.depth_steps < 0:
        raise ValueError("--depth-steps must be non-negative")
    if args.depth_min <= 0:
        raise ValueError("--depth-min must be positive")
    if args.max_delta_rho < 0:
        raise ValueError("--max-delta-rho must be non-negative")
    if args.max_delta_rho_ratio < 0:
        raise ValueError("--max-delta-rho-ratio must be non-negative")
    if args.depth_scale_precision <= 0:
        raise ValueError("--depth-scale-precision must be positive")
    if args.depth_scale_percentile <= 0 or args.depth_scale_percentile > 100:
        raise ValueError("--depth-scale-percentile must be in (0,100]")
    if args.train_scale <= 0.0 or args.train_scale > 1.0:
        raise ValueError("--train-scale must be in (0,1]")
    if args.satd_block_size <= 0 or (args.satd_block_size & (args.satd_block_size - 1)) != 0:
        raise ValueError("--satd-block-size must be a positive power of two")
    if args.pairs_per_step < 0:
        raise ValueError("--pairs-per-step must be non-negative")
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
