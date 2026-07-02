#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# NPZ
# ============================================================

def npz_scalar_to_str(x):
    arr = np.asarray(x)
    v = arr.item() if arr.shape == () else arr.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def load_vggt_npz(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)

    for k in ["depth_original", "extrinsic", "intrinsic_original"]:
        if k not in z:
            raise KeyError(f"{npz_path}: missing key '{k}'")

    depth = z["depth_original"].astype(np.float32)
    extrinsic = z["extrinsic"].astype(np.float32)
    intrinsic = z["intrinsic_original"].astype(np.float32)

    # Optional batch/channel cleanup
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]

    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]

    if depth.ndim != 3:
        raise ValueError(f"{npz_path}: depth_original must be SxHxW. Got {depth.shape}")

    if extrinsic.ndim != 3:
        raise ValueError(f"{npz_path}: extrinsic must be Sx3x4 or Sx4x4. Got {extrinsic.shape}")

    if intrinsic.ndim != 3 or intrinsic.shape[1:] != (3, 3):
        raise ValueError(f"{npz_path}: intrinsic_original must be Sx3x3. Got {intrinsic.shape}")

    n = depth.shape[0]

    if extrinsic.shape[0] != n or intrinsic.shape[0] != n:
        raise ValueError(
            f"{npz_path}: frame count mismatch: "
            f"depth={depth.shape[0]}, extrinsic={extrinsic.shape[0]}, intrinsic={intrinsic.shape[0]}"
        )

    if "frame_indices" in z:
        frame_indices = z["frame_indices"].astype(np.int64).tolist()
    else:
        frame_indices = list(range(n))

    rap_name = None
    if "rap_name" in z:
        rap_name = npz_scalar_to_str(z["rap_name"])

    rap_index = None
    if "rap_index" in z:
        rap_index = int(np.asarray(z["rap_index"]).item())

    return {
        "depth": depth,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "frame_indices": [int(x) for x in frame_indices],
        "rap_name": rap_name,
        "rap_index": rap_index,
    }


# ============================================================
# Naming
# ============================================================

def infer_sequence_and_rap(npz_path: Path, rap_name_from_npz=None, rap_index_from_npz=None):
    stem = npz_path.stem

    for suffix in [
        "_vggt_omega_outputs",
        "_outputs",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

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
# Fixed-point depth scale
# ============================================================

def choose_fixed_point_depth_scale(
    depth: np.ndarray,
    percentile: float,
    precision: int,
    max_code: int = 1023,
):
    """
    Convention:

      depth_scale_int = round(depth_scale_float * precision)
      depth_scale_real = depth_scale_int / precision

      depth_y = round(depth_original / depth_scale_real)
      depth_decoded = depth_y * depth_scale_real

    Header stores:
      depth_scale = depth_scale_int
      depth_scale_precision = precision
    """
    valid = np.isfinite(depth) & (depth > 0)

    if not np.any(valid):
        scale_float = 1.0
        scale_int = int(round(scale_float * precision))
        scale_int = max(1, scale_int)
        scale_real = scale_int / float(precision)

        return {
            "depth_scale": int(scale_int),
            "depth_scale_precision": int(precision),
            "depth_scale_real": float(scale_real),
            "depth_scale_float_before_fixed_point": float(scale_float),
            "depth_ref": float(max_code),
            "depth_percentile": float(percentile),
            "max_code": int(max_code),
            "scale_policy": "fallback_no_valid_depth",
            "encode_formula": "depth_y = round(depth_original / (depth_scale / depth_scale_precision))",
            "decode_formula": "depth = depth_y * depth_scale / depth_scale_precision",
        }

    vals = depth[valid].astype(np.float64)
    depth_ref = float(np.percentile(vals, percentile))

    if not np.isfinite(depth_ref) or depth_ref <= 0:
        depth_ref = float(np.max(vals))

    if not np.isfinite(depth_ref) or depth_ref <= 0:
        depth_ref = float(max_code)

    scale_float = depth_ref / float(max_code)

    # User-requested policy: multiply by 1e5 and round.
    scale_int = int(round(scale_float * float(precision)))

    # Avoid zero scale.
    scale_int = max(1, scale_int)

    scale_real = scale_int / float(precision)

    return {
        "depth_scale": int(scale_int),
        "depth_scale_precision": int(precision),
        "depth_scale_real": float(scale_real),
        "depth_scale_float_before_fixed_point": float(scale_float),
        "depth_ref": float(depth_ref),
        "depth_percentile": float(percentile),
        "max_code": int(max_code),
        "scale_policy": "fixed_point_round",
        "encode_formula": "depth_y = round(depth_original / (depth_scale / depth_scale_precision))",
        "decode_formula": "depth = depth_y * depth_scale / depth_scale_precision",
    }


def quantize_depth_with_fixed_point_scale(depth: np.ndarray, depth_meta: dict):
    scale_real = float(depth_meta["depth_scale_real"])
    max_code = int(depth_meta["max_code"])

    if scale_real <= 0:
        raise ValueError("depth_scale_real must be positive")

    y = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=max_code * scale_real,
        neginf=0.0,
    )

    y = np.round(y / scale_real)
    y = np.clip(y, 0, max_code).astype(np.dtype("<u2"))

    return y


def compute_depth_quant_stats(depth: np.ndarray, yq: np.ndarray, depth_meta: dict):
    scale_real = float(depth_meta["depth_scale_real"])
    max_code = int(depth_meta["max_code"])

    valid = np.isfinite(depth) & (depth > 0)

    if not np.any(valid):
        return {
            "valid_count": 0,
            "clip_ratio": 0.0,
            "mae": None,
            "rmse": None,
            "max_abs_err": None,
        }

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


def write_depth_yuv420p10le(out_path: Path, depth: np.ndarray, depth_meta: dict):
    n, h, w = depth.shape

    if w % 2 or h % 2:
        raise ValueError(f"YUV420 requires even resolution. Got {w}x{h}")

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

    summary = {
        "frame_stats": all_stats,
        "mean_mae": float(np.mean(valid_maes)) if valid_maes else None,
        "mean_rmse": float(np.mean(valid_rmses)) if valid_rmses else None,
        "max_clip_ratio": float(np.max(clip_ratios)) if clip_ratios else 0.0,
    }

    return summary


# ============================================================
# Pose conversion
# ============================================================

def extrinsic_to_4x4(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)

    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = E[:3, :3]
        T[:3, 3] = E[:3, 3]
        return T

    if E.shape == (4, 4):
        return E.copy()

    raise ValueError(f"Unsupported extrinsic shape: {E.shape}")


def rt_cur_to_prev_from_extrinsics(extrinsics: np.ndarray):
    """
    VGGT extrinsic:
      W2C = camera_from_world

    Output:
      poc 0: identity
      poc i: X_prev = R * X_cur + t

    T_cur_to_prev = W2C_prev @ C2W_cur
    """
    n = extrinsics.shape[0]
    W2Cs = [extrinsic_to_4x4(extrinsics[i]) for i in range(n)]

    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)

    for i in range(1, n):
        W2C_prev = W2Cs[i - 1]
        C2W_cur = np.linalg.inv(W2Cs[i])

        T = W2C_prev @ C2W_cur

        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3].astype(np.float64)

        rvec, _ = cv2.Rodrigues(R)

        rvecs[i] = rvec.reshape(3).astype(np.float32)
        tvecs[i] = t.reshape(3).astype(np.float32)

    rvecs[0] = 0.0
    tvecs[0] = 0.0

    return rvecs, tvecs


def rt_gop_local_from_extrinsics(extrinsics: np.ndarray):
    """
    Alternative:
      poc i is relative to GOP first frame.

    poc 0:
      identity

    T_rel_i = W2C_i @ C2W_0
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

        rvec, _ = cv2.Rodrigues(R)

        rvecs[i] = rvec.reshape(3).astype(np.float32)
        tvecs[i] = t.reshape(3).astype(np.float32)

    rvecs[0] = 0.0
    tvecs[0] = 0.0

    return rvecs, tvecs


# ============================================================
# JSONL
# ============================================================

def intrinsic_to_header_dict(K: np.ndarray, z_sign: float):
    K = np.asarray(K, dtype=np.float64)

    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "z_sign": float(z_sign),
    }


def write_camparam_jsonl(
    out_path: Path,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrinsic: np.ndarray,
    depth_meta: dict,
    quant_summary: dict,
    z_sign: float,
    pose_mode: str,
    width: int,
    height: int,
    source_npz: Path,
    depth_yuv_path: Path,
):
    header = {
        "type": "header",

        # Fixed-point integer scale.
        # Decoder:
        #   depth = depth_y * depth_scale / depth_scale_precision
        "depth_scale": int(depth_meta["depth_scale"]),
        "depth_scale_precision": int(depth_meta["depth_scale_precision"]),

        # Debug/readability only. Encoder/decoder does not need this if it uses precision.
        "depth_scale_real": float(depth_meta["depth_scale_real"]),

        "width": int(width),
        "height": int(height),
        "bit_depth": 10,
        "depth_yuv": str(depth_yuv_path.name),

        "intrinsic": intrinsic_to_header_dict(intrinsic[0], z_sign=z_sign),

        "camera_param": "rvec_tvec_6d",
        "pose_mode": pose_mode,

        "depth_quant": {
            **depth_meta,
            "quant_summary": {
                "mean_mae": quant_summary["mean_mae"],
                "mean_rmse": quant_summary["mean_rmse"],
                "max_clip_ratio": quant_summary["max_clip_ratio"],
            },
        },

        "frame_line_format": {
            "type": "omitted",
            "fields": ["poc", "rvec", "tvec"],
        },

        "source_npz": str(source_npz.resolve()),
    }

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(header, ensure_ascii=False) + "\n")

        for poc in range(rvecs.shape[0]):
            rec = {
                "poc": int(poc),
                "rvec": [float(x) for x in rvecs[poc]],
                "tvec": [float(x) for x in tvecs[poc]],
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Processing
# ============================================================

def process_one_npz(npz_path: Path, args):
    data = load_vggt_npz(npz_path)

    depth = data["depth"]
    extrinsics = data["extrinsic"]
    intrinsic = data["intrinsic"]

    n, h, w = depth.shape

    sequence_name, rap_name = infer_sequence_and_rap(
        npz_path,
        rap_name_from_npz=data["rap_name"],
        rap_index_from_npz=data["rap_index"],
    )

    out_cam = npz_path.parent / f"{sequence_name}_camParam_{rap_name}.jsonl"
    out_depth = npz_path.parent / f"{sequence_name}_depth_{rap_name}.yuv"

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

    quant_summary = write_depth_yuv420p10le(
        out_path=out_depth,
        depth=depth,
        depth_meta=depth_meta,
    )

    if args.pose_mode == "current_to_previous":
        rvecs, tvecs = rt_cur_to_prev_from_extrinsics(extrinsics)
    elif args.pose_mode == "gop_local":
        rvecs, tvecs = rt_gop_local_from_extrinsics(extrinsics)
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
    )

    valid = np.isfinite(depth) & (depth > 0)
    dmin = float(np.min(depth[valid])) if np.any(valid) else 0.0
    dmax = float(np.max(depth[valid])) if np.any(valid) else 0.0

    print(f"[OK] {npz_path}")
    print(f"     raw depth range     : {dmin:.6g} ~ {dmax:.6g}")
    print(f"     depth_ref           : {depth_meta['depth_ref']:.6g} @ p{args.depth_percentile}")
    print(f"     scale_float         : {depth_meta['depth_scale_float_before_fixed_point']:.9g}")
    print(f"     depth_scale int     : {depth_meta['depth_scale']}")
    print(f"     precision           : {depth_meta['depth_scale_precision']}")
    print(f"     depth_scale real    : {depth_meta['depth_scale_real']:.9g}")
    print(f"     quant mean MAE      : {quant_summary['mean_mae']}")
    print(f"     quant mean RMSE     : {quant_summary['mean_rmse']}")
    print(f"     max clip ratio      : {quant_summary['max_clip_ratio']:.6g}")
    print(f"     camParam            : {out_cam}")
    print(f"     depth yuv           : {out_depth}")

    return True


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
        help="Root folder. All NPZ files under this folder are recursively converted.",
    )

    ap.add_argument(
        "--pattern",
        default="*_vggt_omega_outputs.npz",
        help="Recursive NPZ glob pattern.",
    )

    ap.add_argument(
        "--depth-percentile",
        type=float,
        default=99.9,
        help="Depth percentile mapped close to 1023. Default: 99.9",
    )

    ap.add_argument(
        "--depth-scale-precision",
        type=int,
        default=100000,
        help="Fixed-point precision for depth_scale. Default: 100000.",
    )

    ap.add_argument(
        "--z-sign",
        type=float,
        default=1.0,
        help="Use 1.0 for VGGT/OpenCV +Z forward.",
    )

    ap.add_argument(
        "--pose-mode",
        choices=["current_to_previous", "gop_local"],
        default="current_to_previous",
    )

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
    print(f"pose_mode             : {args.pose_mode}")
    print(f"depth_percentile      : {args.depth_percentile}")
    print(f"depth_scale_precision : {args.depth_scale_precision}")

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
