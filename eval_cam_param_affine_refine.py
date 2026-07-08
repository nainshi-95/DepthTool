#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_camparam_depth_pair_warp_psnr.py

Evaluate camera/depth output after camParam conversion.

Inputs:
  - camParam JSONL produced by convert script
  - depth YUV420p10le produced by convert script
  - original source YUV420 sequence

For each target/ref pair inside one GOP:
  1) read target depth from depth YUV
  2) back-project target pixels using target K and target depth
  3) transform target camera coordinates to ref camera coordinates
  4) project into ref image
  5) backward-remap original ref Y into target domain
  6) compare with original target Y on valid projection mask

Supported camParam pose modes:
  - current_to_previous: X_prev = T_i X_cur, poc0 identity
  - gop_local          : X_i = T_i X_0, poc0 identity
  - absolute           : X_i = W2C_i X_world

Coordinate convention:
  target pixel -> reference pixel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def rt_to_4x4(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rvec_to_R(rvec)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def frame_size_yuv420(width: int, height: int, bitdepth: int) -> int:
    bytes_per_sample = 1 if bitdepth <= 8 else 2
    return (width * height + 2 * (width // 2) * (height // 2)) * bytes_per_sample


def read_y_frame_yuv420(
    path: str | Path,
    width: int,
    height: int,
    bitdepth: int,
    frame_idx: int,
) -> np.ndarray:
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    fs = frame_size_yuv420(width, height, bitdepth)
    y_samples = width * height

    with open(path, "rb") as f:
        f.seek(int(frame_idx) * fs)
        y = np.fromfile(f, dtype=dtype, count=y_samples)

    if y.size != y_samples:
        raise RuntimeError(f"Cannot read Y frame idx={frame_idx} from {path}")

    return y.reshape(height, width)


def write_yuv420_y_only(path: str | Path, y: np.ndarray, bitdepth: int) -> None:
    y = np.asarray(y)
    h, w = y.shape

    with open(path, "wb") as f:
        if bitdepth <= 8:
            yy = np.clip(np.rint(y), 0, 255).astype(np.uint8)
            uv = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        else:
            maxv = (1 << bitdepth) - 1
            yy = np.clip(np.rint(y), 0, maxv).astype("<u2")
            uv = np.full((h // 2, w // 2), 1 << (bitdepth - 1), dtype="<u2")

        f.write(yy.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())


def save_gray_png(path: str | Path, y: np.ndarray, bitdepth: int) -> None:
    if bitdepth <= 8:
        out = np.clip(y, 0, 255).astype(np.uint8)
    else:
        out = np.clip(
            y.astype(np.float32) / float(1 << (bitdepth - 8)),
            0,
            255,
        ).astype(np.uint8)
    cv2.imwrite(str(path), out)


def calc_psnr_metrics(
    target_y: np.ndarray,
    pred_y: np.ndarray,
    valid: np.ndarray,
    bitdepth: int,
) -> dict[str, Any]:
    valid = valid.astype(bool)
    valid_count = int(np.count_nonzero(valid))
    total = int(valid.size)

    if valid_count <= 0:
        return {
            "valid_count": 0,
            "total_count": total,
            "valid_ratio": 0.0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "psnr": None,
        }

    diff = target_y.astype(np.float32)[valid] - pred_y.astype(np.float32)[valid]
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    maxv = float((1 << bitdepth) - 1)
    psnr = 999.0 if mse <= 1e-12 else float(10.0 * np.log10((maxv * maxv) / mse))

    return {
        "valid_count": valid_count,
        "total_count": total,
        "valid_ratio": float(valid_count / max(total, 1)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
    }


# ============================================================
# camParam JSONL loading
# ============================================================

def load_camparam_jsonl(path: str | Path, pose_mode_override: str = "auto") -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    header: Optional[dict[str, Any]] = None
    frames: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if isinstance(obj, dict) and obj.get("type") == "header":
                header = obj
            else:
                frames.append(obj)

    if header is None:
        raise RuntimeError(f"No header line in camParam JSONL: {path}")

    if not frames:
        raise RuntimeError(f"No frame records in camParam JSONL: {path}")

    n = int(header.get("frame_count", len(frames)))

    if len(frames) != n:
        raise ValueError(f"Frame count mismatch: header={n}, lines={len(frames)}")

    width = int(header["width"])
    height = int(header["height"])

    base_intr = header.get("intrinsic", None)
    if not isinstance(base_intr, dict):
        raise KeyError("header['intrinsic'] missing or invalid")

    z_sign = float(base_intr.get("z_sign", 1.0))

    pose_mode = str(header.get("pose_mode", "current_to_previous"))
    if pose_mode_override != "auto":
        pose_mode = pose_mode_override

    cur = np.array(
        [
            float(base_intr["fx"]),
            float(base_intr["fy"]),
            float(base_intr["cx"]),
            float(base_intr["cy"]),
        ],
        dtype=np.float64,
    )

    K = np.zeros((n, 3, 3), dtype=np.float64)
    rvecs = np.zeros((n, 3), dtype=np.float64)
    tvecs = np.zeros((n, 3), dtype=np.float64)
    frame_indices = np.zeros((n,), dtype=np.int64)

    fixed_intrinsic = (
        str(header.get("intrinsic_delta_mode", "")).startswith("fixed")
        or int(header.get("intrinsic_delta_bits_per_frame", -1) or -1) == 0
    )

    for i, rec in enumerate(frames):
        frame_indices[i] = int(rec.get("frame_idx", i))
        rvecs[i] = np.asarray(rec.get("rvec", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        tvecs[i] = np.asarray(rec.get("tvec", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

        if i > 0 and not fixed_intrinsic:
            delta = np.asarray(
                rec.get("intrinsic_delta", [0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(4)
            cur = cur + delta

        K[i] = np.array(
            [
                [cur[0], 0.0, cur[2]],
                [0.0, cur[1], cur[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    depth_scale = float(header.get("depth_scale", 0)) / float(header.get("depth_scale_precision", 1))
    if "depth_scale_real" in header:
        depth_scale = float(header["depth_scale_real"])

    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"Invalid depth scale in header: {depth_scale}")

    return {
        "path": str(path),
        "header": header,
        "frames": frames,
        "width": width,
        "height": height,
        "frame_count": n,
        "frame_indices": frame_indices,
        "K": K,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "pose_mode": pose_mode,
        "z_sign": z_sign,
        "depth_scale_real": depth_scale,
        "depth_yuv_name": header.get("depth_yuv"),
        "fixed_intrinsic": bool(fixed_intrinsic),
    }


def resolve_depth_yuv(
    camparam_path: Path,
    cam: dict[str, Any],
    explicit_depth_yuv: str,
) -> Path:
    if explicit_depth_yuv:
        p = Path(explicit_depth_yuv).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    name = cam.get("depth_yuv_name")
    if not name:
        raise RuntimeError("Depth YUV path is not provided and header has no depth_yuv")

    p = Path(str(name))

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(camparam_path.parent / p)
        candidates.append(camparam_path.parent / p.name)

    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(f"Cannot resolve depth_yuv={name} near {camparam_path}")


# ============================================================
# Pose reconstruction
# ============================================================

def reconstruct_w2c_from_records(
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    pose_mode: str,
) -> np.ndarray:
    n = int(rvecs.shape[0])
    W2C = np.zeros((n, 4, 4), dtype=np.float64)
    W2C[0] = np.eye(4, dtype=np.float64)

    if pose_mode == "current_to_previous":
        for i in range(1, n):
            T_cur_to_prev = rt_to_4x4(rvecs[i], tvecs[i])

            # convert script convention:
            #   X_prev = T_cur_to_prev * X_cur
            #
            # with GOP-local W2C:
            #   T_cur_to_prev = W2C[i-1] @ inv(W2C[i])
            # therefore:
            #   W2C[i] = inv(T_cur_to_prev) @ W2C[i-1]
            W2C[i] = np.linalg.inv(T_cur_to_prev) @ W2C[i - 1]

    elif pose_mode == "gop_local":
        for i in range(n):
            W2C[i] = rt_to_4x4(rvecs[i], tvecs[i])

    elif pose_mode == "absolute":
        for i in range(n):
            W2C[i] = rt_to_4x4(rvecs[i], tvecs[i])

    else:
        raise ValueError(f"Unsupported pose_mode: {pose_mode}")

    return W2C


# ============================================================
# Depth loading
# ============================================================

def read_depth_yuv420p10le(
    path: str | Path,
    width: int,
    height: int,
    frame_count: int,
    depth_scale_real: float,
) -> np.ndarray:
    path = Path(path)
    dtype = np.dtype("<u2")

    frame_samples = width * height + 2 * (width // 2) * (height // 2)
    expected_samples = frame_count * frame_samples

    raw = np.fromfile(path, dtype=dtype)

    if raw.size < expected_samples:
        raise RuntimeError(
            f"Depth YUV too small: {path}, samples={raw.size}, expected={expected_samples}"
        )

    depth = np.empty((frame_count, height, width), dtype=np.float32)
    off = 0
    y_samples = width * height

    for i in range(frame_count):
        y = raw[off:off + y_samples].reshape(height, width)
        depth[i] = y.astype(np.float32) * float(depth_scale_real)
        off += frame_samples

    return depth


# ============================================================
# Pair generation
# ============================================================

def parse_pairs(text: str, default_weight: float = 1.0) -> list[tuple[int, int, float, str]]:
    out: list[tuple[int, int, float, str]] = []

    if not text or not text.strip():
        return out

    for tok in re.split(r"[,;\s]+", text.strip()):
        if not tok:
            continue

        tok = tok.replace("->", ":")
        parts = tok.split(":")

        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid pair token: {tok}. Use target:ref[:weight]")

        target = int(parts[0])
        ref = int(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else default_weight
        out.append((target, ref, weight, "cli"))

    return out


def generate_adjacent_pairs(
    n: int,
    bidirectional: bool,
    weight: float,
) -> list[tuple[int, int, float, str]]:
    out = []

    for i in range(1, n):
        out.append((i, i - 1, weight, "adjacent"))
        if bidirectional:
            out.append((i - 1, i, weight, "adjacent_rev"))

    return out


def generate_dyadic_pairs(
    n: int,
    bidirectional: bool,
    weight: float,
) -> list[tuple[int, int, float, str]]:
    acc: dict[tuple[int, int], tuple[int, int, float, str]] = {}

    def add(t: int, r: int, w: float, kind: str) -> None:
        if t == r or not (0 <= t < n and 0 <= r < n):
            return

        key = (int(t), int(r))

        if key in acc:
            old = acc[key]
            acc[key] = (old[0], old[1], old[2] + w, old[3] + "+" + kind)
        else:
            acc[key] = (key[0], key[1], w, kind)

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


def build_pairs(n: int, args: argparse.Namespace) -> list[tuple[int, int, float, str]]:
    if args.pairs.strip():
        pairs = parse_pairs(args.pairs, default_weight=args.pair_weight)

    elif args.pair_source == "adjacent":
        pairs = generate_adjacent_pairs(
            n,
            bidirectional=not args.no_bidirectional_pairs,
            weight=args.pair_weight,
        )

    elif args.pair_source == "dyadic":
        pairs = generate_dyadic_pairs(
            n,
            bidirectional=not args.no_bidirectional_pairs,
            weight=args.pair_weight,
        )

    elif args.pair_source == "all":
        pairs = [
            (t, r, args.pair_weight, "all")
            for t in range(n)
            for r in range(n)
            if t != r
        ]

        if args.no_bidirectional_pairs:
            pairs = [(t, r, w, k) for (t, r, w, k) in pairs if t > r]

    else:
        raise ValueError(args.pair_source)

    checked = []
    seen = set()

    for t, r, w, kind in pairs:
        if not (0 <= t < n and 0 <= r < n):
            raise ValueError(f"Pair out of range for N={n}: {t}->{r}")

        key = (int(t), int(r))

        if key in seen:
            continue

        seen.add(key)
        checked.append((key[0], key[1], float(w), str(kind)))

    if args.max_pairs > 0:
        checked = checked[:int(args.max_pairs)]

    if not checked:
        raise RuntimeError("No pairs selected")

    return checked


# ============================================================
# Warping
# ============================================================

def camera_map_target_to_ref(
    target: int,
    ref: int,
    width: int,
    height: int,
    K_all: np.ndarray,
    W2C: np.ndarray,
    depth_target: np.ndarray,
    z_sign: float,
    z_min: float,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_t = K_all[int(target)]
    K_r = K_all[int(ref)]

    fx_t = float(K_t[0, 0])
    fy_t = float(K_t[1, 1])
    cx_t = float(K_t[0, 2])
    cy_t = float(K_t[1, 2])

    fx_r = float(K_r[0, 0])
    fy_r = float(K_r[1, 1])
    cx_r = float(K_r[0, 2])
    cy_r = float(K_r[1, 2])

    T_ref_from_target = W2C[int(ref)] @ np.linalg.inv(W2C[int(target)])
    R_rel = T_ref_from_target[:3, :3]
    t_rel = T_ref_from_target[:3, 3]

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    xs_full = np.arange(width, dtype=np.float64)

    for y0 in range(0, height, max(1, int(row_batch))):
        y1 = min(height, y0 + int(row_batch))

        ys = np.arange(y0, y1, dtype=np.float64)
        xs, yy = np.meshgrid(xs_full, ys)

        dep = depth_target[y0:y1, :].reshape(-1).astype(np.float64)

        ray_x = (xs.reshape(-1) - cx_t) / fx_t
        ray_y = (yy.reshape(-1) - cy_t) / fy_t
        ray_z = np.full_like(ray_x, float(z_sign), dtype=np.float64)

        X_target = dep[:, None] * np.stack([ray_x, ray_y, ray_z], axis=1)
        X_ref = X_target @ R_rel.T + t_rel[None, :]

        z = X_ref[:, 2]
        z_safe = np.where(np.abs(z) > 1e-9, z, np.where(z >= 0, 1e-9, -1e-9))

        mx = fx_r * (X_ref[:, 0] / z_safe) + cx_r
        my = fy_r * (X_ref[:, 1] / z_safe) + cy_r

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


def remap_ref_to_target(
    ref_y: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    interp = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation]

    return cv2.remap(
        ref_y.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


# ============================================================
# Evaluation
# ============================================================

def source_frame_idx_for_poc(
    poc: int,
    frame_indices: np.ndarray,
    args: argparse.Namespace,
) -> int:
    if args.source_index_mode == "frame_idx":
        return int(args.seq_start) + int(frame_indices[int(poc)])

    return int(args.seq_start) + int(poc)


def evaluate(args: argparse.Namespace) -> None:
    camparam_path = Path(args.camparam).expanduser().resolve()
    cam = load_camparam_jsonl(camparam_path, pose_mode_override=args.pose_mode)

    width = int(args.width) if args.width > 0 else int(cam["width"])
    height = int(args.height) if args.height > 0 else int(cam["height"])

    if width != cam["width"] or height != cam["height"]:
        raise ValueError(
            f"Size mismatch: args={width}x{height}, "
            f"camParam={cam['width']}x{cam['height']}"
        )

    depth_yuv = resolve_depth_yuv(camparam_path, cam, args.depth_yuv)

    depth = read_depth_yuv420p10le(
        depth_yuv,
        width=width,
        height=height,
        frame_count=cam["frame_count"],
        depth_scale_real=cam["depth_scale_real"],
    )

    W2C = reconstruct_w2c_from_records(
        cam["rvecs"],
        cam["tvecs"],
        cam["pose_mode"],
    )

    pairs = build_pairs(cam["frame_count"], args)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if out_dir is not None:
        ensure_dir(out_dir)

        if args.save_warp_yuv or args.save_png or args.save_mask:
            ensure_dir(out_dir / "warps")

    print("[INFO] camParam     :", camparam_path)
    print("[INFO] depth_yuv    :", depth_yuv)
    print("[INFO] seq_yuv      :", Path(args.seq_yuv).resolve())
    print("[INFO] frames/size  :", cam["frame_count"], f"{width}x{height}")
    print("[INFO] pose_mode    :", cam["pose_mode"])
    print("[INFO] z_sign       :", cam["z_sign"])
    print("[INFO] depth_scale  :", cam["depth_scale_real"])
    print("[INFO] pairs        :", len(pairs))

    rows: list[dict[str, Any]] = []

    for idx, (target, ref, weight, kind) in enumerate(pairs, start=1):
        tar_idx = source_frame_idx_for_poc(target, cam["frame_indices"], args)
        ref_idx = source_frame_idx_for_poc(ref, cam["frame_indices"], args)

        target_y = read_y_frame_yuv420(
            args.seq_yuv,
            width,
            height,
            args.seq_bitdepth,
            tar_idx,
        )

        ref_y = read_y_frame_yuv420(
            args.seq_yuv,
            width,
            height,
            args.seq_bitdepth,
            ref_idx,
        )

        map_x, map_y, valid = camera_map_target_to_ref(
            target=target,
            ref=ref,
            width=width,
            height=height,
            K_all=cam["K"],
            W2C=W2C,
            depth_target=depth[target],
            z_sign=cam["z_sign"],
            z_min=args.z_min,
            row_batch=args.row_batch,
        )

        pred = remap_ref_to_target(
            ref_y,
            map_x,
            map_y,
            interpolation=args.interp,
        )

        metrics = calc_psnr_metrics(
            target_y,
            pred,
            valid,
            args.seq_bitdepth,
        )

        row = {
            "pair_index": idx,
            "target_poc": int(target),
            "ref_poc": int(ref),
            "target_frame_idx": int(tar_idx),
            "ref_frame_idx": int(ref_idx),
            "kind": kind,
            "weight": float(weight),
            **metrics,
        }

        rows.append(row)

        psnr_str = "None" if row["psnr"] is None else f"{row['psnr']:.4f}"

        print(
            f"[PAIR {idx:03d}/{len(pairs):03d}] "
            f"t={target:03d}(src {tar_idx}) <- r={ref:03d}(src {ref_idx}) "
            f"valid={row['valid_ratio']:.4f} "
            f"psnr={psnr_str} "
            f"mae={row['mae']} "
            f"kind={kind}"
        )

        if out_dir is not None and (args.save_warp_yuv or args.save_png or args.save_mask):
            tag = f"t{target:03d}_r{ref:03d}"
            warp_dir = out_dir / "warps"

            if args.save_warp_yuv:
                write_yuv420_y_only(
                    warp_dir / f"warp_{tag}.yuv",
                    pred,
                    args.seq_bitdepth,
                )

            if args.save_png:
                save_gray_png(
                    warp_dir / f"warp_{tag}.png",
                    pred,
                    args.seq_bitdepth,
                )

            if args.save_mask:
                mask_img = valid.astype(np.uint8) * 255
                cv2.imwrite(str(warp_dir / f"valid_{tag}.png"), mask_img)

    psnrs = np.array(
        [r["psnr"] for r in rows if r["psnr"] is not None],
        dtype=np.float64,
    )

    valid_ratios = np.array(
        [r["valid_ratio"] for r in rows],
        dtype=np.float64,
    )

    summary = {
        "camparam": str(camparam_path),
        "depth_yuv": str(depth_yuv),
        "seq_yuv": str(Path(args.seq_yuv).resolve()),
        "frame_count": int(cam["frame_count"]),
        "width": width,
        "height": height,
        "pose_mode": cam["pose_mode"],
        "source_index_mode": args.source_index_mode,
        "pair_source": args.pair_source if not args.pairs.strip() else "cli",
        "pair_count": len(rows),
        "psnr_mean": float(np.mean(psnrs)) if psnrs.size else None,
        "psnr_median": float(np.median(psnrs)) if psnrs.size else None,
        "psnr_min": float(np.min(psnrs)) if psnrs.size else None,
        "psnr_max": float(np.max(psnrs)) if psnrs.size else None,
        "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios.size else None,
        "rows": rows,
    }

    print("============================================================")
    print("Summary")
    print("============================================================")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))

    if out_dir is not None:
        csv_path = out_dir / "pair_warp_psnr.csv"
        json_path = out_dir / "pair_warp_psnr_summary.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("[OUT] csv    :", csv_path)
        print("[OUT] summary:", json_path)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate GOP pair warping PSNR using converted camParam JSONL + depth YUV."
    )

    ap.add_argument("--camparam", required=True, help="Converted *_camParam_*.jsonl")
    ap.add_argument("--depth-yuv", default="", help="Converted depth YUV. If omitted, resolved from camParam header.")
    ap.add_argument("--seq-yuv", required=True, help="Original source YUV420 sequence used for color PSNR")
    ap.add_argument("--seq-bitdepth", type=int, choices=[8, 10], default=10)

    ap.add_argument("--width", type=int, default=0, help="Optional sanity override; default from camParam header")
    ap.add_argument("--height", type=int, default=0, help="Optional sanity override; default from camParam header")
    ap.add_argument("--output-dir", default="", help="Folder for CSV/JSON and optional warp outputs")

    ap.add_argument(
        "--pose-mode",
        choices=["auto", "current_to_previous", "gop_local", "absolute"],
        default="auto",
    )

    ap.add_argument(
        "--source-index-mode",
        choices=["frame_idx", "poc"],
        default="frame_idx",
        help="frame_idx uses JSONL frame_idx to read source YUV; poc uses local GOP index",
    )

    ap.add_argument("--seq-start", type=int, default=0, help="Offset added to selected source frame index")

    ap.add_argument("--pairs", default="", help="Explicit pairs: target:ref[:weight], e.g. 16:0,16:32")
    ap.add_argument("--pair-source", choices=["adjacent", "dyadic", "all"], default="dyadic")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--no-bidirectional-pairs", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)

    ap.add_argument("--z-min", type=float, default=1e-4)
    ap.add_argument("--row-batch", type=int, default=64)
    ap.add_argument("--interp", choices=["nearest", "linear", "cubic"], default="linear")

    ap.add_argument("--save-warp-yuv", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--save-mask", action="store_true")

    args = ap.parse_args()

    if args.row_batch <= 0:
        raise ValueError("--row-batch must be positive")

    return args


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
