#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warp a reference YUV frame to a target frame using VGGT-Omega depth + camera output.

Expected inputs from run_vggt_omega_yuv.py default output:
  - input original YUV sequence
  - *_depth_inverse_yuv420p10le.yuv
  - *_camera.jsonl

Default warping direction:
  target pixel + target depth -> world -> reference camera -> reference pixel
  then cv2.remap(ref_frame, ref_x, ref_y) to synthesize target view.

Example:
  python warp_vggt_omega_yuv.py \
    --yuv /path/to/input_1920x1080_420p10le.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p10le \
    --depth-yuv out/test_000_015_depth_inverse_yuv420p10le.yuv \
    --camera-jsonl out/test_000_015_camera.jsonl \
    --ref-idx 0 --tar-idx 7 \
    --output-prefix out/warp_ref000_to_tar007

Outputs:
  <prefix>_warped.yuv                 single-frame YUV420 warped ref image
  <prefix>_valid_mask_yuv420p.yuv     single-frame 8-bit YUV420 valid mask
  <prefix>_map_stats.json             projection/validity statistics
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Literal

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc

PixFmt = Literal["yuv420p", "yuv420p10le"]
DepthMode = Literal["linear", "inverse"]
InvalidFill = Literal["black", "edge", "copy_target", "neutral"]


def normalize_pix_fmt(s: str) -> PixFmt:
    s = s.lower().replace("-", "").replace("_", "")
    aliases = {
        "420p": "yuv420p",
        "yuv420p": "yuv420p",
        "i420": "yuv420p",
        "420p8": "yuv420p",
        "420p10le": "yuv420p10le",
        "yuv420p10le": "yuv420p10le",
        "i010": "yuv420p10le",
    }
    if s not in aliases:
        raise ValueError(f"Unsupported pix-fmt: {s}. Use yuv420p or yuv420p10le.")
    return aliases[s]  # type: ignore[return-value]


def frame_size_bytes(width: int, height: int, pix_fmt: PixFmt) -> int:
    if width % 2 or height % 2:
        raise ValueError("YUV420 requires even width and height.")
    samples = width * height + 2 * ((width // 2) * (height // 2))
    return samples if pix_fmt == "yuv420p" else samples * 2


def read_yuv420_frame(
    path: str,
    frame_idx: int,
    width: int,
    height: int,
    pix_fmt: PixFmt,
    tenbit_shift_right: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = frame_size_bytes(width, height, pix_fmt)
    with open(path, "rb") as f:
        f.seek(frame_idx * fs)
        raw = f.read(fs)
    if len(raw) != fs:
        raise EOFError(f"Cannot read frame {frame_idx} from {path}: expected {fs} bytes, got {len(raw)}")

    y_n = width * height
    uv_n = (width // 2) * (height // 2)

    if pix_fmt == "yuv420p":
        arr = np.frombuffer(raw, dtype=np.uint8)
        y = arr[:y_n].reshape(height, width).copy()
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).copy()
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).copy()
    else:
        arr = np.frombuffer(raw, dtype="<u2")
        if tenbit_shift_right > 0:
            arr = arr >> tenbit_shift_right
        y = arr[:y_n].reshape(height, width).copy()
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).copy()
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).copy()
    return y, u, v


def write_yuv420_frame(
    path: str,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pix_fmt: PixFmt,
    bit_depth: int,
) -> None:
    maxv = (1 << bit_depth) - 1
    y = np.clip(np.rint(y), 0, maxv)
    u = np.clip(np.rint(u), 0, maxv)
    v = np.clip(np.rint(v), 0, maxv)
    with open(path, "wb") as f:
        if pix_fmt == "yuv420p":
            f.write(y.astype(np.uint8).tobytes())
            f.write(u.astype(np.uint8).tobytes())
            f.write(v.astype(np.uint8).tobytes())
        else:
            f.write(y.astype("<u2").tobytes())
            f.write(u.astype("<u2").tobytes())
            f.write(v.astype("<u2").tobytes())


def write_mask_yuv420p(path: str, mask: np.ndarray) -> None:
    h, w = mask.shape
    y = (mask.astype(np.uint8) * 255)
    u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes())


def load_camera_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "frame_idx" not in rec:
                raise ValueError(f"camera jsonl line {line_no} has no frame_idx")
            records.append(rec)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_depth_from_npz(npz_path: str, tar_idx: int) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    if "frame_indices" not in z or "depth_original" not in z:
        raise ValueError("NPZ must contain frame_indices and depth_original")
    frame_indices = z["frame_indices"].astype(np.int64)
    matches = np.where(frame_indices == tar_idx)[0]
    if len(matches) != 1:
        raise ValueError(f"target frame {tar_idx} not found uniquely in {npz_path}; matches={matches.tolist()}")
    return z["depth_original"][int(matches[0])].astype(np.float32)


def load_quantized_depth_yuv(
    depth_yuv: str,
    depth_frame_pos: int,
    width: int,
    height: int,
    depth_pix_fmt: PixFmt,
    depth_meta: dict,
) -> np.ndarray:
    y, _, _ = read_yuv420_frame(depth_yuv, depth_frame_pos, width, height, depth_pix_fmt)
    bit_depth = 8 if depth_pix_fmt == "yuv420p" else 10
    max_code = float((1 << bit_depth) - 1)
    qmin = float(depth_meta["quant_min"])
    qmax = float(depth_meta["quant_max"])
    mode: DepthMode = depth_meta.get("depth_quant_mode", "inverse")

    q = y.astype(np.float32) / max_code
    qsrc = qmin + q * (qmax - qmin)

    if mode == "linear":
        depth = qsrc
    elif mode == "inverse":
        eps = max(1e-12, abs(qmax - qmin) * 1e-9)
        depth = 1.0 / np.maximum(qsrc, eps)
    else:
        raise ValueError(f"Unsupported depth_quant_mode: {mode}")

    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def as_k3(k: list | np.ndarray) -> np.ndarray:
    K = np.asarray(k, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected K shape 3x3, got {K.shape}")
    return K


def as_rt34(e: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(e, dtype=np.float64)
    if E.shape == (3, 4):
        R = E[:, :3]
        t = E[:, 3]
    elif E.shape == (4, 4):
        R = E[:3, :3]
        t = E[:3, 3]
    else:
        raise ValueError(f"Expected extrinsic shape 3x4 or 4x4, got {E.shape}")
    return R, t


def make_backward_map(
    depth_tar: np.ndarray,
    K_ref: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    K_tar: np.ndarray,
    R_tar: np.ndarray,
    t_tar: np.ndarray,
    min_depth: float = 1e-8,
    chunk_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """For each target pixel, compute corresponding reference pixel coordinate."""
    h, w = depth_tar.shape
    inv_K_tar = np.linalg.inv(K_tar)
    Rt_tar = R_tar.T

    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)

    total_z_valid = 0
    total_in_front_ref = 0
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        ys, xs = np.mgrid[y0:y1, 0:w]
        z = depth_tar[y0:y1].astype(np.float64)
        depth_ok = np.isfinite(z) & (z > min_depth)
        total_z_valid += int(depth_ok.sum())

        ones = np.ones_like(z)
        pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), ones], axis=0).reshape(3, -1)
        rays_tar = inv_K_tar @ pix
        z_flat = z.reshape(-1)
        x_tar = rays_tar * z_flat[None, :]

        # camera_from_world: X_cam = R * X_world + t
        # world_from_target: X_world = R_tar.T * (X_tar - t_tar)
        x_world = Rt_tar @ (x_tar - t_tar.reshape(3, 1))
        x_ref = R_ref @ x_world + t_ref.reshape(3, 1)

        zr = x_ref[2]
        in_front = zr > min_depth
        total_in_front_ref += int((in_front & depth_ok.reshape(-1)).sum())

        proj = K_ref @ x_ref
        xr = proj[0] / np.maximum(proj[2], min_depth)
        yr = proj[1] / np.maximum(proj[2], min_depth)

        inside = (xr >= 0.0) & (xr <= w - 1.0) & (yr >= 0.0) & (yr <= h - 1.0)
        ok = depth_ok.reshape(-1) & in_front & inside & np.isfinite(xr) & np.isfinite(yr)

        mx = map_x[y0:y1].reshape(-1)
        my = map_y[y0:y1].reshape(-1)
        vv = valid[y0:y1].reshape(-1)
        mx[ok] = xr[ok].astype(np.float32)
        my[ok] = yr[ok].astype(np.float32)
        vv[ok] = True

    stats = {
        "pixels": int(h * w),
        "target_depth_valid": int(total_z_valid),
        "target_depth_valid_ratio": float(total_z_valid / max(h * w, 1)),
        "in_front_of_ref_camera": int(total_in_front_ref),
        "projection_inside_ref": int(valid.sum()),
        "projection_inside_ref_ratio": float(valid.mean()),
    }
    return map_x, map_y, valid, stats


def remap_plane(
    plane: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: float,
) -> np.ndarray:
    remapped = cv2.remap(
        plane.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=float(border_value),
    )
    # cv2 border is not enough because invalid positions have map -1 and can still sample border in some modes.
    remapped[~valid] = border_value
    return remapped


def chroma_maps_from_luma(map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map_x.shape
    cw, ch = w // 2, h // 2
    # Approximate chroma-center mapping by resizing luma maps, then convert full-res ref coords to chroma coords.
    cmx = cv2.resize(map_x, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cmy = cv2.resize(map_y, (cw, ch), interpolation=cv2.INTER_LINEAR) * 0.5
    cvalid_f = cv2.resize(valid.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    cvalid = cvalid_f > 0.999
    cmx[~cvalid] = -1.0
    cmy[~cvalid] = -1.0
    return cmx.astype(np.float32), cmy.astype(np.float32), cvalid


def fill_invalid_with_target(
    warped: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    valid_luma: np.ndarray,
    valid_chroma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target is None:
        return warped
    wy, wu, wv = warped
    ty, tu, tv = target
    out_y = wy.copy()
    out_u = wu.copy()
    out_v = wv.copy()
    out_y[~valid_luma] = ty.astype(np.float32)[~valid_luma]
    out_u[~valid_chroma] = tu.astype(np.float32)[~valid_chroma]
    out_v[~valid_chroma] = tv.astype(np.float32)[~valid_chroma]
    return out_y, out_u, out_v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warp reference YUV frame to target using VGGT-Omega depth/camera")
    p.add_argument("--yuv", required=True, help="Original source YUV sequence")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    p.add_argument("--ref-idx", type=int, required=True, help="absolute frame index of reference frame in original YUV")
    p.add_argument("--tar-idx", type=int, required=True, help="absolute frame index of target frame in original YUV")
    p.add_argument("--camera-jsonl", required=True, help="*_camera.jsonl from run_vggt_omega_yuv.py")
    p.add_argument("--depth-yuv", default=None, help="quantized depth YUV from run_vggt_omega_yuv.py")
    p.add_argument("--depth-pix-fmt", default="yuv420p10le", help="depth YUV pix fmt; default yuv420p10le")
    p.add_argument("--npz", default=None, help="optional *_vggt_omega_outputs.npz. If set, uses raw float depth instead of quantized depth YUV")
    p.add_argument("--output-prefix", required=True)

    p.add_argument("--tenbit-shift-right", type=int, default=0,
                   help="Use 0 for normal yuv420p10le. Use 6 only if samples are MSB-aligned in uint16.")
    p.add_argument("--interp", choices=["linear", "nearest", "cubic"], default="linear")
    p.add_argument("--border", choices=["constant", "replicate"], default="constant")
    p.add_argument("--invalid-fill", choices=["black", "neutral", "copy_target"], default="black",
                   help="How to fill pixels with no valid projection. copy_target reads target frame and fills holes with target.")
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--write-mask", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pix_fmt: PixFmt = normalize_pix_fmt(args.pix_fmt)
    depth_pix_fmt: PixFmt = normalize_pix_fmt(args.depth_pix_fmt)
    bit_depth = 8 if pix_fmt == "yuv420p" else 10
    maxv = (1 << bit_depth) - 1
    neutral = 128 if bit_depth == 8 else 512

    if args.npz is None and args.depth_yuv is None:
        raise ValueError("Provide either --depth-yuv or --npz")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    records = load_camera_jsonl(args.camera_jsonl)
    pos_by_frame = {int(r["frame_idx"]): i for i, r in enumerate(records)}
    if args.ref_idx not in pos_by_frame:
        raise ValueError(f"ref_idx {args.ref_idx} not found in camera JSONL. Available: {sorted(pos_by_frame)[:10]}...")
    if args.tar_idx not in pos_by_frame:
        raise ValueError(f"tar_idx {args.tar_idx} not found in camera JSONL. Available: {sorted(pos_by_frame)[:10]}...")

    ref_pos = pos_by_frame[args.ref_idx]
    tar_pos = pos_by_frame[args.tar_idx]
    ref_rec = records[ref_pos]
    tar_rec = records[tar_pos]

    K_ref = as_k3(ref_rec["intrinsic_original"])
    K_tar = as_k3(tar_rec["intrinsic_original"])
    R_ref, t_ref = as_rt34(ref_rec["extrinsic"])
    R_tar, t_tar = as_rt34(tar_rec["extrinsic"])

    if args.npz:
        depth_tar = load_depth_from_npz(args.npz, args.tar_idx)
        depth_source = args.npz
    else:
        depth_meta = tar_rec.get("depth_output", {})
        for key in ["quant_min", "quant_max", "depth_quant_mode"]:
            if key not in depth_meta:
                raise ValueError(f"camera JSONL target record has no depth_output.{key}; use --npz instead")
        depth_tar = load_quantized_depth_yuv(
            args.depth_yuv,
            tar_pos,
            args.width,
            args.height,
            depth_pix_fmt,
            depth_meta,
        )
        depth_source = args.depth_yuv

    if depth_tar.shape != (args.height, args.width):
        raise ValueError(f"depth shape {depth_tar.shape} != {(args.height, args.width)}")

    ref_y, ref_u, ref_v = read_yuv420_frame(
        args.yuv, args.ref_idx, args.width, args.height, pix_fmt, args.tenbit_shift_right
    )

    map_x, map_y, valid, stats = make_backward_map(
        depth_tar=depth_tar,
        K_ref=K_ref,
        R_ref=R_ref,
        t_ref=t_ref,
        K_tar=K_tar,
        R_tar=R_tar,
        t_tar=t_tar,
        min_depth=args.min_depth,
        chunk_rows=args.chunk_rows,
    )

    if args.interp == "nearest":
        interp = cv2.INTER_NEAREST
    elif args.interp == "cubic":
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR

    border_mode = cv2.BORDER_REPLICATE if args.border == "replicate" else cv2.BORDER_CONSTANT
    y_fill = 0.0 if args.invalid_fill in ["black", "copy_target"] else float(neutral)
    uv_fill = float(neutral)

    wy = remap_plane(ref_y, map_x, map_y, valid, interp, border_mode, y_fill)
    cmx, cmy, cvalid = chroma_maps_from_luma(map_x, map_y, valid)
    wu = remap_plane(ref_u, cmx, cmy, cvalid, interp, border_mode, uv_fill)
    wv = remap_plane(ref_v, cmx, cmy, cvalid, interp, border_mode, uv_fill)

    if args.invalid_fill == "copy_target":
        target = read_yuv420_frame(args.yuv, args.tar_idx, args.width, args.height, pix_fmt, args.tenbit_shift_right)
        wy, wu, wv = fill_invalid_with_target((wy, wu, wv), target, valid, cvalid)

    out_yuv = args.output_prefix + "_warped.yuv"
    write_yuv420_frame(out_yuv, wy, wu, wv, pix_fmt, bit_depth)

    out_mask = args.output_prefix + "_valid_mask_yuv420p.yuv"
    if args.write_mask:
        write_mask_yuv420p(out_mask, valid)

    stats.update(
        {
            "source_yuv": os.path.abspath(args.yuv),
            "depth_source": os.path.abspath(depth_source),
            "camera_jsonl": os.path.abspath(args.camera_jsonl),
            "ref_idx": int(args.ref_idx),
            "tar_idx": int(args.tar_idx),
            "ref_camera_jsonl_position": int(ref_pos),
            "tar_camera_jsonl_position": int(tar_pos),
            "width": int(args.width),
            "height": int(args.height),
            "pix_fmt": pix_fmt,
            "depth_min": float(np.nanmin(depth_tar)),
            "depth_max": float(np.nanmax(depth_tar)),
            "depth_mean": float(np.nanmean(depth_tar)),
            "output_warped_yuv": os.path.abspath(out_yuv),
            "output_valid_mask_yuv420p": os.path.abspath(out_mask) if args.write_mask else None,
            "note": "Backward warp: target depth -> target camera -> world -> reference camera -> reference pixel.",
        }
    )
    out_stats = args.output_prefix + "_map_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done")
    print(f"  warped yuv : {out_yuv}")
    if args.write_mask:
        print(f"  valid mask : {out_mask}")
    print(f"  stats      : {out_stats}")
    print(f"  valid ratio: {stats['projection_inside_ref_ratio']:.6f}")


if __name__ == "__main__":
    main()
