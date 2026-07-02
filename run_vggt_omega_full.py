#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run VGGT-Omega-1B-512 on a raw YUV420 sequence and export
  1) depth as YUV420p / YUV420p10le
  2) raw float depth/confidence as NPZ
  3) camera parameters as JSONL

Supports overlapping GOP/RAP split:
  --gop-size 33 gives:
    rap0: 0  ~ 32
    rap1: 32 ~ 64
    rap2: 64 ~ 96
  Each RAP is separately fed into NN, with 1 selected-frame overlap.

Example:
  python run_vggt_omega_yuv.py \
    --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
    --yuv input_1920x1080_420p10le.yuv \
    --width 1920 --height 1080 \
    --pix-fmt yuv420p10le \
    --start 0 --end 64 \
    --gop-size 33 \
    --output-prefix out/test
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


PixFmt = Literal["yuv420p", "yuv420p10le"]
MatrixName = Literal["bt601", "bt709", "bt2020"]
RangeName = Literal["limited", "full"]
ResizeMode = Literal["balanced", "max_size"]
DepthQuantMode = Literal["linear", "inverse"]
DepthNormMode = Literal["global", "per-frame", "fixed"]
FillCropMode = Literal["edge", "zero"]


@dataclass
class PreprocessMeta:
    src_width: int
    src_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    model_width: int
    model_height: int
    resize_mode: str
    image_resolution: int
    patch_size: int

    @property
    def scale_x(self) -> float:
        return self.model_width / self.crop_width

    @property
    def scale_y(self) -> float:
        return self.model_height / self.crop_height


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
    f,
    frame_idx: int,
    width: int,
    height: int,
    pix_fmt: PixFmt,
    tenbit_shift_right: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = frame_size_bytes(width, height, pix_fmt)
    f.seek(frame_idx * fs)
    raw = f.read(fs)
    if len(raw) != fs:
        raise EOFError(f"Cannot read frame {frame_idx}: expected {fs} bytes, got {len(raw)}")

    y_n = width * height
    uv_n = (width // 2) * (height // 2)

    if pix_fmt == "yuv420p":
        arr = np.frombuffer(raw, dtype=np.uint8)
        y = arr[:y_n].reshape(height, width).astype(np.float32)
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).astype(np.float32)
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype="<u2")
        if tenbit_shift_right > 0:
            arr = arr >> tenbit_shift_right
        y = arr[:y_n].reshape(height, width).astype(np.float32)
        u = arr[y_n : y_n + uv_n].reshape(height // 2, width // 2).astype(np.float32)
        v = arr[y_n + uv_n : y_n + 2 * uv_n].reshape(height // 2, width // 2).astype(np.float32)

    return y, u, v


def yuv420_to_rgb_uint8(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bit_depth: int,
    matrix: MatrixName = "bt709",
    value_range: RangeName = "limited",
) -> np.ndarray:
    """Convert planar YUV420 to RGB uint8. Returns HxWx3."""
    h, w = y.shape
    u_full = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
    v_full = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)

    if matrix == "bt601":
        kr, kb = 0.299, 0.114
    elif matrix == "bt709":
        kr, kb = 0.2126, 0.0722
    elif matrix == "bt2020":
        kr, kb = 0.2627, 0.0593
    else:
        raise ValueError(matrix)

    kg = 1.0 - kr - kb
    scale = float(1 << (bit_depth - 8))

    if value_range == "limited":
        yy = (y - 16.0 * scale) / (219.0 * scale)
        cb = (u_full - 128.0 * scale) / (224.0 * scale)
        cr = (v_full - 128.0 * scale) / (224.0 * scale)
    elif value_range == "full":
        maxv = float((1 << bit_depth) - 1)
        yy = y / maxv
        cb = u_full / maxv - 0.5
        cr = v_full / maxv - 0.5
    else:
        raise ValueError(value_range)

    r = yy + (2.0 - 2.0 * kr) * cr
    b = yy + (2.0 - 2.0 * kb) * cb
    g = (yy - kr * r - kb * b) / kg

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def crop_to_supported_aspect(width: int, height: int) -> tuple[int, int, int, int]:
    """Matches VGGT-Omega load_fn.py: keep aspect ratio height/width in [0.5, 2.0]."""
    min_ar, max_ar = 0.5, 2.0
    ar = height / max(width, 1)

    if ar < min_ar:
        crop_w = min(width, max(1, int(round(height / min_ar))))
        left = max((width - crop_w) // 2, 0)
        return left, 0, crop_w, height

    if ar > max_ar:
        crop_h = min(height, max(1, int(round(width * max_ar))))
        top = max((height - crop_h) // 2, 0)
        return 0, top, width, crop_h

    return 0, 0, width, height


def round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(round(float(value) / patch_size)) * patch_size)


def target_shape(
    crop_width: int,
    crop_height: int,
    image_resolution: int,
    patch_size: int,
    mode: ResizeMode,
) -> tuple[int, int]:
    """Returns model_height, model_width, matching VGGT-Omega load_fn.py."""
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    ar = crop_height / max(crop_width, 1)

    if mode == "balanced":
        token_number = (image_resolution // patch_size) ** 2
        w_patches = math.sqrt(token_number / ar)
        h_patches = token_number / w_patches
        w_patches = max(1, int(round(w_patches)))
        h_patches = max(1, int(round(h_patches)))
        return h_patches * patch_size, w_patches * patch_size

    if mode == "max_size":
        if ar >= 1.0:
            out_h = image_resolution
            out_w = round_to_patch_multiple(image_resolution / ar, patch_size)
        else:
            out_w = image_resolution
            out_h = round_to_patch_multiple(image_resolution * ar, patch_size)
        return out_h, out_w

    raise ValueError(mode)


def preprocess_rgb_frames(
    rgb_frames: list[np.ndarray],
    image_resolution: int = 512,
    mode: ResizeMode = "balanced",
    patch_size: int = 16,
) -> tuple[torch.Tensor, PreprocessMeta]:
    if not rgb_frames:
        raise ValueError("No frames")

    src_h, src_w = rgb_frames[0].shape[:2]
    left, top, crop_w, crop_h = crop_to_supported_aspect(src_w, src_h)
    model_h, model_w = target_shape(crop_w, crop_h, image_resolution, patch_size, mode)

    tensors: list[torch.Tensor] = []

    for rgb in rgb_frames:
        if rgb.shape[:2] != (src_h, src_w):
            raise ValueError("All frames must have the same shape")

        crop = rgb[top : top + crop_h, left : left + crop_w]
        img = Image.fromarray(crop, mode="RGB")
        img = img.resize((model_w, model_h), Image.Resampling.BICUBIC)

        arr = np.asarray(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensors.append(ten)

    meta = PreprocessMeta(
        src_width=src_w,
        src_height=src_h,
        crop_left=left,
        crop_top=top,
        crop_width=crop_w,
        crop_height=crop_h,
        model_width=model_w,
        model_height=model_h,
        resize_mode=mode,
        image_resolution=image_resolution,
        patch_size=patch_size,
    )

    return torch.stack(tensors, dim=0), meta


def intrinsic_to_original(K_model: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """Convert K from resized/cropped model coordinates to original image coordinates."""
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K_model[0, 0] / meta.scale_x
    K[1, 1] = K_model[1, 1] / meta.scale_y
    K[0, 2] = K_model[0, 2] / meta.scale_x + meta.crop_left
    K[1, 2] = K_model[1, 2] / meta.scale_y + meta.crop_top
    return K


def resize_depth_to_original(
    depth_model: np.ndarray,
    meta: PreprocessMeta,
    fill_crop: FillCropMode = "edge",
) -> np.ndarray:
    crop_depth = cv2.resize(
        depth_model.astype(np.float32),
        (meta.crop_width, meta.crop_height),
        interpolation=cv2.INTER_LINEAR,
    )

    if meta.crop_width == meta.src_width and meta.crop_height == meta.src_height:
        return crop_depth

    if fill_crop == "zero":
        full = np.zeros((meta.src_height, meta.src_width), dtype=np.float32)
        full[
            meta.crop_top : meta.crop_top + meta.crop_height,
            meta.crop_left : meta.crop_left + meta.crop_width,
        ] = crop_depth
        return full

    pad_top = meta.crop_top
    pad_left = meta.crop_left
    pad_bottom = meta.src_height - meta.crop_top - meta.crop_height
    pad_right = meta.src_width - meta.crop_left - meta.crop_width

    return np.pad(
        crop_depth,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )


def finite_positive_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def depth_source_for_quant(depth: np.ndarray, mode: DepthQuantMode) -> np.ndarray:
    if mode == "linear":
        return depth.astype(np.float32)

    if mode == "inverse":
        out = np.full_like(depth, np.nan, dtype=np.float32)
        m = finite_positive_mask(depth)
        out[m] = 1.0 / np.maximum(depth[m], 1e-12)
        return out

    raise ValueError(mode)


def compute_quant_range(
    qsrc: np.ndarray,
    norm_mode: DepthNormMode,
    frame_idx: int,
    fixed_min: float | None,
    fixed_max: float | None,
    clip_percentile: tuple[float, float],
) -> tuple[float, float]:
    if norm_mode == "fixed":
        if fixed_min is None or fixed_max is None:
            raise ValueError("--depth-norm fixed requires --depth-min and --depth-max")
        return float(fixed_min), float(fixed_max)

    arr = qsrc[frame_idx] if norm_mode == "per-frame" else qsrc
    valid = np.isfinite(arr)

    if not np.any(valid):
        return 0.0, 1.0

    lo_p, hi_p = clip_percentile
    vals = arr[valid]

    lo = float(np.percentile(vals, lo_p))
    hi = float(np.percentile(vals, hi_p))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def quantize_depth_to_y(
    qsrc_frame: np.ndarray,
    qmin: float,
    qmax: float,
    out_bit_depth: int,
) -> np.ndarray:
    max_code = (1 << out_bit_depth) - 1
    x = np.nan_to_num(qsrc_frame, nan=qmin, posinf=qmax, neginf=qmin)
    x = np.clip(x, qmin, qmax)
    q = np.round((x - qmin) * (max_code / (qmax - qmin))).astype(np.uint16)
    return np.clip(q, 0, max_code).astype(np.uint16)


def write_depth_yuv420(
    path: str,
    depth: np.ndarray,
    mode: DepthQuantMode,
    norm_mode: DepthNormMode,
    out_pix_fmt: PixFmt,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
    clip_percentile: tuple[float, float] = (0.0, 100.0),
) -> list[dict]:
    """Write depth as Y-only 420 file. U/V are neutral. Returns quant metadata per frame."""
    n, h, w = depth.shape

    if w % 2 or h % 2:
        raise ValueError("YUV420 depth output requires even width and height")

    out_bit_depth = 8 if out_pix_fmt == "yuv420p" else 10
    neutral = 128 if out_bit_depth == 8 else 512
    qsrc = depth_source_for_quant(depth, mode)

    metadata = []

    with open(path, "wb") as f:
        for i in range(n):
            qmin, qmax = compute_quant_range(
                qsrc,
                norm_mode,
                i,
                fixed_min,
                fixed_max,
                clip_percentile,
            )

            yq = quantize_depth_to_y(qsrc[i], qmin, qmax, out_bit_depth)

            if out_bit_depth == 8:
                y = yq.astype(np.uint8)
                uv = np.full((h // 2, w // 2), neutral, dtype=np.uint8)
            else:
                y = yq.astype("<u2", copy=False)
                uv = np.full((h // 2, w // 2), neutral, dtype="<u2")

            f.write(y.tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())

            metadata.append(
                {
                    "depth_quant_mode": mode,
                    "depth_norm_mode": norm_mode,
                    "quant_min": qmin,
                    "quant_max": qmax,
                    "bit_depth": out_bit_depth,
                    "note": (
                        "Y stores depth directly"
                        if mode == "linear"
                        else "Y stores inverse depth = 1/depth"
                    ),
                }
            )

    return metadata


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.removeprefix("module."): v for k, v in state.items()}

    model.load_state_dict(state)


def make_rap_frame_groups(
    frame_indices: list[int],
    gop_size: int,
    drop_incomplete_gop: bool = False,
) -> list[list[int]]:
    """
    GOP/RAP overlap split.

    gop_size=33, frame_indices=0..64:
      rap0: 0..32
      rap1: 32..64

    Step is gop_size - 1.
    Therefore adjacent RAPs overlap by exactly one selected frame.
    """
    if not frame_indices:
        return []

    if gop_size <= 0:
        return [frame_indices]

    if gop_size < 2:
        raise ValueError("--gop-size must be >= 2 when enabled")

    step = gop_size - 1
    groups: list[list[int]] = []

    pos = 0
    n = len(frame_indices)

    while pos < n:
        group = frame_indices[pos : pos + gop_size]

        if drop_incomplete_gop and len(group) < gop_size:
            break

        if group:
            groups.append(group)

        if group and group[-1] == frame_indices[-1]:
            break

        pos += step

    return groups


def run_one_rap(
    args: argparse.Namespace,
    model: torch.nn.Module,
    pix_fmt: PixFmt,
    out_depth_pix_fmt: PixFmt,
    frame_indices: list[int],
    rap_output_prefix: str,
    rap_index: int,
    rap_name: str,
) -> dict:
    src_bit_depth = 8 if pix_fmt == "yuv420p" else 10

    print(
        f"[{rap_name}] Reading frames {frame_indices[0]}..{frame_indices[-1]} "
        f"({len(frame_indices)} frames)"
    )

    rgb_frames: list[np.ndarray] = []

    with open(args.yuv, "rb") as f:
        for idx in frame_indices:
            y, u, v = read_yuv420_frame(
                f,
                idx,
                args.width,
                args.height,
                pix_fmt,
                tenbit_shift_right=args.tenbit_shift_right,
            )

            rgb = yuv420_to_rgb_uint8(
                y,
                u,
                v,
                bit_depth=src_bit_depth,
                matrix=args.matrix,
                value_range=args.value_range,
            )

            rgb_frames.append(rgb)

    images, meta = preprocess_rgb_frames(
        rgb_frames,
        image_resolution=args.image_resolution,
        mode=args.resize_mode,
        patch_size=args.patch_size,
    )

    print(f"[{rap_name}] Model input tensor: {tuple(images.shape)}  (S,C,H,W)")

    if args.save_model_input_rgb:
        rgb_dir = rap_output_prefix + "_model_input_rgb"
        os.makedirs(rgb_dir, exist_ok=True)

        for i, t in enumerate(images):
            arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(
                os.path.join(rgb_dir, f"{frame_indices[i]:06d}.png")
            )

    print(f"[{rap_name}] Running inference")

    with torch.inference_mode():
        pred = model(images.to("cuda", non_blocking=True))
        extrinsic_t, intrinsic_t = encoding_to_camera(
            pred["pose_enc"],
            pred["images"].shape[-2:],
        )

    depth_model = pred["depth"].detach().float().cpu().numpy()
    conf_model = pred["depth_conf"].detach().float().cpu().numpy()
    pose_enc = pred["pose_enc"].detach().float().cpu().numpy()
    intrinsic_model = intrinsic_t.detach().float().cpu().numpy()
    extrinsic = extrinsic_t.detach().float().cpu().numpy()

    # Drop batch dim.
    if depth_model.shape[0] == 1:
        depth_model = depth_model[0]
        conf_model = conf_model[0]
        pose_enc = pose_enc[0]
        intrinsic_model = intrinsic_model[0]
        extrinsic = extrinsic[0]

    # depth: S,H,W,1 -> S,H,W
    if depth_model.ndim == 4 and depth_model.shape[-1] == 1:
        depth_model = depth_model[..., 0]

    # confidence도 S,H,W,1일 수 있음.
    if conf_model.ndim == 4 and conf_model.shape[-1] == 1:
        conf_model = conf_model[..., 0]

    print(f"[{rap_name}] Depth model shape: {depth_model.shape}")

    depth_orig = np.stack(
        [
            resize_depth_to_original(depth_model[i], meta, args.fill_crop)
            for i in range(len(frame_indices))
        ],
        axis=0,
    ).astype(np.float32)

    conf_orig = np.stack(
        [
            resize_depth_to_original(conf_model[i], meta, args.fill_crop)
            for i in range(len(frame_indices))
        ],
        axis=0,
    ).astype(np.float32)

    K_orig = np.stack(
        [
            intrinsic_to_original(intrinsic_model[i], meta)
            for i in range(len(frame_indices))
        ],
        axis=0,
    )

    depth_yuv_path = rap_output_prefix + f"_depth_{args.depth_quant_mode}_{out_depth_pix_fmt}.yuv"

    quant_meta = write_depth_yuv420(
        depth_yuv_path,
        depth_orig,
        mode=args.depth_quant_mode,
        norm_mode=args.depth_norm,
        out_pix_fmt=out_depth_pix_fmt,
        fixed_min=args.depth_min,
        fixed_max=args.depth_max,
        clip_percentile=tuple(args.clip_percentile),
    )

    conf_yuv_path = None

    if args.save_conf_yuv:
        conf_yuv_path = rap_output_prefix + f"_depth_conf_{out_depth_pix_fmt}.yuv"

        write_depth_yuv420(
            conf_yuv_path,
            conf_orig,
            mode="linear",
            norm_mode="global",
            out_pix_fmt=out_depth_pix_fmt,
            clip_percentile=tuple(args.clip_percentile),
        )

    npz_path = rap_output_prefix + "_vggt_omega_outputs.npz"

    np.savez_compressed(
        npz_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        rap_index=np.asarray(rap_index, dtype=np.int32),
        rap_name=np.asarray(rap_name, dtype=object),
        depth_original=depth_orig,
        depth_conf_original=conf_orig,
        depth_model=depth_model.astype(np.float32),
        depth_conf_model=conf_model.astype(np.float32),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic_model=intrinsic_model.astype(np.float32),
        intrinsic_original=K_orig.astype(np.float32),
        pose_enc=pose_enc.astype(np.float32),
        preprocess_meta=np.asarray(json.dumps(asdict(meta)), dtype=object),
    )

    camera_jsonl_path = rap_output_prefix + "_camera.jsonl"

    with open(camera_jsonl_path, "w", encoding="utf-8") as f:
        for i, frame_idx in enumerate(frame_indices):
            rec = {
                "rap_index": int(rap_index),
                "rap_name": rap_name,
                "frame_idx": int(frame_idx),
                "rap_frame_index": int(i),
                "source": {
                    "yuv": os.path.abspath(args.yuv),
                    "width": args.width,
                    "height": args.height,
                    "pix_fmt": pix_fmt,
                    "matrix": args.matrix,
                    "range": args.value_range,
                },
                "preprocess": asdict(meta),
                "camera_convention": {
                    "extrinsic": "camera_from_world, OpenCV coordinates, 3x4 [R|T]",
                    "intrinsic_model": "K in VGGT resized/cropped model-input pixel coordinates",
                    "intrinsic_original": "K converted back to original YUV pixel coordinates",
                    "depth_scale": "same arbitrary scene scale as predicted camera translation",
                },
                "extrinsic": extrinsic[i].astype(float).tolist(),
                "intrinsic_model": intrinsic_model[i].astype(float).tolist(),
                "intrinsic_original": K_orig[i].astype(float).tolist(),
                "pose_enc_9d": pose_enc[i].astype(float).tolist(),
                "depth_output": {
                    "depth_yuv": os.path.abspath(depth_yuv_path),
                    "float_npz": os.path.abspath(npz_path),
                    **quant_meta[i],
                },
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest_path = rap_output_prefix + "_manifest.json"

    manifest = {
        "rap_index": int(rap_index),
        "rap_name": rap_name,
        "output_prefix": os.path.abspath(rap_output_prefix),
        "frame_indices": [int(x) for x in frame_indices],
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "num_frames": len(frame_indices),
        "depth_yuv": os.path.abspath(depth_yuv_path),
        "camera_jsonl": os.path.abspath(camera_jsonl_path),
        "npz": os.path.abspath(npz_path),
        "confidence_yuv": os.path.abspath(conf_yuv_path) if conf_yuv_path else None,
        "preprocess": asdict(meta),
        "notes": [
            "Depth Y plane is quantized for YUV storage; use NPZ for raw float depth.",
            "For inverse mode, Y stores 1/depth, not depth.",
            "Extrinsic is camera_from_world in OpenCV coordinates, matching VGGT-Omega pose_enc utility.",
            "When using GOP/RAP split, overlapped frames are independently inferred and stored in adjacent RAP outputs.",
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[{rap_name}] Done")
    print(f"  depth yuv : {depth_yuv_path}")
    print(f"  camera    : {camera_jsonl_path}")
    print(f"  raw npz   : {npz_path}")
    print(f"  manifest  : {manifest_path}")

    if conf_yuv_path:
        print(f"  conf yuv  : {conf_yuv_path}")

    del pred
    del images
    torch.cuda.empty_cache()

    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT-Omega YUV420 test runner")

    p.add_argument("--checkpoint", required=True, help="VGGT-Omega model.pt path")
    p.add_argument("--yuv", required=True, help="Input raw YUV420 sequence")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    p.add_argument("--start", type=int, required=True, help="inclusive frame index")
    p.add_argument("--end", type=int, required=True, help="inclusive frame index")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output-prefix", required=True)

    p.add_argument("--gop-size", type=int, default=0)
    p.add_argument("--rap-name", type=str, default="rap")
    p.add_argument("--drop-incomplete-gop", action="store_true")

    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--resize-mode", choices=["balanced", "max_size"], default="balanced")
    p.add_argument("--patch-size", type=int, default=16)

    p.add_argument("--matrix", choices=["bt601", "bt709", "bt2020"], default="bt709")
    p.add_argument("--range", choices=["limited", "full"], default="limited", dest="value_range")
    p.add_argument(
        "--tenbit-shift-right",
        type=int,
        default=0,
        help="Use 0 for normal yuv420p10le. Use 6 only if your 10-bit samples are MSB-aligned in uint16.",
    )

    p.add_argument(
        "--output-depth-pix-fmt",
        choices=["yuv420p", "yuv420p10le"],
        default="yuv420p10le",
    )
    p.add_argument("--depth-quant-mode", choices=["linear", "inverse"], default="inverse")
    p.add_argument("--depth-norm", choices=["global", "per-frame", "fixed"], default="global")
    p.add_argument("--depth-min", type=float, default=None)
    p.add_argument("--depth-max", type=float, default=None)
    p.add_argument("--clip-percentile", nargs=2, type=float, default=(0.0, 100.0))
    p.add_argument("--fill-crop", choices=["edge", "zero"], default="edge")

    p.add_argument("--save-conf-yuv", action="store_true")
    p.add_argument("--save-model-input-rgb", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    pix_fmt: PixFmt = normalize_pix_fmt(args.pix_fmt)
    out_depth_pix_fmt: PixFmt = normalize_pix_fmt(args.output_depth_pix_fmt)

    if args.start < 0 or args.end < args.start:
        raise ValueError("Require 0 <= start <= end")

    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    if args.gop_size == 1:
        raise ValueError("--gop-size must be 0 or >= 2")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    all_frame_indices = list(range(args.start, args.end + 1, args.stride))

    rap_groups = make_rap_frame_groups(
        all_frame_indices,
        gop_size=args.gop_size,
        drop_incomplete_gop=args.drop_incomplete_gop,
    )

    if not rap_groups:
        raise RuntimeError("No RAP groups were generated")

    print(f"Total selected frames: {len(all_frame_indices)}")
    print(f"Number of RAP groups: {len(rap_groups)}")

    for rap_idx, group in enumerate(rap_groups):
        rap_name = f"{args.rap_name}{rap_idx}" if args.gop_size > 0 else args.rap_name
        print(f"  {rap_name}: {group[0]}..{group[-1]} ({len(group)} frames)")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. VGGT-Omega forward uses CUDA autocast in the released code.")

    print("Loading VGGT-Omega checkpoint")
    model = VGGTOmega().eval()
    load_checkpoint(model, args.checkpoint)
    model = model.to("cuda")

    rap_manifests = []

    for rap_idx, frame_indices in enumerate(rap_groups):
        if args.gop_size > 0:
            rap_name = f"{args.rap_name}{rap_idx}"
            rap_output_prefix = f"{args.output_prefix}_{rap_name}"
        else:
            rap_name = args.rap_name
            rap_output_prefix = args.output_prefix

        manifest = run_one_rap(
            args=args,
            model=model,
            pix_fmt=pix_fmt,
            out_depth_pix_fmt=out_depth_pix_fmt,
            frame_indices=frame_indices,
            rap_output_prefix=rap_output_prefix,
            rap_index=rap_idx,
            rap_name=rap_name,
        )

        rap_manifests.append(manifest)

    if args.gop_size > 0:
        all_manifest_path = args.output_prefix + "_all_rap_manifest.json"

        all_manifest = {
            "input_yuv": os.path.abspath(args.yuv),
            "width": args.width,
            "height": args.height,
            "pix_fmt": pix_fmt,
            "start": args.start,
            "end": args.end,
            "stride": args.stride,
            "gop_size": args.gop_size,
            "overlap_selected_frames": 1,
            "rap_name_prefix": args.rap_name,
            "num_raps": len(rap_manifests),
            "raps": rap_manifests,
            "notes": [
                "Each RAP is fed independently into VGGT-Omega.",
                "Adjacent RAPs overlap by one selected frame.",
                "For --gop-size 33 and --stride 1, RAP ranges are 0..32, 32..64, 64..96, ...",
            ],
        }

        with open(all_manifest_path, "w", encoding="utf-8") as f:
            json.dump(all_manifest, f, indent=2, ensure_ascii=False)

        print("All RAPs done")
        print(f"  all manifest: {all_manifest_path}")
    else:
        print("Done")


if __name__ == "__main__":
    main()
