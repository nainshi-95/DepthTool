#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_satd_refine_fixed_cam.py

GOP/frame-level depth refinement with fixed camera K, R|t.

- K, R|t are read from the same JSONL style as the projection/eval script.
- Reference/target Y frames are read from YUV420 sequence.
- Base depth is read from reconstructed depth YUV420p10le.
- Learnable low-resolution inverse-depth offset grid is optimized.
- Loss is multi-pair projection residual SATD + regularization.
- No OpenCV required. Uses PyTorch + NumPy only.

Typical use:
  python depth_satd_refine_fixed_cam.py \
    --seq-yuv recon_or_orig.yuv \
    --depth-yuv recon_depth.yuv \
    --param-jsonl cam_param.jsonl \
    --width 3840 --height 2160 --bit-depth 10 \
    --pairs 0:16,32:16 \
    --num-steps 300 \
    --offset-stride 64 \
    --max-delta-rho-ratio 0.01 \
    --out-depth-yuv refined_depth.yuv \
    --out-json refine_stats.json \
    --out-delta-npz delta_rho.npz

Notes:
  - This script optimizes target-frame depth only.
  - For pair ref:tar, the target depth D_tar is refined.
  - Reference frames are fixed.
  - If --target-yuv is omitted, --seq-yuv is used as both reference and target.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Basic utilities
# ============================================================

def align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def calc_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top):
    pad_right = coded_w - src_w - pad_left
    pad_bottom = coded_h - src_h - pad_top
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"Invalid padding: src={src_w}x{src_h}, coded={coded_w}x{coded_h}, "
            f"pad_left={pad_left}, pad_top={pad_top}"
        )
    return pad_right, pad_bottom


def validate_yuv420_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top, pad_right, pad_bottom):
    vals = {
        "src_w": src_w, "src_h": src_h,
        "coded_w": coded_w, "coded_h": coded_h,
        "pad_left": pad_left, "pad_top": pad_top,
        "pad_right": pad_right, "pad_bottom": pad_bottom,
    }
    for name, v in vals.items():
        if v < 0:
            raise ValueError(f"{name} must be non-negative: {v}")
        if v % 2 != 0:
            raise ValueError(f"{name} must be even for YUV420: {v}")


def active_slice(src_w, src_h, pad_left, pad_top):
    return slice(pad_top, pad_top + src_h), slice(pad_left, pad_left + src_w)


def pad_2d_edge_np(arr, coded_w, coded_h, pad_left, pad_top):
    h, w = arr.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top
    return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")


def crop_active_np(arr, src_w, src_h, pad_left, pad_top):
    return arr[pad_top:pad_top + src_h, pad_left:pad_left + src_w]


def calc_psnr_np(a, b, bit_depth, mask=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = a - b
    if mask is not None:
        m = mask.astype(bool)
        if not np.any(m):
            return float("nan")
        d = d[m]
    mse = float(np.mean(d * d))
    if mse == 0.0:
        return float("inf")
    maxv = (1 << bit_depth) - 1
    return 10.0 * math.log10((maxv * maxv) / mse)


def json_safe(x):
    if x is None:
        return None
    try:
        xf = float(x)
        if math.isnan(xf):
            return None
        if math.isinf(xf):
            return "inf" if xf > 0 else "-inf"
        return xf
    except Exception:
        return x


# ============================================================
# YUV420 IO
# ============================================================

def yuv_dtype(bit_depth: int):
    return np.uint8 if bit_depth <= 8 else np.dtype("<u2")


def frame_size_yuv420(w: int, h: int, bit_depth: int) -> int:
    bps = 1 if bit_depth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def count_frames_yuv420(path: str, w: int, h: int, bit_depth: int) -> int:
    fs = frame_size_yuv420(w, h, bit_depth)
    size = os.path.getsize(path)
    trailing = size % fs
    if trailing:
        print(f"[WARN] trailing bytes ignored: {path}, trailing={trailing}")
    return size // fs


def read_yuv420_frame(path: str, idx: int, w: int, h: int, bit_depth: int):
    dtype = yuv_dtype(bit_depth)
    y_size = w * h
    uv_size = (w // 2) * (h // 2)
    fs = frame_size_yuv420(w, h, bit_depth)

    with open(path, "rb") as f:
        f.seek(idx * fs)
        y = np.fromfile(f, dtype=dtype, count=y_size)
        u = np.fromfile(f, dtype=dtype, count=uv_size)
        v = np.fromfile(f, dtype=dtype, count=uv_size)

    if y.size != y_size or u.size != uv_size or v.size != uv_size:
        raise EOFError(f"Cannot read frame {idx} from {path}")

    return (
        y.reshape(h, w),
        u.reshape(h // 2, w // 2),
        v.reshape(h // 2, w // 2),
    )


def read_y_frame(path: str, idx: int, w: int, h: int, bit_depth: int):
    y, _, _ = read_yuv420_frame(path, idx, w, h, bit_depth)
    return y


def write_yuv420_frame(fp, y, u, v):
    fp.write(np.ascontiguousarray(y).tobytes())
    fp.write(np.ascontiguousarray(u).tobytes())
    fp.write(np.ascontiguousarray(v).tobytes())


def write_depth_yuv420p10_frame(fp, depth_y_10, maxv=1023):
    y = np.clip(np.rint(depth_y_10), 0, maxv).astype("<u2")
    h, w = y.shape
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    write_yuv420_frame(fp, y, uv, uv)


# ============================================================
# JSONL camera parsing
# ============================================================

def load_param_jsonl(path: str):
    header = None
    frames = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if obj.get("type") in ["header", "intrinsic"]:
                header = obj
            elif "poc" in obj:
                frames[int(obj["poc"])] = obj

    if header is None:
        raise RuntimeError("header line not found in param jsonl")
    if not frames:
        raise RuntimeError("no frame lines found in param jsonl")

    return header, frames


def get_depth_scale_real_from_header(header):
    if "depth_scale_precision" in header:
        precision = float(header["depth_scale_precision"])
        if precision <= 0:
            raise ValueError("depth_scale_precision must be positive")
        return float(header["depth_scale"]) / precision

    if "depth_scale_real" in header:
        return float(header["depth_scale_real"])

    if "depth_scale" not in header:
        raise RuntimeError("depth_scale not found in header")

    return float(header["depth_scale"])


def make_padded_intrinsic_from_original(intr, pad_left, pad_top):
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]) + float(pad_left),
        "cy": float(intr["cy"]) + float(pad_top),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }


def intrinsic_copy(intr):
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }


def add_intrinsic_delta(intr, delta):
    return {
        "fx": float(intr["fx"]) + float(delta[0]),
        "fy": float(intr["fy"]) + float(delta[1]),
        "cx": float(intr["cx"]) + float(delta[2]),
        "cy": float(intr["cy"]) + float(delta[3]),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }


def build_frame_intrinsics(header, frames, max_idx, pad_left, pad_top):
    intrs = {}

    if "intrinsic_dec_first" in header:
        intr0 = intrinsic_copy(header["intrinsic_dec_first"])
    elif "intrinsic_gt_padded0" in header:
        intr0 = intrinsic_copy(header["intrinsic_gt_padded0"])
    elif "intrinsic" in header:
        intr0 = make_padded_intrinsic_from_original(header["intrinsic"], pad_left, pad_top)
    else:
        raise RuntimeError("No intrinsic information found in JSONL header")

    intrs[0] = intr0

    for i in range(1, int(max_idx) + 1):
        f = frames.get(i, {})

        if "intrinsic_tar_dec" in f:
            intrs[i] = intrinsic_copy(f["intrinsic_tar_dec"])
        elif "intrinsic_dec" in f:
            intrs[i] = intrinsic_copy(f["intrinsic_dec"])
        elif "intrinsic" in f:
            intrs[i] = make_padded_intrinsic_from_original(f["intrinsic"], pad_left, pad_top)
        elif "intrinsic_delta" in f:
            delta = np.asarray(f["intrinsic_delta"], dtype=np.float64).reshape(4)
            intrs[i] = add_intrinsic_delta(intrs[i - 1], delta)
        else:
            intrs[i] = intrs[i - 1].copy()

    return intrs


# ============================================================
# Pose helpers
# ============================================================

def rodrigues_to_matrix_np(rvec):
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))

    if theta < 1e-12:
        K = np.array(
            [
                [0.0, -r[2], r[1]],
                [r[2], 0.0, -r[0]],
                [-r[1], r[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + K

    k = r / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def _as_extrinsic_matrix(frame):
    for key in ["extrinsic_abs", "extrinsic", "camera_extrinsic", "cam_from_world"]:
        if key in frame:
            E = np.asarray(frame[key], dtype=np.float64)
            if E.shape == (4, 4):
                return E[:3, :4]
            if E.shape == (3, 4):
                return E
            if E.size == 12:
                return E.reshape(3, 4)
            if E.size == 16:
                return E.reshape(4, 4)[:3, :4]

    if "rvec_abs" in frame and "tvec_abs" in frame:
        R = rodrigues_to_matrix_np(frame["rvec_abs"])
        t = np.asarray(frame["tvec_abs"], dtype=np.float64).reshape(3)
        return np.concatenate([R, t.reshape(3, 1)], axis=1)

    return None


def _frame_relative_rt_current_to_previous(frame):
    if "rt_dec" in frame:
        rt = frame["rt_dec"]
        rvec = rt["rvec"]
        tvec = rt["tvec"]
    elif "rvec" in frame and "tvec" in frame:
        rvec = frame["rvec"]
        tvec = frame["tvec"]
    else:
        raise RuntimeError("Frame has no relative pose. Expected rt_dec or rvec/tvec.")

    return rodrigues_to_matrix_np(rvec), np.asarray(tvec, dtype=np.float64).reshape(3)


def _invert_rt(R, t):
    Ri = R.T
    ti = -Ri @ t
    return Ri, ti


def _compose_rt(R2, t2, R1, t1):
    return R2 @ R1, R2 @ t1 + t2


def get_target_to_reference_rt(frames, ref_idx: int, tar_idx: int):
    ref_idx = int(ref_idx)
    tar_idx = int(tar_idx)

    if ref_idx == tar_idx:
        return {
            "R": np.eye(3, dtype=np.float64),
            "t": np.zeros(3, dtype=np.float64),
            "source": "identity",
        }

    f_ref = frames.get(ref_idx, {})
    f_tar = frames.get(tar_idx, {})

    E_ref = _as_extrinsic_matrix(f_ref)
    E_tar = _as_extrinsic_matrix(f_tar)

    if E_ref is not None and E_tar is not None:
        R_ref, t_ref = E_ref[:, :3], E_ref[:, 3]
        R_tar, t_tar = E_tar[:, :3], E_tar[:, 3]

        R = R_ref @ R_tar.T
        t = t_ref - R @ t_tar

        return {
            "R": R.astype(np.float64),
            "t": t.astype(np.float64),
            "source": "absolute_extrinsic",
        }

    # Fallback: compose current-to-previous relative transforms.
    if tar_idx > ref_idx:
        R_tot = np.eye(3, dtype=np.float64)
        t_tot = np.zeros(3, dtype=np.float64)

        for p in range(tar_idx, ref_idx, -1):
            if p not in frames:
                raise RuntimeError(f"Missing frame {p} for relative pose composition")
            R_p, t_p = _frame_relative_rt_current_to_previous(frames[p])
            R_tot, t_tot = _compose_rt(R_p, t_p, R_tot, t_tot)
    else:
        R_fwd = np.eye(3, dtype=np.float64)
        t_fwd = np.zeros(3, dtype=np.float64)

        for p in range(ref_idx, tar_idx, -1):
            if p not in frames:
                raise RuntimeError(f"Missing frame {p} for relative pose composition")
            R_p, t_p = _frame_relative_rt_current_to_previous(frames[p])
            R_fwd, t_fwd = _compose_rt(R_p, t_p, R_fwd, t_fwd)

        R_tot, t_tot = _invert_rt(R_fwd, t_fwd)

    return {
        "R": R_tot.astype(np.float64),
        "t": t_tot.astype(np.float64),
        "source": "composed_current_to_previous",
    }


# ============================================================
# Pair parsing
# ============================================================

def parse_pairs_string(s: str) -> List[Tuple[int, int]]:
    out = []
    s = str(s).strip()
    if not s:
        return out

    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad pair item: {item}. Expected ref:tar")
        a, b = item.split(":", 1)
        out.append((int(a), int(b)))

    return out


def load_pairs(args):
    pairs = []

    if args.pairs:
        pairs.extend(parse_pairs_string(args.pairs))

    if args.pairs_json:
        with open(args.pairs_json, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            obj = obj.get("pairs", [])

        for p in obj:
            if isinstance(p, dict):
                pairs.append((int(p["ref"]), int(p["tar"])))
            else:
                pairs.append((int(p[0]), int(p[1])))

    if not pairs:
        raise ValueError("No pairs provided. Use --pairs or --pairs-json.")

    return pairs


# ============================================================
# Torch projection and warp
# ============================================================

def torch_from_np_y(y_np, bit_depth, device):
    maxv = float((1 << bit_depth) - 1)
    t = torch.from_numpy(y_np.astype(np.float32) / maxv).to(device)
    return t


def pad_2d_edge_torch(x, coded_h, coded_w, pad_left, pad_top):
    # x: H,W
    h, w = x.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top
    if pad_right == 0 and pad_bottom == 0 and pad_left == 0 and pad_top == 0:
        return x
    x4 = x[None, None]
    y4 = F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return y4[0, 0]


def make_projection_precompute_torch(coded_w, coded_h, intr_tar, intr_ref, device):
    fx_t = float(intr_tar["fx"])
    fy_t = float(intr_tar["fy"])
    cx_t = float(intr_tar["cx"])
    cy_t = float(intr_tar["cy"])

    fx_r = float(intr_ref["fx"])
    fy_r = float(intr_ref["fy"])
    cx_r = float(intr_ref["cx"])
    cy_r = float(intr_ref["cy"])

    z_sign = float(intr_tar.get("z_sign", intr_ref.get("z_sign", 1.0)))

    y, x = torch.meshgrid(
        torch.arange(coded_h, dtype=torch.float32, device=device),
        torch.arange(coded_w, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_norm = (x - cx_t) / fx_t
    y_norm = (y - cy_t) / fy_t

    return {
        "coded_w": int(coded_w),
        "coded_h": int(coded_h),
        "fx_ref": fx_r,
        "fy_ref": fy_r,
        "cx_ref": cx_r,
        "cy_ref": cy_r,
        "z_sign": z_sign,
        "x_norm": x_norm,
        "y_norm": y_norm,
    }


def backward_map_torch(depth_linear, precomp, rt):
    """
    depth_linear: [H,W], target frame depth in linear units.
    Returns:
      map_x, map_y, valid
    """
    x_norm = precomp["x_norm"]
    y_norm = precomp["y_norm"]

    fx = float(precomp["fx_ref"])
    fy = float(precomp["fy_ref"])
    cx = float(precomp["cx_ref"])
    cy = float(precomp["cy_ref"])
    z_sign = float(precomp["z_sign"])
    coded_w = int(precomp["coded_w"])
    coded_h = int(precomp["coded_h"])

    R = torch.as_tensor(rt["R"], dtype=torch.float32, device=depth_linear.device).reshape(3, 3)
    t = torch.as_tensor(rt["t"], dtype=torch.float32, device=depth_linear.device).reshape(3)

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
        & (map_x <= coded_w - 1)
        & (map_y >= 0.0)
        & (map_y <= coded_h - 1)
    )

    return map_x, map_y, valid


def warp_y_torch(ref_y_norm, map_x, map_y, valid, invalid_fill="zero"):
    """
    ref_y_norm: [H,W], normalized [0,1]
    map_x/map_y: target pixel -> reference pixel coordinate
    """
    h, w = ref_y_norm.shape

    gx = 2.0 * map_x / max(w - 1, 1) - 1.0
    gy = 2.0 * map_y / max(h - 1, 1) - 1.0

    grid = torch.stack([gx, gy], dim=-1)[None]  # [1,H,W,2]
    src = ref_y_norm[None, None]               # [1,1,H,W]

    out = F.grid_sample(
        src,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]

    if invalid_fill == "zero":
        out = torch.where(valid, out, torch.zeros_like(out))
    elif invalid_fill == "target":
        # handled outside if needed
        pass
    else:
        raise ValueError(f"Unsupported invalid_fill: {invalid_fill}")

    return out


# ============================================================
# SATD and regularization
# ============================================================

def hadamard4(device):
    H = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return H


def satd4x4_loss(residual, valid=None, active_region=None, reduction="mean"):
    """
    residual: [H,W], normalized pixel residual.
    valid: optional bool [H,W].
    active_region: optional (ys, xs) slices.
    """
    if active_region is not None:
        ys, xs = active_region
        residual = residual[ys, xs]
        if valid is not None:
            valid = valid[ys, xs]

    h, w = residual.shape
    h4 = (h // 4) * 4
    w4 = (w // 4) * 4

    residual = residual[:h4, :w4]

    if valid is not None:
        valid = valid[:h4, :w4]
        vb = valid.reshape(h4 // 4, 4, w4 // 4, 4).permute(0, 2, 1, 3)
        block_ok = torch.all(vb, dim=(2, 3))
    else:
        block_ok = None

    rb = residual.reshape(h4 // 4, 4, w4 // 4, 4).permute(0, 2, 1, 3)  # [Bh,Bw,4,4]

    Hm = hadamard4(residual.device)
    coeff = torch.einsum("ij,bwjk,kl->bwil", Hm, rb, Hm.t())
    satd = torch.sum(torch.abs(coeff), dim=(2, 3)) * 0.5

    if block_ok is not None:
        satd = satd[block_ok]

    if satd.numel() == 0:
        return residual.new_tensor(0.0), 0

    if reduction == "sum":
        return torch.sum(satd), int(satd.numel())

    return torch.mean(satd), int(satd.numel())


def tv_l1(x):
    """
    x: [T,1,H,W] or [1,1,H,W]
    """
    loss = x.new_tensor(0.0)

    if x.shape[-1] > 1:
        loss = loss + torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
    if x.shape[-2] > 1:
        loss = loss + torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :]))

    return loss


def temporal_tv_l1(x, active_frame_mask=None):
    """
    x: [T,1,H,W]
    """
    if x.shape[0] <= 1:
        return x.new_tensor(0.0)

    diff = torch.abs(x[1:] - x[:-1])

    if active_frame_mask is not None:
        # Use only adjacent pairs where both frames are active.
        m0 = active_frame_mask[:-1]
        m1 = active_frame_mask[1:]
        m = (m0 & m1).float().view(-1, 1, 1, 1)
        if torch.sum(m) <= 0:
            return x.new_tensor(0.0)
        return torch.sum(diff * m) / torch.clamp(torch.sum(m) * diff.shape[1] * diff.shape[2] * diff.shape[3], min=1.0)

    return torch.mean(diff)


# ============================================================
# Depth model
# ============================================================

class LowResInverseDepthOffset(torch.nn.Module):
    def __init__(
        self,
        num_frames: int,
        coded_h: int,
        coded_w: int,
        offset_stride: int,
        max_delta_rho: float,
        device,
    ):
        super().__init__()

        self.num_frames = int(num_frames)
        self.coded_h = int(coded_h)
        self.coded_w = int(coded_w)
        self.offset_stride = int(offset_stride)
        self.max_delta_rho = float(max_delta_rho)

        h_lr = max(1, math.ceil(coded_h / offset_stride))
        w_lr = max(1, math.ceil(coded_w / offset_stride))

        self.raw = torch.nn.Parameter(torch.zeros(num_frames, 1, h_lr, w_lr, dtype=torch.float32, device=device))

    def delta_full(self, frame_idx: int):
        raw_one = self.raw[int(frame_idx):int(frame_idx) + 1]
        delta_lr = torch.tanh(raw_one) * self.max_delta_rho

        delta = F.interpolate(
            delta_lr,
            size=(self.coded_h, self.coded_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return delta

    def all_delta_lr(self):
        return torch.tanh(self.raw) * self.max_delta_rho


def compute_refined_depth_linear(
    base_depth_y_np,
    depth_scale_real,
    model,
    frame_idx,
    args,
    device,
):
    """
    base_depth_y_np: source-size depth sample Y, uint16 or float.
    Returns:
      refined depth linear padded tensor [coded_h,coded_w]
      base depth linear padded tensor [coded_h,coded_w]
      delta rho full tensor [coded_h,coded_w]
    """
    depth_y = torch.from_numpy(base_depth_y_np.astype(np.float32)).to(device)
    depth_linear = depth_y * float(depth_scale_real)

    depth_linear_pad = pad_2d_edge_torch(
        depth_linear,
        args.coded_height_resolved,
        args.coded_width_resolved,
        args.pad_left,
        args.pad_top,
    )

    depth_linear_pad = torch.clamp(depth_linear_pad, min=float(args.depth_min_linear))
    rho_base = 1.0 / depth_linear_pad

    delta = model.delta_full(frame_idx)
    rho_refined = rho_base + delta

    rho_min = 1.0 / max(float(args.depth_max_linear), 1e-8)
    rho_max = 1.0 / max(float(args.depth_min_linear), 1e-8)
    rho_refined = torch.clamp(rho_refined, min=rho_min, max=rho_max)

    depth_refined = 1.0 / rho_refined

    return depth_refined, depth_linear_pad, delta


# ============================================================
# Cache
# ============================================================

class FrameCache:
    def __init__(self, seq_yuv, target_yuv, depth_yuv, width, height, bit_depth):
        self.seq_yuv = str(seq_yuv)
        self.target_yuv = str(target_yuv) if target_yuv else str(seq_yuv)
        self.depth_yuv = str(depth_yuv)
        self.width = int(width)
        self.height = int(height)
        self.bit_depth = int(bit_depth)

        self.ref_y_cache: Dict[int, np.ndarray] = {}
        self.tar_y_cache: Dict[int, np.ndarray] = {}
        self.depth_y_cache: Dict[int, np.ndarray] = {}

    def get_ref_y(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx not in self.ref_y_cache:
            y = read_y_frame(self.seq_yuv, frame_idx, self.width, self.height, self.bit_depth)
            self.ref_y_cache[frame_idx] = y
        return self.ref_y_cache[frame_idx]

    def get_tar_y(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx not in self.tar_y_cache:
            y = read_y_frame(self.target_yuv, frame_idx, self.width, self.height, self.bit_depth)
            self.tar_y_cache[frame_idx] = y
        return self.tar_y_cache[frame_idx]

    def get_depth_y(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx not in self.depth_y_cache:
            y = read_y_frame(self.depth_yuv, frame_idx, self.width, self.height, 10)
            self.depth_y_cache[frame_idx] = y
        return self.depth_y_cache[frame_idx]


# ============================================================
# Optional pseudo MV loading
# ============================================================

def load_pseudo_mv_for_pair(mv_dir: Optional[str], ref_idx: int, tar_idx: int):
    """
    Optional regularizer.

    Expected file:
      {mv_dir}/mv_{ref}_{tar}.npz

    Supported keys:
      - map_x, map_y
      or
      - mv_x, mv_y

    Optional:
      - mask
    """
    if not mv_dir:
        return None

    path = Path(mv_dir) / f"mv_{ref_idx}_{tar_idx}.npz"
    if not path.exists():
        return None

    obj = np.load(path)

    if "map_x" in obj and "map_y" in obj:
        map_x = obj["map_x"].astype(np.float32)
        map_y = obj["map_y"].astype(np.float32)
    elif "mv_x" in obj and "mv_y" in obj:
        mv_x = obj["mv_x"].astype(np.float32)
        mv_y = obj["mv_y"].astype(np.float32)
        h, w = mv_x.shape
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        map_x = xx + mv_x
        map_y = yy + mv_y
    else:
        raise RuntimeError(f"{path} must contain map_x/map_y or mv_x/mv_y")

    mask = obj["mask"].astype(np.bool_) if "mask" in obj else np.ones_like(map_x, dtype=np.bool_)

    return {
        "map_x": map_x,
        "map_y": map_y,
        "mask": mask,
        "path": str(path),
    }


def pseudo_mv_loss_torch(map_x, map_y, pseudo, active_region, device):
    pmx = torch.from_numpy(pseudo["map_x"].astype(np.float32)).to(device)
    pmy = torch.from_numpy(pseudo["map_y"].astype(np.float32)).to(device)
    m = torch.from_numpy(pseudo["mask"].astype(np.bool_)).to(device)

    ys, xs = active_region
    map_x_a = map_x[ys, xs]
    map_y_a = map_y[ys, xs]
    pmx_a = pmx[ys, xs]
    pmy_a = pmy[ys, xs]
    m_a = m[ys, xs]

    valid = m_a & torch.isfinite(pmx_a) & torch.isfinite(pmy_a)

    if torch.count_nonzero(valid) < 16:
        return map_x.new_tensor(0.0)

    dx = map_x_a[valid] - pmx_a[valid]
    dy = map_y_a[valid] - pmy_a[valid]
    e = torch.sqrt(dx * dx + dy * dy + 1e-6)

    # Charbonnier-like robust loss
    return torch.mean(e)


# ============================================================
# One training loss pass
# ============================================================

def compute_pair_loss(
    pair,
    frame_cache,
    model,
    intrs,
    frames,
    depth_scale_real,
    args,
    device,
    precomp_cache,
    rt_cache,
    pseudo_cache,
):
    ref_idx, tar_idx = int(pair[0]), int(pair[1])

    ref_y_np = frame_cache.get_ref_y(ref_idx)
    tar_y_np = frame_cache.get_tar_y(tar_idx)
    depth_y_np = frame_cache.get_depth_y(tar_idx)

    ref_y = torch_from_np_y(ref_y_np, args.bit_depth, device)
    tar_y = torch_from_np_y(tar_y_np, args.bit_depth, device)

    ref_y_pad = pad_2d_edge_torch(
        ref_y,
        args.coded_height_resolved,
        args.coded_width_resolved,
        args.pad_left,
        args.pad_top,
    )

    tar_y_pad = pad_2d_edge_torch(
        tar_y,
        args.coded_height_resolved,
        args.coded_width_resolved,
        args.pad_left,
        args.pad_top,
    )

    depth_refined, depth_base, delta = compute_refined_depth_linear(
        base_depth_y_np=depth_y_np,
        depth_scale_real=depth_scale_real,
        model=model,
        frame_idx=tar_idx,
        args=args,
        device=device,
    )

    pc_key = (ref_idx, tar_idx)
    if pc_key not in precomp_cache:
        precomp_cache[pc_key] = make_projection_precompute_torch(
            args.coded_width_resolved,
            args.coded_height_resolved,
            intr_tar=intrs[tar_idx],
            intr_ref=intrs[ref_idx],
            device=device,
        )

    if pc_key not in rt_cache:
        rt_cache[pc_key] = get_target_to_reference_rt(frames, ref_idx, tar_idx)

    precomp = precomp_cache[pc_key]
    rt = rt_cache[pc_key]

    map_x, map_y, valid = backward_map_torch(depth_refined, precomp, rt)
    pred = warp_y_torch(ref_y_pad, map_x, map_y, valid, invalid_fill="zero")

    residual = tar_y_pad - pred

    active = active_slice(args.width, args.height, args.pad_left, args.pad_top)

    if args.satd_valid_only:
        valid_for_satd = valid
    else:
        valid_for_satd = None

    satd, satd_blocks = satd4x4_loss(
        residual=residual,
        valid=valid_for_satd,
        active_region=active,
        reduction=args.satd_reduction,
    )

    loss = args.lambda_satd * satd

    # Optional pseudo MV geometry regularizer.
    mv_loss = residual.new_tensor(0.0)
    if args.pseudo_mv_dir and args.lambda_pseudo_mv > 0.0:
        if pc_key not in pseudo_cache:
            pseudo_cache[pc_key] = load_pseudo_mv_for_pair(args.pseudo_mv_dir, ref_idx, tar_idx)

        pseudo = pseudo_cache[pc_key]
        if pseudo is not None:
            mv_loss = pseudo_mv_loss_torch(map_x, map_y, pseudo, active, device)
            loss = loss + args.lambda_pseudo_mv * mv_loss

    # Per-frame delta regularizers.
    delta_mag = torch.mean(torch.abs(delta))
    delta_tv = tv_l1(delta[None, None])

    loss = loss + args.lambda_delta_mag * delta_mag
    loss = loss + args.lambda_delta_tv * delta_tv

    with torch.no_grad():
        valid_ratio_active = float(torch.mean(valid[active].float()).item())
        pred_np = None
        tar_np = None

    stats = {
        "ref": ref_idx,
        "tar": tar_idx,
        "satd": float(satd.detach().cpu().item()),
        "satd_blocks": int(satd_blocks),
        "mv_loss": float(mv_loss.detach().cpu().item()),
        "delta_mag": float(delta_mag.detach().cpu().item()),
        "delta_tv": float(delta_tv.detach().cpu().item()),
        "valid_ratio_active": valid_ratio_active,
    }

    return loss, stats


def compute_full_loss(
    pairs,
    frame_cache,
    model,
    intrs,
    frames,
    depth_scale_real,
    args,
    device,
    precomp_cache,
    rt_cache,
    pseudo_cache,
    active_frame_mask,
):
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    pair_stats = []

    for pair in pairs:
        l, st = compute_pair_loss(
            pair=pair,
            frame_cache=frame_cache,
            model=model,
            intrs=intrs,
            frames=frames,
            depth_scale_real=depth_scale_real,
            args=args,
            device=device,
            precomp_cache=precomp_cache,
            rt_cache=rt_cache,
            pseudo_cache=pseudo_cache,
        )
        total_loss = total_loss + l
        pair_stats.append(st)

    # Temporal TV on low-res offset grid.
    if args.lambda_delta_temp > 0.0:
        delta_lr_all = model.all_delta_lr()
        temp = temporal_tv_l1(delta_lr_all, active_frame_mask=active_frame_mask)
        total_loss = total_loss + args.lambda_delta_temp * temp
        temp_val = float(temp.detach().cpu().item())
    else:
        temp_val = 0.0

    # Low-res spatial TV over entire model grid.
    if args.lambda_delta_lr_tv > 0.0:
        delta_lr_all = model.all_delta_lr()
        lr_tv = tv_l1(delta_lr_all)
        total_loss = total_loss + args.lambda_delta_lr_tv * lr_tv
        lr_tv_val = float(lr_tv.detach().cpu().item())
    else:
        lr_tv_val = 0.0

    return total_loss, pair_stats, {
        "delta_temp": temp_val,
        "delta_lr_tv": lr_tv_val,
    }


# ============================================================
# Evaluation and output
# ============================================================

@torch.no_grad()
def evaluate_pairs(
    pairs,
    frame_cache,
    model,
    intrs,
    frames,
    depth_scale_real,
    args,
    device,
):
    precomp_cache = {}
    rt_cache = {}
    pseudo_cache = {}
    out = []

    for pair in pairs:
        ref_idx, tar_idx = int(pair[0]), int(pair[1])
        ref_y_np = frame_cache.get_ref_y(ref_idx)
        tar_y_np = frame_cache.get_tar_y(tar_idx)
        depth_y_np = frame_cache.get_depth_y(tar_idx)

        ref_y = torch_from_np_y(ref_y_np, args.bit_depth, device)
        tar_y = torch_from_np_y(tar_y_np, args.bit_depth, device)

        ref_y_pad = pad_2d_edge_torch(ref_y, args.coded_height_resolved, args.coded_width_resolved, args.pad_left, args.pad_top)
        tar_y_pad = pad_2d_edge_torch(tar_y, args.coded_height_resolved, args.coded_width_resolved, args.pad_left, args.pad_top)

        pc_key = (ref_idx, tar_idx)

        precomp = make_projection_precompute_torch(
            args.coded_width_resolved,
            args.coded_height_resolved,
            intr_tar=intrs[tar_idx],
            intr_ref=intrs[ref_idx],
            device=device,
        )

        rt = get_target_to_reference_rt(frames, ref_idx, tar_idx)

        # Base depth eval
        depth_y_t = torch.from_numpy(depth_y_np.astype(np.float32)).to(device)
        depth_base = depth_y_t * float(depth_scale_real)
        depth_base_pad = pad_2d_edge_torch(
            depth_base,
            args.coded_height_resolved,
            args.coded_width_resolved,
            args.pad_left,
            args.pad_top,
        )
        depth_base_pad = torch.clamp(depth_base_pad, min=float(args.depth_min_linear))

        map_x_b, map_y_b, valid_b = backward_map_torch(depth_base_pad, precomp, rt)
        pred_b = warp_y_torch(ref_y_pad, map_x_b, map_y_b, valid_b, invalid_fill="zero")

        # Refined depth eval
        depth_refined, _, _ = compute_refined_depth_linear(
            base_depth_y_np=depth_y_np,
            depth_scale_real=depth_scale_real,
            model=model,
            frame_idx=tar_idx,
            args=args,
            device=device,
        )
        map_x_r, map_y_r, valid_r = backward_map_torch(depth_refined, precomp, rt)
        pred_r = warp_y_torch(ref_y_pad, map_x_r, map_y_r, valid_r, invalid_fill="zero")

        active = active_slice(args.width, args.height, args.pad_left, args.pad_top)

        satd_b, nb = satd4x4_loss(tar_y_pad - pred_b, valid_b if args.satd_valid_only else None, active, reduction="mean")
        satd_r, nr = satd4x4_loss(tar_y_pad - pred_r, valid_r if args.satd_valid_only else None, active, reduction="mean")

        pred_b_np = (pred_b.detach().cpu().numpy() * ((1 << args.bit_depth) - 1)).astype(np.float32)
        pred_r_np = (pred_r.detach().cpu().numpy() * ((1 << args.bit_depth) - 1)).astype(np.float32)
        tar_np = (tar_y_pad.detach().cpu().numpy() * ((1 << args.bit_depth) - 1)).astype(np.float32)

        active_np = np.zeros_like(tar_np, dtype=bool)
        active_np[active] = True

        valid_b_np = valid_b.detach().cpu().numpy().astype(bool)
        valid_r_np = valid_r.detach().cpu().numpy().astype(bool)

        out.append({
            "ref": ref_idx,
            "tar": tar_idx,
            "base_psnr_active": json_safe(calc_psnr_np(pred_b_np, tar_np, args.bit_depth, mask=active_np)),
            "refined_psnr_active": json_safe(calc_psnr_np(pred_r_np, tar_np, args.bit_depth, mask=active_np)),
            "base_psnr_valid_active": json_safe(calc_psnr_np(pred_b_np, tar_np, args.bit_depth, mask=active_np & valid_b_np)),
            "refined_psnr_valid_active": json_safe(calc_psnr_np(pred_r_np, tar_np, args.bit_depth, mask=active_np & valid_r_np)),
            "base_satd": float(satd_b.detach().cpu().item()),
            "refined_satd": float(satd_r.detach().cpu().item()),
            "base_valid_ratio_active": float(np.mean(valid_b_np[active])),
            "refined_valid_ratio_active": float(np.mean(valid_r_np[active])),
            "satd_blocks_base": int(nb),
            "satd_blocks_refined": int(nr),
        })

    return out


@torch.no_grad()
def write_refined_depth_yuv(
    out_path,
    depth_yuv,
    model,
    depth_scale_real,
    args,
    device,
):
    depth_frames = count_frames_yuv420(depth_yuv, args.width, args.height, 10)
    maxv = 1023

    with open(out_path, "wb") as fp:
        for fi in range(depth_frames):
            depth_y_np = read_y_frame(depth_yuv, fi, args.width, args.height, 10).astype(np.float32)

            if fi < model.num_frames:
                depth_refined, _, _ = compute_refined_depth_linear(
                    base_depth_y_np=depth_y_np,
                    depth_scale_real=depth_scale_real,
                    model=model,
                    frame_idx=fi,
                    args=args,
                    device=device,
                )

                depth_refined_np = crop_active_np(
                    depth_refined.detach().cpu().numpy(),
                    args.width,
                    args.height,
                    args.pad_left,
                    args.pad_top,
                )
                depth_y_out = depth_refined_np / float(depth_scale_real)
            else:
                depth_y_out = depth_y_np

            depth_y_out = np.clip(np.rint(depth_y_out), 0, maxv).astype(np.float32)
            write_depth_yuv420p10_frame(fp, depth_y_out, maxv=maxv)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seq-yuv", required=True, help="Reference YUV420 sequence. Usually codec recon/reference sequence.")
    p.add_argument("--target-yuv", default="", help="Target YUV420 sequence for loss. If omitted, --seq-yuv is used.")
    p.add_argument("--depth-yuv", required=True, help="Reconstructed/base depth YUV420p10le.")
    p.add_argument("--param-jsonl", required=True)

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--bit-depth", type=int, default=10)

    p.add_argument("--coded-width", type=int, default=0)
    p.add_argument("--coded-height", type=int, default=0)
    p.add_argument("--pad-left", type=int, default=0)
    p.add_argument("--pad-top", type=int, default=0)

    p.add_argument("--pairs", default="", help="Comma separated ref:tar pairs, e.g. 0:16,32:16")
    p.add_argument("--pairs-json", default="", help="JSON list or {'pairs':[[ref,tar],...]}")

    p.add_argument("--seq-start", type=int, default=0, help="Reserved. Pair indices are currently interpreted as direct frame indices.")

    p.add_argument("--offset-stride", type=int, default=64, help="Low-res offset grid stride in coded luma pixels.")
    p.add_argument("--max-delta-rho", type=float, default=0.0, help="Absolute max inverse-depth offset. If 0, derived from --max-delta-rho-ratio.")
    p.add_argument("--max-delta-rho-ratio", type=float, default=0.01, help="Max delta rho = ratio * robust rho range.")

    p.add_argument("--depth-min-linear", type=float, default=1e-6)
    p.add_argument("--depth-max-linear", type=float, default=0.0, help="If 0, uses 1023 * depth_scale_real.")

    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")

    p.add_argument("--lambda-satd", type=float, default=1.0)
    p.add_argument("--lambda-delta-mag", type=float, default=1.0)
    p.add_argument("--lambda-delta-tv", type=float, default=5.0)
    p.add_argument("--lambda-delta-lr-tv", type=float, default=0.0)
    p.add_argument("--lambda-delta-temp", type=float, default=1.0)

    p.add_argument("--satd-valid-only", action="store_true", help="Use only fully valid 4x4 blocks for SATD.")
    p.add_argument("--satd-reduction", choices=["mean", "sum"], default="mean")

    p.add_argument("--pseudo-mv-dir", default="", help="Optional dir containing mv_ref_tar.npz for pseudo MV regularizer.")
    p.add_argument("--lambda-pseudo-mv", type=float, default=0.0)

    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--out-depth-yuv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-delta-npz", default="")

    return p.parse_args()


def validate_and_resolve_args(args):
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.bit_depth <= 0:
        raise ValueError("bad bit depth")
    if args.offset_stride <= 0:
        raise ValueError("offset_stride must be positive")
    if args.num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if args.depth_min_linear <= 0:
        raise ValueError("depth_min_linear must be positive")

    coded_w = int(args.coded_width) if args.coded_width else align_to(args.width + args.pad_left, 4)
    coded_h = int(args.coded_height) if args.coded_height else align_to(args.height + args.pad_top, 4)

    pad_right, pad_bottom = calc_padding(
        args.width,
        args.height,
        coded_w,
        coded_h,
        args.pad_left,
        args.pad_top,
    )

    validate_yuv420_padding(
        args.width,
        args.height,
        coded_w,
        coded_h,
        args.pad_left,
        args.pad_top,
        pad_right,
        pad_bottom,
    )

    args.coded_width_resolved = coded_w
    args.coded_height_resolved = coded_h
    args.pad_right_resolved = pad_right
    args.pad_bottom_resolved = pad_bottom

    if args.target_yuv == "":
        args.target_yuv = args.seq_yuv

    return args


def derive_max_delta_rho(args, frame_cache, target_frames, depth_scale_real):
    vals = []

    for fi in sorted(target_frames):
        d_y = frame_cache.get_depth_y(fi).astype(np.float64)
        d = np.maximum(d_y * float(depth_scale_real), float(args.depth_min_linear))
        rho = 1.0 / d
        vals.append(rho.reshape(-1))

    if not vals:
        raise RuntimeError("No target frames for delta-rho range derivation")

    rho_all = np.concatenate(vals)
    p5 = float(np.percentile(rho_all, 5.0))
    p95 = float(np.percentile(rho_all, 95.0))
    robust_range = max(p95 - p5, 1e-12)

    return float(args.max_delta_rho_ratio) * robust_range, {
        "rho_p5": p5,
        "rho_p95": p95,
        "rho_robust_range": robust_range,
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    args = validate_and_resolve_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] device: {device}")

    pairs = load_pairs(args)
    max_pair_idx = max(max(r, t) for r, t in pairs)
    target_frames = sorted(set(t for _, t in pairs))

    header, frames = load_param_jsonl(args.param_jsonl)
    depth_scale_real = get_depth_scale_real_from_header(header)

    if args.depth_max_linear <= 0:
        args.depth_max_linear = 1023.0 * float(depth_scale_real)

    intrs = build_frame_intrinsics(
        header=header,
        frames=frames,
        max_idx=max(max_pair_idx, max(frames.keys())),
        pad_left=args.pad_left,
        pad_top=args.pad_top,
    )

    seq_count = count_frames_yuv420(args.seq_yuv, args.width, args.height, args.bit_depth)
    target_count = count_frames_yuv420(args.target_yuv, args.width, args.height, args.bit_depth)
    depth_count = count_frames_yuv420(args.depth_yuv, args.width, args.height, 10)

    for r, t in pairs:
        if r < 0 or r >= seq_count:
            raise RuntimeError(f"ref frame {r} outside seq count {seq_count}")
        if t < 0 or t >= target_count:
            raise RuntimeError(f"target frame {t} outside target count {target_count}")
        if t < 0 or t >= depth_count:
            raise RuntimeError(f"target depth frame {t} outside depth count {depth_count}")

    frame_cache = FrameCache(
        seq_yuv=args.seq_yuv,
        target_yuv=args.target_yuv,
        depth_yuv=args.depth_yuv,
        width=args.width,
        height=args.height,
        bit_depth=args.bit_depth,
    )

    if args.max_delta_rho > 0:
        max_delta_rho = float(args.max_delta_rho)
        rho_stats = {"mode": "absolute", "max_delta_rho": max_delta_rho}
    else:
        max_delta_rho, rho_stats = derive_max_delta_rho(args, frame_cache, target_frames, depth_scale_real)
        rho_stats["mode"] = "ratio"
        rho_stats["max_delta_rho"] = max_delta_rho
        rho_stats["max_delta_rho_ratio"] = float(args.max_delta_rho_ratio)

    print(f"[INFO] pairs: {pairs}")
    print(f"[INFO] target frames: {target_frames}")
    print(f"[INFO] depth_scale_real: {depth_scale_real}")
    print(f"[INFO] depth linear range: [{args.depth_min_linear}, {args.depth_max_linear}]")
    print(f"[INFO] max_delta_rho: {max_delta_rho:.8e}")
    print(f"[INFO] coded size: {args.coded_width_resolved}x{args.coded_height_resolved}")
    print(f"[INFO] offset stride: {args.offset_stride}")

    # Use frame indices directly, so model num frames must cover max frame index.
    model_num_frames = max(depth_count, max_pair_idx + 1)

    model = LowResInverseDepthOffset(
        num_frames=model_num_frames,
        coded_h=args.coded_height_resolved,
        coded_w=args.coded_width_resolved,
        offset_stride=args.offset_stride,
        max_delta_rho=max_delta_rho,
        device=device,
    ).to(device)

    # Freeze non-target frames by gradient masking.
    active_frame_mask_np = np.zeros(model_num_frames, dtype=np.bool_)
    for fi in target_frames:
        active_frame_mask_np[int(fi)] = True
    active_frame_mask = torch.from_numpy(active_frame_mask_np).to(device)

    def mask_grad_hook(grad):
        m = active_frame_mask.float().view(-1, 1, 1, 1)
        return grad * m

    model.raw.register_hook(mask_grad_hook)

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    precomp_cache = {}
    rt_cache = {}
    pseudo_cache = {}

    stats_history = []

    print("[INFO] Initial evaluation...")
    initial_eval = evaluate_pairs(
        pairs=pairs,
        frame_cache=frame_cache,
        model=model,
        intrs=intrs,
        frames=frames,
        depth_scale_real=depth_scale_real,
        args=args,
        device=device,
    )
    for e in initial_eval:
        print(
            f"  pair {e['ref']}->{e['tar']} | "
            f"base SATD={e['base_satd']:.6f}, refined SATD={e['refined_satd']:.6f}, "
            f"base PSNR={e['base_psnr_active']}, refined PSNR={e['refined_psnr_active']}"
        )

    for step in range(args.num_steps):
        opt.zero_grad(set_to_none=True)

        loss, pair_stats, extra_stats = compute_full_loss(
            pairs=pairs,
            frame_cache=frame_cache,
            model=model,
            intrs=intrs,
            frames=frames,
            depth_scale_real=depth_scale_real,
            args=args,
            device=device,
            precomp_cache=precomp_cache,
            rt_cache=rt_cache,
            pseudo_cache=pseudo_cache,
            active_frame_mask=active_frame_mask,
        )

        loss.backward()
        opt.step()

        st = {
            "step": int(step),
            "loss": float(loss.detach().cpu().item()),
            "pair_stats": pair_stats,
            **extra_stats,
        }
        stats_history.append(st)

        if step % args.log_interval == 0 or step == args.num_steps - 1:
            satd_avg = float(np.mean([p["satd"] for p in pair_stats])) if pair_stats else 0.0
            valid_avg = float(np.mean([p["valid_ratio_active"] for p in pair_stats])) if pair_stats else 0.0
            dmag_avg = float(np.mean([p["delta_mag"] for p in pair_stats])) if pair_stats else 0.0
            dtv_avg = float(np.mean([p["delta_tv"] for p in pair_stats])) if pair_stats else 0.0

            print(
                f"[{step:04d}/{args.num_steps}] "
                f"loss={st['loss']:.6f} "
                f"satd={satd_avg:.6f} "
                f"valid={valid_avg:.4f} "
                f"dmag={dmag_avg:.3e} "
                f"dtv={dtv_avg:.3e} "
                f"temp={extra_stats['delta_temp']:.3e}"
            )

    print("[INFO] Final evaluation...")
    final_eval = evaluate_pairs(
        pairs=pairs,
        frame_cache=frame_cache,
        model=model,
        intrs=intrs,
        frames=frames,
        depth_scale_real=depth_scale_real,
        args=args,
        device=device,
    )

    for e in final_eval:
        print(
            f"  pair {e['ref']}->{e['tar']} | "
            f"base SATD={e['base_satd']:.6f}, refined SATD={e['refined_satd']:.6f}, "
            f"base PSNR={e['base_psnr_active']}, refined PSNR={e['refined_psnr_active']}, "
            f"valid {e['base_valid_ratio_active']:.4f}->{e['refined_valid_ratio_active']:.4f}"
        )

    print(f"[INFO] Writing refined depth YUV: {args.out_depth_yuv}")
    write_refined_depth_yuv(
        out_path=args.out_depth_yuv,
        depth_yuv=args.depth_yuv,
        model=model,
        depth_scale_real=depth_scale_real,
        args=args,
        device=device,
    )

    if args.out_delta_npz:
        delta_lr = model.all_delta_lr().detach().cpu().numpy()
        np.savez_compressed(
            args.out_delta_npz,
            delta_rho_lr=delta_lr,
            active_frame_mask=active_frame_mask_np,
            max_delta_rho=np.array([max_delta_rho], dtype=np.float64),
            offset_stride=np.array([args.offset_stride], dtype=np.int32),
            pairs=np.asarray(pairs, dtype=np.int32),
        )
        print(f"[INFO] Wrote delta NPZ: {args.out_delta_npz}")

    summary = {
        "args": {
            k: json_safe(v)
            for k, v in vars(args).items()
            if not k.endswith("_resolved")
        },
        "resolved": {
            "coded_width": args.coded_width_resolved,
            "coded_height": args.coded_height_resolved,
            "pad_right": args.pad_right_resolved,
            "pad_bottom": args.pad_bottom_resolved,
        },
        "pairs": [{"ref": int(r), "tar": int(t)} for r, t in pairs],
        "target_frames": [int(x) for x in target_frames],
        "depth_scale_real": depth_scale_real,
        "rho_stats": rho_stats,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history_tail": stats_history[-20:],
        "outputs": {
            "out_depth_yuv": args.out_depth_yuv,
            "out_delta_npz": args.out_delta_npz,
            "out_json": args.out_json,
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[INFO] Wrote JSON: {args.out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
