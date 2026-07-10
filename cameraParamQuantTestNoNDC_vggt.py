#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-parameter quantization and backward-warp simulation.

Main change from the per-pixel-depth version:
  - Before projection, the decoded current depth is approximated independently
    in each block by an inverse-depth plane

        1 / z(x, y) = a * (x - cx) + b * (y - cy) + c

  - The default block size is 16x16.
  - The plane-reconstructed z is then used by the same pixel-coordinate camera
    projection.
  - If an a/b/c fit is underdetermined or predicts a non-positive inverse depth,
    the block falls back to the least-squares constant inverse-depth model.

Projection convention:
  target current pixel/depth -> current camera 3D -> reference camera ->
  reference pixel coordinate.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

def align_to(x, a):
    return ((x + a - 1) // a) * a


def calc_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top):
    pad_right = coded_w - src_w - pad_left
    pad_bottom = coded_h - src_h - pad_top

    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"Invalid padding: src=({src_w}x{src_h}), "
            f"coded=({coded_w}x{coded_h}), "
            f"pad_left={pad_left}, pad_top={pad_top}"
        )

    return pad_right, pad_bottom


def validate_yuv420_padding(
    src_w,
    src_h,
    coded_w,
    coded_h,
    pad_left,
    pad_top,
    pad_right,
    pad_bottom,
):
    vals = {
        "src_w": src_w,
        "src_h": src_h,
        "coded_w": coded_w,
        "coded_h": coded_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }

    for name, v in vals.items():
        if v < 0:
            raise ValueError(f"{name} must be non-negative: {v}")

    for name, v in vals.items():
        if v % 2 != 0:
            raise ValueError(f"{name} must be even for YUV420: {v}")


def pad_2d_edge(arr, coded_w, coded_h, pad_left, pad_top):
    h, w = arr.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top

    return np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )


def pad_yuv420_edge(y, u, v, coded_w, coded_h, pad_left, pad_top):
    y_pad = pad_2d_edge(y, coded_w, coded_h, pad_left, pad_top)

    u_pad = pad_2d_edge(
        u,
        coded_w // 2,
        coded_h // 2,
        pad_left // 2,
        pad_top // 2,
    )

    v_pad = pad_2d_edge(
        v,
        coded_w // 2,
        coded_h // 2,
        pad_left // 2,
        pad_top // 2,
    )

    return y_pad, u_pad, v_pad


def active_slice(src_w, src_h, pad_left, pad_top):
    return (
        slice(pad_top, pad_top + src_h),
        slice(pad_left, pad_left + src_w),
    )


def calc_psnr(a, b, bit_depth):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))

    if mse == 0.0:
        return float("inf")

    maxv = (1 << bit_depth) - 1
    return 10.0 * np.log10((maxv * maxv) / mse)


def calc_float_metrics(a, b, valid=None, peak=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)

    if not np.any(mask):
        return {
            "valid_count": 0,
            "mae": None,
            "rmse": None,
            "psnr": None,
            "max_abs_err": None,
        }

    d = b[mask] - a[mask]
    mse = float(np.mean(d * d))
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(d)))

    if peak is None:
        peak = float(np.max(a[mask]))
    peak = max(float(peak), 1e-12)
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10((peak * peak) / mse)

    return {
        "valid_count": int(np.count_nonzero(mask)),
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "max_abs_err": max_abs,
    }


def json_safe_float(x):
    if x is None:
        return None
    x = float(x)
    if np.isinf(x):
        return "inf"
    if np.isnan(x):
        return None
    return x


# ============================================================
# YUV
# ============================================================

def frame_size_yuv420(w, h, bit_depth):
    bps = 1 if bit_depth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def count_frames(path, w, h, bit_depth):
    return os.path.getsize(path) // frame_size_yuv420(w, h, bit_depth)


def yuv_dtype(bit_depth):
    return np.uint8 if bit_depth <= 8 else np.dtype("<u2")


def read_yuv420(path, idx, w, h, bit_depth):
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
        raise RuntimeError(f"Cannot read frame {idx}: {path}")

    return (
        y.reshape(h, w),
        u.reshape(h // 2, w // 2),
        v.reshape(h // 2, w // 2),
    )


def write_yuv420(path, y, u, v):
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(y).tobytes())
        f.write(np.ascontiguousarray(u).tobytes())
        f.write(np.ascontiguousarray(v).tobytes())


def write_depth_linear_as_yuv420p10le(path, depth_linear, depth_scale_real):
    if depth_scale_real <= 0.0:
        raise ValueError("depth_scale_real must be positive")

    h, w = depth_linear.shape
    y = np.clip(
        np.rint(np.asarray(depth_linear, dtype=np.float64) / depth_scale_real),
        0,
        1023,
    ).astype("<u2")
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    write_yuv420(path, y, uv, uv)


# ============================================================
# Param JSONL
# ============================================================

def load_param_jsonl(path):
    header = None
    frames = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            if obj.get("type") in ["header", "intrinsic"]:
                header = obj
            elif "poc" in obj:
                frames[int(obj["poc"])] = obj

    if header is None:
        raise RuntimeError("header line not found in param jsonl")
    if "depth_scale" not in header:
        raise RuntimeError("depth_scale not found in header")
    if "intrinsic" not in header:
        raise RuntimeError("intrinsic not found in header")
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

    return float(header["depth_scale"])


def get_depth_scale_precision_from_header(header):
    if "depth_scale_precision" in header:
        return int(header["depth_scale_precision"])
    return None


# ============================================================
# Quantization
# ============================================================

def quant_u(value, lo, hi, bits):
    qmax = (1 << bits) - 1
    q = np.round((value - lo) / (hi - lo) * qmax)
    q = np.clip(q, 0, qmax).astype(np.int32)
    dec = q.astype(np.float32) / qmax * (hi - lo) + lo
    clipped = (value < lo) | (value > hi)
    return q, dec, clipped


def signed_q_abs_max(bits):
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


def quant_s(value, step, bits):
    q_abs_max = signed_q_abs_max(bits)
    qmin = -q_abs_max
    qmax = q_abs_max
    q = np.round(value / step)
    clipped = (q < qmin) | (q > qmax)
    q = np.clip(q, qmin, qmax).astype(np.int32)
    dec = q.astype(np.float32) * step
    return q, dec, clipped


def make_padded_intrinsic_from_original(intr, pad_left, pad_top):
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]) + float(pad_left),
        "cy": float(intr["cy"]) + float(pad_top),
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


def quantize_intrinsic_16(intr, w, h, f_max=4.0, c_min=-1.0, c_max=2.0):
    fx_n = intr["fx"] / w
    fy_n = intr["fy"] / h
    cx_n = intr["cx"] / w
    cy_n = intr["cy"] / h

    q_fx, d_fx, c_fx = quant_u(fx_n, -f_max, f_max, 16)
    q_fy, d_fy, c_fy = quant_u(fy_n, -f_max, f_max, 16)
    q_cx, d_cx, c_cx = quant_u(cx_n, c_min, c_max, 16)
    q_cy, d_cy, c_cy = quant_u(cy_n, c_min, c_max, 16)

    intr_dec = {
        "fx": float(d_fx * w),
        "fy": float(d_fy * h),
        "cx": float(d_cx * w),
        "cy": float(d_cy * h),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }

    intr_q = {
        "fx": int(q_fx),
        "fy": int(q_fy),
        "cx": int(q_cx),
        "cy": int(q_cy),
    }

    clipped = bool(c_fx or c_fy or c_cx or c_cy)
    return intr_q, intr_dec, clipped


def quantize_intrinsic_delta_4(delta, step, bits):
    delta = np.asarray(delta, dtype=np.float32).reshape(4)
    q, dec, clipped = quant_s(delta, step=step, bits=bits)
    return q.astype(np.int32), dec.astype(np.float32), clipped


def param6_from_frame(frame, depth_scale_real):
    r = np.array(frame["rvec"], dtype=np.float32)
    t = np.array(frame["tvec"], dtype=np.float32) / float(depth_scale_real)
    return np.concatenate([r, t], axis=0)


def rt_from_param6(p, depth_scale_real):
    return {
        "rvec": p[:3].astype(float).tolist(),
        "tvec": (p[3:] * float(depth_scale_real)).astype(float).tolist(),
    }


# ============================================================
# Signed truncated Exp-Golomb bit count
# ============================================================

def signed_to_code_num(x):
    x = int(x)
    if x == 0:
        return 0
    return 2 * x - 1 if x > 0 else -2 * x


def ue_exp_golomb_bits(code_num):
    code_num = int(code_num)
    if code_num < 0:
        raise ValueError("code_num must be non-negative")
    k = (code_num + 1).bit_length() - 1
    return 2 * k + 1


def signed_truncated_exp_golomb_bits(x, q_abs_max):
    x = int(x)

    if x < -q_abs_max or x > q_abs_max:
        raise ValueError(
            f"x={x} outside signed truncated range [-{q_abs_max}, {q_abs_max}]"
        )

    code_num = signed_to_code_num(x)
    max_code_num = 2 * q_abs_max

    if code_num > max_code_num:
        raise ValueError(
            f"code_num={code_num} outside truncated range [0, {max_code_num}]"
        )

    return ue_exp_golomb_bits(code_num)


def q_residual_bits_signed_trunc_exp_golomb(q_residual, q_abs_max):
    bits_each = [
        signed_truncated_exp_golomb_bits(int(v), q_abs_max)
        for v in q_residual
    ]
    return bits_each, int(sum(bits_each))


# ============================================================
# Predictor
# ============================================================

def predict_from_history(decoded_hist, pred_n, pred_degree):
    if not decoded_hist:
        return np.zeros(6, dtype=np.float32)

    m = min(len(decoded_hist), pred_n)
    y = np.stack(decoded_hist[-m:], axis=0).astype(np.float32)

    if m == 1:
        return y[-1].copy()

    deg = min(pred_degree, m - 1)
    x = np.arange(m, dtype=np.float32)
    x_next = np.array([m], dtype=np.float32)
    a = np.vander(x, N=deg + 1, increasing=True)
    b = np.vander(x_next, N=deg + 1, increasing=True)
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    pred = b @ coef
    return pred.reshape(6).astype(np.float32)


# ============================================================
# 16x16 inverse-depth block plane model
# ============================================================

def fit_inverse_depth_plane_block(depth_block, min_depth, min_valid_samples):
    """Fit inv-z = a*x_local + b*y_local + c for one block.

    Returns:
      reconstructed_depth, valid_mask, [a,b,c], used_constant_fallback

    The original finite positive-depth mask is retained. Projection therefore
    does not suddenly treat an originally invalid depth sample as valid.
    """
    z = np.asarray(depth_block, dtype=np.float64)
    h, w = z.shape

    valid = np.isfinite(z) & (z > float(min_depth))
    n_valid = int(np.count_nonzero(valid))

    recon = np.zeros((h, w), dtype=np.float64)
    if n_valid == 0:
        return recon, valid, np.array([0.0, 0.0, 0.0]), True

    xs = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    ys = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
    xx, yy = np.meshgrid(xs, ys)
    inv_z = 1.0 / z[valid]
    constant_c = float(np.mean(inv_z))

    use_constant = n_valid < max(3, int(min_valid_samples))
    coeff = np.array([0.0, 0.0, constant_c], dtype=np.float64)

    if not use_constant:
        design = np.stack(
            [xx[valid], yy[valid], np.ones(n_valid, dtype=np.float64)],
            axis=1,
        )
        try:
            candidate, _, rank, _ = np.linalg.lstsq(design, inv_z, rcond=None)
            if rank >= 3 and np.isfinite(candidate).all():
                coeff = candidate.astype(np.float64)
            else:
                use_constant = True
        except np.linalg.LinAlgError:
            use_constant = True

    pred_inv = coeff[0] * xx + coeff[1] * yy + coeff[2]

    # A valid depth plane requires strictly positive inverse depth over every
    # sample that will be projected. Fall back to the LS constant model rather
    # than clipping the plane nonlinearly.
    if (
        use_constant
        or not np.isfinite(pred_inv[valid]).all()
        or np.any(pred_inv[valid] <= 0.0)
    ):
        coeff = np.array([0.0, 0.0, constant_c], dtype=np.float64)
        pred_inv = np.full((h, w), constant_c, dtype=np.float64)
        use_constant = True

    recon[valid] = 1.0 / pred_inv[valid]
    return recon, valid, coeff, use_constant


def model_depth_with_inverse_planes(
    depth_linear,
    block_size=16,
    min_depth=1e-8,
    min_valid_samples=3,
):
    """Approximate a depth frame with independent inverse-depth block planes."""
    z = np.asarray(depth_linear, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"depth_linear must be 2D, got {z.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    h, w = z.shape
    recon = np.zeros_like(z, dtype=np.float64)
    valid_out = np.zeros_like(z, dtype=bool)

    blocks = []
    num_blocks = 0
    num_abc_blocks = 0
    num_constant_blocks = 0
    num_empty_blocks = 0

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            bh = min(block_size, h - by)
            bw = min(block_size, w - bx)
            block = z[by : by + bh, bx : bx + bw]

            block_recon, block_valid, coeff, constant_fallback = (
                fit_inverse_depth_plane_block(
                    block,
                    min_depth=min_depth,
                    min_valid_samples=min_valid_samples,
                )
            )

            recon[by : by + bh, bx : bx + bw] = block_recon
            valid_out[by : by + bh, bx : bx + bw] = block_valid

            valid_count = int(np.count_nonzero(block_valid))
            num_blocks += 1

            if valid_count == 0:
                num_empty_blocks += 1
            elif constant_fallback:
                num_constant_blocks += 1
            else:
                num_abc_blocks += 1

            blocks.append(
                {
                    "x": int(bx),
                    "y": int(by),
                    "w": int(bw),
                    "h": int(bh),
                    "valid_count": valid_count,
                    "a": float(coeff[0]),
                    "b": float(coeff[1]),
                    "c": float(coeff[2]),
                    "constant_fallback": bool(constant_fallback),
                }
            )

    metrics = calc_float_metrics(z, recon, valid=valid_out)
    summary = {
        "model": "block_inverse_depth_plane",
        "equation": "1/z = a*(x-cx) + b*(y-cy) + c",
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "num_abc_blocks": int(num_abc_blocks),
        "num_constant_blocks": int(num_constant_blocks),
        "num_empty_blocks": int(num_empty_blocks),
        "abc_ratio": float(num_abc_blocks / max(num_blocks, 1)),
        "constant_ratio": float(num_constant_blocks / max(num_blocks, 1)),
        "valid_ratio": float(np.mean(valid_out)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "max_abs_err": metrics["max_abs_err"],
    }

    return recon.astype(np.float32), valid_out, summary, blocks


# ============================================================
# Fast pixel-coordinate projection
# ============================================================

def make_projection_precompute_dual(w, h, intr_tar, intr_ref):
    """Target unprojection uses intr_tar; reference projection uses intr_ref."""
    fx_t = float(intr_tar["fx"])
    fy_t = float(intr_tar["fy"])
    cx_t = float(intr_tar["cx"])
    cy_t = float(intr_tar["cy"])

    fx_r = float(intr_ref["fx"])
    fy_r = float(intr_ref["fy"])
    cx_r = float(intr_ref["cx"])
    cy_r = float(intr_ref["cy"])

    z_sign = float(intr_tar.get("z_sign", intr_ref.get("z_sign", 1.0)))

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    x_norm = (x - cx_t) / fx_t
    y_norm = (y - cy_t) / fy_t

    return {
        "w": int(w),
        "h": int(h),
        "fx_ref": fx_r,
        "fy_ref": fy_r,
        "cx_ref": cx_r,
        "cy_ref": cy_r,
        "z_sign": z_sign,
        "x_norm": x_norm.astype(np.float32),
        "y_norm": y_norm.astype(np.float32),
    }


def backward_map_fast_pixel_coord_dual(depth_linear, precomp, rt):
    w = precomp["w"]
    h = precomp["h"]

    fx = precomp["fx_ref"]
    fy = precomp["fy_ref"]
    cx = precomp["cx_ref"]
    cy = precomp["cy_ref"]
    z_sign = precomp["z_sign"]

    x_norm = precomp["x_norm"]
    y_norm = precomp["y_norm"]
    z = depth_linear.astype(np.float32)

    rvec = np.array(rt["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(rt["tvec"], dtype=np.float32)
    rotation, _ = cv2.Rodrigues(rvec)
    rotation = rotation.astype(np.float32)

    tx = float(tvec[0])
    ty = float(tvec[1])
    tz = float(tvec[2])

    kx = (
        rotation[0, 0] * x_norm
        + rotation[0, 1] * y_norm
        + rotation[0, 2] * z_sign
    )
    ky = (
        rotation[1, 0] * x_norm
        + rotation[1, 1] * y_norm
        + rotation[1, 2] * z_sign
    )
    kz = (
        rotation[2, 0] * x_norm
        + rotation[2, 1] * y_norm
        + rotation[2, 2] * z_sign
    )

    xp = z * kx + tx
    yp = z * ky + ty
    zp = z * kz + tz
    denom = np.maximum(np.abs(zp), 1e-8)

    map_x = fx * (xp / denom) + cx
    map_y = fy * (yp / denom) + cy

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(z)
        & np.isfinite(zp)
        & (zp * z_sign > 0)
        & (map_x >= 0.0)
        & (map_x <= w - 1)
        & (map_y >= 0.0)
        & (map_y <= h - 1)
        & (z > 0.0)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y, valid


# ============================================================
# Remap / Warp
# ============================================================

def remap_plane(src, map_x, map_y, bit_depth, border_value):
    maxv = (1 << bit_depth) - 1

    dst = cv2.remap(
        src.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )

    dst = np.clip(np.round(dst), 0, maxv)
    return dst.astype(np.uint8 if bit_depth <= 8 else np.dtype("<u2"))


def downsample_luma_map_to_chroma_map(map_x, map_y):
    h, w = map_x.shape

    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("luma map size must be even for YUV420")

    uv_h = h // 2
    uv_w = w // 2
    mx = map_x.reshape(uv_h, 2, uv_w, 2)
    my = map_y.reshape(uv_h, 2, uv_w, 2)

    valid = (mx >= 0.0) & (my >= 0.0)
    cnt = np.sum(valid, axis=(1, 3)).astype(np.float32)
    sum_x = np.sum(np.where(valid, mx, 0.0), axis=(1, 3))
    sum_y = np.sum(np.where(valid, my, 0.0), axis=(1, 3))

    avg_x = np.full((uv_h, uv_w), -1.0, dtype=np.float32)
    avg_y = np.full((uv_h, uv_w), -1.0, dtype=np.float32)
    ok = cnt > 0
    avg_x[ok] = sum_x[ok] / cnt[ok]
    avg_y[ok] = sum_y[ok] / cnt[ok]

    map_x_uv = avg_x * 0.5
    map_y_uv = avg_y * 0.5
    map_x_uv[~ok] = -1.0
    map_y_uv[~ok] = -1.0

    return map_x_uv.astype(np.float32), map_y_uv.astype(np.float32)


def backward_warp_yuv420_bilinear(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    y = remap_plane(prev_y, map_x, map_y, bit_depth, 0)
    map_x_uv, map_y_uv = downsample_luma_map_to_chroma_map(map_x, map_y)
    neutral = 128 if bit_depth <= 8 else 512
    u = remap_plane(prev_u, map_x_uv, map_y_uv, bit_depth, neutral)
    v = remap_plane(prev_v, map_x_uv, map_y_uv, bit_depth, neutral)
    return y, u, v


# ============================================================
# Optional subblk4 + 6tap torch
# ============================================================

LUMA_6TAP_32_NP = np.array([
    [0,   0, 256,   0,   0, 0],
    [0,  -4, 253,   9,  -2, 0],
    [1,  -7, 249,  17,  -4, 0],
    [1, -10, 245,  25,  -6, 1],
    [1, -13, 241,  34,  -8, 1],
    [2, -16, 235,  44, -10, 1],
    [2, -18, 229,  53, -12, 2],
    [2, -20, 223,  63, -14, 2],
    [2, -22, 217,  72, -15, 2],
    [3, -23, 209,  82, -17, 2],
    [3, -24, 202,  92, -19, 2],
    [3, -25, 194, 101, -20, 3],
    [3, -25, 185, 111, -21, 3],
    [3, -26, 178, 121, -23, 3],
    [3, -25, 168, 131, -24, 3],
    [3, -25, 159, 141, -25, 3],
    [3, -25, 150, 150, -25, 3],
    [3, -25, 141, 159, -25, 3],
    [3, -24, 131, 168, -25, 3],
    [3, -23, 121, 178, -26, 3],
    [3, -21, 111, 185, -25, 3],
    [3, -20, 101, 194, -25, 3],
    [2, -19,  92, 202, -24, 3],
    [2, -17,  82, 209, -23, 3],
    [2, -15,  72, 217, -22, 2],
    [2, -14,  63, 223, -20, 2],
    [2, -12,  53, 229, -18, 2],
    [1, -10,  44, 235, -16, 2],
    [1,  -8,  34, 241, -13, 1],
    [1,  -6,  25, 245, -10, 1],
    [0,  -4,  17, 249,  -7, 1],
    [0,  -2,   9, 253,  -4, 0],
], dtype=np.float32)


def make_subblk4_avg_flow_map_fast(map_x, map_y):
    h, w = map_x.shape

    if h % 4 != 0 or w % 4 != 0:
        raise ValueError("subblk4 mode requires width/height multiple of 4")

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    flow_x = map_x - xx
    flow_y = map_y - yy
    valid = (map_x >= 0.0) & (map_y >= 0.0)

    bh = h // 4
    bw = w // 4
    fx4 = flow_x.reshape(bh, 4, bw, 4)
    fy4 = flow_y.reshape(bh, 4, bw, 4)
    vd4 = valid.reshape(bh, 4, bw, 4)

    corner_valid = (
        vd4[:, 0, :, 0]
        & vd4[:, 0, :, 3]
        & vd4[:, 3, :, 0]
        & vd4[:, 3, :, 3]
    )

    avg_fx = (
        fx4[:, 0, :, 0]
        + fx4[:, 0, :, 3]
        + fx4[:, 3, :, 0]
        + fx4[:, 3, :, 3]
    ) * 0.25

    avg_fy = (
        fy4[:, 0, :, 0]
        + fy4[:, 0, :, 3]
        + fy4[:, 3, :, 0]
        + fy4[:, 3, :, 3]
    ) * 0.25

    avg_fx = np.repeat(np.repeat(avg_fx, 4, axis=0), 4, axis=1)
    avg_fy = np.repeat(np.repeat(avg_fy, 4, axis=0), 4, axis=1)
    blk_valid = np.repeat(np.repeat(corner_valid, 4, axis=0), 4, axis=1)

    out_x = xx + avg_fx
    out_y = yy + avg_fy
    out_x[~blk_valid] = -1.0
    out_y[~blk_valid] = -1.0
    return out_x.astype(np.float32), out_y.astype(np.float32)


def remap_plane_subblk4_6tap_torch(src, map_x, map_y, bit_depth, device=None):
    try:
        import torch
        import torch.nn.functional as f
    except ImportError as exc:
        raise ImportError(
            "subblk4_6tap_torch mode requires torch. "
            "Use --warp-filter bilinear or install torch."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    h, w = src.shape
    maxv = (1 << bit_depth) - 1
    sub_x, sub_y = make_subblk4_avg_flow_map_fast(map_x, map_y)

    valid_np = (
        (sub_x >= 0.0)
        & (sub_y >= 0.0)
        & (sub_x <= w - 1)
        & (sub_y <= h - 1)
    )

    sx = torch.from_numpy(sub_x).to(device=device, dtype=torch.float32)
    sy = torch.from_numpy(sub_y).to(device=device, dtype=torch.float32)
    valid = torch.from_numpy(valid_np).to(device=device)

    ix = torch.floor(sx).to(torch.long)
    iy = torch.floor(sy).to(torch.long)
    frac_x = torch.round((sx - ix.float()) * 32.0).to(torch.long)
    frac_y = torch.round((sy - iy.float()) * 32.0).to(torch.long)

    carry_x = frac_x >= 32
    carry_y = frac_y >= 32
    ix = ix + carry_x.long()
    iy = iy + carry_y.long()
    frac_x = torch.where(carry_x, torch.zeros_like(frac_x), frac_x)
    frac_y = torch.where(carry_y, torch.zeros_like(frac_y), frac_y)
    ix = ix.clamp(0, w - 1)
    iy = iy.clamp(0, h - 1)

    src_t = torch.from_numpy(src.astype(np.float32)).to(device).view(1, 1, h, w)
    src_pad = f.pad(src_t, (2, 3, 2, 3), mode="replicate")
    patches_all = f.unfold(src_pad, kernel_size=(6, 6), stride=1)
    col_idx = (iy * w + ix).reshape(-1)
    patches = patches_all[0, :, col_idx]

    coeff = torch.from_numpy(LUMA_6TAP_32_NP).to(
        device=device,
        dtype=torch.float32,
    )
    cx = coeff[frac_x.reshape(-1)]
    cy = coeff[frac_y.reshape(-1)]
    weight = (cy[:, :, None] * cx[:, None, :]).reshape(-1, 36)

    val = torch.sum(patches.transpose(0, 1) * weight, dim=1)
    val = torch.round(val / 65536.0).clamp(0, maxv).reshape(h, w)
    val = torch.where(valid, val, torch.zeros_like(val))
    out = val.detach().cpu().numpy()
    return out.astype(np.uint8 if bit_depth <= 8 else np.dtype("<u2"))


def backward_warp_yuv420_subblk4_6tap_torch(
    prev_y,
    prev_u,
    prev_v,
    map_x,
    map_y,
    bit_depth,
    torch_device=None,
):
    wy = remap_plane_subblk4_6tap_torch(
        prev_y,
        map_x,
        map_y,
        bit_depth,
        device=torch_device,
    )

    map_x_uv, map_y_uv = downsample_luma_map_to_chroma_map(map_x, map_y)
    neutral = 128 if bit_depth <= 8 else 512
    wu = remap_plane(prev_u, map_x_uv, map_y_uv, bit_depth, neutral)
    wv = remap_plane(prev_v, map_x_uv, map_y_uv, bit_depth, neutral)
    return wy, wu, wv


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seq-yuv", required=True)
    ap.add_argument("--depth-yuv", required=True)
    ap.add_argument("--param-jsonl", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)

    ap.add_argument(
        "--seq-start",
        type=int,
        default=0,
        help=(
            "Frame index offset in seq-yuv. "
            "For rap1 starting at original frame 32, use --seq-start 32."
        ),
    )

    ap.add_argument("--coded-width", type=int, default=None)
    ap.add_argument("--coded-height", type=int, default=None)
    ap.add_argument("--pad-left", type=int, default=0)
    ap.add_argument("--pad-top", type=int, default=0)
    ap.add_argument("--bit-depth", type=int, default=10)

    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--out-q-jsonl", required=True)
    ap.add_argument(
        "--out-plane-depth-yuv",
        default="",
        help=(
            "Optional reconstructed block-plane depth YUV420p10le. "
            "Frames are written in local POC order."
        ),
    )

    ap.add_argument("--pred-n", type=int, default=3)
    ap.add_argument("--pred-degree", type=int, default=2)
    ap.add_argument("--ext-bits", type=int, default=16)
    ap.add_argument("--r-step", type=float, default=1.0 / 16.0)
    ap.add_argument("--t-step-norm", type=float, default=1.0 / 4.0)

    ap.add_argument("--intr-delta-bits", type=int, default=16)
    ap.add_argument("--intr-step", type=float, default=1.0 / 16.0)
    ap.add_argument("--intr-f-max", type=float, default=4.0)
    ap.add_argument("--intr-c-min", type=float, default=-1.0)
    ap.add_argument("--intr-c-max", type=float, default=2.0)

    ap.add_argument("--depth-scale-bits", type=int, default=16)

    ap.add_argument(
        "--depth-projection-mode",
        choices=["block_inv_plane", "per_pixel"],
        default="block_inv_plane",
        help=(
            "Depth used by camera projection. Default block_inv_plane fits "
            "1/z=a*x+b*y+c independently in each block."
        ),
    )
    ap.add_argument(
        "--depth-plane-block-size",
        type=int,
        default=16,
        help="Inverse-depth plane block size. Default: 16.",
    )
    ap.add_argument(
        "--depth-plane-min-depth",
        type=float,
        default=1e-8,
        help="Depth values <= this are treated as invalid.",
    )
    ap.add_argument(
        "--depth-plane-min-valid-samples",
        type=int,
        default=3,
        help="Below this count, use a constant inverse-depth plane.",
    )
    ap.add_argument(
        "--log-depth-plane-blocks",
        action="store_true",
        help="Write all per-block a,b,c records to the output JSONL.",
    )

    ap.add_argument(
        "--warp-filter",
        choices=["bilinear", "subblk4_6tap_torch"],
        default="bilinear",
    )
    ap.add_argument("--torch-device", default=None)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.r_step <= 0:
        raise ValueError("--r-step must be positive")
    if args.t_step_norm <= 0:
        raise ValueError("--t-step-norm must be positive")
    if args.intr_step <= 0:
        raise ValueError("--intr-step must be positive")
    if args.depth_plane_block_size <= 0:
        raise ValueError("--depth-plane-block-size must be positive")
    if args.depth_plane_min_depth < 0:
        raise ValueError("--depth-plane-min-depth must be non-negative")
    if args.depth_plane_min_valid_samples <= 0:
        raise ValueError("--depth-plane-min-valid-samples must be positive")

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_jsonl = Path(args.param_jsonl)
    out_yuv = Path(args.out_yuv)
    out_q_jsonl = Path(args.out_q_jsonl)
    out_plane_depth_yuv = (
        Path(args.out_plane_depth_yuv) if args.out_plane_depth_yuv else None
    )

    output_paths = [out_yuv, out_q_jsonl]
    if out_plane_depth_yuv is not None:
        output_paths.append(out_plane_depth_yuv)

    for path in output_paths:
        if path.exists():
            if args.overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"Output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    src_w = int(args.width)
    src_h = int(args.height)
    bit_depth = int(args.bit_depth)
    pad_left = int(args.pad_left)
    pad_top = int(args.pad_top)

    coded_w = (
        int(args.coded_width)
        if args.coded_width is not None
        else align_to(src_w + pad_left, 4)
    )
    coded_h = (
        int(args.coded_height)
        if args.coded_height is not None
        else align_to(src_h + pad_top, 4)
    )

    pad_right, pad_bottom = calc_padding(
        src_w,
        src_h,
        coded_w,
        coded_h,
        pad_left,
        pad_top,
    )

    validate_yuv420_padding(
        src_w,
        src_h,
        coded_w,
        coded_h,
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    )

    header, frames = load_param_jsonl(param_jsonl)
    pose_mode = header.get("pose_mode", "current_to_previous")
    if pose_mode != "current_to_previous":
        raise RuntimeError(
            f"This script expects pose_mode='current_to_previous', got '{pose_mode}'."
        )

    depth_scale_real = get_depth_scale_real_from_header(header)
    depth_scale_precision = get_depth_scale_precision_from_header(header)
    if depth_scale_real <= 0:
        raise ValueError(f"Invalid depth_scale_real: {depth_scale_real}")

    intr_gt_original0 = header["intrinsic"]
    intr_gt_padded0 = make_padded_intrinsic_from_original(
        intr_gt_original0,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    intr_q0, intr_dec0, intr_clip0 = quantize_intrinsic_16(
        intr_gt_padded0,
        coded_w,
        coded_h,
        f_max=args.intr_f_max,
        c_min=args.intr_c_min,
        c_max=args.intr_c_max,
    )

    seq_count = count_frames(seq_yuv, src_w, src_h, bit_depth)
    depth_count = count_frames(depth_yuv, src_w, src_h, 10)

    if args.seq_start < 0:
        raise ValueError("--seq-start must be non-negative")

    available_seq_count = seq_count - args.seq_start
    if available_seq_count <= 0:
        raise RuntimeError(
            f"--seq-start {args.seq_start} is outside seq-yuv frame count {seq_count}"
        )

    max_poc = min(available_seq_count, depth_count, max(frames.keys()) + 1)
    if max_poc <= 0:
        raise RuntimeError(
            f"No frames to process: available_seq_count={available_seq_count}, "
            f"depth_count={depth_count}, param_frames={len(frames)}"
        )

    decoded_hist = []
    decoded_intrinsics = [intr_dec0]

    q_abs_max_ext = signed_q_abs_max(args.ext_bits)
    q_abs_max_intr = signed_q_abs_max(args.intr_delta_bits)

    header_intrinsic_bits = 4 * 16
    depth_scale_bits = int(args.depth_scale_bits)
    z_sign_bits = 1
    header_bits = header_intrinsic_bits + depth_scale_bits + z_sign_bits

    total_ext_bits = 0
    total_ext_bits_r = 0
    total_ext_bits_t = 0
    total_ext_bits_each = np.zeros(6, dtype=np.int64)

    total_intr_delta_bits = 0
    total_intr_delta_bits_each = np.zeros(4, dtype=np.int64)
    total_intr_delta_frames = 0
    total_intr_clipped_frames = 0
    total_coded_frames = 0
    total_clipped_frames = 0

    sum_mae_active = 0.0
    sum_mae_coded = 0.0
    sum_psnr_active = 0.0
    sum_psnr_coded = 0.0
    psnr_count_active = 0
    psnr_count_coded = 0

    plane_maes = []
    plane_rmses = []
    plane_psnrs = []
    total_plane_blocks = 0
    total_plane_abc_blocks = 0
    total_plane_constant_blocks = 0
    total_plane_empty_blocks = 0

    ys_active, xs_active = active_slice(src_w, src_h, pad_left, pad_top)

    print("============================================================")
    print("Input summary")
    print("============================================================")
    print(f"seq_yuv               : {seq_yuv}")
    print(f"depth_yuv             : {depth_yuv}")
    print(f"param_jsonl           : {param_jsonl}")
    print(f"seq frames total      : {seq_count}")
    print(f"seq_start             : {args.seq_start}")
    print(f"depth frames          : {depth_count}")
    print(f"process frames        : {max_poc}")
    print(f"depth_scale real      : {depth_scale_real}")
    print(f"pose_mode             : {pose_mode}")
    print(f"depth projection mode : {args.depth_projection_mode}")
    print(f"depth plane block     : {args.depth_plane_block_size}x{args.depth_plane_block_size}")
    print(f"warp_filter           : {args.warp_filter}")
    print("============================================================")

    with open(out_q_jsonl, "w", encoding="utf-8") as fq:
        fq.write(
            json.dumps(
                {
                    "type": "header",
                    "source_size": {"width": src_w, "height": src_h},
                    "coded_size": {"width": coded_w, "height": coded_h},
                    "padding": {
                        "left": pad_left,
                        "top": pad_top,
                        "right": pad_right,
                        "bottom": pad_bottom,
                    },
                    "seq_start": int(args.seq_start),
                    "projection_mode": "fast_pixel_coordinate_no_ndc_dual_intrinsic",
                    "depth_projection_mode": args.depth_projection_mode,
                    "depth_plane": {
                        "equation": "1/z = a*(x-cx) + b*(y-cy) + c",
                        "block_size": int(args.depth_plane_block_size),
                        "min_depth": float(args.depth_plane_min_depth),
                        "min_valid_samples": int(args.depth_plane_min_valid_samples),
                        "coefficient_quantization": "none",
                        "fallback": "constant inverse depth c=mean(1/z)",
                        "invalid_input_depth": "preserved invalid",
                    },
                    "warp_filter": args.warp_filter,
                    "depth_padding": "edge",
                    "image_padding": "edge",
                    "depth_scale_header": header["depth_scale"],
                    "depth_scale_precision": depth_scale_precision,
                    "depth_scale_real": depth_scale_real,
                    "intrinsic_gt_original0": intr_gt_original0,
                    "intrinsic_gt_padded0": intr_gt_padded0,
                    "intrinsic_q16_first": intr_q0,
                    "intrinsic_dec_first": intr_dec0,
                    "intrinsic_first_clipped": intr_clip0,
                    "intrinsic_delta_code": "signed_truncated_exp_golomb",
                    "intrinsic_delta_bits": args.intr_delta_bits,
                    "intrinsic_delta_q_abs_max": q_abs_max_intr,
                    "intrinsic_delta_step": args.intr_step,
                    "extrinsic_bits": args.ext_bits,
                    "extrinsic_q_abs_max": q_abs_max_ext,
                    "r_step": args.r_step,
                    "t_step_norm": args.t_step_norm,
                    "pred_n": args.pred_n,
                    "pred_degree": args.pred_degree,
                    "bit_count": {
                        "header_intrinsic_bits": header_intrinsic_bits,
                        "depth_scale_bits": depth_scale_bits,
                        "z_sign_bits": z_sign_bits,
                        "header_bits": header_bits,
                    },
                    "param6_order": [
                        "rx",
                        "ry",
                        "rz",
                        "tx_over_depth_scale_real",
                        "ty_over_depth_scale_real",
                        "tz_over_depth_scale_real",
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for poc in range(max_poc):
            seq_idx = args.seq_start + poc
            cur_y, cur_u, cur_v = read_yuv420(
                seq_yuv,
                seq_idx,
                src_w,
                src_h,
                bit_depth,
            )
            cur_y_pad, cur_u_pad, cur_v_pad = pad_yuv420_edge(
                cur_y,
                cur_u,
                cur_v,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            )

            if poc == 0:
                write_yuv420(out_yuv, cur_y_pad, cur_u_pad, cur_v_pad)
                p0_dec = np.zeros(6, dtype=np.float32)
                decoded_hist.append(p0_dec)

                # Also model/write POC 0 depth so optional debug output has the
                # same frame count as the warped output.
                depth_y0, _, _ = read_yuv420(depth_yuv, poc, src_w, src_h, 10)
                depth_linear0 = depth_y0.astype(np.float32) * float(depth_scale_real)
                depth_linear0_pad = pad_2d_edge(
                    depth_linear0,
                    coded_w,
                    coded_h,
                    pad_left,
                    pad_top,
                ).astype(np.float32)

                if args.depth_projection_mode == "block_inv_plane":
                    depth_used0, _, plane_stats0, plane_blocks0 = (
                        model_depth_with_inverse_planes(
                            depth_linear0_pad,
                            block_size=args.depth_plane_block_size,
                            min_depth=args.depth_plane_min_depth,
                            min_valid_samples=args.depth_plane_min_valid_samples,
                        )
                    )
                else:
                    depth_used0 = depth_linear0_pad
                    plane_blocks0 = []
                    plane_stats0 = {
                        "model": "per_pixel",
                        "block_size": 0,
                        "num_blocks": 0,
                        "num_abc_blocks": 0,
                        "num_constant_blocks": 0,
                        "num_empty_blocks": 0,
                        "valid_ratio": float(np.mean(depth_used0 > 0.0)),
                        "mae": 0.0,
                        "rmse": 0.0,
                        "psnr": float("inf"),
                        "max_abs_err": 0.0,
                    }

                if out_plane_depth_yuv is not None:
                    write_depth_linear_as_yuv420p10le(
                        out_plane_depth_yuv,
                        depth_used0,
                        depth_scale_real,
                    )

                rec0 = {
                    "poc": 0,
                    "seq_idx": int(seq_idx),
                    "q_residual": [0, 0, 0, 0, 0, 0],
                    "q_residual_bits": [0, 0, 0, 0, 0, 0],
                    "q_residual_total_bits": 0,
                    "param6_dec": p0_dec.astype(float).tolist(),
                    "intrinsic_delta_gt": [0.0, 0.0, 0.0, 0.0],
                    "q_intrinsic_delta": [0, 0, 0, 0],
                    "q_intrinsic_delta_bits": [0, 0, 0, 0],
                    "q_intrinsic_delta_total_bits": 0,
                    "intrinsic_dec": intr_dec0,
                    "depth_plane_stats": plane_stats0,
                    "mae_y_active": 0.0,
                    "mae_y_coded": 0.0,
                    "psnr_y_active": "inf",
                    "psnr_y_coded": "inf",
                }
                if args.log_depth_plane_blocks:
                    rec0["depth_plane_blocks"] = plane_blocks0
                fq.write(json.dumps(rec0, ensure_ascii=False) + "\n")

                print(f"[{poc:04d}/{max_poc - 1:04d}] copy first frame, seq_idx={seq_idx}")
                continue

            if poc not in frames:
                raise RuntimeError(f"POC {poc} not found in param jsonl")

            frame = frames[poc]
            if "intrinsic_delta" not in frame:
                raise RuntimeError(
                    f"POC {poc} has no intrinsic_delta. "
                    f"Regenerate camParam JSONL with per-frame intrinsic_delta."
                )

            intr_delta_gt = np.array(
                frame["intrinsic_delta"],
                dtype=np.float32,
            ).reshape(4)
            q_intr, d_intr, clip_intr = quantize_intrinsic_delta_4(
                intr_delta_gt,
                step=args.intr_step,
                bits=args.intr_delta_bits,
            )
            q_intr_bits_each, q_intr_bits_total = (
                q_residual_bits_signed_trunc_exp_golomb(
                    q_intr,
                    q_abs_max=q_abs_max_intr,
                )
            )

            total_intr_delta_bits += q_intr_bits_total
            total_intr_delta_bits_each += np.array(q_intr_bits_each, dtype=np.int64)
            total_intr_delta_frames += 1
            intrinsic_clipped = bool(np.any(clip_intr))
            if intrinsic_clipped:
                total_intr_clipped_frames += 1

            intr_dec_cur = add_intrinsic_delta(decoded_intrinsics[-1], d_intr)
            decoded_intrinsics.append(intr_dec_cur)
            intr_dec_ref = decoded_intrinsics[poc - 1]
            intr_dec_tar = decoded_intrinsics[poc]

            p_gt = param6_from_frame(frame, depth_scale_real)
            p_pred = predict_from_history(
                decoded_hist,
                pred_n=args.pred_n,
                pred_degree=args.pred_degree,
            )
            residual = p_gt - p_pred

            q_r, d_r, clip_r = quant_s(
                residual[:3],
                step=args.r_step,
                bits=args.ext_bits,
            )
            q_t, d_t, clip_t = quant_s(
                residual[3:],
                step=args.t_step_norm,
                bits=args.ext_bits,
            )
            q_residual = np.concatenate([q_r, q_t]).astype(np.int32)
            q_bits_each, q_bits_total = q_residual_bits_signed_trunc_exp_golomb(
                q_residual,
                q_abs_max=q_abs_max_ext,
            )

            total_ext_bits += q_bits_total
            total_ext_bits_r += sum(q_bits_each[:3])
            total_ext_bits_t += sum(q_bits_each[3:])
            total_ext_bits_each += np.array(q_bits_each, dtype=np.int64)
            total_coded_frames += 1

            p_dec = p_pred.copy()
            p_dec[:3] += d_r
            p_dec[3:] += d_t
            decoded_hist.append(p_dec)
            rt_dec = rt_from_param6(p_dec, depth_scale_real)

            depth_y, _, _ = read_yuv420(
                depth_yuv,
                poc,
                src_w,
                src_h,
                10,
            )
            depth_linear = depth_y.astype(np.float32) * float(depth_scale_real)
            depth_linear_pad = pad_2d_edge(
                depth_linear,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            ).astype(np.float32)

            if args.depth_projection_mode == "block_inv_plane":
                depth_for_projection, depth_plane_valid, plane_stats, plane_blocks = (
                    model_depth_with_inverse_planes(
                        depth_linear_pad,
                        block_size=args.depth_plane_block_size,
                        min_depth=args.depth_plane_min_depth,
                        min_valid_samples=args.depth_plane_min_valid_samples,
                    )
                )

                total_plane_blocks += int(plane_stats["num_blocks"])
                total_plane_abc_blocks += int(plane_stats["num_abc_blocks"])
                total_plane_constant_blocks += int(
                    plane_stats["num_constant_blocks"]
                )
                total_plane_empty_blocks += int(plane_stats["num_empty_blocks"])

                if plane_stats["mae"] is not None:
                    plane_maes.append(float(plane_stats["mae"]))
                if plane_stats["rmse"] is not None:
                    plane_rmses.append(float(plane_stats["rmse"]))
                if plane_stats["psnr"] is not None and np.isfinite(
                    float(plane_stats["psnr"])
                ):
                    plane_psnrs.append(float(plane_stats["psnr"]))
            else:
                depth_for_projection = depth_linear_pad
                depth_plane_valid = depth_for_projection > args.depth_plane_min_depth
                plane_blocks = []
                plane_stats = {
                    "model": "per_pixel",
                    "block_size": 0,
                    "num_blocks": 0,
                    "num_abc_blocks": 0,
                    "num_constant_blocks": 0,
                    "num_empty_blocks": 0,
                    "valid_ratio": float(np.mean(depth_plane_valid)),
                    "mae": 0.0,
                    "rmse": 0.0,
                    "psnr": float("inf"),
                    "max_abs_err": 0.0,
                }

            if out_plane_depth_yuv is not None:
                write_depth_linear_as_yuv420p10le(
                    out_plane_depth_yuv,
                    depth_for_projection,
                    depth_scale_real,
                )

            projection_precomp = make_projection_precompute_dual(
                coded_w,
                coded_h,
                intr_tar=intr_dec_tar,
                intr_ref=intr_dec_ref,
            )
            map_x, map_y, map_valid = backward_map_fast_pixel_coord_dual(
                depth_linear=depth_for_projection,
                precomp=projection_precomp,
                rt=rt_dec,
            )

            # Preserve the depth-model validity decision explicitly.
            map_valid &= depth_plane_valid
            map_x[~map_valid] = -1.0
            map_y[~map_valid] = -1.0

            prev_seq_idx = args.seq_start + poc - 1
            prev_y, prev_u, prev_v = read_yuv420(
                seq_yuv,
                prev_seq_idx,
                src_w,
                src_h,
                bit_depth,
            )
            prev_y_pad, prev_u_pad, prev_v_pad = pad_yuv420_edge(
                prev_y,
                prev_u,
                prev_v,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            )

            if args.warp_filter == "bilinear":
                wy, wu, wv = backward_warp_yuv420_bilinear(
                    prev_y_pad,
                    prev_u_pad,
                    prev_v_pad,
                    map_x,
                    map_y,
                    bit_depth,
                )
            elif args.warp_filter == "subblk4_6tap_torch":
                wy, wu, wv = backward_warp_yuv420_subblk4_6tap_torch(
                    prev_y_pad,
                    prev_u_pad,
                    prev_v_pad,
                    map_x,
                    map_y,
                    bit_depth,
                    torch_device=args.torch_device,
                )
            else:
                raise ValueError(args.warp_filter)

            write_yuv420(out_yuv, wy, wu, wv)

            mae_y_coded = float(
                np.mean(
                    np.abs(wy.astype(np.float32) - cur_y_pad.astype(np.float32))
                )
            )
            mae_y_active = float(
                np.mean(
                    np.abs(
                        wy[ys_active, xs_active].astype(np.float32)
                        - cur_y_pad[ys_active, xs_active].astype(np.float32)
                    )
                )
            )
            psnr_y_coded = calc_psnr(wy, cur_y_pad, bit_depth)
            psnr_y_active = calc_psnr(
                wy[ys_active, xs_active],
                cur_y_pad[ys_active, xs_active],
                bit_depth,
            )

            clipped = bool(np.any(clip_r) or np.any(clip_t))
            if clipped:
                total_clipped_frames += 1

            sum_mae_active += mae_y_active
            sum_mae_coded += mae_y_coded
            if np.isfinite(psnr_y_active):
                sum_psnr_active += psnr_y_active
                psnr_count_active += 1
            if np.isfinite(psnr_y_coded):
                sum_psnr_coded += psnr_y_coded
                psnr_count_coded += 1

            out_rec = {
                "poc": int(poc),
                "seq_idx": int(seq_idx),
                "q_residual": q_residual.astype(int).tolist(),
                "q_residual_bits": q_bits_each,
                "q_residual_total_bits": int(q_bits_total),
                "param6_pred": p_pred.astype(float).tolist(),
                "param6_dec": p_dec.astype(float).tolist(),
                "param6_gt": p_gt.astype(float).tolist(),
                "rt_dec": rt_dec,
                "extrinsic_clipped": clipped,
                "intrinsic_delta_gt": intr_delta_gt.astype(float).tolist(),
                "q_intrinsic_delta": q_intr.astype(int).tolist(),
                "intrinsic_delta_dec": d_intr.astype(float).tolist(),
                "q_intrinsic_delta_bits": q_intr_bits_each,
                "q_intrinsic_delta_total_bits": int(q_intr_bits_total),
                "intrinsic_ref_dec": intr_dec_ref,
                "intrinsic_tar_dec": intr_dec_tar,
                "intrinsic_clipped": intrinsic_clipped,
                "depth_plane_stats": plane_stats,
                "projection_valid_ratio": float(np.mean(map_valid)),
                "mae_y_active": mae_y_active,
                "mae_y_coded": mae_y_coded,
                "psnr_y_active": json_safe_float(psnr_y_active),
                "psnr_y_coded": json_safe_float(psnr_y_coded),
            }
            if args.log_depth_plane_blocks:
                out_rec["depth_plane_blocks"] = plane_blocks
            fq.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            print(
                f"[{poc:04d}/{max_poc - 1:04d}] "
                f"seq_idx={seq_idx}, "
                f"planePSNR={json_safe_float(plane_stats['psnr'])}, "
                f"planeABC={plane_stats['num_abc_blocks']}, "
                f"planeConst={plane_stats['num_constant_blocks']}, "
                f"valid={np.mean(map_valid):.4f}, "
                f"Y-PSNR-active={psnr_y_active:.3f} dB, "
                f"Y-MAE-active={mae_y_active:.3f}, "
                f"ext_bits={q_bits_total}, "
                f"intr_bits={q_intr_bits_total}"
            )

    total_bits = header_bits + total_ext_bits + total_intr_delta_bits

    print("============================================================")
    print("Padding / projection summary")
    print("============================================================")
    print(f"source size           : {src_w}x{src_h}")
    print(f"coded size            : {coded_w}x{coded_h}")
    print(
        f"padding               : L={pad_left}, T={pad_top}, "
        f"R={pad_right}, B={pad_bottom}"
    )
    print("projection            : fast pixel-coordinate, dual intrinsic, no NDC")
    print(f"depth mode            : {args.depth_projection_mode}")
    print(f"inverse-plane block   : {args.depth_plane_block_size}")
    print(f"warp filter           : {args.warp_filter}")

    if args.depth_projection_mode == "block_inv_plane":
        print("------------------------------------------------------------")
        print("Inverse-depth plane summary")
        print("------------------------------------------------------------")
        print(f"total blocks          : {total_plane_blocks}")
        print(f"ABC blocks            : {total_plane_abc_blocks}")
        print(f"constant fallback     : {total_plane_constant_blocks}")
        print(f"empty blocks          : {total_plane_empty_blocks}")
        if plane_maes:
            print(f"avg plane MAE         : {np.mean(plane_maes):.9g}")
        if plane_rmses:
            print(f"avg plane RMSE        : {np.mean(plane_rmses):.9g}")
        if plane_psnrs:
            print(f"avg plane PSNR        : {np.mean(plane_psnrs):.3f} dB")

    print("============================================================")
    print("Metric summary")
    print("============================================================")
    if total_coded_frames > 0:
        avg_mae_active = sum_mae_active / total_coded_frames
        avg_mae_coded = sum_mae_coded / total_coded_frames
        avg_psnr_active = (
            sum_psnr_active / psnr_count_active
            if psnr_count_active > 0
            else float("inf")
        )
        avg_psnr_coded = (
            sum_psnr_coded / psnr_count_coded
            if psnr_count_coded > 0
            else float("inf")
        )
        print(f"avg MAE active        : {avg_mae_active:.3f}")
        print(f"avg MAE coded         : {avg_mae_coded:.3f}")
        print(f"avg PSNR active       : {avg_psnr_active:.3f} dB")
        print(f"avg PSNR coded        : {avg_psnr_coded:.3f} dB")

    print("============================================================")
    print("Bit summary")
    print("============================================================")
    print(f"header bits           : {header_bits} bits")
    print(f"extrinsic frames      : {total_coded_frames}")
    print(f"extrinsic clipped     : {total_clipped_frames}")
    print(f"extrinsic total bits  : {total_ext_bits} bits")

    if total_coded_frames > 0:
        avg_ext_bits = total_ext_bits / total_coded_frames
        avg_r_bits = total_ext_bits_r / total_coded_frames
        avg_t_bits = total_ext_bits_t / total_coded_frames
        avg_bits_each = total_ext_bits_each.astype(np.float64) / total_coded_frames
        print(f"avg ext bits/frame    : {avg_ext_bits:.3f}")
        print(f"avg rotation bits     : {avg_r_bits:.3f} / frame")
        print(f"avg translation bits  : {avg_t_bits:.3f} / frame")
        print(
            "avg ext bits each     : "
            f"rx={avg_bits_each[0]:.3f}, "
            f"ry={avg_bits_each[1]:.3f}, "
            f"rz={avg_bits_each[2]:.3f}, "
            f"tx={avg_bits_each[3]:.3f}, "
            f"ty={avg_bits_each[4]:.3f}, "
            f"tz={avg_bits_each[5]:.3f}"
        )

    print(f"intrinsic delta frames: {total_intr_delta_frames}")
    print(f"intrinsic clipped     : {total_intr_clipped_frames}")
    print(f"intrinsic delta bits  : {total_intr_delta_bits} bits")

    if total_intr_delta_frames > 0:
        avg_intr_bits = total_intr_delta_bits / total_intr_delta_frames
        avg_intr_each = (
            total_intr_delta_bits_each.astype(np.float64)
            / total_intr_delta_frames
        )
        print(f"avg intr bits/frame   : {avg_intr_bits:.3f}")
        print(
            "avg intr bits each    : "
            f"dfx={avg_intr_each[0]:.3f}, "
            f"dfy={avg_intr_each[1]:.3f}, "
            f"dcx={avg_intr_each[2]:.3f}, "
            f"dcy={avg_intr_each[3]:.3f}"
        )

    print(f"total bits            : {total_bits} bits")
    print("============================================================")
    print("Done.")
    print(f"warped yuv            : {out_yuv}")
    print(f"q jsonl               : {out_q_jsonl}")
    if out_plane_depth_yuv is not None:
        print(f"plane depth yuv       : {out_plane_depth_yuv}")


if __name__ == "__main__":
    main()
