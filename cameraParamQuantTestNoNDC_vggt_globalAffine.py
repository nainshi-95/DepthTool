#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project previous frame to current frame using decoded depth/camera parameters,
with optional global 2CP/3CP affine bias correction over cam-proj flow.

Main addition:
  --global-affine-bias

Flow:
  1) Generate cam-proj backward map from current depth/camera.
  2) Warp previous frame with cam-proj map only.
  3) Estimate one global affine correction from cam-warped Y to current Y
     using only valid active-region pixels.
  4) Convert affine correction to 2 or 3 control-point biases.
  5) Quantize/decode the CP bias and estimate bits assuming 50:50 bin probability.
  6) Apply decoded affine bias to cam-proj map and perform final warp.
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

    mse = np.mean((a - b) ** 2)

    if mse == 0:
        return float("inf")

    maxv = (1 << bit_depth) - 1
    return 10.0 * np.log10((maxv * maxv) / mse)


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


def intrinsic_to_vec4(intr):
    return np.array(
        [
            float(intr["fx"]),
            float(intr["fy"]),
            float(intr["cx"]),
            float(intr["cy"]),
        ],
        dtype=np.float32,
    )


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
    """
    delta order:
      [dfx, dfy, dcx, dcy]

    q = round(delta / step)
    dec_delta = q * step
    """
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

    if x > 0:
        return 2 * x - 1

    return -2 * x


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
            f"x={x} outside signed truncated range "
            f"[-{q_abs_max}, {q_abs_max}]"
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

    A = np.vander(x, N=deg + 1, increasing=True)
    b = np.vander(x_next, N=deg + 1, increasing=True)

    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = b @ coef

    return pred.reshape(6).astype(np.float32)


# ============================================================
# Fast Pixel-coordinate Projection
# ============================================================

def make_projection_precompute_dual(w, h, intr_tar, intr_ref):
    """
    target unprojection uses intr_tar.
    reference projection uses intr_ref.

      X_tar = [(x-cx_tar)/fx_tar*z,
               (y-cy_tar)/fy_tar*z,
               z_sign*z]

      X_ref = R * X_tar + t

      map_x = fx_ref * X_ref.x / |X_ref.z| + cx_ref
      map_y = fy_ref * X_ref.y / |X_ref.z| + cy_ref
    """
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

    R, _ = cv2.Rodrigues(rvec)
    R = R.astype(np.float32)

    tx = float(tvec[0])
    ty = float(tvec[1])
    tz = float(tvec[2])

    kx = R[0, 0] * x_norm + R[0, 1] * y_norm + R[0, 2] * z_sign
    ky = R[1, 0] * x_norm + R[1, 1] * y_norm + R[1, 2] * z_sign
    kz = R[2, 0] * x_norm + R[2, 1] * y_norm + R[2, 2] * z_sign

    Xp = z * kx + tx
    Yp = z * ky + ty
    Zp = z * kz + tz

    denom = np.maximum(np.abs(Zp), 1e-8)

    map_x = fx * (Xp / denom) + cx
    map_y = fy * (Yp / denom) + cy

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (Zp * z_sign > 0)
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

    return map_x, map_y


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

    if bit_depth <= 8:
        return dst.astype(np.uint8)

    return dst.astype(np.dtype("<u2"))


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
# Global affine bias over cam-proj flow
# ============================================================

def make_valid_bool_mask(map_x, map_y, w, h, erode=0, active_region=None):
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= w - 1)
        & (map_y >= 0.0)
        & (map_y <= h - 1)
    )

    if active_region is not None:
        ys, xs = active_region
        active = np.zeros_like(valid, dtype=bool)
        active[ys, xs] = True
        valid &= active

    if erode > 0:
        k = 2 * int(erode) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_u8 = (valid.astype(np.uint8) * 255)
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        valid = mask_u8 > 0

    return valid


def make_valid_u8_mask(map_x, map_y, w, h, erode=0, active_region=None):
    return make_valid_bool_mask(
        map_x,
        map_y,
        w,
        h,
        erode=erode,
        active_region=active_region,
    ).astype(np.uint8) * 255


def bilinear_sample_float_np(src, x, y):
    """Vectorized bilinear sampling. Returns (value, valid)."""
    src_f = src.astype(np.float32, copy=False)
    h, w = src_f.shape

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= w - 1)
        & (y >= 0.0)
        & (y <= h - 1)
    )

    xc = np.clip(x, 0.0, float(w - 1))
    yc = np.clip(y, 0.0, float(h - 1))

    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    fx = xc - x0.astype(np.float32)
    fy = yc - y0.astype(np.float32)

    p00 = src_f[y0, x0]
    p01 = src_f[y0, x1]
    p10 = src_f[y1, x0]
    p11 = src_f[y1, x1]

    a = p00 * (1.0 - fx) + p01 * fx
    b = p10 * (1.0 - fx) + p11 * fx
    val = a * (1.0 - fy) + b * fy

    val = np.where(valid, val, 0.0).astype(np.float32)
    return val, valid


def local_patch_sad_cost(
    cur_y,
    ref_y,
    map_x,
    map_y,
    cx,
    cy,
    rdx,
    rdy,
    block_radius,
):
    h, w = cur_y.shape
    br = int(block_radius)

    x0 = int(cx) - br
    x1 = int(cx) + br + 1
    y0 = int(cy) - br
    y1 = int(cy) + br + 1

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return float("inf")

    cur_patch = cur_y[y0:y1, x0:x1].astype(np.float32)
    mx = map_x[y0:y1, x0:x1].astype(np.float32) + float(rdx)
    my = map_y[y0:y1, x0:x1].astype(np.float32) + float(rdy)

    ref_patch, valid = bilinear_sample_float_np(ref_y, mx, my)

    if not np.all(valid):
        return float("inf")

    return float(np.mean(np.abs(cur_patch - ref_patch)))


def find_best_residual_for_sample(
    cur_y,
    ref_y,
    map_x,
    map_y,
    x,
    y,
    search_range,
    block_radius,
    subpel_levels=0,
    min_cur_var=0.0,
):
    h, w = cur_y.shape
    br = int(block_radius)

    x0 = int(x) - br
    x1 = int(x) + br + 1
    y0 = int(y) - br
    y1 = int(y) + br + 1

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False, 0.0, 0.0, float("inf")

    if min_cur_var > 0.0:
        cur_patch = cur_y[y0:y1, x0:x1].astype(np.float32)
        if float(np.var(cur_patch)) < float(min_cur_var):
            return False, 0.0, 0.0, float("inf")

    best_cost = float("inf")
    best_dx = 0.0
    best_dy = 0.0

    sr = int(search_range)
    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            cost = local_patch_sad_cost(
                cur_y,
                ref_y,
                map_x,
                map_y,
                x,
                y,
                float(dx),
                float(dy),
                block_radius,
            )

            if cost < best_cost:
                best_cost = cost
                best_dx = float(dx)
                best_dy = float(dy)

    if not np.isfinite(best_cost):
        return False, 0.0, 0.0, float("inf")

    refine_steps = []
    if int(subpel_levels) >= 1:
        refine_steps.append(0.5)
    if int(subpel_levels) >= 2:
        refine_steps.append(0.25)

    for step in refine_steps:
        base_dx = best_dx
        base_dy = best_dy
        for oy in [-step, 0.0, step]:
            for ox in [-step, 0.0, step]:
                dx = base_dx + ox
                dy = base_dy + oy
                cost = local_patch_sad_cost(
                    cur_y,
                    ref_y,
                    map_x,
                    map_y,
                    x,
                    y,
                    dx,
                    dy,
                    block_radius,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_dx = dx
                    best_dy = dy

    return True, float(best_dx), float(best_dy), float(best_cost)


def collect_residual_samples_local_search(
    cur_y,
    ref_y,
    map_x,
    map_y,
    valid_mask,
    active_region,
    sample_step,
    search_range,
    block_radius,
    subpel_levels=0,
    max_sample_cost=-1.0,
    min_cur_var=0.0,
):
    ys, xs = active_region
    y_start = ys.start
    y_stop = ys.stop
    x_start = xs.start
    x_stop = xs.stop

    br = int(block_radius)
    step = max(1, int(sample_step))

    samples = []
    costs = []

    y0 = y_start + br
    y1 = y_stop - br
    x0 = x_start + br
    x1 = x_stop - br

    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            if not bool(valid_mask[y, x]):
                continue

            ok, dx, dy, cost = find_best_residual_for_sample(
                cur_y,
                ref_y,
                map_x,
                map_y,
                x,
                y,
                search_range=search_range,
                block_radius=block_radius,
                subpel_levels=subpel_levels,
                min_cur_var=min_cur_var,
            )

            if not ok:
                continue

            if max_sample_cost is not None and float(max_sample_cost) >= 0.0:
                if cost > float(max_sample_cost):
                    continue

            samples.append([float(x), float(y), float(dx), float(dy)])
            costs.append(float(cost))

    if not samples:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(samples, dtype=np.float32), np.asarray(costs, dtype=np.float32)


def affine_coeff_to_matrix(ax, ay, w, h):
    """Convert normalized residual affine coefficients to 2x3 matrix.

    Model:
      dx = ax0 + ax1 * ((x-cx)/w) + ax2 * ((y-cy)/h)
      dy = ay0 + ay1 * ((x-cx)/w) + ay2 * ((y-cy)/h)
      x' = x + dx
      y' = y + dy
    """
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    ww = max(float(w), 1e-12)
    hh = max(float(h), 1e-12)

    ax0, ax1, ax2 = [float(v) for v in ax]
    ay0, ay1, ay2 = [float(v) for v in ay]

    A = np.array(
        [
            [
                1.0 + ax1 / ww,
                ax2 / hh,
                ax0 - ax1 * cx / ww - ax2 * cy / hh,
            ],
            [
                ay1 / ww,
                1.0 + ay2 / hh,
                ay0 - ay1 * cx / ww - ay2 * cy / hh,
            ],
        ],
        dtype=np.float32,
    )

    return A


def fit_affine_residual_robust_lstsq(
    samples,
    w,
    h,
    min_samples=12,
    robust_iters=2,
    outlier_percent=25.0,
):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1, 4)

    stats = {
        "num_samples_initial": int(samples.shape[0]),
        "num_samples_final": 0,
        "mean_fit_error": None,
        "median_fit_error": None,
        "max_fit_error": None,
    }

    if samples.shape[0] < int(min_samples):
        return None, None, False, stats

    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    ww = max(float(w), 1e-12)
    hh = max(float(h), 1e-12)

    work = samples.copy()
    ax = None
    ay = None
    err = None

    num_passes = max(1, int(robust_iters) + 1)

    for it in range(num_passes):
        x = work[:, 0].astype(np.float64)
        y = work[:, 1].astype(np.float64)
        dx = work[:, 2].astype(np.float64)
        dy = work[:, 3].astype(np.float64)

        nx = (x - cx) / ww
        ny = (y - cy) / hh
        X = np.stack([np.ones_like(nx), nx, ny], axis=1)

        ax, _, _, _ = np.linalg.lstsq(X, dx, rcond=None)
        ay, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)

        pred_dx = X @ ax
        pred_dy = X @ ay
        err = np.sqrt((pred_dx - dx) ** 2 + (pred_dy - dy) ** 2)

        if it >= num_passes - 1:
            break

        pct = float(outlier_percent)
        if pct <= 0.0:
            continue

        thr = np.percentile(err, max(0.0, min(100.0, 100.0 - pct)))
        keep = err <= thr

        if int(np.count_nonzero(keep)) < int(min_samples):
            break

        work = work[keep]

    if ax is None or ay is None or err is None:
        return None, None, False, stats

    stats["num_samples_final"] = int(work.shape[0])
    stats["mean_fit_error"] = float(np.mean(err))
    stats["median_fit_error"] = float(np.median(err))
    stats["max_fit_error"] = float(np.max(err))

    A = affine_coeff_to_matrix(ax, ay, w, h)
    coeff = {
        "ax": [float(v) for v in ax],
        "ay": [float(v) for v in ay],
    }

    return A.astype(np.float32), coeff, True, stats


def estimate_global_affine_bias_local_search(
    cur_y,
    ref_y,
    map_x,
    map_y,
    w,
    h,
    active_region,
    valid_erode=2,
    sample_step=16,
    search_range=4,
    block_radius=2,
    subpel_levels=0,
    max_sample_cost=-1.0,
    min_cur_var=0.0,
    min_samples=12,
    robust_iters=2,
    outlier_percent=25.0,
):
    """Estimate global residual affine without ECC.

    Method:
      valid active-region sparse sampling
      + local block matching around cam-proj map
      + robust least-squares affine fitting to residual MV samples.
    """
    valid_mask = make_valid_bool_mask(
        map_x,
        map_y,
        w,
        h,
        erode=valid_erode,
        active_region=active_region,
    )

    samples, costs = collect_residual_samples_local_search(
        cur_y=cur_y,
        ref_y=ref_y,
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid_mask,
        active_region=active_region,
        sample_step=sample_step,
        search_range=search_range,
        block_radius=block_radius,
        subpel_levels=subpel_levels,
        max_sample_cost=max_sample_cost,
        min_cur_var=min_cur_var,
    )

    A, coeff, ok, stats = fit_affine_residual_robust_lstsq(
        samples,
        w,
        h,
        min_samples=min_samples,
        robust_iters=robust_iters,
        outlier_percent=outlier_percent,
    )

    stats["estimator"] = "local_search_lstsq_no_ecc"
    stats["num_samples_collected"] = int(samples.shape[0])
    stats["sample_step"] = int(sample_step)
    stats["search_range"] = int(search_range)
    stats["block_radius"] = int(block_radius)
    stats["subpel_levels"] = int(subpel_levels)
    stats["max_sample_cost"] = None if float(max_sample_cost) < 0.0 else float(max_sample_cost)
    stats["min_cur_var"] = float(min_cur_var)
    stats["robust_iters"] = int(robust_iters)
    stats["outlier_percent"] = float(outlier_percent)
    stats["mean_sample_cost"] = float(np.mean(costs)) if costs.size > 0 else None
    stats["median_sample_cost"] = float(np.median(costs)) if costs.size > 0 else None
    stats["affine_coeff"] = coeff

    if not ok:
        return np.eye(2, 3, dtype=np.float32), False, stats

    return A.astype(np.float32), True, stats


def affine_cp_points(w, h, cp_num):
    """Return CP coordinates used for coding.

    cp_num=2 follows a 4-parameter affine/similarity-style convention:
      CP0=(0,0), CP1=(w,0), CP2 is derived.

    cp_num=3 follows full 6-parameter affine:
      CP0=(0,0), CP1=(w,0), CP2=(0,h).
    """
    if cp_num == 2:
        return np.array(
            [
                [0.0, 0.0],
                [float(w), 0.0],
            ],
            dtype=np.float32,
        )
    if cp_num == 3:
        return np.array(
            [
                [0.0, 0.0],
                [float(w), 0.0],
                [0.0, float(h)],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"cp_num must be 2 or 3, got {cp_num}")


def affine_matrix_to_cp_bias(A, w, h, cp_num=3):
    """Convert affine coordinate warp to coded CP bias.

    For cp_num=3, all three CP biases are extracted from the estimated affine.
    For cp_num=2, only CP0/CP1 are extracted.  The decoder reconstructs a
    constrained 4-parameter affine from those two CPs.

    bias = warped_cp - original_cp
    """
    src = affine_cp_points(w, h, cp_num)
    ones = np.ones((src.shape[0], 1), dtype=np.float32)
    src_homo = np.concatenate([src, ones], axis=1)

    dst = src_homo @ np.asarray(A, dtype=np.float32).T
    bias = dst - src

    return bias.astype(np.float32)


def cp_bias_to_affine_matrix(cp_bias, w, h, cp_num=3):
    """Reconstruct affine matrix from decoded CP bias.

    cp_num=3: full affine from CP0/CP1/CP2.
    cp_num=2: constrained 4-parameter model from CP0/CP1:
      x' = a*x - b*y + tx
      y' = b*x + a*y + ty
    where CP0 gives tx/ty and CP1 gives a/b.
    """
    cp_bias = np.asarray(cp_bias, dtype=np.float32).reshape(cp_num, 2)

    if cp_num == 3:
        src = affine_cp_points(w, h, 3)
        dst = src + cp_bias
        A = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        return A.astype(np.float32)

    if cp_num == 2:
        dx0, dy0 = float(cp_bias[0, 0]), float(cp_bias[0, 1])
        dx1, dy1 = float(cp_bias[1, 0]), float(cp_bias[1, 1])

        ww = max(float(w), 1e-12)
        tx = dx0
        ty = dy0

        # CP0 original=(0,0), warped=(dx0,dy0)
        # CP1 original=(w,0), warped=(w+dx1,dy1)
        # For x'=a*x-b*y+tx, y'=b*x+a*y+ty:
        a = (float(w) + dx1 - tx) / ww
        b = (dy1 - ty) / ww

        return np.array(
            [
                [a, -b, tx],
                [b,  a, ty],
            ],
            dtype=np.float32,
        )

    raise ValueError(f"cp_num must be 2 or 3, got {cp_num}")


def apply_affine_bias_to_map(map_x, map_y, A):
    """
    final_map = cam_proj_map + affine_bias(x,y)

    affine_bias is generated in current-picture coordinate domain.
    """
    h, w = map_x.shape

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    x2 = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]
    y2 = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]

    bias_x = x2 - xx
    bias_y = y2 - yy

    out_x = map_x + bias_x
    out_y = map_y + bias_y

    valid = (
        np.isfinite(out_x)
        & np.isfinite(out_y)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (out_x >= 0.0)
        & (out_x <= w - 1)
        & (out_y >= 0.0)
        & (out_y <= h - 1)
    )

    out_x = out_x.astype(np.float32)
    out_y = out_y.astype(np.float32)

    out_x[~valid] = -1.0
    out_y[~valid] = -1.0

    return out_x, out_y


def quantize_affine_cp_bias(cp_bias, step, bits, cp_num=3):
    """
    cp_bias: [cp_num,2] in pixel units.
    Components are coded with signed truncated Exp-Golomb.
    Bit estimate assumes 50:50 bin probability, so bin count = bit count.
    """
    cp_bias = np.asarray(cp_bias, dtype=np.float32).reshape(cp_num, 2)
    flat = cp_bias.reshape(cp_num * 2)

    q, dec, clipped = quant_s(flat, step=step, bits=bits)

    q_abs_max = signed_q_abs_max(bits)
    bits_each, bits_total = q_residual_bits_signed_trunc_exp_golomb(
        q,
        q_abs_max=q_abs_max,
    )

    return (
        q.astype(np.int32).reshape(cp_num, 2),
        dec.astype(np.float32).reshape(cp_num, 2),
        clipped.reshape(cp_num, 2),
        bits_each,
        int(bits_total),
    )


def affine_cp_component_names(cp_num):
    names = []
    for i in range(cp_num):
        names.append(f"cp{i}_dx")
        names.append(f"cp{i}_dy")
    return names


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
        import torch.nn.functional as F
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

    src_t = torch.from_numpy(src.astype(np.float32)).to(device)
    src_t = src_t.view(1, 1, h, w)

    src_pad = F.pad(src_t, (2, 3, 2, 3), mode="replicate")

    patches_all = F.unfold(src_pad, kernel_size=(6, 6), stride=1)

    col_idx = (iy * w + ix).reshape(-1)
    patches = patches_all[0, :, col_idx]

    coeff = torch.from_numpy(LUMA_6TAP_32_NP).to(device=device, dtype=torch.float32)

    cx = coeff[frac_x.reshape(-1)]
    cy = coeff[frac_y.reshape(-1)]

    weight = (cy[:, :, None] * cx[:, None, :]).reshape(-1, 36)

    val = torch.sum(patches.transpose(0, 1) * weight, dim=1)

    val = torch.round(val / 65536.0)
    val = val.clamp(0, maxv)

    val = val.reshape(h, w)
    val = torch.where(valid, val, torch.zeros_like(val))

    out = val.detach().cpu().numpy()

    if bit_depth <= 8:
        return out.astype(np.uint8)

    return out.astype(np.dtype("<u2"))


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

    ap.add_argument("--pred-n", type=int, default=3)
    ap.add_argument("--pred-degree", type=int, default=2)

    # User-requested defaults
    ap.add_argument("--ext-bits", type=int, default=16)
    ap.add_argument("--r-step", type=float, default=1.0 / 16.0)
    ap.add_argument("--t-step-norm", type=float, default=1.0 / 4.0)

    # Intrinsic delta coding
    ap.add_argument("--intr-delta-bits", type=int, default=16)
    ap.add_argument(
        "--intr-step",
        type=float,
        default=1.0 / 16.0,
        help=(
            "Intrinsic delta quantization step in pixel units. "
            "q_intr_delta = round(delta / intr_step). Default: 1/16 pixel."
        ),
    )

    ap.add_argument("--intr-f-max", type=float, default=4.0)
    ap.add_argument("--intr-c-min", type=float, default=-1.0)
    ap.add_argument("--intr-c-max", type=float, default=2.0)

    ap.add_argument(
        "--depth-scale-bits",
        type=int,
        default=16,
        help=(
            "Bit estimate for fixed-point depth_scale integer. "
            "depth_scale_precision is assumed known or fixed by design."
        ),
    )

    ap.add_argument(
        "--warp-filter",
        choices=["bilinear", "subblk4_6tap_torch"],
        default="bilinear",
    )

    ap.add_argument(
        "--torch-device",
        default=None,
        help="cuda, cpu, cuda:0, etc. Used only for subblk4_6tap_torch.",
    )

    # Global affine bias over cam-proj flow.
    ap.add_argument("--global-affine-bias", action="store_true")
    ap.add_argument(
        "--affine-cp-num",
        type=int,
        choices=[2, 3],
        default=3,
        help=(
            "Number of affine control points to code. "
            "3 = full 6-parameter affine, 2 = constrained 4-parameter affine using CP0/CP1."
        ),
    )
    ap.add_argument(
        "--affine-cp-step",
        type=float,
        default=1.0 / 16.0,
        help="Affine CP bias quantization step in pixel units.",
    )
    ap.add_argument(
        "--affine-cp-bits",
        type=int,
        default=16,
        help="Signed truncated Exp-Golomb range for affine CP bias.",
    )
    ap.add_argument(
        "--affine-valid-erode",
        type=int,
        default=2,
        help="Erode valid active-region mask before affine local-search sampling.",
    )
    ap.add_argument(
        "--affine-sample-step",
        type=int,
        default=16,
        help="Sparse sampling step for no-ECC local-search affine estimation.",
    )
    ap.add_argument(
        "--affine-search-range",
        type=int,
        default=4,
        help="Integer-pel residual search range around cam-proj position.",
    )
    ap.add_argument(
        "--affine-block-radius",
        type=int,
        default=2,
        help="Patch radius for local SAD matching. 2 means 5x5 block.",
    )
    ap.add_argument(
        "--affine-subpel-levels",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="0: integer only, 1: add half-pel refinement, 2: add half+quarter refinement.",
    )
    ap.add_argument(
        "--affine-max-sample-cost",
        type=float,
        default=-1.0,
        help="Reject local samples whose average SAD cost exceeds this. Negative disables.",
    )
    ap.add_argument(
        "--affine-min-cur-var",
        type=float,
        default=0.0,
        help="Reject local samples with current patch variance below this threshold.",
    )
    ap.add_argument(
        "--affine-min-samples",
        type=int,
        default=12,
        help="Minimum residual samples required for affine fitting.",
    )
    ap.add_argument(
        "--affine-robust-iters",
        type=int,
        default=2,
        help="Number of outlier-rejection refit iterations.",
    )
    ap.add_argument(
        "--affine-outlier-percent",
        type=float,
        default=25.0,
        help="Percentage of largest fitting-error samples removed at each robust iteration.",
    )

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.r_step <= 0:
        raise ValueError("--r-step must be positive")
    if args.t_step_norm <= 0:
        raise ValueError("--t-step-norm must be positive")
    if args.intr_step <= 0:
        raise ValueError("--intr-step must be positive")
    if args.affine_cp_step <= 0:
        raise ValueError("--affine-cp-step must be positive")
    if args.affine_cp_bits < 2:
        raise ValueError("--affine-cp-bits must be >= 2")
    if args.affine_sample_step <= 0:
        raise ValueError("--affine-sample-step must be positive")
    if args.affine_search_range < 0:
        raise ValueError("--affine-search-range must be non-negative")
    if args.affine_block_radius < 0:
        raise ValueError("--affine-block-radius must be non-negative")
    if args.affine_min_samples < 3:
        raise ValueError("--affine-min-samples must be >= 3")
    if args.affine_robust_iters < 0:
        raise ValueError("--affine-robust-iters must be non-negative")
    if args.affine_outlier_percent < 0 or args.affine_outlier_percent >= 100:
        raise ValueError("--affine-outlier-percent must be in [0, 100)")

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_jsonl = Path(args.param_jsonl)

    out_yuv = Path(args.out_yuv)
    out_q_jsonl = Path(args.out_q_jsonl)

    for p in [out_yuv, out_q_jsonl]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}")

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

    # Header first-frame intrinsic absolute coding.
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

    max_poc = min(
        available_seq_count,
        depth_count,
        max(frames.keys()) + 1,
    )

    if max_poc <= 0:
        raise RuntimeError(
            f"No frames to process: available_seq_count={available_seq_count}, "
            f"depth_count={depth_count}, param_frames={len(frames)}"
        )

    decoded_hist = []

    q_abs_max_ext = signed_q_abs_max(args.ext_bits)
    q_abs_max_intr = signed_q_abs_max(args.intr_delta_bits)
    q_abs_max_affine_cp = signed_q_abs_max(args.affine_cp_bits)

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

    total_affine_flag_bits = 0
    total_affine_cp_bits = 0
    total_affine_frames = 0
    total_affine_success_frames = 0
    total_affine_clipped_frames = 0
    total_affine_bits_each = np.zeros(args.affine_cp_num * 2, dtype=np.int64)

    total_coded_frames = 0
    total_clipped_frames = 0

    sum_mae_active = 0.0
    sum_mae_coded = 0.0
    sum_psnr_active = 0.0
    sum_psnr_coded = 0.0
    psnr_count_active = 0
    psnr_count_coded = 0

    sum_psnr_active_cam_only = 0.0
    psnr_count_active_cam_only = 0

    ys_active, xs_active = active_slice(src_w, src_h, pad_left, pad_top)

    decoded_intrinsics = [intr_dec0]

    print("============================================================")
    print("Input summary")
    print("============================================================")
    print(f"seq_yuv               : {seq_yuv}")
    print(f"depth_yuv             : {depth_yuv}")
    print(f"param_jsonl           : {param_jsonl}")
    print(f"seq frames total      : {seq_count}")
    print(f"seq_start             : {args.seq_start}")
    print(f"depth frames          : {depth_count}")
    print(f"param max poc         : {max(frames.keys())}")
    print(f"process frames        : {max_poc}")
    print(f"depth_scale header    : {header['depth_scale']}")
    print(f"depth_scale_precision : {depth_scale_precision}")
    print(f"depth_scale_real      : {depth_scale_real}")
    print(f"pose_mode             : {pose_mode}")
    print(f"warp_filter           : {args.warp_filter}")
    print(f"global affine bias    : {args.global_affine_bias}")
    print("affine estimator      : local_search_lstsq_no_ecc")
    print("------------------------------------------------------------")
    print(f"ext-bits              : {args.ext_bits}")
    print(f"r-step                : {args.r_step}")
    print(f"t-step-norm           : {args.t_step_norm}")
    print(f"intr-delta-bits       : {args.intr_delta_bits}")
    print(f"intr-step             : {args.intr_step}")
    print(f"affine-cp-num         : {args.affine_cp_num}")
    print(f"affine-cp-bits        : {args.affine_cp_bits}")
    print(f"affine-cp-step        : {args.affine_cp_step}")
    print(f"affine-sample-step    : {args.affine_sample_step}")
    print(f"affine-search-range   : {args.affine_search_range}")
    print(f"affine-block-radius   : {args.affine_block_radius}")
    print(f"affine-subpel-levels  : {args.affine_subpel_levels}")
    print("============================================================")

    with open(out_q_jsonl, "w", encoding="utf-8") as fq:
        fq.write(json.dumps({
            "type": "header",

            "source_size": {
                "width": src_w,
                "height": src_h,
            },
            "coded_size": {
                "width": coded_w,
                "height": coded_h,
            },
            "padding": {
                "left": pad_left,
                "top": pad_top,
                "right": pad_right,
                "bottom": pad_bottom,
            },

            "seq_start": int(args.seq_start),

            "projection_mode": "fast_pixel_coordinate_no_ndc_dual_intrinsic",
            "warp_filter": args.warp_filter,
            "precompute": [
                "target: x_norm=(x-cx_tar)/fx_tar",
                "target: y_norm=(y-cy_tar)/fy_tar",
                "reference: fx_ref,fy_ref,cx_ref,cy_ref cached",
            ],
            "depth_padding": "edge",
            "image_padding": "edge",

            "depth_scale_header": header["depth_scale"],
            "depth_scale_precision": depth_scale_precision,
            "depth_scale_real": depth_scale_real,
            "depth_decode_formula": (
                "depth_linear = depth_y * depth_scale / depth_scale_precision"
                if depth_scale_precision is not None
                else "depth_linear = depth_y * depth_scale"
            ),

            "intrinsic_gt_original0": intr_gt_original0,
            "intrinsic_gt_padded0": intr_gt_padded0,
            "intrinsic_q16_first": intr_q0,
            "intrinsic_dec_first": intr_dec0,
            "intrinsic_first_clipped": intr_clip0,

            "intrinsic_delta_code": "signed_truncated_exp_golomb",
            "intrinsic_delta_bits": args.intr_delta_bits,
            "intrinsic_delta_q_abs_max": q_abs_max_intr,
            "intrinsic_delta_q_range": [-q_abs_max_intr, q_abs_max_intr],
            "intrinsic_delta_step": args.intr_step,
            "intrinsic_delta_order": ["dfx", "dfy", "dcx", "dcy"],
            "intrinsic_decode": (
                "intrinsic_dec[poc] = intrinsic_dec[poc-1] "
                "+ q_intrinsic_delta[poc] * intrinsic_delta_step"
            ),

            "extrinsic_bits": args.ext_bits,
            "extrinsic_q_abs_max": q_abs_max_ext,
            "extrinsic_q_range": [-q_abs_max_ext, q_abs_max_ext],
            "r_step": args.r_step,
            "t_step_norm": args.t_step_norm,
            "pred_n": args.pred_n,
            "pred_degree": args.pred_degree,

            "global_affine_bias": {
                "enabled": bool(args.global_affine_bias),
                "mode": "final_map = cam_proj_map + affine_bias(x,y)",
                "estimator": "local_search_lstsq_no_ecc",
                "fit_region": "valid cam-proj pixels inside active source region only",
                "cp_num": int(args.affine_cp_num),
                "cp_convention": (
                    ["CP0=(0,0)", "CP1=(coded_w,0)"]
                    if args.affine_cp_num == 2
                    else ["CP0=(0,0)", "CP1=(coded_w,0)", "CP2=(0,coded_h)"]
                ),
                "cp_num_note": (
                    "2CP reconstructs constrained 4-parameter affine; CP2 is derived from CP0/CP1"
                    if args.affine_cp_num == 2
                    else "3CP reconstructs full 6-parameter affine"
                ),
                "cp_step": args.affine_cp_step,
                "cp_bits": args.affine_cp_bits,
                "cp_q_abs_max": q_abs_max_affine_cp,
                "bit_model": "signed truncated Exp-Golomb, 50:50 bin probability, 1 bin = 1 bit",
                "local_search": {
                    "sample_step": args.affine_sample_step,
                    "search_range": args.affine_search_range,
                    "block_radius": args.affine_block_radius,
                    "subpel_levels": args.affine_subpel_levels,
                    "valid_erode": args.affine_valid_erode,
                    "max_sample_cost": None if args.affine_max_sample_cost < 0.0 else args.affine_max_sample_cost,
                    "min_cur_var": args.affine_min_cur_var,
                    "min_samples": args.affine_min_samples,
                    "robust_iters": args.affine_robust_iters,
                    "outlier_percent": args.affine_outlier_percent,
                },
            },

            "bit_count": {
                "header_intrinsic_bits": header_intrinsic_bits,
                "depth_scale_bits": depth_scale_bits,
                "z_sign_bits": z_sign_bits,
                "header_bits": header_bits,
                "extrinsic_code": "signed_truncated_exp_golomb",
                "intrinsic_delta_code": "signed_truncated_exp_golomb",
                "affine_cp_code": "signed_truncated_exp_golomb_50_50_bit_estimate",
            },

            "param6_order": [
                "rx",
                "ry",
                "rz",
                "tx_over_depth_scale_real",
                "ty_over_depth_scale_real",
                "tz_over_depth_scale_real",
            ],
        }, ensure_ascii=False) + "\n")

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

                fq.write(json.dumps({
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

                    "global_affine_bias": {
                        "enabled": bool(args.global_affine_bias),
                        "success": False,
                        "flag_bits_50_50": 0,
                        "cp_bits_total_50_50": 0,
                    },

                    "mae_y_active": 0.0,
                    "mae_y_coded": 0.0,
                    "psnr_y_active": "inf",
                    "psnr_y_coded": "inf",
                }, ensure_ascii=False) + "\n")

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

            # ------------------------------------------------------------
            # Intrinsic delta coding
            # ------------------------------------------------------------
            intr_delta_gt = np.array(frame["intrinsic_delta"], dtype=np.float32).reshape(4)

            q_intr, d_intr, clip_intr = quantize_intrinsic_delta_4(
                intr_delta_gt,
                step=args.intr_step,
                bits=args.intr_delta_bits,
            )

            q_intr_bits_each, q_intr_bits_total = q_residual_bits_signed_trunc_exp_golomb(
                q_intr,
                q_abs_max=q_abs_max_intr,
            )

            total_intr_delta_bits += q_intr_bits_total
            total_intr_delta_bits_each += np.array(q_intr_bits_each, dtype=np.int64)
            total_intr_delta_frames += 1

            intrinsic_clipped = bool(np.any(clip_intr))
            if intrinsic_clipped:
                total_intr_clipped_frames += 1

            intr_dec_cur = add_intrinsic_delta(
                decoded_intrinsics[-1],
                d_intr,
            )

            decoded_intrinsics.append(intr_dec_cur)

            intr_dec_ref = decoded_intrinsics[poc - 1]
            intr_dec_tar = decoded_intrinsics[poc]

            # ------------------------------------------------------------
            # Extrinsic coding
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Depth
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Projection with dual intrinsic
            # target: current intrinsic
            # ref   : previous intrinsic
            # ------------------------------------------------------------
            projection_precomp = make_projection_precompute_dual(
                coded_w,
                coded_h,
                intr_tar=intr_dec_tar,
                intr_ref=intr_dec_ref,
            )

            map_x, map_y = backward_map_fast_pixel_coord_dual(
                depth_linear=depth_linear_pad,
                precomp=projection_precomp,
                rt=rt_dec,
            )

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

            # ------------------------------------------------------------
            # Cam-proj only warp first. This is used both as baseline metric
            # and as input for global affine residual estimation.
            # ------------------------------------------------------------
            if args.warp_filter == "bilinear":
                wy_cam, wu_cam, wv_cam = backward_warp_yuv420_bilinear(
                    prev_y_pad,
                    prev_u_pad,
                    prev_v_pad,
                    map_x,
                    map_y,
                    bit_depth,
                )
            elif args.warp_filter == "subblk4_6tap_torch":
                wy_cam, wu_cam, wv_cam = backward_warp_yuv420_subblk4_6tap_torch(
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

            psnr_y_active_cam_only = calc_psnr(
                wy_cam[ys_active, xs_active],
                cur_y_pad[ys_active, xs_active],
                bit_depth,
            )

            if np.isfinite(psnr_y_active_cam_only):
                sum_psnr_active_cam_only += psnr_y_active_cam_only
                psnr_count_active_cam_only += 1

            # ------------------------------------------------------------
            # Optional global affine bias over cam-proj flow.
            # ------------------------------------------------------------
            affine_enabled = bool(args.global_affine_bias)
            affine_success = False
            affine_est_stats = {}
            affine_flag_bits = 0
            affine_cp_bits_total = 0
            affine_cp_bits_each = [0] * (args.affine_cp_num * 2)
            affine_cp_q = np.zeros((args.affine_cp_num, 2), dtype=np.int32)
            affine_cp_dec = np.zeros((args.affine_cp_num, 2), dtype=np.float32)
            affine_cp_clipped = False
            affine_matrix_est = np.eye(2, 3, dtype=np.float32)
            affine_matrix_dec = np.eye(2, 3, dtype=np.float32)

            map_x_final = map_x
            map_y_final = map_y

            if affine_enabled:
                # One flag per coded frame after POC0. 50:50 assumption => 1 bit.
                affine_flag_bits = 1
                total_affine_flag_bits += affine_flag_bits
                total_affine_frames += 1

                affine_matrix_est, affine_success, affine_est_stats = estimate_global_affine_bias_local_search(
                    cur_y=cur_y_pad,
                    ref_y=prev_y_pad,
                    map_x=map_x,
                    map_y=map_y,
                    w=coded_w,
                    h=coded_h,
                    active_region=(ys_active, xs_active),
                    valid_erode=args.affine_valid_erode,
                    sample_step=args.affine_sample_step,
                    search_range=args.affine_search_range,
                    block_radius=args.affine_block_radius,
                    subpel_levels=args.affine_subpel_levels,
                    max_sample_cost=args.affine_max_sample_cost,
                    min_cur_var=args.affine_min_cur_var,
                    min_samples=args.affine_min_samples,
                    robust_iters=args.affine_robust_iters,
                    outlier_percent=args.affine_outlier_percent,
                )

                if affine_success:
                    total_affine_success_frames += 1

                    cp_bias_est = affine_matrix_to_cp_bias(
                        affine_matrix_est,
                        coded_w,
                        coded_h,
                        cp_num=args.affine_cp_num,
                    )

                    (
                        affine_cp_q,
                        affine_cp_dec,
                        affine_cp_clip_arr,
                        affine_cp_bits_each,
                        affine_cp_bits_total,
                    ) = quantize_affine_cp_bias(
                        cp_bias_est,
                        step=args.affine_cp_step,
                        bits=args.affine_cp_bits,
                        cp_num=args.affine_cp_num,
                    )

                    affine_cp_clipped = bool(np.any(affine_cp_clip_arr))

                    if affine_cp_clipped:
                        total_affine_clipped_frames += 1

                    total_affine_cp_bits += affine_cp_bits_total
                    total_affine_bits_each += np.array(
                        affine_cp_bits_each,
                        dtype=np.int64,
                    )

                    affine_matrix_dec = cp_bias_to_affine_matrix(
                        affine_cp_dec,
                        coded_w,
                        coded_h,
                        cp_num=args.affine_cp_num,
                    )

                    map_x_final, map_y_final = apply_affine_bias_to_map(
                        map_x,
                        map_y,
                        affine_matrix_dec,
                    )

            # ------------------------------------------------------------
            # Final warp.
            # ------------------------------------------------------------
            if args.warp_filter == "bilinear":
                wy, wu, wv = backward_warp_yuv420_bilinear(
                    prev_y_pad,
                    prev_u_pad,
                    prev_v_pad,
                    map_x_final,
                    map_y_final,
                    bit_depth,
                )
            elif args.warp_filter == "subblk4_6tap_torch":
                wy, wu, wv = backward_warp_yuv420_subblk4_6tap_torch(
                    prev_y_pad,
                    prev_u_pad,
                    prev_v_pad,
                    map_x_final,
                    map_y_final,
                    bit_depth,
                    torch_device=args.torch_device,
                )
            else:
                raise ValueError(args.warp_filter)

            write_yuv420(out_yuv, wy, wu, wv)

            mae_y_coded = float(np.mean(np.abs(
                wy.astype(np.float32) - cur_y_pad.astype(np.float32)
            )))

            mae_y_active = float(np.mean(np.abs(
                wy[ys_active, xs_active].astype(np.float32)
                - cur_y_pad[ys_active, xs_active].astype(np.float32)
            )))

            psnr_y_coded = calc_psnr(
                wy,
                cur_y_pad,
                bit_depth,
            )

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

            fq.write(json.dumps({
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

                "global_affine_bias": {
                    "enabled": affine_enabled,
                    "cp_num": int(args.affine_cp_num),
                    "success": affine_success,
                    "estimator": "local_search_lstsq_no_ecc",
                    "estimator_stats": affine_est_stats,
                    "flag_bits_50_50": int(affine_flag_bits),
                    "cp_q": affine_cp_q.astype(int).tolist(),
                    "cp_dec": affine_cp_dec.astype(float).tolist(),
                    "cp_bits_each_50_50": affine_cp_bits_each,
                    "cp_bits_total_50_50": int(affine_cp_bits_total),
                    "cp_clipped": bool(affine_cp_clipped),
                    "matrix_est": affine_matrix_est.astype(float).tolist(),
                    "matrix_dec": affine_matrix_dec.astype(float).tolist(),
                    "psnr_y_active_cam_only": json_safe_float(psnr_y_active_cam_only),
                    "psnr_y_active_final": json_safe_float(psnr_y_active),
                },

                "mae_y_active": mae_y_active,
                "mae_y_coded": mae_y_coded,
                "psnr_y_active": json_safe_float(psnr_y_active),
                "psnr_y_coded": json_safe_float(psnr_y_coded),
            }, ensure_ascii=False) + "\n")

            print(
                f"[{poc:04d}/{max_poc - 1:04d}] "
                f"seq_idx={seq_idx}, "
                f"Y-PSNR-cam={psnr_y_active_cam_only:.3f} dB, "
                f"Y-PSNR-active={psnr_y_active:.3f} dB, "
                f"Y-PSNR-coded={psnr_y_coded:.3f} dB, "
                f"Y-MAE-active={mae_y_active:.3f}, "
                f"Y-MAE-coded={mae_y_coded:.3f}, "
                f"ext_clip={clipped}, "
                f"intr_clip={intrinsic_clipped}, "
                f"affine={affine_success}, "
                f"affine_bits={affine_flag_bits + affine_cp_bits_total}, "
                f"ext_bits={q_bits_total}, "
                f"intr_bits={q_intr_bits_total}"
            )

    total_bits = (
        header_bits
        + total_ext_bits
        + total_intr_delta_bits
        + total_affine_flag_bits
        + total_affine_cp_bits
    )

    print("============================================================")
    print("Padding / projection summary")
    print("============================================================")
    print(f"source size           : {src_w}x{src_h}")
    print(f"coded size            : {coded_w}x{coded_h}")
    print(
        f"padding               : "
        f"L={pad_left}, T={pad_top}, "
        f"R={pad_right}, B={pad_bottom}"
    )
    print(f"seq_start             : {args.seq_start}")
    print("projection            : fast pixel-coordinate, dual intrinsic, no NDC")
    print(f"warp filter           : {args.warp_filter}")
    print("image/depth padding   : edge")
    print("------------------------------------------------------------")
    print("Depth scale")
    print("------------------------------------------------------------")
    print(f"depth_scale header    : {header['depth_scale']}")
    print(f"depth_scale precision : {depth_scale_precision}")
    print(f"depth_scale real      : {depth_scale_real}")
    print("decoder formula       : depth_linear = depth_y * depth_scale_real")
    print("------------------------------------------------------------")
    print("Intrinsic")
    print("------------------------------------------------------------")
    print(f"first intrinsic gt    : {intr_gt_original0}")
    print(f"first intrinsic padded: {intr_gt_padded0}")
    print(f"first intrinsic dec   : {intr_dec0}")
    print(f"first intrinsic clip  : {intr_clip0}")
    print(f"intrinsic delta step  : {args.intr_step}")
    print(f"intrinsic delta bits  : {args.intr_delta_bits}")
    print(f"intrinsic q range     : [-{q_abs_max_intr}, {q_abs_max_intr}]")
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
        avg_psnr_cam_only = (
            sum_psnr_active_cam_only / psnr_count_active_cam_only
            if psnr_count_active_cam_only > 0
            else float("inf")
        )

        print(f"avg MAE active        : {avg_mae_active:.3f}")
        print(f"avg MAE coded         : {avg_mae_coded:.3f}")
        print(f"avg PSNR cam-only act : {avg_psnr_cam_only:.3f} dB")
        print(f"avg PSNR active       : {avg_psnr_active:.3f} dB")
        print(f"avg PSNR coded        : {avg_psnr_coded:.3f} dB")
        if np.isfinite(avg_psnr_cam_only) and np.isfinite(avg_psnr_active):
            print(f"avg affine gain active: {avg_psnr_active - avg_psnr_cam_only:+.3f} dB")

    print("============================================================")
    print("Bit summary")
    print("============================================================")
    print(f"header intrinsic bits : {header_intrinsic_bits} bits")
    print(f"  fx, fy, cx, cy      : 16 bits each")
    print(f"depth_scale bits      : {depth_scale_bits} bits")
    print(f"z_sign bits           : {z_sign_bits} bits")
    print(f"header bits           : {header_bits} bits")
    print("------------------------------------------------------------")
    print(f"extrinsic code        : signed truncated Exp-Golomb")
    print(f"extrinsic q range     : [-{q_abs_max_ext}, {q_abs_max_ext}]")
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

    print("------------------------------------------------------------")
    print(f"intrinsic delta code  : signed truncated Exp-Golomb")
    print(f"intrinsic delta range : [-{q_abs_max_intr}, {q_abs_max_intr}]")
    print(f"intrinsic delta frames: {total_intr_delta_frames}")
    print(f"intrinsic clipped     : {total_intr_clipped_frames}")
    print(f"intrinsic delta bits  : {total_intr_delta_bits} bits")

    if total_intr_delta_frames > 0:
        avg_intr_bits = total_intr_delta_bits / total_intr_delta_frames
        avg_intr_each = total_intr_delta_bits_each.astype(np.float64) / total_intr_delta_frames

        print(f"avg intr bits/frame   : {avg_intr_bits:.3f}")
        print(
            "avg intr bits each    : "
            f"dfx={avg_intr_each[0]:.3f}, "
            f"dfy={avg_intr_each[1]:.3f}, "
            f"dcx={avg_intr_each[2]:.3f}, "
            f"dcy={avg_intr_each[3]:.3f}"
        )

    print("------------------------------------------------------------")
    print("Global affine bias")
    print("------------------------------------------------------------")
    print(f"affine enabled        : {args.global_affine_bias}")
    print(f"affine frames         : {total_affine_frames}")
    print(f"affine success frames : {total_affine_success_frames}")
    print(f"affine clipped frames : {total_affine_clipped_frames}")
    print(f"affine flag bits      : {total_affine_flag_bits} bits")
    print(f"affine CP bits        : {total_affine_cp_bits} bits")
    print(f"affine total bits     : {total_affine_flag_bits + total_affine_cp_bits} bits")
    print(f"affine cp num         : {args.affine_cp_num}")
    print(f"affine cp step        : {args.affine_cp_step}")
    print(f"affine cp bits range  : [-{q_abs_max_affine_cp}, {q_abs_max_affine_cp}]")
    print("affine estimator      : local_search_lstsq_no_ecc")
    print(f"affine sample step    : {args.affine_sample_step}")
    print(f"affine search range   : {args.affine_search_range}")
    print(f"affine block radius   : {args.affine_block_radius}")
    print(f"affine subpel levels  : {args.affine_subpel_levels}")
    print(f"affine robust iters   : {args.affine_robust_iters}")
    print(f"affine outlier pct    : {args.affine_outlier_percent}")

    if total_affine_success_frames > 0:
        avg_affine_cp_bits = total_affine_cp_bits / total_affine_success_frames
        avg_affine_total_bits = (
            total_affine_flag_bits + total_affine_cp_bits
        ) / max(total_affine_frames, 1)
        avg_affine_each = total_affine_bits_each.astype(np.float64) / total_affine_success_frames

        print(f"avg affine CP bits/frame    : {avg_affine_cp_bits:.3f}")
        print(f"avg affine total bits/frame : {avg_affine_total_bits:.3f}")
        avg_affine_parts = [
            f"{name}={val:.3f}"
            for name, val in zip(affine_cp_component_names(args.affine_cp_num), avg_affine_each)
        ]
        print("avg affine bits each        : " + ", ".join(avg_affine_parts))

    print("------------------------------------------------------------")
    print(f"total bits            : {total_bits} bits")
    print("============================================================")
    print("Done.")
    print(f"warped yuv            : {out_yuv}")
    print(f"q jsonl               : {out_q_jsonl}")


if __name__ == "__main__":
    main()
