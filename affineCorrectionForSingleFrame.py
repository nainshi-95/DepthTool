#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame ref->target camera-projection warp test.

This script is for testing arbitrary ref/target pairs, e.g. ref=0 -> tar=16.
It optionally applies the same non-ECC vectorized LK/SSD global affine bias
used in the sequence simulation. No cv2.findTransformECC() is used.

Flow:
  1) Read ref frame, target frame, target depth, and camera parameters.
  2) Build target->reference projection map.
  3) Warp reference frame to target frame.
  4) Optional: estimate global affine bias from cam-warped Y to target Y
     using valid active-region pixels only.
  5) Quantize/decode affine CP bias, apply it to the map, and warp again.
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

def normalize_for_ecc(img, bit_depth):
    x = img.astype(np.float32)
    maxv = float((1 << bit_depth) - 1)
    return np.clip(x / maxv, 0.0, 1.0).astype(np.float32)


def make_valid_u8_mask(map_x, map_y, w, h, erode=0, active_region=None):
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

    mask = (valid.astype(np.uint8) * 255)

    if erode > 0:
        k = 2 * int(erode) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def bilinear_sample_vectorized(img, xs, ys):
    """Vectorized bilinear sampling for float32 grayscale images."""
    h, w = img.shape

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    valid = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & (xs >= 0.0)
        & (xs <= w - 1)
        & (ys >= 0.0)
        & (ys <= h - 1)
    )

    xs_safe = np.clip(xs, 0.0, float(w - 1))
    ys_safe = np.clip(ys, 0.0, float(h - 1))

    x0 = np.floor(xs_safe).astype(np.int32)
    y0 = np.floor(ys_safe).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    fx = xs_safe - x0.astype(np.float32)
    fy = ys_safe - y0.astype(np.float32)

    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]

    a = p00 * (1.0 - fx) + p01 * fx
    b = p10 * (1.0 - fx) + p11 * fx
    out = a * (1.0 - fy) + b * fy

    out = out.astype(np.float32)
    out[~valid] = 0.0

    return out, valid


def params_to_affine_matrix_lk(p, w, h):
    """
    LK parameterization:
      x' = x + p0 + p1*xn + p2*yn
      y' = y + p3 + p4*xn + p5*yn
      xn = (x-cx)/w, yn = (y-cy)/h

    Return 2x3 matrix A such that [x',y'] = A*[x,y,1].
    """
    p = np.asarray(p, dtype=np.float64).reshape(6)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    ww = max(float(w), 1e-12)
    hh = max(float(h), 1e-12)

    return np.array(
        [
            [1.0 + p[1] / ww, p[2] / hh, p[0] - p[1] * cx / ww - p[2] * cy / hh],
            [p[4] / ww, 1.0 + p[5] / hh, p[3] - p[4] * cx / ww - p[5] * cy / hh],
        ],
        dtype=np.float32,
    )


def estimate_global_affine_bias_lk(
    cur_y,
    cam_warp_y,
    valid_mask_u8,
    bit_depth,
    max_iters=30,
    eps=1e-4,
    sample_step=4,
    normalize="zero_mean",
    damping=1e-6,
    max_update=4.0,
    robust_iters=0,
    outlier_percent=0.0,
    min_samples=128,
    min_keep_ratio=0.25,
):
    """
    Fast non-ECC affine estimator with optional robust outlier rejection.

    Objective:
      align cam-proj-only warped Y to current Y using a vectorized
      forward-additive Lucas-Kanade / SSD affine fitting loop.

    Robust loop:
      1) fit affine with current inlier set
      2) compute per-sample residual under fitted affine
      3) remove the largest residual samples by percentile
      4) repeat fitting

    This is useful for large POC distance where occlusion, object motion,
    depth error, and projection holes produce unstable samples.

    Returns:
      A, success, score, stats
    where score is negative RMSE after fitting, so larger is better.
    """
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if min_keep_ratio <= 0.0 or min_keep_ratio > 1.0:
        raise ValueError("min_keep_ratio must be in (0,1]")

    mask = valid_mask_u8 > 0
    if sample_step > 1:
        sample_mask = np.zeros_like(mask, dtype=bool)
        sample_mask[::sample_step, ::sample_step] = True
        mask &= sample_mask

    ys_i, xs_i = np.nonzero(mask)
    total_initial = int(xs_i.size)
    if total_initial < int(min_samples):
        stats = {
            "num_samples_initial": total_initial,
            "num_samples_final": 0,
            "robust_iters_requested": int(robust_iters),
            "robust_iters_done": 0,
            "reason": "too_few_initial_samples",
        }
        return np.eye(2, 3, dtype=np.float32), False, None, stats

    maxv = float((1 << bit_depth) - 1)
    T_img = np.clip(cur_y.astype(np.float32) / maxv, 0.0, 1.0)
    I_img = np.clip(cam_warp_y.astype(np.float32) / maxv, 0.0, 1.0)

    # Central-difference gradients of input image. This maps cleanly to C++.
    gy, gx = np.gradient(I_img)
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)

    xs = xs_i.astype(np.float32)
    ys = ys_i.astype(np.float32)
    T_all = T_img[ys_i, xs_i].astype(np.float32)

    h, w = cur_y.shape
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    xn = ((xs - cx) / max(float(w), 1e-12)).astype(np.float32)
    yn = ((ys - cy) / max(float(h), 1e-12)).astype(np.float32)

    def sample_with_params(p):
        xw = xs + float(p[0]) + float(p[1]) * xn + float(p[2]) * yn
        yw = ys + float(p[3]) + float(p[4]) * xn + float(p[5]) * yn
        I, valid_i = bilinear_sample_vectorized(I_img, xw, yw)
        Ix, valid_x = bilinear_sample_vectorized(gx, xw, yw)
        Iy, valid_y = bilinear_sample_vectorized(gy, xw, yw)
        valid = valid_i & valid_x & valid_y
        return I, Ix, Iy, valid

    def make_residual(Tv, Iv):
        if normalize == "none":
            e = Tv - Iv
            scale = 1.0
        elif normalize == "zero_mean":
            e = (Tv - np.mean(Tv)) - (Iv - np.mean(Iv))
            scale = 1.0
        elif normalize == "zncc_approx":
            Tv0 = Tv - np.mean(Tv)
            Iv0 = Iv - np.mean(Iv)
            std_t = float(np.std(Tv0)) + 1e-6
            std_i = float(np.std(Iv0)) + 1e-6
            e = Tv0 / std_t - Iv0 / std_i
            scale = 1.0 / std_i
        else:
            raise ValueError(f"Unknown LK normalize mode: {normalize}")
        return e.astype(np.float32), float(scale)

    def compute_errors(p, keep_mask):
        I, _, _, valid_warp = sample_with_params(p)
        valid = keep_mask & valid_warp
        if np.count_nonzero(valid) < int(min_samples):
            return None, valid
        e, _scale = make_residual(T_all[valid], I[valid])
        err = np.full(xs.shape[0], np.inf, dtype=np.float32)
        err[valid] = np.abs(e).astype(np.float32)
        return err, valid

    def fit_once(p_init, keep_mask):
        p_cur = np.asarray(p_init, dtype=np.float64).reshape(6).copy()
        final_rmse = None
        iters_done = 0
        num_valid_last = 0

        for it in range(int(max_iters)):
            I, Ix, Iy, valid_warp = sample_with_params(p_cur)
            valid = keep_mask & valid_warp
            num_valid = int(np.count_nonzero(valid))
            num_valid_last = num_valid
            if num_valid < int(min_samples):
                break

            Tv = T_all[valid]
            Iv = I[valid]
            Ixv = Ix[valid]
            Iyv = Iy[valid]
            xnv = xn[valid]
            ynv = yn[valid]

            e, scale = make_residual(Tv, Iv)

            # J is dI/dp. Linearized residual: e_new = e - J*dp.
            J = np.empty((e.size, 6), dtype=np.float32)
            J[:, 0] = Ixv * scale
            J[:, 1] = Ixv * xnv * scale
            J[:, 2] = Ixv * ynv * scale
            J[:, 3] = Iyv * scale
            J[:, 4] = Iyv * xnv * scale
            J[:, 5] = Iyv * ynv * scale

            H = (J.T @ J).astype(np.float64)
            b = (J.T @ e.astype(np.float32)).astype(np.float64)
            H += np.eye(6, dtype=np.float64) * float(damping)

            try:
                dp = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break

            dp_norm = float(np.linalg.norm(dp))
            if not np.isfinite(dp_norm):
                break

            if max_update is not None and max_update > 0 and dp_norm > float(max_update):
                dp *= float(max_update) / dp_norm
                dp_norm = float(max_update)

            p_cur += dp
            final_rmse = float(np.sqrt(np.mean(e.astype(np.float64) ** 2)))
            iters_done = it + 1

            if dp_norm < float(eps):
                break

        return p_cur, final_rmse, iters_done, num_valid_last

    p = np.zeros(6, dtype=np.float64)
    keep = np.ones(xs.shape[0], dtype=bool)

    robust_iters = max(0, int(robust_iters))
    outlier_percent = float(outlier_percent)
    outlier_percent = max(0.0, min(outlier_percent, 95.0))

    history = []
    final_rmse = None
    final_lk_iters = 0
    final_valid = 0

    # Number of robust rounds.  robust_iters=0 means one normal LK fit only.
    num_rounds = robust_iters + 1
    for r in range(num_rounds):
        p, final_rmse, lk_iters, num_valid = fit_once(p, keep)
        final_lk_iters += int(lk_iters)
        final_valid = int(num_valid)

        err, valid_for_err = compute_errors(p, keep)
        if err is None:
            history.append({
                "round": int(r),
                "num_keep_before": int(np.count_nonzero(keep)),
                "num_valid": int(np.count_nonzero(valid_for_err)),
                "rmse": json_safe_float(final_rmse),
                "reason": "too_few_valid_for_error",
            })
            break

        finite_err = err[np.isfinite(err)]
        mean_abs = float(np.mean(finite_err)) if finite_err.size else None
        median_abs = float(np.median(finite_err)) if finite_err.size else None
        p90_abs = float(np.percentile(finite_err, 90.0)) if finite_err.size else None

        history.append({
            "round": int(r),
            "num_keep_before": int(np.count_nonzero(keep)),
            "num_valid": int(finite_err.size),
            "rmse": json_safe_float(final_rmse),
            "mean_abs_residual": json_safe_float(mean_abs),
            "median_abs_residual": json_safe_float(median_abs),
            "p90_abs_residual": json_safe_float(p90_abs),
        })

        # Last round: do not remove more outliers.
        if r >= robust_iters or outlier_percent <= 0.0:
            break

        valid_idx = np.flatnonzero(np.isfinite(err))
        if valid_idx.size < int(min_samples):
            break

        keep_ratio = max(float(min_keep_ratio), 1.0 - outlier_percent / 100.0)
        num_keep_target = int(round(float(valid_idx.size) * keep_ratio))
        num_keep_target = max(int(min_samples), min(num_keep_target, int(valid_idx.size)))

        # Keep lowest residual samples. Deterministic and C++-friendly.
        order = np.argsort(err[valid_idx], kind="stable")
        selected = valid_idx[order[:num_keep_target]]
        new_keep = np.zeros_like(keep, dtype=bool)
        new_keep[selected] = True

        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    A = params_to_affine_matrix_lk(p, w, h)

    if final_rmse is None or not np.isfinite(final_rmse) or final_valid < int(min_samples):
        stats = {
            "num_samples_initial": total_initial,
            "num_samples_final": int(np.count_nonzero(keep)),
            "num_valid_final": int(final_valid),
            "robust_iters_requested": int(robust_iters),
            "robust_iters_done": max(0, len(history) - 1),
            "lk_iters_total": int(final_lk_iters),
            "history": history,
            "reason": "fit_failed_or_too_few_final_samples",
        }
        return np.eye(2, 3, dtype=np.float32), False, None, stats

    score = -float(final_rmse)
    stats = {
        "num_samples_initial": total_initial,
        "num_samples_final": int(np.count_nonzero(keep)),
        "num_valid_final": int(final_valid),
        "robust_iters_requested": int(robust_iters),
        "robust_iters_done": max(0, len(history) - 1),
        "outlier_percent": float(outlier_percent),
        "min_keep_ratio": float(min_keep_ratio),
        "min_samples": int(min_samples),
        "lk_iters_total": int(final_lk_iters),
        "final_rmse": float(final_rmse),
        "history": history,
    }
    return A.astype(np.float32), True, float(score), stats


def affine_local_matrix_to_global(A_local, x0, y0):
    """Convert an affine matrix estimated on a cropped block coordinate system
    into the full-picture coordinate system.

    local:  [xl', yl'] = A_local * [xl, yl, 1]
    global: x = xl + x0, y = yl + y0
    """
    A = np.asarray(A_local, dtype=np.float64).reshape(2, 3)
    x0 = float(x0)
    y0 = float(y0)

    A_global = np.empty((2, 3), dtype=np.float32)
    A_global[0, 0] = A[0, 0]
    A_global[0, 1] = A[0, 1]
    A_global[0, 2] = x0 + A[0, 2] - A[0, 0] * x0 - A[0, 1] * y0
    A_global[1, 0] = A[1, 0]
    A_global[1, 1] = A[1, 1]
    A_global[1, 2] = y0 + A[1, 2] - A[1, 0] * x0 - A[1, 1] * y0
    return A_global.astype(np.float32)


def affine_bias_at_points(A, pts):
    """Return residual bias [x'-x, y'-y] at arbitrary global points."""
    A = np.asarray(A, dtype=np.float64).reshape(2, 3)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    src = np.concatenate([pts, ones], axis=1)
    dst = src @ A.T
    return (dst - pts).astype(np.float64)


def block_affine_sample_points(x0, y0, x1, y1):
    """Five representative points for comparing/fitting block affine models.

    x1/y1 are exclusive block boundaries, so using them is consistent with
    picture-level CP convention CP1=(w,0), CP2=(0,h).
    """
    x0 = float(x0)
    y0 = float(y0)
    x1 = float(x1)
    y1 = float(y1)
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x0, y1],
            [x1, y1],
            [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
        ],
        dtype=np.float64,
    )


def fit_global_affine_from_block_models(blocks, keep_mask, w, h, weight_mode="equal"):
    """Fit one global affine residual bias to local block affine models.

    Each local block affine contributes bias samples at block corners and center.
    The fitted model is the same normalized 6-parameter residual model used by LK:
      dx = p0 + p1*xn + p2*yn
      dy = p3 + p4*xn + p5*yn
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if len(blocks) == 0 or np.count_nonzero(keep_mask) < 3:
        return np.eye(2, 3, dtype=np.float32), None

    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    ww = max(float(w), 1e-12)
    hh = max(float(h), 1e-12)

    rows = []
    dxs = []
    dys = []
    weights = []

    for bi, b in enumerate(blocks):
        if not keep_mask[bi]:
            continue

        pts = b["fit_points"]
        bias = b["fit_bias"]

        xn = (pts[:, 0] - cx) / ww
        yn = (pts[:, 1] - cy) / hh
        basis = np.stack([np.ones_like(xn), xn, yn], axis=1)

        if weight_mode == "equal":
            # Equal block weight regardless of block valid-sample count.
            wt = np.full(pts.shape[0], 1.0 / max(pts.shape[0], 1), dtype=np.float64)
        elif weight_mode == "valid_count":
            # More valid/reliable blocks receive higher weight, but avoid extreme dominance.
            wt_block = np.sqrt(max(float(b.get("num_valid", 1)), 1.0))
            wt = np.full(pts.shape[0], wt_block / max(pts.shape[0], 1), dtype=np.float64)
        else:
            raise ValueError(f"Unknown affine block weight mode: {weight_mode}")

        rows.append(basis)
        dxs.append(bias[:, 0])
        dys.append(bias[:, 1])
        weights.append(wt)

    if not rows:
        return np.eye(2, 3, dtype=np.float32), None

    X = np.concatenate(rows, axis=0).astype(np.float64)
    dx = np.concatenate(dxs, axis=0).astype(np.float64)
    dy = np.concatenate(dys, axis=0).astype(np.float64)
    wt = np.concatenate(weights, axis=0).astype(np.float64)
    sw = np.sqrt(np.maximum(wt, 1e-12))

    Xw = X * sw[:, None]
    dxw = dx * sw
    dyw = dy * sw

    try:
        coef_x, _, _, _ = np.linalg.lstsq(Xw, dxw, rcond=None)
        coef_y, _, _, _ = np.linalg.lstsq(Xw, dyw, rcond=None)
    except np.linalg.LinAlgError:
        return np.eye(2, 3, dtype=np.float32), None

    p = np.array(
        [coef_x[0], coef_x[1], coef_x[2], coef_y[0], coef_y[1], coef_y[2]],
        dtype=np.float64,
    )
    A = params_to_affine_matrix_lk(p, w, h)
    return A.astype(np.float32), p


def compute_block_model_errors(blocks, keep_mask, A_global):
    keep_mask = np.asarray(keep_mask, dtype=bool)
    errors = np.full(len(blocks), np.inf, dtype=np.float64)

    for bi, b in enumerate(blocks):
        if not keep_mask[bi]:
            continue
        pts = b["fit_points"]
        local_bias = b["fit_bias"]
        global_bias = affine_bias_at_points(A_global, pts)
        diff = global_bias - local_bias
        l2 = np.sqrt(np.sum(diff * diff, axis=1))
        errors[bi] = float(np.sqrt(np.mean(l2 * l2)))

    return errors


def estimate_blockwise_global_affine_bias_lk(
    cur_y,
    cam_warp_y,
    valid_mask_u8,
    bit_depth,
    block_size=128,
    block_sample_step=8,
    block_lk_iters=20,
    block_lk_eps=1e-4,
    normalize="zero_mean",
    damping=1e-6,
    max_update=4.0,
    min_block_samples=64,
    min_block_valid_ratio=0.05,
    robust_iters=3,
    outlier_percent=20.0,
    remove_worst_count=0,
    min_keep_blocks=8,
    weight_mode="equal",
):
    """Estimate one global affine using block-wise local affine candidates.

    Procedure:
      1) Split valid active region into non-overlapping block_size x block_size blocks.
      2) Estimate a local LK affine for each block independently.
      3) Fit a single global affine that best covers all local block affine models.
      4) Remove block-affine outliers with the largest model disagreement.
      5) Refit the global affine.

    This is intended for large ref->target POC distance where a direct global LK
    fit can be destabilized by occlusion/object motion/depth errors.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_sample_step <= 0:
        raise ValueError("block_sample_step must be positive")
    if min_block_samples <= 0:
        raise ValueError("min_block_samples must be positive")
    if min_keep_blocks <= 0:
        raise ValueError("min_keep_blocks must be positive")

    h, w = cur_y.shape
    valid_bool = valid_mask_u8 > 0
    blocks = []
    total_blocks = 0
    skipped_too_few = 0
    skipped_failed = 0

    for y0 in range(0, h, int(block_size)):
        y1 = min(y0 + int(block_size), h)
        for x0 in range(0, w, int(block_size)):
            x1 = min(x0 + int(block_size), w)
            total_blocks += 1

            block_valid = valid_bool[y0:y1, x0:x1]
            valid_count = int(np.count_nonzero(block_valid))
            area = max((x1 - x0) * (y1 - y0), 1)
            valid_ratio = float(valid_count) / float(area)
            if valid_count < int(min_block_samples) or valid_ratio < float(min_block_valid_ratio):
                skipped_too_few += 1
                continue

            A_local, ok, score, st = estimate_global_affine_bias_lk(
                cur_y=cur_y[y0:y1, x0:x1],
                cam_warp_y=cam_warp_y[y0:y1, x0:x1],
                valid_mask_u8=valid_mask_u8[y0:y1, x0:x1],
                bit_depth=bit_depth,
                max_iters=block_lk_iters,
                eps=block_lk_eps,
                sample_step=block_sample_step,
                normalize=normalize,
                damping=damping,
                max_update=max_update,
                robust_iters=0,
                outlier_percent=0.0,
                min_samples=min_block_samples,
                min_keep_ratio=1.0,
            )

            if not ok:
                skipped_failed += 1
                continue

            A_global_local = affine_local_matrix_to_global(A_local, x0, y0)
            pts = block_affine_sample_points(x0, y0, x1, y1)
            bias = affine_bias_at_points(A_global_local, pts)

            blocks.append({
                "block_index": int(len(blocks)),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "num_valid": int(valid_count),
                "valid_ratio": float(valid_ratio),
                "local_score_negative_rmse": json_safe_float(score),
                "local_stats": {
                    "num_samples_initial": st.get("num_samples_initial"),
                    "num_samples_final": st.get("num_samples_final"),
                    "num_valid_final": st.get("num_valid_final"),
                    "final_rmse": st.get("final_rmse"),
                    "lk_iters_total": st.get("lk_iters_total"),
                },
                "A_local_global_coords": A_global_local.astype(float).tolist(),
                "fit_points": pts,
                "fit_bias": bias,
            })

    n = len(blocks)
    if n < int(min_keep_blocks):
        stats = {
            "mode": "block_lk",
            "total_blocks": int(total_blocks),
            "local_success_blocks": int(n),
            "skipped_too_few": int(skipped_too_few),
            "skipped_failed": int(skipped_failed),
            "reason": "too_few_successful_blocks",
        }
        return np.eye(2, 3, dtype=np.float32), False, None, stats

    keep = np.ones(n, dtype=bool)
    robust_iters = max(0, int(robust_iters))
    outlier_percent = max(0.0, min(float(outlier_percent), 95.0))
    remove_worst_count = max(0, int(remove_worst_count))

    history = []
    A_global = np.eye(2, 3, dtype=np.float32)
    errors = np.full(n, np.inf, dtype=np.float64)

    for r in range(robust_iters + 1):
        A_global, p_global = fit_global_affine_from_block_models(
            blocks,
            keep,
            w,
            h,
            weight_mode=weight_mode,
        )
        if p_global is None:
            break

        errors = compute_block_model_errors(blocks, keep, A_global)
        finite = errors[np.isfinite(errors)]
        kept_before = int(np.count_nonzero(keep))
        if finite.size == 0:
            break

        hist = {
            "round": int(r),
            "kept_blocks_before": int(kept_before),
            "mean_block_model_error": float(np.mean(finite)),
            "median_block_model_error": float(np.median(finite)),
            "p90_block_model_error": float(np.percentile(finite, 90.0)),
            "max_block_model_error": float(np.max(finite)),
        }

        worst_order = np.argsort(errors)[::-1]
        top = []
        for bi in worst_order[:min(8, worst_order.size)]:
            if not np.isfinite(errors[bi]):
                continue
            b = blocks[int(bi)]
            top.append({
                "block_index": int(bi),
                "x0": int(b["x0"]),
                "y0": int(b["y0"]),
                "x1": int(b["x1"]),
                "y1": int(b["y1"]),
                "error": float(errors[bi]),
            })
        hist["worst_blocks"] = top
        history.append(hist)

        if r >= robust_iters:
            break

        valid_idx = np.flatnonzero(keep & np.isfinite(errors))
        if valid_idx.size <= int(min_keep_blocks):
            break

        if remove_worst_count > 0:
            remove_count = min(remove_worst_count, max(0, valid_idx.size - int(min_keep_blocks)))
        else:
            remove_count = int(np.ceil(valid_idx.size * outlier_percent / 100.0))
            remove_count = max(1 if outlier_percent > 0.0 else 0, remove_count)
            remove_count = min(remove_count, max(0, valid_idx.size - int(min_keep_blocks)))

        if remove_count <= 0:
            break

        order = valid_idx[np.argsort(errors[valid_idx])[::-1]]
        remove_idx = order[:remove_count]
        keep[remove_idx] = False
        history[-1]["removed_blocks"] = [int(x) for x in remove_idx.tolist()]

    kept_final = int(np.count_nonzero(keep))
    if kept_final < int(min_keep_blocks):
        stats = {
            "mode": "block_lk",
            "total_blocks": int(total_blocks),
            "local_success_blocks": int(n),
            "kept_final": int(kept_final),
            "history": history,
            "reason": "too_few_final_blocks",
        }
        return np.eye(2, 3, dtype=np.float32), False, None, stats

    final_errors = compute_block_model_errors(blocks, keep, A_global)
    finite_final = final_errors[np.isfinite(final_errors)]
    final_rmse = float(np.sqrt(np.mean(finite_final * finite_final))) if finite_final.size else None
    score = -final_rmse if final_rmse is not None else None

    # Keep detailed block model list compact enough for JSON by omitting raw fit_points/fit_bias.
    block_summaries = []
    for bi, b in enumerate(blocks):
        block_summaries.append({
            "block_index": int(bi),
            "x0": int(b["x0"]),
            "y0": int(b["y0"]),
            "x1": int(b["x1"]),
            "y1": int(b["y1"]),
            "num_valid": int(b["num_valid"]),
            "valid_ratio": float(b["valid_ratio"]),
            "kept_final": bool(keep[bi]),
            "final_model_error": json_safe_float(final_errors[bi]),
            "local_score_negative_rmse": b.get("local_score_negative_rmse"),
        })

    stats = {
        "mode": "block_lk",
        "block_size": int(block_size),
        "block_sample_step": int(block_sample_step),
        "block_lk_iters": int(block_lk_iters),
        "block_lk_eps": float(block_lk_eps),
        "min_block_samples": int(min_block_samples),
        "min_block_valid_ratio": float(min_block_valid_ratio),
        "weight_mode": weight_mode,
        "total_blocks": int(total_blocks),
        "local_success_blocks": int(n),
        "skipped_too_few": int(skipped_too_few),
        "skipped_failed": int(skipped_failed),
        "robust_iters_requested": int(robust_iters),
        "outlier_percent": float(outlier_percent),
        "remove_worst_count": int(remove_worst_count),
        "min_keep_blocks": int(min_keep_blocks),
        "kept_final": int(kept_final),
        "final_block_rmse": json_safe_float(final_rmse),
        "history": history,
        "blocks": block_summaries,
    }

    return A_global.astype(np.float32), True, json_safe_float(score), stats

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
# Pose helpers for arbitrary ref -> target
# ============================================================

def _as_extrinsic_matrix(frame):
    """Return camera_from_world [R|t] if absolute extrinsic is available."""
    for key in ["extrinsic_abs", "extrinsic", "camera_extrinsic", "cam_from_world"]:
        if key in frame:
            E = np.asarray(frame[key], dtype=np.float64)
            if E.shape == (4, 4):
                E = E[:3, :4]
            elif E.shape == (3, 4):
                pass
            elif E.size == 12:
                E = E.reshape(3, 4)
            elif E.size == 16:
                E = E.reshape(4, 4)[:3, :4]
            else:
                continue
            return E

    if "rvec_abs" in frame and "tvec_abs" in frame:
        rvec = np.asarray(frame["rvec_abs"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(frame["tvec_abs"], dtype=np.float64).reshape(3)
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        E = np.concatenate([R.astype(np.float64), tvec.reshape(3, 1)], axis=1)
        return E

    return None


def _frame_relative_rt_current_to_previous(frame):
    """Return R,t for X_prev = R * X_cur + t."""
    if "rt_dec" in frame:
        rt = frame["rt_dec"]
        rvec = np.asarray(rt["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(rt["tvec"], dtype=np.float64).reshape(3)
    elif "rvec" in frame and "tvec" in frame:
        rvec = np.asarray(frame["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(frame["tvec"], dtype=np.float64).reshape(3)
    else:
        raise RuntimeError(
            "Frame has no relative pose. Expected either rt_dec or rvec/tvec."
        )

    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R.astype(np.float64), tvec.astype(np.float64)


def _invert_rt(R, t):
    Ri = R.T
    ti = -Ri @ t
    return Ri, ti


def _compose_rt(R2, t2, R1, t1):
    """Compose X2 = R2*(R1*X + t1)+t2."""
    return R2 @ R1, R2 @ t1 + t2


def get_target_to_reference_rt(frames, ref_idx, tar_idx):
    """Return rt dict mapping target camera coordinates to reference camera coordinates.

    Preferred path:
      absolute camera_from_world extrinsic for both frames.

    Fallback:
      compose adjacent current_to_previous transforms. This supports arbitrary
      ref/tar order as long as all intermediate relative poses exist.
    """
    ref_idx = int(ref_idx)
    tar_idx = int(tar_idx)

    if ref_idx == tar_idx:
        return {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0], "source": "identity"}

    f_ref = frames.get(ref_idx, {})
    f_tar = frames.get(tar_idx, {})

    E_ref = _as_extrinsic_matrix(f_ref)
    E_tar = _as_extrinsic_matrix(f_tar)

    if E_ref is not None and E_tar is not None:
        R_ref = E_ref[:, :3]
        t_ref = E_ref[:, 3]
        R_tar = E_tar[:, :3]
        t_tar = E_tar[:, 3]

        # camera_from_world convention:
        #   X_cam = R * X_world + t
        # target -> reference:
        #   X_ref = R_ref * R_tar^T * X_tar + t_ref - R_ref * R_tar^T * t_tar
        R = R_ref @ R_tar.T
        t = t_ref - R @ t_tar
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        return {
            "rvec": rvec.reshape(3).astype(float).tolist(),
            "tvec": t.reshape(3).astype(float).tolist(),
            "source": "absolute_extrinsic",
        }

    # Fallback: compose current_to_previous transforms.
    if tar_idx > ref_idx:
        R_tot = np.eye(3, dtype=np.float64)
        t_tot = np.zeros(3, dtype=np.float64)
        for p in range(tar_idx, ref_idx, -1):
            if p not in frames:
                raise RuntimeError(f"Missing frame {p} for relative pose composition")
            R_p, t_p = _frame_relative_rt_current_to_previous(frames[p])
            R_tot, t_tot = _compose_rt(R_p, t_p, R_tot, t_tot)
    else:
        # Compose ref -> tar, then invert to get tar -> ref.
        R_fwd = np.eye(3, dtype=np.float64)
        t_fwd = np.zeros(3, dtype=np.float64)
        for p in range(ref_idx, tar_idx, -1):
            if p not in frames:
                raise RuntimeError(f"Missing frame {p} for relative pose composition")
            R_p, t_p = _frame_relative_rt_current_to_previous(frames[p])
            R_fwd, t_fwd = _compose_rt(R_p, t_p, R_fwd, t_fwd)
        R_tot, t_tot = _invert_rt(R_fwd, t_fwd)

    rvec, _ = cv2.Rodrigues(R_tot.astype(np.float64))
    return {
        "rvec": rvec.reshape(3).astype(float).tolist(),
        "tvec": t_tot.reshape(3).astype(float).tolist(),
        "source": "composed_current_to_previous",
    }


def _maybe_padded_intrinsic(intr, pad_left, pad_top, assume_already_padded=False):
    if assume_already_padded:
        return {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "z_sign": float(intr.get("z_sign", 1.0)),
        }
    return make_padded_intrinsic_from_original(intr, pad_left, pad_top)


def build_frame_intrinsics(header, frames, max_idx, pad_left, pad_top):
    """Build per-frame intrinsics for raw JSONL or q-jsonl outputs."""
    intrs = {}

    if "intrinsic_dec_first" in header:
        intr0 = _maybe_padded_intrinsic(header["intrinsic_dec_first"], pad_left, pad_top, True)
    elif "intrinsic_gt_padded0" in header:
        intr0 = _maybe_padded_intrinsic(header["intrinsic_gt_padded0"], pad_left, pad_top, True)
    elif "intrinsic" in header:
        # Header intrinsic is assumed original image coordinate unless q-jsonl already says otherwise.
        intr0 = make_padded_intrinsic_from_original(header["intrinsic"], pad_left, pad_top)
    else:
        raise RuntimeError("No intrinsic information found in JSONL header")

    intrs[0] = intr0

    for i in range(1, int(max_idx) + 1):
        f = frames.get(i, {})

        if "intrinsic_tar_dec" in f:
            intrs[i] = _maybe_padded_intrinsic(f["intrinsic_tar_dec"], pad_left, pad_top, True)
        elif "intrinsic_dec" in f:
            intrs[i] = _maybe_padded_intrinsic(f["intrinsic_dec"], pad_left, pad_top, True)
        elif "intrinsic" in f:
            intrs[i] = make_padded_intrinsic_from_original(f["intrinsic"], pad_left, pad_top)
        elif "intrinsic_delta" in f:
            delta = np.asarray(f["intrinsic_delta"], dtype=np.float32).reshape(4)
            intrs[i] = add_intrinsic_delta(intrs[i - 1], delta)
        else:
            intrs[i] = intrs[i - 1].copy()

    return intrs


def write_mask_yuv420(path, mask_y, bit_depth):
    """Write a single-frame YUV420 mask for visualization."""
    maxv = (1 << bit_depth) - 1
    y = np.where(mask_y > 0, maxv, 0).astype(np.uint8 if bit_depth <= 8 else np.dtype("<u2"))
    h, w = y.shape
    neutral = 128 if bit_depth <= 8 else 512
    dtype = np.uint8 if bit_depth <= 8 else np.dtype("<u2")
    u = np.full((h // 2, w // 2), neutral, dtype=dtype)
    v = np.full((h // 2, w // 2), neutral, dtype=dtype)
    write_yuv420(path, y, u, v)

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
    ap.add_argument("--bit-depth", type=int, default=10)

    ap.add_argument("--ref-idx", type=int, required=True, help="Reference frame index, e.g. 0")
    ap.add_argument("--tar-idx", type=int, required=True, help="Target/current frame index, e.g. 16")
    ap.add_argument(
        "--seq-start",
        type=int,
        default=0,
        help="Frame offset in seq-yuv. Actual YUV index = seq_start + ref/tar idx.",
    )

    ap.add_argument("--coded-width", type=int, default=None)
    ap.add_argument("--coded-height", type=int, default=None)
    ap.add_argument("--pad-left", type=int, default=0)
    ap.add_argument("--pad-top", type=int, default=0)

    ap.add_argument("--out-yuv", required=True, help="Final warped ref->target YUV")
    ap.add_argument("--out-json", required=True, help="Single-frame warp stats JSON")
    ap.add_argument("--out-cam-yuv", default=None, help="Optional cam-proj-only warped YUV")
    ap.add_argument("--out-target-yuv", default=None, help="Optional padded target frame YUV")
    ap.add_argument("--out-mask-yuv", default=None, help="Optional valid mask YUV")

    ap.add_argument(
        "--warp-filter",
        choices=["bilinear", "subblk4_6tap_torch"],
        default="bilinear",
    )
    ap.add_argument("--torch-device", default=None)

    # Affine bias: off by default, same decoded application path as sequence test.
    ap.add_argument("--global-affine-bias", action="store_true")
    ap.add_argument(
        "--affine-estimator",
        choices=["lk", "block_lk"],
        default="block_lk",
        help="lk: direct global LK. block_lk: estimate local block affines then robustly aggregate one global affine.",
    )
    ap.add_argument("--affine-cp-num", type=int, choices=[2, 3], default=3)
    ap.add_argument("--affine-cp-step", type=float, default=1.0)
    ap.add_argument("--affine-cp-bits", type=int, default=16)
    ap.add_argument("--affine-valid-erode", type=int, default=2)
    ap.add_argument("--affine-lk-iters", type=int, default=30)
    ap.add_argument("--affine-lk-eps", type=float, default=1e-4)
    ap.add_argument("--affine-lk-sample-step", type=int, default=4)
    ap.add_argument(
        "--affine-lk-normalize",
        choices=["none", "zero_mean", "zncc_approx"],
        default="zero_mean",
    )
    ap.add_argument("--affine-lk-damping", type=float, default=1e-6)
    ap.add_argument("--affine-lk-max-update", type=float, default=4.0)
    ap.add_argument(
        "--affine-lk-robust-iters",
        type=int,
        default=0,
        help="Number of robust outlier-removal rounds after LK fitting.",
    )
    ap.add_argument(
        "--affine-lk-outlier-percent",
        type=float,
        default=0.0,
        help="Percent of largest residual LK samples removed per robust round.",
    )
    ap.add_argument(
        "--affine-lk-min-samples",
        type=int,
        default=128,
        help="Minimum number of LK samples required after outlier removal.",
    )
    ap.add_argument(
        "--affine-lk-min-keep-ratio",
        type=float,
        default=0.25,
        help="Lower bound on sample keep ratio during robust outlier removal.",
    )

    # Block-wise local affine aggregation estimator.
    ap.add_argument("--affine-block-size", type=int, default=128)
    ap.add_argument("--affine-block-sample-step", type=int, default=8)
    ap.add_argument("--affine-block-lk-iters", type=int, default=20)
    ap.add_argument("--affine-block-lk-eps", type=float, default=1e-4)
    ap.add_argument("--affine-block-min-samples", type=int, default=64)
    ap.add_argument("--affine-block-min-valid-ratio", type=float, default=0.05)
    ap.add_argument("--affine-block-robust-iters", type=int, default=3)
    ap.add_argument("--affine-block-outlier-percent", type=float, default=20.0)
    ap.add_argument(
        "--affine-block-remove-worst-count",
        type=int,
        default=0,
        help="If >0, remove this many worst local-affine blocks per robust round instead of using percent.",
    )
    ap.add_argument("--affine-block-min-keep-blocks", type=int, default=8)
    ap.add_argument(
        "--affine-block-weight-mode",
        choices=["equal", "valid_count"],
        default="equal",
        help="How to weight local block affine models when fitting the final global affine.",
    )

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.ref_idx == args.tar_idx:
        raise ValueError("--ref-idx and --tar-idx must be different")
    if args.seq_start < 0:
        raise ValueError("--seq-start must be non-negative")
    if args.affine_cp_step <= 0:
        raise ValueError("--affine-cp-step must be positive")
    if args.affine_cp_bits < 2:
        raise ValueError("--affine-cp-bits must be >= 2")
    if args.affine_lk_sample_step <= 0:
        raise ValueError("--affine-lk-sample-step must be positive")
    if args.affine_lk_robust_iters < 0:
        raise ValueError("--affine-lk-robust-iters must be non-negative")
    if args.affine_lk_outlier_percent < 0.0 or args.affine_lk_outlier_percent >= 100.0:
        raise ValueError("--affine-lk-outlier-percent must be in [0,100)")
    if args.affine_lk_min_samples <= 0:
        raise ValueError("--affine-lk-min-samples must be positive")
    if args.affine_lk_min_keep_ratio <= 0.0 or args.affine_lk_min_keep_ratio > 1.0:
        raise ValueError("--affine-lk-min-keep-ratio must be in (0,1]")
    if args.affine_block_size <= 0:
        raise ValueError("--affine-block-size must be positive")
    if args.affine_block_sample_step <= 0:
        raise ValueError("--affine-block-sample-step must be positive")
    if args.affine_block_lk_iters <= 0:
        raise ValueError("--affine-block-lk-iters must be positive")
    if args.affine_block_min_samples <= 0:
        raise ValueError("--affine-block-min-samples must be positive")
    if args.affine_block_min_valid_ratio < 0.0 or args.affine_block_min_valid_ratio > 1.0:
        raise ValueError("--affine-block-min-valid-ratio must be in [0,1]")
    if args.affine_block_robust_iters < 0:
        raise ValueError("--affine-block-robust-iters must be non-negative")
    if args.affine_block_outlier_percent < 0.0 or args.affine_block_outlier_percent >= 100.0:
        raise ValueError("--affine-block-outlier-percent must be in [0,100)")
    if args.affine_block_remove_worst_count < 0:
        raise ValueError("--affine-block-remove-worst-count must be non-negative")
    if args.affine_block_min_keep_blocks <= 0:
        raise ValueError("--affine-block-min-keep-blocks must be positive")

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_jsonl = Path(args.param_jsonl)

    out_paths = [Path(args.out_yuv), Path(args.out_json)]
    for opt in [args.out_cam_yuv, args.out_target_yuv, args.out_mask_yuv]:
        if opt is not None:
            out_paths.append(Path(opt))

    for p in out_paths:
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

    coded_w = int(args.coded_width) if args.coded_width is not None else align_to(src_w + pad_left, 4)
    coded_h = int(args.coded_height) if args.coded_height is not None else align_to(src_h + pad_top, 4)

    pad_right, pad_bottom = calc_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top)
    validate_yuv420_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top, pad_right, pad_bottom)

    header, frames = load_param_jsonl(param_jsonl)
    depth_scale_real = get_depth_scale_real_from_header(header)
    depth_scale_precision = get_depth_scale_precision_from_header(header)
    if depth_scale_real <= 0:
        raise ValueError(f"Invalid depth_scale_real: {depth_scale_real}")

    max_idx = max(int(args.ref_idx), int(args.tar_idx), max(frames.keys()))
    intrs = build_frame_intrinsics(header, frames, max_idx, pad_left, pad_top)
    intr_ref = intrs[int(args.ref_idx)]
    intr_tar = intrs[int(args.tar_idx)]

    rt_tar_to_ref = get_target_to_reference_rt(frames, int(args.ref_idx), int(args.tar_idx))

    seq_count = count_frames(seq_yuv, src_w, src_h, bit_depth)
    depth_count = count_frames(depth_yuv, src_w, src_h, 10)

    ref_seq_idx = args.seq_start + args.ref_idx
    tar_seq_idx = args.seq_start + args.tar_idx

    if ref_seq_idx < 0 or ref_seq_idx >= seq_count:
        raise RuntimeError(f"ref_seq_idx={ref_seq_idx} outside seq-yuv frame count {seq_count}")
    if tar_seq_idx < 0 or tar_seq_idx >= seq_count:
        raise RuntimeError(f"tar_seq_idx={tar_seq_idx} outside seq-yuv frame count {seq_count}")
    if args.tar_idx < 0 or args.tar_idx >= depth_count:
        raise RuntimeError(f"tar_idx={args.tar_idx} outside depth-yuv frame count {depth_count}")

    ys_active, xs_active = active_slice(src_w, src_h, pad_left, pad_top)

    ref_y, ref_u, ref_v = read_yuv420(seq_yuv, ref_seq_idx, src_w, src_h, bit_depth)
    tar_y, tar_u, tar_v = read_yuv420(seq_yuv, tar_seq_idx, src_w, src_h, bit_depth)

    ref_y_pad, ref_u_pad, ref_v_pad = pad_yuv420_edge(ref_y, ref_u, ref_v, coded_w, coded_h, pad_left, pad_top)
    tar_y_pad, tar_u_pad, tar_v_pad = pad_yuv420_edge(tar_y, tar_u, tar_v, coded_w, coded_h, pad_left, pad_top)

    if args.out_target_yuv is not None:
        write_yuv420(Path(args.out_target_yuv), tar_y_pad, tar_u_pad, tar_v_pad)

    depth_y, _, _ = read_yuv420(depth_yuv, args.tar_idx, src_w, src_h, 10)
    depth_linear = depth_y.astype(np.float32) * float(depth_scale_real)
    depth_linear_pad = pad_2d_edge(depth_linear, coded_w, coded_h, pad_left, pad_top).astype(np.float32)

    projection_precomp = make_projection_precompute_dual(
        coded_w,
        coded_h,
        intr_tar=intr_tar,
        intr_ref=intr_ref,
    )

    map_x, map_y = backward_map_fast_pixel_coord_dual(
        depth_linear=depth_linear_pad,
        precomp=projection_precomp,
        rt=rt_tar_to_ref,
    )

    valid_mask_u8 = make_valid_u8_mask(
        map_x,
        map_y,
        coded_w,
        coded_h,
        erode=args.affine_valid_erode,
        active_region=(ys_active, xs_active),
    )

    if args.out_mask_yuv is not None:
        write_mask_yuv420(Path(args.out_mask_yuv), valid_mask_u8, bit_depth)

    if args.warp_filter == "bilinear":
        wy_cam, wu_cam, wv_cam = backward_warp_yuv420_bilinear(
            ref_y_pad, ref_u_pad, ref_v_pad, map_x, map_y, bit_depth
        )
    elif args.warp_filter == "subblk4_6tap_torch":
        wy_cam, wu_cam, wv_cam = backward_warp_yuv420_subblk4_6tap_torch(
            ref_y_pad, ref_u_pad, ref_v_pad, map_x, map_y, bit_depth, torch_device=args.torch_device
        )
    else:
        raise ValueError(args.warp_filter)

    if args.out_cam_yuv is not None:
        write_yuv420(Path(args.out_cam_yuv), wy_cam, wu_cam, wv_cam)

    psnr_y_active_cam_only = calc_psnr(wy_cam[ys_active, xs_active], tar_y_pad[ys_active, xs_active], bit_depth)
    psnr_y_coded_cam_only = calc_psnr(wy_cam, tar_y_pad, bit_depth)
    mae_y_active_cam_only = float(np.mean(np.abs(
        wy_cam[ys_active, xs_active].astype(np.float32) - tar_y_pad[ys_active, xs_active].astype(np.float32)
    )))

    affine_enabled = bool(args.global_affine_bias)
    affine_success = False
    affine_score = None
    affine_flag_bits = 0
    affine_cp_bits_total = 0
    affine_cp_bits_each = [0] * (args.affine_cp_num * 2)
    affine_cp_q = np.zeros((args.affine_cp_num, 2), dtype=np.int32)
    affine_cp_dec = np.zeros((args.affine_cp_num, 2), dtype=np.float32)
    affine_cp_clipped = False
    affine_matrix_est = np.eye(2, 3, dtype=np.float32)
    affine_matrix_dec = np.eye(2, 3, dtype=np.float32)
    affine_lk_stats = {}
    map_x_final = map_x
    map_y_final = map_y

    if affine_enabled:
        affine_flag_bits = 1
        if args.affine_estimator == "lk":
            affine_matrix_est, affine_success, affine_score, affine_lk_stats = estimate_global_affine_bias_lk(
                cur_y=tar_y_pad,
                cam_warp_y=wy_cam,
                valid_mask_u8=valid_mask_u8,
                bit_depth=bit_depth,
                max_iters=args.affine_lk_iters,
                eps=args.affine_lk_eps,
                sample_step=args.affine_lk_sample_step,
                normalize=args.affine_lk_normalize,
                damping=args.affine_lk_damping,
                max_update=args.affine_lk_max_update,
                robust_iters=args.affine_lk_robust_iters,
                outlier_percent=args.affine_lk_outlier_percent,
                min_samples=args.affine_lk_min_samples,
                min_keep_ratio=args.affine_lk_min_keep_ratio,
            )
        elif args.affine_estimator == "block_lk":
            affine_matrix_est, affine_success, affine_score, affine_lk_stats = estimate_blockwise_global_affine_bias_lk(
                cur_y=tar_y_pad,
                cam_warp_y=wy_cam,
                valid_mask_u8=valid_mask_u8,
                bit_depth=bit_depth,
                block_size=args.affine_block_size,
                block_sample_step=args.affine_block_sample_step,
                block_lk_iters=args.affine_block_lk_iters,
                block_lk_eps=args.affine_block_lk_eps,
                normalize=args.affine_lk_normalize,
                damping=args.affine_lk_damping,
                max_update=args.affine_lk_max_update,
                min_block_samples=args.affine_block_min_samples,
                min_block_valid_ratio=args.affine_block_min_valid_ratio,
                robust_iters=args.affine_block_robust_iters,
                outlier_percent=args.affine_block_outlier_percent,
                remove_worst_count=args.affine_block_remove_worst_count,
                min_keep_blocks=args.affine_block_min_keep_blocks,
                weight_mode=args.affine_block_weight_mode,
            )
        else:
            raise ValueError(args.affine_estimator)

        if affine_success:
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

    if args.warp_filter == "bilinear":
        wy_final, wu_final, wv_final = backward_warp_yuv420_bilinear(
            ref_y_pad, ref_u_pad, ref_v_pad, map_x_final, map_y_final, bit_depth
        )
    elif args.warp_filter == "subblk4_6tap_torch":
        wy_final, wu_final, wv_final = backward_warp_yuv420_subblk4_6tap_torch(
            ref_y_pad, ref_u_pad, ref_v_pad, map_x_final, map_y_final, bit_depth, torch_device=args.torch_device
        )
    else:
        raise ValueError(args.warp_filter)

    write_yuv420(Path(args.out_yuv), wy_final, wu_final, wv_final)

    psnr_y_active_final = calc_psnr(wy_final[ys_active, xs_active], tar_y_pad[ys_active, xs_active], bit_depth)
    psnr_y_coded_final = calc_psnr(wy_final, tar_y_pad, bit_depth)
    mae_y_active_final = float(np.mean(np.abs(
        wy_final[ys_active, xs_active].astype(np.float32) - tar_y_pad[ys_active, xs_active].astype(np.float32)
    )))

    valid_ratio_active = float(np.count_nonzero(valid_mask_u8[ys_active, xs_active]) / max(src_w * src_h, 1))
    valid_ratio_coded = float(np.count_nonzero(valid_mask_u8) / max(coded_w * coded_h, 1))

    result = {
        "ref_idx": int(args.ref_idx),
        "tar_idx": int(args.tar_idx),
        "ref_seq_idx": int(ref_seq_idx),
        "tar_seq_idx": int(tar_seq_idx),
        "source_size": {"width": src_w, "height": src_h},
        "coded_size": {"width": coded_w, "height": coded_h},
        "padding": {"left": pad_left, "top": pad_top, "right": pad_right, "bottom": pad_bottom},
        "depth_scale_header": header.get("depth_scale"),
        "depth_scale_precision": depth_scale_precision,
        "depth_scale_real": depth_scale_real,
        "pose": {
            "target_to_reference_rt": rt_tar_to_ref,
        },
        "intrinsic_ref": intr_ref,
        "intrinsic_tar": intr_tar,
        "projection_mode": "target depth backward projection, ref->target warp",
        "valid_ratio_active": valid_ratio_active,
        "valid_ratio_coded": valid_ratio_coded,
        "warp_filter": args.warp_filter,
        "cam_proj_only": {
            "psnr_y_active": json_safe_float(psnr_y_active_cam_only),
            "psnr_y_coded": json_safe_float(psnr_y_coded_cam_only),
            "mae_y_active": mae_y_active_cam_only,
        },
        "global_affine_bias": {
            "enabled": affine_enabled,
            "estimator": args.affine_estimator,
            "success": affine_success,
            "score_negative_rmse": json_safe_float(affine_score),
            "cp_num": int(args.affine_cp_num),
            "cp_step": float(args.affine_cp_step),
            "cp_bits": int(args.affine_cp_bits),
            "flag_bits_50_50": int(affine_flag_bits),
            "cp_q": affine_cp_q.astype(int).tolist(),
            "cp_dec": affine_cp_dec.astype(float).tolist(),
            "cp_bits_each_50_50": affine_cp_bits_each,
            "cp_bits_total_50_50": int(affine_cp_bits_total),
            "total_bits_50_50": int(affine_flag_bits + affine_cp_bits_total),
            "cp_clipped": bool(affine_cp_clipped),
            "matrix_est": affine_matrix_est.astype(float).tolist(),
            "matrix_dec": affine_matrix_dec.astype(float).tolist(),
            "lk": {
                "iters": int(args.affine_lk_iters),
                "eps": float(args.affine_lk_eps),
                "sample_step": int(args.affine_lk_sample_step),
                "normalize": args.affine_lk_normalize,
                "damping": float(args.affine_lk_damping),
                "max_update": float(args.affine_lk_max_update),
                "robust_iters": int(args.affine_lk_robust_iters),
                "outlier_percent": float(args.affine_lk_outlier_percent),
                "min_samples": int(args.affine_lk_min_samples),
                "min_keep_ratio": float(args.affine_lk_min_keep_ratio),
                "stats": affine_lk_stats,
            },
            "block_lk": {
                "block_size": int(args.affine_block_size),
                "block_sample_step": int(args.affine_block_sample_step),
                "block_lk_iters": int(args.affine_block_lk_iters),
                "block_lk_eps": float(args.affine_block_lk_eps),
                "block_min_samples": int(args.affine_block_min_samples),
                "block_min_valid_ratio": float(args.affine_block_min_valid_ratio),
                "block_robust_iters": int(args.affine_block_robust_iters),
                "block_outlier_percent": float(args.affine_block_outlier_percent),
                "block_remove_worst_count": int(args.affine_block_remove_worst_count),
                "block_min_keep_blocks": int(args.affine_block_min_keep_blocks),
                "block_weight_mode": args.affine_block_weight_mode,
            },
        },
        "final": {
            "psnr_y_active": json_safe_float(psnr_y_active_final),
            "psnr_y_coded": json_safe_float(psnr_y_coded_final),
            "mae_y_active": mae_y_active_final,
            "gain_y_active_vs_cam_only": json_safe_float(psnr_y_active_final - psnr_y_active_cam_only),
        },
        "outputs": {
            "out_yuv": str(args.out_yuv),
            "out_cam_yuv": args.out_cam_yuv,
            "out_target_yuv": args.out_target_yuv,
            "out_mask_yuv": args.out_mask_yuv,
            "out_json": str(args.out_json),
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("============================================================")
    print("Single-frame ref -> target warp")
    print("============================================================")
    print(f"ref -> tar             : {args.ref_idx} -> {args.tar_idx}")
    print(f"ref_seq_idx/tar_seq_idx: {ref_seq_idx} / {tar_seq_idx}")
    print(f"pose source            : {rt_tar_to_ref.get('source')}")
    print(f"source size            : {src_w}x{src_h}")
    print(f"coded size             : {coded_w}x{coded_h}")
    print(f"valid ratio active     : {valid_ratio_active:.4f}")
    print(f"warp filter            : {args.warp_filter}")
    print("------------------------------------------------------------")
    print(f"cam-only PSNR active   : {psnr_y_active_cam_only:.3f} dB")
    print(f"cam-only PSNR coded    : {psnr_y_coded_cam_only:.3f} dB")
    print(f"final PSNR active      : {psnr_y_active_final:.3f} dB")
    print(f"final PSNR coded       : {psnr_y_coded_final:.3f} dB")
    print(f"active gain            : {psnr_y_active_final - psnr_y_active_cam_only:+.3f} dB")
    print("------------------------------------------------------------")
    print(f"affine enabled         : {affine_enabled}")
    print(f"affine success         : {affine_success}")
    print(f"affine cp num          : {args.affine_cp_num}")
    print(f"affine cp step         : {args.affine_cp_step}")
    print(f"affine cp q            : {affine_cp_q.astype(int).tolist()}")
    print(f"affine cp dec          : {affine_cp_dec.astype(float).tolist()}")
    print(f"affine bits            : {affine_flag_bits + affine_cp_bits_total}")
    if affine_lk_stats:
        print(f"lk samples init/final  : {affine_lk_stats.get('num_samples_initial')} / {affine_lk_stats.get('num_samples_final')}")
        print(f"lk robust iters        : {affine_lk_stats.get('robust_iters_done')} / {affine_lk_stats.get('robust_iters_requested')}")
        print(f"lk final rmse          : {affine_lk_stats.get('final_rmse')}")
    print("------------------------------------------------------------")
    print(f"warped yuv             : {args.out_yuv}")
    print(f"json                   : {args.out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
