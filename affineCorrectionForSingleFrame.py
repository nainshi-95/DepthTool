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
):
    """
    Fast non-ECC affine estimator.

    It minimizes an SSD/zero-mean-SSD alignment objective between current Y and
    cam-proj-only warped Y using a vectorized forward-additive Lucas-Kanade loop.

    This is intended as a Python simulation path that is much closer to a C++
    implementation than cv2.findTransformECC(), while avoiding the extremely
    slow sparse local-search prototype.

    Returns:
      A, success, score
    where score is negative RMSE after fitting, so larger is better.
    """
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")

    mask = valid_mask_u8 > 0
    if sample_step > 1:
        sample_mask = np.zeros_like(mask, dtype=bool)
        sample_mask[::sample_step, ::sample_step] = True
        mask &= sample_mask

    ys, xs = np.nonzero(mask)
    if xs.size < 64:
        return np.eye(2, 3, dtype=np.float32), False, None

    maxv = float((1 << bit_depth) - 1)
    T_img = np.clip(cur_y.astype(np.float32) / maxv, 0.0, 1.0)
    I_img = np.clip(cam_warp_y.astype(np.float32) / maxv, 0.0, 1.0)

    # Central-difference gradients of input image.  This maps cleanly to C++.
    gy, gx = np.gradient(I_img)
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    T = T_img[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float32)

    h, w = cur_y.shape
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    xn = ((xs - cx) / max(float(w), 1e-12)).astype(np.float32)
    yn = ((ys - cy) / max(float(h), 1e-12)).astype(np.float32)

    p = np.zeros(6, dtype=np.float64)
    final_rmse = None

    for _iter in range(int(max_iters)):
        xw = xs + float(p[0]) + float(p[1]) * xn + float(p[2]) * yn
        yw = ys + float(p[3]) + float(p[4]) * xn + float(p[5]) * yn

        I, valid = bilinear_sample_vectorized(I_img, xw, yw)
        Ix, valid_x = bilinear_sample_vectorized(gx, xw, yw)
        Iy, valid_y = bilinear_sample_vectorized(gy, xw, yw)
        valid &= valid_x & valid_y

        if np.count_nonzero(valid) < 64:
            break

        Tv = T[valid]
        Iv = I[valid]
        Ixv = Ix[valid]
        Iyv = Iy[valid]
        xnv = xn[valid]
        ynv = yn[valid]

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

        # Avoid catastrophic jumps on hard frames.
        if max_update is not None and max_update > 0 and dp_norm > float(max_update):
            dp *= float(max_update) / dp_norm
            dp_norm = float(max_update)

        p += dp
        final_rmse = float(np.sqrt(np.mean(e.astype(np.float64) ** 2)))

        if dp_norm < float(eps):
            break

    A = params_to_affine_matrix_lk(p, w, h)

    if final_rmse is None or not np.isfinite(final_rmse):
        return np.eye(2, 3, dtype=np.float32), False, None

    # score: larger is better, similar role to ECC cc for logging only.
    score = -final_rmse
    return A.astype(np.float32), True, float(score)


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
    map_x_final = map_x
    map_y_final = map_y

    if affine_enabled:
        affine_flag_bits = 1
        affine_matrix_est, affine_success, affine_score = estimate_global_affine_bias_lk(
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
        )

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
            "estimator": "lk_non_ecc",
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
    print("------------------------------------------------------------")
    print(f"warped yuv             : {args.out_yuv}")
    print(f"json                   : {args.out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
