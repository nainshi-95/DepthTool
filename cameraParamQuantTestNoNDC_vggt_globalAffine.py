#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame ref->target camera-projection warp test WITHOUT OpenCV.

This is a no-cv2 replacement for the geometry-oriented structure-ECC test.
Since cv2.findTransformECC(), cv2.remap(), cv2.Scharr(), cv2.Rodrigues(),
cv2.getAffineTransform(), etc. are unavailable, this script implements:

  - Rodrigues rvec -> rotation matrix in NumPy
  - bilinear remap in NumPy
  - Scharr-like structure image in NumPy
  - valid + active + strong-structure mask in NumPy
  - global affine residual fitting by vectorized Lucas-Kanade / SSD alignment

Recommended geometry-oriented mode:
  Use Scharr structure images, keep the top ~35% structure pixels inside the
  valid active region, and fit one global 3CP affine residual transform.

CP modes:
  --affine-cp-num 2: constrained 4-param affine from CP0/CP1
  --affine-cp-num 3: full 3CP affine

Flow:
  1) Read ref frame, target frame, target depth, and camera parameters.
  2) Build target->reference projection map.
  3) Warp reference frame to target frame by cam-proj only.
  4) Optional: estimate global affine bias from structure(cam-warped Y) to
     structure(target Y), using valid active strong-structure pixels only.
  5) Quantize/decode CP bias, apply it to the map, and warp again.
"""

import argparse
import json
import os
from pathlib import Path

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
            f"Invalid padding: src=({src_w}x{src_h}), coded=({coded_w}x{coded_h}), "
            f"pad_left={pad_left}, pad_top={pad_top}"
        )
    return pad_right, pad_bottom


def validate_yuv420_padding(src_w, src_h, coded_w, coded_h, pad_left, pad_top, pad_right, pad_bottom):
    vals = {
        "src_w": src_w, "src_h": src_h, "coded_w": coded_w, "coded_h": coded_h,
        "pad_left": pad_left, "pad_top": pad_top, "pad_right": pad_right, "pad_bottom": pad_bottom,
    }
    for name, v in vals.items():
        if v < 0:
            raise ValueError(f"{name} must be non-negative: {v}")
        if v % 2 != 0:
            raise ValueError(f"{name} must be even for YUV420: {v}")


def pad_2d_edge(arr, coded_w, coded_h, pad_left, pad_top):
    h, w = arr.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top
    return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")


def pad_yuv420_edge(y, u, v, coded_w, coded_h, pad_left, pad_top):
    y_pad = pad_2d_edge(y, coded_w, coded_h, pad_left, pad_top)
    u_pad = pad_2d_edge(u, coded_w // 2, coded_h // 2, pad_left // 2, pad_top // 2)
    v_pad = pad_2d_edge(v, coded_w // 2, coded_h // 2, pad_left // 2, pad_top // 2)
    return y_pad, u_pad, v_pad


def active_slice(src_w, src_h, pad_left, pad_top):
    return slice(pad_top, pad_top + src_h), slice(pad_left, pad_left + src_w)


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
    return y.reshape(h, w), u.reshape(h // 2, w // 2), v.reshape(h // 2, w // 2)


def write_yuv420(path, y, u, v):
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(y).tobytes())
        f.write(np.ascontiguousarray(u).tobytes())
        f.write(np.ascontiguousarray(v).tobytes())


def write_mask_yuv420(path, mask_y, bit_depth):
    maxv = (1 << bit_depth) - 1
    dtype = yuv_dtype(bit_depth)
    y = np.where(mask_y > 0, maxv, 0).astype(dtype)
    h, w = y.shape
    neutral = 128 if bit_depth <= 8 else 512
    u = np.full((h // 2, w // 2), neutral, dtype=dtype)
    v = np.full((h // 2, w // 2), neutral, dtype=dtype)
    write_yuv420(path, y, u, v)


# ============================================================
# JSONL / quantization
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
    if "intrinsic" not in header and "intrinsic_dec_first" not in header and "intrinsic_gt_padded0" not in header:
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


def signed_q_abs_max(bits):
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


def quant_s(value, step, bits):
    q_abs_max = signed_q_abs_max(bits)
    qmin = -q_abs_max
    qmax = q_abs_max
    q = np.round(np.asarray(value, dtype=np.float32) / float(step))
    clipped = (q < qmin) | (q > qmax)
    q = np.clip(q, qmin, qmax).astype(np.int32)
    dec = q.astype(np.float32) * float(step)
    return q, dec, clipped


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
        raise ValueError(f"x={x} outside signed truncated range [-{q_abs_max}, {q_abs_max}]")
    return ue_exp_golomb_bits(signed_to_code_num(x))


def q_residual_bits_signed_trunc_exp_golomb(q_residual, q_abs_max):
    bits_each = [signed_truncated_exp_golomb_bits(int(v), q_abs_max) for v in q_residual]
    return bits_each, int(sum(bits_each))


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


def _maybe_padded_intrinsic(intr, pad_left, pad_top, assume_already_padded=False):
    if assume_already_padded:
        return {
            "fx": float(intr["fx"]), "fy": float(intr["fy"]),
            "cx": float(intr["cx"]), "cy": float(intr["cy"]),
            "z_sign": float(intr.get("z_sign", 1.0)),
        }
    return make_padded_intrinsic_from_original(intr, pad_left, pad_top)


def build_frame_intrinsics(header, frames, max_idx, pad_left, pad_top):
    intrs = {}
    if "intrinsic_dec_first" in header:
        intr0 = _maybe_padded_intrinsic(header["intrinsic_dec_first"], pad_left, pad_top, True)
    elif "intrinsic_gt_padded0" in header:
        intr0 = _maybe_padded_intrinsic(header["intrinsic_gt_padded0"], pad_left, pad_top, True)
    elif "intrinsic" in header:
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


# ============================================================
# Rotation / pose helpers without cv2
# ============================================================

def rodrigues_to_matrix(rvec):
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + K
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


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
        R = rodrigues_to_matrix(frame["rvec_abs"])
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
        raise RuntimeError("Frame has no relative pose. Expected either rt_dec or rvec/tvec.")
    return rodrigues_to_matrix(rvec), np.asarray(tvec, dtype=np.float64).reshape(3)


def _invert_rt(R, t):
    Ri = R.T
    ti = -Ri @ t
    return Ri, ti


def _compose_rt(R2, t2, R1, t1):
    return R2 @ R1, R2 @ t1 + t2


def get_target_to_reference_rt(frames, ref_idx, tar_idx):
    ref_idx = int(ref_idx)
    tar_idx = int(tar_idx)
    if ref_idx == tar_idx:
        return {"R": np.eye(3, dtype=np.float64), "t": np.zeros(3, dtype=np.float64), "source": "identity"}

    f_ref = frames.get(ref_idx, {})
    f_tar = frames.get(tar_idx, {})
    E_ref = _as_extrinsic_matrix(f_ref)
    E_tar = _as_extrinsic_matrix(f_tar)
    if E_ref is not None and E_tar is not None:
        R_ref, t_ref = E_ref[:, :3], E_ref[:, 3]
        R_tar, t_tar = E_tar[:, :3], E_tar[:, 3]
        R = R_ref @ R_tar.T
        t = t_ref - R @ t_tar
        return {"R": R.astype(np.float64), "t": t.astype(np.float64), "source": "absolute_extrinsic"}

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

    return {"R": R_tot.astype(np.float64), "t": t_tot.astype(np.float64), "source": "composed_current_to_previous"}


# ============================================================
# Projection
# ============================================================

def make_projection_precompute_dual(w, h, intr_tar, intr_ref):
    fx_t, fy_t = float(intr_tar["fx"]), float(intr_tar["fy"])
    cx_t, cy_t = float(intr_tar["cx"]), float(intr_tar["cy"])
    fx_r, fy_r = float(intr_ref["fx"]), float(intr_ref["fy"])
    cx_r, cy_r = float(intr_ref["cx"]), float(intr_ref["cy"])
    z_sign = float(intr_tar.get("z_sign", intr_ref.get("z_sign", 1.0)))
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_norm = (x - cx_t) / fx_t
    y_norm = (y - cy_t) / fy_t
    return {
        "w": int(w), "h": int(h),
        "fx_ref": fx_r, "fy_ref": fy_r, "cx_ref": cx_r, "cy_ref": cy_r,
        "z_sign": z_sign,
        "x_norm": x_norm.astype(np.float32), "y_norm": y_norm.astype(np.float32),
    }


def backward_map_fast_pixel_coord_dual(depth_linear, precomp, rt):
    w, h = precomp["w"], precomp["h"]
    fx, fy = precomp["fx_ref"], precomp["fy_ref"]
    cx, cy = precomp["cx_ref"], precomp["cy_ref"]
    z_sign = precomp["z_sign"]
    x_norm, y_norm = precomp["x_norm"], precomp["y_norm"]
    z = depth_linear.astype(np.float32)
    R = np.asarray(rt["R"], dtype=np.float32).reshape(3, 3)
    t = np.asarray(rt["t"], dtype=np.float32).reshape(3)

    kx = R[0, 0] * x_norm + R[0, 1] * y_norm + R[0, 2] * z_sign
    ky = R[1, 0] * x_norm + R[1, 1] * y_norm + R[1, 2] * z_sign
    kz = R[2, 0] * x_norm + R[2, 1] * y_norm + R[2, 2] * z_sign

    Xp = z * kx + float(t[0])
    Yp = z * ky + float(t[1])
    Zp = z * kz + float(t[2])
    denom = np.maximum(np.abs(Zp), 1e-8)
    map_x = fx * (Xp / denom) + cx
    map_y = fy * (Yp / denom) + cy
    valid = (
        np.isfinite(map_x) & np.isfinite(map_y) & (Zp * z_sign > 0) &
        (map_x >= 0.0) & (map_x <= w - 1) & (map_y >= 0.0) & (map_y <= h - 1) & (z > 0.0)
    )
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y


# ============================================================
# No-OpenCV remap / filters / masks
# ============================================================

def bilinear_sample(src, x, y, border_value=0.0):
    src_f = src.astype(np.float32, copy=False)
    h, w = src_f.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)

    x0 = np.floor(np.clip(x, 0.0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0.0, h - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = np.clip(x - x0.astype(np.float32), 0.0, 1.0)
    fy = np.clip(y - y0.astype(np.float32), 0.0, 1.0)

    p00 = src_f[y0, x0]
    p01 = src_f[y0, x1]
    p10 = src_f[y1, x0]
    p11 = src_f[y1, x1]
    a = p00 * (1.0 - fx) + p01 * fx
    b = p10 * (1.0 - fx) + p11 * fx
    out = a * (1.0 - fy) + b * fy
    out = np.where(valid, out, float(border_value)).astype(np.float32)
    return out


def remap_plane(src, map_x, map_y, bit_depth, border_value):
    maxv = (1 << bit_depth) - 1
    dst = bilinear_sample(src, map_x, map_y, border_value=float(border_value))
    dst = np.clip(np.round(dst), 0, maxv)
    return dst.astype(yuv_dtype(bit_depth))


def downsample_luma_map_to_chroma_map(map_x, map_y):
    h, w = map_x.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("luma map size must be even for YUV420")
    uv_h, uv_w = h // 2, w // 2
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


def morph_mask(mask_u8, radius, op):
    radius = int(radius)
    if radius <= 0:
        return mask_u8.astype(np.uint8).copy()
    m = mask_u8 > 0
    pad_value = False if op == "dilate" else True
    p = np.pad(m, ((radius, radius), (radius, radius)), mode="constant", constant_values=pad_value)
    out = np.zeros_like(m, dtype=bool) if op == "dilate" else np.ones_like(m, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            s = p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
            if op == "dilate":
                out |= s
            elif op == "erode":
                out &= s
            else:
                raise ValueError(op)
    return (out.astype(np.uint8) * 255)


def make_valid_u8_mask(map_x, map_y, w, h, erode=0, active_region=None):
    valid = (
        np.isfinite(map_x) & np.isfinite(map_y) &
        (map_x >= 0.0) & (map_x <= w - 1) & (map_y >= 0.0) & (map_y <= h - 1)
    )
    if active_region is not None:
        ys, xs = active_region
        active = np.zeros_like(valid, dtype=bool)
        active[ys, xs] = True
        valid &= active
    mask = valid.astype(np.uint8) * 255
    if int(erode) > 0:
        mask = morph_mask(mask, int(erode), "erode")
    return mask


def normalize_for_fit(img, bit_depth):
    x = img.astype(np.float32)
    maxv = float((1 << bit_depth) - 1)
    return np.clip(x / maxv, 0.0, 1.0).astype(np.float32)


def gaussian_blur_np(img, radius):
    radius = int(radius)
    if radius <= 0:
        return img.astype(np.float32, copy=False)
    sigma = max(radius / 2.0, 0.5)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    tmp = np.pad(img.astype(np.float32), ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i, kv in enumerate(k):
        out += kv * tmp[:, i:i + img.shape[1]]
    tmp = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    out2 = np.zeros_like(img, dtype=np.float32)
    for i, kv in enumerate(k):
        out2 += kv * tmp[i:i + img.shape[0], :]
    return out2


def scharr_grad_np(y):
    y = y.astype(np.float32, copy=False)
    p = np.pad(y, ((1, 1), (1, 1)), mode="edge")
    # Sign is irrelevant for magnitude/abs; this matches a Scharr-like derivative scale.
    gx = (
        -3.0 * p[:-2, :-2] + 3.0 * p[:-2, 2:] +
        -10.0 * p[1:-1, :-2] + 10.0 * p[1:-1, 2:] +
        -3.0 * p[2:, :-2] + 3.0 * p[2:, 2:]
    )
    gy = (
        -3.0 * p[:-2, :-2] -10.0 * p[:-2, 1:-1] -3.0 * p[:-2, 2:] +
         3.0 * p[2:, :-2] +10.0 * p[2:, 1:-1] +3.0 * p[2:, 2:]
    )
    return gx.astype(np.float32), gy.astype(np.float32)


def make_structure_image(img, bit_depth, mode="scharr_mag", log_gain=20.0, pre_blur=0):
    y = normalize_for_fit(img, bit_depth)
    if int(pre_blur) > 0:
        y = gaussian_blur_np(y, int(pre_blur))
    gx, gy = scharr_grad_np(y)
    mode = str(mode)
    if mode == "scharr_mag":
        s = np.sqrt(gx * gx + gy * gy)
    elif mode == "scharr_l1":
        s = np.abs(gx) + np.abs(gy)
    elif mode == "scharr_x":
        s = np.abs(gx)
    elif mode == "scharr_y":
        s = np.abs(gy)
    elif mode == "scharr_x_weighted":
        s = 0.75 * np.abs(gx) + 0.25 * np.abs(gy)
    else:
        raise ValueError(f"Unsupported structure mode: {mode}")
    if float(log_gain) > 0.0:
        s = np.log1p(float(log_gain) * s)
    m = float(np.max(s))
    if m > 1e-8:
        s = s / m
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def make_structure_mask_u8(structure, base_mask_u8, keep_percent=35.0, dilate=1):
    base = base_mask_u8 > 0
    valid_vals = structure[base]
    stats = {
        "base_count": int(np.count_nonzero(base)),
        "keep_percent": float(keep_percent),
        "threshold": None,
        "structure_count": 0,
        "final_count": 0,
        "final_ratio_vs_base": 0.0,
    }
    if valid_vals.size < 100:
        return base_mask_u8.copy(), stats
    keep_percent = max(0.1, min(100.0, float(keep_percent)))
    if keep_percent >= 99.999:
        thr = float(np.min(valid_vals))
        mask = base.copy()
    else:
        thr = float(np.percentile(valid_vals, 100.0 - keep_percent))
        mask = base & (structure >= thr)
    stats["threshold"] = thr
    stats["structure_count"] = int(np.count_nonzero(mask))
    mask_u8 = mask.astype(np.uint8) * 255
    if int(dilate) > 0:
        mask_u8 = morph_mask(mask_u8, int(dilate), "dilate")
        mask_u8 = np.where(base, mask_u8, 0).astype(np.uint8)
    stats["final_count"] = int(np.count_nonzero(mask_u8))
    if stats["base_count"] > 0:
        stats["final_ratio_vs_base"] = float(stats["final_count"] / stats["base_count"])
    return mask_u8, stats


# ============================================================
# No-OpenCV structure LK affine estimator
# ============================================================

def affine_p_to_matrix(p, w, h):
    p = np.asarray(p, dtype=np.float64).reshape(6)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    W = max(float(w), 1e-12)
    H = max(float(h), 1e-12)
    # x' = x + p0 + p1*(x-cx)/W + p2*(y-cy)/H
    # y' = y + p3 + p4*(x-cx)/W + p5*(y-cy)/H
    a = 1.0 + p[1] / W
    b = p[2] / H
    c = p[0] - p[1] * cx / W - p[2] * cy / H
    d = p[4] / W
    e = 1.0 + p[5] / H
    f = p[3] - p[4] * cx / W - p[5] * cy / H
    return np.array([[a, b, c], [d, e, f]], dtype=np.float32)


def identity_transform_matrix(cp_num):
    return np.eye(2, 3, dtype=np.float32)


def prepare_structure_images_and_mask(cur_y, cam_warp_y, valid_mask_u8, bit_depth,
                                      structure_mode="scharr_mag", structure_keep_percent=35.0,
                                      structure_mask_dilate=1, structure_log_gain=20.0,
                                      structure_pre_blur=0):
    template = make_structure_image(cur_y, bit_depth, mode=structure_mode,
                                    log_gain=structure_log_gain, pre_blur=structure_pre_blur)
    inp = make_structure_image(cam_warp_y, bit_depth, mode=structure_mode,
                               log_gain=structure_log_gain, pre_blur=structure_pre_blur)
    mask, mask_stats = make_structure_mask_u8(template, valid_mask_u8,
                                              keep_percent=structure_keep_percent,
                                              dilate=structure_mask_dilate)
    stats = {
        "fit_input": "structure_lk_noopencv",
        "structure_mode": str(structure_mode),
        "structure_keep_percent": float(structure_keep_percent),
        "structure_mask_dilate": int(structure_mask_dilate),
        "structure_log_gain": float(structure_log_gain),
        "structure_pre_blur": int(structure_pre_blur),
        "mask_count_valid": int(np.count_nonzero(valid_mask_u8)),
        "mask_count_used": int(np.count_nonzero(mask)),
        "structure_mask": mask_stats,
    }
    return template, inp, mask, stats


def estimate_global_affine_bias_structure_lk(
    cur_y,
    cam_warp_y,
    valid_mask_u8,
    bit_depth,
    w,
    h,
    structure_mode="scharr_mag",
    structure_keep_percent=35.0,
    structure_mask_dilate=1,
    structure_log_gain=20.0,
    structure_pre_blur=0,
    lk_sample_step=4,
    lk_iters=50,
    lk_eps=1e-4,
    lk_normalize="none",
    lk_weight="structure",
    lk_damping=1.0,
):
    template, inp, fit_mask_u8, stats = prepare_structure_images_and_mask(
        cur_y=cur_y,
        cam_warp_y=cam_warp_y,
        valid_mask_u8=valid_mask_u8,
        bit_depth=bit_depth,
        structure_mode=structure_mode,
        structure_keep_percent=structure_keep_percent,
        structure_mask_dilate=structure_mask_dilate,
        structure_log_gain=structure_log_gain,
        structure_pre_blur=structure_pre_blur,
    )

    mask = fit_mask_u8 > 0
    step = max(1, int(lk_sample_step))
    sub = np.zeros_like(mask, dtype=bool)
    sub[::step, ::step] = True
    mask &= sub

    ys, xs = np.nonzero(mask)
    if xs.size < 100:
        stats.update({"sample_count": int(xs.size), "success_reason": "too_few_samples"})
        return identity_transform_matrix(3), False, None, stats, fit_mask_u8

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    T = template[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float32)

    if lk_normalize == "zero_mean":
        T_mean = float(np.mean(T))
        T_std = float(np.std(T) + 1e-6)
        T_fit = (T - T_mean) / T_std
        inp_fit = (inp - T_mean) / T_std
    elif lk_normalize == "none":
        T_fit = T
        inp_fit = inp
    else:
        raise ValueError(f"Unsupported --affine-lk-normalize: {lk_normalize}")

    grad_x, grad_y = scharr_grad_np(inp_fit)

    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    xn = (xs - cx) / max(float(w), 1e-12)
    yn = (ys - cy) / max(float(h), 1e-12)

    if lk_weight == "structure":
        weight = np.sqrt(np.maximum(template[ys.astype(np.int32), xs.astype(np.int32)], 1e-4)).astype(np.float32)
    elif lk_weight == "none":
        weight = np.ones_like(xs, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported --affine-lk-weight: {lk_weight}")

    p = np.zeros(6, dtype=np.float64)
    history = []
    success = False

    for it in range(int(lk_iters)):
        xw = xs + p[0] + p[1] * xn + p[2] * yn
        yw = ys + p[3] + p[4] * xn + p[5] * yn
        inside = (xw >= 1.0) & (xw <= w - 2) & (yw >= 1.0) & (yw <= h - 2)
        if np.count_nonzero(inside) < 100:
            break

        xwi = xw[inside]
        ywi = yw[inside]
        I = bilinear_sample(inp_fit, xwi, ywi, border_value=0.0)
        Ix = bilinear_sample(grad_x, xwi, ywi, border_value=0.0)
        Iy = bilinear_sample(grad_y, xwi, ywi, border_value=0.0)
        e = T_fit[inside] - I

        xi = xn[inside]
        yi = yn[inside]
        wi = weight[inside]

        J = np.stack([Ix, Ix * xi, Ix * yi, Iy, Iy * xi, Iy * yi], axis=1).astype(np.float64)
        ew = (e * wi).astype(np.float64)
        Jw = J * wi[:, None].astype(np.float64)
        Hm = Jw.T @ Jw
        b = Jw.T @ ew
        Hm.flat[::7] += 1e-6

        try:
            dp = np.linalg.solve(Hm, b)
        except np.linalg.LinAlgError:
            dp = np.linalg.lstsq(Hm, b, rcond=None)[0]

        dp *= float(lk_damping)
        p += dp
        dp_norm = float(np.linalg.norm(dp))
        mean_abs_e = float(np.mean(np.abs(e)))
        history.append({"iter": int(it), "dp_norm": dp_norm, "mean_abs_error": mean_abs_e, "sample_count": int(np.count_nonzero(inside))})
        if dp_norm < float(lk_eps):
            success = True
            break
    else:
        success = True

    M = affine_p_to_matrix(p, w, h)
    stats.update({
        "sample_count": int(xs.size),
        "lk_sample_step": int(lk_sample_step),
        "lk_iters_requested": int(lk_iters),
        "lk_iters_done": int(len(history)),
        "lk_eps": float(lk_eps),
        "lk_normalize": str(lk_normalize),
        "lk_weight": str(lk_weight),
        "lk_damping": float(lk_damping),
        "p_final": p.astype(float).tolist(),
        "history_tail": history[-5:],
    })
    score = -history[-1]["mean_abs_error"] if history else None
    return M.astype(np.float32), bool(success), score, stats, fit_mask_u8


# ============================================================
# CP / transform helpers
# ============================================================

def transform_cp_points(w, h, cp_num):
    cp_num = int(cp_num)
    if cp_num == 2:
        pts = [[0.0, 0.0], [float(w), 0.0]]
    elif cp_num == 3:
        pts = [[0.0, 0.0], [float(w), 0.0], [0.0, float(h)]]
    else:
        raise ValueError("No-OpenCV LK version supports --affine-cp-num 2 or 3 only")
    return np.asarray(pts, dtype=np.float32)


def apply_transform_to_points(M, pts):
    M = np.asarray(M, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)
    return (homo @ M.T).astype(np.float32)


def transform_matrix_to_cp_bias(M, w, h, cp_num=3):
    src = transform_cp_points(w, h, cp_num)
    dst = apply_transform_to_points(M, src)
    return (dst - src).astype(np.float32)


def cp_bias_to_transform_matrix(cp_bias, w, h, cp_num=3):
    cp_num = int(cp_num)
    cp_bias = np.asarray(cp_bias, dtype=np.float32).reshape(cp_num, 2)
    if cp_num == 2:
        dx0, dy0 = float(cp_bias[0, 0]), float(cp_bias[0, 1])
        dx1, dy1 = float(cp_bias[1, 0]), float(cp_bias[1, 1])
        ww = max(float(w), 1e-12)
        tx, ty = dx0, dy0
        a = (float(w) + dx1 - tx) / ww
        b = (dy1 - ty) / ww
        return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)
    if cp_num == 3:
        dx0, dy0 = float(cp_bias[0, 0]), float(cp_bias[0, 1])
        dx1, dy1 = float(cp_bias[1, 0]), float(cp_bias[1, 1])
        dx2, dy2 = float(cp_bias[2, 0]), float(cp_bias[2, 1])
        W = max(float(w), 1e-12)
        H = max(float(h), 1e-12)
        # Source CPs: (0,0), (W,0), (0,H)
        c = dx0
        f = dy0
        a = (W + dx1 - c) / W
        d = (dy1 - f) / W
        b = (dx2 - c) / H
        e = (H + dy2 - f) / H
        return np.array([[a, b, c], [d, e, f]], dtype=np.float32)
    raise ValueError("No-OpenCV LK version supports --affine-cp-num 2 or 3 only")


def apply_affine_bias_to_map(map_x, map_y, M):
    h, w = map_x.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    x2 = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
    y2 = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
    out_x = map_x + (x2 - xx)
    out_y = map_y + (y2 - yy)
    valid = (
        np.isfinite(out_x) & np.isfinite(out_y) & (map_x >= 0.0) & (map_y >= 0.0) &
        (out_x >= 0.0) & (out_x <= w - 1) & (out_y >= 0.0) & (out_y <= h - 1)
    )
    out_x = out_x.astype(np.float32)
    out_y = out_y.astype(np.float32)
    out_x[~valid] = -1.0
    out_y[~valid] = -1.0
    return out_x, out_y


def quantize_transform_cp_bias(cp_bias, step, bits, cp_num=3):
    cp_num = int(cp_num)
    cp_bias = np.asarray(cp_bias, dtype=np.float32).reshape(cp_num, 2)
    flat = cp_bias.reshape(cp_num * 2)
    q, dec, clipped = quant_s(flat, step=step, bits=bits)
    q_abs_max = signed_q_abs_max(bits)
    bits_each, bits_total = q_residual_bits_signed_trunc_exp_golomb(q, q_abs_max=q_abs_max)
    return q.astype(np.int32).reshape(cp_num, 2), dec.astype(np.float32).reshape(cp_num, 2), clipped.reshape(cp_num, 2), bits_each, int(bits_total)


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
    ap.add_argument("--ref-idx", type=int, required=True)
    ap.add_argument("--tar-idx", type=int, required=True)
    ap.add_argument("--seq-start", type=int, default=0)
    ap.add_argument("--coded-width", type=int, default=None)
    ap.add_argument("--coded-height", type=int, default=None)
    ap.add_argument("--pad-left", type=int, default=0)
    ap.add_argument("--pad-top", type=int, default=0)
    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-cam-yuv", default=None)
    ap.add_argument("--out-target-yuv", default=None)
    ap.add_argument("--out-mask-yuv", default=None)

    ap.add_argument("--global-affine-bias", action="store_true")
    ap.add_argument("--affine-cp-num", type=int, choices=[2, 3], default=3)
    ap.add_argument("--affine-cp-step", type=float, default=1.0)
    ap.add_argument("--affine-cp-bits", type=int, default=16)
    ap.add_argument("--affine-valid-erode", type=int, default=2)

    # Structure LK settings: no cv2.findTransformECC.
    ap.add_argument("--structure-mode", choices=["scharr_mag", "scharr_l1", "scharr_x", "scharr_y", "scharr_x_weighted"], default="scharr_mag")
    ap.add_argument("--structure-keep-percent", type=float, default=35.0)
    ap.add_argument("--structure-mask-dilate", type=int, default=1)
    ap.add_argument("--structure-log-gain", type=float, default=20.0)
    ap.add_argument("--structure-pre-blur", type=int, default=0)
    ap.add_argument("--affine-lk-sample-step", type=int, default=4)
    ap.add_argument("--affine-lk-iters", type=int, default=50)
    ap.add_argument("--affine-lk-eps", type=float, default=1e-4)
    ap.add_argument("--affine-lk-normalize", choices=["none", "zero_mean"], default="none")
    ap.add_argument("--affine-lk-weight", choices=["none", "structure"], default="structure")
    ap.add_argument("--affine-lk-damping", type=float, default=1.0)

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
    if args.structure_keep_percent <= 0 or args.structure_keep_percent > 100:
        raise ValueError("--structure-keep-percent must be in (0, 100]")
    if args.structure_mask_dilate < 0 or args.structure_pre_blur < 0:
        raise ValueError("structure mask/blur parameters must be non-negative")

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

    src_w, src_h = int(args.width), int(args.height)
    bit_depth = int(args.bit_depth)
    pad_left, pad_top = int(args.pad_left), int(args.pad_top)
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

    projection_precomp = make_projection_precompute_dual(coded_w, coded_h, intr_tar=intr_tar, intr_ref=intr_ref)
    map_x, map_y = backward_map_fast_pixel_coord_dual(depth_linear=depth_linear_pad, precomp=projection_precomp, rt=rt_tar_to_ref)
    valid_mask_u8 = make_valid_u8_mask(map_x, map_y, coded_w, coded_h, erode=args.affine_valid_erode, active_region=(ys_active, xs_active))

    wy_cam, wu_cam, wv_cam = backward_warp_yuv420_bilinear(ref_y_pad, ref_u_pad, ref_v_pad, map_x, map_y, bit_depth)
    if args.out_cam_yuv is not None:
        write_yuv420(Path(args.out_cam_yuv), wy_cam, wu_cam, wv_cam)

    psnr_y_active_cam_only = calc_psnr(wy_cam[ys_active, xs_active], tar_y_pad[ys_active, xs_active], bit_depth)
    psnr_y_coded_cam_only = calc_psnr(wy_cam, tar_y_pad, bit_depth)
    mae_y_active_cam_only = float(np.mean(np.abs(wy_cam[ys_active, xs_active].astype(np.float32) - tar_y_pad[ys_active, xs_active].astype(np.float32))))

    affine_enabled = bool(args.global_affine_bias)
    affine_success = False
    affine_score = None
    affine_flag_bits = 0
    affine_cp_bits_total = 0
    affine_cp_bits_each = [0] * (args.affine_cp_num * 2)
    affine_cp_q = np.zeros((args.affine_cp_num, 2), dtype=np.int32)
    affine_cp_dec = np.zeros((args.affine_cp_num, 2), dtype=np.float32)
    affine_cp_clipped = False
    affine_matrix_est = identity_transform_matrix(args.affine_cp_num)
    affine_matrix_dec = identity_transform_matrix(args.affine_cp_num)
    fit_stats = {"fit_input": "none", "mask_count_valid": int(np.count_nonzero(valid_mask_u8)), "mask_count_used": int(np.count_nonzero(valid_mask_u8))}
    fit_mask_u8 = valid_mask_u8.copy()
    map_x_final, map_y_final = map_x, map_y

    if affine_enabled:
        affine_flag_bits = 1
        affine_matrix_est, affine_success, affine_score, fit_stats, fit_mask_u8 = estimate_global_affine_bias_structure_lk(
            cur_y=tar_y_pad,
            cam_warp_y=wy_cam,
            valid_mask_u8=valid_mask_u8,
            bit_depth=bit_depth,
            w=coded_w,
            h=coded_h,
            structure_mode=args.structure_mode,
            structure_keep_percent=args.structure_keep_percent,
            structure_mask_dilate=args.structure_mask_dilate,
            structure_log_gain=args.structure_log_gain,
            structure_pre_blur=args.structure_pre_blur,
            lk_sample_step=args.affine_lk_sample_step,
            lk_iters=args.affine_lk_iters,
            lk_eps=args.affine_lk_eps,
            lk_normalize=args.affine_lk_normalize,
            lk_weight=args.affine_lk_weight,
            lk_damping=args.affine_lk_damping,
        )
        if affine_success:
            cp_bias_est = transform_matrix_to_cp_bias(affine_matrix_est, coded_w, coded_h, cp_num=args.affine_cp_num)
            affine_cp_q, affine_cp_dec, affine_cp_clip_arr, affine_cp_bits_each, affine_cp_bits_total = quantize_transform_cp_bias(
                cp_bias_est, step=args.affine_cp_step, bits=args.affine_cp_bits, cp_num=args.affine_cp_num
            )
            affine_cp_clipped = bool(np.any(affine_cp_clip_arr))
            affine_matrix_dec = cp_bias_to_transform_matrix(affine_cp_dec, coded_w, coded_h, cp_num=args.affine_cp_num)
            map_x_final, map_y_final = apply_affine_bias_to_map(map_x, map_y, affine_matrix_dec)

    if args.out_mask_yuv is not None:
        write_mask_yuv420(Path(args.out_mask_yuv), fit_mask_u8, bit_depth)

    wy_final, wu_final, wv_final = backward_warp_yuv420_bilinear(ref_y_pad, ref_u_pad, ref_v_pad, map_x_final, map_y_final, bit_depth)
    write_yuv420(Path(args.out_yuv), wy_final, wu_final, wv_final)

    psnr_y_active_final = calc_psnr(wy_final[ys_active, xs_active], tar_y_pad[ys_active, xs_active], bit_depth)
    psnr_y_coded_final = calc_psnr(wy_final, tar_y_pad, bit_depth)
    mae_y_active_final = float(np.mean(np.abs(wy_final[ys_active, xs_active].astype(np.float32) - tar_y_pad[ys_active, xs_active].astype(np.float32))))

    valid_ratio_active = float(np.count_nonzero(valid_mask_u8[ys_active, xs_active]) / max(src_w * src_h, 1))
    valid_ratio_coded = float(np.count_nonzero(valid_mask_u8) / max(coded_w * coded_h, 1))
    fit_mask_ratio_active = float(np.count_nonzero(fit_mask_u8[ys_active, xs_active]) / max(src_w * src_h, 1))
    fit_mask_ratio_coded = float(np.count_nonzero(fit_mask_u8) / max(coded_w * coded_h, 1))

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
        "pose": {"target_to_reference_rt": {"R": rt_tar_to_ref["R"].astype(float).tolist(), "t": rt_tar_to_ref["t"].astype(float).tolist(), "source": rt_tar_to_ref.get("source")}},
        "intrinsic_ref": intr_ref,
        "intrinsic_tar": intr_tar,
        "projection_mode": "target depth backward projection, ref->target warp",
        "valid_ratio_active": valid_ratio_active,
        "valid_ratio_coded": valid_ratio_coded,
        "fit_mask_ratio_active": fit_mask_ratio_active,
        "fit_mask_ratio_coded": fit_mask_ratio_coded,
        "cam_proj_only": {
            "psnr_y_active": json_safe_float(psnr_y_active_cam_only),
            "psnr_y_coded": json_safe_float(psnr_y_coded_cam_only),
            "mae_y_active": mae_y_active_cam_only,
        },
        "global_affine_bias": {
            "enabled": affine_enabled,
            "estimator": "structure_lk_noopencv",
            "motion_type": "affine",
            "success": affine_success,
            "score": json_safe_float(affine_score),
            "cp_num": int(args.affine_cp_num),
            "cp_semantics": "3CP full affine" if int(args.affine_cp_num) == 3 else "2CP constrained affine",
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
            "fit": fit_stats,
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
    print("Single-frame ref -> target warp, no OpenCV")
    print("============================================================")
    print(f"ref -> tar             : {args.ref_idx} -> {args.tar_idx}")
    print(f"ref_seq_idx/tar_seq_idx: {ref_seq_idx} / {tar_seq_idx}")
    print(f"pose source            : {rt_tar_to_ref.get('source')}")
    print(f"source size            : {src_w}x{src_h}")
    print(f"coded size             : {coded_w}x{coded_h}")
    print(f"valid ratio active     : {valid_ratio_active:.4f}")
    print(f"fit mask ratio active  : {fit_mask_ratio_active:.4f}")
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
    print(f"structure mode         : {args.structure_mode}")
    print(f"structure keep percent : {args.structure_keep_percent}")
    print(f"LK sample step         : {args.affine_lk_sample_step}")
    print(f"LK iters done          : {fit_stats.get('lk_iters_done')}")
    print(f"affine cp q            : {affine_cp_q.astype(int).tolist()}")
    print(f"affine cp dec          : {affine_cp_dec.astype(float).tolist()}")
    print(f"affine bits            : {affine_flag_bits + affine_cp_bits_total}")
    print("------------------------------------------------------------")
    print(f"warped yuv             : {args.out_yuv}")
    print(f"json                   : {args.out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
