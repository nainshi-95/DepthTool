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


# ============================================================
# YUV
# ============================================================

def frame_size_yuv420(w, h, bit_depth):
    bps = 1 if bit_depth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def count_frames(path, w, h, bit_depth):
    return os.path.getsize(path) // frame_size_yuv420(w, h, bit_depth)


def read_yuv420(path, idx, w, h, bit_depth):
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
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

    return header, frames


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
        raise ValueError("ext-bits must be >= 2")
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
    """
    오른쪽/아래쪽 padding이면 pad_left=0, pad_top=0이므로 intrinsic 변경 없음.
    왼쪽/위쪽 padding이 있으면 principal point만 이동.
    """
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]) + float(pad_left),
        "cy": float(intr["cy"]) + float(pad_top),
        "z_sign": float(intr.get("z_sign", -1.0)),
    }


def quantize_intrinsic_16(intr, w, h, f_max=4.0, c_min=-1.0, c_max=2.0):
    """
    w, h는 decoder가 아는 coded size 사용.
    right/bottom padding만이면 cx, cy 자체는 그대로지만 normalize denominator는 coded_w/h.
    """
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
        "z_sign": float(intr.get("z_sign", -1.0)),
    }

    intr_q = {
        "fx": int(q_fx),
        "fy": int(q_fy),
        "cx": int(q_cx),
        "cy": int(q_cy),
    }

    clipped = bool(c_fx or c_fy or c_cx or c_cy)

    return intr_q, intr_dec, clipped


def param6_from_frame(frame, depth_scale):
    r = np.array(frame["rvec"], dtype=np.float32)
    t = np.array(frame["tvec"], dtype=np.float32) / depth_scale
    return np.concatenate([r, t], axis=0)


def rt_from_param6(p, depth_scale):
    return {
        "rvec": p[:3].astype(float).tolist(),
        "tvec": (p[3:] * depth_scale).astype(float).tolist(),
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

def make_projection_precompute(w, h, intr):
    """
    picture size와 intrinsic이 고정이면 한 번만 계산.

    NDC 아님.
    pixel-coordinate 기반:
      x_norm = (x - cx) / fx
      y_norm = (y - cy) / fy
    """
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    z_sign = float(intr["z_sign"])

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    return {
        "w": int(w),
        "h": int(h),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "z_sign": z_sign,
        "x_norm": x_norm.astype(np.float32),
        "y_norm": y_norm.astype(np.float32),
    }


def backward_map_fast_pixel_coord(depth_linear, precomp, rt):
    """
    NDC / ProjectionMatrix / InvProjectionMatrix 미사용.

    target pixel + target depth -> target camera coord -> previous camera coord
    -> previous pixel coordinate.

    기존:
      X = x_norm * z
      Y = y_norm * z
      Z = z_sign * z
      Xp = R * [X,Y,Z] + t

    최적화:
      Xp = z * (R00*x_norm + R01*y_norm + R02*z_sign) + tx
      Yp = z * (R10*x_norm + R11*y_norm + R12*z_sign) + ty
      Zp = z * (R20*x_norm + R21*y_norm + R22*z_sign) + tz
    """
    w = precomp["w"]
    h = precomp["h"]

    fx = precomp["fx"]
    fy = precomp["fy"]
    cx = precomp["cx"]
    cy = precomp["cy"]
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
    return dst.astype(np.uint16)


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


def backward_warp_yuv420(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    y = remap_plane(prev_y, map_x, map_y, bit_depth, 0)

    map_x_uv, map_y_uv = downsample_luma_map_to_chroma_map(map_x, map_y)

    neutral = 128 if bit_depth <= 8 else 512

    u = remap_plane(prev_u, map_x_uv, map_y_uv, bit_depth, neutral)
    v = remap_plane(prev_v, map_x_uv, map_y_uv, bit_depth, neutral)

    return y, u, v


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

    ap.add_argument("--coded-width", type=int, default=None)
    ap.add_argument("--coded-height", type=int, default=None)

    ap.add_argument("--pad-left", type=int, default=0)
    ap.add_argument("--pad-top", type=int, default=0)

    ap.add_argument("--bit-depth", type=int, default=10)

    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--out-q-jsonl", required=True)

    ap.add_argument("--pred-n", type=int, default=3)
    ap.add_argument("--pred-degree", type=int, default=2)

    ap.add_argument("--ext-bits", type=int, default=8)
    ap.add_argument("--r-step", type=float, default=2 ** -12)
    ap.add_argument("--t-step-norm", type=float, default=2 ** -10)

    ap.add_argument("--intr-f-max", type=float, default=4.0)
    ap.add_argument("--intr-c-min", type=float, default=-1.0)
    ap.add_argument("--intr-c-max", type=float, default=2.0)

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

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

    src_w = args.width
    src_h = args.height
    bit_depth = args.bit_depth

    pad_left = args.pad_left
    pad_top = args.pad_top

    coded_w = (
        args.coded_width
        if args.coded_width is not None
        else align_to(src_w + pad_left, 4)
    )

    coded_h = (
        args.coded_height
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

    depth_scale = float(header["depth_scale"])
    intr_gt_original = header["intrinsic"]

    intr_gt_padded = make_padded_intrinsic_from_original(
        intr_gt_original,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    intr_q, intr_dec, intr_clip = quantize_intrinsic_16(
        intr_gt_padded,
        coded_w,
        coded_h,
        f_max=args.intr_f_max,
        c_min=args.intr_c_min,
        c_max=args.intr_c_max,
    )

    # ------------------------------------------------------------
    # 핵심 precompute
    # ------------------------------------------------------------
    projection_precomp = make_projection_precompute(
        coded_w,
        coded_h,
        intr_dec,
    )

    seq_count = count_frames(seq_yuv, src_w, src_h, bit_depth)
    depth_count = count_frames(depth_yuv, src_w, src_h, 10)
    max_poc = min(seq_count, depth_count, max(frames.keys()) + 1)

    decoded_hist = []

    q_abs_max = signed_q_abs_max(args.ext_bits)

    intrinsic_bits = 4 * 16
    depth_scale_bits = 4
    z_sign_bits = 1
    header_bits = intrinsic_bits + depth_scale_bits + z_sign_bits

    total_ext_bits = 0
    total_ext_bits_r = 0
    total_ext_bits_t = 0
    total_ext_bits_each = np.zeros(6, dtype=np.int64)
    total_coded_frames = 0
    total_clipped_frames = 0

    ys_active, xs_active = active_slice(src_w, src_h, pad_left, pad_top)

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

            "projection_mode": "fast_pixel_coordinate_no_ndc",
            "precompute": [
                "x_norm=(x-cx)/fx",
                "y_norm=(y-cy)/fy",
                "fx,fy,cx,cy,z_sign cached",
            ],
            "depth_padding": "edge",
            "image_padding": "edge",

            "depth_scale": depth_scale,

            "intrinsic_gt_original": intr_gt_original,
            "intrinsic_gt_padded": intr_gt_padded,
            "intrinsic_q16": intr_q,
            "intrinsic_dec": intr_dec,
            "intrinsic_clipped": intr_clip,

            "extrinsic_bits": args.ext_bits,
            "extrinsic_q_abs_max": q_abs_max,
            "extrinsic_q_range": [-q_abs_max, q_abs_max],
            "r_step": args.r_step,
            "t_step_norm": args.t_step_norm,
            "pred_n": args.pred_n,
            "pred_degree": args.pred_degree,

            "bit_count": {
                "intrinsic_bits": intrinsic_bits,
                "depth_scale_bits": depth_scale_bits,
                "z_sign_bits": z_sign_bits,
                "header_bits": header_bits,
                "extrinsic_code": "signed_truncated_exp_golomb",
            },

            "param6_order": [
                "rx",
                "ry",
                "rz",
                "tx_over_depth_scale",
                "ty_over_depth_scale",
                "tz_over_depth_scale",
            ],
        }) + "\n")

        for poc in range(max_poc):
            cur_y, cur_u, cur_v = read_yuv420(
                seq_yuv,
                poc,
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
                    "q_residual": [0, 0, 0, 0, 0, 0],
                    "q_residual_bits": [0, 0, 0, 0, 0, 0],
                    "q_residual_total_bits": 0,
                    "param6_dec": p0_dec.astype(float).tolist(),
                    "mae_y_active": 0.0,
                    "mae_y_coded": 0.0,
                }) + "\n")
                continue

            if poc not in frames:
                raise RuntimeError(f"POC {poc} not found in param jsonl")

            p_gt = param6_from_frame(frames[poc], depth_scale)

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
                q_abs_max=q_abs_max,
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

            rt_dec = rt_from_param6(p_dec, depth_scale)

            # ------------------------------------------------------------
            # depth padding
            # ------------------------------------------------------------
            depth_y, _, _ = read_yuv420(
                depth_yuv,
                poc,
                src_w,
                src_h,
                10,
            )

            depth_linear = depth_y.astype(np.float32) * depth_scale

            depth_linear_pad = pad_2d_edge(
                depth_linear,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            ).astype(np.float32)

            # ------------------------------------------------------------
            # Fast projection
            # NDC 없음.
            # ProjectionMatrix 없음.
            # InvProjectionMatrix 없음.
            # x_norm/y_norm precompute 사용.
            # ------------------------------------------------------------
            map_x, map_y = backward_map_fast_pixel_coord(
                depth_linear=depth_linear_pad,
                precomp=projection_precomp,
                rt=rt_dec,
            )

            prev_y, prev_u, prev_v = read_yuv420(
                seq_yuv,
                poc - 1,
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

            wy, wu, wv = backward_warp_yuv420(
                prev_y_pad,
                prev_u_pad,
                prev_v_pad,
                map_x,
                map_y,
                bit_depth,
            )

            write_yuv420(out_yuv, wy, wu, wv)

            mae_y_coded = float(np.mean(np.abs(
                wy.astype(np.float32) - cur_y_pad.astype(np.float32)
            )))

            mae_y_active = float(np.mean(np.abs(
                wy[ys_active, xs_active].astype(np.float32)
                - cur_y_pad[ys_active, xs_active].astype(np.float32)
            )))

            clipped = bool(np.any(clip_r) or np.any(clip_t))
            if clipped:
                total_clipped_frames += 1

            fq.write(json.dumps({
                "poc": poc,
                "q_residual": q_residual.astype(int).tolist(),
                "q_residual_bits": q_bits_each,
                "q_residual_total_bits": q_bits_total,
                "param6_pred": p_pred.astype(float).tolist(),
                "param6_dec": p_dec.astype(float).tolist(),
                "param6_gt": p_gt.astype(float).tolist(),
                "clipped": clipped,
                "mae_y_active": mae_y_active,
                "mae_y_coded": mae_y_coded,
            }) + "\n")

            print(
                f"[{poc:04d}/{max_poc - 1:04d}] "
                f"Y-MAE-active={mae_y_active:.3f}, "
                f"Y-MAE-coded={mae_y_coded:.3f}, "
                f"clipped={clipped}, "
                f"param_bits={q_bits_total}, "
                f"bits_each={q_bits_each}"
            )

    total_bits = header_bits + total_ext_bits

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
    print("projection            : fast pixel-coordinate, no NDC")
    print("precompute            : x_norm, y_norm")
    print("image/depth padding   : edge")
    print("------------------------------------------------------------")
    print("Intrinsic")
    print("------------------------------------------------------------")
    print(f"original intrinsic    : {intr_gt_original}")
    print(f"padded intrinsic gt   : {intr_gt_padded}")
    print(f"padded intrinsic dec  : {intr_dec}")
    print(f"intrinsic clipped     : {intr_clip}")
    print("============================================================")
    print("Bit summary")
    print("============================================================")
    print(f"intrinsic bits        : {intrinsic_bits} bits")
    print(f"  fx, fy, cx, cy      : 16 bits each")
    print(f"depth_scale bits      : {depth_scale_bits} bits")
    print(f"z_sign bits           : {z_sign_bits} bits")
    print(f"header bits           : {header_bits} bits")
    print("------------------------------------------------------------")
    print(f"extrinsic code        : signed truncated Exp-Golomb")
    print(f"extrinsic q range     : [-{q_abs_max}, {q_abs_max}]")
    print(f"coded frames          : {total_coded_frames}")
    print(f"clipped frames        : {total_clipped_frames}")
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
            "avg bits each         : "
            f"rx={avg_bits_each[0]:.3f}, "
            f"ry={avg_bits_each[1]:.3f}, "
            f"rz={avg_bits_each[2]:.3f}, "
            f"tx={avg_bits_each[3]:.3f}, "
            f"ty={avg_bits_each[4]:.3f}, "
            f"tz={avg_bits_each[5]:.3f}"
        )

    print("------------------------------------------------------------")
    print(f"total bits            : {total_bits} bits")
    print("============================================================")
    print("Done.")
    print(f"warped yuv            : {out_yuv}")
    print(f"q jsonl               : {out_q_jsonl}")


if __name__ == "__main__":
    main()






























import torch
import torch.nn.functional as F

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
    assert h % 4 == 0 and w % 4 == 0

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
    """
    src: numpy 2D, uint16/uint8
    map_x/map_y: luma pixel-wise map
    output: numpy 2D, same dtype as src

    4x4 대표 flow + 6tap interpolation.
    """
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

    # tap range: ix-2 ... ix+3
    # asymmetric pad: left/top=2, right/bottom=3
    src_pad = F.pad(src_t, (2, 3, 2, 3), mode="replicate")

    # patches: [1, 36, h*w]
    patches_all = F.unfold(src_pad, kernel_size=(6, 6), stride=1)

    # each output pixel wants patch whose top-left corresponds to ix, iy
    col_idx = (iy * w + ix).reshape(-1)
    patches = patches_all[0, :, col_idx]  # [36, h*w]

    coeff = torch.from_numpy(LUMA_6TAP_32_NP).to(device)

    cx = coeff[frac_x.reshape(-1)]  # [N, 6]
    cy = coeff[frac_y.reshape(-1)]  # [N, 6]

    # 2D separable filter coeff: cy outer cx
    weight = (cy[:, :, None] * cx[:, None, :]).reshape(-1, 36)  # [N, 36]

    val = torch.sum(patches.transpose(0, 1) * weight, dim=1)

    # coeff precision 8 + 8 = 16
    val = torch.round(val / 65536.0)
    val = val.clamp(0, maxv)

    val = val.reshape(h, w)
    val = torch.where(valid, val, torch.zeros_like(val))

    out = val.detach().cpu().numpy()

    if bit_depth <= 8:
        return out.astype(np.uint8)
    return out.astype(np.uint16)







def backward_warp_yuv420_subblk4_6tap_torch(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    wy = remap_plane_subblk4_6tap_torch(
        prev_y,
        map_x,
        map_y,
        bit_depth,
    )

    # U/V는 일단 기존 bilinear 유지
    map_x_uv, map_y_uv = downsample_luma_map_to_chroma_map(map_x, map_y)
    neutral = 128 if bit_depth <= 8 else 512

    wu = remap_plane(prev_u, map_x_uv, map_y_uv, bit_depth, neutral)
    wv = remap_plane(prev_v, map_x_uv, map_y_uv, bit_depth, neutral)

    return wy, wu, wv


















