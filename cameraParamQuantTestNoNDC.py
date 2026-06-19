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


def validate_yuv420_padding(src_w, src_h, coded_w, coded_h,
                            pad_left, pad_top, pad_right, pad_bottom):
    values = {
        "src_w": src_w,
        "src_h": src_h,
        "coded_w": coded_w,
        "coded_h": coded_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }

    for name, v in values.items():
        if v < 0:
            raise ValueError(f"{name} must be non-negative: {v}")

    # 4:2:0 chroma plane 때문에 padding도 짝수 단위가 안전함.
    for name, v in values.items():
        if v % 2 != 0:
            raise ValueError(
                f"{name} must be even for YUV420 padding: {v}"
            )


def pad_2d_edge(arr, coded_w, coded_h, pad_left, pad_top):
    h, w = arr.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top

    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"Cannot pad array {w}x{h} to {coded_w}x{coded_h} "
            f"with pad_left={pad_left}, pad_top={pad_top}"
        )

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
    ys = slice(pad_top, pad_top + src_h)
    xs = slice(pad_left, pad_left + src_w)
    return ys, xs


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
    """
    bits=8이면 signed residual range는 [-127, 127].
    2's complement의 [-128, 127]이 아니라 symmetric range를 사용.
    """
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
    원본 image coordinate 기준 intrinsic을 padded image coordinate 기준으로 변환.

    오른쪽/아래쪽 padding만 있으면 pad_left=pad_top=0이므로 그대로.
    왼쪽 padding이 있으면 cx += pad_left.
    위쪽 padding이 있으면 cy += pad_top.
    아래쪽 padding은 cy를 바꾸지 않음.
    오른쪽 padding은 cx를 바꾸지 않음.
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
    intrinsic은 고정 16bit 양자화.

    여기서 w, h는 decoder가 아는 coded picture size를 넣는다.
    즉 padded projection을 하려면 coded_w, coded_h 기준으로 quant/dequant한다.

    fx, fy는 각각 width/height로 normalize.
    cx, cy도 각각 width/height로 normalize.
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
    """
    signed Exp-Golomb mapping:
       0 -> 0
      +1 -> 1
      -1 -> 2
      +2 -> 3
      -2 -> 4
      ...
    """
    x = int(x)
    if x == 0:
        return 0
    if x > 0:
        return 2 * x - 1
    return -2 * x


def ue_exp_golomb_bits(code_num):
    """
    unsigned Exp-Golomb code length.
    code_num=0 -> 1 bit
    code_num=1,2 -> 3 bits
    code_num=3~6 -> 5 bits
    """
    code_num = int(code_num)
    if code_num < 0:
        raise ValueError("code_num must be non-negative")

    k = (code_num + 1).bit_length() - 1
    return 2 * k + 1


def signed_truncated_exp_golomb_bits(x, q_abs_max):
    """
    signed residual x in [-q_abs_max, q_abs_max]의 bit 수 계산.
    truncated range 밖이면 error.
    """
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
    """
    decoded_hist에는 이미 복원된 param6만 저장됨.
    predictor도 반드시 decoded 값만 사용.
    """
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
# Pixel-coordinate Backward Warp
# ============================================================

def make_grid(w, h):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return x, y


def backward_map_from_depth_and_params_pixel_coord(
    depth_linear,
    intr,
    rt,
    w,
    h,
):
    """
    NDC / ProjectionMatrix / InvProjectionMatrix를 전혀 사용하지 않는 방식.

    target padded pixel (x, y)와 target depth z로 target camera space 점을 만들고,
    decoded Rt로 previous camera space로 보낸 뒤,
    같은 intrinsic으로 previous image coordinate에 project한다.

    x_cam = (x - cx) / fx * z
    y_cam = (y - cy) / fy * z
    z_cam = z_sign * z

    padding이 있더라도 intr이 padded coordinate 기준이면 그대로 동작한다.
    """
    x, y = make_grid(w, h)

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    z_sign = float(intr["z_sign"])

    z = depth_linear.astype(np.float32)

    X = np.empty((h, w, 3), dtype=np.float32)
    X[..., 0] = (x - cx) / fx * z
    X[..., 1] = (y - cy) / fy * z
    X[..., 2] = z_sign * z

    rvec = np.array(rt["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(rt["tvec"], dtype=np.float32).reshape(1, 1, 3)

    R, _ = cv2.Rodrigues(rvec)

    # 기존 코드와 동일한 convention 유지:
    # Xp = X @ R.T + t
    Xp = X @ R.T + tvec

    zprev = np.maximum(np.abs(Xp[..., 2]), 1e-8)

    map_x = fx * (Xp[..., 0] / zprev) + cx
    map_y = fy * (Xp[..., 1] / zprev) + cy

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (Xp[..., 2] * z_sign > 0)
        & (map_x >= 0)
        & (map_x <= w - 1)
        & (map_y >= 0)
        & (map_y <= h - 1)
        & (z > 0)
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
    """
    4:2:0용 luma map -> chroma map 변환.

    기존처럼 cv2.resize 후 0.5를 곱하면 invalid=-1이 주변과 섞일 수 있다.
    여기서는 2x2 luma block의 valid sample만 평균낸 뒤 chroma coordinate로 변환한다.

    luma coordinate -> chroma coordinate:
      x_uv = x_luma * 0.5
      y_uv = y_luma * 0.5
    """
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
    h, w = prev_y.shape

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

    # 원본 source size
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)

    # decoder/coded picture size
    # 생략하면 pad_left/pad_top을 포함한 뒤 4의 배수로 자동 align
    ap.add_argument("--coded-width", type=int, default=None)
    ap.add_argument("--coded-height", type=int, default=None)

    # padding 위치
    # 일반적인 right/bottom padding이면 둘 다 0.
    # left/top padding을 시뮬레이션하려면 여기에 값 지정.
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

    # ------------------------------------------------------------
    # 핵심:
    # 원본 intrinsic을 padded image coordinate 기준으로 변환한 뒤
    # coded_w/coded_h 기준으로 quant/dequant한다.
    # decoder는 원본 W/H 없이 coded_w/coded_h만 알면 됨.
    # ------------------------------------------------------------
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

    seq_count = count_frames(seq_yuv, src_w, src_h, bit_depth)
    depth_count = count_frames(depth_yuv, src_w, src_h, 10)
    max_poc = min(seq_count, depth_count, max(frames.keys()) + 1)

    decoded_hist = []

    # ------------------------------------------------------------
    # Bit count constants
    # ------------------------------------------------------------
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

            "projection_mode": "pixel_coordinate_no_ndc",
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

            # ------------------------------------------------------------
            # GT camera parameter
            # ------------------------------------------------------------
            p_gt = param6_from_frame(frames[poc], depth_scale)

            # predictor uses decoded parameters only
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
            # Depth:
            # 원본 depth를 읽고, coded picture size로 edge padding.
            # projection은 padded depth 전체에 대해 수행.
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
            # Pixel-coordinate backward projection.
            # NDC / ProjectionMatrix 사용 안 함.
            # ------------------------------------------------------------
            map_x, map_y = backward_map_from_depth_and_params_pixel_coord(
                depth_linear=depth_linear_pad,
                intr=intr_dec,
                rt=rt_dec,
                w=coded_w,
                h=coded_h,
            )

            # ------------------------------------------------------------
            # Reference picture:
            # 이전 GT frame을 읽고 coded picture size로 edge padding.
            # decoder 입장에서는 padded picture 전체가 참조 가능.
            # ------------------------------------------------------------
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
    print("projection            : pixel-coordinate, no NDC")
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
