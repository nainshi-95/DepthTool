import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


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


def quantize_intrinsic_16(intr, w, h, f_max=4.0, c_min=-1.0, c_max=2.0):
    """
    intrinsic은 고정 16bit 양자화.
    fx, fy는 각각 width/height로 normalize 후 [0, f_max]에서 uint16.
    cx, cy는 각각 width/height로 normalize 후 [c_min, c_max]에서 uint16.
    """
    fx_n = intr["fx"] / w
    fy_n = intr["fy"] / h
    cx_n = intr["cx"] / w
    cy_n = intr["cy"] / h

    q_fx, d_fx, c_fx = quant_u(fx_n, 0.0, f_max, 16)
    q_fy, d_fy, c_fy = quant_u(fy_n, 0.0, f_max, 16)
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


# ============================================================
# Rebased absolute extrinsic param6
# ============================================================

def param6_from_frame(frame, depth_scale):
    """
    입력 JSONL의 rvec/tvec는 POC0 기준 rebased absolute W2C라고 가정.

    W2C_rebased[poc] = W2C[poc] @ C2W[0]

    param6 order:
        rx, ry, rz,
        tx / depth_scale,
        ty / depth_scale,
        tz / depth_scale
    """
    r = np.array(frame["rvec"], dtype=np.float32)
    t = np.array(frame["tvec"], dtype=np.float32) / depth_scale

    return np.concatenate([r, t], axis=0).astype(np.float32)


def abs_w2c_matrix_from_param6(p_abs, depth_scale):
    """
    Rebased absolute param6 -> W2C 4x4 matrix.

    X_cam = R * X_world_rebased + t
    """
    rvec = np.array(p_abs[:3], dtype=np.float32).reshape(3, 1)
    tvec = np.array(p_abs[3:], dtype=np.float32) * depth_scale

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = tvec.astype(np.float32)

    return T


def invert_rigid_4x4(T):
    """
    Rigid transform inverse.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t

    return Ti


def derive_rt_cur_to_prev_from_abs_param6(p_abs_cur, p_abs_prev, depth_scale):
    """
    Rebased absolute W2C 두 개로 cur -> prev relative transform 생성.

    W2C_cur:
        X_cur = W2C_cur * X_world_rebased

    W2C_prev:
        X_prev = W2C_prev * X_world_rebased

    필요한 backward warp 변환:
        X_prev = T_cur_to_prev * X_cur

    따라서:
        T_cur_to_prev = W2C_prev @ C2W_cur
    """
    W2C_cur = abs_w2c_matrix_from_param6(p_abs_cur, depth_scale)
    W2C_prev = abs_w2c_matrix_from_param6(p_abs_prev, depth_scale)

    C2W_cur = invert_rigid_4x4(W2C_cur)

    T = W2C_prev @ C2W_cur

    R_rel = T[:3, :3].astype(np.float32)
    t_rel = T[:3, 3].astype(np.float32)

    rvec_rel, _ = cv2.Rodrigues(R_rel)

    return {
        "rvec": rvec_rel.reshape(3).astype(float).tolist(),
        "tvec": t_rel.reshape(3).astype(float).tolist(),
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
    decoded_hist에는 이미 복원된 rebased absolute param6만 저장됨.
    predictor도 반드시 decoded absolute 값만 사용.
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
# Warp
# ============================================================

def make_grid(w, h):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return x, y


def backward_map_from_depth_and_params(depth_linear, intr, rt, w, h):
    x, y = make_grid(w, h)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    z_sign = intr["z_sign"]

    z = depth_linear.astype(np.float32)

    X = np.empty((h, w, 3), dtype=np.float32)
    X[..., 0] = (x - cx) / fx * z
    X[..., 1] = (y - cy) / fy * z
    X[..., 2] = z_sign * z

    rvec = np.array(rt["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(rt["tvec"], dtype=np.float32).reshape(1, 1, 3)

    R, _ = cv2.Rodrigues(rvec)

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
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )

    dst = np.clip(np.round(dst), 0, maxv)

    if bit_depth <= 8:
        return dst.astype(np.uint8)

    return dst.astype(np.uint16)


def backward_warp_yuv420(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    h, w = prev_y.shape
    uv_w = w // 2
    uv_h = h // 2

    y = remap_plane(prev_y, map_x, map_y, bit_depth, 0)

    map_x_uv = cv2.resize(
        map_x,
        (uv_w, uv_h),
        interpolation=cv2.INTER_LINEAR,
    ) * 0.5

    map_y_uv = cv2.resize(
        map_y,
        (uv_w, uv_h),
        interpolation=cv2.INTER_LINEAR,
    ) * 0.5

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

    w = args.width
    h = args.height
    bit_depth = args.bit_depth

    header, frames = load_param_jsonl(param_jsonl)

    depth_scale = float(header["depth_scale"])
    intr_gt = header["intrinsic"]

    input_extrinsic_type = header.get(
        "extrinsic_type",
        "absolute_world_to_camera_rebased_to_poc0",
    )

    if input_extrinsic_type != "absolute_world_to_camera_rebased_to_poc0":
        print(
            "[WARN] input extrinsic_type is not "
            f"'absolute_world_to_camera_rebased_to_poc0': {input_extrinsic_type}"
        )

    intr_q, intr_dec, intr_clip = quantize_intrinsic_16(
        intr_gt,
        w,
        h,
        f_max=args.intr_f_max,
        c_min=args.intr_c_min,
        c_max=args.intr_c_max,
    )

    seq_count = count_frames(seq_yuv, w, h, bit_depth)
    depth_count = count_frames(depth_yuv, w, h, 10)

    if not frames:
        raise RuntimeError("no frame parameters found in param jsonl")

    if 0 not in frames:
        raise RuntimeError("POC 0 not found in param jsonl")

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

    with open(out_q_jsonl, "w", encoding="utf-8") as fq:
        fq.write(json.dumps({
            "type": "header",

            "input_extrinsic_type": "absolute_world_to_camera_rebased_to_poc0",
            "coded_extrinsic_type": "absolute_world_to_camera_rebased_to_poc0",
            "extrinsic_rotation": "rodrigues_rvec",
            "extrinsic_translation": "tvec",
            "anchor_poc": 0,
            "anchor_pose": "identity",

            "depth_scale": depth_scale,
            "intrinsic_q16": intr_q,
            "intrinsic_dec": intr_dec,
            "intrinsic_gt": intr_gt,
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
                "poc0_extrinsic_bits": 0,
            },

            "param6_order": [
                "rebased_abs_rx",
                "rebased_abs_ry",
                "rebased_abs_rz",
                "rebased_abs_tx_over_depth_scale",
                "rebased_abs_ty_over_depth_scale",
                "rebased_abs_tz_over_depth_scale",
            ],
        }) + "\n")

        for poc in range(max_poc):
            cur_y, cur_u, cur_v = read_yuv420(seq_yuv, poc, w, h, bit_depth)

            if poc not in frames:
                raise RuntimeError(f"POC {poc} not found in param jsonl")

            # GT rebased absolute camera parameter
            p_gt = param6_from_frame(frames[poc], depth_scale)

            # ----------------------------------------------------
            # POC 0: rebased absolute anchor
            # ----------------------------------------------------
            if poc == 0:
                write_yuv420(out_yuv, cur_y, cur_u, cur_v)

                # 입력 JSONL이 정상이라면 이 값은 거의 [0,0,0,0,0,0]
                p0_dec = p_gt.copy()
                decoded_hist.append(p0_dec)

                fq.write(json.dumps({
                    "poc": 0,
                    "mode": "rebased_absolute_anchor_uncoded",
                    "q_residual": [0, 0, 0, 0, 0, 0],
                    "q_residual_bits": [0, 0, 0, 0, 0, 0],
                    "q_residual_total_bits": 0,
                    "param6_dec": p0_dec.astype(float).tolist(),
                    "param6_gt": p_gt.astype(float).tolist(),
                    "mae_y": 0.0,
                }) + "\n")

                print(f"[{poc:04d}/{max_poc - 1:04d}] copy original, rebased absolute anchor")
                continue

            # ----------------------------------------------------
            # Predict rebased absolute parameter
            # ----------------------------------------------------
            p_prev_dec = decoded_hist[-1]

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

            # ----------------------------------------------------
            # Warp 직전에 decoded absolute cur/prev로 relative RT 생성
            # ----------------------------------------------------
            rt_dec = derive_rt_cur_to_prev_from_abs_param6(
                p_abs_cur=p_dec,
                p_abs_prev=p_prev_dec,
                depth_scale=depth_scale,
            )

            # depth는 현재 frame depth를 그대로 사용
            depth_y, _, _ = read_yuv420(depth_yuv, poc, w, h, 10)
            depth_linear = depth_y.astype(np.float32) * depth_scale

            map_x, map_y = backward_map_from_depth_and_params(
                depth_linear=depth_linear,
                intr=intr_dec,
                rt=rt_dec,
                w=w,
                h=h,
            )

            # 영상 참조는 항상 이전 GT frame 사용
            prev_y, prev_u, prev_v = read_yuv420(
                seq_yuv,
                poc - 1,
                w,
                h,
                bit_depth,
            )

            wy, wu, wv = backward_warp_yuv420(
                prev_y,
                prev_u,
                prev_v,
                map_x,
                map_y,
                bit_depth,
            )

            write_yuv420(out_yuv, wy, wu, wv)

            mae_y = float(np.mean(np.abs(
                wy.astype(np.float32) - cur_y.astype(np.float32)
            )))

            clipped = bool(np.any(clip_r) or np.any(clip_t))

            if clipped:
                total_clipped_frames += 1

            decoded_hist.append(p_dec)

            fq.write(json.dumps({
                "poc": poc,
                "mode": "rebased_absolute_predictive",
                "q_residual": q_residual.astype(int).tolist(),
                "q_residual_bits": q_bits_each,
                "q_residual_total_bits": q_bits_total,

                "param6_pred": p_pred.astype(float).tolist(),
                "param6_dec": p_dec.astype(float).tolist(),
                "param6_gt": p_gt.astype(float).tolist(),

                "rt_cur_to_prev_dec": rt_dec,

                "clipped": clipped,
                "mae_y": mae_y,
            }) + "\n")

            print(
                f"[{poc:04d}/{max_poc - 1:04d}] "
                f"Y-MAE={mae_y:.3f}, clipped={clipped}, "
                f"param_bits={q_bits_total}, "
                f"bits_each={q_bits_each}"
            )

    total_bits = header_bits + total_ext_bits

    print("============================================================")
    print("Bit summary")
    print("============================================================")
    print(f"intrinsic bits        : {intrinsic_bits} bits")
    print(f"  fx, fy, cx, cy      : 16 bits each")
    print(f"depth_scale bits      : {depth_scale_bits} bits")
    print(f"z_sign bits           : {z_sign_bits} bits")
    print(f"header bits           : {header_bits} bits")
    print("------------------------------------------------------------")
    print("extrinsic type        : rebased absolute WorldToCamera")
    print("anchor POC            : 0")
    print("anchor pose           : identity")
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
    print(f"warped yuv : {out_yuv}")
    print(f"q jsonl    : {out_q_jsonl}")


if __name__ == "__main__":
    main()
