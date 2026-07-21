#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_quant_warp_merged_gop_jsonl.py

Camera-parameter quantization and backward-warp simulation for the merged
multi-GOP JSONL produced by the GOP merge script.

Required JSONL properties
-------------------------
Header:
    pose_mode == "current_to_previous"
    gops[*].gop_idx
    gops[*].depth_scale_real

Frame record:
    camera_record_idx
    gop_idx
    gop_name
    local_poc
    poc
    frame_idx
    depth_frame_idx
    depth_source_gop_idx
    rvec
    tvec
    intrinsic
    intrinsic_delta
    depth_scale_real

Merged-stream behavior
----------------------
* Camera records are processed in JSONL camera_record_idx order.
* Predictor history is reset independently at every GOP local_poc 0.
* The first record of each GOP is implicit R=I, t=0 and copied unchanged.
* For local_poc > 0, the current current-to-previous relative pose is
  predicted from reconstructed relative-pose history.
* With --pred-n 1 --pred-degree 0:
      predictor = previous reconstructed relative pose
      residual  = current relative pose - predictor
* Translation is normalized by the current camera GOP's depth_scale_real,
  matching the original low-delay script.
* Target depth is read using depth_frame_idx and dequantized with the scale of
  depth_source_gop_idx. This is essential for overlapping GOP boundaries,
  where the earlier GOP owns the single stored depth frame.
* Intrinsic reconstruction is reset per GOP. The first intrinsic of each GOP
  is quantized as four 16-bit values, and later intrinsic_delta values are
  predictively accumulated.
* Output YUV is written in camera-record order. Therefore an overlapping POC
  appears twice, once for each GOP-local camera record.

Projection:
    current target pixel/depth -> previous reference camera -> previous pixel

Depth projection modes:
    per_pixel
    block_inv_plane:
        1/z = a*(x-cx) + b*(y-cy) + c

Output:
    --out-yuv:
        one YUV frame per camera JSONL frame record, in camera_record_idx order
    --out-q-jsonl:
        one coding/metric record per camera JSONL frame record
    --out-plane-depth-yuv:
        optional, one modeled target-depth frame per camera record
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

def align_to(x: int, a: int) -> int:
    return ((int(x) + int(a) - 1) // int(a)) * int(a)


def calc_padding(
    src_w: int,
    src_h: int,
    coded_w: int,
    coded_h: int,
    pad_left: int,
    pad_top: int,
) -> tuple[int, int]:
    pad_right = coded_w - src_w - pad_left
    pad_bottom = coded_h - src_h - pad_top
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"Invalid padding: src=({src_w}x{src_h}), "
            f"coded=({coded_w}x{coded_h}), "
            f"pad_left={pad_left}, pad_top={pad_top}"
        )
    return pad_right, pad_bottom


def validate_yuv420_padding(**vals: int) -> None:
    for name, value in vals.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative: {value}")
        if int(value) % 2:
            raise ValueError(f"{name} must be even for YUV420: {value}")


def pad_2d_edge(
    arr: np.ndarray,
    coded_w: int,
    coded_h: int,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    h, w = arr.shape
    pad_right = coded_w - w - pad_left
    pad_bottom = coded_h - h - pad_top
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError("negative padding")
    return np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )


def pad_yuv420_edge(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coded_w: int,
    coded_h: int,
    pad_left: int,
    pad_top: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        pad_2d_edge(y, coded_w, coded_h, pad_left, pad_top),
        pad_2d_edge(
            u,
            coded_w // 2,
            coded_h // 2,
            pad_left // 2,
            pad_top // 2,
        ),
        pad_2d_edge(
            v,
            coded_w // 2,
            coded_h // 2,
            pad_left // 2,
            pad_top // 2,
        ),
    )


def active_slice(
    src_w: int,
    src_h: int,
    pad_left: int,
    pad_top: int,
) -> tuple[slice, slice]:
    return (
        slice(pad_top, pad_top + src_h),
        slice(pad_left, pad_left + src_w),
    )


def calc_psnr(
    a: np.ndarray,
    b: np.ndarray,
    bit_depth: int,
    valid: np.ndarray | None = None,
) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return float("nan")
    mse = float(np.mean((aa[mask] - bb[mask]) ** 2))
    if mse <= 0:
        return float("inf")
    maxv = float((1 << int(bit_depth)) - 1)
    return float(10.0 * np.log10((maxv * maxv) / mse))


def calc_mae(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(aa[mask] - bb[mask])))


def calc_float_metrics(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray | None = None,
    peak: float | None = None,
) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
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
    diff = bb[mask] - aa[mask]
    mse = float(np.mean(diff * diff))
    if peak is None:
        peak = float(np.max(np.abs(aa[mask])))
    peak = max(float(peak), 1e-12)
    return {
        "valid_count": int(np.count_nonzero(mask)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "psnr": (
            float("inf")
            if mse <= 0
            else float(10.0 * np.log10((peak * peak) / mse))
        ),
        "max_abs_err": float(np.max(np.abs(diff))),
    }


def json_safe_float(x: Any) -> Any:
    if x is None:
        return None
    value = float(x)
    if np.isnan(value):
        return None
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


# ============================================================
# YUV
# ============================================================

def frame_size_yuv420(w: int, h: int, bit_depth: int) -> int:
    bps = 1 if int(bit_depth) <= 8 else 2
    return (
        int(w) * int(h)
        + 2 * (int(w) // 2) * (int(h) // 2)
    ) * bps


def count_frames(
    path: str | Path,
    w: int,
    h: int,
    bit_depth: int,
) -> int:
    frame_size = frame_size_yuv420(w, h, bit_depth)
    size = os.path.getsize(path)
    trailing = size % frame_size
    if trailing:
        print(
            f"[WARN] trailing bytes ignored: {path}, trailing={trailing}",
            flush=True,
        )
    return size // frame_size


def yuv_dtype(bit_depth: int):
    return np.uint8 if int(bit_depth) <= 8 else np.dtype("<u2")


def read_yuv420(
    path: str | Path,
    idx: int,
    w: int,
    h: int,
    bit_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = int(idx)
    if idx < 0:
        raise ValueError(f"negative frame index: {idx}")
    dtype = yuv_dtype(bit_depth)
    y_size = int(w) * int(h)
    uv_size = (int(w) // 2) * (int(h) // 2)
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


def write_yuv420(
    path: str | Path,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> None:
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(y).tobytes())
        f.write(np.ascontiguousarray(u).tobytes())
        f.write(np.ascontiguousarray(v).tobytes())


def write_depth_linear_as_yuv420p10le(
    path: str | Path,
    depth_linear: np.ndarray,
    depth_scale_real: float,
) -> None:
    if float(depth_scale_real) <= 0:
        raise ValueError("depth_scale_real must be positive")
    h, w = depth_linear.shape
    y = np.clip(
        np.rint(
            np.asarray(depth_linear, dtype=np.float64)
            / float(depth_scale_real)
        ),
        0,
        1023,
    ).astype("<u2")
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    write_yuv420(path, y, uv, uv)


# ============================================================
# Merged JSONL
# ============================================================

def intrinsic_dict(obj: dict[str, Any]) -> dict[str, float]:
    return {
        "fx": float(obj["fx"]),
        "fy": float(obj["fy"]),
        "cx": float(obj["cx"]),
        "cy": float(obj["cy"]),
        "z_sign": float(obj.get("z_sign", 1.0)),
    }


def load_merged_param_jsonl(
    path: str | Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, float],
]:
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                if header is None:
                    header = obj
                continue
            if obj.get("type") != "frame":
                continue

            required = [
                "gop_idx",
                "local_poc",
                "poc",
                "rvec",
                "tvec",
                "intrinsic",
                "intrinsic_delta",
            ]
            missing = [key for key in required if key not in obj]
            if missing:
                raise KeyError(
                    f"{path}:{line_no}: missing required fields {missing}"
                )

            rec = dict(obj)
            rec["_line_no"] = int(line_no)
            rec["camera_record_idx"] = int(
                rec.get("camera_record_idx", len(records))
            )
            rec["gop_idx"] = int(rec["gop_idx"])
            rec["gop_name"] = str(
                rec.get("gop_name", f"gop{rec['gop_idx']}")
            )
            rec["local_poc"] = int(rec["local_poc"])
            rec["poc"] = int(rec["poc"])
            rec["frame_idx"] = int(
                rec.get("frame_idx", rec["poc"])
            )
            rec["depth_frame_idx"] = int(
                rec.get("depth_frame_idx", rec["poc"])
            )
            rec["depth_source_gop_idx"] = int(
                rec.get("depth_source_gop_idx", rec["gop_idx"])
            )
            rec["rvec"] = (
                np.asarray(rec["rvec"], dtype=np.float32)
                .reshape(3)
            )
            rec["tvec"] = (
                np.asarray(rec["tvec"], dtype=np.float32)
                .reshape(3)
            )
            rec["intrinsic"] = intrinsic_dict(rec["intrinsic"])
            rec["intrinsic_delta"] = (
                np.asarray(rec["intrinsic_delta"], dtype=np.float32)
                .reshape(4)
            )
            if "depth_scale_real" in rec:
                rec["depth_scale_real"] = float(rec["depth_scale_real"])
            records.append(rec)

    if header is None:
        raise RuntimeError("header line not found")
    if not records:
        raise RuntimeError("no frame records found")
    if header.get("pose_mode") != "current_to_previous":
        raise RuntimeError(
            "This script requires merged JSONL generated with "
            "--pose-mode current_to_previous"
        )

    records.sort(
        key=lambda r: (
            int(r["camera_record_idx"]),
            int(r["_line_no"]),
        )
    )

    seen_record_indices: set[int] = set()
    for rec in records:
        idx = int(rec["camera_record_idx"])
        if idx in seen_record_indices:
            raise ValueError(f"duplicate camera_record_idx: {idx}")
        seen_record_indices.add(idx)

    by_gop: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_gop.setdefault(int(rec["gop_idx"]), []).append(rec)

    for gop_idx, gop_records in by_gop.items():
        gop_records.sort(key=lambda r: int(r["local_poc"]))
        local = [int(r["local_poc"]) for r in gop_records]
        if local != list(range(len(local))):
            raise ValueError(
                f"GOP {gop_idx}: local_poc must be contiguous 0..N-1, "
                f"got {local}"
            )
        if not np.allclose(gop_records[0]["rvec"], 0, atol=1e-7):
            raise ValueError(f"GOP {gop_idx}: local_poc 0 rvec must be zero")
        if not np.allclose(gop_records[0]["tvec"], 0, atol=1e-7):
            raise ValueError(f"GOP {gop_idx}: local_poc 0 tvec must be zero")

    scale_by_gop: dict[int, float] = {}
    for item in header.get("gops", []):
        if (
            isinstance(item, dict)
            and "gop_idx" in item
            and "depth_scale_real" in item
        ):
            scale_by_gop[int(item["gop_idx"])] = float(
                item["depth_scale_real"]
            )

    for rec in records:
        if (
            rec["gop_idx"] not in scale_by_gop
            and "depth_scale_real" in rec
        ):
            scale_by_gop[rec["gop_idx"]] = float(
                rec["depth_scale_real"]
            )

    missing_scales = sorted(
        {
            int(rec["gop_idx"])
            for rec in records
            if int(rec["gop_idx"]) not in scale_by_gop
        }
        | {
            int(rec["depth_source_gop_idx"])
            for rec in records
            if int(rec["depth_source_gop_idx"]) not in scale_by_gop
        }
    )
    if missing_scales:
        raise RuntimeError(
            f"depth_scale_real unavailable for GOPs {missing_scales}"
        )

    for gop_idx, scale in scale_by_gop.items():
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"invalid depth_scale_real for GOP {gop_idx}: {scale}"
            )

    return header, records, by_gop, scale_by_gop


# ============================================================
# Quantization
# ============================================================

def quant_u(
    value: Any,
    lo: float,
    hi: float,
    bits: int,
):
    qmax = (1 << int(bits)) - 1
    q = np.round((value - lo) / (hi - lo) * qmax)
    q = np.clip(q, 0, qmax).astype(np.int32)
    dec = q.astype(np.float32) / qmax * (hi - lo) + lo
    clipped = (value < lo) | (value > hi)
    return q, dec, clipped


def signed_q_abs_max(bits: int) -> int:
    if int(bits) < 2:
        raise ValueError("bits must be >=2")
    return (1 << (int(bits) - 1)) - 1


def quant_s(
    value: np.ndarray,
    step: float,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_abs_max = signed_q_abs_max(bits)
    q = np.round(
        np.asarray(value, dtype=np.float64) / float(step)
    )
    clipped = (q < -q_abs_max) | (q > q_abs_max)
    q = np.clip(q, -q_abs_max, q_abs_max).astype(np.int32)
    dec = q.astype(np.float32) * float(step)
    return q, dec, clipped


def make_padded_intrinsic_from_original(
    intr: dict[str, float],
    pad_left: int,
    pad_top: int,
) -> dict[str, float]:
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]) + float(pad_left),
        "cy": float(intr["cy"]) + float(pad_top),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }


def add_intrinsic_delta(
    intr: dict[str, float],
    delta: np.ndarray,
) -> dict[str, float]:
    return {
        "fx": float(intr["fx"]) + float(delta[0]),
        "fy": float(intr["fy"]) + float(delta[1]),
        "cx": float(intr["cx"]) + float(delta[2]),
        "cy": float(intr["cy"]) + float(delta[3]),
        "z_sign": float(intr.get("z_sign", 1.0)),
    }


def quantize_intrinsic_16(
    intr: dict[str, float],
    w: int,
    h: int,
    f_max: float = 4.0,
    c_min: float = -1.0,
    c_max: float = 2.0,
) -> tuple[dict[str, int], dict[str, float], bool]:
    fx_n = float(intr["fx"]) / float(w)
    fy_n = float(intr["fy"]) / float(h)
    cx_n = float(intr["cx"]) / float(w)
    cy_n = float(intr["cy"]) / float(h)

    q_fx, d_fx, c_fx = quant_u(fx_n, -f_max, f_max, 16)
    q_fy, d_fy, c_fy = quant_u(fy_n, -f_max, f_max, 16)
    q_cx, d_cx, c_cx = quant_u(cx_n, c_min, c_max, 16)
    q_cy, d_cy, c_cy = quant_u(cy_n, c_min, c_max, 16)

    return (
        {
            "fx": int(q_fx),
            "fy": int(q_fy),
            "cx": int(q_cx),
            "cy": int(q_cy),
        },
        {
            "fx": float(d_fx * w),
            "fy": float(d_fy * h),
            "cx": float(d_cx * w),
            "cy": float(d_cy * h),
            "z_sign": float(intr.get("z_sign", 1.0)),
        },
        bool(c_fx or c_fy or c_cx or c_cy),
    )


def quantize_intrinsic_delta_4(
    delta: np.ndarray,
    step: float,
    bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, dec, clipped = quant_s(
        np.asarray(delta, dtype=np.float32).reshape(4),
        step,
        bits,
    )
    return q.astype(np.int32), dec.astype(np.float32), clipped


def param6_from_frame(
    frame: dict[str, Any],
    camera_gop_depth_scale_real: float,
) -> np.ndarray:
    r = np.asarray(frame["rvec"], dtype=np.float32).reshape(3)
    t = (
        np.asarray(frame["tvec"], dtype=np.float32).reshape(3)
        / float(camera_gop_depth_scale_real)
    )
    return np.concatenate([r, t], axis=0)


def rt_from_param6(
    p: np.ndarray,
    camera_gop_depth_scale_real: float,
) -> dict[str, list[float]]:
    p = np.asarray(p, dtype=np.float32).reshape(6)
    return {
        "rvec": p[:3].astype(float).tolist(),
        "tvec": (
            p[3:] * float(camera_gop_depth_scale_real)
        ).astype(float).tolist(),
    }


# ============================================================
# Signed truncated Exp-Golomb
# ============================================================

def signed_to_code_num(x: int) -> int:
    x = int(x)
    if x == 0:
        return 0
    return 2 * x - 1 if x > 0 else -2 * x


def ue_exp_golomb_bits(code_num: int) -> int:
    code_num = int(code_num)
    if code_num < 0:
        raise ValueError("code_num must be non-negative")
    k = (code_num + 1).bit_length() - 1
    return 2 * k + 1


def signed_truncated_exp_golomb_bits(
    x: int,
    q_abs_max: int,
) -> int:
    x = int(x)
    if x < -q_abs_max or x > q_abs_max:
        raise ValueError(
            f"x={x} outside range [-{q_abs_max},{q_abs_max}]"
        )
    return ue_exp_golomb_bits(signed_to_code_num(x))


def q_residual_bits_signed_trunc_exp_golomb(
    q_residual: np.ndarray,
    q_abs_max: int,
) -> tuple[list[int], int]:
    bits_each = [
        signed_truncated_exp_golomb_bits(int(v), q_abs_max)
        for v in np.asarray(q_residual).reshape(-1)
    ]
    return bits_each, int(sum(bits_each))


# ============================================================
# Predictor
# ============================================================

def predict_from_history(
    decoded_hist: list[np.ndarray],
    pred_n: int,
    pred_degree: int,
) -> np.ndarray:
    if not decoded_hist:
        return np.zeros(6, dtype=np.float32)

    m = min(len(decoded_hist), int(pred_n))
    y = np.stack(decoded_hist[-m:], axis=0).astype(np.float32)

    if m == 1:
        return y[-1].copy()

    degree = min(int(pred_degree), m - 1)
    x = np.arange(m, dtype=np.float32)
    x_next = np.array([m], dtype=np.float32)
    a = np.vander(x, N=degree + 1, increasing=True)
    b = np.vander(x_next, N=degree + 1, increasing=True)
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return (b @ coef).reshape(6).astype(np.float32)


# ============================================================
# Inverse-depth block plane
# ============================================================

def fit_inverse_depth_plane_block(
    depth_block: np.ndarray,
    min_depth: float,
    min_valid_samples: int,
):
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
            [
                xx[valid],
                yy[valid],
                np.ones(n_valid, dtype=np.float64),
            ],
            axis=1,
        )
        try:
            candidate, _, rank, _ = np.linalg.lstsq(
                design,
                inv_z,
                rcond=None,
            )
            if rank >= 3 and np.isfinite(candidate).all():
                coeff = candidate.astype(np.float64)
            else:
                use_constant = True
        except np.linalg.LinAlgError:
            use_constant = True

    pred_inv = coeff[0] * xx + coeff[1] * yy + coeff[2]

    if (
        use_constant
        or not np.isfinite(pred_inv[valid]).all()
        or np.any(pred_inv[valid] <= 0)
    ):
        coeff = np.array([0.0, 0.0, constant_c], dtype=np.float64)
        pred_inv = np.full((h, w), constant_c, dtype=np.float64)
        use_constant = True

    recon[valid] = 1.0 / pred_inv[valid]
    return recon, valid, coeff, use_constant


def model_depth_with_inverse_planes(
    depth_linear: np.ndarray,
    block_size: int = 16,
    min_depth: float = 1e-8,
    min_valid_samples: int = 3,
):
    z = np.asarray(depth_linear, dtype=np.float64)
    h, w = z.shape
    recon = np.zeros_like(z, dtype=np.float64)
    valid_out = np.zeros_like(z, dtype=bool)
    blocks: list[dict[str, Any]] = []

    num_blocks = 0
    num_abc_blocks = 0
    num_constant_blocks = 0
    num_empty_blocks = 0

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            bh = min(block_size, h - by)
            bw = min(block_size, w - bx)
            block_recon, block_valid, coeff, constant = (
                fit_inverse_depth_plane_block(
                    z[by:by + bh, bx:bx + bw],
                    min_depth,
                    min_valid_samples,
                )
            )
            recon[by:by + bh, bx:bx + bw] = block_recon
            valid_out[by:by + bh, bx:bx + bw] = block_valid

            valid_count = int(np.count_nonzero(block_valid))
            num_blocks += 1
            if valid_count == 0:
                num_empty_blocks += 1
            elif constant:
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
                    "constant_fallback": bool(constant),
                }
            )

    metrics = calc_float_metrics(z, recon, valid_out)
    summary = {
        "model": "block_inverse_depth_plane",
        "equation": "1/z=a*(x-cx)+b*(y-cy)+c",
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "num_abc_blocks": int(num_abc_blocks),
        "num_constant_blocks": int(num_constant_blocks),
        "num_empty_blocks": int(num_empty_blocks),
        "abc_ratio": float(num_abc_blocks / max(num_blocks, 1)),
        "constant_ratio": float(
            num_constant_blocks / max(num_blocks, 1)
        ),
        "valid_ratio": float(np.mean(valid_out)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "max_abs_err": metrics["max_abs_err"],
    }
    return (
        recon.astype(np.float32),
        valid_out,
        summary,
        blocks,
    )


# ============================================================
# Projection / warp
# ============================================================

def make_projection_precompute_dual(
    w: int,
    h: int,
    intr_tar: dict[str, float],
    intr_ref: dict[str, float],
):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return {
        "w": int(w),
        "h": int(h),
        "fx_ref": float(intr_ref["fx"]),
        "fy_ref": float(intr_ref["fy"]),
        "cx_ref": float(intr_ref["cx"]),
        "cy_ref": float(intr_ref["cy"]),
        "z_sign": float(
            intr_tar.get(
                "z_sign",
                intr_ref.get("z_sign", 1.0),
            )
        ),
        "x_norm": (
            (x - float(intr_tar["cx"]))
            / float(intr_tar["fx"])
        ).astype(np.float32),
        "y_norm": (
            (y - float(intr_tar["cy"]))
            / float(intr_tar["fy"])
        ).astype(np.float32),
    }


def backward_map_fast_pixel_coord_dual(
    depth_linear: np.ndarray,
    precomp: dict[str, Any],
    rt: dict[str, list[float]],
):
    w = precomp["w"]
    h = precomp["h"]
    x_norm = precomp["x_norm"]
    y_norm = precomp["y_norm"]
    z_sign = float(precomp["z_sign"])
    z = depth_linear.astype(np.float32)

    rotation, _ = cv2.Rodrigues(
        np.asarray(rt["rvec"], dtype=np.float32).reshape(3, 1)
    )
    rotation = rotation.astype(np.float32)
    t = np.asarray(rt["tvec"], dtype=np.float32).reshape(3)

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

    xp = z * kx + float(t[0])
    yp = z * ky + float(t[1])
    zp = z * kz + float(t[2])

    denom = np.maximum(np.abs(zp), 1e-8)
    map_x = (
        float(precomp["fx_ref"]) * (xp / denom)
        + float(precomp["cx_ref"])
    )
    map_y = (
        float(precomp["fy_ref"]) * (yp / denom)
        + float(precomp["cy_ref"])
    )

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(z)
        & np.isfinite(zp)
        & (zp * z_sign > 0)
        & (map_x >= 0)
        & (map_x <= w - 1)
        & (map_y >= 0)
        & (map_y <= h - 1)
        & (z > 0)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y, valid


def remap_plane(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    bit_depth: int,
    border_value: int,
) -> np.ndarray:
    maxv = (1 << int(bit_depth)) - 1
    dst = cv2.remap(
        src.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    dst = np.clip(np.round(dst), 0, maxv)
    return dst.astype(
        np.uint8 if int(bit_depth) <= 8 else np.dtype("<u2")
    )


def downsample_luma_map_to_chroma_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map_x.shape
    if h % 2 or w % 2:
        raise ValueError("luma map must be even-sized")
    uv_h, uv_w = h // 2, w // 2
    mx = map_x.reshape(uv_h, 2, uv_w, 2)
    my = map_y.reshape(uv_h, 2, uv_w, 2)
    valid = (mx >= 0) & (my >= 0)
    count = np.sum(valid, axis=(1, 3)).astype(np.float32)
    sum_x = np.sum(np.where(valid, mx, 0), axis=(1, 3))
    sum_y = np.sum(np.where(valid, my, 0), axis=(1, 3))
    avg_x = np.full((uv_h, uv_w), -1, dtype=np.float32)
    avg_y = np.full((uv_h, uv_w), -1, dtype=np.float32)
    ok = count > 0
    avg_x[ok] = sum_x[ok] / count[ok]
    avg_y[ok] = sum_y[ok] / count[ok]
    map_x_uv = avg_x * 0.5
    map_y_uv = avg_y * 0.5
    map_x_uv[~ok] = -1
    map_y_uv[~ok] = -1
    return map_x_uv, map_y_uv


def backward_warp_yuv420_bilinear(
    ref_y: np.ndarray,
    ref_u: np.ndarray,
    ref_v: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    bit_depth: int,
):
    y = remap_plane(ref_y, map_x, map_y, bit_depth, 0)
    map_x_uv, map_y_uv = downsample_luma_map_to_chroma_map(
        map_x,
        map_y,
    )
    neutral = 128 if int(bit_depth) <= 8 else 512
    u = remap_plane(
        ref_u,
        map_x_uv,
        map_y_uv,
        bit_depth,
        neutral,
    )
    v = remap_plane(
        ref_v,
        map_x_uv,
        map_y_uv,
        bit_depth,
        neutral,
    )
    return y, u, v


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Low-delay current-to-previous pose quantization/warp simulation "
            "for merged multi-GOP camera JSONL."
        )
    )

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
    ap.add_argument("--out-plane-depth-yuv", default="")

    ap.add_argument("--pred-n", type=int, default=1)
    ap.add_argument("--pred-degree", type=int, default=0)
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
    )
    ap.add_argument(
        "--depth-plane-block-size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--depth-plane-min-depth",
        type=float,
        default=1e-8,
    )
    ap.add_argument(
        "--depth-plane-min-valid-samples",
        type=int,
        default=3,
    )
    ap.add_argument("--log-depth-plane-blocks", action="store_true")

    ap.add_argument(
        "--metric-valid-only",
        action="store_true",
        help="Compute warp metrics only on valid projection samples.",
    )
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.pred_n <= 0:
        raise ValueError("--pred-n must be positive")
    if args.pred_degree < 0:
        raise ValueError("--pred-degree must be non-negative")
    if args.r_step <= 0 or args.t_step_norm <= 0:
        raise ValueError("pose quantization steps must be positive")
    if args.intr_step <= 0:
        raise ValueError("--intr-step must be positive")
    if args.depth_plane_block_size <= 0:
        raise ValueError("--depth-plane-block-size must be positive")

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_jsonl = Path(args.param_jsonl)
    out_yuv = Path(args.out_yuv)
    out_q_jsonl = Path(args.out_q_jsonl)
    out_plane_depth_yuv = (
        Path(args.out_plane_depth_yuv)
        if args.out_plane_depth_yuv
        else None
    )

    for path in [seq_yuv, depth_yuv, param_jsonl]:
        if not path.is_file():
            raise FileNotFoundError(path)

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
        src_w=src_w,
        src_h=src_h,
        coded_w=coded_w,
        coded_h=coded_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )

    (
        header,
        records,
        records_by_gop,
        scale_by_gop,
    ) = load_merged_param_jsonl(param_jsonl)

    if int(header.get("width", src_w)) != src_w:
        raise ValueError("JSONL width does not match --width")
    if int(header.get("height", src_h)) != src_h:
        raise ValueError("JSONL height does not match --height")

    seq_count = count_frames(seq_yuv, src_w, src_h, bit_depth)
    depth_count = count_frames(depth_yuv, src_w, src_h, 10)

    max_seq_idx = max(int(r["frame_idx"]) for r in records)
    max_depth_idx = max(int(r["depth_frame_idx"]) for r in records)
    if max_seq_idx >= seq_count:
        raise RuntimeError(
            f"sequence YUV has {seq_count} frames but JSONL needs "
            f"frame_idx={max_seq_idx}"
        )
    if max_depth_idx >= depth_count:
        raise RuntimeError(
            f"depth YUV has {depth_count} frames but JSONL needs "
            f"depth_frame_idx={max_depth_idx}"
        )

    q_abs_max_ext = signed_q_abs_max(args.ext_bits)
    q_abs_max_intr = signed_q_abs_max(args.intr_delta_bits)

    header_intrinsic_bits_per_gop = 4 * 16
    depth_scale_bits_per_gop = int(args.depth_scale_bits)
    z_sign_bits_per_gop = 1
    header_bits_per_gop = (
        header_intrinsic_bits_per_gop
        + depth_scale_bits_per_gop
        + z_sign_bits_per_gop
    )
    header_bits = header_bits_per_gop * len(records_by_gop)

    total_ext_bits = 0
    total_ext_bits_r = 0
    total_ext_bits_t = 0
    total_ext_bits_each = np.zeros(6, dtype=np.int64)
    total_intr_delta_bits = 0
    total_intr_delta_bits_each = np.zeros(4, dtype=np.int64)
    total_coded_frames = 0
    total_clipped_frames = 0
    total_intr_clipped_frames = 0

    metric_rows: list[dict[str, float]] = []
    gop_summary: dict[int, dict[str, Any]] = {}

    ys_active, xs_active = active_slice(
        src_w,
        src_h,
        pad_left,
        pad_top,
    )

    print("=" * 72)
    print(f"camera records       : {len(records)}")
    print(f"GOP count            : {len(records_by_gop)}")
    print(f"sequence frames      : {seq_count}")
    print(f"depth frames         : {depth_count}")
    print(f"predictor            : pred_n={args.pred_n}, degree={args.pred_degree}")
    print(f"output order         : camera_record_idx")
    print("=" * 72)

    # Per-GOP closed-loop states.
    state_by_gop: dict[int, dict[str, Any]] = {}

    with open(out_q_jsonl, "w", encoding="utf-8") as fq:
        fq.write(
            json.dumps(
                {
                    "type": "header",
                    "source_param_jsonl": str(param_jsonl.resolve()),
                    "source_size": {"width": src_w, "height": src_h},
                    "coded_size": {"width": coded_w, "height": coded_h},
                    "padding": {
                        "left": pad_left,
                        "top": pad_top,
                        "right": pad_right,
                        "bottom": pad_bottom,
                    },
                    "input_pose_mode": "current_to_previous",
                    "output_frame_order": "camera_record_idx",
                    "overlap_behavior": (
                        "overlap POC is output once per GOP-local camera record"
                    ),
                    "predictor": {
                        "pred_n": int(args.pred_n),
                        "pred_degree": int(args.pred_degree),
                        "pred_n_1_degree_0": (
                            "previous reconstructed relative pose"
                        ),
                    },
                    "translation_normalization": (
                        "tvec / camera_GOP_depth_scale_real"
                    ),
                    "depth_dequantization": (
                        "depth_code * depth_source_GOP_depth_scale_real"
                    ),
                    "depth_scale_by_gop": {
                        str(k): float(v)
                        for k, v in sorted(scale_by_gop.items())
                    },
                    "depth_projection_mode": args.depth_projection_mode,
                    "depth_plane": {
                        "equation": "1/z=a*(x-cx)+b*(y-cy)+c",
                        "block_size": int(args.depth_plane_block_size),
                        "min_depth": float(args.depth_plane_min_depth),
                        "min_valid_samples": int(
                            args.depth_plane_min_valid_samples
                        ),
                    },
                    "extrinsic_bits": int(args.ext_bits),
                    "extrinsic_q_abs_max": int(q_abs_max_ext),
                    "r_step": float(args.r_step),
                    "t_step_norm": float(args.t_step_norm),
                    "intrinsic_delta_bits": int(args.intr_delta_bits),
                    "intrinsic_delta_q_abs_max": int(q_abs_max_intr),
                    "intrinsic_delta_step": float(args.intr_step),
                    "header_bits_per_gop": int(header_bits_per_gop),
                    "header_bits_total": int(header_bits),
                    "param6_order": [
                        "rx",
                        "ry",
                        "rz",
                        "tx_over_camera_gop_depth_scale",
                        "ty_over_camera_gop_depth_scale",
                        "tz_over_camera_gop_depth_scale",
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for output_idx, rec in enumerate(records):
            gop_idx = int(rec["gop_idx"])
            local_poc = int(rec["local_poc"])
            camera_scale = float(scale_by_gop[gop_idx])
            depth_owner_gop = int(rec["depth_source_gop_idx"])
            depth_scale = float(scale_by_gop[depth_owner_gop])

            cur_y, cur_u, cur_v = read_yuv420(
                seq_yuv,
                int(rec["frame_idx"]),
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

            depth_y, _, _ = read_yuv420(
                depth_yuv,
                int(rec["depth_frame_idx"]),
                src_w,
                src_h,
                10,
            )
            depth_linear = (
                depth_y.astype(np.float32) * depth_scale
            )
            depth_linear_pad = pad_2d_edge(
                depth_linear,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            ).astype(np.float32)

            if args.depth_projection_mode == "block_inv_plane":
                (
                    depth_for_projection,
                    depth_plane_valid,
                    plane_stats,
                    plane_blocks,
                ) = model_depth_with_inverse_planes(
                    depth_linear_pad,
                    block_size=int(args.depth_plane_block_size),
                    min_depth=float(args.depth_plane_min_depth),
                    min_valid_samples=int(
                        args.depth_plane_min_valid_samples
                    ),
                )
            else:
                depth_for_projection = depth_linear_pad
                depth_plane_valid = (
                    depth_for_projection > args.depth_plane_min_depth
                )
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
                    depth_scale,
                )

            if local_poc == 0:
                # Start a completely independent closed loop for this GOP.
                intr_gt_padded0 = (
                    make_padded_intrinsic_from_original(
                        rec["intrinsic"],
                        pad_left,
                        pad_top,
                    )
                )
                intr_q0, intr_dec0, intr_clip0 = (
                    quantize_intrinsic_16(
                        intr_gt_padded0,
                        coded_w,
                        coded_h,
                        f_max=args.intr_f_max,
                        c_min=args.intr_c_min,
                        c_max=args.intr_c_max,
                    )
                )

                p0_dec = np.zeros(6, dtype=np.float32)
                state_by_gop[gop_idx] = {
                    "decoded_hist": [p0_dec],
                    "decoded_intrinsics": [intr_dec0],
                    "records": [rec],
                }

                write_yuv420(
                    out_yuv,
                    cur_y_pad,
                    cur_u_pad,
                    cur_v_pad,
                )

                out_rec = {
                    "type": "frame",
                    "output_frame_idx": int(output_idx),
                    "camera_record_idx": int(
                        rec["camera_record_idx"]
                    ),
                    "gop_idx": gop_idx,
                    "gop_name": rec["gop_name"],
                    "local_poc": local_poc,
                    "poc": int(rec["poc"]),
                    "frame_idx": int(rec["frame_idx"]),
                    "depth_frame_idx": int(
                        rec["depth_frame_idx"]
                    ),
                    "depth_source_gop_idx": depth_owner_gop,
                    "camera_gop_depth_scale_real": camera_scale,
                    "depth_owner_scale_real": depth_scale,
                    "is_overlap": bool(rec.get("is_overlap", False)),
                    "is_anchor": True,
                    "q_residual": [0] * 6,
                    "q_residual_bits": [0] * 6,
                    "q_residual_total_bits": 0,
                    "param6_dec": p0_dec.astype(float).tolist(),
                    "intrinsic_q16_first": intr_q0,
                    "intrinsic_dec": intr_dec0,
                    "intrinsic_first_clipped": bool(intr_clip0),
                    "intrinsic_delta_gt": [0.0] * 4,
                    "q_intrinsic_delta": [0] * 4,
                    "q_intrinsic_delta_bits": [0] * 4,
                    "q_intrinsic_delta_total_bits": 0,
                    "depth_plane_stats": {
                        **plane_stats,
                        "psnr": json_safe_float(
                            plane_stats.get("psnr")
                        ),
                    },
                    "projection_valid_ratio": 1.0,
                    "mae_y_active": 0.0,
                    "mae_y_coded": 0.0,
                    "psnr_y_active": "inf",
                    "psnr_y_coded": "inf",
                }
                if args.log_depth_plane_blocks:
                    out_rec["depth_plane_blocks"] = plane_blocks
                fq.write(
                    json.dumps(out_rec, ensure_ascii=False) + "\n"
                )

                gop_summary[gop_idx] = {
                    "gop_idx": gop_idx,
                    "gop_name": rec["gop_name"],
                    "record_count": 1,
                    "coded_count": 0,
                    "ext_bits": 0,
                    "intr_bits": 0,
                    "header_bits": int(header_bits_per_gop),
                    "psnr_active_values": [],
                    "mae_active_values": [],
                }

                print(
                    f"[{output_idx:04d}/{len(records)-1:04d}] "
                    f"{rec['gop_name']} local=0 POC={rec['poc']}: "
                    "copy GOP anchor"
                )
                continue

            if gop_idx not in state_by_gop:
                raise RuntimeError(
                    f"GOP {gop_idx}: local_poc {local_poc} encountered "
                    "before local_poc 0"
                )

            state = state_by_gop[gop_idx]
            decoded_hist = state["decoded_hist"]
            decoded_intrinsics = state["decoded_intrinsics"]
            previous_record = state["records"][-1]

            if int(previous_record["local_poc"]) != local_poc - 1:
                raise RuntimeError(
                    f"GOP {gop_idx}: non-adjacent processing order"
                )

            intr_delta_gt = np.asarray(
                rec["intrinsic_delta"],
                dtype=np.float32,
            ).reshape(4)
            q_intr, d_intr, clip_intr = (
                quantize_intrinsic_delta_4(
                    intr_delta_gt,
                    step=args.intr_step,
                    bits=args.intr_delta_bits,
                )
            )
            q_intr_bits_each, q_intr_bits_total = (
                q_residual_bits_signed_trunc_exp_golomb(
                    q_intr,
                    q_abs_max_intr,
                )
            )

            intr_dec_cur = add_intrinsic_delta(
                decoded_intrinsics[-1],
                d_intr,
            )
            decoded_intrinsics.append(intr_dec_cur)
            intr_dec_ref = decoded_intrinsics[-2]
            intr_dec_tar = decoded_intrinsics[-1]

            p_gt = param6_from_frame(rec, camera_scale)
            p_pred = predict_from_history(
                decoded_hist,
                args.pred_n,
                args.pred_degree,
            )
            residual = p_gt - p_pred

            q_r, d_r, clip_r = quant_s(
                residual[:3],
                args.r_step,
                args.ext_bits,
            )
            q_t, d_t, clip_t = quant_s(
                residual[3:],
                args.t_step_norm,
                args.ext_bits,
            )
            q_residual = np.concatenate([q_r, q_t]).astype(
                np.int32
            )
            q_bits_each, q_bits_total = (
                q_residual_bits_signed_trunc_exp_golomb(
                    q_residual,
                    q_abs_max_ext,
                )
            )

            p_dec = p_pred.copy()
            p_dec[:3] += d_r
            p_dec[3:] += d_t
            decoded_hist.append(p_dec)
            rt_dec = rt_from_param6(p_dec, camera_scale)

            projection_precomp = make_projection_precompute_dual(
                coded_w,
                coded_h,
                intr_tar=intr_dec_tar,
                intr_ref=intr_dec_ref,
            )
            map_x, map_y, map_valid = (
                backward_map_fast_pixel_coord_dual(
                    depth_for_projection,
                    projection_precomp,
                    rt_dec,
                )
            )
            map_valid &= depth_plane_valid
            map_x[~map_valid] = -1
            map_y[~map_valid] = -1

            ref_y, ref_u, ref_v = read_yuv420(
                seq_yuv,
                int(previous_record["frame_idx"]),
                src_w,
                src_h,
                bit_depth,
            )
            ref_y_pad, ref_u_pad, ref_v_pad = pad_yuv420_edge(
                ref_y,
                ref_u,
                ref_v,
                coded_w,
                coded_h,
                pad_left,
                pad_top,
            )

            wy, wu, wv = backward_warp_yuv420_bilinear(
                ref_y_pad,
                ref_u_pad,
                ref_v_pad,
                map_x,
                map_y,
                bit_depth,
            )
            write_yuv420(out_yuv, wy, wu, wv)

            metric_mask = map_valid if args.metric_valid_only else None
            mae_y_coded = calc_mae(
                wy,
                cur_y_pad,
                metric_mask,
            )
            mae_y_active = calc_mae(
                wy[ys_active, xs_active],
                cur_y_pad[ys_active, xs_active],
                (
                    map_valid[ys_active, xs_active]
                    if args.metric_valid_only
                    else None
                ),
            )
            psnr_y_coded = calc_psnr(
                wy,
                cur_y_pad,
                bit_depth,
                metric_mask,
            )
            psnr_y_active = calc_psnr(
                wy[ys_active, xs_active],
                cur_y_pad[ys_active, xs_active],
                bit_depth,
                (
                    map_valid[ys_active, xs_active]
                    if args.metric_valid_only
                    else None
                ),
            )

            clipped = bool(np.any(clip_r) or np.any(clip_t))
            intrinsic_clipped = bool(np.any(clip_intr))

            total_ext_bits += int(q_bits_total)
            total_ext_bits_r += int(sum(q_bits_each[:3]))
            total_ext_bits_t += int(sum(q_bits_each[3:]))
            total_ext_bits_each += np.asarray(
                q_bits_each,
                dtype=np.int64,
            )
            total_intr_delta_bits += int(q_intr_bits_total)
            total_intr_delta_bits_each += np.asarray(
                q_intr_bits_each,
                dtype=np.int64,
            )
            total_coded_frames += 1
            total_clipped_frames += int(clipped)
            total_intr_clipped_frames += int(intrinsic_clipped)

            summary = gop_summary[gop_idx]
            summary["record_count"] += 1
            summary["coded_count"] += 1
            summary["ext_bits"] += int(q_bits_total)
            summary["intr_bits"] += int(q_intr_bits_total)
            if np.isfinite(psnr_y_active):
                summary["psnr_active_values"].append(
                    float(psnr_y_active)
                )
            if np.isfinite(mae_y_active):
                summary["mae_active_values"].append(
                    float(mae_y_active)
                )

            metric_rows.append(
                {
                    "psnr_active": psnr_y_active,
                    "psnr_coded": psnr_y_coded,
                    "mae_active": mae_y_active,
                    "mae_coded": mae_y_coded,
                }
            )

            out_rec = {
                "type": "frame",
                "output_frame_idx": int(output_idx),
                "camera_record_idx": int(rec["camera_record_idx"]),
                "gop_idx": gop_idx,
                "gop_name": rec["gop_name"],
                "local_poc": local_poc,
                "poc": int(rec["poc"]),
                "frame_idx": int(rec["frame_idx"]),
                "reference_frame_idx": int(
                    previous_record["frame_idx"]
                ),
                "reference_poc": int(previous_record["poc"]),
                "depth_frame_idx": int(rec["depth_frame_idx"]),
                "depth_source_gop_idx": depth_owner_gop,
                "camera_gop_depth_scale_real": camera_scale,
                "depth_owner_scale_real": depth_scale,
                "is_overlap": bool(rec.get("is_overlap", False)),
                "is_anchor": False,
                "q_residual": q_residual.astype(int).tolist(),
                "q_residual_bits": q_bits_each,
                "q_residual_total_bits": int(q_bits_total),
                "param6_gt": p_gt.astype(float).tolist(),
                "param6_pred": p_pred.astype(float).tolist(),
                "param6_residual": residual.astype(float).tolist(),
                "param6_dec": p_dec.astype(float).tolist(),
                "rt_dec": rt_dec,
                "extrinsic_clipped": clipped,
                "intrinsic_delta_gt": intr_delta_gt.astype(float).tolist(),
                "q_intrinsic_delta": q_intr.astype(int).tolist(),
                "intrinsic_delta_dec": d_intr.astype(float).tolist(),
                "q_intrinsic_delta_bits": q_intr_bits_each,
                "q_intrinsic_delta_total_bits": int(
                    q_intr_bits_total
                ),
                "intrinsic_ref_dec": intr_dec_ref,
                "intrinsic_tar_dec": intr_dec_tar,
                "intrinsic_clipped": intrinsic_clipped,
                "depth_plane_stats": {
                    **plane_stats,
                    "psnr": json_safe_float(
                        plane_stats.get("psnr")
                    ),
                },
                "projection_valid_ratio": float(np.mean(map_valid)),
                "mae_y_active": json_safe_float(mae_y_active),
                "mae_y_coded": json_safe_float(mae_y_coded),
                "psnr_y_active": json_safe_float(psnr_y_active),
                "psnr_y_coded": json_safe_float(psnr_y_coded),
            }
            if args.log_depth_plane_blocks:
                out_rec["depth_plane_blocks"] = plane_blocks
            fq.write(
                json.dumps(out_rec, ensure_ascii=False) + "\n"
            )

            state["records"].append(rec)

            print(
                f"[{output_idx:04d}/{len(records)-1:04d}] "
                f"{rec['gop_name']} local={local_poc} "
                f"POC={rec['poc']} refPOC={previous_record['poc']} "
                f"depthOwner={depth_owner_gop} "
                f"valid={np.mean(map_valid):.4f} "
                f"PSNR={psnr_y_active:.3f} "
                f"extBits={q_bits_total} "
                f"intrBits={q_intr_bits_total}"
            )

    total_bits = (
        int(header_bits)
        + int(total_ext_bits)
        + int(total_intr_delta_bits)
    )

    print("=" * 72)
    print("Bit summary")
    print("=" * 72)
    print(f"GOP header bits      : {header_bits}")
    print(f"coded pose records   : {total_coded_frames}")
    print(f"extrinsic bits       : {total_ext_bits}")
    print(f"intrinsic delta bits : {total_intr_delta_bits}")
    print(f"total bits           : {total_bits}")
    if total_coded_frames:
        print(
            f"avg ext bits/frame   : "
            f"{total_ext_bits / total_coded_frames:.3f}"
        )
        print(
            f"avg rot bits/frame   : "
            f"{total_ext_bits_r / total_coded_frames:.3f}"
        )
        print(
            f"avg trn bits/frame   : "
            f"{total_ext_bits_t / total_coded_frames:.3f}"
        )
        avg_each = (
            total_ext_bits_each.astype(np.float64)
            / total_coded_frames
        )
        print(
            "avg ext bits each    : "
            f"rx={avg_each[0]:.3f}, ry={avg_each[1]:.3f}, "
            f"rz={avg_each[2]:.3f}, tx={avg_each[3]:.3f}, "
            f"ty={avg_each[4]:.3f}, tz={avg_each[5]:.3f}"
        )
    print(f"extrinsic clipped    : {total_clipped_frames}")
    print(f"intrinsic clipped    : {total_intr_clipped_frames}")

    print("=" * 72)
    print("GOP summary")
    print("=" * 72)
    for gop_idx in sorted(gop_summary):
        summary = gop_summary[gop_idx]
        psnr_values = summary.pop("psnr_active_values")
        mae_values = summary.pop("mae_active_values")
        summary["avg_psnr_active"] = (
            float(np.mean(psnr_values))
            if psnr_values
            else None
        )
        summary["avg_mae_active"] = (
            float(np.mean(mae_values))
            if mae_values
            else None
        )
        summary["total_bits"] = (
            summary["header_bits"]
            + summary["ext_bits"]
            + summary["intr_bits"]
        )
        print(
            f"{summary['gop_name']}: "
            f"records={summary['record_count']}, "
            f"coded={summary['coded_count']}, "
            f"bits={summary['total_bits']}, "
            f"avgPSNR={summary['avg_psnr_active']}"
        )

    if metric_rows:
        valid_psnr = [
            row["psnr_active"]
            for row in metric_rows
            if np.isfinite(row["psnr_active"])
        ]
        valid_mae = [
            row["mae_active"]
            for row in metric_rows
            if np.isfinite(row["mae_active"])
        ]
        print("=" * 72)
        print("Metric summary")
        print("=" * 72)
        print(
            f"avg active PSNR      : "
            f"{np.mean(valid_psnr) if valid_psnr else None}"
        )
        print(
            f"avg active MAE       : "
            f"{np.mean(valid_mae) if valid_mae else None}"
        )

    print("=" * 72)
    print(f"warped YUV           : {out_yuv}")
    print(f"quantized JSONL      : {out_q_jsonl}")
    if out_plane_depth_yuv is not None:
        print(f"modeled depth YUV    : {out_plane_depth_yuv}")


if __name__ == "__main__":
    main()
