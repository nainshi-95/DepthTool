#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_relative_pose_dpcm_ra_warp.py

Evaluate the previous adjacent-relative-pose coding method using:

  1) merged camera JSONL produced with:
         --pose-mode current_to_previous
  2) original sequence YUV420p10le
  3) merged depth YUV420p10le

Input relative-pose convention:
    For GOP-local frame i > 0:

        X_{i-1} = R_step_i X_i + t_step_i

    JSONL stores:
        rvec[i] = Rodrigues(R_step_i)
        tvec[i] = t_step_i

    GOP local_poc 0 stores:
        rvec = [0,0,0]
        tvec = [0,0,0]

Closed-loop DPCM:
    The six relative-pose components are predicted from previously
    reconstructed relative-pose components.

    Predictor modes:
      previous:
          pred_i = reconstructed_signal_{i-1}

      linear:
          pred_i = 2*reconstructed_signal_{i-1}
                   - reconstructed_signal_{i-2}

      zero:
          pred_i = 0

    For the first coded relative pose, predictor is always zero.

    Residual:
        e_i = signal_i - pred_i

    Quantization:
        q_i = round(e_i / qstep)
        e_hat_i = q_i * qstep
        signal_hat_i = pred_i + e_hat_i

    Six signed q_i components are estimated with signed Exp-Golomb order-0.

Absolute GOP-local reconstruction:
    Let T_step_i map current camera i to previous camera i-1:

        T_step_i = T_{i-1} @ inverse(T_i)

    Therefore:

        T_i = inverse(T_step_i) @ T_{i-1}

    Starting from:
        T_0 = identity

    Expanded:
        R_i = R_step_i.T @ R_{i-1}
        t_i = R_step_i.T @ (t_{i-1} - t_step_i)

Warping:
    For target t and reference r:

        R_rel = R_r @ R_t.T
        t_rel = t_r - R_rel @ t_t

        X_ref = R_rel X_target + t_rel

    The reference Y image is backward-remapped into the target domain
    using the target depth frame.

RA pairs:
    Default pair source is dyadic. Explicit GOP-local pairs can be supplied.

PSNR:
    Baseline uses absolute poses reconstructed from the original unquantized
    current-to-previous JSONL signals.
    Quantized uses absolute poses reconstructed from closed-loop DPCM signals.

    Both are evaluated over:
        valid_common = valid_baseline & valid_quantized

Outputs:
    <prefix>_summary.json
    <prefix>_per_setting.csv
    <prefix>_per_gop.csv
    <prefix>_per_pair.csv
    <prefix>_per_frame_bits.csv

Example:
    python simulate_relative_pose_dpcm_ra_warp.py \
        --input-jsonl sequence_camParam_merged.jsonl \
        --sequence-yuv sequence_1920x1080_yuv420p10le.yuv \
        --depth-yuv sequence_depth_merged.yuv \
        --width 1920 \
        --height 1080 \
        --predictor previous \
        --rot-qsteps 1e-6,2e-6,5e-6,1e-5 \
        --trans-qsteps 1e-6,2e-6,5e-6,1e-5 \
        --paired-qsteps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


# ============================================================
# Geometry
# ============================================================

def R_from_rvec(rvec: Iterable[float]) -> np.ndarray:
    R, _ = cv2.Rodrigues(
        np.asarray(list(rvec), dtype=np.float64).reshape(3, 1)
    )
    return R.astype(np.float64)


def rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(
        np.asarray(R, dtype=np.float64).reshape(3, 3)
    )
    return rvec.reshape(3).astype(np.float64)


def K_from_record(rec: dict[str, Any]) -> np.ndarray:
    intr = rec.get("intrinsic")
    if not isinstance(intr, dict):
        raise KeyError("frame record has no intrinsic dictionary")

    return np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# Exp-Golomb
# ============================================================

def signed_to_code_num(value: int) -> int:
    value = int(value)
    if value > 0:
        return 2 * value - 1
    if value < 0:
        return -2 * value
    return 0


def ue_bits(code_num: int) -> int:
    code_num = int(code_num)
    if code_num < 0:
        raise ValueError("code_num must be non-negative")
    return 2 * ((code_num + 1).bit_length() - 1) + 1


def se_bits(value: int) -> int:
    return ue_bits(signed_to_code_num(value))


def vector_se_bits(values: np.ndarray) -> tuple[int, list[int]]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    component_bits = [se_bits(int(v)) for v in values]
    return int(sum(component_bits)), component_bits


# ============================================================
# YUV420p10le
# ============================================================

def frame_size_yuv420p10le(width: int, height: int) -> int:
    if width % 2 or height % 2:
        raise ValueError("YUV420 requires even width and height")
    samples = width * height + 2 * (width // 2) * (height // 2)
    return samples * 2


def count_yuv420p10le_frames(
    path: Path,
    width: int,
    height: int,
) -> int:
    frame_size = frame_size_yuv420p10le(width, height)
    size = path.stat().st_size
    trailing = size % frame_size
    if trailing:
        print(
            f"[WARN] trailing bytes ignored: {path}, trailing={trailing}",
            flush=True,
        )
    return size // frame_size


def read_yuv420p10le_y(
    path: Path,
    width: int,
    height: int,
    frame_idx: int,
) -> np.ndarray:
    frame_idx = int(frame_idx)
    if frame_idx < 0:
        raise ValueError(f"negative frame index: {frame_idx}")

    frame_size = frame_size_yuv420p10le(width, height)
    y_count = width * height

    with path.open("rb") as f:
        f.seek(frame_idx * frame_size)
        y = np.fromfile(f, dtype="<u2", count=y_count)

    if y.size != y_count:
        raise EOFError(
            f"cannot read frame {frame_idx} from {path}: "
            f"{y.size}/{y_count} Y samples"
        )

    return y.reshape(height, width)


class LumaReader:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.cache: dict[int, np.ndarray] = {}

    def read(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in self.cache:
            self.cache[frame_idx] = read_yuv420p10le_y(
                self.path,
                self.width,
                self.height,
                frame_idx,
            ).astype(np.float32)
        return self.cache[frame_idx]


class DepthReader:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        scale_by_gop: dict[int, float],
    ) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.scale_by_gop = scale_by_gop
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def read(
        self,
        depth_frame_idx: int,
        owner_gop_idx: int,
    ) -> np.ndarray:
        key = (int(depth_frame_idx), int(owner_gop_idx))

        if key in self.cache:
            return self.cache[key]

        if owner_gop_idx not in self.scale_by_gop:
            raise KeyError(
                f"no depth_scale_real for depth owner GOP {owner_gop_idx}"
            )

        code = read_yuv420p10le_y(
            self.path,
            self.width,
            self.height,
            depth_frame_idx,
        ).astype(np.float32)

        depth = code * float(self.scale_by_gop[owner_gop_idx])
        self.cache[key] = depth
        return depth


# ============================================================
# JSONL loading
# ============================================================

def load_camera_jsonl(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                continue

            if obj.get("type") == "header":
                if not header:
                    header = obj
                continue

            if obj.get("type") != "frame":
                continue

            required = ["gop_idx", "rvec", "tvec", "intrinsic"]
            missing = [key for key in required if key not in obj]
            if missing:
                raise KeyError(
                    f"{path}:{line_no}: missing keys {missing}"
                )

            rvec = np.asarray(obj["rvec"], dtype=np.float64).reshape(-1)
            tvec = np.asarray(obj["tvec"], dtype=np.float64).reshape(-1)

            if rvec.size != 3 or tvec.size != 3:
                raise ValueError(
                    f"{path}:{line_no}: rvec/tvec must contain 3 values"
                )

            rec = dict(obj)
            rec["_line_no"] = int(line_no)
            rec["_rvec_np"] = rvec
            rec["_tvec_np"] = tvec
            rec["_K_np"] = K_from_record(rec)

            rec["gop_idx"] = int(rec["gop_idx"])
            rec["gop_name"] = str(
                rec.get("gop_name", f"gop{rec['gop_idx']}")
            )
            rec["local_poc"] = int(
                rec.get("local_poc", rec.get("poc", 0))
            )
            rec["poc"] = int(rec.get("poc", rec["local_poc"]))
            rec["frame_idx"] = int(
                rec.get("frame_idx", rec["poc"])
            )
            rec["depth_frame_idx"] = int(
                rec.get("depth_frame_idx", rec["poc"])
            )
            rec["depth_source_gop_idx"] = int(
                rec.get("depth_source_gop_idx", rec["gop_idx"])
            )

            frames.append(rec)

    if not frames:
        raise RuntimeError(f"No frame records found in {path}")

    pose_mode = header.get("pose_mode")
    if pose_mode is not None and pose_mode != "current_to_previous":
        raise ValueError(
            f"JSONL pose_mode is '{pose_mode}', but this simulator requires "
            "'current_to_previous'"
        )

    return header, frames


def group_frames_by_gop(
    frames: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}

    for rec in frames:
        groups.setdefault(int(rec["gop_idx"]), []).append(rec)

    for gop_idx, recs in groups.items():
        recs.sort(
            key=lambda r: (
                int(r["local_poc"]),
                int(r["poc"]),
                int(r["_line_no"]),
            )
        )

        local_pocs = [int(r["local_poc"]) for r in recs]

        if len(local_pocs) != len(set(local_pocs)):
            raise ValueError(
                f"GOP {gop_idx}: duplicate local_poc values"
            )

        if not local_pocs or local_pocs[0] != 0:
            raise ValueError(
                f"GOP {gop_idx}: first local_poc must be 0"
            )

        if local_pocs != list(range(len(local_pocs))):
            raise ValueError(
                f"GOP {gop_idx}: local_poc must be contiguous 0..N-1, "
                f"got {local_pocs}"
            )

        if not np.allclose(recs[0]["_rvec_np"], 0.0, atol=1e-7):
            raise ValueError(
                f"GOP {gop_idx}: local_poc 0 rvec is not zero"
            )

        if not np.allclose(recs[0]["_tvec_np"], 0.0, atol=1e-7):
            raise ValueError(
                f"GOP {gop_idx}: local_poc 0 tvec is not zero"
            )

    return groups


def build_depth_scale_by_gop(
    header: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[int, float]:
    scales: dict[int, float] = {}

    gops = header.get("gops")
    if isinstance(gops, list):
        for item in gops:
            if (
                isinstance(item, dict)
                and "gop_idx" in item
                and "depth_scale_real" in item
            ):
                scales[int(item["gop_idx"])] = float(
                    item["depth_scale_real"]
                )

    for rec in frames:
        gop_idx = int(rec["gop_idx"])
        if gop_idx not in scales and "depth_scale_real" in rec:
            scales[gop_idx] = float(rec["depth_scale_real"])

    if not scales:
        raise RuntimeError(
            "No depth_scale_real found in JSONL header or frame records"
        )

    for gop_idx, scale in scales.items():
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"invalid depth scale for GOP {gop_idx}: {scale}"
            )

    return scales


# ============================================================
# Relative signal reconstruction
# ============================================================

def reconstruct_absolute_from_relative(
    relative_rvecs: np.ndarray,
    relative_tvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct GOP-local absolute W2C poses.

    Input at index i:
        X_{i-1} = R_step_i X_i + t_step_i

    Output:
        X_i = R_abs_i X_0 + t_abs_i
    """
    relative_rvecs = np.asarray(relative_rvecs, dtype=np.float64)
    relative_tvecs = np.asarray(relative_tvecs, dtype=np.float64)

    if relative_rvecs.shape != relative_tvecs.shape:
        raise ValueError("relative rvec/tvec shape mismatch")
    if relative_rvecs.ndim != 2 or relative_rvecs.shape[1] != 3:
        raise ValueError("relative poses must have shape [N,3]")

    n = relative_rvecs.shape[0]

    R_abs = np.zeros((n, 3, 3), dtype=np.float64)
    t_abs = np.zeros((n, 3), dtype=np.float64)

    R_abs[0] = np.eye(3, dtype=np.float64)
    t_abs[0] = 0.0

    for i in range(1, n):
        R_step = R_from_rvec(relative_rvecs[i])
        t_step = relative_tvecs[i]

        R_abs[i] = R_step.T @ R_abs[i - 1]
        t_abs[i] = R_step.T @ (
            t_abs[i - 1] - t_step
        )

    return R_abs, t_abs


@dataclass
class CodingOptions:
    rot_qstep: float
    trans_qstep: float
    predictor: str
    first_frame_bits: int
    per_gop_overhead_bits: int
    per_frame_overhead_bits: int


def quantize_to_index(
    value: np.ndarray,
    qstep: float,
) -> np.ndarray:
    if qstep <= 0:
        raise ValueError("qstep must be positive")

    return np.rint(
        np.asarray(value, dtype=np.float64) / float(qstep)
    ).astype(np.int64)


def simulate_relative_dpcm(
    gop_idx: int,
    records: list[dict[str, Any]],
    options: CodingOptions,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """
    Quantize and reconstruct JSONL current-to-previous relative signals.

    Returns:
        GOP bit summary
        per-frame rows
        reconstructed absolute R [N,3,3]
        reconstructed absolute t [N,3]
    """
    n = len(records)

    source_r = np.stack(
        [np.asarray(r["_rvec_np"], dtype=np.float64) for r in records],
        axis=0,
    )
    source_t = np.stack(
        [np.asarray(r["_tvec_np"], dtype=np.float64) for r in records],
        axis=0,
    )

    reconstructed_r = np.zeros_like(source_r)
    reconstructed_t = np.zeros_like(source_t)

    previous_r = np.zeros(3, dtype=np.float64)
    previous_t = np.zeros(3, dtype=np.float64)
    previous2_r = np.zeros(3, dtype=np.float64)
    previous2_t = np.zeros(3, dtype=np.float64)

    total_rotation_bits = 0
    total_translation_bits = 0
    total_overhead_bits = int(options.per_gop_overhead_bits)

    total_rot_abs_qindex = 0
    total_trans_abs_qindex = 0
    total_rot_zero = 0
    total_trans_zero = 0
    total_components = 0

    frame_rows: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        if i == 0:
            r_pred = np.zeros(3, dtype=np.float64)
            t_pred = np.zeros(3, dtype=np.float64)
            r_res = np.zeros(3, dtype=np.float64)
            t_res = np.zeros(3, dtype=np.float64)
            q_r = np.zeros(3, dtype=np.int64)
            q_t = np.zeros(3, dtype=np.int64)
            r_component_bits = [0, 0, 0]
            t_component_bits = [0, 0, 0]
            rotation_bits = 0
            translation_bits = 0
            overhead_bits = int(options.first_frame_bits)
            r_hat = np.zeros(3, dtype=np.float64)
            t_hat = np.zeros(3, dtype=np.float64)

        else:
            if i == 1 or options.predictor == "zero":
                r_pred = np.zeros(3, dtype=np.float64)
                t_pred = np.zeros(3, dtype=np.float64)

            elif options.predictor == "previous":
                r_pred = previous_r.copy()
                t_pred = previous_t.copy()

            elif options.predictor == "linear":
                if i == 2:
                    r_pred = previous_r.copy()
                    t_pred = previous_t.copy()
                else:
                    r_pred = 2.0 * previous_r - previous2_r
                    t_pred = 2.0 * previous_t - previous2_t

            else:
                raise ValueError(options.predictor)

            r_res = source_r[i] - r_pred
            t_res = source_t[i] - t_pred

            q_r = quantize_to_index(
                r_res,
                options.rot_qstep,
            )
            q_t = quantize_to_index(
                t_res,
                options.trans_qstep,
            )

            r_hat = (
                r_pred
                + q_r.astype(np.float64) * float(options.rot_qstep)
            )
            t_hat = (
                t_pred
                + q_t.astype(np.float64) * float(options.trans_qstep)
            )

            rotation_bits, r_component_bits = vector_se_bits(q_r)
            translation_bits, t_component_bits = vector_se_bits(q_t)
            overhead_bits = int(options.per_frame_overhead_bits)

        reconstructed_r[i] = r_hat
        reconstructed_t[i] = t_hat

        total_rotation_bits += int(rotation_bits)
        total_translation_bits += int(translation_bits)
        total_overhead_bits += int(overhead_bits)

        if i > 0:
            total_rot_abs_qindex += int(np.sum(np.abs(q_r)))
            total_trans_abs_qindex += int(np.sum(np.abs(q_t)))
            total_rot_zero += int(np.count_nonzero(q_r == 0))
            total_trans_zero += int(np.count_nonzero(q_t == 0))
            total_components += 3

        R_source = R_from_rvec(source_r[i])
        R_hat = R_from_rvec(r_hat)
        R_error = R_source @ R_hat.T
        relative_rot_error = float(
            np.linalg.norm(rvec_from_R(R_error))
        )
        relative_trans_error = float(
            np.linalg.norm(source_t[i] - t_hat)
        )

        frame_rows.append(
            {
                "rot_qstep": float(options.rot_qstep),
                "trans_qstep": float(options.trans_qstep),
                "predictor": str(options.predictor),
                "gop_idx": int(gop_idx),
                "gop_name": str(rec["gop_name"]),
                "local_poc": int(rec["local_poc"]),
                "poc": int(rec["poc"]),
                "is_anchor": i == 0,
                "source_r_x": float(source_r[i, 0]),
                "source_r_y": float(source_r[i, 1]),
                "source_r_z": float(source_r[i, 2]),
                "source_t_x": float(source_t[i, 0]),
                "source_t_y": float(source_t[i, 1]),
                "source_t_z": float(source_t[i, 2]),
                "pred_r_x": float(r_pred[0]),
                "pred_r_y": float(r_pred[1]),
                "pred_r_z": float(r_pred[2]),
                "pred_t_x": float(t_pred[0]),
                "pred_t_y": float(t_pred[1]),
                "pred_t_z": float(t_pred[2]),
                "res_r_x": float(r_res[0]),
                "res_r_y": float(r_res[1]),
                "res_r_z": float(r_res[2]),
                "res_t_x": float(t_res[0]),
                "res_t_y": float(t_res[1]),
                "res_t_z": float(t_res[2]),
                "q_r_x": int(q_r[0]),
                "q_r_y": int(q_r[1]),
                "q_r_z": int(q_r[2]),
                "q_t_x": int(q_t[0]),
                "q_t_y": int(q_t[1]),
                "q_t_z": int(q_t[2]),
                "bits_r_x": int(r_component_bits[0]),
                "bits_r_y": int(r_component_bits[1]),
                "bits_r_z": int(r_component_bits[2]),
                "bits_t_x": int(t_component_bits[0]),
                "bits_t_y": int(t_component_bits[1]),
                "bits_t_z": int(t_component_bits[2]),
                "rotation_bits": int(rotation_bits),
                "translation_bits": int(translation_bits),
                "overhead_bits": int(overhead_bits),
                "total_bits": int(
                    rotation_bits + translation_bits + overhead_bits
                ),
                "reconstructed_r_x": float(r_hat[0]),
                "reconstructed_r_y": float(r_hat[1]),
                "reconstructed_r_z": float(r_hat[2]),
                "reconstructed_t_x": float(t_hat[0]),
                "reconstructed_t_y": float(t_hat[1]),
                "reconstructed_t_z": float(t_hat[2]),
                "relative_rot_error_rad": relative_rot_error,
                "relative_rot_error_deg": float(
                    np.degrees(relative_rot_error)
                ),
                "relative_trans_error": relative_trans_error,
            }
        )

        previous2_r = previous_r.copy()
        previous2_t = previous_t.copy()
        previous_r = r_hat.copy()
        previous_t = t_hat.copy()

    R_abs_recon, t_abs_recon = reconstruct_absolute_from_relative(
        reconstructed_r,
        reconstructed_t,
    )

    total_bits = (
        total_rotation_bits
        + total_translation_bits
        + total_overhead_bits
    )

    summary = {
        "rot_qstep": float(options.rot_qstep),
        "trans_qstep": float(options.trans_qstep),
        "predictor": str(options.predictor),
        "gop_idx": int(gop_idx),
        "gop_name": str(records[0]["gop_name"]),
        "frame_count": int(n),
        "coded_frame_count": int(max(0, n - 1)),
        "rotation_bits": int(total_rotation_bits),
        "translation_bits": int(total_translation_bits),
        "overhead_bits": int(total_overhead_bits),
        "total_bits": int(total_bits),
        "bits_per_camera_record": float(total_bits / n),
        "bits_per_coded_frame": (
            float(total_bits / (n - 1))
            if n > 1
            else 0.0
        ),
        "rot_mean_abs_qindex": (
            float(total_rot_abs_qindex / total_components)
            if total_components
            else 0.0
        ),
        "trans_mean_abs_qindex": (
            float(total_trans_abs_qindex / total_components)
            if total_components
            else 0.0
        ),
        "rot_zero_ratio": (
            float(total_rot_zero / total_components)
            if total_components
            else 1.0
        ),
        "trans_zero_ratio": (
            float(total_trans_zero / total_components)
            if total_components
            else 1.0
        ),
    }

    return summary, frame_rows, R_abs_recon, t_abs_recon


# ============================================================
# RA pairs
# ============================================================

def parse_explicit_pairs(
    text: str,
) -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []

    if not text.strip():
        return pairs

    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue

        token = token.replace("->", ":")
        parts = token.split(":")

        if len(parts) != 2:
            raise ValueError(
                f"bad pair '{token}', use target:ref"
            )

        pairs.append(
            (
                int(parts[0]),
                int(parts[1]),
                "explicit",
            )
        )

    return pairs


def generate_adjacent_pairs(
    local_pocs: list[int],
    bidirectional: bool,
) -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []

    for i in range(1, len(local_pocs)):
        current = local_pocs[i]
        previous = local_pocs[i - 1]

        pairs.append(
            (current, previous, "adjacent")
        )

        if bidirectional:
            pairs.append(
                (previous, current, "adjacent_reverse")
            )

    return pairs


def generate_dyadic_pairs(
    local_pocs: list[int],
    bidirectional: bool,
) -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(target: int, ref: int, kind: str) -> None:
        key = (int(target), int(ref))
        if target == ref or key in seen:
            return
        seen.add(key)
        pairs.append((int(target), int(ref), kind))

    def recurse(left: int, right: int, level: int) -> None:
        if right <= left + 1:
            return

        middle = (left + right) // 2

        left_poc = local_pocs[left]
        middle_poc = local_pocs[middle]
        right_poc = local_pocs[right]

        add(
            middle_poc,
            left_poc,
            f"dyadic_L{level}_left",
        )
        add(
            middle_poc,
            right_poc,
            f"dyadic_L{level}_right",
        )

        if bidirectional:
            add(
                left_poc,
                middle_poc,
                f"dyadic_L{level}_left_reverse",
            )
            add(
                right_poc,
                middle_poc,
                f"dyadic_L{level}_right_reverse",
            )

        recurse(left, middle, level + 1)
        recurse(middle, right, level + 1)

    if len(local_pocs) >= 2:
        add(
            local_pocs[-1],
            local_pocs[0],
            "dyadic_endpoint",
        )
        if bidirectional:
            add(
                local_pocs[0],
                local_pocs[-1],
                "dyadic_endpoint_reverse",
            )

    recurse(0, len(local_pocs) - 1, 0)

    return pairs


def build_pairs_for_gop(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[int, int, str]]:
    local_pocs = [int(r["local_poc"]) for r in records]

    if args.pairs:
        pairs = parse_explicit_pairs(args.pairs)
    elif args.pair_source == "adjacent":
        pairs = generate_adjacent_pairs(
            local_pocs,
            bidirectional=not args.no_bidirectional_pairs,
        )
    elif args.pair_source == "dyadic":
        pairs = generate_dyadic_pairs(
            local_pocs,
            bidirectional=not args.no_bidirectional_pairs,
        )
    else:
        raise ValueError(args.pair_source)

    available = set(local_pocs)
    checked: list[tuple[int, int, str]] = []

    for target, ref, kind in pairs:
        if target not in available or ref not in available:
            if args.skip_missing_pairs:
                continue
            raise ValueError(
                f"GOP {records[0]['gop_idx']}: pair {target}->{ref} "
                f"is outside local_poc set {sorted(available)}"
            )
        checked.append((target, ref, kind))

    if not checked:
        raise RuntimeError(
            f"GOP {records[0]['gop_idx']}: no valid evaluation pairs"
        )

    return checked


# ============================================================
# Warping
# ============================================================

def camera_map(
    width: int,
    height: int,
    K_target: np.ndarray,
    K_ref: np.ndarray,
    R_target: np.ndarray,
    t_target: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    depth_target: np.ndarray,
    z_sign: float,
    z_min: float,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_rel = R_ref @ R_target.T
    t_rel = t_ref - R_rel @ t_target

    fx_t = float(K_target[0, 0])
    fy_t = float(K_target[1, 1])
    cx_t = float(K_target[0, 2])
    cy_t = float(K_target[1, 2])

    fx_r = float(K_ref[0, 0])
    fy_r = float(K_ref[1, 1])
    cx_r = float(K_ref[0, 2])
    cy_r = float(K_ref[1, 2])

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    valid_all = np.zeros((height, width), dtype=bool)

    xs_full = np.arange(width, dtype=np.float64)

    for y0 in range(0, height, row_batch):
        y1 = min(height, y0 + row_batch)

        ys = np.arange(y0, y1, dtype=np.float64)
        xs, yy = np.meshgrid(xs_full, ys)

        ray_x = (xs - cx_t) / fx_t
        ray_y = (yy - cy_t) / fy_t

        rays = np.stack(
            [
                ray_x.reshape(-1),
                ray_y.reshape(-1),
                np.full(
                    (y1 - y0) * width,
                    float(z_sign),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        dep = (
            depth_target[y0:y1]
            .reshape(-1)
            .astype(np.float64)
        )

        X_target = dep[:, None] * rays
        X_ref = X_target @ R_rel.T + t_rel[None, :]

        z = X_ref[:, 2]
        z_safe = np.where(
            np.abs(z) > 1e-12,
            z,
            np.where(z >= 0, 1e-12, -1e-12),
        )

        map_x_values = fx_r * (X_ref[:, 0] / z_safe) + cx_r
        map_y_values = fy_r * (X_ref[:, 1] / z_safe) + cy_r

        valid = (
            np.isfinite(map_x_values)
            & np.isfinite(map_y_values)
            & np.isfinite(dep)
            & (dep > 0)
            & (z * float(z_sign) > float(z_min))
            & (map_x_values >= 0)
            & (map_x_values <= width - 1)
            & (map_y_values >= 0)
            & (map_y_values <= height - 1)
        )

        map_x[y0:y1] = map_x_values.reshape(
            y1 - y0,
            width,
        ).astype(np.float32)

        map_y[y0:y1] = map_y_values.reshape(
            y1 - y0,
            width,
        ).astype(np.float32)

        valid_all[y0:y1] = valid.reshape(
            y1 - y0,
            width,
        )

    map_x[~valid_all] = -1.0
    map_y[~valid_all] = -1.0

    return map_x, map_y, valid_all


def warp_reference_to_target(
    reference_y: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        reference_y.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def masked_error_stats(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    max_value: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(mask))

    if count == 0:
        return {
            "count": 0,
            "sse": 0.0,
            "mae": None,
            "mse": None,
            "psnr": None,
        }

    difference = (
        target.astype(np.float64)[mask]
        - prediction.astype(np.float64)[mask]
    )

    sse = float(np.sum(difference * difference))
    mse = float(sse / count)
    mae = float(np.mean(np.abs(difference)))

    psnr = (
        999.0
        if mse <= 1e-30
        else float(
            10.0
            * math.log10(
                float(max_value) ** 2 / mse
            )
        )
    )

    return {
        "count": count,
        "sse": sse,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }


def evaluate_pair(
    target_record: dict[str, Any],
    reference_record: dict[str, Any],
    baseline_target_pose: tuple[np.ndarray, np.ndarray],
    baseline_reference_pose: tuple[np.ndarray, np.ndarray],
    quantized_target_pose: tuple[np.ndarray, np.ndarray],
    quantized_reference_pose: tuple[np.ndarray, np.ndarray],
    luma_reader: LumaReader,
    depth_reader: DepthReader,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_y = luma_reader.read(
        int(target_record["frame_idx"])
    )
    reference_y = luma_reader.read(
        int(reference_record["frame_idx"])
    )

    target_depth = depth_reader.read(
        int(target_record["depth_frame_idx"]),
        int(target_record["depth_source_gop_idx"]),
    )

    K_target = np.asarray(
        target_record["_K_np"],
        dtype=np.float64,
    )
    K_reference = np.asarray(
        reference_record["_K_np"],
        dtype=np.float64,
    )

    R_target_base, t_target_base = baseline_target_pose
    R_reference_base, t_reference_base = baseline_reference_pose

    R_target_quant, t_target_quant = quantized_target_pose
    R_reference_quant, t_reference_quant = quantized_reference_pose

    map_x_base, map_y_base, valid_base = camera_map(
        width=args.width,
        height=args.height,
        K_target=K_target,
        K_ref=K_reference,
        R_target=R_target_base,
        t_target=t_target_base,
        R_ref=R_reference_base,
        t_ref=t_reference_base,
        depth_target=target_depth,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.row_batch,
    )

    map_x_quant, map_y_quant, valid_quant = camera_map(
        width=args.width,
        height=args.height,
        K_target=K_target,
        K_ref=K_reference,
        R_target=R_target_quant,
        t_target=t_target_quant,
        R_ref=R_reference_quant,
        t_ref=t_reference_quant,
        depth_target=target_depth,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.row_batch,
    )

    prediction_base = warp_reference_to_target(
        reference_y,
        map_x_base,
        map_y_base,
    )
    prediction_quant = warp_reference_to_target(
        reference_y,
        map_x_quant,
        map_y_quant,
    )

    valid_common = valid_base & valid_quant

    if args.valid_erode > 0:
        kernel_size = 2 * int(args.valid_erode) + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        valid_common = (
            cv2.erode(
                valid_common.astype(np.uint8),
                kernel,
                iterations=1,
            )
            > 0
        )

    max_value = float((1 << args.bitdepth) - 1)

    baseline_common = masked_error_stats(
        target_y,
        prediction_base,
        valid_common,
        max_value,
    )
    quantized_common = masked_error_stats(
        target_y,
        prediction_quant,
        valid_common,
        max_value,
    )

    baseline_native = masked_error_stats(
        target_y,
        prediction_base,
        valid_base,
        max_value,
    )
    quantized_native = masked_error_stats(
        target_y,
        prediction_quant,
        valid_quant,
        max_value,
    )

    psnr_drop = None
    if (
        baseline_common["psnr"] is not None
        and quantized_common["psnr"] is not None
    ):
        psnr_drop = float(
            baseline_common["psnr"]
            - quantized_common["psnr"]
        )

    return {
        "common_valid_count": int(baseline_common["count"]),
        "common_valid_ratio": float(
            baseline_common["count"]
            / (args.width * args.height)
        ),
        "baseline_common_sse": float(baseline_common["sse"]),
        "baseline_common_mse": baseline_common["mse"],
        "baseline_common_psnr": baseline_common["psnr"],
        "quantized_common_sse": float(quantized_common["sse"]),
        "quantized_common_mse": quantized_common["mse"],
        "quantized_common_psnr": quantized_common["psnr"],
        "psnr_drop_common": psnr_drop,
        "baseline_native_valid_ratio": float(np.mean(valid_base)),
        "baseline_native_psnr": baseline_native["psnr"],
        "quantized_native_valid_ratio": float(np.mean(valid_quant)),
        "quantized_native_psnr": quantized_native["psnr"],
    }


# ============================================================
# Qsteps / output
# ============================================================

def parse_float_list(text: str) -> list[float]:
    values: list[float] = []

    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue

        value = float(token)

        if value <= 0:
            raise ValueError(
                f"qstep must be positive: {value}"
            )

        values.append(value)

    if not values:
        raise ValueError("empty qstep list")

    return values


def make_qstep_pairs(
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    rotation_qsteps = (
        parse_float_list(args.rot_qsteps)
        if args.rot_qsteps
        else [float(args.rot_qstep)]
    )

    translation_qsteps = (
        parse_float_list(args.trans_qsteps)
        if args.trans_qsteps
        else [float(args.trans_qstep)]
    )

    if args.paired_qsteps:
        if len(rotation_qsteps) != len(translation_qsteps):
            raise ValueError(
                "--paired-qsteps requires equal numbers of qsteps"
            )

        return list(
            zip(rotation_qsteps, translation_qsteps)
        )

    return [
        (rotation_qstep, translation_qstep)
        for rotation_qstep in rotation_qsteps
        for translation_qstep in translation_qsteps
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Simulate closed-loop DPCM of adjacent current-to-previous "
            "camera poses and evaluate RA warp PSNR."
        )
    )

    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument(
        "--sequence-yuv",
        required=True,
        help="Original YUV420p10le sequence",
    )
    ap.add_argument(
        "--depth-yuv",
        required=True,
        help="Merged depth YUV420p10le",
    )
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--output-prefix", default="")

    ap.add_argument(
        "--predictor",
        choices=["previous", "linear", "zero"],
        default="previous",
    )

    ap.add_argument("--rot-qstep", type=float, default=1e-5)
    ap.add_argument("--trans-qstep", type=float, default=1e-5)
    ap.add_argument("--rot-qsteps", default="")
    ap.add_argument("--trans-qsteps", default="")
    ap.add_argument("--paired-qsteps", action="store_true")

    ap.add_argument(
        "--pair-source",
        choices=["dyadic", "adjacent"],
        default="dyadic",
    )
    ap.add_argument(
        "--pairs",
        default="",
        help="Explicit GOP-local target:ref pairs, e.g. 16:0,16:32",
    )
    ap.add_argument(
        "--no-bidirectional-pairs",
        action="store_true",
    )
    ap.add_argument(
        "--skip-missing-pairs",
        action="store_true",
    )

    ap.add_argument(
        "--bitdepth",
        type=int,
        choices=[10],
        default=10,
    )
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--z-min", type=float, default=1e-4)
    ap.add_argument("--row-batch", type=int, default=64)
    ap.add_argument("--valid-erode", type=int, default=1)

    ap.add_argument("--first-frame-bits", type=int, default=0)
    ap.add_argument("--per-gop-overhead-bits", type=int, default=0)
    ap.add_argument("--per-frame-overhead-bits", type=int, default=0)

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.row_batch <= 0:
        raise ValueError("--row-batch must be positive")
    if args.valid_erode < 0:
        raise ValueError("--valid-erode must be non-negative")
    if args.rot_qstep <= 0 or args.trans_qstep <= 0:
        raise ValueError("qsteps must be positive")

    for name in [
        "first_frame_bits",
        "per_gop_overhead_bits",
        "per_frame_overhead_bits",
    ]:
        if int(getattr(args, name)) < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be non-negative"
            )

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    sequence_yuv = Path(args.sequence_yuv).expanduser().resolve()
    depth_yuv = Path(args.depth_yuv).expanduser().resolve()

    for path in [input_jsonl, sequence_yuv, depth_yuv]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.output_prefix:
        output_prefix = Path(
            args.output_prefix
        ).expanduser().resolve()
    else:
        stem = input_jsonl.with_suffix("")
        output_prefix = stem.parent / (
            stem.name
            + f"_relative_dpcm_{args.predictor}_ra_warp"
        )

    header, frames = load_camera_jsonl(input_jsonl)
    groups = group_frames_by_gop(frames)
    depth_scale_by_gop = build_depth_scale_by_gop(
        header,
        frames,
    )

    sequence_frame_count = count_yuv420p10le_frames(
        sequence_yuv,
        args.width,
        args.height,
    )
    depth_frame_count = count_yuv420p10le_frames(
        depth_yuv,
        args.width,
        args.height,
    )

    maximum_sequence_index = max(
        int(rec["frame_idx"])
        for rec in frames
    )
    maximum_depth_index = max(
        int(rec["depth_frame_idx"])
        for rec in frames
    )

    if maximum_sequence_index >= sequence_frame_count:
        raise ValueError(
            f"sequence YUV contains {sequence_frame_count} frames, "
            f"but JSONL requires frame_idx={maximum_sequence_index}"
        )

    if maximum_depth_index >= depth_frame_count:
        raise ValueError(
            f"depth YUV contains {depth_frame_count} frames, "
            f"but JSONL requires depth_frame_idx={maximum_depth_index}"
        )

    luma_reader = LumaReader(
        sequence_yuv,
        args.width,
        args.height,
    )
    depth_reader = DepthReader(
        depth_yuv,
        args.width,
        args.height,
        depth_scale_by_gop,
    )

    qstep_pairs = make_qstep_pairs(args)

    records_by_gop_local: dict[
        int,
        dict[int, dict[str, Any]]
    ] = {}

    baseline_pose_by_gop: dict[
        int,
        tuple[np.ndarray, np.ndarray]
    ] = {}

    pairs_by_gop: dict[
        int,
        list[tuple[int, int, str]]
    ] = {}

    for gop_idx, records in groups.items():
        records_by_gop_local[gop_idx] = {
            int(record["local_poc"]): record
            for record in records
        }

        source_relative_r = np.stack(
            [record["_rvec_np"] for record in records],
            axis=0,
        )
        source_relative_t = np.stack(
            [record["_tvec_np"] for record in records],
            axis=0,
        )

        baseline_pose_by_gop[gop_idx] = (
            reconstruct_absolute_from_relative(
                source_relative_r,
                source_relative_t,
            )
        )

        pairs_by_gop[gop_idx] = build_pairs_for_gop(
            records,
            args,
        )

    per_setting_rows: list[dict[str, Any]] = []
    per_gop_rows: list[dict[str, Any]] = []
    per_pair_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    settings_json: list[dict[str, Any]] = []

    for setting_idx, (
        rotation_qstep,
        translation_qstep,
    ) in enumerate(qstep_pairs):
        options = CodingOptions(
            rot_qstep=float(rotation_qstep),
            trans_qstep=float(translation_qstep),
            predictor=str(args.predictor),
            first_frame_bits=int(args.first_frame_bits),
            per_gop_overhead_bits=int(args.per_gop_overhead_bits),
            per_frame_overhead_bits=int(args.per_frame_overhead_bits),
        )

        setting_gop_rows: list[dict[str, Any]] = []
        setting_pair_rows: list[dict[str, Any]] = []
        setting_frame_rows: list[dict[str, Any]] = []

        for gop_idx in sorted(groups):
            (
                gop_summary,
                frame_rows,
                R_absolute_quantized,
                t_absolute_quantized,
            ) = simulate_relative_dpcm(
                gop_idx,
                groups[gop_idx],
                options,
            )

            R_absolute_baseline, t_absolute_baseline = (
                baseline_pose_by_gop[gop_idx]
            )

            pair_rows_for_gop: list[dict[str, Any]] = []

            for pair_idx, (
                target_local_poc,
                reference_local_poc,
                pair_kind,
            ) in enumerate(pairs_by_gop[gop_idx]):
                target_record = records_by_gop_local[gop_idx][
                    target_local_poc
                ]
                reference_record = records_by_gop_local[gop_idx][
                    reference_local_poc
                ]

                metrics = evaluate_pair(
                    target_record=target_record,
                    reference_record=reference_record,
                    baseline_target_pose=(
                        R_absolute_baseline[target_local_poc],
                        t_absolute_baseline[target_local_poc],
                    ),
                    baseline_reference_pose=(
                        R_absolute_baseline[reference_local_poc],
                        t_absolute_baseline[reference_local_poc],
                    ),
                    quantized_target_pose=(
                        R_absolute_quantized[target_local_poc],
                        t_absolute_quantized[target_local_poc],
                    ),
                    quantized_reference_pose=(
                        R_absolute_quantized[reference_local_poc],
                        t_absolute_quantized[reference_local_poc],
                    ),
                    luma_reader=luma_reader,
                    depth_reader=depth_reader,
                    args=args,
                )

                pair_row = {
                    "setting_idx": int(setting_idx),
                    "rot_qstep": float(rotation_qstep),
                    "trans_qstep": float(translation_qstep),
                    "predictor": str(args.predictor),
                    "gop_idx": int(gop_idx),
                    "gop_name": str(
                        groups[gop_idx][0]["gop_name"]
                    ),
                    "pair_idx": int(pair_idx),
                    "pair_kind": str(pair_kind),
                    "target_local_poc": int(target_local_poc),
                    "reference_local_poc": int(reference_local_poc),
                    "target_poc": int(target_record["poc"]),
                    "reference_poc": int(reference_record["poc"]),
                    "target_frame_idx": int(
                        target_record["frame_idx"]
                    ),
                    "reference_frame_idx": int(
                        reference_record["frame_idx"]
                    ),
                    **metrics,
                }

                pair_rows_for_gop.append(pair_row)
                setting_pair_rows.append(pair_row)

            valid_gop_pairs = [
                row
                for row in pair_rows_for_gop
                if row["baseline_common_psnr"] is not None
                and row["quantized_common_psnr"] is not None
                and row["common_valid_count"] > 0
            ]

            if valid_gop_pairs:
                mean_baseline_psnr = float(
                    np.mean(
                        [
                            row["baseline_common_psnr"]
                            for row in valid_gop_pairs
                        ]
                    )
                )
                mean_quantized_psnr = float(
                    np.mean(
                        [
                            row["quantized_common_psnr"]
                            for row in valid_gop_pairs
                        ]
                    )
                )
                mean_psnr_drop = float(
                    np.mean(
                        [
                            row["psnr_drop_common"]
                            for row in valid_gop_pairs
                        ]
                    )
                )

                pooled_count = int(
                    sum(
                        row["common_valid_count"]
                        for row in valid_gop_pairs
                    )
                )
                pooled_baseline_sse = float(
                    sum(
                        row["baseline_common_sse"]
                        for row in valid_gop_pairs
                    )
                )
                pooled_quantized_sse = float(
                    sum(
                        row["quantized_common_sse"]
                        for row in valid_gop_pairs
                    )
                )

                maximum_value_squared = float(
                    ((1 << args.bitdepth) - 1) ** 2
                )

                pooled_baseline_mse = (
                    pooled_baseline_sse / pooled_count
                )
                pooled_quantized_mse = (
                    pooled_quantized_sse / pooled_count
                )

                pooled_baseline_psnr = float(
                    10.0
                    * math.log10(
                        maximum_value_squared
                        / max(pooled_baseline_mse, 1e-30)
                    )
                )
                pooled_quantized_psnr = float(
                    10.0
                    * math.log10(
                        maximum_value_squared
                        / max(pooled_quantized_mse, 1e-30)
                    )
                )
                pooled_psnr_drop = float(
                    pooled_baseline_psnr
                    - pooled_quantized_psnr
                )
            else:
                mean_baseline_psnr = None
                mean_quantized_psnr = None
                mean_psnr_drop = None
                pooled_count = 0
                pooled_baseline_psnr = None
                pooled_quantized_psnr = None
                pooled_psnr_drop = None

            gop_summary.update(
                {
                    "setting_idx": int(setting_idx),
                    "pair_count": int(len(pair_rows_for_gop)),
                    "valid_pair_count": int(len(valid_gop_pairs)),
                    "mean_baseline_common_psnr": mean_baseline_psnr,
                    "mean_quantized_common_psnr": mean_quantized_psnr,
                    "mean_psnr_drop_common": mean_psnr_drop,
                    "pooled_common_valid_count": int(pooled_count),
                    "pooled_baseline_psnr": pooled_baseline_psnr,
                    "pooled_quantized_psnr": pooled_quantized_psnr,
                    "pooled_psnr_drop": pooled_psnr_drop,
                }
            )

            setting_gop_rows.append(gop_summary)
            setting_frame_rows.extend(frame_rows)

        valid_setting_pairs = [
            row
            for row in setting_pair_rows
            if row["baseline_common_psnr"] is not None
            and row["quantized_common_psnr"] is not None
            and row["common_valid_count"] > 0
        ]

        total_rotation_bits = int(
            sum(row["rotation_bits"] for row in setting_gop_rows)
        )
        total_translation_bits = int(
            sum(row["translation_bits"] for row in setting_gop_rows)
        )
        total_overhead_bits = int(
            sum(row["overhead_bits"] for row in setting_gop_rows)
        )
        total_bits = (
            total_rotation_bits
            + total_translation_bits
            + total_overhead_bits
        )

        total_camera_records = int(
            sum(row["frame_count"] for row in setting_gop_rows)
        )
        total_coded_frames = int(
            sum(row["coded_frame_count"] for row in setting_gop_rows)
        )

        if valid_setting_pairs:
            mean_baseline_psnr = float(
                np.mean(
                    [
                        row["baseline_common_psnr"]
                        for row in valid_setting_pairs
                    ]
                )
            )
            mean_quantized_psnr = float(
                np.mean(
                    [
                        row["quantized_common_psnr"]
                        for row in valid_setting_pairs
                    ]
                )
            )
            mean_psnr_drop = float(
                np.mean(
                    [
                        row["psnr_drop_common"]
                        for row in valid_setting_pairs
                    ]
                )
            )
            median_psnr_drop = float(
                np.median(
                    [
                        row["psnr_drop_common"]
                        for row in valid_setting_pairs
                    ]
                )
            )
            max_psnr_drop = float(
                np.max(
                    [
                        row["psnr_drop_common"]
                        for row in valid_setting_pairs
                    ]
                )
            )

            pooled_count = int(
                sum(
                    row["common_valid_count"]
                    for row in valid_setting_pairs
                )
            )
            pooled_baseline_sse = float(
                sum(
                    row["baseline_common_sse"]
                    for row in valid_setting_pairs
                )
            )
            pooled_quantized_sse = float(
                sum(
                    row["quantized_common_sse"]
                    for row in valid_setting_pairs
                )
            )

            maximum_value_squared = float(
                ((1 << args.bitdepth) - 1) ** 2
            )

            pooled_baseline_mse = (
                pooled_baseline_sse / pooled_count
            )
            pooled_quantized_mse = (
                pooled_quantized_sse / pooled_count
            )

            pooled_baseline_psnr = float(
                10.0
                * math.log10(
                    maximum_value_squared
                    / max(pooled_baseline_mse, 1e-30)
                )
            )
            pooled_quantized_psnr = float(
                10.0
                * math.log10(
                    maximum_value_squared
                    / max(pooled_quantized_mse, 1e-30)
                )
            )
            pooled_psnr_drop = float(
                pooled_baseline_psnr
                - pooled_quantized_psnr
            )
        else:
            mean_baseline_psnr = None
            mean_quantized_psnr = None
            mean_psnr_drop = None
            median_psnr_drop = None
            max_psnr_drop = None
            pooled_count = 0
            pooled_baseline_psnr = None
            pooled_quantized_psnr = None
            pooled_psnr_drop = None

        setting_summary = {
            "setting_idx": int(setting_idx),
            "rot_qstep": float(rotation_qstep),
            "trans_qstep": float(translation_qstep),
            "predictor": str(args.predictor),
            "gop_count": int(len(setting_gop_rows)),
            "pair_count": int(len(setting_pair_rows)),
            "valid_pair_count": int(len(valid_setting_pairs)),
            "camera_record_count": int(total_camera_records),
            "coded_frame_count": int(total_coded_frames),
            "rotation_bits": int(total_rotation_bits),
            "translation_bits": int(total_translation_bits),
            "overhead_bits": int(total_overhead_bits),
            "total_bits": int(total_bits),
            "total_bytes_ceil": int((total_bits + 7) // 8),
            "bits_per_camera_record": (
                float(total_bits / total_camera_records)
                if total_camera_records
                else 0.0
            ),
            "bits_per_coded_frame": (
                float(total_bits / total_coded_frames)
                if total_coded_frames
                else 0.0
            ),
            "mean_baseline_common_psnr": mean_baseline_psnr,
            "mean_quantized_common_psnr": mean_quantized_psnr,
            "mean_psnr_drop_common": mean_psnr_drop,
            "median_psnr_drop_common": median_psnr_drop,
            "max_psnr_drop_common": max_psnr_drop,
            "pooled_common_valid_count": int(pooled_count),
            "pooled_baseline_psnr": pooled_baseline_psnr,
            "pooled_quantized_psnr": pooled_quantized_psnr,
            "pooled_psnr_drop": pooled_psnr_drop,
        }

        per_setting_rows.append(setting_summary)
        per_gop_rows.extend(setting_gop_rows)
        per_pair_rows.extend(setting_pair_rows)
        per_frame_rows.extend(setting_frame_rows)

        settings_json.append(
            {
                "summary": setting_summary,
                "per_gop": setting_gop_rows,
            }
        )

        print("=" * 80)
        print(
            f"setting {setting_idx}: predictor={args.predictor}, "
            f"rot_qstep={rotation_qstep:.9g}, "
            f"trans_qstep={translation_qstep:.9g}"
        )
        print(
            f"  pose bits             : {total_bits} "
            f"({(total_bits + 7) // 8} bytes ceil)"
        )
        print(
            f"  bits / coded frame    : "
            f"{setting_summary['bits_per_coded_frame']:.4f}"
        )
        print(
            f"  mean baseline PSNR    : {mean_baseline_psnr}"
        )
        print(
            f"  mean quantized PSNR   : {mean_quantized_psnr}"
        )
        print(
            f"  mean PSNR drop        : {mean_psnr_drop}"
        )
        print(
            f"  pooled baseline PSNR  : {pooled_baseline_psnr}"
        )
        print(
            f"  pooled quantized PSNR : {pooled_quantized_psnr}"
        )
        print(
            f"  pooled PSNR drop      : {pooled_psnr_drop}"
        )

    result = {
        "input_jsonl": str(input_jsonl),
        "sequence_yuv": str(sequence_yuv),
        "depth_yuv": str(depth_yuv),
        "width": int(args.width),
        "height": int(args.height),
        "sequence_frame_count": int(sequence_frame_count),
        "depth_frame_count": int(depth_frame_count),
        "input_header": header,
        "depth_scale_by_gop": {
            str(key): float(value)
            for key, value in sorted(depth_scale_by_gop.items())
        },
        "coding_model": {
            "input_pose_mode": "current_to_previous",
            "input_relative_transform": (
                "X_{i-1}=R_step_i X_i+t_step_i"
            ),
            "predictor": str(args.predictor),
            "first_relative_predictor": "zero",
            "residual": (
                "source adjacent-relative rvec/tvec minus predicted "
                "reconstructed adjacent-relative rvec/tvec"
            ),
            "quantization": "independent uniform scalar quantization",
            "bit_estimation": (
                "signed Exp-Golomb order-0 over six quantized residual integers"
            ),
            "absolute_reconstruction": (
                "R_i=R_step_i.T@R_{i-1}; "
                "t_i=R_step_i.T@(t_{i-1}-t_step_i)"
            ),
        },
        "warp_model": {
            "pair_source": (
                "explicit" if args.pairs else args.pair_source
            ),
            "bidirectional": bool(
                not args.no_bidirectional_pairs
            ),
            "relative_target_to_reference": (
                "R_rel=R_ref@R_target.T; "
                "t_rel=t_ref-R_rel@t_target"
            ),
            "metric": (
                "baseline and quantized pose warps are evaluated on their "
                "common valid mask"
            ),
        },
        "options": vars(args),
        "settings": settings_json,
    }

    summary_path = Path(
        str(output_prefix) + "_summary.json"
    )
    setting_csv_path = Path(
        str(output_prefix) + "_per_setting.csv"
    )
    gop_csv_path = Path(
        str(output_prefix) + "_per_gop.csv"
    )
    pair_csv_path = Path(
        str(output_prefix) + "_per_pair.csv"
    )
    frame_csv_path = Path(
        str(output_prefix) + "_per_frame_bits.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    write_csv(setting_csv_path, per_setting_rows)
    write_csv(gop_csv_path, per_gop_rows)
    write_csv(pair_csv_path, per_pair_rows)
    write_csv(frame_csv_path, per_frame_rows)

    print("=" * 80)
    print(f"summary JSON       : {summary_path}")
    print(f"per-setting CSV    : {setting_csv_path}")
    print(f"per-GOP CSV        : {gop_csv_path}")
    print(f"per-pair CSV       : {pair_csv_path}")
    print(f"per-frame bits CSV : {frame_csv_path}")


if __name__ == "__main__":
    main()
