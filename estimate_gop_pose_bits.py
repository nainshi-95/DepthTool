#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_gop_pose_exp_golomb_bits.py

Estimate pose signaling bits from the merged camera JSONL produced by
merge_gop_geometry_gop0_relative.py.

Expected frame records:
    {
      "type": "frame",
      "gop_idx": ...,
      "gop_name": ...,
      "local_poc": ...,
      "poc": ...,
      "rvec": [rx, ry, rz],
      "tvec": [tx, ty, tz]
    }

Pose convention of the input:
    Every GOP is independently anchored at local_poc 0.

        R_0 = I
        t_0 = 0

        X_i = R_i X_0 + t_i

Coding model:
    - GOP local_poc 0 is implicit and costs zero pose-residual bits.
    - Frames are processed in ascending POC order within each GOP.
    - Encoder and decoder both use the previous reconstructed pose.

Rotation:
    R_res_i = R_i @ R_rec_prev.T
    r_res_i = Rodrigues^{-1}(R_res_i)
    q_r_i   = round(r_res_i / rot_qstep)
    r_hat_i = q_r_i * rot_qstep
    R_rec_i = Rodrigues(r_hat_i) @ R_rec_prev

Translation:
    t_res_i = t_i - t_rec_prev
    q_t_i   = round(t_res_i / trans_qstep)
    t_hat_i = q_t_i * trans_qstep
    t_rec_i = t_rec_prev + t_hat_i

Each signed quantized integer is mapped to an unsigned codeNum:
    0  -> 0
    +1 -> 1
    -1 -> 2
    +2 -> 3
    -2 -> 4
    ...

Unsigned Exp-Golomb order-0 length:
    bits = 2 * floor(log2(codeNum + 1)) + 1

Outputs:
    <prefix>_summary.json
    <prefix>_per_gop.csv
    <prefix>_per_frame.csv

Examples:
    python estimate_gop_pose_exp_golomb_bits.py \
        --input-jsonl seq_camParam_merged.jsonl \
        --rot-qstep 1e-5 \
        --trans-qstep 1e-5

    # Sweep multiple qsteps:
    python estimate_gop_pose_exp_golomb_bits.py \
        --input-jsonl seq_camParam_merged.jsonl \
        --rot-qsteps 1e-6,2e-6,5e-6,1e-5 \
        --trans-qsteps 1e-6,2e-6,5e-6,1e-5

Notes:
    - This is an approximate syntax-bit estimate.
    - It does not include CABAC context coding, flags, headers, alignment,
      qstep signaling, GOP identifiers, or container overhead unless enabled
      through the fixed overhead options.
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
# Basic rotation helpers
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


# ============================================================
# Exp-Golomb bit estimation
# ============================================================

def signed_to_code_num(value: int) -> int:
    """
    Signed integer mapping:
         0 -> 0
        +1 -> 1
        -1 -> 2
        +2 -> 3
        -2 -> 4
    """
    value = int(value)
    if value > 0:
        return 2 * value - 1
    if value < 0:
        return -2 * value
    return 0


def ue_bits(code_num: int) -> int:
    """
    Unsigned Exp-Golomb order-0 code length.

        length = 2 * floor(log2(code_num + 1)) + 1
    """
    code_num = int(code_num)
    if code_num < 0:
        raise ValueError("code_num must be non-negative")
    return 2 * ((code_num + 1).bit_length() - 1) + 1


def se_bits(value: int) -> int:
    return ue_bits(signed_to_code_num(int(value)))


def vector_se_bits(values: np.ndarray) -> tuple[int, list[int]]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    component_bits = [se_bits(int(v)) for v in values]
    return int(sum(component_bits)), component_bits


# ============================================================
# JSONL loading
# ============================================================

def load_camera_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {e}"
                ) from e

            if not isinstance(obj, dict):
                continue

            if obj.get("type") == "header" and not header:
                header = obj
                continue

            if obj.get("type") != "frame":
                continue

            required = ["gop_idx", "rvec", "tvec"]
            missing = [k for k in required if k not in obj]
            if missing:
                raise KeyError(
                    f"{path}:{line_no}: missing frame keys {missing}"
                )

            rvec = np.asarray(obj["rvec"], dtype=np.float64).reshape(-1)
            tvec = np.asarray(obj["tvec"], dtype=np.float64).reshape(-1)

            if rvec.size != 3 or tvec.size != 3:
                raise ValueError(
                    f"{path}:{line_no}: rvec/tvec must each have 3 values"
                )

            rec = dict(obj)
            rec["_line_no"] = line_no
            rec["_rvec_np"] = rvec
            rec["_tvec_np"] = tvec
            rec["gop_idx"] = int(rec["gop_idx"])
            rec["local_poc"] = int(
                rec.get("local_poc", rec.get("poc", 0))
            )
            rec["poc"] = int(
                rec.get("poc", rec["local_poc"])
            )
            rec["gop_name"] = str(
                rec.get("gop_name", f"gop{rec['gop_idx']}")
            )
            frames.append(rec)

    if not frames:
        raise RuntimeError(f"No frame records found in {path}")

    return header, frames


def group_frames_by_gop(
    frames: list[dict[str, Any]],
    sort_key: str,
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}

    for rec in frames:
        groups.setdefault(int(rec["gop_idx"]), []).append(rec)

    for gop_idx, recs in groups.items():
        if sort_key == "poc":
            recs.sort(
                key=lambda r: (
                    int(r["poc"]),
                    int(r["local_poc"]),
                    int(r["_line_no"]),
                )
            )
        elif sort_key == "local_poc":
            recs.sort(
                key=lambda r: (
                    int(r["local_poc"]),
                    int(r["poc"]),
                    int(r["_line_no"]),
                )
            )
        else:
            raise ValueError(sort_key)

        local_pocs = [int(r["local_poc"]) for r in recs]
        if len(local_pocs) != len(set(local_pocs)):
            raise ValueError(
                f"GOP {gop_idx}: duplicate local_poc values: {local_pocs}"
            )

    return groups


# ============================================================
# Coding simulation
# ============================================================

@dataclass
class CodingOptions:
    rot_qstep: float
    trans_qstep: float
    first_frame_bits: int
    per_gop_overhead_bits: int
    per_frame_overhead_bits: int
    include_first_frame_residual: bool


def quantize_to_index(value: np.ndarray, qstep: float) -> np.ndarray:
    if qstep <= 0.0:
        raise ValueError("qstep must be positive")
    return np.rint(
        np.asarray(value, dtype=np.float64) / float(qstep)
    ).astype(np.int64)


def simulate_one_gop(
    gop_idx: int,
    records: list[dict[str, Any]],
    options: CodingOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise ValueError(f"GOP {gop_idx} is empty")

    first = records[0]

    if int(first["local_poc"]) != 0:
        raise ValueError(
            f"GOP {gop_idx}: first coded record has local_poc="
            f"{first['local_poc']}, expected 0"
        )

    # Decoder's implicit GOP anchor.
    R_rec_prev = np.eye(3, dtype=np.float64)
    t_rec_prev = np.zeros(3, dtype=np.float64)

    per_frame: list[dict[str, Any]] = []

    total_rot_bits = 0
    total_trans_bits = 0
    total_overhead_bits = int(options.per_gop_overhead_bits)
    total_rot_abs_qindex = 0
    total_trans_abs_qindex = 0
    total_rot_components = 0
    total_trans_components = 0
    total_rot_zero = 0
    total_trans_zero = 0

    rot_recon_angle_errors: list[float] = []
    trans_recon_l2_errors: list[float] = []

    for order_idx, rec in enumerate(records):
        R_true = R_from_rvec(rec["_rvec_np"])
        t_true = np.asarray(rec["_tvec_np"], dtype=np.float64).reshape(3)

        is_anchor = int(rec["local_poc"]) == 0

        if is_anchor and not options.include_first_frame_residual:
            rot_res = np.zeros(3, dtype=np.float64)
            trans_res = np.zeros(3, dtype=np.float64)
            q_rot = np.zeros(3, dtype=np.int64)
            q_trans = np.zeros(3, dtype=np.int64)
            rot_bits = 0
            trans_bits = 0
            rot_component_bits = [0, 0, 0]
            trans_component_bits = [0, 0, 0]

            R_rec = np.eye(3, dtype=np.float64)
            t_rec = np.zeros(3, dtype=np.float64)

            overhead_bits = int(options.first_frame_bits)
        else:
            # Rotation residual relative to previous reconstructed rotation.
            R_res = R_true @ R_rec_prev.T
            rot_res = rvec_from_R(R_res)
            q_rot = quantize_to_index(
                rot_res,
                options.rot_qstep,
            )
            rot_res_hat = (
                q_rot.astype(np.float64)
                * float(options.rot_qstep)
            )
            R_res_hat = R_from_rvec(rot_res_hat)
            R_rec = R_res_hat @ R_rec_prev

            # Translation difference from previous reconstructed absolute tvec.
            trans_res = t_true - t_rec_prev
            q_trans = quantize_to_index(
                trans_res,
                options.trans_qstep,
            )
            trans_res_hat = (
                q_trans.astype(np.float64)
                * float(options.trans_qstep)
            )
            t_rec = t_rec_prev + trans_res_hat

            rot_bits, rot_component_bits = vector_se_bits(q_rot)
            trans_bits, trans_component_bits = vector_se_bits(q_trans)
            overhead_bits = int(options.per_frame_overhead_bits)

        total_rot_bits += int(rot_bits)
        total_trans_bits += int(trans_bits)
        total_overhead_bits += int(overhead_bits)

        total_rot_abs_qindex += int(np.sum(np.abs(q_rot)))
        total_trans_abs_qindex += int(np.sum(np.abs(q_trans)))
        total_rot_components += 3
        total_trans_components += 3
        total_rot_zero += int(np.count_nonzero(q_rot == 0))
        total_trans_zero += int(np.count_nonzero(q_trans == 0))

        # Reconstruction error measured against the current original local pose.
        R_err = R_true @ R_rec.T
        r_err = rvec_from_R(R_err)
        rot_angle_error = float(np.linalg.norm(r_err))
        trans_l2_error = float(np.linalg.norm(t_true - t_rec))

        rot_recon_angle_errors.append(rot_angle_error)
        trans_recon_l2_errors.append(trans_l2_error)

        frame_total_bits = (
            int(rot_bits)
            + int(trans_bits)
            + int(overhead_bits)
        )

        per_frame.append(
            {
                "gop_idx": int(gop_idx),
                "gop_name": str(rec["gop_name"]),
                "coding_order_idx": int(order_idx),
                "local_poc": int(rec["local_poc"]),
                "poc": int(rec["poc"]),
                "is_anchor": bool(is_anchor),
                "rot_qstep": float(options.rot_qstep),
                "trans_qstep": float(options.trans_qstep),
                "rot_res_x": float(rot_res[0]),
                "rot_res_y": float(rot_res[1]),
                "rot_res_z": float(rot_res[2]),
                "trans_res_x": float(trans_res[0]),
                "trans_res_y": float(trans_res[1]),
                "trans_res_z": float(trans_res[2]),
                "rot_q_x": int(q_rot[0]),
                "rot_q_y": int(q_rot[1]),
                "rot_q_z": int(q_rot[2]),
                "trans_q_x": int(q_trans[0]),
                "trans_q_y": int(q_trans[1]),
                "trans_q_z": int(q_trans[2]),
                "rot_bits_x": int(rot_component_bits[0]),
                "rot_bits_y": int(rot_component_bits[1]),
                "rot_bits_z": int(rot_component_bits[2]),
                "trans_bits_x": int(trans_component_bits[0]),
                "trans_bits_y": int(trans_component_bits[1]),
                "trans_bits_z": int(trans_component_bits[2]),
                "rot_bits": int(rot_bits),
                "trans_bits": int(trans_bits),
                "overhead_bits": int(overhead_bits),
                "total_bits": int(frame_total_bits),
                "rot_recon_angle_error_rad": rot_angle_error,
                "rot_recon_angle_error_deg": float(
                    np.degrees(rot_angle_error)
                ),
                "trans_recon_l2_error": trans_l2_error,
                "recon_rvec_x": float(rvec_from_R(R_rec)[0]),
                "recon_rvec_y": float(rvec_from_R(R_rec)[1]),
                "recon_rvec_z": float(rvec_from_R(R_rec)[2]),
                "recon_tvec_x": float(t_rec[0]),
                "recon_tvec_y": float(t_rec[1]),
                "recon_tvec_z": float(t_rec[2]),
            }
        )

        R_rec_prev = R_rec
        t_rec_prev = t_rec

    coded_frame_count = sum(
        0 if (r["is_anchor"] and not options.include_first_frame_residual) else 1
        for r in per_frame
    )

    total_bits = (
        total_rot_bits
        + total_trans_bits
        + total_overhead_bits
    )

    summary = {
        "gop_idx": int(gop_idx),
        "gop_name": str(records[0]["gop_name"]),
        "frame_count": int(len(records)),
        "coded_frame_count": int(coded_frame_count),
        "first_poc": int(records[0]["poc"]),
        "last_poc": int(records[-1]["poc"]),
        "rot_qstep": float(options.rot_qstep),
        "trans_qstep": float(options.trans_qstep),
        "rotation_bits": int(total_rot_bits),
        "translation_bits": int(total_trans_bits),
        "overhead_bits": int(total_overhead_bits),
        "total_bits": int(total_bits),
        "bits_per_all_frame": float(total_bits / len(records)),
        "bits_per_coded_frame": (
            float(total_bits / coded_frame_count)
            if coded_frame_count > 0
            else 0.0
        ),
        "rot_mean_abs_qindex": (
            float(total_rot_abs_qindex / total_rot_components)
            if total_rot_components
            else 0.0
        ),
        "trans_mean_abs_qindex": (
            float(total_trans_abs_qindex / total_trans_components)
            if total_trans_components
            else 0.0
        ),
        "rot_zero_ratio": (
            float(total_rot_zero / total_rot_components)
            if total_rot_components
            else 1.0
        ),
        "trans_zero_ratio": (
            float(total_trans_zero / total_trans_components)
            if total_trans_components
            else 1.0
        ),
        "rot_recon_angle_error_mean_rad": float(
            np.mean(rot_recon_angle_errors)
        ),
        "rot_recon_angle_error_max_rad": float(
            np.max(rot_recon_angle_errors)
        ),
        "rot_recon_angle_error_mean_deg": float(
            np.degrees(np.mean(rot_recon_angle_errors))
        ),
        "rot_recon_angle_error_max_deg": float(
            np.degrees(np.max(rot_recon_angle_errors))
        ),
        "trans_recon_l2_error_mean": float(
            np.mean(trans_recon_l2_errors)
        ),
        "trans_recon_l2_error_max": float(
            np.max(trans_recon_l2_errors)
        ),
    }

    return summary, per_frame


# ============================================================
# Qstep parsing and output
# ============================================================

def parse_float_list(text: str) -> list[float]:
    vals = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v <= 0.0:
            raise ValueError(f"qstep must be positive: {v}")
        vals.append(v)

    if not vals:
        raise ValueError("empty qstep list")

    return vals


def make_qstep_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.rot_qsteps:
        rot_steps = parse_float_list(args.rot_qsteps)
    else:
        rot_steps = [float(args.rot_qstep)]

    if args.trans_qsteps:
        trans_steps = parse_float_list(args.trans_qsteps)
    else:
        trans_steps = [float(args.trans_qstep)]

    if args.paired_qsteps:
        if len(rot_steps) != len(trans_steps):
            raise ValueError(
                "--paired-qsteps requires equal numbers of rotation "
                "and translation qsteps"
            )
        return list(zip(rot_steps, trans_steps))

    return [
        (rq, tq)
        for rq in rot_steps
        for tq in trans_steps
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate closed-loop predictive GOP pose signaling bits "
            "using signed Exp-Golomb coding."
        )
    )

    ap.add_argument(
        "--input-jsonl",
        required=True,
        help="Merged camera JSONL with GOP-frame-0-relative poses",
    )
    ap.add_argument(
        "--output-prefix",
        default="",
        help=(
            "Output path prefix. Default: input filename without suffix "
            "+ '_pose_bits'"
        ),
    )

    ap.add_argument(
        "--rot-qstep",
        type=float,
        default=1e-5,
        help="Rotation Rodrigues residual qstep in radians",
    )
    ap.add_argument(
        "--trans-qstep",
        type=float,
        default=1e-5,
        help="Translation residual qstep",
    )
    ap.add_argument(
        "--rot-qsteps",
        default="",
        help="Comma-separated rotation qstep sweep",
    )
    ap.add_argument(
        "--trans-qsteps",
        default="",
        help="Comma-separated translation qstep sweep",
    )
    ap.add_argument(
        "--paired-qsteps",
        action="store_true",
        help=(
            "Pair rot/trans qstep lists by position instead of taking "
            "their Cartesian product"
        ),
    )

    ap.add_argument(
        "--sort-key",
        choices=["poc", "local_poc"],
        default="poc",
        help="Coding order inside each GOP",
    )
    ap.add_argument(
        "--include-first-frame-residual",
        action="store_true",
        help=(
            "Code frame 0 residual too. Normally disabled because GOP "
            "frame 0 is implicit I/0."
        ),
    )

    ap.add_argument(
        "--first-frame-bits",
        type=int,
        default=0,
        help=(
            "Optional fixed overhead charged to each implicit GOP anchor"
        ),
    )
    ap.add_argument(
        "--per-gop-overhead-bits",
        type=int,
        default=0,
        help="Optional fixed syntax/header overhead per GOP",
    )
    ap.add_argument(
        "--per-frame-overhead-bits",
        type=int,
        default=0,
        help="Optional fixed syntax overhead per coded non-anchor frame",
    )

    args = ap.parse_args()

    if args.rot_qstep <= 0.0:
        raise ValueError("--rot-qstep must be positive")
    if args.trans_qstep <= 0.0:
        raise ValueError("--trans-qstep must be positive")

    for name in [
        "first_frame_bits",
        "per_gop_overhead_bits",
        "per_frame_overhead_bits",
    ]:
        if int(getattr(args, name)) < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be non-negative"
            )

    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        output_prefix = input_path.with_suffix("")
        output_prefix = output_prefix.parent / (
            output_prefix.name + "_pose_bits"
        )

    header, frames = load_camera_jsonl(input_path)
    groups = group_frames_by_gop(
        frames,
        args.sort_key,
    )
    qstep_pairs = make_qstep_pairs(args)

    all_gop_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    sweep_summaries: list[dict[str, Any]] = []

    for rot_qstep, trans_qstep in qstep_pairs:
        options = CodingOptions(
            rot_qstep=float(rot_qstep),
            trans_qstep=float(trans_qstep),
            first_frame_bits=int(args.first_frame_bits),
            per_gop_overhead_bits=int(args.per_gop_overhead_bits),
            per_frame_overhead_bits=int(args.per_frame_overhead_bits),
            include_first_frame_residual=bool(
                args.include_first_frame_residual
            ),
        )

        gop_rows_this_setting: list[dict[str, Any]] = []
        frame_rows_this_setting: list[dict[str, Any]] = []

        for gop_idx in sorted(groups):
            gop_summary, frame_rows = simulate_one_gop(
                gop_idx,
                groups[gop_idx],
                options,
            )
            gop_rows_this_setting.append(gop_summary)
            frame_rows_this_setting.extend(frame_rows)

        total_rot_bits = int(
            sum(r["rotation_bits"] for r in gop_rows_this_setting)
        )
        total_trans_bits = int(
            sum(r["translation_bits"] for r in gop_rows_this_setting)
        )
        total_overhead_bits = int(
            sum(r["overhead_bits"] for r in gop_rows_this_setting)
        )
        total_bits = (
            total_rot_bits
            + total_trans_bits
            + total_overhead_bits
        )
        total_frames = int(
            sum(r["frame_count"] for r in gop_rows_this_setting)
        )
        total_coded_frames = int(
            sum(r["coded_frame_count"] for r in gop_rows_this_setting)
        )

        sweep_summary = {
            "rot_qstep": float(rot_qstep),
            "trans_qstep": float(trans_qstep),
            "gop_count": int(len(gop_rows_this_setting)),
            "camera_record_count": int(total_frames),
            "coded_frame_count": int(total_coded_frames),
            "rotation_bits": int(total_rot_bits),
            "translation_bits": int(total_trans_bits),
            "overhead_bits": int(total_overhead_bits),
            "total_bits": int(total_bits),
            "total_bytes_ceil": int((total_bits + 7) // 8),
            "bits_per_camera_record": (
                float(total_bits / total_frames)
                if total_frames
                else 0.0
            ),
            "bits_per_coded_frame": (
                float(total_bits / total_coded_frames)
                if total_coded_frames
                else 0.0
            ),
            "mean_gop_bits": float(
                np.mean(
                    [r["total_bits"] for r in gop_rows_this_setting]
                )
            ),
            "max_gop_bits": int(
                max(r["total_bits"] for r in gop_rows_this_setting)
            ),
            "mean_rot_recon_error_deg": float(
                np.mean(
                    [
                        r["rot_recon_angle_error_mean_deg"]
                        for r in gop_rows_this_setting
                    ]
                )
            ),
            "max_rot_recon_error_deg": float(
                max(
                    r["rot_recon_angle_error_max_deg"]
                    for r in gop_rows_this_setting
                )
            ),
            "mean_trans_recon_l2_error": float(
                np.mean(
                    [
                        r["trans_recon_l2_error_mean"]
                        for r in gop_rows_this_setting
                    ]
                )
            ),
            "max_trans_recon_l2_error": float(
                max(
                    r["trans_recon_l2_error_max"]
                    for r in gop_rows_this_setting
                )
            ),
        }

        sweep_summaries.append(sweep_summary)
        all_gop_rows.extend(gop_rows_this_setting)
        all_frame_rows.extend(frame_rows_this_setting)

        print("=" * 72)
        print(
            f"rot_qstep={rot_qstep:.9g}, "
            f"trans_qstep={trans_qstep:.9g}"
        )
        print(
            f"  total bits       : {total_bits} "
            f"({(total_bits + 7) // 8} bytes ceil)"
        )
        print(f"  rotation bits    : {total_rot_bits}")
        print(f"  translation bits : {total_trans_bits}")
        print(f"  overhead bits    : {total_overhead_bits}")
        print(
            f"  bits / record    : "
            f"{sweep_summary['bits_per_camera_record']:.4f}"
        )
        print(
            f"  bits / coded frm : "
            f"{sweep_summary['bits_per_coded_frame']:.4f}"
        )
        print(
            f"  max rot err deg  : "
            f"{sweep_summary['max_rot_recon_error_deg']:.9g}"
        )
        print(
            f"  max trans error  : "
            f"{sweep_summary['max_trans_recon_l2_error']:.9g}"
        )

    result = {
        "input_jsonl": str(input_path),
        "input_header": header,
        "coding_model": {
            "gop_anchor": (
                "local_poc 0 is implicit R=I,t=0 unless "
                "--include-first-frame-residual is used"
            ),
            "coding_order": str(args.sort_key),
            "rotation_predictor": (
                "previous reconstructed rotation"
            ),
            "rotation_residual": (
                "rvec(log(R_true @ R_reconstructed_previous.T))"
            ),
            "rotation_reconstruction": (
                "R_rec_i = Rodrigues(q_r * rot_qstep) @ R_rec_prev"
            ),
            "translation_predictor": (
                "previous reconstructed absolute local tvec"
            ),
            "translation_residual": (
                "t_true_i - t_reconstructed_previous"
            ),
            "translation_reconstruction": (
                "t_rec_i = t_rec_prev + q_t * trans_qstep"
            ),
            "signed_mapping": (
                "0->0,+1->1,-1->2,+2->3,-2->4,..."
            ),
            "exp_golomb_order": 0,
            "ue_length_formula": (
                "2*floor(log2(codeNum+1))+1"
            ),
            "excluded_overhead": (
                "CABAC contexts, flags, qstep signaling, byte alignment, "
                "containers and other syntax unless fixed overhead options "
                "are explicitly supplied"
            ),
        },
        "options": {
            "sort_key": args.sort_key,
            "include_first_frame_residual": bool(
                args.include_first_frame_residual
            ),
            "first_frame_bits": int(args.first_frame_bits),
            "per_gop_overhead_bits": int(
                args.per_gop_overhead_bits
            ),
            "per_frame_overhead_bits": int(
                args.per_frame_overhead_bits
            ),
            "paired_qsteps": bool(args.paired_qsteps),
        },
        "sweep": sweep_summaries,
        "per_gop": all_gop_rows,
    }

    summary_path = Path(str(output_prefix) + "_summary.json")
    gop_csv_path = Path(str(output_prefix) + "_per_gop.csv")
    frame_csv_path = Path(str(output_prefix) + "_per_frame.csv")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    write_csv(gop_csv_path, all_gop_rows)
    write_csv(frame_csv_path, all_frame_rows)

    print("=" * 72)
    print(f"summary JSON : {summary_path}")
    print(f"per-GOP CSV  : {gop_csv_path}")
    print(f"per-frame CSV: {frame_csv_path}")


if __name__ == "__main__":
    main()
