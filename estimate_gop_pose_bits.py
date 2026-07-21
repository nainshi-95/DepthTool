#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_ra_pose_bits_and_warp_psnr.py

Estimate:
1) pose signaling bits using closed-loop predictive quantization and
   signed Exp-Golomb order-0 coding;
2) camera-projection warping quality on an original YUV420p10le sequence
   using the merged depth YUV and each quantized pose reconstruction.

Input camera JSONL is expected to be produced by:
    merge_gop_geometry_gop0_relative.py

Input pose convention, independently for every GOP:
    local_poc 0:
        R_0 = I
        t_0 = 0

    local_poc i:
        X_i = R_i X_0 + t_i

Closed-loop pose coding:
    Rotation:
        R_res_i = R_i @ R_rec_prev.T
        r_res_i = Rodrigues^{-1}(R_res_i)
        q_r_i   = round(r_res_i / rot_qstep)
        R_rec_i = Rodrigues(q_r_i * rot_qstep) @ R_rec_prev

    Translation:
        t_res_i = t_i - t_rec_prev
        q_t_i   = round(t_res_i / trans_qstep)
        t_rec_i = t_rec_prev + q_t_i * trans_qstep

    GOP local_poc 0 is implicit and costs zero residual bits by default.

Warping:
    For target t and reference r:
        R_rel = R_r @ R_t.T
        t_rel = t_r - R_rel @ t_t

        X_t   = depth_t * K_t^{-1} [x, y, 1]^T
        X_r   = R_rel X_t + t_rel
        q_r   = K_r X_r / X_r.z

    The reference Y frame is backward-remapped into the target domain.

RA pair generation:
    Default --pair-source dyadic recursively builds hierarchical random-access
    relations over each GOP:
        midpoint -> left endpoint
        midpoint -> right endpoint
    and, unless --no-bidirectional-pairs is used:
        left endpoint -> midpoint
        right endpoint -> midpoint

    You can instead use adjacent pairs or provide explicit local-poc pairs.

PSNR comparison:
    For every pair and qstep:
      - baseline pose warp is computed from the unquantized JSONL pose;
      - quantized pose warp is computed from the closed-loop reconstruction;
      - both PSNRs are evaluated on the SAME common valid mask:
            valid_common = valid_baseline & valid_quantized
      - PSNR drop:
            baseline_common_psnr - quantized_common_psnr

Outputs:
    <prefix>_summary.json
    <prefix>_per_setting.csv
    <prefix>_per_gop.csv
    <prefix>_per_pair.csv
    <prefix>_per_frame_bits.csv

Example:
    python estimate_ra_pose_bits_and_warp_psnr.py \
        --input-jsonl sequence_camParam_merged.jsonl \
        --sequence-yuv sequence_1920x1080_10bit.yuv \
        --depth-yuv sequence_depth_merged.yuv \
        --width 1920 --height 1080 \
        --rot-qsteps 1e-6,2e-6,5e-6,1e-5 \
        --trans-qsteps 1e-6,2e-6,5e-6,1e-5 \
        --paired-qsteps

Important:
    - This estimates only signed Exp-Golomb residual bits plus optional fixed
      overhead configured on the CLI.
    - It does not model CABAC contexts, qstep signaling, byte alignment,
      syntax flags, or container overhead.
    - Depth YUV is interpreted using the depth scale of the GOP that OWNS each
      merged depth frame, not necessarily the current camera-record GOP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


# ============================================================
# Geometry helpers
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

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
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
    return (
        width * height
        + 2 * (width // 2) * (height // 2)
    ) * 2


def count_yuv420p10le_frames(
    path: Path,
    width: int,
    height: int,
) -> int:
    return path.stat().st_size // frame_size_yuv420p10le(width, height)


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
            f"cannot read frame {frame_idx} from {path}; "
            f"got {y.size}/{y_count} Y samples"
        )

    return y.reshape(height, width)


# ============================================================
# JSONL
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
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {e}"
                ) from e

            if not isinstance(obj, dict):
                continue

            if obj.get("type") == "header":
                if not header:
                    header = obj
                continue

            if obj.get("type") != "frame":
                continue

            for key in ["gop_idx", "rvec", "tvec", "intrinsic"]:
                if key not in obj:
                    raise KeyError(
                        f"{path}:{line_no}: missing frame key '{key}'"
                    )

            rvec = np.asarray(obj["rvec"], dtype=np.float64).reshape(-1)
            tvec = np.asarray(obj["tvec"], dtype=np.float64).reshape(-1)

            if rvec.size != 3 or tvec.size != 3:
                raise ValueError(
                    f"{path}:{line_no}: rvec/tvec must each contain 3 values"
                )

            rec = dict(obj)
            rec["_line_no"] = int(line_no)
            rec["_rvec_np"] = rvec
            rec["_tvec_np"] = tvec
            rec["_R_np"] = R_from_rvec(rvec)
            rec["_K_np"] = K_from_record(rec)

            rec["gop_idx"] = int(rec["gop_idx"])
            rec["gop_name"] = str(
                rec.get("gop_name", f"gop{rec['gop_idx']}")
            )
            rec["local_poc"] = int(
                rec.get("local_poc", rec.get("poc", 0))
            )
            rec["poc"] = int(
                rec.get("poc", rec["local_poc"])
            )
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

        if local_pocs[0] != 0:
            raise ValueError(
                f"GOP {gop_idx}: first local_poc is {local_pocs[0]}, not 0"
            )

    return groups


def build_depth_scale_by_gop(
    header: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[int, float]:
    out: dict[int, float] = {}

    gops = header.get("gops")
    if isinstance(gops, list):
        for g in gops:
            if not isinstance(g, dict):
                continue
            if "gop_idx" in g and "depth_scale_real" in g:
                out[int(g["gop_idx"])] = float(g["depth_scale_real"])

    for rec in frames:
        gi = int(rec["gop_idx"])
        if gi not in out and "depth_scale_real" in rec:
            out[gi] = float(rec["depth_scale_real"])

    if not out:
        raise RuntimeError(
            "No depth_scale_real found in JSONL header or frame records"
        )

    for gi, value in out.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"invalid depth scale for GOP {gi}: {value}"
            )

    return out


# ============================================================
# RA pairs
# ============================================================

def parse_explicit_pairs(
    text: str,
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    if not text.strip():
        return out

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

        out.append(
            (
                int(parts[0]),
                int(parts[1]),
                "explicit",
            )
        )

    return out


def generate_adjacent_pairs(
    local_pocs: list[int],
    bidirectional: bool,
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    for i in range(1, len(local_pocs)):
        cur = local_pocs[i]
        prev = local_pocs[i - 1]
        out.append((cur, prev, "adjacent"))

        if bidirectional:
            out.append((prev, cur, "adjacent_reverse"))

    return out


def generate_dyadic_pairs(
    local_pocs: list[int],
    bidirectional: bool,
) -> list[tuple[int, int, str]]:
    """
    Generate hierarchical RA pairs over the sorted local POC positions.

    This works for arbitrary monotonically increasing local_poc values.
    The recursion is based on list positions, not numeric midpoint equality.
    """
    out: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    def add(target: int, ref: int, kind: str) -> None:
        key = (int(target), int(ref))
        if target == ref or key in seen:
            return
        seen.add(key)
        out.append((int(target), int(ref), kind))

    def rec(left_idx: int, right_idx: int, level: int) -> None:
        if right_idx <= left_idx + 1:
            return

        mid_idx = (left_idx + right_idx) // 2

        left_poc = local_pocs[left_idx]
        mid_poc = local_pocs[mid_idx]
        right_poc = local_pocs[right_idx]

        add(mid_poc, left_poc, f"dyadic_L{level}_left")
        add(mid_poc, right_poc, f"dyadic_L{level}_right")

        if bidirectional:
            add(left_poc, mid_poc, f"dyadic_L{level}_left_reverse")
            add(right_poc, mid_poc, f"dyadic_L{level}_right_reverse")

        rec(left_idx, mid_idx, level + 1)
        rec(mid_idx, right_idx, level + 1)

    if len(local_pocs) >= 2:
        # Include the two GOP endpoints because long-term endpoint references
        # are part of the RA hierarchy even when no midpoint exists.
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

    rec(0, len(local_pocs) - 1, 0)
    return out


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
                f"not found in local_poc set {sorted(available)}"
            )
        checked.append((target, ref, kind))

    if not checked:
        raise RuntimeError(
            f"GOP {records[0]['gop_idx']}: no valid pairs"
        )

    return checked


# ============================================================
# Pose reconstruction and bit simulation
# ============================================================

@dataclass
class CodingOptions:
    rot_qstep: float
    trans_qstep: float
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


def simulate_pose_coding(
    gop_idx: int,
    records: list[dict[str, Any]],
    options: CodingOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    R_rec_prev = np.eye(3, dtype=np.float64)
    t_rec_prev = np.zeros(3, dtype=np.float64)

    recon_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_frame: list[dict[str, Any]] = []

    total_rot_bits = 0
    total_trans_bits = 0
    total_overhead_bits = int(options.per_gop_overhead_bits)

    for coding_idx, rec in enumerate(records):
        local_poc = int(rec["local_poc"])
        R_true = np.asarray(rec["_R_np"], dtype=np.float64)
        t_true = np.asarray(rec["_tvec_np"], dtype=np.float64)

        if local_poc == 0:
            rot_res = np.zeros(3, dtype=np.float64)
            trans_res = np.zeros(3, dtype=np.float64)
            q_rot = np.zeros(3, dtype=np.int64)
            q_trans = np.zeros(3, dtype=np.int64)
            rot_component_bits = [0, 0, 0]
            trans_component_bits = [0, 0, 0]
            rot_bits = 0
            trans_bits = 0
            overhead_bits = int(options.first_frame_bits)

            R_rec = np.eye(3, dtype=np.float64)
            t_rec = np.zeros(3, dtype=np.float64)
        else:
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
            R_rec = R_from_rvec(rot_res_hat) @ R_rec_prev

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

        R_err = R_true @ R_rec.T
        rot_error_rad = float(
            np.linalg.norm(rvec_from_R(R_err))
        )
        trans_error = float(
            np.linalg.norm(t_true - t_rec)
        )

        recon_pose[local_poc] = (
            R_rec.copy(),
            t_rec.copy(),
        )

        per_frame.append(
            {
                "rot_qstep": float(options.rot_qstep),
                "trans_qstep": float(options.trans_qstep),
                "gop_idx": int(gop_idx),
                "gop_name": str(rec["gop_name"]),
                "coding_order_idx": int(coding_idx),
                "local_poc": local_poc,
                "poc": int(rec["poc"]),
                "is_anchor": local_poc == 0,
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
                "rotation_bits": int(rot_bits),
                "translation_bits": int(trans_bits),
                "overhead_bits": int(overhead_bits),
                "total_bits": int(
                    rot_bits + trans_bits + overhead_bits
                ),
                "rot_recon_error_rad": rot_error_rad,
                "rot_recon_error_deg": float(
                    np.degrees(rot_error_rad)
                ),
                "trans_recon_l2_error": trans_error,
            }
        )

        R_rec_prev = R_rec
        t_rec_prev = t_rec

    total_bits = (
        total_rot_bits
        + total_trans_bits
        + total_overhead_bits
    )

    summary = {
        "rot_qstep": float(options.rot_qstep),
        "trans_qstep": float(options.trans_qstep),
        "gop_idx": int(gop_idx),
        "gop_name": str(records[0]["gop_name"]),
        "frame_count": int(len(records)),
        "coded_frame_count": int(max(0, len(records) - 1)),
        "rotation_bits": int(total_rot_bits),
        "translation_bits": int(total_trans_bits),
        "overhead_bits": int(total_overhead_bits),
        "total_bits": int(total_bits),
        "bits_per_camera_record": float(
            total_bits / len(records)
        ),
        "bits_per_coded_frame": (
            float(total_bits / (len(records) - 1))
            if len(records) > 1
            else 0.0
        ),
    }

    return summary, per_frame, recon_pose


# ============================================================
# Depth cache
# ============================================================

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
                f"no depth scale for owner GOP {owner_gop_idx}"
            )

        code = read_yuv420p10le_y(
            self.path,
            self.width,
            self.height,
            depth_frame_idx,
        ).astype(np.float32)

        depth = code * float(
            self.scale_by_gop[owner_gop_idx]
        )

        self.cache[key] = depth
        return depth


class LumaReader:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
    ) -> None:
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


# ============================================================
# Warping and PSNR
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

    map_x = np.full(
        (height, width),
        -1.0,
        dtype=np.float32,
    )
    map_y = np.full(
        (height, width),
        -1.0,
        dtype=np.float32,
    )
    valid_all = np.zeros(
        (height, width),
        dtype=bool,
    )

    xs_full = np.arange(
        width,
        dtype=np.float64,
    )

    for y0 in range(0, height, row_batch):
        y1 = min(height, y0 + row_batch)

        ys = np.arange(
            y0,
            y1,
            dtype=np.float64,
        )

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

        eps = 1e-12
        z_safe = np.where(
            np.abs(z) > eps,
            z,
            np.where(z >= 0, eps, -eps),
        )

        mx = fx_r * (X_ref[:, 0] / z_safe) + cx_r
        my = fy_r * (X_ref[:, 1] / z_safe) + cy_r

        valid = (
            np.isfinite(mx)
            & np.isfinite(my)
            & np.isfinite(dep)
            & (dep > 0)
            & (z * float(z_sign) > float(z_min))
            & (mx >= 0)
            & (mx <= width - 1)
            & (my >= 0)
            & (my <= height - 1)
        )

        map_x[y0:y1] = mx.reshape(
            y1 - y0,
            width,
        ).astype(np.float32)

        map_y[y0:y1] = my.reshape(
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
    pred: np.ndarray,
    mask: np.ndarray,
    max_value: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)

    count = int(np.count_nonzero(mask))

    if count <= 0:
        return {
            "count": 0,
            "sse": 0.0,
            "mae": None,
            "mse": None,
            "psnr": None,
        }

    diff = (
        target.astype(np.float64)[mask]
        - pred.astype(np.float64)[mask]
    )

    sse = float(np.sum(diff * diff))
    mse = float(sse / count)
    mae = float(np.mean(np.abs(diff)))

    psnr = (
        999.0
        if mse <= 1e-30
        else float(
            10.0
            * math.log10(
                (float(max_value) ** 2) / mse
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
    target_rec: dict[str, Any],
    ref_rec: dict[str, Any],
    baseline_target_pose: tuple[np.ndarray, np.ndarray],
    baseline_ref_pose: tuple[np.ndarray, np.ndarray],
    quant_target_pose: tuple[np.ndarray, np.ndarray],
    quant_ref_pose: tuple[np.ndarray, np.ndarray],
    luma_reader: LumaReader,
    depth_reader: DepthReader,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_y = luma_reader.read(
        int(target_rec["frame_idx"])
    )
    ref_y = luma_reader.read(
        int(ref_rec["frame_idx"])
    )

    depth_target = depth_reader.read(
        int(target_rec["depth_frame_idx"]),
        int(target_rec["depth_source_gop_idx"]),
    )

    K_target = np.asarray(
        target_rec["_K_np"],
        dtype=np.float64,
    )
    K_ref = np.asarray(
        ref_rec["_K_np"],
        dtype=np.float64,
    )

    R_t_base, t_t_base = baseline_target_pose
    R_r_base, t_r_base = baseline_ref_pose

    R_t_quant, t_t_quant = quant_target_pose
    R_r_quant, t_r_quant = quant_ref_pose

    mx_base, my_base, valid_base = camera_map(
        width=args.width,
        height=args.height,
        K_target=K_target,
        K_ref=K_ref,
        R_target=R_t_base,
        t_target=t_t_base,
        R_ref=R_r_base,
        t_ref=t_r_base,
        depth_target=depth_target,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.row_batch,
    )

    mx_quant, my_quant, valid_quant = camera_map(
        width=args.width,
        height=args.height,
        K_target=K_target,
        K_ref=K_ref,
        R_target=R_t_quant,
        t_target=t_t_quant,
        R_ref=R_r_quant,
        t_ref=t_r_quant,
        depth_target=depth_target,
        z_sign=args.z_sign,
        z_min=args.z_min,
        row_batch=args.row_batch,
    )

    pred_base = warp_reference_to_target(
        ref_y,
        mx_base,
        my_base,
    )

    pred_quant = warp_reference_to_target(
        ref_y,
        mx_quant,
        my_quant,
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

    base_common = masked_error_stats(
        target_y,
        pred_base,
        valid_common,
        max_value,
    )

    quant_common = masked_error_stats(
        target_y,
        pred_quant,
        valid_common,
        max_value,
    )

    base_native = masked_error_stats(
        target_y,
        pred_base,
        valid_base,
        max_value,
    )

    quant_native = masked_error_stats(
        target_y,
        pred_quant,
        valid_quant,
        max_value,
    )

    psnr_drop = None
    if (
        base_common["psnr"] is not None
        and quant_common["psnr"] is not None
    ):
        psnr_drop = float(
            base_common["psnr"]
            - quant_common["psnr"]
        )

    return {
        "common_valid_count": int(base_common["count"]),
        "common_valid_ratio": float(
            base_common["count"]
            / (args.width * args.height)
        ),
        "baseline_common_sse": float(base_common["sse"]),
        "baseline_common_mse": base_common["mse"],
        "baseline_common_psnr": base_common["psnr"],
        "quantized_common_sse": float(quant_common["sse"]),
        "quantized_common_mse": quant_common["mse"],
        "quantized_common_psnr": quant_common["psnr"],
        "psnr_drop_common": psnr_drop,
        "baseline_native_valid_ratio": float(
            np.mean(valid_base)
        ),
        "baseline_native_psnr": base_native["psnr"],
        "quantized_native_valid_ratio": float(
            np.mean(valid_quant)
        ),
        "quantized_native_psnr": quant_native["psnr"],
    }


# ============================================================
# Qstep and CSV
# ============================================================

def parse_float_list(text: str) -> list[float]:
    vals: list[float] = []

    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue

        value = float(token)

        if value <= 0:
            raise ValueError(
                f"qstep must be positive: {value}"
            )

        vals.append(value)

    if not vals:
        raise ValueError("empty qstep list")

    return vals


def make_qstep_pairs(
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    rot_steps = (
        parse_float_list(args.rot_qsteps)
        if args.rot_qsteps
        else [float(args.rot_qstep)]
    )

    trans_steps = (
        parse_float_list(args.trans_qsteps)
        if args.trans_qsteps
        else [float(args.trans_qstep)]
    )

    if args.paired_qsteps:
        if len(rot_steps) != len(trans_steps):
            raise ValueError(
                "--paired-qsteps requires equal list lengths"
            )
        return list(zip(rot_steps, trans_steps))

    return [
        (r, t)
        for r in rot_steps
        for t in trans_steps
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "Estimate predictive pose Exp-Golomb bits and RA camera-warp "
            "PSNR degradation for pose qstep sweeps."
        )
    )

    ap.add_argument(
        "--input-jsonl",
        required=True,
    )
    ap.add_argument(
        "--sequence-yuv",
        required=True,
        help="Original sequence in YUV420p10le",
    )
    ap.add_argument(
        "--depth-yuv",
        required=True,
        help="Merged depth YUV420p10le produced by the GOP merge script",
    )
    ap.add_argument(
        "--width",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--height",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--output-prefix",
        default="",
    )

    ap.add_argument(
        "--rot-qstep",
        type=float,
        default=1e-5,
    )
    ap.add_argument(
        "--trans-qstep",
        type=float,
        default=1e-5,
    )
    ap.add_argument(
        "--rot-qsteps",
        default="",
    )
    ap.add_argument(
        "--trans-qsteps",
        default="",
    )
    ap.add_argument(
        "--paired-qsteps",
        action="store_true",
    )

    ap.add_argument(
        "--pair-source",
        choices=["dyadic", "adjacent"],
        default="dyadic",
    )
    ap.add_argument(
        "--pairs",
        default="",
        help=(
            "Explicit GOP-local target:ref pairs; when set, overrides "
            "--pair-source. Example: 16:0,16:32"
        ),
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
    ap.add_argument(
        "--z-sign",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--z-min",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--row-batch",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--valid-erode",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--first-frame-bits",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--per-gop-overhead-bits",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--per-frame-overhead-bits",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width/height")
    if args.row_batch <= 0:
        raise ValueError("--row-batch must be positive")
    if args.valid_erode < 0:
        raise ValueError("--valid-erode must be non-negative")

    input_jsonl = Path(
        args.input_jsonl
    ).expanduser().resolve()
    sequence_yuv = Path(
        args.sequence_yuv
    ).expanduser().resolve()
    depth_yuv = Path(
        args.depth_yuv
    ).expanduser().resolve()

    for path in [
        input_jsonl,
        sequence_yuv,
        depth_yuv,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.output_prefix:
        output_prefix = Path(
            args.output_prefix
        ).expanduser().resolve()
    else:
        stem = input_jsonl.with_suffix("")
        output_prefix = stem.parent / (
            stem.name + "_ra_pose_bits_warp"
        )

    header, frames = load_camera_jsonl(input_jsonl)
    groups = group_frames_by_gop(frames)
    scale_by_gop = build_depth_scale_by_gop(
        header,
        frames,
    )

    seq_frame_count = count_yuv420p10le_frames(
        sequence_yuv,
        args.width,
        args.height,
    )
    depth_frame_count = count_yuv420p10le_frames(
        depth_yuv,
        args.width,
        args.height,
    )

    max_seq_idx = max(int(r["frame_idx"]) for r in frames)
    max_depth_idx = max(int(r["depth_frame_idx"]) for r in frames)

    if max_seq_idx >= seq_frame_count:
        raise ValueError(
            f"sequence YUV has {seq_frame_count} frames but JSONL "
            f"requires frame_idx {max_seq_idx}"
        )

    if max_depth_idx >= depth_frame_count:
        raise ValueError(
            f"depth YUV has {depth_frame_count} frames but JSONL "
            f"requires depth_frame_idx {max_depth_idx}"
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
        scale_by_gop,
    )

    qstep_pairs = make_qstep_pairs(args)

    per_setting_rows: list[dict[str, Any]] = []
    per_gop_rows: list[dict[str, Any]] = []
    per_pair_rows: list[dict[str, Any]] = []
    per_frame_bit_rows: list[dict[str, Any]] = []
    settings_json: list[dict[str, Any]] = []

    # Baseline original poses indexed by GOP and local_poc.
    baseline_pose_by_gop: dict[
        int,
        dict[int, tuple[np.ndarray, np.ndarray]]
    ] = {}

    rec_by_gop_local: dict[
        int,
        dict[int, dict[str, Any]]
    ] = {}

    pairs_by_gop: dict[
        int,
        list[tuple[int, int, str]]
    ] = {}

    for gop_idx, records in groups.items():
        baseline_pose_by_gop[gop_idx] = {
            int(r["local_poc"]): (
                np.asarray(r["_R_np"], dtype=np.float64),
                np.asarray(r["_tvec_np"], dtype=np.float64),
            )
            for r in records
        }

        rec_by_gop_local[gop_idx] = {
            int(r["local_poc"]): r
            for r in records
        }

        pairs_by_gop[gop_idx] = build_pairs_for_gop(
            records,
            args,
        )

    for setting_idx, (rot_qstep, trans_qstep) in enumerate(
        qstep_pairs
    ):
        options = CodingOptions(
            rot_qstep=float(rot_qstep),
            trans_qstep=float(trans_qstep),
            first_frame_bits=int(args.first_frame_bits),
            per_gop_overhead_bits=int(args.per_gop_overhead_bits),
            per_frame_overhead_bits=int(args.per_frame_overhead_bits),
        )

        setting_gop_rows: list[dict[str, Any]] = []
        setting_pair_rows: list[dict[str, Any]] = []
        setting_frame_rows: list[dict[str, Any]] = []

        for gop_idx in sorted(groups):
            gop_summary, frame_bits, recon_pose = simulate_pose_coding(
                gop_idx,
                groups[gop_idx],
                options,
            )

            pair_rows_this_gop: list[dict[str, Any]] = []

            for pair_idx, (
                target_local,
                ref_local,
                pair_kind,
            ) in enumerate(pairs_by_gop[gop_idx]):
                target_rec = rec_by_gop_local[gop_idx][target_local]
                ref_rec = rec_by_gop_local[gop_idx][ref_local]

                metrics = evaluate_pair(
                    target_rec=target_rec,
                    ref_rec=ref_rec,
                    baseline_target_pose=baseline_pose_by_gop[
                        gop_idx
                    ][target_local],
                    baseline_ref_pose=baseline_pose_by_gop[
                        gop_idx
                    ][ref_local],
                    quant_target_pose=recon_pose[target_local],
                    quant_ref_pose=recon_pose[ref_local],
                    luma_reader=luma_reader,
                    depth_reader=depth_reader,
                    args=args,
                )

                row = {
                    "setting_idx": int(setting_idx),
                    "rot_qstep": float(rot_qstep),
                    "trans_qstep": float(trans_qstep),
                    "gop_idx": int(gop_idx),
                    "gop_name": str(
                        groups[gop_idx][0]["gop_name"]
                    ),
                    "pair_idx": int(pair_idx),
                    "pair_kind": pair_kind,
                    "target_local_poc": int(target_local),
                    "ref_local_poc": int(ref_local),
                    "target_poc": int(target_rec["poc"]),
                    "ref_poc": int(ref_rec["poc"]),
                    "target_frame_idx": int(
                        target_rec["frame_idx"]
                    ),
                    "ref_frame_idx": int(
                        ref_rec["frame_idx"]
                    ),
                    **metrics,
                }

                pair_rows_this_gop.append(row)
                setting_pair_rows.append(row)

            valid_pair_rows = [
                r
                for r in pair_rows_this_gop
                if r["baseline_common_psnr"] is not None
                and r["quantized_common_psnr"] is not None
                and r["common_valid_count"] > 0
            ]

            if valid_pair_rows:
                mean_base_psnr = float(
                    np.mean(
                        [
                            r["baseline_common_psnr"]
                            for r in valid_pair_rows
                        ]
                    )
                )

                mean_quant_psnr = float(
                    np.mean(
                        [
                            r["quantized_common_psnr"]
                            for r in valid_pair_rows
                        ]
                    )
                )

                mean_drop = float(
                    np.mean(
                        [
                            r["psnr_drop_common"]
                            for r in valid_pair_rows
                        ]
                    )
                )

                pooled_count = int(
                    sum(
                        r["common_valid_count"]
                        for r in valid_pair_rows
                    )
                )

                pooled_base_sse = float(
                    sum(
                        r["baseline_common_sse"]
                        for r in valid_pair_rows
                    )
                )

                pooled_quant_sse = float(
                    sum(
                        r["quantized_common_sse"]
                        for r in valid_pair_rows
                    )
                )

                maxv2 = float(
                    ((1 << args.bitdepth) - 1) ** 2
                )

                pooled_base_mse = (
                    pooled_base_sse / pooled_count
                )
                pooled_quant_mse = (
                    pooled_quant_sse / pooled_count
                )

                pooled_base_psnr = float(
                    10.0
                    * math.log10(
                        maxv2 / max(pooled_base_mse, 1e-30)
                    )
                )

                pooled_quant_psnr = float(
                    10.0
                    * math.log10(
                        maxv2 / max(pooled_quant_mse, 1e-30)
                    )
                )

                pooled_drop = float(
                    pooled_base_psnr - pooled_quant_psnr
                )
            else:
                mean_base_psnr = None
                mean_quant_psnr = None
                mean_drop = None
                pooled_count = 0
                pooled_base_psnr = None
                pooled_quant_psnr = None
                pooled_drop = None

            gop_summary.update(
                {
                    "setting_idx": int(setting_idx),
                    "pair_count": int(len(pair_rows_this_gop)),
                    "valid_pair_count": int(
                        len(valid_pair_rows)
                    ),
                    "mean_baseline_common_psnr": mean_base_psnr,
                    "mean_quantized_common_psnr": mean_quant_psnr,
                    "mean_psnr_drop_common": mean_drop,
                    "pooled_common_valid_count": int(
                        pooled_count
                    ),
                    "pooled_baseline_psnr": pooled_base_psnr,
                    "pooled_quantized_psnr": pooled_quant_psnr,
                    "pooled_psnr_drop": pooled_drop,
                }
            )

            setting_gop_rows.append(gop_summary)
            setting_frame_rows.extend(frame_bits)

        valid_setting_pairs = [
            r
            for r in setting_pair_rows
            if r["baseline_common_psnr"] is not None
            and r["quantized_common_psnr"] is not None
            and r["common_valid_count"] > 0
        ]

        total_bits = int(
            sum(r["total_bits"] for r in setting_gop_rows)
        )
        total_rotation_bits = int(
            sum(r["rotation_bits"] for r in setting_gop_rows)
        )
        total_translation_bits = int(
            sum(r["translation_bits"] for r in setting_gop_rows)
        )
        total_overhead_bits = int(
            sum(r["overhead_bits"] for r in setting_gop_rows)
        )

        total_camera_records = int(
            sum(r["frame_count"] for r in setting_gop_rows)
        )
        total_coded_frames = int(
            sum(r["coded_frame_count"] for r in setting_gop_rows)
        )

        if valid_setting_pairs:
            mean_base_psnr = float(
                np.mean(
                    [
                        r["baseline_common_psnr"]
                        for r in valid_setting_pairs
                    ]
                )
            )

            mean_quant_psnr = float(
                np.mean(
                    [
                        r["quantized_common_psnr"]
                        for r in valid_setting_pairs
                    ]
                )
            )

            mean_drop = float(
                np.mean(
                    [
                        r["psnr_drop_common"]
                        for r in valid_setting_pairs
                    ]
                )
            )

            median_drop = float(
                np.median(
                    [
                        r["psnr_drop_common"]
                        for r in valid_setting_pairs
                    ]
                )
            )

            max_drop = float(
                np.max(
                    [
                        r["psnr_drop_common"]
                        for r in valid_setting_pairs
                    ]
                )
            )

            pooled_count = int(
                sum(
                    r["common_valid_count"]
                    for r in valid_setting_pairs
                )
            )

            pooled_base_sse = float(
                sum(
                    r["baseline_common_sse"]
                    for r in valid_setting_pairs
                )
            )

            pooled_quant_sse = float(
                sum(
                    r["quantized_common_sse"]
                    for r in valid_setting_pairs
                )
            )

            maxv2 = float(
                ((1 << args.bitdepth) - 1) ** 2
            )

            pooled_base_mse = (
                pooled_base_sse / pooled_count
            )
            pooled_quant_mse = (
                pooled_quant_sse / pooled_count
            )

            pooled_base_psnr = float(
                10.0
                * math.log10(
                    maxv2 / max(pooled_base_mse, 1e-30)
                )
            )

            pooled_quant_psnr = float(
                10.0
                * math.log10(
                    maxv2 / max(pooled_quant_mse, 1e-30)
                )
            )

            pooled_drop = float(
                pooled_base_psnr - pooled_quant_psnr
            )
        else:
            mean_base_psnr = None
            mean_quant_psnr = None
            mean_drop = None
            median_drop = None
            max_drop = None
            pooled_count = 0
            pooled_base_psnr = None
            pooled_quant_psnr = None
            pooled_drop = None

        setting_summary = {
            "setting_idx": int(setting_idx),
            "rot_qstep": float(rot_qstep),
            "trans_qstep": float(trans_qstep),
            "gop_count": int(len(setting_gop_rows)),
            "pair_count": int(len(setting_pair_rows)),
            "valid_pair_count": int(
                len(valid_setting_pairs)
            ),
            "camera_record_count": int(
                total_camera_records
            ),
            "coded_frame_count": int(
                total_coded_frames
            ),
            "rotation_bits": int(
                total_rotation_bits
            ),
            "translation_bits": int(
                total_translation_bits
            ),
            "overhead_bits": int(
                total_overhead_bits
            ),
            "total_bits": int(total_bits),
            "total_bytes_ceil": int(
                (total_bits + 7) // 8
            ),
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
            "mean_baseline_common_psnr": (
                mean_base_psnr
            ),
            "mean_quantized_common_psnr": (
                mean_quant_psnr
            ),
            "mean_psnr_drop_common": mean_drop,
            "median_psnr_drop_common": median_drop,
            "max_psnr_drop_common": max_drop,
            "pooled_common_valid_count": int(
                pooled_count
            ),
            "pooled_baseline_psnr": (
                pooled_base_psnr
            ),
            "pooled_quantized_psnr": (
                pooled_quant_psnr
            ),
            "pooled_psnr_drop": pooled_drop,
        }

        per_setting_rows.append(setting_summary)
        per_gop_rows.extend(setting_gop_rows)
        per_pair_rows.extend(setting_pair_rows)
        per_frame_bit_rows.extend(setting_frame_rows)

        settings_json.append(
            {
                "summary": setting_summary,
                "per_gop": setting_gop_rows,
            }
        )

        print("=" * 80)
        print(
            f"setting {setting_idx}: "
            f"rot_qstep={rot_qstep:.9g}, "
            f"trans_qstep={trans_qstep:.9g}"
        )
        print(
            f"  pose bits              : {total_bits} "
            f"({(total_bits + 7) // 8} bytes ceil)"
        )
        print(
            f"  bits/coded frame       : "
            f"{setting_summary['bits_per_coded_frame']:.4f}"
        )
        print(
            f"  mean baseline PSNR     : "
            f"{mean_base_psnr}"
        )
        print(
            f"  mean quantized PSNR    : "
            f"{mean_quant_psnr}"
        )
        print(
            f"  mean PSNR drop         : "
            f"{mean_drop}"
        )
        print(
            f"  pooled baseline PSNR   : "
            f"{pooled_base_psnr}"
        )
        print(
            f"  pooled quantized PSNR  : "
            f"{pooled_quant_psnr}"
        )
        print(
            f"  pooled PSNR drop       : "
            f"{pooled_drop}"
        )

    result = {
        "input_jsonl": str(input_jsonl),
        "sequence_yuv": str(sequence_yuv),
        "depth_yuv": str(depth_yuv),
        "width": int(args.width),
        "height": int(args.height),
        "sequence_frame_count": int(
            seq_frame_count
        ),
        "depth_frame_count": int(
            depth_frame_count
        ),
        "input_header": header,
        "depth_scale_by_gop": {
            str(k): float(v)
            for k, v in sorted(scale_by_gop.items())
        },
        "coding_model": {
            "anchor": (
                "Each GOP local_poc 0 is implicit R=I,t=0"
            ),
            "rotation_predictor": (
                "previous quantized reconstructed rotation"
            ),
            "rotation_residual": (
                "Rodrigues(R_true @ R_rec_prev.T)"
            ),
            "translation_predictor": (
                "previous quantized reconstructed local absolute tvec"
            ),
            "translation_residual": (
                "t_true - t_rec_prev"
            ),
            "integer_code": (
                "signed Exp-Golomb order-0 independently for 6 components"
            ),
        },
        "warp_model": {
            "pair_source": (
                "explicit" if args.pairs else args.pair_source
            ),
            "bidirectional": bool(
                not args.no_bidirectional_pairs
            ),
            "relative_pose": (
                "R_rel=R_ref@R_target.T; "
                "t_rel=t_ref-R_rel@t_target"
            ),
            "depth": (
                "target merged-depth frame decoded with its owner GOP scale"
            ),
            "metric": (
                "baseline and quantized warps compared on their common "
                "valid mask; PSNR drop=baseline_common-quantized_common"
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

    write_csv(
        setting_csv_path,
        per_setting_rows,
    )
    write_csv(
        gop_csv_path,
        per_gop_rows,
    )
    write_csv(
        pair_csv_path,
        per_pair_rows,
    )
    write_csv(
        frame_csv_path,
        per_frame_bit_rows,
    )

    print("=" * 80)
    print(f"summary JSON      : {summary_path}")
    print(f"per-setting CSV   : {setting_csv_path}")
    print(f"per-GOP CSV       : {gop_csv_path}")
    print(f"per-pair CSV      : {pair_csv_path}")
    print(f"per-frame bits CSV: {frame_csv_path}")


if __name__ == "__main__":
    main()
