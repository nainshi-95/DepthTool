#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera/depth metadata stream simulation for hierarchical RA and LDB.

Purpose
-------
This script simulates the proposed signaling without writing a real bitstream.
It performs four stages:

1. Encoder-side tool ON/OFF decision in codec coding order.
   - RA: hierarchical coding order, nearest eligible lower-TID past/future refs.
   - LDB: display/coding order, nearest and second-nearest eligible past refs.
   - A picture whose previous-group pose is OFF is excluded from later
     camera-projection references in that geometry group.
   - Reset anchors are always available in their new geometry group.

2. Encoder-side serialization into a numeric CSV stream.
   - Fixed depth/camera reset period: 32 pictures.
   - Records are carried by the currently decoded picture, but target picture
     records are serialized in display order.
   - At a reset boundary POC B, the previous group endpoint record for B is
     written first, then the new group K(B) reset record is written.
   - Tool-OFF pictures omit Rt.
   - Tool-ON Rt is coded as a quantized relative transform from the last
     Tool-ON picture in the same 32-picture geometry group.

3. Decoder-side parsing of that CSV stream.
   - A single last_loaded_poc pointer prevents duplicate loading.
   - Relative Rt is reconstructed into the same group-relative absolute Rt.

4. Encoder/decoder consistency checks.
   - Flags, quantized deltas, reconstructed Rt, K, load pointers, current-pose
     availability, and selected-reference availability are checked.
   - Any mismatch is written to *_mismatch.csv and causes a non-zero exit code.

Input
-----
The merged camera JSONL and depth YUV420p10le used by the previous simulation.
Overlapping GOP boundary camera records are supported.

Important model choice
----------------------
A reset-boundary picture has two geometry roles:

  previous group endpoint: previous-group K + optional Rt + Tool flag
  next group anchor       : next-group K + identity Rt, always available

Therefore POC 32 can be OFF as endpoint of group [0, 32] while still being the
mandatory identity anchor of group [32, 64].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


RESET_PERIOD = 32
EPS = 1e-10


# ============================================================
# Data classes
# ============================================================

@dataclass
class CameraFrame:
    gop_idx: int
    gop_name: str
    local_poc: int
    poc: int
    depth_frame_idx: int
    rvec: np.ndarray
    tvec: np.ndarray
    K: np.ndarray
    z_sign: float
    depth_scale_real: float


@dataclass
class SequenceFrame:
    poc: int
    depth_frame_idx: int
    depth_scale_real: float
    z_sign: float
    K_source: np.ndarray
    T_global: np.ndarray  # camera_poc_from_camera_0


@dataclass
class RefEval:
    ref_poc: int
    direction: str
    valid_points: int
    avg_mv_px: Optional[float]
    motion4096: Optional[float]
    points: List[dict] = field(default_factory=list)


@dataclass
class Decision:
    poc: int
    tid: int
    coding_order: int
    is_intra: bool
    is_reset_boundary: bool
    previous_group_base: Optional[int]
    tool_enabled: bool
    reference_eligible: bool
    final_motion4096: Optional[float]
    reason: str
    ref0: Optional[RefEval] = None
    ref1: Optional[RefEval] = None


@dataclass
class EncodedPoseState:
    group_base: int
    target_poc: int
    tool_enabled: bool
    last_on_before: int
    q_rx: Optional[int] = None
    q_ry: Optional[int] = None
    q_rz: Optional[int] = None
    q_tx: Optional[int] = None
    q_ty: Optional[int] = None
    q_tz: Optional[int] = None
    delta_recon: Optional[np.ndarray] = None
    pose_recon: Optional[np.ndarray] = None


@dataclass
class StreamRecord:
    seq: int
    coding_order: int
    carrier_poc: int
    record_type: str
    target_poc: Optional[int] = None
    group_base: Optional[int] = None
    tool_enabled: Optional[int] = None
    last_on_before: Optional[int] = None
    q_rx: Optional[int] = None
    q_ry: Optional[int] = None
    q_rz: Optional[int] = None
    q_tx: Optional[int] = None
    q_ty: Optional[int] = None
    q_tz: Optional[int] = None
    q_fx: Optional[int] = None
    q_fy: Optional[int] = None
    q_cx: Optional[int] = None
    q_cy: Optional[int] = None
    z_sign: Optional[float] = None
    note: str = ""


@dataclass
class DecoderPoseState:
    group_base: int
    target_poc: int
    tool_enabled: bool
    last_on_before: int
    q_values: Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
    pose_recon: Optional[np.ndarray]


# ============================================================
# Small helpers
# ============================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(v: Optional[float]) -> str | float:
    return "" if v is None else float(v)


def safe_int(v: Optional[int]) -> str | int:
    return "" if v is None else int(v)


def previous_group_base(poc: int) -> Optional[int]:
    """Group that owns the endpoint representation of POC.

    POC 32 belongs to previous group 0 for its optional endpoint Rt record,
    while also becoming the mandatory identity anchor of new group 32.
    """
    if poc <= 0:
        return None
    return ((poc - 1) // RESET_PERIOD) * RESET_PERIOD


def current_group_base(poc: int) -> int:
    return (poc // RESET_PERIOD) * RESET_PERIOD


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def T_to_rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rvec = cv2.Rodrigues(np.asarray(T[:3, :3], np.float64))[0].reshape(3)
    tvec = np.asarray(T[:3, 3], np.float64).reshape(3)
    return rvec, tvec


def compose_relative(target_global: np.ndarray, ref_global: np.ndarray) -> np.ndarray:
    """Return camera_target_from_camera_ref."""
    return target_global @ np.linalg.inv(ref_global)


def matrix_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def quantize_scalar(value: float, qstep: float) -> int:
    if qstep <= 0:
        raise ValueError("qstep must be positive")
    return int(np.rint(float(value) / qstep))


def dequantize_scalar(qvalue: int, qstep: float) -> float:
    return float(qvalue) * qstep


def quantize_transform(
    T: np.ndarray,
    rot_qstep: float,
    trans_qstep: float,
) -> Tuple[Tuple[int, int, int, int, int, int], np.ndarray]:
    rvec, tvec = T_to_rt(T)
    q = (
        quantize_scalar(rvec[0], rot_qstep),
        quantize_scalar(rvec[1], rot_qstep),
        quantize_scalar(rvec[2], rot_qstep),
        quantize_scalar(tvec[0], trans_qstep),
        quantize_scalar(tvec[1], trans_qstep),
        quantize_scalar(tvec[2], trans_qstep),
    )
    rq = np.array(
        [
            dequantize_scalar(q[0], rot_qstep),
            dequantize_scalar(q[1], rot_qstep),
            dequantize_scalar(q[2], rot_qstep),
        ],
        dtype=np.float64,
    )
    tq = np.array(
        [
            dequantize_scalar(q[3], trans_qstep),
            dequantize_scalar(q[4], trans_qstep),
            dequantize_scalar(q[5], trans_qstep),
        ],
        dtype=np.float64,
    )
    return q, rt_to_T(rq, tq)


def quantize_K(K: np.ndarray, qstep: float) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    values = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    q = tuple(quantize_scalar(v, qstep) for v in values)
    Kq = np.array(
        [
            [dequantize_scalar(q[0], qstep), 0.0, dequantize_scalar(q[2], qstep)],
            [0.0, dequantize_scalar(q[1], qstep), dequantize_scalar(q[3], qstep)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (int(q[0]), int(q[1]), int(q[2]), int(q[3])), Kq


# ============================================================
# Camera JSONL and global pose reconstruction
# ============================================================

def read_camera_jsonl(path: Path) -> Tuple[dict, List[CameraFrame]]:
    header = None
    frames: List[CameraFrame] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if obj.get("type") == "header":
                if header is not None:
                    raise ValueError(f"{path}: duplicate header at line {line_no}")
                header = obj
                continue
            if obj.get("type") != "frame":
                continue

            intr = obj["intrinsic"]
            K = np.array(
                [
                    [float(intr["fx"]), 0.0, float(intr["cx"])],
                    [0.0, float(intr["fy"]), float(intr["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            frames.append(
                CameraFrame(
                    gop_idx=int(obj["gop_idx"]),
                    gop_name=str(obj.get("gop_name", f"gop{obj['gop_idx']}")),
                    local_poc=int(obj["local_poc"]),
                    poc=int(obj["poc"]),
                    depth_frame_idx=int(obj["depth_frame_idx"]),
                    rvec=np.asarray(obj["rvec"], np.float64).reshape(3),
                    tvec=np.asarray(obj["tvec"], np.float64).reshape(3),
                    K=K,
                    z_sign=float(intr.get("z_sign", 1.0)),
                    depth_scale_real=float(obj["depth_scale_real"]),
                )
            )

    if header is None:
        raise ValueError(f"{path}: header not found")
    if not frames:
        raise ValueError(f"{path}: no frame records")
    return header, frames


def reconstruct_local_gop_poses(
    frames: Sequence[CameraFrame],
    pose_mode: str,
) -> Dict[int, np.ndarray]:
    """Return camera_localPoc_from_camera_groupStart for one source GOP."""
    ordered = sorted(frames, key=lambda x: x.local_poc)
    raw = {f.local_poc: rt_to_T(f.rvec, f.tvec) for f in ordered}
    out: Dict[int, np.ndarray] = {}

    if pose_mode == "current_to_previous":
        first = ordered[0].local_poc
        out[first] = np.eye(4, dtype=np.float64)
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            if cur.local_poc != prev.local_poc + 1:
                raise ValueError(
                    f"source GOP {cur.gop_idx}: non-contiguous local POC "
                    f"{prev.local_poc}->{cur.local_poc}"
                )
            # raw[cur] = camera_(cur-1)_from_camera_cur
            out[cur.local_poc] = np.linalg.inv(raw[cur.local_poc]) @ out[prev.local_poc]

    elif pose_mode == "gop_local":
        first = ordered[0].local_poc
        inv_first = np.linalg.inv(raw[first])
        for f in ordered:
            out[f.local_poc] = raw[f.local_poc] @ inv_first

    elif pose_mode == "absolute":
        first = ordered[0].local_poc
        inv_first = np.linalg.inv(raw[first])
        for f in ordered:
            out[f.local_poc] = raw[f.local_poc] @ inv_first

    else:
        raise ValueError(f"unsupported pose mode: {pose_mode}")

    return out


def build_sequence_frames(
    source_frames: Sequence[CameraFrame],
    pose_mode: str,
    overlap_tolerance: float,
) -> Tuple[Dict[int, SequenceFrame], List[str]]:
    """Chain source-GOP local poses into one global pose sequence.

    The merged JSONL may contain the same absolute boundary POC twice. The
    source GOP starting at that POC is preferred for K_source, while the global
    pose is checked against the previous source GOP endpoint.
    """
    grouped: Dict[int, List[CameraFrame]] = {}
    for f in source_frames:
        grouped.setdefault(f.gop_idx, []).append(f)

    groups = sorted(grouped.items(), key=lambda kv: min(x.poc for x in kv[1]))
    global_pose: Dict[int, np.ndarray] = {}
    preferred: Dict[int, CameraFrame] = {}
    warnings: List[str] = []

    for gop_idx, gframes in groups:
        ordered = sorted(gframes, key=lambda f: f.local_poc)
        local = reconstruct_local_gop_poses(ordered, pose_mode)
        base_poc = min(f.poc for f in ordered)

        if base_poc in global_pose:
            group_base_global = global_pose[base_poc]
        elif not global_pose:
            group_base_global = np.eye(4, dtype=np.float64)
        else:
            known = sorted(global_pose)
            raise ValueError(
                f"source GOP {gop_idx} starts at POC {base_poc}, but no overlap "
                f"with reconstructed sequence ending at POC {known[-1]}"
            )

        for f in ordered:
            candidate = local[f.local_poc] @ group_base_global
            if f.poc in global_pose:
                err = matrix_error(candidate, global_pose[f.poc])
                if err > overlap_tolerance:
                    warnings.append(
                        f"overlap POC {f.poc}: global-pose disagreement "
                        f"{err:.6e} > {overlap_tolerance:.6e}; existing pose kept"
                    )
            else:
                global_pose[f.poc] = candidate

            old = preferred.get(f.poc)
            # local_poc==0 is preferred because it carries the next-group K.
            if old is None or (f.local_poc == 0 and old.local_poc != 0):
                preferred[f.poc] = f

    pocs = sorted(global_pose)
    if not pocs or pocs[0] != 0:
        raise ValueError("sequence must contain POC 0")
    missing = [p for p in range(pocs[-1] + 1) if p not in global_pose]
    if missing:
        preview = ",".join(map(str, missing[:16]))
        raise ValueError(f"non-contiguous absolute POCs; missing: {preview}")

    out: Dict[int, SequenceFrame] = {}
    for poc in pocs:
        f = preferred[poc]
        out[poc] = SequenceFrame(
            poc=poc,
            depth_frame_idx=f.depth_frame_idx,
            depth_scale_real=f.depth_scale_real,
            z_sign=f.z_sign,
            K_source=f.K.copy(),
            T_global=global_pose[poc].copy(),
        )

    return out, warnings


def build_reset_K_bank(frames: Dict[int, SequenceFrame]) -> Dict[int, np.ndarray]:
    max_poc = max(frames)
    bank: Dict[int, np.ndarray] = {}
    for base in range(0, max_poc + 1, RESET_PERIOD):
        if base not in frames:
            raise ValueError(f"reset base POC {base} missing from camera sequence")
        bank[base] = frames[base].K_source.copy()
    return bank


# ============================================================
# Depth YUV reader
# ============================================================

class DepthYUV420P10LE:
    def __init__(self, path: Path, width: int, height: int):
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("YUV420 requires positive even width/height")

        self.path = path
        self.width = width
        self.height = height
        self.y_samples = width * height
        self.uv_samples = (width // 2) * (height // 2)
        self.frame_bytes = (self.y_samples + 2 * self.uv_samples) * 2

        size = path.stat().st_size
        if size % self.frame_bytes:
            raise ValueError(
                f"{path}: invalid file size {size} for frame size {self.frame_bytes}"
            )
        self.frame_count = size // self.frame_bytes
        self.fp = path.open("rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.fp.close()

    def read_y(self, frame_idx: int) -> np.ndarray:
        if not (0 <= frame_idx < self.frame_count):
            raise IndexError(f"depth frame {frame_idx} outside [0,{self.frame_count})")
        self.fp.seek(frame_idx * self.frame_bytes)
        raw = self.fp.read(self.y_samples * 2)
        if len(raw) != self.y_samples * 2:
            raise EOFError(f"short Y read at frame {frame_idx}")
        return np.frombuffer(raw, dtype="<u2").reshape(self.height, self.width)


# ============================================================
# Coding order and temporal IDs
# ============================================================

def assign_interval_tids(start: int, end: int) -> Dict[int, int]:
    """Hierarchical TID for integer POCs in [start, end]."""
    pocs = list(range(start, end + 1))
    tid: Dict[int, int] = {start: 0, end: 0}

    def rec(lo_idx: int, hi_idx: int, level: int) -> None:
        if hi_idx - lo_idx <= 1:
            return
        mid_idx = (lo_idx + hi_idx) // 2
        mid_poc = pocs[mid_idx]
        tid[mid_poc] = min(tid.get(mid_poc, level), level)
        rec(lo_idx, mid_idx, level + 1)
        rec(mid_idx, hi_idx, level + 1)

    rec(0, len(pocs) - 1, 1)
    fallback = max(tid.values(), default=0) + 1
    for p in pocs:
        tid.setdefault(p, fallback)
    return tid


def build_coding_schedule(
    max_poc: int,
    mode: str,
    gop_size: int,
) -> Tuple[List[int], Dict[int, int]]:
    if mode == "ldb":
        return list(range(max_poc + 1)), {p: 0 for p in range(max_poc + 1)}

    if mode != "ra":
        raise ValueError(mode)

    order: List[int] = [0]
    tids: Dict[int, int] = {0: 0}
    start = 0
    while start < max_poc:
        end = min(start + gop_size, max_poc)
        local_tid = assign_interval_tids(start, end)
        for p, t in local_tid.items():
            tids[p] = min(tids.get(p, t), t)

        interval_pocs = list(range(start + 1, end + 1))
        interval_pocs.sort(key=lambda p: (local_tid[p], p))
        order.extend(interval_pocs)
        start = end

    if sorted(order) != list(range(max_poc + 1)):
        raise AssertionError("coding schedule does not contain every POC exactly once")
    return order, tids


# ============================================================
# Projection and motion evaluation
# ============================================================

def sample_points(width: int, height: int, inset: int) -> List[Tuple[str, int, int]]:
    x0 = min(max(0, inset), width - 1)
    y0 = min(max(0, inset), height - 1)
    x1 = max(0, width - 1 - inset)
    y1 = max(0, height - 1 - inset)
    return [
        ("top_left", x0, y0),
        ("top_right", x1, y0),
        ("bottom_left", x0, y1),
        ("bottom_right", x1, y1),
        ("center", (width - 1) // 2, (height - 1) // 2),
    ]


def project_target_to_reference(
    x: float,
    y: float,
    depth: float,
    target: SequenceFrame,
    reference: SequenceFrame,
    K_target: np.ndarray,
    K_reference: np.ndarray,
) -> Optional[Tuple[float, float]]:
    if not np.isfinite(depth) or depth <= 0:
        return None

    ray = np.linalg.inv(K_target) @ np.array([x, y, 1.0], dtype=np.float64)
    X_target = ray * (depth * target.z_sign)

    T_ref_from_target = reference.T_global @ np.linalg.inv(target.T_global)
    X_ref = T_ref_from_target[:3, :3] @ X_target + T_ref_from_target[:3, 3]

    z = float(X_ref[2])
    if not np.isfinite(z) or abs(z) < 1e-12:
        return None
    if z * reference.z_sign <= 0:
        return None

    q = K_reference @ X_ref
    if abs(float(q[2])) < 1e-12:
        return None
    u = float(q[0] / q[2])
    v = float(q[1] / q[2])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return u, v


def evaluate_reference(
    target_poc: int,
    ref_poc: int,
    direction: str,
    frames: Dict[int, SequenceFrame],
    K_bank: Dict[int, np.ndarray],
    depth_code: np.ndarray,
    width: int,
    height: int,
    inset: int,
) -> RefEval:
    target = frames[target_poc]
    reference = frames[ref_poc]
    target_base = current_group_base(target_poc)
    ref_base = current_group_base(ref_poc)
    K_target = K_bank[target_base]
    K_reference = K_bank[ref_base]

    motions: List[float] = []
    points: List[dict] = []
    for name, x, y in sample_points(width, height, inset):
        code = int(depth_code[y, x])
        depth = code * target.depth_scale_real
        projected = project_target_to_reference(
            float(x), float(y), float(depth), target, reference, K_target, K_reference
        )
        if projected is None:
            points.append(
                {
                    "name": name,
                    "valid": False,
                    "x": x,
                    "y": y,
                    "depth": float(depth),
                    "ref_x": None,
                    "ref_y": None,
                    "mv_px": None,
                }
            )
            continue

        u, v = projected
        mv_px = math.hypot(u - x, v - y)
        motions.append(mv_px)
        points.append(
            {
                "name": name,
                "valid": True,
                "x": x,
                "y": y,
                "depth": float(depth),
                "ref_x": u,
                "ref_y": v,
                "mv_px": mv_px,
            }
        )

    if motions:
        avg_mv_px = float(np.mean(np.asarray(motions, np.float64)))
        motion4096 = avg_mv_px / ((width + height) / 2.0) * 4096.0
    else:
        avg_mv_px = None
        motion4096 = None

    return RefEval(
        ref_poc=ref_poc,
        direction=direction,
        valid_points=len(motions),
        avg_mv_px=avg_mv_px,
        motion4096=motion4096,
        points=points,
    )


def combine_motion(
    evals: Sequence[Optional[RefEval]],
    min_valid_points: int,
    mode: str,
) -> Tuple[Optional[float], str]:
    values = [
        ev.motion4096
        for ev in evals
        if ev is not None
        and ev.valid_points >= min_valid_points
        and ev.motion4096 is not None
    ]
    if not values:
        return None, "no_valid_reference"
    vals = [float(v) for v in values]
    if mode == "min":
        return min(vals), "min_reference_motion"
    if mode == "mean":
        return float(np.mean(vals)), "mean_reference_motion"
    if mode == "max":
        return max(vals), "max_reference_motion"
    raise ValueError(mode)


# ============================================================
# Encoder decision simulation
# ============================================================

def group_range_for_decision(poc: int) -> Tuple[int, int]:
    base = previous_group_base(poc)
    if base is None:
        return 0, 0
    return base, base + RESET_PERIOD


def processed_ref_is_eligible(
    current_poc: int,
    ref_poc: int,
    decisions: Dict[int, Decision],
) -> bool:
    base, end = group_range_for_decision(current_poc)
    if not (base <= ref_poc <= end):
        return False
    if ref_poc == base:
        # New-group reset anchor: K + identity Rt is mandatory and independent
        # of the previous-group endpoint flag of the same picture.
        return True
    d = decisions.get(ref_poc)
    return bool(d is not None and d.reference_eligible)


def choose_ra_refs(
    current_poc: int,
    current_tid: int,
    tids: Dict[int, int],
    decisions: Dict[int, Decision],
) -> Tuple[Optional[int], Optional[int]]:
    base, end = group_range_for_decision(current_poc)
    processed = set(decisions)

    # The reset anchor is conceptually available once its picture has appeared.
    # In this generated schedule that condition is always true before children.
    candidates: List[int] = []
    if base < current_poc:
        candidates.append(base)
    candidates.extend(p for p in processed if base < p <= end and p != current_poc)

    eligible: List[int] = []
    for p in sorted(set(candidates)):
        if not processed_ref_is_eligible(current_poc, p, decisions):
            continue
        # Endpoint TID0 is tested against its reset anchor as a special case.
        if current_tid == 0 and current_poc == end:
            if p < current_poc:
                eligible.append(p)
            continue
        if tids.get(p, 0) < current_tid:
            eligible.append(p)

    past = [p for p in eligible if p < current_poc]
    future = [p for p in eligible if p > current_poc]
    past_ref = min(past, key=lambda p: (current_poc - p, -p)) if past else None
    future_ref = min(future, key=lambda p: (p - current_poc, p)) if future else None
    return past_ref, future_ref


def choose_ldb_refs(
    current_poc: int,
    decisions: Dict[int, Decision],
) -> Tuple[Optional[int], Optional[int]]:
    base, _ = group_range_for_decision(current_poc)
    candidates = []
    if base < current_poc:
        candidates.append(base)
    candidates.extend(p for p in decisions if base < p < current_poc)
    eligible = [
        p
        for p in sorted(set(candidates), reverse=True)
        if processed_ref_is_eligible(current_poc, p, decisions)
    ]
    ref0 = eligible[0] if len(eligible) >= 1 else None
    ref1 = eligible[1] if len(eligible) >= 2 else None
    return ref0, ref1


def simulate_decisions(
    frames: Dict[int, SequenceFrame],
    K_bank: Dict[int, np.ndarray],
    depth_reader: DepthYUV420P10LE,
    width: int,
    height: int,
    mode: str,
    coding_order: Sequence[int],
    tids: Dict[int, int],
    intra_period: int,
    threshold: float,
    ref_combine: str,
    min_valid_points: int,
    corner_inset: int,
    prune_unused_intra: bool,
) -> List[Decision]:
    decisions: Dict[int, Decision] = {}

    for coding_idx, poc in enumerate(coding_order):
        tid = tids[poc]
        is_intra = (poc % intra_period) == 0
        is_reset = (poc % RESET_PERIOD) == 0
        prev_base = previous_group_base(poc)

        if poc == 0:
            decisions[poc] = Decision(
                poc=poc,
                tid=tid,
                coding_order=coding_idx,
                is_intra=True,
                is_reset_boundary=True,
                previous_group_base=None,
                tool_enabled=True,
                reference_eligible=True,
                final_motion4096=None,
                reason="initial_reset_anchor",
            )
            continue

        # Intra pictures do not use CamProj for themselves. Their previous-group
        # endpoint pose is provisionally kept only so enabled leading pictures
        # may reference them; it can be pruned later when unused.
        if is_intra:
            decisions[poc] = Decision(
                poc=poc,
                tid=tid,
                coding_order=coding_idx,
                is_intra=True,
                is_reset_boundary=is_reset,
                previous_group_base=prev_base,
                tool_enabled=True,
                reference_eligible=True,
                final_motion4096=None,
                reason="intra_previous_group_pose_provisional_on",
            )
            continue

        if mode == "ra":
            ref0_poc, ref1_poc = choose_ra_refs(poc, tid, tids, decisions)
            dir0, dir1 = "past", "future"
        else:
            ref0_poc, ref1_poc = choose_ldb_refs(poc, decisions)
            dir0, dir1 = "l0_past", "l1_past"

        depth = depth_reader.read_y(frames[poc].depth_frame_idx)
        ref0 = (
            evaluate_reference(
                poc, ref0_poc, dir0, frames, K_bank, depth,
                width, height, corner_inset,
            )
            if ref0_poc is not None
            else None
        )
        ref1 = (
            evaluate_reference(
                poc, ref1_poc, dir1, frames, K_bank, depth,
                width, height, corner_inset,
            )
            if ref1_poc is not None
            else None
        )

        final_motion, combine_reason = combine_motion(
            [ref0, ref1], min_valid_points, ref_combine
        )
        if final_motion is None:
            tool_on = True
            reason = "insufficient_valid_points_keep_on"
        elif final_motion < threshold:
            tool_on = False
            reason = f"{combine_reason}_below_threshold"
        else:
            tool_on = True
            reason = f"{combine_reason}_at_or_above_threshold"

        decisions[poc] = Decision(
            poc=poc,
            tid=tid,
            coding_order=coding_idx,
            is_intra=False,
            is_reset_boundary=is_reset,
            previous_group_base=prev_base,
            tool_enabled=tool_on,
            reference_eligible=tool_on,
            final_motion4096=final_motion,
            reason=reason,
            ref0=ref0,
            ref1=ref1,
        )

    if prune_unused_intra:
        used_by_enabled: Dict[int, int] = {}
        for d in decisions.values():
            if not d.tool_enabled or d.is_intra:
                continue
            for ev in (d.ref0, d.ref1):
                if ev is not None:
                    used_by_enabled[ev.ref_poc] = used_by_enabled.get(ev.ref_poc, 0) + 1

        for poc, d in decisions.items():
            if poc == 0 or not d.is_intra:
                continue
            if used_by_enabled.get(poc, 0) == 0:
                d.tool_enabled = False
                d.reference_eligible = False
                d.reason = "intra_previous_group_pose_pruned_unused"
            else:
                d.reason = "intra_previous_group_pose_used_by_enabled_picture"

    return sorted(decisions.values(), key=lambda d: d.coding_order)


# ============================================================
# Encoder-side stream serialization
# ============================================================

def encode_pose_record(
    poc: int,
    decision: Decision,
    frames: Dict[int, SequenceFrame],
    last_on_poc: Dict[int, int],
    last_recon_pose: Dict[int, np.ndarray],
    rot_qstep: float,
    trans_qstep: float,
) -> EncodedPoseState:
    base = previous_group_base(poc)
    if base is None:
        raise ValueError("POC 0 has no previous-group pose record")

    if base not in last_on_poc:
        last_on_poc[base] = base
        last_recon_pose[base] = np.eye(4, dtype=np.float64)

    before = last_on_poc[base]
    if not decision.tool_enabled:
        return EncodedPoseState(
            group_base=base,
            target_poc=poc,
            tool_enabled=False,
            last_on_before=before,
        )

    target_rel = compose_relative(frames[poc].T_global, frames[base].T_global)
    last_rel_gt = compose_relative(frames[before].T_global, frames[base].T_global)
    delta_gt = target_rel @ np.linalg.inv(last_rel_gt)
    q, delta_recon = quantize_transform(delta_gt, rot_qstep, trans_qstep)
    pose_recon = delta_recon @ last_recon_pose[base]

    state = EncodedPoseState(
        group_base=base,
        target_poc=poc,
        tool_enabled=True,
        last_on_before=before,
        q_rx=q[0], q_ry=q[1], q_rz=q[2],
        q_tx=q[3], q_ty=q[4], q_tz=q[5],
        delta_recon=delta_recon,
        pose_recon=pose_recon,
    )
    last_on_poc[base] = poc
    last_recon_pose[base] = pose_recon
    return state


def serialize_stream(
    coding_order: Sequence[int],
    decisions_by_poc: Dict[int, Decision],
    frames: Dict[int, SequenceFrame],
    K_bank: Dict[int, np.ndarray],
    rot_qstep: float,
    trans_qstep: float,
    k_qstep: float,
) -> Tuple[List[StreamRecord], Dict[int, EncodedPoseState], Dict[int, np.ndarray], List[dict]]:
    records: List[StreamRecord] = []
    encoded_pose: Dict[int, EncodedPoseState] = {}
    encoded_K: Dict[int, np.ndarray] = {}
    pointer_trace: List[dict] = []

    last_loaded_poc = -1
    last_on_poc: Dict[int, int] = {}
    last_recon_pose: Dict[int, np.ndarray] = {}
    seq = 0

    for coding_idx, carrier_poc in enumerate(coding_order):
        before = last_loaded_poc
        records.append(
            StreamRecord(
                seq=seq,
                coding_order=coding_idx,
                carrier_poc=carrier_poc,
                record_type="SLICE",
                note="picture/slice header start",
            )
        )
        seq += 1

        while last_loaded_poc < carrier_poc:
            target = last_loaded_poc + 1

            if target == 0:
                qk, Kq = quantize_K(K_bank[0], k_qstep)
                encoded_K[0] = Kq
                records.append(
                    StreamRecord(
                        seq=seq,
                        coding_order=coding_idx,
                        carrier_poc=carrier_poc,
                        record_type="RESET_K",
                        target_poc=0,
                        group_base=0,
                        q_fx=qk[0], q_fy=qk[1], q_cx=qk[2], q_cy=qk[3],
                        z_sign=frames[0].z_sign,
                        note="initial group K; group Rt identity",
                    )
                )
                seq += 1
                last_on_poc[0] = 0
                last_recon_pose[0] = np.eye(4, dtype=np.float64)
                last_loaded_poc = 0
                continue

            decision = decisions_by_poc[target]
            state = encode_pose_record(
                target, decision, frames, last_on_poc, last_recon_pose,
                rot_qstep, trans_qstep,
            )
            encoded_pose[target] = state
            records.append(
                StreamRecord(
                    seq=seq,
                    coding_order=coding_idx,
                    carrier_poc=carrier_poc,
                    record_type="FRAME",
                    target_poc=target,
                    group_base=state.group_base,
                    tool_enabled=int(state.tool_enabled),
                    last_on_before=state.last_on_before,
                    q_rx=state.q_rx, q_ry=state.q_ry, q_rz=state.q_rz,
                    q_tx=state.q_tx, q_ty=state.q_ty, q_tz=state.q_tz,
                    note=(
                        "tool ON: quantized delta Rt follows"
                        if state.tool_enabled
                        else "tool OFF: Rt omitted"
                    ),
                )
            )
            seq += 1
            last_loaded_poc = target

            # Boundary ordering: previous-group endpoint first, new K second.
            if target % RESET_PERIOD == 0:
                qk, Kq = quantize_K(K_bank[target], k_qstep)
                encoded_K[target] = Kq
                records.append(
                    StreamRecord(
                        seq=seq,
                        coding_order=coding_idx,
                        carrier_poc=carrier_poc,
                        record_type="RESET_K",
                        target_poc=target,
                        group_base=target,
                        q_fx=qk[0], q_fy=qk[1], q_cx=qk[2], q_cy=qk[3],
                        z_sign=frames[target].z_sign,
                        note="new group K after previous-group endpoint record; Rt identity",
                    )
                )
                seq += 1
                last_on_poc[target] = target
                last_recon_pose[target] = np.eye(4, dtype=np.float64)

        pointer_trace.append(
            {
                "coding_order": coding_idx,
                "carrier_poc": carrier_poc,
                "last_loaded_before": before,
                "last_loaded_after": last_loaded_poc,
                "loaded_count": max(0, last_loaded_poc - before),
            }
        )

    return records, encoded_pose, encoded_K, pointer_trace


# ============================================================
# Decoder-side parser
# ============================================================

def parse_stream(
    records: Sequence[StreamRecord],
    rot_qstep: float,
    trans_qstep: float,
    k_qstep: float,
) -> Tuple[
    Dict[int, DecoderPoseState],
    Dict[int, np.ndarray],
    List[dict],
    List[dict],
]:
    decoded_pose: Dict[int, DecoderPoseState] = {}
    decoded_K: Dict[int, np.ndarray] = {}
    trace: List[dict] = []
    errors: List[dict] = []

    grouped: Dict[int, List[StreamRecord]] = {}
    carrier_info: Dict[int, Tuple[int, int]] = {}
    for r in records:
        grouped.setdefault(r.coding_order, []).append(r)
        carrier_info[r.coding_order] = (r.coding_order, r.carrier_poc)

    last_loaded_poc = -1
    last_on_poc: Dict[int, int] = {}
    last_recon_pose: Dict[int, np.ndarray] = {}

    for coding_idx in sorted(grouped):
        rows = sorted(grouped[coding_idx], key=lambda r: r.seq)
        carrier_poc = rows[0].carrier_poc
        before = last_loaded_poc

        if rows[0].record_type != "SLICE":
            errors.append(
                {
                    "category": "stream_order",
                    "poc": carrier_poc,
                    "detail": "carrier does not start with SLICE record",
                }
            )

        for r in rows[1:]:
            if r.record_type == "FRAME":
                if r.target_poc is None or r.group_base is None or r.tool_enabled is None:
                    errors.append(
                        {
                            "category": "malformed_frame_record",
                            "poc": carrier_poc,
                            "detail": f"seq={r.seq}",
                        }
                    )
                    continue
                expected = last_loaded_poc + 1
                if r.target_poc != expected:
                    errors.append(
                        {
                            "category": "load_pointer",
                            "poc": r.target_poc,
                            "detail": f"expected target {expected}, got {r.target_poc}",
                        }
                    )
                    last_loaded_poc = r.target_poc
                else:
                    last_loaded_poc = expected

                base = r.group_base
                if base not in last_on_poc:
                    last_on_poc[base] = base
                    last_recon_pose[base] = np.eye(4, dtype=np.float64)
                before_on = last_on_poc[base]

                if r.last_on_before != before_on:
                    errors.append(
                        {
                            "category": "last_on_pointer",
                            "poc": r.target_poc,
                            "detail": f"stream={r.last_on_before}, decoder={before_on}",
                        }
                    )

                qvals = (r.q_rx, r.q_ry, r.q_rz, r.q_tx, r.q_ty, r.q_tz)
                if r.tool_enabled:
                    if any(v is None for v in qvals):
                        errors.append(
                            {
                                "category": "missing_rt",
                                "poc": r.target_poc,
                                "detail": "tool ON but one or more q Rt values are absent",
                            }
                        )
                        continue
                    rq = np.array(
                        [
                            dequantize_scalar(int(r.q_rx), rot_qstep),
                            dequantize_scalar(int(r.q_ry), rot_qstep),
                            dequantize_scalar(int(r.q_rz), rot_qstep),
                        ],
                        dtype=np.float64,
                    )
                    tq = np.array(
                        [
                            dequantize_scalar(int(r.q_tx), trans_qstep),
                            dequantize_scalar(int(r.q_ty), trans_qstep),
                            dequantize_scalar(int(r.q_tz), trans_qstep),
                        ],
                        dtype=np.float64,
                    )
                    delta = rt_to_T(rq, tq)
                    pose = delta @ last_recon_pose[base]
                    last_on_poc[base] = r.target_poc
                    last_recon_pose[base] = pose
                else:
                    if any(v is not None for v in qvals):
                        errors.append(
                            {
                                "category": "unexpected_rt",
                                "poc": r.target_poc,
                                "detail": "tool OFF but q Rt value is present",
                            }
                        )
                    pose = None

                decoded_pose[r.target_poc] = DecoderPoseState(
                    group_base=base,
                    target_poc=r.target_poc,
                    tool_enabled=bool(r.tool_enabled),
                    last_on_before=before_on,
                    q_values=qvals,
                    pose_recon=pose,
                )

            elif r.record_type == "RESET_K":
                if (
                    r.group_base is None
                    or r.q_fx is None or r.q_fy is None
                    or r.q_cx is None or r.q_cy is None
                ):
                    errors.append(
                        {
                            "category": "malformed_k_record",
                            "poc": carrier_poc,
                            "detail": f"seq={r.seq}",
                        }
                    )
                    continue
                base = r.group_base
                Kq = np.array(
                    [
                        [dequantize_scalar(r.q_fx, k_qstep), 0.0, dequantize_scalar(r.q_cx, k_qstep)],
                        [0.0, dequantize_scalar(r.q_fy, k_qstep), dequantize_scalar(r.q_cy, k_qstep)],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                decoded_K[base] = Kq
                last_on_poc[base] = base
                last_recon_pose[base] = np.eye(4, dtype=np.float64)
                # The initial RESET_K record also establishes POC 0 as loaded.
                # Later reset records follow the FRAME record of the same
                # boundary POC, so they must not advance the pointer again.
                if base == 0 and last_loaded_poc == -1:
                    last_loaded_poc = 0

            else:
                errors.append(
                    {
                        "category": "unknown_record_type",
                        "poc": carrier_poc,
                        "detail": r.record_type,
                    }
                )

        if last_loaded_poc < carrier_poc:
            errors.append(
                {
                    "category": "current_poc_not_loaded",
                    "poc": carrier_poc,
                    "detail": f"last_loaded={last_loaded_poc}",
                }
            )

        trace.append(
            {
                "coding_order": coding_idx,
                "carrier_poc": carrier_poc,
                "last_loaded_before": before,
                "last_loaded_after": last_loaded_poc,
                "loaded_count": max(0, last_loaded_poc - before),
            }
        )

    return decoded_pose, decoded_K, trace, errors


# ============================================================
# Consistency checks
# ============================================================

def validate_encoder_decoder(
    decisions: Sequence[Decision],
    coding_order: Sequence[int],
    encoded_pose: Dict[int, EncodedPoseState],
    encoded_K: Dict[int, np.ndarray],
    encoder_trace: Sequence[dict],
    decoded_pose: Dict[int, DecoderPoseState],
    decoded_K: Dict[int, np.ndarray],
    decoder_trace: Sequence[dict],
    parser_errors: Sequence[dict],
    tolerance: float,
) -> List[dict]:
    mismatches: List[dict] = list(parser_errors)
    by_poc = {d.poc: d for d in decisions}

    for poc, enc in sorted(encoded_pose.items()):
        dec = decoded_pose.get(poc)
        if dec is None:
            mismatches.append(
                {"category": "missing_decoded_pose_record", "poc": poc, "detail": ""}
            )
            continue
        if bool(enc.tool_enabled) != bool(dec.tool_enabled):
            mismatches.append(
                {
                    "category": "tool_flag",
                    "poc": poc,
                    "detail": f"encoder={enc.tool_enabled}, decoder={dec.tool_enabled}",
                }
            )
        if enc.group_base != dec.group_base:
            mismatches.append(
                {
                    "category": "group_base",
                    "poc": poc,
                    "detail": f"encoder={enc.group_base}, decoder={dec.group_base}",
                }
            )
        if enc.last_on_before != dec.last_on_before:
            mismatches.append(
                {
                    "category": "last_on_before",
                    "poc": poc,
                    "detail": f"encoder={enc.last_on_before}, decoder={dec.last_on_before}",
                }
            )
        enc_q = (enc.q_rx, enc.q_ry, enc.q_rz, enc.q_tx, enc.q_ty, enc.q_tz)
        if enc_q != dec.q_values:
            mismatches.append(
                {
                    "category": "quantized_rt",
                    "poc": poc,
                    "detail": f"encoder={enc_q}, decoder={dec.q_values}",
                }
            )
        if enc.tool_enabled:
            if enc.pose_recon is None or dec.pose_recon is None:
                mismatches.append(
                    {"category": "missing_reconstructed_rt", "poc": poc, "detail": ""}
                )
            else:
                err = matrix_error(enc.pose_recon, dec.pose_recon)
                if err > tolerance:
                    mismatches.append(
                        {
                            "category": "reconstructed_rt",
                            "poc": poc,
                            "detail": f"max_abs_error={err:.12e}",
                        }
                    )

        if bool(by_poc[poc].tool_enabled) != bool(enc.tool_enabled):
            mismatches.append(
                {
                    "category": "decision_vs_stream_flag",
                    "poc": poc,
                    "detail": f"decision={by_poc[poc].tool_enabled}, stream={enc.tool_enabled}",
                }
            )

    for base, Kenc in sorted(encoded_K.items()):
        Kdec = decoded_K.get(base)
        if Kdec is None:
            mismatches.append(
                {"category": "missing_decoded_K", "poc": base, "detail": ""}
            )
        else:
            err = matrix_error(Kenc, Kdec)
            if err > tolerance:
                mismatches.append(
                    {
                        "category": "decoded_K",
                        "poc": base,
                        "detail": f"max_abs_error={err:.12e}",
                    }
                )

    if len(encoder_trace) != len(decoder_trace):
        mismatches.append(
            {
                "category": "trace_length",
                "poc": "",
                "detail": f"encoder={len(encoder_trace)}, decoder={len(decoder_trace)}",
            }
        )
    else:
        for e, d in zip(encoder_trace, decoder_trace):
            if e != d:
                mismatches.append(
                    {
                        "category": "pointer_trace",
                        "poc": e["carrier_poc"],
                        "detail": f"encoder={e}, decoder={d}",
                    }
                )

    # Verify that every enabled non-intra current picture and all of its chosen
    # refs have a decoder-side geometry representation in the same reset group.
    for d in decisions:
        if d.poc == 0 or not d.tool_enabled or d.is_intra:
            continue
        base = previous_group_base(d.poc)
        if base is None:
            continue
        current_state = decoded_pose.get(d.poc)
        if current_state is None or not current_state.tool_enabled:
            mismatches.append(
                {
                    "category": "enabled_current_pose_unavailable",
                    "poc": d.poc,
                    "detail": "",
                }
            )
        for ev in (d.ref0, d.ref1):
            if ev is None:
                continue
            rp = ev.ref_poc
            if rp == base:
                if base not in decoded_K:
                    mismatches.append(
                        {
                            "category": "reset_anchor_K_unavailable",
                            "poc": d.poc,
                            "detail": f"ref={rp}, base={base}",
                        }
                    )
            else:
                rs = decoded_pose.get(rp)
                if rs is None or not rs.tool_enabled or rs.group_base != base:
                    mismatches.append(
                        {
                            "category": "selected_ref_pose_unavailable",
                            "poc": d.poc,
                            "detail": f"ref={rp}, expected_group={base}",
                        }
                    )

    return mismatches


# ============================================================
# Output writers
# ============================================================

def write_decisions_csv(path: Path, decisions: Sequence[Decision]) -> None:
    ensure_parent(path)
    fields = [
        "coding_order", "poc", "tid", "is_intra", "is_reset_boundary",
        "previous_group_base", "ref0_direction", "ref0_poc",
        "ref0_valid_points", "ref0_avg_mv_px", "ref0_motion4096",
        "ref1_direction", "ref1_poc", "ref1_valid_points",
        "ref1_avg_mv_px", "ref1_motion4096", "final_motion4096",
        "tool_enabled", "reference_eligible", "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in decisions:
            row = {
                "coding_order": d.coding_order,
                "poc": d.poc,
                "tid": d.tid,
                "is_intra": int(d.is_intra),
                "is_reset_boundary": int(d.is_reset_boundary),
                "previous_group_base": safe_int(d.previous_group_base),
                "final_motion4096": safe_float(d.final_motion4096),
                "tool_enabled": int(d.tool_enabled),
                "reference_eligible": int(d.reference_eligible),
                "reason": d.reason,
            }
            for prefix, ev in (("ref0", d.ref0), ("ref1", d.ref1)):
                row[f"{prefix}_direction"] = "" if ev is None else ev.direction
                row[f"{prefix}_poc"] = "" if ev is None else ev.ref_poc
                row[f"{prefix}_valid_points"] = 0 if ev is None else ev.valid_points
                row[f"{prefix}_avg_mv_px"] = "" if ev is None else ev.avg_mv_px
                row[f"{prefix}_motion4096"] = "" if ev is None else ev.motion4096
            w.writerow(row)


def write_stream_csv(path: Path, records: Sequence[StreamRecord]) -> None:
    ensure_parent(path)
    fields = [
        "seq", "coding_order", "carrier_poc", "record_type", "target_poc",
        "group_base", "tool_enabled", "last_on_before",
        "q_rx", "q_ry", "q_rz", "q_tx", "q_ty", "q_tz",
        "q_fx", "q_fy", "q_cx", "q_cy", "z_sign", "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) if getattr(r, k) is not None else "" for k in fields})


def write_pose_states_csv(
    path: Path,
    encoded: Dict[int, EncodedPoseState],
    decoded: Dict[int, DecoderPoseState],
) -> None:
    ensure_parent(path)
    fields = [
        "poc", "group_base", "tool_enabled", "last_on_before",
        "q_rx", "q_ry", "q_rz", "q_tx", "q_ty", "q_tz",
        "encoder_r00", "encoder_r01", "encoder_r02", "encoder_tx",
        "encoder_r10", "encoder_r11", "encoder_r12", "encoder_ty",
        "encoder_r20", "encoder_r21", "encoder_r22", "encoder_tz",
        "decoder_r00", "decoder_r01", "decoder_r02", "decoder_tx",
        "decoder_r10", "decoder_r11", "decoder_r12", "decoder_ty",
        "decoder_r20", "decoder_r21", "decoder_r22", "decoder_tz",
        "max_abs_matrix_error",
    ]

    def put_matrix(row: dict, prefix: str, T: Optional[np.ndarray]) -> None:
        names = [
            ("r00", 0, 0), ("r01", 0, 1), ("r02", 0, 2), ("tx", 0, 3),
            ("r10", 1, 0), ("r11", 1, 1), ("r12", 1, 2), ("ty", 1, 3),
            ("r20", 2, 0), ("r21", 2, 1), ("r22", 2, 2), ("tz", 2, 3),
        ]
        for name, y, x in names:
            row[f"{prefix}_{name}"] = "" if T is None else float(T[y, x])

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for poc in sorted(encoded):
            e = encoded[poc]
            d = decoded.get(poc)
            q = (e.q_rx, e.q_ry, e.q_rz, e.q_tx, e.q_ty, e.q_tz)
            row = {
                "poc": poc,
                "group_base": e.group_base,
                "tool_enabled": int(e.tool_enabled),
                "last_on_before": e.last_on_before,
                "q_rx": safe_int(q[0]), "q_ry": safe_int(q[1]), "q_rz": safe_int(q[2]),
                "q_tx": safe_int(q[3]), "q_ty": safe_int(q[4]), "q_tz": safe_int(q[5]),
            }
            put_matrix(row, "encoder", e.pose_recon)
            put_matrix(row, "decoder", None if d is None else d.pose_recon)
            row["max_abs_matrix_error"] = (
                ""
                if e.pose_recon is None or d is None or d.pose_recon is None
                else matrix_error(e.pose_recon, d.pose_recon)
            )
            w.writerow(row)


def write_trace_csv(path: Path, trace: Sequence[dict]) -> None:
    ensure_parent(path)
    fields = [
        "coding_order", "carrier_poc", "last_loaded_before",
        "last_loaded_after", "loaded_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trace)


def write_mismatch_csv(path: Path, mismatches: Sequence[dict]) -> None:
    ensure_parent(path)
    fields = ["category", "poc", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in mismatches:
            w.writerow(
                {
                    "category": m.get("category", ""),
                    "poc": m.get("poc", ""),
                    "detail": m.get("detail", ""),
                }
            )


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    header: dict,
    decisions: Sequence[Decision],
    records: Sequence[StreamRecord],
    mismatches: Sequence[dict],
    warnings: Sequence[str],
) -> None:
    ensure_parent(path)
    tool_off = [d for d in decisions if d.poc > 0 and not d.tool_enabled]
    non_intra = [d for d in decisions if d.poc > 0 and not d.is_intra]
    frame_records = [r for r in records if r.record_type == "FRAME"]
    rt_records = [r for r in frame_records if r.tool_enabled == 1]
    reset_records = [r for r in records if r.record_type == "RESET_K"]
    summary = {
        "format": "camproj_metadata_stream_sim_v1",
        "sequence_name": header.get("sequence_name"),
        "mode": args.mode,
        "gop_size": args.gop_size,
        "intra_period": args.intra_period,
        "depth_reset_period": RESET_PERIOD,
        "offset_threshold": args.offset_threshold,
        "ref_combine": args.ref_combine,
        "rot_qstep": args.rot_qstep,
        "trans_qstep": args.trans_qstep,
        "k_qstep": args.k_qstep,
        "picture_count": len(decisions),
        "tool_off_count_excluding_poc0": len(tool_off),
        "tool_off_ratio_excluding_poc0": len(tool_off) / max(1, len(decisions) - 1),
        "tool_off_ratio_non_intra": (
            sum(not d.tool_enabled for d in non_intra) / len(non_intra)
            if non_intra else 0.0
        ),
        "stream_record_count": len(records),
        "frame_flag_record_count": len(frame_records),
        "rt_present_record_count": len(rt_records),
        "rt_omitted_record_count": len(frame_records) - len(rt_records),
        "reset_K_record_count": len(reset_records),
        "mismatch_count": len(mismatches),
        "encoder_decoder_match": len(mismatches) == 0,
        "warnings": list(warnings),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--camera-jsonl", required=True)
    ap.add_argument(
        "--depth-yuv",
        default=None,
        help="Merged depth YUV420p10le. If omitted, use JSONL header depth_yuv.",
    )
    ap.add_argument("--output-prefix", default=None)
    ap.add_argument("--mode", choices=["ra", "ldb"], default="ra")
    ap.add_argument("--gop-size", type=int, default=32)
    ap.add_argument("--intra-period", type=int, default=32)
    ap.add_argument("--offset-threshold", type=float, required=True)
    ap.add_argument("--ref-combine", choices=["min", "mean", "max"], default="min")
    ap.add_argument("--min-valid-points", type=int, default=3)
    ap.add_argument("--corner-inset", type=int, default=0)
    ap.add_argument(
        "--rot-qstep",
        type=float,
        default=1e-6,
        help="Rodrigues-vector delta quantization step in radians.",
    )
    ap.add_argument(
        "--trans-qstep",
        type=float,
        default=1e-5,
        help="Translation delta quantization step in input camera units.",
    )
    ap.add_argument(
        "--k-qstep",
        type=float,
        default=1e-3,
        help="fx/fy/cx/cy quantization step.",
    )
    ap.add_argument(
        "--no-prune-unused-intra",
        action="store_true",
        help="Keep every intra picture's previous-group endpoint pose.",
    )
    ap.add_argument("--overlap-tolerance", type=float, default=1e-5)
    ap.add_argument("--match-tolerance", type=float, default=1e-12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.gop_size <= 0:
        raise ValueError("--gop-size must be positive")
    if args.intra_period <= 0:
        raise ValueError("--intra-period must be positive")
    if args.mode == "ra" and args.intra_period % args.gop_size != 0:
        raise ValueError("For this RA simulator, --intra-period must be a multiple of --gop-size")
    if args.offset_threshold < 0:
        raise ValueError("--offset-threshold must be nonnegative")
    if not (1 <= args.min_valid_points <= 5):
        raise ValueError("--min-valid-points must be in [1,5]")
    if args.corner_inset < 0:
        raise ValueError("--corner-inset must be nonnegative")
    if args.rot_qstep <= 0 or args.trans_qstep <= 0 or args.k_qstep <= 0:
        raise ValueError("all quantization steps must be positive")

    camera_path = Path(args.camera_jsonl)
    header, source_frames = read_camera_jsonl(camera_path)
    width = int(header["width"])
    height = int(header["height"])
    pose_mode = str(header["pose_mode"])

    if args.depth_yuv:
        depth_path = Path(args.depth_yuv)
    else:
        depth_name = header.get("depth_yuv")
        if not depth_name:
            raise ValueError("--depth-yuv omitted and JSONL header has no depth_yuv")
        depth_path = camera_path.parent / str(depth_name)

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = camera_path.with_name(
            f"{camera_path.stem}_{args.mode}_g{args.gop_size}_ip{args.intra_period}_stream_sim"
        )

    frames, warnings = build_sequence_frames(
        source_frames, pose_mode, args.overlap_tolerance
    )
    K_bank = build_reset_K_bank(frames)
    max_poc = max(frames)
    coding_order, tids = build_coding_schedule(max_poc, args.mode, args.gop_size)

    with DepthYUV420P10LE(depth_path, width, height) as depth_reader:
        decisions = simulate_decisions(
            frames=frames,
            K_bank=K_bank,
            depth_reader=depth_reader,
            width=width,
            height=height,
            mode=args.mode,
            coding_order=coding_order,
            tids=tids,
            intra_period=args.intra_period,
            threshold=args.offset_threshold,
            ref_combine=args.ref_combine,
            min_valid_points=args.min_valid_points,
            corner_inset=args.corner_inset,
            prune_unused_intra=not args.no_prune_unused_intra,
        )

    decisions_by_poc = {d.poc: d for d in decisions}
    records, encoded_pose, encoded_K, encoder_trace = serialize_stream(
        coding_order=coding_order,
        decisions_by_poc=decisions_by_poc,
        frames=frames,
        K_bank=K_bank,
        rot_qstep=args.rot_qstep,
        trans_qstep=args.trans_qstep,
        k_qstep=args.k_qstep,
    )

    decoded_pose, decoded_K, decoder_trace, parser_errors = parse_stream(
        records,
        rot_qstep=args.rot_qstep,
        trans_qstep=args.trans_qstep,
        k_qstep=args.k_qstep,
    )

    mismatches = validate_encoder_decoder(
        decisions=decisions,
        coding_order=coding_order,
        encoded_pose=encoded_pose,
        encoded_K=encoded_K,
        encoder_trace=encoder_trace,
        decoded_pose=decoded_pose,
        decoded_K=decoded_K,
        decoder_trace=decoder_trace,
        parser_errors=parser_errors,
        tolerance=args.match_tolerance,
    )

    decisions_path = Path(str(prefix) + "_decisions.csv")
    stream_path = Path(str(prefix) + "_stream.csv")
    states_path = Path(str(prefix) + "_pose_states.csv")
    enc_trace_path = Path(str(prefix) + "_encoder_trace.csv")
    dec_trace_path = Path(str(prefix) + "_decoder_trace.csv")
    mismatch_path = Path(str(prefix) + "_mismatch.csv")
    summary_path = Path(str(prefix) + "_summary.json")

    write_decisions_csv(decisions_path, decisions)
    write_stream_csv(stream_path, records)
    write_pose_states_csv(states_path, encoded_pose, decoded_pose)
    write_trace_csv(enc_trace_path, encoder_trace)
    write_trace_csv(dec_trace_path, decoder_trace)
    write_mismatch_csv(mismatch_path, mismatches)
    write_summary_json(
        summary_path, args, header, decisions, records, mismatches, warnings
    )

    off_count = sum(d.poc > 0 and not d.tool_enabled for d in decisions)
    print(f"[CONFIG] mode={args.mode}, GOP={args.gop_size}, IntraPeriod={args.intra_period}, reset={RESET_PERIOD}")
    print(f"[SEQUENCE] POC 0..{max_poc}, pictures={len(frames)}")
    print(f"[DECISION] tool_off={off_count}/{max(1, len(decisions)-1)}")
    print(f"[STREAM] records={len(records)}, frame_records={len(encoded_pose)}, K_resets={len(encoded_K)}")
    print(f"[CHECK] mismatches={len(mismatches)}")
    for w in warnings:
        print(f"[WARN] {w}")
    print(f"[OK] decisions : {decisions_path}")
    print(f"[OK] stream    : {stream_path}")
    print(f"[OK] states    : {states_path}")
    print(f"[OK] enc trace : {enc_trace_path}")
    print(f"[OK] dec trace : {dec_trace_path}")
    print(f"[OK] mismatch  : {mismatch_path}")
    print(f"[OK] summary   : {summary_path}")

    if mismatches:
        print("[FAIL] encoder/decoder mismatch detected", file=sys.stderr)
        raise SystemExit(2)
    print("[PASS] encoder/decoder metadata reconstruction is identical")


if __name__ == "__main__":
    main()
