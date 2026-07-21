#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate GOP-level camera-projection enable/disable in hierarchical RA.

Inputs
------
1) Camera JSONL produced by the merge script in the prompt.
2) Merged depth YUV420p10le produced by the same script.

Simulation
----------
- Camera records are grouped by gop_idx.
- GOP-local camera poses are reconstructed as camera-from-GOP0 transforms.
- A hierarchical-RA temporal ID is assigned recursively:
      endpoints -> TID 0
      midpoint  -> TID 1
      quarter points -> TID 2, ...
- Pictures are processed in RA dependency order.
- For each non-anchor picture, the nearest already processed, reference-eligible
  lower-TID picture is selected.
- Five target points are tested:
      four corners + center
- Target depth and target/reference camera parameters are used to project each
  target point into the selected reference picture.
- Pixel displacement is normalized by:
      (width + height) / 2
- If the selected metric is smaller than --offset-threshold, the current
  picture is marked tool-off.
- A tool-off picture is excluded from future camera-projection references.

Outputs
-------
- <prefix>_frames.csv   : picture-level decisions and 5-point measurements
- <prefix>_summary.json : sequence/GOP-level statistics
- <prefix>_mask.jsonl   : compact per-picture tool/reference eligibility

Important assumptions
---------------------
- Depth is camera-Z depth, reconstructed as:
      depth_code * depth_scale_real
- rvec/tvec follow the conventions written by the supplied merge script.
- The first and last picture of each GOP are treated as RA anchors and remain
  reference-eligible by default. Use --allow-anchor-off to test otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


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
    record: dict


@dataclass
class Decision:
    frame: CameraFrame
    tid: int
    coding_order: int
    is_anchor: bool
    reference_poc: Optional[int]
    reference_local_poc: Optional[int]
    temporal_distance: Optional[int]
    valid_points: int
    mean_motion_px: Optional[float]
    rms_motion_px: Optional[float]
    max_motion_px: Optional[float]
    metric_px: Optional[float]
    metric_norm: Optional[float]
    tool_enabled: bool
    reference_eligible: bool
    reason: str
    point_results: List[dict]


# ============================================================
# JSONL / pose handling
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
            typ = obj.get("type")

            if typ == "header":
                if header is not None:
                    raise ValueError(f"{path}: multiple headers")
                header = obj
                continue

            if typ != "frame":
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
                    record=obj,
                )
            )

    if header is None:
        raise ValueError(f"{path}: header not found")
    if not frames:
        raise ValueError(f"{path}: no frame records")

    return header, frames


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(rvec.reshape(3, 1))[0]
    T[:3, 3] = tvec
    return T


def reconstruct_gop_camera_from_origin(
    frames: Sequence[CameraFrame],
    pose_mode: str,
) -> Dict[int, np.ndarray]:
    """
    Return A[i] = camera_i_from_camera_gop0.

    current_to_previous:
        input T = camera_(i-1)_from_camera_i
        A_i = inv(T) @ A_(i-1)

    gop_local:
        input T = camera_i_from_camera_0
        A_i = T

    absolute:
        input T = camera_i_from_world
        convert to camera_i_from_camera_gop0:
            A_i = T_i @ inv(T_0)
    """
    ordered = sorted(frames, key=lambda x: x.local_poc)
    raw = {fr.local_poc: rt_to_T(fr.rvec, fr.tvec) for fr in ordered}
    out: Dict[int, np.ndarray] = {}

    if pose_mode == "current_to_previous":
        first = ordered[0].local_poc
        out[first] = np.eye(4, dtype=np.float64)

        for prev, cur in zip(ordered[:-1], ordered[1:]):
            if cur.local_poc != prev.local_poc + 1:
                raise ValueError(
                    f"GOP {cur.gop_idx}: current_to_previous requires "
                    f"contiguous local_poc, got {prev.local_poc}->{cur.local_poc}"
                )
            T_prev_from_cur = raw[cur.local_poc]
            out[cur.local_poc] = (
                np.linalg.inv(T_prev_from_cur) @ out[prev.local_poc]
            )

    elif pose_mode == "gop_local":
        for fr in ordered:
            out[fr.local_poc] = raw[fr.local_poc]

    elif pose_mode == "absolute":
        T0_inv = np.linalg.inv(raw[ordered[0].local_poc])
        for fr in ordered:
            out[fr.local_poc] = raw[fr.local_poc] @ T0_inv

    else:
        raise ValueError(f"unsupported pose_mode: {pose_mode}")

    return out


# ============================================================
# Depth reader
# ============================================================

class DepthYUV420P10LE:
    def __init__(self, path: Path, width: int, height: int):
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("YUV420 width/height must be positive and even")

        self.path = path
        self.width = width
        self.height = height
        self.y_samples = width * height
        self.uv_samples = (width // 2) * (height // 2)
        self.frame_samples = self.y_samples + 2 * self.uv_samples
        self.frame_bytes = self.frame_samples * 2

        size = path.stat().st_size
        if size % self.frame_bytes != 0:
            raise ValueError(
                f"{path}: size {size} is not a multiple of frame size "
                f"{self.frame_bytes}"
            )
        self.frame_count = size // self.frame_bytes
        self.fp = path.open("rb")

    def close(self) -> None:
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_y(self, frame_idx: int) -> np.ndarray:
        if not (0 <= frame_idx < self.frame_count):
            raise IndexError(
                f"depth frame_idx={frame_idx}, frame_count={self.frame_count}"
            )

        self.fp.seek(frame_idx * self.frame_bytes)
        raw = self.fp.read(self.y_samples * 2)
        if len(raw) != self.y_samples * 2:
            raise EOFError(f"short read at depth frame {frame_idx}")

        return np.frombuffer(raw, dtype="<u2").reshape(
            self.height, self.width
        )


# ============================================================
# Hierarchical RA structure
# ============================================================

def assign_hierarchical_tid(local_pocs: Sequence[int]) -> Dict[int, int]:
    """
    Assign TIDs recursively over the sorted pictures.

    For 0..32:
      0,32 -> TID 0
      16   -> TID 1
      8,24 -> TID 2
      ...
    """
    pocs = sorted(set(int(x) for x in local_pocs))
    if len(pocs) == 1:
        return {pocs[0]: 0}

    tid: Dict[int, int] = {pocs[0]: 0, pocs[-1]: 0}

    def recurse(lo_idx: int, hi_idx: int, level: int) -> None:
        if hi_idx - lo_idx <= 1:
            return

        mid_idx = (lo_idx + hi_idx) // 2
        mid_poc = pocs[mid_idx]
        tid[mid_poc] = min(tid.get(mid_poc, level), level)

        recurse(lo_idx, mid_idx, level + 1)
        recurse(mid_idx, hi_idx, level + 1)

    recurse(0, len(pocs) - 1, 1)

    # Non-power-of-two / irregular GOP safety.
    next_tid = max(tid.values(), default=0) + 1
    for p in pocs:
        if p not in tid:
            tid[p] = next_tid

    return tid


def ra_coding_order(
    frames: Sequence[CameraFrame],
    tids: Dict[int, int],
) -> List[CameraFrame]:
    """
    Anchors first, then increasing TID. POC is only a deterministic tie-break.
    """
    return sorted(
        frames,
        key=lambda fr: (tids[fr.local_poc], fr.local_poc),
    )


def select_nearest_reference(
    current: CameraFrame,
    tid: int,
    processed: Sequence[Decision],
    tie_break: str,
    motion_evaluator,
) -> Tuple[Optional[Decision], Optional[dict]]:
    """
    Candidate restriction:
      - already processed
      - reference_eligible
      - lower TID than current

    Among minimum temporal-distance candidates:
      lower_poc   -> lower absolute POC
      higher_poc  -> higher absolute POC
      lower_motion -> evaluate tied candidates and choose smaller motion
    """
    candidates = [
        d
        for d in processed
        if d.reference_eligible and d.tid < tid
    ]
    if not candidates:
        return None, None

    min_dist = min(
        abs(current.local_poc - d.frame.local_poc)
        for d in candidates
    )
    tied = [
        d
        for d in candidates
        if abs(current.local_poc - d.frame.local_poc) == min_dist
    ]

    if len(tied) == 1:
        ref = tied[0]
        return ref, motion_evaluator(ref.frame)

    if tie_break == "lower_poc":
        ref = min(tied, key=lambda d: d.frame.local_poc)
        return ref, motion_evaluator(ref.frame)

    if tie_break == "higher_poc":
        ref = max(tied, key=lambda d: d.frame.local_poc)
        return ref, motion_evaluator(ref.frame)

    if tie_break == "lower_motion":
        evaluated = [(d, motion_evaluator(d.frame)) for d in tied]

        def key(item):
            _, ev = item
            metric = ev.get("selected_metric_norm")
            return (
                float("inf") if metric is None else metric,
                item[0].frame.local_poc,
            )

        return min(evaluated, key=key)

    raise ValueError(tie_break)


# ============================================================
# Projection and five-point motion
# ============================================================

def five_points(width: int, height: int, inset: int) -> List[Tuple[str, int, int]]:
    x0 = min(max(0, inset), width - 1)
    y0 = min(max(0, inset), height - 1)
    x1 = max(0, width - 1 - inset)
    y1 = max(0, height - 1 - inset)
    xc = (width - 1) // 2
    yc = (height - 1) // 2

    return [
        ("top_left", x0, y0),
        ("top_right", x1, y0),
        ("bottom_left", x0, y1),
        ("bottom_right", x1, y1),
        ("center", xc, yc),
    ]


def project_target_point_to_reference(
    x: float,
    y: float,
    depth: float,
    K_target: np.ndarray,
    K_ref: np.ndarray,
    A_target: np.ndarray,
    A_ref: np.ndarray,
    z_sign: float,
) -> Optional[Tuple[float, float, float]]:
    """
    A_i = camera_i_from_camera_gop0.

    target camera -> GOP0 camera -> reference camera:
        X_ref = A_ref @ inv(A_target) @ X_target
    """
    if not np.isfinite(depth) or depth <= 0:
        return None

    ray = np.linalg.inv(K_target) @ np.array([x, y, 1.0], np.float64)
    X_t = ray * (depth * z_sign)

    T_ref_from_target = A_ref @ np.linalg.inv(A_target)
    X_r = (
        T_ref_from_target[:3, :3] @ X_t
        + T_ref_from_target[:3, 3]
    )

    z = float(X_r[2])
    if not np.isfinite(z) or abs(z) < 1e-12 or z * z_sign <= 0:
        return None

    q = K_ref @ X_r
    u = float(q[0] / q[2])
    v = float(q[1] / q[2])

    if not (np.isfinite(u) and np.isfinite(v)):
        return None

    return u, v, z


def evaluate_five_point_motion(
    target: CameraFrame,
    reference: CameraFrame,
    target_depth_code: np.ndarray,
    poses: Dict[int, np.ndarray],
    width: int,
    height: int,
    inset: int,
    metric_name: str,
) -> dict:
    norm_denom = (width + height) / 2.0
    results = []
    motions = []

    for name, x, y in five_points(width, height, inset):
        code = int(target_depth_code[y, x])
        depth = code * target.depth_scale_real

        pr = project_target_point_to_reference(
            x=float(x),
            y=float(y),
            depth=float(depth),
            K_target=target.K,
            K_ref=reference.K,
            A_target=poses[target.local_poc],
            A_ref=poses[reference.local_poc],
            z_sign=target.z_sign,
        )

        if pr is None:
            results.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "depth_code": code,
                    "depth": float(depth),
                    "valid": False,
                    "ref_x": None,
                    "ref_y": None,
                    "motion_px": None,
                    "motion_norm": None,
                }
            )
            continue

        u, v, _ = pr
        motion = math.hypot(u - x, v - y)
        motions.append(motion)
        results.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "depth_code": code,
                "depth": float(depth),
                "valid": True,
                "ref_x": u,
                "ref_y": v,
                "motion_px": motion,
                "motion_norm": motion / norm_denom,
            }
        )

    if motions:
        a = np.asarray(motions, np.float64)
        mean_px = float(np.mean(a))
        rms_px = float(np.sqrt(np.mean(a * a)))
        max_px = float(np.max(a))
    else:
        mean_px = rms_px = max_px = None

    metric_map = {
        "mean": mean_px,
        "rms": rms_px,
        "max": max_px,
    }
    selected_px = metric_map[metric_name]
    selected_norm = (
        None if selected_px is None else selected_px / norm_denom
    )

    return {
        "points": results,
        "valid_points": len(motions),
        "mean_motion_px": mean_px,
        "rms_motion_px": rms_px,
        "max_motion_px": max_px,
        "selected_metric_px": selected_px,
        "selected_metric_norm": selected_norm,
    }


# ============================================================
# GOP simulation
# ============================================================

def simulate_gop(
    frames: Sequence[CameraFrame],
    pose_mode: str,
    depth_reader: DepthYUV420P10LE,
    width: int,
    height: int,
    threshold: float,
    metric_name: str,
    min_valid_points: int,
    inset: int,
    tie_break: str,
    allow_anchor_off: bool,
) -> List[Decision]:
    by_local = {fr.local_poc: fr for fr in frames}
    if len(by_local) != len(frames):
        raise ValueError(
            f"GOP {frames[0].gop_idx}: duplicate local_poc records"
        )

    tids = assign_hierarchical_tid(list(by_local))
    order = ra_coding_order(frames, tids)
    poses = reconstruct_gop_camera_from_origin(frames, pose_mode)

    anchors = {
        min(by_local),
        max(by_local),
    }

    processed: List[Decision] = []
    depth_cache: Dict[int, np.ndarray] = {}

    def get_depth(fr: CameraFrame) -> np.ndarray:
        arr = depth_cache.get(fr.depth_frame_idx)
        if arr is None:
            arr = depth_reader.read_y(fr.depth_frame_idx)
            depth_cache.clear()
            depth_cache[fr.depth_frame_idx] = arr
        return arr

    for coding_order, fr in enumerate(order):
        tid = tids[fr.local_poc]
        is_anchor = fr.local_poc in anchors

        if is_anchor and not allow_anchor_off:
            processed.append(
                Decision(
                    frame=fr,
                    tid=tid,
                    coding_order=coding_order,
                    is_anchor=True,
                    reference_poc=None,
                    reference_local_poc=None,
                    temporal_distance=None,
                    valid_points=0,
                    mean_motion_px=None,
                    rms_motion_px=None,
                    max_motion_px=None,
                    metric_px=None,
                    metric_norm=None,
                    tool_enabled=True,
                    reference_eligible=True,
                    reason="anchor_forced_on",
                    point_results=[],
                )
            )
            continue

        target_depth = get_depth(fr)

        def evaluator(ref_fr: CameraFrame) -> dict:
            return evaluate_five_point_motion(
                target=fr,
                reference=ref_fr,
                target_depth_code=target_depth,
                poses=poses,
                width=width,
                height=height,
                inset=inset,
                metric_name=metric_name,
            )

        ref_decision, ev = select_nearest_reference(
            current=fr,
            tid=tid,
            processed=processed,
            tie_break=tie_break,
            motion_evaluator=evaluator,
        )

        if ref_decision is None or ev is None:
            # Conservative: no valid RA reference means do not remove Rt.
            tool_on = True
            eligible = True
            reason = "no_eligible_lower_tid_reference"
            ref_poc = None
            ref_local_poc = None
            temporal_distance = None
            valid_points = 0
            mean_px = rms_px = max_px = metric_px = metric_norm = None
            points = []
        else:
            ref_poc = ref_decision.frame.poc
            ref_local_poc = ref_decision.frame.local_poc
            temporal_distance = abs(fr.local_poc - ref_local_poc)
            valid_points = int(ev["valid_points"])
            mean_px = ev["mean_motion_px"]
            rms_px = ev["rms_motion_px"]
            max_px = ev["max_motion_px"]
            metric_px = ev["selected_metric_px"]
            metric_norm = ev["selected_metric_norm"]
            points = ev["points"]

            if valid_points < min_valid_points or metric_norm is None:
                tool_on = True
                eligible = True
                reason = "insufficient_valid_points_keep_on"
            elif metric_norm < threshold:
                tool_on = False
                eligible = False
                reason = "motion_below_threshold"
            else:
                tool_on = True
                eligible = True
                reason = "motion_at_or_above_threshold"

        processed.append(
            Decision(
                frame=fr,
                tid=tid,
                coding_order=coding_order,
                is_anchor=is_anchor,
                reference_poc=ref_poc,
                reference_local_poc=ref_local_poc,
                temporal_distance=temporal_distance,
                valid_points=valid_points,
                mean_motion_px=mean_px,
                rms_motion_px=rms_px,
                max_motion_px=max_px,
                metric_px=metric_px,
                metric_norm=metric_norm,
                tool_enabled=tool_on,
                reference_eligible=eligible,
                reason=reason,
                point_results=points,
            )
        )

    return sorted(processed, key=lambda d: d.frame.local_poc)


# ============================================================
# Output
# ============================================================

POINT_NAMES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
]


def point_map(decision: Decision) -> Dict[str, dict]:
    return {p["name"]: p for p in decision.point_results}


def write_frames_csv(path: Path, decisions: Sequence[Decision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "gop_idx",
        "gop_name",
        "local_poc",
        "poc",
        "depth_frame_idx",
        "tid",
        "coding_order",
        "is_anchor",
        "reference_local_poc",
        "reference_poc",
        "temporal_distance",
        "valid_points",
        "mean_motion_px",
        "rms_motion_px",
        "max_motion_px",
        "metric_px",
        "metric_norm",
        "tool_enabled",
        "reference_eligible",
        "reason",
    ]
    for name in POINT_NAMES:
        fields += [
            f"{name}_depth",
            f"{name}_ref_x",
            f"{name}_ref_y",
            f"{name}_motion_px",
            f"{name}_motion_norm",
            f"{name}_valid",
        ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for d in decisions:
            row = {
                "gop_idx": d.frame.gop_idx,
                "gop_name": d.frame.gop_name,
                "local_poc": d.frame.local_poc,
                "poc": d.frame.poc,
                "depth_frame_idx": d.frame.depth_frame_idx,
                "tid": d.tid,
                "coding_order": d.coding_order,
                "is_anchor": int(d.is_anchor),
                "reference_local_poc": d.reference_local_poc,
                "reference_poc": d.reference_poc,
                "temporal_distance": d.temporal_distance,
                "valid_points": d.valid_points,
                "mean_motion_px": d.mean_motion_px,
                "rms_motion_px": d.rms_motion_px,
                "max_motion_px": d.max_motion_px,
                "metric_px": d.metric_px,
                "metric_norm": d.metric_norm,
                "tool_enabled": int(d.tool_enabled),
                "reference_eligible": int(d.reference_eligible),
                "reason": d.reason,
            }

            pm = point_map(d)
            for name in POINT_NAMES:
                p = pm.get(name)
                row[f"{name}_depth"] = None if p is None else p["depth"]
                row[f"{name}_ref_x"] = None if p is None else p["ref_x"]
                row[f"{name}_ref_y"] = None if p is None else p["ref_y"]
                row[f"{name}_motion_px"] = (
                    None if p is None else p["motion_px"]
                )
                row[f"{name}_motion_norm"] = (
                    None if p is None else p["motion_norm"]
                )
                row[f"{name}_valid"] = (
                    0 if p is None else int(p["valid"])
                )

            w.writerow(row)


def write_mask_jsonl(
    path: Path,
    header: dict,
    decisions: Sequence[Decision],
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        out_header = {
            "type": "header",
            "format": "camproj_ra_simulation_mask_v1",
            "source_sequence": header.get("sequence_name"),
            "width": int(header["width"]),
            "height": int(header["height"]),
            "pose_mode": header["pose_mode"],
            "metric": args.metric,
            "offset_threshold": args.offset_threshold,
            "normalization": "(width + height) / 2",
            "min_valid_points": args.min_valid_points,
            "tie_break": args.tie_break,
            "allow_anchor_off": bool(args.allow_anchor_off),
        }
        f.write(json.dumps(out_header, ensure_ascii=False) + "\n")

        for d in decisions:
            rec = {
                "type": "frame",
                "gop_idx": d.frame.gop_idx,
                "gop_name": d.frame.gop_name,
                "local_poc": d.frame.local_poc,
                "poc": d.frame.poc,
                "tid": d.tid,
                "reference_local_poc": d.reference_local_poc,
                "reference_poc": d.reference_poc,
                "metric_norm": d.metric_norm,
                "tool_enabled": d.tool_enabled,
                "reference_eligible": d.reference_eligible,
                "reason": d.reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_summary(
    header: dict,
    decisions: Sequence[Decision],
    args,
) -> dict:
    by_gop: Dict[int, List[Decision]] = {}
    for d in decisions:
        by_gop.setdefault(d.frame.gop_idx, []).append(d)

    gop_summary = {}
    for gop_idx, ds in sorted(by_gop.items()):
        non_anchor = [d for d in ds if not d.is_anchor]
        off = [d for d in ds if not d.tool_enabled]
        evaluated = [d for d in ds if d.metric_norm is not None]

        gop_summary[str(gop_idx)] = {
            "gop_name": ds[0].frame.gop_name,
            "picture_count": len(ds),
            "non_anchor_picture_count": len(non_anchor),
            "tool_off_count": len(off),
            "tool_off_ratio_all": len(off) / len(ds) if ds else 0.0,
            "tool_off_ratio_non_anchor": (
                len(off) / len(non_anchor) if non_anchor else 0.0
            ),
            "reference_excluded_count": sum(
                not d.reference_eligible for d in ds
            ),
            "mean_metric_norm": (
                float(np.mean([d.metric_norm for d in evaluated]))
                if evaluated
                else None
            ),
            "max_metric_norm": (
                float(np.max([d.metric_norm for d in evaluated]))
                if evaluated
                else None
            ),
        }

    non_anchor_all = [d for d in decisions if not d.is_anchor]
    off_all = [d for d in decisions if not d.tool_enabled]

    return {
        "format": "camproj_ra_simulation_summary_v1",
        "sequence_name": header.get("sequence_name"),
        "camera_json_pose_mode": header["pose_mode"],
        "width": int(header["width"]),
        "height": int(header["height"]),
        "normalization_denominator": (
            int(header["width"]) + int(header["height"])
        ) / 2.0,
        "metric": args.metric,
        "offset_threshold": args.offset_threshold,
        "min_valid_points": args.min_valid_points,
        "corner_inset": args.corner_inset,
        "tie_break": args.tie_break,
        "allow_anchor_off": bool(args.allow_anchor_off),
        "total_picture_count": len(decisions),
        "total_non_anchor_picture_count": len(non_anchor_all),
        "total_tool_off_count": len(off_all),
        "tool_off_ratio_all": (
            len(off_all) / len(decisions) if decisions else 0.0
        ),
        "tool_off_ratio_non_anchor": (
            len(off_all) / len(non_anchor_all)
            if non_anchor_all
            else 0.0
        ),
        "gops": gop_summary,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera-jsonl", required=True)
    ap.add_argument(
        "--depth-yuv",
        default=None,
        help=(
            "Merged depth YUV420p10le. If omitted, use the depth_yuv "
            "filename in the JSONL header relative to the JSONL directory."
        ),
    )
    ap.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix. Default: <camera-jsonl-stem>_ra_sim"
        ),
    )
    ap.add_argument(
        "--offset-threshold",
        type=float,
        required=True,
        help=(
            "Normalized motion threshold. Tool is off when metric_norm "
            "is strictly smaller than this value."
        ),
    )
    ap.add_argument(
        "--metric",
        choices=["mean", "rms", "max"],
        default="mean",
        help="Aggregation of the valid five-point pixel motions.",
    )
    ap.add_argument(
        "--min-valid-points",
        type=int,
        default=3,
        help=(
            "Minimum valid projected points required to turn the tool off. "
            "Otherwise the picture is conservatively kept on."
        ),
    )
    ap.add_argument(
        "--corner-inset",
        type=int,
        default=0,
        help=(
            "Move corner samples inward by this many pixels. Useful when "
            "corner depth is often invalid."
        ),
    )
    ap.add_argument(
        "--tie-break",
        choices=["lower_poc", "higher_poc", "lower_motion"],
        default="lower_motion",
        help=(
            "How to choose between equidistant lower-TID references."
        ),
    )
    ap.add_argument(
        "--allow-anchor-off",
        action="store_true",
        help=(
            "Also evaluate first/last GOP anchor pictures. By default they "
            "remain reference-eligible."
        ),
    )

    args = ap.parse_args()

    if args.offset_threshold < 0:
        raise ValueError("--offset-threshold must be nonnegative")
    if not (1 <= args.min_valid_points <= 5):
        raise ValueError("--min-valid-points must be in [1, 5]")
    if args.corner_inset < 0:
        raise ValueError("--corner-inset must be nonnegative")

    camera_path = Path(args.camera_jsonl)
    header, frames = read_camera_jsonl(camera_path)

    width = int(header["width"])
    height = int(header["height"])
    pose_mode = str(header["pose_mode"])

    if args.depth_yuv:
        depth_path = Path(args.depth_yuv)
    else:
        depth_name = header.get("depth_yuv")
        if not depth_name:
            raise ValueError(
                "--depth-yuv omitted and header has no depth_yuv field"
            )
        depth_path = camera_path.parent / depth_name

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = camera_path.with_name(
            camera_path.stem + "_ra_sim"
        )

    grouped: Dict[int, List[CameraFrame]] = {}
    for fr in frames:
        grouped.setdefault(fr.gop_idx, []).append(fr)

    all_decisions: List[Decision] = []

    with DepthYUV420P10LE(depth_path, width, height) as depth_reader:
        for gop_idx, gop_frames in sorted(grouped.items()):
            decisions = simulate_gop(
                frames=sorted(gop_frames, key=lambda x: x.local_poc),
                pose_mode=pose_mode,
                depth_reader=depth_reader,
                width=width,
                height=height,
                threshold=args.offset_threshold,
                metric_name=args.metric,
                min_valid_points=args.min_valid_points,
                inset=args.corner_inset,
                tie_break=args.tie_break,
                allow_anchor_off=args.allow_anchor_off,
            )
            all_decisions.extend(decisions)

            off = sum(not d.tool_enabled for d in decisions)
            print(
                f"[GOP {gop_idx}] pictures={len(decisions)}, "
                f"tool_off={off}, ratio={off / len(decisions):.4f}"
            )

    csv_path = Path(str(prefix) + "_frames.csv")
    summary_path = Path(str(prefix) + "_summary.json")
    mask_path = Path(str(prefix) + "_mask.jsonl")

    write_frames_csv(csv_path, all_decisions)
    write_mask_jsonl(mask_path, header, all_decisions, args)

    summary = build_summary(header, all_decisions, args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] frames CSV : {csv_path}")
    print(f"[OK] summary     : {summary_path}")
    print(f"[OK] mask JSONL  : {mask_path}")
    print(
        f"[TOTAL] tool_off={summary['total_tool_off_count']}/"
        f"{summary['total_picture_count']} "
        f"({summary['tool_off_ratio_all']:.4%})"
    )


if __name__ == "__main__":
    main()
