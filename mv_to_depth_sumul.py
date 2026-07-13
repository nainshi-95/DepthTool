#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VERSION: 2026-07-13-projection-y-psnr-from-input-depth-v1

Projection Y-PSNR evaluator using an externally supplied depth-map YUV.

No RDO, no depth encoding simulation, no plane fitting, no predictor generation,
no residual coding, and no probability model are used.

For each target POC:
  1) choose one or more reference POCs,
  2) read reference video Y,
  3) read reference depth Y from --input-depth-yuv,
  4) forward-project reference Y into the target camera using that depth,
  5) compare projected Y against target video Y over valid projected pixels,
  6) report frame and sequence projection Y-PSNR.

POCs listed in --exclude-pocs, e.g. 0,32, are excluded from measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


@dataclass
class Camera:
    poc: int
    K: np.ndarray
    W2C: np.ndarray
    C2W: np.ndarray
    z_sign: float


def frame_size_420p10le(width: int, height: int) -> int:
    y = width * height
    uv = (width // 2) * (height // 2)
    return 2 * (y + 2 * uv)


def count_frames(path: str, width: int, height: int) -> int:
    fs = frame_size_420p10le(width, height)
    size = os.path.getsize(path)
    trailing = size % fs
    if trailing:
        print(f"[WARN] trailing bytes ignored: {path}: {trailing}")
    return size // fs


def read_y(fp, poc: int, width: int, height: int, stored_shift: int) -> np.ndarray:
    fp.seek(poc * frame_size_420p10le(width, height))
    raw = fp.read(width * height * 2)
    if len(raw) != width * height * 2:
        raise EOFError(f"Cannot read Y plane at POC {poc}")
    y = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    if stored_shift:
        y = np.right_shift(y, stored_shift)
    return y.astype(np.float64)


def rt4(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def load_cameras(path: str) -> Tuple[Dict[str, Any], Dict[int, Camera]]:
    header: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") in ("header", "intrinsic"):
                header = obj
            elif "poc" in obj:
                records.append(obj)

    if header is None or not records:
        raise RuntimeError("Invalid camera JSONL")

    records.sort(key=lambda x: int(x["poc"]))
    intr = header["intrinsic"]
    base = np.array(
        [intr["fx"], intr["fy"], intr["cx"], intr["cy"]],
        dtype=np.float64,
    )
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )
    z_sign = 1.0 if float(intr.get("z_sign", 1.0)) >= 0.0 else -1.0
    pose_mode = str(header.get("pose_mode", "current_to_previous"))

    current_intrinsic = base.copy()
    previous_w2c = np.eye(4, dtype=np.float64)
    cameras: Dict[int, Camera] = {}

    for order, record in enumerate(records):
        poc = int(record["poc"])
        delta = np.asarray(
            record.get("intrinsic_delta", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        current_intrinsic = (
            base.copy() if fixed_intrinsic else current_intrinsic + delta
        )
        K = np.array(
            [
                [current_intrinsic[0], 0.0, current_intrinsic[2]],
                [0.0, current_intrinsic[1], current_intrinsic[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        Trec = rt4(record["rvec"], record["tvec"])
        if pose_mode == "current_to_previous":
            W2C = (
                np.eye(4, dtype=np.float64)
                if order == 0
                else np.linalg.inv(Trec) @ previous_w2c
            )
        elif pose_mode in ("gop_local", "absolute"):
            W2C = Trec
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")

        C2W = np.linalg.inv(W2C)
        cameras[poc] = Camera(poc, K, W2C, C2W, z_sign)
        previous_w2c = W2C

    return header, cameras


def get_depth_scale_real(header: Dict[str, Any]) -> float:
    if "depth_scale_precision" in header:
        precision = float(header["depth_scale_precision"])
        if precision <= 0.0:
            raise ValueError("depth_scale_precision must be positive")
        return float(header["depth_scale"]) / precision
    if "depth_scale_real" in header:
        return float(header["depth_scale_real"])
    return float(header["depth_scale"])


def parse_poc_set(text: str) -> Set[int]:
    result: Set[int] = set()
    text = str(text).strip()
    if not text:
        return result

    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            if lo < 0 or hi < lo:
                raise ValueError(f"Invalid POC range: {token}")
            result.update(range(lo, hi + 1))
        else:
            poc = int(token)
            if poc < 0:
                raise ValueError("POCs must be non-negative")
            result.add(poc)
    return result


def load_refs_from_mv_csv(path: str, total_frames: int) -> List[List[int]]:
    refs: List[List[int]] = [[] for _ in range(total_frames)]
    if not path:
        return refs

    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        required = {"poc", "ref_poc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"MV CSV missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            try:
                poc = int(row["poc"])
                ref_poc = int(row["ref_poc"])
            except Exception as exc:
                raise RuntimeError(f"Invalid MV CSV row {line_no}: {row}") from exc

            if 0 <= poc < total_frames and ref_poc != poc:
                if ref_poc not in refs[poc]:
                    refs[poc].append(ref_poc)
    return refs


def build_ra_order(start: int, end: int, gop: int) -> List[int]:
    if end <= start:
        return []

    order = [start]
    seen = {start}

    def add_midpoints(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        if mid in seen or mid <= lo or mid >= hi:
            return
        order.append(mid)
        seen.add(mid)
        add_midpoints(lo, mid)
        add_midpoints(mid, hi)

    lo = start
    last = end - 1
    while lo < last:
        hi = min(lo + gop, last)
        if hi not in seen:
            order.append(hi)
            seen.add(hi)
        add_midpoints(lo, hi)
        lo = hi

    if sorted(order) != list(range(start, end)):
        raise RuntimeError("RA order generation failed")
    return order


def forward_project_y(
    reference_y: np.ndarray,
    reference_depth_y: np.ndarray,
    reference_camera: Camera,
    target_camera: Camera,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-project reference Y to target with nearest-pixel z-buffering."""
    height, width = reference_y.shape

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )

    depth = reference_depth_y * depth_scale
    depth_valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )

    rays = np.stack(
        [
            (x - reference_camera.K[0, 2]) / reference_camera.K[0, 0],
            (y - reference_camera.K[1, 2]) / reference_camera.K[1, 1],
            np.full_like(x, reference_camera.z_sign),
        ],
        axis=-1,
    )
    X_ref = rays * depth[..., None]

    M = target_camera.W2C @ reference_camera.C2W
    X_tar = X_ref @ M[:3, :3].T + M[:3, 3]

    tar_depth = target_camera.z_sign * X_tar[..., 2]
    front = depth_valid & np.isfinite(tar_depth) & (tar_depth > 1e-10)
    safe_depth = np.where(front, tar_depth, 1.0)

    u = target_camera.K[0, 0] * X_tar[..., 0] / safe_depth + target_camera.K[0, 2]
    v = target_camera.K[1, 1] * X_tar[..., 1] / safe_depth + target_camera.K[1, 2]

    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)

    valid = (
        front
        & np.isfinite(u)
        & np.isfinite(v)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )

    projected = np.zeros((height, width), dtype=np.float64)
    projected_valid = np.zeros((height, width), dtype=bool)

    if not np.any(valid):
        return projected, projected_valid

    dst_index = vi[valid] * width + ui[valid]
    src_depth = tar_depth[valid]
    src_y = reference_y[valid]

    order = np.argsort(src_depth)
    dst_index = dst_index[order]
    src_y = src_y[order]

    unique_index, first = np.unique(dst_index, return_index=True)
    projected.reshape(-1)[unique_index] = src_y[first]
    projected_valid.reshape(-1)[unique_index] = True

    return projected, projected_valid


def combine_projections(
    projections: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros_like(projections[0][0], dtype=np.float64)
    count = np.zeros_like(total, dtype=np.float64)

    for projected, valid in projections:
        total[valid] += projected[valid]
        count[valid] += 1.0

    valid = count > 0.0
    out = np.zeros_like(total)
    out[valid] = total[valid] / count[valid]
    return out, valid


def calculate_metrics(
    target_y: np.ndarray,
    projected_y: np.ndarray,
    valid: np.ndarray,
    peak: float,
) -> Dict[str, float]:
    valid_pixels = int(np.count_nonzero(valid))
    total_pixels = int(valid.size)

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "valid_ratio": 0.0,
            "sse": 0.0,
            "mse": 0.0,
            "psnr": float("nan"),
            "mae": 0.0,
            "max_abs_error": 0.0,
        }

    diff = target_y[valid] - projected_y[valid]
    sse = float(np.sum(diff * diff, dtype=np.float64))
    mse = sse / valid_pixels
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)

    return {
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_pixels / total_pixels,
        "sse": sse,
        "mse": mse,
        "psnr": psnr,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure projection Y-PSNR using an input depth-map YUV directly."
    )
    p.add_argument("--video-yuv", required=True)
    p.add_argument("--input-depth-yuv", required=True)
    p.add_argument("--camera-param", required=True)
    p.add_argument("--mv-csv", default="")

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)

    p.add_argument("--exclude-pocs", default="0,32")
    p.add_argument("--coding-order", choices=["ra", "sequential"], default="ra")
    p.add_argument("--ra-gop-size", type=int, default=32)
    p.add_argument("--default-ref-offset", type=int, default=1)
    p.add_argument("--max-refs", type=int, default=2)
    p.add_argument(
        "--reference-mode",
        choices=["first", "average"],
        default="average",
    )

    p.add_argument("--video-stored-bit-shift", type=int, choices=[0, 6], default=0)
    p.add_argument("--depth-stored-bit-shift", type=int, choices=[0, 6], default=0)
    p.add_argument("--peak-value", type=float, default=1023.0)
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--max-depth", type=float, default=1e9)
    p.add_argument("--min-valid-ratio", type=float, default=0.0)

    p.add_argument("--out-frame-csv", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError("Invalid YUV420 resolution")
    if a.start_frame < 0 or a.num_frames < 0:
        raise ValueError("Invalid frame range")
    if a.ra_gop_size <= 0 or a.default_ref_offset <= 0 or a.max_refs <= 0:
        raise ValueError("Invalid coding/reference configuration")
    if not (0.0 <= a.min_valid_ratio <= 1.0):
        raise ValueError("--min-valid-ratio must be in [0,1]")

    outputs = [Path(a.out_frame_csv), Path(a.out_summary_json)]
    for path in outputs:
        if path.exists():
            if not a.overwrite:
                raise FileExistsError(f"Output exists: {path}")
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    header, cameras = load_cameras(a.camera_param)
    depth_scale = get_depth_scale_real(header)

    video_frames = count_frames(a.video_yuv, a.width, a.height)
    depth_frames = count_frames(a.input_depth_yuv, a.width, a.height)
    total_frames = min(video_frames, depth_frames)
    if video_frames != depth_frames:
        print(
            f"[WARN] frame-count mismatch: video={video_frames}, "
            f"depth={depth_frames}, using={total_frames}"
        )

    start = a.start_frame
    end = total_frames if a.num_frames == 0 else min(total_frames, start + a.num_frames)
    if start >= end:
        raise ValueError("Invalid frame range")

    coding_order = (
        build_ra_order(start, end, a.ra_gop_size)
        if a.coding_order == "ra"
        else list(range(start, end))
    )
    excluded = parse_poc_set(a.exclude_pocs)
    refs_from_csv = load_refs_from_mv_csv(a.mv_csv, total_frames)

    fields = [
        "poc", "excluded", "measured", "reference_pocs", "reference_mode",
        "valid_pixels", "valid_ratio", "sse_y", "mse_y", "psnr_y",
        "mae_y", "max_abs_error_y", "skip_reason",
    ]

    frame_rows: List[Dict[str, Any]] = []
    total_sse = 0.0
    total_valid_pixels = 0
    measured_frames = 0

    with open(a.video_yuv, "rb") as video_fp, open(a.input_depth_yuv, "rb") as depth_fp, open(
        a.out_frame_csv, "w", newline="", encoding="utf-8"
    ) as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=fields)
        writer.writeheader()

        for order_idx, poc in enumerate(coding_order, start=1):
            row: Dict[str, Any] = {
                "poc": poc,
                "excluded": int(poc in excluded),
                "measured": 0,
                "reference_pocs": "",
                "reference_mode": a.reference_mode,
                "valid_pixels": 0,
                "valid_ratio": 0.0,
                "sse_y": 0.0,
                "mse_y": 0.0,
                "psnr_y": "",
                "mae_y": 0.0,
                "max_abs_error_y": 0.0,
                "skip_reason": "",
            }

            if poc in excluded:
                row["skip_reason"] = "excluded_poc"
            elif poc not in cameras:
                row["skip_reason"] = "missing_target_camera"
            else:
                refs: List[int] = []
                for ref in refs_from_csv[poc]:
                    if (
                        0 <= ref < total_frames
                        and ref in cameras
                        and ref != poc
                        and ref not in refs
                    ):
                        refs.append(ref)

                fallback = poc - a.default_ref_offset
                if not refs and 0 <= fallback < total_frames and fallback in cameras:
                    refs.append(fallback)

                refs = refs[:a.max_refs]
                if a.reference_mode == "first":
                    refs = refs[:1]
                row["reference_pocs"] = "|".join(map(str, refs))

                if not refs:
                    row["skip_reason"] = "no_reference"
                else:
                    target_y = read_y(
                        video_fp, poc, a.width, a.height, a.video_stored_bit_shift
                    )
                    projections: List[Tuple[np.ndarray, np.ndarray]] = []

                    for ref in refs:
                        ref_y = read_y(
                            video_fp, ref, a.width, a.height, a.video_stored_bit_shift
                        )
                        ref_depth_y = read_y(
                            depth_fp, ref, a.width, a.height, a.depth_stored_bit_shift
                        )
                        projections.append(
                            forward_project_y(
                                ref_y,
                                ref_depth_y,
                                cameras[ref],
                                cameras[poc],
                                depth_scale,
                                a.min_depth,
                                a.max_depth,
                            )
                        )

                    if a.reference_mode == "average" and len(projections) > 1:
                        projected_y, valid = combine_projections(projections)
                    else:
                        projected_y, valid = projections[0]

                    metrics = calculate_metrics(target_y, projected_y, valid, a.peak_value)
                    row.update(
                        {
                            "valid_pixels": metrics["valid_pixels"],
                            "valid_ratio": metrics["valid_ratio"],
                            "sse_y": metrics["sse"],
                            "mse_y": metrics["mse"],
                            "psnr_y": metrics["psnr"],
                            "mae_y": metrics["mae"],
                            "max_abs_error_y": metrics["max_abs_error"],
                        }
                    )

                    if metrics["valid_pixels"] == 0:
                        row["skip_reason"] = "no_valid_projection"
                    elif metrics["valid_ratio"] < a.min_valid_ratio:
                        row["skip_reason"] = "below_min_valid_ratio"
                    else:
                        row["measured"] = 1
                        measured_frames += 1
                        total_sse += metrics["sse"]
                        total_valid_pixels += int(metrics["valid_pixels"])

            writer.writerow(row)
            frame_rows.append(row)

            progress = order_idx / len(coding_order)
            bar_width = 30
            filled = int(round(bar_width * progress))
            if row["measured"]:
                status = f"PSNR={float(row['psnr_y']):.4f}"
            else:
                status = row["skip_reason"]
            print(
                f"\r[{'#' * filled}{'-' * (bar_width - filled)}] "
                f"{order_idx}/{len(coding_order)} POC={poc} {status}",
                end="",
                flush=True,
            )
    print()

    overall_mse = total_sse / total_valid_pixels if total_valid_pixels > 0 else 0.0
    overall_psnr = (
        float("nan")
        if total_valid_pixels == 0
        else float("inf")
        if overall_mse == 0.0
        else 10.0 * math.log10((a.peak_value * a.peak_value) / overall_mse)
    )

    frame_psnrs = [float(r["psnr_y"]) for r in frame_rows if r["measured"]]
    mean_frame_psnr = float(np.mean(frame_psnrs)) if frame_psnrs else float("nan")

    summary = {
        "version": "2026-07-13-projection-y-psnr-from-input-depth-v1",
        "metric": "projection Y PSNR",
        "rdo_simulation": False,
        "encoding_simulation": False,
        "plane_fitting": False,
        "depth_source": "--input-depth-yuv used directly as reference depth",
        "video_yuv": a.video_yuv,
        "input_depth_yuv": a.input_depth_yuv,
        "camera_param": a.camera_param,
        "mv_csv": a.mv_csv,
        "width": a.width,
        "height": a.height,
        "start_frame": start,
        "end_frame_exclusive": end,
        "coding_order": a.coding_order,
        "coding_poc_order": coding_order,
        "ra_gop_size": a.ra_gop_size,
        "excluded_pocs": sorted(excluded),
        "reference_mode": a.reference_mode,
        "default_ref_offset": a.default_ref_offset,
        "max_refs": a.max_refs,
        "depth_scale_real": depth_scale,
        "peak_value": a.peak_value,
        "measured_frame_count": measured_frames,
        "aggregate_valid_pixels": total_valid_pixels,
        "overall_sse_y": total_sse,
        "overall_mse_y": overall_mse,
        "overall_projection_psnr_y": overall_psnr,
        "mean_frame_projection_psnr_y": mean_frame_psnr,
        "overall_metric_scope": "valid projected Y pixels of non-excluded measured frames",
        "frames": frame_rows,
        "out_frame_csv": a.out_frame_csv,
    }

    with open(a.out_summary_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(f"Measured frames               : {measured_frames}")
    print(f"Aggregate valid pixels        : {total_valid_pixels}")
    print(f"Overall projection MSE-Y      : {overall_mse:.12f}")
    print(f"Overall projection PSNR-Y     : {overall_psnr:.6f} dB")
    print(f"Mean frame projection PSNR-Y  : {mean_frame_psnr:.6f} dB")
    print(f"Frame CSV                     : {a.out_frame_csv}")
    print(f"Summary JSON                  : {a.out_summary_json}")


if __name__ == "__main__":
    main()
