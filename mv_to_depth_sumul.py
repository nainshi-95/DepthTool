#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VERSION: 2026-07-13-depth-y-psnr-only

Compare GT and externally reconstructed depth YUV420p10le sequences in the
stored 10-bit Y-sample domain. No RDO, encoding simulation, plane fitting,
motion prediction, or forward warping is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


def frame_size(width: int, height: int) -> int:
    # YUV420p10le: every Y/U/V sample occupies 2 bytes.
    return width * height * 3


def count_frames(path: str, width: int, height: int) -> int:
    fs = frame_size(width, height)
    size = os.path.getsize(path)
    if size % fs:
        print(f"[WARN] {path}: trailing bytes ignored: {size % fs}")
    return size // fs


def read_y(fp, frame_idx: int, width: int, height: int, stored_bit_shift: int) -> np.ndarray:
    fp.seek(frame_idx * frame_size(width, height))
    raw = fp.read(width * height * 2)
    if len(raw) != width * height * 2:
        raise EOFError(f"Cannot read Y plane of frame {frame_idx}")
    y = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    if stored_bit_shift:
        y = np.right_shift(y, stored_bit_shift)
    return y.astype(np.float64)


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
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError(f"Invalid POC range: {token}")
            result.update(range(lo, hi + 1))
        else:
            poc = int(token)
            if poc < 0:
                raise ValueError("POCs must be non-negative")
            result.add(poc)
    return result


def metric_from_sse(sse: float, samples: int, peak: float) -> tuple[float, float]:
    if samples <= 0:
        return 0.0, float("inf")
    mse = sse / samples
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((peak * peak) / mse)
    return mse, psnr


def prepare_output(path_text: str, overwrite: bool) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Direct Y-plane PSNR comparison for depth YUV420p10le"
    )
    p.add_argument("--gt-depth-yuv", required=True)
    p.add_argument("--recon-depth-yuv", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0,
                   help="0 means all mutually available frames")
    p.add_argument("--exclude-pocs", default="",
                   help="Excluded POCs/ranges, e.g. '0,32' or '0,32,64-96'")
    p.add_argument("--gt-stored-bit-shift", type=int, choices=[0, 6], default=0)
    p.add_argument("--recon-stored-bit-shift", type=int, choices=[0, 6], default=0)
    p.add_argument("--peak-value", type=float, default=1023.0)
    p.add_argument("--out-frame-csv", default="")
    p.add_argument("--out-summary-json", default="")
    p.add_argument("--include-excluded-rows", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError("YUV420 width and height must be positive even numbers")
    if a.start_frame < 0 or a.num_frames < 0:
        raise ValueError("Invalid frame range")
    if a.peak_value <= 0:
        raise ValueError("--peak-value must be positive")

    gt_count = count_frames(a.gt_depth_yuv, a.width, a.height)
    recon_count = count_frames(a.recon_depth_yuv, a.width, a.height)
    common_count = min(gt_count, recon_count)
    if gt_count != recon_count:
        print(f"[WARN] frame-count mismatch: GT={gt_count}, recon={recon_count}; using {common_count}")

    start = a.start_frame
    if start >= common_count:
        raise ValueError(f"start frame {start} is outside common frame count {common_count}")
    end = common_count if a.num_frames == 0 else min(common_count, start + a.num_frames)
    excluded = parse_poc_set(a.exclude_pocs)

    csv_path = prepare_output(a.out_frame_csv, a.overwrite)
    json_path = prepare_output(a.out_summary_json, a.overwrite)

    fields = [
        "poc", "excluded", "num_y_pixels", "sse_y", "mse_y", "psnr_y",
        "mae_y", "max_abs_error_y"
    ]
    rows: List[Dict[str, object]] = []
    total_sse = 0.0
    included_pixels = 0
    included_frames = 0
    excluded_frames = 0

    csv_fp = open(csv_path, "w", newline="", encoding="utf-8") if csv_path else None
    writer = csv.DictWriter(csv_fp, fieldnames=fields) if csv_fp else None
    if writer:
        writer.writeheader()

    try:
        with open(a.gt_depth_yuv, "rb") as gt_fp, open(a.recon_depth_yuv, "rb") as rec_fp:
            n = end - start
            for order, poc in enumerate(range(start, end), 1):
                is_excluded = poc in excluded
                if is_excluded and not a.include_excluded_rows:
                    excluded_frames += 1
                else:
                    gt_y = read_y(gt_fp, poc, a.width, a.height, a.gt_stored_bit_shift)
                    rec_y = read_y(rec_fp, poc, a.width, a.height, a.recon_stored_bit_shift)
                    diff = gt_y - rec_y
                    abs_diff = np.abs(diff)
                    sse = float(np.sum(diff * diff, dtype=np.float64))
                    pixels = a.width * a.height
                    mse, psnr = metric_from_sse(sse, pixels, a.peak_value)
                    row = {
                        "poc": poc,
                        "excluded": int(is_excluded),
                        "num_y_pixels": pixels,
                        "sse_y": sse,
                        "mse_y": mse,
                        "psnr_y": psnr,
                        "mae_y": float(np.mean(abs_diff)),
                        "max_abs_error_y": float(np.max(abs_diff)),
                    }
                    if writer:
                        writer.writerow(row)
                    if is_excluded:
                        excluded_frames += 1
                    else:
                        rows.append(row)
                        total_sse += sse
                        included_pixels += pixels
                        included_frames += 1

                ratio = order / n
                width = 30
                fill = int(round(width * ratio))
                status = "EXCLUDED" if is_excluded else f"PSNR-Y={psnr:.4f} dB"
                print(f"\r[{'#'*fill}{'-'*(width-fill)}] {order}/{n} POC={poc} {status}",
                      end="", flush=True)
    finally:
        if csv_fp:
            csv_fp.close()
    print()

    overall_mse, overall_psnr = metric_from_sse(total_sse, included_pixels, a.peak_value)
    mean_frame_psnr = float(np.mean([float(r["psnr_y"]) for r in rows])) if rows else float("inf")
    mean_frame_mae = float(np.mean([float(r["mae_y"]) for r in rows])) if rows else 0.0
    max_abs_error = float(max((float(r["max_abs_error_y"]) for r in rows), default=0.0))

    summary = {
        "metric": "direct depth-code Y-plane PSNR",
        "metric_domain": "10-bit Y code values; not physical-depth PSNR and not projected texture/video PSNR",
        "gt_depth_yuv": a.gt_depth_yuv,
        "recon_depth_yuv": a.recon_depth_yuv,
        "width": a.width,
        "height": a.height,
        "gt_frame_count": gt_count,
        "recon_frame_count": recon_count,
        "common_frame_count": common_count,
        "start_frame": start,
        "end_frame_exclusive": end,
        "included_frame_count": included_frames,
        "excluded_frame_count": excluded_frames,
        "exclude_pocs": sorted(excluded),
        "gt_stored_bit_shift": a.gt_stored_bit_shift,
        "recon_stored_bit_shift": a.recon_stored_bit_shift,
        "peak_value": a.peak_value,
        "included_y_pixel_count": included_pixels,
        "overall_sse_y": total_sse,
        "overall_mse_y": overall_mse,
        "overall_psnr_y": overall_psnr,
        "overall_psnr_definition": "10*log10(peak^2/(sum included SSE/sum included Y pixels))",
        "arithmetic_mean_frame_psnr_y": mean_frame_psnr,
        "average_frame_mae_y": mean_frame_mae,
        "maximum_abs_error_y": max_abs_error,
        "per_frame_metrics": rows,
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"GT depth YUV       : {a.gt_depth_yuv}")
    print(f"Recon depth YUV    : {a.recon_depth_yuv}")
    print(f"Frame range        : [{start}, {end})")
    print(f"Excluded POCs      : {sorted(excluded) or 'none'}")
    print(f"Included frames    : {included_frames}")
    print(f"Excluded frames    : {excluded_frames}")
    print(f"Overall SSE-Y      : {total_sse:.6f}")
    print(f"Overall MSE-Y      : {overall_mse:.12f}")
    print(f"Overall PSNR-Y     : {overall_psnr:.6f} dB")
    print(f"Mean frame PSNR-Y  : {mean_frame_psnr:.6f} dB")
    print(f"Average frame MAE-Y: {mean_frame_mae:.6f}")
    print(f"Maximum abs error-Y: {max_abs_error:.6f}")
    if csv_path:
        print(f"Frame CSV          : {csv_path}")
    if json_path:
        print(f"Summary JSON       : {json_path}")


if __name__ == "__main__":
    main()
