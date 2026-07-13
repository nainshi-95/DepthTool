#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover sparse depth maps from codec motion vectors and camera parameters.

Input MV CSV header:
    poc,x,y,w,h,list,ref_poc,mv_x,mv_y

The CSV is expected to contain one row per 4x4 motion subblock and reference.
Two rows with the same (poc,x,y,w,h) are treated as bi-prediction.

Geometry convention:
    reference_position = current_position + mv

For a current pixel p=(u,v), current camera ray d, and current->reference
camera transform X_ref = R * (z*d) + t, the observed reference pixel produces
two linear equations in the current-camera depth z. The script solves them by
least squares. For bi-prediction, both references are solved jointly when they
agree; otherwise the better-conditioned single-reference estimate is selected.

Output defaults:
  - 33 frames, 1920x1080, YUV420p10le
  - unknown/invalid depth locations remain Y=0
  - U and V are fixed to 512
  - depth_y = round(depth_real / depth_scale_real)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MotionRecord:
    poc: int
    x: int
    y: int
    w: int
    h: int
    list_name: str
    ref_poc: int
    mv_x: float
    mv_y: float


@dataclass
class DepthEstimate:
    depth: float
    residual: float
    conditioning: float
    reprojection_error: float
    valid: bool
    source: str


def progress_iter(iterable, total, desc="Processing", min_interval=0.5):
    """Yield items while printing a dependency-free progress bar to stderr."""
    start = time.perf_counter()
    last = 0.0
    total = max(int(total), 0)
    width = 32

    for idx, item in enumerate(iterable, start=1):
        now = time.perf_counter()
        if idx == 1 or idx == total or now - last >= min_interval:
            elapsed = max(now - start, 1e-9)
            rate = idx / elapsed
            remaining = (total - idx) / rate if rate > 0 else float("inf")
            frac = idx / total if total > 0 else 1.0
            filled = min(width, int(round(frac * width)))
            bar = "#" * filled + "-" * (width - filled)
            eta = "--:--" if not math.isfinite(remaining) else time.strftime("%M:%S", time.gmtime(max(0.0, remaining)))
            sys.stderr.write(
                f"\r{desc} [{bar}] {idx}/{total} "
                f"({frac * 100:6.2f}%) | {rate:8.1f} groups/s | ETA {eta}"
            )
            sys.stderr.flush()
            last = now
        yield item

    if total == 0:
        sys.stderr.write(f"\r{desc} [" + "#" * width + "] 0/0 (100.00%)\n")
    else:
        elapsed = max(time.perf_counter() - start, 1e-9)
        rate = total / elapsed
        sys.stderr.write(
            f"\r{desc} [" + "#" * width + f"] {total}/{total} "
            f"(100.00%) | {rate:8.1f} groups/s | elapsed "
            f"{time.strftime('%M:%S', time.gmtime(elapsed))}\n"
        )
    sys.stderr.flush()


def rodrigues_to_matrix(rvec: Sequence[float]) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        x, y, z = r
        k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + k
    axis = r / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rt_to_4x4(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rodrigues_to_matrix(rvec)
    t[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return t


def intrinsic_vec_to_matrix(v: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in v]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid intrinsic: {v}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_camera_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header = None
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Camera JSONL parse failure at line {line_no}: {exc}") from exc
            if obj.get("type") == "header":
                if header is not None:
                    raise ValueError("Multiple camera JSONL headers found")
                header = obj
            elif "poc" in obj:
                records.append(obj)
    if header is None or not records:
        raise ValueError("Camera JSONL header or frame records not found")
    for key in ("width", "height", "depth_scale", "intrinsic", "pose_mode"):
        if key not in header:
            raise KeyError(f"Camera JSONL header is missing '{key}'")
    return header, sorted(records, key=lambda r: int(r["poc"]))


def get_depth_scale_real(header: Dict[str, Any]) -> float:
    if "depth_scale_precision" in header:
        precision = float(header["depth_scale_precision"])
        if precision <= 0.0:
            raise ValueError("depth_scale_precision must be positive")
        scale = float(header["depth_scale"]) / precision
    elif "depth_scale_real" in header:
        scale = float(header["depth_scale_real"])
    else:
        scale = float(header["depth_scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid depth_scale_real: {scale}")
    return scale


def build_camera_lookup(header: Dict[str, Any], frame_records: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    pose_mode = str(header["pose_mode"])
    intr0 = header["intrinsic"]
    base_intr = np.array([intr0["fx"], intr0["fy"], intr0["cx"], intr0["cy"]], dtype=np.float64)
    z_sign = 1.0 if float(intr0.get("z_sign", 1.0)) >= 0.0 else -1.0
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )
    pocs = [int(r["poc"]) for r in frame_records]
    if len(set(pocs)) != len(pocs):
        raise ValueError("Duplicate POC in camera JSONL")
    if pose_mode == "current_to_previous" and pocs != list(range(len(frame_records))):
        raise ValueError("current_to_previous requires consecutive local POCs 0..N-1")

    cur_intr = base_intr.copy()
    prev_w2c = np.eye(4, dtype=np.float64)
    lookup: Dict[int, Dict[str, Any]] = {}

    for order, rec in enumerate(frame_records):
        poc = int(rec["poc"])
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64).reshape(4)
        cur_intr = base_intr.copy() if fixed_intrinsic else cur_intr + delta
        k = intrinsic_vec_to_matrix(cur_intr)
        t_rec = rt_to_4x4(rec["rvec"], rec["tvec"])

        if pose_mode == "current_to_previous":
            w2c = np.eye(4, dtype=np.float64) if order == 0 else np.linalg.inv(t_rec) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            w2c = t_rec
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")

        lookup[poc] = {
            "K": k,
            "W2C": w2c,
            "C2W": np.linalg.inv(w2c),
            "z_sign": z_sign,
        }
        prev_w2c = w2c
    return lookup


def load_motion_csv(path: Path) -> List[MotionRecord]:
    required = {"poc", "x", "y", "w", "h", "list", "ref_poc", "mv_x", "mv_y"}
    records: List[MotionRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("MV CSV has no header")
        missing = required - {str(x).strip() for x in reader.fieldnames}
        if missing:
            raise KeyError(f"MV CSV missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                rec = MotionRecord(
                    poc=int(float(row["poc"])),
                    x=int(float(row["x"])),
                    y=int(float(row["y"])),
                    w=int(float(row["w"])),
                    h=int(float(row["h"])),
                    list_name=str(row["list"]).strip(),
                    ref_poc=int(float(row["ref_poc"])),
                    mv_x=float(row["mv_x"]),
                    mv_y=float(row["mv_y"]),
                )
            except Exception as exc:
                raise ValueError(f"Invalid MV CSV row at line {line_no}: {row}") from exc
            if not np.isfinite([rec.mv_x, rec.mv_y]).all():
                raise ValueError(f"Non-finite MV at CSV line {line_no}")
            records.append(rec)
    return records


def pixel_ray(u: float, v: float, cam: Dict[str, Any]) -> np.ndarray:
    k = np.asarray(cam["K"], dtype=np.float64)
    return np.array(
        [(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], cam["z_sign"]],
        dtype=np.float64,
    )


def current_to_reference_transform(cam_cur: Dict[str, Any], cam_ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(cam_ref["W2C"]) @ np.asarray(cam_cur["C2W"])
    return m[:3, :3], m[:3, 3]


def make_depth_equations(
    u_cur: float,
    v_cur: float,
    mv_x: float,
    mv_y: float,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
    mv_sign: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    u_ref = u_cur + mv_sign * mv_x
    v_ref = v_cur + mv_sign * mv_y
    d = pixel_ray(u_cur, v_cur, cam_cur)
    r, t = current_to_reference_transform(cam_cur, cam_ref)
    rd = r @ d
    k = np.asarray(cam_ref["K"], dtype=np.float64)
    du = u_ref - k[0, 2]
    dv = v_ref - k[1, 2]
    ax = du * rd[2] - k[0, 0] * rd[0]
    bx = k[0, 0] * t[0] - du * t[2]
    ay = dv * rd[2] - k[1, 1] * rd[1]
    by = k[1, 1] * t[1] - dv * t[2]
    return np.array([ax, ay]), np.array([bx, by]), float(u_ref), float(v_ref)


def solve_depth(a: np.ndarray, b: np.ndarray, min_conditioning: float) -> Optional[Tuple[float, float, float]]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    conditioning = float(np.dot(a, a))
    if not np.isfinite(conditioning) or conditioning < min_conditioning:
        return None
    depth = float(np.dot(a, b) / conditioning)
    residual = float(np.sqrt(np.mean((a * depth - b) ** 2)))
    if not np.isfinite(depth) or not np.isfinite(residual):
        return None
    return depth, residual, conditioning


def project_with_depth(u: float, v: float, depth: float, cam_cur: Dict[str, Any], cam_ref: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    d = pixel_ray(u, v, cam_cur)
    r, t = current_to_reference_transform(cam_cur, cam_ref)
    p = r @ (depth * d) + t
    z = float(cam_ref["z_sign"] * p[2])
    if not np.isfinite(z) or z <= 1e-12:
        return None
    k = np.asarray(cam_ref["K"], dtype=np.float64)
    pu = float(k[0, 0] * p[0] / z + k[0, 2])
    pv = float(k[1, 1] * p[1] / z + k[1, 2])
    return (pu, pv) if np.isfinite([pu, pv]).all() else None


def estimate_single(
    rec: MotionRecord,
    u: float,
    v: float,
    cam_cur: Dict[str, Any],
    cam_ref: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Optional[DepthEstimate], np.ndarray, np.ndarray]:
    a, b, u_obs, v_obs = make_depth_equations(
        u, v, rec.mv_x * args.mv_scale, rec.mv_y * args.mv_scale,
        cam_cur, cam_ref, args.mv_sign,
    )
    solved = solve_depth(a, b, args.min_conditioning)
    if solved is None:
        return None, a, b
    depth, residual, conditioning = solved
    p = project_with_depth(u, v, depth, cam_cur, cam_ref)
    reproj = float("inf") if p is None else math.hypot(p[0] - u_obs, p[1] - v_obs)
    valid = (
        np.isfinite(depth)
        and args.min_depth <= depth <= args.max_depth
        and np.isfinite(reproj)
        and reproj <= args.max_reprojection_error
    )
    return DepthEstimate(depth, residual, conditioning, reproj, bool(valid), f"ref_{rec.ref_poc}"), a, b


def estimate_depth_for_pixel(
    records: Sequence[MotionRecord],
    u: float,
    v: float,
    camera_lookup: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
) -> Optional[DepthEstimate]:
    if not records:
        return None
    poc = records[0].poc
    cam_cur = camera_lookup.get(poc)
    if cam_cur is None:
        return None

    singles: List[DepthEstimate] = []
    all_a: List[np.ndarray] = []
    all_b: List[np.ndarray] = []
    usable: List[MotionRecord] = []

    for rec in records:
        cam_ref = camera_lookup.get(rec.ref_poc)
        if cam_ref is None or rec.ref_poc == poc:
            continue
        est, a, b = estimate_single(rec, u, v, cam_cur, cam_ref, args)
        all_a.append(a)
        all_b.append(b)
        usable.append(rec)
        if est is not None and est.valid:
            singles.append(est)

    if not all_a:
        return None

    joint = None
    solved = solve_depth(np.concatenate(all_a), np.concatenate(all_b), args.min_conditioning)
    if solved is not None:
        depth, residual, conditioning = solved
        errors = []
        for rec in usable:
            p = project_with_depth(u, v, depth, cam_cur, camera_lookup[rec.ref_poc])
            if p is None:
                errors.append(float("inf"))
            else:
                u_obs = u + args.mv_sign * rec.mv_x * args.mv_scale
                v_obs = v + args.mv_sign * rec.mv_y * args.mv_scale
                errors.append(math.hypot(p[0] - u_obs, p[1] - v_obs))
        reproj = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
        joint = DepthEstimate(
            depth, residual, conditioning, reproj,
            bool(args.min_depth <= depth <= args.max_depth and reproj <= args.max_reprojection_error),
            "joint",
        )

    if len(records) == 1:
        return singles[0] if singles else (joint if joint and joint.valid else None)

    if args.bi_mode == "joint":
        return joint if joint and joint.valid else None

    if args.bi_mode == "average":
        if not singles:
            return None
        return DepthEstimate(
            float(np.mean([s.depth for s in singles])),
            float(np.mean([s.residual for s in singles])),
            float(np.sum([s.conditioning for s in singles])),
            float(np.mean([s.reprojection_error for s in singles])),
            True,
            "bi_average",
        )

    def score(s: DepthEstimate) -> float:
        return s.conditioning / (1.0 + s.reprojection_error ** 2 + s.residual ** 2)

    best = max(singles, key=score) if singles else None
    if args.bi_mode == "best":
        return best

    if len(singles) >= 2:
        ds = np.array([s.depth for s in singles], dtype=np.float64)
        rel_spread = float((np.max(ds) - np.min(ds)) / max(np.mean(np.abs(ds)), 1e-12))
        if rel_spread <= args.bi_relative_threshold and joint is not None and joint.valid:
            joint.source = "robust_joint"
            return joint
        return best

    if len(singles) == 1:
        return singles[0]
    return joint if joint and joint.valid else None


def estimate_rows_center_vectorized(records, camera_lookup, args):
    """Vectorized single-reference depth solve at each 4x4 block center.

    Returns arrays: depth, residual, conditioning, reprojection_error, valid.
    Camera transforms are evaluated once per (poc, ref_poc) pair rather than
    once per pixel/block.
    """
    n = len(records)
    depth = np.full(n, np.nan, dtype=np.float64)
    residual = np.full(n, np.inf, dtype=np.float64)
    conditioning = np.zeros(n, dtype=np.float64)
    reproj = np.full(n, np.inf, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    pair_to_indices = defaultdict(list)
    for i, rec in enumerate(records):
        if rec.poc in camera_lookup and rec.ref_poc in camera_lookup and rec.ref_poc != rec.poc:
            pair_to_indices[(rec.poc, rec.ref_poc)].append(i)

    pair_items = list(pair_to_indices.items())
    for (poc, ref_poc), idxs in progress_iter(
        pair_items, len(pair_items), desc="Solving camera pairs"
    ):
        idx = np.asarray(idxs, dtype=np.int64)
        cam_cur = camera_lookup[poc]
        cam_ref = camera_lookup[ref_poc]
        kc = np.asarray(cam_cur["K"], dtype=np.float64)
        kr = np.asarray(cam_ref["K"], dtype=np.float64)
        m = np.asarray(cam_ref["W2C"], dtype=np.float64) @ np.asarray(cam_cur["C2W"], dtype=np.float64)
        r = m[:3, :3]
        t = m[:3, 3]
        z_sign_cur = float(cam_cur["z_sign"])
        z_sign_ref = float(cam_ref["z_sign"])

        u = np.fromiter((records[i].x + (records[i].w - 1) * 0.5 for i in idxs), dtype=np.float64)
        v = np.fromiter((records[i].y + (records[i].h - 1) * 0.5 for i in idxs), dtype=np.float64)
        mvx = np.fromiter((records[i].mv_x for i in idxs), dtype=np.float64) * args.mv_scale
        mvy = np.fromiter((records[i].mv_y for i in idxs), dtype=np.float64) * args.mv_scale
        u_obs = u + args.mv_sign * mvx
        v_obs = v + args.mv_sign * mvy

        dx = (u - kc[0, 2]) / kc[0, 0]
        dy = (v - kc[1, 2]) / kc[1, 1]
        rd0 = r[0, 0] * dx + r[0, 1] * dy + r[0, 2] * z_sign_cur
        rd1 = r[1, 0] * dx + r[1, 1] * dy + r[1, 2] * z_sign_cur
        rd2 = r[2, 0] * dx + r[2, 1] * dy + r[2, 2] * z_sign_cur

        du = u_obs - kr[0, 2]
        dv = v_obs - kr[1, 2]
        ax = du * rd2 - kr[0, 0] * rd0
        bx = kr[0, 0] * t[0] - du * t[2]
        ay = dv * rd2 - kr[1, 1] * rd1
        by = kr[1, 1] * t[1] - dv * t[2]

        cond = ax * ax + ay * ay
        good = np.isfinite(cond) & (cond >= args.min_conditioning)
        z = np.full_like(cond, np.nan)
        z[good] = (ax[good] * bx[good] + ay[good] * by[good]) / cond[good]
        res = np.full_like(cond, np.inf)
        res[good] = np.sqrt(0.5 * ((ax[good] * z[good] - bx[good]) ** 2 + (ay[good] * z[good] - by[good]) ** 2))

        xp = z * rd0 + t[0]
        yp = z * rd1 + t[1]
        zp = z * rd2 + t[2]
        front = np.isfinite(zp) & (zp * z_sign_ref > 1e-12)
        denom = np.maximum(np.abs(zp), 1e-12)
        pu = kr[0, 0] * xp / denom + kr[0, 2]
        pv = kr[1, 1] * yp / denom + kr[1, 2]
        err = np.hypot(pu - u_obs, pv - v_obs)

        ok = (
            good & front & np.isfinite(z) & np.isfinite(err)
            & (z >= args.min_depth) & (z <= args.max_depth)
            & (err <= args.max_reprojection_error)
        )
        depth[idx] = z
        residual[idx] = res
        conditioning[idx] = cond
        reproj[idx] = err
        valid[idx] = ok

    return depth, residual, conditioning, reproj, valid


def fill_center_constant_fast(records, camera_lookup, depth_real, args):
    """Fast path: solve each CSV row once, then combine uni/bi rows by block."""
    z, res, cond, err, ok = estimate_rows_center_vectorized(records, camera_lookup, args)
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec.poc, rec.x, rec.y, rec.w, rec.h)].append(i)

    source_counts = defaultdict(int)
    uni_groups = bi_groups = written_groups = invalid_groups = out_groups = 0
    items = list(groups.items())
    for (poc, x, y, w, h), idxs in progress_iter(items, len(items), desc="Filling 4x4 blocks"):
        if not (0 <= poc < args.num_frames) or w <= 0 or h <= 0:
            out_groups += 1
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(args.width, x + w), min(args.height, y + h)
        if x0 >= x1 or y0 >= y1:
            out_groups += 1
            continue
        good = [i for i in idxs if ok[i]]
        if len(idxs) == 1:
            uni_groups += 1
        else:
            bi_groups += 1
        if not good:
            invalid_groups += 1
            continue

        if len(good) == 1:
            chosen = float(z[good[0]])
            source = f"ref_{records[good[0]].ref_poc}"
        else:
            depths = np.asarray([z[i] for i in good], dtype=np.float64)
            if args.bi_mode == "average":
                chosen = float(np.mean(depths)); source = "bi_average"
            elif args.bi_mode == "best":
                scores = np.asarray([cond[i] / (1.0 + err[i] ** 2 + res[i] ** 2) for i in good])
                j = good[int(np.argmax(scores))]
                chosen = float(z[j]); source = f"ref_{records[j].ref_poc}"
            else:
                rel_spread = float((np.max(depths) - np.min(depths)) / max(np.mean(np.abs(depths)), 1e-12))
                if args.bi_mode == "joint" or rel_spread <= args.bi_relative_threshold:
                    weights = np.asarray([max(cond[i], args.min_conditioning) for i in good], dtype=np.float64)
                    chosen = float(np.sum(weights * depths) / np.sum(weights))
                    source = "robust_joint" if args.bi_mode == "robust_joint" else "joint"
                else:
                    scores = np.asarray([cond[i] / (1.0 + err[i] ** 2 + res[i] ** 2) for i in good])
                    j = good[int(np.argmax(scores))]
                    chosen = float(z[j]); source = f"ref_{records[j].ref_poc}"

        depth_real[poc, y0:y1, x0:x1] = chosen
        pixels = (y1 - y0) * (x1 - x0)
        source_counts[source] += pixels
        written_groups += 1

    return groups, source_counts, uni_groups, bi_groups, written_groups, invalid_groups, out_groups


def group_motion_records(records: Iterable[MotionRecord]) -> Dict[Tuple[int, int, int, int, int], List[MotionRecord]]:
    groups: Dict[Tuple[int, int, int, int, int], List[MotionRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.poc, rec.x, rec.y, rec.w, rec.h)].append(rec)
    return groups


def write_yuv420p10le(out_path: Path, depth_code: np.ndarray) -> None:
    n, h, w = depth_code.shape
    if w % 2 or h % 2:
        raise ValueError("YUV420 requires even width and height")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    with out_path.open("wb") as f:
        for poc in range(n):
            f.write(np.ascontiguousarray(depth_code[poc].astype("<u2")).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description="Recover sparse depth from 4x4 codec MVs and camparam_v2 JSONL")
    p.add_argument("--mv-csv", required=True)
    p.add_argument("--camera-param", required=True)
    p.add_argument("--out-yuv", required=True)
    p.add_argument("--out-summary-json", default="")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--mv-scale", type=float, default=1.0)
    p.add_argument("--mv-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    p.add_argument("--sample-mode", choices=["per_pixel", "center_constant"], default="center_constant", help="center_constant is the fast vectorized default; per_pixel is much slower")
    p.add_argument("--bi-mode", choices=["robust_joint", "joint", "average", "best"], default="robust_joint")
    p.add_argument("--bi-relative-threshold", type=float, default=0.25)
    p.add_argument("--min-conditioning", type=float, default=1e-10)
    p.add_argument("--max-reprojection-error", type=float, default=4.0)
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--max-depth", type=float, default=float("inf"))
    p.add_argument("--strict-4x4", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    mv_csv = Path(args.mv_csv)
    camera_path = Path(args.camera_param)
    out_yuv = Path(args.out_yuv)
    if out_yuv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_yuv}; use --overwrite")
    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width and height")

    header, frame_records = load_camera_jsonl(camera_path)
    if int(header["width"]) != args.width or int(header["height"]) != args.height:
        raise ValueError(
            f"Resolution mismatch: camera={header['width']}x{header['height']}, "
            f"requested={args.width}x{args.height}"
        )
    camera_lookup = build_camera_lookup(header, frame_records)
    missing = [i for i in range(args.num_frames) if i not in camera_lookup]
    if missing:
        raise ValueError(f"Camera parameters missing for POCs: {missing}")

    depth_scale_real = get_depth_scale_real(header)
    max_real_code = 1023.0 * depth_scale_real
    records = load_motion_csv(mv_csv)
    if args.strict_4x4:
        bad = next((r for r in records if r.w != 4 or r.h != 4), None)
        if bad:
            raise ValueError(f"Non-4x4 record: {bad}")

    depth_real = np.zeros((args.num_frames, args.height, args.width), dtype=np.float32)

    if args.sample_mode == "center_constant":
        (groups, source_counts, uni_groups, bi_groups, written_groups,
         invalid_groups, out_groups) = fill_center_constant_fast(
            records, camera_lookup, depth_real, args
        )
    else:
        groups = group_motion_records(records)
        source_counts: Dict[str, int] = defaultdict(int)
        uni_groups = bi_groups = written_groups = invalid_groups = out_groups = 0
        sorted_groups = sorted(groups.items())

        for (poc, x, y, w, h), group in progress_iter(
            sorted_groups,
            total=len(sorted_groups),
            desc="Recovering depth",
        ):
            if not (0 <= poc < args.num_frames) or w <= 0 or h <= 0:
                out_groups += 1
                continue
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(args.width, x + w), min(args.height, y + h)
            if x0 >= x1 or y0 >= y1:
                out_groups += 1
                continue
            usable = [r for r in group if r.ref_poc in camera_lookup and r.ref_poc != poc]
            if not usable:
                invalid_groups += 1
                continue
            if len(usable) == 1:
                uni_groups += 1
            else:
                bi_groups += 1
    
            written = 0
            if args.sample_mode == "center_constant":
                u = x + (w - 1) * 0.5
                v = y + (h - 1) * 0.5
                est = estimate_depth_for_pixel(usable, u, v, camera_lookup, args)
                if est is not None and est.valid:
                    depth_real[poc, y0:y1, x0:x1] = est.depth
                    written = (y1 - y0) * (x1 - x0)
                    source_counts[est.source] += written
            else:
                for py in range(y0, y1):
                    for px in range(x0, x1):
                        est = estimate_depth_for_pixel(usable, float(px), float(py), camera_lookup, args)
                        if est is not None and est.valid:
                            depth_real[poc, py, px] = est.depth
                            source_counts[est.source] += 1
                            written += 1
    
            if written:
                written_groups += 1
            else:
                invalid_groups += 1
    depth_code = np.zeros_like(depth_real, dtype="<u2")
    valid = np.isfinite(depth_real) & (depth_real > 0.0)
    depth_code[valid] = np.clip(np.rint(depth_real[valid] / depth_scale_real), 1, 1023).astype("<u2")
    write_yuv420p10le(out_yuv, depth_code)

    nonzero = depth_code > 0
    summary = {
        "mv_csv": str(mv_csv),
        "camera_param": str(camera_path),
        "out_yuv": str(out_yuv),
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "pose_mode": header["pose_mode"],
        "depth_scale_real": depth_scale_real,
        "maximum_representable_real_depth": max_real_code,
        "mv_convention": "ref=cur+mv" if args.mv_sign > 0 else "ref=cur-mv",
        "sample_mode": args.sample_mode,
        "bi_mode": args.bi_mode,
        "csv_rows": len(records),
        "subblock_groups": len(groups),
        "uni_groups": uni_groups,
        "bi_groups": bi_groups,
        "written_groups": written_groups,
        "invalid_groups": invalid_groups,
        "out_of_range_groups": out_groups,
        "nonzero_output_pixels": int(np.count_nonzero(nonzero)),
        "nonzero_output_ratio": float(np.mean(nonzero)),
        "source_counts": dict(source_counts),
        "output_format": "YUV420p10le; Y=depth code; U=V=512; missing=0",
    }
    if np.any(nonzero):
        vals = depth_real[nonzero]
        summary.update({
            "recovered_depth_min": float(np.min(vals)),
            "recovered_depth_mean": float(np.mean(vals)),
            "recovered_depth_max": float(np.max(vals)),
            "clipped_to_1023_pixels": int(np.count_nonzero(valid & (depth_real >= max_real_code))),
        })
    if args.out_summary_json:
        path = Path(args.out_summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(f"Output YUV          : {out_yuv}")
    print(f"Frames / size       : {args.num_frames}, {args.width}x{args.height}")
    print(f"Depth scale real    : {depth_scale_real:.12g}")
    print(f"Groups uni / bi     : {uni_groups} / {bi_groups}")
    print(f"Written / invalid   : {written_groups} / {invalid_groups}")
    print(f"Nonzero depth pixels: {np.count_nonzero(nonzero)}")


if __name__ == "__main__":
    main()
