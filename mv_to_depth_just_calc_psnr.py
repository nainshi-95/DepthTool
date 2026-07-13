#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-warp projection Y metric evaluator.

- No RDO / encoding simulation.
- Target depth is used for backward mapping into each reference.
- RA reference structure follows recursive hierarchical intervals:
  16 <- (0,32), 8 <- (0,16), 4 <- (0,8), 24 <- (16,32), etc.
- MV CSV L0/L1 references may override the generated hierarchical reference pair.
- Bidirectional result: 1:1 blend where both directions are valid, otherwise use the valid side.
- Metrics are computed only on valid pixels.
- Output YUV is finally written in display POC order.
- Excluded/unavailable frames are copied from GT.
- For warped frames, invalid luma pixels are written as zero; chroma is copied from GT.
"""
from __future__ import annotations

import argparse, csv, json, math, os, tempfile
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


@dataclass
class Frame:
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def frame_size(w: int, h: int) -> int:
    return w * h * 3


def count_frames(path: str, w: int, h: int) -> int:
    fs = frame_size(w, h)
    size = os.path.getsize(path)
    if size % fs:
        print(f"[WARN] trailing bytes ignored: {path}: {size % fs}")
    return size // fs


def read_frame(fp, poc: int, w: int, h: int, shift: int) -> Frame:
    fs = frame_size(w, h)
    fp.seek(poc * fs)
    raw = fp.read(fs)
    if len(raw) != fs:
        raise EOFError(f"Cannot read frame POC={poc}")
    a = np.frombuffer(raw, dtype="<u2")
    ny, nc = w * h, (w // 2) * (h // 2)
    y = a[:ny].reshape(h, w)
    u = a[ny:ny + nc].reshape(h // 2, w // 2)
    v = a[ny + nc:].reshape(h // 2, w // 2)
    if shift:
        y, u, v = y >> shift, u >> shift, v >> shift
    return Frame(y.astype(np.float64), u.astype(np.float64), v.astype(np.float64))


def read_y(fp, poc: int, w: int, h: int, shift: int) -> np.ndarray:
    fp.seek(poc * frame_size(w, h))
    raw = fp.read(w * h * 2)
    if len(raw) != w * h * 2:
        raise EOFError(f"Cannot read Y POC={poc}")
    y = np.frombuffer(raw, dtype="<u2").reshape(h, w)
    if shift:
        y = y >> shift
    return y.astype(np.float64)


def write_frame(fp, f: Frame, shift: int) -> None:
    def pack(x: np.ndarray) -> bytes:
        x = np.clip(np.rint(x), 0, 1023).astype(np.uint16)
        if shift:
            x = x << shift
        return np.ascontiguousarray(x.astype("<u2")).tobytes()
    fp.write(pack(f.y)); fp.write(pack(f.u)); fp.write(pack(f.v))


def rt4(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def load_cameras(path: str) -> Tuple[Dict[str, Any], Dict[int, Camera]]:
    header, records = None, []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") in ("header", "intrinsic"):
                header = o
            elif "poc" in o:
                records.append(o)
    if header is None or not records:
        raise RuntimeError("Invalid camera JSONL")

    records.sort(key=lambda x: int(x["poc"]))
    intr = header["intrinsic"]
    base = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]], np.float64)
    fixed = header.get("intrinsic_mode") == "rap_fixed" or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    z_sign = 1.0 if float(intr.get("z_sign", 1.0)) >= 0 else -1.0
    pose_mode = str(header.get("pose_mode", "current_to_previous"))

    cur = base.copy(); prev_w2c = np.eye(4, dtype=np.float64); cams = {}
    for order, rec in enumerate(records):
        poc = int(rec["poc"])
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), np.float64)
        cur = base.copy() if fixed else cur + delta
        K = np.array([[cur[0], 0, cur[2]], [0, cur[1], cur[3]], [0, 0, 1]], np.float64)
        T = rt4(rec["rvec"], rec["tvec"])
        if pose_mode == "current_to_previous":
            W2C = np.eye(4, dtype=np.float64) if order == 0 else np.linalg.inv(T) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            W2C = T
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")
        cams[poc] = Camera(poc, K, W2C, np.linalg.inv(W2C), z_sign)
        prev_w2c = W2C
    return header, cams


def depth_scale_real(header: Dict[str, Any]) -> float:
    if "depth_scale_precision" in header:
        p = float(header["depth_scale_precision"])
        if p <= 0:
            raise ValueError("depth_scale_precision must be positive")
        return float(header["depth_scale"]) / p
    return float(header.get("depth_scale_real", header["depth_scale"]))


def parse_pocs(text: str) -> Set[int]:
    out: Set[int] = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = map(int, token.split("-", 1))
            if a < 0 or b < a:
                raise ValueError(f"Invalid POC range: {token}")
            out.update(range(a, b + 1))
        else:
            v = int(token)
            if v < 0:
                raise ValueError("POCs must be non-negative")
            out.add(v)
    return out


def norm_list(v: str) -> str:
    v = str(v).strip().upper()
    if v in ("0", "L0", "LIST0", "FORWARD"):
        return "L0"
    if v in ("1", "L1", "LIST1", "BACKWARD"):
        return "L1"
    return v


def load_refs(path: str, n: int) -> Tuple[List[List[int]], List[List[int]]]:
    l0, l1 = [[] for _ in range(n)], [[] for _ in range(n)]
    if not path:
        return l0, l1
    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        rd = csv.DictReader(fp)
        if {"poc", "ref_poc"} - set(rd.fieldnames or []):
            raise RuntimeError("MV CSV needs poc, ref_poc")
        has_list = "list" in set(rd.fieldnames or [])
        for line_no, row in enumerate(rd, 2):
            try:
                poc, ref = int(row["poc"]), int(row["ref_poc"])
                li = norm_list(row["list"]) if has_list else ""
            except Exception as e:
                raise RuntimeError(f"Bad MV CSV row {line_no}: {row}") from e
            if not (0 <= poc < n) or ref == poc:
                continue
            dst = l1[poc] if li == "L1" else l0[poc] if li == "L0" else (l0[poc] if ref < poc else l1[poc])
            if ref not in dst:
                dst.append(ref)
    return l0, l1


def build_ra_order(start: int, end: int, gop: int) -> List[int]:
    order, seen = [start], {start}
    def mids(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        m = (lo + hi) // 2
        if m in seen or m <= lo or m >= hi:
            return
        order.append(m); seen.add(m); mids(lo, m); mids(m, hi)
    lo, last = start, end - 1
    while lo < last:
        hi = min(lo + gop, last)
        if hi not in seen:
            order.append(hi); seen.add(hi)
        mids(lo, hi); lo = hi
    if sorted(order) != list(range(start, end)):
        raise RuntimeError("RA order generation failed")
    return order


def build_hierarchical_ra_ref_map(
    start: int,
    end: int,
    gop: int,
) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
    """Build decoder-available hierarchical RA references for every POC.

    For a full 32-frame interval this produces, for example:
      32 <- (0, None)
      16 <- (0, 32)
       8 <- (0, 16)
       4 <- (0, 8)
      12 <- (8, 16)
      24 <- (16, 32)

    The references are the two endpoints of the recursive interval whose
    midpoint is the current POC. Both endpoints are decoded earlier in the
    coding order generated by build_ra_order().
    """
    if end <= start:
        return {}

    refs: Dict[int, Tuple[Optional[int], Optional[int]]] = {start: (None, None)}

    def assign_midpoints(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            return
        refs[mid] = (lo, hi)
        assign_midpoints(lo, mid)
        assign_midpoints(mid, hi)

    lo = start
    last = end - 1
    while lo < last:
        hi = min(lo + gop, last)

        # The right GOP anchor is decoded from the left anchor first.
        refs[hi] = (lo, None)
        assign_midpoints(lo, hi)
        lo = hi

    # Defensive fallback for a truncated range or unusual GOP length.
    for poc in range(start, end):
        refs.setdefault(poc, (poc - 1 if poc > start else None, None))

    return refs


def backward_warp(ref_y: np.ndarray, tar_depth_y: np.ndarray, cref: Camera, ctar: Camera,
                  scale: float, min_depth: float, max_depth: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = tar_depth_y.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = tar_depth_y * scale
    valid_d = np.isfinite(d) & (d >= min_depth) & (d <= max_depth)
    rays = np.stack([(x - ctar.K[0, 2]) / ctar.K[0, 0],
                     (y - ctar.K[1, 2]) / ctar.K[1, 1],
                     np.full_like(x, ctar.z_sign)], axis=-1)
    Xt = rays * d[..., None]
    M = cref.W2C @ ctar.C2W
    Xr = Xt @ M[:3, :3].T + M[:3, 3]
    zr = cref.z_sign * Xr[..., 2]
    front = valid_d & np.isfinite(zr) & (zr > 1e-10)
    safe = np.where(front, zr, 1.0)
    mx = cref.K[0, 0] * Xr[..., 0] / safe + cref.K[0, 2]
    my = cref.K[1, 1] * Xr[..., 1] / safe + cref.K[1, 2]
    valid = front & np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    warped = cv2.remap(ref_y.astype(np.float32), mx.astype(np.float32), my.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float64)
    warped[~valid] = 0
    return warped, valid


def blend(a: Optional[Tuple[np.ndarray, np.ndarray]], b: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if a is None and b is None:
        raise ValueError("No direction to blend")
    if b is None:
        return a  # type: ignore
    if a is None:
        return b
    ay, av = a; by, bv = b
    valid = av | bv; both = av & bv
    out = np.zeros_like(ay)
    out[both] = 0.5 * (ay[both] + by[both])
    out[av & ~bv] = ay[av & ~bv]
    out[bv & ~av] = by[bv & ~av]
    return out, valid


def satd4(target: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    h, w = target.shape
    r = np.where(valid, target - pred, 0.0)
    ph, pw = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    p = np.zeros((ph, pw), np.float64); p[:h, :w] = r
    H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], np.float64)
    total = 0.0
    for yy in range(0, ph, 4):
        for xx in range(0, pw, 4):
            vm = valid[yy:min(yy+4,h), xx:min(xx+4,w)]
            if vm.size == 0 or not np.any(vm):
                continue
            total += float(np.sum(np.abs(H @ p[yy:yy+4, xx:xx+4] @ H.T))) / 2.0
    return total


def metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, peak: float) -> Dict[str, float]:
    n = int(np.count_nonzero(valid)); total = valid.size
    if n == 0:
        return dict(valid_pixels=0, valid_ratio=0.0, sse=0.0, mse=0.0, psnr=float("nan"), mae=0.0, max_abs_error=0.0, satd=0.0, satd_per_valid_pixel=0.0)
    diff = gt[valid] - pred[valid]
    sse = float(np.sum(diff * diff)); mse = sse / n
    psnr = float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)
    satd = satd4(gt, pred, valid)
    return dict(valid_pixels=n, valid_ratio=n/total, sse=sse, mse=mse, psnr=psnr,
                mae=float(np.mean(np.abs(diff))), max_abs_error=float(np.max(np.abs(diff))),
                satd=satd, satd_per_valid_pixel=satd/n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backward-warp projection metrics with RA bidirectional references")
    p.add_argument("--video-yuv", required=True)
    p.add_argument("--input-depth-yuv", required=True)
    p.add_argument("--camera-param", required=True)
    p.add_argument("--mv-csv", default="")
    p.add_argument("--width", type=int, required=True); p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0); p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--exclude-pocs", default="0,32")
    p.add_argument("--coding-order", choices=["ra","sequential"], default="ra")
    p.add_argument("--ra-gop-size", type=int, default=32); p.add_argument("--default-ref-offset", type=int, default=1)
    p.add_argument("--video-stored-bit-shift", type=int, choices=[0,6], default=0)
    p.add_argument("--depth-stored-bit-shift", type=int, choices=[0,6], default=0)
    p.add_argument("--output-stored-bit-shift", type=int, choices=[0,6], default=0)
    p.add_argument("--peak-value", type=float, default=1023.0)
    p.add_argument("--min-depth", type=float, default=1e-8); p.add_argument("--max-depth", type=float, default=1e9)
    p.add_argument("--min-valid-ratio", type=float, default=0.0)
    p.add_argument("--out-warped-yuv", required=True); p.add_argument("--out-frame-csv", required=True); p.add_argument("--out-summary-json", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError("Invalid YUV420 resolution")
    if a.ra_gop_size <= 0 or a.default_ref_offset <= 0 or not (0 <= a.min_valid_ratio <= 1):
        raise ValueError("Invalid arguments")
    for p in map(Path, [a.out_warped_yuv, a.out_frame_csv, a.out_summary_json]):
        if p.exists():
            if not a.overwrite: raise FileExistsError(f"Output exists: {p}")
            p.unlink()
        p.parent.mkdir(parents=True, exist_ok=True)

    header, cams = load_cameras(a.camera_param); scale = depth_scale_real(header)
    nv, nd = count_frames(a.video_yuv, a.width, a.height), count_frames(a.input_depth_yuv, a.width, a.height)
    total = min(nv, nd)
    if nv != nd: print(f"[WARN] frame count mismatch video={nv}, depth={nd}, using={total}")
    start = a.start_frame; end = total if a.num_frames == 0 else min(total, start + a.num_frames)
    if start >= end: raise ValueError("Invalid frame range")

    order = build_ra_order(start, end, a.ra_gop_size) if a.coding_order == "ra" else list(range(start, end))
    hierarchical_refs = build_hierarchical_ra_ref_map(start, end, a.ra_gop_size)
    excluded = parse_pocs(a.exclude_pocs); l0s, l1s = load_refs(a.mv_csv, total)
    fields = ["poc","coding_order_index","excluded","measured","ref_l0","ref_l1","direction_count","valid_pixels","valid_ratio","sse_y","mse_y","psnr_y","mae_y","max_abs_error_y","satd_y","satd_per_valid_pixel","skip_reason"]
    rows: Dict[int, Dict[str, Any]] = {}; total_sse = total_satd = 0.0; total_valid = measured = 0

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="bwproj_", suffix=".yuv", delete=False) as tf:
            temp_path = tf.name
        with open(a.video_yuv,"rb") as vf, open(a.input_depth_yuv,"rb") as df, open(temp_path,"wb+") as of:
            for oi, poc in enumerate(order):
                gt = read_frame(vf, poc, a.width, a.height, a.video_stored_bit_shift)
                out = Frame(gt.y.copy(), gt.u.copy(), gt.v.copy())
                row = {k:"" for k in fields}; row.update(poc=poc,coding_order_index=oi,excluded=int(poc in excluded),measured=0,ref_l0="",ref_l1="",direction_count=0,valid_pixels=0,valid_ratio=0.0,sse_y=0.0,mse_y=0.0,psnr_y="",mae_y=0.0,max_abs_error_y=0.0,satd_y=0.0,satd_per_valid_pixel=0.0,skip_reason="")

                if poc in excluded:
                    row["skip_reason"] = "excluded_poc_gt_inserted"
                elif poc not in cams:
                    row["skip_reason"] = "missing_target_camera_gt_inserted"
                else:
                    if a.coding_order == "ra":
                        generated_l0, generated_l1 = hierarchical_refs[poc]
                    else:
                        generated_l0 = poc - a.default_ref_offset
                        generated_l1 = None

                    # Use explicit codec references from the MV CSV when present.
                    # Otherwise use the recursively generated hierarchical RA pair.
                    r0 = next(
                        (r for r in l0s[poc] if start <= r < end and r in cams and r != poc),
                        generated_l0,
                    )
                    r1 = next(
                        (r for r in l1s[poc] if start <= r < end and r in cams and r != poc),
                        generated_l1,
                    )
                    if r0 is not None and not (start <= r0 < end and r0 in cams and r0 != poc): r0 = None
                    if r1 is not None and not (start <= r1 < end and r1 in cams and r1 != poc): r1 = None
                    if r0 == r1: r1 = None
                    row["ref_l0"], row["ref_l1"] = "" if r0 is None else r0, "" if r1 is None else r1

                    if r0 is None and r1 is None:
                        row["skip_reason"] = "no_reference_gt_inserted"
                    else:
                        td = read_y(df, poc, a.width, a.height, a.depth_stored_bit_shift)
                        w0 = backward_warp(read_y(vf,r0,a.width,a.height,a.video_stored_bit_shift),td,cams[r0],cams[poc],scale,a.min_depth,a.max_depth) if r0 is not None else None
                        w1 = backward_warp(read_y(vf,r1,a.width,a.height,a.video_stored_bit_shift),td,cams[r1],cams[poc],scale,a.min_depth,a.max_depth) if r1 is not None else None
                        row["direction_count"] = int(w0 is not None) + int(w1 is not None)
                        pred, valid = blend(w0, w1)
                        m = metrics(gt.y, pred, valid, a.peak_value)
                        row.update(valid_pixels=m["valid_pixels"],valid_ratio=m["valid_ratio"],sse_y=m["sse"],mse_y=m["mse"],psnr_y=m["psnr"],mae_y=m["mae"],max_abs_error_y=m["max_abs_error"],satd_y=m["satd"],satd_per_valid_pixel=m["satd_per_valid_pixel"])
                        # For every frame where backward warping was attempted,
                        # invalid Y samples are written as zero rather than copied from GT.
                        out.y.fill(0.0)
                        out.y[valid] = np.clip(np.rint(pred[valid]), 0, 1023)

                        if m["valid_pixels"] == 0:
                            row["skip_reason"] = "no_valid_projection_zero_y"
                        elif m["valid_ratio"] < a.min_valid_ratio:
                            row["skip_reason"] = "below_min_valid_ratio_zero_invalid_y"
                        else:
                            row["measured"] = 1; measured += 1; total_sse += m["sse"]; total_satd += m["satd"]; total_valid += int(m["valid_pixels"])

                of.seek((poc-start)*frame_size(a.width,a.height)); write_frame(of,out,a.output_stored_bit_shift); rows[poc]=row
                prog=(oi+1)/len(order); bw=30; fill=int(round(prog*bw)); status=(f"PSNR={float(row['psnr_y']):.4f} SATD={float(row['satd_y']):.1f}" if row["measured"] else row["skip_reason"])
                print(f"\r[{'#'*fill}{'-'*(bw-fill)}] {oi+1}/{len(order)} POC={poc} {status}",end="",flush=True)
        print()

        with open(temp_path,"rb") as ti, open(a.out_warped_yuv,"wb") as fo:
            fs=frame_size(a.width,a.height)
            for poc in range(start,end):
                ti.seek((poc-start)*fs); raw=ti.read(fs)
                if len(raw)!=fs: raise EOFError(f"Missing temporary POC {poc}")
                fo.write(raw)
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

    ordered=[rows[p] for p in range(start,end)]
    with open(a.out_frame_csv,"w",newline="",encoding="utf-8") as fp:
        wr=csv.DictWriter(fp,fieldnames=fields); wr.writeheader(); wr.writerows(ordered)
    mse=total_sse/total_valid if total_valid else 0.0
    psnr=float("nan") if total_valid==0 else float("inf") if mse==0 else 10*math.log10(a.peak_value*a.peak_value/mse)
    mean_psnr=float(np.mean([float(r["psnr_y"]) for r in ordered if r["measured"]])) if measured else float("nan")
    summary={"version":"2026-07-13-backward-warp-projection-hierarchical-ra-v4","warping":"backward mapping using target depth","ra_reference_rule":"recursive interval endpoints; e.g. 16<-(0,32), 8<-(0,16), 4<-(0,8), 24<-(16,32)","generated_ra_reference_map":{str(p): [r0, r1] for p, (r0, r1) in hierarchical_refs.items()},"mv_csv_reference_override":bool(a.mv_csv),"bidirectional_blend":"1:1 where both valid, otherwise single valid direction","metric_scope":"union of L0/L1 valid warped Y pixels only","output_rule":"for attempted warps: valid Y=warped and invalid Y=0; chroma=GT; excluded/missing-camera/no-reference frames=GT","video_yuv":a.video_yuv,"input_depth_yuv":a.input_depth_yuv,"camera_param":a.camera_param,"mv_csv":a.mv_csv,"width":a.width,"height":a.height,"start_frame":start,"end_frame_exclusive":end,"coding_order":a.coding_order,"coding_poc_order":order,"output_poc_order":list(range(start,end)),"ra_gop_size":a.ra_gop_size,"excluded_pocs":sorted(excluded),"depth_source":"target POC depth","depth_scale_real":scale,"measured_frame_count":measured,"aggregate_valid_pixels":total_valid,"overall_sse_y":total_sse,"overall_mse_y":mse,"overall_projection_psnr_y":psnr,"mean_frame_projection_psnr_y":mean_psnr,"overall_satd_y":total_satd,"overall_satd_per_valid_pixel":total_satd/total_valid if total_valid else 0.0,"frames":ordered,"out_warped_yuv":a.out_warped_yuv,"out_frame_csv":a.out_frame_csv}
    with open(a.out_summary_json,"w",encoding="utf-8") as fp: json.dump(summary,fp,indent=2,ensure_ascii=False)
    print(f"Measured frames              : {measured}")
    print(f"Aggregate valid pixels       : {total_valid}")
    print(f"Overall backward PSNR-Y      : {psnr:.6f} dB")
    print(f"Mean frame backward PSNR-Y   : {mean_psnr:.6f} dB")
    print(f"Overall backward SATD-Y      : {total_satd:.6f}")
    print(f"Warped YUV                   : {a.out_warped_yuv}")
    print(f"Frame CSV                    : {a.out_frame_csv}")
    print(f"Summary JSON                 : {a.out_summary_json}")


if __name__ == "__main__":
    main()
