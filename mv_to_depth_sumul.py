#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-block depth predictor candidate RDO simulator.

Inputs
------
- GT depth YUV420p10le: Y is depth code, z = Y * depth_scale_real.
- camParam JSONL.
- Optional MV CSV: poc,x,y,w,h,list,ref_poc,mv_x,mv_y.

For every fixed block, the encoder fits the GT inverse-depth plane
  1/z = a*(x-cx) + b*(y-cy) + c
and evaluates these predictors without block split:
  direct_zero, left, top, top_left, spatial_all,
  fw_ref_<POC>, fw_average.

Forward-warp candidates are built once per reference/current pair in a
low-resolution buffer. Reference depth is always GT depth, by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
class Plane:
    a: float
    b: float
    c: float
    cx: float
    cy: float


@dataclass
class MVRecord:
    poc: int
    x: int
    y: int
    w: int
    h: int
    list_id: str
    ref_poc: int
    mv_x: float
    mv_y: float


@dataclass
class CandidateResult:
    name: str
    pred: Plane
    recon: Plane
    q: Tuple[int, int, int]
    candidate_bits: int
    residual_bits: int
    bits: int
    sse: float
    cost: float
    recon_y: np.ndarray


def ceil_log2(v: int) -> int:
    return 0 if v <= 1 else int(math.ceil(math.log2(v)))


def signed_to_code_num(v: int) -> int:
    return 0 if v == 0 else (2 * v - 1 if v > 0 else -2 * v)


def se_bits(v: int) -> int:
    u = signed_to_code_num(int(v))
    return 2 * int(math.floor(math.log2(u + 1))) + 1


def frame_size(w: int, h: int) -> int:
    return w * h * 3


def count_frames(path: str, w: int, h: int) -> int:
    fs = frame_size(w, h)
    size = os.path.getsize(path)
    if size % fs:
        print(f"[WARN] trailing bytes ignored: {size % fs}")
    return size // fs


def read_y(fp, idx: int, w: int, h: int) -> np.ndarray:
    fp.seek(idx * frame_size(w, h))
    raw = fp.read(w * h * 2)
    if len(raw) != w * h * 2:
        raise EOFError(f"Cannot read frame {idx}")
    return np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float64)


def write_depth_frame(fp, y: np.ndarray, w: int, h: int) -> None:
    y16 = np.clip(np.rint(y), 0, 1023).astype("<u2")
    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    fp.write(np.ascontiguousarray(y16).tobytes())
    fp.write(uv.tobytes())
    fp.write(uv.tobytes())


def rt4(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def load_cameras(path: str) -> Tuple[Dict[str, Any], Dict[int, Camera]]:
    header = None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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

    cur = base.copy()
    prev_w2c = np.eye(4, dtype=np.float64)
    cams: Dict[int, Camera] = {}
    for order, rec in enumerate(records):
        poc = int(rec["poc"])
        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), np.float64)
        cur = base.copy() if fixed else cur + delta
        K = np.array([[cur[0], 0, cur[2]], [0, cur[1], cur[3]], [0, 0, 1]], np.float64)
        Trec = rt4(rec["rvec"], rec["tvec"])
        if pose_mode == "current_to_previous":
            W2C = np.eye(4, dtype=np.float64) if order == 0 else np.linalg.inv(Trec) @ prev_w2c
        elif pose_mode in ("gop_local", "absolute"):
            W2C = Trec
        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")
        C2W = np.linalg.inv(W2C)
        cams[poc] = Camera(poc, K, W2C, C2W, z_sign)
        prev_w2c = W2C
    return header, cams


def depth_scale_real(header: Dict[str, Any]) -> float:
    if "depth_scale_precision" in header:
        p = float(header["depth_scale_precision"])
        if p <= 0:
            raise ValueError("depth_scale_precision must be positive")
        return float(header["depth_scale"]) / p
    return float(header.get("depth_scale_real", header["depth_scale"]))


def scaled_camera(cam: Camera, sx: float, sy: float) -> Camera:
    K = cam.K.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return Camera(cam.poc, K, cam.W2C.copy(), cam.C2W.copy(), cam.z_sign)


def load_mv_csv(path: str, nframes: int, bs: int):
    rows = [[] for _ in range(nframes)]
    refs = [{} for _ in range(nframes)]
    if not path:
        return rows, refs
    req = {"poc", "x", "y", "w", "h", "list", "ref_poc", "mv_x", "mv_y"}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or req - set(rd.fieldnames):
            raise RuntimeError(f"Bad MV CSV header; missing={sorted(req - set(rd.fieldnames or []))}")
        for line_no, r in enumerate(rd, 2):
            try:
                o = MVRecord(int(r["poc"]), int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), str(r["list"]), int(r["ref_poc"]), float(r["mv_x"]), float(r["mv_y"]))
            except Exception as e:
                raise RuntimeError(f"Bad MV CSV row {line_no}: {r}") from e
            if 0 <= o.poc < nframes:
                rows[o.poc].append(o)
                key = (o.x // bs, o.y // bs)
                a = refs[o.poc].setdefault(key, [])
                if o.ref_poc not in a:
                    a.append(o.ref_poc)
    return rows, refs


class GridCache:
    def __init__(self):
        self.cache = {}

    def get(self, w: int, h: int):
        key = (w, h)
        if key not in self.cache:
            xx, yy = np.meshgrid(np.arange(w, dtype=np.float64) - (w - 1) / 2.0,
                                 np.arange(h, dtype=np.float64) - (h - 1) / 2.0)
            self.cache[key] = (xx, yy)
        return self.cache[key]


def fit_plane(z: np.ndarray, cx: float, cy: float, grid: GridCache, min_depth: float, mask=None) -> Optional[Plane]:
    z = np.asarray(z, np.float64)
    h, w = z.shape
    xx, yy = grid.get(w, h)
    valid = np.isfinite(z) & (z > min_depth)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return None
    inv = 1.0 / z[valid]
    if n >= 3:
        A = np.stack([xx[valid], yy[valid], np.ones(n)], axis=1)
        try:
            c, _, rank, _ = np.linalg.lstsq(A, inv, rcond=None)
            if rank >= 3 and np.isfinite(c).all():
                return Plane(float(c[0]), float(c[1]), float(c[2]), cx, cy)
        except np.linalg.LinAlgError:
            pass
    return Plane(0.0, 0.0, float(np.mean(inv)), cx, cy)


def recenter(p: Plane, cx: float, cy: float) -> Plane:
    return Plane(p.a, p.b, p.c + p.a * (cx - p.cx) + p.b * (cy - p.cy), cx, cy)


def average_planes(ps: Sequence[Plane], cx: float, cy: float) -> Optional[Plane]:
    if not ps:
        return None
    q = [recenter(p, cx, cy) for p in ps]
    return Plane(float(np.mean([p.a for p in q])), float(np.mean([p.b for p in q])), float(np.mean([p.c for p in q])), cx, cy)


def render_plane(p: Plane, w: int, h: int, grid: GridCache, min_depth: float, max_depth: float):
    xx, yy = grid.get(w, h)
    inv = p.a * xx + p.b * yy + p.c
    valid = np.isfinite(inv) & (inv > 1.0 / max_depth)
    z = np.zeros((h, w), np.float64)
    z[valid] = 1.0 / inv[valid]
    valid &= (z >= min_depth) & (z <= max_depth)
    z[~valid] = 0.0
    return z, valid


def resize_depth_valid(z: np.ndarray, ow: int, oh: int):
    valid = np.isfinite(z) & (z > 0)
    num = cv2.resize(np.where(valid, z, 0).astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float64)
    den = cv2.resize(valid.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float64)
    out = np.zeros((oh, ow), np.float64)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out, ok


def forward_warp_lowres(zref: np.ndarray, cref: Camera, ccur: Camera, ow: int, oh: int):
    sh, sw = zref.shape
    sx, sy = ow / sw, oh / sh
    z, valid = resize_depth_valid(zref, ow, oh)
    cr = scaled_camera(cref, sx, sy)
    cc = scaled_camera(ccur, sx, sy)
    x, y = np.meshgrid(np.arange(ow, dtype=np.float64), np.arange(oh, dtype=np.float64))
    rays = np.stack([(x - cr.K[0, 2]) / cr.K[0, 0], (y - cr.K[1, 2]) / cr.K[1, 1], np.full_like(x, cr.z_sign)], axis=-1)
    Xr = rays * z[..., None]
    M = cc.W2C @ cr.C2W
    Xc = Xr @ M[:3, :3].T + M[:3, 3]
    d = cc.z_sign * Xc[..., 2]
    front = valid & np.isfinite(d) & (d > 1e-10)
    safe = np.where(front, d, 1.0)
    u = cc.K[0, 0] * Xc[..., 0] / safe + cc.K[0, 2]
    v = cc.K[1, 1] * Xc[..., 1] / safe + cc.K[1, 2]
    good = front & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= ow - 1) & (v >= 0) & (v <= oh - 1)
    zbuf = np.full(ow * oh, np.inf, np.float64)
    if np.any(good):
        uu, vv, dd = u[good], v[good], d[good]
        x0, y0 = np.floor(uu).astype(np.int64), np.floor(vv).astype(np.int64)
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < ow) & (yi >= 0) & (yi < oh)
                if np.any(ok):
                    np.minimum.at(zbuf, yi[ok] * ow + xi[ok], dd[ok])
    out = zbuf.reshape(oh, ow)
    mask = np.isfinite(out)
    out[~mask] = 0.0
    return out, mask


def extract_fw_block(z: np.ndarray, m: np.ndarray, bx: int, by: int, bw: int, bh: int, fw: int, fh: int):
    h, w = z.shape
    x0 = max(0, min(int(math.floor(bx * w / fw)), w - 1))
    y0 = max(0, min(int(math.floor(by * h / fh)), h - 1))
    x1 = max(x0 + 1, min(int(math.ceil((bx + bw) * w / fw)), w))
    y1 = max(y0 + 1, min(int(math.ceil((by + bh) * h / fh)), h))
    zs, ms = z[y0:y1, x0:x1], m[y0:y1, x0:x1].astype(np.float32)
    num = cv2.resize((zs * ms).astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    den = cv2.resize(ms, (bw, bh), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    out = np.zeros((bh, bw), np.float64)
    ok = den > 0.25
    out[ok] = num[ok] / np.maximum(den[ok], 1e-8)
    return out, ok


def combine_fw(blocks):
    if not blocks:
        return None
    total = np.zeros_like(blocks[0][0])
    count = np.zeros_like(total)
    for z, m in blocks:
        total[m] += z[m]
        count[m] += 1
    valid = count > 0
    out = np.zeros_like(total)
    out[valid] = total[valid] / count[valid]
    return out, valid


def eval_candidate(name: str, pred: Plane, actual: Plane, gt_y: np.ndarray, scale: float,
                   cand_bits: int, qs, lam: float, grid: GridCache, min_depth: float, max_depth: float):
    residual = np.array([actual.a - pred.a, actual.b - pred.b, actual.c - pred.c])
    qs = np.asarray(qs, np.float64)
    q = np.rint(residual / qs).astype(np.int64)
    d = q * qs
    rec = Plane(pred.a + float(d[0]), pred.b + float(d[1]), pred.c + float(d[2]), actual.cx, actual.cy)
    z, valid = render_plane(rec, gt_y.shape[1], gt_y.shape[0], grid, min_depth, max_depth)
    ry = np.zeros_like(z)
    ry[valid] = z[valid] / scale
    ry = np.clip(np.rint(ry), 0, 1023)
    diff = gt_y - ry
    sse = float(np.sum(diff * diff))
    rb = sum(se_bits(int(v)) for v in q)
    bits = cand_bits + rb
    return CandidateResult(name, pred, rec, (int(q[0]), int(q[1]), int(q[2])), cand_bits, rb, bits, sse, sse + lam * bits, ry)


def qp_steps(args):
    s = 2.0 ** ((args.qp - args.qp_ref) / 6.0)
    return args.qa_base * s, args.qb_base * s, args.qc_base * s


def derive_lambda(args):
    return args.lambda_rd if args.lambda_rd is not None else args.lambda_scale * 2.0 ** ((args.qp - 12.0) / 3.0)


def parse_args():
    p = argparse.ArgumentParser(description="GT depth fixed-block candidate RDO simulator")
    p.add_argument("--gt-depth-yuv", required=True)
    p.add_argument("--camera-param", required=True)
    p.add_argument("--mv-csv", default="")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--forward-downsample", type=int, default=4)
    p.add_argument("--max-forward-refs", type=int, default=2)
    p.add_argument("--default-ref-offset", type=int, default=1)
    p.add_argument("--min-forward-valid-ratio", type=float, default=0.25)
    p.add_argument("--qp", type=int, default=37)
    p.add_argument("--qp-ref", type=int, default=37)
    p.add_argument("--qa-base", type=float, default=1e-6)
    p.add_argument("--qb-base", type=float, default=1e-6)
    p.add_argument("--qc-base", type=float, default=1e-4)
    p.add_argument("--lambda-rd", type=float, default=None)
    p.add_argument("--lambda-scale", type=float, default=0.57)
    p.add_argument("--min-depth", type=float, default=1e-8)
    p.add_argument("--max-depth", type=float, default=1e9)
    p.add_argument("--out-recon-yuv", required=True)
    p.add_argument("--out-block-csv", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError("Invalid YUV420 resolution")
    if a.block_size <= 0 or a.forward_downsample <= 0:
        raise ValueError("Invalid block/downsample size")
    if min(a.qa_base, a.qb_base, a.qc_base) <= 0:
        raise ValueError("qsteps must be positive")

    outs = [Path(a.out_recon_yuv), Path(a.out_block_csv), Path(a.out_summary_json)]
    for p in outs:
        if p.exists():
            if a.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)

    header, cams = load_cameras(a.camera_param)
    scale = depth_scale_real(header)
    total = count_frames(a.gt_depth_yuv, a.width, a.height)
    start = a.start_frame
    end = total if a.num_frames == 0 else min(total, start + a.num_frames)
    if start < 0 or start >= end:
        raise ValueError("Invalid frame range")

    rows, block_refs = load_mv_csv(a.mv_csv, total, a.block_size)
    qa, qb, qc = qp_steps(a)
    lam = derive_lambda(a)
    lw = int(math.ceil(a.width / a.forward_downsample))
    lh = int(math.ceil(a.height / a.forward_downsample))
    grid = GridCache()

    print(f"block={a.block_size}, no split, forward={lw}x{lh}, QP={a.qp}")
    print(f"qsteps={qa:.6g},{qb:.6g},{qc:.6g}, lambda={lam:.6g}, depth_scale={scale:.12g}")

    fields = ["poc","x","y","w","h","candidate_count","candidate_idx","candidate","candidate_bits","residual_bits","total_bits","q_a","q_b","q_c","pred_a","pred_b","pred_c","gt_a","gt_b","gt_c","recon_a","recon_b","recon_c","sse","cost","forward_refs"]
    frames = []
    global_counts: Dict[str, int] = {}
    total_bits = 0
    total_sse = 0.0

    with open(a.gt_depth_yuv, "rb") as fp, open(a.out_recon_yuv, "wb") as outfp, open(a.out_block_csv, "w", newline="", encoding="utf-8") as cfp:
        wr = csv.DictWriter(cfp, fieldnames=fields); wr.writeheader()
        gt_cache: Dict[int, np.ndarray] = {}
        def gt(poc):
            if poc not in gt_cache:
                gt_cache[poc] = read_y(fp, poc, a.width, a.height)
            return gt_cache[poc]

        for ord_idx, poc in enumerate(range(start, end)):
            if poc not in cams:
                raise KeyError(f"No camera for POC {poc}")
            gy = gt(poc); gz = gy * scale
            recon = np.zeros_like(gy)

            frame_refs = []
            for r in rows[poc]:
                if r.ref_poc != poc and r.ref_poc in cams and r.ref_poc not in frame_refs:
                    frame_refs.append(r.ref_poc)
            if not frame_refs and poc - a.default_ref_offset in cams:
                frame_refs = [poc - a.default_ref_offset]
            frame_refs = frame_refs[:a.max_forward_refs]

            fw_cache = {}
            for ref in frame_refs:
                if 0 <= ref < total:
                    fw_cache[ref] = forward_warp_lowres(gt(ref) * scale, cams[ref], cams[poc], lw, lh)

            rec_planes = {}
            fbits = 0; fsse = 0.0; fcounts = {}; nblocks = 0
            gw = int(math.ceil(a.width / a.block_size)); gh = int(math.ceil(a.height / a.block_size))
            for by_idx in range(gh):
                by = by_idx * a.block_size; bh = min(a.block_size, a.height - by)
                for bx_idx in range(gw):
                    bx = bx_idx * a.block_size; bw = min(a.block_size, a.width - bx); nblocks += 1
                    cx = bx + (bw - 1) / 2.0; cy = by + (bh - 1) / 2.0
                    gyb = gy[by:by+bh, bx:bx+bw]; gzb = gz[by:by+bh, bx:bx+bw]
                    actual = fit_plane(gzb, cx, cy, grid, a.min_depth) or Plane(0,0,1/a.max_depth,cx,cy)
                    cands = [("direct_zero", Plane(0,0,0,cx,cy))]
                    spatial = []
                    for name, key in (("left",(bx_idx-1,by_idx)),("top",(bx_idx,by_idx-1)),("top_left",(bx_idx-1,by_idx-1))):
                        if key in rec_planes:
                            p = recenter(rec_planes[key], cx, cy); cands.append((name,p)); spatial.append(p)
                    if len(spatial) >= 2:
                        cands.append(("spatial_all", average_planes(spatial,cx,cy)))

                    refs = []
                    for ref in block_refs[poc].get((bx_idx,by_idx), []):
                        if ref != poc and ref in cams and ref not in refs:
                            refs.append(ref)
                    for ref in frame_refs:
                        if ref not in refs:
                            refs.append(ref)
                    refs = refs[:a.max_forward_refs]
                    fw_blocks = []
                    for ref in refs:
                        if ref not in fw_cache and 0 <= ref < total and ref in cams:
                            fw_cache[ref] = forward_warp_lowres(gt(ref)*scale, cams[ref], cams[poc], lw, lh)
                        if ref not in fw_cache:
                            continue
                        zb, mb = extract_fw_block(*fw_cache[ref], bx, by, bw, bh, a.width, a.height)
                        if np.mean(mb) < a.min_forward_valid_ratio:
                            continue
                        p = fit_plane(zb, cx, cy, grid, a.min_depth, mb)
                        if p is not None:
                            cands.append((f"fw_ref_{ref}", p)); fw_blocks.append((zb,mb))
                    if len(fw_blocks) >= 2:
                        comb = combine_fw(fw_blocks)
                        if comb is not None and np.mean(comb[1]) >= a.min_forward_valid_ratio:
                            p = fit_plane(comb[0], cx, cy, grid, a.min_depth, comb[1])
                            if p is not None:
                                cands.append(("fw_average",p))

                    cbits = ceil_log2(len(cands))
                    results = [eval_candidate(n,p,actual,gyb,scale,cbits,(qa,qb,qc),lam,grid,a.min_depth,a.max_depth) for n,p in cands]
                    idx,best = min(enumerate(results), key=lambda x:x[1].cost)
                    recon[by:by+bh,bx:bx+bw] = best.recon_y
                    rec_planes[(bx_idx,by_idx)] = best.recon
                    fbits += best.bits; fsse += best.sse
                    fcounts[best.name] = fcounts.get(best.name,0)+1
                    global_counts[best.name] = global_counts.get(best.name,0)+1
                    wr.writerow({"poc":poc,"x":bx,"y":by,"w":bw,"h":bh,"candidate_count":len(cands),"candidate_idx":idx,"candidate":best.name,"candidate_bits":best.candidate_bits,"residual_bits":best.residual_bits,"total_bits":best.bits,"q_a":best.q[0],"q_b":best.q[1],"q_c":best.q[2],"pred_a":best.pred.a,"pred_b":best.pred.b,"pred_c":best.pred.c,"gt_a":actual.a,"gt_b":actual.b,"gt_c":actual.c,"recon_a":best.recon.a,"recon_b":best.recon.b,"recon_c":best.recon.c,"sse":best.sse,"cost":best.cost,"forward_refs":"|".join(map(str,refs))})

            write_depth_frame(outfp,recon,a.width,a.height)
            mse = fsse/(a.width*a.height); psnr = float("inf") if mse==0 else 10*math.log10(1023**2/mse)
            frames.append({"poc":poc,"num_blocks":nblocks,"frame_bits":fbits,"frame_bpp":fbits/(a.width*a.height),"frame_sse":fsse,"frame_mse":mse,"frame_psnr":psnr,"frame_reference_pocs":frame_refs,"forward_buffers_built":sorted(fw_cache.keys()),"candidate_selection_counts":fcounts})
            total_bits += fbits; total_sse += fsse
            ratio=(ord_idx+1)/(end-start); width=30; fill=int(round(width*ratio)); print(f"\r[{'#'*fill}{'-'*(width-fill)}] {ord_idx+1}/{end-start} POC={poc} bits={fbits} PSNR={psnr:.3f}",end="",flush=True)
    print()

    pixels = a.width*a.height*(end-start); mse=total_sse/pixels; psnr=float("inf") if mse==0 else 10*math.log10(1023**2/mse)
    summary={"gt_depth_yuv":a.gt_depth_yuv,"camera_param":a.camera_param,"mv_csv":a.mv_csv,"width":a.width,"height":a.height,"start_frame":start,"num_frames":end-start,"block_size":a.block_size,"block_split":False,"forward_downsample":a.forward_downsample,"forward_buffer_size":[lw,lh],"reference_depth_source":"GT depth frame","depth_scale_real":scale,"plane_equation":"1/z = a*(x-cx)+b*(y-cy)+c","qp":a.qp,"qp_ref":a.qp_ref,"quantization":{"qa":qa,"qb":qb,"qc":qc,"formula":"base*2^((QP-QP_ref)/6)"},"lambda_rd":lam,"rate_model":{"candidate_idx_bits":"ceil(log2(N_available))","residual_bits":"signed Exp-Golomb for q_a,q_b,q_c"},"candidate_types":["direct_zero","left","top","top_left","spatial_all","fw_ref_<POC>","fw_average"],"total_bits":total_bits,"overall_bpp":total_bits/pixels,"overall_sse":total_sse,"overall_mse":mse,"overall_psnr":psnr,"candidate_selection_counts":global_counts,"frames":frames,"out_recon_yuv":a.out_recon_yuv,"out_block_csv":a.out_block_csv}
    with open(a.out_summary_json,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f"Recon YUV : {a.out_recon_yuv}\nBlock CSV : {a.out_block_csv}\nSummary   : {a.out_summary_json}\nTotal bits: {total_bits}\nBPP       : {summary['overall_bpp']:.9f}\nPSNR      : {psnr:.6f} dB")


if __name__ == "__main__":
    main()

