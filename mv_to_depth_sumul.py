#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VERSION: 2026-07-13-v6-projection-y-satd-rdo

Fixed-block depth predictor candidate RDO simulator.

Inputs
------
- GT depth YUV420p10le: Y is depth code, z = Y * depth_scale_real.
- camParam JSONL.
- Sequence YUV420p10le used to evaluate projection-domain Y SATD.
- Optional MV CSV: poc,x,y,w,h,list,ref_poc,mv_x,mv_y.

For every fixed block, the encoder fits the GT inverse-depth plane
  1/z = a*(x-cx) + b*(y-cy) + c
and evaluates these predictors without block split:
  direct_zero, left, top, top_left, spatial_all,
  fw_ref_<POC>, fw_average.

Spatial candidates are derived from the neighboring blocks' MV observations and
camera parameters. Forward-warp candidates are built once per reconstructed
reference/current pair in a low-resolution buffer. Reference depth is reconstructed
depth, never GT depth. RDO distortion is projection-domain Y SATD, not depth SSE.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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



def ue_bits(v: int) -> int:
    """Unsigned Exp-Golomb bit count for a non-negative integer."""
    v = int(v)
    if v < 0:
        raise ValueError("ue_bits expects a non-negative integer")
    return 2 * int(math.floor(math.log2(v + 1))) + 1


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
    block_rows = [{} for _ in range(nframes)]
    if not path:
        return rows, refs, block_rows
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
                block_rows[o.poc].setdefault(key, []).append(o)
    return rows, refs, block_rows


def pixel_ray(u: float, v: float, cam: Camera) -> np.ndarray:
    return np.array([
        (u - cam.K[0, 2]) / cam.K[0, 0],
        (v - cam.K[1, 2]) / cam.K[1, 1],
        cam.z_sign,
    ], dtype=np.float64)


def project_point(X: np.ndarray, cam: Camera) -> Optional[np.ndarray]:
    depth = cam.z_sign * float(X[2])
    if not np.isfinite(depth) or depth <= 1e-10:
        return None
    return np.array([
        cam.K[0, 0] * float(X[0]) / depth + cam.K[0, 2],
        cam.K[1, 1] * float(X[1]) / depth + cam.K[1, 2],
    ], dtype=np.float64)


def solve_depth_from_mv(row: MVRecord, cams: Dict[int, Camera],
                        min_parallax: float, max_reproj_error: float,
                        min_depth: float, max_depth: float) -> Optional[Tuple[float,float,float,float]]:
    if row.poc not in cams or row.ref_poc not in cams:
        return None
    cur = cams[row.poc]
    ref = cams[row.ref_poc]
    u = row.x + (row.w - 1) * 0.5
    v = row.y + (row.h - 1) * 0.5
    ur = u + row.mv_x
    vr = v + row.mv_y
    ray = pixel_ray(u, v, cur)
    M = ref.W2C @ cur.C2W
    R, t = M[:3,:3], M[:3,3]
    q = R @ ray
    fx, fy = float(ref.K[0,0]), float(ref.K[1,1])
    cx, cy = float(ref.K[0,2]), float(ref.K[1,2])
    zs = float(ref.z_sign)
    du, dv = ur - cx, vr - cy
    A = np.array([
        du * zs * q[2] - fx * q[0],
        dv * zs * q[2] - fy * q[1],
    ], dtype=np.float64)
    B = np.array([
        fx * t[0] - du * zs * t[2],
        fy * t[1] - dv * zs * t[2],
    ], dtype=np.float64)
    denom = float(A @ A)
    if not np.isfinite(denom) or denom < min_parallax * min_parallax:
        return None
    z = float((A @ B) / denom)
    if not np.isfinite(z) or z < min_depth or z > max_depth:
        return None
    pred = project_point(z * q + t, ref)
    if pred is None:
        return None
    err = float(np.linalg.norm(pred - np.array([ur, vr], dtype=np.float64)))
    if not np.isfinite(err) or err > max_reproj_error:
        return None
    return u, v, z, err


def fit_plane_from_mv_rows(rows: Sequence[MVRecord], cams: Dict[int, Camera],
                           cx: float, cy: float, min_parallax: float,
                           max_reproj_error: float, min_depth: float,
                           max_depth: float, min_points: int) -> Optional[Plane]:
    pts = []
    for row in rows:
        solved = solve_depth_from_mv(row, cams, min_parallax, max_reproj_error,
                                     min_depth, max_depth)
        if solved is not None:
            pts.append(solved)
    if len(pts) < min_points:
        return None
    x = np.asarray([v[0] for v in pts], np.float64)
    y = np.asarray([v[1] for v in pts], np.float64)
    z = np.asarray([v[2] for v in pts], np.float64)
    err = np.asarray([v[3] for v in pts], np.float64)
    A = np.stack([x-cx, y-cy, np.ones_like(x)], axis=1)
    inv = 1.0 / z
    w = 1.0 / (1.0 + err*err)
    sw = np.sqrt(np.maximum(w, 1e-12))
    try:
        c, _, rank, _ = np.linalg.lstsq(A*sw[:,None], inv*sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.isfinite(c).all():
        return Plane(0.0, 0.0, float(np.average(inv, weights=w)), cx, cy)
    return Plane(float(c[0]), float(c[1]), float(c[2]), cx, cy)


def build_ra_order(start: int, end: int, gop: int) -> List[int]:
    if end <= start:
        return []
    order = [start]
    seen = {start}
    def mids(lo: int, hi: int):
        if hi-lo <= 1:
            return
        mid = (lo+hi)//2
        if mid in seen or mid <= lo or mid >= hi:
            return
        order.append(mid); seen.add(mid)
        mids(lo, mid); mids(mid, hi)
    lo = start
    last = end-1
    while lo < last:
        hi = min(lo+gop, last)
        if hi not in seen:
            order.append(hi); seen.add(hi)
        mids(lo, hi)
        lo = hi
    if sorted(order) != list(range(start,end)):
        raise RuntimeError('RA order generation failed')
    return order


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



def hadamard4x4(block: np.ndarray) -> np.ndarray:
    """Integer 4x4 Hadamard transform used for SATD."""
    x = np.asarray(block, dtype=np.float64)
    if x.shape != (4, 4):
        raise ValueError("hadamard4x4 expects a 4x4 block")
    t = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        a0 = x[i, 0] + x[i, 3]
        a1 = x[i, 1] + x[i, 2]
        a2 = x[i, 1] - x[i, 2]
        a3 = x[i, 0] - x[i, 3]
        t[i, 0] = a0 + a1
        t[i, 1] = a3 + a2
        t[i, 2] = a0 - a1
        t[i, 3] = a3 - a2
    y = np.empty((4, 4), dtype=np.float64)
    for j in range(4):
        a0 = t[0, j] + t[3, j]
        a1 = t[1, j] + t[2, j]
        a2 = t[1, j] - t[2, j]
        a3 = t[0, j] - t[3, j]
        y[0, j] = a0 + a1
        y[1, j] = a3 + a2
        y[2, j] = a0 - a1
        y[3, j] = a3 - a2
    return y


def satd_4x4_tiled(residual: np.ndarray) -> float:
    """Sum 4x4 SATD over a block. Partial edge tiles are zero padded."""
    r = np.asarray(residual, dtype=np.float64)
    h, w = r.shape
    total = 0.0
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            tile = np.zeros((4, 4), dtype=np.float64)
            ph = min(4, h - y)
            pw = min(4, w - x)
            tile[:ph, :pw] = r[y:y + ph, x:x + pw]
            total += 0.5 * float(np.sum(np.abs(hadamard4x4(tile))))
    return total


def backward_project_reference_y_block(
    depth_y_block: np.ndarray,
    target_y_block: np.ndarray,
    reference_y: np.ndarray,
    current_cam: Camera,
    reference_cam: Camera,
    bx: int,
    by: int,
    depth_scale: float,
    min_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-project reference Y to the current block using current candidate depth.

    Invalid projected samples are filled with the collocated reference block. This
    fallback is decoder-available and prevents candidates with many invalid samples
    from receiving an artificially low distortion.
    """
    bh, bw = depth_y_block.shape
    yy, xx = np.meshgrid(
        np.arange(by, by + bh, dtype=np.float64),
        np.arange(bx, bx + bw, dtype=np.float64),
        indexing="ij",
    )
    depth = np.asarray(depth_y_block, np.float64) * float(depth_scale)
    valid_depth = np.isfinite(depth) & (depth > min_depth)
    rays = np.stack([
        (xx - current_cam.K[0, 2]) / current_cam.K[0, 0],
        (yy - current_cam.K[1, 2]) / current_cam.K[1, 1],
        np.full_like(xx, current_cam.z_sign),
    ], axis=-1)
    Xc = rays * depth[..., None]
    M = reference_cam.W2C @ current_cam.C2W
    Xr = Xc @ M[:3, :3].T + M[:3, 3]
    ref_depth = reference_cam.z_sign * Xr[..., 2]
    front = valid_depth & np.isfinite(ref_depth) & (ref_depth > 1e-10)
    safe = np.where(front, ref_depth, 1.0)
    map_x = reference_cam.K[0, 0] * Xr[..., 0] / safe + reference_cam.K[0, 2]
    map_y = reference_cam.K[1, 1] * Xr[..., 1] / safe + reference_cam.K[1, 2]
    rh, rw = reference_y.shape
    valid = (
        front & np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0.0) & (map_x <= rw - 1)
        & (map_y >= 0.0) & (map_y <= rh - 1)
    )
    projected = cv2.remap(
        np.asarray(reference_y, np.float32),
        map_x.astype(np.float32), map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)
    collocated = np.asarray(reference_y[by:by + bh, bx:bx + bw], np.float64)
    if collocated.shape != (bh, bw):
        collocated = cv2.resize(
            np.asarray(reference_y, np.float32), (bw, bh),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float64)
    projected[~valid] = collocated[~valid]
    return projected, valid


def make_projection_satd_evaluator(
    poc: int,
    bx: int,
    by: int,
    target_y_block: np.ndarray,
    reference_pocs: Sequence[int],
    sequence_y_getter: Callable[[int], np.ndarray],
    cams: Dict[int, Camera],
    depth_scale: float,
    min_depth: float,
) -> Callable[[np.ndarray], Tuple[float, int, float]]:
    """Return a candidate-depth -> (best SATD, ref POC, valid ratio) evaluator."""
    usable_refs = [r for r in reference_pocs if r in cams and r != poc]

    def evaluate(depth_y_block: np.ndarray) -> Tuple[float, int, float]:
        best: Optional[Tuple[float, int, float]] = None
        for ref_poc in usable_refs:
            try:
                ref_y = sequence_y_getter(ref_poc)
            except (EOFError, IndexError):
                continue
            projected, valid = backward_project_reference_y_block(
                depth_y_block, target_y_block, ref_y,
                cams[poc], cams[ref_poc], bx, by,
                depth_scale, min_depth,
            )
            satd = satd_4x4_tiled(np.asarray(target_y_block, np.float64) - projected)
            item = (float(satd), int(ref_poc), float(np.mean(valid)))
            if best is None or item[0] < best[0]:
                best = item
        if best is not None:
            return best
        # Normally every inter frame has a decoded reference. Keep deterministic
        # behavior for malformed/no-reference input by using depth-domain SATD.
        fallback = satd_4x4_tiled(
            np.asarray(target_y_block, np.float64) - np.asarray(depth_y_block, np.float64)
        )
        return float(fallback), -1, 0.0

    return evaluate


class SimpleAdaptiveProb:
    """Very small categorical probability model.

    After a committed symbol:
      selected     += update_step
      every unused -= update_step
    Then probabilities are clipped and normalized to sum to one.
    Bit cost is -log2(probability), renormalized over available symbols.
    """

    def __init__(self, symbols, update_step=0.025, p_min=0.01, name=""):
        self.symbols = list(symbols)
        if not self.symbols:
            raise ValueError("SimpleAdaptiveProb needs at least one symbol")
        if update_step < 0.0:
            raise ValueError("update_step must be non-negative")
        if p_min <= 0.0 or p_min * len(self.symbols) >= 1.0:
            raise ValueError("invalid p_min")
        self.update_step = float(update_step)
        self.p_min = float(p_min)
        self.name = str(name)
        self.probs = {x: 1.0 / len(self.symbols) for x in self.symbols}

    def bits(self, symbol, available=None):
        if symbol not in self.probs:
            raise KeyError(f"unknown symbol {symbol} in {self.name}")
        av = self.symbols if available is None else [x for x in available if x in self.probs]
        if symbol not in av:
            raise KeyError(f"unavailable symbol {symbol} in {self.name}")
        if len(av) <= 1:
            return 0.0
        norm = sum(self.probs[x] for x in av)
        prob = self.probs[symbol] / max(norm, 1e-15)
        return -math.log2(max(prob, 1e-15))

    def update(self, selected):
        if selected not in self.probs:
            raise KeyError(f"unknown selected symbol {selected} in {self.name}")
        for symbol in self.symbols:
            if symbol == selected:
                self.probs[symbol] += self.update_step
            else:
                self.probs[symbol] -= self.update_step
        self._normalize()

    def _normalize(self):
        values = np.array([max(self.p_min, self.probs[x]) for x in self.symbols], dtype=np.float64)
        # Iterative floor projection followed by normalization.
        for _ in range(16):
            values /= max(float(np.sum(values)), 1e-15)
            low = values < self.p_min
            if not np.any(low):
                break
            values[low] = self.p_min
        values /= max(float(np.sum(values)), 1e-15)
        for symbol, value in zip(self.symbols, values):
            self.probs[symbol] = float(value)

    def snapshot(self):
        return dict(self.probs)


class AdaptiveResidualCoder:
    """Adaptive residual syntax for q_a/q_b/q_c.

    q_a and q_b share one probability distribution. q_c uses an independent
    distribution. Each coefficient is coded as:
      1) zero/nonzero adaptive symbol
      2) if nonzero, one fixed sign bit
      3) adaptive magnitude class: 1, 2, 3, 4, or gt4
      4) for gt4, ue(abs(q)-5)

    The coding order is q_a, q_b, q_c. Therefore q_b uses the a/b model after
    q_a has already updated it, exactly as a causal decoder would.
    """

    MAG_SYMBOLS = ["1", "2", "3", "4", "gt4"]

    def __init__(self, update_step: float, p_min: float):
        self.ab_zero = SimpleAdaptiveProb(["zero", "nonzero"], update_step, p_min, "residual_ab_zero")
        self.ab_mag = SimpleAdaptiveProb(self.MAG_SYMBOLS, update_step, p_min, "residual_ab_mag")
        self.c_zero = SimpleAdaptiveProb(["zero", "nonzero"], update_step, p_min, "residual_c_zero")
        self.c_mag = SimpleAdaptiveProb(self.MAG_SYMBOLS, update_step, p_min, "residual_c_mag")

    @staticmethod
    def magnitude_symbol(q: int) -> str:
        m = abs(int(q))
        if m <= 0:
            raise ValueError("magnitude_symbol requires nonzero q")
        return str(m) if m <= 4 else "gt4"

    @staticmethod
    def _one_bits(q: int, zero_model: SimpleAdaptiveProb, mag_model: SimpleAdaptiveProb, update: bool) -> float:
        q = int(q)
        nz = q != 0
        zero_symbol = "nonzero" if nz else "zero"
        bits = zero_model.bits(zero_symbol)
        if update:
            zero_model.update(zero_symbol)
        if not nz:
            return bits

        bits += 1.0  # sign is fixed 50:50
        mag = abs(q)
        mag_symbol = AdaptiveResidualCoder.magnitude_symbol(q)
        bits += mag_model.bits(mag_symbol)
        if mag > 4:
            bits += float(ue_bits(mag - 5))
        if update:
            mag_model.update(mag_symbol)
        return bits

    def bits_each(self, q: Sequence[int]) -> Tuple[float, float, float]:
        if len(q) != 3:
            raise ValueError("Residual vector must have three coefficients")
        temp = copy.deepcopy(self)
        ba = temp._one_bits(int(q[0]), temp.ab_zero, temp.ab_mag, True)
        bb = temp._one_bits(int(q[1]), temp.ab_zero, temp.ab_mag, True)
        bc = temp._one_bits(int(q[2]), temp.c_zero, temp.c_mag, True)
        return float(ba), float(bb), float(bc)

    def bits(self, q: Sequence[int]) -> float:
        return float(sum(self.bits_each(q)))

    def update(self, q: Sequence[int]) -> None:
        if len(q) != 3:
            raise ValueError("Residual vector must have three coefficients")
        self._one_bits(int(q[0]), self.ab_zero, self.ab_mag, True)
        self._one_bits(int(q[1]), self.ab_zero, self.ab_mag, True)
        self._one_bits(int(q[2]), self.c_zero, self.c_mag, True)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "ab_zero_nonzero": self.ab_zero.snapshot(),
            "ab_magnitude": self.ab_mag.snapshot(),
            "c_zero_nonzero": self.c_zero.snapshot(),
            "c_magnitude": self.c_mag.snapshot(),
        }


def candidate_probability_class(name: str) -> str:
    return "fw_ref" if name.startswith("fw_ref_") else name


def adaptive_candidate_bits(model: SimpleAdaptiveProb, selected_name: str, available_names) -> float:
    selected_class = candidate_probability_class(selected_name)
    available_classes = [candidate_probability_class(x) for x in available_names]
    unique_classes = list(dict.fromkeys(available_classes))
    class_bits = model.bits(selected_class, unique_classes)
    multiplicity = sum(x == selected_class for x in available_classes)
    return class_bits + (math.log2(multiplicity) if multiplicity > 1 else 0.0)


@dataclass
class RDResult:
    mode: str
    name: str
    pred: Plane
    recon: Plane
    q: Tuple[int, int, int]
    candidate_idx: int
    mode_bits: float
    candidate_bits: float
    residual_bits: float
    residual_bits_each: Tuple[float, float, float]
    bits: float
    depth_sse: float
    projection_satd: float
    projection_ref_poc: int
    projection_valid_ratio: float
    cost: float
    recon_y: np.ndarray
    residual_present: bool
    buffer_no_signal: bool


def truncated_unary_bits(idx: int, n: int) -> int:
    if n <= 1:
        return 0
    return idx if idx == n - 1 else idx + 1


def candidate_idx_bits(idx: int, n: int, coding: str) -> int:
    if n <= 1:
        return 0
    if coding == "fixed":
        return ceil_log2(n)
    if coding == "truncated_unary":
        return truncated_unary_bits(idx, n)
    raise ValueError(coding)


def syntax_mode_bits(mode: str) -> int:
    if mode == "direct":
        return 1
    if mode in ("predictor_only", "predictor_residual"):
        return 2
    if mode == "buffer_reuse":
        return 0
    raise ValueError(mode)


class DepthReuseBuffer:
    def __init__(self, fw: int, fh: int, scale: int, min_depth: float):
        self.fw = fw
        self.fh = fh
        self.scale = scale
        self.min_depth = min_depth
        self.w = int(math.ceil(fw / scale))
        self.h = int(math.ceil(fh / scale))
        self.depth = np.zeros((self.h, self.w), np.float64)
        self.valid = np.zeros((self.h, self.w), bool)

    def rect(self, x: int, y: int, w: int, h: int):
        x0 = max(0, x // self.scale)
        y0 = max(0, y // self.scale)
        x1 = min(self.w, int(math.ceil((x + w) / self.scale)))
        y1 = min(self.h, int(math.ceil((y + h) / self.scale)))
        return x0, y0, x1, y1

    def can_reuse(self, x: int, y: int, w: int, h: int) -> bool:
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        return x1 > x0 and y1 > y0 and bool(np.all(self.valid[y0:y1, x0:x1]))

    def reconstruct(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        if not np.all(self.valid[y0:y1, x0:x1]):
            raise RuntimeError("invalid depth-buffer reuse")
        return cv2.resize(self.depth[y0:y1, x0:x1].astype(np.float32),
                          (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    def commit(self, x: int, y: int, z: np.ndarray):
        h, w = z.shape
        x0, y0, x1, y1 = self.rect(x, y, w, h)
        tw, th = x1 - x0, y1 - y0
        if tw <= 0 or th <= 0:
            return
        valid = np.isfinite(z) & (z > self.min_depth)
        num = cv2.resize(np.where(valid, z, 0).astype(np.float32),
                         (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
        den = cv2.resize(valid.astype(np.float32),
                         (tw, th), interpolation=cv2.INTER_AREA).astype(np.float64)
        ok = den > 1e-6
        val = np.zeros((th, tw), np.float64)
        val[ok] = num[ok] / den[ok]
        self.depth[y0:y1, x0:x1][ok] = val[ok]
        self.valid[y0:y1, x0:x1][ok] = True


def plane_to_recon_y(p: Plane, gt_y: np.ndarray, scale: float, grid: GridCache,
                     min_depth: float, max_depth: float):
    z, valid = render_plane(p, gt_y.shape[1], gt_y.shape[0], grid, min_depth, max_depth)
    ry = np.zeros_like(z)
    ry[valid] = z[valid] / scale
    return np.clip(np.rint(ry), 0, 1023), z


def make_direct(actual: Plane, gt_y: np.ndarray, scale: float, qs, lam: float,
                grid: GridCache, min_depth: float, max_depth: float,
                adaptive_mode_bits: float, residual_coder: AdaptiveResidualCoder,
                distortion_evaluator: Callable[[np.ndarray], Tuple[float, int, float]]) -> RDResult:
    qs = np.asarray(qs, np.float64)
    q = np.rint(np.array([actual.a, actual.b, actual.c]) / qs).astype(np.int64)
    d = q * qs
    rec = Plane(float(d[0]), float(d[1]), float(d[2]), actual.cx, actual.cy)
    ry, _ = plane_to_recon_y(rec, gt_y, scale, grid, min_depth, max_depth)
    depth_sse = float(np.sum((gt_y - ry) ** 2))
    projection_satd, projection_ref_poc, projection_valid_ratio = distortion_evaluator(ry)
    mb = float(adaptive_mode_bits)
    rb_each = residual_coder.bits_each(q)
    rb = float(sum(rb_each))
    bits = mb + rb
    return RDResult("direct", "direct", Plane(0,0,0,actual.cx,actual.cy), rec,
                    tuple(map(int,q)), -1, mb, 0.0, rb, rb_each, bits, depth_sse,
                    projection_satd, projection_ref_poc, projection_valid_ratio,
                    projection_satd + lam * bits, ry, True, False)


def make_predictor_only(name: str, pred: Plane, idx: int, n: int, gt_y: np.ndarray,
                        scale: float, coding: str, lam: float, grid: GridCache,
                        min_depth: float, max_depth: float, adaptive_mode_bits: float,
                        adaptive_candidate_bits_value: float,
                        distortion_evaluator: Callable[[np.ndarray], Tuple[float, int, float]]) -> RDResult:
    ry, _ = plane_to_recon_y(pred, gt_y, scale, grid, min_depth, max_depth)
    depth_sse = float(np.sum((gt_y - ry) ** 2))
    projection_satd, projection_ref_poc, projection_valid_ratio = distortion_evaluator(ry)
    mb = float(adaptive_mode_bits)
    cb = float(adaptive_candidate_bits_value)
    bits = mb + cb
    return RDResult("predictor_only", name, pred, pred, (0,0,0), idx,
                    mb, cb, 0.0, (0.0, 0.0, 0.0), bits, depth_sse,
                    projection_satd, projection_ref_poc, projection_valid_ratio,
                    projection_satd + lam * bits, ry, False, False)


def make_predictor_residual(name: str, pred: Plane, actual: Plane, idx: int, n: int,
                            gt_y: np.ndarray, scale: float, coding: str, qs, lam: float,
                            grid: GridCache, min_depth: float, max_depth: float,
                            adaptive_mode_bits: float, adaptive_candidate_bits_value: float,
                            residual_coder: AdaptiveResidualCoder,
                            distortion_evaluator: Callable[[np.ndarray], Tuple[float, int, float]]) -> RDResult:
    qs = np.asarray(qs, np.float64)
    q = np.rint(np.array([actual.a-pred.a, actual.b-pred.b, actual.c-pred.c]) / qs).astype(np.int64)
    d = q * qs
    rec = Plane(pred.a+float(d[0]), pred.b+float(d[1]), pred.c+float(d[2]), actual.cx, actual.cy)
    ry, _ = plane_to_recon_y(rec, gt_y, scale, grid, min_depth, max_depth)
    depth_sse = float(np.sum((gt_y - ry) ** 2))
    projection_satd, projection_ref_poc, projection_valid_ratio = distortion_evaluator(ry)
    mb = float(adaptive_mode_bits)
    cb = float(adaptive_candidate_bits_value)
    rb_each = residual_coder.bits_each(q)
    rb = float(sum(rb_each))
    bits = mb + cb + rb
    return RDResult("predictor_residual", name, pred, rec, tuple(map(int,q)), idx,
                    mb, cb, rb, rb_each, bits, depth_sse,
                    projection_satd, projection_ref_poc, projection_valid_ratio,
                    projection_satd + lam * bits, ry, True, False)


def make_buffer_reuse(gt_y: np.ndarray, z: np.ndarray, scale: float, cx: float, cy: float,
                      grid: GridCache, min_depth: float,
                      distortion_evaluator: Callable[[np.ndarray], Tuple[float, int, float]]) -> RDResult:
    valid = np.isfinite(z) & (z > min_depth)
    ry = np.zeros_like(z)
    ry[valid] = z[valid] / scale
    ry = np.clip(np.rint(ry), 0, 1023)
    depth_sse = float(np.sum((gt_y - ry) ** 2))
    projection_satd, projection_ref_poc, projection_valid_ratio = distortion_evaluator(ry)
    p = fit_plane(z, cx, cy, grid, min_depth, valid) or Plane(0,0,0,cx,cy)
    return RDResult("buffer_reuse", "depth_buffer", p, p, (0,0,0), -1,
                    0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, depth_sse,
                    projection_satd, projection_ref_poc, projection_valid_ratio,
                    projection_satd, ry, False, True)


def qp_steps(args):
    s = 2.0 ** ((args.qp - args.qp_ref) / 6.0)
    return args.qa_base * s, args.qb_base * s, args.qc_base * s


def derive_lambda(args):
    return args.lambda_rd if args.lambda_rd is not None else args.lambda_scale * 2.0 ** ((args.qp - 12.0) / 3.0)



def parse_poc_set(text: str) -> set[int]:
    """Parse comma-separated POCs and inclusive ranges, e.g. 0,32,64-96."""
    result: set[int] = set()
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
            if hi < lo:
                raise ValueError(f"Invalid POC range: {token}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(token))
    if any(x < 0 for x in result):
        raise ValueError("POCs must be non-negative")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="GT depth fixed-block candidate RDO simulator with predictor-only and implicit depth-buffer reuse")
    p.add_argument("--gt-depth-yuv", required=True)
    p.add_argument("--sequence-yuv", required=True, help="Original/reconstructed video YUV420p10le used for projection Y SATD")
    p.add_argument("--sequence-stored-shift", type=int, default=0, help="Right shift applied after reading 16-bit sequence samples (e.g. 6 for MSB-aligned 10-bit)")
    p.add_argument("--camera-param", required=True)
    p.add_argument("--mv-csv", default="")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--forward-downsample", type=int, default=4)
    p.add_argument("--depth-buffer-downsample", type=int, default=4)
    p.add_argument("--max-forward-refs", type=int, default=2)
    p.add_argument("--default-ref-offset", type=int, default=1)
    p.add_argument(
        "--zero-anchor-pocs",
        default="",
        help=(
            "Comma-separated POCs/ranges to exclude from depth coding RDO and "
            "write as all-zero depth frames, e.g. '0,32' or '0,32,64-96'. "
            "These POCs cost zero depth bits and do not update probability models."
        ),
    )
    p.add_argument(
        "--exclude-zero-anchors-as-forward-refs",
        action="store_true",
        help=(
            "Also prohibit POCs listed by --zero-anchor-pocs from being used as "
            "reconstructed-depth forward-warp references."
        ),
    )
    p.add_argument("--coding-order", choices=["ra","sequential"], default="ra")
    p.add_argument("--ra-gop-size", type=int, default=32)
    p.add_argument("--min-mv-plane-points", type=int, default=3)
    p.add_argument("--min-parallax", type=float, default=1e-6)
    p.add_argument("--max-mv-reproj-error", type=float, default=1.5)
    p.add_argument("--min-forward-valid-ratio", type=float, default=0.25)
    p.add_argument("--candidate-idx-coding", choices=["fixed","truncated_unary"], default="truncated_unary")
    p.add_argument("--disable-buffer-reuse", action="store_true")
    p.add_argument("--disable-predictor-only", action="store_true")
    p.add_argument("--disable-predictor-residual", action="store_true")
    p.add_argument("--disable-direct", action="store_true")
    p.add_argument("--qp", type=int, default=37)
    p.add_argument("--qp-ref", type=int, default=37)
    p.add_argument("--qa-base", type=float, default=5e-3)
    p.add_argument("--qb-base", type=float, default=5e-3)
    p.add_argument("--qc-base", type=float, default=5e-2)
    p.add_argument("--prob-update-step", type=float, default=0.025)
    p.add_argument("--prob-min", type=float, default=0.01)
    p.add_argument("--prob-reset", choices=["sequence", "frame"], default="sequence")
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
    if min(a.block_size, a.forward_downsample, a.depth_buffer_downsample) <= 0:
        raise ValueError("Invalid block/downsample size")
    if min(a.qa_base, a.qb_base, a.qc_base) <= 0:
        raise ValueError("qsteps must be positive")
    if a.prob_update_step < 0.0:
        raise ValueError("--prob-update-step must be non-negative")
    if not (0.0 < a.prob_min < 1.0 / 6.0):
        raise ValueError("--prob-min must be in (0, 1/6)")
    if a.min_mv_plane_points < 1:
        raise ValueError("--min-mv-plane-points must be positive")
    if a.min_parallax <= 0 or a.max_mv_reproj_error < 0:
        raise ValueError("Invalid MV-depth reliability thresholds")
    if a.ra_gop_size <= 0:
        raise ValueError("--ra-gop-size must be positive")
    if a.disable_direct and a.disable_predictor_only and a.disable_predictor_residual:
        raise ValueError("All explicit coding modes are disabled")

    outs = [Path(a.out_recon_yuv), Path(a.out_block_csv), Path(a.out_summary_json)]
    for p in outs:
        if p.exists():
            if a.overwrite: p.unlink()
            else: raise RuntimeError(f"Output exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)

    header, cams = load_cameras(a.camera_param)
    scale = depth_scale_real(header)
    total = count_frames(a.gt_depth_yuv, a.width, a.height)
    sequence_total = count_frames(a.sequence_yuv, a.width, a.height)
    if sequence_total < total:
        raise ValueError(f"sequence YUV has fewer frames ({sequence_total}) than depth YUV ({total})")
    start = a.start_frame
    end = total if a.num_frames == 0 else min(total, start + a.num_frames)
    if start < 0 or start >= end: raise ValueError("Invalid frame range")

    zero_anchor_pocs = parse_poc_set(a.zero_anchor_pocs)
    zero_anchor_pocs_in_range = sorted(p for p in zero_anchor_pocs if start <= p < end)

    rows, block_refs, block_mv_rows = load_mv_csv(a.mv_csv, total, a.block_size)
    qa, qb, qc = qp_steps(a); lam = derive_lambda(a)
    lw = int(math.ceil(a.width / a.forward_downsample)); lh = int(math.ceil(a.height / a.forward_downsample))
    dbw = int(math.ceil(a.width / a.depth_buffer_downsample)); dbh = int(math.ceil(a.height / a.depth_buffer_downsample))
    grid = GridCache()
    coding_order = (build_ra_order(start, end, a.ra_gop_size)
                    if a.coding_order == "ra" else list(range(start, end)))

    mode_symbols = [
        x for x, disabled in (
            ("direct", a.disable_direct),
            ("predictor_only", a.disable_predictor_only),
            ("predictor_residual", a.disable_predictor_residual),
        ) if not disabled
    ]
    candidate_symbols = ["left", "top", "top_left", "spatial_all", "fw_ref", "fw_average"]

    def new_probability_models():
        return (
            SimpleAdaptiveProb(mode_symbols, a.prob_update_step, a.prob_min, "mode"),
            SimpleAdaptiveProb(candidate_symbols, a.prob_update_step, a.prob_min, "candidate"),
            AdaptiveResidualCoder(a.prob_update_step, a.prob_min),
        )

    sequence_mode_model, sequence_candidate_model, sequence_residual_coder = new_probability_models()

    print(f"block={a.block_size}, no split, forward={lw}x{lh}, reuse={dbw}x{dbh}, QP={a.qp}")
    print(f"qsteps={qa:.6g},{qb:.6g},{qc:.6g}, lambda={lam:.6g}, depth_scale={scale:.12g}")
    print(f"adaptive probability: step={a.prob_update_step}, p_min={a.prob_min}, reset={a.prob_reset}")
    print(f"zero anchor POCs    : {zero_anchor_pocs_in_range or 'none'}")
    print(f"exclude as FW refs  : {a.exclude_zero_anchors_as_forward_refs}")
    print(f"coding order        : {a.coding_order} {coding_order[:16]}{'...' if len(coding_order)>16 else ''}")
    print("spatial source      : neighboring-block MV-derived depth")
    print("forward source      : reconstructed reference depth")

    fields = ["poc","x","y","w","h","mode","mode_probability_before","candidate_count","candidate_idx","candidate","candidate_probability_before","mode_bits","candidate_bits","residual_bits","residual_bits_a","residual_bits_b","residual_bits_c","total_bits","residual_present","buffer_no_signal","q_a","q_b","q_c","pred_a","pred_b","pred_c","gt_a","gt_b","gt_c","recon_a","recon_b","recon_c","depth_sse","projection_satd","projection_ref_poc","projection_valid_ratio","cost","forward_refs"]
    frames=[]; global_modes={}; global_cands={}; total_bits=0.0; total_sse=0.0; total_satd=0.0
    simulated_frame_count = 0
    excluded_anchor_output_sse = 0.0

    reconstructed_depth_frames: Dict[int, np.ndarray] = {}
    with open(a.gt_depth_yuv,"rb") as fp, open(a.sequence_yuv,"rb") as seqfp, open(a.out_recon_yuv,"wb+") as outfp, open(a.out_block_csv,"w",newline="",encoding="utf-8") as cfp:
        wr=csv.DictWriter(cfp,fieldnames=fields); wr.writeheader(); gt_cache={}
        def gt(poc):
            if poc not in gt_cache: gt_cache[poc]=read_y(fp,poc,a.width,a.height)
            return gt_cache[poc]
        sequence_cache = {}
        def sequence_y(poc):
            if poc not in sequence_cache:
                y = read_y(seqfp,poc,a.width,a.height)
                if a.sequence_stored_shift:
                    y = np.floor(y / float(1 << a.sequence_stored_shift))
                sequence_cache[poc] = y
            return sequence_cache[poc]

        for ord_idx,poc in enumerate(coding_order):
            if a.prob_reset == "frame":
                mode_model, candidate_model, residual_coder = new_probability_models()
            else:
                mode_model, candidate_model, residual_coder = (
                    sequence_mode_model, sequence_candidate_model, sequence_residual_coder)
            if poc not in cams: raise KeyError(f"No camera for POC {poc}")
            gy=gt(poc); gz=gy*scale; recon=np.zeros_like(gy)
            sequence_target = sequence_y(poc)

            # Explicitly excluded I/anchor frame: no block RDO, no depth buffer,
            # no probability update, and zero depth is written to the output.
            if poc in zero_anchor_pocs:
                reconstructed_depth_frames[poc] = recon.copy()
                outfp.seek((poc-start) * frame_size(a.width, a.height))
                write_depth_frame(outfp, recon, a.width, a.height)
                anchor_sse = float(np.sum(gy * gy))
                anchor_mse = anchor_sse / (a.width * a.height)
                anchor_psnr = (
                    float("inf") if anchor_mse == 0.0
                    else 10.0 * math.log10(1023.0 ** 2 / anchor_mse)
                )
                excluded_anchor_output_sse += anchor_sse
                frames.append({
                    "poc": poc,
                    "excluded_anchor": True,
                    "zero_filled": True,
                    "num_blocks": 0,
                    "frame_bits": 0.0,
                    "frame_bpp": 0.0,
                    "frame_sse": anchor_sse,
                    "frame_mse": anchor_mse,
                    "frame_psnr": anchor_psnr,
                    "frame_reference_pocs": [],
                    "forward_buffers_built": [],
                    "mode_selection_counts": {},
                    "candidate_selection_counts": {},
                    "buffer_reuse_ratio": 0.0,
                    "predictor_only_ratio": 0.0,
                    "predictor_residual_ratio": 0.0,
                    "direct_ratio": 0.0,
                    "final_mode_probabilities": mode_model.snapshot(),
                    "final_candidate_probabilities": candidate_model.snapshot(),
                    "final_residual_probabilities": residual_coder.snapshot(),
                })
                ratio=(ord_idx+1)/len(coding_order); width=30; fill=int(round(width*ratio))
                print(
                    f"\r[{'#'*fill}{'-'*(width-fill)}] {ord_idx+1}/{len(coding_order)} "
                    f"POC={poc} ANCHOR-SKIP bits=0 zero-filled",
                    end="", flush=True,
                )
                continue

            simulated_frame_count += 1

            def ref_allowed(ref_poc: int) -> bool:
                return not (
                    a.exclude_zero_anchors_as_forward_refs
                    and ref_poc in zero_anchor_pocs
                )

            frame_refs=[]
            for r in rows[poc]:
                if (
                    r.ref_poc != poc
                    and r.ref_poc in cams
                    and ref_allowed(r.ref_poc)
                    and r.ref_poc not in frame_refs
                ):
                    frame_refs.append(r.ref_poc)
            default_ref = poc - a.default_ref_offset
            if not frame_refs and default_ref in cams and ref_allowed(default_ref):
                frame_refs = [default_ref]
            frame_refs=frame_refs[:a.max_forward_refs]
            fw_cache = {}
            for ref in frame_refs:
                if ref in reconstructed_depth_frames:
                    fw_cache[ref] = forward_warp_lowres(
                        reconstructed_depth_frames[ref] * scale, cams[ref], cams[poc], lw, lh)
            reuse=DepthReuseBuffer(a.width,a.height,a.depth_buffer_downsample,a.min_depth)
            rec_planes={}; fbits=0.0; fsse=0.0; fsatd=0.0; fmodes={}; fcands={}; nblocks=0
            gw=int(math.ceil(a.width/a.block_size)); gh=int(math.ceil(a.height/a.block_size))

            for byi in range(gh):
                by=byi*a.block_size; bh=min(a.block_size,a.height-by)
                for bxi in range(gw):
                    bx=bxi*a.block_size; bw=min(a.block_size,a.width-bx); nblocks+=1
                    cx=bx+(bw-1)/2; cy=by+(bh-1)/2
                    gyb=gy[by:by+bh,bx:bx+bw]; gzb=gz[by:by+bh,bx:bx+bw]
                    actual=fit_plane(gzb,cx,cy,grid,a.min_depth) or Plane(0,0,1/a.max_depth,cx,cy)
                    cands=[]; refs=[]
                    target_y_block = sequence_target[by:by+bh,bx:bx+bw]

                    distortion_refs = []
                    for ref in block_refs[poc].get((bxi,byi),[]):
                        if ref != poc and ref in cams and ref_allowed(ref) and ref not in distortion_refs:
                            distortion_refs.append(ref)
                    for ref in frame_refs:
                        if ref not in distortion_refs:
                            distortion_refs.append(ref)
                    distortion_refs = distortion_refs[:a.max_forward_refs]
                    distortion_evaluator = make_projection_satd_evaluator(
                        poc, bx, by, target_y_block, distortion_refs, sequence_y,
                        cams, scale, a.min_depth)

                    if not a.disable_buffer_reuse and reuse.can_reuse(bx,by,bw,bh):
                        mode_prob_before = 1.0
                        candidate_prob_before = 0.0
                        best=make_buffer_reuse(gyb,reuse.reconstruct(bx,by,bw,bh),scale,cx,cy,grid,a.min_depth,distortion_evaluator)
                    else:
                        spatial_mv_groups = []
                        spatial_mv_group_count = 0
                        for name,key in (("left",(bxi-1,byi)),("top",(bxi,byi-1)),("top_left",(bxi-1,byi-1))):
                            mv_group = block_mv_rows[poc].get(key, [])
                            if mv_group:
                                p = fit_plane_from_mv_rows(
                                    mv_group, cams, cx, cy, a.min_parallax,
                                    a.max_mv_reproj_error, a.min_depth, a.max_depth,
                                    a.min_mv_plane_points)
                                if p is not None:
                                    cands.append((name,p))
                                    spatial_mv_groups.extend(mv_group)
                                    spatial_mv_group_count += 1
                        if spatial_mv_group_count >= 2:
                            p = fit_plane_from_mv_rows(
                                spatial_mv_groups, cams, cx, cy, a.min_parallax,
                                a.max_mv_reproj_error, a.min_depth, a.max_depth,
                                a.min_mv_plane_points)
                            if p is not None:
                                cands.append(("spatial_all",p))

                        for ref in block_refs[poc].get((bxi,byi),[]):
                            if (
                                ref != poc
                                and ref in cams
                                and ref_allowed(ref)
                                and ref not in refs
                            ):
                                refs.append(ref)
                        for ref in frame_refs:
                            if ref not in refs: refs.append(ref)
                        refs=refs[:a.max_forward_refs]; fw_blocks=[]
                        for ref in refs:
                            if ref not in fw_cache and ref in reconstructed_depth_frames and ref in cams:
                                fw_cache[ref]=forward_warp_lowres(
                                    reconstructed_depth_frames[ref]*scale,cams[ref],cams[poc],lw,lh)
                            if ref not in fw_cache: continue
                            zb,mb=extract_fw_block(*fw_cache[ref],bx,by,bw,bh,a.width,a.height)
                            if np.mean(mb)<a.min_forward_valid_ratio: continue
                            p=fit_plane(zb,cx,cy,grid,a.min_depth,mb)
                            if p is not None: cands.append((f"fw_ref_{ref}",p)); fw_blocks.append((zb,mb))
                        if len(fw_blocks)>=2:
                            comb=combine_fw(fw_blocks)
                            if comb is not None and np.mean(comb[1])>=a.min_forward_valid_ratio:
                                p=fit_plane(comb[0],cx,cy,grid,a.min_depth,comb[1])
                                if p is not None: cands.append(("fw_average",p))

                        results=[]
                        available_modes = []
                        if not a.disable_direct:
                            available_modes.append("direct")
                        if cands and not a.disable_predictor_only:
                            available_modes.append("predictor_only")
                        if cands and not a.disable_predictor_residual:
                            available_modes.append("predictor_residual")

                        available_candidate_names = [name for name, _ in cands]

                        if not a.disable_direct:
                            results.append(make_direct(
                                actual, gyb, scale, (qa,qb,qc), lam, grid,
                                a.min_depth, a.max_depth,
                                mode_model.bits("direct", available_modes), residual_coder, distortion_evaluator))

                        for idx,(name,pred) in enumerate(cands):
                            cbits = adaptive_candidate_bits(candidate_model, name, available_candidate_names)
                            if not a.disable_predictor_only:
                                results.append(make_predictor_only(
                                    name,pred,idx,len(cands),gyb,scale,a.candidate_idx_coding,
                                    lam,grid,a.min_depth,a.max_depth,
                                    mode_model.bits("predictor_only", available_modes), cbits, distortion_evaluator))
                            if not a.disable_predictor_residual:
                                results.append(make_predictor_residual(
                                    name,pred,actual,idx,len(cands),gyb,scale,a.candidate_idx_coding,
                                    (qa,qb,qc),lam,grid,a.min_depth,a.max_depth,
                                    mode_model.bits("predictor_residual", available_modes), cbits,
                                    residual_coder, distortion_evaluator))

                        if not results: raise RuntimeError(f"No RDO mode at POC={poc}, block=({bx},{by})")
                        best=min(results,key=lambda r:r.cost)

                        mode_prob_before = mode_model.probs[best.mode] / sum(
                            mode_model.probs[x] for x in available_modes)
                        if best.mode in ("predictor_only", "predictor_residual"):
                            selected_class = candidate_probability_class(best.name)
                            av_classes = [candidate_probability_class(x) for x in available_candidate_names]
                            uniq_classes = list(dict.fromkeys(av_classes))
                            candidate_prob_before = (
                                candidate_model.probs[selected_class] /
                                sum(candidate_model.probs[x] for x in uniq_classes) /
                                sum(x == selected_class for x in av_classes)
                            )
                        else:
                            candidate_prob_before = 0.0

                        mode_model.update(best.mode)
                        if best.mode in ("predictor_only", "predictor_residual"):
                            candidate_model.update(candidate_probability_class(best.name))
                        if best.residual_present:
                            residual_coder.update(best.q)

                        reuse.commit(bx,by,best.recon_y*scale)

                    recon[by:by+bh,bx:bx+bw]=best.recon_y; rec_planes[(bxi,byi)]=best.recon
                    fbits+=best.bits; fsse+=best.depth_sse; fsatd+=best.projection_satd
                    fmodes[best.mode]=fmodes.get(best.mode,0)+1; fcands[best.name]=fcands.get(best.name,0)+1
                    global_modes[best.mode]=global_modes.get(best.mode,0)+1; global_cands[best.name]=global_cands.get(best.name,0)+1
                    wr.writerow({"poc":poc,"x":bx,"y":by,"w":bw,"h":bh,"mode":best.mode,"mode_probability_before":mode_prob_before,"candidate_count":len(cands),"candidate_idx":best.candidate_idx,"candidate":best.name,"candidate_probability_before":candidate_prob_before,"mode_bits":best.mode_bits,"candidate_bits":best.candidate_bits,"residual_bits":best.residual_bits,"residual_bits_a":best.residual_bits_each[0],"residual_bits_b":best.residual_bits_each[1],"residual_bits_c":best.residual_bits_each[2],"total_bits":best.bits,"residual_present":int(best.residual_present),"buffer_no_signal":int(best.buffer_no_signal),"q_a":best.q[0],"q_b":best.q[1],"q_c":best.q[2],"pred_a":best.pred.a,"pred_b":best.pred.b,"pred_c":best.pred.c,"gt_a":actual.a,"gt_b":actual.b,"gt_c":actual.c,"recon_a":best.recon.a,"recon_b":best.recon.b,"recon_c":best.recon.c,"depth_sse":best.depth_sse,"projection_satd":best.projection_satd,"projection_ref_poc":best.projection_ref_poc,"projection_valid_ratio":best.projection_valid_ratio,"cost":best.cost,"forward_refs":"|".join(map(str,refs))})

            reconstructed_depth_frames[poc] = recon.copy()
            outfp.seek((poc-start) * frame_size(a.width, a.height))
            write_depth_frame(outfp,recon,a.width,a.height)
            mse=fsse/(a.width*a.height); psnr=float("inf") if mse==0 else 10*math.log10(1023**2/mse)
            frames.append({"poc":poc,"num_blocks":nblocks,"frame_bits":fbits,"frame_bpp":fbits/(a.width*a.height),"frame_sse":fsse,"frame_projection_satd":fsatd,"frame_mse":mse,"frame_psnr":psnr,"frame_reference_pocs":frame_refs,"forward_buffers_built":sorted(fw_cache.keys()),"mode_selection_counts":fmodes,"candidate_selection_counts":fcands,"buffer_reuse_ratio":fmodes.get("buffer_reuse",0)/nblocks,"predictor_only_ratio":fmodes.get("predictor_only",0)/nblocks,"predictor_residual_ratio":fmodes.get("predictor_residual",0)/nblocks,"direct_ratio":fmodes.get("direct",0)/nblocks,"final_mode_probabilities":mode_model.snapshot(),"final_candidate_probabilities":candidate_model.snapshot(),"final_residual_probabilities":residual_coder.snapshot()})
            total_bits+=fbits; total_sse+=fsse; total_satd+=fsatd
            ratio=(ord_idx+1)/len(coding_order); width=30; fill=int(round(width*ratio))
            print(f"\r[{'#'*fill}{'-'*(width-fill)}] {ord_idx+1}/{len(coding_order)} POC={poc} bits={fbits:.1f} reuse={fmodes.get('buffer_reuse',0)/nblocks:.3f} predOnly={fmodes.get('predictor_only',0)/nblocks:.3f} depthPSNR={psnr:.3f} SATD={fsatd:.1f}",end="",flush=True)
    print()

    simulated_pixels = a.width * a.height * simulated_frame_count
    if simulated_pixels > 0:
        mse = total_sse / simulated_pixels
        psnr = float("inf") if mse == 0 else 10 * math.log10(1023**2 / mse)
        overall_bpp = total_bits / simulated_pixels
    else:
        mse = 0.0
        psnr = float("inf")
        overall_bpp = 0.0
    summary={
        "gt_depth_yuv":a.gt_depth_yuv,
        "sequence_yuv":a.sequence_yuv,
        "sequence_stored_shift":a.sequence_stored_shift,
        "camera_param":a.camera_param,
        "mv_csv":a.mv_csv,
        "width":a.width,
        "height":a.height,
        "start_frame":start,
        "num_frames":end-start,
        "simulated_frame_count":simulated_frame_count,
        "zero_anchor_pocs":sorted(zero_anchor_pocs),
        "zero_anchor_pocs_in_range":zero_anchor_pocs_in_range,
        "exclude_zero_anchors_as_forward_refs":bool(a.exclude_zero_anchors_as_forward_refs),
        "zero_anchor_rule":"no block RDO, zero depth output, zero depth bits, no buffer/probability update",
        "block_size":a.block_size,
        "block_split":False,"coding_order":a.coding_order,"coding_poc_order":coding_order,"ra_gop_size":a.ra_gop_size,
        "forward_downsample":a.forward_downsample,
        "forward_buffer_size":[lw,lh],
        "depth_buffer_downsample":a.depth_buffer_downsample,
        "depth_buffer_size":[dbw,dbh],
        "depth_buffer_rule":"all covered cells valid -> implicit zero-bit reuse; otherwise run explicit RDO and commit winner",
        "reference_depth_source":"previously reconstructed depth frame only",
        "depth_scale_real":scale,
        "plane_equation":"1/z = a*(x-cx)+b*(y-cy)+c",
        "qp":a.qp,
        "qp_ref":a.qp_ref,
        "quantization":{"qa":qa,"qb":qb,"qc":qc,"formula":"base*2^((QP-QP_ref)/6)"},
        "lambda_rd":lam,
        "rdo_distortion":"projection-domain Y SATD using current candidate depth and best available reference",
        "satd_transform":"4x4 Hadamard, sum(abs(coeff))/2, tiled with zero padding",
        "projection_invalid_fill":"collocated reference Y block",
        "rate_model":{
            "buffer_reuse":"0 bits",
            "direct":"adaptive mode bits + adaptive residual syntax",
            "predictor_only":"adaptive mode bits + adaptive candidate bits",
            "predictor_residual":"adaptive mode bits + adaptive candidate bits + adaptive residual syntax",
            "candidate_idx_coding":"adaptive -log2(p) with fw_ref class split among occurrences"
        },
        "probability_model":{
            "update":"selected += step; every unselected symbol -= step; clip and normalize",
            "update_step":a.prob_update_step,
            "p_min":a.prob_min,
            "reset":a.prob_reset,
            "mode_symbols":mode_symbols,
            "candidate_symbols":candidate_symbols,
            "final_mode_probabilities":sequence_mode_model.snapshot(),
            "final_candidate_probabilities":sequence_candidate_model.snapshot(),
            "residual_syntax": {
                "ab_shared_distribution": True,
                "c_separate_distribution": True,
                "coding_order": ["q_a", "q_b", "q_c"],
                "coefficient_syntax": "adaptive zero/nonzero; fixed 1-bit sign; adaptive magnitude class 1/2/3/4/gt4; ue(abs(q)-5) for gt4",
                "final_probabilities": sequence_residual_coder.snapshot()
            }
        },
        "candidate_types":["left(MV-derived)","top(MV-derived)","top_left(MV-derived)","spatial_all(MV-derived)","fw_ref_<POC>(recon-depth)","fw_average(recon-depth)"],
        "mode_types":["buffer_reuse","predictor_only","predictor_residual","direct"],
        "total_bits":total_bits,
        "overall_bpp":overall_bpp,
        "overall_sse":total_sse,
        "overall_projection_satd":total_satd,
        "overall_mse":mse,
        "overall_psnr":psnr,
        "overall_metric_scope":"non-excluded simulated frames only",
        "excluded_anchor_output_sse":excluded_anchor_output_sse,
        "mode_selection_counts":global_modes,
        "candidate_selection_counts":global_cands,
        "frames":frames,
        "out_recon_yuv":a.out_recon_yuv,
        "out_block_csv":a.out_block_csv
    }
    with open(a.out_summary_json,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f"Recon YUV : {a.out_recon_yuv}\nBlock CSV : {a.out_block_csv}\nSummary   : {a.out_summary_json}\nTotal bits: {total_bits:.3f}\nBPP       : {overall_bpp:.9f}\nProjection SATD: {total_satd:.3f}\nDepth PSNR: {psnr:.6f} dB")


if __name__ == "__main__":
    main()
