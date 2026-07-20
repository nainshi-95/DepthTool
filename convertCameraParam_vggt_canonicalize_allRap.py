#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge rap0...rapK geometry NPZ files.

Output:
- one camera JSONL:
    overlapping boundary POC is written once per RAP (two camera lines)
- one depth YUV420p10le:
    each absolute POC is written once
    for overlap, the earlier RAP/GOP depth is used

Supported NPZ:
1) VGGT: depth_original, extrinsic, intrinsic_original
2) canonical: depth_canonical-like, K_fixed, rvec_abs_*, tvec_abs_*
"""

from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def scalar_str(x: Any) -> str:
    a = np.asarray(x)
    v = a.item() if a.shape == () else a.tolist()
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def safe_name(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).rstrip(" .")
    return s or "unnamed"


def first_key(z, names):
    return next((k for k in names if k in z), None)


def R_from_rvec(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def rvec_from_R(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


def to4(E):
    E = np.asarray(E, np.float64)
    if E.shape == (4, 4):
        return E.copy()
    if E.shape == (3, 4):
        T = np.eye(4)
        T[:3] = E
        return T
    raise ValueError(f"bad extrinsic shape: {E.shape}")


def abs_pose_to_extrinsic(rv, tv):
    rv, tv = np.asarray(rv, np.float64), np.asarray(tv, np.float64)
    if rv.shape != tv.shape or rv.ndim != 2 or rv.shape[1] != 3:
        raise ValueError(f"bad pose shapes: {rv.shape}, {tv.shape}")
    E = np.zeros((len(rv), 3, 4), np.float64)
    for i in range(len(rv)):
        E[i, :3, :3] = R_from_rvec(rv[i])
        E[i, :3, 3] = tv[i]
    return E.astype(np.float32)


def convert_pose(E, mode):
    W = [to4(x) for x in E]
    n = len(W)
    rv = np.zeros((n, 3), np.float32)
    tv = np.zeros((n, 3), np.float32)

    if mode == "current_to_previous":
        for i in range(1, n):
            T = W[i - 1] @ np.linalg.inv(W[i])
            rv[i] = rvec_from_R(T[:3, :3])
            tv[i] = T[:3, 3]
    elif mode == "gop_local":
        C2W0 = np.linalg.inv(W[0])
        for i in range(1, n):
            T = W[i] @ C2W0
            rv[i] = rvec_from_R(T[:3, :3])
            tv[i] = T[:3, 3]
    elif mode == "absolute":
        for i, T in enumerate(W):
            rv[i] = rvec_from_R(T[:3, :3])
            tv[i] = T[:3, 3]
    else:
        raise ValueError(mode)
    return rv, tv


def cleanup_depth(d, path, key):
    d = np.asarray(d, np.float32)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim == 4 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 3:
        raise ValueError(f"{path}: {key} must be [N,H,W], got {d.shape}")
    return d


def repeat_K(K, n):
    K = np.asarray(K, np.float32)
    if K.shape == (3, 3):
        return np.repeat(K[None], n, axis=0)
    if K.shape == (n, 3, 3):
        return K
    raise ValueError(f"bad K shape: {K.shape}")


def infer_seq_rap(path: Path, rap_name=None, rap_idx=None):
    stem = path.stem
    for suffix in [
        "_fixedK_gop_nn_frame_scale_geometry",
        "_fixedK_gop_nn_scale_geometry",
        "_fixedK_gop_nn_geometry",
        "_fixedK_gop_geometry",
        "_canonical_geometry",
        "_vggt_omega_outputs",
        "_outputs",
    ]:
        while stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    m = re.match(r"^(.*)_rap(\d+)$", stem)
    if m:
        return m.group(1), f"rap{m.group(2)}", int(m.group(2))

    if rap_name and re.fullmatch(r"rap\d+", rap_name):
        ri = int(rap_name[3:])
        return stem, rap_name, ri
    if rap_idx is not None:
        return stem, f"rap{int(rap_idx)}", int(rap_idx)
    return stem, "rap0", 0


def load_npz(path: Path, input_mode: str):
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        dk = first_key(z, ["depth_canonical", "depth_final",
                           "depth_optimized", "depth_modified"])
        rk = first_key(z, ["rvec_abs_final", "rvec_abs_stage3_joint",
                           "rvec_abs_stage2_t_nn", "rvec_abs_stage1_rt",
                           "rvec_abs_init"])
        tk = first_key(z, ["tvec_abs_final", "tvec_abs_stage3_joint",
                           "tvec_abs_stage3_joint_unscaled",
                           "tvec_abs_stage2_t_nn", "tvec_abs_stage1_rt",
                           "tvec_abs_init"])
        canonical = dk is not None and "K_fixed" in keys and rk and tk
        vggt = all(k in keys for k in
                   ["depth_original", "extrinsic", "intrinsic_original"])

        mode = input_mode
        if mode == "auto":
            mode = "canonical" if canonical else "vggt" if vggt else "unknown"

        if mode == "canonical":
            if not canonical:
                raise KeyError("not a canonical geometry NPZ")
            depth = cleanup_depth(z[dk], path, dk)
            n = len(depth)
            K = repeat_K(z["K_fixed"], n)
            E = abs_pose_to_extrinsic(z[rk], z[tk])
            fixed_K = True
            source_type = "canonical_fixedK_optimized"
        elif mode == "vggt":
            if not vggt:
                raise KeyError("not a VGGT geometry NPZ")
            depth = cleanup_depth(z["depth_original"], path, "depth_original")
            n = len(depth)
            E = np.asarray(z["extrinsic"], np.float32)
            K = np.asarray(z["intrinsic_original"], np.float32)
            if E.ndim == 4 and E.shape[0] == 1:
                E = E[0]
            if K.ndim == 4 and K.shape[0] == 1:
                K = K[0]
            fixed_K = False
            source_type = "vggt_original"
        else:
            raise KeyError("unsupported NPZ")

        if E.shape[0] != n or E.shape[1:] not in [(3, 4), (4, 4)]:
            raise ValueError(f"bad extrinsic: {E.shape}")
        if K.shape != (n, 3, 3):
            raise ValueError(f"bad intrinsic: {K.shape}")

        frame_indices = (np.asarray(z["frame_indices"], np.int64).reshape(-1)
                         if "frame_indices" in z else np.arange(n))
        if len(frame_indices) != n:
            raise ValueError("frame_indices length mismatch")

        rn = scalar_str(z["rap_name"]) if "rap_name" in z else None
        ri = int(np.asarray(z["rap_index"]).item()) if "rap_index" in z else None

    seq, rap_name, rap_idx = infer_seq_rap(path, rn, ri)
    return dict(path=path, sequence=seq, rap_name=rap_name, rap_idx=rap_idx,
                depth=depth.astype(np.float32), E=E.astype(np.float32),
                K=K.astype(np.float32), fixed_K=fixed_K,
                source_type=source_type,
                frame_indices=[int(x) for x in frame_indices])


def Kdict(K, z_sign):
    return dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]),
                z_sign=float(z_sign))


def Kvec(K):
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float64)


def Kdelta(Ks, force_zero):
    out = np.zeros((len(Ks), 4), np.float32)
    if not force_zero and len(Ks) > 1:
        v = np.stack([Kvec(K) for K in Ks])
        out[1:] = v[1:] - v[:-1]
    return out


def choose_scale(depth, percentile, precision):
    valid = np.isfinite(depth) & (depth > 0)
    ref = float(np.percentile(depth[valid], percentile)) if np.any(valid) else 1023.
    if not np.isfinite(ref) or ref <= 0:
        ref = 1023.
    sf = ref / 1023.
    si = max(1, int(round(sf * precision)))
    return dict(depth_scale=si, depth_scale_precision=precision,
                depth_scale_real=si / precision, depth_ref=ref,
                depth_percentile=percentile, max_code=1023)


def write_depth(path, depth, meta):
    n, h, w = depth.shape
    if h % 2 or w % 2:
        raise ValueError("YUV420 requires even width/height")
    ensure_parent(path)
    uv = np.full((h // 2, w // 2), 512, "<u2")
    sr = float(meta["depth_scale_real"])
    maes, rmses, clips = [], [], []
    with open(path, "wb") as f:
        for d in depth:
            y = np.nan_to_num(d, nan=0., posinf=1023*sr, neginf=0.)
            y = np.clip(np.round(y / sr), 0, 1023).astype("<u2")
            f.write(y.tobytes()); f.write(uv.tobytes()); f.write(uv.tobytes())
            valid = np.isfinite(d) & (d > 0)
            if np.any(valid):
                e = y.astype(np.float32)[valid] * sr - d[valid]
                maes.append(float(np.mean(np.abs(e))))
                rmses.append(float(np.sqrt(np.mean(e*e))))
                clips.append(float(np.mean(y[valid] >= 1023)))
    return dict(mean_mae=float(np.mean(maes)) if maes else None,
                mean_rmse=float(np.mean(rmses)) if rmses else None,
                max_clip_ratio=max(clips) if clips else 0.)


def process_sequence(seq, items, args):
    items = sorted(items, key=lambda x: x["rap_idx"])
    h, w = items[0]["depth"].shape[1:]

    for it in items:
        if it["depth"].shape[1:] != (h, w):
            raise ValueError(f"resolution mismatch: {it['path']}")
        it["rvec"], it["tvec"] = convert_pose(it["E"], args.pose_mode)

    # Earlier RAP wins for depth at duplicate POC.
    depth_by_poc, depth_owner = {}, {}
    camera_count_by_poc = {}
    for it in items:
        for li, poc in enumerate(it["frame_indices"]):
            camera_count_by_poc[poc] = camera_count_by_poc.get(poc, 0) + 1
            if poc not in depth_by_poc:
                depth_by_poc[poc] = it["depth"][li]
                depth_owner[poc] = it["rap_idx"]

    pocs = sorted(depth_by_poc)
    if pocs != list(range(pocs[0], pocs[-1] + 1)):
        raise ValueError("merged depth POCs are not contiguous")
    poc_to_depth_idx = {p: i for i, p in enumerate(pocs)}
    merged_depth = np.stack([depth_by_poc[p] for p in pocs])

    out_dir = Path(args.output_dir) if args.output_dir else items[0]["path"].parent
    tag = f"_{args.output_tag.strip('_')}" if args.output_tag.strip("_") else ""
    base = safe_name(seq)
    out_yuv = out_dir / f"{base}_depth_merged{tag}.yuv"
    out_json = out_dir / f"{base}_camParam_merged{tag}.jsonl"

    if not args.overwrite and (out_yuv.exists() or out_json.exists()):
        print(f"[SKIP] exists: {seq}")
        return False

    meta = choose_scale(merged_depth, args.depth_percentile,
                        args.depth_scale_precision)
    qstats = write_depth(out_yuv, merged_depth, meta)

    overlaps = sorted(p for p, c in camera_count_by_poc.items() if c > 1)
    header = {
        "type": "header",
        "format": "camparam_v3_multi_rap_merged",
        "sequence_name": seq,
        "rap_count": len(items),
        "camera_record_count": sum(len(x["frame_indices"]) for x in items),
        "unique_depth_frame_count": len(pocs),
        "depth_frame_pocs": pocs,
        "overlap_pocs": overlaps,
        "overlap_policy": {
            "camera": "all RAP-local records kept; overlap POC has two lines",
            "depth": "one frame only; earlier RAP/GOP wins"
        },
        "width": w, "height": h, "bit_depth": 10,
        "depth_yuv": out_yuv.name,
        **meta,
        "depth_quant_summary": qstats,
        "pose_mode": args.pose_mode,
        "pose_convention": {
            "current_to_previous":
                "X_prev=R*X_cur+t; each RAP local_poc 0 is identity",
            "gop_local":
                "X_i=R*X_0+t; each RAP local_poc 0 is identity",
            "absolute": "camera_from_world in each NPZ coordinate system"
        }[args.pose_mode],
        "raps": [{
            "rap_idx": x["rap_idx"], "rap_name": x["rap_name"],
            "source_npz": str(x["path"].resolve()),
            "frame_indices": x["frame_indices"],
            "fixed_intrinsic": x["fixed_K"],
            "initial_intrinsic": Kdict(x["K"][0], args.z_sign)
        } for x in items]
    }

    ensure_parent(out_json)
    record_idx = 0
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for it in items:
            kd = Kdelta(it["K"], it["fixed_K"])
            for li, poc in enumerate(it["frame_indices"]):
                rec = {
                    "type": "frame",
                    "camera_record_idx": record_idx,
                    "rap_idx": it["rap_idx"],
                    "rap_name": it["rap_name"],
                    "local_poc": li,
                    "poc": poc,
                    "frame_idx": poc,
                    "depth_frame_idx": poc_to_depth_idx[poc],
                    "depth_source_rap_idx": depth_owner[poc],
                    "is_overlap": camera_count_by_poc[poc] > 1,
                    "is_depth_owner": depth_owner[poc] == it["rap_idx"],
                    "rvec": [float(x) for x in it["rvec"][li]],
                    "tvec": [float(x) for x in it["tvec"][li]],
                    "intrinsic": Kdict(it["K"][li], args.z_sign),
                    "intrinsic_delta": [float(x) for x in kd[li]]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                record_idx += 1

    print(f"[OK] {seq}")
    print(f"     RAPs           : {[x['rap_name'] for x in items]}")
    print(f"     overlap POCs   : {overlaps}")
    print(f"     camera records : {record_idx}")
    print(f"     depth frames   : {len(pocs)}")
    print(f"     camera JSONL   : {out_json}")
    print(f"     depth YUV      : {out_yuv}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--input-mode", choices=["auto", "vggt", "canonical"],
                    default="auto")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--output-tag", default="")
    ap.add_argument("--depth-percentile", type=float, default=99.9)
    ap.add_argument("--depth-scale-precision", type=int, default=100000)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--pose-mode",
                    choices=["current_to_previous", "gop_local", "absolute"],
                    default="current_to_previous")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-skip-unrecognized", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f"not a directory: {root}")

    groups = {}
    for p in sorted(root.rglob(args.pattern)):
        try:
            d = load_npz(p, args.input_mode)
            groups.setdefault(d["sequence"], []).append(d)
        except Exception as e:
            if args.no_skip_unrecognized:
                raise
            print(f"[SKIP] {p}: {type(e).__name__}: {e}")

    ok = fail = skip = 0
    for seq, items in sorted(groups.items()):
        try:
            if process_sequence(seq, items, args):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {seq}: {type(e).__name__}: {e}")

    print(f"Done. converted={ok}, skipped={skip}, failed={fail}")


if __name__ == "__main__":
    main()
