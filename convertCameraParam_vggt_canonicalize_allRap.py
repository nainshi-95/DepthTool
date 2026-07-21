#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient merge of gop0...gopK geometry NPZ files.

The output format and overlap policy are the same as the original script:

- Camera JSONL:
    all GOP-local camera records are retained;
    an overlapping boundary POC therefore has two camera records.
- Depth YUV420p10le:
    each absolute POC is written once;
    when GOPs overlap, the earlier GOP owns the depth frame.

Memory strategy:
1) The initial directory scan stores only small NPZ descriptors, not depth/E/K.
2) Only one GOP NPZ is fully loaded at a time.
3) Merged depth is streamed directly to YUV; np.stack(merged_depth) is avoided.
4) Quantization statistics are accumulated in row chunks.
5) Camera data is reloaded GOP-by-GOP while writing JSONL.

Supported NPZ:
1) VGGT:
   depth_original, extrinsic, intrinsic_original
2) Canonical:
   depth_canonical-like, K_fixed, rvec_abs_*, tvec_abs_*
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

DEPTH_KEYS = [
    "depth_canonical",
    "depth_final",
    "depth_optimized",
    "depth_modified",
]

RVEC_KEYS = [
    "rvec_abs_final",
    "rvec_abs_stage3_joint",
    "rvec_abs_stage2_t_nn",
    "rvec_abs_stage1_rt",
    "rvec_abs_init",
]

TVEC_KEYS = [
    "tvec_abs_final",
    "tvec_abs_stage3_joint",
    "tvec_abs_stage3_joint_unscaled",
    "tvec_abs_stage2_t_nn",
    "tvec_abs_stage1_rt",
    "tvec_abs_init",
]


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
    return cv2.Rodrigues(
        np.asarray(v, np.float64).reshape(3, 1)
    )[0]


def rvec_from_R(R):
    return cv2.Rodrigues(
        np.asarray(R, np.float64)
    )[0].reshape(3)


def to4(E):
    E = np.asarray(E, np.float64)
    if E.shape == (4, 4):
        return E.copy()
    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3] = E
        return T
    raise ValueError(f"bad extrinsic shape: {E.shape}")


def abs_pose_to_extrinsic(rv, tv):
    rv = np.asarray(rv, np.float64)
    tv = np.asarray(tv, np.float64)

    if rv.shape != tv.shape or rv.ndim != 2 or rv.shape[1] != 3:
        raise ValueError(f"bad pose shapes: {rv.shape}, {tv.shape}")

    E = np.zeros((len(rv), 3, 4), np.float32)
    for i in range(len(rv)):
        E[i, :3, :3] = R_from_rvec(rv[i]).astype(np.float32)
        E[i, :3, 3] = tv[i].astype(np.float32)
    return E


def convert_pose(E, mode):
    """
    E is small compared with depth, so this function may construct one
    temporary 4x4 matrix per frame without materially affecting RAM use.
    """
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
    d = np.asarray(d)

    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim == 4 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 3:
        raise ValueError(
            f"{path}: {key} must be [N,H,W], got {d.shape}"
        )

    # Avoid a copy when NPZ already stores float32.
    return d.astype(np.float32, copy=False)


def repeat_K(K, n):
    K = np.asarray(K)

    if K.shape == (3, 3):
        # K is tiny; repeating it has negligible memory cost.
        return np.repeat(
            K.astype(np.float32, copy=False)[None],
            n,
            axis=0,
        )

    if K.shape == (n, 3, 3):
        return K.astype(np.float32, copy=False)

    raise ValueError(f"bad K shape: {K.shape}")


def infer_seq_gop(
    path: Path,
    source_group_name=None,
    source_group_idx=None,
):
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

    m = re.match(
        r"^(.*)_(?:rap|gop)(\d+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if m:
        idx = int(m.group(2))
        return m.group(1), f"gop{idx}", idx

    if (
        source_group_name
        and re.fullmatch(
            r"(?:rap|gop)\d+",
            source_group_name,
            flags=re.IGNORECASE,
        )
    ):
        idx = int(re.search(r"\d+$", source_group_name).group())
        return stem, f"gop{idx}", idx

    if source_group_idx is not None:
        idx = int(source_group_idx)
        return stem, f"gop{idx}", idx

    return stem, "gop0", 0


# ============================================================
# NPZ inspection/loading
# ============================================================

def inspect_npz(path: Path, input_mode: str) -> Dict[str, Any]:
    """
    Read only enough information to group and validate the NPZ.

    No depth/E/K array is retained after this function returns.
    """
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)

        dk = first_key(z, DEPTH_KEYS)
        rk = first_key(z, RVEC_KEYS)
        tk = first_key(z, TVEC_KEYS)

        canonical = (
            dk is not None
            and "K_fixed" in keys
            and rk is not None
            and tk is not None
        )
        vggt = all(
            k in keys
            for k in [
                "depth_original",
                "extrinsic",
                "intrinsic_original",
            ]
        )

        mode = input_mode
        if mode == "auto":
            mode = (
                "canonical"
                if canonical
                else "vggt"
                if vggt
                else "unknown"
            )

        if mode == "canonical":
            if not canonical:
                raise KeyError("not a canonical geometry NPZ")
            depth_key = dk
            source_type = "canonical_fixedK_optimized"

        elif mode == "vggt":
            if not vggt:
                raise KeyError("not a VGGT geometry NPZ")
            depth_key = "depth_original"
            source_type = "vggt_original"

        else:
            raise KeyError("unsupported NPZ")

        # This may decompress one depth array temporarily, but it is released
        # immediately and is never retained across files.
        depth = cleanup_depth(z[depth_key], path, depth_key)
        n, h, w = depth.shape

        frame_indices = (
            np.asarray(z["frame_indices"], np.int64).reshape(-1)
            if "frame_indices" in z
            else np.arange(n, dtype=np.int64)
        )
        if len(frame_indices) != n:
            raise ValueError("frame_indices length mismatch")

        rn = scalar_str(z["rap_name"]) if "rap_name" in z else None
        ri = (
            int(np.asarray(z["rap_index"]).item())
            if "rap_index" in z
            else None
        )

        del depth

    seq, gop_name, gop_idx = infer_seq_gop(path, rn, ri)

    return {
        "path": path,
        "sequence": seq,
        "gop_name": gop_name,
        "gop_idx": gop_idx,
        "mode": mode,
        "depth_key": depth_key,
        "rvec_key": rk,
        "tvec_key": tk,
        "source_type": source_type,
        "n": int(n),
        "height": int(h),
        "width": int(w),
        "frame_indices": [int(x) for x in frame_indices],
    }


def load_geometry(
    desc: Dict[str, Any],
    need_depth: bool = True,
    need_camera: bool = True,
) -> Dict[str, Any]:
    """
    Load one GOP only.

    Callers must discard the returned dictionary before loading the next GOP.
    """
    path = desc["path"]
    mode = desc["mode"]

    out: Dict[str, Any] = {
        "path": path,
        "sequence": desc["sequence"],
        "gop_name": desc["gop_name"],
        "gop_idx": desc["gop_idx"],
        "frame_indices": desc["frame_indices"],
        "source_type": desc["source_type"],
    }

    with np.load(path, allow_pickle=True) as z:
        if need_depth:
            out["depth"] = cleanup_depth(
                z[desc["depth_key"]],
                path,
                desc["depth_key"],
            )

        if need_camera:
            n = desc["n"]

            if mode == "canonical":
                out["K"] = repeat_K(z["K_fixed"], n)
                out["E"] = abs_pose_to_extrinsic(
                    z[desc["rvec_key"]],
                    z[desc["tvec_key"]],
                )
                out["fixed_K"] = True

            elif mode == "vggt":
                E = np.asarray(z["extrinsic"])
                K = np.asarray(z["intrinsic_original"])

                if E.ndim == 4 and E.shape[0] == 1:
                    E = E[0]
                if K.ndim == 4 and K.shape[0] == 1:
                    K = K[0]

                E = E.astype(np.float32, copy=False)
                K = K.astype(np.float32, copy=False)

                if E.shape[0] != n or E.shape[1:] not in [
                    (3, 4),
                    (4, 4),
                ]:
                    raise ValueError(f"bad extrinsic: {E.shape}")
                if K.shape != (n, 3, 3):
                    raise ValueError(f"bad intrinsic: {K.shape}")

                out["E"] = E
                out["K"] = K
                out["fixed_K"] = False

            else:
                raise ValueError(mode)

    if need_depth:
        d = out["depth"]
        if d.shape != (
            desc["n"],
            desc["height"],
            desc["width"],
        ):
            raise ValueError(
                f"{path}: depth shape changed unexpectedly: {d.shape}"
            )

    return out


# ============================================================
# Camera/depth metadata
# ============================================================

def Kdict(K, z_sign):
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "z_sign": float(z_sign),
    }


def Kvec(K):
    return np.array(
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        np.float64,
    )


def Kdelta(Ks, force_zero):
    out = np.zeros((len(Ks), 4), np.float32)

    if not force_zero and len(Ks) > 1:
        # Ks is only N x 3 x 3, so this is small.
        v = np.empty((len(Ks), 4), np.float64)
        for i, K in enumerate(Ks):
            v[i] = Kvec(K)
        out[1:] = v[1:] - v[:-1]

    return out


def choose_scale(depth, percentile, precision):
    """
    Preserve the original exact percentile definition.

    This creates a valid-value vector for one GOP only. The original script
    kept all GOP depth arrays plus merged_depth simultaneously; this version
    keeps only this GOP in memory.
    """
    valid = np.isfinite(depth) & (depth > 0)

    if np.any(valid):
        ref = float(np.percentile(depth[valid], percentile))
    else:
        ref = 1023.0

    if not np.isfinite(ref) or ref <= 0:
        ref = 1023.0

    sf = ref / 1023.0
    si = max(1, int(round(sf * precision)))

    return {
        "depth_scale": si,
        "depth_scale_precision": precision,
        "depth_scale_real": si / precision,
        "depth_ref": ref,
        "depth_percentile": percentile,
        "max_code": 1023,
    }


# ============================================================
# Low-memory depth writing/statistics
# ============================================================

def quantize_depth_frame(d, scale_real):
    """
    Quantize one frame with bounded temporary memory.

    Peak frame-side temporary memory:
      one float32 work image + one uint16 output image.
    """
    work = np.array(d, dtype=np.float32, copy=True)

    max_depth = np.float32(1023.0 * scale_real)
    np.nan_to_num(
        work,
        copy=False,
        nan=0.0,
        posinf=float(max_depth),
        neginf=0.0,
    )
    work /= np.float32(scale_real)
    np.rint(work, out=work)
    np.clip(work, 0.0, 1023.0, out=work)

    return work.astype("<u2", copy=False)


def frame_quant_stats(
    d,
    y,
    scale_real,
    row_chunk=256,
):
    """
    Compute MAE/RMSE/clip ratio without materializing full-frame masked arrays.
    """
    h = d.shape[0]

    count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    clip_count = 0

    sr = np.float32(scale_real)

    for y0 in range(0, h, row_chunk):
        y1 = min(h, y0 + row_chunk)

        ds = d[y0:y1]
        ys = y[y0:y1]

        valid = np.isfinite(ds) & (ds > 0)
        n = int(np.count_nonzero(valid))
        if n == 0:
            continue

        # Chunk-sized arrays only.
        err = ys.astype(np.float32) * sr - ds
        ev = err[valid]

        count += n
        abs_sum += float(
            np.sum(np.abs(ev), dtype=np.float64)
        )
        sq_sum += float(
            np.sum(ev * ev, dtype=np.float64)
        )
        clip_count += int(
            np.count_nonzero(ys[valid] >= 1023)
        )

    if count == 0:
        return None

    return {
        "mae": abs_sum / count,
        "rmse": float(np.sqrt(sq_sum / count)),
        "clip_ratio": clip_count / count,
    }


def finalize_quant_summary(per_gop):
    summary = {}

    for gop_idx, stat in sorted(per_gop.items()):
        maes = stat["maes"]
        rmses = stat["rmses"]
        clips = stat["clips"]

        summary[str(gop_idx)] = {
            "frame_count": int(stat["frame_count"]),
            "mean_mae": (
                float(np.mean(maes))
                if maes
                else None
            ),
            "mean_rmse": (
                float(np.mean(rmses))
                if rmses
                else None
            ),
            "max_clip_ratio": (
                max(clips)
                if clips
                else 0.0
            ),
        }

    return summary


def write_depth_streaming(
    path: Path,
    items: List[Dict[str, Any]],
    pocs: List[int],
    depth_owner: Dict[int, int],
    gop_depth_meta: Dict[int, Dict[str, Any]],
    row_chunk: int,
):
    """
    Write unique POCs in ascending order while loading at most one GOP depth
    array at a time.
    """
    ensure_parent(path)

    h = items[0]["height"]
    w = items[0]["width"]

    if h % 2 or w % 2:
        raise ValueError("YUV420 requires even width/height")

    item_by_gop = {it["gop_idx"]: it for it in items}
    local_index_by_gop = {
        it["gop_idx"]: {
            poc: li
            for li, poc in enumerate(it["frame_indices"])
        }
        for it in items
    }

    uv = np.full((h // 2, w // 2), 512, dtype="<u2")
    uv_bytes = uv.tobytes()
    del uv

    per_gop = {}

    current_gop: Optional[int] = None
    current_depth: Optional[np.ndarray] = None

    with open(path, "wb") as f:
        for poc in pocs:
            owner = depth_owner[poc]

            if owner != current_gop:
                current_depth = None
                gc.collect()

                geom = load_geometry(
                    item_by_gop[owner],
                    need_depth=True,
                    need_camera=False,
                )
                current_depth = geom["depth"]
                current_gop = owner
                del geom

            assert current_depth is not None

            li = local_index_by_gop[owner][poc]
            d = current_depth[li]

            meta = gop_depth_meta[owner]
            sr = float(meta["depth_scale_real"])
            if sr <= 0:
                raise ValueError(
                    f"invalid depth scale for GOP {owner}: {sr}"
                )

            y = quantize_depth_frame(d, sr)

            f.write(y.tobytes())
            f.write(uv_bytes)
            f.write(uv_bytes)

            stat = per_gop.setdefault(
                owner,
                {
                    "maes": [],
                    "rmses": [],
                    "clips": [],
                    "frame_count": 0,
                },
            )
            stat["frame_count"] += 1

            q = frame_quant_stats(
                d,
                y,
                sr,
                row_chunk=row_chunk,
            )
            if q is not None:
                stat["maes"].append(float(q["mae"]))
                stat["rmses"].append(float(q["rmse"]))
                stat["clips"].append(
                    float(q["clip_ratio"])
                )

            del y

    current_depth = None
    gc.collect()

    return finalize_quant_summary(per_gop)


# ============================================================
# Sequence processing
# ============================================================

def build_sequence_index(items):
    """
    Determine unique depth POCs, owners, overlap counts, and output indices
    using descriptor metadata only.
    """
    depth_owner: Dict[int, int] = {}
    camera_count_by_poc: Dict[int, int] = {}

    for it in items:
        for poc in it["frame_indices"]:
            camera_count_by_poc[poc] = (
                camera_count_by_poc.get(poc, 0) + 1
            )
            if poc not in depth_owner:
                depth_owner[poc] = it["gop_idx"]

    pocs = sorted(depth_owner)

    if not pocs:
        raise ValueError("no frames")

    if pocs != list(range(pocs[0], pocs[-1] + 1)):
        raise ValueError("merged depth POCs are not contiguous")

    poc_to_depth_idx = {
        poc: i
        for i, poc in enumerate(pocs)
    }

    overlaps = sorted(
        poc
        for poc, count in camera_count_by_poc.items()
        if count > 1
    )

    return (
        pocs,
        depth_owner,
        camera_count_by_poc,
        poc_to_depth_idx,
        overlaps,
    )


def compute_gop_metadata(items, args):
    """
    Compute exact depth scale and initial intrinsic for each GOP while keeping
    only one GOP loaded.
    """
    gop_depth_meta = {}
    gop_header_info = {}

    for it in items:
        geom = load_geometry(
            it,
            need_depth=True,
            need_camera=True,
        )

        depth = geom["depth"]
        K = geom["K"]

        dm = choose_scale(
            depth,
            args.depth_percentile,
            args.depth_scale_precision,
        )

        gop_depth_meta[it["gop_idx"]] = dm
        gop_header_info[it["gop_idx"]] = {
            "fixed_K": bool(geom["fixed_K"]),
            "initial_K": np.array(
                K[0],
                dtype=np.float32,
                copy=True,
            ),
        }

        del geom, depth, K
        gc.collect()

    return gop_depth_meta, gop_header_info


def write_camera_jsonl(
    out_json,
    header,
    items,
    args,
    poc_to_depth_idx,
    depth_owner,
    camera_count_by_poc,
    gop_depth_meta,
):
    ensure_parent(out_json)

    record_idx = 0

    with open(out_json, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(header, ensure_ascii=False) + "\n"
        )

        for it in items:
            geom = load_geometry(
                it,
                need_depth=False,
                need_camera=True,
            )

            K = geom["K"]
            rvec, tvec = convert_pose(
                geom["E"],
                args.pose_mode,
            )
            kd = Kdelta(K, geom["fixed_K"])
            dm = gop_depth_meta[it["gop_idx"]]

            for li, poc in enumerate(it["frame_indices"]):
                rec = {
                    "type": "frame",
                    "camera_record_idx": record_idx,
                    "gop_idx": it["gop_idx"],
                    "gop_name": it["gop_name"],
                    "local_poc": li,
                    "poc": poc,
                    "frame_idx": poc,
                    "depth_frame_idx": poc_to_depth_idx[poc],
                    "depth_source_gop_idx": depth_owner[poc],
                    "is_overlap": (
                        camera_count_by_poc[poc] > 1
                    ),
                    "is_depth_owner": (
                        depth_owner[poc] == it["gop_idx"]
                    ),
                    "rvec": [
                        float(x)
                        for x in rvec[li]
                    ],
                    "tvec": [
                        float(x)
                        for x in tvec[li]
                    ],
                    "intrinsic": Kdict(
                        K[li],
                        args.z_sign,
                    ),
                    "intrinsic_delta": [
                        float(x)
                        for x in kd[li]
                    ],
                    "depth_scale": int(
                        dm["depth_scale"]
                    ),
                    "depth_scale_precision": int(
                        dm["depth_scale_precision"]
                    ),
                    "depth_scale_real": float(
                        dm["depth_scale_real"]
                    ),
                    "depth_ref": float(
                        dm["depth_ref"]
                    ),
                    "depth_percentile": float(
                        dm["depth_percentile"]
                    ),
                }

                f.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                record_idx += 1

            del geom, K, rvec, tvec, kd
            gc.collect()

    return record_idx


def process_sequence(seq, items, args):
    items = sorted(items, key=lambda x: x["gop_idx"])

    h = items[0]["height"]
    w = items[0]["width"]

    for it in items:
        if (it["height"], it["width"]) != (h, w):
            raise ValueError(
                f"resolution mismatch: {it['path']}"
            )

    (
        pocs,
        depth_owner,
        camera_count_by_poc,
        poc_to_depth_idx,
        overlaps,
    ) = build_sequence_index(items)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else items[0]["path"].parent
    )
    tag = (
        f"_{args.output_tag.strip('_')}"
        if args.output_tag.strip("_")
        else ""
    )
    base = safe_name(seq)

    out_yuv = (
        out_dir
        / f"{base}_depth_merged{tag}.yuv"
    )
    out_json = (
        out_dir
        / f"{base}_camParam_merged{tag}.jsonl"
    )

    if (
        not args.overwrite
        and (out_yuv.exists() or out_json.exists())
    ):
        print(f"[SKIP] exists: {seq}")
        return False

    # Pass 1: one GOP at a time, exact scale and initial K.
    gop_depth_meta, gop_header_info = compute_gop_metadata(
        items,
        args,
    )

    # Pass 2: write unique depth frames directly to YUV.
    qstats_by_gop = write_depth_streaming(
        out_yuv,
        items,
        pocs,
        depth_owner,
        gop_depth_meta,
        row_chunk=args.stats_row_chunk,
    )

    header = {
        "type": "header",
        "format": "camparam_v4_multi_gop_merged",
        "sequence_name": seq,
        "gop_count": len(items),
        "camera_record_count": sum(
            len(x["frame_indices"])
            for x in items
        ),
        "unique_depth_frame_count": len(pocs),
        "depth_frame_pocs": pocs,
        "overlap_pocs": overlaps,
        "overlap_policy": {
            "camera": (
                "all GOP-local records kept; "
                "overlap POC has two lines"
            ),
            "depth": (
                "one frame only; earlier GOP wins"
            ),
        },
        "width": w,
        "height": h,
        "bit_depth": 10,
        "depth_yuv": out_yuv.name,
        "depth_scale_mode": "per_gop",
        "depth_scale_signal": (
            "written in every frame record "
            "using that GOP scale"
        ),
        "depth_scale_precision": int(
            args.depth_scale_precision
        ),
        "depth_quant_summary_by_gop": qstats_by_gop,
        "pose_mode": args.pose_mode,
        "pose_convention": {
            "current_to_previous": (
                "X_prev=R*X_cur+t; each GOP "
                "local_poc 0 is identity"
            ),
            "gop_local": (
                "X_i=R*X_0+t; each GOP "
                "local_poc 0 is identity"
            ),
            "absolute": (
                "camera_from_world in each "
                "NPZ coordinate system"
            ),
        }[args.pose_mode],
        "gops": [
            {
                "gop_idx": x["gop_idx"],
                "gop_name": x["gop_name"],
                "source_npz": str(
                    x["path"].resolve()
                ),
                "frame_indices": x["frame_indices"],
                "fixed_intrinsic": (
                    gop_header_info[x["gop_idx"]][
                        "fixed_K"
                    ]
                ),
                "initial_intrinsic": Kdict(
                    gop_header_info[x["gop_idx"]][
                        "initial_K"
                    ],
                    args.z_sign,
                ),
                "depth_scale": int(
                    gop_depth_meta[x["gop_idx"]][
                        "depth_scale"
                    ]
                ),
                "depth_scale_precision": int(
                    gop_depth_meta[x["gop_idx"]][
                        "depth_scale_precision"
                    ]
                ),
                "depth_scale_real": float(
                    gop_depth_meta[x["gop_idx"]][
                        "depth_scale_real"
                    ]
                ),
                "depth_ref": float(
                    gop_depth_meta[x["gop_idx"]][
                        "depth_ref"
                    ]
                ),
                "depth_percentile": float(
                    gop_depth_meta[x["gop_idx"]][
                        "depth_percentile"
                    ]
                ),
            }
            for x in items
        ],
    }

    # Pass 3: camera JSONL, one GOP at a time.
    record_idx = write_camera_jsonl(
        out_json,
        header,
        items,
        args,
        poc_to_depth_idx,
        depth_owner,
        camera_count_by_poc,
        gop_depth_meta,
    )

    print(f"[OK] {seq}")
    print(
        f"     GOPs           : "
        f"{[x['gop_name'] for x in items]}"
    )
    print(f"     overlap POCs   : {overlaps}")
    print(f"     camera records : {record_idx}")
    print(f"     depth frames   : {len(pocs)}")
    print("     depth scales   :")

    for it in items:
        dm = gop_depth_meta[it["gop_idx"]]
        print(
            f"       {it['gop_name']}: "
            f"int={dm['depth_scale']}, "
            f"real={dm['depth_scale_real']:.9g}"
        )

    print(f"     camera JSONL   : {out_json}")
    print(f"     depth YUV      : {out_yuv}")

    return True


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument(
        "--input-mode",
        choices=["auto", "vggt", "canonical"],
        default="auto",
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--output-tag", default="")
    ap.add_argument(
        "--depth-percentile",
        type=float,
        default=99.9,
    )
    ap.add_argument(
        "--depth-scale-precision",
        type=int,
        default=100000,
    )
    ap.add_argument(
        "--stats-row-chunk",
        type=int,
        default=256,
        help=(
            "Rows per chunk for quantization statistics. "
            "Smaller values reduce temporary RAM."
        ),
    )
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument(
        "--pose-mode",
        choices=[
            "current_to_previous",
            "gop_local",
            "absolute",
        ],
        default="current_to_previous",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--no-skip-unrecognized",
        action="store_true",
    )

    args = ap.parse_args()

    if args.depth_scale_precision <= 0:
        raise ValueError(
            "--depth-scale-precision must be positive"
        )
    if args.stats_row_chunk <= 0:
        raise ValueError(
            "--stats-row-chunk must be positive"
        )
    if not (0.0 <= args.depth_percentile <= 100.0):
        raise ValueError(
            "--depth-percentile must be in [0, 100]"
        )

    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f"not a directory: {root}")

    # Store descriptors only. Large arrays are never retained here.
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for p in sorted(root.rglob(args.pattern)):
        try:
            desc = inspect_npz(p, args.input_mode)
            groups.setdefault(
                desc["sequence"],
                [],
            ).append(desc)

        except Exception as e:
            if args.no_skip_unrecognized:
                raise
            print(
                f"[SKIP] {p}: "
                f"{type(e).__name__}: {e}"
            )

        # Release temporary arrays from inspection promptly.
        gc.collect()

    ok = 0
    fail = 0
    skip = 0

    for seq, items in sorted(groups.items()):
        try:
            if process_sequence(seq, items, args):
                ok += 1
            else:
                skip += 1

        except Exception as e:
            fail += 1
            print(
                f"[FAIL] {seq}: "
                f"{type(e).__name__}: {e}"
            )

        gc.collect()

    print(
        f"Done. converted={ok}, "
        f"skipped={skip}, failed={fail}"
    )


if __name__ == "__main__":
    main()
