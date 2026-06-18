#!/usr/bin/env python3
# eval_recon_depth_backward_warp_psnr.py
#
# Purpose:
#   Evaluate projection-prediction quality using an externally reconstructed
#   depth map, e.g. VTM-reconstructed depth.
#
#   For each current frame t:
#     1. Read reconstructed depth Y(t), yuv420p10le.
#     2. Convert stored depth sample to linear camera depth:
#          linear_depth = depth_y * nearClipPlane(current_camera)
#     3. Back-project each current pixel using current camera and recon depth.
#     4. Project the 3D point into previous/reference camera.
#     5. Bilinear-sample previous/reference GT video frame.
#     6. Compare the backward-warped predictor with current GT video frame.
#     7. Write predicted YUV and per-frame CSV/JSON metrics.
#
# Notes:
#   - Camera parser supports JSON, JSON array, JSON object with frames, and JSONL.
#   - Matrix dict format e00~e33 is transposed, matching the previously verified
#     forward-warp convention.
#   - Default input/output format is yuv420p10le.
#
# Example:
#   python eval_recon_depth_backward_warp_psnr.py ^
#     --recon-depth recon_depth.yuv ^
#     --gt-video texture_1920x1080_10bit.yuv ^
#     --camera-param camera.txt ^
#     --width 1920 ^
#     --height 1080 ^
#     --start-frame 0 ^
#     --num-frames 16 ^
#     --ref-offset 1 ^
#     --out-pred-yuv pred_from_recon_depth.yuv ^
#     --out-csv projection_psnr.csv ^
#     --out-json projection_psnr_summary.json

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# Camera JSON / JSONL / matrix utilities
# ============================================================

def load_camera_json(path: str):
    """
    Supports:
      1. JSON object
      2. JSON array
      3. JSONL, one camera frame per line

    This follows the convention from the previously verified forward-warp code.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_err:
        entries = []

        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as jsonl_err:
                raise RuntimeError(
                    f"Failed to parse camera parameter file as JSON or JSONL: {path}\n"
                    f"JSON error={json_err}\n"
                    f"JSONL error at line {line_no}: {jsonl_err}"
                ) from jsonl_err

            entries.append(obj)

        if not entries:
            raise RuntimeError(f"Camera parameter file is empty or invalid: {path}")

        return entries


def has_camera_matrices(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    keys = set(obj.keys())

    inv_keys = {
        "InvProjectionMatrix",
        "invProjectionMatrix",
        "InverseProjectionMatrix",
    }
    proj_keys = {
        "ProjectionMatrix",
        "projectionMatrix",
    }
    w2c_keys = {
        "WorldToCameraMatrix",
        "worldToCameraMatrix",
        "ViewMatrix",
    }
    c2w_keys = {
        "CameraToWorldMatrix",
        "cameraToWorldMatrix",
        "CameraToWorldMarix",
        "cameraToWorldMarix",
        "InvViewMatrix",
    }

    return (
        bool(keys & inv_keys)
        and bool(keys & proj_keys)
        and bool(keys & w2c_keys)
        and bool(keys & c2w_keys)
    )


def camera_frame_poc(entry: Dict[str, Any], fallback_idx: int) -> int:
    for key in ["frames", "frame", "Frame", "frameIdx", "frame_idx", "poc", "POC"]:
        if isinstance(entry, dict) and key in entry:
            try:
                return int(entry[key])
            except Exception:
                pass

    return int(fallback_idx)


def extract_camera_entries(cams_json):
    entries = []

    if isinstance(cams_json, list):
        entries = cams_json

    elif isinstance(cams_json, dict):
        for key in [
            "frames",
            "Frames",
            "cameras",
            "Cameras",
            "cameraFrames",
            "CameraFrames",
            "camera_params",
            "cameraParams",
        ]:
            if key in cams_json and isinstance(cams_json[key], list):
                entries = cams_json[key]
                break

        if not entries and has_camera_matrices(cams_json):
            entries = [cams_json]

        if not entries:
            numeric_items = []
            for k, v in cams_json.items():
                if isinstance(v, dict) and has_camera_matrices(v):
                    try:
                        numeric_items.append((int(k), v))
                    except Exception:
                        pass

            if numeric_items:
                numeric_items.sort(key=lambda x: x[0])
                entries = [v for _, v in numeric_items]

    if not entries:
        raise RuntimeError("Cannot parse camera parameter structure")

    return entries


def build_camera_lookup(cams_json):
    entries = extract_camera_entries(cams_json)

    lookup = {}
    pocs = []

    for i, entry in enumerate(entries):
        poc = camera_frame_poc(entry, i)

        lookup[poc] = entry
        pocs.append(poc)

        # Fallback by list index.
        if i not in lookup:
            lookup[i] = entry

    return lookup, sorted(set(pocs))


def get_camera(camera_lookup, frame_idx: int):
    if frame_idx not in camera_lookup:
        raise KeyError(f"Camera for frame {frame_idx} not found")
    return camera_lookup[frame_idx]


def get_value_by_alias(entry: Dict[str, Any], aliases: List[str], required: bool = True):
    for key in aliases:
        if isinstance(entry, dict) and key in entry:
            return entry[key]

    if required:
        raise KeyError(f"Missing key. aliases={aliases}")

    return None


def get_near_clip(entry: Dict[str, Any]) -> float:
    val = get_value_by_alias(
        entry,
        ["nearClipPlane", "NearClipPlane", "near_clip_plane", "near", "Near"],
        required=False,
    )

    if val is None:
        return 1.0

    return float(val)


def get_matrix(frame_entry: Dict[str, Any], logical_name_or_aliases):
    """
    Important:
      If matrix is stored as dict e00~e33, return mat.T.

    This matches the previously verified forward-warp code:
      # 중요: dict e00~e33 matrix는 transpose해서 사용
      return mat.T

    Accepts either:
      get_matrix(cam, "ProjectionMatrix")
      get_matrix(cam, ["ProjectionMatrix", "projectionMatrix"])
    """
    alias_map = {
        "InvProjectionMatrix": [
            "InvProjectionMatrix",
            "invProjectionMatrix",
            "InverseProjectionMatrix",
        ],
        "ProjectionMatrix": [
            "ProjectionMatrix",
            "projectionMatrix",
        ],
        "WorldToCameraMatrix": [
            "WorldToCameraMatrix",
            "worldToCameraMatrix",
            "ViewMatrix",
        ],
        "CameraToWorldMatrix": [
            "CameraToWorldMatrix",
            "cameraToWorldMatrix",
            "CameraToWorldMarix",
            "cameraToWorldMarix",
            "InvViewMatrix",
        ],
    }

    if isinstance(logical_name_or_aliases, str):
        aliases = alias_map.get(logical_name_or_aliases, [logical_name_or_aliases])
    else:
        aliases = list(logical_name_or_aliases)

    obj = get_value_by_alias(frame_entry, aliases, required=True)

    if isinstance(obj, dict):
        mat = np.zeros((4, 4), dtype=np.float64)

        for r in range(4):
            for c in range(4):
                key = f"e{r}{c}"
                if key not in obj:
                    raise KeyError(f"Missing matrix key {key} for aliases={aliases}")
                mat[r, c] = float(obj[key])

        return mat.T

    mat = np.array(obj, dtype=np.float64)

    if mat.shape == (16,):
        mat = mat.reshape(4, 4)

    if mat.shape != (4, 4):
        raise ValueError(f"matrix shape is {mat.shape}, expected 4x4. aliases={aliases}")

    return mat


def camera_has_required_mats(cam: Dict[str, Any]) -> bool:
    try:
        get_matrix(cam, "InvProjectionMatrix")
        get_matrix(cam, "ProjectionMatrix")
        get_matrix(cam, "WorldToCameraMatrix")
        get_matrix(cam, "CameraToWorldMatrix")
        return True
    except Exception:
        return False


# ============================================================
# YUV420 IO
# ============================================================

def get_yuv420_frame_size_bytes(w: int, h: int, bit_depth: int) -> int:
    y_size = w * h
    uv_size = (w // 2) * (h // 2)
    samples = y_size + 2 * uv_size
    bytes_per_sample = 1 if bit_depth <= 8 else 2
    return samples * bytes_per_sample


def count_yuv420_frames(path: str, w: int, h: int, bit_depth: int) -> int:
    frame_size = get_yuv420_frame_size_bytes(w, h, bit_depth)
    file_size = os.path.getsize(path)

    frame_count = file_size // frame_size
    trailing = file_size % frame_size

    if trailing != 0:
        print(f"[WARN] trailing bytes ignored: {path}, trailing={trailing}")

    return frame_count


def yuv_dtype(bit_depth: int):
    if bit_depth <= 8:
        return np.uint8
    return np.dtype("<u2")


def read_yuv420_frame(fp, idx: int, w: int, h: int, bit_depth: int):
    frame_size = get_yuv420_frame_size_bytes(w, h, bit_depth)
    bytes_per_sample = 1 if bit_depth <= 8 else 2

    fp.seek(idx * frame_size)

    y_count = w * h
    uv_w = w // 2
    uv_h = h // 2
    uv_count = uv_w * uv_h

    dt = yuv_dtype(bit_depth)

    y_raw = fp.read(y_count * bytes_per_sample)
    u_raw = fp.read(uv_count * bytes_per_sample)
    v_raw = fp.read(uv_count * bytes_per_sample)

    if len(y_raw) != y_count * bytes_per_sample:
        raise EOFError(f"Failed to read Y frame idx={idx}")
    if len(u_raw) != uv_count * bytes_per_sample or len(v_raw) != uv_count * bytes_per_sample:
        raise EOFError(f"Failed to read UV frame idx={idx}")

    y = np.frombuffer(y_raw, dtype=dt).reshape(h, w).astype(np.float64)
    u = np.frombuffer(u_raw, dtype=dt).reshape(uv_h, uv_w).astype(np.float64)
    v = np.frombuffer(v_raw, dtype=dt).reshape(uv_h, uv_w).astype(np.float64)

    return y, u, v


def read_yuv420_y_frame(fp, idx: int, w: int, h: int, bit_depth: int):
    frame_size = get_yuv420_frame_size_bytes(w, h, bit_depth)
    bytes_per_sample = 1 if bit_depth <= 8 else 2

    fp.seek(idx * frame_size)

    y_count = w * h
    dt = yuv_dtype(bit_depth)

    y_raw = fp.read(y_count * bytes_per_sample)
    if len(y_raw) != y_count * bytes_per_sample:
        raise EOFError(f"Failed to read depth Y frame idx={idx}")

    return np.frombuffer(y_raw, dtype=dt).reshape(h, w).astype(np.float64)


def write_yuv420_frame(fp, y, u, v, bit_depth: int):
    maxv = (1 << bit_depth) - 1

    if bit_depth <= 8:
        dt = np.uint8
    else:
        dt = np.dtype("<u2")

    y_out = np.clip(np.rint(y), 0, maxv).astype(dt)
    u_out = np.clip(np.rint(u), 0, maxv).astype(dt)
    v_out = np.clip(np.rint(v), 0, maxv).astype(dt)

    fp.write(y_out.tobytes())
    fp.write(u_out.tobytes())
    fp.write(v_out.tobytes())


# ============================================================
# Geometry
# ============================================================

def pixel_rays_camera(u, v, width: int, height: int, inv_proj):
    x_ndc = ((u + 0.5) / float(width)) * 2.0 - 1.0
    y_ndc = 1.0 - ((v + 0.5) / float(height)) * 2.0

    ones = np.ones_like(x_ndc, dtype=np.float64)
    p_ndc = np.stack([x_ndc, y_ndc, ones, ones], axis=-1)

    p_view_h = p_ndc @ inv_proj.T
    w = p_view_h[..., 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)

    p_view = p_view_h[..., :3] / w
    z_abs = np.maximum(np.abs(p_view[..., 2:3]), 1e-12)

    # Same convention as previously verified:
    # p_view_scaled = p_view / abs(p_view.z) * linear_depth
    return p_view / z_abs


def make_backward_map_cur_to_ref(depth_y_cur, cam_cur, cam_ref, width: int, height: int):
    inv_proj_cur = get_matrix(cam_cur, "InvProjectionMatrix")
    c2w_cur = get_matrix(cam_cur, "CameraToWorldMatrix")
    w2c_ref = get_matrix(cam_ref, "WorldToCameraMatrix")
    proj_ref = get_matrix(cam_ref, "ProjectionMatrix")

    near_cur = get_near_clip(cam_cur)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )

    rays = pixel_rays_camera(xx, yy, width, height, inv_proj_cur)

    # Stored depth sample Y -> actual linear depth.
    linear_z = depth_y_cur.astype(np.float64) * near_cur

    p_cur = rays * linear_z[..., None]
    ones = np.ones((height, width, 1), dtype=np.float64)
    p_cur_h = np.concatenate([p_cur, ones], axis=-1)

    p_world = p_cur_h @ c2w_cur.T
    p_ref = p_world @ w2c_ref.T
    clip = p_ref @ proj_ref.T

    cw = clip[..., 3]
    valid = np.abs(cw) > 1e-12

    ndc_x = np.zeros_like(cw)
    ndc_y = np.zeros_like(cw)

    ndc_x[valid] = clip[..., 0][valid] / cw[valid]
    ndc_y[valid] = clip[..., 1][valid] / cw[valid]

    map_x = (ndc_x + 1.0) * 0.5 * width - 0.5
    map_y = (1.0 - ndc_y) * 0.5 * height - 0.5

    valid = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(linear_z)
        & (linear_z > 0)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y <= height - 1.0)
    )

    return map_x, map_y, valid


def bilinear_sample(img, map_x, map_y, valid, fill):
    h, w = img.shape

    safe_x = np.where(np.isfinite(map_x), map_x, 0.0)
    safe_y = np.where(np.isfinite(map_y), map_y, 0.0)

    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid2 = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < w)
        & (y1 < h)
    )

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    dx = safe_x - x0
    dy = safe_y - y0

    v00 = img[y0c, x0c]
    v01 = img[y0c, x1c]
    v10 = img[y1c, x0c]
    v11 = img[y1c, x1c]

    out = (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v01
        + (1.0 - dx) * dy * v10
        + dx * dy * v11
    )

    out = np.where(valid2, out, fill)

    return out.astype(np.float64), valid2


def downsample_map_for_chroma(map_x, map_y, valid):
    h, w = map_x.shape
    hc = h // 2
    wc = w // 2

    mx = map_x[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    my = map_y[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) / 2.0
    mv = valid[: hc * 2, : wc * 2].reshape(hc, 2, wc, 2).mean(axis=(1, 3)) >= 0.5

    return mx, my, mv


# ============================================================
# Metrics
# ============================================================

def compute_metrics(orig, pred, maxv: int, mask=None, prefix=""):
    diff = orig.astype(np.float64) - pred.astype(np.float64)

    if mask is not None:
        mask = mask.astype(bool)
        if not np.any(mask):
            return {
                f"{prefix}mae": float("nan"),
                f"{prefix}mse": float("nan"),
                f"{prefix}rmse": float("nan"),
                f"{prefix}psnr": float("nan"),
                f"{prefix}max_error": float("nan"),
            }
        diff = diff[mask]

    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((maxv * maxv) / mse)
    max_error = float(np.max(np.abs(diff)))

    return {
        f"{prefix}mae": mae,
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}psnr": psnr,
        f"{prefix}max_error": max_error,
    }


def weighted_yuv420_psnr(y_mse, u_mse, v_mse, maxv):
    if not all(math.isfinite(x) for x in [y_mse, u_mse, v_mse]):
        return float("nan")

    mse = (4.0 * y_mse + u_mse + v_mse) / 6.0
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((maxv * maxv) / mse)


# ============================================================
# Backward warp evaluation
# ============================================================

def make_fill_plane(shape, mode: str, maxv: int, ref_plane=None):
    if mode == "zero":
        return np.zeros(shape, dtype=np.float64)
    if mode == "neutral":
        return np.full(shape, maxv // 2, dtype=np.float64)
    if mode == "ref_same":
        if ref_plane is None:
            raise ValueError("ref_same fill requires ref_plane")
        return ref_plane
    raise ValueError(f"unknown fill mode: {mode}")


def backward_warp_ref_to_cur(
    ref_yuv,
    cur_yuv,
    recon_depth_y_cur,
    cam_ref,
    cam_cur,
    args,
):
    ref_y, ref_u, ref_v = ref_yuv
    cur_y, cur_u, cur_v = cur_yuv

    h, w = cur_y.shape

    map_x, map_y, valid_y = make_backward_map_cur_to_ref(
        recon_depth_y_cur,
        cam_cur,
        cam_ref,
        w,
        h,
    )

    maxv = (1 << args.video_bit_depth) - 1

    fill_y = make_fill_plane(cur_y.shape, args.invalid_fill, maxv, ref_plane=ref_y)
    pred_y, valid_y = bilinear_sample(ref_y, map_x, map_y, valid_y, fill_y)

    mx_c, my_c, valid_c = downsample_map_for_chroma(map_x, map_y, valid_y)

    fill_u = make_fill_plane(cur_u.shape, args.invalid_fill, maxv, ref_plane=ref_u)
    fill_v = make_fill_plane(cur_v.shape, args.invalid_fill, maxv, ref_plane=ref_v)

    pred_u, valid_u = bilinear_sample(ref_u, mx_c, my_c, valid_c, fill_u)
    pred_v, valid_v = bilinear_sample(ref_v, mx_c, my_c, valid_c, fill_v)

    # Round to integer sample domain before metrics and writing.
    pred_y = np.clip(np.rint(pred_y), 0, maxv)
    pred_u = np.clip(np.rint(pred_u), 0, maxv)
    pred_v = np.clip(np.rint(pred_v), 0, maxv)

    y_all = compute_metrics(cur_y, pred_y, maxv, prefix="y_")
    u_all = compute_metrics(cur_u, pred_u, maxv, prefix="u_")
    v_all = compute_metrics(cur_v, pred_v, maxv, prefix="v_")

    y_valid = compute_metrics(cur_y, pred_y, maxv, mask=valid_y, prefix="y_valid_")
    u_valid = compute_metrics(cur_u, pred_u, maxv, mask=valid_u, prefix="u_valid_")
    v_valid = compute_metrics(cur_v, pred_v, maxv, mask=valid_v, prefix="v_valid_")

    yuv_psnr = weighted_yuv420_psnr(
        y_all["y_mse"],
        u_all["u_mse"],
        v_all["v_mse"],
        maxv,
    )
    yuv_valid_psnr = weighted_yuv420_psnr(
        y_valid["y_valid_mse"],
        u_valid["u_valid_mse"],
        v_valid["v_valid_mse"],
        maxv,
    )

    stats = {
        "valid_y_ratio": float(np.mean(valid_y)),
        "valid_u_ratio": float(np.mean(valid_u)),
        "valid_v_ratio": float(np.mean(valid_v)),
        "valid_uv_ratio": float(np.mean(valid_u & valid_v)),
        "yuv420_psnr": yuv_psnr,
        "yuv420_valid_psnr": yuv_valid_psnr,
        **y_all,
        **u_all,
        **v_all,
        **y_valid,
        **u_valid,
        **v_valid,
    }

    return (pred_y, pred_u, pred_v), stats


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate backward-warp projection PSNR using externally reconstructed "
            "depth map, e.g. VTM-reconstructed depth."
        )
    )

    p.add_argument("--recon-depth", required=True, help="Externally reconstructed depth YUV420 sequence")
    p.add_argument("--gt-video", required=True, help="GT video YUV420 sequence")
    p.add_argument("--camera-param", required=True, help="Camera parameter JSON/JSONL/TXT")

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)

    p.add_argument("--depth-bit-depth", type=int, default=10)
    p.add_argument("--video-bit-depth", type=int, default=10)

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)

    p.add_argument(
        "--ref-offset",
        type=int,
        default=1,
        help="Use frame t-ref_offset as reference. Default: previous frame.",
    )

    p.add_argument(
        "--invalid-fill",
        choices=["ref_same", "zero", "neutral"],
        default="ref_same",
        help="Fill strategy for out-of-view pixels.",
    )

    p.add_argument(
        "--skip-no-ref",
        action="store_true",
        help="Skip frames whose reference index is negative. If not set, copy current frame for output and exclude from averages.",
    )

    p.add_argument("--out-pred-yuv", default="backward_pred_from_recon_depth.yuv")
    p.add_argument("--out-csv", default="backward_projection_psnr.csv")
    p.add_argument("--out-json", default="backward_projection_psnr_summary.json")

    return p.parse_args()


def validate_args(args):
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")

    if args.width % 2 or args.height % 2:
        raise ValueError("YUV420 requires even width and height")

    if args.depth_bit_depth not in [8, 10, 12, 16]:
        raise ValueError("--depth-bit-depth should be one of 8, 10, 12, 16")

    if args.video_bit_depth not in [8, 10, 12, 16]:
        raise ValueError("--video-bit-depth should be one of 8, 10, 12, 16")

    if args.ref_offset <= 0:
        raise ValueError("--ref-offset must be positive")

    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")


def finite_mean(values):
    vals = []
    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                vals.append(fv)
        except Exception:
            pass

    if not vals:
        return float("nan")

    return float(np.mean(vals))


def main():
    args = parse_args()
    validate_args(args)

    depth_total = count_yuv420_frames(
        args.recon_depth,
        args.width,
        args.height,
        args.depth_bit_depth,
    )
    video_total = count_yuv420_frames(
        args.gt_video,
        args.width,
        args.height,
        args.video_bit_depth,
    )

    total = min(depth_total, video_total)

    if total <= 0:
        raise RuntimeError("No complete frames found")

    if args.start_frame >= total:
        raise ValueError(f"bad --start-frame={args.start_frame}, total={total}")

    end = total if args.num_frames == 0 else min(total, args.start_frame + args.num_frames)

    cams_json = load_camera_json(args.camera_param)
    camera_lookup, camera_pocs = build_camera_lookup(cams_json)

    if not camera_pocs:
        raise RuntimeError("No camera POCs found")

    maxv_video = (1 << args.video_bit_depth) - 1

    rows = []

    pred_fp = open(args.out_pred_yuv, "wb") if args.out_pred_yuv else None

    try:
        with open(args.recon_depth, "rb") as depth_fp, open(args.gt_video, "rb") as video_fp:
            for fi in range(args.start_frame, end):
                ref_idx = fi - args.ref_offset

                if ref_idx < 0:
                    if args.skip_no_ref:
                        print(f"Frame {fi:4d} | skip: ref_idx={ref_idx}")
                        continue

                    cur_yuv = read_yuv420_frame(
                        video_fp,
                        fi,
                        args.width,
                        args.height,
                        args.video_bit_depth,
                    )

                    if pred_fp:
                        write_yuv420_frame(pred_fp, cur_yuv[0], cur_yuv[1], cur_yuv[2], args.video_bit_depth)

                    row = {
                        "frame": fi,
                        "ref_frame": ref_idx,
                        "used_for_average": 0,
                        "note": "no reference; copied current frame",
                        "valid_y_ratio": 0.0,
                        "valid_uv_ratio": 0.0,
                        "y_psnr": float("inf"),
                        "u_psnr": float("inf"),
                        "v_psnr": float("inf"),
                        "yuv420_psnr": float("inf"),
                    }
                    rows.append(row)
                    print(f"Frame {fi:4d} | no ref, copied current frame")
                    continue

                cam_cur = get_camera(camera_lookup, fi)
                cam_ref = get_camera(camera_lookup, ref_idx)

                if not camera_has_required_mats(cam_cur):
                    raise RuntimeError(f"camera frame {fi} does not have required matrices")
                if not camera_has_required_mats(cam_ref):
                    raise RuntimeError(f"camera frame {ref_idx} does not have required matrices")

                recon_depth_y = read_yuv420_y_frame(
                    depth_fp,
                    fi,
                    args.width,
                    args.height,
                    args.depth_bit_depth,
                )
                cur_yuv = read_yuv420_frame(
                    video_fp,
                    fi,
                    args.width,
                    args.height,
                    args.video_bit_depth,
                )
                ref_yuv = read_yuv420_frame(
                    video_fp,
                    ref_idx,
                    args.width,
                    args.height,
                    args.video_bit_depth,
                )

                pred_yuv, stats = backward_warp_ref_to_cur(
                    ref_yuv=ref_yuv,
                    cur_yuv=cur_yuv,
                    recon_depth_y_cur=recon_depth_y,
                    cam_ref=cam_ref,
                    cam_cur=cam_cur,
                    args=args,
                )

                if pred_fp:
                    write_yuv420_frame(pred_fp, pred_yuv[0], pred_yuv[1], pred_yuv[2], args.video_bit_depth)

                row = {
                    "frame": fi,
                    "ref_frame": ref_idx,
                    "used_for_average": 1,
                    "near_cur": get_near_clip(cam_cur),
                    "near_ref": get_near_clip(cam_ref),
                    **stats,
                }
                rows.append(row)

                print(
                    f"Frame {fi:4d} <- {ref_idx:4d} | "
                    f"Y-PSNR={stats['y_psnr']:.3f} | "
                    f"Y-valid-PSNR={stats['y_valid_psnr']:.3f} | "
                    f"YUV-PSNR={stats['yuv420_psnr']:.3f} | "
                    f"validY={stats['valid_y_ratio']:.3f}"
                )

    finally:
        if pred_fp:
            pred_fp.close()

    if not rows:
        raise RuntimeError("No frames processed")

    with open(args.out_csv, "w", newline="") as f:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = [r for r in rows if int(r.get("used_for_average", 0)) == 1]

    avg = {}
    if avg_rows:
        for k in sorted(set().union(*(r.keys() for r in avg_rows))):
            avg[k] = finite_mean(r.get(k) for r in avg_rows)

    summary = {
        **vars(args),
        "depth_total_frames": depth_total,
        "video_total_frames": video_total,
        "processed_rows": len(rows),
        "averaged_frames": len(avg_rows),
        "camera_poc_min": int(min(camera_pocs)),
        "camera_poc_max": int(max(camera_pocs)),
        "average": avg,
        "csv": args.out_csv,
        "pred_yuv": args.out_pred_yuv,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done.")
    print(f"CSV      : {args.out_csv}")
    print(f"Summary  : {args.out_json}")
    if args.out_pred_yuv:
        print(f"Pred YUV : {args.out_pred_yuv}")

    if avg_rows:
        print(f"Average Y-PSNR       : {avg.get('y_psnr', float('nan')):.3f}")
        print(f"Average valid Y-PSNR : {avg.get('y_valid_psnr', float('nan')):.3f}")
        print(f"Average YUV420-PSNR  : {avg.get('yuv420_psnr', float('nan')):.3f}")


if __name__ == "__main__":
    main()
