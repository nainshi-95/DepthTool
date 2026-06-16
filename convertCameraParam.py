import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# YUV
# ============================================================

def frame_size_yuv420(w, h, bit_depth):
    bps = 1 if bit_depth <= 8 else 2
    return (w * h + 2 * (w // 2) * (h // 2)) * bps


def count_frames(path, w, h, bit_depth):
    return os.path.getsize(path) // frame_size_yuv420(w, h, bit_depth)


def read_yuv420(path, idx, w, h, bit_depth):
    dtype = np.uint8 if bit_depth <= 8 else np.uint16

    y_size = w * h
    uv_size = (w // 2) * (h // 2)
    fs = frame_size_yuv420(w, h, bit_depth)

    with open(path, "rb") as f:
        f.seek(idx * fs)
        y = np.fromfile(f, dtype=dtype, count=y_size)
        u = np.fromfile(f, dtype=dtype, count=uv_size)
        v = np.fromfile(f, dtype=dtype, count=uv_size)

    if y.size != y_size or u.size != uv_size or v.size != uv_size:
        raise RuntimeError(f"Cannot read frame {idx}: {path}")

    return (
        y.reshape(h, w),
        u.reshape(h // 2, w // 2),
        v.reshape(h // 2, w // 2),
    )


def write_yuv420(path, y, u, v):
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(y).tobytes())
        f.write(np.ascontiguousarray(u).tobytes())
        f.write(np.ascontiguousarray(v).tobytes())


# ============================================================
# Camera
# ============================================================

def load_camera_file(path):
    text = Path(path).read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def has_mats(x):
    if not isinstance(x, dict):
        return False

    return (
        ("InvProjectionMatrix" in x or "invProjectionMatrix" in x)
        and ("WorldToCameraMatrix" in x or "worldToCameraMatrix" in x)
        and (
            "CameraToWorldMatrix" in x
            or "cameraToWorldMatrix" in x
            or "CameraToWorldMarix" in x
            or "cameraToWorldMarix" in x
        )
    )


def build_camera_lookup(obj):
    if isinstance(obj, list):
        entries = obj

    elif isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
        entries = obj["frames"]

    elif isinstance(obj, dict) and has_mats(obj):
        entries = [obj]

    elif isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if isinstance(v, dict) and has_mats(v):
                items.append((int(k), v))
        entries = [v for _, v in sorted(items)]

    else:
        raise RuntimeError("Unsupported camera parameter format")

    lookup = {}
    pocs = []

    for i, e in enumerate(entries):
        poc = i

        for k in ["frames", "frame", "frameIdx", "frame_idx", "poc", "POC"]:
            if k in e:
                poc = int(e[k])
                break

        lookup[poc] = e
        lookup.setdefault(i, e)
        pocs.append(poc)

    return lookup, sorted(set(pocs))


def get_alias(cam, names):
    for n in names:
        if n in cam:
            return cam[n]

    raise KeyError(names)


def get_near(cam):
    return float(get_alias(cam, ["nearClipPlane", "NearClipPlane"]))


def get_matrix(cam, name):
    aliases = {
        "InvProjectionMatrix": [
            "InvProjectionMatrix",
            "invProjectionMatrix",
        ],
        "WorldToCameraMatrix": [
            "WorldToCameraMatrix",
            "worldToCameraMatrix",
        ],
        "CameraToWorldMatrix": [
            "CameraToWorldMatrix",
            "cameraToWorldMatrix",
            "CameraToWorldMarix",
            "cameraToWorldMarix",
        ],
    }

    obj = get_alias(cam, aliases[name])

    if isinstance(obj, dict):
        m = np.zeros((4, 4), dtype=np.float32)

        for r in range(4):
            for c in range(4):
                m[r, c] = float(obj[f"e{r}{c}"])

        # 기존 코드의 convention 유지
        return m.T

    m = np.array(obj, dtype=np.float32)

    if m.shape == (16,):
        m = m.reshape(4, 4)

    return m.astype(np.float32)


# ============================================================
# Intrinsic / RT
# ============================================================

def make_grid(w, h):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return x, y


def derive_intrinsic_4(cam, w, h):
    invP = get_matrix(cam, "InvProjectionMatrix")

    xs = np.linspace(0, w - 1, 32, dtype=np.float32)
    ys = np.linspace(0, h - 1, 18, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)

    x_ndc = (u + 0.5) / w * 2.0 - 1.0
    y_ndc = 1.0 - (v + 0.5) / h * 2.0

    p = np.stack(
        [x_ndc, y_ndc, np.ones_like(x_ndc), np.ones_like(x_ndc)],
        axis=-1,
    ).reshape(-1, 4)

    q = p @ invP.T
    q = q[:, :3] / np.maximum(q[:, 3:4], 1e-8)

    zabs = np.maximum(np.abs(q[:, 2]), 1e-8)

    rx = q[:, 0] / zabs
    ry = q[:, 1] / zabs

    fx, cx = np.linalg.lstsq(
        np.stack([rx, np.ones_like(rx)], axis=1),
        u.reshape(-1),
        rcond=None,
    )[0]

    fy, cy = np.linalg.lstsq(
        np.stack([ry, np.ones_like(ry)], axis=1),
        v.reshape(-1),
        rcond=None,
    )[0]

    z_sign = float(np.sign(np.median(q[:, 2])))

    if z_sign == 0:
        z_sign = -1.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "z_sign": z_sign,
    }


def derive_rt_cur_to_prev(cam_cur, cam_prev):
    """
    Relative extrinsic for warping.

    X_prev = R * X_cur + t
    """
    C2W_cur = get_matrix(cam_cur, "CameraToWorldMatrix")
    W2C_prev = get_matrix(cam_prev, "WorldToCameraMatrix")

    T = W2C_prev @ C2W_cur

    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)

    rvec, _ = cv2.Rodrigues(R)

    return {
        "rvec": rvec.reshape(3).astype(float).tolist(),
        "tvec": t.reshape(3).astype(float).tolist(),
    }


def derive_absolute_extrinsic(cam):
    """
    Absolute extrinsic for JSONL saving.

    WorldToCamera 기준:
        X_cam = R * X_world + t
    """
    W2C = get_matrix(cam, "WorldToCameraMatrix")

    R = W2C[:3, :3].astype(np.float32)
    t = W2C[:3, 3].astype(np.float32)

    rvec, _ = cv2.Rodrigues(R)

    return {
        "rvec": rvec.reshape(3).astype(float).tolist(),
        "tvec": t.reshape(3).astype(float).tolist(),
    }


# ============================================================
# Backward warp using converted params
# ============================================================

def backward_map_from_depth_and_params(depth_linear, intr, rt, w, h):
    x, y = make_grid(w, h)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    z_sign = intr["z_sign"]

    z = depth_linear.astype(np.float32)

    X = np.empty((h, w, 3), dtype=np.float32)
    X[..., 0] = (x - cx) / fx * z
    X[..., 1] = (y - cy) / fy * z
    X[..., 2] = z_sign * z

    rvec = np.array(rt["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(rt["tvec"], dtype=np.float32).reshape(1, 1, 3)

    R, _ = cv2.Rodrigues(rvec)

    Xp = X @ R.T + tvec

    zprev = np.maximum(np.abs(Xp[..., 2]), 1e-8)

    map_x = fx * (Xp[..., 0] / zprev) + cx
    map_y = fy * (Xp[..., 1] / zprev) + cy

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (Xp[..., 2] * z_sign > 0)
        & (map_x >= 0)
        & (map_x <= w - 1)
        & (map_y >= 0)
        & (map_y <= h - 1)
        & (z > 0)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y


def remap_plane(src, map_x, map_y, bit_depth, border_value):
    maxv = (1 << bit_depth) - 1

    dst = cv2.remap(
        src.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )

    dst = np.clip(np.round(dst), 0, maxv)

    if bit_depth <= 8:
        return dst.astype(np.uint8)

    return dst.astype(np.uint16)


def backward_warp_yuv420(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    h, w = prev_y.shape

    uv_w = w // 2
    uv_h = h // 2

    y = remap_plane(prev_y, map_x, map_y, bit_depth, 0)

    map_x_uv = cv2.resize(
        map_x,
        (uv_w, uv_h),
        interpolation=cv2.INTER_LINEAR,
    ) * 0.5

    map_y_uv = cv2.resize(
        map_y,
        (uv_w, uv_h),
        interpolation=cv2.INTER_LINEAR,
    ) * 0.5

    neutral = 128 if bit_depth <= 8 else 512

    u = remap_plane(prev_u, map_x_uv, map_y_uv, bit_depth, neutral)
    v = remap_plane(prev_v, map_x_uv, map_y_uv, bit_depth, neutral)

    return y, u, v


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seq-yuv", required=True)
    ap.add_argument("--depth-yuv", required=True)
    ap.add_argument("--param-txt", required=True)

    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--bit-depth", type=int, default=10)

    ap.add_argument("--out-yuv", required=True)
    ap.add_argument("--out-param-jsonl", required=True)

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_txt = Path(args.param_txt)
    out_yuv = Path(args.out_yuv)
    out_param = Path(args.out_param_jsonl)

    for p in [out_yuv, out_param]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}")

    w = args.width
    h = args.height
    bit_depth = args.bit_depth

    cam_obj = load_camera_file(param_txt)
    cams, camera_pocs = build_camera_lookup(cam_obj)

    seq_count = count_frames(seq_yuv, w, h, bit_depth)
    depth_count = count_frames(depth_yuv, w, h, 10)

    max_poc = min(seq_count, depth_count, max(camera_pocs) + 1)

    # intrinsic은 첫 frame 기준으로 1번만 사용
    intr = derive_intrinsic_4(cams[0], w, h)

    depth_scale = get_near(cams[0])

    with open(out_param, "w", encoding="utf-8") as fp:
        fp.write(json.dumps({
            "type": "header",
            "depth_scale": depth_scale,
            "intrinsic": intr,
            "extrinsic_type": "absolute_world_to_camera",
            "extrinsic_rotation": "rodrigues_rvec",
            "extrinsic_translation": "tvec",
        }) + "\n")

        for poc in range(max_poc):
            cur_y, cur_u, cur_v = read_yuv420(
                seq_yuv,
                poc,
                w,
                h,
                bit_depth,
            )

            # JSONL 저장용 absolute extrinsic
            abs_ext = derive_absolute_extrinsic(cams[poc])

            # 첫 frame은 그대로 copy
            if poc == 0:
                write_yuv420(out_yuv, cur_y, cur_u, cur_v)

                fp.write(json.dumps({
                    "poc": poc,
                    "rvec": abs_ext["rvec"],
                    "tvec": abs_ext["tvec"],
                }) + "\n")

                print(f"[{poc:04d}/{max_poc - 1:04d}] copy original")
                continue

            depth_y, _, _ = read_yuv420(
                depth_yuv,
                poc,
                w,
                h,
                10,
            )

            depth_linear = depth_y.astype(np.float32) * get_near(cams[poc])

            # warp 계산용 상대변환
            rt_rel = derive_rt_cur_to_prev(cams[poc], cams[poc - 1])

            map_x, map_y = backward_map_from_depth_and_params(
                depth_linear=depth_linear,
                intr=intr,
                rt=rt_rel,
                w=w,
                h=h,
            )

            prev_y, prev_u, prev_v = read_yuv420(
                seq_yuv,
                poc - 1,
                w,
                h,
                bit_depth,
            )

            wy, wu, wv = backward_warp_yuv420(
                prev_y,
                prev_u,
                prev_v,
                map_x,
                map_y,
                bit_depth,
            )

            write_yuv420(out_yuv, wy, wu, wv)

            mae_y = float(
                np.mean(
                    np.abs(
                        wy.astype(np.float32)
                        - cur_y.astype(np.float32)
                    )
                )
            )

            # JSONL에는 absolute extrinsic 저장
            fp.write(json.dumps({
                "poc": poc,
                "rvec": abs_ext["rvec"],
                "tvec": abs_ext["tvec"],
                "mae_y": mae_y,
            }) + "\n")

            print(f"[{poc:04d}/{max_poc - 1:04d}] Y-MAE={mae_y:.3f}")

    print("Done.")
    print(f"warped yuv : {out_yuv}")
    print(f"params     : {out_param}")


if __name__ == "__main__":
    main()
