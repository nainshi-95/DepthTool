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

    return y.reshape(h, w), u.reshape(h // 2, w // 2), v.reshape(h // 2, w // 2)


def write_yuv420(path, y, u, v):
    with open(path, "ab") as f:
        f.write(np.ascontiguousarray(y).tobytes())
        f.write(np.ascontiguousarray(u).tobytes())
        f.write(np.ascontiguousarray(v).tobytes())


def write_depth_yuv420p10le(path, depth_y, w, h):
    u = np.full((h // 2, w // 2), 512, dtype=np.uint16)
    v = np.full((h // 2, w // 2), 512, dtype=np.uint16)
    write_yuv420(path, depth_y.astype(np.uint16), u, v)


# ============================================================
# Camera parsing
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
        and ("ProjectionMatrix" in x or "projectionMatrix" in x)
        and ("WorldToCameraMatrix" in x or "worldToCameraMatrix" in x)
        and (
            "CameraToWorldMatrix" in x
            or "cameraToWorldMatrix" in x
            or "CameraToWorldMarix" in x
            or "cameraToWorldMarix" in x
        )
    )


def camera_poc(entry, fallback):
    for k in ["frames", "frame", "frameIdx", "frame_idx", "poc", "POC"]:
        if k in entry:
            return int(entry[k])
    return fallback


def build_camera_lookup(obj):
    if isinstance(obj, list):
        entries = obj
    elif isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
        entries = obj["frames"]
    elif isinstance(obj, dict) and has_mats(obj):
        entries = [obj]
    else:
        items = []
        for k, v in obj.items():
            if isinstance(v, dict) and has_mats(v):
                items.append((int(k), v))
        entries = [v for _, v in sorted(items)]

    lookup = {}
    pocs = []

    for i, e in enumerate(entries):
        poc = camera_poc(e, i)
        lookup[poc] = e
        lookup.setdefault(i, e)
        pocs.append(poc)

    return lookup, sorted(set(pocs))


def get_alias(entry, names):
    for n in names:
        if n in entry:
            return entry[n]
    raise KeyError(names)


def get_near(cam):
    return float(get_alias(cam, ["nearClipPlane", "NearClipPlane"]))


def get_matrix(cam, name):
    aliases = {
        "InvProjectionMatrix": ["InvProjectionMatrix", "invProjectionMatrix"],
        "ProjectionMatrix": ["ProjectionMatrix", "projectionMatrix"],
        "WorldToCameraMatrix": ["WorldToCameraMatrix", "worldToCameraMatrix"],
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
        return m.T

    m = np.array(obj, dtype=np.float32)
    if m.shape == (16,):
        m = m.reshape(4, 4)
    return m


# ============================================================
# Intrinsic / extrinsic
# ============================================================

def make_grid(w, h):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    x_ndc = (x + 0.5) / w * 2.0 - 1.0
    y_ndc = 1.0 - (y + 0.5) / h * 2.0
    return x, y, x_ndc, y_ndc


def derive_intrinsic_4(cam, w, h):
    """
    ProjectionMatrix convention에 직접 의존하지 않고,
    InvProjectionMatrix로 pixel ray를 복원한 뒤
    u = fx * X/abs(Z) + cx
    v = fy * Y/abs(Z) + cy
    로 fitting한다.
    """
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

    A_x = np.stack([rx, np.ones_like(rx)], axis=1)
    A_y = np.stack([ry, np.ones_like(ry)], axis=1)

    fx, cx = np.linalg.lstsq(A_x, u.reshape(-1), rcond=None)[0]
    fy, cy = np.linalg.lstsq(A_y, v.reshape(-1), rcond=None)[0]

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


def derive_extrinsic_6_cur_to_prev(cam_cur, cam_prev):
    """
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


# ============================================================
# Depth generation
# ============================================================

def compact_idx_from_poc(poc, first_poc, step, count):
    d = poc - first_poc
    if d < 0 or d % step != 0:
        return None
    idx = d // step
    if idx < 0 or idx >= count:
        return None
    return idx


def forward_depth_to_target(depth_linear_src, cam_src, cam_tgt, w, h):
    invP = get_matrix(cam_src, "InvProjectionMatrix")
    C2W = get_matrix(cam_src, "CameraToWorldMatrix")
    W2C = get_matrix(cam_tgt, "WorldToCameraMatrix")
    P = get_matrix(cam_tgt, "ProjectionMatrix")

    _, _, x_ndc, y_ndc = make_grid(w, h)

    p = np.stack(
        [x_ndc, y_ndc, np.ones_like(x_ndc), np.ones_like(x_ndc)],
        axis=-1,
    )

    q = p @ invP.T
    q = q[..., :3] / np.maximum(q[..., 3:4], 1e-8)

    q = q / np.maximum(np.abs(q[..., 2:3]), 1e-8)
    q = q * depth_linear_src[..., None]

    q4 = np.concatenate([q, np.ones((h, w, 1), dtype=np.float32)], axis=-1)

    world = q4 @ C2W.T
    tgt = world @ W2C.T
    clip = tgt @ P.T

    wc = clip[..., 3]
    ok_w = np.abs(wc) > 1e-8

    ndc_x = clip[..., 0] / np.where(ok_w, wc, 1.0)
    ndc_y = clip[..., 1] / np.where(ok_w, wc, 1.0)

    mx = (ndc_x + 1.0) * 0.5 * w - 0.5
    my = (1.0 - ndc_y) * 0.5 * h - 0.5
    z = np.abs(tgt[..., 2]).astype(np.float32)

    valid = (
        ok_w
        & np.isfinite(mx)
        & np.isfinite(my)
        & np.isfinite(z)
        & (z > 0)
        & (mx >= 0)
        & (mx <= w - 1)
        & (my >= 0)
        & (my <= h - 1)
    )

    zbuf = np.full(h * w, np.inf, dtype=np.float32)

    mxv = mx[valid].astype(np.float32)
    myv = my[valid].astype(np.float32)
    zv = z[valid].astype(np.float32)

    x0 = np.floor(mxv).astype(np.int32)
    y0 = np.floor(myv).astype(np.int32)

    for dy in [0, 1]:
        for dx in [0, 1]:
            xi = x0 + dx
            yi = y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            np.minimum.at(zbuf, yi[ok] * w + xi[ok], zv[ok])

    out = zbuf.reshape(h, w)
    valid = np.isfinite(out)
    out[~valid] = 0.0
    return out, valid


def fill_nearest(depth, valid):
    if valid.all():
        return depth

    try:
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(~valid, return_indices=True)
        filled = depth.copy()
        filled[~valid] = depth[idx[0][~valid], idx[1][~valid]]
        return filled
    except Exception:
        filled = depth.copy()
        mask = valid.copy()
        k = np.ones((3, 3), dtype=np.float32)

        for _ in range(1024):
            if mask.all():
                break

            mf = mask.astype(np.float32)
            s = cv2.filter2D(filled * mf, -1, k, borderType=cv2.BORDER_REPLICATE)
            c = cv2.filter2D(mf, -1, k, borderType=cv2.BORDER_REPLICATE)

            fill = (~mask) & (c > 0)
            filled[fill] = s[fill] / np.maximum(c[fill], 1e-8)
            mask[fill] = True

        if not mask.all():
            filled[~mask] = np.median(filled[mask])

        return filled


def get_full_depth_y(
    poc,
    depth_yuv,
    depth_count,
    cams,
    w,
    h,
    first_poc,
    step,
):
    if poc == 0:
        y, _, _ = read_yuv420(depth_yuv, 0, w, h, 10)
        return y.astype(np.uint16)

    idx = compact_idx_from_poc(poc, first_poc, step, depth_count)
    if idx is not None:
        y, _, _ = read_yuv420(depth_yuv, idx, w, h, 10)
        return y.astype(np.uint16)

    src_poc = poc - 1
    src_idx = compact_idx_from_poc(src_poc, first_poc, step, depth_count)
    if src_idx is None:
        raise RuntimeError(f"Cannot make depth for POC {poc}")

    src_y, _, _ = read_yuv420(depth_yuv, src_idx, w, h, 10)

    cam_src = cams[src_poc]
    cam_tgt = cams[poc]

    src_linear = src_y.astype(np.float32) * get_near(cam_src)

    tgt_sparse, valid = forward_depth_to_target(
        src_linear,
        cam_src,
        cam_tgt,
        w,
        h,
    )

    tgt_filled = fill_nearest(tgt_sparse, valid)
    tgt_y = np.round(tgt_filled / max(get_near(cam_tgt), 1e-8))
    return np.clip(tgt_y, 0, 1023).astype(np.uint16)


# ============================================================
# Backward warp using depth + intrinsic 4 + extrinsic 6
# ============================================================

def backward_maps_from_params(depth_linear_cur, intr_cur, intr_prev, ext_cur_to_prev, w, h):
    x, y, _, _ = make_grid(w, h)

    fx = intr_cur["fx"]
    fy = intr_cur["fy"]
    cx = intr_cur["cx"]
    cy = intr_cur["cy"]
    zsign = intr_cur["z_sign"]

    z = depth_linear_cur.astype(np.float32)

    X = np.empty((h, w, 3), dtype=np.float32)
    X[..., 0] = (x - cx) / fx * z
    X[..., 1] = (y - cy) / fy * z
    X[..., 2] = zsign * z

    rvec = np.array(ext_cur_to_prev["rvec"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(ext_cur_to_prev["tvec"], dtype=np.float32).reshape(1, 1, 3)
    R, _ = cv2.Rodrigues(rvec)

    Xp = X @ R.T + tvec

    zprev = np.maximum(np.abs(Xp[..., 2]), 1e-8)

    map_x = intr_prev["fx"] * (Xp[..., 0] / zprev) + intr_prev["cx"]
    map_y = intr_prev["fy"] * (Xp[..., 1] / zprev) + intr_prev["cy"]

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (Xp[..., 2] * intr_prev["z_sign"] > 0)
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

    warped = cv2.remap(
        src.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )

    warped = np.clip(np.round(warped), 0, maxv)

    if bit_depth <= 8:
        return warped.astype(np.uint8)
    return warped.astype(np.uint16)


def backward_warp_yuv420(prev_y, prev_u, prev_v, map_x, map_y, bit_depth):
    h, w = prev_y.shape
    uv_h, uv_w = h // 2, w // 2

    y = remap_plane(prev_y, map_x, map_y, bit_depth, 0)

    map_x_uv = cv2.resize(map_x, (uv_w, uv_h), interpolation=cv2.INTER_LINEAR) * 0.5
    map_y_uv = cv2.resize(map_y, (uv_w, uv_h), interpolation=cv2.INTER_LINEAR) * 0.5

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
    ap.add_argument("--out-depth-yuv", required=True)
    ap.add_argument("--out-param-jsonl", required=True)

    ap.add_argument("--depth-first-poc", type=int, default=1)
    ap.add_argument("--depth-poc-step", type=int, default=2)

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    seq_yuv = Path(args.seq_yuv)
    depth_yuv = Path(args.depth_yuv)
    param_txt = Path(args.param_txt)

    out_yuv = Path(args.out_yuv)
    out_depth_yuv = Path(args.out_depth_yuv)
    out_param = Path(args.out_param_jsonl)

    for p in [out_yuv, out_depth_yuv, out_param]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}")

    w = args.width
    h = args.height

    cam_obj = load_camera_file(param_txt)
    cams, camera_pocs = build_camera_lookup(cam_obj)

    seq_count = count_frames(seq_yuv, w, h, args.bit_depth)
    depth_count = count_frames(depth_yuv, w, h, 10)

    last_depth_poc = args.depth_first_poc + (depth_count - 1) * args.depth_poc_step
    max_poc = min(seq_count - 1, max(camera_pocs), last_depth_poc + 1)

    intr_cache = {}

    def intr(poc):
        if poc not in intr_cache:
            intr_cache[poc] = derive_intrinsic_4(cams[poc], w, h)
        return intr_cache[poc]

    with open(out_param, "w", encoding="utf-8") as fparam:
        for poc in range(max_poc + 1):
            cam_cur = cams[poc]
            depth_y = get_full_depth_y(
                poc=poc,
                depth_yuv=depth_yuv,
                depth_count=depth_count,
                cams=cams,
                w=w,
                h=h,
                first_poc=args.depth_first_poc,
                step=args.depth_poc_step,
            )

            write_depth_yuv420p10le(out_depth_yuv, depth_y, w, h)

            if poc == 0:
                y, u, v = read_yuv420(seq_yuv, 0, w, h, args.bit_depth)
                write_yuv420(out_yuv, y, u, v)

                rec = {
                    "poc": 0,
                    "intrinsic": intr(0),
                    "extrinsic_cur_to_prev": {
                        "rvec": [0.0, 0.0, 0.0],
                        "tvec": [0.0, 0.0, 0.0],
                    },
                }
                fparam.write(json.dumps(rec) + "\n")
                continue

            cam_prev = cams[poc - 1]

            intrinsic_cur = intr(poc)
            intrinsic_prev = intr(poc - 1)
            extrinsic = derive_extrinsic_6_cur_to_prev(cam_cur, cam_prev)

            depth_linear = depth_y.astype(np.float32) * get_near(cam_cur)

            map_x, map_y = backward_maps_from_params(
                depth_linear_cur=depth_linear,
                intr_cur=intrinsic_cur,
                intr_prev=intrinsic_prev,
                ext_cur_to_prev=extrinsic,
                w=w,
                h=h,
            )

            prev_y, prev_u, prev_v = read_yuv420(
                seq_yuv,
                poc - 1,
                w,
                h,
                args.bit_depth,
            )

            wy, wu, wv = backward_warp_yuv420(
                prev_y,
                prev_u,
                prev_v,
                map_x,
                map_y,
                args.bit_depth,
            )

            write_yuv420(out_yuv, wy, wu, wv)

            rec = {
                "poc": poc,
                "intrinsic": intrinsic_cur,
                "intrinsic_prev": intrinsic_prev,
                "extrinsic_cur_to_prev": extrinsic,
            }
            fparam.write(json.dumps(rec) + "\n")

            print(f"[{poc:04d}/{max_poc:04d}] warped")

    print("Done.")
    print(f"warped yuv   : {out_yuv}")
    print(f"full depth   : {out_depth_yuv}")
    print(f"params jsonl : {out_param}")


if __name__ == "__main__":
    main()
