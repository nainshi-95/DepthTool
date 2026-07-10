#!/usr/bin/env python3
"""Patch depth_inv_plane_backward_warp_sim_fixed.py for camparam_v2 JSONL.

Changes:
  1) removes the plain temporal plane candidate;
  2) reads camparam_v2_vggt_or_canonical JSONL;
  3) reconstructs per-frame intrinsics and absolute W2C/C2W poses;
  4) converts stored depth sample Y to real camera depth with
       depth_real = Y * depth_scale / depth_scale_precision;
  5) replaces Unity projection-matrix geometry with K-based pinhole geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CAMERA_SECTION = r'''# ============================================================
# Camera JSONL v2 / pose reconstruction
# ============================================================

def rodrigues_to_matrix(rvec):
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))

    if theta < 1e-12:
        # First-order form is more stable around zero.
        x, y, z = r
        k = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + k

    axis = r / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * k
        + (1.0 - math.cos(theta)) * (k @ k)
    )


def rt_to_4x4(rvec, tvec):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rodrigues_to_matrix(rvec)
    t[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return t


def intrinsic_vec_to_matrix(v):
    fx, fy, cx, cy = [float(x) for x in v]

    if not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError(f"non-finite intrinsic: {v}")

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"fx/fy must be positive: fx={fx}, fy={fy}")

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_camera_json(path):
    """Read camparam_v2_vggt_or_canonical JSONL.

    The first JSON object must be the header and each following object must be
    a frame record containing poc, rvec, tvec, and intrinsic_delta.
    """
    header = None
    frames = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse camera JSONL at line {line_no}: {path}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Camera JSONL line {line_no} must be an object, got {type(obj)}"
                )

            if obj.get("type") == "header":
                if header is not None:
                    raise ValueError(f"Multiple camera headers found: {path}")
                header = obj
            else:
                frames.append(obj)

    if header is None:
        raise ValueError(f"Camera JSONL header not found: {path}")

    required_header = [
        "width",
        "height",
        "depth_scale",
        "depth_scale_precision",
        "intrinsic",
        "pose_mode",
    ]

    for k in required_header:
        if k not in header:
            raise KeyError(f"Camera JSONL header is missing '{k}': {path}")

    if not frames:
        raise ValueError(f"Camera JSONL has no frame records: {path}")

    required_frame = ["poc", "rvec", "tvec", "intrinsic_delta"]

    for i, rec in enumerate(frames):
        for k in required_frame:
            if k not in rec:
                raise KeyError(f"Camera frame record {i} is missing '{k}': {path}")

    return {"header": header, "frames": frames}


def get_depth_scale_real_from_header(header):
    # IMPORTANT: depth_scale is fixed-point. The real depth step is the
    # integer value divided by depth_scale_precision.
    scale_int = float(header["depth_scale"])
    precision = float(header["depth_scale_precision"])

    if precision <= 0.0:
        raise ValueError("depth_scale_precision must be positive")

    scale_real = scale_int / precision

    if not np.isfinite(scale_real) or scale_real <= 0.0:
        raise ValueError(
            f"invalid real depth scale: {scale_int} / {precision} = {scale_real}"
        )

    stored = header.get("depth_scale_real")

    if stored is not None:
        stored = float(stored)
        tol = max(1e-12, abs(scale_real) * 1e-7)

        if not math.isclose(stored, scale_real, rel_tol=1e-7, abs_tol=tol):
            print(
                "[WARN] header depth_scale_real differs from "
                "depth_scale/depth_scale_precision; using the integer ratio: "
                f"stored={stored}, derived={scale_real}"
            )

    return scale_real


def build_camera_lookup(camera_json):
    header = camera_json["header"]
    frame_records = sorted(camera_json["frames"], key=lambda r: int(r["poc"]))

    pocs = [int(r["poc"]) for r in frame_records]

    if len(set(pocs)) != len(pocs):
        raise ValueError("duplicate POC in camera JSONL")

    pose_mode = str(header["pose_mode"])

    if pose_mode == "current_to_previous":
        expected = list(range(len(frame_records)))

        if pocs != expected:
            raise ValueError(
                "current_to_previous camera JSONL requires consecutive local POCs "
                f"0..N-1, got first values={pocs[:8]}"
            )

    intr0 = header["intrinsic"]
    base_intr = np.array(
        [
            float(intr0["fx"]),
            float(intr0["fy"]),
            float(intr0["cx"]),
            float(intr0["cy"]),
        ],
        dtype=np.float64,
    )
    z_sign = float(intr0.get("z_sign", 1.0))

    if z_sign == 0.0 or not np.isfinite(z_sign):
        raise ValueError(f"invalid z_sign: {z_sign}")

    z_sign = 1.0 if z_sign > 0.0 else -1.0
    fixed_intrinsic = (
        header.get("intrinsic_mode") == "rap_fixed"
        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"
    )
    depth_scale_real = get_depth_scale_real_from_header(header)

    by_poc = {}
    by_frame_idx = {}
    ordered = []

    cur_intr = base_intr.copy()
    prev_w2c = np.eye(4, dtype=np.float64)

    for order, rec in enumerate(frame_records):
        poc = int(rec["poc"])
        frame_idx = int(rec.get("frame_idx", poc))

        delta = np.asarray(rec.get("intrinsic_delta", [0, 0, 0, 0]), dtype=np.float64)

        if delta.shape != (4,):
            raise ValueError(
                f"POC {poc}: intrinsic_delta must have 4 values, got {delta.shape}"
            )

        if fixed_intrinsic:
            cur_intr = base_intr.copy()
        else:
            # Header intrinsic is frame-0 K. The written frame-0 delta is zero;
            # later records carry the delta from the previous frame.
            cur_intr = cur_intr + delta

        k = intrinsic_vec_to_matrix(cur_intr)
        t_rec = rt_to_4x4(rec["rvec"], rec["tvec"])

        if pose_mode == "current_to_previous":
            if order == 0:
                w2c = np.eye(4, dtype=np.float64)
            else:
                # JSONL stores X_prev = T_prev_from_cur * X_cur.
                # Therefore W2C_cur = inv(T_prev_from_cur) * W2C_prev.
                try:
                    w2c = np.linalg.inv(t_rec) @ prev_w2c
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f"POC {poc}: singular current_to_previous pose") from exc

        elif pose_mode in ("gop_local", "absolute"):
            # gop_local: X_i = T_i * X_0, where frame-0 camera is the world.
            # absolute : T_i is already camera_from_world.
            w2c = t_rec

        else:
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")

        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"POC {poc}: singular W2C") from exc

        cam = {
            "poc": poc,
            "frame_idx": frame_idx,
            "K": k,
            "W2C": w2c,
            "C2W": c2w,
            "z_sign": z_sign,
            "depth_scale_real": depth_scale_real,
            "pose_mode": pose_mode,
        }

        by_poc[poc] = cam
        by_frame_idx.setdefault(frame_idx, cam)
        ordered.append(cam)
        prev_w2c = w2c

    declared_count = header.get("frame_count")

    if declared_count is not None and int(declared_count) != len(ordered):
        raise ValueError(
            f"camera frame_count mismatch: header={declared_count}, records={len(ordered)}"
        )

    return {
        "header": header,
        "by_poc": by_poc,
        "by_frame_idx": by_frame_idx,
        "ordered": ordered,
    }


def get_camera(lookup, frame_idx):
    # The generated depth YUV is RAP-local, so local POC is the primary key.
    if frame_idx in lookup["by_poc"]:
        return lookup["by_poc"][frame_idx]

    # Fallback supports a caller indexing a global sequence by frame_idx.
    if frame_idx in lookup["by_frame_idx"]:
        return lookup["by_frame_idx"][frame_idx]

    raise KeyError(f"camera for frame/POC {frame_idx} not found")


def camera_has_required_mats(cam):
    try:
        k = np.asarray(cam["K"], dtype=np.float64)
        w2c = np.asarray(cam["W2C"], dtype=np.float64)
        c2w = np.asarray(cam["C2W"], dtype=np.float64)
        scale = float(cam["depth_scale_real"])
        z_sign = float(cam["z_sign"])

        return (
            k.shape == (3, 3)
            and w2c.shape == (4, 4)
            and c2w.shape == (4, 4)
            and np.isfinite(k).all()
            and np.isfinite(w2c).all()
            and np.isfinite(c2w).all()
            and np.isfinite(scale)
            and scale > 0.0
            and z_sign in (-1.0, 1.0)
        )
    except Exception:
        return False


def get_depth_scale_real(cam):
    scale = float(cam["depth_scale_real"])

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid camera depth scale: {scale}")

    return scale
'''


GEOMETRY_SECTION = r'''# ============================================================
# Camera geometry: K-based pinhole model
# ============================================================

def pixel_rays_camera(u, v, cam):
    """Return camera rays whose absolute camera-Z component is one.

    Real camera depth is therefore represented by:
      X_cam = ray(u, v) * depth_real
    where depth_real = depth_y * depth_scale_real.
    """
    k = np.asarray(cam["K"], dtype=np.float64)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    z_sign = float(cam["z_sign"])

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    rx = (u - cx) / fx
    ry = (v - cy) / fy
    rz = np.full_like(rx, z_sign, dtype=np.float64)

    return np.stack([rx, ry, rz], axis=-1)


def project_camera_points(points_cam, cam):
    """Project camera-space points with the same z_sign convention."""
    p = np.asarray(points_cam, dtype=np.float64)
    k = np.asarray(cam["K"], dtype=np.float64)
    z_sign = float(cam["z_sign"])

    depth = z_sign * p[..., 2]
    front = np.isfinite(depth) & (depth > 1e-12)
    safe_depth = np.where(front, depth, 1.0)

    u = k[0, 0] * (p[..., 0] / safe_depth) + k[0, 2]
    v = k[1, 1] * (p[..., 1] / safe_depth) + k[1, 2]

    return u, v, depth, front


def forward_project_depth_to_target_view(
    source_depth_linear,
    cam_source,
    cam_target,
    width,
    height,
    splat_mode="bilinear",
):
    """Forward-project real source-camera depth into the target camera."""
    if splat_mode not in ["nearest", "bilinear"]:
        raise ValueError(f"Unsupported splat_mode: {splat_mode}")

    yy, xx = np.indices((height, width), dtype=np.float64)
    z_src = source_depth_linear.astype(np.float64)
    rays = pixel_rays_camera(xx, yy, cam_source)
    p_src_view = rays * z_src[..., None]

    ones = np.ones((height, width, 1), dtype=np.float64)
    p_src_view4 = np.concatenate([p_src_view, ones], axis=-1)
    p_world = p_src_view4 @ np.asarray(cam_source["C2W"]).T
    p_tgt_cam = p_world @ np.asarray(cam_target["W2C"]).T

    map_x, map_y, z_tgt, front = project_camera_points(
        p_tgt_cam[..., :3],
        cam_target,
    )

    valid_src = (
        front
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(z_src)
        & np.isfinite(z_tgt)
        & (z_src > 0.0)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )

    zbuf = np.full((height * width,), np.inf, dtype=np.float64)

    if splat_mode == "nearest":
        xi = np.rint(map_x[valid_src]).astype(np.int64)
        yi = np.rint(map_y[valid_src]).astype(np.int64)
        zi = z_tgt[valid_src]

        xi = np.clip(xi, 0, width - 1)
        yi = np.clip(yi, 0, height - 1)

        flat_idx = yi * width + xi
        np.minimum.at(zbuf, flat_idx, zi)

    else:
        mx = map_x[valid_src].astype(np.float64)
        my = map_y[valid_src].astype(np.float64)
        zi = z_tgt[valid_src].astype(np.float64)

        x0 = np.floor(mx).astype(np.int64)
        y0 = np.floor(my).astype(np.int64)

        for dy_off in [0, 1]:
            for dx_off in [0, 1]:
                xi = x0 + dx_off
                yi = y0 + dy_off

                ok = (
                    (xi >= 0)
                    & (xi < width)
                    & (yi >= 0)
                    & (yi < height)
                )

                if not np.any(ok):
                    continue

                flat_idx = yi[ok] * width + xi[ok]
                np.minimum.at(zbuf, flat_idx, zi[ok])

    target_depth_linear = zbuf.reshape(height, width)
    target_valid = np.isfinite(target_depth_linear)
    target_depth_linear[~target_valid] = 0.0

    return target_depth_linear, target_valid


def fit_3d_plane(points):
    if points.shape[0] < 3:
        return None

    c = np.mean(points, axis=0)
    q = points - c

    try:
        _, s, vh = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if len(s) < 2 or s[1] < 1e-9:
        return None

    n = vh[-1]
    norm = np.linalg.norm(n)

    if norm < 1e-12:
        return None

    n = n / norm
    d = -float(np.dot(n, c))

    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def transform_plane_src_to_tgt(plane_src, cam_src, cam_tgt):
    c2w_src = np.asarray(cam_src["C2W"], dtype=np.float64)
    w2c_tgt = np.asarray(cam_tgt["W2C"], dtype=np.float64)

    # Column-vector semantic transform: X_tgt = M * X_src.
    m = w2c_tgt @ c2w_src

    try:
        plane_tgt = np.linalg.inv(m).T @ plane_src
    except np.linalg.LinAlgError:
        return None

    n = plane_tgt[:3]
    norm = np.linalg.norm(n)

    if norm < 1e-12:
        return None

    return plane_tgt / norm


def image_inv_plane_to_3d_plane(leaf, cam, frame_w, frame_h, args):
    del frame_w, frame_h  # K already describes the source image coordinates.

    depth_scale_real = get_depth_scale_real(cam)
    ns = max(2, int(args.plane_warp_samples))

    xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)
    ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)

    depth_y = inv_plane_to_depth_value(leaf.plane, uu, vv, args)
    depth_real = depth_y * depth_scale_real

    rays = pixel_rays_camera(uu, vv, cam)
    pts = rays.reshape(-1, 3) * depth_real.reshape(-1, 1)

    valid = np.isfinite(pts).all(axis=1) & (depth_real.reshape(-1) > 0)
    pts = pts[valid]

    if pts.shape[0] < 3:
        return None

    return fit_3d_plane(pts)


def render_3d_plane_to_depth_block(
    plane_cam,
    cam_cur,
    x,
    y,
    w,
    h,
    frame_w,
    frame_h,
    args,
):
    del frame_w, frame_h

    depth_scale_real = get_depth_scale_real(cam_cur)

    gx = np.arange(x, x + w, dtype=np.float64)
    gy = np.arange(y, y + h, dtype=np.float64)
    uu, vv = np.meshgrid(gx, gy)

    rays = pixel_rays_camera(uu, vv, cam_cur)

    n = plane_cam[:3]
    d = plane_cam[3]
    denom = np.sum(rays * n.reshape(1, 1, 3), axis=-1)

    valid = np.abs(denom) > 1e-12
    depth_real = np.full((h, w), np.nan, dtype=np.float64)
    depth_real[valid] = -d / denom[valid]

    valid = valid & np.isfinite(depth_real) & (depth_real > 0)
    valid_ratio = float(np.mean(valid))

    if valid_ratio < args.plane_warp_min_valid_ratio:
        return None

    # Convert real camera depth back to the stored 10-bit depth sample domain.
    depth_y = depth_real / depth_scale_real
    depth_y = np.clip(depth_y, args.depth_eps, args.max_value)

    if not np.all(valid):
        med = np.median(depth_y[valid]) if np.any(valid) else args.max_value
        depth_y[~valid] = med

    return depth_y


def make_plane_warp_candidate(ctx, x, y, w, h, cx, cy, args, grid):
    if ctx is None:
        return None

    r = leaf_covering_point(ctx.prev_store, cx, cy)

    if r is None:
        return None

    plane3d_src = image_inv_plane_to_3d_plane(
        r,
        ctx.cam_prev,
        ctx.frame_w,
        ctx.frame_h,
        args,
    )

    if plane3d_src is None:
        return None

    plane3d_cur = transform_plane_src_to_tgt(
        plane3d_src,
        ctx.cam_prev,
        ctx.cam_cur,
    )

    if plane3d_cur is None:
        return None

    depth_block = render_3d_plane_to_depth_block(
        plane3d_cur,
        ctx.cam_cur,
        x,
        y,
        w,
        h,
        ctx.frame_w,
        ctx.frame_h,
        args,
    )

    if depth_block is None:
        return None

    _, _, pinv = grid.get(w, h)
    return fit_inv_depth_plane_from_depth_block(depth_block, pinv, cx, cy, args)


def make_candidates(
    store,
    x,
    y,
    w,
    h,
    cx,
    cy,
    max_cands,
    plane_warp_ctx,
    args,
    grid,
):
    cand = []
    conv = {}

    if args.plane_warp_candidate and plane_warp_ctx is not None:
        p = make_plane_warp_candidate(
            plane_warp_ctx,
            x,
            y,
            w,
            h,
            cx,
            cy,
            args,
            grid,
        )

        if p is not None:
            conv["plane_warp"] = p
            cand.append(("plane_warp", p))

    items = [
        ("left", best_left(store, x, y, w, h)),
        ("top", best_top(store, x, y, w, h)),
        ("top_left", top_left(store, x, y)),
        ("top_right", top_right(store, x, y, w)),
    ]

    for name, r in items:
        if r is not None:
            p = plane_to_center(r.plane, cx, cy)
            conv[name] = p
            cand.append((name, p))

    if "left" in conv and "top" in conv:
        l = conv["left"]
        t = conv["top"]

        cand.append(
            (
                "avg_left_top",
                Plane(
                    0.5 * (l.a + t.a),
                    0.5 * (l.b + t.b),
                    0.5 * (l.c + t.c),
                    cx,
                    cy,
                ),
            )
        )

    return cand[:max_cands]
'''


OLD_TEMPORAL_LOOKUP = r'''def temporal_center(prev_store, cx, cy):
    if not prev_store:
        return None

    for r in prev_store:
        if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
            return r

    return None
'''

NEW_LEAF_LOOKUP = r'''def leaf_covering_point(store, cx, cy):
    if not store:
        return None

    for r in store:
        if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
            return r

    return None
'''

OLD_MAKE_CAND_CALL = r'''    cands = make_candidates(
        store=store,
        prev_store=prev_store,
        x=x,
        y=y,
        w=w,
        h=h,
        cx=cx,
        cy=cy,
        max_cands=args.max_candidates,
        use_temporal=args.temporal_candidate,
        plane_warp_ctx=plane_warp_ctx,
        args=args,
        grid=grid,
    )
'''

NEW_MAKE_CAND_CALL = r'''    cands = make_candidates(
        store=store,
        x=x,
        y=y,
        w=w,
        h=h,
        cx=cx,
        cy=cy,
        max_cands=args.max_candidates,
        plane_warp_ctx=plane_warp_ctx,
        args=args,
        grid=grid,
    )
'''

OLD_BACKWARD_MAP = r'''def make_backward_map_cur_to_prev(depth_y_cur, cam_cur, cam_prev, width, height):
    """
    Backward map from current pixels to previous-frame pixel coordinates.

    Uses the same convention as the verified full-depth projection script:
      - depth_y is stored sample Y
      - actual current linear depth = depth_y * nearClipPlane_current
      - inv-projection ray is normalized by abs(view_z)
      - row-vector matrix application uses @ M.T
    """
    inv_proj_cur = get_matrix(cam_cur, INV_PROJ_ALIASES)
    c2w_cur = get_matrix(cam_cur, C2W_ALIASES)
    w2c_prev = get_matrix(cam_prev, W2C_ALIASES)
    proj_prev = get_matrix(cam_prev, PROJ_ALIASES)

    near_cur = get_near_clip(cam_cur)

    _, _, x_ndc, y_ndc = make_ndc_grid(width, height)

    p_ndc = np.stack(
        [x_ndc, y_ndc, np.ones_like(x_ndc), np.ones_like(x_ndc)],
        axis=-1,
    )

    p_view_h = p_ndc @ inv_proj_cur.T
    p_view = p_view_h[..., :3] / np.maximum(p_view_h[..., 3:4], 1e-12)

    denom = np.maximum(np.abs(p_view[..., 2:3]), 1e-12)
    linear_z = depth_y_cur.astype(np.float64) * near_cur
    p_cur = p_view / denom * linear_z[..., None]

    ones = np.ones((height, width, 1), dtype=np.float64)
    p_cur_h = np.concatenate([p_cur, ones], axis=-1)

    p_world = p_cur_h @ c2w_cur.T
    p_prev_cam = p_world @ w2c_prev.T
    clip = p_prev_cam @ proj_prev.T

    cw = clip[..., 3]
    good_w = np.abs(cw) > 1e-12

    ndc_x = clip[..., 0] / np.where(good_w, cw, 1.0)
    ndc_y = clip[..., 1] / np.where(good_w, cw, 1.0)

    map_x = (ndc_x + 1.0) * 0.5 * width - 0.5
    map_y = (1.0 - ndc_y) * 0.5 * height - 0.5

    valid = (
        good_w
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(linear_z)
        & np.isfinite(p_prev_cam[..., 2])
        & (linear_z > 0.0)
        & (np.abs(p_prev_cam[..., 2]) > 1e-12)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y <= height - 1.0)
    )

    return map_x, map_y, valid
'''

NEW_BACKWARD_MAP = r'''def make_backward_map_cur_to_prev(depth_y_cur, cam_cur, cam_prev, width, height):
    """Backward map from current pixels to previous-frame pixel coordinates.

    Stored depth is fixed-point coded. Camera geometry must use:
      depth_real = depth_y * depth_scale / depth_scale_precision
    """
    yy, xx = np.indices((height, width), dtype=np.float64)
    rays_cur = pixel_rays_camera(xx, yy, cam_cur)

    depth_real = (
        depth_y_cur.astype(np.float64)
        * get_depth_scale_real(cam_cur)
    )
    p_cur = rays_cur * depth_real[..., None]

    ones = np.ones((height, width, 1), dtype=np.float64)
    p_cur_h = np.concatenate([p_cur, ones], axis=-1)

    p_world = p_cur_h @ np.asarray(cam_cur["C2W"], dtype=np.float64).T
    p_prev_cam = p_world @ np.asarray(cam_prev["W2C"], dtype=np.float64).T

    map_x, map_y, depth_prev, front = project_camera_points(
        p_prev_cam[..., :3],
        cam_prev,
    )

    valid = (
        front
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(depth_real)
        & np.isfinite(depth_prev)
        & (depth_real > 0.0)
        & (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y <= height - 1.0)
    )

    return map_x, map_y, valid
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_section(text: str, start_marker: str, end_marker: str, new: str, label: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"{label}: section markers not found")
    return text[:start] + new.rstrip() + "\n\n\n" + text[end:]


def patch_source(text: str) -> str:
    text = replace_once(
        text,
        '        "temporal",\n',
        '',
        "remove temporal adaptive symbol",
    )

    text = replace_once(
        text,
        OLD_TEMPORAL_LOOKUP,
        NEW_LEAF_LOOKUP,
        "replace temporal leaf lookup",
    )

    text = text.replace(
        "# Spatial / temporal candidates",
        "# Spatial candidates / previous-frame leaf lookup",
        1,
    )

    text = replace_section(
        text,
        "# ============================================================\n# Camera JSON / TXT / matrices\n# ============================================================",
        "# ============================================================\n# Camera geometry\n# ============================================================",
        CAMERA_SECTION,
        "camera parser section",
    )

    text = replace_section(
        text,
        "# ============================================================\n# Camera geometry\n# ============================================================",
        "# ============================================================\n# Mode evaluation\n# ============================================================",
        GEOMETRY_SECTION,
        "camera geometry section",
    )

    text = replace_once(
        text,
        OLD_MAKE_CAND_CALL,
        NEW_MAKE_CAND_CALL,
        "make_candidates call",
    )

    text = replace_once(
        text,
        OLD_BACKWARD_MAP,
        NEW_BACKWARD_MAP,
        "backward map",
    )

    text = replace_once(
        text,
        '    p.add_argument("--temporal-candidate", action="store_true")\n',
        '',
        "remove temporal CLI option",
    )

    text = replace_once(
        text,
        '    p.add_argument("--camera-param", required=True, help="camera JSON/TXT/JSONL")\n',
        '    p.add_argument("--camera-param", required=True, help="camparam_v2 JSONL")\n',
        "camera help",
    )

    old_main = '''    cam_json = load_camera_json(args.camera_param)\n    camera_lookup = build_camera_lookup(cam_json)\n\n    grid = GridCache()\n'''
    new_main = '''    cam_json = load_camera_json(args.camera_param)\n    cam_header = cam_json["header"]\n\n    if int(cam_header["width"]) != args.width or int(cam_header["height"]) != args.height:\n        raise ValueError(\n            "camera/depth resolution mismatch: "\n            f"camera={cam_header['width']}x{cam_header['height']}, "\n            f"input={args.width}x{args.height}"\n        )\n\n    camera_lookup = build_camera_lookup(cam_json)\n    print(\n        "Camera depth scale real: "\n        f"{get_depth_scale_real_from_header(cam_header):.12g} "\n        "(depth_scale / depth_scale_precision)"\n    )\n\n    grid = GridCache()\n'''
    text = replace_once(text, old_main, new_main, "main camera initialization")

    text = text.replace(
        "# Inverse-depth plane compression simulation + camera-plane candidate\n# + backward projection predictor.",
        "# Inverse-depth plane compression simulation + camera-plane candidate\n# + backward projection predictor using camparam_v2 JSONL.",
        1,
    )

    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Original depth_inv_plane_backward_warp_sim_fixed.py")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.in_place and args.output is not None:
        raise ValueError("Use either --in-place or --output")

    if args.in_place:
        out = args.input
    elif args.output is not None:
        out = args.output
    else:
        out = args.input.with_name(args.input.stem + "_camparam_v2.py")

    src = args.input.read_text(encoding="utf-8")
    patched = patch_source(src)

    if out == args.input:
        backup = args.input.with_suffix(args.input.suffix + ".bak")
        backup.write_text(src, encoding="utf-8")
        print(f"Backup: {backup}")

    out.write_text(patched, encoding="utf-8")
    print(f"Patched: {out}")


if __name__ == "__main__":
    main()
