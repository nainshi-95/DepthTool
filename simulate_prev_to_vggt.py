#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

CAMERA_AND_GEOMETRY_REPLACEMENT = '\n# ============================================================\n# Camera JSONL v2 produced by the VGGT/canonical converter\n# ============================================================\n\ndef load_camera_json(path):\n    """Load camParam v2 JSONL and normalize it to {header, frames}."""\n    with open(path, "r", encoding="utf-8") as f:\n        text = f.read()\n\n    try:\n        obj = json.loads(text)\n    except json.JSONDecodeError:\n        obj = None\n\n    if isinstance(obj, dict) and "header" in obj and "frames" in obj:\n        header = obj["header"]\n        frames = obj["frames"]\n    else:\n        header = None\n        frames = []\n        for line_no, line in enumerate(text.splitlines(), start=1):\n            line = line.strip()\n            if not line:\n                continue\n            try:\n                rec = json.loads(line)\n            except json.JSONDecodeError as exc:\n                raise RuntimeError(\n                    f"camParam JSONL parse error at line {line_no}: {exc}"\n                ) from exc\n\n            if rec.get("type") == "header":\n                if header is not None:\n                    raise ValueError("camParam contains multiple header records")\n                header = rec\n            else:\n                frames.append(rec)\n\n    if not isinstance(header, dict):\n        raise ValueError("camParam v2 requires one header record")\n    if not frames:\n        raise ValueError("camParam v2 contains no frame records")\n\n    required_header = (\n        "width",\n        "height",\n        "depth_scale",\n        "depth_scale_precision",\n        "intrinsic",\n        "pose_mode",\n    )\n    for key in required_header:\n        if key not in header:\n            raise KeyError(f"camParam header missing \'{key}\'")\n\n    return {"header": header, "frames": frames}\n\n\ndef rodrigues_to_matrix(rvec):\n    r = np.asarray(rvec, dtype=np.float64).reshape(3)\n    theta = float(np.linalg.norm(r))\n\n    if theta < 1e-12:\n        x, y, z = r\n        Kx = np.array(\n            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],\n            dtype=np.float64,\n        )\n        return np.eye(3, dtype=np.float64) + Kx\n\n    axis = r / theta\n    x, y, z = axis\n    Kx = np.array(\n        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],\n        dtype=np.float64,\n    )\n    return (\n        np.eye(3, dtype=np.float64)\n        + math.sin(theta) * Kx\n        + (1.0 - math.cos(theta)) * (Kx @ Kx)\n    )\n\n\ndef rt_to_4x4(rvec, tvec):\n    T = np.eye(4, dtype=np.float64)\n    T[:3, :3] = rodrigues_to_matrix(rvec)\n    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)\n    return T\n\n\ndef intrinsic_vec_to_matrix(v):\n    fx, fy, cx, cy = [float(x) for x in v]\n    if not np.isfinite([fx, fy, cx, cy]).all():\n        raise ValueError(f"non-finite intrinsic: {v}")\n    if fx <= 0.0 or fy <= 0.0:\n        raise ValueError(f"invalid focal length: fx={fx}, fy={fy}")\n    return np.array(\n        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],\n        dtype=np.float64,\n    )\n\n\ndef get_depth_scale_real_from_header(header):\n    precision = float(header["depth_scale_precision"])\n    if precision <= 0.0:\n        raise ValueError("depth_scale_precision must be positive")\n\n    scale = float(header["depth_scale"]) / precision\n    if not np.isfinite(scale) or scale <= 0.0:\n        raise ValueError(f"invalid depth_scale_real: {scale}")\n    return scale\n\n\ndef build_camera_lookup(camera_json):\n    """Reconstruct K, W2C and C2W for every frame."""\n    header = camera_json["header"]\n    records = sorted(camera_json["frames"], key=lambda r: int(r["poc"]))\n\n    intr0 = header["intrinsic"]\n    base_intr = np.array(\n        [intr0["fx"], intr0["fy"], intr0["cx"], intr0["cy"]],\n        dtype=np.float64,\n    )\n    z_sign = 1.0 if float(intr0.get("z_sign", 1.0)) > 0.0 else -1.0\n    pose_mode = str(header["pose_mode"])\n    depth_scale_real = get_depth_scale_real_from_header(header)\n\n    fixed_intrinsic = (\n        header.get("intrinsic_mode") == "rap_fixed"\n        or header.get("intrinsic_delta_mode") == "fixed_zero_delta"\n    )\n\n    by_poc = {}\n    by_frame_idx = {}\n    cur_intr = base_intr.copy()\n    prev_w2c = np.eye(4, dtype=np.float64)\n\n    for order, rec in enumerate(records):\n        if "poc" not in rec or "rvec" not in rec or "tvec" not in rec:\n            raise KeyError(\n                "each camParam frame record requires poc, rvec and tvec"\n            )\n\n        poc = int(rec["poc"])\n        frame_idx = int(rec.get("frame_idx", poc))\n\n        delta = np.asarray(\n            rec.get("intrinsic_delta", [0.0, 0.0, 0.0, 0.0]),\n            dtype=np.float64,\n        ).reshape(4)\n\n        if fixed_intrinsic:\n            cur_intr = base_intr.copy()\n        elif order == 0:\n            cur_intr = base_intr.copy()\n        else:\n            cur_intr = cur_intr + delta\n\n        K = intrinsic_vec_to_matrix(cur_intr)\n        T_rec = rt_to_4x4(rec["rvec"], rec["tvec"])\n\n        if pose_mode == "current_to_previous":\n            if order == 0:\n                w2c = np.eye(4, dtype=np.float64)\n            else:\n                try:\n                    w2c = np.linalg.inv(T_rec) @ prev_w2c\n                except np.linalg.LinAlgError as exc:\n                    raise ValueError(\n                        f"singular current_to_previous transform at POC {poc}"\n                    ) from exc\n        elif pose_mode in ("gop_local", "absolute"):\n            w2c = T_rec\n        else:\n            raise ValueError(f"unsupported pose_mode: {pose_mode}")\n\n        try:\n            c2w = np.linalg.inv(w2c)\n        except np.linalg.LinAlgError as exc:\n            raise ValueError(f"singular W2C at POC {poc}") from exc\n\n        cam = {\n            "poc": poc,\n            "frame_idx": frame_idx,\n            "K": K,\n            "W2C": w2c,\n            "C2W": c2w,\n            "z_sign": z_sign,\n            "depth_scale_real": depth_scale_real,\n        }\n\n        by_poc[poc] = cam\n        by_frame_idx.setdefault(frame_idx, cam)\n        prev_w2c = w2c\n\n    return {\n        "header": header,\n        "by_poc": by_poc,\n        "by_frame_idx": by_frame_idx,\n    }\n\n\ndef get_camera(camera_lookup, frame_idx):\n    if frame_idx in camera_lookup["by_poc"]:\n        return camera_lookup["by_poc"][frame_idx]\n    if frame_idx in camera_lookup["by_frame_idx"]:\n        return camera_lookup["by_frame_idx"][frame_idx]\n    raise KeyError(f"camera for frame/POC {frame_idx} not found")\n\n\ndef camera_has_required_mats(cam):\n    try:\n        K = np.asarray(cam["K"], dtype=np.float64)\n        W2C = np.asarray(cam["W2C"], dtype=np.float64)\n        C2W = np.asarray(cam["C2W"], dtype=np.float64)\n        scale = float(cam["depth_scale_real"])\n        z_sign = float(cam["z_sign"])\n        return (\n            K.shape == (3, 3)\n            and W2C.shape == (4, 4)\n            and C2W.shape == (4, 4)\n            and np.isfinite(K).all()\n            and np.isfinite(W2C).all()\n            and np.isfinite(C2W).all()\n            and np.isfinite(scale)\n            and scale > 0.0\n            and z_sign in (-1.0, 1.0)\n        )\n    except Exception:\n        return False\n\n\ndef get_depth_scale_real(cam):\n    return float(cam["depth_scale_real"])\n\n\n# ============================================================\n# Camera geometry for camParam v2\n# ============================================================\n\ndef pixel_rays_camera(u, v, cam):\n    """Return camera rays whose signed optical-axis depth has magnitude 1."""\n    K = np.asarray(cam["K"], dtype=np.float64)\n    z_sign = float(cam["z_sign"])\n\n    u = np.asarray(u, dtype=np.float64)\n    v = np.asarray(v, dtype=np.float64)\n\n    rx = (u - K[0, 2]) / K[0, 0]\n    ry = (v - K[1, 2]) / K[1, 1]\n    rz = np.full_like(rx, z_sign)\n\n    return np.stack([rx, ry, rz], axis=-1)\n\n\ndef project_camera_points(points_cam, cam):\n    p = np.asarray(points_cam, dtype=np.float64)\n    K = np.asarray(cam["K"], dtype=np.float64)\n    z_sign = float(cam["z_sign"])\n\n    depth = z_sign * p[..., 2]\n    front = np.isfinite(depth) & (depth > 1e-12)\n    safe_depth = np.where(front, depth, 1.0)\n\n    u = K[0, 0] * (p[..., 0] / safe_depth) + K[0, 2]\n    v = K[1, 1] * (p[..., 1] / safe_depth) + K[1, 2]\n\n    return u, v, depth, front\n\n\ndef fit_3d_plane(points):\n    points = np.asarray(points, dtype=np.float64)\n    if points.shape[0] < 3:\n        return None\n\n    center = np.mean(points, axis=0)\n    q = points - center\n\n    try:\n        _, s, vh = np.linalg.svd(q, full_matrices=False)\n    except np.linalg.LinAlgError:\n        return None\n\n    if len(s) < 2 or s[1] < 1e-9:\n        return None\n\n    n = vh[-1]\n    norm = float(np.linalg.norm(n))\n    if norm < 1e-12:\n        return None\n\n    n = n / norm\n    d = -float(np.dot(n, center))\n    return np.array([n[0], n[1], n[2], d], dtype=np.float64)\n\n\ndef transform_plane_src_to_tgt(plane_src, cam_src, cam_tgt):\n    M = (\n        np.asarray(cam_tgt["W2C"], dtype=np.float64)\n        @ np.asarray(cam_src["C2W"], dtype=np.float64)\n    )\n\n    try:\n        plane_tgt = np.linalg.inv(M).T @ np.asarray(\n            plane_src, dtype=np.float64\n        )\n    except np.linalg.LinAlgError:\n        return None\n\n    norm = float(np.linalg.norm(plane_tgt[:3]))\n    if norm < 1e-12:\n        return None\n    return plane_tgt / norm\n\n\ndef image_inv_plane_to_3d_plane(leaf, cam, frame_w, frame_h, args):\n    del frame_w, frame_h\n\n    ns = max(2, int(args.plane_warp_samples))\n    xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)\n    ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)\n    uu, vv = np.meshgrid(xs, ys)\n\n    depth_y = inv_plane_to_depth_value(leaf.plane, uu, vv, args)\n    depth_real = depth_y * get_depth_scale_real(cam)\n\n    rays = pixel_rays_camera(uu, vv, cam)\n    points = rays.reshape(-1, 3) * depth_real.reshape(-1, 1)\n\n    valid = (\n        np.isfinite(points).all(axis=1)\n        & np.isfinite(depth_real.reshape(-1))\n        & (depth_real.reshape(-1) > 0.0)\n    )\n    points = points[valid]\n\n    if points.shape[0] < 3:\n        return None\n    return fit_3d_plane(points)\n\n\ndef render_3d_plane_to_depth_block(\n    plane_cam,\n    cam_cur,\n    x,\n    y,\n    w,\n    h,\n    frame_w,\n    frame_h,\n    args,\n):\n    del frame_w, frame_h\n\n    gx = np.arange(x, x + w, dtype=np.float64)\n    gy = np.arange(y, y + h, dtype=np.float64)\n    uu, vv = np.meshgrid(gx, gy)\n    rays = pixel_rays_camera(uu, vv, cam_cur)\n\n    n = np.asarray(plane_cam[:3], dtype=np.float64)\n    d = float(plane_cam[3])\n\n    denom = (\n        n[0] * rays[..., 0]\n        + n[1] * rays[..., 1]\n        + n[2] * rays[..., 2]\n    )\n\n    valid = np.abs(denom) > 1e-12\n    depth_real = np.full((h, w), np.nan, dtype=np.float64)\n    depth_real[valid] = -d / denom[valid]\n\n    valid = valid & np.isfinite(depth_real) & (depth_real > 0.0)\n    valid_ratio = float(np.mean(valid))\n\n    if valid_ratio < args.plane_warp_min_valid_ratio:\n        return None\n\n    depth_y = depth_real / get_depth_scale_real(cam_cur)\n    depth_y = np.clip(depth_y, args.depth_eps, args.max_value)\n\n    if not np.all(valid):\n        fill = (\n            float(np.median(depth_y[valid]))\n            if np.any(valid)\n            else float(args.max_value)\n        )\n        depth_y[~valid] = fill\n\n    return depth_y\n\n\n'
BACKWARD_MAP_REPLACEMENT = 'def make_backward_map_cur_to_prev(\n    depth_y_cur,\n    cam_cur,\n    cam_prev,\n    width,\n    height,\n):\n    yy, xx = np.meshgrid(\n        np.arange(height, dtype=np.float64),\n        np.arange(width, dtype=np.float64),\n        indexing="ij",\n    )\n\n    rays = pixel_rays_camera(xx, yy, cam_cur)\n    depth_real = (\n        np.asarray(depth_y_cur, dtype=np.float64)\n        * get_depth_scale_real(cam_cur)\n    )\n    points_cur = rays * depth_real[..., None]\n\n    M = (\n        np.asarray(cam_prev["W2C"], dtype=np.float64)\n        @ np.asarray(cam_cur["C2W"], dtype=np.float64)\n    )\n    points_flat = points_cur.reshape(-1, 3)\n    points_h = np.concatenate(\n        [\n            points_flat,\n            np.ones((points_flat.shape[0], 1), dtype=np.float64),\n        ],\n        axis=1,\n    )\n    points_prev = (points_h @ M.T)[:, :3].reshape(height, width, 3)\n\n    map_x, map_y, _, front = project_camera_points(points_prev, cam_prev)\n\n    valid = (\n        front\n        & np.isfinite(map_x)\n        & np.isfinite(map_y)\n        & np.isfinite(depth_real)\n        & (depth_real > 0.0)\n        & (map_x >= 0.0)\n        & (map_y >= 0.0)\n        & (map_x <= width - 1.0)\n        & (map_y <= height - 1.0)\n    )\n\n    return map_x, map_y, valid\n\n\n'


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_pos = text.find(start)
    if start_pos < 0:
        raise RuntimeError(f"start marker not found: {start!r}")
    end_pos = text.find(end, start_pos)
    if end_pos < 0:
        raise RuntimeError(f"end marker not found: {end!r}")
    return text[:start_pos] + replacement + text[end_pos:]


def patch_source(source: str) -> str:
    camera_marker = (
        "# ============================================================\n"
        "# Camera JSON / JSONL / matrices\n"
        "# ============================================================\n"
    )
    source = replace_between(
        source,
        camera_marker,
        "def make_plane_warp_candidate(",
        CAMERA_AND_GEOMETRY_REPLACEMENT,
    )

    source = replace_between(
        source,
        "def make_backward_map_cur_to_prev(",
        "def downsample_map_for_chroma(",
        BACKWARD_MAP_REPLACEMENT,
    )

    old_lookup = "    camera_lookup = build_camera_lookup(cam_json)\n"
    new_lookup = (
        "    camera_lookup = build_camera_lookup(cam_json)\n"
        "    cam_header = camera_lookup[\"header\"]\n"
        "    if int(cam_header[\"width\"]) != args.width or int(cam_header[\"height\"]) != args.height:\n"
        "        raise ValueError(\n"
        "            f\"camera/depth resolution mismatch: \"\n"
        "            f\"cam={cam_header['width']}x{cam_header['height']}, \"\n"
        "            f\"input={args.width}x{args.height}\"\n"
        "        )\n"
    )
    if old_lookup not in source:
        raise RuntimeError("camera lookup insertion point not found")
    source = source.replace(old_lookup, new_lookup, 1)

    source = source.replace(
        "# where Y is stored depth sample, e.g. Z_real = Y * near.",
        "# where real camera depth = Y * depth_scale_real from camParam JSONL.",
        1,
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")

    source = input_path.read_text(encoding="utf-8")
    patched = patch_source(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")
    py_compile.compile(str(output_path), doraise=True)
    print(f"Patched simulator written to: {output_path}")


if __name__ == "__main__":
    main()
