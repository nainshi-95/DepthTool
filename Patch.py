#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply adaptive temporal sampling / exact 3D plane conversion /
candidate deduplication patch to the original simulator.

Usage:
    python apply_adaptive_temporal_patch.py \
        --input projection_satd_depth_sim.py \
        --output projection_satd_depth_sim_v2.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PATCH_TEXT = r"""*** Begin Patch
*** Update File: projection_satd_depth_sim.py
@@
-  * five-point temporal plane resampling for L0/L1
+  * adaptive multi-point temporal plane resampling for L0/L1
+  * exact inverse-depth-image-plane -> 3D-plane conversion
+  * block-domain duplicate candidate pruning
@@
-# Camera geometry / five-point temporal resampling
+# Camera geometry / adaptive multi-point temporal resampling
@@
 def fit_3d_plane(points: np.ndarray) -> Optional[np.ndarray]:
@@
     return np.array([n[0], n[1], n[2], -float(np.dot(n, center))], dtype=np.float64)


-def image_inv_plane_to_3d_plane(
+def image_inv_plane_to_3d_plane_sampled(
     leaf: LeafRecord,
     cam: Dict[str, Any],
     args: argparse.Namespace,
 ) -> Optional[np.ndarray]:
+    """Legacy sampled conversion retained as an optional fallback."""
     ns = max(2, int(args.plane_warp_samples))
     xs = np.linspace(leaf.x, leaf.x + leaf.w - 1, ns, dtype=np.float64)
     ys = np.linspace(leaf.y, leaf.y + leaf.h - 1, ns, dtype=np.float64)
@@
     valid = np.isfinite(points).all(axis=1) & (depth_real.reshape(-1) > 0.0)
     return fit_3d_plane(points[valid])


+def image_inv_plane_to_3d_plane_direct(
+    leaf: LeafRecord,
+    cam: Dict[str, Any],
+) -> Optional[np.ndarray]:
+    """Convert an image inverse-depth plane to a camera-space 3D plane exactly.
+
+    The coded image plane is
+
+        invY(u,v) = a (u-pcx) + b (v-pcy) + c
+
+    and real camera depth is
+
+        d = depth_scale_real / invY.
+
+    With the ray convention used by this script,
+
+        X = ((u-cx)/fx) d
+        Y = ((v-cy)/fy) d
+        Z = z_sign d
+
+    the corresponding camera-space plane is derived analytically, without
+    sampling or SVD fitting.
+    """
+    p = leaf.plane
+    k = np.asarray(cam["K"], dtype=np.float64)
+    fx = float(k[0, 0])
+    fy = float(k[1, 1])
+    cam_cx = float(k[0, 2])
+    cam_cy = float(k[1, 2])
+    z_sign = float(cam["z_sign"])
+    depth_scale_real = get_depth_scale_real(cam)
+
+    # invY(u,v) = alpha*u + beta*v + gamma
+    alpha = float(p.a)
+    beta = float(p.b)
+    gamma = float(p.c - p.a * p.cx - p.b * p.cy)
+
+    plane = np.array(
+        [
+            alpha * fx,
+            beta * fy,
+            z_sign * (alpha * cam_cx + beta * cam_cy + gamma),
+            -depth_scale_real,
+        ],
+        dtype=np.float64,
+    )
+    if not np.isfinite(plane).all():
+        return None
+    norm = float(np.linalg.norm(plane[:3]))
+    if norm < 1e-15:
+        return None
+    return plane / norm
+
+
+def image_inv_plane_to_3d_plane(
+    leaf: LeafRecord,
+    cam: Dict[str, Any],
+    args: argparse.Namespace,
+) -> Optional[np.ndarray]:
+    if args.reference_plane_3d == "sampled":
+        return image_inv_plane_to_3d_plane_sampled(leaf, cam, args)
+
+    direct = image_inv_plane_to_3d_plane_direct(leaf, cam)
+    if direct is not None:
+        return direct
+
+    # Defensive fallback for malformed or numerically degenerate inputs.
+    return image_inv_plane_to_3d_plane_sampled(leaf, cam, args)
+
+
@@
-def block_five_sample_points(x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
-    # Pixel-center locations near the four corners plus center.
-    xs = np.array(
-        [
-            float(x),
-            float(x + w - 1),
-            float(x),
-            float(x + w - 1),
-            x + (w - 1) / 2.0,
-        ],
-        dtype=np.float64,
-    )
-    ys = np.array(
-        [
-            float(y),
-            float(y),
-            float(y + h - 1),
-            float(y + h - 1),
-            y + (h - 1) / 2.0,
-        ],
-        dtype=np.float64,
-    )
-    return xs, ys
+def unique_sample_points(
+    points: Sequence[Tuple[float, float]],
+) -> Tuple[np.ndarray, np.ndarray]:
+    seen = set()
+    out: List[Tuple[float, float]] = []
+    for px, py in points:
+        key = (round(float(px), 9), round(float(py), 9))
+        if key not in seen:
+            seen.add(key)
+            out.append((float(px), float(py)))
+    return (
+        np.asarray([p[0] for p in out], dtype=np.float64),
+        np.asarray([p[1] for p in out], dtype=np.float64),
+    )
+
+
+def temporal_sample_level_points(
+    x: int,
+    y: int,
+    w: int,
+    h: int,
+    level: int,
+    final_grid: int,
+) -> Tuple[np.ndarray, np.ndarray]:
+    """Return cumulative block samples for one adaptive refinement level.
+
+    level 0: four corners + center
+    level 1: 3x3 grid, therefore adding edge midpoints
+    level 2: configurable dense grid
+    """
+    x0 = float(x)
+    x1 = float(x + w - 1)
+    y0 = float(y)
+    y1 = float(y + h - 1)
+    xc = 0.5 * (x0 + x1)
+    yc = 0.5 * (y0 + y1)
+
+    points: List[Tuple[float, float]] = [
+        (x0, y0), (x1, y0), (x0, y1), (x1, y1), (xc, yc)
+    ]
+    if level >= 1:
+        xs = np.linspace(x0, x1, 3, dtype=np.float64)
+        ys = np.linspace(y0, y1, 3, dtype=np.float64)
+        points.extend((float(px), float(py)) for py in ys for px in xs)
+    if level >= 2:
+        n = max(3, int(final_grid))
+        xs = np.linspace(x0, x1, n, dtype=np.float64)
+        ys = np.linspace(y0, y1, n, dtype=np.float64)
+        points.extend((float(px), float(py)) for py in ys for px in xs)
+    return unique_sample_points(points)
@@
 def sample_visible_depths_from_projected_leaves(
@@
     return depth_y, valid


-def make_five_point_temporal_candidates(
+def fit_inv_plane_from_samples_with_quality(
+    sample_x: np.ndarray,
+    sample_y: np.ndarray,
+    depth_y: np.ndarray,
+    valid: np.ndarray,
+    cx: float,
+    cy: float,
+    args: argparse.Namespace,
+) -> Tuple[Optional[Plane], float, float]:
+    valid2 = (
+        np.asarray(valid, dtype=bool)
+        & np.isfinite(sample_x)
+        & np.isfinite(sample_y)
+        & np.isfinite(depth_y)
+        & (depth_y >= args.depth_eps)
+        & (depth_y <= args.max_value)
+    )
+    valid_ratio = float(np.mean(valid2)) if valid2.size else 0.0
+    plane = fit_inv_plane_from_samples(
+        sample_x, sample_y, depth_y, valid2, cx, cy, args
+    )
+    if plane is None:
+        return None, float("inf"), valid_ratio
+
+    inv_gt = 1.0 / np.clip(
+        depth_y[valid2], args.depth_eps, args.max_value
+    )
+    inv_pred = eval_inv_plane_value(
+        plane, sample_x[valid2], sample_y[valid2]
+    )
+    rmse = math.sqrt(float(np.mean((inv_pred - inv_gt) ** 2)))
+    return plane, rmse, valid_ratio
+
+
+def adaptive_temporal_samples(
+    ctx: PlaneWarpContext,
+    x: int,
+    y: int,
+    w: int,
+    h: int,
+    cx: float,
+    cy: float,
+    args: argparse.Namespace,
+) -> Dict[str, Any]:
+    rec0 = get_projected_leaf_cache(ctx, "l0", args)
+    rec1 = (
+        get_projected_leaf_cache(ctx, "l1", args)
+        if ctx.l1_store is not None and ctx.cam_l1_low is not None
+        else []
+    )
+
+    max_level = max(0, min(int(args.temporal_adaptive_levels), 2))
+    result: Dict[str, Any] = {}
+    for level in range(max_level + 1):
+        sx, sy = temporal_sample_level_points(
+            x, y, w, h, level, args.temporal_adaptive_grid
+        )
+        d0, v0 = sample_visible_depths_from_projected_leaves(
+            rec0, ctx.cam_cur_low, sx, sy, args
+        )
+        p0, e0, r0 = fit_inv_plane_from_samples_with_quality(
+            sx, sy, d0, v0, cx, cy, args
+        )
+
+        d1 = np.full_like(d0, np.nan)
+        v1 = np.zeros_like(v0)
+        p1: Optional[Plane] = None
+        e1 = float("inf")
+        r1 = 0.0
+        if rec1:
+            d1, v1 = sample_visible_depths_from_projected_leaves(
+                rec1, ctx.cam_cur_low, sx, sy, args
+            )
+            p1, e1, r1 = fit_inv_plane_from_samples_with_quality(
+                sx, sy, d1, v1, cx, cy, args
+            )
+
+        result = {
+            "sx": sx, "sy": sy,
+            "d0": d0, "v0": v0, "p0": p0, "rmse0": e0, "ratio0": r0,
+            "d1": d1, "v1": v1, "p1": p1, "rmse1": e1, "ratio1": r1,
+            "level": level,
+        }
+
+        # Refinement is requested when coverage is poor or the current plane
+        # does not explain the sampled inverse depths sufficiently well.
+        side0_good = (
+            p0 is not None
+            and r0 >= args.temporal_adaptive_min_valid_ratio
+            and (
+                args.temporal_adaptive_target_inv_rmse <= 0.0
+                or e0 <= args.temporal_adaptive_target_inv_rmse
+            )
+        )
+        side1_good = (
+            not rec1
+            or (
+                p1 is not None
+                and r1 >= args.temporal_adaptive_min_valid_ratio
+                and (
+                    args.temporal_adaptive_target_inv_rmse <= 0.0
+                    or e1 <= args.temporal_adaptive_target_inv_rmse
+                )
+            )
+        )
+        if side0_good and side1_good:
+            break
+
+    return result
+
+
+def make_adaptive_temporal_candidates(
     ctx: Optional[PlaneWarpContext],
@@
 ) -> List[Tuple[str, Plane]]:
     if ctx is None:
         return []

-    sx, sy = block_five_sample_points(x, y, w, h)
-    rec0 = get_projected_leaf_cache(ctx, "l0", args)
-    d0, v0 = sample_visible_depths_from_projected_leaves(
-        rec0, ctx.cam_cur_low, sx, sy, args
-    )
-    p0 = fit_inv_plane_from_samples(sx, sy, d0, v0, cx, cy, args)
-
-    p1 = None
-    d1 = np.full_like(d0, np.nan)
-    v1 = np.zeros_like(v0)
-    if ctx.l1_store is not None and ctx.cam_l1_low is not None:
-        rec1 = get_projected_leaf_cache(ctx, "l1", args)
-        d1, v1 = sample_visible_depths_from_projected_leaves(
-            rec1, ctx.cam_cur_low, sx, sy, args
-        )
-        p1 = fit_inv_plane_from_samples(sx, sy, d1, v1, cx, cy, args)
+    sampled = adaptive_temporal_samples(ctx, x, y, w, h, cx, cy, args)
+    sx = sampled["sx"]
+    sy = sampled["sy"]
+    d0 = sampled["d0"]
+    v0 = sampled["v0"]
+    p0 = sampled["p0"]
+    d1 = sampled["d1"]
+    v1 = sampled["v1"]
+    p1 = sampled["p1"]
@@
     return out


+def candidate_comparison_points(
+    x: int,
+    y: int,
+    w: int,
+    h: int,
+    grid_size: int,
+) -> Tuple[np.ndarray, np.ndarray]:
+    n = max(2, int(grid_size))
+    xs = np.linspace(x, x + w - 1, n, dtype=np.float64)
+    ys = np.linspace(y, y + h - 1, n, dtype=np.float64)
+    xx, yy = np.meshgrid(xs, ys)
+    return xx.reshape(-1), yy.reshape(-1)
+
+
+def candidate_planes_are_similar(
+    p0: Plane,
+    p1: Plane,
+    sample_x: np.ndarray,
+    sample_y: np.ndarray,
+    args: argparse.Namespace,
+) -> bool:
+    d0 = inv_plane_to_depth_value(p0, sample_x, sample_y, args)
+    d1 = inv_plane_to_depth_value(p1, sample_x, sample_y, args)
+    if not np.isfinite(d0).all() or not np.isfinite(d1).all():
+        return False
+
+    abs_diff = np.abs(d0 - d1)
+    rel_diff = abs_diff / np.maximum(np.maximum(np.abs(d0), np.abs(d1)), 1.0)
+    return (
+        float(np.max(abs_diff)) <= args.candidate_dedup_max_abs_depth
+        and float(np.mean(rel_diff)) <= args.candidate_dedup_mean_rel_depth
+    )
+
+
+def remove_duplicate_candidates(
+    candidates: Sequence[Tuple[str, Plane]],
+    x: int,
+    y: int,
+    w: int,
+    h: int,
+    args: argparse.Namespace,
+) -> List[Tuple[str, Plane]]:
+    if not args.candidate_dedup or len(candidates) <= 1:
+        return list(candidates)
+
+    sx, sy = candidate_comparison_points(
+        x, y, w, h, args.candidate_dedup_grid
+    )
+    kept: List[Tuple[str, Plane]] = []
+    for name, plane in candidates:
+        duplicate = any(
+            candidate_planes_are_similar(plane, kept_plane, sx, sy, args)
+            for _, kept_plane in kept
+        )
+        if not duplicate:
+            kept.append((name, plane))
+    return kept
+
+
 def make_candidates(
@@
     converted: Dict[str, Plane] = {}

     if args.plane_warp_candidate:
-        for name, p in make_five_point_temporal_candidates(
+        for name, p in make_adaptive_temporal_candidates(
             plane_warp_ctx, x, y, w, h, cx, cy, args
         ):
@@
-    return candidates[: args.max_candidates]
+    # Ordering is intentional: temporal candidates keep priority when a
+    # spatial candidate predicts almost the same depth surface.
+    candidates = remove_duplicate_candidates(candidates, x, y, w, h, args)
+    return candidates[: args.max_candidates]
@@
     p.add_argument("--plane-warp-samples", type=int, default=5)
+    p.add_argument(
+        "--reference-plane-3d",
+        choices=["direct", "sampled"],
+        default="direct",
+        help="direct uses the exact inverse-depth-plane to 3D-plane formula",
+    )
+    p.add_argument(
+        "--temporal-adaptive-levels",
+        type=int,
+        default=2,
+        help="0: five points, 1: add 3x3 samples, 2: add dense grid",
+    )
+    p.add_argument(
+        "--temporal-adaptive-grid",
+        type=int,
+        default=5,
+        help="final dense sampling grid size for adaptive level 2",
+    )
+    p.add_argument(
+        "--temporal-adaptive-min-valid-ratio",
+        type=float,
+        default=0.70,
+        help="refine sampling while valid projected coverage is below this",
+    )
+    p.add_argument(
+        "--temporal-adaptive-target-inv-rmse",
+        type=float,
+        default=0.0,
+        help="refine while inverse-depth fitting RMSE exceeds this; 0 disables",
+    )
     p.add_argument(
         "--temporal-sample-max-inv-rmse",
@@
         help="reject five-point candidate if inverse-depth fit RMSE exceeds this; 0 disables",
     )
+    p.add_argument(
+        "--candidate-dedup",
+        dest="candidate_dedup",
+        action="store_true",
+        help="remove candidates producing nearly identical block depths",
+    )
+    p.add_argument(
+        "--no-candidate-dedup",
+        dest="candidate_dedup",
+        action="store_false",
+    )
+    p.set_defaults(candidate_dedup=True)
+    p.add_argument("--candidate-dedup-grid", type=int, default=3)
+    p.add_argument(
+        "--candidate-dedup-max-abs-depth",
+        type=float,
+        default=2.0,
+        help="maximum depth-code difference at all comparison samples",
+    )
+    p.add_argument(
+        "--candidate-dedup-mean-rel-depth",
+        type=float,
+        default=0.002,
+        help="maximum mean relative depth difference",
+    )
@@
     if args.block_size <= 0 or args.max_qt_depth < 0:
         raise ValueError("invalid block configuration")
+    if not 0 <= args.temporal_adaptive_levels <= 2:
+        raise ValueError("--temporal-adaptive-levels must be in [0,2]")
+    if args.temporal_adaptive_grid < 3:
+        raise ValueError("--temporal-adaptive-grid must be at least 3")
+    if not 0.0 <= args.temporal_adaptive_min_valid_ratio <= 1.0:
+        raise ValueError("--temporal-adaptive-min-valid-ratio must be in [0,1]")
+    if args.candidate_dedup_grid < 2:
+        raise ValueError("--candidate-dedup-grid must be at least 2")
+    if args.candidate_dedup_max_abs_depth < 0.0:
+        raise ValueError("--candidate-dedup-max-abs-depth must be nonnegative")
+    if args.candidate_dedup_mean_rel_depth < 0.0:
+        raise ValueError("--candidate-dedup-mean-rel-depth must be nonnegative")
@@
-        "temporal_candidate": "five_target_points_nearest_visible_projected_leaf_refit",
+        "temporal_candidate": "adaptive_multi_point_nearest_visible_projected_leaf_refit",
+        "reference_plane_3d_conversion": args.reference_plane_3d,
+        "candidate_deduplication": {
+            "enabled": args.candidate_dedup,
+            "grid": args.candidate_dedup_grid,
+            "max_abs_depth": args.candidate_dedup_max_abs_depth,
+            "mean_rel_depth": args.candidate_dedup_mean_rel_depth,
+        },
*** End Patch
"""


def run_command(cmd, cwd):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def apply_patch(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        work_file = tmp_dir / "projection_satd_depth_sim.py"
        patch_file = tmp_dir / "change.patch"

        shutil.copy2(input_path, work_file)
        patch_file.write_text(PATCH_TEXT, encoding="utf-8")

        applied = False
        errors = []

        patch_exe = shutil.which("patch")
        if patch_exe:
            code, stdout, stderr = run_command(
                [patch_exe, "-p0", "-i", str(patch_file)],
                cwd=tmp_dir,
            )
            if code == 0:
                applied = True
            else:
                errors.append(
                    "patch command failed:\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )

        if not applied:
            git_exe = shutil.which("git")
            if git_exe:
                code, stdout, stderr = run_command(
                    [git_exe, "apply", "--unsafe-paths", str(patch_file)],
                    cwd=tmp_dir,
                )
                if code == 0:
                    applied = True
                else:
                    errors.append(
                        "git apply failed:\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}"
                    )

        if not applied:
            raise RuntimeError(
                "Could not apply the patch. Install either 'patch' or 'git', "
                "and verify that the input file exactly matches the supplied base code.\n\n"
                + "\n\n".join(errors)
            )

        shutil.copy2(work_file, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="original Python file")
    parser.add_argument("--output", required=True, help="patched output Python file")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if input_path == output_path:
        raise ValueError("input and output paths must be different")

    apply_patch(input_path, output_path)
    print(f"Patched file written to: {output_path}")


if __name__ == "__main__":
    main()
