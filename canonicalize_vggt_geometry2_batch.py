#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py

It recursively finds VGGT-Omega runner outputs:

    xxx_vggt_omega_outputs.npz
    xxx_camera.jsonl              optional, auto-detected

and runs:

    optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py
        --npz xxx_vggt_omega_outputs.npz
        --camera-jsonl xxx_camera.jsonl
        --out-prefix DST/relative/path/xxx
        --overwrite

Example:

python batch_optimize_vggt_npz.py \
  --src-root out/vggt_raw \
  --dst-root out/fixedK_depth \
  --optimizer-script optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py \
  --force \
  -- \
  --stage1-iters 300 \
  --stage2-iters 300 \
  --stage3-iters 150 \
  --stage4-iters 200 \
  --sample-stride 8 \
  --max-samples-per-pair 60000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


NPZ_SUFFIX = "_vggt_omega_outputs.npz"
CAMERA_SUFFIX = "_camera.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def validate_vggt_npz(npz_path: Path) -> bool:
    required = {"depth_original", "extrinsic", "intrinsic_original"}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = set(data.files)
        missing = required - keys
        if missing:
            log(f"SKIP invalid NPZ: {npz_path} / missing {sorted(missing)}")
            return False
        return True
    except Exception as e:
        log(f"SKIP unreadable NPZ: {npz_path} / {e}")
        return False


def find_vggt_npz_files(src_root: Path, pattern: str) -> list[Path]:
    files = sorted(src_root.rglob(pattern))
    return [p for p in files if p.is_file()]


def derive_base_name(npz_path: Path) -> str:
    name = npz_path.name
    if name.endswith(NPZ_SUFFIX):
        return name[: -len(NPZ_SUFFIX)]
    return npz_path.stem


def find_camera_jsonl(npz_path: Path) -> Path | None:
    base = derive_base_name(npz_path)
    candidate = npz_path.with_name(base + CAMERA_SUFFIX)
    return candidate if candidate.is_file() else None


def make_out_prefix(
    npz_path: Path,
    src_root: Path,
    dst_root: Path,
    layout: str,
) -> Path:
    base = derive_base_name(npz_path)

    if layout == "preserve":
        rel_dir = npz_path.parent.relative_to(src_root)
        out_dir = dst_root / rel_dir
    elif layout == "flat":
        out_dir = dst_root
    else:
        raise ValueError(layout)

    return out_dir / base


def expected_outputs(out_prefix: Path) -> list[Path]:
    name = out_prefix.name
    parent = out_prefix.parent
    return [
        parent / f"{name}_fixedK_gop_nn_geometry.npz",
        parent / f"{name}_fixedK_gop_nn_cam.jsonl",
        parent / f"{name}_fixedK_gop_nn_depth_linear_yuv420p10le.yuv",
        parent / f"{name}_fixedK_gop_nn_manifest.json",
    ]


def already_done(out_prefix: Path) -> bool:
    # manifest 기준으로 완료 여부 판단
    manifest = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_manifest.json")
    return manifest.is_file()


def run_one(
    npz_path: Path,
    camera_jsonl: Path | None,
    out_prefix: Path,
    optimizer_script: Path,
    python_exe: str,
    optimizer_args: list[str],
    force: bool,
    dry_run: bool,
) -> int:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(optimizer_script),
        "--npz",
        str(npz_path),
        "--out-prefix",
        str(out_prefix),
    ]

    if camera_jsonl is not None:
        cmd += ["--camera-jsonl", str(camera_jsonl)]

    if force:
        cmd += ["--overwrite"]

    cmd += optimizer_args

    log("RUN:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)

    if dry_run:
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Recursively batch-run fixed-K GOP NN-depth optimizer on VGGT-Omega NPZ outputs."
    )

    p.add_argument("--src-root", required=True, help="Folder containing first-script outputs")
    p.add_argument("--dst-root", required=True, help="Folder to save optimized outputs")
    p.add_argument(
        "--optimizer-script",
        required=True,
        help="Path to optimize_fixedK_rt_depth_nn_gop_smooth_predloss.py",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run optimizer script. Default: current Python",
    )
    p.add_argument(
        "--pattern",
        default=f"*{NPZ_SUFFIX}",
        help=f"NPZ search pattern. Default: *{NPZ_SUFFIX}",
    )
    p.add_argument(
        "--layout",
        choices=["preserve", "flat"],
        default="preserve",
        help="preserve: keep relative folders under dst-root, flat: save all outputs directly in dst-root",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Pass --overwrite to optimizer and rerun even if output exists",
    )
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Open NPZ and skip files that do not contain required VGGT keys",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining NPZ files even if one fails",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )

    args, extra = p.parse_known_args()

    # Allows:
    #   batch.py ... -- --stage1-iters 100 ...
    if extra and extra[0] == "--":
        extra = extra[1:]

    return args, extra


def main() -> None:
    args, optimizer_args = parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    optimizer_script = Path(args.optimizer_script).resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"src-root not found: {src_root}")

    if not optimizer_script.is_file():
        raise FileNotFoundError(f"optimizer-script not found: {optimizer_script}")

    npz_files = find_vggt_npz_files(src_root, args.pattern)

    log(f"Source root : {src_root}")
    log(f"Dest root   : {dst_root}")
    log(f"Optimizer   : {optimizer_script}")
    log(f"Found NPZ   : {len(npz_files)}")

    if not npz_files:
        return

    success = 0
    skipped = 0
    failed = 0

    for idx, npz_path in enumerate(npz_files, start=1):
        rel = npz_path.relative_to(src_root)
        log("=" * 72)
        log(f"[{idx}/{len(npz_files)}] {rel}")

        if args.skip_invalid and not validate_vggt_npz(npz_path):
            skipped += 1
            continue

        camera_jsonl = find_camera_jsonl(npz_path)
        if camera_jsonl is None:
            log("Camera JSONL: not found, running without --camera-jsonl")
        else:
            log(f"Camera JSONL: {camera_jsonl.name}")

        out_prefix = make_out_prefix(
            npz_path=npz_path,
            src_root=src_root,
            dst_root=dst_root,
            layout=args.layout,
        )

        if already_done(out_prefix) and not args.force:
            log(f"SKIP already done: {out_prefix.with_name(out_prefix.name + '_fixedK_gop_nn_manifest.json')}")
            skipped += 1
            continue

        ret = run_one(
            npz_path=npz_path,
            camera_jsonl=camera_jsonl,
            out_prefix=out_prefix,
            optimizer_script=optimizer_script,
            python_exe=args.python,
            optimizer_args=optimizer_args,
            force=args.force,
            dry_run=args.dry_run,
        )

        if ret == 0:
            success += 1
            log("OK")
        else:
            failed += 1
            log(f"FAILED returncode={ret}")
            if not args.continue_on_error:
                raise RuntimeError(f"Optimizer failed on {npz_path}")

    log("=" * 72)
    log(f"Done. success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
