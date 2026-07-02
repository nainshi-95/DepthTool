#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# NPZ loading
# ============================================================

def load_vggt_npz(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)

    required = ["extrinsic", "intrinsic_original"]
    for k in required:
        if k not in z:
            raise KeyError(f"{npz_path}: missing key '{k}'")

    extrinsic = z["extrinsic"].astype(np.float32)
    intrinsic = z["intrinsic_original"].astype(np.float32)

    if extrinsic.ndim != 3:
        raise ValueError(f"{npz_path}: extrinsic must be Sx3x4 or Sx4x4. Got {extrinsic.shape}")

    if intrinsic.ndim != 3 or intrinsic.shape[1:] != (3, 3):
        raise ValueError(f"{npz_path}: intrinsic_original must be Sx3x3. Got {intrinsic.shape}")

    n = extrinsic.shape[0]

    if intrinsic.shape[0] != n:
        raise ValueError(
            f"{npz_path}: frame count mismatch: "
            f"extrinsic={extrinsic.shape[0]}, intrinsic={intrinsic.shape[0]}"
        )

    if "frame_indices" in z:
        frame_indices = z["frame_indices"].astype(np.int64).tolist()
    else:
        frame_indices = list(range(n))

    if len(frame_indices) != n:
        raise ValueError(
            f"{npz_path}: frame_indices length {len(frame_indices)} != frame count {n}"
        )

    rap_name = None
    if "rap_name" in z:
        rap_name = npz_scalar_to_str(z["rap_name"])

    rap_index = None
    if "rap_index" in z:
        rap_index = int(np.asarray(z["rap_index"]).item())

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "frame_indices": [int(x) for x in frame_indices],
        "rap_name": rap_name,
        "rap_index": rap_index,
    }


def npz_scalar_to_str(x) -> str:
    arr = np.asarray(x)

    if arr.shape == ():
        v = arr.item()
    else:
        v = arr.tolist()

    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")

    return str(v)


# ============================================================
# File naming
# ============================================================

def infer_sequence_and_rap(npz_path: Path, rap_name_from_npz=None, rap_index_from_npz=None):
    """
    Examples:
      BasketballDrive_rap0_vggt_omega_outputs.npz
        -> sequence_name = BasketballDrive
        -> rap_name      = rap0

      test_000_064_rap12_vggt_omega_outputs.npz
        -> sequence_name = test_000_064
        -> rap_name      = rap12

      something.npz
        -> sequence_name = something
        -> rap_name      = rap0
    """
    stem = npz_path.stem

    # Remove common VGGT suffix.
    for suffix in [
        "_vggt_omega_outputs",
        "_outputs",
        "_npz",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    m = re.search(r"^(?P<seq>.+?)_(?P<rap>rap\d+)$", stem)

    if m:
        sequence_name = m.group("seq")
        rap_name = m.group("rap")
        return sequence_name, rap_name

    if rap_name_from_npz is not None and re.fullmatch(r"rap\d+", rap_name_from_npz):
        rap_name = rap_name_from_npz
    elif rap_index_from_npz is not None:
        rap_name = f"rap{rap_index_from_npz}"
    else:
        rap_name = "rap0"

    # If filename already ends with rapK but regex missed for some reason.
    m2 = re.search(r"(rap\d+)$", stem)
    if m2:
        rap_name = m2.group(1)
        sequence_name = stem[: -len(rap_name)].rstrip("_")
        if not sequence_name:
            sequence_name = stem
    else:
        sequence_name = stem

    return sequence_name, rap_name


def make_output_path(npz_path: Path, sequence_name: str, rap_name: str, overwrite_name=None):
    if overwrite_name:
        return npz_path.parent / overwrite_name

    return npz_path.parent / f"{sequence_name}_camParam_{rap_name}.jsonl"


# ============================================================
# Pose conversion
# ============================================================

def extrinsic_to_4x4(E: np.ndarray, translation_scale: float = 1.0) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)

    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = E[:3, :3]
        T[:3, 3] = E[:3, 3] * translation_scale
        return T

    if E.shape == (4, 4):
        T = E.copy()
        T[:3, 3] *= translation_scale
        return T

    raise ValueError(f"Unsupported extrinsic shape: {E.shape}")


def rt_cur_to_prev_from_extrinsics(
    extrinsics: np.ndarray,
    translation_scale: float,
):
    """
    VGGT extrinsic convention:
      W2C = camera_from_world

    Output pose mode:
      X_prev = R * X_cur + t

    For poc 0:
      rvec = 0
      tvec = 0

    For poc i:
      T_cur_to_prev = W2C_prev @ C2W_cur
    """
    n = extrinsics.shape[0]

    W2Cs = [
        extrinsic_to_4x4(extrinsics[i], translation_scale=translation_scale)
        for i in range(n)
    ]

    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)

    for i in range(1, n):
        W2C_prev = W2Cs[i - 1]
        C2W_cur = np.linalg.inv(W2Cs[i])

        T = W2C_prev @ C2W_cur

        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3].astype(np.float64)

        rvec, _ = cv2.Rodrigues(R)

        rvecs[i] = rvec.reshape(3).astype(np.float32)
        tvecs[i] = t.reshape(3).astype(np.float32)

    # numerical cleanup
    rvecs[0] = 0.0
    tvecs[0] = 0.0

    return rvecs, tvecs


def rt_gop_local_from_extrinsics(
    extrinsics: np.ndarray,
    translation_scale: float,
):
    """
    Alternative pose mode:
      GOP 첫 frame 기준 absolute pose.

    poc 0:
      identity

    poc i:
      T_rel_i = W2C_i @ C2W_0
    """
    n = extrinsics.shape[0]

    W2Cs = [
        extrinsic_to_4x4(extrinsics[i], translation_scale=translation_scale)
        for i in range(n)
    ]

    C2W_0 = np.linalg.inv(W2Cs[0])

    rvecs = np.zeros((n, 3), dtype=np.float32)
    tvecs = np.zeros((n, 3), dtype=np.float32)

    for i in range(1, n):
        T = W2Cs[i] @ C2W_0

        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3].astype(np.float64)

        rvec, _ = cv2.Rodrigues(R)

        rvecs[i] = rvec.reshape(3).astype(np.float32)
        tvecs[i] = t.reshape(3).astype(np.float32)

    rvecs[0] = 0.0
    tvecs[0] = 0.0

    return rvecs, tvecs


# ============================================================
# JSONL writing
# ============================================================

def intrinsic_to_header_dict(K: np.ndarray, z_sign: float):
    K = np.asarray(K, dtype=np.float64)

    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "z_sign": float(z_sign),
    }


def write_camparam_jsonl(
    out_path: Path,
    frame_indices,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: float,
    z_sign: float,
    pose_mode: str,
    width: int | None = None,
    height: int | None = None,
    source_npz: Path | None = None,
):
    header = {
        "type": "header",
        "depth_scale": float(depth_scale),
        "intrinsic": intrinsic_to_header_dict(intrinsic[0], z_sign=z_sign),
        "camera_param": "rvec_tvec_6d",
        "pose_mode": pose_mode,
        "frame_line_format": {
            "type": "omitted",
            "fields": ["poc", "rvec", "tvec"],
        },
    }

    if width is not None:
        header["width"] = int(width)

    if height is not None:
        header["height"] = int(height)

    if source_npz is not None:
        header["source_npz"] = str(source_npz.resolve())

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(header, ensure_ascii=False) + "\n")

        for poc in range(len(frame_indices)):
            rec = {
                "poc": int(poc),
                "rvec": [float(x) for x in rvecs[poc]],
                "tvec": [float(x) for x in tvecs[poc]],
            }

            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Recursive processing
# ============================================================

def find_npz_files(root: Path, pattern: str):
    return sorted(root.rglob(pattern))


def process_one_npz(
    npz_path: Path,
    args,
):
    data = load_vggt_npz(npz_path)

    sequence_name, rap_name = infer_sequence_and_rap(
        npz_path,
        rap_name_from_npz=data["rap_name"],
        rap_index_from_npz=data["rap_index"],
    )

    out_path = make_output_path(
        npz_path,
        sequence_name=sequence_name,
        rap_name=rap_name,
        overwrite_name=None,
    )

    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] exists: {out_path}")
        return

    extrinsics = data["extrinsic"]
    intrinsic = data["intrinsic"]
    frame_indices = data["frame_indices"]

    if args.pose_mode == "current_to_previous":
        rvecs, tvecs = rt_cur_to_prev_from_extrinsics(
            extrinsics,
            translation_scale=args.depth_scale,
        )
    elif args.pose_mode == "gop_local":
        rvecs, tvecs = rt_gop_local_from_extrinsics(
            extrinsics,
            translation_scale=args.depth_scale,
        )
    else:
        raise ValueError(args.pose_mode)

    write_camparam_jsonl(
        out_path=out_path,
        frame_indices=frame_indices,
        rvecs=rvecs,
        tvecs=tvecs,
        intrinsic=intrinsic,
        depth_scale=args.depth_scale,
        z_sign=args.z_sign,
        pose_mode=args.pose_mode,
        width=args.width,
        height=args.height,
        source_npz=npz_path,
    )

    print(f"[OK] {npz_path}")
    print(f"     -> {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
        help="Root folder. All NPZ files under this folder are recursively converted.",
    )

    ap.add_argument(
        "--pattern",
        default="*.npz",
        help="Recursive NPZ glob pattern. Default: *.npz",
    )

    ap.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale applied to camera translation. Must match the depth YUV scale.",
    )

    ap.add_argument(
        "--z-sign",
        type=float,
        default=1.0,
        help="Use 1.0 for VGGT/OpenCV +Z forward depth. Use -1.0 only if your warp code expects -Z forward.",
    )

    ap.add_argument(
        "--pose-mode",
        choices=["current_to_previous", "gop_local"],
        default="current_to_previous",
        help=(
            "current_to_previous: poc i stores frame i -> frame i-1 transform. "
            "gop_local: poc i stores transform relative to GOP first frame."
        ),
    )

    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    root = Path(args.root)

    if not root.is_dir():
        raise RuntimeError(f"Not a directory: {root}")

    if args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")

    npz_files = find_npz_files(root, args.pattern)

    if not npz_files:
        print(f"No NPZ files found under: {root}")
        return

    print(f"Found NPZ files: {len(npz_files)}")
    print(f"pose_mode   : {args.pose_mode}")
    print(f"depth_scale : {args.depth_scale}")
    print(f"z_sign      : {args.z_sign}")

    ok = 0
    fail = 0

    for npz_path in npz_files:
        try:
            process_one_npz(npz_path, args)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {npz_path}")
            print(f"       {type(exc).__name__}: {exc}")

    print("Done.")
    print(f"converted: {ok}")
    print(f"failed   : {fail}")


if __name__ == "__main__":
    main()
