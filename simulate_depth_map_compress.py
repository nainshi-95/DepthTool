#!/usr/bin/env python3
# depth_plane_sim.py

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Plane:
    a: float
    b: float
    c: float
    cx: float
    cy: float


@dataclass
class ModeResult:
    mode: str
    candidate_name: str
    plane: Plane
    recon_block: np.ndarray
    bits: float
    sse: float
    cost: float
    q_values: Tuple[int, ...]


def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return int(math.ceil(math.log2(x)))


def exp_golomb_len_unsigned(u: int) -> int:
    """
    Unsigned Exp-Golomb code length.
    code_num = u
    length = 2 * floor(log2(code_num + 1)) + 1
    """
    if u < 0:
        raise ValueError("unsigned Exp-Golomb input must be non-negative")
    return 2 * int(math.floor(math.log2(u + 1))) + 1


def signed_to_code_num(v: int) -> int:
    """
    Signed integer mapping:
      0  -> 0
      +1 -> 1
      -1 -> 2
      +2 -> 3
      -2 -> 4
    """
    if v == 0:
        return 0
    if v > 0:
        return 2 * v - 1
    return -2 * v


def exp_golomb_len_signed(v: int) -> int:
    return exp_golomb_len_unsigned(signed_to_code_num(v))


def quantize(x: float, qstep: float) -> int:
    return int(np.rint(x / qstep))


def dequantize(q: int, qstep: float) -> float:
    return float(q) * qstep


def make_plane_grids(block_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local coordinates centered at block center.
    For block_size=16:
      x, y = -7.5 ... +7.5
    """
    coords = np.arange(block_size, dtype=np.float64) - (block_size - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords)
    design = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.ones(block_size * block_size)],
        axis=1,
    )
    pinv = np.linalg.pinv(design)
    return xx, yy, pinv


def fit_plane(block: np.ndarray, pinv: np.ndarray, cx: float, cy: float) -> Plane:
    """
    Least squares fitting:
      depth(x,y) ~= a*x + b*y + c
    """
    coeff = pinv @ block.astype(np.float64).reshape(-1)
    a, b, c = coeff.tolist()
    return Plane(a=a, b=b, c=c, cx=cx, cy=cy)


def plane_to_current_center(src: Plane, cur_cx: float, cur_cy: float) -> Plane:
    """
    Convert neighboring plane to current block center.

    src.c is depth at src center.
    Current center c is:
      c_cur = src.c + src.a * (cur_cx - src.cx) + src.b * (cur_cy - src.cy)
    """
    dcx = cur_cx - src.cx
    dcy = cur_cy - src.cy
    c_cur = src.c + src.a * dcx + src.b * dcy
    return Plane(a=src.a, b=src.b, c=c_cur, cx=cur_cx, cy=cur_cy)


def render_plane_block(
    plane: Plane,
    xx: np.ndarray,
    yy: np.ndarray,
    max_value: int,
) -> np.ndarray:
    block = plane.a * xx + plane.b * yy + plane.c
    block = np.rint(block)
    block = np.clip(block, 0, max_value)
    return block.astype(np.float64)


def block_sse(orig: np.ndarray, recon: np.ndarray) -> float:
    diff = orig.astype(np.float64) - recon.astype(np.float64)
    return float(np.sum(diff * diff))


def read_yuv420p10le_y_frame(
    fp,
    frame_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Read only Y plane from yuv420p10le.

    Assumption:
      Y: width * height uint16 little-endian samples
      U: width/2 * height/2 uint16
      V: width/2 * height/2 uint16

    Total frame size:
      width * height * 3 bytes
    """
    y_samples = width * height
    y_bytes = y_samples * 2
    frame_size = width * height * 3

    fp.seek(frame_idx * frame_size)
    raw = fp.read(y_bytes)
    if len(raw) != y_bytes:
        raise EOFError(f"Failed to read frame {frame_idx}")

    y = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    return y.astype(np.float64)


def count_yuv420p10le_frames(path: str, width: int, height: int) -> int:
    frame_size = width * height * 3
    size = os.path.getsize(path)
    return size // frame_size


def pad_to_block_multiple(
    img: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if pad_h == 0 and pad_w == 0:
        return img.copy(), h, w

    padded = np.pad(
        img,
        pad_width=((0, pad_h), (0, pad_w)),
        mode="edge",
    )
    return padded, h + pad_h, w + pad_w


def write_yuv420p10le_frame(
    fp,
    y: np.ndarray,
    width: int,
    height: int,
    max_value: int,
):
    y_u16 = np.clip(np.rint(y), 0, max_value).astype("<u2")
    fp.write(y_u16.tobytes())

    uv_h = height // 2
    uv_w = width // 2
    uv_value = min(512, max_value)
    uv = np.full((uv_h, uv_w), uv_value, dtype="<u2")
    fp.write(uv.tobytes())
    fp.write(uv.tobytes())


def make_candidates(
    plane_store: Dict[Tuple[int, int], Plane],
    bx: int,
    by: int,
    block_size: int,
    cur_cx: float,
    cur_cy: float,
) -> List[Tuple[str, Plane]]:
    """
    Raster-scan available neighbors:
      left
      top
      top-left
      top-right
      avg(left, top)
    """
    candidates: List[Tuple[str, Plane]] = []

    neighbor_keys = [
        ("left", (bx - block_size, by)),
        ("top", (bx, by - block_size)),
        ("top_left", (bx - block_size, by - block_size)),
        ("top_right", (bx + block_size, by - block_size)),
    ]

    converted: Dict[str, Plane] = {}

    for name, key in neighbor_keys:
        if key in plane_store:
            p = plane_to_current_center(plane_store[key], cur_cx, cur_cy)
            converted[name] = p
            candidates.append((name, p))

    if "left" in converted and "top" in converted:
        l = converted["left"]
        t = converted["top"]
        avg = Plane(
            a=0.5 * (l.a + t.a),
            b=0.5 * (l.b + t.b),
            c=0.5 * (l.c + t.c),
            cx=cur_cx,
            cy=cur_cy,
        )
        candidates.append(("avg_left_top", avg))

    return candidates


def eval_direct_mode(
    orig_block: np.ndarray,
    actual: Plane,
    xx: np.ndarray,
    yy: np.ndarray,
    qa_step: float,
    qb_step: float,
    qc_step: float,
    lambda_rd: float,
    mode_bits: int,
    max_value: int,
) -> ModeResult:
    qa = quantize(actual.a, qa_step)
    qb = quantize(actual.b, qb_step)
    qc = quantize(actual.c, qc_step)

    recon_plane = Plane(
        a=dequantize(qa, qa_step),
        b=dequantize(qb, qb_step),
        c=dequantize(qc, qc_step),
        cx=actual.cx,
        cy=actual.cy,
    )

    recon_block = render_plane_block(recon_plane, xx, yy, max_value)
    sse = block_sse(orig_block, recon_block)

    bits = float(mode_bits)
    bits += exp_golomb_len_signed(qa)
    bits += exp_golomb_len_signed(qb)

    if qc >= 0:
        bits += exp_golomb_len_unsigned(qc)
    else:
        bits += exp_golomb_len_signed(qc)

    cost = sse + lambda_rd * bits

    return ModeResult(
        mode="direct",
        candidate_name="none",
        plane=recon_plane,
        recon_block=recon_block,
        bits=bits,
        sse=sse,
        cost=cost,
        q_values=(qa, qb, qc),
    )


def eval_copy_modes(
    orig_block: np.ndarray,
    candidates: List[Tuple[str, Plane]],
    xx: np.ndarray,
    yy: np.ndarray,
    lambda_rd: float,
    mode_bits: int,
    max_value: int,
) -> List[ModeResult]:
    results: List[ModeResult] = []

    if not candidates:
        return results

    cand_bits = ceil_log2(len(candidates))

    for cand_name, cand_plane in candidates:
        recon_block = render_plane_block(cand_plane, xx, yy, max_value)
        sse = block_sse(orig_block, recon_block)

        bits = float(mode_bits + cand_bits)
        cost = sse + lambda_rd * bits

        results.append(
            ModeResult(
                mode="copy",
                candidate_name=cand_name,
                plane=cand_plane,
                recon_block=recon_block,
                bits=bits,
                sse=sse,
                cost=cost,
                q_values=(),
            )
        )

    return results


def eval_delta_modes(
    orig_block: np.ndarray,
    actual: Plane,
    candidates: List[Tuple[str, Plane]],
    xx: np.ndarray,
    yy: np.ndarray,
    qa_step: float,
    qb_step: float,
    qc_step: float,
    lambda_rd: float,
    mode_bits: int,
    max_value: int,
) -> List[ModeResult]:
    results: List[ModeResult] = []

    if not candidates:
        return results

    cand_bits = ceil_log2(len(candidates))

    for cand_name, pred in candidates:
        da = actual.a - pred.a
        db = actual.b - pred.b
        dc = actual.c - pred.c

        qda = quantize(da, qa_step)
        qdb = quantize(db, qb_step)
        qdc = quantize(dc, qc_step)

        recon_plane = Plane(
            a=pred.a + dequantize(qda, qa_step),
            b=pred.b + dequantize(qdb, qb_step),
            c=pred.c + dequantize(qdc, qc_step),
            cx=actual.cx,
            cy=actual.cy,
        )

        recon_block = render_plane_block(recon_plane, xx, yy, max_value)
        sse = block_sse(orig_block, recon_block)

        bits = float(mode_bits + cand_bits)
        bits += exp_golomb_len_signed(qda)
        bits += exp_golomb_len_signed(qdb)
        bits += exp_golomb_len_signed(qdc)

        cost = sse + lambda_rd * bits

        results.append(
            ModeResult(
                mode="delta",
                candidate_name=cand_name,
                plane=recon_plane,
                recon_block=recon_block,
                bits=bits,
                sse=sse,
                cost=cost,
                q_values=(qda, qdb, qdc),
            )
        )

    return results


def compute_metrics(
    orig: np.ndarray,
    recon: np.ndarray,
    max_value: int,
) -> Dict[str, float]:
    diff = orig.astype(np.float64) - recon.astype(np.float64)

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(mse))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((max_value * max_value) / mse)

    max_err = float(np.max(np.abs(diff)))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "max_error": max_err,
    }


def simulate_one_frame(
    depth: np.ndarray,
    frame_idx: int,
    block_size: int,
    qa_step: float,
    qb_step: float,
    qc_step: float,
    lambda_rd: float,
    mode_bits: int,
    max_value: int,
    xx: np.ndarray,
    yy: np.ndarray,
    pinv: np.ndarray,
    block_csv_writer: Optional[csv.DictWriter] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = depth.shape
    padded, hp, wp = pad_to_block_multiple(depth, block_size)

    recon_padded = np.zeros_like(padded, dtype=np.float64)
    plane_store: Dict[Tuple[int, int], Plane] = {}

    total_bits = 0.0
    total_sse = 0.0

    mode_count = {
        "direct": 0,
        "copy": 0,
        "delta": 0,
    }

    candidate_count: Dict[str, int] = {}

    zero_delta_count = 0
    delta_mode_count = 0

    for by in range(0, hp, block_size):
        for bx in range(0, wp, block_size):
            block = padded[by : by + block_size, bx : bx + block_size]

            cur_cx = bx + (block_size - 1) / 2.0
            cur_cy = by + (block_size - 1) / 2.0

            actual = fit_plane(block, pinv, cur_cx, cur_cy)

            candidates = make_candidates(
                plane_store=plane_store,
                bx=bx,
                by=by,
                block_size=block_size,
                cur_cx=cur_cx,
                cur_cy=cur_cy,
            )

            mode_results: List[ModeResult] = []

            mode_results.append(
                eval_direct_mode(
                    orig_block=block,
                    actual=actual,
                    xx=xx,
                    yy=yy,
                    qa_step=qa_step,
                    qb_step=qb_step,
                    qc_step=qc_step,
                    lambda_rd=lambda_rd,
                    mode_bits=mode_bits,
                    max_value=max_value,
                )
            )

            mode_results.extend(
                eval_copy_modes(
                    orig_block=block,
                    candidates=candidates,
                    xx=xx,
                    yy=yy,
                    lambda_rd=lambda_rd,
                    mode_bits=mode_bits,
                    max_value=max_value,
                )
            )

            mode_results.extend(
                eval_delta_modes(
                    orig_block=block,
                    actual=actual,
                    candidates=candidates,
                    xx=xx,
                    yy=yy,
                    qa_step=qa_step,
                    qb_step=qb_step,
                    qc_step=qc_step,
                    lambda_rd=lambda_rd,
                    mode_bits=mode_bits,
                    max_value=max_value,
                )
            )

            best = min(mode_results, key=lambda r: r.cost)

            recon_padded[by : by + block_size, bx : bx + block_size] = best.recon_block
            plane_store[(bx, by)] = best.plane

            total_bits += best.bits
            total_sse += best.sse

            mode_count[best.mode] += 1
            candidate_count[best.candidate_name] = candidate_count.get(best.candidate_name, 0) + 1

            if best.mode == "delta":
                delta_mode_count += 1
                if len(best.q_values) == 3 and best.q_values == (0, 0, 0):
                    zero_delta_count += 1

            if block_csv_writer is not None:
                q0 = best.q_values[0] if len(best.q_values) > 0 else ""
                q1 = best.q_values[1] if len(best.q_values) > 1 else ""
                q2 = best.q_values[2] if len(best.q_values) > 2 else ""

                block_csv_writer.writerow(
                    {
                        "frame": frame_idx,
                        "bx": bx,
                        "by": by,
                        "mode": best.mode,
                        "candidate": best.candidate_name,
                        "bits": best.bits,
                        "sse": best.sse,
                        "cost": best.cost,
                        "q0": q0,
                        "q1": q1,
                        "q2": q2,
                        "actual_a": actual.a,
                        "actual_b": actual.b,
                        "actual_c": actual.c,
                        "recon_a": best.plane.a,
                        "recon_b": best.plane.b,
                        "recon_c": best.plane.c,
                    }
                )

    recon = recon_padded[:h, :w]
    metrics = compute_metrics(depth, recon, max_value=max_value)

    num_pixels = h * w
    num_blocks = (hp // block_size) * (wp // block_size)

    summary = {
        "frame": frame_idx,
        "width": w,
        "height": h,
        "padded_width": wp,
        "padded_height": hp,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "bits": total_bits,
        "bpp": total_bits / num_pixels,
        "sse": total_sse,
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "max_error": metrics["max_error"],
        "direct_blocks": mode_count["direct"],
        "copy_blocks": mode_count["copy"],
        "delta_blocks": mode_count["delta"],
        "direct_ratio": mode_count["direct"] / num_blocks,
        "copy_ratio": mode_count["copy"] / num_blocks,
        "delta_ratio": mode_count["delta"] / num_blocks,
        "zero_delta_blocks": zero_delta_count,
        "zero_delta_ratio_in_delta": (
            zero_delta_count / delta_mode_count if delta_mode_count > 0 else 0.0
        ),
    }

    for name, count in candidate_count.items():
        safe_name = name.replace("-", "_")
        summary[f"candidate_{safe_name}_count"] = count

    return recon, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Block-wise depth plane compression simulation for yuv420p10le depth maps."
    )

    parser.add_argument("--input", required=True, help="Input depth yuv420p10le path")
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="0 means all remaining frames",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Square block size. Example: 8, 16, 32",
    )

    parser.add_argument(
        "--lambda-rd",
        type=float,
        default=0.0,
        help="R-D lambda. cost = SSE + lambda * bits",
    )

    parser.add_argument("--qa", type=float, default=0.25, help="Quant step for a")
    parser.add_argument("--qb", type=float, default=0.25, help="Quant step for b")
    parser.add_argument("--qc", type=float, default=2.0, help="Quant step for c")

    parser.add_argument(
        "--mode-bits",
        type=int,
        default=2,
        help="Simplified mode signaling bits per block",
    )

    parser.add_argument(
        "--max-value",
        type=int,
        default=1023,
        help="Max depth sample value. For 10-bit, use 1023.",
    )

    parser.add_argument(
        "--out-csv",
        default="depth_plane_frame_stats.csv",
        help="Frame-level CSV output",
    )

    parser.add_argument(
        "--out-json",
        default="depth_plane_summary.json",
        help="Overall JSON summary output",
    )

    parser.add_argument(
        "--out-recon-yuv",
        default="",
        help="Optional reconstructed yuv420p10le output path",
    )

    parser.add_argument(
        "--out-block-csv",
        default="",
        help="Optional block-level CSV output path. Can be very large.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    if args.qa <= 0 or args.qb <= 0 or args.qc <= 0:
        raise ValueError("--qa, --qb, --qc must be positive")

    if args.width % 2 != 0 or args.height % 2 != 0:
        raise ValueError("yuv420p10le requires even width and height")

    total_frames = count_yuv420p10le_frames(args.input, args.width, args.height)

    if args.start_frame < 0 or args.start_frame >= total_frames:
        raise ValueError(
            f"--start-frame must be in [0, {total_frames - 1}], got {args.start_frame}"
        )

    if args.num_frames == 0:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, args.start_frame + args.num_frames)

    xx, yy, pinv = make_plane_grids(args.block_size)

    frame_summaries: List[Dict[str, float]] = []

    recon_fp = None
    if args.out_recon_yuv:
        recon_fp = open(args.out_recon_yuv, "wb")

    block_csv_fp = None
    block_csv_writer = None

    if args.out_block_csv:
        block_csv_fp = open(args.out_block_csv, "w", newline="")
        block_csv_writer = csv.DictWriter(
            block_csv_fp,
            fieldnames=[
                "frame",
                "bx",
                "by",
                "mode",
                "candidate",
                "bits",
                "sse",
                "cost",
                "q0",
                "q1",
                "q2",
                "actual_a",
                "actual_b",
                "actual_c",
                "recon_a",
                "recon_b",
                "recon_c",
            ],
        )
        block_csv_writer.writeheader()

    try:
        with open(args.input, "rb") as fp:
            for frame_idx in range(args.start_frame, end_frame):
                depth = read_yuv420p10le_y_frame(
                    fp=fp,
                    frame_idx=frame_idx,
                    width=args.width,
                    height=args.height,
                )

                recon, summary = simulate_one_frame(
                    depth=depth,
                    frame_idx=frame_idx,
                    block_size=args.block_size,
                    qa_step=args.qa,
                    qb_step=args.qb,
                    qc_step=args.qc,
                    lambda_rd=args.lambda_rd,
                    mode_bits=args.mode_bits,
                    max_value=args.max_value,
                    xx=xx,
                    yy=yy,
                    pinv=pinv,
                    block_csv_writer=block_csv_writer,
                )

                frame_summaries.append(summary)

                if recon_fp is not None:
                    write_yuv420p10le_frame(
                        fp=recon_fp,
                        y=recon,
                        width=args.width,
                        height=args.height,
                        max_value=args.max_value,
                    )

                print(
                    f"Frame {frame_idx:4d} | "
                    f"bpp={summary['bpp']:.5f} | "
                    f"MAE={summary['mae']:.4f} | "
                    f"RMSE={summary['rmse']:.4f} | "
                    f"PSNR={summary['psnr']:.3f} | "
                    f"D/C/Δ={summary['direct_ratio']:.3f}/"
                    f"{summary['copy_ratio']:.3f}/"
                    f"{summary['delta_ratio']:.3f}"
                )

    finally:
        if recon_fp is not None:
            recon_fp.close()
        if block_csv_fp is not None:
            block_csv_fp.close()

    if not frame_summaries:
        raise RuntimeError("No frames processed")

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = sorted(set().union(*(s.keys() for s in frame_summaries)))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in frame_summaries:
            writer.writerow(s)

    avg = {}
    numeric_keys = [
        "bits",
        "bpp",
        "mae",
        "mse",
        "rmse",
        "psnr",
        "max_error",
        "direct_ratio",
        "copy_ratio",
        "delta_ratio",
        "zero_delta_ratio_in_delta",
    ]

    for k in numeric_keys:
        vals = [float(s[k]) for s in frame_summaries if k in s and math.isfinite(float(s[k]))]
        if vals:
            avg[k] = float(np.mean(vals))

    total_bits = float(sum(s["bits"] for s in frame_summaries))
    total_pixels = float(args.width * args.height * len(frame_summaries))

    overall = {
        "input": args.input,
        "width": args.width,
        "height": args.height,
        "start_frame": args.start_frame,
        "num_processed_frames": len(frame_summaries),
        "block_size": args.block_size,
        "qa": args.qa,
        "qb": args.qb,
        "qc": args.qc,
        "lambda_rd": args.lambda_rd,
        "mode_bits": args.mode_bits,
        "max_value": args.max_value,
        "total_bits": total_bits,
        "overall_bpp": total_bits / total_pixels,
        "average": avg,
        "frame_csv": args.out_csv,
        "recon_yuv": args.out_recon_yuv,
        "block_csv": args.out_block_csv,
    }

    with open(args.out_json, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("Done.")
    print(f"Frame CSV : {args.out_csv}")
    print(f"Summary   : {args.out_json}")
    if args.out_recon_yuv:
        print(f"Recon YUV : {args.out_recon_yuv}")
    if args.out_block_csv:
        print(f"Block CSV : {args.out_block_csv}")
    print(f"Overall bpp: {overall['overall_bpp']:.6f}")


if __name__ == "__main__":
    main()
