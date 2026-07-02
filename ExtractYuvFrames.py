#!/usr/bin/env python3
# extract_yuv_frames.py

import argparse
import os


def normalize_pix_fmt(fmt: str) -> str:
    fmt = fmt.lower().replace("-", "").replace("_", "")

    if fmt in ["420p", "yuv420p", "i420", "yuv420p8"]:
        return "yuv420p"

    if fmt in ["420p10le", "yuv420p10le", "i010"]:
        return "yuv420p10le"

    raise ValueError(f"Unsupported pix_fmt: {fmt}")


def get_frame_size(width: int, height: int, pix_fmt: str) -> int:
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("YUV420 requires even width and height.")

    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    samples = y_size + uv_size * 2

    if pix_fmt == "yuv420p":
        return samples

    if pix_fmt == "yuv420p10le":
        return samples * 2

    raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")


def extract_yuv_frames(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    pix_fmt: str,
    start_idx: int,
    end_idx: int,
):
    pix_fmt = normalize_pix_fmt(pix_fmt)
    frame_size = get_frame_size(width, height, pix_fmt)

    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")

    if end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")

    input_size = os.path.getsize(input_path)
    total_frames = input_size // frame_size

    if input_size % frame_size != 0:
        print(f"Warning: input file has trailing bytes: {input_size % frame_size}")

    if start_idx >= total_frames:
        raise ValueError(f"start_idx {start_idx} >= total_frames {total_frames}")

    if end_idx >= total_frames:
        raise ValueError(f"end_idx {end_idx} >= total_frames {total_frames}")

    num_frames = end_idx - start_idx + 1
    byte_offset = start_idx * frame_size
    byte_count = num_frames * frame_size

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        fin.seek(byte_offset)

        remaining = byte_count
        chunk_size = 1024 * 1024 * 64

        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = fin.read(read_size)

            if not data:
                raise EOFError("Unexpected EOF while reading input YUV.")

            fout.write(data)
            remaining -= len(data)

    print("Done")
    print(f"Input        : {input_path}")
    print(f"Output       : {output_path}")
    print(f"Resolution   : {width}x{height}")
    print(f"Pixel format : {pix_fmt}")
    print(f"Frame range  : {start_idx} ~ {end_idx}")
    print(f"Num frames   : {num_frames}")
    print(f"Frame size   : {frame_size} bytes")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="input YUV path")
    parser.add_argument("--output", required=True, help="output YUV path")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--pix-fmt", required=True, help="yuv420p / 420p / yuv420p10le / 420p10le")
    parser.add_argument("--start", type=int, required=True, help="start frame index, inclusive")
    parser.add_argument("--end", type=int, required=True, help="end frame index, inclusive")

    args = parser.parse_args()

    extract_yuv_frames(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        pix_fmt=args.pix_fmt,
        start_idx=args.start,
        end_idx=args.end,
    )


if __name__ == "__main__":
    main()
