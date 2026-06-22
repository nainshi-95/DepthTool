#!/usr/bin/env python3
import argparse
import os
import numpy as np


def plane_shape(w, h, plane, chroma):
    if plane == "Y":
        return h, w

    if chroma == "420":
        return h // 2, w // 2
    elif chroma == "422":
        return h, w // 2
    elif chroma == "444":
        return h, w
    elif chroma == "400":
        raise ValueError("YUV400 has no chroma plane")
    else:
        raise ValueError(f"Unsupported chroma format: {chroma}")


def frame_layout(w, h, bitdepth, chroma):
    dtype = np.uint8 if bitdepth <= 8 else np.dtype("<u2")
    bytes_per_sample = np.dtype(dtype).itemsize

    planes = ["Y"] if chroma == "400" else ["Y", "Cb", "Cr"]

    layout = []
    offset_samples = 0

    for p in planes:
        ph, pw = plane_shape(w, h, p, chroma)
        samples = ph * pw
        layout.append((p, ph, pw, offset_samples, samples))
        offset_samples += samples

    frame_samples = offset_samples
    frame_bytes = frame_samples * bytes_per_sample

    return dtype, bytes_per_sample, layout, frame_samples, frame_bytes


def compare_yuv(enc_path, dec_path, w, h, bitdepth, chroma, max_frames=None):
    dtype, bps, layout, frame_samples, frame_bytes = frame_layout(
        w, h, bitdepth, chroma
    )

    enc_size = os.path.getsize(enc_path)
    dec_size = os.path.getsize(dec_path)

    enc_frames = enc_size // frame_bytes
    dec_frames = dec_size // frame_bytes
    num_frames = min(enc_frames, dec_frames)

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    print(f"enc size      : {enc_size} bytes, {enc_frames} frames")
    print(f"dec size      : {dec_size} bytes, {dec_frames} frames")
    print(f"compare frames: {num_frames}")
    print(f"frame bytes   : {frame_bytes}")
    print(f"dtype         : {dtype}")
    print()

    with open(enc_path, "rb") as fe, open(dec_path, "rb") as fd:
        for poc in range(num_frames):
            enc_buf = fe.read(frame_bytes)
            dec_buf = fd.read(frame_bytes)

            enc = np.frombuffer(enc_buf, dtype=dtype)
            dec = np.frombuffer(dec_buf, dtype=dtype)

            if enc.size != frame_samples or dec.size != frame_samples:
                print(f"Short frame at POC {poc}")
                return

            if np.array_equal(enc, dec):
                continue

            for plane, ph, pw, start, samples in layout:
                e = enc[start:start + samples].reshape(ph, pw)
                d = dec[start:start + samples].reshape(ph, pw)

                diff = e != d
                if not diff.any():
                    continue

                y, x = np.argwhere(diff)[0]

                enc_val = int(e[y, x])
                dec_val = int(d[y, x])
                delta = dec_val - enc_val

                sample_offset_in_frame = start + y * pw + x
                byte_offset = poc * frame_bytes + sample_offset_in_frame * bps

                print("FIRST MISMATCH FOUND")
                print(f"POC/frame     : {poc}")
                print(f"plane         : {plane}")
                print(f"x, y          : {x}, {y}")
                print(f"enc value     : {enc_val}")
                print(f"dec value     : {dec_val}")
                print(f"delta dec-enc : {delta}")
                print(f"sample offset : {sample_offset_in_frame} in frame")
                print(f"byte offset   : {byte_offset} in file")

                if plane != "Y" and chroma == "420":
                    print(f"approx luma position: x={x * 2}, y={y * 2}")

                return

    print("No mismatch found in compared frames.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc", required=True, help="encoder reconstructed yuv")
    parser.add_argument("--dec", required=True, help="decoder output yuv")
    parser.add_argument("--width", "-w", type=int, required=True)
    parser.add_argument("--height", "-H", type=int, required=True)
    parser.add_argument("--bitdepth", "-b", type=int, default=10)
    parser.add_argument("--chroma", default="420", choices=["400", "420", "422", "444"])
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    compare_yuv(
        enc_path=args.enc,
        dec_path=args.dec,
        w=args.width,
        h=args.height,
        bitdepth=args.bitdepth,
        chroma=args.chroma,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
  
