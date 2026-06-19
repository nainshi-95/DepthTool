import argparse
import os
import numpy as np


def ceil_mul(x, m):
    return ((x + m - 1) // m) * m


def frame_size_yuv420p10le(w, h):
    # yuv420p10le = uint16 little-endian per sample
    y = w * h
    u = (w // 2) * (h // 2)
    v = u
    return (y + u + v) * 2


def reflect_pad_right_bottom(plane, out_h, out_w):
    h, w = plane.shape
    pad_b = out_h - h
    pad_r = out_w - w

    if pad_b == 0 and pad_r == 0:
        return plane

    # np.pad(mode="reflect")는 edge 값을 반복하지 않는 reflect
    # 예: [0,1,2] 오른쪽 pad 2 -> [0,1,2,1,0]
    return np.pad(
        plane,
        pad_width=((0, pad_b), (0, pad_r)),
        mode="reflect"
    )


def pad_yuv420p10le(input_path, output_path, width, height):
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("yuv420p10le 4:2:0 input width/height must be even.")

    out_w = ceil_mul(width, 4)
    out_h = ceil_mul(height, 4)

    in_frame_size = frame_size_yuv420p10le(width, height)
    file_size = os.path.getsize(input_path)

    if file_size % in_frame_size != 0:
        raise ValueError(
            f"Input file size is not a multiple of one frame size. "
            f"file_size={file_size}, frame_size={in_frame_size}"
        )

    num_frames = file_size // in_frame_size

    y_size = width * height
    c_w = width // 2
    c_h = height // 2
    c_size = c_w * c_h

    out_c_w = out_w // 2
    out_c_h = out_h // 2

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        for frame_idx in range(num_frames):
            raw = np.fromfile(fin, dtype="<u2", count=y_size + 2 * c_size)

            if raw.size != y_size + 2 * c_size:
                raise RuntimeError(f"Failed to read frame {frame_idx}")

            y = raw[:y_size].reshape(height, width)
            u = raw[y_size:y_size + c_size].reshape(c_h, c_w)
            v = raw[y_size + c_size:].reshape(c_h, c_w)

            y_pad = reflect_pad_right_bottom(y, out_h, out_w)
            u_pad = reflect_pad_right_bottom(u, out_c_h, out_c_w)
            v_pad = reflect_pad_right_bottom(v, out_c_h, out_c_w)

            y_pad.astype("<u2", copy=False).tofile(fout)
            u_pad.astype("<u2", copy=False).tofile(fout)
            v_pad.astype("<u2", copy=False).tofile(fout)

    print("Done")
    print(f"Input  : {width}x{height}")
    print(f"Output : {out_w}x{out_h}")
    print(f"Frames : {num_frames}")
    print(f"Saved  : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input yuv420p10le file")
    parser.add_argument("output", help="output padded yuv420p10le file")
    parser.add_argument("--width", "-w", type=int, required=True)
    parser.add_argument("--height", "-H", type=int, required=True)

    args = parser.parse_args()

    pad_yuv420p10le(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
