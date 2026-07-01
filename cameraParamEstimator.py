import argparse
import numpy as np
import cv2
import os

def read_yuv_frame(file_path, width, height, frame_idx, fmt='420p'):
    """ YUV 파일에서 특정 프레임을 읽어 Y, U, V 배열로 반환합니다. """
    bpp = 2 if fmt == '420p10le' else 1
    dtype = np.uint16 if fmt == '420p10le' else np.uint8
    
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size
    offset = frame_idx * frame_size * bpp

    with open(file_path, 'rb') as f:
        f.seek(offset)
        y_data = np.fromfile(f, dtype=dtype, count=y_size).reshape((height, width))
        u_data = np.fromfile(f, dtype=dtype, count=uv_size).reshape((height // 2, width // 2))
        v_data = np.fromfile(f, dtype=dtype, count=uv_size).reshape((height // 2, width // 2))
        
    return y_data, u_data, v_data

def write_yuv_frame(file_path, y, u, v, fmt='420p'):
    """ Y, U, V 배열을 YUV 파일에 append 합니다. """
    dtype = np.uint16 if fmt == '420p10le' else np.uint8
    with open(file_path, 'ab') as f:
        f.write(y.astype(dtype).tobytes())
        f.write(u.astype(dtype).tobytes())
        f.write(v.astype(dtype).tobytes())

def get_chroma_homography(H_y):
    """ 
    Y(Luma)용 Homography를 4:2:0 UV(Chroma) 해상도에 맞게 변환합니다. 
    H_uv = S * H_y * S^-1 (S는 2배 스케일링 행렬)
    """
    S = np.array([[2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]])
    S_inv = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 1.0]])
    return S @ H_y @ S_inv

def main():
    parser = argparse.ArgumentParser(description="Block-wise Homography Backward Warping (YUV)")
    parser.add_argument('-i', '--input', required=True, help="Input YUV file")
    parser.add_argument('-o', '--output', required=True, help="Output Warped YUV file")
    parser.add_argument('-w', '--width', type=int, required=True, help="Width of video")
    parser.add_argument('-e', '--height', type=int, required=True, help="Height of video")
    parser.add_argument('-f', '--format', choices=['420p', '420p10le'], default='420p', help="YUV format")
    parser.add_argument('-r', '--refidx', type=int, required=True, help="Reference frame index")
    parser.add_argument('-t', '--taridx', type=int, required=True, help="Target frame index")
    parser.add_argument('-b', '--blocksize', type=int, default=128, help="Block size for local homography")
    
    args = parser.parse_args()

    # 기존 출력 파일이 있으면 삭제
    if os.path.exists(args.output):
        os.remove(args.output)

    # 1. 프레임 로드
    y_ref, u_ref, v_ref = read_yuv_frame(args.input, args.width, args.height, args.refidx, args.format)
    y_tar, u_tar, v_tar = read_yuv_frame(args.input, args.width, args.height, args.taridx, args.format)

    # OpenCV 처리를 위해 8bit 정규화 (모션 추정용)
    if args.format == '420p10le':
        ref_gray = (y_ref / 4).astype(np.uint8)
        tar_gray = (y_tar / 4).astype(np.uint8)
    else:
        ref_gray = y_ref
        tar_gray = y_tar

    # 2. Target -> Ref 방향의 Dense Optical Flow 계산 (특징점 부족 방지)
    flow = cv2.calcOpticalFlowFarneback(tar_gray, ref_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Warped 결과를 담을 빈 배열 생성
    y_warped = np.zeros_like(y_tar)
    u_warped = np.zeros_like(u_tar)
    v_warped = np.zeros_like(v_tar)

    max_val = 1023 if args.format == '420p10le' else 255
    border_mode = cv2.BORDER_REPLICATE

    # 3. 블록 단위 순회 및 Homography 피팅
    bs = args.blocksize
    for y in range(0, args.height, bs):
        for x in range(0, args.width, bs):
            y_end = min(y + bs, args.height)
            x_end = min(x + bs, args.width)
            
            # 현재 블록의 그리드 좌표 생성
            grid_y, grid_x = np.mgrid[y:y_end, x:x_end]
            pts_tar = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
            
            # Flow를 이용해 맵핑되는 Ref 좌표 생성
            flow_block = flow[y:y_end, x:x_end]
            pts_ref = pts_tar + flow_block.reshape(-1, 2)

            # RANSAC으로 Block-wise Homography (Target -> Ref) 계산
            H, mask = cv2.findHomography(pts_tar, pts_ref, cv2.RANSAC, 3.0)
            
            # Homography를 찾지 못한 경우 (fallback: Identity)
            if H is None:
                H = np.eye(3, dtype=np.float32)

            # 4. Y(Luma) Warping (전체 이미지를 Warp 후 블록 부분만 잘라오기)
            # 최적화를 위해서는 패딩된 ROI만 Warp해야 하지만, 프로토타입 직관성을 위해 전체 사용
            warped_ref_y = cv2.warpPerspective(y_ref.astype(np.float32), H, (args.width, args.height), flags=cv2.INTER_LINEAR, borderMode=border_mode)
            y_warped[y:y_end, x:x_end] = np.clip(warped_ref_y[y:y_end, x:x_end], 0, max_val)

            # 5. U, V(Chroma) Warping
            H_uv = get_chroma_homography(H)
            uv_w = args.width // 2
            uv_h = args.height // 2
            cx, cy = x // 2, y // 2
            cx_end, cy_end = x_end // 2, y_end // 2

            warped_ref_u = cv2.warpPerspective(u_ref.astype(np.float32), H_uv, (uv_w, uv_h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
            warped_ref_v = cv2.warpPerspective(v_ref.astype(np.float32), H_uv, (uv_w, uv_h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
            
            u_warped[cy:cy_end, cx:cx_end] = np.clip(warped_ref_u[cy:cy_end, cx:cx_end], 0, max_val)
            v_warped[cy:cy_end, cx:cx_end] = np.clip(warped_ref_v[cy:cy_end, cx:cx_end], 0, max_val)

    # 결과 저장
    write_yuv_frame(args.output, y_warped, u_warped, v_warped, args.format)
    print(f"Warping complete! Saved to {args.output}")

if __name__ == "__main__":
    main()
