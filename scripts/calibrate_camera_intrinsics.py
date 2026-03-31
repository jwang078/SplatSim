"""
Fisheye camera intrinsic calibration from a checkerboard video using OpenCV.
Outputs a JSON compatible with SplatSim's Camera class (FoVx, FoVy, resolution).

Since SplatSim's Camera class is pinhole-only, this script:
  1. Calibrates the fisheye distortion model (cv2.fisheye)
  2. Computes an optimal undistorted pinhole camera matrix
  3. Outputs FoVx/FoVy from the undistorted intrinsics

When using images with SplatSim, undistort them first using the saved
camera_matrix + dist_coeffs, then use the undistorted FoVx/FoVy.

Usage:
    python scripts/calibrate_camera_intrinsics.py \
        --video "/home/jennyw2/data/gopro calibration/GX019771.MP4" \
        --board-size 9 6 \
        --square-size 25.0 \
        --num-frames 40 \
        --output calibration.json --show
"""

import argparse
import json
import math
import cv2
import numpy as np
from pathlib import Path


def focal2fov(focal: float, pixels: int) -> float:
    """Convert focal length to field of view (matches gaussian_splatting/utils/graphics_utils.py)."""
    return 2 * math.atan(pixels / (2 * focal))


def extract_candidate_frames(video_path: str, num_frames: int) -> list[tuple[int, np.ndarray]]:
    """Sample num_frames evenly spaced frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((int(idx), frame))
    cap.release()
    print(f"Extracted {len(frames)} candidate frames from {total_frames} total")
    return frames


def detect_corners(
    frames: list[tuple[int, np.ndarray]],
    board_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect checkerboard corners in each frame.

    Returns obj_points, img_points, detected_frames, image_size.
    Points are shaped for cv2.fisheye: obj (1, N, 3), img (1, N, 2).
    detected_frames: list of BGR images where corners were found.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []
    detected_frames = []
    image_size = None
    used_indices = []

    for frame_idx, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(corners_refined.reshape(1, -1, 2))
            detected_frames.append(frame.copy())
            used_indices.append(frame_idx)

    print(f"Checkerboard detected in {len(obj_points)}/{len(frames)} candidate frames")
    print(f"  Frame indices used: {used_indices}")
    return obj_points, img_points, detected_frames, image_size


MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


def _resize_to_fit(grid: np.ndarray) -> np.ndarray:
    """Resize grid image to fit within MAX_DISPLAY_WIDTH x MAX_DISPLAY_HEIGHT."""
    gh, gw = grid.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / gw, MAX_DISPLAY_HEIGHT / gh, 1.0)
    if scale < 1.0:
        grid = cv2.resize(grid, (int(gw * scale), int(gh * scale)))
    return grid


def show_frame_grid(frames: list[np.ndarray], title: str = "Frames"):
    """Show a grid of frames (no corners)."""
    n = len(frames)
    if n == 0:
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    h, w = frames[0].shape[:2]
    thumb_w = 480
    scale = thumb_w / w
    thumb_h = int(h * scale)

    grid_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                vis = cv2.resize(frames[idx], (thumb_w, thumb_h))
                cv2.putText(vis, f"#{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                vis = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            row_imgs.append(vis)
        grid_rows.append(np.hstack(row_imgs))
    grid = _resize_to_fit(np.vstack(grid_rows))

    cv2.imshow(f"{title} ({n}) - press any key", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_frames_with_corners(
    frames: list[np.ndarray],
    img_points: list[np.ndarray],
    board_size: tuple[int, int],
):
    """Show a grid of frames with detected corners drawn on them."""
    n = len(frames)
    if n == 0:
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    h, w = frames[0].shape[:2]
    thumb_w = 480
    scale = thumb_w / w
    thumb_h = int(h * scale)

    grid_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                vis = frames[idx].copy()
                corners = img_points[idx].reshape(-1, 1, 2).astype(np.float32)
                cv2.drawChessboardCorners(vis, board_size, corners, True)
                vis = cv2.resize(vis, (thumb_w, thumb_h))
                cv2.putText(vis, f"#{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                vis = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            row_imgs.append(vis)
        grid_rows.append(np.hstack(row_imgs))
    grid = _resize_to_fit(np.vstack(grid_rows))

    cv2.imshow(f"Frames used for calibration ({n} total) - press any key", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_fisheye(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    detected_frames: list[np.ndarray],
    image_size: tuple[int, int],
    square_size: float,
    board_size: tuple[int, int],
) -> dict:
    """Run fisheye calibration, compute undistorted pinhole intrinsics for SplatSim."""
    # Scale object points by square size
    scaled_obj = [pts * square_size for pts in obj_points]

    w, h = image_size
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    # Iteratively drop ill-conditioned frames that cause calibration to fail
    while len(scaled_obj) >= 5:
        try:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                scaled_obj, img_points, image_size, K, D,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            break  # success
        except cv2.error as e:
            msg = str(e)
            if "CALIB_CHECK_COND" in msg and "input array" in msg:
                bad_idx = int(msg.split("input array")[1].split()[0])
                print(f"  Dropping ill-conditioned frame {bad_idx} ({len(scaled_obj)-1} remaining)")
                scaled_obj.pop(bad_idx)
                img_points.pop(bad_idx)
                detected_frames.pop(bad_idx)
                K = np.zeros((3, 3))
                D = np.zeros((4, 1))
            elif "InitExtrinsics" in msg or "norm_u1" in msg:
                # Fisheye model can't initialize — fall back to standard OpenCV calibration
                print(f"  Fisheye InitExtrinsics failed. Falling back to standard calibration...")
                # Reshape points from fisheye format [1, N, 3] to standard [N, 1, 3]
                std_obj = [o.reshape(-1, 3).astype(np.float32) for o in scaled_obj]
                std_img = [p.reshape(-1, 1, 2).astype(np.float32) for p in img_points]
                ret, K, dist_std, rvecs, tvecs = cv2.calibrateCamera(
                    std_obj, std_img, image_size, np.zeros((3, 3)), np.zeros(5),
                )
                # Convert standard 5-param distortion (k1,k2,p1,p2,k3) to fisheye 4-param (k1,k2,k3,k4)
                # This is approximate but gives a reasonable starting point
                ds = dist_std.ravel()
                D = np.array([[ds[0]], [ds[1]], [ds[4] if len(ds) > 4 else 0.0], [0.0]])
                print(f"  Standard calibration succeeded (RMS: {ret:.4f} px)")
                break
            else:
                raise
    else:
        raise RuntimeError(f"Too many ill-conditioned frames — only {len(scaled_obj)} left, need >= 5")

    # Compute per-frame reprojection error
    per_frame_errors = []
    for i in range(len(scaled_obj)):
        projected, _ = cv2.fisheye.projectPoints(
            scaled_obj[i].reshape(1, -1, 3), rvecs[i], tvecs[i], K, D,
        )
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / projected.shape[1]
        per_frame_errors.append(err)

    print(f"\nFisheye calibration complete!")
    print(f"  RMS reprojection error: {ret:.4f} px")
    print(f"  Mean per-frame error:   {np.mean(per_frame_errors):.4f} px")
    print(f"  Max per-frame error:    {np.max(per_frame_errors):.4f} px")
    print(f"  Image size: {w}x{h}")
    print(f"\nFisheye camera matrix K:\n{K}")
    print(f"\nFisheye distortion D (k1,k2,k3,k4): {D.ravel()}")

    # Compute optimal undistorted camera matrix (balance=0 keeps all valid pixels,
    # balance=1 keeps all source pixels). balance=0 is best for SplatSim since
    # it avoids black borders.
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=0.0,
    )

    fx_undist = new_K[0, 0]
    fy_undist = new_K[1, 1]
    cx_undist = new_K[0, 2]
    cy_undist = new_K[1, 2]

    FoVx = focal2fov(fx_undist, w)
    FoVy = focal2fov(fy_undist, h)

    print(f"\n--- Undistorted pinhole intrinsics (for SplatSim) ---")
    print(f"  Undistorted K:\n{new_K}")
    print(f"  resolution: [{w}, {h}]")
    print(f"  FoVx: {FoVx:.6f} rad ({math.degrees(FoVx):.2f} deg)")
    print(f"  FoVy: {FoVy:.6f} rad ({math.degrees(FoVy):.2f} deg)")
    print(f"  fx: {fx_undist:.2f}, fy: {fy_undist:.2f}")
    print(f"  cx: {cx_undist:.2f}, cy: {cy_undist:.2f}")

    # Visualize the frames that survived filtering with corners drawn
    print(f"\n  Showing {len(detected_frames)} frames used for calibration (press any key to close)...")
    show_frames_with_corners(detected_frames, img_points, board_size)

    return {
        # SplatSim Camera fields (from undistorted pinhole)
        "resolution": [w, h],
        "FoVx": FoVx,
        "FoVy": FoVy,
        # Undistorted pinhole intrinsics
        "undistorted_fx": float(fx_undist),
        "undistorted_fy": float(fy_undist),
        "undistorted_cx": float(cx_undist),
        "undistorted_cy": float(cy_undist),
        "undistorted_camera_matrix": new_K.tolist(),
        # Original fisheye intrinsics (needed for undistortion)
        "fisheye_camera_matrix": K.tolist(),
        "fisheye_dist_coeffs": D.ravel().tolist(),
        "fisheye_fx": float(K[0, 0]),
        "fisheye_fy": float(K[1, 1]),
        "fisheye_cx": float(K[0, 2]),
        "fisheye_cy": float(K[1, 2]),
        # Metadata
        "image_width": w,
        "image_height": h,
        "rms_reprojection_error": ret,
        "num_frames_used": len(obj_points),
        "square_size_mm": square_size,
    }


def save_results(results: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def undistort_frame(frame: np.ndarray, results: dict) -> np.ndarray:
    """Undistort a single frame using the fisheye calibration results."""
    K = np.array(results["fisheye_camera_matrix"])
    D = np.array(results["fisheye_dist_coeffs"])
    new_K = np.array(results["undistorted_camera_matrix"])
    h, w = frame.shape[:2]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2,
    )
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def print_splatsim_snippet(results: dict):
    """Print copy-pasteable code for gsplat fisheye and original pinhole pipelines."""
    w, h = results["resolution"]
    K = results["fisheye_camera_matrix"]
    D = results["fisheye_dist_coeffs"]

    print(f"""
========================================================
  gsplat fisheye rendering (paste into get_wrist_camera)
========================================================

# Fisheye intrinsic matrix K from calibration
fisheye_K = torch.tensor([
    [{K[0][0]:.4f}, {K[0][1]:.4f}, {K[0][2]:.4f}],
    [{K[1][0]:.4f}, {K[1][1]:.4f}, {K[1][2]:.4f}],
    [{K[2][0]:.4f}, {K[2][1]:.4f}, {K[2][2]:.4f}],
], dtype=torch.float32, device="cuda")

# Fisheye distortion coefficients (k1, k2, k3, k4)
fisheye_D = torch.tensor(
    [{D[0]:.10f}, {D[1]:.10f}, {D[2]:.10f}, {D[3]:.10f}],
    dtype=torch.float32, device="cuda",
)

splatsim_camera = SplatSimCamera(
    camera=camera,
    pipeline=self.base_camera.pipeline,
    background=self.base_camera.background,
    camera_model="fisheye",
    intrinsic_matrix=fisheye_K,
    radial_coeffs=fisheye_D,
)

========================================================
  Original pinhole pipeline (undistort images first)
========================================================

resolution = [{w}, {h}]
FoVx = {results['FoVx']:.10f}  # {math.degrees(results['FoVx']):.2f} deg
FoVy = {results['FoVy']:.10f}  # {math.degrees(results['FoVy']):.2f} deg
""")


def show_undistorted_comparison(video_path: str, results: dict):
    """Show a side-by-side of original vs undistorted for a middle frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    undistorted = undistort_frame(frame, results)

    # Scale down for display
    scale = 0.35
    orig_small = cv2.resize(frame, None, fx=scale, fy=scale)
    undist_small = cv2.resize(undistorted, None, fx=scale, fy=scale)
    combined = np.hstack([orig_small, undist_small])

    cv2.imshow("Original (left) vs Undistorted (right) - press any key", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Fisheye camera calibration from checkerboard video")
    parser.add_argument("--video", required=True, help="Path to checkerboard video")
    parser.add_argument("--board-size", nargs=2, type=int, default=[8, 6],
                        help="Inner corners of checkerboard (cols rows), default: 8, 6")
    parser.add_argument("--square-size", type=float, default=25.0,
                        help="Size of one square in mm (default: 25.0)")
    parser.add_argument("--num-frames", type=int, default=40,
                        help="Number of candidate frames to sample (default: 40)")
    parser.add_argument("--output", default="calibration.json",
                        help="Output JSON path (default: calibration.json)")
    parser.add_argument("--show", action="store_true",
                        help="Show undistorted comparison after calibration")
    args = parser.parse_args()

    video_path = args.video
    board_size = tuple(args.board_size)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Video: {video_path}")
    print(f"Board size (inner corners): {board_size[0]}x{board_size[1]}")
    print(f"Square size: {args.square_size} mm")
    print(f"Sampling {args.num_frames} candidate frames\n")

    frames = extract_candidate_frames(video_path, args.num_frames)
    show_frame_grid([f for _, f in frames], title="Sampled candidate frames")
    obj_points, img_points, detected_frames, image_size = detect_corners(frames, board_size)

    if len(obj_points) < 5:
        print(f"\nERROR: Only {len(obj_points)} frames with detected corners. Need at least 5.")
        print("Try adjusting --board-size or check that the video has a clear checkerboard.")
        return

    results = calibrate_fisheye(obj_points, img_points, detected_frames, image_size, args.square_size, board_size)
    save_results(results, args.output)
    print_splatsim_snippet(results)

    if args.show:
        show_undistorted_comparison(video_path, results)


if __name__ == "__main__":
    main()
