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
    num_frames = min(num_frames, total_frames)
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
            # Store a thumbnail — full-res frames (~16 MB each) are only needed for display
            thumb_w = 480
            thumb_h = int(frame.shape[0] * thumb_w / frame.shape[1])
            detected_frames.append(cv2.resize(frame, (thumb_w, thumb_h)))
            used_indices.append(frame_idx)

    print(f"Checkerboard detected in {len(obj_points)}/{len(frames)} candidate frames")
    print(f"  Frame indices used: {used_indices}")
    return obj_points, img_points, detected_frames, image_size


def save_corners_cache(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    detected_frames: list[np.ndarray],
    image_size: tuple[int, int],
    cache_dir: Path,
):
    """Save corner detection results to <cache_dir>/corners.npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "corners.npz",
        obj_points=np.stack(obj_points),
        img_points=np.stack(img_points),
        detected_frames=np.stack(detected_frames),
        image_size=np.array(image_size),
    )
    print(f"Corner detection results cached to {cache_dir / 'corners.npz'}")


def load_corners_cache(
    cache_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Load corner detection results from <cache_dir>/corners.npz."""
    npz_path = cache_dir / "corners.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache not found: {npz_path}")
    data = np.load(npz_path)
    obj_points = list(data["obj_points"])
    img_points = list(data["img_points"])
    detected_frames = list(data["detected_frames"])
    image_size = tuple(data["image_size"].tolist())
    print(f"Loaded corner detection results from {npz_path}")
    print(f"  {len(obj_points)} frames, image size: {image_size}")
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


MAX_GRID_FRAMES = 50


def show_frame_grid(frames: list[np.ndarray], title: str = "Frames", save_path: str | None = None):
    """Show a grid of frames (no corners). Capped at MAX_GRID_FRAMES. Optionally save to save_path."""
    frames = frames[:MAX_GRID_FRAMES]
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

    if save_path is not None:
        cv2.imwrite(save_path, grid)
        print(f"  Saved frame grid to {save_path}")

    cv2.imshow(f"{title} ({n}) - press any key", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_frames_with_corners(
    frames: list[np.ndarray],
    img_points: list[np.ndarray],
    board_size: tuple[int, int],
    image_size: tuple[int, int],
    save_path: str | None = None
):
    """Show a grid of frames with detected corners drawn on them.

    frames are thumbnails; image_size is the original full-res (w, h) used to
    scale corner coordinates before drawing.
    """
    n = len(frames)
    if n == 0:
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    thumb_h, thumb_w = frames[0].shape[:2]
    orig_w, orig_h = image_size
    scale_x = thumb_w / orig_w
    scale_y = thumb_h / orig_h

    grid_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                vis = frames[idx].copy()
                corners = img_points[idx].reshape(-1, 1, 2).astype(np.float32)
                corners = corners * np.array([[[scale_x, scale_y]]], dtype=np.float32)
                cv2.drawChessboardCorners(vis, board_size, corners, True)
                cv2.putText(vis, f"#{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                vis = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            row_imgs.append(vis)
        grid_rows.append(np.hstack(row_imgs))
    grid = _resize_to_fit(np.vstack(grid_rows))

    cv2.imshow(f"Frames used for calibration ({n} total) - press any key", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the grid with corners drawn
    if save_path is not None:
        cv2.imwrite(save_path, grid)
        print(f"  Saved detected frames with corners grid to {save_path}")


def _passes_solvepnp(obj_pts: np.ndarray, img_pts: np.ndarray, image_size: tuple[int, int]) -> bool:
    """Return True if solvePnP can find a pose for this frame (quick degenerate-frame check)."""
    w, h = image_size
    f = float(max(w, h))
    K_dummy = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    obj = obj_pts.reshape(-1, 3).astype(np.float32)
    img = img_pts.reshape(-1, 1, 2).astype(np.float32)
    success, _, _ = cv2.solvePnP(obj, img, K_dummy, None)
    return bool(success)


def calibrate_fisheye(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    detected_frames: list[np.ndarray],
    image_size: tuple[int, int],
    square_size: float,
    board_size: tuple[int, int],
    cache_dir: Path,
) -> dict:
    """Run fisheye calibration, compute undistorted pinhole intrinsics for SplatSim."""
    # Scale object points by square size
    scaled_obj = [pts * square_size for pts in obj_points]

    # Pre-filter frames whose geometry is degenerate (collinear points, bad detections, etc.)
    # solvePnP is a cheap sanity check — if it can't solve the pose, fisheye won't either.
    valid = [_passes_solvepnp(o, p, image_size) for o, p in zip(scaled_obj, img_points)]
    n_removed = valid.count(False)
    if n_removed:
        print(f"  Pre-filter: removing {n_removed} degenerate frames that fail solvePnP ({sum(valid)} remaining)")
        scaled_obj      = [x for x, v in zip(scaled_obj, valid) if v]
        img_points      = [x for x, v in zip(img_points, valid) if v]
        detected_frames = [x for x, v in zip(detected_frames, valid) if v]

    w, h = image_size
    print("image size:", image_size)

    # Initial guess scaled from a similar camera/lens at 960x540:
    #   f=435.45, cx=479.12, cy=274.46, D=[0.05, 0.07, -0.11, 0.05]
    # Scale: fx/fy by (target/source) per axis; cx/cy by same; D is dimensionless.
    # Initial guess scaled from a similar camera/lens at 960x540:
    #   f=435.45, cx=479.12, cy=274.46, D=[0.05, 0.07, -0.11, 0.05]
    # fx=fy (square pixels; use width scale factor: 2704/960 ≈ 2.817).
    # cx/cy scaled per axis. D is dimensionless — no scaling needed.
    K = np.array([
        [776.0756, 0.0000, 1343.6819],
        [0.0000, 778.6178, 1006.0610],
        [0.0000, 0.0000, 1.0000],
    ], dtype=np.float64)
    D = np.array([-0.0228169664, -0.0162816270, 0.0000000000, 0.0000000000], dtype=np.float64)
    K_init = K.copy()
    D_init = D.copy()

    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        # + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        # Fix k3/k4 to prevent overfitting when frame count is low.
        # Remove these once you have 40+ well-distributed frames.
        + cv2.fisheye.CALIB_FIX_K3
        + cv2.fisheye.CALIB_FIX_K4
    )

    # Save pre-filtered copies — the while loop mutates these lists,
    # so keep originals in case we need the standard-calibration fallback.
    scaled_obj_prefiltered = list(scaled_obj)
    img_points_prefiltered = list(img_points)

    # Iteratively drop ill-conditioned  frames that cause calibration to fail.
    # Uses explicit calibration_succeeded flag so fallback logic is centralised below.
    calibration_succeeded = False
    used_standard_fallback = False
    dist_std = None
    init_extrinsics_drops = 0
    ret, rvecs, tvecs = 0.0, [], []
    # If InitExtrinsics keeps failing we're dropping random frames without knowing which is
    # bad; bail out to the standard fallback once we've exhausted this many attempts.
    max_init_extrinsics_drops = float("inf") # max(5, len(scaled_obj_prefiltered) // 3)

    show_frames_with_corners(detected_frames, img_points, board_size, image_size, save_path=str(cache_dir / "detected_frames_with_corners_grid_prefilter.png"))


    while len(scaled_obj) >= 5:
        try:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                scaled_obj, img_points, image_size, K, D,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            calibration_succeeded = True
            break
        except cv2.error as e:
            msg = str(e)
            if "CALIB_CHECK_COND" in msg and "input array" in msg:
                bad_idx = int(msg.split("input array")[1].split()[0])
                print(f"  Dropping ill-conditioned frame {bad_idx} ({len(scaled_obj)-1} remaining)")
                scaled_obj.pop(bad_idx)
                img_points.pop(bad_idx)
                detected_frames.pop(bad_idx)
                K = K_init.copy()
                D = D_init.copy()
            elif "InitExtrinsics" in msg or "norm_u1" in msg:
                init_extrinsics_drops += 1
                if init_extrinsics_drops > max_init_extrinsics_drops:
                    print(f"  InitExtrinsics still failing after {init_extrinsics_drops} drops — "
                          f"fisheye model can't initialize on this data, falling back to standard")
                    break
                print(f"  InitExtrinsics failure — dropping last frame ({len(scaled_obj)-1} remaining)")
                scaled_obj.pop(-1)
                img_points.pop(-1)
                detected_frames.pop(-1)
                K = K_init.copy()
                D = D_init.copy()
            else:
                raise

    if not calibration_succeeded:
        raise RuntimeError("Fisheye calibration failed after filtering frames. Try checking the video quality.")

    # Compute per-frame mean reprojection error (mean Euclidean distance per point)
    per_frame_errors = []
    for i in range(len(scaled_obj)):
        if used_standard_fallback:
            std_obj_i = scaled_obj[i].reshape(-1, 3).astype(np.float32)
            projected, _ = cv2.projectPoints(std_obj_i, rvecs[i], tvecs[i], K, dist_std)
            pts = img_points[i].reshape(-1, 2).astype(np.float32)
        else:
            projected, _ = cv2.fisheye.projectPoints(
                scaled_obj[i].reshape(1, -1, 3), rvecs[i], tvecs[i], K, D,
            )
            pts = img_points[i].reshape(-1, 2).astype(projected.dtype)
        proj_flat = projected.reshape(-1, 2)
        err = float(np.linalg.norm(pts - proj_flat, axis=1).mean())
        per_frame_errors.append(err)

    sorted_frame_errors = sorted(enumerate(per_frame_errors), key=lambda x: -x[1])

    print(f"\nFisheye calibration complete!")
    print(f"  RMS reprojection error: {ret:.4f} px")
    print(f"  Mean per-frame error:   {np.mean(per_frame_errors):.4f} px")
    print(f"  Max per-frame error:    {np.max(per_frame_errors):.4f} px")
    print(f"  Per-frame errors (worst first):")
    for frame_i, frame_err in sorted_frame_errors:
        print(f"    frame {frame_i:3d}: {frame_err:.3f} px")
    print(f"  Image size: {w}x{h}")
    print(f"\nFisheye camera matrix K:\n{K}")
    print(f"\nFisheye distortion D (k1,k2,k3,k4): {D.ravel()}")

    # Compute optimal undistorted camera matrix (balance=0 crops to valid pixels only,
    # balance=1 keeps all source pixels with black borders).
    # balance=0 can return NaN when distortion coefficients are large/unstable;
    # fall back to balance=1 in that case.
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=0.0,
    )
    # Sanity check: fx should be in the same ballpark as the fisheye focal length.
    # With near-zero or negative-only D, the function can return a degenerate result
    # (fx << 1) without triggering NaN. Fall back to the fisheye K in that case,
    # which is valid because distortion is negligible.
    if np.any(np.isnan(new_K)) or new_K[0, 0] < K[0, 0] * 0.1:
        print("  WARNING: estimateNewCameraMatrixForUndistortRectify returned a degenerate result "
              f"(fx={new_K[0,0]:.4f}). Falling back to fisheye K as undistorted K "
              "(valid: distortion coefficients are near-zero).")
        new_K = K.copy()

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
    show_frames_with_corners(detected_frames, img_points, board_size, image_size, save_path=str(cache_dir / "detected_frames_with_corners_grid_postfilter.png"))
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
        "square_size_m": square_size,
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


def show_undistorted_comparison(frame: np.ndarray, results: dict, save_path: str | None = None):
    """Show a side-by-side of original vs undistorted for a middle frame."""

    undistorted = undistort_frame(frame, results)

    # Scale down for display
    scale = 0.35
    orig_small = cv2.resize(frame, None, fx=scale, fy=scale)
    undist_small = cv2.resize(undistorted, None, fx=scale, fy=scale)

    show_frame_grid(
        [orig_small, undist_small],
        title="Original (left) vs Undistorted (right) - press any key",
        save_path=save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Fisheye camera calibration from checkerboard video")
    parser.add_argument("--video", help="Path to checkerboard video", required=False)
    parser.add_argument("--image_folder", help="Path to folder of checkerboard images (alternative to --video)", required=False)
    parser.add_argument("--load-cache", action="store_true",
                        help="Load corner detection results from cache instead of processing video")
    parser.add_argument("--board-size", nargs=2, type=int, default=[8, 6],
                        help="Inner corners of checkerboard (cols rows), default: 8, 6")
    parser.add_argument("--square-size", type=float, default=0.0425,
                        help="Size of one square in meters (default: 0.0425)")
    parser.add_argument("--num-frames", type=int, default=40,
                        help="Number of candidate frames to sample (default: 40)")
    parser.add_argument("--output", default="calibration.json",
                        help="Output JSON path (default: calibration.json)")
    args = parser.parse_args()

    if not args.video and not args.image_folder:
        parser.error("--video or --image_folder is required")

    video_path = args.video
    image_folder = args.image_folder
    if video_path is not None:
        print(f"Video path: {video_path}")
        video_p = Path(video_path)
        base_folder = video_p.parent
        cache_dir = video_p.parent / f"{video_p.stem}_calibration_cache"
    elif image_folder is not None:
        print(f"Image folder: {image_folder}")
        base_folder = Path(image_folder)
        cache_dir = Path(image_folder) / "calibration_cache"
    else:
        raise ValueError("Unexpected error: neither video nor image folder provided")
    board_size = tuple(args.board_size)


    if args.load_cache:
        print(f"Loading corner detection results from cache: {cache_dir}\n")
        obj_points, img_points, detected_frames, image_size = load_corners_cache(cache_dir)
        cached_grid = cache_dir / "detected_frames_grid.png"
        if cached_grid.exists():
            grid_img = cv2.imread(str(cached_grid))
            if grid_img is not None:
                cv2.imshow(f"Filtered candidate frames w/ corners ({len(detected_frames)}) - press any key", grid_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        if video_path:
            if not video_p.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            print(f"Video: {video_path}")
        if image_folder:
            img_folder_p = Path(image_folder)
            if not img_folder_p.exists() or not img_folder_p.is_dir():
                raise FileNotFoundError(f"Image folder not found or not a directory: {image_folder}")
            print(f"Image folder: {image_folder}")
        print(f"Board size (inner corners): {board_size[0]}x{board_size[1]}")
        print(f"Square size: {args.square_size} m")
        print(f"Sampling {args.num_frames} candidate frames\n")

        if video_path:
            print("Extracting candidate frames from video...")
            frames = extract_candidate_frames(video_path, args.num_frames)
            show_frame_grid(
                [f for _, f in frames],
                title="Sampled candidate frames",
                save_path=str(cache_dir / "candidate_frames_grid.png"),
            )
        elif image_folder:
            print(f"Loading checkerboard images from folder: {image_folder}")
            image_paths = sorted(Path(image_folder).glob("*"))
            frames = []
            for idx, img_path in enumerate(image_paths):
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append((idx, img))
            if not frames:
                print(f"No valid images found in {image_folder}")
                return
            print(f"Loaded {len(frames)} images from {image_folder}")
            
        print("\nDetecting corners in candidate frames...")
        obj_points, img_points, detected_frames, image_size = detect_corners(frames, board_size)
        del frames  # free ~1.6 GB of full-res candidate frames before caching
        print("\nSaving corner detection results to cache...")
        save_corners_cache(obj_points, img_points, detected_frames, image_size, cache_dir)
        show_frame_grid(
            detected_frames,
            title="Filtered candidate frames w/ corners",
            save_path=str(cache_dir / "detected_frames_grid.png"),
        )

    if len(obj_points) < 5:
        print(f"\nERROR: Only {len(obj_points)} frames with detected corners. Need at least 5.")
        print("Try adjusting --board-size or check that the video has a clear checkerboard.")
        return

    print("\nCalibrating fisheye camera intrinsics...")
    results = calibrate_fisheye(obj_points, img_points, detected_frames, image_size, args.square_size, board_size, cache_dir)
    save_results(results, args.output)
    print_splatsim_snippet(results)

    print("\nShowing undistorted comparison (press any key to close)...")
    # Get a middle frame for undistortion comparison
    if video_path:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame {mid_frame_idx} from video for undistortion comparison")
            return
    elif image_folder:
        # Pick an image from the image folder (e.g. the middle one) for undistortion comparison
        image_paths = sorted(Path(image_folder).glob("*"))
        if not image_paths:
            print(f"No valid images found in {image_folder} for undistortion comparison")
            return
        mid_image_path = image_paths[len(image_paths) // 2]
        frame = cv2.imread(str(mid_image_path))
        if frame is None:
            print(f"Failed to read image {mid_image_path} for undistortion comparison")
            return
    else:
        print("Unexpected error: no video or image folder provided for undistortion comparison")
        return
    show_undistorted_comparison(frame, results, save_path=str(cache_dir / "undistorted_comparison.png"))


if __name__ == "__main__":
    main()
