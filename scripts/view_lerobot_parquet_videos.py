#!/usr/bin/env python3
"""
Script to view videos (sequences of images) from a parquet dataset.
Finds all base_rgb* and wrist_rgb* columns and displays them as a collage.
"""

import argparse
import glob
import io
import os

import cv2
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lerobot_parquet_utils import parse_episodes

# Label bar height in pixels drawn above each image tile
_LABEL_H = 24


def load_parquet_dataset(parquet_folder: str, episodes: list[int] | None = None) -> pd.DataFrame:
    """Load parquet files from a folder, optionally filtering to specific episodes.

    Uses parquet predicate pushdown so only matching row groups are read into memory.
    """
    print(f"Loading parquet files from {parquet_folder}...")
    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")
    print(f"Found {len(parquet_files)} parquet files")

    filters = [("episode_index", "in", episodes)] if episodes is not None else None
    dfs = []
    for f in parquet_files:
        chunk = pd.read_parquet(f, filters=filters)
        if not chunk.empty:
            dfs.append(chunk)
    if not dfs:
        raise ValueError(f"No frames found for episodes {episodes}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} frames")
    return df


def load_parquet_episode_index(parquet_folder: str) -> list[int]:
    """Return sorted unique episode indices without loading image data."""
    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")
    episodes: set[int] = set()
    for f in parquet_files:
        table = pq.read_table(f, columns=["episode_index"])
        episodes.update(table.column("episode_index").to_pylist())
    return sorted(episodes)


def decode_image(image_data: dict) -> np.ndarray:
    """Decode image from bytes stored in parquet."""
    img_bytes = image_data["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def find_image_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (base_cols, wrist_cols) — all columns matching base_rgb* and wrist_rgb*, sorted."""
    base_cols = sorted(c for c in df.columns if "base_rgb" in c.lower())
    wrist_cols = sorted(c for c in df.columns if "wrist_rgb" in c.lower())
    return base_cols, wrist_cols


def _labeled_tile(img_rgb: np.ndarray, label: str) -> np.ndarray:
    """Return a BGR tile: label bar on top, image (original size) below."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Draw label bar
    bar = np.zeros((_LABEL_H, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, _LABEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.concatenate([bar, img_bgr], axis=0)


def make_collage(row: pd.Series, base_cols: list[str], wrist_cols: list[str]) -> np.ndarray:
    """
    Build a collage with base_rgb* on the top row and wrist_rgb* on the bottom row.
    Each tile has a label bar. Tiles/rows are padded with black if sizes differ.
    """
    def build_row(cols: list[str]) -> np.ndarray | None:
        tiles = []
        for col in cols:
            img = decode_image(row[col])
            label = col.removeprefix("observation.images.")
            tiles.append(_labeled_tile(img, label))
        if not tiles:
            return None
        # Pad tiles to the same height before concatenating horizontally
        max_h = max(t.shape[0] for t in tiles)
        padded = []
        for t in tiles:
            if t.shape[0] < max_h:
                pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
                t = np.concatenate([t, pad], axis=0)
            padded.append(t)
        return np.concatenate(padded, axis=1)

    base_row = build_row(base_cols)
    wrist_row = build_row(wrist_cols)

    if base_row is None and wrist_row is None:
        raise ValueError("No image columns found")
    if base_row is None:
        return wrist_row  # type: ignore[return-value]
    if wrist_row is None:
        return base_row

    # Pad narrower row to match the wider one
    w_base, w_wrist = base_row.shape[1], wrist_row.shape[1]
    if w_base < w_wrist:
        pad = np.zeros((base_row.shape[0], w_wrist - w_base, 3), dtype=np.uint8)
        base_row = np.concatenate([base_row, pad], axis=1)
    elif w_wrist < w_base:
        pad = np.zeros((wrist_row.shape[0], w_base - w_wrist, 3), dtype=np.uint8)
        wrist_row = np.concatenate([wrist_row, pad], axis=1)

    return np.concatenate([base_row, wrist_row], axis=0)


def generate_output_path(parquet_folder: str, episode_str: str | None) -> str:
    """Generate output video path based on parquet folder and episode."""
    parts = parquet_folder.rstrip("/").split("/")
    chunk_name = parts[-1]
    dataset_name = parts[-3]

    output_dir = "outputs/view_lerobot_parquet_video"
    ep_str = f"episode_{episode_str}" if episode_str is not None else "all_episodes"
    output_filename = f"{dataset_name}_{chunk_name}_{ep_str}.mp4"
    return os.path.join(output_dir, output_filename)



def _prepare_df(df: pd.DataFrame, episodes: list[int] | None = None) -> pd.DataFrame:
    """Filter by episodes and sort by (episode_index, frame_index)."""
    if episodes is not None:
        if "episode_index" in df.columns:
            df = df[df["episode_index"].isin(episodes)].copy()
            print(f"Filtered to episodes {episodes}: {len(df)} frames")
        else:
            print("Warning: episode_index column not found, using all frames")

    sort_cols = [c for c in ["episode_index", "frame_index"] if c in df.columns]
    if not sort_cols and "index" in df.columns:
        sort_cols = ["index"]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df


def save_video(df: pd.DataFrame, output_path: str, episodes: list[int] | None = None, fps: int = 30):
    """Save collage video of all base_rgb* and wrist_rgb* columns to a file."""
    df = _prepare_df(df, episodes)

    base_cols, wrist_cols = find_image_columns(df)
    print(f"Base columns: {base_cols}")
    print(f"Wrist columns: {wrist_cols}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving {len(df)} frames at {fps} FPS to {output_path}")

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')

    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]
        collage_bgr = make_collage(row, base_cols, wrist_cols)
        collage_rgb = cv2.cvtColor(collage_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(collage_rgb)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(df)} frames")

    writer.close()
    print(f"Video saved to {output_path}")


def play_video(df: pd.DataFrame, episodes: list[int] | None = None, fps: int = 30):
    """Play collage of all base_rgb* and wrist_rgb* columns."""
    df = _prepare_df(df, episodes)

    base_cols, wrist_cols = find_image_columns(df)
    print(f"Base columns: {base_cols}")
    print(f"Wrist columns: {wrist_cols}")

    delay_ms = int(1000 / fps)
    ep_label = str(episodes) if episodes is not None else "all"
    window_name = f"base_rgb* | wrist_rgb* (episodes: {ep_label})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Playing {len(df)} frames at {fps} FPS")
    print("Press 'space' to start/pause, 'q' to quit, 'n' for next frame when paused")

    paused = True
    frame_idx = 0
    window_sized = False

    has_episode_col = "episode_index" in df.columns

    # Build per-episode frame ranges for the episode progress bar
    if has_episode_col:
        episode_list = sorted(df["episode_index"].unique())
    else:
        episode_list = [0]

    # Precompute episode frame counts for resetting the frame bar per episode
    if has_episode_col:
        ep_frame_counts = {ep: int((df["episode_index"] == ep).sum()) for ep in episode_list}
    else:
        ep_frame_counts = {0: len(df)}

    ep_pbar = tqdm(episode_list, desc="Episode", position=0, leave=True)
    frame_pbar = tqdm(total=ep_frame_counts[episode_list[0]], desc="Frame  ", position=1, leave=True)

    current_ep = None

    while frame_idx < len(df):
        row = df.iloc[frame_idx]
        collage = make_collage(row, base_cols, wrist_cols)

        if not window_sized:
            cv2.resizeWindow(window_name, collage.shape[1], collage.shape[0])
            window_sized = True

        # Reset frame bar when episode changes
        if has_episode_col:
            ep_num = int(row["episode_index"])
            if ep_num != current_ep:
                if current_ep is not None:
                    ep_pbar.update(1)
                current_ep = ep_num
                frame_pbar.reset(total=ep_frame_counts[ep_num])

        cv2.imshow(window_name, collage)

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('n') and paused:
            frame_idx += 1
            frame_pbar.update(1)
            continue

        if not paused:
            frame_idx += 1
            frame_pbar.update(1)

    # Finalize progress bars
    ep_pbar.update(1)  # count last episode
    ep_pbar.close()
    frame_pbar.close()

    cv2.destroyAllWindows()


def episode_lengths(parquet_folder: str, subset: list[int] | None = None):
    """Print average episode length across all episodes and optionally a subset."""
    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")

    ep_counts: dict[int, int] = {}
    for f in parquet_files:
        table = pq.read_table(f, columns=["episode_index"])
        for ep in table.column("episode_index").to_pylist():
            ep_counts[ep] = ep_counts.get(ep, 0) + 1

    all_lengths = [ep_counts[ep] for ep in sorted(ep_counts)]
    print(f"Total episodes: {len(all_lengths)}")
    print(f"Avg episode length (all): {np.mean(all_lengths):.1f} frames  (min={min(all_lengths)}, max={max(all_lengths)})")

    if subset is not None:
        missing = [ep for ep in subset if ep not in ep_counts]
        if missing:
            print(f"Warning: episodes not found in dataset: {missing}")
        subset_lengths = [ep_counts[ep] for ep in subset if ep in ep_counts]
        if subset_lengths:
            print(f"Avg episode length (subset of {len(subset_lengths)} episodes): {np.mean(subset_lengths):.1f} frames  (min={min(subset_lengths)}, max={max(subset_lengths)})")


def list_episodes(parquet_folder: str):
    """List available episodes without loading image data."""
    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")

    ep_counts: dict[int, int] = {}
    for f in parquet_files:
        table = pq.read_table(f, columns=["episode_index"])
        for ep in table.column("episode_index").to_pylist():
            ep_counts[ep] = ep_counts.get(ep, 0) + 1

    print(f"Available episodes: {sorted(ep_counts)}")
    for ep in sorted(ep_counts):
        print(f"  Episode {ep}: {ep_counts[ep]} frames")


def main():
    parser = argparse.ArgumentParser(description="View all base_rgb* and wrist_rgb* columns as a collage from a parquet dataset")
    parser.add_argument(
        "--parquet_folder",
        type=str,
        default="/home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_1strrtpath_base/data/chunk-000",
        help="Path to folder containing parquet files",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episodes to display: single int '3', range '0-5', or comma-separated '0,2,4' or '0-3,7' (default: all)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frames per second (default: 30)",
    )

    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="List available episodes and exit",
    )
    parser.add_argument(
        "--episode-lengths",
        action="store_true",
        help="Print average episode length for all episodes and optionally --episode subset, then exit",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video to file instead of displaying interactively",
    )

    args = parser.parse_args()

    if args.list_episodes:
        list_episodes(args.parquet_folder)
        return

    episodes = parse_episodes(args.episode) if args.episode is not None else None

    if args.episode_lengths:
        episode_lengths(args.parquet_folder, subset=episodes)
        return

    df = load_parquet_dataset(args.parquet_folder, episodes)

    if args.save_video:
        output_path = generate_output_path(args.parquet_folder, args.episode)
        save_video(df, output_path, episodes, args.fps)
    else:
        play_video(df, episodes, args.fps)


if __name__ == "__main__":
    main()
