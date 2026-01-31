#!/usr/bin/env python3
"""
Script to view videos (sequences of images) from a parquet dataset.
Reads base_rgb or wrist_rgb columns and displays them as video.
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


def load_parquet_dataset(parquet_folder: str) -> pd.DataFrame:
    """Load all parquet files from a folder into a single DataFrame."""
    print(f"Loading parquet files from {parquet_folder}...")
    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")
    print(f"Found {len(parquet_files)} parquet files")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"Loaded {len(df)} total frames")
    return df


def decode_image(image_data: dict) -> np.ndarray:
    """Decode image from bytes stored in parquet."""
    img_bytes = image_data["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def get_image_column(df: pd.DataFrame, column_name: str) -> str:
    """Find the full column name for the requested image type."""
    # Look for columns containing the requested name
    image_columns = [c for c in df.columns if column_name in c.lower() and "image" in c.lower()]

    if not image_columns:
        # Try direct match
        if column_name in df.columns:
            return column_name
        # List available columns
        available = [c for c in df.columns if "image" in c.lower() or "rgb" in c.lower()]
        raise ValueError(f"Column '{column_name}' not found. Available image columns: {available}")

    return image_columns[0]


def generate_output_path(parquet_folder: str, episode_index: int | None) -> str:
    """Generate output video path based on parquet folder and episode."""
    # Extract dataset name and chunk from parquet folder path
    # e.g., /home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_1strrtpath_base/data/chunk-000
    parts = parquet_folder.rstrip("/").split("/")
    chunk_name = parts[-1]  # e.g., chunk-000
    dataset_name = parts[-3]  # e.g., splatsim_approach_lever_1strrtpath_base

    output_dir = "outputs/view_lerobot_parquet_video"
    ep_str = f"episode_{episode_index}" if episode_index is not None else "all_episodes"
    output_filename = f"{dataset_name}_{chunk_name}_{ep_str}.mp4"
    return os.path.join(output_dir, output_filename)


def save_video(df: pd.DataFrame, column_name: str, output_path: str, episode_index: int = None, fps: int = 30):
    """Save video from the specified column to a file."""
    # Filter by episode if specified
    if episode_index is not None:
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_index].copy()
            print(f"Filtered to episode {episode_index}: {len(df)} frames")
        else:
            print("Warning: episode_index column not found, using all frames")

    # Sort by frame index if available
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    elif "index" in df.columns:
        df = df.sort_values("index")

    # Get the actual column name
    col = get_image_column(df, column_name)
    print(f"Using column: {col}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving {len(df)} frames at {fps} FPS to {output_path}")

    # Use imageio for better codec support (H.264)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')

    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]
        img = decode_image(row[col])  # RGB format, which imageio expects

        writer.append_data(img)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(df)} frames")

    writer.close()
    print(f"Video saved to {output_path}")


def play_video(df: pd.DataFrame, column_name: str, episode_index: int = None, fps: int = 30):
    """Play video from the specified column."""
    # Filter by episode if specified
    if episode_index is not None:
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_index].copy()
            print(f"Filtered to episode {episode_index}: {len(df)} frames")
        else:
            print("Warning: episode_index column not found, showing all frames")

    # Sort by frame index if available
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    elif "index" in df.columns:
        df = df.sort_values("index")

    # Get the actual column name
    col = get_image_column(df, column_name)
    print(f"Using column: {col}")

    # Calculate delay between frames
    delay_ms = int(1000 / fps)

    window_name = f"Video: {col} (Episode {episode_index if episode_index is not None else 'all'})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Playing {len(df)} frames at {fps} FPS")
    print("Press 'space' to start, 'q' to quit, 'space' to pause/resume, 'n' for next frame when paused")

    paused = True  # Start paused
    frame_idx = 0

    while frame_idx < len(df):
        row = df.iloc[frame_idx]
        img = decode_image(row[col])

        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('n') and paused:
            frame_idx += 1
            continue

        if not paused:
            frame_idx += 1

    cv2.destroyAllWindows()


def list_episodes(df: pd.DataFrame):
    """List available episodes in the dataset."""
    if "episode_index" in df.columns:
        episodes = df["episode_index"].unique()
        print(f"Available episodes: {sorted(episodes)}")
        for ep in sorted(episodes):
            ep_df = df[df["episode_index"] == ep]
            print(f"  Episode {ep}: {len(ep_df)} frames")
    else:
        print("No episode_index column found")


def main():
    parser = argparse.ArgumentParser(description="View videos from parquet dataset")
    parser.add_argument(
        "--parquet_folder",
        type=str,
        default="/home/jennyw2/.cache/huggingface/lerobot/JennyWWW/splatsim_approach_lever_1strrtpath_base/data/chunk-000",
        help="Path to folder containing parquet files",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="base_rgb",
        help="Image column to display (e.g., 'base_rgb' or 'wrist_rgb')",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to display (default: all episodes)",
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
        "--save_video",
        action="store_true",
        help="Save video to file instead of displaying interactively",
    )

    args = parser.parse_args()

    df = load_parquet_dataset(args.parquet_folder)

    if args.list_episodes:
        list_episodes(df)
        return

    if args.save_video:
        output_path = generate_output_path(args.parquet_folder, args.episode)
        save_video(df, args.column, output_path, args.episode, args.fps)
    else:
        play_video(df, args.column, args.episode, args.fps)


if __name__ == "__main__":
    main()
