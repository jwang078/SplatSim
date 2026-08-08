#!/usr/bin/env python3
"""
Script to view videos (sequences of images) from a LeRobot dataset.

Supports both storage layouts:
  * image-in-parquet: base_rgb*/wrist_rgb* columns hold encoded PNG/JPEG bytes.
  * video-backed (LeRobot v2.x/v3.0): the parquet holds only state/action, and
    frames live in mp4 files under <root>/videos/. The mp4s are located via
    meta/info.json + meta/episodes/*.parquet and decoded on demand.
"""

import argparse
import glob
import io
import json
import os
from collections import OrderedDict

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
    skipped = []
    for f in parquet_files:
        try:
            chunk = pd.read_parquet(f, filters=filters)
        except Exception as e:
            # A live recording leaves its NEWEST file without a parquet footer
            # ("magic bytes not found in footer") until the round finalizes — a
            # truncated/partial write looks identical. Skip it with a warning
            # instead of crashing, so the finalized files (and the episode you
            # asked for, if it's in one of them) still load.
            skipped.append((os.path.basename(f), f"{type(e).__name__}: {e}"))
            continue
        if not chunk.empty:
            dfs.append(chunk)
    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable parquet file(s) "
              f"(likely an in-progress/partial write):")
        for name, err in skipped:
            print(f"  {name}: {err}")
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
        try:
            table = pq.read_table(f, columns=["episode_index"])
        except Exception as e:
            # Same in-progress/partial-write tolerance as load_parquet_dataset.
            print(f"WARNING: skipping unreadable parquet {os.path.basename(f)}: "
                  f"{type(e).__name__}: {e}")
            continue
        episodes.update(table.column("episode_index").to_pylist())
    return sorted(episodes)


def decode_image(image_data: dict) -> np.ndarray:
    """Decode image from bytes stored in parquet."""
    img_bytes = image_data["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def find_dataset_root(parquet_folder: str) -> str | None:
    """Walk up from a data/chunk-XXX folder to the dataset root (the dir with meta/info.json)."""
    path = os.path.abspath(parquet_folder)
    for _ in range(4):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "meta", "info.json")):
            return path
    return None


class VideoFrameSource:
    """Decodes frames of video-backed LeRobot datasets (dtype == "video" features).

    v3.0 packs many episodes into one mp4 per camera; meta/episodes/*.parquet says
    which file an episode lives in and at what timestamp it starts. v2.x writes one
    mp4 per (camera, episode), which we detect from the {episode_index} placeholder
    in info.json's video_path template.

    Frames are decoded a whole episode at a time (sequential decode is ~1000x faster
    than seeking per frame) and the last few episodes are kept in an LRU cache so
    stepping backwards / replaying doesn't re-decode.
    """

    def __init__(self, dataset_root: str, max_cached_episodes: int = 8):
        self.root = dataset_root
        with open(os.path.join(dataset_root, "meta", "info.json")) as f:
            info = json.load(f)
        self.info = info
        self.video_path_tmpl = info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self.keys = sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") == "video")
        self.per_episode_files = "{episode_index" in self.video_path_tmpl
        self._ep_meta = None if self.per_episode_files else self._load_episode_meta()
        self._cache: OrderedDict[tuple[str, int], list[np.ndarray]] = OrderedDict()
        self._max_cached = max_cached_episodes
        self._containers: dict[str, object] = {}

    def _load_episode_meta(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(
                f"Video-backed dataset at {self.root} has no meta/episodes/*.parquet — "
                "cannot locate episode segments inside the mp4 files."
            )
        cols_needed = ["episode_index", "length"] + [
            f"videos/{k}/{sfx}" for k in self.keys for sfx in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ]
        dfs = []
        for f in files:
            df = pd.read_parquet(f, columns=None)
            dfs.append(df[[c for c in cols_needed if c in df.columns]])
        meta = pd.concat(dfs, ignore_index=True).set_index("episode_index")
        return meta

    def _locate(self, key: str, ep: int) -> tuple[str, float, int | None]:
        """Return (mp4 path, start timestamp in that file, expected frame count)."""
        if self.per_episode_files:
            chunk = ep // int(self.info.get("chunks_size", 1000))
            rel = self.video_path_tmpl.format(
                video_key=key, episode_chunk=chunk, episode_index=ep, chunk_index=chunk, file_index=ep
            )
            return os.path.join(self.root, rel), 0.0, None

        if ep not in self._ep_meta.index:
            raise KeyError(f"Episode {ep} not present in meta/episodes")
        row = self._ep_meta.loc[ep]
        rel = self.video_path_tmpl.format(
            video_key=key,
            chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]),
        )
        length = int(row["length"]) if "length" in row.index else None
        return os.path.join(self.root, rel), float(row[f"videos/{key}/from_timestamp"]), length

    def _container(self, path: str):
        import av

        if path not in self._containers:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Video file not found: {path}")
            container = av.open(path)
            # Must be set before the codec opens (i.e. before the first decode).
            container.streams.video[0].thread_type = "AUTO"
            self._containers[path] = container
        return self._containers[path]

    def _decode_episode(self, key: str, ep: int) -> list[np.ndarray]:
        path, from_ts, length = self._locate(key, ep)
        container = self._container(path)
        stream = container.streams.video[0]
        # Seek to the keyframe at/just before the episode start, then drop the
        # pre-roll frames by timestamp.
        container.seek(int(from_ts / stream.time_base), stream=stream, backward=True)
        eps = 1e-4
        frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            if float(frame.pts * stream.time_base) < from_ts - eps:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if length is not None and len(frames) >= length:
                break
        if not frames:
            raise RuntimeError(f"Decoded 0 frames for {key} episode {ep} from {path}")
        return frames

    def get_frame(self, key: str, ep: int, frame_in_ep: int) -> np.ndarray:
        cache_key = (key, ep)
        frames = self._cache.get(cache_key)
        if frames is None:
            frames = self._decode_episode(key, ep)
            self._cache[cache_key] = frames
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(cache_key)
        # Clamp: an episode's last frame can be missing if the mp4 was truncated.
        return frames[min(frame_in_ep, len(frames) - 1)]


def find_image_columns(
    df: pd.DataFrame, video_source: VideoFrameSource | None = None, key_filter: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Return (base_cols, wrist_cols) — all image keys matching base_rgb* and wrist_rgb*, sorted.

    Keys come from parquet columns holding encoded images, plus (for video-backed
    datasets) the video feature keys of `video_source`.
    """
    names = list(df.columns) + (list(video_source.keys) if video_source is not None else [])
    if key_filter:
        names = [n for n in names if any(f.lower() in n.lower() for f in key_filter)]
    base_cols = sorted(set(c for c in names if "base_rgb" in c.lower()))
    wrist_cols = sorted(set(c for c in names if "wrist_rgb" in c.lower()))
    return base_cols, wrist_cols


def _labeled_tile(img_rgb: np.ndarray, label: str) -> np.ndarray:
    """Return a BGR tile: label bar on top, image (original size) below."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Draw label bar
    bar = np.zeros((_LABEL_H, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, _LABEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.concatenate([bar, img_bgr], axis=0)


def make_collage(
    row: pd.Series,
    base_cols: list[str],
    wrist_cols: list[str],
    video_source: VideoFrameSource | None = None,
) -> np.ndarray:
    """
    Build a collage with base_rgb* on the top row and wrist_rgb* on the bottom row.
    Each tile has a label bar. Tiles/rows are padded with black if sizes differ.
    """
    def get_image(col: str) -> np.ndarray:
        if video_source is not None and col in video_source.keys:
            return video_source.get_frame(col, int(row["episode_index"]), int(row["frame_index"]))
        return decode_image(row[col])

    def build_row(cols: list[str]) -> np.ndarray | None:
        tiles = []
        for col in cols:
            img = get_image(col)
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


def generate_output_path(parquet_folder: str, episode_str: str | None, fmt: str = "mp4") -> str:
    """Generate output video path based on parquet folder, episode, and format."""
    parts = parquet_folder.rstrip("/").split("/")
    chunk_name = parts[-1]
    dataset_name = parts[-3]

    output_dir = "outputs/view_lerobot_parquet_video"
    ep_str = f"episode_{episode_str}" if episode_str is not None else "all_episodes"
    output_filename = f"{dataset_name}_{chunk_name}_{ep_str}.{fmt}"
    return os.path.join(output_dir, output_filename)



def _check_image_keys(base_cols, wrist_cols, video_source: VideoFrameSource | None) -> None:
    """Fail early (with an actionable message) instead of on the first frame."""
    if base_cols or wrist_cols:
        return
    if video_source is not None:
        raise ValueError(
            "No base_rgb*/wrist_rgb* image keys found. Video features in this dataset: "
            f"{video_source.keys} (check --keys filter)"
        )
    raise ValueError(
        "No image columns in the parquet and no video-backed dataset root found. "
        "Point --parquet_folder at <dataset_root>/data/chunk-XXX so meta/info.json "
        "and videos/ can be located."
    )


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


def save_video(
    df: pd.DataFrame,
    output_path: str,
    episodes: list[int] | None = None,
    fps: int = 30,
    gif_fps: int = 10,
    video_source: VideoFrameSource | None = None,
    key_filter: list[str] | None = None,
):
    """Save collage video of all base_rgb* and wrist_rgb* columns to a file.

    GIFs are subsampled to ~gif_fps (keeping every Nth frame) so playback stays
    real-time while the file shrinks proportionally.
    """
    df = _prepare_df(df, episodes)

    base_cols, wrist_cols = find_image_columns(df, video_source, key_filter)
    print(f"Base columns: {base_cols}")
    print(f"Wrist columns: {wrist_cols}")
    _check_image_keys(base_cols, wrist_cols, video_source)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.lower().endswith(".gif"):
        stride = max(1, round(fps / gif_fps))
        if stride > 1:
            df = df.iloc[::stride]
            fps = fps / stride
            print(f"GIF: keeping every {stride}th frame ({len(df)} frames at {fps:g} FPS)")
        print(f"Saving {len(df)} frames at {fps:g} FPS to {output_path}")
        writer = imageio.get_writer(output_path, fps=fps, loop=0)
    else:
        print(f"Saving {len(df)} frames at {fps} FPS to {output_path}")
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')

    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]
        collage_bgr = make_collage(row, base_cols, wrist_cols, video_source)
        collage_rgb = cv2.cvtColor(collage_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(collage_rgb)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(df)} frames")

    writer.close()
    print(f"Video saved to {output_path}")


def play_video(
    df: pd.DataFrame,
    episodes: list[int] | None = None,
    fps: int = 30,
    pause_every_episode: bool = False,
    video_source: VideoFrameSource | None = None,
    key_filter: list[str] | None = None,
):
    """Play collage of all base_rgb* and wrist_rgb* columns."""
    df = _prepare_df(df, episodes)

    base_cols, wrist_cols = find_image_columns(df, video_source, key_filter)
    print(f"Base columns: {base_cols}")
    print(f"Wrist columns: {wrist_cols}")
    _check_image_keys(base_cols, wrist_cols, video_source)

    delay_ms = int(1000 / fps)
    ep_label = str(episodes) if episodes is not None else "all"
    window_name = f"base_rgb* | wrist_rgb* (episodes: {ep_label})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Playing {len(df)} frames at {fps} FPS")
    hint = (
        "Press 'space' to start/pause, 'q' to quit, 'n' for next frame when paused, "
        "'b' for previous, 'e' to jump to next episode, 'r' (or Backspace) when "
        "paused to replay the episode from its start"
    )
    if pause_every_episode:
        hint += (
            "\n[pause-every-episode] will auto-pause on LAST frame of each "
            "episode AND FIRST frame of the next (two space-presses per "
            "boundary; overlay text labels which is which)."
        )
    print(hint)

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

    # First / last row-of-df indices for each episode. Used by 'e' (jump-to-
    # next) and by --pause-every-episode (we auto-pause at these specific
    # frame_idx values to bracket boundaries).
    ep_start_indices: dict[int, int] = {}
    ep_end_indices: dict[int, int] = {}
    if has_episode_col:
        last_ep = None
        for i in range(len(df)):
            ep = int(df.iloc[i]["episode_index"])
            if ep != last_ep:
                ep_start_indices[ep] = i
                if last_ep is not None:
                    ep_end_indices[last_ep] = i - 1
                last_ep = ep
        if last_ep is not None:
            ep_end_indices[last_ep] = len(df) - 1
    else:
        ep_start_indices = {0: 0}
        ep_end_indices = {0: len(df) - 1}

    # Auto-pause schedule: which frame indices should pause when reached for
    # the first time, and what overlay label to show. We include every
    # episode-END and every episode-START EXCEPT the very first start
    # (frame 0 is already paused by the initial `paused=True` above; an
    # extra "START OF EPISODE N" overlay there would be redundant).
    auto_pause_label: dict[int, str] = {}
    if pause_every_episode and has_episode_col:
        # End-of-episode pauses (skip the very last episode's end — there's
        # no "next" to advance to; the playback just terminates naturally).
        for ep in episode_list[:-1]:
            auto_pause_label[ep_end_indices[ep]] = "END"
        # Start-of-episode pauses (skip the very first episode's start as
        # noted above — frame 0 is already the initial-pause).
        for ep in episode_list[1:]:
            auto_pause_label[ep_start_indices[ep]] = "START"
    already_auto_paused_at: set[int] = set()  # don't re-pause on the same frame after user resumes

    ep_pbar = tqdm(episode_list, desc="Episode", position=0, leave=True)
    frame_pbar = tqdm(total=ep_frame_counts[episode_list[0]], desc="Frame  ", position=1, leave=True)

    current_ep = None

    def _draw_overlay(img: np.ndarray, label: str, ep_num: int, frame_in_ep: int, total_in_ep: int) -> np.ndarray:
        """Stamp a big bottom banner on the collage with the auto-pause kind
        (END / START), the episode number, the frame index within the
        episode, and a usage hint. Drawn as a semi-transparent black bar
        with white text so it's readable over any video content. Returns
        a new image (does not mutate the input).
        """
        out = img.copy()
        h, w = out.shape[:2]
        # Banner height proportional to image height, with floor so text fits.
        bar_h = max(80, int(h * 0.12))
        # Semi-transparent black bar across the bottom.
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, dst=out)
        # Two lines of text: big label + smaller hint.
        if label == "END":
            big = f"END OF EPISODE {ep_num}  (frame {frame_in_ep + 1}/{total_in_ep})"
            small = "[space] -> next episode    [r]/[backspace] -> replay this episode"
            big_color = (50, 80, 255)  # red-ish in BGR (terminal feedback for "stop")
        else:
            big = f"START OF EPISODE {ep_num}  (frame {frame_in_ep + 1}/{total_in_ep})"
            small = "[space] -> play this episode    [r]/[backspace] -> replay previous episode"
            big_color = (80, 255, 80)  # green-ish in BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Big label.
        big_scale = max(0.6, w / 1200.0)
        big_thick = max(2, int(big_scale * 2))
        (tw, th), _ = cv2.getTextSize(big, font, big_scale, big_thick)
        cv2.putText(
            out, big,
            org=((w - tw) // 2, h - bar_h + th + 10),
            fontFace=font, fontScale=big_scale, color=big_color,
            thickness=big_thick, lineType=cv2.LINE_AA,
        )
        # Hint below.
        small_scale = max(0.4, big_scale * 0.55)
        small_thick = max(1, int(small_scale * 2))
        (sw, sh), _ = cv2.getTextSize(small, font, small_scale, small_thick)
        cv2.putText(
            out, small,
            org=((w - sw) // 2, h - 12),
            fontFace=font, fontScale=small_scale, color=(255, 255, 255),
            thickness=small_thick, lineType=cv2.LINE_AA,
        )
        return out

    while frame_idx < len(df):
        row = df.iloc[frame_idx]
        collage = make_collage(row, base_cols, wrist_cols, video_source)

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
        else:
            ep_num = 0

        # Auto-pause-at-boundary decision. Fire when we land on a boundary
        # frame for the first time (so users can space through; subsequent
        # iterations on the same frame_idx while paused don't re-pause).
        label_for_overlay: str | None = None
        if frame_idx in auto_pause_label and frame_idx not in already_auto_paused_at:
            paused = True
            already_auto_paused_at.add(frame_idx)
            label_for_overlay = auto_pause_label[frame_idx]
        elif paused and frame_idx in auto_pause_label:
            # Still sitting on the auto-paused frame after we already
            # consumed the auto-pause (e.g. user pressed b to backstep
            # then n forward). Keep showing the overlay so they see the
            # banner state matches the frame they're looking at.
            label_for_overlay = auto_pause_label[frame_idx]

        if label_for_overlay is not None:
            # Frame-within-episode position for the overlay text.
            frame_in_ep = frame_idx - ep_start_indices[ep_num]
            total_in_ep = ep_frame_counts[ep_num]
            collage = _draw_overlay(collage, label_for_overlay, ep_num, frame_in_ep, total_in_ep)

        cv2.imshow(window_name, collage)

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if paused and label_for_overlay == "END":
                # Special transition: at end-of-episode pause, advance one
                # frame (to next episode's first frame) but stay PAUSED so
                # the START-of-episode auto-pause can fire immediately.
                # This bracket-the-boundary behavior is the whole point of
                # --pause-every-episode.
                frame_idx += 1
                continue
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('n') and paused:
            frame_idx += 1
            frame_pbar.update(1)
            continue
        elif key == ord('b') and paused and frame_idx > 0:
            frame_idx -= 1
            frame_pbar.n = max(frame_pbar.n - 1, 0)
            frame_pbar.refresh()
            continue
        elif key in (ord('r'), 8) and paused:  # 8 = Backspace
            # Replay an episode from its first frame and auto-play. Which
            # episode: at a START-of-episode pause screen, "the episode that
            # just finished" is the PREVIOUS one (the current frame is
            # already the next episode's first frame); everywhere else
            # (END screen, or manually paused mid-episode) it's the current
            # episode.
            target_ep = ep_num
            if label_for_overlay == "START" and has_episode_col:
                prev_eps = [e for e in episode_list if e < ep_num]
                if prev_eps:
                    target_ep = prev_eps[-1]
            # Re-arm the target's END auto-pause so the replay pauses there
            # again. Its START pause stays consumed — we're playing straight
            # through from the first frame, a start-banner would be noise.
            already_auto_paused_at.discard(ep_end_indices.get(target_ep, -1))
            frame_idx = ep_start_indices.get(target_ep, 0)
            paused = False
            # None forces the episode-change block above to reset the frame
            # progress bar WITHOUT bumping ep_pbar (that guard checks
            # `current_ep is not None`), so replays don't inflate the count.
            current_ep = None
            print(f"[r] replaying episode {target_ep}")
            continue
        elif key == ord('e'):
            # Jump to the first frame of the NEXT episode (skip current).
            if has_episode_col:
                next_ep_starts = [i for i in ep_start_indices.values() if i > frame_idx]
                if next_ep_starts:
                    frame_idx = min(next_ep_starts)
                    continue
                else:
                    print("[e] no more episodes — already in the last one")

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
    print(f"Total frames: {sum(ep_counts.values())}")


def state_stats(parquet_folder: str):
    """Print per-dimension statistics for observation.state, including mean absolute velocity, acceleration, and jerk."""
    import pyarrow.parquet as pq

    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")

    # Check if observation.state exists
    sample = pq.read_table(parquet_files[0], columns=None).schema
    col_names = [field.name for field in sample]
    if "observation.state" not in col_names:
        print("No 'observation.state' column found in dataset.")
        return

    all_states = []
    all_episodes = []
    all_frames = []
    for f in parquet_files:
        cols = ["episode_index", "frame_index", "observation.state"]
        cols = [c for c in cols if c in col_names]
        table = pq.read_table(f, columns=cols)
        df_chunk = table.to_pandas()
        all_states.append(np.stack(df_chunk["observation.state"].to_numpy()))
        if "episode_index" in df_chunk.columns:
            all_episodes.append(df_chunk["episode_index"].to_numpy())
        if "frame_index" in df_chunk.columns:
            all_frames.append(df_chunk["frame_index"].to_numpy())

    states = np.concatenate(all_states, axis=0)  # (N, D)
    n_dims = states.shape[1]

    # Compute per-episode derivatives to avoid cross-episode transitions
    nan = np.full(n_dims, float("nan"))
    if all_episodes:
        episodes = np.concatenate(all_episodes, axis=0)
        frames = np.concatenate(all_frames, axis=0) if all_frames else None
        abs_vels,   vel_eps   = [], []
        abs_accels, accel_eps = [], []
        abs_jerks,  jerk_eps  = [], []
        for ep in np.unique(episodes):
            mask = episodes == ep
            ep_states = states[mask]
            if frames is not None:
                order = np.argsort(frames[mask])
                ep_states = ep_states[order]
            n = len(ep_states)
            if n > 1:
                abs_vels.append(np.abs(np.diff(ep_states, axis=0)))
                vel_eps.append(np.full(n - 1, ep))
            if n > 2:
                abs_accels.append(np.abs(np.diff(ep_states, n=2, axis=0)))
                accel_eps.append(np.full(n - 2, ep))
            if n > 3:
                abs_jerks.append(np.abs(np.diff(ep_states, n=3, axis=0)))
                jerk_eps.append(np.full(n - 3, ep))

        all_abs_vel   = np.concatenate(abs_vels,   axis=0) if abs_vels   else None
        all_abs_accel = np.concatenate(abs_accels, axis=0) if abs_accels else None
        all_abs_jerk  = np.concatenate(abs_jerks,  axis=0) if abs_jerks  else None
        vel_ep_arr   = np.concatenate(vel_eps)   if vel_eps   else None
        accel_ep_arr = np.concatenate(accel_eps) if accel_eps else None
        jerk_ep_arr  = np.concatenate(jerk_eps)  if jerk_eps  else None

        mean_abs_vel   = all_abs_vel.mean(axis=0)   if all_abs_vel   is not None else nan
        mean_abs_accel = all_abs_accel.mean(axis=0) if all_abs_accel is not None else nan
        mean_abs_jerk  = all_abs_jerk.mean(axis=0)  if all_abs_jerk  is not None else nan

        def ep_of_nz_min(arr, ep_arr):
            if arr is None:
                return [None] * n_dims
            result = []
            for d in range(n_dims):
                col = arr[:, d]
                nz = np.flatnonzero(col)
                idx = nz[np.argmin(col[nz])] if len(nz) else np.argmin(col)
                result.append(int(ep_arr[idx]))
            return result

        def ep_of_max(arr, ep_arr):
            return [int(ep_arr[np.argmax(arr[:, d])]) for d in range(n_dims)] if arr is not None else [None] * n_dims

        state_min_ep = [int(episodes[np.argmin(states[:, d])]) for d in range(n_dims)]
        state_max_ep = [int(episodes[np.argmax(states[:, d])]) for d in range(n_dims)]
        vel_min_ep   = ep_of_nz_min(all_abs_vel,   vel_ep_arr)
        vel_max_ep   = ep_of_max(all_abs_vel,   vel_ep_arr)
        accel_min_ep = ep_of_nz_min(all_abs_accel, accel_ep_arr)
        accel_max_ep = ep_of_max(all_abs_accel, accel_ep_arr)
        jerk_min_ep  = ep_of_nz_min(all_abs_jerk,  jerk_ep_arr)
        jerk_max_ep  = ep_of_max(all_abs_jerk,  jerk_ep_arr)
        n_episodes = len(np.unique(episodes))
    else:
        mean_abs_vel   = np.mean(np.abs(np.diff(states, n=1, axis=0)), axis=0)
        mean_abs_accel = np.mean(np.abs(np.diff(states, n=2, axis=0)), axis=0)
        mean_abs_jerk  = np.mean(np.abs(np.diff(states, n=3, axis=0)), axis=0)
        episodes = None
        state_min_ep = state_max_ep = [None] * n_dims
        vel_min_ep = vel_max_ep = accel_min_ep = accel_max_ep = jerk_min_ep = jerk_max_ep = [None] * n_dims
        all_abs_vel = all_abs_accel = all_abs_jerk = None
        vel_ep_arr = accel_ep_arr = jerk_ep_arr = None
        n_episodes = "?"

    print(f"\n=== observation.state statistics ({states.shape[0]} frames, {n_episodes} episodes, {n_dims} dims) ===\n")

    def fv(val, ep):
        """Format a value+episode as '0.1234[42]' left-padded to width 16."""
        ep_str = f"[{ep}]" if ep is not None else ""
        return f"{val:.4f}{ep_str}"

    col_w = 16
    S = " | "  # group separator
    header = (
        f"{'Dim':<5}{S}{'Mean':>10}  {'Std':>10}  {'Min[ep]':>{col_w}}  {'Max[ep]':>{col_w}}"
        f"{S}{'Mean|v|':>10}  {'Min|v|[ep]':>{col_w}}  {'Max|v|[ep]':>{col_w}}"
        f"{S}{'Mean|a|':>10}  {'Min|a|[ep]':>{col_w}}  {'Max|a|[ep]':>{col_w}}"
        f"{S}{'Mean|j|':>10}  {'Min|j|[ep]':>{col_w}}  {'Max|j|[ep]':>{col_w}}"
    )
    print(header)

    def nz_min(col):
        nz = col[col != 0]
        return nz.min() if len(nz) else float("nan")

    for d in range(n_dims):
        v_min = nz_min(all_abs_vel[:, d])   if all_abs_vel   is not None else float("nan")
        v_max = all_abs_vel[:, d].max()      if all_abs_vel   is not None else float("nan")
        a_min = nz_min(all_abs_accel[:, d]) if all_abs_accel is not None else float("nan")
        a_max = all_abs_accel[:, d].max()   if all_abs_accel is not None else float("nan")
        j_min = nz_min(all_abs_jerk[:, d])  if all_abs_jerk  is not None else float("nan")
        j_max = all_abs_jerk[:, d].max()    if all_abs_jerk  is not None else float("nan")
        print(
            f"{d:<5}{S}{states[:, d].mean():>10.4f}  {states[:, d].std():>10.4f}  "
            f"{fv(states[:, d].min(), state_min_ep[d]):>{col_w}}  {fv(states[:, d].max(), state_max_ep[d]):>{col_w}}"
            f"{S}{mean_abs_vel[d]:>10.6f}  {fv(v_min, vel_min_ep[d]):>{col_w}}  {fv(v_max, vel_max_ep[d]):>{col_w}}"
            f"{S}{mean_abs_accel[d]:>10.6f}  {fv(a_min, accel_min_ep[d]):>{col_w}}  {fv(a_max, accel_max_ep[d]):>{col_w}}"
            f"{S}{mean_abs_jerk[d]:>10.6f}  {fv(j_min, jerk_min_ep[d]):>{col_w}}  {fv(j_max, jerk_max_ep[d]):>{col_w}}"
        )

    # Global min/max across all dims for derivatives
    def global_nz_min_ep(arr, ep_arr):
        if arr is None:
            return float("nan"), None
        flat_nz = np.flatnonzero(arr)
        if len(flat_nz) == 0:
            return float("nan"), None
        idx = flat_nz[np.argmin(arr.ravel()[flat_nz])]
        r, _ = np.unravel_index(idx, arr.shape)
        return arr.ravel()[idx], int(ep_arr[r])

    def global_max_ep(arr, ep_arr):
        if arr is None:
            return float("nan"), None
        idx = np.argmax(arr)
        r, _ = np.unravel_index(idx, arr.shape)
        return arr.max(), int(ep_arr[r])

    state_global_min_idx = np.argmin(states)
    state_global_max_idx = np.argmax(states)
    state_global_min_r, _ = np.unravel_index(state_global_min_idx, states.shape)
    state_global_max_r, _ = np.unravel_index(state_global_max_idx, states.shape)
    state_global_min_ep = int(episodes[state_global_min_r]) if episodes is not None else None
    state_global_max_ep = int(episodes[state_global_max_r]) if episodes is not None else None

    v_global_min, v_global_min_ep = global_nz_min_ep(all_abs_vel,   vel_ep_arr)
    v_global_max, v_global_max_ep = global_max_ep(all_abs_vel,     vel_ep_arr)
    a_global_min, a_global_min_ep = global_nz_min_ep(all_abs_accel, accel_ep_arr)
    a_global_max, a_global_max_ep = global_max_ep(all_abs_accel,    accel_ep_arr)
    j_global_min, j_global_min_ep = global_nz_min_ep(all_abs_jerk,  jerk_ep_arr)
    j_global_max, j_global_max_ep = global_max_ep(all_abs_jerk,     jerk_ep_arr)

    sep = "-" * len(header)
    print(sep)
    print(
        f"{'all':<5}{S}{states.mean():>10.4f}  {states.std():>10.4f}  "
        f"{fv(states.min(), state_global_min_ep):>{col_w}}  {fv(states.max(), state_global_max_ep):>{col_w}}"
        f"{S}{mean_abs_vel.mean():>10.6f}  {fv(v_global_min, v_global_min_ep):>{col_w}}  {fv(v_global_max, v_global_max_ep):>{col_w}}"
        f"{S}{mean_abs_accel.mean():>10.6f}  {fv(a_global_min, a_global_min_ep):>{col_w}}  {fv(a_global_max, a_global_max_ep):>{col_w}}"
        f"{S}{mean_abs_jerk.mean():>10.6f}  {fv(j_global_min, j_global_min_ep):>{col_w}}  {fv(j_global_max, j_global_max_ep):>{col_w}}"
    )
    print("  * Min|v|, Min|a|, Min|j| are the minimum nonzero absolute values.")
    print()


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
        "--keys",
        type=str,
        default=None,
        help="Comma-separated substrings to filter which image/video keys are shown, "
        "e.g. --keys stretch (default: all base_rgb*/wrist_rgb* keys)",
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
        "--state-stats",
        action="store_true",
        help="Print per-dimension statistics (mean, std, min, max, mean absolute velocity) for observation.state and exit",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video to file instead of displaying interactively",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format for --save_video (default: mp4)",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=10,
        help="Target FPS for GIF output; frames are subsampled from --fps to keep "
        "real-time playback speed (default: 10)",
    )
    parser.add_argument(
        "--pause-every-episode",
        action="store_true",
        help="When playing back, auto-pause TWICE at each episode boundary: "
        "once on the LAST frame of episode N (overlay: 'END OF EPISODE N'), "
        "then again on the FIRST frame of episode N+1 (overlay: 'START OF "
        "EPISODE N+1'). Space advances through each pause. Lets you "
        "clearly see how an episode stops and how the next one starts. "
        "No effect on --save_video.",
    )

    args = parser.parse_args()

    if args.list_episodes:
        list_episodes(args.parquet_folder)
        return

    if args.state_stats:
        state_stats(args.parquet_folder)
        return

    episodes = parse_episodes(args.episode) if args.episode is not None else None

    if args.episode_lengths:
        episode_lengths(args.parquet_folder, subset=episodes)
        return

    df = load_parquet_dataset(args.parquet_folder, episodes)

    # Video-backed datasets keep no images in the parquet; look for the dataset
    # root so frames can be decoded from <root>/videos/*.mp4 instead.
    video_source = None
    base_cols, wrist_cols = find_image_columns(df)
    if not base_cols and not wrist_cols:
        root = find_dataset_root(args.parquet_folder)
        if root is not None:
            video_source = VideoFrameSource(root)
            print(f"No image columns in parquet — decoding video features from {root}/videos")

    key_filter = [k.strip() for k in args.keys.split(",")] if args.keys else None

    if args.save_video:
        output_path = generate_output_path(args.parquet_folder, args.episode, args.format)
        save_video(
            df, output_path, episodes, args.fps, gif_fps=args.gif_fps,
            video_source=video_source, key_filter=key_filter,
        )
    else:
        play_video(
            df, episodes, args.fps, pause_every_episode=args.pause_every_episode,
            video_source=video_source, key_filter=key_filter,
        )


if __name__ == "__main__":
    main()
