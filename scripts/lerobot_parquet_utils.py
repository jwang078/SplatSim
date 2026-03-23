"""Shared utilities for LeRobot parquet dataset scripts."""

import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def load_episode_meta(meta_episodes_dir: str, episode_indices: list[int] | None = None) -> list[dict]:
    """Load episode metadata rows from meta/episodes/chunk-*/*.parquet.

    Args:
        meta_episodes_dir: Path to the meta/episodes directory.
        episode_indices: If given, only return rows for these episode indices.

    Returns:
        List of dicts, one per episode, with all metadata columns.
    """
    files = sorted(glob.glob(os.path.join(meta_episodes_dir, "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {meta_episodes_dir}")

    rows = []
    for f in files:
        filters = [("episode_index", "in", episode_indices)] if episode_indices is not None else None
        table = pq.read_table(f, filters=filters)
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            rows.append(row)
    return rows


def parse_object_configs(raw) -> list[dict]:
    """Parse splatsim_object_configs field (may be a JSON string or already a list)."""
    if isinstance(raw, str):
        return json.loads(raw)
    if raw is None:
        return []
    return raw


def parse_episodes(episode_str: str) -> list[int]:
    """Parse episode argument into a list of ints.

    Supports:
      - Single int:              '3'
      - Range:                   '0-5'
      - Comma-separated:         '0,2,4'
      - Mixed:                   '0-3,7,9-11'
      - Python/numpy array fmt:  '[3, 8, 23, 38]'
    """
    episode_str = episode_str.strip().strip("[]")
    indices = []
    for part in episode_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def episode_frame_counts(meta_episodes_dir: str) -> dict[int, int]:
    """Return {episode_index: frame_count} for all episodes, without loading image data."""
    files = sorted(glob.glob(os.path.join(meta_episodes_dir, "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {meta_episodes_dir}")
    counts: dict[int, int] = {}
    for f in files:
        table = pq.read_table(f, columns=["episode_index", "length"])
        for ep, length in zip(
            table.column("episode_index").to_pylist(),
            table.column("length").to_pylist(),
        ):
            counts[ep] = length
    return counts
