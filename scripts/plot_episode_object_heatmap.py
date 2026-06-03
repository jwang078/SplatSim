#!/usr/bin/env python3
"""
Plot a 2D heatmap of initial XY positions for each object across benchmark episodes.

Each object gets its own color; density is shown by color intensity (darker = more episodes
with an object at that position). All objects are overlaid on the same axes with a legend.

Usage:
    python scripts/plot_episode_object_heatmap.py \
        --dataset ~/.cache/huggingface/lerobot/JennyWWW/eval_splatsim_approach_lever_benchmark_1000 \
        [--episode '3,8,23' | '[3,8,23]' | '0-99']  # optional subset
        [--skip table wall]                           # objects to exclude
        [--output heatmap.png]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, uniform_filter

from lerobot_parquet_utils import load_episode_meta, parse_episodes, parse_object_configs

X_LIM = (-0.75, 0.75)
Y_LIM = (-0.2, 1.0)
GRID_BINS = 80  # resolution of the KDE grid

# Physical footprint (x_size, y_size) in metres for objects whose center point is plotted.
# The heatmap for these objects is convolved with a box kernel of this size so the full
# occupied area is visible rather than just the centre point.
OBJECT_FOOTPRINTS: dict[str, tuple[float, float]] = {
    "box1": (0.1524, 0.508),   # thinkpad box:  15.24 cm x 50.8 cm
    "box2": (0.508, 0.0762),   # starwars box:  50.8 cm x 7.62 cm
}

# Fixed reference objects drawn as labeled rectangles (center_x, center_y, width, height).
STATIC_RECTANGLES = [
    {"label": "small_engine_new", "cx": -0.48, "cy": 0.36, "w": 0.376, "h": 0.342},
]


SISBOT_URDF = "/home/jennyw2/code/SplatSim/splatsim/robot_definitions/urdf/sisbot.urdf"
SISBOT_BASE_POS = [0.0, 0.0, -0.088]
SISBOT_ARM_JOINT_INDICES = [1, 2, 3, 4, 5, 6]  # shoulder_pan .. wrist_3
SISBOT_EE_LINK_INDEX = 7  # ee_link


def _load_first_frame_states(dataset_root: str, episode_indices: list[int]) -> dict[int, np.ndarray]:
    """Load observation.state from the first frame of each requested episode.

    Returns {episode_index: state_array}. Episodes not found are omitted.
    """
    parquet_files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "*.parquet")))
    if not parquet_files:
        return {}
    wanted = set(episode_indices)
    result: dict[int, np.ndarray] = {}
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        except Exception:
            continue
        df = df[df["episode_index"].isin(wanted)]
        if df.empty:
            continue
        first_frames = df.sort_values("frame_index").groupby("episode_index").first()
        for ep_idx, row in first_frames.iterrows():
            if ep_idx not in result:
                result[int(ep_idx)] = np.asarray(row["observation.state"])
        if wanted.issubset(result):
            break
    return result


def compute_ee_positions(rows: list[dict], dataset_root: str | None = None) -> np.ndarray:
    """Return (N, 2) array of EE XY positions computed via pybullet FK (headless).

    Primary source: splatsim_robot_config.articulation_config.initial_joint_positions.
    Fallback: observation.state from the first data frame of each episode (requires
    dataset_root so the data parquet files can be found).

    Episodes where neither source is available are skipped with a warning.
    """
    import json
    import pybullet as p

    client = p.connect(p.DIRECT)
    robot = p.loadURDF(SISBOT_URDF, basePosition=SISBOT_BASE_POS,
                       useFixedBase=True, physicsClientId=client)

    # Pre-load first-frame states for any episode that might need the fallback.
    needs_fallback: list[int] = [
        int(r["episode_index"])
        for r in rows
        if r.get("splatsim_robot_config") is None and r.get("episode_index") is not None
    ]
    fallback_states: dict[int, np.ndarray] = {}
    if needs_fallback and dataset_root is not None:
        print(f"  Falling back to observation.state for {len(needs_fallback)} episode(s) missing splatsim_robot_config")
        fallback_states = _load_first_frame_states(dataset_root, needs_fallback)
    elif needs_fallback:
        print(f"  Warning: {len(needs_fallback)} episode(s) missing splatsim_robot_config and no dataset_root provided; those episodes will be skipped")

    xy = []
    skipped = 0
    for row in rows:
        cfg = row.get("splatsim_robot_config")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)

        joints: list[float] | None = None
        if cfg is not None:
            joints = cfg["articulation_config"]["initial_joint_positions"][:6]
        else:
            ep_idx = row.get("episode_index")
            state = fallback_states.get(ep_idx) if ep_idx is not None else None
            if state is not None:
                joints = state[:6].tolist()
            else:
                if ep_idx is not None:
                    print(f"  Warning: episode {ep_idx} has no robot config or state data; skipping EE position")
                skipped += 1
                continue

        assert joints is not None
        for ji, jval in zip(SISBOT_ARM_JOINT_INDICES, joints):
            p.resetJointState(robot, ji, jval, physicsClientId=client)
        pos = p.getLinkState(robot, SISBOT_EE_LINK_INDEX, physicsClientId=client)[0]
        xy.append((pos[0], pos[1]))

    p.disconnect(client)
    if skipped:
        print(f"  Skipped {skipped} episode(s) with no joint data")
    return np.array(xy) if xy else np.empty((0, 2))


def build_position_map(rows: list[dict], skip: set[str], skip_static: bool = True) -> dict[str, np.ndarray]:
    """Return {object_name: array of shape (N, 2)} of XY initial positions.

    If skip_static=True (default), objects whose XY position never varies across
    episodes are automatically excluded (e.g. table, wall).
    """
    positions: dict[str, list[tuple[float, float]]] = {}
    missing_count = 0
    for row in rows:
        raw = row.get("splatsim_object_configs")
        if raw is None:
            missing_count += 1
            continue
        for obj in parse_object_configs(raw):
            name = obj.get("name", "unknown")
            if name in skip:
                continue
            pos = obj.get("initial_position")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            positions.setdefault(name, []).append((x, y))

    if missing_count:
        print(f"  Warning: {missing_count} episode(s) have no splatsim_object_configs; skipped from object position map")

    result = {}
    for name, pts in positions.items():
        arr = np.array(pts)
        if skip_static and arr.shape[0] > 1 and np.allclose(arr, arr[0]):
            print(f"  Skipping static object '{name}' (position never varies)")
            continue
        result[name] = arr
    return result


def make_alpha_cmap(base_color):
    """Create a colormap that goes from transparent to `base_color`."""
    r, g, b, _ = plt.cm.colors.to_rgba(base_color) if isinstance(base_color, str) else (*base_color[:3], 1)
    return LinearSegmentedColormap.from_list(
        "alpha_cmap",
        [(r, g, b, 0.0), (r, g, b, 1.0)],
    )


def plot_heatmap(
    position_map: dict[str, np.ndarray],
    subset_label: str,
    output: str | None,
    robot_ee_pts: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Use tab10 for distinct per-object colors
    palette = plt.colormaps["tab10"]
    names = sorted(position_map)

    xs = np.linspace(X_LIM[0], X_LIM[1], GRID_BINS + 1)
    ys = np.linspace(Y_LIM[0], Y_LIM[1], GRID_BINS + 1)

    # Track per-object colors so we can draw footprint outlines later
    object_colors: dict[str, tuple] = {}

    legend_handles = []
    for i, name in enumerate(names):
        pts = position_map[name]
        if len(pts) == 0:
            continue

        color = palette(i % 10)
        object_colors[name] = color
        cmap = make_alpha_cmap(color)

        # 2D histogram counts
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[xs, ys])
        # H is (x_bins, y_bins); imshow expects (y, x) with origin='lower'
        H = H.T

        # If this object has a known physical footprint, convolve with a box kernel
        # so the full occupied area is shown rather than just the centre point.
        if name in OBJECT_FOOTPRINTS:
            x_size, y_size = OBJECT_FOOTPRINTS[name]
            bin_w = (X_LIM[1] - X_LIM[0]) / GRID_BINS
            bin_h = (Y_LIM[1] - Y_LIM[0]) / GRID_BINS
            kx = max(1, round(x_size / bin_w))
            ky = max(1, round(y_size / bin_h))
            H = uniform_filter(H, size=(ky, kx), mode="constant")

        # Normalize to [0,1] by max bin, then scale vmax so that the overall
        # brightness reflects what fraction of episodes landed in the peak bin.
        # A peak bin hit by 5% of episodes → full opacity; less dense → proportionally dimmer.
        # This keeps darkness comparable between subset and full-dataset plots.
        VMAX_REF = 0.05  # fraction of episodes for "full opacity"
        if H.max() > 0:
            peak_fraction = H.max() / len(pts)
            H_norm = H / H.max()
            vmax = min(1.0, VMAX_REF / peak_fraction)
        else:
            H_norm = H
            vmax = 1.0

        ax.imshow(
            H_norm,
            origin="lower",
            extent=[X_LIM[0], X_LIM[1], Y_LIM[0], Y_LIM[1]],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="bilinear",
        )

        # Legend proxy patch
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor=color, label=f"{name} (n={len(pts)})"))

    # Overlay robot EE positions as a separate heatmap layer
    if robot_ee_pts is not None and len(robot_ee_pts) > 0:
        from matplotlib.patches import Patch
        ee_color = (1.0, 0.85, 0.0, 1.0)  # yellow
        ee_cmap = make_alpha_cmap(ee_color)
        H_ee, _, _ = np.histogram2d(robot_ee_pts[:, 0], robot_ee_pts[:, 1], bins=[xs, ys])
        H_ee = H_ee.T
        # Smooth EE positions with a Gaussian kernel. Scale sigma so sparser
        # subsets get more spread — target ~sqrt(1000/n) * base_sigma.
        base_sigma_m = 0.04  # 4 cm base spread at 1000 episodes
        scale = np.log(1000) / np.log(max(len(robot_ee_pts), 2))  # log scale: gentler growth
        sigma_m = base_sigma_m * scale
        bin_size = (X_LIM[1] - X_LIM[0]) / GRID_BINS
        sigma_bins = max(0.5, sigma_m / bin_size)
        H_ee = gaussian_filter(H_ee, sigma=sigma_bins)
        if H_ee.max() > 0:
            ee_peak_fraction = H_ee.max() / len(robot_ee_pts)
            H_ee = H_ee / H_ee.max()
            ee_vmax = min(1.0, 0.05 / ee_peak_fraction)
        else:
            ee_vmax = 1.0
        ax.imshow(
            H_ee,
            origin="lower",
            extent=[X_LIM[0], X_LIM[1], Y_LIM[0], Y_LIM[1]],
            aspect="auto",
            cmap=ee_cmap,
            vmin=0,
            vmax=ee_vmax,
            interpolation="bilinear",
        )
        legend_handles.append(Patch(facecolor=ee_color, label=f"robot EE (n={len(robot_ee_pts)})"))

    # Draw footprint outlines for objects with known physical sizes.
    # Placed at the median position of their point cloud; colored like their heatmap but darker.
    from matplotlib.patches import Rectangle
    for name, (x_size, y_size) in OBJECT_FOOTPRINTS.items():
        if name not in position_map or name not in object_colors:
            continue
        pts = position_map[name]
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        r, g, b, _ = object_colors[name]
        dark_color = (r * 0.7, g * 0.7, b * 0.7)  # darken by 30%
        patch = Rectangle(
            (cx - x_size / 2, cy - y_size / 2), x_size, y_size,
            linewidth=1.5, edgecolor=dark_color, facecolor="none",
            linestyle="--", zorder=5,
        )
        ax.add_patch(patch)
        ax.text(cx, cy, name, color="black", fontsize=7,
                ha="center", va="center", zorder=6)

    # Draw robot base footprint
    from matplotlib.patches import Circle
    robot_circle = Circle((0, 0), radius=0.15, linewidth=1.5, edgecolor="black",
                           facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(robot_circle)
    ax.text(0, 0, "robot", color="black", fontsize=7, ha="center", va="center", zorder=6)

    # Draw fixed reference rectangles (e.g. small_engine_new at its fixed position)
    for rect in STATIC_RECTANGLES:
        x0 = rect["cx"] - rect["w"] / 2
        y0 = rect["cy"] - rect["h"] / 2
        patch = Rectangle((x0, y0), rect["w"], rect["h"],
                           linewidth=1.5, edgecolor="black", facecolor="none",
                           linestyle="--", zorder=5)
        ax.add_patch(patch)
        ax.text(rect["cx"], rect["cy"], rect["label"],
                color="black", fontsize=7, ha="center", va="center", zorder=6)

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    title = "Initial Object Positions"
    if subset_label:
        title += f"\n{subset_label}"
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Heatmap of initial object XY positions across episodes")
    parser.add_argument(
        "--dataset",
        default="JennyWWW/splatsim_approach_lever_11_50failsrrtpi05",
        help="Repo ID (e.g. 'JennyWWW/my-dataset') or absolute path to the dataset root",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Explicit path to the dataset's data/ directory, bypassing auto-resolution",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Episode subset: '3', '0-99', '0,2,4', or '[3,8,23]'. Default: all episodes.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Additional object names to exclude from the plot (static objects are always auto-excluded)",
    )
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="Include objects whose position never varies (disabled by default)",
    )
    parser.add_argument(
        "--invert-episode",
        action="store_true",
        help="Plot all episodes NOT in --episode (complement of the subset)",
    )
    parser.add_argument(
        "--robot-ee",
        action="store_true",
        help="Overlay robot end-effector XY positions computed via pybullet FK",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Suffix appended to the output filename, e.g. 'benchmark1000' → episode_object_heatmap_benchmark1000.png",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path entirely (default: auto-generated from --experiment_name and --episode)",
    )
    args = parser.parse_args()

    # Resolve repo ID or path → dataset root (resolve_dataset_dir returns data/,
    # so go up one level to get the root that contains meta/ and data/).
    dataset_root = str(resolve_dataset_dir(args.dataset, args.dataset_dir).parent)
    meta_dir = os.path.join(dataset_root, "meta", "episodes")
    episode_indices = parse_episodes(args.episode) if args.episode else None

    print(f"Loading episode metadata from {meta_dir}...")
    if args.invert_episode and episode_indices is not None:
        # Load all episodes then filter to the complement
        all_rows = load_episode_meta(meta_dir, episode_indices=None)
        exclude = set(episode_indices)
        rows = [r for r in all_rows if r.get("episode_index") not in exclude]
    else:
        rows = load_episode_meta(meta_dir, episode_indices)
    print(f"Loaded {len(rows)} episodes")

    skip = set(args.skip or [])
    position_map = build_position_map(rows, skip=skip, skip_static=not args.include_static)

    for name, pts in sorted(position_map.items()):
        print(f"  {name}: {len(pts)} positions")

    robot_ee_pts = None
    if args.robot_ee:
        print("Computing robot EE positions via FK...")
        robot_ee_pts = compute_ee_positions(rows, dataset_root=dataset_root)
        print(f"  robot EE: {len(robot_ee_pts)} positions")

    # Build output path
    if args.output:
        output = args.output
    else:
        parts = ["episode_object_heatmap"]
        if args.experiment_name:
            parts.append(args.experiment_name)
        if episode_indices is not None:
            if args.invert_episode:
                parts.append(f"not_{len(episode_indices)}subset")
            else:
                parts.append(f"{len(episode_indices)}subset")
        output = os.path.join("outputs", "_".join(parts) + ".png")

    exp_name = args.experiment_name or "unnamed"
    if episode_indices is not None:
        if args.invert_episode:
            subset_label = f"Experiment: {exp_name} (not in {len(episode_indices)} subset)"
        else:
            subset_label = f"Experiment: {exp_name} ({len(episode_indices)} subset)"
    else:
        subset_label = f"Experiment: {exp_name} (all)"
    plot_heatmap(position_map, subset_label=subset_label, output=output, robot_ee_pts=robot_ee_pts)


if __name__ == "__main__":
    main()
