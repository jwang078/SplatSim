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
import os

import matplotlib.pyplot as plt
import numpy as np
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


def compute_ee_positions(rows: list[dict]) -> np.ndarray:
    """Return (N, 2) array of EE XY positions computed via pybullet FK (headless).

    Loads the URDF once, then teleports joints for each episode and reads ee_link.
    """
    import json
    import pybullet as p

    client = p.connect(p.DIRECT)
    robot = p.loadURDF(SISBOT_URDF, basePosition=SISBOT_BASE_POS,
                       useFixedBase=True, physicsClientId=client)

    xy = []
    for row in rows:
        cfg = row.get("splatsim_robot_config")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        if cfg is None:
            continue
        joints = cfg["articulation_config"]["initial_joint_positions"][:6]
        for ji, jval in zip(SISBOT_ARM_JOINT_INDICES, joints):
            p.resetJointState(robot, ji, jval, physicsClientId=client)
        pos = p.getLinkState(robot, SISBOT_EE_LINK_INDEX, physicsClientId=client)[0]
        xy.append((pos[0], pos[1]))

    p.disconnect(client)
    return np.array(xy)


def build_position_map(rows: list[dict], skip: set[str], skip_static: bool = True) -> dict[str, np.ndarray]:
    """Return {object_name: array of shape (N, 2)} of XY initial positions.

    If skip_static=True (default), objects whose XY position never varies across
    episodes are automatically excluded (e.g. table, wall).
    """
    positions: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        for obj in parse_object_configs(row.get("splatsim_object_configs")):
            name = obj.get("name", "unknown")
            if name in skip:
                continue
            pos = obj.get("initial_position")
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            positions.setdefault(name, []).append((x, y))

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
        default="/home/jennyw2/.cache/huggingface/lerobot/JennyWWW/eval_splatsim_approach_lever_benchmark_1000",
        help="Root of the LeRobot dataset cache directory",
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

    meta_dir = os.path.join(args.dataset, "meta", "episodes")
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
        robot_ee_pts = compute_ee_positions(rows)
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
