"""Segment a vegetation splat into hard trunk / soft twigs / soft foliage,
with a visualization artifact for EVERY stage so each step can be inspected.

Outputs (under --outdir):
  <name>_trunk_hard.ply     gaussian PLY subset -> input to collision meshing
  <name>_soft_cost.npz      xyz + per-point weight + class for the cost field
  <name>_seg_labels.npy     full-length class array (aligned to input PLY)
  <name>_seg_params.json    every threshold used (reproducibility)
  <name>_class_preview.ply  class-colored point cloud (SuperSplat/CloudCompare)
  <name>_weight_preview.ply weight-colormapped point cloud
  viz/01_input_rgb.png            input cloud in its own colors
  viz/02_opacity_prefilter.png    kept vs dropped points
  viz/03_color_classes.png        after HSV stage only
  viz/04_hue_histogram.png        hue distribution + threshold windows
  viz/05_branch_radius.png        radius estimates + hard/soft threshold
  viz/06_final_classes.png        final labels (trunk red/twig orange/leaf green)
  viz/07_weights.png              soft-cost weight heatmap
  viz/08_confusion.png            (only if a <input>.truth.npy exists)
  viz/09_forced_hard.png          (only with --force-hard-diff)

Usage:
  python scripts/segment_vine_splat.py data/vine_seg/synthetic/vine.ply \
      --outdir data/vine_seg/synthetic [--params my_params.json] [--show]
      [--force-hard-diff vegetation_only_subset.ply]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splatsim.utils.splat_ply_io import (
    read_gaussian_ply,
    write_gaussian_ply_subset,
    write_rgb_preview_ply,
)
from splatsim.utils import splat_segmentation as seg

MAX_PLOT_POINTS = 120_000


def _downsample(n: int, rng=np.random.default_rng(0)):
    if n <= MAX_PLOT_POINTS:
        return np.arange(n)
    return rng.choice(n, MAX_PLOT_POINTS, replace=False)


def scatter_panels(xyz, colors, title, path, point_size=0.6):
    """Three orthographic views (front xz, side yz, top xy) in one figure."""
    idx = _downsample(len(xyz))
    x, y, z = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
    c = colors[idx] if colors.ndim == 2 else colors[idx]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (h, v, hl, vl) in zip(
        axes, [(x, z, "x", "z"), (y, z, "y", "z"), (x, y, "x", "y")]
    ):
        ax.scatter(h, v, s=point_size, c=c, linewidths=0)
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.set_aspect("equal")
    fig.suptitle(f"{title}  (showing {len(idx)}/{len(xyz)} pts)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  viz -> {path}")


def class_colors(labels):
    out = np.zeros((len(labels), 3))
    for cls, col in seg.CLASS_COLORS.items():
        out[labels == cls] = col
    return out


def legend_text():
    return " | ".join(
        f"{seg.CLASS_NAMES[c]}={col}"
        for c, col in [(seg.TRUNK_HARD, "red"), (seg.TWIG_SOFT, "orange"),
                       (seg.FOLIAGE, "green"), (seg.GRAPE, "purple"),
                       (seg.UNKNOWN, "gray"), (seg.DROPPED, "lt-gray")]
    )


def plot_hue_histogram(result, path):
    keep = result.labels != seg.DROPPED
    hue, sat = result.hsv[keep, 0], result.hsv[keep, 1]
    p = result.params
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(hue, bins=180, range=(0, 360), color="0.4")
    ax.axvspan(*p.green_hue_range, alpha=0.25, color="green",
               label=f"green window {p.green_hue_range}")
    ax.axvspan(*p.brown_hue_range, alpha=0.25, color="peru",
               label=f"brown window {p.brown_hue_range}")
    for i, (lo, hi) in enumerate(p.grape_hue_ranges):
        label = f"grape windows {p.grape_hue_ranges}" if i == 0 else None
        ax.axvspan(lo, min(hi, 360), alpha=0.25, color="purple", label=label)
        if hi > 360:  # wrap-around window continues past the 0deg seam
            ax.axvspan(0, hi - 360, alpha=0.25, color="purple")
    ax.set_xlabel("hue (deg)")
    ax.set_ylabel("# gaussians")
    ax.set_title(f"Hue distribution (kept points; median sat={np.median(sat):.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  viz -> {path}")


def plot_radius_histogram(result, path):
    r = result.branch_radius
    r = r[np.isfinite(r)]
    p = result.params
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(r):
        ax.hist(r * 1000.0, bins=80, range=(0, max(25, r.max() * 1000)),
                color="0.4")
    ax.axvline(p.hard_min_radius * 1000.0, color="red",
               label=f"hard_min_radius = {p.hard_min_radius*1000:.0f} mm")
    ax.set_xlabel("estimated local branch radius (mm)")
    ax.set_ylabel("# branch points")
    ax.set_title("Branch radius estimates — right of red line becomes TRUNK_HARD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  viz -> {path}")


def plot_confusion(labels, truth, path):
    classes = [seg.FOLIAGE, seg.GRAPE, seg.TWIG_SOFT, seg.TRUNK_HARD, seg.DROPPED]
    names = [seg.CLASS_NAMES[c] for c in classes]
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    for i, t in enumerate(classes):
        for j, l in enumerate(classes):
            mat[i, j] = int(((truth == t) & (labels == l)).sum())
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("truth")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    acc = np.trace(mat) / max(mat.sum(), 1)
    ax.set_title(f"Truth vs predicted (acc={acc:.1%})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  viz -> {path}  (accuracy {acc:.1%})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input_ply")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--params", help="SegmentationParams JSON to load")
    ap.add_argument("--show", action="store_true",
                    help="also open an interactive open3d window at the end")
    ap.add_argument("--force-hard-diff", metavar="SUBSET_PLY",
                    help="PLY holding the vegetation-only subset of the input "
                    "(same scan, identical xyz values). Input points NOT in "
                    "it are man-made structure (trellis/frame): kept ones are "
                    "forced to TRUNK_HARD and excluded from the soft cost.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    vizdir = outdir / "viz"
    vizdir.mkdir(parents=True, exist_ok=True)
    name = Path(args.input_ply).stem

    params = (
        seg.SegmentationParams.from_json(args.params)
        if args.params else seg.SegmentationParams()
    )

    print(f"reading {args.input_ply} ...")
    cloud = read_gaussian_ply(args.input_ply)
    print(f"  {len(cloud)} gaussians")

    scatter_panels(cloud.xyz, cloud.rgb, "01 input (splat RGB)",
                   vizdir / "01_input_rgb.png")

    print("segmenting ...")
    result = seg.segment_vegetation_splat(
        cloud.xyz, cloud.rgb, cloud.opacity, params, scales=cloud.scales
    )

    structure = forced = None
    if args.force_hard_diff:
        # Points absent from the vegetation-only subset are man-made
        # structure (trellis/frame): rigid by definition, so they bypass the
        # color/radius heuristics and go straight to the hard collision mesh.
        subset = read_gaussian_ply(args.force_hard_diff)

        def _keys(xyz):
            buf = np.ascontiguousarray(xyz.astype(np.float32))
            return buf.view([("", np.float32)] * 3).ravel()

        structure = ~np.isin(_keys(cloud.xyz), _keys(subset.xyz))
        forced = structure & (result.labels != seg.DROPPED)
        result.labels[forced] = seg.TRUNK_HARD
        result.weights[forced] = 0.0
        print(
            f"force-hard diff vs {args.force_hard_diff}: "
            f"{int(structure.sum())} structure points, "
            f"{int(forced.sum())} kept -> TRUNK_HARD, "
            f"{int((structure & ~forced).sum())} stay dropped (opacity prefilter)"
        )

    kept = result.labels != seg.DROPPED
    drop_colors = np.where(kept[:, None], [[0.2, 0.2, 0.8]], [[0.85, 0.85, 0.85]])
    scatter_panels(
        cloud.xyz, drop_colors,
        f"02 opacity prefilter (blue kept={kept.sum()}, "
        f"gray dropped={len(cloud)-kept.sum()}, thr={params.min_opacity})",
        vizdir / "02_opacity_prefilter.png",
    )
    scatter_panels(cloud.xyz, class_colors(result.color_labels),
                   f"03 color stage only — {legend_text()}",
                   vizdir / "03_color_classes.png")
    plot_hue_histogram(result, vizdir / "04_hue_histogram.png")
    plot_radius_histogram(result, vizdir / "05_branch_radius.png")
    scatter_panels(cloud.xyz, class_colors(result.labels),
                   f"06 FINAL classes — {legend_text()}",
                   vizdir / "06_final_classes.png")

    w = result.weights
    wnorm = w / max(w.max(), 1e-9)
    weight_rgb = plt.get_cmap("viridis")(wnorm)[:, :3]
    weight_rgb[w <= 0] = [0.9, 0.9, 0.9]
    scatter_panels(cloud.xyz, weight_rgb,
                   f"07 soft-cost weights (viridis, max={w.max():.2f}; "
                   f"multipliers={params.class_multiplier})",
                   vizdir / "07_weights.png")

    truth_path = Path(args.input_ply).with_suffix(".truth.npy")
    if truth_path.exists():
        plot_confusion(result.labels, np.load(truth_path),
                       vizdir / "08_confusion.png")

    if structure is not None:
        forced_colors = np.tile([[0.75, 0.9, 0.75]], (len(cloud), 1))
        forced_colors[structure & ~forced] = [0.85, 0.85, 0.85]
        forced_colors[forced] = [0.7, 0.1, 0.1]
        scatter_panels(
            cloud.xyz, forced_colors,
            f"09 forced-hard structure (diff vs {Path(args.force_hard_diff).name}) "
            f"— red forced hard={int(forced.sum())}, "
            f"lt-gray structure dropped={int((structure & ~forced).sum())}, "
            f"pale-green vegetation={int((~structure).sum())}",
            vizdir / "09_forced_hard.png",
        )

    # ---- artifacts ----
    n_hard = write_gaussian_ply_subset(
        cloud, result.hard_mask, outdir / f"{name}_trunk_hard.ply"
    )
    soft = result.soft_mask
    np.savez_compressed(
        outdir / f"{name}_soft_cost.npz",
        xyz=cloud.xyz[soft],
        weight=result.weights[soft],
        class_id=result.labels[soft],
        params_json=json.dumps(asdict(params)),
    )
    np.save(outdir / f"{name}_seg_labels.npy", result.labels)
    params.to_json(outdir / f"{name}_seg_params.json")
    write_rgb_preview_ply(cloud.xyz, class_colors(result.labels),
                          outdir / f"{name}_class_preview.ply")
    write_rgb_preview_ply(cloud.xyz, weight_rgb,
                          outdir / f"{name}_weight_preview.ply")

    counts = {seg.CLASS_NAMES[c]: int((result.labels == c).sum())
              for c in np.unique(result.labels)}
    print(f"\nclass counts: {counts}")
    print(f"trunk-hard PLY: {n_hard} gaussians -> {name}_trunk_hard.ply")
    print(f"soft-cost points: {int(soft.sum())} -> {name}_soft_cost.npz")
    print(f"all viz PNGs in {vizdir}/")

    if args.show:
        import open3d as o3d

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud.xyz)
        pc.colors = o3d.utility.Vector3dVector(class_colors(result.labels))
        o3d.visualization.draw_geometries([pc], window_name="final classes")


if __name__ == "__main__":
    main()
