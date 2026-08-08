"""Regenerate grape_targets.json from a scene's segmentation output.

Re-clusters the GRAPE-class points and writes the bunch list the vine env
consumes — including the ``peduncle`` field (top of the bunch, where it
attaches to the vine) that older target files predate.

Points come from the pair of npz files the segmentation pipeline already
wrote, so no PLY re-read and no splat->sim transform are needed:
  <scene>_soft_cost.npz          xyz + weight + class_id   (splat frame)
  <scene>_cost_field_sim.npz     points + weights          (SIM frame)
Both carry one row per soft point in the same order (verified by comparing
the weight columns), so the class ids from the first index straight into the
sim-frame coordinates of the second.

Usage:
    python scripts/regen_grape_targets.py                       # vine_and_trellis
    python scripts/regen_grape_targets.py --scene-dir data/vine_seg/<scene>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from splatsim.utils import grape_targets as G
from splatsim.utils import splat_segmentation as seg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path,
                    default=Path("data/vine_seg/vine_and_trellis"))
    ap.add_argument("--name", default=None,
                    help="asset basename (default: scene dir name)")
    ap.add_argument("--eps", type=float, default=0.025,
                    help="DBSCAN-ish linking radius (m)")
    ap.add_argument("--min-points", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the bunches without writing the JSON")
    args = ap.parse_args()

    name = args.name or args.scene_dir.name
    soft = np.load(args.scene_dir / f"{name}_soft_cost.npz", allow_pickle=False)
    simf = np.load(args.scene_dir / f"{name}_cost_field_sim.npz", allow_pickle=False)

    if soft["weight"].shape != simf["weights"].shape or not np.allclose(
            soft["weight"], simf["weights"]):
        raise SystemExit(
            "soft_cost.npz and cost_field_sim.npz do not describe the same "
            "point set in the same order — cannot map class ids into sim "
            "coordinates. Re-run the segmentation so both are regenerated "
            "together.")

    pts_sim = np.asarray(simf["points"], dtype=np.float64)
    class_id = np.asarray(soft["class_id"])
    print(f"{len(pts_sim):,} soft points, {(class_id == seg.GRAPE).sum():,} "
          f"of class GRAPE")

    bunches = seg.cluster_labeled_points(
        pts_sim, class_id, seg.GRAPE, eps=args.eps,
        min_points=args.min_points, transform=None,
    )
    print(f"\n{len(bunches)} bunch(es), largest first:")
    for i, b in enumerate(bunches):
        c = np.asarray(b["center"])
        ped = np.asarray(b["peduncle"])
        print(f"  {i}: n={b['n_points']:5d}  center={np.round(c, 3).tolist()}  "
              f"peduncle={np.round(ped, 3).tolist()}  "
              f"(peduncle is {(ped[2] - c[2]) * 100:+.1f} cm in z)")

    # ALWAYS the detector file, never grape_targets_manual.json — hand
    # annotation is not automation's to overwrite.
    out = args.scene_dir / G.AUTO_TARGETS_NAME
    if args.dry_run:
        print(f"\n--dry-run: not writing {out}")
        return
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:                       # noqa: BLE001
            existing = []
        if G.is_manual(existing):
            raise SystemExit(
                f"REFUSING to overwrite {out}: it contains hand-marked "
                f"targets ({sum(1 for b in existing if b.get('manual'))} of "
                f"{len(existing)}). Move them to "
                f"{args.scene_dir / G.MANUAL_TARGETS_NAME} (which this script "
                f"never writes and the env prefers), or delete the file if "
                f"you really mean to discard the annotation.")
    out.write_text(json.dumps(bunches, indent=1))
    manual = args.scene_dir / G.MANUAL_TARGETS_NAME
    print(f"\nwrote {out}")
    if manual.exists():
        print(f"NOTE {manual.name} exists and takes precedence — the env will "
              f"keep using the hand-marked targets, not what was just written.")


if __name__ == "__main__":
    main()
