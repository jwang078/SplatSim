"""Inspect a vine collision URDF + soft-cost field inside PyBullet.

Headless mode (default) — automated verification, writes artifacts:
  viz/12_pybullet_probe.png    camera render of the loaded collision body
  stdout probe table           getClosestPoints distance + soft cost at probe
                               points (inside trunk / in canopy / free space)

GUI mode (--gui) — interactive inspection:
  - trunk collision mesh loaded as a fixed body
  - soft-cost source points scattered as colored debug points (viridis by w)
  - a red probe sphere driven by x/y/z sliders; distance-to-trunk and
    soft-cost at the probe are re-printed whenever it moves

Usage:
  python scripts/visualize_vine_collision.py \
      --urdf data/vine_seg/synthetic/vine.urdf \
      --soft-npz data/vine_seg/synthetic/vine_soft_cost.npz [--gui]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

from splatsim.utils.soft_cost_field import SoftCostField


def probe_report(client, body_id, field, probes):
    print(f"\n{'probe (m)':<28}{'dist to trunk (m)':<20}{'soft cost':<12}note")
    for label, point in probes:
        col = client.createCollisionShape(pb.GEOM_SPHERE, radius=0.008)
        probe_body = client.createMultiBody(0, baseCollisionShapeIndex=col,
                                            basePosition=point)
        cps = client.getClosestPoints(probe_body, body_id, distance=2.0)
        dist = min((c[8] for c in cps), default=float("inf"))
        cost = field.cost_at(np.array([point]))[0] if field else float("nan")
        client.removeBody(probe_body)
        print(f"{str(np.round(point, 3)):<28}{dist:<20.4f}{cost:<12.3f}{label}")


def render_scene(client, out_path, target, distance=1.8):
    view = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target, distance=distance,
        yaw=45, pitch=-25, roll=0, upAxisIndex=2)
    proj = client.computeProjectionMatrixFOV(fov=55, aspect=4 / 3,
                                             nearVal=0.05, farVal=10)
    w, h, rgb, _, _ = client.getCameraImage(
        1280, 960, view, proj, renderer=pb.ER_TINY_RENDERER)
    img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, img)
    print(f"viz -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--soft-npz", help="segmentation soft_cost.npz (optional)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--outdir", default=None,
                    help="viz output dir (default: <urdf dir>/viz)")
    args = ap.parse_args()

    import pybullet_utils.bullet_client as bc

    client = bc.BulletClient(pb.GUI if args.gui else pb.DIRECT)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())

    body_id = client.loadURDF(args.urdf, useFixedBase=True)
    aabb_min, aabb_max = client.getAABB(body_id)
    center = (np.array(aabb_min) + np.array(aabb_max)) / 2
    extent = np.array(aabb_max) - np.array(aabb_min)
    print(f"loaded {args.urdf}")
    print(f"  AABB min={np.round(aabb_min, 3)} max={np.round(aabb_max, 3)}")

    field = None
    if args.soft_npz:
        field = SoftCostField.from_config({"npz_path": args.soft_npz})
        print(f"  soft-cost field: {len(field.points)} pts, "
              f"grid {tuple(field.grid.shape)}, max={field.max_cost():.2f}")

    # probes: trunk center (should collide), canopy centroid (soft cost > 0,
    # positive trunk distance), far free space (no contact, zero cost)
    probes = [("inside/near trunk", center),
              ("free space", center + extent * 1.5 + 0.5)]
    if field is not None:
        canopy = field.points[field.weights > np.percentile(field.weights, 80)]
        probes.insert(1, ("dense canopy", canopy.mean(axis=0)))
    probe_report(client, body_id, field, probes)

    outdir = Path(args.outdir) if args.outdir else Path(args.urdf).parent / "viz"

    if not args.gui:
        render_scene(client, outdir / "12_pybullet_probe.png", center,
                     distance=max(2.0 * float(extent.max()), 1.0))
        client.disconnect()
        return

    # ---------------- interactive GUI mode ----------------
    if field is not None:
        field.draw_in_pybullet(client)
        print("soft points drawn (viridis: dark=low cost, yellow=high)")

    sliders = [client.addUserDebugParameter(axis, lo, hi, float(c))
               for axis, lo, hi, c in zip(
                   "xyz",
                   center - extent * 1.5,
                   center + extent * 1.5,
                   center)]
    vis = client.createVisualShape(pb.GEOM_SPHERE, radius=0.015,
                                   rgbaColor=[1, 0, 0, 0.9])
    col = client.createCollisionShape(pb.GEOM_SPHERE, radius=0.015)
    probe_body = client.createMultiBody(0, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=center)
    last = None
    shot_idx = 0
    print("drag x/y/z sliders to move the red probe; Ctrl+C to quit")
    print("rotate/zoom with the mouse, then press 'c' to save a screenshot "
          f"of the current view into {outdir}/")
    while client.isConnected():
        keys = client.getKeyboardEvents()
        if ord("c") in keys and keys[ord("c")] & pb.KEY_WAS_TRIGGERED:
            # capture exactly what the GUI camera shows right now
            cam = client.getDebugVisualizerCamera()
            w_res, h_res, view, proj = cam[0], cam[1], cam[2], cam[3]
            _, _, rgb, _, _ = client.getCameraImage(
                w_res, h_res, view, proj,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            outdir.mkdir(parents=True, exist_ok=True)
            shot = outdir / f"gui_screenshot_{shot_idx:02d}.png"
            img = np.reshape(rgb, (h_res, w_res, 4))[:, :, :3].astype(np.uint8)
            plt.imsave(shot, img)
            shot_idx += 1
            print(f"\nsaved {shot}")
        pos = [client.readUserDebugParameter(s) for s in sliders]
        if last is None or np.linalg.norm(np.array(pos) - last) > 1e-4:
            last = np.array(pos)
            client.resetBasePositionAndOrientation(probe_body, pos, [0, 0, 0, 1])
            cps = client.getClosestPoints(probe_body, body_id, distance=2.0)
            dist = min((c[8] for c in cps), default=float("inf"))
            cost = field.cost_at(np.array([pos]))[0] if field else float("nan")
            print(f"\rprobe={np.round(pos, 3)}  trunk-dist={dist: .4f} m  "
                  f"soft-cost={cost:8.3f}", end="")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
