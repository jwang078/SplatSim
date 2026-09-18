"""Check a robot folder before launching it: load the URDF, derive what the
simulator will use (arm joints, gripper, cameras, EE link, action size) and
print it in one screen, with warnings for the things that usually go wrong.

    python scripts/check_robot.py my_robot          # a data/assets/<name>/ or data/stages/<name>/ entry
    python scripts/check_robot.py path/to/robot.urdf # a bare URDF, no yaml (derive everything)

Exit code 0 = the simulator will accept it; 1 = something needs fixing (the
message says what). Nothing is rendered and no scene is loaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pybullet as p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot", help="registry entry name, or a .urdf path")
    ap.add_argument("--wrist-cam-ver", type=int, default=2, help="fisheye calibration for a legacy wrist_camera_link_name")
    args = ap.parse_args()

    from splatsim.configs import registry
    from splatsim.robots.robot_spec import RobotSpec
    from splatsim.utils.paths import resolve_splatsim_path

    if args.robot.endswith(".urdf"):
        cfg = {"urdf_path": str(Path(args.robot).resolve()), "robot": {}}
        name = Path(args.robot).stem
    else:
        cfg = registry.get(args.robot)
        if cfg is None:
            print(f"ERROR: no entry named {args.robot!r} under data/assets/ or data/stages/ "
                  f"(known: {sorted(registry.load_all())})")
            return 1
        name = args.robot
    urdf = resolve_splatsim_path(str(cfg.get("urdf_path", "")))
    if not urdf or not os.path.exists(urdf):
        print(f"ERROR: urdf_path {cfg.get('urdf_path')!r} -> {urdf} does not exist")
        return 1

    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(urdf, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    except p.error as e:
        print(f"ERROR: pybullet cannot load {urdf}: {e}\n       (missing mesh files are the usual cause — paths in the URDF are relative to the URDF)")
        return 1
    try:
        spec = RobotSpec.derive(p, body, cfg, name=name, wrist_cam_ver=args.wrist_cam_ver, urdf_path=urdf)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(spec.summary())
    if not args.robot.endswith(".urdf"):
        lp = registry.labels_path(name)
        if lp.exists():
            import numpy as np
            labels = np.load(lp); key = registry.labels_key(name)
            used = sorted(set(np.unique(labels).astype(int).tolist()))
            if key:
                names = [key["links"].get(str(v), "?") for v in used]
                print(f"  splat labels: {len(labels)} gaussians over {len(used)} links ({lp.name}; key {lp.with_suffix('.json').name})")
                print("    " + ", ".join(f"{v}={n}" for v, n in zip(used, names)))
            else:
                print(f"  splat labels: {len(labels)} gaussians over link indices {used} ({lp.name}; no key file — "
                      f"values are PyBullet link indices of the URDF, -1 = base)")
    warnings = []
    if spec.legacy and "robot" not in cfg:
        warnings.append("no `robot:` block — everything above was derived from the URDF (fine; add the block only to override)")
    if not spec.cameras:
        warnings.append("no cameras: observations will have base_rgb only, and camera-framed goals are disabled")
    if spec.gripper.kind == "per_joint" and spec.gripper.command_dim > 1:
        warnings.append(f"gripper is per_joint with {spec.gripper.command_dim} commands — if the fingers move together, "
                        "declare `gripper: {kind: synergies, open: [...], closed: [...]}` to command it with one value")
    if spec.gripper.kind != "none" and spec.gripper.tip_link_index is None:
        warnings.append("gripper tip: several leaf links, tip will be their centroid (set gripper.tip_link to pin it)")
    for j in [spec.joints[i] for i in spec.arm_joint_indices]:
        if j.lower >= j.upper:
            warnings.append(f"arm joint {j.name!r} has no limits in the URDF (lower >= upper); IK treats it as continuous")
    for w in warnings:
        print(f"  ! {w}")
    print("OK — the simulator will accept this robot." if not any("ERROR" in w for w in warnings) else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
