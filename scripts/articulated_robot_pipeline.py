#!/usr/bin/env python
"""Kept for muscle memory. The pipeline is now scripts/segment_stage_asset.py,
which does the same in three named steps (pcd -> align in CloudCompare ->
transform -> labels) for any body of a stage, not only the robot:

    python scripts/segment_stage_asset.py <stage> --asset robot pcd
    python scripts/segment_stage_asset.py <stage> --asset robot transform <matrix>
    python scripts/segment_stage_asset.py <stage> --asset robot labels --show

`--robot_name X` here runs `X --asset robot all --show`.
"""
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    name = None
    for i, a in enumerate(args):
        if a == "--robot_name" and i + 1 < len(args):
            name = args[i + 1]
        elif a.startswith("--robot_name="):
            name = a.split("=", 1)[1]
    if name is None:
        print(__doc__)
        sys.exit(2)
    print(f"[articulated_robot_pipeline] -> scripts/segment_stage_asset.py {name} --asset robot all --show")
    sys.exit(subprocess.call([sys.executable, "scripts/segment_stage_asset.py", name, "--asset", "robot", "all", "--show"]))
