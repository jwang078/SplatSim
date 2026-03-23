#!/usr/bin/env python3
"""Convert a SplatSim zarr trajectory file to LeRobot format and push to hub.

Usage:
    python scripts/zarr_to_lerobot.py \
        --zarr-path output/JennyWWW_eval_splatsim_approach_lever_benchmarking_1000_trajectories.zarr \
        --repo-id JennyWWW/eval_splatsim_approach_lever_benchmarking_1000 \
        --task "Approach the lever" \
        --fps 20

Data mapping from zarr -> LeRobot:
    zarr qs (T, 6) float32        -> action (T, 7) float32  [pad gripper=0]
    zarr qs (T, 6) float32        -> observation.state (T, 7) float32  [same — no physics readback in zarr]
    zarr {cam}_letterbox (T,H,W,C) float32 0-255 -> observation.images.{cam}_letterbox (3,H,W) uint8
    obstacle metadata (.zattrs)   -> episode metadata
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_MODES = [
    "base_rgb_letterbox",
    "base_rgb_stretch",
    "wrist_rgb_letterbox",
    "wrist_rgb_stretch",
]

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def create_dataset(repo_id: str, fps: int) -> LeRobotDataset:
    features = {
        **{
            f"observation.images.{cam}": {
                "dtype": "image",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            }
            for cam in CAMERA_MODES
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="lerobot_splatsim",
        use_videos=True,
        features=features,
    )


def zarr_img_to_lerobot(img_hwc_f32: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) float32 0-255 -> (C, H, W) uint8."""
    img = np.clip(img_hwc_f32, 0, 255).astype(np.uint8)
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


def main():
    parser = argparse.ArgumentParser(description="Convert SplatSim zarr -> LeRobot dataset")
    parser.add_argument("--zarr-path", required=True, help="Path to .zarr file")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (user/dataset)")
    parser.add_argument("--task", default="Approach the lever", help="Task description string")
    parser.add_argument("--fps", type=int, default=20, help="Dataset FPS (default: 20)")
    parser.add_argument("--push-to-hub", action="store_true", default=True, help="Push to HuggingFace Hub after creation")
    parser.add_argument("--no-push", action="store_true", help="Skip push to hub")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes (for testing)")
    args = parser.parse_args()

    push = args.push_to_hub and not args.no_push

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"ERROR: zarr path not found: {zarr_path}")
        sys.exit(1)

    print(f"Opening zarr: {zarr_path}")
    root = zarr.open(str(zarr_path), "r")
    trajs_group = root["trajectories"]
    scenarios = sorted(trajs_group.keys())
    total = len(scenarios) if args.max_episodes is None else min(len(scenarios), args.max_episodes)
    print(f"Found {len(scenarios)} scenarios, converting {total}")

    print(f"Creating LeRobot dataset: {args.repo_id}")
    # Clear any stale/corrupt local cache before creating fresh
    import shutil, os
    local_dir = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    if os.path.exists(local_dir):
        print(f"Removing existing local cache at {local_dir}")
        shutil.rmtree(local_dir)
    dataset = create_dataset(args.repo_id, args.fps)

    for ep_idx, scenario_name in enumerate(scenarios[:total]):
        if ep_idx % 50 == 0:
            print(f"  Episode {ep_idx}/{total} ...")

        obs_grp = trajs_group[scenario_name]["obstacle_config_00"]
        traj_grp = obs_grp["traj_00"]

        # Load joint trajectory (T, 6)
        qs = traj_grp["qs"][:]  # (T, 6) float32

        # Load images for each camera/mode
        cam_images = {}
        for cam in CAMERA_MODES:
            if cam in traj_grp:
                arr = traj_grp[cam][:]  # (T, H, W, C) float32, values 0-255
                cam_images[cam] = arr
            else:
                cam_images[cam] = None

        # Skip episodes missing any image data
        if any(v is None for v in cam_images.values()):
            print(f"  Skipping {scenario_name}: missing image data")
            continue

        # Use the minimum length across qs and all image arrays (some may be shorter)
        T = qs.shape[0]
        for arr in cam_images.values():
            if arr is not None:
                T = min(T, arr.shape[0])

        # Pad to 7 DOF (gripper = 0)
        state_action = np.zeros((T, 7), dtype=np.float32)
        state_action[:, :6] = qs[:T]

        # Load obstacle metadata
        obstacle_meta = json.loads(obs_grp.attrs.get("metadata", "{}"))

        # Add frames to dataset
        for t in range(T):
            frame = {
                "observation.state": state_action[t],
                "action": state_action[t],
                "task": args.task,
            }
            for cam in CAMERA_MODES:
                if cam_images[cam] is not None:
                    frame[f"observation.images.{cam}"] = zarr_img_to_lerobot(cam_images[cam][t])
            dataset.add_frame(frame)

        # Save episode with metadata — serialize obstacle_info as JSON string to avoid
        # parquet schema mismatches between episodes (empty obstacles list infers list<null>)
        dataset.save_episode(episode_metadata={"obstacle_info": json.dumps(obstacle_meta), "zarr_scenario": scenario_name})

    print(f"Finalizing dataset ({total} episodes)...")
    dataset.finalize()

    if push:
        print(f"Pushing to hub as '{args.repo_id}'...")
        try:
            dataset.push_to_hub()
            print("Successfully pushed to hub.")
        except Exception as e:
            print(f"ERROR pushing to hub: {e}")
            print("Dataset is saved locally. You can retry with: dataset.push_to_hub()")
            sys.exit(1)
    else:
        print("Skipping push to hub (--no-push).")

    print("Done.")


if __name__ == "__main__":
    main()
