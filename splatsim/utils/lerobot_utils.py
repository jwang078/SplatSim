"""Shared utilities for building LeRobot dataset frames and feature specs.

These functions are sim-agnostic: they work with any observation dict that
contains numpy image arrays and joint_positions / gripper_position keys.
Used by both the sim server's _render_and_save_episode and by run_env_sim.py
(which may be running against the real UR robot instead of the simulator).
"""

import os
import shutil
from typing import List, Optional

import numpy as np


def build_lerobot_features(image_keys: List[str], num_dofs: int = 6) -> dict:
    """Build the standard LeRobotDataset feature spec for SplatSim recordings.

    Args:
        image_keys: obs dict keys to use as images
                    (e.g. ["base_rgb_letterbox"] from the sim,
                     or ["base_rgb"] from a real-robot wrapper)
        num_dofs: number of arm joints (gripper adds 1 more, so state/action
                  shape will be num_dofs + 1)
    """
    dof_names = [f"joint_{i+1}" for i in range(num_dofs)] + ["gripper"]
    return {
        **{
            f"observation.images.{key}": {
                "dtype": "image",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            }
            for key in image_keys
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (num_dofs + 1,),
            "names": dof_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (num_dofs + 1,),
            "names": dof_names,
        },
    }


def load_lerobot_dataset(repo_id: str) -> Optional["LeRobotDataset"]:
    """Load an existing LeRobot dataset, preferring local cache over the hub.

    ``LeRobotDataset(repo_id)`` without ``root`` always hits the hub to
    validate the repo, failing with ``RepositoryNotFoundError`` for datasets
    that were only saved locally and never pushed.  This function passes
    ``root=local_dir`` to load from local cache without a hub request.

    When a local dir exists but is partially initialized (e.g. only info.json,
    no tasks.parquet — can happen after a crash before the first episode was
    saved), ``load_metadata()`` raises ``FileNotFoundError`` which causes
    LeRobot to fall back to the hub.  If the hub also 404s, we clean up the
    partial dir and return None so the caller can create a fresh dataset.

    Returns None if the dataset is not found locally or on the hub.
    """
    from huggingface_hub.errors import RepositoryNotFoundError as HubNotFoundError
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    local_dir = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}")
    local_info = os.path.join(local_dir, "meta", "info.json")

    if os.path.exists(local_info):
        try:
            dataset = LeRobotDataset(repo_id, root=local_dir)
            print(f"[LeRobot] Loaded existing dataset from local cache ({dataset.meta.total_episodes} episodes).")
            return dataset
        except HubNotFoundError:
            # Local dataset is incomplete (triggered hub fallback) and hub also 404s.
            # Clean up so the caller can create fresh.
            print(f"[LeRobot] Local dataset at {local_dir} is incomplete and not on hub; will create fresh.")
            shutil.rmtree(local_dir)
            return None
        except Exception as e:
            print(f"[LeRobot] WARNING: Failed to load local dataset: {e}")

    try:
        dataset = LeRobotDataset(repo_id)
        print(f"[LeRobot] Loaded dataset from hub ({dataset.meta.total_episodes} episodes).")
        return dataset
    except Exception as e:
        print(f"[LeRobot] Dataset not found locally or on hub: {e}")
        # Clean up any partial dir left by the failed load attempt.
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        return None


def create_lerobot_dataset(
    repo_id: str,
    fps: int,
    image_keys: List[str],
    num_dofs: int = 6,
    robot_type: str = "lerobot_splatsim",
) -> "LeRobotDataset":
    """Create a fresh LeRobot dataset with standard SplatSim settings."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        use_videos=True,
        features=build_lerobot_features(image_keys, num_dofs),
    )


def finalize_lerobot_dataset(dataset: "LeRobotDataset") -> None:
    """Discard any partial in-progress episode and finalize the dataset.

    Calls ``clear_episode_buffer()`` first (no-op if not supported) to drop
    any frames that were added but never committed via ``save_episode()``.
    """
    try:
        dataset.clear_episode_buffer()
    except Exception:
        pass
    dataset.finalize()


def push_lerobot_to_hub(dataset: "LeRobotDataset") -> None:
    """Push dataset to hub, retrying with a new repo_id on failure.

    Loops until the push succeeds or the user presses Enter to skip.
    """
    while True:
        repo_id = dataset.repo_id
        print(f"[LeRobot] Pushing dataset to hub as '{repo_id}'...")
        try:
            dataset.push_to_hub()
            print(f"[LeRobot] Successfully pushed to hub as '{repo_id}'.")
            return
        except Exception as e:
            print(f"[LeRobot] ERROR: Failed to push to hub: {e}")
            print("[LeRobot] Repo ID should be in 'username/dataset_name' format.")
            print("[LeRobot] Make sure you are authenticated with `huggingface-cli login`.")
            new_repo_id = input("[LeRobot] Enter a new repo_id to retry (or press Enter to skip): ").strip()
            if not new_repo_id:
                print("[LeRobot] Skipping push to hub. Dataset is saved locally.")
                return
            dataset.repo_id = new_repo_id


def build_lerobot_frame(
    action_7: np.ndarray,
    obs: dict,
    image_keys: List[str],
    task: str = "",
) -> dict:
    """Build a frame dict suitable for LeRobotDataset.add_frame().

    Convention: ``obs`` should be the *post-step* observation — the robot state
    after the physics step that followed ``action_7``.  Both
    ``_render_and_save_episode`` (traj-gen) and ``run_env_sim.py`` (interactive)
    follow this convention so that saved datasets are directly comparable.

    Args:
        action_7: 7-DOF commanded joint state [j0..j5, gripper]
        obs: observation dict from get_observations()
        image_keys: obs dict keys to include as images (values must be
                    (C, H, W) float32 arrays in [0, 1])
        task: task description string (from get_task_description() or "")
    """
    state_7 = np.zeros(7, dtype=np.float32)
    state_7[:6] = np.asarray(obs["joint_positions"])[:6]
    state_7[6] = np.asarray(obs.get("gripper_position", [0.0]))[0]

    frame: dict = {
        "observation.state": state_7,
        "action": np.asarray(action_7, dtype=np.float32),
        "task": task,
    }
    for key in image_keys:
        if obs.get(key) is not None:
            frame[f"observation.images.{key}"] = obs[key]
    return frame
