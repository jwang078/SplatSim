import numpy as np
import pybullet as p
import pybullet_data
import yaml
import time
import argparse
import zarr
import os
import shutil
from tqdm import tqdm

from splatsim.utils.rrt_path_utils import *


def main(args):
    N_SAMPLES = int(args.robot_update_rate * args.time_per_traj)

    if not args.no_save:
        os.makedirs("output", exist_ok=True)
        output_dir = os.path.join("output", args.experiment_name + ".zarr")
        if args.delete_existing and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        root_output = zarr.open(output_dir, mode="a")

        if "joint_trajectories" not in root_output:
            traj_ds = root_output.create_dataset(
                "joint_trajectories",
                shape=(0, N_SAMPLES, args.dof),
                chunks=(10_000, N_SAMPLES, args.dof),  # example chunk size
                dtype="f4",  # float32
                maxshape=(None, N_SAMPLES, args.dof),
            )
        else:
            traj_ds = root_output["joint_trajectories"]
            # Assert that the existing DOF matches args.DOF
            assert (
                traj_ds.ndim == 3
            ), f"Existing dataset has ndim {traj_ds.ndim}, expected 3"
            assert (
                traj_ds.shape[1] == N_SAMPLES
            ), f"Existing dataset samples {traj_ds.shape[1]} does not match expected {N_SAMPLES}"
            assert (
                traj_ds.shape[2] == args.dof
            ), f"Existing dataset DOF {traj_ds.shape[1]} does not match args.DOF {args.dof}"
        # Start q_start from the end of this dataset
        if traj_ds.shape[0] > 0:
            q_start = traj_ds[-1, -1, :].copy()
            print("Continuing from existing dataset, starting at:", q_start)
        else:
            q_start = None
            print("Did not find existing data, starting fresh for q_start.")
    else:
        q_start = None

    ll, ul, obstacle_ids, robot_id, joint_indices = setup_env(args)

    if q_start is not None and args.start is not None:
        q_start = np.array(args.start)[: args.dof]
    else:
        q_start = get_random_joint_angles_without_collision(
            robot_id, joint_indices, obstacle_ids, ll, ul
        )
        print("Random collision-free start:", q_start)

    if args.goal is not None:
        q_goal = np.array(args.goal)[: args.dof]
    else:
        q_goal = get_random_joint_angles_without_collision(
            robot_id, joint_indices, obstacle_ids, ll, ul
        )
        print("Random collision-free goal:", q_goal)

    buffer = []
    save_interval_s = 60  # 1 minute
    t_last_save = time.time()
    num_traj_collected = 0
    tqdm_bar = tqdm(total=args.num_trajectories)
    while num_traj_collected < args.num_trajectories:
        path = get_path(
            q_start,
            q_goal,
            robot_id,
            joint_indices,
            obstacle_ids,
            ll,
            ul,
            args.time_per_traj,
            args.robot_update_rate,
            rrt_vis_fps=args.rrt_vis_fps,
            use_gui=args.gui,
        )
        if path is None:
            print("Retrying path planning with new random goal...")
            q_goal = get_random_joint_angles_without_collision(
                robot_id, joint_indices, obstacle_ids, ll, ul
            )
            print("New random collision-free goal:", q_goal)
            continue
        if not args.no_save:
            buffer.append(path)
            # Periodically flush to disk
            if time.time() - t_last_save > save_interval_s:
                all_data = np.array(buffer)
                old_size = traj_ds.shape[0]
                new_size = old_size + all_data.shape[0]
                traj_ds.resize((new_size, N_SAMPLES, args.dof))
                traj_ds[old_size:new_size, :, :] = all_data
                print(
                    f"Saved {all_data.shape[0]} new samples to disk. Total is now {new_size}."
                )
                buffer = []
                t_last_save = time.time()
        # For next trajectory, start where we ended
        q_start = path[-1]
        # New random goal
        q_goal = get_random_joint_angles_without_collision(
            robot_id, joint_indices, obstacle_ids, ll, ul
        )
        print("New random collision-free goal:", q_goal)
        tqdm_bar.update(1)
        num_traj_collected += 1
    tqdm_bar.close()
    p.disconnect()


if __name__ == "__main__":
    with open(
        "/home/jennyw2/code/SplatSim/configs/object_configs/objects.yaml", "r"
    ) as file:
        object_config = yaml.safe_load(file)
    robot_name = "robot_iphone_w_engine"
    robot_config = object_config[robot_name]
    SISBOT_PATH = "/home/jennyw2/code/SplatSim/" + robot_config["urdf_path"][0]
    initial_joint_positions = np.array(robot_config["joint_states"][0][1:8])
    num_obstacles = 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf", required=False, help="Path to robot URDF", default=SISBOT_PATH
    )
    parser.add_argument(
        "--start",
        nargs="+",
        type=float,
        help="start joint values (7 floats)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--goal",
        nargs="+",
        type=float,
        help="goal joint values (7 floats)",
        required=False,
        default=None,
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--planner_method",
        type=str,
        default="L-BFGS-B",
        help="Optimization method for scipy minimize",
    )
    parser.add_argument(
        "--cuboids_fn",
        type=str,
        default="/home/jennyw2/code/fabrics/outputs/cuboids_voxel0.050.npz",
        help="Path to npz file with cuboids",
    )

    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--rrt_vis_fps", type=int, default=5)
    parser.add_argument("--time_per_traj", type=float, default=6.0, help="seconds")
    parser.add_argument("--robot_update_rate", type=int, default=20, help="Hz")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="test",
        help="Name for the experiment (for saving results)",
    )
    parser.add_argument(
        "--delete_existing",
        action="store_true",
        help="If set, clear existing output directory",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=1,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not save trajectories to disk",
    )
    parser.add_argument(
        "--use_chomp",
        action="store_true",
        help="If set, use CHOMP-like smoothing after RRT and shortcutting. Note: this implementation doesn't work",
    )
    args = parser.parse_args()
    main(args)
