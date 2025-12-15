import numpy as np
import pybullet as p
import pybullet_data
import yaml
import time
import argparse
import zarr
import os
import shutil
import json
from tqdm import tqdm

from splatsim.utils.rrt_path_utils import *

import pybullet as p
import numpy as np
import re

from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info


# -0.2 for x because we don't want to put obstacles in the wall
TABLE_SPACE = ((-0.2, 0.5), (-0.5, 0.5), (0.0, 0.8))  # x,y,z

def get_random_joint_angles_without_collision(robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits, max_tries=10000, verbose=True):
    sample_fn = get_sample_fn(robot_id, joint_indices)
    for _ in range(max_tries):
        # trying out the pybullet planning version
        q = sample_fn()

        if not state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.01, verbose=verbose):
            return q
    raise RuntimeError("Failed to find collision-free joint angles after many tries")

def get_link_index_by_name(robot_id: int, link_name: str) -> int:
    """
    Finds the PyBullet link index given the link's name (as defined in the URDF/SDF).

    Args:
        client_id: The physics client ID.
        robot_id: The PyBullet object ID of the robot.
        link_name: The string name of the link to search for (e.g., "ee_link").

    Returns:
        The integer link index if found, or -1 if not found.
    """
    
    # PyBullet represents links using the joint index that leads to them.
    num_joints = p.getNumJoints(robot_id)
    
    # Iterate through all joints (indices 0 to num_joints - 1)
    for i in range(num_joints):
        # Get Joint Info returns a tuple with 18 elements.
        # The 12th element (index 12) is the link name (or child link frame name).
        joint_info = p.getJointInfo(robot_id, i)
        current_link_name = joint_info[12].decode("utf-8") # Link name is a byte string, so decode it
        
        if current_link_name == link_name:
            # The joint index is also the link index in PyBullet
            return i 
            
    # If the loop finishes without finding the link
    print(f"Warning: Link '{link_name}' not found for robot ID {robot_id}.")
    return -1

def add_random_obstacles(min_obstacles, max_obstacles, robot_id, robot_qs_to_avoid, base_ee_traj):
    new_obj_ids = []
    new_obj_infos = []
    num_obstacles = np.random.randint(min_obstacles, max_obstacles + 1)
    # Don't put obstacles too close to the start or end configurations
    MIN_TIME_PROPORTION = 0.2
    MAX_TIME_PROPORTION = 0.8
    for _ in range(num_obstacles):
        success = False
        while not success:
            success = True
            # Random position and orientation
            # pos = np.random.uniform(
            #     [TABLE_SPACE[0][0], TABLE_SPACE[1][0], TABLE_SPACE[2][0]],
            #     [TABLE_SPACE[0][1], TABLE_SPACE[1][1], TABLE_SPACE[2][1]]
            # )
            time_proportion = np.random.uniform(MIN_TIME_PROPORTION, MAX_TIME_PROPORTION)
            pos = base_ee_traj[int(len(base_ee_traj) * time_proportion)]
            orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, np.pi)])

            # Random box size
            lx = np.random.uniform(0.02, 0.30)
            ly = np.random.uniform(0.02, 0.30)
            lz = np.random.uniform(0.02, 0.30)

            # Create collision and visual shapes
            body_id = create_box(lx, ly, lz, color=BLUE)
            set_pose(body_id, Pose(point=pos))

            # Check for collisions with the robot
            for robot_q in robot_qs_to_avoid:
                set_robot_joint_positions(robot_id, list(range(p.getNumJoints(robot_id))), robot_q)
                p.stepSimulation()
                collisions = p.getClosestPoints(bodyA=body_id, bodyB=robot_id, distance=0.05)
                if len(collisions) > 0:
                    success = False

            if not success:
                p.removeBody(body_id)  # remove and try again

        new_obj_ids.append(body_id)
        new_obj_infos.append({
            "type": "cuboid",
            "pos": list(pos),
            "orn": [0, 0, 0, 1],
            "size": (lx, ly, lz),
        })

    return new_obj_ids, new_obj_infos

def get_end_effector_pos(
    robot_id: int, 
    ee_link_index: int, 
    joint_positions: list[float],
    joint_indices
) -> list[float]:
    """
    Calculates the forward kinematics for a robot's end-effector 
    given a list of joint positions.

    Args:
        client_id: The physics client ID.
        robot_id: The PyBullet object ID of the robot.
        ee_link_index: The PyBullet link index of the end-effector.
        joint_positions: A list of joint angle values (radians for revolute, meters for prismatic).

    Returns:
        A list [x, y, z] representing the end-effector's position.
    """
    
    # NOTE: You MUST set the robot's joint positions before calling FK.
    # PyBullet's FK function internally uses the current joint state of the robot body.
    
    # 1. Get the indices of the joints that will be moved. 
    # This is necessary because not all links are moving joints (e.g., fixed joints).
    # Assuming 'joint_positions' corresponds to the moving joints in order.
    num_joints = p.getNumJoints(robot_id)
    movable_joints = joint_indices
    

    # 2. Reset the joint state of the robot in the simulation.
    for joint_index, position in zip(movable_joints, joint_positions):
        # The 'targetVelocity' and 'maxForce' are set to 0 to instantaneously place the joint
        p.resetJointState(
            robot_id, 
            joint_index, 
            targetValue=position, 
            targetVelocity=0,
        )

    # 3. Calculate Forward Kinematics
    # PyBullet's FK function will return the position and orientation 
    # of the link specified by 'ee_link_index'
    link_state = p.getLinkState(
        robot_id, 
        ee_link_index, 
        computeForwardKinematics=True, # Explicitly compute FK
    )

    # The format is: (linkWorldPosition, linkWorldOrientation, localInertiaPosition, ...)
    ee_pos = link_state[0]
    # ee_orn = link_state[1] # If you needed the orientation

    return list(ee_pos)


def get_path_through_keyframes(keyframes, robot_id, joint_indices, obstacle_ids, ll, ul,
                                time_per_segment, robot_update_rate, rrt_vis_fps=5,
                                use_gui=False, verbose=True):
    """
    Generate a trajectory that goes through all keyframes in order.

    Args:
        keyframes: List of joint configurations. Each can be:
                   - (6,) array: 6 DOF joint config (gripper defaults to open)
                   - (7,) array: 6 DOF joint config + gripper state (0=open, 1=close)
        robot_id, joint_indices, obstacle_ids, ll, ul: Robot and environment info
        time_per_segment: Time allocated for each segment between keyframes
        robot_update_rate: Hz
        rrt_vis_fps: Visualization FPS
        use_gui: Whether to visualize
        verbose: Print debug info

    Returns:
        Concatenated trajectory array with gripper states, or None if any segment fails
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes to generate a path")

    all_segments = []
    all_gripper_states = []

    # Generate path for each consecutive pair of keyframes
    for i in range(len(keyframes) - 1):
        kf_start = np.array(keyframes[i])
        kf_goal = np.array(keyframes[i + 1])

        # Extract joint positions (first 6 DOF) and gripper states
        q_start = kf_start[:6]
        q_goal = kf_goal[:6]
        gripper_start = kf_start[6] if len(kf_start) > 6 else 0.0
        gripper_goal = kf_goal[6] if len(kf_goal) > 6 else 0.0

        if verbose:
            gripper_str_start = "closed" if gripper_start > 0.5 else "open"
            gripper_str_goal = "closed" if gripper_goal > 0.5 else "open"
            print(f"  Planning segment {i+1}/{len(keyframes)-1}: keyframe {i} (gripper {gripper_str_start}) -> keyframe {i+1} (gripper {gripper_str_goal})")

        segment_path = get_path(
            q_start, q_goal,
            robot_id, joint_indices,
            obstacle_ids,
            ll, ul,
            time_per_segment, robot_update_rate,
            rrt_vis_fps=rrt_vis_fps,
            use_gui=use_gui,
            verbose=verbose
        )

        if segment_path is None:
            if verbose:
                print(f"  Failed to generate path for segment {i+1}")
            return None

        segment_path = np.array(segment_path)
        all_segments.append(segment_path)

        # Interpolate gripper state linearly across the segment
        num_points = len(segment_path)
        gripper_trajectory = np.linspace(gripper_start, gripper_goal, num_points)
        all_gripper_states.append(gripper_trajectory)

    # Concatenate all segments
    # Remove the first point of each segment after the first to avoid duplicates at keyframes
    concatenated_path = [all_segments[0]]
    concatenated_gripper = [all_gripper_states[0]]

    for segment, gripper in zip(all_segments[1:], all_gripper_states[1:]):
        concatenated_path.append(segment[1:])  # Skip first point (duplicate of previous segment's end)
        concatenated_gripper.append(gripper[1:])

    full_trajectory = np.vstack(concatenated_path)
    full_gripper = np.concatenate(concatenated_gripper)

    # Add gripper state as 7th column
    full_trajectory_with_gripper = np.column_stack([full_trajectory, full_gripper])

    if verbose:
        print(f"  Generated full trajectory with {len(full_trajectory_with_gripper)} points (7 DOF with gripper) through {len(keyframes)} keyframes")

    return full_trajectory_with_gripper


def main(args):
    with open(
        "/home/jennyw2/code/SplatSim/configs/object_configs/objects.yaml", "r"
    ) as file:
        object_config = yaml.safe_load(file)
    robot_name = "robot_iphone_w_engine"
    robot_config = object_config[robot_name]
    SISBOT_PATH = "/home/jennyw2/code/SplatSim/" + robot_config["urdf_path"][0]
    initial_joint_positions = np.array(robot_config["joint_states"][0][1:8])
    robot_base_position = robot_config["base_position"][0]

    experiment_name = f"{args.keyframes_file.split("/")[-1]}_{args.experiment_name}"

    # Load keyframes from file if provided
    if args.keyframes_file is not None:
        with open(args.keyframes_file, 'r') as f:
            trajectories_dict = json.load(f)

        print(f"Loaded keyframes from {args.keyframes_file}")
        print(f"Number of trajectories: {len(trajectories_dict)}")

        # Convert dictionary to list of (name, keyframes) tuples
        # Preserve the order and convert keyframes to numpy arrays
        all_trajectories_keyframes = [
            (name, [np.array(kf) for kf in trajectory])
            for name, trajectory in trajectories_dict.items()
        ]

        # Print summary
        for i, (name, traj) in enumerate(all_trajectories_keyframes[:3]):  # Show first 3
            print(f"  '{name}': {len(traj)} keyframes")
        if len(all_trajectories_keyframes) > 3:
            print(f"  ... and {len(all_trajectories_keyframes) - 3} more trajectories")
    else:
        all_trajectories_keyframes = None

    if not args.no_save:
        os.makedirs("output", exist_ok=True)
        output_dir = os.path.join("output", experiment_name + ".zarr")
        if args.delete_existing and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        root_output = zarr.open(output_dir, mode="a")

        scenarios_group = root_output.require_group("trajectories")

        # Find existing scenario indices
        scenario_re = re.compile(r"^scenario_(\d+)$")
        existing_ids = []
        for name in scenarios_group.group_keys():
            m = scenario_re.match(name)
            if m:
                existing_ids.append(int(m.group(1)))

        # Start from next index
        num_prev_scenarios = (max(existing_ids) + 1) if existing_ids else 0
        print(f"Resuming at scenario index {num_prev_scenarios}")

    # Environment setup
    ll, ul, obstacle_ids, robot_id, joint_indices = setup_env(args, robot_base_position, use_old_walls=args.use_old_walls, use_obstacles=not args.no_obstacles)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,      # Distance from target
        cameraYaw=180,             # 0 degrees = looking from +y towards origin
        cameraPitch=-30,         # -30 degrees = looking down at ~30 degree angle
        cameraTargetPosition=[0, 0, 0.3]  # Look at point above the floor
    )

    scenario_i = 0

    # Determine total number of trajectories to generate
    if all_trajectories_keyframes is not None:
        # When using keyframes from file, generate one optimized path per trajectory
        total_trajectories = len(all_trajectories_keyframes)
        print(f"Will generate {total_trajectories} optimized trajectories from keyframes")
    else:
        total_trajectories = args.num_trajectories

    scenario_pbar = tqdm(total=total_trajectories, desc="Trajectories")

    while scenario_i < total_trajectories:
        # Generate keyframes if not provided from file
        if all_trajectories_keyframes is None:
            # Generate random keyframes
            num_keyframes = np.random.randint(args.min_keyframes, args.max_keyframes + 1)
            current_keyframes = []
            trajectory_name = None

            if args.verbose:
                print(f"\nGenerating {num_keyframes} random keyframes...")

            for kf_i in range(num_keyframes):
                kf = get_random_joint_angles_without_collision(
                    robot_id, joint_indices, obstacle_ids, ll, ul, verbose=args.verbose
                )
                current_keyframes.append(kf)
                if args.verbose:
                    print(f"  Keyframe {kf_i + 1}/{num_keyframes}: {kf}")
        else:
            # Use keyframes from file - each trajectory has its own name and set of keyframes
            trajectory_name, current_keyframes = all_trajectories_keyframes[scenario_i]
            num_keyframes = len(current_keyframes)

            if args.verbose:
                print(f"\n[Trajectory {scenario_i + 1}/{total_trajectories}] '{trajectory_name}'")
                print(f"  Loaded {num_keyframes} keyframes from file")
                for kf_i, kf in enumerate(current_keyframes):
                    print(f"  Keyframe {kf_i + 1}: {kf}")

        if args.verbose:
            name_str = f" '{trajectory_name}'" if trajectory_name else ""
            print(f"\n[Trajectory {scenario_i + 1}/{total_trajectories}]{name_str} Generating optimized trajectory through {num_keyframes} keyframes...")

        # Generate path through all keyframes
        full_trajectory = get_path_through_keyframes(
            current_keyframes,
            robot_id, joint_indices,
            obstacle_ids,
            ll, ul,
            args.time_per_segment,
            args.robot_update_rate,
            rrt_vis_fps=args.rrt_vis_fps,
            use_gui=args.gui,
            verbose=args.verbose
        )

        if full_trajectory is None:
            if args.verbose:
                print("  Failed to generate trajectory through keyframes")
            # For file-based keyframes, skip this trajectory and move to next
            if all_trajectories_keyframes is not None:
                if args.verbose:
                    print("  Skipping to next trajectory...")
                scenario_i += 1
                scenario_pbar.update(1)
            continue

        # Save trajectory
        if not args.no_save:
            N_SAMPLES = len(full_trajectory)
            scenario_name = f"scenario_{scenario_i + num_prev_scenarios:04d}"
            if scenario_name in scenarios_group:
                del scenarios_group[scenario_name]  # replace if re-running
            scenario_grp = scenarios_group.create_group(scenario_name)

            # Create obstacle config group (no obstacles in this simplified version)
            # Save keyframes with gripper states (convert to list for JSON serialization)
            keyframes_with_gripper = []
            for kf in current_keyframes:
                kf_array = kf if isinstance(kf, np.ndarray) else np.array(kf)
                keyframes_with_gripper.append(kf_array.tolist())

            metadata = {
                "obstacles": [],
                "num_keyframes": num_keyframes,
                "keyframes": keyframes_with_gripper
            }

            # Add trajectory name if provided
            if trajectory_name is not None:
                metadata["trajectory_name"] = trajectory_name

            obstacle_config_grp = scenario_grp.create_group("obstacle_config_00")
            obstacle_config_grp.attrs["metadata"] = json.dumps(metadata)

            # Save trajectory
            traj_grp = obstacle_config_grp.create_group("traj_00")
            # Note: trajectory now has 7 DOF (6 joints + gripper)
            traj_grp.create_dataset("qs", shape=full_trajectory.shape, data=full_trajectory, dtype="f4")

            if args.verbose:
                name_str = f" '{trajectory_name}'" if trajectory_name else ""
                print(f"  Saved {scenario_name}{name_str} with {len(full_trajectory)} trajectory points through {num_keyframes} keyframes")

        scenario_i += 1
        scenario_pbar.update(1)

    scenario_pbar.close()
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

    parser = argparse.ArgumentParser(
        description="Generate robot trajectories through keyframes using RRT path planning"
    )
    parser.add_argument(
        "--urdf", required=False, help="Path to robot URDF", default=SISBOT_PATH
    )
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
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
    parser.add_argument(
        "--use_old_walls",
        action="store_true",
        help="The old walls (pre-December 2025) had the robot facing forwards when the base was at 0. Now, the robot is facing forward when you rotate the base to be 90 degrees"
    )

    parser.add_argument(
        "--no_obstacles",
        action="store_true"
    )

    parser.add_argument("--dof", type=int, default=6, help="Degrees of freedom")
    parser.add_argument("--rrt_vis_fps", type=int, default=10, help="Visualization FPS for RRT path")
    parser.add_argument("--time_per_segment", type=float, default=2.0,
                        help="Time in seconds allocated for each segment between keyframes")
    parser.add_argument("--robot_update_rate", type=int, default=20, help="Robot update rate in Hz")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="Name for the experiment (for saving results)",
    )
    parser.add_argument(
        "--delete_existing",
        action="store_true",
        help="If set, clear existing output directory",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not save trajectories to disk",
    )

    # Keyframe-related arguments
    parser.add_argument(
        "--keyframes_file",
        type=str,
        default=None,
        help="Path to .json file containing dictionary of named trajectories. "
             "Each trajectory is a list of keyframes. Each keyframe can be: "
             "6 DOF [j1, j2, j3, j4, j5, j6] (gripper defaults to open), or "
             "7 DOF [j1, j2, j3, j4, j5, j6, gripper] where gripper is 0 (open) or 1 (close). "
             "Format: {\"traj1\": [[j1, j2, j3, j4, j5, j6, 0], [j1, j2, j3, j4, j5, j6, 1], ...], ...}. "
             "If not provided, random keyframes will be generated.",
    )
    parser.add_argument(
        "--min_keyframes", type=int, default=2,
        help="Minimum number of keyframes to generate (only used if --keyframes_file is not provided)"
    )
    parser.add_argument(
        "--max_keyframes", type=int, default=5,
        help="Maximum number of keyframes to generate (only used if --keyframes_file is not provided)"
    )
    parser.add_argument(
        "--num_trajectories", type=int, default=100,
        help="Number of trajectories to generate (only used if --keyframes_file is not provided)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose debug information"
    )

    args = parser.parse_args()
    main(args)
