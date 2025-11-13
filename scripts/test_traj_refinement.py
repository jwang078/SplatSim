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

        # q = np.random.uniform(lower_limits, upper_limits)
        # if not state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.0) and not check_self_collision(robot_id, joint_indices):
        #     return q

        # random end effector pose then use inverse kinematics
        # pos = np.random.uniform(
        #     [TABLE_SPACE[0][0], TABLE_SPACE[1][0], TABLE_SPACE[2][0]],
        #     [TABLE_SPACE[0][1], TABLE_SPACE[1][1], TABLE_SPACE[2][1]]
        # )
        # orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, np.pi)])
        # # inverse kinematics with obstacle ids
        # q = p.calculateInverseKinematics(robot_id, joint_indices[-1], pos, orn)

        # trying out the pybullet planning version
        q = sample_fn()

        if not state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.01, verbose=verbose):
            return q
    raise RuntimeError("Failed to find collision-free joint angles after many tries")

def check_intersection_raycast(object_id: int, points: list[list[float]]) -> bool:
    """
    Checks if a cuboid (object_id) intersects with any of the given XYZ points 
    by casting a tiny ray at each point's location.

    Args:
        client_id: The physics client ID (e.g., p.connect(p.GUI)).
        object_id: The PyBullet object ID of the cuboid.
        points: A list of [x, y, z] lists.

    Returns:
        True if any point is intersecting (inside) the cuboid, False otherwise.
    """
    TINY_OFFSET = 0.001  # A small offset for the ray
    
    # Get the cuboid's visual shape data to approximate its bounds (optional, but good for context)
    # shapes = p.getVisualShapeData(object_id, physicsClientId=client_id)
    # cuboid_dims = shapes[0][3]  # [half_extents_x, half_extents_y, half_extents_z]
    
    # Get the cuboid's current position and orientation
    cuboid_pos, _ = p.getBasePositionAndOrientation(object_id) #, physicsClientId=client_id)
    
    # 1. Cast a ray from just outside the point to just inside, or just a tiny ray
    for point in points:
        # Cast a tiny ray: from (point - offset) to (point + offset)
        # If the point is inside the cuboid, the ray test will report a hit
        start_pos = [point[0] - TINY_OFFSET, point[1], point[2]]
        end_pos = [point[0] + TINY_OFFSET, point[1], point[2]]

        # The ray test returns a list of hit data (closest fraction, hit body, link index, hit position, hit normal)
        results = p.rayTest([start_pos], [end_pos]) #, physicsClientId=client_id)
        
        # Check if the ray hit the target object
        # The result format is: (hitFraction, hitBodyUniqueId, linkIndex, hitPosition, hitSurfaceNormal)
        if results and results[0][1] == object_id:
            # We found an intersection
            return True

    return False

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

            # collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_l, half_w, half_h])
            # visual_shape = p.createVisualShape(
            #     p.GEOM_BOX,
            #     halfExtents=[half_l, half_w, half_h],
            #     rgbaColor=[0, 0, 1, 1]
            # )

            # # Create the actual rigid body in the world
            # body_id = p.createMultiBody(
            #     baseMass=0,  # static obstacle (no gravity)
            #     baseCollisionShapeIndex=collision_shape,
            #     baseVisualShapeIndex=visual_shape,
            #     basePosition=pos,
            #     baseOrientation=orn
            # )

            # Check for collisions with the robot
            for robot_q in robot_qs_to_avoid:
                set_robot_joint_positions(robot_id, list(range(p.getNumJoints(robot_id))), robot_q)
                p.stepSimulation()
                collisions = p.getClosestPoints(bodyA=body_id, bodyB=robot_id, distance=0.05)
                if len(collisions) > 0:
                    success = False

            # check if the box lies on the trajectory
            # import pdb; pdb.set_trace()
            # if not check_intersection_raycast(body_id, base_ee_traj):
            #     success = False

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

    N_SAMPLES = int(args.robot_update_rate * args.time_per_traj)

    if not args.no_save:
        os.makedirs("output", exist_ok=True)
        output_dir = os.path.join("output", args.experiment_name + ".zarr")
        if args.delete_existing and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        root_output = zarr.open(output_dir, mode="a")

        scenarios_group = root_output.require_group("trajectories")

        # Find existing traj indices like traj_0007, traj_0123, ...
        traj_re = re.compile(r"^traj_(\d+)$")
        existing_ids = []
        for name, node in scenarios_group.items():
            if isinstance(node, zarr.hierarchy.Group):
                m = traj_re.match(name)
                if m:
                    existing_ids.append(int(m.group(1)))

        # Start from next index
        num_prev_traj = (max(existing_ids) + 1) if existing_ids else 0
        print(f"Resuming at traj index {num_prev_traj}")

    # Environment setup
    ll, ul, obstacle_ids, robot_id, joint_indices = setup_env(args, robot_base_position)

    base_traj_i = 0
    base_traj_pbar = tqdm(total=args.num_base_trajectories, desc="Base Trajectories")
    while base_traj_i < args.num_base_trajectories:
        saved_base_traj = False
        q_start = get_random_joint_angles_without_collision(
            robot_id, joint_indices, obstacle_ids, ll, ul, verbose=args.verbose
        )
        if args.q_goal is None:
            q_goal = get_random_joint_angles_without_collision(
                robot_id, joint_indices, obstacle_ids, ll, ul, verbose=args.verbose
            )
        else:
            q_goal = args.q_goal

        base_traj = get_path(
            q_start, q_goal,
            robot_id, joint_indices,
            obstacle_ids,
            ll, ul,
            args.time_per_traj, args.robot_update_rate,
            rrt_vis_fps=args.rrt_vis_fps,
            use_gui=args.gui,
            verbose=args.verbose
        )

        if base_traj is not None:
            base_traj = np.array(base_traj)  # (N_SAMPLES, dof)
        else:
            if args.verbose:
                print("  Failed to generate base trajectory, retrying...")
            continue

        # This will accumulate all obstacle-specific modified trajectories for this base traj
        modified_trajs = []

        ee_link_index = get_link_index_by_name(robot_id, "ee_link")
        base_ee_traj = np.array([
            get_end_effector_pos(robot_id, ee_link_index, q, joint_indices)
            for q in base_traj
        ])

        obstacle_i = 1
        obstacle_pbar = tqdm(total=args.obstacles_per_base_trajectory + 1, desc="Obstacle Configurations")
        num_obstacle_fails = 0
        while obstacle_i < args.obstacles_per_base_trajectory + 1 and num_obstacle_fails < args.max_obstacle_fails_per_base_traj:
            num_paths_for_this_obstacle = 0
            # This also checks for collisions with the robot
            robot_qs_to_avoid = [q_start, q_goal]
            new_obj_ids, new_obj_infos = add_random_obstacles(args.min_obstacles, args.max_obstacles, robot_id, robot_qs_to_avoid, base_ee_traj)
            for path_per_obstacle_i in range(args.obstacles_per_base_trajectory):
            # for path_per_obstacle_i in tqdm(range(args.obstacles_per_base_trajectory), desc="Paths per Obstacle Configuration"):
                num_fails = 0
                path = None
                while path is None and num_fails < args.max_fails:
                    path = get_path(
                        q_start, q_goal,
                        robot_id, joint_indices,
                        obstacle_ids + new_obj_ids,
                        ll, ul,
                        args.time_per_traj, args.robot_update_rate,
                        rrt_vis_fps=args.rrt_vis_fps,
                        use_gui=args.gui,
                        verbose=args.verbose,
                    )
                    if path is None:
                        num_fails += 1
                        if args.verbose:
                            print(f"    Retry {num_fails}/{args.max_fails}")

                if path is not None:
                    modified_trajs.append(path)
                    num_paths_for_this_obstacle += 1
                else:
                    if args.verbose:
                        print(f"    Failed to find a path after {args.max_fails} attempts. Skipping this obstacle configuration.")
                if args.verbose and num_fails >= args.max_fails:
                    print("    Moving to next obstacle configuration due to repeated failures.")

            if num_paths_for_this_obstacle == 0:
                num_obstacle_fails += 1

            obstacle_info = {
                "obstacles": new_obj_infos,
            }
            if not args.no_save and len(modified_trajs) > 0 and len(modified_trajs) > 0:
                # Save one Zarr subgroup per base trajectory
                if not saved_base_traj:
                    traj_name = f"scenario_{base_traj_i + num_prev_traj:04d}"
                    if traj_name in scenarios_group:
                        del scenarios_group[traj_name]  # replace if re-running
                    scenario_grp = scenarios_group.create_group(traj_name)
                    no_obstacle_scenario_grp = scenario_grp.create_group("obstacle_config_00")
                    traj_grp = no_obstacle_scenario_grp.create_group("traj_00")
                    traj_grp.create_dataset("qs", data=base_traj, dtype="f4", chunks=(N_SAMPLES, args.dof))
                    no_obstacle_scenario_grp.attrs["metadata"] = json.dumps({"obstacles": []})
                    saved_base_traj = True
                    base_traj_i += 1

                obstacle_grp = scenario_grp.create_group(f"obstacle_config_{obstacle_i:02d}")
                obstacle_grp.attrs["metadata"] = json.dumps(obstacle_info)

                # Save modified trajectory as concatenated array
                for i in range(len(modified_trajs)):
                    traj_grp = obstacle_grp.create_group(f"traj_{i:02d}")
                    traj_grp.create_dataset(f"qs", data=modified_trajs[i], dtype="f4")
                # all_modified_trajs = np.stack(modified_trajs, axis=0)  # (num_modified, N_SAMPLES, dof)
                # obstacle_grp.create_dataset("qs", data=all_modified_trajs, dtype="f4", chunks=(1, N_SAMPLES, args.dof))
                if args.verbose:
                    print(f"  Saved {traj_name} with {len(modified_trajs)} modified trajectories")
                obstacle_i += 1
                obstacle_pbar.update(1)

            # Remove the newly added obstacles before the next iteration
            for oid in new_obj_ids:
                p.removeBody(oid)
        obstacle_pbar.close()
        base_traj_pbar.update(1)
    base_traj_pbar.close()

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf", required=False, help="Path to robot URDF", default=SISBOT_PATH
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
    parser.add_argument("--rrt_vis_fps", type=int, default=10)
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
        "--no-save",
        action="store_true",
        help="If set, do not save trajectories to disk",
    )
    parser.add_argument(
        "--min_obstacles", type=int, default=1, help="Minimum number of obstacles in the scene"
    )
    parser.add_argument(
        "--max_obstacles", type=int, default=3, help="Number of obstacles in the scene"
    )
    parser.add_argument(
        "--max_fails", type=int, default=2, help="Max number of planning failures before going to the next trajectory"
    )
    parser.add_argument(
        "--obstacles_per_base_trajectory", type=int, default=3, help="Number of different obstacle configurations to try per base trajectory"
    )
    parser.add_argument(
        "--max_obstacle_fails_per_base_traj", type=int, default=20, help="Number of different obstacle configurations to try per base trajectory before doing another trajectory (assuming all failed)"
    )
    parser.add_argument(
        "--paths_per_obstacle", type=int, default=2, help="Number of different paths to to get per obstacle configuration per base trajectory"
    )
    parser.add_argument(
        "--num_base_trajectories", type=int, default=10_000, help="Number of base trajectories to generate"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true"
    )
    parser.add_argument(
        "--q_goal",
        nargs='+',  # Expect one or more arguments
        type=float, # Convert each argument to a float
        default=None,
        help="Set to None to use a random goal. Otherwise, provide a space-separated list of floats."
    )
    # To the left side of the engine
    # (0.8704628188464882, -2.4071832524933336, 2.190265808315341, -2.6430289436412373, -1.255085236341607, -1.9464706594109809, 0.4383191524999254)
    args = parser.parse_args()
    main(args)
