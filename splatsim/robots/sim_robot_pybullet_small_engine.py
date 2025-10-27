import time

import torch
import numpy as np
import mujoco
import mujoco.viewer
import zmq

assert mujoco.viewer is mujoco.viewer
import pybullet as p
from pybullet_planning import plan_joint_motion, get_movable_joints

from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
    TrajectoryPathSegment,
    GripperPathSegment,
    GripperState,
)
from splatsim.utils.transform_utils import rotation_matrix_to_euler_angles
from scipy.optimize import minimize

from tqdm import tqdm


class SmallEnginePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG_NAME = None
    ENV_CONFIG = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths
    

    
    def optimize_trajectory(self, start_pos, start_quat, end_pos, end_quat, initial_joint_positions):
        print("Optimizing trajectory...")
        MAX_ITER = 200
        FTOL = 1e-6 # 1e-6
        pbar = tqdm(total=MAX_ITER, desc="Optimization Progress", unit="iters")
        def optimization_callback(x_k):
            # Since we don't get the iteration count, we just advance the bar by 1
            pbar.update(1)

        def get_ee_pose(q):
            """
            Computes the end-effector (EE) position (x, y, z) and orientation (quaternion)
            for a given joint configuration q using PyBullet's forward kinematics.
            """
            # 1. Set the robot's joint state
            for idx, j in enumerate(joint_ids[:ee_joint_number+1]):  # Only set up to the EE joint
                p.resetJointState(robot, j, q[idx])

            # 2. Get the link state (FK)
            link_state = p.getLinkState(robot, ee_link_index)
            
            # link_state[0] is the position, link_state[1] is the orientation (quaternion)
            pos = np.array(link_state[0])
            quat = np.array(link_state[1])
            
            return pos, quat

        def traj_tracking_objective(Q_flat):
            """
            The new total objective function.
            Q_flat is the flattened (T*nq) trajectory array.
            """
            Q = Q_flat.reshape(T, -1)

            Q_ee_pos_start, Q_ee_quat_start = get_ee_pose(Q[:1])
            Q_ee_pos_end, Q_ee_quat_end = get_ee_pose(Q[-1:])

            

            # if abs(w_track_pos) > 0 or abs(w_track_orient) > 0:
            #     # tracking cost for end effector pose
            #     # 1. Forward Kinematics: Convert joint angles to EE poses
            #     Q_pos_traj, Q_quat_traj = get_ee_poses_traj(Q)
                
            #     # 2. Tracking Cost
            #     # Position (XYZ) tracking cost (Squared Euclidean distance)
            #     pos_track = np.sum((Q_pos_traj - ee_pos_ref)**2)
                
            #     # Orientation (Quat) tracking cost (Sum of squared quaternion error)
            #     orient_track = np.sum([quat_error_sq(Q_quat_traj[i], ee_quat_ref[i]) for i in range(T)])

            #     ee_track_cost = w_track_pos * pos_track + w_track_orient * orient_track
            # else:
            #     ee_track_cost = 0.0
            ee_track_cost = 0.0

            joint_track = np.sum((Q - q_ref)**2)

            # smoothness on velocities
            dQ = np.diff(Q, axis=0) / dt
            smooth = np.sum(dQ**2)

            # collision loss: sum over waypoints
            dists = min_clearance_traj(Q)
            # coll = np.sum(exponential(dists)**2)
            coll = np.sum(hinge(dists)**2)

            cost = w_track*joint_track + w_smooth*smooth + w_collision*coll  + ee_track_cost

            # 3. Total Cost
            return cost

        # method = L-BFGS-B # took 2 mins for 15 iters
        # method = SLSQP # took 12 mins and 200 iters
        res = minimize(objective,
                    x0=q0.ravel(),
                    method=args.planner_method,
                    bounds=bounds,
                    options=dict(maxiter=MAX_ITER, ftol=FTOL),
                    callback=optimization_callback)

    def serve_loop(self) -> None:
        # To be called in the parent's serve()
        if self.serve_mode == self.SERVE_MODES.INTERACTIVE:
            self.pybullet_client.stepSimulation()
            time.sleep(1 / 240)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_DEMOS:
            initial_joint_positions = self.randomize_ee_pose()

            success = self.plan_execute_record_trajectory(
                initial_joint_positions, self.joint_signs
            )
            if success:
                self.trajectory_count += 1

            if self.trajectory_count > self.MAX_TRAJECTORY_COUNT:
                print(
                    f"Exiting record_demos mode because max trajectory count of {self.MAX_TRAJECTORY_COUNT} was reached in folder {self.path}"
                )
                self.set_serve_mode(self.SERVE_MODES.INTERACTIVE)
        else:
            raise ValueError(f"Unknown serve mode {self.serve_mode}. ")


class UprightRobotSmallEnginePybulletRobotServer(SmallEnginePybulletRobotServer):
    ENV_CONFIG_NAME = "upright_robot_small_engine"

    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "small_engine",
                "splat_object_name": "small_engine",
                "grasp_config": [],
                "randomize_pose": False,
                "table_pos": [0.3, 0.55],
                "table_quat": [0, 0, 1, 0],
            },
            # {
            #     "object_name": "small_engine",
            #     "splat_object_name": "small_engine",
            #     "grasp_config": [],
            #     "randomize_pose": True,
            # },
            {
                "object_name": "plastic_apple",
                "splat_object_name": "plastic_apple",
                "grasp_config": [],
                "randomize_pose": True,
            },
        ]
    }
