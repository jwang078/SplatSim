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


class BWAPybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG_NAME = None
    ENV_CONFIG = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths

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


class OpenSpaceBWAPybulletRobotServer(BWAPybulletRobotServer):
    ENV_CONFIG_NAME = "open_space_bwa"

    ENV_CONFIG = {
        "objects": [
            # {
            #     "object_name": "small_engine",
            #     "splat_object_name": "small_engine",
            #     "grasp_config": [],
            #     "randomize_pose": False,
            #     "table_pos": [0.3, 0.55],
            #     "table_quat": [0, 0, 1, 0],
            #     "rotation_range_z": [0, 0],
            # },
            # {
            #     "object_name": "plastic_apple",
            #     "splat_object_name": "plastic_apple",
            #     "grasp_config": [],
            #     "randomize_pose": True,
            #     "rotation_range_z": [0, 0],
            # },
            # {
            #     "object_name": "redblock",
            #     "splat_object_name": "redblock",
            #     "grasp_config": [],
            #     "randomize_pose": True,
            #     "rotation_range_z": [0, 0],
            # },
        ]
    }
