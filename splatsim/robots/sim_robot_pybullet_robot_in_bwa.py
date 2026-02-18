import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import mujoco
import mujoco.viewer
import zmq

assert mujoco.viewer is mujoco.viewer
import pybullet as p

from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
)

from splatsim.utils.robot_splat_render_utils import transform_object, get_curr_link_states


class BWAPybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG = None
    background_splat_name = "bwa_open_space"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths

    def serve_loop(self) -> None:
        # To be called in the parent's serve()
        # GENERATE_TRAJECTORIES is handled by the base class serve() method.

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
                self.serve_mode = self.SERVE_MODES.INTERACTIVE
        elif self.serve_mode in (self.SERVE_MODES.GENERATE_TRAJECTORIES, self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE):
            pass  # Handled by base class serve()
        else:
            raise ValueError(f"Unknown serve mode {self.serve_mode}. ")

    # =========================================================================
    # Gym Environment Interface
    # =========================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict

        Returns:
            observation: Initial observation dict
            info: Dict with initial info
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._step_count = 0
        self._episode_started = True

        # From GENERATE_DEMOS: randomize_ee_pose()
        initial_joints = self.randomize_ee_pose()
        self.teleport_joint_state(self.splatsim_robot, initial_joints)
        self.open_gripper()

        # Let simulation settle
        for _ in range(100):
            self.pybullet_client.stepSimulation()

        return self._get_gym_observation(), {"is_success": False}

    def compute_reward(self) -> float:
        """Compute sparse reward based on success."""
        return 1.0 if self.check_success() else 0.0

    def check_success(self) -> bool:
        """Check if the task goal is achieved.

        TODO: Implement task-specific success criteria for BWA environment.
        """
        return False

    def check_terminated(self) -> bool:
        """Check if episode should terminate."""
        return self.check_success()


class OpenSpaceBWAPybulletRobotServer(BWAPybulletRobotServer):
    ENV_CONFIG = {
        "name": "open_space_bwa",
        "objects": [
            # {
            #     "name": "small_engine",
            #     "splat_name": "small_engine",
            #     "grasp_config": [],
            #     "randomize_pose": False,
            #     "table_pos": [0.3, 0.55],
            #     "table_quat": [0, 0, 1, 0],
            #     "rotation_range_z": [0, 0],
            # },
            # {
            #     "name": "plastic_apple",
            #     "splat_name": "plastic_apple",
            #     "grasp_config": [],
            #     "randomize_pose": True,
            #     "rotation_range_z": [0, 0],
            # },
            # {
            #     "name": "redblock",
            #     "splat_name": "redblock",
            #     "grasp_config": [],
            #     "randomize_pose": True,
            #     "rotation_range_z": [0, 0],
            # },
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-1, 0, 0.0], quat)


        # Put the robot upside down and facing the back
        # pybullet quaternion convention is (x, y, z, w)
        self.pybullet_client.resetBasePositionAndOrientation(
            self.splatsim_robot.sim_id, [0, 0, 1.0], [0, 1, 0, 0]
        )
        (
            object_pos,
            object_quat,
        ) = self.pybullet_client.getBasePositionAndOrientation(
            self.splatsim_robot.sim_id
        )
        _ = transform_object(
            splatsim_obj=self.splatsim_robot,
            pos=torch.tensor(object_pos, device='cuda').float(),
            quat=torch.tensor(object_quat, device='cuda').float().roll(1),
            use_base_position=True,
            inplace=True
        )

