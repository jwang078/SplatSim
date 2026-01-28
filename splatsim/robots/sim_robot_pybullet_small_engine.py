import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import zmq

from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
)

class SmallEnginePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG_NAME = None
    ENV_CONFIG = None

    # Success criteria: target end effector pose
    TARGET_EE_POS = (-0.003126271918487248, 0.4626016957140267, 0.31067939915838083)
    TARGET_EE_QUAT = (-0.5883302720488017, 0.318663764807395, 0.472865116611213, 0.5733406295406109)

    # Tolerance for success check
    POS_TOLERANCE_M = 0.03  # 3 centimeters
    QUAT_TOLERANCE_DEG = 10.0  # 10 degrees

    # TODO reformat this to have a "task"-like variable that sets the task
    # That then informs self.TRAJECTORY_GEN_CONFIG["q_goal"], etc
    # Also sets the success and reward criteria

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _register_trajectory_obstacles(self):
        """Register static obstacles with the trajectory generator. Override in subclasses."""
        pass

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths

    def serve_loop(self) -> None:
        # To be called in the parent's serve()

        # Check mode dropdown for mode changes
        new_mode = self._check_mode_buttons()
        if new_mode is not None:
            self._handle_mode_transition(new_mode)

        # Check GUI buttons for trajectory generation control (start/stop)
        start_pressed, stop_pressed = self._splatsim_gui.check_traj_buttons()

        if start_pressed and self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
            # Sync GUI values to config before starting
            self._splatsim_gui.save_to_config(self.trajectory_generator.config)
            # Register static obstacles with the trajectory generator
            self._register_trajectory_obstacles()
            # Switch to active trajectory generation
            self._handle_mode_transition(self.SERVE_MODES.GENERATE_TRAJECTORIES)
            print(f"[GUI] Started trajectory generation with config: {self.trajectory_generator.config}")

        if stop_pressed:
            if self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
                # Stop active generation, go back to idle
                self._handle_mode_transition(self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE)
            elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
                # From idle, go back to interactive
                self._handle_mode_transition(self.SERVE_MODES.INTERACTIVE)

        if self.serve_mode == self.SERVE_MODES.INTERACTIVE:
            self.pybullet_client.stepSimulation()
            time.sleep(1 / 240)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
            # Idle mode - just step simulation while user configures settings
            self.pybullet_client.stepSimulation()
            time.sleep(1 / 240)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_DEMOS:
            raise NotImplementedError()
            initial_joint_positions = self.randomize_ee_pose()

            success = self.plan_execute_record_trajectory(
                initial_joint_positions, self.splatsim_robot.articulation_config.joint_signs
            )
            if success:
                self.trajectory_count += 1

            if self.trajectory_count > self.MAX_TRAJECTORY_COUNT:
                print(
                    f"Exiting record_demos mode because max trajectory count of {self.MAX_TRAJECTORY_COUNT} was reached in folder {self.path}"
                )
                self._handle_mode_transition(self.SERVE_MODES.INTERACTIVE)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            # Handle active trajectory generation mode
            self.trajectory_generator.generate_trajectory_batch()

            if self.trajectory_generator.is_complete():
                print(f"[GUI] Completed trajectory generation. Switching to idle mode.")
                self._handle_mode_transition(self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE)
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

        is_success, info = self.check_success_metrics()

        info = {"is_success": is_success, **info}

        return self._get_gym_observation(), info

    # def compute_reward(self) -> float:
    #     """Compute sparse reward based on success."""

    #     return 1.0 if self.check_success() else 0.0

    def check_success_metrics(self) -> tuple[bool, dict]:
        """Check if the task goal is achieved.

        Returns True if the end effector is within POS_TOLERANCE_M (meters) and
        QUAT_TOLERANCE_DEG (degrees) of the target pose.
        """
        success = True

        pos, quat = self.get_current_ee_pose()

        # Check position distance
        pos_diff = np.linalg.norm(np.array(pos) - np.array(self.TARGET_EE_POS))
        if pos_diff > self.POS_TOLERANCE_M:
            success = False

        # Check quaternion distance (angle between orientations)
        # Quaternion dot product gives cos(theta/2) where theta is the rotation angle
        q1 = np.array(quat)
        q2 = np.array(self.TARGET_EE_QUAT)
        dot = np.abs(np.dot(q1, q2))  # abs handles q and -q representing same rotation
        dot = np.clip(dot, -1.0, 1.0)  # Numerical stability
        angle_rad = 2 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        if angle_deg <= self.QUAT_TOLERANCE_DEG:
            success = False

        info = {
            "position_error_m": pos_diff,
            "orientation_error_deg": angle_deg,
        }

        return success, info

    def check_terminated(self) -> bool:
        """Check if episode should terminate."""
        success, info = self.check_success_metrics()
        return success


class UprightRobotSmallEnginePybulletRobotServer(SmallEnginePybulletRobotServer):
    TABLE_LIMITS = ((0.2, 0.6), (-0.5, 0.5))

    ENV_CONFIG = {
        "name": "upright_robot_small_engine",
        "objects": [
            {
                "object_name": "small_engine",
                "splat_object_name": "small_engine",
                "grasp_config": [],
                "randomize_pose": False,
                "table_pos": [0.3, 0.55],
                "table_quat": [0, 0, 1, 0],
                "rotation_range_z": [0, 0],
            },
            {
                "object_name": "plastic_apple",
                "splat_object_name": "plastic_apple",
                "grasp_config": [],
                "randomize_pose": True,
                "rotation_range_z": [0, 0],
            },
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)

        # Register obstacles if trajectory generator was initialized via CLI
        if self.trajectory_generator is not None:
            self._register_trajectory_obstacles()

    def _register_trajectory_obstacles(self):
        """Register wall and plane as obstacles for trajectory generation."""
        if self.trajectory_generator is not None:
            self.trajectory_generator.register_obstacle(self.wall)
            self.trajectory_generator.register_obstacle(self.plane)

class UprightRobotSmallEngineNewPybulletRobotServer(SmallEnginePybulletRobotServer):
    # This new lab bench scene has the robot rotated 90 degrees because it was installed rotated D:
    background_splat_name = "robot_iphone_w_engine_new"

    TABLE_LIMITS = ((-0.5, 0.5), (0.2, 0.6))

    ENV_CONFIG = {
        "name": "upright_robot_small_engine_new",
        "objects": [
            {
                "object_name": "small_engine_new",
                "splat_object_name": "small_engine_new",
                "grasp_config": [],
                "randomize_pose": False,
                "rotation_range_z": [0, 0],
                "is_in_scene_splat": True,
                "table_pos": [-0.48, 0.36],
                # "table_pos": [-0.565, 0.35],
                "table_quat": [0, 0, -0.7071068, 0.7071068],
            },
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, 0])

        # place a wall in -0.6 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([-np.pi/2, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [0.0, -0.2, 0.0], quat)
        # self.wall = self.pybullet_client.loadURDF("plane.urdf", [0.0, -0.16, 0.0], quat)

        # Set initial camera position on the opposite side of the wall (positive y side)
        # Camera looks at the origin from the positive y side, above the floor
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=2.0,      # Distance from target
            cameraYaw=180,             # 0 degrees = looking from +y towards origin
            cameraPitch=-30,         # -30 degrees = looking down at ~30 degree angle
            cameraTargetPosition=[0, 0, 0.3]  # Look at point above the floor
        )

        # Register obstacles if trajectory generator was initialized via CLI
        if self.trajectory_generator is not None:
            self._register_trajectory_obstacles()

    def _register_trajectory_obstacles(self):
        """Register wall and plane as obstacles for trajectory generation."""
        if self.trajectory_generator is not None:
            self.trajectory_generator.register_obstacle(self.wall)
            self.trajectory_generator.register_obstacle(self.plane)