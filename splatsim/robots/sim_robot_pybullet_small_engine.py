import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import zmq

from splatsim.configs import (
    EnvConfig,
    TaskConfig,
    CuboidObjectConfig,
    SplatObjectConfig,
)
from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
)

from splatsim.utils.rrt_path_utils import compute_camera_alignment_score

class SmallEnginePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG: EnvConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths

    def serve_loop(self) -> None:
        # To be called in the parent's serve()

        if self.serve_mode == self.SERVE_MODES.INTERACTIVE:
            print("in collision?", self.is_robot_in_collision())
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
                self.serve_mode = self.SERVE_MODES.INTERACTIVE
        elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            # Handle active trajectory generation mode
            self.trajectory_generator.generate_trajectory_batch()

            if self.trajectory_generator.is_complete():
                print(f"[GUI] Completed trajectory generation. Switching to idle mode.")
                self.serve_mode = self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE
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

        # Smoothness tracking
        self._prev_action = None
        self._action_delta = 0.0
        self._prev_action_delta = 0.0
        self._action_accel = 0.0
        self._prev_action_accel = 0.0
        self._action_jerk = 0.0

        # From GENERATE_DEMOS: randomize_ee_pose()
        initial_joints = self.randomize_ee_pose()
        self.teleport_joint_state(self.splatsim_robot, initial_joints)
        self.open_gripper()

        # Let simulation settle
        for _ in range(100):
            self.pybullet_client.stepSimulation()

        metrics = self.check_metrics()

        info = {"is_success": metrics['is_success'], **metrics}

        return self._get_gym_observation(), info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one control step, tracking action smoothness."""
        # Track action delta (velocity) for smoothness metric
        if self._prev_action is not None:
            self._action_delta = np.linalg.norm(np.array(action) - self._prev_action)
        else:
            self._action_delta = 0.0

        # Track action acceleration (delta in action_delta)
        self._action_accel = np.abs(self._action_delta - self._prev_action_delta)

        # Track action jerk (delta in action_accel)
        self._action_jerk = np.abs(self._action_accel - self._prev_action_accel)

        # Update previous values for next step
        self._prev_action = np.array(action)
        self._prev_action_delta = self._action_delta
        self._prev_action_accel = self._action_accel

        return super().step(action)

    def compute_reward(self) -> float:
        """Compute sparse reward based on success."""

        return 1.0 if self.check_success() else 0.0
    
    def check_success(self) -> bool:
        metrics = self.check_metrics()
        return metrics['is_success']

    def check_metrics(self) -> dict:
        """Check if the task goal is achieved.

        Returns True if the end effector is within pos_tolerance_m (meters) and
        quat_tolerance_deg (degrees) of the target pose.
        """
        assert self.ENV_CONFIG.task is not None, "SmallEngine env requires a task config"
        task_config = self.ENV_CONFIG.task
        target_ee_pos = task_config.target_ee_pos
        target_ee_quat = task_config.target_ee_quat
        pos_tolerance_m = task_config.pos_tolerance_m
        quat_tolerance_deg = task_config.quat_tolerance_deg

        success = True

        pos, quat = self.get_current_ee_pose()

        # Check position distance
        pos_diff = np.linalg.norm(np.array(pos) - np.array(target_ee_pos))
        if pos_diff > pos_tolerance_m:
            success = False

        # Check quaternion distance (angle between orientations)
        # Quaternion dot product gives cos(theta/2) where theta is the rotation angle
        q1 = np.array(quat)
        q2 = np.array(target_ee_quat)
        dot = np.abs(np.dot(q1, q2))  # abs handles q and -q representing same rotation
        dot = np.clip(dot, -1.0, 1.0)  # Numerical stability
        angle_rad = 2 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        if angle_deg <= quat_tolerance_deg:
            success = False

        cam_position, cam_rotation = self.get_wrist_camera_transform()
        # Camera forward direction (assumes +Z axis in local frame)
        cam_forward = cam_rotation[:, 2]
        cam_looks_at_goal_score = compute_camera_alignment_score(cam_position, cam_forward, target_ee_pos)

        in_collision = self.is_robot_in_collision()
        if in_collision:
            success = False

        metrics = {
            "is_success": success,
            "position_error_m": pos_diff,
            "orientation_error_deg": angle_deg,
            "cam_looks_at_goal_score": cam_looks_at_goal_score,
            "action_delta": self._action_delta,
            "action_accel": self._action_accel,
            "action_jerk": self._action_jerk,
            "in_collision": in_collision,
        }

        return metrics

    def check_terminated(self) -> bool:
        """Check if episode should terminate."""
        metrics = self.check_metrics()
        return metrics['is_success']

class UprightRobotSmallEngineNewPybulletRobotServer(SmallEnginePybulletRobotServer):
    # This new lab bench scene has the robot rotated 90 degrees because it was installed rotated D:
    background_splat_name = "robot_iphone_w_engine_new"

    ENV_CONFIG = EnvConfig(
        name="upright_robot_small_engine_new",
        task=TaskConfig(
            task_description="<control_mode> joint <control_mode>",
            target_ee_pos=(-0.003126271918487248, 0.4626016957140267, 0.31067939915838083),
            target_ee_quat=(-0.5883302720488017, 0.318663764807395, 0.472865116611213, 0.5733406295406109),
            pos_tolerance_m=0.03,  # 3 centimeters
            quat_tolerance_deg=10.0,  # 10 degrees
        ),
        objects=[
            SplatObjectConfig(
                object_name="small_engine_new",
                splat_object_name="small_engine_new",
                grasp_config=[],
                randomize_pose=False,
                rotation_range_z=(0, 0),
                is_in_scene_splat=True,
                table_pos=(-0.48, 0.36),
                table_quat=(0, 0, -0.7071068, 0.7071068),
            ),
            # table has a plane for objects to sit on at z = 0
            CuboidObjectConfig(
                object_name="table",
                size=(1.5, 0.90, 0.05),
                position=(0, 0.25, -0.025),
                mass=0,
                color_rgb=(223, 205, 192),
            ),
            # wall is at -0.2 on y axis
            CuboidObjectConfig(
                object_name="wall",
                size=(3.0, 0.05, 1.5),
                position=(0, -0.225, 0.75),
                mass=0,
                color_rgb=(223, 205, 192),
            ),

            # SplatObjectConfig(
            #     object_name="thinkpad_box",
            #     splat_object_name="thinkpad_box",
            #     grasp_config=[],
            #     randomize_pose=False,
            #     rotation_range_z=(0, 0),
            #     is_in_scene_splat=False,
            #     table_pos=(0.48, 0.20),
            #     table_quat=(0, 0, 0, 1),
            # ),
            # SplatObjectConfig(
            #     object_name="starwars_box",
            #     splat_object_name="starwars_box",
            #     grasp_config=[],
            #     randomize_pose=False,
            #     rotation_range_z=(0, 0),
            #     is_in_scene_splat=False,
            #     table_pos=(0.48, 0.40),
            #     table_quat=(0, 0, 0, 1),
            # ),
        ],
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set initial camera position on the opposite side of the wall (positive y side)
        # Camera looks at the origin from the positive y side, above the floor
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=2.0,      # Distance from target
            cameraYaw=180,             # 0 degrees = looking from +y towards origin
            cameraPitch=-30,         # -30 degrees = looking down at ~30 degree angle
            cameraTargetPosition=[0, 0, 0.3]  # Look at point above the floor
        )
