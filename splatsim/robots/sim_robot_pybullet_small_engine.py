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
    TrajectoryGenModeConfig,
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
        pass

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

        # This now also randomizes the robot's joints
        self.randomize_objects()

        # From GENERATE_DEMOS: randomize_ee_pose()
        # initial_joints = self.randomize_ee_pose()
        # self.teleport_joint_state(self.splatsim_robot, initial_joints)
        # self.open_gripper()

        # # Let simulation settle
        # for _ in range(100):
        #     self.pybullet_client.stepSimulation()

        metrics = self.check_metrics()

        info = {"is_success": metrics['is_success'], **metrics}

        return self.get_observations(), info

    def _physics_step(self, action: np.ndarray) -> None:
        """Track action smoothness metrics then advance physics."""
        if self._prev_action is not None:
            self._action_delta = np.linalg.norm(np.array(action) - self._prev_action)
        else:
            self._action_delta = 0.0

        self._action_accel = np.abs(self._action_delta - self._prev_action_delta)
        self._action_jerk = np.abs(self._action_accel - self._prev_action_accel)

        self._prev_action = np.array(action)
        self._prev_action_delta = self._action_delta
        self._prev_action_accel = self._action_accel

        super()._physics_step(action)

    def compute_reward_from_metrics(self, metrics: dict) -> float:
        return 1.0 if metrics['is_success'] else 0.0

    def check_terminated_from_metrics(self, metrics: dict) -> bool:
        if self.ENV_CONFIG.terminate_on_collision:
            return metrics['is_success'] or metrics['in_collision']
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
        print(f"Current EE position: {pos}")
        print(f"Position difference: {pos_diff:.4f} m (tolerance: {pos_tolerance_m:.4f} m)")
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

        print(f"Current EE orientation (quat): {quat}")
        print(f"Orientation difference: {angle_deg:.2f} deg (tolerance: {quat_tolerance_deg:.2f} deg)")

        if angle_deg > quat_tolerance_deg:
            success = False

        cam_position, cam_rotation = self.get_wrist_camera_transform()
        # Camera forward direction (assumes +Z axis in local frame)
        cam_forward = cam_rotation[:, 2]
        cam_looks_at_goal_score = compute_camera_alignment_score(cam_position, cam_forward, target_ee_pos)

        in_collision = self.is_robot_in_collision(obstacle_clearance=0.0)
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


class UprightRobotSmallEngineNewPybulletRobotServer(SmallEnginePybulletRobotServer):
    # This new lab bench scene has the robot rotated 90 degrees because it was installed rotated D:
    background_splat_name = "robot_iphone_w_engine_new"

    ENV_CONFIG = EnvConfig(
        name="upright_robot_small_engine_new",
        task=TaskConfig(
            task_description="<control_mode> joint <control_mode>",

            # Approach lever
            # q_goal=[1.33936567, -1.52838483, 1.92282924, -1.21754169, -0.53407075, -0.73042029]
            # Ran self.get_current_ee_pose()
            target_ee_pos=(-0.10123532289544344, 0.5484031509107826, 0.26692192875731213),
            target_ee_quat=(0.8074376258351692, 0.1106042613918073, -0.5450490313370774, 0.19680632913133583),
            
            pos_tolerance_m=0.03,  # 3 centimeters
            quat_tolerance_deg=10.0,  # 10 degrees
        ),
        objects=[
            SplatObjectConfig(
                name="small_engine_new",
                splat_name="small_engine_new",
                grasp_configs=[],
                randomize_pose=False,
                rotation_range_z=(0, 0),
                load_splat=False, # Because it's already in the scene splat
                position_range_x=(-0.48, -0.48),
                position_range_y=(0.36, 0.36),
                base_quat=(0, 0, -0.7071068, 0.7071068),
            ),
            # table has a plane for objects to sit on at z = 0
            CuboidObjectConfig(
                name="table",
                size=(1.5, 1.0, 0.05),
                # size=(1.5, 0.90, 0.05),
                randomize_pose=False,
                position_range_x=(0, 0),
                position_range_y=(0.3, 0.3),
                # position_range_y=(0.25, 0.25),
                position_range_z=(-0.025, -0.025),
                mass=0,
                color_rgb=(223, 205, 192),
                load_splat=False,
                skip_collision_robot_links=[0],  # Robot is mounted on the table; shoulder_link (link 0) is always within 1cm of the table surface
            ),
            # # wall is at -0.2 on y axis
            CuboidObjectConfig(
                name="wall",
                size=(3.0, 0.05, 1.5),
                randomize_pose=False,
                position_range_x=(0, 0),
                position_range_y=(-0.225, -0.225),
                position_range_z=(0.75, 0.75),
                # position=(0, -0.225, 0.75),
                mass=0,
                color_rgb=(223, 205, 192),
                load_splat=False,
            ),
            SplatObjectConfig(
                name="box1",
                splat_name="thinkpad_box",
                grasp_configs=[],
                randomize_pose=True,
                rotation_range_z=(0, 0),

                # Parallel boxes
                # position_range_x=(-0.2, 0.5),
                # position_range_y=(0.15, 0.3),
                # base_quat=(0, 0, 0, 1),

                # boxes at 90 degree angle
                position_range_x=(0.15, 0.5),
                position_range_y=(0.3, 0.5),
                base_quat=(0, 0, 0.707, 0.707),

                scaling_range_x=(0.9, 1.1),
                scaling_range_y=(0.9, 1.1),
                scaling_range_z=(0.9, 1.1),

                use_aabb_collision=True, # Box is axis-aligned, so AABB is exact and faster than PyBullet collision checks
            ),
            SplatObjectConfig(
                name="box2",
                splat_name="starwars_box",
                grasp_configs=[],
                randomize_pose=True,
                rotation_range_z=(0, 0),

                # Parallel boxes
                # position_range_x=(-0.2, 0.5),
                # position_range_y=(0.5, 0.7),
                # base_quat=(0, 0, 1, 0), #rotated 180 degrees about z

                # Boxes at 90 degree angle
                position_range_x=(-0.4, 0.3),
                position_range_y=(0.6, 0.8),
                base_quat=(0, 0, 1, 0), #rotated 180 degrees about z

                scaling_range_x=(0.9, 1.1),
                scaling_range_y=(0.9, 1.1),
                scaling_range_z=(0.9, 1.1),

                use_aabb_collision=True, # Box is axis-aligned, so AABB is exact and faster than PyBullet collision checks
            ),
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

    def _get_default_trajectory_gen_config(self) -> TrajectoryGenModeConfig:
        assert self.ENV_CONFIG.task is not None, "SmallEngine env requires a task config"
        assert self.ENV_CONFIG.task.target_ee_pos is not None, "SmallEngine task config requires target_ee_pos"
        assert self.ENV_CONFIG.task.target_ee_quat is not None, "SmallEngine task config requires target_ee_quat"
        return TrajectoryGenModeConfig(
            ee_pos_goal=list(self.ENV_CONFIG.task.target_ee_pos),
            ee_quat_goal=list(self.ENV_CONFIG.task.target_ee_quat),
            debug_visualize=False
        )
