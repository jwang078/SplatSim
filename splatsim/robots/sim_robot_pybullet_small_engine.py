import time

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
                initial_joint_positions, self.splatsim_robot.articulation_config.joint_signs
            )
            if success:
                self.trajectory_count += 1

            if self.trajectory_count > self.MAX_TRAJECTORY_COUNT:
                print(
                    f"Exiting record_demos mode because max trajectory count of {self.MAX_TRAJECTORY_COUNT} was reached in folder {self.path}"
                )
                self.set_serve_mode(self.SERVE_MODES.INTERACTIVE)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            # Handle trajectory generation mode
            self.trajectory_generator.generate_trajectory_batch()

            if self.trajectory_generator.is_complete():
                print(f"Completed trajectory generation. Exiting.")
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

    # Environment-specific trajectory generation defaults
    TRAJECTORY_GEN_CONFIG = {
        "num_base_trajectories": 10_000,
        "obstacles_per_base_trajectory": 3,
        "paths_per_obstacle": 2,
        "min_obstacles": 1,
        "max_obstacles": 3,
        "max_fails": 2,
        "max_obstacle_fails_per_base_traj": 20,
        "time_per_traj": 6.0,
        "robot_update_rate": 20,
        "rrt_vis_fps": 10,
        "use_obstacles": False,  # No extra obstacles for small engine by default
        "q_start": None,
        "q_goal": None,
        "cuboids_fn": None,
        "render_images": False,
        "save_base_trajectory": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)

        # TODO temporary until wall and plane are splatsim objects
        if self.trajectory_generator is not None:
            self.trajectory_generator.register_obstacle(self.wall)
            self.trajectory_generator.register_obstacle(self.plane)

class UprightRobotSmallEngineNewPybulletRobotServer(SmallEnginePybulletRobotServer):
    # This new lab bench scene has the robot rotated 90 degrees because it was installed rotated D:
    ENV_CONFIG_NAME = "upright_robot_small_engine_new"
    background_splat_name = "robot_iphone_w_engine_new"

    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "small_engine_new",
                "splat_object_name": "small_engine_new",
                "grasp_config": [],
                "randomize_pose": False,
                "rotation_range_z": [0, 0],
                "is_in_scene_splat": True,
                "table_pos": [-0.565, 0.35],
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

        # TODO temporary until wall and plane are splatsim objects
        if self.trajectory_generator is not None:
            self.trajectory_generator.register_obstacle(self.wall)
            self.trajectory_generator.register_obstacle(self.plane)