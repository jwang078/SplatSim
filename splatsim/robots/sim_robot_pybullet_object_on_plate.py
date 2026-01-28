import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import mujoco
import mujoco.viewer
import zmq
import random

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


class ObjectOnPlatePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG_NAME = None
    ENV_CONFIG = None
    background_splat_name = "robot_iphone"
    TABLE_LIMITS = ((0.2, 0.6), (-0.5, 0.5))


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []

        object = 0  # Object of interest; to be grasped

        # get object position
        (object_pos, object_quat,) = self.pybullet_client.getBasePositionAndOrientation(
            self.urdf_object_list[object]
        )

        # create transformation matrix from the object position and orientation
        object_transformation = np.eye(4)
        object_transformation[:3, :3] = np.array(
            self.pybullet_client.getMatrixFromQuaternion(object_quat)
        ).reshape(3, 3)
        object_transformation[:3, 3] = np.array(object_pos)

        # get the end effector position and orientation according to self.apple_grasp_pose
        ee_transformation = object_transformation @ self.grasp_poses[object]

        # get pregrasp to grasp path
        pre_grasp2grasp_path, pregrasp_transformation = self.pre_grasp_to_grasp(
            ee_transformation
        )
        if pre_grasp2grasp_path is None:
            return []  # Failure

        # get the joint positions using the inverse kinematics
        ee_pos = pregrasp_transformation[:3, 3]
        # convert transformation matrix to euler angles
        ee_euler = rotation_matrix_to_euler_angles(ee_transformation[:3, :3])
        ee_quat = self.pybullet_client.getQuaternionFromEuler(ee_euler)

        # get the joint positions using the inverse kinematics
        joint_positions = self.pybullet_client.calculateInverseKinematics(
            self.splatsim_robot.sim_id,
            6,
            ee_pos,
            ee_quat,
            maxNumIterations=100000,
            residualThreshold=1e-10,
        )
        joint_positions = list(joint_positions)

        # compute the path from the current joint positions to the target joint positions
        ik_joints = get_movable_joints(self.splatsim_robot.sim_id)
        ik_joint_positions = []
        path = plan_joint_motion(
            self.splatsim_robot.sim_id,
            ik_joints,
            joint_positions,
            obstacles=[
                self.plane,
                self.urdf_object_list[0],
                self.urdf_object_list[1],
            ],
            self_collisions=False,
        )
        if path is None:
            return []  # Failure

        # set the joints to the last joint positions of path
        # reset the joint positions to the initial joint positions
        # Note: Doesn't reset gripper open/close state
        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(self.splatsim_robot.sim_id, i, path[0][i - 1])

        all_paths.append(
            TrajectoryPathSegment(
                path=path, gripper_pos=0, gripper_velocity=0.2, threshold=0.001
            )
        )
        all_paths.append(
            TrajectoryPathSegment(
                path=pre_grasp2grasp_path,
                gripper_pos=0,
                gripper_velocity=0.2,
                threshold=0.001,
            )
        )
        all_paths.append(GripperPathSegment(target_state=GripperState.CLOSE))
        all_paths.append(
            TrajectoryPathSegment(
                path=pre_grasp2grasp_path[::-1],
                gripper_pos=1,  # this is closed, unlike the GripperState enum that says 0 is closed
            )
        )

        # set the joint angle to pre_grasp2grasp_path[0]
        # Note: doesn't affect gripper open/close state
        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id, i, pre_grasp2grasp_path[0][i - 1]
            )

        # now plan the path from pre_grasp to intermediate position
        ee_pos = self.pybullet_client.getLinkState(self.splatsim_robot.sim_id, 6)[0]
        intermediate_ee_pos = [ee_pos[0], ee_pos[1], 0.4]
        intermediate_ee_quat = self.initial_ee_quat
        intermediate_joint_positions = self.pybullet_client.calculateInverseKinematics(
            self.splatsim_robot.sim_id,
            6,
            intermediate_ee_pos,
            intermediate_ee_quat,
            maxNumIterations=100000,
            residualThreshold=1e-10,
        )

        # compute the path from the current joint positions to the target joint positions
        path = plan_joint_motion(
            self.splatsim_robot.sim_id,
            ik_joints,
            intermediate_joint_positions,
            obstacles=[self.plane, self.urdf_object_list[-1]],
            self_collisions=False,
        )
        if path is None:
            return []  # Failure
        # set the joints to the last joint positions of path
        # reset the joint positions to the initial joint positions
        # Note: doesn't affect gripper open/close state
        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(self.splatsim_robot.sim_id, i, path[-1][i - 1])

        all_paths.append(
            TrajectoryPathSegment(
                path=path,
                gripper_pos=1,
            )
        )

        # now plan the path from intermediate to drop location
        path = plan_joint_motion(
            self.splatsim_robot.sim_id,
            ik_joints,
            self.drop_ee_joint,
            obstacles=[self.plane, self.urdf_object_list[-1]],
            self_collisions=False,
        )
        if path is None:
            return []  # Failure
        # set the joints to the last joint positions of path
        # reset the joint positions to the initial joint positions
        # Note: doesn't affect gripper open/close state
        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(self.splatsim_robot.sim_id, i, path[-1][i - 1])

        all_paths.append(
            TrajectoryPathSegment(
                path=path,
                gripper_pos=1,
            )
        )

        all_paths.append(
            GripperPathSegment(
                target_state=GripperState.OPEN,
            )
        )

        # TODO why is this here. Can it be moved outside?
        if self.skip_recording_first:
            for i in range(1, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id, i, initial_joint_positions[i - 1]
                )

        # create a path from the drop location to the initial joint positions
        path = [
            np.array(self.drop_ee_joint[:6]) * 0.1 * (10 - i)
            + np.array(self.initial_joint_state[:6]) * 0.1 * (i)
            for i in range(1, 11)
        ]
        all_paths.append(
            TrajectoryPathSegment(
                path=path,
                gripper_pos=0,
            )
        )

        path = [self.initial_joint_state for _ in range(5)]
        all_paths.append(
            TrajectoryPathSegment(
                path=path,
                gripper_pos=0,
            )
        )

        if len(all_paths) != 9:
            print("WARNING: Incorrect number of segments in the path")
            # Failure
            return []

        return all_paths
    
    def randomize_plate_and_drop_pose(self):
        # randomize plate and drop location
        # [0.3, -0.5, 0.07]
        while True:
            x = random.uniform(0.2, 0.8)
            y = random.uniform(-0.4, 0.4)
            z = 0.0

            # get obj[0] position
            (
                object_pos,
                object_quat,
            ) = self.pybullet_client.getBasePositionAndOrientation(
                self.urdf_object_list[0]
            )

            # check the distance between the object and the drop location
            if np.linalg.norm(np.array(object_pos)[:2] - np.array([x, y])) > 0.2:
                break

        euler_z = 0
        quat = self.pybullet_client.getQuaternionFromEuler([0, 0, euler_z])

        self.pybullet_client.resetBasePositionAndOrientation(
            self.urdf_object_list[-1], [x, y, z], quat
        )

        self.drop_ee_pos = [x, y, 0.3]

        # calculate the drop ee joint
        self.drop_ee_joint = self.pybullet_client.calculateInverseKinematics(
            self.splatsim_robot.sim_id,
            6,
            self.drop_ee_pos,
            self.drop_ee_quat,
            maxNumIterations=100000,
        )



    def serve_loop(self) -> None:
        # To be called in the parent's serve()
        if self.serve_mode == self.SERVE_MODES.INTERACTIVE:
            self.pybullet_client.stepSimulation()
            time.sleep(1 / 240)
        elif self.serve_mode == self.SERVE_MODES.GENERATE_DEMOS:
            # self.get_camera_image_from_end_effector()
            self.randomize_object_pose()
            self.randomize_plate_and_drop_pose()

            # Let the simulation settle
            for i in range(10000):
                self.pybullet_client.stepSimulation()
                self.open_gripper()
                for k in range(1, self.num_dofs()):
                    self.pybullet_client.resetJointState(
                        self.splatsim_robot.sim_id,
                        k,
                        self.initial_joint_state[k - 1] * self.joint_signs[k - 1],
                    )
            self.pybullet_client.stepSimulation()

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

        # From GENERATE_DEMOS (serve_loop)
        self.randomize_object_pose()
        self.randomize_plate_and_drop_pose()

        # Get initial joint state and joint signs
        initial_joints = self.splatsim_robot.articulation_config.initial_joint_positions
        joint_signs = self.splatsim_robot.articulation_config.joint_signs

        # Let simulation settle with robot in initial position
        for _ in range(10000):
            self.pybullet_client.stepSimulation()
            self.open_gripper()
            for k in range(1, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id,
                    k,
                    initial_joints[k - 1] * joint_signs[k - 1],
                )

        return self._get_gym_observation(), {"is_success": False}

    def compute_reward(self) -> float:
        """Compute sparse reward based on success."""
        return 1.0 if self.check_success() else 0.0

    def check_success(self) -> bool:
        """Check if objects are placed on the plate.

        Refactored from eval_trajectory_success().
        """
        # Check the MSE of XY position of objects with the drop location
        for i in range(len(self.splatsim_objects) - 1):
            splatsim_obj = self.splatsim_objects[i]
            if splatsim_obj == self.splatsim_robot or splatsim_obj == self.splatsim_background:
                continue
            if splatsim_obj.sim_id is None:
                continue

            object_pos, _ = self.pybullet_client.getBasePositionAndOrientation(
                splatsim_obj.sim_id
            )
            mse = (object_pos[0] - self.drop_ee_pos[0]) ** 2 + (
                object_pos[1] - self.drop_ee_pos[1]
            ) ** 2

            if mse > 0.03:
                return False
        return True

    def check_terminated(self) -> bool:
        """Check if episode should terminate."""
        return self.check_success()


class AppleOnPlatePybulletRobotServer(ObjectOnPlatePybulletRobotServer):
    ENV_CONFIG_NAME = "apple_on_plate"

    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "plastic_apple",
                "splat_object_name": "plastic_apple",
                "grasp_config": [PybulletRobotServerBase.GRASP_CONFIGS["apple"]],
                "rotation_range_z": [0, 0],
            },
            {
                "object_name": "plate",
                "splat_object_name": "plate",
                "grasp_config": [],
                "rotation_range_z": [-np.pi / 6, np.pi / 6],
            },
        ]
    }


class BananaOnPlatePybulletRobotServer(ObjectOnPlatePybulletRobotServer):
    ENV_CONFIG_NAME = "banana_on_plate"

    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "plastic_banana",
                "splat_object_name": "plastic_banana",
                "grasp_config": [
                    PybulletRobotServerBase.GRASP_CONFIGS["banana1"],
                    PybulletRobotServerBase.GRASP_CONFIGS["banana2"],
                ],
                "rotation_range_z": [0, 0],
            },
            {
                "object_name": "plate",
                "splat_object_name": "plate",
                "grasp_config": [],
                "rotation_range_z": [-np.pi / 6, np.pi / 6],
            },
        ]
    }


class OrangeOnPlatePybulletRobotServer(ObjectOnPlatePybulletRobotServer):
    ENV_CONFIG_NAME = "orange_on_plate"

    ENV_CONFIG = {
        "objects": [
            {
                "object_name": "plastic_orange",
                "splat_object_name": "plastic_orange",
                "grasp_config": [PybulletRobotServerBase.GRASP_CONFIGS["orange"]],
                "rotation_range_z": [0, 0],
            },
            {
                "object_name": "plate",
                "splat_object_name": "plate",
                "grasp_config": [],
                "rotation_range_z": [-np.pi / 6, np.pi / 6],
            },
        ]
    }
