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
from splatsim.configs.env_config import GraspConfig


class ObjectOnPlatePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG = None
    background_splat_name = "robot_iphone"
    # TODO: Add table object to ENV_CONFIG and remove this hardcoded TABLE_LIMITS
    # TABLE_LIMITS = ((0.2, 0.6), (-0.5, 0.5))

    # object_rot is only x and y. Since it's a tabletop, z is randomized
    GRASP_CONFIGS = {
        "orange": GraspConfig(
            grasp_pose=np.array(
                [
                    [0.03420832, 0.29551898, 0.95472421, -0.08157158],
                    [-0.82904722, 0.54187654, -0.13802362, -0.14110232],
                    [-0.55813126, -0.7867899, 0.26353588, 0.20728098],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            object_rot=np.array([0, 0])
        ),
        "banana1": GraspConfig(
            grasp_pose=np.array(
                [
                    [-0.13784676, -0.14873802, 0.97922177, 0.01055928],
                    [-0.98239786, 0.14637033, -0.11606107, -0.06527538],
                    [-0.12606632, -0.97798401, -0.16629659, 0.23013977],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            object_rot=np.array([0, 0])
        ),
        "banana2": GraspConfig(
            grasp_pose=np.array(
                [
                    [0.12773567, 0.02665088, -0.99145011, 0.00692899],
                    [-0.87105321, 0.481048, -0.09929316, -0.14203231],
                    [0.47428884, 0.87628908, 0.08466133, -0.20627994],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            object_rot=np.array([0, np.pi]),
        ),
        "apple": GraspConfig(
            grasp_pose=np.array(
                [
                    [-0.12515046, -0.0412762, 0.99127879, 0.00471373],
                    [-0.98896543, -0.07464537, -0.12796658, 0.01413896],
                    [0.07927635, -0.99635553, -0.03147883, 0.27105228],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            object_rot=np.array([0, 0]),
        ),
        # self.strawberry_grasp_pose = np.array([[-0.19612399,  0.06661985,  0.97831344 ,-0.03194745],
        #                                 [-0.90997152, -0.38409934, -0.15626751,  0.10821076],
        #                                 [ 0.36535902, -0.92088517,  0.13595326,  0.23474673],
        #                                 [ 0.,          0. ,         0. ,         1.        ]])
        "strawberry": GraspConfig(
            grasp_pose=np.array(
                [
                    [6.03600159e-04, 4.74883933e-01, 8.80048229e-01, -1.17034260e-01],
                    [-7.31850150e-01, -5.99512796e-01, 3.24005810e-01, 1.57542460e-01],
                    [6.81465328e-01, -6.44258999e-01, 3.47182012e-01, 1.72402069e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            object_rot=np.array([0, 0]),
        ),
    }

    # TODO is there a plastic strawberry env?


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # add plane
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)

        # TODO add this functionality back:
        # # reset the box position
        # for splatsim_obj in self.splatsim_objects:
        #     if splatsim_obj.name == "plate":
        #         self.pybullet_client.resetBasePositionAndOrientation(
        #             splatsim_obj.sim_id,
        #             [0.3, -0.5, 0.02],
        #             p.getQuaternionFromEuler([0, 0, np.pi / 2]),
        #         )
        #         break
        # set the drop location for the apple and banana
        # self.drop_ee_pos = [0.3, -0.5, 0.3]
        # self.drop_ee_euler = [-np.pi / 2, 0, -np.pi / 2]
        # self.drop_ee_quat = self.pybullet_client.getQuaternionFromEuler(
        #     self.drop_ee_euler
        # )
        # limits are +-pi of the initial joint positions
        # self.drop_ee_joint = self.pybullet_client.calculateInverseKinematics(
        #     self.splatsim_robot.sim_id,
        #     6,
        #     self.drop_ee_pos,
        #     self.drop_ee_quat,
        #     maxNumIterations=100000,
        #     residualThreshold=1e-10,
        #     lowerLimits=self.lower_limits,
        #     upperLimits=self.upper_limits,
        # )
        # print("drop_ee_joint", self.drop_ee_joint)
        # set the joint positions to the drop location
        # for i in range(1, self.num_dofs()):
        #     self.pybullet_client.resetJointState(
        #         self.splatsim_robot.sim_id, i, self.drop_ee_joint[i - 1]
        #     )
        # change the friction of the plane
        # self.pybullet_client.changeDynamics(self.plane, -1, lateralFriction=random.uniform(0.2, 1.1))

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

            self.teleport_joint_state(self.splatsim_robot, self.splatsim_robot.articulation_config.initial_joint_positions)

            self.pybullet_client.stepSimulation()

            initial_joint_positions = self.randomize_ee_pose()

            success = self.plan_execute_record_trajectory(
                initial_joint_positions, 
                self.splatsim_robot.articulation_config.joint_signs
            )
            if success:
                self.trajectory_count += 1

            if self.trajectory_count > self.MAX_TRAJECTORY_COUNT:
                print(
                    f"Exiting record_demos mode because max trajectory count of {self.MAX_TRAJECTORY_COUNT} was reached in folder {self.path}"
                )
                self.serve_mode = self.SERVE_MODES.INTERACTIVE
        elif self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            # Handle trajectory generation mode
            self.trajectory_generator.generate_trajectory_batch()

            if self.trajectory_generator.is_complete():
                print(f"[GUI] Completed trajectory generation. Switching to interactive mode.")
                self.serve_mode = self.SERVE_MODES.INTERACTIVE
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

        # Let simulation settle with robot in initial position
        self.teleport_joint_state(self.splatsim_robot, initial_joints)

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
    ENV_CONFIG = {
        "name": "apple_on_plate",
        "objects": [
            {
                "object_name": "plastic_apple",
                "splat_object_name": "plastic_apple",
                "grasp_config": [ObjectOnPlatePybulletRobotServer.GRASP_CONFIGS["apple"]],
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
    ENV_CONFIG = {
        "name": "banana_on_plate",
        "objects": [
            {
                "object_name": "plastic_banana",
                "splat_object_name": "plastic_banana",
                "grasp_config": [
                    ObjectOnPlatePybulletRobotServer.GRASP_CONFIGS["banana1"],
                    ObjectOnPlatePybulletRobotServer.GRASP_CONFIGS["banana2"],
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
    ENV_CONFIG = {
        "name": "orange_on_plate",
        "objects": [
            {
                "object_name": "plastic_orange",
                "splat_object_name": "plastic_orange",
                "grasp_config": [ObjectOnPlatePybulletRobotServer.GRASP_CONFIGS["orange"]],
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
