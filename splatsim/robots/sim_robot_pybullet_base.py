import pickle
import threading
import time
from typing import Any, Dict, Optional, List
import enum
import random
from collections import namedtuple
import math
import os
import pickle
import shutil
import yaml
from argparse import ArgumentParser

from dataclasses import dataclass, field
import numpy as np
import threading
import queue
from e3nn import o3


import torch
import numpy as np
import mujoco
import mujoco.viewer
import zmq
from splatsim.robots.robot import Robot

import cv2
from torchvision.transforms.functional import to_pil_image

assert mujoco.viewer is mujoco.viewer
from gaussian_splatting.scene.cameras import Camera
from gaussian_renderer import render
import urdf_models.models_data as md
import pybullet as p
from pybullet_planning.interfaces.robots.collision import pairwise_collision

from pybullet_planning import RED, BLUE, GREEN
from pybullet_planning import Pose
from pybullet_planning import set_pose
from pybullet_planning import create_box
import pybullet_data
from splatsim.utils.robot_splat_render_utils import (
    get_segmented_indices,
    transform_means,
    get_transfomration_list,
    transform_object,
    get_curr_link_states,
    crop_splat,
    SplatSimObject,
)
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams, Namespace
from gaussian_splatting.scene import Scene

from splatsim.utils.transform_utils import rotation_matrix_to_euler_angles


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                # print(f"Received request: {method}, {args}")
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "set_object_pose":
                    result = self._robot.set_object_pose(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "create_object":
                    splatsim_object = self._robot.create_object(**args)
                    result = None
                elif method == "delete_object":
                    result = self._robot.delete_object(**args)
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass
                # print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class GripperState(str, enum.Enum):
    OPEN = 1
    CLOSE = 0


@dataclass
class PathSegment:
    path_type: str = field(init=False)


@dataclass
class TrajectoryPathSegment(PathSegment):
    path: np.ndarray
    gripper_pos: float
    gripper_velocity: float = 0.2
    threshold: float = 1e-2

    def __post_init__(self):
        self.path_type = "trajectory"


@dataclass
class GripperPathSegment(PathSegment):
    target_state: GripperState
    num_steps: int = 160

    def __post_init__(self):
        self.path_type = "gripper"


class PybulletRobotServerBase:
    MAX_TRAJECTORY_COUNT = 500

    TABLE_LIMITS = ((0.2, 0.6), (-0.5, 0.5))

    # Enum for serve modes
    class SERVE_MODES(enum.Enum):
        GENERATE_DEMOS = "generate_demos"
        INTERACTIVE = "interactive"

    # object_rot is only x and y. Since it's a tabletop, z is randomized
    GRASP_CONFIGS = {
        "orange": {
            "grasp_pose": np.array(
                [
                    [0.03420832, 0.29551898, 0.95472421, -0.08157158],
                    [-0.82904722, 0.54187654, -0.13802362, -0.14110232],
                    [-0.55813126, -0.7867899, 0.26353588, 0.20728098],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "object_rot": [0, 0],
        },
        "banana1": {
            "grasp_pose": np.array(
                [
                    [-0.13784676, -0.14873802, 0.97922177, 0.01055928],
                    [-0.98239786, 0.14637033, -0.11606107, -0.06527538],
                    [-0.12606632, -0.97798401, -0.16629659, 0.23013977],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "object_rot": [0, 0],
        },
        "banana2": {
            "grasp_pose": np.array(
                [
                    [0.12773567, 0.02665088, -0.99145011, 0.00692899],
                    [-0.87105321, 0.481048, -0.09929316, -0.14203231],
                    [0.47428884, 0.87628908, 0.08466133, -0.20627994],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "object_rot": [0, np.pi],
        },
        "apple": {
            "grasp_pose": np.array(
                [
                    [-0.12515046, -0.0412762, 0.99127879, 0.00471373],
                    [-0.98896543, -0.07464537, -0.12796658, 0.01413896],
                    [0.07927635, -0.99635553, -0.03147883, 0.27105228],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "object_rot": [0, 0],
        },
        # self.strawberry_grasp_pose = np.array([[-0.19612399,  0.06661985,  0.97831344 ,-0.03194745],
        #                                 [-0.90997152, -0.38409934, -0.15626751,  0.10821076],
        #                                 [ 0.36535902, -0.92088517,  0.13595326,  0.23474673],
        #                                 [ 0.,          0. ,         0. ,         1.        ]])
        "strawberry": {
            "grasp_pose": np.array(
                [
                    [6.03600159e-04, 4.74883933e-01, 8.80048229e-01, -1.17034260e-01],
                    [-7.31850150e-01, -5.99512796e-01, 3.24005810e-01, 1.57542460e-01],
                    [6.81465328e-01, -6.44258999e-01, 3.47182012e-01, 1.72402069e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "object_rot": [0, 0],
        },
    }

    # TODO is there a plastic strawberry env?

    ENV_CONFIG = None  # To be set in a subclass

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        use_gripper: bool = True,
        serve_mode: str = SERVE_MODES.GENERATE_DEMOS,
        use_link_centers: bool = True,
        robot_name: str = "robot_iphone",
        camera_names: List[str] = ["base_rgb"],
        cam_i: int = 254,
        object_config_path: str = "./configs/object_configs/objects.yaml",
    ):
        self.serve_mode = serve_mode
        self.use_link_centers = use_link_centers
        self.robot_name = robot_name
        self.camera_names = camera_names
        self.cam_i = cam_i

        # load labels.npy
        self.robot_labels = np.load(
            "./data/labels_path/" + self.robot_name + "_labels.npy"
        )
        self.robot_labels = torch.from_numpy(self.robot_labels).to(device="cuda").long()
        self.transformations_cache = None

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self.pybullet_client = p
        self.object_config_path = object_config_path
        self.grasp_poses = {}
        self.pybullet_client.connect(p.GUI)
        self.pybullet_client.setAdditionalSearchPath(
            "./submodules/pybullet-playground-wrapper/pybullet_playground/urdf/pybullet_ur5_gripper/urdf"
        )

        with open(self.object_config_path, "r") as file:
            self.object_config = yaml.safe_load(file)

        self.models_lib = md.model_lib()

        self.splatsim_objects: List[SplatSimObject] = []
        self.splatsim_robot = None # Need this initialization as a placeholder for self.create_object()
        self.splatsim_robot: SplatSimObject = self.create_object(
            object_name="robot", splat_object_name=self.robot_name
        )

        # self.pybullet_client.resetBasePositionAndOrientation(
        #     self.splatsim_robot.sim_id, [0, 0, 0.4], [-1, 0, 0, 0]
        # )

        # self.background_splat_name = self.robot_name
        # self.splatsim_background = self.splatsim_robot

        self.background_splat_name = "bwa_open_space" # self.robot_name
        # The background uses the robot's full splat, but crops out the robot
        self.splatsim_background: SplatSimObject = self.create_object(
            object_name="background",
            splat_object_name=self.background_splat_name,
            keep_within_aabb=False,
            load_urdf=False,
        )
        self.splatsim_background.is_articulated = False

        self.skip_recording_first = 0

        for i in range(self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)):
            info = self.pybullet_client.getJointInfo(self.splatsim_robot.sim_id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id

        self.use_gripper = use_gripper
        if self.use_gripper:
            self.setup_gripper()
        # else:
        #     self.setup_spatula()
        #     pass

        # self.offsets = [np.pi / 2, 0, 0, 0, 0, 0, 0]
        # This has an extra 0 at the beginning for the world joint, and then another 0 for a fixed joint, I think
        self.initial_joint_state = self.splatsim_robot.object_config["joint_states"][0]
        # Remove the extra 0 for the world joint
        self.initial_joint_state = self.initial_joint_state[1:]
        # + 1 joint for the world joint at the beginning which will be skipped
        num_joints = self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)
        if len(self.initial_joint_state) > num_joints:
            print(
                f"Warning: Provided initial joint positions ({len(self.initial_joint_state)}) exceed the number of joints ({num_joints}). Truncating to {num_joints} positions."
            )
            self.initial_joint_state = self.initial_joint_state[:num_joints]
        # This is a no-op
        self.joint_signs = [1] * len(self.initial_joint_state)

        # self.initial_joint_state = [0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

        # model_lib = md.model_lib()
        # objectid = self.pybullet_client.loadURDF(model_lib['potato_chip_1'], [0.5, 0.15, 0])

        # random euler angles for the orientation of the object
        # euler_z = random.uniform(-np.pi, np.pi)
        # random quaternion for the orientation of the object

        for object_cfg in self.ENV_CONFIG["objects"]:
            object_name = object_cfg["object_name"]
            splat_object_name = object_cfg["splat_object_name"]
            grasp_config = object_cfg["grasp_config"]

            # Already adds the splatsim_object to self.splatsim_objects
            splatsim_object = self.create_object(
                object_name, splat_object_name=splat_object_name
            )

            splatsim_object.grasp_configs = grasp_config

        self.randomize_object_pose()

        # reset the box position
        for splatsim_obj in self.splatsim_objects:
            if splatsim_obj.name == "plate":
                self.pybullet_client.resetBasePositionAndOrientation(
                    splatsim_obj.sim_id,
                    [0.3, -0.5, 0.02],
                    p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                )
                break

        # set the drop location for the apple and banana
        self.drop_ee_pos = [0.3, -0.5, 0.3]
        self.drop_ee_euler = [-np.pi / 2, 0, -np.pi / 2]
        self.drop_ee_quat = self.pybullet_client.getQuaternionFromEuler(
            self.drop_ee_euler
        )

        # set initial joint positions
        for i in range(1, len(self.initial_joint_state)):
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id, i, self.initial_joint_state[i - 1]
            )

        # limits are +-pi of the initial joint positions
        lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
        upper_limits = [np.pi, 0, np.pi, np.pi, np.pi, np.pi]
        self.drop_ee_joint = self.pybullet_client.calculateInverseKinematics(
            self.splatsim_robot.sim_id,
            6,
            self.drop_ee_pos,
            self.drop_ee_quat,
            maxNumIterations=100000,
            residualThreshold=1e-10,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
        )

        print("drop_ee_joint", self.drop_ee_joint)

        # set the joint positions to the drop location

        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id, i, self.drop_ee_joint[i - 1]
            )

        # change the friction of the object
        for splatsim_obj in self.splatsim_objects:
            if splatsim_obj.sim_id is not None:
                self.pybullet_client.changeDynamics(
                    splatsim_obj.sim_id, -1, lateralFriction=1.5
                )
                # rolling friction
                self.pybullet_client.changeDynamics(
                    splatsim_obj.sim_id, -1, rollingFriction=0
                )
                inertia = p.getDynamicsInfo(splatsim_obj.sim_id, -1)[2]

        # add gravity
        self.pybullet_client.setGravity(0, 0, -9.81)

        # add plane
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.pybullet_client.loadURDF("plane.urdf", [0, 0, -0.022])

        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = self.pybullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.wall = self.pybullet_client.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)

        ## add stage
        self.stage = 0

        # change the friction of the plane
        # self.pybullet_client.changeDynamics(self.plane, -1, lateralFriction=random.uniform(0.2, 1.1))

        # set time step
        self.pybullet_client.setTimeStep(1 / 240)

        # current gripper state
        self.current_gripper_action = 0

        # trajectory path
        with open("configs/folder_configs.yaml", "r") as f:
            folder_config = yaml.safe_load(f)
        self.path = folder_config["traj_folder"]
        # get no of folders in the path
        self.trajectory_count = len(os.listdir(self.path))

        # step simulation
        for i in range(100):
            self.pybullet_client.stepSimulation()
            # time.sleep(1/240)

        # Placeholder object for rendering purposes
        self.scene_gaussian = GaussianModel(3)

        if "base_rgb" in self.camera_names:
            source_path = self.splatsim_background.object_config["source_path"]
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source path not found: {source_path}")

            model_path = self.splatsim_background.object_config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")

            parser = ArgumentParser(description="Testing script parameters")
            self.pipeline = PipelineParams(parser)
            model = ModelParams(parser, sentinel=True)
            dataset = model.extract(
                Namespace(
                    sh_degree=3,
                    # TODO get these from the object config
                    source_path=source_path,
                    model_path=model_path,
                    images="images",
                    depths="",
                    resolution=-1,
                    white_background=False,
                    train_test_exp=False,
                    data_device="cuda",
                    eval=False,
                )
            )
            self.cam_scale = 1 #2
            temp_gaussian_model = GaussianModel(3)
            self.scene = Scene(
                dataset,
                temp_gaussian_model, # This is just used for camera initialization
                load_iteration=-1,
                shuffle=False,
                resolution_scales=[self.cam_scale],
                train_cam_indices=[self.cam_i],
                test_cam_indices=[], # we're using train cameras
            )

            bg_color = [1, 1, 1]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            self.base_camera = self.setup_camera_from_dataset(
                cam_i=self.cam_i, use_train=True
            )
        else:
            self.base_camera = None
            self.scene = None
            self.pipeline = None
            self.background = None

        self.wrist_camera_link_index = None
        if "wrist_rgb" in self.camera_names:
            # Get the index of the wrist_camera_link
            if "wrist_camera_link_name" in self.splatsim_robot.object_config:
                wrist_camera_link_name = self.splatsim_robot.object_config[
                    "wrist_camera_link_name"
                ]
                num_joints = p.getNumJoints(self.splatsim_robot.sim_id)
                for i in range(num_joints):
                    info = p.getJointInfo(self.splatsim_robot.sim_id, i)
                    if info[12].decode("utf-8") == wrist_camera_link_name:
                        self.wrist_camera_link_index = i
                        break
                if self.wrist_camera_link_index is None:
                    raise ValueError(
                        f"Cannot find wrist camera link name {wrist_camera_link_name}"
                    )
            else:
                raise ValueError(
                    f"wrist_camera_link_name attribute not defined in object config of robot {self.robot_name}, yet wrist camera was requested"
                )

    def num_dofs(self) -> int:
        return 7

    def get_joint_state(self) -> np.ndarray:
        # return self._joint_state
        joint_states = []
        num_joints = self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)
        for i in range(1, num_joints):
            joint_states.append(
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[0]
            )
        return np.array(joint_states)

    def get_joint_state_dummy(self) -> np.ndarray:
        # return self._joint_state
        joint_states = []
        num_joints = self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)
        for i in range(1, num_joints):
            joint_states.append(
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[0]
            )
        return np.array(joint_states)

    def set_serve_mode(self, serve_mode: SERVE_MODES) -> None:
        if not isinstance(serve_mode, self.SERVE_MODES):
            print(
                f"ERROR: Expected serve_mode to be an enum instance of SERVE_MODES, got {type(serve_mode)}"
            )
        else:
            print(f"Setting serve mode to {serve_mode}")
            self.serve_mode = serve_mode

    def load_gaussian_splat(self, gaussians, object_config):
        if "ply_path" in object_config:
            ply_path = object_config["ply_path"]
            gaussians.load_ply(ply_path)
        elif "source_path" in object_config and "model_path" in object_config:
            source_path = object_config["source_path"]
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source path not found: {source_path}")

            model_path = object_config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")

            parser = ArgumentParser(description="Testing script parameters")
            model = ModelParams(parser, sentinel=True)
            dataset = model.extract(
                Namespace(
                    sh_degree=3,
                    source_path=source_path,
                    model_path=model_path,
                    images="images",
                    depths="",
                    resolution=-1,
                    white_background=False,
                    train_test_exp=False,
                    data_device="cuda",
                    eval=False,
                )
            )
            # This loads the .ply file into gaussians
            scene = Scene(
                dataset,
                gaussians,
                load_iteration=-1,
                shuffle=False,
                resolution_scales=[],
                train_cam_indices=[],
                test_cam_indices=[] ,
            )
            del scene
        else:
            raise ValueError("Could not load gaussian splat")
        
        # Disable gradients on this gaussian splat b/c we're not optimizing
        gaussians._xyz.requires_grad = False
        gaussians._rotation.requires_grad = False
        gaussians._opacity.requires_grad = False
        gaussians._features_rest.requires_grad = False
        gaussians._features_dc.requires_grad = False
        gaussians._scaling.requires_grad = False
            
        return gaussians

    def delete_object(self, object_name):
        index = [splatsim_obj.name for splatsim_obj in self.splatsim_objects].index(
            object_name
        )
        splatsim_obj = self.splatsim_objects.pop(index)

        # Explicitly delete some values
        del splatsim_obj.gaussians
        if splatsim_obj.sim_id is not None:
            p.removeBody(splatsim_obj.sim_id)

    def create_object(
        self,
        object_name,
        object_config={},
        splat_object_name=None,
        keep_within_aabb=True,
        load_urdf=True,
    ):
        if len(object_config) == 0 and splat_object_name is None:
            splat_object_name = object_name  # TODO when is this wrong?

        gaussians = GaussianModel(3)

        # Find object config
        if splat_object_name is not None:
            object_config = self.object_config.get(splat_object_name, {})
        if object_config.get("object_type", None) == "cuboid":
            # Use the redblock object's gaussian splat b/c it's a nice rectangular prism
            # object_config has higher priority than the redblock config if there are overlapping attributes
            object_config = {**self.object_config["redblock"], **object_config}
        elif len(object_config) == 0:
            print("WARNING: No object config found for ", splat_object_name)

        use_fixed_base = object_config.get("use_fixed_base", False)
        global_scaling = object_config.get("global_scaling", 1)
        is_articulated = object_config.get("is_articulated", False)
        base_position = object_config.get("base_position", [[0, 0, 0]])[0]

        self.load_gaussian_splat(gaussians, object_config)

        # Find possible URDF config
        if object_name in self.models_lib.model_name_list:
            urdf_path = self.models_lib[splat_object_name]
        elif "urdf_path" in object_config:
            urdf_path = object_config["urdf_path"][0]
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        else:
            urdf_path = None

        if load_urdf:
            if is_articulated:
                flags = self.pybullet_client.URDF_USE_IMPLICIT_CYLINDER
            else:
                flags = 0

            if urdf_path is not None:
                # This object has an associated urdf file. Use that
                # TODO possibly do custom quat
                quat = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
                object_loaded = self.pybullet_client.loadURDF(
                    urdf_path,
                    base_position,
                    quat,
                    globalScaling=global_scaling,
                    useFixedBase=use_fixed_base,
                    flags=flags,
                )
                mass = self.pybullet_client.getDynamicsInfo(object_loaded, -1)[0]
            elif object_config.get("object_type", None) is not None:
                # Find primitive shape config
                object_type = object_config["object_type"]
                if object_type == "cuboid":
                    # position, orientation, size
                    # TODO orientation
                    lx, ly, lz = object_config["size"]
                    position = object_config["position"]

                    if load_urdf:
                        # Apply global scaling
                        lx, ly, lz = (
                            lx * global_scaling,
                            ly * global_scaling,
                            lz * global_scaling,
                        )
                        position = [
                            position[0] * global_scaling,
                            position[1] * global_scaling,
                            position[2] * global_scaling,
                        ]

                        object_loaded = create_box(lx, ly, lz, color=BLUE)
                        set_pose(object_loaded, Pose(point=position))
                        # TODO set orientation
                        mass = (
                            0
                            if use_fixed_base or "mass" not in object_config
                            else object_config["mass"]
                        )
                        self.pybullet_client.changeDynamics(object_loaded, -1, mass=mass)
                    else:
                        object_loaded = None
                        mass = 0

                    # Customize the redblock splat rectanglular prism to have the right dimensions
                    robot_scale = self.splatsim_robot.transformations_cache["transformation_scale"]
                    for axis, actual_len in zip(range(3), [lx, ly, lz]):
                        redblock_len = gaussians._xyz[:, axis].max() - gaussians._xyz[:, axis].min()
                        ratio = actual_len / redblock_len / robot_scale
                        gaussians._xyz[:, axis] = gaussians._xyz[:, axis] * ratio
                        # Do the ratio in exponential space
                        gaussians._scaling[:, axis] = torch.log(gaussians.get_scaling[:, axis] * ratio)

                    # TODO change this to a brownish color and also adjust the size of the block to the size of the cuboid
                else:
                    raise ValueError(f"Cannot create unknown object type {object_type}")
            elif len(gaussians._xyz) > 0: # The gaussian splat was loaded, but there's no urdf
                # Create a box that covers the gaussian means in the gaussian splat. This is an approximation
                lx = max(gaussians._xyz[:, 0]) - min(gaussians._xyz[:, 0])
                ly = max(gaussians._xyz[:, 1]) - min(gaussians._xyz[:, 1])
                lz = max(gaussians._xyz[:, 2]) - min(gaussians._xyz[:, 2])

                position = object_config.get("position", [0, 0, 0])

                # Apply global scaling
                lx, ly, lz = (
                    lx * global_scaling,
                    ly * global_scaling,
                    lz * global_scaling,
                )
                position = [
                    position[0] * global_scaling,
                    position[1] * global_scaling,
                    position[2] * global_scaling,
                ]

                object_loaded = create_box(lx, ly, lz, color=BLUE)
                set_pose(object_loaded, Pose(point=position))
                # TODO set orientation
                mass = (
                    0
                    if use_fixed_base or "mass" not in object_config
                    else object_config["mass"]
                )
                self.pybullet_client.changeDynamics(object_loaded, -1, mass=mass)
            else:
                raise ValueError(
                    f"Could not parse object config for object name {object_name}"
                )
        else:
            object_loaded = None
            mass = 0

        transformations_cache = self.create_transformations_cache(object_config)

        splatsim_obj = SplatSimObject(
            name=object_name,
            splat_name=splat_object_name,
            gaussians=gaussians,
            transformations_cache=transformations_cache,
            is_articulated=is_articulated,
            sim_id=object_loaded,
            mass=mass,
            object_config=object_config
        )

        # Transform the xyz, rotation, and shs features to the canonical frame (the world frame for the simulator)
        # We will work in the coordinate frame of the simulator from now on
        Trans_canonical = torch.from_numpy(np.array(splatsim_obj.object_config['transformation']['matrix'])).to(device=splatsim_obj.gaussians.get_xyz.device).float() # shape (4, 4)
        xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj = transform_object(
            splatsim_obj=splatsim_obj, splatsim_robot=splatsim_obj, transform=Trans_canonical
        )

        splatsim_obj.gaussians._xyz = xyz_obj
        splatsim_obj.gaussians._rotation = rot_obj
        splatsim_obj.gaussians._opacity = opacity_obj
        splatsim_obj.gaussians._features_rest = features_rest_obj
        splatsim_obj.gaussians._features_dc = features_dc_obj
        splatsim_obj.gaussians._scaling = scales_obj

        if self.splatsim_robot is None and object_name == "robot":
            # This is trying to initialize the robot
            crop_splat(splatsim_obj, splatsim_obj, keep_within_aabb=keep_within_aabb)
        else:
            crop_splat(splatsim_obj, self.splatsim_robot, keep_within_aabb=keep_within_aabb)

        self.splatsim_objects.append(splatsim_obj)
        return splatsim_obj

    def set_object_pose(
        self,
        object_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
        use_gravity: bool = True,
    ) -> None:
        """Set the pose of an object in the simulation."""
        if object_name not in [
            splatsim_obj.splat_name for splatsim_obj in self.splatsim_objects
        ]:
            print(f"Object {object_name} not found in splat_object_name_list.")
            return

        object_i = [splatsim_obj.name for splatsim_obj in self.splatsim_objects].index(
            object_name
        )
        splatsim_obj = self.splatsim_objects[object_i]

        if splatsim_obj.sim_id is None:
            raise ValueError(
                "Cannot set pose of object not represented in pybullet (ex: has urdf)"
            )
    
        self.pybullet_client.resetBasePositionAndOrientation(
            splatsim_obj.sim_id, position, orientation
        )

        if not use_gravity:
            # Make the object static so that it doesn't move
            self.pybullet_client.changeDynamics(splatsim_obj.sim_id, -1, mass=0)
        else:
            self.pybullet_client.changeDynamics(
                splatsim_obj.sim_id,
                -1,
                mass=splatsim_obj.mass,
            )

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs(), (
            f"Expected joint state of length {self.num_dofs()}, "
            f"got {len(joint_state)}."
        )

        for i in range(1, self.num_dofs()):
            self.pybullet_client.setJointMotorControl2(
                self.splatsim_robot.sim_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_state[i - 1],
                force=1000,
                # force=250,
            )

        if self.use_gripper:
            self.move_gripper((1 - joint_state[-1]) * 0.085)
            self.current_gripper_action = joint_state[-1]

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def create_transformations_cache(self, object_config):
        # object_config is in the format of an object in configs/object_configs/objects.yaml
        object_transformation = np.array(object_config["transformation"]["matrix"])
        object_transformation = (
            torch.tensor(object_transformation).to(device="cuda").float()
        )
        object_transformation_scale = torch.pow(
            torch.linalg.det(object_transformation[:3, :3]), 1 / 3
        )
        object_transformation_inv = torch.inverse(object_transformation)
        object_transformation_inv_scale = torch.pow(
            torch.linalg.det(object_transformation_inv[:3, :3]), 1 / 3
        )
        transformations_cache = {
            "transformation": object_transformation,
            "transformation_scale": object_transformation_scale,
            "inv_transformation": object_transformation_inv,
            "inv_transformation_scale": object_transformation_inv_scale,
        }
        return transformations_cache

    def get_wrist_camera(self):
        if self.wrist_camera_link_index is None:
            print("WARNING: No wrist camera index found")
            return None

        uid = 0
        colmap_id = 1

        # Get the pose of the wrist_camera_link
        link_state = p.getLinkState(
            self.splatsim_robot.sim_id,
            self.wrist_camera_link_index,
            computeForwardKinematics=True,
        )

        # robot_transformation = np.array(
        #     self.splatsim_robot.object_config["transformation"]["matrix"]
        # )
        if self.transformations_cache is None:
            robot_transformation = torch.tensor(
                self.splatsim_robot.object_config["transformation"]["matrix"],
                device="cuda",
            )
            robot_transformation_inv = torch.linalg.inv(robot_transformation)
        else:
            robot_transformation = self.transformations_cache[self.robot_name][
                "transformation"
            ]
            robot_transformation_inv = self.transformations_cache[self.robot_name][
                "inv_transformation"
            ]

        T = torch.tensor(
            link_state[0], device=robot_transformation.device
        ).float()  # xyz position in world frame
        quat = link_state[1]
        R = (
            torch.tensor(
                p.getMatrixFromQuaternion(quat), device=robot_transformation.device
            )
            .reshape(3, 3)
            .float()
        )
        Trans_cam_world = torch.eye(4, device=R.device)
        Trans_cam_world[:3, :3] = R
        Trans_cam_world[:3, 3] = T

        robot_transformation[:3, 3] = robot_transformation[:3, 3]
        Trans_cam_splat = torch.matmul(robot_transformation_inv, Trans_cam_world)

        FoVx = 1.375955594372348
        FoVy = 1.1025297299614814

        image_width = 640
        image_height = 480
        image_name = "wrist_rgb"
        image = torch.zeros((3, image_height, image_width)).float()

        # Original camera-to-world transform
        R_cw = Trans_cam_splat[:3, :3]
        T_cw = Trans_cam_splat[:3, 3]
        scale = torch.pow(torch.linalg.det(R_cw[:3, :3]), 1 / 3)
        R_cw = R_cw / scale
        T_cw = T_cw / scale
        Trans_cam_splat_wo_scale = torch.eye(4, device=R_cw.device)
        Trans_cam_splat_wo_scale[:3, :3] = R_cw
        Trans_cam_splat_wo_scale[:3, 3] = T_cw

        # Convert to world-to-camera
        Rt_wc = torch.linalg.inv(Trans_cam_splat_wo_scale)
        T_wc = Rt_wc[:3, 3]

        resolution = (image_width, image_height)
        depth_params = None
        invdepthmap = None

        # I really don't understand why this combination of rotation and translation matrices fixes calibration...
        camera = Camera(
            resolution,
            colmap_id,
            R_cw.detach().cpu().numpy(),
            T_wc.detach().cpu().numpy(),
            FoVx,
            FoVy,
            depth_params,
            to_pil_image(image),
            invdepthmap,
            # gt_mask_alpha,
            image_name,
            uid,
            scale=scale.detach().cpu().numpy(),
        )

        return camera

    def prep_image_rendering(self, data) -> Dict[str, np.ndarray]:
        # Transform each object splat to be in the right pose
        del self.scene_gaussian._xyz
        del self.scene_gaussian._rotation
        del self.scene_gaussian._opacity
        del self.scene_gaussian._features_rest
        del self.scene_gaussian._features_dc
        del self.scene_gaussian._scaling

        xyz_obj_list = []
        rot_obj_list = []
        opacity_obj_list = []
        scales_obj_list = []
        features_dc_obj_list = []
        features_rest_obj_list = []
        for i in range(len(self.splatsim_objects)):
            splatsim_obj = self.splatsim_objects[i]
            if splatsim_obj.is_articulated:
                assert (
                    splatsim_obj == self.splatsim_robot
                ), "Other articulated objects are not implemented yet"

                # Gets transformations for all links of the robot based on the current simulation
                transformations_list = get_transfomration_list(
                    splatsim_obj.sim_id, self.initial_link_states
                )

                # TODO generalize this to "every articulated object" instead of just the robot
                # TODO does this need to be done every time?
                # Ah. it's because xyz gets overwritten
                segmented_list, xyz = get_segmented_indices(
                    splatsim_obj=splatsim_obj,
                    # splatsim_robot=self.splatsim_robot,  # This is to get robot transformations
                    robot_labels=self.robot_labels,
                )

                (
                    xyz_obj,
                    rot_obj,
                    opacity_obj,
                    scales_obj,
                    features_rest_obj,
                    features_dc_obj,
                ) = transform_means(
                    splatsim_obj=splatsim_obj,
                    splatsim_robot=self.splatsim_robot,  # this is to get robot transformations
                    xyz=xyz,
                    segmented_list=segmented_list,
                    transformations_list=transformations_list,
                )

                # # Now, apply the object's transformation within the simulation. Assume there is no scaling within pybullet
                # # xyz_obj and rot_obj need to be changed
                # (
                #     object_pos,
                #     object_quat,
                # ) = self.pybullet_client.getBasePositionAndOrientation(
                #     self.splatsim_objects[i].sim_id
                # )

                # # xyz_obj, rotation_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj

                # (
                #     xyz_obj,
                #     rot_obj,
                #     opacity_obj,
                #     scales_obj,
                #     features_dc_obj,
                #     features_rest_obj,
                # ) = transform_splat(
                #     xyz_obj,
                #     rot_obj,
                #     opacity_obj,
                #     scales_obj,
                #     features_dc_obj,
                #     features_rest_obj,
                #     torch.tensor(object_pos).to(xyz_obj.device),
                #     torch.tensor(object_quat).to(xyz_obj.device)
                # )

            else:
                if splatsim_obj.sim_id is not None:
                    cur_object_position = torch.tensor(
                        data[splatsim_obj.name + "_position"], device="cuda"
                    ).float()
                    base_position = torch.tensor(
                        splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0],
                        device="cuda",
                    ).float()
                    cur_object_rotation = torch.tensor(
                        data[splatsim_obj.name + "_orientation"], device="cuda"
                    ).float()
                else:
                    # TODO currently, objects that aren't urdfs in sim don't ever move
                    cur_object_position = torch.tensor([0, 0, 0], device="cuda").float()
                    base_position = torch.tensor([0, 0, 0], device="cuda").float()
                    cur_object_rotation = torch.tensor(
                        [0, 0, 0, 1], device="cuda"
                    ).float()
                cur_object_position = cur_object_position - base_position
                cur_object_rotation = torch.roll(
                    cur_object_rotation,
                    1,
                )
                (
                    xyz_obj,
                    rot_obj,
                    opacity_obj,
                    scales_obj,
                    features_dc_obj,
                    features_rest_obj,
                ) = transform_object(
                    splatsim_obj=splatsim_obj,
                    splatsim_robot=self.splatsim_robot,
                    pos=cur_object_position,
                    quat=cur_object_rotation,
                )

            xyz_obj_list.append(xyz_obj)
            rot_obj_list.append(rot_obj)
            opacity_obj_list.append(opacity_obj)
            scales_obj_list.append(scales_obj)
            features_dc_obj_list.append(features_dc_obj)
            features_rest_obj_list.append(features_rest_obj)

        # Combine splats of robot and of objects
        with torch.no_grad():
            # gaussians.active_sh_degree = 0
            self.scene_gaussian._xyz = torch.cat(
                xyz_obj_list,
                dim=0,
            )
            self.scene_gaussian._rotation = torch.cat(
                rot_obj_list,
                dim=0,
            )
            self.scene_gaussian._opacity = torch.cat(
                opacity_obj_list,
                dim=0,
            )
            self.scene_gaussian._features_rest = torch.cat(
                features_rest_obj_list,
                dim=0,
            )
            self.scene_gaussian._features_dc = torch.cat(
                features_dc_obj_list,
                dim=0,
            )
            self.scene_gaussian._scaling = torch.cat(
                scales_obj_list,
                dim=0,
            )

    def render_image(self, camera_name):
        if camera_name == "base_rgb":
            camera = self.base_camera
        elif camera_name == "wrist_rgb":
            camera = self.get_wrist_camera()
            if camera is None:
                return None
        else:
            raise ValueError(f"Unknown camera name {camera_name}")
        
        rendering = render(camera, self.scene_gaussian, self.pipeline, self.background)[
            "render"
        ].cpu()
        # If you index "depth" instead of "render", you get the depth image

        # save the image
        return rendering

    def setup_camera_from_dataset(self, cam_i, use_train=True):
        # Assume that self.cam_train_indices and self.cam_test_indices have already singled out
        # the camera of interest. Return the first camera in the list
        if use_train:
            camera = self.scene.getTrainCameras(scale=self.cam_scale)[0]
        else:
            camera = self.scene.getTestCameras(scale=self.cam_scale)[0]

        # The camera was saved in the world frame of the self.splatsim_background object.
        # Transform it to be in the coordinate frame of the simulator

    #         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug,
    #     antialiasing=pipe.antialiasing





        # device = camera.world_view_transform.device
        # Trans_canonical = torch.from_numpy(np.array(self.splatsim_background.object_config['transformation']['matrix'])).to(device=device).float() # shape (4, 4)
        # scale_obj = torch.pow(torch.linalg.det(Trans_canonical[:3, :3]), 1/3)
        # R = Trans_canonical[:3, :3].clone()
        # T = Trans_canonical[:3, 3].clone()
        # Trans_canonical[:3, :3] = Trans_canonical[:3, :3] / scale_obj
        # Trans_canonical[:3, 3] = Trans_canonical[:3, 3] / scale_obj
        
        # P_matrix = torch.matmul(camera.full_proj_transform, torch.linalg.inv(camera.world_view_transform))

        # camera.world_view_transform = camera.world_view_transform @ Trans_canonical.T
        # # camera.world_view_transform = camera.world_view_transform @ torch.inverse(Trans_canonical).T
        # camera.camera_center = R @ camera.camera_center + T
        # # camera.camera_center = (Trans_canonical @ torch.concatenate([camera.camera_center, torch.tensor([1.0], dtype=torch.float32, device=device)]))[:3]
        # # This is calculated with the updated camera.world_view_transform
        # camera.full_proj_transform = torch.matmul(P_matrix, camera.world_view_transform)






        # 1. Define device and Trans_canonical
        device = camera.world_view_transform.device
        Trans_canonical_full = torch.from_numpy(
            np.array(self.splatsim_background.object_config['transformation']['matrix'])
        ).to(device=device).float()

        # 2. Get the camera's pose in the SPLAT'S LOCAL FRAME
        # V_original is the view matrix in the splat's local frame.
        # M_CW_original is the Camera-to-World (C2W) matrix in the splat's local frame.
        # We assume V_original's inverse is the C2W matrix.
        V_original_inv = torch.linalg.inv(camera.world_view_transform.clone()).T
        M_CW_original_local = V_original_inv

        # 3. Calculate the new ABSOLUTE Camera-to-World matrix
        # M_CW_world = T_world_from_local @ M_CW_local
        # This matrix now has the full scale (s) embedded in it.
        # background camera to background frame; background frame to world
        # want: background camera to world
        M_CW_new_world_scaled = torch.matmul(Trans_canonical_full, M_CW_original_local)
        # M_CW_new_world_scaled = torch.matmul(M_CW_original_local, Trans_canonical_full)

        # 4. Decompose the new world pose (just like get_wrist_camera)
        # We need to extract R_cw (scale-free), T_wc (scale-free), and scale (s)
        
        # Get the scaled rotation and translation components
        R_cw_scaled = M_CW_new_world_scaled[:3, :3]
        T_cw_scaled = M_CW_new_world_scaled[:3, 3]

        # Get the scalar scale
        scale = torch.pow(torch.linalg.det(R_cw_scaled), 1/3)
        scale_np = scale.detach().cpu().numpy()

        # Normalize the components to be scale-free
        R_cw_normalized = R_cw_scaled / scale
        T_cw_normalized = T_cw_scaled / scale

        # 5. Build the final SCALE-FREE C2W pose matrix
        M_CW_new_world_pose = torch.eye(4, device=device, dtype=torch.float32)
        M_CW_new_world_pose[:3, :3] = R_cw_normalized
        M_CW_new_world_pose[:3, 3] = T_cw_normalized

        # 6. Calculate T_wc (World-to-Camera Translation)
        # This is the translation component of the final View Matrix (V_new)
        V_new_world = torch.linalg.inv(M_CW_new_world_pose)
        T_wc = V_new_world[:3, 3]

        # Convert to numpy for the Camera constructor
        R_cw_np = R_cw_normalized.detach().cpu().numpy()
        T_wc_np = T_wc.detach().cpu().numpy()

        # 7. Initialize the New Camera
        # (Assuming other parameters are loaded as before)
        resolution = (camera.alpha_mask.shape[2], camera.alpha_mask.shape[1])
        image = torch.zeros((3, resolution[1], resolution[0])).float()
        depth_params = None

        new_camera = Camera(
            resolution,
            camera.colmap_id,
            R_cw_np,           # R_cw (scale-free C2W rotation)
            T_wc_np,           # T_wc (scale-free W2C translation)
            camera.FoVx,
            camera.FoVy,
            depth_params,
            to_pil_image(image),
            camera.invdepthmap,
            camera.image_name,
            camera.uid,
            scale=scale_np,    # The separate scale scalar
        )

        return new_camera



        # # 1. Define device and Trans_canonical (from your previous code)
        # device = camera.world_view_transform.device
        # Trans_canonical_full = torch.from_numpy(np.array(self.splatsim_background.object_config['transformation']['matrix'])).to(device=device).float()

        # # 2. Scale Normalization: Create T_pose (scale-normalized transformation)
        # scale = torch.pow(torch.linalg.det(Trans_canonical_full[:3, :3]), 1/3)
        # Trans_pose = Trans_canonical_full.clone()
        # Trans_pose[:3, :3] = Trans_pose[:3, :3] / scale
        # # removed
        # Trans_pose[:3, 3] = Trans_pose[:3, 3] / scale # CRITICAL: Scale the translation

        # # 3. Calculate Original Camera-to-World Matrix (M_CW, original)
        # # V_original is typically the transpose of M_CW_original, but since V_original
        # # is not guaranteed to be a pure R|T matrix (it's V=P_inv @ P_full), we use the inverse.
        # V_original_inv = torch.linalg.inv(camera.world_view_transform.clone()).T
        # # The translation is in the 4th row, so ensure the inverse yields an R|T matrix.
        # # For a row-vector V: M_CW = V^T. Let's assume M_CW = V_inv for safety.
        # M_CW_original = V_original_inv

        # # 4. Calculate New Camera-to-World Matrix (M_CW, new)
        # # M_CW_new = M_CW_original @ T_pose
        # M_CW_new = torch.matmul(M_CW_original, Trans_pose)

        # # Extract and Normalize R_cw
        # R_cw = M_CW_new[:3, :3]
        # T_cw = M_CW_new[:3, 3] # This is the scale-normalized T_cw

        # # The R_cw matrix here is already the scale-normalized rotation from the M_CW_new calculation.
        # # You could re-normalize if needed, but M_CW_new should be scale-free in R.
        # # scale_recheck = torch.pow(torch.linalg.det(R_cw[:3, :3]), 1 / 3) # Should be close to 1.0

        # # Calculate T_wc (World-to- Translation)
        # # T_wc is the translation component of the VIEW MATRIX (M_CW_new)^-1
        # Rt_wc = torch.linalg.inv(M_CW_new) # This is the new V_new
        # T_wc = Rt_wc[:3, 3]

        # # Convert to numpy for the Camera constructor
        # R_cw_np = R_cw.detach().cpu().numpy()
        # T_wc_np = T_wc.detach().cpu().numpy()
        # scale_np = scale.detach().cpu().numpy()

        # # 5. Initialize the New Camera

        # # Define placeholders for other parameters (assuming they are set elsewhere)
        # depth_params = None

        # # t doesnt have scale, but r has scale
        # resolution = (camera.alpha_mask.shape[2], camera.alpha_mask.shape[1])
        # image = torch.zeros((3, resolution[1], resolution[0])).float() # Dummy image

        # # No way to get the original depth params, and it's only used to initialize invdepthmap, which we have
        # depth_params = None

        # new_camera = Camera(
        #     resolution,
        #     camera.colmap_id,
        #     R_cw_np,           # R_cw (Camera-to-World Rotation)
        #     T_wc_np,           # T_wc (World-to-Camera Translation)
        #     camera.FoVx,
        #     camera.FoVy,
        #     depth_params,
        #     to_pil_image(image), # to_pil_image utility needed here
        #     camera.invdepthmap,
        #     camera.image_name,
        #     camera.uid,
        #     scale=scale_np,
        # )

        # return new_camera

    def get_current_ee_pose(self):
        dummy_ee_pos, dummy_ee_quat = (
            self.pybullet_client.getLinkState(self.splatsim_robot.sim_id, 6)[0],
            self.pybullet_client.getLinkState(self.splatsim_robot.sim_id, 6)[1],
        )
        return dummy_ee_pos, dummy_ee_quat

    def get_current_object_pose(self, object_name=None, object_id=None):
        if object_name is not None:
            if object_name not in [
                splatsim_obj.name for splatsim_obj in self.splatsim_objects
            ]:
                raise ValueError(
                    f"Object name '{object_name}' not found when querying its pose."
                )
            queried_object_id = [
                splatsim_obj.name for splatsim_obj in self.splatsim_objects
            ].index(object_name)
            if object_id is not None:
                assert object_id == queried_object_id
            object_id = queried_object_id
        elif object_id is None:
            raise ValueError("No object_name or object_id given!")

        body_id = self.splatsim_objects[object_id].sim_id
        pos, quat = self.pybullet_client.getBasePositionAndOrientation(body_id)
        return pos, quat

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self.get_joint_state()
        joint_positions_dummy = self.get_joint_state_dummy()
        joint_velocities = np.array(
            [
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[1]
                for i in range(7)
            ]
        )

        dummy_ee_pos, dummy_ee_quat = self.get_current_ee_pose()
        # get the euler angles from the quaternion
        dummy_ee_euler = self.pybullet_client.getEulerFromQuaternion(dummy_ee_quat)

        # print the euler angles and the reconstructed quaternion
        if self.use_gripper:
            self.current_gripper_state = self.get_current_gripper_state() / 0.8
        else:
            self.current_gripper_state = 0.0

        # combine the position and euler angles and self.current_gripper_state to get the state
        state = np.concatenate(
            [dummy_ee_pos, dummy_ee_euler, [self.current_gripper_state]]
        )
        action = np.concatenate(
            [dummy_ee_pos, dummy_ee_euler, [self.current_gripper_action]]
        )

        # Target object position and orientation

        observations = {
            "joint_positions": joint_positions[:7],
            "all_joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_positions_dummy": joint_positions_dummy,
            "ee_pos_quat": dummy_ee_quat,
            "state": state,
            "action": action,
        }

        # gripper_position is for gello integration. It's a shame that it intersects with self.splat_object_name_list convetion
        observations["gripper_position"] = [self.current_gripper_state]

        for i in range(len(self.splatsim_objects)):
            if self.splatsim_objects[i] == self.splatsim_background:
                continue
            (
                object_pos,
                object_quat,
            ) = self.pybullet_client.getBasePositionAndOrientation(
                self.splatsim_objects[i].sim_id
            )
            observations[self.splatsim_objects[i].name + "_position"] = object_pos
            observations[self.splatsim_objects[i].name + "_orientation"] = object_quat

        if len(self.camera_names) > 0:
            self.prep_image_rendering(data=observations)
            with torch.no_grad():
                for camera_name in self.camera_names:
                    observations[camera_name] = self.render_image(camera_name=camera_name)
        for camera_name in ["base_rgb", "wrist_rgb"]:
            if camera_name not in observations:
                observations[camera_name] = None

        return observations

    def randomize_object_pose(self):
        collison_between_objects = True
        while collison_between_objects:
            collison_between_objects = False
            for i in range(len(self.splatsim_objects)):
                if (
                    self.splatsim_objects[i] == self.splatsim_robot
                    or self.splatsim_objects[i] == self.splatsim_background
                ):
                    continue  # only randomize the objects, not the robot
                my_object_env_config = [
                    conf
                    for conf in self.ENV_CONFIG["objects"]
                    if conf["object_name"] == self.splatsim_objects[i].name
                ][0]
                my_object_config = self.splatsim_objects[i].object_config
                if my_object_env_config.get("table_pos", None) is not None:
                    table_pos = my_object_env_config["table_pos"]
                    table_quat = my_object_env_config.get("table_quat", [0, 0, 0, 1])
                    base_position = my_object_config.get("base_position", [[0, 0, 0]])[
                        0
                    ]
                    pos = [
                        table_pos[0] + base_position[0],
                        table_pos[1] + base_position[1],
                        0.0 + base_position[2],
                    ]
                    if self.splatsim_objects[i].sim_id is None:
                        import pdb

                        pdb.set_trace()
                    self.pybullet_client.resetBasePositionAndOrientation(
                        self.splatsim_objects[i].sim_id,
                        pos,
                        table_quat,
                    )
                elif my_object_env_config.get("randomize_pose", True):
                    # randomly reset the object position and orientation
                    # TODO remove hardcoding
                    x = random.uniform(self.TABLE_LIMITS[0][0], self.TABLE_LIMITS[0][1])
                    y = random.uniform(self.TABLE_LIMITS[1][0], self.TABLE_LIMITS[1][1])
                    base_position = my_object_config.get("base_position", [[0, 0, 0]])[
                        0
                    ]
                    pos = [
                        x + base_position[0],
                        y + base_position[1],
                        0.0 + base_position[2],
                    ]
                    # random euler angles for the orientation of the object
                    euler_z = random.uniform(
                        my_object_env_config["rotation_range_z"][0],
                        my_object_env_config["rotation_range_z"][1],
                    )
                    # random quaternion for the orientation of the object
                    # get object name from the object id
                    if len(self.splatsim_objects[i].grasp_configs) > 0:
                        grasp_config = random.choice(
                            self.splatsim_objects[i].grasp_configs
                        )
                    else:
                        grasp_config = {"grasp_pose": [], "object_rot": [0, 0, 0]}
                    self.grasp_poses[i] = grasp_config["grasp_pose"]
                    object_rot = grasp_config["object_rot"]
                    quat = self.pybullet_client.getQuaternionFromEuler(
                        [object_rot[0], object_rot[1], euler_z]
                    )
                    self.pybullet_client.resetBasePositionAndOrientation(
                        self.splatsim_objects[i].sim_id, pos, quat
                    )

            for i in range(len(self.splatsim_objects)):
                if self.splatsim_objects[i].sim_id is None:
                    continue
                for j in range(len(self.splatsim_objects)):
                    if self.splatsim_objects[j].sim_id is None:
                        continue
                    if i != j:
                        collison_between_objects_1 = pairwise_collision(
                            self.splatsim_objects[i].sim_id,
                            self.splatsim_objects[j].sim_id,
                        )
                        if collison_between_objects_1:
                            collison_between_objects = True
                            break

    def randomize_ee_pose(self):
        # generating random initial joint state using random end effector position and orientation
        random_ee_pos, random_ee_quat = self.get_random_ee_pose()

        # joint angles using inverse kinematics
        initial_joint_positions = self.pybullet_client.calculateInverseKinematics(
            self.splatsim_robot.sim_id,
            6,
            random_ee_pos,
            random_ee_quat,
            maxNumIterations=100000,
            residualThreshold=1e-10,
        )

        # reset the joint positions to the initial joint positions
        for i in range(1, self.num_dofs()):
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id, i, initial_joint_positions[i - 1]
            )
        # TODO possibly randomize gripper state here, too
        # Though that might have to edit initial_joint_positions
        return initial_joint_positions

    def get_random_ee_pose(self):
        # random end effector position
        if random.uniform(0, 1) > 0.2:
            random_ee_pos = np.array(
                [
                    random.uniform(
                        self.TABLE_LIMITS[0][0], self.TABLE_LIMITS[0][1] + 0.1
                    ),
                    random.uniform(
                        self.TABLE_LIMITS[1][0] - 0.1, self.TABLE_LIMITS[1][1] + 0.1
                    ),
                    random.uniform(0.25, 0.65),
                ]
            )
        else:
            # get object position
            (
                object_pos,
                object_quat,
            ) = self.pybullet_client.getBasePositionAndOrientation(
                self.splatsim_objects[0].sim_id
            )
            random_x = random.uniform(-0.105, 0.105)
            random_y = random.uniform(-0.105, 0.105)
            random_z = random.uniform(0.25, 0.3)
            random_ee_pos = np.array(
                [
                    object_pos[0] + random_x,
                    object_pos[1] + random_y,
                    object_pos[2] + random_z,
                ]
            )
        # random_ee_pos = np.array([random.uniform(0.2, 0.5), random.uniform(-0.6, 0.6), random.uniform(0.2, 0.65)])

        # get the euler angles from the quaternion
        # get quaternion from euler angles
        random_ee_quat = self.initial_ee_quat

        return random_ee_pos, random_ee_quat

    def follow_paths_and_record(self, all_paths: List[PathSegment]):
        for path_segment in all_paths:
            if isinstance(path_segment, TrajectoryPathSegment):
                self.follow_trajectory_and_record(
                    path=path_segment.path,
                    gripper_pos=path_segment.gripper_pos,
                    gripper_velocity=path_segment.gripper_velocity,
                    threshold=path_segment.threshold,
                    use_current_iters=True,  # Seems to always be default true
                )
            elif isinstance(path_segment, GripperPathSegment):
                if path_segment.target_state == GripperState.OPEN:
                    self.open_gripper_and_record(num_steps=path_segment.num_steps)
                elif path_segment.target_state == GripperState.CLOSE:
                    self.close_gripper_and_record(num_steps=path_segment.num_steps)
                else:
                    raise ValueError(
                        f"Unknown GripperPathSegment.target_state value {path_segment.target_state}"
                    )
            else:
                raise ValueError(f"Unknown path segment type {type(path_segment)}")

    def eval_trajectory_success(self):
        # check the mse of xy position of the objects with the drop location
        for i in range(len(self.splatsim_objects) - 1):
            object_pos, _ = self.pybullet_client.getBasePositionAndOrientation(
                self.splatsim_objects[i].sim_id
            )
            mse = (object_pos[0] - self.drop_ee_pos[0]) ** 2 + (
                object_pos[1] - self.drop_ee_pos[1]
            ) ** 2

            if mse > 0.03:
                print("object not placed correctly")
                return False
        return True

    def open_gripper(self):
        """Open the gripper."""
        self.move_gripper(0.084)
        self.current_gripper_action = GripperState.OPEN  # 1

    def close_gripper(self):
        """Close the gripper."""
        self.move_gripper(0.0)
        self.current_gripper_action = GripperState.CLOSE  # 0

    def plan_execute_record_trajectory(self, initial_joint_positions, joint_signs):
        # Returns whether it was a success

        self.trajectory_length = 0

        # make path+trajectory_count folder
        trajectory_folder = os.path.join(self.path, str(self.trajectory_count).zfill(3))
        print("Generating trajectory in folder:", trajectory_folder)
        os.makedirs(trajectory_folder, exist_ok=True)

        all_paths = self.plan_given_this_state(initial_joint_positions)

        for i in range(100):
            self.pybullet_client.stepSimulation()
            self.open_gripper()
            for k in range(1, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id,
                    k,
                    initial_joint_positions[k - 1] * joint_signs[k - 1],
                )

        if len(all_paths) == 0:
            self.delete_trajectory_folder()
            return False

        self.follow_paths_and_record(all_paths)

        # evaluate the success of the trajectory
        correct_trajectory = self.eval_trajectory_success()

        if correct_trajectory:
            return True
        else:
            self.delete_trajectory_folder()
            return False

    def serve(self) -> None:
        # Prepare for teleport by removing forces
        for i in range(len(self.initial_joint_state)):
            self.pybullet_client.setJointMotorControl2(
                self.splatsim_robot.sim_id,
                i,
                self.pybullet_client.VELOCITY_CONTROL,
                force=0,
            )
        # Reset joint states by teleporting
        for i in range(1, len(self.initial_joint_state)):
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id,
                i,
                self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
            )
        self.initial_link_states = get_curr_link_states(
            self.splatsim_robot.sim_id, self.use_link_centers
        )

        # get end effector position and orientation
        ee_pos, ee_quat = self.get_current_ee_pose()
        self.iniital_ee_quat = ee_quat

        for i in range(1, self.num_dofs()):
            self.pybullet_client.setJointMotorControl2(
                self.splatsim_robot.sim_id,
                i,
                p.VELOCITY_CONTROL,
                targetPosition=self.initial_joint_state[i - 1]
                * self.joint_signs[i - 1],
                force=250,
                maxVelocity=0.2,
            )
        self.close_gripper()

        # get initial ee position and orientation
        self.initial_ee_pos, self.initial_ee_quat = self.get_current_ee_pose()
        # print joint angles
        joint_states = []
        for i in range(1, len(self.initial_joint_state)):
            joint_states.append(
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[0]
            )

        # set to initial joint state
        for i in range(10000):
            for i in range(1, len(self.initial_joint_state)):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id,
                    i,
                    self.initial_joint_state[i - 1] * self.joint_signs[i - 1],
                )
            self.pybullet_client.stepSimulation()

        # start the zmq server
        self._zmq_server_thread.start()

        print("Ready to serve.")

        while True:
            self.serve_loop()

    def serve_loop():
        raise NotImplementedError()

    def plan_given_this_state(self, initial_joint_positions):
        raise NotImplementedError()

    def delete_trajectory_folder(self):
        shutil.rmtree(os.path.join(self.path, str(self.trajectory_count).zfill(3)))

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.splatsim_robot.sim_id)
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.splatsim_robot.sim_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[
                2
            ]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    jointID,
                    self.pybullet_client.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0,
                )
            info = jointInfo(
                jointID,
                jointName,
                jointType,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                controllable,
            )
            self.joints.append(info)

    def setup_gripper(self):
        self.__parse_joint_info__()
        self.gripper_range = [0, 0.085]

        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            # "finger_joint": 1, # TODO: is this left_outer_knuckle_joint?
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        # self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.pybullet_client.createConstraint(
                self.splatsim_robot.sim_id,
                self.mimic_parent_id,
                self.splatsim_robot.sim_id,
                joint_id,
                jointType=self.pybullet_client.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            self.pybullet_client.changeConstraint(
                c, gearRatio=-multiplier, maxForce=10, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def get_link_pose(self, body, link):
        result = self.pybullet_client.getLinkState(body, link)
        return result[4], result[5]

    def move_gripper(self, open_length, velocity=2):
        if not self.use_gripper:
            return
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin(
            (open_length - 0.010) / 0.1143
        )  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(
            self.splatsim_robot.sim_id,
            self.mimic_parent_id,
            self.pybullet_client.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=velocity,
        )

    def get_current_gripper_state(self):
        return self.pybullet_client.getJointState(
            self.splatsim_robot.sim_id, self.mimic_parent_id
        )[0]

    def get_camera_image_from_end_effector(self):

        cam_fov = 90
        near_plane = 0.01
        far_plane = 100
        # Get the pose of the end effector
        end_effector_state = self.pybullet_client.getLinkState(
            self.splatsim_robot.sim_id, 8
        )
        end_effector_pos = end_effector_state[4]
        end_effector_orn = end_effector_state[5]

        # Convert the quaternion orientation to a rotation matrix
        rot_matrix = self.pybullet_client.getMatrixFromQuaternion(end_effector_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Define the camera position relative to the end effector
        camera_position = np.array([0, 0, 0.1])  # Adjust as needed
        camera_position_world = end_effector_pos + rot_matrix.dot(camera_position)

        # Define the camera view direction
        camera_target_position = np.array([0, 0, 1])  # Adjust as needed
        camera_target_position_world = end_effector_pos + rot_matrix.dot(
            camera_target_position
        )

        # Compute the view matrix
        view_matrix = self.pybullet_client.computeViewMatrix(
            camera_position_world, camera_target_position_world, [0, 0, 1]
        )

        # Compute the projection matrix
        projection_matrix = self.pybullet_client.computeProjectionMatrixFOV(
            cam_fov, 1.0, near_plane, far_plane
        )

        # Get the camera image
        (
            width,
            height,
            rgb_img,
            depth_img,
            seg_img,
        ) = self.pybullet_client.getCameraImage(
            320,
            240,
            view_matrix,
            projection_matrix,
            flags=self.pybullet_client.ER_NO_SEGMENTATION_MASK,
        )

        return rgb_img

    def pre_grasp_to_grasp(self, transformation_matrix):

        # get the end effector position and orientation according to self.apple_grasp_pose
        ee_transformation = transformation_matrix

        # get approach vector
        approach_vector = ee_transformation[:3, :3][:, 1] / np.linalg.norm(
            ee_transformation[:3, :3][:, 1]
        )

        # check the angle of the approach vector with the -ve z axis
        angle = np.arccos(np.dot(approach_vector, np.array([0, 0, -1])))
        if angle > 0.1 or angle < -0.1:
            print("angle is greater than 0.1")
            return None, None

        # check the angle of the approach vector with the y axis
        angle = np.arccos(np.dot(approach_vector, np.array([0, 1, 0])))
        if np.abs(angle - np.pi / 2) > 0.1:
            print("angle is greater than 0.1")
            return None, None

        # check the angle of the approach vector with the x axis
        angle = np.arccos(np.dot(approach_vector, np.array([1, 0, 0])))
        print(angle)

        # get the pre-grasp position
        pre_grasp_pos = ee_transformation[:3, 3] - 0.1 * approach_vector

        # pre-grasp transformation
        pre_grasp_transformation = np.eye(4)
        pre_grasp_transformation[:3, 3] = pre_grasp_pos
        pre_grasp_transformation[:3, :3] = ee_transformation[:3, :3]

        # get joint positions going from pre-grasp to grasp
        path = []
        for i in range(11):
            ee_pos = pre_grasp_pos + 0.01 * i * approach_vector
            ee_quat = self.pybullet_client.getQuaternionFromEuler(
                rotation_matrix_to_euler_angles(ee_transformation[:3, :3])
            )
            joint_positions = self.pybullet_client.calculateInverseKinematics(
                self.splatsim_robot.sim_id,
                6,
                ee_pos,
                ee_quat,
                maxNumIterations=100000,
                residualThreshold=1e-10,
            )
            path.append(joint_positions)

        return path, pre_grasp_transformation

    def close_gripper_and_record(self, num_steps=160):
        for i in range(num_steps):
            self.close_gripper()
            self.pybullet_client.stepSimulation()

            observations = self.get_observations()
            # save the observations in the correct format zfill(5)
            if i % 20 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

    def open_gripper_and_record(self, num_steps=160):
        for i in range(num_steps):
            self.open_gripper()
            self.pybullet_client.stepSimulation()

            observations = self.get_observations()
            # save the observations
            if i % 20 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

    def follow_trajectory_and_record(
        self,
        path,
        gripper_pos,
        use_current_iters=True,
        gripper_velocity=2,
        threshold=1e-2,
    ):
        # Note: This function also saves observations
        k = 0
        loop_iters = 0
        current_iters = 0
        while k < len(path):
            error = 0

            for j in range(1, self.num_dofs()):
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=path[k][j - 1],
                    force=250,
                    maxVelocity=0.2,
                )

            # get current joint positions
            joint_states = []
            for i in range(1, self.num_dofs()):
                joint_states.append(
                    self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[0]
                )

            error = np.linalg.norm(np.array(joint_states) - path[k][:6])

            self.move_gripper((1 - gripper_pos) * 0.084, velocity=gripper_velocity)

            if error < threshold:
                k += 1
                current_iters = 0

            current_iters += 1

            if use_current_iters:
                if current_iters > 200:
                    k += 1
                    current_iters = 0

            self.current_gripper_action = gripper_pos
            if loop_iters % 50 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                # get observations
                observations = self.get_observations()
                # save the observations with trajectory length as pickle file in format 0000x.pkl
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

            if loop_iters > 10000:
                break

            self.pybullet_client.stepSimulation()

            loop_iters += 1

        if not self.skip_recording_first:

            self.trajectory_length += 1
            # get observations
            observations = self.get_observations()
            # save the observations with trajectory length as pickle file in format 0000x.pkl
            with open(
                os.path.join(
                    self.path,
                    str(self.trajectory_count).zfill(3),
                    str(self.trajectory_length).zfill(5) + ".pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(observations, f)

    def shutdown(self):
        # Say to shut down
        pass
