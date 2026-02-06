import pickle
import threading
import time
from typing import Any, Dict, Optional, List, Tuple
import gymnasium
from gymnasium import spaces
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

# import urdf_models.models_data as md
import pybullet as p
from pybullet_planning.interfaces.robots.collision import pairwise_collision, pairwise_link_collision

from pybullet_planning import RED, BLUE, GREEN
from pybullet_planning import Pose
from pybullet_planning import set_pose
from pybullet_planning import create_box
import pybullet_data
from splatsim.utils.robot_splat_render_utils import (
    get_segmented_indices,
    transform_means,
    get_transformation_list,
    transform_object,
    get_curr_link_states,
    crop_splat,
    create_cuboid_gaussians,
    SplatSimObject,
    ArticulationConfig,
)
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams, Namespace
from gaussian_splatting.scene import Scene

from splatsim.utils.transform_utils import rotation_matrix_to_euler_angles
from splatsim.utils.image_utils import letterbox
from splatsim.utils.trajectory_generation import TrajectoryGenerator
from splatsim.utils.splatsim_gui import SplatSimGui
from splatsim.configs import (
    DebugModes,
    EnvConfig,
    CuboidObjectConfig,
    SplatObjectConfig,
)

from pathlib import Path

# Get the splatsim package root directory
SPLATSIM_ROOT = Path(__file__).resolve().parent.parent.parent


def resize_image(img: np.ndarray, output_size: Tuple[int, int], mode: str = "letterbox") -> np.ndarray:
    """Resize image to output_size using the specified mode.

    Args:
        img: Input image in CHW format (channels, height, width), float32 in [0, 1]
        output_size: Target (height, width)
        mode: Resize mode - "letterbox" or "stretch"

    Returns:
        Resized image in CHW format, float32 in [0, 1]
    """
    if mode == "letterbox":
        return letterbox(img, output_size=output_size)
    elif mode == "stretch":
        # Convert from CHW to HWC for cv2
        img_hwc = np.transpose(img, (1, 2, 0))
        # Resize using cv2 (stretches to fill, ignoring aspect ratio)
        img_resized = cv2.resize(img_hwc, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
        # Convert back to CHW
        return np.transpose(img_resized, (2, 0, 1))
    else:
        raise ValueError(f"Unknown image resize mode: {mode}. Use 'letterbox' or 'stretch'.")


def resolve_splatsim_path(path: str) -> str:
    """Resolve a path, making relative paths relative to SPLATSIM_ROOT.

    This allows configs to use relative paths like './splatsim/...' that work
    regardless of the current working directory.
    """
    if os.path.isabs(path):
        return path
    resolved = SPLATSIM_ROOT / path
    return str(resolved)


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
                elif method == "get_ee_pos":
                    result = self._robot.get_current_ee_pose()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "teleport_joint_state":
                    result = self._robot.teleport_joint_state(
                        self._robot.splatsim_robot, **args
                    )
                elif method == "set_object_pose":
                    result = self._robot.set_object_pose(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "create_object":
                    splatsim_object = self._robot.create_object(**args)
                    result = None
                elif method == "delete_object":
                    result = self._robot.delete_object(**args)
                elif method == "clear_temp_objects":
                    result = self._robot.clear_temp_objects()
                elif method == "disable_rendering":
                    result = self._robot.disable_rendering()
                elif method == "enable_rendering":
                    result = self._robot.enable_rendering()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass
                # Timeout occurred, check if the stop event is set
            except (zmq.error.ContextTerminated, zmq.error.ZMQError):
                break  # Socket/context was closed during shutdown

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._socket.close()
        except zmq.error.ZMQError:
            pass
        try:
            self._context.term()
        except zmq.error.ZMQError:
            pass


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


@dataclass
class SplatSimCamera:
    camera: Optional[Camera]
    pipeline: Optional[PipelineParams]
    background: Optional[torch.tensor]
    tracked_link_index: Optional[str] = None


class PybulletRobotServerBase:
    MAX_TRAJECTORY_COUNT = 500

    @property
    def TABLE_LIMITS(self):
        """Compute table limits from ENV_CONFIG.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max)) computed from the table object in ENV_CONFIG.

        Raises:
            ValueError: If no table object is defined in ENV_CONFIG.
        """
        for obj in self.ENV_CONFIG.objects:
            if obj.object_name == "table" and isinstance(obj, CuboidObjectConfig):
                position = obj.position
                size = obj.size
                x_center, y_center = position[0], position[1]
                x_half, y_half = size[0] / 2, size[1] / 2
                return ((x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half))
        raise ValueError("TABLE_LIMITS requested but no 'table' object is defined in ENV_CONFIG['objects'].")

    # Gym environment constants
    _max_episode_steps = 400 # Can overwrite this in a child class
    # 240Hz physics / 30Hz control
    _physics_steps_per_action = 8  # Can overwrite this in a child class

    # This is the default splat name. Overwrite it in a child class of PybulletRobotServerBase
    background_splat_name = None

    # Enum for serve modes
    class SERVE_MODES(enum.Enum):
        GENERATE_DEMOS = "generate_demos"

        INTERACTIVE = "interactive"

        GENERATE_TRAJECTORIES = "generate_trajectories"
        GENERATE_TRAJECTORIES_IDLE = "generate_trajectories_idle"

    # Alias to the shared DebugModes enum for backwards compatibility
    DEBUG_MODES = DebugModes

    lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
    upper_limits = [np.pi, 0, np.pi, np.pi, np.pi, np.pi]

    ENV_CONFIG: EnvConfig # To be set in a subclass

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        use_gripper: bool = True,
        serve_mode: SERVE_MODES = SERVE_MODES.GENERATE_DEMOS,
        robot_name: str = "robot_iphone",
        camera_names: List[str] = ["base_rgb"],
        cam_i: int = 254,
        image_width: int = 640,
        image_height: Optional[int] = None,
        debug_mode: DEBUG_MODES = DEBUG_MODES.OFF,
        trajectory_gen_config: Optional[dict] = None,
        image_resize_mode: str = "letterbox",
    ):
        self.serve_mode = serve_mode
        self.robot_name = robot_name
        self.camera_names = camera_names
        self.cam_i = cam_i
        self.image_width = image_width
        self.image_height = image_height
        self.use_gripper = use_gripper
        if isinstance(debug_mode, str):
            debug_mode = self.DEBUG_MODES(debug_mode)
        assert debug_mode in self.DEBUG_MODES, f"debug_mode must be one of {list(self.DEBUG_MODES)}, got {debug_mode}"
        self.debug_mode = debug_mode
        # Image resize mode: "letterbox" (preserve aspect ratio, pad) or "stretch" (fill entire area)
        self.image_resize_mode = image_resize_mode

        # load labels.npy
        self.robot_labels = np.load(
            str(SPLATSIM_ROOT / "data" / "labels_path" / f"{self.robot_name}_labels.npy")
        )
        self.robot_labels = torch.from_numpy(self.robot_labels).to(device="cuda").long()

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        print(f"Listening on {host}:{port}")

        # Queue for passing display requests from ZMQ thread to main thread
        self._display_queue: queue.Queue = queue.Queue()
        self._main_thread_id = threading.current_thread().ident

        # Populate this on the fly if it's needed
        self.base_cuboid_gaussians = None

        ## add stage
        self.stage = 0

        self.do_render_from_splat = True

        # Placeholder object for rendering purposes
        self.scene_gaussian = GaussianModel(3)

        self.object_config_path = str(SPLATSIM_ROOT / "configs" / "object_configs" / "objects.yaml")
        self.grasp_poses = {}

        self.pybullet_client = p
        self.pybullet_client.connect(p.GUI)
        self.pybullet_client.setAdditionalSearchPath(
            str(SPLATSIM_ROOT.parent / "submodules" / "pybullet-playground-wrapper" / "pybullet_playground" / "urdf" / "pybullet_ur5_gripper" / "urdf")
        )
        # Enable GUI for trajectory generation controls (sliders/buttons)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        # set time step
        self.pybullet_client.setTimeStep(1 / 240)
        # add gravity
        self.pybullet_client.setGravity(0, 0, -9.81)

        # For plane.urdf
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())


        with open(self.object_config_path, "r") as file:
            self.object_config = yaml.safe_load(file)

        # TODO this mock class fails for the apple on plate task, etc
        class MockModelsLib:
            model_name_list = {}

        self.models_lib = MockModelsLib  # md.model_lib()

        self.splatsim_objects: List[SplatSimObject] = []
        self.splatsim_robot: SplatSimObject = self.create_object(
            object_name="robot", splat_object_name=self.robot_name
        )

        # Initialize trajectory generation config and generator
        # TrajectoryGenerator has the default values; we only merge subclass and CLI overrides
        cli_config = trajectory_gen_config or {}
        trajectory_gen_config = {
            # For example, from a child class
            **getattr(self, 'TRAJECTORY_GEN_CONFIG', {}),
            **cli_config
        }
        self.trajectory_generator = TrajectoryGenerator(
            pybullet_client=self.pybullet_client,
            robot_id=self.splatsim_robot.sim_id,
            joint_indices=list(range(1, 7)), # excludes gripper
            env_config_name=self.ENV_CONFIG.name,
            get_ee_link_fn=lambda: self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id) - 1,
            splatsim_objects=self.splatsim_objects,
            wrist_camera_link_name=self.splatsim_robot.object_config.get("wrist_camera_link_name", None),
            **trajectory_gen_config,
        )

        self._setup_interactive_gui()

        if self.debug_mode == self.DEBUG_MODES.NO_BACKGROUND:
            print("[Debug mode] no_background, using robot as background")
            self.background_splat_name = self.robot_name
            self.splatsim_background = self.splatsim_robot
        else:
            # The background uses the robot's full splat, but crops out the robot
            self.splatsim_background: SplatSimObject = self.create_object(
                object_name="background",
                splat_object_name=self.background_splat_name,
                keep_within_aabb=False,
                load_urdf=False,
                can_articulate=False,
            )
        if self.debug_mode == self.DEBUG_MODES.ROTATE_BASE_CAM:
            print(
                "[Debug Mode] Setting base_rgb camera to be adjustable using pybullet GUI debug camera"
            )

        self.skip_recording_first = 0

        for i in range(self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)):
            info = self.pybullet_client.getJointInfo(self.splatsim_robot.sim_id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id

        for object_cfg in self.ENV_CONFIG.objects:
            object_name = object_cfg.object_name
            splat_object_name = getattr(object_cfg, "splat_object_name", None)
            nested_object_config = getattr(object_cfg, "object_config", None) or {}
            object_config = {
                **self.object_config.get(object_name, {}),
                **nested_object_config,
                **object_cfg.to_dict(),
            }
            grasp_config = getattr(object_cfg, "grasp_config", []) or []

            # Already adds the splatsim_object to self.splatsim_objects
            splatsim_object = self.create_object(
                object_name,
                splat_object_name=splat_object_name,
                object_config=object_config,
            )

            splatsim_object.grasp_configs = grasp_config

        self.randomize_object_pose()

        # TODO put all trajectory saving into TrajectoryGenerator
        # trajectory path
        with open(resolve_splatsim_path("configs/folder_configs.yaml"), "r") as f:
            folder_config = yaml.safe_load(f)
        self.path = folder_config["traj_folder"]
        # get no of folders in the path
        if os.path.exists(self.path):
            self.trajectory_count = len(os.listdir(self.path))
        else:
            self.trajectory_count = 0

        # Always set up the base camera because wrist_rgb is initialized from it
        self.base_camera = self.setup_camera_from_dataset(
            self.splatsim_background.object_config, cam_i=self.cam_i, use_train=True
        )

        if "wrist_rgb" in self.camera_names:
            # TODO Hardcoding a splat dataset / recovered camera from a GoPro
            self.wrist_camera = self.setup_camera_from_dataset(
                self.object_config["bwa_open_space"], cam_i=3, use_train=True
            )
        else:
            # Make a dummy camera
            self.wrist_camera = SplatSimCamera(
                camera=None,
                pipeline=None,
                background=None,
            )

        # Add the index of the wrist_camera_link to the wrist camera if it's available
        if "wrist_camera_link_name" in self.splatsim_robot.object_config:
            wrist_camera_link_name = self.splatsim_robot.object_config[
                "wrist_camera_link_name"
            ]
            num_joints = p.getNumJoints(self.splatsim_robot.sim_id)
            for i in range(num_joints):
                info = p.getJointInfo(self.splatsim_robot.sim_id, i)
                if info[12].decode("utf-8") == wrist_camera_link_name:
                    self.wrist_camera.tracked_link_index = i
                    break
            if self.wrist_camera.tracked_link_index is None:
                raise ValueError(
                    f"Cannot find wrist camera link name {wrist_camera_link_name}"
                )
        else:
            raise ValueError(
                f"wrist_camera_link_name attribute not defined in object config of robot {self.robot_name}, yet wrist camera was requested"
            )

        # current gripper state
        self.current_gripper_action = GripperState.OPEN
        self.teleport_joint_state(self.splatsim_robot, self.splatsim_robot.articulation_config.initial_joint_positions)

        # Gym-related state
        self._step_count = 0
        self._episode_started = False

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

    def load_urdf(self, splatsim_obj):
        # This must be called after the gaussians are finalized
        # ex: after the gaussians are transformed to be in the simulator's coordinate frame
        use_fixed_base = splatsim_obj.object_config.get("use_fixed_base", False)
        global_scaling = splatsim_obj.object_config.get("global_scaling", 1)
        is_articulated = splatsim_obj.object_config.get("is_articulated", False)
        base_position = splatsim_obj.object_config.get("base_position", [[0, 0, 0]])[0]

        # Find possible URDF config
        if splatsim_obj.name in self.models_lib.model_name_list:
            urdf_path = self.models_lib[splatsim_obj.splat_name]
        elif "urdf_path" in splatsim_obj.object_config:
            urdf_path = resolve_splatsim_path(splatsim_obj.object_config["urdf_path"][0])
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        else:
            urdf_path = None

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
        elif splatsim_obj.object_config.get("object_type", None) is not None:
            # Find primitive shape config
            object_type = splatsim_obj.object_config["object_type"]
            if object_type == "cuboid":
                # position, orientation, size
                # TODO orientation
                lx, ly, lz = splatsim_obj.object_config["size"]
                position = splatsim_obj.object_config["position"]
                color_rgb = splatsim_obj.object_config.get("color_rgb", BLUE)

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

                # TODO check if this box is created with (0,0,0) at the center of the box
                object_loaded = create_box(lx, ly, lz, color=color_rgb)
                set_pose(object_loaded, Pose(point=position))
                # TODO set orientation
                mass = (
                    0
                    if use_fixed_base or "mass" not in splatsim_obj.object_config
                    else splatsim_obj.object_config["mass"]
                )
                self.pybullet_client.changeDynamics(object_loaded, -1, mass=mass)
            else:
                raise ValueError(f"Cannot create unknown object type {object_type}")
        elif (
            len(splatsim_obj.gaussians._xyz) > 0
        ):  # The gaussian splat was loaded, but there's no urdf
            # Create a box that covers the gaussian means in the gaussian splat. This is an approximation
            lx = max(splatsim_obj.gaussians._xyz[:, 0]) - min(
                splatsim_obj.gaussians._xyz[:, 0]
            )
            ly = max(splatsim_obj.gaussians._xyz[:, 1]) - min(
                splatsim_obj.gaussians._xyz[:, 1]
            )
            lz = max(splatsim_obj.gaussians._xyz[:, 2]) - min(
                splatsim_obj.gaussians._xyz[:, 2]
            )

            position = splatsim_obj.object_config.get("position", [0, 0, 0])

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
                if use_fixed_base or "mass" not in splatsim_obj.object_config
                else splatsim_obj.object_config["mass"]
            )
            self.pybullet_client.changeDynamics(object_loaded, -1, mass=mass)
        else:
            raise ValueError(
                f"Could not parse object config for object name {splatsim_obj.name}"
            )

        splatsim_obj.sim_id = object_loaded
        splatsim_obj.mass = mass

        # Set friction
        self.pybullet_client.changeDynamics(
            splatsim_obj.sim_id, -1, lateralFriction=1.5
        )
        self.pybullet_client.changeDynamics(
            splatsim_obj.sim_id, -1, rollingFriction=0
        )

    def load_gaussian_splat(self, splatsim_obj):
        # Most of these representations are in the splat frame, so we need to transform to simulator frame
        need_transform_to_simulator_frame = True

        if "ply_path" in splatsim_obj.object_config:
            ply_path = splatsim_obj.object_config["ply_path"]
            splatsim_obj.gaussians.load_ply(ply_path)
        elif (
            "source_path" in splatsim_obj.object_config
            and "model_path" in splatsim_obj.object_config
        ):
            source_path = resolve_splatsim_path(splatsim_obj.object_config["source_path"])
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source path not found: {source_path}")

            model_path = resolve_splatsim_path(splatsim_obj.object_config["model_path"])
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
                splatsim_obj.gaussians,
                load_iteration=-1,
                shuffle=False,
                resolution_scales=[],
                train_cam_indices=[],
                test_cam_indices=[],
            )
            del scene
        elif splatsim_obj.object_config.get("object_type", None) == "cuboid":
            lx, ly, lz = splatsim_obj.object_config["size"]
            # Default to brown color for rendering
            color_rgb = splatsim_obj.object_config.get("color_rgb", (87, 51, 33)) 
            cuboid_params = create_cuboid_gaussians(
                side_lengths=(lx, ly, lz),
                spacing=0.005,
                color_rgb=color_rgb,
            )
            splatsim_obj.gaussians._xyz = cuboid_params["_xyz"]
            splatsim_obj.gaussians._rotation = cuboid_params["_rotation"]
            splatsim_obj.gaussians._opacity = cuboid_params["_opacity"]
            splatsim_obj.gaussians._features_rest = cuboid_params["_features_rest"]
            splatsim_obj.gaussians._features_dc = cuboid_params["_features_dc"]
            splatsim_obj.gaussians._scaling = cuboid_params["_scaling"]
            need_transform_to_simulator_frame = False
        else:
            import pdb

            pdb.set_trace()
            raise ValueError("Could not load gaussian splat")

        # Disable gradients on this gaussian splat b/c we're not optimizing
        splatsim_obj.gaussians._xyz.requires_grad = False
        splatsim_obj.gaussians._rotation.requires_grad = False
        splatsim_obj.gaussians._opacity.requires_grad = False
        splatsim_obj.gaussians._features_rest.requires_grad = False
        splatsim_obj.gaussians._features_dc.requires_grad = False
        splatsim_obj.gaussians._scaling.requires_grad = False

        # Transform the xyz, rotation, and shs features to the canonical frame (the world frame for the simulator)
        # We will work in the coordinate frame of the simulator from now on
        if need_transform_to_simulator_frame:
            Trans_canonical = (
                torch.from_numpy(
                    np.array(splatsim_obj.object_config["transformation"]["matrix"])
                )
                .to(device=splatsim_obj.gaussians.get_xyz.device)
                .float()
            )  # shape (4, 4)
            _ = transform_object(
                splatsim_obj=splatsim_obj,
                transform=Trans_canonical,
                # This transform_object is to go from splat frame to simulator frame. the transformation matrix already accounted for the base position
                use_base_position=False,  # False,
                inplace=True,
            )

        return splatsim_obj

    def delete_object(self, object_name):
        index = [splatsim_obj.name for splatsim_obj in self.splatsim_objects].index(
            object_name
        )
        splatsim_obj = self.splatsim_objects.pop(index)

        # Explicitly delete some values
        del splatsim_obj.gaussians
        if splatsim_obj.sim_id is not None:
            p.removeBody(splatsim_obj.sim_id)

        # Invalidate scene gaussian buffers so they get reinitialized without the deleted object
        self._invalidate_scene_gaussian_buffers()

    def clear_temp_objects(self):
        non_temp_object_names = [
            self.splatsim_robot.name,
            self.splatsim_background.name,
        ] + [obj_cfg.object_name for obj_cfg in self.ENV_CONFIG.objects]
        all_object_names = [splatsim_obj.name for splatsim_obj in self.splatsim_objects]
        deleted_obj_names = []
        for obj_name in all_object_names:
            if obj_name not in non_temp_object_names:
                self.delete_object(obj_name)
                deleted_obj_names.append(obj_name)
        return deleted_obj_names

    def create_object(
        self,
        object_name,
        object_config={},
        splat_object_name=None,
        keep_within_aabb=True,
        load_urdf=True,
        can_articulate=True,
    ):
        load_splat = object_config.get("load_splat", True)
        if len(object_config) == 0 and splat_object_name is None:
            splat_object_name = object_name  # TODO when is this wrong?

        # Find object config
        if splat_object_name is not None:
            object_config = {
                **self.object_config.get(splat_object_name, {}),
                **object_config,
            }
        elif len(object_config) == 0:
            print("WARNING: No object config found for ", splat_object_name)

        is_articulated = can_articulate and object_config.get("is_articulated", False)

        splatsim_obj = SplatSimObject(
            name=object_name,
            splat_name=splat_object_name,
            gaussians=GaussianModel(3),
            is_articulated=is_articulated,
            # set sim_id and mass later
            object_config=object_config,
        )

        if load_splat:
            self.load_gaussian_splat(splatsim_obj)
        else:
            splatsim_obj.gaussians = None

        if load_urdf:
            self.load_urdf(splatsim_obj)
        else:
            splatsim_obj.sim_id = None
            splatsim_obj.mass = 0

        crop_splat(splatsim_obj, keep_within_aabb=keep_within_aabb)

        if is_articulated:
            assert object_name == "robot", "Only the robot can be articulated for now"
            if object_name == "robot":
                self.splatsim_robot = splatsim_obj
                if self.use_gripper:
                    self.setup_gripper()
                # else:
                #     self.setup_spatula()
                #     pass
                self.open_gripper()

            initial_joint_positions = splatsim_obj.object_config["joint_states"][0]
            # remove world joint
            initial_joint_positions = initial_joint_positions[1:]
            num_joints = self.pybullet_client.getNumJoints(splatsim_obj.sim_id)
            if len(initial_joint_positions) > num_joints:
                print(
                    f"Warning: Provided initial joint positions ({len(initial_joint_positions)}) exceed the number of joints ({num_joints}). Truncating to {num_joints} positions."
                )
                initial_joint_positions = initial_joint_positions[:num_joints]

            segmentation_labels = np.load(
                resolve_splatsim_path("./data/labels_path/" + splatsim_obj.splat_name + "_labels.npy")
            )
            segmentation_labels = (
                torch.from_numpy(segmentation_labels)
                .to(device=splatsim_obj.gaussians._xyz.device)
                .long()
            )

            joint_signs = [1] * len(initial_joint_positions)
            # TODO doesn't the ur5 have some joint signs inverted based on the gello repo documentation / readme?
            # joint_signs[2] = -1
            splatsim_obj.articulation_config = ArticulationConfig(
                initial_joint_positions=initial_joint_positions,
                joint_signs=joint_signs,  # placeholder values; integration with gello
                segmentation_labels=segmentation_labels,
            )

            segmented_list = get_segmented_indices(
                splatsim_obj=splatsim_obj,
            )
            splatsim_obj.articulation_config.segmented_list = segmented_list

            # Use the config to find these values
            self.teleport_joint_state(splatsim_obj, initial_joint_positions)
            # Let the gripper move (it isn't teleporting rn)
            for _ in range(100):
                self.pybullet_client.stepSimulation()
            initial_link_poses = get_curr_link_states(splatsim_obj.sim_id)
            splatsim_obj.articulation_config.initial_link_poses = initial_link_poses

        self.splatsim_objects.append(splatsim_obj)

        # Invalidate scene gaussian buffers so they get reinitialized with the new object
        self._invalidate_scene_gaussian_buffers()

        return splatsim_obj

    def set_object_pose(
        self,
        object_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
        use_gravity: bool = True,
    ) -> None:
        """Set the pose of an object in the simulation."""
        # if object_name not in [
        #     splatsim_obj.splat_name for splatsim_obj in self.splatsim_objects
        # ]:
        #     print(f"Object {object_name} not found in splat_object_name_list.")
        #     return

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

    def teleport_joint_state(
        self, splatsim_obj: SplatSimObject, joint_state: List[float]
    ) -> None:
        """Set the joint states of an articulated object in the simulation and hold position."""
        if not splatsim_obj.is_articulated:
            raise ValueError(f"Object {splatsim_obj.name} is not articulated.")

        if splatsim_obj.sim_id is None:
            raise ValueError(
                "Cannot set joint states of object not represented in pybullet (ex: has urdf)"
            )

        num_joints = self.pybullet_client.getNumJoints(splatsim_obj.sim_id)
        if len(joint_state) > num_joints - 1:
            raise ValueError(
                f"Expected at most {num_joints - 1} joint states, got {len(joint_state)}."
            )

        for i in range(1, min(len(joint_state), num_joints)):
            target_position = (
                joint_state[i - 1]
                * splatsim_obj.articulation_config.joint_signs[i - 1]
            )
            # Teleport joint to target position
            self.pybullet_client.resetJointState(
                splatsim_obj.sim_id,
                i,
                target_position,
            )
            # Set position control to hold the joint at target position
            self.pybullet_client.setJointMotorControl2(
                splatsim_obj.sim_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=150,
                maxVelocity=3.14,
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
                # Set a more realistic force for the robot
                force=150,
                maxVelocity=3.14,
            )

        if self.use_gripper:
            self.move_gripper((1 - joint_state[-1]) * 0.085)
            self.current_gripper_action = joint_state[-1]

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def disable_rendering(self):
        self.do_render_from_splat = False

    def enable_rendering(self):
        self.do_render_from_splat = True

    def get_wrist_camera_transform(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.wrist_camera.tracked_link_index is None:
            print("WARNING: No wrist camera index found")
            return None

        # Get the pose of the wrist_camera_link
        link_state = p.getLinkState(
            self.splatsim_robot.sim_id,
            self.wrist_camera.tracked_link_index,
            computeForwardKinematics=True,
        )

        # Original camera-to-world transform
        T_cw = np.array(link_state[0]).astype(np.float32)  # xyz position in world frame

        quat = link_state[1]
        R_cw = (
            np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3).astype(np.float32)
        )

        return T_cw, R_cw

    def get_wrist_camera(self):
        transform_pair = self.get_wrist_camera_transform()
        if transform_pair is None:
            return None
        T_cw, R_cw = transform_pair

        T_wc = -R_cw.T @ T_cw

        # TODO is this needed?
        # scale = self.base_camera.scale if self.base_camera is not None else 1.0
        resolution = (
            self.base_camera.camera.image_width,
            self.base_camera.camera.image_height,
        )

        # Use actual camera resolution and FOV from COLMAP calibration
        # COLMAP camera model: OPENCV_FISHEYE
        # COLMAP params: width=960, height=540, fx=1152, fy=1152, cx=480, cy=270
        # Distortion k1-k4 = [0,0,0,0] (no distortion, lucky!)
        # Computed FOV: FoVx=0.789582 rad (45.24°), FoVy=0.460439 rad (26.38°)
        # resolution = (960, 540)
        # resolution = (960, 540)
        # fovx = 0.789582  # radians
        # fovy = 0.460439  # radians

        colmap_id = 0
        uid = 0
        depth_params = None
        invdepthmap = None
        image_name = "wrist_camera"
        image = torch.zeros((3, resolution[0], resolution[1]), dtype=torch.float32)

        fov_scale_x = 1
        fovx = self.base_camera.camera.FoVx * fov_scale_x
        fovy = 2 * np.atan(np.tan(self.base_camera.camera.FoVy / 2) * fov_scale_x)

        camera = Camera(
            resolution,
            colmap_id,
            R_cw,
            T_wc,
            fovx,
            fovy,
            depth_params,
            to_pil_image(image),
            invdepthmap,
            image_name,
            uid,
            scale=1,  # scale
        )

        splatsim_camera = SplatSimCamera(
            camera=camera,
            pipeline=self.base_camera.pipeline,
            background=self.base_camera.background,
        )

        return splatsim_camera

    def _init_scene_gaussian_buffers(self):
        """Initialize pre-allocated buffers for scene_gaussian to avoid fragmentation.

        Called automatically on first render and after objects are created/deleted.
        """
        # Calculate total number of gaussians across all rendered objects
        total_gaussians = 0
        self._scene_gaussian_offsets = {}  # Map object name to (start, end) indices

        for splatsim_obj in self.splatsim_objects:
            if splatsim_obj.gaussians is None:
                continue
            if (splatsim_obj.object_config.get("is_in_scene_splat", False)
                and not splatsim_obj.object_config.get("randomize_pose", True)):
                continue
            if self.debug_mode == DebugModes.NO_BACKGROUND and splatsim_obj == self.splatsim_background:
                continue

            n_gaussians = splatsim_obj.gaussians.get_xyz.shape[0]
            self._scene_gaussian_offsets[splatsim_obj.name] = (total_gaussians, total_gaussians + n_gaussians)
            total_gaussians += n_gaussians

        # Pre-allocate scene gaussian buffers
        device = 'cuda'
        self._scene_gaussian_buffers_initialized = True
        self.scene_gaussian._xyz = torch.zeros(total_gaussians, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._rotation = torch.zeros(total_gaussians, 4, device=device, dtype=torch.float32)
        self.scene_gaussian._opacity = torch.zeros(total_gaussians, 1, device=device, dtype=torch.float32)
        self.scene_gaussian._scaling = torch.zeros(total_gaussians, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._features_dc = torch.zeros(total_gaussians, 1, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._features_rest = torch.zeros(total_gaussians, 15, 3, device=device, dtype=torch.float32)

    def _invalidate_scene_gaussian_buffers(self):
        """Mark scene gaussian buffers as needing reinitialization.

        Call this when objects are created or deleted.
        """
        self._scene_gaussian_buffers_initialized = False

    def prep_image_rendering(self, data) -> Dict[str, np.ndarray]:
        # Initialize buffers on first call
        if not getattr(self, '_scene_gaussian_buffers_initialized', False):
            self._init_scene_gaussian_buffers()

        # Transform each object splat to be in the right pose and copy into pre-allocated buffers
        with torch.no_grad():
            for i in range(len(self.splatsim_objects)):
                splatsim_obj = self.splatsim_objects[i]
                if splatsim_obj.gaussians is None:
                    continue

                # Skip objects not in the offset map (filtered during init)
                if splatsim_obj.name not in self._scene_gaussian_offsets:
                    continue

                start_idx, end_idx = self._scene_gaussian_offsets[splatsim_obj.name]

                # Build output_slices dict for zero-copy writes directly into scene_gaussian buffers
                output_slices = {
                    '_xyz': self.scene_gaussian._xyz[start_idx:end_idx],
                    '_rotation': self.scene_gaussian._rotation[start_idx:end_idx],
                    '_opacity': self.scene_gaussian._opacity[start_idx:end_idx],
                    '_scaling': self.scene_gaussian._scaling[start_idx:end_idx],
                    '_features_dc': self.scene_gaussian._features_dc[start_idx:end_idx],
                    '_features_rest': self.scene_gaussian._features_rest[start_idx:end_idx],
                }

                if splatsim_obj.is_articulated:
                    assert (
                        splatsim_obj == self.splatsim_robot
                    ), "Other articulated objects are not implemented yet"

                    # Gets transformations for all links of the robot based on the current simulation
                    transformations_list = get_transformation_list(splatsim_obj)

                    # TODO generalize this to "every articulated object" instead of just the robot
                    transform_means(
                        splatsim_obj=splatsim_obj,
                        transformations_list=transformations_list,
                        use_base_position=True,
                        inplace=False,
                        output_slices=output_slices,
                    )

                else:
                    if splatsim_obj.sim_id is not None:
                        # Reuse cached tensors and copy data to avoid allocating new GPU memory each step
                        if 'position_tensor' not in splatsim_obj._cache:
                            splatsim_obj._cache['position_tensor'] = torch.zeros(3, device="cuda", dtype=torch.float32)
                            splatsim_obj._cache['rotation_tensor'] = torch.zeros(4, device="cuda", dtype=torch.float32)
                            splatsim_obj._cache['rotation_rolled'] = torch.zeros(4, device="cuda", dtype=torch.float32)
                        cur_object_position = splatsim_obj._cache['position_tensor']
                        cur_object_position.copy_(torch.as_tensor(data[splatsim_obj.name + "_position"], device="cuda"))
                        rot_raw = splatsim_obj._cache['rotation_tensor']
                        rot_raw.copy_(torch.as_tensor(data[splatsim_obj.name + "_orientation"], device="cuda"))
                        # Roll quaternion from xyzw to wxyz format using pre-allocated tensor
                        cur_object_rotation = splatsim_obj._cache['rotation_rolled']
                        cur_object_rotation[0] = rot_raw[3]  # w
                        cur_object_rotation[1:4] = rot_raw[0:3]  # xyz
                    else:
                        # Static objects: cache a zero position and identity rotation (already in wxyz format)
                        if 'static_position' not in splatsim_obj._cache:
                            splatsim_obj._cache['static_position'] = torch.tensor([0, 0, 0], device="cuda").float()
                            # Identity quaternion in wxyz format: (1, 0, 0, 0)
                            splatsim_obj._cache['static_rotation'] = torch.tensor([1, 0, 0, 0], device="cuda").float()
                        cur_object_position = splatsim_obj._cache['static_position']
                        cur_object_rotation = splatsim_obj._cache['static_rotation']

                    transform_object(
                        splatsim_obj=splatsim_obj,
                        pos=cur_object_position,
                        quat=cur_object_rotation,
                        use_base_position=True,
                        inplace=False,
                        output_slices=output_slices,
                    )

    def get_pybullet_debug_camera_as_splat_camera(self) -> SplatSimCamera:
        """Convert PyBullet's debug camera to a Camera object for Gaussian splatting."""
        # This function is the inverse of this post: https://stackoverflow.com/a/75355212

        # Get PyBullet camera info
        camera_info = p.getDebugVisualizerCamera()
        # Pybullet view matrix is major-column order
        view_matrix = np.array(camera_info[2]).reshape(4, 4).T

        Tc = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).reshape(4, 4)

        T = np.linalg.inv(view_matrix) @ Tc

        R = T[:3, :3]
        t = T[:3, 3]

        R_cw_final = R
        T_wc_final = -R.T @ t

        # TODO is this needed?
        # scale = self.base_camera.scale if self.base_camera is not None else 1.0
        resolution = (
            self.base_camera.camera.image_width,
            self.base_camera.camera.image_height,
        )
        colmap_id = 0
        uid = 0
        depth_params = None
        invdepthmap = None
        image_name = "pybullet_debug_camera"
        image = torch.zeros((3, resolution[0], resolution[1]), dtype=torch.float32)

        new_camera = Camera(
            resolution,
            colmap_id,
            R_cw_final,  # Use the fixed rotation
            T_wc_final,  # Use the fixed translation
            self.base_camera.camera.FoVx,
            self.base_camera.camera.FoVy,
            depth_params,
            to_pil_image(image),
            invdepthmap,
            image_name,
            uid,
            scale=1,  # scale,
        )

        splatsim_camera = SplatSimCamera(
            camera=new_camera,
            pipeline=self.base_camera.pipeline,
            background=self.base_camera.background,
        )

        return splatsim_camera

    def set_pybullet_camera_to_match_base(self):
        """Set PyBullet's debug camera to match the base camera view."""

        if self.base_camera is None:
            print("No base camera available")
            return

        # Get base camera parameters
        R_cw = self.base_camera.camera.R
        T_wc = self.base_camera.camera.T
        scale = self.base_camera.camera.scale

        # Camera position in world space
        camera_pos = -R_cw.T @ T_wc
        camera_pos = np.array([camera_pos[0], camera_pos[2], camera_pos[1]])

        # good
        # camera_pos = np.array([-0.2063907, -6.2722306,  1.5897055]) * scale

        # bad
        # camera_pos = np.array([-0.2063907,  1.5897055,  6.2722306])
        # camera_pos = T_wc * scale

        # Camera's forward direction (camera looks down -Z axis)
        forward_cam = np.array([0, 0, -1])
        forward_world = R_cw @ forward_cam

        # Find target by ray-casting forward from camera
        # Use a reasonable distance (e.g., distance to origin)
        target_distance = np.linalg.norm(camera_pos)  # Distance to origin as estimate
        target = camera_pos + forward_world * target_distance

        # Or directly use origin if robot is there
        target = np.array([0.0, 0.0, 0.0])

        # Distance from camera to target
        distance = np.linalg.norm(target - camera_pos)

        # Compute yaw and pitch
        # Forward vector from camera to target
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Yaw: angle in XY plane
        yaw = np.rad2deg(np.arctan2(forward[1], forward[0]))

        # Pitch: angle from XY plane
        pitch = np.rad2deg(np.arcsin(forward[2]))

        print(f"Setting PyBullet camera to match base camera:")
        print(f"  Camera position: {camera_pos}")
        print(f"  Target: {target}")
        print(f"  Distance: {distance:.3f}")
        print(f"  Yaw: {yaw:.1f}°")
        print(f"  Pitch: {pitch:.1f}°")

        # Set PyBullet camera
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=list(target),
        )

        print("PyBullet camera updated!\n")

    def render_image(self, camera_name):
        if camera_name == "base_rgb":
            if self.debug_mode != DebugModes.OFF:
                camera = self.get_pybullet_debug_camera_as_splat_camera()
            else:
                camera = self.base_camera
        elif camera_name == "wrist_rgb":
            camera = self.get_wrist_camera()
            if camera is None:
                return None
        else:
            raise ValueError(f"Unknown camera name {camera_name}")

        rendering = render(
            camera.camera, self.scene_gaussian, camera.pipeline, camera.background
        )["render"].cpu().numpy()
        # If you index "depth" instead of "render", you get the depth image

        # save the image (always as numpy array)
        return rendering

    def setup_camera_from_dataset(
        self, splatsim_obj_object_config, cam_i, use_train=True
    ) -> SplatSimCamera:
        ###################################################################
        # Load the gaussian splat dataset to get camera parameters
        ###################################################################
        source_path = resolve_splatsim_path(splatsim_obj_object_config["source_path"])
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        model_path = resolve_splatsim_path(splatsim_obj_object_config["model_path"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        parser = ArgumentParser(description="Testing script parameters")
        pipeline = PipelineParams(parser)
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
        # arbitrary as long as it's consistent between initialization and setup_camera_from_dataset()
        # because we're going to overwrite the resolution when transforming the camera
        cam_scale = 2
        temp_gaussian_model = GaussianModel(3)
        scene = Scene(
            dataset,
            temp_gaussian_model,  # This is just used for camera initialization
            load_iteration=-1,
            shuffle=False,
            resolution_scales=[cam_scale],
            train_cam_indices=[cam_i],
            test_cam_indices=[],  # we're using train cameras
        )

        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ###################################################################
        # Load the camera from the dataset
        ###################################################################

        # Assume that self.cam_train_indices and self.cam_test_indices have already singled out
        # the camera of interest. Return the first camera in the list
        if use_train:
            camera = scene.getTrainCameras(scale=cam_scale)[0]
        else:
            camera = scene.getTestCameras(scale=cam_scale)[0]

        del scene
        del temp_gaussian_model
        del dataset

        ###################################################################
        # Transform the camera to be in the simulator frame instead of the splatsim background object's splat frame
        ####################################################################

        # 1. Define device and Trans_canonical
        device = camera.world_view_transform.device
        Trans_canonical_full = (
            torch.from_numpy(
                np.array(splatsim_obj_object_config["transformation"]["matrix"])
            )
            .to(device=device)
            .float()
        )

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

        # 4. Extract rotation, translation, and scale from the transformation
        # Use SVD to properly decompose the scaled transformation (same as transform_object)
        A = M_CW_new_world_scaled[:3, :3]  # Rotation + Scale
        t = M_CW_new_world_scaled[:3, 3]  # Translation

        # Decompose A = U @ S @ Vh to get pure rotation and scale
        U, S_vec, Vh = torch.linalg.svd(A)
        R_mat = U @ Vh

        # Fix reflection if det(R) < 0
        if torch.linalg.det(R_mat) < 0:
            U[:, -1] *= -1
            S_vec[-1] *= -1
            R_mat = U @ Vh

        # Get uniform scale (use geometric mean of singular values)
        scale = torch.pow(S_vec.prod(), 1 / 3)
        scale_np = scale.detach().cpu().numpy()

        # Build C2W matrix with pure rotation and original translation
        # Don't divide translation - it should stay as-is from the transformation
        M_CW_new_world_pose = torch.eye(4, device=device, dtype=torch.float32)
        M_CW_new_world_pose[:3, :3] = R_mat
        M_CW_new_world_pose[:3, 3] = t  # Keep original translation

        # 5. Calculate T_wc (World-to-Camera Translation)
        V_new_world = torch.linalg.inv(M_CW_new_world_pose)
        T_wc = V_new_world[:3, 3]

        # Convert to numpy for the Camera constructor
        R_cw_np = R_mat.detach().cpu().numpy()
        T_wc_np = T_wc.detach().cpu().numpy()

        # 7. Initialize the New Camera
        # (Assuming other parameters are loaded as before)
        resolution = (camera.alpha_mask.shape[2], camera.alpha_mask.shape[1])
        if self.image_width is None:
            if self.image_height is None:
                image_width = resolution[0]
                image_height = resolution[1]
            else:
                image_width = int(resolution[0] * self.image_width / resolution[1])
                image_height = self.image_height
        else:
            if self.image_height is None:
                image_width = self.image_width
                image_height = int(resolution[1] * self.image_width / resolution[0])
            else:
                image_width = self.image_width
                image_height = self.image_height

        resolution = (image_width, image_height)
        image = torch.zeros((3, resolution[1], resolution[0])).float()
        depth_params = None

        # Adjust FoV to compensate for scene scaling
        # FoV relates to distance via: tan(FoV/2) = viewport_size / (2 * distance)
        # When distance is scaled by 'scale', we need: tan(FoV_new/2) = tan(FoV_old/2) * scale
        # Therefore: FoV_new = 2 * atan(tan(FoV_old/2) * scale)
        # FoVx_adjusted = 2 * np.arctan(np.tan(camera.FoVx / 2) / scale_np)
        # FoVy_adjusted = 2 * np.arctan(np.tan(camera.FoVy / 2) / scale_np)

        new_camera = Camera(
            resolution,
            camera.colmap_id,
            R_cw_np,  # Pure rotation (orthonormal)
            T_wc_np,  # W2C translation (scale-free)
            camera.FoVx,  # FoV adjusted for scene scaling
            camera.FoVy,  # FoV adjusted for scene scaling
            depth_params,
            to_pil_image(image),
            camera.invdepthmap,
            camera.image_name,
            camera.uid,
            scale=camera.scale,  # Uniform scale extracted from transformation
        )

        splatsim_camera = SplatSimCamera(
            camera=new_camera,
            pipeline=pipeline,
            background=background,
            tracked_link_index=None,  # Set this outside of this loop
        )

        return splatsim_camera

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

    def get_task_description(self) -> str:
        return self.ENV_CONFIG.task_description

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
            # Snap the gripper state to 0 or 1 if they're very close
            if self.current_gripper_state > 0.95:
                self.current_gripper_state = 1.0
            elif self.current_gripper_state < 0.05:
                self.current_gripper_state = 0.0
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

        if self.do_render_from_splat and len(self.camera_names) > 0:
            self.prep_image_rendering(data=observations)
            with torch.no_grad():
                for camera_name in self.camera_names:
                    # render_image returns numpy array directly
                    observations[camera_name] = self.render_image(
                        camera_name=camera_name
                    )
                    # Resize to 224x224 using configured mode (letterbox or stretch)
                    if observations[camera_name] is not None:
                        observations[camera_name] = resize_image(observations[camera_name], (224, 224), mode=self.image_resize_mode)

            # Display the rendered observations
            self.display_observations(observations)

        for camera_name in self.camera_names:
            # For example, when self.do-render_from_splat is False
            if camera_name not in observations:
                observations[camera_name] = None

        return observations

    def display_observations(self, observations: Dict[str, Any]) -> None:
        """Display rendered RGB observations using cv2.imshow()

        Args:
            observations: Dictionary containing rendered images as torch tensors (C, H, W)
        """
        frames_to_display = {}
        for img_obs_name in ["wrist_rgb", "base_rgb"]:
            if img_obs_name not in observations or observations[img_obs_name] is None:
                continue
            frame = observations[img_obs_name]

            # Convert from tensor if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            # Convert from CxHxW to HxWxC
            frame = np.transpose(frame, (1, 2, 0))

            # Convert from [0, 1] float to [0, 255] uint8
            frame = (frame * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            frames_to_display[img_obs_name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # If on main thread, display directly; otherwise queue for main thread
        if threading.current_thread().ident == self._main_thread_id:
            self._do_display(frames_to_display)
        else:
            self._display_queue.put(frames_to_display)

    def _do_display(self, frames: Dict[str, np.ndarray]) -> None:
        """Actually display frames using cv2.imshow. Must be called from main thread."""
        for img_obs_name, frame in frames.items():
            cv2.imshow(img_obs_name, frame)
        cv2.waitKey(1)

    def _process_display_queue(self) -> None:
        """Process any pending display requests from the queue. Call from main thread."""
        while not self._display_queue.empty():
            try:
                frames = self._display_queue.get_nowait()
                self._do_display(frames)
            except queue.Empty:
                break

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
                    for conf in self.ENV_CONFIG.objects
                    if conf.object_name == self.splatsim_objects[i].name
                ][0]
                my_object_config = self.splatsim_objects[i].object_config
                table_pos = getattr(my_object_env_config, "table_pos", None)
                if table_pos is not None:
                    table_quat = getattr(my_object_env_config, "table_quat", (0, 0, 0, 1))
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
                elif my_object_env_config.randomize_pose:
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
                        my_object_env_config.rotation_range_z[0],
                        my_object_env_config.rotation_range_z[1],
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

    def randomize_ee_pose(self, max_attempts=100):
        # generating random initial joint state using random end effector position and orientation
        for attempt in range(max_attempts):
            random_ee_pos, random_ee_quat = self.get_random_ee_pose()

            # joint angles using inverse kinematics
            initial_joint_positions = self.pybullet_client.calculateInverseKinematics(
                self.splatsim_robot.sim_id,
                6,
                random_ee_pos,
                random_ee_quat,
                maxNumIterations=100000,
                residualThreshold=1e-10,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
            )

            # Wrap joint angles to [-pi, pi]
            initial_joint_positions = tuple(
                ((angle + np.pi) % (2 * np.pi)) - np.pi
                for angle in initial_joint_positions
            )

            # reset the joint positions to the initial joint positions
            for i in range(1, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id, i, initial_joint_positions[i - 1]
                )

            # Check for collisions between robot and scene objects
            # Only detect actual penetration (negative contact distance), not just proximity
            collision_detected = False
            for splatsim_obj in self.splatsim_objects:
                if splatsim_obj.sim_id is not None and splatsim_obj != self.splatsim_robot:
                    contacts = self.pybullet_client.getClosestPoints(
                        self.splatsim_robot.sim_id, splatsim_obj.sim_id, distance=0.0
                    )
                    # Check if any contact has negative distance (actual penetration)
                    for contact in contacts:
                        contact_distance = contact[8]  # contactDistance is at index 8
                        if contact_distance < -0.001:  # Small tolerance for numerical stability
                            collision_detected = True
                            break
                    if collision_detected:
                        break

            if not collision_detected:
                # TODO possibly randomize gripper state here, too
                # Though that might have to edit initial_joint_positions
                return initial_joint_positions

        # If we exhausted all attempts, return the last configuration with a warning
        print(f"Warning: Could not find collision-free EE pose after {max_attempts} attempts")
        return initial_joint_positions

    def get_random_ee_pose(self):
        if self.TABLE_LIMITS is None:
            raise NotImplementedError(
                "TABLE_LIMITS must be set in the environment to randomize end effector pose."
            )
        # random end effector position
        # if random.uniform(0, 1) > 0.2:
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
        # TODO move this logic to the object_on_plate environment
        # else:
        #     # get object position
        #     (
        #         object_pos,
        #         object_quat,
        #     ) = self.pybullet_client.getBasePositionAndOrientation(
        #         self.splatsim_objects[0].sim_id
        #     )
        #     random_x = random.uniform(-0.105, 0.105)
        #     random_y = random.uniform(-0.105, 0.105)
        #     random_z = random.uniform(0.25, 0.3)
        #     random_ee_pos = np.array(
        #         [
        #             object_pos[0] + random_x,
        #             object_pos[1] + random_y,
        #             object_pos[2] + random_z,
        #         ]
        #     )
        # random_ee_pos = np.array([random.uniform(0.2, 0.5), random.uniform(-0.6, 0.6), random.uniform(0.2, 0.65)])

        # get the euler angles from the quaternion
        # get quaternion from euler angles
        # TODO move this logic to the object_on_plate environment (end effector pointing down)
        # random_ee_quat = self.initial_ee_quat

        # random_ee_quat is any an arbitrary orientation
        random_ee_quat = self.pybullet_client.getQuaternionFromEuler(
            [
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
            ]
        )

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
        self.reset()

        # start the zmq server
        self._zmq_server_thread.start()

        print("Ready to serve.")

        # # Set motors to hold current position to prevent arm from falling due to gravity
        # for i in range(1, self.num_dofs()):
        #     current_pos = self.pybullet_client.getJointState(
        #         self.splatsim_robot.sim_id, i
        #     )[0]
        #     self.pybullet_client.setJointMotorControl2(
        #         self.splatsim_robot.sim_id,
        #         i,
        #         self.pybullet_client.POSITION_CONTROL,
        #         targetPosition=current_pos,
        #         force=500,  # Enough force to hold against gravity
        #     )
        
        while True:
            # Process any pending display requests from ZMQ thread
            self._process_display_queue()

            # Check mode dropdown for mode changes
            new_mode = self._check_mode_buttons()
            if new_mode is not None:
                self._handle_mode_transition(new_mode)

            # Check debug mode dropdown for changes
            self._check_debug_mode()

            # Check GUI buttons for trajectory generation control (start/stop)
            start_pressed, stop_pressed = self._splatsim_gui.check_traj_buttons()

            if start_pressed and self.serve_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
                # Sync GUI values to config before starting
                self._splatsim_gui.save_to_config(self.trajectory_generator.config)
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

            self.serve_loop()

    def serve_loop(self):
        raise NotImplementedError()

    def plan_given_this_state(self, initial_joint_positions):
        raise NotImplementedError()

    def delete_trajectory_folder(self):
        shutil.rmtree(os.path.join(self.path, str(self.trajectory_count).zfill(3)))

    def stop(self) -> None:
        """Stop the robot server and clean up resources.

        Safe to call whether or not serve() was called (gym mode vs ZMQ server mode).
        """
        # Only join the thread if it was actually started
        if self._zmq_server_thread.is_alive():
            self._zmq_server.stop()
            self._zmq_server_thread.join(timeout=2.0)

        # Clean up GUI if it exists
        if hasattr(self, '_splatsim_gui') and self._splatsim_gui is not None:
            self._splatsim_gui.stop()
            self._splatsim_gui = None

        # Disconnect pybullet
        if hasattr(self, 'pybullet_client') and self.pybullet_client is not None:
            try:
                self.pybullet_client.disconnect()
            except Exception:
                pass  # Already disconnected

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass  # Ignore errors during garbage collection

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
        # Snap the gripper state to 0 or 1 if they're very close
        gripper_state = self.pybullet_client.getJointState(
            self.splatsim_robot.sim_id, self.mimic_parent_id
        )[0]
        return gripper_state

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

    # =========================================================================
    # Gym Environment Interface
    # =========================================================================

    @property
    def action_space(self) -> spaces.Box:
        """Define the action space.

        Returns:
            Box space for 7 DOF actions (6 joints + gripper)
        """
        # Joint limits - can be refined from URDF if needed
        low = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0.0], dtype=np.float32)
        high = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        """Define the observation space.

        Returns:
            Dict space with agent_pos (state) and pixels dict (images)
            in LeRobot-compatible format.
        """
        # Build pixels dict space for each camera
        pixels_dict = {}
        for camera_name in self.camera_names:
            pixels_dict[camera_name] = spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3), dtype=np.uint8
            )

        obs_dict = {
            "agent_pos": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,), dtype=np.float32
            ),
            "pixels": spaces.Dict(pixels_dict)
        }
        return spaces.Dict(obs_dict)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one control step in the environment.

        Args:
            action: Joint positions array of shape (7,) - 6 joints + gripper

        Returns:
            observation: Dict with state and images
            reward: Float reward signal
            terminated: True if episode ended due to success/failure condition
            truncated: True if episode ended due to time limit
            info: Dict with 'is_success' key
        """
        assert self._episode_started, "Must call reset() before step()"

        # Apply action via existing method
        self.command_joint_state(action)

        # Step physics simulation (multiple substeps for stability)
        for _ in range(self._physics_steps_per_action):
            self.pybullet_client.stepSimulation()

        self._step_count += 1

        # Get new observation
        observation = self._get_gym_observation()

        # Compute reward, termination, success
        reward = self.compute_reward()
        is_success = self.check_success()
        terminated = self.check_terminated()
        truncated = self._step_count >= self._max_episode_steps

        metrics = self.check_metrics()

        info = {
            "is_success": is_success,
            "step_count": self._step_count,
            **metrics
        }

        return observation, reward, terminated, truncated, info

    def _get_gym_observation(self) -> Dict[str, Any]:
        """Get observation in Gym-compatible format.

        Returns:
            Dict with:
                - 'state': np.ndarray of joint positions + gripper (7,)
                - camera images by name (e.g., 'base_rgb', 'wrist_rgb')
        """
        raw_obs = self.get_observations()

        # Joint state (6 joints + gripper)
        joint_positions = raw_obs["joint_positions"][:6]
        gripper_state = raw_obs.get("gripper_position", [0.0])
        if isinstance(gripper_state, (list, np.ndarray)):
            gripper_state = gripper_state[0] if len(gripper_state) > 0 else 0.0

        # Use "agent_pos" key for LeRobot compatibility
        gym_obs = {
            "agent_pos": np.concatenate([joint_positions, [gripper_state]]).astype(np.float32)
        }

        # Images - use "pixels" dict format for LeRobot compatibility
        # LeRobot expects: {"pixels": {"base_rgb": img, "wrist_rgb": img}} for multi-camera
        # or {"pixels": img} for single camera
        pixels = {}
        for camera_name in self.camera_names:
            img = raw_obs.get(camera_name)
            if img is not None:
                if hasattr(img, 'cpu'):
                    img = img.cpu().numpy()
                # Convert from (C, H, W) float32 to (H, W, C) uint8 for LeRobot
                if img.shape[0] == 3:  # (C, H, W) format
                    img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                pixels[camera_name] = img
            else:
                # Provide placeholder if camera not available
                pixels[camera_name] = np.zeros((224, 224, 3), dtype=np.uint8)

        # Use dict format for pixels (LeRobot expects this for multi-camera)
        gym_obs["pixels"] = pixels

        return gym_obs

    def check_truncated(self) -> bool:
        """Check if episode should be truncated due to time limit.

        Returns:
            True if step limit exceeded
        """
        return self._step_count >= self._max_episode_steps

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Subclasses must implement this method with task-specific reset logic.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict

        Returns:
            observation: Initial observation dict
            info: Dict with initial info

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement reset()")

    def compute_reward(self) -> float:
        """Compute the reward for the current state.

        Subclasses must implement this method with task-specific reward logic.

        Returns:
            Float reward value

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")

    def check_success(self) -> bool:
        """Check if the current episode achieved success.

        Subclasses must implement this method with task-specific success criteria.

        Returns:
            True if success condition is met

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement check_success()")
    
    def check_metrics(self) -> Dict[str, Any]:
        """Check additional metrics for the current episode.

        Subclasses can implement this method to return task-specific metrics.

        Returns:
            Dict of metric names to values
        """
        return {}

    def check_terminated(self) -> bool:
        """Check if episode should terminate due to success or failure.

        Subclasses must implement this method with task-specific termination conditions.

        Returns:
            True if episode should terminate

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement check_terminated()")

    # =========================================================================
    # Mode-based GUI Controls
    # =========================================================================

    def _setup_interactive_gui(self):
        """Create GUI controls using Tkinter.

        This launches the SplatSimGui window which includes:
        - Mode switching buttons (Interactive / Trajectory Gen)
        - Debug mode dropdown
        - Trajectory generation parameters
        - Start/Stop trajectory generation buttons
        """
        # Initialize the Tkinter GUI (runs in separate thread)
        config = self.trajectory_generator.config
        initial_mode = self.serve_mode.value  # Use the enum's string value
        self._splatsim_gui: SplatSimGui = SplatSimGui(
            config,
            initial_mode,
            debug_mode_enum=self.DEBUG_MODES,
            initial_debug_mode=self.debug_mode,
        )
        self._splatsim_gui.start()

    def _check_mode_buttons(self) -> Optional['PybulletRobotServerBase.SERVE_MODES']:
        """Check if a mode button was pressed and return new mode if so.

        Returns:
            New SERVE_MODES value if a button was pressed, None otherwise.
        """
        interactive_pressed, traj_gen_pressed = self._splatsim_gui.check_mode_buttons()

        if interactive_pressed and self.serve_mode != self.SERVE_MODES.INTERACTIVE:
            return self.SERVE_MODES.INTERACTIVE

        # Trajectory gen button goes to IDLE mode first (user must click Start to begin)
        if traj_gen_pressed and self.serve_mode not in (
            self.SERVE_MODES.GENERATE_TRAJECTORIES,
            self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE
        ):
            return self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE

        return None

    def _check_debug_mode(self):
        """Check if debug mode has changed in the GUI and update self.debug_mode."""
        new_debug_mode = self._splatsim_gui.get_debug_mode()
        if new_debug_mode is None:
            return  # GUI not yet initialized
        if new_debug_mode != self.debug_mode:
            print(f"[GUI] Debug mode changed: {self.debug_mode.value} -> {new_debug_mode.value}")
            self.debug_mode = new_debug_mode

    def _handle_mode_transition(self, new_mode: 'PybulletRobotServerBase.SERVE_MODES'):
        """Handle mode transition.

        Args:
            new_mode: The new SERVE_MODES value to transition to.
        """
        if new_mode == self.SERVE_MODES.INTERACTIVE:
            print(f"[GUI] Switched to Interactive mode")

        elif new_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
            print(f"[GUI] Switched to Trajectory Generation mode (idle - click Start to begin)")

        elif new_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            print(f"[GUI] Trajectory generation started")

        self.serve_mode = new_mode

        # Update the GUI mode display
        self._splatsim_gui.set_mode(new_mode.value)

    def shutdown(self):
        """Clean up resources.

        Another name for self.stop(). This is to fit with the gello api
        """
        self.stop()
