"""A SplatSim node as a lerobot Robot (`--robot.type=splatsim_lerobot`); moved here from the fork's robots/splatsim_lerobot/."""

from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


class ImageResizeMode(str, Enum):
    """Image resize modes for SplatSim observations.

    LETTERBOX: Resize while keeping aspect ratio, pad with black bars.
               Good for pretrained VLAs (PI0, PI0.5) that expect preserved aspect ratios.
    STRETCH: Resize to fill entire target size without keeping aspect ratio.
             Good for policies trained from scratch (diffusion) that benefit from
             using the full image area without black padding.
    """

    LETTERBOX = "letterbox"
    STRETCH = "stretch"


@RobotConfig.register_subclass("splatsim_lerobot")
@dataclass
class SplatSimLerobotConfig(RobotConfig):
    """Configuration for SplatSim LeRobot simulation robot"""

    # ZMQ connection settings
    port: int | None = None
    hostname: str = "127.0.0.1"

    # Camera ports
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001

    # Control rate for RobotEnv
    control_rate_hz: int = 50

    # Image size for observations
    image_width: int = 224
    image_height: int = 224

    # Image resize mode:
    # - "letterbox": Resize keeping aspect ratio, pad with black bars (good for pretrained VLAs)
    # - "stretch": Resize to fill entire area without keeping aspect ratio (good for diffusion)
    image_resize_modes: list[str] = field(default_factory=lambda: ["letterbox"])

    # Related to camera ports
    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # More of a SplatSim config, which do not exist on a camera port
    # Camera names to include in observations (e.g., ["base_rgb"], ["wrist_rgb"], or ["base_rgb", "wrist_rgb"])
    camera_names: list[str] = field(default_factory=lambda: ["base_rgb", "wrist_rgb"])

    # Joint names (can be configured to match your dataset)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
    )


import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import torch
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from splatsim.utils.image_utils import letterbox

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


def resize_image(img: np.ndarray, output_size: tuple[int, int], mode: str = "letterbox") -> np.ndarray:
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


class SplatSimLerobot(Robot):
    """Robot interface for SplatSim simulation using RobotEnv from gello"""

    config_class = SplatSimLerobotConfig
    name = "splatsim_lerobot"

    def __init__(self, config: SplatSimLerobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_client = None
        self.env = None  # RobotEnv instance
        self._is_connected = False
        # Cameras are managed through RobotEnv, not LeRobot's camera system
        # But lerobot-record expects a cameras dict for counting threads
        self.cameras = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space"""
        features = {}

        # Add each configured camera (with resize mode suffix to match get_observation keys)
        for camera_name in self.config.camera_names:
            for image_resize_mode in self.config.image_resize_modes:
                features[f"{camera_name}_{image_resize_mode}"] = (
                    self.config.image_height,
                    self.config.image_width,
                    3,
                )

        # Add individual state features for each joint (LeRobot expects named joints)
        for joint_name in self.config.joint_names:
            features[f"state.{joint_name}"] = float

        return features

    @cached_property
    def action_features(self) -> dict:
        """Define action space"""
        features = {}

        # Add individual action features for each joint (LeRobot expects named joints)
        for joint_name in self.config.joint_names:
            features[f"action.{joint_name}"] = float

        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to ZMQ robot and create RobotEnv"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Create ZMQ robot client
        self.robot_client = ZMQClientRobot(port=self.config.port, host=self.config.hostname)

        # Don't create camera clients - SplatSim provides images through robot observations
        # The camera data comes embedded in the robot's get_obs() response
        camera_clients = {}

        # Create RobotEnv with robot only (no separate camera clients)
        self.env = RobotEnv(
            robot=self.robot_client,
            control_rate_hz=self.config.control_rate_hz,
            camera_dict=camera_clients,
        )

        self._is_connected = True
        logger.info(f"{self} connected to {self.config.hostname}:{self.config.port}")

    @property
    def is_calibrated(self) -> bool:
        """Simulation robots don't require calibration"""
        return True

    def calibrate(self) -> None:
        """Simulation robots don't require calibration"""
        pass

    def configure(self) -> None:
        """Apply any runtime configuration"""
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from RobotEnv (includes images from robot_obs)"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get observation from RobotEnv - this includes images!
        obs = self.env.get_obs()

        # Format observation to match LeRobot expectations
        lerobot_obs = {}

        # Process each configured camera
        for camera_name in self.config.camera_names:
            for image_resize_mode in self.config.image_resize_modes:
                key = f"{camera_name}_{image_resize_mode}"
                if key in obs:
                    img = obs[key]
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    # (C, H, W) -> (H, W, C)
                    if img.ndim == 3 and img.shape[0] <= 4:
                        img = img.transpose(1, 2, 0)
                    lerobot_obs[key] = img
                else:
                    img = obs.get(camera_name)
                    if img is not None:
                        # Convert to numpy if needed
                        if isinstance(img, torch.Tensor):
                            img = img.detach().cpu().numpy()

                        # Resize to configured size using configured mode
                        img_resized = resize_image(
                            img,
                            output_size=(self.config.image_height, self.config.image_width),
                            mode=image_resize_mode,
                        )
                        # Change from (C, H, W) to (H, W, C)
                        lerobot_obs[key] = img_resized.transpose(1, 2, 0)
            if not any(k for k in lerobot_obs if k.startswith(camera_name)):
                logger.warning(f"No {camera_name} in observation!")

        # Process joint positions (state)
        joint_positions = obs.get("joint_positions")
        if joint_positions is not None:
            if isinstance(joint_positions, torch.Tensor):
                joint_positions = joint_positions.detach().cpu().numpy()
            if not isinstance(joint_positions, np.ndarray):
                joint_positions = np.array(joint_positions)

            # Append gripper state if needed (7 DOF vs 6 DOF)
            if joint_positions.shape[0] == 6:
                gripper_pos = obs.get("gripper_position", 0.0)
                if isinstance(gripper_pos, np.ndarray):
                    gripper_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
                elif isinstance(gripper_pos, torch.Tensor):
                    gripper_pos = gripper_pos.item()
                joint_positions = np.append(joint_positions, gripper_pos)

            # Add individual joint states (LeRobot expects named joints, not a single array)
            for i, joint_name in enumerate(self.config.joint_names):
                if i < len(joint_positions):
                    lerobot_obs[f"state.{joint_name}"] = float(joint_positions[i])

        return lerobot_obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to simulation via RobotEnv.step()"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract action values
        if isinstance(action, dict):
            # Try different possible keys
            action_array = action.get("action", action.get("joint_positions"))
            if action_array is None:
                # Try to reconstruct from individual joint actions
                # First try with 'action.' prefix
                action_list = [action.get(f"action.{name}") for name in self.config.joint_names]
                if all(a is not None for a in action_list):
                    action_array = np.array(action_list)
                else:
                    # Try without prefix (policy output format)
                    action_list = [action.get(name) for name in self.config.joint_names]
                    if all(a is not None for a in action_list):
                        action_array = np.array(action_list)
                    else:
                        raise ValueError(f"Could not extract action from dict: {action.keys()}")
        else:
            action_array = action

        if isinstance(action_array, torch.Tensor):
            action_array = action_array.detach().cpu().numpy()

        # Ensure it's a 1D array
        if action_array.ndim > 1:
            action_array = action_array.squeeze()

        # Remove gripper if your sim expects 6 DOF
        # action_to_send = action_array[:6] if len(action_array) == 7 else action_array
        action_to_send = action_array

        # Send to robot via RobotEnv.step() - this returns the next observation
        # but we don't use it here since get_observation() will be called separately
        self.env.step(action_to_send)

        # Return the action that was sent (in the original format)
        if isinstance(action, dict):
            return action
        else:
            return {"action": action_array}

    def disconnect(self):
        """Disconnect from robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
        self.robot_client = None
        self.env = None
        logger.info(f"{self} disconnected.")
