from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from lerobot.robots.robot import Robot, RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
from splatsim.utils.image_utils import letterbox

@dataclass
class ZMQSimRobotConfig(RobotConfig):
    """Configuration for ZMQ-based simulation robot"""
    robot_type: str = "zmq_sim_robot"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    # Define your cameras (for LeRobot metadata purposes)
    cameras: dict = None
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = {
                "base_rgb": OpenCVCameraConfig(
                    type="zmq",
                    width=224,
                    height=224,
                    fps=30
                )
            }

class ZMQSimRobot(Robot):
    """Robot interface for ZMQ-based simulation using RobotEnv"""
    
    def __init__(self, config: ZMQSimRobotConfig):
        super().__init__(config)
        self.robot_type = config.robot_type
        self.robot_port = config.robot_port
        self.hostname = config.hostname
        self.base_camera_port = config.base_camera_port
        self.wrist_camera_port = config.wrist_camera_port
        
        self.robot_client = None
        self.env = None  # RobotEnv instance
        
    @property
    def name(self) -> str:
        return "zmq_sim_robot"
    
    def connect(self):
        """Connect to ZMQ robot and create RobotEnv"""
        # Create ZMQ robot client
        self.robot_client = ZMQClientRobot(
            port=self.robot_port,
            host=self.hostname
        )
        
        # Create camera clients dict for RobotEnv
        camera_clients = {
            "base": ZMQClientCamera(
                port=self.base_camera_port,
                host=self.hostname
            ),
            # Add wrist camera if needed
            # "wrist": ZMQClientCamera(
            #     port=self.wrist_camera_port,
            #     host=self.hostname
            # ),
        }
        
        # Create RobotEnv with robot and cameras
        self.env = RobotEnv(
            robot=self.robot_client,
            control_rate_hz=50,  # Match your control rate
            camera_dict=camera_clients
        )
        
        self.is_connected = True
        print(f"Connected to ZMQ robot at {self.hostname}:{self.robot_port}")
    
    def disconnect(self):
        """Disconnect from robot"""
        self.is_connected = False
    
    def get_observation(self) -> dict:
        """Get current observation from RobotEnv (includes images from robot_obs)"""
        # Get observation from RobotEnv - this includes images!
        obs = self.env.get_obs()
        
        # Format observation to match LeRobot expectations
        lerobot_obs = {}
        
        # Process base_rgb image (comes from robot_obs in RobotEnv)
        base_rgb = obs.get("base_rgb")
        if base_rgb is not None:
            # Convert to (C, H, W) format and resize to 224x224
            if isinstance(base_rgb, torch.Tensor):
                base_rgb = base_rgb.detach().cpu().numpy()
            base_rgb_resized = letterbox(base_rgb, output_size=(224, 224))
            lerobot_obs["images.base_rgb"] = torch.from_numpy(base_rgb_resized).float()
        
        # Process wrist_rgb if available
        wrist_rgb = obs.get("wrist_rgb")
        if wrist_rgb is not None:
            if isinstance(wrist_rgb, torch.Tensor):
                wrist_rgb = wrist_rgb.detach().cpu().numpy()
            wrist_rgb_resized = letterbox(wrist_rgb, output_size=(224, 224))
            lerobot_obs["images.wrist_rgb"] = torch.from_numpy(wrist_rgb_resized).float()
        
        # Process joint positions (state)
        joint_positions = obs.get("joint_positions")
        if joint_positions is not None:
            if not isinstance(joint_positions, np.ndarray):
                joint_positions = np.array(joint_positions)
            
            # Append gripper state if needed (7 DOF vs 6 DOF)
            if joint_positions.shape[0] == 6:
                gripper_pos = obs.get("gripper_position", 0.0)
                if isinstance(gripper_pos, np.ndarray):
                    gripper_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
                joint_positions = np.append(joint_positions, gripper_pos)
            
            lerobot_obs["state"] = torch.from_numpy(joint_positions).float()
        
        return lerobot_obs
    
    def send_action(self, action: dict) -> dict:
        """Send action to simulation via RobotEnv.step()"""
        # Extract action values
        if isinstance(action, dict):
            action_array = action.get("action", action.get("joint_positions"))
        else:
            action_array = action
            
        if isinstance(action_array, torch.Tensor):
            action_array = action_array.detach().cpu().numpy()
        
        # Ensure it's a 1D array
        if action_array.ndim > 1:
            action_array = action_array.squeeze()
        
        # Remove gripper if your sim expects 6 DOF
        if len(action_array) == 7:
            action_array = action_array[:6]
        
        # Send to robot via RobotEnv.step() - this returns the next observation
        # but we don't use it here since get_observation() will be called separately
        self.env.step(action_array)
        
        return action
    
    @property 
    def action_features(self) -> dict:
        """Define action space"""
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),  # 6 joints + gripper
                "names": [
                    "joint_1",
                    "joint_2", 
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper"
                ]
            }
        }
    
    @property
    def observation_features(self) -> dict:
        """Define observation space"""
        features = {
            "images.base_rgb": {
                "dtype": "video",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"]
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "joint_1",
                    "joint_2",
                    "joint_3", 
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper"
                ]
            }
        }
        
        # Add wrist camera if you have it
        # features["images.wrist_rgb"] = {
        #     "dtype": "video",
        #     "shape": (3, 224, 224),
        #     "names": ["channels", "height", "width"]
        # }
        
        return features