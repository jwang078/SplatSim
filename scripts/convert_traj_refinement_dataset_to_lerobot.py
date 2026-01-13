import os
import zarr
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def letterbox(img, output_size=(224, 224)):
    """
    add black bars to the sides of the image to make it square, then resize
    """
    # Use (c, h, w) input with a numpy array
    c, h, w = img.shape
    scale = min(output_size[1] / w, output_size[0] / h)
    
    # New dimensions that maintain aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 1. Resize the image so the long side fits
    # Opencv expects (h, w, c)
    img_resized = cv2.resize(img.transpose(1, 2, 0), (new_w, new_h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    
    # 2. Create a black canvas
    canvas = np.zeros((3, output_size[0], output_size[1]), dtype=np.uint8)
    
    # 3. Paste the resized image into the center of the canvas
    offset_x = (output_size[1] - new_w) // 2
    offset_y = (output_size[0] - new_h) // 2
    canvas[:, offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img_resized
    
    return canvas

def convert_splatsim_to_lerobot(
    input_zarr_path: str, 
    repo_id: str, 
    fps: int = 50
):
    # 1. Open Source Dataset
    input_path = Path(input_zarr_path).expanduser()
    print(f"Loading Zarr from: {input_path}")
    src_root = zarr.open(str(input_path), mode='r')
    traj_root = src_root.get('trajectories')

    # 2. Initialize LeRobot Dataset
    # We define the schema based on your SplatSim requirements
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="lerobot_splatsim",
        use_videos=True,
        features={
            "observation.images.base_rgb": {
                "dtype": "image",
                "shape": (3, 224, 224),  # Pi05 expects 224x224
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            },
        },
    )

    # 3. Flatten and Add Data
    for scenario in tqdm(sorted(traj_root.keys()), desc="Scenarios"):
        scenario_group = traj_root[scenario]
        
        for config in scenario_group.keys():
            config_group = scenario_group[config]
            
            for traj_name in config_group.keys():
                traj_group = config_group[traj_name]
                
                # Extract Data
                qs = traj_group.get('qs')[:]          # Joint angles (T, 7)
                base_rgb = traj_group.get('base_rgb') # Image sequence (T, H, W, 3)
                
                if qs is None or base_rgb is None:
                    continue

                # Process Frames
                for i in range(len(qs)):
                    # Image Preprocessing: Resize and Transpose to CxHxW
                    img = base_rgb[i]
                    img_resized = letterbox(img, output_size=(224, 224)) # 224x224 is the usual size for paligemma / pi0 (224x224, 448x448, and 896x896 pixels)
                    img_torch = np.transpose(img_resized, (2, 0, 1)) # HWC -> CHW

                    # State and Action mapping
                    # (Note: In your previous script you used non_gripper_qs, 
                    # here we use the full 7-DoF state/action)
                    state = qs[i].astype(np.float32)
                    
                    # For Imitation Learning, action at time T is typically 
                    # the state at time T+1. Adjust if your dataset is offset.
                    action = qs[i].astype(np.float32) 

                    lerobot_dataset.add_frame({
                        "observation.images.base_rgb": img_torch,
                        "observation.state": state,
                        "action": action,
                        "task": f"Go to home position", # Important for Pi0
                    })
                
                # Signal the end of an episode to save metadata
                lerobot_dataset.save_episode()

    # 4. Finalize
    print("\nFinalizing dataset...")
    lerobot_dataset.finalize()
    
    # Optional: Push to Hugging Face
    # lerobot_dataset.push_to_hub()
    print(f"Success! Dataset stored at: {lerobot_dataset.root}")

if __name__ == "__main__":
    convert_splatsim_to_lerobot(
        input_zarr_path="output/obstacles_on_path_onegoal_20dataset_simple.zarr",
        repo_id="JennyWWW/splatsim_lerobot_dataset_obstacles_on_path_onegoal_20dataset_simple"
    )