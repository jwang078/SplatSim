import zarr
import numpy as np
import os
import shutil
from typing import Dict, Union, Optional
import numcodecs

from typing import Dict
import torch
import numpy as np

# From the diffusion_policy repo
# Run this with
# PYTHONPATH=~/code/diffusion_policy python scripts/convert_traj_refinement_dataset_to_diffusion_policy.py
from diffusion_policy.common.replay_buffer import ReplayBuffer

def create_replay_buffer_from_splatsim_data(input_zarr_path: str, output_zarr_path: str, dof: int = 7, horizon: int = 16):
    """
    Flattens the Splatsim dataset into a linear ReplayBuffer structure.

    Args:
        input_zarr_path: The path to the root of the hierarchical Zarr dataset.
        output_zarr_path: The path where the new, flattened ReplayBuffer will be saved.
        dof: The degrees of freedom for the joint angle data (expected: 7).
    """
    print(f"Loading hierarchical Zarr dataset from: {input_zarr_path}")

    print("Using time horizon", horizon)
    
    # 1. Open the source Zarr dataset
    try:
        src_root = zarr.open(os.path.expanduser(input_zarr_path), mode='r')
    except Exception as e:
        print(f"Error opening input Zarr: {e}")
        return

    # Ensure the output directory is clean
    if os.path.exists(output_zarr_path):
        print(f"Removing existing output directory: {output_zarr_path}")
        shutil.rmtree(output_zarr_path)
    
    # 2. Create an empty destination Zarr group for the ReplayBuffer
    dest_store = zarr.DirectoryStore(os.path.expanduser(output_zarr_path))
    dest_root = zarr.group(store=dest_store)
    replay_buffer = ReplayBuffer.create_empty_zarr(root=dest_root)
    
    # Define required shapes for the Diffusion Policy dataset class
    IMAGE_SHAPE_HWC = (96, 96, 3) 
    
    # 3. Iterate through the source dataset
    trajectory_count = 0
    
    # Get the 'trajectories' group
    traj_root = src_root.get('trajectories')
    if not traj_root:
        print("Error: 'trajectories' group not found in the input Zarr root.")
        return
        
    for scenario_name in traj_root.keys():
        scenario_group = traj_root[scenario_name]
        
        for config_name in scenario_group.keys():
            config_group = scenario_group[config_name]
            
            # The innermost group may contain multiple 'traj_##' keys
            for traj_name in config_group.keys():
                traj_group = config_group[traj_name]
                
                # Check for the qs data
                qs_array = traj_group.get('qs')
                if qs_array is None:
                    print(f"Warning: 'qs' not found in {scenario_name}/{config_name}/{traj_name}. Skipping.")
                    continue

                # Load joint angle data
                # Expected shape is (120, 7)
                qs_data = qs_array[:] 
                
                # Input validation
                if not (qs_data.ndim == 2 and qs_data.shape[1] == dof):
                    print(f"Error: Expected qs shape (-1, {dof}), but found {qs_data.shape} in {traj_name}. Skipping.")
                    continue
                
                base_rgb_data = traj_group.get("base_rgb")
                if not (base_rgb_data.ndim == 4 and base_rgb_data.shape[0] == qs_data.shape[0] and base_rgb_data.shape[3] == 3):
                    print(f"Error: Expected base_rgb shape ({qs_data.shape[0]}, xx, xx, 3), but found {base_rgb_data.shape} in {traj_name}. Skipping.")
                    continue

                # 4. Prepare data dictionary for ReplayBuffer.add_episode
                # Use qs for both 'state' and 'action'
                data_to_add = {
                    # State is the past
                    'state': qs_data.astype(np.float32)[:-horizon + 1],    # (T-1, DOF)
                    'base_rgb': base_rgb_data.astype(np.float32)[:-horizon + 1], # (T-1, image_height, image_width, 3)

                    # Action is the future
                    'action': qs_data.astype(np.float32)[horizon - 1:],   # (T-1, DOF)
                }

                # 6. Add the episode to the ReplayBuffer
                replay_buffer.add_episode(data_to_add)
                trajectory_count += 1

    print(f"\n✅ Flattening complete. Added {trajectory_count} trajectories.")
    print(f"Total steps in ReplayBuffer: {replay_buffer.n_steps}")
    print(f"Output saved to: {output_zarr_path}")

# --- Example of How to Call ---
# You would need to adjust the DOF in your SplatsimObstacleAvoidanceDataset class
# to 'DOF = 7' to match your data.

dof_count = 7
horizon = 16
input_path = 'output/obstacles_on_path_onegoal.zarr'
output_path = f'output/obstacles_on_path_onegoal_diffusionpolicy_horizon{horizon}.zarr'
create_replay_buffer_from_splatsim_data(input_path, output_path, dof=dof_count, horizon=horizon)