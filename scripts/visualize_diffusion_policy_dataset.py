import zarr
import numpy as np
import os
import shutil
from typing import Dict, Union, Optional
import numcodecs

from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import cv2


# From the diffusion_policy repo
# Run this with
# PYTHONPATH=~/code/diffusion_policy python scripts/convert_traj_refinement_dataset_to_diffusion_policy.py
from diffusion_policy.common.replay_buffer import ReplayBuffer

zarr_path = f'output/obstacles_on_path_onegoal_noshift_6dof_100dataset_diffusionpolicy.zarr'
# zarr_zip_path = "/home/jennyw2/code/SplatSim/output/obstacles_on_path_onegoal_5traj_noshift_6dof_diffusionpolicy.zarr.zip"

if not os.path.exists(zarr_path):
    raise FileNotFoundError(f"Output Zarr path does not exist: {zarr_path}")

dest_store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
dest_root = zarr.group(store=dest_store)
replay_buffer = ReplayBuffer.create_from_group(group=dest_root)

def display_video(replay_buffer: ReplayBuffer, episode_index: int):
    # 1. Load an episode from the ReplayBuffer
    episode = replay_buffer.get_episode(episode_index)

    import pdb; pdb.set_trace()
    images = episode['robot0_base_rgb']  # (T, H, W, 3)
    # convert float32 images to uint8
    images = (images - images.min()) / (images.max() - images.min()) * 255.0
    images = images.astype(np.uint8)
    # display video
    for i in range(images.shape[0]):
        cv2.imshow("frame", images[i][:, :, ::-1])
        cv2.waitKey(20)

display_video(replay_buffer, 0)
import pdb; pdb.set_trace()
