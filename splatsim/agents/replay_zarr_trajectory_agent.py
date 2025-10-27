import numpy as np
from typing import List
import pybullet as p
import pybullet_data
from splatsim.agents.agent import Agent
import pickle
import os
from tqdm import tqdm
from gello.env import RobotEnv
import cv2
import zarr
import re
import json

class ReplayZarrTrajectoryAgent(Agent): 
    def __init__(self, traj_folder: str, env: RobotEnv, save_images: bool = False, step_size=5):
        # TODO later put the step size to 1 when using a better machine
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6
        self.step_size = step_size

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.last_action = np.array([0, 0, 0, 0, 0, 0, 1])  # 7-DoF

        self.traj_folder = traj_folder
        traj_folder_basename = ".".join(os.path.basename(traj_folder).split(".")[:-1])
        self.image_folder = os.path.join("output", traj_folder_basename + '_images')
        if save_images:
            os.makedirs(self.image_folder, exist_ok=True)
        self.save_images = save_images
        self.traj_index = 0
        self.t = 0

        assert traj_folder.endswith('.zarr'), "Currently only .zarr trajectory folder is supported."
        # Load the entire zarr file into memory
        self.trajs_groups = zarr.open(traj_folder, mode='r')['trajectories']
        traj_re = re.compile(r"^traj_(\d+)$")
        existing_ids = []
        for name, node in self.trajs_groups.items():
            if isinstance(node, zarr.hierarchy.Group):
                m = traj_re.match(name)
                if m:
                    existing_ids.append(int(m.group(1)))

        # Get the list of traj_xxxx subfolders that exist in self.trajectories and use regex to get only those folders
        existing_ids.sort()
        self.trajectories = []
        for traj_id in existing_ids:
            traj_name = f'traj_{traj_id:04d}'
            trajs_group = self.trajs_groups[traj_name]
            base_q = trajs_group['base_q'][:]
            self.trajectories.append({
                "qs": base_q,
                "metadata": {},
                "path_type": "base_traj",
                "name": f"{traj_name}_no_obstacles"
            })
            
            obstacle_re = re.compile(r"^obstacle_config_(\d+)$")
            for obstacle_name, node in trajs_group.items():
                m = obstacle_re.match(obstacle_name)
                if m:
                    obstacle_config = trajs_group[obstacle_name]

                    modified_q = np.array(obstacle_config["modified_q"]).reshape(-1, 120, 7) # TODO test if this is right
                    # obstacle config is inside attrs
                    obstacle_config_json = json.loads(obstacle_config.attrs['metadata'])
                    for path_i in range(len(modified_q)):
                        self.trajectories.append({
                            "qs": modified_q[path_i],
                            "metadata": obstacle_config_json,
                            "path_type": "modified_traj",
                            "name": f"{traj_name}_{obstacle_name}_{path_i+1}"
                        })

            path_names = [traj["name"] for traj in self.trajectories]

    def load_next_recorded_trajectory(self):
        self.traj_index += 1
        self.t = 0

    def next_trajectory_step(self):
        if self.traj_index >= len(self.trajectories):
            return None
        
        traj = np.array(self.trajectories[self.traj_index]['qs'])

        if self.t >= len(traj):
            self.load_next_recorded_trajectory()
            if self.traj_index >= len(self.trajectories):
                return None  # No more trajectory steps available
            
        # TODO save it corrrectly so it doesn't need this :7
        
        cur_joint = traj[self.t, :7]
        cur_joint = cur_joint.tolist()
        # Add the world joint to the recorded joint state
        # cur_joint = [0] + cur_joint 
        cur_joint = np.array(cur_joint)
        self.t += 1 * 5

        # TODO Other objects not supported at the moment
        # object_list = [object_position_key[:-len("_position")] for object_position_key in data.keys() if object_position_key.endswith("_position")]
        # # gripper_position is for gello integration. Ignore this value
        # if "gripper" in object_list:
        #     object_list.remove("gripper")
        # for object_name in object_list:
        #     cur_object_position = np.array(data[object_name + '_position'])
        #     cur_object_rotation = np.array(data[object_name + '_orientation'])
        #     # cur_object_rotation = np.roll(cur_object_rotation, 1)
        #     # Disable gravity for objects when replaying a trajectory so that there's no jittering
        #     self.env._robot.set_object_pose(object_name, cur_object_position, cur_object_rotation, use_gravity=False)

        return cur_joint

    def act(self, obs):
        angles = self.next_trajectory_step()
        if angles is None:
            print("No more trajectory steps available.")
            return self.last_action
        else:
            if self.save_images:
                for image_name in [image_name for image_name in obs.keys() if image_name.endswith("_rgb") and obs[image_name] is not None]:
                    frame = obs[image_name]
                    frame = np.transpose(frame.detach().cpu().numpy(), (1, 2, 0))  # CxHxW -> HxWxC
                    frame = (frame * 255).astype(np.uint8)
                    image_index = len(os.listdir(self.image_folder))
                    image_path = os.path.join(self.image_folder, f"{image_name}_{image_index:05d}.png")
                    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.last_action = angles
            return angles
        
    # To convert the folder of images to a video, run this ffmpeg command:
    # ffmpeg -framerate 4 -i output/test_images/base_rgb_%05d.png -c:v libx264 -pix_fmt yuv420p output/test_images/out.mp4
    # rm output/test_images/base_rgb_*.png
