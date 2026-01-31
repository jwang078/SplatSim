import numpy as np
import pybullet_data
from splatsim.agents.agent import Agent
import os
from gello.env import RobotEnv
import zarr
import re
import json
from collections import defaultdict
from splatsim.utils.agent_state_utils import AGENT_STATE

class ReplayZarrTrajectoryAgent(Agent): 
    NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS = 50

    def __init__(self, traj_folder: str, env: RobotEnv, save_images: bool = False, step_size=1):
        # TODO later put the step size to 1 when using a better machine
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6
        self.step_size = step_size
        self.image_buffers = defaultdict(lambda: [])


        self.settle_steps_remaining = self.NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.state = AGENT_STATE.SETTLING
        self.env._robot.disable_rendering()

        self.last_action = None

        self.traj_folder = traj_folder
        self.save_images = save_images
        self.traj_index = -1 # for testing purposes, start with one with an obstacle
        self.t = 0

        assert traj_folder.endswith('.zarr'), "Currently only .zarr trajectory folder is supported."
        # Load the entire zarr file into memory
        self.scenarios_groups = zarr.open(traj_folder, mode='a')['trajectories']
        scenario_re = re.compile(r"^scenario_(\d+)$")
        existing_ids = []
        for name in self.scenarios_groups.keys():
            m = scenario_re.match(name)
            if m:
                existing_ids.append(int(m.group(1)))

        # Get the list of traj_xxxx subfolders that exist in self.trajectories and use regex to get only those folders
        existing_ids.sort()
        self.trajectories = []
        for scenario_id in existing_ids:
            scenario_name = f'scenario_{scenario_id:04d}'
            scenarios_group = self.scenarios_groups[scenario_name]

            obstacle_re = re.compile(r"^obstacle_config_(\d+)$")
            for obstacle_name in scenarios_group.keys():
                if obstacle_re.match(obstacle_name):
                    obstacle_config = scenarios_group[obstacle_name]
                    # obstacle config is inside .zattrs
                    obstacle_config_json = json.loads(obstacle_config.attrs['metadata'])

                    traj_re = re.compile(r"^traj_(\d+)$")
                    for trajs_name in obstacle_config.keys():
                        if traj_re.match(trajs_name):
                            traj_group = obstacle_config[trajs_name]
                            qs = np.array(traj_group['qs'])
                            assert qs.ndim == 2
                            
                            self.trajectories.append({
                                "qs": qs,
                                "metadata": obstacle_config_json,
                                "path_type": "modified_traj",
                                "name": f"{scenario_name}_{obstacle_name}_{trajs_name}",
                                "zarr_group": traj_group,
                            })
        self.loaded_obstacle_names = []

        self.load_next_recorded_trajectory()

    def load_next_recorded_trajectory(self):
        # Clean up the previous trajectory
        # This clears everything, even if the object was created by another script
        deleted_obj_names = self.env._robot.clear_temp_objects()

        for obstacle_name in self.loaded_obstacle_names:
            if obstacle_name not in deleted_obj_names:
                # Still have to clean it up
                self.env._robot.delete_object(obstacle_name)
        
        if self.save_images:
            for key in self.image_buffers.keys():
                if len(self.image_buffers[key]) > 0:
                    # Create the zarr dataset
                    zarr_group = self.trajectories[self.traj_index]["zarr_group"]
                    if key in zarr_group:
                        del zarr_group[key] # replace if re-running
                    # images_group = zarr_group.create_group("images")
                    images = np.stack(self.image_buffers[key], axis=0)
                    zarr_group.create_dataset(key, data=images, dtype="f4")
                self.image_buffers[key] = []

        self.traj_index += 1
        self.t = 0

        # Load new trajectory

        # Load new static obstacles
        self.loaded_obstacle_names = []

        obstacle_config_json = self.trajectories[self.traj_index]["metadata"]
        path_name = self.trajectories[self.traj_index]["name"]
        for i, obstacle in enumerate(obstacle_config_json['obstacles']):
            if obstacle['type'] == "cuboid":
                # has pos, orn, and size attributes
                obstacle_config = {
                    "object_type": obstacle['type'],
                    "position": obstacle["pos"],
                    "orientation": obstacle["orn"],
                    "size": obstacle["size"],
                }
                obstacle_name = f"{path_name}_cuboid{i}"
                self.env._robot.create_object(obstacle_name, obstacle_config, use_gravity=False)
                self.loaded_obstacle_names.append(obstacle_name)
            else:
                raise ValueError(f"Unknown obstacle type {obstacle['type']}")

    def next_trajectory_step(self):
        if self.last_action is not None and self.state == AGENT_STATE.SETTLING:
            if self.settle_steps_remaining > 0:
                self.settle_steps_remaining -= 1
                if self.settle_steps_remaining == 0:
                    # Tell the simulator to get the rendering for the next timestep ready
                    self.env._robot.enable_rendering()
            else:
                # Finished settling, switch to executing trajectory
                # TODO check for off by one errors
                self.state = AGENT_STATE.EXECUTING_TRAJ
                self.env._robot.enable_rendering()
                print("Shifting from settling to executing")
            cur_joint = self.last_action
        elif self.last_action is None or self.state == AGENT_STATE.EXECUTING_TRAJ:
            if self.traj_index >= len(self.trajectories):
                return None

            if self.t >= len(self.trajectories[self.traj_index]['qs']):
                self.load_next_recorded_trajectory()
                if self.traj_index >= len(self.trajectories):
                    return None  # No more trajectory steps available
                self.state = AGENT_STATE.SETTLING
                # TODO check for off by one errors
                self.settle_steps_remaining = self.NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS
                self.env._robot.disable_rendering()
                print("Shifting from executing to settling")
                
            # TODO save it corrrectly so it doesn't need this :7
            cur_joint = self.trajectories[self.traj_index]['qs'][self.t, :7]
            cur_joint = cur_joint.tolist()
            cur_joint = np.array(cur_joint)
            self.t += 1
        else:
            raise ValueError(f"Unknown state {self.state}")
        
        if cur_joint.shape[0] == 6:
            cur_joint = np.append(cur_joint, 0) # assume gripper is in the open position

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
                    self.image_buffers[image_name].append(frame)
            self.last_action = angles
            return angles
        
    # To convert the folder of images to a video, run this ffmpeg command:
    # ffmpeg -framerate 4 -i output/test_images/base_rgb_%05d.png -c:v libx264 -pix_fmt yuv420p output/test_images/out.mp4
    # rm output/test_images/base_rgb_*.png
