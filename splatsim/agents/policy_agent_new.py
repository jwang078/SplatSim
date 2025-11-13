import numpy as np
import pybullet_data
from splatsim.agents.agent import Agent
import os
from gello.env import RobotEnv
import zarr
import re
import json
from collections import defaultdict

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import torch
import dill
import hydra

class DiffusionPolicyAgent(Agent):
    def __init__(
        self,
        env: RobotEnv,
        save_images: bool = False,
        ckpt_path="/home/jennyw2/code/diffusion_policy/data/outputs/2025.11.13/16.04.06_train_diffusion_unet_hybrid_splatsim_obstacle_avoidance/checkpoints/epoch=0050-test_mean_score=0.000.ckpt",
        image_names=["base_rgb"],
    ):
        # TODO later put the step size to 1 when using a better machine
        self.ckpt_path = ckpt_path
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6
        self.image_names = image_names
        self.obs_buffers = {**{key: [] for key in self.image_names}, **{"agent_pos": []}}

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.obs_buffer = defaultdict(lambda: [])
        self.save_images = save_images

        self.policy = None
        self.policy_cfg = None
        self.load_policy()

    def load_policy(self):
        print("Loading policy from:", self.ckpt_path)
        payload = torch.load(open(self.ckpt_path, "rb"), pickle_module=dill)
        self.policy_cfg = payload["cfg"]
        cls = hydra.utils.get_class(self.policy_cfg._target_)
        workspace = cls(self.policy_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # diffusion model
        self.policy: BaseImagePolicy
        self.policy = workspace.model
        if self.policy_cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device("cuda")
        self.policy.eval().to(device)

        # set inference params
        self.policy.num_inference_steps = 16  # DDIM inference iterations
        self.policy.n_action_steps = (
            self.policy.horizon - self.policy.n_obs_steps + 1
        )

        self.policy.reset()
        print("Successfully loaded policy")

    def get_obs_for_policy(self):
        # TODO this format can be taken from the configs probably
        # TODO uh is it true that you don't need normalization on the observations? is it done internally?
        return {key: np.stack(self.obs_buffer[key])[None] for key in self.obs_buffer}

    def next_trajectory_step(self):
        if self.traj_index >= len(self.trajectories):
            return None

        traj = np.array(self.trajectories[self.traj_index]["qs"])

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

        return cur_joint
    
    def add_obs_to_buffer(self, obs):
        # The buffers have the oldest image at the 0th index
        # Add the image observations and also the agent pose (joint_positions)
        self.obs_buffer["agent_pos"].append(obs["joint_positions"])
        for image_name in self.image_names:
            self.obs_buffer[image_name].append(obs[image_name])
        if len(self.obs_buffer[self.image_names[0]]) > self.policy_cfg.horizon:
            for obs_name in self.obs_buffer.keys():
                self.obs_buffer[obs_name].pop(0)

    def act(self, obs):
        # Fill the time horizon buffer until it represents the full time horizon
        while not len(self.obs_buffer[self.image_names[0]]) == self.policy_cfg.horizon:
            self.add_obs_to_buffer(obs)
        
        policy_obs = self.get_obs_for_policy()
        action = self.policy.predict_action(policy_obs)
        # has keys action and action_pred
        # action_pred is (1, 16, 7). Take the 7 DOF angles
        angles = action["action_pred"][0, 1].detach().cpu().numpy()

        if self.save_images:
            for image_name in [
                image_name
                for image_name in obs.keys()
                if image_name.endswith("_rgb") and obs[image_name] is not None
            ]:
                frame = obs[image_name]
                frame = np.transpose(
                    frame.detach().cpu().numpy(), (1, 2, 0)
                )  # CxHxW -> HxWxC
                frame = (frame * 255).astype(np.uint8)
                self.obs_buffers[image_name].append(frame)

        self.add_obs_to_buffer(obs)

        return angles

    # To convert the folder of images to a video, run this ffmpeg command:
    # ffmpeg -framerate 4 -i output/test_images/base_rgb_%05d.png -c:v libx264 -pix_fmt yuv420p output/test_images/out.mp4
    # rm output/test_images/base_rgb_*.png
