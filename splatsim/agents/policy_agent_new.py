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
        # ckpt_path="/home/jennyw2/code/diffusion_policy/data/outputs/2025.11.13/16.04.06_train_diffusion_unet_hybrid_splatsim_obstacle_avoidance/checkpoints/epoch=0050-test_mean_score=0.000.ckpt",
        # residual_to_first_step: bool = False, #True,
        
        # good ish
        # ckpt_path="/home/jennyw2/code/diffusion_policy/data/outputs/2025.11.14/10.39.52_train_diffusion_unet_hybrid_splatsim_obstacle_avoidance/checkpoints/latest.ckpt",
        # residual_to_first_step: bool = False, #True,
        
        # 11/18/25 residual policy
        # ckpt_path="/home/jennyw2/code/diffusion_policy/data/outputs/2025.11.18/15.42.35_train_diffusion_unet_hybrid_splatsim_obstacle_avoidance/checkpoints/epoch=0060-test_mean_score=0.000.ckpt",
        # residual_to_first_step: bool = True,

        # 11/19/25 umi repo with 100 scenario dataset
        # ckpt_path="/home/jennyw2/code/universal_manipulation_interface/data/outputs/2025.11.19/11.39.12_train_diffusion_unet_timm_splatsim_umi/checkpoints/epoch=0050-train_loss=0.035.ckpt",
        # residual_to_first_step: bool = False,

        # 12/1/25 umi repo with 5traj dataset
        # ckpt_path="/home/jennyw2/code/universal_manipulation_interface/data/outputs/2025.12.01/14.23.13_train_diffusion_unet_timm_splatsim_umi/checkpoints/epoch=0051-train_loss=0.039.ckpt",
        # residual_to_first_step: bool = False,

        # 12/1/25 umi repo with 20 traj simple dataset
        # ckpt_path="/home/jennyw2/code/universal_manipulation_interface/data/outputs/2025.12.01/18.02.45_train_diffusion_unet_timm_splatsim_umi/checkpoints/epoch=0009-train_loss=0.074.ckpt",
        # but with the images rescaled to 224 224 for the training dataset
        ckpt_path="/home/jennyw2/code/universal_manipulation_interface/data/outputs/2025.12.01/19.10.44_train_diffusion_unet_timm_splatsim_umi/checkpoints/epoch=0025-train_loss=0.046.ckpt",
        residual_to_first_step: bool = False,

        image_names=["base_rgb"],
    ):
        # TODO later put the step size to 1 when using a better machine
        self.ckpt_path = ckpt_path
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6
        self.image_names = image_names
        self.obs_buffers = {**{key: [] for key in self.image_names}, **{"agent_pos": []}}

        self.residual_to_first_step = residual_to_first_step

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.obs_buffer = defaultdict(lambda: [])
        self.save_images = save_images
        self.last_action = None

        self.policy = None
        self.policy_cfg = None
        self.load_policy()

        self.env._robot.enable_rendering()

        self.pred_trajectory_buffer = []
        self.pred_trajectory_buffer_i = float("inf")

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
            16 - 2 + 1
            # self.policy.horizon - self.policy.n_obs_steps + 1
        )
        self.target_obs_buffer_len = 2

        # self.policy.reset()
        print("Successfully loaded policy")

    def get_obs_for_policy(self):
        # TODO this format can be taken from the configs probably
        obs = {key: torch.tensor(np.stack(self.obs_buffer[key])[None], device="cuda") for key in self.obs_buffer}
        actual_obs = {
            "robot0_base_rgb": obs["base_rgb"],
            "robot0_qs": obs["agent_pos"][:, :, :6],
            "robot0_gripper_width": obs["agent_pos"][:, :, 6:7],
        }
        return actual_obs
    
    def add_obs_to_buffer(self, obs):
        # The buffers have the oldest image at the 0th index
        # Add the image observations and also the agent pose (joint_positions)
        self.obs_buffer["agent_pos"].append(obs["joint_positions"])
        for image_name in self.image_names:
            self.obs_buffer[image_name].append(obs[image_name])
        if len(self.obs_buffer[self.image_names[0]]) > self.target_obs_buffer_len:
            for obs_name in self.obs_buffer.keys():
                self.obs_buffer[obs_name].pop(0)

    def act(self, obs):
        # Fill the time horizon buffer until it represents the full time horizon
        while len(self.obs_buffer[self.image_names[0]]) < self.target_obs_buffer_len - 1:
            print("doing init obs")
            self.add_obs_to_buffer(obs)

        self.add_obs_to_buffer(obs)
        
        if self.pred_trajectory_buffer_i >= 1 * len(self.pred_trajectory_buffer):
            policy_obs = self.get_obs_for_policy()
            print("obs", policy_obs['robot0_qs'][:, :, 0])
            action = self.policy.predict_action(policy_obs)
            # action_pred is (1, 16, 7). Take the 7 DOF angles
            self.pred_trajectory_buffer = action["action"][0].detach().cpu().numpy()
            print("action", self.pred_trajectory_buffer[:, 0])
            if self.residual_to_first_step:
                # TODO make sure that the action really is in residual form
                import pdb; pdb.set_trace()
                self.pred_trajectory_buffer[1:] += self.pred_trajectory_buffer[0]
            self.pred_trajectory_buffer_i = 0

            # This does not help. the similarity scores are bad
            # compute cosine similarity between self.last_action and each step of self.pred_trajectory_buffer
            # if self.last_action is not None:
            #     sim = [np.linalg.norm(np.array(self.last_action) - self.pred_trajectory_buffer[i]) for i in range(len(self.pred_trajectory_buffer))]
            #     # sim = [np.array(self.last_action) @ self.pred_trajectory_buffer[i] / (np.linalg.norm(self.last_action) * np.linalg.norm(self.pred_trajectory_buffer[i]) + 1e-8) for i in range(len(self.pred_trajectory_buffer))]
            #     sim_i = np.argmax(sim)
            #     print(f"Similarity index: {sim_i}, sims: {sim}")
            #     self.pred_trajectory_buffer_i = sim_i
            print("update!")

        angles = self.pred_trajectory_buffer[self.pred_trajectory_buffer_i]
        self.pred_trajectory_buffer_i += 1

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

        if len(angles) == 6:
            # add the gripper
            angles = np.concatenate([angles, [0]])

        self.last_action = angles

        return angles

    # To convert the folder of images to a video, run this ffmpeg command:
    # ffmpeg -framerate 4 -i output/test_images/base_rgb_%05d.png -c:v libx264 -pix_fmt yuv420p output/test_images/out.mp4
    # rm output/test_images/base_rgb_*.png
