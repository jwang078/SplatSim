import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from splatsim.agents.agent import Agent
from splatsim.robots.dynamixel import DynamixelRobot

import torch
import dill
import hydra
# from diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy import DiffusionUnetImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import pybullet as p
import cv2
import matplotlib.pyplot as plt
import copy

@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB1", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
    #     joint_ids=(1, 2, 3, 4, 5, 6, 7),
    #     joint_offsets=(
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         -1 * np.pi / 2 + 2 * np.pi,
    #         1 * np.pi / 2,
    #         1 * np.pi / 2,
    #     ),
    #     joint_signs=(1, 1, 1, 1, 1, 1, 1),
    #     gripper_config=(8, 279, 279 - 50),
    # ),
    # panda
    # "/dev/cu.usbserial-FT3M9NVB": DynamixelRobotConfig(
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            3 * np.pi / 2,
            2 * np.pi / 2,
            1 * np.pi / 2,
            4 * np.pi / 2,
            -2 * np.pi / 2 + 2 * np.pi,
            3 * np.pi / 2,
            4 * np.pi / 2,
        ),
        joint_signs=(1, -1, 1, 1, 1, -1, 1),
        gripper_config=(8, 195, 152),
    ),
    # Left UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2 + 0 * np.pi,
            0 * np.pi + np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
    ),
    # Right UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi + 0 * np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
    ),
    # Custom UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISNEF-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
         3*np.pi/2, 6*np.pi/2, 4*np.pi/2, 4*np.pi/2, 1*np.pi/2, 1*np.pi/2 
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 110, 68),
    ),
}

class DiffusionAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        # self.policy = 
        print('Loading policy')

        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.07.16/00.55.58_train_diffusion_unet_image_real_image/checkpoints/epoch=0150-train_loss=0.009.ckpt'
        # ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.07.17/23.07.05_train_diffusion_unet_image_real_image/checkpoints/epoch=0250-train_loss=0.007.ckpt'
        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.07.23/13.49.47_train_diffusion_unet_image_real_image/checkpoints/epoch=0150-train_loss=0.010.ckpt'

        # ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.07.30/02.38.06_train_diffusion_unet_image_real_image/checkpoints/epoch=0150-train_loss=0.011.ckpt'
        # ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.08.17/16.47.40_train_diffusion_unet_image_real_image/checkpoints/epoch=0550-train_loss=0.002.ckpt'
        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.08.18/23.47.50_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt'
        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.08.24/18.58.30_train_diffusion_unet_image_real_image/checkpoints/epoch=0300-train_loss=0.002.ckpt'
        
        ############### Final Checkpoint for apple picking ###############
        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.08.26/12.35.48_train_diffusion_unet_image_real_image/checkpoints/epoch=0250-train_loss=0.002.ckpt'
        ##################################################################


        ############### Final Checkpoint for orange on plate ###############
        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.08.31/04.12.24_train_diffusion_unet_image_real_image/checkpoints/epoch=0300-train_loss=0.002.ckpt'
        ####################################################################

        ############### Real world dataset apple picking ###############
        # ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.09.11/03.00.47_train_diffusion_unet_image_real_image/checkpoints/epoch=0250-train_loss=0.008.ckpt'
        ####################################################################

        ckpt_path = '/home/nomaan/Desktop/corl24/main/diffusion_policy/diffusion_policy/data/outputs/2024.09.11/10.07.03_train_diffusion_unet_image_real_image_assembly/checkpoints/epoch=0550-train_loss=0.001.ckpt'


        # 96x96 image
        ckpt_path = "/home/jennyw2/code/diffusion_policy/data/outputs/2025.08.25/18.25.55_train_diffusion_unet_hybrid_splatsim_object_on_plate/checkpoints/epoch=0100-test_mean_score=0.000.ckpt"
        # 240x320 image <- this does not even run
        # ckpt_path = "/home/jennyw2/data/diffusion_policy/checkpoints/epoch=0050-test_mean_score=0.000.ckpt"

        self.DOF = 7

        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        if 'diffusion' in cfg.name:
            # diffusion model
            self.policy: BaseImagePolicy
            self.policy = workspace.model
            if cfg.training.use_ema:
                self.policy = workspace.ema_model


            device = torch.device('cuda')
            self.policy.eval().to(device)

            # set inference params
            self.policy.num_inference_steps = 16 # DDIM inference iterations
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1
        
        p.connect(p.DIRECT)
        # TOOD put this in configs
        self.dummy_robot = p.loadURDF("/home/jennyw2/code/SplatSim/splatsim/robot_definitions/urdf/sisbot.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.dummy_robot, [0, 0, -0.1], [0, 0, 0, 1])
        
        p.setGravity(0, 0, -9.81)
        # p.setRealTimeSimulation(1)
        p.setTimeStep(1/240)
        #set initial joint positions
        initial_joint_state = [0, -1.57, 1.57, -1.57, -1.57, 0]
        self.initial_joint_state = initial_joint_state
        joint_signs = [1, 1, 1, 1, 1, 1]
        for i in range(1, self.DOF):
            p.resetJointState(self.dummy_robot, i, initial_joint_state[i-1]*joint_signs[i-1])
        # p.stepSimulation()    


        ee_pos, ee_quat = p.getLinkState(self.dummy_robot, 6)[0], p.getLinkState(self.dummy_robot, 6)[1]
        self.correct_ee_quat = ee_quat

        self.cur_index = -1
        self.cur_joint_list = None

        self.last_image_obs = None
        self.last_state_obs = None

        self.policy.reset()

        print('policy loaded')
        self.cur_total_steps = 0


    def act(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        '''
        obs_dict: must include "obs" key
        '''

        #set joint positions to the pybullet robot
        for i in range(1, self.DOF):
                p.resetJointState(self.dummy_robot, i, obs_dict['joint_positions'][i-1])
        #get end effector pose from the pybullet robot
        ee_pos, ee_quat = p.getLinkState(self.dummy_robot, 6)[0], p.getLinkState(self.dummy_robot, 6)[1]
        ee_euler = p.getEulerFromQuaternion(ee_quat)
        obs_dict['state'] = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_euler[0], ee_euler[1], ee_euler[2], obs_dict['gripper_position'][0]])
        print('gripper position true:', obs_dict['gripper_position'][0])
        

        #resize the image to 480x640x3 to 240x320x3
        # target size
        def resize_and_center_crop(img, image_width, image_height):
            target_w, target_h = image_width, image_height
            h, w = img.shape[:2]

            # compute scaling factor to fill target (like ImageOps.fit)
            scale = max(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # resize with LANCZOS
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # for some reason, this interpolation gives values outside of the range [0, 1]
            # TODO does this need to go to the dataset generation script, too?
            # resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))

            # center crop
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

            return cropped
        # image = resize_and_center_crop(np.transpose(obs_dict['base_rgb'].detach().cpu().numpy()), 92, 92)
        image = resize_and_center_crop(np.transpose(obs_dict['base_rgb'].detach().cpu().numpy().transpose(0, 2, 1)), 92, 92)

        # image = cv2.resize(np.transpose(obs_dict['base_rgb'].detach().cpu().numpy(), (1, 2, 0)), (320, 240))
        # image = cv2.resize(obs_dict['wrist_rgb'], (320, 240))

        # TODO generalize this code to any history length (not just 2)
    
        # plt.imsave('image.png', image)
        #make image sharper
        # TODO what is this additional noise o-o
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()/255.0
        # Apparently this was really important. tho this also has to be added to the training data
        # image = image + 0.1*torch.randn_like(image)

        image_2 = image
        # image_2 = cv2.resize(obs_dict['base_rgb'], (320, 240))
        
        # plt.imsave('image_2.png', image_2)
        # image_2 = image_2.transpose(2, 0, 1)
        # image_2 = np.expand_dims(image_2, axis=0)
        # image_2 = np.expand_dims(image_2, axis=0)
        # image_2 = torch.from_numpy(image_2).float()/255.0
        # image_2 = image_2 + 0.1*torch.randn_like(image_2)

        if self.last_image_obs is None:
            self.last_image_obs = image
            self.last_image_obs_1 = image_2
            self.last_state_obs = obs_dict['state'][:].reshape(1, 1, self.DOF)
            self.last_state_obs = torch.from_numpy(self.last_state_obs).float()



        cur_state_obs = obs_dict['state'][:].reshape(1, 1, self.DOF)
        cur_state_obs = torch.from_numpy(cur_state_obs).float()


        image_out = torch.cat(( self.last_image_obs, image,  ), dim=1)
        image_out_1 = torch.cat(( self.last_image_obs_1, image_2,  ), dim=1)
        state_out = torch.cat((  self.last_state_obs, cur_state_obs, ), dim=1)
            
        
            
        # If new action sequence needs to be predicted, predict it.
        # Otherwise, return the next action in the preplaned sequence
        if self.cur_index == -1 or self.cur_joint_list is None:
        # if True:  
            print('new step')
            obs_dict_1 = {
                'image':  image_out,
                'agent_pos': state_out
            }

            # obs_dict_1 = {
            #     'camera_1' :  image_out,
            #     'camera_2' :  image_out_1,
            #     'robot_eef_pose': state_out
            # }
            result = self.policy.predict_action(obs_dict_1)
            
            # result = {
            #     'action': action,
            #     'action_pred': action_pred
            # }
            
            # Return the first action in the sequence and store the rest for later
            self.cur_index = 0
            self.cur_joint_list = []
            # Shape of result['action']: (batch=1, horizon=15, action_dim=7)
            # This gets rid of the batch dimension
            self.cur_joint_list_1 = result['action'][0].detach().cpu().numpy()
            #reverse the order of the joints
            # Takes the first 3 timesteps of the predicted plan
            # Not sure what -12 is from
            for i in range(0, len(self.cur_joint_list_1)-12):
            # for i in range(4):
                for k in range(1):
                    # self.cur_joint_list.append(copy.deepcopy(out_1))
                    self.cur_joint_list.append(self.cur_joint_list_1[i])
                
            #reverse the order of the joints
                    
            

        # If self.cur_index == 1 (because cur_joint_list takes the first 3 steps)
        # So does this mean it happens every other step?
        if True:
        # This was the original
        # if self.cur_index == len(self.cur_joint_list) - 2:
            self.last_image_obs = copy.deepcopy(image)
            self.last_image_obs_1 = copy.deepcopy(image_2)
            self.last_state_obs = copy.deepcopy(cur_state_obs)
        
        action_pred = self.cur_joint_list[self.cur_index]


        # if action_pred[2] < 0.235:
        #     action_pred[2] = 0.235

        pred_mode = "ee_pose"
        if pred_mode == "ee_pose":
            # The diffusion policy was predicting xyzrpy for rotation and translation of the end effector
            # Not joint angles
            ee_pose = [action_pred[0], action_pred[1], action_pred[2]] 
            ee_quat = p.getQuaternionFromEuler( action_pred[3:6])
            print('ee_pose:', ee_pose, 'ee_quat:', ee_quat)

            # ee_pose = [action_pred[0]*0.5 + obs_dict['state'][0]*0.5, action_pred[1]*0.5 + obs_dict['state'][1]*0.5, 0.095]

            # Convert end effector pose to joint angles using inverse kinematics
            dummy_joint_pos = p.calculateInverseKinematics(self.dummy_robot, 6, ee_pose , ee_quat,
                residualThreshold=0.00001, maxNumIterations=100000, 
                # lowerLimits=[self.initial_joint_state[k] - np.pi/2 for k in range(6)],
                lowerLimits=[obs_dict['joint_positions'][k] - np.pi for k in range(6)],
                # upperLimits=[self.initial_joint_state[k] + np.pi/2 for k in range(6)],
                upperLimits=[obs_dict['joint_positions'][k] + np.pi for k in range(6)],
                jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
                restPoses=[0* np.pi, -0.5* np.pi, 0.5* np.pi, -0.5* np.pi, -0.5* np.pi, 0]
                )
            
            # check the error between the end effector pose and the calculated pose
            for i in range(1, self.DOF):
                p.resetJointState(self.dummy_robot, i, dummy_joint_pos[i-1])
            new_ee_pos, new_ee_quat = p.getLinkState(self.dummy_robot, 6)[0], p.getLinkState(self.dummy_robot, 6)[1]
            new_ee_euler = p.getEulerFromQuaternion(new_ee_quat)
            new_ee_pos = np.array(new_ee_pos)
            # print('ee_pos:', new_ee_pos, 'ee_quat:', new_ee_quat, 'ee_euler:', new_ee_euler)
            # print('target_ee_pos:', ee_pose, 'target_ee_quat:', ee_quat, 'target_ee_euler:', action_pred[3:6])

            # print('error in ee pos:', np.linalg.norm(np.array(new_ee_pos) - np.array(ee_pose)))
            # print('error in ee euler:', np.linalg.norm(np.array(new_ee_euler) - np.array(action_pred[3:6])))
                
            # calculate difference between current and target joint angles
            joint_diff = np.array(dummy_joint_pos)[:6] - np.array(obs_dict['joint_positions'])[:6]
            self.cur_total_steps += 1
            # Maybe this is: if the IK fails 400 times in a row, reset the diffusion model
            # if self.cur_total_steps > 400:
            #     self.policy.reset()
            if np.linalg.norm(joint_diff) < 0.01 :
                self.cur_index += 1
                self.cur_total_steps = 0
        elif pred_mode == "joint_angles":
            print("action pred:", action_pred)
            dummy_joint_pos = action_pred
            self.cur_index += 1
            self.cur_total_steps = 0
        else:
            raise ValueError('pred_mode must be either ee_pose or joint_angles')

        if self.cur_index == len(self.cur_joint_list):
        # if True:
            self.cur_index = -1
            self.cur_joint_list = None

        joints = np.array(dummy_joint_pos)[:6]
        # joints = np.array([1.5681470689206045, -1.068216007103522, 2.1378836578411438, -2.6390424613000025, -1.5699116232851198, -0.0018527878551533776])
        # Append gripper action
        joints = np.append(joints, action_pred[-1])

        return joints   



if __name__ == "__main__":
    demo_agent = DiffusionAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0")
    obs = {
        "image": np.zeros((1, 1,  3, 240, 320)),
        "agent_pos": np.zeros((1, 4, 2)),
        # "joint_positions": np.array([0, 0, 0, 0, 0, 0, 0]),
        # "joint_velocities": np.array([0, 0, 0, 0, 0, 0, 0]),
        # "ee_pos_quat": np.zeros(7),
        # "gripper_position": np.array([0]),
    }

    action = demo_agent.act(obs)
    print('Action:', action)