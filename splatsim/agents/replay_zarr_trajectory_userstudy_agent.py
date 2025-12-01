import numpy as np
import pybullet_data
from splatsim.agents.agent import Agent
import os
from gello.env import RobotEnv
import zarr
import re
import json
from collections import defaultdict
import glob
import time

from splatsim.agents.gello_agent import GelloAgent

from pynput import keyboard

from dataclasses import dataclass

def get_answer_on_1_to_5_scale(question):
    while True:
        answer = input(question + " (1-5): ")
        try:
            value = int(answer)
            if 1 <= value <= 5:
                return value
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")

def get_answer_y_or_n(question):
    while True:
        answer = input(question + " (y/n): ")
        value = answer.strip().lower()
        if value in ['y', 'n']:
            return value
        else:
            print("Please enter either y or n.")

class ReplayZarrTrajectoryUserStudyAgent(Agent): 
    NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS = 100

    @dataclass
    class STATE:
        EXECUTING_TRAJ: str = "EXECUTING_TRAJ"
        SETTLING: str = "SETTLING"
        GELLO: str = "GELLO"
        QUITTING_GELLO: str = "QUITTING_GELLO"

    def __init__(self, traj_folder: str, env: RobotEnv, save_images: bool = False, step_size=3,
                 gello_port=None, gello_start_joints=None):
        # TODO later put the step size to 1 when using a better machine
        self.robot = None
        # TODO does this need to be set?
        self.joint_signs = [1] * 6
        self.step_size = step_size
        self.image_buffers = defaultdict(lambda: [])

        self.gello_traj_to_save = []

        self.gello_port = gello_port
        self.gello_start_joints = gello_start_joints
        self.init_gello()

        self.user_name = input("Name? ").strip()

        self.settle_steps_remaining = self.NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS

        # env is using for setting the pose of recorded objects in the scene
        self.env = env

        self.state = self.STATE.SETTLING
        self.env._robot.disable_rendering()
        self.env._robot.clear_temp_objects()

        self.last_action = self.env._robot.get_joint_state()

        self.traj_folder = traj_folder
        self.save_images = save_images
        self.traj_index = 2*4 - 1 #-1 # for testing purposes, start with one with an obstacle
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

            # User does gello teleop once per scenario. Save it under obstacle_config_00
            ref_traj = scenarios_group["obstacle_config_00"]['traj_00']['qs']
            self.env._robot.teleport_joint_state(ref_traj[-1:].squeeze())
            ee_pos, ee_quat = self.env._robot.get_ee_pos()
            self.trajectories.append({
                "qs": None,
                # Put a box where the end effector will be
                "metadata": {"obstacles": [{
                    "type": "cuboid",
                    "pos": list(ee_pos),
                    "orn": [0, 0, 0, 1],
                    "size": (0.1, 0.1, 0.1),
                }]},
                "path_type": "gello_traj",
                "ref_traj": ref_traj,
                "name": f"{scenario_name}_obstacle_config_00_traj_{self.user_name}",
                "traj_group": scenarios_group["obstacle_config_00"]['traj_00'],
                "scenario_group": scenarios_group,
            })
            # let the env itself teleport back to the original robot joint state

            # # Create the user's profile
            # if self.user_name in scenarios_group:
            #     do_overwrite = do_overwrite == True or get_answer_y_or_n("Overwrite existing save?") == "y"
            #     if do_overwrite:
            #         del scenarios_group[self.user_name]
            #     else:
            #         raise RuntimeError("Not overwriting profile")
            # scenarios_group.create_group(self.user_name)
            # scenarios_group[self.user_name].attrs['question_answers'] = json.dumps({})

            obstacle_re = re.compile(r"^obstacle_config_(\d+)$")
            for obstacle_name in sorted(scenarios_group.keys()):
                if obstacle_re.match(obstacle_name):
                    obstacle_group = scenarios_group[obstacle_name]
                    # obstacle config is inside .zattrs
                    metadata = json.loads(obstacle_group.attrs['metadata'])


                    # Save the user study questions under each of the obstacle configs
                    if "user_study" not in obstacle_group.attrs:
                        # user_study:
                        # > question_answers:
                        # > > question1:
                        # > > > user1: 3
                        # > > > user2: 2
                        # > > question2:
                        # > > > user1: 3
                        # ...
                        obstacle_group.attrs["user_study"] = {"question_answers": defaultdict(lambda: {})}

                    traj_re = re.compile(r"^traj_(\d+)$")
                    for trajs_name in obstacle_group.keys():
                        if traj_re.match(trajs_name):
                            traj_group = obstacle_group[trajs_name]
                            qs = np.array(traj_group['qs'])
                            assert qs.ndim == 2

                            user_traj_name = f"qs_{self.user_name}"
                            if user_traj_name in traj_group:
                                del traj_group[user_traj_name]
                                # Later, do a create_dataset

                            if obstacle_name == "obstacle_config_00":
                                # User sees this trajectory with default settings only (no obstalces)
                                self.trajectories.append({
                                    "qs": qs,
                                    "metadata": metadata,
                                    "path_type": "original_traj",
                                    "name": f"{scenario_name}_{obstacle_name}_{trajs_name}_basetraj",
                                    "traj_group": traj_group,
                                    "scenario_group": scenarios_group,
                                })
                            else:
                                # User sees trajectory without obstacles
                                self.trajectories.append({
                                    "qs": qs,
                                    "metadata": {"obstacles": []},
                                    "path_type": "modified_traj_no_obstacles",
                                    "name": f"{scenario_name}_{obstacle_name}_{trajs_name}_modifiednoshowobstacles",
                                    "traj_group": traj_group,
                                    "scenario_group": scenarios_group,
                                })

                                # User sees trajectory with obstacles
                                self.trajectories.append({
                                    "qs": qs,
                                    "metadata": metadata,
                                    "path_type": "modified_traj",
                                    "name": f"{scenario_name}_{obstacle_name}_{trajs_name}_modified",
                                    "traj_group": traj_group,
                                    "scenario_group": scenarios_group,
                                })
                    
        self.loaded_obstacle_names = []

        self.load_next_recorded_trajectory()

        # Initialize gello keyboard listener to quit gello mode
        self.gello_keyboard_listener = None

    def get_gello_on_press(self):
        def gello_on_press(key):
            try:
                if key.char == 'q':
                    print('Key q pressed. Quitting gello mode listener and loop.')
                    self.state = self.STATE.QUITTING_GELLO
                    # Returning False stops the pynput listener
                    return False 
            except AttributeError:
                # Handle special keys (like 'esc', 'space') which don't have a .char attribute
                # if key == keyboard.Key.esc:
                #     print('Escape pressed. Stopping listener and loop.')
                pass
        return gello_on_press

    def init_gello(self):
        gello_port = self.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                self.gello_port = usb_ports[0]
                print('all usb ports:', usb_ports)
                print(f"using port {gello_port}")
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )
        if self.gello_start_joints is None:
            reset_joints = np.deg2rad(
                [0, -90, 90, -90, -90, 0, 0]
            )  # Change this to your own reset joints
        else:
            reset_joints = self.gello_start_joints
            reset_joints = np.array(reset_joints)
        self.gello_start_joints = reset_joints

        self.gello_agent = GelloAgent(port=self.gello_port, start_joints=self.gello_start_joints)


        # assert os.path.exists(self.gello_port), self.gello_port
        # assert self.gello_port in PORT_CONFIG_MAP, f"Port {self.gello_port} not in config map"

        # config = PORT_CONFIG_MAP[self.gello_port]
        # self.gello_robot = config.make_robot(port=self.gello_port, start_joints=self.gello_start_joints)

    def save_curr_trajectory(self):
        # Clean up the previous trajectory
        # This clears everything, even if the object was created by another script
        deleted_obj_names = self.env._robot.clear_temp_objects()

        for obstacle_name in self.loaded_obstacle_names:
            if obstacle_name not in deleted_obj_names:
                # Still have to clean it up
                self.env._robot.delete_object(obstacle_name)

        if len(self.gello_traj_to_save) > 0:
            trajectory = self.trajectories[self.traj_index]
            trajectory["traj_group"].create_dataset(f"qs_{self.user_name}", data=self.gello_traj_to_save, dtype="f4")
            self.gello_traj_to_save = []
        
        if self.save_images:
            for key in self.image_buffers.keys():
                if len(self.image_buffers[key]) > 0:
                    # Create the zarr dataset
                    traj_group = self.trajectories[self.traj_index]["traj_group"]
                    if key in traj_group:
                        del traj_group[key] # replace if re-running
                    images = np.stack(self.image_buffers[key], axis=0)
                    traj_group.create_dataset(key, data=images, dtype="f4")
                self.image_buffers[key] = []

    def load_next_recorded_trajectory(self):
        self.traj_index += 1
        self.t = 0

        self.env._robot.clear_temp_objects()

        # Load new trajectory

        # Load new static obstacles
        self.loaded_obstacle_names = []
    
        metadata = self.trajectories[self.traj_index]["metadata"]
        path_name = self.trajectories[self.traj_index]["name"]
        for i, obstacle in enumerate(metadata['obstacles']):
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
            
        if self.trajectories is not None and self.trajectories[self.traj_index] is not None and self.trajectories[self.traj_index]["qs"] is not None:
            self.env._robot.teleport_joint_state(self.trajectories[self.traj_index]["qs"][0])
            
    def ask_user_questions_and_record(self, trajectory):
        print("\n\n\n")

        answer_goal_alignment = get_answer_on_1_to_5_scale(
            "Did the robot move as you intended?"
        )

        # if trajectory["path_type"] in ["modified_traj", "modified_traj_no_obstacles"]:
        #     print("\nThe robot was avoiding obstacles")
        # elif trajectory["path_type"] == "original_traj":
        #     print("\nThe robot was taking a shorter path to the goal")
        # elif trajectory["path_type"] == "gello_traj":
        #     print("\nThe robot was following your trajectory")
        # else:
        #     raise ValueError(f"Unknown trajectory path type {trajectory['path_type']}")
        
        # if trajectory["path_type"] != "user_traj":
        #     answer_trust_obstacle = get_answer_on_1_to_5_scale("To what extent did you feel the robot had reasons for deviating from your intended path, even if you couldn't see the reason?")
        # else:
        #     answer_trust_obstacle = None

        answer_safety = get_answer_on_1_to_5_scale("How safe was the robot?")

        # answer_trust_autonomous = get_answer_on_1_to_5_scale("How much would you trust this robot to execute your intended trajectory autonomously?")
        
        answers_dict = {
            "Did the robot move as you intended?": answer_goal_alignment,
            # "To what extent did you feel the robot had reasons for deviating from your intended path, even if you couldn't see the reason?": answer_trust_obstacle,
            "How safe was the robot?": answer_safety,
            # "How much would you trust this robot to execute your intended trajectory autonomously?": answer_trust_autonomous
        }

        scenario_attrs = trajectory['scenario_group'].attrs
        if 'question_answers' not in scenario_attrs:
            scenario_attrs["question_answers"] = {}
        answers_json = scenario_attrs["question_answers"]
        if trajectory["name"] not in answers_json:
            answers_json[trajectory["name"]] = {}
        if self.user_name not in answers_json[trajectory["name"]]:
            answers_json[trajectory["name"]][self.user_name] = {}
        answers_json[trajectory["name"]][self.user_name] = answers_dict
        trajectory['scenario_group'].attrs['question_answers'] = answers_json

    def next_trajectory_step(self):
        if self.last_action is not None and self.state == self.STATE.SETTLING:
            if self.settle_steps_remaining > 0:
                self.settle_steps_remaining -= 1
            else:
                # Finished settling, switch to executing trajectory
                # TODO check for off by one errors
                if self.trajectories[self.traj_index]['path_type'] == "gello_traj":
                    self.state = self.STATE.GELLO
                    self.gello_traj_to_save = []
                    # Start the thread that will stop the gello mode on keyboard press
                    self.gello_keyboard_listener = keyboard.Listener(on_press=self.get_gello_on_press())
                    self.gello_keyboard_listener.start()
                else:
                    self.state = self.STATE.EXECUTING_TRAJ
                self.env._robot.enable_rendering()
                print("Shifting from settling to executing")
            cur_joint = self.last_action
        elif self.last_action is None or self.state == self.STATE.EXECUTING_TRAJ:
            if self.traj_index >= len(self.trajectories):
                return None
            
            traj = np.array(self.trajectories[self.traj_index]['qs'])

            if self.t >= len(traj):
                self.save_curr_trajectory()
                self.ask_user_questions_and_record(self.trajectories[self.traj_index])
                self.load_next_recorded_trajectory()

                if self.traj_index >= len(self.trajectories):
                    return None  # No more trajectory steps available
                self.state = self.STATE.SETTLING
                # TODO check for off by one errors
                self.settle_steps_remaining = self.NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS
                self.env._robot.disable_rendering()
                print("Shifting from executing to settling")
                
            # TODO save it corrrectly so it doesn't need this :7
            cur_joint = traj[self.t, :7]
            cur_joint = cur_joint.tolist()
            cur_joint = np.array(cur_joint)
            self.t += self.step_size
        elif self.state == self.STATE.GELLO:
            cur_joint = self.gello_agent.act(obs=None)
            self.gello_traj_to_save.append(cur_joint)
            # cur_joint = np.array([-1] * 6 + [0])
            # Make sure it has CPU to look for keyboard presses
            time.sleep(1)
        elif self.state == self.STATE.QUITTING_GELLO:
            # Do this on the main thread. Can't do this in the keyboard thread
            # Go to next trajectory
            self.save_curr_trajectory()
            self.load_next_recorded_trajectory()
            self.state = self.STATE.SETTLING
            self.settle_steps_remaining = self.NUM_STEPS_TO_SETTLE_BETWEEN_TRAJS
            self.env._robot.disable_rendering()
            print("Shifting from executing to settling")
            cur_joint = self.last_action
        else:
            raise ValueError(f"Unknown state {self.state}")
        
        if cur_joint.shape[0] == 6:
            cur_joint = np.append(cur_joint, 0) # assume gripper is in the open position (1)
        cur_joint = cur_joint[:7]

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
