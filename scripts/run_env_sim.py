import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro
import termcolor


from splatsim.utils.lerobot_utils import (
    build_lerobot_frame,
    create_lerobot_dataset,
    finalize_lerobot_dataset,
    load_lerobot_dataset,
    push_lerobot_to_hub,
)


from splatsim.agents.agent import BimanualAgent, DummyAgent
from splatsim.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from splatsim.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
from splatsim.utils.agent_state_utils import AGENT_STATE
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig



import yaml



def print_color(*args, color=None, attrs=(), **kwargs):
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    policy_guidance_mode: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/data/bc_data"
    bimanual: bool = False
    verbose: bool = False

    save_lerobot_dataset: bool = False
    lerobot_dataset_repo_id_base: str = "JennyWWW/splatsim_lerobot_dataset"
    save_lerobot_dataset_on_interrupt: bool = False


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=False)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from splatsim.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from splatsim.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print('all usb ports:', usb_ports)
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
                reset_joints = np.array(reset_joints)
            agent = GelloAgent(port=gello_port, start_joints=reset_joints)
            # curr_joints = env.get_obs()["joint_positions"]
            
            # #if not np array, then convert to np array
            # if not isinstance(curr_joints, np.ndarray):
            #     curr_joints = np.array(curr_joints)
            
            # if reset_joints.shape == curr_joints.shape:
            #     max_delta = (np.abs(curr_joints - reset_joints)).max()
            #     steps = min(int(max_delta / 0.01), 100)

            #     for jnt in np.linspace(curr_joints, reset_joints, steps):
            #         env.step(jnt)
            #         time.sleep(0.001)
            startup_steps = 2
            query_new_joints_per_startup_step = True
        elif args.agent == "quest":
            from splatsim.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            startup_steps = 100
            query_new_joints_per_startup_step = True
        elif args.agent == "spacemouse":
            from splatsim.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
            startup_steps = 100
            query_new_joints_per_startup_step = True
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
            startup_steps = 2
            query_new_joints_per_startup_step = True
        elif args.agent == "policy":
            from splatsim.agents.policy_agent import DiffusionAgent
            agent = DiffusionAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0")
            startup_steps = 25
            query_new_joints_per_startup_step = True
        elif args.agent == "policy6DOF":
            from splatsim.agents.policy_agent_6DOF import DiffusionAgent
            agent = DiffusionAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0")
            startup_steps = 25
            query_new_joints_per_startup_step = True
        elif args.agent == "policy_diffusion":
            from splatsim.agents.policy_agent_new import DiffusionPolicyAgent
            startup_steps = 2
            agent = DiffusionPolicyAgent(env=env, save_images=False)
            query_new_joints_per_startup_step = True
        elif args.agent == "policy_diffusion_and_save":
            from splatsim.agents.policy_agent_new import DiffusionPolicyAgent
            startup_steps = 16
            agent = DiffusionPolicyAgent(env=env, save_images=True, num_startup_steps=startup_steps)
            query_new_joints_per_startup_step = True
        elif args.agent == "policy_diffusion":
            from splatsim.agents.policy_agent_new import DiffusionPolicyAgent
            agent = DiffusionPolicyAgent(env=env, save_images=True)
            startup_steps = 25
            query_new_joints_per_startup_step = True
        elif args.agent == "servoing":
            from splatsim.agents.servoing_agent import ServoingAgent
            agent = ServoingAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0")
            startup_steps = 25
            query_new_joints_per_startup_step = True
        elif args.agent == "interface":
            from splatsim.agents.interface_agent import InterfaceAgent
            agent = InterfaceAgent()
            startup_steps = 2
            query_new_joints_per_startup_step = True
        elif args.agent == "text_interface":
            from splatsim.agents.text_interface_agent import TextInterfaceAgent
            agent = TextInterfaceAgent()
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "slider_interface":
            from splatsim.agents.slider_interface_agent import SliderInterfaceAgent
            agent = SliderInterfaceAgent()
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "joystick_interface":
            from splatsim.agents.joystick_interface_agent import JoystickInterfaceAgent
            kwargs = {}
            if args.robot_type is not None:
                kwargs["robot_name"] = args.robot_type
            agent = JoystickInterfaceAgent(**kwargs)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "keyboard_interface":
            from splatsim.agents.keyboard_interface_agent import KeyboardInterfaceAgent
            kwargs = {}
            if args.robot_type is not None:
                kwargs["robot_name"] = args.robot_type
            if args.policy_guidance_mode:
                kwargs["delta_mode"] = True
            agent = KeyboardInterfaceAgent(**kwargs)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "replay_trajectory":
            from splatsim.agents.replay_trajectory_agent import ReplayTrajectoryAgent
            with open("configs/folder_configs.yaml", "r") as file:
                folder_config = yaml.safe_load(file)
            traj_folder = folder_config["traj_folder"]
            agent = ReplayTrajectoryAgent(traj_folder=traj_folder, env=env, save_images=False)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "replay_trajectory_and_save":
            from splatsim.agents.replay_trajectory_agent import ReplayTrajectoryAgent
            with open("configs/folder_configs.yaml", "r") as file:
                folder_config = yaml.safe_load(file)
            traj_folder = folder_config["traj_folder"]
            agent = ReplayTrajectoryAgent(traj_folder=traj_folder, env=env, save_images=True)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "replay_zarr_trajectory":
            from splatsim.agents.replay_zarr_trajectory_agent import ReplayZarrTrajectoryAgent
            with open("configs/folder_configs.yaml", "r") as file:
                folder_config = yaml.safe_load(file)
            # traj_folder = "./output/obstacles_on_path_onegoal.zarr" # temporary hardcoding
            # traj_folder = "./output/obstacles_on_path_onegoal.zarr" # temporary hardcoding
            # traj_folder = "/home/jennyw2/code/SplatSim/output/obstacles_on_path_onegoal_20dataset_simple.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/small_engine_keyframes_default.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/eval_go_to_nominal_10.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/go_to_nominal_10.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/upright_robot_small_engine_new_test_trajectories.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/upright_robot_small_engine_new_trajectories.zarr"
            # traj_folder = "output/upright_robot_small_engine_new_trajectories_sapi05.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/upright_robot_small_engine_new_1strrtpath_trajectories.zarr"
            # traj_folder = "/home/jennyw2/code/SplatSim/output/upright_robot_small_engine_new_5thrrtpath_trajectories.zarr"
            traj_folder = "/home/jennyw2/code/SplatSim/output/upright_robot_small_engine_new_1path_trajectories.zarr"
            agent = ReplayZarrTrajectoryAgent(traj_folder=traj_folder, env=env, save_images=False)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "replay_zarr_trajectory_and_save":
            from splatsim.agents.replay_zarr_trajectory_agent import ReplayZarrTrajectoryAgent
            with open("configs/folder_configs.yaml", "r") as file:
                folder_config = yaml.safe_load(file)
            # traj_folder = "./output/obstacles_on_path_onegoal.zarr" # temporary hardcoding
            # traj_folder = "./output/obstacles_on_path_onegoal.zarr" # temporary hardcoding
            traj_folder = "/home/jennyw2/code/SplatSim/output/obstacles_on_path_onegoal_20dataset_simple.zarr"
            agent = ReplayZarrTrajectoryAgent(traj_folder=traj_folder, env=env, save_images=True)
            startup_steps = 2
            query_new_joints_per_startup_step = False
        elif args.agent == "replay_zarr_trajectory_user_study":
            from splatsim.agents.replay_zarr_trajectory_userstudy_agent import ReplayZarrTrajectoryUserStudyAgent
            with open("configs/folder_configs.yaml", "r") as file:
                folder_config = yaml.safe_load(file)
            traj_folder = "output/obstacles_on_path_10_userstudy.zarr"

            agent = ReplayZarrTrajectoryUserStudyAgent(traj_folder=traj_folder, env=env, save_images=False, gello_port=args.gello_port, gello_start_joints=args.start_joints)
            startup_steps = 2
            query_new_joints_per_startup_step = False

        elif args.agent == "lerobot":
            # Point to your local checkpoint or Hugging Face hub ID
            # checkpoint = "outputs/pi05_finetune/checkpoints/last/pretrained_model"

            # Does ok with going to nominal position
            # checkpoint = "outputs/pi05_training/checkpoints/last/pretrained_model"

            # Approach lever of engine
            checkpoint = "/home/jennyw2/code/SplatSim/outputs/pi05_training_approach_lever/checkpoints/last/pretrained_model"
            from splatsim.agents.lerobot_agent import LeRobotAgent
            agent = LeRobotAgent(checkpoint_path=checkpoint)
            startup_steps = 16
            query_new_joints_per_startup_step = False
        else:
            raise ValueError("Invalid agent name")
        
    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

    # No more importing after this point

    # going to start position
    print("Going to start position")
    obs = env.get_obs()
    if obs is not None:
        if query_new_joints_per_startup_step:
            start_pos = agent.act(obs)
        else:
            start_pos = obs["joint_positions"]
        joints = obs["joint_positions"]
        
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)
        
        #if start pose is of dimension 7 and joints is of dimension 6, then add last element to joints
        if start_pos.shape[0] == 7 and joints.shape[0] == 6:
            joints = np.append(joints, start_pos[-1])

        
        print('start_pos:', start_pos)
        print('joints:', joints)

        abs_deltas = np.abs(start_pos - joints)
        id_max_joint_delta = np.argmax(abs_deltas)

        # max_joint_delta = 0.8
        # if abs_deltas[id_max_joint_delta] > max_joint_delta:
        #     id_mask = abs_deltas > max_joint_delta
        #     print()
        #     ids = np.arange(len(id_mask))[id_mask]
        #     for i, delta, joint, current_j in zip(
        #         ids,
        #         abs_deltas[id_mask],
        #         start_pos[id_mask],
        #         joints[id_mask],
        #     ):
        #         print(
        #             f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
        #         )
        #     return

        print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
        assert len(start_pos) == len(
            joints
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

        max_delta = 0.05
        command_joints = start_pos
        
        for _ in range(startup_steps):
            obs = env.get_obs()
            if query_new_joints_per_startup_step:
                command_joints = agent.act(obs)
            current_joints = obs["joint_positions"]
            if not isinstance(current_joints, np.ndarray):
                current_joints = np.array(current_joints)
            
            # obs joint_positions is truncated to num_dofs (6) in
            # _raw_obs_to_gym_obs, but the agent outputs num_dofs + 1 (with the
            # gripper). Pad current_joints with the agent's gripper value so the
            # delta subtraction and the resulting action are (num_dofs + 1)-dim.
            if command_joints.shape[0] == current_joints.shape[0] + 1:
                current_joints = np.append(current_joints, command_joints[-1])

            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            print("command:", command_joints)
            print("current:", current_joints)
            print("delta:", delta)
            print("")
            
                
            # env.step expects the full action (num_dofs arm joints + 1 gripper);
            # do NOT strip the gripper element. current_joints was padded to
            # match command_joints above, so current_joints + delta is already
            # the correct (num_dofs + 1)-dim action.
            env.step(current_joints + delta) # ------------------------------


    obs = env.get_obs()
    if obs is not None:
        joints = obs["joint_positions"]
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)
        
        action = agent.act(obs)
        
        if action.shape[0] == 7 and joints.shape[0] == 6:
            joints = np.append(joints, action[-1])
        
    
    # if (action - joints > 0.5).any():
    #     print("Action is too big")

    #     # print which joints are too big
    #     joint_index = np.where(action - joints > 0.8)
    #     for j in joint_index:
    #         print(
    #             f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
    #         )

    if args.use_save_interface:
        kb_interface = KBReset()

    lerobot_saver = None
    if args.save_lerobot_dataset:
        # Detect image keys from obs: any 3-D numpy array (C, H, W).
        # Sim obs already has pre-resized keys like "base_rgb_letterbox";
        # real-robot wrappers may expose "base_rgb" (raw) — both work here.
        image_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3]
        print(f"Saving {len(image_keys)} image keys in observation to lerobot dataset:", image_keys)

        dataset_repo_id = f"{args.lerobot_dataset_repo_id_base}"
        print(f"Saving to LeRobot dataset with repo ID: {dataset_repo_id}")

        lerobot_saver = load_lerobot_dataset(dataset_repo_id)
        if lerobot_saver is None:
            print(f"Creating new LeRobot dataset: {dataset_repo_id}")
            lerobot_saver = create_lerobot_dataset(dataset_repo_id, fps=30, image_keys=image_keys, num_dofs=6)

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    interrupted = False
    start_time = time.time()
    started_executing_traj = False
    try:
        while True:
            loop_start = time.time()

            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = agent.act(obs)

            if getattr(agent, 'state', AGENT_STATE.UNKNOWN) == AGENT_STATE.EXECUTING_TRAJ:
                started_executing_traj = True

            if getattr(agent, 'state', AGENT_STATE.UNKNOWN) == AGENT_STATE.SETTLING and started_executing_traj:
                if lerobot_saver is not None:
                    lerobot_saver.save_episode()
                    print("In run env sim saving episode")
                started_executing_traj = False

            dt = datetime.datetime.now()
            if args.use_save_interface:
                state = kb_interface.update()
                if state == "start":
                    dt_time = datetime.datetime.now()
                    save_path = (
                        Path(args.data_dir).expanduser()
                        / args.agent
                        / dt_time.strftime("%m%d_%H%M%S")
                    )
                    save_path.mkdir(parents=True, exist_ok=True)
                    print(f"Saving to {save_path}")
                elif state == "save":
                    assert save_path is not None, "something went wrong"
                    #check whether each element of dict is not None
                    if obs is not None and all(value is not None for value in obs.values()):
                        save_frame(save_path, dt, obs, action)
                        print('success')
        
                elif state == "normal":
                    save_path = None
                else:
                    raise ValueError(f"Invalid state {state}")
                
            if args.policy_guidance_mode:
                # action is (7,) delta or all-NaN when no keys are held.
                # Send as [1, 1, 7] so the wrapper receives it with a batch and time dimension.
                if action is not None and not np.all(np.isnan(action)):
                    guidance = action.reshape(1, 1, -1)  # [1, 1, 7]
                else:
                    guidance = None
                env.queue_policy_guidance_action(guidance)
                # Refresh obs without rendering so the keyboard agent has
                # up-to-date joint positions for IK seeding, without contending
                # with the lerobot eval loop's image renders on the sim server.
                obs = env.get_obs(render_images=False)
            else:
                # env.step expects the full (num_dofs + 1)-dim action including
                # the gripper at index num_dofs. Previously this stripped the
                # last element when `flag` was set (agent=7 vs obs=6), which
                # silently dropped the gripper command after num_dofs changed
                # 7 -> 6 (obs joint_positions is truncated to num_dofs in
                # _raw_obs_to_gym_obs). Send the action unmodified.
                obs = env.step(action)
                # Save frame after step: state = post-physics obs (matches traj-gen convention)
                if getattr(agent, 'state', AGENT_STATE.UNKNOWN) == AGENT_STATE.EXECUTING_TRAJ and lerobot_saver is not None and action is not None:
                    try:
                        task = robot_client.get_task_description()
                    except Exception:
                        task = ""
                    lerobot_saver.add_frame(build_lerobot_frame(action, obs, image_keys, task=task))

            loop_end = time.time()
            loop_duration = loop_end - loop_start
            # Keep the time locked at a fixed rate
            sleep_time = max(0, (1 / 30) - (loop_duration))
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        interrupted = True
        print_color("\nStopping run_env_sim.py", color="red", attrs=("bold",))
    finally:
        if lerobot_saver is not None:
            if started_executing_traj and args.save_lerobot_dataset_on_interrupt:
                print("Saving in-progress episode...")
                lerobot_saver.save_episode()
            finalize_lerobot_dataset(lerobot_saver)
            if args.save_lerobot_dataset_on_interrupt:
                push_lerobot_to_hub(lerobot_saver)
        else:
            print("lerobot_saver is not initialized, so not pushing to hub")


if __name__ == "__main__":
    main(tyro.cli(Args))
