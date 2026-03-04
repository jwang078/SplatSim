from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import types
import yaml

from splatsim.robots.robot import BimanualRobot, PrintRobot
from splatsim.configs import DebugModes
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"
    gaussian_path : str = "/home/jennyw2/data/output/robot_iphone/point_cloud/iteration_30000/point_cloud.ply"
    robot_name: str = "robot_iphone"

    # Debug mode for PyBullet visualization
    debug_mode: DebugModes = DebugModes.OFF


def launch_robot_server(args: Args):
    # Match lerobot's precision settings so dataset collection
    # uses the same tf32 precision as evaluation
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    with open("configs/object_configs/objects.yaml", "r") as f:
        object_config = yaml.safe_load(f)

    has_wrist_camera = object_config[args.robot_name].get("wrist_camera_link_name", None) is not None
    if has_wrist_camera:
        camera_names = ["base_rgb", "wrist_rgb"]
    else:
        camera_names = ["base_rgb"]
    use_gripper = object_config[args.robot_name].get("use_gripper", True)
    
    port = args.robot_port
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        from splatsim.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )

    elif args.robot == "sim_ur_pybullet_push":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_push import PybulletRobotServer

        server = PybulletRobotServer(
           port=port, host=args.hostname,
        )

    elif args.robot == "sim_ur_pybullet_cup":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_cup import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_pick_planner import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_pick_place_planner import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_assembly import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_articulated import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_deformable import PybulletRobotServer

        server = PybulletRobotServer(
           port=port, host=args.hostname,
        )

    elif args.robot == "sim_ur_pybullet_orange":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import OrangeOnPlatePybulletRobotServer

        server = OrangeOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=OrangeOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_orange_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import OrangeOnPlatePybulletRobotServer

        server = OrangeOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=OrangeOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
            camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
            debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_apple":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_apple_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_apple_interactive-nosplat":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import AppleOnPlatePybulletRobotServer

        server = AppleOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=AppleOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_small_engine_new_interactive":
        from splatsim.robots.sim_robot_pybullet_small_engine import UprightRobotSmallEngineNewPybulletRobotServer

        server = UprightRobotSmallEngineNewPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=UprightRobotSmallEngineNewPybulletRobotServer.SERVE_MODES.INTERACTIVE,
        #    camera_names=[], robot_name=args.robot_name, use_gripper=use_gripper,

        #    camera_names=["base_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,

           camera_names=["base_rgb", "wrist_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           image_resize_modes=['letterbox'],
        #    camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
        #    image_width=224, image_height=224
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_small_engine_new_interactive_stretchimg":
        from splatsim.robots.sim_robot_pybullet_small_engine import UprightRobotSmallEngineNewPybulletRobotServer

        server = UprightRobotSmallEngineNewPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=UprightRobotSmallEngineNewPybulletRobotServer.SERVE_MODES.INTERACTIVE,
        #    camera_names=[], robot_name=args.robot_name, use_gripper=use_gripper,

        #    camera_names=["base_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,

           camera_names=["base_rgb", "wrist_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           image_resize_modes=['stretch'],
        #    camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
        #    image_width=224, image_height=224
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_small_engine_new_interactive_norender":
        from splatsim.robots.sim_robot_pybullet_small_engine import UprightRobotSmallEngineNewPybulletRobotServer

        server = UprightRobotSmallEngineNewPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=UprightRobotSmallEngineNewPybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=[], robot_name=args.robot_name, use_gripper=use_gripper,
           image_width=96, image_height=96,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_open_bwa_interactive":
        from splatsim.robots.sim_robot_pybullet_robot_in_bwa import OpenSpaceBWAPybulletRobotServer

        server = OpenSpaceBWAPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=OpenSpaceBWAPybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=5, use_gripper=use_gripper,

        #    image_width=224, image_height=224

           # with the actual engine on table
        #    camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_banana":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import BananaOnPlatePybulletRobotServer

        server = BananaOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=BananaOnPlatePybulletRobotServer.SERVE_MODES.GENERATE_DEMOS,
           camera_names=[], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_pybullet_banana_interactive":
        from splatsim.robots.sim_robot_pybullet_object_on_plate import BananaOnPlatePybulletRobotServer

        server = BananaOnPlatePybulletRobotServer(
           port=port, host=args.hostname, serve_mode=BananaOnPlatePybulletRobotServer.SERVE_MODES.INTERACTIVE,
           camera_names=camera_names, robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           debug_mode=args.debug_mode
        )

    elif args.robot == "sim_ur_splat":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        # gripper_xml = None
        # from splatsim.robots.sim_robot_pybullet import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_splat_6DOF import PybulletRobotServer
        # from splatsim.robots.sim_robot_pybullet_splat_servoing import PybulletRobotServer
        from splatsim.robots.sim_robot_pybullet_splat_servoing_improved import GaussianRenderServer
        server = GaussianRenderServer(
           port=port, host=args.hostname, gaussian_path=args.gaussian_path
        )

    elif args.robot == "sim_panda":
        from splatsim.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )

    elif args.robot == "sim_xarm":
        from splatsim.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )

    else:
        if args.robot == "xarm":
            from splatsim.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur":
            from splatsim.robots.ur import URRobot

            robot = URRobot(robot_ip=args.robot_ip, no_gripper=False)
        elif args.robot == "panda":
            from splatsim.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=args.robot_ip, no_gripper=False)
        elif args.robot == "bimanual_ur":
            from splatsim.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.11")
            _robot_r = URRobot(robot_ip="192.168.1.11")
            robot = BimanualRobot(_robot_l, _robot_r)
        
        elif args.robot == "ur_pybullet":
            from splatsim.robots.ur_pybullet import URRobotPybullet
            robot = URRobotPybullet(robot_ip=args.robot_ip, no_gripper=True)

        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
    
    
    try:
        server.serve()
    except KeyboardInterrupt:
        if hasattr(server, "shutdown") and type(getattr(server, "shutdown")) == types.MethodType:
            server.shutdown()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
