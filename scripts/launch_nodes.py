from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
    # Robot splat/URDF name to load. When None (default), the server variant
    # picked by `--robot` provides its own class-level
    # `DEFAULT_ROBOT_NAME` (see `PybulletRobotServerBase.DEFAULT_ROBOT_NAME`
    # + subclass overrides). Callers should only pass this when overriding
    # per-run (e.g. running the small_engine env against a different
    # splat trained on a modified scene). Bash orchestrator scripts
    # deliberately omit `--robot_name` so the class default flows through
    # — matches the LeRobot side's default resolution and eliminates the
    # need to keep two hardcoded strings in sync across repos.
    robot_name: Optional[str] = None

    # Debug mode for PyBullet visualization
    debug_mode: DebugModes = DebugModes.OFF

    # When set, the server starts in EVAL_BENCHMARK serve mode and cycles
    # through the pre-recorded scenarios from this LeRobot dataset on each
    # env.reset(). Leave None to use the robot variant's default serve mode.
    eval_benchmark_repo_id: Optional[str] = None
    # Optional subset of episode indices (e.g. [3, 8, 23]). None = all episodes.
    eval_benchmark_subset: Optional[List[int]] = None

    # Wrist camera model version (see WRIST_CAM_FISHEYE_CALIBRATIONS in
    # splatsim/robots/sim_robot_pybullet_base.py):
    #   0 = pinhole using base camera intrinsics (matches pre-fisheye datasets)
    #   1 = fisheye, original 2704x2028 GoPro calibration
    #   2 = fisheye, recalibrated 1920x1080 GoPro calibration (default)
    # Matches `PybulletRobotServerBase.__init__`'s Python default (2) and
    # LeRobot's `SplatSimEnv.wrist_cam_ver` default (2). Keeping this in
    # lockstep prevents client-server calibration mismatch: the client's
    # LeRobot config passes wrist_cam_ver=2 to the server over ZMQ for
    # its ObsResampler, and if the launch script had a different default
    # the sim server would render with ver=1 intrinsics while lerobot
    # thought it was ver=2.
    wrist_cam_ver: int = 2

    # When True, connect pybullet in DIRECT (no GUI) mode. Skips OpenGL
    # context creation entirely → no display required, ~3-5x faster for
    # physics-only workloads. Gaussian splat rendering is unavailable in
    # this mode. Intended for fast batch operations like trajectory
    # replay + collision-check filtering.
    headless: bool = False

    # Clearances used by the robot server's ``check_metrics()`` when
    # reporting ``info["in_collision"]``. Defaults preserve historical
    # behavior (0 m on both = actual contact / penetration only). Set
    # non-zero (typically matching the DAgger SA wrapper's
    # ``rrt_obstacle_clearance`` / ``rrt_self_collision_clearance``) so
    # downstream collision filters / termination triggers consider
    # near-misses as collisions too. Forwarded to the robot server's
    # constructor; ignored by robot variants that don't accept the
    # corresponding kwargs.
    in_collision_obstacle_clearance: float = 0.0
    in_collision_self_collision_clearance: float = 0.0


def _resolve_default_robot_name(robot_variant: str) -> str:
    """Map a `--robot` variant to its class-level `DEFAULT_ROBOT_NAME`.

    Lazily imports the specific server subclass so this helper doesn't
    pull in every variant's deps for a single launch. Falls back to the
    base class's `DEFAULT_ROBOT_NAME` (historical `"robot_iphone"`) when
    the variant isn't recognized here — matches how `--robot_name`
    defaulted before this refactor.

    Add a new branch when you add a new robot variant with an env-
    specific splat: `elif robot_variant == "sim_ur_pybullet_new_env": ...`
    """
    if robot_variant in (
        "sim_ur_pybullet_small_engine_new_interactive",
        "sim_ur_pybullet_small_engine_new_interactive_strict",
    ):
        from splatsim.robots.sim_robot_pybullet_small_engine import (
            UprightRobotSmallEngineNewPybulletRobotServer,
        )
        return UprightRobotSmallEngineNewPybulletRobotServer.DEFAULT_ROBOT_NAME
    # Fallback: base class default.
    from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase
    return PybulletRobotServerBase.DEFAULT_ROBOT_NAME


def launch_robot_server(args: Args):
    # Match lerobot's precision settings so dataset collection
    # uses the same tf32 precision as evaluation
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve --robot_name from the robot variant's class default when the
    # user didn't pass an explicit override. Downstream code paths (YAML
    # lookup, per-branch `robot_name=args.robot_name` on the server
    # constructor) all consume the same resolved value, so bash callers
    # who omit `--robot_name` get the env-specific canonical splat name
    # (e.g. `robot_iphone_w_engine_curtain` for small_engine) without any
    # hardcoded string on the launch side.
    if args.robot_name is None:
        args.robot_name = _resolve_default_robot_name(args.robot)

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

        # Auto-switch to EVAL_BENCHMARK mode when an eval-benchmark dataset is
        # provided so the server cycles through its scenarios on each reset.
        serve_mode = (
            UprightRobotSmallEngineNewPybulletRobotServer.SERVE_MODES.EVAL_BENCHMARK
            if args.eval_benchmark_repo_id is not None
            else UprightRobotSmallEngineNewPybulletRobotServer.SERVE_MODES.INTERACTIVE
        )
        server = UprightRobotSmallEngineNewPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=serve_mode,
           camera_names=["base_rgb", "wrist_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           image_resize_modes=['letterbox', 'stretch'],
           debug_mode=args.debug_mode,
           eval_benchmark_repo_id=args.eval_benchmark_repo_id,
           eval_benchmark_subset=args.eval_benchmark_subset,
           wrist_cam_ver=args.wrist_cam_ver,
           headless=args.headless,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
        )

    elif args.robot == "sim_ur_pybullet_small_engine_new_interactive_strict":
        # Tighter success-tolerance variant for DAgger-style intervention
        # recording: the loose eval-time threshold (3 cm / 10 deg) cuts off
        # precise RRT corrections before they reach the exact goal pose.
        from splatsim.robots.sim_robot_pybullet_small_engine import UprightRobotSmallEngineNewStrictPybulletRobotServer

        serve_mode = (
            UprightRobotSmallEngineNewStrictPybulletRobotServer.SERVE_MODES.EVAL_BENCHMARK
            if args.eval_benchmark_repo_id is not None
            else UprightRobotSmallEngineNewStrictPybulletRobotServer.SERVE_MODES.INTERACTIVE
        )
        server = UprightRobotSmallEngineNewStrictPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=serve_mode,
           camera_names=["base_rgb", "wrist_rgb"], robot_name=args.robot_name, cam_i=3, use_gripper=use_gripper,
           image_resize_modes=['letterbox', 'stretch'],
           debug_mode=args.debug_mode,
           eval_benchmark_repo_id=args.eval_benchmark_repo_id,
           eval_benchmark_subset=args.eval_benchmark_subset,
           wrist_cam_ver=args.wrist_cam_ver,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
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
