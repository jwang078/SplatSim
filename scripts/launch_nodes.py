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

    # Keep the Tkinter "SplatSim Controls" panel even when --headless. pybullet
    # still connects DIRECT (no 3D OpenGL window, EGL GPU rendering, no ~30 Hz
    # render-loop throttle), but the control panel launches so you can pick modes,
    # tune the trajectory config, and press Start interactively — with fast
    # headless rendering. Needs a display for Tkinter (use on a workstation, not a
    # display-less node). No effect without --headless (a GUI run already shows
    # the panel). Only wired for the pybullet sim envs.
    control_gui: bool = False

    # Path to a trajectory-generator config JSON exported from the SplatSim GUI
    # ("Export Config" in the Traj Gen panel). When set, its values are applied
    # to the server's trajectory generator at startup — so you can tune settings
    # interactively in the GUI, export, then re-run generation elsewhere (e.g.
    # headless) with the exact same config. Loaded field-by-field and tolerant of
    # schema drift (see splatsim/utils/config_io.py); only affects envs that have
    # a trajectory generator.
    traj_config_file: Optional[str] = None

    # When True, skip the per-step gsplat camera render: get_observations
    # returns None for every image key but still runs physics, joint/EE
    # state, metrics, and RRT. Much faster when you don't need images (e.g.
    # scripted/collision-only runs). Unlike --headless this keeps the GUI
    # and the base camera loaded, so you can flip rendering back on at
    # runtime via the server's enable_rendering().
    no_camera_rendering: bool = False

    # Client-driven physics: when True, physics only steps in response to a
    # client `command_joint_state` call — the main serve loop's autonomous
    # 240 Hz step is disabled. Eliminates the "sim races ahead while the
    # policy is thinking" pathology visible with slow policies (e.g. a
    # diffusion U-Net taking 200-500 ms per chunk-boundary inference; the
    # sim was previously advancing physics under the last commanded target
    # during that entire window, so the robot visibly jumped forward when
    # the client resumed and could catch up). Each command runs
    # `physics_substeps_per_command` (default 8) `stepSimulation` calls,
    # matching the async default rate of 240 Hz when the client is running
    # at 30 Hz. Only affects INTERACTIVE / EVAL_BENCHMARK* modes.
    sync_physics_to_client: bool = False
    physics_substeps_per_command: int = 8

    # Image-observation source: "splat" (Gaussian-splat render), "pybullet"
    # (fast PyBullet getCameraImage — works without splat assets), or "none"
    # (state/action-only, fastest). Overrides --no_camera_rendering when set.
    # If unset, the env's own default is used (splat envs -> splat; the planar
    # env -> pybullet). Also switchable at runtime via the GUI "Render mode"
    # dropdown.
    render_mode: Optional[str] = None

    # Clearances used by the robot server's ``check_metrics()`` when
    # reporting ``info["in_collision"]``. Default 5 mm (bumped from historical
    # 0.0 = penetration-only) because PyBullet's constraint solver holds rigid
    # bodies at ~0 mm gap under contact force — a link pressed against an
    # obstacle stays a rounding hair above zero penetration, so the old
    # penetration-only default silently missed "arm slid into obstacle" and
    # "arm folded onto own body" cases. 5 mm matches the SA wrapper's
    # ``rrt_self_collision_clearance=0.005`` default so env-terminate agrees
    # with what the planner treats as a collision. Forwarded to the robot
    # server's constructor; ignored by robot variants that don't accept the
    # corresponding kwargs. Pass ``0`` explicitly to restore penetration-only.
    in_collision_obstacle_clearance: float = 0.005
    in_collision_self_collision_clearance: float = 0.005


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
    if robot_variant in ("sim_pybullet_planar_interactive",
                         "sim_pybullet_planar_oracle_interactive",
                         "sim_pybullet_planar_oracle_simple_interactive"):
        from splatsim.robots.sim_robot_pybullet_planar import (
            Planar3JointPybulletRobotServer,
        )
        return Planar3JointPybulletRobotServer.DEFAULT_ROBOT_NAME  # all share robot_name=planar_3joint
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

    # Resolve the image-observation source. --render_mode wins; else the legacy
    # --no_camera_rendering maps to NONE; else None = let the env pick its
    # default (splat envs -> splat; planar -> pybullet via RENDER_PYBULLET_CAMERA).
    from splatsim.configs.mode_config import RenderMode
    if args.render_mode is not None:
        resolved_render_mode = RenderMode(args.render_mode)
    elif args.no_camera_rendering:
        resolved_render_mode = RenderMode.NONE
    else:
        resolved_render_mode = None

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
           show_control_gui=args.control_gui,
           render_mode=resolved_render_mode,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
           sync_physics_to_client=args.sync_physics_to_client,
           physics_substeps_per_command=args.physics_substeps_per_command,
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
           headless=args.headless,
           show_control_gui=args.control_gui,
           render_mode=resolved_render_mode,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
           sync_physics_to_client=args.sync_physics_to_client,
           physics_substeps_per_command=args.physics_substeps_per_command,
        )

    elif args.robot == "sim_pybullet_planar_interactive":
        # Fast planar 3-joint arm. RENDER_SPLATS=False -> no splat assets/base
        # camera, so SPLAT is unavailable; it defaults to the PyBullet camera
        # (RENDER_PYBULLET_CAMERA=True). Pass --render_mode {pybullet,none} to
        # pick, or the GUI "Render mode" dropdown at runtime.
        from splatsim.robots.sim_robot_pybullet_planar import Planar3JointPybulletRobotServer

        # Auto-switch to EVAL_BENCHMARK when an eval-benchmark dataset is given so
        # the server cycles through its recorded scenarios (via the reset's
        # benchmark_start_index) instead of doing a RANDOM reset — and so the GUI
        # opens on the Eval Benchmark tab. Mirrors the small_engine branch.
        serve_mode = (
            Planar3JointPybulletRobotServer.SERVE_MODES.EVAL_BENCHMARK
            if args.eval_benchmark_repo_id is not None
            else Planar3JointPybulletRobotServer.SERVE_MODES.INTERACTIVE
        )
        server = Planar3JointPybulletRobotServer(
           port=port, host=args.hostname, serve_mode=serve_mode,
           camera_names=["base_rgb"], robot_name=args.robot_name, use_gripper=use_gripper,
           render_mode=resolved_render_mode,
           debug_mode=args.debug_mode,
           eval_benchmark_repo_id=args.eval_benchmark_repo_id,
           eval_benchmark_subset=args.eval_benchmark_subset,
           headless=args.headless,
           show_control_gui=args.control_gui,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
           sync_physics_to_client=args.sync_physics_to_client,
           physics_substeps_per_command=args.physics_substeps_per_command,
        )

    elif args.robot in ("sim_pybullet_planar_oracle_interactive",
                        "sim_pybullet_planar_oracle_simple_interactive"):
        # Oracle-state planar env: observation.state carries exact object coords
        # (goal + obstacles) for a STATE-ONLY policy. `_simple_` = 0 obstacles.
        # We KEEP the base_rgb camera so the SplatSim GUI's pybullet render panel
        # works (blank with camera_names=[]) and you get a free vision dataset;
        # the policy stays image-free because training uses --cameras=state
        # (input_features = observation.state only).
        from splatsim.robots.sim_robot_pybullet_planar import (
            Planar3JointOraclePybulletRobotServer,
            Planar3JointOracleSimplePybulletRobotServer,
        )
        _oracle_cls = (
            Planar3JointOracleSimplePybulletRobotServer
            if args.robot == "sim_pybullet_planar_oracle_simple_interactive"
            else Planar3JointOraclePybulletRobotServer
        )
        # See the sim_pybullet_planar_interactive branch: EVAL_BENCHMARK when a
        # benchmark dataset is given (cycles recorded scenarios + opens the Eval
        # Benchmark tab), else INTERACTIVE.
        serve_mode = (
            _oracle_cls.SERVE_MODES.EVAL_BENCHMARK
            if args.eval_benchmark_repo_id is not None
            else _oracle_cls.SERVE_MODES.INTERACTIVE
        )
        server = _oracle_cls(
           port=port, host=args.hostname, serve_mode=serve_mode,
           camera_names=["base_rgb"], robot_name=args.robot_name, use_gripper=use_gripper,
           render_mode=resolved_render_mode,
           debug_mode=args.debug_mode,
           eval_benchmark_repo_id=args.eval_benchmark_repo_id,
           eval_benchmark_subset=args.eval_benchmark_subset,
           headless=args.headless,
           show_control_gui=args.control_gui,
           in_collision_obstacle_clearance=args.in_collision_obstacle_clearance,
           in_collision_self_collision_clearance=args.in_collision_self_collision_clearance,
           sync_physics_to_client=args.sync_physics_to_client,
           physics_substeps_per_command=args.physics_substeps_per_command,
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

    # Apply a GUI-exported trajectory-generator config, if given. Field-driven +
    # schema-drift tolerant (config_io), and updates the config IN PLACE so the
    # generator and GUI keep the same object.
    if getattr(args, "traj_config_file", None):
        gen = getattr(server, "trajectory_generator", None)
        if gen is not None and getattr(gen, "config", None) is not None:
            from splatsim.utils.config_io import update_dataclass_json
            update_dataclass_json(gen.config, args.traj_config_file, warn=print)
            print(f"[launch] Applied traj config from {args.traj_config_file}")
        else:
            print(f"[launch] --traj_config_file set but '{args.robot}' has no trajectory "
                  f"generator; ignoring.")

    try:
        server.serve()
    except KeyboardInterrupt:
        if hasattr(server, "shutdown") and type(getattr(server, "shutdown")) == types.MethodType:
            server.shutdown()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
