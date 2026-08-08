import logging
import pickle
import threading
import time
from typing import Any, ClassVar, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)
import gymnasium
from gymnasium import spaces
import enum
import random
from collections import namedtuple
import math
import os
import pickle
import shutil
import yaml
import shutil
from argparse import ArgumentParser

from dataclasses import dataclass, field, asdict
import numpy as np
import quaternion
import threading
import json

import torch
import numpy as np
import mujoco
import mujoco.viewer
import zmq
from splatsim.robots.robot import Robot
from splatsim.rendering.gsplat_renderer import render_gsplat

import cv2
from torchvision.transforms.functional import to_pil_image

assert mujoco.viewer is mujoco.viewer
from gaussian_splatting.scene.cameras import Camera
from gaussian_renderer import render

# import urdf_models.models_data as md
import pybullet as p
from pybullet_planning.interfaces.robots.collision import pairwise_collision, pairwise_link_collision

from pybullet_planning import RED, BLUE, GREEN
from pybullet_planning import Pose
from pybullet_planning import set_pose
from pybullet_planning import create_box
import pybullet_data
from splatsim.utils.robot_splat_render_utils import (
    get_segmented_indices,
    transform_means,
    get_transformation_list,
    transform_object,
    get_curr_link_states,
    crop_splat,
    create_cuboid_gaussians,
)
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams, Namespace
from gaussian_splatting.scene import Scene

from splatsim.utils.transform_utils import rotation_matrix_to_euler_angles
from splatsim.utils.image_utils import letterbox
from splatsim.utils.trajectory_generation import TrajectoryGenerator
from splatsim.utils.splatsim_gui import SplatSimGui
from splatsim.configs.env_config import (
    DebugModes,
    EnvConfig,
    CuboidObjectConfig,
    GraspConfig,
    SplatObjectConfig,
    ObjectConfig,
    SplatSimObject,
    ArticulationConfig,
)
from splatsim.configs.mode_config import TrajectoryGenModeConfig, ImageResizeMode, RenderMode
from splatsim.utils import rrt_path_utils
from splatsim.utils.rrt_path_utils import _COLLISION_CLEARANCE, RuckigCloudUnavailableError
from collections import defaultdict

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.transforms import ImageTransformConfig, ImageTransformsConfig

from pathlib import Path

from splatsim.utils.paths import SPLATSIM_ROOT, resolve_splatsim_path  # noqa: F401
from splatsim.utils.lerobot_utils import (
    build_lerobot_features,
    build_lerobot_frame,
    create_lerobot_dataset,
    finalize_lerobot_dataset,
    load_lerobot_dataset,
    push_lerobot_to_hub,
)


def resize_image(img: np.ndarray, output_size: Tuple[int, int], mode: 'ImageResizeMode') -> np.ndarray:
    """Resize image to output_size using the specified mode.

    Args:
        img: Input image in CHW format (channels, height, width), float32 in [0, 1]
        output_size: Target (height, width)
        mode: ImageResizeMode enum value

    Returns:
        Resized image in CHW format, float32 in [0, 1]
    """
    if mode == ImageResizeMode.LETTERBOX:
        return letterbox(img, output_size=output_size)
    elif mode == ImageResizeMode.STRETCH:
        # Convert from CHW to HWC for cv2
        img_hwc = np.transpose(img, (1, 2, 0))
        # Resize using cv2 (stretches to fill, ignoring aspect ratio)
        img_resized = cv2.resize(img_hwc, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
        # Convert back to CHW
        return np.transpose(img_resized, (2, 0, 1))
    else:
        raise ValueError(f"Unknown image resize mode: {mode}.")


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()
        self._policy_guidance_action = None
        self._policy_guidance_lock = threading.Lock()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                # print(f"Received request: {method}, {args}")
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "get_ee_pos":
                    result = self._robot.get_current_ee_pose()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(self._robot.splatsim_robot, **args)
                elif method == "set_policy_guidance_action":
                    with self._policy_guidance_lock:
                        self._policy_guidance_action = args["joint_state"]
                    result = None
                elif method == "teleport_joint_state":
                    result = self._robot.teleport_joint_state(
                        self._robot.splatsim_robot, **args
                    )
                elif method == "set_object_pose":
                    result = self._robot.set_object_pose(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations(render_images=args.get("render_images", True))
                    with self._policy_guidance_lock:
                        if self._policy_guidance_action is not None:
                            result["policy_guidance_chunk"] = self._policy_guidance_action.copy()
                elif method == "create_object":
                    splatsim_object = self._robot.create_object(**args)
                    result = None
                elif method == "delete_object":
                    result = self._robot.delete_object(**args)
                elif method == "clear_temp_objects":
                    result = self._robot.clear_temp_objects()
                elif method == "disable_rendering":
                    result = self._robot.disable_rendering()
                elif method == "enable_rendering":
                    result = self._robot.enable_rendering()
                elif method == "reset":
                    # TODO rename stuff so that this handle reset is truly just reset
                    obs, info = self._robot._handle_reset(**args)
                    with self._policy_guidance_lock:
                        if self._policy_guidance_action is not None:
                            obs["policy_guidance_chunk"] = self._policy_guidance_action.copy()
                    result = obs, info
                elif method == "check_metrics":
                    if hasattr(self._robot, "check_metrics"):
                        result = self._robot.check_metrics()
                    else:
                        result = {"error": "check_metrics not supported"}
                elif method == "get_env_config":
                    result = self._robot.get_env_config()
                elif method == "set_eval_benchmark_indices":
                    result = self._robot.set_eval_benchmark_indices(**args)
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass
                # Timeout occurred, check if the stop event is set
            except (zmq.error.ContextTerminated, zmq.error.ZMQError):
                break  # Socket/context was closed during shutdown

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._socket.close()
        except zmq.error.ZMQError:
            pass
        try:
            self._context.term()
        except zmq.error.ZMQError:
            pass


class GripperState(str, enum.Enum):
    OPEN = 1
    CLOSE = 0


@dataclass
class PathSegment:
    path_type: str = field(init=False)


@dataclass
class TrajectoryPathSegment(PathSegment):
    path: np.ndarray
    gripper_pos: float
    gripper_velocity: float = 0.2
    threshold: float = 1e-2

    def __post_init__(self):
        self.path_type = "trajectory"


@dataclass
class GripperPathSegment(PathSegment):
    target_state: GripperState
    num_steps: int = 160

    def __post_init__(self):
        self.path_type = "gripper"


@dataclass
class SplatSimCamera:
    camera: Optional[Camera]
    pipeline: Optional[PipelineParams]
    background: Optional[torch.tensor]
    tracked_link_index: Optional[int] = None
    # Fisheye rendering via gsplat
    camera_model: str = "pinhole"  # "pinhole" or "fisheye"
    intrinsic_matrix: Optional[torch.Tensor] = None  # [3,3] fisheye K on CUDA
    radial_coeffs: Optional[torch.Tensor] = None  # [4] (k1,k2,k3,k4) on CUDA


# Fisheye -> pinhole rectification zoom. The wrist lens is an ULTRA-wide GoPro
# fisheye; both render backends present it to the policy as a RECTIFIED "wide"
# (max-mode-like) pinhole view, never as the raw circular fisheye:
#   * splat path   — renders fisheye, then undistorts (_rectify_fisheye_image)
#                    with K scaled by this factor.
#   * pybullet path — has no fisheye projection, so it renders a pinhole
#                    directly at the EQUIVALENT rectified FoV
#                    (_effective_fovy_rad).
# Shared here so the two backends can never drift apart — bump it in one place
# and both the splat wrist and the pybullet wrist re-zoom together.
FISHEYE_RECTIFY_ZOOM = 2.2

# Wrist camera fisheye calibrations indexed by wrist_cam_ver. Version 0 is
# pinhole (no entry here). New calibrations from
# scripts/calibrate_camera_intrinsics.py should be appended with the next id.
WRIST_CAM_FISHEYE_CALIBRATIONS: Dict[int, Dict[str, Any]] = {
    1: {
        "CAL_W": 2704, "CAL_H": 2028,
        "CAL_FX": 775.5615, "CAL_FY": 778.0103,
        "CAL_CX": 1343.6974, "CAL_CY": 1005.3416,
        "D": [-0.0232652411, -0.0160767049, 0.0, 0.0],
    },
    2: {
        "CAL_W": 1920, "CAL_H": 1080,
        "CAL_FX": 777.86654216, "CAL_FY": 767.71982274,
        "CAL_CX": 973.16480901, "CAL_CY": 524.25398954,
        "D": [0.16369808, -0.15318689, 0.10608916, -0.02891525],
    },
}


class PybulletRobotServerBase:
    MAX_TRAJECTORY_COUNT = 500
    # ── URDF-specific self-collision skip pairs ──────────────────────────────
    # Non-adjacent link pairs to EXCLUDE from self-collision checks. Use for
    # URDF link pairs that are structurally close at every reachable joint
    # config — without skipping them, any non-zero self_collision_clearance
    # would flag every valid pose. Tied to the URDF, not the scenario, so a
    # CLASS attribute is the natural home: subclass envs override it once;
    # every consumer (env-side `is_robot_in_collision`, the trajectory
    # generator, the LeRobot SA wrapper via the dispatched oracle env
    # config) reads from this single source of truth.
    #
    # Default: no skips (suitable for URDFs without geometrically-close
    # non-adjacent link pairs). Override at class scope in subclass envs
    # — see `SmallEnginePybulletRobotServer` for the UR robot's
    # `[(0, 2)]` (base_link vs upper_arm_link, ~4 mm apart due to the
    # shoulder bracket).
    #
    # Authoring rule: pair tuples should be `(int, int)` and order doesn't
    # matter ((a,b) is treated identically to (b,a) downstream).
    #
    # NOTE: this holds only the ROBOT-SPECIFIC (arm/wrist) pairs, addressed by
    # index. The gripper-internal pairs are shared across every robot that
    # mounts the Robotiq 2F-85 and are declared BY NAME in
    # `GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES` below — they're resolved to this
    # robot's indices and merged in at construction (see
    # `_init_self_collision_skip_pairs`). So after __init__,
    # `self.SELF_COLLISION_SKIP_PAIRS` = these class-declared arm pairs + the
    # resolved gripper pairs. Subclasses only need to list arm pairs.
    SELF_COLLISION_SKIP_PAIRS: ClassVar[list[tuple[int, int]]] = []

    # ── Gripper self-collision skip pairs, shared by NAME across robots ───────
    # The Robotiq 2F-85 is a 4-bar linkage whose inner-finger / inner-finger-pad
    # collision meshes overlap the inner knuckle by design (~13 mm). Any
    # collision query run with a non-zero self_collision_clearance (RRT planner,
    # trajectory generator) would otherwise flag every gripper-present pose.
    #
    # Declared BY LINK NAME (not index) because the gripper links are identical
    # across every robot that mounts this gripper, but their numeric indices
    # differ with the arm's joint count (e.g. `left_inner_finger` is index 11 on
    # the UR5 but 7 on the 3-joint planar arm). `_init_self_collision_skip_pairs`
    # resolves these to the loaded robot's indices — so the SAME definition
    # serves the UR envs and the planar debug env. For the UR5 they resolve to
    # exactly the original hardcoded (11,13),(12,13),(16,18),(17,18), keeping
    # small_engine's skippable set unchanged. Change the gripper skip contract
    # here, once, and every robot with this gripper follows.
    GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES: ClassVar[list[tuple[str, str]]] = [
        ("left_inner_finger", "left_inner_knuckle"),
        ("left_inner_finger_pad", "left_inner_knuckle"),
        ("right_inner_finger", "right_inner_knuckle"),
        ("right_inner_finger_pad", "right_inner_knuckle"),
        # Outer↔inner knuckle contact is a NATURAL Robotiq 2f resting contact
        # — the two knuckles sit ~2.35 mm apart at every gripper pose (measured
        # via SA-wrapper shield diagnostic). Not adding this pair means every
        # collision check with `self_collision_clearance ≥ 2.35 mm` (e.g. the
        # wrapper's future-chunk shield at 5 mm, or the planner at 5 mm)
        # falsely reports "in collision" regardless of arm pose. Add both
        # sides so a closed OR open gripper is always cleared. Distance is
        # preserved by the Robotiq 2f mimic-parallelogram — both knuckles
        # are children of `robotiq_arg2f_base_link` via mimic-tied revolute
        # joints, so their relative geometry stays constant even as the
        # gripper opens/closes.
        ("left_outer_knuckle", "left_inner_knuckle"),
        ("right_outer_knuckle", "right_inner_knuckle"),
        # Same story for outer_finger↔inner_knuckle: a CONSTANT ~5.7 mm resting
        # gap (measured identical across the full gripper open→close range, both
        # sides — the parallelogram keeps outer_finger and inner_knuckle a fixed
        # distance apart). Without skipping it, any self_collision_clearance ≥
        # 5.7 mm (e.g. the trajectory generator's default 1 cm) falsely reports a
        # self-collision at every pose, which flooded the planar env's random-q
        # probe. Never an actual collision (5.7 mm > 0 always), so safe to skip.
        ("left_outer_finger", "left_inner_knuckle"),
        ("right_outer_finger", "right_inner_knuckle"),
    ]

    def _resolve_link_name_pairs(self, name_pairs):
        """Map (link_name, link_name) pairs to (link_index, link_index) for the
        loaded robot URDF. Modular: the same names resolve to the right indices
        regardless of how many arm joints precede the gripper. Pairs whose links
        aren't present are silently dropped."""
        name2idx = {}
        num_joints = self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)
        for i in range(num_joints):
            info = self.pybullet_client.getJointInfo(self.splatsim_robot.sim_id, i)
            name2idx[info[12].decode("utf-8")] = i
        return [
            (name2idx[a], name2idx[b])
            for a, b in name_pairs
            if a in name2idx and b in name2idx
        ]

    # Parent-child link name-pairs that ARE geometrically able to collide
    # despite being adjacent, and therefore SHOULD be checked. Default: empty
    # — the URDF-adjacency skip in `check_links_in_collision` catches natural
    # joint-pivot overlap that every UR5/Robotiq-style robot has by design
    # (small_engine relies on this to avoid false-firing at every arm config).
    # Override on robots with URDFs where the child link's body can fold onto
    # the parent's body at extreme joint angles — e.g., the planar 3-DOF arm
    # where |joint_2| ≈ π folds link_2 back onto link_1. Names are resolved to
    # PyBullet link indices at URDF-load time by `_init_check_adjacent_pairs`;
    # the resolved integer list is stored on `self.SELF_COLLISION_CHECK_ADJACENT_PAIRS`
    # and consumed by `is_robot_in_collision` + the RRT/shield checks.
    CHECK_ADJACENT_LINK_PAIRS_NAMES: ClassVar[list[tuple[str, str]]] = []
    # Per-instance resolved (int, int) form of CHECK_ADJACENT_LINK_PAIRS_NAMES.
    # Populated by `_init_check_adjacent_pairs` right after URDF load.
    SELF_COLLISION_CHECK_ADJACENT_PAIRS: ClassVar[list[tuple[int, int]]] = []

    def _init_check_adjacent_pairs(self) -> None:
        """Resolve `CHECK_ADJACENT_LINK_PAIRS_NAMES` (link-name pairs) into
        PyBullet link indices and stash on `self.SELF_COLLISION_CHECK_ADJACENT_PAIRS`.
        Called once, right after the robot URDF is loaded (in the same slot
        as `_init_self_collision_skip_pairs`) so every consumer sees the
        resolved list. Empty class default → empty instance list = skip all
        adjacent pairs (legacy behavior)."""
        resolved = self._resolve_link_name_pairs(
            list(type(self).CHECK_ADJACENT_LINK_PAIRS_NAMES)
        )
        # Dedup while preserving order; treat (a,b) == (b,a).
        seen = set()
        merged = []
        for a, b in resolved:
            key = frozenset((int(a), int(b)))
            if key not in seen:
                seen.add(key)
                merged.append((int(a), int(b)))
        # Per-instance shadow of the ClassVar (link indices depend on URDF).
        self.SELF_COLLISION_CHECK_ADJACENT_PAIRS = merged  # type: ignore[misc]

    def _init_self_collision_skip_pairs(self) -> None:
        """Merge the resolved gripper name-pairs into `SELF_COLLISION_SKIP_PAIRS`.

        Sets an instance attribute = the class-declared ARM pairs + the Robotiq
        gripper pairs resolved from `GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES`
        (only when a gripper is in use). Called once, right after the robot URDF
        is loaded and BEFORE the trajectory generator is built, so every consumer
        that reads `self.SELF_COLLISION_SKIP_PAIRS` (is_robot_in_collision,
        `_get_default_trajectory_gen_config`, `get_env_config`) sees the full,
        robot-specific list."""
        arm_pairs = list(type(self).SELF_COLLISION_SKIP_PAIRS)
        gripper_pairs = (
            self._resolve_link_name_pairs(self.GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES)
            if self.use_gripper else []
        )
        # Dedup while preserving order; treat (a,b) == (b,a).
        seen = set()
        merged = []
        for a, b in arm_pairs + gripper_pairs:
            key = frozenset((int(a), int(b)))
            if key not in seen:
                seen.add(key)
                merged.append((int(a), int(b)))
        # Per-instance shadow of the ClassVar: gripper indices can't be known
        # until the URDF is loaded, so they're resolved here, not at class scope.
        self.SELF_COLLISION_SKIP_PAIRS = merged  # type: ignore[misc]
        # Also resolve the check-adjacent whitelist (mirror mechanism, both
        # need URDF loaded to translate link names → indices).
        self._init_check_adjacent_pairs()

    def _apply_joint_damping(self) -> None:
        """Apply `JOINT_DAMPING` viscous friction to the arm DOFs (joints
        1..num_dofs). No-op when JOINT_DAMPING == 0 (the default). Modular home
        for the 'URDFs declare zero damping, real joints have friction' fix so
        every env — not just the planar one — gets it by setting the class attr."""
        if not self.JOINT_DAMPING:
            return
        for j in range(1, self.num_dofs() + 1):
            self.pybullet_client.changeDynamics(
                self.splatsim_robot.sim_id, j, jointDamping=self.JOINT_DAMPING
            )

    # Headroom multiplier applied when CONTROL_MAX_VELOCITY is None (servo cap
    # tied to the plan's max_joint_vel). MUST be > 1: with a 1.0x tie the servo
    # is velocity-clamped to exactly the reference speed, so any tracking error
    # accumulated during a sustained max-velocity section can NEVER close — it
    # plateaus until the plan decelerates. Observed in planar_3joint_3 ep33: a
    # 9 s single-joint max-speed sweep held a constant 0.24 rad (14 deg) lag.
    # 1.25x lets the servo catch up during cruise while still preventing the
    # violent waypoint-chasing overshoot the tie exists to avoid.
    CONTROL_VELOCITY_HEADROOM: ClassVar[float] = 1.25

    # DEBUG (--debug_fast_control): near-unlimited servo caps so the arm snaps
    # to each commanded target within a physics tick or two. Makes latency
    # artifacts (shadow-vs-splat lag, wrist-camera pose staleness, obs-vs-state
    # skew) visible: at debug speed a one-tick lag is a large pose error
    # instead of sub-millimeter. NEVER for data collection — the dynamics are
    # completely unrealistic.
    DEBUG_FAST_CONTROL_FORCE: ClassVar[float] = 1e5
    DEBUG_FAST_CONTROL_MAX_VELOCITY: ClassVar[float] = 1e3  # rad/s
    # The gripper mimic fingers are coupled to the drive joint only by
    # JOINT_GEAR constraints (normally maxForce=10 — see setup_gripper) and
    # the drive joint by its URDF effort limit. Debug-speed arm snaps swing
    # the finger links with inertial torques far above 10 N·m, so the
    # follower finger visibly flaps; scale both up in lockstep with the arm.
    DEBUG_FAST_CONTROL_GRIPPER_FORCE: ClassVar[float] = 1e4

    # NORMAL-mode mimic rigidity (real 2F-85 fingers are STRUCTURALLY rigid —
    # they never flap). When > 0, move_gripper position-holds every mimic
    # child at its gear-consistent angle with this force, targets refreshed
    # on every call so the holds can never jam the mimic at a stale target.
    # Measured effect: free-space arm motion is ≤5 deg asymmetry either way,
    # but CONTACT (fingers brushing an obstacle) bends the fingers 50+ deg at
    # the historical gear-only coupling — with strong holds the fingers stay
    # rigid and the ARM (CONTROL_FORCE) stalls against the obstacle instead,
    # matching real-robot behavior. TRADEOFF: while closing ON a grasped
    # object, finger force is capped by this value instead of the gear's
    # 10 N·m, so envs where realistic crush force matters should keep 0
    # (historical behavior). Approach/no-grasp envs can set it high.
    GRIPPER_MIMIC_HOLD_FORCE: ClassVar[float] = 0.0

    def _control_force(self) -> float:
        """POSITION_CONTROL force for the arm joints: `CONTROL_FORCE`, or the
        near-unlimited debug value when `debug_fast_control` is on."""
        if getattr(self, "debug_fast_control", False):
            return self.DEBUG_FAST_CONTROL_FORCE
        return self.CONTROL_FORCE

    # NOTE: debug_fast_control deliberately does NOT raise positionGain on the
    # hold motors. A gain-1.0 servo was tried and is UNSTABLE after kinematic
    # teleports: the teleported pose isn't a gravity equilibrium, so small
    # errors appear each tick and the stiff servo pumps energy into the
    # linkage — measured as sustained 30-100 deg finger-flap ringing. The
    # teleport in command_joint_state provides the speed; the motors (default
    # gain) only hold against gravity, which is stable.

    def _control_max_velocity(self) -> float:
        """Resolve the POSITION_CONTROL maxVelocity: an explicit
        `CONTROL_MAX_VELOCITY` float, or — when it's None — the trajectory
        generator's planned `max_joint_vel` times CONTROL_VELOCITY_HEADROOM
        (see that attr: an exact 1.0x tie turns transient lag into a
        persistent offset on long max-speed sections). Falls back to 3.14
        before the generator exists (construction-time teleports snap via
        resetJointState, so the cap is moot there anyway).
        `debug_fast_control` overrides everything with the near-unlimited
        debug cap."""
        if getattr(self, "debug_fast_control", False):
            return self.DEBUG_FAST_CONTROL_MAX_VELOCITY
        if self.CONTROL_MAX_VELOCITY is not None:
            return self.CONTROL_MAX_VELOCITY
        tg = getattr(self, "trajectory_generator", None)
        if tg is not None:
            return tg.config.max_joint_vel * self.CONTROL_VELOCITY_HEADROOM
        return 3.14

    # Robot splat / URDF name used to look up gaussians + collision assets
    # from `configs/object_configs/objects.yaml`. Read by:
    #   * `launch_nodes.py` as the default when `--robot_name` isn't passed
    #   * (indirectly) LeRobot's `SplatSimEnv.robot_name` and
    #     `SharedAutonomyConfig.robot_name` defaults — kept in sync manually,
    #     no cross-repo import path from Python config → LeRobot
    #
    # Subclass override lives at class scope so every consumer sees the SAME
    # canonical value without needing to pass it through every constructor.
    # Change here and nowhere else — bash scripts pass no `--robot_name` by
    # default, LeRobot side has its own matching default (or query via
    # `get_env_config()` at env-init).
    DEFAULT_ROBOT_NAME: ClassVar[str] = "robot_iphone"

    # ADDITIONAL pairs skipped ONLY by the env-side eval-terminate check
    # `is_robot_in_collision` (via `--env.terminate_on_collision=true` /
    # `check_metrics`) — NOT by the RRT planner, trajectory generator, or
    # controller-side collision predicates.
    #
    # Populate this with pairs the audit flags as CRITICAL_MUST_UNSKIP
    # (they kinematically penetrate at some workload configs) but that
    # PyBullet's constraint solver KICKS on when RRT paths go through
    # them — leaving them in the strict `SELF_COLLISION_SKIP_PAIRS`
    # produces recorded joint spikes. The wider `_eval_terminate_skip_pairs()`
    # helper below unions this list with `SELF_COLLISION_SKIP_PAIRS`; the
    # env's `check_metrics` passes the union to `is_robot_in_collision`
    # so eval-terminate silently accepts those URDF-mesh-overlap configs
    # while the planner keeps rejecting them.
    #
    # Empty base default → subclasses opt in per-URDF (see
    # `SmallEnginePybulletRobotServer` for the wrist-camera-mesh entries).
    SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA: ClassVar[list[tuple[int, int]]] = []

    def _eval_terminate_skip_pairs(self) -> list[tuple[int, int]] | None:
        """Union of the strict skip list + the eval-terminate-only extras.
        Consumers: `check_metrics`'s `is_robot_in_collision(...)` call.
        Kept as a method (not a class attr) so both class attributes can be
        overridden freely in subclasses without needing to keep a derived
        third attribute in sync. Returns None (not []) when the union is
        empty so `check_links_in_collision`'s skip-set builder path
        collapses cleanly."""
        combined = list(self.SELF_COLLISION_SKIP_PAIRS) + list(
            self.SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA
        )
        return combined or None

    @property
    def TABLE_LIMITS(self):
        """Compute table limits from ENV_CONFIG.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max), (0, 0)) computed from the table object in ENV_CONFIG.

        Raises:
            ValueError: If no table object is defined in ENV_CONFIG.
        """
        for obj in self.ENV_CONFIG.objects:
            if obj.name == "table" and isinstance(obj, CuboidObjectConfig):
                position = obj.position
                size = obj.size
                x_center, y_center = position[0], position[1]
                x_half, y_half = size[0] / 2, size[1] / 2
                return ((x_center - x_half, x_center + x_half), (y_center - y_half, y_center + y_half), (0, 0))
        raise ValueError("TABLE_LIMITS requested but no 'table' object is defined in ENV_CONFIG['objects'].")

    # Gym environment constants
    _max_episode_steps = 400 # Can overwrite this in a child class
    # 240Hz physics / 30Hz control
    _physics_steps_per_action = 8  # Can overwrite this in a child class

    # Per-joint viscous damping (N·m·s/rad) applied to the ARM DOFs at
    # construction. Real robot joints have viscous + Coulomb friction (bearings,
    # gears) plus servo velocity feedback, so nonzero damping is MORE physical
    # than the URDFs' declared `damping="0"` — it stops light arms from ringing
    # around goal waypoints. Default 0.0 preserves historical behavior for envs
    # that don't opt in; subclasses set a value appropriate to their arm's
    # inertia (heavy UR5 needs little/none; a light planar arm needs more).
    # Applied via `_apply_joint_damping()`; ideally calibrated to the real
    # robot's settling response.
    JOINT_DAMPING: ClassVar[float] = 0.0

    # POSITION_CONTROL gains used when driving the arm to a commanded joint
    # target (command_joint_state / teleport_joint_state's hold).
    #
    # CONTROL_MAX_VELOCITY caps how fast the PD servo chases a waypoint. The
    # trajectory is a ~30 Hz reference executed by a 240 Hz servo (each waypoint
    # held for several substeps), so the servo tries to CLOSE each held waypoint
    # fast — meaning to track the plan faithfully (not overshoot it) this cap
    # should be ~the planned per-joint velocity. Two modes:
    #   * a float -> fixed cap. Default 3.14 (historical UR5 value; fine for the
    #                heavy UR5 whose inertia keeps its force-limited peak near
    #                the plan even at a high cap).
    #   * None    -> TIE it to the trajectory generator's `max_joint_vel`, so the
    #                servo may move at exactly the planned velocity and no faster.
    #                Single source of truth — change `max_joint_vel` and both the
    #                planned motion AND the execution cap follow. Used by the
    #                light planar arm. Resolved via `_control_max_velocity()`.
    CONTROL_FORCE = 150.0
    CONTROL_MAX_VELOCITY: ClassVar[Optional[float]] = 3.14

    # ── Fast PyBullet-native camera (an alternative to splat rendering) ───────
    # When True, get_observations renders image observations with PyBullet's
    # getCameraImage (a fixed third-person camera) instead of the Gaussian
    # splat. This is cheap, needs no splat assets, and works for ANY env — so
    # it's the fast/portable image source (e.g. the planar debug env). Rendered
    # per active camera key, resized to 224x224 like the splat path. Independent
    # of RENDER_SPLATS / render_from_splat. Envs set the camera pose below.
    RENDER_PYBULLET_CAMERA: ClassVar[bool] = False
    # Fixed camera extrinsics (world frame) + intrinsics. Defaults frame the
    # planar env's X-Z workspace from the -Y side; other envs override.
    #
    # Two ways to specify the pose (pick one per env):
    #   * explicit eye/target/up (below), or
    #   * orbit params PYBULLET_CAMERA_{YAW,PITCH,DISTANCE} around
    #     PYBULLET_CAMERA_TARGET — matches resetDebugVisualizerCamera's
    #     yaw/pitch/distance, so an env can reuse the debug-view framing it
    #     already tuned. When YAW is not None the orbit form takes precedence.
    PYBULLET_CAMERA_EYE = (0.0, -1.6, 0.2)
    PYBULLET_CAMERA_TARGET = (0.0, 0.0, 0.2)
    PYBULLET_CAMERA_UP = (0.0, 0.0, 1.0)
    PYBULLET_CAMERA_YAW = None       # degrees; None -> use eye/target/up
    PYBULLET_CAMERA_PITCH = None     # degrees
    PYBULLET_CAMERA_DISTANCE = None  # meters
    PYBULLET_CAMERA_FOV = 60.0
    PYBULLET_CAMERA_NEAR = 0.05
    PYBULLET_CAMERA_FAR = 5.0
    # Native render resolution. Match the encoder's 224x224 input so the resize
    # modes are a no-op (letterbox/stretch of a 224 square = identity) — rendering
    # smaller and upscaling just blurs small task objects and wastes their pixel
    # budget. Lower these ONLY if you need headless (CPU tiny-renderer) speed and
    # accept softer small objects; GUI/OpenGL renders 224 in ~1-3 ms regardless.
    PYBULLET_CAMERA_WIDTH = 224
    PYBULLET_CAMERA_HEIGHT = 224

    # ── Splat shadow compositing (--splat_shadows / GUI checkbox) ─────────────
    # Depth-cue aid for the SPLAT render mode: shadows are computed from
    # PyBullet collision geometry (a camera-frustum ray per pixel finds
    # each surface point, a reversed light ray tests occlusion — see
    # `_raycast_splat_shadow_mask` for why it's raycast rather than a
    # shadow-mapped render) and multiplied into the splat image, so the arm
    # casts a visible contact shadow onto the engine/table, making "about to
    # crash" readable. The shadow lands on URDF collision geometry, not the
    # splat surface, so it can be off by the URDF-vs-splat mismatch — fine for
    # a proximity cue. NOTE: this modifies the images policies see/record;
    # keep it OFF for dataset recording and eval unless the covariate shift
    # is intentional.
    # World-frame position of the point light rays are cast toward. Overhead
    # and slightly to the side so the shadow separates from the caster
    # instead of hiding directly beneath it.
    SPLAT_SHADOW_LIGHT_DIRECTION: ClassVar = (0.4, 0.8, 3.0)
    # 0..1 — how dark a fully-shadowed pixel gets (0 = invisible, 1 = black).
    # Tuned by eye on the small-engine table: a subtle cue, not a lighting
    # effect.
    SPLAT_SHADOW_STRENGTH: ClassVar[float] = 0.20
    # Per-episode multiplicative jitter on SPLAT_SHADOW_STRENGTH: every
    # env.reset() resamples the EFFECTIVE strength as
    # STRENGTH * U(1 - J, 1 + J) (so 0.1 = +/-10%), via
    # `_resample_splat_shadow_strength` (called from _reset_episode_state,
    # so it applies to every env subclass sharing that reset path). Mild
    # visual domain randomization so a policy trained with shadows on
    # doesn't overfit to one exact shadow darkness. Draws from np.random,
    # which reset() seeds first — reproducible under seeded resets. 0
    # disables (fixed strength). Constant WITHIN an episode.
    SPLAT_SHADOW_STRENGTH_JITTER: ClassVar[float] = 0.1
    # False (default): EVERY body casts shadows — robot, engine, boxes —
    # including convex-hull self-shadowing, at IDENTICAL ray cost (only the
    # hit classification differs). True restores the original robot-only
    # casting: useful if scene shadows double-darken regions where the splat
    # already baked in the real capture-time shadows, or to keep the shadow
    # cue exclusively about the robot.
    SPLAT_SHADOW_ROBOT_ONLY: ClassVar[bool] = False
    # Camera rays that look through MORE robot collision thickness than this
    # (metres) are robot pixels — the splat shows the arm there, so they never
    # receive shadows. Thinner traversals are the silhouette band (collision
    # hulls fatter than the visuals) where shadows must continue across.
    # ~half a UR5 link diameter: grazing chords measure < ~3 cm, a straight
    # pass through one link ~8 cm. See the thickness filter in
    # _raycast_splat_shadow_mask — this replaced a layer-count cutoff that
    # made curled-arm self-overlap pixels flicker.
    SPLAT_SHADOW_MAX_ROBOT_THICKNESS: ClassVar[float] = 0.05
    # Contrast gain on the shade term before STRENGTH caps it. The raycast
    # mask is already full-contrast (a pixel is either lit or robot-shadowed),
    # so 1.0 = neutral; kept as a knob for envs layering softer masks.
    SPLAT_SHADOW_GAIN: ClassVar[float] = 1.0

    # Master switch for all Gaussian-splat assets. When True (default), the
    # constructor loads the robot's per-point labels, the robot + background
    # splats, and the base camera — everything needed to render. When False,
    # ALL of that is skipped: no labels.npy, no splat plys, no segmentation,
    # no background object, no base camera. The env then runs as a pure
    # PyBullet physics sim exposing oracle state (joint positions, EE pose,
    # per-object poses) via get_observations — fast, asset-free, no rendering.
    # Subclasses that only ever run non-rendered (oracle-info) set this False;
    # `background_splat_name` may then stay None. Orthogonal to the per-step
    # `render_from_splat` flag, which only gates rendering of already-loaded
    # splats; RENDER_SPLATS=False avoids loading them in the first place.
    RENDER_SPLATS: ClassVar[bool] = True

    # This is the default splat name. Overwrite it in a child class of PybulletRobotServerBase
    background_splat_name = None

    # This sets up the camera. None if it is the same as background_splat_name
    base_camera_splat_name = None

    # Hand-measured base-camera extrinsics override (SIM world frame; see
    # setup_camera_from_dataset docstring). Historically hardcoded at the
    # call site — values below reproduce that exactly for the engine envs.
    # Scene-specific envs (e.g. the vine env) override these class attrs to
    # frame their own scene. None disables the corresponding override and
    # falls back to the dataset camera's own extrinsics.
    BASE_CAMERA_OVERRIDE_XYZ: ClassVar = (0, 1.20, 0.61)
    BASE_CAMERA_OVERRIDE_RPY: ClassVar = (np.pi / 2 - 15 * np.pi / 180, np.pi, 0)
    BASE_CAMERA_OVERRIDE_DIST_INC: ClassVar = 0.55

    # COLMAP PINHOLE (undistorted sparse), camera_id=3 — intrinsics only; no wrist splat folder needed.
    wrist_colmap_camera_id = 3
    wrist_colmap_width = 943
    wrist_colmap_height = 530
    wrist_colmap_fx = 509.5795213749339
    wrist_colmap_fy = 509.5795213749339

    # Enum for serve modes
    class SERVE_MODES(enum.Enum):
        GENERATE_DEMOS = "generate_demos"

        INTERACTIVE = "interactive"

        GENERATE_TRAJECTORIES = "generate_trajectories"
        GENERATE_TRAJECTORIES_IDLE = "generate_trajectories_idle"

        EVAL_BENCHMARK = "eval_benchmark"
        EVAL_BENCHMARK_IDLE = "eval_benchmark_idle"

    # Serve modes eligible for `_sync_physics_to_client` gating. Only the
    # user-driven modes (INTERACTIVE / EVAL_BENCHMARK*) qualify: their main-
    # loop cadence exists purely to advance physics in the absence of client
    # commands, so replacing that cadence with client-driven stepping is a
    # clean 1:1 substitute. The trajectory-generation modes drive their own
    # planned commands from the main thread and can't cede physics timing
    # to an external client — sync mode is a no-op for them.
    _SYNC_ELIGIBLE_MODES = frozenset({
        SERVE_MODES.INTERACTIVE,
        SERVE_MODES.EVAL_BENCHMARK,
        SERVE_MODES.EVAL_BENCHMARK_IDLE,
    })

    # Alias to the shared DebugModes enum for backwards compatibility
    DEBUG_MODES = DebugModes

    @property
    def serve_mode(self) -> 'PybulletRobotServerBase.SERVE_MODES':
        """Current serve mode. Reads from the GUI when available."""
        if self._splatsim_gui is not None:
            return self.SERVE_MODES(self._splatsim_gui.mode)
        return self._serve_mode

    @serve_mode.setter
    def serve_mode(self, value: 'PybulletRobotServerBase.SERVE_MODES'):
        """Set the serve mode. Updates the GUI when available."""
        self._serve_mode = value
        if self._splatsim_gui is not None:
            self._splatsim_gui.set_mode(value.value)

    lower_limits = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
    upper_limits = [np.pi, 0, np.pi, np.pi, np.pi, np.pi]

    ENV_CONFIG: EnvConfig # To be set in a subclass

    # ── Oracle info (observation.environment_state) ───────────────────────────
    # Privileged world state (object poses) recorded as a SEPARATE
    # observation.environment_state feature (FeatureType.ENV) in EVERY dataset, so
    # a single recording can train BOTH image-based policies (ignore it) AND
    # oracle/state-based policies (consume it). Enabled by default for any env
    # with scene objects; set ORACLE_RECORD_ENV_STATE=False to opt out.
    #   ORACLE_STATE_COORD_INDICES — which position axes to record per object
    #     (default full 3D (x,y,z); planar arms override to (0,2) = x,z).
    #   ORACLE_STATE_INCLUDE_QUAT  — also record each object's (x,y,z,w) quaternion.
    #   ORACLE_OBJECT_NAMES        — explicit object subset (None = all
    #     ENV_CONFIG.objects, in that STABLE order so the layout is fixed).
    #   ORACLE_STATE_INCLUDE_EE_POS — ALSO append the end-effector's position
    #     (via get_current_ee_pose, wrist_camera_link frame) AFTER the object
    #     coords, so the object layout stays a stable prefix. The EE pos is
    #     redundant with observation.state's joint angles (FK), but giving
    #     it to a state-only policy explicitly removes the need to learn FK —
    #     goal-reaching becomes a direct env_state comparison.
    #   ORACLE_STATE_EE_COORD_INDICES — which position axes the EE entry uses.
    #     None (default) = same as ORACLE_STATE_COORD_INDICES. Override when
    #     objects and EE need DIFFERENT dimensionality — e.g. small_engine's
    #     boxes sit ON the table (z constant → record x,y only) while the EE
    #     moves in full 3D (record x,y,z).
    #   ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST — ALSO append per-link SIGNED
    #     minimum distances to any obstacle. For each link index in
    #     ORACLE_LINK_OBSTACLE_DIST_LINKS (or auto-detected movable arm links
    #     when None), we compute min over obstacles of PyBullet's getClosestPoints
    #     signed distance and append the resulting scalar. Negative values mean
    #     "the link is INSIDE the obstacle's inflated volume by this much" (out-
    #     of-the-box PyBullet semantics — see check_links_in_collision for the
    #     same signal). Obstacles here are the objects named in
    #     ORACLE_LINK_OBSTACLE_DIST_OBSTACLE_NAMES; if None we auto-select any
    #     ENV_CONFIG object whose name matches ORACLE_LINK_OBSTACLE_NAME_PATTERN
    #     (default: starts with "obstacle"). Missing obstacles or no-obstacle
    #     scenes get sentinel value ORACLE_LINK_OBSTACLE_DIST_SENTINEL (default
    #     1.0 m — "very far, safe") per link so the env_state layout stays a
    #     fixed width across scenarios that vary obstacle count.
    #     Positioned LAST in the env_state layout so the pre-existing prefix
    #     (objects + optional EE) stays intact for older checkpoints/datasets.
    #     Motivation: with EE xy in env_state, policies often shortcut-learn
    #     an EE-space controller that ignores obstacle geometry — because
    #     obstacles collide with ARM LINKS (not the EE), and joint-space
    #     reasoning about link poses is harder than "move EE toward goal".
    #     Adding link-obstacle distances directly gives the policy a
    #     salient obstacle signal it can't ignore.
    ORACLE_RECORD_ENV_STATE: ClassVar[bool] = True
    ORACLE_STATE_COORD_INDICES: ClassVar[tuple] = (0, 1, 2)
    ORACLE_STATE_INCLUDE_QUAT: ClassVar[bool] = False
    ORACLE_OBJECT_NAMES: ClassVar[Optional[List[str]]] = None
    ORACLE_STATE_INCLUDE_EE_POS: ClassVar[bool] = False
    ORACLE_STATE_EE_COORD_INDICES: ClassVar[Optional[tuple]] = None
    ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST: ClassVar[bool] = False
    # Per-env tolerances applied when a caller passes strict_goal_tolerances=True
    # to __init__. See `_apply_strict_goal_tolerances`. Defaults match the
    # historical `UprightRobotSmallEngineNewStrictPybulletRobotServer` values
    # (5 mm / 2°) — precise-manipulation appropriate. Reacher-style envs
    # should override up (e.g. planar → 1 cm) since the arm can't converge
    # to millimeters without unstable oscillation at the goal.
    STRICT_POS_TOLERANCE_M: ClassVar[float] = 0.005
    STRICT_QUAT_TOLERANCE_DEG: ClassVar[float] = 2.0
    ORACLE_LINK_OBSTACLE_DIST_LINKS: ClassVar[Optional[Tuple[int, ...]]] = None
    ORACLE_LINK_OBSTACLE_DIST_OBSTACLE_NAMES: ClassVar[Optional[List[str]]] = None
    ORACLE_LINK_OBSTACLE_NAME_PATTERN: ClassVar[str] = "obstacle"
    ORACLE_LINK_OBSTACLE_DIST_SENTINEL: ClassVar[float] = 1.0
    ORACLE_LINK_OBSTACLE_DIST_MAX_QUERY: ClassVar[float] = 1.0

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        use_gripper: bool = True,
        serve_mode: SERVE_MODES = SERVE_MODES.GENERATE_DEMOS,
        robot_name: str = "robot_iphone",
        camera_names: List[str] = ["base_rgb"],
        cam_i: int = 254,
        image_width: int = 640,
        image_height: Optional[int] = None,
        debug_mode: DEBUG_MODES = DEBUG_MODES.OFF,
        image_resize_modes: Optional[List[ImageResizeMode]] = None,
        eval_benchmark_repo_id: Optional[str] = None,
        eval_benchmark_subset: Optional[List[int]] = None,
        use_gsplat: bool = True,
        wrist_cam_ver: int = 2,
        headless: bool = False,
        render_from_splat: bool = True,
        render_mode: Optional['RenderMode'] = None,
        # Composite PyBullet-rendered shadows onto the SPLAT render as a depth
        # cue (see SPLAT_SHADOW_* class attrs). Default OFF — alters the images
        # policies see, so opt in per launch (--splat_shadows) or via the GUI
        # "Splat shadows" checkbox.
        splat_shadows: bool = False,
        # DEBUG: near-unlimited servo force + maxVelocity (see the
        # DEBUG_FAST_CONTROL_* class attrs) so the arm snaps to commanded
        # targets — for exposing latency artifacts (a one-tick lag becomes a
        # large visible pose error). Default OFF; never for data collection.
        debug_fast_control: bool = False,
        show_control_gui: bool = False,
        sync_physics_to_client: bool = False,
        physics_substeps_per_command: int = 8,
        # Tighten the success-tolerance thresholds so RRT-style corrections
        # (which converge to the exact goal pose) don't terminate the episode
        # under the loose eval-time thresholds — e.g. small_engine at 3 cm /
        # 10° would consider the arm "done" while the RRT is still 2 cm and
        # 8° from the target, cutting the recorded intervention chunk short
        # and starving the trained policy of the last-mile corrections.
        # Applied via `_apply_strict_goal_tolerances()` after ENV_CONFIG is
        # materialized; each env class picks its own strict values via
        # `STRICT_POS_TOLERANCE_M` / `STRICT_QUAT_TOLERANCE_DEG` class-vars
        # (defaults below match the historical small_engine strict variant).
        # Set True unconditionally in intervention-recording sim launches
        # (e.g. dagger_orchestrate); leave False for eval / freerun where
        # the loose thresholds are the intended success semantics.
        strict_goal_tolerances: bool = False,
        # DEBUG: make every non-robot object PHYSICS-transparent to the robot
        # (setCollisionFilterPair robot<->object = 0) while keeping it fully
        # present everywhere else: rendered, in oracle env_state, and in the
        # distance-based `in_collision` near-miss metrics (getClosestPoints
        # ignores collision filters). Lets closed-loop debugging (e.g. blend
        # ratio comparisons) run without contact dynamics confounding the
        # trajectories — the robot passes through obstacles, but the policy
        # still SEES them and metrics still report the would-be collisions.
        phantom_obstacles: bool = False,
    ):
        self._splatsim_gui = None
        # Monotonic env-mutation clock. Bumped by every discrete mutation of
        # sim state (commanded control ticks, teleports, resets, object
        # add/move/delete) and stamped into every get_observations() result
        # as obs["state_version"]. Mutating RPCs that clients need to order
        # against observations (teleport_joint_state) RETURN the post-bump
        # value, so a client can tell whether an observation in hand was
        # captured before or after its mutation — without trusting call
        # ordering in its own control loop. The ZMQ REP socket serializes
        # requests, so bump/stamp ordering is well-defined cross-process.
        self._state_version: int = 0
        self._phantom_obstacles = bool(phantom_obstacles)
        self._serve_mode = serve_mode
        self._eval_benchmark_repo_id = eval_benchmark_repo_id or ""
        self._eval_benchmark_subset: Optional[List[int]] = eval_benchmark_subset
        # Sync-physics-to-client mode: when True, the main serve loop stops
        # auto-stepping physics at 240 Hz. Instead, each `command_joint_state`
        # ZMQ call runs `physics_substeps_per_command` `stepSimulation()` calls
        # synchronously before returning to the client. This makes physics
        # advance ONLY in response to a client command, eliminating the "sim
        # races ahead while client is thinking" pathology visible with slow
        # policies (e.g. diffusion policy inference at chunk boundaries: the
        # client blocks for 100-500 ms while running the U-Net; without this
        # flag the sim keeps stepping and the last commanded target keeps
        # pulling the robot forward). Off by default (legacy async behavior).
        # Only affects `SERVE_MODES.INTERACTIVE` and `EVAL_BENCHMARK*` — the
        # trajectory-generation modes need their own async physics for planning.
        self._sync_physics_to_client: bool = bool(sync_physics_to_client)
        # Number of physics substeps per client command. Physics runs at 240 Hz
        # internally; a policy issuing commands at 30 Hz should set this to 8
        # so wall-clock physics rate matches the async default (240 Hz) when
        # the client is keeping up. Higher = coarser physics steps per command.
        self._physics_substeps_per_command: int = max(1, int(physics_substeps_per_command))
        # Sync-physics main-thread signaling (see below): the ZMQ handler
        # thread must NOT call `stepSimulation` directly when pybullet is in
        # GUI mode — the OpenGL context is bound to the main thread and calls
        # from a different thread deadlock waiting for context switch. Instead
        # `command_joint_state` posts a step-request to these primitives and
        # blocks; the main serve loop consumes the request, runs the steps on
        # the main thread, and signals completion. Also correct (and no-op)
        # in DIRECT mode — stepSimulation is thread-safe there, but routing
        # through the main thread costs one extra loop iter and keeps ONE
        # code path for both connection modes.
        self._sync_step_request_ticks: int = 0
        self._sync_step_lock: threading.Lock = threading.Lock()
        self._sync_step_pending_event: threading.Event = threading.Event()
        self._sync_step_done_event: threading.Event = threading.Event()
        # Sync-render primitives: mirror sync_step above, but for camera image
        # rendering. In headless (DIRECT) mode, PyBullet's camera path uses the
        # EGL plugin — and EGL contexts are bound to the thread that INITIALIZED
        # them (the main thread that first called p.getCameraImage). Calling
        # p.getCameraImage from the ZMQ handler thread returns a null GL vendor
        # on NVIDIA drivers and can segfault mid-render (observed as
        # `ven = (null)` in the sim log followed by SIGSEGV).
        # Fix: marshal the render call to the main serve loop via a signal
        # handshake, same shape as _sync_step_*. The handler thread posts a
        # request, blocks on _sync_render_done_event; the main loop consumes
        # the request in _consume_sync_render_request(), runs the render
        # inline on the main thread, and signals completion.
        # `_main_thread` is captured in serve() so _render_pybullet_camera can
        # skip the marshaling round-trip when it's already called on the main
        # thread (e.g. trajectory generation, in-process gym env step).
        self._sync_render_request_camera: Optional[str] = None
        # What the marshaled call should do: "rgb" (PyBullet camera render) or
        # "snapshot_masks" (atomic link-state + shadow-mask capture). See
        # _sync_main_thread_call.
        self._sync_render_request_kind: str = "rgb"
        # Link-state snapshot accompanying the render request, so the marshaled
        # wrist render uses the pose captured with the observation instead of
        # the live pose at main-thread consume time (see _render_pybullet_camera).
        self._sync_render_request_link_states = None
        self._sync_render_result: Optional[np.ndarray] = None
        self._sync_render_error: Optional[BaseException] = None
        self._sync_render_lock: threading.Lock = threading.Lock()
        self._sync_render_pending_event: threading.Event = threading.Event()
        self._sync_render_done_event: threading.Event = threading.Event()
        self._main_thread: Optional[threading.Thread] = None
        # Headless mode: connect pybullet in DIRECT (no GUI) for fast
        # physics-only use cases like trajectory replay + collision filtering.
        # The pybullet 3D window is suppressed and the PYBULLET-camera path
        # switches to EGL (TINY fallback). Gaussian-SPLAT rendering is
        # CUDA-side (gsplat, no pybullet GL dependency) and keeps working —
        # pick the image source with --render_mode.
        self._headless = headless
        # Decouple the pybullet 3D WINDOW from the Tkinter CONTROL panel: when set
        # alongside headless, pybullet still connects DIRECT (no OpenGL window,
        # EGL GPU rendering, no ~30 Hz render-loop throttle) BUT the "SplatSim
        # Controls" window is still launched — so you can pick modes / tune the
        # trajectory config / press Start from the panel while rendering fast.
        # Requires a display for Tkinter (a workstation, not a display-less node);
        # no-op unless headless (a GUI connection already shows the panel).
        self._show_control_gui = show_control_gui
        self._strict_goal_tolerances = bool(strict_goal_tolerances)
        if self._strict_goal_tolerances:
            self._apply_strict_goal_tolerances()
        self.robot_name = robot_name
        self.camera_names = camera_names
        self.cam_i = cam_i
        self.image_width = image_width
        self.image_height = image_height
        self.use_gripper = use_gripper
        self.use_gsplat = use_gsplat
        # Selects the wrist camera model used in get_wrist_camera (see
        # WRIST_CAM_FISHEYE_CALIBRATIONS for fisheye versions):
        #   0 = pinhole, using the base camera's intrinsics. Reproduces
        #       pre-fisheye datasets and is useful for A/B-testing the
        #       fisheye visual covariate shift.
        #   1 = fisheye, original 2704x2028 GoPro calibration.
        #   2 = fisheye, recalibrated 1920x1080 GoPro calibration.
        if wrist_cam_ver != 0 and wrist_cam_ver not in WRIST_CAM_FISHEYE_CALIBRATIONS:
            raise ValueError(
                f"Unknown wrist_cam_ver={wrist_cam_ver}; expected 0 (pinhole) or one of "
                f"{sorted(WRIST_CAM_FISHEYE_CALIBRATIONS.keys())} (fisheye)."
            )
        self.wrist_cam_ver = wrist_cam_ver
        # When the wrist camera is aligned to the real camera's mounting pose,
        # its optical center sits inside the gaussians of the physical camera
        # body that were captured in the robot splat — so the wrist view just
        # renders the black interior of that blob. When True, the gaussians
        # belonging to the wrist_camera_link segment are made transparent for
        # the wrist render only (the base/third-person view still shows the
        # camera). Set False to render the wrist camera body as-is.
        self.mask_wrist_camera_body = True
        # Cached robot-relative gaussian indices of the wrist camera body,
        # computed lazily from the KNN segmentation. None until first use.
        self._wrist_cam_occluder_rel_idx = None
        self.splatsim_robot = None
        self.splatsim_background = None
        self.scene_gaussian = None
        # Set by load_scenario_file(pin=True): reset() reuses this
        # arrangement instead of re-running the randomize/solvability
        # search. None = randomise as before.
        self._pinned_scenario = None
        self.base_camera = None
        self.wrist_camera = None
        self._lerobot_saver = None
        self._eval_benchmark_episode_index = -1
        self.pybullet_client = p
        self.splatsim_objects = []

        if isinstance(debug_mode, str):
            debug_mode = self.DEBUG_MODES(debug_mode)
        assert debug_mode in self.DEBUG_MODES, f"debug_mode must be one of {list(self.DEBUG_MODES)}, got {debug_mode}"
        self.debug_mode = debug_mode
        # Image resize modes: list of ImageResizeMode values; defaults to both letterbox and stretch
        if image_resize_modes is None:
            image_resize_modes = [ImageResizeMode.LETTERBOX, ImageResizeMode.STRETCH]
        self.image_resize_modes: List[ImageResizeMode] = [
            m if isinstance(m, ImageResizeMode) else ImageResizeMode(m)
            for m in image_resize_modes
        ]

        # load labels.npy (only needed for splat rendering — see RENDER_SPLATS)
        if self.RENDER_SPLATS:
            self.robot_labels = np.load(
                str(SPLATSIM_ROOT / "data" / "labels_path" / f"{self.robot_name}_labels.npy")
            )
            self.robot_labels = torch.from_numpy(self.robot_labels).to(device="cuda").long()
        else:
            self.robot_labels = None

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        print(f"Listening on {host}:{port}")

        # Populate this on the fly if it's needed
        self.base_cuboid_gaussians = None

        ## add stage
        self.stage = 0

        # Gates the per-step gsplat render in get_observations. When False, the
        # render block is skipped entirely (no prep_image_rendering /
        # render_image) and every {cam}_{mode} observation key is set to None —
        # physics, joint/EE state, metrics, and RRT all still run. Toggle at
        # runtime via disable_rendering()/enable_rendering(); set the initial
        # value here from the constructor for fast, image-free runs.
        # Image-observation source, selectable at launch (--render_mode) and at
        # runtime via the GUI dropdown. Two independent runtime gates back it:
        #   * do_render_from_splat     — Gaussian-splat render in get_observations
        #   * do_render_from_pybullet  — PyBullet getCameraImage in get_observations
        # Resolve the initial RenderMode: explicit `render_mode` wins; otherwise
        # fall back to the legacy `render_from_splat` bool + the env's
        # RENDER_PYBULLET_CAMERA class default (PyBullet takes precedence when
        # both are on, matching the pre-dropdown get_observations order).
        if render_mode is None:
            if self.RENDER_PYBULLET_CAMERA:
                render_mode = RenderMode.PYBULLET
            elif render_from_splat:
                render_mode = RenderMode.SPLAT
            else:
                render_mode = RenderMode.NONE
        render_mode = RenderMode(render_mode)  # accept enum or str
        self._apply_render_mode(render_mode)
        # Launch-time preference, kept separate from the runtime gates above:
        # paths that re-enable rendering after a temporary disable (e.g.
        # _render_and_save_episode in trajectory-gen mode) and the LeRobot
        # dataset schema guard consult these so the recorded-image contract
        # matches how the server was launched instead of a mid-run toggle.
        self._initial_render_mode = render_mode
        self._render_from_splat_default = (render_mode == RenderMode.SPLAT)

        # Composite PyBullet shadows onto the splat render (see SPLAT_SHADOW_*
        # class attrs). Runtime-toggleable via the GUI "Splat shadows" checkbox,
        # polled each serve-loop tick in _check_camera_rendering_toggle.
        self.splat_shadows = bool(splat_shadows)
        # DEBUG servo override — read by _control_force/_control_max_velocity.
        self.debug_fast_control = bool(debug_fast_control)
        if self.debug_fast_control:
            print(
                "[control] DEBUG_FAST_CONTROL ON: servo force "
                f"{self.DEBUG_FAST_CONTROL_FORCE:g}, maxVelocity "
                f"{self.DEBUG_FAST_CONTROL_MAX_VELOCITY:g} rad/s — latency "
                "debugging only, dynamics are unrealistic."
            )
        # Per-tick shadow masks keyed by camera name, captured atomically with
        # the link-state snapshot by _capture_obs_snapshot_and_masks and
        # popped by _composite_splat_shadows during the same tick's renders.
        self._tick_shadow_masks: Dict[str, np.ndarray] = {}
        # Per-tick robot screen silhouettes (mask resolution), written by
        # _protect_robot_silhouette and consumed by _composite_splat_shadows
        # to re-assert the protection at full render resolution.
        self._tick_shadow_robot_silhouettes: Dict[str, np.ndarray] = {}
        # Static-scene shadow cache keyed by camera name (see the cache block
        # in _raycast_splat_shadow_mask). Validated per computation against
        # scene/view signatures — never needs explicit invalidation.
        self._shadow_static_cache: Dict[str, dict] = {}
        # This episode's effective shadow strength, resampled per reset by
        # _resample_splat_shadow_strength. None (pre-first-reset) falls back
        # to the class SPLAT_SHADOW_STRENGTH.
        self._splat_shadow_strength_current: Optional[float] = None

        # Placeholder object for rendering purposes
        self.scene_gaussian = GaussianModel(3)

        self.grasp_poses = {}

        # Trajectory + goal validated during the most recent reset (see
        # `_check_scenario_solvable`). `randomize_objects` accepts a scene only
        # once a full RRT path to the goal is found; that path is stashed here so
        # the first recording can reuse it instead of re-planning. None until a
        # goal-directed env sets it; cleared when the scene changes elsewhere.
        self._cached_reset_trajectory = None
        self._cached_reset_goal_q = None

        # GUI connection by default; DIRECT (headless) when `headless=True`.
        # DIRECT skips OpenGL context creation entirely, so the process can run
        # without a display; physics-only operations (collision queries,
        # joint state teleport) and the CUDA-side splat render are unaffected
        # (the PYBULLET-camera path switches to EGL/TINY — see below).
        self._pb_client_id = self.pybullet_client.connect(
            p.DIRECT if self._headless else p.GUI
        )
        # In headless (DIRECT) mode getCameraImage would otherwise fall back to
        # the CPU software renderer (ER_TINY_RENDERER, ~80 ms/frame on the small-
        # engine scene). Load PyBullet's EGL plugin so ER_BULLET_HARDWARE_OPENGL
        # renders OFFSCREEN on the GPU (~a few ms) with no window — the fast path
        # for headless recording / eval. Best-effort: absent EGL we transparently
        # fall back to TINY (see `_render_pybullet_camera`). GUI mode already has
        # a GL context, so it doesn't need this (its getCameraImage is instead
        # rate-locked to the GUI's ~30 Hz render loop).
        self._egl_plugin_id = self._try_load_egl_renderer() if self._headless else None
        # Tell rrt_path_utils which physicsClient our subsequent calls target,
        # so its existing call sites that don't thread `physics_client_id=` keep
        # working when more than one PyBullet server is connected in this
        # process (e.g. lerobot's shared autonomy wrapper running alongside us).
        rrt_path_utils.set_default_client_id(self._pb_client_id)
        self.pybullet_client.setAdditionalSearchPath(
            str(SPLATSIM_ROOT.parent / "submodules" / "pybullet-playground-wrapper" / "pybullet_playground" / "urdf" / "pybullet_ur5_gripper" / "urdf")
        )
        # Enable GUI for trajectory generation controls (sliders/buttons)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        # Disable the visualizer's RGB/depth/segmentation PREVIEW tiles. With
        # them on (pybullet default), every getCameraImage call in GUI mode
        # additionally renders and copies preview buffers into the visualizer
        # window — a large per-frame tax on image-producing loops (traj-gen's
        # _render_and_save_episode, eval rendering). The main 3D view is
        # unaffected; only the small camera-preview widgets disappear.
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # Align the GUI visualizer's light with the splat-shadow light so the
        # shadows in the pybullet 3D window fall in the SAME direction as the
        # raycast shadows composited onto the splat render — with the pybullet
        # default light the two disagreed, which reads as a bug when comparing
        # views. No-op in headless DIRECT mode (no visualizer).
        self.pybullet_client.configureDebugVisualizer(
            lightPosition=list(self.SPLAT_SHADOW_LIGHT_DIRECTION)
        )

        # set time step
        self.pybullet_client.setTimeStep(1 / 240)
        # add gravity
        self.pybullet_client.setGravity(0, 0, -9.81)

        # For plane.urdf
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # TODO this mock class fails for the apple on plate task, etc
        class MockModelsLib:
            model_name_list = {}

        self.models_lib = MockModelsLib  # md.model_lib()

        self.splatsim_objects: List[SplatSimObject] = []
        self._skip_pairs: set = set()
        # `self.SELF_COLLISION_SKIP_PAIRS` (class attribute, declared at the
        # top of `PybulletRobotServerBase`) is the single source of truth for
        # URDF-specific self-collision exclusions. NOT an instance attr — so
        # subclass overrides are at class scope and propagate without an
        # __init__ pass.
        self.splatsim_robot: SplatSimObject = self.create_object(
            SplatObjectConfig(
                name="robot",
                splat_name=self.robot_name,
                randomize_pose=False,
                rotation_range_z=(0, 0),
                position_range_x=(0, 0),
                position_range_y=(0, 0),
                # Explicit z-range so the (fixed-base, never-moved) robot never
                # falls back to TABLE_LIMITS — lets tableless envs (e.g. the
                # planar oracle env) construct. Existing table envs already
                # resolved TABLE_LIMITS[2] to (0, 0), so behavior is unchanged.
                position_range_z=(0, 0),
                # When RENDER_SPLATS is False the robot loads its URDF (physics
                # + articulation) but no Gaussian splat / segmentation.
                load_splat=self.RENDER_SPLATS,
            )
        )

        # Resolve + merge the shared Robotiq gripper self-collision skip pairs
        # (by name) into SELF_COLLISION_SKIP_PAIRS now that the URDF is loaded,
        # BEFORE the trajectory generator reads them below.
        self._init_self_collision_skip_pairs()

        # Apply per-joint viscous damping to the arm DOFs (URDFs declare none).
        self._apply_joint_damping()

        # Initialize trajectory generation config and generator
        self.trajectory_generator = TrajectoryGenerator(
            pybullet_client=self.pybullet_client,
            pb_client_id=self._pb_client_id,
            robot_id=self.splatsim_robot.sim_id,
            joint_indices=list(range(1, self.num_dofs() + 1)), # excludes gripper
            env_config_name=self.ENV_CONFIG.name,
            get_ee_link_fn=lambda: self._get_ee_link_index(),
            splatsim_objects=self.splatsim_objects,
            wrist_camera_link_name=self.splatsim_robot.config.wrist_camera_link_name,
            trajectory_gen_config=self._get_default_trajectory_gen_config(),
            # Envs with a soft-cost field (EnvConfig.soft_cost) get cost-aware
            # trajectory generation; None for binary-obstacle envs (no-op).
            soft_cost_payload=getattr(self.ENV_CONFIG, "soft_cost", None),
        )

        self._setup_interactive_gui()

        # The background uses the robot's full splat, but crops out the robot.
        # Skipped entirely when not rendering (no background splat asset).
        if not self.RENDER_SPLATS:
            self.splatsim_background = None
        else:
            if self.background_splat_name is None:
                raise ValueError(f"background_splat_name has not been set for env {type(self)}")
            self.splatsim_background = self.create_object(
                SplatObjectConfig(
                    name="background",
                    splat_name=self.background_splat_name,
                    keep_within_aabb=False,
                    load_urdf=False,
                    is_articulated=False,
                    randomize_pose=False,
                    rotation_range_z=(0, 0),
                    position_range_x=(0, 0),
                    position_range_y=(0, 0),
                )
            )

        self.skip_recording_first = 0

        for object_config in self.ENV_CONFIG.objects:
            # Already adds the splatsim_object to self.splatsim_objects
            self.create_object(
                object_config=object_config,
            )

        # TODO put all trajectory saving into TrajectoryGenerator
        # trajectory path
        with open(resolve_splatsim_path("configs/folder_configs.yaml"), "r") as f:
            folder_config = yaml.safe_load(f)
        self.path = folder_config["traj_folder"]
        # get no of folders in the path
        if os.path.exists(self.path):
            self.trajectory_count = len(os.listdir(self.path))
        else:
            self.trajectory_count = 0

        # Base camera + wrist_rgb camera come from splat datasets, so they're
        # only set up when rendering. Without rendering, both are None/dummy;
        # the wrist_camera object is still created below so its tracked_link_index
        # (the EE reference link) can be resolved from the URDF.
        if not self.RENDER_SPLATS:
            self.base_camera = None
            # Dummy camera; tracked_link_index filled in by the EE-link block below.
            self.wrist_camera = SplatSimCamera(camera=None, pipeline=None, background=None)
        else:
            # Always set up the base camera because wrist_rgb is initialized from it
            if self.background_splat_name == self.base_camera_splat_name or self.base_camera_splat_name is None:
                self.splatsim_base_camera = self.splatsim_background
            else:
                self.splatsim_base_camera: SplatSimObject = self.create_object(
                    SplatObjectConfig(
                        name="base_camera",
                        splat_name=self.base_camera_splat_name,
                        keep_within_aabb=False,
                        load_urdf=False,
                        is_articulated=False,
                        randomize_pose=False,
                        rotation_range_z=(0, 0),
                        position_range_x=(0, 0),
                        position_range_y=(0, 0),
                    )
                )
            self.base_camera = self.setup_camera_from_dataset(
                self.splatsim_base_camera.config, cam_i=self.cam_i, use_train=True,
                override_xyz=self.BASE_CAMERA_OVERRIDE_XYZ,
                override_rpy=self.BASE_CAMERA_OVERRIDE_RPY,
                override_dist_inc=self.BASE_CAMERA_OVERRIDE_DIST_INC,
            )
            if self.splatsim_base_camera is not self.splatsim_background:
                self.delete_object(self.splatsim_base_camera.config.name)

            if "wrist_rgb" in self.camera_names:
                # Intrinsics from COLMAP cameras.txt (PINHOLE id=3); pose comes from get_wrist_camera().
                self.wrist_camera = SplatSimCamera(
                    camera=None,
                    pipeline=self.base_camera.pipeline,
                    background=self.base_camera.background,
                )
            else:
                # Make a dummy camera
                self.wrist_camera = SplatSimCamera(
                    camera=None,
                    pipeline=None,
                    background=None,
                )

        # Add the index of the wrist_camera_link to the wrist camera if it's available
        if self.splatsim_robot.config.wrist_camera_link_name is not None:
            wrist_camera_link_name = self.splatsim_robot.config.wrist_camera_link_name
            num_joints = p.getNumJoints(self.splatsim_robot.sim_id)
            for i in range(num_joints):
                info = p.getJointInfo(self.splatsim_robot.sim_id, i)
                if info[12].decode("utf-8") == wrist_camera_link_name:
                    self.wrist_camera.tracked_link_index = i
                    break
            if self.wrist_camera.tracked_link_index is None:
                raise ValueError(
                    f"Cannot find wrist camera link name {wrist_camera_link_name}"
                )
        else:
            raise ValueError(
                f"wrist_camera_link_name attribute not defined in object config of robot {self.robot_name}, yet wrist camera was requested"
            )

        # current gripper state
        self.current_gripper_action = GripperState.OPEN
        self.teleport_joint_state(self.splatsim_robot, self.splatsim_robot.config.articulation_config.initial_joint_positions)

        # Gym-related state
        self._step_count = 0
        self._episode_started = False

    def num_dofs(self) -> int:
        return 6

    def state_dim(self) -> int:
        """Width of observation.state = [joints, gripper] (proprioception only).
        Always num_dofs + 1. Privileged world state (object coords) is NOT packed
        in here — it goes into a SEPARATE observation.environment_state feature
        (see env_state_dim / oracle_environment_state), because policies like the
        diffusion policy require an image OR a distinct environment_state input
        and normalize the two feature types independently."""
        return self.num_dofs() + 1

    def _apply_strict_goal_tolerances(self) -> None:
        """Tighten the success-tolerance thresholds to the STRICT_* class-vars.

        Called once at __init__ when the caller passes
        `strict_goal_tolerances=True`. Motivation: RRT-style intervention
        recording plans until the arm reaches the EXACT goal pose (mm-scale),
        but the loose eval-time thresholds (e.g. 3 cm / 10° for small_engine,
        6 cm for planar) mark the episode "successful" long before RRT
        converges — the recorded chunk gets cut short and the trained policy
        never sees the last-mile corrections it will need to actually finish
        the task. Enabling strict tolerances at record time forces RRT to
        run until it truly reaches the goal, giving cleaner demonstration
        data. Env-time success semantics are unaffected: only the recording
        sim tightens, and only when the caller asks.

        Default handles both tolerance-representation patterns in the
        codebase:

          (1) `ENV_CONFIG.task: TaskConfig` (small_engine, vine) — replaced
              via `dataclasses.replace` at the instance level, shadowing the
              class ENV_CONFIG so sibling instances aren't affected.
          (2) `self.pos_tolerance_m` class attribute (planar) — reassigned
              at the instance level; the class attribute (and other
              instances) is left untouched.

        Envs that don't use either pattern (or want custom tightening logic
        beyond pos+quat) can override this method entirely.
        """
        # Pattern 1: TaskConfig-based tolerances.
        task = getattr(self.ENV_CONFIG, "task", None)
        if task is not None:
            import dataclasses as _dc
            new_task = _dc.replace(
                task,
                pos_tolerance_m=float(self.STRICT_POS_TOLERANCE_M),
                quat_tolerance_deg=float(self.STRICT_QUAT_TOLERANCE_DEG),
            )
            # Instance-shadow so we don't mutate the class-level ENV_CONFIG.
            self.ENV_CONFIG = _dc.replace(self.ENV_CONFIG, task=new_task)
        # Pattern 2: raw `pos_tolerance_m` class attribute (planar-style).
        # `hasattr` catches both class-attr and instance-attr; the assignment
        # creates an instance shadow either way.
        if hasattr(self, "pos_tolerance_m"):
            self.pos_tolerance_m = float(self.STRICT_POS_TOLERANCE_M)
        print(
            f"[strict_goal] tolerances tightened: "
            f"pos={self.STRICT_POS_TOLERANCE_M} m, quat={self.STRICT_QUAT_TOLERANCE_DEG}°"
        )

    def _oracle_object_names(self) -> List[str]:
        """Scene objects whose poses go into observation.environment_state, in a
        STABLE order (ENV_CONFIG.objects order, or the explicit ORACLE_OBJECT_NAMES
        subset) so the env-state layout is fixed across resets. Empty when this
        env records no oracle info."""
        if not self.ORACLE_RECORD_ENV_STATE:
            return []
        if self.ORACLE_OBJECT_NAMES is not None:
            return list(self.ORACLE_OBJECT_NAMES)
        env_config = getattr(self, "ENV_CONFIG", None)
        if env_config is None:
            return []
        return [obj.name for obj in env_config.objects]

    def _oracle_ee_coord_indices(self) -> tuple:
        """Axes recorded for the trailing EE entry: the explicit override, or
        the per-object indices when None (historical behavior — planar's (0,2)
        EE stays unchanged)."""
        if self.ORACLE_STATE_EE_COORD_INDICES is not None:
            return tuple(self.ORACLE_STATE_EE_COORD_INDICES)
        return tuple(self.ORACLE_STATE_COORD_INDICES)

    def _oracle_per_object_dim(self) -> int:
        """Number of scalars recorded per object: position coords + optional quat."""
        return len(self.ORACLE_STATE_COORD_INDICES) + (4 if self.ORACLE_STATE_INCLUDE_QUAT else 0)

    def env_state_dim(self) -> int:
        """Width of observation.environment_state (a FeatureType.ENV feature):
        privileged object poses for oracle/state-based policies, plus the
        trailing EE position when ORACLE_STATE_INCLUDE_EE_POS, plus the
        trailing per-link min-obstacle distances when
        ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST. Computed from the recorded
        objects × per-object dim, so it always matches
        oracle_environment_state(). 0 → no environment_state feature."""
        dim = len(self._oracle_object_names()) * self._oracle_per_object_dim()
        if self.ORACLE_RECORD_ENV_STATE and self.ORACLE_STATE_INCLUDE_EE_POS:
            dim += len(self._oracle_ee_coord_indices())
        if self.ORACLE_RECORD_ENV_STATE and self.ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST:
            # One scalar per link — width doesn't depend on obstacle count
            # (we take min-over-obstacles per link).
            dim += len(self._resolve_link_obstacle_dist_links())
        return dim

    def _resolve_link_obstacle_dist_links(self) -> Tuple[int, ...]:
        """Which PyBullet link indices to compute per-link obstacle distances for.

        Explicit ORACLE_LINK_OBSTACLE_DIST_LINKS wins; otherwise auto-detect
        the movable arm links via `pybullet.getJointInfo(...).jointType != FIXED`
        on the robot body (matches "actually-controllable arm segments"). Falls
        back to num_dofs() distinct link indices [1..num_dofs] when the robot
        isn't loaded yet (e.g. width query before serve()).
        Auto-detect skips the base link (index -1) which never moves and would
        always return the same static distance to obstacles.
        """
        if self.ORACLE_LINK_OBSTACLE_DIST_LINKS is not None:
            return tuple(int(i) for i in self.ORACLE_LINK_OBSTACLE_DIST_LINKS)
        # Auto-detect: enumerate movable joints on the robot body.
        robot_obj = getattr(self, "splatsim_robot", None)
        if robot_obj is None or not hasattr(robot_obj, "sim_id"):
            # Robot not loaded yet — env_state_dim() may be called during
            # config validation before serve(). Use num_dofs() as the width
            # (matches typical arm-joint count).
            return tuple(range(self.num_dofs()))
        movable = []
        n_joints = self.pybullet_client.getNumJoints(robot_obj.sim_id)
        for j in range(n_joints):
            info = self.pybullet_client.getJointInfo(robot_obj.sim_id, j)
            joint_type = info[2]
            # p.JOINT_FIXED = 4; anything else is movable (revolute, prismatic, ...)
            if joint_type != 4:
                movable.append(j)
        # Cap at num_dofs() so a gripper joint (JOINT_PRISMATIC) doesn't sneak
        # in — the callers care about arm-segment collision, not gripper.
        return tuple(movable[: self.num_dofs()])

    def _resolve_link_obstacle_dist_obstacles(self) -> List[str]:
        """Which scene objects count as 'obstacles' for the link-obstacle
        distance feature. Explicit ORACLE_LINK_OBSTACLE_DIST_OBSTACLE_NAMES
        wins; otherwise auto-select any object whose name starts with
        ORACLE_LINK_OBSTACLE_NAME_PATTERN. Returns an empty list for envs
        with no matching objects (e.g. the obstacle-free simple planar variant)
        — the caller then emits the sentinel for every link, keeping layout
        width fixed across envs at the cost of an uninformative signal in
        those envs.
        """
        if self.ORACLE_LINK_OBSTACLE_DIST_OBSTACLE_NAMES is not None:
            return list(self.ORACLE_LINK_OBSTACLE_DIST_OBSTACLE_NAMES)
        env_config = getattr(self, "ENV_CONFIG", None)
        if env_config is None:
            return []
        pattern = self.ORACLE_LINK_OBSTACLE_NAME_PATTERN
        return [obj.name for obj in env_config.objects if obj.name.startswith(pattern)]

    def _compute_link_obstacle_min_distances(self) -> List[float]:
        """Per-link signed minimum distance to any obstacle, in the order
        returned by _resolve_link_obstacle_dist_links().

        For each (link, obstacle) pair, runs `pybullet.getClosestPoints` with
        a query distance of ORACLE_LINK_OBSTACLE_DIST_MAX_QUERY (1 m default)
        and takes the smallest `contactDistance` (index 8) — signed such that
        negative means the link is INSIDE the obstacle's inflated volume by
        that amount (see check_links_in_collision for the same signal). Per
        link we take the min across obstacles. When no closest-point info is
        returned (obstacles farther than MAX_QUERY) OR when there are no
        obstacles at all (obstacle-free variants), returns the sentinel value
        ORACLE_LINK_OBSTACLE_DIST_SENTINEL for that link — "very far, safe."
        Layout width stays fixed at len(links) regardless of scene state.
        """
        links = self._resolve_link_obstacle_dist_links()
        obstacle_names = self._resolve_link_obstacle_dist_obstacles()
        sentinel = float(self.ORACLE_LINK_OBSTACLE_DIST_SENTINEL)
        if not obstacle_names:
            return [sentinel] * len(links)

        # Resolve obstacle names → PyBullet body ids via self.splatsim_objects
        # (canonical scene registry populated by create_object / load_urdf; the
        # same list check_links_in_collision reads through when the caller
        # passes obstacle_names). O(N_objects) per frame; N is tiny.
        name_to_sim_id = {
            obj.config.name: obj.sim_id
            for obj in getattr(self, "splatsim_objects", [])
            if hasattr(obj, "config") and hasattr(obj, "sim_id")
        }
        obstacle_ids = [name_to_sim_id[n] for n in obstacle_names if n in name_to_sim_id]
        if not obstacle_ids:
            return [sentinel] * len(links)

        robot_obj = getattr(self, "splatsim_robot", None)
        if robot_obj is None or not hasattr(robot_obj, "sim_id"):
            return [sentinel] * len(links)

        robot_id = robot_obj.sim_id
        cid = getattr(self.pybullet_client, "_client", 0)
        max_q = float(self.ORACLE_LINK_OBSTACLE_DIST_MAX_QUERY)

        out: List[float] = []
        for link_i in links:
            min_d = sentinel
            for obs_id in obstacle_ids:
                pts = self.pybullet_client.getClosestPoints(
                    bodyA=robot_id, bodyB=obs_id, distance=max_q,
                    linkIndexA=int(link_i), linkIndexB=-1,
                )
                for pt in pts:
                    d = float(pt[8])  # contactDistance (signed)
                    if d < min_d:
                        min_d = d
            out.append(min_d)
        return out

    def _get_default_trajectory_gen_config(self) -> TrajectoryGenModeConfig:
        # Use all the default values
        return TrajectoryGenModeConfig()

    def _traj_env_asserted_fields(self) -> Dict[str, Any]:
        """Trajectory-gen config fields THIS ENV owns: every field where the
        env's `_get_default_trajectory_gen_config()` differs from the plain
        TrajectoryGenModeConfig defaults. These encode environment IDENTITY
        (task goal pose, q_goal_bias, self-collision skip pairs, scene-tuned
        clearances/soft-cost) — not generation tuning — so an imported config
        file must never override them (see reassert_env_traj_config_fields).
        Computed by diff, so nothing is hardcoded per env."""
        import dataclasses as _dc

        env_cfg = self._get_default_trajectory_gen_config()
        plain = TrajectoryGenModeConfig()
        return {
            f.name: getattr(env_cfg, f.name)
            for f in _dc.fields(env_cfg)
            if getattr(env_cfg, f.name) != getattr(plain, f.name)
        }

    def reassert_env_traj_config_fields(self, cfg) -> List[str]:
        """Re-assert env-owned fields on a trajectory-gen config IN PLACE and
        return the names of fields that were changed.

        Call after applying an imported config file (GUI Import button,
        --traj_config_file). Motivation: config files are exported per
        SESSION, and importing one exported from a DIFFERENT env silently
        wipes this env's identity fields — observed as a planar-arm export
        (3-joint q_start, null goal) nulling the small-engine lever goal, so
        generation quietly produced random-goal episodes. Imports keep their
        generation TUNING (speeds, smoothing, counts); env identity always
        wins. Additionally clears q_start/q_goal/q_goal_bias whose length
        doesn't match this robot's DOF (cross-env files carry those as plain
        user tuning, so the ownership diff can't catch them)."""
        import copy as _copy

        changed: List[str] = []
        for name, val in self._traj_env_asserted_fields().items():
            if getattr(cfg, name) != val:
                setattr(cfg, name, _copy.deepcopy(val))
                changed.append(name)
        nd = self.num_dofs()
        for name in ("q_start", "q_goal", "q_goal_bias"):
            v = getattr(cfg, name, None)
            if v is not None and len(v) != nd:
                setattr(cfg, name, None)
                changed.append(f"{name}(cleared: {len(v)} joints != {nd} DOF)")
        return changed

    def get_joint_state(self) -> np.ndarray:
        # return self._joint_state
        joint_states = []
        num_joints = self.pybullet_client.getNumJoints(self.splatsim_robot.sim_id)
        for i in range(1, num_joints):
            joint_states.append(
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[0]
            )
        return np.array(joint_states)

    def load_urdf(self, splatsim_obj: SplatSimObject, physics_scale: float = 1.0):
        # This must be called after the gaussians are finalized
        # ex: after the gaussians are transformed to be in the simulator's coordinate frame
        use_fixed_base = splatsim_obj.config.use_fixed_base
        is_articulated = splatsim_obj.config.is_articulated
        base_position = splatsim_obj.config.base_position

        if is_articulated:
            flags = (
                self.pybullet_client.URDF_USE_IMPLICIT_CYLINDER
                | self.pybullet_client.URDF_USE_SELF_COLLISION
                | self.pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            )
        else:
            flags = 0

        if type(splatsim_obj.config) == SplatObjectConfig:
            # Find possible URDF config
            if splatsim_obj.config.name in self.models_lib.model_name_list:
                urdf_path = self.models_lib[splatsim_obj.config.splat_name]
            elif splatsim_obj.config.urdf_path is not None:
                urdf_path = resolve_splatsim_path(splatsim_obj.config.urdf_path)
                if not os.path.exists(urdf_path):
                    raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            else:
                raise ValueError(f"urdf_path not found for object {splatsim_obj.config.name}")

            # TODO possibly do custom quat
            quat = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
            object_loaded = self.pybullet_client.loadURDF(
                urdf_path,
                base_position,
                quat,
                globalScaling=physics_scale,  # Visual scaling (scaling_range_x/y/z) is applied to the gaussian splat at reset time via randomize_object_scale(), not at URDF load time.
                useFixedBase=use_fixed_base,
                flags=flags,
            )
            mass = self.pybullet_client.getDynamicsInfo(object_loaded, -1)[0]
        elif type(splatsim_obj.config) == CuboidObjectConfig:
            # Find primitive shape config
            # position, orientation, size
            # TODO orientation
            lx, ly, lz = splatsim_obj.config.size
            # position = splatsim_obj.config.position
            color_rgb = splatsim_obj.config.color_rgb

            # position = [
            #     position[0] * global_scaling,
            #     position[1] * global_scaling,
            #     position[2] * global_scaling,
            # ]

            # config.color_rgb uses the 0-255 int convention (shared with the
            # splat generator, which normalizes internally — see
            # create_cuboid_gaussians). create_box expects RGBA floats in
            # [0, 1]; passing the raw ints clamps every channel to 1.0 and the
            # pybullet visual renders white regardless of the configured
            # color. Normalize + append alpha so the URDF box matches the
            # generated splat color.
            rgba = tuple(float(c) / 255.0 for c in color_rgb) + (1.0,)
            # TODO check if this box is created with (0,0,0) at the center of the box
            object_loaded = create_box(lx, ly, lz, color=rgba)
            # set_pose(object_loaded, Pose(point=position))
            # TODO set orientation
            if use_fixed_base:
                splatsim_obj.config.mass = 0.0
            self.pybullet_client.changeDynamics(object_loaded, -1, mass=splatsim_obj.config.mass)
        else:
            raise ValueError(
                f"Could not parse object config for object name {splatsim_obj.config.name}"
            )

        splatsim_obj.sim_id = object_loaded

        # Set friction
        self.pybullet_client.changeDynamics(
            splatsim_obj.sim_id, -1, lateralFriction=1.5
        )
        self.pybullet_client.changeDynamics(
            splatsim_obj.sim_id, -1, rollingFriction=0
        )

    def load_gaussian_splat(self, splatsim_obj: SplatSimObject):
        # Most of these representations are in the splat frame, so we need to transform to simulator frame
        if type(splatsim_obj.config) == SplatObjectConfig:
            if splatsim_obj.config.ply_path is not None:
                splatsim_obj.gaussians.load_ply(splatsim_obj.config.ply_path)
            elif splatsim_obj.config.model_path is not None:
                model_path = resolve_splatsim_path(splatsim_obj.config.model_path)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path not found: {model_path}")

                pc_dir = os.path.join(model_path, "point_cloud")
                iteration = max(int(f.split("_")[-1]) for f in os.listdir(pc_dir))
                ply_path = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")
                splatsim_obj.gaussians.load_ply(ply_path)
            else:
                raise ValueError(f"Object {splatsim_obj.config.name} has no ply_path or model_path")
            
            # Transform the xyz, rotation, and shs features to the canonical frame (the world frame for the simulator)
            # We will work in the coordinate frame of the simulator from now on
            assert splatsim_obj.config.transformation is not None
            Trans_canonical = (
                torch.from_numpy(
                    np.array(splatsim_obj.config.transformation.matrix)
                )
                .to(device=splatsim_obj.gaussians.get_xyz.device)
                .float()
            )  # shape (4, 4)
            _ = transform_object(
                splatsim_obj=splatsim_obj,
                transform=Trans_canonical,
                # This transform_object is to go from splat frame to simulator frame. the transformation matrix already accounted for the base position
                use_base_position=False,  # False,
                inplace=True,
            )

            # Detach tensors before crop_splat since transform_object produces non-leaf tensors
            # Splat is loaded at scale=1; per-axis scaling is applied at reset time via randomize_object_scale()
            splatsim_obj.gaussians._xyz = splatsim_obj.gaussians._xyz.detach()
            splatsim_obj.gaussians._rotation = splatsim_obj.gaussians._rotation.detach()
            splatsim_obj.gaussians._opacity = splatsim_obj.gaussians._opacity.detach()
            splatsim_obj.gaussians._features_rest = splatsim_obj.gaussians._features_rest.detach()
            splatsim_obj.gaussians._features_dc = splatsim_obj.gaussians._features_dc.detach()
            splatsim_obj.gaussians._scaling = splatsim_obj.gaussians._scaling.detach()

            crop_splat(splatsim_obj, keep_within_aabb=splatsim_obj.config.keep_within_aabb)

        elif type(splatsim_obj.config) == CuboidObjectConfig:
            assert splatsim_obj.config.size is not None
            lx, ly, lz = splatsim_obj.config.size
            # CuboidObjectConfig does not use scaling_range; size is fixed at load time
            # Default to brown color for rendering
            cuboid_params = create_cuboid_gaussians(
                side_lengths=(lx, ly, lz),
                spacing=0.005,
                color_rgb=splatsim_obj.config.color_rgb,
            )
            splatsim_obj.gaussians._xyz = cuboid_params["_xyz"]
            splatsim_obj.gaussians._rotation = cuboid_params["_rotation"]
            splatsim_obj.gaussians._opacity = cuboid_params["_opacity"]
            splatsim_obj.gaussians._features_rest = cuboid_params["_features_rest"]
            splatsim_obj.gaussians._features_dc = cuboid_params["_features_dc"]
            splatsim_obj.gaussians._scaling = cuboid_params["_scaling"]
        else:
            raise ValueError("Could not load gaussian splat")
        
        # Disable gradients on this gaussian splat b/c we're not optimizing
        splatsim_obj.gaussians._xyz = splatsim_obj.gaussians._xyz.detach()
        splatsim_obj.gaussians._rotation = splatsim_obj.gaussians._rotation.detach()
        splatsim_obj.gaussians._opacity = splatsim_obj.gaussians._opacity.detach()
        splatsim_obj.gaussians._features_rest = splatsim_obj.gaussians._features_rest.detach()
        splatsim_obj.gaussians._features_dc = splatsim_obj.gaussians._features_dc.detach()
        splatsim_obj.gaussians._scaling = splatsim_obj.gaussians._scaling.detach()
        
        return splatsim_obj

    def _invalidate_reset_trajectory_cache(self) -> None:
        """Drop the reset-validated trajectory cache: the scene (or robot start
        pose) it was planned against has changed, so the cached path is stale.
        Guarded so it's safe to call during construction (before
        `trajectory_generator` exists)."""
        self._cached_reset_trajectory = None
        self._cached_reset_goal_q = None
        tg = getattr(self, "trajectory_generator", None)
        if tg is not None:
            tg._cached_base_traj = None

    def delete_object(self, object_name):
        self._bump_state_version()
        index = [splatsim_obj.config.name for splatsim_obj in self.splatsim_objects].index(
            object_name
        )
        splatsim_obj = self.splatsim_objects.pop(index)
        self._recompute_skip_pairs()

        # Explicitly delete some values
        del splatsim_obj.gaussians
        if splatsim_obj.sim_id is not None:
            p.removeBody(splatsim_obj.sim_id)

        # Invalidate scene gaussian buffers so they get reinitialized without the deleted object
        self._invalidate_scene_gaussian_buffers()
        self._invalidate_reset_trajectory_cache()

    def clear_temp_objects(self):
        self._bump_state_version()
        non_temp_object_names = [
            self.splatsim_robot.config.name,
        ] + (
            [self.splatsim_background.config.name]
            if self.splatsim_background is not None else []
        ) + [obj_cfg.name for obj_cfg in self.ENV_CONFIG.objects]
        all_object_names = [splatsim_obj.config.name for splatsim_obj in self.splatsim_objects]
        deleted_obj_names = []
        for obj_name in all_object_names:
            if obj_name not in non_temp_object_names:
                self.delete_object(obj_name)
                deleted_obj_names.append(obj_name)
        return deleted_obj_names

    def create_object(
        self,
        object_config: ObjectConfig,
    ):
        """
        Create a splatsim object from configs
        """
        self._bump_state_version()
        splatsim_obj = SplatSimObject(
            gaussians=GaussianModel(3),
            config=object_config,
            # set sim_id and mass later
        )

        if splatsim_obj.config.load_urdf:
            self.load_urdf(splatsim_obj)
        else:
            splatsim_obj.sim_id = None
            splatsim_obj.mass = 0

        if splatsim_obj.config.is_articulated:
            assert splatsim_obj.config.articulation_config is not None
            assert splatsim_obj.config.name == "robot", "Only the robot can be articulated for now"
            assert type(splatsim_obj.config) == SplatObjectConfig, "Only splat objects can be articulated for now"

            articulation_config = splatsim_obj.config.articulation_config

            if splatsim_obj.config.name == "robot":
                self.splatsim_robot = splatsim_obj
                if self.use_gripper:
                    self.setup_gripper()
                self.open_gripper()

            if articulation_config.joint_signs is None:
                # Default to all 1's
                articulation_config.joint_signs = [1] * len(articulation_config.initial_joint_positions)

            num_joints = self.pybullet_client.getNumJoints(splatsim_obj.sim_id)
            if len(articulation_config.initial_joint_positions) > num_joints:
                print(
                    f"Warning: Provided initial joint positions ({len(articulation_config.initial_joint_positions)}) exceed the number of joints ({num_joints}). Truncating to {num_joints} positions."
                )
                articulation_config.initial_joint_positions = articulation_config.initial_joint_positions[:num_joints]
                articulation_config.joint_signs = articulation_config.joint_signs[:num_joints]

            # Use the config to find these values.
            self.teleport_joint_state(splatsim_obj, splatsim_obj.config.articulation_config.initial_joint_positions)
            # Capture the transform-reference link poses at EXACTLY
            # initial_joint_positions (resetJointState just set every joint,
            # including the gripper mimic joints). get_curr_link_states uses
            # computeForwardKinematics=True, so this is valid without stepping.
            #
            # CRITICAL: the robot must stay at this pose until AFTER
            # load_gaussian_splat runs, and the settle loop must NOT run before
            # it. Two reasons:
            #   1. The gripper mimic children are no longer position-held (that
            #      jammed the mimic — see teleport_joint_state/setup_gripper), so
            #      stepping now would let move_gripper drive the gripper away
            #      from initial_joint_positions (toward open).
            #   2. load_gaussian_splat -> transform_object(inplace=True)
            #      RE-captures initial_link_poses at the CURRENT joint pose. If
            #      the gripper has settled open by then, that overwrites this
            #      reference with the open pose, and since the splat's per-link
            #      labels were assigned at initial_joint_positions in
            #      articulated_robot_pipeline, the gripper gaussians end up
            #      anchored to a mismatched reference and never open in-render.
            # So: capture here, load the splat at the same pose (re-capture is
            # then consistent), THEN settle below.
            initial_link_poses = get_curr_link_states(splatsim_obj.sim_id)
            articulation_config.initial_link_poses = initial_link_poses

        if splatsim_obj.config.load_splat:
            self.load_gaussian_splat(splatsim_obj)
        else:
            splatsim_obj.gaussians = None

        if splatsim_obj.config.is_articulated:
            assert splatsim_obj.config.articulation_config is not None
            assert splatsim_obj.config.name == "robot", "Only the robot can be articulated for now"
            assert type(splatsim_obj.config) == SplatObjectConfig, "Only splat objects can be articulated for now"
            articulation_config = splatsim_obj.config.articulation_config

            # Per-point segmentation only exists when a splat was loaded. With
            # load_splat=False (RENDER_SPLATS off) gaussians is None, so skip —
            # the articulation still works from the URDF alone (physics + FK).
            if splatsim_obj.config.load_splat:
                segmentation_labels = np.load(
                    resolve_splatsim_path("./data/labels_path/" + splatsim_obj.config.splat_name + "_labels.npy")
                )
                segmentation_labels = (
                    torch.from_numpy(segmentation_labels)
                    .to(device=splatsim_obj.gaussians._xyz.device)
                    .long()
                )
                splatsim_obj.segmentation_labels = segmentation_labels

                segmented_list = get_segmented_indices(
                    splatsim_obj=splatsim_obj,
                )
                articulation_config.segmented_list = segmented_list
                # Segmentation changed → drop the cached wrist-cam occluder indices
                # so they're recomputed from the new segmented_list on next render.
                self._wrist_cam_occluder_rel_idx = None

            # Now that the transform reference (initial_link_poses) is captured
            # and the splat is loaded/segmented — all at initial_joint_positions
            # — let physics settle for the live starting state. This is where
            # the gripper actually moves (move_gripper drives it per the gripper
            # command in initial_joint_positions), rendered relative to the
            # reference above.
            for _ in range(100):
                self.pybullet_client.stepSimulation()

        # Set the position of the object
        self.randomize_object_scale(splatsim_obj)
        self.randomize_object_pose(splatsim_obj)

        self.splatsim_objects.append(splatsim_obj)
        self._recompute_skip_pairs()

        # Invalidate scene gaussian buffers so they get reinitialized with the new object
        self._invalidate_scene_gaussian_buffers()
        self._invalidate_reset_trajectory_cache()

        return splatsim_obj

    def set_object_pose(
        self,
        object_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
        use_gravity: bool = True,
    ) -> None:
        """Set the pose of an object in the simulation."""
        self._bump_state_version()
        # if object_name not in [
        #     splatsim_obj.splat_name for splatsim_obj in self.splatsim_objects
        # ]:
        #     print(f"Object {object_name} not found in splat_name_list.")
        #     return

        object_i = [splatsim_obj.config.name for splatsim_obj in self.splatsim_objects].index(
            object_name
        )
        splatsim_obj = self.splatsim_objects[object_i]

        if splatsim_obj.sim_id is None:
            raise ValueError(
                "Cannot set pose of object not represented in pybullet (ex: has urdf)"
            )

        self.pybullet_client.resetBasePositionAndOrientation(
            splatsim_obj.sim_id, position, orientation
        )

        if not use_gravity:
            # Make the object static so that it doesn't move
            self.pybullet_client.changeDynamics(splatsim_obj.sim_id, -1, mass=0)
        else:
            self.pybullet_client.changeDynamics(
                splatsim_obj.sim_id,
                -1,
                mass=splatsim_obj.mass,
            )

        # An object moved → any reset-validated path is stale.
        self._invalidate_reset_trajectory_cache()

    def _apply_phantom_obstacles(self) -> None:
        """Disable robot<->object PHYSICS collision for every loaded object.

        No-op unless the server was constructed with phantom_obstacles=True.
        Distance queries (near-miss `in_collision`, planner clearances) are
        unaffected — getClosestPoints ignores collision filters — so metrics
        keep reporting would-be collisions while the robot passes through.
        Idempotent; call after objects are (re)created.
        """
        if not getattr(self, "_phantom_obstacles", False):
            return
        rid = self.splatsim_robot.sim_id
        n_links = self.pybullet_client.getNumJoints(rid)
        for obj in self.splatsim_objects:
            if obj is self.splatsim_robot or obj.sim_id is None or obj.sim_id == rid:
                continue
            for link in range(-1, n_links):
                self.pybullet_client.setCollisionFilterPair(
                    obj.sim_id, rid, -1, link, enableCollision=0
                )

    def _bump_state_version(self) -> int:
        """Advance the env-mutation clock; returns the new version.

        Call from every code path that mutates sim state. Observations are
        stamped with the current value at capture (see get_observations), so
        `obs["state_version"] < version_returned_by_a_mutation` ⇔ the obs
        predates that mutation.
        """
        self._state_version += 1
        return int(self._state_version)

    def teleport_joint_state(
        self, splatsim_obj: SplatSimObject, joint_state: Tuple[float, ...]
    ) -> int:
        """Set the joint states of an articulated object in the simulation and
        hold position. Returns the post-teleport state_version so the caller
        can detect observations captured before this teleport (stale)."""
        if not splatsim_obj.config.is_articulated:
            raise ValueError(f"Object {splatsim_obj.config.name} is not articulated.")
        if splatsim_obj.config.articulation_config is None:
            raise ValueError(f"Object {splatsim_obj.config.name} has no articulation config.")
        
        if splatsim_obj.sim_id is None:
            raise ValueError(
                "Cannot set joint states of object not represented in pybullet (ex: has urdf)"
            )

        num_joints = self.pybullet_client.getNumJoints(splatsim_obj.sim_id)
        if len(joint_state) > num_joints - 1:
            raise ValueError(
                f"Expected at most {num_joints - 1} joint states, got {len(joint_state)}."
            )

        signs = splatsim_obj.config.articulation_config.joint_signs

        # Snap EVERY provided joint to its target with resetJointState — this
        # includes the gripper's mimic joints, so the gripper starts in the
        # correct (e.g. open) pose at init.
        for i in range(0, min(len(joint_state), num_joints - 1)):
            self.pybullet_client.resetJointState(
                splatsim_obj.sim_id,
                i + 1, # Assuming the first joint index is 1 (0 is often a fixed joint), adjust if necessary
                joint_state[i] * signs[i],
            )

        # Hold ONLY the arm DOFs (joints 1..num_dofs) with POSITION_CONTROL.
        # The gripper's mimic CHILD joints must stay motor-free (they were put
        # in VELOCITY_CONTROL force=0 by __parse_joint_info__) so the
        # JOINT_GEAR mimic + move_gripper can actuate them; holding them here
        # with force=150 would overpower the gears (force=10) and freeze the
        # gripper. The gripper is actuated solely via move_gripper.
        for i in range(0, min(len(joint_state), self.num_dofs())):
            self.pybullet_client.setJointMotorControl2(
                splatsim_obj.sim_id,
                i + 1, # Assuming the first joint index is 1 (0 is often a fixed joint), adjust if necessary
                p.POSITION_CONTROL,
                targetPosition=joint_state[i] * signs[i],
                force=self._control_force(),
                maxVelocity=self._control_max_velocity(),
            )
        # step_physics=False keeps the teleport ATOMIC. Without this the
        # command's post-set physics-step loop (added by the sync-to-client
        # feature) would step 8× immediately after the resetJointState above,
        # letting the position controller pull the arm off the teleport
        # target — the exact opposite of what a teleport should do.
        self.command_joint_state(splatsim_obj, np.array(joint_state), step_physics=False)
        # Returned over ZMQ to the teleporting client (the RRT source), which
        # uses it to recognize observations captured before this teleport.
        return self._bump_state_version()

    def command_joint_state(
        self,
        splatsim_obj: SplatSimObject,
        joint_state: np.ndarray,
        step_physics: bool = True,
    ) -> None:
        self._bump_state_version()
        # Only drive the arm DOFs (joints 1..num_dofs). Anything beyond that is
        # the fixed ee/gripper-mount joints and the gripper's mimic joints.
        # Driving the gripper's mimic CHILD joints here with independent
        # POSITION_CONTROL (force=150) overpowers the JOINT_GEAR mimic
        # constraints (force=10) that move_gripper relies on, freezing the
        # gripper — so the gripper is actuated ONLY via move_gripper below.
        # (joint_state may carry 18 entries at init; the extra trailing values
        # must not be applied as per-joint position targets on gripper links.)
        if self.debug_fast_control:
            # KINEMATIC snap: teleport the arm straight to the target, then
            # the position holds below just maintain it. Servo-only "fast"
            # settings (huge force/maxVelocity/positionGain) still take ~3
            # ticks to converge and slam the linkage hard enough that the
            # constraint solver visibly flaps the gripper fingers (25-65 deg
            # transients, measured) — resetJointState sidesteps the dynamics
            # entirely: exact target next tick, zero inertial transient.
            # Contact physics during the motion is skipped (the arm can pass
            # through objects between commands) — debug only, never for data.
            for i in range(0, min(len(joint_state), self.num_dofs())):
                self.pybullet_client.resetJointState(
                    splatsim_obj.sim_id, i + 1, joint_state[i]
                )
        for i in range(0, min(len(joint_state), self.num_dofs())):
            self.pybullet_client.setJointMotorControl2(
                splatsim_obj.sim_id,
                i + 1, # Assuming the first joint index is 1 (0 is often a fixed joint), adjust if necessary
                p.POSITION_CONTROL,
                targetPosition=joint_state[i],
                # Set a more realistic force for the robot
                force=self._control_force(),
                maxVelocity=self._control_max_velocity(),
            )

        if splatsim_obj.config.name == "robot" and self.use_gripper:
            # Gripper is COMMANDED as an extra trailing entry AT INDEX
            # num_dofs() of the joint_state array — the caller convention
            # is [q_arm_0, ..., q_arm_{num_dofs-1}, q_gripper]. Callers
            # that only want to command the arm (e.g. RRT source teleport
            # where the gripper state must survive the RRT recovery
            # unchanged) can pass a shorter array — length == num_dofs() —
            # and the gripper move is silently skipped. Prevents
            # `IndexError: index N is out of bounds for axis 0 with size N`
            # from an arm-only teleport call; the gripper's current motor
            # target stays in effect from the last full-length command.
            if len(joint_state) > self.num_dofs():
                self.move_gripper((1 - joint_state[self.num_dofs()]) * 0.085)
                self.current_gripper_action = joint_state[self.num_dofs()]

        # Client-driven physics: step N times per commanded action. See
        # `_sync_physics_to_client` docstring on __init__. Only fires when:
        #   (a) the flag is on,
        #   (b) the current serve mode is eligible (auto-step is gated),
        #   (c) the ZMQ serve loop is actually running — otherwise this
        #       method is being called from the in-process `_physics_step`
        #       (SplatSimGymEnv.step path), which ALREADY loops
        #       stepSimulation itself; adding our N here would double-step,
        #   (d) `step_physics=True` — teleport_joint_state (which internally
        #       calls this to set the hold-position target after resetJointState)
        #       passes False so the teleport stays ATOMIC. Without this, a
        #       teleport would resetJointState → set target → then physics-step
        #       8 times, and the position controller would drag the arm away
        #       from where we just placed it — undoing the teleport.
        #
        # The stepSimulation itself runs on the MAIN thread via a signal
        # handshake — pybullet in GUI mode binds the OpenGL context to the
        # thread that called p.connect(p.GUI), and calling stepSimulation
        # from a different thread deadlocks waiting for the context. We're
        # on the ZMQ handler thread here; post the request to the main
        # thread's serve loop and block until it signals completion.
        if (
            step_physics
            and self._sync_physics_to_client
            and self.serve_mode in self._SYNC_ELIGIBLE_MODES
            and self._zmq_server_thread.is_alive()
        ):
            with self._sync_step_lock:
                self._sync_step_request_ticks = int(self._physics_substeps_per_command)
                self._sync_step_done_event.clear()
                self._sync_step_pending_event.set()
            # Wait for the main serve loop to run the requested steps.
            # Timeout is generous — the main loop wakes every 1/240 s and
            # each step is cheap; N=8 completes in well under a second even
            # with rendering. A missed signal (loop dead / GUI hang) surfaces
            # as a warning here rather than an indefinite ZMQ block.
            _completed = self._sync_step_done_event.wait(timeout=5.0)
            if not _completed:
                logger.warning(
                    "sync_physics_to_client: main-thread step timeout (5 s). "
                    "Serve loop may be blocked; skipping the substep and continuing."
                )
                with self._sync_step_lock:
                    self._sync_step_request_ticks = 0
                    self._sync_step_pending_event.clear()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    @property
    def render_mode(self) -> 'RenderMode':
        """Current image-observation source as a RenderMode, derived from the
        two runtime gates (PyBullet takes precedence if somehow both set)."""
        if self.do_render_from_pybullet:
            return RenderMode.PYBULLET
        if self.do_render_from_splat:
            return RenderMode.SPLAT
        return RenderMode.NONE

    def _apply_render_mode(self, mode: 'RenderMode') -> None:
        """Set the two runtime render gates from a single RenderMode. Mutually
        exclusive — at most one image source is active at a time."""
        mode = RenderMode(mode)
        self.do_render_from_splat = (mode == RenderMode.SPLAT)
        self.do_render_from_pybullet = (mode == RenderMode.PYBULLET)

    def _available_render_modes(self) -> List['RenderMode']:
        """Render modes this env can actually produce (for the GUI dropdown).
        SPLAT is offered only when splat assets are loaded (RENDER_SPLATS);
        NONE and the PyBullet camera work for every env."""
        modes = [RenderMode.NONE, RenderMode.PYBULLET]
        if self.RENDER_SPLATS:
            modes.insert(1, RenderMode.SPLAT)
        return modes

    def disable_rendering(self):
        # Legacy on/off API (ZMQ): turn OFF all image rendering.
        self._apply_render_mode(RenderMode.NONE)

    def enable_rendering(self):
        # Legacy on/off API (ZMQ): restore the launch-time render mode.
        self._apply_render_mode(getattr(self, "_initial_render_mode", RenderMode.SPLAT))

    def get_wrist_camera_transform(self, cached_link_states=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.wrist_camera.tracked_link_index is None:
            print("WARNING: No wrist camera index found")
            return None

        # Use link frame origin (indices 4/5), not CoM (indices 0/1), since
        # wrist_camera_link has no inertial and its frame is set by the joint xyz offset.
        link_idx = int(self.wrist_camera.tracked_link_index)
        if cached_link_states is not None and link_idx < len(cached_link_states):
            # Use cached state for synchronization
            cached_state = cached_link_states[link_idx]
            T_cw = np.array(cached_state["link_frame_pos"]).astype(np.float32)
            quat = cached_state["link_frame_q"]
        else:
            # Fall back to direct query (backward compatibility)
            link_state = p.getLinkState(
                self.splatsim_robot.sim_id,
                link_idx,
                computeForwardKinematics=True,
            )
            T_cw = np.array(link_state[4]).astype(np.float32)
            quat = link_state[5]

        R_cw = (
            np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3).astype(np.float32)
        )

        return T_cw, R_cw

    def get_wrist_camera(self, cached_link_states=None):
        transform_pair = self.get_wrist_camera_transform(cached_link_states=cached_link_states)
        if transform_pair is None:
            return None
        T_cw, R_cw = transform_pair

        T_wc = -R_cw.T @ T_cw

        # Fisheye calibration (from scripts/calibrate_camera_intrinsics.py);
        # preserve the wrist camera's native aspect ratio (CAL_W : CAL_H)
        # rather than conforming to the base camera's aspect.
        fisheye_cal = WRIST_CAM_FISHEYE_CALIBRATIONS.get(self.wrist_cam_ver)

        H = self.base_camera.camera.image_height
        if fisheye_cal is not None:
            # Preserve the fisheye's native aspect (CAL_W : CAL_H).
            W = int(round(H * fisheye_cal["CAL_W"] / fisheye_cal["CAL_H"]))
        else:
            # Pinhole: match base camera resolution exactly so both cameras
            # produce identical-shape frames downstream.
            W = self.base_camera.camera.image_width
        resolution = (W, H)

        fovx = self.base_camera.camera.FoVx
        fovy = 2 * np.atan(np.tan(self.base_camera.camera.FoVy / 2))

        colmap_id = self.wrist_colmap_camera_id
        uid = 0
        depth_params = None
        invdepthmap = None
        image_name = "wrist_camera"
        image = torch.zeros((3, resolution[0], resolution[1]), dtype=torch.float32)

        camera = Camera(
            resolution,
            colmap_id,
            R_cw,
            T_wc,
            fovx,
            fovy,
            depth_params,
            to_pil_image(image),
            invdepthmap,
            image_name,
            uid,
            scale=1,  # scale
        )

        if fisheye_cal is not None:
            # Aspect ratio is preserved, so sx == sy; scale K uniformly.
            scale = H / fisheye_cal["CAL_H"]
            fisheye_K = torch.tensor([
                [fisheye_cal["CAL_FX"] * scale, 0.0,                            fisheye_cal["CAL_CX"] * scale],
                [0.0,                            fisheye_cal["CAL_FY"] * scale, fisheye_cal["CAL_CY"] * scale],
                [0.0,                            0.0,                            1.0],
            ], dtype=torch.float32, device="cuda")
            # Distortion coefficients are resolution-independent.
            fisheye_D = torch.tensor(
                fisheye_cal["D"],
                dtype=torch.float32, device="cuda",
            )

            splatsim_camera = SplatSimCamera(
                camera=camera,
                pipeline=self.base_camera.pipeline,
                background=self.base_camera.background,
                camera_model="fisheye",
                intrinsic_matrix=fisheye_K,
                radial_coeffs=fisheye_D,
            )
        else:
            # Pinhole render using base camera's intrinsics (FoV inherited via the
            # Camera object's fovx/fovy above). Matches the pre-a161cad6 default.
            splatsim_camera = SplatSimCamera(
                camera=camera,
                pipeline=self.base_camera.pipeline,
                background=self.base_camera.background,
                camera_model="pinhole",
            )

        return splatsim_camera

    def _init_scene_gaussian_buffers(self):
        """Initialize pre-allocated buffers for scene_gaussian to avoid fragmentation.

        Called automatically on first render and after objects are created/deleted.
        """
        # Calculate total number of gaussians across all rendered objects
        total_gaussians = 0
        self._scene_gaussian_offsets = {}  # Map object name to (start, end) indices

        for splatsim_obj in self.splatsim_objects:
            if splatsim_obj.gaussians is None:
                continue
            if not splatsim_obj.config.load_splat:
                continue
            if self.debug_mode == DebugModes.NO_BACKGROUND and splatsim_obj == self.splatsim_background:
                continue

            n_gaussians = splatsim_obj.gaussians.get_xyz.shape[0]
            self._scene_gaussian_offsets[splatsim_obj.config.name] = (total_gaussians, total_gaussians + n_gaussians)
            total_gaussians += n_gaussians

        # Pre-allocate scene gaussian buffers
        device = 'cuda'
        self._scene_gaussian_buffers_initialized = True
        self.scene_gaussian._xyz = torch.zeros(total_gaussians, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._rotation = torch.zeros(total_gaussians, 4, device=device, dtype=torch.float32)
        self.scene_gaussian._opacity = torch.zeros(total_gaussians, 1, device=device, dtype=torch.float32)
        self.scene_gaussian._scaling = torch.zeros(total_gaussians, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._features_dc = torch.zeros(total_gaussians, 1, 3, device=device, dtype=torch.float32)
        self.scene_gaussian._features_rest = torch.zeros(total_gaussians, 15, 3, device=device, dtype=torch.float32)

        # Bump the generation so every per-object pose cache in
        # prep_image_rendering invalidates: the buffers were just reallocated (and
        # the offsets remapped), so nothing they previously held survives.
        self._scene_gaussian_generation = getattr(self, "_scene_gaussian_generation", 0) + 1

    def _invalidate_scene_gaussian_buffers(self):
        """Mark scene gaussian buffers as needing reinitialization.

        Call this when objects are created or deleted. The reinit bumps
        `_scene_gaussian_generation`, which also drops every per-object pose cache
        used by prep_image_rendering.
        """
        self._scene_gaussian_buffers_initialized = False

    def _prep_cache_hit(self, splatsim_obj, pose_key) -> bool:
        """True when this object's gaussians in the scene buffers are already up
        to date for `pose_key`, so prep_image_rendering can skip transforming it.

        `pose_key` must capture EVERYTHING the object's transform depends on — its
        pose for a rigid object, its link states for an articulated one. Paired
        with the buffer generation so a realloc / offset remap always forces a
        re-transform. On a miss the key is recorded, so the caller must go on to
        actually do the transform.
        """
        gen = getattr(self, "_scene_gaussian_generation", 0)
        if (
            splatsim_obj._cache.get("prep_pose_key") == pose_key
            and splatsim_obj._cache.get("prep_gen") == gen
        ):
            return True
        splatsim_obj._cache["prep_pose_key"] = pose_key
        splatsim_obj._cache["prep_gen"] = gen
        return False

    def _invalidate_prep_pose_cache(self):
        """Force prep_image_rendering to re-transform every rigid object next frame.

        The pose cache keys off an object's POSE, so a move is detected
        automatically. Call this only when an object's GAUSSIANS are mutated in
        place while it stays still (recolouring, opacity edits, crop_splat) —
        otherwise the buffers would keep serving the pre-edit splat.
        """
        for splatsim_obj in self.splatsim_objects:
            splatsim_obj._cache.pop("prep_pose_key", None)

    def prep_image_rendering(self, data, cached_link_states=None):
        # Initialize buffers on first call
        if not getattr(self, '_scene_gaussian_buffers_initialized', False):
            self._init_scene_gaussian_buffers()

        # Transform each object splat to be in the right pose and copy into pre-allocated buffers
        with torch.no_grad():
            for i in range(len(self.splatsim_objects)):
                splatsim_obj = self.splatsim_objects[i]
                if splatsim_obj.gaussians is None:
                    continue

                # Skip objects not in the offset map (filtered during init)
                if splatsim_obj.config.name not in self._scene_gaussian_offsets:
                    continue

                start_idx, end_idx = self._scene_gaussian_offsets[splatsim_obj.config.name]

                # Build output_slices dict for zero-copy writes directly into scene_gaussian buffers
                output_slices = {
                    '_xyz': self.scene_gaussian._xyz[start_idx:end_idx],
                    '_rotation': self.scene_gaussian._rotation[start_idx:end_idx],
                    '_opacity': self.scene_gaussian._opacity[start_idx:end_idx],
                    '_scaling': self.scene_gaussian._scaling[start_idx:end_idx],
                    '_features_dc': self.scene_gaussian._features_dc[start_idx:end_idx],
                    '_features_rest': self.scene_gaussian._features_rest[start_idx:end_idx],
                }

                if splatsim_obj.config.is_articulated:
                    assert (
                        splatsim_obj == self.splatsim_robot
                    ), "Other articulated objects are not implemented yet"

                    # DIRTY CHECK (articulated). The robot's gaussians are a pure
                    # function of its LINK STATES — initial_link_poses is static
                    # config — so if no link moved, the buffers already hold the
                    # right answer. This is by far the most expensive object in
                    # the scene (~99% of prep once rigid objects are cached), so
                    # skipping it turns an idle frame (GUI polling a paused scene,
                    # between eval episodes) from ~55 ms into ~1 ms. During an
                    # actual rollout the arm moves every frame, so this correctly
                    # misses every time and costs only the link-state query.
                    #
                    # The link states are resolved ONCE and used as BOTH the key
                    # and the transform input, so PyBullet is queried only once —
                    # keying on joint positions instead would risk disagreeing
                    # with a caller-supplied cached_link_states.
                    link_states = (
                        cached_link_states
                        if cached_link_states is not None
                        else get_curr_link_states(splatsim_obj.sim_id, use_link_centers=True)
                    )
                    pose_key = tuple(
                        (s["pos"], s["q"], s["link_frame_pos"], s["link_frame_q"])
                        for s in link_states
                    )
                    if self._prep_cache_hit(splatsim_obj, pose_key):
                        continue

                    # Gets transformations for all links of the robot based on the current simulation
                    transformations_list = get_transformation_list(splatsim_obj, cached_link_states=link_states)

                    # TODO generalize this to "every articulated object" instead of just the robot
                    transform_means(
                        splatsim_obj=splatsim_obj,
                        transformations_list=transformations_list,
                        use_base_position=True,
                        inplace=False,
                        output_slices=output_slices,
                    )

                else:
                    # DIRTY CHECK (rigid). A rigid object's gaussians only need
                    # re-transforming when its POSE changed — the result is
                    # otherwise bit-identical to what the buffers already hold.
                    # Static scenery (the background splat is usually the largest
                    # buffer in the scene) and idle props would otherwise burn a
                    # full transform every frame for nothing: ~29% of
                    # prep_image_rendering on the small-engine scene.
                    #
                    # The key is the raw pose straight from `data`, so ANY motion
                    # (including a teleport via set_object_pose) is picked up on
                    # the next frame automatically. Gaussians mutated in place
                    # without moving are the one case this can't see — call
                    # _invalidate_prep_pose_cache() there.
                    if splatsim_obj.sim_id is not None:
                        pose_key = (
                            tuple(data[splatsim_obj.config.name + "_position"]),
                            tuple(data[splatsim_obj.config.name + "_orientation"]),
                        )
                    else:
                        pose_key = "static"  # no sim body -> pinned at the origin
                    if self._prep_cache_hit(splatsim_obj, pose_key):
                        continue

                    if splatsim_obj.sim_id is not None:
                        # Reuse cached tensors and copy data to avoid allocating new GPU memory each step
                        if 'position_tensor' not in splatsim_obj._cache:
                            splatsim_obj._cache['position_tensor'] = torch.zeros(3, device="cuda", dtype=torch.float32)
                            splatsim_obj._cache['rotation_tensor'] = torch.zeros(4, device="cuda", dtype=torch.float32)
                            splatsim_obj._cache['rotation_rolled'] = torch.zeros(4, device="cuda", dtype=torch.float32)
                        cur_object_position = splatsim_obj._cache['position_tensor']
                        cur_object_position.copy_(torch.as_tensor(data[splatsim_obj.config.name + "_position"], device="cuda"))
                        rot_raw = splatsim_obj._cache['rotation_tensor']
                        rot_raw.copy_(torch.as_tensor(data[splatsim_obj.config.name + "_orientation"], device="cuda"))
                        # Roll quaternion from xyzw to wxyz format using pre-allocated tensor
                        cur_object_rotation = splatsim_obj._cache['rotation_rolled']
                        cur_object_rotation[0] = rot_raw[3]  # w
                        cur_object_rotation[1:4] = rot_raw[0:3]  # xyz
                    else:
                        # Static objects: cache a zero position and identity rotation (already in wxyz format)
                        if 'static_position' not in splatsim_obj._cache:
                            splatsim_obj._cache['static_position'] = torch.tensor([0, 0, 0], device="cuda").float()
                            # Identity quaternion in wxyz format: (1, 0, 0, 0)
                            splatsim_obj._cache['static_rotation'] = torch.tensor([1, 0, 0, 0], device="cuda").float()
                        cur_object_position = splatsim_obj._cache['static_position']
                        cur_object_rotation = splatsim_obj._cache['static_rotation']

                    transform_object(
                        splatsim_obj=splatsim_obj,
                        pos=cur_object_position,
                        quat=cur_object_rotation,
                        use_base_position=True,
                        inplace=False,
                        output_slices=output_slices,
                    )

    def get_pybullet_debug_camera_as_splat_camera(self) -> Optional[SplatSimCamera]:
        """Convert PyBullet's debug camera to a Camera object for Gaussian
        splatting. Returns None when no live visualizer camera exists —
        headless (DIRECT) connections have no debug visualizer, and
        getDebugVisualizerCamera then returns an all-zero (singular) view
        matrix; callers fall back to the normal base camera."""
        # This function is the inverse of this post: https://stackoverflow.com/a/75355212

        # Get PyBullet camera info
        camera_info = p.getDebugVisualizerCamera()
        # Pybullet view matrix is major-column order
        view_matrix = np.array(camera_info[2]).reshape(4, 4).T
        if abs(np.linalg.det(view_matrix)) < 1e-9:
            return None

        Tc = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        ).reshape(4, 4)

        T = np.linalg.inv(view_matrix) @ Tc

        R = T[:3, :3]
        t = T[:3, 3]

        R_cw_final = R
        T_wc_final = -R.T @ t

        # TODO is this needed?
        # scale = self.base_camera.scale if self.base_camera is not None else 1.0
        resolution = (
            self.base_camera.camera.image_width,
            self.base_camera.camera.image_height,
        )
        colmap_id = 0
        uid = 0
        depth_params = None
        invdepthmap = None
        image_name = "pybullet_debug_camera"
        image = torch.zeros((3, resolution[0], resolution[1]), dtype=torch.float32)

        new_camera = Camera(
            resolution,
            colmap_id,
            R_cw_final,  # Use the fixed rotation
            T_wc_final,  # Use the fixed translation
            self.base_camera.camera.FoVx,
            self.base_camera.camera.FoVy,
            depth_params,
            to_pil_image(image),
            invdepthmap,
            image_name,
            uid,
            scale=1,  # scale,
        )

        splatsim_camera = SplatSimCamera(
            camera=new_camera,
            pipeline=self.base_camera.pipeline,
            background=self.base_camera.background,
        )

        return splatsim_camera

    def set_pybullet_camera_to_match_base(self):
        """Set PyBullet's debug camera to match the base camera view."""

        if self.base_camera is None:
            print("No base camera available")
            return

        # Get base camera parameters
        R_cw = self.base_camera.camera.R
        T_wc = self.base_camera.camera.T
        scale = self.base_camera.camera.scale

        # Camera position in world space
        camera_pos = -R_cw.T @ T_wc
        camera_pos = np.array([camera_pos[0], camera_pos[2], camera_pos[1]])

        # good
        # camera_pos = np.array([-0.2063907, -6.2722306,  1.5897055]) * scale

        # bad
        # camera_pos = np.array([-0.2063907,  1.5897055,  6.2722306])
        # camera_pos = T_wc * scale

        # Camera's forward direction (camera looks down -Z axis)
        forward_cam = np.array([0, 0, -1])
        forward_world = R_cw @ forward_cam

        # Find target by ray-casting forward from camera
        # Use a reasonable distance (e.g., distance to origin)
        target_distance = np.linalg.norm(camera_pos)  # Distance to origin as estimate
        target = camera_pos + forward_world * target_distance

        # Or directly use origin if robot is there
        target = np.array([0.0, 0.0, 0.0])

        # Distance from camera to target
        distance = np.linalg.norm(target - camera_pos)

        # Compute yaw and pitch
        # Forward vector from camera to target
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Yaw: angle in XY plane
        yaw = np.rad2deg(np.arctan2(forward[1], forward[0]))

        # Pitch: angle from XY plane
        pitch = np.rad2deg(np.arcsin(forward[2]))

        print(f"Setting PyBullet camera to match base camera:")
        print(f"  Camera position: {camera_pos}")
        print(f"  Target: {target}")
        print(f"  Distance: {distance:.3f}")
        print(f"  Yaw: {yaw:.1f}°")
        print(f"  Pitch: {pitch:.1f}°")

        # Set PyBullet camera
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=list(target),
        )

        print("PyBullet camera updated!\n")

    def _get_wrist_cam_occluder_abs_indices(self) -> Optional[torch.Tensor]:
        """Absolute indices into ``scene_gaussian`` of the wrist camera body.

        These are the gaussians the KNN segmentation assigned to
        ``wrist_camera_link`` — i.e. the physical camera captured in the robot
        splat, which occludes the wrist view once the virtual camera is aligned
        to the real mounting pose. Returns None if masking isn't applicable
        (no tracked link, no segmentation, empty segment, or robot not yet in
        the scene_gaussian offset map).
        """
        if not self.mask_wrist_camera_body:
            return None
        if self.wrist_camera is None or self.wrist_camera.tracked_link_index is None:
            return None
        robot = self.splatsim_robot
        if robot is None or robot.config.articulation_config is None:
            return None
        segmented_list = robot.config.articulation_config.segmented_list
        link_idx = int(self.wrist_camera.tracked_link_index)
        if segmented_list is None or link_idx >= len(segmented_list):
            return None

        # Robot-relative indices (into the robot's own gaussian array) are stable
        # across steps, so compute once. The scene_gaussian start offset can
        # change when objects are added/removed, so add it fresh each call.
        if self._wrist_cam_occluder_rel_idx is None:
            rel = segmented_list[link_idx]
            if rel is None or len(rel) == 0:
                return None
            # rel is a CUDA long tensor at runtime (from get_segmented_indices);
            # as_tensor also handles a plain list without copying a tensor.
            self._wrist_cam_occluder_rel_idx = torch.as_tensor(
                rel, device="cuda"
            ).long()

        offsets = getattr(self, "_scene_gaussian_offsets", None)
        if not offsets or robot.config.name not in offsets:
            return None
        start_idx = offsets[robot.config.name][0]
        return self._wrist_cam_occluder_rel_idx + start_idx

    def render_image(self, camera_name, cached_link_states=None):
        if camera_name == "base_rgb":
            camera = None
            if self.debug_mode != DebugModes.OFF:
                # Mirror the pybullet debug-visualizer camera in debug modes.
                # None in headless (DIRECT) runs — no visualizer exists, so
                # debug modes there keep their OTHER effects (e.g.
                # no_background hiding the background splat) but render from
                # the normal base camera instead of crashing on the
                # visualizer's zero view matrix.
                camera = self.get_pybullet_debug_camera_as_splat_camera()
            if camera is None:
                camera = self.base_camera
        elif camera_name == "wrist_rgb":
            camera = self.get_wrist_camera(cached_link_states=cached_link_states)
            if camera is None:
                return None
        else:
            raise ValueError(f"Unknown camera name {camera_name}")

        # For the wrist view, temporarily hide the gaussians of the physical
        # camera body (which the aligned virtual camera sits inside). Save and
        # restore raw opacity so the base view — rendered from the same
        # scene_gaussian buffers — is unaffected regardless of render order.
        occluder_idx = None
        saved_opacity = None
        if camera_name == "wrist_rgb":
            occluder_idx = self._get_wrist_cam_occluder_abs_indices()
            if occluder_idx is not None:
                saved_opacity = self.scene_gaussian._opacity[occluder_idx].clone()
                # Raw opacity → sigmoid; a large negative value renders as ~0 alpha.
                self.scene_gaussian._opacity[occluder_idx] = -1e4

        try:
            if camera.camera_model == "fisheye" or self.use_gsplat:
                rendering = render_gsplat(
                    camera, self.scene_gaussian
                )["render"].cpu().numpy()
            else:
                rendering = render(
                    camera.camera, self.scene_gaussian, camera.pipeline, camera.background
                )["render"].cpu().numpy()
        finally:
            if saved_opacity is not None:
                self.scene_gaussian._opacity[occluder_idx] = saved_opacity
        # If you index "depth" instead of "render", you get the depth image

        # Preprocess fisheye renders to an equivalent pinhole view so datasets
        # (lerobot, zarr) and policy inputs are always rectified.
        if camera.camera_model == "fisheye":
            # import pdb; pdb.set_trace()
            rendering = self._rectify_fisheye_image(rendering, camera)

        # Optional depth cue: darken the splat with PyBullet's shadow mask (see
        # SPLAT_SHADOW_* class attrs). AFTER rectification on purpose — the
        # pybullet mirror of a fisheye wrist is its rectified pinhole view, so
        # that's the geometry the mask matches. Skipped while the base view
        # tracks the pybullet debug-visualizer camera (debug mode): the mask is
        # computed from the normal base-camera pose and wouldn't line up.
        if self.splat_shadows and not (
            camera_name == "base_rgb" and self.debug_mode != DebugModes.OFF
        ):
            rendering = self._composite_splat_shadows(
                rendering, camera_name, cached_link_states=cached_link_states
            )

        # save the image (always as numpy array)
        return rendering

    def _compute_tick_shadow_masks(self, cached_link_states) -> Dict[str, np.ndarray]:
        """This tick's shadow masks for every active camera, consumed by
        `_composite_splat_shadows` during the same get_observations pass.

        MUST run back-to-back with the `get_curr_link_states` snapshot AND
        without physics stepping in between — the ray tests read live body
        poses, and any steps between snapshot and rays lag the shadow behind
        the rendered arm. `_capture_obs_snapshot_and_masks` guarantees this
        by running both inside one marshaled main-thread block."""
        masks: Dict[str, np.ndarray] = {}
        if not self.splat_shadows:
            return masks
        for camera_name in self.camera_names:
            if camera_name == "base_rgb" and self.debug_mode != DebugModes.OFF:
                continue  # render_image skips compositing for the debug camera
            try:
                masks[camera_name] = self._raycast_splat_shadow_mask(
                    camera_name, cached_link_states=cached_link_states
                )
            except Exception as e:
                print(
                    f"[render] splat shadow mask failed for {camera_name} "
                    f"({e}); frame will be unshadowed."
                )
        return masks

    def _composite_splat_shadows(
        self, rendering: np.ndarray, camera_name: str, cached_link_states=None
    ) -> np.ndarray:
        """Multiply the robot-shadow mask into a (C, H, W) splat render.

        Prefers the mask stashed by `_capture_obs_snapshot_and_masks` (computed at
        the exact link-state-capture instant of this tick — no lag vs the
        rendered robot). Direct `render_image` calls outside get_observations
        have no stash; those compute the mask now, pinning the CAMERA pose to
        `cached_link_states` while rays read live body poses. Any failure
        returns the frame unshadowed rather than killing the observation."""
        mask = self._tick_shadow_masks.pop(camera_name, None)
        if mask is None:
            try:
                mask = self._raycast_splat_shadow_mask(
                    camera_name, cached_link_states=cached_link_states
                )
            except Exception as e:
                print(f"[render] splat shadow mask failed ({e}); frame left unshadowed.")
                return rendering
        _, H, W = rendering.shape
        sil = self._tick_shadow_robot_silhouettes.pop(camera_name, None)
        if mask.shape != (H, W):
            upscale = H / float(mask.shape[0])
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            # Re-assert the robot silhouette AT FULL RESOLUTION. The upscale
            # above smears every mask edge over ~`upscale` pixels, which is
            # what let shadows computed for surfaces behind the arm bleed onto
            # the arm and flicker as the caster moved (see
            # _protect_robot_silhouette). ERODED by roughly the smear width so
            # the protection covers the arm itself without punching a lit halo
            # around it — the collision hull is already fatter than the arm's
            # splat visual, so shadows still run right up to the visible edge.
            if sil is not None and sil.any():
                sil_up = cv2.resize(
                    sil.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
                k = max(3, int(round(upscale)) | 1)  # odd kernel >= 3
                sil_up = cv2.erode(sil_up, np.ones((k, k), np.uint8))
                mask[sil_up.astype(bool)] = 1.0
        # shade: 0 = lit, 1 = fully shadowed. The 0.02 deadband keeps
        # near-lit mask values (blur fringes, any future soft mask source)
        # from tinting whole surfaces once SPLAT_SHADOW_GAIN scales the term;
        # the per-episode strength then caps how dark a fully-shadowed pixel
        # gets (SPLAT_SHADOW_STRENGTH, jittered per reset — see
        # _resample_splat_shadow_strength).
        shade = np.clip(
            float(self.SPLAT_SHADOW_GAIN) * np.maximum(0.0, (1.0 - mask) - 0.02),
            0.0, 1.0,
        )
        strength = (
            self._splat_shadow_strength_current
            if self._splat_shadow_strength_current is not None
            else float(self.SPLAT_SHADOW_STRENGTH)
        )
        atten = 1.0 - strength * shade
        return (rendering * atten[None, :, :]).astype(np.float32, copy=False)

    def _rectify_fisheye_image(self, img: np.ndarray, camera: SplatSimCamera) -> np.ndarray:
        """Undistort a fisheye-rendered image to a pinhole view at the same
        intrinsics K, so straight lines in the scene are straight again.
        Fisheye content outside the pinhole's FOV is cropped.

        Args:
            img: (C, H, W) float32 in [0, 1]
            camera: SplatSimCamera with camera_model="fisheye", intrinsic_matrix, radial_coeffs

        Returns:
            Rectified (C, H, W) float32 in [0, 1].
        """
        _, H, W = img.shape
        K = camera.intrinsic_matrix.detach().cpu().numpy().astype(np.float64)
        D = camera.radial_coeffs.detach().cpu().numpy().astype(np.float64).reshape(4, 1)

        # CREATE A NEW K FOR THE OUTPUT
        # A multiplier of 2.0 or 2.5 is likely what you need to match VLA data
        # Want to rectify to regular wide view. the wrist cam calibration rn is for ultra wide view.
        # Shared with the pybullet backend via FISHEYE_RECTIFY_ZOOM so both
        # wrist renders land on the SAME effective field of view.
        zoom_factor = FISHEYE_RECTIFY_ZOOM
        K_target = K.copy()
        K_target[0, 0] *= zoom_factor # fx
        K_target[1, 1] *= zoom_factor # fy

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_target, (W, H), cv2.CV_16SC2,
        )
        img_hwc = np.transpose(img, (1, 2, 0))
        rectified = cv2.remap(
            img_hwc, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.transpose(rectified, (2, 0, 1))

    def setup_camera_from_dataset(
        self,
        splatsim_obj_object_config: SplatObjectConfig,
        cam_i,
        use_train=True,
        override_xyz: Optional[Tuple[float, float, float]] = None,
        override_rpy: Optional[Tuple[float, float, float]] = None,
        override_dist_inc: Optional[float] = None,
    ) -> SplatSimCamera:
        """Load a camera from a gaussian-splat dataset, optionally overriding
        its extrinsics with a hand-measured pose.

        Intrinsics (``FoVx``, ``FoVy``, resolution, ``scale``) always come
        from the dataset. Extrinsics come from the dataset by default but can
        be overridden per-axis:

          * ``override_xyz`` — camera position in the SIMULATOR world frame
            (the same frame the robot's base sits in, typically at
            ~(0, 0, 0)). When provided, replaces the position component of
            the camera-to-world transform.
          * ``override_rpy`` — camera orientation as roll/pitch/yaw (radians)
            using PyBullet's Euler convention (``getQuaternionFromEuler`` —
            R about X, then P about Y, then Y about Z). When provided,
            replaces the rotation component.
          * ``override_dist_inc`` — signed distance along the camera's own
            view axis (metres). POSITIVE values pull the camera BACKWARD
            along the direction it's looking — i.e. further from whatever
            point it's aimed at. NEGATIVE values push it forward (closer).
            Applied AFTER ``override_xyz`` / ``override_rpy``, so the axis
            is derived from the (possibly overridden) rotation and the
            translation shifts along it. Use when the ``override_xyz`` you
            eyeballed puts the camera at the right ANGLE but too close or
            too far — bump ``override_dist_inc`` instead of recomputing
            the xyz coordinates by hand.

        Each override can be provided independently. All ``None`` (default)
        preserves the historical dataset-only behavior.

        Camera-forward convention: gaussian-splatting cameras look down
        their own +Z axis (COLMAP style), so the world-space forward
        direction is the 3rd column of the C2W rotation matrix. The
        ``override_dist_inc`` math relies on this — if a future refactor
        changes the camera-forward axis, revisit the sign convention below.
        """
        ###################################################################
        # Load the gaussian splat dataset to get camera parameters
        ###################################################################
        source_path = resolve_splatsim_path(splatsim_obj_object_config.source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path not found: {source_path}")

        model_path = resolve_splatsim_path(splatsim_obj_object_config.model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        parser = ArgumentParser(description="Testing script parameters")
        pipeline = PipelineParams(parser)
        model = ModelParams(parser, sentinel=True)
        dataset = model.extract(
            Namespace(
                sh_degree=3,
                # TODO get these from the object config
                source_path=source_path,
                model_path=model_path,
                images="images",
                depths="",
                resolution=-1,
                white_background=False,
                train_test_exp=False,
                data_device="cuda",
                eval=False,
            )
        )
        # arbitrary as long as it's consistent between initialization and setup_camera_from_dataset()
        # because we're going to overwrite the resolution when transforming the camera
        cam_scale = 2
        temp_gaussian_model = GaussianModel(3)
        scene = Scene(
            dataset,
            temp_gaussian_model,  # This is just used for camera initialization
            load_iteration=-1,
            shuffle=False,
            resolution_scales=[cam_scale],
            train_cam_indices=[cam_i],
            test_cam_indices=[],  # we're using train cameras
        )

        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ###################################################################
        # Load the camera from the dataset
        ###################################################################

        # Assume that self.cam_train_indices and self.cam_test_indices have already singled out
        # the camera of interest. Return the first camera in the list
        if use_train:
            camera = scene.getTrainCameras(scale=cam_scale)[0]
        else:
            camera = scene.getTestCameras(scale=cam_scale)[0]

        del scene
        del temp_gaussian_model
        del dataset

        ###################################################################
        # Transform the camera to be in the simulator frame instead of the splatsim background object's splat frame
        ####################################################################

        # 1. Define device and Trans_canonical
        device = camera.world_view_transform.device
        Trans_canonical_full = (
            torch.from_numpy(
                np.array(splatsim_obj_object_config.transformation.matrix)
            )
            .to(device=device)
            .float()
        )

        # 2. Get the camera's pose in the SPLAT'S LOCAL FRAME
        # V_original is the view matrix in the splat's local frame.
        # M_CW_original is the Camera-to-World (C2W) matrix in the splat's local frame.
        # We assume V_original's inverse is the C2W matrix.
        V_original_inv = torch.linalg.inv(camera.world_view_transform.clone()).T
        M_CW_original_local = V_original_inv

        # 3. Calculate the new ABSOLUTE Camera-to-World matrix
        # M_CW_world = T_world_from_local @ M_CW_local
        # This matrix now has the full scale (s) embedded in it.
        # background camera to background frame; background frame to world
        # want: background camera to world
        M_CW_new_world_scaled = torch.matmul(Trans_canonical_full, M_CW_original_local)

        # 4. Extract rotation, translation, and scale from the transformation
        # Use SVD to properly decompose the scaled transformation (same as transform_object)
        A = M_CW_new_world_scaled[:3, :3]  # Rotation + Scale
        t = M_CW_new_world_scaled[:3, 3]  # Translation

        # Decompose A = U @ S @ Vh to get pure rotation and scale
        U, S_vec, Vh = torch.linalg.svd(A)
        R_mat = U @ Vh

        # Fix reflection if det(R) < 0
        if torch.linalg.det(R_mat) < 0:
            U[:, -1] *= -1
            S_vec[-1] *= -1
            R_mat = U @ Vh

        # Get uniform scale (use geometric mean of singular values)
        scale = torch.pow(S_vec.prod(), 1 / 3)
        scale_np = scale.detach().cpu().numpy()

        # Build C2W matrix with pure rotation and original translation
        # Don't divide translation - it should stay as-is from the transformation
        M_CW_new_world_pose = torch.eye(4, device=device, dtype=torch.float32)
        M_CW_new_world_pose[:3, :3] = R_mat
        M_CW_new_world_pose[:3, 3] = t  # Keep original translation

        # ---- Optional extrinsic overrides (simulator-frame position + rpy) ----
        # Replace the dataset-derived rotation and/or translation with a
        # hand-specified pose BEFORE recomputing T_wc. Intrinsics are
        # untouched — only the C2W transform gets edited. Both branches are
        # independent so a caller can pin position while keeping rotation
        # (or vice versa). See the docstring for the rpy convention.
        if override_rpy is not None:
            override_quat = p.getQuaternionFromEuler(list(override_rpy))
            R_over_np = np.array(
                p.getMatrixFromQuaternion(override_quat), dtype=np.float32,
            ).reshape(3, 3)
            R_mat = torch.from_numpy(R_over_np).to(device=device, dtype=torch.float32)
            M_CW_new_world_pose[:3, :3] = R_mat
        if override_xyz is not None:
            t_over_np = np.array(override_xyz, dtype=np.float32)
            t = torch.from_numpy(t_over_np).to(device=device, dtype=torch.float32)
            M_CW_new_world_pose[:3, 3] = t

        # dist_inc: shift the camera along its OWN view axis. Applied last
        # so the axis reflects any override_rpy / dataset rotation already
        # sitting in R_mat. Gaussian-splat cameras look down local +Z, so
        # the world-space forward is `R_mat[:, 2]`. Positive dist_inc pulls
        # the camera BACKWARD (opposite the view direction) — further from
        # whatever the camera is looking at. Negative pushes it forward.
        if override_dist_inc is not None:
            forward_world = R_mat[:, 2]
            t = t - float(override_dist_inc) * forward_world
            M_CW_new_world_pose[:3, 3] = t

        # 5. Calculate T_wc (World-to-Camera Translation)
        V_new_world = torch.linalg.inv(M_CW_new_world_pose)
        T_wc = V_new_world[:3, 3]

        print("cam to world transformation:")
        print(M_CW_new_world_pose)

        print('world to cam translation:')
        print(T_wc)

        # Convert to numpy for the Camera constructor
        R_cw_np = R_mat.detach().cpu().numpy()
        T_wc_np = T_wc.detach().cpu().numpy()

        # 7. Initialize the New Camera
        # (Assuming other parameters are loaded as before)
        resolution = (camera.alpha_mask.shape[2], camera.alpha_mask.shape[1])
        if self.image_width is None:
            if self.image_height is None:
                image_width = resolution[0]
                image_height = resolution[1]
            else:
                image_width = int(resolution[0] * self.image_width / resolution[1])
                image_height = self.image_height
        else:
            if self.image_height is None:
                image_width = self.image_width
                image_height = int(resolution[1] * self.image_width / resolution[0])
            else:
                image_width = self.image_width
                image_height = self.image_height

        resolution = (image_width, image_height)
        image = torch.zeros((3, resolution[1], resolution[0])).float()
        depth_params = None

        # Adjust FoV to compensate for scene scaling
        # FoV relates to distance via: tan(FoV/2) = viewport_size / (2 * distance)
        # When distance is scaled by 'scale', we need: tan(FoV_new/2) = tan(FoV_old/2) * scale
        # Therefore: FoV_new = 2 * atan(tan(FoV_old/2) * scale)
        # FoVx_adjusted = 2 * np.arctan(np.tan(camera.FoVx / 2) / scale_np)
        # FoVy_adjusted = 2 * np.arctan(np.tan(camera.FoVy / 2) / scale_np)

        new_camera = Camera(
            resolution,
            camera.colmap_id,
            R_cw_np,  # Pure rotation (orthonormal)
            T_wc_np,  # W2C translation (scale-free)
            camera.FoVx,  # FoV adjusted for scene scaling
            camera.FoVy,  # FoV adjusted for scene scaling
            depth_params,
            to_pil_image(image),
            camera.invdepthmap,
            camera.image_name,
            camera.uid,
            scale=camera.scale,  # Uniform scale extracted from transformation
        )

        splatsim_camera = SplatSimCamera(
            camera=new_camera,
            pipeline=pipeline,
            background=background,
            tracked_link_index=None,  # Set this outside of this loop
        )

        return splatsim_camera

    def _get_ee_link_index(self) -> int:
        """Return the link index used as the end-effector for trajectory planning."""
        return int(self.wrist_camera.tracked_link_index)

    def get_current_ee_pose(self):
        """Current EE pose as the URDF LINK FRAME (getLinkState indices 4/5),
        not the link COM (0/1). Task targets captured from this method are fed
        to the RRT planner, whose IK + FK-accuracy gate operate in the link
        frame (pybullet's calculateInverseKinematics convention) — mixing the
        two silently breaks on any robot whose EE link has a COM offset.
        Identical values for all current robots (wrist_camera_link COM ==
        frame)."""
        ee_link = self._get_ee_link_index()
        link_state = self.pybullet_client.getLinkState(
            self.splatsim_robot.sim_id, ee_link
        )
        return link_state[4], link_state[5]

    def get_current_object_pose(self, object_name=None, object_id=None):
        if object_name is not None:
            if object_name not in [
                splatsim_obj.config.name for splatsim_obj in self.splatsim_objects
            ]:
                raise ValueError(
                    f"Object name '{object_name}' not found when querying its pose."
                )
            queried_object_id = [
                splatsim_obj.config.name for splatsim_obj in self.splatsim_objects
            ].index(object_name)
            if object_id is not None:
                assert object_id == queried_object_id
            object_id = queried_object_id
        elif object_id is None:
            raise ValueError("No object_name or object_id given!")

        body_id = self.splatsim_objects[object_id].sim_id
        pos, quat = self.pybullet_client.getBasePositionAndOrientation(body_id)
        return pos, quat

    def get_task_description(self) -> str:
        return self.ENV_CONFIG.task_description

    def get_env_config(self) -> Dict[str, Any]:
        """Serialize ENV_CONFIG into a pickle-safe dict.

        Used by remote clients (e.g. lerobot's shared autonomy wrapper) to fetch
        obstacle geometry and task goal info for RRT planning. Numpy arrays and
        nested dataclasses are converted to plain Python via dataclasses.asdict.
        """
        from dataclasses import asdict

        def _to_jsonable(obj):
            # Convert numpy arrays / scalars to plain Python so the result pickles
            # cleanly across processes and runs.
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj

        cfg = self.ENV_CONFIG
        objects = []
        for o in cfg.objects:
            d = asdict(o)
            # Tag the concrete type so downstream consumers can dispatch without
            # re-importing the dataclasses (they may not be available client-side).
            d["__type__"] = type(o).__name__
            objects.append(_to_jsonable(d))

        task = _to_jsonable(asdict(cfg.task)) if cfg.task is not None else None

        # Robot config (URDF path + base position) for the planner to load
        # the same arm into its private pybullet client.
        robot_obj = self.splatsim_robot
        robot_cfg_dict = _to_jsonable(asdict(robot_obj.config))
        robot_cfg_dict["__type__"] = type(robot_obj.config).__name__

        # Splatsim scene metadata (mirrors _get_splatsim_episode_metadata).
        # Included here so callers (e.g. intervention_record.py) can fetch
        # everything they need in one ZMQ call rather than adding a new method.
        splatsim_meta = self._get_splatsim_episode_metadata()

        # Current (live) end-effector pose at the time of this call. Used by
        # lerobot's last-mile-debug wrapper to trigger EE-space override
        # without having to re-implement FK client-side. Cheap: get_env_config
        # is called once per env step and FK is just a pybullet link lookup.
        try:
            cur_ee_pos, cur_ee_quat = self.get_current_ee_pose()
            current_ee_pos = list(cur_ee_pos) if cur_ee_pos is not None else None
            current_ee_quat = list(cur_ee_quat) if cur_ee_quat is not None else None
        except Exception:
            current_ee_pos = None
            current_ee_quat = None

        env_config_dict = {
            "name": cfg.name,
            "task": task,
            "task_description": cfg.task_description,
            "terminate_on_collision": cfg.terminate_on_collision,
            "objects": objects,
            "robot": robot_cfg_dict,
            "splatsim_robot_config": splatsim_meta["splatsim_robot_config"],
            "splatsim_background_config": splatsim_meta["splatsim_background_config"],
            "splatsim_object_configs": splatsim_meta["splatsim_object_configs"],
            "current_ee_pos": current_ee_pos,
            "current_ee_quat": current_ee_quat,
            # Forward the URDF's self-collision skip pairs into the dispatched
            # oracle config so the LeRobot SA wrapper's RRT planner can pick
            # them up automatically (no need for the user to also pass
            # `--rrt_self_collision_skip_pairs` on the DAgger CLI for the
            # typical case — the env publishes its own URDF-known exclusions).
            # An explicit SA-config-level override still wins if the user
            # wants to add more pairs per-run.
            "self_collision_skip_pairs": list(self.SELF_COLLISION_SKIP_PAIRS),
            # Physics-stepping mode, so remote clients (viz / blend scripts)
            # can warn when they're talking to a wallclock-stepped sim — a
            # slow policy against an unsynced sim produces "jumpy" rollouts.
            "sync_physics_to_client": bool(self._sync_physics_to_client),
        }
        # Soft-cost payload (cost-aware RRT over pushable vegetation). Only
        # added when the env declares one so binary-obstacle envs publish a
        # byte-identical dict (planner-side config hashing stays stable).
        if getattr(cfg, "soft_cost", None):
            env_config_dict["soft_cost"] = cfg.soft_cost
        return env_config_dict

    def _produces_images(self) -> bool:
        """True if this env emits image observations from ANY source — the
        Gaussian splat (render_from_splat) OR the fast PyBullet camera. Used to
        decide whether the LeRobot dataset should declare image features."""
        return getattr(self, "_initial_render_mode", RenderMode.NONE) != RenderMode.NONE

    def _is_wrist_camera(self, camera_name: Optional[str]) -> bool:
        """True if this camera key should render the wrist-mounted view (and a
        wrist camera link is actually available to render it from)."""
        return (
            camera_name is not None
            and "wrist" in camera_name.lower()
            and self.wrist_camera is not None
            and self.wrist_camera.tracked_link_index is not None
        )

    def _effective_fovy_rad(self, splatsim_camera) -> float:
        """Vertical field-of-view (radians) for the PyBullet pinhole projection.

        For a FISHEYE camera, `camera.FoVy` is only the pinhole-fallback value
        (it's set to the base camera's FoVy in get_wrist_camera) — far too narrow
        for a wide fisheye lens, which is why the pybullet wrist looked zoomed
        in. Derive the vertical FoV from the fisheye intrinsic matrix instead:
        fovy = 2·atan(H / (2·fy)).

        Crucially fy is scaled by FISHEYE_RECTIFY_ZOOM, so this returns the FoV of
        the RECTIFIED (undistorted "wide") view — exactly what the splat backend
        produces via `_rectify_fisheye_image` with the same zoom. That makes the
        pybullet wrist a rectangular wide view matching the splat wrist, instead
        of the raw circular ultra-wide fisheye. For a pinhole camera, use
        `camera.FoVy` directly."""
        cam = splatsim_camera.camera
        K = getattr(splatsim_camera, "intrinsic_matrix", None)
        if getattr(splatsim_camera, "camera_model", "pinhole") == "fisheye" and K is not None:
            fy = float(K[1, 1]) * FISHEYE_RECTIFY_ZOOM
            H = float(cam.image_height)
            if fy > 0 and H > 0:
                return 2.0 * float(np.arctan(H / (2.0 * fy)))
        return float(cam.FoVy)

    def _splatsim_camera_to_pybullet_view(self, splatsim_camera) -> Optional[Tuple[list, list, int, int]]:
        """Convert a SplatSimCamera (the SOURCE OF TRUTH for camera extrinsics +
        intrinsics) into a PyBullet (view_matrix, proj_matrix, render_W, render_H).

        The gsplat ``Camera`` stores COLMAP-convention extrinsics: ``R`` is the
        camera→world rotation and ``T`` is the world→camera translation, with the
        camera looking down its +Z axis and +Y pointing DOWN. So:
            eye     = -R @ T                (camera position in world)
            forward =  R[:, 2]              (world dir the camera looks along)
            up      = -R[:, 1]              (COLMAP +Y is down -> world up = -Y)
        The projection is ALWAYS pinhole at the camera's effective vertical FoV
        (see `_effective_fovy_rad` — a fisheye derives it from its intrinsics
        scaled by FISHEYE_RECTIFY_ZOOM, i.e. the RECTIFIED wide view the splat
        backend also outputs, not the raw circular ultra-wide image). One code
        path serves pinhole and fisheye alike, so the pybullet wrist and the splat
        wrist can't diverge. Rendered at the camera's NATIVE ASPECT (so horizontal
        coverage matches too, not just vertical), height fixed to
        PYBULLET_CAMERA_HEIGHT for speed; `resize_image` then letterboxes/
        stretches to 224 exactly as the splat path does. (The fisheye's fx/fy
        differ by ~1%, so deriving horizontal FoV from the sensor aspect rather
        than fx is off by that much — far below the error already inherent in
        approximating a rectified fisheye with a pinhole.) Returns None if the
        camera has no gsplat Camera."""
        cam = getattr(splatsim_camera, "camera", None)
        if cam is None:
            return None
        R = np.asarray(cam.R, dtype=np.float64).reshape(3, 3)
        T = np.asarray(cam.T, dtype=np.float64).reshape(3)
        eye = -R @ T
        forward = R[:, 2]
        up = -R[:, 1]
        view = self.pybullet_client.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=(eye + forward).tolist(),
            cameraUpVector=up.tolist(),
        )
        H = int(self.PYBULLET_CAMERA_HEIGHT)
        sensor_aspect = float(cam.image_width) / float(cam.image_height)
        # ONE path for every camera (pinhole base AND fisheye wrist): render a
        # rectangular pinhole at the sensor aspect. `_effective_fovy_rad` is the
        # single place that knows how to turn a camera into a pinhole FoV — for a
        # fisheye it returns the RECTIFIED (FISHEYE_RECTIFY_ZOOM) FoV, which is
        # what the splat backend also outputs after `_rectify_fisheye_image`. So
        # the pybullet wrist is a rectangular wide view that matches the splat
        # wrist, rather than the raw circular ultra-wide fisheye.
        aspect = sensor_aspect
        fov_deg = float(np.degrees(self._effective_fovy_rad(splatsim_camera)))
        W = max(1, int(round(H * aspect)))
        proj = self.pybullet_client.computeProjectionMatrixFOV(
            fov=fov_deg,
            aspect=aspect,
            nearVal=self.PYBULLET_CAMERA_NEAR,
            farVal=self.PYBULLET_CAMERA_FAR,
        )
        return view, proj, W, H

    def _try_load_egl_renderer(self) -> Optional[int]:
        """Load PyBullet's EGL offscreen GPU renderer into this DIRECT client.

        Returns the plugin id on success (so `_render_pybullet_camera` uses the
        hardware GL renderer), else None (fall back to the CPU TINY renderer).
        Never raises — a missing EGL library just means the slow path.

        RENDER-MISMATCH NOTE: the headless EGL renderer and the non-headless GUI
        renderer produce VISUALLY DIFFERENT pybullet camera images — most visibly
        a specular sheen on the robot in EGL that the GUI context lacks. It's a
        pybullet EGL-vs-GUI OpenGL-lighting difference, NOT controllable via the
        getCameraImage lighting params or per-link specularColor (both are no-ops
        on the hardware renderer). Consequence: RECORD and EVAL in the SAME mode
        (both headless, or both GUI) — mixing them injects a visual covariate
        shift into the dataset. A one-time warning below flags this at load.
        """
        try:
            import importlib.util
            spec = importlib.util.find_spec("eglRenderer")
            fname = spec.origin if spec is not None else None
        except Exception:
            fname = None
        try:
            if fname:
                pid = self.pybullet_client.loadPlugin(fname, "_eglRendererPlugin")
            else:
                pid = self.pybullet_client.loadPlugin("eglRendererPlugin")
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[render] EGL plugin load failed ({e}); headless camera uses CPU TINY renderer.")
            return None
        if pid is not None and pid >= 0:
            print(f"[render] EGL offscreen renderer loaded (plugin {pid}) — headless GPU camera.")
            print(
                "[render] WARNING: headless (EGL) and non-headless (GUI) render the "
                "pybullet camera DIFFERENTLY (e.g. a specular sheen on the robot in "
                "headless that the GUI lacks). Record your dataset and run eval in "
                "the SAME mode — mixing headless/GUI adds a visual covariate shift."
            )
            return pid
        print("[render] EGL plugin unavailable; headless camera uses CPU TINY renderer.")
        return None

    def _fixed_pybullet_view_proj(self) -> Tuple[list, list, int, int]:
        """The fixed third-person (view, proj) from the PYBULLET_CAMERA_* attrs —
        orbit form (yaw/pitch/distance) if PYBULLET_CAMERA_YAW is set, else
        explicit eye/target/up."""
        W, H = self.PYBULLET_CAMERA_WIDTH, self.PYBULLET_CAMERA_HEIGHT
        if self.PYBULLET_CAMERA_YAW is not None:
            view = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=list(self.PYBULLET_CAMERA_TARGET),
                distance=self.PYBULLET_CAMERA_DISTANCE,
                yaw=self.PYBULLET_CAMERA_YAW,
                pitch=self.PYBULLET_CAMERA_PITCH,
                roll=0,
                upAxisIndex=2,
            )
        else:
            view = self.pybullet_client.computeViewMatrix(
                cameraEyePosition=list(self.PYBULLET_CAMERA_EYE),
                cameraTargetPosition=list(self.PYBULLET_CAMERA_TARGET),
                cameraUpVector=list(self.PYBULLET_CAMERA_UP),
            )
        proj = self.pybullet_client.computeProjectionMatrixFOV(
            fov=self.PYBULLET_CAMERA_FOV,
            aspect=float(W) / float(H),
            nearVal=self.PYBULLET_CAMERA_NEAR,
            farVal=self.PYBULLET_CAMERA_FAR,
        )
        return view, proj, W, H

    def render_video_fallback_frame(
        self, obs: Optional[Dict[str, np.ndarray]] = None, camera_name: str = "base_rgb"
    ) -> Optional[np.ndarray]:
        """One-off CHW float [0,1] frame for eval videos when no policy
        cameras are configured (camera_names=[], e.g. a state-only policy).

        Honors the env's render mode: SPLAT renders the Gaussian splat from
        the base camera (same imagery an image policy would see); PYBULLET —
        or SPLAT with assets unavailable (RENDER_SPLATS=False subclasses) —
        falls back to the fixed PyBullet debug camera. `obs` is accepted so
        callers that already hold a get_observations() dict don't pay for a
        second one; prep_image_rendering reads poses from PyBullet either way.
        """
        if self.do_render_from_splat and self.base_camera is not None:
            if obs is None:
                obs = self.get_observations(render_images=False)
            cached_link_states, self._tick_shadow_masks = (
                self._capture_obs_snapshot_and_masks()
            )
            self.prep_image_rendering(data=obs, cached_link_states=cached_link_states)
            with torch.no_grad():
                img = self.render_image(camera_name=camera_name, cached_link_states=cached_link_states)
            if img is not None:
                return img
        return self._render_pybullet_camera(camera_name)

    def _render_pybullet_camera(
        self, camera_name: Optional[str] = None, cached_link_states=None
    ) -> np.ndarray:
        """Thread-safe wrapper around the PyBullet camera render.

        `cached_link_states` (from `get_curr_link_states` at observation-
        capture time) pins the WRIST camera pose to that snapshot instead of
        the live link state at render time. Without it, the marshal round-trip
        below means the wrist pose is derived on the main thread up to a
        physics tick (or more) AFTER the rest of the observation was captured
        — a small but real wrist-view latency vs the state vector. The image
        content still shows live body poses (GL renders the current world);
        only the camera pose can be snapshot-pinned.

        Callers on the MAIN serve-loop thread run the render inline (fast
        path — matches previous behavior byte-for-byte).

        Callers on any OTHER thread (typically the ZMQ handler thread
        dispatching a get_observations request) marshal the render call to
        the main thread via the _sync_render_* handshake. Necessary because
        in headless (DIRECT) mode PyBullet's camera path uses the EGL plugin,
        and the EGL context is bound to whichever thread first called
        p.getCameraImage — subsequent calls from a different thread return a
        null GL vendor and segfault on NVIDIA drivers (observed as
        `ven = (null)` in sim logs immediately preceding SIGSEGV).

        The marshaling round-trip is a no-op when either (a) serve() hasn't
        run yet — `_main_thread` is None, no serve loop to marshal to; or
        (b) we ARE the main thread; or (c) there's no ZMQ server thread /
        it isn't alive. In those cases we just execute directly.
        """
        return self._sync_main_thread_call("rgb", camera_name, cached_link_states)

    def _capture_obs_snapshot_and_masks(self):
        """(cached_link_states, shadow_masks) captured as ONE ATOMIC block on
        the main serve-loop thread.

        Atomicity is the point: in async mode the MAIN thread steps physics
        at 240 Hz. If the link-state snapshot and the shadow-mask raycasts run
        on the ZMQ handler thread, physics advances between (and during) them
        — the rays then see a robot a few ticks AHEAD of the pose the splat
        renders from, and at speed the shadow visibly deforms away from the
        rendered arm. Marshaled to the main thread, the serve loop is busy
        executing this block and cannot step physics until it returns, so the
        snapshot, the rays, and therefore the rendered robot and its shadow
        all describe the same instant. (sync_physics_to_client mode gets this
        for free — physics only steps on command — and main-thread callers
        run inline, which is equally atomic.)"""
        return self._sync_main_thread_call("snapshot_masks")

    def _sync_main_thread_call(self, kind: str, camera_name=None, cached_link_states=None):
        """Run a sync-render-family request on the main serve-loop thread,
        marshaling via the _sync_render_* handshake when called from another
        thread while the serve loop is live. `kind` is "rgb" (PyBullet camera
        render) or "snapshot_masks" (link-state + shadow-mask capture)."""
        current = threading.current_thread()
        needs_marshal = (
            self._main_thread is not None
            and current is not self._main_thread
            and getattr(self, "_zmq_server_thread", None) is not None
            and self._zmq_server_thread.is_alive()
        )
        if not needs_marshal:
            return self._dispatch_sync_render(kind, camera_name, cached_link_states)

        # Post the request; single-slot design mirrors _sync_step_* — safe
        # because ZMQ REP serializes handler-thread requests one at a time,
        # so no two requests can race for the slot.
        with self._sync_render_lock:
            self._sync_render_request_kind = kind
            self._sync_render_request_camera = camera_name
            self._sync_render_request_link_states = cached_link_states
            self._sync_render_result = None
            self._sync_render_error = None
            self._sync_render_done_event.clear()
            self._sync_render_pending_event.set()

        # Timeout matches _sync_step_* (5 s). The main loop wakes every
        # 1/240 s and consumes the request immediately; a missed signal
        # (loop dead / GUI hang) surfaces as a warning rather than blocking
        # the ZMQ thread indefinitely.
        completed = self._sync_render_done_event.wait(timeout=5.0)
        if not completed:
            with self._sync_render_lock:
                self._sync_render_request_kind = "rgb"
                self._sync_render_request_camera = None
                self._sync_render_request_link_states = None
                self._sync_render_pending_event.clear()
            raise RuntimeError(
                "sync_render: main-thread render timeout (5 s). Serve loop may "
                "be blocked; skipping render and continuing."
            )

        with self._sync_render_lock:
            err = self._sync_render_error
            result = self._sync_render_result
            self._sync_render_result = None
            self._sync_render_error = None

        if err is not None:
            # Re-raise on the calling thread so the ZMQ handler surfaces the
            # error to the client rather than silently returning None.
            raise err
        return result

    def _dispatch_sync_render(self, kind: str, camera_name, cached_link_states):
        """Execute a sync-render-family request. Runs on whichever thread owns
        the GL/EGL context (the marshal target, or the caller when inline)."""
        if kind == "snapshot_masks":
            snap = get_curr_link_states(
                self.splatsim_robot.sim_id, use_link_centers=True
            )
            return snap, self._compute_tick_shadow_masks(snap)
        return self._render_pybullet_camera_direct(camera_name, cached_link_states)

    def _render_pybullet_camera_direct(
        self, camera_name: Optional[str] = None, cached_link_states=None
    ) -> np.ndarray:
        """Actual PyBullet camera render — MUST run on the thread that owns the
        EGL context (main serve-loop thread in headless mode). Callers should go
        through `_render_pybullet_camera`, which marshals cross-thread requests
        here via the sync-render handshake.

        Fast, splat-free image source usable by any env. For a wrist camera key
        (see `_is_wrist_camera`) it renders the WRIST-MOUNTED view derived from
        `get_wrist_camera(cached_link_states=...)` — pass the observation's
        link-state snapshot to pin the wrist pose to capture time (None reads
        live state); for a base/third-person key it mirrors `base_camera`
        when a splat base camera exists (matching the splat view's pose + FoV +
        aspect), else the fixed PYBULLET_CAMERA_* pose. Returns a CHW float32 RGB
        image in [0, 1] at the per-view render resolution (resize_image then maps
        it to 224). Uses the GPU OpenGL renderer under a GUI connection, the CPU
        tiny renderer when headless (DIRECT)."""
        view, proj, W, H = self._resolve_pybullet_view_proj(
            camera_name, cached_link_states=cached_link_states
        )
        _, _, rgba, _, _ = self.pybullet_client.getCameraImage(
            W, H, view, proj,
            renderer=self._pybullet_camera_renderer(),
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (H, W, 4))[:, :, :3]
        # HWC uint8 [0,255] -> CHW float32 [0,1] (the format resize_image expects).
        return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))

    def _resolve_pybullet_view_proj(
        self, camera_name: Optional[str], cached_link_states=None
    ) -> Tuple[list, list, int, int]:
        """(view, proj, W, H) for a PyBullet render of `camera_name` — the
        camera-resolution logic shared by the RGB render and the shadow mask.

        `cached_link_states` pins the WRIST camera pose to a link-state
        snapshot (same contract as `render_image`) so a shadow mask lines up
        with the splat frame rendered from that snapshot; None reads the live
        wrist pose (the pybullet RGB path's historical behavior)."""
        view_proj = None
        if self._is_wrist_camera(camera_name):
            # Wrist view from the wrist-camera pose + FoV (a fisheye lens
            # renders as its RECTIFIED pinhole equivalent, matching the splat).
            wrist_cam = self.get_wrist_camera(cached_link_states=cached_link_states)
            if wrist_cam is not None:
                view_proj = self._splatsim_camera_to_pybullet_view(wrist_cam)
        elif self.base_camera is not None:
            # Base/third-person: mirror the SPLAT base camera (pose + FoV) so the
            # pybullet base matches the splat base view instead of a synthetic
            # orbit with an unrelated FoV.
            view_proj = self._splatsim_camera_to_pybullet_view(self.base_camera)
        if view_proj is None:
            # No SplatSimCamera to mirror (e.g. the planar env has no base
            # camera): fall back to the fixed PYBULLET_CAMERA_* third-person pose.
            view_proj = self._fixed_pybullet_view_proj()
        return view_proj

    def _pybullet_camera_renderer(self) -> int:
        # Hardware GL whenever we have a GL context — a GUI connection, or a
        # headless DIRECT client with the EGL plugin loaded. Only fall back to the
        # CPU software renderer when headless AND EGL was unavailable.
        use_hardware_gl = (not self._headless) or (getattr(self, "_egl_plugin_id", None) is not None)
        return p.ER_BULLET_HARDWARE_OPENGL if use_hardware_gl else p.ER_TINY_RENDERER

    def _raycast_splat_shadow_mask(
        self, camera_name: Optional[str], cached_link_states=None
    ) -> np.ndarray:
        """Shadow mask (1 = lit, 0 = in the ROBOT's shadow) for splat shadow
        compositing (see SPLAT_SHADOW_* class attrs).

        PURE RAYCAST — no getCameraImage. PyBullet's renderers can't do this
        job (the EGL hardware plugin silently ignores getCameraImage's
        `shadow`/`lightDirection` args, and once loaded it intercepts even
        explicit ER_TINY_RENDERER requests), and a render would also drag in
        the EGL thread-affinity marshaling, delaying the mask several physics
        ticks past the observation snapshot. Ray tests have neither problem:
        callable from any thread, immediately. Two batches:

          1. Camera rays: near->far frustum ray per quarter-res pixel;
             the first collision hit gives each pixel's surface point + body
             (collision geometry — consistent with what the shadow caster
             uses, and with what actually determines crashes).
          2. Light rays: from each non-robot surface point (nudged 5 mm off
             the surface) to the light position; a pixel is shadowed iff the
             ROBOT body blocks its ray.

        SNAPSHOT SEMANTICS: pass the SAME `cached_link_states` given to
        `render_image` — the wrist camera pose then matches the splat frame
        exactly. The ray tests read live body poses, so call this
        back-to-back with the link-state capture (get_observations stashes
        masks via `_capture_obs_snapshot_and_masks` right at that point) — then the
        caster pose the rays see is the pose the snapshot recorded, and the
        shadow cannot lag the rendered robot.

        All bodies cast shadows by default (SPLAT_SHADOW_ROBOT_ONLY=True
        restores robot-only casting) — the reversed light rays make the
        occlusion test a plain first-hit-distance check, so scene-on-scene
        shadows cost nothing extra. Robot PIXELS are still excluded as
        RECEIVERS: the robot is one multibody of adjacent convex hulls, and
        shadows landing on it would mostly be acne from sibling links.

        Returns (h, w) float32 in [0, 1] at quarter camera resolution,
        lightly blurred for soft edges; the compositor resizes to the splat
        resolution (shadow masks are low-frequency)."""
        view, proj, W, H = self._resolve_pybullet_view_proj(
            camera_name, cached_link_states=cached_link_states
        )
        w, h = max(1, W // 4), max(1, H // 4)
        mask = np.ones((h, w), dtype=np.float32)
        robot = getattr(self, "splatsim_robot", None)
        robot_id = getattr(robot, "sim_id", None)
        if robot_id is None:
            return mask

        def _ray_batch(froms: np.ndarray, tos: np.ndarray) -> list:
            hits: list = []
            BATCH = 16000  # pybullet MAX_RAY_INTERSECTION_BATCH_SIZE is 16384
            for s in range(0, len(froms), BATCH):
                hits.extend(
                    self.pybullet_client.rayTestBatch(
                        froms[s : s + BATCH].tolist(),
                        tos[s : s + BATCH].tolist(),
                        numThreads=0,
                    )
                )
            return hits

        light = np.asarray(self.SPLAT_SHADOW_LIGHT_DIRECTION, dtype=np.float64)

        # ── STATIC-SCENE CACHE ────────────────────────────────────────────
        # Receivers, their surface points, and the static-geometry shadow
        # state depend only on (camera pose, non-robot scene state, light).
        # Both are hashed into signatures checked on EVERY mask computation —
        # any moved body, moved articulated part (e.g. the engine's
        # prismatic cap), created/removed object, or light change misses the
        # cache and triggers a full rebuild; nothing is hardcoded about
        # which bodies count as static. On a hit, only the ROBOT's occlusion
        # needs fresh rays — and only for static-lit receivers whose light
        # segment passes through the robot's (inflated) AABB. The base
        # camera is fixed, so it hits every tick; the wrist camera moves
        # with the arm, so its view signature misses and it rebuilds (same
        # cost as the uncached path).
        # The wrist camera rides the arm — its view signature changes every
        # frame, so it can never hit the cache; skip the signature work AND
        # the cache-building extras in the rebuild below (the continuation
        # rays that resolve static occluders hidden behind the robot are
        # only needed to keep a stored cache correct for later frames).
        cacheable = not self._is_wrist_camera(camera_name)
        scene_sig = view_sig = None
        cache = None
        if cacheable:
            scene_sig = self._static_scene_signature()
            view_sig = (w, h) + tuple(
                np.round(np.asarray(view, dtype=np.float64), 6).tolist()
            )
            cache = self._shadow_static_cache.get(camera_name)
        if (
            cache is not None
            and cache["scene_sig"] == scene_sig
            and cache["view_sig"] == view_sig
        ):
            recv_idx = cache["recv_idx"]
            surface = cache["surface"]
            shadowed = cache["static_shadowed"].copy()
            test = np.nonzero(~shadowed)[0]
            if len(test):
                lo, hi = self._robot_shadow_aabb()
                test = test[self._segments_intersect_aabb(light, surface[test], lo, hi)]
            if len(test):
                sub = surface[test]
                dirs = sub - light[None, :]
                dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
                hits = _ray_batch(np.broadcast_to(light, sub.shape), sub + 0.002 * dirs)
                if self.SPLAT_SHADOW_ROBOT_ONLY:
                    robot_hit = np.fromiter(
                        (hh[0] == robot_id for hh in hits), dtype=bool, count=len(hits)
                    )
                else:
                    # Scene is unchanged, so for a static-lit receiver the
                    # only possible NEW first hit before the surface is the
                    # robot; same 8 mm slack as the rebuild path.
                    hit_pts = np.array(
                        [hh[3] if hh[0] >= 0 else (0.0, 0.0, 0.0) for hh in hits],
                        dtype=np.float64,
                    )
                    gaps = np.linalg.norm(hit_pts - sub, axis=1)
                    robot_hit = np.fromiter(
                        (hh[0] == robot_id for hh in hits), dtype=bool, count=len(hits)
                    ) & (gaps > 0.008)
                shadowed[test] = robot_hit
            mask.ravel()[recv_idx[shadowed]] = 0.0
            return self._protect_robot_silhouette(
                cv2.GaussianBlur(mask, (3, 3), 0), camera_name, view, proj
            )

        # ── FULL REBUILD (cache miss) ─────────────────────────────────────
        # Frustum rays: unproject every pixel at the near and far planes.
        # pybullet view/proj are COLUMN-major.
        V = np.asarray(view, dtype=np.float64).reshape(4, 4).T
        P = np.asarray(proj, dtype=np.float64).reshape(4, 4).T
        inv_vp = np.linalg.inv(P @ V)
        xs_ndc = 2.0 * (np.arange(w) + 0.5) / w - 1.0
        ys_ndc = 1.0 - 2.0 * (np.arange(h) + 0.5) / h  # pixel row 0 = NDC y +1
        gx, gy = np.meshgrid(xs_ndc, ys_ndc)  # (h, w)
        flat_x, flat_y = gx.ravel(), gy.ravel()

        def _unproject(z_ndc: float) -> np.ndarray:
            ndc = np.stack(
                [flat_x, flat_y, np.full_like(flat_x, z_ndc), np.ones_like(flat_x)],
                axis=1,
            )
            pts = ndc @ inv_vp.T
            return pts[:, :3] / pts[:, 3:4]

        cam_tos = _unproject(1.0)  # far plane
        # Rays start at the camera EYE (from the view matrix: eye = -R^T t),
        # NOT at the near plane (_unproject(-1.0)). Near-plane starts have a
        # 5 cm blind zone: with a box pressed right up against the wrist
        # camera, the rays began INSIDE the box hull (which PyBullet rays
        # never register), passed through, and landed on the TABLE behind it
        # — so the mask composited the table's shadow pattern onto the
        # close-up box pixels. From the eye, the first hit is the box face
        # the splat is actually showing.
        eye = -V[:3, :3].T @ V[:3, 3]
        cam_froms = np.broadcast_to(eye, cam_tos.shape)

        # Camera pass with ROBOT PASS-THROUGH: a ray whose first hit is the
        # robot is continued past that hit (up to 3 layers) so the surface
        # BEHIND the robot still receives shadow. Without this, shadows get
        # visible holes: collision hulls are fatter than the splat visuals,
        # so with a curled arm a band of pixels around/behind the arm hits
        # robot collision geometry even though the SPLAT shows the table
        # there — those pixels were excluded from receiving, cutting the
        # arm's shadow into disconnected blobs. Pixels that are still robot
        # after 3 layers stay lit (genuinely deep inside the robot's
        # silhouette, where the splat shows the robot itself anyway).
        n_rays = len(cam_froms)
        ray_dirs = cam_tos - cam_froms
        ray_dirs /= np.maximum(np.linalg.norm(ray_dirs, axis=1, keepdims=True), 1e-9)
        surf_ids = np.full(n_rays, -1, dtype=np.int64)
        surf_pts = np.zeros((n_rays, 3), dtype=np.float64)
        first_entry = np.zeros((n_rays, 3), dtype=np.float64)
        has_entry = np.zeros(n_rays, dtype=bool)
        froms = cam_froms
        active = np.arange(n_rays)
        for _layer in range(3):
            hits = _ray_batch(froms, cam_tos[active])
            ids = np.fromiter((h[0] for h in hits), dtype=np.int64, count=len(hits))
            surf_ids[active] = ids
            hit_sel = ids >= 0
            hit_rows = active[hit_sel]
            surf_pts[hit_rows] = np.array([hits[i][3] for i in np.nonzero(hit_sel)[0]],
                                          dtype=np.float64)
            through = ids == robot_id
            thru_rows = active[through]
            newly = thru_rows[~has_entry[thru_rows]]
            first_entry[newly] = surf_pts[newly]  # first robot-hull entry point
            has_entry[newly] = True
            active = thru_rows
            if len(active) == 0:
                break
            # Re-cast from just past the robot surface toward the far plane.
            froms = surf_pts[active] + 1e-3 * ray_dirs[active]

        # Shadow receivers: pixels whose (pass-through) camera ray reached a
        # non-robot body.
        recv = (surf_ids >= 0) & (surf_ids != robot_id)

        # ROBOT-THICKNESS filter on pass-through pixels: pass-through exists
        # for the thin silhouette band where collision hulls are fatter than
        # the splat visuals — there the ray only GRAZES a hull (a chord of a
        # couple cm). A pixel whose ray traverses a THICK robot stack (curled
        # arm: upper link over lower link) is deep inside the silhouette,
        # where the splat shows the ROBOT — shadowing the surface behind it
        # painted dark blotches on the rendered arm, and because the layer
        # cutoff sat right at curled-arm depths the blotches FLICKERED as the
        # pose crept. Thickness = first hull entry -> last hull back-face
        # (one reverse ray from the receiving surface toward the entry; rays
        # never register the body they start in, so the back-face is the
        # first reverse hit). Thicker than SPLAT_SHADOW_MAX_ROBOT_THICKNESS
        # => reclassify as robot pixel (stably lit).
        # NOT applied to the wrist (uncacheable) camera: its near field IS
        # robot — the gripper right in front of the lens measures ~4-6 cm of
        # hull, riding exactly on the threshold, and filtering there made
        # thousands of pixels flip-flop (measured 4 -> 5544 flickery px).
        # The wrist view showed no curled-arm flicker without the filter.
        thru_recv = recv & has_entry if cacheable else np.zeros_like(recv)
        if thru_recv.any():
            rows = np.nonzero(thru_recv)[0]
            rev_hits = _ray_batch(
                surf_pts[rows] - 0.001 * ray_dirs[rows], first_entry[rows]
            )
            max_t = float(self.SPLAT_SHADOW_MAX_ROBOT_THICKNESS)
            for i, hh in enumerate(rev_hits):
                if hh[0] == robot_id:
                    t = np.linalg.norm(np.asarray(hh[3]) - first_entry[rows[i]])
                    if t > max_t:
                        recv[rows[i]] = False

        if not recv.any():
            return mask
        recv_idx = np.nonzero(recv)[0]
        surface = surf_pts[recv_idx]

        # Cast REVERSED — from the light down to each surface point — never
        # surface-to-light. A surface-to-light ray needs a nudge off the
        # receiver (or it re-hits its own body), and any nudge creates a
        # dead zone: with the gripper hovering within the nudge distance of
        # the table (plus collision-hull fat), the ray STARTS INSIDE the
        # gripper's hull, which PyBullet rays never register — so exactly at
        # near-contact the shadow vanished in a gripper-shaped hole. From
        # the light (open air) the ray cleanly hits whatever is first:
        # robot => shadowed; the receiver's own surface or any other body
        # => lit. Targets extend 2 mm past the surface so a robot hull
        # already penetrating the receiver (touching contact) still
        # registers as the first hit.
        to_surf = surface - light[None, :]
        dirs = to_surf / np.maximum(np.linalg.norm(to_surf, axis=1, keepdims=True), 1e-9)
        light_hits = _ray_batch(
            np.broadcast_to(light, surface.shape), surface + 0.002 * dirs
        )
        n_recv = len(surface)
        hit_ids_l = np.fromiter(
            (hit[0] for hit in light_hits), dtype=np.int64, count=n_recv
        )
        static_shadowed = np.zeros(n_recv, dtype=bool)
        if self.SPLAT_SHADOW_ROBOT_ONLY:
            robot_shadowed = hit_ids_l == robot_id
        else:
            # EVERY body casts: shadowed iff the light's first hit lands
            # meaningfully BEFORE the receiver point — no matter whose
            # surface blocks. For convex hulls this also yields correct
            # self-shadowing (a point visible from the light IS the first
            # intersection of its own hull; a far-side point is occluded by
            # its own near side). The 8 mm slack absorbs the hit landing a
            # hair early on the receiver's own face at grazing light angles.
            hit_pts_l = np.array(
                [hit[3] if hit[0] >= 0 else (0.0, 0.0, 0.0) for hit in light_hits],
                dtype=np.float64,
            )
            gap = np.linalg.norm(hit_pts_l - surface, axis=1)
            occluded = (hit_ids_l >= 0) & (gap > 0.008)
            robot_first = hit_ids_l == robot_id
            robot_shadowed = occluded & robot_first
            static_shadowed = occluded & ~robot_first
            # Resolve static occlusion currently HIDDEN BEHIND the robot, so
            # the cached static state stays correct once the robot moves
            # away: continue robot-first rays past the robot (2 layers).
            # Cache-only work — skipped for uncacheable (wrist) views.
            pend = np.nonzero(robot_first)[0] if cacheable else np.empty(0, dtype=np.int64)
            froms2 = hit_pts_l[pend] + 1e-3 * dirs[pend]
            for _layer in range(2):
                if len(pend) == 0:
                    break
                hits2 = _ray_batch(froms2, surface[pend] + 0.002 * dirs[pend])
                ids2 = np.fromiter(
                    (hh[0] for hh in hits2), dtype=np.int64, count=len(hits2)
                )
                pts2 = np.array(
                    [hh[3] if hh[0] >= 0 else (0.0, 0.0, 0.0) for hh in hits2],
                    dtype=np.float64,
                )
                gap2 = np.linalg.norm(pts2 - surface[pend], axis=1)
                static_shadowed[pend[(ids2 >= 0) & (ids2 != robot_id) & (gap2 > 0.008)]] = True
                again = ids2 == robot_id
                froms2 = pts2[again] + 1e-3 * dirs[pend[again]]
                pend = pend[again]

        if cacheable:
            self._shadow_static_cache[camera_name] = {
                "scene_sig": scene_sig,
                "view_sig": view_sig,
                "recv_idx": recv_idx,
                "surface": surface,
                "static_shadowed": static_shadowed,
            }

        mask.ravel()[recv_idx[static_shadowed | robot_shadowed]] = 0.0
        # Soften edges; the compositor's strength cap sets final darkness.
        return self._protect_robot_silhouette(
            cv2.GaussianBlur(mask, (3, 3), 0), camera_name, view, proj
        )

    def _protect_robot_silhouette(
        self, mask: np.ndarray, camera_name: Optional[str], view, proj
    ) -> np.ndarray:
        """Force the mask LIT wherever the robot currently occludes the camera,
        and stash that silhouette for the compositor to re-apply at full
        resolution.

        Why this exists: the mask is computed at a fraction of the splat
        render's resolution (measured 56x99 mask vs 359x640 render = 6.4x),
        so the mask's own blur plus the compositor's bilinear upscale smear
        every shadow edge across ~6-13 splat pixels. Shadows legitimately
        computed for surfaces BEHIND the arm therefore bleed forward onto the
        arm, and as the caster moves that bleed switches on and off — read as
        "a shadow flickering on the shoulder link, cast from a metre away".
        The cached path makes it structural rather than occasional: its
        receiver set is deliberately robot-independent (camera rays pass
        THROUGH the arm so the cache stays valid as the arm moves), so nothing
        downstream knew where the arm was at all.

        Fixing it needs the arm's CURRENT screen coverage, which can't be
        cached — hence a small dedicated ray pass, restricted to the robot's
        projected AABB so it stays cheap enough to run every frame on the
        cached path too.

        Note this only enforces a decision already made elsewhere: robot
        pixels are excluded as shadow RECEIVERS (see the receiver mask in
        `_raycast_splat_shadow_mask`), because the robot is a chain of
        adjacent convex hulls whose sibling links would shadow-acne each
        other. Without this, that exclusion held at mask resolution but was
        undone by the upscale."""
        sil = self._robot_screen_silhouette(camera_name, view, proj, mask.shape)
        if sil is None:
            return mask
        self._tick_shadow_robot_silhouettes[camera_name] = sil
        mask = mask.copy()
        mask[sil] = 1.0  # crisp at mask res — applied AFTER the blur
        return mask

    def _robot_screen_silhouette(
        self, camera_name: Optional[str], view, proj, shape
    ) -> Optional[np.ndarray]:
        """Boolean (h, w) of mask pixels whose camera ray hits the robot first.

        Casts only within the robot's projected world-AABB bounding box (the
        arm covers a modest slice of frame), so this stays affordable on the
        per-frame cached path."""
        h, w = shape
        robot = getattr(self, "splatsim_robot", None)
        robot_id = getattr(robot, "sim_id", None)
        if robot_id is None:
            return None
        lo, hi = self._robot_shadow_aabb()

        V = np.asarray(view, dtype=np.float64).reshape(4, 4).T
        P = np.asarray(proj, dtype=np.float64).reshape(4, 4).T
        VP = P @ V
        # Project the 8 AABB corners to NDC -> pixel box (clipped to frame).
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        clip = corners @ VP.T
        wclip = clip[:, 3]
        if np.all(wclip <= 1e-9):
            return None
        # Corners behind the camera can't be projected meaningfully; when any
        # is, fall back to the whole frame rather than a bogus (tiny) box.
        if np.any(wclip <= 1e-9):
            x0, x1, y0, y1 = 0, w, 0, h
        else:
            ndc = clip[:, :3] / wclip[:, None]
            px = (ndc[:, 0] * 0.5 + 0.5) * w
            py = (0.5 - ndc[:, 1] * 0.5) * h
            x0 = max(0, int(np.floor(px.min())) - 1)
            x1 = min(w, int(np.ceil(px.max())) + 1)
            y0 = max(0, int(np.floor(py.min())) - 1)
            y1 = min(h, int(np.ceil(py.max())) + 1)
        sil = np.zeros((h, w), dtype=bool)
        if x1 <= x0 or y1 <= y0:
            return sil

        inv_vp = np.linalg.inv(VP)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xs = xs.ravel(); ys = ys.ravel()
        ndc_x = 2.0 * (xs + 0.5) / w - 1.0
        ndc_y = 1.0 - 2.0 * (ys + 0.5) / h
        far = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=1)
        far = far @ inv_vp.T
        far = far[:, :3] / far[:, 3:4]
        eye = -V[:3, :3].T @ V[:3, 3]
        froms = np.broadcast_to(eye, far.shape)
        hit_ids = []
        BATCH = 16000
        for s0 in range(0, len(far), BATCH):
            hits = self.pybullet_client.rayTestBatch(
                froms[s0 : s0 + BATCH].tolist(), far[s0 : s0 + BATCH].tolist(),
                numThreads=0,
            )
            hit_ids.extend(hh[0] for hh in hits)
        sil[ys, xs] = np.fromiter(
            (i == robot_id for i in hit_ids), dtype=bool, count=len(hit_ids)
        )
        return sil

    def _resample_splat_shadow_strength(self) -> None:
        """Resample this episode's effective shadow strength:
        SPLAT_SHADOW_STRENGTH * U(1 - JITTER, 1 + JITTER), clipped to [0, 1].
        Called on every env.reset() (via _reset_episode_state) AFTER reset
        seeds np.random, so seeded resets reproduce the same strength."""
        base_strength = float(self.SPLAT_SHADOW_STRENGTH)
        jitter = float(self.SPLAT_SHADOW_STRENGTH_JITTER)
        if jitter > 0:
            base_strength *= float(np.random.uniform(1.0 - jitter, 1.0 + jitter))
        self._splat_shadow_strength_current = float(np.clip(base_strength, 0.0, 1.0))

    def _static_scene_signature(self) -> tuple:
        """Pose + joint-state signature of every NON-robot body (plus the
        light position), rounded to 0.1 mm / 1e-4 rad. Compared on every
        shadow-mask computation: any moved body, moved articulated part
        (e.g. the engine's prismatic cap), created/removed object, or light
        change invalidates the static shadow cache. Nothing is hardcoded
        about which bodies count as static — a body IS static exactly while
        its signature entry doesn't change."""
        pc = self.pybullet_client
        robot_id = getattr(getattr(self, "splatsim_robot", None), "sim_id", None)
        sig = [tuple(round(float(v), 6) for v in self.SPLAT_SHADOW_LIGHT_DIRECTION)]
        for bi in range(pc.getNumBodies()):
            uid = pc.getBodyUniqueId(bi)
            if uid == robot_id:
                continue
            # 3 decimals (1 mm / ~0.06 deg): coarse enough that RESTING-contact
            # micro-jitter (sub-mm solver noise on objects sitting on the
            # table) can't flip the signature — a per-frame flip means a full
            # rebuild every tick, whose layer decisions visibly flicker.
            # Real object motion exceeds 1 mm immediately.
            pos, orn = pc.getBasePositionAndOrientation(uid)
            entry = (uid,) + tuple(round(v, 3) for v in pos) + tuple(
                round(v, 3) for v in orn
            )
            nj = pc.getNumJoints(uid)
            if nj:
                entry += tuple(
                    round(js[0], 3) for js in pc.getJointStates(uid, list(range(nj)))
                )
            sig.append(entry)
        return tuple(sig)

    def _robot_shadow_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Union AABB over every robot link, inflated 3 cm — a cheap bound
        for 'could the robot possibly block this light segment'."""
        pc = self.pybullet_client
        rid = self.splatsim_robot.sim_id
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for link in range(-1, pc.getNumJoints(rid)):
            a, b = pc.getAABB(rid, link)
            lo = np.minimum(lo, a)
            hi = np.maximum(hi, b)
        return lo - 0.03, hi + 0.03

    @staticmethod
    def _segments_intersect_aabb(
        start: np.ndarray, ends: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        """Vectorized slab test: for each i, does the segment start->ends[i]
        intersect the AABB [lo, hi]? `start` is a single (3,) point (the
        light); `ends` is (N, 3). Returns an (N,) bool array."""
        d = ends - start[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo[None, :] - start[None, :]) / d
            t2 = (hi[None, :] - start[None, :]) / d
        # Axis-parallel segments (d==0): intersect that slab iff start lies
        # inside it — encode as an always/never-true t-interval.
        d0 = d == 0
        inside = (start >= lo) & (start <= hi)  # (3,)
        t1 = np.where(d0, np.where(inside[None, :], -np.inf, np.inf), t1)
        t2 = np.where(d0, np.where(inside[None, :], np.inf, -np.inf), t2)
        tmin = np.minimum(t1, t2).max(axis=1)
        tmax = np.maximum(t1, t2).min(axis=1)
        return (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0)

    def _consume_sync_render_request(self) -> None:
        """Main-thread consumer for the sync-to-main-thread camera-render
        handshake. Mirrors `_consume_sync_step_request` — non-blocking check;
        if no request pending, returns immediately. When a request IS pending,
        runs the render on THIS thread (which owns the EGL context in headless
        mode), stores result / exception, signals done.

        Called from the main serve loop next to `_consume_sync_step_request`,
        so a ZMQ handler-thread `get_observations(render_images=True)` sees at
        most a 1/240 s round-trip latency (loop's sleep granularity) per camera.
        """
        if not self._sync_render_pending_event.is_set():
            return
        with self._sync_render_lock:
            kind = self._sync_render_request_kind
            camera_name = self._sync_render_request_camera
            cached_link_states = self._sync_render_request_link_states
            self._sync_render_request_kind = "rgb"
            self._sync_render_request_camera = None
            self._sync_render_request_link_states = None
            self._sync_render_pending_event.clear()
        try:
            result = self._dispatch_sync_render(kind, camera_name, cached_link_states)
            err = None
        except BaseException as e:
            result = None
            err = e
        with self._sync_render_lock:
            self._sync_render_result = result
            self._sync_render_error = err
        self._sync_render_done_event.set()

    def oracle_environment_state(self, observations: Dict[str, Any]) -> list:
        """Privileged world state (object poses) for oracle/state-based policies,
        recorded in EVERY dataset as a SEPARATE observation.environment_state
        feature (FeatureType.ENV) — NOT appended to observation.state, because
        policies like the diffusion policy treat environment_state as a distinct
        conditioning input with its own normalization. Image-based training simply
        ignores it, so one recording serves both policy families.

        For each object in `_oracle_object_names()` it appends the selected world
        position coords (ORACLE_STATE_COORD_INDICES) and, if
        ORACLE_STATE_INCLUDE_QUAT, the (x, y, z, w) orientation. `observations`
        already holds each object's ``<name>_position`` / ``<name>_orientation``
        (populated above), so this reads them directly. Consumed by BOTH
        `build_lerobot_frame` (recording) and `_raw_obs_to_gym_obs` (eval), so the
        recorded and eval env-state vectors are identical by construction. Its
        length always equals `env_state_dim()`."""
        coords: list = []
        for name in self._oracle_object_names():
            pos = observations.get(name + "_position")
            if pos is None:
                # Object not present this step — keep the layout fixed with zeros
                # so len == env_state_dim() (a missing object is a scene bug, not
                # a variable-width state).
                coords.extend([0.0] * self._oracle_per_object_dim())
                continue
            coords.extend(float(pos[i]) for i in self.ORACLE_STATE_COORD_INDICES)
            if self.ORACLE_STATE_INCLUDE_QUAT:
                quat = observations.get(name + "_orientation") or (0.0, 0.0, 0.0, 1.0)
                coords.extend(float(q) for q in quat)
        # EE position LAST so the per-object layout above stays a stable prefix
        # (old recordings without it are the first len-N slice of new ones).
        if self.ORACLE_RECORD_ENV_STATE and self.ORACLE_STATE_INCLUDE_EE_POS:
            ee_pos, _ = self.get_current_ee_pose()
            coords.extend(float(ee_pos[i]) for i in self._oracle_ee_coord_indices())
        # Per-link min-obstacle distances AFTER EE, so the (objects + EE)
        # prefix stays intact for older checkpoints/datasets. Sentinel value
        # for envs with no obstacles keeps layout width fixed.
        if self.ORACLE_RECORD_ENV_STATE and self.ORACLE_STATE_INCLUDE_LINK_OBSTACLE_DIST:
            coords.extend(self._compute_link_obstacle_min_distances())
        return coords

    def get_observations(self, render_images: bool = True) -> Dict[str, np.ndarray]:
        joint_positions = self.get_joint_state()
        joint_velocities = np.array(
            [
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i)[1]
                for i in range(self.num_dofs() + 1)
            ]
        )

        dummy_ee_pos, dummy_ee_quat = self.get_current_ee_pose()
        # get the euler angles from the quaternion
        dummy_ee_euler = self.pybullet_client.getEulerFromQuaternion(dummy_ee_quat)

        # print the euler angles and the reconstructed quaternion
        if self.use_gripper:
            self.current_gripper_state = self.get_current_gripper_state() / 0.8
            # Snap the gripper state to 0 or 1 if they're reasonably close.
            # The wider thresholds (0.2 / 0.8 instead of 0.05 / 0.95) account
            # for residual rest-position drift after physics settling — e.g.
            # in some scene configurations the open gripper rests at ~0.05
            # rather than exactly 0, which a tighter threshold misses.
            if self.current_gripper_state > 0.8:
                self.current_gripper_state = 1.0
            elif self.current_gripper_state < 0.2:
                self.current_gripper_state = 0.0
        else:
            self.current_gripper_state = 0.0

        # combine the position and euler angles and self.current_gripper_state to get the state
        state = np.concatenate(
            [dummy_ee_pos, dummy_ee_euler, [self.current_gripper_state]]
        )
        action = np.concatenate(
            [dummy_ee_pos, dummy_ee_euler, [self.current_gripper_action]]
        )

        # Target object position and orientation

        observations = {
            "joint_positions": joint_positions[:self.num_dofs() + 1],
            "all_joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": dummy_ee_quat,
            "state": state,
            "action": action,
        }

        # gripper_position is for gello integration. It's a shame that it intersects with self.splat_name_list convetion
        observations["gripper_position"] = [self.current_gripper_state]

        for i in range(len(self.splatsim_objects)):
            if self.splatsim_objects[i] == self.splatsim_background:
                continue
            (
                object_pos,
                object_quat,
            ) = self.pybullet_client.getBasePositionAndOrientation(
                self.splatsim_objects[i].sim_id
            )
            observations[self.splatsim_objects[i].config.name + "_position"] = object_pos
            observations[self.splatsim_objects[i].config.name + "_orientation"] = object_quat
            # Keep current pose fields in sync with PyBullet state
            self.splatsim_objects[i].config.current_position = list(object_pos)
            self.splatsim_objects[i].config.current_quat = list(object_quat)

        # Privileged world state (e.g. object coords for a state-only policy),
        # exposed as a SEPARATE observation.environment_state feature — NOT packed
        # into observation.state. Empty by default; subclasses override
        # oracle_environment_state(). Both the recorded dataset (build_lerobot_frame)
        # and the eval gym obs (_raw_obs_to_gym_obs) read this SAME key, so
        # record-time and eval-time env-state vectors stay identical.
        observations["environment_state"] = self.oracle_environment_state(observations)

        if render_images and self.do_render_from_pybullet and len(self.camera_names) > 0:
            # Fast PyBullet-native camera path (no splat). Rendered PER camera:
            # a wrist key ("wrist_rgb") gets the wrist-mounted view (from the
            # wrist link pose via SplatSimCamera); others get the fixed
            # third-person view. One link-state snapshot is captured up front
            # and threaded through every render (incl. the cross-thread
            # marshal), pinning the wrist camera pose to THIS observation's
            # capture instant — matching the splat path's cached_link_states
            # contract instead of reading a live pose a tick or more later.
            cached_link_states = get_curr_link_states(
                self.splatsim_robot.sim_id, use_link_centers=True
            )
            for camera_name in self.camera_names:
                raw_img = self._render_pybullet_camera(camera_name, cached_link_states)
                for mode in self.image_resize_modes:
                    key = f"{camera_name}_{mode.value}"
                    observations[key] = resize_image(raw_img, (224, 224), mode=mode)
            self.display_observations(observations)
        elif (
            render_images and self.do_render_from_splat
            and self.base_camera is not None  # splat render needs a base camera
            and len(self.camera_names) > 0
        ):
            # Capture the link-state snapshot AND the shadow masks in one
            # atomic main-thread block (see _capture_obs_snapshot_and_masks):
            # the shadow rays read live body poses, so any physics stepping
            # between snapshot and rays would deform the shadow away from the
            # rendered arm at speed.
            cached_link_states, self._tick_shadow_masks = (
                self._capture_obs_snapshot_and_masks()
            )

            self.prep_image_rendering(data=observations, cached_link_states=cached_link_states)
            with torch.no_grad():
                for camera_name in self.camera_names:
                    # render_image returns raw numpy array (CHW float32)
                    raw_img = self.render_image(
                        camera_name=camera_name,
                        cached_link_states=cached_link_states
                    )
                    # Store one resized copy per active resize mode under {cam}_{mode} key
                    for mode in self.image_resize_modes:
                        key = f"{camera_name}_{mode.value}"
                        if raw_img is not None:
                            observations[key] = resize_image(raw_img, (224, 224), mode=mode)
                        else:
                            observations[key] = None

            # Display the rendered observations
            self.display_observations(observations)

        for camera_name in self.camera_names:
            # For example, when self.do_render_from_splat is False
            for mode in self.image_resize_modes:
                key = f"{camera_name}_{mode.value}"
                if key not in observations:
                    observations[key] = None

        # Monotonic env-mutation clock stamped at capture time. Consumers
        # holding the version returned by a mutation (e.g. the SA wrapper's
        # teleport_joint_state) compare: obs with state_version < that value
        # was captured BEFORE the mutation and is stale with respect to it.
        # See _bump_state_version for the bump sites.
        observations["state_version"] = int(self._state_version)

        return observations

    def display_observations(self, observations: Dict[str, Any]) -> None:
        """Display rendered RGB observations in the SplatSim GUI.

        Args:
            observations: Dictionary containing rendered images as torch tensors (C, H, W)
        """
        if self._splatsim_gui is None:
            return

        frames_to_display = {}
        display_mode = self.image_resize_modes[0] if self.image_resize_modes else None
        for camera_name in self.camera_names:
            key = f"{camera_name}_{display_mode.value}" if display_mode else camera_name
            if key not in observations or observations[key] is None:
                continue
            frame = observations[key]

            # Convert from tensor if needed
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            # Convert from CxHxW to HxWxC
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                frame = np.transpose(frame, (1, 2, 0))

            # Convert from [0, 1] float to [0, 255] uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            frames_to_display[camera_name] = frame

        if frames_to_display:
            self._splatsim_gui.update_camera_images(frames_to_display)

    def randomize_object_pose(self, splatsim_obj: SplatSimObject):
        if splatsim_obj.sim_id is None:
            return

        cfg = splatsim_obj.config
        bp = cfg.base_position
        bq = cfg.base_quat

        x_range = self.TABLE_LIMITS[0] if cfg.position_range_x is None else cfg.position_range_x
        y_range = self.TABLE_LIMITS[1] if cfg.position_range_y is None else cfg.position_range_y
        z_range = self.TABLE_LIMITS[2] if cfg.position_range_z is None else cfg.position_range_z

        curr_pos, curr_quat = self.pybullet_client.getBasePositionAndOrientation(splatsim_obj.sim_id)
        curr_euler_z = self.pybullet_client.getEulerFromQuaternion(curr_quat)[2]
        base_z = self.pybullet_client.getEulerFromQuaternion(bq)[2]
        # Subtract base_z to get rotation relative to base orientation, then normalize to [0, 2pi]
        # to match rotation_range_z convention (rotation_range_z is defined relative to bq)
        _rot_eps = 1e-3  # tolerance for floating point drift in rotation check
        curr_euler_z_rel = (curr_euler_z - base_z) % (2 * np.pi)
        # 2pi and 0 are the same angle; if we're within eps of 2pi, treat as 0
        if curr_euler_z_rel > 2 * np.pi - _rot_eps:
            curr_euler_z_rel = 0.0
        in_range = np.all(np.array(curr_pos) >= np.array([x_range[0] + bp[0], y_range[0] + bp[1], z_range[0] + bp[2]])) \
            and np.all(np.array(curr_pos) <= np.array([x_range[1] + bp[0], y_range[1] + bp[1], z_range[1] + bp[2]])) \
            and cfg.rotation_range_z[0] - _rot_eps <= curr_euler_z_rel <= cfg.rotation_range_z[1] + _rot_eps
        if not (splatsim_obj.config.randomize_pose or (not splatsim_obj.config.randomize_pose and not in_range)):
            return
        
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        z = random.uniform(z_range[0], z_range[1])
        pos = [x + bp[0], y + bp[1], z + bp[2]]
        euler_z = random.uniform(cfg.rotation_range_z[0], cfg.rotation_range_z[1])

        quat = self.pybullet_client.getQuaternionFromEuler(
            [0, 0, euler_z]
        )
        quat = np.quaternion(quat[3], quat[0], quat[1], quat[2]) * np.quaternion(*bq)
        quat = [quat.w, quat.x, quat.y, quat.z]
        
        self.pybullet_client.resetBasePositionAndOrientation(splatsim_obj.sim_id, pos, quat)
        splatsim_obj.config.initial_position = list(pos)
        splatsim_obj.config.initial_quat = list(quat)

    def randomize_object_scale(self, splatsim_obj: SplatSimObject):
        cfg = splatsim_obj.config

        in_range = np.all(
            np.array(cfg.current_scale) >= np.array([cfg.scaling_range_x[0], cfg.scaling_range_y[0], cfg.scaling_range_z[0]])) and np.all(
            np.array(cfg.current_scale) <= np.array([cfg.scaling_range_x[1], cfg.scaling_range_y[1], cfg.scaling_range_z[1]])
        )
        if not (splatsim_obj.config.randomize_scale or (not splatsim_obj.config.randomize_scale and not in_range)):
            return  # No change in scale, skip

        new_sx = random.uniform(*cfg.scaling_range_x)
        new_sy = random.uniform(*cfg.scaling_range_y)
        new_sz = random.uniform(*cfg.scaling_range_z)
        new_scale = np.array([new_sx, new_sy, new_sz])

        if np.allclose(new_scale, np.array(cfg.current_scale)):
            return  # No change in scale, skip

        old_scale = np.array(cfg.current_scale)
        ratio = new_scale / old_scale

        device = splatsim_obj.gaussians._xyz.device
        dtype = splatsim_obj.gaussians._xyz.dtype
        ratio_t = torch.tensor(ratio, device=device, dtype=dtype)
        splatsim_obj.gaussians._xyz = splatsim_obj.gaussians._xyz * ratio_t
        splatsim_obj.gaussians._scaling = splatsim_obj.gaussians._scaling + torch.tensor(
            np.log(ratio), device=splatsim_obj.gaussians._scaling.device, dtype=splatsim_obj.gaussians._scaling.dtype
        )

        splatsim_obj.config.current_scale = new_scale.tolist()
        splatsim_obj.config.initial_scale = new_scale.tolist()

        # Reload URDF at new scale (PyBullet only supports globalScaling at load time)
        # Save and restore current pose so scale doesn't reset position/orientation
        if splatsim_obj.sim_id is not None:
            pos, quat = self.pybullet_client.getBasePositionAndOrientation(splatsim_obj.sim_id)
            old_sim_id = splatsim_obj.sim_id
            physics_scale = float(np.cbrt(np.prod(new_scale)))  # geometric mean for uniform physics approximation
            self.load_urdf(splatsim_obj, physics_scale=physics_scale)
            self.pybullet_client.resetBasePositionAndOrientation(splatsim_obj.sim_id, pos, quat)
            self.pybullet_client.removeBody(old_sim_id)

    def _objects_collide(self, obj_i: SplatSimObject, obj_j: SplatSimObject) -> bool:
        """Check collision between two objects.

        Both use_aabb_collision=True  → fast AABB overlap test (no PyBullet narrowphase).
        Otherwise                     → PyBullet pairwise_collision.

        PyBullet's getAABB returns world-space (aabbMin, aabbMax) for a
        specific link, accounting for position, orientation, and scale.

        Two-object box URDFs in this project (thinkpad_box, starwars_box, ...)
        wrap their collision geometry in a `<link name="base_link">` that sits
        BELOW a `<link name="world"/>` root via a fixed joint at
        `xyz="0 0 0.15875"`. In PyBullet's numbering that puts the world link
        at linkIndex=-1 (with NO collision shape → zero-volume AABB at the
        origin) and the actual box collision at linkIndex=0. The default
        `getAABB(sim_id)` queries linkIndex=-1 and returns the empty world
        AABB — so two boxes anywhere in the scene have "AABBs" that are both
        zero-volume points at (0, 0, 0), and this overlap check always
        returned False regardless of where the visible boxes actually were.
        Fix: union the per-link AABBs across every link of each body.
        """
        if obj_i.config.use_aabb_collision and obj_j.config.use_aabb_collision:
            min_i, max_i = self._body_world_aabb(obj_i.sim_id)
            min_j, max_j = self._body_world_aabb(obj_j.sim_id)
            # Overlap on all three axes ↔ collision.
            return all(max_i[k] > min_j[k] and max_j[k] > min_i[k] for k in range(3))
        return pairwise_collision(obj_i.sim_id, obj_j.sim_id)

    def _body_world_aabb(self, body_id: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Union of all-link world-space AABBs for a body.

        `getAABB(body_id)` alone queries linkIndex=-1 (the base/root link),
        which is empty for URDFs that keep their collision block on a child
        link — e.g. the box URDFs whose root is `<link name="world"/>` and
        whose collision `<box>` lives on a downstream `base_link`. Iterating
        `range(-1, getNumJoints(body_id))` covers every link (base + all
        joint-attached children) and returns the outer bounding box that
        actually contains the visible geometry.
        """
        num_joints = self.pybullet_client.getNumJoints(body_id)
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        # linkIndex range: -1 (base) through num_joints-1 (each joint's child link).
        for link_idx in range(-1, num_joints):
            lo, hi = self.pybullet_client.getAABB(body_id, linkIndex=link_idx)
            # Empty/absent collision shapes surface as a zero-volume AABB at
            # the link's origin — they can't shrink the union (they're a
            # point on the boundary of any real geometry we've already seen)
            # BUT they'd wrongly EXPAND the union if the link's origin sits
            # outside the real geometry (e.g. the root world link at (0,0,0)
            # while the collision box is at z=0.16). Skip zero-volume AABBs
            # so the union reflects real collision extent only.
            if lo == hi:
                continue
            for k in range(3):
                if lo[k] < mins[k]:
                    mins[k] = lo[k]
                if hi[k] > maxs[k]:
                    maxs[k] = hi[k]
        # If every link was zero-volume (no collision anywhere), fall back to
        # base AABB so downstream code doesn't dereference inf.
        if any(m == float("inf") for m in mins):
            return self.pybullet_client.getAABB(body_id)
        return (tuple(mins), tuple(maxs))

    def _resolve_goal_ee_target(self) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[list]]]:
        """Hook: the goal the scenario must be solvable to, as
        ``(ee_pos, ee_quat, q_goal_bias | None)``, or ``None`` to fall back to
        the legacy goal-CONFIG-exists check (`check_able_to_solve`).

        Base default returns None (envs without a goal-directed reachability
        requirement — e.g. object-on-plate — keep the legacy behavior).
        Goal-directed envs override:
          * small_engine → the fixed `ENV_CONFIG.task.target_ee_{pos,quat}`.
          * planar       → position-IK to the (moving) target block, FK'd to a
                           reachable pose + that config as the bias.
        See `_check_scenario_solvable`."""
        return None

    def _check_scenario_solvable(self, q_start) -> bool:
        """True iff the current scene is completable. For goal-directed envs
        (`_resolve_goal_ee_target` returns a target) this attempts a FULL RRT
        path from `q_start` to the goal and caches it for reuse; for others it
        falls back to `check_able_to_solve` (goal config exists)."""
        goal = self._resolve_goal_ee_target()
        if goal is None:
            return self.trajectory_generator.check_able_to_solve(q_start=q_start)

        target_ee_pos, target_ee_quat, q_goal_bias = goal
        path = self.trajectory_generator.try_plan_to_goal(
            q_start, target_ee_pos, target_ee_quat, q_goal_bias=q_goal_bias
        )
        if path is None:
            self._cached_reset_trajectory = None
            self._cached_reset_goal_q = None
            return False
        # Stash the validated path + goal so the first recording can skip
        # re-planning (see generate_trajectory_batch's cache consumption). Both
        # the server copy (for oracle info) and the generator's consume-once
        # slot are set.
        self._cached_reset_trajectory = path
        self._cached_reset_goal_q = getattr(
            self.trajectory_generator._planner, "_last_chosen_q_goal", None
        )
        self.trajectory_generator._cached_base_traj = path
        return True

    def randomize_objects(self, max_attempts: int = 100):
        collision = True
        able_to_solve = False
        attempt = 0
        while collision or not able_to_solve:
            attempt += 1
            if attempt > max_attempts:
                # Bounded so a scene that's never solvable (e.g. obstacles that
                # always wall off the target) can't spin forever. Accept the last
                # arrangement; check_metrics/eval will still report it honestly.
                print(
                    f"[randomize_objects] Warning: no solvable scenario after "
                    f"{max_attempts} attempts; accepting the last arrangement."
                )
                break
            collision = False
            able_to_solve = True

            for splatsim_obj in random.sample(self.splatsim_objects, len(self.splatsim_objects)):
                # if splatsim_obj == self.splatsim_robot or splatsim_obj == self.splatsim_background:
                #     continue
                self.randomize_object_scale(splatsim_obj)
                self.randomize_object_pose(splatsim_obj)

            # TODO the better solution is to make this randomize all articulated joints of any splatsim object (this just does the robot rn)
            # Returns None if no collision-free arm pose exists for this object
            # arrangement (and does NOT teleport in that case). Re-randomize the
            # whole scene rather than committing to a colliding robot pose.
            if self.randomize_ee_pose() is None:
                collision = True  # force another loop iteration
                continue

            # Gather candidate object pairs (unique, skipping table and un-fixable pairs)
            candidates = []
            objs = self.splatsim_objects
            for i in range(len(objs)):
                obj_i = objs[i]
                if obj_i.sim_id is None or obj_i.config.name == "table":
                    continue
                for j in range(i + 1, len(objs)):
                    obj_j = objs[j]
                    if obj_j.sim_id is None or obj_j.config.name == "table":
                        continue
                    if not obj_i.config.randomize_pose and not obj_j.config.randomize_pose:
                        continue  # Fixed pair — collision cannot be resolved by re-randomizing
                    both_aabb = obj_i.config.use_aabb_collision and obj_j.config.use_aabb_collision
                    candidates.append((0 if both_aabb else 1, obj_i, obj_j))  # AABB pairs (0) sort first

            # Check for collisions — AABB pairs first for maximum early-exit speed
            for _, obj_i, obj_j in sorted(candidates, key=lambda t: t[0]):
                if self._objects_collide(obj_i, obj_j):
                    collision = True
                    break

            # Colliding scenes can't be fixed by planning — skip the expensive
            # RRT solvability check and re-randomize.
            if collision:
                continue

            # Verify the scene is actually completable (goal + path for
            # goal-directed envs; goal-config-exists otherwise).
            able_to_solve = self._check_scenario_solvable(self.get_joint_state())
    
    SCENARIO_FIELDS = ("splatsim_robot_config", "splatsim_object_configs",
                       "splatsim_background_config")

    def apply_scenario(self, scenario: dict) -> None:
        """Put the scene into a recorded ORACLE SCENARIO.

        ``scenario`` uses the same three fields the episode metadata and
        ``get_env_config()`` already publish — robot config (whose
        ``articulation_config.initial_joint_positions`` is teleported to) and
        per-object ``initial_position`` / ``initial_quat`` / ``initial_scale``.
        One format for recorded episodes, ZMQ oracle info, and cached launch
        scenarios, so they cannot drift apart.
        """
        def _parse_ep_field(val):
            """Parquet stores these as JSON strings; parse back if needed."""
            if isinstance(val, str):
                return json.loads(val)
            return val

        robot_cfg = _parse_ep_field(scenario.get("splatsim_robot_config"))
        object_configs = _parse_ep_field(
            scenario.get("splatsim_object_configs")) or []
        if robot_cfg is not None:
            initial_joints = robot_cfg["articulation_config"]["initial_joint_positions"]
            self.teleport_joint_state(self.splatsim_robot, initial_joints)
            # self.command_joint_state(self.splatsim_robot, np.concatenate([initial_joints[:self.num_dofs()], [0]]))

        # Build a name→object lookup for the live scene objects
        obj_by_name = {obj.config.name: obj for obj in self.splatsim_objects}

        # Restore each non-robot, non-background object
        if robot_cfg is None and len(object_configs) == 0:
            raise ValueError("scenario has neither a robot config nor object configs")
        for obj_cfg_dict in object_configs:
            name = obj_cfg_dict["name"]
            splatsim_obj = obj_by_name.get(name)
            if splatsim_obj is None:
                print(f"[restore_episode_scenario] Object '{name}' not found in live scene, skipping.")
                continue
            if splatsim_obj.sim_id is None:
                continue

            pos = obj_cfg_dict["initial_position"]
            quat = obj_cfg_dict["initial_quat"]
            scale = obj_cfg_dict["initial_scale"]

            # Restore scale first (reloads URDF, so must happen before pose set)
            target_scale = np.array(scale)
            if not np.allclose(target_scale, np.array(splatsim_obj.config.current_scale)):
                old_scale = np.array(splatsim_obj.config.current_scale)
                ratio = target_scale / old_scale
                device = splatsim_obj.gaussians._xyz.device
                dtype = splatsim_obj.gaussians._xyz.dtype
                splatsim_obj.gaussians._xyz = splatsim_obj.gaussians._xyz * torch.tensor(ratio, device=device, dtype=dtype)
                splatsim_obj.gaussians._scaling = splatsim_obj.gaussians._scaling + torch.tensor(
                    np.log(ratio), device=splatsim_obj.gaussians._scaling.device, dtype=splatsim_obj.gaussians._scaling.dtype
                )
                splatsim_obj.config.current_scale = target_scale.tolist()
                splatsim_obj.config.initial_scale = target_scale.tolist()
                physics_scale = float(np.cbrt(np.prod(target_scale)))
                saved_pos, saved_quat = self.pybullet_client.getBasePositionAndOrientation(splatsim_obj.sim_id)
                old_sim_id = splatsim_obj.sim_id
                self.load_urdf(splatsim_obj, physics_scale=physics_scale)
                self.pybullet_client.resetBasePositionAndOrientation(splatsim_obj.sim_id, saved_pos, saved_quat)
                self.pybullet_client.removeBody(old_sim_id)

            # Restore pose
            self.pybullet_client.resetBasePositionAndOrientation(splatsim_obj.sim_id, pos, quat)
            splatsim_obj.config.current_position = list(pos)
            splatsim_obj.config.current_quat = list(quat)
            splatsim_obj.config.initial_position = list(pos)
            splatsim_obj.config.initial_quat = list(quat)

        # print("restored state. now doing step simulation")
        # # TODO does this help
        # for _ in range(100000):
        #     self.pybullet_client.stepSimulation()
        # print("step simulation done")
        


    def scenario_dict(self) -> dict:
        """Current scene as an oracle scenario (the format apply_scenario eats)."""
        meta = self._get_splatsim_episode_metadata()
        return {k: meta[k] for k in self.SCENARIO_FIELDS if k in meta}

    def save_scenario_file(self, path) -> None:
        """Write the current scene to a JSON scenario file.

        Point `launch_nodes.py --scenario_file` at it to start the simulator
        directly in this arrangement, skipping the randomize/solvability
        search — which costs up to `randomize_objects`' 100 attempts, each
        running goal IK, and dominates launch time on hard scenes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.scenario_dict(), indent=1))
        print(f"[scenario] saved -> {path}")

    def load_scenario_file(self, path, pin: bool = True) -> dict:
        """Apply a saved scenario file. ``pin`` also makes every later
        reset() reuse it instead of re-randomising."""
        scenario = json.loads(Path(path).read_text())
        self.apply_scenario(scenario)
        if pin:
            self._pinned_scenario = scenario
        print(f"[scenario] loaded <- {path}"
              + (" (pinned: resets reuse it)" if pin else ""))
        return scenario

    def restore_episode_scenario(self, episode_index: int) -> None:
        """Restore the environment to the state recorded at the start of a
        LeRobot episode. Thin wrapper over `apply_scenario` — see it for the
        format."""
        if self._lerobot_saver is None:
            raise RuntimeError("No LeRobot dataset loaded. Call _init_lerobot_dataset() first.")
        ep = self._lerobot_saver.meta.episodes[episode_index]
        self.apply_scenario({k: ep.get(k) for k in self.SCENARIO_FIELDS})

    def randomize_ee_pose(self, max_attempts=100) -> Optional[Tuple[float, ...]]:
        # generating random initial joint state using random end effector position and orientation
        initial_joint_positions: Optional[Tuple[float, ...]] = None
        found_collision_free = False
        for attempt in range(max_attempts):
            random_ee_pos, random_ee_quat = self.get_random_ee_pose()

            # joint angles using inverse kinematics.
            # maxNumIterations: PyBullet's DLS solver converges (or plateaus at
            # a local minimum) well under 200 iterations — profiled: identical
            # FK error at 200 vs 100000 iters, but 100000 costs ~14 ms on
            # reachable poses and ~380 ms on unreachable ones (the residual
            # threshold never triggers, so all iterations burn). With random
            # EE samples frequently unreachable, a 100k cap made this loop —
            # and env reset — take seconds. 512 caps the cost at ~2 ms/attempt
            # with no accuracy loss; the collision check downstream rejects
            # bad solutions either way.
            initial_joint_positions = self.pybullet_client.calculateInverseKinematics(
                self.splatsim_robot.sim_id,
                # IK end-effector link = the last arm link, which is the child of
                # joint `num_dofs` (UR5: link 6 = wrist_3; a 3-DOF arm: link 3).
                # Was hardcoded 6; num_dofs() is identical for the UR5 and
                # generalizes to other DOF counts. NOTE: the rest of this method
                # is still tied to a horizontal-table Cartesian workspace
                # (get_random_ee_pose needs TABLE_LIMITS) and length-num_dofs
                # null-space limits, so table-less arms (e.g. the planar env)
                # override randomize_ee_pose with joint-space sampling instead.
                self.num_dofs(),
                random_ee_pos,
                random_ee_quat,
                maxNumIterations=512,
                residualThreshold=1e-10,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
            )

            # Wrap joint angles to [-pi, pi]
            initial_joint_positions = tuple(
                ((angle + np.pi) % (2 * np.pi)) - np.pi
                for angle in initial_joint_positions
            )

            # reset the joint positions to the initial joint positions
            for i in range(0, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id, i + 1, initial_joint_positions[i]
                )

            # Validate at the PLANNER's clearance contract (traj-gen config:
            # e.g. 2 cm obstacle / 0.5 cm self), not is_robot_in_collision's
            # looser defaults (1 cm / 0). A reset pose that is clean at 1 cm
            # but dirty at 2 cm makes RRTToGoalPlanner.plan() fire its
            # start-escape, which silently shifts the recorded trajectory's
            # start away from this pose (metadata/frame-0 mismatch in saved
            # datasets, visible as a "jump" between episode restore and
            # replay). Matching contracts here means escape only fires for
            # genuine mid-run wedges, not at episode start.
            _tg_cfg = getattr(getattr(self, "trajectory_generator", None), "config", None)
            if not self.is_robot_in_collision(
                obstacle_clearance=(
                    _tg_cfg.obstacle_clearance if _tg_cfg is not None else _COLLISION_CLEARANCE
                ),
                self_collision_clearance=(
                    _tg_cfg.self_collision_clearance if _tg_cfg is not None else 0.0
                ),
            ):
                found_collision_free = True
                break

        if not found_collision_free or initial_joint_positions is None:
            # No collision-free arm pose for this object arrangement. Return None
            # (WITHOUT teleporting) so the caller re-randomizes the whole scene
            # rather than committing the robot to a colliding pose. Teleporting to
            # a colliding pose here is what caused the post-reset stutter (arm
            # shoved by contact) and the stale (often closed) gripper — the robot
            # is only ever teleported on the success path below, which snaps +
            # holds the arm and opens the gripper.
            print(f"Warning: Could not find collision-free EE pose after {max_attempts} attempts; "
                  f"returning None to trigger re-randomization.")
            return None

        # Collision-free pose found → teleport to it. teleport_joint_state does
        # resetJointState (snap) + POSITION_CONTROL hold on the arm +
        # move_gripper, so the robot reaches the pose and stays held.
        #
        # calculateInverseKinematics returns values for ALL movable joints (6 arm
        # + the gripper mimic joints); index num_dofs of that raw output is the
        # finger joint's *current* IK value, NOT a gripper command. Feeding it
        # straight to teleport would make command_joint_state -> move_gripper read
        # a stale finger angle and stick. So build the canonical action instead:
        # 6 arm joints + an explicit gripper-open command (0 => open).
        # TODO randomize the gripper state here if desired.
        initial_joint_positions = tuple(initial_joint_positions[:self.num_dofs()]) + (0.0,)
        self.teleport_joint_state(self.splatsim_robot, initial_joint_positions)
        if self.splatsim_robot.config.articulation_config is not None:
            self.splatsim_robot.config.articulation_config.initial_joint_positions = list(initial_joint_positions)
        return initial_joint_positions

    def _consume_sync_step_request(self) -> None:
        """Main-thread consumer for the sync-to-client physics-step handshake.

        Runs one main-loop iteration's worth of a pending step-request posted
        by the ZMQ handler thread from `command_joint_state`. Non-blocking:
        if no request is pending, returns immediately (letting the caller
        proceed to the standard time.sleep). When a request IS pending, runs
        the ALL of the requested substeps in one go — the ZMQ thread is
        blocked on `_sync_step_done_event.wait`, so latency-per-step matters
        (spreading the substeps across successive main-loop iters would
        multiply the ZMQ round-trip by 1/240 s per step).

        Called instead of `stepSimulation()` in the sync-eligible serve
        modes' main-loop branches; see `command_joint_state`'s docstring
        for why this signaling handshake exists (GUI-mode pybullet thread
        affinity).
        """
        if not self._sync_step_pending_event.is_set():
            return
        with self._sync_step_lock:
            n = int(self._sync_step_request_ticks)
            self._sync_step_request_ticks = 0
            self._sync_step_pending_event.clear()
        for _ in range(n):
            self.pybullet_client.stepSimulation()
        self._sync_step_done_event.set()

    def _recompute_skip_pairs(self):
        self._skip_pairs = {
            (link_idx, obj.sim_id)
            for obj in self.splatsim_objects
            if obj.sim_id is not None
            for link_idx in (obj.config.skip_collision_robot_links or [])
        }

    def is_robot_in_collision(
        self,
        obstacle_clearance=_COLLISION_CLEARANCE,
        verbose=False,
        return_kind=False,
        self_collision_clearance=0.0,
        self_collision_skip_pairs=None,
        self_collision_check_adjacent_pairs=None,
    ):
        """Check whether the robot is in collision with any obstacle or itself.

        Args:
            obstacle_clearance: Distance threshold for obstacle checks. 0.0 = actual penetration only.
            verbose: If True, print the offending pair on hit.
            return_kind: If False (default), returns bool — keeps backward compat
                with the existing teleport-loop caller (`if not self.is_robot_in_collision()`).
                If True, returns `(in_collision: bool, kind: str | None)` where
                `kind` is "obstacle" / "self" / None. The eval-time `check_metrics`
                path uses this to surface the cause in per-episode metrics →
                eval_info.json so post-hoc analysis can break down failures.
            self_collision_skip_pairs: OVERRIDE the self-collision skip list.
                ``None`` (default) uses the STRICT list
                (`self.SELF_COLLISION_SKIP_PAIRS`) — matches the RRT planner's
                contract, so the reset-time "find collision-free start"
                loop and the trajectory-gen collision predicates see the
                same "in collision" outcome as the planner. Eval-terminate
                (`check_metrics`) passes `self._eval_terminate_skip_pairs()`
                (union of STRICT + `SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA`)
                so URDF-mesh-overlap artifacts that produce solver kicks
                — pairs the planner must reject, but that the ENV shouldn't
                terminate on — get silently accepted at the terminate check.

        Returns:
            bool, or (bool, str | None) when return_kind=True.
        """
        joint_indices = list(range(1, self.num_dofs() + 1))
        obstacles = [
            obj for obj in self.splatsim_objects
            if obj.sim_id is not None and obj != self.splatsim_robot
        ]
        obstacle_ids = [obj.sim_id for obj in obstacles]
        # Pass human-readable names so verbose output says
        # `Collision: robot wrist_link(7) vs obstacle small_engine_new(id=1)`
        # rather than `vs obstacle 1`. Skipped when verbose=False.
        # SplatSimObject stores its display name at `obj.config.name`
        # (not `obj.name`) — the old `getattr(obj, "name", ...)` always
        # missed and fell back to the sim_id string.
        obstacle_names = {
            obj.sim_id: getattr(getattr(obj, "config", None), "name", None) or str(obj.sim_id)
            for obj in obstacles
        }
        # Resolve skip pairs: caller-provided override wins; else fall back
        # to the strict list (RRT contract). `or None` keeps
        # `check_links_in_collision`'s skip-set builder path on the
        # empty-list case.
        _skip_pairs_for_check = (
            self_collision_skip_pairs
            if self_collision_skip_pairs is not None
            else (self.SELF_COLLISION_SKIP_PAIRS or None)
        )
        # Fall back to the class-declared adjacent-check whitelist when the
        # caller doesn't override. Empty → parent-child pairs get skipped as
        # before (legacy behavior for small_engine et al). Robots that declared
        # CHECK_ADJACENT_LINK_PAIRS_NAMES (e.g., planar_3joint) get their
        # link_1↔link_2 / link_2↔link_3 fold-over check.
        _check_adjacent_for_call = (
            self_collision_check_adjacent_pairs
            if self_collision_check_adjacent_pairs is not None
            else (self.SELF_COLLISION_CHECK_ADJACENT_PAIRS or None)
        )
        return rrt_path_utils.check_links_in_collision(
            self.splatsim_robot.sim_id, joint_indices, q=None, obstacle_ids=obstacle_ids, skip_pairs=self._skip_pairs,
            obstacle_clearance=obstacle_clearance,
            self_collision_clearance=self_collision_clearance,
            self_collision_skip_pairs=_skip_pairs_for_check,
            self_collision_check_adjacent_pairs=_check_adjacent_for_call,
            verbose=verbose, obstacle_names=obstacle_names,
            return_kind=return_kind,
        )

    def get_random_ee_pose(self):
        if self.TABLE_LIMITS is None:
            raise NotImplementedError(
                "TABLE_LIMITS must be set in the environment to randomize end effector pose."
            )
        # random end effector position
        # if random.uniform(0, 1) > 0.2:
        random_ee_pos = np.array(
            [
                random.uniform(
                    self.TABLE_LIMITS[0][0], self.TABLE_LIMITS[0][1] + 0.1
                ),
                random.uniform(
                    self.TABLE_LIMITS[1][0] - 0.1, self.TABLE_LIMITS[1][1] + 0.1
                ),
                random.uniform(0.25, 0.65),
            ]
        )
        # TODO move this logic to the object_on_plate environment
        # else:
        #     # get object position
        #     (
        #         object_pos,
        #         object_quat,
        #     ) = self.pybullet_client.getBasePositionAndOrientation(
        #         self.splatsim_objects[0].sim_id
        #     )
        #     random_x = random.uniform(-0.105, 0.105)
        #     random_y = random.uniform(-0.105, 0.105)
        #     random_z = random.uniform(0.25, 0.3)
        #     random_ee_pos = np.array(
        #         [
        #             object_pos[0] + random_x,
        #             object_pos[1] + random_y,
        #             object_pos[2] + random_z,
        #         ]
        #     )
        # random_ee_pos = np.array([random.uniform(0.2, 0.5), random.uniform(-0.6, 0.6), random.uniform(0.2, 0.65)])

        # get the euler angles from the quaternion
        # get quaternion from euler angles
        # TODO move this logic to the object_on_plate environment (end effector pointing down)
        # random_ee_quat = self.initial_ee_quat

        # random_ee_quat is any an arbitrary orientation
        random_ee_quat = self.pybullet_client.getQuaternionFromEuler(
            [
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
            ]
        )

        return random_ee_pos, random_ee_quat

    def follow_paths_and_record(self, all_paths: List[PathSegment]):
        for path_segment in all_paths:
            if isinstance(path_segment, TrajectoryPathSegment):
                self.follow_trajectory_and_record(
                    path=path_segment.path,
                    gripper_pos=path_segment.gripper_pos,
                    gripper_velocity=path_segment.gripper_velocity,
                    threshold=path_segment.threshold,
                    use_current_iters=True,  # Seems to always be default true
                )
            elif isinstance(path_segment, GripperPathSegment):
                if path_segment.target_state == GripperState.OPEN:
                    self.open_gripper_and_record(num_steps=path_segment.num_steps)
                elif path_segment.target_state == GripperState.CLOSE:
                    self.close_gripper_and_record(num_steps=path_segment.num_steps)
                else:
                    raise ValueError(
                        f"Unknown GripperPathSegment.target_state value {path_segment.target_state}"
                    )
            else:
                raise ValueError(f"Unknown path segment type {type(path_segment)}")

    def eval_trajectory_success(self):
        # check the mse of xy position of the objects with the drop location
        for i in range(len(self.splatsim_objects) - 1):
            object_pos, _ = self.pybullet_client.getBasePositionAndOrientation(
                self.splatsim_objects[i].sim_id
            )
            mse = (object_pos[0] - self.drop_ee_pos[0]) ** 2 + (
                object_pos[1] - self.drop_ee_pos[1]
            ) ** 2

            if mse > 0.03:
                print("object not placed correctly")
                return False
        return True

    def open_gripper(self):
        """Open the gripper."""
        self.move_gripper(0.084)
        self.current_gripper_action = GripperState.OPEN  # 1

    def close_gripper(self):
        """Close the gripper."""
        self.move_gripper(0.0)
        self.current_gripper_action = GripperState.CLOSE  # 0

    def plan_execute_record_trajectory(self, initial_joint_positions, joint_signs):
        # Returns whether it was a success

        self.trajectory_length = 0

        # make path+trajectory_count folder
        trajectory_folder = os.path.join(self.path, str(self.trajectory_count).zfill(3))
        print("Generating trajectory in folder:", trajectory_folder)
        os.makedirs(trajectory_folder, exist_ok=True)

        all_paths = self.plan_given_this_state(initial_joint_positions)

        for i in range(100):
            self.pybullet_client.stepSimulation()
            self.open_gripper()
            for k in range(0, self.num_dofs()):
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id,
                    k + 1,
                    initial_joint_positions[k] * joint_signs[k],
                )

        if len(all_paths) == 0:
            self.delete_trajectory_folder()
            return False

        self.follow_paths_and_record(all_paths)

        # evaluate the success of the trajectory
        correct_trajectory = self.eval_trajectory_success()

        if correct_trajectory:
            return True
        else:
            self.delete_trajectory_folder()
            return False

    # =========================================================================
    # Mode Transition Hooks
    # =========================================================================

    def _enter_mode(self, mode: 'PybulletRobotServerBase.SERVE_MODES'):
        """Called when entering a new serve mode. Override in subclasses for custom behavior."""
        if mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            # Invalidate any cached planner so the next plan() call rebuilds
            # RRTToGoalPlanner with the CURRENT config values. Without this,
            # a user who ran trajgen once (planner built with defaults), then
            # tweaked config values in the GUI (e.g. Trajopt Passes), then
            # restarted trajgen — would see the OLD cached planner with OLD
            # values, because `_ensure_planner` at trajectory_generation.py
            # :361 caches on first use and never invalidates. Explicit reset
            # here at mode-entry makes the "Start Trajectory Generation"
            # button a hard reset of the planner too.
            self.trajectory_generator._planner = None
            # Update active resize modes from traj config before creating the dataset
            traj_config = self.trajectory_generator.config
            active_modes = []
            if traj_config.render_letterbox:
                active_modes.append(ImageResizeMode.LETTERBOX)
            if traj_config.render_stretch:
                active_modes.append(ImageResizeMode.STRETCH)
            if active_modes:
                self.image_resize_modes = active_modes
            # Dataset init failures (schema mismatch with the rendering mode,
            # corrupt/incompatible cache, hub errors) must NOT crash the
            # server — surface them in the GUI and bounce back to idle so the
            # user can fix the repo id / rendering toggle and retry.
            try:
                self._init_lerobot_dataset(traj_config.lerobot_repo_id)
            except Exception as e:
                print(f"[LeRobot] Cannot start trajectory generation — dataset init failed: {e}")
                if self._splatsim_gui is not None:
                    self._splatsim_gui.set_status(
                        f"ERROR: dataset '{traj_config.lerobot_repo_id}' cannot be loaded — see console"
                    )
                self.serve_mode = self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE
                return
            # Resume trajectory_count from existing dataset so we don't restart at 0
            if self._lerobot_saver is not None:
                self.trajectory_generator.trajectory_count = self._lerobot_saver.meta.total_episodes
                print(f"[LeRobot] Resuming trajectory generation at index {self.trajectory_generator.trajectory_count}")
        elif mode == self.SERVE_MODES.EVAL_BENCHMARK:
            if self._splatsim_gui is not None:
                self._splatsim_gui.save_to_config(self._splatsim_gui._eval_config, prefix="eval_benchmark")
            gui_repo_id = self._splatsim_gui._eval_config.lerobot_repo_id if self._splatsim_gui is not None else None
            repo_id = gui_repo_id if self._eval_benchmark_repo_id is None or len(self._eval_benchmark_repo_id) == 0 else self._eval_benchmark_repo_id
            if repo_id is None or len(str(repo_id).strip()) == 0:
                print("[EvalBenchmark] No LeRobot repo id configured (GUI field empty and no --eval_benchmark_repo_id).")
                if self._splatsim_gui is not None:
                    self._splatsim_gui.set_status("ERROR: enter a LeRobot repo id, then press Load Dataset")
                self.serve_mode = self.SERVE_MODES.EVAL_BENCHMARK_IDLE
                return
            self._eval_benchmark_episode_index = -1
            self._splatsim_gui.set_status(f"Loading repo_id {repo_id}")
            try:
                self._init_lerobot_dataset(repo_id)
            except Exception as e:
                print(f"[EvalBenchmark] Dataset init failed: {e}")
                if self._splatsim_gui is not None:
                    self._splatsim_gui.set_status(f"ERROR: dataset '{repo_id}' cannot be loaded — see console")
                self.serve_mode = self.SERVE_MODES.EVAL_BENCHMARK_IDLE
                return
            # Parse episode subset from GUI config string (e.g. "3,8,23" or "[3,8,23]")
            subset_str = ""
            if self._splatsim_gui is not None:
                subset_str = self._splatsim_gui._eval_config.episode_subset_str.strip()
            if subset_str:
                cleaned = subset_str.strip("[]")
                self._eval_benchmark_subset = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
            if self._lerobot_saver is not None:
                total = self._lerobot_saver.meta.total_episodes
                if self._eval_benchmark_subset is None:
                    self._eval_benchmark_subset = list(range(total))
                if self._splatsim_gui is not None:
                    self._splatsim_gui.set_status(f"Loaded {len(self._eval_benchmark_subset)} / {total} episodes — ready (reset to start at episode 1)")
                    self._splatsim_gui.set_eval_episode_options(self._eval_benchmark_subset)
            else:
                raise ValueError(f"self._lerobot_saver failed in initialization")
        elif mode == self.SERVE_MODES.EVAL_BENCHMARK_IDLE:
            self._eval_benchmark_episode_index = -1
            self._lerobot_saver = None

    def _exit_mode(self, mode: 'PybulletRobotServerBase.SERVE_MODES'):
        """Called when exiting a serve mode. Override in subclasses for custom behavior."""
        if mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
            traj_config = self.trajectory_generator.config
            self._finalize_lerobot_dataset(push_to_hub=traj_config.push_to_hub)
        elif mode == self.SERVE_MODES.EVAL_BENCHMARK:
            self._lerobot_saver = None
            self._eval_benchmark_episode_index = -1
            if self._splatsim_gui is not None:
                self._splatsim_gui.set_status("")

    # =========================================================================
    # Eval Benchmark Episode Navigation
    # =========================================================================

    def _eval_benchmark_next_episode(self):
        """Advance to the next episode in the benchmark dataset and restore its scenario.

        Called by _handle_reset() when in EVAL_BENCHMARK mode, so both the GUI Reset Env
        button and external policy reset() calls always advance to the next episode.
        No-ops if already at the last episode.
        """
        if self._lerobot_saver is None:
            print("[EvalBenchmark] No dataset loaded.")
            return self.reset()
        self._eval_benchmark_episode_index += 1
        if self._eval_benchmark_episode_index >= len(self._eval_benchmark_subset):
            print(f"[EvalBenchmark] Reached end of sequence ({len(self._eval_benchmark_subset)} episodes), wrapping around.")
            self._eval_benchmark_episode_index = 0
        episode_index = self._eval_benchmark_subset[self._eval_benchmark_episode_index]
        self.restore_episode_scenario(episode_index)
        if hasattr(self, "_reset_episode_state"):
            self._reset_episode_state()
        if self._splatsim_gui is not None:
            self._splatsim_gui.set_status(
                f"Episode: {self._eval_benchmark_episode_index + 1} / {len(self._eval_benchmark_subset)}"
            )
            self._splatsim_gui.set_eval_episode_index(episode_index)
        metrics = self.check_metrics() if hasattr(self, "check_metrics") else {}
        info = {"is_success": metrics.get("is_success", False), **metrics}
        obs = self.get_observations()
        return obs, info

    def _eval_benchmark_prev_episode(self):
        """Go back to the previous episode and restore its scenario (GUI Prev button only).

        Allows stepping back to index -1 (the uninitialized state before episode 1),
        so that the next reset() call will land on episode 1 rather than episode 2.
        """
        if self._lerobot_saver is None or self._eval_benchmark_episode_index < 0:
            return
        self._eval_benchmark_episode_index -= 1
        if self._eval_benchmark_episode_index == -1:
            if self._splatsim_gui is not None:
                self._splatsim_gui.set_status(
                    f"Ready — reset to start at episode 1 / {len(self._eval_benchmark_subset)}"
                )
            return
        episode_id = self._eval_benchmark_subset[self._eval_benchmark_episode_index]
        self.restore_episode_scenario(episode_id)
        if self._splatsim_gui is not None:
            self._splatsim_gui.set_status(
                f"Episode: {self._eval_benchmark_episode_index + 1} / {len(self._eval_benchmark_subset)}"
            )
            self._splatsim_gui.set_eval_episode_index(episode_id)
        return self.get_observations()

    def _eval_benchmark_goto_episode(self, subset_pos: int):
        """Jump to a position in the eval subset and restore its scenario (GUI dropdown)."""
        if self._lerobot_saver is None:
            return
        if not (0 <= subset_pos < len(self._eval_benchmark_subset)):
            print(f"[EvalBenchmark] Subset position {subset_pos} out of range [0, {len(self._eval_benchmark_subset)}).")
            return
        self._eval_benchmark_episode_index = subset_pos
        episode_id = self._eval_benchmark_subset[subset_pos]
        self.restore_episode_scenario(episode_id)
        if self._splatsim_gui is not None:
            self._splatsim_gui.set_status(
                f"Episode: {subset_pos + 1} / {len(self._eval_benchmark_subset)}"
            )
            self._splatsim_gui.set_eval_episode_index(episode_id)
        return self.get_observations()

    def set_eval_benchmark_indices(self, indices):
        """Replace the eval-benchmark playlist and rewind the counter.

        The playlist is a SEQUENCE (not a set): duplicates ARE preserved
        AND played in the order given. Example — passing ``[2, 2, 3, 10, 10]``
        makes the next five ``env.reset()`` calls in EVAL_BENCHMARK mode
        replay scenarios 2, 2, 3, 10, 10 in that order (then wrap to 2).

        Use case: the blending script needs to replay each source-intervention
        episode in the SAME obstacle configuration it was recorded from. It
        builds a playlist matching ``source_scenario_idx`` per source episode
        (with duplicates for multi-episodes-per-scenario) and calls this
        once; each subsequent reset advances one slot forward.

        This differs from the ``options={"benchmark_start_index": N}`` knob on
        ``_handle_reset`` — that jumps the counter per-reset (still indexing
        into the CURRENT playlist), and doesn't allow the caller to rewrite
        the playlist. Use this method when your rollout order is data-driven
        (playlist known up front); use ``benchmark_start_index`` for ad-hoc
        jumps within a fixed subset.

        Resets ``self._eval_benchmark_episode_index = -1`` so the very next
        ``_eval_benchmark_next_episode()`` (i.e. the next ``env.reset()``)
        lands on ``indices[0]``, not ``indices[1]`` — matching the counter's
        "pre-first-reset" invariant used everywhere else on this class.
        """
        if not isinstance(indices, list):
            raise TypeError(f"indices must be a list, got {type(indices).__name__}")
        if not indices:
            raise ValueError("indices must be non-empty")
        for i, x in enumerate(indices):
            if not isinstance(x, int) or isinstance(x, bool):
                raise TypeError(f"indices[{i}] must be int, got {type(x).__name__}")
        # Store as-is: order + duplicates preserved. Do NOT dedup, do NOT sort.
        self._eval_benchmark_subset = list(indices)
        self._eval_benchmark_episode_index = -1
        if self._splatsim_gui is not None:
            self._splatsim_gui.set_status(
                f"Playlist set: {len(self._eval_benchmark_subset)} episodes — ready (next reset = playlist[0])"
            )
            # GUI dropdown will show duplicates as duplicate entries; that's a
            # UX quirk of duplicates, not a functional issue (playback is
            # counter-driven, not dropdown-driven, in playlist mode).
            self._splatsim_gui.set_eval_episode_options(self._eval_benchmark_subset)

    def _eval_benchmark_replay_episode(self):
        """Replay the current episode's recorded observation.state via motor control.

        Visualization for saved datasets (including state-only ones from
        --no_camera_rendering trajectory generation): restores the episode's
        recorded scene (object poses + robot start — restore_episode_scenario
        resets object poses too, so future non-static objects still begin
        correctly; their in-episode motion is re-simulated by physics rather
        than replayed), teleports the robot to frame 0's observation.state,
        then COMMANDS every recorded state at the dataset's fps through
        command_joint_state + physics substeps — the same drive path normal
        execution uses (step() / _render_and_save_episode) — so the replay
        exhibits realistic PD tracking dynamics rather than kinematic snaps.

        The gripper entry (state[num_dofs], 0=open..1=closed — same convention
        for obs and action) is driven through move_gripper inside
        command_joint_state, so the mimic linkage follows physically. Blocks
        the serve loop for the duration of the episode (deliberate — no
        play/pause).
        """
        if self._lerobot_saver is None or not self._eval_benchmark_subset:
            print("[EvalBenchmark] No dataset/episodes loaded — cannot replay.")
            return
        episode_id = self._eval_benchmark_subset[self._eval_benchmark_episode_index]

        # Read the episode's states via a state-only column select — cheap, and
        # works for image-free datasets (no video decoding involved).
        try:
            table = self._lerobot_saver.select_columns(
                ["episode_index", "frame_index", "observation.state"]
            ).to_pandas()
        except Exception as e:
            print(f"[EvalBenchmark] Failed to read observation.state for replay: {e}")
            return
        ep_rows = table[table["episode_index"] == episode_id].sort_values("frame_index")
        states = [np.asarray(s, dtype=np.float64) for s in ep_rows["observation.state"].tolist()]
        if not states:
            print(f"[EvalBenchmark] Episode {episode_id} has no frames — nothing to replay.")
            return

        fps = float(
            getattr(self._lerobot_saver.meta, "fps", None)
            or self.trajectory_generator.config.robot_update_rate
        )

        # Reset the scene to the recorded episode start, then snap the robot to
        # the first recorded state (restore uses initial_joint_positions, which
        # should match frame 0 — the explicit teleport guarantees it).
        self.restore_episode_scenario(episode_id)
        self.teleport_joint_state(self.splatsim_robot, list(states[0]))

        print(f"[EvalBenchmark] Replaying episode {episode_id}: {len(states)} frames @ {fps:.0f} Hz")
        dt = 1.0 / fps
        for i, state in enumerate(states):
            frame_start = time.time()
            # Drive the robot the same way normal execution does (step() /
            # _render_and_save_episode): PD motor commands + physics substeps,
            # NOT a kinematic teleport — so the replay shows realistic tracking
            # dynamics for the recorded state sequence. (The initial teleport
            # above only sets the exact recorded start.)
            self.command_joint_state(self.splatsim_robot, np.asarray(state, dtype=np.float64))
            for _ in range(self._physics_steps_per_action):
                self.pybullet_client.stepSimulation()
            # Refresh GUI camera thumbnails (no-op when camera rendering is off).
            self.get_observations()
            if self._splatsim_gui is not None and i % max(1, int(fps)) == 0:
                self._splatsim_gui.set_status(
                    f"Replaying ep {episode_id}: frame {i + 1}/{len(states)}"
                )
            time.sleep(max(0.0, dt - (time.time() - frame_start)))

        if self._splatsim_gui is not None:
            self._splatsim_gui.set_status(
                f"Replayed ep {episode_id} ({len(states)} frames @ {fps:.0f} Hz)"
            )

    # =========================================================================
    # LeRobot Dataset Lifecycle
    # =========================================================================

    def _create_lerobot_dataset(
        self,
        repo_id: str,
        fps: Optional[int] = None,
        image_keys: Optional[List[str]] = None,
    ) -> LeRobotDataset:
        """Create a fresh LeRobot dataset, defaulting fps/image_keys from instance config."""
        if fps is None:
            fps = self.trajectory_generator.config.robot_update_rate
        if image_keys is None:
            # Declare image features only if this env actually emits images from
            # SOME source — the splat OR the fast PyBullet camera. With neither
            # (e.g. --no_camera_rendering and no PyBullet camera), frames are
            # state/action-only, else LeRobot's validate_frame rejects every
            # add_frame with "Missing features: observation.images.*".
            if not self._produces_images():
                image_keys = []
            else:
                image_keys = [
                    f"{cam}_{mode.value}"
                    for cam in self.camera_names
                    for mode in self.image_resize_modes
                ]
        return create_lerobot_dataset(
            repo_id, fps, image_keys, self.num_dofs(),
            state_dim=self.state_dim(), env_state_dim=self.env_state_dim(),
        )

    def _init_lerobot_dataset(self, repo_id: str):
        """Initialize or load a LeRobot dataset by repo_id."""
        if not repo_id:
            print("[LeRobot] No lerobot_repo_id configured, skipping LeRobot dataset creation.")
            self._lerobot_saver = None
            return

        self._lerobot_saver = load_lerobot_dataset(repo_id)

        if self._lerobot_saver is None:
            print(f"[LeRobot] Creating fresh dataset for {repo_id}")
            self._lerobot_saver = self._create_lerobot_dataset(repo_id)
            return

        # Resuming an existing dataset: its feature schema must agree with the
        # current rendering mode, or every add_frame will fail deep inside
        # LeRobot's validate_frame. Fail fast here with a clear remedy instead.
        # Design choice: image-free runs declare NO observation.images.* keys
        # (rather than zero-filled frames) so state-only consumers work
        # unchanged while image consumers error loudly — a no-render dataset
        # used for image training SHOULD fail, not silently train on black.
        # NOTE: compare against _produces_images() (RenderMode-aware: SPLAT or
        # PYBULLET camera both count), NOT the SPLAT-specific
        # _render_from_splat_default — the latter is False in PYBULLET camera
        # mode and would spuriously reject perfectly valid image datasets.
        dataset_image_keys = [
            k for k in self._lerobot_saver.meta.features if k.startswith("observation.images.")
        ]
        if dataset_image_keys and not self._produces_images():
            # Drop the half-loaded dataset so a failed init leaves no active
            # saver behind (callers catch this error and bounce back to idle;
            # _exit_mode's finalize must not touch the rejected dataset).
            self._lerobot_saver = None
            raise ValueError(
                f"[LeRobot] Dataset '{repo_id}' declares image features "
                f"({sorted(dataset_image_keys)}) but the server's render mode is "
                f"NONE (--no_camera_rendering), so frames would have no images. "
                f"Either relaunch with an image-producing render mode (splat or "
                f"pybullet camera) or use a fresh repo_id for a state/action-only "
                f"dataset."
            )
        if not dataset_image_keys and self._produces_images():
            self._lerobot_saver = None
            raise ValueError(
                f"[LeRobot] Dataset '{repo_id}' is state/action-only (no image "
                f"features), but the server is rendering images — frames would "
                f"carry extra image keys the schema rejects. Either relaunch "
                f"with --no_camera_rendering to keep appending state-only "
                f"episodes, or use a fresh repo_id for an image dataset."
            )

    def _finalize_lerobot_dataset(self, push_to_hub: bool = False):
        """Finalize and optionally push the LeRobot dataset."""
        if self._lerobot_saver is None:
            return
        print("[LeRobot] Finalizing dataset...")
        finalize_lerobot_dataset(self._lerobot_saver)
        if push_to_hub:
            push_lerobot_to_hub(self._lerobot_saver)
        self._lerobot_saver = None

    # =========================================================================
    # Scene State Save / Restore
    # =========================================================================

    def _save_scene_state(self) -> dict:
        """Snapshot positions, orientations, and joint states of all splatsim_objects."""
        state = {}
        for obj in self.splatsim_objects:
            if obj.sim_id is None or obj == self.splatsim_background:
                continue
            if obj.config.is_articulated:
                num_joints = self.pybullet_client.getNumJoints(obj.sim_id)
                joint_states = [
                    self.pybullet_client.getJointState(obj.sim_id, i)
                    for i in range(num_joints)
                ]
                state[obj.config.name] = {
                    "pos_orn": self.pybullet_client.getBasePositionAndOrientation(obj.sim_id),
                    "joint_states": [(js[0], js[1]) for js in joint_states],
                }
            else:
                state[obj.config.name] = {
                    "pos_orn": self.pybullet_client.getBasePositionAndOrientation(obj.sim_id),
                }
        return state

    def _restore_scene_state(self, state: dict):
        """Restore all object positions, orientations, and joint states from snapshot."""
        for obj in self.splatsim_objects:
            if obj.config.name not in state:
                continue
            saved = state[obj.config.name]
            pos, orn = saved["pos_orn"]
            self.pybullet_client.resetBasePositionAndOrientation(obj.sim_id, pos, orn)
            if obj.config.is_articulated and "joint_states" in saved:
                for i, (jp, jv) in enumerate(saved["joint_states"]):
                    self.pybullet_client.resetJointState(obj.sim_id, i, jp, jv)

    # =========================================================================
    # Trajectory Generation + Rendering Pipeline
    # =========================================================================

    def _is_stop_requested(self) -> bool:
        """Check if the user has pressed Stop in the GUI (non-consuming peek)."""
        return self._splatsim_gui.peek_button("stop_traj")

    def _generate_and_render_one_episode(self):
        """Generate trajectories, render each step, and save to LeRobot + Zarr."""
        self.reset()
        # Snapshot scene state AFTER reset, BEFORE trajectory generation
        scene_state = self._save_scene_state()

        # Start from the current state
        self.trajectory_generator.config.q_start = self.get_joint_state()[:self.num_dofs()].tolist()

        episodes = self.trajectory_generator.generate_trajectory_batch()
        if episodes is None:
            return  # Planning failed, will retry next iteration

        for episode in episodes:
            if self._is_stop_requested():
                print("[TrajectoryGen] Stop requested, finishing current batch early.")
                break
            # Restore scene to post-reset state before rendering each episode
            self._restore_scene_state(scene_state)
            # Freeze the visualizer's world redraw for the episode's frame
            # loop: in GUI mode every stepSimulation + getCameraImage syncs
            # with the 3D view, which caps batch rendering near realtime.
            # With the redraw frozen the loop runs compute-bound (often
            # several× realtime); progress stays visible via the SplatSim
            # GUI camera thumbnails, which are fed by getCameraImage and
            # still render fine while the world view is frozen.
            if not self._headless:
                self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            try:
                self._render_and_save_episode(episode)
            finally:
                if not self._headless:
                    self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Config sub-fields that are STATIC per robot/scene (identical in every
    # episode) and are never read back by any consumer. They are stripped from
    # the payload built below — see `_strip_static_config_fields`.
    #
    # `segmented_list` (the per-link gaussian point-index segmentation) is by
    # far the worst: ~865 KB. Written once per episode it made a 500-episode
    # LeRobot `meta/episodes` table ~500 MB / ~1 MB per row, and a single row
    # read ~19 ms. LeRobot's video-decode path reads that row once per camera
    # per sample, so it dominated dataloader time for video-backed training
    # (measured: data_s 0.85 s/batch, GPU idle ~90% of the time). It also rode
    # along in every `get_env_config` ZMQ reply to the SA wrapper.
    #
    # What scenario restore + the analysis scripts actually consume is only
    # `articulation_config.initial_joint_positions` (which genuinely varies per
    # episode) and the per-object initial pose/scale — all kept, all tiny.
    _STATIC_ARTICULATION_FIELDS = ("segmented_list",)

    @classmethod
    def _strip_static_config_fields(cls, d: dict) -> dict:
        """Drop static, never-read bulk from a serialized object config.

        Kept deliberately narrow: only the named articulation sub-fields are
        removed, so every other field survives for provenance/debuggability.
        """
        art = d.get("articulation_config")
        if isinstance(art, dict):
            for key in cls._STATIC_ARTICULATION_FIELDS:
                art.pop(key, None)
        return d

    def _get_splatsim_episode_metadata(self) -> dict:
        """Build the splatsim-specific episode metadata dict for LeRobot save_episode().

        Contains JSON-serialisable configs for the robot, background, and all
        non-robot/non-background objects currently in the scene, minus the
        static bulk listed in `_STATIC_ARTICULATION_FIELDS`.
        """
        def _config_to_dict(cfg):
            d = asdict(cfg)
            d = json.loads(json.dumps(d, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
            return self._strip_static_config_fields(d)

        return {
            "splatsim_robot_config": _config_to_dict(self.splatsim_robot.config),
            "splatsim_background_config": (
                _config_to_dict(self.splatsim_background.config)
                if self.splatsim_background is not None else None
            ),
            "splatsim_object_configs": [
                _config_to_dict(obj.config) for obj in self.splatsim_objects
                if obj is not self.splatsim_robot and obj is not self.splatsim_background
            ],
        }

    def _render_and_save_episode(self, episode: dict):
        """Step through a trajectory using motor control, render images, save frames to LeRobot + Zarr.

        At each step the planner's target joint configuration is sent via command_joint_state,
        physics is stepped for _physics_steps_per_action substeps, then the actual post-physics
        joint state is read back.  This matches eval-time dynamics so that the saved
        observation.state reflects what the policy will actually observe at inference time.

        action  = planner target q (what the policy should output)
        state   = actual joint positions after physics stepping (what the policy observes)
        """
        joint_trajectory = episode["joint_positions"]  # (N, DOF)
        obstacle_info = episode.get("obstacle_info", {"obstacles": []})
        zarr_group = episode.get("zarr_group")

        # Set up obstacles from metadata
        loaded_obstacle_names = []
        for i, obstacle in enumerate(obstacle_info.get("obstacles", [])):
            if obstacle["type"] == "cuboid":
                obstacle_name = f"_render_obstacle_cuboid{i}"
                obstacle_config = CuboidObjectConfig(
                    name=obstacle_name,
                    size=tuple(obstacle["size"]),
                    position=tuple(obstacle["pos"]),
                    base_quat=tuple(obstacle["orn"]),
                    load_splat=True,
                    load_urdf=True,
                    randomize_pose=False,
                )
                self.create_object(obstacle_config)
                loaded_obstacle_names.append(obstacle_name)
            else:
                print(f"[Render] Unknown obstacle type: {obstacle['type']}, skipping.")

        # Ensure rendering is enabled — but only when the env's authoritative
        # render mode produces images at all (SPLAT or PYBULLET; see
        # _produces_images / _initial_render_mode). This call exists to undo a
        # temporary disable_rendering() from earlier phases (e.g. planning);
        # it must not override RenderMode.NONE (--no_camera_rendering), where
        # trajectory-gen episodes are saved state/action-only
        # (build_lerobot_frame and the Zarr collector both skip None images).
        # NOTE: checking _render_from_splat_default here would be wrong — that
        # flag is SPLAT-specific, and in PYBULLET camera mode it is False even
        # though the env produces images, which made every episode fall into
        # the mismatch-warning branch below.
        if self._produces_images():
            self.enable_rendering()
        elif self._lerobot_saver is not None and any(
            k.startswith("observation.images.") for k in self._lerobot_saver.meta.features
        ):
            # Mid-run mismatch: an image-schema dataset is already open (created
            # while rendering was on) but rendering has since been toggled off
            # in the GUI. The dataset schema is fixed at creation, so frames
            # WITHOUT images would fail LeRobot's validate_frame on add_frame.
            # The schema wins for the current dataset: force rendering back on
            # for this episode and tell the user how to get a state-only run.
            print(
                "[TrajectoryGen] WARNING: camera rendering was toggled off, but the "
                "open LeRobot dataset declares image features — re-enabling rendering "
                "for this episode to keep frames schema-consistent. For a state-only "
                "dataset, stop generation and restart it with rendering off (a fresh "
                "repo_id) instead."
            )
            self.enable_rendering()

        # Teleport to the first waypoint so the robot starts at the right configuration
        # before we begin issuing motor commands.
        if len(joint_trajectory) > 0:
            q0 = joint_trajectory[0]
            self.teleport_joint_state(self.splatsim_robot, list(q0))
            # Sync the episode metadata's robot start to the trajectory's ACTUAL
            # first waypoint. The planner may legally shift the start away from
            # the reset pose (its start-escape runs when the reset pose violates
            # the planner's clearance contract — reset validates at looser
            # clearances), and save_episode() reads initial_joint_positions for
            # splatsim_robot_config. Without this sync, eval-benchmark restore
            # (metadata) and replay (recorded frames) disagree about where the
            # episode starts — exactly the "restore shows one pose, replay
            # starts from another" bug.
            art_cfg = self.splatsim_robot.config.articulation_config
            if art_cfg is not None:
                prev_q = np.asarray(art_cfg.initial_joint_positions[:self.num_dofs()], dtype=np.float64)
                drift = float(np.max(np.abs(prev_q - np.asarray(q0[:self.num_dofs()], dtype=np.float64))))
                if drift > 0.01:
                    print(f"[TrajectoryGen] Planned start differs from reset pose by "
                          f"{drift:.3f} rad (planner start-escape) — syncing episode "
                          f"metadata to the actual trajectory start.")
                art_cfg.initial_joint_positions = [float(v) for v in q0]

        # Collect image buffers for Zarr saving
        image_buffers = defaultdict(list)

        stopped_early = False
        for step_idx in range(len(joint_trajectory)):
            if self._is_stop_requested():
                print(f"[TrajectoryGen] Stop requested at step {step_idx}/{len(joint_trajectory)}, saving partial episode.")
                stopped_early = True
                break

            q = joint_trajectory[step_idx]

            # Build a 7-DOF command (6 joints + gripper open = 0)
            action_7 = np.zeros(self.num_dofs() + 1, dtype=np.float32)
            action_7[:len(q)] = q

            # Command the robot and step the physics simulation
            self.command_joint_state(self.splatsim_robot, action_7)
            for _ in range(self._physics_steps_per_action):
                self.pybullet_client.stepSimulation()

            # Read back the actual post-physics joint state
            obs = self.get_observations()

            # Save to LeRobot dataset. Derive image_keys from the dataset's
            # DECLARED schema (not camera_names × resize_modes) so frames carry
            # exactly the image features the dataset expects — e.g. a
            # state-only dataset (created with rendering off) stays state-only
            # even if the GUI toggles rendering back on mid-run for display.
            if self._lerobot_saver is not None:
                image_keys = [
                    k[len("observation.images."):]
                    for k in self._lerobot_saver.meta.features
                    if k.startswith("observation.images.")
                ]
                self._lerobot_saver.add_frame(
                    build_lerobot_frame(
                        action_7, obs, image_keys,
                        task=self.get_task_description(),
                        num_dofs=self.num_dofs(),
                    )
                )

            # Collect images for Zarr saving
            for cam in self.camera_names:
                for mode in self.image_resize_modes:
                    key = f"{cam}_{mode.value}"
                    if obs.get(key) is not None:
                        # Convert from (C, H, W) float [0,1] to (H, W, C) uint8
                        img = obs[key]
                        img = np.transpose(img, (1, 2, 0))  # CxHxW -> HxWxC
                        img = (img * 255).astype(np.uint8)
                        image_buffers[key].append(img)

        # Save episode to LeRobot (skip partial episodes from early stop)
        if not stopped_early and self._lerobot_saver is not None:
            self._lerobot_saver.save_episode(episode_metadata=self._get_splatsim_episode_metadata())

        # Save images to Zarr (save even if partial — zarr is more forgiving)
        if zarr_group is not None:
            for cam_name, frames in image_buffers.items():
                if len(frames) > 0:
                    images = np.stack(frames, axis=0)  # (T, H, W, C)
                    if cam_name in zarr_group:
                        del zarr_group[cam_name]
                    zarr_group.create_dataset(cam_name, data=images, dtype="f4")

        # Clean up obstacles
        for obstacle_name in loaded_obstacle_names:
            self.delete_object(obstacle_name)

    def _handle_reset(self, seed=None, options=None):
        """Unified reset entry point for serve() loop and ZMQ server.

        In EVAL_BENCHMARK mode, advances to the next episode in
        ``self._eval_benchmark_subset`` instead of calling the subclass
        reset(), so external policies that call reset() always get the next
        deterministic scenario without needing to know the serve mode.

        Scenario selection vs. policy randomness
        ----------------------------------------
        The scenario INDEX is determined ENTIRELY by the subset + the
        per-reset counter (which wraps at end-of-subset). The ``seed``
        argument does NOT affect which scenario runs — its only role is
        seeding the env/policy randomness. This lets callers re-evaluate
        the same subset with multiple seeds (varying policy stochasticity
        only) and get identical scenario coverage across runs.

        Caller-side override
        --------------------
        Callers that need to force a specific starting scenario (e.g.
        lerobot-train's per-batch eval wanting determinism when
        eval_n_episodes < len(subset)) can pass
        ``options={"benchmark_start_index": N}`` to set the next
        ``_eval_benchmark_next_episode()`` to land on
        ``subset[N % len(subset)]``. Without this, the counter just
        increments from wherever it left off (which wraps to 0 cleanly
        when eval_n_episodes == len(subset) — the common case).

        In all other modes, delegates to the subclass reset().
        """
        # Reset is a mutation: bump BEFORE so the obs captured during/after
        # the reset are stamped ≥ this version, invalidating pre-reset obs.
        self._bump_state_version()
        if self.serve_mode == self.SERVE_MODES.EVAL_BENCHMARK:
            if (
                options is not None
                and "benchmark_start_index" in options
                and self._eval_benchmark_subset
            ):
                n = len(self._eval_benchmark_subset)
                self._eval_benchmark_episode_index = (int(options["benchmark_start_index"]) % n) - 1
            return self._eval_benchmark_next_episode()
        else:
            return self.reset(seed=seed, options=options)

    def serve(self) -> None:
        self.reset()

        self._lerobot_saver = None
        _prev_serve_mode = self.serve_mode
        # The mode-transition detector below only fires when serve_mode CHANGES,
        # so a mode set by the constructor (e.g. EVAL_BENCHMARK from
        # launch_nodes.py --eval_benchmark_repo_id=...) would never go through
        # _enter_mode and the LeRobot dataset / GUI status would never be
        # initialized. Explicitly enter the initial mode here.
        #
        # CRITICAL: this MUST happen BEFORE the ZMQ thread starts. If ZMQ
        # accepts a reset() request from an external policy (e.g. lerobot-eval)
        # before _enter_mode finishes initializing EVAL_BENCHMARK state, the
        # _handle_reset → _eval_benchmark_next_episode path sees subset=None
        # and _lerobot_saver=None and falls through to a random self.reset()
        # — so the policy records its first "scenario 0" against a random
        # scenario, and the next reset (whose seed pin then works) skips
        # subset[0] straight to subset[1]. Letting _enter_mode complete first
        # guarantees the first external reset hits a fully-initialized
        # benchmark state.
        self._enter_mode(self.serve_mode)

        # Capture the main serve-loop thread BEFORE ZMQ starts. Used by
        # _render_pybullet_camera to decide whether it needs to marshal its
        # GL call to us (headless-EGL context is bound to this thread) or
        # can execute directly. `threading.current_thread()` here IS the
        # thread that will run the while-True loop below.
        self._main_thread = threading.current_thread()

        # start the zmq server only after benchmark state is ready
        self._zmq_server_thread.start()

        print("Ready to serve.")

        try:
            while True:
                # Let the GUI handle all mode/button transitions
                self._splatsim_gui.process_mode_transitions()

                # Reset env button — available in all modes
                if self._splatsim_gui.check_button(SplatSimGui.BTN_RESET_ENV):
                    print("[GUI] Reset Env pressed — resetting environment (randomized).")
                    # A user pressing Reset wants a FRESH scene: bypass any
                    # pinned --scenario_file arrangement (see the small-engine
                    # reset's pin gate).
                    self._handle_reset(options={"force_randomize": True})

                # Check debug mode dropdown for changes
                self._check_debug_mode()

                # Sync camera rendering with the GUI checkbox
                self._check_camera_rendering_toggle()

                # Detect and handle mode transitions
                current_mode = self.serve_mode
                if _prev_serve_mode != current_mode:
                    self._exit_mode(_prev_serve_mode)
                    self._enter_mode(current_mode)
                    _prev_serve_mode = current_mode

                if current_mode == self.SERVE_MODES.INTERACTIVE:
                    # In sync-to-client mode, `command_joint_state` posts a
                    # step-request to `_sync_step_request_ticks` and blocks.
                    # We consume it here on the MAIN thread — pybullet's GUI
                    # OpenGL context is bound to this thread, so calling
                    # stepSimulation from the ZMQ handler thread would
                    # deadlock. Still sleep to keep the GUI responsive and
                    # cede the GIL to the ZMQ thread when idle.
                    if self._sync_physics_to_client:
                        self._consume_sync_step_request()
                    else:
                        self.pybullet_client.stepSimulation()
                    # Also consume any pending camera-render request. Always
                    # active (not gated on _sync_physics_to_client) because
                    # the EGL-context-thread-affinity issue is unconditional
                    # in headless mode — we always want camera renders to
                    # execute on this thread.
                    self._consume_sync_render_request()
                    time.sleep(1 / 240)
                elif current_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
                    # Idle mode - just step simulation while user configures settings
                    self.pybullet_client.stepSimulation()
                    self._consume_sync_render_request()
                    time.sleep(1 / 240)
                elif current_mode == self.SERVE_MODES.GENERATE_DEMOS:
                    raise NotImplementedError()
                    initial_joint_positions = self.randomize_ee_pose()

                    success = self.plan_execute_record_trajectory(
                        initial_joint_positions, self.splatsim_robot.articulation_config.joint_signs
                    )
                    if success:
                        self.trajectory_count += 1

                    if self.trajectory_count > self.MAX_TRAJECTORY_COUNT:
                        print(
                            f"Exiting record_demos mode because max trajectory count of {self.MAX_TRAJECTORY_COUNT} was reached in folder {self.path}"
                        )
                        self.serve_mode = self.SERVE_MODES.INTERACTIVE
                elif current_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES:
                    # Generate trajectories, render images, and save to LeRobot + Zarr
                    try:
                        self._generate_and_render_one_episode()
                    except RuckigCloudUnavailableError as e:
                        # ruckig backend's cloud API down / daily rate limit hit: every
                        # further episode would fail identically. Stop
                        # generation gracefully (per-episode dataset state
                        # was already finalized by the failing episode's own
                        # cleanup) and drop to idle so the server + GUI stay
                        # alive.
                        msg = (
                            "Trajectory generation STOPPED: ruckig-backend cloud "
                            f"unavailable ({e}). Generated "
                            f"{self.trajectory_generator.trajectory_count} episode(s) "
                            "before stopping; the dataset up to here is intact. "
                            "Resume generation after the rate limit resets."
                        )
                        print(f"[serve] {msg}")
                        if self._splatsim_gui is not None:
                            self._splatsim_gui.set_status(
                                "STOPPED: ruckig-backend cloud rate-limited/unreachable"
                            )
                        self.serve_mode = self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE
                        continue

                    # Update GUI status with current progress
                    tgen = self.trajectory_generator
                    total = tgen.config.num_base_trajectories
                    done = tgen.trajectory_count
                    if self._splatsim_gui is not None:
                        self._splatsim_gui.set_status(f"Trajectory: {done} / {total}")

                    if self.trajectory_generator.is_complete():
                        print(f"[GUI] Completed trajectory generation. Switching to idle mode.")
                        if self._splatsim_gui is not None:
                            self._splatsim_gui.set_status(f"Done: {done} / {total} trajectories")
                        self.serve_mode = self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE
                elif current_mode == self.SERVE_MODES.EVAL_BENCHMARK_IDLE:
                    # See INTERACTIVE branch above for sync-to-client rationale.
                    if self._sync_physics_to_client:
                        self._consume_sync_step_request()
                    else:
                        self.pybullet_client.stepSimulation()
                    self._consume_sync_render_request()
                    time.sleep(1 / 240)
                elif current_mode == self.SERVE_MODES.EVAL_BENCHMARK:
                    # See INTERACTIVE branch above for sync-to-client rationale.
                    if self._sync_physics_to_client:
                        self._consume_sync_step_request()
                    else:
                        self.pybullet_client.stepSimulation()
                    self._consume_sync_render_request()
                    time.sleep(1 / 240)
                    # Handle Next/Prev/dropdown buttons from the eval benchmark panel
                    from splatsim.utils.splatsim_gui import EvalBenchmarkModePanel
                    if self._splatsim_gui.check_button(EvalBenchmarkModePanel.BTN_NEXT):
                        self._eval_benchmark_next_episode()
                    if self._splatsim_gui.check_button(EvalBenchmarkModePanel.BTN_PREV):
                        self._eval_benchmark_prev_episode()
                    if self._splatsim_gui.check_button(EvalBenchmarkModePanel.BTN_REPLAY):
                        self._eval_benchmark_replay_episode()
                    # Read and immediately consume the dropdown value. We must
                    # reset it to "—" after reading so that programmatic updates
                    # from set_eval_episode_index() (called by _eval_benchmark_next_episode
                    # and _eval_benchmark_goto_episode) don't get re-read by this
                    # loop as a new user selection, causing duplicate restore calls.
                    selected = self._splatsim_gui.get_value(EvalBenchmarkModePanel.EPISODE_SELECT_KEY)
                    if selected is not None and selected != "—":
                        self._splatsim_gui.set_value(EvalBenchmarkModePanel.EPISODE_SELECT_KEY, "—")
                        try:
                            episode_id = int(selected)
                            current_pos = self._eval_benchmark_episode_index
                            current_id = (
                                self._eval_benchmark_subset[current_pos]
                                if 0 <= current_pos < len(self._eval_benchmark_subset)
                                else None
                            )
                            # The position-mismatch guard below is not enough when the
                            # subset contains DUPLICATE ids (e.g. a blend-replay playlist
                            # like [6, 17, 40, 40, 41, ...] from set_eval_benchmark_indices):
                            # .index() maps the echo to the FIRST duplicate, which differs
                            # from the counter on every later occurrence — rewinding the
                            # counter mid-run on each reset. An echo always carries the
                            # CURRENT episode's id, so only treat differing ids as a jump.
                            if episode_id != current_id and episode_id in self._eval_benchmark_subset:
                                subset_pos = self._eval_benchmark_subset.index(episode_id)
                                if subset_pos != current_pos:
                                    self._eval_benchmark_goto_episode(subset_pos)
                        except (ValueError, TypeError):
                            pass
                else:
                    raise ValueError(f"Unknown serve mode {current_mode}. ")

                self.serve_loop()
        except Exception as e:
            import traceback
            print(f"[serve] ERROR: {e}")
            traceback.print_exc()
            raise
        finally:
            self._finalize_lerobot_dataset()

    def serve_loop(self):
        raise NotImplementedError()

    def plan_given_this_state(self, initial_joint_positions):
        raise NotImplementedError()

    def delete_trajectory_folder(self):
        shutil.rmtree(os.path.join(self.path, str(self.trajectory_count).zfill(3)))

    def stop(self) -> None:
        """Stop the robot server and clean up resources.

        Safe to call whether or not serve() was called (gym mode vs ZMQ server mode).
        """
        # Only join the thread if it was actually started
        if self._zmq_server_thread.is_alive():
            self._zmq_server.stop()
            self._zmq_server_thread.join(timeout=2.0)

        # Clean up GUI if it exists — must happen before pybullet disconnect
        # so the Tk thread can shut down gracefully
        if self._splatsim_gui is not None:
            self._splatsim_gui.stop()
            self._splatsim_gui = None

        # Free GPU memory held by gaussian splat models and CUDA tensors
        self._cleanup_gpu_resources()

        # Disconnect pybullet
        if self.pybullet_client is not None:
            try:
                self.pybullet_client.disconnect()
            except Exception:
                pass  # Already disconnected

    def _cleanup_gpu_resources(self) -> None:
        """Release all CUDA tensors and gaussian splat models."""
        import gc

        # Clear gaussian splat models from splatsim objects
        for obj in self.splatsim_objects:
            if obj.gaussians is not None:
                del obj.gaussians
                obj.gaussians = None
            obj._cache.clear()
        self.splatsim_objects.clear()

        # Clear the background splat
        if self.splatsim_background is not None:
            del self.splatsim_background.gaussians
        self.splatsim_background = None

        # Clear the robot splat
        if self.splatsim_robot is not None:
            del self.splatsim_robot.gaussians
        self.splatsim_robot = None

        # Clear scene gaussian and labels
        del self.scene_gaussian
        self.scene_gaussian = None
        del self.robot_labels
        self.robot_labels = None

        # Clear camera resources
        for cam in (self.base_camera, self.wrist_camera):
            if cam is not None:
                if cam.background is not None:
                    del cam.background
                if cam.camera is not None:
                    del cam.camera
        self.base_camera = None
        self.wrist_camera = None

        # Force garbage collection and release CUDA cache
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass  # Ignore errors during garbage collection

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.splatsim_robot.sim_id)
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.splatsim_robot.sim_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[
                2
            ]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    jointID,
                    self.pybullet_client.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0,
                )
            info = jointInfo(
                jointID,
                jointName,
                jointType,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                controllable,
            )
            self.joints.append(info)

    def setup_gripper(self):
        self.__parse_joint_info__()
        self.gripper_range = [0, 0.085]

        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            # "finger_joint": 1, # TODO: is this left_outer_knuckle_joint?
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        # self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.pybullet_client.createConstraint(
                self.splatsim_robot.sim_id,
                self.mimic_parent_id,
                self.splatsim_robot.sim_id,
                joint_id,
                jointType=self.pybullet_client.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            # maxForce stays 10 in EVERY mode, including debug_fast_control.
            # Stiffening the gears for debug mode was tried and is actively
            # harmful: the gear's erp term tracks an internal reference that
            # does NOT follow resetJointState, so after a debug-mode teleport
            # a stiff gear drives the fingers at a sustained ~15 rad/s
            # against the position holds (measured). At maxForce=10 that
            # residual fight is negligible, and debug mode doesn't need the
            # gears anyway — move_gripper position-holds every mimic child.
            self.pybullet_client.changeConstraint(
                c, gearRatio=-multiplier, maxForce=10, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

        # Disable PHYSICS self-collision among the gripper's own links. The
        # Robotiq 2F-85 is a 4-bar linkage whose inner-finger and inner-knuckle
        # collision meshes overlap by design; with URDF_USE_SELF_COLLISION on,
        # those overlaps (e.g. left_inner_finger vs left_inner_knuckle) register
        # as penetrating contacts that jam the mimic — the gripper gets pinned
        # in a squeezed mid-range and can neither fully open nor fully close.
        # Collision with EXTERNAL objects (grasping) is a separate filter and is
        # unaffected. (SELF_COLLISION_SKIP_PAIRS only feeds the RRT planner's
        # collision check, not the physics engine, so it doesn't help here.)
        gripper_joint_names = {
            "finger_joint", "left_outer_finger_joint", "left_inner_finger_joint",
            "left_inner_finger_pad_joint", "left_inner_knuckle_joint",
            "right_outer_knuckle_joint", "right_outer_finger_joint",
            "right_inner_finger_joint", "right_inner_finger_pad_joint",
            "right_inner_knuckle_joint",
        }
        gripper_link_ids = [j.id for j in self.joints if j.name in gripper_joint_names]
        for a in range(len(gripper_link_ids)):
            for b in range(a + 1, len(gripper_link_ids)):
                self.pybullet_client.setCollisionFilterPair(
                    self.splatsim_robot.sim_id, self.splatsim_robot.sim_id,
                    gripper_link_ids[a], gripper_link_ids[b], enableCollision=0,
                )

        # Unify the PHYSICS-side skip contract with the RRT PLANNER's skip
        # contract. Every pair in SELF_COLLISION_SKIP_PAIRS is a URDF-artifact
        # overlap the planner already ignores; if physics still enforces them,
        # planner-accepted configs get pinned by constraint forces at the mesh
        # overlap (drift-abort). See analyze_forearm_wrist2_penetration.py:
        # forearm↔wrist_2 has a ~120° wrist_1 zone where the UR5 stock
        # collision hulls overlap by design, which was jamming joint 3 during
        # RRT chunk execution. NOT extended to
        # SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA — those pairs the
        # planner still checks, so disabling them at the physics level would
        # let RRT accept a plan whose execution silently produced mesh
        # overlap.
        for link_a, link_b in getattr(self, "SELF_COLLISION_SKIP_PAIRS", ()):
            self.pybullet_client.setCollisionFilterPair(
                self.splatsim_robot.sim_id, self.splatsim_robot.sim_id,
                link_a, link_b, enableCollision=0,
            )

    def get_link_pose(self, body, link):
        result = self.pybullet_client.getLinkState(body, link)
        return result[4], result[5]

    def move_gripper(self, open_length, velocity=2):
        if not self.use_gripper:
            return
        # open_length = np.clip(open_length, *self.gripper_range)
        # Don't throw an error when out of range
        open_angle = 0.715 - math.asin(np.clip(
            (open_length - 0.010) / 0.1143,
            -1,
            1
        ))  # angle calculation
        # Control the mimic gripper joint(s). Debug-fast mode raises the hold
        # force so arm-snap inertia can't back-drive the finger drive joint
        # (the gear constraints are stiffened to match in setup_gripper).
        drive_force = self.joints[self.mimic_parent_id].maxForce
        if self.debug_fast_control:
            drive_force = max(drive_force, self.DEBUG_FAST_CONTROL_GRIPPER_FORCE)
            # Debug mode: the whole gripper snaps like the arm does — the
            # realistic per-call `velocity` cap (2 rad/s) is far too slow to
            # reject arm-snap disturbances on the finger links.
            velocity = self.DEBUG_FAST_CONTROL_MAX_VELOCITY
        self.pybullet_client.setJointMotorControl2(
            self.splatsim_robot.sim_id,
            self.mimic_parent_id,
            self.pybullet_client.POSITION_CONTROL,
            targetPosition=open_angle,
            force=drive_force,
            maxVelocity=velocity,
        )
        if self.debug_fast_control:
            # KINEMATIC snap + hold for every mimic CHILD, mirroring the
            # arm's debug path in command_joint_state: teleport each finger
            # joint to its gear-consistent angle and position-hold it there.
            # The JOINT_GEAR coupling alone is a velocity-level constraint —
            # even at DEBUG_FAST_CONTROL_GRIPPER_FORCE it lets the fingers
            # flap 25-65 deg during a debug-speed arm snap (measured;
            # transient, self-recovering, but visually a twitching finger).
            # Normally children MUST stay motor-free (a position hold at a
            # STALE target jams the mimic — see the teleport/setup notes);
            # here it's safe because parent and child targets are set
            # TOGETHER from the same open_angle on every call, so the
            # motors and gears always agree.
            self.pybullet_client.resetJointState(
                self.splatsim_robot.sim_id, self.mimic_parent_id, open_angle
            )
            for child_id, mult in self.mimic_child_multiplier.items():
                self.pybullet_client.resetJointState(
                    self.splatsim_robot.sim_id, child_id, mult * open_angle
                )
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    child_id,
                    self.pybullet_client.POSITION_CONTROL,
                    targetPosition=mult * open_angle,
                    force=self.DEBUG_FAST_CONTROL_GRIPPER_FORCE,
                    maxVelocity=velocity,
                )
        elif self.GRIPPER_MIMIC_HOLD_FORCE > 0:
            # NORMAL-mode mimic rigidity (see the class attr): position-hold
            # the mimic children at their gear-consistent angles so contact
            # can't bend the follower fingers away from the drive joint.
            # Targets refresh on every call — always in agreement with the
            # parent target and the gears, so no stale-target jam.
            for child_id, mult in self.mimic_child_multiplier.items():
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    child_id,
                    self.pybullet_client.POSITION_CONTROL,
                    targetPosition=mult * open_angle,
                    force=self.GRIPPER_MIMIC_HOLD_FORCE,
                    maxVelocity=velocity,
                )

    def get_current_gripper_state(self):
        # Snap the gripper state to 0 or 1 if they're very close
        gripper_state = self.pybullet_client.getJointState(
            self.splatsim_robot.sim_id, self.mimic_parent_id
        )[0]
        return gripper_state

    def get_camera_image_from_end_effector(self):

        cam_fov = 90
        near_plane = 0.01
        far_plane = 100
        # Get the pose of the end effector
        end_effector_state = self.pybullet_client.getLinkState(
            self.splatsim_robot.sim_id, 8
        )
        end_effector_pos = end_effector_state[4]
        end_effector_orn = end_effector_state[5]

        # Convert the quaternion orientation to a rotation matrix
        rot_matrix = self.pybullet_client.getMatrixFromQuaternion(end_effector_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Define the camera position relative to the end effector
        camera_position = np.array([0, 0, 0.1])  # Adjust as needed
        camera_position_world = end_effector_pos + rot_matrix.dot(camera_position)

        # Define the camera view direction
        camera_target_position = np.array([0, 0, 1])  # Adjust as needed
        camera_target_position_world = end_effector_pos + rot_matrix.dot(
            camera_target_position
        )

        # Compute the view matrix
        view_matrix = self.pybullet_client.computeViewMatrix(
            camera_position_world, camera_target_position_world, [0, 0, 1]
        )

        # Compute the projection matrix
        projection_matrix = self.pybullet_client.computeProjectionMatrixFOV(
            cam_fov, 1.0, near_plane, far_plane
        )

        # Get the camera image
        (
            width,
            height,
            rgb_img,
            depth_img,
            seg_img,
        ) = self.pybullet_client.getCameraImage(
            320,
            240,
            view_matrix,
            projection_matrix,
            flags=self.pybullet_client.ER_NO_SEGMENTATION_MASK,
        )

        return rgb_img

    def pre_grasp_to_grasp(self, transformation_matrix):

        # get the end effector position and orientation according to self.apple_grasp_pose
        ee_transformation = transformation_matrix

        # get approach vector
        approach_vector = ee_transformation[:3, :3][:, 1] / np.linalg.norm(
            ee_transformation[:3, :3][:, 1]
        )

        # check the angle of the approach vector with the -ve z axis
        angle = np.arccos(np.dot(approach_vector, np.array([0, 0, -1])))
        if angle > 0.1 or angle < -0.1:
            print("angle is greater than 0.1")
            return None, None

        # check the angle of the approach vector with the y axis
        angle = np.arccos(np.dot(approach_vector, np.array([0, 1, 0])))
        if np.abs(angle - np.pi / 2) > 0.1:
            print("angle is greater than 0.1")
            return None, None

        # check the angle of the approach vector with the x axis
        angle = np.arccos(np.dot(approach_vector, np.array([1, 0, 0])))
        print(angle)

        # get the pre-grasp position
        pre_grasp_pos = ee_transformation[:3, 3] - 0.1 * approach_vector

        # pre-grasp transformation
        pre_grasp_transformation = np.eye(4)
        pre_grasp_transformation[:3, 3] = pre_grasp_pos
        pre_grasp_transformation[:3, :3] = ee_transformation[:3, :3]

        # get joint positions going from pre-grasp to grasp
        path = []
        for i in range(11):
            ee_pos = pre_grasp_pos + 0.01 * i * approach_vector
            ee_quat = self.pybullet_client.getQuaternionFromEuler(
                rotation_matrix_to_euler_angles(ee_transformation[:3, :3])
            )
            # maxNumIterations=512 (was 100000): same rationale as
            # randomize_ee_pose — DLS converges well under 200 iterations and
            # the residual threshold never early-exits, so 100k only burned time.
            joint_positions = self.pybullet_client.calculateInverseKinematics(
                self.splatsim_robot.sim_id,
                6,
                ee_pos,
                ee_quat,
                maxNumIterations=512,
                residualThreshold=1e-10,
            )
            path.append(joint_positions)

        return path, pre_grasp_transformation

    def close_gripper_and_record(self, num_steps=160):
        for i in range(num_steps):
            self.close_gripper()
            self.pybullet_client.stepSimulation()

            observations = self.get_observations()
            # save the observations in the correct format zfill(5)
            if i % 20 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

    def open_gripper_and_record(self, num_steps=160):
        for i in range(num_steps):
            self.open_gripper()
            self.pybullet_client.stepSimulation()

            observations = self.get_observations()
            # save the observations
            if i % 20 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

    def follow_trajectory_and_record(
        self,
        path,
        gripper_pos,
        use_current_iters=True,
        gripper_velocity=2,
        threshold=1e-2,
    ):
        # Note: This function also saves observations
        k = 0
        loop_iters = 0
        current_iters = 0
        while k < len(path):
            error = 0

            for j in range(0, self.num_dofs()):
                self.pybullet_client.setJointMotorControl2(
                    self.splatsim_robot.sim_id,
                    j + 1, # Assuming the first joint index is 1 (0 is often a fixed joint), adjust if necessary
                    p.POSITION_CONTROL,
                    targetPosition=path[k][j],
                    force=250,
                    maxVelocity=0.2,
                )

            # get current joint positions
            joint_states = []
            for i in range(0, self.num_dofs()):
                joint_states.append(
                    self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i + 1)[0]
                )

            error = np.linalg.norm(np.array(joint_states) - path[k][:self.num_dofs()])

            self.move_gripper((1 - gripper_pos) * 0.084, velocity=gripper_velocity)

            if error < threshold:
                k += 1
                current_iters = 0

            current_iters += 1

            if use_current_iters:
                if current_iters > 200:
                    k += 1
                    current_iters = 0

            self.current_gripper_action = gripper_pos
            if loop_iters % 50 == 0 and not self.skip_recording_first:
                self.trajectory_length += 1
                # get observations
                observations = self.get_observations()
                # save the observations with trajectory length as pickle file in format 0000x.pkl
                with open(
                    os.path.join(
                        self.path,
                        str(self.trajectory_count).zfill(3),
                        str(self.trajectory_length).zfill(5) + ".pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(observations, f)

            if loop_iters > 10000:
                break

            self.pybullet_client.stepSimulation()

            loop_iters += 1

        if not self.skip_recording_first:

            self.trajectory_length += 1
            # get observations
            observations = self.get_observations()
            # save the observations with trajectory length as pickle file in format 0000x.pkl
            with open(
                os.path.join(
                    self.path,
                    str(self.trajectory_count).zfill(3),
                    str(self.trajectory_length).zfill(5) + ".pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(observations, f)

    # =========================================================================
    # Gym Environment Interface
    # =========================================================================

    @property
    def action_space(self) -> spaces.Box:
        """Define the action space.

        Returns:
            Box space for 7 DOF actions (6 joints + gripper)
        """
        # Joint limits - can be refined from URDF if needed
        low = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0.0], dtype=np.float32)
        high = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        """Define the observation space.

        Returns:
            Dict space with agent_pos (state) and pixels dict (images)
            in LeRobot-compatible format.
        """
        # Build pixels dict space for each camera+mode combination
        pixels_dict = {}
        for camera_name in self.camera_names:
            for mode in self.image_resize_modes:
                key = f"{camera_name}_{mode.value}"
                pixels_dict[key] = spaces.Box(
                    low=0, high=255,
                    shape=(224, 224, 3), dtype=np.uint8
                )

        obs_dict = {
            "agent_pos": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_dofs() + 1,), dtype=np.float32
            ),
            "pixels": spaces.Dict(pixels_dict)
        }
        # Privileged world state (object coords) for oracle/state-only policies,
        # exposed as a separate observation.environment_state feature. Only
        # declared when this env actually emits it (env_state_dim > 0).
        if self.env_state_dim() > 0:
            obs_dict["environment_state"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.env_state_dim(),), dtype=np.float32
            )
        return spaces.Dict(obs_dict)

    def _physics_step(self, action: np.ndarray) -> None:
        """Apply action and advance the physics simulation by one control step."""
        assert self._episode_started, "Must call reset() before step()"

        self.command_joint_state(self.splatsim_robot, action)

        for _ in range(self._physics_steps_per_action):
            self.pybullet_client.stepSimulation()

        self._step_count += 1

    def step(self, action: np.ndarray):
        """Single env step returning the **raw** observation dict (gym signature).

        Single source of truth for what a step does on a local server: apply
        action, fetch raw observations, compute reward/termination, build info.
        ``SplatSimGymEnv.step`` wraps this and applies ``_to_gym_obs`` for the
        gym observation space; the recording wrapper
        (``TeleopRecordingWrapper._step_raw``) calls this directly to get the
        unconverted dict — which has ``{cam}_{mode}`` keys for every resize
        mode. The signature matches ``_ZMQBackend.step`` so callers don't
        branch on backend type.
        """
        self._physics_step(action)
        raw_obs = self.get_observations()
        metrics = self.check_metrics()
        is_success = metrics.get("is_success", False)
        reward = self.compute_reward_from_metrics(metrics)
        terminated = self.check_terminated_from_metrics(metrics)
        truncated = self._step_count >= self._max_episode_steps
        info = {
            k: v.item() if isinstance(v, np.generic) else v
            for k, v in {"is_success": is_success, "step_count": self._step_count, **metrics}.items()
        }
        return raw_obs, reward, terminated, truncated, info

    def check_truncated(self) -> bool:
        """Check if episode should be truncated due to time limit.

        Returns:
            True if step limit exceeded
        """
        return self._step_count >= self._max_episode_steps

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Subclasses must implement this method with task-specific reset logic.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict

        Returns:
            observation: Initial observation dict
            info: Dict with initial info

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement reset()")

    def compute_reward(self) -> float:
        """Compute the reward for the current state.

        Subclasses must implement this method with task-specific reward logic.

        Returns:
            Float reward value

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")

    def check_success(self) -> bool:
        """Check if the current episode achieved success.

        Subclasses must implement this method with task-specific success criteria.

        Returns:
            True if success condition is met

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement check_success()")
    
    def check_metrics(self) -> Dict[str, Any]:
        """Check additional metrics for the current episode.

        Subclasses can implement this method to return task-specific metrics.

        Returns:
            Dict of metric names to values
        """
        return {}

    def check_terminated(self) -> bool:
        """Check if episode should terminate due to success or failure.

        Subclasses must implement this method with task-specific termination conditions.

        Returns:
            True if episode should terminate

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement check_terminated()")

    def compute_reward_from_metrics(self, metrics: Dict[str, Any]) -> float:
        return self.compute_reward()

    def check_terminated_from_metrics(self, metrics: Dict[str, Any]) -> bool:
        return self.check_terminated()

    # =========================================================================
    # Mode-based GUI Controls
    # =========================================================================

    def _setup_interactive_gui(self):
        """Create GUI controls using Tkinter.

        This launches the SplatSimGui window which includes:
        - Mode switching buttons (Interactive / Trajectory Gen)
        - Debug mode dropdown
        - Trajectory generation parameters
        - Start/Stop trajectory generation buttons

        In fully-headless mode (``self._headless`` and NOT
        ``self._show_control_gui``), the SplatSimGui object is still constructed
        (so every ``self._splatsim_gui.X`` reference elsewhere resolves), but
        ``.start()`` is NOT called — no Tkinter root, no thread, no "SplatSim
        Controls" window. SplatSimGui's read methods fall back to
        ``dict.get(key, default)`` and its write methods iterate empty/none Tk
        state, so calls from the main loop become safe no-ops.

        With ``self._show_control_gui`` set alongside headless, ``.start()`` IS
        called: pybullet stays DIRECT (no 3D window, EGL rendering) but the
        Tkinter control panel launches so a user can drive modes/config/Start
        interactively. Needs a display for Tkinter.
        """
        # Initialize the Tkinter GUI (runs in separate thread)
        config = self.trajectory_generator.config

        # TODO add some env configs to the gui on startup like self._eval_benchmark_repo_id

        initial_mode = self.serve_mode.value  # Use the enum's string value
        self._splatsim_gui: SplatSimGui = SplatSimGui(
            config,
            initial_mode,
            debug_mode_enum=self.DEBUG_MODES,
            initial_debug_mode=self.debug_mode,
            # Seed the "Render mode" dropdown from the launch --render_mode so
            # GUI state matches server state. `available_render_modes` hides
            # options this env can't do (SPLAT needs loaded splat assets).
            initial_render_mode=self._initial_render_mode,
            available_render_modes=self._available_render_modes(),
            # Seed the "Splat shadows" checkbox from the launch --splat_shadows
            # so GUI state matches server state; polled alongside render mode.
            initial_splat_shadows=self.splat_shadows,
            # Called by the Traj Gen panel after "Import Config" so env-owned
            # fields (task goal, skip pairs, DOF-checked q_*) survive a config
            # file exported from a different env. Reads only env class state —
            # safe from the GUI thread.
            traj_env_reassert_fn=self.reassert_env_traj_config_fields,
        )
        if self._headless and not self._show_control_gui:
            # Fully headless (batch / display-less): skip the Tkinter mainloop
            # entirely. The GUI object's initialization is pure Python (no Tk
            # widgets created until _build_ui runs inside the thread), so it's
            # safe to leave it in this "constructed but not started" state.
            return
        # Start the Tkinter control panel. This runs whenever we're NOT fully
        # headless — including the headless + show_control_gui combo, where
        # pybullet has no 3D window but the panel still drives modes/config.
        self._splatsim_gui.start()

    def _check_debug_mode(self):
        """Check if debug mode has changed in the GUI and update self.debug_mode."""
        new_debug_mode = self._splatsim_gui.get_debug_mode()
        if new_debug_mode is None:
            return  # GUI not yet initialized
        if new_debug_mode != self.debug_mode:
            print(f"[GUI] Debug mode changed: {self.debug_mode.value} -> {new_debug_mode.value}")
            self.debug_mode = new_debug_mode

    def _check_camera_rendering_toggle(self):
        """Sync the image-observation source with the GUI's "Render mode"
        dropdown (Splat / PyBullet camera / None).

        Polled each serve-loop tick (like _check_debug_mode). On change, this
        updates BOTH the runtime gates (via _apply_render_mode) AND the
        launch-time preference (_initial_render_mode / _render_from_splat_default)
        consulted by _render_and_save_episode's re-enable and the LeRobot dataset
        schema guard — so a GUI change behaves exactly like relaunching with a
        different --render_mode.
        NOTE: change the mode BEFORE starting trajectory generation with a
        lerobot repo_id — the dataset schema (image vs state-only) is fixed at
        dataset creation, and the schema guard rejects a mid-run mismatch.
        """
        if self._splatsim_gui is None:
            return
        # Sync the "Splat shadows" checkbox (same poll cadence as the render
        # mode). Track whether it changed so we can refresh the thumbnails
        # below — but only when splat imagery is what's on screen.
        shadows_changed = False
        new_shadows = self._splatsim_gui.get_splat_shadows()
        if new_shadows is not None and bool(new_shadows) != self.splat_shadows:
            print(f"[GUI] Splat shadows: {self.splat_shadows} -> {bool(new_shadows)}")
            self.splat_shadows = bool(new_shadows)
            shadows_changed = True
        new_mode = self._splatsim_gui.get_render_mode()
        if new_mode is None:
            return  # GUI not yet initialized
        if new_mode != self.render_mode:
            print(f"[GUI] Render mode changed: {self.render_mode.value} -> {new_mode.value}")
            self._apply_render_mode(new_mode)
            self._initial_render_mode = new_mode
            self._render_from_splat_default = (new_mode == RenderMode.SPLAT)
            if new_mode == RenderMode.NONE:
                # Clear the (now-stale) camera thumbnails; display_observations
                # is never called while rendering is off, so nothing else would
                # refresh or remove them.
                self._splatsim_gui.clear_camera_images()
            else:
                # Render the CURRENT sim state in the new mode and push it to the
                # GUI thumbnails right away, so switching to Splat/PyBullet updates
                # the images immediately instead of waiting for the next
                # observation request. get_observations() renders per the active
                # mode and calls display_observations() internally. (Also covers
                # a shadows toggle landing on the same tick — one render, not two.)
                try:
                    self.get_observations()
                except Exception as e:
                    print(f"[GUI] Render on switch to {new_mode.value} failed: {e}")
        elif shadows_changed and self.render_mode == RenderMode.SPLAT:
            # Shadows toggled while splat imagery is displayed: re-render the
            # current sim state immediately (same pattern as the mode switch
            # above) so the thumbnails reflect the checkbox without waiting for
            # the next observation request. Other modes need no refresh — the
            # shadow composite only exists on the splat path.
            try:
                self.get_observations()
            except Exception as e:
                print(f"[GUI] Render on splat-shadows toggle failed: {e}")

    def shutdown(self):
        """Clean up resources.

        Another name for self.stop(). This is to fit with the gello api
        """
        self.stop()
