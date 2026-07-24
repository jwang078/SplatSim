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
from splatsim.utils.rrt_path_utils import _COLLISION_CLEARANCE
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

    def _control_max_velocity(self) -> float:
        """Resolve the POSITION_CONTROL maxVelocity: an explicit
        `CONTROL_MAX_VELOCITY` float, or — when it's None — the trajectory
        generator's planned `max_joint_vel`, so the servo tracks the plan
        without overshooting it. Falls back to 3.14 before the generator exists
        (construction-time teleports snap via resetJointState, so the cap is moot
        there anyway)."""
        if self.CONTROL_MAX_VELOCITY is not None:
            return self.CONTROL_MAX_VELOCITY
        tg = getattr(self, "trajectory_generator", None)
        if tg is not None:
            return tg.config.max_joint_vel
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
    ORACLE_RECORD_ENV_STATE: ClassVar[bool] = True
    ORACLE_STATE_COORD_INDICES: ClassVar[tuple] = (0, 1, 2)
    ORACLE_STATE_INCLUDE_QUAT: ClassVar[bool] = False
    ORACLE_OBJECT_NAMES: ClassVar[Optional[List[str]]] = None

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
        show_control_gui: bool = False,
        sync_physics_to_client: bool = False,
        physics_substeps_per_command: int = 8,
    ):
        self._splatsim_gui = None
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
        # Headless mode: connect pybullet in DIRECT (no GUI) for fast
        # physics-only use cases like trajectory replay + collision filtering.
        # Splat rendering and any GUI-dependent features are unavailable;
        # callers needing those should leave this False.
        self._headless = headless
        # Decouple the pybullet 3D WINDOW from the Tkinter CONTROL panel: when set
        # alongside headless, pybullet still connects DIRECT (no OpenGL window,
        # EGL GPU rendering, no ~30 Hz render-loop throttle) BUT the "SplatSim
        # Controls" window is still launched — so you can pick modes / tune the
        # trajectory config / press Start from the panel while rendering fast.
        # Requires a display for Tkinter (a workstation, not a display-less node);
        # no-op unless headless (a GUI connection already shows the panel).
        self._show_control_gui = show_control_gui
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
        # without a display and physics-only operations (collision queries,
        # joint state teleport) are unaffected. Splat rendering won't work in
        # this mode.
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
                override_xyz=(0, 1.20, 0.61),
                override_rpy=(np.pi/2 - 15*np.pi/180, np.pi, 0),
                override_dist_inc=0.55
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

    def _oracle_per_object_dim(self) -> int:
        """Number of scalars recorded per object: position coords + optional quat."""
        return len(self.ORACLE_STATE_COORD_INDICES) + (4 if self.ORACLE_STATE_INCLUDE_QUAT else 0)

    def env_state_dim(self) -> int:
        """Width of observation.environment_state (a FeatureType.ENV feature):
        privileged object poses for oracle/state-based policies. Computed from the
        recorded objects × per-object dim, so it always matches
        oracle_environment_state(). 0 → no environment_state feature."""
        return len(self._oracle_object_names()) * self._oracle_per_object_dim()

    def _get_default_trajectory_gen_config(self) -> TrajectoryGenModeConfig:
        # Use all the default values
        return TrajectoryGenModeConfig()

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

    def teleport_joint_state(
        self, splatsim_obj: SplatSimObject, joint_state: Tuple[float, ...]
    ) -> None:
        """Set the joint states of an articulated object in the simulation and hold position."""
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
                force=self.CONTROL_FORCE,
                maxVelocity=self._control_max_velocity(),
            )
        # step_physics=False keeps the teleport ATOMIC. Without this the
        # command's post-set physics-step loop (added by the sync-to-client
        # feature) would step 8× immediately after the resetJointState above,
        # letting the position controller pull the arm off the teleport
        # target — the exact opposite of what a teleport should do.
        self.command_joint_state(splatsim_obj, np.array(joint_state), step_physics=False)

    def command_joint_state(
        self,
        splatsim_obj: SplatSimObject,
        joint_state: np.ndarray,
        step_physics: bool = True,
    ) -> None:
        # Only drive the arm DOFs (joints 1..num_dofs). Anything beyond that is
        # the fixed ee/gripper-mount joints and the gripper's mimic joints.
        # Driving the gripper's mimic CHILD joints here with independent
        # POSITION_CONTROL (force=150) overpowers the JOINT_GEAR mimic
        # constraints (force=10) that move_gripper relies on, freezing the
        # gripper — so the gripper is actuated ONLY via move_gripper below.
        # (joint_state may carry 18 entries at init; the extra trailing values
        # must not be applied as per-joint position targets on gripper links.)
        for i in range(0, min(len(joint_state), self.num_dofs())):
            self.pybullet_client.setJointMotorControl2(
                splatsim_obj.sim_id,
                i + 1, # Assuming the first joint index is 1 (0 is often a fixed joint), adjust if necessary
                p.POSITION_CONTROL,
                targetPosition=joint_state[i],
                # Set a more realistic force for the robot
                force=self.CONTROL_FORCE,
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

    def get_pybullet_debug_camera_as_splat_camera(self) -> SplatSimCamera:
        """Convert PyBullet's debug camera to a Camera object for Gaussian splatting."""
        # This function is the inverse of this post: https://stackoverflow.com/a/75355212

        # Get PyBullet camera info
        camera_info = p.getDebugVisualizerCamera()
        # Pybullet view matrix is major-column order
        view_matrix = np.array(camera_info[2]).reshape(4, 4).T

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
            if self.debug_mode != DebugModes.OFF:
                camera = self.get_pybullet_debug_camera_as_splat_camera()
            else:
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

        # save the image (always as numpy array)
        return rendering

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
        ee_link = self._get_ee_link_index()
        dummy_ee_pos, dummy_ee_quat = (
            self.pybullet_client.getLinkState(self.splatsim_robot.sim_id, ee_link)[0],
            self.pybullet_client.getLinkState(self.splatsim_robot.sim_id, ee_link)[1],
        )
        return dummy_ee_pos, dummy_ee_quat

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

        return {
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
        }

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

    def _render_pybullet_camera(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render one camera view with PyBullet's getCameraImage.

        Fast, splat-free image source usable by any env. For a wrist camera key
        (see `_is_wrist_camera`) it renders the WRIST-MOUNTED view derived from
        `get_wrist_camera()`; for a base/third-person key it mirrors `base_camera`
        when a splat base camera exists (matching the splat view's pose + FoV +
        aspect), else the fixed PYBULLET_CAMERA_* pose. Returns a CHW float32 RGB
        image in [0, 1] at the per-view render resolution (resize_image then maps
        it to 224). Uses the GPU OpenGL renderer under a GUI connection, the CPU
        tiny renderer when headless (DIRECT)."""
        view_proj = None
        if self._is_wrist_camera(camera_name):
            # Wrist view from the live wrist-camera pose + FoV (a fisheye lens
            # renders as its RECTIFIED pinhole equivalent, matching the splat).
            wrist_cam = self.get_wrist_camera()
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
        view, proj, W, H = view_proj

        # Hardware GL whenever we have a GL context — a GUI connection, or a
        # headless DIRECT client with the EGL plugin loaded. Only fall back to the
        # CPU software renderer when headless AND EGL was unavailable.
        use_hardware_gl = (not self._headless) or (getattr(self, "_egl_plugin_id", None) is not None)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if use_hardware_gl else p.ER_TINY_RENDERER
        _, _, rgba, _, _ = self.pybullet_client.getCameraImage(
            W, H, view, proj,
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (H, W, 4))[:, :, :3]
        # HWC uint8 [0,255] -> CHW float32 [0,1] (the format resize_image expects).
        return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))

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
            # third-person view.
            for camera_name in self.camera_names:
                raw_img = self._render_pybullet_camera(camera_name)
                for mode in self.image_resize_modes:
                    key = f"{camera_name}_{mode.value}"
                    observations[key] = resize_image(raw_img, (224, 224), mode=mode)
            self.display_observations(observations)
        elif (
            render_images and self.do_render_from_splat
            and self.base_camera is not None  # splat render needs a base camera
            and len(self.camera_names) > 0
        ):
            # Capture link state snapshot for synchronized rendering
            cached_link_states = get_curr_link_states(
                self.splatsim_robot.sim_id,
                use_link_centers=True
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
    
    def restore_episode_scenario(self, episode_index: int) -> None:
        """Restore the environment to the exact state recorded at the start of a LeRobot episode.

        Reads the splatsim_robot_config, splatsim_object_configs, and splatsim_background_config
        fields saved in the episode metadata and applies the recorded initial_position,
        initial_quat, and initial_scale to each matching live object.

        For the robot, the saved initial_joint_positions from the articulation_config are
        teleported directly. For scene objects, the recorded pose is set via PyBullet and the
        scale is restored by adjusting the Gaussian splat and reloading the URDF.

        Args:
            episode_index: The episode whose saved scenario should be restored.
        """
        if self._lerobot_saver is None:
            raise RuntimeError("No LeRobot dataset loaded. Call _init_lerobot_dataset() first.")

        ep = self._lerobot_saver.meta.episodes[episode_index]

        def _parse_ep_field(val):
            """Parquet stores these as JSON strings; parse back to dict/list if needed."""
            if isinstance(val, str):
                return json.loads(val)
            return val

        # Restore robot joint positions
        robot_cfg = _parse_ep_field(ep.get("splatsim_robot_config"))
        if robot_cfg is not None:
            initial_joints = robot_cfg["articulation_config"]["initial_joint_positions"]
            self.teleport_joint_state(self.splatsim_robot, initial_joints)
            # self.command_joint_state(self.splatsim_robot, np.concatenate([initial_joints[:self.num_dofs()], [0]]))

        # Build a name→object lookup for the live scene objects
        obj_by_name = {obj.config.name: obj for obj in self.splatsim_objects}

        # Restore each non-robot, non-background object
        object_configs = _parse_ep_field(ep.get("splatsim_object_configs")) or []

        if robot_cfg is None and len(object_configs) == 0:
            raise ValueError(f"No robot config or object configs found for episode {episode_index}")
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

    def _get_splatsim_episode_metadata(self) -> dict:
        """Build the splatsim-specific episode metadata dict for LeRobot save_episode().

        Contains JSON-serialisable configs for the robot, background, and all
        non-robot/non-background objects currently in the scene.
        """
        def _config_to_dict(cfg):
            d = asdict(cfg)
            return json.loads(json.dumps(d, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))

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

        # start the zmq server only after benchmark state is ready
        self._zmq_server_thread.start()

        print("Ready to serve.")

        try:
            while True:
                # Let the GUI handle all mode/button transitions
                self._splatsim_gui.process_mode_transitions()

                # Reset env button — available in all modes
                if self._splatsim_gui.check_button(SplatSimGui.BTN_RESET_ENV):
                    print("[GUI] Reset Env pressed — resetting environment.")
                    self._handle_reset()

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
                    time.sleep(1 / 240)
                elif current_mode == self.SERVE_MODES.GENERATE_TRAJECTORIES_IDLE:
                    # Idle mode - just step simulation while user configures settings
                    self.pybullet_client.stepSimulation()
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
                    self._generate_and_render_one_episode()

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
                    time.sleep(1 / 240)
                elif current_mode == self.SERVE_MODES.EVAL_BENCHMARK:
                    # See INTERACTIVE branch above for sync-to-client rationale.
                    if self._sync_physics_to_client:
                        self._consume_sync_step_request()
                    else:
                        self.pybullet_client.stepSimulation()
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
                            if episode_id in self._eval_benchmark_subset:
                                subset_pos = self._eval_benchmark_subset.index(episode_id)
                                if subset_pos != self._eval_benchmark_episode_index:
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
        # Control the mimic gripper joint(s)
        self.pybullet_client.setJointMotorControl2(
            self.splatsim_robot.sim_id,
            self.mimic_parent_id,
            self.pybullet_client.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
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
                # mode and calls display_observations() internally.
                try:
                    self.get_observations()
                except Exception as e:
                    print(f"[GUI] Render on switch to {new_mode.value} failed: {e}")

    def shutdown(self):
        """Clean up resources.

        Another name for self.stop(). This is to fit with the gello api
        """
        self.stop()
