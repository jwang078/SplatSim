import enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


class ImageResizeMode(enum.Enum):
    LETTERBOX = "letterbox"
    STRETCH = "stretch"


# `PathSelectionStrategy` is owned by the shared planner
# (`splatsim.utils.rrt_to_goal`, the canonical RRTToGoalPlanner used by BOTH
# SplatSim trajectory-gen and the LeRobot DAgger/SA intervention side) and
# re-exported here so configs and the trajectory generator use ONE enum object
# — identity-equal to what `RRTToGoalPlanner` compares against internally.
# Values: ee_arc_length / joint_arc_length / joint_velocity_match /
# min_pair_clearance / camera_scoring. CAMERA_SCORING is SplatSim's wrist-camera
# view-angle scorer (only meaningful when a wrist camera link is configured).
from splatsim.utils.rrt_to_goal import (  # noqa: E402,F401
    IkGoalSelectionStrategy,
    PathSelectionStrategy,
)


@dataclass
class SplatSimModeConfig:
    """Base configuration for a SplatSim mode."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InteractiveModeConfig(SplatSimModeConfig):
    """Configuration for interactive mode."""
    pass


@dataclass
class TrajectoryGenModeConfig(SplatSimModeConfig):
    """Configuration for trajectory generation mode."""

    num_base_trajectories: int = 100
    obstacles_per_base_trajectory: int = 0
    paths_per_obstacle: int = 0
    min_obstacles: int = 1
    max_obstacles: int = 3
    max_fails: int = 2
    max_obstacle_fails_per_base_traj: int = 20
    time_per_traj: float = 6.0
    robot_update_rate: int = 30
    rrt_vis_fps: int = 10
    use_obstacles: bool = True
    q_start: Optional[List[float] | np.ndarray] = None
    q_goal: Optional[List[float] | np.ndarray] = None # ex: 7-dof-joint robot configuration goal
    ee_pos_start: Optional[List[float] | np.ndarray] = None   # [x, y, z] end-effector position start
    ee_quat_start: Optional[List[float] | np.ndarray] = None  # [x, y, z, w] end-effector quaternion start
    ee_pos_goal: Optional[List[float] | np.ndarray] = None    # [x, y, z] end-effector position goal
    ee_quat_goal: Optional[List[float] | np.ndarray] = None   # [x, y, z, w] end-effector quaternion goal
    # Canonical 6-DOF joint config to prefer as the goal IK solution. If set and
    # collision-free for the current scene, it is used as the primary q_goal so
    # demos converge to a shared joint configuration. If it's in collision (e.g.
    # a randomized obstacle blocks it), the planner falls back to random-seed IK.
    q_goal_bias: Optional[List[float] | np.ndarray] = None
    num_ik_candidates: int = 32                   # number of IK solutions to try for EE goals
    cuboids_fn: Optional[str] = None
    render_images: bool = False
    save_base_trajectory: bool = True
    disable_camera_scoring_for_rrt: bool = False
    num_path_candidates: int = 5
    max_path_attempts: int = 15
    # Radians to perturb the RRT start/goal endpoints between path candidates so
    # BiRRT explores distinct branches (path_selection then picks the best of
    # them). Larger = more diverse candidates. Forwarded to
    # RRTToGoalPlanner(path_perturbation_scale=).
    path_perturbation_scale: float = 0.05
    # RRT planner-time collision clearances, in meters. Reject any path bringing
    # the robot within this margin of an obstacle / of itself — gives execution
    # drift margin so a planning-time near-miss doesn't become a real collision
    # (wedge) under the position controller's tracking lag. Defaults match the
    # DAgger/SA intervention side. Higher = safer but fewer plannable paths in
    # tight scenes. Forwarded to `get_path` / `check_links_in_collision`.
    # NOTE: self_collision_clearance relies on `self_collision_skip_pairs` to
    # exclude structurally-close URDF pairs, else every pose reads as colliding.
    obstacle_clearance: float = 0.02
    self_collision_clearance: float = 0.005
    # Which scoring strategy picks the winning path among the RRT candidates
    # generated per IK goal. Default `MIN_PAIR_CLEARANCE` — picks the path whose
    # tightest non-adjacent link-pair gap is LARGEST, avoiding pretzeled / near-
    # self-collision poses that wedge during execution (matches the DAgger/SA
    # intervention side). `CAMERA_SCORING` picks the path whose wrist-camera view
    # best satisfies `k_exp`/`k_sig`/`threshold` against the target EE pose;
    # `EE_ARC_LENGTH` / `JOINT_ARC_LENGTH` minimize cartesian / joint path
    # length. See `PathSelectionStrategy` for full descriptions. Stored as the
    # enum value string; parsed back to enum at use.
    #
    # `disable_camera_scoring_for_rrt` below is DEPRECATED — only consulted when
    # `path_selection == CAMERA_SCORING`, forcing it to `EE_ARC_LENGTH`.
    path_selection: str = PathSelectionStrategy.MIN_PAIR_CLEARANCE.value
    # How to pick the IK GOAL among the candidates for the target EE pose,
    # BEFORE running RRT (orthogonal to `path_selection`, which picks the PATH).
    # Stored as the enum value string; parsed back at use.
    #   * "joint_distance" (default) — pick the IK solution NEAREST q_start
    #     (minimal joint reconfiguration). Avoids the far wrist-flipped IK
    #     branch that requires a large wrist sweep and self-collides
    #     (wrist_3 vs forearm) during execution. Same strategy the DAgger/SA
    #     intervention side uses. `path_selection` still picks among the paths
    #     TO that nearest goal.
    #   * "none" — no IK-goal pre-selection: try ALL IK candidates and let
    #     `path_selection` (e.g. camera_scoring) pick the winner across them
    #     (historical multi-candidate behavior).
    ik_goal_selection: str = IkGoalSelectionStrategy.JOINT_DISTANCE.value
    # Ruckig segmentation. True: split the trajectory at sharp (>45°) corners
    # with a forced zero-velocity STOP at each — safer but produces bursty
    # "start-stop" motion. False (default for trajectory-gen): a single ruckig
    # pass with intermediate positions → continuous motion through corners.
    # Forwarded to RRTToGoalPlanner(segment_at_sharp_corners=).
    segment_at_sharp_corners: bool = False
    # Non-adjacent link pairs to EXCLUDE from self-collision checks. Use for
    # URDF link pairs that are structurally close at every reachable joint
    # config (e.g. UR's base_link(0) vs upper_arm_link(2), ~4 mm apart due
    # to shoulder bracket geometry) — without skipping them, any non-zero
    # `self_collision_clearance` flags every valid pose as a collision.
    # Format: list of [link_a, link_b] pairs. Order doesn't matter
    # ((a,b) == (b,a)). None = no skips. Subclass envs typically set this
    # in their `_get_default_trajectory_gen_config()` to declare their
    # URDF's known-close pairs.
    self_collision_skip_pairs: Optional[List[List[int]]] = None
    # When True, the IK-candidate collision filter in `_solve_ik` skips
    # gripper-finger ⟷ obstacle pairs (auto-detected by URDF link name —
    # any link whose name contains "finger" or "knuckle"). Motivation:
    # at grasp goals the fingers are intentionally within mm of the
    # target object, and the global `obstacle_clearance` (2 cm) rejects
    # nearby IK branches — leaving only far-away IK branches and forcing
    # `_resolve_ee_pose_to_q_candidates` to give up because "all
    # candidates within N mm of small_engine". The arm-link clearance
    # check still runs on every other link, and the full RRT path
    # search / per-tick / future-chunk shield checks downstream are
    # UNCHANGED — only the IK candidate filter is relaxed.
    # Mirrors `SharedAutonomyConfig.rrt_ik_skip_gripper_obstacle_pairs`
    # on the LeRobot side so SplatSim's env-side reset-time IK feasibility
    # check uses the same relaxation the runtime RRT does.
    ik_skip_gripper_obstacle_pairs: bool = True
    k_exp: float = 5.0
    k_sig: float = 15.0
    threshold: float = 0.4
    save_zarr: bool = False
    lerobot_repo_id: str = ""
    push_to_hub: bool = True
    render_letterbox: bool = True
    render_stretch: bool = True
    debug_visualize: bool = False  # Visualize q_start, q_goal, and trajectory in PyBullet GUI
    verbose: bool = True

    def __post_init__(self):
        has_ee_start = self.ee_pos_start is not None or self.ee_quat_start is not None
        if has_ee_start and self.q_start is not None:
            raise ValueError(
                "TrajectoryGenModeConfig: Cannot specify both q_start and ee_pos_start/ee_quat_start. "
                "Set q_start=None when using end-effector pose starts."
            )
        if self.ee_pos_start is not None and len(self.ee_pos_start) != 3:
            raise ValueError(f"ee_pos_start must be length 3 (x, y, z), got {len(self.ee_pos_start)}")
        if self.ee_quat_start is not None and len(self.ee_quat_start) != 4:
            raise ValueError(f"ee_quat_start must be length 4 (x, y, z, w), got {len(self.ee_quat_start)}")

        has_ee_goal = self.ee_pos_goal is not None or self.ee_quat_goal is not None
        if has_ee_goal and self.q_goal is not None:
            raise ValueError(
                "TrajectoryGenModeConfig: Cannot specify both q_goal and ee_pos_goal/ee_quat_goal. "
                "Set q_goal=None when using end-effector pose goals."
            )
        if self.ee_pos_goal is not None and len(self.ee_pos_goal) != 3:
            raise ValueError(f"ee_pos_goal must be length 3 (x, y, z), got {len(self.ee_pos_goal)}")
        if self.ee_quat_goal is not None and len(self.ee_quat_goal) != 4:
            raise ValueError(f"ee_quat_goal must be length 4 (x, y, z, w), got {len(self.ee_quat_goal)}")


@dataclass
class EvalBenchmarkModeConfig(SplatSimModeConfig):
    """Configuration for eval benchmark mode."""
    lerobot_repo_id: str = ""
    episode_subset_str: str = ""  # Comma-separated episode indices, e.g. "3,8,23" or "[3,8,23]". Blank = all.
