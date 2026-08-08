import logging
import numpy as np
import zarr
import os
import re
import json
from typing import Optional, Tuple, List
from pybullet_planning import create_box, set_pose, Pose, RED, BLUE

from splatsim.configs import TrajectoryGenModeConfig
from splatsim.configs.mode_config import PathSelectionStrategy
from splatsim.utils import rrt_path_utils
from splatsim.utils.rrt_to_goal import RRTToGoalPlanner, RRTPlanningError
from splatsim.configs.env_config import SplatSimObject

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Helper class for RRT-based trajectory generation."""

    RRT_PERTURBATION_SCALE = 0.001  # Radians to perturb start/goal for path diversity. Setting numpy random seeds didn't work

    def __init__(
        self,
        pybullet_client,
        robot_id: int,
        joint_indices: list,
        env_config_name: str,
        get_ee_link_fn,  # Callback to get EE link index
        splatsim_objects: List[SplatSimObject],
        wrist_camera_link_name: Optional[str] = None,
        trajectory_gen_config: Optional[TrajectoryGenModeConfig] = None,
        pb_client_id: int = 0,
        soft_cost_payload: Optional[dict] = None,
    ):
        """
        Initialize trajectory generator.

        Args:
            pybullet_client: PyBullet client instance (the pybullet module)
            pb_client_id: Integer pybullet client id (used as physicsClientId=
                by the shared RRTToGoalPlanner). Distinct from pybullet_client,
                which is the module.
            robot_id: Robot body ID in PyBullet
            joint_indices: List of movable joint indices
            env_config_name: Environment name for output directory
            get_ee_link_fn: Function that returns EE link index
            splatsim_objects: List of SplatSimObject instances
            wrist_camera_link_name: Name of wrist camera link for camera-aware scoring
            trajectory_gen_config: Trajectory generation configuration (uses defaults if None)
            soft_cost_payload: Optional env_config-style soft-cost dict
                (EnvConfig.soft_cost). The generator's planner never calls
                load_obstacles (it plans against the env's live world), so
                the field must be attached explicitly or cost-aware scoring
                silently stays off for in-env trajectory generation.
        """
        self.pb = pybullet_client
        self._pb_client_id = pb_client_id
        self.robot_id = robot_id
        self.joint_indices = joint_indices

        if trajectory_gen_config is None:
            trajectory_gen_config = TrajectoryGenModeConfig()
        self.config = trajectory_gen_config
        self.env_config_name = env_config_name
        self.get_ee_link_fn = get_ee_link_fn
        self.splatsim_objects = splatsim_objects
        self.soft_cost_payload = soft_cost_payload

        # Store and resolve camera link for scoring
        self.wrist_camera_link_name = wrist_camera_link_name
        self.camera_link_index = None

        if self.wrist_camera_link_name:
            num_joints = self.pb.getNumJoints(self.robot_id)
            for i in range(num_joints):
                info = self.pb.getJointInfo(self.robot_id, i)
                link_name = info[12].decode("utf-8")
                if link_name == self.wrist_camera_link_name:
                    self.camera_link_index = i
                    break

            if self.camera_link_index is None:
                print(f"Warning: Camera link '{self.wrist_camera_link_name}' not found. Camera scoring disabled.")

        # Load joint limits
        self.lower_limits, self.upper_limits = rrt_path_utils.get_joint_limits(
            robot_id, joint_indices
        )

        # Canonical shared planner. TrajectoryGenerator now DELEGATES all
        # RRT/IK/path-scoring/parametrization work to RRTToGoalPlanner (the same planner
        # the LeRobot DAgger/SA intervention side uses), so SplatSim
        # trajectory-gen and intervention share ONE implementation. Obstacles
        # are pushed into the planner via `_sync_planner_obstacles()` before
        # each plan (we do NOT call planner.load_obstacles(), which would
        # delete/recreate bodies in this shared sim client).
        #
        # Built LAZILY (on first `_sync_planner_obstacles`) rather than here:
        # the constructor needs `self.get_ee_link_fn()`, which reads
        # `self.wrist_camera` — and the parent robot server sets that up AFTER
        # constructing this generator, so calling it now would AttributeError.
        self._planner = None

        # Optional pre-validated base trajectory (T, num_dofs), set by the env's
        # reset via `_check_scenario_solvable` after it already planned a path to
        # prove the scenario is solvable. `generate_trajectory_batch` consumes it
        # ONCE instead of re-planning the base path (obstacle variations still
        # plan fresh). None disables the shortcut.
        self._cached_base_traj = None

        # Static obstacles from environment (walls, table, etc.)
        self.loaded_obstacle_ids = []

        # Load cuboid obstacles if specified
        if self.config.use_obstacles and self.config.cuboids_fn:
            self.loaded_obstacle_ids = self._load_cuboid_obstacles(self.config.cuboids_fn)

        # Zarr storage (initialized lazily when start_generation() is called)
        self.trajectory_count = 0
        self.zarr_root = None
        self.scenarios_group = None

    def get_obstacle_ids(self) -> List[int]:
        return self.loaded_obstacle_ids + [
            obj.sim_id for obj in self.splatsim_objects
            if obj.sim_id is not None
            and obj.config.name != "robot"
        ]

    def get_obstacle_names(self) -> dict:
        """Returns a dict mapping PyBullet body ID -> object name for verbose collision messages."""
        return {
            obj.sim_id: obj.config.name
            for obj in self.splatsim_objects
            if obj.sim_id is not None
        }

    def get_skip_pairs(self) -> set:
        """Returns a set of (robot_link_index, obstacle_body_id) pairs to skip during collision checks."""
        pairs = set()
        for obj in self.splatsim_objects:
            if obj.sim_id is not None and obj.config.skip_collision_robot_links:
                for link_idx in obj.config.skip_collision_robot_links:
                    pairs.add((link_idx, obj.sim_id))
        return pairs

    def _discover_gripper_finger_link_indices(self) -> list[int]:
        """Enumerate URDF link indices whose name contains 'finger' or 'knuckle'.

        Used by the IK collision filter when
        `TrajectoryGenModeConfig.ik_skip_gripper_obstacle_pairs` is True.
        Name-based detection covers the standard Robotiq 2F-85 subtree
        (left/right outer_finger, inner_finger, inner_finger_pad,
        outer_knuckle, inner_knuckle) without hard-coding indices.
        Cached after first call.
        """
        if getattr(self, "_gripper_finger_link_indices_cache", None) is not None:
            return self._gripper_finger_link_indices_cache
        finger_links: list[int] = []
        n_joints = self.pb.getNumJoints(self.robot_id)
        for j in range(n_joints):
            info = self.pb.getJointInfo(self.robot_id, j)
            raw_name = info[12]
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            lname = name.lower()
            if "finger" in lname or "knuckle" in lname:
                finger_links.append(j)
        self._gripper_finger_link_indices_cache = finger_links
        return finger_links

    def _ik_augmented_skip_pairs(self) -> set:
        """Skip-pair set used by the IK candidate filter in `_solve_ik`.

        When `self.config.ik_skip_gripper_obstacle_pairs` is True, this
        is `self.get_skip_pairs() ∪ {(finger_link, obs_id) for finger × obstacle}`.
        Otherwise it's `self.get_skip_pairs()` unchanged. Called per-`_solve_ik`
        so newly-registered obstacles are picked up automatically.
        """
        base = self.get_skip_pairs()
        if not getattr(self.config, "ik_skip_gripper_obstacle_pairs", False):
            return base
        finger_links = self._discover_gripper_finger_link_indices()
        if not finger_links:
            return base
        augmented = set(base)
        for link in finger_links:
            for obs_id in self.get_obstacle_ids():
                augmented.add((link, obs_id))
        return augmented

    def register_obstacle(self, sim_id: int):
        """Register an obstacle by its PyBullet body ID."""
        # This is a temporary function. Every obstacle should ideally be a splatsim object
        if sim_id not in self.loaded_obstacle_ids:
            self.loaded_obstacle_ids.append(sim_id)

    def _load_cuboid_obstacles(self, cuboids_fn: str) -> List[int]:
        """Load cuboid obstacles from NPZ file and create PyBullet bodies."""

        cuboid_bboxes = rrt_path_utils.load_cuboids(cuboids_fn)
        obstacle_ids = []

        for cuboid_bbox in cuboid_bboxes:
            cx, cy, cz, lx, ly, lz = cuboid_bbox
            obs = create_box(lx, ly, lz, color=RED)
            set_pose(obs, Pose(point=[cx, cy, cz]))
            obstacle_ids.append(obs)

        return obstacle_ids

    def _init_zarr_storage(self):
        """Initialize Zarr storage for trajectory data."""
        repo_id = self.config.lerobot_repo_id
        if repo_id:
            # Use repo_id as the folder name (replace '/' with '_' for filesystem safety)
            safe_name = repo_id.replace("/", "_")
            output_dir = os.path.join("output", f"{safe_name}_trajectories.zarr")
        else:
            output_dir = os.path.join("output", f"{self.env_config_name}_trajectories.zarr")
        self.zarr_root = zarr.open(output_dir, mode="a")
        self.scenarios_group = self.zarr_root.require_group("trajectories")

        # Find existing trajectory indices to resume
        traj_re = re.compile(r"^scenario_(\d+)$")
        existing_ids = []
        for name in self.scenarios_group.keys():
            node = self.scenarios_group[name]
            if isinstance(node, zarr.hierarchy.Group):
                m = traj_re.match(name)
                if m:
                    existing_ids.append(int(m.group(1)))

        # Resume from next index
        self.trajectory_count = (max(existing_ids) + 1) if existing_ids else 0
        print(f"[TrajectoryGenerator] Initialized storage. Resuming at index {self.trajectory_count}")

    def is_complete(self) -> bool:
        """Check if we've generated all requested trajectories."""
        return self.trajectory_count >= self.config.num_base_trajectories
    
    def check_able_to_solve(self, q_start: Optional[List[float] | np.ndarray]) -> bool:
        """Check if we can solve for at least one valid start/goal configuration."""
        curr_q_start = self.config.q_start
        curr_ee_pos_start = self.config.ee_pos_start
        curr_ee_quat_start = self.config.ee_quat_start

        curr_joint_positions = [s[0] for s in self.pb.getJointStates(self.robot_id, self.joint_indices)]

        self.config.q_start = q_start
        self.config.ee_pos_start = None
        self.config.ee_quat_start = None
        q_start, q_goals = self._get_start_and_goal_qs()

        self.config.q_start = curr_q_start
        self.config.ee_pos_start = curr_ee_pos_start
        self.config.ee_quat_start = curr_ee_quat_start

        for idx, qi in zip(self.joint_indices, curr_joint_positions):
            self.pb.resetJointState(self.robot_id, idx, qi)
            self.pb.setJointMotorControl2(self.robot_id, idx, self.pb.POSITION_CONTROL, targetPosition=qi)

        return q_start is not None and len(q_goals) > 0

    def try_plan_to_goal(
        self,
        q_start,
        target_ee_pos,
        target_ee_quat,
        q_goal_bias=None,
    ) -> Optional[np.ndarray]:
        """Attempt goal-IK + a full RRT path from ``q_start`` to the EE pose.

        Shared 'is this scenario actually solvable — and here's the demo path'
        primitive used by env resets. STRONGER than ``check_able_to_solve``,
        which only checks a goal CONFIG exists; this also verifies a
        collision-free PATH reaches it. Returns the time-parametrized joint path
        ``(T, num_dofs)`` on success (the chosen goal config is published on
        ``self._planner._last_chosen_q_goal``), or ``None`` on
        ``RRTPlanningError``. The robot's joint state is restored by
        ``planner.plan`` itself, so this is side-effect-free on the pose."""
        self._ensure_planner()
        self._sync_planner_obstacles()
        q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)[: len(self.joint_indices)]
        try:
            traj, _ = self._planner.plan(
                q_start,
                np.asarray(target_ee_pos, dtype=np.float64),
                np.asarray(target_ee_quat, dtype=np.float64),
                q_goal_bias=(None if q_goal_bias is None else np.asarray(q_goal_bias, dtype=np.float64)),
            )
            return traj
        except RRTPlanningError:
            return None

    def _get_start_and_goal_qs(self) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Get collision free start and goal joint configurations, resolving EE pose goals to joint space via IK if needed.

        TODO make this function and the later functions able to handle multiple q_starts
        """
        # 1. Get start/goal configurations
        q_start = self.config.q_start
        has_ee_start = (
            self.config.ee_pos_start is not None
            or self.config.ee_quat_start is not None
        )
        if q_start is not None:
            all_q_starts = [np.array(q_start)]
        elif has_ee_start:
            all_q_starts = self._resolve_ee_pose_to_q_candidates(
                self.config.ee_pos_start, self.config.ee_quat_start, label="start"
            )
            if len(all_q_starts) == 0:
                return None, []
        else:
            all_q_starts = [self._get_random_collision_free_q()]

        q_start = all_q_starts[0]

        q_goal = self.config.q_goal
        has_ee_goal = (
            self.config.ee_pos_goal is not None
            or self.config.ee_quat_goal is not None
        )

        if has_ee_goal:
            # Resolve EE goal to joint-space candidates via IK
            q_goal_candidates = self._resolve_ee_goal_to_q_goals()
            if len(q_goal_candidates) == 0:
                return None, []
            q_goal = q_goal_candidates[0]
            q_goal_fallbacks = q_goal_candidates[1:]
        elif q_goal is None:
            q_goal = self._get_random_collision_free_q()
            q_goal_fallbacks = []
        else:
            q_goal = np.array(q_goal)
            q_goal_fallbacks = []

        # 2. Generate base trajectory (with optional camera scoring + fallback goals)
        all_q_goals = [q_goal] + q_goal_fallbacks

        if self.config.debug_visualize:
            for i, qs in enumerate(all_q_starts):
                rrt_path_utils.show_joint_config_in_gui(self.robot_id, self.joint_indices, qs)
                input(f"[debug_visualize] q_start candidate {i+1}/{len(all_q_starts)}. Press Enter for next...")
            for i, qg in enumerate(all_q_goals):
                rrt_path_utils.show_joint_config_in_gui(self.robot_id, self.joint_indices, qg)
                input(f"[debug_visualize] q_goal candidate {i+1}/{len(all_q_goals)}. Press Enter for next...")
        return q_start, all_q_goals

    def _ensure_planner(self) -> "RRTToGoalPlanner":
        """Build the shared RRTToGoalPlanner on first use (see the deferral
        note in __init__ — `get_ee_link_fn()` isn't safe to call until the
        parent robot server has finished setting up `self.wrist_camera`).
        Cached after the first call. Kinematic limits are hardcoded 0.5/1.0/10.0
        to match the conservative smoothing `generate_trajectory_batch`
        historically applied."""
        if self._planner is None:
            self._planner = RRTToGoalPlanner(
                pb_client=self._pb_client_id,
                robot_id=self.robot_id,
                joint_indices=list(self.joint_indices),
                ee_link_index=self.get_ee_link_fn(),
                num_dofs=len(self.joint_indices),
                fps=self.config.robot_update_rate,
                lower_limits=self.lower_limits,
                upper_limits=self.upper_limits,
                num_ik_candidates=self.config.num_ik_candidates,
                max_joint_vel=self.config.max_joint_vel,
                max_joint_acc=self.config.max_joint_acc,
                max_joint_jerk=self.config.max_joint_jerk,
                # Parametrize only the FINAL trajectory (cheap linear-densify checks
                # per candidate) so we parametrize once per plan, not once per
                # candidate — that overwhelmed it during batch trajectory-gen.
                parametrize_per_candidate=False,
                segment_at_sharp_corners=self.config.segment_at_sharp_corners,
                path_selection=PathSelectionStrategy(self.config.path_selection),
                # Config-driven (default "joint_distance"): pick the IK goal
                # nearest q_start to avoid the far wrist-flipped branch that
                # self-collides during execution. "none" falls back to scoring
                # all IK candidates via path_selection. See TrajectoryGenModeConfig.
                ik_goal_selection=self.config.ik_goal_selection,
                num_path_candidates_per_ik=self.config.num_path_candidates,
                max_path_attempts_per_ik=self.config.max_path_attempts,
                path_perturbation_scale=self.config.path_perturbation_scale,
                rrt_smooth_iterations=self.config.rrt_smooth_iterations,
                elastic_smooth_passes=self.config.elastic_smooth_passes,
                trajopt_passes=self.config.trajopt_passes,
                trajopt_lr=self.config.trajopt_lr,
                trajopt_smoothness_weight=self.config.trajopt_smoothness_weight,
                trajopt_collision_weight=self.config.trajopt_collision_weight,
                trajopt_collision_threshold=self.config.trajopt_collision_threshold,
                trajopt_fd_step=self.config.trajopt_fd_step,
                obstacle_clearance_factor=self.config.obstacle_clearance_factor,
                final_approach_dist=self.config.final_approach_dist,
                final_approach_vel_scale=self.config.final_approach_vel_scale,
                final_approach_acc_scale=self.config.final_approach_acc_scale,
                uniform_path_speed=self.config.uniform_path_speed,
                freeze_visualizer_during_plan=self.config.freeze_visualizer_during_plan,
                obstacle_clearance=self.config.obstacle_clearance,
                self_collision_clearance=self.config.self_collision_clearance,
                self_collision_skip_pairs=self.config.self_collision_skip_pairs,
                # Match _solve_ik's IK collision filter: at grasp goals the
                # gripper fingers are intentionally within mm of the target, so
                # skip finger⟷obstacle pairs during IK candidate resolution.
                # Without this the planner's goal-IK rejects every grasp pose
                # that trajectory_generation._solve_ik accepts, yielding
                # "No collision-free IK solution found" despite valid candidates.
                ik_skip_gripper_obstacle_pairs=self.config.ik_skip_gripper_obstacle_pairs,
                wrist_camera_link_index=self.camera_link_index,
                camera_k_exp=self.config.k_exp,
                camera_k_sig=self.config.k_sig,
                camera_threshold=self.config.threshold,
                # Soft-cost usage ("off"/"score"/"guided") + scoring weight.
                # Config default is "off" (small_engine/planar have no field
                # and stay cost-blind by construction); envs with a field
                # opt in via their traj-gen config (vine -> "guided").
                soft_cost_mode=getattr(self.config, "soft_cost_mode", "off"),
                soft_cost_weight=getattr(self.config, "soft_cost_weight", 1.0),
                # Surface-ring sampling + max reduction (see
                # RRTToGoalPlanner._config_soft_cost_points). getattr-guarded
                # like the two above so an older config JSON without these
                # fields still loads.
                soft_cost_surface_samples=getattr(
                    self.config, "soft_cost_surface_samples", 6),
                soft_cost_aggregation=getattr(
                    self.config, "soft_cost_aggregation", "max"),
            )
            # This planner never sees load_obstacles (it plans against the
            # env's live world), so the soft-cost field must be attached
            # explicitly — otherwise cost-aware scoring silently stays off
            # for in-env trajectory generation.
            if self.soft_cost_payload:
                try:
                    self._planner.set_soft_cost_field(self.soft_cost_payload)
                except Exception:
                    logger.exception(
                        "TrajectoryGenerator: failed to attach soft-cost "
                        "field %s — continuing cost-blind",
                        self.soft_cost_payload,
                    )
        return self._planner

    def _sync_planner_obstacles(self, additional_obstacles=None):
        """Push the current obstacle set into the shared planner.

        We deliberately DO NOT call ``planner.load_obstacles()`` — that would
        delete/recreate bodies in this shared sim client. Instead we set the
        three attributes the planner reads (``_loaded_obstacle_ids``,
        ``_obstacle_names``, ``_skip_pairs``) directly from the generator's
        existing obstacle helpers.
        """
        self._ensure_planner()
        ids = list(self.get_obstacle_ids())
        if additional_obstacles:
            ids += list(additional_obstacles)
        self._planner._loaded_obstacle_ids = ids
        self._planner._obstacle_names = self.get_obstacle_names()
        # Use BASE skip pairs (per-obstacle skip_collision_robot_links from the
        # env config) for the planner's global collision check. Path checks —
        # RRT tree, linear densify, time-parametrized, final gate — see the
        # gripper's real collision mesh here so mid-trajectory gripper⟷obstacle
        # contact is detected and rejected. Previously this line pushed
        # `_ik_augmented_skip_pairs()` here, which added (finger, obstacle) for
        # EVERY obstacle × EVERY finger — silently masking gripper-vs-engine /
        # gripper-vs-any-obstacle collisions during traversal. The augmentation
        # is still applied at `_solve_ik` (grasp goals need fingers within
        # obstacle_clearance of the target); when a goal's LAST waypoint fails
        # the path check because fingers sit inside the target's clearance
        # buffer, the fix is to mark the target's env-config
        # `skip_collision_robot_links` explicitly rather than universally
        # blinding every check.
        self._planner._skip_pairs = self.get_skip_pairs()

    def _fk_ee_pose(self, q) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-kinematics a joint config ``q`` to an EE world pose.

        Snaps the robot on the shared client to ``q`` and reads the EE link
        state. Returns ``(pos(3,), quat(4,))``. Used to bridge the generator's
        joint-space goal resolution (``_get_start_and_goal_qs``) into the
        planner's EE-pose-based ``plan()`` API.
        """
        ee_link_index = self.get_ee_link_fn()
        for idx, qi in zip(self.joint_indices, np.asarray(q).reshape(-1)):
            self.pb.resetJointState(self.robot_id, idx, float(qi))
        link_state = self.pb.getLinkState(
            self.robot_id, ee_link_index, computeForwardKinematics=True
        )
        # URDF link frame (4/5), not COM (0/1) — must match what pybullet IK
        # solves for and what the planner's FK-accuracy gate compares
        # (identical for links whose COM sits at the frame origin).
        return np.asarray(link_state[4]), np.asarray(link_state[5])

    def generate_trajectory_batch(self):
        """Generate one base trajectory with multiple obstacle configurations.

        Returns:
            Optional[List[dict]]: List of episode dicts, each containing:
                - "joint_positions": np.ndarray of shape (N, DOF)
                - "obstacle_info": dict with obstacle metadata (e.g. {"obstacles": [...]})
                - "zarr_group": zarr.Group reference for saving images later
            Returns None if planning failed.
        """
        if self.config.save_zarr:
            self._init_zarr_storage()

        # Reuse the path already validated during env.reset() if one was cached,
        # skipping the (expensive) base-path re-plan. Consumed exactly once.
        cached = self._cached_base_traj
        self._cached_base_traj = None
        if cached is not None and len(cached) >= 2:
            base_traj = np.asarray(cached, dtype=np.float64)
            q_start = base_traj[0].copy()
            q_goal = base_traj[-1].copy()
            self._sync_planner_obstacles()
        else:
            q_start, all_q_goals = self._get_start_and_goal_qs()
            if q_start is None or len(all_q_goals) == 0:
                print("[TrajectoryGenerator] Failed to get valid start/goal configurations. Skipping this trajectory.")
                return None

            # Delegate RRT/IK/path-scoring/parametrization to the shared planner. We keep
            # `_get_start_and_goal_qs()` for goal RESOLUTION (EE-goal vs direct-q
            # vs random → joint config), then bridge into the planner's EE-pose
            # API by FK'ing the chosen joint goal to an EE pose. `q_goal_bias` is
            # seeded with that joint config so the planner's IK converges back to
            # the same branch. The returned `base_traj` is ALREADY time-parametrized
            # (with the same conservative 0.5/1.0/10.0 limits configured on the
            # planner), so no standalone parametrization pass runs here.
            self._sync_planner_obstacles()
            q_goal = all_q_goals[0]

            # Prefer a user-specified EE-pose goal as the target for fidelity;
            # otherwise FK the resolved joint goal to an EE pose.
            if self.config.ee_pos_goal is not None and self.config.ee_quat_goal is not None:
                target_ee_pos = np.asarray(self.config.ee_pos_goal, dtype=np.float64)
                target_ee_quat = np.asarray(self.config.ee_quat_goal, dtype=np.float64)
            else:
                target_ee_pos, target_ee_quat = self._fk_ee_pose(q_goal)

            try:
                base_traj, _escape_end_q = self._planner.plan(
                    q_start,
                    target_ee_pos,
                    target_ee_quat,
                    q_goal_bias=q_goal,
                )
            except RRTPlanningError as e:
                print(f"[TrajectoryGenerator] RRT planning failed: {e}. Skipping this trajectory (will retry next iteration).")
                return None  # Failed, will retry next iteration

        # Debug visualization: show the chosen start/goal then play back the trajectory
        if self.config.debug_visualize:
            rrt_path_utils.show_joint_config_in_gui(self.robot_id, self.joint_indices, q_start)
            input("[debug_visualize] Showing chosen q_start. Press Enter to show chosen q_goal...")
            rrt_path_utils.show_joint_config_in_gui(self.robot_id, self.joint_indices, q_goal)
            input("[debug_visualize] Showing chosen q_goal. Press Enter to play trajectory...")
            rrt_path_utils.playback_path_in_gui(
                base_traj, self.robot_id, self.joint_indices,
                path_name="base_traj",
                fps=self.config.robot_update_rate,
            )

        # Append gripper state (hardcoded 0 = open) as the last column of each q.
        base_traj = np.hstack([base_traj, np.zeros((base_traj.shape[0], 1), dtype=base_traj.dtype)])

        # 2b. Optionally extend the path so it holds the last position for a
        # second (opt-in via pad_stopped_last_frames; default off — the frozen
        # tail is otherwise dead frames that inflate episode length).
        if self.config.pad_stopped_last_frames:
            num_extra_steps = int(1 * self.config.robot_update_rate)
            last_q = base_traj[-1]
            extra_steps = np.tile(last_q, (num_extra_steps, 1))
            base_traj = np.vstack((base_traj, extra_steps))

        # 3. Get EE trajectory for obstacle placement
        base_ee_traj = self._get_ee_trajectory(base_traj)

        results = []

        # 4. Save base trajectory (no obstacles) if configured
        if self.config.save_base_trajectory:
            obstacle_info = {"obstacles": []}
            zarr_group = self._save_trajectory_zarr(
                base_traj, self.trajectory_count, 0, 0, obstacle_info
            ) if self.config.save_zarr else None
            results.append({
                "joint_positions": base_traj,
                "obstacle_info": obstacle_info,
                "zarr_group": zarr_group,
            })

        # 5. Generate trajectories with obstacles
        if self.config.obstacles_per_base_trajectory > 0:
            for obstacle_i in range(
                1, self.config.obstacles_per_base_trajectory + 1
            ):
                # Add random obstacles
                obstacle_ids, obstacle_infos = self._add_random_obstacles(
                    [q_start, q_goal], base_ee_traj
                )

                obstacle_info = {"obstacles": obstacle_infos}

                # Plan the same start→goal with the extra obstacles loaded.
                # Sync the planner's obstacle set to include the freshly-added
                # random obstacles, then re-derive the EE-pose target for the
                # goal (FK on the shared client) so the planner can solve IK.
                self._sync_planner_obstacles(additional_obstacles=obstacle_ids)
                if self.config.ee_pos_goal is not None and self.config.ee_quat_goal is not None:
                    obs_target_ee_pos = np.asarray(self.config.ee_pos_goal, dtype=np.float64)
                    obs_target_ee_quat = np.asarray(self.config.ee_quat_goal, dtype=np.float64)
                else:
                    obs_target_ee_pos, obs_target_ee_quat = self._fk_ee_pose(q_goal)

                # Generate multiple paths per obstacle configuration
                for path_i in range(self.config.paths_per_obstacle):
                    try:
                        modified_traj, _ = self._planner.plan(
                            q_start,
                            obs_target_ee_pos,
                            obs_target_ee_quat,
                            q_goal_bias=q_goal,
                        )
                    except RRTPlanningError:
                        modified_traj = None

                    if modified_traj is not None:
                        # Optionally hold the last position for a second (opt-in;
                        # see the base_traj site above).
                        if self.config.pad_stopped_last_frames:
                            num_extra_steps = int(1 * self.config.robot_update_rate)
                            last_q = modified_traj[-1]
                            extra_steps = np.tile(last_q, (num_extra_steps, 1))
                            modified_traj = np.vstack((modified_traj, extra_steps))

                        zarr_group = self._save_trajectory_zarr(
                            modified_traj,
                            self.trajectory_count,
                            obstacle_i,
                            path_i,
                            obstacle_info,
                        ) if self.config.save_zarr else None
                        results.append({
                            "joint_positions": modified_traj,
                            "obstacle_info": obstacle_info,
                            "zarr_group": zarr_group,
                        })

                # Remove obstacles for next iteration
                self._remove_obstacles(obstacle_ids)

        self.trajectory_count += 1
        return results

    def _get_random_collision_free_q(self) -> np.ndarray:
        """Generate random collision-free joint configuration.

        Must pass BOTH skip-pair sets so the collision check honors the
        same contract the path-check downstream uses:
          * ``skip_pairs`` — (robot_link, obstacle_body_id), from
            per-obstacle ``skip_collision_robot_links`` config.
          * ``self_collision_skip_pairs`` — non-adjacent robot-robot
            pairs that are structurally close by URDF design (e.g. UR
            base_link ⟷ upper_arm, Robotiq inner_finger ⟷ inner_knuckle
            mesh overlap). Without these, `env.reset() → randomize_objects
            → check_able_to_solve` at `self_collision_clearance > 0`
            fails on every sample and the outer while-loop hangs.
        """
        return rrt_path_utils.get_random_joint_angles_without_collision(
            self.robot_id,
            self.joint_indices,
            self.get_obstacle_ids(),
            self.lower_limits,
            self.upper_limits,
            max_tries=10000,
            verbose=self.config.verbose,
            skip_pairs=self.get_skip_pairs(),
            # Match the path-check contract: BOTH threshold and skip list
            # come from the same TrajectoryGenModeConfig so the sampler
            # sees the same "what counts as a collision" definition the
            # rest of the trajectory generator uses.
            self_collision_clearance=self.config.self_collision_clearance,
            self_collision_skip_pairs=self.config.self_collision_skip_pairs,
        )

    # =========================================================================
    # End-Effector Goal Resolution via IK
    # =========================================================================

    def _resolve_ee_goal_to_q_goals(self) -> List[np.ndarray]:
        """Resolve end-effector pose goal(s) to joint-space goal candidates via IK.

        If ``q_goal_bias`` is configured, the first IK attempt is seeded at the
        bias (so demos converge to a shared joint configuration when feasible);
        the remaining attempts use random seeds as fallbacks for cases where
        the bias is blocked.
        """
        seed_q_bias = (
            np.array(self.config.q_goal_bias, dtype=np.float64)
            if self.config.q_goal_bias is not None else None
        )
        return self._resolve_ee_pose_to_q_candidates(
            self.config.ee_pos_goal, self.config.ee_quat_goal,
            label="goal", seed_q_bias=seed_q_bias,
        )

    def _resolve_ee_pose_to_q_candidates(
        self,
        ee_pos: Optional[List[float]],
        ee_quat: Optional[List[float]],
        label: str = "pose",
        seed_q_bias: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Resolve an end-effector pose to joint-space candidates via IK.

        Handles three cases:
        1. ee_pos + ee_quat: Full pose. Run IK from multiple random seeds.
        2. ee_pos only: Sample multiple orientations, run IK for each.
        3. ee_quat only: Sample multiple positions via FK, run IK for each.

        If ``seed_q_bias`` is provided, the first IK attempt is seeded at the
        bias so the canonical solution is preferred when feasible. The
        remaining attempts use random seeds as fallbacks.

        Returns:
            List of collision-free q candidates (may be empty if all IK attempts fail).
        """
        num_candidates = self.config.num_ik_candidates
        ee_link_index = self.get_ee_link_fn()

        candidates = []

        if ee_pos is not None and ee_quat is not None:
            for i in range(num_candidates):
                seed_q = seed_q_bias if (i == 0 and seed_q_bias is not None) else None
                q = self._solve_ik(ee_pos, ee_quat, ee_link_index, seed_q=seed_q)
                if q is not None:
                    candidates.append(q)

        elif ee_pos is not None:
            for i in range(num_candidates):
                sampled_quat = self._sample_random_quaternion()
                seed_q = seed_q_bias if (i == 0 and seed_q_bias is not None) else None
                q = self._solve_ik(ee_pos, sampled_quat, ee_link_index, seed_q=seed_q)
                if q is not None:
                    candidates.append(q)

        elif ee_quat is not None:
            for i in range(num_candidates):
                sampled_pos = self._sample_reachable_position()
                seed_q = seed_q_bias if (i == 0 and seed_q_bias is not None) else None
                q = self._solve_ik(sampled_pos, ee_quat, ee_link_index, seed_q=seed_q)
                if q is not None:
                    candidates.append(q)

        candidates = self._deduplicate_q_candidates(candidates)

        if len(candidates) == 0:
            print(f"[TrajectoryGenerator] Failed to find any valid IK solution for EE {label} "
                  f"(tried {num_candidates} candidates). "
                  f"ee_pos={ee_pos}, ee_quat={ee_quat}")
        else:
            print(f"[TrajectoryGenerator] Resolved EE {label} to {len(candidates)} IK candidate(s)")

        return candidates

    def _solve_ik(
        self,
        ee_pos: List[float],
        ee_quat: List[float],
        ee_link_index: int,
        seed_q: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Solve IK for a given EE pose. Returns collision-free joint config or None.

        Args:
            seed_q: Optional joint config to use as the IK seed (and rest pose).
                When provided, PyBullet's damped-least-squares solver converges to
                the IK branch nearest to ``seed_q``. If None, a random seed is
                used (gives diverse solutions across calls).
        """
        is_biased = seed_q is not None
        if seed_q is None:
            seed_q = self._get_random_q_within_limits()
        for idx, qi in zip(self.joint_indices, seed_q):
            self.pb.resetJointState(self.robot_id, idx, qi)

        # Pass jointRanges (= upper - lower) so PyBullet activates null-space IK
        # and actually uses restPoses to bias the solution toward seed_q. Without
        # jointRanges, restPoses is silently ignored and only the joint-state
        # seed weakly influences the converged branch.
        joint_ranges = (self.upper_limits - self.lower_limits).tolist()
        # maxNumIterations=512 (was 100000): DLS converges or plateaus well
        # under 200 iterations — profiled identical FK error at 200 vs 100k,
        # but 100k costs ~14-380 ms/call (residualThreshold=1e-10 never
        # early-exits). This runs num_ik_candidates (32) times per goal
        # resolution — including check_able_to_solve inside the env-reset
        # randomize_objects loop — so the cap turns a multi-second reset
        # stall into ~60 ms. Bad/inaccurate solutions are still rejected by
        # the bias-drift check and collision filter below.
        q_solution = self.pb.calculateInverseKinematics(
            self.robot_id,
            ee_link_index,
            ee_pos,
            ee_quat,
            maxNumIterations=512,
            residualThreshold=1e-10,
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=joint_ranges,
            restPoses=list(seed_q),
        )

        # Extract only the joints we control
        q_solution = np.array(q_solution[:len(self.joint_indices)])

        # Wrap to [-pi, pi]
        q_solution = ((q_solution + np.pi) % (2 * np.pi)) - np.pi

        # Check joint limits
        if np.any(q_solution < self.lower_limits) or np.any(q_solution > self.upper_limits):
            return None

        # If we explicitly seeded with a bias, reject IK results that wandered
        # too far from it — null-space IK on a 6-DOF arm can still flip branches
        # when the residual is small, and the caller will retry with random seeds.
        if is_biased:
            seed_arr = np.asarray(seed_q)
            wrapped_diff = ((q_solution - seed_arr + np.pi) % (2 * np.pi)) - np.pi
            max_drift = float(np.max(np.abs(wrapped_diff)))
            if max_drift > np.pi / 3:  # 60° per-joint tolerance
                if self.config.verbose:
                    print(f"[TrajectoryGenerator] IK seeded with q_goal_bias drifted "
                          f"{np.degrees(max_drift):.1f}° from seed; rejecting and "
                          f"falling back to random-seed IK.")
                return None
            if self.config.verbose:
                print(f"[TrajectoryGenerator] IK seeded with q_goal_bias converged "
                      f"within {np.degrees(max_drift):.2f}° of seed.")

        # `_ik_augmented_skip_pairs()` extends `get_skip_pairs()` with
        # gripper-finger ⟷ obstacle pairs when
        # `config.ik_skip_gripper_obstacle_pairs=True`. Grasp-goal IKs
        # otherwise fail on legit finger-near-target proximity; the arm
        # links are still checked normally, and RRT path search /
        # runtime shield keep gripper vs obstacle strict.
        #
        # verbose=False: IK candidate acceptance is called in a hot loop
        # (up to `num_ik_candidates` per goal × per env.reset sample),
        # and each collision hit was firing a per-pair "Collision: ..."
        # print that dominated the profile. Rejection is signalled by
        # the return value; the diagnostic prints aren't needed here.
        # Flip to True locally for one-off debugging.
        in_col = rrt_path_utils.check_links_in_collision(
            self.robot_id, self.joint_indices, q_solution,
            self.get_obstacle_ids(),
            verbose=False,
            obstacle_names=self.get_obstacle_names(),
            skip_pairs=self._ik_augmented_skip_pairs(),
            obstacle_clearance=self.config.obstacle_clearance,
            self_collision_clearance=self.config.self_collision_clearance,
            self_collision_skip_pairs=self.config.self_collision_skip_pairs,
        )
        if self.config.debug_visualize:
            rrt_path_utils.show_joint_config_in_gui(self.robot_id, self.joint_indices, q_solution)
            input(f"[debug_visualize] IK candidate: {'IN COLLISION — rejecting' if in_col else 'collision-free — keeping'}. Press Enter...")
        if in_col:
            return None

        # Verify IK accuracy via FK
        for idx, qi in zip(self.joint_indices, q_solution):
            self.pb.resetJointState(self.robot_id, idx, qi)
        link_state = self.pb.getLinkState(self.robot_id, ee_link_index, computeForwardKinematics=True)

        actual_pos = np.array(link_state[0])
        if np.linalg.norm(actual_pos - np.array(ee_pos)) > 0.005:  # 5mm tolerance
            return None

        # If ee_quat_goal was user-specified, also check orientation accuracy
        if self.config.ee_quat_goal is not None:
            actual_quat = np.array(link_state[1])
            target_quat = np.array(ee_quat)
            dot = np.abs(np.dot(actual_quat, target_quat))
            dot = np.clip(dot, -1.0, 1.0)
            angle_error_deg = np.degrees(2 * np.arccos(dot))
            if angle_error_deg > 1.0:  # 1 degree tolerance
                return None

        return q_solution

    def _get_random_q_within_limits(self) -> np.ndarray:
        """Generate random joint configuration within joint limits (no collision check)."""
        return np.random.uniform(self.lower_limits, self.upper_limits)

    def _sample_random_quaternion(self) -> List[float]:
        """Sample a uniformly random unit quaternion in PyBullet (x,y,z,w) format."""
        euler = [
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
        ]
        return list(self.pb.getQuaternionFromEuler(euler))

    def _sample_reachable_position(self) -> List[float]:
        """Sample a position in the robot's reachable workspace via FK on a random config."""
        random_q = self._get_random_q_within_limits()
        for idx, qi in zip(self.joint_indices, random_q):
            self.pb.resetJointState(self.robot_id, idx, qi)
        ee_link_index = self.get_ee_link_fn()
        link_state = self.pb.getLinkState(self.robot_id, ee_link_index, computeForwardKinematics=True)
        return list(link_state[0])

    def _deduplicate_q_candidates(
        self, candidates: List[np.ndarray], threshold_rad: float = 0.1
    ) -> List[np.ndarray]:
        """Remove near-duplicate joint configurations."""
        if len(candidates) <= 1:
            return candidates
        unique = [candidates[0]]
        for candidate in candidates[1:]:
            is_duplicate = any(
                np.linalg.norm(candidate - existing) < threshold_rad
                for existing in unique
            )
            if not is_duplicate:
                unique.append(candidate)
        return unique

    def _get_ee_trajectory(self, joint_trajectory: np.ndarray) -> np.ndarray:
        """Compute end-effector positions for joint trajectory."""
        ee_link_index = self.get_ee_link_fn()
        ee_positions = []

        for q in joint_trajectory:
            # Set joint positions
            for idx, qi in zip(self.joint_indices, q):
                self.pb.resetJointState(self.robot_id, idx, qi)

            # Get EE position
            link_state = self.pb.getLinkState(
                self.robot_id, ee_link_index, computeForwardKinematics=True
            )
            ee_positions.append(list(link_state[0]))

        return np.array(ee_positions)

    def _add_random_obstacles(
        self, robot_qs_to_avoid: List[np.ndarray], base_ee_traj: np.ndarray
    ) -> Tuple[List[int], List[dict]]:
        """Add random cuboid obstacles avoiding robot configurations."""
        # Adapted from test_traj_refinement.py add_random_obstacles
        new_obj_ids = []
        new_obj_infos = []
        num_obstacles = np.random.randint(
            self.config.min_obstacles, self.config.max_obstacles + 1
        )

        MIN_TIME_PROPORTION = 0.2
        MAX_TIME_PROPORTION = 0.8

        for _ in range(num_obstacles):
            success = False
            while not success:
                success = True

                # Random position along EE trajectory
                time_proportion = np.random.uniform(
                    MIN_TIME_PROPORTION, MAX_TIME_PROPORTION
                )
                pos = base_ee_traj[int(len(base_ee_traj) * time_proportion)]
                orn = self.pb.getQuaternionFromEuler(
                    [0, 0, np.random.uniform(0, np.pi)]
                )

                # Random box size
                lx = np.random.uniform(0.02, 0.30)
                ly = np.random.uniform(0.02, 0.30)
                lz = np.random.uniform(0.02, 0.30)

                # Create obstacle
                body_id = create_box(lx, ly, lz, color=BLUE)
                set_pose(body_id, Pose(point=pos))

                # Check for collisions with robot at start/goal
                for robot_q in robot_qs_to_avoid:
                    rrt_path_utils.set_robot_joint_positions(
                        self.robot_id, self.joint_indices, robot_q
                    )
                    self.pb.stepSimulation()
                    collisions = self.pb.getClosestPoints(
                        bodyA=body_id, bodyB=self.robot_id, distance=0.05
                    )
                    if len(collisions) > 0:
                        success = False

                if not success:
                    self.pb.removeBody(body_id)

            new_obj_ids.append(body_id)
            new_obj_infos.append(
                {
                    "type": "cuboid",
                    "pos": list(pos),
                    "orn": [0, 0, 0, 1],
                    "size": (lx, ly, lz),
                }
            )

        return new_obj_ids, new_obj_infos

    def _remove_obstacles(self, obstacle_ids: List[int]):
        """Remove obstacles from simulation."""
        for oid in obstacle_ids:
            self.pb.removeBody(oid)

    def _save_trajectory_zarr(
        self,
        trajectory: np.ndarray,
        scenario_idx: int,
        obstacle_config_idx: int,
        traj_idx: int,
        obstacle_info: dict,
    ):
        """Save trajectory to Zarr format.

        Returns:
            zarr.Group: The trajectory group where data was saved, for adding images later.
        """
        # Create hierarchy: scenario_XXXX/obstacle_config_XX/traj_XX
        scenario_name = f"scenario_{scenario_idx:04d}"
        obstacle_name = f"obstacle_config_{obstacle_config_idx:02d}"
        traj_name = f"traj_{traj_idx:02d}"

        # Get or create groups
        if scenario_name not in self.scenarios_group:
            scenario_grp = self.scenarios_group.create_group(scenario_name)
        else:
            scenario_grp = self.scenarios_group[scenario_name]

        if obstacle_name not in scenario_grp:
            obstacle_grp = scenario_grp.create_group(obstacle_name)
            obstacle_grp.attrs["metadata"] = json.dumps(obstacle_info)
        else:
            obstacle_grp = scenario_grp[obstacle_name]

        if traj_name not in obstacle_grp:
            traj_grp = obstacle_grp.create_group(traj_name)
        else:
            traj_grp = obstacle_grp[traj_name]

        # Save trajectory data
        N_SAMPLES = trajectory.shape[0]
        DOF = trajectory.shape[1]

        if "qs" in traj_grp:
            del traj_grp["qs"]

        traj_grp.create_dataset(
            "qs", data=trajectory, dtype="f4", chunks=(N_SAMPLES, DOF)
        )

        return traj_grp
