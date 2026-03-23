import numpy as np
import zarr
import os
import time
import re
import json
from typing import Optional, Tuple, List
from pybullet_planning import create_box, set_pose, Pose, RED, BLUE

from splatsim.configs import TrajectoryGenModeConfig
from splatsim.utils import rrt_path_utils
from splatsim.configs.env_config import SplatSimObject

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
    ):
        """
        Initialize trajectory generator.

        Args:
            pybullet_client: PyBullet client instance
            robot_id: Robot body ID in PyBullet
            joint_indices: List of movable joint indices
            env_config_name: Environment name for output directory
            get_ee_link_fn: Function that returns EE link index
            splatsim_objects: List of SplatSimObject instances
            wrist_camera_link_name: Name of wrist camera link for camera-aware scoring
            trajectory_gen_config: Trajectory generation configuration (uses defaults if None)
        """
        self.pb = pybullet_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices

        if trajectory_gen_config is None:
            trajectory_gen_config = TrajectoryGenModeConfig()
        self.config = trajectory_gen_config
        self.env_config_name = env_config_name
        self.get_ee_link_fn = get_ee_link_fn
        self.splatsim_objects = splatsim_objects

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

        q_start, all_q_goals = self._get_start_and_goal_qs()
        if q_start is None or len(all_q_goals) == 0:
            print("[TrajectoryGenerator] Failed to get valid start/goal configurations. Skipping this trajectory.")
            return None

        result = self._plan_with_fallback_goals(q_start, all_q_goals)
        if result is None:
            return None  # Failed, will retry next iteration
        base_traj, q_goal = result

        # Apply ruckig time-parametrization once on the final trajectory.
        dof = len(self.joint_indices)
        tries = 0
        max_tries = 5
        while tries < max_tries:
            try:
                base_traj = rrt_path_utils.ruckig_parametrize_path(
                    base_traj,
                    max_joint_vel=np.full(dof, 0.5),
                    max_joint_acc=np.full(dof, 1.0),
                    max_joint_jerk=np.full(dof, 10.0),
                    control_hz=self.config.robot_update_rate,
                )
                break
            except Exception as e:
                tries += 1
                print(f"Ruckig path smoothing failed with exception {e}")
                print(f"Retry {tries} / {max_tries}")
                if tries >= max_tries:
                    raise RuntimeError(f"Ruckig path smoothing failed after {max_tries} attempts: {e}")
                time.sleep(10)

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

        # 2b. Extend the path so that it stays at the last position for a second
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

                # Generate multiple paths per obstacle configuration
                for path_i in range(self.config.paths_per_obstacle):
                    modified_traj = self._plan_rrt_path(
                        q_start, q_goal, additional_obstacles=obstacle_ids
                    )

                    if modified_traj is not None:
                        # Extend the path so that it stays at the last position for a second
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
        """Generate random collision-free joint configuration."""
        return rrt_path_utils.get_random_joint_angles_without_collision(
            self.robot_id,
            self.joint_indices,
            self.get_obstacle_ids(),
            self.lower_limits,
            self.upper_limits,
            max_tries=10000,
            verbose=self.config.verbose,
            skip_pairs=self.get_skip_pairs(),
        )

    # =========================================================================
    # End-Effector Goal Resolution via IK
    # =========================================================================

    def _resolve_ee_goal_to_q_goals(self) -> List[np.ndarray]:
        """Resolve end-effector pose goal(s) to joint-space goal candidates via IK."""
        return self._resolve_ee_pose_to_q_candidates(
            self.config.ee_pos_goal, self.config.ee_quat_goal, label="goal"
        )

    def _resolve_ee_pose_to_q_candidates(
        self,
        ee_pos: Optional[List[float]],
        ee_quat: Optional[List[float]],
        label: str = "pose",
    ) -> List[np.ndarray]:
        """Resolve an end-effector pose to joint-space candidates via IK.

        Handles three cases:
        1. ee_pos + ee_quat: Full pose. Run IK from multiple random seeds.
        2. ee_pos only: Sample multiple orientations, run IK for each.
        3. ee_quat only: Sample multiple positions via FK, run IK for each.

        Returns:
            List of collision-free q candidates (may be empty if all IK attempts fail).
        """
        num_candidates = self.config.num_ik_candidates
        ee_link_index = self.get_ee_link_fn()

        candidates = []

        if ee_pos is not None and ee_quat is not None:
            for _ in range(num_candidates):
                q = self._solve_ik(ee_pos, ee_quat, ee_link_index)
                if q is not None:
                    candidates.append(q)

        elif ee_pos is not None:
            for _ in range(num_candidates):
                sampled_quat = self._sample_random_quaternion()
                q = self._solve_ik(ee_pos, sampled_quat, ee_link_index)
                if q is not None:
                    candidates.append(q)

        elif ee_quat is not None:
            for _ in range(num_candidates):
                sampled_pos = self._sample_reachable_position()
                q = self._solve_ik(sampled_pos, ee_quat, ee_link_index)
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
    ) -> Optional[np.ndarray]:
        """Solve IK for a given EE pose. Returns collision-free joint config or None."""
        # Seed robot at random joint config for diverse IK solutions
        seed_q = self._get_random_q_within_limits()
        for idx, qi in zip(self.joint_indices, seed_q):
            self.pb.resetJointState(self.robot_id, idx, qi)

        q_solution = self.pb.calculateInverseKinematics(
            self.robot_id,
            ee_link_index,
            ee_pos,
            ee_quat,
            maxNumIterations=100000,
            residualThreshold=1e-10,
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
        )

        # Extract only the joints we control
        q_solution = np.array(q_solution[:len(self.joint_indices)])

        # Wrap to [-pi, pi]
        q_solution = ((q_solution + np.pi) % (2 * np.pi)) - np.pi

        # Check joint limits
        if np.any(q_solution < self.lower_limits) or np.any(q_solution > self.upper_limits):
            return None

        in_col = rrt_path_utils.check_links_in_collision(
            self.robot_id, self.joint_indices, q_solution,
            self.get_obstacle_ids(),
            verbose=True,
            obstacle_names=self.get_obstacle_names(),
            skip_pairs=self.get_skip_pairs(),
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

    # =========================================================================
    # Multi-Candidate Planning (for EE goals with multiple IK solutions)
    # =========================================================================

    def _plan_with_fallback_goals(
        self,
        q_start: np.ndarray,
        q_goal_candidates: List[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Try RRT planning with each goal candidate until one succeeds.

        Returns:
            Tuple of (trajectory, q_goal_used) or None if all candidates fail.
        """
        for i, q_goal in enumerate(q_goal_candidates):
            if not self.config.disable_camera_scoring_for_rrt:
                if self.camera_link_index is None:
                    raise ValueError(
                        "Camera scoring enabled but camera link not available. "
                        f"Check that wrist_camera_link_name='{self.wrist_camera_link_name}' exists in robot URDF."
                    )

                num_candidates = self.config.num_path_candidates
                max_attempts = self.config.max_path_attempts
                candidate_paths = self._generate_multiple_path_candidates(
                    q_start, q_goal, num_candidates, max_attempts
                )

                if len(candidate_paths) > 0:
                    target_position, _ = self._get_camera_link_pose(q_goal)
                    k_exp = self.config.k_exp
                    k_sig = self.config.k_sig
                    threshold = self.config.threshold

                    scored_paths = []
                    for j, path in enumerate(candidate_paths):
                        score = self._compute_camera_score(path, target_position, k_exp, k_sig, threshold)
                        scored_paths.append((score, j, path))

                    scored_paths.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_idx, best_path = scored_paths[0]
                    print(f"[TrajectoryGenerator] IK candidate {i}: best camera score = {best_score:.4f}")
                    return best_path, q_goal
            else:
                base_traj = self._plan_rrt_path(q_start, q_goal)
                if base_traj is not None:
                    print(f"[TrajectoryGenerator] RRT succeeded with IK candidate {i}")
                    return base_traj, q_goal

        print(f"[TrajectoryGenerator] RRT planning failed for all {len(q_goal_candidates)} IK goal candidate(s). See per-candidate diagnostics above.")
        return None

    def _plan_rrt_path(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        additional_obstacles: List[int] = None,
    ) -> Optional[np.ndarray]:
        """Plan collision-free path using RRT-Connect."""
        obstacles = self.get_obstacle_ids()
        if additional_obstacles:
            obstacles.extend(additional_obstacles)

        path = rrt_path_utils.get_path(
            q_start,
            q_goal,
            self.robot_id,
            self.joint_indices,
            obstacles,
            self.lower_limits,
            self.upper_limits,
            self.config.robot_update_rate,
            use_gui=self.config.debug_visualize,
            verbose=self.config.verbose,
            obstacle_names=self.get_obstacle_names(),
            skip_pairs=self.get_skip_pairs(),
        )

        # Snap endpoints to exact q_start/q_goal. Smoothing and resampling can
        # introduce small drift that shifts EE orientation at the goal.
        if path is not None:
            path[0] = q_start
            path[-1] = q_goal

        return path

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

    def _get_camera_link_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera link pose for a joint configuration.

        Args:
            q: Joint configuration (DOF,)

        Returns:
            position: Camera position in world frame (3,)
            rotation_matrix: Camera orientation as 3x3 rotation matrix
        """
        # Set robot to configuration q
        for idx, qi in zip(self.joint_indices, q):
            self.pb.resetJointState(self.robot_id, idx, qi)

        # Get camera link state
        link_state = self.pb.getLinkState(
            self.robot_id, self.camera_link_index, computeForwardKinematics=True
        )

        position = np.array(link_state[0])
        orientation_quat = np.array(link_state[1])  # [x, y, z, w]

        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(self.pb.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)

        return position, rotation_matrix

    def _compute_camera_score(
        self,
        path: np.ndarray,
        target_position: np.ndarray,
        k_exp: float,
        k_sig: float,
        threshold: float,
    ) -> float:
        """
        Compute camera-aware score for a trajectory path.

        Higher score = camera better aligned with target throughout path.
        Combines exponential reward with sigmoid gating.

        Args:
            path: Trajectory (N_SAMPLES, DOF)
            target_position: Target position in world frame (3,)
            k_exp: Exponential sharpness (default: 5.0)
            k_sig: Sigmoid sharpness (default: 15.0)
            threshold: Alignment threshold (default: 0.4)

        Returns:
            Average score across sampled waypoints
        """
        if self.camera_link_index is None:
            return 0.0  # Camera scoring unavailable

        # Sample waypoints (use 10 points max to reduce computation)
        num_samples = min(len(path), 10)
        sample_indices = np.linspace(0, len(path) - 1, num_samples, dtype=int)

        scores = []
        for idx in sample_indices:
            q = path[idx]

            # Get camera pose
            cam_position, cam_rotation = self._get_camera_link_pose(q)

            # Camera forward direction (assumes +Z axis in local frame)
            cam_forward = cam_rotation[:, 2]

            # Use utility function for single-timestep score
            waypoint_score = rrt_path_utils.compute_camera_alignment_score(
                cam_position, cam_forward, target_position, k_exp, k_sig, threshold
            )
            scores.append(waypoint_score)

        return float(np.mean(scores))

    def _generate_multiple_path_candidates(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        num_candidates: int,
        max_attempts: int,
        additional_obstacles: List[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate multiple candidate paths using RRT (adaptive approach).

        Attempts to generate num_candidates valid paths, up to max_attempts tries.
        Uses perturbations to start/goal configurations to ensure path diversity.

        Args:
            q_start: Start configuration
            q_goal: Goal configuration
            num_candidates: Target number of valid paths
            max_attempts: Maximum planning attempts
            additional_obstacles: Optional obstacle IDs

        Returns:
            List of valid paths (each: (N_SAMPLES, DOF))
        """
        paths = []
        attempts = 0

        # Perturbation magnitude (radians) - configurable via config
        perturbation_scale = self.RRT_PERTURBATION_SCALE

        while len(paths) < num_candidates and attempts < max_attempts:
            attempts += 1

            # First path uses exact start/goal, subsequent paths use perturbations
            if len(paths) == 0 and attempts == 0:
                plan_start = q_start
                plan_goal = q_goal
            else:
                # Add small random perturbations to force RRT to explore different paths
                start_perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=q_start.shape)
                goal_perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=q_goal.shape)

                # Clip to joint limits
                plan_start = np.clip(q_start + start_perturbation, self.lower_limits, self.upper_limits)
                plan_goal = np.clip(q_goal + goal_perturbation, self.lower_limits, self.upper_limits)

            path = self._plan_rrt_path(plan_start, plan_goal, additional_obstacles)

            if path is not None:
                # Ensure exact goal is included at the end. This is because we perturbed goal to get multiple path candidates
                if not np.allclose(path[-1], q_goal):
                    path = np.vstack([path, q_goal])
                paths.append(path)

                if self.config.verbose:
                    perturbation_info = "" if len(paths) == 1 else f", perturbed={perturbation_scale:.3f}"
                    print(f"Generated path {len(paths)}/{num_candidates} (attempt {attempts}/{max_attempts}{perturbation_info})")
                attempts = 0  # Reset attempts after a successful path

        if self.config.verbose and len(paths) < num_candidates:
            print(f"Warning: Only generated {len(paths)}/{num_candidates} valid paths after {max_attempts} attempts")

        return paths

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
