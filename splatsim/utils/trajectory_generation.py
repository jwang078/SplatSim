import numpy as np
import zarr
import os
import re
import json
from typing import Optional, Tuple, List
from splatsim.utils import rrt_path_utils
from splatsim.utils.robot_splat_render_utils import SplatSimObject


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
        num_base_trajectories: int = 100,
        obstacles_per_base_trajectory: int = 0,
        paths_per_obstacle: int = 0,
        min_obstacles: int = 1,
        max_obstacles: int = 3,
        max_fails: int = 2,
        max_obstacle_fails_per_base_traj: int = 20,
        time_per_traj: float = 6.0,
        robot_update_rate: int = 20,
        rrt_vis_fps: int = 10,
        use_obstacles: bool = True,
        q_start: Optional[np.ndarray] = None,
        q_goal: Optional[np.ndarray] = np.array([1.33936567, -1.52838483, 1.92282924, -1.21754169, -0.53407075, -0.73042029]),
        cuboids_fn: Optional[str] = None,
        render_images: bool = False,
        save_base_trajectory: bool = True,
        disable_camera_scoring_for_rrt: bool = False,
        num_path_candidates: int = 5,
        max_path_attempts: int = 20,
        k_exp: float = 5.0,
        k_sig: float = 15.0,
        threshold: float = 0.4,
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
            (other kwargs): Trajectory generation configuration parameters
        """
        self.pb = pybullet_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices

        # Store all config parameters
        self.config = {
            "num_base_trajectories": num_base_trajectories,
            "obstacles_per_base_trajectory": obstacles_per_base_trajectory,
            "paths_per_obstacle": paths_per_obstacle,
            "min_obstacles": min_obstacles,
            "max_obstacles": max_obstacles,
            "max_fails": max_fails,
            "max_obstacle_fails_per_base_traj": max_obstacle_fails_per_base_traj,
            "time_per_traj": time_per_traj,
            "robot_update_rate": robot_update_rate,
            "rrt_vis_fps": rrt_vis_fps,
            "use_obstacles": use_obstacles,
            "q_start": q_start,
            "q_goal": q_goal,
            "cuboids_fn": cuboids_fn,
            "render_images": render_images,
            "save_base_trajectory": save_base_trajectory,
            "disable_camera_scoring_for_rrt": disable_camera_scoring_for_rrt,
            "num_path_candidates": num_path_candidates,
            "max_path_attempts": max_path_attempts,
            "k_exp": k_exp,
            "k_sig": k_sig,
            "threshold": threshold,
            "experiment_name": "",
        }
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
        if self.config.get("use_obstacles") and self.config.get("cuboids_fn"):
            self.loaded_obstacle_ids = self._load_cuboid_obstacles(self.config["cuboids_fn"])

        # Zarr storage (initialized lazily when start_generation() is called)
        self.trajectory_count = 0
        self.zarr_root = None
        self.scenarios_group = None

    def get_obstacle_ids(self) -> List[int]:
        return self.loaded_obstacle_ids + [
            obj.sim_id for obj in self.splatsim_objects if obj.sim_id is not None and obj.name != "robot"
        ]

    def register_obstacle(self, sim_id: int):
        """Register an obstacle by its PyBullet body ID."""
        # This is a temporary function. Every obstacle should ideally be a splatsim object
        if sim_id not in self.loaded_obstacle_ids:
            self.loaded_obstacle_ids.append(sim_id)

    def _load_cuboid_obstacles(self, cuboids_fn: str) -> List[int]:
        """Load cuboid obstacles from NPZ file and create PyBullet bodies."""
        from pybullet_planning import create_box, set_pose, Pose, RED

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
        experiment_name = self.config.get("experiment_name", "")
        if experiment_name:
            output_dir = os.path.join("output", f"{self.env_config_name}_{experiment_name}_trajectories.zarr")
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
        return self.trajectory_count >= self.config["num_base_trajectories"]

    def generate_trajectory_batch(self):
        """Generate one base trajectory with multiple obstacle configurations."""
        # Take into account any folder naming from experiment_name
        self._init_zarr_storage()

        # 1. Get start/goal configurations
        q_start = self.config.get("q_start")
        if q_start is None:
            q_start = self._get_random_collision_free_q()
        else:
            q_start = np.array(q_start)

        q_goal = self.config.get("q_goal")
        if q_goal is None:
            q_goal = self._get_random_collision_free_q()
        else:
            q_goal = np.array(q_goal)


        # 2. Generate base trajectory (with optional camera scoring)
        if not self.config.get("disable_camera_scoring_for_rrt", False):
            # Camera-aware mode: verify camera link is available BEFORE generating paths
            if self.camera_link_index is None:
                raise ValueError(
                    "Camera scoring enabled (disable_camera_scoring_for_rrt=False) but camera link not available. "
                    f"Check that wrist_camera_link_name='{self.wrist_camera_link_name}' exists in robot URDF."
                )

            # Generate multiple candidates and select best
            num_candidates = self.config.get("num_path_candidates", 5)
            max_attempts = self.config.get("max_path_attempts", 20)

            candidate_paths = self._generate_multiple_path_candidates(
                q_start, q_goal, num_candidates, max_attempts
            )

            if len(candidate_paths) == 0:
                return  # Failed, will retry next iteration

            # Compute target position (camera pose at goal)
            target_position, _ = self._get_camera_link_pose(q_goal)

            # Score all candidates
            k_exp = self.config.get("k_exp", 5.0)
            k_sig = self.config.get("k_sig", 15.0)
            threshold = self.config.get("threshold", 0.4)

            scored_paths = []
            for i, path in enumerate(candidate_paths):
                score = self._compute_camera_score(path, target_position, k_exp, k_sig, threshold)
                scored_paths.append((score, i, path))

                if self.config.get("verbose", False):
                    print(f"Path {i}: score = {score:.4f}")

            # Select path with highest score
            scored_paths.sort(key=lambda x: x[0], reverse=True)
            best_score, best_idx, base_traj = scored_paths[0]
            print("All scores:", [score for score, _, _ in scored_paths])

            if self.config.get("verbose", False):
                print(f"Selected path {best_idx} with score {best_score:.4f}")
        else:
            # Standard mode: generate single path
            base_traj = self._plan_rrt_path(q_start, q_goal)

            if base_traj is None:
                return  # Failed, will retry next iteration

        # 2b. Extend the path so that it stays at the last position for a second
        num_extra_steps = int(1 * self.config["robot_update_rate"])
        last_q = base_traj[-1]
        extra_steps = np.tile(last_q, (num_extra_steps, 1))
        base_traj = np.vstack((base_traj, extra_steps))

        # 3. Get EE trajectory for obstacle placement
        base_ee_traj = self._get_ee_trajectory(base_traj)

        # 4. Save base trajectory (no obstacles) if configured
        if self.config.get("save_base_trajectory", True):
            self._save_trajectory_zarr(
                base_traj, self.trajectory_count, 0, 0, {"obstacles": []}
            )

        # 5. Generate trajectories with obstacles
        if self.config["obstacles_per_base_trajectory"] > 0:
            for obstacle_i in range(
                1, self.config["obstacles_per_base_trajectory"] + 1
            ):
                # Add random obstacles
                obstacle_ids, obstacle_infos = self._add_random_obstacles(
                    [q_start, q_goal], base_ee_traj
                )

                # Generate multiple paths per obstacle configuration
                for path_i in range(self.config["paths_per_obstacle"]):
                    modified_traj = self._plan_rrt_path(
                        q_start, q_goal, additional_obstacles=obstacle_ids
                    )

                    if modified_traj is not None:
                        # Extend the path so that it stays at the last position for a second
                        num_extra_steps = int(1 * self.config["robot_update_rate"])
                        last_q = modified_traj[-1]
                        extra_steps = np.tile(last_q, (num_extra_steps, 1))
                        modified_traj = np.vstack((modified_traj, extra_steps))

                        self._save_trajectory_zarr(
                            modified_traj,
                            self.trajectory_count,
                            obstacle_i,
                            path_i,
                            {"obstacles": obstacle_infos},
                        )

                # Remove obstacles for next iteration
                self._remove_obstacles(obstacle_ids)

        self.trajectory_count += 1

    def _get_random_collision_free_q(self) -> np.ndarray:
        """Generate random collision-free joint configuration."""
        return rrt_path_utils.get_random_joint_angles_without_collision(
            self.robot_id,
            self.joint_indices,
            self.get_obstacle_ids(),
            self.lower_limits,
            self.upper_limits,
            max_tries=10000,
            verbose=self.config.get("verbose", False),
        )

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

        return rrt_path_utils.get_path(
            q_start,
            q_goal,
            self.robot_id,
            self.joint_indices,
            obstacles,
            self.lower_limits,
            self.upper_limits,
            self.config["time_per_traj"],
            self.config["robot_update_rate"],
            rrt_vis_fps=self.config.get("rrt_vis_fps", 10),
            use_gui=False,
            verbose=self.config.get("verbose", False),
        )

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
            if len(paths) == 0:
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
                paths.append(path)

                if self.config.get("verbose", False):
                    perturbation_info = "" if len(paths) == 1 else f", perturbed={perturbation_scale:.3f}"
                    print(f"Generated path {len(paths)}/{num_candidates} (attempt {attempts}/{max_attempts}{perturbation_info})")

        if self.config.get("verbose", False) and len(paths) < num_candidates:
            print(f"Warning: Only generated {len(paths)}/{num_candidates} valid paths after {max_attempts} attempts")

        return paths

    def _add_random_obstacles(
        self, robot_qs_to_avoid: List[np.ndarray], base_ee_traj: np.ndarray
    ) -> Tuple[List[int], List[dict]]:
        """Add random cuboid obstacles avoiding robot configurations."""
        from pybullet_planning import create_box, set_pose, Pose, BLUE

        # Adapted from test_traj_refinement.py add_random_obstacles
        new_obj_ids = []
        new_obj_infos = []
        num_obstacles = np.random.randint(
            self.config["min_obstacles"], self.config["max_obstacles"] + 1
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
        """Save trajectory to Zarr format."""
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

        # TODO: If render_images is True, execute trajectory and save observations
        if self.config.get("render_images", False):
            # Call back to base class method to execute and record
            pass
