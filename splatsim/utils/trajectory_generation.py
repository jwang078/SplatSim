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

    def __init__(
        self,
        pybullet_client,
        robot_id: int,
        joint_indices: list,
        config: dict,
        env_config_name: str,
        get_ee_link_fn,  # Callback to get EE link index
        splatsim_objects: List[SplatSimObject],
    ):
        """
        Initialize trajectory generator.

        Args:
            pybullet_client: PyBullet client instance
            robot_id: Robot body ID in PyBullet
            joint_indices: List of movable joint indices
            config: Trajectory generation configuration dictionary
            env_config_name: Environment name for output directory
            get_ee_link_fn: Function that returns EE link index
        """
        self.pb = pybullet_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.config = config
        self.env_config_name = env_config_name
        self.get_ee_link_fn = get_ee_link_fn
        self.splatsim_objects = splatsim_objects

        # Load joint limits
        self.lower_limits, self.upper_limits = rrt_path_utils.get_joint_limits(
            robot_id, joint_indices
        )

        # Static obstacles from environment (walls, table, etc.)
        self.loaded_obstacle_ids = []

        # Load cuboid obstacles if specified
        if config.get("use_obstacles") and config.get("cuboids_fn"):
            self.loaded_obstacle_ids = self._load_cuboid_obstacles(config["cuboids_fn"])

        # Initialize Zarr storage
        self.trajectory_count = 0
        self._init_zarr_storage()

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
        print(f"Resuming trajectory generation at index {self.trajectory_count}")

    def is_complete(self) -> bool:
        """Check if we've generated all requested trajectories."""
        return self.trajectory_count >= self.config["num_base_trajectories"]

    def generate_trajectory_batch(self):
        """Generate one base trajectory with multiple obstacle configurations."""
        # 1. Get start/goal configurations
        q_start = self.config.get("q_start")
        if q_start is None:
            q_start = self._get_random_collision_free_q()

        q_goal = self.config.get("q_goal")
        if q_goal is None:
            q_goal = self._get_random_collision_free_q()

        # 2. Generate base trajectory
        base_traj = self._plan_rrt_path(q_start, q_goal)
        if base_traj is None:
            return  # Failed, will retry next iteration

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
