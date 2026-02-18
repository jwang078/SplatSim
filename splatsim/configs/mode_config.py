from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
    robot_update_rate: int = 20
    rrt_vis_fps: int = 10
    use_obstacles: bool = True
    q_start: Optional[List[float]] = None
    q_goal: Optional[List[float]] = None # ex: 7-dof-joint robot configuration goal
    ee_pos_goal: Optional[List[float]] = None    # [x, y, z] end-effector position goal
    ee_quat_goal: Optional[List[float]] = None   # [x, y, z, w] end-effector quaternion goal
    num_ik_candidates: int = 8                    # number of IK solutions to try for EE goals
    cuboids_fn: Optional[str] = None
    render_images: bool = False
    save_base_trajectory: bool = True
    disable_camera_scoring_for_rrt: bool = False
    num_path_candidates: int = 5
    max_path_attempts: int = 20
    k_exp: float = 5.0
    k_sig: float = 15.0
    threshold: float = 0.4
    lerobot_repo_id: str = ""
    push_to_hub: bool = False
    debug_visualize: bool = False  # Visualize q_start, q_goal, and trajectory in PyBullet GUI

    def __post_init__(self):
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
