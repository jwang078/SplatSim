import enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


class ImageResizeMode(enum.Enum):
    LETTERBOX = "letterbox"
    STRETCH = "stretch"


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
    q_start: Optional[List[float] | np.ndarray] = None
    q_goal: Optional[List[float] | np.ndarray] = None # ex: 7-dof-joint robot configuration goal
    ee_pos_start: Optional[List[float] | np.ndarray] = None   # [x, y, z] end-effector position start
    ee_quat_start: Optional[List[float] | np.ndarray] = None  # [x, y, z, w] end-effector quaternion start
    ee_pos_goal: Optional[List[float] | np.ndarray] = None    # [x, y, z] end-effector position goal
    ee_quat_goal: Optional[List[float] | np.ndarray] = None   # [x, y, z, w] end-effector quaternion goal
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
