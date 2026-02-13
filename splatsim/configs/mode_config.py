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
    q_goal: Optional[List[float]] = field(
        default_factory=lambda: [1.33936567, -1.52838483, 1.92282924, -1.21754169, -0.53407075, -0.73042029]
    )
    cuboids_fn: Optional[str] = None
    render_images: bool = False
    save_base_trajectory: bool = True
    disable_camera_scoring_for_rrt: bool = False
    num_path_candidates: int = 5
    max_path_attempts: int = 20
    k_exp: float = 5.0
    k_sig: float = 15.0
    threshold: float = 0.4
    experiment_name: str = ""
