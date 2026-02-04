import enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union


class DebugModes(enum.Enum):
    """Debug modes for PybulletRobotServerBase."""
    OFF = "off"  # Normal operation, no debug features
    NO_BACKGROUND = "no_background"  # Use robot as background (no separate background splat)
    ROTATE_BASE_CAM = "rotate_base_cam"  # Allow rotating base camera via pybullet GUI


@dataclass
class CuboidObjectConfig:
    """Procedurally generated cuboid object."""

    object_name: str
    object_type: str = "cuboid"
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 0.0
    color_rgb: Tuple[int, int, int] = (128, 128, 128)
    randomize_pose: bool = False
    rotation_range_z: Tuple[float, float] = (0.0, 0.0)
    load_splat: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplatObjectConfig:
    """Splat-rendered object."""

    object_name: str
    splat_object_name: Optional[str] = None
    grasp_config: List = field(default_factory=list)
    randomize_pose: bool = True
    rotation_range_z: Tuple[float, float] = (0.0, 6.283)
    table_pos: Optional[Tuple[float, float]] = None
    table_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    load_splat: bool = True
    is_in_scene_splat: bool = False
    object_config: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


ObjectConfig = Union[CuboidObjectConfig, SplatObjectConfig]


@dataclass
class TaskConfig:
    """Task-specific goals and tolerances."""

    target_ee_pos: Tuple[float, float, float]
    target_ee_quat: Tuple[float, float, float, float]
    pos_tolerance_m: float = 0.03
    quat_tolerance_deg: float = 10.0
    task_description: str = ""


@dataclass
class EnvConfig:
    """Environment configuration."""

    name: str
    objects: List[ObjectConfig] = field(default_factory=list)
    task_description: str = ""
    task: Optional[TaskConfig] = None
