import enum
from dataclasses import dataclass, field, asdict, fields
from typing import TYPE_CHECKING, List, Optional, Tuple, Any, Dict, get_origin, get_args, Union
import numpy as np
from pathlib import Path
import yaml
import copy
from abc import ABC

_SPLATSIM_ROOT = Path(__file__).resolve().parent.parent.parent





def Default(value: Any) -> Any:
    """Helper to define the global fallback value within the field metadata."""
    return field(default=None, metadata={'global_fallback': value})


class DebugModes(enum.Enum):
    """Debug modes for PybulletRobotServerBase."""
    OFF = "off"  # Normal operation, no debug features
    NO_BACKGROUND = "no_background"  # Use robot as background (no separate background splat)
    ROTATE_BASE_CAM = "rotate_base_cam"  # Allow rotating base camera via pybullet GUI


@dataclass
class GraspConfig:
    """Good grasp poses relative to this object"""
    grasp_pose: np.ndarray
    object_rot: np.ndarray

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass
class Transformation:
    matrix: np.ndarray
    
@dataclass
class AABB:
    """AABB config for this object"""
    bounding_box: np.ndarray
    urdf_bbox_adjustment: np.ndarray

@dataclass
class ArticulationConfig:
    # joint_states: Optional[np.ndarray] = Default(None)
    initial_joint_positions: List[float]
    # To handle joint direction conventions; initialize with placeholder
    joint_signs: List[int] = Default(None)
    # List of splat indices per joint
    segmented_list: Optional[List[List[int]]] = Default(None)
    # List of (pos, quat) per link at initial joint positions]]
    initial_link_poses: Optional[List[Tuple[List[float], List[float]]]] = Default(None)

@dataclass
class ObjectConfig(ABC):
    """Abstract base class fora config for an object in SplatSim"""
    name: str

    base_position: List[float] = Default(lambda: [0, 0, 0])
    base_quat: Tuple[float, float, float, float] = Default((0.0, 0.0, 0.0, 1.0))
    
    source_path: Optional[str] = Default(None)
    model_path: Optional[str] = Default(None)
    is_articulated: Optional[bool] = Default(False)
    articulation_config: Optional[ArticulationConfig] = Default(None)
    use_fixed_base: Optional[bool] = Default(False)
    scaling_range_x: Tuple[float, float] = Default((1.0, 1.0))
    scaling_range_y: Tuple[float, float] = Default((1.0, 1.0))
    scaling_range_z: Tuple[float, float] = Default((1.0, 1.0))
    wrist_camera_link_name: Optional[str] = Default(None)
    grasp_configs: Optional[List[GraspConfig]] = Default(list)
    load_splat: Optional[bool] = Default(True)
    load_urdf: Optional[bool] = Default(True)
    randomize_pose: Optional[bool] = Default(True)
    randomize_scale: Optional[bool] = Default(True)
    skip_collision_robot_links: Optional[List[int]] = Default(list)  # Robot link indices to skip when checking collisions against this object.
    use_aabb_collision: Optional[bool] = Default(False)  # If True, use fast AABB overlap test instead of PyBullet pairwise_collision for object-object checks.

    rotation_range_z: Tuple[float, float] = Default((0.0, 0.0))

    # Defaults to TABLE_LIMITS if not specified, otherwise uses the provided range for randomization
    position_range_x: Optional[Tuple[float, float]] = Default(None)
    position_range_y: Optional[Tuple[float, float]] = Default(None)
    position_range_z: Optional[Tuple[float, float]] = Default(None)

    # Live state — kept in sync with PyBullet each observation step
    current_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    current_quat: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    current_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Initial state — snapshotted at episode start after randomization
    initial_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_quat: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    initial_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


    _YAML_CACHE: Optional[Dict[str, Any]] = None
    _OBJECT_CONFIG_PATH: Path = _SPLATSIM_ROOT / "configs/object_configs/objects.yaml"

    def __post_init__(self):
        """Resolves Priority: Instance > YAML > Global Fallback"""
        if type(self) is ObjectConfig:
            raise TypeError("ObjectConfig is an abstract class.")
        
        yaml_data = self._load_yaml_config()
        
        for f in fields(self):
            if 'global_fallback' in f.metadata:
                current_val = getattr(self, f.name)
                
                if current_val is None:
                    # 1. Resolve raw value (YAML or Fallback)
                    if yaml_data is None or yaml_data.get(f.name) is None:
                        fallback = f.metadata['global_fallback']
                        val_to_set = fallback() if callable(fallback) and not isinstance(fallback, (str, tuple)) else copy.deepcopy(fallback)
                    else:
                        val_to_set = yaml_data.get(f.name)

                    # 2. AUTOMATIC HYDRATION
                    # Handle Optional[Type] by extracting the inner type
                    target_type = f.type
                    origin = get_origin(target_type)
                    if origin is Union: # Optional[T] is Union[T, None]
                        args = get_args(target_type)
                        # Pick the one that isn't NoneType
                        target_type = next((a for a in args if a is not type(None)), target_type)

                    # If the target type is a dataclass and we have a dict from YAML
                    from dataclasses import is_dataclass
                    if is_dataclass(target_type) and isinstance(val_to_set, dict):
                        # This turns the dict into the proper object automatically
                        val_to_set = target_type(**val_to_set)
                    
                    # 3. Final Assignment
                    setattr(self, f.name, val_to_set)

        # Validation
        if self.is_articulated and self.articulation_config is None:
            raise TypeError(f"is_articulated is True but articulation_config is missing for {self.name}")

    @classmethod
    def _get_yaml_data(cls) -> Dict[str, Dict[str, Any]]:
        """Loads the YAML file once and stores it in memory"""
        if cls._YAML_CACHE is None:
            if cls._OBJECT_CONFIG_PATH.exists():
                with open(cls._OBJECT_CONFIG_PATH, "r") as f:
                    cls._YAML_CACHE = yaml.safe_load(f) or {}
            else:
                cls._YAML_CACHE = {}
        if len(cls._YAML_CACHE) == 0:
            print(f"WARNING: Could not load object config file at {cls._OBJECT_CONFIG_PATH}")
        return cls._YAML_CACHE

    def _load_yaml_config(self) -> Dict[str, Any]:
        splat_name = getattr(self, "splat_name", None)
        if splat_name is None:
            # Fall back to object name
            splat_name = getattr(self, "object_name", None)
        if splat_name is None:
            return {}
        all_configs = self._get_yaml_data()
        if splat_name in all_configs:
            return all_configs[splat_name]
        

@dataclass
class CuboidObjectConfig(ObjectConfig):
    """Procedurally generated cuboid object."""
    size: Tuple[float, float, float] = Default((1.0, 1.0, 1.0))
    position: Tuple[float, float, float] = Default((0.0, 0.0, 0.0))
    mass: Optional[float] = Default(0.0)
    color_rgb: Tuple[int, int, int] = Default((0, 0, 255))
    randomize_pose: Optional[bool] = Default(False)
    use_aabb_collision: Optional[bool] = Default(True)  # Cuboids are axis-aligned boxes; AABB is exact.

    def to_dict(self) -> dict:
        return asdict(self)

# kw_only is python 3.10+
@dataclass(kw_only=True)
class SplatObjectConfig(ObjectConfig):
    """Splat-rendered object."""
    splat_name: str

    urdf_path: Optional[str] = Default(None)
    transformation: Optional[Transformation] = Default(None)
    aabb: AABB = Default(None)
    randomize_pose: Optional[bool] = Default(True)
    rotation_range_z: Tuple[float, float] = Default((0.0, 6.283))
    object_config: Optional[dict] = Default(None)
    keep_within_aabb: bool = Default(True)

    ply_path: Optional[str] = Default(None)
    model_path: Optional[str] = Default(None)
    source_path: Optional[str] = Default(None)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskConfig:
    """Task-specific goals and tolerances."""

    target_ee_pos: Tuple[float, float, float]
    target_ee_quat: Tuple[float, float, float, float]
    # Optional 6-DOF canonical goal joint config. When set, the trajectory
    # generator seeds IK from this config so demos converge to a shared joint
    # configuration when feasible (falling back to random-seed IK if blocked).
    q_goal_bias: Optional[Tuple[float, ...]] = None
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
    terminate_on_collision: bool = False

@dataclass
class SplatSimObject:
    # name: str
    # splat_name: str
    config: ObjectConfig
    sim_id: Optional[int] = None
    mass: float = 0.0 # Default to static object
    gaussians: Any = None
    # List of which link each point belongs to
    segmentation_labels: Optional[List[int]] = None
    # grasp_configs: List[dict] = field(default_factory=list)
    # object_config: dict[str, Any] = None # The format in configs/object_configs/objects.yaml
    transformations_cache: dict[Any] = None
    # is_articulated: bool = False # For example, the robot has is_articulated=True. An object with is_articulated should have articulation_config
    # articulation_config: Optional[ArticulationConfig] = None
    _cache: dict = field(default_factory=dict)  # Cache for GPU tensors to avoid recreating each step
