"""Gymnasium wrapper for SplatSim environments."""

import threading
from typing import Any, Dict, Optional, Type

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np

from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase


# Thread-safe counter for assigning unique ports to each environment instance
_port_counter = 5556
_port_lock = threading.Lock()


class SplatSimGymEnv(gym.Env):
    """Thin Gymnasium wrapper - delegates all logic to robot_server.

    Observation space is a Dict with:
        - "state": Box(7,) for joint positions + gripper
        - "base_rgb": Box(3, 224, 224) if enabled
        - "wrist_rgb": Box(3, 224, 224) if enabled
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        robot_server: PybulletRobotServerBase,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 300,
    ):
        """Initialize the Gym environment.

        Args:
            robot_server: Instance of PybulletRobotServerBase or subclass
            render_mode: 'rgb_array' for pixel observations
            max_episode_steps: Maximum steps per episode (default 400, matching Aloha at 50fps = 8 seconds)
        """
        super().__init__()
        self.robot_server = robot_server
        self.render_mode = render_mode
        self.action_space = robot_server.action_space
        self.observation_space = robot_server.observation_space
        self._max_episode_steps = max_episode_steps

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        return self.robot_server.step(action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        return self.robot_server.reset(seed=seed, options=options)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode='rgb_array', None otherwise.
            If multiple *_rgb observations exist, they are stacked horizontally.
        """
        if self.render_mode == "rgb_array":
            obs = self.robot_server.get_observations()

            # Collect all RGB observations
            rgb_images = []
            for key in sorted(obs.keys()):
                if key.endswith("_rgb") and obs[key] is not None:
                    img = obs[key]
                    if hasattr(img, 'cpu'):
                        img = img.cpu().numpy()
                    # Convert from (C, H, W) -> (H, W, C) for rendering
                    img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
                    rgb_images.append(img)

            if rgb_images:
                # Pad images to the same height if needed
                max_height = max(img.shape[0] for img in rgb_images)
                padded_images = []
                for img in rgb_images:
                    if img.shape[0] < max_height:
                        pad_height = max_height - img.shape[0]
                        padding = np.zeros((pad_height, img.shape[1], img.shape[2]), dtype=img.dtype)
                        img = np.concatenate([img, padding], axis=0)
                    padded_images.append(img)
                # Stack all RGB images horizontally
                return np.concatenate(padded_images, axis=1)
        return None

    def close(self):
        """Clean up resources."""
        pass

    @property
    def task_description(self) -> str:
        """Return the task description for LeRobot compatibility.

        This is used by language-conditioned policies (e.g., PI0, PI05) to condition
        the action prediction on a natural language task description.
        """
        return self.robot_server.get_task_description()

    @property
    def unwrapped(self) -> PybulletRobotServerBase:
        """Return the underlying robot server."""
        return self.robot_server


# Registry of available environments
ENV_REGISTRY: Dict[str, Type[PybulletRobotServerBase]] = {}


def register_env(name: str, cls: Type[PybulletRobotServerBase]):
    """Register an environment class with a name."""
    ENV_REGISTRY[name] = cls


def _populate_registry():
    """Populate registry with supported environments."""
    from splatsim.robots.sim_robot_pybullet_small_engine import (
        SmallEnginePybulletRobotServer,
        UprightRobotSmallEngineNewPybulletRobotServer,
    )
    from splatsim.robots.sim_robot_pybullet_object_on_plate import (
        ObjectOnPlatePybulletRobotServer,
        AppleOnPlatePybulletRobotServer,
        BananaOnPlatePybulletRobotServer,
        OrangeOnPlatePybulletRobotServer,
    )
    from splatsim.robots.sim_robot_pybullet_robot_in_bwa import (
        BWAPybulletRobotServer,
        OpenSpaceBWAPybulletRobotServer,
    )

    register_env("small_engine", SmallEnginePybulletRobotServer)
    register_env("upright_small_engine_new", UprightRobotSmallEngineNewPybulletRobotServer)
    register_env("object_on_plate", ObjectOnPlatePybulletRobotServer)
    register_env("apple_on_plate", AppleOnPlatePybulletRobotServer)
    register_env("banana_on_plate", BananaOnPlatePybulletRobotServer)
    register_env("orange_on_plate", OrangeOnPlatePybulletRobotServer)
    register_env("bwa", BWAPybulletRobotServer)
    register_env("open_space_bwa", OpenSpaceBWAPybulletRobotServer)


def _get_next_port() -> int:
    """Get the next available port for a new environment instance.

    Thread-safe to support parallel environment creation.
    """
    global _port_counter
    with _port_lock:
        port = _port_counter
        _port_counter += 1
    return port


def make_single_env(
    env_name: str,
    cfg: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
    port: Optional[int] = None,
) -> SplatSimGymEnv:
    """Create a single SplatSim Gym environment.

    Args:
        env_name: Name of environment (must be in ENV_REGISTRY)
        cfg: Configuration dict passed to robot server constructor.
        render_mode: 'rgb_array' or None
        port: ZMQ server port. If None, auto-assigns a unique port.

    Returns:
        SplatSimGymEnv instance
    """
    if not ENV_REGISTRY:
        _populate_registry()

    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )

    cfg = cfg or {}

    robot_server_cls = ENV_REGISTRY[env_name]

    # Auto-assign port if not specified
    if port is None:
        port = _get_next_port()

    # Create robot server with config
    robot_server = robot_server_cls(
        serve_mode=PybulletRobotServerBase.SERVE_MODES.INTERACTIVE,
        port=port,
        **cfg
    )

    # Wrap in Gym interface
    return SplatSimGymEnv(robot_server, render_mode=render_mode)


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
    base_port: Optional[int] = None,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """Create vectorized SplatSim environments for LeRobot.

    Args:
        n_envs: Number of parallel environments per task
        use_async_envs: Use AsyncVectorEnv (subprocess) vs SyncVectorEnv
        cfg: Configuration dict with:
            - env_names: List of environment names to create
            - render_mode: Optional render mode
            - Additional kwargs passed to robot servers
        base_port: Starting port for ZMQ servers. Each env gets base_port + i.
            If None, auto-assigns unique ports.

    Returns:
        Nested dict: {task_name: {env_index: VectorEnv}}
        For single-task: {"default": {0: VectorEnv}}
    """
    if not ENV_REGISTRY:
        _populate_registry()

    cfg = cfg or {}
    env_names = cfg.pop("env_names", list(ENV_REGISTRY.keys())[:1])
    render_mode = cfg.pop("render_mode", None)

    if isinstance(env_names, str):
        env_names = [env_names]

    VecEnvCls = AsyncVectorEnv if use_async_envs else SyncVectorEnv

    result: Dict[str, Dict[int, gym.vector.VectorEnv]] = {}

    env_idx = 0
    for env_name in env_names:
        # Create factory functions with unique ports for each env
        env_fns = []
        for i in range(n_envs):
            # Calculate port for this specific environment instance
            if base_port is not None:
                port = base_port + env_idx
            else:
                port = None  # Will auto-assign in make_single_env

            # Capture variables in closure
            def make_env_fn(name=env_name, c=cfg.copy(), rm=render_mode, p=port):
                return make_single_env(name, cfg=c, render_mode=rm, port=p)

            env_fns.append(make_env_fn)
            env_idx += 1

        # Create vectorized environment
        vec_env = VecEnvCls(env_fns)

        result[env_name] = {0: vec_env}

    return result


def list_envs() -> list:
    """List all available environment names."""
    if not ENV_REGISTRY:
        _populate_registry()
    return list(ENV_REGISTRY.keys())
