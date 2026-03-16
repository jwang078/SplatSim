"""Gymnasium wrapper for SplatSim environments."""

import threading
from typing import Any, Dict, List, Optional, Tuple, Type

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import torch

from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase


def _raw_obs_to_gym_obs(raw_obs: Dict[str, Any], num_dofs: int, camera_names: list, image_resize_modes: list) -> Dict[str, Any]:
    """Convert raw robot observations to gym format (agent_pos + pixels dict).

    Works with both SplatSim and real robot servers — any source that returns
    joint_positions, gripper_position, and {cam}_{mode} image keys.
    """
    joint_positions = np.array(raw_obs["joint_positions"][:num_dofs], dtype=np.float32)
    gripper = raw_obs.get("gripper_position", [0.0])
    if isinstance(gripper, (list, np.ndarray)):
        gripper = float(gripper[0]) if len(gripper) > 0 else 0.0

    agent_pos = np.concatenate([joint_positions, [gripper]]).astype(np.float32)

    pixels = {}
    for camera_name in camera_names:
        for mode in image_resize_modes:
            mode_str = mode.value if hasattr(mode, "value") else str(mode)
            key = f"{camera_name}_{mode_str}"
            img = raw_obs.get(key)
            if img is not None:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                img = (img * 255).clip(0, 255).astype(np.uint8)
                pixels[key] = img
            else:
                pixels[key] = np.zeros((224, 224, 3), dtype=np.uint8)

    gym_obs = {"agent_pos": agent_pos, "pixels": pixels}

    if "policy_guidance_action" in raw_obs:
        gym_obs["policy_guidance_action"] = np.array(raw_obs["policy_guidance_action"], dtype=np.float32)

    return gym_obs


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
    ):
        """Initialize the Gym environment.

        Args:
            robot_server: Instance of PybulletRobotServerBase or subclass
            render_mode: 'rgb_array' for pixel observations
        """
        super().__init__()
        self.robot_server = robot_server
        self.render_mode = render_mode
        self.action_space = robot_server.action_space
        self.observation_space = robot_server.observation_space
        self._max_episode_steps = robot_server._max_episode_steps

    def _to_gym_obs(self, raw_obs):
        gym_obs = _raw_obs_to_gym_obs(
            raw_obs,
            num_dofs=self.robot_server.num_dofs(),
            camera_names=self.robot_server.camera_names,
            image_resize_modes=self.robot_server.image_resize_modes,
        )
        # Remap pixels to match observation_space keys exactly.
        # PybulletRobotServerBase uses "{cam}_{mode}" keys; _ZMQBackend uses bare "{cam}" keys.
        declared_keys = set(self.observation_space["pixels"].spaces.keys())
        if declared_keys != set(gym_obs["pixels"].keys()):
            # Build a map from declared key -> best matching converted key
            remap = {}
            for dk in declared_keys:
                if dk in gym_obs["pixels"]:
                    remap[dk] = gym_obs["pixels"][dk]
                else:
                    # declared key is bare cam name; find matching "{cam}_{mode}" key
                    for ck, v in gym_obs["pixels"].items():
                        if ck.startswith(dk + "_"):
                            remap[dk] = v
                            break
            gym_obs["pixels"] = remap
        return gym_obs

    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        self.robot_server._physics_step(action)
        raw_obs = self.robot_server.get_observations()
        metrics = self.robot_server.check_metrics()
        is_success = metrics["is_success"]
        reward = self.robot_server.compute_reward_from_metrics(metrics)
        terminated = self.robot_server.check_terminated_from_metrics(metrics)
        truncated = self.robot_server._step_count >= self.robot_server._max_episode_steps
        info = {"is_success": is_success, "step_count": self.robot_server._step_count, **metrics}
        return self._to_gym_obs(raw_obs), reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        raw_obs, info = self.robot_server.reset(seed=seed, options=options)
        return self._to_gym_obs(raw_obs), info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode='rgb_array', None otherwise.
            If multiple *_rgb observations exist, they are stacked horizontally.
        """
        if self.render_mode == "rgb_array":
            obs = self.robot_server.get_observations()

            # Collect one RGB image per camera (first resize mode only, to avoid duplicates)
            rgb_images = []
            camera_names = getattr(self.robot_server, "camera_names", [])
            image_resize_modes = getattr(self.robot_server, "image_resize_modes", [])
            if camera_names and image_resize_modes:
                # Use the first resize mode to pick one image per camera
                first_mode = image_resize_modes[0]
                mode_str = first_mode.value if hasattr(first_mode, "value") else str(first_mode)
                keys_to_render = [f"{cam}_{mode_str}" for cam in camera_names]
            else:
                # Fallback: collect any key containing "_rgb", one per unique camera prefix
                seen_cameras = set()
                keys_to_render = []
                for key in sorted(obs.keys()):
                    if "_rgb" in key:
                        # Use the part up to the last underscore as the camera name
                        cam_prefix = key.rsplit("_", 1)[0]
                        if cam_prefix not in seen_cameras:
                            seen_cameras.add(cam_prefix)
                            keys_to_render.append(key)

            for key in keys_to_render:
                img = obs.get(key)
                if img is not None:
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
        if self.robot_server is not None:
            self.robot_server.stop()
            self.robot_server = None

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


def _is_port_available(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _get_next_port() -> int:
    """Get the next available port for a new environment instance.

    Thread-safe to support parallel environment creation.
    Skips occupied ports.
    """
    global _port_counter
    with _port_lock:
        while not _is_port_available(_port_counter):
            _port_counter += 1
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
        cfg["port"] = _get_next_port()

    robot_server = robot_server_cls(
        serve_mode=PybulletRobotServerBase.SERVE_MODES.INTERACTIVE,
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


class _ZMQBackend:
    """Duck-typed adapter so ZMQSplatSimGymEnv can reuse SplatSimGymEnv unchanged."""

    _max_episode_steps = 400

    def __init__(self, client, camera_names, image_resize_modes, num_dofs, image_height, image_width):
        self._client = client
        self._num_dofs = num_dofs
        self.camera_names = camera_names
        self.image_resize_modes = image_resize_modes
        self.action_space = spaces.Box(
            low=np.array([-np.pi] * num_dofs + [0.0], dtype=np.float32),
            high=np.array([np.pi] * num_dofs + [1.0], dtype=np.float32),
        )
        # observation_space uses bare camera names (no mode suffix) to match lerobot's features_map.
        # policy_guidance_action is always declared; NaN when no guidance process is active.
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(num_dofs + 1,), dtype=np.float32),
            "pixels": spaces.Dict({
                cam: spaces.Box(low=0, high=255, shape=(image_height, image_width, 3), dtype=np.uint8)
                for cam in camera_names
            }),
            "policy_guidance_action": spaces.Box(low=-np.inf, high=np.inf, shape=(num_dofs + 1,), dtype=np.float32),
        })

    def _get_policy_guidance(self, raw_obs):
        pga = raw_obs.get("policy_guidance_action")
        if pga is not None:
            return np.array(pga, dtype=np.float32)
        return np.full(self._num_dofs + 1, np.nan, dtype=np.float32)

    def step(self, action):
        self._client.command_joint_state(action)
        raw_obs = self._client.get_observations()
        raw_obs["policy_guidance_action"] = self._get_policy_guidance(raw_obs)
        metrics = self._client.get_metrics() if hasattr(self._client, "get_metrics") else {"is_success": False}
        terminated = metrics.get("is_success", False)
        return raw_obs, float(terminated), terminated, False, metrics

    def reset(self, seed=None, options=None):
        raw_obs, info = self._client.reset(seed=seed, options=options)
        raw_obs["policy_guidance_action"] = self._get_policy_guidance(raw_obs)
        return raw_obs, info

    def num_dofs(self):                return self._num_dofs
    def get_observations(self):        return self._client.get_observations()
    def get_task_description(self):    return ""
    def stop(self):                    self._client.close()


class ZMQSplatSimGymEnv(SplatSimGymEnv):
    """SplatSimGymEnv backed by an already-running simulator connected via ZMQ.

    Use via external_port in the lerobot config when the simulator is launched
    separately (e.g. alongside gello for shared-autonomy eval).
    """

    def __init__(self, host, port, camera_names, image_resize_modes,
                 num_dofs=6, image_height=224, image_width=224, render_mode=None):
        from gello.zmq_core.robot_node import ZMQClientRobot

        backend = _ZMQBackend(
            ZMQClientRobot(port=port, host=host),
            camera_names, image_resize_modes, num_dofs, image_height, image_width,
        )
        super().__init__(robot_server=backend, render_mode=render_mode)  # type: ignore[arg-type]

    def step(self, action: np.ndarray):
        """Send joint command and read back observations asynchronously."""
        raw_obs, reward, terminated, truncated, info = self.robot_server.step(action)  # type: ignore[union-attr]
        return self._to_gym_obs(raw_obs), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super(SplatSimGymEnv, self).reset(seed=seed)
        raw_obs, info = self.robot_server.reset(seed=seed, options=options)  # type: ignore[union-attr]
        return self._to_gym_obs(raw_obs), info


def list_envs() -> list:
    """List all available environment names."""
    if not ENV_REGISTRY:
        _populate_registry()
    return list(ENV_REGISTRY.keys())
