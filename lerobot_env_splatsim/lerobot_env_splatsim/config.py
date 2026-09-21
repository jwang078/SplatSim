"""The `splatsim` LeRobot environment config (moved here from the lerobot fork's envs/configs.py)."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("splatsim")
@dataclass
class SplatSimEnv(EnvConfig):
    """Configuration for SplatSim Gym environment.

    SplatSim is a Gaussian splatting-based robot simulation environment.
    The actual Gym environment is defined in the splatsim package and must be
    registered with gymnasium before use.

    Example usage:
        lerobot-eval \\
            --policy.path=your/checkpoint \\
            --env.type=splatsim \\
            --env.task=upright_small_engine_new \\
            --eval.n_episodes=50
    """

    task: str = "upright_small_engine_new"
    fps: int = 30
    episode_length: int = 400  # 8 seconds at 50 fps, matching Aloha
    render_mode: str = "rgb_array"

    # Task description for language-conditioned policies (e.g., PI0, PI05)
    # This should match the task description used during training
    task_description: str | None = None

    # SplatSim-specific config
    # Default aligned with SmallEnginePybulletRobotServer's current
    # `background_splat_name` — the robot splat + scene splat must come from
    # the same Gaussian training session or the wrist camera renders into
    # empty space (correct pose in the wrong coordinate frame). If you swap
    # scene splats on the SplatSim side, update both here and every other
    # `robot_iphone_w_engine_*` default in this repo.
    robot_name: str = "robot_iphone_w_engine_curtain"
    cam_i: int = 3
    camera_names: list[str] = field(default_factory=lambda: ["base_rgb", "wrist_rgb"])
    use_gripper: bool = True
    debug_mode: str = "off"

    port: int | None = None

    # Run in eval_benchmark mode: restore pre-recorded episode scenarios on each reset().
    # Set to the LeRobot repo ID (e.g. "user/my-eval-dataset") of the dataset whose
    # episode scenarios should be cycled through. Each env.reset() advances to the next episode.
    eval_benchmark_repo_id: str | None = None

    # Optional subset of episode indices to evaluate in eval_benchmark mode.
    # If None, all episodes in the dataset are used (0..N-1).
    # Example: [3, 8, 23, 38] to evaluate only those episodes from the benchmark dataset.
    eval_benchmark_subset: list[int] | None = None

    # Connect to an already-running SplatSim server instead of launching a new one.
    # When set, lerobot-eval uses ZMQSplatSimGymEnv on this port rather than
    # spawning a PybulletRobotServerBase. Useful for shared-autonomy eval where
    # gello is also connected to the same running simulator.
    external_port: int | None = None
    external_host: str = "127.0.0.1"

    # When True, the in-process PybulletRobotServerBase connects via p.DIRECT
    # instead of p.GUI — no pybullet visualizer window. Has no effect when
    # `external_port` is set, since that path uses ZMQSplatSimGymEnv (no local
    # pybullet client) and the external sim's GUI mode is controlled by ITS
    # own --headless flag at launch (see SplatSim's scripts/launch_nodes.py).
    # Wired by `dagger_orchestrate.sh --headless` for fast batch runs.
    headless: bool = False

    # When True (paired with `headless`), the in-process robot server still
    # launches its Tkinter "SplatSim Controls" panel: pybullet stays DIRECT
    # (fast, no 3D window) but the control panel is available for mode picking
    # / live tuning. Maps to the server ctor's `show_control_gui` — the same
    # surface launch_nodes.py's --control_gui flag drives for external sims.
    # Only injected when True so server classes predating the kwarg keep
    # working. No effect in non-headless mode (GUI already shows the panel)
    # or when `external_port` is set (the external sim owns its GUI mode).
    control_gui: bool = False

    # When True, the in-process robot server composites PyBullet-computed
    # shadows onto its Gaussian-splat renders (depth cue: the arm casts a
    # visible shadow onto the table/objects). Maps to the server ctor's
    # `splat_shadows` — the same surface launch_nodes.py's --splat_shadows
    # flag drives for external sims. Only injected when True so server
    # classes predating the kwarg keep working. NOTE: this CHANGES the
    # images the policy sees, so eval imagery only matches training imagery
    # when the recording sim used the same setting — keep it consistent
    # across a lineage (dagger_orchestrate.sh --splat_shadows sets every
    # phase at once). No effect when `external_port` is set (the external
    # sim owns its own rendering config).
    splat_shadows: bool = False

    # When True, the gym env exposes get_env_config() so the policy / wrapper can
    # access obstacle geometry and the task goal (q_goal_bias, target_ee_pos/quat).
    # Required for the shared autonomy wrapper's "RRT to Goal" mode.
    include_oracle_info: bool = False

    # When True, end the episode the first step `info["in_collision"]` is
    # true. SplatSim's underlying env already publishes `in_collision` in
    # the info dict on every step (it's the same predicate used by the
    # intervention controller's collision trigger), so this wrapper just
    # forces gymnasium's `terminated=True` based on that. Default False
    # preserves historical behavior (episodes run until success / truncate
    # / out-of-bounds). Use case: cleaner eval metrics ("success rate AT
    # FIRST COLLISION" vs "success rate within episode_length steps")
    # without changing intervention recording semantics — intervention
    # recording leaves this False so the collision trigger handles it,
    # while training-time eval can set it True.
    terminate_on_collision: bool = False

    # Wrist camera model version (see WRIST_CAM_FISHEYE_CALIBRATIONS in
    # splatsim/robots/sim_robot_pybullet_base.py):
    #   0 = pinhole using base camera intrinsics (matches pre-fisheye datasets)
    #   1 = fisheye, original 2704x2028 GoPro calibration
    #   2 = fisheye, recalibrated 1920x1080 GoPro calibration (default)
    wrist_cam_ver: int = 2

    # Teleop recording: save pure-teleop (ratio=0) segments to a LeRobot dataset.
    # Set to a repo ID (e.g. "user/teleop-data") to enable; None to disable.
    teleop_dataset_repo_id: str | None = None
    teleop_min_episode_length: int = 60  # discard segments shorter than this
    # When True (default, backward-compat), TeleopRecordingWrapper.close() pushes
    # the finalized dataset to HuggingFace Hub. Set False to keep the dataset
    # local-only (used by dagger_orchestrate.sh's offline mode to avoid round-
    # tripping each round's intervention dataset through the Hub).
    teleop_push_to_hub: bool = True
    # Short-episode behavior. True (default, legacy): episodes shorter than
    # `teleop_min_episode_length` get padded by repeating the last committed
    # frame. False: drop those episodes entirely instead of padding. The
    # padded frames are exact repeats (state diff = 0, action = last
    # commanded), so they train the policy with ~min_episode_length samples
    # of `obs = (near_goal, near_goal) → action = hold`, which biases the
    # diffusion score field toward "freeze when close to goal" — manifests
    # at eval as the policy stopping a few cm short of goal. For DAgger /
    # intervention recording, set this False (pass
    # `--env.teleop_pad_short_episodes=false` in --intervention_extra_args).
    teleop_pad_short_episodes: bool = True
    # Recorder-side state-discontinuity threshold (joint-L2 of Δstate
    # between consecutive real frames, radians). When exceeded,
    # TeleopRecordingWrapper finalizes the current episode and starts a
    # fresh one — regardless of whether the upstream RRT source signaled
    # a teleport.
    #
    # The source-side signal (`force_episode_split_next_real_frame`, set
    # by `_teleport_env_to_q_start`) only covers LEROBOT-driven env
    # teleports (lookback rewind, escape from q_start-in-collision,
    # request_retry_after_collision). It cannot cover PyBullet's
    # constraint-solver position corrections — those fire when a robot
    # link physically penetrates an obstacle during time-parametrized RRT
    # execution (env-physics, never touches our code), and produce
    # multi-rad state jumps with no source-side hook to signal them.
    # The recorder-side threshold check IS the only mechanism that can
    # split on those.
    #
    # Default 0.15 rad/frame sits between parametrizer-bounded RRT motion
    # (~0.1 rad/frame max) and typical teleport / collision-correction
    # magnitudes (~0.3-3 rad). Set to 0 to disable (source-side signal
    # still applies). Tighten if your recorded data still shows
    # within-episode jumps; loosen if normal fast motion false-splits.
    teleop_state_jump_split_threshold_rad: float = 0.15

    # Image dimensions
    observation_height: int = 224
    observation_width: int = 224

    # Image resize mode:
    # - "letterbox": Resize keeping aspect ratio, pad with black bars (good for pretrained VLAs)
    # - "stretch": Resize to fill entire area without keeping aspect ratio (good for diffusion)
    image_resize_modes: list[str] = field(default_factory=lambda: ["letterbox"])

    # Arm DOF count. For the ZMQ (external_port) path this sizes the gym
    # action/observation spaces; the in-process path reads it from the spawned
    # server. Default 6 (UR5); set 3 for the planar arm. state/action dims are
    # num_dofs + 1 (gripper) — keep the three consistent (the env profiles /
    # train_sweep.sh derive state_dim = action_dim = num_dofs + 1).
    num_dofs: int = 6
    # State dimension (num_dofs joints + 1 gripper). Default 7 (UR5).
    state_dim: int = 7
    # Action dimension (num_dofs joints + 1 gripper). Default 7 (UR5).
    action_dim: int = 7
    # Environment-state dimension: width of a SEPARATE observation.environment_state
    # feature (FeatureType.ENV) holding privileged world state (object coords) for
    # oracle/state-only policies. 0 → no environment_state feature (default).
    env_state_dim: int = 0

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
            "pixels/base_rgb": f"{OBS_IMAGES}.base_rgb",
            "pixels/wrist_rgb": f"{OBS_IMAGES}.wrist_rgb",
        }
    )

    def __post_init__(self):
        # Set state + action features from the configured dims (the default
        # `features` factory hardcodes 7 — override so non-UR5 arms like the
        # 3-joint planar arm get the right shapes, e.g. state_dim/action_dim=4).
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))

        # Privileged world state (object coords) as a distinct FeatureType.ENV
        # input for oracle/state-only policies (e.g. the diffusion policy, which
        # requires an image OR an environment_state). Omitted when env_state_dim=0.
        if self.env_state_dim > 0:
            self.features["environment_state"] = PolicyFeature(
                type=FeatureType.ENV, shape=(self.env_state_dim,)
            )
        else:
            self.features.pop("environment_state", None)

        # Set image features - always use "pixels/<camera_name>" format for consistency
        # This maps to "observation.images.<camera_name>" in LeRobot format
        for cam_name in self.camera_names:
            self.features[f"pixels/{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        # When teleop recording is configured, force the SplatSim server to
        # render every ImageResizeMode regardless of what the policy itself
        # uses. This way the saved dataset always has every variant
        # ({cam}_letterbox, {cam}_stretch, ...) available for downstream
        # training of policies that may want a different resize mode. The
        # policy's preprocessor (rename_map) still picks whichever one it
        # was trained on.
        server_image_resize_modes = self.image_resize_modes
        if self.teleop_dataset_repo_id is not None:
            from splatsim.configs.mode_config import ImageResizeMode

            server_image_resize_modes = [m.value for m in ImageResizeMode]

        cfg = {
            "robot_name": self.robot_name,
            "camera_names": self.camera_names,
            "cam_i": self.cam_i,
            "use_gripper": self.use_gripper,
            "debug_mode": self.debug_mode,
            "image_resize_modes": server_image_resize_modes,
            "port": self.port,
            "wrist_cam_ver": self.wrist_cam_ver,
        }
        # Include task_description if provided (for language-conditioned policies)
        if self.task_description is not None:
            cfg["task_description"] = self.task_description
        # Pass eval_benchmark_repo_id so the robot server loads the dataset on startup
        if self.eval_benchmark_repo_id is not None:
            cfg["eval_benchmark_repo_id"] = self.eval_benchmark_repo_id
        if self.eval_benchmark_subset is not None:
            cfg["eval_benchmark_subset"] = self.eval_benchmark_subset
        return {
            "cfg": cfg,
            "render_mode": self.render_mode,
        }

    def create_envs(self, n_envs: int, use_async_envs: bool = False) -> dict:
        # Honour use_async_envs even at n_envs=1: callers that wrap a single
        # env around a process-isolated splatsim (e.g. dataset-augmentation
        # scripts that already hold a pybullet GUI client in the parent
        # process for the SharedAutonomyPolicyWrapper) need the env to live
        # in a worker process so its own pybullet GUI client doesn't collide
        # with the parent's.
        env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
        splatsim_render_mode = self.gym_kwargs.get("render_mode", "rgb_array")

        # ---- Teleop recording (shared by ZMQ + local-spawn branches) ---- #
        teleop_context = None
        teleop_dataset = None
        image_keys = None
        if self.teleop_dataset_repo_id is not None:
            from splatsim.configs.mode_config import ImageResizeMode
            from splatsim.utils.lerobot_utils import create_lerobot_dataset, load_lerobot_dataset

            from .recording import TeleopRecordingContext

            teleop_context = TeleopRecordingContext.get_instance()
            # Push the user-configured discontinuity threshold into the
            # singleton context so TeleopRecordingWrapper.step() picks it
            # up. Set here (not as a TeleopRecordingWrapper ctor arg) so
            # external callers that share the singleton read the same
            # value.
            teleop_context.state_jump_split_threshold_rad = float(self.teleop_state_jump_split_threshold_rad)
            image_keys = [f"{cam}_{mode.value}" for cam in self.camera_names for mode in ImageResizeMode]
            teleop_dataset = load_lerobot_dataset(self.teleop_dataset_repo_id)
            if teleop_dataset is None:
                # Forward EnvConfig's arm/state/env-state dims so the created
                # dataset's `observation.state`, `action`, and (optional)
                # `observation.environment_state` feature shapes match what the
                # env actually emits. Prior default (num_dofs=6, state_dim=None,
                # env_state_dim=0) was UR5-hardcoded — a planar_3joint run
                # (num_dofs=3, env_state_dim=6) then hit
                # "expected shape (7,) got (4,)" at the first frame commit.
                teleop_dataset = create_lerobot_dataset(
                    self.teleop_dataset_repo_id,
                    fps=self.fps,
                    image_keys=image_keys,
                    num_dofs=self.num_dofs,
                    state_dim=self.state_dim,
                    env_state_dim=self.env_state_dim,
                )

        # Locals captured into _make_splatsim closures below.
        task = self.task
        episode_length = self.episode_length
        teleop_min_episode_length = self.teleop_min_episode_length
        teleop_push_to_hub = self.teleop_push_to_hub
        teleop_pad_short_episodes = self.teleop_pad_short_episodes
        terminate_on_collision = self.terminate_on_collision

        class _CollisionTerminationWrapper(gym.Wrapper):
            """Force `terminated=True` whenever the env reports
            `info["in_collision"]`. SplatSim publishes this key on every
            step (it's the same predicate the intervention controller
            uses for its collision trigger), so we just lift it into the
            gymnasium done-flag so the rollout loop stops the episode
            immediately. No-op if `in_collision` is missing from info.
            """

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if info.get("in_collision"):
                    terminated = True
                return obs, reward, terminated, truncated, info

        def _wrap_for_collision_termination(env):
            """Apply the collision-termination wrapper when configured.
            Pure no-op when terminate_on_collision is False — same as
            other optional wrappers in this file."""
            if terminate_on_collision:
                env = _CollisionTerminationWrapper(env)
            return env

        def _wrap_for_recording(env):
            """Apply TeleopRecordingWrapper when teleop recording is configured."""
            if teleop_context is not None and teleop_dataset is not None:
                from .recording import TeleopRecordingWrapper

                # image_keys is populated in the same block as teleop_dataset
                # above; assert for the type checker's benefit.
                assert image_keys is not None
                env = TeleopRecordingWrapper(
                    env,
                    context=teleop_context,
                    dataset=teleop_dataset,
                    image_keys=image_keys,
                    task=task,
                    min_episode_length=teleop_min_episode_length,
                    push_to_hub=teleop_push_to_hub,
                    pad_short_episodes=teleop_pad_short_episodes,
                )
            return env

        if self.external_port is not None:
            from splatsim.gym_env import ZMQSplatSimGymEnv

            external_host = self.external_host
            external_port = self.external_port
            include_oracle_info = self.include_oracle_info
            camera_names = self.camera_names
            image_resize_modes = self.image_resize_modes
            observation_height = self.observation_height
            observation_width = self.observation_width
            num_dofs = self.num_dofs
            env_state_dim = self.env_state_dim

            def _make_splatsim():
                env = ZMQSplatSimGymEnv(
                    host=external_host,
                    port=external_port,
                    camera_names=camera_names,
                    image_resize_modes=image_resize_modes,
                    num_dofs=num_dofs,
                    image_height=observation_height,
                    image_width=observation_width,
                    render_mode=splatsim_render_mode,
                    max_episode_steps=episode_length,
                    include_oracle_info=include_oracle_info,
                    env_state_dim=env_state_dim,
                )
                # Collision-termination first, then teleop recording (so the
                # recorder sees the same terminated flag that the rollout
                # loop sees; otherwise it would record a partial episode
                # without knowing the episode ended early).
                env = _wrap_for_collision_termination(env)
                return _wrap_for_recording(env)
        else:
            from splatsim.gym_env import make_single_env
            from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase

            splatsim_cfg = self.gym_kwargs.get("cfg", {})
            # Inject the SplatSimEnv.headless field so the in-process
            # PybulletRobotServerBase connects via p.DIRECT instead of p.GUI.
            # No-op when headless is False (the default) — robot server's
            # ctor default leaves GUI mode on.
            splatsim_cfg = {**splatsim_cfg, "headless": self.headless}
            # Keep the Tk control panel alongside headless when asked. Only
            # inject when True: make_single_env passes cfg as **kwargs to the
            # server ctor, and not every server class accepts show_control_gui.
            if self.control_gui:
                splatsim_cfg = {**splatsim_cfg, "show_control_gui": True}
            # Splat shadow compositing for the in-process eval sim. Same
            # inject-only-when-True rule as control_gui above (cfg is
            # splatted as **kwargs into the server ctor, and older server
            # classes don't accept the kwarg).
            if self.splat_shadows:
                splatsim_cfg = {**splatsim_cfg, "splat_shadows": True}
            splatsim_serve_mode = (
                PybulletRobotServerBase.SERVE_MODES.EVAL_BENCHMARK
                if self.eval_benchmark_repo_id is not None
                else PybulletRobotServerBase.SERVE_MODES.INTERACTIVE
            )

            def _make_splatsim():
                env = make_single_env(
                    task,
                    cfg=splatsim_cfg,
                    render_mode=splatsim_render_mode,
                    serve_mode=splatsim_serve_mode,
                )
                # Local mode honours the lerobot-side episode_length cap. The
                # underlying robot server's _max_episode_steps drives both the
                # gym env's truncation and the rollout loop's max_steps query.
                if hasattr(env, "robot_server") and env.robot_server is not None:
                    env.robot_server._max_episode_steps = episode_length
                if hasattr(env, "_max_episode_steps"):
                    env._max_episode_steps = episode_length
                # Collision-termination first, then teleop recording (so the
                # recorder sees the same terminated flag that the rollout
                # loop sees; otherwise it would record a partial episode
                # without knowing the episode ended early).
                env = _wrap_for_collision_termination(env)
                return _wrap_for_recording(env)

        # When the caller asks for AsyncVectorEnv, use the "forkserver" start
        # method to match how the base EnvConfig.create_envs spawns workers —
        # avoids fork-after-CUDA hazards if the parent has loaded a policy on
        # GPU (forkserver forks early before CUDA / threads come up).
        extra_kwargs: dict = {}
        if env_cls is gym.vector.AsyncVectorEnv:
            extra_kwargs["context"] = "forkserver"

        try:
            from gymnasium.vector import AutoresetMode

            vec = env_cls(
                [_make_splatsim for _ in range(n_envs)],
                # NEXT_STEP: on the termination step, final_info is populated (needed by lerobot_eval
                # to read is_success). The actual auto-reset fires on the *next* step call, which
                # never happens since the rollout loop exits on done=True.
                autoreset_mode=AutoresetMode.NEXT_STEP,
                **extra_kwargs,
            )
        except ImportError:
            vec = env_cls([_make_splatsim for _ in range(n_envs)], **extra_kwargs)
        return {"splatsim": {0: vec}}
