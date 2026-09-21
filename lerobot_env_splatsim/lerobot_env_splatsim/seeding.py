"""Helpers for seeding a SplatSim Gym env to a specific dataset frame's state.

The lerobot env factory only exposes a Gym vector env API (`reset`, `step`). For the
visualize / data-relabelling use case we need to start a rollout from an arbitrary
``(episode_index, frame_index)`` state.

``seed_splatsim_env_to_state`` bridges that gap: it calls ``reset(seed=...)`` so the
env loads the right benchmark scenario, then teleports the robot to the requested
joint state. The teleport works for BOTH backends: in-process pybullet envs are
teleported directly via ``PybulletRobotServerBase.teleport_joint_state``, and ZMQ
(out-of-process) envs forward the same call through ``_ZMQBackend.teleport_joint_state``
→ ``ZMQClientRobot`` → the SplatSim server's dispatch loop (which handles
``teleport_joint_state``). Only backends lacking the method (non-SplatSim servers)
fall back to the episode-initial pose, with a debug log.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def _add_batch_dim(value: Any) -> Any:
    """Add a leading batch dim of size 1 to numpy arrays in a (possibly nested) dict."""
    if isinstance(value, dict):
        return {k: _add_batch_dim(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value[np.newaxis, ...]
    return value


def set_env_benchmark_indices(vec_env: gym.vector.VectorEnv, indices: list[int]) -> None:
    """Replace the sim server's EVAL_BENCHMARK playlist.

    Order + duplicates are preserved — passing ``[2, 2, 3, 10, 10]`` makes the
    next 5 ``vec_env.reset()`` calls return scenarios ``2, 2, 3, 10, 10`` in
    that order. Use this when your rollout order is data-driven up front (e.g.
    the blending script wanting each replay to run in the ORIGINAL scenario
    of the source intervention episode it's blending — so the playlist is
    exactly the source ``source_scenario_idx`` sequence).

    Called ONCE before a batch of resets; the sim's internal counter then
    walks the playlist naturally on each subsequent ``reset()``. The counter
    is rewound to -1 by this call, so the FIRST reset lands on ``indices[0]``.

    Works with both the ZMQ backend (``_ZMQBackend.set_eval_benchmark_indices``
    → ``ZMQClientRobot`` → server) and the in-process backend
    (``PybulletRobotServerBase.set_eval_benchmark_indices`` directly).

    For ad-hoc single-reset scenario jumps within an EXISTING playlist, use
    ``seed_splatsim_env_to_state(benchmark_start_index=N)`` instead.
    """
    if not hasattr(vec_env, "envs"):
        raise TypeError(f"set_env_benchmark_indices requires a SyncVectorEnv; got {type(vec_env).__name__}.")
    if len(vec_env.envs) != 1:
        raise ValueError(f"set_env_benchmark_indices only supports n_envs=1 (got {len(vec_env.envs)}).")
    single_env = vec_env.envs[0]
    # Same accessor as seed_splatsim_env_to_state — handles both the ZMQ
    # backend (has .robot_server) and the in-process backend (unwrapped is the
    # PybulletRobotServerBase itself).
    robot_server = getattr(single_env, "robot_server", None) or single_env.unwrapped
    if not hasattr(robot_server, "set_eval_benchmark_indices"):
        raise AttributeError(
            f"{type(robot_server).__name__} does not expose set_eval_benchmark_indices; "
            "either the SplatSim server was launched without EVAL_BENCHMARK mode, or the "
            "server-side method wasn't wired up on this backend."
        )
    robot_server.set_eval_benchmark_indices(list(indices))


def seed_splatsim_env_to_state(
    vec_env: gym.vector.VectorEnv,
    *,
    joint_state: np.ndarray | None = None,
    joint_velocities: np.ndarray | list | None = None,
    num_dofs: int = 6,
    seed: list[int] | None = None,
    benchmark_start_index: int | None = None,
) -> dict[str, Any]:
    """Reset ``vec_env`` and optionally teleport the robot to ``joint_state``.

    The teleport fires after the reset for BOTH backends: local (in-process
    pybullet) envs directly, ZMQ (out-of-process) envs via the SplatSim
    server's ``teleport_joint_state`` dispatch (forwarded by ``_ZMQBackend``).
    Only backends without the method (non-SplatSim servers) skip it — the
    robot then stays at the scenario-initial pose, logged at debug level.

    Args:
        vec_env: A SyncVectorEnv with n_envs=1 wrapping a SplatSim env.
        joint_state: Raw joint configuration to teleport to. Shape ``(num_dofs,)``
            or ``(num_dofs + 1,)`` (with gripper).
        num_dofs: Number of arm degrees of freedom. Defaults to 6.
        seed: Forwarded to ``vec_env.reset(seed=seed)``. Contrary to what its
            name suggests, this DOES NOT select the benchmark scenario in
            SplatSim's EVAL_BENCHMARK mode — the server uses an internal
            per-reset counter for scenario selection and treats ``seed`` purely
            as env/policy randomness. To force a specific scenario, pass
            ``benchmark_start_index`` (see below).
        benchmark_start_index: When set, forwarded to ``vec_env.reset`` as
            ``options={"benchmark_start_index": N}`` — SplatSim's
            EVAL_BENCHMARK ``_handle_reset`` then positions its internal
            counter so THIS reset lands on ``subset[N % len(subset)]``. This is
            the ONLY way to force a deterministic scenario per reset. Common
            use: replay an intervention episode in the same scenario it was
            recorded from (pass the source episode's ``source_scenario_idx``).
            Assumes subset is the identity map (e.g. ``[0..99]``); if the sim
            was launched with a non-identity subset, callers must resolve
            ``subset.index(scenario_idx)`` themselves before passing.

    Returns:
        Batched gym observation dict matching the shape ``vec_env.step()`` produces.
    """
    if not hasattr(vec_env, "envs"):
        raise TypeError(f"seed_splatsim_env_to_state requires a SyncVectorEnv; got {type(vec_env).__name__}.")
    if len(vec_env.envs) != 1:
        raise ValueError(f"seed_splatsim_env_to_state only supports n_envs=1 (got {len(vec_env.envs)}).")

    # Reset loads the benchmark scenario (object poses + episode-start robot joints).
    # gymnasium's SyncVectorEnv.reset broadcasts a single options dict to every
    # sub-env, so a bare dict here is what each sub-env sees — no per-sub-env
    # wrapping needed. On the ZMQ path the dict pickles through the reset
    # request end-to-end and lands as `options` in the server's _handle_reset.
    reset_options: dict[str, Any] | None = None
    if benchmark_start_index is not None:
        reset_options = {"benchmark_start_index": int(benchmark_start_index)}
    env_obs, _info = vec_env.reset(seed=seed, options=reset_options)

    # Joint teleport — works for the in-process pybullet backend AND ZMQ
    # SplatSim servers (_ZMQBackend forwards teleport_joint_state to the
    # server's dispatch loop; the sentinel `splatsim_robot` attr keeps the
    # call signature uniform across backends).
    if joint_state is not None:
        single_env = vec_env.envs[0]
        # `unwrapped` for both lookups: gymnasium wrappers refuse to forward
        # underscore-prefixed attributes, so a wrapped env would silently
        # return the PRE-teleport reset obs if we probed `_to_gym_obs` on it.
        # (Requires SplatSimGymEnv.unwrapped to follow the gymnasium contract
        # and return the base ENV — an old SplatSim override returned the
        # robot server, which skipped the post-teleport obs refresh below.)
        base_env = single_env.unwrapped
        robot_server = getattr(base_env, "robot_server", None) or base_env

        if hasattr(robot_server, "teleport_joint_state") and hasattr(robot_server, "splatsim_robot"):
            js = np.asarray(joint_state, dtype=np.float64).reshape(-1)
            n_set = min(js.shape[0], num_dofs + 1)
            if joint_velocities is not None:
                # Velocity-preserving teleport (resetJointState targetVelocity
                # on the server): a blend rollout seeding an intervention
                # episode's start state must also restore the velocity the
                # policy handed off — a rest-seeded robot lurches at its
                # first command and its obs history contradicts the source
                # episode's moving handoff. Older backends without the
                # parameter fall back to position-only.
                jv = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)[:n_set].tolist()
                try:
                    robot_server.teleport_joint_state(robot_server.splatsim_robot, js[:n_set].tolist(), jv)
                except TypeError:
                    logging.debug("teleport_joint_state lacks joint_velocities — position-only seed.")
                    robot_server.teleport_joint_state(robot_server.splatsim_robot, js[:n_set].tolist())
            else:
                robot_server.teleport_joint_state(robot_server.splatsim_robot, js[:n_set].tolist())
            raw_obs = robot_server.get_observations()
            if hasattr(base_env, "_to_gym_obs"):
                env_obs = _add_batch_dim(base_env._to_gym_obs(raw_obs))
        else:
            logger.debug(
                "seed_splatsim_env_to_state: teleport_joint_state not available "
                "on this backend (non-SplatSim server?). Robot starts at "
                "episode-initial pose."
            )

    return env_obs
