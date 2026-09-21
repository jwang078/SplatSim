"""Teleop episode recording (moved here from the lerobot fork's policies/teleop_recording.py). during shared autonomy evaluation.

When the shared autonomy slider is at ratio=0 (pure teleop), this module
captures observation/action frames and saves them as episodes in a LeRobot
dataset.  Short segments (below a configurable threshold) are padded with
policy frames.

Teleop frames are buffered in memory and trimmed (leading/trailing no-ops
removed) before being committed to the dataset.  This avoids saving frames
where the user hasn't started pressing keys yet or has released keys before
switching out of teleop mode.

Architecture
------------
``TeleopRecordingContext`` is a singleton that bridges the policy wrapper
(which writes the current ratio) and the ``TeleopRecordingWrapper`` gym
wrapper (which reads the ratio to decide whether to record).

``TeleopRecordingWrapper`` is a standard ``gym.Wrapper`` applied to each
individual env *before* vectorisation.  It intercepts ``step`` / ``reset``
/ ``close`` and manages episode boundaries.
"""

from __future__ import annotations

import contextlib
import logging
from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


# Origin of an action sample, used by the recorder to decide whether to keep it.
#
# Built with the Enum functional API because we want a 101-member family
# `BLEND_INTERVENTION_<pct>` alongside the 4 base values. Enumerating those
# 101 members by hand would be untenable; the comprehension below generates
# them at module load time.
#
# Members:
#   TELEOP / RRT             — "real" frames, always committed to the dataset.
#   POLICY                   — skipped, except when ratio==0 (committed for
#                              legacy callers that haven't migrated to TELEOP).
#   PADDING                  — post-teleop pad-to-min frames; committed only
#                              to reach min_episode_length.
#   BLEND_INTERVENTION_<pct> — controller-driven blend frames; committed.
#     <pct> is the INTERVENTION (guidance) share of the blend, 000..100:
#       _000 = 0% intervention (pure policy)
#       _100 = 100% intervention (verbatim controller drive — what
#              OracleGoalGuidanceSource emits for its VERBATIM playback)
#     NOTE the sign convention: this is the COMPLEMENT of the wrapper's
#     `forward_flow_ratio` (where 0.0 = pure guidance and 1.0 = pure
#     policy). A caller tagging frames from a wrapper ratio must pass
#     (1 - forward_flow_ratio) to `blend_at_ratio`, not the ratio itself.
#     The downstream dataset can then filter / weight by intervention share.
_FRAME_SOURCE_BASE = {"TELEOP": "teleop", "RRT": "rrt", "POLICY": "policy", "PADDING": "padding"}
_FRAME_SOURCE_BLEND = {f"BLEND_INTERVENTION_{i:03d}": f"blend_intervention_{i:03d}" for i in range(101)}
FrameSource = Enum(  # type: ignore[misc]
    "FrameSource",
    {**_FRAME_SOURCE_BASE, **_FRAME_SOURCE_BLEND},
    module=__name__,
)


def _blend_at_ratio(cls, ratio: float) -> FrameSource:  # type: ignore[valid-type]
    """Return the `BLEND_INTERVENTION_<pct>` member for the given ratio in [0, 1].

    `ratio` is the INTERVENTION (guidance) share — 1.0 = verbatim controller
    drive, 0.0 = pure policy. This is the COMPLEMENT of the wrapper's
    forward_flow_ratio; pass (1 - forward_flow_ratio) when converting.
    Clamped to [0, 1] and rounded to 2 decimal places, then formatted
    as a 3-digit zero-padded integer matching the enum member naming. Examples:
        0.00 → BLEND_INTERVENTION_000
        0.65 → BLEND_INTERVENTION_065
        1.00 → BLEND_INTERVENTION_100
    """
    idx = int(round(max(0.0, min(1.0, float(ratio))) * 100))
    return cls[f"BLEND_INTERVENTION_{idx:03d}"]


FrameSource.blend_at_ratio = classmethod(_blend_at_ratio)  # type: ignore[attr-defined]


def is_committed_frame_source(source, ratio: float) -> bool:
    """Recorder-side filter: True iff a frame with the given source should be committed.

    Centralized here so the wrapper's step() and any future readers stay in sync.
    The legacy `POLICY + ratio==0` case is preserved for callers that haven't
    migrated to setting a more specific source.
    """
    if source in (FrameSource.TELEOP, FrameSource.RRT):
        return True
    if source.name.startswith("BLEND_INTERVENTION_"):
        return True
    return source is FrameSource.POLICY and ratio == 0.0


# ---------------------------------------------------------------------------
# Shared context singleton
# ---------------------------------------------------------------------------


class TeleopRecordingContext:
    """Shared state between SharedAutonomyPolicyWrapper and TeleopRecordingWrapper.

    The policy wrapper writes ``ratio`` and ``has_guidance`` on every
    ``select_action`` call; the recording wrapper reads them on every ``step``.
    """

    _instance: TeleopRecordingContext | None = None

    def __init__(self) -> None:
        self.ratio: float = 1.0  # current forward_flow_ratio
        self.has_guidance: bool = False  # True when user is actively pressing keys
        self.recording: bool = False
        self.episode_frame_count: int = 0
        self.min_episode_length: int = 60
        self.total_saved_episodes: int = 0
        self.padding: bool = False
        self.discard_requested: bool = False
        # Set by the policy wrapper each step. The recorder uses this to decide
        # whether to record (TELEOP/RRT) and whether the frame counts toward
        # min_episode_length. POLICY frames are skipped entirely.
        self.frame_source: FrameSource = FrameSource.POLICY
        # Index of the eval-benchmark scenario the current episode came from.
        # Pushed by the intervention controller on each scenario reset; the
        # wrapper reads it in _finish_episode and stores it as a per-episode
        # column in the dataset's episodes parquet (via save_episode's
        # episode_metadata kwarg). None when not running under a controller
        # that knows the scenario index (e.g. interactive teleop).
        self.source_scenario_idx: int | None = None
        # Splatsim scene configs for the current scenario. Set by the
        # intervention controller after each env.reset() and cleared on exit.
        # Saved into each episode's metadata so the dataset is self-contained
        # (same fields as episodes recorded via normal data collection).
        self.splatsim_robot_config: dict | None = None
        self.splatsim_object_configs: list | None = None
        self.splatsim_background_config: dict | None = None
        # When True, the recording wrapper accumulates finished episodes in
        # an in-memory pending list instead of writing them to the dataset.
        # The caller then invokes ``commit_pending_episodes()`` (save them)
        # or ``discard_pending_episodes()`` (drop them) once it has the
        # information needed to decide. False = legacy immediate-save
        # behavior, used by interactive teleop where there is no upstream
        # decision to defer to.
        self.defer_episode_saves: bool = False
        # When > 0, the next _flush_buffer call drops this many ADDITIONAL
        # frames from the start of the RRT segment, AFTER the standard
        # leading-trim removes has_guidance=False frames. Set by the
        # intervention controller when it triggers RRT from a rest-start
        # trigger (e.g. "time stall") to drop the n_obs_steps - 1 velocity-
        # artifact frames at the segment onset where the robot is
        # accelerating from a stopped state. Auto-consumed (reset to 0) by
        # _flush_buffer so it only affects one segment per set.
        self.rrt_extra_leading_trim: int = 0
        # When True (default — interactive teleop legacy), episodes shorter
        # than min_episode_length get padded by repeating the last committed
        # frame until they reach the minimum. When False, those episodes are
        # DROPPED entirely instead of padded. The intervention controller
        # sets this to False: pad frames in RRT-recorded episodes are
        # near-goal-state frozen repeats that train the diffusion policy's
        # score field to "hold position when close to goal" — at eval time
        # that fires as "freeze a few cm short of goal", which is the
        # dominant failure mode in DAgger rounds with short final-approach
        # interventions. See diagnostic in plot_episode_onset_deltas.py
        # (any episode where ds[20:].mean() == 0 is a padded episode).
        self.pad_short_episodes: bool = True
        # Signal: "the env's joint state was just mutated outside the normal
        # env.step path — finish the current episode and start a fresh one
        # on the NEXT real frame." Set by _teleport_env_to_q_start whenever
        # the planner explicitly teleports the env (lookback rewind, escape
        # from q_start-in-collision, request_retry_after_collision). The
        # flag is consumed (cleared) by the wrapper's step() the next time
        # it observes is_real_frame=True.
        #
        # Why we ALSO need the recorder-side threshold check below: this
        # signal only covers LEROBOT-driven teleports. PyBullet's
        # constraint solver applies position corrections when the robot
        # link physically penetrates an obstacle during time-parametrized
        # RRT execution — those state jumps come from env physics, never
        # touch any of our code, and so can't be source-signaled. The
        # recorder-side threshold check at TeleopRecording.step() catches
        # them by inspecting the actual Δstate at the recorder level.
        self.force_episode_split_next_real_frame: bool = False
        # Recorder-side state-discontinuity threshold (joint-L2 of Δstate
        # between consecutive real frames, radians). When exceeded,
        # TeleopRecording.step() finalizes the current episode and starts
        # a fresh one — REGARDLESS of whether anyone signaled a teleport.
        #
        # Backup for the source-side signal above: the signal can't reach
        # PyBullet's penetration-correction state changes (env-physics,
        # invisible to lerobot). The threshold check is the only
        # mechanism that can split on those.
        #
        # Contract: each recorded episode is one continuous trajectory.
        # Any inter-frame Δstate > this threshold ends the prior episode
        # there, no exceptions.
        #
        # Default 0.15 rad/frame ≈ 4.5 rad/s at 30 Hz, well above parametrizer-
        # bounded RRT motion (~0.1 rad/frame max) but below typical
        # teleport magnitudes (~0.3-3 rad). Set to 0 to disable.
        self.state_jump_split_threshold_rad: float = 0.15

    @classmethod
    def get_instance(cls) -> TeleopRecordingContext:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


# ---------------------------------------------------------------------------
# Gymnasium recording wrapper
# ---------------------------------------------------------------------------


class TeleopRecordingWrapper(gym.Wrapper):
    """Records pure-teleop (ratio == 0) segments to a LeRobot dataset.

    Wraps an individual (non-vectorised) SplatSim gym env.  Each contiguous
    run of ratio == 0 steps becomes a separate episode.  Segments shorter
    than ``min_episode_length`` are padded with policy frames.

    Teleop frames are buffered in memory and trimmed (leading/trailing
    no-guidance frames removed) before being committed to the dataset.

    When recording, calls ``robot_server.step()`` directly to obtain the
    raw observation dict (which has ``{cam}_{mode}`` image keys for all
    resize modes).  This avoids a second ZMQ round-trip.
    """

    def __init__(
        self,
        env: gym.Env,
        context: TeleopRecordingContext,
        dataset: Any,  # LeRobotDataset — imported lazily to avoid circular deps
        image_keys: list[str],
        task: str,
        min_episode_length: int = 60,
        push_to_hub: bool = True,
        pad_short_episodes: bool = True,
    ) -> None:
        super().__init__(env)
        self._context = context
        self._dataset = dataset
        self._image_keys = image_keys
        self._task = task
        # Whether the DATASET SCHEMA declares an observation.environment_state
        # feature. When it does, every emitted frame must supply that key
        # (dataset_writer.validate_frame raises Missing features on any
        # frame without it). Env exposes it in gym_obs["environment_state"]
        # when env_state_dim > 0 — copy it through in _build_frame.
        _feats = getattr(getattr(self._dataset, "meta", None), "features", None) or {}
        self._include_environment_state = "observation.environment_state" in _feats
        self._min_episode_length = min_episode_length
        # When False, finalize the dataset locally but skip the
        # `self._dataset.push_to_hub()` call in close(). Useful for offline
        # pipelines (e.g. the DAgger orchestrator) that don't want to round-
        # trip each round's intervention dataset through HuggingFace Hub.
        self._push_to_hub = push_to_hub
        # Mirror to the context so _finish_episode can branch on it.
        # See TeleopRecordingContext.pad_short_episodes docstring + envs/
        # configs.py:teleop_pad_short_episodes for the rationale. Setting
        # this False (intervention recording) drops short episodes instead
        # of padding them — last-frame-repeat padding teaches the diffusion
        # policy to freeze near goal.
        self._context.pad_short_episodes = pad_short_episodes

        self._recording: bool = False
        self._padding: bool = False
        self._episode_frame_count: int = 0
        self._committed_frame_count: int = 0  # frames actually in the dataset
        self._frame_buffer: list[tuple[dict, bool]] = []  # (frame, has_guidance)
        # Most recently committed frame; reused as the pad value for short
        # episodes that ended (e.g. successful goal reached) before reaching
        # min_episode_length, since no further env steps are available to
        # source live padding frames from.
        self._last_committed_frame: dict | None = None
        # Previous REAL-FRAME observation.state, kept across step() calls
        # for the recorder-side state-discontinuity check (see
        # context.state_jump_split_threshold_rad). Reset to None on
        # _finish_episode / _discard_episode so the next episode's first
        # real frame doesn't false-split against the previous tail state.
        self._prev_real_frame_state: np.ndarray | None = None
        self._context.min_episode_length = min_episode_length

        # When ``context.defer_episode_saves`` is True, finished episodes go
        # here instead of being written to the dataset. The first list holds
        # frames for the episode currently being built; the second is
        # finalized episodes waiting on a commit/discard call. Both are
        # ignored in immediate-save mode.
        self._in_progress_episode_frames: list[dict] = []
        self._pending_episodes: list[tuple[list[dict], dict | None]] = []

        # Pre-import hf_xet so it's in sys.modules before any KeyboardInterrupt
        # can corrupt Python's import machinery (causes push_to_hub to fail).
        with contextlib.suppress(ImportError):
            import hf_xet  # noqa: F401

    # -- internal helpers ---------------------------------------------------

    def _build_frame(self, action: np.ndarray, gym_obs: dict, raw_obs: dict) -> dict:
        """Build a LeRobot frame dict from gym obs (state) and raw obs (images).

        Uses ``gym_obs["agent_pos"]`` for observation.state and ``raw_obs``
        for images (which are in CHW float32 [0,1] format with
        ``{cam}_{mode}`` keys).
        """
        frame: dict[str, Any] = {
            "observation.state": gym_obs["agent_pos"].astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self._task,
        }
        if self._include_environment_state:
            env_state = gym_obs.get("environment_state")
            if env_state is None:
                # The dataset schema promised an environment_state feature
                # (from env_state_dim > 0 at create-time), but the current
                # step's gym_obs didn't carry one. That's a real invariant
                # break — e.g. someone constructed the env with a different
                # env_state_dim than at teleop_dataset creation. Fail loud
                # here rather than let dataset_writer's validate_frame raise
                # a less-informative "Missing features" error downstream.
                raise RuntimeError(
                    "Teleop dataset declares 'observation.environment_state' "
                    "but gym_obs has no 'environment_state' key. Check that "
                    "the SplatSim env's env_state_dim matches what the "
                    "dataset was created with."
                )
            if isinstance(env_state, torch.Tensor):
                env_state = env_state.cpu().numpy()
            frame["observation.environment_state"] = np.asarray(env_state, dtype=np.float32)
        for key in self._image_keys:
            img = raw_obs.get(key)
            if img is None:
                raise RuntimeError(
                    f"Image key '{key}' missing from server observations. "
                    f"Launch with --env.image_resize_modes listing all modes "
                    f'from ImageResizeMode enum (e.g. \'["stretch", "letterbox"]\').'
                )
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            frame[f"observation.images.{key}"] = img.astype(np.float32)
        return frame

    def _step_raw(self, action: np.ndarray):
        """Step via robot_server, build frame, return (frame, gym_obs, rew, term, trunc, info)."""
        raw_obs, reward, terminated, truncated, info = self.env.robot_server.step(action)
        # This path bypasses SplatSimGymEnv.step, so mirror its env-mutation
        # clock stamping (obs → info) — otherwise recorded (guidance-active)
        # frames lack the stamp and the SA wrapper's teleport-staleness gate
        # is blind exactly when controller-driven teleports happen (mid-RRT).
        _ver = raw_obs.get("state_version")
        if _ver is not None:
            info["state_version"] = int(_ver)
        gym_obs = self.env._to_gym_obs(raw_obs)
        frame = self._build_frame(action, gym_obs, raw_obs)
        return frame, gym_obs, reward, terminated, truncated, info

    # -- buffer / trim / flush ----------------------------------------------

    def _buffer_frame(self, frame: dict) -> None:
        """Buffer a teleop frame for later trimming and flushing."""
        self._frame_buffer.append((frame, self._context.has_guidance))

    def _emit_frame(self, frame: dict) -> None:
        """Send a frame to the dataset (immediate) or the in-progress
        deferred buffer (deferred mode). All paths that previously called
        ``self._dataset.add_frame`` should go through this so the immediate
        and deferred modes share one code path.
        """
        if self._context.defer_episode_saves:
            # Snapshot — dataset.add_frame mutates the dict by popping "task",
            # so we match the same contract on the eventual replay.
            self._in_progress_episode_frames.append(dict(frame))
        else:
            self._dataset.add_frame(frame)

    def _trim_frames(
        self,
        frames: list[tuple[dict, bool]],
        *,
        extra_leading: int = 0,
    ) -> tuple[list[tuple[dict, bool]], int, int, int]:
        """Trim leading and trailing no-guidance frames from the buffer.

        Returns ``(kept_frames, n_leading_trimmed, n_trailing_trimmed,
        n_onset_trimmed)``. ``n_onset_trimmed`` reports how many of the
        ``extra_leading`` request were actually dropped (capped by what
        remained after the standard leading-trim).
        """
        if not frames:
            return frames, 0, 0, 0

        # 1. Trim trailing no-guidance frames
        end = len(frames)
        while end > 0 and not frames[end - 1][1]:
            end -= 1
        n_trailing = len(frames) - end
        if end == 0:
            return [], 0, n_trailing, 0

        # 2. Trim leading no-guidance frames
        start = 0
        while start < end and not frames[start][1]:
            start += 1
        n_leading = start
        if start >= end:
            return [], n_leading, n_trailing, 0

        # 3. Drop the first ``extra_leading`` real frames to suppress
        # velocity-from-rest artifacts at RRT segment onset (e.g. when
        # the trigger reason rewinds via lookback + start_vel=0).
        n_onset = 0
        if extra_leading > 0:
            n_onset = min(end - start, extra_leading)
            start += n_onset

        if start >= end:
            return [], n_leading, n_trailing, n_onset

        return frames[start:end], n_leading, n_trailing, n_onset

    def _flush_buffer(self) -> None:
        """Trim the teleop buffer and commit survivors to the dataset."""
        if not self._frame_buffer:
            return
        # Consume the per-segment onset-trim budget set by the intervention
        # controller (reset regardless of buffer state so a leftover value
        # can't bleed into the next segment).
        extra_leading = self._context.rrt_extra_leading_trim
        self._context.rrt_extra_leading_trim = 0
        trimmed, n_leading, n_trailing, n_onset = self._trim_frames(
            self._frame_buffer, extra_leading=extra_leading
        )
        # Snapshot the last surviving frame BEFORE add_frame mutates it
        # (LeRobotDataset.add_frame pops "task" for separate handling, so
        # post-call the dict is missing fields required by validate_frame).
        # We need a clean copy in case _finish_episode pads short episodes
        # by re-adding this frame.
        last_frame_snapshot = dict(trimmed[-1][0]) if trimmed else None
        for frame, _ in trimmed:
            self._emit_frame(frame)
        n_trimmed = n_leading + n_trailing + n_onset
        if n_trimmed > 0:
            logger.info(
                "[TeleopRecording] Trimmed %d frames (buffer %d → kept %d): "
                "leading=%d (pre-RRT-trigger policy + planning frames, "
                "has_guidance=False), trailing=%d (post-chunk policy frames "
                "before _finish_episode fires, has_guidance=False — usually "
                "0 since auto-cancel calls _finish_episode immediately), "
                "onset=%d (velocity-from-rest artifact frames dropped at "
                "RRT segment start; set by the controller when the trigger "
                "rewinds via lookback)",
                n_trimmed,
                len(self._frame_buffer),
                len(trimmed),
                n_leading,
                n_trailing,
                n_onset,
            )
        self._committed_frame_count += len(trimmed)
        if last_frame_snapshot is not None:
            self._last_committed_frame = last_frame_snapshot
        self._frame_buffer.clear()

    # -- episode lifecycle --------------------------------------------------

    def _finish_episode(self) -> None:
        """Flush any buffered frames, then save or discard the episode."""
        if not self._recording:
            return
        if self._frame_buffer:
            self._flush_buffer()
        # Short-episode handling. Two strategies, selected by
        # `context.pad_short_episodes` (env config: teleop_pad_short_episodes):
        #   True (default — interactive teleop legacy): pad by repeating the
        #     last committed frame until the episode hits min_episode_length.
        #     OK for human teleop where the demonstrator releases controls at
        #     goal (frozen-at-goal frames are part of the natural demo).
        #   False (intervention recording, pass
        #     `--env.teleop_pad_short_episodes=false` in
        #     --intervention_extra_args): DROP the episode entirely instead.
        #     Padded frames are exact repeats (state diff = 0, action = last
        #     commanded), so they train the policy with ~min_episode_length
        #     samples of `obs = (near_goal, near_goal) → action = hold` —
        #     biases the diffusion score field toward "freeze near goal" and
        #     manifests at eval as the policy stopping a few cm short of
        #     goal. Dropping 2-6 short demos per round trades a small amount
        #     of intervention data for keeping the score field clean.
        if self._committed_frame_count > 0 and self._committed_frame_count < self._min_episode_length:
            if not self._context.pad_short_episodes:
                logger.info(
                    f"[TeleopRecording] Dropping short episode "
                    f"({self._committed_frame_count} < {self._min_episode_length} frames; "
                    f"pad_short_episodes=False — last-frame-repeat padding would train "
                    f"'freeze near goal' into the policy)."
                )
                self._in_progress_episode_frames = []
                self._committed_frame_count = 0
                self._last_committed_frame = None
                self._frame_buffer = []
                self._recording = False
                self._padding = False
                self._episode_frame_count = 0
                # Clear the discontinuity anchor like the normal-path tail
                # does, so the next episode's first real frame doesn't
                # false-split against this dropped episode's tail.
                self._prev_real_frame_state = None
                # Push the cleared state to the shared context so the SA GUI's
                # poll sees `recording=False` and reverts the REC label to
                # "Not recording". Without this, the prior "REC N/60 frames
                # (too short)" line stays on screen indefinitely (and bleeds
                # into the next scenario) because no other write to
                # ctx.recording happens until the next real frame.
                self._sync_context()
                return
            elif self._last_committed_frame is not None:
                n_pad = self._min_episode_length - self._committed_frame_count
                for _ in range(n_pad):
                    self._emit_frame(dict(self._last_committed_frame))
                self._committed_frame_count += n_pad
                logger.info(
                    f"[TeleopRecording] Padded {n_pad} repeated frame(s) of the final "
                    f"observation to reach min_episode_length ({self._min_episode_length})."
                )
        if self._committed_frame_count >= self._min_episode_length:
            scenario_idx = self._context.source_scenario_idx
            # Always include the source_scenario_idx key (None when unset) so
            # the per-episode metadata schema is uniform across runs. Mixing
            # rows that have the key with rows that don't crashes
            # _flush_metadata_buffer (pa.Table.from_pydict requires equal
            # column lengths) and silently strands the buffered rows.
            episode_metadata: dict | None = {
                "source_scenario_idx": int(scenario_idx) if scenario_idx is not None else None,
                "splatsim_robot_config": self._context.splatsim_robot_config,
                "splatsim_object_configs": self._context.splatsim_object_configs,
                "splatsim_background_config": self._context.splatsim_background_config,
            }
            if self._context.defer_episode_saves:
                # Move the in-progress frames into the pending list and
                # leave the dataset's writer untouched. The caller will
                # decide later whether to commit or drop them.
                self._pending_episodes.append((self._in_progress_episode_frames, episode_metadata))
                self._in_progress_episode_frames = []
                logger.info(
                    f"[TeleopRecording] Buffered episode for deferred save "
                    f"({self._committed_frame_count} frames, "
                    f"{len(self._pending_episodes)} pending"
                    + (f", source_scenario_idx={scenario_idx}" if scenario_idx is not None else "")
                    + ")"
                )
            else:
                self._dataset.save_episode(episode_metadata=episode_metadata)
                logger.info(
                    f"[TeleopRecording] Saved episode ({self._committed_frame_count} frames, "
                    f"total episodes: {self._dataset.meta.total_episodes}"
                    + (f", source_scenario_idx={scenario_idx}" if scenario_idx is not None else "")
                    + ")"
                )
        else:
            if self._context.defer_episode_saves:
                # Drop the in-progress deferred buffer; nothing was sent to
                # the dataset's writer in this mode.
                self._in_progress_episode_frames = []
            else:
                self._dataset.clear_episode_buffer()
            logger.info(
                f"[TeleopRecording] Discarded too-short episode "
                f"({self._committed_frame_count} < {self._min_episode_length} after trimming)"
            )
        self._recording = False
        self._padding = False
        self._episode_frame_count = 0
        self._committed_frame_count = 0
        self._frame_buffer.clear()
        self._last_committed_frame = None
        # Clear the discontinuity-check anchor so the next episode's first
        # real frame doesn't false-split against this episode's tail.
        self._prev_real_frame_state = None
        self._sync_context()

    def _discard_episode(self) -> None:
        """Discard the current recording segment (triggered by GUI button)."""
        if not self._recording:
            return
        self._frame_buffer.clear()
        self._last_committed_frame = None
        if self._context.defer_episode_saves:
            self._in_progress_episode_frames = []
        else:
            self._dataset.clear_episode_buffer()
        logger.info(f"[TeleopRecording] Manually discarded episode ({self._episode_frame_count} frames)")
        self._recording = False
        self._padding = False
        self._episode_frame_count = 0
        self._committed_frame_count = 0
        # Match _finish_episode: clear the discontinuity anchor so the next
        # episode's first real frame doesn't false-split against this one.
        self._prev_real_frame_state = None
        self._sync_context()

    # -- deferred-save commit/discard ---------------------------------------

    def flush_in_progress_episode(self) -> bool:
        """Force-finalize any in-progress recording (frames in ``_frame_buffer``
        from a recording stream the env never transitioned out of).

        Normally ``_finish_episode`` is triggered when the next ``step()`` sees
        ``frame_source`` flip back to POLICY. If the caller bails out of the
        env loop while still in TELEOP / RRT (e.g. env declared success
        mid-RRT-execution), no further step happens and the recording sits
        stranded in ``_frame_buffer``. Calling this drains it through the
        normal episode-finish path so the resulting episode either lands in
        ``_pending_episodes`` (deferred mode) or saves immediately
        (immediate mode), and the wrapper's recording state resets.

        Returns True if a finalization occurred, False if there was nothing
        to do.
        """
        if not self._recording:
            return False
        self._finish_episode()
        return True

    def commit_pending_episodes(self) -> int:
        """Replay every pending episode to the dataset (add_frame + save_episode).

        No-op in immediate-save mode (pending list is always empty there).
        Safe to call any time; in-progress recordings are unaffected. Returns
        the number of episodes saved.
        """
        n = 0
        for frames, meta in self._pending_episodes:
            for frame in frames:
                self._dataset.add_frame(frame)
            self._dataset.save_episode(episode_metadata=meta)
            n += 1
        if n > 0:
            logger.info(
                f"[TeleopRecording] Committed {n} pending episode(s); "
                f"total episodes: {self._dataset.meta.total_episodes}"
            )
        self._pending_episodes.clear()
        return n

    def discard_pending_episodes(self) -> int:
        """Drop every pending episode without saving. No-op in immediate-save
        mode. Returns the number of episodes dropped."""
        n = len(self._pending_episodes)
        if n > 0:
            total_frames = sum(len(frames) for frames, _ in self._pending_episodes)
            logger.info(f"[TeleopRecording] Discarded {n} pending episode(s) ({total_frames} frames)")
        self._pending_episodes.clear()
        return n

    def _sync_context(self) -> None:
        """Push local recording state to the shared context for GUI display."""
        self._context.recording = self._recording
        self._context.padding = self._padding
        self._context.episode_frame_count = self._episode_frame_count
        self._context.total_saved_episodes = self._dataset.meta.total_episodes

    # -- gym.Wrapper overrides ----------------------------------------------

    def step(self, action: np.ndarray):
        if self._context.discard_requested:
            self._discard_episode()
            self._context.discard_requested = False

        # A frame is "real" (counts toward min_episode_length, gets saved as a
        # genuine sample) when its source is TELEOP, RRT, any BLEND_INTERVENTION_<pct>,
        # or POLICY+ratio==0 (legacy contract for callers that haven't migrated).
        # See `is_committed_frame_source` in this module.
        source = self._context.frame_source
        is_real_frame = is_committed_frame_source(source, self._context.ratio)

        if is_real_frame:
            if self._padding:
                # We were padding the prior cycle's recorded segment when a
                # NEW cycle's first real-frame arrived. Historical behavior
                # treated this as "continue the same episode" — but that's
                # silently wrong for RRT intervention recording: the new
                # cycle starts at a TELEPORTED env state (lookback rewind or
                # escape), so concatenating produces a single recorded
                # episode with a multi-rad state discontinuity in the middle.
                # That discontinuity then pollutes the chunk-relative action
                # stats (chunks straddling the boundary report 1+ rad
                # rel-actions, blowing up aggregated normalization range).
                #
                # Fix: treat the prior-segment + padding as ONE episode,
                # finalize it, and start fresh for the new cycle. Each RRT
                # cycle gets its own episode → no in-episode teleports →
                # chunks never span teleports → clean rel-action stats.
                # `pad_short_episodes=False` flag still applies; if the
                # prior segment was too short to keep, _finish_episode
                # drops it.
                self._padding = False
                self._finish_episode()
            elif self._context.force_episode_split_next_real_frame:
                # Source signaled a mid-cycle teleport just happened (most
                # commonly: request_retry_after_collision → _do_plan →
                # escape → env teleport while mode stayed EXECUTING). The
                # wrapper would normally NOT observe a frame_source
                # transition (stays RRT through the retry), so without this
                # flag the post-teleport frames would get appended to the
                # same episode that contains pre-teleport frames. Force a
                # split here so the recorded episode boundary aligns with
                # the planner's chunk boundary. The flag is consumed (cleared)
                # exactly once per fire, then recording starts fresh.
                logger.info(
                    "[TeleopRecording] Mid-cycle teleport signal received — "
                    "finalizing prior cycle's recording and starting a fresh "
                    "episode for the post-teleport chunk."
                )
                self._context.force_episode_split_next_real_frame = False
                self._finish_episode()
            self._recording = True
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            # Recorder-side state-discontinuity detection. Historically
            # this ALSO split the episode on the theory that PyBullet's
            # constraint-solver position corrections (kicked in when the
            # time-parametrized RRT chunk grazes/penetrates an obstacle)
            # would pollute rel-action stats for the straddling chunk if
            # left inside one episode. In practice the split produced
            # confusing "one RRT plan → three episode segments, two of
            # them dropped as too short" output that hindered debugging.
            # Now: log the jump at WARNING level so it's visible in the
            # timeline, but leave the episode intact. One RRT plan = one
            # recorded episode. If you need the split behavior back, wrap
            # the log with a `_finish_episode()` / `_recording=True` block
            # like it used to have.
            new_state = frame.get("observation.state")
            jump_thr = self._context.state_jump_split_threshold_rad
            if self._prev_real_frame_state is not None and new_state is not None and jump_thr > 0:
                jump = float(np.linalg.norm(new_state - self._prev_real_frame_state))
                if jump > jump_thr:
                    logger.warning(
                        "[TeleopRecording] State discontinuity (Δs=%.4f rad > "
                        "threshold %.4f) DURING episode — likely physics-solver "
                        "correction from an obstacle-penetrating RRT waypoint. "
                        "NOT splitting the episode (kept as one). Investigate "
                        "the plan quality if this fires often.",
                        jump,
                        jump_thr,
                    )
            self._buffer_frame(frame)
            if new_state is not None:
                # Anchor for the next tick's discontinuity check. Plain
                # reference is safe — observation.state is freshly
                # constructed each frame by _build_frame.
                self._prev_real_frame_state = np.asarray(new_state, dtype=np.float32)
            self._episode_frame_count = self._committed_frame_count + len(self._frame_buffer)
            self._sync_context()
            return gym_obs, reward, terminated, truncated, info

        # --- Real-frame stream just ended: flush buffer ---
        if self._recording and self._frame_buffer:
            self._flush_buffer()
            self._episode_frame_count = self._committed_frame_count

        if self._recording and self._committed_frame_count < self._min_episode_length:
            self._padding = True

        if self._padding:
            frame, gym_obs, reward, terminated, truncated, info = self._step_raw(action)
            # Snapshot before add_frame, which pops "task" from the dict.
            self._last_committed_frame = dict(frame)
            self._emit_frame(frame)
            self._committed_frame_count += 1
            self._episode_frame_count = self._committed_frame_count
            self._sync_context()
            if self._committed_frame_count >= self._min_episode_length:
                self._finish_episode()
            return gym_obs, reward, terminated, truncated, info

        self._finish_episode()
        return self.env.step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._finish_episode()
        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        from splatsim.utils.lerobot_utils import finalize_lerobot_dataset

        try:
            self._finish_episode()
        except Exception:
            logger.exception("[TeleopRecording] Error finishing episode during close")
        try:
            finalize_lerobot_dataset(self._dataset)
            logger.info("[TeleopRecording] Dataset finalised.")
            if not self._push_to_hub:
                logger.info(
                    "[TeleopRecording] Skipping Hub push (push_to_hub=False); "
                    f"dataset is local at {self._dataset.root}"
                )
                super().close()
                return
            logger.info(f"[TeleopRecording] Pushing dataset to hub as '{self._dataset.repo_id}'...")
            self._dataset.push_to_hub()
            logger.info("[TeleopRecording] Successfully pushed to hub.")
        except Exception:
            # After KeyboardInterrupt, the import system is often corrupted.
            # Fall back to a subprocess with a clean Python interpreter.
            import subprocess
            import sys

            root = str(self._dataset.root)
            repo_id = self._dataset.repo_id
            print("[TeleopRecording] In-process push failed, retrying in subprocess...")
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from huggingface_hub import HfApi; "
                        f"HfApi().upload_folder(folder_path='{root}', "
                        f"repo_id='{repo_id}', repo_type='dataset', "
                        "ignore_patterns=['images/'])",
                    ],
                    timeout=300,
                )
                if result.returncode == 0:
                    print("[TeleopRecording] Successfully pushed to hub (via subprocess).")
                else:
                    print(f"[TeleopRecording] Subprocess push failed (exit code {result.returncode}).")
                    print(f"Dataset saved locally at: {root}")
            except Exception as e2:
                print(f"[TeleopRecording] Subprocess push also failed: {e2}")
                print(f"Dataset saved locally at: {root}")
        super().close()
