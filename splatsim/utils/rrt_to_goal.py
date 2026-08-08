"""RRT-to-goal planner for the shared autonomy wrapper.

Wraps SplatSim's RRT (`splatsim.utils.rrt_path_utils.get_path`) and TOPP-RA time
parametrization with a small interface tailored to the wrapper's needs:

  * Loads obstacle bodies from a serialized env config (sent over ZMQ from the
    SplatSim server) into a caller-provided pybullet client.
  * Plans a joint-space path from the current joints to a fixed goal config.
  * Returns waypoints sampled at the wrapper's control rate.

The planner does not own a pybullet client. It writes into the client owned by
the wrapper (created in ``SharedAutonomyPolicyWrapper.__init__``); this client
is private to the wrapper and never touches the env-side simulator, so the
planner works whether the env is sim or real-robot.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pybullet as p

from lerobot.policies.guidance.base import GuidanceMode


class PathSelectionStrategy(Enum):
    """How `RRTToGoalPlanner.plan()` picks among IK-goal-candidate paths.

    All strategies score each successful candidate path and pick the minimum;
    they differ only in what they minimize:

    * ``EE_ARC_LENGTH`` (default) — Euclidean distance traversed by the EE
      link in cartesian space. Penalizes wide swings that hurt DAgger data
      quality even when the joint-space length is small.
    * ``JOINT_ARC_LENGTH`` — sum of joint-space L2 distances between
      consecutive waypoints. Legacy behavior; tends to prefer paths that
      happen to land near `q_start` in configuration space even if the EE
      swings wide.
    * ``JOINT_VELOCITY_MATCH`` — cosine distance between the candidate's
      initial direction (averaged over the first few path samples) and
      the robot's recent direction (averaged over the last few samples
      before the trigger). Picks the path that starts off in the same
      direction the robot was already moving, minimizing the velocity
      discontinuity at the trigger moment. Direction-only (not
      magnitude-matching) because the candidate's raw waypoint deltas
      are in different units than the robot's per-tick velocity — see
      ``_path_velocity_deviation``. When the robot's recent velocity is
      near zero (typical from a collision/stall trigger), falls back to
      EE_ARC_LENGTH. Requires `recent_joint_velocity` to be passed to
      `plan()`; raises `RRTPlanningError` if not.
    * ``MIN_PAIR_CLEARANCE`` — picks the candidate whose path maintains
      the LARGEST minimum distance between any non-adjacent robot link
      pair, evaluated at every waypoint. "Larger min" = "more comfortable
      arm pose at every point along the path." Specifically targets the
      pretzeled-pose failure mode: BiRRT can find feasible paths that
      pass through configurations where normally-distant links (e.g.
      gripper near shoulder) come close together — those configurations
      aren't COLLISIONS, but they're hard for diffusion policies to
      learn from. Picking the path with the largest min-pair gap
      systematically avoids them when multiple candidates exist.
      The structurally-close pairs declared via the planner's
      ``self_collision_skip_pairs`` (URDF noise like UR base_link vs
      upper_arm_link, ~4 mm apart at every config) are EXCLUDED from
      the scoring so they don't dominate the min — otherwise every path
      would tie at ~4 mm and the strategy would be useless.
      Cost: one ``getClosestPoints`` query per non-adjacent pair per
      waypoint. For a UR-class robot (~24 links → ~250 non-adjacent
      pairs after filtering) with 20 waypoints, this is ~5K queries per
      candidate path. With ``rrt_num_path_candidates_per_ik=5`` that's
      ~25K per IK goal — ~250 ms at 0.01 ms/query. Use a query distance
      cap so far-apart pairs early-out cheaply.
    Note: scoring metrics that depend ONLY on the goal state (e.g. joint
    distance from start to candidate goal) belong in
    ``IkGoalSelectionStrategy`` instead — those don't need a planned path
    to evaluate and are properties of the IK candidate, not the path.
    """

    EE_ARC_LENGTH = "ee_arc_length"
    JOINT_ARC_LENGTH = "joint_arc_length"
    JOINT_VELOCITY_MATCH = "joint_velocity_match"
    MIN_PAIR_CLEARANCE = "min_pair_clearance"
    CAMERA_SCORING = "camera_scoring"


class IkGoalSelectionStrategy(Enum):
    """How `RRTToGoalPlanner.plan()` picks AMONG the IK candidates BEFORE
    running RRT, based on goal-state geometry alone (no path required).

    When this strategy is set on the planner, candidates are scored by
    their goal-state property (lower = better), tried in order, and the
    FIRST successful RRT plan wins — ``path_selection`` is unused because
    each path is already to a different goal, so cross-path comparison
    is meaningless once we've decided which goal to commit to.

    When the strategy is left unset (None), the planner falls back to
    running RRT against EVERY IK candidate and using ``path_selection``
    to score the resulting paths (historical multi-candidate behavior).

    * ``JOINT_DISTANCE`` — minimize ``||q_candidate - q_start||``. For
      redundant arms (7-DOF, multiple IK solutions per EE pose) this
      picks the goal configuration requiring the LEAST joint
      reconfiguration. Biases the planner toward keeping the policy
      "in its current mode" — elbow-up stays elbow-up, wrist-flip stays
      unflipped. Useful when the policy's training data is multimodal
      (multiple IK branches) and you want intervention data to
      consistently commit to whichever branch the policy is already on.
      Works even when the policy was stationary at trigger time (unlike
      JOINT_VELOCITY_MATCH which needs a velocity history).
    """

    JOINT_DISTANCE = "joint_distance"
    # Sentinel for "no IK-goal pre-selection": try ALL IK candidates and let
    # `path_selection` (e.g. camera_scoring / min_pair_clearance) pick the winner
    # across them — the historical multi-candidate behavior. Normalized to Python
    # None inside RRTToGoalPlanner.__init__ so plan() takes the
    # `_ik_goal_selection is None` branch. Exposed as an enum member so config
    # dropdowns (SplatSim GUI) can offer it alongside JOINT_DISTANCE.
    NONE = "none"


class SoftCostMode(Enum):
    """How a loaded soft-cost field (pushable vegetation) is used by
    `RRTToGoalPlanner`. Enum mirror of the planner's string
    ``soft_cost_mode`` kwarg so config dropdowns (SplatSim GUI) can offer
    the options; the planner itself accepts plain strings.

    * ``OFF`` — ignore the field entirely (binary planning only).
    * ``SCORE`` — candidates are generated cost-blind; ``weight *
      path-integral(cost)`` is added to the path-selection score, so the
      least-exposed of the generated candidates wins. Cheap, but shortcut
      smoothing tends to collapse all candidates onto the same straight
      (often high-cost) route, so the effect is limited in practice.
    * ``GUIDED`` — SCORE plus cost-aware GENERATION: T-RRT transition test
      on every tree-extension step and cost-gated shortcut / elastic /
      trajopt smoothing (see ``rrt_path_utils.cost_aware_birrt``). Paths
      actively route around the field. Costs more planning time (one field
      lookup per extension step); no effect when no field is loaded.
    """

    OFF = "off"
    SCORE = "score"
    GUIDED = "guided"


logger = logging.getLogger(__name__)


# `RRTMode` is aliased to the unified `GuidanceMode` so external callers like
# `InterventionController` (which imports `RRTMode` and compares against
# `RRTMode.IDLE/PLANNING/EXECUTING`) keep working unchanged after the SA-wrapper
# guidance-source refactor. The two enums have byte-identical members and string
# values; the alias is transparent.
RRTMode = GuidanceMode


@dataclass
class RRTRuntimeState:
    """All RRT-mode state owned by the shared autonomy wrapper.

    Bundled into a single dataclass so the wrapper class stays tidy. Access from
    the wrapper looks like ``self._rrt.mode``, ``self._rrt.chunk[i]``, etc.
    """

    mode: RRTMode = RRTMode.IDLE
    chunk: np.ndarray | None = None  # [T, num_dofs] joint waypoints
    step: int = 0  # next index into chunk
    # Optional hint set by the caller (e.g. the intervention controller)
    # BEFORE triggering: the number of waypoints the caller intends to
    # execute before cancelling and handing control back. Used purely for
    # informative logging ("executing X / Y waypoints"); the wrapper itself
    # always drains the full chunk unless explicitly cancelled.
    target_steps: int | None = None
    cancel_requested: bool = False
    planner: RRTToGoalPlanner | None = None
    oracle_env_config: dict | None = None
    oracle_config_hash: str | None = None  # detect config changes
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Last successfully-planned IK goal (joint config). Set by RRTGuidanceSource
    # from the planner's `_last_chosen_q_goal` after each successful plan.
    # Consumed by the retry-on-collision path so we know which IK branch
    # to exclude when re-planning.
    chosen_q_goal: np.ndarray | None = None
    # IK branches whose paths collided when executed in sim. Appended on
    # each retry, passed back into planner.plan() via `exclude_q_goals` so
    # subsequent plans skip them. Reset to [] at scenario start (the
    # source's normal reset path) — within-scenario only.
    excluded_q_goals: list[np.ndarray] = field(default_factory=list)
    # When True for the next _do_plan invocation: skip the pre-jump
    # lookback sampling AND the teleport-to-q_start entirely. q_start is
    # read from the wrapper's CURRENT joint state (no rewind), and the parametrizer
    # is invoked with `start_vel = recent_joint_velocity` so the
    # parametrized trajectory begins at velocity-continuous matching the
    # robot's actual motion. Set per-trigger by RRTGuidanceSource.trigger();
    # consumed (and cleared back to False) by _do_plan at the start of
    # each planning call.
    no_lookback: bool = False


class RRTPlanningError(RuntimeError):
    """Raised when RRT planning fails with a recognizable cause (start/goal
    in collision, no path found within iteration budget, etc.)."""


def _canonical_for_hash(value) -> str:
    """Render value to a canonical string so functionally-identical configs
    hash to the same key.

    Specifically:
      * Sort dict keys (Python preserves insertion order, but the env can
        emit the same dict with different key orderings across calls).
      * Round floats to 6 decimal places — the env's quaternion conversion
        sometimes flips -0.0 vs 0.0 and emits sub-nanometer position
        jitter; without rounding, repr() of those differs and invalidates
        the cache for the same physical scene.
    """
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_canonical_for_hash(k)}:{_canonical_for_hash(v)}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_for_hash(v) for v in value) + "]"
    if isinstance(value, float):
        rounded = round(value, 6)
        # Normalize -0.0 → 0.0 so the sign bit doesn't break the cache.
        if rounded == 0:
            rounded = 0.0
        return repr(rounded)
    return repr(value)


def _hash_config(cfg: dict) -> str:
    """Hash JUST the obstacle-relevant portion of the oracle env config.

    The full ``env_config`` dict includes transient per-step fields like
    ``current_ee_pos`` that change every tick — hashing the whole dict
    invalidates the cache every step and forces a full obstacle reload.
    ``load_obstacles`` only reads ``cfg["objects"]``, so that's the only
    thing the cache key needs to track.

    Uses a canonical-rendering helper that sorts dict keys and rounds floats
    so a static scene produces a stable hash across env ticks, even when
    the env emits ``-0.0`` vs ``0.0`` or sub-µm float jitter.

    Hash is for cache invalidation only, not security. usedforsecurity=False
    silences bandit's B324 warning about SHA1.
    """
    objs = cfg.get("objects", []) if isinstance(cfg, dict) else []
    return hashlib.sha1(_canonical_for_hash(objs).encode("utf-8"), usedforsecurity=False).hexdigest()



class RRTToGoalPlanner:
    """Plans a joint-space trajectory to a fixed goal using SplatSim's RRT.

    Owns no pybullet client; all calls write into ``pb_client``. Not thread-safe
    on its own — the caller serializes access (the wrapper holds an RRT lock and
    only invokes the planner from a single worker thread).
    """

    def __init__(
        self,
        pb_client: int,
        robot_id: int,
        joint_indices: list[int],
        ee_link_index: int,
        num_dofs: int,
        fps: int,
        lower_limits: np.ndarray | None = None,
        upper_limits: np.ndarray | None = None,
        num_ik_candidates: int = 16,
        max_joint_vel: float = 0.5,
        max_joint_acc: float = 1.0,
        max_joint_jerk: float = 10.0,
        parametrize_per_candidate: bool = True,
        path_selection: PathSelectionStrategy = PathSelectionStrategy.EE_ARC_LENGTH,
        velocity_match_window: int = 3,
        segment_at_sharp_corners: bool = True,
        ik_goal_selection: IkGoalSelectionStrategy | str | None = None,
        num_path_candidates_per_ik: int = 1,
        max_path_attempts_per_ik: int = 5,
        path_perturbation_scale: float = 0.001,
        rrt_smooth_iterations: int = 50,
        elastic_smooth_passes: int = 0,
        # CHOMP-lite trajopt smoothing (soft collision + smoothness). 0 = off.
        # See trajopt_smooth_path in rrt_path_utils.py for cost formulation.
        # Default 15 matches both SharedAutonomyConfig and
        # TrajectoryGenModeConfig (which passes it through explicitly).
        trajopt_passes: int = 15,
        trajopt_lr: float = 0.02,
        trajopt_smoothness_weight: float = 1.0,
        trajopt_collision_weight: float = 5.0,
        trajopt_collision_threshold: float = 0.10,
        trajopt_fd_step: float = 0.01,
        # Run trajopt + elastic smoothing on the SELECTED candidate only,
        # instead of on every candidate before ranking. See
        # `_postprocess_path` for the cost/fidelity trade-off.
        postprocess_after_ranking: bool = True,
        final_approach_dist: float = 0.0,
        final_approach_vel_scale: float = 0.3,
        final_approach_acc_scale: float = 0.25,
        uniform_path_speed: bool = False,
        freeze_visualizer_during_plan: bool = False,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
        self_collision_skip_pairs: list[tuple[int, int]] | None = None,
        diagnostic_log_pairs: str = "off",
        escape_clearance_factor: float = 1.5,
        rewind_clearance_factor: float | None = None,
        obstacle_clearance_factor: float = 0.5,
        ik_skip_gripper_obstacle_pairs: bool = False,
        wrist_camera_link_index: int | None = None,
        camera_k_exp: float = 5.0,
        camera_k_sig: float = 15.0,
        camera_threshold: float = 0.4,
        soft_cost_mode: str = "score",
        soft_cost_weight: float = 1.0,
        soft_cost_sample_spacing: float = 0.05,
        soft_cost_surface_samples: int = 6,
        soft_cost_aggregation: str = "max",
        soft_cost_debug_draw: bool = False,
        soft_cost_guided_params: dict | None = None,
    ) -> None:
        self._pb_client = pb_client
        self._robot_id = robot_id
        # Canonical link scope for EVERY planner-owned collision check —
        # excludes ONLY the world frame (-1). The static base_link (0) IS
        # included so gripper/arm swinging back into the robot's own mount
        # is caught by the self-collision check.
        #
        # PRIOR VERSION excluded base_link too, on the theory that its AABB
        # sits ~7 mm above the table top and would false-fire obstacle_collision.
        # In practice (a) the planar env has no table body loaded, (b) the
        # small-engine/UR5 setups place base_link ≥ 100 mm above any obstacle,
        # and (c) the failure mode the exclusion caused (gripper→own-base
        # collision silently ignored) is worse than the false-fire it
        # prevented. If any future env truly has base_link overlapping an
        # obstacle AABB, add (0, obstacle_body_id) to that env's
        # skip_collision_robot_links — the obstacle-side skip mechanism is
        # already separate from self_collision_skip_pairs, so silencing the
        # obstacle false-fire won't disable the self-check.
        #
        # The shared helper `_current_pose_in_planner_collision` uses this
        # scope; the time-parametrized / linear-densified / raw-path /
        # start-in-collision / `check_chunk_collision` checks MUST use the
        # same one or escape's "safe" verdict disagrees with RRT's per-
        # waypoint verdict and the planner cascades into 5-retry backoff at
        # every intervention. See `_current_pose_in_planner_collision`'s
        # docstring for the mismatch-cascade background.
        _n_pb_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        self._planner_link_indices_to_check: list[int] = list(range(0, _n_pb_joints))
        self._planner_num_pb_joints: int = _n_pb_joints
        # Actual gripper config for this plan() call, snapped onto the
        # planner's pybullet client so every downstream collision check
        # (BiRRT collision_fn, IK candidate check, time-parametrized check,
        # linear-densified check) uses the SAME gripper geometry the escape
        # rewind check used. Without this, birrt_path's `set_robot_joint_positions`
        # call forces `open_gripper()` — collision checks then run on OPEN
        # (wide-finger) geometry while escape's check ran on CLOSED (actual)
        # geometry, so escape says "safe" and RRT immediately says "colliding
        # at waypoint 0" for grasp tasks where the env's gripper is closing.
        # Set per plan() invocation; refreshed via `_snap_gripper_to_actual`
        # after every get_path/birrt_path call (which resets it to open).
        self._current_actual_gripper_q: float | None = None
        self._joint_indices = list(joint_indices)
        self._ee_link_index = ee_link_index
        self._num_dofs = num_dofs
        # CAMERA_SCORING (PathSelectionStrategy.CAMERA_SCORING) params — only
        # consulted when that strategy is active. `wrist_camera_link_index=None`
        # makes `_path_camera_score` a no-op (returns 0), so the intervention
        # side never needs to supply camera params.
        self._wrist_camera_link_index = wrist_camera_link_index
        self._camera_k_exp = float(camera_k_exp)
        self._camera_k_sig = float(camera_k_sig)
        self._camera_threshold = float(camera_threshold)
        # Set per-candidate inside the planning loop so `_path_camera_score`
        # can FK the goal EE position as the camera's aim target.
        self._score_goal_q: np.ndarray | None = None
        # Soft-cost (pushable vegetation) scoring. The field itself arrives
        # via env_config["soft_cost"] in load_obstacles; these knobs only
        # control how it is used. With no field loaded (every binary-obstacle
        # env: small_engine, planar_3joint, ...) all of this is a None-check
        # — behavior and cost are identical to the pre-soft-cost planner.
        #   soft_cost_mode: "score" adds weight * path-integral(cost) to the
        #     path-selection score (hard collision checks are untouched);
        #     "guided" does everything "score" does AND makes path GENERATION
        #     cost-aware — T-RRT transition test on tree growth, cost-gated
        #     shortcut/elastic/trajopt smoothing (see rrt_path_utils
        #     cost_aware_birrt). Use "guided" when candidates must actively
        #     route around the field, not just be ranked by it: with "score"
        #     alone every candidate is generated cost-blind and shortcut
        #     smoothing collapses them onto the same (often high-cost)
        #     straight route, leaving the scorer nothing to choose between.
        #     "off" ignores a loaded field entirely.
        #   soft_cost_weight: scale of the soft term relative to the base
        #     score. Fields are normalized to max=1 at build, so the term is
        #     commensurate with EE/joint arc length (both ~"meters").
        #   soft_cost_sample_spacing: spacing (m) of extra samples between
        #     link origins when evaluating the field along the arm.
        #   soft_cost_debug_draw: draw the winning trajectory's EE trace in
        #     the GUI colored by local soft cost (green=free .. red=dense).
        #   soft_cost_guided_params: optional dict of T-RRT overrides
        #     forwarded to get_path(trrt_params=...) in guided mode
        #     (max_iterations, max_time, t_init, t_min, alpha, nfail_max,
        #     restarts, direct_cost_threshold, fallback_to_binary).
        #     None = rrt_path_utils._TRRT_DEFAULTS (10 s/attempt, 1 restart,
        #     then fall back to plain binary birrt so a hard scene degrades
        #     to score-mode quality instead of stalling).
        if isinstance(soft_cost_mode, SoftCostMode):
            soft_cost_mode = soft_cost_mode.value
        if soft_cost_mode not in ("off", "score", "guided"):
            raise ValueError(
                "soft_cost_mode must be 'off', 'score' or 'guided', "
                f"got {soft_cost_mode!r}"
            )
        self._soft_cost_mode = soft_cost_mode
        self._soft_cost_weight = float(soft_cost_weight)
        self._soft_cost_sample_spacing = float(soft_cost_sample_spacing)
        # Surface-ring sampling + reduction: see `_config_soft_cost_points`
        # and `_aggregate_soft_cost` for why the defaults are 6 rings and
        # "max" rather than centerline-only + "mean". Set surface_samples=0
        # and aggregation="mean" to restore the pre-2026-08-01 behavior.
        self._soft_cost_surface_samples = int(soft_cost_surface_samples)
        if soft_cost_aggregation not in ("max", "mean"):
            raise ValueError(
                "soft_cost_aggregation must be 'max' or 'mean', "
                f"got {soft_cost_aggregation!r}"
            )
        self._soft_cost_aggregation = soft_cost_aggregation
        self._link_radii_cache: np.ndarray | None = None
        self._soft_cost_debug_draw = bool(soft_cost_debug_draw)
        self._soft_cost_guided_params = (
            dict(soft_cost_guided_params) if soft_cost_guided_params else None
        )
        self._soft_cost_field = None  # SoftCostField | None; set by load_obstacles
        self._fps = fps
        self._lower_limits = (
            np.asarray(lower_limits, dtype=np.float64)
            if lower_limits is not None
            else -np.pi * np.ones(num_dofs)
        )
        self._upper_limits = (
            np.asarray(upper_limits, dtype=np.float64)
            if upper_limits is not None
            else np.pi * np.ones(num_dofs)
        )
        self._num_ik_candidates = num_ik_candidates
        self._max_joint_vel = max_joint_vel
        self._max_joint_acc = max_joint_acc
        self._max_joint_jerk = max_joint_jerk
        # Parametrization frequency inside plan(). True (default): parametrize +
        # dense-check EACH candidate path so a candidate whose smoothed spline
        # curves into an obstacle is rejected and the next tried (max robustness;
        # the DAgger/SA intervention side wants this). False: per-candidate checks
        # use a cheap LINEAR-DENSIFY collision check (no parametrization) and it runs
        # EXACTLY ONCE on the winning path. SplatSim trajectory-gen sets False so
        # it doesn't re-parametrize once per candidate (and, on the
        # ruckig fallback backend, doesn't hit its cloud API that often).
        self._parametrize_per_candidate = bool(parametrize_per_candidate)
        self._path_selection = path_selection
        # MIN_PAIR_CLEARANCE diagnostic toggle (validated upstream by the
        # SA config's __post_init__). Controls the structural-offender
        # probe inside `_path_min_pair_clearance`. See the SA config for
        # the semantics of "off" / "first" / "always".
        if diagnostic_log_pairs not in ("off", "first", "always"):
            raise ValueError(
                f"diagnostic_log_pairs must be one of 'off'/'first'/'always', got {diagnostic_log_pairs!r}"
            )
        self._diagnostic_log_pairs = diagnostic_log_pairs
        # Forwarded to parametrize_path. True (default) = historical
        # per-segment mode with zero velocity at each sharp corner. False =
        # a single continuous parametrization pass (no forced internal
        # stops). Empirically indistinguishable on typical manipulation
        # RRT plans; True is the safer default.
        self._segment_at_sharp_corners = bool(segment_at_sharp_corners)
        # When set, the planner SHORT-CIRCUITS the multi-candidate
        # path-scoring loop: candidates are sorted by their IK-goal score
        # (lower = better, see IkGoalSelectionStrategy) and tried in
        # order — the FIRST successful RRT plan wins. `path_selection` is
        # ignored in that mode because there's no path-vs-path comparison
        # to make (each path goes to a different goal). When None
        # (default), the original "try all + score by path_selection"
        # behavior runs. Accepts the enum or its string value
        # ("joint_distance") for ergonomic config wiring.
        if ik_goal_selection is None:
            self._ik_goal_selection = None
        else:
            _ik = (
                IkGoalSelectionStrategy(ik_goal_selection)
                if isinstance(ik_goal_selection, str)
                else ik_goal_selection
            )
            # NONE is a config/GUI-facing sentinel for "no IK-goal
            # pre-selection" — collapse it to Python None so plan() falls back
            # to scoring paths across all IK candidates via path_selection.
            self._ik_goal_selection = None if _ik == IkGoalSelectionStrategy.NONE else _ik
        # Per-IK multi-path scoring (ports SplatSim's
        # TrajectoryGenerator._generate_multiple_path_candidates pattern).
        # When num_path_candidates_per_ik > 1, the planner runs RRT
        # multiple times per IK candidate — first attempt at exact
        # endpoints, subsequent attempts with both q_start and q_goal
        # randomly perturbed by ±path_perturbation_scale to nudge RRT's
        # sampler down different branches — then `path_selection` picks
        # the best path among them for that IK. This is what makes
        # `path_selection` non-trivial when `ik_goal_selection` is also
        # set: each IK gets several path candidates, the best one wins
        # for that IK, and then the IK ordering decides which IK's best
        # path is used.
        # max_path_attempts_per_ik caps the total RRT calls per IK
        # (counter resets between successes, so it's actually max attempts
        # BETWEEN successes — matches SplatSim's loop semantics).
        self._num_path_candidates_per_ik = int(num_path_candidates_per_ik)
        self._max_path_attempts_per_ik = int(max_path_attempts_per_ik)
        self._path_perturbation_scale = float(path_perturbation_scale)
        # Random-shortcut smoothing iterations forwarded to get_path /
        # pybullet_planning smooth_path. Shortcutting is what straightens
        # RRT's detours — the parametrizer only shapes timing, not
        # the geometric path. Iterations past convergence are cheap (a
        # candidate segment is collision-checked only when it would shorten
        # the path), so higher values mainly trade a little planning time
        # for visibly less erratic paths.
        self._rrt_smooth_iterations = int(rrt_smooth_iterations)
        # Corner-rounding relaxation after shortcut smoothing (see
        # rrt_path_utils.elastic_smooth_path). 0 = off (historical behavior).
        # Enable for cluttered scenes (vine canopy) where shortcuts collide
        # and jagged joint-space corners survive into the trajectory as wobble.
        self._elastic_smooth_passes = int(elastic_smooth_passes)
        # CHOMP-lite trajopt smoothing after (or in place of) elastic. Explicit
        # repulsive collision cost pushes waypoints away from obstacles rather
        # than just refusing entry into collision. Combined with the Laplacian
        # smoothness term this gives paths with genuinely wider clearance,
        # reducing the sensitivity of downstream imitation policies to small
        # obstacle-position changes (fewer boundary flip-flops between
        # nearly-identical scenarios). Off by default; see the trajopt_*
        # forwarding into get_path() below for how the args flow through.
        self._trajopt_passes = int(trajopt_passes)
        self._trajopt_lr = float(trajopt_lr)
        self._trajopt_smoothness_weight = float(trajopt_smoothness_weight)
        self._trajopt_collision_weight = float(trajopt_collision_weight)
        self._trajopt_collision_threshold = float(trajopt_collision_threshold)
        self._trajopt_fd_step = float(trajopt_fd_step)
        self._postprocess_after_ranking = bool(postprocess_after_ranking)
        # Final-approach taper forwarded to parametrize_path: brake to
        # a stop `final_approach_dist` rad (joint-space L2) before the goal,
        # then creep the last stretch at scaled-down vel/acc so the PD-tracked
        # robot doesn't carry momentum past the final waypoint. 0.0 = off
        # (historical behavior — DAgger runtime keeps this off by default).
        self._final_approach_dist = float(final_approach_dist)
        self._final_approach_vel_scale = float(final_approach_vel_scale)
        self._final_approach_acc_scale = float(final_approach_acc_scale)
        # Equalize joint-space path speed across sections (per-section velocity
        # caps proportional to section direction). Removes the direction-
        # anisotropy surging of box limits — see parametrize_path.
        self._uniform_path_speed = bool(uniform_path_speed)
        # Freeze the pybullet visualizer's world redraw for the duration of
        # plan(). Planning issues thousands of resetJointState +
        # per-collision-check stepSimulation calls; under a GUI connection
        # each syncs with the redraw, dominating planning wall-time. Off by
        # default (DAgger side likes watching the planner explore); SplatSim
        # trajectory-gen enables it for batch throughput. No-op on DIRECT
        # clients.
        self._freeze_visualizer_during_plan = bool(freeze_visualizer_during_plan)
        # Collision clearances threaded into every check_links_in_collision
        # + get_path call so RRT plans paths with the configured margin.
        # None = use SplatSim's defaults (_COLLISION_CLEARANCE = 0.01 m
        # obstacle, self = 0.0 m). Stored as `_obstacle_clearance_override`
        # and `_self_collision_clearance_override`; downstream sites consult
        # them via the `_obstacle_clearance_arg` / `_self_clearance_arg`
        # helpers below so we don't have to special-case None at every
        # callsite.
        self._obstacle_clearance_override = (
            float(obstacle_clearance) if obstacle_clearance is not None else None
        )
        self._self_collision_clearance_override = (
            float(self_collision_clearance) if self_collision_clearance is not None else None
        )
        # Pre-build the kwargs dicts the callsites pass to
        # check_links_in_collision / get_path. None = empty dict so SplatSim's
        # defaults stand (omit kwarg → check_links_in_collision uses
        # _COLLISION_CLEARANCE / self_collision_clearance=0.0).
        self._collision_kwargs: dict = {}
        if self._obstacle_clearance_override is not None:
            self._collision_kwargs["obstacle_clearance"] = self._obstacle_clearance_override
        if self._self_collision_clearance_override is not None:
            self._collision_kwargs["self_collision_clearance"] = self._self_collision_clearance_override
        # Time-parametrized checks use a LOOSER obstacle clearance than the
        # sparse/dense RRT checks. Motivation: the parametrizer's C² spline naturally
        # bulges outside the linear chord at sharp corners; if it had to
        # satisfy the same clearance as the raw path, the reject rate would
        # go through the roof even for paths whose smoothed form only grazes
        # the buffer (say, 1.5 cm on a 2 cm requirement). Halving (default
        # 0.5) says: RRT-clean at 2 cm, parametrizer-clean at 1 cm — still catches
        # actual penetration, doesn't reject cornering bulges. Applies to
        # obstacle clearance only; self-collision clearance stays symmetric.
        # Only the `_smooth_and_check_collision` callsites and stage 3 of
        # `_validate_final_trajectory` use these parametrizer-scaled kwargs.
        self._smoothed_obstacle_clearance_factor = float(obstacle_clearance_factor)
        # Skip-pair list goes into the same kwargs dict so every
        # `check_links_in_collision(**self._collision_kwargs)` / `get_path(...)`
        # callsite picks it up automatically — same pattern as the clearance
        # overrides. Empty / None list means we omit the kwarg entirely so
        # SplatSim's defaults stand (no skips).
        if self_collision_skip_pairs:
            self._collision_kwargs["self_collision_skip_pairs"] = [
                tuple(p) for p in self_collision_skip_pairs
            ]
        # Multiplier on `_effective_obstacle_clearance()` used by
        # `_escape_collision` to set how far past the BiRRT collision
        # threshold the escape pushes before declaring "clear". Provides
        # a buffer so subsequent motion doesn't immediately dip
        # back into the threshold (cascade-retry the user observed:
        # escape stops at exactly the planner clearance → the trajectory moves
        # 1 step toward goal → controller's per-tick check fires → retry
        # → escape re-runs to the same barely-safe config → repeat).
        # Default 1.5× gives ~50% margin (e.g., 3cm escape for a 2cm
        # planner clearance). Set to 1.0 to restore the historical
        # "escape stops exactly at threshold" behavior.
        self._escape_clearance_factor = float(escape_clearance_factor)
        # Separate factor for `_escape_via_policy_history_rewind`. Rewind
        # picks a real historical policy frame (already well-formed, robot
        # was moving normally there) rather than synthesizing a config at
        # the boundary like contact-normal escape does — so it doesn't
        # need as much margin to dodge the ramp-up cascade. None = inherit
        # `escape_clearance_factor` (back-compat). Lower values let rewind
        # accept frames closer to "now" (more in-distribution for the
        # policy) at the cost of less margin against the per-tick check's
        # ramp-up dip.
        self._rewind_clearance_factor = (
            float(rewind_clearance_factor)
            if rewind_clearance_factor is not None
            else float(escape_clearance_factor)
        )
        # Policy-frame-history context used by `_escape_via_policy_history_rewind`
        # — the highest-priority entry in `_try_escape_chain`. Refreshed by the
        # source via `set_policy_history_context()` before each `plan()` call.
        # `_policy_history_ref` is a non-owning reference to the wrapper's
        # `_actual_q_history` deque (newest entry at [-1]); `_policy_history_max_lookback`
        # caps how far back the escape can walk (= wrapper's
        # `_frames_since_last_rrt_end` so the rewind never lands in a prior
        # RRT cycle's trajectory). Both None/0 by default → the rewind
        # method returns None and the chain falls through to the existing
        # contact-normal / self-collision-gradient escapes.
        self._policy_history_ref: object | None = None
        self._policy_history_max_lookback: int = 0
        # Published by plan() before parametrization — last successful IK goal as
        # joint config. Consumed by RRTGuidanceSource's retry-on-collision
        # to track which IK branch to exclude when re-planning.
        self._last_chosen_q_goal: np.ndarray | None = None
        # Window over which to average velocities for the JOINT_VELOCITY_MATCH
        # strategy. Applied to BOTH the candidate path's leading edge and the
        # robot's trailing velocity history. 3 samples ≈ 100 ms at 30 Hz —
        # enough to smooth jitter, short enough that "recent" still means recent.
        self._velocity_match_window = int(velocity_match_window)
        self._loaded_obstacle_ids: list[int] = []  # only oracle-loaded bodies
        self._obstacle_names: dict[int, str] = {}
        self._skip_pairs: set[tuple[int, int]] = set()
        self._loaded_config_hash: str | None = None
        # PyBullet's calculateInverseKinematics expects null-space arrays
        # (lowerLimits / upperLimits / jointDamping / restPoses) sized to the
        # number of MOVABLE joints in the URDF, not just the arm DOFs we plan
        # over. Cache the count once so the IK calls below can pad correctly.
        self._num_movable_joints = sum(
            1
            for j in range(p.getNumJoints(self._robot_id, physicsClientId=self._pb_client))
            if p.getJointInfo(self._robot_id, j, physicsClientId=self._pb_client)[2] != p.JOINT_FIXED
        )
        # When True, the IK candidate filter inside `_solve_ik` ignores
        # gripper-finger ⟷ obstacle pairs. Motivation: at grasp goals the
        # gripper fingers are intentionally within mm of the target object,
        # which the global obstacle clearance (e.g. 2 cm) rejects — leaving
        # only far-away IK branches and forcing RRT into long detours. The
        # arm-link clearance check still runs on every other link, and the
        # full RRT path / per-tick / future-chunk shield checks downstream
        # are UNCHANGED. Auto-detects finger/knuckle link indices by URDF
        # link name at construction time.
        self._ik_skip_gripper_obstacle_pairs = bool(ik_skip_gripper_obstacle_pairs)
        self._gripper_finger_link_indices: list[int] = (
            self._discover_gripper_finger_link_indices()
            if self._ik_skip_gripper_obstacle_pairs
            else []
        )
        if self._ik_skip_gripper_obstacle_pairs:
            logger.info(
                "IK collision filter: skipping gripper-finger ⟷ obstacle pairs "
                "(auto-detected link indices: %s)",
                self._gripper_finger_link_indices,
            )

    def _discover_gripper_finger_link_indices(self) -> list[int]:
        """Enumerate URDF link indices whose name contains 'finger' or 'knuckle'.

        Used by the IK collision filter when ``ik_skip_gripper_obstacle_pairs``
        is True. Name-based detection covers the standard Robotiq 2F-85
        subtree (left/right outer_finger, inner_finger, inner_finger_pad,
        outer_knuckle, inner_knuckle) without hard-coding indices that
        could shift between URDFs.
        """
        finger_links: list[int] = []
        for j in range(p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)):
            info = p.getJointInfo(self._robot_id, j, physicsClientId=self._pb_client)
            raw_name = info[12]
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            lname = name.lower()
            if "finger" in lname or "knuckle" in lname:
                finger_links.append(j)
        return finger_links

    def _ik_skip_pairs(self) -> set[tuple[int, int]]:
        """Skip-pair set used by the IK candidate filter in ``_solve_ik``.

        When ``ik_skip_gripper_obstacle_pairs`` is True, this is
        ``self._skip_pairs ∪ {(finger_link, obs_id) for finger × obstacle}``.
        Otherwise it's ``self._skip_pairs`` unchanged.
        """
        if not self._ik_skip_gripper_obstacle_pairs or not self._gripper_finger_link_indices:
            return self._skip_pairs
        augmented = set(self._skip_pairs)
        for link in self._gripper_finger_link_indices:
            for obs_id in self._loaded_obstacle_ids:
                augmented.add((link, obs_id))
        return augmented

    def _effective_obstacle_clearance(self) -> float:
        """Override or SplatSim's _COLLISION_CLEARANCE default (0.01).

        Used by `_escape_collision` whose contact-normal math is keyed off
        the SAME clearance threshold the BiRRT collision_fn uses, so the
        escape moves the robot to a state the planner considers safe.
        """
        if self._obstacle_clearance_override is not None:
            return self._obstacle_clearance_override
        # Lazy import to keep splatsim out of module-level import surface.
        from splatsim.utils.rrt_path_utils import _COLLISION_CLEARANCE

        return _COLLISION_CLEARANCE

    def _smoothed_collision_kwargs(self) -> dict:
        """Kwargs dict for time-parametrized collision checks. Same as
        `self._collision_kwargs` but with `obstacle_clearance` scaled by
        `_smoothed_obstacle_clearance_factor` (default 0.5). Sparse/dense RRT
        checks keep the full clearance; only the smoothed-spline check uses
        this looser bound so the parametrizer's natural cornering bulge past the linear
        chord doesn't cause spurious rejects. Self-collision clearance and
        skip-pairs are preserved as-is (the bulge argument doesn't apply to
        self-collision — those pairs are geometric constants of the URDF).
        """
        kwargs = dict(self._collision_kwargs)
        base_obs = self._effective_obstacle_clearance()
        kwargs["obstacle_clearance"] = base_obs * self._smoothed_obstacle_clearance_factor
        return kwargs

    def _current_pose_in_planner_collision(
        self,
        return_kind: bool = False,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
    ) -> bool | tuple[bool, str | None]:
        """Canonical "does the robot's CURRENT joint state collide by the
        planner's contract?" check. NO snap — caller has already positioned
        the robot (either via `is_q_in_collision`'s pre-check snap, or by
        the escape methods' iteration loop). Uses the exact same
        parameters + link-scope as `is_q_in_collision`:

          * `link_indices_to_check = range(0, n_joints)` — excludes ONLY
            the world frame (-1). base_link (0) IS included so the
            gripper/arm swinging back into the robot's own mount is caught
            by the self-collision check. (The prior "range(1, n_joints)"
            scope silently ignored those gripper→base collisions.) If a
            specific env has base_link geometrically overlapping an
            obstacle AABB, silence it via env-config `skip_pairs` —
            obstacle-side skips leave the (0, X) self-check live.
          * `skip_pairs = self._skip_pairs` — env-config per-obstacle skips.
          * `**self._collision_kwargs` — same obstacle/self_collision
            clearances and skip_pairs the BiRRT collision_fn uses.

        Any callsite that decides "am I in collision by RRT standards?"
        MUST go through this helper (or `is_q_in_collision`, which wraps
        it). Prior bug: escape methods (`_escape_collision`,
        `_escape_self_collision_gradient`) called `check_links_in_collision`
        directly WITHOUT `link_indices_to_check`, so they used the broader
        default scope `range(-1, n_joints)` — flagging (world/base_link, X)
        pairs that the RRT planner ignores. Escape sometimes declared
        "success" when RRT still saw a collision (or vice-versa), producing
        infinite loops in the intervention controller.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        kwargs = dict(self._collision_kwargs)
        if obstacle_clearance is not None:
            kwargs["obstacle_clearance"] = float(obstacle_clearance)
        if self_collision_clearance is not None:
            kwargs["self_collision_clearance"] = float(self_collision_clearance)
        result = check_links_in_collision(
            self._robot_id,
            self._joint_indices,
            None,
            self._loaded_obstacle_ids,
            obstacle_names=self._obstacle_names,
            link_indices_to_check=self._planner_link_indices_to_check,
            skip_pairs=self._skip_pairs,
            verbose=False,
            physics_client_id=self._pb_client,
            return_kind=return_kind,
            **kwargs,
        )
        if return_kind:
            colliding, kind = result
            return bool(colliding), kind
        return bool(result)

    def _snap_gripper_to_actual(self) -> None:
        """Snap the planner's pybullet gripper joints (URDF indices
        n_dof+1..num_pb_joints) to ``self._current_actual_gripper_q``.
        No-op when actual_gripper_q was not passed to `plan()` (legacy
        callers, or the wrapper's obs.state excludes the gripper dim).

        MUST be called (a) at the top of `plan()` after storing
        `_current_actual_gripper_q`, and (b) after every `get_path` /
        `birrt_path` call — those go through `set_robot_joint_positions`
        which forces `open_gripper()` (resets all gripper joints to 0.0
        = wide-open geometry). Without the re-snap, every downstream
        `check_links_in_collision(q=arm_only)` runs with wide-open fingers
        while the real env robot's fingers are typically closing around
        an object; the geometric mismatch is exactly the "escape says
        safe / parametrized path collides at waypoint 0" cascade.
        """
        if self._current_actual_gripper_q is None:
            return
        gripper_val = float(self._current_actual_gripper_q)
        # Joint 0 isn't in `_joint_indices` (fixed base attach), arm
        # joints occupy [1, n_dof]; gripper joints start at n_dof+1.
        # Same convention as `is_q_in_collision` at line ~718 and
        # `check_chunk_collision` at line ~3341.
        for idx in range(self._num_dofs + 1, self._planner_num_pb_joints):
            p.resetJointState(self._robot_id, idx, gripper_val, physicsClientId=self._pb_client)

    def is_q_in_collision(
        self,
        q: np.ndarray,
        return_kind: bool = False,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
    ) -> bool | tuple[bool, str | None]:
        """Snap-and-check whether a single joint config is in collision.

        Uses the same `check_links_in_collision` contract as the BiRRT
        planner — same obstacle list (whatever the most recent
        `load_obstacles` cached), same `self_collision_skip_pairs`, same
        robot.

        Clearance handling:
          * `obstacle_clearance` / `self_collision_clearance` args, when
            provided, OVERRIDE the planner's `_collision_kwargs` values for
            this call only. Designed for the controller's per-tick
            "is current state in collision?" probe, which wants an
            INTERMEDIATE clearance (catches wedges without flagging
            legitimate goal-approach near-contacts) distinct from the
            planner's path-clearance (which keeps non-goal waypoints
            clear).
          * When both args are None, defaults to `_collision_kwargs`
            (planner clearance). Skip-pairs are always inherited from
            `_collision_kwargs` — they're URDF-structural, not policy-
            dependent.

        Side effect: leaves the planning robot AT ``q`` on exit. The caller
        owns the restore-to-prior-state if needed. (For controller-tick
        use, the planner robot's state is ephemeral between RRT calls, so
        there's no observable side effect on planning.)

        Returns:
          - When ``return_kind=False`` (default): bool.
          - When ``return_kind=True``: (bool, kind) where kind is
            ``"obstacle"`` / ``"self"`` / None — matches
            ``check_links_in_collision(return_kind=True)`` semantics.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        n_dof = len(self._joint_indices)
        if q_arr.size < n_dof:
            # Caller may pass DOF+gripper; we only need the DOF slice. If
            # they passed fewer than n_dof, refuse — no safe interpretation.
            raise ValueError(
                f"is_q_in_collision: q has size {q_arr.size}, need >= {n_dof} (n_dof)"
            )
        q_dof = q_arr[:n_dof]
        for j_idx, qi in zip(self._joint_indices, q_dof.tolist(), strict=True):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
        # Set the gripper joints (URDF indices ≥ n_dof+1; joint 0 is the
        # fixed base-attach joint that's not in `_joint_indices`) to match
        # the env's ACTUAL gripper config from `q[-1]`. CRITICAL for
        # avoiding false-positive `obstacle_collision` triggers:
        # `check_links_in_collision(q=...)` internally calls
        # `set_robot_joint_positions` which forces `open_gripper(robot_id)`
        # — that resets every joint ≥ 7 to 0.0 (the URDF's open pose).
        # For grasp tasks where the env's gripper closes during the
        # approach, the planner's OPEN-gripper geometry has wider fingers
        # than the env's CLOSED gripper. When the wrist is near the goal
        # object (intentional, that's where the approach lands), the
        # planner's open fingers touch the object and `is_q_in_collision`
        # returns True — firing a spurious retry every tick even though
        # the env's actually-closed gripper would have plenty of clearance.
        # Per-callsite manual reset + q=None below avoids this entirely.
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        # Joint 0 isn't in `_joint_indices` (likely a fixed attach), so arm joints
        # occupy [1, n_dof]; gripper joints start at n_dof+1. Matches
        # `open_gripper`'s `range(7, num_joints)` convention when n_dof=6.
        if q_arr.size > n_dof:
            # q carries the gripper (the [joints, gripper] state layout).
            gripper_val = float(q_arr[n_dof])
            for idx in range(n_dof + 1, num_joints):
                p.resetJointState(self._robot_id, idx, gripper_val, physicsClientId=self._pb_client)
        else:
            # q has NO gripper dim (e.g. --exclude_gripper_from_state sliced it out
            # of observation.state). DON'T leave the gripper at whatever stale
            # value the planner robot last held — a garbage / mimic-inconsistent
            # gripper self-collides at nearly every arm pose, which makes the
            # escape→RRT recovery fail at waypoint 0 every time. Force the OPEN
            # pose (all gripper joints 0.0). Excluding the gripper from state
            # implies it's constant/irrelevant to the task (a non-grasp reach), so
            # open is the correct, always-valid assumption.
            for idx in range(n_dof + 1, num_joints):
                p.resetJointState(self._robot_id, idx, 0.0, physicsClientId=self._pb_client)
        # Delegate the actual check to `_current_pose_in_planner_collision`
        # — the canonical "planner-contract" check every callsite (this
        # method, all `_escape_*` methods) shares. See that helper's
        # docstring for the scope + parameter rationale.
        colliding, kind = self._current_pose_in_planner_collision(
            return_kind=True,
            obstacle_clearance=obstacle_clearance,
            self_collision_clearance=self_collision_clearance,
        )
        if return_kind:
            return bool(colliding), kind
        return bool(colliding)

    def describe_collision_at(
        self,
        q: np.ndarray,
        obstacle_clearance: float | None = None,
        self_collision_clearance: float | None = None,
        query_radius: float = 0.5,
    ) -> dict | None:
        """Debug probe: name the absolute-closest link pair at ``q``.

        Snaps the planning robot to ``q`` (arm + actual gripper config from
        ``q[n_dof]`` when present, same gripper-snap fix as
        ``is_q_in_collision``), then iterates ALL (robot_link, obstacle)
        and non-adjacent (robot_link_a, robot_link_b) pairs via
        ``getClosestPoints(distance=query_radius)`` (default 0.5 m so we
        catch the closest pair even when it's WELL above the violation
        threshold — useful for diagnosing the "is_q_in_collision said True
        but no pair under threshold" case).

        Returns a dict describing the closest obstacle pair AND the closest
        self pair, with ``in_violation`` flags computed against the given
        clearances. Returns ``None`` only when the planner has no obstacles
        loaded AND no robot links to check (degenerate setup).

        Honors the planner's existing skip-pairs (URDF-structural).
        """
        from splatsim.utils.rrt_path_utils import are_adjacent_links

        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        n_dof = len(self._joint_indices)
        if q_arr.size < n_dof:
            return None
        q_dof = q_arr[:n_dof]
        for j_idx, qi in zip(self._joint_indices, q_dof.tolist(), strict=True):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
        if q_arr.size > n_dof:
            gripper_val = float(q_arr[n_dof])
            num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
            for idx in range(n_dof + 1, num_joints):
                p.resetJointState(self._robot_id, idx, gripper_val, physicsClientId=self._pb_client)

        obs_thr = (
            float(obstacle_clearance)
            if obstacle_clearance is not None
            else self._collision_kwargs.get("obstacle_clearance", 0.01)
        )
        self_thr = (
            float(self_collision_clearance)
            if self_collision_clearance is not None
            else self._collision_kwargs.get("self_collision_clearance", 0.0)
        )
        _self_skip_raw = self._collision_kwargs.get("self_collision_skip_pairs") or []
        self_skip: set[frozenset[int]] = {frozenset((int(a), int(b))) for a, b in _self_skip_raw}

        n_links = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        # Iterate links 0..n_links-1 — same scope as is_q_in_collision /
        # check_chunk_collision (which now include base_link so gripper-
        # into-mount self-collisions are caught). The world frame (-1) is
        # still skipped because it doesn't move and isn't a robot link.
        link_indices = list(range(0, n_links))

        def _link_name(idx: int) -> str:
            if idx == -1:
                return "base(-1)"
            info = p.getJointInfo(self._robot_id, idx, physicsClientId=self._pb_client)
            raw = info[12]
            name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            return f"{name}({idx})"

        closest_obs: dict | None = None
        closest_self: dict | None = None

        for link_i in link_indices:
            for obs in self._loaded_obstacle_ids:
                if (link_i, obs) in self._skip_pairs:
                    continue
                pts = p.getClosestPoints(
                    bodyA=self._robot_id,
                    bodyB=obs,
                    distance=query_radius,
                    linkIndexA=link_i,
                    physicsClientId=self._pb_client,
                )
                for pt in pts:
                    dist = float(pt[8])
                    if closest_obs is None or dist < closest_obs["distance_m"]:
                        closest_obs = {
                            "kind": "obstacle",
                            "link_a_idx": link_i,
                            "link_a_name": _link_name(link_i),
                            "link_b_idx": int(obs),
                            "link_b_name": self._obstacle_names.get(obs, str(obs)),
                            "distance_m": dist,
                            "threshold_m": obs_thr,
                            "in_violation": dist < obs_thr,
                        }

        import itertools as _it
        for a, b in _it.combinations(link_indices, 2):
            if frozenset((a, b)) in self_skip:
                continue
            if are_adjacent_links(self._robot_id, a, b, physics_client_id=self._pb_client):
                continue
            pts = p.getClosestPoints(
                self._robot_id,
                self._robot_id,
                query_radius,
                linkIndexA=a,
                linkIndexB=b,
                physicsClientId=self._pb_client,
            )
            for pt in pts:
                dist = float(pt[8])
                if closest_self is None or dist < closest_self["distance_m"]:
                    closest_self = {
                        "kind": "self",
                        "link_a_idx": a,
                        "link_a_name": _link_name(a),
                        "link_b_idx": b,
                        "link_b_name": _link_name(b),
                        "distance_m": dist,
                        "threshold_m": self_thr,
                        "in_violation": dist < self_thr,
                    }

        if closest_obs is None and closest_self is None:
            return None
        # Pick whichever closest pair is "more violating" — prefer the one
        # that's actually below its threshold. If neither violates, return
        # the closest one as primary (diagnoses the discrepancy case).
        candidates = [c for c in (closest_obs, closest_self) if c is not None]
        primary = next((c for c in candidates if c["in_violation"]), None)
        if primary is None:
            primary = min(candidates, key=lambda c: c["distance_m"] - c["threshold_m"])
        # Sanity check: when this probe says no pair is in violation, but
        # is_q_in_collision (which uses the SAME check_links_in_collision
        # call as the controller's per-tick + RRT path checks) might still
        # return True, call the underlying check with verbose=True so its
        # own "Collision:" / "Self-collision:" print line surfaces the
        # ACTUAL offending pair to stdout. Lets us catch the case where
        # the iteration loop here diverges from the planner's loop
        # (different broadphase cache, hidden filter, etc.).
        any_violation = any(c["in_violation"] for c in candidates)
        if not any_violation:
            from splatsim.utils.rrt_path_utils import check_links_in_collision as _clc
            sanity_kwargs = {"obstacle_clearance": obs_thr, "self_collision_clearance": self_thr}
            if _self_skip_raw:
                sanity_kwargs["self_collision_skip_pairs"] = _self_skip_raw
            sanity_in_coll, sanity_kind = _clc(
                self._robot_id,
                self._joint_indices,
                None,
                self._loaded_obstacle_ids,
                link_indices_to_check=link_indices,
                skip_pairs=self._skip_pairs,
                obstacle_names=self._obstacle_names,
                verbose=True,
                physics_client_id=self._pb_client,
                return_kind=True,
                **sanity_kwargs,
            )
            if sanity_in_coll:
                logger.warning(
                    "describe_collision_at: my probe found no pair below threshold "
                    "(closest_obs=%.2fmm, closest_self=%.2fmm), but check_links_in_collision "
                    "returned (True, %r). See 'Collision:' / 'Self-collision:' stdout line for actual offender.",
                    closest_obs["distance_m"] * 1000 if closest_obs else float("inf"),
                    closest_self["distance_m"] * 1000 if closest_self else float("inf"),
                    sanity_kind,
                )
        return {**primary, "closest_obstacle": closest_obs, "closest_self": closest_self}

    # ------------------------------------------------------------------ #
    #  Obstacle loading                                                  #
    # ------------------------------------------------------------------ #

    def load_obstacles(self, env_config: dict) -> list[int]:
        """Populate the wrapper's pybullet client with obstacles from ``env_config``.

        Idempotent: hashes the input and short-circuits when the config hasn't
        changed since the last load. On a cache miss, removes only the bodies
        this planner previously loaded (leaves any wrapper-owned hardcoded
        fallback obstacles untouched), then loads the new set.

        Returns the list of pybullet body IDs for the oracle obstacles.
        """
        cfg_hash = _hash_config(env_config)
        if cfg_hash == self._loaded_config_hash:
            return list(self._loaded_obstacle_ids)
        logger.info("load_obstacles: cache miss (hash %s) — loading fresh", cfg_hash[:12])

        # Tear down previously-loaded oracle bodies.
        for body_id in self._loaded_obstacle_ids:
            with contextlib.suppress(p.error):
                p.removeBody(body_id, physicsClientId=self._pb_client)
        self._loaded_obstacle_ids.clear()
        self._obstacle_names.clear()
        self._skip_pairs.clear()

        for obj in env_config.get("objects", []):
            body_id = self._load_one_object(obj)
            if body_id is None:
                continue
            self._loaded_obstacle_ids.append(body_id)
            name = obj.get("name", f"body_{body_id}")
            self._obstacle_names[body_id] = name
            for link_idx in obj.get("skip_collision_robot_links") or []:
                self._skip_pairs.add((int(link_idx), body_id))
            # Verify the collision shape actually exists at the expected place
            # by reading the body's AABB in our pybullet client. If a body has
            # no collision shape, getAABB returns the base-frame AABB only
            # (~zero-volume), which is the smoking gun for "planner ignores
            # this obstacle".
            try:
                num_links = p.getNumJoints(body_id, physicsClientId=self._pb_client)
                aabbs = [
                    p.getAABB(body_id, linkIndex=link_i, physicsClientId=self._pb_client)
                    for link_i in range(-1, num_links)
                ]
                logger.info(
                    "_load_one_object: name=%s body_id=%d num_links=%d AABBs=%s",
                    name,
                    body_id,
                    num_links,
                    aabbs,
                )
            except p.error as e:
                logger.warning("getAABB failed for %s (body_id=%d): %s", name, body_id, e)

        # Optional soft-cost payload (pushable vegetation). Travels in the
        # same env-config dict as `objects`; absent for binary-obstacle envs,
        # in which case the field stays None and scoring is unchanged.
        self._soft_cost_field = None
        soft_payload = env_config.get("soft_cost")
        if soft_payload:
            try:
                from splatsim.utils.soft_cost_field import SoftCostField

                self._soft_cost_field = SoftCostField.from_config(soft_payload)
                logger.info(
                    "load_obstacles: soft-cost field loaded from %s "
                    "(%d pts, grid %s, mode=%s, weight=%.3f)",
                    soft_payload.get("npz_path"),
                    len(self._soft_cost_field.points),
                    tuple(self._soft_cost_field.grid.shape),
                    self._soft_cost_mode,
                    self._soft_cost_weight,
                )
            except Exception:
                logger.exception(
                    "load_obstacles: failed to load soft_cost payload %s — "
                    "continuing WITHOUT soft-cost scoring",
                    soft_payload,
                )

        self._loaded_config_hash = cfg_hash
        logger.info(
            "RRTToGoalPlanner.load_obstacles: %d obstacle(s) loaded (%s)",
            len(self._loaded_obstacle_ids),
            ", ".join(self._obstacle_names.values()) or "<none>",
        )
        return list(self._loaded_obstacle_ids)

    def _load_one_object(self, obj: dict) -> int | None:
        """Create a pybullet body for ``obj`` (a serialized ObjectConfig dict).

        Returns the body id, or None if the type is unsupported / no collision
        geometry is available.
        """
        obj_type = obj.get("__type__", "")
        position = self._resolve_position(obj)
        quat = self._resolve_quat(obj)
        scale = obj.get("current_scale") or [1.0, 1.0, 1.0]
        logger.info(
            "_load_one_object: name=%s type=%s position=%s quat=%s scale=%s "
            "(raw: current_pos=%s, base_pos=%s, range_x=%s, range_y=%s, range_z=%s)",
            obj.get("name"),
            obj_type,
            position,
            quat,
            scale,
            obj.get("current_position"),
            obj.get("base_position"),
            obj.get("position_range_x"),
            obj.get("position_range_y"),
            obj.get("position_range_z"),
        )

        if obj_type == "CuboidObjectConfig":
            # SplatSim's create_box uses the raw size with no scale multiplication
            # (CuboidObjectConfig has no scaling_range; size is fixed at load time).
            size = obj.get("size") or (1.0, 1.0, 1.0)
            half = [s / 2.0 for s in size]
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self._pb_client)
            # Visual shape with the color the sim uses. Same source of truth
            # (CuboidObjectConfig.color_rgb, published in the oracle env
            # config as `color_rgb` — three ints in 0-255). Without a
            # visual shape pybullet renders the collision shape in a default
            # gray, making planner-side pybullet-GUI diagnostics ambiguous
            # (obstacle_1 vs obstacle_2 vs block indistinguishable). Alpha
            # fixed at 1.0 — the oracle config carries no transparency.
            _rgb = obj.get("color_rgb")
            _visual_kwargs = {}
            if _rgb is not None and len(_rgb) >= 3:
                _visual_kwargs["rgbaColor"] = [
                    float(_rgb[0]) / 255.0,
                    float(_rgb[1]) / 255.0,
                    float(_rgb[2]) / 255.0,
                    1.0,
                ]
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self._pb_client, **_visual_kwargs
            )
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=list(position),
                baseOrientation=list(quat),
                physicsClientId=self._pb_client,
            )

        if obj_type == "SplatObjectConfig":
            urdf_path = obj.get("urdf_path")
            if not urdf_path:
                logger.debug("Skipping splat obstacle '%s' (no urdf_path)", obj.get("name"))
                return None
            try:
                from splatsim.utils.paths import resolve_splatsim_path

                resolved = resolve_splatsim_path(urdf_path)
            except Exception:
                resolved = urdf_path
            # Mirror SplatSim's two-step load: open the URDF at `base_position`
            # with identity orientation (matches SplatSim's `load_urdf`), then
            # call `resetBasePositionAndOrientation` to teleport it to the
            # final placement with the actual quaternion (matches their
            # `randomize_object_pose` which always applies the final pose via
            # reset). Some URDFs bake in mesh transforms that interact badly
            # with a non-identity quaternion at load time, so combining both
            # steps into a single loadURDF call can render the visual mesh in
            # the wrong place even though the link origin is correct.
            base_position = obj.get("base_position") or [0.0, 0.0, 0.0]
            # PyBullet only supports uniform globalScaling. SplatSim approximates
            # per-axis scaling with the geometric mean (cbrt of product); see
            # randomize_object_scale in sim_robot_pybullet_base.py.
            if scale:
                physics_scale = float(np.cbrt(float(scale[0]) * float(scale[1]) * float(scale[2])))
            else:
                physics_scale = 1.0
            try:
                body_id = p.loadURDF(
                    str(resolved),
                    basePosition=list(base_position),
                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                    useFixedBase=True,
                    globalScaling=physics_scale,
                    physicsClientId=self._pb_client,
                )
            except p.error as e:
                logger.warning("Failed to load splat obstacle '%s' from %s: %s", obj.get("name"), resolved, e)
                return None
            p.resetBasePositionAndOrientation(
                body_id,
                list(position),
                list(quat),
                physicsClientId=self._pb_client,
            )
            return body_id

        logger.debug("Unsupported object type '%s' (name=%s) — skipping", obj_type, obj.get("name"))
        return None

    @staticmethod
    def _resolve_position(obj: dict) -> tuple[float, float, float]:
        """Mirror SplatSim's pose-placement formula.

        ``randomize_object_pose`` (sim_robot_pybullet_base.py) computes
        ``pos = [x + bp[0], y + bp[1], z + bp[2]]`` where ``(x,y,z)`` is the
        sample from ``position_range_*`` (or its midpoint when no random) and
        ``bp = base_position``. Picking just one of the two — as the previous
        implementation did — drops the YAML-provided height offset
        (e.g. ``small_engine_new`` has ``base_position=[0, 0, 0.180955]`` to
        sit on the table; the table-relative xy comes from position_range).

        Preference order:
          1. ``current_position`` if the sim has updated it past the default
             [0,0,0] (i.e. a get_observations has fired post-placement).
          2. ``initial_position`` (set by ``randomize_object_pose`` at episode
             start) for the same reason.
          3. ``base_position + position_range_midpoint`` as the formula
             fallback when neither live field is populated yet.
        """
        for key in ("current_position", "initial_position"):
            v = obj.get(key)
            if v and any(abs(float(x)) > 1e-12 for x in v):
                return (float(v[0]), float(v[1]), float(v[2]))
        bp = obj.get("base_position") or [0.0, 0.0, 0.0]
        rx = obj.get("position_range_x") or (0.0, 0.0)
        ry = obj.get("position_range_y") or (0.0, 0.0)
        rz = obj.get("position_range_z") or (0.0, 0.0)
        return (
            float(bp[0]) + (float(rx[0]) + float(rx[1])) / 2.0,
            float(bp[1]) + (float(ry[0]) + float(ry[1])) / 2.0,
            float(bp[2]) + (float(rz[0]) + float(rz[1])) / 2.0,
        )

    @staticmethod
    def _resolve_quat(obj: dict) -> tuple[float, float, float, float]:
        """Pick the most informative orientation, with the same precedence as
        ``_resolve_position``. Skips current_quat / initial_quat when they're
        still the identity default (since asdict will always include them as
        [0,0,0,1] until the sim updates them); falls through to base_quat,
        which is YAML-provided and meaningful for static configs.
        """
        identity = (0.0, 0.0, 0.0, 1.0)
        for key in ("current_quat", "initial_quat"):
            v = obj.get(key)
            if v and len(v) == 4 and any(abs(float(x) - d) > 1e-9 for x, d in zip(v, identity, strict=True)):
                return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        bq = obj.get("base_quat")
        if bq and len(bq) == 4:
            return (float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3]))
        return identity

    # ------------------------------------------------------------------ #
    #  Planning                                                          #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        q_start: np.ndarray,
        target_ee_pos: np.ndarray,
        target_ee_quat: np.ndarray,
        q_goal_bias: np.ndarray | None = None,
        recent_joint_velocity: np.ndarray | None = None,
        exclude_q_goals: list[np.ndarray] | None = None,
        start_vel: np.ndarray | None = None,
        actual_gripper_q: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Plan a joint-space trajectory to an end-effector pose.

        Mirrors SplatSim's ``TrajectoryGenerator._resolve_ee_pose_to_q_candidates``
        + ``_plan_with_fallback_goals``: solves IK to ``(target_ee_pos, target_ee_quat)``
        from multiple seeds — the first attempt seeded by ``q_goal_bias`` (so demos
        converge to a canonical configuration when feasible), the rest from random
        seeds — and runs RRT against each collision-free candidate until one
        succeeds.  This is much less constrained than fixing ``q_goal_bias`` as
        the only goal, which often has no valid path through cluttered scenes.

        `exclude_q_goals`: optional list of joint configurations to FILTER OUT
        of the IK candidate set before planning. Used by the retry-on-collision
        path: when a path executed in sim collides (typically because the parametrizer
        smoothing curved through an obstacle the RRT-raw path didn't), the
        source adds that path's q_goal to this list and re-calls plan() — the
        filter discards any candidate within ~0.05 rad (per-joint L2) of an
        excluded goal, so we don't immediately re-pick the same IK branch.
        Empty / None preserves historical behavior.

        After a successful plan, the chosen q_goal is also published on
        ``self._last_chosen_q_goal`` so callers can record it without
        scraping the trajectory's terminal pose.

        Returns ``(traj, escape_end_q)`` where:
          * ``traj`` is the time-parametrized RRT chunk of shape (T, num_dofs)
            sampled at ``self._fps``. **NEW (formerly traj contained
            prepended escape waypoints):** the escape segment is no longer
            included in ``traj``. Callers that have access to the env must
            teleport the robot to ``escape_end_q`` before executing ``traj``,
            so the env's robot is physically at ``traj[0]`` at chunk t=0.
          * ``escape_end_q`` is the collision-free joint config the planner
            escaped to (== ``traj[0]``), or ``None`` if no escape was needed.
            Used as a signal to the source: "if non-None, env teleport is
            required". When the historical lookback path also wants to
            teleport q_start_full into the env, ``escape_end_q``'s teleport
            REPLACES that (you don't want to teleport to the wedged config
            first; the planner already moved past it).

        Why no longer prepend escape: the escape waypoints were intentionally
        un-smoothed (large per-step deltas to overcome PD-controller contact
        forces in sim), which produced 10×-mean-delta outlier frames at the
        start of recorded intervention episodes. Those outliers contaminated
        the DAgger training distribution (diffusion policy's score field
        learned to associate wedged-state observations with discrete
        pushout actions — a sim-PD artifact, not a transferable skill).
        Teleporting in the env achieves the same physical end-state without
        recording the artifact. The planner's iterative escape search is
        unchanged — only the env-side replay is bypassed.

        Raises ``RRTPlanningError`` if no IK candidate can be reached.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision
        # parametrize_path is no longer used inline here — the per-IK
        # loop now calls self._smooth_and_check_collision which owns both
        # the parametrize and the dense collision check.

        q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)[: self._num_dofs]
        target_ee_pos = np.asarray(target_ee_pos, dtype=np.float64).reshape(-1)[:3]
        target_ee_quat = np.asarray(target_ee_quat, dtype=np.float64).reshape(-1)[:4]
        # Snap the planner's gripper joints to the env's ACTUAL gripper
        # config BEFORE any collision check runs. Every downstream check
        # (start-in-collision precheck, escape rewind, IK candidate filter,
        # BiRRT sample/extend, time-parametrized dense check) then evaluates
        # against a gripper geometry that matches the real robot's fingers.
        # Without this, birrt_path's `set_robot_joint_positions` forces
        # `open_gripper()` and every check downstream runs on wide-open
        # fingers while the real robot's fingers are closing around an
        # object — the exact cause of the "escape rewound to safe frame,
        # time-parametrized path collides at waypoint 0" cascade. Re-snapped
        # after every get_path call inside `_generate_paths_for_ik` (which
        # goes through set_robot_joint_positions again). None from the
        # caller = leave gripper untouched (legacy behavior).
        self._current_actual_gripper_q = (
            float(actual_gripper_q) if actual_gripper_q is not None else None
        )
        self._snap_gripper_to_actual()
        if q_goal_bias is not None:
            q_goal_bias = np.asarray(q_goal_bias, dtype=np.float64).reshape(-1)[: self._num_dofs]

        # Snapshot every joint (positions and velocities) so we can fully restore
        # the robot's pose after planning — the wrapper's pybullet client is
        # shared with FK/IK code paths and would otherwise be left in an
        # arbitrary state.
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        saved_joint_states: list[tuple[int, float, float]] = []
        for i in range(n_joints):
            s = p.getJointState(self._robot_id, i, physicsClientId=self._pb_client)
            saved_joint_states.append((i, float(s[0]), float(s[1])))

        # Freeze the visualizer's world redraw while planning when configured
        # (freeze_visualizer_during_plan). Historically always-disabled for
        # speed, then re-enabled per user request for debugging visibility —
        # now a per-consumer choice: traj-gen batches freeze it (planning is
        # GUI-redraw-bound otherwise), the DAgger side keeps it visible.
        _froze_viz = False
        if self._freeze_visualizer_during_plan:
            try:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pb_client
                )
                _froze_viz = True
            except Exception:
                pass

        try:
            # If the start config is in collision (the policy got stuck), try to
            # escape along the aggregated outward contact normal before planning.
            # The robot is the only thing in the wrapper's pybullet client besides
            # the loaded obstacles, so all contacts are robot↔obstacle.
            escape_path: np.ndarray | None = None
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                q_start,
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=True,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **self._collision_kwargs,
            ):
                logger.info(
                    "Start config in collision; attempting escape chain (contact-normal → self-collision gradient)..."
                )
                escape_path = self._try_escape_chain(q_start)
                if escape_path is None:
                    raise RRTPlanningError(
                        "Start in collision and all escape modes "
                        "(contact-normal, self-collision gradient) failed to clear it"
                    )
                logger.info("Escape produced %d waypoint(s); replanning from new start", len(escape_path))
                q_start = escape_path[-1].copy()

            # Resolve the target EE pose to multiple collision-free joint-space
            # candidates. First attempt is seeded with q_goal_bias (when provided).
            candidates = self._resolve_ee_pose_to_q_candidates(target_ee_pos, target_ee_quat, q_goal_bias)
            if len(candidates) == 0:
                raise RRTPlanningError("No collision-free IK solution found for target EE pose")
            n_before_exclude = len(candidates)
            # Retry-on-collision filter: drop any IK candidate whose joint
            # config is close to one previously found to collide when its
            # path was executed. The tolerance is generous (0.05 rad per-
            # joint L2) so we drop the SAME IK branch, not a genuinely
            # different solution that happens to be near it.
            if exclude_q_goals:
                tol = 0.05
                filtered = []
                for q in candidates:
                    q_arr = np.asarray(q)
                    if any(np.linalg.norm(q_arr - np.asarray(eq)) < tol for eq in exclude_q_goals):
                        continue
                    filtered.append(q)
                if len(filtered) < n_before_exclude:
                    logger.info(
                        "Collision-history filter dropped %d of %d IK candidate(s)",
                        n_before_exclude - len(filtered),
                        n_before_exclude,
                    )
                candidates = filtered
                if not candidates:
                    raise RRTPlanningError(
                        f"All {n_before_exclude} IK candidate(s) were excluded by "
                        f"the collision-history filter — no untried branch remains"
                    )
            logger.info("Resolved EE goal to %d collision-free IK candidate(s)", len(candidates))

            # Planning loop. The two scoring axes work TOGETHER (not
            # mutually exclusive):
            #
            #   * `ik_goal_selection` (when set) — sorts IK candidates by
            #     goal-state geometry (e.g. joint distance from q_start),
            #     best-first. The planner walks through candidates in
            #     this order. When None, candidates keep their original
            #     order from the IK solver.
            #
            #   * `path_selection` — scores the actual planned path(s) to
            #     a given IK goal. With one RRT attempt per IK there's
            #     a single path per candidate, but path_selection still
            #     determines which IK candidate's path wins when MULTIPLE
            #     candidates yield successful RRT plans.
            #
            # Algorithm:
            #   For each IK candidate (in IK-sorted order):
            #     - Run RRT once. If failure, fall through to next IK.
            #     - On success, score the path by path_selection.
            #     - Track the running best. If `early_exit_on_first_ik` is
            #       True (the IK-goal-selection contract: "first IK with
            #       any successful path wins"), break immediately. If
            #       False (no IK ordering — original "try-all-and-rank"
            #       behavior), keep going to find the minimum-path-score
            #       winner across all IKs.
            if self._path_selection == PathSelectionStrategy.JOINT_VELOCITY_MATCH:
                if recent_joint_velocity is None:
                    raise RRTPlanningError(
                        "PathSelectionStrategy.JOINT_VELOCITY_MATCH requires `recent_joint_velocity` "
                        "to be passed to plan(); none provided. Either supply a velocity history or "
                        "switch to EE_ARC_LENGTH / JOINT_ARC_LENGTH."
                    )
                recent_vel = np.asarray(recent_joint_velocity, dtype=np.float64).reshape(-1)[: self._num_dofs]
            else:
                recent_vel = None

            path_score_label, path_score_units = {
                PathSelectionStrategy.EE_ARC_LENGTH: ("EE arc-length", "m"),
                PathSelectionStrategy.JOINT_ARC_LENGTH: ("joint arc-length", "rad"),
                PathSelectionStrategy.JOINT_VELOCITY_MATCH: ("joint-velocity deviation", "rad/step"),
                # MIN_PAIR_CLEARANCE scoring returns NEGATED min distance — the
                # score "label" reflects the underlying quantity (min pair gap,
                # meters), but lower is better as elsewhere because of the
                # negation. Reader interpretation: more negative = larger
                # actual gap = safer path.
                PathSelectionStrategy.MIN_PAIR_CLEARANCE: ("neg min-pair clearance", "m"),
                PathSelectionStrategy.CAMERA_SCORING: ("neg camera alignment", ""),
            }[self._path_selection]

            # Decide the IK try-order.
            if self._ik_goal_selection is not None:
                ik_score_label, ik_score_units = {
                    IkGoalSelectionStrategy.JOINT_DISTANCE: ("joint Δ from start", "rad"),
                }[self._ik_goal_selection]
                ordered = sorted(
                    enumerate(candidates),
                    key=lambda iq: self._score_ik_candidate(iq[1], q_start),
                )
                early_exit_on_first_ik = True
                ik_order_log = f"IK-sorted by {ik_score_label}"
            else:
                ordered = list(enumerate(candidates))
                early_exit_on_first_ik = False
                ik_order_log = "IK in original solver order"
                ik_score_label, ik_score_units = None, None

            path = None
            chosen_q_goal = None
            best_path_score = float("inf")
            chosen_ik_score: float | None = None
            traj: np.ndarray | None = None  # cached smoothed path of the winning IK
            _loop_ordered = ordered  # alias kept so logging uses a stable name
            for tried, (orig_i, q_goal) in enumerate(ordered, start=1):
                ik_score = (
                    self._score_ik_candidate(q_goal, q_start) if self._ik_goal_selection is not None else None
                )
                if ik_score is not None:
                    logger.info(
                        "IK candidate %d/%d (orig idx %d) %s=%.3f %s — running RRT (up to %d path candidate(s))",
                        tried,
                        len(_loop_ordered),
                        orig_i + 1,
                        ik_score_label,
                        ik_score,
                        ik_score_units,
                        self._num_path_candidates_per_ik,
                    )
                else:
                    logger.info(
                        "Trying RRT against IK candidate %d/%d (up to %d path candidate(s))",
                        tried,
                        len(_loop_ordered),
                        self._num_path_candidates_per_ik,
                    )
                # Generate one OR more RRT paths to this IK candidate.
                # With num_path_candidates_per_ik=1 (default) this is a
                # single get_path call; with >1, the helper perturbs
                # endpoints to force RRT to find distinct paths.
                candidate_paths = self._generate_paths_for_ik(q_start, q_goal)
                if not candidate_paths:
                    continue
                # Among this IK's path candidates, pick the best-scored one
                # whose PARAMETRIZED form is collision-free. Score all
                # candidates, sort best-first, then parametrize + dense-check
                # each in score order — take the first that passes. Falling
                # back to a lower-scored path WITHIN the same IK goal is
                # almost always preferable to giving up on the IK goal and
                # trying the next IK (which may itself fail the same way).
                # When num_path_candidates_per_ik=1, this collapses to "try
                # the single path; if its parametrized form collides, treat this IK
                # as failed".
                # CAMERA_SCORING aims the wrist camera at this goal's EE
                # position — stash the goal so `_path_camera_score` can FK it.
                self._score_goal_q = q_goal
                _scored_cps: list[tuple[float, np.ndarray]] = [
                    (self._score_candidate(cp, recent_vel), cp) for cp in candidate_paths
                ]
                _scored_cps.sort(key=lambda x: x[0])
                local_best_score = float("inf")
                local_best_path: np.ndarray | None = None
                local_best_traj: np.ndarray | None = None
                for _rank, (_s, _cp) in enumerate(_scored_cps, start=1):
                    # Snap endpoints to the exact start/goal so smoothing
                    # doesn't drift (same fix as the post-loop block used
                    # to do for the global winner).
                    _cp_snapped = list(_cp)
                    _cp_snapped[0] = q_start
                    _cp_snapped[-1] = q_goal
                    _cp_arr = np.asarray(_cp_snapped, dtype=np.float64)
                    # NOTE: trajopt + elastic do NOT run here. They are applied
                    # exactly once, to the GLOBAL winner, after both loops (see
                    # the `_postprocess_path` block below `_last_chosen_q_goal`).
                    # Running them per candidate — even lazily, in score order —
                    # meant every gate rejection paid a full trajopt: a scene
                    # where all 5 candidates fail cost 5 runs per IK goal and
                    # produced nothing. Gate first, optimize the survivor.
                    # BOTH modes: cheap linear-densified check of the RAW path
                    # first, at the planner's FULL collision contract. This is
                    # exactly what `_validate_final_trajectory` stage 2 asserts
                    # on the winner — parametrize_per_candidate=True mode (see that kwarg)
                    # historically skipped it and validated candidates only on
                    # their time-parametrized form at the PARAMETRIZER-SCALED (looser)
                    # clearance, so a raw path inside the full clearance band
                    # sailed through the loop and then died at the final gate
                    # as a hard RRTPlanningError instead of a recoverable
                    # "try the next candidate" (vine bench: finger pad at
                    # 12.9 mm passed the 1 cm parametrizer-scaled check, failed the
                    # 2 cm gate).
                    _cp_checked, _coll_idx = self._densify_and_check_collision(_cp_arr)
                    _coll_kind = "linear-densified"
                    if _coll_idx is None and self._parametrize_per_candidate:
                        _cp_checked, _coll_idx = self._smooth_and_check_collision(
                            _cp_arr, start_vel,
                        )
                        _coll_kind = "time-parametrized"
                    if _coll_idx is not None:
                        logger.info(
                            "  path %d/%d (score=%.4f): %s path collides "
                            "at waypoint %d — trying next path candidate.",
                            _rank,
                            len(_scored_cps),
                            _s,
                            _coll_kind,
                            _coll_idx,
                        )
                        continue
                    # Densify mode: linear-densify passed, but the parametrizer's C² spline
                    # can curve wider than the linear chord near sharp corners
                    # and collide with an obstacle the chord clears. Do the
                    # parametrization check NOW (one parametrize call per surviving
                    # candidate); if it fails, try the next-best candidate.
                    # Historically we warned + executed the colliding trajectory
                    # anyway — that's how the gripper/wrist ended up bumping
                    # box obstacles at corners. Bounded cost: at most
                    # len(_scored_cps) parametrize calls per IK in the worst case,
                    # ~1-2 in practice.
                    if not self._parametrize_per_candidate:
                        _smoothed_traj, _smoothed_coll = self._smooth_and_check_collision(
                            _cp_checked, start_vel,
                        )
                        if _smoothed_coll is not None:
                            logger.info(
                                "  path %d/%d (score=%.4f): linear-densify clean "
                                "but time-parametrized collides at waypoint %d/%d "
                                "— trying next path candidate.",
                                _rank,
                                len(_scored_cps),
                                _s,
                                _smoothed_coll,
                                _smoothed_traj.shape[0],
                            )
                            continue
                        # Time-parametrized is also clean. Cache the smoothed
                        # trajectory so the post-outer-loop block doesn't
                        # need to re-parametrize it.
                        _cp_checked = _smoothed_traj
                    # Found one whose checked form is collision-free.
                    local_best_score = _s
                    # The POST-PROCESSED, endpoint-snapped path — this is what
                    # produced `local_best_traj`, so it is what the final
                    # gate's sparse-waypoint stage must validate.
                    local_best_path = _cp_arr
                    # Both branches now store the time-parametrized trajectory.
                    local_best_traj = _cp_checked
                    break
                if local_best_path is None:
                    logger.warning(
                        "IK candidate %d/%d: all %d path(s) had time-parametrized "
                        "collisions — moving to next IK candidate.",
                        tried,
                        len(_loop_ordered),
                        len(_scored_cps),
                    )
                    continue
                assert local_best_traj is not None
                # MIN_PAIR_CLEARANCE's raw score is the NEGATED min distance —
                # flip the sign and rename the label for the log so the user
                # sees "min-pair clearance = 0.012 m" instead of the
                # confusing "neg min-pair clearance = -0.012 m". The
                # underlying score field stays negated for sort consistency
                # with the other strategies' lower-is-better convention.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _display_val = -local_best_score
                    _display_label = "min-pair clearance"
                    _display_units = "m (more = better)"
                else:
                    _display_val = local_best_score
                    _display_label = path_score_label
                    _display_units = path_score_units
                logger.info(
                    "IK candidate %d/%d: %d path(s) generated; best path %s=%.4f %s",
                    tried,
                    len(_loop_ordered),
                    len(candidate_paths),
                    _display_label,
                    _display_val,
                    _display_units,
                )
                if local_best_score < best_path_score:
                    best_path_score = local_best_score
                    path = local_best_path
                    traj = local_best_traj  # cache the smoothed path so we don't re-parametrize below
                    chosen_q_goal = q_goal
                    chosen_ik_score = ik_score
                    if early_exit_on_first_ik:
                        # IK-goal-selection contract: best IK with ANY successful
                        # plan wins. Within that IK, path_selection picked the
                        # best of its candidate paths above. Other IKs aren't
                        # tried.
                        break

            if path is None or chosen_q_goal is None:
                raise RRTPlanningError(f"RRT failed for all {len(_loop_ordered)} IK goal candidate(s)")
            if chosen_ik_score is not None:
                # Same per-strategy display fix as the per-IK log above —
                # MIN_PAIR_CLEARANCE shows positive min distance.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _disp_score = -best_path_score
                    _disp_label = "min-pair clearance"
                    _disp_units = "m (more = better)"
                else:
                    _disp_score = best_path_score
                    _disp_label = path_score_label
                    _disp_units = path_score_units
                logger.info(
                    "Picked path (%s; chose first IK with successful plan) — IK %s=%.3f %s, path %s=%.4f %s",
                    ik_order_log,
                    ik_score_label,
                    chosen_ik_score,
                    ik_score_units,
                    _disp_label,
                    _disp_score,
                    _disp_units,
                )
            else:
                # Same per-strategy display fix.
                if self._path_selection == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
                    _disp_score = -best_path_score
                    _disp_label = "min-pair clearance"
                    _disp_units = "m (more = better)"
                else:
                    _disp_score = best_path_score
                    _disp_label = path_score_label
                    _disp_units = path_score_units
                logger.info(
                    "Picked best path (%s) by %s=%.4f %s",
                    ik_order_log,
                    _disp_label,
                    _disp_score,
                    _disp_units,
                )

            # Publish the chosen IK goal so callers (RRTGuidanceSource) can
            # track it across plan() calls without having to scrape the
            # trajectory's terminal pose. Set BEFORE parametrization so the value
            # reflects the actual IK solution, not the post-parametrization endpoint
            # (which might drift by tiny amounts from boundary effects).
            self._last_chosen_q_goal = np.asarray(chosen_q_goal, dtype=np.float64).copy()

            # Trajopt + elastic, ONCE, on the global winner. Everything above
            # ranked and gated the shortcut-smoothed geometry, so exactly one
            # path per plan() reaches these passes — the expensive ones (a
            # trajopt pass is 2*DOF min-distance queries per waypoint, ~10 ms
            # each on a concave mesh).
            #
            # The winner is already gate-clean, and both passes are internally
            # hard-collision-gated, so this can only refine it. Re-gate anyway
            # (the optimized geometry is what gets executed) and fall back to
            # the pre-optimization pair on failure — never worse than not
            # having run them, and never a wasted rejection.
            if self._postprocess_after_ranking:
                _opt_path = self._postprocess_path(path)
                if _opt_path is not path:
                    _opt_traj = None
                    _, _opt_coll = self._densify_and_check_collision(_opt_path)
                    if _opt_coll is None:
                        _opt_traj, _opt_coll = self._smooth_and_check_collision(
                            _opt_path, start_vel,
                        )
                    if _opt_coll is None and _opt_traj is not None:
                        path, traj = _opt_path, _opt_traj
                    else:
                        logger.info(
                            "trajopt/elastic output collides at waypoint %d — "
                            "keeping the pre-optimization path.",
                            _opt_coll,
                        )

            # `traj` is already populated from the per-IK loop above
            # (_smooth_and_check_collision was called there per path
            # candidate; the winning path's smoothed form was cached).
            # Historically, time parametrization happened here as a single
            # post-loop step on the global winner, and the smoothed path
            # was never collision-checked — so a spline that curved
            # through obstacles would silently reach the env. Per-IK
            # smoothing+checking moved that logic up, so we just use the
            # cached `traj` here.
            #
            # Why escape isn't prepended: the escape segment was historically
            # PREPENDED raw (un-smoothed) so each escape waypoint became one
            # env-step command — needed because the simulator's PD controller
            # wouldn't overcome contact forces with smooth sub-samples.
            # That left 10×-mean-delta outlier frames in the recorded dataset
            # that corrupted the diffusion policy's score field. Now: escape
            # segment is NOT included in `traj` — `escape_end_q` is returned
            # separately so callers can teleport the env's robot directly to
            # the post-escape config, achieving the same physical end-state
            # without recording the artifact.
            assert traj is not None  # guaranteed when path/chosen_q_goal are set

            # Densify mode (parametrize_per_candidate=False) USED TO parametrize
            # the winner here (one cloud-API call per plan) and then log a
            # WARNING if the smoothed path collided but execute it anyway —
            # which is how gripper/wrist tips ended up bumping obstacles at
            # corners where the parametrizer's spline curved outside the linear chord.
            # The per-candidate loop above now does the parametrization check inline
            # and caches the smoothed trajectory in `local_best_traj` when it
            # passes, so `traj` is already time-parametrized and clean here in
            # both modes — no additional parametrize call needed.

            # Final three-stage collision gate on the trajectory we're about
            # to hand back. Stage 3 (time-parametrized) is what the per-IK loop
            # already checks — running it here catches drift between check and
            # return. Stages 1 (sparse RRT) and 2 (linear-densified) are
            # diagnostic breadcrumbs: if we ever see collisions in the
            # recorded data despite this gate, the log line tells us which
            # resolution first exposed it (and thus WHERE the earlier check
            # missed). See `_validate_final_trajectory` docstring for the
            # motivation.
            _final_gate = self._validate_final_trajectory(path, traj)
            if _final_gate is not None:
                _stage, _idx = _final_gate
                _n_wp = (
                    path.shape[0] if _stage == "sparse RRT waypoints"
                    else traj.shape[0] if _stage == "time-parametrized trajectory"
                    else -1
                )
                raise RRTPlanningError(
                    f"Final collision gate failed: {_stage} collides at "
                    f"waypoint {_idx}"
                    + (f"/{_n_wp}" if _n_wp > 0 else "")
                    + " — refusing to return a colliding trajectory."
                )

            # Escape segment is NO LONGER prepended (see docstring + previous
            # comment for the rationale). Instead, return the post-escape
            # config so the caller can teleport the env's robot directly to
            # `traj[0]`. ``escape_path[-1]`` equals ``traj[0]`` by construction
            # (the planner ran from `q_start = escape_path[-1].copy()`), so the
            # teleport puts the robot exactly where the chunk begins.
            escape_end_q: np.ndarray | None = None
            if escape_path is not None and len(escape_path) >= 1:
                escape_end_q = np.asarray(escape_path[-1], dtype=np.float64)

            self._debug_draw_soft_cost_path(traj)
            return traj, escape_end_q
        finally:
            if _froze_viz:
                try:
                    p.configureDebugVisualizer(
                        p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pb_client
                    )
                except Exception:
                    pass
            for i, pos, vel in saved_joint_states:
                p.resetJointState(self._robot_id, i, pos, vel, physicsClientId=self._pb_client)
            # Re-assert POSITION_CONTROL holds on the ARM joints at the
            # restored positions. Planning internals (IK candidate sampling,
            # set_robot_joint_positions(hold=True) in collision checks) issue
            # setJointMotorControl2 with their own targets; resetJointState
            # alone leaves those STALE motor targets active, so on a shared
            # live sim client the PD motors drag the robot away from the
            # restored pose toward the last sampled config as soon as physics
            # steps again (e.g. env reset's settle loop right after the
            # reset-time plan-feasibility check) — observed as "robot doesn't
            # return to the start configuration after the RRT check".
            # Arm joints only: position-holding the gripper mimic children
            # would overpower the JOINT_GEAR mimic and freeze the gripper
            # (see setup_gripper). Harmless on the DAgger side, where the
            # planning client is separate from the live env.
            _restored_pos = {i: pos for i, pos, _vel in saved_joint_states}
            for j_idx in self._joint_indices:
                if j_idx in _restored_pos:
                    p.setJointMotorControl2(
                        self._robot_id,
                        j_idx,
                        p.POSITION_CONTROL,
                        targetPosition=_restored_pos[j_idx],
                        force=150,
                        maxVelocity=3.14,
                        physicsClientId=self._pb_client,
                    )
            # (No render re-enable here: the pre-plan disable above was removed
            # per user request. Left as a marker so it's obvious this was
            # deliberate, not forgotten.)

    def _validate_final_trajectory(
        self,
        path: np.ndarray,
        traj: np.ndarray,
    ) -> tuple[str, int] | None:
        """Belt-and-suspenders three-stage collision check on the trajectory
        that plan() is about to return. The per-IK loop already rejects
        colliding candidates, but this is the last line of defense against
        anything that could slip through: floating-point re-evaluation drift,
        joint-state changes between check and return, or a subtle
        parameter/order difference between the per-candidate check and this
        one.

        Ordered increasing-granularity so the log identifies WHICH
        representation first failed — a diagnostic breadcrumb when we
        eventually see the "still colliding" report despite the inline check:
          1. Sparse RRT waypoints (`path`)     — BiRRT should emit clean
                                                  waypoints; failure here means
                                                  the raw check was off.
          2. Linear-densified `path` (0.02 rad) — catches obstacles between
                                                  sparse waypoints that the
                                                  chord skips over.
          3. Time-parametrized `traj`             — the actually-executed path;
                                                  catches obstacles that
                                                  the parametrizer's spline curves into
                                                  at sharp corners the linear
                                                  chord clears.

        Returns None on success. On failure returns
        `(stage_name, waypoint_idx)` for the first stage/waypoint that
        collided; caller raises RRTPlanningError with that context so
        trajectory-gen retries the scenario instead of recording a
        colliding replay.
        """
        from splatsim.utils.rrt_path_utils import (
            check_links_in_collision,
            resample_path_by_distance,
        )
        # Stage 1: sparse RRT waypoints.
        for k in range(path.shape[0]):
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                path[k],
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **self._collision_kwargs,
            ):
                return ("sparse RRT waypoints", k)
        # Stage 2: linear-densified path at ~0.02 rad joint spacing (matches
        # _densify_and_check_collision's step size).
        if path.shape[0] >= 2:
            total_joint_travel = float(np.sum(np.abs(np.diff(path, axis=0))))
            n_points = int(np.clip(total_joint_travel / 0.02, path.shape[0], 2000))
            dense = np.asarray(
                resample_path_by_distance(path, n_points), dtype=np.float64,
            )
            for k in range(dense.shape[0]):
                if check_links_in_collision(
                    self._robot_id,
                    self._joint_indices,
                    dense[k],
                    self._loaded_obstacle_ids,
                    obstacle_names=self._obstacle_names,
                    skip_pairs=self._skip_pairs,
                    verbose=False,
                    physics_client_id=self._pb_client,
                    link_indices_to_check=self._planner_link_indices_to_check,
                    **self._collision_kwargs,
                ):
                    return ("linear-densified path", k)
        # Stage 3: time-parametrized trajectory (the one actually replayed).
        # Uses the parametrizer-scaled obstacle clearance (default 0.5×) — a bulge
        # that stays within the halved bound but outside the raw path's
        # bound is still safe to execute, no reason to reject.
        _smoothed_kwargs = self._smoothed_collision_kwargs()
        for k in range(traj.shape[0]):
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                traj[k],
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **_smoothed_kwargs,
            ):
                return ("time-parametrized trajectory", k)
        return None

    def _densify_and_check_collision(
        self,
        rrt_waypoints: np.ndarray,
    ) -> tuple[np.ndarray, int | None]:
        """Cheap per-candidate collision validation WITHOUT time parametrization. Linearly
        densifies the raw BiRRT path (so collision checks don't tunnel between
        sparse waypoints) and dense-checks each interpolated config under the
        planner's collision contract. Returns ``(rrt_waypoints, first_coll_idx)``
        — the RAW path is returned UNCHANGED (the caller parametrizes the
        winner exactly once after the candidate loop); ``first_coll_idx is None``
        means collision-free.

        Used when ``parametrize_per_candidate=False`` (SplatSim trajectory-gen) so
        plan() parametrizes only once per call instead of once per
        candidate. Caller restores joint state (plan()'s finally block).
        """
        from splatsim.utils.rrt_path_utils import (
            check_links_in_collision,
            resample_path_by_distance,
        )

        rrt_waypoints = np.asarray(rrt_waypoints, dtype=np.float64)
        if rrt_waypoints.shape[0] < 2:
            return rrt_waypoints, None
        # Densify to ~0.02 rad joint spacing (fine enough that a link can't
        # tunnel through a thin obstacle between samples), capped so long paths
        # don't blow up the check count.
        total_joint_travel = float(np.sum(np.abs(np.diff(rrt_waypoints, axis=0))))
        n_points = int(np.clip(total_joint_travel / 0.02, rrt_waypoints.shape[0], 2000))
        dense = np.asarray(resample_path_by_distance(rrt_waypoints, n_points), dtype=np.float64)
        for k in range(dense.shape[0]):
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                dense[k],
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **self._collision_kwargs,
            ):
                return rrt_waypoints, k
        return rrt_waypoints, None

    def _postprocess_path(self, path: np.ndarray) -> np.ndarray:
        """Run trajopt + elastic smoothing on ONE path, outside `get_path`.

        These two passes used to run inside `get_path`, i.e. on EVERY path
        candidate, before `plan()` had scored any of them — so a 5-candidate
        IK paid 5x their cost and threw 4 of the results away. `plan()`
        already sorts candidates by score and tries them one at a time, so
        the cheapest correct place for them is right before the collision
        gate of the candidate actually being considered: normally one run per
        plan, and at most one per candidate the gate rejects.

        Trade-off: ranking now scores the SHORTCUT-SMOOTHED path rather than
        the fully post-processed one. Trajopt's repulsion term only fires
        within `trajopt_collision_threshold` of an obstacle, so candidates
        that hug obstacles grow while free-space candidates only shrink —
        enough asymmetry to flip near-ties in arc-length-style metrics.
        `MIN_PAIR_CLEARANCE` is the strategy most exposed to this, since
        raising clearance is precisely what trajopt does; set
        `postprocess_after_ranking=False` to restore rank-after-postprocess
        if that matters more than the planning time.

        Pass ORDER is preserved from `get_path`: trajopt first, then elastic.
        Elastic runs second on purpose — trajopt's outward push can create
        sharp corners at the bow apex that the parametrizer renders as hard
        decelerations, and the Laplacian pass rounds them off (see the
        ordering comment in `rrt_path_utils.get_path`).

        Both passes are internally hard-collision-gated and revert any sweep
        that would break the path, so this cannot turn a clean candidate into
        a colliding one; the caller still gates the result, which keeps the
        contract identical to the old in-`get_path` placement.

        Returns the path unchanged when neither pass is enabled, when the
        path is too short to optimize (< 3 waypoints — no interior points),
        or when `postprocess_after_ranking` is off (in which case `get_path`
        already did the work).
        """
        from splatsim.utils.rrt_path_utils import (
            elastic_smooth_path,
            min_distance_to_obstacles,
            trajopt_smooth_path,
        )
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        if not self._postprocess_after_ranking:
            return path
        if path.shape[0] < 3:
            return path
        if not self._trajopt_passes and not self._elastic_smooth_passes:
            return path

        def _collision_fn(q):
            return check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                q,
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **self._collision_kwargs,
            )

        # "guided" soft-cost mode: same q -> cost lookup get_path would have
        # handed these passes, so the cost-gating behaves identically.
        _cost_fn = self._rrt_config_cost_fn()
        out = np.asarray(path, dtype=np.float64)

        if self._trajopt_passes:
            def _distance_fn(q):
                return min_distance_to_obstacles(
                    self._robot_id,
                    self._joint_indices,
                    q,
                    self._loaded_obstacle_ids,
                    # Matches get_path: the hinge cost is zero past the
                    # threshold, and getClosestPoints cost explodes with the
                    # query margin on large concave meshes.
                    max_dist=max(float(self._trajopt_collision_threshold) * 2.0, 0.2),
                )

            out = np.asarray(
                trajopt_smooth_path(
                    out,
                    collision_fn=_collision_fn,
                    distance_fn=_distance_fn,
                    passes=int(self._trajopt_passes),
                    lr=float(self._trajopt_lr),
                    smoothness_weight=float(self._trajopt_smoothness_weight),
                    collision_weight=float(self._trajopt_collision_weight),
                    collision_threshold=float(self._trajopt_collision_threshold),
                    fd_step=float(self._trajopt_fd_step),
                    config_cost_fn=_cost_fn,
                ),
                dtype=np.float64,
            )

        if self._elastic_smooth_passes and out.shape[0] >= 3:
            out = np.asarray(
                elastic_smooth_path(
                    out,
                    _collision_fn,
                    passes=int(self._elastic_smooth_passes),
                    config_cost_fn=_cost_fn,
                ),
                dtype=np.float64,
            )

        # `min_distance_to_obstacles` goes through `set_robot_joint_positions`,
        # which forces the gripper OPEN and steps physics. Re-snap so the
        # collision gate below sees the env's actual gripper geometry — the
        # same restore `_generate_paths_for_ik` does after every get_path.
        self._snap_gripper_to_actual()
        return out

    def _smooth_and_check_collision(
        self,
        rrt_waypoints: np.ndarray,
        start_vel: np.ndarray | None,
    ) -> tuple[np.ndarray, int | None]:
        """Time-parametrize a raw RRT path (TOPP-RA by default; see
        rrt_path_utils.parametrize_path) and dense-check the smoothed
        result for collisions. Returns ``(smoothed_traj, first_colliding_idx)``;
        ``first_colliding_idx is None`` means the smoothed path is
        collision-free.

        Used inside ``plan()``'s per-IK loop to try each candidate path's
        smoothed form BEFORE settling on one. Without this, the BiRRT raw
        check (which only validates discrete waypoints) lets the parametrizer's
        continuous spline curve through obstacles at sharp corners; the
        bad chunk then reaches the env, robot collides, and the controller
        has to retry from inside an already-contaminated teleop buffer.

        Cost dominated by the per-waypoint pybullet getClosestPoints calls.
        Bails on the first collision so a known-bad path is cheap to reject.
        Caller is responsible for joint-state restore (plan()'s finally).
        """
        from splatsim.utils.rrt_path_utils import (
            check_links_in_collision,
            parametrize_path,
        )

        if rrt_waypoints.shape[0] < 2:
            # Single-waypoint path — nothing to parametrize and nothing the
            # raw collision check would have missed.
            return rrt_waypoints, None

        max_vel = np.full(self._num_dofs, self._max_joint_vel)
        max_acc = np.full(self._num_dofs, self._max_joint_acc)
        max_jerk = np.full(self._num_dofs, self._max_joint_jerk)
        _parametrize_kwargs: dict = {}
        if start_vel is not None:
            _parametrize_kwargs["start_vel"] = np.asarray(start_vel, dtype=np.float64).reshape(-1)[
                : self._num_dofs
            ]
        traj = np.asarray(
            parametrize_path(
                rrt_waypoints,
                max_vel,
                max_acc,
                max_jerk,
                control_hz=self._fps,
                segment_at_sharp_corners=self._segment_at_sharp_corners,
                final_approach_dist=self._final_approach_dist,
                final_approach_vel_scale=self._final_approach_vel_scale,
                final_approach_acc_scale=self._final_approach_acc_scale,
                uniform_path_speed=self._uniform_path_speed,
                **_parametrize_kwargs,
            ),
            dtype=np.float64,
        )
        # Parametrizer-scaled clearance: the smoothed C² spline naturally curves
        # outside the linear chord at sharp corners, so a stricter-than-raw
        # obstacle_clearance would reject too many good candidates. Default
        # factor 0.5 (RRT ≥2 cm ⇒ parametrized ≥1 cm) — still catches real
        # penetration.
        _kwargs = self._smoothed_collision_kwargs()
        import os as _os
        _dbg_wp0 = _os.environ.get("SPLATSIM_RRT_DEBUG_WP0")
        for k in range(traj.shape[0]):
            if check_links_in_collision(
                self._robot_id,
                self._joint_indices,
                traj[k],
                self._loaded_obstacle_ids,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=False,
                physics_client_id=self._pb_client,
                link_indices_to_check=self._planner_link_indices_to_check,
                **_kwargs,
            ):
                # Waypoint 0 == q_start (the just-escaped, supposedly-safe frame).
                # Its collision here contradicts the escape's own safety check, so
                # dump WHY (kind, links, gripper-state sensitivity, whether
                # is_q_in_collision agrees) to localize the escape↔parametrization
                # inconsistency. Gated on SPLATSIM_RRT_DEBUG_WP0.
                if _dbg_wp0 and k == 0:
                    self._debug_waypoint0_collision(traj[0], _kwargs)
                return traj, k
        return traj, None

    def _debug_waypoint0_collision(self, q_arm: np.ndarray, collision_kwargs: dict) -> None:
        """Diagnose why the just-escaped q_start (== waypoint 0) collides in the
        parametrization check when the escape's own `is_q_in_collision` cleared it.

        Prints, for this exact arm config:
          1. the parametrization check's kind + colliding link pair (verbose re-check),
          2. the gripper joint value the parametrization check actually used,
          3. gripper SENSITIVITY — does the collision persist with the gripper
             forced open vs closed? (if it toggles, gripper state is the culprit),
          4. whether `is_q_in_collision` (the escape's checker) AGREES.
        Enabled by SPLATSIM_RRT_DEBUG_WP0; best-effort, never raises.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        try:
            rid, cid = self._robot_id, self._pb_client
            n = self._num_dofs
            njoints = p.getNumJoints(rid, physicsClientId=cid)
            grip_idx = list(range(n + 1, njoints))  # gripper joints (arm = [1..n])

            def _recheck(verbose=False):
                _, kind = check_links_in_collision(
                    rid, self._joint_indices, q_arm, self._loaded_obstacle_ids,
                    obstacle_names=self._obstacle_names, skip_pairs=self._skip_pairs,
                    verbose=verbose, physics_client_id=cid,
                    link_indices_to_check=self._planner_link_indices_to_check,
                    return_kind=True, **collision_kwargs,
                )
                return kind

            cur_grip = [round(float(p.getJointState(rid, gi, physicsClientId=cid)[0]), 3) for gi in grip_idx]
            logger.warning("[wp0-debug] waypoint-0 collides. check used gripper joints=%s", cur_grip)
            logger.warning("[wp0-debug]   colliding pair (verbose): kind=%s", _recheck(verbose=True))

            # Gripper sensitivity: force open (0.0) vs closed (0.8), re-check.
            for label, gval in (("open(0.0)", 0.0), ("closed(0.8)", 0.8)):
                for gi in grip_idx:
                    p.resetJointState(rid, gi, gval, physicsClientId=cid)
                logger.warning("[wp0-debug]   gripper=%s -> collision kind=%s", label, _recheck())
            # Restore the gripper the parametrization check had left.
            for gi, gv in zip(grip_idx, cur_grip):
                p.resetJointState(rid, gi, float(gv), physicsClientId=cid)

            # Does the escape's own checker agree on this exact config?
            iqc = self.is_q_in_collision(np.asarray(q_arm, dtype=np.float64), return_kind=True)
            logger.warning(
                "[wp0-debug]   is_q_in_collision (escape's checker) says: %s   "
                "(parametrized obstacle_clr=%.4f)",
                iqc, collision_kwargs.get("obstacle_clearance", -1),
            )
        except Exception as e:  # never let debug break planning
            logger.warning("[wp0-debug] diagnostic failed: %s", e)

    def _score_candidate(
        self,
        path: np.ndarray,
        recent_joint_velocity: np.ndarray | None,
    ) -> float:
        """Dispatch to the active path-selection strategy. Lower = better.

        `recent_joint_velocity` is consulted only when the strategy is
        JOINT_VELOCITY_MATCH; for other strategies it is ignored.

        When a soft-cost field is loaded (vegetation scenes) and
        soft_cost_mode == "score", ``weight * path-integral(cost)`` is ADDED
        to whichever base strategy is active, so among hard-feasible
        candidates the one brushing the least foliage wins. With no field
        loaded this is a None-check — identical to the historical scorer.
        """
        strategy = self._path_selection
        if strategy == PathSelectionStrategy.EE_ARC_LENGTH:
            base = self._path_ee_arc_length(path)
        elif strategy == PathSelectionStrategy.JOINT_ARC_LENGTH:
            base = self._path_joint_arc_length(path)
        elif strategy == PathSelectionStrategy.JOINT_VELOCITY_MATCH:
            assert recent_joint_velocity is not None  # caller guarantees, see plan()
            base = self._path_velocity_deviation(path, recent_joint_velocity)
        elif strategy == PathSelectionStrategy.MIN_PAIR_CLEARANCE:
            base = self._path_min_pair_clearance(path)
        elif strategy == PathSelectionStrategy.CAMERA_SCORING:
            base = self._path_camera_score(path)
        else:
            raise ValueError(f"Unknown PathSelectionStrategy: {strategy!r}")
        if not self._soft_cost_active():
            return base
        soft = self._path_soft_cost(path)
        logger.debug(
            "_score_candidate: base(%s)=%.4f + soft_cost=%.4f (weight=%.3f)",
            strategy, base, self._soft_cost_weight * soft, self._soft_cost_weight,
        )
        return base + self._soft_cost_weight * soft

    def set_soft_cost_field(self, field_or_payload) -> None:
        """Attach a SoftCostField (or an env-config-style payload dict)
        directly. For callers that never go through ``load_obstacles`` —
        e.g. the in-env TrajectoryGenerator, which plans against the env's
        live pybullet world where obstacles already exist as bodies."""
        if field_or_payload is None or hasattr(field_or_payload, "cost_at"):
            self._soft_cost_field = field_or_payload
            return
        from splatsim.utils.soft_cost_field import SoftCostField

        self._soft_cost_field = SoftCostField.from_config(field_or_payload)

    def _soft_cost_active(self) -> bool:
        """Whether the soft-cost term participates in candidate SCORING.
        True for both "score" and "guided" (guided is score + cost-aware
        generation)."""
        return (
            self._soft_cost_field is not None
            and self._soft_cost_mode in ("score", "guided")
            and self._soft_cost_weight > 0.0
        )

    def _config_soft_cost(self, q: np.ndarray) -> float:
        """Scalar soft cost of a single configuration: aggregated field cost
        over the arm surface sample points (same sampling and reduction as
        `_path_soft_cost`, one config). Mutates the planning client's joint
        state — same contract as `_config_soft_cost_points`."""
        pts = self._config_soft_cost_points(q)
        return self._aggregate_soft_cost(self._soft_cost_field.cost_at(pts))

    def _rrt_config_cost_fn(self):
        """The `config_cost_fn` handed to rrt_path_utils.get_path: a
        q -> float soft-cost lookup in "guided" mode, None otherwise (which
        makes get_path run the exact historical binary pipeline)."""
        if self._soft_cost_field is None or self._soft_cost_mode != "guided":
            return None
        return self._config_soft_cost

    def _link_radii(self) -> np.ndarray:
        """Per-link cross-sectional radii, cached. Delegates to
        soft_cost_field.link_radii so the planner and any other soft-cost
        consumer measure the arm the same way."""
        if getattr(self, "_link_radii_cache", None) is None:
            from splatsim.utils.pybullet_client import BulletClientShim
            from splatsim.utils.soft_cost_field import link_radii
            self._link_radii_cache = link_radii(
                BulletClientShim(self._pb_client), self._robot_id,
                self._planner_link_indices_to_check,
                joint_indices=self._joint_indices,
            )
        return self._link_radii_cache

    def _config_soft_cost_points(self, q: np.ndarray) -> np.ndarray:
        """World-space sample points on the arm's SURFACE at joint config
        ``q``. Sampling lives in soft_cost_field.link_surface_points so the
        planner, goal-pose ranking and any diagnostic all measure the same
        geometry; see that function for why surfaces beat the centreline.
        Mutates the planning client's joint state; caller relies on plan()'s
        finally block to restore it."""
        from splatsim.utils.soft_cost_field import link_surface_points

        for idx, qi in zip(self._joint_indices,
                           np.asarray(q, dtype=np.float64).reshape(-1)):
            p.resetJointState(self._robot_id, idx, float(qi),
                              physicsClientId=self._pb_client)
        origins = [
            p.getLinkState(self._robot_id, link_i, computeForwardKinematics=True,
                           physicsClientId=self._pb_client)[4]
            for link_i in self._planner_link_indices_to_check
        ]
        return link_surface_points(
            origins, self._link_radii(),
            spacing=self._soft_cost_sample_spacing,
            n_ring=int(self._soft_cost_surface_samples),
        )

    def _aggregate_soft_cost(self, costs: np.ndarray) -> float:
        """Reduce per-sample costs to one scalar. Delegates to
        soft_cost_field.aggregate_soft_cost (see it for why "max")."""
        from splatsim.utils.soft_cost_field import aggregate_soft_cost

        return aggregate_soft_cost(costs, self._soft_cost_aggregation)

    def _path_soft_cost(self, path: np.ndarray) -> float:
        """Path integral of the soft-cost field along the arm's sweep.

        Densifies the sparse RRT path (same helper the MIN_PAIR_CLEARANCE
        scorer uses), evaluates the field at arm sample points per dense
        waypoint, and integrates mean cost against joint-space arc length so
        the result is invariant to densification resolution. Field grids are
        normalized to max=1, so the value is roughly "radians traveled
        weighted by foliage density" — commensurate with arc-length scores.
        """
        assert self._soft_cost_field is not None
        dense = self._densify_path_for_scoring(np.asarray(path, dtype=np.float64))
        costs = np.empty(dense.shape[0])
        for k in range(dense.shape[0]):
            pts = self._config_soft_cost_points(dense[k])
            costs[k] = self._aggregate_soft_cost(
                self._soft_cost_field.cost_at(pts))
        if dense.shape[0] < 2:
            return float(costs[0])
        seg_len = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        seg_cost = 0.5 * (costs[:-1] + costs[1:])
        return float(np.sum(seg_len * seg_cost))

    def _debug_draw_soft_cost_path(self, traj: np.ndarray) -> None:
        """GUI-only: draw the winning trajectory's EE trace colored by local
        soft cost (green=free, red=max). Gated by soft_cost_debug_draw."""
        if not (self._soft_cost_debug_draw and self._soft_cost_field is not None):
            return
        try:
            idxs = np.linspace(0, traj.shape[0] - 1,
                               min(traj.shape[0], 120), dtype=int)
            ee = []
            for k in idxs:
                for idx, qi in zip(self._joint_indices, traj[k]):
                    p.resetJointState(self._robot_id, idx, float(qi),
                                      physicsClientId=self._pb_client)
                ee.append(p.getLinkState(
                    self._robot_id, self._ee_link_index,
                    computeForwardKinematics=True,
                    physicsClientId=self._pb_client,
                )[0])
            ee_arr = np.asarray(ee)
            c = self._soft_cost_field.cost_at(ee_arr)
            cmax = max(float(c.max()), 1e-9)
            for a, b, ca in zip(ee_arr[:-1], ee_arr[1:], c[:-1]):
                frac = float(ca) / cmax
                p.addUserDebugLine(
                    a.tolist(), b.tolist(),
                    lineColorRGB=[frac, 1.0 - frac, 0.0],
                    lineWidth=3, lifeTime=30,
                    physicsClientId=self._pb_client,
                )
        except Exception:
            logger.exception("soft-cost debug draw failed (non-fatal)")

    def _get_camera_link_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK the wrist-camera link at joint config ``q``. Returns
        ``(position(3,), rotation_matrix(3,3))`` in world frame. Mutates the
        planning client's joint state; caller relies on plan()'s finally block
        to restore it (same contract as ``_path_min_pair_clearance``)."""
        for idx, qi in zip(self._joint_indices, np.asarray(q, dtype=np.float64).reshape(-1)):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)
        link_state = p.getLinkState(
            self._robot_id,
            self._wrist_camera_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._pb_client,
        )
        position = np.asarray(link_state[0], dtype=np.float64)
        rot = np.asarray(p.getMatrixFromQuaternion(link_state[1]), dtype=np.float64).reshape(3, 3)
        return position, rot

    def _path_camera_score(self, path: np.ndarray) -> float:
        """PathSelectionStrategy.CAMERA_SCORING scorer. Rewards paths that keep
        the wrist camera aimed at the goal EE position throughout the motion.
        Ported from SplatSim's TrajectoryGenerator._compute_camera_score; reuses
        the shared ``rrt_path_utils.compute_camera_alignment_score``. Returns the
        NEGATED mean alignment (higher align = better → negate so lower = better,
        matching the dispatcher convention, same trick as MIN_PAIR_CLEARANCE)."""
        from splatsim.utils.rrt_path_utils import compute_camera_alignment_score

        if self._wrist_camera_link_index is None or self._score_goal_q is None:
            return 0.0  # camera scoring unavailable → neutral (no-op)
        # Aim target = EE-link world position at the goal config.
        for idx, qi in zip(self._joint_indices, np.asarray(self._score_goal_q, dtype=np.float64).reshape(-1)):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)
        target_position = np.asarray(
            p.getLinkState(
                self._robot_id,
                self._ee_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._pb_client,
            )[0],
            dtype=np.float64,
        )
        num_samples = min(len(path), 10)
        idxs = np.linspace(0, len(path) - 1, num_samples, dtype=int)
        scores = []
        for k in idxs:
            cam_pos, cam_rot = self._get_camera_link_pose(path[k])
            cam_forward = cam_rot[:, 2]  # camera +Z local axis
            scores.append(
                compute_camera_alignment_score(
                    cam_pos,
                    cam_forward,
                    target_position,
                    self._camera_k_exp,
                    self._camera_k_sig,
                    self._camera_threshold,
                )
            )
        return -float(np.mean(scores)) if scores else 0.0

    def _score_ik_candidate(
        self,
        q_candidate: np.ndarray,
        q_start: np.ndarray,
    ) -> float:
        """Score a candidate IK goal under the active IkGoalSelectionStrategy.
        Lower = better. Pure goal-state geometry — no planned path needed.
        """
        strategy = self._ik_goal_selection
        if strategy == IkGoalSelectionStrategy.JOINT_DISTANCE:
            return float(np.linalg.norm(np.asarray(q_candidate) - np.asarray(q_start)))
        raise ValueError(f"Unknown IkGoalSelectionStrategy: {strategy!r}")

    def _generate_paths_for_ik(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate up to `num_path_candidates_per_ik` distinct RRT paths
        from q_start to q_goal. First attempt uses exact endpoints;
        subsequent attempts perturb both endpoints by ±perturbation_scale
        to nudge RRT's random sampler down different branches. Each
        successful path's terminal point is snapped back to the exact
        q_goal so the final pose is consistent across candidates.

        Ports the multi-candidate pattern from SplatSim's
        `TrajectoryGenerator._generate_multiple_path_candidates`: the
        per-IK loop runs at most `max_path_attempts_per_ik` consecutive
        attempts between successes, so num_attempts is a soft cap, not
        a hard total. Returns the list of successful paths (may be
        empty if every attempt failed).

        Skipped (and reduces to a single get_path call) when
        num_path_candidates_per_ik == 1 — preserves the original single-
        attempt behavior with zero overhead.
        """
        # Lazy import — keeps the module-level import surface free of
        # splatsim (an optional dep) and matches the pattern used in
        # plan(). Resolves to the same `get_path` used there.
        from splatsim.utils.rrt_path_utils import get_path

        # With `postprocess_after_ranking` on (the default), trajopt and
        # elastic are NOT run per candidate here — `plan()` applies them via
        # `_postprocess_path` to the candidate it actually selects. Zeroing
        # them means get_path stops after RRT + shortcut smoothing, which is
        # what the scorer then ranks.
        _gp_trajopt = 0 if self._postprocess_after_ranking else self._trajopt_passes
        _gp_elastic = (
            0 if self._postprocess_after_ranking else self._elastic_smooth_passes
        )

        num_target = self._num_path_candidates_per_ik
        # "guided" soft-cost mode: hand get_path a q -> cost lookup so RRT
        # growth + smoothing avoid the field, not just the final scorer.
        # None in every other mode / with no field — historical pipeline.
        _cost_fn = self._rrt_config_cost_fn()
        if num_target <= 1:
            attempt = get_path(
                q_start,
                q_goal,
                self._robot_id,
                self._joint_indices,
                self._loaded_obstacle_ids,
                self._lower_limits,
                self._upper_limits,
                self._fps,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=True,
                physics_client_id=self._pb_client,
                max_smooth_iterations=self._rrt_smooth_iterations,
                elastic_smooth_passes=_gp_elastic,
                trajopt_passes=_gp_trajopt,
                trajopt_lr=self._trajopt_lr,
                trajopt_smoothness_weight=self._trajopt_smoothness_weight,
                trajopt_collision_weight=self._trajopt_collision_weight,
                trajopt_collision_threshold=self._trajopt_collision_threshold,
                trajopt_fd_step=self._trajopt_fd_step,
                actual_gripper_q=self._current_actual_gripper_q,
                config_cost_fn=_cost_fn,
                trrt_params=self._soft_cost_guided_params,
                **self._collision_kwargs,
            )
            # get_path → set_robot_joint_positions → open_gripper resets the
            # gripper to 0.0. Re-snap so downstream time-parametrized / dense
            # collision checks see the actual gripper geometry.
            self._snap_gripper_to_actual()
            return [np.asarray(attempt, dtype=np.float64)] if attempt is not None else []

        paths: list[np.ndarray] = []
        attempts = 0
        max_attempts = self._max_path_attempts_per_ik
        scale = self._path_perturbation_scale
        # Early abort on a dead IK goal: attempts against one q_goal differ
        # only by +-scale endpoint perturbation, so RRT failures are near-
        # perfectly correlated — an IK branch whose approach corridor is
        # sealed at the configured clearance fails ALL max_attempts (vine
        # bench: 15 consecutive ~2s failures before plan() moved to the next
        # IK candidate). If the first few attempts produce zero paths, give
        # up on this goal and let the caller try the next IK candidate.
        early_abort_after = 10**6
        while len(paths) < num_target and attempts < max_attempts:
            if not paths and attempts >= early_abort_after:
                logger.info(
                    "IK goal looks unplannable (%d/%d attempts, 0 paths) — "
                    "skipping to the next IK candidate.",
                    attempts, max_attempts,
                )
                break
            attempts += 1
            if len(paths) == 0:
                plan_start = q_start
                plan_goal = q_goal
            else:
                plan_start = np.clip(
                    q_start + np.random.uniform(-scale, scale, size=q_start.shape),
                    self._lower_limits,
                    self._upper_limits,
                )
                plan_goal = np.clip(
                    q_goal + np.random.uniform(-scale, scale, size=q_goal.shape),
                    self._lower_limits,
                    self._upper_limits,
                )
            attempt = get_path(
                plan_start,
                plan_goal,
                self._robot_id,
                self._joint_indices,
                self._loaded_obstacle_ids,
                self._lower_limits,
                self._upper_limits,
                self._fps,
                obstacle_names=self._obstacle_names,
                skip_pairs=self._skip_pairs,
                verbose=True,
                physics_client_id=self._pb_client,
                max_smooth_iterations=self._rrt_smooth_iterations,
                elastic_smooth_passes=_gp_elastic,
                trajopt_passes=_gp_trajopt,
                trajopt_lr=self._trajopt_lr,
                trajopt_smoothness_weight=self._trajopt_smoothness_weight,
                trajopt_collision_weight=self._trajopt_collision_weight,
                trajopt_collision_threshold=self._trajopt_collision_threshold,
                trajopt_fd_step=self._trajopt_fd_step,
                actual_gripper_q=self._current_actual_gripper_q,
                config_cost_fn=_cost_fn,
                trrt_params=self._soft_cost_guided_params,
                **self._collision_kwargs,
            )
            # get_path resets the gripper open — re-snap for the next
            # iteration's checks and for the caller's parametrize/dense check.
            self._snap_gripper_to_actual()
            if attempt is None:
                continue
            arr = np.asarray(attempt, dtype=np.float64)
            # Snap initial pose back to the exact q_start. The perturbed
            # plan_start exists ONLY to diversify RRT's exploration; it must
            # not leak into the executed trajectory. arr[0] is what the parametrizer
            # parametrizes from and what the robot is commanded to on the
            # first frame, but the robot is physically at the true q_start.
            # A perturbed arr[0] (up to ±path_perturbation_scale per joint)
            # teleports the command on frame 1 → a single-frame joint spike
            # at the RRT onset (and at every mid-trajectory replan), worst on
            # otherwise-static wrist joints. Prepend the true q_start so the parametrizer
            # accelerates from the robot's actual config over several frames
            # instead of jumping. Mirrors the terminal q_goal snap below; the
            # added lead-in segment is still dense-collision-checked in
            # _smooth_and_check_collision, so it can only reject a path, never
            # smuggle a colliding one through.
            if not np.allclose(arr[0], q_start):
                arr = np.vstack([q_start, arr])
            # Snap terminal pose back to the exact q_goal so every
            # candidate's endpoint is identical (perturbation only
            # affects the middle of the path).
            if not np.allclose(arr[-1], q_goal):
                arr = np.vstack([arr, q_goal])
            paths.append(arr)
            attempts = 0  # reset between successes — matches SplatSim semantics
        return paths

    def _path_joint_arc_length(self, path: np.ndarray) -> float:
        """Sum of joint-space L2 distances between consecutive waypoints.

        Cheap: no pybullet calls. Tends to favor candidates that land close
        to `q_start` in configuration space even if the EE swings wide;
        use EE_ARC_LENGTH if you care more about cartesian path tidiness.
        """
        if path.shape[0] < 2:
            return 0.0
        deltas = np.diff(path, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _path_velocity_deviation(
        self,
        path: np.ndarray,
        recent_joint_velocity: np.ndarray,
    ) -> float:
        """Cosine distance between candidate's initial DIRECTION and the
        robot's recent DIRECTION (lower = better aligned).

        An earlier version computed an L2 distance between magnitudes
        directly, but that conflates two different units: the candidate's
        ``leading_deltas`` are spatial deltas between raw RRT waypoints
        (which aren't uniformly time-spaced), while ``recent_joint_velocity``
        is a real per-step velocity (rad / control-tick). Comparing their
        magnitudes ranked paths in a way that didn't survive the parametrizer's
        time-parametrization, producing sustained high-velocity stretches
        in some recorded trajectories (e.g. one joint at 5 rad/s for many
        consecutive frames). Direction-only comparison sidesteps the
        unit mismatch — we ask "does the candidate START in the same
        direction the robot was already moving" without trying to match
        magnitudes that aren't comparable.

        Fallback: when the robot's recent velocity is near zero (typical
        when RRT triggers from a collision/stall — no direction to match),
        we delegate to EE arc-length so the candidate selection isn't
        random. Threshold is sized to per-step deltas at 30 Hz, where
        ~5e-4 rad/step ≈ 1.5e-2 rad/s of total joint motion across the
        ``velocity_match_window`` — below that, the direction is dominated
        by sensor noise.
        """
        if path.shape[0] < 2:
            # Degenerate path — no motion to compare against. Return a large
            # penalty so any candidate with real motion wins over this one.
            return float("inf")
        recent_vel = recent_joint_velocity.reshape(-1)[: self._num_dofs]
        recent_norm = float(np.linalg.norm(recent_vel))
        if recent_norm < 5e-4:
            # No meaningful direction — fall back to EE arc length to avoid
            # picking among candidates at random.
            return self._path_ee_arc_length(path)
        window = max(1, min(self._velocity_match_window, path.shape[0] - 1))
        leading_deltas = np.diff(path[: window + 1], axis=0)  # [window, num_dofs]
        candidate_vel = leading_deltas.mean(axis=0)
        cand_norm = float(np.linalg.norm(candidate_vel))
        if cand_norm < 1e-9:
            # Candidate path starts with all-zero deltas (degenerate RRT
            # sampling). Maximally misaligned with any nonzero recent_vel.
            return 1.0
        cos_sim = float(np.dot(candidate_vel, recent_vel) / (cand_norm * recent_norm))
        # Clip to [-1, 1] to guard against rounding errors, then 1 - cos:
        # range becomes [0, 2], with 0 = perfectly aligned, 2 = opposite.
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 1.0 - cos_sim

    def _path_ee_arc_length(self, path: np.ndarray) -> float:
        """Sum of Euclidean distances between consecutive end-effector world
        positions along a joint-space ``path`` of shape ``[N, num_dofs]``.

        Uses fast FK: resetJointState (no physics step) on the arm joints,
        then ``getLinkState(..., computeForwardKinematics=True)``. This leaves
        the robot at the path's terminal config — callers in ``plan()`` are
        responsible for restoring joint state from the snapshot taken on entry.
        """
        if path.shape[0] < 2:
            return 0.0
        ee_positions = np.empty((path.shape[0], 3), dtype=np.float64)
        for k in range(path.shape[0]):
            for j_idx, qi in zip(self._joint_indices, path[k].tolist(), strict=True):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
            link_state = p.getLinkState(
                self._robot_id,
                self._ee_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._pb_client,
            )
            ee_positions[k] = link_state[0]  # worldLinkFramePosition
        return float(np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)))

    @staticmethod
    def _densify_path_for_scoring(path: np.ndarray, max_step_rad: float = 0.05) -> np.ndarray:
        """Linearly interpolate between consecutive waypoints so adjacent
        dense waypoints differ by no more than ``max_step_rad`` (per-joint
        L-infinity). Used by ``_path_min_pair_clearance`` to score the
        trajectory the robot will actually execute, not just RRT's sparse
        sample points — RRT can return paths with 4-6 waypoints whose
        interpolated configs in between pass through tighter self-collision
        configs than the waypoints themselves.

        Densification is PURELY a scoring-side concern: callers of
        ``_path_min_pair_clearance`` pass the sparse RRT path, this helper
        densifies internally, the SAME sparse path is what gets returned
        from ``plan()`` and handed to the parametrizer. So the actual execution
        trajectory is unchanged; only the score reflects the dense view.

        Args:
            path: ``[N, num_dofs]`` sparse joint-space path from RRT.
            max_step_rad: max allowed per-joint L-infinity step between
                consecutive dense waypoints. 0.05 rad (~3°) matches the
                RRT-Connect collision-check resolution, so links move on
                the order of 10 mm between dense waypoints — enough to
                catch self-collisions in the 5-50 mm gap range we care
                about for scoring.

        Returns:
            ``[M, num_dofs]`` densified path with M >= N. The first and
            last waypoints are preserved exactly; intermediate sparse
            waypoints are preserved as the boundary points of segments.
        """
        if path.shape[0] < 2:
            return path  # nothing to interpolate between
        dense_chunks: list[np.ndarray] = []
        for k in range(path.shape[0] - 1):
            q0 = path[k]
            q1 = path[k + 1]
            # Number of intermediate steps so per-step L-infinity delta is
            # at most max_step_rad. n_steps=1 means just [q0, q1] (no extra).
            max_delta = float(np.max(np.abs(q1 - q0)))
            n_steps = max(1, int(np.ceil(max_delta / max_step_rad)))
            # Sample n_steps+1 points from q0 to q1 INCLUSIVE; drop the
            # last so the next segment's first point isn't duplicated.
            ts = np.linspace(0.0, 1.0, n_steps + 1)
            seg = q0[None, :] + ts[:, None] * (q1 - q0)[None, :]
            dense_chunks.append(seg[:-1])
        # Append the final waypoint that we trimmed off the last segment.
        dense_chunks.append(path[-1:].reshape(1, -1))
        return np.concatenate(dense_chunks, axis=0)

    def _path_min_pair_clearance(self, path: np.ndarray) -> float:
        """Negated minimum non-adjacent-link-pair distance over a joint-space
        path. Lower returned value = larger clearance = SAFER path.

        For each waypoint along the path, the robot is snapped to that joint
        config and every non-adjacent link pair (excluding the
        URDF-structurally-close pairs declared via
        ``self_collision_skip_pairs``) is queried for its actual minimum
        distance. The smallest such distance across the path is what's
        returned (negated, since the planner's scoring convention is
        lower-is-better and we want LARGER clearance to win).

        IMPORTANT: the input ``path`` is RRT's SPARSE waypoint sequence,
        but scoring is done on a DENSIFIED copy (linear interpolation,
        per-joint L-infinity step ≤ 0.05 rad) so configurations between
        the sparse waypoints aren't missed. The sparse path is what
        the parametrizer sees — densification is internal to scoring only. See
        ``_densify_path_for_scoring`` for the rationale.

        Query distance cap is set to ``_PAIR_CLEARANCE_QUERY_CAP`` (10 cm
        by default) so far-apart pairs early-out cheaply in pybullet's
        ``getClosestPoints``. A waypoint where every pair is > cap apart
        contributes +cap to the running minimum (saturated), which keeps
        the score comparable across paths with mostly-roomy waypoints.

        Side effect: leaves the robot at the dense path's terminal config
        (= the sparse path's terminal config). The ``plan()`` caller takes
        a joint-state snapshot on entry and restores from it after scoring.
        """
        if path.shape[0] < 1:
            return 0.0  # degenerate; no waypoints to evaluate
        # Densify so RRT's sparse waypoints don't hide tighter
        # configurations between them. Hardcoded step matches the BiRRT
        # collision-check resolution so the scorer never "sees" finer
        # granularity than the planner could have rejected at planning
        # time — keeps the scoring grounded in what was actually feasible.
        scoring_path = self._densify_path_for_scoring(path, max_step_rad=0.05)
        # 10 cm cap — beyond this the pair isn't influencing the score.
        # Larger cap = more compute per query; smaller = more pairs saturate
        # at the cap, making the score less discriminating.
        _PAIR_CLEARANCE_QUERY_CAP = 0.10  # noqa: N806 — read as a constant, kept uppercase for clarity

        # Lazy import — matches the pattern used elsewhere in this file
        # (e.g. plan_segment) to keep splatsim out of the module-level
        # import surface.
        from splatsim.utils.rrt_path_utils import are_adjacent_links

        # Build the structural-skip set once. Stored on the planner via
        # self._collision_kwargs["self_collision_skip_pairs"] (a list of
        # (a, b) tuples) when set by the SA config / env oracle dispatch.
        _skip_raw = self._collision_kwargs.get("self_collision_skip_pairs") or []
        _skip_set = {frozenset((int(a), int(b))) for a, b in _skip_raw}

        # All robot links (-1 = body base + every joint's child link). Match
        # check_links_in_collision's default to keep the scoring consistent
        # with the feasibility check.
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)
        all_links = list(range(-1, n_joints))

        # Precompute non-adjacent + non-skip pair list once (it's a property
        # of the URDF, not the path). Saves the adjacency check on every
        # waypoint.
        pair_list: list[tuple[int, int]] = []
        for a, b in itertools.combinations(all_links, 2):
            if frozenset((a, b)) in _skip_set:
                continue
            if are_adjacent_links(self._robot_id, a, b, physics_client_id=self._pb_client):
                continue
            pair_list.append((a, b))

        # Track the minimum across the DENSE path (interior + waypoints),
        # AND remember which pair set it. The full per-pair min/max table
        # is dumped on EVERY call (per-scenario, per-RRT-trigger) so the
        # user can spot structural offenders across all paths the robot
        # actually plans through — not just the first one (which may not
        # be representative of all path topologies).
        min_dist_over_path = _PAIR_CLEARANCE_QUERY_CAP
        min_pair: tuple[int, int] | None = None
        # Diagnostic mode determines (a) whether to log the per-pair table
        # at all, (b) whether to use the larger 5m getClosestPoints cap
        # for tracking, and (c) whether to log once per planner instance
        # or every call. See SharedAutonomyConfig.rrt_diagnostic_log_pairs.
        diag_mode = self._diagnostic_log_pairs
        diag_first_pending = diag_mode == "first" and not getattr(self, "_min_pair_diag_logged", False)
        should_log_table = diag_mode == "always" or diag_first_pending
        # Cap rules: when logging the full table we use 5m so far-apart
        # pairs are also captured; when off (or already-first-logged) we
        # use the lean 10cm cap so scoring loop stays fast (~200ms vs
        # ~500-1000ms). Score itself is invariant — see clamp below.
        scoring_cap = 5.0 if should_log_table else _PAIR_CLEARANCE_QUERY_CAP
        # Per-pair MIN/MAX only tracked when logging the table — small
        # dict overhead, but pointless if we're not going to print it.
        per_pair_min: dict[tuple[int, int], float] = {} if should_log_table else {}
        per_pair_max: dict[tuple[int, int], float] = {} if should_log_table else {}
        for k in range(scoring_path.shape[0]):
            # Snap to this waypoint's joint config (FK only, no physics).
            for j_idx, qi in zip(self._joint_indices, scoring_path[k].tolist(), strict=True):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
            for a, b in pair_list:
                pts = p.getClosestPoints(
                    self._robot_id,
                    self._robot_id,
                    distance=scoring_cap,
                    linkIndexA=a,
                    linkIndexB=b,
                    physicsClientId=self._pb_client,
                )
                if not pts:
                    continue  # > cap → saturated, doesn't update the min
                # pts[i][8] is the contact-distance field (negative = penetration).
                d = float(pts[0][8])
                # min_dist_over_path tracking clamps to the original
                # _PAIR_CLEARANCE_QUERY_CAP so the score the planner sees
                # is identical regardless of whether we used the lean
                # 10cm cap or the diagnostic's larger 5m cap.
                if d < _PAIR_CLEARANCE_QUERY_CAP and d < min_dist_over_path:
                    min_dist_over_path = d
                    min_pair = (a, b)
                if should_log_table:
                    prev_min = per_pair_min.get((a, b), scoring_cap)
                    if d < prev_min:
                        per_pair_min[(a, b)] = d
                    prev_max = per_pair_max.get((a, b), float("-inf"))
                    if d > prev_max:
                        per_pair_max[(a, b)] = d

        link_name = (  # noqa: E731 — compact local helper used only inside this loop
            lambda i: "WORLD"
            if i == -1
            else p.getJointInfo(self._robot_id, i, physicsClientId=self._pb_client)[12].decode()
        )
        # Dump EVERY non-adjacent pair that produced a close-points hit
        # over this path, sorted ascending by min distance. Includes a
        # range column (max-min) so structural pairs (range ~ 0 mm = rigid
        # sub-chain) are visually distinct from articulating pairs (range
        # > tens of mm = real motion). Gated by diag_mode — see comments
        # at the top of this block.
        # _STRUCTURAL_RANGE_MM_HINT: range below this is auto-flagged in
        # the log as "STRUCTURAL?" — a hint, not a hard threshold. The
        # user makes the final call on whether to skip.
        if should_log_table:
            if diag_first_pending:
                # First-mode bookkeeping — mark so subsequent calls
                # skip the table dump and use the lean cap.
                self._min_pair_diag_logged = True
            _STRUCTURAL_RANGE_MM_HINT = 5.0  # noqa: N806 — read as a constant, kept uppercase for clarity
            ranked = sorted(per_pair_min.items(), key=lambda kv: kv[1])
            logger.info(
                "MIN_PAIR_CLEARANCE structural-offender probe — "
                "ALL %d non-adjacent pairs that had a hit over this scored "
                "path (%d waypoints). Columns: min / max / range; pairs "
                "with range < %.1f mm are likely structural (joint motion "
                "doesn't move them apart) and good skip-list candidates:",
                len(ranked),
                scoring_path.shape[0],
                _STRUCTURAL_RANGE_MM_HINT,
            )
            for (a, b), dmin in ranked:
                dmax = per_pair_max[(a, b)]
                drange = dmax - dmin
                flag = "STRUCTURAL?" if drange * 1000.0 < _STRUCTURAL_RANGE_MM_HINT else "          "
                logger.info(
                    "  %s pair (%d,%d) %s vs %s : min %.3f / max %.3f / range %.3f mm",
                    flag,
                    a,
                    b,
                    link_name(a),
                    link_name(b),
                    dmin * 1000.0,
                    dmax * 1000.0,
                    drange * 1000.0,
                )
            n_structural = sum(
                1
                for (a, b), dmin in ranked
                if (per_pair_max[(a, b)] - dmin) * 1000.0 < _STRUCTURAL_RANGE_MM_HINT
            )
            logger.info(
                "MIN_PAIR_CLEARANCE structural-offender probe summary: "
                "%d/%d pairs flagged STRUCTURAL? (range < %.1f mm on this path).",
                n_structural,
                len(ranked),
                _STRUCTURAL_RANGE_MM_HINT,
            )
        # Per-call floor-pair line: always-on when diag is "always" or
        # when diag is "first" (since the per-pair table is also gated
        # by the same conditions, this stays consistent — first-mode
        # gets the table + floor line on first call, then silence).
        # In "off" mode there's no diagnostic output at all.
        if should_log_table and min_pair is not None:
            logger.info(
                "  → score floor set by pair (%d,%d) %s vs %s at %.3f mm",
                min_pair[0],
                min_pair[1],
                link_name(min_pair[0]),
                link_name(min_pair[1]),
                min_dist_over_path * 1000.0,
            )

        # Negate so lower returned value = larger clearance = better path.
        return -min_dist_over_path

    def _ik_null_space_kwargs(self, seed_q: np.ndarray) -> dict:
        """Build the null-space kwargs for calculateInverseKinematics, padded
        to ``self._num_movable_joints``. PyBullet silently disables damping
        (and the other null-space hints) when the array sizes don't match the
        URDF's total movable-joint count, so we extend the planner's per-arm
        arrays with permissive defaults for the trailing gripper joints.
        """
        n_extra = max(0, self._num_movable_joints - self._num_dofs)
        ll = self._lower_limits.tolist() + [-np.pi] * n_extra
        ul = self._upper_limits.tolist() + [np.pi] * n_extra
        jr = [u - lo for lo, u in zip(ll, ul, strict=True)]
        rp = list(seed_q) + [0.0] * n_extra
        return {
            "lowerLimits": ll,
            "upperLimits": ul,
            "jointRanges": jr,
            "restPoses": rp,
            "jointDamping": [0.1] * self._num_movable_joints,
        }

    # ------------------------------------------------------------------ #
    #  Collision escape (used when the policy got the robot stuck)       #
    # ------------------------------------------------------------------ #

    def _escape_collision(
        self,
        q_start: np.ndarray,
        max_iters: int = 60,
        step_size: float = 0.01,
        max_per_iter_joint_jump: float = 0.2,
        stall_iters: int = 6,
        lift_fallback_step: float = 0.015,
    ) -> np.ndarray | None:
        """Move the arm out of collision along the aggregated outward contact normal.

        Each iteration: query getClosestPoints between the robot and every loaded
        oracle obstacle, weight each (link, obstacle) pair by how deep it is past
        the planner's collision clearance, sum the contact normals (which point
        from obstacle → robot), and apply a small EE-position step in that
        direction via IK. Stops when no pair is within the clearance buffer (or
        when ``max_iters`` is reached).

        Returns a joint-space trajectory ``[N, num_dofs]`` whose first row is
        ``q_start`` and last row is a collision-free config (with the standard
        clearance), or ``None`` if escape did not converge. Caller is responsible
        for restoring the robot's joint state afterwards (the surrounding
        ``plan()`` ``finally`` block handles this).

        Self-collisions and within-clearance pairs whose ``(link, obstacle)`` is
        in ``skip_pairs`` are ignored, matching the planner's collision check.
        """
        # Use the configured obstacle clearance (or SplatSim's default if
        # not overridden) — BUT INFLATED by `escape_clearance_factor`
        # (default 1.5×) for the escape's stop condition. Reasoning:
        #
        #   Without the inflation, escape stops at exactly the BiRRT
        #   collision threshold (e.g., 0.02 m). The planner then starts
        #   RRT from this barely-safe config and the robot begins executing
        #   the resulting trajectory. The very first trajectory waypoint
        #   moves slightly toward the goal — which for approach/grasp
        #   tasks is itself near the obstacle — and the robot dips
        #   BELOW the threshold within 1-2 chunk steps. The controller's
        #   per-tick `is_in_collision_at` check (which uses the same
        #   in-progress clearance, typically matched to the planner's)
        #   then fires `obstacle_collision`, retry triggers, escape
        #   re-runs to the same barely-safe config, repeat. Cascade.
        #
        #   With the inflation, escape pushes the robot to ≥1.5× the
        #   threshold (e.g., 0.03 m for a 0.02 m planner clearance). The
        #   1cm buffer gives the trajectory room to move toward the goal for a
        #   handful of ticks before tripping the controller's check.
        #
        # Weight formula at line ~1781 keeps the original `_eff_clearance`
        # so per-pair contributions stay calibrated for normal escape;
        # only the close-points query distance and the n_pairs-based stop
        # condition use the inflated threshold.
        _eff_clearance_raw = self._effective_obstacle_clearance()
        _eff_clearance = _eff_clearance_raw * self._escape_clearance_factor

        q = np.asarray(q_start, dtype=np.float64).copy()
        waypoints: list[np.ndarray] = [q.copy()]
        prev_max_pen: float | None = None
        no_progress_iters = 0
        lift_mode = False  # switch to a deterministic +z lift after stall

        for it in range(max_iters):
            # Snap pybullet to the current candidate config.
            for j_idx, qi in zip(self._joint_indices, q, strict=False):
                p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)

            escape_dir = np.zeros(3)
            max_pen = 0.0
            n_pairs = 0
            for obs_id in self._loaded_obstacle_ids:
                close_points = (
                    p.getClosestPoints(
                        self._robot_id,
                        obs_id,
                        distance=_eff_clearance,
                        physicsClientId=self._pb_client,
                    )
                    or []
                )
                for c in close_points:
                    link_idx = c[3]  # linkIndexA
                    if (link_idx, obs_id) in self._skip_pairs:
                        continue
                    normal = np.asarray(c[7], dtype=np.float64)  # B → A
                    dist = float(c[8])
                    weight = max(_eff_clearance - dist, 1e-3)
                    escape_dir = escape_dir + normal * weight
                    max_pen = max(max_pen, max(0.0, -dist))
                    n_pairs += 1

            if n_pairs == 0:
                # No obstacle close-pairs left, but the planner's standard
                # collision check (which ALSO covers self-collisions between
                # non-adjacent robot links) may still report colliding — e.g.
                # when the policy curled the arm into itself with the gripper
                # near the shoulder. Contact-normal escape can't help with
                # self-tangles (no obstacle to repel from), but a +z lift
                # straightens the arm and breaks the tangle. Switch to lift
                # mode and continue iterating.
                #
                # Uses the shared planner-contract check so this inner-loop
                # "am I still in collision?" decision matches what RRT (and
                # `is_q_in_collision`) would say — otherwise escape can
                # declare success on a config the planner still rejects
                # (link-scope divergence between raw check_links_in_collision
                # and is_q_in_collision was producing the intervention-
                # controller infinite loop in DAgger interventions).
                still_colliding = self._current_pose_in_planner_collision()
                if not still_colliding:
                    break  # truly clear
                if not lift_mode:
                    logger.info(
                        "No obstacle pairs but check_links_in_collision still "
                        "reports collision (likely self-collision from curled arm) "
                        "— switching to +z lift fallback.",
                    )
                    lift_mode = True

            # Track penetration progress; if max_pen hasn't decreased meaningfully
            # over `stall_iters` iterations we're stuck (e.g. contact normals
            # cancel each other out, or IK can't realise the requested EE step).
            # Switch to a deterministic +z lift in world frame as a fallback.
            if prev_max_pen is not None and max_pen >= prev_max_pen - 1e-4:
                no_progress_iters += 1
            else:
                no_progress_iters = 0
            prev_max_pen = max_pen
            if no_progress_iters >= stall_iters and not lift_mode:
                logger.info(
                    "Escape stalled after %d iter(s) at penetration=%.4fm; "
                    "switching to deterministic +z lift fallback.",
                    it + 1,
                    max_pen,
                )
                lift_mode = True

            if lift_mode:
                escape_dir = np.array([0.0, 0.0, 1.0])
                step = lift_fallback_step
            else:
                norm = float(np.linalg.norm(escape_dir))
                # Wedged between opposing surfaces (~zero net normal) — fall
                # back to a straight up-lift.
                escape_dir = np.array([0.0, 0.0, 1.0]) if norm < 1e-9 else escape_dir / norm
                # Step further when actually penetrating; up to ~6× the base step.
                step = step_size * (1.0 + min(max_pen / _eff_clearance, 5.0))

            # FK at the current config gives us the EE pose to displace.
            ee_state = p.getLinkState(
                self._robot_id,
                self._ee_link_index,
                computeForwardKinematics=True,
                physicsClientId=self._pb_client,
            )
            ee_pos = np.asarray(ee_state[4], dtype=np.float64)
            ee_quat = np.asarray(ee_state[5], dtype=np.float64)
            target_pos = ee_pos + escape_dir * step

            joint_poses = p.calculateInverseKinematics(
                self._robot_id,
                self._ee_link_index,
                target_pos.tolist(),
                ee_quat.tolist(),
                **self._ik_null_space_kwargs(q),
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=self._pb_client,
            )
            q_new = np.asarray(joint_poses[: self._num_dofs], dtype=np.float64)
            q_new = ((q_new + np.pi) % (2 * np.pi)) - np.pi

            # Clamp to limits instead of bailing — escape is best-effort
            # recovery, not precise IK. If clamping leaves q_new == q (no
            # forward progress), the stall detector above will trip after
            # `stall_iters` iterations and switch to the deterministic +z
            # lift fallback. Bailing on the very first IK overshoot makes the
            # escape return None on iter 0 in the common case where IK wants
            # to move joint 1 through its 0-rad upper limit.
            q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
            # If IK overshoots the per-iter jump cap, scale the step down so we
            # still make forward progress (rather than skipping the iteration).
            max_jump = float(np.max(np.abs(q_new - q)))
            if max_jump > max_per_iter_joint_jump:
                scale = max_per_iter_joint_jump / max_jump
                q_new = q + (q_new - q) * scale

            waypoints.append(q_new.copy())
            q = q_new
            if (it + 1) % 10 == 0:
                logger.info(
                    "Escape iter %d/%d: max_pen=%.4fm, n_pairs=%d, mode=%s",
                    it + 1,
                    max_iters,
                    max_pen,
                    n_pairs,
                    "lift" if lift_mode else "contact-normal",
                )

        # Final verification that we cleared the standard collision check.
        for j_idx, qi in zip(self._joint_indices, q, strict=False):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)
        # Final verification uses the planner-contract check so a "success"
        # here means EXACTLY "RRT would accept this pose."
        if self._current_pose_in_planner_collision():
            logger.warning(
                "Escape failed after %d iter(s) — final config still in collision "
                "(max_pen this loop=%.4fm; lift_mode_used=%s). Returning None.",
                len(waypoints) - 1,
                prev_max_pen if prev_max_pen is not None else 0.0,
                lift_mode,
            )
            return None

        logger.info(
            "Escape succeeded after %d waypoint(s) (lift_mode_used=%s).",
            len(waypoints) - 1,
            lift_mode,
        )
        return np.asarray(waypoints, dtype=np.float64)

    def _set_joints_to(self, q: np.ndarray) -> None:
        """Snap pybullet joints to a config (no physics step). Used to restore
        the planning client's state between escape attempts in
        ``_try_escape_chain`` and at the start of gradient-escape iters.
        """
        for j_idx, qi in zip(self._joint_indices, q, strict=False):
            p.resetJointState(self._robot_id, j_idx, float(qi), physicsClientId=self._pb_client)

    def _find_worst_self_collision_pair(
        self,
        query_cap: float = 0.10,
    ) -> tuple[tuple[int, int] | None, float]:
        """Find the non-adjacent link pair with the smallest current clearance.

        Iterates all non-adjacent link pairs (respecting
        ``self_collision_skip_pairs``), queries getClosestPoints, returns
        the pair with the lowest distance (most penetrating / closest).
        Pairs farther than ``query_cap`` apart are ignored (pybullet
        returns no points for those — early-out per pair).

        Returns ``(pair, distance)`` or ``(None, +inf)`` if no pair is
        within ``query_cap``. Distance can be NEGATIVE (penetration).

        Reads the planner's currently-snapped joint state — caller must
        ensure pybullet is at the desired q before calling.
        """
        from splatsim.utils.rrt_path_utils import are_adjacent_links

        _skip_raw = self._collision_kwargs.get("self_collision_skip_pairs") or []
        _skip_set = {frozenset((int(a), int(b))) for a, b in _skip_raw}
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._pb_client)

        worst_dist = float("inf")
        worst_pair: tuple[int, int] | None = None
        for a, b in itertools.combinations(list(range(-1, n_joints)), 2):
            if frozenset((a, b)) in _skip_set:
                continue
            if are_adjacent_links(self._robot_id, a, b, physics_client_id=self._pb_client):
                continue
            pts = p.getClosestPoints(
                self._robot_id,
                self._robot_id,
                distance=query_cap,
                linkIndexA=a,
                linkIndexB=b,
                physicsClientId=self._pb_client,
            )
            if not pts:
                continue
            d = float(pts[0][8])
            if d < worst_dist:
                worst_dist = d
                worst_pair = (a, b)
        return worst_pair, worst_dist

    def _escape_self_collision_gradient(
        self,
        q_start: np.ndarray,
        max_iters: int = 60,
        step_size: float = 0.02,
        eps: float = 0.005,
    ) -> np.ndarray | None:
        """Escape SELF-collision via finite-difference gradient ascent on the
        worst-pair clearance.

        The default ``_escape_collision`` handles OBSTACLE collisions
        (contact-normal IK steps) and falls back to ``+z lift`` when no
        obstacle pair is detected. But the lift fallback can't fix
        self-collisions: e.g., forearm vs wrist_camera depends on
        wrist_1/wrist_2/wrist_3 joint angles, not EE height — lifting
        the assembly leaves the offending pair untouched.

        This method instead does direct joint-space search. Each iter:
          1. Find the worst-clearance non-adjacent link pair (the "active pair").
          2. Compute per-joint finite-difference gradient of that pair's
             clearance w.r.t. each joint: ∂d/∂q_i ≈ (d(q+ε e_i) − d(q−ε e_i)) / 2ε.
          3. Step q in the gradient direction (toward larger clearance).
          4. Re-check; stop if the standard collision_fn clears.

        ``query_cap`` for pair distances is 10 cm; pairs beyond that don't
        influence the gradient (gracefully — pybullet returns no points).

        Cost: 2 × num_dofs getClosestPoints queries per iter (plus the
        worst-pair scan, which is ~N² pair queries each iter, but bounded
        by skip_pairs filtering). Typical: ~12-25 queries per iter ×
        20-60 iters = 0.3-1.5 sec total. Cheap enough for an escape that
        only fires on failed plans.

        Returns waypoints ``[N, num_dofs]`` with ``waypoints[0] == q_start``
        and ``waypoints[-1]`` a cleared config, or None on failure.

        Side effect: leaves pybullet at the final attempted q. Caller is
        responsible for restoring (the surrounding plan() finally block
        already snapshots and restores).
        """
        # Use the configured self_collision_clearance as the success threshold.
        # The standard check is "clearance >= threshold". Default 0.0 = no
        # penetration. We add a tiny positive epsilon so the gradient
        # has a target slightly above the threshold (prevents oscillating
        # at exactly the threshold boundary).
        target_clearance = self._collision_kwargs.get("self_collision_clearance", 0.0) or 0.0
        target_clearance_plus = target_clearance + 1e-4

        q = np.asarray(q_start, dtype=np.float64).copy()
        waypoints: list[np.ndarray] = [q.copy()]

        for it in range(max_iters):
            self._set_joints_to(q)
            # Find worst non-adjacent self-pair at current q.
            worst_pair, worst_dist = self._find_worst_self_collision_pair()

            if worst_pair is not None and worst_dist >= target_clearance_plus:  # noqa: SIM102
                # Worst pair cleared. Confirm via the planner-contract check
                # (which also covers obstacle collisions AND uses the same
                # link-scope as RRT / is_q_in_collision) before returning.
                if not self._current_pose_in_planner_collision():
                    logger.info(
                        "Self-collision gradient escape: cleared after %d iter(s) "
                        "(final worst pair %s at %.4fm).",
                        it,
                        worst_pair,
                        worst_dist,
                    )
                    return np.asarray(waypoints, dtype=np.float64)
            if worst_pair is None:
                # No non-adjacent pair within query cap → either we already
                # cleared OR the collision is between adjacent links (which
                # we don't optimize). Random kick + continue.
                q_new = q + np.random.uniform(-step_size, step_size, self._num_dofs) * 0.5
                q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
                waypoints.append(q_new.copy())
                q = q_new
                continue

            # Compute finite-difference gradient of worst-pair clearance.
            # Two queries per joint (±ε), so 2 * num_dofs getClosestPoints calls.
            a, b = worst_pair
            grad = np.zeros(self._num_dofs)
            for i in range(self._num_dofs):
                d_pair = 0.0
                for sign, delta in ((+1, +eps), (-1, -eps)):
                    q_perturbed = q.copy()
                    q_perturbed[i] = float(
                        np.clip(
                            q_perturbed[i] + delta,
                            self._lower_limits[i],
                            self._upper_limits[i],
                        )
                    )
                    self._set_joints_to(q_perturbed)
                    pts = p.getClosestPoints(
                        self._robot_id,
                        self._robot_id,
                        distance=0.10,
                        linkIndexA=a,
                        linkIndexB=b,
                        physicsClientId=self._pb_client,
                    )
                    d = float(pts[0][8]) if pts else 0.10
                    d_pair += sign * d
                grad[i] = d_pair / (2 * eps)
            # Restore pybullet to current q (perturbations mutated it).
            self._set_joints_to(q)

            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-6:
                # Flat region — random kick to escape the saddle.
                q_new = q + np.random.uniform(-step_size, step_size, self._num_dofs)
            else:
                q_new = q + step_size * grad / grad_norm
            q_new = np.clip(q_new, self._lower_limits, self._upper_limits)
            waypoints.append(q_new.copy())
            q = q_new

            if (it + 1) % 10 == 0:
                logger.info(
                    "Self-collision gradient escape iter %d/%d: worst pair (%d,%d) at %.4fm, ||grad||=%.4f",
                    it + 1,
                    max_iters,
                    worst_pair[0],
                    worst_pair[1],
                    worst_dist,
                    grad_norm,
                )

        # Final check via the planner-contract helper so "success" matches
        # what RRT / is_q_in_collision would say.
        self._set_joints_to(q)
        if self._current_pose_in_planner_collision():
            logger.warning(
                "Self-collision gradient escape failed after %d iter(s) "
                "(final worst pair %s at %.4fm). Returning None.",
                max_iters,
                worst_pair,
                worst_dist,
            )
            return None
        logger.info(
            "Self-collision gradient escape succeeded after %d waypoint(s).",
            len(waypoints) - 1,
        )
        return np.asarray(waypoints, dtype=np.float64)

    def set_policy_history_context(
        self,
        history: object | None,
        max_lookback: int,
    ) -> None:
        """Refresh the policy-frame context that `_escape_via_policy_history_rewind`
        reads. Called by the source (`_do_plan`) before each `plan()` invocation
        so the highest-priority escape sees the up-to-date deque + counter.

        Args:
            history: non-owning reference to the wrapper's `_actual_q_history`
                deque (newest entry at [-1]). None disables the rewind escape.
            max_lookback: cap on how far back to walk the deque (in ticks).
                Typically set to `wrapper._frames_since_last_rrt_end` so the
                rewind never lands in a prior RRT cycle's trajectory. Zero or
                negative disables the rewind escape.
        """
        self._policy_history_ref = history
        self._policy_history_max_lookback = int(max_lookback)

    def _escape_via_policy_history_rewind(
        self, q_start: np.ndarray
    ) -> np.ndarray | None:
        """Highest-priority escape method (`_try_escape_chain` calls this
        first). Walks the policy's frame history newest→oldest and returns
        the first config whose clearance ≥ `escape_clearance_factor ×
        obstacle_clearance`.

        Properties when it succeeds:
          - q_start is IN-DISTRIBUTION for the policy: it's a config the
            policy was actually at at some recent tick. Subsequent
            (obs, RRT-action) recordings start on-manifold.
          - q_start has the same clearance margin as the contact-normal
            escape would target (both use `escape_clearance_factor`), so
            the trajectory has the same headroom for the first chunk steps.

        Returns None (so the chain falls through to the contact-normal
        and self-collision-gradient methods) when:
          - no history context was set on the planner (caller never invoked
            `set_policy_history_context`),
          - `_policy_history_max_lookback <= 0` (no policy-driven frames
            in the buffer — mid-RRT retry, back-to-back shield-triggered
            cycle, start of episode),
          - or no frame within the lookback window satisfies the inflated
            clearance (the policy was in/near collision throughout the
            recent buffer — typical contact-normal escape case).

        On success returns `[q_start, q_rewound_dof]` — 2 waypoints matching
        the `[N, num_dofs]` return contract of the other escape methods.
        `plan()` takes the last waypoint as the new q_start and `_do_plan`
        teleports the env there via the existing `escape_end_q` path.
        """
        hist = self._policy_history_ref
        max_lookback = self._policy_history_max_lookback
        if hist is None or max_lookback <= 0:
            return None
        try:
            hist_len = len(hist)  # type: ignore[arg-type]
        except TypeError:
            return None
        if hist_len == 0:
            return None
        # Use the rewind-specific factor (typically lower than the contact-
        # normal escape factor — rewind picks a real historical frame, so
        # it doesn't need as much margin to avoid the ramp-up cascade).
        factor = self._rewind_clearance_factor
        inflated_obstacle = self._effective_obstacle_clearance() * factor
        # Walk newest→oldest, bounded by both max_lookback and the actual
        # buffer length. steps_back=1 means "1 tick ago" = the entry that
        # was at [-1] one tick ago, i.e. `hist[-1]` is the most recent
        # actual_q (this tick's). Start at 1 to skip the current tick
        # (which IS the colliding q_start by definition).
        n_walk = min(int(max_lookback), hist_len)
        n_dof = self._num_dofs
        for steps_back in range(1, n_walk + 1):
            entry = hist[-steps_back]  # type: ignore[index]
            q_full = np.asarray(entry, dtype=np.float64).reshape(-1)
            if q_full.size < n_dof:
                continue
            # is_q_in_collision already handles arm-joint snap, gripper-
            # joint snap (from q_full[n_dof] when present), q=None pass
            # to check_links_in_collision, and skip-pairs from
            # _collision_kwargs. Returns False when the inflated clearance
            # is satisfied across all link pairs.
            in_coll = self.is_q_in_collision(q_full, obstacle_clearance=inflated_obstacle)
            if not in_coll:
                q_dof = q_full[:n_dof].copy()
                logger.info(
                    "Policy-history rewind: found safe frame %d ticks back "
                    "(clearance ≥ %.4f m, factor=%.2fx); using as escape target.",
                    steps_back,
                    inflated_obstacle,
                    factor,
                )
                return np.stack([q_start, q_dof], axis=0)
        # No safe frame in the searched window — log so the user can tell
        # how often rewind ALMOST succeeded vs never had a chance (which
        # affects whether lowering the factor would help). Falls through to
        # contact-normal escape in `_try_escape_chain`.
        logger.info(
            "Policy-history rewind: no safe frame within last %d tick(s) at "
            "clearance ≥ %.4f m (factor=%.2fx) — falling through to "
            "contact-normal escape.",
            n_walk,
            inflated_obstacle,
            factor,
        )
        return None

    def _try_escape_chain(self, q_start: np.ndarray) -> np.ndarray | None:
        """Run all escape modes in sequence, restoring pybullet joints to
        ``q_start`` between attempts so each mode starts from the same
        ground-truth pose.

        Order:
          0. Policy-history rewind (``_escape_via_policy_history_rewind``).
             Highest-priority entry — if the policy's recent frame history
             contains a config with the inflated clearance margin, the
             escape "teleports back in time" to that on-manifold config
             instead of synthesizing a contact-normal-pushed config the
             policy has never seen. Returns None to fall through when
             history is unavailable (start of episode, mid-RRT retry,
             back-to-back shield-triggered cycle) OR when no historical
             frame has enough clearance.
          1. Contact-normal escape (``_escape_collision``). Best for
             obstacle collisions — pushes the EE along the aggregated
             contact normal. Falls back to ``+z lift`` internally when it
             stalls or sees no obstacle pairs.
          2. Self-collision gradient escape
             (``_escape_self_collision_gradient``). Best for wrist-
             pretzel / arm-on-arm self-collisions that ``+z lift`` can't
             fix because the offending pair's clearance depends on joint
             angles, not EE position.

        Returns the first successful escape's waypoints (with
        ``waypoints[0] == q_start``), or None if all modes failed. On
        failure, the pybullet state is restored to q_start so the caller
        sees a consistent rollback point.
        """
        # Ensure pybullet is at q_start before the first attempt.
        self._set_joints_to(q_start)
        waypoints = self._escape_via_policy_history_rewind(q_start)
        if waypoints is not None:
            return waypoints
        # History rewind unavailable or no safe frame — restore pybullet to
        # q_start (the rewind itself shouldn't mutate, but defensive) and
        # try the contact-normal escape.
        self._set_joints_to(q_start)
        waypoints = self._escape_collision(q_start)
        if waypoints is not None:
            return waypoints
        # Contact-normal failed — restore pybullet to q_start (it's been
        # mutated by 60 iters of escape attempts) and try the gradient mode.
        logger.info(
            "Contact-normal escape failed — restoring q_start and trying self-collision gradient escape.",
        )
        self._set_joints_to(q_start)
        waypoints = self._escape_self_collision_gradient(q_start)
        if waypoints is not None:
            return waypoints
        # All failed — leave pybullet at q_start for the caller.
        self._set_joints_to(q_start)
        return None

    # ------------------------------------------------------------------ #
    #  IK candidate resolution (mirrors SplatSim's TrajectoryGenerator)  #
    # ------------------------------------------------------------------ #

    def _resolve_ee_pose_to_q_candidates(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        q_goal_bias: np.ndarray | None,
    ) -> list[np.ndarray]:
        """Sample multiple IK solutions for the EE pose, biased toward q_goal_bias.

        First attempt seeds with q_goal_bias (when provided), the rest with
        random configurations, so the canonical solution is preferred when
        feasible and we still find alternate IK branches when it isn't.
        """
        candidates: list[np.ndarray] = []
        for i in range(self._num_ik_candidates):
            seed_q = q_goal_bias if (i == 0 and q_goal_bias is not None) else None
            q = self._solve_ik(ee_pos, ee_quat, seed_q=seed_q)
            if q is not None:
                candidates.append(q)
        return self._deduplicate_q_candidates(candidates)

    def _solve_ik(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        seed_q: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve IK for a target EE pose. Returns a collision-free q or None.

        Mirrors ``TrajectoryGenerator._solve_ik`` (splatsim/utils/trajectory_generation.py)
        but operates on the wrapper's private pybullet client.
        """
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        is_biased = seed_q is not None
        if seed_q is None:
            seed_q = np.random.uniform(self._lower_limits, self._upper_limits)
        for idx, qi in zip(self._joint_indices, seed_q, strict=False):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)

        # Pass null-space arrays sized to the URDF's full movable-joint count
        # so PyBullet actually engages null-space IK (it silently disables it
        # when the array sizes don't match), giving us bias toward seed_q.
        # maxNumIterations=512 (was 100000): DLS converges or plateaus well
        # under 200 iterations — profiled identical FK error at 200 vs 100k,
        # but 100k costs ~14-380 ms/call since residualThreshold=1e-10 never
        # early-exits. This runs num_ik_candidates times per plan; the FK
        # accuracy verification below rejects any insufficiently-converged
        # solution, so the cap trades no correctness.
        q_solution = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link_index,
            list(ee_pos),
            list(ee_quat),
            **self._ik_null_space_kwargs(np.asarray(seed_q, dtype=np.float64)),
            maxNumIterations=512,
            residualThreshold=1e-10,
            physicsClientId=self._pb_client,
        )
        q_solution = np.array(q_solution[: len(self._joint_indices)])

        # Wrap to [-pi, pi]
        q_solution = ((q_solution + np.pi) % (2 * np.pi)) - np.pi

        if np.any(q_solution < self._lower_limits) or np.any(q_solution > self._upper_limits):
            return None

        # If we explicitly seeded with q_goal_bias, reject IK results that wandered
        # too far — null-space IK on a 6-DOF arm can still flip branches.
        if is_biased:
            wrapped_diff = ((q_solution - seed_q + np.pi) % (2 * np.pi)) - np.pi
            max_drift = float(np.max(np.abs(wrapped_diff)))
            if max_drift > np.pi / 3:  # 60° per-joint tolerance
                logger.debug(
                    "IK seeded with q_goal_bias drifted %.1f° from seed; falling back to random-seed IK.",
                    np.degrees(max_drift),
                )
                return None

        if check_links_in_collision(
            self._robot_id,
            self._joint_indices,
            q_solution,
            self._loaded_obstacle_ids,
            obstacle_names=self._obstacle_names,
            skip_pairs=self._ik_skip_pairs(),
            verbose=False,
            physics_client_id=self._pb_client,
            link_indices_to_check=self._planner_link_indices_to_check,
            **self._collision_kwargs,
        ):
            return None

        # Verify IK accuracy via FK. Compare the URDF LINK FRAME pose
        # (indices 4/5), not the link COM (0/1): calculateInverseKinematics
        # solves for the link frame, so a robot whose EE link has a COM
        # offset (e.g. KUKA iiwa flange, ~20 mm) would otherwise fail this
        # gate for EVERY solution. Identical for links with COM == frame
        # (all current SplatSim robots' wrist_camera_link).
        for idx, qi in zip(self._joint_indices, q_solution, strict=False):
            p.resetJointState(self._robot_id, idx, float(qi), physicsClientId=self._pb_client)
        link_state = p.getLinkState(
            self._robot_id,
            self._ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._pb_client,
        )
        actual_pos = np.array(link_state[4])
        if np.linalg.norm(actual_pos - ee_pos) > 0.005:  # 5 mm tolerance
            return None
        actual_quat = np.array(link_state[5])
        dot = float(np.clip(abs(np.dot(actual_quat, ee_quat)), -1.0, 1.0))
        if np.degrees(2 * np.arccos(dot)) > 5.0:  # 5° tolerance
            return None

        return q_solution

    @staticmethod
    def _deduplicate_q_candidates(
        candidates: list[np.ndarray], threshold_rad: float = 0.1
    ) -> list[np.ndarray]:
        """Remove near-duplicate joint configurations."""
        if len(candidates) <= 1:
            return candidates
        kept: list[np.ndarray] = []
        for c in candidates:
            if all(
                float(np.max(np.abs(((c - k + np.pi) % (2 * np.pi)) - np.pi))) > threshold_rad for k in kept
            ):
                kept.append(c)
        return kept


def check_chunk_collision(
    pb_client: int,
    robot_id: int,
    joint_indices: list[int],
    q_current: np.ndarray,
    chunk_dof_actions: np.ndarray,
    action_format: str,
    obstacle_ids: list[int],
    obstacle_clearance: float | None = None,
    self_collision_clearance: float | None = None,
    self_collision_skip_pairs: list[tuple[int, int]] | None = None,
    skip_pairs: set[tuple[int, int]] | list[tuple[int, int]] | None = None,
    obstacle_names: list[str] | None = None,
    link_indices_to_check: list[int] | None = None,
    actual_gripper_q: float | None = None,
) -> tuple[bool, int | None, str | None]:
    """Forward-kinematics safety sweep over a predicted action chunk.

    Snaps the planning-client robot through each future joint config that
    would result from executing ``chunk_dof_actions`` and reports whether
    any of them collides (obstacle or self) under the same collision contract
    RRT uses (``check_links_in_collision``).

    Used by the SharedAutonomyPolicyWrapper's future-chunk predictive shield:
    when ``rrt_collision_detection=future_chunk``, this is called every
    select_action tick to decide whether to preempt the policy and trigger
    RRT BEFORE the colliding waypoint actually executes. No teleport / no
    rewind — the wrapper triggers from the robot's CURRENT, continuous-motion
    state.

    Args:
        pb_client: pybullet physics client id (the planning one, with
            obstacles already loaded by RRTToGoalPlanner.load_obstacles).
        robot_id: planning robot's body id.
        joint_indices: movable joint indices for the DOF arm (length = num_dofs).
        q_current: (num_dofs,) current robot joint state. Used as the
            integration base for ``action_format='rel'`` and to restore the
            robot pose on exit.
        chunk_dof_actions: (n_steps, num_dofs) — the future joint actions
            from the policy chunk, already DOF-sliced (gripper dim dropped)
            and denormalized into raw joint-space units (radians).
        action_format: ``"rel"`` — offset from chunk-START state (NOT a
            per-step delta; see body of this function for the math). Each
            ``future_q[k] = q_current + chunk[k]``. ``"abs"`` — absolute
            joint targets per step (``future_q[k] = chunk[k]``).
        obstacle_ids: bodies in the planning client to check robot links
            against (matches what RRT uses).
        obstacle_clearance / self_collision_clearance / self_collision_skip_pairs:
            same semantics as the planner's ``_collision_kwargs``. Defaulted
            to SplatSim's built-in defaults when None.
        obstacle_names: optional pretty-print names for logging the offender.

    Returns:
        ``(any_collides, first_step_idx, kind)``:
        - any_collides: True if any future config collides.
        - first_step_idx: 0-indexed offset into chunk_dof_actions where
          collision was first detected, or None if no collision.
        - kind: ``"obstacle"`` or ``"self"`` (from check_links_in_collision)
          identifying which kind of collision tripped the check, or None.

    Side effect: leaves the planning robot at q_current on exit (so
    subsequent RRT planning starts from the same config). Each step
    snapshot uses p.resetJointState, which doesn't run physics.
    """
    if chunk_dof_actions.shape[0] == 0:
        return False, None, None
    if action_format not in ("rel", "abs"):
        raise ValueError(f"action_format must be 'rel' or 'abs', got {action_format!r}")

    # Lazy import — keep optional dependency surface contained.
    from splatsim.utils.rrt_path_utils import check_links_in_collision

    n_dof = len(joint_indices)
    if chunk_dof_actions.shape[1] != n_dof:
        raise ValueError(
            f"chunk_dof_actions has {chunk_dof_actions.shape[1]} action dims; "
            f"expected num_dofs={n_dof}. Caller must DOF-slice before passing."
        )
    q_current_arr = np.asarray(q_current, dtype=np.float64).reshape(-1)[:n_dof]

    # Build the absolute future joint trajectory.
    #
    # IMPORTANT — 'rel' is NOT a per-step delta format. It's "offset from
    # the chunk-START obs state". The training-time `to_relative_actions`
    # (and inference-time `to_absolute_actions`) broadcasts a SINGLE state
    # across all chunk timesteps (see
    # ``relative_action_processor.to_absolute_actions:122-126`` —
    # ``state_offset.unsqueeze(-2)`` widens the state to time dim, then
    # ``+=`` adds it to every step). So target k = chunk[k] + q_current,
    # NOT cumsum(chunk[..k]) + q_current. Using cumsum here would
    # overestimate the predicted motion by ~k×, causing the FK shield to
    # hallucinate collisions far beyond where the policy is actually
    # committing to go.
    #
    # For 'abs', chunk[k] IS the absolute joint target k directly.
    chunk_arr = np.asarray(chunk_dof_actions, dtype=np.float64)
    if action_format == "rel":  # noqa: SIM108 — branches are differently commented; ternary would lose the explanatory comments
        future_qs = q_current_arr[None, :] + chunk_arr
    else:  # abs
        future_qs = chunk_arr

    # Snap & check each future q. Return early on first collision so the
    # cost is bounded by the position of the offending step.
    collide_kwargs = {}
    if obstacle_clearance is not None:
        collide_kwargs["obstacle_clearance"] = obstacle_clearance
    if self_collision_clearance is not None:
        collide_kwargs["self_collision_clearance"] = self_collision_clearance
    if self_collision_skip_pairs:
        collide_kwargs["self_collision_skip_pairs"] = self_collision_skip_pairs
    if skip_pairs:
        # Forward env-config's per-obstacle `skip_collision_robot_links`
        # (e.g. base_link ⟷ table) so the shield uses the SAME contract as
        # RRT planning / the controller's per-tick check. Without this the
        # shield would false-fire on structural near-contacts at the same
        # clearance the planner already excuses.
        collide_kwargs["skip_pairs"] = skip_pairs
    if obstacle_names is not None:
        collide_kwargs["obstacle_names"] = obstacle_names
    # When the caller doesn't scope the link set, default to everything
    # downstream of (and including) the STATIC base_link — i.e., skip only
    # the world frame (-1). base_link (0) IS included so the shield catches
    # gripper/arm swinging back into the robot's own mount, matching the
    # planner's `_planner_link_indices_to_check` scope. If a specific env
    # has base_link geometrically overlapping an obstacle AABB (e.g., an
    # older UR5+table setup where the mount is very close to the table
    # top), silence it per-env via `skip_pairs`, which is obstacle-side
    # only — the base_link↔gripper self-check stays live.
    if link_indices_to_check is None:
        n_joints = p.getNumJoints(robot_id, physicsClientId=pb_client)
        link_indices_to_check = list(range(0, n_joints))
    collide_kwargs["link_indices_to_check"] = link_indices_to_check

    # Pre-snap the gripper joints (URDF indices ≥ n_dof+1) ONCE to the
    # env's actual gripper config from `actual_gripper_q`. Mirrors the fix
    # in `RRTToGoalPlanner.is_q_in_collision` — without it, the per-waypoint
    # `check_links_in_collision(q=future_qs[k])` call internally invokes
    # `set_robot_joint_positions(q)` which forces `open_gripper(robot_id)`
    # (resets every joint ≥7 to 0.0). For grasp tasks where the env's
    # gripper closes during approach, the planner's OPEN-gripper geometry
    # has wider fingers than the env's CLOSED gripper; predicted future_qs
    # that bring the EE near a target object (intentional, that's the
    # approach goal) then false-fire the shield because the OPEN fingers
    # geometrically overlap the object even though the env's CLOSED fingers
    # would have clearance. We snap once here (gripper is constant across
    # the chunk's FK projection since chunk_dof_actions excludes the gripper
    # dim) then pass q=None per waypoint to skip the redundant gripper-open
    # call. Skipped when actual_gripper_q is None (legacy callers).
    n_dof = len(joint_indices)
    if actual_gripper_q is not None:
        num_joints = p.getNumJoints(robot_id, physicsClientId=pb_client)
        for idx in range(n_dof + 1, num_joints):
            p.resetJointState(robot_id, idx, float(actual_gripper_q), physicsClientId=pb_client)
    try:
        for k in range(future_qs.shape[0]):
            if actual_gripper_q is not None:
                # Snap arm joints manually; gripper joints stay pre-snapped
                # above. Pass q=None to check_links_in_collision so it
                # doesn't re-invoke set_robot_joint_positions (which would
                # reopen the gripper AND run stepSimulation per waypoint).
                for j_idx, qi in zip(joint_indices, future_qs[k].tolist(), strict=True):
                    p.resetJointState(robot_id, j_idx, float(qi), physicsClientId=pb_client)
                _q_arg = None
            else:
                # Legacy path: caller didn't pass actual_gripper_q, so let
                # check_links_in_collision do its full snap (including
                # the open_gripper side effect, matching the historical
                # behavior).
                _q_arg = future_qs[k].tolist()
            # verbose=False keeps the per-tick log quiet. The caller's
            # "Future-chunk shield: predicted X collision at step Y"
            # line is the only shield output in production. Flip to True
            # temporarily if you need per-pair attribution for debugging.
            colliding, kind = check_links_in_collision(
                robot_id,
                joint_indices,
                _q_arg,
                obstacle_ids,
                verbose=False,
                physics_client_id=pb_client,
                return_kind=True,
                **collide_kwargs,
            )
            if colliding:
                return True, k, kind
        return False, None, None
    finally:
        # Restore the robot to q_current so subsequent RRT planning starts
        # from the same physical state the wrapper just queried.
        for j_idx, qi in zip(joint_indices, q_current_arr.tolist(), strict=True):
            p.resetJointState(robot_id, j_idx, float(qi), physicsClientId=pb_client)


def extract_task_goal(
    env_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Pull the RRT goal (target EE pose + optional q_goal_bias seed) from the env config.

    Returns ``(target_ee_pos, target_ee_quat, q_goal_bias_or_none)`` when the
    task has a defined target EE pose, or ``None`` otherwise. The caller should
    surface ``None`` as a planning error rather than silently falling back.
    """
    task = env_config.get("task") if env_config else None
    if not task:
        return None
    pos = task.get("target_ee_pos")
    quat = task.get("target_ee_quat")
    if pos is None or quat is None:
        return None
    bias = task.get("q_goal_bias")
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat, dtype=np.float64),
        np.asarray(bias, dtype=np.float64) if bias is not None else None,
    )
