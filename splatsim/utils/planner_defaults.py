"""Canonical defaults for the RRT planner's smoothing / time-parametrization
pipeline — the SINGLE source of truth shared by every endpoint.

Why this module exists. Three places used to carry their own copies of these
numbers: ``RRTToGoalPlanner.__init__`` (the shared planner), SplatSim's
``TrajectoryGenModeConfig`` (trajectory generation), and lerobot's
``SharedAutonomyConfig`` (interventions). They drifted — interventions ran
``rrt_smooth_iterations=50`` and ``elastic_smooth_passes=0`` while
trajectory-gen ran 200/30, and the missing elastic pass shipped visible speed
judder into recorded DAgger chunks before anyone noticed. Now:

  * ``RRTToGoalPlanner.__init__`` takes its defaults FROM here;
  * ``TrajectoryGenModeConfig`` fields default FROM here (its exported JSON
    configs still override per env, as before);
  * lerobot's ``SharedAutonomyConfig`` cannot import splatsim (optional
    dependency), so its planner-forwarded fields default to ``None`` =
    "inherit the planner default" and are only forwarded when explicitly set.

Change a value here and every endpoint that has not explicitly overridden it
follows. The values themselves are the trajectory-generation-proven set (the
pipeline that produced the clean demo datasets).

Location note: this lives in ``splatsim.utils`` (no package ``__init__``
side effects), NOT ``splatsim.configs`` — ``configs/__init__`` imports
mode_config which imports rrt_to_goal which needs these defaults, so hosting
it there made ``import splatsim.utils.rrt_to_goal`` circular.
``splatsim.configs.planner_defaults`` remains as a re-export shim.

Deliberately import-light: this module is imported by both the planner and the
config layer, so it must never pull in pybullet/lerobot/torch.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerDefaults:
    # Kinematic limits handed to time parametrization (per joint).
    max_joint_vel: float = 0.5
    max_joint_acc: float = 1.0
    # NOTE: honored by the "retimed" (default)/"follower"/"chained"/"ruckig"
    # backends; the bare "toppra" backend has no third-order term and
    # ignores it.
    max_joint_jerk: float = 10.0

    # Random-shortcut smoothing iterations on the raw BiRRT path. 200 (not the
    # planner's old 50): leftover zigzags survive into the trajectory as
    # direction changes, each forcing a decelerate/re-accelerate.
    rrt_smooth_iterations: int = 200

    # Corner-rounding relaxation AFTER trajopt, BEFORE parametrization. 30:
    # trajopt's FD repulsion leaves waypoint-scale jitter (worst when its RDP
    # decimation falls back to raw dense output) that renders as 5-8 Hz speed
    # judder; elastic irons it out. Ordering rationale lives in
    # rrt_path_utils.get_path.
    elastic_smooth_passes: int = 30

    # CHOMP-lite trajopt (soft collision repulsion + smoothness), run once on
    # the winning candidate (deferred postprocess).
    trajopt_passes: int = 15
    trajopt_lr: float = 0.02
    trajopt_smoothness_weight: float = 1.0
    trajopt_collision_weight: float = 5.0
    trajopt_collision_threshold: float = 0.10
    trajopt_fd_step: float = 0.01

    # Corner handling during parametrization: False = carry speed (corners
    # blended within rrt_path_utils.DEFAULT_CORNER_BLEND_RAD); True = stop at
    # sharp (>45 deg) corners.
    segment_at_sharp_corners: bool = False

    # Equalize joint-space path speed across sections — removes the
    # direction-anisotropy surging of per-joint box velocity limits.
    uniform_path_speed: bool = True

    # Final-approach taper: brake over the last `dist` rad of joint-space arc
    # to scaled-down vel/acc so a PD-tracked robot doesn't carry momentum past
    # the goal.
    final_approach_dist: float = 0.15
    final_approach_vel_scale: float = 0.5
    final_approach_acc_scale: float = 0.25


PLANNER_DEFAULTS = PlannerDefaults()
