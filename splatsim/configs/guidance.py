"""Guidance lifecycle enum shared across the SplatSim / LeRobot boundary.

`GuidanceMode` is owned here, not in LeRobot, because SplatSim's planner
(`splatsim.utils.rrt_to_goal.RRTRuntimeState.mode`) needs it and SplatSim must
import without LeRobot installed — LeRobot is the dependent side (it imports the
gym env, the robot server and the planner from here). LeRobot's
`lerobot.policies.guidance.base` re-exports this exact object, so
`state.mode == RRTMode.IDLE` comparisons stay valid across the repo boundary.

Deliberately dependency-free: this module is imported during
`splatsim.configs` package init.
"""

from enum import Enum


class GuidanceMode(Enum):
    """Method-triggered lifecycle state of a guidance source.

    For observation-driven sources, this stays at IDLE; activation is
    decided by `is_active()` based on observation content.
    """

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
