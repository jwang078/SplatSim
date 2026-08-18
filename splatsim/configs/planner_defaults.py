"""Compatibility re-export. The canonical module lives in
``splatsim.utils.planner_defaults`` — it must be importable WITHOUT running
``splatsim/configs/__init__`` (which imports mode_config, which imports
rrt_to_goal, which needs these defaults: a cycle otherwise)."""

from splatsim.utils.planner_defaults import PLANNER_DEFAULTS, PlannerDefaults

__all__ = ["PLANNER_DEFAULTS", "PlannerDefaults"]
