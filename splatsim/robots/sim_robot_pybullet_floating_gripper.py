"""Floating-gripper visualization env.

A 6-DoF "virtual arm" (floating_gripper.urdf: 3 prismatic + 3 revolute joints
that ARE the end-effector pose, plus the shared Robotiq 2F-85) for visualizing
recorded end-effector trajectories — e.g. UMI/roboharvest demos converted to
LeRobot format — via the eval-benchmark replay machinery.

state/action layout: [x, y, z, rx, ry, rz, gripper]
  * position in metres, orientation as INTRINSIC XYZ Euler angles
    (scipy `Rotation.as_euler('XYZ')`), matching the URDF joint chain
  * gripper: 0 = open, 1 = closed (standard SplatSim convention)

Launch:
  python scripts/launch_nodes.py --robot sim_pybullet_floating_gripper \
      --eval_benchmark_repo_id <repo_id>
then use the GUI's Eval Benchmark tab (Replay Episode) to play back episodes.
"""

from splatsim.configs.env_config import EnvConfig
from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase


class FloatingGripperPybulletRobotServer(PybulletRobotServerBase):
    """Pure-physics floating gripper; no splats, no task objects."""

    DEFAULT_ROBOT_NAME = "floating_gripper"

    # No splat assets; render through PyBullet's camera instead.
    RENDER_SPLATS = False
    RENDER_PYBULLET_CAMERA = True
    # Frame the typical UMI workspace (trajectories roughly x,y in [-0.5, 0.5],
    # z in [0.6, 1.4] in the tag/EKF frame).
    PYBULLET_CAMERA_EYE = (1.6, -1.6, 1.4)
    PYBULLET_CAMERA_TARGET = (0.0, -0.2, 1.0)
    PYBULLET_CAMERA_FOV = 60.0

    # The UR-specific link-index skip pairs don't apply to this URDF. The
    # shared Robotiq gripper pairs are resolved by NAME in the base class
    # (GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES) and merged in automatically.
    SELF_COLLISION_SKIP_PAIRS = []
    SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA = []
    CHECK_ADJACENT_LINK_PAIRS_NAMES = []

    # No scene objects: replay only needs the robot itself.
    ENV_CONFIG = EnvConfig(
        name="floating_gripper",
        objects=[],
        task_description="Floating-gripper EEF trajectory replay",
    )

    # No objects -> nothing to record as privileged env state.
    ORACLE_RECORD_ENV_STATE = False

    def num_dofs(self) -> int:
        # Joints 1..6 are the virtual pose DOFs (x, y, z, rx, ry, rz); joint 0
        # is the fixed world_joint and everything after 6 is the gripper.
        return 6
