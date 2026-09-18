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

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from splatsim.configs.env_config import EnvConfig, SplatObjectConfig
from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase

logger = logging.getLogger(__name__)


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

    # =========================================================================
    # Gym Environment Interface
    # =========================================================================
    #
    # This env is REPLAY-ONLY: there is no task, no objects and no goal, so the
    # reward/success/termination hooks are constant. They still have to exist —
    # PybulletRobotServerBase declares them abstract and `serve()` calls
    # `reset()` before the ZMQ thread starts.

    # Pose the virtual arm starts (and re-starts) at: the URDF zero
    # configuration, i.e. EE at the origin of the tag/EKF frame with identity
    # orientation. Episode replay teleports to the recorded frame-0 pose
    # immediately afterwards, so this only matters for the pre-replay idle view.
    RESET_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the floating gripper to its home pose with the gripper open.

        No randomization: `randomize_objects()` has nothing to place, and a
        randomized start pose would only add a jump cut before the first
        replayed frame.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._step_count = 0
        self._episode_started = True

        self.teleport_joint_state(self.splatsim_robot, list(self.RESET_JOINTS))
        # Match the small-engine reset: nothing else in the reset path touches
        # the gripper joints, so without this the previous episode's closed
        # gripper bleeds into frame 0 of the next one.
        self.open_gripper()

        for _ in range(100):
            self.pybullet_client.stepSimulation()

        return self.get_observations(), {"is_success": False}

    def compute_reward(self) -> float:
        return 0.0

    def check_success(self) -> bool:
        return False

    def check_terminated(self) -> bool:
        # Replay ends when the recorded episode runs out, never from here.
        return False

    def serve_loop(self) -> None:
        # Like the small-engine env: the base `serve()` already does all the
        # per-mode stepping (INTERACTIVE / EVAL_BENCHMARK), and there is no
        # trajectory-generation mode here, so this hook has nothing to add.
        pass

    def plan_given_this_state(self, initial_joint_positions):
        # No motion planning in a replay env; the recorded trajectory IS the plan.
        return []

    # =========================================================================
    # Gimbal-safe orientation tracking
    # =========================================================================
    #
    # The URDF's revolute chain is an INTRINSIC XYZ Euler decomposition
    # (R = Rx(j4) @ Ry(j5) @ Rz(j6)), which is singular at j5 = ±90°: there
    # only (j4 - j6) is determined, so j4 and j6 individually become
    # ill-conditioned. Recorded UMI grape demos live right on that edge
    # (measured |j5| up to 87.3°), and `Rotation.as_euler('XYZ')` returns the
    # canonical branch frame-by-frame with no memory of the previous one — so
    # a 26° pose change gets encoded as a 127° jump in j4 AND j6 that flips
    # back on the next frame. The pose ENDPOINTS are exact, but
    # command_joint_state drives the arm with POSITION_CONTROL, which
    # interpolates LINEARLY IN JOINT SPACE: mid-interpolation the wrist swings
    # through orientations that are on no part of the recorded trajectory —
    # visibly, the gripper points down (with gravity) at the twist while the
    # demo had it pointing up.
    #
    # Fix, applied per command below:
    #   1. Re-branch. Rx(a+π) @ Ry(π-b) @ Rz(c+π) == Rx(a) @ Ry(b) @ Rz(c), so
    #      every Euler triple has a second representation of the SAME rotation.
    #      Pick whichever of the two (and whichever 2π wrap of each angle) is
    #      closest to the joints we are currently at. Measured on
    #      local/umi_eef_jenny_test: worst-case per-frame joint step drops
    #      126.7° -> 92.0° (ep0) and 131.8° -> 77.1° (ep1) with the replayed
    #      pose unchanged to 3e-5 degrees.
    #   2. Snap. Even a continuous target is interpolated through joint space,
    #      and near the singularity that path still bulges away from the
    #      recorded orientation. This env has no task objects and no contact
    #      dynamics worth preserving — it is a trajectory VIEWER — so the six
    #      virtual DOFs are set kinematically and every rendered frame shows
    #      the recorded pose exactly. The gripper still goes through
    #      `move_gripper` (base class) so the Robotiq mimic linkage is
    #      physical.
    #
    # What this does NOT fix: jitter in the recorded poses themselves (the
    # tags drop out and the EEF pose steps up to 45° between frames on this
    # dataset). That is a capture-side problem — smooth it in the calibration
    # pipeline, not here, where any filtering would silently misreport what
    # the dataset actually contains.

    # Kinematically snap the 6 virtual pose DOFs instead of servoing to them.
    SNAP_POSE_DOFS = True

    @staticmethod
    def _equivalent_euler_branches(euler: np.ndarray) -> List[np.ndarray]:
        """The two intrinsic-XYZ triples that denote the same rotation."""
        rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])
        return [
            np.array([rx, ry, rz]),
            np.array([rx + np.pi, np.pi - ry, rz + np.pi]),
        ]

    def _continuous_pose_dofs(self, joint_state: np.ndarray) -> np.ndarray:
        """Re-express `joint_state`'s orientation on the branch nearest to the
        arm's CURRENT joint angles. Position and gripper pass through."""
        if len(joint_state) < 6:
            return joint_state
        prev = np.array(
            [
                self.pybullet_client.getJointState(self.splatsim_robot.sim_id, i + 1)[0]
                for i in range(3, 6)
            ]
        )
        two_pi = 2.0 * np.pi
        best = None
        for cand in self._equivalent_euler_branches(np.asarray(joint_state[3:6], float)):
            # Slide each angle by whole turns to sit nearest `prev`; Rx/Ry/Rz
            # are 2π-periodic so this is free.
            cand = cand + two_pi * np.round((prev - cand) / two_pi)
            dist = float(np.abs(cand - prev).sum())
            if best is None or dist < best[0]:
                best = (dist, cand)
        out = np.array(joint_state, dtype=np.float64)
        out[3:6] = best[1]
        return out

    def command_joint_state(self, splatsim_obj, joint_state, step_physics: bool = True) -> None:
        if splatsim_obj is self.splatsim_robot:
            joint_state = self._continuous_pose_dofs(np.asarray(joint_state, dtype=np.float64))
            if self.SNAP_POSE_DOFS:
                for i in range(min(len(joint_state), self.num_dofs())):
                    # targetVelocity=0 because a snap is kinematic: carrying
                    # the pre-snap velocity into the next step would let the
                    # integrator drift off the pose we just set.
                    self.pybullet_client.resetJointState(
                        splatsim_obj.sim_id, i + 1, float(joint_state[i]),
                        targetVelocity=0.0,
                    )
        super().command_joint_state(splatsim_obj, joint_state, step_physics=step_physics)

    def teleport_joint_state(self, splatsim_obj, joint_state, joint_velocities=None) -> int:
        # Same re-branching on the replay's frame-0 teleport, so the very first
        # commanded frame is already measured against a compatible branch.
        if splatsim_obj is self.splatsim_robot:
            joint_state = self._continuous_pose_dofs(np.asarray(joint_state, dtype=np.float64))
        return super().teleport_joint_state(
            splatsim_obj, joint_state, joint_velocities=joint_velocities
        )


class VineFloatingGripperPybulletRobotServer(FloatingGripperPybulletRobotServer):
    """Floating gripper replayed INSIDE the scanned grape-vine splat.

    Same replay machinery as the base class, but the frame is photoreal: the
    `vine_scene` Gaussian scan renders as the background and the vine's
    collision mesh loads at the origin (its splat->sim transform is pre-baked
    into the URDF — see `vine_scene/vine_and_trellis` in data/stages/vine_scene/stage.yaml), so a recorded UMI
    trajectory can be watched against the actual grape clusters it was recorded
    on.

    The gripper was never scanned, so it has no gaussians of its own — it is
    drawn in from PyBullet and depth-composited against the splat
    (RENDER_ROBOT_SPLAT=False + COMPOSITE_PYBULLET_ROBOT=True), which is what
    makes the canopy occlude it as it reaches in.

    FRAME WARNING: the recorded poses are in UMI's tag/EKF frame and the splat
    is in SplatSim's sim frame. These are different captures and NOT the same
    frame — on local/umi_eef_jenny_smooth the track spans y in [-0.35, -0.05]
    while the vine occupies y in [0.15, 1.06], i.e. they do not even overlap,
    so with the default identity transform the gripper flies in empty space in
    front of the vine. Set UMI_TO_SIM_* below once you have the extrinsic, or
    use ALIGN_TO_BUNCH_INDEX for a translation-only placeholder that parks the
    track on a bunch so its scale and shape can be eyeballed against the fruit.
    """

    # Vine assets, shared with the UR5 vine env so the two stay in sync.
    RENDER_SPLATS = True
    background_splat_name = "vine_scene"
    # No robot scan exists for a floating UMI gripper — composite it from
    # PyBullet instead of looking for gaussians and labels that do not exist.
    RENDER_ROBOT_SPLAT = False
    COMPOSITE_PYBULLET_ROBOT = True

    # Splat base camera framing the canopy (same viewpoint the UR5 vine env
    # uses — verified to show the vine and its bunches).
    BASE_CAMERA_OVERRIDE_XYZ = (0.65, -0.75, 0.85)
    BASE_CAMERA_OVERRIDE_RPY = (-1.725719, 0.0, 0.674741)
    BASE_CAMERA_OVERRIDE_DIST_INC = 0.0

    RENDER_PYBULLET_CAMERA = True
    PYBULLET_CAMERA_EYE = (0.5, -1.1, 1.0)
    PYBULLET_CAMERA_TARGET = (-0.4, 0.55, 0.5)
    PYBULLET_CAMERA_FOV = 65.0

    # UMI tag/EKF frame -> SplatSim sim frame. Identity by default because the
    # true extrinsic is not recoverable from the dataset alone; fill it in when
    # you have it (translation in metres, rotation as extrinsic-xyz euler rad).
    UMI_TO_SIM_TRANSLATION = (0.0, 0.0, 0.0)
    UMI_TO_SIM_ROTATION_RPY = (0.0, 0.0, 0.0)
    # Placeholder alignment: translate the whole replayed track so its centroid
    # lands on this grape bunch (largest-first index, same ordering the UR5 vine
    # env's TARGET_BUNCH_INDEX uses). None = off. Rotation is NOT solved for, so
    # this shows scale and path shape against the fruit, not true registration.
    ALIGN_TO_BUNCH_INDEX = None

    ENV_CONFIG = EnvConfig(
        name="vine_floating_gripper",
        task_description="UMI EEF trajectory replay on the scanned grape vine",
        terminate_on_collision=False,
        objects=[
            SplatObjectConfig(
                name="vine",
                splat_name="vine_scene/vine_and_trellis",
                grasp_configs=[],
                randomize_pose=False,
                rotation_range_z=(0, 0),
                position_range_x=(0, 0),
                position_range_y=(0, 0),
                position_range_z=(0, 0),
                # Collision URDF is pre-baked in sim frame -> loads at origin.
                # The vine's gaussians are already part of the `vine_scene`
                # background scan, so loading them again would double them.
                load_splat=False,
            ),
        ],
    )

    def __init__(self, *args, **kwargs):
        # BEFORE super(): the base constructor teleports the robot to its
        # initial joint positions, which routes through our frame-mapping
        # override — the transform has to exist by then.
        self._align_bunch_center = None
        self._align_bunch_offset = None
        self._umi_to_sim = self._resolve_umi_to_sim()
        super().__init__(*args, **kwargs)

    def _resolve_umi_to_sim(self) -> np.ndarray:
        """4x4 UMI-frame -> sim-frame transform applied to every replayed pose."""
        from scipy.spatial.transform import Rotation

        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", self.UMI_TO_SIM_ROTATION_RPY).as_matrix()
        T[:3, 3] = np.asarray(self.UMI_TO_SIM_TRANSLATION, dtype=np.float64)
        if self.ALIGN_TO_BUNCH_INDEX is None:
            return T

        # Placeholder alignment: shift so the track centroid sits on a bunch.
        # Needs the episode states, which only the replay has — resolve lazily
        # there and cache; here we just record that it was requested.
        try:
            from splatsim.robots.sim_robot_pybullet_vine import (
                VineGrapeReachPybulletRobotServer as _Vine,
            )
            from splatsim.utils import grape_targets

            bunches = grape_targets.load_targets(_Vine.GRAPE_TARGETS_JSON)
            self._align_bunch_center = np.asarray(
                bunches[int(self.ALIGN_TO_BUNCH_INDEX)]["center"], dtype=np.float64
            )
            print(
                f"[vine-fg] ALIGN_TO_BUNCH_INDEX={self.ALIGN_TO_BUNCH_INDEX}: replayed "
                f"track will be translated onto bunch centre "
                f"{np.round(self._align_bunch_center, 3)} — PLACEHOLDER, not a "
                f"calibrated extrinsic."
            )
        except Exception:
            logger.exception("[vine-fg] could not load grape bunches; alignment disabled")
            self._align_bunch_center = None
        return T

    def _apply_umi_to_sim(self, joint_state: np.ndarray) -> np.ndarray:
        """Map one [x, y, z, rx, ry, rz, gripper] pose into the sim frame."""
        from scipy.spatial.transform import Rotation

        if len(joint_state) < 6:
            return joint_state
        out = np.array(joint_state, dtype=np.float64)
        T = self._umi_to_sim
        offset = getattr(self, "_align_bunch_offset", None)
        pos = T[:3, :3] @ out[:3] + T[:3, 3]
        if offset is not None:
            pos = pos + offset
        rot = Rotation.from_matrix(T[:3, :3]) * Rotation.from_euler("XYZ", out[3:6])
        out[:3] = pos
        out[3:6] = rot.as_euler("XYZ")
        return out

    def _eval_benchmark_replay_episode(self):
        # Resolve the placeholder bunch alignment from THIS episode's states —
        # it is defined as "centroid of the replayed track", which is not known
        # until the episode is picked.
        self._align_bunch_offset = None
        center = getattr(self, "_align_bunch_center", None)
        if center is not None:
            states = self._current_episode_states()
            if states is not None and len(states):
                T = self._umi_to_sim
                pts = (T[:3, :3] @ np.asarray(states)[:, :3].T).T + T[:3, 3]
                self._align_bunch_offset = center - pts.mean(axis=0)
        return super()._eval_benchmark_replay_episode()

    def _current_episode_states(self):
        """The current benchmark episode's observation.state rows, or None."""
        if self._lerobot_saver is None or not self._eval_benchmark_subset:
            return None
        try:
            episode_id = self._eval_benchmark_subset[self._eval_benchmark_episode_index]
            table = self._lerobot_saver.select_columns(
                ["episode_index", "frame_index", "observation.state"]
            ).to_pandas()
            rows = table[table["episode_index"] == episode_id].sort_values("frame_index")
            return np.stack(rows["observation.state"].to_numpy())
        except Exception:
            logger.exception("[vine-fg] could not read episode states for alignment")
            return None

    def command_joint_state(self, splatsim_obj, joint_state, step_physics: bool = True) -> None:
        if splatsim_obj is self.splatsim_robot:
            joint_state = self._apply_umi_to_sim(np.asarray(joint_state, dtype=np.float64))
        super().command_joint_state(splatsim_obj, joint_state, step_physics=step_physics)

    def teleport_joint_state(self, splatsim_obj, joint_state, joint_velocities=None) -> int:
        if splatsim_obj is self.splatsim_robot:
            joint_state = self._apply_umi_to_sim(np.asarray(joint_state, dtype=np.float64))
        return super().teleport_joint_state(
            splatsim_obj, joint_state, joint_velocities=joint_velocities
        )
