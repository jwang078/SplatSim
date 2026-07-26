"""Fast, non-rendered planar 3-joint arm env.

A minimal server for the simple planar (RRR) arm defined in
`splatsim/robot_definitions/urdf/planar_3joint.urdf`. It runs as a pure PyBullet
physics sim with NO Gaussian-splat rendering — `RENDER_SPLATS` is False, so the
constructor loads only the URDF (arm + Robotiq gripper) and skips every splat
asset (labels.npy, robot/background splats, base camera).

`get_observations()` still returns full oracle state (joint positions, EE pose,
per-object poses), which is exactly what an oracle-info policy consumes. Use
this env for fast iteration; use the splat envs when you need rendered images.

It is a DEBUG PLATFORM for the small_engine env: it inherits from
`SmallEnginePybulletRobotServer` and reuses its reset / object-placement /
smoothness-tracking / serve machinery verbatim, so bugs ironed out here (and
shared fixes) carry over to small_engine. The only planar-specific overrides are:
  * the arm-start hook `randomize_ee_pose()` — the planar arm has a FIXED start
    pose (no 6-DOF IK randomization), and
  * `check_metrics()` — a planar reach-to-block task instead of a fixed EE goal.

The UR-specific `SELF_COLLISION_SKIP_PAIRS` (arm/wrist indices) do NOT apply to
the 3-joint arm and are reset to []; the shared Robotiq GRIPPER pairs still come
from the base's `GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES` (resolved by name).

Everything scene-specific lives in `ENV_CONFIG`. Launch it like small_engine:

    python scripts/launch_nodes.py \
        --robot sim_pybullet_planar_interactive \
        --robot_port 6001 --robot_name planar_3joint \
        --wrist_cam_ver=2 --no_camera_rendering
"""

from typing import Any, Dict, Optional

import numpy as np

from splatsim.configs import (
    EnvConfig,
    CuboidObjectConfig,
    TrajectoryGenModeConfig,
)
from splatsim.robots.sim_robot_pybullet_small_engine import (
    SmallEnginePybulletRobotServer,
)


class PlanarPybulletRobotServer(SmallEnginePybulletRobotServer):
    """No-render planar reach env sharing small_engine's reset/placement code.

    Task: reach the object named ``TARGET_OBJECT_NAME`` with the gripper.
    Success = EE within ``pos_tolerance_m`` of the target."""

    # Pure physics — no splat assets loaded. `background_splat_name` may stay
    # None because the background-splat requirement is gated behind this flag.
    RENDER_SPLATS = False

    # The UR5 arm/wrist skip pairs inherited from SmallEnginePybulletRobotServer
    # use UR link INDICES that are wrong for the 3-joint planar arm, so clear
    # them. The shared Robotiq GRIPPER pairs are NOT listed here — they live in
    # the base's GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES (by link name) and are
    # resolved + merged in at construction, identically for both robots.
    SELF_COLLISION_SKIP_PAIRS = []
    SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA = []
    # Empty. Turns out the planar arm's URDF ISN'T slim enough to avoid the
    # natural pivot overlap either — pybullet's getClosestPoints between
    # adjacent parent-child links returns near-zero at every reachable joint
    # config, not just at fold-over angles. Whitelisting these pairs made
    # `is_robot_in_collision` reject every randomized start pose in the
    # scene-solvability check → randomize_objects gave up on 100/100 attempts.
    # Non-adjacent pairs (link_1 ↔ link_3) still get checked normally and
    # catch the important fold-over case (elbow folded back over shoulder).
    CHECK_ADJACENT_LINK_PAIRS_NAMES = []

    # Reach-task parameters (override per concrete env if desired). Success
    # tolerance (6 cm) exceeds the block's half-diagonal (~4.3 cm for a 5 cm
    # cube), so the gripper reaching the block always registers as SUCCESS
    # before it could register as an obstacle collision.
    TARGET_OBJECT_NAME = "block"
    OBSTACLE_OBJECT_NAMES = ("obstacle_1", "obstacle_2")
    pos_tolerance_m = 0.06

    # ── Object placement: annulus around the base pivot (all 4 quadrants) ─────
    # Randomizable objects are placed on an annulus around the arm's base pivot
    # in the vertical X-Z plane, sampling the FULL circle (theta over 0..2pi) so
    # they land in ANY of the four quadrants around the base — not stuck in the
    # single +X/+Z rectangle. A rectangular position range can't express this;
    # `randomize_object_pose` is overridden below to do polar sampling.
    #
    # Center = base pivot (joint_1): world_joint lifts base_link to z=0.1 and
    # joint_1 sits 0.1 above that -> (x=0, z=0.2). Radii stay within the planar
    # arm's ~0.9 m reach so the target stays reachable; obstacles get a wider
    # band. Per-object radius ranges (meters) fall back to the default.
    PLACEMENT_CENTER_XZ = (0.0, 0.2)
    PLACEMENT_RADIUS_RANGE_DEFAULT = (0.25, 0.60)
    PLACEMENT_RADIUS_RANGES = {
        "block": (0.30, 0.55),
        "obstacle_1": (0.20, 0.62),
        "obstacle_2": (0.20, 0.62),
    }

    # ── Controller / joint dynamics for goal-settling ────────────────────────
    # The arm links are light (masses 0.5-1.0, tiny inertias) and the URDF has
    # zero joint damping, so the default UR5 controller (maxVelocity 3.14) lets
    # the 240 Hz servo close each held ~30 Hz waypoint faster than the planned
    # ~0.5 rad/s, overshooting + ringing. Two fixes:
    #   * CONTROL_MAX_VELOCITY = None TIES the servo cap to the trajectory
    #     generator's max_joint_vel, so the servo may move at exactly the planned
    #     velocity and no faster — one knob (max_joint_vel) drives both planning
    #     and execution. (Measured: a fixed 1.0 was inert here — the force-limited
    #     peak was 0.686 < 1.0 — so only tying to the plan actually binds.)
    #   * per-joint damping (URDF has none) dissipates residual oscillation.
    # Tune via the generator's max_joint_vel (planning+control) and JOINT_DAMPING.
    CONTROL_MAX_VELOCITY = None
    JOINT_DAMPING = 2.0

    # Fast, splat-free image observations via PyBullet's getCameraImage. True
    # side view along +Y (the X-Z plane normal), so block (x,z) maps 1:1 to
    # pixels with no depth ambiguity. Framed TIGHT: the eye distance is set so
    # the object placement disk (radius up to ~0.66 m around the pivot at
    # (0,0,0.2)) fills ~85% of the frame — small task objects (5-7 cm cubes)
    # otherwise occupy too few pixels for precise localization. The frame can't
    # go tighter without clipping edge-placed objects (bounded by
    # PLACEMENT_RADIUS_RANGES); to make cubes bigger still, shrink that annulus
    # or enlarge the cubes. Rendered at the base's native 224² (no upscale).
    RENDER_PYBULLET_CAMERA = True
    PYBULLET_CAMERA_EYE = (0.0, -1.1,0.2)   # closer than base default -> ~85% fill
    PYBULLET_CAMERA_TARGET = (0.0, 0.0, 0.2)  # pivot; centers the workspace disk
    PYBULLET_CAMERA_FOV = 60.0

    # The planar arm has 3 revolute joints (indices 1,2,3); joint index 0 is
    # the fixed world_joint, and everything after 3 is the gripper. The base
    # class hardcodes 6 for the UR5, so it MUST be overridden here.
    def num_dofs(self) -> int:
        return 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The target block is a non-colliding marker: disable its PHYSICS
        # collision with every robot link so the gripper can reach its center
        # (and never stutters against it). The QUERY/planner side is handled by
        # its skip_collision_robot_links (see ENV_CONFIG). Done once here — the
        # block isn't rescaled, so its body id is stable.
        self._disable_target_physics_collision()
        # NOTE: joint damping is applied by the base via JOINT_DAMPING (set
        # below) — no longer done inline here.

    def _disable_target_physics_collision(self) -> None:
        rid = self.splatsim_robot.sim_id
        block = next(
            (o for o in self.splatsim_objects if o.config.name == self.TARGET_OBJECT_NAME),
            None,
        )
        if block is None or block.sim_id is None:
            return
        n = self.pybullet_client.getNumJoints(rid)
        for link in range(-1, n):
            self.pybullet_client.setCollisionFilterPair(
                block.sim_id, rid, -1, link, enableCollision=0
            )

    def _resolve_goal_ee_target(self) -> Optional[tuple]:
        """Goal for the reset reachability check: reach the (moving) target
        block. Position-IK the EE to the block center (the block is a marker, so
        its center is a valid collision-free goal), FK that config to a
        concretely-reachable EE pose, and return it plus the config as the IK
        bias so the planner converges to the same branch."""
        block_pos, _ = self.get_current_object_pose(object_name=self.TARGET_OBJECT_NAME)
        rid = self.splatsim_robot.sim_id
        ee_link = self._get_ee_link_index()
        joints = list(range(1, self.num_dofs() + 1))

        saved = [self.pybullet_client.getJointState(rid, j)[0] for j in joints]
        ik = self.pybullet_client.calculateInverseKinematics(
            rid, ee_link, list(block_pos), maxNumIterations=200, residualThreshold=1e-6
        )
        q_goal = np.array(ik[: self.num_dofs()])
        for j, qi in zip(joints, q_goal):
            self.pybullet_client.resetJointState(rid, j, float(qi))
        ls = self.pybullet_client.getLinkState(rid, ee_link, computeForwardKinematics=True)
        ee_pos, ee_quat = np.array(ls[0]), np.array(ls[1])
        # Restore the pre-query joint state (the caller's start pose).
        for j, qi in zip(joints, saved):
            self.pybullet_client.resetJointState(rid, j, float(qi))
        return ee_pos, ee_quat, q_goal

    def _get_default_trajectory_gen_config(self) -> TrajectoryGenModeConfig:
        """Trajectory-gen config for the planar env. Mirrors Upright
        small_engine's override (feeds the resolved self-collision skip pairs —
        the same Robotiq gripper pairs, by name) and inherits the global default
        `self_collision_clearance` (1 cm).

        This env used to force `self_collision_clearance=0.0` because the shared
        `randomize_objects()` loop's `_get_random_collision_free_q` probe (no
        fixed RRT goal, unlike small_engine's goal-directed IK) flagged constant
        Robotiq gripper near-misses at 1 cm. Those pairs (outer↔inner knuckle
        ~2.4 mm AND outer_finger↔inner_knuckle ~5.7 mm) are now in the shared
        GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES, so 1 cm no longer false-flags and
        the override is gone — planar plans with the same clearance as everything
        else."""
        return TrajectoryGenModeConfig(
            self_collision_skip_pairs=[list(p) for p in self.SELF_COLLISION_SKIP_PAIRS] or None,
            # Equalize joint-space path speed across sections so the light arm
            # doesn't sprint-then-brake into the goal (the surge that the PD
            # controller tracks as overshoot/oscillation). Complements the
            # env's lowered CONTROL_MAX_VELOCITY + joint damping.
            uniform_path_speed=True,
            debug_visualize=False,
        )

    def randomize_ee_pose(self, max_attempts: int = 100) -> Optional[tuple]:
        """Arm-start hook consumed by the shared `randomize_objects()` loop.

        The UR envs sample random collision-free 6-DOF EE poses via IK (hardcoded
        EE link 6 + 6-DOF limits + Cartesian sampling over TABLE_LIMITS) — none
        of which applies to a 3-DOF planar arm. The planar-appropriate analogue
        is JOINT-space sampling: draw the 3 arm joints uniformly within their
        URDF limits and keep the first draw that's collision-free for the current
        object arrangement. Returns the full (q1,q2,q3, gripper=open) action, or
        None if no collision-free start is found in `max_attempts` (so
        `randomize_objects()` re-rolls the whole scene rather than committing to
        a colliding start)."""
        rid = self.splatsim_robot.sim_id
        # Arm DOFs are joints 1..num_dofs; read their limits from the URDF so
        # this stays correct if the joint ranges change.
        limits = []
        for j in range(1, self.num_dofs() + 1):
            info = self.pybullet_client.getJointInfo(rid, j)
            lo, hi = info[8], info[9]  # jointLowerLimit, jointUpperLimit
            limits.append((lo, hi))

        for _ in range(max_attempts):
            # (q1, q2, q3, gripper=0) — gripper open at start.
            action = tuple(np.random.uniform(lo, hi) for lo, hi in limits) + (0.0,)
            self.teleport_joint_state(self.splatsim_robot, action)
            if not self.is_robot_in_collision():
                # Persist as the episode's initial pose (mirrors the base env), so
                # get_env_config / demo recording reflect the actual start.
                if self.splatsim_robot.config.articulation_config is not None:
                    self.splatsim_robot.config.articulation_config.initial_joint_positions = list(action)
                return action
        return None

    def randomize_object_pose(self, splatsim_obj) -> None:
        """Place randomizable objects on an annulus around the base pivot,
        sampling the full circle so they span all four quadrants (see the
        PLACEMENT_* attributes). The robot and any fixed-pose object fall back to
        the base rectangular-range behavior (the robot stays put).

        Called per object by the shared `randomize_objects()` loop, which then
        checks object-object / robot-start collisions and re-rolls on overlap —
        so polar placement composes with the shared collision-free guarantee."""
        cfg = splatsim_obj.config
        if (
            splatsim_obj is self.splatsim_robot
            or not cfg.randomize_pose
            or splatsim_obj.sim_id is None
        ):
            return super().randomize_object_pose(splatsim_obj)

        r_min, r_max = self.PLACEMENT_RADIUS_RANGES.get(
            cfg.name, self.PLACEMENT_RADIUS_RANGE_DEFAULT
        )
        theta = np.random.uniform(0.0, 2.0 * np.pi)   # full circle -> all 4 quadrants
        r = np.random.uniform(r_min, r_max)
        cx, cz = self.PLACEMENT_CENTER_XZ
        pos = [cx + r * np.cos(theta), 0.0, cz + r * np.sin(theta)]  # y=0: planar
        quat = list(cfg.base_quat)

        self.pybullet_client.resetBasePositionAndOrientation(splatsim_obj.sim_id, pos, quat)
        cfg.current_position = list(pos)
        cfg.current_quat = list(quat)
        cfg.initial_position = list(pos)
        cfg.initial_quat = list(quat)

    def check_metrics(self) -> dict:
        """Planar reach metric: gripper within pos_tolerance_m of the target
        block. Keeps the same collision + action-smoothness fields as
        small_engine's check_metrics so downstream consumers see a familiar
        shape."""
        ee_pos, _ = self.get_current_ee_pose()
        # Target block is always in ENV_CONFIG; get_current_object_pose (base)
        # raises if it's missing, which is the right fail-loud behavior.
        target_pos, _ = self.get_current_object_pose(object_name=self.TARGET_OBJECT_NAME)
        pos_diff = float(np.linalg.norm(np.array(ee_pos) - np.array(target_pos)))
        success = pos_diff <= self.pos_tolerance_m

        # Robot-vs-scene collision. Uses the eval-terminate skip union (same as
        # small_engine) so the resolved gripper pairs are honored, AND the same
        # near-miss clearances (from --in_collision_*_clearance) so a near-miss
        # counts as a collision identically to small_engine — otherwise the
        # constructor clearances would be silently ignored here.
        in_collision, collision_kind = self.is_robot_in_collision(
            obstacle_clearance=self._in_collision_obstacle_clearance,
            self_collision_clearance=self._in_collision_self_collision_clearance,
            self_collision_skip_pairs=self._eval_terminate_skip_pairs(),
            return_kind=True,
        )
        _CK_CODE = {None: 0, "obstacle": 1, "self": 2}
        return {
            "is_success": success,
            "distance_to_target_m": pos_diff,
            "in_collision": in_collision,
            "collision_kind": collision_kind,
            "collision_kind_code": _CK_CODE[collision_kind],
            "action_delta": self._action_delta,
            "action_accel": self._action_accel,
            "action_jerk": self._action_jerk,
        }

    def get_env_config(self) -> Dict[str, Any]:
        """Extend the base's oracle env config with a DYNAMIC task goal.

        The base class serializes `self.ENV_CONFIG.task`, but planar's
        ENV_CONFIG.task is None because the target is the (randomized)
        block, not a fixed pose. Without a task goal the LeRobot SA
        wrapper's RRT source can't plan — every trigger fails with
        "no task.target_ee_pos" and future_chunk shield calls loop.
        Compute the goal on-the-fly from the current block pose using
        `_resolve_goal_ee_target` (already used for the reset reachability
        check), and inject it into the returned dict."""
        env_cfg = super().get_env_config()
        goal = self._resolve_goal_ee_target()
        if goal is not None:
            ee_pos, ee_quat, q_goal = goal
            # Keep the (possibly-existing) task fields from ENV_CONFIG.task
            # if any were declared statically; dynamic goal overrides just
            # the pose fields the RRT source needs.
            task = dict(env_cfg.get("task") or {})
            task["target_ee_pos"] = [float(x) for x in ee_pos]
            task["target_ee_quat"] = [float(x) for x in ee_quat]
            task["q_goal_bias"] = [float(x) for x in q_goal]
            env_cfg["task"] = task
        return env_cfg

    # ── Oracle state (privileged) ─────────────────────────────────────────────
    # Every planar recording carries a SEPARATE observation.environment_state
    # feature (FeatureType.ENV):
    #   [<block x,z>, <obstacle_i x,z>...]
    # i.e. exact object coords (goal = the block) for EVERY scene object in
    # ENV_CONFIG order, so an oracle policy sees the goal AND what to avoid.
    # observation.state stays pure proprioception [joint_1, joint_2, joint_3,
    # gripper]. This lets one dataset train both an image policy (ignores
    # environment_state) and a state-only policy (consumes it — a pure control
    # problem, no perception, so it trains fast). Only the X-Z axes are recorded
    # (planar plane; Y is a constant 0). The generic recording logic lives in the
    # base (oracle_environment_state / env_state_dim); this only picks the axes.
    ORACLE_STATE_COORD_INDICES = (0, 2)  # world x, z


def _planar_env_config(name: str, obstacle_count: int) -> EnvConfig:
    """Build a planar reach EnvConfig: a floating target block + `obstacle_count`
    floating obstacles. obstacle_count=0 is the simplest debug scene (reach
    only). Object placement is POLAR (see PLACEMENT_* / randomize_object_pose)."""
    objects = [
        # Floating, non-colliding target MARKER (see the class docstring notes).
        CuboidObjectConfig(
            name="block", size=(0.05, 0.05, 0.05), randomize_pose=True, mass=0,
            color_rgb=(80, 120, 200), load_splat=False,
            skip_collision_robot_links=list(range(20)),
        ),
    ]
    for i in range(1, obstacle_count + 1):
        objects.append(CuboidObjectConfig(
            name=f"obstacle_{i}", size=(0.07, 0.07, 0.07), randomize_pose=True,
            mass=0, color_rgb=(200, 90, 70), load_splat=False,
        ))
    return EnvConfig(name=name, terminate_on_collision=False, objects=objects)


class Planar3JointPybulletRobotServer(PlanarPybulletRobotServer):
    """Concrete planar reach env: a 3-joint arm in the vertical X-Z plane, a
    floating target block, and two floating obstacles that may or may not block
    the path between the arm and the block. All oracle-info, no rendering."""

    DEFAULT_ROBOT_NAME = "planar_3joint"
    # No splat rendering, so no background splat is needed.
    background_splat_name = None

    # Block (non-colliding reach target/marker) + 2 solid obstacles that may or
    # may not block the arm→block corridor. See _planar_env_config.
    ENV_CONFIG = _planar_env_config("planar_3joint", obstacle_count=2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Side-on debug camera looking along -Y at the X-Z operating plane,
        # centered on the base pivot so all four quadrants are in view. Only
        # meaningful with a GUI window — in headless (DIRECT) mode there's no
        # debug visualizer, so skip it.
        if not self._headless:
            self.pybullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.9,
                cameraYaw=0,        # look along -Y toward the X-Z plane
                cameraPitch=-5,
                cameraTargetPosition=[0.0, 0.0, 0.2],
            )


class Planar3JointOraclePybulletRobotServer(Planar3JointPybulletRobotServer):
    """Oracle-state variant: same 2-obstacle reach scene, but a separate
    observation.environment_state carries [block(x,z), obstacle_1(x,z),
    obstacle_2(x,z)] — exact coords, no perception — alongside proprioceptive
    observation.state [joints, gripper]. For a state-only diffusion policy (no
    image) that trains fast; use it to isolate control from vision."""

    DEFAULT_ROBOT_NAME = "planar_3joint"  # same URDF/objects.yaml entry
    ENV_CONFIG = _planar_env_config("planar_3joint_oracle", obstacle_count=2)


class Planar3JointOracleSimplePybulletRobotServer(Planar3JointOraclePybulletRobotServer):
    """Simplest debug scene: oracle state, ZERO obstacles (pure reach).
    observation.state = [joints, gripper]; observation.environment_state =
    [block(x,z)]. Should train to ~100% in a few thousand steps — the green
    baseline that confirms the pipeline/action space are correct."""

    ENV_CONFIG = _planar_env_config("planar_3joint_oracle_simple", obstacle_count=0)
