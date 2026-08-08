import dataclasses
import random
import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import zmq

from splatsim.configs import (
    EnvConfig,
    TaskConfig,
    CuboidObjectConfig,
    SplatObjectConfig,
    TrajectoryGenModeConfig,
)
from splatsim.robots.sim_robot_pybullet_base import (
    PybulletRobotServerBase,
)

from splatsim.utils.rrt_path_utils import compute_camera_alignment_score

class SmallEnginePybulletRobotServer(PybulletRobotServerBase):
    # To fill in with subclasses
    ENV_CONFIG: EnvConfig

    # Mild per-joint viscous damping for the UR5 arm (URDF declares none). Real
    # servo-controlled joints have friction; the heavy UR5 is already near
    # critically damped, so a small value suffices (vs. the light planar arm's
    # 2.0). Applied by the base's `_apply_joint_damping()`. Calibrate to the
    # real UR5's settling response; note nonzero damping changes recorded
    # dynamics vs. datasets captured at damping=0.
    JOINT_DAMPING = 0.5

    # Rigid gripper mimic (see the base attr): the small-engine tasks are
    # approach/touch tasks — the gripper never closes on an object, so
    # capping finger crush force is irrelevant here, while the historical
    # gear-only coupling let obstacle contact visibly bend a single finger
    # 50+ deg (the real 2F-85 is structurally rigid and never does this).
    GRIPPER_MIMIC_HOLD_FORCE = 500.0

    # UR URDF: base_link(0) and upper_arm_link(2) are non-adjacent (separated
    # by shoulder_link(1)) but the shoulder bracket places upper_arm_link's
    # lower face ~4 mm above base_link's top face. Any self_collision_clearance
    # > 0.004 m flags this as a self-collision at every valid joint config,
    # which breaks IK + RRT planning. Excluded here so callers can crank up
    # the threshold without false positives.
    #
    # SINGLE SOURCE OF TRUTH: this class attribute is consumed by THREE
    # downstream paths:
    #   1. `is_robot_in_collision` (env-side termination check on collision).
    #   2. `_get_default_trajectory_gen_config` → trajectory generator (GUI).
    #   3. `get_env_config()` → dispatched to LeRobot SA wrapper over ZMQ
    #      → SA wrapper's RRT planner.
    # Changing this list updates all three; the user does not have to pass
    # `--rrt_self_collision_skip_pairs` separately on the DAgger CLI.
    #
    # If you ever swap robots, audit this list against the new URDF's link
    # adjacency + AABB layout. Other envs (object_on_plate, assembly, ...)
    # inherit the base class's empty default — add their own override here
    # if their URDF needs it.
    # This list was PRUNED from 40 entries to 17 based on the audit at
    # `my_scripts/audit_self_collision_skip_pairs.py` (10 000 uniform-random
    # joint configs + 467 workload-representative configs from
    # `splatsim_approach_lever_12_clean/data/chunk-000/file-000.parquet`,
    # mimic-joint-aware sampling).
    #
    # The dropped 23 entries fell into two audit classes:
    #   * CRITICAL_MUST_UNSKIP (3 pairs): (0,2), (3,5), (4,19). The comments
    #     called these URDF-fixed but 22-35% of sampled configs actually had
    #     the pair penetrating by 22-41 mm. The planner was silently
    #     accepting configs where physics disagrees → runtime PyBullet
    #     solver kicks the joint state to un-penetrate → recorded joint
    #     spike (matched the `joint_spike` anomaly class flagged by
    #     `dagger_detect_dataset_anomalies.py`).
    #   * REDUNDANT_UNSKIP_OK (19 pairs): (4,7), (4,8), (4,12), (4,13),
    #     (4,17), (4,18), (5,10), (5,11), (5,12), (5,15), (5,16), (5,17),
    #     (5,19), (6,8), (6,10), (6,11), (6,12), (6,15), (6,16), (6,17).
    #     All wrist_1/2/3 ↔ some-finger/knuckle-part pairs where the min
    #     distance across every sampled config stays above 20 mm — they
    #     never come near the runtime `self_collision_clearance` buffer,
    #     so skipping them had no effect. Removed for clarity; behavior
    #     unchanged.
    #
    # The 17 kept below are the STRUCTURAL_KEEP class (range < 1 mm across
    # 10 467 configs) plus (4,6) which sits at a constant 11-13 mm — a
    # BORDERLINE case that stays below 20 mm but has never penetrated.
    #
    # If you ever swap robots or change the URDF, re-run
    # `my_scripts/audit_self_collision_skip_pairs.py --urdf <new.urdf>`
    # and update this list from the audit output.
    SELF_COLLISION_SKIP_PAIRS = [
        # ---- UR wrist_1 vs wrist_3 URDF floor (~12 mm constant) ----
        # BORDERLINE per audit: min 11.44 mm, range 1.18 mm, 0% penetrated.
        # Compact UR wrist geometry pins these ~12 mm apart regardless of
        # arm articulation. Keeping so runtime self_collision_clearance
        # thresholds > 11 mm don't trip on every config.
        (4, 6),
        # ---- CRITICAL_MUST_UNSKIP pairs from the audit — re-added ----
        # The audit flagged these as pairs that ACTUALLY penetrate in the
        # sampled distribution (22-35% of workload configs, max 22-41 mm
        # interpenetration per `getClosestPoints`). Kinematically that's a
        # collision, but PyBullet's constraint solver tolerates the
        # interpenetration without forces (URDF joint constraints hold, no
        # observable solver kick). These are mesh-overlap artifacts, not
        # dynamics events.
        #
        # SELF_COLLISION_SKIP_PAIRS is the SINGLE source of truth for THREE
        # downstream paths (see the class-level comment above), and the
        # env-side eval-terminate check `is_robot_in_collision` (called
        # when --env.terminate_on_collision=true) reads this list too. When
        # the audit removed these three pairs, the eval-terminate began
        # firing on 60-70% of workload configs — the same URDF-artifact
        # overlaps the physics engine ignores. Restored here so RRT
        # planning AND eval-terminate share the same "this pair is a URDF
        # artifact, ignore it" contract. Trade-off: RRT paths may pass
        # through these penetrations, but that's what pre-audit behavior
        # was and it never caused issues (the joint-spike case that
        # motivated the audit was actually (3, 19) forearm ↔ wrist_camera,
        # which was NEVER in the skip list and still isn't).
        (0, 2),   # base_link vs upper_arm_link — mesh artifact at shoulder.
                  # base has no dynamics that can be kicked; RRT tolerating this
                  # penetration hasn't produced joint spikes.
        (3, 5),   # forearm_link vs wrist_2_link — URDF mesh floor at
                  # elbow-wrist junction (~12 mm). No solver kicks observed.
        #
        # NOT in STRICT list — see SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA
        # below for the pairs eval-terminate silently accepts but RRT still
        # rejects (wrist-camera / wrist_1 mesh overlaps that produce solver
        # kicks in traj-gen paths).
        # ---- Wrist camera rigidly downstream of wrist_3 ----
        # Fixed sensor attachment via ee_link. Distances URDF-determined
        # for these two pairs (audit: (6,19) 26.80 mm constant; (7,19)
        # 65.76 mm constant).
        (6, 19),  # wrist_3 vs wrist_camera_link
        (7, 19),  # ee_link vs wrist_camera_link
        # ---- Wrist_2 → ee_link / gripper_base (rigid via ee_link) ----
        # Audit: constant 18.80 / 42.11 mm across all configs.
        (5, 7),   # wrist_2 vs ee_link
        (5, 8),   # wrist_2 vs robotiq_arg2f_base_link
        # ---- Wrist_2 → inner/outer knuckles (rigid downstream) ----
        # Audit: constant 91-100 mm across all configs.
        (5, 9),   # wrist_2 vs left_outer_knuckle
        (5, 13),  # wrist_2 vs left_inner_knuckle
        (5, 14),  # wrist_2 vs right_outer_knuckle
        (5, 18),  # wrist_2 vs right_inner_knuckle
        # ---- Wrist_3 → inner/outer knuckles (rigid downstream) ----
        # Audit: constant 57-66 mm across all configs.
        (6, 9),   # wrist_3 vs left_outer_knuckle
        (6, 13),  # wrist_3 vs left_inner_knuckle
        (6, 14),  # wrist_3 vs right_outer_knuckle
        (6, 18),  # wrist_3 vs right_inner_knuckle
        # ---- Gripper-internal URDF mesh overlaps (Robotiq 2F-85 design) ----
        # No longer listed here by index. The Robotiq gripper is shared across
        # robots, so its skip pairs live in the base class's
        # GRIPPER_SELF_COLLISION_SKIP_PAIR_NAMES (by link name) and are resolved
        # + merged into SELF_COLLISION_SKIP_PAIRS at construction. For this UR5
        # they resolve to exactly the old hardcoded indices —
        # (11,13),(12,13),(16,18),(17,18) — so this env's skippable set is
        # UNCHANGED, but the planar debug env now shares the identical
        # definition. See _init_self_collision_skip_pairs.
    ]

    # Additional pairs skipped ONLY by the env-side eval-terminate check
    # (`is_robot_in_collision` called from `check_metrics`) — NOT by the RRT
    # planner, trajectory generator, controller-side collision predicates,
    # or the reset-time "find collision-free start pose" scan.
    #
    # Rationale: these three wrist-camera / wrist_1 mesh-overlap pairs
    # kinematically penetrate at some workload configs (audit flags them
    # CRITICAL_MUST_UNSKIP), AND PyBullet's constraint solver DOES kick them
    # when RRT paths pass through those configs — producing the "teleport +
    # trailing joint spike" pattern in trajectory-gen recordings. So the
    # planner must reject them (kept out of SELF_COLLISION_SKIP_PAIRS
    # above). But the ENV shouldn't terminate on them either — they're
    # URDF-mesh artifacts, and if the physics happens to land on one
    # briefly it shouldn't cost the episode.
    #
    # Consumed exclusively by `_eval_terminate_skip_pairs()` (base class),
    # which unions this with SELF_COLLISION_SKIP_PAIRS. `check_metrics`
    # below passes the union to `is_robot_in_collision`.
    SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA = [
        (2, 4),   # upper_arm ↔ wrist_1 — mesh overlap in extreme arm-curl configs
        (3, 19),  # forearm ↔ wrist_camera_link — original joint-spike pair
        (4, 19),  # wrist_1 ↔ wrist_camera_link — same wrist-cam mesh class
    ]

    def __init__(
        self,
        in_collision_obstacle_clearance: float = 0.005,
        in_collision_self_collision_clearance: float = 0.005,
        **kwargs,
    ):
        # Near-miss clearances used by check_metrics()'s is_robot_in_collision()
        # when reporting `in_collision`. Default 5 mm (bumped from historical
        # 0.0 = penetration-only) because PyBullet's constraint solver holds
        # rigid bodies at ~0 mm gap under contact force — a link PRESSED against
        # an obstacle stays a rounding hair above zero penetration, so the old
        # penetration-only check silently missed "arm slid into obstacle" and
        # "arm folded onto own body" cases. 5 mm matches the SA wrapper's
        # `rrt_self_collision_clearance=0.005` default so env-terminate agrees
        # with what the planner treats as a collision. Lives on THIS shared base
        # so every SmallEngine-derived env — the UR5 concrete classes AND the
        # planar debug arm — accepts --in_collision_*_clearance identically
        # (check_metrics here already reads these attributes). Callers who
        # NEED penetration-only semantics (legacy scripts, unit tests) can
        # still pass `--in_collision_obstacle_clearance=0` to restore.
        self._in_collision_obstacle_clearance = float(in_collision_obstacle_clearance)
        self._in_collision_self_collision_clearance = float(in_collision_self_collision_clearance)
        super().__init__(**kwargs)

    def plan_given_this_state(self, initial_joint_positions):
        all_paths = []
        return all_paths

    def serve_loop(self) -> None:
        pass

    def _resolve_goal_ee_target(self):
        """Goal for the reset-time reachability check: the fixed task EE pose.

        Upgrades this env's reset from 'a goal CONFIG exists' to 'a full RRT
        PATH from the random start reaches the goal' (see base
        `_check_scenario_solvable`). Returns None if no task pose is configured,
        which falls back to the legacy `check_able_to_solve`."""
        task = self.ENV_CONFIG.task
        if task is None or task.target_ee_pos is None or task.target_ee_quat is None:
            return None
        bias = list(task.q_goal_bias) if task.q_goal_bias is not None else None
        return (
            np.asarray(task.target_ee_pos, dtype=np.float64),
            np.asarray(task.target_ee_quat, dtype=np.float64),
            bias,
        )

    # =========================================================================
    # Gym Environment Interface
    # =========================================================================

    def _reset_episode_state(self):
        self._step_count = 0
        self._episode_started = True
        # Per-episode shadow-strength jitter (base class; no-op visual change
        # unless splat_shadows is on). Runs after reset() seeds np.random.
        self._resample_splat_shadow_strength()
        self._prev_action = None
        self._action_delta = 0.0
        self._prev_action_delta = 0.0
        self._action_accel = 0.0
        self._prev_action_accel = 0.0
        self._action_jerk = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict

        Returns:
            observation: Initial observation dict
            info: Dict with initial info
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._reset_episode_state()

        # A pinned scenario (launch_nodes --scenario_file) replaces the
        # randomize/solvability search: that search can burn 100 attempts,
        # each running goal IK, which is minutes on a scene where the goal is
        # hard to solve. Reusing a known-good arrangement makes launch and
        # interactive/eval resets deterministic and fast.
        #
        # NOT during trajectory GENERATION though: randomize_objects() is
        # what varies the box arrangement AND the robot's start joints per
        # episode — honoring the pin there froze every recorded episode to
        # one identical scene + start pose (zero dataset diversity; observed
        # as "reset never randomizes anything"). Generation-mode resets
        # therefore always randomize; the pin keeps serving the launch-time
        # setup and non-generation resets. An explicit
        # options={"force_randomize": True} bypasses the pin from ANY mode —
        # the GUI "Reset Env" button sends it, since a user pressing Reset
        # is asking for a fresh scene, not a replay of the pinned one.
        generating = self.serve_mode in (
            self.SERVE_MODES.GENERATE_TRAJECTORIES,
            self.SERVE_MODES.GENERATE_DEMOS,
        )
        force_randomize = bool(options.get("force_randomize")) if options else False
        if (
            getattr(self, "_pinned_scenario", None) is not None
            and not generating
            and not force_randomize
        ):
            self.apply_scenario(self._pinned_scenario)
        else:
            # This now also randomizes the robot's joints
            self.randomize_objects()

        # From GENERATE_DEMOS: randomize_ee_pose()
        # initial_joints = self.randomize_ee_pose()
        # self.teleport_joint_state(self.splatsim_robot, initial_joints)
        # (open_gripper is now unconditional — see below.)

        # Explicitly open the gripper on every reset. Constructor-time
        # setup (`sim_robot_pybullet_base.py`, ~line 881) calls this once
        # at env init, but nothing else in the reset flow touches the
        # gripper joints — `randomize_objects` and `randomize_ee_pose`
        # only reset arm joints (indices 1..num_dofs), and
        # `is_robot_in_collision` calls check_links_in_collision(q=None)
        # which doesn't invoke `set_robot_joint_positions`'s side-effect
        # `open_gripper`. Without this call, whichever gripper state the
        # PRIOR episode ended at (e.g. closed=1.0 after a grasp) bleeds
        # into the next episode's frame 0 — then trajectory playback
        # commands gripper=0 partway through, and the recorded state
        # drifts 1.0 → 0.0 mid-episode. Manifests in
        # dagger_detect_dataset_anomalies as GRIPPER_DRIFT with range
        # [0.0000, 1.0000] on ~1/6 of episodes.
        self.open_gripper()

        # # Let simulation settle
        for _ in range(1000):
            self.pybullet_client.stepSimulation()

        metrics = self.check_metrics()

        info = {"is_success": metrics['is_success'], **metrics}

        return self.get_observations(), info

    def _physics_step(self, action: np.ndarray) -> None:
        """Track action smoothness metrics then advance physics."""
        if self._prev_action is not None:
            self._action_delta = np.linalg.norm(np.array(action) - self._prev_action)
        else:
            self._action_delta = 0.0

        self._action_accel = np.abs(self._action_delta - self._prev_action_delta)
        self._action_jerk = np.abs(self._action_accel - self._prev_action_accel)

        self._prev_action = np.array(action)
        self._prev_action_delta = self._action_delta
        self._prev_action_accel = self._action_accel

        super()._physics_step(action)

    def compute_reward_from_metrics(self, metrics: dict) -> float:
        return 1.0 if metrics['is_success'] else 0.0

    def check_terminated_from_metrics(self, metrics: dict) -> bool:
        if self.ENV_CONFIG.terminate_on_collision:
            return metrics['is_success'] or metrics['in_collision']
        return metrics['is_success']

    def check_metrics(self) -> dict:
        """Check if the task goal is achieved.

        Returns True if the end effector is within pos_tolerance_m (meters) and
        quat_tolerance_deg (degrees) of the target pose.
        """
        assert self.ENV_CONFIG.task is not None, "SmallEngine env requires a task config"
        task_config = self.ENV_CONFIG.task
        target_ee_pos = task_config.target_ee_pos
        target_ee_quat = task_config.target_ee_quat
        pos_tolerance_m = task_config.pos_tolerance_m
        quat_tolerance_deg = task_config.quat_tolerance_deg

        success = True

        pos, quat = self.get_current_ee_pose()

        # Check position distance
        pos_diff = np.linalg.norm(np.array(pos) - np.array(target_ee_pos))
        # print(f"Position difference: {pos_diff:.4f} m (tolerance: {pos_tolerance_m:.4f} m)")
        if pos_diff > pos_tolerance_m:
            success = False

        # Check quaternion distance (angle between orientations)
        # Quaternion dot product gives cos(theta/2) where theta is the rotation angle
        q1 = np.array(quat)
        q2 = np.array(target_ee_quat)
        dot = np.abs(np.dot(q1, q2))  # abs handles q and -q representing same rotation
        dot = np.clip(dot, -1.0, 1.0)  # Numerical stability
        angle_rad = 2 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        # print(f"Orientation difference: {angle_deg:.2f} deg (tolerance: {quat_tolerance_deg:.2f} deg)")

        if angle_deg > quat_tolerance_deg:
            success = False

        cam_position, cam_rotation = self.get_wrist_camera_transform()
        # Camera forward direction (assumes +Z axis in local frame)
        cam_forward = cam_rotation[:, 2]
        cam_looks_at_goal_score = compute_camera_alignment_score(cam_position, cam_forward, target_ee_pos)

        # Capture both the bool AND the collision kind ("obstacle" / "self" /
        # None) in one call so the metrics dict can surface the cause to
        # downstream consumers (eval_info.json per-episode data → DAgger
        # plots / failure-mode analysis). check_links_in_collision short-
        # circuits on the first match, so the cost is the same as the
        # bool-only path.
        # Use the WIDER eval-terminate skip list (strict RRT list ∪
        # SELF_COLLISION_SKIP_PAIRS_EVAL_TERMINATE_EXTRA) — see the class
        # attribute's docstring for the wrist-camera-mesh rationale.
        # Cached once per call so both the initial + verbose re-check use
        # the identical set.
        _eval_terminate_skips = self._eval_terminate_skip_pairs()
        in_collision, collision_kind = self.is_robot_in_collision(
            obstacle_clearance=self._in_collision_obstacle_clearance,
            self_collision_clearance=self._in_collision_self_collision_clearance,
            self_collision_skip_pairs=_eval_terminate_skips,
            return_kind=True,
        )
        if in_collision:
            success = False
            # Re-check with verbose=True so the per-step log ALSO records the
            # specific link pair (the bool/kind tuple above doesn't carry
            # the offending link names). Cheap: PyBullet's pairwise query is
            # the same work as the initial check; we just print the first
            # match this time. Note: check_metrics is called every physics
            # step, so once a collision is detected (and the env terminates
            # via check_terminated_from_metrics), this branch fires once at
            # the moment of detection and then check_metrics stops being
            # called for that episode — no per-step log spam.
            self.is_robot_in_collision(
                obstacle_clearance=self._in_collision_obstacle_clearance,
                self_collision_clearance=self._in_collision_self_collision_clearance,
                self_collision_skip_pairs=_eval_terminate_skips,
                verbose=True,
            )

        # Numeric encoding of collision_kind so SyncVectorEnv can stack it
        # into an ndarray of shape (num_envs,) and lerobot-eval's
        # info-metrics aggregation (torch.from_numpy + reduce ops) can
        # consume it. The string form below is for verbose logs / direct
        # readers — gets silently filtered out by lerobot-eval's
        # "skip non-numeric" guard so the numeric path is what reaches
        # eval_info.json.
        #   0 = no collision
        #   1 = obstacle (robot link vs scene obstacle)
        #   2 = self    (robot link vs robot link, non-adjacent pair)
        _CK_CODE = {None: 0, "obstacle": 1, "self": 2}
        collision_kind_code = _CK_CODE[collision_kind]
        metrics = {
            "is_success": success,
            "position_error_m": pos_diff,
            "orientation_error_deg": angle_deg,
            "cam_looks_at_goal_score": cam_looks_at_goal_score,
            "action_delta": self._action_delta,
            "action_accel": self._action_accel,
            "action_jerk": self._action_jerk,
            "in_collision": in_collision,
            # collision_kind is "obstacle" / "self" / None — informational
            # string for direct callers (verbose log readers, custom
            # scripts that load the metrics dict). The numeric *_code
            # field below is what flows through lerobot-eval to
            # eval_info.json (see the encoding comment above the
            # _CK_CODE dict).
            "collision_kind": collision_kind,
            "collision_kind_code": collision_kind_code,
        }

        return metrics


class UprightRobotSmallEngineNewPybulletRobotServer(SmallEnginePybulletRobotServer):
    # This new lab bench scene has the robot rotated 90 degrees because it was installed rotated D:
    # background_splat_name = "robot_iphone_w_engine_new"
    #
    # DEFAULT_ROBOT_NAME is the SINGLE SOURCE OF TRUTH for this env's splat
    # / URDF identifier. Both the SplatSim server (via `launch_nodes.py`'s
    # class-default lookup) and LeRobot-side clients (which mirror this
    # string in their own defaults or query it via `get_env_config()` over
    # ZMQ) key off this one attribute. `background_splat_name` derives from
    # it because for this env the robot + background come from the same
    # Gaussian training session — one canonical splat, one coordinate
    # frame, no risk of the pair drifting apart in defaults.
    DEFAULT_ROBOT_NAME = "robot_iphone_w_engine_curtain"
    background_splat_name = DEFAULT_ROBOT_NAME
    # base_camera_splat_name = "robot_iphone_w_engine_new"

    # PyBullet-camera pose for the "pybullet" render mode (dropdown). The base
    # default frames the PLANAR env from the -Y side, which for THIS scene sits
    # behind the wall (y=-0.225) and renders only the wall. Reuse this env's
    # tuned debug-camera framing (orbit form) so the third-person view actually
    # shows the robot + engine + table. NOTE: this camera pose lives on the
    # concrete class (not the shared SmallEngine base) so it doesn't override
    # the planar env's own eye/target pose.
    PYBULLET_CAMERA_TARGET = (0.0, 0.0, 0.3)
    PYBULLET_CAMERA_DISTANCE = 2.0
    PYBULLET_CAMERA_YAW = 180.0
    PYBULLET_CAMERA_PITCH = -30.0

    # ── observation.environment_state layout: 7-wide ─────────────────────────
    #   [box1(x,y), box2(x,y), ee(x,y,z)]
    # Only the RANDOMIZED objects are recorded — engine/table/wall are pinned
    # (randomize_pose=False, fixed ranges), so their coords were 9 constant
    # dims of the historical 15-wide layout carrying zero information (and
    # degenerate min==max normalization stats). The boxes sit ON the table so
    # their z is constant → (x,y) only; the EE moves in full 3D → (x,y,z) via
    # ORACLE_STATE_EE_COORD_INDICES. Mirrors the planar recipe (EE appended
    # after objects; see planar's 8-wide [block,obs1,obs2,ee](x,z)).
    # HISTORY: pre-2026-08-04 recordings (e.g. approach_lever_13_smooth) are
    # 15-wide [engine,table,wall,box1,box2](x,y,z) with NO EE — checkpoints
    # and datasets are NOT width-compatible across this change; re-record or
    # migrate (slice box coords + FK-append EE).
    ORACLE_OBJECT_NAMES = ["box1", "box2"]
    ORACLE_STATE_COORD_INDICES = (0, 1)      # boxes: world x, y (z = table height)
    ORACLE_STATE_INCLUDE_EE_POS = True
    ORACLE_STATE_EE_COORD_INDICES = (0, 1, 2)  # EE: full 3D

    ENV_CONFIG = EnvConfig(
        # name="upright_robot_small_engine_new",
        name="upright_robot_small_engine_curtain",
        task=TaskConfig(
            task_description="<control_mode> joint <control_mode>",

            # Approach lever — canonical goal joint config (6-DOF, no gripper).
            # # Used to seed IK so demos converge to a shared joint configuration.
            # q_goal_bias=(1.33936567, -1.52838483, 1.92282924, -1.21754169, -0.53407075, -0.73042029),
            # # target_ee_{pos,quat} were captured from self.get_current_ee_pose() at this q_goal_bias.
            # target_ee_pos=(-0.10123532289544344, 0.5484031509107826, 0.26692192875731213),
            # # If camera is tilted 18 degrees down
            # target_ee_quat=(0.8074376258351692, 0.1106042613918073, -0.5450490313370774, 0.19680632913133583),
            # # If camera is level with the horizon
            # # target_ee_quat=(0.8282820040827756, 0.02399455087684049, -0.5556401809874196, 0.06809678783251895),

            # Moving goal a bit further from the engine
            # Used to seed IK so demos converge to a shared joint configuration.
            q_goal_bias=(1.223, -1.587, 2.082, -0.925, -0.496, -1.124),
            # target_ee_{pos,quat} were captured from self.get_current_ee_pose() at this q_goal_bias.
            target_ee_pos=(-0.04133536080021404, 0.48531107901173687, 0.2357459331089757),
            # If camera is tilted 18 degrees down
            target_ee_quat=(0.7557648104558216, 0.11587548726295122, -0.6202265575111952, 0.175246797648522),


            pos_tolerance_m=0.03,  # 3 centimeters
            quat_tolerance_deg=10.0,  # 10 degrees
        ),
        objects=[
            SplatObjectConfig(
                name="small_engine_new",
                splat_name="small_engine_new",
                grasp_configs=[],
                randomize_pose=False,
                rotation_range_z=(0, 0),
                load_splat=False, # Because it's already in the scene splat
                position_range_x=(-0.48, -0.48),
                position_range_y=(0.36, 0.36),
                base_quat=(0, 0, -0.7071068, 0.7071068),
            ),
            # table has a plane for objects to sit on at z = 0
            CuboidObjectConfig(
                name="table",
                size=(1.5, 1.0, 0.05),
                # size=(1.5, 0.90, 0.05),
                randomize_pose=False,
                position_range_x=(0, 0),
                position_range_y=(0.3, 0.3),
                # position_range_y=(0.25, 0.25),
                position_range_z=(-0.025, -0.025),
                mass=0,
                color_rgb=(223, 205, 192),
                load_splat=False,
                skip_collision_robot_links=[0],  # Robot is mounted on the table; shoulder_link (link 0) is always within 1cm of the table surface
            ),
            # # wall is at -0.2 on y axis
            CuboidObjectConfig(
                name="wall",
                size=(3.0, 0.05, 1.5),
                randomize_pose=False,
                position_range_x=(0, 0),
                position_range_y=(-0.225, -0.225),
                position_range_z=(0.75, 0.75),
                # position=(0, -0.225, 0.75),
                mass=0,
                color_rgb=(223, 205, 192),
                load_splat=False,
            ),
            SplatObjectConfig(
                name="box1",
                splat_name="thinkpad_box",
                grasp_configs=[],
                randomize_pose=True,
                rotation_range_z=(0, 0),

                # Parallel boxes
                # position_range_x=(-0.2, 0.5),
                # position_range_y=(0.15, 0.3),
                # base_quat=(0, 0, 0, 1),

                # boxes at 90 degree angle
                position_range_x=(0.15, 0.5),
                position_range_y=(0.3, 0.5),
                base_quat=(0, 0, 0.707, 0.707),

                scaling_range_x=(0.9, 1.1),
                scaling_range_y=(0.9, 1.1),
                scaling_range_z=(0.9, 1.1),

                use_aabb_collision=True, # Box is axis-aligned, so AABB is exact and faster than PyBullet collision checks
            ),
            SplatObjectConfig(
                name="box2",
                splat_name="starwars_box",
                grasp_configs=[],
                randomize_pose=True,
                rotation_range_z=(0, 0),

                # Parallel boxes
                # position_range_x=(-0.2, 0.5),
                # position_range_y=(0.5, 0.7),
                # base_quat=(0, 0, 1, 0), #rotated 180 degrees about z

                # Boxes at 90 degree angle
                position_range_x=(-0.4, 0.3),
                position_range_y=(0.6, 0.8),
                base_quat=(0, 0, 1, 0), #rotated 180 degrees about z

                scaling_range_x=(0.9, 1.1),
                scaling_range_y=(0.9, 1.1),
                scaling_range_z=(0.9, 1.1),

                use_aabb_collision=True, # Box is axis-aligned, so AABB is exact and faster than PyBullet collision checks
            ),
        ],
    )

    def __init__(self, **kwargs):
        # in_collision_*_clearance is handled by SmallEnginePybulletRobotServer
        # (shared with the planar env); it flows through **kwargs to super().
        super().__init__(**kwargs)
        # Set initial camera position on the opposite side of the wall (positive y side)
        # Camera looks at the origin from the positive y side, above the floor
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=2.0,      # Distance from target
            cameraYaw=180,             # 0 degrees = looking from +y towards origin
            cameraPitch=-30,         # -30 degrees = looking down at ~30 degree angle
            cameraTargetPosition=[0, 0, 0.3]  # Look at point above the floor
        )

    def _get_default_trajectory_gen_config(self) -> TrajectoryGenModeConfig:
        assert self.ENV_CONFIG.task is not None, "SmallEngine env requires a task config"
        assert self.ENV_CONFIG.task.target_ee_pos is not None, "SmallEngine task config requires target_ee_pos"
        assert self.ENV_CONFIG.task.target_ee_quat is not None, "SmallEngine task config requires target_ee_quat"
        return TrajectoryGenModeConfig(
            ee_pos_goal=list(self.ENV_CONFIG.task.target_ee_pos),
            ee_quat_goal=list(self.ENV_CONFIG.task.target_ee_quat),
            q_goal_bias=(
                list(self.ENV_CONFIG.task.q_goal_bias)
                if self.ENV_CONFIG.task.q_goal_bias is not None else None
            ),
            # Read from the env class's single source of truth so changing
            # `SELF_COLLISION_SKIP_PAIRS` updates the trajectory generator
            # automatically. Convert tuples → lists for draccus/JSON
            # round-trip friendliness (TrajectoryGenModeConfig declares
            # this as List[List[int]]).
            self_collision_skip_pairs=[list(p) for p in self.SELF_COLLISION_SKIP_PAIRS] or None,
            debug_visualize=False
        )


class UprightRobotSmallEngineNewStrictPybulletRobotServer(UprightRobotSmallEngineNewPybulletRobotServer):
    """Tighter success-tolerance variant of the upright_small_engine_new task.

    Used for DAgger-style intervention recording (`my_scripts/intervention_record.py`)
    so the loose eval-time success threshold doesn't cut off precise RRT
    corrections before they reach the exact goal pose. The goal pose itself
    (q_goal_bias / target_ee_pos / target_ee_quat) is unchanged; only the
    `is_success` thresholds are tightened.
    """

    assert UprightRobotSmallEngineNewPybulletRobotServer.ENV_CONFIG.task is not None
    ENV_CONFIG = dataclasses.replace(
        UprightRobotSmallEngineNewPybulletRobotServer.ENV_CONFIG,
        task=dataclasses.replace(
            UprightRobotSmallEngineNewPybulletRobotServer.ENV_CONFIG.task,
            pos_tolerance_m=0.005,    # 5 mm (vs. 30 mm)
            quat_tolerance_deg=2.0,   # 2 deg (vs. 10 deg)
        ),
    )
