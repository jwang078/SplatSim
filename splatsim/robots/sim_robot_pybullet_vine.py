"""Vine grape-reach environment.

Extends the small-engine env the same way the planar env does: same UR5
(`robot_iphone_w_engine_curtain` — sisbot.urdf at its usual base position),
but the scene is the real scanned grape vine at the origin. The vine's
splat->sim transform is baked into its collision URDF (collision_frame: sim
in its scene.yaml), so the object loads at identity.

Task: reach an end-effector pose `GRAPE_STANDOFF_M` short of a grape bunch
(bunch clusters from the build's grape_targets.json under data/scenes/, produced by
the segmentation pipeline). The env also publishes a `soft_cost` payload in
its oracle env config so the RRT planner runs cost-aware over the foliage
(hard trunk mesh stays a binary obstacle).

Class attrs are the knobs; subclass or edit to retarget:
  TARGET_BUNCH_INDEX   which bunch the task aims at; None = random per reset
  GRAPE_STANDOFF_M     how close the gripper should get
  GRAPE_TARGETS_JSON / SOFT_COST_NPZ / VINE_SPLAT_NAME  asset locations
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from splatsim.configs import scene_registry
from splatsim.configs.env_config import EnvConfig, SplatObjectConfig, TaskConfig
from splatsim.robots.sim_robot_pybullet_small_engine import (
    SmallEnginePybulletRobotServer,
)
from splatsim.utils import grape_targets

logger = logging.getLogger(__name__)


# Same UR5 mount as the small-engine env (objects.yaml base_position).
_ROBOT_BASE_POS = np.array([0.0, 0.0, -0.088])


def _build_frame(vine_splat_name: str):
    """(collision_frame, splat->sim 4x4 or None) for a registered segmentation
    build. Everything the build produced — collision URDF, cost field, grape
    targets — shares ONE frame declaration, `collision_frame` in its
    scene.yaml, and one matrix, the scan's `transformation`:
      sim   -> artifacts were baked into sim frame; transform is None.
      splat -> artifacts are in the scan frame; transform is applied at load
               (load_urdf for the URDF, here for targets + cost field)."""
    cfg = scene_registry.get(vine_splat_name) or {}
    frame = cfg.get("collision_frame", "sim")
    if frame == "sim":
        return frame, None
    T = (cfg.get("transformation") or {}).get("matrix")
    if T is None:
        raise ValueError(f"{vine_splat_name}: collision_frame='splat' but no transformation")
    return frame, np.asarray(T, dtype=np.float64)


def _vine_env_config(
    name: str,
    bunch_index: int,
    standoff_m: float,
    grape_targets_json: Path,
    soft_cost_npz: Path,
    vine_splat_name: str = "vine_and_trellis",
    assets_transform=None,
) -> EnvConfig:
    """Build the vine reach EnvConfig. Pure function so tests/subclasses can
    build variants (different bunch, standoff, assets) without subclassing.
    ``assets_transform`` is the scan's splat->sim matrix when the build's
    artifacts are in scan frame (collision_frame: splat), else None."""
    task = None
    try:
        bunches = grape_targets.load_targets(grape_targets_json, transform=assets_transform)
        bunch = bunches[bunch_index]
        # Horizontal approach: aim from the robot column at bunch height so
        # the gripper closes in level with the bunch rather than from above.
        from_point = np.array([_ROBOT_BASE_POS[0], _ROBOT_BASE_POS[1],
                               bunch["center"][2]])
        pos, quat = grape_targets.approach_pose(
            bunch["center"], from_point, standoff=standoff_m
        )
        task = TaskConfig(
            task_description=f"reach grape bunch {bunch_index}",
            target_ee_pos=tuple(float(v) for v in pos),
            target_ee_quat=tuple(float(v) for v in quat),
            pos_tolerance_m=0.03,
            quat_tolerance_deg=15.0,
        )
    except Exception:
        logger.exception(
            "vine env: could not build grape task from %s — env loads with "
            "no task target", grape_targets_json,
        )

    soft_cost = None
    if soft_cost_npz.exists():
        soft_cost = {"npz_path": str(soft_cost_npz)}
        if assets_transform is not None:
            # Raw (xyz + weight) npz: SoftCostField re-rasterises the grid in
            # sim frame after applying this. A prebuilt grid cannot be re-posed.
            soft_cost["transform"] = assets_transform.tolist()
    else:
        logger.warning(
            "vine env: soft-cost npz missing (%s) — planner will run "
            "binary-only", soft_cost_npz,
        )

    return EnvConfig(
        name=name,
        task=task,
        terminate_on_collision=False,
        objects=[
            SplatObjectConfig(
                name="vine",
                splat_name=vine_splat_name,
                grasp_configs=[],
                randomize_pose=False,
                rotation_range_z=(0, 0),
                position_range_x=(0, 0),
                position_range_y=(0, 0),
                # Explicit z-range: a missing range falls back to
                # TABLE_LIMITS, which raises in this tableless env.
                position_range_z=(0, 0),
                # Collision URDF placement follows the build's collision_frame
                # (load_urdf); splat visual (when enabled) by transformation.matrix.
                load_splat=False,
            ),
        ],
        soft_cost=soft_cost,
    )


class VineGrapeReachPybulletRobotServer(SmallEnginePybulletRobotServer):
    """UR5 (small-engine mount) reaching toward a grape bunch on the scanned
    vine, with cost-aware RRT payload published in the oracle env config."""

    DEFAULT_ROBOT_NAME = "robot_iphone_w_engine_curtain"
    # Splat rendering: background = the FULL vine highbay scan (whole scene —
    # vine, trellis, room; the `vine` object keeps load_splat=False since it
    # is a subset of the same scan). Robot splat comes from the robot's own
    # (engine-scene) scan and is articulated as usual.
    RENDER_SPLATS = True
    background_splat_name = "vine_scene"

    # Base splat camera: eye (0.65, -0.75, 0.85) looking at (-0.55, 0.75,
    # 0.55) — robot arm in the foreground, vine canopy + grape bunches
    # behind it (verified render: viz/16_splat_render_full.png). The rpy is
    # the COLMAP-convention look-at (x right, y down, z forward) converted
    # to pybullet's extrinsic-XYZ euler; recompute with
    # scipy Rotation.from_matrix(...).as_euler("xyz") if the eye/target move.
    BASE_CAMERA_OVERRIDE_XYZ = (0.65, -0.75, 0.85)
    BASE_CAMERA_OVERRIDE_RPY = (-1.725719, 0.0, 0.674741)
    BASE_CAMERA_OVERRIDE_DIST_INC = 0.0

    # PyBullet debug camera stays available as a fallback render mode.
    # Framed to show robot + vine canopy.
    RENDER_PYBULLET_CAMERA = True
    PYBULLET_CAMERA_EYE = (0.5, -1.1, 1.0)
    PYBULLET_CAMERA_TARGET = (-0.4, 0.55, 0.5)
    PYBULLET_CAMERA_FOV = 65.0

    # The vine is a segmentation BUILD registered under data/scenes/ — its
    # scene.yaml carries urdf/ply paths and the scan's splat->sim transform,
    # and the folder holds the grape targets + cost field. Retarget to another
    # vine (or another build of the same scan) by changing this one name; see
    # README "Data layout". This build is the trellis-inclusive one: trellis
    # gaussians are forced into the hard collision mesh via
    # `segment_vine_splat.py --force-hard-diff vine_only.ply`.
    VINE_SPLAT_NAME = "vine_and_trellis"
    _SEG_DIR = scene_registry.scene_dir(VINE_SPLAT_NAME)
    # Prefers grape_targets_manual.json when present (see
    # grape_targets.resolve_targets_json): hand annotation outranks detector
    # output, because colour segmentation cannot see green fruit and this
    # prop has red, purple AND green bunches.
    GRAPE_TARGETS_JSON = grape_targets.resolve_targets_json(_SEG_DIR)
    # One frame declaration for every artifact of the build (see _build_frame).
    # Baked builds ship the prebuilt sim-frame grid; unbaked ones ship the raw
    # soft-cost points, rasterised at load after the transform.
    ASSETS_FRAME, ASSETS_TRANSFORM = _build_frame(VINE_SPLAT_NAME)
    SOFT_COST_NPZ = (
        _SEG_DIR / f"{VINE_SPLAT_NAME}_cost_field_sim.npz" if ASSETS_FRAME == "sim"
        else _SEG_DIR / f"{VINE_SPLAT_NAME}_soft_cost.npz"
    )

    @classmethod
    def load_grape_targets(cls) -> list:
        """Bunch targets in SIM frame regardless of how the build was written.
        Use this rather than grape_targets.load_targets(cls.GRAPE_TARGETS_JSON)."""
        return grape_targets.load_targets(cls.GRAPE_TARGETS_JSON, transform=cls.ASSETS_TRANSFORM)
    # Which bunch the task aims at. None (default) = a NEW random bunch at
    # every reset, drawn from the bunches the goal search can reach (a bunch
    # whose search fails is dropped from the draw for the session and
    # logged). Set an int to pin one bunch, e.g. for an eval benchmark.
    TARGET_BUNCH_INDEX = None
    # Bunch used for the STATIC floor task pose built at class-definition
    # time (no robot yet); the live target comes from _pick_target_bunch.
    STATIC_TASK_BUNCH_INDEX = 0
    GRAPE_STANDOFF_M = 0.10
    # Gripper (cutter) approach direction in the wrist_camera_link frame,
    # measured by FK as the direction of wrist -> finger-pad midpoint; see
    # grape_targets.tool_tip_vector, which derives this and
    # GRIPPER_TIP_OFFSET_M as the direction and length of that one vector.
    GRIPPER_AIM_AXIS = (-0.032, 0.221, 0.975)
    # Axis the goal pose actually AIMS, in the wrist_camera_link frame.
    #
    # This is the CAMERA's optical axis (+Z), not GRIPPER_AIM_AXIS. The goal
    # is an IMAGING pose: what matters is that the bunch lands centred in the
    # wrist view, so the axis pointed at the fruit must be the one the camera
    # looks down. Aiming the cutter axis instead (the previous behaviour) put
    # the fruit ~13 deg off-centre in frame, since the two axes differ by that
    # much. The cutter is still served by the POSITION — see AIM_AT_PEDUNCLE.
    CAMERA_FORWARD_AXIS = (0.0, 0.0, 1.0)
    # Position the tool relative to the bunch's PEDUNCLE (where it joins the
    # vine) rather than its centre, so that after this imaging step a simple
    # straight-ahead nudge takes the cutter to the stem — the thing you cut.
    # The camera still centres on the bunch centre (see CAMERA_FORWARD_AXIS),
    # which is what makes this an imaging pose rather than a cutting one.
    AIM_AT_PEDUNCLE = True
    # EE link (wrist_camera_link) to fingertip distance along the aim axis,
    # measured via FK. GRAPE_STANDOFF_M is fingertip-to-bunch — without this
    # offset the wrist sits at the standoff and the fingers overshoot the
    # bunch by the whole gripper length.
    GRIPPER_TIP_OFFSET_M = 0.196
    # Camera roll at the goal pose, specified ABSOLUTELY: the world direction
    # the wrist camera's image-up should point toward. (0, 0, -1) = upside
    # down. Aiming the gripper at the bunch leaves the spin about the aim
    # axis free, and grape_targets solves in closed form for the spin that
    # best matches this.
    #
    # A relative GRIPPER_ROLL_OFFSET_DEG=180 was tried first and does NOT
    # work: the roll is measured from the position-only IK's natural
    # orientation, which is an arbitrary branch, so when that came out
    # sideways +180 just gave the opposite sideways. Tune live with
    # scripts/tune_goal_pose.py.
    GRIPPER_CAMERA_UP_WORLD = (0.0, 0.0, -1.0)
    GRIPPER_ROLL_OFFSET_DEG = 0.0

    ENV_CONFIG = _vine_env_config(
        name="vine_grape_reach",
        bunch_index=STATIC_TASK_BUNCH_INDEX,
        # The static target is expressed at the EE link, so add the tip
        # offset: GRAPE_STANDOFF_M is fingertip-to-bunch. Without this the
        # static pose put the fingers 10 cm INSIDE the bunch and no IK could
        # reach it.
        standoff_m=GRAPE_STANDOFF_M + GRIPPER_TIP_OFFSET_M,
        grape_targets_json=GRAPE_TARGETS_JSON,
        soft_cost_npz=SOFT_COST_NPZ,
        vine_splat_name=VINE_SPLAT_NAME,
        assets_transform=ASSETS_TRANSFORM,
    )

    # Start-pose randomization: per-joint deltas (rad) around the home
    # config. Wide on the shoulder pan so episodes start on either side of
    # the canopy (the traversal case where cost-aware planning matters),
    # narrower elsewhere so starts stay upright and vine-facing.
    START_JOINT_DELTA = (1.0, 0.35, 0.35, 0.5, 0.5, 0.8)

    def randomize_ee_pose(self, max_attempts: int = 100):
        """Arm-start hook consumed by the shared `randomize_objects()` loop.

        The base UR implementation Cartesian-samples over TABLE_LIMITS —
        this env has no table. Sample JOINT space around the home config
        instead (planar does the same over full URDF limits; a 6-DOF UR5
        needs the around-home restriction or starts come out folded/facing
        away). Returns (q1..q6, gripper=open) or None so randomize_objects
        re-rolls the scene."""
        rid = self.splatsim_robot.sim_id
        art = self.splatsim_robot.config.articulation_config
        # Capture the pristine home ONCE — the loop below persists each
        # episode's start into initial_joint_positions (mirroring the base
        # env), which would otherwise make the sampling center drift.
        if not hasattr(self, "_home_arm_q"):
            self._home_arm_q = np.asarray(
                art.initial_joint_positions[: self.num_dofs()], dtype=np.float64
            )
        # Per-joint sampling width; a robot with more joints than the tuple
        # has entries gets the last value for the rest.
        d = list(self.START_JOINT_DELTA)
        deltas = np.asarray((d + [d[-1]] * self.num_dofs())[: self.num_dofs()])
        limits = []
        for j in self.robot_spec.arm_joint_indices:
            info = self.pybullet_client.getJointInfo(rid, j)
            limits.append((info[8], info[9]))

        for _ in range(max_attempts):
            q = self._home_arm_q + np.random.uniform(-deltas, deltas)
            q = np.clip(q, [lo for lo, _ in limits], [hi for _, hi in limits])
            action = tuple(float(v) for v in q) + (0.0,)  # gripper open
            self.teleport_joint_state(self.splatsim_robot, action)
            if not self.is_robot_in_collision():
                if art is not None:
                    art.initial_joint_positions = list(action)
                return action
        return None

    def __init__(self, **kwargs):
        # Default to SPLAT rendering: with no explicit render_mode the base
        # ctor prefers PYBULLET whenever RENDER_PYBULLET_CAMERA is True (we
        # keep that True only as a debug fallback). Explicit --render_mode
        # still wins; the GUI dropdown can switch at runtime.
        if kwargs.get("render_mode") is None:
            from splatsim.configs.mode_config import RenderMode

            kwargs["render_mode"] = RenderMode.SPLAT
        # Default cameras: base + wrist (direct construction; launch_nodes
        # passes camera_names explicitly from objects.yaml wrist-camera info).
        kwargs.setdefault("camera_names", ["base_rgb", "wrist_rgb"])
        super().__init__(**kwargs)
        # The trajectory generator snapshotted the STATIC task goal at
        # construction (before the wrist camera / EE link existed). Refine it
        # to the robot-derived grasp-aligned goal now that FK is available,
        # so GUI/batch trajectory generation aims at the same pose as
        # get_env_config and _resolve_goal_ee_target.
        if getattr(self, "viewer_mode", False):
            return   # viewer: no goal solving, no soft-cost overlay setup
        try:
            pos, quat, q_seed = self._grape_goal()
            cfg = self.trajectory_generator.config
            cfg.ee_pos_goal = [float(v) for v in pos]
            cfg.ee_quat_goal = [float(v) for v in quat]
            cfg.q_goal_bias = [float(v) for v in q_seed]
        except Exception:
            logger.exception(
                "vine env: could not refine trajectory-gen goal — GUI batch "
                "generation will use the STATIC task pose (bunch centre, "
                "horizontal approach, no fingertip offset). Episodes will "
                "still aim at the bunch but from the wrong distance; fix the "
                "goal search before recording."
            )
        # GUI-only overlay of the SOFT vegetation (leaves/twigs/grapes the
        # pipeline kept OUT of the hard collision mesh), viridis by cost
        # weight. Debug points, drawn once: no bodies, no broadphase entry,
        # invisible to getClosestPoints and to TinyRenderer camera images —
        # zero effect on collision checking or stepSimulation. No-op when
        # headless (DIRECT).
        if self.DRAW_SOFT_POINTS_IN_GUI and self.ENV_CONFIG.soft_cost:
            try:
                from splatsim.utils.soft_cost_field import draw_soft_points_in_gui
                import pybullet as _pb

                draw_soft_points_in_gui(
                    _pb,
                    self.ENV_CONFIG.soft_cost["npz_path"],
                    physics_client_id=self._pb_client_id,
                    max_points=self.SOFT_POINTS_GUI_MAX,
                )
            except Exception:
                logger.exception("vine env: soft-point GUI overlay failed (non-fatal)")

    # GUI overlay of soft-cost vegetation points (see __init__). Set False
    # to declutter the GUI view.
    DRAW_SOFT_POINTS_IN_GUI = True
    # How many soft points the GUI overlay draws. One addUserDebugPoints call
    # (a single debug item), but the point count is real GL load, so tools
    # that also need a responsive GUI can turn it down.
    SOFT_POINTS_GUI_MAX = 40000

    def _get_default_trajectory_gen_config(self):
        import dataclasses

        # Anti-wobble settings (elastic_smooth_passes=30,
        # uniform_path_speed=True) are TrajectoryGenModeConfig defaults now —
        # no per-env override needed; tune them live in the GUI Traj Gen panel.
        # The env-OWNED goal. Once __init__ has refined it, this is the
        # search-derived imaging pose (cached in _grape_goal_cache) — so
        # reassert_env_traj_config_fields re-applies THAT after a config push
        # or import, never the static floor. Before refinement (or if it
        # failed) it is the static task pose: bunch centre, horizontal
        # approach, fingertip offset applied. Some EE goal must always be
        # set — with none, TrajectoryGenerator._get_start_and_goal_qs plans
        # every episode after the first to a RANDOM collision-free config.
        refined = getattr(self, "_grape_goal_cache", None)
        task = self.ENV_CONFIG.task
        if refined is not None:
            pos, quat, q_seed = refined
            goal = dict(ee_pos_goal=[float(v) for v in pos],
                        ee_quat_goal=[float(v) for v in quat],
                        q_goal_bias=[float(v) for v in q_seed])
        elif task is not None:
            goal = dict(ee_pos_goal=list(task.target_ee_pos),
                        ee_quat_goal=list(task.target_ee_quat))
        else:
            goal = {}
        return dataclasses.replace(
            super()._get_default_trajectory_gen_config(),
            **goal,
            # Grasp goals put the fingers within mm of the vine by design —
            # enable the planner's finger<->obstacle IK filter.
            ik_skip_gripper_obstacle_pairs=True,
            # Max feasible obstacle leeway for this scene (clearance sweep,
            # 2026-07-28, trellis-inclusive mesh): 2 cm plans reliably from
            # home AND canopy-adjacent starts (measured arm-link min
            # 2.1-2.4 cm); 3 cm is geometrically impossible — grasp-adjacent
            # gripper links sit ~2.5 cm from the bunch's branch and start
            # poses hug the canopy. Don't raise past 0.02 here.
            obstacle_clearance=0.02,
            # Cost-aware GENERATION (T-RRT transition test + cost-gated
            # smoothing), not just cost-aware candidate scoring: with
            # "score" every candidate is generated cost-blind and shortcut
            # smoothing collapses them onto the same route through the
            # canopy. Toggle to "score"/"off" (GUI Traj Gen panel or config
            # JSON) to A/B against cost-blind planning.
            soft_cost_mode="guided",
            # 100, not 5: the candidate score is `joint_arc_length + weight *
            # exposure_integral`. On this scene arc length runs 5-13 rad while
            # the exposure integral runs 0.07-0.68, so at weight 5 the soft
            # term moved the score by ~1% and the scorer simply picked the
            # SHORTEST candidate — which measured as the dirtiest of 7. The
            # winner switches to the cleanest candidate at weight >= 50 and is
            # stable from there through 500; 100 sits in the middle of that
            # plateau. Re-derive with scripts/test_vine_grape_rrt.py if the
            # cost function or scene changes.
            soft_cost_weight=100.0,
            # CHOMP-lite trajopt OFF for this scene: its FD collision
            # gradient issues ~12 min-distance queries per waypoint per pass,
            # each a 1 m-radius getClosestPoints + stepSimulation against the
            # 40k-tri concave vine mesh — observed minutes-to-HOURS per
            # trajectory (2026-07-30). Elastic smoothing + the soft-cost
            # gates provide the smoothing/clearance behavior here instead.
            trajopt_passes=0,
        )

    def _arm_config_collides(self, q) -> bool:
        """collision_fn for goal generation: True if arm config ``q`` puts
        the robot in (hard-obstacle or self) collision. Fingers are checked
        against obstacles too: grapes/foliage are NOT in the hard mesh (only
        trunk + trellis), and finger placement is rigid w.r.t. the EE pose,
        so any goal accepted here passes the planner's final q_goal gate —
        skipping finger pairs let finger-through-trellis-wire grasps through
        that RRT then rejected, burning every IK candidate.
        check_links_in_collision saves/restores joint state itself."""
        from splatsim.utils.rrt_path_utils import check_links_in_collision

        rid = self.splatsim_robot.sim_id
        arm = list(self.robot_spec.arm_joint_indices)
        obstacle_ids = [o.sim_id for o in self.splatsim_objects
                        if o.sim_id is not None and o is not self.splatsim_robot]
        return bool(check_links_in_collision(
            rid, arm, q, obstacle_ids,
            self_collision_skip_pairs=self.SELF_COLLISION_SKIP_PAIRS,
        ))

    def _soft_cost_score_fn(self):
        """q -> soft cost, for RANKING goal-IK candidates (lower = cleaner).

        Reuses the planner's own ``_config_soft_cost`` rather than rolling a
        second metric, so the pose the goal search picks is scored by exactly
        what the planner will later optimise — surface-ring sampling, max
        reduction, same field. A goal chosen by a different metric would keep
        handing the planner poses it then judges as buried in foliage.

        Returns None (no ranking, first acceptable candidate wins) when the
        planner or field is unavailable — e.g. the field npz is missing, or
        this runs before the trajectory generator can build its planner.
        """
        try:
            gen = getattr(self, "trajectory_generator", None)
            if gen is None:
                return None
            planner = gen._ensure_planner()
            if planner._soft_cost_field is None:
                return None
            return planner._config_soft_cost
        except Exception:
            logger.exception(
                "vine env: soft-cost scorer unavailable — goal IK will take "
                "the first acceptable candidate instead of the cleanest")
            return None

    def _grape_goal(self):
        """ROBOT-DERIVED grape IMAGING goal: position-only IK to the standoff
        point, then aim the WRIST CAMERA's optical axis at the bunch centre
        so the fruit lands centred in frame. Candidates are collision-checked
        so the chosen elbow branch doesn't intersect the trunk mesh.

        Candidates are additionally RANKED by the planner's soft-cost
        function, so among poses that image the bunch equally well the one
        whose arm brushes the least foliage/fruit wins.

        Position and aim serve different jobs here. The tool is placed at a
        standoff from the PEDUNCLE (where the bunch joins the vine), so the
        follow-up heuristic — drive straight ahead — brings the cutter to the
        stem. The camera meanwhile centres on the bunch CENTRE, which is what
        makes this a good imaging pose. Set AIM_AT_PEDUNCLE=False to go back
        to positioning off the bunch centre.

        Single source of truth for the goal — used by get_env_config
        (published to the external RRT planner over ZMQ) AND
        _resolve_goal_ee_target (reset-time solvability check + in-env
        trajectory generation), so all consumers aim at the same pose.
        Cached per bunch; `_grape_goal_cache` is the goal for the ACTIVE
        bunch (`active_bunch_index`). Returns (pos, quat, q_seed) or raises."""
        idx = self.active_bunch_index
        by_bunch = self.__dict__.setdefault("_grape_goal_by_bunch", {})
        cached = by_bunch.get(idx)
        if cached is not None:
            self._grape_goal_cache = cached
            return cached
        from splatsim.utils.goal_pose import GoalPoseSpec, solve_goal_pose

        bunches = self.load_grape_targets()
        bunch = bunches[idx]
        if bunch.get("peduncle") is None and self.AIM_AT_PEDUNCLE:
            logger.warning(
                "vine env: %s has no 'peduncle' field — positioning off the "
                "bunch centre. Run scripts/regen_grape_targets.py.",
                self.GRAPE_TARGETS_JSON,
            )
        arm = list(self.robot_spec.arm_joint_indices)
        art = self.splatsim_robot.config.articulation_config
        goal = solve_goal_pose(
            self.pybullet_client,
            self.splatsim_robot.sim_id,
            self._get_ee_link_index(),
            arm,
            bunch,
            GoalPoseSpec.from_env_class(type(self)),
            collision_fn=self._arm_config_collides,
            score_fn=self._soft_cost_score_fn(),
            clouds=self._scene_clouds(),
            q_home=np.asarray(art.initial_joint_positions[: self.num_dofs()], dtype=np.float64),
            ik_fn=self._planner_ik_fn(),
        )
        by_bunch[idx] = goal
        self._grape_goal_cache = goal
        return goal

    # ------------------------------------------------------------ bunch choice
    @property
    def active_bunch_index(self) -> int:
        """The bunch this episode aims at: the pinned TARGET_BUNCH_INDEX, or
        whatever _pick_target_bunch drew at the last reset (bunch 0 until
        the first reset)."""
        if self.TARGET_BUNCH_INDEX is not None:
            return int(self.TARGET_BUNCH_INDEX)
        return int(getattr(self, "_active_bunch_index", 0))

    def _pick_target_bunch(self) -> int:
        """Choose the episode's bunch (random when TARGET_BUNCH_INDEX is
        None), solve its goal, and push that goal into the trajectory-gen
        config so every consumer — reset solvability check, batch
        generation, published env config — aims at the same bunch. Bunches
        whose goal search fails are dropped from the draw for the session."""
        if getattr(self, "viewer_mode", False):
            return self.active_bunch_index
        n = len(self.load_grape_targets())
        if self.TARGET_BUNCH_INDEX is not None:
            order = [int(self.TARGET_BUNCH_INDEX)]
        else:
            bad = self.__dict__.setdefault("_unreachable_bunches", set())
            pool = [i for i in range(n) if i not in bad]
            if not pool:
                logger.error("vine env: every bunch failed the goal search — keeping bunch %d",
                             self.active_bunch_index)
                pool = [self.active_bunch_index]
            first = int(np.random.choice(pool))
            order = [first] + [i for i in pool if i != first]
        for idx in order:
            self._active_bunch_index = idx
            self._grape_goal_cache = None
            if idx not in self.__dict__.get("_grape_goal_by_bunch", {}):
                self._trajgen_status(f"solving goal pose for grape bunch {idx}")
            try:
                pos, quat, q_seed = self._grape_goal()
            except Exception:
                logger.exception("vine env: bunch %d has no reachable goal — dropping it", idx)
                self.__dict__.setdefault("_unreachable_bunches", set()).add(idx)
                continue
            gen = getattr(self, "trajectory_generator", None)
            if gen is not None:
                gen.config.ee_pos_goal = [float(v) for v in pos]
                gen.config.ee_quat_goal = [float(v) for v in quat]
                gen.config.q_goal_bias = [float(v) for v in q_seed]
            logger.info("vine env: target bunch %d (%s)", idx,
                        "pinned" if self.TARGET_BUNCH_INDEX is not None else "random")
            return idx
        return self.active_bunch_index

    def reset(self, seed=None, options=None):
        # Re-draw the target BEFORE randomize_objects runs its solvability
        # check, so the cached reset path and the batch goal agree.
        self._pick_target_bunch()
        return super().reset(seed=seed, options=options)

    def _planner_ik_fn(self):
        """The planner's goal-IK acceptance (5 mm / 5 deg / obstacle
        clearance), for filtering search candidates — so the goal the env
        publishes is one the reset-time plan can actually reach. None when
        the planner isn't available yet; the search then uses its own,
        looser filter."""
        try:
            gen = getattr(self, "trajectory_generator", None)
            if gen is None:
                return None
            planner = gen._ensure_planner()
            gen._sync_planner_obstacles()
            return planner._solve_ik
        except Exception:
            logger.exception("vine env: planner IK unavailable for goal search")
            return None

    def _scene_clouds(self):
        """Point clouds for the task-space goal search (GoalPoseSpec.solver
        == "search"), built once from this build's registry entry: the
        scan's grapes_only.ply (scan frame -> scan transformation), and the
        cost-field points + hard collision mesh (which follow the build's
        collision_frame — ASSETS_TRANSFORM is None when baked). Returns None,
        and the solver falls back to ik, if the grapes PLY is missing."""
        cached = getattr(self, "_scene_clouds_cache", None)
        if cached is not None:
            return cached
        from splatsim.utils.goal_pose import load_scene_clouds
        from splatsim.utils.paths import resolve_splatsim_path
        cfg = scene_registry.get(self.VINE_SPLAT_NAME) or {}
        grapes_ply = Path(resolve_splatsim_path(cfg["model_path"])) / \
            "point_cloud" / "iteration_30000" / "grapes_only.ply"
        if not grapes_ply.exists():
            logger.warning("vine env: %s missing — goal search unavailable, "
                           "using the ik solver", grapes_ply)
            self._scene_clouds_cache = None
            return None
        client, rid = self.pybullet_client, self.splatsim_robot.sim_id
        scan_T = np.asarray(cfg["transformation"]["matrix"], dtype=np.float64)
        self._scene_clouds_cache = load_scene_clouds(
            client, rid, self._get_ee_link_index(),
            gripper_links=list(self.robot_spec.gripper.link_indices),
            grapes_ply=grapes_ply, grapes_transform=scan_T,
            veg_npz=self.SOFT_COST_NPZ, veg_transform=self.ASSETS_TRANSFORM,
            hard_urdf=resolve_splatsim_path(cfg["urdf_path"]),
            hard_transform=self.ASSETS_TRANSFORM,
        )
        return self._scene_clouds_cache

    def _resolve_goal_ee_target(self):
        """Override small_engine's static-task version: that would feed the
        static (possibly unreachable) quat to the reset-time solvability
        check and trajectory generator. Use the robot-derived goal instead;
        fall back to the inherited behavior if it cannot be computed."""
        try:
            pos, quat, q_seed = self._grape_goal()
            return np.asarray(pos), np.asarray(quat), list(q_seed)
        except Exception:
            logger.exception(
                "vine env: robot-derived grape goal failed — falling back to "
                "the static task target"
            )
            return super()._resolve_goal_ee_target()

    def get_env_config(self) -> dict:
        """Publish the static config, then refine the grape task with the
        robot-derived goal (mirrors the planar env's dynamic-task injection)."""
        cfg_dict = super().get_env_config()
        try:
            pos, quat, q_seed = self._grape_goal()
            if cfg_dict.get("task") is None:
                cfg_dict["task"] = {}
            cfg_dict["task"]["target_ee_pos"] = [float(v) for v in pos]
            cfg_dict["task"]["target_ee_quat"] = [float(v) for v in quat]
            cfg_dict["task"]["q_goal_bias"] = [float(v) for v in q_seed]
            cfg_dict["task"]["task_description"] = f"reach grape bunch {self.active_bunch_index}"
            cfg_dict["task"]["bunch_index"] = int(self.active_bunch_index)
        except Exception:
            logger.exception(
                "vine env: runtime grape-task refinement failed — publishing "
                "the static task unchanged"
            )
        return cfg_dict
