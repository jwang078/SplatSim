"""Shared goal-pose generation: one definition of "where should the arm be
to image a target", used by the env server AND by standalone tools.

Everything about how a goal pose is specified lives here — the wrist-camera
convention, the camera-up modes, the position-vs-aim split, the parameter
defaults — so `tune_goal_pose.py` cannot drift from what the env actually
generates. Before this existed the tuner re-derived all of it and quietly
disagreed with the env (different aim axis, different tip offset, no
soft-cost ranking), which is exactly the class of bug a tuning tool must not
have: you tune against one thing and ship another.

The heavy lifting stays in `grape_targets.reachable_approach_pose`; this is
the layer that decides WHAT to ask it for.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

from splatsim.utils import grape_targets

# --------------------------------------------------------------- conventions
# The wrist camera looks down `wrist_camera_link` +Z and its image-up is -Y
# (COLMAP +Y is down), with NO offset between link frame and camera frame.
# Copied from PybulletRobotServerBase.get_wrist_camera_transform /
# _splat_camera_from_gsplat; keep in sync if the robot's camera mounting
# convention ever changes.
CAMERA_FORWARD_AXIS = (0.0, 0.0, 1.0)
CAMERA_UP_AXIS = (0.0, -1.0, 0.0)


class CameraUpMode(str, Enum):
    """Which way the wrist camera's image-up should point in WORLD space."""

    OFF = "off"          # unconstrained — whatever roll IK lands on
    UPRIGHT = "upright"  # image-up toward world +Z
    INVERTED = "inverted"  # image-up toward world -Z (upside down)

    def world_up(self):
        """World direction for `align_up_world`, or None when unconstrained."""
        if self is CameraUpMode.UPRIGHT:
            return (0.0, 0.0, 1.0)
        if self is CameraUpMode.INVERTED:
            return (0.0, 0.0, -1.0)
        return None

    @classmethod
    def from_world_up(cls, vec) -> "CameraUpMode":
        """Inverse of `world_up` — lets an env express its preference as a
        vector (GRIPPER_CAMERA_UP_WORLD) and still round-trip to a mode."""
        if vec is None:
            return cls.OFF
        return cls.INVERTED if float(np.asarray(vec)[2]) < 0 else cls.UPRIGHT


def wrist_camera_pose(client, robot_id: int, ee_link: int):
    """``(eye, forward, up)`` of the wrist camera at the CURRENT joint state,
    in world coordinates, using the convention above."""
    st = client.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    eye = np.asarray(st[4], dtype=np.float64)
    rot = np.asarray(client.getMatrixFromQuaternion(st[5]),
                     dtype=np.float64).reshape(3, 3)
    return eye, rot[:, 2], -rot[:, 1]


def wrist_view_matrix(client, robot_id: int, ee_link: int):
    """PyBullet view matrix looking through the wrist camera right now."""
    eye, forward, up = wrist_camera_pose(client, robot_id, ee_link)
    return client.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=(eye + forward).tolist(),
        cameraUpVector=up.tolist(),
    )


def camera_up_state(quat, client=None):
    """Classify an achieved goal orientation. Returns
    ``(state, tilt_deg, up_world)`` where state is "upright" / "UPSIDE DOWN"
    / "SIDEWAYS".

    The deadband matters: a bare ``up_z < 0`` test labels a HORIZONTAL camera
    upside down, which is what the position-relaxation search often lands on,
    so that test reports success for poses that are nothing of the kind.
    """
    import pybullet as pb

    rot = np.asarray(
        (client.getMatrixFromQuaternion(quat) if client is not None
         else pb.getMatrixFromQuaternion(quat)), dtype=np.float64).reshape(3, 3)
    up = -rot[:, 1]
    uz = float(up[2])
    state = "upright" if uz > 0.5 else "UPSIDE DOWN" if uz < -0.5 else "SIDEWAYS"
    tilt = float(np.degrees(np.arccos(np.clip(uz, -1.0, 1.0))))
    return state, tilt, up


# ------------------------------------------------------------------- the spec
@dataclasses.dataclass
class GoalPoseSpec:
    """Everything that defines an imaging goal pose. Envs own the values;
    tools read them off the env so both ask for the same thing."""

    standoff_m: float = 0.10
    # None = measure from the URDF by FK (grape_targets.tool_tip_vector).
    tip_offset_m: float | None = None
    # Axis pointed AT the target. Defaults to the camera's optical axis, so
    # the fruit lands centred in frame; aiming the tool axis instead puts it
    # off-centre by the angle between the two (~13 deg on this gripper).
    aim_axis_local: tuple = CAMERA_FORWARD_AXIS
    camera_up_axis_local: tuple = CAMERA_UP_AXIS
    camera_up: CameraUpMode = CameraUpMode.OFF
    roll_offset_deg: float = 0.0
    max_aim_error_deg: float = 12.0
    max_up_error_deg: float = 45.0
    position_relax_m: float = 0.08
    position_relax_step_m: float = 0.02
    ik_seed: int = 0
    # Search budget. More seeds = more elbow/wrist branches tried, which is
    # what finds solutions on bunches where the obvious branch collides;
    # ik_enough_candidates stops the sweep once that many pass every hard
    # gate, so the budget is only spent where solutions are scarce.
    ik_random_seeds: int = 12
    ik_enough_candidates: int = 5
    # False (default) = an unmet camera-up request raises instead of silently
    # returning a differently-rolled pose.
    allow_roll_fallback: bool = False
    # Position off the bunch PEDUNCLE (cut-ready) rather than its centre; the
    # camera still aims at the centre.
    aim_at_peduncle: bool = True
    # Which solver `solve_goal_pose` runs.
    #   "search" (default) — task-space sample / score / IK-filter over a
    #             shell of approach directions around the bunch
    #             (ee_pose_search). Finds a pose on every bunch of the
    #             highbay scene; needs the scene clouds (see SceneClouds).
    #   "ik"     — the older config-space solver: one approach direction
    #             (from the robot column, horizontal), fixed standoff, then IK
    #             + roll matching. Manages 2/7 bunches under the same
    #             constraints. Kept as the fallback when search finds nothing
    #             or no clouds were given.
    solver: str = "search"
    # Pool size matters more than it looks: candidates are ranked by
    # task-space score, but the planner's goal gate (5 mm / 5 deg / 2 cm
    # obstacle clearance on the WHOLE arm) rejects most of them — measured
    # on the highbay bunch 0: 200 dirs / top 25 -> 0-1 survivors, 400 / 40
    # -> ~6. Search cost is a few seconds either way.
    search_directions: int = 400
    search_top_k: int = 40
    # IK attempts per candidate: the first is seeded from home and usually
    # drifts past the planner's 60-degree seed tolerance, so the random-seed
    # retries are what actually find the branch.
    search_ik_attempts: int = 6
    # Rounds of the search to run before giving up. Survivors are few (a
    # handful out of 40) and the IK seeds are random, so one round can come
    # up empty on a bunch that is perfectly reachable; each retry draws
    # fresh seeds and doubles the per-candidate attempts.
    search_retries: int = 3
    # Search standoff window around standoff_m (metres below / above).
    search_standoff_below_m: float = 0.04
    search_standoff_above_m: float = 0.06

    @classmethod
    def from_env_class(cls, env_cls, **overrides) -> "GoalPoseSpec":
        """Build from an env server class's attributes, so the env stays the
        single source of truth and tools inherit its choices."""
        up_world = getattr(env_cls, "GRIPPER_CAMERA_UP_WORLD", None)
        spec = cls(
            standoff_m=float(getattr(env_cls, "GRAPE_STANDOFF_M", 0.10)),
            tip_offset_m=None,
            aim_axis_local=tuple(getattr(env_cls, "CAMERA_FORWARD_AXIS",
                                         CAMERA_FORWARD_AXIS)),
            camera_up=CameraUpMode.from_world_up(up_world),
            roll_offset_deg=float(getattr(env_cls, "GRIPPER_ROLL_OFFSET_DEG", 0.0)),
            aim_at_peduncle=bool(getattr(env_cls, "AIM_AT_PEDUNCLE", True)),
        )
        for k, v in overrides.items():
            setattr(spec, k, v)
        return spec


def resolve_targets(bunch: dict, spec: GoalPoseSpec):
    """``(reach_pt, look_at)`` for a bunch: where the TOOL goes versus what
    the CAMERA centres on.

    The tool sits off the peduncle (the stem a cutter must reach, so a
    straight-ahead nudge after imaging arrives at it) while the camera
    centres on the bunch centre. Falls back to the centre for target files
    written before the peduncle field existed.
    """
    center = np.asarray(bunch["center"], dtype=np.float64)
    ped = bunch.get("peduncle")
    if ped is None or not spec.aim_at_peduncle:
        return center, center
    return np.asarray(ped, dtype=np.float64), center


# ------------------------------------------------------------ scene clouds
@dataclasses.dataclass
class SceneClouds:
    """Point clouds the task-space search scores against. All in SIM frame.
    Build once per scene with `load_scene_clouds`; reuse across bunches."""

    grapes: np.ndarray          # segmented fruit gaussians (N, 3)
    veg: np.ndarray             # soft-cost points: foliage / twigs (M, 3)
    grip_centers: np.ndarray    # gripper modelled as spheres, EE-local
    grip_radii: np.ndarray
    hard: np.ndarray | None     # hard collision mesh samples, or None
    grapes_tree: object = None  # cKDTree over `grapes`

    def __post_init__(self):
        from scipy.spatial import cKDTree
        if self.grapes_tree is None:
            self.grapes_tree = cKDTree(self.grapes)


def _apply(points, T):
    if T is None:
        return np.asarray(points, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


def sample_collision_mesh(urdf_path, transform=None, step: int = 1):
    """Point-sample the collision OBJ a URDF wraps: vertices plus per-face
    centroids (vertices alone leave gaps across large triangles, and a gap
    here is a gripper that passes through a trellis wire unnoticed).
    ``transform`` moves a scan-frame mesh into sim frame; None for a baked
    one. Returns None when the URDF has no mesh collision."""
    from pathlib import Path
    urdf = Path(str(urdf_path))
    obj = None
    if urdf.exists():
        for ln in urdf.read_text().splitlines():
            if "<mesh filename=" in ln and "collision" in ln:
                obj = urdf.parent / ln.split('filename="')[1].split('"')[0]
                break
    if obj is None or not obj.exists():
        return None
    verts, faces = [], []
    for ln in obj.read_text().splitlines():
        if ln.startswith("v "):
            verts.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    pts = [v]
    if faces:
        pts.append(v[np.asarray(faces, dtype=int)].mean(axis=1))
    return _apply(np.concatenate(pts)[::step], transform)


def load_scene_clouds(client, robot_id: int, ee_link: int, gripper_links,
                      grapes_ply, grapes_transform, veg_npz, veg_transform,
                      hard_urdf, hard_transform) -> SceneClouds:
    """Assemble `SceneClouds` for one segmentation build. Each source gets
    its own transform because they need not share a frame: the grapes PLY is
    a scan-frame splat subset (pass the scan's splat->sim matrix), while the
    cost-field points and collision mesh follow the build's collision_frame
    (None when baked to sim, the same matrix when not)."""
    from splatsim.utils import ee_pose_search as eps
    from splatsim.utils.splat_ply_io import read_gaussian_ply
    grapes = _apply(read_gaussian_ply(grapes_ply).xyz, grapes_transform)
    data = np.load(veg_npz, allow_pickle=False)
    # prebuilt grid (baked build) stores "points"; raw segmentation "xyz"
    veg = _apply(data["points"] if "points" in data else data["xyz"], veg_transform)
    gc, gr = eps.gripper_spheres(client, robot_id, ee_link, list(gripper_links))
    hard = sample_collision_mesh(hard_urdf, hard_transform)
    return SceneClouds(grapes=grapes, veg=veg, grip_centers=gc, grip_radii=gr,
                       hard=hard)


def search_goal_poses(client, robot_id: int, ee_link: int, joint_indices,
                      bunch: dict, spec: GoalPoseSpec, clouds: SceneClouds,
                      joint_limits, q_home, collision_fn=None,
                      direction_hint=None, direction_max_angle_deg=180.0,
                      ik_attempts: int = 4, ik_fn=None):
    """Task-space search: sample poses on a shell around the bunch, score
    them (visibility, gripper clearance, approach corridor, camera-up,
    standoff, hard-geometry margin), keep a diverse top-K, then IK-filter
    with a full-arm collision check.

    Returns ``(candidates, message)`` — candidates are
    ``[(score, pos, quat_xyzw, q, terms), ...]`` best first, possibly empty.
    The single source of the pipeline: the env, tune_goal_pose.py and
    optimize_ee_poses.py all come through here.

    ``ik_fn(pos, quat, seed_q | None) -> q | None`` replaces the built-in
    IK + collision filter when given. Pass the planner's own ``_solve_ik`` so
    a candidate is accepted here iff the planner will accept it later — the
    built-in filter (10 mm / 15 deg, penetration-only collision) is looser
    than the planner's goal gate (5 mm / 5 deg, 2 cm obstacle clearance),
    and a goal that passes the loose gate but not the strict one makes every
    reset plan fail with "No collision-free IK solution"."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as R
    from splatsim.utils import ee_pose_search as eps

    ctr = np.asarray(bunch["center"], dtype=np.float64)
    ext = max(bunch.get("extent", [0.1, 0.1, 0.1]))
    tp = clouds.grapes[np.asarray(
        clouds.grapes_tree.query_ball_point(ctr, 0.5 * ext + 0.04))]
    if len(tp) == 0:
        return [], "no grape points near this target"
    d, _ = cKDTree(tp).query(clouds.veg)
    cl = eps.build_clouds(clouds.grapes, clouds.veg, clouds.grip_centers,
                          clouds.grip_radii, occluder_pts=clouds.veg[d > 0.03],
                          hard_pts=clouds.hard)
    tip = spec.tip_offset_m
    if tip is None:
        _, tip = grape_targets.tool_tip_vector(client, robot_id, ee_link)
    sspec = eps.SearchSpec(
        n_directions=spec.search_directions, top_k=spec.search_top_k,
        standoff_range=(max(spec.standoff_m - spec.search_standoff_below_m, 0.03),
                        spec.standoff_m + spec.search_standoff_above_m),
        camera_up_world=spec.camera_up.world_up(),
        aim_axis_local=tuple(spec.aim_axis_local),
        up_axis_local=tuple(spec.camera_up_axis_local),
        tip_offset_m=float(tip),
        base_xyz=tuple(np.asarray(client.getBasePositionAndOrientation(robot_id)[0])),
    )
    if direction_hint is not None:
        sspec.direction_hint = tuple(direction_hint)
        sspec.direction_max_angle_deg = float(direction_max_angle_deg)
    pos_s, rot_s, ap_s = eps.sample_poses(ctr, sspec)
    n_raw = len(eps.fibonacci_directions(sspec.n_directions)) * sspec.n_standoffs
    note = ""
    if len(pos_s) < 0.05 * n_raw:
        # Distant bunches leave only a sliver of the shell inside reach —
        # "0 feasible" from 5 samples is a sampling problem, not geometry.
        note = (f" [only {len(pos_s)}/{n_raw} poses inside reach; bunch is "
                f"{np.linalg.norm(ctr - np.asarray(sspec.base_xyz)):.2f} m from base]")
    if not len(pos_s):
        return [], "no candidate poses (reach prefilter)" + note
    sc = np.full(len(pos_s), -1.0); terms = [None] * len(pos_s); n_hard = 0
    for i in range(len(pos_s)):
        if eps.gripper_hits_hard(pos_s[i], rot_s[i], cl):
            n_hard += 1
            continue
        sc[i], terms[i] = eps.score_pose(pos_s[i], rot_s[i], ap_s[i], cl, sspec, tp)
    ok = np.flatnonzero(sc >= 0)
    if not len(ok):
        return [], "every pose hits the trellis" + note
    keep = ok[eps.select_diverse(pos_s[ok], rot_s[ok], sc[ok], sspec)]
    joint_limits = np.asarray(joint_limits, dtype=np.float64)
    joint_indices = list(joint_indices)
    saved = [client.getJointState(robot_id, j)[0] for j in joint_indices]
    out = []
    if ik_fn is not None:
        for i in keep:
            quat = R.from_matrix(rot_s[i]).as_quat()
            for att in range(ik_attempts):
                q = ik_fn(pos_s[i], quat, np.asarray(q_home) if att == 0 else None)
                if q is not None:
                    out.append((float(sc[i]), pos_s[i], np.asarray(quat), np.asarray(q), terms[i]))
                    break
        out.sort(key=lambda t: -t[0])
        return out, (f"{len(out)}/{len(keep)} IK-feasible (planner gate) of "
                     f"{len(pos_s)} sampled ({n_hard} hit trellis)" + note)
    try:
        for i in keep:
            quat = R.from_matrix(rot_s[i]).as_quat()
            for att in range(ik_attempts):
                seed = (np.asarray(q_home) if att == 0 else
                        np.random.uniform(joint_limits[:, 0], joint_limits[:, 1]))
                for j, qq in zip(joint_indices, seed):
                    client.resetJointState(robot_id, j, float(qq))
                sol = client.calculateInverseKinematics(
                    robot_id, ee_link, pos_s[i].tolist(), list(quat),
                    maxNumIterations=300, residualThreshold=1e-9)
                q = np.asarray(sol[:len(joint_indices)])
                for j, qq in zip(joint_indices, q):
                    client.resetJointState(robot_id, j, float(qq))
                st = client.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
                if np.linalg.norm(np.asarray(st[4]) - pos_s[i]) > 0.01:
                    continue
                if R.from_matrix(rot_s[i].T @ R.from_quat(st[5]).as_matrix()
                                 ).magnitude() > np.radians(15):
                    continue
                if collision_fn is not None and collision_fn(q):
                    continue
                out.append((float(sc[i]), pos_s[i], np.asarray(st[5]), q, terms[i]))
                break
    finally:
        for j, qq in zip(joint_indices, saved):
            client.resetJointState(robot_id, j, float(qq))
    out.sort(key=lambda t: -t[0])
    return out, (f"{len(out)}/{len(keep)} IK-feasible of {len(pos_s)} sampled "
                 f"({n_hard} hit trellis)" + note)


def solve_goal_pose(client, robot_id: int, ee_link: int, joint_indices,
                    bunch: dict, spec: GoalPoseSpec, from_point=None,
                    collision_fn=None, score_fn=None, clouds=None,
                    joint_limits=None, q_home=None, ik_fn=None):
    """Generate the imaging goal pose for ``bunch``.

    Runs the solver the spec names. With ``solver="search"`` and ``clouds``
    given, the task-space search runs first and its best candidate is
    returned; if it finds nothing (or no clouds were supplied) the
    config-space ``ik`` solver runs as the fallback.

    Returns ``(pos, quat, q_seed)``; raises ValueError when no pose satisfies
    the spec (including an unmet camera-up request, unless the spec allows
    the roll fallback).
    """
    if spec.solver == "search":
        if clouds is None:
            import logging
            logging.getLogger(__name__).warning(
                "solve_goal_pose: solver='search' but no scene clouds given — "
                "falling back to the ik solver")
        else:
            if joint_limits is None:
                joint_limits = [client.getJointInfo(robot_id, j)[8:10]
                                for j in joint_indices]
            if q_home is None:
                q_home = [client.getJointState(robot_id, j)[0] for j in joint_indices]
            import logging
            log = logging.getLogger(__name__)
            msg = ""
            for rnd in range(max(1, spec.search_retries)):
                cands, msg = search_goal_poses(
                    client, robot_id, ee_link, joint_indices, bunch, spec, clouds,
                    joint_limits, q_home, collision_fn=collision_fn, ik_fn=ik_fn,
                    ik_attempts=spec.search_ik_attempts * (2 ** rnd))
                if cands:
                    if rnd:
                        log.info("solve_goal_pose: search succeeded on retry %d (%s)", rnd, msg)
                    _, pos, quat, q, _ = cands[0]
                    return np.asarray(pos), np.asarray(quat), np.asarray(q)
                log.info("solve_goal_pose: search round %d found no pose (%s)", rnd, msg)
            log.warning(
                "solve_goal_pose: search found no pose after %d rounds (%s) — "
                "falling back to the ik solver", max(1, spec.search_retries), msg)
    reach_pt, look_at = resolve_targets(bunch, spec)
    tip = spec.tip_offset_m
    if tip is None:
        _, tip = grape_targets.tool_tip_vector(client, robot_id, ee_link)
    if from_point is None:
        base = np.asarray(client.getBasePositionAndOrientation(robot_id)[0],
                          dtype=np.float64)
        from_point = np.array([base[0], base[1], reach_pt[2]])
    return grape_targets.reachable_approach_pose(
        client, robot_id, ee_link, list(joint_indices), reach_pt,
        look_at=look_at,
        standoff=spec.standoff_m,
        from_point=from_point,
        aim_axis_local=spec.aim_axis_local,
        camera_up_axis_local=spec.camera_up_axis_local,
        align_up_world=spec.camera_up.world_up(),
        roll_offset_deg=spec.roll_offset_deg,
        max_aim_error_deg=spec.max_aim_error_deg,
        max_up_error_deg=spec.max_up_error_deg,
        position_relax_m=spec.position_relax_m,
        position_relax_step_m=spec.position_relax_step_m,
        ik_seed=spec.ik_seed,
        ik_random_seeds=spec.ik_random_seeds,
        ik_enough_candidates=spec.ik_enough_candidates,
        allow_roll_fallback=spec.allow_roll_fallback,
        collision_fn=collision_fn,
        tool_tip_offset=tip,
        score_fn=score_fn,
    )


# ------------------------------------------------------------- projection
def project_points(points_world, splatsim_camera, rectify_zoom: float = 1.0):
    """Project world points into a SplatSimCamera's IMAGE pixels.

    Returns ``(uv, valid)`` — ``uv`` is (N, 2) float pixel coordinates and
    ``valid`` marks points in front of the camera.

    Matches what ``render_image`` actually hands back. That matters for the
    fisheye wrist: the render is RECTIFIED before it is returned
    (``_rectify_fisheye_image`` remaps it to a pinhole at K with fx/fy scaled
    by FISHEYE_RECTIFY_ZOOM), so projecting with the RAW fisheye intrinsics
    and distortion would put markers in the wrong place. Pass the same zoom
    the server uses and this is exact pinhole math against the rectified
    image; pass 1.0 for an already-pinhole camera.

    Duck-types the camera (``.camera.R/.T/.image_width/.image_height``,
    ``.intrinsic_matrix``, ``.camera_model``) so this module needs no import
    from the robot-server package.
    """
    cam = splatsim_camera.camera
    rot = np.asarray(cam.R, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(cam.T, dtype=np.float64).reshape(3)
    eye = -rot @ trans
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    # Camera axes are the columns of R (z forward, y down — COLMAP), so the
    # camera-frame coordinates are just (p - eye) projected onto them.
    p_cam = (pts - eye) @ rot
    z = p_cam[:, 2]

    width, height = float(cam.image_width), float(cam.image_height)
    k = getattr(splatsim_camera, "intrinsic_matrix", None)
    if k is not None:
        k = np.asarray(k.detach().cpu().numpy() if hasattr(k, "detach") else k,
                       dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        if getattr(splatsim_camera, "camera_model", "pinhole") == "fisheye":
            fx *= rectify_zoom
            fy *= rectify_zoom
    else:
        fx = (width / 2.0) / np.tan(float(cam.FoVx) * 0.5)
        fy = (height / 2.0) / np.tan(float(cam.FoVy) * 0.5)
        cx, cy = width / 2.0, height / 2.0

    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    return np.stack([u, v], axis=1), z > 1e-6
