"""Grape-bunch reach targets: load clustered bunches and build approach poses.

Bunch clusters come from ``splat_segmentation.cluster_labeled_points`` (class
GRAPE) and are cached as JSON (e.g. data/vine_seg/<scene>/grape_targets.json,
written by scripts/segment_vine_splat.py users or ad hoc). This module is the
single place that turns a bunch center into an end-effector goal pose, so the
env task config and standalone planner tests stay consistent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


MANUAL_TARGETS_NAME = "grape_targets_manual.json"
AUTO_TARGETS_NAME = "grape_targets.json"


def resolve_targets_json(scene_dir):
    """Pick a scene's bunch list, preferring HAND-ANNOTATED targets.

    Two files, with a deliberate split of ownership:
      grape_targets_manual.json  authored by scripts/mark_grape_targets.py.
                                 Automation must never write this.
      grape_targets.json         detector output; regenerated freely.

    The manual file wins when present, so hand annotation survives any
    re-run of the segmentation pipeline. Delete it to fall back to the
    detector once the real one lands.

    Colour-based detection cannot see green fruit (its hue windows key on
    purple/red), which is why hand annotation has to outrank it rather than
    merely seed it.
    """
    scene_dir = Path(scene_dir)
    manual = scene_dir / MANUAL_TARGETS_NAME
    return manual if manual.exists() else scene_dir / AUTO_TARGETS_NAME


def is_manual(targets) -> bool:
    """True if any bunch in the list was hand-marked — used to refuse
    clobbering annotation work from an automated path."""
    return any(b.get("manual") for b in targets)


def load_targets(json_path: str | Path) -> list:
    """List of bunch dicts ({center, n_points, extent}), largest first."""
    return json.loads(Path(json_path).read_text())


def compute_targets(
    ply_path: str | Path,
    labels_path: str | Path,
    transform: np.ndarray | list | None = None,
    eps: float = 0.025,
    min_points: int = 150,
) -> list:
    """Recompute bunch clusters from a segmented splat (sim frame if
    ``transform`` given). Slower than load_targets; use for regeneration."""
    from splatsim.utils import splat_segmentation as seg
    from splatsim.utils.splat_ply_io import read_gaussian_ply

    cloud = read_gaussian_ply(ply_path)
    labels = np.load(labels_path)
    return seg.cluster_labeled_points(
        cloud.xyz, labels, seg.GRAPE, eps=eps, min_points=min_points,
        transform=transform,
    )


def gripper_finger_links(client, robot_id: int) -> list:
    """URDF link indices whose name contains 'finger' or 'knuckle' — the
    Robotiq subtree. Same name-based detection as the RRT planner's
    ``ik_skip_gripper_obstacle_pairs`` filter, so goal generation and the
    planner's IK gate agree on which links may legitimately be within
    millimeters of the target at a grasp pose."""
    links = []
    for j in range(client.getNumJoints(robot_id)):
        name = client.getJointInfo(robot_id, j)[12].decode().lower()
        if "finger" in name or "knuckle" in name:
            links.append(j)
    return links


def tool_tip_vector(client, robot_id: int, ee_link: int,
                    finger_links: list | None = None) -> tuple:
    """``(aim_axis_local, tip_offset_m)`` measured from the URDF by FK.

    Both are the same quantity: the vector from the EE link origin to the
    fingertip-pad midpoint, expressed in the EE LINK frame. Its DIRECTION is
    the tool's approach axis (``aim_axis_local``) and its LENGTH is the
    EE-link-to-fingertip distance (``tool_tip_offset``). They are rigid
    geometry, not free parameters — deriving both here keeps them from
    drifting apart, which hand-copied constants invite (a stale tip offset
    silently makes the fingers over- or under-shoot the target by the error).

    Prefers ``*_pad`` links (the actual contact surfaces); falls back to all
    finger/knuckle links. NOTE this depends on the CURRENT gripper opening —
    the pads swing as the fingers close — so call it at the opening the goal
    pose assumes (typically fully open).
    """
    links = finger_links if finger_links is not None else gripper_finger_links(
        client, robot_id)
    pads = [li for li in links
            if "pad" in client.getJointInfo(robot_id, li)[12].decode().lower()]
    if not pads:
        pads = links
    if not pads:
        raise ValueError("no finger/knuckle links found on this robot")
    state = client.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    origin = np.asarray(state[4], dtype=np.float64)
    rot = np.asarray(client.getMatrixFromQuaternion(state[5]),
                     dtype=np.float64).reshape(3, 3)
    mid = np.mean([
        np.asarray(client.getLinkState(robot_id, li,
                                       computeForwardKinematics=True)[4],
                   dtype=np.float64)
        for li in pads], axis=0)
    v = rot.T @ (mid - origin)          # into the EE link frame
    length = float(np.linalg.norm(v))
    return v / max(length, 1e-9), length


def reachable_approach_pose(
    client,
    robot_id: int,
    ee_link: int,
    joint_indices: list,
    bunch_center: np.ndarray | list,
    standoff: float = 0.10,
    from_point: np.ndarray | list | None = None,
    aim_axis_local: np.ndarray | list | None = None,
    look_at: np.ndarray | Sequence[float] | None = None,
    max_aim_error_deg: float = 12.0,
    max_up_error_deg: float = 45.0,
    allow_roll_fallback: bool = False,
    position_relax_m: float = 0.08,
    position_relax_step_m: float = 0.02,
    ik_seed: int = 0,
    ik_random_seeds: int = 12,
    ik_enough_candidates: int = 5,
    collision_fn=None,
    score_fn=None,
    tool_tip_offset: float = 0.0,
    roll_offset_deg: float = 0.0,
    align_up_world: np.ndarray | Sequence[float] | None = None,
    camera_up_axis_local: np.ndarray | Sequence[float] = (0.0, -1.0, 0.0),
) -> tuple:
    """Like :func:`approach_pose`, but the ORIENTATION comes from the robot:
    solve position-only IK for the standoff point and adopt the achieved EE
    orientation. Guarantees a feasible (pos, quat) pair regardless of the
    tool-frame convention (hand-built look-at quats are easy to get ~90deg
    wrong), and returns the IK config as a natural ``q_goal_bias`` seed.

    ``aim_axis_local`` (unit-ish vector in the EE-LINK frame, e.g. the
    measured gripper approach axis) additionally makes the tool FACE the
    bunch — grasp-ready — instead of whatever wrist-down orientation
    position-only IK happens to produce:
      1. position-only IK -> natural orientation R0
      2. minimal rotation taking R0's aim axis onto the bunch direction ->
         target orientation (preserves as much of the natural, reachable
         orientation as possible)
      3. orientation-constrained IK, verified (position + aim alignment);
         on failure retries a sweep of rolls about the aim direction and
         random IK re-seeds (different elbow/wrist branches)
    ``ik_random_seeds`` / ``ik_enough_candidates`` size the search. Each
    candidate EE position sweeps 10 roll offsets x (3 + ``ik_random_seeds``)
    IK seeds; more seeds means more elbow/wrist branches tried, which is what
    rescues bunches where the obvious branch collides. ``ik_enough_candidates``
    stops the sweep once that many candidates have passed EVERY hard gate, so
    the larger budget costs nothing on easy targets and is only spent where
    solutions are actually scarce. The kept candidates are then ranked by
    ``score_fn`` and the cheapest returned (so this is "best of N found",
    not "first found").

    ``score_fn(q) -> float`` (lower = better, e.g. a soft-cost lookup) makes
    the search RANK every fully-acceptable candidate instead of returning the
    first one found. Several rolls and elbow branches reach the same
    viewpoint but differ greatly in how much soft vegetation the arm pushes
    through, and take-first has no reason to pick the clean one. None keeps
    the original first-acceptable behaviour.

    ``collision_fn(q) -> bool`` (True = colliding) additionally rejects
    candidates whose CONFIG collides — the same aimed pose can be reached
    with elbow placements that do or don't sweep through hard obstacles.
    Raises ValueError if no aligned (and collision-free) solution verifies.

    ``look_at``: what ``aim_axis_local`` is pointed at, defaulting to
    ``bunch_center``. Splitting the two lets an IMAGING pose sit at a
    cut-ready standoff from the peduncle while the camera stays centred on
    the fruit. NOTE ``tool_tip_offset`` is still applied along the approach
    RAY (toward ``bunch_center``); when the aim axis is the camera axis and
    the tool axis differs from it by a few degrees, the tip lands within a
    few mm of the nominal standoff rather than exactly on it.

    ``tool_tip_offset``: distance from the EE LINK to the fingertips along
    the aim axis. ``standoff`` is then measured from the FINGERTIPS — the
    thing you actually want near the target. Without this, the wrist link
    sits at the standoff and the fingers overshoot the target by the whole
    gripper length (~0.2 m on the Robotiq/UR5).

    Aiming the tool at the bunch fixes only two of the three orientation
    DOF; the roll about the aim axis is free. Two ways to pin it down:

    ``align_up_world`` (RECOMMENDED — absolute): a WORLD direction the
    camera's image-up should point toward. The roll achieving it is solved in
    closed form, so the result does NOT depend on which branch the
    position-only IK happened to land on. ``(0, 0, 1)`` = upright,
    ``(0, 0, -1)`` = upside down. Alignment is exact only up to the aim
    constraint — image-up must stay perpendicular to the aim axis, so what is
    achieved is the requested direction projected into that plane. Degenerate
    when the aim is parallel to the requested up (aiming straight up/down);
    that case falls back to ``roll_offset_deg`` alone.

    ``roll_offset_deg`` (RELATIVE — for nudging): degrees added on top. With
    ``align_up_world=None`` this is measured from the position-only IK's
    natural orientation, which is ARBITRARY — if that solve returns a
    sideways wrist then ``roll_offset_deg=180`` yields the OTHER sideways
    wrist, not an upside-down one. That is why absolute alignment is
    preferred for specifying camera orientation.

    ``allow_roll_fallback``: when the camera-up request cannot be met
    anywhere in the relaxation ball, return the closest-rolled pose (with a
    warning) instead of raising. Default False — an unmet camera-up request
    is an error, because a silently upright pose in an "inverted" dataset is
    far harder to notice than a failed solve.

    ``camera_up_axis_local``: which EE-link axis is the camera's image-up.
    Defaults to -Y, matching PybulletRobotServerBase's convention (camera
    looks down wrist_camera_link +Z, image-up -Y, COLMAP +Y being down).

    The roll sweep still runs around the chosen value, so a blocked preferred
    roll degrades to a nearby reachable one rather than failing outright —
    check the returned quat if the preference must be honored exactly.

    Restores the robot's joint state before returning.
    Returns (pos(3,), quat(x,y,z,w), q_seed(len(joint_indices),)).
    """
    from scipy.spatial.transform import Rotation

    center = np.asarray(bunch_center, dtype=np.float64)
    if from_point is None:
        from_point, _ = client.getBasePositionAndOrientation(robot_id)
    origin = np.asarray(from_point, dtype=np.float64)
    direction = center - origin
    direction /= max(np.linalg.norm(direction), 1e-9)
    pos = center - direction * (standoff + tool_tip_offset)
    # POSITION and AIM are decoupled: `bunch_center` places the tool (for a
    # cutter that is the PEDUNCLE — the stem you cut, so a straight-ahead
    # nudge reaches it), while `look_at` is what the aim axis points at (for
    # an imaging pose that is the bunch CENTRE, so the fruit lands centred in
    # frame). They coincide when look_at is omitted, which is the historical
    # behaviour.
    look_pt = (center if look_at is None
               else np.asarray(look_at, dtype=np.float64))

    # Null-space arrays sized to the URDF's FULL movable-joint count —
    # pybullet silently disables null-space IK on size mismatch, and plain
    # DLS IK diverges badly on many-DOF chains (gripper URDFs have 20+
    # movable joints). Same trick as RRTToGoalPlanner._ik_null_space_kwargs.
    import pybullet as _pb

    movable = [j for j in range(client.getNumJoints(robot_id))
               if client.getJointInfo(robot_id, j)[2] != _pb.JOINT_FIXED]
    lowers, uppers, ranges = [], [], []
    for j in movable:
        info = client.getJointInfo(robot_id, j)
        lo, hi = info[8], info[9]
        if lo > hi:  # continuous joint
            lo, hi = -2 * np.pi, 2 * np.pi
        lowers.append(lo)
        uppers.append(hi)
        ranges.append(hi - lo)

    def solve(pos, target_quat=None, seed=None):
        """IK (optionally orientation-constrained) -> (pos_err, quat, q, R)."""
        if seed is not None:
            for j, qi in zip(joint_indices, seed):
                client.resetJointState(robot_id, j, float(qi))
        rest = [client.getJointState(robot_id, j)[0] for j in movable]
        args = [robot_id, ee_link, np.asarray(pos, dtype=np.float64).tolist()]
        if target_quat is not None:
            args.append(list(target_quat))
        sol = client.calculateInverseKinematics(
            *args,
            lowerLimits=lowers, upperLimits=uppers,
            jointRanges=ranges, restPoses=rest,
            maxNumIterations=512, residualThreshold=1e-10,
        )
        q = np.array(sol[: len(joint_indices)])
        for j, qi in zip(joint_indices, q):
            client.resetJointState(robot_id, j, float(qi))
        state = client.getLinkState(robot_id, ee_link,
                                    computeForwardKinematics=True)
        # link frame (4/5), matching pybullet IK + the planner's FK gate
        err = float(np.linalg.norm(np.asarray(state[4]) - np.asarray(pos)))
        quat = np.asarray(state[5])
        R = Rotation.from_quat(quat).as_matrix()
        return err, quat, q, R

    saved = [client.getJointState(robot_id, j)[0] for j in joint_indices]
    diag = {"best": None, "rejected_colliding": 0}

    def attempt(pos):
        """Best pose AT THIS EE POSITION. Returns (quat, q, up_err) for a
        solution passing position + aim + collision, or None if there is
        none. ``up_err`` is the achieved camera-up error (0 when no camera-up
        was requested), so the caller can tell a fully-satisfying result from
        one that only got the aim right."""
        err0, quat0, q0, R0 = solve(pos)
        if err0 > 0.01:
            return None
        if aim_axis_local is None:
            return (quat0, q0, 0.0)

        # Aim FROM THIS POSITION at the bunch: when the position is relaxed
        # sideways, the tool must re-point at the target, so the aim axis is
        # recomputed per candidate rather than reusing the nominal ray.
        aim_dir = look_pt - np.asarray(pos, dtype=np.float64)
        aim_dir = aim_dir / max(np.linalg.norm(aim_dir), 1e-9)

        def aim_err_deg(R):
            a = R @ axis_local
            return float(np.degrees(
                np.arccos(np.clip(a @ aim_dir, -1.0, 1.0))))

        # Wrist-flip seeds. Rolling the tool ~180 deg about the aim axis is
        # very nearly a half turn of the FINAL wrist joint, but null-space IK
        # pulls toward restPoses (= the seed), so seeded from the upright q0
        # it converges straight back to the upright branch and the camera-up
        # request is silently lost. Offering the flipped branch explicitly is
        # what makes an inverted-camera request actually reachable.
        _flip = []
        for _sgn in (1.0, -1.0):
            _qf = np.array(q0, dtype=np.float64)
            _qf[-1] = np.clip(_qf[-1] + _sgn * np.pi, arm_lo[-1], arm_hi[-1])
            _flip.append(_qf)
        seeds = [q0] + _flip + [rng.uniform(arm_lo, arm_hi)
                                for _ in range(int(ik_random_seeds))]
        if ik_seed:
            rng.shuffle(seeds)

        # Minimal rotation aligning the natural aim axis onto the bunch
        # direction — constant across the roll sweep, so hoisted out of it.
        a_world = R0 @ axis_local
        v = np.cross(a_world, aim_dir)
        s_ = np.linalg.norm(v)
        angle = float(np.arctan2(s_, a_world @ aim_dir))
        R_corr = (Rotation.identity() if s_ < 1e-9
                  else Rotation.from_rotvec(v / s_ * angle))

        # Absolute camera roll: solve for the spin about the aim direction
        # that takes the camera's image-up closest to `align_up_world`.
        # Without this the roll is measured from R0, an arbitrary IK branch —
        # a sideways R0 makes "+180" merely the opposite sideways.
        w_target = None
        base_roll_deg = 0.0
        if align_up_world is not None:
            w = np.asarray(align_up_world, dtype=np.float64)
            # image-up must be perpendicular to the aim axis; use the
            # component of the request that actually lies in that plane.
            w = w - (w @ aim_dir) * aim_dir
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-6:
                w /= w_norm
                w_target = w
                u0 = (R_corr * Rotation.from_matrix(R0)).apply(up_local)
                base_roll_deg = float(np.degrees(np.arctan2(
                    np.cross(u0, w) @ aim_dir, u0 @ w)))

        def up_err_deg(R):
            """Angle between the ACHIEVED camera image-up and the request."""
            if w_target is None:
                return 0.0
            u = R @ up_local
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            return float(np.degrees(
                np.arccos(np.clip(u @ w_target, -1.0, 1.0))))

        local_best = None  # aim-valid + collision-free, but wrong roll
        accepted = []  # (score, quat, q, up_err) passing EVERY hard gate
        # Preferred roll first, then the spread around it.
        for _roll_delta in (0, 20, -20, 45, -45, 90, -90, 135, -135, 180):
            roll_deg = base_roll_deg + roll_offset_deg + _roll_delta
            R_roll = Rotation.from_rotvec(aim_dir * np.radians(roll_deg))
            target = (R_roll * R_corr * Rotation.from_matrix(R0)).as_quat()
            for seed in seeds:
                err, quat, q, R = solve(pos, target_quat=target, seed=seed)
                aim = aim_err_deg(R)
                if err <= 0.005 and aim <= max_aim_error_deg:
                    if collision_fn is not None and collision_fn(q):
                        diag["rejected_colliding"] += 1
                        continue
                    # The target quat carries the requested roll, but IK is
                    # free to converge to a DIFFERENT roll that still points
                    # the aim axis at the bunch — position+aim alone would
                    # accept it and silently discard the camera-up request.
                    ue = up_err_deg(R)
                    if w_target is None or ue <= max_up_error_deg:
                        if score_fn is None:
                            return (quat, q, ue)
                        # RANK rather than take-first: several rolls/elbow
                        # branches reach the same viewpoint, and they differ a
                        # lot in how much foliage the arm buries itself in.
                        # Collect a handful, then stop — the search budget
                        # exists to FIND solutions on tricky bunches, not to
                        # exhaustively score easy ones.
                        accepted.append((float(score_fn(q)), quat, q, ue))
                        if len(accepted) >= int(ik_enough_candidates):
                            break
                        continue
                    if local_best is None or ue < local_best[2]:
                        local_best = (quat, q, ue)
                    continue
                b = diag["best"]
                if b is None or (aim, err) < (b[0], b[1]):
                    diag["best"] = (aim, err, quat, q)
            if len(accepted) >= int(ik_enough_candidates):
                break  # enough good options; stop sweeping rolls
        if accepted:
            accepted.sort(key=lambda t: t[0])
            diag["chosen_score"] = accepted[0][0]
            diag["n_accepted"] = len(accepted)
            return accepted[0][1], accepted[0][2], accepted[0][3]
        return local_best

    try:
        axis_local = None
        if aim_axis_local is not None:
            axis_local = np.asarray(aim_axis_local, dtype=np.float64)
            axis_local /= np.linalg.norm(axis_local)
        up_local = np.asarray(camera_up_axis_local, dtype=np.float64)
        up_local /= np.linalg.norm(up_local)

        # arm joint limits for random IK re-seeds (elbow-branch diversity)
        arm_lo = np.array([lowers[movable.index(j)] for j in joint_indices])
        arm_hi = np.array([uppers[movable.index(j)] for j in joint_indices])
        # ik_seed varies the random elbow/wrist re-seeds, so callers can ask
        # for a DIFFERENT solution at identical parameters (the tuner's
        # "resample IK" button). Seed 0 keeps the historical deterministic
        # ordering; any other seed also shuffles the fixed seeds (q0 + the two
        # wrist flips) into the random ones, since otherwise those are always
        # tried first and would keep returning the same solution.
        rng = np.random.default_rng(ik_seed)

        # Candidate EE positions, nominal first then increasing displacement.
        # Relaxing POSITION is preferred over relaxing ORIENTATION: the
        # camera roll is what downstream policies see in every frame, while a
        # few cm of standoff/offset is behaviourally irrelevant. So when the
        # requested camera-up cannot be met at the nominal point (typically
        # because the flipped wrist swings the gripper into the canopy), we
        # step the goal away rather than accept a sideways camera.
        cand = [pos]
        if align_up_world is not None and position_relax_m > 0.0:
            back = -direction
            e1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(direction, np.array([1.0, 0.0, 0.0]))
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(direction, e1)
            e2 /= np.linalg.norm(e2)
            dirs = [back, e2, -e2, e1, -e1,
                    (back + e2) / np.sqrt(2.0), (back - e2) / np.sqrt(2.0),
                    (back + e1) / np.sqrt(2.0), (back - e1) / np.sqrt(2.0)]
            step = max(position_relax_step_m, 1e-3)
            n_steps = int(np.floor(position_relax_m / step + 1e-9))
            for k in range(1, n_steps + 1):
                for u in dirs:
                    cand.append(pos + u * (k * step))

        fallback = None  # (displacement, pos, quat, q, up_err)
        for cpos in cand:
            got = attempt(cpos)
            if got is None:
                continue
            quat, q, ue = got
            if align_up_world is None or ue <= max_up_error_deg:
                if not np.allclose(cpos, pos):
                    logger.info(
                        "reachable_approach_pose: camera-up request needed a "
                        "%.0f mm EE translation; goal moved %s -> %s",
                        np.linalg.norm(cpos - pos) * 1000.0,
                        np.round(pos, 3).tolist(), np.round(cpos, 3).tolist(),
                    )
                return cpos, quat, q
            d = float(np.linalg.norm(cpos - pos))
            if fallback is None or ue < fallback[4]:
                fallback = (d, cpos, quat, q, ue)

        if fallback is not None and not allow_roll_fallback:
            # STRICT (default): a camera-up request is a requirement, not a
            # hint. Silently returning an upright/sideways pose when inverted
            # was asked for is worse than failing — the caller gets a goal
            # that looks fine and only shows up as inconsistent imagery much
            # later. Callers who genuinely prefer any-reachable-pose over
            # correct-roll can pass allow_roll_fallback=True.
            raise ValueError(
                f"no collision-free IK at {np.round(pos, 3).tolist()} achieved "
                f"the requested camera up within {max_up_error_deg:.0f} deg, "
                f"even allowing {position_relax_m*1000:.0f} mm of EE "
                f"translation (closest was {fallback[4]:.0f} deg off). Widen "
                f"position_relax_m / max_up_error_deg, or pass "
                f"allow_roll_fallback=True to accept a differently-rolled pose."
            )
        if fallback is not None:
            logger.warning(
                "reachable_approach_pose at %s: no collision-free IK achieved "
                "the requested camera up within %.0f deg, even allowing %.0f mm "
                "of EE translation; returning the closest (%.0f deg off). Tool "
                "roll differs from the request.",
                np.round(pos, 3).tolist(), max_up_error_deg,
                position_relax_m * 1000.0, fallback[4],
            )
            return fallback[1], fallback[2], fallback[3]

        best = diag["best"]
        if best is None:
            raise ValueError(
                f"standoff point {np.round(pos, 3).tolist()} unreachable "
                "(position-only IK did not converge)"
            )
        raise ValueError(
            f"no grasp-aligned IK at {np.round(pos, 3).tolist()} — best "
            f"aim error {best[0]:.0f} deg, pos error {best[1]*1000:.0f} mm, "
            f"{diag['rejected_colliding']} aligned candidate(s) rejected as colliding"
        )
    finally:
        for j, qi in zip(joint_indices, saved):
            client.resetJointState(robot_id, j, float(qi))


def approach_pose(
    bunch_center: np.ndarray | list,
    from_point: np.ndarray | list,
    standoff: float = 0.10,
    approach_axis: str = "z",
) -> tuple:
    """EE goal pose ``standoff`` meters short of the bunch, tool axis aimed
    at it.

    The goal position sits on the line from ``from_point`` (typically the
    robot base) to the bunch center, pulled back by ``standoff``; the
    orientation is a look-at frame whose ``approach_axis`` (+z for the
    Robotiq-on-UR5 tool convention) points at the bunch.

    Returns (pos(3,), quat(x,y,z,w)).
    """
    from scipy.spatial.transform import Rotation

    center = np.asarray(bunch_center, dtype=np.float64)
    origin = np.asarray(from_point, dtype=np.float64)
    to_bunch = center - origin
    dist = np.linalg.norm(to_bunch)
    if dist < 1e-9:
        raise ValueError("bunch center coincides with from_point")
    direction = to_bunch / dist
    pos = center - direction * standoff

    z = direction
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=1)  # columns = tool x/y/z in world
    if approach_axis == "x":
        rot = rot[:, [2, 1, 0]] * np.array([1, 1, -1])
    elif approach_axis == "y":
        rot = rot[:, [0, 2, 1]] * np.array([1, 1, -1])
    quat = Rotation.from_matrix(rot).as_quat()  # (x, y, z, w)
    return pos, quat
