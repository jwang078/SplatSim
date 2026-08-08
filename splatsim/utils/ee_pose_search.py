"""Task-space search over end-effector poses, with IK as a post-filter.

WHY THIS EXISTS (and why it is the right way round)
---------------------------------------------------
The older path (`grape_targets.reachable_approach_pose`) samples in
CONFIGURATION space — IK seeds x roll offsets — and evaluates the result in
task space. That couples "is this a good viewpoint" to "did IK happen to land
there", so the pose you get is whichever feasible solution turned up first,
the search cannot express *why* one viewpoint beats another, and a viewpoint
that is merely awkward for one IK branch is indistinguishable from one that
is genuinely bad.

Here the order is inverted, which is what the literature does:

  * Grasp synthesis — GPD (ten Pas et al. 2017), 6-DOF GraspNet (Mousavian
    et al. 2019), Contact-GraspNet (Sundermeyer et al. 2021) all GENERATE and
    SCORE 6-DOF poses from a point cloud, then check reachability afterwards.
  * Viewpoint / next-best-view planning — the NBV line (Connolly 1985; Scott
    et al. survey 2003) and agricultural viewpoint planning for fruit
    (Zaenker et al.; Burusa et al. on active vision in glasshouse crops)
    optimise a camera pose against a visibility/coverage objective, then ask
    whether the arm can achieve it.

Both reduce to: score poses over a point cloud, then filter by reachability.
That shared shape is the reason this module is task-agnostic — the objective
is a weighted sum of named terms, so "inspect a bunch" and "grasp it from
below" differ only in weights and in which approach directions are sampled,
not in machinery.

THE ONE REAL PITFALL
--------------------
Pure optimise-then-filter can return nothing: the argmax pose is often
unreachable, and you discover that only after committing to it. The standard
fixes, both used here, are to (a) keep a DIVERSE TOP-K rather than an argmax,
so the IK filter has alternatives, and (b) prune obviously-unreachable
regions before scoring. A full fix is a precomputed reachability/capability
map (Zacharias et al. 2007; Vahrenkamp et al.) used to bias sampling — a
natural next step, not implemented here.

Geometry note: the gripper is modelled as spheres rigidly attached to the EE
link, so the pose search is arm-agnostic and fast (KD-tree queries, no
physics engine in the inner loop). The ARM is deliberately ignored until the
IK stage, where a full-body collision check runs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from splatsim.utils.goal_pose import CAMERA_FORWARD_AXIS, CAMERA_UP_AXIS


# --------------------------------------------------------------- gripper model
def gripper_spheres(client, robot_id: int, ee_link: int, link_indices,
                    radii=None):
    """Sphere approximation of the gripper, expressed in the EE-LINK frame.

    Built by FK at the CURRENT gripper opening: each link contributes a sphere
    at its origin with the link's cross-sectional radius. Rigid w.r.t. the EE
    link, so a candidate EE pose maps them into the world with one transform —
    which is what makes scoring thousands of poses cheap.

    Returns ``(centers_local (N,3), radii (N,))``.
    """
    from splatsim.utils.soft_cost_field import link_radii as _link_radii

    if radii is None:
        radii = _link_radii(client, robot_id, link_indices)
    st = client.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos = np.asarray(st[4], dtype=np.float64)
    ee_rot = np.asarray(client.getMatrixFromQuaternion(st[5]),
                        dtype=np.float64).reshape(3, 3)
    centers = []
    for li in link_indices:
        p = np.asarray(client.getLinkState(robot_id, li,
                                           computeForwardKinematics=True)[4],
                       dtype=np.float64)
        centers.append(ee_rot.T @ (p - ee_pos))
    return np.asarray(centers), np.asarray(radii, dtype=np.float64)


# ------------------------------------------------------------------ pose model
def look_at_rotation(forward_world, up_hint_world, aim_axis_local,
                     up_axis_local):
    """Rotation putting ``aim_axis_local`` on ``forward_world`` and
    ``up_axis_local`` as close to ``up_hint_world`` as the first constraint
    allows. Returns a (3,3) rotation, or None if degenerate."""
    f = np.asarray(forward_world, dtype=np.float64)
    n = np.linalg.norm(f)
    if n < 1e-9:
        return None
    f = f / n
    up = np.asarray(up_hint_world, dtype=np.float64)
    up = up - (up @ f) * f          # only the component orthogonal to the aim
    if np.linalg.norm(up) < 1e-6:   # aim parallel to the up hint
        tmp = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        up = tmp - (tmp @ f) * f
    up = up / np.linalg.norm(up)
    # Local frame: aim axis -> f, up axis -> up.
    a = np.asarray(aim_axis_local, dtype=np.float64)
    a = a / np.linalg.norm(a)
    u = np.asarray(up_axis_local, dtype=np.float64)
    u = u - (u @ a) * a
    u = u / np.linalg.norm(u)
    local = np.stack([np.cross(u, a), u, a], axis=1)     # columns
    world = np.stack([np.cross(up, f), up, f], axis=1)
    return world @ local.T


def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` roughly-uniform unit vectors on the sphere (Fibonacci lattice).
    Uniform coverage matters: clustered samples silently bias which approach
    directions get explored."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


@dataclasses.dataclass
class SearchSpec:
    """What poses to consider and how to score them."""

    # --- sampling ---
    n_directions: int = 400          # approach directions on the sphere
    standoff_range: tuple = (0.08, 0.18)   # fingertip-to-target distance (m)
    n_standoffs: int = 3
    n_rolls: int = 1                 # >1 samples roll; 1 uses the up hint only
    # Restrict approach directions. (0,0,-1) with max_angle 60 gives
    # "from below", the structurally clear side of a hanging bunch; None =
    # all directions. This is the knob that turns inspection into grasping.
    direction_hint: tuple | None = None
    direction_max_angle_deg: float = 180.0

    # --- geometry ---
    aim_axis_local: tuple = CAMERA_FORWARD_AXIS
    up_axis_local: tuple = CAMERA_UP_AXIS
    camera_up_world: tuple | None = (0.0, 0.0, -1.0)
    tip_offset_m: float = 0.196
    fov_deg: float = 60.0

    # --- scoring weights (interpretable, all terms in [0,1]) ---
    w_visibility: float = 1.0        # cluster visible + unoccluded
    w_clearance: float = 1.0         # gripper not buried in vegetation
    w_approach: float = 0.7          # clear corridor behind the tool
    w_camera_up: float = 0.5         # image-up matches the request
    w_standoff: float = 0.3          # prefer the middle of the standoff range
    w_hard: float = 0.8              # margin from trellis / thick branches

    # --- feasibility prefilter (cheap, before scoring) ---
    base_xyz: tuple = (0.0, 0.0, -0.088)
    # EE distance from base (m). Upper bound is intentionally a little past
    # the nominal UR5 reach: it is a cheap PREFILTER whose only job is to
    # avoid scoring hopeless poses, and IK is the real arbiter. Too tight and
    # distant bunches (this scene has several past 1.2 m) get starved of
    # samples and report "infeasible" when they are merely under-sampled.
    reach_range: tuple = (0.25, 1.05)

    # --- selection ---
    top_k: int = 40
    nms_position_m: float = 0.05
    nms_angle_deg: float = 20.0


def sample_poses(target_center, spec: SearchSpec):
    """Candidate EE poses aimed at ``target_center``.

    Returns ``(positions (M,3), rotations (M,3,3), approach_dirs (M,3))``.
    The EE sits BEHIND the target along the approach direction by
    ``standoff + tip_offset``, so ``standoff`` is measured fingertip-to-target
    rather than wrist-to-target.
    """
    c = np.asarray(target_center, dtype=np.float64)
    dirs = fibonacci_directions(spec.n_directions)
    if spec.direction_hint is not None:
        h = np.asarray(spec.direction_hint, dtype=np.float64)
        h = h / np.linalg.norm(h)
        keep = np.degrees(np.arccos(np.clip(dirs @ h, -1, 1))) <= spec.direction_max_angle_deg
        dirs = dirs[keep]
    standoffs = np.linspace(spec.standoff_range[0], spec.standoff_range[1],
                            max(spec.n_standoffs, 1))
    up_hint = (np.asarray(spec.camera_up_world, dtype=np.float64)
               if spec.camera_up_world is not None else np.array([0.0, 0.0, 1.0]))

    base = np.asarray(spec.base_xyz, dtype=np.float64)
    lo, hi = spec.reach_range
    pos, rot, appr = [], [], []
    for d in dirs:                       # d points FROM the target outward
        for s in standoffs:
            p = c + d * (s + spec.tip_offset_m)
            r = float(np.linalg.norm(p - base))
            if not (lo <= r <= hi):      # cheap reachability prefilter
                continue
            rolls = ([None] if spec.n_rolls <= 1
                     else np.linspace(0, 2 * np.pi, spec.n_rolls, endpoint=False))
            for roll in rolls:
                hint = up_hint
                if roll is not None:
                    # rotate the up hint about the aim to sample roll
                    f = -d
                    ca, sa = np.cos(roll), np.sin(roll)
                    hint = (up_hint * ca + np.cross(f, up_hint) * sa
                            + f * (f @ up_hint) * (1 - ca))
                R = look_at_rotation(-d, hint, spec.aim_axis_local,
                                     spec.up_axis_local)
                if R is None:
                    continue
                pos.append(p); rot.append(R); appr.append(d)
    if not pos:
        return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 3))
    return np.asarray(pos), np.asarray(rot), np.asarray(appr)


# ---------------------------------------------------------------- scoring
def _occluded_fraction(eye, targets, veg_tree, clear_radius, n_steps=6):
    """Fraction of ``targets`` whose straight line from ``eye`` passes within
    ``clear_radius`` of vegetation — i.e. is occluded.

    Deliberately coarse (a few samples per ray). Exact visibility on a
    gaussian cloud is a rendering problem; what the objective needs is a
    monotone signal that prefers unobstructed views."""
    if len(targets) == 0:
        return 1.0
    ts = np.linspace(0.15, 0.9, n_steps)[:, None, None]
    pts = eye[None, None, :] + ts * (targets[None, :, :] - eye[None, None, :])
    d, _ = veg_tree.query(pts.reshape(-1, 3))
    hit = (d.reshape(n_steps, len(targets)) < clear_radius).any(axis=0)
    return float(hit.mean())


def score_pose(pos, rot, appr, clouds, spec: SearchSpec, target_pts):
    """Score one EE pose. Returns ``(total, terms_dict)`` with every term in
    [0,1] so the weights are directly comparable and the breakdown is
    inspectable — which is what makes tuning tractable."""
    aim = rot @ np.asarray(spec.aim_axis_local, dtype=np.float64)
    aim /= np.linalg.norm(aim)

    # --- visibility: target points inside the FOV cone, not occluded ---
    v = target_pts - pos
    dist = np.linalg.norm(v, axis=1)
    ang = np.degrees(np.arccos(np.clip((v / dist[:, None]) @ aim, -1, 1)))
    in_fov = ang <= spec.fov_deg * 0.5
    frac_fov = float(in_fov.mean())
    occ = _occluded_fraction(pos, target_pts[in_fov], clouds["occ_tree"],
                             clouds["occlusion_radius"])
    visibility = frac_fov * (1.0 - occ)

    # --- clearance: vegetation intruding into the gripper's own volume ---
    centers = pos + (rot @ clouds["grip_centers"].T).T
    n_intrude = 0
    for cen, rad in zip(centers, clouds["grip_radii"]):
        n_intrude += len(clouds["veg_tree"].query_ball_point(cen, rad))
    clearance = 1.0 / (1.0 + n_intrude / 50.0)

    # --- approach corridor: free space BEHIND the tool along the aim axis ---
    back = pos + aim[None, :] * (-np.linspace(0.02, 0.25, 6))[:, None]
    d_back, _ = clouds["veg_tree"].query(back)
    approach = float(np.clip(d_back.min() / 0.08, 0.0, 1.0))

    # --- camera-up agreement ---
    if spec.camera_up_world is None:
        cam_up = 1.0
    else:
        up = rot @ np.asarray(spec.up_axis_local, dtype=np.float64)
        want = np.asarray(spec.camera_up_world, dtype=np.float64)
        want = want / np.linalg.norm(want)
        cam_up = float(np.clip((up @ want + 1.0) / 2.0, 0.0, 1.0))

    # --- standoff: prefer the middle of the allowed band ---
    tip = pos + aim * spec.tip_offset_m
    s = float(np.linalg.norm(tip - target_pts.mean(axis=0)))
    lo, hi = spec.standoff_range
    mid = 0.5 * (lo + hi)
    standoff = float(np.clip(1.0 - abs(s - mid) / max(hi - lo, 1e-6), 0.0, 1.0))

    hard = hard_clearance_score(pos, rot, clouds)

    terms = {"visibility": visibility, "clearance": clearance,
             "approach": approach, "camera_up": cam_up, "standoff": standoff,
             "hard": hard}
    total = (spec.w_visibility * visibility + spec.w_clearance * clearance
             + spec.w_approach * approach + spec.w_camera_up * cam_up
             + spec.w_standoff * standoff + spec.w_hard * hard)
    total /= (spec.w_visibility + spec.w_clearance + spec.w_approach
              + spec.w_camera_up + spec.w_standoff + spec.w_hard)
    return total, terms


def select_diverse(pos, rot, scores, spec: SearchSpec):
    """Greedy non-maximum suppression in SE(3): best first, then skip anything
    within ``nms_position_m`` AND ``nms_angle_deg`` of something already kept.

    Diversity is the whole point — a top-K of near-identical poses gives the
    IK filter nothing to fall back on when the best one is unreachable."""
    order = np.argsort(-scores)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if np.linalg.norm(pos[i] - pos[j]) < spec.nms_position_m:
                cos = (np.trace(rot[i].T @ rot[j]) - 1.0) / 2.0
                if np.degrees(np.arccos(np.clip(cos, -1, 1))) < spec.nms_angle_deg:
                    ok = False
                    break
        if ok:
            keep.append(int(i))
        if len(keep) >= spec.top_k:
            break
    return np.asarray(keep, dtype=int)


def gripper_hits_hard(pos, rot, clouds, margin: float = 0.0) -> bool:
    """True if the gripper at this EE pose intersects HARD geometry.

    Hard geometry (trellis wires + thick branches) is rigid: unlike foliage
    it cannot be pushed aside, so this is a feasibility GATE, not a cost.
    Checked during pose scoring because the vegetation cloud does not
    represent it — the hard mesh was deliberately excluded from the soft
    field, so a gripper impaled on a trellis wire otherwise scores as
    perfectly clear and only fails much later at the IK stage (or, if the arm
    happens to reach it, not at all).
    """
    tree = clouds.get("hard_tree")
    if tree is None:
        return False
    centers = pos + (rot @ clouds["grip_centers"].T).T
    for cen, rad in zip(centers, clouds["grip_radii"]):
        if tree.query_ball_point(cen, rad + margin, return_length=True):
            return True
    return False


def hard_clearance_score(pos, rot, clouds, full_at: float = 0.05) -> float:
    """Margin to hard geometry in [0,1] — 0 touching, 1 at ``full_at`` metres.

    A gate alone makes every non-colliding pose look equally good, including
    ones a millimetre off a trellis wire that any IK or tracking error will
    turn into a real collision. This prefers standing off from rigid stuff.
    """
    tree = clouds.get("hard_tree")
    if tree is None:
        return 1.0
    centers = pos + (rot @ clouds["grip_centers"].T).T
    d, _ = tree.query(centers)
    return float(np.clip((d - clouds["grip_radii"]).min() / full_at, 0.0, 1.0))


def build_clouds(grape_pts, veg_pts, grip_centers, grip_radii,
                 occluder_pts=None, hard_pts=None,
                 occlusion_radius: float = 0.02):
    """Bundle the scene point clouds + gripper model the scorer needs.

    Two separate clouds, deliberately:
      veg_tree  everything solid, used for GRIPPER CLEARANCE and the approach
                corridor — the target's own fruit belongs here, since burying
                the gripper in it is still bad.
      hard_tree trellis + thick branches, RIGID. A feasibility gate (and a
                margin term), never merely a cost — it cannot be pushed
                aside, and it is absent from the soft field by construction.
      occ_tree  occluders only, used for VISIBILITY. The target bunch must be
                removed from it: otherwise the fruit you are trying to see
                counts as blocking the view of itself, and every pose scores
                near-zero visibility regardless of how good it is.
    """
    veg = np.asarray(veg_pts, dtype=np.float64)
    occ = veg if occluder_pts is None else np.asarray(occluder_pts, dtype=np.float64)
    hard = None if hard_pts is None else np.asarray(hard_pts, dtype=np.float64)
    return {
        "hard_pts": hard,
        "hard_tree": None if hard is None or not len(hard) else cKDTree(hard),
        "grape_pts": np.asarray(grape_pts, dtype=np.float64),
        "veg_pts": veg,
        "veg_tree": cKDTree(veg),
        "occ_tree": cKDTree(occ) if len(occ) else cKDTree(np.zeros((1, 3)) + 1e6),
        "grip_centers": np.asarray(grip_centers, dtype=np.float64),
        "grip_radii": np.asarray(grip_radii, dtype=np.float64),
        "occlusion_radius": float(occlusion_radius),
    }
