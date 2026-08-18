import numpy as np
import math
import os
import pybullet as p
import pybullet_data
import time
import itertools

import pybullet as p
import pybullet_data
from pybullet_planning import RED, smooth_path
from pybullet_planning import Pose
from pybullet_planning import get_movable_joints, create_box, set_pose, get_extend_fn
from pybullet_planning import get_sample_fn, get_distance_fn, birrt
import numpy as np
import time

# Optional for smooth interpolation
try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


def load_cuboids(cuboid_path):
    data = np.load(cuboid_path, allow_pickle=True)
    R = data['R']
    cuboids = data['cuboids']

    # The cuboids were saved in pybullet-space, not splat space
    cuboid_points = cuboids

    # # Apply R to each point in cuboids
    # cuboid_points = np.array([
    #     ((R @ np.array([x0, y0, z0, 1]).T)[:3],
    #     (R @ np.array([x1, y1, z1, 1]).T)[:3])
    #     for (x0, x1, y0, y1, z0, z1) in cuboids
    # ])
    # # Sort order
    # cuboid_points = np.array([
    #     (min(point[0][0], point[1][0]), max(point[0][0], point[1][0]),
    #      min(point[0][1], point[1][1]), max(point[0][1], point[1][1]),
    #      min(point[0][2], point[1][2]), max(point[0][2], point[1][2]))
    #      for point in cuboid_points
    # ])
    # Convert to center + size for length, width, height
    cuboid_bboxes = np.array([
        [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
         for (x0, x1, y0, y1, z0, z1) in cuboid_points
    ])
    return cuboid_bboxes

def world_to_local(link_world_pos, link_world_orn, point_world):
    """
    Convert world point to link local coordinates using pybullet transforms.
    """
    # invert transform (link_world_pos, link_world_orn)
    inv_pos, inv_orn = p.invertTransform(link_world_pos, link_world_orn)
    local_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, point_world, [0,0,0,1])
    return local_pos

def contact_tuple_debug(pt):
    # Useful for debugging getClosestPoints tuple layout
    print("closestPoint tuple:", pt)
    # Common indices (may vary): 5=posOnA,6=posOnB,7=normalOnB,8=distance
    try:
        print("posA", pt[5], "posB", pt[6], "normalOnB", pt[7], "dist", pt[8])
    except Exception:
        pass

###########################
# Utility / Collision API #
###########################

_COLLISION_CLEARANCE = 0.01  # 1 cm clearance for all collision checks


# ---------------------------------------------------------------------------
# Multi-client support
# ---------------------------------------------------------------------------
#
# All PyBullet calls in this module need a `physicsClientId`. Historically the
# functions used the implicit default (client 0), which is fine when only one
# server is connected — but breaks down in setups where two PyBullet clients
# coexist (for example, lerobot's shared autonomy wrapper running in the same
# process as a local SplatSim simulator). To keep older callers working, we
# accept an optional `physics_client_id=` kwarg on every function and resolve
# it via `_resolve_client_id`. Modules that own a single client (e.g. the
# SplatSim server) call `set_default_client_id(...)` once at startup so their
# subsequent calls don't have to thread the id; cross-client callers
# (e.g. the wrapper) pass `physics_client_id=` explicitly.

_DEFAULT_CLIENT_ID: int = 0  # PyBullet's implicit default client


def set_default_client_id(client_id: int) -> None:
    """Set the default ``physicsClientId`` used by this module.

    Called once by long-lived single-client setups (SplatSim's
    PybulletRobotServerBase). Cross-client callers should pass
    ``physics_client_id=`` to each function instead.
    """
    global _DEFAULT_CLIENT_ID
    _DEFAULT_CLIENT_ID = int(client_id)


def _resolve_client_id(physics_client_id):
    return _DEFAULT_CLIENT_ID if physics_client_id is None else int(physics_client_id)

def check_links_in_collision(robot_id, joint_indices, q, obstacle_ids, link_indices_to_check=None, verbose=False, obstacle_names=None, self_collision_clearance=0.0, skip_pairs=None, obstacle_clearance=None, physics_client_id=None, return_kind=False, self_collision_skip_pairs=None, self_collision_check_adjacent_pairs=None):
    """
    Single source-of-truth collision checker.

    Checks the robot at configuration q (or the current state if q is None) against obstacles and itself.
      1. Each link in link_indices_to_check against every obstacle (obstacle_clearance, default 1 cm).
      2. Self-collision between all non-adjacent link pairs in link_indices_to_check (0 clearance by default).

    Args:
        robot_id: PyBullet body ID of the robot.
        joint_indices: Movable joint indices (used to set configuration).
        q: Joint configuration to check. If provided, the robot is moved to q for
            the check and RESTORED to its prior state (position + velocity +
            position-hold) on `joint_indices` before returning — the call is
            side-effect-free on those joints (the gripper is left open, as
            set_robot_joint_positions opens it for these demos). If None, uses
            the robot's current joint state (no teleport, nothing to restore).
        obstacle_ids: List of PyBullet body IDs to treat as obstacles.
        link_indices_to_check: Links to check. None = all links (base link -1 + all joints).
        verbose: If True, print the first collision found.
        obstacle_names: Optional dict mapping body ID -> name string for readable verbose output.
        self_collision_clearance: Distance threshold for self-collision checks (default 0.0 = actual intersection only).
            Use 0.0 to avoid false positives when arm links are legitimately close (e.g. IK solutions).
        skip_pairs: Optional set of (robot_link_index, obstacle_body_id) tuples to skip.
            Used to exclude known always-touching pairs (e.g. shoulder_link vs table).
        self_collision_skip_pairs: Optional iterable of (link_a, link_b) tuples to
            skip in the SELF-collision check (independent of obstacles). Used to
            exclude non-adjacent link pairs that the URDF geometry places
            structurally close (e.g. UR robot's base_link(0) vs upper_arm_link(2),
            naturally ~4 mm apart due to the shoulder bracket). Without this,
            any non-zero `self_collision_clearance` falsely flags every valid
            joint config. Pairs are compared in BOTH orders ((a,b) == (b,a))
            so the caller doesn't need to canonicalize.
        obstacle_clearance: Distance threshold for obstacle checks. Defaults to _COLLISION_CLEARANCE (1 cm).
            Pass 0.0 to detect only actual penetration.
        return_kind: If False (default), returns a bool — keeps backward-compat
            with the ~9 existing RRT callers that use this as a truth test.
            If True, returns `(in_collision: bool, kind: str | None)` where
            `kind` is "obstacle" or "self" on hit, None otherwise. Used by
            the eval-time env metrics dict to record WHY an episode terminated.

    Returns:
        bool (default) or (bool, str | None) (when return_kind=True).
        - bool: True if any collision detected.
        - kind: "obstacle" for robot-vs-obstacle, "self" for self-collision,
                None for no collision. Reports the FIRST match found; with
                obstacles checked before self-collisions, "obstacle" takes
                precedence when both happen on the same query.
    """
    cid = _resolve_client_id(physics_client_id)
    if obstacle_clearance is None:
        obstacle_clearance = _COLLISION_CLEARANCE

    # When a configuration q is provided we mutate the robot to check it. Snapshot
    # the joints being moved so the check is side-effect-free: this runs on the
    # LIVE shared robot thousands of times per plan, and leaving it at the last
    # checked q silently corrupts the robot for every subsequent caller (which is
    # exactly the class of bug that bit randomize_ee_pose). q=None means "check
    # the current state" — nothing to set or restore.
    #
    # Kinematic teleport ONLY: `getClosestPoints` reads link poses from the
    # solver directly, so we don't need `stepSimulation` or motor control
    # commands here. Previously this called `set_robot_joint_positions` which
    # did `resetJointState` + `setJointMotorControl2` + `open_gripper` + a
    # full `p.stepSimulation()`, then followed up with another stepSimulation
    # here — two physics steps at 5-20 ms each per query, dominating the
    # cost of `_get_random_collision_free_q`'s inner loop (env.reset spent
    # seconds looking for a collision-free start config). Bare `resetJointState`
    # is the same "snap kinematically" primitive that `teleport_joint_state`
    # uses; matches its semantics without needing the SplatSimObject wrapping.
    _saved_joint_states = None
    if q is not None:
        _saved_joint_states = p.getJointStates(robot_id, joint_indices, physicsClientId=cid)
        for idx, qi in zip(joint_indices, q):
            p.resetJointState(robot_id, idx, float(qi), physicsClientId=cid)

    try:
        if link_indices_to_check is None:
            link_indices_to_check = list(range(-1, p.getNumJoints(robot_id, physicsClientId=cid)))

        def _robot_link_name(link_i):
            if link_i == -1:
                return "base_link(-1)"
            info = p.getJointInfo(robot_id, link_i, physicsClientId=cid)
            return f"{info[12].decode('utf-8')}({link_i})"

        def _obs_name(obs):
            if obstacle_names and obs in obstacle_names:
                return f"{obstacle_names[obs]}(id={obs})"
            return str(obs)

        # Check robot links against obstacles.
        # linkIndexB is intentionally omitted so PyBullet checks all links of the obstacle body,
        # not just its base link (-1). This matters for multi-link obstacle bodies (splat objects, boxes).
        for link_i in link_indices_to_check:
            for obs in obstacle_ids:
                if skip_pairs and (link_i, obs) in skip_pairs:
                    continue
                pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=obstacle_clearance,
                                         linkIndexA=link_i, physicsClientId=cid)
                if len(pts) > 0:
                    if verbose:
                        print(f"Collision: robot {_robot_link_name(link_i)} vs obstacle {_obs_name(obs)}")
                    return (True, "obstacle") if return_kind else True

        # Pre-normalize the self-collision skip pairs into a frozenset of
        # frozensets so (a,b) and (b,a) lookups both hit. Cheap (small N) and
        # done once per call so the hot loop just does set membership.
        _self_skip = None
        if self_collision_skip_pairs:
            _self_skip = {frozenset((int(a), int(b))) for a, b in self_collision_skip_pairs}
        # `self_collision_check_adjacent_pairs`: force-INCLUDE these adjacent
        # (parent-child) pairs in the self-collision check. Default: skip all
        # adjacent pairs (correct when parent-child geometry legitimately touches
        # at the joint pivot — e.g., small_engine's UR5+Robotiq URDF). Robots
        # whose extreme joint angles can fold a child link's BODY onto its
        # parent's (e.g., the planar 3-DOF arm at |joint_2| ≈ π) list those
        # pairs here so the check catches the fold-over case.
        _check_adjacent = None
        if self_collision_check_adjacent_pairs:
            _check_adjacent = {
                frozenset((int(a), int(b))) for a, b in self_collision_check_adjacent_pairs
            }

        # Check self-collisions between non-adjacent link pairs (plus the
        # whitelisted-adjacent pairs from `_check_adjacent`).
        for a, b in itertools.combinations(link_indices_to_check, 2):
            if _self_skip is not None and frozenset((a, b)) in _self_skip:
                continue
            if are_adjacent_links(robot_id, a, b, physics_client_id=cid):
                # Adjacent by URDF topology. Default: skip (natural joint-pivot
                # overlap). Override: caller explicitly listed this pair.
                if _check_adjacent is None or frozenset((a, b)) not in _check_adjacent:
                    continue
            if len(p.getClosestPoints(robot_id, robot_id, self_collision_clearance, linkIndexA=a, linkIndexB=b, physicsClientId=cid)) > 0:
                if verbose:
                    print(f"Self-collision: robot {_robot_link_name(a)} vs {_robot_link_name(b)}")
                return (True, "self") if return_kind else True

        return (False, None) if return_kind else False
    finally:
        # Restore the joints we moved to their pre-check state (position +
        # velocity) and re-apply POSITION_CONTROL at the pre-check position
        # so the robot ends this call exactly as it entered. Only the arm
        # `joint_indices` are restored; the gripper is untouched by the
        # kinematic-teleport path above. (Historically the pose-set used
        # `set_robot_joint_positions` which forced the gripper open — that
        # side-effect is intentionally gone now; the check reflects actual
        # gripper state.)
        if _saved_joint_states is not None:
            for idx, st in zip(joint_indices, _saved_joint_states):
                p.resetJointState(robot_id, idx, st[0], st[1], physicsClientId=cid)
                p.setJointMotorControl2(
                    robot_id, idx, p.POSITION_CONTROL,
                    targetPosition=st[0], force=150, maxVelocity=3.14,
                    physicsClientId=cid,
                )


def state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=None, link_indices_to_check=None, verbose=True):
    """Deprecated: use check_links_in_collision instead. Kept for backwards compatibility."""
    return check_links_in_collision(
        robot_id, joint_indices, q, obstacle_ids,
        link_indices_to_check=link_indices_to_check,
        verbose=verbose,
    )

def get_movable_joints(robot_id):
    """Return list of joint indices for revolute/continuous/prismatic joints that we consider movable."""
    n = p.getNumJoints(robot_id)
    joints = []
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        jtype = info[2]
        # 0 = revolute, 1 = prismatic, 2 = planar, 3 = fixed, 4 = floating, 5 = fixed? (varies)
        # We'll accept revolute (0) and prismatic (1) and continuous (-1 sometimes). Skip fixed (3).
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC): #, p.JOINT_CONTINUOUS):
            joints.append(i)
    return joints

def get_joint_limits(robot_id, joint_indices):
    """Get lower and upper limits for provided joints; if limits are invalid, use default (-pi, pi)."""
    lowers = []
    uppers = []
    for j in joint_indices:
        info = p.getJointInfo(robot_id, j)
        lower = info[8]
        upper = info[9]
        # If limits are huge or equal, fallback to -pi..pi
        if lower > upper or abs(upper - lower) < 1e-6:
            lower, upper = -math.pi, math.pi
        lowers.append(lower)
        uppers.append(upper)
    return np.array(lowers), np.array(uppers)

def set_robot_joint_positions(robot_id, joint_indices, q, hold=True, physics_client_id=None):
    cid = _resolve_client_id(physics_client_id)
    for idx, qi in zip(joint_indices, q):
        p.resetJointState(robot_id, idx, qi, physicsClientId=cid)
        if hold:
            p.setJointMotorControl2(
                robot_id, idx, p.POSITION_CONTROL,
                targetPosition=qi, force=150, maxVelocity=3.14,
                physicsClientId=cid,
            )
    # Always assume that the robot gripper is open in these demos
    open_gripper(robot_id, physics_client_id=cid)
    p.stepSimulation(physicsClientId=cid)

def min_distance_to_obstacles(robot_id, joint_indices, q, obstacle_ids, link_indices_to_check=None, max_dist=5.0):
    """Return minimum distance between robot (at q) and the set of obstacles (useful for soft cost)."""
    set_robot_joint_positions(robot_id, joint_indices, q)
    if link_indices_to_check is None:
        link_indices = list(range(0, p.getNumJoints(robot_id)))
    else:
        link_indices = link_indices_to_check

    min_d = max_dist
    for link_i in link_indices:
        for obs in obstacle_ids:
            pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=max_dist, linkIndexA=link_i, linkIndexB=-1)
            for pt in pts:
                d = pt[8]  # contactDistance
                if d < min_d:
                    min_d = d
                    if min_d <= 0.0:
                        return min_d
    return min_d


###########################
# Utilities: Time parametrization / spline
###########################

def joints_to_trajectory(path, total_time=5.0, use_cubic_spline=True):
    """
    path: list of joint vectors (M x DOF)
    Returns function q(t) for t in [0, total_time] sampled discretely and a discrete array of samples.
    If scipy is available, uses cubic spline interpolation per joint.
    """
    M = len(path)
    DOF = len(path[0])
    times = np.linspace(0, total_time, M)
    path_arr = np.array(path)  # M x DOF
    if use_cubic_spline and SCIPY_AVAILABLE and M >= 4:
        splines = [CubicSpline(times, path_arr[:, j], bc_type='clamped') for j in range(DOF)]
        def sample_traj(n_samples=100):
            ts = np.linspace(0, total_time, n_samples)
            qs = np.stack([spl(ts) for spl in splines], axis=1)  # n x DOF
            return ts, qs
        return sample_traj
    else:
        def sample_traj(n_samples=100):
            ts = np.linspace(0, total_time, n_samples)
            qs = []
            for t in ts:
                s = t / total_time * (M - 1)
                i = int(np.floor(s))
                alpha = s - i
                if i >= M - 1:
                    q = path_arr[-1].copy()
                else:
                    q = (1 - alpha) * path_arr[i] + alpha * path_arr[i + 1]
                qs.append(q)
            return ts, np.array(qs)
        return sample_traj

def setup_env(args, robot_base_position, use_old_walls=False, use_obstacles=True):
    if args.gui:
        cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load plane + obstacles here for demo; user should load their 30-300 cuboids and collect their body ids
    plane = p.loadURDF("plane.urdf")

    if use_old_walls:
        # place a wall in -0.4 at x axis using plane.urdf
        # wall is perpendicular to the plane
        quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        wall = p.loadURDF("plane.urdf", [-0.4, 0, 0.0], quat)
    else:
        quat = p.getQuaternionFromEuler([-np.pi/2, np.pi / 2, 0])
        wall = p.loadURDF("plane.urdf", [0.0, -0.4, 0.0], quat)

    if use_obstacles:
        cuboid_bboxes = load_cuboids(args.cuboids_fn)
    else:
        cuboid_bboxes = None

    # load robot
    flags = p.URDF_USE_INERTIA_FROM_FILE
    robot_id = p.loadURDF(args.urdf, useFixedBase=True, flags=flags, basePosition=robot_base_position)

    # get joints
    joint_indices = get_movable_joints(robot_id)
    if len(joint_indices) != 7:
        print("Warning: detected movable joints:", len(joint_indices), "expected 6 (no dof for gripper) .")
        print("taking the first 7")
        joint_indices = joint_indices[:6]

    ll, ul = get_joint_limits(robot_id, joint_indices)

    obstacle_ids = []
    if use_obstacles:
        for cuboid_bbox in cuboid_bboxes:
            cx, cy, cz, lx, ly, lz = cuboid_bbox
            obs = create_box(lx, ly, lz, color=RED)
            set_pose(obs, Pose(point=[cx, cy, cz]))
            obstacle_ids.append(obs)
    obstacle_ids.append(plane)
    obstacle_ids.append(wall)

    return ll, ul, obstacle_ids, robot_id, joint_indices

def get_random_joint_angles_without_collision(robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits, max_tries=10000, verbose=True, link_indices_to_check=None, skip_pairs=None, self_collision_clearance=0.0, self_collision_skip_pairs=None) -> np.ndarray:
    """Sample a random collision-free joint config.

    Two-part collision contract mirroring `check_links_in_collision`:
      * `skip_pairs` — (robot_link, obstacle_body_id) pairs to skip in
        robot-vs-obstacle checks (per-obstacle skip list).
      * `self_collision_skip_pairs` + `self_collision_clearance` —
        non-adjacent robot-link pair skip list and near-contact
        threshold for self-collision. When the caller uses a non-zero
        threshold (e.g. env's reset-time `check_able_to_solve` under
        the new `TrajectoryGenModeConfig.self_collision_clearance`),
        the skip list MUST be forwarded — otherwise structurally-close
        URDF pairs (Robotiq inner_finger/inner_knuckle mesh overlap,
        UR base/upper_arm) trip on every sample and the caller hangs
        after `max_tries` failures.
    """
    sample_fn = get_sample_fn(robot_id, joint_indices)
    for _ in range(max_tries):
        q = sample_fn()
        if not check_links_in_collision(
            robot_id,
            joint_indices,
            q,
            obstacle_ids,
            link_indices_to_check=link_indices_to_check,
            verbose=verbose,
            skip_pairs=skip_pairs,
            self_collision_clearance=self_collision_clearance,
            self_collision_skip_pairs=self_collision_skip_pairs,
        ):
            return np.array(q)
    raise RuntimeError("Failed to find collision-free joint angles after many tries")

def check_self_collision(robot_id, joint_indices, distance=0.0):
    """
    Returns True if any self-collision is detected.
    Note: for some reason, this always returns true
    """
    for linkA_i in range(len(joint_indices)):       # -1 = base link
        for linkB_i in range(linkA_i + 1, len(joint_indices)):
            linkA = joint_indices[linkA_i]
            linkB = joint_indices[linkB_i]

            # Skip adjacent links (they are usually connected by joints)
            if are_adjacent_links(robot_id, linkA, linkB):
                continue
            pts = p.getClosestPoints(robot_id, robot_id, distance, linkIndexA=linkA, linkIndexB=linkB)
            if len(pts) > 0:
                return True
    return False

_GRIPPER_LINK_START = 7  # Links 7+ are gripper links; arm links are 0-6 inclusive

# Cache: {(client_id, robot_id, min_link, max_link): is_adjacent_bool}.
# URDF adjacency is a purely-topological property that doesn't change
# with joint state — so a single per-(robot, client) lookup can be
# reused across every collision check for the life of the process. The
# uncached path was calling `p.getJointInfo` TWICE per non-adjacent
# pair PER collision check, and `check_links_in_collision` iterates
# ~190 non-adjacent pairs per query — that's 380 API round-trips per
# query just to filter adjacency, which the audit script bypasses
# entirely (it pre-filters by index enumeration). Caching turns 380
# API calls per query into 190 dict hits after the first query fills
# the cache.
_ADJACENCY_CACHE: dict[tuple[int, int, int, int], bool] = {}


def are_adjacent_links(robot_id, linkA, linkB, physics_client_id=None):
    """
    Returns True if the link pair should be skipped for self-collision checking.
    Two cases:
      1. Directly connected links (parent-child relationship).
      2. Both links are gripper links (joint index >= 7) — gripper geometry
         overlaps by design so any intra-gripper pair is excluded.

    Result is cached in `_ADJACENCY_CACHE` after the first (cid, robot_id,
    linkA, linkB) tuple is resolved. Adjacency is a URDF-topology property
    that doesn't change while the robot body lives; a fresh `loadURDF`
    call gets a fresh `robot_id` so cache staleness across robot reloads
    is impossible by key construction.
    """
    cid = _resolve_client_id(physics_client_id)
    # Both gripper links: always skip. Fast-path — no cache lookup needed
    # since the check is O(1) integer comparison anyway.
    if linkA >= _GRIPPER_LINK_START and linkB >= _GRIPPER_LINK_START:
        return True
    if linkA == -1 or linkB == -1:
        return False
    # Canonical (low, high) key so callers can pass either order.
    a, b = (linkA, linkB) if linkA < linkB else (linkB, linkA)
    key = (cid, robot_id, a, b)
    cached = _ADJACENCY_CACHE.get(key)
    if cached is not None:
        return cached
    parentA = p.getJointInfo(robot_id, linkA, physicsClientId=cid)[16]
    parentB = p.getJointInfo(robot_id, linkB, physicsClientId=cid)[16]
    result = (parentA == linkB) or (parentB == linkA)
    _ADJACENCY_CACHE[key] = result
    return result

def _make_uniform_sample_fn(lower_limits, upper_limits):
    """Self-contained sample_fn that doesn't query PyBullet (so it doesn't
    care which client is the default). Returns uniform random configurations
    within the supplied limits."""
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)

    def fn():
        return tuple(np.random.uniform(lower, upper))
    return fn


def _make_l2_distance_fn():
    """Self-contained distance_fn (Euclidean in joint space)."""
    def fn(q1, q2):
        return float(np.linalg.norm(np.asarray(q2) - np.asarray(q1)))
    return fn


def _make_linear_extend_fn(resolutions):
    """Self-contained extend_fn that linearly interpolates between two configs
    at the supplied per-joint resolutions. No PyBullet calls (so client-id
    agnostic). Yields intermediate configurations."""
    resolutions = np.asarray(resolutions, dtype=np.float64)

    def fn(q1, q2):
        q1a = np.asarray(q1, dtype=np.float64)
        q2a = np.asarray(q2, dtype=np.float64)
        diff = q2a - q1a
        n_steps = max(int(np.ceil(np.max(np.abs(diff) / resolutions))), 1)
        for i in range(1, n_steps + 1):
            yield tuple(q1a + diff * (i / n_steps))
    return fn


###########################
# Cost-aware (soft-cost / T-RRT) planning
###########################
#
# The binary planning stack above treats the world as free/colliding. Scenes
# with PUSHABLE geometry (vine foliage, twigs, grapes) add a continuous
# soft-cost field on top: brushing it is allowed but should be avoided.
# Historically the cost only entered as a POST-HOC candidate score
# (RRTToGoalPlanner._score_candidate) — every candidate was still GENERATED
# cost-blind, and `birrt`'s check_direct + shortcut smoothing collapse paths
# onto the straight (often high-cost) route, so the score had little to pick
# from. The pieces below make generation itself cost-aware:
#
#   * `cost_aware_birrt`  — bidirectional RRT-Connect with a T-RRT-style
#     stochastic transition test (Jaillet et al. 2010): every tree extension
#     step must pass `collision_fn` AND a cost-uphill filter with adaptive
#     temperature, so trees preferentially grow through low-cost space while
#     retaining probabilistic completeness (temperature rises after repeated
#     rejections, so constrained/high-cost corridors — e.g. the grasp goal
#     inside the canopy — remain reachable).
#   * `cost_aware_smooth_path` — random-shortcut smoothing that accepts a
#     shortcut only if it does not INCREASE the path's cost integral;
#     plain `smooth_path` would happily straighten a cost-avoiding detour
#     right back through the canopy.
#   * `elastic_smooth_path(config_cost_fn=...)` — the corner-rounding
#     relaxation gains the same guard on each midpoint-pull.
#
# All of it is opt-in via a `config_cost_fn(q) -> float` callable (None =
# exact historical behavior); RRTToGoalPlanner passes one only when
# `soft_cost_mode == "guided"` and a field is loaded, so binary-obstacle
# envs (small_engine, planar_3joint, ...) are untouched.

# T-RRT transition-test defaults. Cost fields are normalized to max=1, so
# these are in "field units": T_INIT is the uphill cost step accepted with
# probability ~e^-1 at start; temperature heats (*= alpha) after NFAIL_MAX
# consecutive rejections and cools proportionally to accepted uphill steps.
# MAX_TIME (seconds) bounds each attempt's wall clock: a failing cost-aware
# attempt is FAR more expensive per iteration than binary birrt (cost lookup
# on every extension step + more iterations), and the planner's retry ladder
# (IK candidates x path attempts x restarts) multiplies it — without the
# bound a hard scene ground for HOURS in the failure path (observed
# 2026-07-29 on the vine bench). On timeout/failure get_rrt_plan falls back
# to plain binary birrt, so guided mode degrades to score-mode behavior
# instead of stalling.
_TRRT_DEFAULTS = dict(
    # 400 iterations: after the transition-first ordering + vectorized NN
    # (2026-07-30) a FULL-BUDGET failure costs ~0.4-1 s wall, so iterations
    # (not max_time) are the binding constraint — at 150 the vine bench still
    # fell back to binary on several candidates that a longer search
    # connects. max_time stays the hard backstop.
    max_iterations=400,
    max_time=30.0,
    # t_init calibration (vine bench, re-measured 2026-08-01): t_init is the
    # uphill cost step accepted with probability ~e^-1 at start, so it wants
    # to sit near the MEDIAN uphill step of the active cost function —
    # rejecting roughly the worse half at start, with cooling tightening it
    # from there.
    #
    # The previous value (0.005) was calibrated against the old cost function
    # (centerline sampling, MEAN reduction), whose per-step uphill deltas ran
    # ~0.001-0.01. The current cost function (surface rings + MAX reduction —
    # see RRTToGoalPlanner._config_soft_cost_points) is ~26x less diluted, and
    # its measured deltas are p50=0.027, p75=0.051, p90=0.111. Leaving t_init
    # at 0.005 against those deltas would reject essentially every uphill step
    # (exp(-0.027/0.005) ~ 0.5%) and freeze both trees.
    t_init=0.027,
    t_min=1e-6,
    alpha=2.0,
    nfail_max=10,
)


class _CostNode:
    """Tree node for cost_aware_birrt: config + parent link + cached cost."""

    __slots__ = ("q", "parent", "cost")

    def __init__(self, q, parent=None, cost=0.0):
        self.q = np.asarray(q, dtype=np.float64)
        self.parent = parent
        self.cost = float(cost)

    def retrace(self):
        seq, node = [], self
        while node is not None:
            seq.append(node.q)
            node = node.parent
        return seq[::-1]


class _CostTree:
    """Node list + growing numpy buffer of configs for O(n)-vectorized
    nearest-neighbor. The per-node-Python-lambda `min(tree, key=...)` scan
    was a measurable slice of cost_aware_birrt wall time once trees reach
    hundreds of nodes (2 NN scans per iteration x 150 iterations); one
    `argmin` over a contiguous array is ~100x cheaper per scan."""

    __slots__ = ("nodes", "_buf", "_n")

    def __init__(self, root: _CostNode):
        self.nodes = [root]
        self._buf = np.empty((64, root.q.size), dtype=np.float64)
        self._buf[0] = root.q
        self._n = 1

    def __len__(self):
        return self._n

    def add(self, node: _CostNode) -> None:
        if self._n == self._buf.shape[0]:
            self._buf = np.concatenate([self._buf, np.empty_like(self._buf)])
        self._buf[self._n] = node.q
        self.nodes.append(node)
        self._n += 1

    def nearest(self, target: np.ndarray) -> _CostNode:
        """Joint-space-L2 nearest node (matches _make_l2_distance_fn)."""
        d2 = ((self._buf[: self._n] - target) ** 2).sum(axis=1)
        return self.nodes[int(np.argmin(d2))]


class _TransitionTest:
    """T-RRT adaptive-temperature transition test (shared by both trees).

    accept downhill always; accept uphill with p = exp(-dcost / T). Per the
    original T-RRT rule (Jaillet et al. 2010), cooling after an accepted
    uphill step is PROPORTIONAL to the cost increase — T /= alpha^(dcost /
    t_init) — so crossing a real ridge cools sharply while the tiny uphill
    gradients of a smooth field's tails barely cool at all (a flat "halve
    on every accept" collapses T to t_min within a few dozen accepts in
    such fields and freezes both trees). Heating: after `nfail_max`
    consecutive rejections T *= alpha, so a planner stalled against a cost
    ridge (e.g. a grasp goal inside the canopy) gradually relaxes until
    progress resumes. Distance-normalization is skipped because extension
    steps have ~constant length (extend_fn resolution).
    """

    def __init__(self, t_init, t_min, alpha, nfail_max):
        self.T = float(t_init)
        self.t_init = float(t_init)
        self.t_min = float(t_min)
        self.alpha = float(alpha)
        self.nfail_max = int(nfail_max)
        self.nfail = 0

    def __call__(self, cost_from, cost_to) -> bool:
        dcost = cost_to - cost_from
        if dcost <= 0.0:
            return True
        if np.random.random() < math.exp(-dcost / max(self.T, self.t_min)):
            self.T = max(
                self.T / (self.alpha ** (dcost / self.t_init)), self.t_min
            )
            self.nfail = 0
            return True
        self.nfail += 1
        if self.nfail > self.nfail_max:
            self.T *= self.alpha
            self.nfail = 0
        return False


def _cost_extend_towards(tree, target, extend_fn, collision_fn,
                         config_cost_fn, transition, swap=False):
    """`extend_towards` (pybullet_planning.primitives) with a per-step
    transition test: each new step must be accepted by the cost filter
    relative to its parent step AND be collision-free. The transition test
    runs FIRST: a cost lookup (FK + trilinear grid read) is ~10x cheaper
    than a collision check against a large concave mesh, so every
    cost-rejected step skips the collision query entirely — in dense-field
    regions that's most of the rejected work. Returns (last_node, success)."""
    near = tree.nearest(np.asarray(target, dtype=np.float64))
    extend = list(extend_fn(near.q, target))
    if swap:  # asymmetric_extend: goal-tree extensions run the reversed edge
        extend = list(reversed(list(extend_fn(target, near.q))))
    last = near
    n_safe = 0
    for q in extend:
        c = float(config_cost_fn(q))
        if not transition(last.cost, c):
            break
        if collision_fn(q):
            break
        last = _CostNode(q, parent=last, cost=c)
        tree.add(last)
        n_safe += 1
    return last, n_safe == len(extend)


def cost_aware_birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn,
                     collision_fn, config_cost_fn,
                     max_iterations=None, max_time=None, t_init=None,
                     t_min=None, alpha=None, nfail_max=None, verbose=False,
                     line_bias=0.3, line_bias_std=0.3,
                     lower_limits=None, upper_limits=None):
    """Bidirectional RRT-Connect with a T-RRT transition test on every
    extension step. Drop-in for pybullet_planning's `birrt` when a
    `config_cost_fn(q) -> float` is available (normalized soft-cost field).

    Unlike `birrt` there is NO check_direct fast path: the straight
    start-goal segment being collision-free says nothing about its cost, and
    accepting it unconditionally is exactly how cost-blind planning arcs
    through the canopy. A caller wanting that shortcut can pre-check the
    direct segment's cost itself (get_rrt_plan does, with a cost gate).

    `distance_fn` is accepted for signature compatibility with `birrt` but
    nearest-neighbor lookups use vectorized joint-space L2 (what every
    caller passes anyway) — see _CostTree.

    Corridor-biased sampling: with probability `line_bias` the sample is
    drawn from the straight start-goal segment plus N(0, line_bias_std^2)
    per-joint noise (clipped to `lower_limits`/`upper_limits` when given)
    instead of uniformly over the joint box. Reach-style solutions live
    near that corridor (the cost detour is an offset from it, covered by
    the noise), while uniform 6-DOF samples mostly grow the trees into
    irrelevant space — the bias cuts iterations-to-connect and the rate of
    full-budget failures. 0.0 disables (pure uniform sampling).

    Returns a list of configs (start..goal) or None.
    """
    d = _TRRT_DEFAULTS
    max_iterations = d["max_iterations"] if max_iterations is None else int(max_iterations)
    max_time = d["max_time"] if max_time is None else float(max_time)
    transition = _TransitionTest(
        d["t_init"] if t_init is None else t_init,
        d["t_min"] if t_min is None else t_min,
        d["alpha"] if alpha is None else alpha,
        d["nfail_max"] if nfail_max is None else nfail_max,
    )
    if collision_fn(q_start) or collision_fn(q_goal):
        return None
    start_time = time.time()
    q_start = np.asarray(q_start, dtype=np.float64)
    q_goal = np.asarray(q_goal, dtype=np.float64)
    ll = None if lower_limits is None else np.asarray(lower_limits, dtype=np.float64)
    ul = None if upper_limits is None else np.asarray(upper_limits, dtype=np.float64)

    def _sample_target():
        if line_bias > 0.0 and np.random.random() < line_bias:
            q = (q_start + np.random.random() * (q_goal - q_start)
                 + np.random.normal(0.0, line_bias_std, size=q_start.shape))
            if ll is not None and ul is not None:
                q = np.clip(q, ll, ul)
            return q
        return np.asarray(sample_fn(), dtype=np.float64)

    tree_a = _CostTree(_CostNode(q_start, cost=float(config_cost_fn(q_start))))
    tree_b = _CostTree(_CostNode(q_goal, cost=float(config_cost_fn(q_goal))))
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"cost-aware birrt: TIMEOUT after {max_time:.0f}s "
                      f"({iteration} iterations, "
                      f"{len(tree_a) + len(tree_b)} nodes)")
            return None
        swap = len(tree_a) > len(tree_b)
        tree1, tree2 = (tree_b, tree_a) if swap else (tree_a, tree_b)

        target = _sample_target()
        last1, _ = _cost_extend_towards(
            tree1, target, extend_fn, collision_fn,
            config_cost_fn, transition, swap)
        last2, success = _cost_extend_towards(
            tree2, last1.q, extend_fn, collision_fn,
            config_cost_fn, transition, not swap)
        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            if verbose:
                print(f"cost-aware birrt: {iteration + 1} iterations, "
                      f"{len(tree_a) + len(tree_b)} nodes, "
                      f"T={transition.T:.2e}, "
                      f"{time.time() - start_time:.1f}s")
            return path1[:-1] + path2[::-1]
    if verbose:
        print(f"cost-aware birrt: FAILED after {max_iterations} iterations "
              f"({len(tree_a) + len(tree_b)} nodes, T={transition.T:.2e}, "
              f"{time.time() - start_time:.1f}s)")
    return None


def _path_cost_integral(points, config_cost_fn):
    """Sum of mean-endpoint-cost * joint-space segment length over a
    waypoint sequence — same trapezoid form as
    RRTToGoalPlanner._path_soft_cost, evaluated on the given points as-is."""
    pts = [np.asarray(q, dtype=np.float64) for q in points]
    if len(pts) < 2:
        return 0.0
    costs = [float(config_cost_fn(q)) for q in pts]
    total = 0.0
    for a, b, ca, cb in zip(pts[:-1], pts[1:], costs[:-1], costs[1:]):
        total += float(np.linalg.norm(b - a)) * 0.5 * (ca + cb)
    return total


def cost_aware_smooth_path(path, extend_fn, collision_fn, config_cost_fn,
                           max_smooth_iterations=50, cost_tolerance=1e-3):
    """Random-shortcut smoothing that refuses shortcuts that raise the
    soft-cost integral. Same move as pybullet_planning's `smooth_path` (pick
    two random waypoints, replace the intermediate stretch with the straight
    extend_fn segment when collision-free) plus one extra gate: the
    replacement segment's cost integral must not exceed the replaced
    stretch's by more than `cost_tolerance` (absolute, in normalized-cost x
    radians units). Without the gate, shortcutting undoes every detour the
    cost-aware tree growth just paid for."""
    pts = [np.asarray(q, dtype=np.float64) for q in path]
    for _ in range(max_smooth_iterations):
        if len(pts) <= 2:
            return pts
        i, j = sorted(np.random.randint(0, len(pts), 2))
        if j <= i + 1:
            continue
        shortcut = [pts[i]] + list(extend_fn(pts[i], pts[j]))
        if len(shortcut) >= (j - i + 1):
            continue  # not actually shorter
        if any(collision_fn(q) for q in shortcut[1:-1]):
            continue
        old_cost = _path_cost_integral(pts[i:j + 1], config_cost_fn)
        new_cost = _path_cost_integral(shortcut, config_cost_fn)
        if new_cost > old_cost + cost_tolerance:
            continue
        pts = pts[:i + 1] + shortcut[1:-1] + pts[j:]
    return pts


def get_rrt_plan(robot_id, joint_indices, obstacle_ids, q_start, q_goal,
                 lower_limits=None, upper_limits=None, resolutions=None,
                 verbose=True, obstacle_names=None, skip_pairs=None,
                 physics_client_id=None,
                 obstacle_clearance=None, self_collision_clearance=None,
                 self_collision_skip_pairs=None,
                 actual_gripper_q=None,
                 config_cost_fn=None, trrt_params=None):
    """Plan a joint-space path from q_start to q_goal with bidirectional RRT.

    `physics_client_id` controls which PyBullet server every call goes to,
    so this works correctly even when multiple clients are connected (e.g.
    SplatSim's GUI server + lerobot's wrapper's DIRECT client).

    `lower_limits`/`upper_limits`/`resolutions` let callers provide joint
    bounds and step sizes directly so we don't need pybullet_planning's
    helpers (which would query the *default* client and might see a
    different body at the same id). When omitted we fall back to those
    helpers — fine when only one client is connected.

    `config_cost_fn` (q -> float, normalized soft cost of a configuration):
    when provided, planning runs `cost_aware_birrt` (T-RRT transition test
    on every extension step) instead of pybullet_planning's `birrt`, so tree
    growth itself avoids high-cost regions. `birrt`'s check_direct fast path
    is replaced by a cost-gated direct check: the straight segment is
    accepted only when it is collision-free AND its mean cost is ~zero.
    None (default) = exact historical binary planning. `trrt_params` is an
    optional dict overriding `_TRRT_DEFAULTS` keys (max_iterations, t_init,
    t_min, alpha, nfail_max) plus "restarts" (extra attempts, default 2)
    and "direct_cost_threshold" (mean-cost gate for the direct segment,
    default 0.01).
    """
    cid = _resolve_client_id(physics_client_id)
    if verbose:
        print("Planning with pybullet planning...")
    set_robot_joint_positions(robot_id, joint_indices, q_start, physics_client_id=cid)
    # `set_robot_joint_positions` internally calls `open_gripper()` — resets
    # every gripper joint to 0.0. Re-snap them to the env's actual gripper
    # config so BiRRT's per-sample `collision_fn` (which uses bare
    # resetJointState on arm joints only) evaluates against the SAME finger
    # geometry the caller's outer collision predicates expect. Without this,
    # BiRRT samples paths against wide-open finger geometry while ruckig-
    # smoothed / dense-checked paths use actual (typically closed) fingers,
    # producing "escape says safe / RRT says colliding" cascade failures on
    # grasp tasks.
    if actual_gripper_q is not None:
        _n_pb_joints = p.getNumJoints(robot_id, physicsClientId=cid)
        _dof = len(joint_indices)
        _gv = float(actual_gripper_q)
        for _idx in range(_dof + 1, _n_pb_joints):
            p.resetJointState(robot_id, _idx, _gv, physicsClientId=cid)

    if lower_limits is not None and upper_limits is not None:
        sample_fn = _make_uniform_sample_fn(lower_limits, upper_limits)
        distance_fn = _make_l2_distance_fn()
        extend_fn = _make_linear_extend_fn(
            resolutions if resolutions is not None else [0.05] * len(joint_indices)
        )
    else:
        sample_fn = get_sample_fn(robot_id, joint_indices)
        distance_fn = get_distance_fn(robot_id, joint_indices)
        extend_fn = get_extend_fn(robot_id, joint_indices)

    # Clearance kwargs forwarded into every collision check so RRT's
    # sample/extend/smooth/start/goal checks all use the same configured
    # margin. None falls through to check_links_in_collision's defaults
    # (_COLLISION_CLEARANCE = 0.01 obstacle, self = 0.0).
    _ccheck_kwargs = {}
    if obstacle_clearance is not None:
        _ccheck_kwargs["obstacle_clearance"] = obstacle_clearance
    if self_collision_clearance is not None:
        _ccheck_kwargs["self_collision_clearance"] = self_collision_clearance
    if self_collision_skip_pairs:
        _ccheck_kwargs["self_collision_skip_pairs"] = self_collision_skip_pairs

    # Link scope for every collision check in this BiRRT invocation.
    # MUST match the RRTToGoalPlanner's `_current_pose_in_planner_collision`
    # scope (which excludes ONLY the world frame -1 — base_link 0 IS
    # included so gripper-into-own-mount self-collisions are caught) so the
    # escape chain's "safe" verdict agrees with the BiRRT collision_fn's
    # verdict. Prior mismatches caused escape to find a config that RRT's
    # `collision_fn` immediately declared in-collision (or vice versa)
    # → cascade of 5-retry backoffs. Any obstacle false-fires against
    # base_link should be silenced per-env via `skip_pairs` (the obstacle-
    # side skip mechanism is separate from self_collision_skip_pairs, so
    # silencing an obstacle pair doesn't disable the self-check).
    _n_pb_joints = p.getNumJoints(robot_id, physicsClientId=cid)
    _link_indices_to_check = list(range(0, _n_pb_joints))

    def collision_fn(q):
        return check_links_in_collision(robot_id, joint_indices, q, obstacle_ids,
                                         skip_pairs=skip_pairs, physics_client_id=cid,
                                         link_indices_to_check=_link_indices_to_check,
                                         **_ccheck_kwargs)

    if config_cost_fn is None:
        path = birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
    else:
        tp = dict(trrt_params or {})
        restarts = int(tp.pop("restarts", 1))
        fallback_to_binary = bool(tp.pop("fallback_to_binary", True))
        # 0.002: a "free pass" straight line must be genuinely near-zero
        # cost. At the old 0.01 the vine bench accepted directs with mean
        # cost 0.006-0.009, which over a ~3 rad path is a ~0.02-0.03
        # exposure integral — comparable to a whole cost-aware plan's total.
        direct_cost_threshold = float(tp.pop("direct_cost_threshold", 0.002))
        # Cost-gated equivalent of birrt's check_direct: take the straight
        # segment only when it is collision-free AND essentially cost-free —
        # otherwise it deserves real (cost-aware) planning.
        direct = [np.asarray(q_start, dtype=np.float64)] + [
            np.asarray(q, dtype=np.float64)
            for q in extend_fn(q_start, q_goal)
        ]
        path = None
        if not any(collision_fn(q) for q in direct):
            direct_mean_cost = float(
                np.mean([config_cost_fn(q) for q in direct]))
            if direct_mean_cost <= direct_cost_threshold:
                if verbose:
                    print("cost-aware birrt: direct segment is collision-free "
                          f"and low-cost (mean {direct_mean_cost:.4f}) — using it")
                path = direct
        if path is None:
            for _attempt in range(restarts + 1):
                path = cost_aware_birrt(
                    q_start, q_goal, distance_fn, sample_fn, extend_fn,
                    collision_fn, config_cost_fn, verbose=verbose,
                    lower_limits=lower_limits, upper_limits=upper_limits,
                    **tp)
                if path is not None:
                    break
        if path is None and fallback_to_binary:
            # The cost-aware tree couldn't connect within its time/iteration
            # budget (T-RRT's transition test makes hard scenes MUCH more
            # expensive to fail on than binary birrt). Fall back to plain
            # binary planning: the result is still cost-GATED downstream
            # (shortcut/elastic/trajopt gates all use config_cost_fn) and
            # cost-SCORED by the planner, so this degrades to score-mode
            # quality for this candidate instead of stalling the pipeline.
            if verbose:
                print("cost-aware birrt: falling back to binary birrt "
                      "(cost-aware attempts exhausted)")
            # max_iterations=150: pybullet_planning's default is 20 per
            # restart — decorative next to the 2x400 cost-aware budget it is
            # supposed to rescue (vine bench: the stock fallback essentially
            # never connected). Binary iterations are cheap (no cost lookups),
            # so a real budget here turns "fallback failed too" into a
            # score-mode-quality path.
            path = birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn,
                         collision_fn, max_iterations=150)
    if path is None:
        start_in_col = check_links_in_collision(robot_id, joint_indices, q_start, obstacle_ids, verbose=True, obstacle_names=obstacle_names, skip_pairs=skip_pairs, physics_client_id=cid, link_indices_to_check=_link_indices_to_check, **_ccheck_kwargs)
        goal_in_col = check_links_in_collision(robot_id, joint_indices, q_goal, obstacle_ids, verbose=True, obstacle_names=obstacle_names, skip_pairs=skip_pairs, physics_client_id=cid, link_indices_to_check=_link_indices_to_check, **_ccheck_kwargs)
        if start_in_col and goal_in_col:
            print("PyBullet planning failed: both q_start and q_goal are in collision.")
        elif start_in_col:
            print("PyBullet planning failed: q_start is in collision.")
        elif goal_in_col:
            print("PyBullet planning failed: q_goal is in collision.")
        else:
            print("PyBullet planning failed: start and goal are collision-free individually, but no path was found (environment may be too constrained or max iterations exhausted).")
        return None

    path = np.array(path)

    # Sometimes BiRRT returns the path from goal → start; flip to start → goal.
    # `np.ndarray` doesn't have `.reverse()` (that's a list-only method) — must
    # use numpy slicing. Pre-fix this errored as
    # `AttributeError: 'numpy.ndarray' object has no attribute 'reverse'`
    # whenever the wrong-direction branch fired, which under
    # multi-candidate generation became common (each candidate is an
    # independent BiRRT run).
    if ((path[0] - q_start) ** 2).sum() > ((path[0] - q_goal) ** 2).sum():
        path = path[::-1]
    if verbose:
        print("RRT raw path length:", len(path))
    return path

def resample_path_by_distance(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resamples a path to have a specific number of points, spaced
    evenly by distance (arc length) along the path.

    Args:
        path: The original path (N, DOF) array.
        n_points: The desired number of points.

    Returns:
        The new, resampled path (n_points, DOF) array.
    """
    if not isinstance(path, np.ndarray):
        path = np.array(path)
        
    n_original_points, dof = path.shape
    if n_original_points < 2:
        # Not enough points to interpolate
        return path

    # 1. Calculate the distance between each original point
    # diffs is (N-1, DOF)
    diffs = np.diff(path, axis=0)
    # dists is (N-1,)
    dists = np.linalg.norm(diffs, axis=1)

    # 2. Calculate the cumulative distance (arc length) at each original point
    # cum_dists is (N,)
    cum_dists = np.zeros(n_original_points)
    cum_dists[1:] = np.cumsum(dists)
    total_dist = cum_dists[-1]

    # 3. Create the new, evenly spaced distance markers
    # new_dists is (n_points,)
    new_dists = np.linspace(0, total_dist, num=n_points)
    
    # 4. Create an empty array for the new path
    resampled_path = np.zeros((n_points, dof))
    
    # 5. Interpolate each joint (column)
    for i in range(dof):
        joint_original = path[:, i]
        # Use cum_dists as the 'x' axis and joint_original as the 'y' axis
        # Use new_dists as the new 'x' axis to query
        joint_new = np.interp(new_dists, cum_dists, joint_original)
        resampled_path[:, i] = joint_new
        
    return resampled_path

def _ruckig_run_segment(
    waypoints: np.ndarray,
    start_vel: np.ndarray,
    start_acc: np.ndarray,
    end_vel: np.ndarray,
    end_acc: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    per_section_max_velocity: list | None = None,
    per_section_max_acceleration: list | None = None,
) -> tuple:
    """Run ruckig on a single segment. Returns (samples, final_vel, final_acc).

    per_section_max_velocity / per_section_max_acceleration: optional
    per-section limit lists (length = number of waypoint gaps =
    len(waypoints) - 1, each entry a DOF-length list). Used by the
    final-approach taper to slow only the sections near the goal while
    ruckig plans ONE trajectory through them (it picks the section-boundary
    velocities itself). None = uniform limits (historical behavior).
    """
    from ruckig import InputParameter, Ruckig, Trajectory, Synchronization, ControlInterface  # type: ignore

    dof = waypoints.shape[1]
    dt = 1.0 / control_hz
    n_intermediates = len(waypoints) - 2
    otg = Ruckig(dof, dt, max(n_intermediates, 0))

    inp = InputParameter(dof)
    inp.control_interface = ControlInterface.Position
    inp.synchronization = Synchronization.Phase
    inp.current_position = waypoints[0].tolist()
    inp.current_velocity = start_vel.tolist()
    inp.current_acceleration = start_acc.tolist()
    if n_intermediates > 0:
        inp.intermediate_positions = waypoints[1:-1].tolist()
    inp.target_position = waypoints[-1].tolist()
    inp.target_velocity = end_vel.tolist()
    inp.target_acceleration = end_acc.tolist()
    inp.max_velocity = max_joint_vel.tolist()
    inp.max_acceleration = max_joint_acc.tolist()
    inp.max_jerk = max_joint_jerk.tolist()
    if per_section_max_velocity is not None:
        inp.per_section_max_velocity = list(per_section_max_velocity)
    if per_section_max_acceleration is not None:
        inp.per_section_max_acceleration = list(per_section_max_acceleration)

    traj = Trajectory(dof)
    try:
        result = otg.calculate(inp, traj)
    except Exception as e:
        # Community ruckig delegates intermediate-waypoint problems to a
        # cloud API; when it is unreachable or rate-limited every subsequent
        # call fails identically — surface that as a distinct, fatal-for-now
        # error so generation loops stop gracefully instead of crashing or
        # endlessly re-rolling scenes (RuckigError message looks like:
        # "could not reach cloud API server, error code: 429 ...").
        msg = str(e)
        if type(e).__name__ == "RuckigError" and (
                "cloud API" in msg or "Rate limit" in msg):
            raise RuckigCloudUnavailableError(msg.strip()) from e
        raise
    if result < 0:
        raise RuntimeError(f"Ruckig trajectory calculation failed with result: {result}")

    duration = traj.duration
    dt = 1.0 / control_hz
    ts = np.arange(0, duration, dt)
    ts = np.append(ts, duration)
    samples = np.array([traj.at_time(t)[0] for t in ts])
    final_vel = np.array(traj.at_time(duration)[1])
    final_acc = np.array(traj.at_time(duration)[2])
    return samples, final_vel, final_acc


class RuckigCloudUnavailableError(RuntimeError):
    """The ruckig cloud API is unreachable or rate-limited (e.g. HTTP 429
    "Rate limit exceeded: 1000 per 1 day").

    Deliberately NOT an RRTPlanningError subclass: planning-failure handlers
    retry with a new scene/candidate, which is pointless (and an infinite
    re-roll loop) when every future ruckig call will fail the same way.
    Callers that drive long-running generation loops should catch THIS and
    stop gracefully (see the GENERATE_TRAJECTORIES branch of
    PybulletRobotServerBase.serve)."""


class TrajectoryParametrizationError(RuntimeError):
    """A parametrization backend failed on THIS path (e.g. ruckig's step-1
    solver throwing on a boundary state). Unlike RuckigCloudUnavailableError
    the failure is input-specific, not systemic: the right response is to
    discard the candidate path and try another, exactly as for a candidate
    whose smoothed form collides (see RRTPlanner._smooth_and_check_collision).
    Raised instead of the raw ruckig.RuckigError so callers can be selective
    without importing ruckig."""


def _rdp_joint_path(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on an (N, DOF) joint path. Returns kept indices."""
    if len(points) < 3:
        return np.arange(len(points))
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            d = np.linalg.norm(points[a + 1:b] - points[a], axis=1)
        else:
            u = seg / seg_len
            rel = points[a + 1:b] - points[a]
            proj = rel @ u
            d = np.linalg.norm(rel - np.outer(proj, u), axis=1)
        i = int(np.argmax(d))
        if d[i] > epsilon:
            mid = a + 1 + i
            keep[mid] = True
            stack.append((a, mid))
            stack.append((mid, b))
    return np.flatnonzero(keep)


def elastic_smooth_path(
    path,
    collision_fn,
    passes: int = 30,
    alpha: float = 0.5,
    densify_step: float = 0.10,
    decimate_eps: float = 0.015,
    config_cost_fn=None,
    cost_increase_tol: float = 0.002,
) -> np.ndarray:
    """Corner-ROUNDING smoother for paths in tight scenes.

    Random-shortcut smoothing (`smooth_path`) removes a corner only when the
    straight line PAST it is collision-free — in narrow corridors (e.g. a
    robot threading a vine canopy) those shortcuts collide and the RRT's
    jagged 50-90 deg joint-space corners survive, which downstream ruckig
    turns into visible wobble (zero-velocity stops at sharp corners, speed
    surging between sections). This pass BENDS instead of cutting: densify
    the path, then iteratively pull each interior waypoint toward the
    midpoint of its neighbors, keeping the move only if the perturbed
    section stays collision-free. Corners relax into gentle arcs that hug
    the corridor. Finally decimates with joint-space RDP so ruckig receives
    a modest number of smooth waypoints.

    Args:
        path: (N, DOF) waypoints (list or array).
        collision_fn: q -> bool (True = colliding), same contract as
            smooth_path's.
        passes: max relaxation sweeps (early-exits on convergence).
        alpha: blend factor per sweep (0..1, higher = stronger pull).
        densify_step: max per-joint L-inf spacing (rad) of the densified
            path the relaxation runs on.
        decimate_eps: RDP tolerance (rad) for the final decimation.
        config_cost_fn: optional q -> float soft-cost lookup (normalized
            field). When set, a midpoint-pull is additionally rejected if it
            raises the waypoint's cost by more than `cost_increase_tol` —
            corner rounding through straightening is exactly a mini-shortcut,
            and unguarded it drags cost-avoiding detours back into the
            canopy. None (default) = historical behavior.
        cost_increase_tol: per-waypoint cost increase allowed per pull (in
            normalized-cost units, field max = 1). 0.002 because typical
            per-config costs in a real vegetation field are ~0.005-0.03 —
            a 0.01 tolerance made the gate effectively inert.
    """
    wp = np.asarray(path, dtype=np.float64)
    if wp.shape[0] < 3:
        return wp

    def _segment_collides(a, b, step=0.05):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for k in range(1, n + 1):
            if collision_fn(a + (b - a) * (k / n)):
                return True
        return False

    def _path_collides(points):
        return any(_segment_collides(a, b)
                   for a, b in zip(points[:-1], points[1:]))

    # densify so corners have room to round
    dense = [wp[0]]
    for a, b in zip(wp[:-1], wp[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / densify_step)))
        for k in range(1, n + 1):
            dense.append(a + (b - a) * (k / n))
    wp = np.array(dense)

    # Per-move checks (candidate + edge midpoints) are LOCAL and go stale as
    # neighbors move in later iterations, so each pass is verified against
    # the full 0.05-rad-densified path and reverted if it broke — the
    # returned path is always gate-clean (the planner's final collision gate
    # re-densifies at the same resolution).
    for _ in range(passes):
        snapshot = wp.copy()
        changed = False
        for i in range(1, len(wp) - 1):
            target = 0.5 * (wp[i - 1] + wp[i + 1])
            cand = (1.0 - alpha) * wp[i] + alpha * target
            if np.max(np.abs(cand - wp[i])) < 1e-5:
                continue
            if (collision_fn(cand)
                    or collision_fn(0.5 * (cand + wp[i - 1]))
                    or collision_fn(0.5 * (cand + wp[i + 1]))):
                continue
            if config_cost_fn is not None and (
                    float(config_cost_fn(cand))
                    > float(config_cost_fn(wp[i])) + cost_increase_tol):
                continue
            wp[i] = cand
            changed = True
        if not changed:
            break
        if _path_collides(wp):
            wp = snapshot
            break

    # decimate; a too-aggressive RDP chord can graze an obstacle, so verify
    # and tighten the tolerance (and finally fall back to undecimated).
    for eps in (decimate_eps, decimate_eps / 2, decimate_eps / 4):
        out = wp[_rdp_joint_path(wp, eps)]
        if not _path_collides(out):
            return out
    return wp


def trajopt_smooth_path(
    path,
    collision_fn,
    distance_fn,
    passes: int = 30,
    lr: float = 0.02,
    smoothness_weight: float = 1.0,
    collision_weight: float = 5.0,
    collision_threshold: float = 0.10,
    fd_step: float = 0.01,
    densify_step: float = 0.10,
    decimate_eps: float = 0.04,
    config_cost_fn=None,
    cost_increase_tol: float = 0.002,
) -> np.ndarray:
    """CHOMP-lite post-RRT trajectory optimizer with soft collision + smoothness.

    `config_cost_fn` (optional, q -> normalized soft cost): gradient steps
    that raise a waypoint's soft cost by more than `cost_increase_tol` are
    rejected, so the smoothness pull cannot drag a cost-avoiding detour back
    into high-cost (vegetation) space. None = historical behavior.

    Extends `elastic_smooth_path` by adding an EXPLICIT REPULSIVE collision
    cost — a hinge on min-signed-distance-to-obstacles that activates when a
    waypoint is within `collision_threshold` meters of any obstacle. Gradient
    is computed by central finite differences per joint per waypoint;
    combined with the analytical Laplacian smoothness gradient this pushes
    waypoints AWAY from obstacles (not just refusing to step INTO them) while
    minimizing curvature.

    Contrast with `elastic_smooth_path`:
      - `elastic_smooth_path`: hard reject on collision, no repulsion. Corners
        relax toward the neighbor-midpoint until the pull would collide.
      - `trajopt_smooth_path`: soft cost with gradient descent. Waypoints
        near an obstacle get pushed to larger clearance even if the Laplacian
        alone wouldn't move them, because the cost term keeps growing as
        clearance shrinks.

    Rationale: raw RRT paths are only marginally collision-free — they take
    whatever path samples pass the collision predicate, which typically hugs
    obstacles. Post-shortcut paths inherit that geometry. Adding a soft
    collision cost with a clearance threshold gives the optimizer a
    continuous signal to prefer wider-clearance homotopy-equivalent paths
    over tight-clearance ones. Combined with the paper reference (CHOMP /
    "1001 Demos" trajectory optimization), this is the standard "make RRT
    output more deterministic and safer" post-processing step.

    Args:
        path: (N, DOF) waypoints from RRT / earlier smoothing pass.
        collision_fn: q -> bool. Hard-collision predicate (True = colliding).
            Used only for gate validation — reject candidate perturbations
            that would enter collision, revert to the last-known-good.
        distance_fn: q -> float. Min signed distance to any obstacle, in
            meters. Positive = safe, negative = in collision. Recommended
            impl: `lambda q: min_distance_to_obstacles(robot_id,
            joint_indices, q, obstacle_ids, physics_client_id=cid)`.
        passes: max gradient-descent sweeps (early-exits on convergence).
        lr: gradient step size per sweep (rad).
        smoothness_weight: weight of the Laplacian smoothness term
            (||q_i - 0.5*(q_{i-1}+q_{i+1})||²). Higher = smoother path,
            less obstacle repulsion.
        collision_weight: weight of the soft-collision hinge cost. Higher =
            path pushed further from obstacles at the cost of length /
            smoothness.
        collision_threshold: distance below which the collision cost
            activates. `hinge(threshold - d)²` — quadratic when close, zero
            when far. In your planar env `0.10` m ≈ ½ link width — a good
            starting value; tune down if paths become too conservative.
        fd_step: central-diff step for the collision gradient (rad).
        densify_step: densify the path to this max L∞ joint-spacing before
            optimizing (matches `elastic_smooth_path`), so gradient descent
            has enough interior waypoints to bend cleanly.
        decimate_eps: joint-space RDP tolerance for the final decimation.
            Default 0.04 rad (~2.3° per joint) is coarser than
            elastic_smooth_path's 0.015 default because the trajopt optimizer
            actively SPREADS waypoints (pushing them away from obstacles into
            genuinely different joint configs), so RDP can't collapse them
            as aggressively as it can with elastic's more-collinear output.
            At 0.015 the output ends up with 30+ waypoints on a modest RRT
            path, which triggers a ruckig warning ("please reduce/filter the
            number of waypoints for better results"). 0.04 keeps the final
            geometry under the parametrizer-friendly ~15-waypoint cap.

    Returns:
        (M, DOF) optimized + decimated waypoints. M is typically < N (RDP
        decimation removes redundant intermediate points).

    Cost budget:
        Per pass: N × 2·DOF collision-distance queries (finite-diff
        gradient) + N collision checks (candidate validation). At ~1 ms
        per query, `passes=30` on a densified 20-waypoint path ≈ 3 s per
        trajectory. Acceptable for offline data-generation; use lower
        `passes` for online / retrieval-hot paths.
    """
    wp = np.asarray(path, dtype=np.float64).copy()
    if wp.shape[0] < 3:
        return wp
    dof = wp.shape[1]
    # One-line trace so callers can confirm trajopt ran (matches the terseness
    # of the surrounding "Planning with pybullet planning..." /
    # "RRT raw path length: N" prints elsewhere in the pipeline).
    print(f"[trajopt] {passes} passes  in={wp.shape[0]} waypoints  "
          f"weights=(smooth={smoothness_weight}, coll={collision_weight}@thresh={collision_threshold}m)")

    def _segment_collides(a, b, step=0.05):
        # Reused from elastic_smooth_path — dense sub-sampling collision
        # check on a straight segment, matching the RRT resolution.
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for k in range(1, n + 1):
            if collision_fn(a + (b - a) * (k / n)):
                return True
        return False

    def _path_collides(points):
        return any(_segment_collides(a, b)
                   for a, b in zip(points[:-1], points[1:]))

    def _collision_cost(q):
        d = float(distance_fn(q))
        gap = max(0.0, collision_threshold - d)
        return gap * gap  # quadratic hinge

    # Densify (matches elastic_smooth_path) — corners need room to bend.
    dense = [wp[0]]
    for a, b in zip(wp[:-1], wp[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / densify_step)))
        for k in range(1, n + 1):
            dense.append(a + (b - a) * (k / n))
    wp = np.array(dense)
    N = wp.shape[0]

    # Gradient-descent sweeps. Endpoints are FIXED (start config + goal
    # config must be preserved — they were carefully chosen by RRT to be
    # the reachable start/end).
    for _pass in range(passes):
        snapshot = wp.copy()
        changed = False
        for i in range(1, N - 1):
            # Smoothness gradient (Laplacian on the trajectory):
            #   ∇_i ||q_i - 0.5*(q_{i-1}+q_{i+1})||² ∝ 2*q_i - q_{i-1} - q_{i+1}
            grad_smooth = 2.0 * wp[i] - wp[i - 1] - wp[i + 1]
            # Collision gradient (central finite differences per joint).
            # Small `fd_step` = accurate but sensitive to distance-fn noise;
            # 0.01 rad matches the resolution PyBullet's collision checks
            # resolve at, so signal ≥ discretization noise.
            grad_coll = np.zeros(dof)
            for j in range(dof):
                q_p = wp[i].copy(); q_p[j] += fd_step
                q_m = wp[i].copy(); q_m[j] -= fd_step
                grad_coll[j] = (_collision_cost(q_p) - _collision_cost(q_m)) / (2 * fd_step)
            grad = smoothness_weight * grad_smooth + collision_weight * grad_coll
            if np.max(np.abs(grad)) < 1e-6:
                continue
            cand = wp[i] - lr * grad
            # Hard gate: reject the step if the candidate collides OR the
            # segments to its neighbors would collide (matches
            # elastic_smooth_path's local validation).
            if (collision_fn(cand)
                    or collision_fn(0.5 * (cand + wp[i - 1]))
                    or collision_fn(0.5 * (cand + wp[i + 1]))):
                continue
            if config_cost_fn is not None and (
                    float(config_cost_fn(cand))
                    > float(config_cost_fn(wp[i])) + cost_increase_tol):
                continue
            wp[i] = cand
            changed = True
        if not changed:
            break
        # Full-path re-validation after each sweep — local checks go stale
        # as neighbors move, so any pass that would leave a stale collision
        # is reverted (matches elastic_smooth_path's snapshot pattern).
        if _path_collides(wp):
            wp = snapshot
            break

    # Decimate — RDP tolerance may allow the reduced chord to graze an
    # obstacle, so verify and retry with tighter tolerances before falling
    # back to the undecimated dense path (matches elastic_smooth_path).
    for eps in (decimate_eps, decimate_eps / 2, decimate_eps / 4):
        out = wp[_rdp_joint_path(wp, eps)]
        if not _path_collides(out):
            print(f"[trajopt]   out={out.shape[0]} waypoints (RDP eps={eps:.4f} rad)")
            return out
    print(f"[trajopt]   out={wp.shape[0]} waypoints (RDP fallback — every decimation collided)")
    return wp


def _find_sharp_waypoint_indices(waypoints: np.ndarray, threshold_deg: float) -> list:
    """Return indices of waypoints where the turn angle exceeds threshold_deg."""
    sharp = []
    for i in range(1, len(waypoints) - 1):
        v_in = waypoints[i] - waypoints[i - 1]
        v_out = waypoints[i + 1] - waypoints[i]
        n_in = np.linalg.norm(v_in)
        n_out = np.linalg.norm(v_out)
        if n_in < 1e-9 or n_out < 1e-9:
            continue
        cos_angle = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle > threshold_deg:
            sharp.append(i)
    return sharp


def _prepare_section_limits(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    final_approach_dist: float,
    final_approach_vel_scale: float,
    final_approach_acc_scale: float,
    uniform_path_speed: bool,
) -> tuple:
    """Build the PER-SECTION vel/acc limit lists shared by both time
    parametrization backends (ruckig and toppra).

    A "section" is the span between consecutive waypoints, so the returned
    lists have length ``len(waypoints) - 1`` and each entry is a DOF-length
    list. Two independent effects are folded in:

      * final-approach taper (``final_approach_dist > 0``) — insert a split
        waypoint that far (joint-space arc length) before the goal and scale
        every section from the split onward.
      * ``uniform_path_speed`` — cap each section's per-joint velocity so its
        L2 path speed matches every other section's.

    Returns ``(waypoints, max_joint_vel, max_joint_acc, per_section_vel,
    per_section_acc)``. ``waypoints`` may have ONE extra row (the inserted
    final-approach split point); the global vel/acc arrays are returned
    scaled when the whole path lies inside the approach zone. Either
    per-section list is None when that effect is inactive (uniform limits).
    """
    # Final-approach taper via ruckig PER-SECTION limits: insert a split
    # waypoint `final_approach_dist` of joint-space arc length before the
    # goal and give every section from the split onward the scaled-down
    # vel/acc limits. ONE ruckig problem plans through it, so ruckig itself
    # chooses the section-boundary velocity (time-optimal AND feasible).
    # This replaces an earlier two-call design that hand-computed a handoff
    # velocity at the split — that was fragile: too fast an entry forced a
    # command overshoot-and-reverse at the goal, too slow (or a zero
    # handoff) taught the policy to stop short of the goal. With
    # per-section limits neither failure mode is possible: the scales are
    # preferences, not correctness-critical.
    per_section_vel: list | None = None
    per_section_acc: list | None = None
    if final_approach_dist and final_approach_dist > 0 and waypoints.shape[0] >= 2:
        scaled_vel = np.asarray(max_joint_vel, dtype=np.float64) * float(final_approach_vel_scale)
        scaled_acc = np.asarray(max_joint_acc, dtype=np.float64) * float(final_approach_acc_scale)
        seg_vecs = np.diff(waypoints, axis=0)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        total_len = float(seg_lens.sum())
        if total_len <= final_approach_dist:
            # Whole path is inside the approach zone — cap the global limits.
            max_joint_vel = scaled_vel
            max_joint_acc = scaled_acc
        else:
            # Walk backward from the goal to find the split point at
            # final_approach_dist of joint-space arc length.
            remaining = float(final_approach_dist)
            i = len(seg_lens) - 1
            while i > 0 and remaining > seg_lens[i]:
                remaining -= seg_lens[i]
                i -= 1
            seg_len = float(seg_lens[i])
            t = (seg_len - remaining) / seg_len if seg_len > 1e-12 else 0.0
            t = min(max(t, 0.0), 1.0)
            split_pt = waypoints[i] + t * seg_vecs[i]
            # Insert the split as a real waypoint unless it coincides with an
            # existing one; sections at/after it get the scaled limits.
            if (
                np.linalg.norm(split_pt - waypoints[i]) > 1e-9
                and np.linalg.norm(split_pt - waypoints[i + 1]) > 1e-9
            ):
                waypoints = np.vstack([waypoints[: i + 1], split_pt, waypoints[i + 1 :]])
                first_scaled_section = i + 1
            elif np.linalg.norm(split_pt - waypoints[i]) <= 1e-9:
                first_scaled_section = i  # split == waypoint i
            else:
                first_scaled_section = i + 1  # split == waypoint i+1
            base_vel = np.asarray(max_joint_vel, dtype=np.float64)
            base_acc = np.asarray(max_joint_acc, dtype=np.float64)
            n_sections = waypoints.shape[0] - 1
            per_section_vel = [
                (scaled_vel if s >= first_scaled_section else base_vel).tolist()
                for s in range(n_sections)
            ]
            per_section_acc = [
                (scaled_acc if s >= first_scaled_section else base_acc).tolist()
                for s in range(n_sections)
            ]

    if uniform_path_speed and waypoints.shape[0] >= 2:
        # Equalize JOINT-SPACE PATH SPEED across sections. Per-joint box
        # velocity limits are direction-anisotropic: a section moving one
        # joint tops out at max_vel, while a section spread across all N
        # joints legally reaches max_vel*sqrt(N) of L2 path speed — the
        # time-optimal profile sprints there, then brakes for the next
        # section, which reads as surging/jerky execution. Capping each
        # section's per-joint velocity at v_path * |unit_dir_j| bounds every
        # section's L2 path speed to the same v_path (the section's smallest
        # existing per-joint cap, so final-approach scaling composes). The
        # 0.1 floor keeps near-stationary joints controllable through
        # waypoint transitions. Acceleration limits are left as-is.
        n_sections = waypoints.shape[0] - 1
        base_vel = np.asarray(max_joint_vel, dtype=np.float64)
        if per_section_vel is None:
            per_section_vel = [base_vel.tolist() for _ in range(n_sections)]
        uniformed = []
        for s in range(n_sections):
            sec_cap = np.asarray(per_section_vel[s], dtype=np.float64)
            d = waypoints[s + 1] - waypoints[s]
            seg_norm = float(np.linalg.norm(d))
            if seg_norm > 1e-12:
                dir_abs = np.abs(d) / seg_norm
                v_path = float(np.min(sec_cap))
                sec_cap = np.minimum(sec_cap, v_path * np.maximum(dir_abs, 0.1))
            uniformed.append(sec_cap.tolist())
        per_section_vel = uniformed
    return waypoints, max_joint_vel, max_joint_acc, per_section_vel, per_section_acc


def ruckig_parametrize_path(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    sharp_angle_threshold_deg: float = 45.0,
    segment_at_sharp_corners: bool = True,
    start_vel: np.ndarray | None = None,
    start_acc: np.ndarray | None = None,
    final_approach_dist: float = 0.0,
    final_approach_vel_scale: float = 0.3,
    final_approach_acc_scale: float = 0.25,
    end_vel: np.ndarray | None = None,
    uniform_path_speed: bool = False,
) -> np.ndarray:
    """
    Time-optimal path parametrization using Ruckig.

    Two modes, controlled by `segment_at_sharp_corners`:

    * (default, True) — split the path at sharp-angle waypoints (angle >
      threshold) and run ruckig per-segment. This matches the historical
      behavior of this function: the robot decelerates to zero velocity at
      each sharp corner before re-accelerating into the next segment. Safe
      because the underlying RRT plans for typical manipulation tasks
      don't have many sharp corners, so segmentation usually produces a
      single segment anyway; only complex multi-obstacle plans see a
      visible difference. Empirical comparison on lever-grasp interventions
      (d5_fast_03dag vs d5jvm_g0_03dag, 2026-06-10) showed no observable
      duration difference between the two modes, so True is the
      conservative default.

    * (False) — ONE ruckig call across the full path. Intermediate
      waypoints are passed as `inp.intermediate_positions`; ruckig
      optimizes cornering with no forced zero-velocity stops at internal
      corners. Useful for paths with many sharp corners in joint space
      where the per-segment stops produce visibly stuttering motion. For
      the manipulation tasks we've tested this gives no measurable
      speedup vs True, so it's opt-in.

    Args:
        waypoints: (N, DOF) joint-space waypoints.
        max_joint_vel: (DOF,) max joint velocities in rad/s.
        max_joint_acc: (DOF,) max joint accelerations in rad/s^2.
        max_joint_jerk: (DOF,) max joint jerks in rad/s^3.
        control_hz: Output sample rate in Hz.
        sharp_angle_threshold_deg: Used only when segment_at_sharp_corners=True.
        segment_at_sharp_corners: Per-corner zero-velocity-stop mode.
            Default True (historical / safe). Pass False to use a single
            ruckig call across the whole path with no internal forced stops.
        start_vel: (DOF,) initial joint velocity at the FIRST sample. Use the
            policy's last commanded velocity for a smooth handoff at
            intervention trigger time. Default None = zeros (cold start).
        start_acc: (DOF,) initial joint acceleration. Default None = zeros.
        final_approach_dist: Joint-space L2 distance (rad) before the FINAL
            waypoint at which the "final approach" begins: a split waypoint is
            inserted there and every section from it to the goal gets the
            scaled-down vel/acc limits below, via ruckig PER-SECTION limits in
            a single trajectory problem. Ruckig chooses the split-boundary
            velocity itself (time-optimal and feasible), so the profile
            neither stops short of the goal nor enters the approach too fast
            to brake (both failure modes of hand-picking a handoff state).
            Motivation: a uniform-limit time-optimal profile brakes at max
            deceleration right up to the last sample; the PD-tracked physical
            robot carries momentum PAST the goal and gets dragged back by the
            hold — demonstrations then teach the policy to overshoot. A
            low-acceleration final approach is trivial to track, while
            intermediate motion keeps the full limits.
            0.0 (default) disables — identical to historical behavior.
        final_approach_vel_scale: Velocity limit scale for sections inside the
            final approach (only used when final_approach_dist > 0).
        final_approach_acc_scale: Acceleration limit scale for sections inside
            the final approach (only used when final_approach_dist > 0). Jerk
            is not scaled — it only shapes the (now small) accel ramps.
        end_vel: (DOF,) target joint velocity at the FINAL waypoint. Default
            None = zeros (come to rest at the goal — historical behavior).

    Returns:
        (M, DOF) trajectory sampled at control_hz.
    """
    waypoints = np.array(waypoints)
    dof = waypoints.shape[1]
    zeros = np.zeros(dof)
    start_vel = zeros if start_vel is None else np.asarray(start_vel, dtype=np.float64)
    # Ruckig rejects (or behaves badly on) initial states outside the limits;
    # a handoff velocity estimated from noisy history can nick past
    # max_joint_vel. Clamp elementwise — the profile then decelerates from
    # the limit, which is the intended physical behavior anyway.
    start_vel = np.clip(
        start_vel,
        -np.asarray(max_joint_vel, dtype=np.float64),
        np.asarray(max_joint_vel, dtype=np.float64),
    )
    start_acc = zeros if start_acc is None else np.asarray(start_acc, dtype=np.float64)
    end_vel = zeros if end_vel is None else np.asarray(end_vel, dtype=np.float64)

    (
        waypoints,
        max_joint_vel,
        max_joint_acc,
        per_section_vel,
        per_section_acc,
    ) = _prepare_section_limits(
        waypoints,
        max_joint_vel,
        max_joint_acc,
        final_approach_dist,
        final_approach_vel_scale,
        final_approach_acc_scale,
        uniform_path_speed,
    )

    if not segment_at_sharp_corners:
        # Fast path: single ruckig call. `_ruckig_run_segment` already passes
        # waypoints[1:-1] as intermediate_positions, so ruckig handles
        # corner decel internally without zero-velocity stops.
        samples, _, _ = _ruckig_run_segment(
            waypoints,
            start_vel=start_vel,
            start_acc=start_acc,
            end_vel=end_vel,
            end_acc=zeros,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            max_joint_jerk=max_joint_jerk,
            control_hz=control_hz,
            per_section_max_velocity=per_section_vel,
            per_section_max_acceleration=per_section_acc,
        )
        return samples

    # Legacy per-segment mode.
    sharp_indices = _find_sharp_waypoint_indices(waypoints, sharp_angle_threshold_deg)
    split_points = sorted(set([0] + sharp_indices + [len(waypoints) - 1]))
    segments = [
        waypoints[split_points[k]: split_points[k + 1] + 1]
        for k in range(len(split_points) - 1)
    ]
    all_samples = []
    prev_end_vel = start_vel
    prev_end_acc = start_acc
    for seg_idx, seg in enumerate(segments):
        is_last = seg_idx == len(segments) - 1
        # Sharp internal boundaries stop at zero velocity (segments are split
        # exactly at sharp corners); the FINAL point targets the caller's
        # end_vel (zeros by default — come to rest; the final-approach split
        # passes a creep-speed handoff velocity here).
        seg_end_vel = end_vel if is_last else zeros
        seg_end_acc = zeros
        # Slice the global per-section limit arrays (built by the
        # final-approach taper; None when disabled) to this segment's span:
        # global section s covers waypoints[s] -> waypoints[s+1], so segment k
        # (waypoints split_points[k]..split_points[k+1]) owns sections
        # split_points[k]..split_points[k+1]-1.
        _seg_psv = (
            per_section_vel[split_points[seg_idx]: split_points[seg_idx + 1]]
            if per_section_vel is not None else None
        )
        _seg_psa = (
            per_section_acc[split_points[seg_idx]: split_points[seg_idx + 1]]
            if per_section_acc is not None else None
        )
        samples, end_v, end_a = _ruckig_run_segment(
            seg,
            start_vel=prev_end_vel,
            start_acc=prev_end_acc,
            end_vel=seg_end_vel,
            end_acc=seg_end_acc,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            max_joint_jerk=max_joint_jerk,
            control_hz=control_hz,
            per_section_max_velocity=_seg_psv,
            per_section_max_acceleration=_seg_psa,
        )
        if not is_last:
            samples = samples[:-1]
        all_samples.append(samples)
        prev_end_vel = end_v
        prev_end_acc = end_a
    return np.concatenate(all_samples, axis=0)

# ---------------------------------------------------------------------------
# TOPP-RA time parametrization (default backend)
# ---------------------------------------------------------------------------
#
# Why not ruckig: the community ruckig build (0.15.x) solves any problem with
# `intermediate_positions` by calling a CLOUD API — measured at 510-650 ms per
# call on this repo's paths (vs 0.08 ms for a 2-waypoint, intermediate-free
# problem) and rate-limited to 1000 requests/day, which is what
# `RuckigCloudUnavailableError` exists to report. TOPP-RA runs entirely
# locally.
#
# What changes behaviorally:
#   * JERK IS NOT LIMITED. TOPP-RA is a path-velocity method: it optimizes
#     s(t) along a FIXED geometric path subject to velocity/acceleration
#     bounds, and has no third-order term. `max_joint_jerk` is accepted and
#     ignored by the toppra backend. The `ParametrizeConstAccel` sampler
#     produces piecewise-constant acceleration, so jerk spikes at grid
#     boundaries. In practice the grid is fine (~0.01 rad) and the RRT paths
#     are already elastic-smoothed, but if execution looks buzzy this is the
#     first thing to check — `SPLATSIM_TRAJ_BACKEND=ruckig` restores the
#     jerk-limited profile.
#   * `start_acc` is ignored for the same reason (no acceleration boundary
#     condition in a path-velocity formulation).
#   * `start_vel` / `end_vel` are honored only in their component TANGENT to
#     the path (projected onto dq/ds). Any normal component is unrepresentable
#     — the trajectory must follow the collision-checked geometry.
#
# What stays the same: the per-section velocity/acceleration limits produced by
# `_prepare_section_limits` (final-approach taper + uniform_path_speed) are
# applied as path-position-varying constraints, and `segment_at_sharp_corners`
# still forces a zero-velocity stop at each sharp corner.


def _toppra_varying_acc_constraint(alim_func):
    """`JointAccelerationConstraint` with limits that vary along the path.

    toppra ships `JointVelocityConstraintVarying` but no acceleration
    equivalent, and the final-approach taper needs one. Same canonical-linear
    form as the fixed-limit version (a=q', b=q'', F=[I; -I], g=[a_hi; -a_lo]),
    only with `g` recomputed per gridpoint from `alim_func(s)` and
    `identical=False` so the solver reads the per-gridpoint arrays.
    """
    from toppra.constraint import (
        DiscretizationType,
        LinearConstraint,
        canlinear_colloc_to_interpolate,
    )

    class _VaryingAcc(LinearConstraint):
        def __init__(self):
            super().__init__()
            self.dof = int(np.asarray(alim_func(0.0)).shape[0])
            self.identical = False
            # Interpolation (not Collocation): Collocation enforces the bound
            # only AT gridpoints, which let the sampled trajectory overshoot
            # the acceleration limit by ~8% between them.
            self.discretization_type = DiscretizationType.Interpolation
            self._format_string = "    Varying acceleration limit\n"

        def compute_constraint_params(self, path, gridpoints, *args, **kwargs):
            if path.dof != self.dof:
                raise ValueError(
                    f"Wrong dimension: constraint dof ({self.dof}) != path dof ({path.dof})"
                )
            ps = np.asarray(path(gridpoints, order=1)).reshape(-1, self.dof)
            pss = np.asarray(path(gridpoints, order=2)).reshape(-1, self.dof)
            n = len(gridpoints)
            eye = np.eye(self.dof)
            F = np.zeros((n, 2 * self.dof, self.dof))
            F[:, : self.dof, :] = eye
            F[:, self.dof :, :] = -eye
            g = np.zeros((n, 2 * self.dof))
            for i, s in enumerate(gridpoints):
                lim = np.asarray(alim_func(float(s)), dtype=np.float64)
                g[i, : self.dof] = lim[:, 1]
                g[i, self.dof :] = -lim[:, 0]
            if self.discretization_type == DiscretizationType.Collocation:
                return ps, pss, np.zeros_like(ps), F, g, None, None
            return canlinear_colloc_to_interpolate(
                ps, pss, np.zeros_like(ps), F, g, None, None,
                gridpoints, identical=False,
            )

    return _VaryingAcc()


def _toppra_path_speed_constraint(vpath_func):
    """Bound the L2 JOINT-SPACE PATH SPEED |dq/dt| to `vpath_func(s)` rad/s.

    This is what `uniform_path_speed` actually wants, and TOPP-RA can say it
    exactly: |dq/dt| = |dq/ds| * ds/dt, so bounding path speed is the direct
    `xbound` `x = (ds/dt)^2 <= (vpath / |dq/ds|)^2` — no per-joint
    approximation involved. |dq/ds| is evaluated on the SPLINE at each
    gridpoint rather than assumed to be 1: `s` is the CHORD arc length of the
    waypoints, and the spline bows outside those chords, so |dq/ds| runs
    slightly above 1 through corners. Assuming 1 let the sampled path speed
    overshoot the cap by ~40% at the corners.

    The ruckig backend cannot express this (it only has per-joint box limits),
    so `_prepare_section_limits` approximates it by scaling each section's
    per-joint caps by that section's CHORD direction with a 0.1 floor. Under
    TOPP-RA that approximation badly over-constrains: the spline tangent
    diverges from the chord direction through corners, so a joint whose chord
    component was near-zero (capped at 0.1*v_path) picks up real motion on the
    spline and throttles the whole profile — measured ~1.7x longer durations.
    The toppra backend therefore passes `uniform_path_speed=False` to
    `_prepare_section_limits` and uses this constraint instead.
    """
    from toppra.constraint import LinearConstraint

    class _PathSpeed(LinearConstraint):
        def __init__(self):
            super().__init__()
            self.dof = None  # not a per-joint constraint; skips the dof check
            self._format_string = "    Path speed limit\n"

        def get_dof(self):
            return self.dof

        def compute_constraint_params(self, path, gridpoints, *args, **kwargs):
            v = np.array([float(vpath_func(float(s))) for s in gridpoints])
            ps = np.asarray(path(gridpoints, order=1)).reshape(len(gridpoints), -1)
            dqds = np.maximum(np.linalg.norm(ps, axis=1), 1e-9)
            xbound = np.zeros((len(gridpoints), 2))
            xbound[:, 0] = 0.0
            xbound[:, 1] = (v / dqds) ** 2
            return None, None, None, None, None, None, xbound

    return _PathSpeed()


# Pre-blend decimation tolerance (rad, joint-space). See the call sites: on a
# DENSE path (trajopt/elastic output, chords ~0.04 rad) corner blending is
# chord-limited — its cut is capped at 45% of the adjacent chord — so even a
# 30 deg corner gets a ~0.01 rad rounding radius and the parametrizer brakes
# to ~sqrt(accel*0.01) ~= 0.1 rad/s at it. Measured on planar_3joint_10: 69%
# of mid-trajectory stop-dips sat at corners GENTLER than 60 deg. Decimating
# to the polyline shape first restores long chords, so blending can round
# those corners enough to carry cruise speed through them.
DEFAULT_PREBLEND_DECIMATE_EPS = 0.02


def _decimate_for_blending(waypoints: np.ndarray, eps: float) -> np.ndarray:
    """RDP-decimate so consecutive chords are long enough for corner blending.

    Deviation from the original polyline is bounded by `eps`, which stacks
    with the blend budget (0.05) to at most ~0.07 rad — still far below every
    collision clearance, and the parametrized trajectory is re-checked against
    collisions downstream regardless.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.shape[0] < 3 or eps <= 0:
        return wp
    # NOTE: RDP's perpendicular-distance metric deletes collinear REVERSAL
    # spurs (out-and-back along one line = zero deviation from the bypass
    # chord). For mid-path spurs that is desirable — they are RRT noise and
    # keeping them measured 1.2x durations. The one LOAD-BEARING reversal —
    # a braking lead-in prepended for a moving handoff — never reaches this
    # code: retimed_parametrize_path splits it off into a 1-D ruckig brake
    # segment before preprocessing (see the brake-out split there).
    return wp[_rdp_joint_path(wp, eps)]


# Joint-space deviation budget (rad, L2) for rounding path corners before
# time parametrization. See `_blend_corners`.
DEFAULT_CORNER_BLEND_RAD = 0.05


def _blend_corners(
    waypoints: np.ndarray,
    budget: float,
    keep_sharp_deg: float | None = None,
    passes: int = 2,
) -> np.ndarray:
    """Round path corners by chamfering, within a joint-space deviation budget.

    Why: a trajectory pinned to the waypoint polyline CANNOT carry speed
    through a sharp joint-space corner — any joint whose velocity reverses
    there must pass through zero, so the parametrizer (correctly) brakes to
    ~0 at the waypoint no matter what `segment_at_sharp_corners` says.
    Measured on real planner output: corners at 0.000 rad/s with the flag
    False. Carrying speed requires geometry that actually turns, i.e. a
    bounded deviation from the polyline.

    Each pass replaces every interior corner V with two points on its chords
    at cut distance d, chosen so the chamfer's deviation from the original
    polyline (d*sin(theta/2), theta = turn angle) never exceeds the per-pass
    budget; repeated passes round the sub-corners. Endpoints never move, and
    cut distances are capped at 45% of the adjacent chord so neighbouring
    chamfers cannot cross.

    Near-reversals get blended too: a chamfered reversal becomes a small
    U-turn the parametrizer can ROLL through at ~sqrt(accel * radius) instead
    of braking to a dead stop mid-trajectory. (An earlier version skipped
    corners >150 deg on the theory that a reversal must brake anyway; that
    turned every leftover RRT reversal into a recorded full stop — the
    "robot pauses at waypoints" artifact. Full speed through a reversal is
    still impossible, but ~0.15 rad/s through a budget-radius U-turn beats
    0.) The excursion tip is shortened by at most the budget, and the
    parametrized trajectory is re-checked against collisions downstream.
    When `keep_sharp_deg` is set (the segment_at_sharp_corners=True mode),
    corners sharper than it are left alone so they still come to the
    intended full stop.

    The blended path deviates from the collision-CHECKED chords by at most
    `budget`; the parametrized trajectory is re-checked downstream
    (`_smooth_and_check_collision`), so a blend that cuts into an obstacle
    rejects that candidate rather than reaching the robot.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.shape[0] < 3 or budget <= 0:
        return wp
    per_pass = float(budget) / max(1, passes)
    keep_sharp = (np.deg2rad(keep_sharp_deg) if keep_sharp_deg is not None else None)

    for _ in range(passes):
        if wp.shape[0] < 3:
            break
        out = [wp[0]]
        for i in range(1, wp.shape[0] - 1):
            prev_pt, v, next_pt = out[-1], wp[i], wp[i + 1]
            d_in, d_out = v - prev_pt, next_pt - v
            n_in, n_out = float(np.linalg.norm(d_in)), float(np.linalg.norm(d_out))
            if n_in < 1e-9 or n_out < 1e-9:
                out.append(v)
                continue
            cos = float(np.clip(np.dot(d_in / n_in, d_out / n_out), -1.0, 1.0))
            theta = float(np.arccos(cos))
            wants_stop = keep_sharp is not None and theta > keep_sharp
            if theta < np.deg2rad(3.0) or wants_stop:
                out.append(v)
                continue
            half_sin = float(np.sin(0.5 * theta))
            cut = min(per_pass / max(half_sin, 1e-6), 0.45 * n_in, 0.45 * n_out)
            if cut < 1e-9:
                out.append(v)
                continue
            out.append(v - d_in / n_in * cut)
            out.append(v + d_out / n_out * cut)
        out.append(wp[-1])
        wp = np.asarray(out, dtype=np.float64)
    return wp


# Max joint-space L2 gap (rad) between waypoints handed to a time
# parametrization. Matches the RRT collision-check resolution (`resolutions =
# [0.05] * dof`), i.e. the spacing at which the path was actually validated.
DEFAULT_PARAM_DENSIFY_STEP = 0.05


def _densify_waypoints(waypoints: np.ndarray, max_step: float) -> np.ndarray:
    """Insert collinear points so no segment is longer than `max_step` (L2, rad).

    Both parametrization backends interpolate BETWEEN waypoints, and neither
    reproduces the straight chord exactly: toppra fits a cubic spline (measured
    up to 0.41 rad off the polyline), and chained ruckig uses
    `Synchronization.Time`, which matches each segment's duration across joints
    but not its shape (up to 0.10 rad). That bow is unvalidated geometry — the
    collision gate cleared the CHORDS — so it shows up as the parametrized path
    colliding where the linear densify check passed, which forces the planner
    onto a farther IK goal and a much longer chunk.

    Deviation shrinks with segment length, so subdividing bounds it directly
    while changing nothing about the geometry: the inserted points lie exactly
    on the chords that were already checked.

    This is NOT redundant with the densify inside `trajopt_smooth_path` /
    `elastic_smooth_path`. Those densify to optimise, then RDP-DECIMATE back
    down (to `decimate_eps` 0.04 / 0.015) before returning, so the parametrizer
    still receives a sparse path. The decimation exists because the ruckig
    CLOUD API warns above ~15 intermediate waypoints — a constraint that does
    not apply to the chained backend (independent single-section solves) or to
    toppra (one spline fit).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.shape[0] < 2 or max_step <= 0:
        return wp
    out = [wp[0]]
    for a, b in zip(wp[:-1], wp[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / max_step))
        for k in range(1, max(n, 1) + 1):
            out.append(a + (b - a) * (k / max(n, 1)))
    return np.asarray(out, dtype=np.float64)


def _dedupe_waypoints(waypoints: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive waypoints closer than `tol` (L2, rad).

    toppra's `SplineInterpolator` needs a strictly increasing path coordinate,
    so zero-length sections — which RRT + RDP decimation do occasionally
    produce, and which the final-approach split can create — must go.
    """
    keep = [0]
    for i in range(1, waypoints.shape[0]):
        if np.linalg.norm(waypoints[i] - waypoints[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) < waypoints.shape[0]:
        # Always preserve the goal: if the last waypoint got merged away the
        # trajectory would silently stop short of it.
        if keep[-1] != waypoints.shape[0] - 1:
            keep[-1] = waypoints.shape[0] - 1
    return waypoints[keep]


def _toppra_run_segment(
    waypoints: np.ndarray,
    sd_start: float,
    sd_end: float,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    control_hz: float,
    per_section_max_velocity: list | None = None,
    per_section_max_acceleration: list | None = None,
    per_section_path_speed: list | None = None,
    gridpoint_spacing: float = 0.01,
) -> tuple:
    """Time-parametrize one geometric segment with TOPP-RA.

    Returns ``(samples, end_path_speed)`` where `samples` is (M, DOF) sampled
    at `control_hz` and `end_path_speed` is the achieved ds/dt at the final
    waypoint (used to chain segments in `segment_at_sharp_corners` mode).

    `per_section_max_velocity` / `per_section_max_acceleration` follow the
    same contract as `_ruckig_run_segment`: one DOF-length entry per waypoint
    gap, or None for uniform limits. `per_section_path_speed` is one SCALAR
    per gap bounding the L2 path speed (see `_toppra_path_speed_constraint`),
    or None to leave path speed unbounded beyond the per-joint caps.
    """
    import toppra as ta
    import toppra.algorithm as ta_algo
    from toppra.constraint import (
        DiscretizationType,
        JointAccelerationConstraint,
        JointVelocityConstraint,
        JointVelocityConstraintVarying,
    )

    dof = waypoints.shape[1]
    # Path coordinate = cumulative joint-space L2 arc length, so `s` is in
    # radians and the gridpoint spacing below is directly interpretable.
    seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(ss[-1])

    def _section_of(s: float) -> int:
        return int(np.clip(np.searchsorted(ss, s, side="right") - 1, 0, len(ss) - 2))

    # Cubic spline through the waypoints: C² geometry with rounded corners,
    # the same cornering behavior the parametrizer's `intermediate_positions` gave. The
    # spline can bow slightly outside the collision-checked chord, which is
    # why callers re-check the parametrized output (see
    # RRTToGoalPlanner._smooth_and_check_collision and
    # `obstacle_clearance_factor`).
    path = ta.SplineInterpolator(ss.tolist(), waypoints)

    if per_section_max_velocity is None:
        vel_c = JointVelocityConstraint(
            np.stack([-max_joint_vel, max_joint_vel], axis=1)
        )
    else:
        psv = np.asarray(per_section_max_velocity, dtype=np.float64)

        def _vlim(s):
            v = psv[_section_of(s)]
            return np.stack([-v, v], axis=1)

        vel_c = JointVelocityConstraintVarying(_vlim)

    if per_section_max_acceleration is None:
        acc_c = JointAccelerationConstraint(
            np.stack([-max_joint_acc, max_joint_acc], axis=1),
            discretization_scheme=DiscretizationType.Interpolation,
        )
    else:
        psa = np.asarray(per_section_max_acceleration, dtype=np.float64)

        def _alim(s):
            a = psa[_section_of(s)]
            return np.stack([-a, a], axis=1)

        acc_c = _toppra_varying_acc_constraint(_alim)

    constraints = [vel_c, acc_c]
    if per_section_path_speed is not None:
        psp = np.asarray(per_section_path_speed, dtype=np.float64)
        constraints.append(
            _toppra_path_speed_constraint(lambda s: psp[_section_of(s)])
        )

    n_grid = int(np.clip(total / gridpoint_spacing, 50, 2000))
    gridpoints = np.linspace(0.0, total, n_grid)

    instance = ta_algo.TOPPRA(
        constraints, path, gridpoints=gridpoints, parametrizer="ParametrizeConstAccel"
    )
    jnt_traj = instance.compute_trajectory(sd_start, sd_end)
    if jnt_traj is None:
        raise RuntimeError(
            f"TOPP-RA failed to parametrize a {waypoints.shape[0]}-waypoint path "
            f"(arc length {total:.3f} rad, sd_start={sd_start:.3f}, sd_end={sd_end:.3f})"
        )

    dt = 1.0 / control_hz
    duration = float(jnt_traj.duration)
    ts = np.arange(0.0, duration, dt)
    ts = np.append(ts, duration)
    samples = np.asarray(jnt_traj(ts)).reshape(-1, dof)
    # ds/dt at the end, for chaining: |dq/dt| / |dq/ds| at s = total.
    qd_end = np.asarray(jnt_traj(duration, 1)).reshape(-1)
    ps_end = np.asarray(path(total, 1)).reshape(-1)
    denom = float(np.dot(ps_end, ps_end))
    end_path_speed = float(np.dot(qd_end, ps_end) / denom) if denom > 1e-12 else 0.0
    return samples, max(end_path_speed, 0.0)


_TOPPRA_LOGGING_CONFIGURED = False


def toppra_parametrize_path(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    sharp_angle_threshold_deg: float = 45.0,
    segment_at_sharp_corners: bool = True,
    start_vel: np.ndarray | None = None,
    start_acc: np.ndarray | None = None,
    final_approach_dist: float = 0.0,
    final_approach_vel_scale: float = 0.3,
    final_approach_acc_scale: float = 0.25,
    end_vel: np.ndarray | None = None,
    uniform_path_speed: bool = False,
    densify_step: float = DEFAULT_PARAM_DENSIFY_STEP,
    corner_blend_rad: float = DEFAULT_CORNER_BLEND_RAD,
) -> np.ndarray:
    """Time-optimal path parametrization using TOPP-RA. Local, no network.

    Drop-in replacement for `ruckig_parametrize_path` — same signature, same
    (M, DOF) output sampled at `control_hz`. See the module comment above
    "TOPP-RA time parametrization" for the behavioral differences
    (`max_joint_jerk` and `start_acc` are accepted and IGNORED; `start_vel` /
    `end_vel` are honored only tangentially to the path).

    `segment_at_sharp_corners` decides what happens at path corners:

      * False (the config default) — CARRY SPEED. Corners are rounded within
        a `corner_blend_rad` joint-space deviation budget (`_blend_corners`)
        and the whole path is parametrized as one problem, so the trajectory
        flows through waypoints near cruise speed. Splitting was never the
        only thing standing between a waypoint and a stop: a polyline-pinned
        trajectory must brake to ~0 at any sharp corner regardless, which is
        why blending — not just not-splitting — is what implements this
        flag's intent. Near-reversals (>150 deg) still brake: a joint whose
        velocity changes sign must pass through zero.
      * True — STOP at corners sharper than `sharp_angle_threshold_deg`
        (split into separate problems with zero boundary speed); milder
        corners are still blended and carried.
    """
    global _TOPPRA_LOGGING_CONFIGURED
    if not _TOPPRA_LOGGING_CONFIGURED:
        # toppra logs one INFO line per solve; quiet it once per process
        # rather than reconfiguring logging on every call.
        import toppra as ta

        ta.setup_logging("WARNING")
        _TOPPRA_LOGGING_CONFIGURED = True

    waypoints = np.asarray(waypoints, dtype=np.float64)
    dof = waypoints.shape[1]
    zeros = np.zeros(dof)
    max_joint_vel = np.asarray(max_joint_vel, dtype=np.float64)
    max_joint_acc = np.asarray(max_joint_acc, dtype=np.float64)
    start_vel = zeros if start_vel is None else np.clip(
        np.asarray(start_vel, dtype=np.float64), -max_joint_vel, max_joint_vel
    )
    end_vel = zeros if end_vel is None else np.asarray(end_vel, dtype=np.float64)

    # Dedupe BEFORE building section limits so section indices line up with
    # the waypoint array the spline is built on.
    waypoints = _dedupe_waypoints(waypoints)
    if waypoints.shape[0] < 2:
        return waypoints
    # Strip redundant dense waypoints FIRST so blending is not chord-limited
    # (see DEFAULT_PREBLEND_DECIMATE_EPS), then round corners (bounded
    # deviation) so speed can be CARRIED through them — a polyline-pinned
    # trajectory must brake to ~0 at any sharp corner. With
    # segment_at_sharp_corners=True, corners past the threshold stay sharp so
    # they still stop; everything milder is blended either way.
    waypoints = _decimate_for_blending(waypoints, DEFAULT_PREBLEND_DECIMATE_EPS)
    waypoints = _blend_corners(
        waypoints, corner_blend_rad,
        keep_sharp_deg=sharp_angle_threshold_deg if segment_at_sharp_corners else None,
    )
    # Merge near-duplicate knots the blend can leave around a SHORT reversal
    # chord (its cut is capped at 45% of the chord, so a ~0.01 rad reversal
    # hop yields knots ~0.004 rad apart across a fold). A cubic spline through
    # such a cluster develops a huge |dq/ds| spike, and the joint-velocity
    # constraint (sd*|q'| <= vmax) then pins the path speed to ~0 across it —
    # observed as the robot PARKED for hundreds of samples mid-trajectory or
    # at the goal (planar_3joint_9 ep75: ~600 frames). Merging at 0.4x the
    # densify spacing (~20 mrad at defaults, still far below every clearance)
    # removes the pathology at its source; 0.1x proved too fine once
    # REVERSAL corners started being blended, whose residue pairs land in the
    # 5-20 mrad gap (1/300 random fold geometries re-pinched).
    waypoints = _dedupe_waypoints(waypoints, tol=max(1e-9, 0.4 * densify_step))
    # Bound how far the spline can bow off the (now corner-blended) chords.
    waypoints = _densify_waypoints(waypoints, densify_step)

    # uniform_path_speed is handled NATIVELY below (as an exact path-speed
    # bound), not by the parametrizer's per-joint chord-direction approximation — see
    # `_toppra_path_speed_constraint` for why that approximation is harmful
    # here. Pass False so `_prepare_section_limits` only builds the
    # final-approach taper.
    (
        waypoints,
        max_joint_vel,
        max_joint_acc,
        per_section_vel,
        per_section_acc,
    ) = _prepare_section_limits(
        waypoints,
        max_joint_vel,
        max_joint_acc,
        final_approach_dist,
        final_approach_vel_scale,
        final_approach_acc_scale,
        uniform_path_speed=False,
    )
    waypoints = np.asarray(waypoints, dtype=np.float64)

    # L2 path-speed cap per section. Matches the ruckig backend's intent:
    # v_path for a section is the smallest per-joint velocity cap in force
    # there, so the final-approach taper's scaled sections creep and the rest
    # run at full speed.
    per_section_path_speed = None
    if uniform_path_speed and waypoints.shape[0] >= 2:
        n_sections = waypoints.shape[0] - 1
        if per_section_vel is None:
            per_section_path_speed = [float(np.min(max_joint_vel))] * n_sections
        else:
            per_section_path_speed = [
                float(np.min(np.asarray(v, dtype=np.float64))) for v in per_section_vel
            ]

    def _tangential_speed(vel: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Project a joint velocity onto the local path direction, in ds/dt.

        `s` is arc length, so |dq/ds| = 1 and ds/dt is just the tangential
        component's magnitude. Negative (backwards along the path) clamps to
        0 — TOPP-RA cannot start by moving away from the goal.
        """
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return 0.0
        return max(float(np.dot(vel, d / n)), 0.0)

    if not segment_at_sharp_corners:
        sd_start = _tangential_speed(start_vel, waypoints[0], waypoints[1])
        sd_end = _tangential_speed(end_vel, waypoints[-2], waypoints[-1])
        # A hot handoff velocity can make the instance uncontrollable (TOPP-RA
        # cannot continue at that entry speed along this geometry — e.g. a
        # sharp turn right after the start). Old cloud-ruckig accepted such
        # starts, so no-lookback intervention plans would otherwise crash
        # here. Halve the entry speed and retry, ending at a full stop: the
        # trajectory then begins slower than the robot's true velocity, which
        # the follower backend absorbs smoothly (it tracks from the TRUE
        # state) and a PD controller merely brakes into.
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                samples, _ = _toppra_run_segment(
                    waypoints,
                    sd_start=sd_start * (0.5 ** attempt) if attempt < 3 else 0.0,
                    sd_end=sd_end,
                    max_joint_vel=max_joint_vel,
                    max_joint_acc=max_joint_acc,
                    control_hz=control_hz,
                    per_section_max_velocity=per_section_vel,
                    per_section_max_acceleration=per_section_acc,
                    per_section_path_speed=per_section_path_speed,
                )
                return samples
            except RuntimeError as e:
                last_err = e
                if sd_start <= 1e-9:
                    break  # not an entry-speed problem — retrying cannot help
        raise last_err

    # Legacy per-segment mode: zero path speed at each sharp corner. Mirrors
    # the ruckig backend's split, including the per-section limit slicing.
    sharp_indices = _find_sharp_waypoint_indices(waypoints, sharp_angle_threshold_deg)
    split_points = sorted(set([0] + sharp_indices + [len(waypoints) - 1]))
    all_samples = []
    prev_sd = _tangential_speed(start_vel, waypoints[0], waypoints[1])
    for k in range(len(split_points) - 1):
        lo, hi = split_points[k], split_points[k + 1]
        seg = waypoints[lo: hi + 1]
        is_last = k == len(split_points) - 2
        sd_end = (
            _tangential_speed(end_vel, waypoints[-2], waypoints[-1]) if is_last else 0.0
        )
        _seg_psv = per_section_vel[lo:hi] if per_section_vel is not None else None
        _seg_psa = per_section_acc[lo:hi] if per_section_acc is not None else None
        _seg_psp = (
            per_section_path_speed[lo:hi]
            if per_section_path_speed is not None else None
        )
        samples, end_sd = _toppra_run_segment(
            seg,
            sd_start=prev_sd,
            sd_end=sd_end,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            control_hz=control_hz,
            per_section_max_velocity=_seg_psv,
            per_section_max_acceleration=_seg_psa,
            per_section_path_speed=_seg_psp,
        )
        if not is_last:
            samples = samples[:-1]
        all_samples.append(samples)
        prev_sd = end_sd
    return np.concatenate(all_samples, axis=0)


# ---------------------------------------------------------------------------
# Chained offline ruckig (jerk-limited, local)
# ---------------------------------------------------------------------------
#
# Why a third backend. The other two each give up something:
#
#   ruckig (cloud)  jerk-limited and time-optimal through the intermediate
#                   waypoints, but the community build solves any
#                   intermediate-waypoint problem via a network call — ~500 ms
#                   each, capped at 1000/day.
#   toppra          local and fast, but has no third-order term: the optimal
#                   s(t) is bang-bang in acceleration, so `max_joint_jerk` is
#                   silently ignored. Measured on real planner output: jerk up
#                   to 44 against a limit of 10, which a PD-tracked robot
#                   renders as ringing around the goal.
#
# This backend gets both by decomposing the problem: ONE single-section ruckig
# solve per waypoint pair. With zero intermediate positions ruckig stays local
# (~0.02 ms), and each section is a proper jerk-limited third-order profile.
# The cost is optimality — velocities at interior waypoints are chosen by a
# local heuristic instead of a global optimisation, so trajectories run a bit
# longer than the cloud's. On this repo's planner output:
#
#   toppra   vel 0.5 acc 1.00   5.32 s   jerk 44.13   RINGS
#   toppra   vel 1.0 acc 0.25   8.15 s   jerk  9.49   clean (the cheapest way
#                                                      to make toppra clean)
#   chained  vel 1.0 acc 1.00   6.60 s   jerk 10.00   clean
#
# NOTE: the table above predates the densify/lookahead rework; measured on
# real planner output afterwards, chained pays 2-5x duration for its
# construction-guaranteed jerk limit. The "follower" backend (toppra pacing +
# online-ruckig tracking, further down) reaches the same limits at toppra
# speed and is the default; chained remains for callers that want the jerk
# guarantee to hold by construction rather than by tracking.


def _polyline_speed_targets(
    points: np.ndarray,
    sec_of_gap: list[int],
    sec_vel: list[np.ndarray],
    sec_acc: list[np.ndarray],
    start_vel: np.ndarray,
    end_vel: np.ndarray,
    stop_indices: set[int],
    max_joint_jerk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamically consistent (velocity, acceleration) targets at every point
    of a densified polyline — the lookahead pass that makes chained ruckig
    fast on dense waypoints.

    Chaining single-section ruckig solves needs a target STATE at each interior
    point. The first version targeted a central-difference velocity and ZERO
    acceleration; on a sparse path that is merely suboptimal, but on a
    densified one (a point every 0.05 rad) it forbids sustaining acceleration
    across points, so the arm saw-tooths and the duration blew up 2.6x the
    moment densification was enabled.

    This computes a classic forward/backward speed profile along the polyline
    instead (the CNC-lookahead pattern):

      * per-point direction u_k: normalized central difference (chord bisector
        at corners, the chord itself on collinear densified runs);
      * per-point caps: V_k = min_j vel_cap_j/|u_kj| and likewise A_k — the
        exact path-speed/path-accel the per-joint box limits allow along u_k,
        honouring per-SECTION limits (final-approach taper, uniform speed);
      * corner cap: v <= sqrt(A * l / theta) for turn angle theta over blend
        length l (the two adjacent gaps), so direction changes get absorbed
        within the acceleration budget instead of demanding an accel spike;
      * forward then backward pass with v_{k+1}^2 <= v_k^2 + 2*A*ds, seeded
        with the projections of the true start/end velocities;
      * acceleration target from the finished profile:
        a_k = u_k * (v_{k+1}^2 - v_k^2) / (2*ds_k), clamped to +-A_k.

    Jerk is deliberately absent here: each ruckig section enforces it exactly,
    stretching its own duration when a target is jerk-tight. The pass only has
    to be accel-consistent for that to stay near-optimal.

    Returns (vel_targets, acc_targets), each (N, dof). Callers should override
    the final row with the true end state (the projection loses any component
    of a nonzero handoff velocity normal to the last chord).
    """
    n_pts = points.shape[0]
    chords = np.diff(points, axis=0)
    ds = np.linalg.norm(chords, axis=1)
    u_gap = chords / np.maximum(ds[:, None], 1e-12)

    u_pt = np.empty_like(points)
    u_pt[0], u_pt[-1] = u_gap[0], u_gap[-1]
    for k in range(1, n_pts - 1):
        m = u_gap[k - 1] + u_gap[k]
        nm = float(np.linalg.norm(m))
        u_pt[k] = m / nm if nm > 1e-9 else u_gap[k]

    V = np.empty(n_pts)
    A = np.empty(n_pts)
    for k in range(n_pts):
        gaps = [g for g in (k - 1, k) if 0 <= g < n_pts - 1]
        vel_cap = np.min([sec_vel[sec_of_gap[g]] for g in gaps], axis=0)
        acc_cap = np.min([sec_acc[sec_of_gap[g]] for g in gaps], axis=0)
        au = np.maximum(np.abs(u_pt[k]), 1e-9)
        V[k] = float(np.min(vel_cap / au))
        A[k] = float(np.min(acc_cap / au))

    for k in range(1, n_pts - 1):
        cos = float(np.clip(np.dot(u_gap[k - 1], u_gap[k]), -1.0, 1.0))
        theta = float(np.arccos(cos))
        if theta > 1e-6:
            blend = float(min(ds[k - 1], ds[k]))
            V[k] = min(V[k], float(np.sqrt(max(A[k], 1e-9) * blend / theta)))
    for k in stop_indices:
        if 0 <= k < n_pts:
            V[k] = 0.0

    v = V.copy()
    v[0] = min(V[0], max(0.0, float(np.dot(start_vel, u_pt[0]))))
    v[-1] = min(V[-1], max(0.0, float(np.dot(end_vel, u_pt[-1]))))
    for k in range(n_pts - 1):                     # forward
        a = min(A[k], A[k + 1])
        v[k + 1] = min(v[k + 1], float(np.sqrt(v[k] ** 2 + 2.0 * a * ds[k])))
    for k in range(n_pts - 2, -1, -1):             # backward
        a = min(A[k], A[k + 1])
        v[k] = min(v[k], float(np.sqrt(v[k + 1] ** 2 + 2.0 * a * ds[k])))

    a_s = np.zeros(n_pts)
    for k in range(n_pts - 1):
        if ds[k] > 1e-12:
            a_s[k] = float(np.clip(
                (v[k + 1] ** 2 - v[k] ** 2) / (2.0 * ds[k]), -A[k], A[k]))
    vel_t = u_pt * v[:, None]
    acc_t = u_pt * a_s[:, None]

    # Ruckig VALIDATES target states, not just trajectories: any target with
    # |v| + a^2/(2*jerk) > v_cap on some joint is rejected outright,
    # WHICHEVER way the acceleration points. Toward the cap, the jerk-limited
    # ramp-down of `a` would overshoot the velocity past the cap AFTER the
    # target; away from the cap (a decelerating joint arriving at cap speed),
    # reaching that acceleration means the velocity was beyond the cap just
    # BEFORE it ("will inevitably have reached a velocity ... that will
    # undercut its minimum velocity limit"). Both directions bind, so clamp
    # unconditionally: |a| <= sqrt(2 * jerk * (v_cap - |v|)). The visible cost
    # is that a joint cruising exactly at its cap gets a zero-acceleration
    # target — ruckig then shapes the entry/exit inside the section instead.
    for k in range(n_pts):
        gaps = [g for g in (k - 1, k) if 0 <= g < n_pts - 1]
        vel_cap = np.min([sec_vel[sec_of_gap[g]] for g in gaps], axis=0)
        headroom = np.maximum(vel_cap - np.abs(vel_t[k]), 0.0)
        a_allow = np.sqrt(2.0 * max_joint_jerk * headroom)
        acc_t[k] = np.clip(acc_t[k], -a_allow, a_allow)
    return vel_t, acc_t


def _chained_solve_section(
    p0, p1, v0, a0, v1, a1, vmax, amax, jmax, control_hz,
):
    """One ruckig section, zero intermediate waypoints — always solved LOCALLY.

    `Synchronization.Time` rather than `Phase`: phase sync locks every joint to
    one scaled profile, which cannot represent a boundary velocity that is not
    parallel to the section's displacement — exactly the case at a blended
    interior waypoint.
    """
    from ruckig import (  # type: ignore
        ControlInterface, InputParameter, Ruckig, Synchronization, Trajectory,
    )

    dof = len(p0)
    otg = Ruckig(dof, 1.0 / control_hz, 0)
    inp = InputParameter(dof)
    inp.control_interface = ControlInterface.Position
    inp.synchronization = Synchronization.Time
    inp.current_position = list(p0)
    inp.current_velocity = list(v0)
    inp.current_acceleration = list(a0)
    inp.target_position = list(p1)
    inp.target_velocity = list(v1)
    inp.target_acceleration = list(a1)
    inp.max_velocity = list(vmax)
    inp.max_acceleration = list(amax)
    inp.max_jerk = list(jmax)
    traj = Trajectory(dof)
    result = otg.calculate(inp, traj)
    if result < 0:
        raise RuntimeError(
            f"chained ruckig section failed (result {result}): "
            f"|dp|={np.linalg.norm(np.asarray(p1) - np.asarray(p0)):.4f}"
        )
    return traj


def _sample_chain_on_global_grid(trajs, total_duration, dof, control_hz):
    """Sample a list of (start_time, Trajectory) on ONE uniform time grid.

    Sampling each section separately and concatenating is wrong: a section's
    duration is almost never a multiple of dt, so every join ends up with two
    samples less than dt apart. Anything that differentiates the result (a
    controller, or a jerk metric) reads that spacing as dt and reports a spike
    the trajectory does not actually contain. One global grid removes the
    artifact by construction.
    """
    dt = 1.0 / control_hz
    ts = np.append(np.arange(0.0, total_duration, dt), total_duration)
    out = np.empty((len(ts), dof), dtype=np.float64)
    idx = 0
    for i, t in enumerate(ts):
        while idx + 1 < len(trajs) and t > trajs[idx][0] + trajs[idx][1].duration + 1e-12:
            idx += 1
        t0, traj = trajs[idx]
        out[i] = traj.at_time(float(np.clip(t - t0, 0.0, traj.duration)))[0]
    return out


def chained_ruckig_parametrize_path(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    sharp_angle_threshold_deg: float = 45.0,
    segment_at_sharp_corners: bool = True,
    start_vel: np.ndarray | None = None,
    start_acc: np.ndarray | None = None,
    final_approach_dist: float = 0.0,
    final_approach_vel_scale: float = 0.3,
    final_approach_acc_scale: float = 0.25,
    end_vel: np.ndarray | None = None,
    uniform_path_speed: bool = False,
    densify_step: float = DEFAULT_PARAM_DENSIFY_STEP,
    corner_blend_rad: float = DEFAULT_CORNER_BLEND_RAD,
) -> np.ndarray:
    """Jerk-limited time parametrization, solved locally. Drop-in for the others.

    Unlike the toppra backend this honours `max_joint_jerk`, and unlike the
    ruckig backend it never touches the network: the path is densified to
    `densify_step` and each gap becomes ONE zero-intermediate ruckig section
    (always solved locally), with the section boundary states supplied by a
    forward/backward lookahead pass (`_polyline_speed_targets`). Densification
    bounds how far the Time-synchronized profile can bow off the
    collision-checked chords — the inserted points lie exactly ON those chords
    — and the lookahead is what keeps a dense chain fast: it lets acceleration
    carry across section boundaries instead of resetting to zero at each one.

    `segment_at_sharp_corners=True` forces a full stop at corners sharper
    than `sharp_angle_threshold_deg`; with False (the config default) corners
    are blended within `corner_blend_rad` and the lookahead's corner cap
    slows only as much as the (now shallow) turn demands.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    dof = waypoints.shape[1]
    zeros = np.zeros(dof)
    max_joint_vel = np.asarray(max_joint_vel, dtype=np.float64)
    max_joint_acc = np.asarray(max_joint_acc, dtype=np.float64)
    max_joint_jerk = np.asarray(max_joint_jerk, dtype=np.float64)
    start_vel = zeros if start_vel is None else np.clip(
        np.asarray(start_vel, dtype=np.float64), -max_joint_vel, max_joint_vel
    )
    start_acc = zeros if start_acc is None else np.asarray(start_acc, dtype=np.float64)
    end_vel = zeros if end_vel is None else np.asarray(end_vel, dtype=np.float64)

    waypoints = _dedupe_waypoints(waypoints)
    if waypoints.shape[0] < 2:
        return waypoints
    # Decimate-then-round (see toppra backend): dense waypoints chord-limit
    # the blend, so strip them to the polyline shape first; the lookahead's
    # corner cap then sees shallow turns instead of braking at every waypoint.
    waypoints = _decimate_for_blending(waypoints, DEFAULT_PREBLEND_DECIMATE_EPS)
    waypoints = _blend_corners(
        waypoints, corner_blend_rad,
        keep_sharp_deg=sharp_angle_threshold_deg if segment_at_sharp_corners else None,
    )
    # Merge blend residue around short reversal chords — see the toppra
    # backend for the parked-robot pathology this prevents.
    waypoints = _dedupe_waypoints(waypoints, tol=max(1e-9, 0.4 * densify_step))

    # Taper / uniform-speed section limits on the SPARSE waypoints first, so
    # section indices line up with the geometry they describe; densify after,
    # tracking which original section every mini-gap belongs to.
    (
        waypoints,
        max_joint_vel,
        max_joint_acc,
        per_section_vel,
        per_section_acc,
    ) = _prepare_section_limits(
        waypoints,
        max_joint_vel,
        max_joint_acc,
        final_approach_dist,
        final_approach_vel_scale,
        final_approach_acc_scale,
        uniform_path_speed,
    )
    waypoints = np.asarray(waypoints, dtype=np.float64)
    n_orig_sections = waypoints.shape[0] - 1
    sec_vel = [
        np.asarray(per_section_vel[i], dtype=np.float64) if per_section_vel is not None
        else max_joint_vel
        for i in range(n_orig_sections)
    ]
    sec_acc = [
        np.asarray(per_section_acc[i], dtype=np.float64) if per_section_acc is not None
        else max_joint_acc
        for i in range(n_orig_sections)
    ]

    sharp = set(
        _find_sharp_waypoint_indices(waypoints, sharp_angle_threshold_deg)
        if segment_at_sharp_corners else []
    )

    # Densify each original section, remembering (a) the original section of
    # every mini-gap (for per-section limits) and (b) where the original
    # waypoints landed (for sharp-corner stops).
    step = float(densify_step) if densify_step and densify_step > 0 else np.inf
    pts: list[np.ndarray] = [waypoints[0]]
    sec_of_gap: list[int] = []
    orig_index_of: list[int] = [0]
    for i in range(n_orig_sections):
        a, b = waypoints[i], waypoints[i + 1]
        n_sub = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n_sub + 1):
            pts.append(a + (b - a) * (k / n_sub))
            sec_of_gap.append(i)
        orig_index_of.append(len(pts) - 1)
    points = np.asarray(pts, dtype=np.float64)
    stop_indices = {orig_index_of[i] for i in sharp}

    vel_targets, acc_targets = _polyline_speed_targets(
        points, sec_of_gap, sec_vel, sec_acc, start_vel, end_vel, stop_indices,
        max_joint_jerk,
    )
    # True end state, not its projection onto the last chord.
    vel_targets[-1] = np.clip(end_vel, -max_joint_vel, max_joint_vel)
    acc_targets[-1] = zeros

    trajs: list = []
    t_start = 0.0
    v_cur, a_cur = start_vel, start_acc
    for g in range(points.shape[0] - 1):
        sec = sec_of_gap[g]
        traj = _chained_solve_section(
            points[g], points[g + 1], v_cur, a_cur,
            vel_targets[g + 1], acc_targets[g + 1],
            sec_vel[sec], sec_acc[sec], max_joint_jerk, control_hz,
        )
        trajs.append((t_start, traj))
        t_start += traj.duration
        v_cur = np.asarray(traj.at_time(traj.duration)[1], dtype=np.float64)
        a_cur = np.asarray(traj.at_time(traj.duration)[2], dtype=np.float64)

    return _sample_chain_on_global_grid(trajs, t_start, dof, control_hz)


def follower_parametrize_path(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    settle_timeout_s: float = 3.0,
    target_lead_ticks: int = 2,
    **toppra_kwargs,
) -> np.ndarray:
    """TOPP-RA pacing + a LOCAL online-ruckig follower for the jerk limit.

    Decomposition: toppra already produces the near-time-optimal, chord-tight
    schedule — its one defect is stepwise acceleration (jerk spikes of ~44
    against a limit of 10, which a PD-tracked robot renders as ringing).
    Rather than re-deriving the schedule under a jerk limit (the chained
    backend; measured 2-8x slower trajectories), track toppra's samples with
    ruckig in ONLINE mode: one `update()` per control tick toward the current
    reference sample, target velocity from the reference's finite difference.
    Single-target updates are solved locally by community ruckig — only
    intermediate-waypoint problems use its cloud API — and each tick replans
    MPC-style, so the follower lags the reference only where the reference
    itself violates the jerk limit and rejoins immediately after.

    Two details keep the follower tight to the reference (measured 24 mm ->
    14 mm of worst-case EE deviation from the collision-checked chords on real
    planner output): the target leads the clock by `target_lead_ticks` so the
    follower is not systematically a step behind, and the reference's own
    finite-difference acceleration is fed as the target acceleration, clamped
    per joint to sqrt(2*jerk*(v_cap-|v|)) — the largest value ruckig's
    target-state validation admits at that speed.

    After the reference ends, the follower keeps stepping toward the final
    sample until it settles (position within ~1e-4 rad, velocity ~0), bounded
    by `settle_timeout_s`; the tail is typically a few ticks.

    All `toppra_kwargs` (densify_step, taper, uniform_path_speed, start_vel,
    ...) pass straight through to `toppra_parametrize_path`.
    """
    from ruckig import (  # type: ignore
        ControlInterface, InputParameter, OutputParameter, Result, Ruckig,
        RuckigError,
    )

    ref = toppra_parametrize_path(
        waypoints, max_joint_vel, max_joint_acc, max_joint_jerk, control_hz,
        **toppra_kwargs,
    )
    ref = np.asarray(ref, dtype=np.float64)
    if ref.shape[0] < 3:
        return ref
    n_ref, dof = ref.shape
    dt = 1.0 / control_hz
    vel_cap = np.asarray(max_joint_vel, dtype=np.float64)

    # Reference velocity by central difference, clamped inside the cap so the
    # target state always passes ruckig's validation.
    v_ref = np.zeros_like(ref)
    v_ref[1:-1] = (ref[2:] - ref[:-2]) * (0.5 * control_hz)
    v_ref = np.clip(v_ref, -vel_cap, vel_cap)
    acc_cap = np.asarray(max_joint_acc, dtype=np.float64)
    jerk_cap = np.asarray(max_joint_jerk, dtype=np.float64)
    a_ref = np.zeros_like(ref)
    a_ref[1:-1] = (ref[2:] - 2.0 * ref[1:-1] + ref[:-2]) * (control_hz ** 2)
    a_allow = np.minimum(
        np.sqrt(2.0 * jerk_cap * np.maximum(vel_cap - np.abs(v_ref), 0.0)),
        acc_cap,
    )
    a_ref = np.clip(a_ref, -a_allow, a_allow)

    # The follower's own state is pinned strictly INSIDE the caps: a current
    # velocity exactly AT max_velocity to the last bit (which out.new_velocity
    # can legitimately produce, and a caller-supplied start_vel can hit) makes
    # ruckig's step-1 synchronization throw RuckigError on some targets. The
    # margin is orders of magnitude below anything physical.
    vel_feed_cap = vel_cap * (1.0 - 1e-6)
    acc_feed_cap = acc_cap * (1.0 - 1e-6)

    start_vel = toppra_kwargs.get("start_vel")
    otg = Ruckig(dof, dt)
    inp = InputParameter(dof)
    out = OutputParameter(dof)
    inp.control_interface = ControlInterface.Position
    inp.current_position = ref[0].tolist()
    inp.current_velocity = (
        np.clip(np.asarray(start_vel, dtype=np.float64), -vel_feed_cap, vel_feed_cap)
        if start_vel is not None else np.zeros(dof)
    ).tolist()
    inp.current_acceleration = [0.0] * dof
    inp.max_velocity = vel_cap.tolist()
    inp.max_acceleration = np.asarray(max_joint_acc, dtype=np.float64).tolist()
    inp.max_jerk = np.asarray(max_joint_jerk, dtype=np.float64).tolist()

    samples = [ref[0].copy()]
    max_ticks = n_ref + int(settle_timeout_s * control_hz)
    # KNOWN DEFECT (deliberate, bounded): clocked targets give the follower a
    # schedule to race. Jerk-lag at accel transients -> catch-up at full
    # per-joint caps -> overshooting the schedule -> the clocked target lands
    # behind -> brief brake toward ~0 while the clock catches up. On real
    # paths this appears as ~0.5 speed dips per episode (to 0.01-0.3 rad/s,
    # 3-6 frames) at positions uncorrelated with path geometry, plus L2
    # overspeed bursts to ~0.7 (per-joint limits still exact). Two quick
    # fixes were tried and REVERTED after failing validation on the dumped
    # real-path corpus: (a) nearest-sample progress targeting deadlocks the
    # moment the follower reaches its nearest sample; (b) capping the tick
    # velocity at ~the reference's local speed starves catch-up, compounds
    # lag, and ends in settle-timeout teleports. A correct fix is a proper
    # arc-length pure-pursuit target (lookahead by distance, monotone
    # projection) — a design change, not a patch; see session notes.
    for tick in range(1, max_ticks):
        k = min(tick + int(target_lead_ticks), n_ref - 1)
        inp.target_position = ref[k].tolist()
        inp.target_velocity = v_ref[k].tolist()
        inp.target_acceleration = a_ref[k].tolist()
        try:
            result = otg.update(inp, out)
        except RuckigError:
            # Input-specific solver failure (seen at exactly-at-limit boundary
            # states, and on some corner-blended reference targets). Before
            # giving up on the path, degrade THIS tick to a position-only
            # target: dropping the velocity/acceleration targets is ruckig's
            # most robust problem form, and the next tick's targets resume
            # normally — one softened tick is invisible in the output. Only
            # if even that fails is the path abandoned, and as a
            # TrajectoryParametrizationError (candidate-discard semantics),
            # never a raw RuckigError — that killed a whole generation
            # worker once.
            inp.target_velocity = [0.0] * dof
            inp.target_acceleration = [0.0] * dof
            try:
                result = otg.update(inp, out)
            except RuckigError as e:
                raise TrajectoryParametrizationError(
                    f"follower ruckig update threw at tick {tick} "
                    f"(position-only retry also failed): {e}"
                ) from e
        if result < 0:
            raise TrajectoryParametrizationError(
                f"follower ruckig update failed (result {result})"
            )
        samples.append(np.asarray(out.new_position, dtype=np.float64))
        inp.current_position = out.new_position
        inp.current_velocity = np.clip(
            np.asarray(out.new_velocity, dtype=np.float64),
            -vel_feed_cap, vel_feed_cap,
        ).tolist()
        inp.current_acceleration = np.clip(
            np.asarray(out.new_acceleration, dtype=np.float64),
            -acc_feed_cap, acc_feed_cap,
        ).tolist()
        if k == n_ref - 1:
            pos_err = float(np.max(np.abs(samples[-1] - ref[-1])))
            vel_mag = float(np.max(np.abs(out.new_velocity)))
            if result == Result.Finished or (pos_err < 1e-4 and vel_mag < 1e-3):
                break
    samples[-1] = ref[-1].copy()  # pin the exact goal
    return np.asarray(samples)


def _erode_box_same(u: np.ndarray, W: int) -> np.ndarray:
    """Trailing sliding-min then centered box-average (same length). Fallback
    smoother for `_jerk_limited_minorant` when scipy's LP is unavailable:
    bounded jerk and never above the input, but charges a fixed W-tick
    plateau at every feature — measurably slower than the LP result."""
    e = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.full(W - 1, u[0]), u]), W
    ).min(axis=1)
    lp = W // 2
    up = np.concatenate([np.full(lp, e[0]), e, np.full(W - 1 - lp, e[-1])])
    return np.convolve(up, np.full(W, 1.0 / W), mode="valid")


def _jerk_limited_minorant(
    v: np.ndarray,
    seg: np.ndarray,
    acc_bound: float,
    jerk_cap: float,
    dt: float,
    factor: np.ndarray | None = None,
    v_start: float = 0.0,
    v_end: float = 0.0,
    strict_onset: bool | None = None,
) -> np.ndarray:
    """Largest speed profile <= `v` with bounded first/second differences.

    This is the exact object the retimer needs, posed as a small sparse LP
    (HiGHS, ~ms at corpus sizes): maximize arc-weighted speed subject to
      * 0 <= u_i <= v_i           — the reference is a hard ceiling
        (time-optimality: exceeding it at a corner breaks accel as v^2/r);
      * |u_{i+1} - u_i| <= acc_bound*dt;
      * |u_{i+1} - 2u_i + u_{i-1}| <= jerk_cap*dt^2, including boundary rows
        with virtual rest states so the launch onset and terminal stop are
        jerk-limited too.
    Compared to fixed-window morphology this rounds each valley/onset with
    the MINIMAL time cost (no plateau-width vs jerk-width coupling), leaves
    already-feasible spans (cruise, taper) untouched, and needs no tuning.
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n = len(v)
    jd = jerk_cap * dt * dt
    ad = acc_bound * dt
    # Optional per-interval budget factor (e.g. 0.5 near curvature so the
    # tangential transient cannot stack with the normal one on the same
    # joints — the LP otherwise brakes bang-bang-late INTO a corner at full
    # budget exactly where curvature onset spends the rest of it).
    f = np.ones(n) if factor is None else np.asarray(factor, dtype=np.float64)

    def loc(idx_list):
        vals = [f[i] for i in idx_list if 0 <= i < n]
        return min(vals) if vals else 1.0

    def virt(i):
        # Boundary states OUTSIDE the profile: the speed the system arrives
        # with (v_start — a shared-autonomy handoff hands over a MOVING
        # robot) and leaves at (v_end, normally rest). Hard-coding these to
        # zero clamped every handoff plan to a from-rest launch ramp
        # (measured: plan starts near 0 speed regardless of policy speed).
        return v_start if i < 0 else v_end
    rows_a, rows_j = n + 1, n + 2  # D1 incl. boundary states; D2 incl. onset row
    # boundary pairs — the (-2,-1,0) row constrains u0 against the virtual
    # pre-state pair (|u0 - v_start| <= jd), without which the LP may take
    # full accel in the very first tick (onset jerk ~3x the cap, measured
    # 24 at a brake-out junction); symmetrically (n-1,n,n+1) for the end.
    A = lil_matrix((2 * (rows_a + rows_j), n))
    b = np.empty(2 * (rows_a + rows_j))
    r = 0
    # First differences over the rest-extended profile [0, u..., 0].
    ext = [(i, i + 1) for i in range(-1, n)]
    for i, k in ext:
        for sign in (1.0, -1.0):
            rhs = ad * loc((i, k))
            if 0 <= i < n:
                A[r, i] = -sign
            else:
                rhs += sign * virt(i)
            if 0 <= k < n:
                A[r, k] = sign
            else:
                rhs -= sign * virt(k)
            b[r] = rhs
            r += 1
    # Second differences over the same extension.
    # The (-2,-1,0) boundary pair row jerk-limits the ONSET against the
    # virtual pre-state (|u0 - v_start| <= jd). Applied only for moving
    # handoffs and brake-out suffixes (strict_onset): a blanket application
    # taxed every from-rest launch a few ticks and cost 1.2x duration on
    # short paths, while plain launches already onset gently because the LP
    # hugs the toppra ceiling's own soft ramp. The symmetric END pair row is
    # never added — it forces a ~0.01 rad/s terminal crawl; the trajectory
    # ENDS at the goal (env holds there) and the taper bounds terminal decel.
    _strict = strict_onset if strict_onset is not None else (v_start > 0.0)
    for m in range(-2 if _strict else -1, n - 1):
        idx = (m, m + 1, m + 2)
        for sign in (1.0, -1.0):
            rhs = jd * loc(idx)
            for pos, coef in zip(idx, (1.0, -2.0, 1.0)):
                if 0 <= pos < n:
                    A[r, pos] = sign * coef
                else:
                    rhs -= sign * coef * virt(pos)
            b[r] = rhs
            r += 1
    res = linprog(
        c=-np.asarray(seg),  # maximize arc-weighted speed ~ minimize duration
        A_ub=A.tocsr()[:r], b_ub=b[:r],
        bounds=list(zip(np.zeros(n), v)),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"speed-minorant LP failed: {res.message}")
    return np.maximum(res.x, 0.0)


# Curvature-ceiling margins for the retimed backend: the fraction of the
# accel / jerk budget granted to the NORMAL (curvature) component when
# capping speed over the smoothed geometry. 0.8/0.8 is the corpus-calibrated
# optimum (tests/follower_acceptance.py: 37/37); the tangential side's own
# headroom lives in `acc_bound` and the LP's 0.85 jerk allocation.
_RT_VACC = 0.8
_RT_VJRK = 0.8


def _brake_out_prefix(
    q0: np.ndarray,
    q1: np.ndarray,
    v0_tangential: float,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
) -> np.ndarray | None:
    """Jerk-limited 1-D braking run from (q0, moving at v0 toward q1) to
    rest at q1, sampled at control_hz. One LOCAL single-target ruckig solve
    (community build solves single-target offline without the cloud).

    EMERGENCY ESCALATION: when the runway is shorter than the nominal-limit
    braking distance (a shield trigger hands over a robot heading toward an
    obstacle with little free space), the solve retries at escalating
    decel/jerk scales — up to 3x accel / 5x jerk. This DELIBERATELY exceeds
    the nominal caps for the braking segment only: the alternative (a
    rest-start chunk) commands the PD to brake even harder implicitly while
    recording out-of-distribution data that starts from zero velocity. An
    explicitly planned emergency brake keeps the chunk starting at the
    robot's true velocity, which is the property DAgger needs. Escalated
    solves are the exception path and the caller logs the runway shortfall.

    Returns (K, dof) samples from q0 to q1 inclusive, or None if ruckig is
    unavailable / no scale fits — callers fall back to the plain pipeline."""
    try:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig
    except Exception:
        return None
    d = q1 - q0
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return None
    u = d / dist
    # 1-D limits: the tightest per-joint cap projected onto the run direction.
    nz = np.abs(u) > 1e-9
    vlim = float(np.min(np.asarray(max_joint_vel)[nz] / np.abs(u)[nz]))
    alim = float(np.min(np.asarray(max_joint_acc)[nz] / np.abs(u)[nz]))
    jlim = float(np.min(np.asarray(max_joint_jerk)[nz] / np.abs(u)[nz]))
    v0 = float(np.clip(v0_tangential, 0.0, max(vlim, v0_tangential)))
    dt = 1.0 / control_hz
    for a_scale, j_scale in ((1.0, 1.0), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)):
        otg = Ruckig(1, dt)
        inp, out = InputParameter(1), OutputParameter(1)
        inp.current_position, inp.current_velocity, inp.current_acceleration = [0.0], [v0], [0.0]
        inp.target_position, inp.target_velocity, inp.target_acceleration = [dist], [0.0], [0.0]
        inp.max_velocity = [max(vlim, v0 * (1 + 1e-6))]
        inp.max_acceleration, inp.max_jerk = [alim * a_scale], [jlim * j_scale]
        ss = [0.0]
        failed = False
        for _ in range(int(10 * control_hz)):  # 10 s hard cap
            res = otg.update(inp, out)
            if res == Result.Error:
                failed = True
                break
            ss.append(float(out.new_position[0]))
            out.pass_to_input(inp)
            if res == Result.Finished:
                break
        else:
            failed = True
        if failed:
            continue
        ss = np.asarray(ss)
        # Overshoot past the runway means this scale can't stop in time —
        # clipping would corrupt the profile into an accel spike (measured
        # 1.5/34); escalate to the next scale instead.
        if float(ss.max()) > dist + 1e-6 or float(ss.min()) < -1e-6:
            continue
        prefix = q0[None, :] + ss[:, None] * u[None, :]
        prefix[-1] = q1
        return prefix
    return None


def retimed_parametrize_path(
    waypoints: np.ndarray,
    max_joint_vel: np.ndarray,
    max_joint_acc: np.ndarray,
    max_joint_jerk: np.ndarray,
    control_hz: float,
    **toppra_kwargs,
) -> np.ndarray:
    """TOPP-RA schedule + offline jerk-limited retiming of its SPEED PROFILE.

    Why this exists: measured on the acceptance corpus (tests/data/
    follower_corpus), toppra's output is already right in every axis the
    trajectory criteria care about — uniform cruise speed, no dips, exact
    endpoints, vel/acc limits — except one: its acceleration is piecewise
    constant in time, so finite-difference jerk spikes to 17-40 against a
    limit of 10, and those spikes are purely TANGENTIAL (on all 37 corpus
    cases the joint-jerk peak coincides with the 1-D speed-profile jerk
    peak; curvature contributes ~nothing at these speeds). The online-ruckig
    "follower" fixed jerk but introduced a clocked-target limit cycle
    (brief near-stops at waypoints — see its KNOWN DEFECT comment).

    So fix ONLY the defective axis, offline and SURGICALLY: keep toppra's
    samples as exact geometry AND its speed-vs-arc profile untouched except
    inside the few spans where its own jerk violates the cap; there, blend
    toward an eroded+box-filtered profile; then rebuild time by integrating
    arc/speed. Three structural rules, each earned by a measured failure:

      * The new speed is defined AS A FUNCTION OF ARC and is pointwise <=
        the reference's speed at that arc (`_erode_box_same`, plus the
        blend). The reference is a hard ceiling — toppra is time-optimal,
        so exceeding it at an accel-limited corner breaks the accel cap as
        v^2/r — and arc-indexing makes ceiling alignment exact by
        construction. (Tick-indexed variants drifted: smoothing costs arc
        mass, and ONE tick of misalignment at a corner already exceeds the
        accel tolerance.) Slowing down only ever helps accel; and real-time
        smoothness bounds tighten, since d/dt = (w/v <= 1) * d/d(index).
      * Smooth ONLY around measured violations (mask dilated ~3W, blended
        over W ticks so the splice is not itself a jerk step). Whole-profile
        erosion shifts the final-approach taper against its own decay and
        costs a harmonic-sum ~12 ticks of tail crawl; surgical application
        leaves ramp and taper schedules as toppra wrote them.
      * Cover the whole arc by integrating time over every interval — never
        truncate or extend a schedule to "reach" the goal (the truncating
        variant teleported; the extending variant crawled and embedded a
        full stop).

    Duration cost measured on the corpus: ~0-2%. All `toppra_kwargs` pass
    through to `toppra_parametrize_path`. `start_vel` handoffs
    (interventions): the toppra reference honors the handoff tangentially,
    and its projection onto the first path segment feeds the LP's virtual
    pre-start state, so the plan begins AT the robot's current speed
    instead of ramping from rest.
    """
    # Leading BRAKE-OUT split (shared-autonomy moving handoffs). When the
    # planner prepended a braking lead-in along the start velocity, the
    # path begins q0 -> p_lead -> reversal -> rest-of-path. A blended
    # reversal is the WRONG representation for toppra: the spline
    # degenerates at the cusp (measured: controllable start-speed window
    # collapsing to 0.016 rad/s, jerk 24) and the handoff speed is lost.
    # The physically exact structure is a SPLIT: a 1-D jerk-limited braking
    # run along the first segment (single local ruckig solve), then the
    # remaining path parametrized from rest at the cusp — the two meet at
    # zero velocity, so the concatenation is smooth by construction.
    _strict_onset = bool(toppra_kwargs.pop("_strict_onset", False))
    _sv = toppra_kwargs.get("start_vel")
    wp_arr = np.asarray(waypoints, dtype=np.float64)
    if _sv is not None and wp_arr.shape[0] >= 3:
        sv = np.asarray(_sv, dtype=np.float64).reshape(-1)[: wp_arr.shape[1]]
        d01 = wp_arr[1] - wp_arr[0]
        d12 = wp_arr[2] - wp_arr[1]
        n01, n12, nsv = (np.linalg.norm(d01), np.linalg.norm(d12), np.linalg.norm(sv))
        if n01 > 1e-9 and n12 > 1e-9 and nsv > 1e-6:
            aligned = float(np.dot(sv, d01)) / (nsv * n01) > 0.7
            cusp = float(np.dot(d01, d12)) / (n01 * n12) < -0.5  # turn > 120 deg
            if aligned and cusp:
                prefix = _brake_out_prefix(
                    wp_arr[0], wp_arr[1], float(np.dot(sv, d01) / n01),
                    max_joint_vel, max_joint_acc, max_joint_jerk, control_hz,
                )
                if prefix is not None:
                    kw2 = dict(toppra_kwargs)
                    kw2.pop("start_vel", None)
                    # The suffix meets the easing brake at REST — its onset
                    # must be jerk-limited too or the junction accel steps
                    # (measured 24 of jerk).
                    kw2["_strict_onset"] = True
                    suffix = retimed_parametrize_path(
                        wp_arr[1:], max_joint_vel, max_joint_acc,
                        max_joint_jerk, control_hz, **kw2,
                    )
                    return np.vstack([prefix[:-1], np.asarray(suffix)])

    ref = toppra_parametrize_path(
        waypoints, max_joint_vel, max_joint_acc, max_joint_jerk, control_hz,
        **toppra_kwargs,
    )
    ref = np.asarray(ref, dtype=np.float64)
    if ref.shape[0] < 3:
        return ref
    dt = 1.0 / control_hz

    seg_all = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    # Drop zero-length intervals (settle/duplicate samples) — they carry no
    # arc and would divide by zero in the time rebuild.
    live = seg_all > 1e-12
    if not live.any():
        return ref
    q_knots = np.concatenate([ref[:1], ref[1:][live]])
    seg_orig = seg_all[live]

    # Sample-scale geometric smoothing (endpoints fixed). Real planner output
    # carries direction kinks of 2-8 deg PER KNOT (trajopt/elastic/chamfer
    # residue); FD jerk crossing such a vertex is v*theta*hz^2 — LINEAR in
    # speed, so no speed profile fixes it acceptably (meeting jerk 10 by
    # slowing means near-stops at every kink, the exact behavior this
    # backend exists to remove), while rounding the vertex geometrically is
    # nearly free (~60 urad deviation per Laplacian pass at a 2.5 deg kink).
    # Ten passes cost ~2-3 mrad total deviation — far under the 0.02 rad
    # obstacle clearance — and are SAFE only because the speed license below
    # is derived from this smoothed geometry itself: with the license taken
    # from the ORIGINAL spacing alone, deeper smoothing misaligned license
    # from curvature and measured fresh accel violations (that instability
    # also killed the measure-and-iterate variants: ceiling scars create
    # jerk at their own edges, becoming next round's violations).
    for _ in range(12):
        q_knots[1:-1] += 0.25 * (q_knots[:-2] - 2.0 * q_knots[1:-1] + q_knots[2:])
    seg = np.linalg.norm(np.diff(q_knots, axis=0), axis=1)
    keep2 = seg > 1e-12
    seg, seg_orig = seg[keep2], seg_orig[keep2]
    q_knots = np.concatenate([q_knots[:1], q_knots[1:][keep2]])
    s_knots = np.concatenate([[0.0], np.cumsum(seg)])
    total = s_knots[-1]
    if total < 1e-9:
        return ref
    v = seg * control_hz  # schedule speed per (live) tick, smoothed geometry
    # License alignment BY ARC, not index: smoothing shifts knot arcs, and an
    # index-aligned license puts the reference's braking profile a knot or
    # two away from where the (moved) corner actually is — at corner
    # tolerances that misalignment is what capped global smoothing at ~12
    # passes. Sample the original speed at the smoothed interval's
    # fractional arc position instead.
    s_orig = np.concatenate([[0.0], np.cumsum(seg_orig)])
    mids_orig = 0.5 * (s_orig[:-1] + s_orig[1:]) / max(s_orig[-1], 1e-12)
    mids_sm = 0.5 * (s_knots[:-1] + s_knots[1:]) / max(total, 1e-12)
    v_lic_arc = np.interp(mids_sm, mids_orig, seg_orig * control_hz)

    # Speed license: the original spacing (the schedule toppra proved
    # feasible) further capped by two ANALYTIC ceilings measured on the
    # SMOOTHED geometry, in the same finite-difference sense the controller
    # (and the acceptance test) will see:
    #   * accel: per-joint FD accel crossing knot k is ~ v^2 * |ddir_j| / h,
    #     so v <= sqrt(0.95 * a_cap * h / max_j |ddir_j|);
    #   * jerk: crossing a residual vertex concentrates a per-joint accel
    #     change of ~ v * |d2dir_j| * hz in one tick, so
    #     v <= 0.9 * J / (hz^2 * max_j |d2dir_j|).
    acc_cap = float(np.max(np.asarray(max_joint_acc, dtype=np.float64)))
    jerk_cap = float(np.min(np.asarray(max_joint_jerk, dtype=np.float64)))
    v_ceil = v_lic_arc
    dirs_k = np.diff(q_knots, axis=0) / np.maximum(seg[:, None], 1e-12)
    if len(dirs_k) >= 7:
        # Multi-scale: FD at output ticks spans ~1-3 knots (tick arc ~ v*dt
        # vs knot spacing ~ v_ref*dt), and consecutive same-direction kinks
        # ADD at tick scale — a per-knot (single-scale) cap measured 30-50%
        # low. Continuum forms per scale m: accel = v^2 * |ddir_m|/(m h),
        # jerk = v^3 * |d2dir_m|/(m h)^2; ceiling = min over scales.
        cap_i = np.full(len(seg), np.inf)
        for m in (1, 2, 3):
            dd = np.abs(dirs_k[m:] - dirs_k[:-m]).max(axis=1)
            d2 = np.abs(
                dirs_k[2 * m:] - 2.0 * dirs_k[m:-m] + dirs_k[:-2 * m]
            ).max(axis=1)
            h_m = m * 0.5 * (seg[m:] + seg[:-m])
            v_acc = np.sqrt(_RT_VACC * acc_cap * h_m / np.maximum(dd, 1e-9))
            h2_m = m * 0.5 * (seg[2 * m:] + seg[:-2 * m])
            v_jrk = np.cbrt(
                _RT_VJRK * jerk_cap * h2_m * h2_m / np.maximum(d2, 1e-9)
            )
            # scatter knot-scale caps onto the intervals they straddle
            for k in range(len(dd)):
                sl = slice(k, min(k + m + 1, len(cap_i)))
                cap_i[sl] = np.minimum(cap_i[sl], v_acc[k])
            for k in range(len(d2)):
                sl = slice(k, min(k + 2 * m + 1, len(cap_i)))
                cap_i[sl] = np.minimum(cap_i[sl], v_jrk[k])
        # Decouple tangential and normal transients: a centered sliding-min
        # widens every CURVATURE dip by K ticks each side, so the LP
        # finishes braking BEFORE curvature turns on and resumes after it
        # ends. Without this the LP brakes bang-bang INTO the corner and
        # the tangential brake-onset jerk lands on the same ticks as the
        # curvature-onset jerk, adding per-joint (measured |a| 1.15-1.33,
        # |j| 12-22 at corner entries with each component individually in
        # budget). Applied to the corner caps ONLY — eroding the full
        # license drags the taper's near-zero tail backward and the
        # integrator crawls it (measured: duration blown on all 37).
        # ...implemented as backward+forward accel propagation at HALF the
        # accel budget: cap[k] = min(cap[k], sqrt(cap[k+1]^2 + 2*(a/2)*h)).
        # Depth-adaptive (a fixed-width erosion left braking ending exactly
        # AT the corner for deep caps), and the halved tangential accel near
        # corners leaves per-joint headroom for the normal component.
        cap_raw = cap_i.copy()
        finite = np.isfinite(cap_i)
        if finite.any():
            brake = acc_cap * 0.5
            capped = np.minimum(cap_i, np.max(v_ceil) * 2.0)
            for k in range(len(capped) - 2, -1, -1):
                capped[k] = min(capped[k], np.sqrt(capped[k + 1] ** 2 + 2.0 * brake * seg[k]))
            for k in range(1, len(capped)):
                capped[k] = min(capped[k], np.sqrt(capped[k - 1] ** 2 + 2.0 * brake * seg[k]))
            cap_i = capped
        v_ceil = np.minimum(v_ceil, np.maximum(cap_i, 0.02))

    # Tangential accel must respect the true cap even where the reference's
    # own slope exceeds it (curvature can help per-joint accel; a scalar
    # profile gets no such help) — the LP simply brakes earlier, which is
    # always feasible for an upper-bounded profile.
    # 0.88: the LP otherwise accelerates at exactly the cap, and two
    # measured amplifiers stack on it per-joint — a few percent of ambient
    # curvature on nominally straight spans (1.057-1.066), and ~10%
    # execution compression on launch ramps, where the arc-aligned license
    # sits slightly above the index schedule on the convex sqrt(2as) ramp.
    acc_bound = 0.88 * acc_cap
    if len(dirs_k) < 7:
        cap_raw = np.full(len(seg), np.inf)
    turn_knots = np.zeros(max(len(seg) - 1, 0))
    if len(dirs_k) >= 2:
        turn_knots = np.arccos(
            np.clip(np.sum(dirs_k[:-1] * dirs_k[1:], axis=1), -1.0, 1.0)
        )
    use_spline = len(s_knots) >= 4
    if use_spline:
        from scipy.interpolate import CubicSpline

        pos_of_arc = CubicSpline(s_knots, q_knots, axis=0, bc_type="natural")

    # The LP variables ARE the knot speeds — the executed field is their
    # linear interpolation, nothing else. (An earlier variant solved the LP
    # per-interval and derived node speeds afterward with a turn-aware
    # mean/min rule; the switching created slope discontinuities the LP
    # never saw, measured as accel -1.2 and jerk 22 at brake onsets whose
    # LP profile was bounded at -1.0/8.5.) The turn-awareness lives in the
    # knot CEILING instead — a constraint the LP smooths across: MIN of the
    # adjacent interval licenses where the path turns (a mean exceeds the
    # corner interval's ceiling and shows up as v^2/r accel), MEAN on
    # straight-ish knots (the min costs a half-interval lag on every ramp).
    # Executed FD slopes can only be GENTLER than the LP's index-domain
    # design: crossing a knot interval takes h/u >= dt of real time.
    # Curved-ramp coupling: normal jerk carries a 2*v*vdot*kappa term —
    # tangential accel RAMPS the centripetal accel — which neither the
    # curvature ceilings (kappa, dkappa terms at fixed speed) nor the LP
    # (tangential only) see. Measured 14.8 of jerk on a launch that curves
    # at 10 deg/tick. Bound the local tangential accel by a <= J/(2 v k)
    # through the LP's per-row budget factor.
    lp_factor = np.ones(len(s_knots))
    if len(dirs_k) >= 2:
        kappa_i = np.zeros(len(seg))
        dd1 = np.linalg.norm(np.diff(dirs_k, axis=0), axis=1)
        h1 = 0.5 * (seg[:-1] + seg[1:])
        kappa_i[:-1] = np.maximum(kappa_i[:-1], dd1 / np.maximum(h1, 1e-9))
        kappa_i[1:] = np.maximum(kappa_i[1:], dd1 / np.maximum(h1, 1e-9))
        a_curv = 0.55 * jerk_cap / (
            2.0 * np.maximum(v_ceil * kappa_i, 1e-9)
        )
        f_i = np.clip(a_curv / max(acc_bound, 1e-9), 0.25, 1.0)
        lp_factor[:-1] = np.minimum(lp_factor[:-1], f_i)
        lp_factor[1:] = np.minimum(lp_factor[1:], f_i)

    c_kn = np.empty(len(s_knots))
    c_kn[0], c_kn[-1] = v_ceil[0], v_ceil[-1]
    if len(v_ceil) > 1:
        pair_min = np.minimum(v_ceil[:-1], v_ceil[1:])
        pair_mean = 0.5 * (v_ceil[:-1] + v_ceil[1:])
        c_kn[1:-1] = np.where(turn_knots > 0.005, pair_min, pair_mean)
    wgt = np.empty(len(s_knots))
    wgt[0], wgt[-1] = 0.5 * seg[0], 0.5 * seg[-1]
    wgt[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    # Tangential handoff speed: shared autonomy hands over a MOVING robot
    # (start_vel = the policy's recent joint velocity, honored tangentially
    # by the toppra reference). The LP's virtual pre-start state must carry
    # it too, or its rest boundary row clamps the first knot to ~acc*dt and
    # every intervention plan launches from zero.
    sv = toppra_kwargs.get("start_vel")
    v_handoff = 0.0
    if sv is not None and len(q_knots) >= 2:
        d0 = q_knots[1] - q_knots[0]
        n0 = float(np.linalg.norm(d0))
        if n0 > 1e-12:
            v_handoff = max(0.0, float(np.dot(
                np.asarray(sv, dtype=np.float64).reshape(-1)[: q_knots.shape[1]],
                d0 / n0,
            )))
    # EXACT-CARRY CONTRACT (shared autonomy): the EXECUTED launch speed is
    # v_handoff — never clamped, no exceptions. The LEGAL profile below is
    # still solved from a clamped boundary state v_legal: on tight geometry
    # the ceiling's first knot can be tiny (toppra's controllable window
    # near a shield-triggered plan measured 0.013 rad/s) and an unclamped
    # boundary makes the LP INFEASIBLE outright (u0 <= c_kn[0] vs
    # u0 >= v_start - acc*dt cannot both hold). The surplus
    # v_handoff - v_legal is carried by a BRAKE-IN OVERLAY in the time
    # rebuild: the first ticks ride a decelerating overspeed profile
    # (nominal decel, escalating only when the path is too short to shed
    # the surplus) until it meets the legal profile, then follow it. The
    # overlay deliberately exceeds the nominal ceilings while it lasts —
    # the policy put the robot in that state, and a planned brake beats
    # pretending the speed away (the clamped launch recorded a 4x speed
    # discontinuity at the policy->RRT seam; eval planar_3joint scenario 2,
    # 2026-08-17).
    v_legal = min(v_handoff, float(c_kn[0]))
    if sv is not None and v_handoff > 0.0:
        logger.info(
            "retimed handoff: |v_start|=%.3f rad/s, tangential=%.3f, start "
            "license c_kn[0]=%.3f -> launch at %.3f",
            float(np.linalg.norm(np.asarray(sv, dtype=np.float64))),
            v_handoff, float(c_kn[0]), v_handoff,
        )

    def _fallback_smooth():
        at_f = np.diff(v) * control_hz
        jt_pk = float(np.abs(np.diff(at_f)).max() * control_hz) if len(at_f) > 1 else 0.0
        W = max(2, int(np.ceil(1.5 * jt_pk / max(jerk_cap, 1e-9))))
        return _erode_box_same(c_kn, W) if len(c_kn) > 2 * W else c_kn

    try:
        node_v = _jerk_limited_minorant(
            c_kn, wgt, acc_bound, 0.85 * jerk_cap, dt, factor=lp_factor,
            v_start=v_legal, strict_onset=(v_legal > 0.0 or _strict_onset),
        )
    except Exception:
        if v_legal > 0.0:
            # Infeasibility can also come from the handoff interacting with
            # tight interior constraints — retry as a from-rest LEGAL
            # profile before falling back to morphology. The executed
            # launch still starts at v_handoff either way: the brake-in
            # overlay below rides above whatever profile this produces.
            try:
                node_v = _jerk_limited_minorant(
                    c_kn, wgt, acc_bound, 0.85 * jerk_cap, dt, factor=lp_factor,
                )
            except Exception:
                node_v = _fallback_smooth()
        else:
            node_v = _fallback_smooth()

    floor = 1e-3 * float(np.max(v))
    # Brake-in overlay (exact-carry contract above): while w_over exceeds
    # the legal profile the tick speed IS w_over, decaying at a_brake; the
    # overlay ends the tick it dips under the legal profile (it always
    # does: w_over decays toward 0, the legal profile is floored). Rung
    # selection: smallest decel whose brake-to-rest arc fits in 90% of the
    # path, so even a profile that never rises above the floor stops the
    # surplus before the goal; the forced last resort handles a handoff
    # hotter than any rung can shed on this runway.
    w_over = v_handoff if v_handoff > float(node_v[0]) + 1e-9 else 0.0
    a_brake = 0.0
    if w_over > 0.0:
        for _scale in (1.0, 1.5, 2.0, 3.0, 5.0):
            a_brake = _scale * acc_cap
            if w_over * w_over / (2.0 * a_brake) <= 0.9 * total:
                break
        else:
            a_brake = w_over * w_over / (1.8 * total)
        logger.info(
            "retimed handoff: carrying %.3f rad/s launch over a %.3f rad/s "
            "start license — brake-in overlay at %.2f rad/s^2 decel (%.1fx "
            "nominal)", v_handoff, float(node_v[0]), a_brake,
            a_brake / max(acc_cap, 1e-9),
        )
    s_list = [0.0]
    arc = 0.0
    max_ticks = int(
        np.ceil(float(np.sum(seg / np.maximum(0.5 * (node_v[:-1] + node_v[1:]), 1e-9))) * control_hz)
    ) + int(3.0 * control_hz)
    while arc < total - 1e-4 and len(s_list) <= max_ticks:
        v1 = max(float(np.interp(arc, s_knots, node_v)), floor)
        vm = max(float(np.interp(arc + 0.5 * v1 * dt, s_knots, node_v)), floor)
        if w_over > 0.0:
            if w_over <= vm:
                w_over = 0.0  # met the legal profile — overlay done
            else:
                vm = w_over
                w_over = max(w_over - a_brake * dt, 0.0)
        arc = min(arc + vm * dt, total)
        s_list.append(arc)
    # Land the last sample ON the goal rather than crawling the final
    # fraction of a mrad at the speed floor (measured ~10 settle ticks).
    s_list[-1] = total
    s_ticks = np.asarray(s_list)
    if use_spline:
        out = pos_of_arc(s_ticks)
    else:
        out = np.empty((len(s_ticks), ref.shape[1]), dtype=np.float64)
        for jnt in range(ref.shape[1]):
            out[:, jnt] = np.interp(s_ticks, s_knots, q_knots[:, jnt])
    out[0], out[-1] = ref[0], ref[-1]
    return out


# Backend selection for `parametrize_path`. Gate for the default: the
# acceptance corpus (tests/follower_acceptance.py, 37 real planner paths) —
# uniform cruise with no dips below the physics floors, no overshoot,
# per-joint vel/acc/jerk within caps, duration within 1.03x + 0.4 s of the
# time-optimal toppra reference.
#
#   backend     acceptance   character
#   retimed     37/37        offline LP retiming of the toppra schedule;
#                            deterministic, local, no ruckig     <- default
#   follower     2/37        online-ruckig tracker; clocked-target limit
#                            cycle dips to 0.01-0.3 rad/s ~0.5x/episode
#                            (see its KNOWN DEFECT comment)
#   toppra       0/37        time-optimal but IGNORES max_joint_jerk
#                            (fails only jerk); accel steps ring under PD
#   chained      (unused)    per-gap offline ruckig; jerk-limited but pays
#                            2-8x duration
#   ruckig       (unused)    community cloud API; networked, 1000/day
TRAJ_BACKEND_ENV = "SPLATSIM_TRAJ_BACKEND"
# Single source of truth for the fallback when the env var is unset — the
# planner's startup log reads this too, so log and dispatch can't disagree.
DEFAULT_TRAJ_BACKEND = "retimed"


def _dump_parametrization(waypoints, traj, kwargs) -> None:
    """Record one (waypoints -> trajectory) pair when SPLATSIM_TRAJ_DUMP is set.

    Diagnostic only, and off unless the env var names a directory. Lets a
    limits sweep run offline against the REAL geometry a closed-loop run
    produced, instead of synthetic waypoints that may not share its
    pathologies.
    """
    out_dir = os.environ.get("SPLATSIM_TRAJ_DUMP")
    if not out_dir:
        return
    try:
        import pathlib, pickle, threading, time as _t

        d = pathlib.Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        # pid+thread+counter: several planners can share a directory.
        global _DUMP_SEQ
        _DUMP_SEQ += 1
        name = f"wp_{os.getpid()}_{threading.get_ident()}_{_DUMP_SEQ:05d}.pkl"
        with open(d / name, "wb") as fh:
            pickle.dump({"waypoints": np.asarray(waypoints),
                         "traj": np.asarray(traj),
                         "kwargs": {k: v for k, v in kwargs.items()}}, fh)
    except Exception:
        pass  # a diagnostic must never break planning


_DUMP_SEQ = 0


def parametrize_path(*args, backend: str | None = None, **kwargs) -> np.ndarray:
    """Time-parametrize a geometric joint path. Dispatches on `backend`.

    `backend`: "follower" (default), "toppra", "chained", or "ruckig". None
    reads the `SPLATSIM_TRAJ_BACKEND` environment variable, defaulting to
    `DEFAULT_TRAJ_BACKEND`. All backends take the identical signature — see
    `toppra_parametrize_path` for the (small) semantic differences.
    """
    if backend is None:
        backend = os.environ.get(TRAJ_BACKEND_ENV, DEFAULT_TRAJ_BACKEND).strip().lower()
    try:
        if backend == "ruckig":
            traj = ruckig_parametrize_path(*args, **kwargs)
        elif backend == "toppra":
            traj = toppra_parametrize_path(*args, **kwargs)
        elif backend in ("chained", "chained_ruckig"):
            traj = chained_ruckig_parametrize_path(*args, **kwargs)
        elif backend == "follower":
            traj = follower_parametrize_path(*args, **kwargs)
        elif backend == "retimed":
            traj = retimed_parametrize_path(*args, **kwargs)
        else:
            traj = None
    except (RuckigCloudUnavailableError, TrajectoryParametrizationError):
        raise  # already classified — systemic vs. this-path-only
    except Exception as e:
        # A raw RuckigError escaping here once crashed a whole generation
        # worker. Classify it as path-specific (the cloud/rate-limit case was
        # converted to RuckigCloudUnavailableError at the call site) so
        # planners treat it like any other rejected candidate. Matched by name
        # to keep the ruckig import lazy.
        if type(e).__name__ == "RuckigError":
            raise TrajectoryParametrizationError(
                f"{backend} backend ruckig failure: {e}"
            ) from e
        raise
    if traj is not None:
        if args:
            _dump_parametrization(args[0], traj, kwargs)
        return traj
    raise ValueError(
        f"Unknown trajectory backend {backend!r} (set {TRAJ_BACKEND_ENV} to "
        "'chained', 'toppra' or 'ruckig')"
    )



def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resamples a path to have a specific number of points using linear interpolation.

    Args:
        path: The original path (N, DOF) array.
        n_points: The desired number of points (e.g., 120).

    Returns:
        The new, resampled path (n_points, DOF) array.
    """
    if not isinstance(path, np.ndarray):
        path = np.array(path)
        
    n_original_points, dof = path.shape
    
    # 1. Create the "x" axis for the original and new paths
    # Original: [0, 1, 2, ..., N-1]
    original_x = np.linspace(0, 1, num=n_original_points)
    
    # New: [0, 0.008, 0.016, ..., 1]
    new_x = np.linspace(0, 1, num=n_points)
    
    # 2. Create an empty array for the new path
    resampled_path = np.zeros((n_points, dof))
    
    # 3. Interpolate each joint (column)
    for i in range(dof):
        joint_original = path[:, i]
        joint_new = np.interp(new_x, original_x, joint_original)
        resampled_path[:, i] = joint_new
        
    return resampled_path

def open_gripper(robot_id, physics_client_id=None):
    cid = _resolve_client_id(physics_client_id)
    # A very hardcoded and temporary solution
    for idx in range(7, p.getNumJoints(robot_id, physicsClientId=cid)):
        p.resetJointState(robot_id, idx, 0.0, physicsClientId=cid)
    p.stepSimulation(physicsClientId=cid)



def get_path(q_start, q_goal, robot_id, joint_indices, obstacle_ids, ll, ul, robot_update_rate, use_gui=False, verbose=True, max_joint_vel=None, max_joint_acc=None, max_joint_jerk=None, obstacle_names=None, skip_pairs=None, physics_client_id=None, obstacle_clearance=None, self_collision_clearance=None, self_collision_skip_pairs=None, max_smooth_iterations=50, actual_gripper_q=None, elastic_smooth_passes=0, trajopt_passes=15, trajopt_lr=0.02, trajopt_smoothness_weight=1.0, trajopt_collision_weight=5.0, trajopt_collision_threshold=0.10, trajopt_fd_step=0.01, config_cost_fn=None, trrt_params=None):
    """Plan + smooth a joint path. With `config_cost_fn` set (soft-cost
    "guided" mode) every stage becomes cost-aware: T-RRT tree growth
    (`cost_aware_birrt`), cost-gated shortcut smoothing
    (`cost_aware_smooth_path`), and cost-gated elastic corner rounding.
    None = historical binary pipeline, bit-for-bit."""
    cid = _resolve_client_id(physics_client_id)
    dof = len(joint_indices)
    if max_joint_vel is None:
        max_joint_vel = np.full(dof, 0.5)   # rad/s
    if max_joint_acc is None:
        max_joint_acc = np.full(dof, 1.0)   # rad/s^2
    if max_joint_jerk is None:
        max_joint_jerk = np.full(dof, 10.0)  # rad/s^3, ~10x max_acc
    # Set joints to q_start (this ALSO forces open_gripper via
    # set_robot_joint_positions — wipes the caller's actual-gripper snap).
    set_robot_joint_positions(robot_id, joint_indices, q_start, physics_client_id=cid)
    # Re-snap the gripper joints (URDF indices dof+1..num_pb_joints) to the
    # actual env gripper if the caller told us the value. Without this every
    # BiRRT sample/extend collision check runs on wide-open finger geometry
    # while the real robot's fingers are typically closing around an object;
    # RRT then falsely reports q_start/q_goal in collision and the intervention
    # cascades to 5-retry backoff. Matches the fix in `RRTToGoalPlanner.plan`
    # and mirrors what `check_chunk_collision` does at line ~3341.
    if actual_gripper_q is not None:
        _n_pb_joints = p.getNumJoints(robot_id, physicsClientId=cid)
        _gv = float(actual_gripper_q)
        for _idx in range(dof + 1, _n_pb_joints):
            p.resetJointState(robot_id, _idx, _gv, physicsClientId=cid)

    # movable_joints = get_movable_joints(robot_id)

    # 0.05 radians per joint, used both for RRT extension and path smoothing.
    resolutions = [0.05] * len(joint_indices)

    # Forward the configured clearance to both the RRT sample/extend
    # collision_fn AND the smooth_path post-processing collision_fn so the
    # whole pipeline uses one consistent margin. See `get_rrt_plan` for the
    # symmetric forwarding inside the BiRRT collision check.
    _ccheck_kwargs = {}
    if obstacle_clearance is not None:
        _ccheck_kwargs["obstacle_clearance"] = obstacle_clearance
    if self_collision_clearance is not None:
        _ccheck_kwargs["self_collision_clearance"] = self_collision_clearance
    if self_collision_skip_pairs:
        _ccheck_kwargs["self_collision_skip_pairs"] = self_collision_skip_pairs

    # RRT-Connect planner — pass joint limits + resolutions so its sample/extend
    # functions don't query PyBullet (which would target the default client).
    rrt_path = get_rrt_plan(
        robot_id, joint_indices, obstacle_ids, q_start, q_goal,
        lower_limits=ll, upper_limits=ul, resolutions=resolutions,
        verbose=verbose, obstacle_names=obstacle_names, skip_pairs=skip_pairs,
        physics_client_id=cid,
        obstacle_clearance=obstacle_clearance,
        self_collision_clearance=self_collision_clearance,
        self_collision_skip_pairs=self_collision_skip_pairs,
        actual_gripper_q=actual_gripper_q,
        config_cost_fn=config_cost_fn,
        trrt_params=trrt_params,
    )
    if rrt_path is None:
        return None


    def collision_fn(q):
        return check_links_in_collision(robot_id, joint_indices, q, obstacle_ids,
                                         skip_pairs=skip_pairs, physics_client_id=cid,
                                         **_ccheck_kwargs)

    extend_fn = _make_linear_extend_fn(resolutions)

    # Random-shortcut smoothing (pybullet_planning): repeatedly pick two random
    # points on the path and replace the intermediate segment with a straight
    # joint-space connection when it is collision-free and shorter. This is
    # what removes RRT's characteristic detours/zigzags — time
    # parametrization downstream only shapes the SPEED profile (vel/acc/jerk),
    # it does not straighten the geometric path, so erratic-looking waypoints
    # must be fixed here. Leftover zigzags survive as direction changes, each
    # of which forces a decelerate/re-accelerate that surfaces as jerk. Iterations beyond convergence are cheap (a candidate is
    # collision-checked only when it would shorten the path).
    if config_cost_fn is None:
        smoothed_path = smooth_path(
            rrt_path.tolist(),
            extend_fn,
            collision_fn,
            max_smooth_iterations=max_smooth_iterations,
        )
    else:
        # Cost-gated shortcutting: a shortcut is accepted only if it does not
        # raise the path's soft-cost integral — plain smooth_path would
        # straighten the cost-avoiding detour right back through the canopy.
        smoothed_path = cost_aware_smooth_path(
            rrt_path.tolist(),
            extend_fn,
            collision_fn,
            config_cost_fn,
            max_smooth_iterations=max_smooth_iterations,
        )

    # CHOMP-lite trajectory optimization (opt-in): adds an EXPLICIT REPULSIVE
    # collision cost on top of Laplacian smoothness, so waypoints get pushed
    # AWAY from obstacles (not just refused entry into collision). With
    # trajopt on, paths take routes with genuinely wider clearance rather
    # than skimming obstacle boundaries, which makes the trained policy less
    # sensitive to small obstacle position changes (fewer discontinuous
    # homotopy-class flips between nearly-identical scenarios). See
    # `trajopt_smooth_path` docstring for the cost formulation. Distance-fn
    # closes over robot_id+obstacle_ids so the caller doesn't need to know
    # about PyBullet internals.
    # RUNS BEFORE elastic_smooth_path (below): the collision-repulsion term
    # can introduce SHARP corners at the bow apex where the outward push
    # fights the smoothness pull — ruckig then decelerates hard at those
    # corners, producing the "jerky start/stop around obstacles" pathology.
    # Running elastic AFTER trajopt lets Laplacian smoothing round those
    # apex corners (with the hard-collision-reject gate preserving trajopt's
    # clearance headroom in the process). Order was elastic→trajopt in the
    # very first draft; swapped to trajopt→elastic after observing exactly
    # this jerkiness.
    if trajopt_passes and smoothed_path is not None and len(smoothed_path) >= 3:
        _run_trajopt = True
    else:
        _run_trajopt = False
        if trajopt_passes:
            # Trace WHY trajopt got skipped, matching the terseness of the
            # [trajopt] trace inside trajopt_smooth_path — otherwise a user
            # who enabled trajopt but only sees the trace on cluttered scenes
            # can't tell whether the flag is off or the shortcut smoother
            # reduced the path to a straight line (no interior waypoints to
            # optimize). "path=None" means RRT itself returned nothing.
            _n = 0 if smoothed_path is None else len(smoothed_path)
            print(f"[trajopt] skipped: shortcut-smoothed path has {_n} waypoints (< 3 interior); nothing to optimize")

    if _run_trajopt:
        def distance_fn(q):
            return min_distance_to_obstacles(
                robot_id, joint_indices, q, obstacle_ids,
                # Query capped near the threshold — beyond it the collision
                # cost is zero anyway, and getClosestPoints cost explodes with
                # the query margin on large concave meshes (vine scene: ~18 ms
                # at 1.0 m vs ~0.4 ms at 0.3 m per link), which stalled env
                # init for hours inside trajopt.
                max_dist=max(float(trajopt_collision_threshold) * 2.0, 0.2),
            )
        smoothed_path = trajopt_smooth_path(
            smoothed_path,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            passes=int(trajopt_passes),
            lr=float(trajopt_lr),
            smoothness_weight=float(trajopt_smoothness_weight),
            collision_weight=float(trajopt_collision_weight),
            collision_threshold=float(trajopt_collision_threshold),
            fd_step=float(trajopt_fd_step),
            config_cost_fn=config_cost_fn,
        ).tolist()

    # Corner-rounding relaxation (opt-in): runs AFTER trajopt so it can round
    # any sharp corners trajopt introduced at the bow apex. On its own (when
    # trajopt is off), it also serves its original purpose — rounding jagged
    # RRT joint-space corners in tight scenes where shortcutting can't
    # collapse them. Both roles use the same Laplacian pull-toward-neighbor-
    # midpoint with hard-collision-reject gate. See elastic_smooth_path
    # docstring.
    if elastic_smooth_passes and smoothed_path is not None and len(smoothed_path) >= 3:
        smoothed_path = elastic_smooth_path(
            smoothed_path, collision_fn, passes=int(elastic_smooth_passes),
            config_cost_fn=config_cost_fn,
        ).tolist()

    # Visualize in GUI if requested
    if use_gui:
        playback_path_in_gui(resample_path_by_distance(smoothed_path, n_points=140), robot_id, joint_indices, path_name="Joint Dist Sampled", fps=robot_update_rate, playback_speed=1.0)

    return smoothed_path

def playback_path_in_gui(path, robot_id, joint_indices, path_name, fps=240, playback_speed=1.0):
    if not p.isConnected():
        print("Not connected to PyBullet GUI.")
        return
    set_robot_joint_positions(robot_id, joint_indices, path[0])
    input(f"Press Enter to play back the {path_name} path...")
    for q in path:
        set_robot_joint_positions(robot_id, joint_indices, q)
        p.stepSimulation()
        time.sleep(1.0 / fps / playback_speed)

def show_joint_config_in_gui(robot_id, joint_indices, q):
    if not p.isConnected():
        print("Not connected to PyBullet GUI.")
        return
    # Use resetJointState only — no stepSimulation, no motor control.
    # stepSimulation would let physics push the robot away from the desired pose
    # (especially with obstacles nearby), so the GUI would not reflect the true config.
    for idx, qi in zip(joint_indices, q):
        p.resetJointState(robot_id, idx, qi)

def compute_camera_alignment_score(
    cam_position: np.ndarray,
    cam_forward: np.ndarray,
    target_position: np.ndarray,
    k_exp: float = 5.0,
    k_sig: float = 15.0,
    threshold: float = 0.4,
) -> float:
    """
    Compute camera alignment score for a single timestep.

    Higher score = camera better aligned with target.
    Combines exponential reward with sigmoid gating.

    Args:
        cam_position: Camera position in world frame (3,)
        cam_forward: Camera forward direction unit vector (3,)
        target_position: Target position in world frame (3,)
        k_exp: Exponential sharpness (default: 5.0)
        k_sig: Sigmoid sharpness (default: 15.0)
        threshold: Alignment threshold (default: 0.4)

    Returns:
        Score for this single timestep
    """
    # Direction from camera to target
    target_direction = target_position - cam_position
    target_distance = np.linalg.norm(target_direction)

    if target_distance < 1e-6:
        alignment = 0.0  # Camera at target
    else:
        target_direction_normalized = target_direction / target_distance
        alignment = np.dot(cam_forward, target_direction_normalized)

    # Scoring function components
    exp_reward = np.exp(k_exp * alignment)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-k_sig * (alignment - threshold)))

    return float(exp_reward * sigmoid_gate)