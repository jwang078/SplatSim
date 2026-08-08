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
) -> np.ndarray:
    """Time-optimal path parametrization using TOPP-RA. Local, no network.

    Drop-in replacement for `ruckig_parametrize_path` — same signature, same
    (M, DOF) output sampled at `control_hz`. See the module comment above
    "TOPP-RA time parametrization" for the behavioral differences
    (`max_joint_jerk` and `start_acc` are accepted and IGNORED; `start_vel` /
    `end_vel` are honored only tangentially to the path).

    `segment_at_sharp_corners=True` splits at corners sharper than
    `sharp_angle_threshold_deg` and stops at zero path speed on each internal
    boundary, matching the ruckig backend's legacy mode. False (the config
    default) parametrizes the whole path in one problem, letting TOPP-RA slow
    for corners as the spline curvature demands rather than stopping dead.
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
        samples, _ = _toppra_run_segment(
            waypoints,
            sd_start=sd_start,
            sd_end=sd_end,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            control_hz=control_hz,
            per_section_max_velocity=per_section_vel,
            per_section_max_acceleration=per_section_acc,
            per_section_path_speed=per_section_path_speed,
        )
        return samples

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


# Backend selection for `parametrize_path`. "toppra" (default) runs locally;
# "ruckig" restores the jerk-limited profile at the cost of a ~0.5 s cloud API
# round trip per call and a 1000/day quota.
TRAJ_BACKEND_ENV = "SPLATSIM_TRAJ_BACKEND"


def parametrize_path(*args, backend: str | None = None, **kwargs) -> np.ndarray:
    """Time-parametrize a geometric joint path. Dispatches on `backend`.

    `backend`: "toppra" (default) or "ruckig". None reads the
    `SPLATSIM_TRAJ_BACKEND` environment variable, defaulting to "toppra".
    Both backends take the identical signature — see
    `toppra_parametrize_path` for the (small) semantic differences.
    """
    if backend is None:
        backend = os.environ.get(TRAJ_BACKEND_ENV, "toppra").strip().lower()
    if backend == "ruckig":
        return ruckig_parametrize_path(*args, **kwargs)
    if backend == "toppra":
        return toppra_parametrize_path(*args, **kwargs)
    raise ValueError(
        f"Unknown trajectory backend {backend!r} (set {TRAJ_BACKEND_ENV} to 'toppra' or 'ruckig')"
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
    # what removes RRT's characteristic detours/zigzags — ruckig downstream
    # only smooths the TIME parametrization (vel/acc/jerk), it does not
    # straighten the geometric path, so erratic-looking waypoints must be
    # fixed here. Iterations beyond convergence are cheap (a candidate is
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