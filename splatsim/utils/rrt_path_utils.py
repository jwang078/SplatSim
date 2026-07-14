import numpy as np
import math
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

def check_links_in_collision(robot_id, joint_indices, q, obstacle_ids, link_indices_to_check=None, verbose=False, obstacle_names=None, self_collision_clearance=0.0, skip_pairs=None, obstacle_clearance=None, physics_client_id=None, return_kind=False, self_collision_skip_pairs=None):
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

        # Check self-collisions between non-adjacent link pairs
        for a, b in itertools.combinations(link_indices_to_check, 2):
            if _self_skip is not None and frozenset((a, b)) in _self_skip:
                continue
            if not are_adjacent_links(robot_id, a, b, physics_client_id=cid):
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


def get_rrt_plan(robot_id, joint_indices, obstacle_ids, q_start, q_goal,
                 lower_limits=None, upper_limits=None, resolutions=None,
                 verbose=True, obstacle_names=None, skip_pairs=None,
                 physics_client_id=None,
                 obstacle_clearance=None, self_collision_clearance=None,
                 self_collision_skip_pairs=None):
    """Plan a joint-space path from q_start to q_goal with bidirectional RRT.

    `physics_client_id` controls which PyBullet server every call goes to,
    so this works correctly even when multiple clients are connected (e.g.
    SplatSim's GUI server + lerobot's wrapper's DIRECT client).

    `lower_limits`/`upper_limits`/`resolutions` let callers provide joint
    bounds and step sizes directly so we don't need pybullet_planning's
    helpers (which would query the *default* client and might see a
    different body at the same id). When omitted we fall back to those
    helpers — fine when only one client is connected.
    """
    cid = _resolve_client_id(physics_client_id)
    if verbose:
        print("Planning with pybullet planning...")
    set_robot_joint_positions(robot_id, joint_indices, q_start, physics_client_id=cid)

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

    def collision_fn(q):
        return check_links_in_collision(robot_id, joint_indices, q, obstacle_ids,
                                         skip_pairs=skip_pairs, physics_client_id=cid,
                                         **_ccheck_kwargs)

    path = birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
    if path is None:
        start_in_col = check_links_in_collision(robot_id, joint_indices, q_start, obstacle_ids, verbose=True, obstacle_names=obstacle_names, skip_pairs=skip_pairs, physics_client_id=cid, **_ccheck_kwargs)
        goal_in_col = check_links_in_collision(robot_id, joint_indices, q_goal, obstacle_ids, verbose=True, obstacle_names=obstacle_names, skip_pairs=skip_pairs, physics_client_id=cid, **_ccheck_kwargs)
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
) -> tuple:
    """Run ruckig on a single segment. Returns (samples, final_vel, final_acc)."""
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

    traj = Trajectory(dof)
    result = otg.calculate(inp, traj)
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

    Returns:
        (M, DOF) trajectory sampled at control_hz.
    """
    waypoints = np.array(waypoints)
    dof = waypoints.shape[1]
    zeros = np.zeros(dof)
    start_vel = zeros if start_vel is None else np.asarray(start_vel, dtype=np.float64)
    start_acc = zeros if start_acc is None else np.asarray(start_acc, dtype=np.float64)

    if not segment_at_sharp_corners:
        # Fast path: single ruckig call. `_ruckig_run_segment` already passes
        # waypoints[1:-1] as intermediate_positions, so ruckig handles
        # corner decel internally without zero-velocity stops.
        samples, _, _ = _ruckig_run_segment(
            waypoints,
            start_vel=start_vel,
            start_acc=start_acc,
            end_vel=zeros,
            end_acc=zeros,
            max_joint_vel=max_joint_vel,
            max_joint_acc=max_joint_acc,
            max_joint_jerk=max_joint_jerk,
            control_hz=control_hz,
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
        # End at zero only when the next boundary is genuinely sharp or it's
        # the trajectory's final point. At other boundaries, pass through.
        is_next_sharp = (
            not is_last and split_points[seg_idx + 1] in sharp_indices
        )
        seg_end_vel = zeros if (is_last or is_next_sharp) else zeros
        seg_end_acc = zeros
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
        )
        if not is_last:
            samples = samples[:-1]
        all_samples.append(samples)
        prev_end_vel = end_v
        prev_end_acc = end_a
    return np.concatenate(all_samples, axis=0)


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



def get_path(q_start, q_goal, robot_id, joint_indices, obstacle_ids, ll, ul, robot_update_rate, use_gui=False, verbose=True, max_joint_vel=None, max_joint_acc=None, max_joint_jerk=None, obstacle_names=None, skip_pairs=None, physics_client_id=None, obstacle_clearance=None, self_collision_clearance=None, self_collision_skip_pairs=None):
    cid = _resolve_client_id(physics_client_id)
    dof = len(joint_indices)
    if max_joint_vel is None:
        max_joint_vel = np.full(dof, 0.5)   # rad/s
    if max_joint_acc is None:
        max_joint_acc = np.full(dof, 1.0)   # rad/s^2
    if max_joint_jerk is None:
        max_joint_jerk = np.full(dof, 10.0)  # rad/s^3, ~10x max_acc
    # Set joints to q_start
    set_robot_joint_positions(robot_id, joint_indices, q_start, physics_client_id=cid)

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
    )
    if rrt_path is None:
        return None


    def collision_fn(q):
        return check_links_in_collision(robot_id, joint_indices, q, obstacle_ids,
                                         skip_pairs=skip_pairs, physics_client_id=cid,
                                         **_ccheck_kwargs)

    extend_fn = _make_linear_extend_fn(resolutions)

    smoothed_path = smooth_path(
        rrt_path.tolist(),
        extend_fn,
        collision_fn,
        max_smooth_iterations=50,
    )

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