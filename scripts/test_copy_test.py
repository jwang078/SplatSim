import numpy as np
import math
import pybullet as p
import pybullet_data
import yaml
import time
import random
import argparse
import zarr
import os
import shutil
from tqdm import tqdm

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

def compute_collision_gradients_at_q(robot_id, joint_indices, q,
                                     obstacle_ids, margin=0.1, link_indices_to_check=None,
                                     safe_distance=0.05, obstacle_gain=1.0):
    """
    Compute joint-space gradient of collision penalty for a single robot configuration q.
    Returns a gradient vector of size len(joint_indices) (numpy).
    - margin: how far to query getClosestPoints (only returns pairs with distance <= margin)
    - safe_distance: cost is active when d < safe_distance
    - obstacle_gain: multiplier on gradient magnitude
    """
    dof = len(joint_indices)
    grad_q = np.zeros(dof, dtype=float)

    # set robot state (use resetJointState for faster non-sim)
    for idx, qi in zip(joint_indices, q):
        p.resetJointState(robot_id, idx, qi)

    # which links to check: default all (0..n-1) + base (-1)
    if link_indices_to_check is None:
        # linkIndexA must be in [-1 .. n-1]. We'll skip -1 because calculateJacobian for -1 is not supported.
        link_indices = list(range(0, p.getNumJoints(robot_id)))
    else:
        link_indices = link_indices_to_check

    # cached zeros for jacobian call
    zeros_q = [0.0]*dof
    dq = [0.0]*dof
    ddq = [0.0]*dof

    for link_idx in link_indices:
        for obs in obstacle_ids:
            pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=margin, linkIndexA=link_idx, linkIndexB=-1)
            if not pts:
                continue
            # process each close point (there can be multiple)
            for pt in pts:
                # tuple structure varies by pybullet version. Common mapping:
                # pt[5] -> positionOnA_world, pt[6] -> positionOnB_world
                # pt[7] -> contactNormalOnB (points from B -> A?), pt[8] -> contactDistance
                try:
                    pos_on_a = pt[5]
                    pos_on_b = pt[6]
                    normal_on_b = np.array(pt[7], dtype=float)
                    dist = float(pt[8])
                except Exception:
                    # If indices don't match your version, print to debug
                    contact_tuple_debug(pt)
                    raise RuntimeError("Unexpected getClosestPoints tuple layout. Print tuple to inspect indices.")

                # We need a normal that points from obstacle toward robot point (so steering away).
                # If normal_on_b is normal on B pointing toward A, then it is the correct direction.
                n_world = normal_on_b / (np.linalg.norm(normal_on_b) + 1e-12)

                # signed distance convention: dist (positive if separated). We want cost when dist < safe_distance
                if dist >= safe_distance:
                    continue

                # compute contact point in link-local coordinates for Jacobian
                # need link world pose
                link_state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=1)
                link_world_pos = link_state[0]
                link_world_orn = link_state[1]
                # convert pos_on_a (world) to local coordinates of the link
                local_pos = world_to_local(link_world_pos, link_world_orn, pos_on_a)

                # get linear jacobian (3 x dof) at local_pos
                # calculateJacobian(robot, linkIndex, localPosition, jointPositions, jointVelocities, jointAccelerations)
                import pdb; pdb.set_trace()
                J_lin, J_ang = p.calculateJacobian(robot_id, link_idx, list(local_pos),
                                                   list(q.tolist()), dq, ddq)
                # J_lin is lists of lists; convert to numpy 3xdof
                J_lin = np.array(J_lin)  # shape (3, dof)
                # gradient of distance d wrt joints is J_lin^T * n_world (approx)
                # cost c = 0.5 * (d - safe)^2  => dc/dd = (d - safe)
                # d is contactDistance (positive if separated), but when penetrating it may be negative.
                hinge = (dist - safe_distance)  # negative when inside margin
                # gradient of cost w.r.t q: hinge * (d grad/dq) = hinge * (J^T * n)
                grad_here = hinge * (J_lin.T.dot(n_world))  # shape (dof,)
                grad_q += obstacle_gain * grad_here

    return grad_q

###########################
# Utility / Collision API #
###########################

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
    return np.array(lowers)[:7], np.array(uppers)[:7]

def set_robot_joint_positions(robot_id, joint_indices, q):
    for idx, qi in zip(joint_indices, q):
        p.resetJointState(robot_id, idx, qi)

def state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.01, link_indices_to_check=None):
    """
    Returns True if any robot link is closer than distance_threshold to any obstacle.
    Uses pybullet.getClosestPoints.
    link_indices_to_check: list of link indices to check; if None, we check all links (0..getNumJoints-1)
    """
    set_robot_joint_positions(robot_id, joint_indices, q)
    # Allow a small sleep for certain simulators, but generally resetJointState is immediate.
    # p.stepSimulation()

    if link_indices_to_check is None:
        # gather all link indices (0..n-1)
        link_indices = list(range(-1, p.getNumJoints(robot_id)))  # include base (-1) and links
    else:
        link_indices = link_indices_to_check

    for link_i in link_indices:
        for obs in obstacle_ids:
            pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=distance_threshold, linkIndexA=link_i, linkIndexB=-1)
            if len(pts) > 0:
                return True
    return False

def min_distance_to_obstacles(robot_id, joint_indices, q, obstacle_ids, link_indices_to_check=None, max_dist=5.0):
    """Return minimum distance between robot (at q) and the set of obstacles (useful for soft cost)."""
    set_robot_joint_positions(robot_id, joint_indices, q)
    if link_indices_to_check is None:
        link_indices = list(range(-1, p.getNumJoints(robot_id)))
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
# RRT-Connect in Jointspace
###########################

class RRTConnect:
    def __init__(self, robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits,
                 step_size=0.2, max_iters=20000, collision_distance=0.01, link_indices_to_check=None):
        self.robot_id = robot_id
        self.jidx = joint_indices
        self.obstacle_ids = obstacle_ids
        self.ll = np.array(lower_limits)
        self.ul = np.array(upper_limits)
        self.step_size = step_size
        self.max_iters = max_iters
        self.collision_distance = collision_distance
        self.link_indices_to_check = link_indices_to_check

    def sample(self):
        return np.random.uniform(self.ll, self.ul)

    def nearest(self, tree_nodes, q):
        # tree_nodes: list of ndarrays
        dists = [np.linalg.norm(q - n) for n in tree_nodes]
        return int(np.argmin(dists))

    def steer(self, q_src, q_tgt):
        v = q_tgt - q_src
        dist = np.linalg.norm(v)
        if dist <= self.step_size:
            return q_tgt.copy()
        else:
            return q_src + v / dist * self.step_size

    def collision_free_segment(self, q_from, q_to, n_steps=10):
        # linear interpolation in joint space; check collisions
        for alpha in np.linspace(0.0, 1.0, n_steps):
            q = (1 - alpha) * q_from + alpha * q_to
            if state_in_collision(self.robot_id, self.jidx, q, self.obstacle_ids,
                                  distance_threshold=self.collision_distance,
                                  link_indices_to_check=self.link_indices_to_check):
                return False
        return True

    def try_connect(self, tree_nodes, tree_parents, q_target):
        """
        Try to extend tree towards q_target (one step each time) until cannot extend or reached.
        Returns: (new_node_index, reached_flag)
        """
        idx_near = self.nearest(tree_nodes, q_target)
        q_near = tree_nodes[idx_near]
        q_new = self.steer(q_near, q_target)
        if not self.collision_free_segment(q_near, q_new):
            return None, False
        tree_nodes.append(q_new)
        tree_parents.append(idx_near)
        reached = np.allclose(q_new, q_target, atol=1e-6) or np.linalg.norm(q_new - q_target) < 1e-6
        return len(tree_nodes) - 1, reached

    def plan(self, q_start, q_goal, timeout=10.0):
        t0 = time.time()
        if state_in_collision(self.robot_id, self.jidx, q_start, self.obstacle_ids,
                              distance_threshold=self.collision_distance,
                              link_indices_to_check=self.link_indices_to_check):
            raise RuntimeError("Start in collision")
        if state_in_collision(self.robot_id, self.jidx, q_goal, self.obstacle_ids,
                              distance_threshold=self.collision_distance,
                              link_indices_to_check=self.link_indices_to_check):
            raise RuntimeError("Goal in collision")

        tree_a_nodes = [np.array(q_start)]
        tree_a_parents = [-1]
        tree_b_nodes = [np.array(q_goal)]
        tree_b_parents = [-1]

        for it in range(self.max_iters):
            if time.time() - t0 > timeout:
                break

            q_rand = self.sample()
            # extend tree A towards q_rand
            idx_new_a, _ = self.try_connect(tree_a_nodes, tree_a_parents, q_rand)
            if idx_new_a is not None:
                # try connect tree B to this new node
                idx_new_b, reached = self.try_connect(tree_b_nodes, tree_b_parents, tree_a_nodes[idx_new_a])
                if idx_new_b is not None and reached:
                    # found connection: reconstruct path
                    path_a = self.reconstruct_path(tree_a_nodes, tree_a_parents, idx_new_a)
                    path_b = self.reconstruct_path(tree_b_nodes, tree_b_parents, idx_new_b)
                    path_b.reverse()
                    path = path_a + path_b
                    return path
            # swap roles
            tree_a_nodes, tree_b_nodes = tree_b_nodes, tree_a_nodes
            tree_a_parents, tree_b_parents = tree_b_parents, tree_a_parents

        raise RuntimeError("RRT-Connect failed (timeout/iterations)")

    def reconstruct_path(self, nodes, parents, idx):
        path = []
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path

###########################
# Shortcut Smoothing
###########################

def get_shortcut_smoothed_path(path, robot_id, joint_indices, obstacle_ids, collision_distance=0.01, iterations=200):
    """
    path: list of joint vectors (ndarray)
    For iterations, pick random i<j and if straight interpolation between them is collision free, remove intermediates.
    """
    if len(path) <= 2:
        return path
    for it in range(iterations):
        n = len(path)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        if j == i + 1:
            continue
        if np.allclose(path[i], path[j]):
            # identical
            path = path[:i+1] + path[j:]
            continue
        # check segment
        rrt = RRTConnect(robot_id, joint_indices, obstacle_ids, np.zeros(len(joint_indices))-math.pi, np.zeros(len(joint_indices))+math.pi,
                         step_size=0.2, max_iters=1)
        if rrt.collision_free_segment(path[i], path[j], n_steps=max(6, j - i)):
            # keep endpoints only
            path = path[:i+1] + path[j:]
    return path

###########################
# CHOMP-like Smoother (simple)
###########################

def chomp_smooth(path, robot_id, joint_indices, obstacle_ids,
                 iterations=200, alpha=0.1, smoothing_weight=100.0, obs_weight=1.0,
                 obstacle_distance_gain=10.0, min_clearance=0.05):
    """
    path: list of joint vectors (ndarray)
    Minimizes smoothness (second derivative) + obstacle cost using finite difference gradients.
    This is a simple and direct implementation — tune iterations/weights for performance.
    """
    path = [np.array(q).astype(float) for q in path]
    N = len(path)
    if N < 3:
        return path

    # fix endpoints
    q0 = path[0].copy()
    qf = path[-1].copy()

    for it in range(iterations):
        grads = [np.zeros_like(path[0]) for _ in range(N)]
        # Smoothness gradient: second finite difference (discrete Laplacian)
        for i in range(1, N-1):
            grads[i] += smoothing_weight * (2 * path[i] - path[i-1] - path[i+1])

        # Obstacle gradient: finite differences per waypoint
        for i in range(1, N-1):  # don't move endpoints
            q = path[i]
            d = min_distance_to_obstacles(robot_id, joint_indices, q, obstacle_ids, max_dist=2.0)
            # soft cost: only when closer than a margin
            margin = min_clearance
            if d < margin:
                # cost = obs_weight * exp(-k*(d))  (we'll use simple hinge gradient)
                # approximate gradient by finite diffs on each joint
                eps = 1e-4
                grad = np.zeros_like(q)
                base_cost = math.exp(-obstacle_distance_gain * (d - margin))
                for j in range(len(q)):
                    q_pert = q.copy()
                    q_pert[j] += eps
                    d2 = min_distance_to_obstacles(robot_id, joint_indices, q_pert, obstacle_ids, max_dist=2.0)
                    # numeric derivative of cost w.r.t joint j
                    cost_pert = math.exp(-obstacle_distance_gain * (d2 - margin))
                    grad[j] = (cost_pert - base_cost) / eps
                grads[i] += obs_weight * grad

        # gradient descent update (project endpoints fixed)
        for i in range(1, N-1):
            path[i] = path[i] - alpha * grads[i]

        # re-lock endpoints
        path[0] = q0
        path[-1] = qf

        # optional early exit if collisions resolved and smooth
        if it % 10 == 0:
            # check collisions
            collision_found = False
            for q in path:
                if state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.0):
                    collision_found = True
                    break
            if not collision_found:
                # continue refining but this is a good sign
                pass

    return path

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

def chomp_with_jacobian(path, robot_id, joint_indices, obstacle_ids,
                       iterations=200, alpha=0.05,
                       smoothing_weight=100.0, safe_distance=0.06,
                       margin=0.2, obstacle_gain=3.0, link_indices_to_check=None,
                       joint_lower=None, joint_upper=None):
    """
    path: list of numpy joint vectors (len M)
    Returns refined path (list of numpy joint vectors).
    - Uses analytic collision gradients computed from PyBullet jacobians.
    - smoothing_weight: weight on Laplacian smoothness (2*q_i - q_{i-1} - q_{i+1})
    - safe_distance: threshold for activating collision cost
    - margin: getClosestPoints query distance (only pairs within margin are considered)
    - obstacle_gain: scales collision gradient magnitude
    - joint_lower / joint_upper: arrays to clamp joint values after update
    """
    path = [np.array(q, dtype=float) for q in path]
    M = len(path)
    DOF = len(path[0])
    if joint_lower is None or joint_upper is None:
        # If not provided, use wide bounds
        joint_lower = np.ones(DOF) * -math.pi
        joint_upper = np.ones(DOF) * math.pi

    q0 = path[0].copy()
    qf = path[-1].copy()

    for it in range(iterations):
        grads = [np.zeros(DOF, dtype=float) for _ in range(M)]

        # Smoothness gradient (discrete second derivative)
        for i in range(1, M-1):
            grads[i] += smoothing_weight * (2 * path[i] - path[i-1] - path[i+1])

        # Collision gradients from jacobian
        for i in range(1, M-1):  # don't move endpoints
            q = path[i]
            # compute analytic collision gradient at this q
            cg = compute_collision_gradients_at_q(robot_id, joint_indices, q,
                                                  obstacle_ids, margin=margin,
                                                  link_indices_to_check=link_indices_to_check,
                                                  safe_distance=safe_distance,
                                                  obstacle_gain=obstacle_gain)
            import pdb; pdb.set_trace()
            grads[i] += cg

        # Gradient descent step
        for i in range(1, M-1):
            path[i] = path[i] - alpha * grads[i]
            # clamp joint limits
            path[i] = np.minimum(np.maximum(path[i], joint_lower), joint_upper)

        # re-lock endpoints
        path[0] = q0
        path[-1] = qf

        # optional quick diagnostics
        if it % 10 == 0:
            # check if any waypoint collides (distance < 0)
            coll_any = False
            for q in path:
                # use direct getClosestPoints with distance=0
                for link_idx in (link_indices_to_check if link_indices_to_check is not None else range(p.getNumJoints(robot_id))):
                    for obs in obstacle_ids:
                        pts = p.getClosestPoints(robot_id, obs, distance=0.0, linkIndexA=link_idx, linkIndexB=-1)
                        if pts:
                            coll_any = True
                            break
                    if coll_any:
                        break
                if coll_any:
                    break
            # print progress
            print(f"[chomp] iter {it}/{iterations}, collision_present={coll_any}")

    return path

def get_random_joint_angles_without_collision(robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits, max_tries=100):
    for _ in range(max_tries):
        q = np.random.uniform(lower_limits, upper_limits)
        if not state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.0):
            return q
    raise RuntimeError("Failed to find collision-free joint angles after many tries")

def setup_env(args):
    if args.gui:
        cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load plane + obstacles here for demo; user should load their 30-300 cuboids and collect their body ids
    plane = p.loadURDF("plane.urdf")

    cuboid_bboxes = load_cuboids(args.cuboids_fn)

    # load robot
    flags = p.URDF_USE_INERTIA_FROM_FILE
    robot_id = p.loadURDF(args.urdf, useFixedBase=True, flags=flags)

    # get joints
    joint_indices = get_movable_joints(robot_id)
    if len(joint_indices) != 7:
        print("Warning: detected movable joints:", len(joint_indices), "expected 7.")

    ll, ul = get_joint_limits(robot_id, joint_indices)

    # For demo: create a few cuboid obstacles (user should replace with their own obstacles and keep their ids)
    obstacle_ids = []
    # Example random obstacles (comment out and use your actual obstacles)
    for cuboid_bbox in cuboid_bboxes:
        cx, cy, cz, lx, ly, lz = cuboid_bbox
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2], rgbaColor=[0.8, 0.3, 0.3, 1])
        obs = p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[cx, cy, cz])
        obstacle_ids.append(obs)
    obstacle_ids.append(plane)

    return ll, ul, obstacle_ids, robot_id, joint_indices

def get_rrt_plan(robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits, q_start, q_goal, step_size=0.2, max_iters=20000, collision_distance=0.01):
    rrt = RRTConnect(robot_id, joint_indices, obstacle_ids, lower_limits, upper_limits, step_size=step_size, max_iters=max_iters, collision_distance=collision_distance)
    print("Planning RRT-Connect...")
    try:
        path = rrt.plan(q_start, q_goal, timeout=15.0)
    except RuntimeError as e:
        print("RRT-Connect planning failed:", e)
        return None
    # Sometimes, the plan is from end to start
    if ((path[0] - q_start)**2).sum() > ((path[0] - q_goal)**2).sum():
        path.reverse()
    print("RRT raw path length:", len(path))
    return path

def get_path(q_start, q_goal, robot_id, joint_indices, obstacle_ids, ll, ul, time_per_traj, robot_update_rate, rrt_vis_fps=5, use_gui=False, use_chomp=False):
    N_SAMPLES = int(robot_update_rate * time_per_traj)
    # Set joints to q_start
    set_robot_joint_positions(robot_id, joint_indices, q_start)

    # Visualize the q start
    if use_gui:
        show_joint_config_in_gui(robot_id, joint_indices, q_start)
        input("Showing start pose. Press Enter to continue...")
        show_joint_config_in_gui(robot_id, joint_indices, q_goal)
        input("Showing goal pose. Press Enter to continue...")

    # RRT-Connect planner
    rrt_path = get_rrt_plan(robot_id, joint_indices, obstacle_ids, ll, ul, q_start, q_goal, step_size=0.2, max_iters=20000, collision_distance=0.01)
    if rrt_path is None:
        return None

    # Shortcut smoothing
    print("Shortcut smoothing...")
    shortcut_rrt_path = get_shortcut_smoothed_path(rrt_path, robot_id, joint_indices, obstacle_ids, iterations=300)
    print("Smoothed path length:", len(shortcut_rrt_path))

    if use_chomp:
        """
        # CHOMP refinement (optional, slower)
        # print("Refining with CHOMP-like optimizer...")
        # path = chomp_smooth(path, robot_id, joint_indices, obstacle_ids,
        #                     iterations=200, alpha=0.05, smoothing_weight=50.0,
        #                     obs_weight=10.0, obstacle_distance_gain=30.0, min_clearance=0.06)
        # print("Refined path length:", len(path))
        # chomp_len = len(path)

        # # Play back the path in GUI
        # if args.gui:
        #     input("Press Enter to show CHOMP-refined path...")
        #     for q in path:
        #         set_robot_joint_positions(robot_id, joint_indices, q)
        #         p.stepSimulation()
        #         time.sleep(1.0 / fps_in_playback * rrt_len / chomp_len)
        """

        chomp_path = chomp_with_jacobian(
            shortcut_rrt_path,
            robot_id,
            joint_indices,
            obstacle_ids,
            iterations=150,
            alpha=0.03,
            smoothing_weight=80.0,
            safe_distance=0.06,
            margin=0.12,
            obstacle_gain=4.0,
            link_indices_to_check=list(range(1, 7)),
            joint_lower=ll,
            joint_upper=ul
        )
    else:
        chomp_path = shortcut_rrt_path

    # Convert to time-parametrized trajectory
    sampler = joints_to_trajectory(chomp_path, total_time=time_per_traj, use_cubic_spline=True)
    ts, time_parametrized_path = sampler(n_samples=N_SAMPLES)

    # Visualize in GUI if requested
    if use_gui:
        playback_path_in_gui(rrt_path, robot_id, joint_indices, path_name="RRT", fps=rrt_vis_fps, playback_speed=1.0)
        playback_path_in_gui(shortcut_rrt_path, robot_id, joint_indices, path_name="Shortcut-Smoothed RRT", fps=rrt_vis_fps, playback_speed=len(shortcut_rrt_path)/ len(rrt_path))
        playback_path_in_gui(chomp_path, robot_id, joint_indices, path_name="CHOMP-Jacobian Smoothed", fps=rrt_vis_fps, playback_speed=len(chomp_path)/ len(shortcut_rrt_path))
        playback_path_in_gui(time_parametrized_path, robot_id, joint_indices, path_name="Time-Parametrized", fps=robot_update_rate, playback_speed=1.0)
    
    return time_parametrized_path

def playback_path_in_gui(path, robot_id, joint_indices, path_name, fps=240, playback_speed=1.0):
    if not p.isConnected():
        print("Not connected to PyBullet GUI.")
        return
    input(f"Press Enter to play back the {path_name} path...")
    for q in path:
        set_robot_joint_positions(robot_id, joint_indices, q)
        p.stepSimulation()
        time.sleep(1.0 / fps / playback_speed)

def show_joint_config_in_gui(robot_id, joint_indices, q):
    if not p.isConnected():
        print("Not connected to PyBullet GUI.")
        return
    set_robot_joint_positions(robot_id, joint_indices, q)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

def main(args):
    N_SAMPLES = int(args.robot_update_rate * args.time_per_traj)

    if not args.no_save:
        os.makedirs("outputs", exist_ok=True)
        load_dir = os.path.join("output", args.experiment_name + ".zarr")
        output_dir = os.path.join("output", args.experiment_name + "_new.zarr")
        # if args.delete_existing and os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        load_root = zarr.open(load_dir, mode='r')
        root_output = zarr.open(output_dir, mode='a')
        
        if 'joint_trajectories' not in root_output:
            traj_ds = root_output.create_dataset(
                'joint_trajectories',
                shape=(0, N_SAMPLES, args.dof),
                chunks=(10_000, args.dof),  # example chunk size
                dtype='f4', # float32
                maxshape=(None, N_SAMPLES, args.dof)
            )
        else:
            traj_ds = root_output['joint_trajectories']
            # Assert that the existing DOF matches args.DOF
            assert traj_ds.ndim == 3, f"Existing dataset has ndim {traj_ds.ndim}, expected 3"
            assert traj_ds.shape[1] == N_SAMPLES, f"Existing dataset samples {traj_ds.shape[1]} does not match expected {N_SAMPLES}"
            assert traj_ds.shape[2] == args.dof, f"Existing dataset DOF {traj_ds.shape[1]} does not match args.DOF {args.dof}"
        # Start q_start from the end of this dataset
        q_start = traj_ds[-1] if traj_ds.shape[0] > 0 else None
        if q_start is not None:
            print("Continuing from existing dataset, starting at:", q_start)
    else:
        q_start = None

    loaded_traj_ds = np.array(load_root["joint_trajectories"]).reshape((-1, N_SAMPLES, args.dof))
    traj_ds.resize((loaded_traj_ds.shape[0], N_SAMPLES, args.dof))
    traj_ds[:loaded_traj_ds.shape[0], :, :] = loaded_traj_ds
    print("Copied existing trajectories, total now:", traj_ds.shape[0])


if __name__ == "__main__":
    with open("/home/jennyw2/code/SplatSim/configs/object_configs/objects.yaml", 'r') as file:
        object_config = yaml.safe_load(file)
    robot_name = "robot_iphone_w_engine"
    robot_config = object_config[robot_name]
    SISBOT_PATH = "/home/jennyw2/code/SplatSim/" + robot_config['urdf_path'][0]
    initial_joint_positions = np.array(robot_config['joint_states'][0][1:8])
    num_obstacles = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=False, help="Path to robot URDF", default=SISBOT_PATH)
    parser.add_argument("--start", nargs='+', type=float, help="start joint values (7 floats)", required=False, default=None)
    parser.add_argument("--goal", nargs='+', type=float, help="goal joint values (7 floats)", required=False, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument('--planner_method', type=str, default='L-BFGS-B', help='Optimization method for scipy minimize')
    parser.add_argument("--cuboids_fn", type=str, default="/home/jennyw2/code/fabrics/outputs/cuboids_voxel0.050.npz", help="Path to npz file with cuboids")

    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--rrt_vis_fps", type=int, default=5)
    parser.add_argument("--time_per_traj", type=float, default=6.0, help="seconds")
    parser.add_argument("--robot_update_rate", type=int, default=20, help="Hz")
    parser.add_argument("--experiment_name", type=str, default="test", help="Name for the experiment (for saving results)")
    parser.add_argument("--delete_existing", action="store_true", help="If set, clear existing output directory")
    parser.add_argument("--num_trajectories", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--no-save", action="store_true", help="If set, do not save trajectories to disk")
    parser.add_argument("--use_chomp", action="store_true", help="If set, use CHOMP-like smoothing after RRT and shortcutting")
    args = parser.parse_args()
    main(args)