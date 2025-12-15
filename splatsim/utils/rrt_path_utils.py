import numpy as np
import math
import pybullet as p
import pybullet_data
import time
import itertools

import pybullet as p
import pybullet_data
from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import plan_joint_motion, get_movable_joints
from pybullet_planning import connect, disconnect, set_camera_pose, load_pybullet, \
    wait_for_user, set_joint_positions, get_movable_joints, plan_joint_motion, \
    get_collision_fn, smooth_path, create_box, set_pose, Point, get_extend_fn
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import dump_world, set_pose
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info
import numpy as np
import time
from pybullet_planning.interfaces.robots import get_collision_fn

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

def state_in_collision(robot_id, joint_indices, q, obstacle_ids, distance_threshold=0.01, link_indices_to_check=None, verbose=True):
    """
    Returns True if any robot link is closer than distance_threshold to any obstacle.
    Uses pybullet.getClosestPoints.
    link_indices_to_check: list of link indices to check; if None, we check all links (0..getNumJoints1)
    """
    set_robot_joint_positions(robot_id, joint_indices, q)
    # Allow a small sleep for certain simulators, but generally resetJointState is immediate.
    p.stepSimulation()

    if link_indices_to_check is None:
        link_indices_to_check = joint_indices

    for link_i in link_indices_to_check:
        for obs in obstacle_ids:
            pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=distance_threshold, linkIndexA=link_i, linkIndexB=1)
            if len(pts) > 0: # TODO I'm not sure why there's always 1 point in collision. setting the 1 to 0 made this always true
                if verbose:
                    print(f"Collision detected between link {link_i} and obstacle {obs} with points {len(pts)}")
                return True
    return False

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

def set_robot_joint_positions(robot_id, joint_indices, q):
    for idx, qi in zip(joint_indices, q):
        p.resetJointState(robot_id, idx, qi)
    # Always assume that the robot gripper is open in these demos
    open_gripper(robot_id)
    p.stepSimulation()

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

def are_adjacent_links(robot_id, linkA, linkB):
    """
    Heuristic: skip link pairs that are directly connected.
    """
    if linkA == -1 or linkB == -1:
        return False
    parentA = p.getJointInfo(robot_id, linkA)[16]
    parentB = p.getJointInfo(robot_id, linkB)[16]
    return parentA == linkB or parentB == linkA

def get_rrt_plan(robot_id, joint_indices, obstacle_ids, q_start, q_goal, verbose=True):
    # Use pybullet planning

    if verbose:
        print("Planning with pybullet planning...")
    # move robot to q_start
    set_robot_joint_positions(robot_id, joint_indices, q_start)

    path = plan_joint_motion(
        robot_id,

        # movable_joints[1:6],
        joint_indices,

        q_goal,
        # start_conf=q_start,
        obstacles=obstacle_ids,
        self_collisions=True, #False,
        # algorithm='rrt_connect',  # or 'birrt' / 'rrt'
        # custom_limits=None,
    )
    if path is None:
        if verbose:
            print("PyBullet planning failed.")
        return None
    
    path = np.array(path)

    # Sometimes, the plan is from end to start
    if ((np.array(path[0]) - np.array(q_start))**2).sum() > ((np.array(path[0]) - np.array(q_goal))**2).sum():
        path.reverse()
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

def open_gripper(robot_id):
    # A very hardcoded and temporary solution
    for idx in range(7, p.getNumJoints(robot_id)):
        p.resetJointState(robot_id, idx, 0.0)
    p.stepSimulation()

def get_path(q_start, q_goal, robot_id, joint_indices, obstacle_ids, ll, ul, time_per_traj, robot_update_rate, rrt_vis_fps=5, use_gui=False, verbose=True):
    N_SAMPLES = int(robot_update_rate * time_per_traj)
    # Set joints to q_start
    set_robot_joint_positions(robot_id, joint_indices, q_start)

    # movable_joints = get_movable_joints(robot_id)

    # RRT-Connect planner
    rrt_path = get_rrt_plan(robot_id, joint_indices, obstacle_ids, q_start, q_goal, verbose=verbose)
    if rrt_path is None:
        return None
    
    
    def collision_all_links(q, distance=0.0):
        # q is a sequence for movable_joints
        set_robot_joint_positions(robot_id, joint_indices, q)
        open_gripper(robot_id)

        # let the solver update (resetJointState is immediate but step once to be safe)
        p.stepSimulation()
        # check base link (-1) and all child links
        for link_i in range(-1, p.getNumJoints(robot_id)):
            for obs in obstacle_ids:
                pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=distance,
                                         linkIndexA=link_i, linkIndexB=-1)
                if len(pts) > 0:
                    return True
        # Optionally check self-collisions (uncomment if desired)
        for a,b in itertools.combinations(range(-1, p.getNumJoints(robot_id)), 2):
            if not are_adjacent_links(robot_id, a, b):
                if len(p.getClosestPoints(robot_id, robot_id, distance, linkIndexA=a, linkIndexB=b))>0:
                    return True
        return False
    collision_fn = collision_all_links
    
    # Only checks a subset of all movable joints (aka doesn't check gripper fingers)
    # collision_fn = get_collision_fn(
    #     robot_id, 
    #     joint_indices, 
    #     obstacles=obstacle_ids,
    #     self_collisions=True
    # )

    # 0.05 radians
    resolutions = [0.05] * len(joint_indices)
    extend_fn = get_extend_fn(
        robot_id, 
        joint_indices, 
        resolutions=resolutions
    )
    
    smoothed_path = smooth_path(
        rrt_path.tolist(),
        extend_fn,
        collision_fn,     # The function to check for collisions
        iterations=50     # Number of shortcutting attempts
    )

    # Convert to time-parametrized trajectory
    # Don't use cubic spline because that might introduce obstacle collisions. Do a lerp
    num_samples = max(N_SAMPLES, len(rrt_path))
    time_parametrized_path = resample_path_by_distance(smoothed_path, num_samples)

    # sampler = joints_to_trajectory(rrt_path, total_time=time_per_traj, use_cubic_spline=True)
    # ts, time_parametrized_path = sampler(n_samples=num_samples)

    # Visualize in GUI if requested
    if use_gui:
        # show_joint_config_in_gui(robot_id, joint_indices, q_start)
        # input("Showing start pose. Press Enter to continue...")
        # show_joint_config_in_gui(robot_id, joint_indices, q_goal)
        # input("Showing goal pose. Press Enter to continue...")
        # playback_path_in_gui(rrt_path, robot_id, joint_indices, path_name="RRT", fps=rrt_vis_fps, playback_speed=1.0)
        playback_path_in_gui(time_parametrized_path, robot_id, joint_indices, path_name="Time-Parametrized", fps=robot_update_rate, playback_speed=1.0)
    
    return time_parametrized_path

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
    set_robot_joint_positions(robot_id, joint_indices, q)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)