# pip install pybullet numpy scipy
import pybullet as p
import pybullet_data
import numpy as np
from scipy.optimize import minimize
import os

import time
from tqdm import tqdm
import yaml

import argparse

import os
import yaml
from argparse import ArgumentParser
from gaussian_splatting.gaussian_renderer import GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams, Namespace
from gaussian_splatting.scene import Scene
import torch

with open(
    "/home/jennyw2/code/SplatSim/configs/object_configs/objects.yaml", "r"
) as file:
    object_config = yaml.safe_load(file)
robot_name = "robot_iphone_w_engine"
robot_config = object_config[robot_name]
SISBOT_PATH = "/home/jennyw2/code/SplatSim/" + robot_config["urdf_path"][0]
initial_joint_positions = np.array(robot_config["joint_states"][0][1:8])
num_obstacles = 1

parser = argparse.ArgumentParser(description="Playback a saved trajectory in PyBullet.")
parser.add_argument("--scenario", type=str, default="big_box")
parser.add_argument(
    "--planner_method",
    type=str,
    choices=["L-BFGS-B", "SLSQP"],
    default="L-BFGS-B",
    help="Optimization method to use.",
)
parser.add_argument(
    "--use_cache",
    action="store_true",
    help="If set, skip the optimization step and only playback the reference trajectory.",
)
args = parser.parse_args()

output_folder = f"output/planner_outputs/{args.planner_method}/{args.scenario}"
os.makedirs(output_folder, exist_ok=True)

ref_path = f"{output_folder}/q_ref.npz"
opt_path = f"{output_folder}/q_opt.npz"

with open("configs/object_configs/objects.yaml", "r") as file:
    object_config = yaml.safe_load(file)

w_track = 1.0  # weight for tracking q_ref
w_track_pos = 0.1  # Position weight
w_track_orient = (
    1  # Orientation weight (often higher due to small quaternion error values)
)
w_smooth = 2e-1  # weight for velocity smoothness
w_collision = 1000  # 1000.0       # weight for collision penalty
safe_margin = 0.05  # meters; penalize if closer than this
search_distance = 1  # meters; only check collisions within this distance
T = 40  # number of waypoints
dt = 0.05

# ---------------------------------
# 4) Helper: set joints / clearance
# ---------------------------------
def set_q(q):
    for idx, j in enumerate(joint_ids):
        p.resetJointState(robot, j, q[idx])


def min_clearance(q):
    """Compute minimum signed-ish clearance robot↔obstacle at config q.
    Uses getClosestPoints; returns positive distance (m)."""
    if obstacle is None:
        return 1.0  # no obstacle, always far
    set_q(q)
    min_d = np.inf
    # Broad phase: check each link body against the obstacle
    for link in joint_ids:
        pts = p.getClosestPoints(
            bodyA=robot,
            bodyB=obstacle,
            distance=search_distance,
            linkIndexA=link,
            linkIndexB=-1,
        )
        if pts:
            d = min(pt[8] for pt in pts)  # pt[8] is distance
            if d < min_d:
                min_d = d
    if min_d is np.inf:
        min_d = 1.0  # far away
    return float(min_d)


# Vectorized min clearance over a whole trajectory
def min_clearance_traj(Q):
    return np.array([min_clearance(q) for q in Q])


def load_gaussian_splat():
    cam_i = 3

    source_path = object_config[robot_name]["source_path"]
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    model_path = object_config[robot_name]["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    parser = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(parser)
    model = ModelParams(parser, sentinel=True)
    dataset = model.extract(
        Namespace(
            sh_degree=3,
            # TODO get these from the object config
            source_path=source_path,
            model_path=model_path,
            images="images",
            depths="",
            resolution=-1,
            white_background=False,
            train_test_exp=False,
            data_device="cuda",
            eval=False,
        )
    )
    gaussians_backup = GaussianModel(dataset.sh_degree)
    # This loads the .ply file into gaussians_backup
    cam_scale = 2  # A scale of 2 produces a smaller image than a scale of 1
    scene = Scene(
        dataset,
        gaussians_backup,
        load_iteration=-1,
        shuffle=False,
        resolution_scales=[cam_scale],
        train_cam_indices=[cam_i],
        test_cam_indices=[
            0
        ],  # Even tho we're not using this, make sure to load at max 1 camera for memory purposes
    )

    # Transform the xyz points and shs to the world frame
    robot_transformation = object_config[robot_name]["transformation"]["matrix"]
    Trans = (
        torch.tensor(robot_transformation)
        .to(device=gaussians_backup._xyz.device)
        .float()
    )
    # scale_robot = torch.pow(torch.linalg.det(Trans[:3, :3]), 1/3)
    inv_transformation_matrix = torch.inverse(Trans)

    # TODO transform the SHS values as well
    gaussians_backup._xyz = torch.matmul(
        torch.cat(
            [gaussians_backup._xyz, torch.ones_like(gaussians_backup._xyz[:, :1])],
            dim=1,
        ),
        Trans.T,
    )[:, :3]

    return gaussians_backup


# -----------------------------
# 1) PyBullet setup (headless)
# -----------------------------
print("Setting up pybullet...")
physics = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.resetDebugVisualizerCamera(
    cameraDistance=1,  # Default Zoom
    cameraYaw=50 + 150,  # 50 is Typical Default Yaw
    cameraPitch=-35,  # Typical Default Pitch (looking down)
    cameraTargetPosition=[0.5, 0.5, 0.5],  # Default Focus Point
)

plane = p.loadURDF("plane.urdf")
# load your manipulator; replace with your URDF + base pose
robot = p.loadURDF("./splatsim/robot_definitions/urdf/sisbot.urdf", useFixedBase=True)
ee_joint_number = 6

if args.scenario == "no_obstacle":
    # No obstacle
    box_col = None
    box_vis = None
    obstacle = None
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "big_box":
    # Simple obstacle: a box
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
    obstacle = p.createMultiBody(
        baseCollisionShapeIndex=box_col,
        baseVisualShapeIndex=box_vis,
        basePosition=[0.6, 0.0, 0.4],
    )
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "small_box":
    # Simple obstacle: a smaller box
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    obstacle = p.createMultiBody(
        baseCollisionShapeIndex=box_col,
        baseVisualShapeIndex=box_vis,
        basePosition=[0.5, 0.0, 0.2],
    )
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "big_sphere":
    # Simple obstacle: a sphere
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2)
    obstacle = p.createMultiBody(
        baseCollisionShapeIndex=sphere_col,
        baseVisualShapeIndex=sphere_vis,
        basePosition=[0.6, 0.0, 0.4],
    )
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "small_sphere":
    # Simple obstacle: a smaller sphere
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1)
    obstacle = p.createMultiBody(
        baseCollisionShapeIndex=sphere_col,
        baseVisualShapeIndex=sphere_vis,
        basePosition=[0.5, 0.0, 0.2],
    )
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "small_engine_side":
    urdf_path = object_config["small_engine"]["urdf_path"][0]
    pos = [0.3, 0.55]
    quat = [0, 0, 1, 0]
    base_position = object_config["small_engine"].get("base_position", [[0, 0, 0]])[0]
    pos = [
        pos[0] + base_position[0],
        pos[1] + base_position[1],
        0 + base_position[2],
    ]
    obstacle = p.loadURDF(urdf_path, pos, quat, useFixedBase=True)
    # Start Pose (XYZ in meters)
    start_pos = [-0.1, 0.7, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.5, 0, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
elif args.scenario == "small_engine_middle":
    urdf_path = object_config["small_engine"]["urdf_path"][0]
    pos = [0.3, 0]
    quat = [0, 0, 1, 0]
    base_position = object_config["small_engine"].get("base_position", [[0, 0, 0]])[0]
    pos = [
        pos[0] + base_position[0],
        pos[1] + base_position[1],
        0 + base_position[2],
    ]
    obstacle = p.loadURDF(urdf_path, pos, quat, useFixedBase=True)
    # Start Pose (XYZ in meters)
    start_pos = [0.3, 0.5, 0.7]
    # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
    start_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])  # e.g., Pitch 90 degrees

    # Goal Pose (XYZ in meters)
    goal_pos = [0.7, -0.3, 0.2]
    # Goal Orientation (Quaternion: [x, y, z, w])
    goal_quat = p.getQuaternionFromEuler([0, np.pi / 2, 0])  # e.g., Roll 90 degrees
# elif args.scenario == "small_engine_splat":
#     # TODO actually this is better implemented within splatsim
#     gaussian_splat = load_gaussian_splat()

#     obstacle = None
#     # Start Pose (XYZ in meters)
#     start_pos = [0.3, 0.5, 0.7]
#     # Start Orientation (Quaternion: [x, y, z, w]) - This is close to a 'tool down' orientation
#     start_quat = p.getQuaternionFromEuler([np.pi/2, 0, 0]) # e.g., Pitch 90 degrees

#     # Goal Pose (XYZ in meters)
#     goal_pos = [0.7, -0.3, 0.2]
#     # Goal Orientation (Quaternion: [x, y, z, w])
#     goal_quat = p.getQuaternionFromEuler([0, np.pi/2, 0]) # e.g., Roll 90 degrees
else:
    raise ValueError(f"Unknown scenario {args.scenario}")


# -----------------------------
# 2) Robot joint metadata
# -----------------------------
print("Getting robot joint info...")
joint_ids = []
lower, upper, vel, torque = [], [], [], []
for j in range(p.getNumJoints(robot)):
    ji = p.getJointInfo(robot, j)
    if ji[2] == p.JOINT_REVOLUTE or ji[2] == p.JOINT_PRISMATIC:
        joint_ids.append(j)
        lower.append(ji[8])
        upper.append(ji[9])
        vel.append(ji[11])
        torque.append(ji[10])
        if len(joint_ids) == ee_joint_number + 1:
            print(
                f"End effector joint name (index {ee_joint_number}):",
                ji[1].decode("utf-8"),
            )

nq = len(joint_ids)
lower = np.array(lower)
upper = np.array(upper)

# The end-effector link index for the KUKA IIWA is typically the last joint/link.
# For the kuka_iiwa/model.urdf, it's usually link 6 (the 7th link/joint since we skip fixed joints).
# We use the index of the last joint ID in our list.
ee_link_index = joint_ids[ee_joint_number]

# ----------------------------------------------------
# 3) Reference trajectory (End-Effector Pose to Joints)
# ----------------------------------------------------
print("Creating reference trajectory via IK...")

# --------------------
# Interpolate the Poses
# --------------------
# Linearly interpolate position (XYZ)
ee_pos_ref = np.linspace(start_pos, goal_pos, T)

# Use SLERP (Spherical Linear Interpolation) for orientation (Quaternions)
ee_quat_ref = np.array(
    [p.getQuaternionSlerp(start_quat, goal_quat, float(i) / (T - 1)) for i in range(T)]
)

# ---------------------------------
# Compute Inverse Kinematics (IK)
# ---------------------------------
q_ref = []
# Get current joint states as initial guess for IK (optional but good practice)
initial_joint_states = [p.getJointState(robot, j)[0] for j in joint_ids]

for i in range(T):
    target_pos = ee_pos_ref[i]
    target_quat = ee_quat_ref[i]

    # Calculate IK. We limit the IK search space to the joint limits for better results.
    joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robot,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        targetOrientation=target_quat,
        lowerLimits=lower.tolist(),
        upperLimits=upper.tolist(),
        jointRanges=(upper - lower).tolist(),
        restPoses=initial_joint_states,
        maxNumIterations=1000,
        residualThreshold=1e-6,
    )
    # The result from IK includes all joints. We only need the moving joints.
    q_ref.append(np.array(joint_angles[:nq]))

q_ref = np.array(q_ref)  # (T, nq)

q_start = q_ref[0]
q_goal = q_ref[-1]


# Save to npz file
np.savez(ref_path, traj=q_ref)

# --- Playback ---
# input("Press Enter to playback trajectory...")
p.setRealTimeSimulation(
    0
)  # Ensure simulation is not running in real-time during playback

for q in q_ref:
    set_q(q)
    p.stepSimulation()
    time.sleep(dt)

# ---------------------------------
# 5) Objective & constraints
# ---------------------------------
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Hides the entire PyBullet GUI/controls
# p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


def optimization_method():
    print("Setting up optimization problem...")

    def hinge(x):
        # positive when inside margin → penalize; zero otherwise
        return np.maximum(0.0, safe_margin - x)

    def exponential(x):
        return np.exp(safe_margin - x)

    # # Joint bounds repeated for all T waypoints
    bounds = [(lower[k], upper[k]) for _ in range(T) for k in range(nq)]

    # # Optional: fix endpoints exactly (hard constraints) by clamping
    q0 = q_ref.copy()
    q0[0] = q_start
    q0[-1] = q_goal

    def get_ee_pose(q):
        """
        Computes the end-effector (EE) position (x, y, z) and orientation (quaternion)
        for a given joint configuration q using PyBullet's forward kinematics.
        """
        # 1. Set the robot's joint state
        for idx, j in enumerate(
            joint_ids[: ee_joint_number + 1]
        ):  # Only set up to the EE joint
            p.resetJointState(robot, j, q[idx])

        # 2. Get the link state (FK)
        link_state = p.getLinkState(robot, ee_link_index)

        # link_state[0] is the position, link_state[1] is the orientation (quaternion)
        pos = np.array(link_state[0])
        quat = np.array(link_state[1])

        return pos, quat

    def get_ee_poses_traj(Q):
        """Computes EE pose for all waypoints in a trajectory Q."""
        poses = [get_ee_pose(q) for q in Q]
        # Unpack into (T, 3) for positions and (T, 4) for quaternions
        pos_traj = np.array([p[0] for p in poses])
        quat_traj = np.array([p[1] for p in poses])
        return pos_traj, quat_traj

    # --- Custom loss function for quaternion orientation error ---
    def quat_error_sq(q1, q2):
        """Calculates squared error between two quaternions (q1, q2)."""
        # The error quaternion is q_err = q1 * inverse(q2)
        # The rotation angle error is proportional to the magnitude of the vector part (x,y,z) of q_err.
        # A simple, effective metric is 1 - |q1 . q2|^2. Or, use the angular distance (2*acos(|q1 . q2|)).
        # PyBullet's built-in distance is sometimes simpler:
        # dot_product = np.abs(np.sum(q1 * q2))
        # return 2.0 * np.arccos(dot_product)**2 # Squared angular distance

        # A robust and common alternative: using the difference vector
        q_diff = p.getDifferenceQuaternion(q1, q2)
        # The magnitude of the difference quaternion's vector part (x, y, z) is a measure of error
        # We use a simple squared distance which is a good proxy for small angles
        return np.sum(np.array(q_diff) ** 2)

    def objective(Q_flat):
        """
        The new total objective function.
        Q_flat is the flattened (T*nq) trajectory array.
        """
        Q = Q_flat.reshape(T, nq)

        if abs(w_track_pos) > 0 or abs(w_track_orient) > 0:
            # tracking cost for end effector pose
            # 1. Forward Kinematics: Convert joint angles to EE poses
            Q_pos_traj, Q_quat_traj = get_ee_poses_traj(Q)

            # 2. Tracking Cost
            # Position (XYZ) tracking cost (Squared Euclidean distance)
            pos_track = np.sum((Q_pos_traj - ee_pos_ref) ** 2)

            # Orientation (Quat) tracking cost (Sum of squared quaternion error)
            orient_track = np.sum(
                [quat_error_sq(Q_quat_traj[i], ee_quat_ref[i]) for i in range(T)]
            )

            ee_track_cost = w_track_pos * pos_track + w_track_orient * orient_track
        else:
            ee_track_cost = 0.0

        joint_track = np.sum((Q - q_ref) ** 2)

        # smoothness on velocities
        dQ = np.diff(Q, axis=0) / dt
        smooth = np.sum(dQ ** 2)

        # collision loss: sum over waypoints
        dists = min_clearance_traj(Q)
        # coll = np.sum(exponential(dists)**2)
        coll = np.sum(hinge(dists) ** 2)

        cost = (
            w_track * joint_track
            + w_smooth * smooth
            + w_collision * coll
            + ee_track_cost
        )

        # 3. Total Cost
        return cost

    # ---------------------------------
    # 6) Optimize (L-BFGS-B, finite diff)
    # ---------------------------------
    print("Optimizing trajectory...")
    MAX_ITER = 200
    FTOL = 1e-6  # 1e-6
    pbar = tqdm(total=MAX_ITER, desc="Optimization Progress", unit="iters")

    def optimization_callback(x_k):
        # Since we don't get the iteration count, we just advance the bar by 1
        pbar.update(1)

    # method = L-BFGS-B # took 2 mins for 15 iters
    # method = SLSQP # took 12 mins and 200 iters
    res = minimize(
        objective,
        x0=q0.ravel(),
        method=args.planner_method,
        bounds=bounds,
        options=dict(maxiter=MAX_ITER, ftol=FTOL),
        callback=optimization_callback,
    )

    Q_opt = res.x.reshape(T, nq)
    print("success:", res.success, " final cost:", res.fun)

    print("Saving optimized trajectory to q_opt.npz...")
    np.savez(opt_path, traj=Q_opt)
    return Q_opt


if args.use_cache and os.path.exists(opt_path):
    Q_opt = np.load(opt_path)["traj"]
else:
    if args.use_cache:
        print(
            f"Warning: Cached trajectory not found at {opt_path}. Running optimization instead."
        )
    Q_opt = optimization_method()

# ---------------------------------
# 7) Playback in PyBullet (GUI optional)
# ---------------------------------
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

print("Playing back optimized trajectory...")
# input("Press Enter to playback trajectory...")
# If you want to visualize: switch to GUI by reconnecting with p.GUI earlier.
for q in Q_opt:
    set_q(q)
    p.stepSimulation()
    time.sleep(dt)

print("Playback the trajectory with:")
print(
    "python test_playback_traj.py --scenario {args.scenario} --planner_method {args.planner_method} --type opt"
)
