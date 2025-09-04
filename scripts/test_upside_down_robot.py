import pybullet as p
import time
import pybullet_data
import math

# --- Main Setup ---
# 1. Connect to the physics server
# p.GUI allows you to see the simulation.
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Used for loading default URDFs

# 2. Configure the simulation
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0) # We will step the simulation manually

# --- Create the Environment ---
# 1. Load the ground plane and a table
planeId = p.loadURDF("plane.urdf")
# tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# --- Robot Setup ---
# !! IMPORTANT !!: Replace this with the actual path to your robot's URDF file.
# For this example, we'll use the KUKA arm that comes with PyBullet.
import yaml
with open("configs/object_configs/objects.yaml", 'r') as f:
    object_config = yaml.safe_load(f)
urdf_path = object_config['robot_iphone']['urdf_path'][0]

# 1. Define the robot's initial position and orientation
# The robot will be placed high up in the Z-axis.
initial_robot_pos = [0, 0, 1.0] 
# To make the robot hang upside down, we rotate it 180 degrees (pi radians) around the X-axis.
upside_down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

# 2. Load the robot from the URDF file
# useFixedBase=True makes the robot hang from its base, as if attached to the ceiling.
robotId = p.loadURDF(
    urdf_path,
    initial_robot_pos,
    upside_down_orientation,
    useFixedBase=True
)
print(f"Robot URDF '{urdf_path}' loaded with ID: {robotId}")

use_ee_drag_slider = p.addUserDebugParameter(
    paramName="Use EE Drag",  # The text that appears in the GUI
    rangeMin=0,               # The 'Off' state
    rangeMax=1,               # The 'On' state
    startValue=1              # Start in the 'On' state
)
prev_use_ee_drag_state = 1



# --- Rectangular Prism (Box) Setup ---
# 1. Define the prism's properties
# TODO put real prism dimensions
# height 13.5 in, length 15in, width 12 in
# 0.3429m, 0.381m, 0.3048m
prism_half_extents = [0.381/2, 0.3048/2, 0.3429/2] # Half of length, width, height
prism_mass = 1
initial_prism_pos_xy = [0.0, 0.0]

# 2. Calculate the correct Z position to be on top of the table
# table_aabb = p.getAABB(tableId)
# table_top_z = table_aabb[1][2] # The maximum Z coordinate of the table's bounding box
# prism_z = table_top_z + prism_half_extents[2] # Place prism's bottom on the table top
prism_z = prism_half_extents[2] # Place prism's bottom on the table top
initial_prism_pos = [initial_prism_pos_xy[0], initial_prism_pos_xy[1], prism_z]

# 3. Create the prism's shape
visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=prism_half_extents,
    rgbaColor=[0.8, 0.2, 0.2, 1] # Red color
)
collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=prism_half_extents
)

# 4. Create the prism object in the world
prismId = p.createMultiBody(
    baseMass=prism_mass,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=initial_prism_pos
)
print(f"Prism created with ID: {prismId}")

# --- Add GUI Sliders for Interactive Control ---
# These sliders will appear in the PyBullet GUI window.
robot_x_slider = p.addUserDebugParameter("Robot X", -1.5, 1.5, initial_robot_pos[0])
robot_y_slider = p.addUserDebugParameter("Robot Y", -1.5, 1.5, initial_robot_pos[1])
robot_z_slider = p.addUserDebugParameter("Robot Z", 0.0, 3.0, initial_robot_pos[2])

prism_x_slider = p.addUserDebugParameter("Prism X", -0.6, 0.6, initial_prism_pos[0])
prism_y_slider = p.addUserDebugParameter("Prism Y", -0.6, 0.6, initial_prism_pos[1])

# This section inspects the robot and creates a slider for each movable joint.
joint_sliders = []
num_joints = p.getNumJoints(robotId)

joint_lower_limits = []
joint_upper_limits = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_id = joint_info[0]
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    joint_lower_limit = joint_info[8]
    joint_upper_limit = joint_info[9]

    # We only want to control 'revolute' (rotating) or 'prismatic' (sliding) joints.
    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
        # Create a slider for this joint
        slider_id = p.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=joint_lower_limit,
            rangeMax=joint_upper_limit,
            startValue=0 # Start at a neutral position
        )
        # Store the joint ID and slider ID for later use
        joint_sliders.append((joint_id, slider_id))
        joint_lower_limits.append(joint_lower_limit)
        joint_upper_limits.append(joint_upper_limit)

print(f"\nCreated sliders for {len(joint_sliders)} controllable joints.")


# --- Main Simulation Loop ---
# Facing Down (Z of EE -> -Z world)
down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
# Facing Up (Z of EE -> +Z world)
up_orn = p.getQuaternionFromEuler([0, 0, 0])  # identity
# Facing Forward (Z of EE -> +X world)
forward_orn = p.getQuaternionFromEuler([0, -math.pi/2, 0])
# Facing Backward (Z of EE -> -X world)
backward_orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
# Facing Left (Z of EE -> +Y world)
left_orn = p.getQuaternionFromEuler([math.pi/2, 0, 0])
# Facing Right (Z of EE -> -Y world)
right_orn = p.getQuaternionFromEuler([-math.pi/2, 0, 0])

ee_points_to_go_to = {
    "top of engine": {
        "pos": (prism_half_extents[0]*0, prism_half_extents[1]*0, prism_half_extents[2]*2),
        "orn": down_orn,
    },
    "close side": {
        "pos": (prism_half_extents[0]*-1, prism_half_extents[1]*0, prism_half_extents[2]*1),
        "orn": forward_orn,
    },
    "far side": {
        "pos": (prism_half_extents[0]*1, prism_half_extents[1]*0, prism_half_extents[2]*1),
        "orn": backward_orn,
    },
    "left side": {
        "pos": (prism_half_extents[0]*0, prism_half_extents[1]*-1, prism_half_extents[2]*1),
        "orn": right_orn,
    },
    "right side": {
        "pos": (prism_half_extents[0]*0, prism_half_extents[1]*1, prism_half_extents[2]*1),
        "orn": left_orn,
    },
    "close table": {
        "pos": (prism_half_extents[0]*-1.2, prism_half_extents[1]*0, prism_half_extents[2]*0),
        "orn": down_orn,
    },
    "far table": {
        "pos": (prism_half_extents[0]*1.2, prism_half_extents[1]*0, prism_half_extents[2]*0),
        "orn": down_orn,
    },
    "left table": {
        "pos": (prism_half_extents[0]*0, prism_half_extents[1]*-1.2, prism_half_extents[2]*0),
        "orn": down_orn,
    },
    "right table": {
        "pos": (prism_half_extents[0]*0, prism_half_extents[1]*1.2, prism_half_extents[2]*0),
        "orn": down_orn,
    },
}
ee_points_to_go_to_keys = list(ee_points_to_go_to.keys())
ee_points_to_go_to_index = 0
use_interactive_mode = True
for i in range(p.getNumJoints(robotId)):
    info = p.getJointInfo(robotId, i)
    joint_id = info[0]
    joint_name = info[1].decode("utf-8")
    if joint_name == "ee_fixed_joint":
        ee_id = joint_id
        break

import numpy as np
try:
    while True:
        if not use_interactive_mode:
            input("Press enter to continue")
            print()

            joint_positions = []
            for joint_id, slider_id in joint_sliders:
                joint_state = p.getJointState(robotId, joint_id)
                current_position = joint_state[0]
                joint_positions.append(current_position)

            target_name = ee_points_to_go_to_keys[ee_points_to_go_to_index % len(ee_points_to_go_to_keys)]

            target_pos = ee_points_to_go_to[target_name]['pos']
            target_orn = ee_points_to_go_to[target_name]['orn']

            joint_angles = p.calculateInverseKinematics(
                robotId, ee_id,
                target_pos, target_orn,
                lowerLimits=[joint_positions[k] - np.pi for k in range(6)],
                upperLimits=[joint_positions[k] + np.pi for k in range(6)],
                jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
                restPoses=[0* np.pi, -0.5* np.pi, 0.5* np.pi, -0.5* np.pi, -0.5* np.pi, 0]
            )
            print(f"Going to target {target_name} with EE pos {target_pos} / EE orn {target_orn}")
            print(f"Joint angles: {joint_angles[:6]}")

            for joint_id, slider_id in joint_sliders:
                if joint_id >= 6:
                    break
                target_position = joint_angles[joint_id]
                
                # Command the joint to move to the target position
                p.resetJointState(
                    robotId, joint_id, target_position
                )
            ee_points_to_go_to_index += 1
            
        else:
            use_ee_drag_state = p.readUserDebugParameter(use_ee_drag_slider)

            # 1. Read the current values from the GUI sliders
            robot_x = p.readUserDebugParameter(robot_x_slider)
            robot_y = p.readUserDebugParameter(robot_y_slider)
            robot_z = p.readUserDebugParameter(robot_z_slider)
            
            prism_x = p.readUserDebugParameter(prism_x_slider)
            prism_y = p.readUserDebugParameter(prism_y_slider)

            # 2. Update the robot's base position using the slider values
            # The orientation remains fixed to keep it upside down.
            p.resetBasePositionAndOrientation(
                robotId,
                [robot_x, robot_y, robot_z],
                upside_down_orientation
            )

            # 3. Update the prism's position using the slider values
            # The Z position and orientation are kept constant so it stays flat on the table.
            p.resetBasePositionAndOrientation(
                prismId,
                [prism_x, prism_y, prism_z],
                p.getQuaternionFromEuler([0, 0, 0])
            )

            if use_ee_drag_state > 0.5:
                # no joint motor control
                # Set sliders to current joint positions
                for joint_id, slider_id in joint_sliders:
                    p.setJointMotorControl2(
                        bodyUniqueId=robotId,
                        jointIndex=joint_id,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=500
                    )
                    joint_state = p.getJointState(robotId, joint_id)
                    current_position = joint_state[0]
                prev_use_ee_drag_state = 1
            else:
                # 4. Control the robot's joints using sliders
                # If we just switched to using the sliders, update the sliders to match current joint positions
                if prev_use_ee_drag_state == 1:
                    new_joint_sliders = []
                    for joint_id, slider_id in joint_sliders:
                        joint_state = p.getJointState(robotId, joint_id)
                        current_position = joint_state[0]
                        # Remove the previous slider

                        # Create a new slider initialized to the current joint position
                        new_slider_id = p.addUserDebugParameter(
                            paramName=p.getJointInfo(robotId, joint_id)[1].decode('utf-8') + "1",
                            rangeMin=p.getJointInfo(robotId, joint_id)[8],
                            rangeMax=p.getJointInfo(robotId, joint_id)[9],
                            startValue=current_position
                        )
                        new_joint_sliders.append((joint_id, new_slider_id))
                    joint_sliders = new_joint_sliders

                for joint_id, slider_id in joint_sliders:
                    # Read the target position for this joint from its slider
                    target_position = p.readUserDebugParameter(slider_id)
                    
                    # Command the joint to move to the target position
                    p.setJointMotorControl2(
                        bodyUniqueId=robotId,
                        jointIndex=joint_id,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_position,
                        # force is the maximum motor force used to reach the target position
                        force=500 
                    )
                prev_use_ee_drag_state = 0
            
        
        # 5. Step the simulation forward
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    # --- Cleanup ---
    p.disconnect()