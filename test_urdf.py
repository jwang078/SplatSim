import pybullet as p
import pybullet_data

import argparse

parser = argparse.ArgumentParser(description='Load and visualize a URDF file in PyBullet.')
parser.add_argument('urdf_path', type=str, help='Path to the URDF file to be loaded.')
args = parser.parse_args()

# Start PyBullet in GUI mode
p.connect(p.GUI)

# Set up the simulation environment
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane
p.loadURDF("plane.urdf")

import urdf_models.models_data as md
models_lib = md.model_lib()

# Load your URDF file
# Replace 'your_robot.urdf' with the path to your actual URDF file
# Make sure the file is in a directory that PyBullet can find, or provide the full path
robot_id = p.loadURDF(
    # models_lib["plastic_apple"],
    args.urdf_path,
    [0, 0, 0],
    [0, 0, 0, 1],
    globalScaling=1,
    useFixedBase=True,
)

# You can adjust the robot's initial position if it appears to be falling through the plane
# p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], [0, 0, 0, 1])

# Run the simulation loop
# try:
#     while True:
#         p.stepSimulation()
#         time.sleep(1./240.) # PyBullet runs at 240Hz by default, so we can sleep to match
# except p.error:
#     # This will catch the error when the GUI is closed
#     pass

input()

p.disconnect()