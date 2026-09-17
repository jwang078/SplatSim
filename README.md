# SplatSim
SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

[Project Page](https://splatsim.github.io) | [Arxiv](https://arxiv.org/abs/2409.10161)



This repository contains the code for the paper "SplatSim". 

## Installation

You'll need conda and an NVIDIA GPU. We've tested on Python 3.12 / CUDA 12.8.

```bash
git clone --recursive git@github.com:jwang078/SplatSim.git ~/code/SplatSim
cd ~/code/SplatSim
conda env create -f environment.yml
conda activate splatsim
./install.sh
```

`install.sh` installs the PyTorch CUDA build, the
Python dependencies, and the source-built submodules (compiling the CUDA
extensions takes a few minutes). It also patches a couple of dependencies that don't build out of the box — see
[Things `install.sh` already handles](#things-installsh-already-handles) below
if you're curious. When it finishes, it import-checks everything and tells you
if anything is off.

If you'd like to drive a physical xArm, add the hardware extras afterwards:
```bash
pip install -e '.[hardware]'
```

### LeRobot

The simulation server runs on its own, but you'll want our fork of LeRobot for
anything involving datasets or policies — recording demos, replaying an
eval-benchmark episode, or driving the sim with a trained policy. Clone it next
to SplatSim:

```bash
git clone git@github.com:jwang078/lerobot.git ~/code/lerobot
pip install -e '~/code/lerobot[dataset]'
```

The `[dataset]` extra is what the recording and replay code needs, so please
keep it.

`install.sh` does this step for you if it finds `../lerobot`. If your checkout
lives somewhere else, point it there with `LEROBOT_DIR=/path/to/lerobot
./install.sh`, or `SKIP_LEROBOT=true ./install.sh` to leave it out for now.

### Things `install.sh` already handles

You don't need to do anything about these — they're listed so the output
doesn't surprise you.

- **`ghalton` and `evdev`** both fail to compile with the conda toolchain, so
  `install.sh` builds ghalton from the patched `submodules/ghalton` fork and
  evdev with the system gcc. This is step 3a in its output.
- **`diff_gaussian_rasterization`** needs two small edits to the upstream
  source (a missing `#include <cstdint>` and a closer near-plane cull so the
  wrist camera can see up close). `install.sh` applies them right before
  compiling, so if `git status` shows those two files modified inside the
  submodule, that's expected.

### If something goes wrong

- **`ModuleNotFoundError: simple_knn`** (or an empty `submodules/` folder) —
  the clone missed its submodules. Run
  `git submodule update --init --recursive`, then `./install.sh` again. If a
  submodule folder has nothing but a `.git` inside even after that, check it
  out directly with `git -C submodules/<name> checkout -f HEAD`.
- **`nvcc: command not found`** — the conda env isn't active. A quick
  `conda activate splatsim` fixes it.
- **Please don't upgrade torch.** It's pinned to 2.11.0+cu128 on purpose: the
  CUDA extensions are compiled against it, and video dataloading gets several
  times slower on other builds. The same goes for the CUDA 12.8 toolchain.
- **Noisy `git status` after installing** — the submodule builds leave a few
  artifacts behind. Harmless, but if you'd like them hidden:
  ```bash
  echo '*.egg-info' >> .git/modules/submodules/ghalton/info/exclude
  echo '*.egg-info' >> .git/modules/submodules/gello_software/modules/third_party/DynamixelSDK/info/exclude
  ```

## Running the rendering code 
### 1. Download the colmap and gaussian-splatting models from the below links:
- [colmap (test_data)](https://drive.google.com/file/d/14D3fFtaPX4GBe9dSJLKAIvUYlgK7fUxS/view?usp=sharing)
- [gaussian-splats (output)](https://drive.google.com/file/d/1rAUkf7l2ZZqG1Bm3ih6cAO5HCd9dSTO-/view?usp=sharing)
- [trajectories (bc_data/gello)](https://drive.google.com/file/d/1NhSBNYMi51hETAspk6vN7F-Ih1134_lt/view?usp=sharing)

`test_data` is the folder name of the output of colmap. `output` is the folder name of the output of gaussian splat generation. `bc_data/gello` is the folder name of the demo trajectories recorded by one of the scripts in this repo.

Assume below that these files are stored under:

- test_data: /home/yourusername/data/test_data
- output: /home/yourusername/data/output
- bc_data/gello: /home/yourusername/data/bc_data/gello

### 2. Configure the configs to match your folder directory structure

#### Open `configs/object_configs/objects.yaml`. The data you downloaded in step 1 is for the robot `robot_iphone`.

Modify `robot_iphone` as below:
- source_path: /home/yourusername/data/test_data/robot_iphone # Path to a folder you downloaded
- model_path: /home/yourusername/data/output/robot_iphone # Path to a folder you downloaded

Modify all `ply_path` attributes to point to `/home/yourusername/data/output/...`, for example for `plastic_apple`

#### Open `configs/folder_configs.yaml`

Modify as follows:
- traj_folder: /home/yourusername/data/bc_data/gello

### 3. Run the rendering script:

Launch the robot server which includes an apple and a plate:
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone
```

In another terminal tab, launch a node that will send the recorded trajectories in `/home/yourusername/data/bc_data/gello` to the server so that it will be rendered:
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

A window should pop up that is a rendering of the robot in the pybullet simulation. If you drag the end effector of the robot around in pybullet, it should be reflected in the render.

The rendered images for trajectory 0 are saved in `{traj_folder}/0/images_1`, for example `/home/yourusername/data/bc_data/gello/0/images_1`.

Congrats! Your static splat is now being simulated! 🚀

# Optional

## Adding a new robot

What if you want to simulate a different robot than the one downloaded above? Or with a new background?

### Create gaussian splat

First, create a gaussian splat of your robot within the scene. Record the joint angles of the robot in your static scene
<details>
<summary>Tips for creating a good gaussian splat of the robot</summary>

- Set the pose of your robot to be in an easy-to-describe state. For example, all exactly 90 degree angles. This is for easier calibration later

- Don't put any objects within the robot's rectangular bounding box. The pipeline currently uses a simple segmentation technique with a rectangular bounding box

- 1-2 minute landscape video on a smartphone or similar

- Capture the scene from all angles

- Make sure to capture areas of finer detail, for example the robot gripper, by sometimes moving the camera closer to it

- Make sure the background is textured; use posters to cover plain walls. This helps when recovering camera poses (structure from motion / colmap)

- Diffuse lighting is best (ex: white paper over a strong light), or else the result will have glares, which will not be updated by the simulation when robot joints are moved
</details>

Train the gaussian splat with the [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo, which is in `submodules/gausisan-splatting-wrapper/gaussian_splatting` in this repo. Software like Polycam unfortunately does not give you the structure from motion outputs that are used for generating camera views in this repo.
<details>
<summary> Summary of how to train splat </summary>

- Create a folder structure for example at `~/data/your_robot_name/input`. Put all images into the `input` folder. For example, you can convert your video to images with ffmpeg (ex: `ffmpeg -i my_video.MOV -qscale:v 1 output_%04d.jpeg`).

- Recover camera poses with structure from motion / colmap. First, install colmap then run the script. This will populate `~/data/your_robot_name` with camera info
```bash
conda install colmap
python submodules/gaussian-splatting-wrapper/gaussian_splatting/convert.py -s ~/data/your_robot_name
```

- Train gaussian splat. This will create a folder that looks like `./output/258f657d-c` from where you run this command, and it contains the `output/258f657-c/point_cloud/iteration_30000/point_cloud.ply` file, which is the gaussian splat.
```bash
python submodules/gaussian-splatting-wrapper/gaussian_splatting/train.py -s ~/data/your_robot_name
```
</details>

### Align the simulator and the splat

#### Configs

Add `your_robot_name` to `configs/object_configs/objects.yaml`. First, copy-paste the attributes from `robot_iphone`.

- Set `model_path` to the folder output of the gaussian splat training (ex: `~/.../.../output/258f657d-c`)

- Set `source_path` to the folder with the image data and colmap outputs (ex: `~/data/your_robot_name/input`)

- Set `joint_states` (radians) to the joint angles that the robot had when the gaussian splat data was collected. There might be an extra 0 preceding the base joint (ex: [0, 0, 1.57, ...])

- If you have a different URDF, change `urdf_path`. Note that `robot_iphone` is a UR5 robot.

#### Convert URDF to point cloud

Run
```bash
python scripts/articulated_robot_pipeline.py --robot_name your_robot_name
```

Verify that the first point cloud visualization has the same joint poses as your robot had in the splat. If not, adjust `joint_states`. Ignore the second visualization for now.

The point cloud is outputted in `data/pcds_path/your_robot_name_pcd.ply`.

#### Align robot coordinate frames in sim and in splat

Download CloudCompare, which visualizes point clouds. 

Open both the URDF point cloud `data/pcds_path/your_robot_name_pcd.ply` and the gaussian splat `output/.../point_cloud/point_cloud/iteration_30000/point_cloud.ply`. The goal is to apply transformations (rotation/translation/scale) *to your splat* such that the robot arm matches between the sim and splat, then you can copy that transformation to a config file. Don't apply transformations to the simulated robot arm.

<details>
<summary> Tips and tricks with CloudCompare </summary>

- To see rgb colors on your trained splat, download `3dgsconverter` and run `3dgsconverter -i point_cloud.ply -o output_cloudcompare.ply -f cc --rgb`. After importing `output_cloudcompare.ply` to CloudCompare, select it in the top left sidebar, then in the bottom left sidebar, set `Properties > Colors` to `RGB`

- Use the Segment tool (scissor in top bar) to crop out the table and other objects, leaving only the robot. Also crop out any wires (which wouldn't be present in the simulated robot). Note: select the point cloud you want to segment on the left toolbar before you click Segment, or else it will try to segment the wrong point cloud, thus making no changes. The points you segmented out re-appear after you save the segmentation because they are now in another group (in the left toolbar). You can deselect it to stop visualizing it. First select create the selection polygon with left clicks, right click to finish your polygon, click either Segment In or Segment Out, then if you want to keep on iterating on this, find a new angle then press the unpause button to start segmenting again. When you're done, press the green checkmark.

- The fastest way to align the robots seems to doing ICP w/o scale adjustment, manual adjustment, then ICP with scale adjustment. To start ICP, ctrl+click both point clouds, then `Tools > Registration > Fine Registration (ICP)`. Set the simulated robot to be the reference and the splat to be the aligned. Uncheck `adjust scale` for this first pass of ICP. Then, manually fix errors with `Translate/Rotate` (click on point cloud in left sidebar then click on the button at the top toolbar). Then run ICP again but with `adjust scaling` checked. If the URDF is fundamentally different from your real robot, prioritize aligning the robot's end effector to mitigate cascading error when doing forward kinematics in the sim.

- You can double-check alignment by setting the floor as visible and seeing if the floor planes are aligned, or by looking at all orthographic views (left toolbar)
</details>

The splat-to-simulator transformation is in `Transformation History` (scroll to the bottom of Properties in the left sidebar). Copy-paste it to `configs/object_configs/objects.yaml` under your_robot_name > transformation > matrix, while fitting the yaml format

#### Double check calibration

Run this script below again. The last visualization that compares the two point clouds should have colors and coordinate frames lined up.

```bash
python scripts/articulated_robot_pipeline.py --robot_name your_robot_name
```

Note: `urdf_bbox_adjustment` can handle cases where your physical robot has an additional attachment compared to the URDF. You can check its effects in the same final visualization

#### Your custom robot can now follow the same recorded joint state trajectories!

Launch the simulation server
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name your_robot_name
```

Set the robot to follow the recorded trajectories.
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

## Generating new trajectories

The trajectories are stored at the folder specified in `configs/trajectory_configs.yaml` (`trajectory_folder`). New trajectories are added as larger folder id numbers, and then the replay_trajectory agent starts playing trajectories from folder 0 and up.

For new trajectories, clear out the trajectory folder. Either change `trajectory_folder` or move the previously generated trajectories to another location.

The provided demos are for placing an apple on a plate. If instead you wanted to generate demos for placing a banana on a plate, run

```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_banana --robot_name your_robot_name
```

This populates the `trajectory_folder`. Then, to replay these new trajectories, do the same visualization setup but with this banana on plate environment:

Launch the simulation server
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_banana_interactive --robot_name your_robot_name
```

Set the robot to follow the recorded trajectories.
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

You can use `splatsim/robots/sim_robot_pybullet_object_on_plate.py` as a template for configuring custom environments.

## Grape vine environment

A UR5 reaching toward a grape bunch on a scanned vine, rendered inside the vine
splat. Launch it with:

```bash
python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive \
    --robot_port 6003 --wrist_cam_ver=2 --control_gui
```

You do not pass `--robot_name`: this variant already knows it renders against
`robot_iphone_w_engine_curtain`. `--wrist_cam_ver=2` picks a fisheye
calibration that ships in the code, so it needs nothing on disk.

### 1. Download the vine assets

- [vine assets (vine_assets.tar.gz)](TODO-DRIVE-LINK) — about 900 MB

Unpack it into the repo. It contains a `data/` folder that merges into the
existing one, and the configs already point at those paths, so there's nothing
to edit:

```bash
tar xzf vine_assets.tar.gz -C ~/code/SplatSim
```

That gives you:

```
data/output/robot_iphone_w_engine_curtain/   gaussian splat of the robot
data/output/vine_scene/                      gaussian splat of the vine highbay scene
data/test_data/vine_scene/                   structure-from-motion output for the vine scene
data/vine_seg/vine_and_trellis/              vine collision mesh, grape targets, soft-cost field
```

### 2. Launch

Run the command at the top of this section. You should see the PyBullet
window, the control GUI, and a splat render with the arm in front of the vine.

<details>
<summary>What each downloaded piece is for</summary>

- `data/output/*/point_cloud/iteration_30000/point_cloud.ply` — the trained
  splats. The robot's is articulated as the arm moves; the vine scene's is the
  rendered background. `grapes_only.ply` alongside it is the segmented fruit,
  used by `scripts/tune_goal_pose.py`.
- `data/test_data/vine_scene/sparse/0/` — camera poses from an
  [hloc](https://github.com/cvg/Hierarchical-Localization) run with
  disk+lightglue. The base camera view is picked from these. This isn't the
  COLMAP `convert.py` route described under *Adding a new robot*, so don't
  expect to regenerate it that way.
- `data/test_data/vine_scene/images/` — the source frames; the server loads
  the one the base camera corresponds to.
- `data/vine_seg/vine_and_trellis/vine_and_trellis.urdf` + `_collision.obj` —
  the hard trunk and trellis, loaded as a PyBullet obstacle. Pre-baked in sim
  frame, so there's no splat-to-sim transform to calibrate.
- `data/vine_seg/vine_and_trellis/grape_targets_manual.json` — the bunch
  centers the task aims at. Hand-annotated, because colour segmentation can't
  see green fruit.
- `data/vine_seg/vine_and_trellis/vine_and_trellis_cost_field_sim.npz` — the
  soft-cost field the RRT planner trades off against, so foliage is a cost
  rather than a wall.

These come from `scripts/segment_vine_splat.py`, `scripts/build_vine_collision.py`
and `scripts/mark_grape_targets.py` if you ever need to rebuild them. If you keep
your splats somewhere else, the three entries to repoint in
`configs/object_configs/objects.yaml` are `robot_iphone_w_engine_curtain`,
`vine_scene` and `vine_and_trellis`.
</details>

### Retargeting the task

The knobs are class attributes on `VineGrapeReachPybulletRobotServer` in
`splatsim/robots/sim_robot_pybullet_vine.py` — `TARGET_BUNCH_INDEX` (which
bunch, largest first), `GRAPE_STANDOFF_M` (how close), and the
`GRIPPER_CAMERA_UP_WORLD` / `CAMERA_FORWARD_AXIS` pair that decides how the
wrist camera frames the fruit. Tune the goal pose live with
`scripts/tune_goal_pose.py` rather than by guessing; the roll in particular is
specified as an absolute world direction because a relative offset is measured
from an arbitrary IK branch.

If the log fills with `vine env: robot-derived grape goal failed — falling back
to the static task target`, the goal solver could not find a collision-free IK
that satisfies `GRIPPER_CAMERA_UP_WORLD` within its tolerance. The sim still
runs, on the static pose from the config. Widen the tolerances or retune with
the script above.

## GELLO integration

[GELLO](https://github.com/wuphilipp/gello_software) can be used as a teleoperation system for collecting demos. Please refer to the GELLO repo for hardware setup and general connectivity. Connect the GELLO via USB.

Start the simulation server in interactive mode:
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name your_robot_name
```

Set the robot to follow GELLO commands and show the save interface GUI:
```bash
python scripts/run_env_sim.py --agent gello --use-save-interface
```

If you move your GELLO, it the simulated robot should move, as well.

In the gray save interface window, you can start and stop recording a new demonstration. Click on the window and press `s` to start recording (the window will turn green). When  you're done recording, click on the window and press `q` (the window will turn red). You can do multiple `s` and `q` recordings.

Play back your recordings with the same command as before (note that new recordings are played last; you can delete or move files out of `trajectory_folder` to view your newly recorded trajectories)
```bash
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

## Visualizing a trained policy in the Splat Sim

TODO for diffusion policy

## A list of TODOs

- [x] ~~Proper instructions for installation. Current instructions might not work.~~
    - [x] Installation instructions should work now.
- [x] ~~Add links to pretrain gaussian-splats and trajectories, so that people can run rendering script.~~
    - [x] Links to pretrain gaussian-splats, colmap and trajectories are added.
- [x] ~~Create a new file for rendering robot and objects, without hardcoding the segmentation and shifting everything to KNN based segmentation.~~
- [x] Clean up the splat folder for only keeping necessary files and easy creation of KNN based segmentation for robots.
- [x] Documentation for the codebase.
- [x] Adding new robots (in sim or any other environment).
- [x] Instructions to generate a trajectory and render it. 
    - [ ] Trajectory format should be specified properly.
- [x] Clean up the gello folder and only keep files that are necessary. 
 