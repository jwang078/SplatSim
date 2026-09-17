# SplatSim
SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

[Project Page](https://splatsim.github.io) | [Arxiv](https://arxiv.org/abs/2409.10161)



This repository contains the code for the paper "SplatSim". 

## Installation

You'll need conda and an NVIDIA GPU. We've tested on Python 3.12 / CUDA 12.8.

```bash
git clone --recursive git@github.com:jwang078/SplatSim.git ~/code/SplatSim
cd ~/code/SplatSim
./install.sh
conda activate splatsim
```

`install.sh` creates the `splatsim` conda env from `environment.yml`, then
installs the PyTorch CUDA build, the Python dependencies, and the source-built
submodules into it (compiling the CUDA extensions takes a few minutes). If a
`splatsim` env already exists it checks it first and asks whether to install
into it or recreate it from scratch — nothing is touched before you answer.
It also patches a couple of dependencies that don't build out of the box — see
[Things `install.sh` already handles](#things-installsh-already-handles) below
if you're curious. When it finishes, it import-checks everything and tells you
if anything is off.

If you'd like to drive a physical xArm, add the hardware extras afterwards:
```bash
pip install -e '.[hardware]'
```

### LeRobot

`install.sh` also clones and installs
[our fork of LeRobot](https://github.com/jwang078/lerobot) into
`external/lerobot` (editable, ignored by git). SplatSim reads and writes its
datasets through it, and it's where the training and DAgger code lives, so
the two are meant to be worked on together. If you already have a checkout,
the installer uses a sibling `../lerobot` automatically, or point it anywhere
with `LEROBOT_DIR=/path/to/lerobot ./install.sh`.

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
- **`nvcc: command not found`** when running things later — the conda env
  isn't active. A quick `conda activate splatsim` fixes it.
- **Please don't upgrade torch.** It's pinned to 2.11.0+cu128 on purpose: the
  CUDA extensions are compiled against it, and video dataloading gets several
  times slower on other builds. The same goes for the CUDA 12.8 toolchain.
- **Noisy `git status` after installing** — the submodule builds leave a few
  artifacts behind. Harmless, but if you'd like them hidden:
  ```bash
  echo '*.egg-info' >> .git/modules/submodules/ghalton/info/exclude
  echo '*.egg-info' >> .git/modules/submodules/gello_software/modules/third_party/DynamixelSDK/info/exclude
  ```

## Running a simulation

A simulation is a robot scan plus a scene scan, each a folder under
`data/scenes/` (see *Data layout* below). Two are available to download and
make a working example: the grape vine, and the UR5 scanned in the engine
scene that it uses as its robot.

### 1. Download the example scenes

One tarball per scene, named after the folder it unpacks to:

- [vine_scene.tar.gz](https://drive.google.com/file/d/1fzErjuOOu85abCVYgvtSJQXf6HzEpZmO/view?usp=drive_link) — the grape vine, scanned in the highbay, ~700 MB
- [robot_iphone_w_engine_curtain.tar.gz](https://drive.google.com/file/d/15OvhOXCvdgjNVPUZB1W28DLXjG9HLpJX/view?usp=drive_link) — the UR5 with its wrist camera, scanned in the engine scene, ~200 MB

Unpack both into the repo's `data/scenes/` folder. Each carries its own
`scene.yaml`, so there's nothing to edit:

```bash
tar xzf vine_scene.tar.gz -C /path/to/SplatSim/data/scenes
tar xzf robot_iphone_w_engine_curtain.tar.gz -C /path/to/SplatSim/data/scenes
```

That gives you:

```
data/scenes/vine_scene/splat/                                  gaussian splat of the vine highbay scene
data/scenes/vine_scene/sfm/                                    structure-from-motion output for it
data/scenes/vine_scene/segmentations/vine_and_trellis/         collision mesh, grape targets, soft-cost field
data/scenes/robot_iphone_w_engine_curtain/splat/               gaussian splat of the robot
```

### 2. Launch

Each environment is a `--robot` variant of `launch_nodes.py`; the vine one is:

```bash
python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive \
    --robot_port 6003 --wrist_cam_ver=2 --control_gui
```

You should see the PyBullet window, the control GUI, and a splat render with
the arm in front of the vine. `--wrist_cam_ver=2` picks a fisheye calibration
that ships in the code; `--robot_name` isn't needed, the variant knows its
robot.

The apple-on-plate demo from the paper (the `test_data` / `output` /
`bc_data/gello` downloads) is not maintained here — use the
[original SplatSim repository](https://github.com/qureshinomaan/SplatSim)
for that.

# Optional

## Data layout

Everything the simulator loads lives under `data/`, one folder per thing,
described by a yaml next to the files: scenes in `data/scenes/<scene>/scene.yaml`,
robots in `data/robots/<robot>/robot.yaml` (a scanned robot can also live
under `data/scenes/`). A scene folder looks like this:

```
data/scenes/<scene>/
    scene.yaml                       transformation, aabb, model_path, source_path, ...
    splat/                           gaussian-splatting output
    sfm/                             COLMAP / hloc output
    segmentations/<build>/
        scene.yaml                   ply_path, urdf_path, collision_frame
        <build>.urdf, <build>_collision.obj, cost field, grape targets, ...
```

- The folder tree is the config: every `scene.yaml` is one object, named
  after its folder, referenced by that flat name in code
  (`splat_name="vine_and_trellis"`). Paths inside it are relative to the
  file; a nested `scene.yaml` inherits from the one above it.
- `collision_frame: splat` means the build's collision mesh, cost field and
  grape targets are in the scan's frame and get the scan's `transformation`
  at load; `sim` means they were baked already. `scripts/build_vine_collision.py`
  sets it.
- `scene.yaml` files are tracked in git; the data next to them is not.

To add a scene: make `data/scenes/<scene>/` with `splat/` and `sfm/`, copy
`data/scenes/vine_scene/scene.yaml` beside them and fill in the transform
(see *Scanning your robot for photoreal rendering*), run the segmentation scripts with
`--outdir data/scenes/<scene>/segmentations/<build>`, and tar the folder for
whoever needs it. `configs/object_configs/objects.yaml` is the older
single-file form; it still loads, but a `scene.yaml` with the same name wins.

## Adding your robot

You need a URDF. Nothing else — the simulator works out the rest from it.

1. Make a folder under `data/robots/` and put the URDF (and its meshes)
   inside. Add a `robot.yaml` next to it that says where the URDF is:
   ```yaml
   urdf_path: my_robot.urdf
   ```
   `data/robots/example_panda/` is a complete, working example — copy it.
2. Check what the simulator sees:
   ```bash
   python scripts/check_robot.py my_robot
   ```
   It prints the arm joints, gripper, cameras and end-effector link it
   derived, and warns about anything it had to guess. If a guess is wrong,
   add the matching key under `robot:` in `robot.yaml` — every key is
   optional and documented in the example.
3. Put it in a scene:
   ```bash
   python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive --robot_name my_robot --viewer
   ```
   `--viewer` works with any environment: it loads the scene and the robot and
   nothing else — no task, no planning, no datasets — so you can look at how the
   robot fits. Without a splat scan of its own the robot is drawn from its
   meshes, composited by depth into the scene. Drop `--viewer` when you want
   the environment's task to run against it.
4. Move it where you want it: press **Robot Placement** in the control window.
   Sliders move the base (x, y, z, yaw) and every arm joint live in the scene;
   **Save placement** writes the pose into your `robot.yaml` so it starts
   there next time (the file's comments are kept).

What the yaml can describe, all optional: which joints are the arm and which
the gripper, how the gripper is commanded (one value for a parallel gripper,
one per finger, or synergies for a coupled hand), any number of cameras and
which link each sits on, the end-effector link, and the base — `fixed`,
`planar` (placed by the sliders, no physics), or `wheeled` (a free body on a
ground plane; its wheel joints are velocity-controlled through
`drive_wheels`). Planning covers the arm; driving a base around is teleop
territory. Details and defaults are in `splatsim/robots/robot_spec.py`.

To render your robot photoreal — as gaussians rather than meshes — it needs a
scan and a calibration; that's the next section.

## Scanning your robot for photoreal rendering

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

Demos are planned by the simulator itself (RRT to the task goal, retimed to
joint limits) and saved as a LeRobot dataset under
`~/.cache/huggingface/lerobot/<repo_id>`.

### From the control GUI

Launch any environment with `--control_gui` — the vine one, for example:

```bash
python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive \
    --robot_port 6003 --wrist_cam_ver=2 --control_gui
```

In the control window, press **Trajectory Gen Mode**. The panel that opens has
every planner knob, but the two you need are:

- **Num Trajectories** — how many episodes to record
- **LeRobot Repo ID (user/name)** — where to save them, e.g. `you/vine_reach`

Press **Start Traj Gen**. Each episode randomises the scene, plans a path,
executes it while rendering, and appends to the dataset. The status line
under the mode buttons says what's happening at each moment — which RRT
attempt, which frame is being recorded, when the episode and dataset are
being saved — so a long plan isn't mistaken for a hang. **Stop** ends the
run early and still finalises what was recorded — it also interrupts a plan
that's still running (within about one RRT attempt), so you don't have to
wait for a slow episode to finish.

In the vine environment each episode also draws a new target bunch at
random; the episode's `task_description` records which one
(`reach grape bunch N`). Pin a bunch with `TARGET_BUNCH_INDEX` on the env
class if you want every episode on the same one. If you tune the other knobs,
**Export Config** saves them to `configs/traj_configs/<env>.json` so the run is
reproducible — see [configs/traj_configs/README.md](configs/traj_configs/README.md).

To watch what you recorded, press **Eval Benchmark**, enter the same repo id,
**Load Dataset**, and step through with **Prev / Next Episode** and
**Replay Episode**.

### In parallel, from one command

One simulator generates one episode at a time. For a bigger dataset,
`launch_trajgen_pool.py` brings up several headless workers and a coordinator
panel, then merges their shards into a single dataset when the target is
reached:

```bash
python scripts/launch_trajgen_pool.py --robot sim_pybullet_vine_interactive \
    --workers 4 --base-repo-id you/vine_reach
```

Set the episode target in the panel and press Start; it runs unattended and
tears the workers down on exit (Ctrl-C included). VRAM is what limits
`--workers`, not CPU cores. To reuse settings you tuned in the GUI, pass the
exported file with `--traj-config-file`; anything else you'd pass to
`launch_nodes.py` goes through `--extra-worker-arg`, and `--dry-run` prints
the commands without launching. The script's docstring lists the rest.

## Tuning the vine task

- `scripts/tune_goal_pose.py --once --target 0 --out goal.png` — with the sim
  running, renders the wrist camera at the goal pose and the base view with
  every bunch marked. The quickest way to see whether the arm is aiming at
  fruit.
- `scripts/mark_grape_targets.py --load` — click on bunches in the splat
  render to add or move targets; saves to
  `data/scenes/vine_scene/segmentations/vine_and_trellis/grape_targets_manual.json`.
- Task knobs are class attributes on `VineGrapeReachPybulletRobotServer` in
  `splatsim/robots/sim_robot_pybullet_vine.py`: `TARGET_BUNCH_INDEX` (`None`
  = random bunch each reset), `GRAPE_STANDOFF_M`, and the camera-framing
  settings, each documented inline.

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
 