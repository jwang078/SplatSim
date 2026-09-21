# SplatSim
SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

[Project Page](https://splatsim.github.io) | [Arxiv](https://arxiv.org/abs/2409.10161)



This repository contains the code for the paper "SplatSim". 

## Installation

You'll need conda and an NVIDIA GPU. We've tested on Python 3.12 / CUDA 12.8.
Older cards work too: `install.sh` reads your GPU's compute capability and
picks a PyTorch build that has kernels for it (CUDA 12.8 for Turing and
newer, CUDA 12.6 for anything older, down to a GTX 10-series).

```bash
git clone --recursive git@github.com:jwang078/SplatSim.git
cd SplatSim
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

<details>
<summary>LeRobot — installed alongside SplatSim, and how to point it at your own checkout</summary>

`install.sh` also clones and installs
[our fork of LeRobot](https://github.com/jwang078/lerobot) into
`external/lerobot` (editable, ignored by git). SplatSim reads and writes its
datasets through it, and it's where the training and DAgger code lives, so
the two are meant to be worked on together. If you already have a checkout,
the installer uses a sibling `../lerobot` automatically, or point it anywhere
with `LEROBOT_DIR=/path/to/lerobot ./install.sh` — and symlinks whichever one
it used to `external/lerobot`, so the layout is the same either way and paths
can just say `external/lerobot`. (`external/` is gitignored, link included.)
</details>

<a id="things-installsh-already-handles"></a>
<details>
<summary>Things <code>install.sh</code> already handles — the dependencies it patches so they build</summary>

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
- **`gsplat`** compiles its CUDA kernels on the first render rather than at
  install time, so `install.sh` clears that path in advance: it links the
  CUDA headers where the host compiler will find them, and on pre-Volta GPUs
  runs `scripts/patch_gsplat_pre_volta.py` (gsplat's backward kernels use a
  Volta-only cooperative-groups call, and one file that won't compile takes
  the whole renderer with it). Then it builds the extension once, so the
  first sim launch doesn't spend minutes on nvcc. Re-run that script by hand
  after any `pip install gsplat`, which restores the stock sources.
</details>

### If something goes wrong

- **`ModuleNotFoundError: simple_knn`** (or an empty `submodules/` folder) —
  the clone missed its submodules. Run
  `git submodule update --init --recursive`, then `./install.sh` again. If a
  submodule folder has nothing but a `.git` inside even after that, check it
  out directly with `git -C submodules/<name> checkout -f HEAD`.
- **`nvcc: command not found`** when running things later — the conda env
  isn't active. A quick `conda activate splatsim` fixes it.
- **`[splat assets] not on disk` at launch** — a scan you haven't downloaded.
  The sim lists what's missing and keeps going rather than aborting, as far
  as the assets allow:
  - only the ROBOT's scan missing — the scene still renders as gaussians and
    the robot is composited in from PyBullet geometry
    (`RENDER_ROBOT_SPLAT=False` + `COMPOSITE_PYBULLET_ROBOT`, the same mode
    the floating-gripper vine env runs in);
  - a SCENE scan missing — nothing photoreal is left, so image observations
    come from the PyBullet camera instead.

  Physics, control, planning, metrics and oracle state work either way.
  Unpack the scene's tarball into `data/stages/` for the full splat render.
- **`CUDA error: no kernel image is available for execution on the device`**
  — torch is a CUDA build (`cuda available: True`) but has no kernels for
  your GPU. Compare `torch.cuda.get_device_capability()` with
  `torch.cuda.get_arch_list()`; if your `sm_XX` isn't in the list, re-run
  `./install.sh` (it now picks the wheel index from the GPU) or force one
  with `TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./install.sh`.
- **Please don't upgrade torch.** It's pinned to 2.11.0 on purpose: the CUDA
  extensions are compiled against it, and video dataloading gets several
  times slower on other builds. The same goes for the CUDA 12.x toolchain.
  Which `+cuXXX` build of 2.11.0 you get is chosen by `install.sh` from your
  GPU — that part is not a pin.
- **Noisy `git status` after installing** — the submodule builds leave a few
  artifacts behind. Harmless, but if you'd like them hidden:
  ```bash
  echo '*.egg-info' >> .git/modules/submodules/ghalton/info/exclude
  echo '*.egg-info' >> .git/modules/submodules/gello_software/modules/third_party/DynamixelSDK/info/exclude
  ```

## Running a simulation

A simulation is a robot scan plus a scene scan, each a folder under
`data/stages/` (see *Data layout* below). Two are available to download and
make a working example: the grape vine, and the UR5 scanned in the engine
scene that it uses as its robot.

### 1. Download the example scenes

One tarball per scene, named after the folder it unpacks to:

- [vine_scene.tar.gz](https://drive.google.com/file/d/1y9-8epz9nQ868EmqGSHOH_tyuGUB_X_8/view?usp=drive_link) — the grape vine prop, scanned in the highbay, ~1 GB
- [robot_iphone_w_engine_curtain.tar.gz](https://drive.google.com/file/d/14CONUnIpUrL6v5KV99zVB8cOJanOUSZt/view?usp=drive_link) — the UR5 with its wrist camera, scanned in the engine scene, ~370 MB

Unpack both into the repo's `data/stages/` folder. Each carries its own
`stage.yaml`, so there's nothing to edit:

```bash
tar xzf /path/to/vine_scene.tar.gz -C data/stages
tar xzf /path/to/robot_iphone_w_engine_curtain.tar.gz -C data/stages
```

That gives you:

```
data/stages/vine_scene/splat/                                  gaussian splat of the vine highbay scene
data/stages/vine_scene/sfm/                                    structure-from-motion output for it
data/stages/vine_scene/segmentations/vine_and_trellis/         collision mesh, grape targets, soft-cost field
data/stages/robot_iphone_w_engine_curtain/splat/               gaussian splat of the robot
```

### 2. Launch

Each environment is a `--robot` variant of `launch_nodes.py`; the vine one is:

```bash
python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive \
    --robot_port 6003 --wrist_cam_ver=2 --control_gui
```

You should see the PyBullet window, the control GUI, and a splat render with
the arm in front of the vine. The scene starts exactly as configured — every
object at its placed pose, the robot at its scan / home pose (or the saved
`default` scenario, if there is one); nothing is randomised or planned until
you press **Reset Env** or start generating trajectories. The control window
opens in **External Control**: the server waits for a controller — a policy,
GELLO, a script — on its port and, with the default
`--sync_physics_to_client`, steps physics only on that controller's
commands, so the scene stands still until one connects. To move the robot
or objects by hand, switch to **Robot Placement**. `--wrist_cam_ver=2` picks a fisheye calibration
that ships in the code; `--robot_name` isn't needed, the variant knows its
robot.

You can also view the small engine environment:

```bash
python scripts/launch_nodes.py     --robot sim_ur_pybullet_small_engine_new_interactive     --rob
ot_port 6001 --robot_name robot_iphone_w_engine_curtain --wrist_cam_ver=2
```

The apple-on-plate demo from the paper (the `test_data` / `output` /
`bc_data/gello` downloads) is not maintained here — use the
[original SplatSim repository](https://github.com/qureshinomaan/SplatSim)
for that.

# Optional

## Using SplatSim from LeRobot

`lerobot_env_splatsim/` is a small LeRobot *environment plugin*. `install.sh` installs it next to
LeRobot, and because its distribution name starts with `lerobot_env_`, `lerobot-train` and
`lerobot-eval` import it automatically. That is what makes `--env.type=splatsim` (and
`--robot.type=splatsim_lerobot`) resolve, with no changes inside LeRobot:

```bash
lerobot-eval --env.type=splatsim --env.task=planar_3joint --env.robot_name=planar_3joint \
             --env.external_port=6023 --policy.path=<checkpoint> --eval.n_episodes=10
```

The env config (`lerobot_env_splatsim/lerobot_env_splatsim/config.py`) either connects to a node you
launched with `scripts/launch_nodes.py` (`--env.external_port`) or starts one in-process. Install it by
hand with `pip install --no-deps -e lerobot_env_splatsim`.

## Data layout

Everything the simulator loads lives under `data/`, one folder per thing,
described by a yaml next to the files. The two kinds borrow their names from
USD:

- **assets** — `data/assets/<asset>/asset.yaml`: a body you can reuse. A URDF
  and its meshes; a `robot:` block if it is a robot. `ur5/`, `ur5e/` and
  `panda/` ship with the repo.
- **stages** — `data/stages/<stage>/stage.yaml`: a scan. The gaussian splat,
  the SfM output, the transform that aligns the splat with the simulator, and
  `asset: <name>` for the body it is a scan of (the engine-scene UR5 scan says
  `asset: ur5`; a stage's own fields override the asset's). A thing scanned
  once, like the cardboard boxes, keeps its URDF in the stage folder instead.

A stage folder looks like this:

```
data/stages/<stage>/
    stage.yaml                       model_path, source_path, transformation, and one block per body under assets:
                                     (asset, base_position, scan_pose, aabb, labels_path)
    splat/                           gaussian-splatting output
    sfm/                             COLMAP / hloc output
    splat_rgb.ply                    the splat as a plain RGB cloud, for CloudCompare
    <body>_urdf_pcd.ply              that body's URDF sampled at its scan_pose, for aligning
    <body>_labels.npy + .json        which URDF link each gaussian of that body belongs to, and the key
    segmentations/<build>/           a body built from the scan (see Making a collision body);
                                     registered under assets: in stage.yaml as <stage>/<build>
        grape_targets_manual.json    what you authored by hand
        byproducts/                  everything the scripts produced: <build>.urdf + _collision.obj,
                                     the cost fields, grape_targets.json, trunk subset, labels, viz/
```

- The folder tree is the config: every yaml is one entry, named after its
  folder and referenced by that name in code; a body listed under a stage's
  `assets:` is `<stage>/<body>` (`splat_name="vine_scene/vine_and_trellis"`).
  Paths inside it are relative to the file; a body under `assets:` inherits
  the stage's own fields (transformation, splat) beneath its own.
- `collision_frame: splat` means the build's collision mesh, cost field and
  grape targets are in the scan's frame and get the scan's `transformation`
  at load; `sim` means they were baked already. `scripts/splat_to_collision.py`
  sets it.
- The yaml files are tracked in git; the data next to them is not.
- A stage with a URDF but no splat (`ply_path`/`model_path`) still works in
  a splat-rendered scene: like an unscanned robot, the body is drawn from
  its PyBullet visual geometry and composited by depth into the render.
  Set `composite_if_no_splat: false` on an object that exists only for
  physics or planning so it stays out of the images. (`load_splat: false`
  is different — it means the object is already part of the background
  scan and must not be drawn twice.)
- Folders extracted under the older names (`data/scenes/`, `data/robots/`,
  `scene.yaml`, `robot.yaml`) still load.
- `data/scenarios/<env>__<robot>__<name>.json` is a saved arrangement: the
  robot's base pose and start joint state (by joint name) and each object's
  pose. It sits on top of the stage and asset and repeats nothing from them.
  `launch_nodes.py` loads `default` unless you pass `--scenario <name>`;
  save more from the Robot Placement panel.

To add a stage: make `data/stages/<stage>/` with `splat/` and `sfm/`, copy
`data/stages/vine_scene/stage.yaml` beside them and fill in the transform
(see *Scanning an asset for photoreal rendering*), run the segmentation scripts with
`--outdir data/stages/<stage>/segmentations/<build>`, and tar the folder for
whoever needs it.

Two fields in a stage are about *this* scan and nothing else: `scan_pose`,
the joint configuration (by joint name) the body was scanned in — the splat
and its labels are relative to it, so it never changes — and `labels_path`.
The pose a robot *starts* episodes in is the asset's
`robot.initial_joint_positions`; a stage that wants a different start writes
its own `robot:` block, which overrides the asset's.

## Adding an asset you have a URDF for (a robot, a box, an engine)

Anything with a URDF is an asset: a folder under `data/assets/` that the
simulator can load into any scene, collide with and move. If all you have
is the body's splat, *Making a collision body from a scan* builds the
collision mesh and URDF from its gaussians instead. The steps below are written for a robot (joints, gripper, cameras); something simpler like a box is the same with just step 1.

The commands below use the shipped UR5: its body is `data/assets/ur5/`, and
`robot_iphone_w_engine_curtain` is the scan of it under `data/stages/`, whose
`stage.yaml` says `asset: ur5`. Either name works wherever a robot is asked
for; the scan name renders the robot photoreal, the asset name draws it from
its meshes.

1. Make a folder under `data/assets/` and put the URDF (and its meshes)
   inside, with an `asset.yaml` next to it. The one line it needs says where
   the URDF is; `data/assets/ur5/asset.yaml` is:
   ```yaml
   urdf_path: ur5.urdf
   base_position: [0.0, 0.0, 0.0]
   base_orientation_rpy: [0.0, 0.0, 0.0]

   robot:
     base: fixed
     arm_joints: auto                # the six UR joints
     ee_link: wrist_camera_link      # goal frame = the wrist camera (imaging tasks)
     initial_joint_positions: [1.570796, -1.570796, 1.570796, -1.570796, -1.570796, 0.0]
     gripper:                        # Robotiq 2F-85: one drive joint, the other five follow it through
       kind: synergies               #   the URDF's <mimic> tags (gear constraints in the sim)
       joints: [finger_joint]
       open: [0.0]                   # drive angle (rad) at command 0
       closed: [0.8]                 #                  ... at command 1
       stroke_m: 0.085               # max opening width, for width-based commands (move_gripper)
     cameras:
       - name: wrist
         link: wrist_camera_link
         model: fisheye_v2           # the GoPro calibration used for the datasets
   ```
   Everything under `robot:` is optional (`auto` or omitted = derived from
   the URDF). `data/assets/ur5/`, `ur5e/` and `panda/` are complete, working
   robot folders — copy one.
2. Check what the simulator sees:
   ```bash
   python scripts/check_robot.py robot_iphone_w_engine_curtain
   ```
   It prints the arm joints, gripper, cameras and end-effector link it
   derived, and warns about anything it had to guess. If a guess is wrong,
   add the matching key under `robot:` in `asset.yaml` — every key is
   optional and documented in `data/assets/panda/asset.yaml`.
3. Put it in a scene:
   ```bash
   python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive \
       --robot_name robot_iphone_w_engine_curtain --viewer --control_gui
   ```
   `--viewer` works with any environment: it loads the scene and the robot and
   nothing else — no task, no planning, no datasets — so you can look at how the
   robot fits. A robot without a scan of its own (`--robot_name ur5e`, say) is
   drawn from its meshes, composited by depth into the scene. Drop `--viewer`
   when you want the environment's task to run against it.
4. **Play** (on by default; off while generating trajectories) in the control window makes the splat render update
   continuously (at the rate next to it) instead of only when something asks
   for an observation — handy for watching the splat and the PyBullet window
   side by side while you move the robot.
5. Move it where you want it: press **Robot Placement** in the control window.
   Sliders move the base (x, y, z, yaw), every arm joint and every gripper
   live in the scene. Type a name and press **Save scenario** to keep the
   arrangement as `data/scenarios/<env>__<robot>__<name>.json` (**Load
   scenario** brings one back; `--scenario <name>` launches into it).
   **Save as robot default (yaml)** instead writes the pose into the robot's
   yaml so it starts there in every scene (comments kept).

The yaml can also say, all optional: which joints are the arm and which the
gripper, how the gripper is commanded (one value per finger, or synergies
for fingers the URDF couples with `<mimic>` — the Robotiq is one of those), the cameras (pinhole with a field of view, or fisheye with
`intrinsics:` inline or as a `calibration.json` from
`scripts/calibrate_camera_intrinsics.py`), the end-effector link, and the
base — `fixed`, `planar`, or `wheeled` (velocity-controlled wheels; planning
covers the arm only). Defaults and details: `splatsim/robots/robot_spec.py`.

A robot with several arms is still one URDF and one folder: list them under
`arms:` instead of `arm_joints` / `ee_link` / `gripper`, each with its own
`ee_link`, gripper and start pose. The state and action vectors are then
every arm's joints in that order followed by every gripper's commands, and
goals refer to the first arm (or `primary_arm`). `data/assets/dual_panda/`
is a working two-arm example.

A mobile robot is the same thing on wheels: `base: wheeled` plus the two
driven `wheel_joints`, and the body is free to roll on a ground plane. In
**Robot Placement** the panel grows Drive sliders (forward m/s, turn rad/s),
and the arrow keys do the same while the PyBullet window has focus; from code
it is `server.drive_base(forward, turn)` (differential drive, geometry read
from the URDF) or `server.drive_wheels([...])` per wheel.
`data/assets/dual_ur5/` is the shipped example — two UR5s with Robotiq
grippers and wrist cameras on a wheeled box, built with
`scripts/make_multi_arm_urdf.py`, which composes any arm URDFs onto a box
base with optional wheels:

```bash
python scripts/launch_nodes.py --robot sim_pybullet_vine_interactive --robot_name dual_ur5 --viewer
```

To render your robot photoreal — as gaussians rather than meshes — it needs a
scan and a calibration; that's the next section.

## Scanning an asset for photoreal rendering

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

A scan is a *stage* (`data/stages/<stage>/stage.yaml`), and every body you
want the simulator to move — the robot, a box, an engine — is one entry
under its `assets:`. Each is cut out of the same splat with the same three
steps; the robot is just the one called `robot`. Look at
`data/stages/robot_iphone_w_engine_curtain/stage.yaml` while reading this.

#### 1. Describe the scan

The examples below use the shipped UR5 scan, `robot_iphone_w_engine_curtain`,
so every command runs as written on the downloaded data; for your own scan,
make `data/stages/<stage>/` with a `stage.yaml`:

- `model_path` — the gaussian-splat training output (ex: `~/.../output/258f657d-c`), or symlink it as `splat/` next to the yaml.
- `source_path` — the images + COLMAP output (ex: `~/data/<stage>/input`), or symlink it as `sfm/`.
- `assets:` — one block per body. For the robot:
  ```yaml
  assets:
    robot:
      asset: ur5                    # its folder under data/assets/ (see *Adding an asset you have a URDF for*)
      base_position: [0.0, 0.0, 0.0]
      scan_pose:                    # joint angles (rad, by joint name) it had during the scan
        shoulder_pan_joint: 1.5708  #   joints you leave out are 0
        ...
  ```
  A box or an apple is the same block without `scan_pose`.

#### 2. Point cloud of the URDF, align it in CloudCompare

```bash
python scripts/segment_stage_asset.py robot_iphone_w_engine_curtain --asset robot pcd
```

`--asset robot` is the block's name under `assets:` in the stage.yaml — call
the block something else and pass that instead. (`robot` is special only in
that the stage name alone refers to it.) The command writes `<stage>/robot_urdf_pcd.ply` (the URDF sampled at `scan_pose`, one
colour per link) and `<stage>/splat_rgb.ply` (the splat as a plain RGB
cloud). Check the first one has the joint pose the robot really had; if
not, fix `scan_pose` and rerun. Then open both in CloudCompare, crop the
splat down to the robot, and align — *moving the splat onto the URDF cloud*,
never the other way round:

<details>
<summary> Tips and tricks with CloudCompare </summary>

- `splat_rgb.ply` already has RGB; select it in the top-left sidebar and set `Properties > Colors` to `RGB` if it shows grey.

- Use the Segment tool (scissor in top bar) to crop out the table and other objects, leaving only the robot. Also crop out any wires (which wouldn't be present in the simulated robot). Note: select the point cloud you want to segment on the left toolbar before you click Segment, or else it will try to segment the wrong point cloud, thus making no changes. The points you segmented out re-appear after you save the segmentation because they are now in another group (in the left toolbar). You can deselect it to stop visualizing it. First select create the selection polygon with left clicks, right click to finish your polygon, click either Segment In or Segment Out, then if you want to keep on iterating on this, find a new angle then press the unpause button to start segmenting again. When you're done, press the green checkmark.

- The fastest way to align the robots seems to doing ICP w/o scale adjustment, manual adjustment, then ICP with scale adjustment. To start ICP, ctrl+click both point clouds, then `Tools > Registration > Fine Registration (ICP)`. Set the simulated robot to be the reference and the splat to be the aligned. Uncheck `adjust scale` for this first pass of ICP. Then, manually fix errors with `Translate/Rotate` (click on point cloud in left sidebar then click on the button at the top toolbar). Then run ICP again but with `adjust scaling` checked. If the URDF is fundamentally different from your real robot, prioritize aligning the robot's end effector to mitigate cascading error when doing forward kinematics in the sim.

- You can double-check alignment by setting the floor as visible and seeing if the floor planes are aligned, or by looking at all orthographic views (left toolbar)
</details>

The result is in `Transformation History` (bottom of Properties in the left
sidebar). Paste its four rows into your `stage.yaml` under `transformation:
matrix:`, formatted like the shipped
`data/stages/robot_iphone_w_engine_curtain/stage.yaml`:

```yaml
transformation:                   # splat frame -> simulator frame, shared by every body in the scan
  matrix:
  - [-0.231955, -0.002806, -0.002667, -0.299566]
  - [0.001594, 0.076423, -0.219032, 0.764096]
  - [0.003528, -0.219019, -0.076393, 0.550882]
  - [0.0, 0.0, 0.0, 1.0]
```

#### 3. Cut the body out and label it

```bash
python scripts/segment_stage_asset.py robot_iphone_w_engine_curtain --asset robot labels --show
```

fits the body's box from the URDF cloud, labels every gaussian inside it
with the nearest URDF link, and writes `aabb`, `labels_path`
(`robot_labels.npy` + a `.json` key saying which value is which link) into
the yaml. The windows show the URDF and the labelled splat in the same
colours; they should sit on the same parts, and the printed centroid
distance should be a few centimetres. If the box clips something the URDF
doesn't have (a wrist camera, cables), widen it with `urdf_bbox_adjustment`
under that body and rerun.

#### More bodies in the same scan

Repeat the same steps for every other body under `assets:` (a box, an
engine), with one difference: the scan is already placed, so instead of
pasting a stage transform you write where the URDF cloud ended up as that
body's `base_position` / `base_orientation_rpy`. Each body is then its own
entry, `<stage>/<name>`, that an environment can load and move. Bear in mind
that cutting a body out of the scan leaves a gap in the surface it stood on.

## Making a collision body from a scan (no URDF)

For objects that have no URDF, we can cut the body's gaussians
out of the splat, turn them into a mesh, and wrap that in a generated URDF,
which is what lets the robot collide with it and lets you place it. The result is a *segmentation build*,
`data/stages/<scan>/segmentations/<build>/`, which is a registry entry like
any other (the vine env's `vine_and_trellis` is one). The examples run on the
shipped `vine_scene`.

1. **Crop the body's gaussians** into a gaussian PLY (it must keep the 3DGS
   fields — SuperSplat's editor exports them; CloudCompare's PLY export does
   not unless you use `3dgsconverter` to convert it back to the non-cloudcompare version with `3dgsconverter -i input_cc.ply -o output_3dgs.ply -f 3dgs`). Put it in the build folder:
   `data/stages/vine_scene/segmentations/<build>/<build>.ply`. For vegetation,
   `scripts/segment_vine_splat.py <ply> --outdir <build dir>` splits it into
   the hard trunk (`byproducts/<build>_trunk_hard.ply`, what gets a mesh) and the soft
   twigs/leaves/grapes (`byproducts/<build>_soft_cost.npz`, what the planner steers
   around), with a picture of every stage under `byproducts/viz/`, though the color thresholding might need to be tuned.

2. **Mesh it and register it:**
   ```bash
   python scripts/splat_to_collision.py \
       data/stages/vine_scene/segmentations/vine_and_trellis/byproducts/vine_and_trellis_trunk_hard.ply \
       --outdir data/stages/vine_scene/segmentations/my_build
   ```
   writes `byproducts/<name>_collision.obj` and `byproducts/<name>.urdf` (fixed
   base, concave) and registers the build under `assets:` in the scan's `stage.yaml` — as
   `vine_scene/my_build`, named after its folder — with `collision_frame:
   splat`, meaning the mesh is in the scan's frame and the scan's
   `transformation` is applied at load: the same transform you found for the
   robot places the vine. The default backend is PlayCanvas's mesher
   (`npm i -g @playcanvas/splat-transform`); `--backend voxel` is an
   in-repo blockier mesher that needs no extra tools. `--voxel-size` sets
   the resolution in the scan's units.

3. **Check it:** `byproducts/viz/10_mesh_overlay.png` must hug the trunk points, and
   ```bash
   python scripts/visualize_collision.py --urdf data/stages/vine_scene/segmentations/my_build/byproducts/my_build.urdf \
       --soft-npz data/stages/vine_scene/segmentations/vine_and_trellis/byproducts/vine_and_trellis_soft_cost.npz --gui
   ```
   loads it in PyBullet with the soft points and a probe sphere. Grape
   targets for the task come from `scripts/regen_grape_targets.py` (or
   `mark_grape_targets.py` by hand); see *Tuning the vine task*.

An environment then refers to the build as `<stage>/<build>`
(`splat_name="vine_scene/vine_and_trellis"`); nothing else needs to know it
was a splat first. Everything the scripts produce — the URDF and mesh the
simulator loads as much as the working files — sits in the build's
`byproducts/`; the build folder's top level is for what you author by hand
(`grape_targets_manual.json`).

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
  `data/stages/vine_scene/segmentations/vine_and_trellis/grape_targets_manual.json`.
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
 