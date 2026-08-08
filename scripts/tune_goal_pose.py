"""Interactive tuner for IK GOAL POSES — no planning, no trajectory optimizer.

Regenerates the task's goal pose whenever you touch a control, teleports the
robot there, and shows what the WRIST CAMERA sees. Only IK runs, so a change
costs milliseconds instead of a trajectory-gen round trip.

Every goal-pose decision comes from `splatsim.utils.goal_pose` — the same
module the env server calls — so what you tune here is what the env
generates. Nothing about aiming, camera-up, peduncle targeting or scoring is
reimplemented in this file; see GoalPoseSpec.from_env_class.

The SIMULATOR RUNS OUT OF PROCESS. Launch it once and leave it up:

    cd ~/code/SplatSim && python scripts/launch_nodes.py \\
        --robot sim_pybullet_vine_interactive \\
        --robot_port 6003 --wrist_cam_ver=2

then run this script (it connects on 6003 by default; --sim-port to change).
If nothing is listening it exits immediately with that command rather than
hanging. Splitting them means the ~30 s splat load happens once per session
instead of once per tweak, and the two pybullet clients cannot contend.

Only the splat RENDER comes from the simulator (teleport + get_observations,
the policy's own image path). Geometry — IK, collision, the 3D view — is
computed locally, so slider drags stay in the millisecond range.

--lite skips the simulator entirely for geometry-only work; the wrist view is
then a pybullet URDF render, in which the grapes are INVISIBLE.

Controls live on the WRIST CAMERA (OpenCV) window, not pybullet's debug
panel: pybullet draws its panel text through the same GL context the splat
renderer uses, and with the soft-cost overlay up that font rendering dies —
its own labels vanish too, so debug sliders are unreadable exactly when the
scene is most worth looking at. cv2 trackbars are independent of that.

    n / p                 next / previous target (wraps)
    r                     resample IK — a different solution at identical
                          parameters (varies elbow/wrist re-seeds)
    q / Esc               quit
    standoff cm           fingertip-to-peduncle distance
    roll deg +180         extra tool spin about the aim axis (offset by 180
                          because trackbars cannot go negative)
    max aim err           how far off-axis a solution may point
    cam up 0/1/2          camera image-up: unconstrained / world +Z / world -Z
    collision chk         1 = reject configs hitting hard obstacles

The fingertip offset is NOT a control: it is rigid URDF geometry, measured by
FK (grape_targets.tool_tip_vector) as the length of the same wrist->fingerpad
vector whose direction is the tool axis. --tip-offset overrides for what-ifs.

Overlays: goal EE frame (X red, Y green, Z blue = aim/camera-forward), the
camera's image-UP in magenta, the fingertip in yellow, the bunch centre
(cyan, what the camera aims at) and the peduncle (orange, what the tool is
positioned off).

Usage:
    python scripts/tune_goal_pose.py                         # full simulator
    python scripts/tune_goal_pose.py --lite                  # fast, no splat
    python scripts/tune_goal_pose.py --control-gui           # + SplatSim panel
    python scripts/tune_goal_pose.py --debug-mode rotate_base_cam
    python scripts/tune_goal_pose.py --env pkg.mod:Class
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc

from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils import grape_targets
from splatsim.utils.pybullet_client import as_client
from splatsim.utils.goal_pose import (
    CameraUpMode, GoalPoseSpec, camera_up_state, project_points,
    resolve_targets, solve_goal_pose, wrist_view_matrix,
)

DEFAULT_ENV = "splatsim.robots.sim_robot_pybullet_vine:VineGrapeReachPybulletRobotServer"


def load_env_class(spec: str):
    mod_name, _, cls_name = spec.partition(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def load_scene(client, env_cls):
    """Robot + obstacle bodies from the env's ENV_CONFIG. Kinematics only —
    no splats, no cameras, no server."""
    robot_cfg = SplatObjectConfig(
        name="robot", splat_name=env_cls.DEFAULT_ROBOT_NAME,
        randomize_pose=False, load_splat=False,
    )
    robot_id = client.loadURDF(
        str(robot_cfg.urdf_path), basePosition=list(robot_cfg.base_position),
        useFixedBase=True,
    )
    init_q = list(robot_cfg.articulation_config.initial_joint_positions)
    movable = [j for j in range(client.getNumJoints(robot_id))
               if client.getJointInfo(robot_id, j)[2] != pb.JOINT_FIXED]
    for j, q in zip(movable, init_q):
        client.resetJointState(robot_id, j, float(q))

    obstacles = []
    for obj in getattr(env_cls.ENV_CONFIG, "objects", []):
        try:
            obstacles.append(client.loadURDF(
                str(obj.urdf_path), basePosition=list(obj.base_position),
                useFixedBase=True))
        except Exception as exc:            # noqa: BLE001 - diagnostic tool
            print(f"  (skipped obstacle {obj.name!r}: {exc})")
    return robot_id, obstacles, np.asarray(init_q[:6], dtype=np.float64)


def find_link(client, robot_id, name):
    for j in range(client.getNumJoints(robot_id)):
        if client.getJointInfo(robot_id, j)[12].decode() == name:
            return j
    return None


DEFAULT_SIM_PORT = 6003
LAUNCH_HINT = """\
The simulator is not running on {host}:{port}.

Start it once, in its own terminal (it stays up across tuner restarts):

    cd ~/code/SplatSim && python scripts/launch_nodes.py \\
        --robot {robot} \\
        --robot_port {port} --wrist_cam_ver=2

then re-run this script (add --sim-port if you used a different port).

Or run without a simulator:

    python scripts/tune_goal_pose.py --lite

--lite loads only the robot + obstacle URDFs. It starts in seconds and the
geometry is identical, but the wrist view is a pybullet URDF render, so the
grapes are INVISIBLE (they are soft-cost points with no collision geometry)
and there is no splat.\
"""


def connect_sim(host: str, port: int, robot_variant: str, timeout_s: float = 4.0):
    """Attach to an ALREADY-RUNNING simulator over ZMQ.

    Out-of-process by design: this tool holds its own pybullet client for IK
    and collision, and a second in-process pybullet + splat renderer in the
    same interpreter is both slow to start and prone to GL contention. Running
    the simulator separately also means it survives tuner restarts, so the
    30-second splat load happens once per session rather than once per tweak.

    Probes with a short timeout and raises SystemExit carrying the exact
    launch command rather than hanging forever on a dead socket (the ZMQ REQ
    client has no default receive timeout).
    """
    import zmq
    from gello.zmq_core.robot_node import ZMQClientRobot

    sim = ZMQClientRobot(port=port, host=host)
    sim._socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sim._socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    sim._socket.setsockopt(zmq.LINGER, 0)
    try:
        dofs = sim.num_dofs()
    except Exception:                                   # noqa: BLE001
        raise SystemExit(LAUNCH_HINT.format(host=host, port=port,
                                            robot=robot_variant))
    # Long timeout for real work: a splat render is far slower than the probe.
    sim._socket.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"connected to simulator at {host}:{port} ({dofs} DoF)")
    return sim


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=DEFAULT_ENV,
                    help="module:Class of the env server (default: vine)")
    ap.add_argument("--roll", type=float, default=None,
                    help="initial roll offset (deg); defaults to the env's "
                         "GRIPPER_ROLL_OFFSET_DEG")
    ap.add_argument("--target", type=int, default=None,
                    help="initial target index (default: env's)")
    ap.add_argument("--tip-offset", type=float, default=None,
                    help="override the FK-measured EE-link-to-fingertip "
                         "distance (m); normally derived from the URDF")
    ap.add_argument("--lite", action="store_true",
                    help="skip the simulator: load only the robot + obstacle "
                         "URDFs. Starts in seconds, but the wrist view is a "
                         "pybullet URDF render (grapes are soft-cost points "
                         "with no collision geometry, so they are INVISIBLE) "
                         "and there is no soft-cost ranking. Geometry-only "
                         "work; the default full-simulator path is what the "
                         "env actually does.")
    ap.add_argument("--sim-port", type=int, default=DEFAULT_SIM_PORT,
                    help="port of an ALREADY-RUNNING simulator "
                         "(scripts/launch_nodes.py --robot_port)")
    ap.add_argument("--sim-host", default="127.0.0.1")
    ap.add_argument("--sim-robot", default="sim_pybullet_vine_interactive",
                    help="robot variant named in the launch hint printed when "
                         "no simulator is found")
    ap.add_argument("--grapes-ply", default=(
        "/home/jennyw2/code/gaussian-splatting/output/"
        "grape_prop_in_highbay2_images_500_500match_800height_"
        "SIMPLE_RADIALcam/point_cloud/iteration_30000/grapes_only.ply"),
        help="segmented grape gaussians, used by search mode for visibility")
    ap.add_argument("--mode", default="search", choices=["ik", "search"],
                    help="initial solver. 'search' = task-space sample/score/"
                         "IK-filter (finds >=1 pose on 7/7 bunches); 'ik' = "
                         "the older config-space solver the env still runs, "
                         "which manages 2/7 under the same constraints. "
                         "Switchable live with the 'mode 0ik/1search' slider.")
    ap.add_argument("--search-directions", type=int, default=200,
                    help="approach directions sampled per search (higher = "
                         "better coverage, slower)")
    ap.add_argument("--search-top-k", type=int, default=25,
                    help="diverse candidates kept for the IK filter")
    ap.add_argument("--cam-fov", type=float, default=60.0,
                    help="wrist preview vertical FoV (deg)")
    ap.add_argument("--once", action="store_true",
                    help="compute once, save the wrist view, exit (no GUI)")
    ap.add_argument("--out", default="goal_wrist_view.png")
    args = ap.parse_args()

    env_cls = load_env_class(args.env)

    # ---------------------------------------------------------------- sim
    # The simulator runs OUT OF PROCESS and is only asked for things a local
    # pybullet cannot provide — splat renders. Geometry (IK, collision, the
    # 3D view) stays local, which keeps every slider drag at millisecond cost
    # and keeps this tool usable with --lite when no simulator is up.
    sim = None
    if not args.lite:
        sim = connect_sim(args.sim_host, args.sim_port, args.sim_robot)

    client = bc.BulletClient(pb.DIRECT if args.once else pb.GUI)
    if not args.once:
        # Turn off the render-preview thumbnails only. COV_ENABLE_GUI is
        # left alone so pybullet's own panel and 3D view stay usable; this
        # tool's controls live on the OpenCV window regardless (see WIN).
        for _flag in (pb.COV_ENABLE_RGB_BUFFER_PREVIEW,
                      pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            client.configureDebugVisualizer(_flag, 0)
    robot_id, obstacles, q_home = load_scene(client, env_cls)

    joint_indices = list(range(1, 7))
    # Arm joint limits, used to re-seed IK in search mode. Continuous joints
    # report lo > hi, which would make np.random.uniform draw backwards.
    joint_limits = []
    for _j in joint_indices:
        _a, _b = client.getJointInfo(robot_id, _j)[8:10]
        joint_limits.append((-2 * np.pi, 2 * np.pi) if _a > _b else (_a, _b))
    joint_limits = np.asarray(joint_limits, dtype=np.float64)
    ee_link = find_link(client, robot_id, "wrist_camera_link")
    if ee_link is None:
        sys.exit("no wrist_camera_link in this robot URDF")

    # Targets: the vine env carries a bunch list; other envs fall back to the
    # single static task target in ENV_CONFIG.
    targets_json = getattr(env_cls, "GRAPE_TARGETS_JSON", None)
    if targets_json is not None:
        bunches = grape_targets.load_targets(targets_json)
    else:
        task = getattr(env_cls.ENV_CONFIG, "task", None)
        if task is None or task.target_ee_pos is None:
            sys.exit(f"{env_cls.__name__} has neither GRAPE_TARGETS_JSON nor "
                     "a task target to aim at")
        bunches = [{"center": list(task.target_ee_pos)}]
    # reach vs look split is resolved by goal_pose.resolve_targets, the same
    # code the env uses — no second interpretation of the bunch dict here.
    centers = [np.asarray(b["center"], dtype=np.float64) for b in bunches]
    print(f"{env_cls.__name__}: {len(centers)} target(s)")

    base_pos = np.asarray(
        client.getBasePositionAndOrientation(robot_id)[0], dtype=np.float64)
    # Aim axis + fingertip distance are the direction and length of ONE rigid
    # vector (wrist -> fingerpad midpoint), so both come from FK rather than
    # being tuned. Measured at the URDF's initial gripper opening, which is
    # what the goal pose assumes.
    _tool_axis, tip_m = grape_targets.tool_tip_vector(client, robot_id, ee_link)
    if args.tip_offset is not None:
        tip_m = float(args.tip_offset)
    # Every goal-pose knob comes from the env class via the SHARED spec, so
    # this tool cannot ask for something different from what the env does.
    base_spec = GoalPoseSpec.from_env_class(env_cls, tip_offset_m=tip_m)
    if args.roll is not None:
        base_spec.roll_offset_deg = float(args.roll)
    aim_axis = tuple(base_spec.aim_axis_local)
    _cls_aim = getattr(env_cls, "GRIPPER_AIM_AXIS", None)
    aim_axis_arr = np.asarray(aim_axis, dtype=np.float64)
    print(f"goal spec (from {env_cls.__name__}): camera_up={base_spec.camera_up.value}, "
          f"peduncle={base_spec.aim_at_peduncle}, "
          f"strict_up={not base_spec.allow_roll_fallback}")
    _cls_tip = getattr(env_cls, "GRIPPER_TIP_OFFSET_M", None)
    print(f"tool geometry from URDF FK: tool axis "
          f"{np.round(_tool_axis, 3).tolist()}, fingertip offset {tip_m*100:.1f} cm")
    print(f"aiming the CAMERA axis {np.round(aim_axis_arr, 3).tolist()} at the "
          f"bunch centre; positioning off the peduncle")
    if _cls_tip is not None and abs(_cls_tip - tip_m) > 0.005:
        print(f"  NOTE {env_cls.__name__}.GRIPPER_TIP_OFFSET_M={_cls_tip} "
              f"disagrees with FK ({tip_m:.3f}) by "
              f"{abs(_cls_tip - tip_m)*1000:.0f} mm. Either the constant is "
              f"stale, or the gripper is at a different opening than when it "
              f"was measured (the pads swing as the fingers close).")
    if _cls_aim is not None and float(
            np.dot(np.asarray(_cls_aim) / np.linalg.norm(_cls_aim), _tool_axis)) < 0.99:
        print(f"  NOTE {env_cls.__name__}.GRIPPER_AIM_AXIS={_cls_aim} "
              f"disagrees with FK {np.round(_tool_axis, 3).tolist()}.")

    skip_pairs = [tuple(x) for x in
                  getattr(env_cls, "SELF_COLLISION_SKIP_PAIRS", [])]

    from splatsim.utils.rrt_path_utils import check_links_in_collision

    # Rank goal candidates by soft cost exactly as the env does, so the poses
    # shown here are the poses the env will actually generate. Only available
    # Built locally; see below.
    # Soft-cost scorer, built LOCALLY from the same field the env uses. It
    # cannot come over ZMQ (the sim exposes observations, not its planner), and
    # a local copy is exact anyway: same npz, same surface sampling, same max
    # reduction as RRTToGoalPlanner.
    score_fn = None
    try:
        from splatsim.utils.rrt_to_goal import RRTToGoalPlanner
        import dataclasses as _dcx
        _env_dict = {"name": env_cls.ENV_CONFIG.name,
                     "objects": [{**_dcx.asdict(o), "__type__": type(o).__name__}
                                 for o in env_cls.ENV_CONFIG.objects],
                     "soft_cost": env_cls.ENV_CONFIG.soft_cost}
        _pl = RRTToGoalPlanner(
            pb_client=client._client, robot_id=robot_id,
            joint_indices=joint_indices, ee_link_index=ee_link, num_dofs=6,
            fps=30, self_collision_skip_pairs=skip_pairs,
            obstacle_clearance=0.02, soft_cost_mode="guided",
            soft_cost_weight=100.0)
        _pl.load_obstacles({**_env_dict, "objects": []})  # field only, no bodies
        if _pl._soft_cost_field is not None:
            score_fn = _pl._config_soft_cost
    except Exception as _exc:                          # noqa: BLE001
        print(f"  (soft-cost scorer unavailable: {_exc})")
    print("goal ranking: " + ("soft-cost (matches the env)" if score_fn
                              else "take-first (no soft-cost field found)"))

    def collides(q):
        return bool(check_links_in_collision(
            robot_id, joint_indices, q, obstacles,
            self_collision_skip_pairs=skip_pairs, obstacle_clearance=0.01))

    # Keys match `solve`'s signature so the --once path and the slider path
    # feed it the same way.
    defaults = dict(
        target_i=args.target if args.target is not None
        else int(getattr(env_cls, "TARGET_BUNCH_INDEX", 0)),
        standoff_cm=float(getattr(env_cls, "GRAPE_STANDOFF_M", 0.10)) * 100.0,
        roll_deg=args.roll if args.roll is not None
        else float(getattr(env_cls, "GRIPPER_ROLL_OFFSET_DEG", 0.0)),
        aim_err_deg=12.0,
        use_coll=True,
        ik_seed=0,
        cam_up=(2.0 if tuple(getattr(env_cls, "GRIPPER_CAMERA_UP_WORLD",
                                     (0, 0, 1)))[2] < 0 else 1.0),
    )

    # Slider value -> shared CameraUpMode (no second private mapping).
    _UP_MODES = {0: CameraUpMode.OFF, 1: CameraUpMode.UPRIGHT,
                 2: CameraUpMode.INVERTED}

    # ------------------------------------------------ SEARCH mode machinery
    # Mode 1 runs the task-space pipeline (sample EE poses -> score -> IK
    # filter) instead of the config-space solver. Its scene clouds are built
    # lazily and its results cached per (target, params) key: the search takes
    # seconds, so it must not re-run on every frame — you scroll the ranked
    # results with the "candidate" slider, which is instant.
    search = {"clouds": None, "key": None, "results": [], "msg": ""}
    # [mode, from_below, candidate] — written by read() each frame so solve()
    # can dispatch without changing its signature (the --once path calls it
    # with the same keyword set as before).
    vals_mode = [1.0 if args.mode == "search" else 0.0, 0.0, 0.0]

    def _ensure_clouds():
        if search["clouds"] is not None:
            return search["clouds"]
        import json as _json
        from scipy.spatial import cKDTree as _KD
        from splatsim.utils import ee_pose_search as _eps
        from splatsim.utils.splat_ply_io import read_gaussian_ply as _rply
        sd = Path(env_cls.GRAPE_TARGETS_JSON).parent
        T = np.asarray(_json.loads((sd / "splat_to_sim.json").read_text()))
        grapes = _rply(args.grapes_ply).xyz @ T[:3, :3].T + T[:3, 3]
        veg = np.asarray(np.load(env_cls.SOFT_COST_NPZ)["points"])
        gl = list(range(7, client.getNumJoints(robot_id)))
        gc, gr = _eps.gripper_spheres(client, robot_id, ee_link, gl)
        from optimize_ee_poses import load_hard_points as _lhp
        search["clouds"] = {"grapes": grapes, "veg": veg, "gc": gc, "gr": gr,
                            "hard": _lhp(env_cls), "gtree": _KD(grapes)}
        print(f"  [search] grapes {len(grapes):,}  veg {len(veg):,}  "
              f"hard {0 if search['clouds']['hard'] is None else len(search['clouds']['hard']):,}")
        return search["clouds"]

    def run_search(ti, standoff_cm, cam_up, use_coll, from_below):
        """Full task-space pipeline for one bunch. Returns ranked feasible
        candidates: [(score, pos, quat, q, terms), ...]."""
        from scipy.spatial import cKDTree as _KD
        from scipy.spatial.transform import Rotation as _R
        from splatsim.utils import ee_pose_search as _eps

        cl = _ensure_clouds()
        b = bunches[ti]
        ctr = np.asarray(b["center"], dtype=np.float64)
        ext = max(b.get("extent", [0.1, 0.1, 0.1]))
        tp = cl["grapes"][np.asarray(
            cl["gtree"].query_ball_point(ctr, 0.5 * ext + 0.04))]
        if len(tp) == 0:
            return [], "no grape points near this target"
        d, _ = _KD(tp).query(cl["veg"])
        clouds = _eps.build_clouds(cl["grapes"], cl["veg"], cl["gc"], cl["gr"],
                                   occluder_pts=cl["veg"][d > 0.03],
                                   hard_pts=cl["hard"])
        spec = _eps.SearchSpec(
            n_directions=args.search_directions, top_k=args.search_top_k,
            standoff_range=(max(standoff_cm / 100.0 - 0.04, 0.03),
                            standoff_cm / 100.0 + 0.06),
            camera_up_world=_UP_MODES.get(int(round(cam_up)),
                                          CameraUpMode.OFF).world_up(),
            tip_offset_m=float(tip_m),
            base_xyz=tuple(np.asarray(
                client.getBasePositionAndOrientation(robot_id)[0])),
        )
        if from_below:
            spec.direction_hint = (0.0, 0.0, -1.0)
            spec.direction_max_angle_deg = 60.0
        pos_s, rot_s, ap_s = _eps.sample_poses(ctr, spec)
        # Distant bunches (this scene has several past 1.2 m) leave only a
        # sliver of the sampling shell inside the arm's reach, so say so —
        # "0 feasible" from 5 samples is a sampling problem, not a geometry
        # one, and the two want different fixes.
        n_raw = len(_eps.fibonacci_directions(spec.n_directions)) * spec.n_standoffs
        if len(pos_s) < 0.05 * n_raw:
            print(f"  [search] WARNING only {len(pos_s)}/{n_raw} poses survived "
                  f"the reach prefilter — bunch is "
                  f"{np.linalg.norm(ctr - np.asarray(spec.base_xyz)):.2f} m "
                  f"from the base; widen SearchSpec.reach_range if the arm "
                  f"really can get there")
        if not len(pos_s):
            return [], "no candidate poses (reach prefilter)"
        sc = np.full(len(pos_s), -1.0); terms = [None] * len(pos_s)
        n_hard = 0
        for i in range(len(pos_s)):
            if _eps.gripper_hits_hard(pos_s[i], rot_s[i], clouds):
                n_hard += 1
                continue
            sc[i], terms[i] = _eps.score_pose(pos_s[i], rot_s[i], ap_s[i],
                                              clouds, spec, tp)
        ok = np.flatnonzero(sc >= 0)
        if not len(ok):
            return [], "every pose hits the trellis"
        keep = ok[_eps.select_diverse(pos_s[ok], rot_s[ok], sc[ok], spec)]
        out = []
        for i in keep:
            quat = _R.from_matrix(rot_s[i]).as_quat()
            for att in range(4):
                seed = (q_home if att == 0 else
                        np.random.uniform(joint_limits[:, 0], joint_limits[:, 1]))
                for j, qq in zip(joint_indices, seed):
                    client.resetJointState(robot_id, j, float(qq))
                sol = client.calculateInverseKinematics(
                    robot_id, ee_link, pos_s[i].tolist(), list(quat),
                    maxNumIterations=300, residualThreshold=1e-9)
                q = np.asarray(sol[:6])
                for j, qq in zip(joint_indices, q):
                    client.resetJointState(robot_id, j, float(qq))
                st = client.getLinkState(robot_id, ee_link,
                                         computeForwardKinematics=True)
                if np.linalg.norm(np.asarray(st[4]) - pos_s[i]) > 0.01:
                    continue
                if _R.from_matrix(rot_s[i].T @ _R.from_quat(
                        st[5]).as_matrix()).magnitude() > np.radians(15):
                    continue
                if use_coll and collides(q):
                    continue
                out.append((float(sc[i]), pos_s[i], np.asarray(st[5]), q,
                            terms[i]))
                break
        out.sort(key=lambda t: -t[0])
        return out, (f"{len(out)}/{len(keep)} IK-feasible of {len(pos_s)} "
                     f"sampled ({n_hard} hit trellis)")

    def solve(target_i, standoff_cm, roll_deg, aim_err_deg, use_coll,
              cam_up, ik_seed):
        """Regenerate the goal pose via the SAME solver the env calls.
        Returns (pos, quat, q, message)."""
        import dataclasses as _dc

        ti = int(np.clip(target_i, 0, len(bunches) - 1))
        if int(round(vals_mode[0])) == 1:
            key = (ti, round(standoff_cm), round(cam_up), bool(use_coll),
                   bool(vals_mode[1]))
            if search["key"] != key:
                print(f"  [search] running for target {ti}...")
                search["results"], search["msg"] = run_search(
                    ti, standoff_cm, cam_up, use_coll, bool(vals_mode[1]))
                search["key"] = key
                print(f"  [search] {search['msg']}")
            res = search["results"]
            if not res:
                return None, None, None, f"SEARCH: {search['msg']}"
            k = int(np.clip(vals_mode[2], 0, len(res) - 1))
            sco, p_, qt_, q_, tm = res[k]
            for j, qq in zip(joint_indices, q_):
                client.resetJointState(robot_id, j, float(qq))
            tt = "  ".join(f"{a}={b:.2f}" for a, b in (tm or {}).items())
            return p_, qt_, q_, (f"SEARCH cand {k+1}/{len(res)} "
                                 f"score={sco:.3f} | {tt}")
        spec = _dc.replace(
            base_spec,
            standoff_m=standoff_cm / 100.0,
            roll_offset_deg=roll_deg,
            max_aim_error_deg=aim_err_deg,
            camera_up=_UP_MODES.get(int(round(cam_up)), CameraUpMode.OFF),
            ik_seed=int(ik_seed),
        )
        try:
            pos, quat, q = solve_goal_pose(
                client, robot_id, ee_link, joint_indices, bunches[ti], spec,
                collision_fn=collides if use_coll else None,
                score_fn=score_fn,
            )
            return pos, quat, q, "OK"
        except ValueError as exc:
            return None, None, None, f"FAILED: {exc}"

    # Which bunch is currently selected, for overlay highlighting. Defined
    # here (not in the GUI-only section) because --once returns before that.
    overlay_state = {"sel": int(defaults["target_i"]) % max(len(bunches), 1)}

    def draw_target_overlay(img, camera, sel_idx):
        """Draw EVERY detected bunch into a rendered image: centre (cyan, what
        the camera aims at), peduncle (orange, what the tool is placed off),
        and an index/size label.

        Drawing all of them — not just the selected one — is the point: a
        cluster label sitting on a leaf or a highlight is obvious against the
        splat, and that is how false positives in grape detection get caught.
        Markers are projected with the same intrinsics the render used, so
        their position is meaningful rather than indicative.
        """
        import cv2 as _cv2

        if camera is None:
            return img
        h, w = img.shape[:2]
        pts, labels = [], []
        for i, b in enumerate(bunches):
            reach, ctr = resolve_targets(b, base_spec)
            pts.append(ctr); labels.append((i, b.get("n_points"), True))
            pts.append(reach); labels.append((i, None, False))
        uv, valid = project_points(np.asarray(pts), camera,
                                   rectify_zoom=1.0)
        out = img.copy()
        for k, ((i, n, is_center), ok) in enumerate(zip(labels, valid)):
            if not ok:
                continue
            x, y = int(round(uv[k, 0])), int(round(uv[k, 1]))
            if not (-20 <= x < w + 20 and -20 <= y < h + 20):
                continue
            sel = (i == sel_idx)
            if is_center:
                col = (0, 255, 255) if sel else (0, 160, 160)   # cyan
                _cv2.circle(out, (x, y), 7 if sel else 5, col, 2 if sel else 1)
                txt = f"{i}" + (f" n={n}" if n is not None else "")
                _cv2.putText(out, txt, (x + 9, y - 6),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                             _cv2.LINE_AA)
            else:
                col = (0, 165, 255) if sel else (0, 110, 170)   # orange
                r = 6 if sel else 4
                _cv2.line(out, (x - r, y - r), (x + r, y + r), col, 2)
                _cv2.line(out, (x - r, y + r), (x + r, y - r), col, 2)
        return out

    def _obs_image(obs, prefix):
        """CHW float32 [0,1] observation -> HWC uint8, or None."""
        img = next((obs[k] for k in (f"{prefix}_letterbox", f"{prefix}_stretch",
                                     prefix) if obs.get(k) is not None), None)
        if img is None:
            return None
        a = np.asarray(img)
        if a.ndim == 3 and a.shape[0] in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        return a[:, :, :3]

    def _sim_observe():
        """Teleport the SIM's robot to our current local config, then pull its
        observations. This is the only thing the simulator is used for — the
        splat render — and going through teleport+observe means the image is
        produced by the policy's own pipeline rather than a lookalike."""
        q = np.asarray([client.getJointState(robot_id, j)[0]
                        for j in joint_indices], dtype=np.float64)
        try:
            sim.teleport_joint_state(np.concatenate([q, [0.0]]))
            return sim.get_observations()
        except Exception as exc:                        # noqa: BLE001
            print(f"  (sim observation failed: {type(exc).__name__}: {exc})")
            return None

    def wrist_view(width=480, height=360):
        if sim is not None:
            obs = _sim_observe()
            if obs is not None:
                img = _obs_image(obs, "wrist_rgb")
                if img is not None:
                    # No camera object over ZMQ, so target markers cannot be
                    # projected into the SIM's image; the 3D pybullet view
                    # carries those overlays instead.
                    return img
        return _pybullet_wrist_view(width, height)

    def base_view():
        """Base-camera splat image from the simulator (None with --lite)."""
        if sim is None:
            return None
        obs = _sim_observe()
        return None if obs is None else _obs_image(obs, "base_rgb")

    def _pybullet_wrist_view(width=480, height=360):
        """Render what the wrist camera sees at the CURRENT joint state.

        Convention copied from PybulletRobotServerBase: the camera looks down
        wrist_camera_link +Z and its image-up is -Y (COLMAP +Y is down), with
        no offset between link frame and camera frame.
        """
        view = wrist_view_matrix(client, robot_id, ee_link)
        proj = client.computeProjectionMatrixFOV(
            fov=args.cam_fov, aspect=width / height, nearVal=0.02, farVal=6.0)
        _, _, rgb, _, _ = client.getCameraImage(
            width, height, view, proj, renderer=pb.ER_TINY_RENDERER)
        return np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)

    def apply(vals):
        # Underscore keys exist only for change-detection and redraw (mode,
        # from-below, candidate); solve() reads them via vals_mode instead, so
        # they must not be splatted into its signature.
        pos, quat, q, msg = solve(
            **{k: v for k, v in vals.items() if not k.startswith("_")})
        if q is not None:
            for j, qi in zip(joint_indices, q):
                client.resetJointState(robot_id, j, float(qi))
        return pos, quat, q, msg

    if args.once:
        # vals_mode is already set from --mode above; --once never reads the
        # trackbars, so this is what selects the solver for it.
        pos, quat, q, msg = apply(defaults)
        print(f"goal: {msg}")
        if q is not None:
            R = np.asarray(client.getMatrixFromQuaternion(quat)).reshape(3, 3)
            print(f"  pos      = {np.round(pos, 4).tolist()}")
            print(f"  quat     = {np.round(quat, 4).tolist()}")
            print(f"  q        = {np.round(q, 4).tolist()}")
            print(f"  cam fwd  = {np.round(R[:, 2], 3).tolist()}")
            print(f"  cam up   = {np.round(-R[:, 1], 3).tolist()}  "
                  f"(world +Z component {(-R[:, 1])[2]:+.3f}; "
                  f"negative = upside down)")
            import matplotlib.pyplot as plt
            plt.imsave(args.out, wrist_view())
            print(f"  wrist view -> {args.out}")
            _b = base_view()
            if _b is not None:
                _bp = args.out.replace(".png", "_base.png")
                plt.imsave(_bp, _b)
                print(f"  base view  -> {_bp}  (all clusters overlaid)")
        return

    # ------------------------------------------------------------- sliders
    # Target index is an integer over a handful of bunches, so it gets prev/
    # next BUTTONS rather than a float slider you have to land exactly on.
    # pybullet turns a debug parameter into a button when min > max; reading
    # it returns a monotonically increasing press COUNT, so a press is
    # detected by the value changing rather than by any pressed/released
    # state. Wraps around, so "next" from the last target returns to 0.
    import cv2

    # Controls live on the OpenCV window, NOT pybullet's debug panel.
    # PyBullet's ExampleBrowser draws its panel text through the same GL
    # context the splat renderer hammers, and with the soft-cost overlay up
    # its font rendering dies — the panel's own labels vanish along with
    # ours, so debug sliders there are unreadable exactly when the scene is
    # most worth looking at. cv2 trackbars render via Qt/GTK, independent of
    # that context, so they keep their labels regardless.
    WIN = "wrist camera (goal pose)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 720, 560)

    # Trackbars are integer-only; roll is stored offset so it can go negative.
    ROLL_OFF = 180
    N_TARGETS = len(bunches)
    _tb = [
        # Target index as a slider rather than prev/next buttons: it is an
        # integer over a handful of bunches, so a slider both replaces the
        # buttons and lets you jump straight to one. n/p keys still step it.
        ("target", 0, max(N_TARGETS - 1, 1),
         int(defaults["target_i"]) % max(N_TARGETS, 1)),
        # Dragging this re-runs IK with a different random seed set, i.e. it
        # IS the "regenerate solution" control; the r key bumps it by one.
        ("ik seed (regen)", 0, 99, 0),
        # 0 = config-space solver (what the env runs today); 1 = task-space
        # sample/score/IK-filter pipeline. Search results are cached, so the
        # "candidate" slider scrolls them instantly.
        ("mode 0ik/1search", 0, 1, 1 if args.mode == "search" else 0),
        ("candidate", 0, 60, 0),
        ("from below 0/1", 0, 1, 0),
        ("standoff cm", 2, 30, int(round(defaults["standoff_cm"]))),
        ("roll deg +180", 0, 360, int(round(defaults["roll_deg"])) + ROLL_OFF),
        ("max aim err", 2, 45, int(round(defaults["aim_err_deg"]))),
        ("cam up 0/1/2", 0, 2, int(round(defaults["cam_up"]))),
        ("collision chk", 0, 1, int(defaults["use_coll"])),
    ]
    for name, lo, hi, init in _tb:
        cv2.createTrackbar(name, WIN, max(init, lo), hi, lambda _v: None)
        cv2.setTrackbarMin(name, WIN, lo)
        cv2.setTrackbarPos(name, WIN, max(min(init, hi), lo))

    # Real buttons too (Qt control panel, opened with Ctrl+P in the image
    # window). Wrapped because createButton exists only on Qt builds.
    def _step_target(delta):
        cur = cv2.getTrackbarPos("target", WIN)
        cv2.setTrackbarPos("target", WIN, (cur + delta) % max(N_TARGETS, 1))

    def _bump_seed():
        cv2.setTrackbarPos("ik seed (regen)", WIN,
                           (cv2.getTrackbarPos("ik seed (regen)", WIN) + 1) % 100)
    try:
        cv2.createButton("<< prev target", lambda *_: _step_target(-1),
                         None, cv2.QT_PUSH_BUTTON, 0)
        cv2.createButton("next target >>", lambda *_: _step_target(1),
                         None, cv2.QT_PUSH_BUTTON, 0)
        cv2.createButton("regenerate IK", lambda *_: _bump_seed(),
                         None, cv2.QT_PUSH_BUTTON, 0)
        _has_buttons = True
    except Exception:                               # noqa: BLE001
        _has_buttons = False

    def read():
        # solve() reads these through vals_mode (see its dispatch) so its
        # signature stays identical to the --once path's.
        vals_mode[0] = cv2.getTrackbarPos("mode 0ik/1search", WIN)
        vals_mode[1] = cv2.getTrackbarPos("from below 0/1", WIN)
        vals_mode[2] = cv2.getTrackbarPos("candidate", WIN)
        return {
            "target_i": cv2.getTrackbarPos("target", WIN),
            "ik_seed": cv2.getTrackbarPos("ik seed (regen)", WIN),
            "standoff_cm": float(cv2.getTrackbarPos("standoff cm", WIN)),
            "roll_deg": float(cv2.getTrackbarPos("roll deg +180", WIN) - ROLL_OFF),
            "aim_err_deg": float(cv2.getTrackbarPos("max aim err", WIN)),
            "cam_up": float(cv2.getTrackbarPos("cam up 0/1/2", WIN)),
            "use_coll": cv2.getTrackbarPos("collision chk", WIN) > 0,
            "_mode": float(vals_mode[0]),
            "_below": float(vals_mode[1]),
            "_cand": float(vals_mode[2]),
        }

    markers: list[int] = []

    def redraw(pos, quat, msg, vals):
        for m in markers:
            client.removeUserDebugItem(m)
        markers.clear()
        _ti = int(np.clip(vals["target_i"], 0, len(centers) - 1))
        overlay_state["sel"] = _ti
        reach_pt, center = resolve_targets(bunches[_ti], base_spec)
        # bunch centre = what the camera centres on (cyan cross)
        for ax in np.eye(3) * 0.03:
            markers.append(client.addUserDebugLine(
                (center - ax).tolist(), (center + ax).tolist(),
                [0, 1, 1], lineWidth=2))
        # peduncle = what the tool is positioned off (orange cross)
        for ax in np.eye(3) * 0.025:
            markers.append(client.addUserDebugLine(
                (reach_pt - ax).tolist(), (reach_pt + ax).tolist(),
                [1, 0.5, 0], lineWidth=3))
        # In search mode show every IK-feasible candidate as a short axis, so
        # the spatial spread of the solution set is visible at a glance and
        # the selected one is seen in context rather than alone.
        if int(round(vals["_mode"])) == 1 and search["results"]:
            sel = int(np.clip(vals["_cand"], 0, len(search["results"]) - 1))
            for k, (sc_, p_, qt_, _q, _t) in enumerate(search["results"]):
                Rk = np.asarray(
                    client.getMatrixFromQuaternion(qt_)).reshape(3, 3)
                col = [0, 1, 0] if k == sel else [0.35, 0.35, 0.35]
                markers.append(client.addUserDebugLine(
                    p_.tolist(), (p_ + Rk[:, 2] * (0.06 if k == sel else 0.03)).tolist(),
                    col, lineWidth=3 if k == sel else 1))
        if pos is not None:
            R = np.asarray(client.getMatrixFromQuaternion(quat)).reshape(3, 3)
            # EE frame: X red, Y green, Z blue (Z == aim == camera forward)
            for k, col in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
                markers.append(client.addUserDebugLine(
                    pos.tolist(), (pos + R[:, k] * 0.08).tolist(),
                    col, lineWidth=3))
            # camera image-UP (magenta) — flips when you roll 180
            up = -R[:, 1]
            markers.append(client.addUserDebugLine(
                pos.tolist(), (pos + up * 0.12).tolist(),
                [1, 0, 1], lineWidth=4))
            tip = pos + R @ _tool_axis * tip_m
            markers.append(client.addUserDebugLine(
                pos.tolist(), tip.tolist(), [1, 1, 0], lineWidth=2))
            # Classify the ACHIEVED camera roll, with a deadband. A bare
            # `up[2] < 0` test labels a HORIZONTAL camera "upside down",
            # which is exactly the up_z ~= -0.001 case the position-relax
            # search often lands on — so the old label read UPSIDE DOWN for
            # poses that were really sideways, i.e. it meant nothing.
            # Classification lives in goal_pose.camera_up_state so the tool
            # and any other consumer agree on what "upside down" means (the
            # deadband matters: a bare up_z<0 calls a HORIZONTAL camera
            # inverted).
            state, tilt, up = camera_up_state(quat, client)
            req_mode = _UP_MODES.get(int(round(vals["cam_up"])), CameraUpMode.OFF)
            req = req_mode.world_up()
            if req is None:
                req_txt = "no cam-up requested"
            else:
                err = np.degrees(np.arccos(np.clip(
                    up @ (np.asarray(req) / np.linalg.norm(req)), -1.0, 1.0)))
                req_txt = (f"req {req_mode.value}, "
                           f"{'MET' if err <= base_spec.max_up_error_deg else 'NOT MET'}"
                           f" ({err:.0f} deg off)")
            txt = (f"{msg} | camera {state} (image-up {tilt:.0f} deg from world +Z)"
                   f" | {req_txt} | fingertip-target "
                   f"{np.linalg.norm(tip - reach_pt)*100:.1f} cm to peduncle")
        else:
            txt = msg
        markers.append(client.addUserDebugText(
            txt, [center[0], center[1], center[2] + 0.25],
            textColorRGB=[0, 0, 0] if pos is not None else [1, 0, 0],
            textSize=1.2))

    print(f"\n{len(bunches)} targets. Focus the '{WIN}' window:\n"
          "  sliders (bottom of the window): target, ik seed (regen),\n"
          "    standoff, roll, max aim err, cam up, collision check\n"
          "  mode 0ik/1search: 1 (default) runs the task-space sample/score/IK\n"
          "    pipeline; 0 is the older config-space solver;\n"
          "    'candidate' then scrolls the ranked feasible poses instantly\n"
          "  keys: n / p = next / prev target, r = regenerate IK, q = quit"
          + ("\n  buttons: Ctrl+P opens the Qt panel with prev / next / "
             "regenerate" if _has_buttons else ""))
    last = None
    while True:
        vals = read()
        if last is None or any(
                abs(vals[k] - last[k]) > 1e-6 if not isinstance(vals[k], bool)
                else vals[k] != last[k] for k in vals):
            pos, quat, q, msg = apply(vals)
            redraw(pos, quat, msg, vals)
            last = vals
            print(f"  target {int(vals['target_i'])} "
                  f"standoff {vals['standoff_cm']:.1f}cm "
                  f"roll {vals['roll_deg']:+.0f} ik_seed {vals['ik_seed']} "
                  f"-> {msg}")
        cv2.imshow(WIN, cv2.cvtColor(wrist_view(), cv2.COLOR_RGB2BGR))
        base = base_view()
        if base is not None:
            cv2.imshow("base camera (goal pose)",
                       cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("n"):
            _step_target(1)
        elif key == ord("p"):
            _step_target(-1)
        elif key == ord("r"):
            # New seed => different elbow/wrist re-seeds, so IK can land on
            # another branch at identical parameters.
            _bump_seed()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
