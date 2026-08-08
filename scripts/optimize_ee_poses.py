"""Task-space EE-pose optimisation for grape inspection, with IK post-filter.

Pipeline (each stage visualised):
  1. load    grape gaussians (grapes_only.ply) + vegetation cloud, sim frame
  2. sample  EE poses on a shell around the target, aimed at it
  3. score   visibility / gripper clearance / approach corridor / camera-up /
             standoff — arm-agnostic, gripper modelled as spheres
  4. select  diverse top-K (SE(3) non-max suppression)
  5. filter  IK + full-arm collision, using the existing planner utilities
  6. report  what survived and why the rest did not

See splatsim/utils/ee_pose_search.py for why this order (task space first,
IK last) rather than the older sample-IK-then-filter approach.

Usage:
    python scripts/optimize_ee_poses.py --target 3
    python scripts/optimize_ee_poses.py --target 3 --from-below   # grasp-style
    python scripts/optimize_ee_poses.py --target 3 --out viz/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc

from splatsim.configs.env_config import SplatObjectConfig
from splatsim.utils import ee_pose_search as eps
from splatsim.utils import grape_targets as G
from splatsim.utils.rrt_path_utils import check_links_in_collision
from splatsim.utils.splat_ply_io import read_gaussian_ply

GRAPES_PLY = ("/home/jennyw2/code/gaussian-splatting/output/"
              "grape_prop_in_highbay2_images_500_500match_800height_"
              "SIMPLE_RADIALcam/point_cloud/iteration_30000/grapes_only.ply")
SCENE_DIR = Path("data/vine_seg/vine_and_trellis")


def load_hard_points(env_cls, step: int = 1):
    """Point-sample the HARD collision mesh (trellis + thick branches).

    Vertices plus per-face centroids: vertices alone leave gaps across large
    triangles, and a gap in this cloud is a gripper that passes through a
    trellis wire unnoticed. The URDF loads the mesh at identity (it was baked
    into sim frame), so these coordinates need no transform.
    """
    obj = None
    for o in getattr(env_cls.ENV_CONFIG, "objects", []):
        urdf = Path(str(o.urdf_path))
        if urdf.exists():
            for ln in urdf.read_text().splitlines():
                if "<mesh filename=" in ln and "collision" in ln:
                    obj = urdf.parent / ln.split('filename="')[1].split('"')[0]
                    break
        if obj is not None:
            break
    if obj is None or not obj.exists():
        return None
    verts, faces = [], []
    for ln in obj.read_text().splitlines():
        if ln.startswith("v "):
            verts.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    pts = [v]
    if faces:
        f = np.asarray(faces, dtype=int)
        pts.append(v[f].mean(axis=1))
    return np.concatenate(pts)[::step]


def load_scene(client, env_cls):
    """Robot + hard obstacles, kinematics only (no splat, no server)."""
    cfg = SplatObjectConfig(name="robot", splat_name=env_cls.DEFAULT_ROBOT_NAME,
                            randomize_pose=False, load_splat=False)
    rid = client.loadURDF(str(cfg.urdf_path),
                          basePosition=list(cfg.base_position), useFixedBase=True)
    init = list(cfg.articulation_config.initial_joint_positions)
    movable = [j for j in range(client.getNumJoints(rid))
               if client.getJointInfo(rid, j)[2] != pb.JOINT_FIXED]
    for j, q in zip(movable, init):
        client.resetJointState(rid, j, float(q))
    obstacles = []
    for o in getattr(env_cls.ENV_CONFIG, "objects", []):
        try:
            obstacles.append(client.loadURDF(str(o.urdf_path),
                                             basePosition=list(o.base_position),
                                             useFixedBase=True))
        except Exception as exc:                    # noqa: BLE001
            print(f"  (skipped obstacle {o.name!r}: {exc})")
    return rid, obstacles, np.asarray(init[:6], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=0, help="bunch index")
    ap.add_argument("--grapes-ply", default=GRAPES_PLY)
    ap.add_argument("--directions", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--from-below", action="store_true",
                    help="restrict approach to the underside of the bunch — "
                         "the structurally clear side for a hanging cluster, "
                         "and the setting a grasp task would use")
    ap.add_argument("--cam-up", default="inverted",
                    choices=["inverted", "upright", "off"])
    ap.add_argument("--out", default="data/vine_seg/vine_and_trellis/viz")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from splatsim.robots.sim_robot_pybullet_vine import (
        VineGrapeReachPybulletRobotServer as V)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- 1. load the scene
    T = np.asarray(json.loads((SCENE_DIR / "splat_to_sim.json").read_text()),
                   dtype=np.float64)
    grapes = read_gaussian_ply(args.grapes_ply).xyz @ T[:3, :3].T + T[:3, 3]
    field = np.load(V.SOFT_COST_NPZ, allow_pickle=False)
    veg = np.asarray(field["points"], dtype=np.float64)
    print(f"grapes {len(grapes):,} pts | vegetation {len(veg):,} pts")

    bunches = G.load_targets(V.GRAPE_TARGETS_JSON)
    bunch = bunches[args.target % len(bunches)]
    centre = np.asarray(bunch["center"], dtype=np.float64)
    # Only the fruit belonging to THIS bunch is the visibility target.
    from scipy.spatial import cKDTree
    gtree = cKDTree(grapes)
    target_pts = grapes[np.asarray(
        gtree.query_ball_point(centre, 0.5 * max(bunch["extent"]) + 0.04))]
    print(f"target bunch {args.target}: {len(target_pts)} grape points")

    client = bc.BulletClient(pb.DIRECT)
    robot_id, obstacles, q_home = load_scene(client, V)
    ee_link = next(j for j in range(client.getNumJoints(robot_id))
                   if client.getJointInfo(robot_id, j)[12].decode()
                   == "wrist_camera_link")
    joint_indices = list(range(1, 7))
    grip_links = list(range(7, client.getNumJoints(robot_id)))
    gc, gr = eps.gripper_spheres(client, robot_id, ee_link, grip_links)
    print(f"gripper model: {len(gc)} spheres, radii "
          f"{gr.min()*100:.1f}-{gr.max()*100:.1f} cm")

    # Occluders exclude the target bunch itself — see build_clouds. Without
    # this the fruit blocks the view of itself and visibility collapses.
    d_to_target, _ = cKDTree(target_pts).query(veg)
    occluders = veg[d_to_target > 0.03]
    print(f"occluders {len(occluders):,} pts "
          f"({len(veg)-len(occluders):,} dropped as target fruit)")
    hard_pts = load_hard_points(V)
    if hard_pts is None:
        print("  WARNING: no hard collision mesh found — the pose stage will "
              "NOT see the trellis/branches (IK stage still will)")
    else:
        print(f"hard geometry {len(hard_pts):,} pts (trellis + thick branches)")
    clouds = eps.build_clouds(grapes, veg, gc, gr, occluder_pts=occluders,
                              hard_pts=hard_pts)
    spec = eps.SearchSpec(
        n_directions=args.directions, top_k=args.top_k,
        camera_up_world={"inverted": (0.0, 0.0, -1.0),
                         "upright": (0.0, 0.0, 1.0),
                         "off": None}[args.cam_up],
        tip_offset_m=float(V.GRIPPER_TIP_OFFSET_M),
        base_xyz=tuple(np.asarray(
            client.getBasePositionAndOrientation(robot_id)[0])),
    )
    if args.from_below:
        spec.direction_hint = (0.0, 0.0, -1.0)
        spec.direction_max_angle_deg = 60.0

    # -------------------------------------------------- 2/3. sample + score
    pos, rot, appr = eps.sample_poses(centre, spec)
    print(f"sampled {len(pos)} poses (after reachability prefilter)")
    if len(pos) == 0:
        raise SystemExit("no candidate poses — loosen reach_range/standoff")
    # Gate on HARD geometry first: the trellis and thick branches are rigid,
    # so a gripper intersecting them is infeasible, not merely low-scoring.
    hard_hit = np.array([eps.gripper_hits_hard(pos[i], rot[i], clouds)
                         for i in range(len(pos))])
    print(f"gripper-vs-hard-geometry: {int(hard_hit.sum())} of {len(pos)} "
          f"poses rejected (trellis / thick branches)")
    scores = np.full(len(pos), -1.0); terms = []
    for i in range(len(pos)):
        if hard_hit[i]:
            terms.append({k: 0.0 for k in ("visibility", "clearance",
                                           "approach", "camera_up",
                                           "standoff", "hard")})
            continue
        sc, t = eps.score_pose(pos[i], rot[i], appr[i], clouds, spec, target_pts)
        scores[i] = sc; terms.append(t)
    if not (scores >= 0).any():
        raise SystemExit("every candidate pose intersects hard geometry")
    _ok = scores[scores >= 0]
    print(f"scores (feasible only): min {_ok.min():.3f}  "
          f"median {np.median(_ok):.3f}  max {_ok.max():.3f}")

    # ------------------------------------------------------- 4. select top-K
    ok_idx = np.flatnonzero(scores >= 0)
    keep = ok_idx[eps.select_diverse(pos[ok_idx], rot[ok_idx],
                                     scores[ok_idx], spec)]
    print(f"selected {len(keep)} diverse candidates")

    # -------------------------------------------------------- 5. IK filter
    def arm_collides(q):
        return bool(check_links_in_collision(
            robot_id, joint_indices, q, obstacles,
            self_collision_skip_pairs=[tuple(x) for x in
                                       V.SELF_COLLISION_SKIP_PAIRS],
            obstacle_clearance=0.01))

    from scipy.spatial.transform import Rotation
    movable = [j for j in range(client.getNumJoints(robot_id))
               if client.getJointInfo(robot_id, j)[2] != pb.JOINT_FIXED]
    lo, hi, rng_ = [], [], []
    for j in movable:
        info = client.getJointInfo(robot_id, j)
        a, b = info[8], info[9]
        if a > b:
            a, b = -2 * np.pi, 2 * np.pi
        lo.append(a); hi.append(b); rng_.append(b - a)

    survivors, reasons = [], {"ok": 0, "ik_pos": 0, "ik_rot": 0, "collision": 0}
    for i in keep:
        quat = Rotation.from_matrix(rot[i]).as_quat()
        best = None
        for attempt in range(6):        # a few seeds: IK is local
            seed = q_home if attempt == 0 else np.random.uniform(
                [lo[movable.index(j)] for j in joint_indices],
                [hi[movable.index(j)] for j in joint_indices])
            for j, qq in zip(joint_indices, seed):
                client.resetJointState(robot_id, j, float(qq))
            sol = client.calculateInverseKinematics(
                robot_id, ee_link, pos[i].tolist(), list(quat),
                lowerLimits=lo, upperLimits=hi, jointRanges=rng_,
                restPoses=[client.getJointState(robot_id, j)[0] for j in movable],
                maxNumIterations=400, residualThreshold=1e-9)
            q = np.asarray(sol[:6])
            for j, qq in zip(joint_indices, q):
                client.resetJointState(robot_id, j, float(qq))
            st = client.getLinkState(robot_id, ee_link,
                                     computeForwardKinematics=True)
            perr = float(np.linalg.norm(np.asarray(st[4]) - pos[i]))
            rerr = float(np.degrees(
                Rotation.from_matrix(
                    rot[i].T @ Rotation.from_quat(st[5]).as_matrix()).magnitude()))
            if perr > 0.01:
                best = best or "ik_pos"; continue
            if rerr > 15.0:
                best = "ik_rot"; continue
            if arm_collides(q):
                best = "collision"; continue
            survivors.append((int(i), q, perr, rerr)); best = "ok"; break
        reasons[best if best else "ik_pos"] += 1
    print(f"IK filter: {reasons['ok']} reachable / {len(keep)} "
          f"(rejected: {reasons['ik_pos']} unreachable, {reasons['ik_rot']} "
          f"orientation, {reasons['collision']} collision)")

    # ------------------------------------------------------ 6. visualisation
    def draw(ax, idxs, title, colour_by=None):
        ax.scatter(veg[::40, 0], veg[::40, 1], veg[::40, 2], s=0.4,
                   c="#9ccc9c", alpha=0.25, linewidths=0)
        ax.scatter(grapes[::4, 0], grapes[::4, 1], grapes[::4, 2], s=1.2,
                   c="#7b1030", alpha=0.5, linewidths=0)
        ax.scatter(*target_pts[::2].T, s=3, c="#e01050", linewidths=0)
        if len(idxs):
            p = pos[idxs]
            a = np.array([rot[i] @ np.asarray(spec.aim_axis_local)
                          for i in idxs])
            cvals = (scores[idxs] if colour_by is None else colour_by)
            q = ax.quiver(p[:, 0], p[:, 1], p[:, 2], a[:, 0], a[:, 1], a[:, 2],
                          length=0.07, normalize=True, linewidth=1.4,
                          cmap="viridis", array=cvals)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, c=cvals, cmap="viridis")
        ax.scatter(*centre, s=90, marker="*", c="cyan", edgecolors="k", zorder=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        lim = 0.45
        ax.set_xlim(centre[0]-lim, centre[0]+lim)
        ax.set_ylim(centre[1]-lim, centre[1]+lim)
        ax.set_zlim(centre[2]-lim, centre[2]+lim)
        ax.view_init(elev=18, azim=-70)

    fig = plt.figure(figsize=(19, 10))
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    draw(ax1, np.arange(len(pos))[np.argsort(-scores)[:250]],
         f"1. sampled poses ({len(pos)}), top 250 by score\n"
         "arrow = camera axis, colour = score")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    draw(ax2, keep, f"2. diverse top-{len(keep)} after SE(3) NMS")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    surv_idx = np.array([s[0] for s in survivors], dtype=int)
    draw(ax3, surv_idx, f"3. IK-FEASIBLE: {len(survivors)} of {len(keep)}\n"
                        "(full-arm collision checked)")

    ax4 = fig.add_subplot(2, 3, 4)
    names = ["visibility", "clearance", "approach", "camera_up", "standoff",
             "hard"]
    data = np.array([[t[n] for n in names] for t in terms])[ok_idx]
    _pos_of = {int(v): k for k, v in enumerate(ok_idx)}
    ax4.boxplot([data[:, k] for k in range(len(names))], tick_labels=names,
                showfliers=False)
    if len(keep):
        _k = [_pos_of[int(i)] for i in keep]
        ax4.plot(range(1, len(names) + 1),
                 [data[_k][:, k].mean() for k in range(len(names))],
                 "o-", c="tab:orange", label="top-K mean")
    if len(surv_idx):
        _s = [_pos_of[int(i)] for i in surv_idx]
        ax4.plot(range(1, len(names) + 1),
                 [data[_s][:, k].mean() for k in range(len(names))],
                 "s-", c="tab:green", label="IK-feasible mean")
    ax4.set_title("4. score terms: all samples vs survivors", fontsize=9)
    ax4.legend(fontsize=7); ax4.tick_params(labelrotation=20)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(scores[ok_idx], bins=40, color="#bbb", label="all samples")
    if len(keep):
        ax5.hist(scores[keep], bins=20, color="tab:orange", label="top-K")
    if len(surv_idx):
        ax5.hist(scores[surv_idx], bins=20, color="tab:green",
                 label="IK-feasible")
    ax5.set_title("5. score distribution", fontsize=9)
    ax5.set_xlabel("score"); ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(2, 3, 6)
    labels = ["reachable", "unreachable", "orientation", "arm collision",
              "hard geom\n(pre-score)"]
    vals = [reasons["ok"], reasons["ik_pos"], reasons["ik_rot"],
            reasons["collision"], int(hard_hit.sum())]
    ax6.bar(labels, vals, color=["tab:green", "#bbb", "tab:orange", "tab:red",
                                 "tab:purple"])
    ax6.set_title("6. rejections: gripper-vs-trellis gate, then IK", fontsize=9)
    ax6.tick_params(labelrotation=15)
    for i, v in enumerate(vals):
        ax6.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    mode = "from-below (grasp-style)" if args.from_below else "all directions"
    fig.suptitle(f"EE-pose optimisation then IK filter — bunch {args.target}, "
                 f"{mode}, cam-up={args.cam_up}", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"30_ee_pose_search_bunch{args.target}" \
                    f"{'_below' if args.from_below else ''}.png"
    fig.savefig(out, dpi=105)

    if survivors:
        best = max(survivors, key=lambda s: scores[s[0]])
        print(f"\nbest feasible pose: score={scores[best[0]]:.3f} "
              f"pos={np.round(pos[best[0]], 3).tolist()} "
              f"(IK pos err {best[2]*1000:.1f} mm, rot err {best[3]:.1f} deg)")
        print("  terms: " + "  ".join(
            f"{k}={v:.2f}" for k, v in terms[best[0]].items()))

    # Absolute path, printed LAST so it is the final thing on screen and can
    # be copied straight into an image viewer regardless of the cwd the
    # script was launched from.
    print(f"\nsaved visualization:\n  {out.resolve()}")


if __name__ == "__main__":
    main()
