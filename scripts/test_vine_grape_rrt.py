"""Cost-aware RRT test on the real vine: UR5 gripper-to-grape reaches.

Builds the exact scene the vine env describes — same UR5 (sisbot.urdf at the
small-engine mount, initial q from objects.yaml), vine collision URDF at the
origin, soft-cost field payload — by serializing
VineGrapeReachPybulletRobotServer.ENV_CONFIG the same way the env publishes
it over ZMQ, then feeding it to RRTToGoalPlanner.load_obstacles. For each
reachable grape bunch it plans once per soft_cost_mode (off / score /
guided) and reports the foliage-exposure integral of the returned
trajectories. "off" and "score" generate identical candidate sets
(cost-blind) and differ only in which candidate wins; "guided" makes
generation itself cost-aware (T-RRT transition test + cost-gated
smoothing).

Outputs (data/vine_seg/vine_and_trellis/viz/):
  14_grape_rrt_paths.png    EE traces over cost-field projections, per bunch
  15_grape_final_pose.png   robot at the winning goal config near the bunch

Usage:  python scripts/test_vine_grape_rrt.py [--bunches 0 1 2] [--weight 3.0]
        [--modes off score guided]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc

from splatsim.configs.env_config import SplatObjectConfig
from splatsim.robots.sim_robot_pybullet_small_engine import (
    SmallEnginePybulletRobotServer,
)
from splatsim.robots.sim_robot_pybullet_vine import (
    VineGrapeReachPybulletRobotServer,
    _ROBOT_BASE_POS,
)
from splatsim.utils import grape_targets
from splatsim.utils.rrt_to_goal import RRTPlanningError, RRTToGoalPlanner
from splatsim.utils.soft_cost_field import SoftCostField

VIZ = Path("data/vine_seg/vine_and_trellis/viz")


def serialize_env_config(cfg) -> dict:
    """Mirror PybulletRobotServerBase.get_env_config's object serialization
    (asdict + __type__) so the planner sees the same dict the env publishes."""
    objects = []
    for o in cfg.objects:
        d = dataclasses.asdict(o)
        d["__type__"] = type(o).__name__
        objects.append(d)
    out = {"name": cfg.name, "objects": objects}
    if cfg.soft_cost:
        out["soft_cost"] = cfg.soft_cost
    return out


def load_ur5(client):
    """Load the small-engine UR5 exactly as objects.yaml describes it."""
    robot_cfg = SplatObjectConfig(
        name="robot", splat_name=VineGrapeReachPybulletRobotServer.DEFAULT_ROBOT_NAME,
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

    ee_link = None
    for j in range(client.getNumJoints(robot_id)):
        if client.getJointInfo(robot_id, j)[12].decode() == "wrist_camera_link":
            ee_link = j
            break
    assert ee_link is not None, "wrist_camera_link not found in robot URDF"
    joint_indices = list(range(1, 7))  # 6 arm joints; 0 is the world joint
    q_start = np.array(init_q[:6])
    return robot_id, joint_indices, ee_link, q_start


def make_planner(client_id, robot_id, joint_indices, ee_link, mode, weight):
    return RRTToGoalPlanner(
        pb_client=client_id, robot_id=robot_id,
        joint_indices=joint_indices, ee_link_index=ee_link,
        num_dofs=6, fps=30,
        num_ik_candidates=12,
        num_path_candidates_per_ik=5,
        max_path_attempts_per_ik=8,
        path_perturbation_scale=0.1,
        rrt_smooth_iterations=30,
        self_collision_skip_pairs=list(
            SmallEnginePybulletRobotServer.SELF_COLLISION_SKIP_PAIRS
        ),
        # grasp goals put fingers within mm of the vine by design
        ik_skip_gripper_obstacle_pairs=True,
        # anti-wobble: round canopy-constrained corners, equalize path speed
        elastic_smooth_passes=30,
        uniform_path_speed=True,
        segment_at_sharp_corners=False,
        # CHOMP-lite trajopt OFF: its min-distance FD gradient (12 distance
        # queries per waypoint per pass, each a 1 m-radius getClosestPoints
        # + stepSimulation against the 40k-tri concave vine mesh) takes
        # MINUTES-TO-HOURS per call on this scene. Matches the pre-trajopt
        # conditions of the recorded exposure baselines.
        trajopt_passes=0,
        soft_cost_mode=mode,
        soft_cost_weight=weight,
    )


def ee_trace(client, robot_id, joint_indices, ee_link, traj):
    pts = []
    for q in traj:
        for j, qi in zip(joint_indices, q):
            client.resetJointState(robot_id, j, float(qi))
        pts.append(client.getLinkState(robot_id, ee_link,
                                       computeForwardKinematics=True)[4])
    return np.asarray(pts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bunches", type=int, nargs="*", default=[0, 1, 2, 5])
    ap.add_argument("--modes", nargs="*", default=["off", "score", "guided"],
                    choices=["off", "score", "guided"],
                    help="soft_cost_modes to A/B; off+score are cost-blind "
                    "generation, guided is cost-aware generation")
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--standoff", type=float, default=0.10)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--start-shift", type=float, default=0.0,
                    help="radians added to the shoulder-pan joint of the "
                    "start config; +-0.8 places the EE on the far side of "
                    "the near canopy so the plan must sweep across it")
    ap.add_argument("--approach", choices=["front", "top", "side"],
                    default="front",
                    help="approach direction: front = from the robot column "
                    "(usually clean), top = down through the upper canopy, "
                    "side = from +x at bunch height")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    client = bc.BulletClient(pb.DIRECT)
    robot_id, joint_indices, ee_link, q_start = load_ur5(client)
    q_start = q_start.copy()
    q_start[0] += args.start_shift

    # one vine body for goal-generation collision checks (the planners load
    # their own copies via load_obstacles; identical static geometry)
    from splatsim.utils.rrt_path_utils import check_links_in_collision

    goal_vine = client.loadURDF(
        "data/vine_seg/vine_and_trellis/vine_and_trellis.urdf",
        useFixedBase=True)
    # Fingers are checked against the mesh too: grapes/foliage are not in
    # the hard mesh (only trunk + trellis), so a finger-clear goal here is
    # exactly what the planner's final q_goal gate accepts — skipping finger
    # pairs let finger-through-wire grasps through that RRT then rejected.

    def goal_config_collides(q):
        return bool(check_links_in_collision(
            robot_id, joint_indices, q, [goal_vine],
            self_collision_skip_pairs=[tuple(p) for p in
                SmallEnginePybulletRobotServer.SELF_COLLISION_SKIP_PAIRS],
            obstacle_clearance=0.01,
        ))

    env_cfg = VineGrapeReachPybulletRobotServer.ENV_CONFIG
    env_dict = serialize_env_config(env_cfg)
    field = SoftCostField.from_config(env_cfg.soft_cost)
    bunches = grape_targets.load_targets(
        VineGrapeReachPybulletRobotServer.GRAPE_TARGETS_JSON
    )

    results = {}  # (bunch, mode) -> dict
    for bi in args.bunches:
        if bi >= len(bunches):
            continue
        center = np.array(bunches[bi]["center"])
        if args.approach == "top":
            from_point = center + np.array([0.0, 0.0, 1.0])
        elif args.approach == "side":
            from_point = center + np.array([1.0, 0.0, 0.0])
        else:  # front
            from_point = np.array([_ROBOT_BASE_POS[0], _ROBOT_BASE_POS[1],
                                   center[2]])
        try:
            pos, quat, q_seed = grape_targets.reachable_approach_pose(
                client, robot_id, ee_link, joint_indices, center,
                standoff=args.standoff, from_point=from_point,
                aim_axis_local=VineGrapeReachPybulletRobotServer.GRIPPER_AIM_AXIS,
                collision_fn=goal_config_collides,
                tool_tip_offset=VineGrapeReachPybulletRobotServer.GRIPPER_TIP_OFFSET_M,
            )
        except ValueError as e:
            print(f"\n=== bunch {bi}: UNREACHABLE — {e}")
            continue
        print(f"\n=== bunch {bi}: center={np.round(center,3).tolist()} "
              f"goal={np.round(pos,3).tolist()} "
              f"(dist from base {np.linalg.norm(center - _ROBOT_BASE_POS):.2f} m)")
        for mode in args.modes:
            random.seed(0)
            np.random.seed(0)
            planner = make_planner(client._client, robot_id, joint_indices,
                                   ee_link, mode, args.weight)
            planner.load_obstacles(env_dict)
            traj = None
            fail = ""
            for attempt in range(args.attempts):
                try:
                    for j, qi in zip(joint_indices, q_start):
                        client.resetJointState(robot_id, j, float(qi))
                    traj, _ = planner.plan(q_start.copy(), pos, quat,
                                           q_goal_bias=q_seed.copy())
                    break
                except RRTPlanningError as e:
                    fail = str(e)
            if traj is None:
                print(f"  mode={mode:5s}: PLANNING FAILED ({fail[:80]})")
                continue
            planner._soft_cost_field = field  # grade both modes identically
            exposure = planner._path_soft_cost(traj)
            trace = ee_trace(client, robot_id, joint_indices, ee_link, traj)
            results[(bi, mode)] = {"traj": traj, "trace": trace,
                                   "exposure": exposure, "goal": pos}
            print(f"  mode={mode:5s}: {traj.shape[0]} wp, "
                  f"foliage exposure = {exposure:.4f}")

    # ------------------------------- viz -------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planned = sorted({bi for bi, _ in results})
    if planned:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        for ax, (h, v, proj_axis, hl, vl) in zip(
            axes, [(0, 2, 1, "x", "z"), (0, 1, 2, "x", "y")]
        ):
            img = field.grid.max(axis=proj_axis)
            img = img.T if proj_axis == 1 else img.T
            ax.imshow(field.grid.max(axis=proj_axis).T, origin="lower",
                      extent=field._extent(proj_axis), cmap="inferno", alpha=0.85)
            for bi in planned:
                for mode, color, style in (("off", "red", "--"),
                                           ("score", "deepskyblue", "-"),
                                           ("guided", "lime", "-")):
                    r = results.get((bi, mode))
                    if r is None:
                        continue
                    t = r["trace"]
                    ax.plot(t[:, h], t[:, v], style, color=color, lw=1.8,
                            label=f"bunch {bi} {mode} "
                                  f"(exp={r['exposure']:.3f})")
                    ax.scatter(*r["goal"][[h, v]], c="cyan", marker="*",
                               s=150, zorder=6, edgecolors="black",
                               linewidths=0.5)
            ax.scatter(*_ROBOT_BASE_POS[[h, v]], c="white", marker="s", s=80,
                       zorder=6, edgecolors="black", label="robot base")
            ax.set_xlabel(f"{hl} (m)")
            ax.set_ylabel(f"{vl} (m)")
        handles, lbls = axes[0].get_legend_handles_labels()
        uniq = dict(zip(lbls, handles))
        axes[0].legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper left")
        fig.suptitle("14 UR5 EE traces to grape bunches over soft-cost field "
                     "(red dashed = off, blue = score, green = guided, "
                     "star = goal)")
        out = VIZ / "14_grape_rrt_paths.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"\nviz -> {out}")

        # final pose render: first planned bunch, most-cost-aware mode's winner
        bi = planned[0]
        r = (results.get((bi, "guided")) or results.get((bi, "score"))
             or results.get((bi, "off")))
        for j, qi in zip(joint_indices, r["traj"][-1]):
            client.resetJointState(robot_id, j, float(qi))
        target = r["goal"]
        view = client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target.tolist(), distance=1.6, yaw=140,
            pitch=-15, roll=0, upAxisIndex=2)
        proj = client.computeProjectionMatrixFOV(fov=55, aspect=4 / 3,
                                                 nearVal=0.05, farVal=10)
        w, h_, rgb, _, _ = client.getCameraImage(1280, 960, view, proj,
                                                 renderer=pb.ER_TINY_RENDERER)
        out2 = VIZ / "15_grape_final_pose.png"
        plt.imsave(out2, np.reshape(rgb, (h_, w, 4))[:, :, :3].astype(np.uint8))
        print(f"viz -> {out2}")

    print("\nsummary (exposure integrals, lower = less foliage brushed):")
    for bi in planned:
        parts = []
        for mode in args.modes:
            e = results.get((bi, mode), {}).get("exposure")
            parts.append(f"{mode}={e:.4f}" if e is not None
                         else f"{mode}=FAILED")
        off = results.get((bi, "off"), {}).get("exposure")
        best_mode, best = None, None
        for mode in args.modes:
            e = results.get((bi, mode), {}).get("exposure")
            if e is not None and (best is None or e < best):
                best_mode, best = mode, e
        verdict = ""
        if off is not None and best is not None and best_mode != "off":
            verdict = (f" -> {best_mode} IMPROVED" if best < off - 1e-6
                       else " -> no improvement over off")
        print(f"  bunch {bi}: " + "  ".join(parts) + verdict)


if __name__ == "__main__":
    main()
