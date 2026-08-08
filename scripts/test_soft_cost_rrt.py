"""End-to-end smoke test for soft-cost-aware RRT planning.

Loads the repo's planar_3joint robot in DIRECT PyBullet with the synthetic
vine (scaled 0.35x, trunk URDF as hard obstacle + soft-cost field) sitting in
the arm's sweep corridor, then plans the same EE goal twice:

  run A: soft_cost_mode="off"    (historical behavior)
  run B: soft_cost_mode="score"  (soft term added to path selection)

Both trajectories are graded with the SAME field; B's foliage exposure should
be <= A's (statistically — RRT is stochastic, so we report, not hard-assert).

Also runs the binary-env regression check: with no `soft_cost` key in the env
config the scorer must return exactly the base strategy's score.

Visualization: viz/13_softcost_rrt_paths.png — EE traces of both runs over
the cost field's max-projections (red=off, blue=score).

Usage:  python scripts/test_soft_cost_rrt.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

DATA = Path("data/vine_seg/synthetic")
ROBOT_URDF = "splatsim/robot_definitions/urdf/planar_3joint.urdf"
JOINT_INDICES = [1, 2, 3]
EE_LINK = 15  # wrist_camera_link (COM == link frame; planner FK gate matches)
VINE_SCALE = 0.35
VINE_POS = [0.66, 0.095, -0.28]  # scaled vine sits in the arm's xz sweep arc
Q_START = np.array([-1.0, 1.4, -0.6])   # EE high (0.64, 0, 0.41)
Q_GOAL_SEED = np.array([1.2, -0.6, 0.2])  # EE low  (0.49, 0, -0.40)


def make_planner(client_id, robot_id, mode, field_payload):
    from splatsim.utils.rrt_to_goal import RRTToGoalPlanner

    planner = RRTToGoalPlanner(
        pb_client=client_id,
        robot_id=robot_id,
        joint_indices=list(JOINT_INDICES),
        ee_link_index=EE_LINK,
        num_dofs=3,
        fps=30,
        num_ik_candidates=8,
        num_path_candidates_per_ik=6,
        max_path_attempts_per_ik=10,
        path_perturbation_scale=0.15,
        rrt_smooth_iterations=30,
        soft_cost_mode=mode,
        soft_cost_weight=5.0,
    )
    env_config = {
        "objects": [{
            "__type__": "SplatObjectConfig",
            "name": "vine_trunk",
            "urdf_path": str((DATA / "vine.urdf").resolve()),
            "base_position": VINE_POS,
            "current_position": VINE_POS,
            "current_quat": [0, 0, 0, 1],
            "current_scale": [VINE_SCALE] * 3,
        }],
    }
    if field_payload:
        env_config["soft_cost"] = field_payload
    planner.load_obstacles(env_config)
    return planner


def ee_trace(client, robot_id, traj):
    pts = []
    for q in traj:
        for j, qi in zip(JOINT_INDICES, q):
            client.resetJointState(robot_id, j, float(qi))
        pts.append(client.getLinkState(robot_id, EE_LINK,
                                       computeForwardKinematics=True)[4])
    return np.asarray(pts)


def main():
    import pybullet_utils.bullet_client as bc

    from splatsim.utils.soft_cost_field import SoftCostField

    client = bc.BulletClient(pb.DIRECT)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = client.loadURDF(ROBOT_URDF, useFixedBase=True)

    s = VINE_SCALE
    payload = {
        "npz_path": str((DATA / "vine_soft_cost.npz").resolve()),
        "transform": [[s, 0, 0, VINE_POS[0]],
                      [0, s, 0, VINE_POS[1]],
                      [0, 0, s, VINE_POS[2]],
                      [0, 0, 0, 1]],
    }
    grading_field = SoftCostField.from_config(payload)

    for j, qi in zip(JOINT_INDICES, Q_GOAL_SEED):
        client.resetJointState(robot_id, j, float(qi))
    goal_state = client.getLinkState(robot_id, EE_LINK, computeForwardKinematics=True)
    # planner's IK + FK-accuracy check use link COM pose (indices 0/1),
    # not the URDF link frame (4/5) — match that convention
    ee_target, ee_quat = np.array(goal_state[0]), np.array(goal_state[1])
    print(f"goal EE pos={np.round(ee_target, 3)} (FK of mirrored config)")

    results = {}
    for mode in ("off", "score"):
        random.seed(0)
        np.random.seed(0)
        planner = make_planner(client._client, robot_id, mode, payload)
        for j, qi in zip(JOINT_INDICES, Q_START):
            client.resetJointState(robot_id, j, float(qi))
        # production callers retry on RRTPlanningError (stochastic planner);
        # mirror that here
        from splatsim.utils.rrt_to_goal import RRTPlanningError

        traj = None
        for attempt in range(5):
            try:
                traj, _ = planner.plan(Q_START.copy(), ee_target, ee_quat,
                                       q_goal_bias=Q_GOAL_SEED.copy())
                break
            except RRTPlanningError as e:
                print(f"  attempt {attempt + 1} failed ({e}); retrying")
        assert traj is not None, f"planning failed after retries (mode={mode})"
        trace = ee_trace(client, robot_id, traj)
        # grade with the planner's own integral (same field for both modes)
        planner._soft_cost_field = grading_field
        exposure = planner._path_soft_cost(traj)
        results[mode] = {"traj": traj, "trace": trace, "exposure": exposure}
        print(f"mode={mode:5s}: {traj.shape[0]} waypoints, "
              f"foliage exposure integral = {exposure:.4f}")

    # ---- deterministic scorer A/B (no RRT stochasticity): a dense blob is
    # placed on the straight start->goal chord in JOINT space; path A goes
    # straight through it, path B detours via a via-config. Cost-blind must
    # prefer A (shorter); cost-aware must prefer B (cleaner).
    q_mid = 0.5 * (Q_START + Q_GOAL_SEED)
    planner_ab = make_planner(client._client, robot_id, "score", payload)
    for j, qi in zip(JOINT_INDICES, q_mid):
        client.resetJointState(robot_id, j, float(qi))
    blob_center = np.array(client.getLinkState(
        robot_id, EE_LINK, computeForwardKinematics=True)[0])
    rng = np.random.default_rng(3)
    blob_xyz = rng.normal(blob_center, [0.06, 0.03, 0.06], (4000, 3))
    blob_npz = DATA / "blob_soft_cost.npz"
    np.savez_compressed(blob_npz, xyz=blob_xyz,
                        weight=np.ones(len(blob_xyz)),
                        class_id=np.ones(len(blob_xyz)))
    from splatsim.utils.soft_cost_field import SoftCostField as SCF

    planner_ab._soft_cost_field = SCF.from_config({"npz_path": str(blob_npz)})
    path_through = np.stack([Q_START, q_mid, Q_GOAL_SEED])
    # find a via-config whose path is LONGER (in EE arc) but CLEANER (in blob
    # cost) than the through path — the classic detour trade-off
    arc_thru = planner_ab._path_ee_arc_length(path_through)
    cost_thru = planner_ab._path_soft_cost(path_through)
    path_around = None
    for off in ([0.0, 1.5, 0.0], [0.0, -1.5, 0.0], [-0.8, 0.0, 0.0],
                [0.8, 0.0, 0.0], [0.0, 1.2, -1.6], [0.0, -1.2, 1.6]):
        cand = np.stack([Q_START, q_mid + np.array(off), Q_GOAL_SEED])
        if (planner_ab._path_ee_arc_length(cand) > arc_thru
                and planner_ab._path_soft_cost(cand) < 0.5 * cost_thru):
            path_around = cand
            break
    assert path_around is not None, "no longer-but-cleaner via-config found"
    scores = {}
    for mode in ("off", "score"):
        planner_ab._soft_cost_mode = mode
        scores[mode] = {
            "through": planner_ab._score_candidate(path_through, None),
            "around": planner_ab._score_candidate(path_around, None),
        }
        pick = min(scores[mode], key=scores[mode].get)
        print(f"scorer A/B mode={mode:5s}: through={scores[mode]['through']:.3f} "
              f"around={scores[mode]['around']:.3f} -> picks '{pick}'")
    assert min(scores["off"], key=scores["off"].get) == "through", \
        "cost-blind should prefer the shorter through-blob path"
    assert min(scores["score"], key=scores["score"].get) == "around", \
        "cost-aware should prefer the detour around the blob"
    print("scorer A/B PASS: soft term flips the selection as designed")

    # ---- regression: no soft_cost key => scorer identical to base strategy
    planner_plain = make_planner(client._client, robot_id, "score", None)
    assert planner_plain._soft_cost_field is None, "field must be None w/o payload"
    fake_path = np.stack([Q_START, Q_START + 0.3])
    base = planner_plain._path_ee_arc_length(fake_path)
    scored = planner_plain._score_candidate(fake_path, None)
    assert scored == base, f"regression: {scored} != base {base}"
    print("regression check PASS: no-payload scorer == base strategy score")

    delta = results["off"]["exposure"] - results["score"]["exposure"]
    print(f"\nexposure off->score delta: {delta:+.4f} "
          f"({'IMPROVED' if delta > 0 else 'no improvement this seed'})")

    # ------------------------------ viz ------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    views = [((0, 2), 1, "x", "z"), ((0, 1), 2, "x", "y")]
    for ax, ((h, v), proj_axis, hl, vl) in zip(axes, views):
        img = grading_field.grid.max(axis=proj_axis)
        if proj_axis == 1:
            img = img.T
        ax.imshow(img if proj_axis == 1 else img.T, origin="lower",
                  extent=grading_field._extent(proj_axis), cmap="inferno",
                  alpha=0.9)
        for mode, color in (("off", "red"), ("score", "deepskyblue")):
            t = results[mode]["trace"]
            ax.plot(t[:, h], t[:, v], color=color, lw=2,
                    label=f"{mode} (exposure={results[mode]['exposure']:.3f})")
            ax.scatter(*t[0, [h, v]], c=color, marker="o", s=60, zorder=5)
            ax.scatter(*t[-1, [h, v]], c=color, marker="*", s=140, zorder=5)
        ax.set_xlabel(f"{hl} (m)")
        ax.set_ylabel(f"{vl} (m)")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("13 RRT EE traces vs soft-cost field  "
                 "(o=start, *=goal; red=cost-blind, blue=cost-aware)")
    out = DATA / "viz" / "13_softcost_rrt_paths.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"viz -> {out}")


if __name__ == "__main__":
    main()
