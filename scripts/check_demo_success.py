"""Sanity-check the RRT demo quality for a planar env WITHOUT recording/training.

For N random resets it plans the RRT path (the same one env.reset() caches),
executes it in physics, and reports whether the gripper actually reached the
block (check_metrics success) and whether it collided. If success rate isn't
high, the DEMOS are the problem — no policy can learn them — so fix that before
blaming the policy.

Usage:
    python scripts/check_demo_success.py [--task planar_3joint_oracle_simple] [--n 30]

Task must be a registered env (see splatsim/gym_env.py):
    planar_3joint | planar_3joint_oracle | planar_3joint_oracle_simple
"""
import argparse
import numpy as np

from splatsim.gym_env import ENV_REGISTRY, _populate_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="planar_3joint_oracle_simple")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--port", type=int, default=6099)
    args = ap.parse_args()

    _populate_registry()
    if args.task not in ENV_REGISTRY:
        raise SystemExit(f"Unknown task '{args.task}'. Available: {sorted(ENV_REGISTRY)}")
    cls = ENV_REGISTRY[args.task]

    srv = cls(
        host="127.0.0.1", port=args.port, robot_name="planar_3joint",
        use_gripper=True, camera_names=[], render_from_splat=False, headless=True,
    )

    n_success = 0
    n_collision = 0
    dists = []
    for t in range(args.n):
        srv.reset(seed=t)
        path = srv._cached_reset_trajectory  # the RRT demo the policy learns
        if path is None:
            print(f"seed={t}: reset produced NO path (unsolvable scene accepted)")
            continue
        rid = srv.splatsim_robot.sim_id
        # snap to the path start, then execute it in physics like recording does
        for j, qi in zip(range(1, srv.num_dofs() + 1), path[0]):
            srv.pybullet_client.resetJointState(rid, j, float(qi))
        for row in path:
            srv.command_joint_state(srv.splatsim_robot, np.concatenate([row, [0.0]]))
            for _ in range(srv._physics_steps_per_action):
                srv.pybullet_client.stepSimulation()
        m = srv.check_metrics()
        n_success += bool(m["is_success"])
        n_collision += bool(m["in_collision"])
        dists.append(m["distance_to_target_m"])

    dists = np.array(dists)
    print("=" * 56)
    print(f"task={args.task}  n={args.n}")
    print(f"  demo success rate : {n_success}/{args.n} = {100*n_success/max(args.n,1):.0f}%")
    print(f"  ended in collision: {n_collision}/{args.n}")
    if len(dists):
        print(f"  final gripper→block dist (m): mean={dists.mean():.3f} "
              f"max={dists.max():.3f} (tol={srv.pos_tolerance_m})")
    print("=" * 56)
    if n_success < 0.9 * args.n:
        print("⚠️  Demo success < 90% — fix the demos (RRT/execution) before training.")
    else:
        print("✓ Demos are good; a state-only policy should be able to learn them.")


if __name__ == "__main__":
    main()
