"""Profile prep_image_rendering() — the splat's per-frame pose->gaussian step.

benchmark_render.py showed prep_image_rendering is ~82% of a splat frame (133 of
162 ms), i.e. the splat RENDERER is not the bottleneck; posing the gaussians is.
This breaks that number down PER SCENE OBJECT so you can see where it goes.

Method: monkey-patch the three helpers prep_image_rendering calls
(get_transformation_list / transform_means for the articulated robot,
transform_object for everything else) with timing wrappers, then run the real
method unmodified. No duplicated loop, so this can't drift from production code.

It also tracks whether each object actually MOVED between frames. Anything that
never moves is being re-transformed every frame for nothing — usually the single
biggest, easiest win (the background splat is typically the largest buffer in
the scene and is completely static).

Usage:
    python scripts/profile_prep_render.py --iters 20
    python scripts/profile_prep_render.py --iters 20 --jitter 0   # static arm too
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch

import splatsim.robots.sim_robot_pybullet_base as base_mod


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _obj_of(args, kwargs):
    """Pull the SplatSimObject out of either call convention."""
    o = kwargs.get("splatsim_obj")
    if o is None and args:
        o = args[0]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_name", default="robot_iphone_w_engine_curtain")
    ap.add_argument("--cameras", default="base_rgb,wrist_rgb")
    ap.add_argument("--wrist_cam_ver", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--port", type=int, default=6096)
    ap.add_argument("--jitter", type=float, default=0.05)
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c]

    from splatsim.robots.sim_robot_pybullet_small_engine import (
        UprightRobotSmallEngineNewPybulletRobotServer as Server,
    )
    print("Loading scene (splats)…")
    srv = Server(
        port=args.port, host="127.0.0.1",
        serve_mode=Server.SERVE_MODES.INTERACTIVE,
        camera_names=cameras, robot_name=args.robot_name, cam_i=3,
        use_gripper=True, image_resize_modes=["letterbox"],
        wrist_cam_ver=args.wrist_cam_ver,
    )

    # ── instrument the three helpers prep_image_rendering calls ───────────────
    timings = defaultdict(list)      # "name::fn" -> [ms]
    collecting = {"on": False}

    def wrap(orig, fn_label):
        def wrapper(*a, **kw):
            if not collecting["on"]:
                return orig(*a, **kw)
            o = _obj_of(a, kw)
            name = getattr(getattr(o, "config", None), "name", "?")
            _sync()
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            _sync()
            timings[f"{name}::{fn_label}"].append((time.perf_counter() - t0) * 1e3)
            return r
        return wrapper

    base_mod.get_transformation_list = wrap(base_mod.get_transformation_list, "get_transformation_list")
    base_mod.transform_means = wrap(base_mod.transform_means, "transform_means")
    base_mod.transform_object = wrap(base_mod.transform_object, "transform_object")

    rid = srv.splatsim_robot.sim_id
    n = srv.num_dofs()
    q0 = np.asarray(srv.get_joint_state()[:n], dtype=float)
    rng = np.random.default_rng(0)
    obs = {}
    poses = defaultdict(set)  # object name -> set of observed poses (movement check)

    def jitter():
        if args.jitter > 0:
            q = q0 + rng.uniform(-args.jitter, args.jitter, size=n)
            for j, qi in zip(range(1, n + 1), q):
                srv.pybullet_client.resetJointState(rid, j, float(qi))
        obs.clear()
        obs.update(srv.get_observations(render_images=False))
        for o in srv.splatsim_objects:
            nm = o.config.name
            pos = obs.get(nm + "_position")
            quat = obs.get(nm + "_orientation")
            if pos is not None:
                poses[nm].add((tuple(np.round(pos, 6)), tuple(np.round(quat, 6))))

    # ── run ───────────────────────────────────────────────────────────────────
    for _ in range(args.warmup):
        jitter()
        with torch.no_grad():
            srv.prep_image_rendering(data=obs)
    _sync()

    collecting["on"] = True
    frame_ms = []
    for _ in range(args.iters):
        jitter()
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            srv.prep_image_rendering(data=obs)
        _sync()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
    collecting["on"] = False

    # ── aggregate per object ──────────────────────────────────────────────────
    offsets = srv._scene_gaussian_offsets
    per_obj = defaultdict(float)
    per_obj_fns = defaultdict(list)
    for key, ms in timings.items():
        name, fn = key.split("::")
        per_obj[name] += float(np.mean(ms))
        per_obj_fns[name].append(fn)

    total_frame = float(np.mean(frame_ms))
    print("\n" + "=" * 96)
    print(f"prep_image_rendering PROFILE  (iters={args.iters}, jitter={args.jitter} rad)")
    print(f"measured frame total: {total_frame:.2f} ms  ({1000 / total_frame:.1f} FPS)")
    print("=" * 96)
    print(f"  {'object':<26} {'gaussians':>11} {'ms/frame':>9} {'%':>6}  {'moves?':>7}  helper")
    print("  " + "-" * 92)

    static_ms = 0.0
    static_g = 0
    for name, ms in sorted(per_obj.items(), key=lambda kv: -kv[1]):
        s, e = offsets.get(name, (0, 0))
        ng = e - s
        moved = len(poses.get(name, ())) > 1
        # the robot is articulated: its pose changes via joints, not base pose
        if name == srv.splatsim_robot.config.name:
            moved = args.jitter > 0
        if not moved:
            static_ms += ms
            static_g += ng
        print(f"  {name:<26} {ng:>11,} {ms:>9.2f} {100 * ms / total_frame:>5.1f}%  "
              f"{'yes' if moved else 'NO':>7}  {'+'.join(sorted(set(per_obj_fns[name])))}")

    # per-helper detail: separates CPU-side pose queries from GPU transforms,
    # which need completely different fixes.
    print("  " + "-" * 92)
    print(f"  {'per-helper detail':<26} {'':>11} {'ms/frame':>9} {'%':>6}")
    for key, ms in sorted(timings.items(), key=lambda kv: -float(np.mean(kv[1]))):
        m = float(np.mean(ms))
        print(f"    {key:<40} {m:>9.2f} {100 * m / total_frame:>5.1f}%")

    accounted = sum(per_obj.values())
    print("  " + "-" * 92)
    print(f"  {'accounted':<26} {sum(e - s for s, e in offsets.values()):>11,} "
          f"{accounted:>9.2f} {100 * accounted / total_frame:>5.1f}%")
    print(f"  {'unaccounted (loop/slicing)':<26} {'':>11} "
          f"{total_frame - accounted:>9.2f} {100 * (total_frame - accounted) / total_frame:>5.1f}%")

    print("\n" + "=" * 96)
    if static_ms > 0:
        print(f"WASTE: {static_ms:.2f} ms/frame ({100 * static_ms / total_frame:.0f}% of prep, "
              f"{static_g:,} gaussians) is spent re-transforming objects that NEVER MOVED.")
        print(f"       Caching those would take prep {total_frame:.0f} -> ~{total_frame - static_ms:.0f} ms/frame.")
    else:
        print("Every object moved this run — no trivially cacheable work found.")
    print("=" * 96)


if __name__ == "__main__":
    main()
