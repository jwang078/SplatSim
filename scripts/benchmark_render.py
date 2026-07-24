"""Measure the image-observation update rate: SPLAT vs PYBULLET camera.

The GUI shows which render mode is active but never times it, so "is the splat
actually the bottleneck?" is guesswork. This answers it directly by running both
backends against the SAME live scene and reporting ms/frame + FPS.

Three levels of measurement, because they answer different questions:

  1. end-to-end  get_observations(render_images=True)
       The REAL update rate — what the GUI / trajectory recording / eval sees.
       Includes per-camera render AND the resize to 224 for every
       image_resize_mode. This is the number to quote.
  2. per-camera  render_image() vs _render_pybullet_camera()
       The raw renderer cost per camera, isolating base vs wrist (the wrist
       fisheye also pays an undistort pass on the splat path).
  3. splat prep  get_curr_link_states() + prep_image_rendering()
       Per-FRAME (not per-camera) splat overhead that transforms the gaussians
       to the current robot pose. Often the real cost — it's why the splat can
       be slow even with one camera.

CUDA is asynchronous, so every timed region is wrapped in cuda.synchronize();
without it the splat looks ~free because the work is still queued on the GPU.
Joints are jittered each iteration so per-pose caches can't inflate results.

Usage (small engine, the splat scene):
    python scripts/benchmark_render.py --robot_name robot_iphone_w_engine_curtain \
        --wrist_cam_ver 2 --iters 50

    # base camera only, more samples
    python scripts/benchmark_render.py --cameras base_rgb --iters 100
"""
import argparse
import time

import numpy as np
import torch

from splatsim.configs.mode_config import RenderMode


def _sync():
    """Wait for queued CUDA work — mandatory for honest GPU timings."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters, warmup, jitter=None):
    """Return a list of per-call wall times in ms."""
    for _ in range(warmup):
        if jitter:
            jitter()
        fn()
        _sync()
    out = []
    for _ in range(iters):
        if jitter:
            jitter()
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _fmt(label, ms, width=34):
    a = np.asarray(ms)
    mean = a.mean()
    return (f"  {label:<{width}} {mean:8.2f} {np.median(a):8.2f} "
            f"{np.percentile(a, 95):8.2f} {a.min():8.2f} {a.max():8.2f} "
            f"{1000.0 / mean:8.1f}")


HEADER = (f"  {'':<34} {'mean':>8} {'median':>8} {'p95':>8} "
          f"{'min':>8} {'max':>8} {'FPS':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_name", default="robot_iphone_w_engine_curtain")
    ap.add_argument("--cameras", default="base_rgb,wrist_rgb",
                    help="comma-separated camera names to render")
    ap.add_argument("--resize_modes", default="letterbox,stretch",
                    help="comma-separated; more modes = more resize cost per frame")
    ap.add_argument("--wrist_cam_ver", type=int, default=2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=6099)
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="radians of random joint motion between frames (0 = static scene)")
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c]
    resize_modes = [m for m in args.resize_modes.split(",") if m]

    from splatsim.robots.sim_robot_pybullet_small_engine import (
        UprightRobotSmallEngineNewPybulletRobotServer as Server,
    )
    print(f"Loading scene (splats)… cameras={cameras} resize_modes={resize_modes}")
    srv = Server(
        port=args.port, host="127.0.0.1",
        serve_mode=Server.SERVE_MODES.INTERACTIVE,
        camera_names=cameras, robot_name=args.robot_name, cam_i=3,
        use_gripper=True, image_resize_modes=resize_modes,
        wrist_cam_ver=args.wrist_cam_ver,
    )

    rid = srv.splatsim_robot.sim_id
    n = srv.num_dofs()
    q0 = np.asarray(srv.get_joint_state()[:n], dtype=float)
    rng = np.random.default_rng(0)

    # prep_image_rendering() reads each object's "<name>_position" out of an
    # observations dict, so keep a fresh one around. Refreshed inside jitter(),
    # which runs BEFORE the timer starts, so this bookkeeping is never timed.
    obs = {}

    def jitter():
        """Perturb the arm so gaussian transforms / caches are recomputed each
        frame, as they would be during a real rollout, then refresh the
        object-pose dict the splat prep consumes. Both are untimed."""
        if args.jitter > 0:
            q = q0 + rng.uniform(-args.jitter, args.jitter, size=n)
            for j, qi in zip(range(1, n + 1), q):
                srv.pybullet_client.resetJointState(rid, j, float(qi))
        obs.clear()
        obs.update(srv.get_observations(render_images=False))

    results = {}

    # ── 1. end-to-end update rate, per render mode ────────────────────────────
    for mode in (RenderMode.SPLAT, RenderMode.PYBULLET):
        srv._apply_render_mode(mode)
        results[f"e2e:{mode.value}"] = _time(
            lambda: srv.get_observations(render_images=True),
            args.iters, args.warmup, jitter,
        )

    # ── 2. raw per-camera render cost ─────────────────────────────────────────
    srv._apply_render_mode(RenderMode.SPLAT)
    for cam in cameras:
        # splat: prep once per frame, then render this camera
        def splat_one(cam=cam):
            with torch.no_grad():
                srv.prep_image_rendering(data=obs)
                srv.render_image(camera_name=cam)
        results[f"splat:{cam}"] = _time(splat_one, args.iters, args.warmup, jitter)
        results[f"pybullet:{cam}"] = _time(
            lambda cam=cam: srv._render_pybullet_camera(cam),
            args.iters, args.warmup, jitter,
        )

    # ── 3. splat per-frame prep (pose -> gaussians), camera-independent ───────
    def prep_only():
        with torch.no_grad():
            srv.prep_image_rendering(data=obs)
    results["splat:prep_image_rendering"] = _time(prep_only, args.iters, args.warmup, jitter)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"RENDER BENCHMARK  (iters={args.iters}, warmup={args.warmup}, "
          f"jitter={args.jitter} rad, cuda_sync=on)")
    print(f"cameras={cameras}  resize_modes={resize_modes}  "
          f"wrist_cam_ver={args.wrist_cam_ver}")
    print("=" * 92)

    print("\n[1] END-TO-END get_observations(render_images=True)  <- the real update rate")
    print(HEADER)
    for mode in ("splat", "pybullet"):
        print(_fmt(f"{mode}", results[f"e2e:{mode}"]))
    s, p = np.mean(results["e2e:splat"]), np.mean(results["e2e:pybullet"])
    print(f"\n  -> pybullet is {s / p:.1f}x faster end-to-end "
          f"({1000 / p:.0f} FPS vs {1000 / s:.0f} FPS)")

    print("\n[2] RAW PER-CAMERA RENDER")
    print(HEADER)
    for cam in cameras:
        print(_fmt(f"splat    {cam}  (incl. prep)", results[f"splat:{cam}"]))
        print(_fmt(f"pybullet {cam}", results[f"pybullet:{cam}"]))

    print("\n[3] SPLAT PER-FRAME PREP (pose -> gaussians, camera-independent)")
    print(HEADER)
    print(_fmt("prep_image_rendering", results["splat:prep_image_rendering"]))
    prep = np.mean(results["splat:prep_image_rendering"])
    print(f"\n  -> prep is {100 * prep / s:.0f}% of the splat end-to-end frame time")
    print("=" * 92)


if __name__ == "__main__":
    main()
