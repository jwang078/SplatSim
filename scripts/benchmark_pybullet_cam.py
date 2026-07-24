"""Micro-benchmark the PyBullet getCameraImage call to find render speedups.

_render_pybullet_camera does one getCameraImage per camera. The knobs that
actually move its cost:
  * shadow            — shadows are a SECOND render pass; shadow=0 ~halves it.
  * renderer          — ER_BULLET_HARDWARE_OPENGL (GPU) vs ER_TINY_RENDERER (CPU).
  * flags             — ER_NO_SEGMENTATION_MASK skips the seg buffer (already on).
  * resolution        — fewer pixels = less fill + less readback.

This times the raw call under each combination against the live scene so you can
see the win and the image cost (shadow=0 changes pixels; the rest shouldn't).

Usage:
    python scripts/benchmark_pybullet_cam.py --iters 60
"""
import argparse
import time

import numpy as np
import pybullet as p


def _bench(client, W, H, view, proj, iters, warmup, **kw):
    for _ in range(warmup):
        client.getCameraImage(W, H, view, proj, **kw)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        client.getCameraImage(W, H, view, proj, **kw)
        out.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(out)


def _line(label, ms, base=None):
    tag = "" if base is None else f"   {base / ms.mean():4.1f}x vs shadow-on-GL"
    return (f"  {label:<42} {ms.mean():7.2f} {np.median(ms):7.2f} "
            f"{ms.min():7.2f} {ms.max():7.2f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_name", default="robot_iphone_w_engine_curtain")
    ap.add_argument("--cameras", default="base_rgb,wrist_rgb")
    ap.add_argument("--wrist_cam_ver", type=int, default=2)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--port", type=int, default=6077)
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c]
    from splatsim.robots.sim_robot_pybullet_small_engine import (
        UprightRobotSmallEngineNewPybulletRobotServer as Server,
    )
    print("Loading scene…")
    srv = Server(
        port=args.port, host="127.0.0.1",
        serve_mode=Server.SERVE_MODES.INTERACTIVE,
        camera_names=cameras, robot_name=args.robot_name, cam_i=3,
        use_gripper=True, image_resize_modes=["letterbox"],
        wrist_cam_ver=args.wrist_cam_ver,
    )
    client = srv.pybullet_client
    GL = p.ER_BULLET_HARDWARE_OPENGL
    TINY = p.ER_TINY_RENDERER
    NOSEG = p.ER_NO_SEGMENTATION_MASK

    for cam in cameras:
        # reuse the production view/proj builder
        if srv._is_wrist_camera(cam):
            vp = srv._splatsim_camera_to_pybullet_view(srv.get_wrist_camera())
        elif srv.base_camera is not None:
            vp = srv._splatsim_camera_to_pybullet_view(srv.base_camera)
        else:
            vp = srv._fixed_pybullet_view_proj()
        view, proj, W, H = vp

        print(f"\n{'=' * 92}\n{cam}  render {W}x{H}\n{'=' * 92}")
        print(f"  {'config':<42} {'mean':>7} {'med':>7} {'min':>7} {'max':>7}")

        # current production config (baseline)
        base = _bench(client, W, H, view, proj, args.iters, args.warmup,
                      renderer=GL, flags=NOSEG)
        print(_line("GL, shadow=default, NOSEG  [CURRENT]", base))

        # shadow off
        print(_line("GL, shadow=0, NOSEG", _bench(
            client, W, H, view, proj, args.iters, args.warmup,
            renderer=GL, flags=NOSEG, shadow=0), base.mean()))

        # shadow off + no depth request via flags where supported
        print(_line("GL, shadow=0, NOSEG+lightdir off", _bench(
            client, W, H, view, proj, args.iters, args.warmup,
            renderer=GL, flags=NOSEG, shadow=0,
            lightDirection=[0, 0, 1]), base.mean()))

        # half resolution (sanity: how much is fill vs fixed overhead)
        hW, hH = max(1, W // 2), max(1, H // 2)
        print(_line(f"GL, shadow=0, {hW}x{hH} (half-res)", _bench(
            client, hW, hH, view, proj, args.iters, args.warmup,
            renderer=GL, flags=NOSEG, shadow=0), base.mean()))

        # CPU tiny renderer for reference
        print(_line("TINY (CPU), shadow=0, NOSEG", _bench(
            client, W, H, view, proj, args.iters, args.warmup,
            renderer=TINY, flags=NOSEG, shadow=0), base.mean()))

        # ── image cost of shadow=0 (does it change pixels?) ──
        _, _, a, _, _ = client.getCameraImage(W, H, view, proj, renderer=GL, flags=NOSEG, shadow=1)
        _, _, b, _, _ = client.getCameraImage(W, H, view, proj, renderer=GL, flags=NOSEG, shadow=0)
        a = np.reshape(np.asarray(a, np.uint8), (H, W, 4))[..., :3].astype(int)
        b = np.reshape(np.asarray(b, np.uint8), (H, W, 4))[..., :3].astype(int)
        d = np.abs(a - b)
        print(f"\n  shadow on->off image diff: max={d.max()}/255  mean={d.mean():.2f}/255  "
              f"changed px={100 * (d.max(-1) > 0).mean():.1f}%")


if __name__ == "__main__":
    main()
