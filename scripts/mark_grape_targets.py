"""Manually mark grape clusters on the rendered splat.

Colour-based detection cannot find every bunch on this prop: the hue windows
key on purple/red fruit, so GREEN grapes are indistinguishable from leaves and
are missed, while reddish twigs and highlights slip in as false positives.
Until the real detector lands, this lets you point at the fruit yourself.

You click on a bunch in the splat render; the tool converts that pixel into a
3D point by projecting the scene's soft (vegetation) point cloud into the same
image and taking the nearest-to-camera point under your cursor. It then grows
a cluster around that seed and summarises it with
`splat_segmentation.cluster_stats` — the SAME function the automatic
clusterer uses — so a hand-marked bunch is indistinguishable downstream from
a detected one, peduncle and all.

The view follows PyBullet's debug camera (DebugModes.ROTATE_BASE_CAM), so
orbit/zoom in the pybullet window and the splat render follows; that is how
you reach bunches hidden from the default viewpoint.

Controls (in the "mark grapes" window):
    left click   mark a cluster at the cursor
    u            undo the last mark
    c            clear all marks
    [ / ]        shrink / grow the cluster radius
    s            save to the scene's grape_targets.json
    q / Esc      quit (without saving)

Usage:
    python scripts/mark_grape_targets.py
    python scripts/mark_grape_targets.py --radius 0.06
    python scripts/mark_grape_targets.py --load    # start from existing targets
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pybullet as pb

from splatsim.utils import grape_targets as G
from splatsim.utils import splat_segmentation as seg
from splatsim.utils.goal_pose import project_points

DEFAULT_ENV = "splatsim.robots.sim_robot_pybullet_vine:VineGrapeReachPybulletRobotServer"


def load_env_class(spec: str):
    mod_name, _, cls_name = spec.partition(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def load_soft_points(env_cls):
    """Vegetation points in SIM frame, with class ids when available.

    `<scene>_cost_field_sim.npz` stores the points already transformed into
    sim coordinates, and `<scene>_soft_cost.npz` stores the matching class
    ids in the same row order — so no PLY read and no splat->sim transform is
    needed here.
    """
    sim_npz = Path(env_cls.SOFT_COST_NPZ)
    simf = np.load(sim_npz, allow_pickle=False)
    pts = np.asarray(simf["points"], dtype=np.float64)
    class_id = None
    soft_npz = sim_npz.with_name(sim_npz.name.replace("_cost_field_sim", "_soft_cost"))
    if soft_npz.exists():
        soft = np.load(soft_npz, allow_pickle=False)
        if soft["weight"].shape == simf["weights"].shape and np.allclose(
                soft["weight"], simf["weights"]):
            class_id = np.asarray(soft["class_id"])
    return pts, class_id


def pick_point(pts_sim, uv_click, cam, rectify_zoom=1.0, pick_px=12):
    """Pixel -> 3D vegetation point.

    Projects every soft point into the current image and returns the one
    NEAREST THE CAMERA within ``pick_px`` of the click — i.e. the visible
    surface the user pointed at, not something behind it. Returns None when
    nothing vegetation-like is under the cursor.
    """
    uv, valid = project_points(pts_sim, cam, rectify_zoom=rectify_zoom)
    du = uv[:, 0] - uv_click[0]
    dv = uv[:, 1] - uv_click[1]
    near = valid & ((du * du + dv * dv) <= pick_px ** 2)
    if not near.any():
        return None
    cand = np.flatnonzero(near)
    rot = np.asarray(cam.camera.R, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(cam.camera.T, dtype=np.float64).reshape(3)
    eye = -rot @ trans
    depth = np.linalg.norm(pts_sim[cand] - eye, axis=1)
    return pts_sim[cand[int(np.argmin(depth))]]


def grow_cluster(tree, pts_sim, seed, radius, class_id=None, min_points=5):
    """Cluster = soft points within ``radius`` of ``seed``, summarised by the
    SHARED ``splat_segmentation.cluster_stats`` so a hand-marked bunch is
    indistinguishable downstream from a detected one (peduncle included).

    Adds ``manual: True`` and, when class ids are available, ``grape_frac`` —
    the fraction of the cluster the colour segmenter thought was fruit. That
    is a useful sanity signal, not a gate: green grapes read as foliage, which
    is exactly why manual marking exists.
    """
    idx = tree.query_ball_point(np.asarray(seed, dtype=np.float64), radius)
    if len(idx) < min_points:
        return None
    idx = np.asarray(idx)
    stats = seg.cluster_stats(pts_sim[idx])
    stats["manual"] = True
    if class_id is not None:
        stats["grape_frac"] = float((class_id[idx] == seg.GRAPE).mean())
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--radius", type=float, default=0.05,
                    help="cluster growth radius around the clicked seed (m)")
    ap.add_argument("--pick-px", type=int, default=12,
                    help="pixel search radius for the click -> point lookup")
    ap.add_argument("--load", action="store_true",
                    help="preload the existing grape_targets.json so you can "
                         "add to / prune it rather than starting empty")
    ap.add_argument("--out", default=None,
                    help="output json (default: the scene's "
                         "grape_targets_manual.json, which automation never "
                         "writes and the env prefers over detector output)")
    ap.add_argument("--port", type=int, default=6089)
    args = ap.parse_args()

    import cv2

    env_cls = load_env_class(args.env)
    # Save to the MANUAL file by default, never to the detector's output:
    # regen_grape_targets.py rewrites grape_targets.json freely, and the env
    # prefers the manual one anyway (grape_targets.resolve_targets_json).
    out_path = Path(args.out) if args.out else (
        Path(env_cls.GRAPE_TARGETS_JSON).parent / G.MANUAL_TARGETS_NAME)
    pts_sim, class_id = load_soft_points(env_cls)
    print(f"{len(pts_sim):,} soft points loaded")

    from splatsim.configs.mode_config import RenderMode
    from splatsim.robots.sim_robot_pybullet_base import FISHEYE_RECTIFY_ZOOM

    print("constructing the env server (loads the splat scene)...")
    server = env_cls(
        port=args.port, host="127.0.0.1",
        serve_mode=env_cls.SERVE_MODES.INTERACTIVE,
        robot_name=env_cls.DEFAULT_ROBOT_NAME,
        camera_names=["base_rgb", "wrist_rgb"],
        render_mode=RenderMode.SPLAT,
        # Base camera follows the pybullet debug camera, so orbiting the
        # pybullet window re-frames the splat render and you can reach
        # bunches occluded from the default viewpoint.
        debug_mode=env_cls.DEBUG_MODES.ROTATE_BASE_CAM,
        headless=False, show_control_gui=False,
    )
    client = server.pybullet_client
    for _flag in (pb.COV_ENABLE_RGB_BUFFER_PREVIEW,
                  pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                  pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
        client.configureDebugVisualizer(_flag, 0)

    marks: list[dict] = []
    if args.load and out_path.exists():
        marks = json.loads(out_path.read_text())
        print(f"loaded {len(marks)} existing target(s) from {out_path}")

    state = {"radius": float(args.radius), "img": None, "cam": None,
             "click": None, "dirty": True}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "mark grapes"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    def render():
        """Native-resolution base render + its camera (see tune_goal_pose for
        why native rather than the letterboxed observation)."""
        server.get_observations()          # assembles scene_gaussian
        cam = server.get_pybullet_debug_camera_as_splat_camera()
        img = server.render_image("base_rgb")
        a = np.asarray(img)
        if a.ndim == 3 and a.shape[0] in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        return a[:, :, :3], cam

    from scipy.spatial import cKDTree
    tree = cKDTree(pts_sim)


    print("\nclick bunches in the 'mark grapes' window; orbit in the pybullet "
          "window to re-frame.\n  u=undo  c=clear  [ ]=radius  s=save  q=quit")
    while True:
        img, cam = render()
        if state["click"] is not None and cam is not None:
            seed = pick_point(pts_sim, state["click"], cam,
                              rectify_zoom=FISHEYE_RECTIFY_ZOOM,
                              pick_px=args.pick_px)
            state["click"] = None
            if seed is None:
                print("  no vegetation point under the cursor — try again")
            else:
                st = grow_cluster(tree, pts_sim, seed, state["radius"],
                                  class_id=class_id)
                if st is None:
                    print(f"  too few points within {state['radius']*100:.0f} cm "
                          "— grow the radius with ']'")
                else:
                    marks.append(st)
                    gf = st.get("grape_frac")
                    print(f"  + bunch {len(marks)-1}: n={st['n_points']} "
                          f"center={np.round(st['center'],3).tolist()}"
                          + (f"  grape_frac={gf:.2f}" if gf is not None else ""))

        # overlay current marks
        if cam is not None and marks:
            allpts, kinds = [], []
            for i, m in enumerate(marks):
                allpts.append(m["center"]); kinds.append((i, True))
                allpts.append(m["peduncle"]); kinds.append((i, False))
            uv, valid = project_points(np.asarray(allpts), cam,
                                       rectify_zoom=FISHEYE_RECTIFY_ZOOM)
            for k, ((i, is_c), ok) in enumerate(zip(kinds, valid)):
                if not ok:
                    continue
                x, y = int(round(uv[k, 0])), int(round(uv[k, 1]))
                if is_c:
                    cv2.circle(img, (x, y), 6, (0, 255, 255), 2)
                    cv2.putText(img, f"{i} n={marks[i]['n_points']}",
                                (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), (0, 165, 255), 2)
                    cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), (0, 165, 255), 2)
        cv2.putText(img, f"marks={len(marks)}  radius={state['radius']*100:.0f}cm",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(win, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("quit without saving" if marks else "quit")
            break
        if key == ord("u") and marks:
            marks.pop()
            print(f"  undo -> {len(marks)} mark(s)")
        elif key == ord("c"):
            marks.clear()
            print("  cleared")
        elif key == ord("]"):
            state["radius"] = min(state["radius"] + 0.01, 0.5)
            print(f"  radius {state['radius']*100:.0f} cm")
        elif key == ord("["):
            state["radius"] = max(state["radius"] - 0.01, 0.01)
            print(f"  radius {state['radius']*100:.0f} cm")
        elif key == ord("s"):
            if not marks:
                print("  nothing to save")
                continue
            # Largest first, matching cluster_labeled_points' ordering, so
            # TARGET_BUNCH_INDEX keeps meaning "the biggest bunch".
            ordered = sorted(marks, key=lambda m: -m["n_points"])
            out_path.write_text(json.dumps(ordered, indent=1))
            print(f"  saved {len(ordered)} target(s) -> {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
