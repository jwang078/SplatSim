#!/usr/bin/env python
"""Segment one body (an asset instance) out of a stage's splat so the simulator
can move it — the robot, a box, an engine — and fill in its block of the
stage.yaml. Three steps, with a manual alignment in the middle (automatic
alignment tends to find odd solutions, so it stays manual):

  1. pcd     sample the URDF at its scan pose into a point cloud you can align
             against the splat, and export the splat as a plain RGB cloud:
                 python scripts/segment_stage_asset.py <stage> --asset robot pcd
             -> data/stages/<stage>/<asset>_urdf_pcd.ply  and  splat_rgb.ply
             Open both in CloudCompare, crop the splat down to the body, align
             (ICP without scale, fix by hand, ICP with scale) and copy the final
             Transformation History matrix.
  2. transform  paste that matrix in:
                 python scripts/segment_stage_asset.py <stage> --asset robot transform matrix.txt
             The first body of a scan defines the scan's transformation
             (splat -> simulator). For every later body the scan is already
             placed, so align the URDF cloud onto the splat instead and pass
             --as-pose: the matrix becomes that body's base pose.
  3. labels  with everything aligned, fit the body's box, label every gaussian
             inside it with the nearest URDF link and save both to the yaml:
                 python scripts/segment_stage_asset.py <stage> --asset robot labels [--show]
             Adjust `urdf_bbox_adjustment` in the yaml and rerun if the box
             clips something the URDF doesn't model (a camera on the wrist, say).

`<asset>` is the instance name under `assets:` in the stage.yaml (the robot
is `robot`; a stage written before `assets:` existed is treated as one body
called robot). `all` runs pcd + labels for a body whose transform is already in.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p

from splatsim.configs import registry
from splatsim.utils.paths import resolve_splatsim_path

SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------- helpers
def entry_for(stage: str, asset: str) -> tuple[str, dict]:
    """(registry name, config) of one body of a stage."""
    name = f"{stage}/{asset}"
    cfg = registry.get(name)
    if cfg is None and asset == "robot":
        name, cfg = stage, registry.get(stage)
    if cfg is None:
        known = sorted(n for n in registry.load_all() if n == stage or n.startswith(stage + "/"))
        raise SystemExit(f"no body {asset!r} in stage {stage!r} (known: {known or 'no such stage'})")
    return name, cfg


def load_body(cfg: dict, gui: bool = False) -> int:
    """The URDF in PyBullet at its base pose and scan pose."""
    p.connect(p.GUI if gui else p.DIRECT)
    urdf = resolve_splatsim_path(cfg["urdf_path"])
    pos = cfg.get("base_position") or [0.0, 0.0, 0.0]
    rpy = cfg.get("base_orientation_rpy") or [0.0, 0.0, 0.0]
    body = p.loadURDF(urdf, basePosition=pos, baseOrientation=p.getQuaternionFromEuler(rpy), useFixedBase=True)
    by_name = {p.getJointInfo(body, j)[1].decode(): j for j in range(p.getNumJoints(body))}
    if cfg.get("scan_pose"):
        for k, v in cfg["scan_pose"].items():
            if k not in by_name:
                raise SystemExit(f"scan_pose names joint {k!r}, which the URDF does not have")
            p.resetJointState(body, by_name[k], float(v))
    else:
        for i, v in enumerate((cfg.get("articulation_config") or {}).get("initial_joint_positions") or []):
            if i + 1 < p.getNumJoints(body):
                p.resetJointState(body, i + 1, float(v))
    return body


def sample_urdf(body: int, imgx: int = 1000, imgy: int = 1000):
    """Points on every link, from depth renders with one link visible at a
    time. Returns (N,3) points and (N,) link indices (-1 = base)."""
    from splatsim.utils.cameras import get_overall_pcd
    links = list(range(-1, p.getNumJoints(body)))
    pts, lab = [], []
    for li in links:
        p.changeVisualShape(body, li, rgbaColor=[1, 1, 1, 0])
    for li in links:
        for lj in links:
            p.changeVisualShape(body, lj, rgbaColor=[1, 1, 1, 1 if lj == li else 0])
        cloud = np.asarray(get_overall_pcd(imgx=imgx, imgy=imgy).points)
        if len(cloud):
            pts.append(cloud)
            lab.append(np.full(len(cloud), li))
    for li in links:
        p.changeVisualShape(body, li, rgbaColor=[1, 1, 1, 1])
    return np.vstack(pts), np.hstack(lab)


def splat_ply(cfg: dict) -> Path:
    if cfg.get("ply_path"):
        return Path(resolve_splatsim_path(cfg["ply_path"]))
    model = resolve_splatsim_path(cfg["model_path"])
    cands = sorted(glob.glob(os.path.join(model, "point_cloud", "iteration_*", "point_cloud.ply")),
                   key=lambda f: int(f.split("iteration_")[-1].split(os.sep)[0]))
    if not cands:
        raise SystemExit(f"no point_cloud.ply under {model}/point_cloud/iteration_*/")
    return Path(cands[-1])


def read_splat(ply: Path):
    """(N,3) gaussian centres and (N,3) RGB in [0,1], in the file's order —
    the same order the simulator loads them in."""
    from plyfile import PlyData
    v = PlyData.read(str(ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if "f_dc_0" in v.data.dtype.names:
        rgb = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1), 0, 1)
    elif "red" in v.data.dtype.names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz, rgb


def write_ply(path: Path, xyz, rgb) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pc)


def parse_matrix(text: str) -> np.ndarray:
    """A 4x4 from a file or a string: CloudCompare's Transformation History
    (4 rows of 4 numbers), JSON, or a flat list of 16."""
    if os.path.exists(text):
        text = Path(text).read_text()
    try:
        m = np.asarray(json.loads(text), dtype=np.float64)
    except (ValueError, json.JSONDecodeError):
        nums = [float(t) for t in text.replace(",", " ").replace("[", " ").replace("]", " ").split()
                if t.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).replace("E", "", 1).replace("+", "", 1).isdigit()
                or t.lstrip("-").replace(".", "", 1).isdigit() or "e" in t.lower()]
        m = np.asarray(nums[-16:], dtype=np.float64) if len(nums) >= 16 else np.asarray(nums)
    m = m.reshape(4, 4)
    return m


def link_colors(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.random((n + 1, 3))


# ------------------------------------------------------------------- steps
def step_pcd(stage: str, asset: str, name: str, cfg: dict, args) -> Path:
    out = registry.entry_dir(name)
    body = load_body(cfg, gui=False)
    X, y = sample_urdf(body)
    colors = link_colors(p.getNumJoints(body))
    pcd_path = out / f"{asset}_urdf_pcd.ply"
    write_ply(pcd_path, X, colors[(y + 1).astype(int)])
    print(f"[{name}] URDF at its scan pose -> {pcd_path}  ({len(X)} points, {p.getNumJoints(body) + 1} links)")
    splat_rgb = out / "splat_rgb.ply"
    if not splat_rgb.exists() or args.force:
        xyz, rgb = read_splat(splat_ply(cfg))
        write_ply(splat_rgb, xyz, rgb)
        print(f"[{name}] splat as a plain RGB cloud -> {splat_rgb}  ({len(xyz)} gaussians)")
    else:
        print(f"[{name}] splat_rgb.ply already there (use --force to rewrite)")
    print("Next: open both in CloudCompare, crop the splat to this body, align (ICP without scale, "
          "adjust by hand, ICP with scale), then\n"
          f"      python scripts/segment_stage_asset.py {stage} --asset {asset} transform <matrix>"
          + ("" if is_first_body(stage, asset) else "  --as-pose"))
    p.disconnect()
    return pcd_path


def is_first_body(stage: str, asset: str) -> bool:
    """True when this body defines the scan's transformation (no other body
    of the stage has one yet / it is the robot)."""
    cfg = registry.get(stage) or {}
    return asset == "robot" or not cfg.get("transformation")


def step_transform(stage: str, asset: str, name: str, cfg: dict, args) -> None:
    M = parse_matrix(args.matrix)
    if args.as_pose:
        pos = np.asarray(cfg.get("base_position") or [0.0, 0.0, 0.0], dtype=np.float64)
        rpy = cfg.get("base_orientation_rpy") or [0.0, 0.0, 0.0]
        R_old = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)
        T_old = np.eye(4); T_old[:3, :3] = R_old; T_old[:3, 3] = pos
        T_new = M @ T_old
        A = T_new[:3, :3]
        scale = float(np.cbrt(abs(np.linalg.det(A))))
        R = A / scale
        q = _quat_from_matrix(R)
        new_rpy = [round(float(v), 6) for v in p.getEulerFromQuaternion(q)]
        new_pos = [round(float(v), 6) for v in T_new[:3, 3]]
        if abs(scale - 1.0) > 0.02:
            print(f"WARNING: the matrix scales by {scale:.3f}; a body's pose cannot carry scale — "
                  f"the URDF is what it is. Dropping the scale.")
        path = registry.write_back(name, {"base_position": new_pos, "base_orientation_rpy": new_rpy})
        print(f"[{name}] base pose <- matrix: position {new_pos}, rpy {new_rpy}  -> {path}")
    else:
        rows = [[round(float(v), 6) for v in row] for row in M]
        path = registry.write_back(stage, {"transformation": {"matrix": rows}})
        print(f"[{stage}] transformation.matrix (splat -> simulator) <- matrix  -> {path}")
    print(f"Next: python scripts/segment_stage_asset.py {stage} --asset {asset} labels --show")


def _quat_from_matrix(R):
    import scipy.spatial.transform as sst
    return sst.Rotation.from_matrix(R).as_quat()   # x, y, z, w


def step_labels(stage: str, asset: str, name: str, cfg: dict, args) -> None:
    if not cfg.get("transformation"):
        raise SystemExit(f"stage {stage!r} has no transformation yet — run `pcd`, align, then `transform`")
    out = registry.entry_dir(name)
    body = load_body(cfg, gui=args.show)
    X, y = sample_urdf(body)
    # the body's box in the simulator frame: the URDF's extent plus the yaml's adjustments
    adj = np.asarray((cfg.get("aabb") or {}).get("urdf_bbox_adjustment") or [[0, 0], [0, 0], [0, 0]], dtype=np.float64)
    # The box is written to the yaml rounded to 4 decimals, and the simulator
    # cuts the gaussians with THAT box, so the labels must be fitted with the
    # rounded box too — every gaussian the loader keeps needs a label.
    box = [[round(float(v), 4) for v in X.min(0) + adj[:, 0]], [round(float(v), 4) for v in X.max(0) + adj[:, 1]]]
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    # the splat into the simulator frame
    xyz, rgb = read_splat(splat_ply(cfg))
    T = np.asarray(cfg["transformation"]["matrix"], dtype=np.float64)
    xyz_sim = xyz @ T[:3, :3].T + T[:3, 3]
    inside = np.where((xyz_sim > lo).all(1) & (xyz_sim < hi).all(1))[0]
    if len(inside) == 0:
        raise SystemExit("no gaussians inside the body's box — is the transform right? (run `pcd` and compare in CloudCompare)")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10).fit(X, y)
    labels = knn.predict(xyz_sim[inside]).astype(np.int32)
    # write: labels (over the gaussians inside the box, in splat order) + key + yaml
    labels_path = out / (Path(cfg["labels_path"]).name if cfg.get("labels_path") else f"{asset}_labels.npy")
    np.save(labels_path, labels)
    classes = {-1: p.getBodyInfo(body)[0].decode()}
    classes.update({j: p.getJointInfo(body, j)[12].decode() for j in range(p.getNumJoints(body))})
    key = registry.write_labels_key(labels_path, classes,
                                    source="scripts/segment_stage_asset.py: KNN against the URDF sampled at the scan pose")
    path = registry.write_back(name, {"aabb": {"bounding_box": box}, "labels_path": labels_path.name})
    counts = {classes[int(v)]: int(c) for v, c in zip(*np.unique(labels, return_counts=True))}
    print(f"[{name}] box {box}\n[{name}] {len(inside)} of {len(xyz)} gaussians inside; per link: {counts}")
    print(f"[{name}] labels -> {labels_path.name} (+ {key.name}); yaml -> {path}")
    cu, cs = X.mean(0), xyz_sim[inside].mean(0)
    print(f"[{name}] alignment check — URDF centroid {np.round(cu, 3).tolist()} vs splat-in-box centroid {np.round(cs, 3).tolist()} "
          f"(|d| = {np.linalg.norm(cu - cs):.3f} m; a few cm is fine, tens of cm means the transform or the box is off)")
    if args.show:
        _show(X, y, xyz_sim[inside], labels, p.getNumJoints(body))
    p.disconnect()


def _show(X, y, S, labels, n_links):
    """Three windows at once: the URDF cloud, the labelled splat, and both
    overlaid — same colour = same link. Closing them all continues."""
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    colors = link_colors(n_links)
    a = o3d.geometry.PointCloud(); a.points = o3d.utility.Vector3dVector(X); a.colors = o3d.utility.Vector3dVector(colors[(y + 1).astype(int)])
    b = o3d.geometry.PointCloud(); b.points = o3d.utility.Vector3dVector(S); b.colors = o3d.utility.Vector3dVector(colors[(labels + 1).astype(int)])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    hint = "3D view — left-drag to rotate, scroll to zoom, right-drag to pan"

    def window(app, title, geoms, x, y):
        w = app.create_window(title, 800, 600, x, y)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(w.renderer)
        mat = rendering.MaterialRecord(); mat.point_size = 3.0
        for i, g in enumerate(geoms):
            scene.scene.add_geometry(f"geometry_{i}", g, mat)
        bounds = geoms[0].get_axis_aligned_bounding_box()
        for g in geoms[1:]:
            bounds += g.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bounds, bounds.get_center())
        label = gui.Label(hint)
        em = w.theme.font_size

        def on_layout(ctx):
            r = w.content_rect
            scene.frame = r
            pref = label.calc_preferred_size(ctx, gui.Widget.Constraints())
            label.frame = gui.Rect(r.x + em, r.y + em, pref.width, pref.height)
        w.set_on_layout(on_layout)
        w.add_child(scene)
        w.add_child(label)

    print("Windows: URDF links (left), splat gaussians coloured by the link they were given (right), both overlaid (below). "
          "Same colours should sit on the same parts. Close all three to continue.")
    app = gui.Application.instance
    app.initialize()
    window(app, "URDF at scan pose", [a, frame], 50, 50)
    window(app, "splat labelled by link", [b, frame], 900, 50)
    window(app, "overlay: URDF + labelled splat", [a, b, frame], 50, 700)
    app.run()


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", help="stage name (data/stages/<stage>/)")
    ap.add_argument("--asset", default="robot", help="instance name under assets: (default robot)")
    ap.add_argument("step", choices=["pcd", "transform", "labels", "all"])
    ap.add_argument("matrix", nargs="?", help="transform: a 4x4 (file or text; CloudCompare's Transformation History pastes fine)")
    ap.add_argument("--as-pose", action="store_true", help="transform: apply the matrix to this body's base pose instead of the scan's transformation")
    ap.add_argument("--show", action="store_true", help="labels: open the Open3D windows to eyeball the result")
    ap.add_argument("--force", action="store_true", help="pcd: rewrite splat_rgb.ply even if it exists")
    args = ap.parse_args()

    name, cfg = entry_for(args.stage, args.asset)
    if not cfg.get("urdf_path"):
        raise SystemExit(f"{name} has no urdf_path / asset — a body needs geometry to segment against")
    if args.step == "pcd":
        step_pcd(args.stage, args.asset, name, cfg, args)
    elif args.step == "transform":
        if not args.matrix:
            raise SystemExit("transform needs the matrix (a file, or the 16 numbers)")
        step_transform(args.stage, args.asset, name, cfg, args)
    elif args.step == "labels":
        step_labels(args.stage, args.asset, name, cfg, args)
    else:
        step_pcd(args.stage, args.asset, name, cfg, args)
        registry.invalidate(); name, cfg = entry_for(args.stage, args.asset)
        step_labels(args.stage, args.asset, name, cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
