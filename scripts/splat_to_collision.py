"""Make a collision body from a piece of a scan: a gaussian PLY (the body's
gaussians, cropped out of the stage's splat) -> watertight mesh + fixed-base
URDF + the stage.yaml that registers it, with alignment visualizations at
every step. This is the splat-first path — no URDF to start from — used for
the grape vine and its trellis; the URDF-first path is
scripts/segment_stage_asset.py.

Backends:
  splat-transform (default) PlayCanvas CLI (npm i -g @playcanvas/splat-transform);
                  runs `--collision-mesh smooth` and imports the .collision.glb.
  voxel           in-repo voxel-face mesher: occupancy grid from the gaussian
                  centers, emit faces between occupied/empty voxel neighbors ->
                  watertight blocky mesh. No external tools.
                  (Same idea as splat-transform's `--collision-mesh faces`.)

Outputs (under --outdir, which should be the build's folder,
data/stages/<scan>/segmentations/<build>/):
  <name>_collision.obj        collision mesh, in the scan's frame — or in SIM
                              frame if --transform is given (baking; the old
                              behaviour, kept for reproducing existing builds)
  <name>.urdf                 fixed-base URDF wrapping the mesh (concave)
  stage.yaml                  registers the build: urdf_path + collision_frame
                              (splat, or sim when baked). The scan's
                              splat->sim matrix is applied at load from the
                              parent stage.yaml, so prefer NOT baking.
  viz/10_mesh_overlay.png     mesh cross-sections overlaid on trunk points —
                              THE alignment check (mesh must hug red points)
  viz/11_mesh_render.png      shaded open3d render of the mesh (if EGL works)

Usage:
  python scripts/splat_to_collision.py \
      data/stages/vine_scene/segmentations/vine_and_trellis/vine_and_trellis_trunk_hard.ply \
      --outdir data/stages/vine_scene/segmentations/vine_and_trellis
      [--backend voxel|splat-transform] [--voxel-size 0.012] [--dilate 1]
      [--transform path/to/4x4.json]   # bake into sim frame (not recommended)

NOTE: --voxel-size is in the input's units. Unbaked, that is the SCAN frame,
which for the highbay scan is ~4x sim scale (transformation has scale ~0.25).
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splatsim.utils.splat_ply_io import read_gaussian_ply

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{obj_rel}" scale="1 1 1"/></geometry>
      <material name="trunk_brown"><color rgba="0.45 0.30 0.15 1.0"/></material>
    </visual>
    <collision concave="yes">
      <geometry><mesh filename="{obj_rel}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""


# --------------------------------------------------------------- voxel mesh
_FACES = [  # (axis, direction, 4 corner offsets in CCW order seen from outside)
    (0, -1, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    (0, +1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    (1, -1, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    (1, +1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    (2, -1, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    (2, +1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
]


def voxel_face_mesh(points: np.ndarray, voxel: float, dilate: int = 0,
                    close: int = 0):
    """Watertight blocky mesh over occupied voxels. Returns (verts, tris).

    ``close`` runs morphological closing (dilate N then erode N) on the
    occupancy grid first: bridges occlusion gaps up to ~2*N voxels between
    nearby fragments WITHOUT inflating the final geometry (unlike ``dilate``,
    which grows it and is meant as a safety margin)."""
    origin = points.min(axis=0) - voxel * (close + 1)
    idx = np.floor((points - origin) / voxel).astype(int)
    shape = idx.max(axis=0) + 2 + close
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    if close > 0:
        from scipy import ndimage

        n_before = int(ndimage.label(occ)[1])
        occ = ndimage.binary_closing(occ, iterations=close)
        n_after = int(ndimage.label(occ)[1])
        print(f"morphological closing x{close}: "
              f"{n_before} -> {n_after} connected fragments")

    for _ in range(dilate):  # optional safety-margin growth by one voxel
        grown = occ.copy()
        grown[1:] |= occ[:-1]
        grown[:-1] |= occ[1:]
        grown[:, 1:] |= occ[:, :-1]
        grown[:, :-1] |= occ[:, 1:]
        grown[:, :, 1:] |= occ[:, :, :-1]
        grown[:, :, :-1] |= occ[:, :, 1:]
        occ = grown

    padded = np.zeros(np.array(occ.shape) + 2, dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = occ

    verts: dict = {}
    tris: list = []

    def vid(key):
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    occupied = np.argwhere(occ)
    for cell in occupied:
        pc = cell + 1
        for axis, dirn, corners in _FACES:
            nb = pc.copy()
            nb[axis] += dirn
            if padded[tuple(nb)]:
                continue  # neighbor occupied -> internal face
            ids = [vid(tuple(cell + off)) for off in corners]
            tris.append([ids[0], ids[1], ids[2]])
            tris.append([ids[0], ids[2], ids[3]])

    vkeys = np.array(sorted(verts, key=verts.get))
    v = origin + vkeys * voxel
    return v, np.asarray(tris, dtype=np.int64)


def write_obj(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"# collision mesh: {len(verts)} verts, {len(tris)} tris\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in tris:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ------------------------------------------------------ splat-transform path
def run_splat_transform(input_ply: Path, outdir: Path, voxel: float):
    exe = shutil.which("splat-transform")
    if exe is None:
        raise RuntimeError(
            "splat-transform not on PATH (npm i -g @playcanvas/splat-transform); "
            "use --backend voxel instead"
        )
    out_voxel = outdir / "st_output.voxel.json"
    cmd = [exe, str(input_ply), "--voxel-params", f"{voxel},0.1",
           "--collision-mesh", "smooth", str(out_voxel)]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("GPU voxelization failed; retrying with -g cpu ...")
        subprocess.run(cmd + ["-g", "cpu"], check=True)
    candidates = list(outdir.glob("*.collision.glb"))
    if not candidates:
        raise RuntimeError(f"expected a .collision.glb next to {out_voxel}")
    glb = candidates[0]

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(glb))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"open3d read 0 vertices from {glb}")
    # splat-transform writes the glb rotated 180deg about z relative to the
    # source PLY frame (verified numerically against the input gaussians;
    # the 10_mesh_overlay.png viz will catch it if this ever changes).
    verts = np.asarray(mesh.vertices) * np.array([-1.0, -1.0, 1.0])
    tris = np.asarray(mesh.triangles)[:, ::-1]  # mirror pair flips winding
    return verts, tris


# ------------------------------------------------------------------- viz
def plot_mesh_overlay(verts, tris, trunk_pts, path, transform_note=""):
    """Orthographic overlays: trunk gaussian centers (red) vs mesh edges
    (black, subsampled). If the mesh hugs the points, alignment is correct."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    if len(edges) > 30000:
        edges = edges[np.random.default_rng(0).choice(len(edges), 30000,
                                                      replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    views = [((0, 2), "x", "z"), ((1, 2), "y", "z"), ((0, 1), "x", "y")]
    pts_idx = (np.random.default_rng(0).choice(
        len(trunk_pts), 40000, replace=False)
        if len(trunk_pts) > 40000 else np.arange(len(trunk_pts)))
    for ax, ((h, v), hl, vl) in zip(axes, views):
        ax.scatter(trunk_pts[pts_idx, h], trunk_pts[pts_idx, v], s=0.5,
                   c="red", linewidths=0, label="trunk gaussians", zorder=1)
        seg = np.stack([verts[edges[:, 0]][:, [h, v]],
                        verts[edges[:, 1]][:, [h, v]]], axis=1)
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection(seg, colors="black", linewidths=0.15,
                                         zorder=2))
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.set_aspect("equal")
        ax.autoscale()
    axes[0].legend(loc="upper left", markerscale=8)
    fig.suptitle(
        f"10 mesh/point alignment — mesh edges must hug red points "
        f"({len(verts)} verts, {len(tris)} tris){transform_note}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  viz -> {path}")


def try_render_mesh(verts, tris, path):
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
        )
        mesh.compute_vertex_normals()
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        eye = center + np.array([1.2, -1.2, 0.8]) * max(bounds.get_extent())
        renderer.scene.camera.look_at(center, eye, [0, 0, 1])
        img = renderer.render_to_image()
        o3d.io.write_image(str(path), img)
        print(f"  viz -> {path}")
    except Exception as e:  # EGL often unavailable headless — non-fatal
        print(f"  (skipped shaded render: {type(e).__name__}: {e})")


def build_capsules(args, cloud, outdir, vizdir, name):
    """Skeleton backend: MST-bridged curve skeleton -> capsule-chain URDF."""
    from splatsim.utils import splat_segmentation as seg
    from splatsim.utils.branch_skeleton import build_branch_skeleton, capsule_urdf

    print("estimating per-point branch radii ...")
    radii = seg.estimate_branch_radii(
        cloud.xyz, np.ones(len(cloud), dtype=bool),
        seg.SegmentationParams(), scales=cloud.scales,
    )
    skel = build_branch_skeleton(
        cloud.xyz, radii,
        node_spacing=args.node_spacing,
        max_bridge_dist=args.max_bridge,
        rdp_epsilon=args.rdp_eps,
        min_capsule_radius=args.min_capsule_radius,
        min_spur_len=args.min_spur_len,
    )
    print(f"skeleton: {len(skel.nodes)} nodes, {len(skel.mst_edges)} MST edges, "
          f"{len(skel.capsules)} capsules; fragments {skel.n_components_before} "
          f"-> {skel.n_components_after} connected components after bridging")

    trunk_pts = cloud.xyz
    note = ""
    if args.transform:
        T = np.asarray(json.loads(Path(args.transform).read_text()), dtype=float)
        for c in skel.capsules:
            c.p0 = c.p0 @ T[:3, :3].T + T[:3, 3]
            c.p1 = c.p1 @ T[:3, :3].T + T[:3, 3]
        trunk_pts = trunk_pts @ T[:3, :3].T + T[:3, 3]
        note = f" [transformed via {args.transform}]"

    urdf_path = outdir / f"{name}_capsules.urdf"
    urdf_path.write_text(capsule_urdf(skel.capsules, f"{name}_trunk_capsules"))
    wrote("URDF (capsules)", urdf_path)

    # viz: capsule segments over trunk points (line width ~ radius)
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    views = [((0, 2), "x", "z"), ((1, 2), "y", "z"), ((0, 1), "x", "y")]
    rng = np.random.default_rng(0)
    idx = (rng.choice(len(trunk_pts), 40000, replace=False)
           if len(trunk_pts) > 40000 else np.arange(len(trunk_pts)))
    for ax, ((h, v), hl, vl) in zip(axes, views):
        ax.scatter(trunk_pts[idx, h], trunk_pts[idx, v], s=0.5, c="red",
                   linewidths=0, label="trunk gaussians", zorder=1)
        segs = [[(c.p0[h], c.p0[v]), (c.p1[h], c.p1[v])] for c in skel.capsules]
        widths = [max(c.radius * 250, 0.8) for c in skel.capsules]
        ax.add_collection(LineCollection(segs, colors="black",
                                         linewidths=widths, alpha=0.55, zorder=2))
        ax.set_xlabel(hl)
        ax.set_ylabel(vl)
        ax.set_aspect("equal")
        ax.autoscale()
    axes[0].legend(loc="upper left", markerscale=8)
    fig.suptitle(
        f"10b capsule skeleton over trunk points — {len(skel.capsules)} capsules, "
        f"{skel.n_components_before}->{skel.n_components_after} fragments{note}"
    )
    p = vizdir / "10b_capsule_skeleton.png"
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"  viz -> {p}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trunk_ply", help="gaussian PLY of the body (a crop of the stage's splat; for vegetation the "
                    "*_trunk_hard.ply from segment_vine_splat.py)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backend", choices=["voxel", "splat-transform", "capsules"],
                    default="splat-transform",
                    help="splat-transform (default): smooth mesh via the "
                    "PlayCanvas CLI; voxel: in-repo mesher with --close gap "
                    "bridging; capsules: experimental skeleton")
    ap.add_argument("--node-spacing", type=float, default=0.03,
                    help="capsules: skeleton node downsample spacing (m)")
    ap.add_argument("--max-bridge", type=float, default=0.15,
                    help="capsules: max gap the skeleton graph may bridge (m)")
    ap.add_argument("--rdp-eps", type=float, default=0.02,
                    help="capsules: chain simplification tolerance (m)")
    ap.add_argument("--min-capsule-radius", type=float, default=0.008,
                    help="capsules: radius floor (m)")
    ap.add_argument("--min-spur-len", type=float, default=0.08,
                    help="capsules: prune leaf chains shorter than this (m)")
    ap.add_argument("--voxel-size", type=float, default=0.012)
    ap.add_argument("--dilate", type=int, default=0,
                    help="grow occupancy by N voxels (safety margin, voxel backend)")
    ap.add_argument("--close", type=int, default=0,
                    help="morphological closing iterations: bridge occlusion "
                    "gaps up to ~2N voxels without inflating (voxel backend)")
    ap.add_argument("--transform", help="JSON file with 4x4 splat->sim matrix "
                    "(applied to the mesh so the OBJ lands in sim frame)")
    ap.add_argument("--max-tris", type=int, default=40000,
                    help="decimate above this triangle count (PyBullet "
                    "segfaults loading very large concave URDF meshes)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    vizdir = outdir / "viz"
    vizdir.mkdir(parents=True, exist_ok=True)
    global _INPUT_PLY
    _INPUT_PLY = os.path.abspath(args.trunk_ply)
    # The build is named after its folder — that is its registry name (a
    # nested stage.yaml is the entry `<folder>`), so two builds of the same
    # input in different folders never collide.
    name = outdir.name

    cloud = read_gaussian_ply(args.trunk_ply)
    print(f"{len(cloud)} trunk gaussians from {args.trunk_ply}")

    if args.backend == "capsules":
        build_capsules(args, cloud, outdir, vizdir, name)
        return

    if args.backend == "voxel":
        verts, tris = voxel_face_mesh(cloud.xyz, args.voxel_size, args.dilate,
                                      args.close)
    else:
        verts, tris = run_splat_transform(Path(args.trunk_ply), outdir,
                                          args.voxel_size)
    print(f"mesh: {len(verts)} verts, {len(tris)} tris ({args.backend})")

    if len(tris) > args.max_tris:
        import open3d as o3d

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
        )
        m = m.simplify_quadric_decimation(target_number_of_triangles=args.max_tris)
        verts = np.asarray(m.vertices)
        tris = np.asarray(m.triangles)
        print(f"decimated to: {len(verts)} verts, {len(tris)} tris "
              f"(--max-tris {args.max_tris})")

    trunk_pts = cloud.xyz
    note = ""
    if args.transform:
        T = np.asarray(json.loads(Path(args.transform).read_text()), dtype=float)
        verts = verts @ T[:3, :3].T + T[:3, 3]
        trunk_pts = trunk_pts @ T[:3, :3].T + T[:3, 3]
        note = f" [transformed to sim frame via {args.transform}]"

    obj_path = outdir / f"{name}_collision.obj"
    write_obj(obj_path, verts, tris)
    wrote("collision mesh (OBJ)", obj_path)

    urdf_path = outdir / f"{name}.urdf"
    urdf_path.write_text(
        URDF_TEMPLATE.format(name=f"{name}_trunk", obj_rel=obj_path.name)
    )
    wrote("URDF (fixed base, concave mesh)", urdf_path)

    write_scene_yaml(outdir, name, urdf_path.name, baked=bool(args.transform))

    plot_mesh_overlay(verts, tris, trunk_pts,
                      vizdir / "10_mesh_overlay.png", note)
    try_render_mesh(verts, tris, vizdir / "11_mesh_render.png")


def wrote(what: str, path) -> None:
    """One output per line, absolute path last (copy/pastable, clickable)."""
    print(f"  {what:<40s} {Path(path).resolve()}")


_INPUT_PLY = ""


def write_scene_yaml(outdir: Path, name: str, urdf_rel: str, baked: bool) -> None:
    """Register the build with the scene registry (data/stages/<scan>/
    segmentations/<build>/stage.yaml). One frame declaration covers every
    artifact of the build — this URDF, the cost field and the grape targets:

      collision_frame: sim    --transform was given; everything is baked into
                              sim frame and loads at identity.
      collision_frame: splat  no --transform; everything is in the scan's
                              frame and the scan's `transformation` (inherited
                              from ../../stage.yaml) is applied at load.

    Only sets the keys this script owns; an existing file keeps its other
    fields (aabb, ply_path, ...)."""
    import yaml
    from splatsim.configs.registry import _edit_yaml_lines
    path = outdir / "stage.yaml"
    frame = "sim" if baked else "splat"
    if path.exists():
        # keep the file's comments and other fields; only set what this build owns
        text = _edit_yaml_lines(path.read_text(), {"urdf_path": urdf_rel, "collision_frame": frame})
        doc = yaml.safe_load(text) or {}
        for k, v in (("name", name), ("base_position", [0.0, 0.0, 0.0]), ("use_fixed_base", True)):
            if k not in doc:
                text = _edit_yaml_lines(text, {k: v})
        path.write_text(text)
    else:
        path.write_text(
            f"# Collision body built from {Path(_INPUT_PLY).name} by scripts/splat_to_collision.py.\n"
            f"# Inherits the scan's transformation from the stage.yaml above; collision_frame\n"
            f"# says which frame the mesh is in ({frame}).\n"
            f"name: {name}\n"
            f"ply_path: {os.path.relpath(_INPUT_PLY, outdir)}\n"
            f"urdf_path: {urdf_rel}\n"
            f"collision_frame: {frame}\n"
            f"base_position: [0.0, 0.0, 0.0]\n"
            f"use_fixed_base: true\n")
    wrote(f"stage yaml (collision_frame: {frame})", path)


if __name__ == "__main__":
    main()
