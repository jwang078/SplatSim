"""Soft-cost field over "pushable" vegetation (leaves, grapes, thin twigs).

Weighted splat points (from scripts/segment_vine_splat.py) are splatted into a
dense voxel grid with a quadratic-falloff kernel; queries are trilinear
lookups (0 outside the grid), cheap enough to run inside RRT path scoring.

The planner consumes this via an ``env_config["soft_cost"]`` payload:

    {
      "npz_path": "data/vine_seg/<scene>/<name>_soft_cost.npz",
      "grid_resolution": 0.01,        # optional, m
      "influence_radius": 0.03,       # optional, m
      "transform": [[...4x4...]],     # optional splat->sim frame matrix
    }

Every field can be visually inspected: ``save_debug_images`` writes
max-projection heatmaps + z-slice mosaics, and ``draw_in_pybullet`` scatters
the source points into a PyBullet GUI colored by weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SoftCostField:
    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        grid_resolution: float = 0.01,
        influence_radius: float = 0.03,
        transform: np.ndarray | None = None,
        metadata: dict | None = None,
        normalize: bool = True,
    ):
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if transform is not None:
            transform = np.asarray(transform, dtype=np.float64)
            points = points @ transform[:3, :3].T + transform[:3, 3]
        self.points = points
        self.weights = weights
        self.resolution = float(grid_resolution)
        self.influence_radius = float(influence_radius)
        self.metadata = metadata or {}

        pad = influence_radius + 2 * grid_resolution
        self.origin = points.min(axis=0) - pad
        upper = points.max(axis=0) + pad
        self.shape = np.maximum(
            np.ceil((upper - self.origin) / grid_resolution).astype(int) + 1, 2
        )
        self.grid = np.zeros(self.shape, dtype=np.float32)
        self._splat_points()
        # Normalize so grid max == 1: makes planner-side weights commensurate
        # with arc-length scores and independent of splat point density.
        if normalize and self.grid.max() > 0:
            self.metadata["normalization_factor"] = float(self.grid.max())
            self.grid /= self.grid.max()

    # ---------------------------------------------------------- construction
    def _splat_points(self) -> None:
        """Accumulate w * (1 - (d/R)^2)^2 into voxels within R of each point."""
        res, rad = self.resolution, self.influence_radius
        reach = int(np.ceil(rad / res))
        offs = np.stack(
            np.meshgrid(*([np.arange(-reach, reach + 1)] * 3), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)

        base = np.floor((self.points - self.origin) / res).astype(int)  # (N,3)
        voxel_centers_base = self.origin + (base + 0.5) * res

        for off in offs:
            centers = voxel_centers_base + off * res
            d2 = ((centers - self.points) ** 2).sum(axis=1)
            k = 1.0 - d2 / (rad * rad)
            hit = k > 0.0
            if not hit.any():
                continue
            idx = base[hit] + off
            ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
            idx = idx[ok]
            vals = (self.weights[hit][ok] * (k[hit][ok] ** 2)).astype(np.float32)
            np.add.at(self.grid, (idx[:, 0], idx[:, 1], idx[:, 2]), vals)

    # ---------------------------------------------------------------- query
    def cost_at(self, query: np.ndarray) -> np.ndarray:
        """Trilinear-interpolated cost at (M, 3) world points; 0 outside."""
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        g = (q - self.origin) / self.resolution - 0.5
        i0 = np.floor(g).astype(int)
        frac = g - i0

        out = np.zeros(len(q))
        valid = np.all((i0 >= 0) & (i0 + 1 < self.shape), axis=1)
        if not valid.any():
            return out
        i0v, fv = i0[valid], frac[valid]
        acc = np.zeros(valid.sum())
        for corner in range(8):
            d = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            w = np.prod(np.where(d, fv, 1.0 - fv), axis=1)
            idx = i0v + d
            acc += w * self.grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        out[valid] = acc
        return out

    def max_cost(self) -> float:
        return float(self.grid.max())

    # ------------------------------------------------------------------ I/O
    def save_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            grid=self.grid,
            origin=self.origin,
            resolution=self.resolution,
            influence_radius=self.influence_radius,
            points=self.points,
            weights=self.weights,
            metadata_json=json.dumps(self.metadata),
        )

    @classmethod
    def from_config(cls, payload: dict) -> "SoftCostField":
        """Build from an env_config['soft_cost'] payload dict."""
        data = np.load(payload["npz_path"], allow_pickle=False)
        if "grid" in data:  # prebuilt field saved via save_npz
            field = cls.__new__(cls)
            field.grid = data["grid"]
            field.origin = data["origin"]
            field.resolution = float(data["resolution"])
            field.influence_radius = float(data["influence_radius"])
            field.shape = np.array(field.grid.shape)
            field.points = data["points"]
            field.weights = data["weights"]
            field.metadata = json.loads(str(data.get("metadata_json", "{}")))
            return field
        # raw segmentation output (xyz + weight [+ class_id])
        return cls(
            points=data["xyz"],
            weights=data["weight"],
            grid_resolution=float(payload.get("grid_resolution", 0.01)),
            influence_radius=float(payload.get("influence_radius", 0.03)),
            transform=payload.get("transform"),
            metadata={"source": str(payload["npz_path"])},
            normalize=bool(payload.get("normalize", True)),
        )

    # ---------------------------------------------------------------- viz
    def save_debug_images(self, out_dir: str | Path, prefix: str = "cost") -> list:
        """Max-projection heatmaps (3 axes) + a z-slice mosaic. Returns paths."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (axis, hl, vl) in zip(
            axes, [(1, "x", "z"), (0, "y", "z"), (2, "x", "y")]
        ):
            proj = self.grid.max(axis=axis)
            if axis != 2:
                proj = proj.T  # put z vertical
            extent = self._extent(axis)
            im = ax.imshow(proj, origin="lower", extent=extent, cmap="inferno",
                           aspect="equal")
            ax.set_xlabel(f"{hl} (m)")
            ax.set_ylabel(f"{vl} (m)")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"soft-cost max projections  (res={self.resolution} m, "
            f"influence={self.influence_radius} m, max={self.max_cost():.2f})"
        )
        p = out_dir / f"{prefix}_max_projections.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

        n_slices = 12
        zs = np.linspace(0, self.shape[2] - 1, n_slices).astype(int)
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        vmax = max(self.max_cost(), 1e-9)
        for ax, zi in zip(axes.ravel(), zs):
            ax.imshow(self.grid[:, :, zi].T, origin="lower", cmap="inferno",
                      vmin=0, vmax=vmax)
            z_m = self.origin[2] + (zi + 0.5) * self.resolution
            ax.set_title(f"z = {z_m:.2f} m", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("soft-cost z-slices (x right, y up)")
        p = out_dir / f"{prefix}_z_slices.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
        return paths

    def _extent(self, axis: int):
        lo, hi = self.origin, self.origin + self.shape * self.resolution
        if axis == 1:  # xz view
            return [lo[0], hi[0], lo[2], hi[2]]
        if axis == 0:  # yz view
            return [lo[1], hi[1], lo[2], hi[2]]
        return [lo[0], hi[0], lo[1], hi[1]]  # xy view

    def draw_in_pybullet(
        self, pb_client, max_points: int = 40000, point_size: float = 3.0
    ) -> int:
        """Scatter source points into a PyBullet GUI colored by weight
        (viridis). Returns the debug-item id."""
        import matplotlib.pyplot as plt

        n = len(self.points)
        idx = (np.random.default_rng(0).choice(n, max_points, replace=False)
               if n > max_points else np.arange(n))
        w = self.weights[idx]
        wn = w / max(w.max(), 1e-9)
        colors = plt.get_cmap("viridis")(wn)[:, :3]
        return pb_client.addUserDebugPoints(
            self.points[idx].tolist(), colors.tolist(), pointSize=point_size
        )


def draw_soft_points_in_gui(
    pb_module,
    npz_path,
    physics_client_id: int = 0,
    max_points: int = 40000,
    point_size: float = 2.0,
    transform=None,
) -> int | None:
    """One-shot GUI overlay of the soft-cost points (leaves/twigs/grapes) —
    the vegetation the pipeline did NOT bake into the hard collision mesh.

    Uses addUserDebugPoints: debug items are not bodies, so they never enter
    the broadphase, never appear in getClosestPoints, and cost nothing in
    stepSimulation — pure GUI-draw overhead, drawn ONCE for a static scene.
    They also do NOT appear in TinyRenderer getCameraImage, so recorded
    observations stay clean.

    Accepts either npz layout: a prebuilt cost field saved via
    SoftCostField.save_npz (``points``/``weights``, already in sim frame) or
    a raw segmentation ``soft_cost.npz`` (``xyz``/``weight``, splat frame —
    pass ``transform`` for those). Colors: viridis by weight (dark = low
    cost, yellow = high/grape). Returns the debug-item id, or None when no
    GUI is connected (DIRECT mode — nothing to draw on).
    """
    import numpy as _np

    info = pb_module.getConnectionInfo(physics_client_id)
    if info.get("connectionMethod") != pb_module.GUI:
        return None
    data = _np.load(npz_path, allow_pickle=False)
    if "points" in data:
        pts, w = data["points"], data["weights"]
    else:
        pts, w = data["xyz"], data["weight"]
        if transform is not None:
            T = _np.asarray(transform, dtype=_np.float64)
            pts = pts @ T[:3, :3].T + T[:3, 3]
    if len(pts) > max_points:
        idx = _np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, w = pts[idx], w[idx]
    import matplotlib.pyplot as _plt

    colors = _plt.get_cmap("viridis")(w / max(float(w.max()), 1e-9))[:, :3]
    return pb_module.addUserDebugPoints(
        pts.tolist(), colors.tolist(), pointSize=point_size,
        physicsClientId=physics_client_id,
    )


def sample_link_chain_points(
    pb_client, robot_id: int, link_indices, spacing: float = 0.05
) -> np.ndarray:
    """World-space sample points along the robot's kinematic chain at the
    CURRENT joint state: link origins + interpolated points between
    consecutive links. Used to evaluate the cost field for a configuration."""
    origins = []
    for li in link_indices:
        if li == -1:
            pos, _ = pb_client.getBasePositionAndOrientation(robot_id)
        else:
            pos = pb_client.getLinkState(robot_id, li)[4]  # worldLinkFramePosition
        origins.append(pos)
    origins = np.asarray(origins)

    samples = [origins]
    for a, b in zip(origins[:-1], origins[1:]):
        seg_len = np.linalg.norm(b - a)
        n_mid = int(seg_len // spacing)
        if n_mid > 0:
            t = (np.arange(1, n_mid + 1) / (n_mid + 1))[:, None]
            samples.append(a + t * (b - a))
    return np.concatenate(samples, axis=0)


def link_radii(pb_client, robot_id: int, link_indices, physics_client_id=None,
               joint_indices=None, lo: float = 0.015, hi: float = 0.08):
    """Approximate cross-sectional radius (m) of each link, measured at the
    ZERO configuration so the value is deterministic rather than depending on
    whatever pose happened to trigger the first call.

    Taken as the MEDIAN of the link's world-AABB half-extents (the long axis
    is the max, the thin axis the min). Approximate by construction — it
    feeds a soft cost, not a collision check — and clamped so a degenerate
    AABB cannot yield a zero or absurd radius.
    """
    saved = None
    if joint_indices is not None:
        saved = pb_client.getJointStates(robot_id, joint_indices)
        for idx in joint_indices:
            pb_client.resetJointState(robot_id, idx, 0.0)
    try:
        out = []
        for li in link_indices:
            a, b = pb_client.getAABB(robot_id, li)
            half = np.sort((np.asarray(b) - np.asarray(a)) / 2.0)
            out.append(float(np.clip(half[1], lo, hi)))
    finally:
        if saved is not None:
            for idx, st in zip(joint_indices, saved):
                pb_client.resetJointState(robot_id, idx, st[0],
                                          targetVelocity=st[1])
    return np.asarray(out, dtype=np.float64)


def link_surface_points(origins, radii, spacing: float = 0.05,
                        n_ring: int = 6) -> np.ndarray:
    """Sample points on the SURFACE of a kinematic chain.

    ``origins`` are consecutive link-frame world positions; between each pair
    the segment is subdivided every ``spacing`` m, and each sample is ringed
    with ``n_ring`` points at the local link radius.

    Why surfaces and not the centreline: a soft-cost field's influence radius
    is typically a few cm, while arm links are 4-7 cm thick, so foliage
    touching a link's SURFACE sits outside the field's support when measured
    from the centreline and reads as exactly zero cost. Ringing inflates the
    ROBOT by its real per-link radius, which a uniformly larger influence
    radius cannot express (fingers and upper arm differ by ~5x).

    ``n_ring=0`` returns centreline points only (historical behaviour).
    """
    origins = np.asarray(origins, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    samples = [origins]
    thetas = (2.0 * np.pi * np.arange(n_ring) / n_ring) if n_ring > 0 else None
    for i, (a, b) in enumerate(zip(origins[:-1], origins[1:])):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        centers = [np.atleast_2d(a)]
        n_mid = int(seg_len // spacing)
        if n_mid > 0:
            t = (np.arange(1, n_mid + 1) / (n_mid + 1))[:, None]
            mids = a + t * seg
            samples.append(mids)
            centers.append(mids)
        if n_ring == 0 or seg_len < 1e-6:
            # Coincident link frames (common inside a gripper tree) give no
            # direction to build a ring around; the origin sample covers them.
            continue
        u = seg / seg_len
        tmp = (np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        e1 = np.cross(u, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1); e2 /= np.linalg.norm(e2)
        r = 0.5 * (radii[i] + radii[i + 1])
        offs = r * (np.cos(thetas)[:, None] * e1 + np.sin(thetas)[:, None] * e2)
        ctr = np.concatenate(centers, axis=0)
        samples.append((ctr[:, None, :] + offs[None, :, :]).reshape(-1, 3))
    return np.concatenate(samples, axis=0)


def aggregate_soft_cost(costs, mode: str = "max") -> float:
    """Reduce per-sample-point field costs to one scalar per configuration.

    "max" = the worst-brushing point on the arm. "mean" dilutes badly: an arm
    carries hundreds of sample points nearly all in free air, so one link
    buried in foliage moves the mean by ~1/N of its true value (measured 26x
    understatement on the vine bench), putting real brushes far below a
    T-RRT transition temperature and letting them through as if free.
    """
    costs = np.asarray(costs, dtype=np.float64)
    return float(np.mean(costs)) if mode == "mean" else float(np.max(costs))
