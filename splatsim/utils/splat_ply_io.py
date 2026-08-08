"""Lightweight gaussian-splat PLY I/O (no torch/CUDA required).

Reads the standard 3DGS PLY layout (x/y/z, f_dc_*, f_rest_*, opacity logit,
scale_* log, rot_*) via plyfile and exposes the *activated* attributes that
segmentation and cost-field construction need. Subset writing copies every
original vertex property row-for-row, so the output remains a valid gaussian
PLY for downstream tools (splat-transform, SuperSplat).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from splatsim.utils.sh_utils import SH2RGB


@dataclass
class GaussianCloud:
    """Activated view of a gaussian splat PLY plus the raw vertex data."""

    xyz: np.ndarray  # (N, 3)
    rgb: np.ndarray  # (N, 3) in [0, 1], from DC spherical harmonics
    opacity: np.ndarray  # (N,) in [0, 1], sigmoid-activated
    scales: np.ndarray  # (N, 3) in world units, exp-activated
    raw_vertex: np.ndarray  # structured array with ALL original properties

    def __len__(self) -> int:
        return self.xyz.shape[0]


def read_gaussian_ply(path: str | Path) -> GaussianCloud:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgb = np.clip(SH2RGB(f_dc.astype(np.float64)), 0.0, 1.0)

    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    else:
        opacity = np.ones(len(xyz))

    scale_names = [n for n in ("scale_0", "scale_1", "scale_2") if n in names]
    if scale_names:
        scales = np.exp(
            np.stack([v[n] for n in scale_names], axis=1).astype(np.float64)
        )
    else:
        scales = np.full((len(xyz), 3), 0.01)

    return GaussianCloud(
        xyz=xyz, rgb=rgb, opacity=opacity, scales=scales, raw_vertex=v
    )


def write_gaussian_ply_subset(
    cloud: GaussianCloud, mask: np.ndarray, out_path: str | Path
) -> int:
    """Write the rows selected by boolean ``mask`` as a valid gaussian PLY."""
    subset = cloud.raw_vertex[np.asarray(mask, dtype=bool)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(subset, "vertex")]).write(str(out_path))
    return len(subset)


def write_rgb_preview_ply(
    xyz: np.ndarray, rgb01: np.ndarray, out_path: str | Path
) -> None:
    """Plain colored point cloud (viewable in SuperSplat/CloudCompare/open3d)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    data = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb255[:, 0], rgb255[:, 1], rgb255[:, 2]
    PlyData([PlyElement.describe(data, "vertex")]).write(str(out_path))
