"""Generate a synthetic grape-vine gaussian splat PLY for pipeline testing.

Builds a procedural vine in the standard 3DGS PLY layout so the whole
segmentation -> cost-field -> collision-mesh pipeline can be exercised and
visually inspected before a real vine scan exists:

  - curved TRUNK (r ~ 2 cm) + two secondary branches (r ~ 1.2 cm)  [hard]
  - many thin twigs (r ~ 2-4 mm) hanging off the branches           [soft]
  - green leaf blobs around twig ends                               [soft]
  - purple grape clusters                                           [soft]
  - low-opacity gray floaters (should be dropped by the prefilter)

Ground-truth class per point is saved alongside (<out>.truth.npy) so the
segmentation stages can be scored, not just eyeballed.

Usage:
  python scripts/make_synthetic_vine_splat.py --out data/vine_seg/synthetic/vine.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

RGB_TRUNK = np.array([0.42, 0.28, 0.14])
RGB_TWIG = np.array([0.50, 0.35, 0.18])
RGB_LEAF = np.array([0.18, 0.55, 0.20])
RGB_GRAPE = np.array([0.35, 0.15, 0.45])
RGB_FLOATER = np.array([0.6, 0.6, 0.6])

# truth ids match splatsim.utils.splat_segmentation semantics
TRUTH_HARD, TRUTH_TWIG, TRUTH_FOLIAGE, TRUTH_DROPPED, TRUTH_GRAPE = 3, 2, 1, 4, 5


def _bezier(p0, p1, p2, t):
    t = t[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _tube(rng, p0, p1, p2, radius, n, jitter=1.0):
    """Points filling a tube around a quadratic bezier curve."""
    t = rng.uniform(0, 1, n)
    centers = _bezier(np.asarray(p0), np.asarray(p1), np.asarray(p2), t)
    offs = rng.normal(0, radius / np.sqrt(2.0), (n, 3)) * jitter
    return centers + offs


def _blob(rng, center, radii, n):
    return np.asarray(center) + rng.normal(0, 1, (n, 3)) * np.asarray(radii)


def build_vine(rng, n_scale=1.0):
    pts, cols, truth = [], [], []

    def add(xyz, base_rgb, cls):
        pts.append(xyz)
        c = np.clip(base_rgb + rng.normal(0, 0.035, (len(xyz), 3)), 0, 1)
        cols.append(c)
        truth.append(np.full(len(xyz), cls, dtype=np.int32))

    # trunk: ground to 1.1 m with a sway; r = 2 cm
    trunk_top = np.array([0.15, 0.05, 1.1])
    add(_tube(rng, [0, 0, 0], [0.1, -0.05, 0.55], trunk_top,
              0.020, int(9000 * n_scale)), RGB_TRUNK, TRUTH_HARD)

    # two secondary branches; r = 1.2 cm  (still hard)
    sec_ends = []
    for sgn in (-1, 1):
        start = _bezier(np.array([0, 0, 0]), np.array([0.1, -0.05, 0.55]),
                        trunk_top, np.array([0.55 + 0.1 * sgn]))[0]
        end = start + np.array([0.35 * sgn, 0.25 * sgn, 0.30])
        mid = (start + end) / 2 + np.array([0, 0.1 * sgn, 0.05])
        add(_tube(rng, start, mid, end, 0.012, int(4000 * n_scale)),
            RGB_TRUNK, TRUTH_HARD)
        sec_ends.append((start, end))

    # thin twigs off trunk + secondaries; r = 2-4 mm (soft)
    twig_tips = []
    for i in range(14):
        src_start, src_end = sec_ends[i % 2]
        base = src_start + (src_end - src_start) * rng.uniform(0.2, 1.0)
        tip = base + rng.uniform([-0.15, -0.15, -0.20], [0.15, 0.15, 0.10])
        mid = (base + tip) / 2 + rng.normal(0, 0.02, 3)
        r = rng.uniform(0.002, 0.004)
        add(_tube(rng, base, mid, tip, r, int(700 * n_scale)),
            RGB_TWIG, TRUTH_TWIG)
        twig_tips.append(tip)

    # leaves: green blobs at twig tips
    for tip in twig_tips:
        for _ in range(rng.integers(2, 5)):
            c = tip + rng.normal(0, 0.05, 3)
            add(_blob(rng, c, [0.045, 0.045, 0.02], int(900 * n_scale)),
                RGB_LEAF, TRUTH_FOLIAGE)

    # grape clusters hanging below some twig tips
    for tip in twig_tips[::3]:
        c = tip + np.array([0, 0, -0.10])
        add(_blob(rng, c, [0.03, 0.03, 0.06], int(1200 * n_scale)),
            RGB_GRAPE, TRUTH_GRAPE)

    # floaters: sparse low-opacity junk
    n_float = int(1500 * n_scale)
    add(rng.uniform([-0.6, -0.6, 0.0], [0.8, 0.8, 1.4], (n_float, 3)),
        RGB_FLOATER, TRUTH_DROPPED)

    xyz = np.concatenate(pts)
    rgb = np.concatenate(cols)
    truth_arr = np.concatenate(truth)

    opacity = rng.uniform(0.75, 0.98, len(xyz))
    opacity[truth_arr == TRUTH_DROPPED] = rng.uniform(
        0.005, 0.04, (truth_arr == TRUTH_DROPPED).sum()
    )
    return xyz, rgb, opacity, truth_arr


def write_3dgs_ply(path, xyz, rgb, opacity, scale=0.004, sh_rest=45):
    n = len(xyz)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    fields += [(f"f_dc_{i}", "f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "f4") for i in range(sh_rest)]
    fields += [("opacity", "f4")]
    fields += [(f"scale_{i}", "f4") for i in range(3)]
    fields += [(f"rot_{i}", "f4") for i in range(4)]

    v = np.zeros(n, dtype=fields)
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    C0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / C0  # RGB2SH
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = f_dc.T.astype(np.float32)
    eps = 1e-6
    op = np.clip(opacity, eps, 1 - eps)
    v["opacity"] = np.log(op / (1 - op)).astype(np.float32)  # inverse sigmoid
    for i in range(3):
        v[f"scale_{i}"] = np.float32(np.log(scale))
    v["rot_0"] = 1.0  # identity quaternion (w,x,y,z)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/vine_seg/synthetic/vine.ply")
    ap.add_argument("--n-scale", type=float, default=1.0,
                    help="point-count multiplier")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xyz, rgb, opacity, truth = build_vine(rng, args.n_scale)
    write_3dgs_ply(args.out, xyz, rgb, opacity)
    truth_path = Path(args.out).with_suffix(".truth.npy")
    np.save(truth_path, truth)
    counts = {int(k): int((truth == k).sum()) for k in np.unique(truth)}
    print(f"wrote {args.out}  ({len(xyz)} gaussians)")
    print(f"wrote {truth_path}  truth counts={counts}")


if __name__ == "__main__":
    main()
