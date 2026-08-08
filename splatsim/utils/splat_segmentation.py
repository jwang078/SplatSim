"""Color + geometry segmentation of a vegetation gaussian splat.

Splits a vine-like splat into three classes:

  TRUNK_HARD  — thick rigid branches: become a binary collision mesh
  TWIG_SOFT   — thin brown branches: robot may push them; soft cost (higher w)
  FOLIAGE     — leaves/grapes: soft cost (baseline w)

Pipeline stages (each returns arrays so every stage can be visualized):
  1. opacity prefilter
  2. HSV color classification (brown vs green, tunable hue/sat/val windows)
  3. kNN fill of unclassified points + kNN label smoothing
  4. local-radius estimation on branch points (PCA axis + perpendicular RMS)
  5. thick/thin split + connected-component cleanup of the hard set
  6. per-point soft-cost weights  w = opacity * class_multiplier[class]

Pure numpy/scipy — no torch, no GUI. Visualization lives in the CLI script.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy.spatial import cKDTree

# Class labels (kept small ints so they serialize into .npy/.npz cleanly)
UNKNOWN = 0
FOLIAGE = 1
TWIG_SOFT = 2
TRUNK_HARD = 3
DROPPED = 4  # failed opacity prefilter
GRAPE = 5  # purple/red fruit — soft, its own cost multiplier

CLASS_NAMES = {
    UNKNOWN: "unknown",
    FOLIAGE: "foliage",
    TWIG_SOFT: "twig_soft",
    TRUNK_HARD: "trunk_hard",
    DROPPED: "dropped",
    GRAPE: "grape",
}

# Colors used by every visualization so all stages read consistently.
CLASS_COLORS = {
    UNKNOWN: (0.55, 0.55, 0.55),
    FOLIAGE: (0.20, 0.70, 0.20),
    TWIG_SOFT: (1.00, 0.60, 0.10),
    TRUNK_HARD: (0.85, 0.10, 0.10),
    DROPPED: (0.85, 0.85, 0.85),
    GRAPE: (0.60, 0.15, 0.70),
}


@dataclass
class SegmentationParams:
    # Stage 1: opacity prefilter
    min_opacity: float = 0.05

    # Stage 2: HSV windows (hue in degrees [0, 360))
    green_hue_range: tuple = (60.0, 170.0)
    green_min_sat: float = 0.15
    brown_hue_range: tuple = (10.0, 55.0)
    brown_min_sat: float = 0.15
    brown_max_val: float = 0.85
    # Grape hue windows (degrees; hi may exceed 360 to express wrap-around,
    # e.g. (335, 372) = burgundy/red grapes spanning the 0deg seam). Purple
    # window catches dark grapes. GREEN grapes are indistinguishable from
    # leaves by hue — they ride with foliage unless proven problematic.
    grape_hue_ranges: tuple = ((250.0, 335.0), (335.0, 368.0))
    grape_min_sat: float = 0.25  # excludes the gray hue-0 spike

    # Stage 3: kNN fill/smooth
    knn_k: int = 15
    smooth_iterations: int = 1

    # Stage 4/5: thin-twig split
    local_radius: float = 0.016  # m, neighborhood for radius estimation
    hard_min_radius: float = 0.008  # m, branch radius above which it is rigid
    # Connectivity radius / minimum size for keeping hard components. eps
    # must exceed the point spacing of SPARSELY reconstructed wood (posts and
    # stakes are often covered by few large gaussians several cm apart) or
    # the cleanup shreds them into fragments and demotes rigid wood to twig.
    hard_component_eps: float = 0.05  # m, connectivity radius for hard set
    min_hard_component: int = 150  # smaller hard islands demoted to twig

    # Stage 6: per-class cost multipliers (THE tunable punishment knob).
    # Twigs slightly costlier than foliage by default; extend with e.g.
    # "grape": 2.0 once grapes get their own class.
    class_multiplier: dict = field(
        default_factory=lambda: {"foliage": 1.0, "twig_soft": 1.5, "grape": 2.0}
    )

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SegmentationParams":
        data = json.loads(Path(path).read_text())
        for key in ("green_hue_range", "brown_hue_range"):
            if key in data:
                data[key] = tuple(data[key])
        if "grape_hue_ranges" in data:
            data["grape_hue_ranges"] = tuple(tuple(r) for r in data["grape_hue_ranges"])
        return cls(**data)


@dataclass
class SegmentationResult:
    labels: np.ndarray  # (N,) int class ids over ALL input points
    color_labels: np.ndarray  # (N,) labels right after color stage (for viz)
    weights: np.ndarray  # (N,) soft-cost weight (0 for hard/dropped)
    branch_radius: np.ndarray  # (N,) local radius estimate, NaN off-branch
    hsv: np.ndarray  # (N, 3) hue[deg]/sat/val (for threshold-tuning plots)
    params: SegmentationParams

    @property
    def hard_mask(self) -> np.ndarray:
        return self.labels == TRUNK_HARD

    @property
    def soft_mask(self) -> np.ndarray:
        return (
            (self.labels == FOLIAGE)
            | (self.labels == TWIG_SOFT)
            | (self.labels == GRAPE)
        )


def classify_by_color(rgb: np.ndarray, params: SegmentationParams) -> tuple:
    """Stage 2: HSV windows -> UNKNOWN / FOLIAGE / branch-candidate mask."""
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0))
    hue = hsv[:, 0] * 360.0
    sat, val = hsv[:, 1], hsv[:, 2]

    g0, g1 = params.green_hue_range
    b0, b1 = params.brown_hue_range
    green = (hue >= g0) & (hue <= g1) & (sat >= params.green_min_sat)
    grape = np.zeros(len(rgb), dtype=bool)
    for lo, hi in params.grape_hue_ranges:
        in_window = ((hue >= lo) & (hue <= hi)) | ((hue + 360.0 >= lo) & (hue + 360.0 <= hi))
        grape |= in_window
    grape &= (sat >= params.grape_min_sat) & ~green
    brown = (
        (hue >= b0) & (hue <= b1)
        & (sat >= params.brown_min_sat) & (val <= params.brown_max_val)
        & ~green & ~grape
    )

    labels = np.full(len(rgb), UNKNOWN, dtype=np.int32)
    labels[green] = FOLIAGE
    labels[grape] = GRAPE
    # Branch points provisionally TWIG_SOFT; stage 5 promotes thick ones.
    labels[brown] = TWIG_SOFT
    hsv_out = np.stack([hue, sat, val], axis=1)
    return labels, hsv_out


def knn_fill_and_smooth(
    xyz: np.ndarray, labels: np.ndarray, params: SegmentationParams
) -> np.ndarray:
    """Stage 3: majority-vote unknowns onto a class, then smooth speckle."""
    labels = labels.copy()
    active = labels != DROPPED
    if active.sum() < params.knn_k + 1:
        return labels
    tree = cKDTree(xyz[active])
    active_idx = np.flatnonzero(active)

    def majority_pass(target_mask: np.ndarray) -> None:
        targets = np.flatnonzero(target_mask)
        if len(targets) == 0:
            return
        _, nn = tree.query(xyz[targets], k=params.knn_k)
        nn_labels = labels[active_idx[nn]]  # (T, k)
        for row, point_i in enumerate(targets):
            votes = nn_labels[row]
            votes = votes[(votes != UNKNOWN) & (votes != DROPPED)]
            if len(votes):
                labels[point_i] = np.bincount(votes).argmax()

    majority_pass(labels == UNKNOWN)
    for _ in range(params.smooth_iterations):
        majority_pass(active)
    return labels


def estimate_branch_radii(
    xyz: np.ndarray,
    branch_mask: np.ndarray,
    params: SegmentationParams,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Stage 4: local branch radius via PCA axis + perpendicular RMS spread.

    For points sampled through a cylinder of radius R the perpendicular RMS
    is ~R/sqrt(2); we scale by sqrt(2) so the estimate is directly comparable
    to `hard_min_radius` in meters.

    ``scales`` (per-gaussian world-unit scales, (N,3)) act as a LOWER BOUND
    on the estimate: big structures (posts, planks) are often reconstructed
    sparsely with few LARGE gaussians, leaving too few neighbors for PCA —
    without the bound those points default to "thin" and rigid wood gets
    classified as pushable twig. A gaussian of scale s cannot be part of a
    branch thinner than s.
    """
    radius = np.full(len(xyz), np.nan)
    branch_idx = np.flatnonzero(branch_mask)
    if len(branch_idx) < 5:
        return radius
    pts = xyz[branch_idx]
    scale_floor = (
        scales[branch_idx].max(axis=1) if scales is not None
        else np.zeros(len(branch_idx))
    )
    tree = cKDTree(pts)
    neighborhoods = tree.query_ball_point(pts, r=params.local_radius)
    for row, nbrs in enumerate(neighborhoods):
        if len(nbrs) < 5:
            radius[branch_idx[row]] = scale_floor[row]
            continue
        local = pts[nbrs] - pts[nbrs].mean(axis=0)
        # principal axis = branch direction
        _, _, vt = np.linalg.svd(local, full_matrices=False)
        axis = vt[0]
        perp = local - np.outer(local @ axis, axis)
        rms = np.sqrt((perp**2).sum(axis=1).mean())
        radius[branch_idx[row]] = max(rms * np.sqrt(2.0), scale_floor[row])
    return radius


def split_hard_soft(
    xyz: np.ndarray,
    labels: np.ndarray,
    branch_radius: np.ndarray,
    params: SegmentationParams,
) -> np.ndarray:
    """Stage 5: promote thick branch points to TRUNK_HARD, then demote small
    disconnected hard islands (dry leaves, floaters) back to TWIG_SOFT."""
    labels = labels.copy()
    branch = labels == TWIG_SOFT
    thick = branch & (branch_radius >= params.hard_min_radius)
    labels[thick] = TRUNK_HARD

    hard_idx = np.flatnonzero(labels == TRUNK_HARD)
    if len(hard_idx) == 0:
        return labels

    # Connected components on the hard set via union-find over radius pairs.
    pts = xyz[hard_idx]
    tree = cKDTree(pts)
    parent = np.arange(len(hard_idx))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in tree.query_pairs(r=params.hard_component_eps):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(len(hard_idx))])
    _, root_inverse, counts = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    small = counts[root_inverse] < params.min_hard_component
    labels[hard_idx[small]] = TWIG_SOFT
    return labels


def compute_weights(
    opacity: np.ndarray, labels: np.ndarray, params: SegmentationParams
) -> np.ndarray:
    """Stage 6: w = opacity * class_multiplier. Hard/dropped points get 0
    (hard geometry is handled by the binary mesh, not the cost field)."""
    weights = np.zeros(len(labels))
    mult = params.class_multiplier
    weights[labels == FOLIAGE] = mult.get("foliage", 1.0)
    weights[labels == TWIG_SOFT] = mult.get("twig_soft", 1.5)
    weights[labels == GRAPE] = mult.get("grape", 2.0)
    # UNKNOWN survivors count as foliage-weight rather than free space:
    weights[labels == UNKNOWN] = mult.get("foliage", 1.0)
    return weights * opacity


def cluster_stats(members: np.ndarray, top_quantile: float = 0.9) -> dict:
    """Summarize a set of cluster member points into the bunch dict the vine
    env consumes: ``center``, ``peduncle``, ``n_points``, ``extent``.

    Shared by the automatic clusterer and by manual marking, so a
    hand-annotated bunch is indistinguishable downstream from a detected one.

    ``peduncle`` is where the cluster hangs from the vine — its TOP — used as
    the reach target for a cutter, while ``center`` stays what a camera is
    aimed at. It is the centroid of the top decile in z, rather than the
    single highest point (one stray gaussian) or ``center + extent/2`` (wrong,
    because ``center`` is the MEAN and the bbox is not centred on it for an
    asymmetric bunch).
    """
    members = np.asarray(members, dtype=np.float64)
    z = members[:, 2]
    top = members[z >= np.quantile(z, top_quantile)]
    return {
        "center": members.mean(axis=0).tolist(),
        "peduncle": top.mean(axis=0).tolist(),
        "n_points": int(members.shape[0]),
        "extent": (members.max(axis=0) - members.min(axis=0)).tolist(),
    }


def cluster_labeled_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    eps: float = 0.025,
    min_points: int = 150,
    transform: np.ndarray | None = None,
) -> list:
    """Group the points of one class into spatial clusters (union-find over
    ``eps``-neighbor pairs). Returns clusters sorted largest-first, each a
    dict with ``center``, ``n_points``, ``extent`` — e.g. grape bunches as
    grasp/approach targets. ``transform`` (4x4) is applied first, so results
    can be produced directly in the sim frame."""
    pts = np.asarray(xyz, dtype=np.float64)[labels == class_id]
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64)
        pts = pts @ T[:3, :3].T + T[:3, 3]
    if len(pts) == 0:
        return []
    tree = cKDTree(pts)
    parent = np.arange(len(pts))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in tree.query_pairs(r=eps):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(len(pts))])
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    clusters = []
    for k in np.argsort(-counts):
        if counts[k] < min_points:
            break
        clusters.append(cluster_stats(pts[inverse == k]))
    return clusters


def segment_vegetation_splat(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    params: SegmentationParams | None = None,
    scales: np.ndarray | None = None,
) -> SegmentationResult:
    """Run all stages. Every intermediate needed for inspection is returned."""
    params = params or SegmentationParams()
    n = len(xyz)

    labels = np.full(n, DROPPED, dtype=np.int32)
    keep = opacity >= params.min_opacity

    color_labels, hsv = classify_by_color(rgb, params)
    labels[keep] = color_labels[keep]
    color_stage = labels.copy()

    labels = knn_fill_and_smooth(xyz, labels, params)
    branch_radius = estimate_branch_radii(xyz, labels == TWIG_SOFT, params, scales)
    labels = split_hard_soft(xyz, labels, branch_radius, params)
    weights = compute_weights(opacity, labels, params)

    return SegmentationResult(
        labels=labels,
        color_labels=color_stage,
        weights=weights,
        branch_radius=branch_radius,
        hsv=hsv,
        params=params,
    )
