"""Curve-skeleton extraction from trunk-hard gaussian points -> capsule chain.

The voxel-face collision mesh is faithful to the points, but occlusion and
the opacity prefilter leave gaps, so branches come out as disconnected
fragments. This module recovers CONNECTED branches:

  1. voxel-downsample trunk points into skeleton nodes (mean position,
     median radius estimate per node)
  2. k-NN graph with edges capped at ``max_bridge_dist`` — long enough to
     bridge occlusion gaps, short enough not to weld separate runners
  3. minimum spanning tree (scipy) -> branch topology (a forest: anything
     farther apart than the bridge cap stays a separate component)
  4. walk degree!=2 anchor nodes -> polyline chains, simplify each with
     Ramer-Douglas-Peucker
  5. one capsule per simplified segment (radius = median of the chain's
     node radii, floored at ``min_capsule_radius``)

Capsules are also better RRT collision primitives than a concave trimesh:
smooth signed distances from getClosestPoints, no giant-mesh loading limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


@dataclass
class Capsule:
    p0: np.ndarray
    p1: np.ndarray
    radius: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class SkeletonResult:
    capsules: list
    nodes: np.ndarray  # (M, 3) skeleton node positions
    node_radius: np.ndarray  # (M,)
    mst_edges: np.ndarray  # (E, 2) node-index pairs
    n_components_before: int  # components at node_spacing contact distance
    n_components_after: int  # components after MST bridging
    params: dict = field(default_factory=dict)


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (returns kept indices)."""
    if len(points) < 3:
        return np.arange(len(points))
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            d = np.linalg.norm(points[a + 1:b] - points[a], axis=1)
        else:
            u = seg / seg_len
            rel = points[a + 1:b] - points[a]
            proj = rel @ u
            d = np.linalg.norm(rel - np.outer(proj, u), axis=1)
        i = int(np.argmax(d))
        if d[i] > epsilon:
            mid = a + 1 + i
            keep[mid] = True
            stack.append((a, mid))
            stack.append((mid, b))
    return np.flatnonzero(keep)


def build_branch_skeleton(
    points: np.ndarray,
    radii: np.ndarray,
    node_spacing: float = 0.03,
    knn: int = 8,
    max_bridge_dist: float = 0.15,
    rdp_epsilon: float = 0.02,
    min_capsule_radius: float = 0.008,
    min_spur_len: float = 0.08,
    prune_passes: int = 3,
) -> SkeletonResult:
    points = np.asarray(points, dtype=np.float64)
    radii = np.nan_to_num(np.asarray(radii, dtype=np.float64), nan=0.0)

    # ---- 1. voxel downsample to nodes
    keys = np.floor(points / node_spacing).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_nodes = inverse.max() + 1
    nodes = np.zeros((n_nodes, 3))
    node_radius = np.zeros(n_nodes)
    counts = np.bincount(inverse)
    for d in range(3):
        nodes[:, d] = np.bincount(inverse, weights=points[:, d]) / counts
    # median-ish radius per node (mean is fine at this granularity)
    node_radius = np.bincount(inverse, weights=radii) / counts

    # ---- 2. kNN candidate edges, capped at max_bridge_dist
    tree = cKDTree(nodes)
    k = min(knn + 1, n_nodes)
    dists, nbrs = tree.query(nodes, k=k)
    rows, cols, vals = [], [], []
    for i in range(n_nodes):
        for d, j in zip(dists[i, 1:], nbrs[i, 1:]):
            if np.isfinite(d) and d <= max_bridge_dist:
                rows.append(i)
                cols.append(int(j))
                vals.append(float(d))
    graph = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    # components at "touching" distance (pre-bridge fragmentation metric)
    touch = graph.copy()
    touch.data = np.where(touch.data <= 2 * node_spacing, touch.data, 0)
    touch.eliminate_zeros()
    n_before = connected_components(touch, directed=False)[0]

    # ---- 3. MST (forest over the bridge-capped graph)
    mst = minimum_spanning_tree(graph)
    mst_coo = mst.tocoo()
    edges = np.stack([mst_coo.row, mst_coo.col], axis=1)
    n_after = connected_components(mst, directed=False)[0]

    # ---- 4. spur pruning: surface noise turns the MST "furry" — thousands
    # of short leaf chains hanging off the true branch path. Iteratively
    # delete leaf chains shorter than min_spur_len; real branch tips longer
    # than the threshold survive.
    edge_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}

    def make_adj():
        adj_: list = [[] for _ in range(n_nodes)]
        for a, b in edge_set:
            adj_[a].append(b)
            adj_[b].append(a)
        return adj_

    for _ in range(prune_passes):
        adj = make_adj()
        degree = np.array([len(a) for a in adj])
        removed_any = False
        for leaf in np.flatnonzero(degree == 1):
            if len(adj[leaf]) != 1:
                continue
            path = [leaf]
            length = 0.0
            prev, cur = -1, leaf
            while True:
                nxts = [n for n in adj[cur] if n != prev]
                if degree[cur] > 2 or (degree[cur] == 1 and cur != leaf) or not nxts:
                    break
                nxt = nxts[0]
                length += float(np.linalg.norm(nodes[cur] - nodes[nxt]))
                path.append(nxt)
                prev, cur = cur, nxt
                if degree[cur] != 2:
                    break
            if length < min_spur_len and len(path) >= 2:
                for a, b in zip(path[:-1], path[1:]):
                    edge_set.discard((min(a, b), max(a, b)))
                removed_any = True
        if not removed_any:
            break
    edges = np.array(sorted(edge_set)) if edge_set else np.zeros((0, 2), dtype=int)

    # ---- 5. chains between anchors (degree != 2), then RDP per chain
    adj = make_adj()
    degree = np.array([len(a) for a in adj])
    visited = set()

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    chains = []

    def walk(start, first):
        chain = [start, first]
        visited.add(edge_key(start, first))
        prev, cur = start, first
        while degree[cur] == 2:
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            if edge_key(cur, nxt) in visited:
                break
            visited.add(edge_key(cur, nxt))
            chain.append(nxt)
            prev, cur = cur, nxt
        return chain

    anchors = np.flatnonzero(degree != 2)
    for a in anchors:
        for nb in adj[a]:
            if edge_key(a, nb) not in visited:
                chains.append(walk(a, nb))
    # leftover pure cycles (all degree 2)
    for a, b in edges:
        if edge_key(a, b) not in visited:
            chains.append(walk(int(a), int(b)))

    # ---- 6. capsules
    capsules = []
    for chain in chains:
        pts = nodes[chain]
        r = max(float(np.median(node_radius[chain])), min_capsule_radius)
        kept = _rdp(pts, rdp_epsilon)
        for i0, i1 in zip(kept[:-1], kept[1:]):
            p0, p1 = pts[i0], pts[i1]
            if np.linalg.norm(p1 - p0) < 1e-6:
                continue
            capsules.append(Capsule(p0=p0, p1=p1, radius=r))

    return SkeletonResult(
        capsules=capsules,
        nodes=nodes,
        node_radius=node_radius,
        mst_edges=edges,
        n_components_before=int(n_before),
        n_components_after=int(n_after),
        params=dict(
            node_spacing=node_spacing, knn=knn,
            max_bridge_dist=max_bridge_dist, rdp_epsilon=rdp_epsilon,
            min_capsule_radius=min_capsule_radius,
            min_spur_len=min_spur_len, prune_passes=prune_passes,
        ),
    )


def capsule_urdf(capsules: list, name: str) -> str:
    """Single-link fixed-base URDF whose collision (and visual) geometry is
    the capsule chain. PyBullet's URDF importer supports <capsule>."""
    from scipy.spatial.transform import Rotation

    blocks = []
    for c in capsules:
        mid = (c.p0 + c.p1) / 2.0
        d = (c.p1 - c.p0) / max(c.length, 1e-12)
        # rotate capsule local +z onto d
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, d)
        s = np.linalg.norm(v)
        if s < 1e-12:
            rpy = (0.0, 0.0, 0.0) if d[2] > 0 else (np.pi, 0.0, 0.0)
        else:
            angle = float(np.arctan2(s, float(z @ d)))
            rpy = tuple(Rotation.from_rotvec(v / s * angle).as_euler("xyz"))
        origin = (
            f'<origin xyz="{mid[0]:.6f} {mid[1]:.6f} {mid[2]:.6f}" '
            f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>'
        )
        geom = f'<geometry><capsule radius="{c.radius:.4f}" length="{c.length:.6f}"/></geometry>'
        blocks.append(
            f"    <collision>{origin}{geom}</collision>\n"
            f'    <visual>{origin}{geom}'
            f'<material name="wood"><color rgba="0.45 0.30 0.15 1.0"/></material></visual>'
        )
    body = "\n".join(blocks)
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
{body}
  </link>
</robot>
"""
