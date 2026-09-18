"""RobotSpec — everything the simulator needs to know about a robot, derived
from its URDF plus an optional `robot:` block in its yaml, so that any robot
(any DOF, any gripper, any number of cameras, fixed or mobile base) is a
drop-in folder rather than a set of class attributes.

    data/assets/<name>/asset.yaml      (or a robot scan under data/stages/)
    data/assets/<name>/my_robot.urdf   (+ meshes)

Minimal asset.yaml — everything else is derived from the URDF:

    urdf_path: my_robot.urdf

Full form (every key optional; `auto` or omitted = derive):

    urdf_path: my_robot.urdf
    base_position: [0, 0, 0]
    base_orientation_rpy: [0, 0, 0]
    robot:
      base: fixed | planar | wheeled
      arm_joints: auto | [joint names]         # planner / IK joints, in order
      wheel_joints: []                          # base: wheeled; velocity-controlled
      ee_link: auto | link name                 # goal / tool frame (auto = last arm link)
      self_collision_skip_pairs: auto | [[link_a, link_b], ...]
      initial_joint_positions: auto | [per arm joint]
      cameras:                                  # any number, [] = none
        - name: wrist                           # observation key "<name>_rgb"
          link: wrist_camera_link
          offset_xyz: [0, 0, 0]
          offset_rpy: [0, 0, 0]
          model: fisheye_v2 | fisheye_v1 | pinhole
          fov_deg: 60                           # pinhole only
      gripper:
        kind: auto | robotiq_2f85 | synergies | per_joint | none
        joints: auto | [joint names]
        open:   [per gripper joint]             # synergies: joint targets at command 0
        closed: [per gripper joint]             #            ... and at command 1
        matrix: [[...]]                         # synergies with K > 1: K x N, q = open + matrix^T c
        tip_link: auto | link name
        contact_links: auto | [link names]

`kind: auto` recognises the Robotiq 2F-85 by its joint names (the shipped
UR5) and keeps its calibrated mimic behaviour; any other gripper becomes
`per_joint` over its movable joints, which always works and can be tightened
to `synergies` later. A URDF with no finger-ish joints is `none`.

The legacy config form (`articulation_config`, `wrist_camera_link_name`,
`use_gripper`) is still read, so existing robot scans need no changes.

Derivation needs the URDF loaded in a pybullet client (joint types, names,
limits come from there) — `RobotSpec.derive(client, body_id, cfg)`. Nothing
in here touches the simulator state.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# pybullet joint type ids
_REVOLUTE, _PRISMATIC, _SPHERICAL, _PLANAR, _FIXED = 0, 1, 2, 3, 4
_MOVABLE = (_REVOLUTE, _PRISMATIC)

# Joint-name patterns that mark gripper joints when `gripper.joints` is auto.
_GRIPPER_NAME_RE = re.compile(r"finger|gripper|knuckle|jaw|hand_joint|thumb|palm", re.I)
_WHEEL_NAME_RE = re.compile(r"wheel", re.I)
# Fixed-camera link naming used by the shipped robots.
_CAMERA_LINK_RE = re.compile(r"camera", re.I)

# The Robotiq 2F-85 as shipped on the UR5: drive joint + mimic children with
# their gear ratios (see PybulletRobotServerBase.setup_gripper). Recognised by
# name so `kind: auto` keeps the calibrated open-length behaviour.
ROBOTIQ_2F85_DRIVE = "finger_joint"
ROBOTIQ_2F85_MIMIC = {
    "right_outer_knuckle_joint": 1,
    "left_inner_knuckle_joint": 1,
    "right_inner_knuckle_joint": 1,
    "left_inner_finger_joint": -1,
    "right_inner_finger_joint": -1,
}
ROBOTIQ_2F85_ALL_JOINTS = [
    "finger_joint", "left_outer_finger_joint", "left_inner_finger_joint",
    "left_inner_finger_pad_joint", "left_inner_knuckle_joint",
    "right_outer_knuckle_joint", "right_outer_finger_joint",
    "right_inner_finger_joint", "right_inner_finger_pad_joint",
    "right_inner_knuckle_joint",
]


@dataclass
class JointInfo:
    index: int
    name: str
    type: int
    lower: float
    upper: float
    max_force: float
    max_velocity: float
    child_link: str
    parent_link_index: int

    @property
    def movable(self) -> bool:
        return self.type in _MOVABLE


@dataclass
class CameraSpec:
    name: str                       # observation key is f"{name}_rgb"
    link: str
    link_index: int
    offset_xyz: tuple = (0.0, 0.0, 0.0)
    offset_rpy: tuple = (0.0, 0.0, 0.0)
    model: str = "pinhole"          # pinhole | fisheye_v1 | fisheye_v2
    fov_deg: Optional[float] = None # pinhole horizontal FoV; None = same as the scene's base camera

    @property
    def obs_key(self) -> str:
        return f"{self.name}_rgb"

    @property
    def fisheye_version(self) -> Optional[int]:
        m = re.fullmatch(r"fisheye_v(\d+)", self.model)
        return int(m.group(1)) if m else None


@dataclass
class GripperSpec:
    kind: str                       # robotiq_2f85 | synergies | per_joint | none
    joint_indices: List[int] = field(default_factory=list)   # movable gripper joints
    joint_names: List[str] = field(default_factory=list)
    link_indices: List[int] = field(default_factory=list)    # every link under the gripper (incl. fixed)
    open: Optional[np.ndarray] = None       # per joint_indices
    closed: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None     # K x N (synergies)
    mimic: Dict[int, tuple] = field(default_factory=dict)    # child joint idx -> (parent idx, multiplier, offset)
    tip_link_index: Optional[int] = None
    contact_link_indices: List[int] = field(default_factory=list)

    @property
    def command_dim(self) -> int:
        if self.kind == "none":
            return 0
        if self.kind == "per_joint":
            return len(self.joint_indices)
        if self.kind == "synergies" and self.matrix is not None:
            return int(self.matrix.shape[0])
        return 1  # robotiq_2f85, single-synergy

    def joint_targets(self, command: Sequence[float]) -> np.ndarray:
        """Joint targets (per joint_indices) for a command vector of
        command_dim values in [0, 1] (0 = open). Not used for robotiq_2f85,
        whose calibrated mimic path lives in the server."""
        c = np.asarray(command, dtype=np.float64).reshape(-1)
        if self.kind == "per_joint":
            lo, hi = self.open, self.closed
            return lo + (hi - lo) * np.clip(c, 0.0, 1.0)
        if self.kind == "synergies":
            if self.matrix is None:
                return self.open + (self.closed - self.open) * float(np.clip(c[0], 0.0, 1.0))
            return self.open + self.matrix.T @ c
        raise ValueError(f"joint_targets not defined for gripper kind {self.kind!r}")


@dataclass
class RobotSpec:
    name: str
    urdf_path: str
    base: str                                   # fixed | planar | wheeled
    joints: List[JointInfo]
    link_names: Dict[int, str]                  # link index -> name (-1 = base)
    arm_joint_indices: List[int]
    wheel_joint_indices: List[int]
    gripper: GripperSpec
    cameras: List[CameraSpec]
    ee_link_index: int
    state_joint_indices: List[int]              # what get_joint_state / teleport vectors index
    initial_joint_positions: np.ndarray         # per state_joint_indices
    joint_signs: np.ndarray                     # per state_joint_indices
    self_collision_skip_pairs: List[tuple]      # (link_a, link_b) indices
    legacy: bool                                # config had no `robot:` block
    has_splat: bool

    # ----------------------------------------------------------- convenience
    @property
    def num_dofs(self) -> int:
        return len(self.arm_joint_indices)

    @property
    def action_dim(self) -> int:
        return self.num_dofs + self.gripper.command_dim

    @property
    def ee_link_name(self) -> str:
        return self.link_names[self.ee_link_index]

    def joint_by_name(self, name: str) -> JointInfo:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"no joint named {name!r} (have: {[j.name for j in self.joints]})")

    def link_index(self, name: str) -> int:
        for idx, n in self.link_names.items():
            if n == name:
                return idx
        raise KeyError(f"no link named {name!r} (have: {sorted(self.link_names.values())})")

    def camera(self, name: str) -> Optional[CameraSpec]:
        for c in self.cameras:
            if c.name == name or c.obs_key == name:
                return c
        return None

    def summary(self) -> str:
        g = self.gripper
        lines = [
            f"robot {self.name!r}: {self.urdf_path}",
            f"  base: {self.base}",
            f"  arm joints ({self.num_dofs}): {[self.joints[i].name for i in self.arm_joint_indices]}",
            f"  ee link: {self.ee_link_name} (index {self.ee_link_index})",
            f"  gripper: {g.kind}" + (f", command dim {g.command_dim}, joints {g.joint_names}" if g.kind != 'none' else ''),
            f"  cameras ({len(self.cameras)}): " + (", ".join(f"{c.name}@{c.link} [{c.model}]" for c in self.cameras) or "none"),
            f"  action vector: {self.num_dofs} arm + {g.command_dim} gripper = {self.action_dim}",
            f"  splat scan: {'yes' if self.has_splat else 'no (rendered from URDF meshes)'}",
        ]
        if self.wheel_joint_indices:
            lines.append(f"  wheel joints: {[self.joints[i].name for i in self.wheel_joint_indices]}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- derive
    @classmethod
    def derive(cls, client, body_id: int, cfg: Dict[str, Any], name: str = "robot",
               wrist_cam_ver: Optional[int] = None, urdf_path: Optional[str] = None) -> "RobotSpec":
        """Build the spec from a loaded pybullet body + its yaml dict
        (a scene_registry entry). Honours the legacy keys when there is no
        `robot:` block."""
        rb: Dict[str, Any] = dict(cfg.get("robot") or {})
        legacy = "robot" not in cfg
        joints = _read_joints(client, body_id)
        link_names = {-1: _base_link_name(client, body_id)}
        for j in joints:
            link_names[j.index] = j.child_link
        movable = [j for j in joints if j.movable]

        def _auto(v): return v is None or (isinstance(v, str) and v.lower() == "auto")

        # --- wheels ---------------------------------------------------------
        wheel_names = rb.get("wheel_joints") or []
        wheel_idx = [_find_joint(joints, n).index for n in wheel_names]
        base = str(rb.get("base") or ("wheeled" if wheel_idx else "fixed")).lower()
        if base not in ("fixed", "planar", "wheeled"):
            raise ValueError(f"{name}: robot.base must be fixed | planar | wheeled, got {base!r}")

        # --- gripper -------------------------------------------------------
        gcfg: Dict[str, Any] = dict(rb.get("gripper") or {})
        kind = str(gcfg.get("kind") or "auto").lower()
        use_gripper_legacy = bool(cfg.get("use_gripper", True))
        names = {j.name for j in joints}
        is_robotiq = ROBOTIQ_2F85_DRIVE in names and all(n in names for n in ROBOTIQ_2F85_MIMIC)
        if kind == "auto":
            if is_robotiq and (not legacy or use_gripper_legacy):
                kind = "robotiq_2f85"
            elif legacy and not use_gripper_legacy:
                kind = "none"
            else:
                auto_g = [j for j in movable if _GRIPPER_NAME_RE.search(j.name) and not _WHEEL_NAME_RE.search(j.name)]
                kind = "per_joint" if auto_g else "none"
        if kind == "robotiq_2f85" and not is_robotiq:
            raise ValueError(f"{name}: gripper.kind robotiq_2f85 but the URDF has no Robotiq 2F-85 joints")

        if kind == "none":
            g_roots: List[int] = []
        elif kind == "robotiq_2f85":
            g_roots = [j.index for j in joints if j.name in ROBOTIQ_2F85_ALL_JOINTS and j.movable]
        elif _auto(gcfg.get("joints")):
            g_roots = [j.index for j in movable
                       if _GRIPPER_NAME_RE.search(j.name) and not _WHEEL_NAME_RE.search(j.name)]
        else:
            g_roots = [_find_joint(joints, n).index for n in gcfg["joints"]]
        g_roots = [i for i in g_roots if i not in wheel_idx]
        # The gripper is a SUBTREE: every movable joint whose link hangs below
        # the declared/detected joints belongs to it (mimic children, extra
        # finger joints) even if it wasn't named — otherwise it would be
        # mistaken for an arm joint.
        g_links_all = _links_under(joints, g_roots) if g_roots else []
        g_joint_idx = sorted(set(g_roots) | {j.index for j in movable if j.index in g_links_all and j.index not in wheel_idx})

        # --- arm -----------------------------------------------------------
        if _auto(rb.get("arm_joints")):
            if legacy:
                n_legacy = int(cfg.get("num_dofs") or 0)
                arm_idx = [j.index for j in movable if j.index not in g_joint_idx and j.index not in wheel_idx]
                if n_legacy:
                    arm_idx = arm_idx[:n_legacy]
            else:
                arm_idx = [j.index for j in movable if j.index not in g_joint_idx and j.index not in wheel_idx]
        else:
            arm_idx = [_find_joint(joints, n).index for n in rb["arm_joints"]]
        if not arm_idx:
            raise ValueError(f"{name}: no arm joints found in the URDF (no movable non-gripper joints)")

        # --- gripper details (needs arm known for link ownership) ----------
        g_links = g_links_all
        g_mimic = _mimic_from_urdf(urdf_path or cfg.get("urdf_path"), joints) if kind in ("synergies", "per_joint") else {}
        # mimic children are driven by their parent, not commanded
        commanded = [i for i in g_joint_idx if i not in g_mimic] if kind != "robotiq_2f85" else g_joint_idx
        gopen = gclosed = gmat = None
        if kind in ("synergies", "per_joint"):
            n = len(commanded)
            lo = np.array([joints[i].lower for i in commanded]); hi = np.array([joints[i].upper for i in commanded])
            gopen = np.asarray(gcfg.get("open"), dtype=np.float64) if gcfg.get("open") is not None else np.where(np.isfinite(lo), lo, 0.0)
            gclosed = np.asarray(gcfg.get("closed"), dtype=np.float64) if gcfg.get("closed") is not None else np.where(np.isfinite(hi), hi, 0.0)
            if len(gopen) != n or len(gclosed) != n:
                raise ValueError(f"{name}: gripper open/closed need {n} values (commanded joints "
                                 f"{[joints[i].name for i in commanded]}), got {len(gopen)}/{len(gclosed)}")
            if gcfg.get("matrix") is not None:
                gmat = np.asarray(gcfg["matrix"], dtype=np.float64)
                if gmat.ndim != 2 or gmat.shape[1] != n:
                    raise ValueError(f"{name}: gripper.matrix must be K x {n}")
                kind = "synergies"
        tip_idx = None
        if kind != "none":
            if not _auto(gcfg.get("tip_link")):
                tip_idx = _link_index(link_names, gcfg["tip_link"])
            else:
                leaves = _leaf_links(joints, g_links)
                tip_idx = leaves[0] if len(leaves) == 1 else None   # >1 leaves: centroid, resolved by FK later
        if not _auto(gcfg.get("contact_links")):
            contact = [_link_index(link_names, n) for n in gcfg["contact_links"]]
        else:
            contact = _leaf_links(joints, g_links) if g_links else []
        gripper = GripperSpec(kind=kind, joint_indices=commanded, joint_names=[joints[i].name for i in commanded],
                              link_indices=g_links, open=gopen, closed=gclosed, matrix=gmat, mimic=g_mimic,
                              tip_link_index=tip_idx, contact_link_indices=contact)

        # --- cameras -------------------------------------------------------
        cams: List[CameraSpec] = []
        if "cameras" in rb:
            for c in rb["cameras"] or []:
                cams.append(CameraSpec(
                    name=str(c["name"]), link=str(c["link"]), link_index=_link_index(link_names, c["link"]),
                    offset_xyz=tuple(c.get("offset_xyz", (0, 0, 0))), offset_rpy=tuple(c.get("offset_rpy", (0, 0, 0))),
                    model=str(c.get("model", "pinhole")),
                    fov_deg=(float(c["fov_deg"]) if c.get("fov_deg") is not None else None)))
        elif cfg.get("wrist_camera_link_name"):
            ver = wrist_cam_ver if wrist_cam_ver else 0
            cams.append(CameraSpec(name="wrist", link=cfg["wrist_camera_link_name"],
                                   link_index=_link_index(link_names, cfg["wrist_camera_link_name"]),
                                   model=(f"fisheye_v{ver}" if ver else "pinhole")))
        seen = set()
        for c in cams:
            if c.name in seen:
                raise ValueError(f"{name}: duplicate camera name {c.name!r}")
            seen.add(c.name)

        # --- EE link -------------------------------------------------------
        if not _auto(rb.get("ee_link")):
            ee = _link_index(link_names, rb["ee_link"])
        elif cams and legacy:
            ee = cams[0].link_index            # shipped convention: goal frame = wrist camera link
        else:
            ee = arm_idx[-1]
            # prefer a fixed "tool"/"ee" link hanging off the last arm link if the URDF has one
            for j in joints:
                if j.parent_link_index == arm_idx[-1] and not j.movable and re.search(r"ee|tool|tcp|flange|hand", j.child_link, re.I):
                    ee = j.index
                    break

        # --- state vector, initial positions, signs -------------------------
        if legacy:
            state_idx = list(range(1, len(joints)))          # historical: joints 1..N-1, all types
        else:
            state_idx = arm_idx + commanded + wheel_idx
        art = cfg.get("articulation_config") or {}
        init_src = rb.get("initial_joint_positions")
        if _auto(init_src):
            init_src = art.get("initial_joint_positions")
        init = np.zeros(len(state_idx))
        if init_src is not None:
            v = np.asarray(init_src, dtype=np.float64).reshape(-1)
            init[: min(len(v), len(init))] = v[: len(init)]
        signs = np.ones(len(state_idx))
        if art.get("joint_signs") is not None:
            v = np.asarray(art["joint_signs"], dtype=np.float64).reshape(-1)
            signs[: min(len(v), len(signs))] = v[: len(signs)]

        # --- self-collision skip pairs --------------------------------------
        skip: List[tuple] = []
        if not _auto(rb.get("self_collision_skip_pairs")):
            skip = [(_link_index(link_names, a), _link_index(link_names, b)) for a, b in rb["self_collision_skip_pairs"]]
        elif not legacy:
            # adjacent links always; every gripper-internal pair (linkages overlap by design)
            for j in joints:
                skip.append((j.parent_link_index, j.index))
            for a in g_links:
                for b in g_links:
                    if a < b:
                        skip.append((a, b))
        # legacy: the server's class-level lists apply (see PybulletRobotServerBase)

        return cls(
            name=name, urdf_path=str(urdf_path or cfg.get("urdf_path")), base=base, joints=joints,
            link_names=link_names, arm_joint_indices=arm_idx, wheel_joint_indices=wheel_idx,
            gripper=gripper, cameras=cams, ee_link_index=ee, state_joint_indices=state_idx,
            initial_joint_positions=init, joint_signs=signs, self_collision_skip_pairs=skip,
            legacy=legacy, has_splat=bool(cfg.get("model_path") or cfg.get("ply_path")),
        )


# ---------------------------------------------------------------- helpers
def _read_joints(client, body_id: int) -> List[JointInfo]:
    out = []
    for i in range(client.getNumJoints(body_id)):
        info = client.getJointInfo(body_id, i)
        out.append(JointInfo(index=i, name=info[1].decode(), type=int(info[2]), lower=float(info[8]),
                             upper=float(info[9]), max_force=float(info[10]), max_velocity=float(info[11]),
                             child_link=info[12].decode(), parent_link_index=int(info[16])))
    return out


def _base_link_name(client, body_id: int) -> str:
    try:
        return client.getBodyInfo(body_id)[0].decode()
    except Exception:
        return "base"


def _find_joint(joints: List[JointInfo], name: str) -> JointInfo:
    for j in joints:
        if j.name == name:
            return j
    raise KeyError(f"no joint named {name!r}; URDF joints: {[j.name for j in joints]}")


def _link_index(link_names: Dict[int, str], name: str) -> int:
    for idx, n in link_names.items():
        if n == name:
            return idx
    raise KeyError(f"no link named {name!r}; URDF links: {sorted(link_names.values())}")


def _links_under(joints: List[JointInfo], root_joint_indices: Sequence[int]) -> List[int]:
    """Every link at or below the given joints' child links (subtree)."""
    children: Dict[int, List[int]] = {}
    for j in joints:
        children.setdefault(j.parent_link_index, []).append(j.index)
    out: List[int] = []
    stack = list(root_joint_indices)
    # the gripper's mount: walk up to the first FIXED joint above the first gripper joint so the
    # gripper base link (e.g. robotiq_arg2f_base_link) is included
    if stack:
        first = min(stack)
        parent = joints[first].parent_link_index
        if parent >= 0 and not joints[parent].movable:
            stack.append(parent)
    seen = set()
    while stack:
        li = stack.pop()
        if li in seen:
            continue
        seen.add(li); out.append(li)
        stack.extend(children.get(li, []))
    return sorted(out)


def _leaf_links(joints: List[JointInfo], links: Sequence[int]) -> List[int]:
    parents = {j.parent_link_index for j in joints}
    return [li for li in links if li not in parents]


def _mimic_from_urdf(urdf_path, joints: List[JointInfo]) -> Dict[int, tuple]:
    """child joint index -> (parent joint index, multiplier, offset) from URDF <mimic> tags."""
    if not urdf_path or not Path(str(urdf_path)).exists():
        return {}
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except ET.ParseError:
        return {}
    by_name = {j.name: j.index for j in joints}
    out: Dict[int, tuple] = {}
    for jt in root.iter("joint"):
        m = jt.find("mimic")
        if m is None:
            continue
        child, parent = jt.get("name"), m.get("joint")
        if child in by_name and parent in by_name:
            out[by_name[child]] = (by_name[parent], float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
    return out
