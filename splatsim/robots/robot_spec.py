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
      self_collision_skip_pairs: auto | [[link_a, link_b], ...]   # by link name; ADDED to the derived (adjacent + gripper-internal) pairs
      initial_joint_positions: auto | [per arm joint]
      cameras:                                  # any number, [] = none
        - name: wrist                           # observation key "<name>_rgb"
          link: wrist_camera_link
          offset_xyz: [0, 0, 0]
          offset_rpy: [0, 0, 0]
          model: pinhole | fisheye | fisheye_v2 | fisheye_v1
          fov_deg: 60                           # pinhole without intrinsics; omit = the scene camera's
          intrinsics: calibration.json          # a file next to the yaml (what
                                                #   scripts/calibrate_camera_intrinsics.py writes), or inline:
          intrinsics: {width: 1920, height: 1080, fx: 777.9, fy: 767.7, cx: 973.2, cy: 524.3,
                       D: [0.164, -0.153, 0.106, -0.029]}   # D: fisheye only (OpenCV k1..k4)

`model: fisheye` needs `intrinsics`; `fisheye_v1` / `fisheye_v2` are the
calibrations that ship in the code (splatsim/robots/camera_calibrations.py)
and need nothing else. A pinhole with `intrinsics` gets its FoV from fx/fy.
      gripper:
        kind: auto | synergies | per_joint | none
        joints: auto | [joint names]
        open:   [per gripper joint]             # synergies: joint targets at command 0
        closed: [per gripper joint]             #            ... and at command 1
        matrix: [[...]]                         # synergies with K > 1: K x N, q = open + matrix^T c
        stroke_m: 0.085                         # max opening width, for width-based commands (optional)
        tip_link: auto | link name
        contact_links: auto | [link names]

Several arms on one body (a mobile manipulator with two arms, say) replace
`arm_joints` / `ee_link` / `gripper` / `initial_joint_positions` with a list;
each arm gets its own, and the state/action vectors are every arm's joints
in this order followed by every gripper's commands in this order:

    robot:
      base: wheeled
      wheel_joints: [...]
      primary_arm: left                         # goals / `ee_link_index` refer to it (default: first)
      arms:
        - name: left
          ee_link: left_tool0
          joints: auto                          # auto = the movable chain base -> ee_link
          initial_joint_positions: auto | [per arm joint]
          gripper: {kind: ..., ...}             # detected under this arm's ee link
        - name: right
          ee_link: right_tool0
      cameras: [...]                            # cameras are per link, not per arm

`kind: auto` takes the finger-ish joints of the URDF (name contains
finger/knuckle/gripper/...): if the URDF couples them with `<mimic>` tags it
is `synergies` with one command per non-mimic joint and open/closed from the
joint limits (the shipped UR5's Robotiq 2F-85 comes out as one drive joint,
0 = open .. 0.8 = closed — its yaml spells that out), otherwise `per_joint`.
A URDF with no finger-ish joints is `none`. `stroke_m` (max opening width)
is only needed by the width-based `move_gripper` API.

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
    model: str = "pinhole"          # pinhole | fisheye | fisheye_v1 | fisheye_v2
    fov_deg: Optional[float] = None # pinhole horizontal FoV; None = same as the scene's base camera
    # Normalised intrinsics: {width, height, fx, fy, cx, cy, D (fisheye), source}.
    # None = a plain pinhole that follows the scene camera (or fov_deg).
    intrinsics: Optional[Dict[str, Any]] = None

    @property
    def obs_key(self) -> str:
        return f"{self.name}_rgb"

    @property
    def is_fisheye(self) -> bool:
        return self.model.startswith("fisheye")

    @property
    def fisheye_version(self) -> Optional[int]:
        m = re.fullmatch(r"fisheye_v(\d+)", self.model)
        return int(m.group(1)) if m else None


@dataclass
class GripperSpec:
    kind: str                       # synergies | per_joint | none
    joint_indices: List[int] = field(default_factory=list)   # movable gripper joints
    joint_names: List[str] = field(default_factory=list)
    link_indices: List[int] = field(default_factory=list)    # every link under the gripper (incl. fixed)
    open: Optional[np.ndarray] = None       # per joint_indices
    closed: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None     # K x N (synergies)
    mimic: Dict[int, tuple] = field(default_factory=dict)    # child joint idx -> (parent idx, multiplier, offset)
    tip_link_index: Optional[int] = None
    contact_link_indices: List[int] = field(default_factory=list)
    stroke_m: Optional[float] = None        # max opening width (m), for width-based commands

    @property
    def command_dim(self) -> int:
        if self.kind == "none":
            return 0
        if self.kind == "per_joint":
            return len(self.joint_indices)
        if self.kind == "synergies" and self.matrix is not None:
            return int(self.matrix.shape[0])
        return 1  # single synergy (open .. closed)

    def joint_targets(self, command: Sequence[float]) -> np.ndarray:
        """Joint targets (per joint_indices) for a command vector of
        command_dim values in [0, 1] (0 = open)."""
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
class ArmSpec:
    """One kinematic chain with its own tool frame and gripper."""
    name: str
    joint_indices: List[int]                    # planner / IK joints of this arm, in order
    ee_link_index: int
    gripper: GripperSpec

    @property
    def num_dofs(self) -> int:
        return len(self.joint_indices)


@dataclass
class RobotSpec:
    name: str
    urdf_path: str
    base: str                                   # fixed | planar | wheeled
    joints: List[JointInfo]
    link_names: Dict[int, str]                  # link index -> name (-1 = base)
    arms: List[ArmSpec]                         # one or more; arms[primary] is what goals refer to
    arm_joint_indices: List[int]                # every arm's joints, arm after arm
    wheel_joint_indices: List[int]
    gripper: GripperSpec                        # the primary arm's gripper
    cameras: List[CameraSpec]
    ee_link_index: int                          # the primary arm's tool frame
    state_joint_indices: List[int]              # what get_joint_state / teleport vectors index

    @property
    def state_joint_names(self) -> List[str]:
        """URDF joint names in state-vector order (the legend for q vectors)."""
        by_index = {j.index: j.name for j in self.joints}
        return [by_index[i] for i in self.state_joint_indices]
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
    def grippers(self) -> List[GripperSpec]:
        """Every arm's gripper, in arm order (the layout of the gripper part
        of the action vector)."""
        return [a.gripper for a in self.arms]

    @property
    def gripper_command_dim(self) -> int:
        return sum(g.command_dim for g in self.grippers)

    @property
    def action_dim(self) -> int:
        return self.num_dofs + self.gripper_command_dim

    @property
    def primary_arm(self) -> ArmSpec:
        for a in self.arms:
            if a.ee_link_index == self.ee_link_index and a.gripper is self.gripper:
                return a
        return self.arms[0]

    def arm(self, name: str) -> ArmSpec:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"no arm named {name!r} (have: {[a.name for a in self.arms]})")

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
        lines = [f"robot {self.name!r}: {self.urdf_path}", f"  base: {self.base}"]
        for a in self.arms:
            g = a.gripper
            tag = f"arm {a.name!r}" + (" (primary)" if len(self.arms) > 1 and a is self.primary_arm else "")
            lines += [
                f"  {tag}: {a.num_dofs} joints {[self.joints[i].name for i in a.joint_indices]}",
                f"    ee link: {self.link_names[a.ee_link_index]} (index {a.ee_link_index})",
                f"    gripper: {g.kind}" + (f", command dim {g.command_dim}, joints {g.joint_names}" if g.kind != 'none' else ''),
            ]
        lines += [
            f"  cameras ({len(self.cameras)}): " + (", ".join(
                f"{c.name}@{c.link} [{c.model}" + (f", {c.intrinsics['width']}x{c.intrinsics['height']} from {c.intrinsics['source']}" if c.intrinsics else "") + "]"
                for c in self.cameras) or "none"),
            f"  action vector: {self.num_dofs} arm + {self.gripper_command_dim} gripper = {self.action_dim}",
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

        # --- arms + grippers ------------------------------------------------
        arms: List[ArmSpec] = []
        if rb.get("arms"):
            claimed: set = set(wheel_idx)
            for i, acfg in enumerate(rb["arms"]):
                acfg = dict(acfg or {})
                aname = str(acfg.get("name") or f"arm{i}")
                if _auto(acfg.get("ee_link")) and _auto(acfg.get("joints")):
                    raise ValueError(f"{name}: arm {aname!r} needs `ee_link` (or an explicit `joints` list)")
                ee_i = None if _auto(acfg.get("ee_link")) else _link_index(link_names, acfg["ee_link"])
                if _auto(acfg.get("joints")):
                    chain = _chain_to(joints, ee_i)
                    a_idx = [j for j in chain if joints[j].movable and j not in claimed]
                else:
                    a_idx = [_find_joint(joints, n).index for n in acfg["joints"]]
                if not a_idx:
                    raise ValueError(f"{name}: arm {aname!r} has no movable joints between the base and {acfg.get('ee_link')!r}")
                if ee_i is None:
                    ee_i = a_idx[-1]
                # this arm's gripper lives under its tool link (or its last joint)
                sub = _links_under(joints, [ee_i])
                g, g_all = _derive_gripper(dict(acfg.get("gripper") or {}), joints, movable, link_names, wheel_idx,
                                           legacy=False, use_gripper_legacy=True, urdf_path=urdf_path or cfg.get("urdf_path"),
                                           name=f"{name}/{aname}", candidate_links=set(sub))
                a_idx = [j for j in a_idx if j not in g_all]
                claimed |= set(a_idx) | set(g_all)
                arms.append(ArmSpec(name=aname, joint_indices=a_idx, ee_link_index=ee_i, gripper=g))
        else:
            g, g_joint_idx = _derive_gripper(dict(rb.get("gripper") or {}), joints, movable, link_names, wheel_idx,
                                             legacy=legacy, use_gripper_legacy=bool(cfg.get("use_gripper", True)),
                                             urdf_path=urdf_path or cfg.get("urdf_path"), name=name)
            if _auto(rb.get("arm_joints")):
                arm_idx = [j.index for j in movable if j.index not in g_joint_idx and j.index not in wheel_idx]
                if legacy:
                    n_legacy = int(cfg.get("num_dofs") or 0)
                    if n_legacy:
                        arm_idx = arm_idx[:n_legacy]
            else:
                arm_idx = [_find_joint(joints, n).index for n in rb["arm_joints"]]
            if not arm_idx:
                raise ValueError(f"{name}: no arm joints found in the URDF (no movable non-gripper joints)")
            arms.append(ArmSpec(name=str(rb.get("name") or "arm"), joint_indices=arm_idx, ee_link_index=-2, gripper=g))
        arm_idx = [j for a in arms for j in a.joint_indices]
        commanded = [j for a in arms for j in a.gripper.joint_indices]
        gripper = arms[0].gripper if _auto(rb.get("primary_arm")) else next(a for a in arms if a.name == rb["primary_arm"]).gripper
        primary = arms[0] if _auto(rb.get("primary_arm")) else next(a for a in arms if a.name == rb["primary_arm"])

        # --- cameras -------------------------------------------------------
        cams: List[CameraSpec] = []
        yaml_dir = Path(cfg.get("entry_dir") or Path(urdf_path or cfg.get("urdf_path") or ".").parent)
        if "cameras" in rb:
            for c in rb["cameras"] or []:
                model = str(c.get("model", "pinhole")).lower()
                cams.append(CameraSpec(
                    name=str(c["name"]), link=str(c["link"]), link_index=_link_index(link_names, c["link"]),
                    offset_xyz=tuple(c.get("offset_xyz", (0, 0, 0))), offset_rpy=tuple(c.get("offset_rpy", (0, 0, 0))),
                    model=model, fov_deg=(float(c["fov_deg"]) if c.get("fov_deg") is not None else None),
                    intrinsics=_camera_intrinsics(c, model, yaml_dir, f"{name}/{c['name']}")))
        elif cfg.get("wrist_camera_link_name"):
            ver = wrist_cam_ver if wrist_cam_ver else 0
            model = f"fisheye_v{ver}" if ver else "pinhole"
            cams.append(CameraSpec(name="wrist", link=cfg["wrist_camera_link_name"],
                                   link_index=_link_index(link_names, cfg["wrist_camera_link_name"]),
                                   model=model, intrinsics=_camera_intrinsics({}, model, yaml_dir, f"{name}/wrist")))
        seen = set()
        for c in cams:
            if c.name in seen:
                raise ValueError(f"{name}: duplicate camera name {c.name!r}")
            seen.add(c.name)

        # --- EE link -------------------------------------------------------
        if rb.get("arms"):
            ee = primary.ee_link_index
        else:
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
            arms[0].ee_link_index = ee

        # --- state vector, initial positions, signs -------------------------
        if legacy:
            state_idx = list(range(1, len(joints)))          # historical: joints 1..N-1, all types
        else:
            state_idx = arm_idx + commanded + wheel_idx
        art = cfg.get("articulation_config") or {}
        init_src = rb.get("initial_joint_positions")
        if rb.get("arms"):
            per_arm = [a.get("initial_joint_positions") for a in rb["arms"]]
            if any(not _auto(v) for v in per_arm):
                init_src = []
                for a, v in zip(arms, per_arm):
                    v = [] if _auto(v) else list(v)
                    if len(v) != a.num_dofs:
                        raise ValueError(f"{name}: arm {a.name!r} initial_joint_positions needs {a.num_dofs} values, got {len(v)}")
                    init_src += v
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
        if not legacy:
            # adjacent links always; every gripper-internal pair (linkages overlap by design)
            for j in joints:
                skip.append((j.parent_link_index, j.index))
            for arm in arms:                      # within ONE gripper, never across grippers
                gl = arm.gripper.link_indices
                for a in gl:
                    for b in gl:
                        if a < b:
                            skip.append((a, b))
        # Declared pairs ADD to the derived ones (they name the URDF's own mesh
        # artifacts, e.g. the UR5's forearm <-> wrist_2 floor); `auto` = derived only.
        if not _auto(rb.get("self_collision_skip_pairs")):
            skip += [(_link_index(link_names, a), _link_index(link_names, b)) for a, b in rb["self_collision_skip_pairs"]]
        # legacy: the server's class-level lists apply (see PybulletRobotServerBase)

        return cls(
            name=name, urdf_path=str(urdf_path or cfg.get("urdf_path")), base=base, joints=joints,
            link_names=link_names, arms=arms, arm_joint_indices=arm_idx, wheel_joint_indices=wheel_idx,
            gripper=gripper, cameras=cams, ee_link_index=ee, state_joint_indices=state_idx,
            initial_joint_positions=init, joint_signs=signs, self_collision_skip_pairs=skip,
            legacy=legacy, has_splat=bool(cfg.get("model_path") or cfg.get("ply_path")),
        )


# ---------------------------------------------------------------- helpers
def _camera_intrinsics(c: Dict[str, Any], model: str, yaml_dir: Path, who: str) -> Optional[Dict[str, Any]]:
    """Normalise a camera's intrinsics to {width, height, fx, fy, cx, cy, D, source}.

    Sources, in order: `intrinsics:` as a path (relative to the yaml) to the
    JSON from scripts/calibrate_camera_intrinsics.py or to a JSON/YAML with
    the normalised keys (or a 3x3 `K`); `intrinsics:` written inline; or the
    shipped table for `fisheye_vN`. Plain pinholes return None."""
    import json
    src = c.get("intrinsics")
    data: Optional[Dict[str, Any]] = None
    source = "inline"
    if isinstance(src, str):
        path = Path(src) if Path(src).is_absolute() else (yaml_dir / src)
        if not path.exists():
            raise FileNotFoundError(f"{who}: intrinsics file {path} not found")
        text = path.read_text()
        data = json.loads(text) if path.suffix.lower() == ".json" else __import__("yaml").safe_load(text)
        source = str(path)
    elif isinstance(src, dict):
        data = dict(src)
    m = re.fullmatch(r"fisheye_v(\d+)", model)
    if data is None:
        if m:
            from splatsim.robots.camera_calibrations import shipped_intrinsics
            return shipped_intrinsics(int(m.group(1)))
        if model == "fisheye":
            raise ValueError(f"{who}: model fisheye needs `intrinsics:` (a calibration file or fx/fy/cx/cy/width/height/D)")
        return None
    # scripts/calibrate_camera_intrinsics.py output -> normalised keys
    if "fisheye_camera_matrix" in data or "fisheye_fx" in data:
        K = data.get("fisheye_camera_matrix")
        out = {"width": data["image_width"], "height": data["image_height"],
               "fx": data.get("fisheye_fx", K and K[0][0]), "fy": data.get("fisheye_fy", K and K[1][1]),
               "cx": data.get("fisheye_cx", K and K[0][2]), "cy": data.get("fisheye_cy", K and K[1][2]),
               "D": list(data.get("fisheye_dist_coeffs", []))}
    else:
        K = data.get("K")
        out = {"width": data.get("width", data.get("image_width")), "height": data.get("height", data.get("image_height")),
               "fx": data.get("fx", K and K[0][0]), "fy": data.get("fy", K and K[1][1]),
               "cx": data.get("cx", K and K[0][2]), "cy": data.get("cy", K and K[1][2]),
               "D": list(data.get("D", data.get("dist_coeffs", [])))}
    missing = [k for k in ("width", "height", "fx", "fy", "cx", "cy") if out.get(k) is None]
    if missing:
        raise ValueError(f"{who}: intrinsics missing {missing} (have keys {sorted(data)})")
    if model.startswith("fisheye") and len(out["D"]) != 4:
        raise ValueError(f"{who}: a fisheye needs 4 distortion coefficients (OpenCV fisheye k1..k4), got {len(out['D'])}")
    out = {k: (float(v) if k not in ("D",) else [float(x) for x in v]) for k, v in out.items()}
    out["width"], out["height"] = int(out["width"]), int(out["height"])
    out["source"] = source
    return out


def _derive_gripper(gcfg: Dict[str, Any], joints: List[JointInfo], movable: List[JointInfo],
                    link_names: Dict[int, str], wheel_idx: List[int], *, legacy: bool, use_gripper_legacy: bool,
                    urdf_path, name: str, candidate_links: Optional[set] = None):
    """Gripper of one arm from its `gripper:` block. `candidate_links`
    restricts auto-detection to a subtree (a multi-arm robot's arm). Returns
    (GripperSpec, every movable gripper joint index incl. mimic children)."""
    def _auto(v): return v is None or (isinstance(v, str) and v.lower() == "auto")
    def _in(j): return candidate_links is None or j.index in candidate_links
    kind = str(gcfg.get("kind") or "auto").lower()
    if kind == "robotiq_2f85":
        raise ValueError(f"{name}: gripper.kind robotiq_2f85 is gone — the Robotiq is an ordinary `synergies` "
                         f"gripper now; see data/assets/ur5/asset.yaml (joints: [finger_joint], open/closed, stroke_m)")
    auto_g = [j for j in movable if _in(j) and _GRIPPER_NAME_RE.search(j.name) and not _WHEEL_NAME_RE.search(j.name)]
    if kind == "auto":
        if legacy and not use_gripper_legacy:
            kind = "none"
        elif not auto_g:
            kind = "none"
        else:
            # coupled fingers (URDF <mimic>) -> one command per drive joint
            mim = _mimic_from_urdf(urdf_path, joints)
            kind = "synergies" if any(j.index in mim for j in auto_g) else "per_joint"
    if kind not in ("none", "synergies", "per_joint"):
        raise ValueError(f"{name}: gripper.kind must be auto | synergies | per_joint | none, got {kind!r}")

    if kind == "none":
        g_roots: List[int] = []
    elif _auto(gcfg.get("joints")):
        g_roots = [j.index for j in auto_g]
    else:
        g_roots = [_find_joint(joints, n).index for n in gcfg["joints"]]
    g_roots = [i for i in g_roots if i not in wheel_idx]
    # The gripper is a SUBTREE: every movable joint whose link hangs below
    # the declared/detected joints belongs to it (mimic children, extra
    # finger joints) even if it wasn't named — otherwise it would be
    # mistaken for an arm joint.
    g_links = _links_under(joints, g_roots) if g_roots else []
    g_joint_idx = sorted(set(g_roots) | {j.index for j in movable if j.index in g_links and j.index not in wheel_idx})

    g_mimic = _mimic_from_urdf(urdf_path, joints) if kind in ("synergies", "per_joint") else {}
    g_mimic = {c: pm for c, pm in g_mimic.items() if c in g_joint_idx}
    # mimic children are driven by their parent, not commanded
    commanded = [i for i in g_joint_idx if i not in g_mimic]
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
    stroke = float(gcfg["stroke_m"]) if gcfg.get("stroke_m") is not None else None
    return GripperSpec(kind=kind, joint_indices=commanded, joint_names=[joints[i].name for i in commanded],
                       link_indices=g_links, open=gopen, closed=gclosed, matrix=gmat, mimic=g_mimic,
                       tip_link_index=tip_idx, contact_link_indices=contact, stroke_m=stroke), g_joint_idx


def _chain_to(joints: List[JointInfo], link_index: int) -> List[int]:
    """Joint indices from the base down to `link_index` (a link's index is
    the index of the joint whose child it is)."""
    chain: List[int] = []
    j = link_index
    while j is not None and j >= 0:
        chain.append(j)
        j = joints[j].parent_link_index
    return chain[::-1]


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
