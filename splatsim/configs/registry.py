"""Asset / stage registry: every folder under `data/` that has an entry yaml,
plus the legacy `configs/object_configs/objects.yaml`.

Layout (Hydra-style — the folder tree IS the config structure; the words are
USD's: an *asset* is a reusable body, a *stage* places assets and carries
what was scanned):

    data/assets/<asset>/asset.yaml           a URDF body (a robot, a box, an
                                             engine) + its meshes. Self-contained.
    data/stages/<stage>/stage.yaml           a scan: splat + alignment transform,
                                             optionally `asset: <name>` for the
                                             body it is a scan of
    data/stages/<stage>/segmentations/<build>/stage.yaml   (nested: inherits)

A stage that contains several segmented bodies lists them under `assets:`,
one block per instance (the robot, a box, an apple ...), each with what is
true of THAT body in THIS scan — `asset:` (or urdf_path), base_position,
scan_pose, aabb, labels_path. The stage's own fields (splat, transformation)
are shared. Every instance is a registry entry `<stage>/<instance>`; the
bare stage name resolves to the instance called `robot` (else the only /
first one) so envs keep saying `splat_name="<stage>"` for the robot, and the
entry also carries `exclude_aabbs` (every instance's box) for when the
stage is loaded as the background.

Each entry holds exactly what one `objects.yaml` entry used to hold
(`ply_path`, `model_path`, `source_path`, `urdf_path`, `labels_path`,
`transformation`, `aabb`, ...) with three differences:

  * paths are relative to the yaml's OWN folder (`splat`, `../../splat/x.ply`),
    so a folder is self-contained and can be tarred up and dropped into
    another checkout. A leading `./` keeps its repo-wide meaning of
    "relative to the repo root" (`./splatsim/robot_definitions/...`), and
    absolute paths pass through;
  * a nested stage.yaml inherits every field of the stage.yaml above it and
    overrides only what it sets. A segmentation build therefore does not
    repeat its scan's `transformation` / `aabb`;
  * `asset: ur5` in a stage pulls in that asset's fields (`urdf_path`, the
    `robot:` block, ...) underneath the stage's own — the stage's opinions
    win, like a USD reference with overrides on the referencing prim.

`name` defaults to the folder name. Names are flat — env code keeps saying
`splat_name="vine_and_trellis"` regardless of nesting.

Still accepted, so nothing already downloaded breaks: the old roots
`data/robots/` and `data/scenes/`, the old file names `robot.yaml` and
`scene.yaml`, and `objects.yaml` (a folder entry with the same name overrides
it, with a warning, so duplicates get cleaned up rather than silently
shadowed).
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from splatsim.utils.paths import SPLATSIM_ROOT

logger = logging.getLogger(__name__)

ASSETS_ROOT = SPLATSIM_ROOT / "data" / "assets"
STAGES_ROOT = SPLATSIM_ROOT / "data" / "stages"
ASSET_FILE = "asset.yaml"
STAGE_FILE = "stage.yaml"
# Pre-2026-09-18 names. Walked / accepted so an existing checkout or an
# already-extracted tarball keeps working; new folders should use the above.
LEGACY_ROOTS = (SPLATSIM_ROOT / "data" / "robots", SPLATSIM_ROOT / "data" / "scenes")
LEGACY_ENTRY_FILES = ("robot.yaml", "scene.yaml")
ENTRY_FILES = (ASSET_FILE, STAGE_FILE) + LEGACY_ENTRY_FILES
LEGACY_OBJECTS_YAML = SPLATSIM_ROOT / "configs" / "object_configs" / "objects.yaml"
# Where per-Gaussian link labels used to be kept, keyed by entry name.
LEGACY_LABELS_DIR = SPLATSIM_ROOT / "data" / "labels_path"

# Fields whose string values are filesystem paths and get rebased from
# yaml-relative to repo-relative on load.
_PATH_FIELDS = ("ply_path", "model_path", "source_path", "urdf_path", "labels_path")

_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_SOURCE: Dict[str, Path] = {}      # name -> file the entry came from
_DIR: Dict[str, Path] = {}         # name -> folder of that entry (folder entries only)


def _rebase_paths(entry: Dict[str, Any], base: Path) -> Dict[str, Any]:
    """yaml-relative -> repo-relative ('./data/stages/...') so every downstream
    consumer can keep calling resolve_splatsim_path unchanged."""
    out = dict(entry)
    for k in _PATH_FIELDS:
        v = out.get(k)
        if not isinstance(v, str) or Path(v).is_absolute() or v.startswith("./"):
            # Absolute, or repo-relative by the existing `./` convention
            # (resolve_splatsim_path) — e.g. `./splatsim/robot_definitions/...`
            # for a URDF that ships with the code. Left untouched.
            continue
        # normpath, not resolve(): folders may be symlinks to wherever the
        # big files really live, and the repo-relative path must survive.
        joined = Path(os.path.normpath(base / v))
        out[k] = "./" + joined.relative_to(SPLATSIM_ROOT).as_posix()
    return out


def _robot_defaults(cfg: Dict[str, Any]) -> None:
    """A robot asset only has to say `urdf_path` and have a `robot:` block.
    Fill in what the object loader expects of an articulated body; the real
    joint vectors are derived from the URDF by RobotSpec once the body is
    loaded."""
    cfg.setdefault("is_articulated", True)
    cfg.setdefault("articulation_config", {"initial_joint_positions": [], "joint_signs": None})
    cfg.setdefault("robot", {})
    # A wheeled base is a free body on the ground plane the server adds for
    # it; everything else is bolted down.
    wheeled = str((cfg["robot"] or {}).get("base", "fixed")).lower() == "wheeled"
    cfg.setdefault("use_fixed_base", not wheeled)
    cfg.setdefault("base_position", [0.0, 0.0, 0.0])


def _merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Child overrides parent, one level deep for mapping values (so a child can
    override `aabb.bounding_box` without restating `aabb.urdf_bbox_adjustment`)."""
    out = copy.deepcopy(parent)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_tree(root: Path) -> Dict[str, Dict[str, Any]]:
    """Walk one root, resolving folder inheritance parent-before-child."""
    entries: Dict[str, Dict[str, Any]] = {}
    if not root.is_dir():
        return entries
    files = sorted((f for pat in ENTRY_FILES for f in root.rglob(pat)), key=lambda p: (len(p.parts), str(p)))
    # A tarball made before the rename may drop a `scene.yaml` / `robot.yaml`
    # next to the `stage.yaml` / `asset.yaml` that git tracks: the tracked
    # file is the current one, so the old name is ignored, not merged.
    files = [f for f in files
             if not (f.name in LEGACY_ENTRY_FILES and any((f.parent / n).exists() for n in (ASSET_FILE, STAGE_FILE)))]
    resolved_by_dir: Dict[Path, Dict[str, Any]] = {}
    for f in files:
        folder = f.parent
        raw = yaml.safe_load(f.read_text()) or {}
        raw = _rebase_paths(raw, folder)
        if isinstance(raw.get("assets"), dict):
            raw["assets"] = {k: _rebase_paths(dict(v or {}), folder) for k, v in raw["assets"].items()}
        # Nearest ancestor entry (already resolved because of the sort).
        parent_cfg: Dict[str, Any] = {}
        for anc in folder.parents:
            if anc in resolved_by_dir:
                parent_cfg = resolved_by_dir[anc]
                break
            if anc == root:
                break
        cfg = _merge({k: v for k, v in parent_cfg.items() if k != "name"}, raw)
        name = cfg.get("name") or folder.name
        cfg["name"] = name
        cfg["entry_dir"] = str(folder)     # where relative paths inside nested blocks (camera intrinsics) resolve
        if name in entries:
            logger.warning("registry: duplicate entry name %r (%s and %s); keeping the latter",
                           name, _SOURCE[name], f)
        entries[name] = cfg
        resolved_by_dir[folder] = cfg
        _SOURCE[name] = f
        _DIR[name] = folder
    return entries


def _flatten_stage_assets(entries: Dict[str, Dict[str, Any]]) -> None:
    """`assets: {instance: {...}}` in a stage -> one entry per instance."""
    for name in list(entries):
        cfg = entries[name]
        inst = cfg.get("assets")
        if not isinstance(inst, dict) or not inst:
            continue
        shared = {k: v for k, v in cfg.items() if k != "assets"}
        boxes = []
        flat: Dict[str, Dict[str, Any]] = {}
        for iname, icfg in inst.items():
            icfg = dict(icfg or {})
            if "aabb" in icfg:
                boxes.append(copy.deepcopy(icfg["aabb"]))
            e = _merge({k: v for k, v in shared.items() if k != "name"}, icfg)
            e["name"] = f"{name}/{iname}"
            e["stage"] = name
            e["instance"] = iname
            flat[e["name"]] = e
            _SOURCE[e["name"]] = _SOURCE[name]
            _DIR[e["name"]] = _DIR[name]
        primary = f"{name}/robot" if f"{name}/robot" in flat else next(iter(flat))
        alias = copy.deepcopy(flat[primary])
        alias["name"] = name
        alias["exclude_aabbs"] = boxes
        entries[name] = alias
        entries.update(flat)


_ASSET_BODY_FIELDS = ("urdf_path", "robot", "use_fixed_base", "is_articulated", "articulation_config",
                      "base_position", "base_orientation_rpy", "num_dofs", "use_gripper", "wrist_camera_link_name")


def _resolve_asset_refs(entries: Dict[str, Dict[str, Any]]) -> None:
    """`asset: <name>` -> that entry's fields underneath this one's."""
    def resolve(name: str, chain: tuple) -> Dict[str, Any]:
        cfg = entries[name]
        ref = cfg.get("asset")
        if not ref or cfg.get("_asset_resolved"):
            return cfg
        if ref in chain:
            raise ValueError(f"registry: circular asset reference {' -> '.join(chain + (ref,))}")
        if ref not in entries:
            raise KeyError(f"registry: {name!r} ({_SOURCE.get(name)}) references asset {ref!r}, "
                           f"which is not under {ASSETS_ROOT} (known: {sorted(entries)})")
        base = resolve(ref, chain + (name,))
        # Only what describes the BODY comes across. A referenced entry may
        # itself be a scan (a box scanned once keeps its URDF in its stage);
        # its splat, transform, box, labels and scan pose belong to that scan.
        body = {k: v for k, v in base.items() if k in _ASSET_BODY_FIELDS}
        merged = _merge(body, cfg)
        merged["_asset_resolved"] = True
        entries[name] = merged
        return merged

    for name in list(entries):
        resolve(name, ())
    for cfg in entries.values():
        cfg.pop("_asset_resolved", None)
        if "robot" in cfg:
            _robot_defaults(cfg)


def load_all() -> Dict[str, Dict[str, Any]]:
    """name -> config dict, for every asset / stage known to the registry."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    entries: Dict[str, Dict[str, Any]] = {}
    if LEGACY_OBJECTS_YAML.exists():
        legacy = yaml.safe_load(LEGACY_OBJECTS_YAML.read_text()) or {}
        for name, cfg in legacy.items():
            entries[name] = dict(cfg or {})
            _SOURCE[name] = LEGACY_OBJECTS_YAML
    tree: Dict[str, Dict[str, Any]] = {}
    for root in (ASSETS_ROOT, STAGES_ROOT) + LEGACY_ROOTS:
        tree.update(_load_tree(root))
    _flatten_stage_assets(tree)
    _resolve_asset_refs(tree)
    for name, cfg in tree.items():
        if name in entries and _SOURCE.get(name) == LEGACY_OBJECTS_YAML:
            logger.warning("registry: %r is defined in both objects.yaml and %s; "
                           "the folder entry wins — delete the objects.yaml entry",
                           name, _SOURCE[name])
        entries[name] = cfg
    if not entries:
        logger.warning("registry: no entries found (%s, %s, %s)",
                       ASSETS_ROOT, STAGES_ROOT, LEGACY_OBJECTS_YAML)
    _CACHE = entries
    return entries


def get(name: str) -> Optional[Dict[str, Any]]:
    return load_all().get(name)


def entry_dir(name: str) -> Path:
    """Folder holding `name`'s yaml. Raises for objects.yaml entries, which
    have no folder of their own."""
    load_all()
    if name not in _DIR:
        raise KeyError(f"{name!r} is not a data/assets or data/stages entry "
                       f"(source: {_SOURCE.get(name, 'unknown')})")
    return _DIR[name]


stage_dir = entry_dir
scene_dir = entry_dir   # pre-rename spelling


def labels_path(name: str) -> Path:
    """Per-Gaussian link labels of `name`'s splat (`labels_path` in its yaml,
    relative to the yaml). Falls back to the old shared folder
    `data/labels_path/<name>_labels.npy` for entries that predate the field."""
    from splatsim.utils.paths import resolve_splatsim_path
    cfg = get(name) or {}
    if cfg.get("labels_path"):
        return Path(resolve_splatsim_path(cfg["labels_path"]))
    return LEGACY_LABELS_DIR / f"{name}_labels.npy"


def labels_key(name: str) -> Optional[Dict[str, Any]]:
    """The legend written next to `labels_path` (`<stem>.json`): `classes`
    maps each integer in the labels array to a name, `source` says how the
    labels were made. Names are just strings — a robot scan's classes are its
    URDF link names, a vine segmentation's could be "grapes" / "trunk" — and
    `load_labels` is where names get matched to whatever the consumer needs.
    None for arrays labelled before the legend existed."""
    import json
    key = labels_path(name).with_suffix(".json")
    if not key.exists():
        return None
    return json.loads(key.read_text())


def write_labels_key(labels_file: Path, classes: Dict[int, str], source: str) -> Path:
    """Write the legend for a labels array next to it (`<stem>.json`).
    `classes`: label value -> name. `source`: how the labels were produced."""
    import json
    key = {
        "labels_file": Path(labels_file).name,
        "meaning": "one value per Gaussian of the splat (same order as its point_cloud.ply); "
                   "`classes` says what each value is",
        "source": source,
        "classes": {str(int(k)): str(v) for k, v in sorted(classes.items(), key=lambda kv: int(kv[0]))},
    }
    out = Path(labels_file).with_suffix(".json")
    out.write_text(json.dumps(key, indent=2) + "\n")
    return out


def load_labels(name: str, client=None, body_id: Optional[int] = None):
    """`name`'s labels as an int array. With a PyBullet body, the values are
    remapped to THAT body's link indices by matching class names to its link
    names (-1 = base), so labels survive a URDF whose link order differs from
    the one they were made against, and classes that are not links of this
    body (a heuristic's "background", say) become -1. Without a key file the
    array is returned as is (the pre-legend convention: values already are
    link indices of the entry's URDF)."""
    import numpy as np
    labels = np.asarray(np.load(labels_path(name))).astype(np.int64)
    key = labels_key(name)
    if key is None or client is None or body_id is None:
        return labels
    link_of = {client.getBodyInfo(body_id)[0].decode(): -1}
    for j in range(client.getNumJoints(body_id)):
        link_of[client.getJointInfo(body_id, j)[12].decode()] = j
    lut: Dict[int, int] = {}
    unmapped = []
    for v, cname in key.get("classes", {}).items():
        if cname in link_of:
            lut[int(v)] = link_of[cname]
        else:
            lut[int(v)] = -1
            unmapped.append(cname)
    if unmapped:
        logger.warning("labels for %r: classes %s are not links of the loaded body; treated as base (-1)",
                       name, unmapped)
    out = np.full_like(labels, -1)
    for v, li in lut.items():
        out[labels == v] = li
    return out


def source_file(name: str) -> Path:
    load_all()
    return _SOURCE[name]


def write_back(name: str, updates: Dict[str, Any]) -> Path:
    """Persist `updates` into whichever file `name` came from. Only touches
    the given keys. For a per-folder yaml the edit is done line by line so
    the file's comments survive (an asset.yaml is also the user's template);
    objects.yaml entries and anything the line editor can't place fall back
    to a full re-dump. A `<stage>/<instance>` entry is edited under its
    `assets: <instance>:` block."""
    path = source_file(name)
    cfg = get(name) or {}
    if cfg.get("instance") and "/" in name:
        updates = {"assets": {cfg["instance"]: updates}}
    if path != LEGACY_OBJECTS_YAML:
        text = path.read_text()
        edited = _edit_yaml_lines(text, updates)
        if edited is not None:
            path.write_text(edited)
            invalidate()
            return path
    doc = yaml.safe_load(path.read_text()) or {}
    target = doc if path != LEGACY_OBJECTS_YAML else doc.setdefault(name, {})
    _deep_update(target, updates)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    invalidate()
    return path


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v


def _flow(v: Any) -> str:
    """One-line YAML for a scalar or a (nested) list of scalars."""
    out = yaml.safe_dump(v, default_flow_style=True, width=10**6).strip()
    return out.split("\n")[0] if out.endswith("\n...") or "\n..." in out else out


def _edit_yaml_lines(text: str, updates: Dict[str, Any]) -> Optional[str]:
    """Replace/insert `key: <flow value>` lines in place, at any nesting
    depth, keeping every other line (comments included). A mapping value is
    written key by key inside the existing block (created if missing); a
    scalar or list value is one flow-style line that replaces the key's
    current line and whatever block hung under it. Returns None only for
    values it cannot express (nothing, currently) so callers can fall back."""
    import re as _re
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    def indent_of(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    def is_blank_or_comment(ln: str) -> bool:
        return ln.strip() == "" or ln.lstrip().startswith("#")

    def find_key(start: int, end: int, indent: int, key: str):
        """Index of `key:` at exactly `indent` within lines[start:end]."""
        pat = _re.compile(rf"^ {{{indent}}}{_re.escape(key)}\s*:")
        for i in range(start, end):
            if pat.match(lines[i]):
                return i
        return None

    def block_end(start: int, indent: int) -> int:
        """One past the last line of the block whose key is at lines[start]
        (child lines are indented deeper; comments/blanks inside count)."""
        i = start + 1
        while i < len(lines) and (is_blank_or_comment(lines[i]) or indent_of(lines[i]) > indent
                                  or (indent_of(lines[i]) == indent and lines[i].lstrip().startswith("- "))):
            i += 1
        while i > start + 1 and is_blank_or_comment(lines[i - 1]) and indent_of(lines[i - 1]) <= indent:
            i -= 1
        return i

    def child_indent(start: int, end: int, indent: int) -> int:
        for i in range(start + 1, end):
            if not is_blank_or_comment(lines[i]):
                return indent_of(lines[i])
        return indent + 2

    def apply(updates_: Dict[str, Any], start: int, end: int, indent: int) -> int:
        """Apply within lines[start:end] at `indent`; returns the new end."""
        for key, value in updates_.items():
            at = find_key(start, end, indent, key)
            if isinstance(value, dict):
                if at is None:
                    lines.insert(end, " " * indent + f"{key}:\n"); at = end; end += 1
                    bend = at + 1
                else:
                    bend = block_end(at, indent)
                    # `key: {inline}` or `key: value` -> make it a block
                    if lines[at].split(":", 1)[1].strip() not in ("", ) and not lines[at].split(":", 1)[1].strip().startswith("#"):
                        lines[at] = " " * indent + f"{key}:\n"; bend = at + 1
                ci = child_indent(at, bend, indent)
                new_bend = apply(value, at + 1, bend, ci)
                end += new_bend - bend
            else:
                line = " " * indent + f"{key}: {_flow(value)}"
                if at is None:
                    lines.insert(end, line + "\n"); end += 1
                else:
                    # keep a trailing `# comment` on the line being replaced
                    m = _re.search(r"\s+(#.*)$", lines[at].rstrip("\n"))
                    if m and not _re.match(r"^\s*[^#]*:\s*#", lines[at]) or (m and lines[at].split(":", 1)[1].strip().startswith("#")):
                        line += "   " + m.group(1)
                    bend = block_end(at, indent)
                    lines[at:bend] = [line + "\n"]
                    end -= (bend - at - 1)
        return end

    apply(updates, 0, len(lines), 0)
    return "".join(lines)


def invalidate() -> None:
    global _CACHE
    _CACHE = None
    _SOURCE.clear()
    _DIR.clear()
