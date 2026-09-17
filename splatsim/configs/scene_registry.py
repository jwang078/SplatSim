"""Object/scene config registry: per-scene `scene.yaml` files + the legacy
`configs/object_configs/objects.yaml`.

Layout (Hydra-style — the folder tree IS the config structure):

    data/scenes/<scene>/scene.yaml
    data/scenes/<scene>/segmentations/<build>/scene.yaml     (nested: inherits)
    data/robots/<robot>/robot.yaml                           (a robot: urdf + optional scan)

Each `scene.yaml` holds exactly what one `objects.yaml` entry used to hold
(`ply_path`, `model_path`, `source_path`, `urdf_path`, `transformation`,
`aabb`, ...) with two differences:

  * paths are relative to the yaml's OWN folder (`splat`, `../../splat/x.ply`),
    so a scene folder is self-contained and can be tarred up and dropped into
    another checkout. A leading `./` keeps its repo-wide meaning of
    "relative to the repo root" (`./splatsim/robot_definitions/...`), and
    absolute paths pass through;
  * a nested scene.yaml inherits every field of the scene.yaml above it and
    overrides only what it sets. A segmentation build therefore does not
    repeat its scan's `transformation` / `aabb`.

`name` defaults to the folder name. Names are flat — env code keeps saying
`splat_name="vine_and_trellis"` regardless of nesting.

`objects.yaml` is deprecated but still loaded; a scene.yaml entry with the same
name overrides it (with a warning, so duplicates get cleaned up rather than
silently shadowed).
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

SCENES_ROOT = SPLATSIM_ROOT / "data" / "scenes"
# Robots are the same kind of entry (urdf_path, splat, transformation, ...)
# but a robot without a scan is not a "scene", so they get their own folder
# and file name. Both roots are walked; both file names are accepted in both.
ROBOTS_ROOT = SPLATSIM_ROOT / "data" / "robots"
SCENE_FILE = "scene.yaml"
ROBOT_FILE = "robot.yaml"
ENTRY_FILES = (SCENE_FILE, ROBOT_FILE)
LEGACY_OBJECTS_YAML = SPLATSIM_ROOT / "configs" / "object_configs" / "objects.yaml"

# Fields whose string values are filesystem paths and get rebased from
# yaml-relative to repo-relative on load.
_PATH_FIELDS = ("ply_path", "model_path", "source_path", "urdf_path")

_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_SOURCE: Dict[str, Path] = {}      # name -> file the entry came from
_SCENE_DIR: Dict[str, Path] = {}   # name -> folder of that scene.yaml (scene entries only)


def _rebase_paths(entry: Dict[str, Any], base: Path) -> Dict[str, Any]:
    """yaml-relative -> repo-relative ('./data/scenes/...') so every downstream
    consumer can keep calling resolve_splatsim_path unchanged."""
    out = dict(entry)
    for k in _PATH_FIELDS:
        v = out.get(k)
        if not isinstance(v, str) or Path(v).is_absolute() or v.startswith("./"):
            # Absolute, or repo-relative by the existing `./` convention
            # (resolve_splatsim_path) — e.g. `./splatsim/robot_definitions/...`
            # for a URDF that ships with the code. Left untouched.
            continue
        # normpath, not resolve(): scene folders may be symlinks to wherever
        # the big files really live, and the repo-relative path must survive.
        joined = Path(os.path.normpath(base / v))
        out[k] = "./" + joined.relative_to(SPLATSIM_ROOT).as_posix()
    return out


def _robot_defaults(cfg: Dict[str, Any]) -> None:
    """A robot folder only has to say `urdf_path`. Fill in what the object
    loader expects of an articulated body; the real joint vectors are
    derived from the URDF by RobotSpec once the body is loaded."""
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


def _load_scene_tree(root: Path) -> Dict[str, Dict[str, Any]]:
    """Walk data/scenes/, resolving inheritance parent-before-child."""
    entries: Dict[str, Dict[str, Any]] = {}
    if not root.is_dir():
        return entries
    files = sorted((f for pat in ENTRY_FILES for f in root.rglob(pat)), key=lambda p: (len(p.parts), str(p)))
    resolved_by_dir: Dict[Path, Dict[str, Any]] = {}
    for f in files:
        folder = f.parent
        raw = yaml.safe_load(f.read_text()) or {}
        raw = _rebase_paths(raw, folder)
        # Nearest ancestor scene.yaml (already resolved because of the sort).
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
        if f.name == ROBOT_FILE or "robot" in cfg:
            _robot_defaults(cfg)
        if name in entries:
            logger.warning("scene registry: duplicate scene name %r (%s and %s); keeping the latter",
                           name, _SOURCE[name], f)
        entries[name] = cfg
        resolved_by_dir[folder] = cfg
        _SOURCE[name] = f
        _SCENE_DIR[name] = folder
    return entries


def load_all() -> Dict[str, Dict[str, Any]]:
    """name -> config dict, for every object known to the registry."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    entries: Dict[str, Dict[str, Any]] = {}
    if LEGACY_OBJECTS_YAML.exists():
        legacy = yaml.safe_load(LEGACY_OBJECTS_YAML.read_text()) or {}
        for name, cfg in legacy.items():
            entries[name] = dict(cfg or {})
            _SOURCE[name] = LEGACY_OBJECTS_YAML
    tree = {}
    for root in (SCENES_ROOT, ROBOTS_ROOT):
        tree.update(_load_scene_tree(root))
    for name, cfg in tree.items():
        if name in entries and _SOURCE.get(name) == LEGACY_OBJECTS_YAML:
            logger.warning("scene registry: %r is defined in both objects.yaml and %s; "
                           "the scene.yaml wins — delete the objects.yaml entry",
                           name, _SOURCE[name])
        entries[name] = cfg
    if not entries:
        logger.warning("scene registry: no object configs found (%s, %s)",
                       SCENES_ROOT, LEGACY_OBJECTS_YAML)
    _CACHE = entries
    return entries


def get(name: str) -> Optional[Dict[str, Any]]:
    return load_all().get(name)


def scene_dir(name: str) -> Path:
    """Folder holding `name`'s scene.yaml. Raises for objects.yaml entries,
    which have no folder of their own."""
    load_all()
    if name not in _SCENE_DIR:
        raise KeyError(f"{name!r} is not a data/scenes entry "
                       f"(source: {_SOURCE.get(name, 'unknown')})")
    return _SCENE_DIR[name]


def source_file(name: str) -> Path:
    load_all()
    return _SOURCE[name]


def write_back(name: str, updates: Dict[str, Any]) -> Path:
    """Persist `updates` into whichever file `name` came from. Only touches
    the given keys. For a per-folder yaml the edit is done line by line so
    the file's comments survive (a robot.yaml is also the user's template);
    objects.yaml entries and anything the line editor can't place fall back
    to a full re-dump."""
    path = source_file(name)
    if path != LEGACY_OBJECTS_YAML:
        text = path.read_text()
        edited = _edit_yaml_lines(text, updates)
        if edited is not None:
            path.write_text(edited)
            invalidate()
            return path
    doc = yaml.safe_load(path.read_text()) or {}
    target = doc if path != LEGACY_OBJECTS_YAML else doc.setdefault(name, {})
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k].update(v)
        else:
            target[k] = v
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    invalidate()
    return path


def _flow(v: Any) -> str:
    """One-line YAML for a scalar or a (nested) list of scalars."""
    return yaml.safe_dump(v, default_flow_style=True, width=10**6).strip()


def _edit_yaml_lines(text: str, updates: Dict[str, Any]) -> Optional[str]:
    """Replace/insert `key: <flow value>` lines in place, keeping every other
    line (comments included). Handles top-level keys and one level of
    nesting (`{"robot": {"initial_joint_positions": [...]}}`). Returns None
    when a value is not representable on one line (nested mappings beyond
    one level, multi-line data) so the caller can fall back."""
    import re as _re
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    def key_line(indent: str, key: str, value: Any) -> str:
        return f"{indent}{key}: {_flow(value)}\n"

    def find_top(key: str):
        for i, ln in enumerate(lines):
            if _re.match(rf"^{_re.escape(key)}\s*:", ln):
                return i
        return None

    def block_end(start: int) -> int:
        """Index one past the last line belonging to the top-level block at `start`."""
        i = start + 1
        while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            # a top-level comment followed by a top-level key ends the block
            if lines[i].lstrip().startswith("#") and not lines[i].startswith(" "):
                j = i
                while j < len(lines) and lines[j].lstrip().startswith("#") and not lines[j].startswith(" "):
                    j += 1
                if j >= len(lines) or not lines[j].startswith(" "):
                    break
            i += 1
        # trailing blank lines belong to the file's layout, not the block
        while i > start + 1 and lines[i - 1].strip() == "":
            i -= 1
        return i

    for key, value in updates.items():
        if isinstance(value, dict):
            top = find_top(key)
            if top is None:
                lines.append(f"{key}:\n"); top = len(lines) - 1
            end = block_end(top)
            for sub, subval in value.items():
                if isinstance(subval, dict):
                    return None
                placed = False
                for i in range(top + 1, end):
                    if _re.match(rf"^\s+{_re.escape(sub)}\s*:", lines[i]):
                        indent = _re.match(r"^(\s+)", lines[i]).group(1)
                        lines[i] = key_line(indent, sub, subval); placed = True
                        break
                if not placed:
                    indent = "  "
                    for i in range(top + 1, end):
                        m = _re.match(r"^(\s+)\S", lines[i])
                        if m and not lines[i].lstrip().startswith("#"):
                            indent = m.group(1); break
                    lines.insert(top + 1, key_line(indent, sub, subval)); end += 1
        else:
            top = find_top(key)
            if top is not None:
                # a block-style value spanning following indented lines is replaced whole
                end = block_end(top)
                lines[top:end] = [key_line("", key, value)]
            else:
                anchor = find_top("urdf_path")
                at = (anchor + 1) if anchor is not None else 0
                lines.insert(at, key_line("", key, value))
    return "".join(lines)


def invalidate() -> None:
    global _CACHE
    _CACHE = None
    _SOURCE.clear()
    _SCENE_DIR.clear()
