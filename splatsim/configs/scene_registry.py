"""Object/scene config registry: per-scene `scene.yaml` files + the legacy
`configs/object_configs/objects.yaml`.

Layout (Hydra-style — the folder tree IS the config structure):

    data/scenes/<scene>/scene.yaml
    data/scenes/<scene>/segmentations/<build>/scene.yaml     (nested: inherits)

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
SCENE_FILE = "scene.yaml"
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
    files = sorted(root.rglob(SCENE_FILE), key=lambda p: len(p.parts))
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
    for name, cfg in _load_scene_tree(SCENES_ROOT).items():
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
    """Persist `updates` into whichever file `name` came from (used by the
    calibration pipeline to store a fitted aabb). Only touches the given keys."""
    path = source_file(name)
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


def invalidate() -> None:
    global _CACHE
    _CACHE = None
    _SOURCE.clear()
    _SCENE_DIR.clear()
