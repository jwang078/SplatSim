"""Generic, schema-drift-tolerant JSON export/import for dataclass configs.

Everything here is driven by ``dataclasses.fields()`` — no config field is ever
named — so changing a config's schema needs NO edits here or in callers:

  * ADD a field     -> old files simply lack it; the class default fills in.
  * REMOVE / RENAME -> unknown keys in a file are ignored (old name dropped, new
                       name gets its default until re-exported).
  * CHANGE a type   -> values load as native JSON; Enums, tuples and nested
                       dataclasses are coerced back using the LIVE default's type.
  * CHANGE a range  -> not validated here; values load as-is (the config /
                       downstream owns validation).

Fields whose value can't be JSON-encoded (e.g. a callable ``cuboids_fn``) are
skipped on export and fall back to the class default on import.

Use with any dataclass:
    save_dataclass_json(cfg, "cfg.json")
    cfg = load_dataclass_json(TrajectoryGenModeConfig, "cfg.json")   # new instance
    update_dataclass_json(existing_cfg, "cfg.json")                  # in place
"""
import dataclasses
import enum
import json
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar("T")


def to_jsonable(value: Any) -> Any:
    """Recursively convert a config value to JSON-native types.

    Nested dataclasses -> dicts, Enums -> their ``.value``, tuples -> lists. Leaf
    values that json can't encode raise in the caller, which then skips the field.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def dataclass_to_dict(obj) -> dict:
    """Field-driven JSON-able dict of a dataclass, skipping any field that can't
    be serialized (a callable, a live handle, ...). Those import as the default."""
    out: dict = {}
    for f in dataclasses.fields(obj):
        try:
            encoded = to_jsonable(getattr(obj, f.name))
            json.dumps(encoded)  # prove it round-trips before committing to it
        except (TypeError, ValueError):
            continue
        out[f.name] = encoded
    return out


def _coerce(value: Any, default: Any) -> Any:
    """Coerce a JSON-loaded ``value`` toward the shape of the live ``default`` so
    a datatype change across versions still loads. Best-effort — on failure the
    raw value is kept (the config/downstream can reject it)."""
    if default is not None:
        if isinstance(default, enum.Enum):
            try:
                return type(default)(value)
            except (ValueError, KeyError):
                return default
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            return apply_dict(default, value)  # recurse into a nested dataclass
        if isinstance(default, tuple) and isinstance(value, list):
            return tuple(value)  # JSON has no tuples; restore tuple-ness
    return value


def apply_dict(obj: T, data: dict, warn: Optional[Callable[[str], None]] = None) -> T:
    """Update dataclass ``obj`` IN PLACE from ``data``; return it.

    Tolerant of schema drift: unknown keys are ignored (optionally reported via
    ``warn``); fields absent from ``data`` keep ``obj``'s current value.
    """
    field_names = {f.name for f in dataclasses.fields(obj)}
    if warn:
        unknown = [k for k in data if k not in field_names]
        if unknown:
            warn(f"ignoring {len(unknown)} unknown config field(s): {sorted(unknown)}")
    for f in dataclasses.fields(obj):
        if f.name in data:
            setattr(obj, f.name, _coerce(data[f.name], getattr(obj, f.name)))
    return obj


def save_dataclass_json(obj, path) -> list:
    """Export a dataclass instance to a JSON file. Returns the names of any fields
    skipped because they weren't serializable."""
    encoded = dataclass_to_dict(obj)
    skipped = [f.name for f in dataclasses.fields(obj) if f.name not in encoded]
    with open(path, "w") as fp:
        json.dump(encoded, fp, indent=2, sort_keys=True)
    return skipped


def load_dataclass_json(cls: Type[T], path, warn: Optional[Callable[[str], None]] = None) -> T:
    """Import and return a fresh ``cls`` from a JSON file (class defaults fill any
    field the file doesn't contain). ``cls`` must be constructible with no args."""
    with open(path) as fp:
        data = json.load(fp)
    return apply_dict(cls(), data, warn=warn)


def update_dataclass_json(obj: T, path, warn: Optional[Callable[[str], None]] = None) -> T:
    """Like ``load_dataclass_json`` but updates an EXISTING instance in place, so
    every reference to ``obj`` sees the imported values."""
    with open(path) as fp:
        data = json.load(fp)
    return apply_dict(obj, data, warn=warn)
