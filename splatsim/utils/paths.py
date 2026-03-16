"""Path utilities shared across SplatSim modules."""

from pathlib import Path

SPLATSIM_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def resolve_splatsim_path(path: str) -> str:
    """Resolve a path, making relative paths relative to SPLATSIM_ROOT.

    This allows configs to use relative paths like './splatsim/...' that work
    regardless of the current working directory.
    """
    if Path(path).is_absolute():
        return path
    return str(SPLATSIM_ROOT / path)
