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


# ---------------------------------------------------------------------------
# Trajectory-generation config files
# ---------------------------------------------------------------------------
#
# One exported config per environment, named after `EnvConfig.name`, so a run
# is reproducible from the env alone: `configs/traj_configs/planar_3joint.json`,
# `.../vine_grape_reach.json`, `.../upright_robot_small_engine_curtain.json`.
#
# The GUI's Traj Gen panel seeds its "Config File (JSON)" box from
# `traj_config_path(env_name)`, so Export/Import default to the current env's
# file instead of one shared scratch file that whichever env ran last would
# overwrite. The box stays editable — nothing forces the convention.
#
# The directory also holds configs named after the DATASET they generate
# rather than an env (e.g. `eval_planar_3joint_benchmark_config.json`); those
# are passed explicitly via `--traj_config_file` and are not auto-discovered.

TRAJ_CONFIG_DIR: Path = SPLATSIM_ROOT / "configs" / "traj_configs"


def traj_config_path(env_name: str, absolute: bool = False) -> str:
    """Conventional config path for ``env_name`` (an ``EnvConfig.name``).

    Relative to the repo root by default — that is what the GUI displays and
    what reads well in a shell command. Pass ``absolute=True`` when the caller
    may run from a different working directory.
    """
    if absolute:
        return str(TRAJ_CONFIG_DIR / f"{env_name}.json")
    return f"configs/traj_configs/{env_name}.json"
