# Trajectory-generation configs

Exported `TrajectoryGenModeConfig` snapshots — every knob the GUI's **Traj Gen**
panel exposes (RRT candidates, trajopt passes, clearances, dataset id, episode
count, …). Keeping one checked-in file per environment makes a generation run
reproducible from the env name alone.

## Naming convention

`configs/traj_configs/<env>.json`, where `<env>` is the environment's
`EnvConfig.name`:

| environment | `EnvConfig.name` | config file |
|---|---|---|
| planar 3-joint arm | `planar_3joint` | `planar_3joint.json` |
| small engine (curtain) | `upright_robot_small_engine_curtain` | `upright_robot_small_engine_curtain.json` |
| vine grape reach | `vine_grape_reach` | `vine_grape_reach.json` |

The path is produced by `splatsim.utils.paths.traj_config_path(env_name)`. The
GUI seeds its "Config File (JSON)" box from it for whichever env is running, so
**Export Config** writes to the right file by default instead of every env
overwriting one shared scratch config. The box stays editable — the convention
is a default, not a constraint.

Only `planar_3joint.json` exists so far. To add another env: launch it, tune the
panel, press **Export Config**.

## Files not named after an env

`planar_3joint_config.json` and `eval_planar_3joint_benchmark_config.json` are
named after the DATASET they generate rather than an environment (they predate
this folder). They are not auto-discovered — pass them explicitly:

```bash
python scripts/launch_nodes.py --robot <env> \
  --traj_config_file configs/traj_configs/eval_planar_3joint_benchmark_config.json
```

## Usage

Headless generation with a pinned config:

```bash
python scripts/launch_nodes.py --robot <env> --headless \
  --traj_config_file configs/traj_configs/planar_3joint.json
```

Parallel generation across N workers — the coordinator preloads the config into
its panel, then pushes it to every worker at Start (see
`scripts/trajgen_coordinator.py`):

```bash
python scripts/trajgen_coordinator.py --ports 6002-6004 \
  --traj-config-file configs/traj_configs/planar_3joint.json
```

## Schema drift

Loading is field-by-field and tolerant in both directions
(`splatsim/utils/config_io.py`): a field the file lacks keeps the class default,
and a field the class no longer has is ignored with a warning. So an older
config still loads after the schema changes — it just picks up defaults for
anything new. Re-export to refresh it.
