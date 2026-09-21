"""SplatSim as a LeRobot environment plugin.

LeRobot's `lerobot-train` / `lerobot-eval` import every installed distribution whose name starts
with `lerobot_env_` (see `lerobot.utils.import_utils.register_third_party_plugins`), so installing
this package is all it takes for `--env.type=splatsim` to resolve:

    pip install -e SplatSim/lerobot_env_splatsim      (install.sh does this next to the lerobot install)

    lerobot-eval --env.type=splatsim --env.task=planar_3joint --env.robot_name=planar_3joint \
                 --env.external_port=6023 --policy.path=... --eval.n_episodes=10

What it registers:
    config.SplatSimEnv            the `splatsim` EnvConfig (ZMQ client to a running node, or an in-process server)
    robot.SplatSimLerobotConfig   the `splatsim_lerobot` RobotConfig (a SplatSim node as a lerobot Robot)
Also here, because the env uses them:
    recording                     TeleopRecordingContext / TeleopRecordingWrapper (record rollouts to a LeRobot dataset)
    seeding                       seed a gym env at a dataset frame's state (data relabelling, visualisation)

Dependencies point one way: this package imports `splatsim` and `lerobot`; neither imports it.
"""

from . import config as config  # noqa: F401  (registers "splatsim")
from . import robot as robot  # noqa: F401  (registers "splatsim_lerobot")
from .config import SplatSimEnv
from .recording import TeleopRecordingContext, TeleopRecordingWrapper

__all__ = ["SplatSimEnv", "TeleopRecordingContext", "TeleopRecordingWrapper"]
__version__ = "0.1.0"
