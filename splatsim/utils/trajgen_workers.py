"""Drive a pool of headless SplatSim servers generating trajectories in parallel.

Trajectory generation is single-threaded per server: one process plans one
episode at a time (RRT + trajopt on the CPU, then splat rendering on the GPU).
Running N server processes multiplies throughput by up to N, bounded by cores
for the planning phase and by VRAM for the rendering phase.

This module is the CLIENT half of that setup — pure ZMQ, no Tkinter and no
pybullet, so it can be driven from a GUI, a script, or a test. The SERVER half
is three dispatch methods in ``ZMQServerRobot.serve``
(``set_traj_config`` / ``set_serve_mode`` / ``get_trajgen_status``).

Layering, so each piece can be read on its own:

    TrajGenWorker      one server, one port. Typed wrappers over the three
                       dispatch methods, plus connect/close.
    TrajGenWorkerPool  N workers. Broadcasts a config (rewriting the dataset
                       repo id per worker), starts/stops them together, and
                       polls their progress.

Typical use::

    pool = TrajGenWorkerPool([6002, 6003, 6004])
    pool.connect()
    pool.push_config(cfg)          # cfg.lerobot_repo_id -> "<base>_worker1", ...
    pool.start_all()
    ...
    for s in pool.statuses():
        print(s.port, s.trajectory_count, "/", s.total)
    pool.stop_all()
    pool.close()

Every call is best-effort per worker: a dead or unreachable worker is recorded
in the returned result rather than raising, so one crashed process never takes
the whole pool down. Callers should surface those failures — a silently-missing
worker just stops contributing episodes and looks identical to a slow one.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Shared secret gating the workers' remote-control methods. Workers bind to
# 127.0.0.1 so they are unreachable from the network, but any LOCAL process
# could otherwise drive them; the token restricts that to this coordinator.
#
# Resolution order, first hit wins:
#   1. $SPLATSIM_CONTROL_TOKEN     — explicit override (CI, multi-user boxes)
#   2. ~/.cache/splatsim/control_token (0600)  — the normal path
#
# A FILE rather than a mandatory env export, because the workers and the
# coordinator are usually launched from different terminals and an env var
# would have to be exported in each; forgetting it in one silently produced a
# pool that rejected every push. The file is created once by
# `ensure_control_token()` (the coordinator does this at startup) and read
# lazily per request, so it works even when the workers started first.
#
# Never a CLI flag: /proc/<pid>/cmdline is world-readable, so a token in argv
# would be visible to exactly the local processes this excludes. The env var
# (/proc/<pid>/environ) and the 0600 file are both owner-only.
CONTROL_TOKEN_ENV = "SPLATSIM_CONTROL_TOKEN"
CONTROL_TOKEN_FILE_ENV = "SPLATSIM_CONTROL_TOKEN_FILE"


def control_token_file() -> "Path":
    from pathlib import Path

    override = os.environ.get(CONTROL_TOKEN_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "splatsim" / "control_token"


def control_token() -> str | None:
    """The shared secret, or None if neither source has one. Never creates."""
    env = os.environ.get(CONTROL_TOKEN_ENV)
    if env:
        return env
    path = control_token_file()
    try:
        token = path.read_text().strip()
    except OSError:
        return None
    return token or None


def ensure_control_token() -> str:
    """Return the shared secret, generating and persisting one if needed.

    Called by the coordinator so first-run needs no setup. Workers only ever
    READ (never create) — a worker that generated its own token would hand the
    pool a different secret per process.
    """
    import secrets
    import stat

    existing = control_token()
    if existing:
        return existing

    path = control_token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    # Create owner-only from the start — writing then chmod'ing would leave a
    # window where the secret is world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    return token

# Default request timeout. Generous because a worker in the middle of an
# episode services ZMQ on its dispatch thread while the main thread is busy
# planning; a loaded box can take a while to get around to the reply.
DEFAULT_TIMEOUT_MS = 5000

# Seconds between consecutive worker starts. See TrajGenWorkerPool.start_all
# for why: an episode alternates a CPU-bound planning phase with a GPU-bound
# rendering phase, and workers launched in lockstep collide on the GPU.
DEFAULT_START_STAGGER_S = 0.5


def worker_repo_id(base_repo_id: str, index: int) -> str:
    """Dataset repo id for worker ``index`` (1-based): ``<base>_worker<N>``.

    Every worker MUST write its own dataset — two servers appending to one
    LeRobot repo interleave episode indices and corrupt it. Shards are merged
    afterwards with lerobot's dataset editor::

        lerobot-edit-dataset --operation.type merge \\
          --operation.repo_ids '["me/data_worker1", "me/data_worker2"]' \\
          --new_repo_id me/data

    That merge preserves arbitrary extra ``meta/episodes`` columns (it copies
    each shard's episodes parquet wholesale and only re-bases the index
    columns), so SplatSim's per-episode config payload survives it.
    """
    return f"{base_repo_id}_worker{index}"


def dataset_root(repo_id: str, root: str | None = None) -> "Path":
    """Local directory for ``repo_id`` in the flat write-mode layout.

    Mirrors ``lerobot_utils.load_lerobot_dataset``: SplatSim always writes to
    ``$HF_LEROBOT_HOME/{repo_id}``, never the revision-safe Hub snapshot cache.
    """
    from pathlib import Path

    if root is not None:
        return Path(root)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def count_episodes(repo_id: str, root: str | None = None) -> int:
    """Episodes already on disk for ``repo_id``; 0 if it does not exist yet.

    Reads ``meta/info.json`` directly rather than constructing a
    ``LeRobotDataset``: this runs for the base dataset plus every shard on each
    Start, and opening a dataset is far heavier than reading one small JSON. A
    partially-initialized directory (crash before the first episode was saved)
    reads as 0, which is the right answer for planning.
    """
    import json

    info = dataset_root(repo_id, root) / "meta" / "info.json"
    if not info.exists():
        return 0
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except (ValueError, OSError):
        return 0


@dataclass
class ShardTarget:
    """One worker's slice of the remaining work."""

    index: int              # 1-based, matches the _workerN suffix
    port: int
    repo_id: str
    existing: int           # episodes already in THIS shard
    to_generate: int        # episodes this run should add
    absolute_target: int    # existing + to_generate
    active: bool = True     # False: excluded from the split (dead worker)

    def describe(self) -> str:
        line = (
            f"worker{self.index}@{self.port} -> {self.repo_id}: "
            f"{self.existing} on disk + {self.to_generate} new = {self.absolute_target}"
        )
        if not self.active:
            line += "  [EXCLUDED — its episodes still count, but it gets no new work]"
        return line


@dataclass
class ShardPlan:
    """How ``target_total`` episodes get split across workers, accounting for
    everything already generated.

    The counting is what makes resume work. Episodes already on disk live in
    two places after a previous run:

      * the BASE dataset (e.g. 100 episodes from a single-process run), and
      * each per-worker SHARD from earlier parallel runs.

    Both count toward the target, so ``remaining = target_total - base_existing
    - sum(shard existing)``. That remainder is split evenly across the workers,
    and each worker is handed an ABSOLUTE target for its own shard
    (``existing + share``) because the server resumes ``trajectory_count`` from
    its shard's episode count on mode entry and stops when
    ``trajectory_count >= num_base_trajectories``.

    Concretely, for base=100, target=200, 5 workers, empty shards: each worker
    gets 20, and the merged result is 100 + 20*5 = 200.
    """

    base_repo_id: str
    base_existing: int
    target_total: int
    shards: list[ShardTarget] = dataclasses.field(default_factory=list)

    @property
    def shard_existing(self) -> int:
        return sum(s.existing for s in self.shards)

    @property
    def already_have(self) -> int:
        return self.base_existing + self.shard_existing

    @property
    def remaining(self) -> int:
        """Episodes still to generate. Never negative — an over-full dataset
        means there is simply nothing to do."""
        return max(0, self.target_total - self.already_have)

    @property
    def working_ports(self) -> list[int]:
        """Ports with a nonzero share. Workers with none would immediately
        report complete and bounce back to idle; skip starting them."""
        return [s.port for s in self.shards if s.to_generate > 0]

    def merge_repo_ids(self) -> list[str]:
        """Inputs for the final merge: the base dataset (when it has episodes)
        followed by every shard that ends up non-empty, in worker order."""
        ids = [self.base_repo_id] if self.base_existing > 0 else []
        ids += [s.repo_id for s in self.shards if s.absolute_target > 0]
        return ids

    def merge_command(self, new_repo_id: str) -> str:
        """The lerobot-edit-dataset invocation that reassembles the shards.

        ``new_repo_id`` is required and must NOT be one of the inputs — a merge
        written over one of its own inputs is unsafe. Consolidating back into
        the base (the normal end state, so the next run's resume sees the full
        dataset) therefore needs a staging id and a swap, which is what
        ``launch_trajgen_pool.consolidate`` does; this helper is for the
        standalone case where you are driving the merge yourself.
        """
        import json as _json

        ids = _json.dumps(self.merge_repo_ids())
        return (
            "lerobot-edit-dataset --operation.type merge \\\n"
            f"  --operation.repo_ids '{ids}' \\\n"
            f"  --new_repo_id {new_repo_id}"
        )

    def describe(self) -> str:
        lines = [
            f"target {self.target_total} episodes; "
            f"{self.base_existing} in base '{self.base_repo_id}' + "
            f"{self.shard_existing} in shards = {self.already_have} already; "
            f"{self.remaining} to generate across {len(self.shards)} worker(s)"
        ]
        lines += [f"  {s.describe()}" for s in self.shards]
        return "\n".join(lines)


def plan_shards(
    base_repo_id: str,
    target_total: int,
    ports: Sequence[int],
    root: str | None = None,
    active_ports: Sequence[int] | None = None,
) -> ShardPlan:
    """Split ``target_total`` across ``ports``, counting what is already on disk.

    The remainder of an uneven split goes to the lowest-numbered workers, so
    5 workers over 102 remaining episodes get 21, 21, 20, 20, 20.

    ``active_ports`` (default: all of ``ports``) limits WHO gets a share of
    the remaining work — used to route around a crashed worker. Crucially,
    ``ports`` must stay the FULL pool even then: a shard's repo id is derived
    from its position in ``ports``, so dropping the dead port from the list
    would silently remap every later worker onto the wrong shard. Excluded
    shards keep their on-disk episodes counted; they just get
    ``to_generate=0``.
    """
    base_existing = count_episodes(base_repo_id, root)
    shard_ids = [worker_repo_id(base_repo_id, i + 1) for i in range(len(ports))]
    # A shard whose root was explicitly overridden can't share one root path;
    # per-shard roots aren't supported, so only the default layout is counted.
    shard_existing = [count_episodes(rid) if root is None else 0 for rid in shard_ids]

    active = set(int(p) for p in (ports if active_ports is None else active_ports))
    remaining = max(0, int(target_total) - base_existing - sum(shard_existing))
    n = sum(1 for p in ports if int(p) in active)
    base_share, extra = (remaining // n, remaining % n) if n else (0, 0)

    shards = []
    nth_active = 0
    for i, port in enumerate(ports):
        if int(port) in active:
            share = base_share + (1 if nth_active < extra else 0)
            nth_active += 1
        else:
            share = 0
        shards.append(
            ShardTarget(
                index=i + 1,
                port=int(port),
                repo_id=shard_ids[i],
                existing=shard_existing[i],
                to_generate=share,
                absolute_target=shard_existing[i] + share,
                active=int(port) in active,
            )
        )
    return ShardPlan(
        base_repo_id=base_repo_id,
        base_existing=base_existing,
        target_total=int(target_total),
        shards=shards,
    )


@dataclass
class WorkerStatus:
    """One worker's progress, as returned by ``get_trajgen_status``."""

    port: int
    reachable: bool
    mode: str | None = None
    trajectory_count: int = 0
    total: int = 0
    repo_id: str | None = None
    env_name: str | None = None
    error: str | None = None

    def describe(self) -> str:
        """One-line human summary, for a GUI status line or a log."""
        if not self.reachable:
            return f"worker@{self.port}: UNREACHABLE ({self.error})"
        return (
            f"worker@{self.port}: {self.trajectory_count}/{self.total} [{self.mode}]"
        )


@dataclass
class BroadcastResult:
    """Outcome of one fan-out call, per worker port."""

    ok: list[int] = dataclasses.field(default_factory=list)
    failed: dict[int, str] = dataclasses.field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        if self.all_ok:
            return f"{len(self.ok)} worker(s) OK"
        return (
            f"{len(self.ok)} OK, {len(self.failed)} FAILED: "
            + ", ".join(f"{p} ({e})" for p, e in sorted(self.failed.items()))
        )


class TrajGenWorker:
    """One headless SplatSim server, addressed over ZMQ.

    Thin and synchronous: each method is one REQ/REP round trip. The socket is
    recreated on timeout because a ZMQ REQ socket that times out mid-exchange
    is left in a broken state and would desync every later request.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        token: str | None = None,
    ):
        self.port = int(port)
        self.host = host
        self.timeout_ms = int(timeout_ms)
        self.token = token if token is not None else control_token()
        self._context = None
        self._socket = None

    # -- connection -----------------------------------------------------

    def connect(self) -> None:
        import zmq

        self.close()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        # Drop queued messages instantly on close instead of blocking the
        # caller (default LINGER waits forever for an unreachable peer).
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

    def _request(self, method: str, args: dict | None = None,
                 timeout_ms: int | None = None) -> Any:
        """One REQ/REP round trip. Raises on transport failure.

        ``timeout_ms`` overrides the receive deadline for this call only —
        rendering a splat scene takes far longer than a status poll.
        """
        import pickle

        if self._socket is None:
            self.connect()
        if timeout_ms is not None:
            import zmq

            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        # Every control request carries the shared secret; the worker
        # default-denies when it has none configured.
        payload = dict(args or {})
        payload["token"] = self.token
        try:
            self._socket.send(pickle.dumps({"method": method, "args": payload}))
            reply = pickle.loads(self._socket.recv())
            if timeout_ms is not None:
                import zmq

                self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            return reply
        except Exception:
            # A timed-out REQ socket can't be reused — the next send() would
            # raise EFSM. Rebuild so a transient failure doesn't poison the
            # worker for the rest of the session.
            self.connect()
            raise

    # -- operations -----------------------------------------------------

    def set_config(self, config_dict: dict) -> Any:
        """Push a trajectory-gen config.

        Accepted in any mode except actively-generating — including
        INTERACTIVE, which is what a freshly-launched worker sits in.
        """
        return self._request("set_traj_config", {"config": config_dict})

    def set_serve_mode(self, mode: str) -> Any:
        return self._request("set_serve_mode", {"mode": mode})

    def start(self) -> Any:
        return self.set_serve_mode("generate_trajectories")

    def stop(self) -> Any:
        return self.set_serve_mode("generate_trajectories_idle")

    def observations(self, timeout_ms: int = 120_000) -> Any:
        """Fetch one observation dict (rendered images included).

        NOT token-gated — this is the pre-existing ``get_observations``
        dispatch the gym/eval clients use, not one of the control methods.

        Slow by default (a splat render on a cold scene), hence the generous
        timeout. Never call this on a worker that is actively generating: the
        render would run on the ZMQ thread while the main thread drives its own
        rendering, and the GL context belongs to the main thread.
        """
        return self._request("get_observations", {"render_images": True},
                             timeout_ms=timeout_ms)

    def last_frames(self, timeout_ms: int | None = None) -> Any:
        """Frames the worker ALREADY rendered — no new rendering triggered.

        Safe to poll while the worker is generating: it reads a cache the
        generation loop fills for free. `observations()` by contrast forces a
        fresh render, which must not happen mid-run.
        """
        return self._request("get_last_frames", timeout_ms=timeout_ms)

    def status(self) -> WorkerStatus:
        """Never raises — an unreachable worker comes back ``reachable=False``."""
        try:
            raw = self._request("get_trajgen_status")
        except Exception as e:
            return WorkerStatus(port=self.port, reachable=False, error=f"{type(e).__name__}: {e}")
        if isinstance(raw, dict) and raw.get("error"):
            return WorkerStatus(port=self.port, reachable=True, error=str(raw["error"]))
        return WorkerStatus(
            port=self.port,
            reachable=True,
            mode=raw.get("mode"),
            trajectory_count=int(raw.get("trajectory_count", 0)),
            total=int(raw.get("total", 0)),
            repo_id=raw.get("lerobot_repo_id"),
            env_name=raw.get("env_name"),
        )


class TrajGenWorkerPool:
    """A set of ``TrajGenWorker``s driven as one unit."""

    def __init__(
        self,
        ports: Iterable[int],
        host: str = "127.0.0.1",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        token: str | None = None,
    ):
        self.token = token if token is not None else control_token()
        self.workers = [
            TrajGenWorker(p, host=host, timeout_ms=timeout_ms, token=self.token)
            for p in ports
        ]

    def __len__(self) -> int:
        return len(self.workers)

    @property
    def ports(self) -> list[int]:
        return [w.port for w in self.workers]

    def connect(self) -> None:
        for w in self.workers:
            w.connect()

    def close(self) -> None:
        for w in self.workers:
            w.close()

    def _broadcast(self, fn, label: str) -> BroadcastResult:
        """Apply ``fn(worker, index)`` to every worker, collecting failures."""
        result = BroadcastResult()
        for i, w in enumerate(self.workers):
            try:
                reply = fn(w, i)
                if isinstance(reply, dict) and reply.get("error"):
                    result.failed[w.port] = str(reply["error"])
                else:
                    result.ok.append(w.port)
            except Exception as e:
                result.failed[w.port] = f"{type(e).__name__}: {e}"
        return result

    def plan_work(
        self,
        config,
        base_repo_id: str | None = None,
        root: str | None = None,
        active_ports: Sequence[int] | None = None,
    ) -> ShardPlan:
        """Work out each worker's share, counting episodes already on disk.

        ``config.num_base_trajectories`` is read as the TOTAL you want to end
        up with, not the number to add — so re-pressing Start with the target
        raised from 100 to 200 generates 100 more, split across the pool,
        rather than 200 more.

        ``active_ports`` restricts who gets NEW work (see ``plan_shards``) —
        the coordinator uses it to reassign a crashed worker's remaining share
        to the survivors.
        """
        base = base_repo_id or getattr(config, "lerobot_repo_id", None)
        if not base:
            raise ValueError(
                "No lerobot_repo_id in the config and no base_repo_id given — "
                "workers would have nowhere to write."
            )
        return plan_shards(
            base, int(config.num_base_trajectories), self.ports, root=root,
            active_ports=active_ports,
        )

    def push_config(
        self,
        config,
        plan: ShardPlan | None = None,
        base_repo_id: str | None = None,
        ports: Sequence[int] | None = None,
    ) -> BroadcastResult:
        """Broadcast ``config`` to every worker, one dataset shard each.

        ``config`` is a ``TrajectoryGenModeConfig`` (or any dataclass config).
        It is serialized ONCE via ``config_io.dataclass_to_dict`` — the same
        encoding the GUI's "Export Config" button writes, so non-serializable
        fields (e.g. ``cuboids_fn``) are dropped identically and each worker
        keeps its own default for them.

        Three fields are then rewritten PER WORKER from ``plan`` (computed via
        ``plan_work`` when not supplied):

          * ``lerobot_repo_id`` -> ``<base>_worker<i>``. Without this every
            worker would append to the same LeRobot dataset and corrupt it.
          * ``num_base_trajectories`` -> that shard's ABSOLUTE target
            (episodes already in the shard + its share of the remaining work).
            Absolute, not a delta, because the server resumes
            ``trajectory_count`` from its shard on mode entry and stops at
            ``trajectory_count >= num_base_trajectories``.
          * ``push_to_hub`` -> False, whatever the panel says. Shards are
            transient — merged into the base and deleted when the pool stops —
            so a worker uploading its shard to the HF hub just publishes
            ``<base>_workerN`` repos that are about to be deleted locally.
            The panel's Push-to-Hub applies to the MERGED dataset instead
            (``launch_trajgen_pool.py`` pushes it after consolidating).

        NOT rewritten here: RNG seeds. If the env's scene randomization is
        seeded deterministically, every worker will generate the SAME episodes
        and you get N copies rather than N times the data. Check how your env
        seeds itself before trusting the shard count.

        ``ports`` limits the push to those workers (e.g. the reachable ones
        when re-pushing around a dead worker — pushing to it would just burn a
        timeout). Skipped workers are reported as ok.
        """
        from splatsim.utils.config_io import dataclass_to_dict

        if plan is None:
            plan = self.plan_work(config, base_repo_id=base_repo_id)
        encoded = dataclass_to_dict(config)
        wanted = None if ports is None else set(int(p) for p in ports)

        def _push(worker: TrajGenWorker, index: int):
            if wanted is not None and worker.port not in wanted:
                return {"ok": True, "skipped": True}
            # Deep-copy so each worker's payload is independent; the encoded
            # dict holds nested lists/dicts that would otherwise be shared.
            payload = copy.deepcopy(encoded)
            shard = plan.shards[index]
            payload["lerobot_repo_id"] = shard.repo_id
            payload["num_base_trajectories"] = shard.absolute_target
            payload["push_to_hub"] = False  # shards are transient; see docstring
            return worker.set_config(payload)

        return self._broadcast(_push, "set_traj_config")

    def start_all(
        self,
        ports: Sequence[int] | None = None,
        stagger_s: float = DEFAULT_START_STAGGER_S,
    ) -> BroadcastResult:
        """Start generation. ``ports`` limits it to a subset (e.g.
        ``plan.working_ports``, skipping workers with a zero share, which would
        otherwise report complete immediately and bounce back to idle).

        ``stagger_s`` pauses between consecutive starts. An episode is two very
        differently-shaped phases — CPU-bound planning (RRT, trajopt) then
        GPU-bound splat rendering — and workers started in the same instant
        march through them in lockstep, so every worker renders at once and
        queues on the one GPU while all 24 cores idle, then all plan at once
        while the GPU idles. Offsetting the starts decorrelates the phases.

        This is an initial offset, not a scheduler: episodes vary in length so
        the phases drift on their own after a while, and a 0.5 s stagger is
        small next to a multi-second episode. It costs nothing and removes the
        worst case (a synchronized render storm on the first episode); it does
        not guarantee they stay out of phase.
        """
        import time

        wanted = None if ports is None else set(int(p) for p in ports)
        first = [True]

        def _start(worker: TrajGenWorker, index: int):
            if wanted is not None and worker.port not in wanted:
                return {"ok": True, "skipped": True}
            if not first[0] and stagger_s > 0:
                time.sleep(stagger_s)
            first[0] = False
            return worker.start()

        return self._broadcast(_start, "start")

    def stop_all(self) -> BroadcastResult:
        return self._broadcast(lambda w, i: w.stop(), "stop")

    def statuses(self) -> list[WorkerStatus]:
        return [w.status() for w in self.workers]

    def preview_frames(
        self,
        port: int | None = None,
        timeout_ms: int = 120_000,
        live: bool = False,
    ) -> tuple[dict, str | None]:
        """One worker's camera images, ready for ``gui.update_camera_images``.

        Returns ``(frames, note)`` — ``frames`` maps camera name to an
        (H, W, 3) uint8 array, and ``note`` is a human-readable explanation
        when it comes back empty (no worker reachable, rendering disabled,
        the request failed). Empty-with-no-explanation would read as "the
        scene is black", which is exactly the wrong impression for a
        does-this-look-right check.

        Only ONE worker is asked (the first reachable, or ``port``): every
        worker builds the same scene, and rendering is expensive.

        ``live=True`` fetches the frames the worker last rendered for its
        dataset instead of forcing a new render — the mode to use WHILE it is
        generating, since it costs no GPU time and cannot collide with the
        main thread's render. ``live=False`` (default) triggers a fresh render,
        which is what you want for a pre-flight look at an idle scene.
        """
        targets = [w for w in self.workers if port is None or w.port == int(port)]
        if not targets:
            return {}, f"no worker on port {port}"

        last_error = None
        for worker in targets:
            try:
                obs = (
                    worker.last_frames() if live
                    else worker.observations(timeout_ms=timeout_ms)
                )
            except Exception as e:
                last_error = f"worker@{worker.port}: {type(e).__name__}: {e}"
                continue
            if isinstance(obs, dict) and obs.get("error"):
                last_error = f"worker@{worker.port}: {obs['error']}"
                continue
            frames = _extract_frames(obs)
            if frames:
                return frames, None
            last_error = (
                f"worker@{worker.port} has not rendered a frame yet (it is "
                "probably planning rather than rendering)" if live else
                f"worker@{worker.port} returned no images — its render mode is "
                "probably NONE (--no_camera_rendering)"
            )
        return {}, last_error

    def detect_env_name(self, statuses: Sequence[WorkerStatus] | None = None) -> str | None:
        """The env every reachable worker reports, or None if they disagree.

        Disagreement means the pool is pointed at a mixed set of simulators —
        their datasets would not merge (different features), so the caller
        should treat None as "don't assume", not "no workers".
        """
        sts = list(statuses) if statuses is not None else self.statuses()
        names = {s.env_name for s in sts if s.reachable and s.env_name}
        return names.pop() if len(names) == 1 else None

    def summarize(
        self,
        statuses: Sequence[WorkerStatus] | None = None,
        plan: ShardPlan | None = None,
    ) -> str:
        """One-line progress, reported against the TARGET YOU ASKED FOR.

        With a ``plan``, progress is counted from DISK (base dataset + every
        shard) against ``plan.target_total``. Disk is the honest source: it is
        what the merge will actually produce, and it is correct in every mode.

        Do NOT sum the workers' own ``total`` fields. Each worker reports its
        own ``num_base_trajectories``, which before a config push is just the
        class default (100) — so two idle workers summed to "0/200" while the
        panel said 100, implying each had been handed the full target when in
        fact none had been handed anything yet. After a push the sum is also
        wrong, in a subtler way: shard targets exclude episodes already in the
        base dataset, so it would under-report the goal.

        Without a plan this falls back to the raw worker-reported sum, clearly
        labelled as such.
        """
        sts = list(statuses) if statuses is not None else self.statuses()
        dead = [s.port for s in sts if not s.reachable]
        live = len(sts) - len(dead)

        if plan is None:
            done = sum(s.trajectory_count for s in sts if s.reachable)
            total = sum(s.total for s in sts if s.reachable)
            line = f"workers report {done}/{total} across {live} live"
        else:
            line = (
                f"{plan.already_have}/{plan.target_total} episodes on disk "
                f"({plan.remaining} to go) across {live} live worker(s)"
            )
        if dead:
            line += f" — UNREACHABLE: {dead}"
        return line

    def per_worker_line(
        self,
        statuses: Sequence[WorkerStatus] | None = None,
    ) -> str:
        """Compact per-shard breakdown, e.g. ``w1 12/50 · w2 11/50``.

        Uses each worker's LIVE counter (which the server resumes from that
        shard's episode count) against the target the worker ITSELF reports —
        i.e. the ``num_base_trajectories`` actually pushed to it. Deliberately
        NOT a freshly recomputed plan's split: that re-divides the remaining
        work evenly on every poll, so under uneven progress a slow worker's
        displayed target visibly shrank (and fast workers' grew) while every
        real target was unchanged.
        """
        sts = list(statuses) if statuses is not None else self.statuses()
        by_port = {s.port: s for s in sts}
        parts = []
        for i, worker in enumerate(self.workers):
            st = by_port.get(worker.port)
            if st is None or not st.reachable:
                parts.append(f"w{i + 1} down")
                continue
            parts.append(f"w{i + 1} {st.trajectory_count}/{st.total}")
        return " · ".join(parts)


def _extract_frames(obs) -> dict:
    """Pull displayable RGB frames out of an observation dict.

    The coordinator does not know the worker's ``camera_names`` or
    ``image_resize_modes``, so instead of reconstructing the
    ``"{camera}_{mode}"`` keys it scans for image-shaped values and strips a
    trailing resize-mode suffix. When a camera was rendered in several modes
    only the first is kept — they are the same view, and showing both would
    just halve the preview size.

    Normalization matches the in-sim GUI's own feed
    (``_update_gui_camera_images``): torch tensor -> numpy, CHW -> HWC,
    float [0,1] -> uint8.
    """
    import numpy as np

    if not isinstance(obs, dict):
        return {}
    try:
        from splatsim.configs.mode_config import ImageResizeMode

        suffixes = [f"_{m.value}" for m in ImageResizeMode]
    except Exception:
        suffixes = []

    frames: dict = {}
    for key, value in obs.items():
        if value is None:
            continue
        frame = value
        if hasattr(frame, "detach"):  # torch tensor
            frame = frame.detach().cpu().numpy()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            continue
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] not in (3, 4):
            continue
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        name = key
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        frames.setdefault(name, frame[:, :, :3])
    return frames


def parse_ports(spec: str) -> list[int]:
    """Parse a worker-port spec: ``"6002,6003"`` or a range ``"6002-6005"``.

    Mixed forms are allowed (``"6002,6010-6012"``). Empty string -> no workers,
    which is the "run generation only in this process" case.
    """
    ports: list[int] = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ports.extend(range(int(lo), int(hi) + 1))
        else:
            ports.append(int(chunk))
    return ports
