"""One command to bring up parallel trajectory generation: workers + panel.

Replaces the manual sequence (export a token, launch N headless servers on N
ports, wait for them, start the coordinator) with a single invocation that also
tears everything down on exit.

    python scripts/launch_trajgen_pool.py \\
        --robot sim_ur_pybullet_small_engine_new_interactive \\
        --workers 4 --base-repo-id JennyWWW/small_engine

That launches 4 headless workers on ports 6002-6005, waits until each is
answering, then opens the Trajectory Gen panel wired to all of them. Set your
episode target in the panel and press Start; each worker generates its own
shard and the coordinator prints the merge command.

The run is UNATTENDED by default: press Start in the panel and it goes all the
way through on its own — the panel closes once the episode target is reached,
the workers are stopped, and their shards are merged back into the base dataset
and removed. You are left with one dataset under the name you asked for, ready
for the next run to resume from. Pass --keep-panel-open to stay in control, or
--no-merge / --keep-shards to skip the cleanup.

Ctrl-C (or closing the panel) stops the coordinator and terminates every worker
it started — including on a crash, so you never leak headless simulators
holding GPU memory. Termination is graceful: each worker gets SIGTERM, aborts
its in-flight episode, and finalizes its LeRobot shard (parquet footers are
only written at finalize, so this step is what keeps the shard readable); the
pool waits up to --stop-grace seconds per run before SIGKILLing stragglers.
Everything the workers saved up to that point is then merged into the base
dataset as usual — a partial run is still consolidated. Should a shard be
unreadable anyway (worker SIGKILLed or crashed hard), it is quarantined to
'<shard>__corrupt' and the merge proceeds with the healthy ones.

Worker logs go to per-port files under --log-dir (default logs/trajgen/) rather
than being interleaved on your terminal; the path of each is printed at launch.

Useful flags:
    --workers N          how many simulators (VRAM is the limit, not cores)
    --base-port P        first port; workers take P .. P+N-1  (default 6002)
    --dry-run            print the commands and exit, launch nothing
    --extra-worker-arg   repeatable passthrough to launch_nodes.py, e.g.
                         --extra-worker-arg --no_camera_rendering

Anything this script does not wrap can still be done by hand — it only
sequences `scripts/launch_nodes.py` and `scripts/trajgen_coordinator.py`.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from splatsim.utils.paths import SPLATSIM_ROOT
from splatsim.utils.trajgen_workers import (
    count_episodes,
    dataset_root,
    ensure_control_token,
    worker_repo_id,
)

DEFAULT_BASE_PORT = 6002


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--robot", required=True,
        help="launch_nodes.py --robot value, e.g. "
             "sim_ur_pybullet_small_engine_new_interactive",
    )
    ap.add_argument(
        "--workers", type=int, default=2,
        help="Number of headless simulators (default 2). Each loads its own "
             "copy of the scene splat into its own CUDA context, so VRAM — not "
             "core count — is the binding limit. Start low and watch nvidia-smi.",
    )
    ap.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT,
                    help=f"First worker port (default {DEFAULT_BASE_PORT}).")
    ap.add_argument("--base-repo-id", default=None,
                    help="Base LeRobot dataset id; worker i writes '<base>_worker<i>'.")
    ap.add_argument("--traj-config-file", default=None,
                    help="Optional config JSON to preload into the panel "
                         "(conventionally configs/traj_configs/<env>.json).")
    ap.add_argument("--merged-repo-id", default=None,
                    help="Repo id for the merge command the coordinator prints.")
    ap.add_argument("--log-dir", default="logs/trajgen",
                    help="Where per-worker logs go (default logs/trajgen).")
    ap.add_argument("--startup-timeout", type=float, default=600.0,
                    help="Seconds to wait for each worker's port to open "
                         "(default 600 — loading splats is slow).")
    ap.add_argument("--extra-worker-arg", action="append", default=[],
                    metavar="ARG",
                    help="Extra argument passed through to launch_nodes.py. "
                         "Repeat for each token, e.g. --extra-worker-arg "
                         "--no_camera_rendering")
    ap.add_argument("--launch-stagger", type=float, default=2.0, metavar="SECONDS",
                    help="Delay between SPAWNING consecutive workers (default "
                         "2.0). Four processes allocating splat VRAM and "
                         "hammering the same PLY files at once is the slowest "
                         "way to start; this is separate from the coordinator's "
                         "--start-stagger, which offsets GENERATION.")
    ap.add_argument("--start-stagger", type=float, default=None, metavar="SECONDS",
                    help="Passed to the coordinator: delay between starting "
                         "generation on consecutive workers, so their CPU-bound "
                         "planning and GPU-bound rendering phases do not run in "
                         "lockstep. Default is the coordinator's own.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run, then exit.")
    ap.add_argument("--no-merge", action="store_true",
                    help="Do not merge when the workers stop; leave the "
                         "per-worker shards on disk to merge by hand.")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Merge, but do NOT delete the per-worker shard "
                         "datasets afterwards.")
    ap.add_argument("--keep-panel-open", action="store_true",
                    help="Leave the panel open after the episode target is "
                         "reached instead of closing it and merging. By "
                         "default the run is unattended: press Start and it "
                         "goes all the way to one consolidated dataset.")
    ap.add_argument("--confirm", action="store_true",
                    help="Ask before merging instead of doing it automatically "
                         "when the workers stop.")
    ap.add_argument("--stop-grace", type=float, default=120.0, metavar="SECONDS",
                    help="How long to wait after SIGTERM for each worker to "
                         "finalize its dataset before SIGKILLing it (default "
                         "120). Killing early corrupts the shard's parquet "
                         "files, so keep this generous.")
    return ap


def worker_command(args, port: int) -> list[str]:
    cmd = [
        sys.executable, "scripts/launch_nodes.py",
        "--robot", args.robot,
        "--headless",
        "--robot_port", str(port),
    ]
    cmd += list(args.extra_worker_arg)
    return cmd


def coordinator_command(args, ports: list[int], dump_config_path: Path) -> list[str]:
    cmd = [
        sys.executable, "scripts/trajgen_coordinator.py",
        "--ports", f"{ports[0]}-{ports[-1]}" if len(ports) > 1 else str(ports[0]),
        # The panel's effective config, written on coordinator exit. It is how
        # panel settings reach the post-run steps in THIS process — e.g.
        # push_to_hub, which applies to the merged dataset, not the shards.
        "--dump-config", str(dump_config_path),
    ]
    if args.base_repo_id:
        cmd += ["--base-repo-id", args.base_repo_id]
    if args.traj_config_file:
        cmd += ["--traj-config-file", args.traj_config_file]
    if args.merged_repo_id:
        cmd += ["--merged-repo-id", args.merged_repo_id]
    if args.start_stagger is not None:
        cmd += ["--start-stagger", str(args.start_stagger)]
    if not args.keep_panel_open:
        cmd += ["--exit-when-done"]
    return cmd


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def wait_for_workers(procs: dict[int, subprocess.Popen], timeout: float) -> list[int]:
    """Block until every worker's port answers. Returns the ports that came up.

    A worker that exits during startup is reported immediately rather than
    waited on for the full timeout — that is the common case when the scene
    assets or the --robot name are wrong, and the log path is the useful thing
    to show.
    """
    deadline = time.time() + timeout
    pending = set(procs)
    ready: list[int] = []
    while pending and time.time() < deadline:
        for port in sorted(pending):
            proc = procs[port]
            if proc.poll() is not None:
                print(f"[pool] worker on {port} EXITED during startup "
                      f"(code {proc.returncode}) — see its log")
                pending.discard(port)
                break
            if port_is_open(port):
                print(f"[pool] worker on {port} is up")
                ready.append(port)
                pending.discard(port)
                break
        else:
            time.sleep(1.0)
    for port in sorted(pending):
        print(f"[pool] worker on {port} did not open its port within "
              f"{timeout:.0f}s — continuing without it")
    return sorted(ready)


def terminate(procs: dict[int, subprocess.Popen], grace_s: float = 120.0) -> None:
    """SIGTERM every worker, wait for it to finalize its dataset, then SIGKILL.

    Runs from a finally block so a crash in the coordinator cannot leave
    headless simulators holding GPU memory.

    The grace period matters: on SIGTERM a worker aborts its current episode
    and finalizes its LeRobot shard (flush pending videos, close parquet
    writers — the parquet FOOTERS are only written here). SIGKILLing before
    that finishes leaves the shard unreadable and its episodes lost, so wait
    generously. Ctrl-C during the wait skips straight to SIGKILL.
    """
    alive = {p: proc for p, proc in procs.items() if proc.poll() is None}
    for port, proc in alive.items():
        print(f"[pool] stopping worker on {port} (pid {proc.pid})")
        proc.terminate()
    if alive:
        print(f"[pool] waiting up to {grace_s:.0f}s for worker(s) to finalize "
              f"their datasets (Ctrl-C to kill immediately)...")
    deadline = time.time() + grace_s
    try:
        for port, proc in alive.items():
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(f"[pool] worker on {port} exited cleanly")
            except subprocess.TimeoutExpired:
                print(f"[pool] pid {proc.pid} still alive after {grace_s:.0f}s — "
                      f"killing (its shard may be corrupt)")
                proc.kill()
    except KeyboardInterrupt:
        print("\n[pool] interrupted again — SIGKILLing remaining workers "
              "(their shards may be corrupt)")
        for proc in alive.values():
            if proc.poll() is None:
                proc.kill()


def discover_shards(base_repo_id: str, n_workers: int) -> tuple[list[str], list[str]]:
    """Split the pool's datasets into (merge inputs, deletable shards).

    Only datasets that actually have episodes are returned: a shard that never
    produced one has no directory to merge or remove. The BASE dataset is a
    merge input when it already holds episodes, but is never deletable — it is
    the thing everything gets consolidated into.
    """
    shards = [
        worker_repo_id(base_repo_id, i + 1)
        for i in range(n_workers)
        if count_episodes(worker_repo_id(base_repo_id, i + 1)) > 0
    ]
    inputs = ([base_repo_id] if count_episodes(base_repo_id) > 0 else []) + shards
    return inputs, shards


def dataset_is_readable(repo_id: str) -> bool:
    """True when every parquet file in the dataset has a valid footer.

    A worker that died without finalizing its LeRobot writer leaves parquet
    files without footers ("Parquet magic bytes not found"), and one such
    shard fails the WHOLE merge. Checking footers up front (pyarrow reads just
    the footer — cheap) lets the merge proceed with the healthy shards.
    """
    import pyarrow.parquet as pq

    for path in sorted(dataset_root(repo_id).rglob("*.parquet")):
        try:
            pq.read_metadata(path)
        except Exception as e:
            print(f"[pool] '{repo_id}' is corrupt: {path.name}: {e}")
            return False
    return True


def quarantine_shard(repo_id: str) -> None:
    """Move a corrupt shard's directory aside so the next run starts it fresh.

    Renamed (to '<root>__corrupt[N]'), never deleted — partial recovery by
    hand stays possible. Leaving it in place would be worse than deleting:
    its info.json episode count would keep counting toward the next run's
    resume target, and the worker assigned to it would fail dataset init on
    the unreadable parquet.
    """
    root = dataset_root(repo_id)
    dest = root.with_name(root.name + "__corrupt")
    n = 2
    while dest.exists():
        dest = root.with_name(f"{root.name}__corrupt{n}")
        n += 1
    root.rename(dest)
    print(f"[pool] quarantined corrupt shard -> {dest}")


def preflight_validate_datasets(base_repo_id: str | None, n_workers: int) -> None:
    """Quarantine unreadable leftover shards BEFORE launching workers.

    A shard left corrupt by a previous run (worker killed pre-finalize) would
    otherwise poison this run twice: its info.json episode count is counted
    toward the resume target, and the worker assigned to it fails dataset
    init on the unreadable parquet and sits idle. A corrupt BASE is fatal —
    it can't be quarantined without losing all previous runs, so stop and let
    the user decide.
    """
    if not base_repo_id:
        return
    if count_episodes(base_repo_id) > 0 and not dataset_is_readable(base_repo_id):
        raise SystemExit(
            f"base dataset '{base_repo_id}' has corrupt parquet files — refusing "
            f"to start. Repair it or move it aside, then rerun."
        )
    for i in range(n_workers):
        rid = worker_repo_id(base_repo_id, i + 1)
        if dataset_root(rid).exists() and not dataset_is_readable(rid):
            quarantine_shard(rid)


def run_merge(inputs: list[str], output_repo_id: str) -> bool:
    """Merge `inputs` into `output_repo_id` via lerobot's dataset editor."""
    import json

    cmd = [
        "lerobot-edit-dataset", "--operation.type", "merge",
        "--operation.repo_ids", json.dumps(inputs),
        "--new_repo_id", output_repo_id,
    ]
    print(f"[pool] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=SPLATSIM_ROOT).returncode == 0


def config_wants_hub_push(dump_config_path: Path) -> bool:
    """Whether the panel's Push-to-Hub was on, from the coordinator's config dump.

    False when the dump is missing or unreadable (coordinator crashed before
    writing it) — never pushing is the safe failure mode, and the message the
    merge step prints tells the user how to push by hand.
    """
    import json

    try:
        return bool(json.loads(dump_config_path.read_text()).get("push_to_hub", False))
    except (OSError, ValueError):
        return False


def generation_start_time(dump_config_path: Path) -> float | None:
    """When the panel's Start was first pressed, from the coordinator's dump.

    None when the dump is missing or the coordinator never started generating
    (panel closed without a Start) — the elapsed-time print is then skipped.
    """
    import json

    try:
        v = json.loads(dump_config_path.read_text()).get("_generation_started_at")
        return float(v) if v is not None else None
    except (OSError, ValueError, TypeError):
        return None


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def push_merged_to_hub(repo_id: str) -> None:
    """One non-interactive attempt to push the consolidated dataset to the hub.

    Deliberately NOT `lerobot_utils.push_lerobot_to_hub`, whose retry loop
    blocks on input() — this runs at the tail of an unattended pool run, where
    a failed push should end the run with the dataset intact locally, not hang
    it on a prompt.
    """
    from splatsim.utils.lerobot_utils import load_lerobot_dataset

    print(f"[pool] pushing '{repo_id}' to the HF hub (panel had Push-to-Hub on)...")
    try:
        dataset = load_lerobot_dataset(repo_id)
        if dataset is None:
            print(f"[pool] could not load '{repo_id}' — push skipped")
            return
        dataset.push_to_hub()
        print(f"[pool] pushed '{repo_id}' to the hub")
    except Exception as e:
        print(f"[pool] hub push FAILED ({type(e).__name__}: {e}) — the dataset "
              f"is intact locally; check `huggingface-cli whoami` and rerun, or "
              f"push it yourself from Python: "
              f"LeRobotDataset.resume('{repo_id}').push_to_hub()")


def consolidate(args, n_workers: int, push_to_hub: bool = False,
                started_at: float | None = None) -> None:
    """Merge the worker shards into one dataset and remove them.

    The result lands on the BASE repo id — not a `_merged` sibling. That is
    what makes resume work across runs: the launcher's next invocation counts episodes in
    the base, so leaving the result under a different name would make the next
    run regenerate everything from zero. `--merged-repo-id` overrides it and
    leaves the base untouched.

    Because the base is usually one of the merge INPUTS, the merge writes to a
    temporary id first and only replaces the base once the episode count checks
    out. Nothing destructive happens before that verification.
    """
    import shutil

    base = args.base_repo_id
    if not base:
        print("[pool] no --base-repo-id — skipping merge; "
              "merge the shards by hand if you want them consolidated")
        return

    inputs, shards = discover_shards(base, n_workers)
    if not shards:
        print("[pool] no worker shards with episodes — nothing to merge")
        return

    # Use whatever the workers managed to save: a shard whose parquet files
    # never got their footers (worker killed before finalizing) is quarantined
    # and the merge proceeds with the healthy ones. A corrupt BASE is
    # different — it cannot be quarantined (it holds all previous runs), so
    # nothing is touched and the run ends with the shards intact on disk.
    if base in inputs and not dataset_is_readable(base):
        print(f"[pool] base dataset '{base}' is corrupt — NOT merging and not "
              f"touching anything. Repair or remove it, then merge the shards "
              f"by hand.")
        return
    bad = [rid for rid in shards if not dataset_is_readable(rid)]
    for rid in bad:
        quarantine_shard(rid)
    shards = [rid for rid in shards if rid not in bad]
    inputs = [rid for rid in inputs if rid not in bad]
    if not shards:
        print("[pool] every worker shard was corrupt — nothing left to merge")
        return

    counts = {rid: count_episodes(rid) for rid in inputs}
    expected = sum(counts.values())
    print(f"[pool] merging {len(inputs)} dataset(s), {expected} episode(s) total:")
    for rid, n in counts.items():
        print(f"[pool]   {rid}: {n}")

    if not _confirm(args, f"merge into '{args.merged_repo_id or base}'"):
        print("[pool] skipped. To do it yourself:")
        staged = args.merged_repo_id or f"{base}__merge_staging"
        print(f"  lerobot-edit-dataset --operation.type merge \\\n"
              f"    --operation.repo_ids '{inputs}' \\\n"
              f"    --new_repo_id {staged}")
        if not args.merged_repo_id:
            print(f"  # then replace '{base}' with '{staged}' so the next run resumes")
        return

    # Explicit target: write there and leave base + shards decisions to the
    # simple path. Default: stage under a temp id so the base is never
    # half-overwritten if the merge dies partway.
    final_id = args.merged_repo_id or base
    staging_id = final_id if final_id not in inputs else f"{base}__merge_staging"

    if not run_merge(inputs, staging_id):
        print("[pool] merge FAILED — nothing deleted, shards left intact")
        return

    merged_n = count_episodes(staging_id)
    if merged_n != expected:
        print(f"[pool] merge produced {merged_n} episodes, expected {expected} — "
              f"refusing to delete anything. Result is at '{staging_id}'.")
        return

    if staging_id != final_id:
        # Swap staging into place. The old base is already fully represented
        # inside the staged merge (verified by the count above), so removing it
        # loses nothing.
        final_root, staging_root = dataset_root(final_id), dataset_root(staging_id)
        print(f"[pool] replacing '{final_id}' with the merged result")
        shutil.rmtree(final_root, ignore_errors=True)
        staging_root.rename(final_root)

    print(f"[pool] merged -> '{final_id}' ({merged_n} episodes)")
    if started_at is not None:
        print(f"[pool] data generation took {format_duration(time.time() - started_at)} "
              f"(Start pressed -> merged)")

    # The MERGED dataset is what Push-to-Hub means; the workers were told
    # push_to_hub=False so their transient shards never reach the hub.
    if push_to_hub:
        push_merged_to_hub(final_id)

    if args.keep_shards:
        print(f"[pool] keeping {len(shards)} shard(s) (--keep-shards)")
        return
    for rid in shards:
        root = dataset_root(rid)
        # Never delete outside the dataset home, and never the thing we just
        # wrote — a mis-derived path here would be unrecoverable.
        if rid == final_id or not root.exists():
            continue
        print(f"[pool] removing shard {rid}")
        shutil.rmtree(root, ignore_errors=True)


def _confirm(args, action: str) -> bool:
    """True if `action` should proceed. Automatic unless --confirm is set."""
    if not args.confirm:
        return True
    try:
        reply = input(f"\nWhen the dataset is done, type continue to {action}: ")
    except EOFError:
        print("[pool] stdin closed — skipping")
        return False
    except KeyboardInterrupt:
        print("\n[pool] cancelled")
        return False
    return reply.strip().lower() in {"continue", "c", "y", "yes"}


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    ports = [args.base_port + i for i in range(args.workers)]
    busy = [p for p in ports if port_is_open(p)]
    if busy and not args.dry_run:
        raise SystemExit(
            f"port(s) {busy} already in use — another pool is probably still "
            f"running. Pick a different --base-port, or stop the old workers."
        )

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = SPLATSIM_ROOT / log_dir

    # Where the coordinator leaves the panel's effective config for the
    # post-run steps (currently just the merged dataset's hub push).
    dump_config_path = log_dir / "coordinator_config.json"

    if args.dry_run:
        print(f"# token file: {'(exists)' if os.environ.get('SPLATSIM_CONTROL_TOKEN') else '~/.cache/splatsim/control_token'}")
        for port in ports:
            print("SPLATSIM_WORKER_LOCKDOWN=1 "
                  + " ".join(worker_command(args, port))
                  + f"  > {log_dir}/worker_{port}.log 2>&1 &")
        print(" ".join(coordinator_command(args, ports, dump_config_path)))
        return

    # Create the shared secret BEFORE the workers start. They read it lazily
    # per request so the order is not strictly required, but doing it here
    # means a worker is never briefly unable to answer the coordinator.
    ensure_control_token()
    log_dir.mkdir(parents=True, exist_ok=True)
    preflight_validate_datasets(args.base_repo_id, args.workers)

    procs: dict[int, subprocess.Popen] = {}
    try:
        for i, port in enumerate(ports):
            if i and args.launch_stagger > 0:
                time.sleep(args.launch_stagger)
            log_path = log_dir / f"worker_{port}.log"
            cmd = worker_command(args, port)
            print(f"[pool] launching worker on {port} -> {log_path}")
            with open(log_path, "w") as log:
                procs[port] = subprocess.Popen(
                    cmd, cwd=SPLATSIM_ROOT, stdout=log, stderr=subprocess.STDOUT,
                    # Own process group, so a Ctrl-C in this terminal reaches
                    # only us; we then shut the workers down deliberately in
                    # `terminate` rather than having them die mid-episode.
                    start_new_session=True,
                    # Coordinator-only mode: the worker rejects any ZMQ
                    # request that lacks the pool's control token, so a stray
                    # gym/teleop client on the same port can't hijack it
                    # mid-generation (see ZMQRobotServer.serve).
                    env={**os.environ, "SPLATSIM_WORKER_LOCKDOWN": "1"},
                )

        print(f"[pool] waiting for {len(ports)} worker(s) to come up "
              f"(up to {args.startup_timeout:.0f}s; splat loading is slow)...")
        ready = wait_for_workers(procs, args.startup_timeout)
        if not ready:
            raise SystemExit(
                f"no workers came up — check {log_dir}/worker_*.log "
                f"(a bad --robot value usually shows there immediately)"
            )
        if len(ready) < len(ports):
            print(f"[pool] continuing with {len(ready)}/{len(ports)} worker(s): {ready}")

        # A dump left over from a previous run must not masquerade as this
        # run's panel settings if the coordinator dies before writing its own.
        dump_config_path.unlink(missing_ok=True)
        cmd = coordinator_command(args, ready, dump_config_path)
        print(f"[pool] starting coordinator: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=SPLATSIM_ROOT)
    except KeyboardInterrupt:
        print("\n[pool] interrupted")
    finally:
        # Workers must be fully stopped before merging: a live worker still
        # holds its dataset writer open, so its shard is not yet consistent.
        terminate(procs, grace_s=args.stop_grace)
        print("[pool] all workers stopped")

    # Reached whether the coordinator exited cleanly, the panel was closed, or
    # Ctrl-C ended the run — a partial run is still worth consolidating.
    if args.no_merge:
        print("[pool] --no-merge set; shards left on disk")
    else:
        try:
            consolidate(args, len(ports),
                        push_to_hub=config_wants_hub_push(dump_config_path),
                        started_at=generation_start_time(dump_config_path))
        except Exception as e:
            print(f"[pool] merge step failed ({type(e).__name__}: {e}) — "
                  "shards left on disk, nothing deleted")


if __name__ == "__main__":
    main()
