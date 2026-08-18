"""Standalone Tk panel that drives N headless SplatSim trajectory-gen workers.

Runs the SAME "Trajectory Gen Mode" panel the in-process GUI shows, but backed
by nothing local: every Start / Stop is broadcast over ZMQ to a pool of
headless servers, and the status line aggregates their progress. Nothing is
generated in this process — it holds no pybullet client, no splat model, and no
GPU memory.

The point is that the config no longer has to exist before the simulators do.
Launch the workers with no --traj_config_file at all; they idle until this
panel pushes a config and presses go.

First pick a shared secret. The workers bind to 127.0.0.1 so nothing off-box
can reach them, but without a secret ANY local process could drive them; the
workers therefore default-deny remote control until one is set. Export it once,
and launch everything from that same shell so both sides inherit it::

    export SPLATSIM_CONTROL_TOKEN=$(openssl rand -hex 16)

Launch the workers (one per GPU-slot you can afford), each on its own port::

    python scripts/launch_nodes.py --robot <env> --headless --robot_port 6002
    python scripts/launch_nodes.py --robot <env> --headless --robot_port 6003
    python scripts/launch_nodes.py --robot <env> --headless --robot_port 6004

Then this panel::

    python scripts/trajgen_coordinator.py --ports 6002-6004 \\
        --base-repo-id JennyWWW/my_dataset

Each worker writes its own shard — ``JennyWWW/my_dataset_worker1``,
``_worker2``, ``_worker3`` — because two servers appending to one LeRobot
dataset would corrupt it.

RESUME is automatic. The panel's "Num Base Trajectories" is read as the TOTAL
you want to end up with, and everything already on disk counts toward it — the
base dataset AND any shards from previous runs. So with 100 episodes already in
``JennyWWW/my_dataset``, a target of 200 and 5 workers, each worker generates
20 and the result is 100 + 20x5 = 200. Press Start again with the target raised
and it tops up from wherever it left off.

The shards are consolidated back into the BASE dataset when the workers stop —
``launch_trajgen_pool.py`` does that automatically, and it is what lets the
next run resume instead of starting over. Run this script standalone and the
shards are left for you to merge yourself.

Alternative to this script: give ONE of the workers ``--control_gui`` and let
it both generate and host the panel. That skips this process entirely but ties
the panel to a worker's lifetime and needs a display on that machine. This
script exists so the workers can be uniform and display-less.
"""

from __future__ import annotations

import argparse
import time

from splatsim.configs.mode_config import TrajectoryGenModeConfig
from splatsim.utils.paths import traj_config_path
from splatsim.utils.splatsim_gui import SplatSimGui, TrajectoryGenModePanel
from splatsim.utils.trajgen_workers import (
    DEFAULT_START_STAGGER_S,
    TrajGenWorkerPool,
    ensure_control_token,
    parse_ports,
)

# Mode strings owned by TrajectoryGenModePanel. The panel flips the GUI's mode
# when Start / Stop are pressed; we watch for the transition rather than
# reaching into its buttons, so the panel stays untouched.
MODE_RUNNING = "generate_trajectories"
MODE_IDLE = "generate_trajectories_idle"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ports", required=True,
        help="Worker ZMQ ports: '6002,6003' or a range '6002-6005' (or both).",
    )
    ap.add_argument("--host", default="127.0.0.1", help="Worker host (default localhost).")
    ap.add_argument(
        "--base-repo-id", default=None,
        help="Base LeRobot dataset id; worker i writes '<base>_worker<i>'. "
             "Defaults to whatever lerobot_repo_id the loaded config carries.",
    )
    ap.add_argument(
        "--traj-config-file", default=None,
        help="Optional JSON to preload into the panel (same format as the "
             "GUI's Export Config), conventionally "
             "configs/traj_configs/<env>.json. Purely a starting point — "
             "everything stays editable in the panel before you press Start.",
    )
    ap.add_argument(
        "--status-interval", type=float, default=3.0,
        help="Seconds between worker status polls (default 3).",
    )
    ap.add_argument(
        "--preview-interval", type=float, default=1.0, metavar="SECONDS",
        help="Seconds between camera-preview refreshes (default 1). While "
             "generating this just fetches the frame the worker already "
             "rendered, so it costs no GPU time. 0 disables the preview.",
    )
    ap.add_argument(
        "--preview-port", type=int, default=None,
        help="Which worker to watch (default: the first reachable one).",
    )
    ap.add_argument(
        "--exit-when-done", action="store_true",
        help="Close the panel automatically once the target is reached, so an "
             "unattended run can proceed straight to the merge instead of "
             "waiting for someone to close the window.",
    )
    ap.add_argument("--timeout-ms", type=int, default=5000, help="Per-request ZMQ timeout.")
    ap.add_argument(
        "--start-stagger", type=float, default=DEFAULT_START_STAGGER_S,
        metavar="SECONDS",
        help="Delay between starting consecutive workers (default "
             f"{DEFAULT_START_STAGGER_S}). An episode is CPU-bound planning "
             "then GPU-bound rendering; starting in lockstep makes every "
             "worker render at once and queue on the one GPU. 0 disables.",
    )
    ap.add_argument(
        "--env-name", default=None,
        help="Override the workers' EnvConfig.name. Normally auto-detected "
             "from the running workers; only sets the panel's Export/Import "
             "default to configs/traj_configs/<env>.json so this process "
             "follows the same per-env convention as the in-sim GUI.",
    )
    ap.add_argument(
        "--merged-repo-id", default=None,
        help="Where the shards get consolidated (default: back into the base "
             "dataset, so the next run resumes from it).",
    )
    ap.add_argument(
        "--dump-config", default=None, metavar="PATH",
        help="Write the panel's EFFECTIVE config (defaults + preloaded file + "
             "panel edits) as JSON to PATH on exit. launch_trajgen_pool.py "
             "reads it to honor panel settings that outlive this process — "
             "e.g. push_to_hub, which applies to the merged dataset the "
             "launcher assembles, not to the per-worker shards.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    ports = parse_ports(args.ports)
    if not ports:
        raise SystemExit(f"--ports {args.ports!r} resolved to no ports")

    # The panel edits THIS object; push_config serializes it on every Start,
    # so edits made in the GUI take effect without any file round trip.
    config = TrajectoryGenModeConfig()
    if args.traj_config_file:
        from splatsim.utils.config_io import update_dataclass_json

        update_dataclass_json(config, args.traj_config_file, warn=lambda m: print(f"[coord] {m}"))
        print(f"[coord] preloaded config from {args.traj_config_file}")
    if args.base_repo_id:
        config.lerobot_repo_id = args.base_repo_id
    if not config.lerobot_repo_id:
        raise SystemExit(
            "No dataset id — pass --base-repo-id or a --traj-config-file that sets "
            "lerobot_repo_id (workers need somewhere to write)."
        )

    # Creates ~/.cache/splatsim/control_token (0600) on first run, so there is
    # no manual export step. Workers read the same file lazily per request,
    # which is why it is fine for them to have started before this line ran.
    ensure_control_token()

    pool = TrajGenWorkerPool(ports, host=args.host, timeout_ms=args.timeout_ms)
    pool.connect()
    print(f"[coord] driving {len(pool)} worker(s) on ports {pool.ports}")

    # Ask the workers what env they are, so Export/Import defaults to
    # configs/traj_configs/<env>.json without the user restating it.
    env_name = args.env_name or pool.detect_env_name()
    if env_name:
        print(f"[coord] env: {env_name} -> {traj_config_path(env_name)}")
    else:
        print("[coord] env unknown (workers unreachable or mixed) — "
              "config path defaults to the generic one")
    print("[coord] " + pool.plan_work(config).describe().replace("\n", "\n[coord] "))

    # Only the traj-gen panel: this process has no interactive sim to drive and
    # no eval dataset to replay, so offering those modes would be a lie.
    gui = SplatSimGui(
        config,
        initial_mode=MODE_IDLE,
        panels=[TrajectoryGenModePanel()],
        # Unlike the in-sim GUI this process has no env to read a name from,
        # so the convention has to be passed in explicitly.
        traj_config_default_path=traj_config_path(env_name) if env_name else None,
    )
    gui.start()
    gui.set_status(pool.summarize(plan=pool.plan_work(config)))
    # Render the scene once before anything is generated, so a misplaced
    # object or a wrong splat is visible BEFORE committing to a long run.
    _refresh_preview(gui, pool, port=args.preview_port)

    prev_mode = MODE_IDLE
    last_poll = 0.0
    last_preview = 0.0
    # Wall-clock of the FIRST Start press, surfaced to launch_trajgen_pool.py
    # through the --dump-config JSON so it can report how long the whole run
    # took once the shards are merged (the merge happens in that process,
    # after this one exits).
    gen_started_at: float | None = None
    # Stall bookkeeping (see _maybe_redistribute): consecutive failed status
    # polls per port, plus give-up tracking across redistribution attempts.
    unreachable_streaks: dict[int, int] = {}
    stall_state = {"last_disk": -1, "attempts": 0}
    try:
        while True:
            # Drives the panel's own buttons (Export / Import / Start / Stop).
            # Start syncs the widgets into `config` and flips the mode, which
            # is exactly the edge we act on below.
            gui.process_mode_transitions()
            mode = gui.mode

            if mode != prev_mode:
                if mode == MODE_RUNNING:
                    if gen_started_at is None:
                        gen_started_at = time.time()
                    # A fresh Start wipes the stall history — dead workers may
                    # have been relaunched, and the target may have changed.
                    unreachable_streaks.clear()
                    stall_state.update(last_disk=-1, attempts=0)
                    _handle_start(gui, pool, config, args.merged_repo_id,
                                  stagger_s=args.start_stagger)
                elif mode == MODE_IDLE:
                    _handle_stop(gui, pool, config, args.merged_repo_id)
                prev_mode = mode

            now = time.time()
            running = mode == MODE_RUNNING
            if now - last_poll >= args.status_interval:
                last_poll = now
                statuses, plan = _poll_status(gui, pool, running=running, config=config)
                for s in statuses:
                    unreachable_streaks[s.port] = (
                        0 if s.reachable else unreachable_streaks.get(s.port, 0) + 1
                    )
                # Completion is detected from DISK, not from the workers'
                # modes: a worker flips itself back to idle the moment its own
                # shard target is met, so "all idle" would fire while other
                # shards were still running.
                if running and plan is not None and plan.remaining == 0:
                    print("[coord] target reached — generation complete")
                    gui.set_mode(MODE_IDLE)
                    if args.exit_when_done:
                        print("[coord] --exit-when-done: closing")
                        break
                elif running:
                    outcome = _maybe_redistribute(
                        gui, pool, config, statuses, plan,
                        unreachable_streaks, stall_state,
                        stagger_s=args.start_stagger,
                    )
                    if outcome == "give_up":
                        gui.set_mode(MODE_IDLE)
                        if args.exit_when_done:
                            print("[coord] --exit-when-done: closing "
                                  "(INCOMPLETE — see the stall messages above)")
                            break
            # Preview runs on its own (faster) timer so the view stays live
            # while generating. While RUNNING it fetches the frame the worker
            # already rendered for its dataset — no new render, no GPU cost,
            # no collision with the main thread. While IDLE it triggers a
            # fresh render, which is what a pre-flight scene check needs.
            if args.preview_interval > 0 and now - last_preview >= args.preview_interval:
                last_preview = now
                _refresh_preview(gui, pool, quiet=True, live=running,
                                 port=args.preview_port)

            if not gui.is_alive():
                print("[coord] panel closed — leaving workers as they are.")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[coord] interrupted — stopping workers.")
        print("[coord] " + pool.stop_all().describe())
    finally:
        gui.stop()
        pool.close()
        if args.dump_config:
            try:
                import json

                from splatsim.utils.config_io import save_dataclass_json

                save_dataclass_json(config, args.dump_config)
                if gen_started_at is not None:
                    # Ride along in the dump rather than a sidecar file; the
                    # underscore keeps it visibly not-a-config-field if the
                    # JSON is ever imported back into a panel.
                    with open(args.dump_config) as fp:
                        data = json.load(fp)
                    data["_generation_started_at"] = gen_started_at
                    with open(args.dump_config, "w") as fp:
                        json.dump(data, fp, indent=2, sort_keys=True)
                print(f"[coord] effective config written to {args.dump_config}")
            except Exception as e:
                print(f"[coord] could not write --dump-config: {type(e).__name__}: {e}")


def _handle_start(gui: SplatSimGui, pool: TrajGenWorkerPool, config, merged_repo_id,
                  stagger_s: float = DEFAULT_START_STAGGER_S):
    """Plan the remaining work, push it to every worker, then start them.

    Config first, start second: a worker refuses a config push while it is
    generating, so the order is load-bearing rather than stylistic.
    """
    # Re-planned on EVERY Start, not once at launch: episode counts on disk
    # change as the workers run, so a second Start must see the new totals.
    plan = pool.plan_work(config)
    print("[coord] " + plan.describe().replace("\n", "\n[coord] "))

    if plan.remaining == 0:
        msg = (
            f"nothing to do — {plan.already_have} episode(s) already on disk "
            f"meets the target of {plan.target_total}"
        )
        print(f"[coord] {msg}")
        gui.set_status(msg)
        gui.set_mode(MODE_IDLE)
        return

    push = pool.push_config(config, plan=plan)
    print(f"[coord] push_config: {push.describe()}")
    if not push.ok:
        gui.set_status(f"START FAILED — no worker accepted the config ({push.describe()})")
        gui.set_mode(MODE_IDLE)
        return

    # Only workers with a nonzero share: one with none is already at its
    # target and would just flip straight back to idle.
    started = pool.start_all(ports=plan.working_ports, stagger_s=stagger_s)
    print(f"[coord] start: {started.describe()}"
          + (f" (staggered {stagger_s}s apart)" if stagger_s > 0 else ""))
    _print_merge_command(plan, merged_repo_id)

    msg = f"generating {plan.remaining} more (target {plan.target_total}) on {len(plan.working_ports)} worker(s)"
    if not push.all_ok or not started.all_ok:
        # Partial success is still useful — the reachable workers generate —
        # but it must be visible, or a half-dead pool reads as a slow one.
        msg += f" — PROBLEMS: push[{push.describe()}] start[{started.describe()}]"
    gui.set_status(msg)


# A worker mid-episode still answers status requests from its ZMQ dispatch
# thread, so "unreachable" really means crashed/killed — but a single failed
# poll can also be a loaded box missing one 5s deadline. Require this many
# CONSECUTIVE failures before treating a worker as dead.
STALL_UNREACHABLE_POLLS = 3
# Redistributions that produce no new episodes on disk before giving up —
# bounds the restart loop when every surviving worker immediately fails too
# (e.g. the ruckig cloud backend rate-limited: each restarted worker stops
# itself again after one failed episode).
STALL_MAX_ATTEMPTS = 3


def _maybe_redistribute(gui: SplatSimGui, pool: TrajGenWorkerPool, config,
                        statuses, plan, streaks: dict, state: dict,
                        stagger_s: float) -> str | None:
    """Detect a STALLED pool and re-route the missing episodes to survivors.

    The stall this fixes: one worker crashes mid-run (its shard short), the
    others finish their own shares and idle, and the disk count never reaches
    the target — historically the coordinator then waited forever. The stalled
    state is precisely: running, ``plan.remaining > 0``, and every worker
    either reachable-and-idle or unreachable for ``STALL_UNREACHABLE_POLLS``
    consecutive polls. Waiting for the survivors to go idle (rather than
    interrupting them the moment a worker dies) is deliberate — a worker
    refuses config pushes while generating, and its in-flight share is not
    knowably "missing" until it stops.

    Returns None (nothing to do / not stalled yet), "redistributed", or
    "give_up" (no survivors, or ``STALL_MAX_ATTEMPTS`` redistributions in a
    row added nothing to disk).
    """
    if plan is None or plan.remaining <= 0:
        return None
    if any(s.reachable and s.mode != MODE_IDLE for s in statuses):
        return None  # someone is still generating — not a stall
    if any(not s.reachable and streaks.get(s.port, 0) < STALL_UNREACHABLE_POLLS
           for s in statuses):
        return None  # too early to tell dead from busy for those workers

    live_ports = [s.port for s in statuses if s.reachable]
    dead_ports = [s.port for s in statuses if not s.reachable]
    short = plan.remaining
    if not live_ports:
        msg = (f"STALLED {short} episode(s) short of {plan.target_total} with "
               f"no reachable workers {dead_ports} — giving up")
        print(f"[coord] {msg}")
        gui.set_status(msg)
        return "give_up"

    # Progress guard: every redistribution must grow the on-disk count before
    # the next one, or the restarted workers are failing without producing
    # anything and restarting them again is a loop, not a fix.
    if plan.already_have > state["last_disk"]:
        state["attempts"] = 0
    state["last_disk"] = plan.already_have
    state["attempts"] += 1
    if state["attempts"] > STALL_MAX_ATTEMPTS:
        msg = (f"STALLED {short} episode(s) short — giving up after "
               f"{STALL_MAX_ATTEMPTS} redistribution(s) added nothing to disk")
        print(f"[coord] {msg}")
        gui.set_status(msg)
        return "give_up"

    print(f"[coord] STALLED: {short} episode(s) short of {plan.target_total}, "
          f"every reachable worker idle"
          + (f", worker(s) {dead_ports} presumed dead" if dead_ports else "")
          + f" — redistributing to {live_ports} "
          f"(attempt {state['attempts']}/{STALL_MAX_ATTEMPTS})")
    new_plan = pool.plan_work(config, active_ports=live_ports)
    print("[coord] " + new_plan.describe().replace("\n", "\n[coord] "))
    push = pool.push_config(config, plan=new_plan, ports=live_ports)
    print(f"[coord] push_config: {push.describe()}")
    if not push.ok:
        print("[coord] redistribution push failed everywhere — giving up")
        return "give_up"
    started = pool.start_all(ports=new_plan.working_ports, stagger_s=stagger_s)
    print(f"[coord] restart: {started.describe()}")
    gui.set_status(
        f"redistributed {short} episode(s) to {len(new_plan.working_ports)} "
        f"worker(s) after stall"
        + (f" (dead: {dead_ports})" if dead_ports else "")
    )
    return "redistributed"


def _refresh_preview(gui: SplatSimGui, pool: TrajGenWorkerPool, quiet: bool = False,
                     live: bool = False, port: int | None = None) -> None:
    """Show one worker's cameras in the panel's Camera Observations area.

    Only one worker is asked — every worker builds the same scene, and a splat
    render is expensive. Failures are reported rather than left as an empty
    black area, which would look like a broken scene instead of a missing one.

    ``live=True`` (used while generating) reads the worker's already-rendered
    frame instead of forcing a new render.
    """
    try:
        frames, note = pool.preview_frames(port=port, live=live)
    except Exception as e:
        if not quiet:
            print(f"[coord] scene preview failed: {type(e).__name__}: {e}")
        return
    if frames:
        gui.update_camera_images(frames)
        if not quiet:
            print(f"[coord] scene preview: {', '.join(sorted(frames))}")
    elif not quiet:
        print(f"[coord] no scene preview — {note}")


def _print_merge_command(plan, merged_repo_id) -> None:
    """Say where the shards end up. Normally that is automatic.

    `launch_trajgen_pool.py` consolidates them into the base dataset once the
    workers stop, so no command is printed for the usual path — printing one
    invited running it by hand into a `<base>_merged` id, which leaves the base
    empty and makes the NEXT run regenerate everything instead of resuming.
    Only a coordinator driven standalone needs the raw invocation.
    """
    target = merged_repo_id or plan.base_repo_id
    shards = [s.repo_id for s in plan.shards if s.to_generate > 0]
    print(f"[coord] shards: {', '.join(shards)}")
    print(f"[coord] these are consolidated into '{target}' when the workers "
          "stop (launch_trajgen_pool.py does it automatically).")


def _handle_stop(gui: SplatSimGui, pool: TrajGenWorkerPool, config=None, merged_repo_id=None) -> None:
    stopped = pool.stop_all()
    print(f"[coord] stop: {stopped.describe()}")
    gui.set_status(f"stopped — {stopped.describe()}")
    # Re-print the merge command against the CURRENT on-disk counts, so the
    # line reflects what actually got generated (shards that stayed empty are
    # left out of the merge).
    if config is not None:
        try:
            _print_merge_command(pool.plan_work(config), merged_repo_id)
        except Exception as e:
            print(f"[coord] could not compute merge command: {e}")


def _poll_status(gui: SplatSimGui, pool: TrajGenWorkerPool, running: bool,
                 config=None):
    """Refresh the status line: episodes on disk vs the target you typed.

    The plan is recomputed here rather than reusing the one from Start, so the
    line stays right when the panel's target is edited while idle. Returns
    ``(statuses, plan)`` so the main loop's done/stall checks reuse this
    poll's snapshot instead of re-querying.
    """
    statuses = pool.statuses()
    plan = None
    if config is not None:
        try:
            plan = pool.plan_work(config)
        except Exception:
            plan = None  # fall back to the raw worker-reported sum
    line = pool.summarize(statuses, plan=plan)
    if running:
        line += "  |  " + pool.per_worker_line(statuses)
    gui.set_status(line)
    if running:
        for s in statuses:
            if not s.reachable:
                print(f"[coord] {s.describe()}")
    return statuses, plan


if __name__ == "__main__":
    main()
