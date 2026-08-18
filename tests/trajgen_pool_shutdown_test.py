#!/usr/bin/env python
"""Acceptance test for graceful Ctrl-C / SIGTERM handling in the trajgen pool.

Encodes the failure seen on 2026-08-17: Ctrl-C on launch_trajgen_pool sent
SIGTERM to the workers, the default handler killed them mid-parquet-write,
every shard's parquet files were left without footers ("Parquet magic bytes
not found"), and the merge failed wholesale — losing the whole run.

What is covered, without booting a simulator:

  1. SHARD VALIDATION — dataset_is_readable accepts a finalized parquet and
     rejects a footerless (truncated) one.
  2. QUARANTINE — quarantine_shard renames a corrupt shard aside (never
     deletes), picks a fresh __corruptN name when one already exists, and
     preflight_validate_datasets applies it while leaving healthy shards and
     the base alone.
  3. STOP GRACE — terminate() waits for a worker that traps SIGTERM and
     finalizes before exiting (the marker file must exist afterwards), and
     still SIGKILLs one that ignores SIGTERM past the grace period.
  4. SERVER FLAGS — PybulletRobotServerBase.request_shutdown sets the flag
     the serve loop polls, and safe_to_interrupt mirrors the dataset
     critical-section marker the launch_nodes signal handler consults.

Usage:
  python tests/trajgen_pool_shutdown_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# dataset_root() resolves HF_LEROBOT_HOME at import time inside lerobot —
# point it at a sandbox BEFORE anything imports lerobot.
_TMP = tempfile.TemporaryDirectory(prefix="trajgen_pool_test_")
os.environ["HF_LEROBOT_HOME"] = str(Path(_TMP.name) / "lerobot")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from launch_trajgen_pool import (  # noqa: E402
    dataset_is_readable,
    preflight_validate_datasets,
    quarantine_shard,
    terminate,
)
from splatsim.utils.trajgen_workers import dataset_root  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not ok else ""))
    PASS += ok
    FAIL += not ok


def make_dataset(repo_id: str, episodes: int, corrupt: bool) -> Path:
    """Fabricate the minimal on-disk shape of a LeRobot shard."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = dataset_root(repo_id)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    path = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pa.table({"x": list(range(10))}), path)
    if corrupt:
        # A killed writer leaves everything but the footer — model that by
        # truncating the tail (footer + magic) off a valid file.
        data = path.read_bytes()
        path.write_bytes(data[: len(data) - 16])
    return root


# ── 1. shard validation ─────────────────────────────────────────────────────

good = make_dataset("test/pool_good", episodes=3, corrupt=False)
bad = make_dataset("test/pool_bad", episodes=2, corrupt=True)

check("readable: finalized parquet accepted", dataset_is_readable("test/pool_good"))
check("readable: truncated parquet rejected", not dataset_is_readable("test/pool_bad"))
check("readable: nonexistent dataset vacuously ok", dataset_is_readable("test/pool_absent"))

# ── 2. quarantine + preflight ───────────────────────────────────────────────

quarantine_shard("test/pool_bad")
check("quarantine: root moved aside", not bad.exists())
check("quarantine: __corrupt sibling created",
      bad.with_name(bad.name + "__corrupt").exists())

bad2 = make_dataset("test/pool_bad", episodes=1, corrupt=True)
quarantine_shard("test/pool_bad")
check("quarantine: second failure gets a fresh name",
      bad2.with_name(bad2.name + "__corrupt2").exists())

base = make_dataset("test/pool_base", episodes=5, corrupt=False)
w1 = make_dataset("test/pool_base_worker1", episodes=2, corrupt=True)
w2 = make_dataset("test/pool_base_worker2", episodes=2, corrupt=False)
preflight_validate_datasets("test/pool_base", n_workers=2)
check("preflight: corrupt shard quarantined", not w1.exists())
check("preflight: healthy shard untouched", w2.exists())
check("preflight: base untouched", base.exists())

corrupt_base_ok = False
make_dataset("test/pool_badbase", episodes=5, corrupt=True)
try:
    preflight_validate_datasets("test/pool_badbase", n_workers=1)
except SystemExit:
    corrupt_base_ok = True
check("preflight: corrupt base is fatal, not quarantined",
      corrupt_base_ok and dataset_root("test/pool_badbase").exists())

# ── 3. terminate() grace period ─────────────────────────────────────────────

WORKER_SRC = textwrap.dedent("""
    import signal, sys, time
    ignore, marker = sys.argv[1] == "ignore", sys.argv[2]
    def handler(signum, frame):
        # graceful worker: "finalize the dataset", then exit
        time.sleep(1.0)
        open(marker, "w").write("finalized")
        sys.exit(0)
    if not ignore:
        signal.signal(signal.SIGTERM, handler)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)
""")

marker = Path(_TMP.name) / "finalized_marker"
graceful = subprocess.Popen([sys.executable, "-c", WORKER_SRC, "trap", str(marker)])
time.sleep(0.5)  # let it install its handler
t0 = time.time()
terminate({6002: graceful}, grace_s=30.0)
check("terminate: waits for graceful worker to finalize",
      marker.exists() and graceful.returncode == 0,
      f"marker={marker.exists()} rc={graceful.returncode}")
check("terminate: returns as soon as the worker exits", time.time() - t0 < 10.0)

stubborn = subprocess.Popen([sys.executable, "-c", WORKER_SRC, "ignore", str(marker) + "2"])
time.sleep(0.5)
terminate({6003: stubborn}, grace_s=2.0)
stubborn.wait(timeout=5.0)
check("terminate: SIGKILLs a worker that ignores SIGTERM past the grace",
      stubborn.returncode == -9, f"rc={stubborn.returncode}")

# ── 4. server-side shutdown flags ───────────────────────────────────────────
# Instantiating the server needs a GPU + scene; the flag protocol is plain
# attribute logic, so exercise it on an uninitialized instance.

from splatsim.robots.sim_robot_pybullet_base import PybulletRobotServerBase  # noqa: E402

srv = object.__new__(PybulletRobotServerBase)
check("server: shutdown not requested initially", not srv._shutdown_requested)
check("server: safe to interrupt initially", srv.safe_to_interrupt)
srv._in_dataset_critical_section = True
check("server: critical section defers interrupts", not srv.safe_to_interrupt)
srv._in_dataset_critical_section = False
srv.request_shutdown()
check("server: request_shutdown sets the serve-loop flag", srv._shutdown_requested)

# ── 5. finalize-after-interrupt leaves a readable dataset ───────────────────
# The serve() finally-path: episodes saved, one PARTIAL episode in the buffer
# (the aborted one), then finalize_lerobot_dataset. The result must be
# readable (footers written) with only the complete episodes.

import numpy as np  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from splatsim.utils.lerobot_utils import finalize_lerobot_dataset  # noqa: E402

ds = LeRobotDataset.create(
    repo_id="test/pool_finalize",
    fps=30,
    features={
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "action": {"dtype": "float32", "shape": (4,), "names": None},
    },
)
for _ in range(2):
    for _ in range(5):
        ds.add_frame({
            "observation.state": np.zeros(4, dtype=np.float32),
            "action": np.zeros(4, dtype=np.float32),
            "task": "t",
        })
    ds.save_episode()
# the episode the Ctrl-C aborted mid-flight:
ds.add_frame({
    "observation.state": np.zeros(4, dtype=np.float32),
    "action": np.zeros(4, dtype=np.float32),
    "task": "t",
})
finalize_lerobot_dataset(ds)

check("finalize: every parquet has a footer", dataset_is_readable("test/pool_finalize"))
reloaded = LeRobotDataset("test/pool_finalize")
check("finalize: partial episode discarded, complete ones kept",
      reloaded.meta.total_episodes == 2,
      f"total_episodes={reloaded.meta.total_episodes}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
