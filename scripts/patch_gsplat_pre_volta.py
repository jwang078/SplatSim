#!/usr/bin/env python3
"""Make gsplat's CUDA extension compile for pre-Volta GPUs (sm_50/sm_60/sm_61).

gsplat JIT-compiles `gsplat_cuda` from its bundled sources the first time a
render happens, for whatever arch the GPU reports. Four of those sources call
`cg::labeled_partition`, which needs the independent thread scheduling
introduced in Volta, so on a GTX 10-series card the build dies with

    error: namespace "cooperative_groups" has no member "labeled_partition"

Every one of those calls is inside a *backward* (gradient) kernel, where it
groups the threads that share a gaussian so their gradients can be summed once
and written with a single atomic. SplatSim only ever renders forward, but the
whole extension is one module: one file that will not compile takes the
renderer with it.

So on pre-Volta we swap in a one-thread group. `warpSum` then reduces over a
group of one (a no-op) and every thread does its own atomic add -- same sums,
more atomics, and the extension compiles. The forward kernels SplatSim
actually uses are untouched.

Idempotent: run it again and it does nothing. install.sh calls it for GPUs
below sm_70; run it by hand after any `pip install gsplat`, which restores the
unpatched sources.
"""

import sys
from pathlib import Path

MARKER = "SPLATSIM_LABELED_PARTITION"
SHIM = """
// --- SplatSim: pre-Volta fallback (see scripts/patch_gsplat_pre_volta.py) ---
// cg::labeled_partition needs sm_70+. It is used only in the backward kernels
// below, to sum a gaussian's gradients across threads before one atomic add.
// A one-thread group gives the same sums via one atomic per thread, and lets
// this file compile for sm_5x/sm_6x so the forward renderer works at all.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define SPLATSIM_LABELED_PARTITION(warp, label)                                \\
    cg::tiled_partition<1>(cg::this_thread_block())
#else
#define SPLATSIM_LABELED_PARTITION(warp, label) cg::labeled_partition(warp, label)
#endif
"""


def gsplat_csrc() -> Path:
    import gsplat  # noqa: F401  (import for its location only)

    return Path(gsplat.__file__).parent / "cuda" / "csrc"


def patch(path: Path) -> bool:
    src = path.read_text()
    if MARKER in src:
        return False
    if "cg::labeled_partition(" not in src:
        return False
    src = src.replace("cg::labeled_partition(", MARKER + "(")
    # Insert the shim after the file's last #include, so it sees <cooperative_groups>.
    lines = src.splitlines(keepends=True)
    last = max(i for i, ln in enumerate(lines) if ln.startswith("#include"))
    lines.insert(last + 1, SHIM)
    path.write_text("".join(lines))
    return True


def main() -> int:
    csrc = gsplat_csrc()
    if not csrc.is_dir():
        print(f"gsplat sources not found at {csrc} — nothing to patch")
        return 0
    touched = [p.name for p in sorted(csrc.glob("*.cu")) if patch(p)]
    print(
        f"gsplat pre-Volta patch: {len(touched)} file(s) patched"
        + (f" ({', '.join(touched)})" if touched else " — already up to date")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
