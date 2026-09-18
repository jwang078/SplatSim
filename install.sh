#!/usr/bin/env bash
# SplatSim installer. One command from a fresh clone:
#
#     ./install.sh
#
# It creates (or, with your say-so, reuses / recreates) the `splatsim` conda
# env from environment.yml, then installs everything into it in the one
# order that works — which is the whole point of this script and cannot be
# expressed in pyproject.toml:
#   1. git submodules            — not a pip concept
#   2. torch from the CUDA wheel index that covers THIS GPU (see the
#      TORCH_INDEX block below) — those builds are not on PyPI
#   3. the rest of the pip deps  — pyproject.toml
#   4. source-built CUDA extensions — their setup.py IMPORTS torch, so torch
#      must already be present; hence --no-build-isolation
#
# Knobs (environment variables):
#   ENV_NAME=splatsim        conda env to create/use
#   ENV_ACTION=use|recreate  what to do if ENV_NAME already exists (else: ask)
#   ENV_ONLY=true            stop after the conda env is ready
#   TORCH_CUDA_ARCH_LIST     GPU archs to compile the CUDA extensions for
#                            (default: the installed GPU's, via nvidia-smi)
#   TORCH_INDEX              pip index for the torch wheels (default: picked
#                            from the GPU's compute capability, see below)
#   LEROBOT_DIR, SKIP_LEROBOT — see below
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ── GPU check — before anything else ───────────────────────────────────────
# Everything below ends in CUDA extensions compiled for THIS machine's GPU,
# so there is no point creating an env or installing anything if the driver
# cannot see one (no GPU, or a kernel update that needs a reboot first).
# nvidia-smi prints its "couldn't communicate with the driver" message on
# stdout, so keep only lines that look like a compute capability (e.g. 6.1).
# (`|| true` because grep exits 1 on no match, which set -e would turn into
# a silent exit before the error message below.)
GPU_ARCHS="$( { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | grep -Ex '[0-9]+\.[0-9]+' | sort -u | tr '\n' ' ' | sed 's/ *$//'; } || true)"
[[ -n "$GPU_ARCHS" ]] || {
    echo "ERROR: nvidia-smi found no GPU (driver not loaded?). Nothing was installed." >&2
    echo "       Fix the driver first (a reboot after a kernel update is the usual cause) and re-run." >&2
    exit 1; }

# ── 0. conda env — decided BEFORE any other work ───────────────────────────
ENV_NAME="${ENV_NAME:-splatsim}"
ENV_ACTION="${ENV_ACTION:-}"
REQUIRED_PY="$(sed -n 's/^\s*-\s*python=\([0-9]*\.[0-9]*\).*/\1/p' environment.yml)"

command -v conda >/dev/null || {
    echo "ERROR: conda not found on PATH. Install miniforge/miniconda first." >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

ask() {  # ask "<prompt>" "<valid letters>" -> echoes the chosen letter
    local prompt="$1" valid="$2" ans
    if [[ ! -t 0 ]]; then
        echo "ERROR: '$ENV_NAME' already exists and stdin is not a terminal." >&2
        echo "       Re-run with ENV_ACTION=use (install into it) or ENV_ACTION=recreate (delete + rebuild)." >&2
        exit 1
    fi
    while true; do
        read -r -p "$prompt " ans
        ans="${ans,,}"
        [[ -n "$ans" && "$valid" == *"$ans"* ]] && { echo "$ans"; return; }
    done
}

if env_exists; then
    # Inspect what's there so the choice is informed: the python minor
    # version must match environment.yml (extensions are built against it)
    # and nvcc must be present (comes from environment.yml's cuda-nvcc).
    env_py="$(conda run -n "$ENV_NAME" python -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "unknown")"
    if conda run -n "$ENV_NAME" bash -c 'command -v nvcc' >/dev/null 2>&1; then env_nvcc="yes"; else env_nvcc="MISSING"; fi
    compatible=true
    [[ "$env_py" == "$REQUIRED_PY" && "$env_nvcc" == "yes" ]] || compatible=false

    say "conda env '$ENV_NAME' already exists (python $env_py, nvcc: $env_nvcc)"
    if [[ -z "$ENV_ACTION" ]]; then
        if $compatible; then
            echo "It looks compatible with environment.yml (python $REQUIRED_PY + nvcc)."
            choice="$(ask "[u]se it and install into it / [r]ecreate it from scratch / [q]uit?" "urq")"
        else
            echo "It does NOT match environment.yml (needs python $REQUIRED_PY and nvcc) — installing into it is unlikely to work."
            choice="$(ask "[r]ecreate it from scratch (deletes the env) / [u]se it anyway / [q]uit?" "ruq")"
        fi
        case "$choice" in u) ENV_ACTION=use ;; r) ENV_ACTION=recreate ;; q) echo "Aborted; nothing was changed."; exit 0 ;; esac
    fi
    case "$ENV_ACTION" in
        use)      say "installing into the existing '$ENV_NAME'" ;;
        recreate) say "removing '$ENV_NAME' and recreating it from environment.yml"
                  set +u; conda deactivate 2>/dev/null || true; set -u
                  conda env remove -n "$ENV_NAME" -y
                  conda env create -f environment.yml -n "$ENV_NAME" -y ;;
        *) echo "ERROR: ENV_ACTION must be 'use' or 'recreate' (got '$ENV_ACTION')" >&2; exit 1 ;;
    esac
else
    say "creating conda env '$ENV_NAME' from environment.yml"
    conda env create -f environment.yml -n "$ENV_NAME" -y
fi

# conda's own activate/deactivate hooks reference unset variables, which
# `set -u` would turn into a hard error — relax it just around them.
set +u
conda activate "$ENV_NAME"
set -u
command -v nvcc >/dev/null || {
    echo "ERROR: nvcc not found in '$ENV_NAME'. It comes from environment.yml (cuda-nvcc); recreate the env (ENV_ACTION=recreate)." >&2; exit 1; }
say "env: $CONDA_PREFIX | python $(python -V 2>&1 | cut -d' ' -f2) | nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
if [[ "${ENV_ONLY:-false}" == "true" ]]; then
    echo "ENV_ONLY=true — conda env is ready; stopping here. Activate with: conda activate $ENV_NAME"; exit 0
fi

# ── knobs for the pip layer ────────────────────────────────────────────────
# PyTorch's CUDA wheels do not all cover the same GPU generations, and a
# wheel with no kernels for your card still reports `cuda available: True`:
# it only fails at the first kernel launch, with "no kernel image is
# available for execution on the device". So the index is chosen from the
# compute capability nvidia-smi reported above, not hard-coded:
#
#   cu126   sm_50 .. sm_90     Maxwell/Pascal/Volta .. Hopper
#   cu128   sm_75 .. sm_120    Turing and newer, incl. Blackwell (RTX 50xx)
#
# i.e. CUDA 12.8 dropped everything before Turing, and CUDA 12.6 predates
# Blackwell. The oldest card in the machine decides, since torch is one
# install for all of them.
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"        # must satisfy pyproject.toml
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
cap_int() { local c="${1%%.*}" m="${1##*.}"; echo $(( 10#$c * 10 + 10#$m )); }
MIN_CAP=9999; MAX_CAP=0
for a in $GPU_ARCHS; do
    n="$(cap_int "$a")"
    if (( n < MIN_CAP )); then MIN_CAP=$n; fi
    if (( n > MAX_CAP )); then MAX_CAP=$n; fi
done
if (( MIN_CAP >= 75 )); then TORCH_CUDA=cu128; else TORCH_CUDA=cu126; fi
if (( MIN_CAP < 75 && MAX_CAP > 90 )); then
    echo "WARNING: this machine mixes a pre-Turing GPU (sm_$MIN_CAP) with a post-Hopper one (sm_$MAX_CAP)." >&2
    echo "         No single torch wheel covers both; building for sm_$MIN_CAP ($TORCH_CUDA)." >&2
    echo "         Set TORCH_INDEX / CUDA_VISIBLE_DEVICES if you want the newer card instead." >&2
fi
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_CUDA}"
# Where LeRobot lives. Default is INSIDE this repo (external/, gitignored) so
# the installer never creates anything outside the directory you ran it in.
# An existing sibling checkout at ../lerobot is used if present (reading it
# is fine; we just won't create one there). LEROBOT_DIR overrides both.
if [[ -z "${LEROBOT_DIR:-}" ]]; then
    if [[ -d "external/lerobot" ]]; then
        LEROBOT_DIR="$PWD/external/lerobot"
    elif [[ -d "../lerobot" ]]; then
        LEROBOT_DIR="$(cd ../lerobot && pwd)"
        echo "NOTE: using the existing LeRobot checkout at $LEROBOT_DIR (set LEROBOT_DIR to override)"
    else
        LEROBOT_DIR="$PWD/external/lerobot"
    fi
fi
SKIP_LEROBOT="${SKIP_LEROBOT:-false}"
export SKIP_LEROBOT   # read by the verify step below

say "1/4  git submodules"
git submodule update --init --recursive

say "2/4  torch stack ($TORCH_INDEX — GPU arch(s): $GPU_ARCHS)"
# Installed BEFORE the pyproject resolve so the range constraints there are
# already satisfied and pip never falls back to a CPU wheel from PyPI.
pip install --index-url "$TORCH_INDEX" \
    "torch==$TORCH_VERSION+$TORCH_CUDA" \
    "torchvision==$TORCHVISION_VERSION+$TORCH_CUDA" \
    "torchaudio==$TORCH_VERSION+$TORCH_CUDA"

# Two dependencies cannot come from the `pip install -e .` resolve below and
# have to be placed FIRST, the same way torch is:
#
#   ghalton  (via pybullet-planning) — the PyPI sdist hard-codes the
#            clang-only flag `-stdlib=libc++`, so conda's gcc rejects it.
#            `submodules/ghalton` is a fork with that line removed.
#   evdev    (via pynput) — it generates its C source from the HOST kernel
#            headers in /usr/include, then compiles against conda's older
#            sysroot, so newly added key codes come out undeclared
#            ("error: 'KEY_LINK_PHONE' undeclared"). Building it with the
#            system gcc keeps the headers and the compiler consistent.
say "3a/4  dependencies that need a non-default toolchain (ghalton, evdev)"
pip install --no-build-isolation submodules/ghalton
CC=/usr/bin/gcc pip install "evdev==1.9.2"

say "3b/4  SplatSim + pip dependencies"
pip install -e .                      # add '.[hardware]' for a physical xArm

say "4/4  source-built submodules (--no-build-isolation: they import torch at build time)"

# torch's build helper needs to know which GPU archs to compile for. Left
# unset it asks the driver itself, but pinning it is explicit and lets
# TORCH_CUDA_ARCH_LIST override (e.g. to build for a different machine).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$GPU_ARCHS}"
echo "NOTE: compiling CUDA extensions for GPU arch(s): $TORCH_CUDA_ARCH_LIST"

# diff-gaussian-rasterization is pinned to upstream graphdeco-inria, which
# SplatSim needs two changes to. Applied here rather than committed because
# the submodule points at a repo we do not control; both are idempotent.
#
#   cstdint      — upstream omits it, so the uint32_t/uintptr_t uses in
#                  rasterizer_impl.h do not compile under gcc 13.
#   near plane   — upstream culls every gaussian closer than 0.2 m to the
#                  camera. The wrist camera works well inside that, so
#                  close-up geometry (grapes, the gripper's own fingers)
#                  would disappear from the render. 0.01 m keeps it.
DGR=submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization
grep -q '#include <cstdint>' "$DGR/cuda_rasterizer/rasterizer_impl.h" || \
    sed -i 's|#include <cuda_runtime_api.h>|#include <cstdint>\n#include <cuda_runtime_api.h>|' \
        "$DGR/cuda_rasterizer/rasterizer_impl.h"
sed -i 's/if (p_view.z <= 0.2f)/if (p_view.z <= 0.01f)/' "$DGR/cuda_rasterizer/auxiliary.h"

pip install -e submodules/gaussian-splatting-wrapper
pip install --no-build-isolation "$DGR"
pip install --no-build-isolation submodules/simple-knn
pip install -e submodules/pybullet-playground-wrapper
pip install -e submodules/gello_software
pip install -r submodules/gello_software/requirements.txt
pip install -e submodules/gello_software/third_party/DynamixelSDK/python

# gsplat does NOT build its CUDA extension at install time — it JIT-compiles
# from its bundled sources on the first render, inside whatever process is
# running the sim. Two things have to be true for that compile to work, and
# both are easier to fix here than to debug at 3am behind a pybullet window.
say "gsplat's runtime JIT compile (prerequisites, then a warm-up build)"

# 1. Headers. The CUDA packages from the `nvidia` channel put their headers in
#    $CONDA_PREFIX/targets/x86_64-linux/include. nvcc adds that directory
#    itself, but gsplat's plain C++ sources are compiled by the host gcc,
#    which does not — so Rasterization.cpp fails on `#include <cuda_runtime.h>`
#    even though nvcc is right there. Link them where everything looks
#    ($CONDA_PREFIX/include is what torch passes as -isystem).
CUDA_TARGET_INC="$CONDA_PREFIX/targets/x86_64-linux/include"
if [[ -d "$CUDA_TARGET_INC" ]]; then
    linked=0
    for hdr in "$CUDA_TARGET_INC"/*; do
        dest="$CONDA_PREFIX/include/$(basename "$hdr")"
        [[ -e "$dest" ]] || { ln -sfn "$hdr" "$dest"; linked=$((linked+1)); }
    done
    echo "NOTE: linked $linked CUDA header(s) into \$CONDA_PREFIX/include"
fi

# 2. Pre-Volta GPUs. gsplat's projection kernels use cg::labeled_partition,
#    which needs sm_70+; without this the extension does not compile at all on
#    a GTX 10-series card. See the script for why it is safe here.
if (( MIN_CAP < 70 )); then
    python scripts/patch_gsplat_pre_volta.py
fi

# Compile it now rather than on the user's first render — it takes minutes,
# and a failure belongs in this script's output, not in a sim session.
python - <<'GSPLAT'
import torch
from gsplat import spherical_harmonics
sh = torch.zeros(4, 1, 3, device="cuda")
dirs = torch.zeros(4, 3, device="cuda")
spherical_harmonics(0, dirs, sh)
print("  gsplat CUDA extension built and loaded")
GSPLAT

# LeRobot: our fork (github.com/jwang078/lerobot) is co-developed with this
# repo — it holds the training / DAgger side, and SplatSim's dataset
# recording + eval-benchmark replay read and write LeRobot datasets through
# it. Cloned into external/lerobot (see LEROBOT_DIR above) if not already
# available, installed editable so both can be worked on together.
# SKIP_LEROBOT=true skips the step entirely (the bare simulator still runs).
LEROBOT_URL="${LEROBOT_URL:-https://github.com/jwang078/lerobot.git}"
if [[ "$SKIP_LEROBOT" != "true" ]]; then
    if [[ ! -d "$LEROBOT_DIR" ]]; then
        say "LeRobot: cloning $LEROBOT_URL -> $LEROBOT_DIR"
        git clone "$LEROBOT_URL" "$LEROBOT_DIR"
    fi
    say "LeRobot (editable, from $LEROBOT_DIR)"
    # [dataset] is what the recording / eval-replay paths import
    # (LeRobotDataset needs `datasets` + torchcodec).
    pip install -e "$LEROBOT_DIR[dataset]"
fi

say "verifying"
python - <<'PY'
import importlib
ok=True
mods=["torch","torchvision","gsplat","pybullet","diff_gaussian_rasterization","simple_knn","splatsim"]
import os
if os.environ.get("SKIP_LEROBOT","false")!="true":
    mods += ["lerobot","lerobot.datasets.lerobot_dataset"]
for m in mods:
    try:
        importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e:
        ok=False; print(f"  {m}: FAIL -> {type(e).__name__}: {e}")
import torch
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
# `cuda available` is not enough: it is True even when the wheel carries no
# kernels for this GPU. Launch one and see.
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").add_(1).sum().item()
        print(f"  cuda kernel on {torch.cuda.get_device_name(0)}: OK")
    except Exception as e:
        ok = False
        cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
        print(f"  cuda kernel: FAIL -> {type(e).__name__}: {e}")
        print(f"    this GPU is {cap}; this torch build has kernels for "
              f"{' '.join(torch.cuda.get_arch_list())}")
        print("    -> wrong wheel index; re-run with TORCH_INDEX set to a CUDA "
              "version whose wheels cover " + cap)
else:
    ok = False
    print("  cuda available: FAIL -> torch cannot see the GPU (CPU-only wheel?)")
raise SystemExit(0 if ok else 1)
PY
say "done — activate with: conda activate $ENV_NAME"
