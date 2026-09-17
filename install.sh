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
#   2. torch from the CUDA 12.8 index — the +cu128 builds are not on PyPI
#   3. the rest of the pip deps  — pyproject.toml
#   4. source-built CUDA extensions — their setup.py IMPORTS torch, so torch
#      must already be present; hence --no-build-isolation
#
# Knobs (environment variables):
#   ENV_NAME=splatsim        conda env to create/use
#   ENV_ACTION=use|recreate  what to do if ENV_NAME already exists (else: ask)
#   ENV_ONLY=true            stop after the conda env is ready
#   LEROBOT_DIR, SKIP_LEROBOT, TORCH_INDEX — see below
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

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
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
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

say "2/4  torch stack (CUDA 12.8 index)"
# Installed BEFORE the pyproject resolve so the range constraints there are
# already satisfied and pip never falls back to a CPU wheel from PyPI.
pip install --index-url "$TORCH_INDEX" \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128"

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
raise SystemExit(0 if ok else 1)
PY
say "done — activate with: conda activate $ENV_NAME"
