#!/usr/bin/env bash
# SplatSim pip-layer installer. Run AFTER creating + activating the conda env:
#
#     conda env create -f environment.yml
#     conda activate splatsim
#     ./install.sh
#
# Order is the whole point of this script and cannot be expressed in
# pyproject.toml:
#   1. git submodules            — not a pip concept
#   2. torch from the CUDA 12.8 index — the +cu128 builds are not on PyPI
#   3. the rest of the pip deps  — pyproject.toml
#   4. source-built CUDA extensions — their setup.py IMPORTS torch, so torch
#      must already be present; hence --no-build-isolation
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd .. 2>/dev/null && pwd)/lerobot}"
SKIP_LEROBOT="${SKIP_LEROBOT:-false}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no conda env active. Run: conda env create -f environment.yml && conda activate splatsim" >&2
    exit 1
fi
command -v nvcc >/dev/null || {
    echo "ERROR: nvcc not found. It comes from environment.yml (cuda-nvcc); is the env active?" >&2; exit 1; }
say "env: $CONDA_PREFIX | python $(python -V 2>&1 | cut -d' ' -f2) | nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"

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

# LeRobot is a co-developed SIBLING checkout, not a pinned dependency — it is
# imported by the dataset/eval integration (splatsim/utils/lerobot_*.py,
# rrt_to_goal.py). Installed editable so both repos can be worked on together.
if [[ "$SKIP_LEROBOT" != "true" ]]; then
    if [[ -d "$LEROBOT_DIR" ]]; then
        say "LeRobot (editable, from $LEROBOT_DIR)"
        # [dataset] is what the recording / eval-replay paths import
        # (LeRobotDataset needs `datasets` + torchcodec).
        pip install -e "$LEROBOT_DIR[dataset]"
    else
        cat >&2 <<MSG

NOTE: LeRobot not found at $LEROBOT_DIR — skipping.
      The sim server runs without it; dataset recording, eval-benchmark
      replay and policy agents will not import until you install it:
          git clone git@github.com:jwang078/lerobot.git "$LEROBOT_DIR"
          pip install -e "$LEROBOT_DIR[dataset]"
      Or re-run with LEROBOT_DIR=/path/to/lerobot ./install.sh
      Set SKIP_LEROBOT=true to silence this.
MSG
    fi
fi

say "verifying"
python - <<'PY'
import importlib
ok=True
for m in ("torch","torchvision","gsplat","pybullet","diff_gaussian_rasterization","simple_knn","splatsim"):
    try:
        importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e:
        ok=False; print(f"  {m}: FAIL -> {type(e).__name__}: {e}")
import torch
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
raise SystemExit(0 if ok else 1)
PY
say "done"
