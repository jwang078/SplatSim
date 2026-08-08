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

say "3/4  SplatSim + pip dependencies"
pip install -e .                      # add '.[hardware]' for a physical xArm

say "4/4  source-built submodules (--no-build-isolation: they import torch at build time)"
pip install -e submodules/gaussian-splatting-wrapper
pip install --no-build-isolation \
    submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization
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
        pip install -e "$LEROBOT_DIR"
    else
        cat >&2 <<MSG

NOTE: LeRobot not found at $LEROBOT_DIR — skipping.
      The LeRobot dataset/eval integration will not import without it:
          git clone git@github.com:jwang078/lerobot.git "$LEROBOT_DIR"
          pip install -e "$LEROBOT_DIR"
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
