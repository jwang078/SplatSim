"""Load and save named eval benchmark presets from configs/eval_benchmark_presets.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

_PRESETS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "eval_benchmark_presets.yaml"
)


@dataclass
class EvalBenchmarkPreset:
    name: str
    lerobot_repo_id: str
    episode_subset_str: str = ""


def _presets_path() -> str:
    return os.path.abspath(_PRESETS_FILE)


def load_presets() -> List[EvalBenchmarkPreset]:
    """Load all presets from the YAML file. Returns [] if the file doesn't exist."""
    path = _presets_path()
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return [
        EvalBenchmarkPreset(
            name=p["name"],
            lerobot_repo_id=p.get("lerobot_repo_id", ""),
            episode_subset_str=p.get("episode_subset_str", ""),
        )
        for p in data.get("presets", [])
    ]


def save_preset(preset: EvalBenchmarkPreset) -> None:
    """Save or overwrite a preset by name in the YAML file."""
    path = _presets_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    presets = load_presets()
    # Replace if name already exists, otherwise append
    presets_dict: Dict[str, EvalBenchmarkPreset] = {p.name: p for p in presets}
    presets_dict[preset.name] = preset
    data = {
        "presets": [
            {
                "name": p.name,
                "lerobot_repo_id": p.lerobot_repo_id,
                "episode_subset_str": p.episode_subset_str,
            }
            for p in presets_dict.values()
        ]
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
