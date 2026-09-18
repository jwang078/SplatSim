"""Pre-2026-09-18 name of `splatsim.configs.registry`. Import that instead."""
from splatsim.configs.registry import *  # noqa: F401,F403
from splatsim.configs.registry import (  # noqa: F401
    load_all, get, entry_dir, stage_dir, scene_dir, labels_path, source_file, write_back, invalidate,
    ASSETS_ROOT, STAGES_ROOT, LEGACY_OBJECTS_YAML,
)
