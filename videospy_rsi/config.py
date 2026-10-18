"""RSI configuration is independent of videospy/evolving and its SQLite state."""

import copy
import math
import os
from pathlib import Path

from .io import read_json

ROOT = Path(__file__).resolve().parents[1]


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(
            result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path=None, backend=None):
    config = read_json(ROOT / "config" / "videospy_rsi" / "default.json")
    if path:
        config = merge(config, read_json(path))
    if backend:
        config["backend"] = backend
    for name, model in config["models"].items():
        for key in ("api_key", "api_base", "model_name"):
            value = model.get(key)
            if isinstance(value, str) and value.startswith("env:"):
                model[key] = os.environ.get(value[4:])
        if config["backend"] != "demo" and not model.get("model_name"):
            raise ValueError(f"Missing model_name for role {name}")
    validate_config(config)
    return config


def validate_config(config):
    if config.get("backend") not in {"demo", "live"}:
        raise ValueError("backend must be demo or live")
    required = {"task", "overview", "clip_skim", "frame_inspect", "assessor", "meta", "outer"}
    if set(config["models"]) != required:
        raise ValueError(f"models must define exactly {sorted(required)}")
    for key in ("max_task_rounds", "meta_every", "max_meta_rounds", "task_candidates",
                "meta_candidates", "episode_rounds", "repair_attempts", "investigations"):
        value = config[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("max_task_rounds", "meta_every", "task_candidates", "meta_candidates", "episode_rounds"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key, value in config["limits"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid resource limit: {key}")
    for key, value in config["selection"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid selection threshold: {key}")


def baseline_config(config):
    # Read only the fixed prompt, never load local credentials from default.yml.
    text = (ROOT / "config" / "videospy" / "default.yml").read_text(encoding="utf-8")
    prompt = text.split("SYSTEM_PROMPT: |", 1)[1]
    prompt = "\n".join(line[2:] if line.startswith("  ") else line
                       for line in prompt.splitlines()).strip()
    model = {"model_name": "rsi-fixed-model", "max_completion_tokens": 4096}
    return {
        "SYSTEM_PROMPT": prompt,
        "agent": {"max_steps": 16, "include_subtitles_in_question": True,
                  "model": {**model, "max_completion_tokens": 1024}},
        "tools": {
            "enabled": ["overview", "clip_skim", "frame_inspect"],
            "overview": {"once_per_task": True, "num_frames": 64, "use_subtitles": True,
                         "subtitle_context_seconds": 5, "model": model},
            "clip_skim": {"num_frames": 16, "max_duration": None, "use_subtitles": True,
                          "subtitle_context_seconds": 5, "model": model},
            "frame_inspect": {"max_frames": 4, "use_subtitles": True,
                              "subtitle_context_seconds": 5, "model": model},
            "memory_search": {"enabled": False},
        },
        "TASK_SKILLS": {},
    }
