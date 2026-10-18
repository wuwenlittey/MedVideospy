import argparse
import copy
import os
from pathlib import Path
from typing import Iterable

import yaml

from videospy.evolved_config import (
    apply_evolution_state,
    format_evolution_summary,
    inference_components,
)


MODEL_KEYS = {
    "model_name",
    "api_base",
    "api_key",
    "api_version",
    "seed",
    "temperature",
    "max_retries",
}
CONFIG_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def merge_configs(base: dict, *overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for override in overrides:
        for key, value in override.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _resolve_model_environment(model_config: dict) -> None:
    for key in ("model_name", "api_base", "api_key", "api_version"):
        value = model_config.get(key)
        if isinstance(value, str):
            model_config[key] = os.getenv(value) or value


def init_config(config: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(config)
    agent_config = config["agent"]
    agent_model = agent_config["model"]

    for key in MODEL_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            agent_model[key] = value
    max_completion_tokens = getattr(args, "max_completion_tokens", None)
    if max_completion_tokens is not None:
        agent_model["max_completion_tokens"] = max_completion_tokens
    if getattr(args, "max_steps", None) is not None:
        agent_config["max_steps"] = args.max_steps
    if getattr(args, "enabled_tools", None) is not None:
        config["tools"]["enabled"] = list(args.enabled_tools)

    _resolve_model_environment(agent_model)
    reflector_model = config.get("evolving", {}).get("reflector", {}).get("model")
    if isinstance(reflector_model, dict):
        _resolve_model_environment(reflector_model)
    embedding_model = config["tools"].get("memory_search", {}).get(
        "embedding_model"
    )
    if isinstance(embedding_model, dict):
        _resolve_model_environment(embedding_model)
    for tool_name in config["tools"]["enabled"]:
        tool_config = config["tools"][tool_name]
        if "model" in tool_config:
            _resolve_model_environment(tool_config["model"])
        for nested_config in tool_config.values():
            if isinstance(nested_config, dict) and "model" in nested_config:
                _resolve_model_environment(nested_config["model"])
    return config


def build_config(
    args: argparse.Namespace,
    config_paths: Iterable[Path] = (),
) -> dict:
    config_overrides = [load_config(Path(path).expanduser()) for path in config_paths]
    config = merge_configs(default_config, *config_overrides)
    inference = config.setdefault("evolving", {}).setdefault("inference", {})
    component_arguments = (
        (getattr(args, "evolved_skill", None), "skill"),
        (getattr(args, "evolved_memory", None), "memory"),
        (getattr(args, "evolved_hyperparameter", None), "hyperparameter"),
    )
    evolution_db = getattr(args, "evolution_db", None)
    if (
        any(value is not None for value, _ in component_arguments)
        and not evolution_db
    ):
        raise ValueError("Evolved component switches require --evolution-db.")
    for argument, key in component_arguments:
        if argument is not None:
            inference[key] = argument
    inference_components(config)
    # Resolve environment-backed model fields before validating an evolution
    # database, then apply explicit CLI overrides after SQLite so CLI remains
    # the highest-precedence source.
    config = init_config(config, argparse.Namespace())
    if evolution_db:
        config = apply_evolution_state(
            config,
            evolution_db,
            getattr(args, "evolution_version", None),
        )
        print(format_evolution_summary(config), flush=True)
    return init_config(config, args)


default_config = load_config(CONFIG_DIR / "default.yml")

__all__ = [
    "build_config",
    "default_config",
    "init_config",
    "load_config",
    "merge_configs",
]
