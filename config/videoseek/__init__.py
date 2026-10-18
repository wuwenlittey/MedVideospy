import argparse
import os
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
LEGACY_CONFIG_DIR = CONFIG_DIR.parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _general_config_path() -> Path:
    local_path = CONFIG_DIR / "general.yml"
    if local_path.is_file():
        return local_path
    legacy_path = LEGACY_CONFIG_DIR / "general.yml"
    if legacy_path.is_file():
        return legacy_path
    return CONFIG_DIR / "general.example.yml"


def init_config(config: dict, args: argparse.Namespace) -> dict:
    for key, value in vars(args).items():
        if key in config and value is not None:
            config[key] = value
    for key in ("api_base", "api_key", "api_version"):
        value = config.get(key)
        if isinstance(value, str):
            config[key] = os.getenv(value) or value
    return config


general_config = load_config(_general_config_path())
prompts_config = load_config(CONFIG_DIR / "prompts.yml")

__all__ = ["general_config", "init_config", "load_config", "prompts_config"]
