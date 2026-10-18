import copy
import hashlib
import json
import logging
from pathlib import Path

import yaml


def create_run_logger(
    path: Path,
) -> tuple[logging.Logger, tuple[logging.Handler, ...]]:
    logger = logging.getLogger(
        f"videospy.evolving.pipeline.run.{path.parent.name}.{id(path)}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handlers = (file_handler, console_handler)
    for handler in handlers:
        logger.addHandler(handler)
    return logger, handlers


def close_run_logger(
    logger: logging.Logger, handlers: tuple[logging.Handler, ...]
) -> None:
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_yaml(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def redact(value, *, ignore_fields: tuple[str, ...] = ()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                lowered in ignore_fields
                or lowered in {"key", "token", "api_token", "access_token"}
                or lowered.endswith("_key")
                or "secret" in lowered
            ):
                continue
            result[key] = redact(item, ignore_fields=ignore_fields)
        return result
    if isinstance(value, list):
        return [redact(item, ignore_fields=ignore_fields) for item in value]
    return value


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_identity(value):
    """Exclude credentials and transport settings, retaining experiment semantics."""
    return redact(value, ignore_fields=("api_base", "max_retries", "timeout", "request_timeout"))


def hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.get("evolving", {}).pop("inference", None)
    return result
