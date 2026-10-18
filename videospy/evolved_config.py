import copy
import hashlib
import re
from itertools import zip_longest
from pathlib import Path

from .evolving.common.schemas import SKILL_PROMPT_CHARS, validate_skill_document

from .evolving.storage.memory_store import EvolutionStateStore, embedding_metadata


HYPERPARAMETER_PATHS = {
    ("agent", "max_steps"),
    ("tools", "overview", "num_frames"),
    ("tools", "overview", "subtitle_context_seconds"),
    ("tools", "clip_skim", "num_frames"),
    ("tools", "clip_skim", "max_duration"),
    ("tools", "clip_skim", "subtitle_context_seconds"),
    ("tools", "frame_inspect", "max_frames"),
    ("tools", "frame_inspect", "subtitle_context_seconds"),
}
NON_NEGATIVE_HYPERPARAMETER_PATHS = {
    ("tools", "overview", "subtitle_context_seconds"),
    ("tools", "clip_skim", "subtitle_context_seconds"),
    ("tools", "frame_inspect", "subtitle_context_seconds"),
}
EVOLUTION_COMPONENTS = ("skill", "memory", "hyperparameter")


def _leaf_paths(value: dict, prefix: tuple[str, ...] = ()):
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Evolution state keys must be strings.")
        path = (*prefix, key)
        if isinstance(item, dict):
            yield from _leaf_paths(item, path)
        else:
            yield path, item


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_hyperparameters(value: dict) -> None:
    invalid = sorted(
        ".".join(path)
        for path, _ in _leaf_paths(value)
        if path not in HYPERPARAMETER_PATHS
    )
    if invalid:
        raise ValueError(
            "Execution hyperparameter snapshots allow only whitelisted fields; "
            f"invalid fields: {invalid}"
        )
    invalid_values = sorted(
        ".".join(path)
        for path, item in _leaf_paths(value)
        if path in HYPERPARAMETER_PATHS
        and (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or (item == 0 and path not in NON_NEGATIVE_HYPERPARAMETER_PATHS)
        )
    )
    if invalid_values:
        raise ValueError(
            "Stored execution hyperparameters must be positive integers, except "
            "subtitle context windows which may be zero; "
            f"invalid fields: {invalid_values}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_components(config: dict) -> dict[str, bool]:
    value = config.get("evolving", {}).get("inference", {})
    if not isinstance(value, dict):
        raise ValueError("evolving.inference must be a mapping.")
    unknown = set(value) - set(EVOLUTION_COMPONENTS)
    if unknown:
        raise ValueError(
            "Unknown evolved inference components: " + ", ".join(sorted(unknown))
        )
    result = {}
    for name in EVOLUTION_COMPONENTS:
        enabled = value.get(name, True)
        if not isinstance(enabled, bool):
            raise ValueError(f"evolving.inference.{name} must be a boolean.")
        result[name] = enabled
    return result


def apply_evolution_state(
    config: dict,
    database_path: str | Path | None,
    version_id: str | None = None,
    components: dict[str, bool] | None = None,
) -> dict:
    """Load one SQLite evolution snapshot over YAML without mutating the database."""
    if database_path is None:
        return copy.deepcopy(config)
    path = Path(database_path).expanduser().resolve()
    store = EvolutionStateStore(path, read_only=True)
    selected_version = store.resolve_version(version_id)
    selected_components = (
        components if components is not None else inference_components(config)
    )
    if set(selected_components) != set(EVOLUTION_COMPONENTS) or any(
        not isinstance(value, bool) for value in selected_components.values()
    ):
        raise ValueError(
            "Evolution components must contain boolean skill, memory, and "
            "hyperparameter values."
        )
    hyperparameters = store.load_hyperparameters(selected_version)
    validate_hyperparameters(hyperparameters)

    result = (
        _merge(config, hyperparameters)
        if selected_components["hyperparameter"]
        else copy.deepcopy(config)
    )
    if selected_components["skill"]:
        result["TASK_SKILLS"] = store.load_skills(selected_version)
    tools = result.setdefault("tools", {})
    memory_config = tools.setdefault("memory_search", {})
    has_memories = store.count_memories(selected_version) > 0
    enabled = tools.setdefault("enabled", [])
    if selected_components["memory"]:
        if has_memories:
            run_metadata = store.run_metadata()
            configured_embedding = embedding_metadata(memory_config)
            expected_embedding = {
                "backend": run_metadata["embedding_backend"],
                "model_name": run_metadata["embedding_model_name"],
                "config_hash": run_metadata["embedding_config_hash"],
            }
            actual_embedding = {
                key: configured_embedding[key] for key in expected_embedding
            }
            if actual_embedding != expected_embedding and run_metadata.get("training_mode") == "offline":
                from .evolving.storage.artifacts import config_identity

                configured_embedding = embedding_metadata(config_identity(memory_config))
                actual_embedding = {
                    key: configured_embedding[key] for key in expected_embedding
                }
            if actual_embedding != expected_embedding:
                raise ValueError(
                    "Memory embedding configuration does not match the "
                    "evolution database."
                )
        memory_config.update(
            {
                "enabled": has_memories,
                "database_path": str(path),
                "version_id": selected_version,
            }
        )
    else:
        memory_config["enabled"] = False
        memory_config.pop("database_path", None)
        memory_config.pop("version_id", None)
    tools["enabled"] = [name for name in enabled if name != "memory_search"]
    result["_evolution"] = {
        "database_path": str(path),
        "database_sha256": _sha256(path),
        "version_id": selected_version,
        "components": selected_components,
        "metadata": store.version_metadata(selected_version),
    }
    return result


def format_evolution_summary(config: dict) -> str:
    evolution = config.get("_evolution")
    if not isinstance(evolution, dict):
        return ""
    path = Path(evolution["database_path"])
    version_id = str(evolution["version_id"])
    components = evolution["components"]
    store = EvolutionStateStore(path, read_only=True)

    lines = [
        "Evolution state loaded for inference:",
        f"  Database: {path}",
        f"  Version: {version_id}",
    ]
    if components["skill"]:
        skills = store.load_skills(version_id)
        task_type_count = 0
        skill_count = 0
        for value in skills.values():
            count = int(bool(value.get("instruction")))
            if count:
                task_type_count += 1
                skill_count += count
        lines.append(
            f"  Skill: enabled, entries={skill_count}, "
            f"task_types={task_type_count}"
        )
    else:
        lines.append("  Skill: disabled")

    if components["memory"]:
        memory_count = store.count_memories(version_id)
        lines.append(
            f"  Memory: enabled, entries={memory_count}, "
            f"search={'active' if memory_count else 'inactive'}"
        )
    else:
        lines.append("  Memory: disabled")

    if components["hyperparameter"]:
        hyperparameters = store.load_hyperparameters(version_id)
        overrides = [
            f"{'.'.join(path)}={value}"
            for path, value in sorted(_leaf_paths(hyperparameters))
        ]
        detail = ", ".join(overrides) if overrides else "none"
        lines.append(
            f"  Hyperparameter: enabled, overrides={len(overrides)} ({detail})"
        )
    else:
        lines.append("  Hyperparameter: disabled")
    return "\n".join(lines)


def task_value(mapping: object, task_type: str):
    if not isinstance(mapping, dict):
        return None
    if task_type in mapping:
        return mapping[task_type]
    for prefix in ("dense_captioning", "region_caption", "video_summary"):
        if task_type.startswith(prefix):
            return mapping.get(prefix)
    return mapping.get("general")


def format_skills(config, task_types):
    mapping = config.get('TASK_SKILLS') or {}
    if not isinstance(mapping, dict):
        raise ValueError('TASK_SKILLS must map task types to document Skills')
    for document in mapping.values():
        validate_skill_document(document)
    groups, identities = [], set()
    for task in dict.fromkeys([*task_types, '__domain__']):
        document = mapping.get(task) if task == '__domain__' else task_value(mapping, task)
        if not document or not document['instruction'].strip():
            continue
        identity = document['instruction'].strip()
        if identity in identities:
            continue
        identities.add(identity)
        # Attach standalone headings to the following paragraph, never injecting an orphan heading.
        blocks, pending = [], []
        for block in re.split(r'\n\s*\n', identity):
            if all(line.lstrip().startswith('#') for line in block.splitlines()):
                pending.append(block)
            else:
                blocks.append('\n\n'.join([*pending, block]))
                pending = []
        groups.append([(task, index, block) for index, block in enumerate(blocks)])
    header = ('# Learned practices\nApply only to matching tasks and evidence. Current observations, '
              'question requirements and tool limits take precedence.\n')
    parts, seen, loaded, skipped = [], set(), [], []
    used = len(header)
    for column in zip_longest(*groups):
        for item in column:
            if item is None:
                continue
            task, index, block = item
            normalized = ' '.join(block.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            part = f"\n{'Domain' if task == '__domain__' else task}:\n{block}\n"
            reference = {'task': task, 'paragraph': index}
            if used + len(part) > SKILL_PROMPT_CHARS:
                skipped.append(reference)
                continue
            parts.append(part)
            used += len(part)
            loaded.append(reference)
    return (header + ''.join(parts) if parts else ''), {
        'paragraphs': loaded, 'budget_skipped': skipped, 'characters': used if parts else 0}


__all__ = [
    "EVOLUTION_COMPONENTS",
    "HYPERPARAMETER_PATHS",
    "apply_evolution_state",
    "format_evolution_summary",
    "format_skills",
    "inference_components",
    "task_value",
    "validate_hyperparameters",
]
