"""Load and validate training manifests, source runs, and trajectories."""
import copy
import hashlib
import json
import logging
import re
from pathlib import Path

from videospy.evolved_config import HYPERPARAMETER_PATHS
from ..storage.artifacts import redact
from .memory_scope import egolife_source_id, medvidbench_source_context


def _nested_value(value: dict, path: tuple[str, ...]):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def initial_hyperparameters(config: dict) -> dict:
    result: dict = {}
    for path in HYPERPARAMETER_PATHS:
        value = _nested_value(config, path)
        if value is None:
            continue
        current = result
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value
    return result


def latest_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark records not found: {path}")
    records: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "sample_key" not in record:
                    raise ValueError("record requires sample_key")
            except ValueError as error:
                logging.getLogger(__name__).warning("Skipping invalid record %s:%d: %s", path, number, error)
                continue
            records[str(record["sample_key"])] = record
    return list(records.values())


def records_for_samples(records: list[dict], samples: list) -> list[dict]:
    by_key = {str(record["sample_key"]): record for record in records}
    selected = []
    seen = set()
    for sample in samples:
        key = str(sample)
        if key not in seen:
            selected.append(by_key[key])
            seen.add(key)
    return selected


def manifest_values(path: Path, benchmark: str) -> list:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        keys = {
            "lvbench": ("sample_ids", "uids", "full"),
            "egolife": ("sample_ids", "question_ids", "full"),
            "medvidbench": ("sample_indices", "indices", "full"),
        }[benchmark]
        value = next((value[key] for key in keys if key in value), None)
    if not isinstance(value, list) or not value:
        raise ValueError("Evolution train manifest must contain a non-empty list.")
    identities = [str(item) for item in value]
    if len(identities) != len(set(identities)):
        raise ValueError("Evolution train manifest must not contain duplicates.")
    return value


def load_trajectory(run_dir: Path, record: dict) -> dict:
    relative = record.get("trajectory")
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"Offline sample is missing a trajectory: {record.get('sample_key')}"
        )
    root = run_dir.resolve()
    path = (run_dir / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Invalid offline trajectory: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Offline trajectory must be a mapping: {path}")
    return value


def load_medvidbench_sources(run_dir, manifest, records):
    """Recover clip scope from records or the dataset recorded by the source run."""
    logger = logging.getLogger(__name__)
    samples = []
    if any(not record.get("memory_source") for record in records) and manifest.get("metadata_path"):
        try:
            samples = json.loads(Path(manifest["metadata_path"]).expanduser().read_text())
        except (OSError, ValueError) as error:
            logger.warning("Cannot read MedVidBench clip metadata: %s", error)
    sources = {}
    for record in records:
        sid = str(record["sample_key"])
        if record.get("memory_source"):
            sources[sid] = record["memory_source"]
            continue
        try:
            sample = samples[int(sid)]
            if str(sample["metadata"].get("video_id", "")) != str(record.get("video_id", "")):
                raise ValueError("Source video differs from dataset metadata")
            sources[sid] = medvidbench_source_context(sample)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            logger.warning("MedVidBench clip scope unavailable for sample %s; keeping trajectory-local scope: %s", sid, error)
            # Unknown clips must never be merged just because their video IDs match.
            sources[sid] = {"benchmark": "medvidbench", "video": record.get("video_id", ""),
                            "source_id": f"medvidbench:trajectory:{run_dir.resolve()}:{sid}",
                            "time_basis": "seconds from this trajectory's frame sequence; original clip unknown"}
    return sources


def load_egolife_timeline(run_dir, manifest):
    """Recover legacy coordinates only when every recorded query boundary matches."""
    if manifest.get('timeline_mapping'):
        return manifest['timeline_mapping'], {'status': 'recorded'}
    try:
        if egolife_source_id(manifest.get('video_root')) != 'egolife:A1_JAKE':
            raise ValueError('Unknown source timeline')
        records = latest_records(run_dir / 'records.jsonl')
        by_key = {str(r['sample_key']): r for r in records}
        boundaries = []
        for key in manifest.get('sample_keys', []):
            record = by_key[str(key)]
            if record.get('status') != 'success':
                continue
            trajectory = load_trajectory(run_dir, record)
            user = next((m.get('content', '') for m in trajectory.get('messages', [])
                         if m.get('role') == 'user'), '')
            match = re.search(r'^Video Duration:\s*([0-9.]+)s', user)
            if not match:
                raise ValueError(f'Sample {key} has no recorded video duration')
            boundaries.append((str(key), record['query_time'], float(match[1])))
        if not boundaries:
            raise ValueError('No recorded query boundaries to validate')
        # Use the recorded dataset when present, otherwise the same local default
        # as evaluation. Source identity and all query boundaries must still agree.
        from benchmark.egolife.run import DEFAULT_VIDEO_ROOT
        from benchmark.egolife.timeline import EgoLifeTimelineReader
        root = Path(manifest['video_root']).expanduser()
        if not root.is_dir():
            root = DEFAULT_VIDEO_ROOT
        timeline = EgoLifeTimelineReader(root)
        for key, query_time, recorded_duration in boundaries:
            timeline.set_query_time(query_time)
            current_duration = round(round(timeline.duration, 2), 1)
            if current_duration != recorded_duration:
                raise ValueError(f'Sample {key} timeline mismatch: recorded={recorded_duration}, local={current_duration}')
        return timeline.time_mapping, {
            'status': 'reconstructed', 'video_root': str(root.resolve()),
            'checked_samples': len(boundaries), 'timeline_id': timeline.time_mapping['id'],
            'validation': 'Source identity and every successful recorded query boundary; video bytes not verified.'}
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        logging.getLogger(__name__).warning('EgoLife source timeline remains unavailable: %s', error)
        return None, {'status': 'unavailable', 'reason': str(error)}


_MISSING_CONFIG_VALUE = object()
_IMPORTANT_CONFIG_FIELDS = {
    "enable_thinking",
    "max_completion_tokens",
    "max_duration",
    "max_frames",
    "max_steps",
    "model_name",
    "num_frames",
    "seed",
    "subtitle_context_seconds",
    "temperature",
    "use_subtitles",
}


def _inference_config(config: dict, benchmark: str) -> dict:
    result = {
        key: copy.deepcopy(config.get(key))
        for key in ("SYSTEM_PROMPT", "TASK_SKILLS", "agent", "tools")
    }
    if benchmark == "egolife":
        result.setdefault("agent", {})["include_subtitles_in_question"] = False
    result.setdefault("tools", {}).pop("memory_search", None)
    return redact(result)


def _important_config_path(path: str) -> bool:
    if path == "SYSTEM_PROMPT" or path.startswith("TASK_SKILLS"):
        return True
    if path == "tools.enabled" or path.startswith("tools.enabled["):
        return True
    return path.rsplit(".", 1)[-1] in _IMPORTANT_CONFIG_FIELDS


def _config_value_summary(value) -> str:
    if value is _MISSING_CONFIG_VALUE:
        return "<missing>"
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if len(encoded) <= 160:
        return encoded
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"<sha256={digest} length={len(encoded)}>"


def _flatten_config(value, path: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        flattened: dict[str, object] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            flattened.update(_flatten_config(item, child_path))
        return flattened
    return {path: value}


def _config_differences(offline: dict, current: dict) -> list[dict[str, object]]:
    offline_values = _flatten_config(offline)
    current_values = _flatten_config(current)
    differences = []
    for path in set(offline_values) | set(current_values):
        offline_value = offline_values.get(path, _MISSING_CONFIG_VALUE)
        current_value = current_values.get(path, _MISSING_CONFIG_VALUE)
        if offline_value == current_value:
            continue
        differences.append(
            {
                "path": path,
                "offline": _config_value_summary(offline_value),
                "current": _config_value_summary(current_value),
                "important": _important_config_path(path),
            }
        )
    return sorted(
        differences,
        key=lambda difference: (
            not bool(difference["important"]),
            str(difference["path"]),
        ),
    )


def format_config_difference_warning(
    differences: list[dict[str, object]],
) -> str:
    lines = [
        "Offline inference Agent configuration has "
        f"{len(differences)} difference(s); continuing with existing trajectories."
    ]
    for difference in differences:
        prefix = "[IMPORTANT] " if difference["important"] else ""
        lines.append(
            f"{prefix}{difference['path']}: offline={difference['offline']}, "
            f"current={difference['current']}"
        )
    return "\n".join(lines)


def _hash_source_files(files: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in files:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_inference_source(
    source: Path,
    benchmark: str,
    samples: list,
    current_config: dict,
) -> tuple[dict, list[dict], list[dict[str, object]]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Offline inference run not found: {source}")
    run_path = source / "run.json"
    records_path = source / "records.jsonl"
    if not run_path.is_file():
        raise FileNotFoundError(f"Offline run manifest not found: {run_path}")
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    expected_fields = {"dataset": benchmark, "agent": "videospy", "mode": "full"}
    mismatches = [
        key for key, value in expected_fields.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Offline inference run does not match evolution training: "
            + ", ".join(mismatches)
        )
    expected_keys = [str(sample) for sample in samples]
    source_keys = [str(key) for key in manifest.get("sample_keys", [])]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Offline inference run contains duplicate sample keys.")
    positions = {key: index for index, key in enumerate(source_keys)}
    missing_keys = [key for key in expected_keys if key not in positions]
    if missing_keys:
        raise ValueError(
            "Offline inference run does not contain training manifest sample keys: "
            + ", ".join(missing_keys)
        )
    if [positions[key] for key in expected_keys] != sorted(
        positions[key] for key in expected_keys
    ):
        raise ValueError(
            "Offline training manifest sample keys must preserve inference run order."
        )
    agent_config = manifest.get("agent_config")
    if not isinstance(agent_config, dict):
        raise ValueError("Offline run.json is missing agent_config.")
    if "_evolution" in agent_config:
        raise ValueError(
            "Offline evolution training requires base VideoSpy inference "
            "without an existing evolution database."
        )
    source_tools = agent_config.get("tools") or {}
    source_memory = source_tools.get("memory_search") or {}
    if source_memory.get("enabled") or "memory_search" in (
        source_tools.get("enabled") or []
    ):
        raise ValueError(
            "Offline evolution training requires base trajectories without "
            "Memory Search."
        )
    config_differences = _config_differences(
        _inference_config(agent_config, benchmark),
        _inference_config(current_config, benchmark),
    )
    records = latest_records(records_path)
    by_key = {str(record["sample_key"]): record for record in records}
    trajectory_files = []
    valid_keys, skipped_keys = [], []
    failures = []
    for key in expected_keys:
        record = by_key.get(key)
        try:
            if record is None:
                raise ValueError(f"Offline inference sample is not complete: {key}")
            if record.get("status") != "success":
                raise ValueError("source inference failed")
            load_trajectory(source, record)
            trajectory_path = (source / record["trajectory"]).resolve()
        except (OSError, ValueError) as error:
            logging.getLogger(__name__).warning("Skipping invalid source sample %s: %s", key, error)
            skipped_keys.append(key)
            failures.append({"stage": "source", "node": key, "reason": str(error)})
            continue
        trajectory_files.append((f"trajectory:{key}", trajectory_path))
        valid_keys.append(key)
    if not valid_keys:
        raise ValueError("Offline inference has no successful training samples after skipping failed records.")
    files = [("run.json", run_path), ("records.jsonl", records_path)]
    files.extend(trajectory_files)
    source_metadata = {
        "path": str(source),
        "dataset": benchmark,
        "sample_keys": valid_keys,
        "source_sample_count": len(source_keys),
        "content_sha256": _hash_source_files(files),
    }
    if skipped_keys:
        source_metadata["skipped_sample_keys"] = skipped_keys
        source_metadata["failures"] = failures
    return source_metadata, [by_key[key] for key in valid_keys], config_differences
