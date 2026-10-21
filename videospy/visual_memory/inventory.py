"""Join a whole trajectory directory to explicit video partitions and QA metadata.

The join never infers train/test membership from QA parity or model correctness.
It reads dataset metadata only and writes a new, auditable output directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .curation import ensure_new_output, read_jsonl


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("QA identity must be a nonempty string or integer")
    result = str(value).strip()
    if not result:
        raise ValueError("QA identity must not be empty")
    return result


def _question(value):
    return " ".join(str(value or "").split())


def _question_matches(expected, observed):
    expected = str(expected).strip().replace("\r\n", "\n")
    observed = str(observed or "").strip().replace("\r\n", "\n")
    if observed.startswith("Question:"):
        observed = observed[len("Question:"):].lstrip()
    return observed == expected or observed.startswith(expected + "\n")


def _video_alias(value):
    return str(value).replace("\\", "/").rstrip("/")


def _check_trajectory_source(trajectory, allowed):
    evidence = []
    for mapping in (trajectory, trajectory.get("metadata", {}), trajectory.get("memory_context", {})):
        if isinstance(mapping, dict):
            for key in ("source_video_id", "video_id", "video_path", "video_path_original", "video"):
                if isinstance(mapping.get(key), (str, int)) and str(mapping[key]):
                    evidence.append(mapping[key])
    diagnostics = trajectory.get("diagnostics") or {}
    memory = (diagnostics.get("visual_memory") or {}) if isinstance(diagnostics, dict) else {}
    events = (memory.get("inspections") or []) if isinstance(memory, dict) else []
    for event in events:
        context = (event.get("context") or {}) if isinstance(event, dict) else {}
        if isinstance(context, dict) and context.get("source_video_id"):
            evidence.append(context["source_video_id"])
    if any(_video_alias(value) not in allowed for value in evidence):
        raise ValueError("dataset_trajectory_video_identity_mismatch")
    return "explicit_video_identity_matched" if evidence else "qa_id_and_question_only_no_recorded_video_id"


def _path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _dataset_rows(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        for key in ("videos", "data", "samples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            # Some exports use an original-video key as the dictionary key.
            if all(isinstance(item, dict) for item in value.values()):
                value = [{"key": key, **item} for key, item in value.items()]
    if not isinstance(value, list) or not value:
        raise ValueError("Dataset must contain video records with qa arrays, or flat QA records")
    records = {}
    for video in value:
        if not isinstance(video, dict):
            raise ValueError("Dataset records must be objects")
        questions = video.get("qa", video.get("qas"))
        if questions is None:
            questions = [video]
        if not isinstance(questions, list):
            raise ValueError("Dataset qa field must be an array")
        keys = {str(video[key]) for key in ("key", "video_path_original", "video_path", "source_video_id")
                if video.get(key) is not None}
        for qa in questions:
            if not isinstance(qa, dict):
                raise ValueError("Each dataset QA must be an object")
            uid = _identity(qa.get("uid", qa.get("id", qa.get("sample_id"))))
            if uid in records:
                raise ValueError(f"Duplicate QA identity in dataset: {uid}")
            if not keys or not _question(qa.get("question")):
                raise ValueError(f"QA {uid} needs original-video identity and question")
            records[uid] = (video, qa, keys)
    return records


def inventory_sources(trajectories_dir, dataset_path, splits_path, output_dir, *, video_root=None):
    """Create sources.jsonl, inventory.json and skipped.jsonl without VLM calls.

    Each split row has dataset_video_key (an exact dataset video key/path),
    source_video_id (the canonical runtime identity), and split. Rows can supply
    an explicit video_path or frame_paths/fps. Otherwise video_root plus the
    dataset video_path_original/video_path locates media. Missing split rows are
    skipped, never treated as training data. Existing outputs are refused.
    """
    trajectories = Path(trajectories_dir).resolve()
    if not trajectories.is_dir():
        raise ValueError("trajectories must be an existing directory")
    dataset, split_file = Path(dataset_path).resolve(), Path(splits_path).resolve()
    output = ensure_new_output(output_dir)
    qa_records = _dataset_rows(dataset)
    dataset_hash, split_hash = _hash(dataset), _hash(split_file)
    partitions = {}
    canonical_splits, patient_splits, media_splits = {}, {}, {}
    for row in read_jsonl(split_file):
        key = _identity(row.get("dataset_video_key"))
        source = _identity(row.get("source_video_id"))
        partition = row.get("split")
        if partition not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid explicit partition for video {key}")
        if key in partitions:
            raise ValueError(f"Duplicate dataset_video_key in split manifest: {key}")
        previous = canonical_splits.setdefault(source, partition)
        if previous != partition:
            raise ValueError(f"Original video occurs in multiple partitions: {source}")
        # Patient IDs can be dataset-local, so namespace them explicitly when
        # different upstream datasets reuse small integer identifiers.
        if row.get("patient_id") is not None:
            patient = _identity(row["patient_id"])
            previous = patient_splits.setdefault(patient, partition)
            if previous != partition:
                raise ValueError(f"Patient occurs in multiple partitions: {patient}")
        media_keys = []
        if row.get("video_path"):
            media_keys.append(_path(row["video_path"], split_file.parent))
        elif row.get("frame_paths"):
            root = Path(_path(row.get("frame_root", "."), split_file.parent))
            prefix = row.get("frame_path_prefix")
            media_keys.extend(_path(item[len(prefix):].lstrip("/\\") if prefix and item.startswith(prefix) else item, root)
                              for item in row["frame_paths"])
        elif video_root is not None:
            matching_videos = [video for video, _, keys in qa_records.values() if key in keys]
            for video in matching_videos[:1]:
                relative = video.get("video_path_original") or video.get("video_path")
                if relative:
                    media_keys.append(_path(relative, Path(video_root).resolve()))
        for media_key in media_keys:
            previous = media_splits.setdefault(media_key, partition)
            if previous != partition:
                raise ValueError("Same resolved media occurs in multiple partitions under different aliases")
        partitions[key] = {**row, "source_video_id": source}

    accepted, skipped, seen_uids = [], [], set()
    task_counts, split_counts, source_outcomes = Counter(), Counter(), Counter()
    for path in sorted(trajectories.rglob("*.json")):
        try:
            trajectory = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
                raise ValueError("not_a_trajectory")
            uid = _identity(trajectory.get("uid", trajectory.get("sample_id", trajectory.get("id", path.stem))))
            if uid not in qa_records:
                raise ValueError("qa_not_in_dataset")
            video, qa, keys = qa_records[uid]
            mappings = [partitions[key] for key in keys if key in partitions]
            if not mappings:
                raise ValueError("missing_explicit_video_partition")
            if len(mappings) != 1:
                raise ValueError("ambiguous_video_partition_aliases")
            partition = mappings[0]
            if partition["split"] == "test":
                raise ValueError("held_out_test_video")
            if not _question_matches(qa["question"], trajectory.get("question")):
                raise ValueError("dataset_trajectory_question_mismatch")
            if uid in seen_uids:
                raise ValueError("duplicate_qa_trajectory")
            task = qa.get("task_name") or qa.get("qa_type") or qa.get("task_type") or qa.get("task_id")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("missing_task_type")
            source = {"trajectory_path": str(path), "source_video_id": partition["source_video_id"],
                      "source_sample_id": uid, "split": partition["split"], "task_type": task.strip()}
            for key in ("patient_id", "source_id", "procedure", "num_frames", "max_frames"):
                if partition.get(key) is not None:
                    source[key] = str(partition[key]) if key in {"patient_id", "source_id"} else partition[key]
            if partition.get("frame_paths"):
                source["frame_paths"] = partition["frame_paths"]
                source["fps"] = partition["fps"]
                source["frame_root"] = _path(partition.get("frame_root", "."), split_file.parent)
                for key in ("frame_path_prefix", "rc_info"):
                    if partition.get(key) is not None:
                        source[key] = partition[key]
            else:
                if partition.get("video_path"):
                    source["video_path"] = _path(partition["video_path"], split_file.parent)
                elif video_root is not None:
                    relative = video.get("video_path_original") or video.get("video_path")
                    if not isinstance(relative, str) or not relative:
                        raise ValueError("missing_dataset_video_path")
                    source["video_path"] = _path(relative, Path(video_root).resolve())
                else:
                    raise ValueError("missing_media_location")
                if not Path(source["video_path"]).is_file():
                    raise ValueError("missing_video_file")
            # Gold is explicit dataset information, never the model's answer.
            if isinstance(qa.get("answer"), str) and isinstance(qa.get("options"), (list, dict)):
                source.update(gold_option=qa["answer"], options=qa["options"])
            source["inventory_provenance"] = {
                "dataset_sha256": dataset_hash, "split_manifest_sha256": split_hash,
                "trajectory_sha256": _hash(path), "dataset_video_key": partition["dataset_video_key"],
                "qa_id": uid,
            }
            allowed_aliases = {_video_alias(value) for value in keys | {source["source_video_id"]}}
            if source.get("video_path"):
                allowed_aliases.add(_video_alias(source["video_path"]))
            aliases = partition.get("trajectory_video_aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(value, str) for value in aliases):
                raise ValueError("trajectory_video_aliases_must_be_a_string_list")
            allowed_aliases.update(_video_alias(value) for value in aliases)
            source["inventory_provenance"]["video_identity_check"] = _check_trajectory_source(trajectory, allowed_aliases)
            accepted.append(source)
            seen_uids.add(uid)
            task_counts[source["task_type"]] += 1
            split_counts[source["split"]] += 1
            outcome = trajectory.get("correct", trajectory.get("is_correct"))
            source_outcomes["correct" if outcome is True else "incorrect" if outcome is False else "unknown"] += 1
        except (ValueError, TypeError, KeyError, OSError) as error:
            skipped.append({"trajectory_path": str(path), "reason": str(error)[:240]})
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    (output / "skipped.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped), encoding="utf-8")
    report = {"status": "ready" if accepted else "no_eligible_sources", "sources": len(accepted),
              "source_videos": len({row["source_video_id"] for row in accepted}),
              "tasks": dict(task_counts), "partitions": dict(split_counts),
              "trajectory_outcomes": dict(source_outcomes), "skipped": len(skipped),
              "skip_reasons": dict(Counter(row["reason"] for row in skipped)),
              "sources_path": str(output / "sources.jsonl"), "dataset_sha256": dataset_hash,
              "split_manifest_sha256": split_hash,
              "note": "Correctness is reported, never used as an inclusion criterion. No model was called."}
    (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
