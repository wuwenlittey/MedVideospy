"""Assemble existing source labels and observed keyframes; zero VLM calls.

The labels are supplied dataset annotations or externally verified descriptions.
Successful and unsuccessful QA trajectories follow the exact same path. Missing
labels/alignment become pending records, never labels generated from pixels.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

from .curation import (_object_hash, _open_reader, _resolve, _sample_indices, _select_call,
                       _selection_answer, _sha256, _text, ensure_new_output, read_jsonl)
from .facts import FACT_TYPES, validate_fact_record
from .grounding import _seconds, resolve_annotation
from .inventory import _check_trajectory_source, _question_matches, _video_alias


DEFAULTS = {"max_images_per_fact": 16, "max_facts_per_video_type": 20,
            "max_total_facts": None, "resume": False}


def _settings(value):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown supervised assembly settings; VLM settings are not accepted")
    result = {**DEFAULTS, **value}
    for key in ("max_images_per_fact", "max_facts_per_video_type", "max_total_facts"):
        number = result[key]
        if key == "max_total_facts" and number is None:
            continue
        if type(number) is not int or number < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["max_images_per_fact"] > 16:
        raise ValueError("max_images_per_fact cannot exceed the online reference limit of 16")
    if type(result["resume"]) is not bool:
        raise ValueError("resume must be boolean")
    return result


def _write(path, value):
    if path.is_symlink():
        raise ValueError("output must not contain symlinks")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.is_symlink():
        raise ValueError("output must not contain symlinks")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _key(source):
    return (_text(source.get("source_video_id"), "source_video_id"),
            _text(str(source["source_sample_id"]) if source.get("source_sample_id") is not None else None,
                  "source_sample_id"))


def _media_identities(source, base):
    if source.get("image_paths"):
        return [str(_resolve(path, base)) for path in source["image_paths"]]
    if source.get("video_path"):
        return [str(_resolve(source["video_path"], base))]
    if source.get("frame_paths"):
        root = _resolve(source["frame_root"], base) if source.get("frame_root") else base
        prefix = source.get("frame_path_prefix")
        return [str(_resolve(item[len(prefix):].lstrip("/\\") if prefix and item.startswith(prefix) else item, root))
                for item in source["frame_paths"]]
    return []


def _prepare(source, base):
    _key(source)
    _text(source.get("task_type"), "task_type")
    if source.get("split") not in {"train", "dev"}:
        raise ValueError("held_out_or_missing_partition")
    if source.get("image_paths"):
        if source.get("trajectory_path") or source.get("video_path") or source.get("frame_paths"):
            raise ValueError("Use either an existing image pair or a trajectory with source media")
        paths = source["image_paths"]
        times = source.get("timestamps")
        if not isinstance(paths, list) or not isinstance(times, list) or not paths or len(paths) != len(times):
            raise ValueError("existing pairs require aligned image_paths and timestamps")
        times = [_seconds(t) for t in times]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("existing pair timestamps must be strictly increasing")
        media_paths = [_resolve(path, base) for path in paths]
        question = _text(source.get("question"), "question")
        metadata = {"kind": "existing_annotated_images", "paths": [str(path) for path in media_paths],
                    "sha256": [_sha256(path) for path in media_paths]}
        plan = {"question": question, "paths": media_paths, "times": times, "media": metadata,
                "trajectory": None, "trajectory_path": None}
    else:
        path = _resolve(source.get("trajectory_path"), base)
        trajectory = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
            raise ValueError("source trajectory must contain messages")
        provenance = source.get("inventory_provenance", {})
        saved_schema = None
        if provenance.get("label_origin") == "trajectory":
            from .trajectory_inventory import _one, trajectory_schema

            saved_schema = trajectory_schema(provenance.get("schema"))
            question = _text(_one(trajectory, saved_schema["question_fields"], "question"), "trajectory question")
            uid = _one(trajectory, saved_schema["id_fields"], "qa_identity", default=path.stem)
        else:
            question = _text(trajectory.get("question"), "trajectory question")
            uid = trajectory.get("uid", trajectory.get("sample_id", trajectory.get("id")))
        if source.get("question") and not _question_matches(source["question"], question):
            raise ValueError("dataset_trajectory_question_mismatch")
        if uid is not None and str(uid) != str(source["source_sample_id"]):
            raise ValueError("dataset_trajectory_qa_identity_mismatch")
        trajectory_hash = _sha256(path)
        if provenance.get("trajectory_sha256") and provenance["trajectory_sha256"] != trajectory_hash:
            raise ValueError("trajectory_changed_since_inventory")
        if provenance.get("label_origin") == "trajectory":
            from .trajectory_inventory import _gold

            expected_gold, _ = _gold(trajectory, question, saved_schema)
            if any(source.get(field) != expected_gold.get(field) for field in ("gold_answer_text", "gold_option", "options")):
                raise ValueError("saved_standard_answer_changed_since_inventory")
        if not provenance.get("video_identity_check"):
            aliases = {_video_alias(source["source_video_id"])}
            if source.get("video_path"):
                aliases.add(_video_alias(source["video_path"]))
                aliases.add(_video_alias(_resolve(source["video_path"], base)))
            aliases.update(_video_alias(value) for value in source.get("trajectory_video_aliases", []))
            _check_trajectory_source(trajectory, aliases)
        if bool(source.get("video_path")) == bool(source.get("frame_paths")):
            raise ValueError("trajectory needs exactly one video_path or frame_paths with fps")
        if source.get("video_path"):
            media_paths = [_resolve(source["video_path"], base)]
        else:
            frame_root = _resolve(source["frame_root"], base) if source.get("frame_root") else base
            prefix = source.get("frame_path_prefix")
            media_paths = [_resolve(item[len(prefix):].lstrip("/\\") if prefix and item.startswith(prefix) else item,
                                    frame_root) for item in source["frame_paths"]]
        metadata = [{"path": str(item), "size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
                    for item in media_paths]
        plan = {"question": question, "trajectory": trajectory, "trajectory_path": str(path),
                "trajectory_sha256": trajectory_hash, "media": metadata}
    plan["media_paths"] = [str(item) for item in media_paths]
    plan["signature"] = _object_hash({"source": source, "media": plan["media"],
                                     "trajectory_sha256": plan.get("trajectory_sha256")})
    return plan


def _observed_frames(source, plan, base, reader_factory):
    if plan["trajectory"] is None:
        frames = [{"index": i, "timestamp": t, "calls": []} for i, t in enumerate(plan["times"])]
        return frames, None, [], 0.0
    reader, _ = reader_factory(source, base)
    frames, calls = {}, []
    for index, message in enumerate(plan["trajectory"]["messages"]):
        if message.get("role") != "tool":
            continue
        try:
            _, result, call, tool, arguments = _select_call(plan["trajectory"], {"tool_message_index": index})
            indices, times, sampling = _sample_indices(reader, tool, arguments, result, source)
            audit = {"tool_message_index": index, "tool_call_id": call["id"], "tool": tool,
                     "sampling": sampling, "status": "usable"}
            for source_index, time in zip(indices, times):
                frame = frames.setdefault(source_index, {"index": source_index, "timestamp": time, "calls": []})
                frame["calls"].append(index)
        except (ValueError, TypeError, KeyError) as error:
            audit = {"tool_message_index": index, "status": "skipped", "reason": str(error)}
        calls.append(audit)
    return [frames[index] for index in sorted(frames)], reader, calls, min(0.5 / float(reader.get_avg_fps()), 0.5)


def _select_frames(frames, annotation, tolerance):
    alignment = annotation["alignment"]
    if alignment["kind"] == "interval":
        selected = [frame for frame in frames if alignment["start"] <= frame["timestamp"] <= alignment["end"]]
    else:
        selected, used = [], set()
        for timestamp in alignment["timestamps"]:
            matches = [frame for frame in frames if abs(frame["timestamp"] - timestamp) <= tolerance + 1e-9]
            if not matches:
                raise ValueError("annotated_keyframe_not_observed")
            frame = min(matches, key=lambda f: (abs(f["timestamp"] - timestamp), f["index"]))
            if frame["index"] in used:
                raise ValueError("annotated_keyframes_collapse_to_same_video_frame")
            selected.append(frame)
            used.add(frame["index"])
    if not selected:
        raise ValueError("no_observed_frames_in_annotated_evidence")
    if len(selected) < 2 and (annotation["fact_type"] == "action" or alignment["scope"] != "frame_set"):
        raise ValueError("insufficient_observed_frames_for_clip_label")
    return selected


def _balance(records, settings):
    groups, seen, duplicate, excluded = defaultdict(lambda: defaultdict(list)), {}, [], []
    for record in sorted(records, key=lambda row: row["case_id"]):
        # Keep scope/conditions/procedure in the key; similar text alone is not equivalence.
        key = _object_hash([record["fact"]["type"], record["answer_text"], record.get("procedure"),
                            record["grounding"]["alignment"]["scope"], record["provenance"]["support_image_sha256"]])
        if key in seen:
            duplicate.append({"case_id": record["case_id"], "kept_case_id": seen[key]})
            continue
        seen[key] = record["case_id"]
        groups[record["fact"]["type"]][record["source_video_id"]].append(record)
    queues = {}
    for kind, videos in groups.items():
        per_video = {}
        for video, rows in videos.items():
            limit = settings["max_facts_per_video_type"]
            excluded.extend(row["case_id"] for row in rows[limit:])
            per_video[video] = deque(rows[:limit])
        queues[kind] = deque()
        while any(per_video.values()):
            for video in sorted(per_video):
                if per_video[video]:
                    queues[kind].append(per_video[video].popleft())
    selected = []
    maximum = settings["max_total_facts"]
    while any(queues.values()) and (maximum is None or len(selected) < maximum):
        for kind in FACT_TYPES:
            if queues.get(kind) and (maximum is None or len(selected) < maximum):
                selected.append(queues[kind].popleft())
    excluded.extend(row["case_id"] for queue in queues.values() for row in queue)
    return selected, duplicate, excluded


def assemble_facts(sources_path, output_dir, config=None, *, annotations_path=None, reader_factory=None):
    """Batch source-grounded assembly; optional annotations join by video AND QA ID.

    Source rows can hold ``supervision`` lists directly, or get them from a
    sidecar. Existing annotated image/text pairs use image_paths/timestamps;
    trajectories use trajectory_path plus original video/frame sequence.
    """
    settings = _settings(config)
    sources_path = Path(sources_path).resolve()
    sources = copy.deepcopy(read_jsonl(sources_path))
    json.dumps(sources, allow_nan=False)  # Reject non-JSON numbers before reserving output.
    keys = [_key(source) for source in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate original-video/QA source identity")
    pending, external = [], defaultdict(list)
    if annotations_path is not None:
        annotations_path = Path(annotations_path).resolve()
        sidecar = read_jsonl(annotations_path)
        json.dumps(sidecar, allow_nan=False)
        for annotation in sidecar:
            key = _key(annotation)
            if key not in keys:
                pending.append({"status": "pending", "reason": "annotation_has_no_matching_source",
                                "source_video_id": key[0], "source_sample_id": key[1], "annotation": annotation})
            else:
                external[key].append(annotation)
    for source, key in zip(sources, keys):
        labels = source.get("supervision", [])
        if not isinstance(labels, list):
            raise ValueError("source.supervision must be a list")
        source["supervision"] = labels + external[key]
        ids = [_text(a.get("annotation_id"), "annotation_id") for a in source["supervision"] if isinstance(a, dict)]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate annotation_id within one source")
    # Explicit partition conflicts invalidate the run, including aliases to the same media.
    partitions = {}
    plans = []
    for source, key in zip(sources, keys):
        identities = [("video", key[0])]
        if source.get("patient_id") is not None:
            identities.append(("patient", str(source["patient_id"])))
        # Register held-out media too, before excluding its source from assembly.
        identities.extend(("media", path) for path in _media_identities(source, sources_path.parent))
        for identity in identities:
            previous = partitions.setdefault(identity, source.get("split"))
            if previous != source.get("split"):
                raise ValueError("original video/patient/media occurs in multiple partitions")
        try:
            plan = _prepare(source, sources_path.parent)
        except (ValueError, TypeError, KeyError, OSError) as error:
            pending.append({"status": "invalid_source", "source_video_id": key[0], "source_sample_id": key[1],
                            "reason": str(error)})
            plans.append(None)
            continue
        for path in plan["media_paths"]:
            identity = ("media", path)
            previous = partitions.setdefault(identity, source["split"])
            if previous != source["split"]:
                raise ValueError("same resolved media occurs in multiple partitions")
        plans.append(plan)
    binding = {"version": 1, "pipeline": "source_grounded_no_vlm", "sources_sha256": _sha256(sources_path),
               "annotations_sha256": _sha256(annotations_path) if annotations_path else None,
               "source_signatures": [plan["signature"] if plan else None for plan in plans],
               "settings": {key: value for key, value in settings.items() if key != "resume"}}
    output = Path(output_dir).absolute()
    if output.exists():
        if not settings["resume"]:
            raise FileExistsError("Assembly output exists; use a new directory or --resume")
        data = Path(__file__).resolve().parents[2] / "data"
        if output.is_symlink() or output.resolve().is_relative_to(data.resolve()):
            raise ValueError("resume requires an output outside data and without symlinks")
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("resume output must not contain symlinks")
        if not (output / "run.json").is_file() or json.loads((output / "run.json").read_text(encoding="utf-8")) != binding:
            raise ValueError("Resume refused: source, labels, alignment, media or configuration changed")
    else:
        output = ensure_new_output(output)
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "run.json", binding)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    records, observations, outcomes = [], [], Counter()
    for source, key, plan in zip(sources, keys, plans):
        if plan is None:
            continue
        prefix = {"source_video_id": key[0], "source_sample_id": key[1], "task_type": source["task_type"]}
        if plan["trajectory"] is not None:
            correct = plan["trajectory"].get("correct", plan["trajectory"].get("is_correct"))
            outcomes["correct" if correct is True else "incorrect" if correct is False else "unknown"] += 1
        try:
            frames, reader, calls, tolerance = _observed_frames(source, plan, sources_path.parent, reader_factory or _open_reader)
        except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
            pending.append({**prefix, "status": "pending", "reason": "cannot_reconstruct_observed_frames", "detail": str(error)})
            continue
        observations.append({**prefix, "observed_timestamps": [frame["timestamp"] for frame in frames], "calls": calls})
        if not source["supervision"]:
            candidate = source.get("gold_answer_text")
            if source.get("gold_option"):
                try:
                    candidate, _ = _selection_answer({"gold_option": source["gold_option"], "options": source.get("options")})
                except ValueError:
                    pass
            pending.append({**prefix, "status": "pending", "reason": "missing_explicit_label_alignment",
                            "dataset_label_candidate": candidate if source.get("inventory_provenance", {}).get("label_origin") != "trajectory" else None,
                            "source_label_candidate": candidate, "observed_timestamps": [f["timestamp"] for f in frames],
                            "note": "Supply the local fact type and an external evidence annotation; QA correctness alone is insufficient."})
        for raw_annotation in source["supervision"]:
            try:
                annotation = resolve_annotation(raw_annotation, source)
                selected = _select_frames(frames, annotation, tolerance)
                if len(selected) > settings["max_images_per_fact"]:
                    raise ValueError("annotated_evidence_exceeds_image_budget; supply explicit annotated keyframes")
                case_id = "source_fact_" + _object_hash([key, annotation, [f["index"] for f in selected]])[:24]
                paths, hashes = [], []
                image_dir = output / "images" / case_id
                image_dir.mkdir(exist_ok=True)
                if reader is None:
                    arrays = []
                    for frame in selected:
                        with Image.open(plan["paths"][frame["index"]]) as original:
                            arrays.append(np.asarray(original.convert("RGB")))
                else:
                    arrays = reader.get_batch([frame["index"] for frame in selected]).asnumpy()
                for i, array in enumerate(arrays):
                    path = image_dir / f"frame_{i:03d}.png"
                    with Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB") as image:
                        if path.exists():
                            with Image.open(path) as existing:
                                if existing.size != image.size or existing.convert("RGB").tobytes() != image.tobytes():
                                    raise ValueError("checkpoint_evidence_changed")
                        else:
                            image.save(path, format="PNG")
                    paths.append(path.relative_to(output).as_posix())
                    hashes.append(_sha256(path))
                ids = list(range(len(selected)))
                text = annotation["label_source"]["text"]
                record = {"record_kind": "visual_fact", "case_id": case_id,
                          "question": plan["question"], "answer_text": text, **prefix, "split": source["split"],
                          "image_paths": paths, "timestamps": [frame["timestamp"] for frame in selected],
                          "evidence_scope": "frame" if len(paths) == 1 else "clip", "verified": False,
                          "verification_note": "Existing source label with explicit external alignment; no VLM extraction or review.",
                          "fact": {"type": annotation["fact_type"], "text": text, "entity": text,
                                   "basis": "source_annotation", "conditions": [], "evidence_frame_ids": ids},
                          "grounding": {**annotation, "annotation_sha256": _object_hash(annotation),
                                        "timestamp_tolerance": tolerance},
                          "verification": {"status": "source_grounded", "human_verified": False,
                                           "decision": "supported_by_source_annotation", "evidence_frame_ids": ids,
                                           "reason": "Label and temporal scope supplied externally; no new medical verification."},
                          "provenance": {"source_manifest_sha256": binding["sources_sha256"],
                                         "annotations_sha256": binding["annotations_sha256"],
                                         "source_signature": plan["signature"],
                                         "trajectory_path": plan["trajectory_path"],
                                         "trajectory_sha256": plan.get("trajectory_sha256"),
                                         "inventory": source.get("inventory_provenance"),
                                         "selected_frames": selected, "observations": calls,
                                         "support_image_sha256": hashes, "vlm_calls": 0}}
                for field in ("procedure", "patient_id", "source_id"):
                    if source.get(field) is not None:
                        record[field] = str(source[field])
                record = validate_fact_record(record)
                checkpoint = output / "checkpoints" / f"{case_id}.json"
                if checkpoint.exists():
                    if json.loads(checkpoint.read_text(encoding="utf-8")) != record:
                        raise ValueError("checkpoint_record_changed")
                else:
                    _write(checkpoint, record)
                records.append(record)
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                if str(error).startswith("checkpoint_"):
                    raise
                pending.append({**prefix, "status": "pending", "reason": str(error), "annotation": raw_annotation})
        # Release native video readers before opening the next video.
        del reader
    selected, duplicates, excluded = _balance(records, settings)
    for name, rows in (("cases.jsonl", selected), ("pending.jsonl", pending)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    by_type = Counter(row["fact"]["type"] for row in selected)
    report = {"pipeline": "source_grounded_no_vlm", "status": "completed_with_pending" if pending else "completed",
              "cases_path": str(output / "cases.jsonl"), "pending_path": str(output / "pending.jsonl"),
              "report_path": str(output / "report.json"), "sources": len(sources),
              "selected_fact_count": len(selected), "source_grounded_before_selection": len(records),
              "facts_by_type": dict(by_type), "missing_fact_types": [kind for kind in FACT_TYPES if not by_type[kind]],
              "facts_by_task": dict(Counter(row["task_type"] for row in selected)),
              "facts_by_video": dict(Counter(row["source_video_id"] for row in selected)),
              "labels_by_source": dict(Counter(row["grounding"]["label_source"]["kind"] for row in selected)),
              "trajectory_outcomes": dict(outcomes), "pending_count": len(pending),
              "pending_reasons": dict(Counter(row["reason"] for row in pending)),
              "duplicate_facts": duplicates, "quota_excluded_fact_ids": excluded,
              "observations": observations, "vlm_calls": 0,
              "note": "Labels are copied from supplied annotations. External label/alignment correctness remains the source's responsibility."}
    _write(output / "report.json", report)
    return report
