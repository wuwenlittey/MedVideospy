"""Inventory evaluator-labelled trajectories without opening a QA dataset.

Field mappings declare which saved values are standard answers. Predictions and
tool observations are never fallback labels. Explicit video partitions remain
mandatory even when the reference answer is already stored in a trajectory.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

from .curation import _answer, _selection_answer, ensure_new_output, read_jsonl
from .inventory import _hash, _identity, _path, _video_alias


DEFAULTS = {
    "id_fields": ["uid", "sample_id", "id"],
    "video_fields": ["source_video_id", "video_id", "video_path_original", "video_path",
                     "metadata.source_video_id", "metadata.video_id"],
    "question_fields": ["question", "metadata.question"],
    "gold_fields": ["gold_option", "correct_answer", "ground_truth", "gt_answer", "reference_answer",
                    "metadata.gold_option", "metadata.correct_answer", "metadata.ground_truth",
                    "evaluation.correct_answer"],
    "options_fields": ["options", "choices", "metadata.options", "metadata.choices"],
    "task_fields": ["task_type", "task_name", "qa_type", "metadata.task_type"],
    "annotations_fields": ["visual_fact_annotations", "supervision", "metadata.visual_fact_annotations"],
    "gold_format": "letter_or_text",
    "options_from_question": True,
    "default_task_type": "unknown",
}
_PREDICTION_FIELDS = {"final_answer", "prediction", "predicted_answer", "model_answer", "model_response",
                      "messages", "model_calls", "diagnostics", "is_correct", "correct"}


def trajectory_schema(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown trajectory inventory schema settings")
    result = {**copy.deepcopy(DEFAULTS), **value}
    for name in DEFAULTS:
        if name.endswith("_fields"):
            fields = result[name]
            if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields):
                raise ValueError(f"{name} must contain explicit dot-separated field paths")
    if any(set(field.split(".")) & _PREDICTION_FIELDS for field in result["gold_fields"]):
        raise ValueError("gold_fields cannot point to predictions, correctness flags or model messages")
    if result["gold_format"] not in {"letter_or_text", "index_0", "index_1"}:
        raise ValueError("gold_format must be letter_or_text, index_0 or index_1")
    if type(result["options_from_question"]) is not bool:
        raise ValueError("options_from_question must be boolean")
    if not isinstance(result["default_task_type"], str) or not result["default_task_type"].strip():
        raise ValueError("default_task_type must be explicit nonempty text")
    return result


def _values(row, fields):
    found = []
    for field in fields:
        value = row
        for key in field.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            found.append((field, value))
    return found


def _one(row, fields, name, *, default=None):
    found = _values(row, fields)
    if not found:
        return default
    if any(value != found[0][1] for _, value in found[1:]):
        raise ValueError("ambiguous_" + name)
    return found[0][1]


def _question_options(question):
    # Only an unambiguous, contiguous single-line A./B./... option block.
    matches = list(re.finditer(r"(?m)^[ \t]*([A-Z])[.\):：][ \t]+([^\r\n]+)", question))
    if len(matches) < 2 or [m[1] for m in matches] != [chr(65 + i) for i in range(len(matches))]:
        raise ValueError("missing_or_ambiguous_options; provide options_fields")
    if any(question[a.end():b.start()].strip() for a, b in zip(matches, matches[1:])):
        raise ValueError("multiline_options_require_explicit_options_fields")
    remainder = question[matches[-1].end():]
    if remainder.strip() and not re.match(r"\r?\n[ \t]*\r?\n", remainder):
        raise ValueError("ambiguous_last_option; provide options_fields")
    return {match[1]: match[2].strip() for match in matches}


def _gold(row, question, settings):
    found = _values(row, settings["gold_fields"])
    if not found:
        raise ValueError("missing_standard_answer; final_answer is not ground truth")
    options = _one(row, settings["options_fields"], "options")
    options_origin = "explicit_fields"
    if options is None and settings["options_from_question"]:
        try:
            options = _question_options(question)
            options_origin = "question_option_block"
        except ValueError:
            options = None
    normalized, texts = [], []
    for field, value in found:
        letter = None
        if settings["gold_format"] in {"index_0", "index_1"}:
            if type(value) is not int:
                raise ValueError("gold_index_must_be_integer")
            index = value - (settings["gold_format"] == "index_1")
            if not 0 <= index < 26:
                raise ValueError("gold_index_out_of_range")
            letter = chr(65 + index)
        elif isinstance(value, str):
            match = re.fullmatch(r"\s*\(?([A-Z])\)?[.)]?\s*", value)
            if match:
                letter = match[1]
        else:
            raise ValueError("gold_requires_letter_or_text; integer indices need explicit gold_format")
        if letter is not None:
            if options is None:
                raise ValueError("standard_option_has_no_option_text")
            text, _ = _selection_answer({"gold_option": letter, "options": options})
        else:
            text = _answer(value)
            # A full saved reference description also works without MC options.
            if options is not None:
                candidates = {chr(65 + i): item for i, item in enumerate(options)} if isinstance(options, list) else options
                if not isinstance(candidates, dict):
                    raise ValueError("options must be a list or letter mapping")
                letters = [key for key, item in candidates.items() if isinstance(item, str) and item.strip() == text]
                if len(letters) != 1:
                    raise ValueError("standard_text_does_not_uniquely_match_options")
                letter = letters[0]
                _selection_answer({"gold_option": letter, "options": options})
        normalized.append(letter)
        texts.append(text)
    if any(text != texts[0] for text in texts[1:]) or len({letter for letter in normalized if letter}) > 1:
        raise ValueError("conflicting_saved_standard_answers")
    result = {"gold_answer_text": texts[0]}
    letter = next((letter for letter in normalized if letter), None)
    if letter is not None:
        result.update(gold_option=letter, options=options)
    return result, {"fields": [field for field, _ in found], "gold_format": settings["gold_format"],
                    "options_origin": options_origin if options is not None else "none_reference_text_only"}


def inventory_trajectories(trajectories_dir, splits_path, output_dir, *, schema=None, video_root=None):
    settings = trajectory_schema(schema)
    directory, split_file = Path(trajectories_dir).resolve(), Path(splits_path).resolve()
    if not directory.is_dir():
        raise ValueError("trajectories must be an existing directory")
    output = ensure_new_output(output_dir)
    partitions, aliases, sample_map, ownership = {}, {}, {}, {}
    split_hash = _hash(split_file)
    for row in read_jsonl(split_file):
        source_id = _identity(row.get("source_video_id"))
        split = row.get("split")
        if split not in {"train", "dev", "test"} or source_id in partitions:
            raise ValueError("Each original video needs one explicit train/dev/test partition row")
        item = {**row, "source_video_id": source_id}
        media_ids = []
        if row.get("frame_paths"):
            item["frame_root"] = _path(row.get("frame_root", "."), split_file.parent)
            prefix = row.get("frame_path_prefix")
            media_ids = [_path(p[len(prefix):].lstrip("/\\") if prefix and p.startswith(prefix) else p,
                               Path(item["frame_root"])) for p in row["frame_paths"]]
        elif row.get("video_path"):
            item["video_path"] = _path(row["video_path"], Path(video_root).resolve() if video_root else split_file.parent)
            media_ids = [item["video_path"]]
        elif video_root and row.get("dataset_video_key"):
            item["video_path"] = _path(row["dataset_video_key"], Path(video_root).resolve())
            media_ids = [item["video_path"]]
        identities = [("media", value) for value in media_ids]
        if row.get("patient_id") is not None:
            identities.append(("patient", str(row["patient_id"])))
        for identity in identities:
            if ownership.setdefault(identity, split) != split:
                raise ValueError("Patient or resolved media occurs in multiple partitions")
        partitions[source_id] = item
        explicit_aliases = row.get("trajectory_video_aliases", [])
        if not isinstance(explicit_aliases, list) or any(not isinstance(a, str) for a in explicit_aliases):
            raise ValueError("trajectory_video_aliases must be a string list")
        for value in [source_id, row.get("dataset_video_key"), row.get("video_path"), item.get("video_path"), *explicit_aliases]:
            if value is not None:
                alias = _video_alias(value)
                if aliases.setdefault(alias, source_id) != source_id:
                    raise ValueError("Ambiguous original-video alias")
        sample_ids = row.get("sample_ids", [])
        if not isinstance(sample_ids, list):
            raise ValueError("sample_ids must be an explicit list")
        for uid in sample_ids:
            uid = _identity(uid)
            if sample_map.setdefault(uid, source_id) != source_id:
                raise ValueError("QA identity assigned to different original videos")
    sources, skipped, seen, outcomes = [], [], set(), Counter()
    for path in sorted(directory.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError("not_a_trajectory")
            uid = _identity(_one(row, settings["id_fields"], "qa_identity", default=path.stem))
            video_values = _values(row, settings["video_fields"])
            candidates = set()
            for _, value in video_values:
                identity = aliases.get(_video_alias(_identity(value)))
                if identity is None:
                    raise ValueError("unmapped_trajectory_video; add an explicit split alias")
                candidates.add(identity)
            if uid in sample_map:
                candidates.add(sample_map[uid])
            if len(candidates) != 1:
                raise ValueError("missing_or_conflicting_video_partition")
            source_id = candidates.pop()
            partition = partitions[source_id]
            if partition["split"] == "test":
                raise ValueError("held_out_test_video")
            if (source_id, uid) in seen:
                raise ValueError("duplicate_qa_trajectory")
            question = _one(row, settings["question_fields"], "question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing_question")
            gold, origin = _gold(row, question, settings)
            source = {"trajectory_path": str(path), "source_video_id": source_id, "source_sample_id": uid,
                      "question": question, "split": partition["split"], **gold,
                      "task_type": _one(row, settings["task_fields"], "task_type", default=settings["default_task_type"])}
            if not isinstance(source["task_type"], str) or not source["task_type"].strip():
                raise ValueError("invalid_task_type")
            for key in ("video_path", "frame_paths", "frame_root", "fps", "frame_path_prefix", "rc_info",
                        "patient_id", "source_id", "procedure", "num_frames", "max_frames"):
                if partition.get(key) is not None:
                    source[key] = partition[key]
            if bool(source.get("video_path")) == bool(source.get("frame_paths")):
                raise ValueError("split row must supply exactly one video_path or frame_paths plus fps")
            if source.get("video_path") and not Path(source["video_path"]).is_file():
                raise ValueError("missing_video_file")
            annotations = _one(row, settings["annotations_fields"], "annotations")
            if annotations is not None:
                if not isinstance(annotations, list):
                    raise ValueError("saved_annotations_must_be_a_list")
                source["supervision"] = annotations
            source["inventory_provenance"] = {
                "label_origin": "trajectory", "trajectory_sha256": _hash(path), "qa_id": uid,
                "split_manifest_sha256": split_hash, "schema": settings,
                "video_identity_check": "explicit_video_identity_matched" if video_values else "explicit_qa_to_video_mapping",
                "gold": origin,
            }
            json.dumps(source, allow_nan=False)
            sources.append(source)
            seen.add((source_id, uid))
            correct = row.get("correct", row.get("is_correct"))
            outcomes["correct" if correct is True else "incorrect" if correct is False else "unknown"] += 1
        except (ValueError, TypeError, KeyError, OSError) as error:
            skipped.append({"trajectory_path": str(path), "reason": str(error)})
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("sources.jsonl", sources), ("skipped.jsonl", skipped)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    report = {"status": "ready" if sources else "no_eligible_sources", "label_origin": "trajectory",
              "sources": len(sources), "source_videos": len({row["source_video_id"] for row in sources}),
              "tasks": dict(Counter(row["task_type"] for row in sources)), "trajectory_outcomes": dict(outcomes),
              "skip_reasons": dict(Counter(row["reason"] for row in skipped)), "skipped": len(skipped),
              "sources_path": str(output / "sources.jsonl"), "split_manifest_sha256": split_hash,
              "dataset_reads": 0, "vlm_calls": 0, "schema": settings}
    (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
