"""Export explicitly selected trajectory observations for human review.

Question-level ground truth is deliberately kept at case level. This module does
not label individual frames, invent visual descriptions, or approve its exports.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from videospy.utils import parse_video_timestamp, validate_time_range


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _answer(value) -> str:
    answer = _text(value, "answer_text")
    if re.fullmatch(r"(?:answer\s*[:：]\s*)?[\s(<\[]*[A-D][\s.)>\]]*", answer, re.I):
        raise ValueError("answer_text must contain the semantic GT answer, not only an option letter.")
    return answer


def _selection_answer(selection: dict):
    """Resolve explicit GT only; never read the trajectory's prediction."""
    direct = _answer(selection["answer_text"]) if "answer_text" in selection else None
    if "gold_option" not in selection:
        if direct is None:
            raise ValueError("Supply answer_text or explicit gold_option plus options.")
        return direct, {"kind": "explicit_selection_answer_text"}
    gold = _text(selection["gold_option"], "gold_option").upper()
    if not re.fullmatch(r"[A-Z]", gold):
        raise ValueError("gold_option must be one explicit option letter, such as C.")
    options = selection.get("options")
    if isinstance(options, list):
        index = ord(gold) - ord("A")
        if not 0 <= index < len(options):
            raise ValueError("gold_option is outside the options list.")
        semantic = _answer(options[index])
    elif isinstance(options, dict):
        if gold not in options:
            raise ValueError("gold_option must be a key in the explicit options mapping.")
        semantic = _answer(options[gold])
    else:
        raise ValueError("gold_option requires an explicit options mapping or ordered list.")
    if direct is not None and direct != semantic:
        raise ValueError("answer_text disagrees with the explicitly selected gold option text.")
    return semantic, {"kind": "explicit_gold_option_and_options", "gold_option": gold,
                      "options_sha256": _object_hash(options), "answer_text_also_provided": direct is not None}


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object.")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no records.")
    return records


def _resolve(value, base: Path) -> Path:
    path = Path(_text(value, "source path")).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def ensure_new_output(path: str | Path) -> Path:
    """Protect repository data and refuse every existing output, including symlinks."""
    raw = Path(path).absolute()
    resolved = raw.resolve()
    data = Path(__file__).resolve().parents[2] / "data"
    if raw.is_relative_to(data) or resolved.is_relative_to(data.resolve()):
        raise ValueError("Repository data directory is read-only; choose an output outside data/.")
    if raw.exists() or raw.is_symlink() or resolved.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {resolved}")
    return resolved


class _Batch:
    def __init__(self, images):
        self.images = images

    def asnumpy(self):
        return np.stack(self.images)


class _ImageSequence:
    """Read-only image sequence; input order is the video timeline."""

    def __init__(self, paths: list[Path], fps: float, rc_info=None, original_paths=None):
        if not paths or not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame_paths must be non-empty and fps must be finite and positive.")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Source frame not found: {path}")
        self.paths, self.fps = paths, fps
        self.rc_index, self.rc_bbox = None, None
        if rc_info:
            start = rc_info.get("start_frame")
            if start in (original_paths or []):
                self.rc_index = original_paths.index(start)
                self.rc_bbox = rc_info.get("start_frame_bbox")

    def __len__(self):
        return len(self.paths)

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        images = []
        for index in indices:
            with Image.open(self.paths[int(index)]) as source:
                image = source.convert("RGB")
                if index == self.rc_index and self.rc_bbox:
                    ImageDraw.Draw(image).rectangle(self.rc_bbox, outline=(0, 255, 0), width=8)
                images.append(np.asarray(image))
        return _Batch(images)


def _open_reader(selection: dict, base: Path):
    has_video = bool(selection.get("video_path"))
    has_frames = bool(selection.get("frame_paths"))
    if has_video == has_frames:
        raise ValueError("Provide exactly one of video_path or frame_paths (plus fps).")
    if has_video:
        path = _resolve(selection["video_path"], base)
        if not path.is_file():
            raise FileNotFoundError(f"Source video not found: {path}")
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        stat = path.stat()
        return reader, {"kind": "video", "path": str(path), "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns}
    originals = selection["frame_paths"]
    if not isinstance(originals, list) or not originals:
        raise ValueError("frame_paths must be a non-empty ordered list.")
    root = _resolve(selection["frame_root"], base) if selection.get("frame_root") else base
    prefix = selection.get("frame_path_prefix")
    if prefix is not None:
        prefix = _text(prefix, "frame_path_prefix")
        if not selection.get("frame_root"):
            raise ValueError("frame_path_prefix remapping requires an explicit frame_root.")
    paths = []
    for original in originals:
        original = _text(original, "frame_paths item")
        if prefix and original.startswith(prefix):
            original = original[len(prefix):].lstrip("/\\")
        paths.append(_resolve(original, root))
    fps = selection.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise ValueError("Image sequences require an explicit numeric fps.")
    return _ImageSequence(paths, fps, selection.get("rc_info"), originals), {
        "kind": "image_sequence", "frame_root": str(root), "fps": fps,
        "frame_path_prefix": prefix, "rc_info": selection.get("rc_info"),
        "ordered_paths_sha256": _object_hash([str(path) for path in paths]),
    }


def _select_call(trajectory: dict, selection: dict):
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Trajectory must contain a messages list.")
    by_id = "tool_call_id" in selection
    by_index = "tool_message_index" in selection
    if by_id == by_index:
        raise ValueError("Select exactly one of tool_call_id or tool_message_index (zero-based).")
    if by_id:
        call_id = _text(selection["tool_call_id"], "tool_call_id")
        matches = [(i, m) for i, m in enumerate(messages)
                   if m.get("role") == "tool" and m.get("tool_call_id") == call_id]
        if len(matches) != 1:
            raise ValueError("tool_call_id must identify exactly one tool result.")
        index, result = matches[0]
    else:
        index = selection["tool_message_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(messages):
            raise ValueError("tool_message_index is outside the messages list.")
        result = messages[index]
        if result.get("role") != "tool":
            raise ValueError("tool_message_index must select a tool result, not an assistant request.")
        call_id = _text(result.get("tool_call_id"), "selected tool_call_id")
    calls = [call for message in messages[:index] if message.get("role") == "assistant"
             for call in (message.get("tool_calls") or []) if call.get("id") == call_id]
    if len(calls) != 1:
        raise ValueError("Selected result must match exactly one preceding assistant tool call.")
    call = calls[0]
    function = call.get("function", {})
    name = function.get("name")
    if name not in {"clip_skim", "frame_inspect"} or result.get("name", name) != name:
        raise ValueError("Only matching clip_skim/frame_inspect observations can be curated.")
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Selected tool observation has no content.")
    if (re.match(r"\s*(?:Duplicate\b|Tool\s+`?\w+`?\s+execution failed)", content, re.I)
            or result.get("success") is False or result.get("succeeded") is False):
        raise ValueError("Failed or skipped tool calls cannot supply evidence cases.")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return index, result, call, name, arguments


def _recorded_image_count(result: dict) -> int | None:
    records = result.get("model_calls") or ([result["model_call"]] if result.get("model_call") else [])
    if not records:
        return None
    for record in records:
        if record.get("metrics", {}).get("error") or not record.get("output"):
            continue
        # Reference examples may precede the current observation. They are not
        # samples of this source video and must not enter its reconstruction.
        for message in reversed(record.get("input", {}).get("messages", [])):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                boundaries = [i for i, part in enumerate(content) if isinstance(part, dict)
                              and part.get("type") == "text"
                              and str(part.get("text", "")).startswith("END HISTORICAL REFERENCES.")]
                current = content[boundaries[-1] + 1:] if boundaries else content
                if not boundaries and any(event.get("status") == "injected"
                                          and event.get("added_reference_images", 0) > 0
                                          for event in result.get("visual_memory", [])):
                    raise ValueError("Cannot separate historical reference images from current observation samples.")
                count = sum(isinstance(part, dict) and part.get("type") == "image_url" for part in current)
                if count:
                    return count
    raise ValueError("Recorded tool model calls contain no successful visual observation.")


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _sample_indices(reader, name: str, arguments: dict, result: dict, selection: dict):
    fps = float(reader.get_avg_fps())
    if not len(reader) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source reader has invalid frame count or FPS.")
    duration = float(getattr(reader, "duration", len(reader) / fps))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source reader has invalid duration.")
    observed_count = _recorded_image_count(result)
    if name == "clip_skim":
        start = parse_video_timestamp(arguments["start_time"])
        end = min(parse_video_timestamp(arguments["end_time"]), duration)
        validate_time_range(start, end, duration)
        count = selection.get("num_frames", observed_count)
        if count is None:
            raise ValueError("No recorded clip image count; selection must supply historical num_frames.")
        count = _positive_int(count, "num_frames")
        first = min(max(int(start * fps), 0), len(reader) - 1)
        last = min(max(int(end * fps), first), len(reader) - 1)
        indices = np.linspace(first, last, min(last - first + 1, count)).astype(int).tolist()
        sampling = {"rule": "clip_floor_linspace", "num_frames": count,
                    "requested_interval": [arguments["start_time"], arguments["end_time"]],
                    "effective_interval": [start, end]}
    else:
        requested = arguments.get("timestamps")
        if not isinstance(requested, list) or not requested:
            raise ValueError("frame_inspect requires a non-empty timestamps list.")
        notice = re.search(r"only the first\s+(\d+)\s+in input order", result["content"])
        if observed_count is None and not notice and "max_frames" not in selection:
            raise ValueError("No recorded frame count; selection must supply historical max_frames.")
        limit = selection.get("max_frames", int(notice[1]) if notice else len(requested))
        limit = _positive_int(limit, "max_frames")
        times = sorted({parse_video_timestamp(value) for value in requested[:limit]})
        displayed_duration = max(duration, round(duration, 1), round(round(duration, 2), 1))
        if any(value > displayed_duration for value in times):
            raise ValueError("Selected frame request exceeded the original video duration.")
        indices = sorted({min(int(min(value, duration) * fps + 0.5), len(reader) - 1) for value in times})
        sampling = {"rule": "frame_nearest_deduplicated", "max_frames": limit,
                    "requested_timestamps": requested, "accepted_timestamps": times}
    if observed_count is not None and observed_count != len(indices):
        raise ValueError(f"Reconstructed {len(indices)} frames but recorded tool saw {observed_count}; "
                         "check source FPS and historical sampling configuration.")
    sampling.update(fps=fps, frame_count=len(reader), duration_seconds=duration,
                    frame_indices=indices, recorded_image_count=observed_count)
    return indices, [index / fps for index in indices], sampling


def validate_manifest(path: str | Path, *, require_verified: bool = False) -> list[dict]:
    """Read-only basic validation; pending review cases are valid unless requested otherwise."""
    path = Path(path).resolve()
    records = read_jsonl(path)
    ids = set()
    for case in records:
        case_id = _text(case.get("case_id"), "case_id")
        if case_id in ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        ids.add(case_id)
        for field in ("question", "source_video_id", "task_type"):
            _text(case.get(field), field)
        _answer(case.get("answer_text"))
        if case.get("split") not in {"train", "dev"}:
            raise ValueError(f"{case_id}: split must be train or dev; never populate memory from test.")
        if case.get("evidence_scope") not in {"frame", "clip"}:
            raise ValueError(f"{case_id}: evidence_scope must be frame or clip.")
        times, paths = case.get("timestamps"), case.get("image_paths")
        if not isinstance(times, list) or not times or not isinstance(paths, list) or len(times) != len(paths):
            raise ValueError(f"{case_id}: timestamps and image_paths must be non-empty aligned lists.")
        if case["evidence_scope"] == "frame" and len(paths) != 1:
            raise ValueError(f"{case_id}: frame evidence must contain exactly one image.")
        if case["evidence_scope"] == "clip" and len(paths) < 2:
            raise ValueError(f"{case_id}: clip evidence needs at least two ordered images.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in times):
            raise ValueError(f"{case_id}: timestamps must be finite, non-negative seconds.")
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError(f"{case_id}: timestamps must be ordered and distinct.")
        if not isinstance(case.get("verified"), bool) or not isinstance(case.get("verification_note"), str):
            raise ValueError(f"{case_id}: verified must be boolean and verification_note must be a string.")
        if case["verified"] and not case["verification_note"].strip():
            raise ValueError(f"{case_id}: reviewed cases require a non-empty verification_note.")
        if require_verified and not case["verified"]:
            raise ValueError(f"{case_id}: pending human verification.")
        for value in paths:
            image_path = _resolve(value, path.parent)
            if not image_path.is_file():
                raise FileNotFoundError(f"{case_id}: image not found: {image_path}")
            with Image.open(image_path) as image:
                image.verify()
    return records


def curate_cases(selections_path: str | Path, output_dir: str | Path, *,
                 reader_factory: Callable | None = None) -> Path:
    """Export one pending case per selected observation, never one GT label per frame.

    Relative input paths are resolved against the selections JSONL. Image sequence
    paths resolve against frame_root when supplied. Outputs are never overwritten.
    """
    selections_path = Path(selections_path).resolve()
    output = ensure_new_output(output_dir)
    selections = read_jsonl(selections_path)
    plans, ids = [], set()
    for selection in selections:
        source_video_id = _text(selection.get("source_video_id"), "source_video_id")
        task_type = _text(selection.get("task_type"), "task_type")
        answer, answer_source = _selection_answer(selection)
        if selection.get("split") not in {"train", "dev"}:
            raise ValueError("Selections require explicit split=train or dev; test is forbidden.")
        trajectory_path = _resolve(selection.get("trajectory_path"), selections_path.parent)
        with trajectory_path.open(encoding="utf-8-sig") as stream:
            trajectory = json.load(stream)
        question = _text(trajectory.get("question"), "trajectory question")
        index, result, call, name, arguments = _select_call(trajectory, selection)
        trajectory_hash = _sha256(trajectory_path)
        identity = {"trajectory": trajectory_hash, "tool_call": call, "source_video_id": source_video_id,
                    "answer_text": answer, "task_type": task_type}
        case_id = selection.get("case_id", "case_" + _object_hash(identity)[:20])
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", case_id):
            raise ValueError("case_id must be a safe filename of at most 96 ASCII characters.")
        if case_id in ids:
            raise ValueError(f"Duplicate selected case_id: {case_id}")
        ids.add(case_id)
        case = {"case_id": case_id, "question": question, "answer_text": answer,
                "source_video_id": source_video_id, "split": selection["split"], "task_type": task_type,
                "evidence_scope": "clip" if name == "clip_skim" else "frame",
                "timestamps": [], "image_paths": [], "verified": False, "verification_note": "",
                "previous_prediction": trajectory.get("final_answer"),
                "provenance": {"trajectory_path": str(trajectory_path), "trajectory_sha256": trajectory_hash,
                               "tool_message_index": index, "tool_call": call,
                               "tool_call_sha256": _object_hash(call),
                               "tool_message_sha256": _object_hash(result),
                               "selection_sha256": _object_hash(selection),
                               "answer_source": answer_source,
                               "label_scope": "question_level_not_individual_frame_fact"}}
        for field in ("source_id", "patient_id", "source_sample_id", "procedure"):
            if field in selection:
                case[field] = _text(selection[field], field)
        plans.append((case, selection, name, arguments, result))
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "cases.jsonl"
    pending = output / "cases.pending.jsonl"
    try:
        with pending.open("x", encoding="utf-8") as stream:
            for case, selection, name, arguments, result in plans:
                opener = reader_factory or _open_reader
                reader, media = opener(selection, selections_path.parent)
                indices, times, sampling = _sample_indices(reader, name, arguments, result, selection)
                images = reader.get_batch(indices).asnumpy()
                case["evidence_scope"] = "frame" if len(indices) == 1 else "clip"
                case_dir = output / "images" / case["case_id"]
                case_dir.mkdir(parents=True, exist_ok=False)
                source_frames = []
                for index, image in zip(indices, images):
                    image_path = case_dir / f"frame_{index:09d}.png"
                    Image.fromarray(image).convert("RGB").save(image_path, format="PNG")
                    case["image_paths"].append(image_path.relative_to(output).as_posix())
                    source = {"frame_index": index, "export_sha256": _sha256(image_path)}
                    if isinstance(reader, _ImageSequence):
                        source.update(path=str(reader.paths[index]), sha256=_sha256(reader.paths[index]))
                    source_frames.append(source)
                case["timestamps"] = times
                case["provenance"].update(media=media, sampling=sampling, source_frames=source_frames,
                                           export_transform="source_RGB_PNG; clip timestamps match tool, resolution retained",
                                           observed_transform="short_side_256" if name == "clip_skim" else "native_resolution",
                                           temporal_coverage="ordered sampled frames, not continuous observation")
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
        validate_manifest(pending)
        pending.rename(manifest)
        report = {"status": "pending_human_review", "count": len(plans),
                  "selections_path": str(selections_path), "selections_sha256": _sha256(selections_path),
                  "warning": "GT labels the question, not every exported frame. Review before indexing."}
        (output / "curation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as error:
        (output / "curation_failed.json").write_text(
            json.dumps({"status": "incomplete_do_not_index", "error": str(error)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise
    return manifest
