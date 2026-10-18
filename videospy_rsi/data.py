"""Normalized, source-disjoint data with explicit label-free runtime views."""

import copy
import json
from pathlib import Path

from .io import read_json, write_json

RUNTIME_FIELDS = {"id", "source_id", "question", "video_path", "frame_paths", "fps",
                  "subtitle_path", "task_type", "duration", "demo", "metric"}


def runtime_view(sample):
    return {key: copy.deepcopy(value) for key, value in sample.items() if key in RUNTIME_FIELDS}


def normalize(text, metric="option"):
    text = str(text).strip()
    if metric == "option":
        return text if len(text) == 1 and text in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else ""
    return " ".join(text.casefold().split())


def correct(record, sample):
    metric = sample.get("metric", "option")
    return record.get("status") == "ok" and normalize(record.get("prediction", ""), metric) == normalize(sample["answer"], metric)


def load_dataset(path):
    data = read_json(path)
    validate_dataset(data)
    base = Path(path).resolve().parent
    for samples in all_groups(data):
        for sample in samples:
            for key in ("video_path", "subtitle_path"):
                if sample.get(key):
                    sample[key] = str((base / sample[key]).resolve())
            if sample.get("frame_paths"):
                sample["frame_paths"] = [str((base / p).resolve()) for p in sample["frame_paths"]]
    return data


def all_groups(data):
    yield from data["diagnostic_batches"]
    yield data["gate"]
    for episode in data["episodes"]:
        for key in ("probe", "inner_gate", "report"):
            yield episode[key]
    yield data.get("test", [])


def validate_dataset(data):
    if not isinstance(data, dict) or not data.get("diagnostic_batches") or not data.get("gate"):
        raise ValueError("Dataset needs diagnostic_batches and gate")
    episodes = data.get("episodes", [])
    if len(episodes) < 2 or len({e.get("id") for e in episodes}) != len(episodes):
        raise ValueError("At least two uniquely identified Meta episodes are required")
    scopes = {"diag": [s for batch in data["diagnostic_batches"] for s in batch],
              "gate": data["gate"], "meta": [], "test": data.get("test", [])}
    for episode in episodes:
        groups = [episode.get(key, []) for key in ("probe", "inner_gate", "report")]
        if any(not group for group in groups):
            raise ValueError("Each Meta episode needs probe, inner_gate and report")
        sources = [{s["source_id"] for s in group} for group in groups]
        if any(sources[i] & sources[j] for i in range(3) for j in range(i)):
            raise ValueError("Meta probe, inner_gate and report must be source-disjoint")
        scopes["meta"].extend(s for group in groups for s in group)
    seen = {}
    for scope, samples in scopes.items():
        for sample in samples:
            required = {"id", "source_id", "question", "answer"}
            if not isinstance(sample, dict) or not required <= sample.keys() or any(
                not isinstance(sample[k], str) or not sample[k].strip() for k in required
            ):
                raise ValueError("Sample needs nonempty id/source_id/question/answer strings")
            if sample.get("metric", "option") not in {"option", "exact"}:
                raise ValueError("First implementation supports option or exact normalized scoring")
            if sample.get("metric", "option") == "option" and not normalize(sample["answer"]):
                raise ValueError("Option gold must be one uppercase letter")
            if not any(sample.get(k) for k in ("video_path", "frame_paths", "demo")):
                raise ValueError("Sample needs a video, frame sequence or demo source")
            source = sample["source_id"]
            if source in seen and seen[source] != scope:
                raise ValueError(f"Source leaks between dataset scopes: {source}")
            seen[source] = scope
    # An identical sample can be reused by explicitly paired episodes, but not
    # carry different labels/questions or appear twice within one execution batch.
    identity = {}
    for group in all_groups(data):
        ids = [s["id"] for s in group]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample ID within a batch")
        for sample in group:
            previous = identity.setdefault(sample["id"], sample)
            if previous != sample:
                raise ValueError("Conflicting sample identities")


def prepare_lvbench(metadata, video_dir, splits, output):
    """Map user-provided, video-disjoint UID splits into the normalized format."""
    by_id = {}
    for line in Path(metadata).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        video = json.loads(line)
        video_id = str(video["key"])
        for qa in video["qa"]:
            by_id[str(qa["uid"])] = {
                "id": str(qa["uid"]), "source_id": video_id, "question": qa["question"],
                "answer": str(qa["answer"]), "video_path": str((Path(video_dir) / f"{video_id}.mp4").resolve()),
                "task_type": "|".join(qa.get("question_type", ["general"])), "metric": "option",
            }
    def resolve(ids):
        try:
            return [copy.deepcopy(by_id[str(sid)]) for sid in ids]
        except KeyError as error:
            raise ValueError(f"Unknown LVBench UID: {error}") from error
    manifest = read_json(splits)
    data = {"diagnostic_batches": [resolve(batch) for batch in manifest["diagnostic_batches"]],
            "gate": resolve(manifest["gate"]), "test": resolve(manifest.get("test", [])),
            "episodes": [{"id": e["id"], "start": e.get("start", "initial"),
                          "seed": e.get("seed", 42), **{k: resolve(e[k]) for k in
                          ("probe", "inner_gate", "report")}} for e in manifest["episodes"]]}
    validate_dataset(data)
    write_json(output, data)


def demo_dataset():
    def sample(name):
        return {"id": name, "source_id": name, "question": "What is the background color?\nA. red\nB. blue",
                "answer": "A", "metric": "option", "demo": True, "task_type": "appearance"}
    return {"diagnostic_batches": [[sample(f"diag-{i}")] for i in range(6)],
            "gate": [sample("gate")], "test": [sample("test")],
            "episodes": [{"id": f"episode-{i}", "start": start, "seed": 42 + i,
                          "probe": [sample(f"probe-{i}")], "inner_gate": [sample(f"inner-{i}")],
                          "report": [sample(f"report-{i}")]} for i, start in enumerate(("initial", "current"))]}
