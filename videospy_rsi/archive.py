"""Immutable snapshots, atomic official pointers and phase checkpoints."""

import copy
import difflib
import json
from pathlib import Path

from .io import digest, public_config, read_json, write_json


class Archive:
    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        run = self.root / "run.json"
        if run.exists():
            if read_json(run)["identity"] != identity:
                raise ValueError("Resume configuration, dataset or implementation changed; use a new run")
        elif any(self.root.iterdir()):
            raise ValueError("New RSI run directory must be empty")
        else:
            write_json(run, {"schema": 1, "identity": identity})

    def snapshot(self, value):
        value = public_config(copy.deepcopy(value))
        value.pop("id", None)
        if value.get("layer") not in {"task", "meta"}:
            raise ValueError("Invalid snapshot layer")
        version = value["layer"] + "-" + digest(value)[:24]
        value["id"] = version
        directory = self.root / "versions" / value["layer"] / version
        target = directory / "manifest.json"
        if target.exists():
            if read_json(target) != value:
                raise ValueError("Snapshot identity collision or modification")
            return value
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in value["files"].items():
            if Path(name).name != name:
                raise ValueError("Snapshot filenames must be flat")
            (directory / name).write_text(source, encoding="utf-8")
        write_json(target, value)
        return value

    def get(self, version):
        if not isinstance(version, str) or Path(version).name != version or not version.startswith(("task-", "meta-")):
            raise ValueError("Invalid version ID")
        layer = version.split("-", 1)[0]
        directory = self.root / "versions" / layer / version
        value = read_json(directory / "manifest.json")
        expected = {key: item for key, item in value.items() if key != "id"}
        if version != layer + "-" + digest(expected)[:24]:
            raise ValueError("Modified snapshot manifest")
        for name, source in value["files"].items():
            if Path(name).name != name or (directory / name).read_text(encoding="utf-8") != source:
                raise ValueError("Modified snapshot source")
        return value

    def state(self):
        path = self.root / "state.json"
        return read_json(path) if path.exists() else None

    def save_state(self, state):
        self.get(state["task_id"])
        self.get(state["meta_id"])
        write_json(self.root / "state.json", state)

    def phase(self, key, inputs, producer):
        path = self.root / "phases" / f"{key}.json"
        identity = digest(inputs)
        if path.exists():
            saved = read_json(path)
            if saved["input_hash"] != identity:
                raise ValueError(f"Checkpoint input changed: {key}")
            return saved["result"]
        result = producer()
        write_json(path, {"input_hash": identity, "result": result})
        return result

    def write(self, key, value):
        write_json(self.root / key, value)

    def attempts(self):
        paths = sorted((self.root / "task_rounds").glob("*/attempt.json"))
        return [read_json(path) for path in paths]

    def patch(self, parent, candidate, key):
        pieces = []
        for name, text in sorted(candidate.get("files", {}).items()):
            pieces.extend(difflib.unified_diff(parent["files"][name].splitlines(True),
                                               text.splitlines(True), fromfile=name, tofile=name))
        path = self.root / key / "source.patch"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(pieces), encoding="utf-8")
        write_json(path.with_name("proposal.json"), candidate)
