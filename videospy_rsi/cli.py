"""Local-only CLI: run, resume, offline demo, inference and LVBench preparation."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import demo_dataset, load_dataset, prepare_lvbench
from .io import read_json, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="videospy-rsi")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True, help="New or compatible resumable RSI run directory")
        sub.add_argument("--config", help="Independent RSI JSON configuration")
        sub.add_argument("--final-test", action="store_true")
        if name == "run":
            sub.add_argument("--dataset", required=True, help="Normalized source-disjoint dataset JSON")
    sub = commands.add_parser("prepare-lvbench")
    for flag in ("metadata", "video-dir", "splits", "output"):
        sub.add_argument("--" + flag, required=True)
    sub = commands.add_parser("infer")
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--task-version")
    sub.add_argument("--config")
    sub.add_argument("--video-path", required=True)
    sub.add_argument("--subtitle-path")
    sub.add_argument("--question", required=True)
    sub.add_argument("--task-type", default="general")
    sub.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare-lvbench":
        prepare_lvbench(args.metadata, args.video_dir, args.splits, args.output)
        print(f"Prepared normalized dataset: {args.output}")
        return 0
    if args.command == "infer":
        from .archive import Archive
        from .runner import Runner
        from .pipeline import implementation_identity
        from .io import public_config, digest
        config = load_config(args.config)
        run = read_json(Path(args.run_dir) / "run.json")
        if run["identity"]["implementation"] != implementation_identity():
            raise ValueError("Frozen runtime implementation changed")
        # A new inference output need not share search budgets, but fixed models
        # must match the accepted run. Credentials are deliberately not hashed.
        saved_config = read_json(Path(args.run_dir) / "settings.json")
        if public_config(config["models"]) != saved_config["models"]:
            raise ValueError("Inference model roles differ from the accepted run")
        archive = Archive(args.run_dir, run["identity"])
        state = archive.state()
        task = archive.get(args.task_version or state["task_id"])
        runner = Runner(archive, config)
        sample = {"id": "inference", "source_id": str(Path(args.video_path).resolve()),
                  "question": args.question, "video_path": str(Path(args.video_path).resolve()),
                  "subtitle_path": str(Path(args.subtitle_path).resolve()) if args.subtitle_path else None,
                  "task_type": args.task_type}
        result = runner.job("infer-" + digest({"sample": sample, "task": task["id"], "seed": config["seed"]})[:20], "task",
                            {"scope": "final_test", "stage": "inference", "seed": config["seed"]},
                            snapshot=task, sample=sample)
        write_json(args.output, result)
        if result["status"] != "ok":
            print(result.get("error"))
            return 1
        print(result["result"]["prediction"])
        return 0
    from .pipeline import EvolutionPipeline
    from .io import public_config
    config = load_config(args.config, backend="demo" if args.command == "demo" else None)
    dataset = demo_dataset() if args.command == "demo" else load_dataset(args.dataset)
    pipeline = EvolutionPipeline(config, dataset, args.output)
    pipeline.archive.write("settings.json", public_config(config))
    state = pipeline.run()
    print(json.dumps(state, ensure_ascii=False, indent=2))
    if args.final_test:
        report = pipeline.final_test()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["complete"] or not report["usage_complete"] or report["failed"]:
            return 1
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
