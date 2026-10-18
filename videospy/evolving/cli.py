import argparse
from pathlib import Path

from config.videospy import build_config

from .pipeline import EvolutionPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m videospy.evolving.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run VideoSpy evolution.")
    run.add_argument(
        "--benchmark", choices=("lvbench", "egolife", "medvidbench"), required=True
    )
    run.add_argument("--train-manifest", required=True)
    run.add_argument("--workers", type=int, default=2, help="Concurrent classification and reflection jobs.")
    run.add_argument("--mode", choices=("full", "lite", "debug"), default="full")
    run.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        default=[],
        help="YAML override. Repeat to compose model, benchmark, and experiment configs.",
    )
    run.add_argument("--output-dir", default=None,
                     help="Parent directory for new runs; defaults to output/evolving/<benchmark>.")
    run.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Existing evolution run directory to resume, or an explicit directory "
            "for a new run."
        ),
    )
    run.add_argument(
        "--inference-run-dir",
        required=True,
        help="Completed full Benchmark run to use as read-only offline training input.",
    )
    run.add_argument(
        "--run-dir-file",
        default=None,
        help="Write the completed run directory to this file for shell workflows.",
    )
    run.add_argument("--reuse-reflection-dir", default=None,
                     help="Read-only current-version reflection checkpoints to reuse when fingerprints match.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config = build_config(
        args,
        config_paths=[Path(path) for path in args.config_paths],
    )
    evolving = config.setdefault("evolving", {})
    evolving["enabled"] = True
    evolving["mode"] = args.mode
    pipeline = EvolutionPipeline(
        benchmark=args.benchmark,
        workers=args.workers,
        train_manifest=args.train_manifest,
        output_dir=args.output_dir or f"output/evolving/{args.benchmark}",
        run_dir=args.run_dir,
        inference_run_dir=args.inference_run_dir,
        config=config,
        config_paths=args.config_paths,
        reuse_reflection_dir=args.reuse_reflection_dir,
    )
    run_dir = pipeline.run()
    if pipeline.result["status"] == "failed":
        print(f"Evolution failed; checkpoints and partial artifacts preserved: {run_dir}")
        return 1
    print(f"Evolution status: {pipeline.result['status']}")
    if pipeline.result.get('evaluation_blockers'):
        for reason in pipeline.result['evaluation_blockers']:
            print(reason)
        print(f"Evolution artifacts: {run_dir}")
        return 1
    if args.run_dir_file is not None:
        Path(args.run_dir_file).write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    print(f"Evolution run: {run_dir}")
    print(f"Evolution database: {run_dir / 'evolution.sqlite3'}")
    print(f"Final version: {run_dir / 'final_version.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
