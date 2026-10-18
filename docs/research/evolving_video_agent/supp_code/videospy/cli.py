import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.videospy import build_config

from .agent import VideoSpyAgent
from .utils import append_commit_id


def build_agent_config(args: argparse.Namespace) -> dict:
    config_paths = getattr(args, "config_paths", None) or []
    return build_config(
        args,
        config_paths=[Path(path) for path in config_paths],
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="videospy")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--user_query", required=True)
    parser.add_argument("--subtitle_path", default=None)
    parser.add_argument("--output_dir", default="./output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--enabled-tools", nargs="*", default=None,
        choices=("overview", "clip_skim", "frame_inspect"),
        help="Override enabled visual tools. Pass no names to disable all tools.",
    )
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        default=[],
        help="Agent YAML override. Repeat to compose multiple configs in order.",
    )
    parser.add_argument(
        "--evolution-db",
        dest="evolution_db",
        default=None,
        help="Versioned SQLite state produced by VideoSpy evolution.",
    )
    parser.add_argument(
        "--evolution-version",
        default=None,
        help="Evolution version ID. Defaults to the final or latest version.",
    )
    parser.add_argument(
        "--evolved-skill", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--evolved-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--evolved-hyperparameter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_version", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
    )
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    video_id = Path(args.video_path).stem
    run_name = append_commit_id(
        f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / "videospy" / "demo" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_agent_config(args)
    agent = VideoSpyAgent(
        config=config,
        video_path=args.video_path,
        subtitle_path=args.subtitle_path,
        output_dir=str(run_dir),
        tools=config["tools"]["enabled"],
        verbose=args.verbose,
    )
    trajectory = agent.run(args.user_query).to_dict()
    prediction = trajectory.get("final_answer", "")
    print(f"Question: {args.user_query}")
    print(f"Prediction: {prediction}")

    (run_dir / "prediction.json").write_text(
        json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
