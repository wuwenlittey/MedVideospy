import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.videoseek import general_config, init_config, load_config, prompts_config
from .agent import VideoSeekAgent


def build_agent_config(args: argparse.Namespace) -> dict:
    config_path = getattr(args, "config_path", None)
    base_config = (
        load_config(Path(config_path).expanduser()) if config_path else general_config
    )
    config = {}
    config.update(base_config)
    config.update(prompts_config)
    config = init_config(config, args)
    return config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="videoseek")
    p.add_argument("--video_path", required=True, help="YouTube URL or local path to video.")
    p.add_argument("--user_query", required=True, help="Question/query towards this video.")
    p.add_argument("--subtitle_path", default=None, help="Local path to subtitle file.")
    p.add_argument(
        "--output_dir",
        default="./output/",
        help="Directory to write outputs (default: ./output/).",
    )
    p.add_argument("--verbose", action="store_true", help="Print agent step logs.")
    p.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Agent YAML config. Defaults to config/videoseek/general.yml.",
    )

    # Allow overriding general.yml keys (optional)
    p.add_argument("--model_name", default=None, help="Model name.")
    p.add_argument("--api_base", default=None, help="API base.")
    p.add_argument("--api_key", default=None, help="API key.")
    p.add_argument("--api_version", default=None, help="API version.")
    p.add_argument(
        "--reasoning_effort", default=None, help="Reasoning effort of the LLM."
    )
    p.add_argument("--seed", type=int, default=None, help="Seed.")
    p.add_argument("--temperature", type=float, default=None, help="Temperature.")
    p.add_argument(
        "--max_tokens", type=int, default=None, help="Max output tokens of the LLM."
    )
    p.add_argument(
        "--max_steps", type=int, default=None, help="Max steps of the VideoSeek agent."
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_agent_config(args)

    video_id = args.video_path.split("/")[-1].split(".")[0]

    run_dir = output_dir / "videoseek" / "demo" / f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run the agent
    agent = VideoSeekAgent(
        config=config,
        video_path=args.video_path,
        subtitle_path=args.subtitle_path,
        output_dir=str(run_dir),
        tools=config["tools"],
        verbose=args.verbose,
    )
    traj = agent.run(args.user_query)
    traj_dict = traj.to_dict()
    prediction = traj_dict.get("final_answer", "")

    print(f"Question: {args.user_query}")
    print(f"Prediction: {prediction}")

    (run_dir / "prediction.json").write_text(json.dumps({"prediction": prediction}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "trajectory.json").write_text(json.dumps(traj_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
