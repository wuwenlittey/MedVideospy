import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.lvbench.utils import add_common_arguments
from benchmark.streaming_cutoff import parse_timestamp
from benchmark.streaming_runner import (
    add_streaming_arguments,
    choices_from_options,
    letter_from_index,
    run_streaming_benchmark,
    select_samples,
)


DEFAULT_PACKAGE_ROOT = Path(
    os.environ.get(
        "STREAMING_BENCHMARKS_ROOT",
        "/share/home/tm1005929190170000/a1111547840/xy/streaming-video-benchmarks/streaming_video_benchmark_package",
    )
)
DEFAULT_QA_PATH = (
    DEFAULT_PACKAGE_ROOT
    / "official_repositories/StreamEQA/data/annotations/Perception.json"
)
DEFAULT_VIDEO_ROOT = DEFAULT_PACKAGE_ROOT / "datasets/hd_epic/videos"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VideoSpy on StreamEQA using HD-EPIC videos."
    )
    add_common_arguments(parser)
    add_streaming_arguments(
        parser,
        qa_path=str(DEFAULT_QA_PATH),
        video_root=str(DEFAULT_VIDEO_ROOT),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split name recorded in sample keys, defaulting to the QA filename.",
    )
    return parser.parse_args(argv)


def resolve_video(sample: dict, video_root: Path) -> Path:
    video_id = sample["video_id"]
    path = video_root / video_id.split("-")[0] / f"{video_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"HD-EPIC video not found: {path}")
    return path


def _query_time(item: dict):
    if item.get("realtime") is None:
        return None
    try:
        value = parse_timestamp(item["realtime"])
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _reject_streameqa_item(item: dict) -> str | None:
    """Return why this official item cannot be asked, or None if it is usable."""
    options = item.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return "options"
    realtime = item.get("realtime")
    if isinstance(realtime, (int, float)) and not isinstance(realtime, bool) and float(realtime) < 0:
        return "negative_time"
    if _query_time(item) is None:
        return "missing_time"
    gt = item.get("gt")
    if isinstance(gt, bool) or not isinstance(gt, int) or not 0 <= gt < 4:
        return "gt"
    return None


def load_dataset(path: Path, split: str | None = None) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("StreamEQA annotation must be a list of questions.")
    split_name = split or path.stem
    samples = []
    skipped = {"options": 0, "negative_time": 0, "missing_time": 0, "gt": 0}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each StreamEQA item must be a mapping.")
        reason = _reject_streameqa_item(item)
        if reason is not None:
            skipped[reason] += 1
            continue
        required = {"id", "task", "video", "question", "options", "gt"}
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"StreamEQA item is missing fields: {missing}")
        samples.append(
            {
                "sample_key": f"{split_name}:{item['id']}",
                "video_id": str(item["video"]),
                "question": str(item["question"]),
                "choices": choices_from_options(item["options"]),
                "answer": letter_from_index(int(item["gt"])),
                "category": str(item["task"]),
                "query_time": _query_time(item),
            }
        )
    if not samples:
        raise ValueError(f"No usable StreamEQA questions in {path}")
    if skipped["missing_time"]:
        print(f"Skipped {skipped['missing_time']} StreamEQA items without a query timestamp.")
    if skipped["negative_time"]:
        print(f"Skipped {skipped['negative_time']} StreamEQA items with a negative timestamp.")
    if skipped["options"]:
        print(f"Skipped {skipped['options']} StreamEQA items without four options.")
    if skipped["gt"]:
        print(f"Skipped {skipped['gt']} StreamEQA items with an invalid answer index.")
    return samples


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    qa_path = Path(args.qa_path).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    samples = select_samples(
        load_dataset(qa_path, getattr(args, "split", None)),
        args.mode,
        getattr(args, "sample_manifest", None),
    )
    return run_streaming_benchmark(
        args,
        dataset_name="streameqa",
        samples=samples,
        resolve_video=lambda sample: resolve_video(sample, video_root),
        extra_manifest={"split": getattr(args, "split", None) or qa_path.stem},
        agent_class=agent_class,
    )


def main(argv: Optional[list[str]] = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
