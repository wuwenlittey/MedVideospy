import argparse
import json
import os
import re
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
    / "official_repositories/StreamingBench/src/data/questions_real.json"
)
DEFAULT_VIDEO_ROOT = DEFAULT_PACKAGE_ROOT / "datasets/streamingbench/videos"
SAMPLE_STEM = re.compile(r"^(sample_\d+)(?:_.+)?$", re.IGNORECASE)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VideoSpy on StreamingBench Real-Time Visual Understanding."
    )
    add_common_arguments(parser)
    add_streaming_arguments(
        parser,
        qa_path=str(DEFAULT_QA_PATH),
        video_root=str(DEFAULT_VIDEO_ROOT),
    )
    return parser.parse_args(argv)


def _sample_id_from_video_path(video_path: str) -> str:
    stem = Path(str(video_path)).stem
    match = SAMPLE_STEM.match(stem)
    if match:
        return match.group(1)
    return stem


def resolve_video(sample: dict, video_root: Path) -> Path:
    sample_id = sample["video_id"]
    candidates = [
        video_root / sample_id / "video.mp4",
        video_root / f"{sample_id}.mp4",
        video_root / Path(sample["source_video_path"]).name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"StreamingBench video not found for {sample_id}: tried {candidates}"
    )


def _iter_entries(value):
    """Official SQA is a list of clip groups; the other files are flat lists."""
    if not isinstance(value, list):
        raise ValueError("StreamingBench annotation must be a list of video entries.")
    for item in value:
        if isinstance(item, list):
            for entry in item:
                yield entry
        else:
            yield item


def _usable_streamingbench_question(item: dict) -> tuple[float, str] | None:
    """Four-choice questions only. Proactive items have no options and are skipped."""
    options = item.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return None
    answer = str(item.get("answer", "")).strip().upper()
    if answer not in "ABCD" or len(answer) != 1:
        return None
    try:
        query_time = parse_timestamp(item.get("time_stamp"))
    except (TypeError, ValueError):
        return None
    if query_time < 0:
        return None
    if not item.get("task_type") or not item.get("question"):
        return None
    return query_time, answer


def load_dataset(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    seen = set()
    skipped = 0
    for entry_index, entry in enumerate(_iter_entries(value)):
        if not isinstance(entry, dict):
            raise ValueError("Each StreamingBench entry must be a mapping.")
        questions = entry.get("questions")
        if not isinstance(questions, list) or not questions:
            skipped += 1
            continue
        source_video_path = str(entry.get("video_path") or "")
        if not source_video_path:
            skipped += 1
            continue
        video_id = _sample_id_from_video_path(source_video_path)
        for question_index, item in enumerate(questions):
            if not isinstance(item, dict):
                skipped += 1
                continue
            usable = _usable_streamingbench_question(item)
            if usable is None:
                skipped += 1
                continue
            query_time, answer = usable
            sample_key = (
                f"{video_id}@{item['time_stamp']}#{entry_index}.{question_index}"
            )
            if sample_key in seen:
                raise ValueError(f"Duplicate StreamingBench sample key: {sample_key}")
            seen.add(sample_key)
            samples.append(
                {
                    "sample_key": sample_key,
                    "video_id": video_id,
                    "source_video_path": source_video_path,
                    "question": str(item["question"]),
                    "choices": choices_from_options(item["options"]),
                    "answer": answer,
                    "category": str(item["task_type"]),
                    "query_time": query_time,
                }
            )
    if skipped:
        print(f"Skipped {skipped} StreamingBench items that are not four-choice questions.")
    return samples


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    qa_path = Path(args.qa_path).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    samples = select_samples(
        load_dataset(qa_path), args.mode, getattr(args, "sample_manifest", None)
    )
    return run_streaming_benchmark(
        args,
        dataset_name="streamingbench",
        samples=samples,
        resolve_video=lambda sample: resolve_video(sample, video_root),
        agent_class=agent_class,
    )


def main(argv: Optional[list[str]] = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
