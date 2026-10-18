import argparse
import copy
import json
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from time import perf_counter
from typing import Optional

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.egolife.timeline import EgoLifeTimelineReader, parse_egolife_time
from benchmark.sampling import get_sampling_stride
from benchmark.evolution_diagnostics import aggregate_evolution_usage, evolution_usage_notes
from benchmark.lvbench.utils import (
    add_common_arguments,
    aggregate_task_metrics,
    append_record,
    attach_task_metrics,
    build_agent_config,
    create_run_dir,
    ensure_run_manifest,
    get_agent_class,
    get_agent_model_name,
    get_agent_tools,
    is_completed_record,
    load_latest_records,
    safe_config,
    tee_output,
    write_json,
    write_markdown_table,
    write_trajectory,
)
from videospy.utils import load_subtitles as load_srt
from videospy.evolving.common.memory_scope import source_context


DEFAULT_DATA_ROOT = Path("./data/EgoLife/lmms-lab_EgoLife")
DEFAULT_QA_PATH = DEFAULT_DATA_ROOT / "EgoLifeQA/EgoLifeQA_A1_JAKE.json"
DEFAULT_VIDEO_ROOT = DEFAULT_DATA_ROOT / "A1_JAKE"
DEFAULT_TRANSCRIPT_ROOT = DEFAULT_DATA_ROOT / "EgoLifeCap/Transcript/A1_JAKE"
DEFAULT_SAMPLE_MANIFEST = str(
    Path(__file__).resolve().parents[2] / "experiments/data_splits/main/egolife/test.json"
)
TRANSCRIPT_PATTERN = re.compile(r"A1_JAKE_DAY(\d+)_(\d{8})$")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VideoSpy on EgoLifeQA A1_JAKE.")
    add_common_arguments(parser)
    parser.set_defaults(agent="videospy", sample_manifest=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument(
        "--transcript-root",
        default=str(DEFAULT_TRANSCRIPT_ROOT),
        help="Official A1_JAKE Transcript directory.",
    )
    return parser.parse_args(argv)


def _answer_letter(sample: dict) -> str:
    answer = str(sample["answer"]).strip().upper()
    if answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"Cannot normalize EgoLifeQA answer: {sample['answer']!r}")
    return answer


def load_dataset(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "questions", "qa", "A1_JAKE"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError("EgoLifeQA annotation must contain a list of questions.")
    samples = []
    required = {
        "ID",
        "query_time",
        "type",
        "question",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "answer",
    }
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each EgoLifeQA item must be a mapping.")
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"EgoLifeQA item is missing fields: {missing}")
        query_time = item["query_time"]
        if not isinstance(query_time, dict):
            raise ValueError("EgoLifeQA query_time must contain date and time fields.")
        parse_egolife_time(query_time)
        day = str(query_time["date"]).upper()
        samples.append(
            {
                "sample_key": str(item["ID"]),
                "question": str(item["question"]),
                "choices": {
                    letter: item[f"choice_{letter.lower()}"] for letter in "ABCD"
                },
                "answer": _answer_letter(item),
                "category": str(item["type"]),
                "query_time": {"date": day, "time": str(query_time["time"])},
                "day": day,
            }
        )
    return samples


def _manifest_values(path: str | Path) -> list[str]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("sample_ids", value.get("question_ids"))
    if not isinstance(value, list):
        raise ValueError("EgoLife manifest must be a JSON list of question IDs.")
    return [str(item) for item in value]


def select_samples(
    samples: list[dict], mode: str, manifest_path: str | Path | None
) -> list[dict]:
    if manifest_path:
        selected = _manifest_values(manifest_path)
        samples_by_key = {item["sample_key"]: item for item in samples}
        unknown = set(selected) - samples_by_key.keys()
        if unknown:
            raise ValueError(f"Unknown EgoLife question IDs: {sorted(unknown)}")
        samples = [samples_by_key[key] for key in selected]
    stride = get_sampling_stride("egolife", mode)
    return samples[::stride]


def normalize_answer(value: str) -> str:
    text = str(value).strip().upper()
    direct = re.fullmatch(r"\(?\s*([A-D])\s*\)?[\s.,:;-]*", text)
    if direct:
        return direct.group(1)
    candidates = re.findall(r"\b([A-D])\b", text)
    return candidates[-1] if candidates else ""


def build_question(sample: dict, timeline: EgoLifeTimelineReader) -> str:
    lines = [sample["question"], ""]
    for letter in "ABCD":
        lines.append(f"{letter}. {sample['choices'][letter]}")
    query_time = sample["query_time"]
    absolute = parse_egolife_time(query_time)
    lines.extend(
        [
            "",
            f"Current query time: {query_time['date']} {query_time['time']}.",
            f"Its timestamp on the provided timeline is {timeline.to_timeline_seconds(absolute):.1f} seconds.",
        ]
    )
    lines.extend(
        [
            "",
            "Output requirement: Return exactly one uppercase option letter: A, B, C, or D.",
        ]
    )
    return "\n".join(lines)


def load_transcripts(
    transcript_root: str | Path, timeline: EgoLifeTimelineReader
) -> list[dict]:
    root = Path(transcript_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EgoLife Transcript directory not found: {root}")
    result = []
    for path in sorted(root.rglob("*.srt")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        match = TRANSCRIPT_PATTERN.fullmatch(path.stem.upper())
        if not match:
            continue
        base_time = parse_egolife_time(f"DAY{match.group(1)}@{match.group(2)}")
        for item in load_srt(str(path)):
            result.append(
                {
                    "start_time": timeline.to_timeline_seconds(
                        base_time + item["start_time"]
                    ),
                    "end_time": timeline.to_timeline_seconds(
                        base_time + item["end_time"]
                    ),
                    "subtitle": item["subtitle"],
                }
            )
    if not result:
        raise FileNotFoundError(f"No official A1_JAKE Transcript SRT found under: {root}")
    result.sort(key=lambda item: (item["start_time"], item["end_time"]))
    return result


def compute_metrics(samples: list[dict], records: dict[str, dict]) -> dict:
    totals = defaultdict(int)
    correct = defaultdict(int)
    overall = 0
    for sample in samples:
        is_correct = records.get(sample["sample_key"], {}).get("prediction") == sample["answer"]
        overall += int(is_correct)
        totals[sample["category"]] += 1
        correct[sample["category"]] += int(is_correct)
    by_category = {
        key: {
            "accuracy": correct[key] / total,
            "correct": correct[key],
            "total": total,
        }
        for key, total in sorted(totals.items())
    }
    return {
        "overall": {
            "accuracy": overall / len(samples) if samples else 0.0,
            "correct": overall,
            "total": len(samples),
        },
        "by_category": by_category,
        "macro_average": (
            sum(item["accuracy"] for item in by_category.values()) / len(by_category)
            if by_category else 0.0
        ),
    }


def write_report(
    run_dir: Path,
    agent_name: str,
    mode: str,
    model_name: str,
    samples: list[dict],
    records: dict[str, dict],
    metrics: dict,
) -> None:
    def make_row(task: str, task_samples: list[dict], accuracy: float) -> list[object]:
        keys = [sample["sample_key"] for sample in task_samples]
        statistics = aggregate_task_metrics(keys, records)
        completed = sum(
            records.get(key, {}).get("status") == "success" for key in keys
        )
        measured_tasks = statistics["measured_tasks"]
        average_tokens = (
            statistics["token_usage"]["total_tokens"] / measured_tasks
            if measured_tasks
            else 0.0
        )
        return [
            task,
            agent_name,
            mode,
            model_name,
            f"{completed}/{len(keys)}",
            f"{accuracy * 100:.1f}",
            f"{statistics['agent_rounds']['average']:.1f}",
            f"{average_tokens:.1f}",
            f"{statistics['elapsed_seconds']['average']:.2f}",
        ]

    rows = [make_row("Overall", samples, metrics["overall"]["accuracy"])]
    for category, category_metrics in metrics["by_category"].items():
        category_samples = [
            sample for sample in samples if sample["category"] == category
        ]
        rows.append(
            make_row(category, category_samples, category_metrics["accuracy"])
        )

    write_markdown_table(
        run_dir / "report.md",
        "EgoLifeQA Run Report",
        [
            "Task",
            "Agent",
            "Mode",
            "Model",
            "Completed",
            "Accuracy (%)",
            "Avg. rounds",
            "Avg. tokens",
            "Avg. time (s)",
        ],
        rows,
        notes=[f"- Macro Average (%): {metrics['macro_average'] * 100:.1f}", '',
               *evolution_usage_notes(aggregate_evolution_usage([s['sample_key'] for s in samples], records))],
    )


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    if args.agent != "videospy":
        raise ValueError("The EgoLife timeline adapter currently supports VideoSpy only.")
    qa_path = Path(args.qa_path).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    transcript_root = Path(args.transcript_root).expanduser().resolve()
    samples = select_samples(
        load_dataset(qa_path), args.mode, getattr(args, "sample_manifest", None),
    )
    selected_keys = [sample["sample_key"] for sample in samples]
    agent_name = args.agent
    config = copy.deepcopy(build_agent_config(args))
    config.setdefault("agent", {})["include_subtitles_in_question"] = False
    model_name = get_agent_model_name(config, agent_name)
    run_dir = create_run_dir(
        "egolife",
        args.mode,
        args.output_dir,
        args.run_dir,
        agent_name,
        model_name,
    )
    progress_stream = sys.stderr
    with tee_output(run_dir / "run.log"):
        print(
            f"Starting EgoLifeQA: agent={agent_name}, mode={args.mode}, "
            f"samples={len(samples)}, workers={getattr(args, 'workers', 4)}, "
            f"verbose={args.verbose}",
            flush=True,
        )
        print(f"Log file: {run_dir / 'run.log'}", flush=True)
        timeline = EgoLifeTimelineReader(video_root)
        transcripts = load_transcripts(transcript_root, timeline)
        ensure_run_manifest(
            run_dir,
            {
                "dataset": "egolife",
                "agent": agent_name,
                "mode": args.mode,
                "qa_path": str(qa_path),
                "video_root": str(video_root),
                "transcript_root": str(transcript_root),
                "timeline_mapping": timeline.time_mapping,
                "sample_keys": selected_keys,
                "agent_config": safe_config(config),
            },
        )
        agent_class = agent_class or get_agent_class(agent_name)
        records_path = run_dir / "records.jsonl"
        latest = load_latest_records(records_path)
        worker_state = local()
        first_video = str(timeline.clips[0][1])

        def run_sample(sample: dict) -> tuple[str, dict, dict | None, str | None]:
            key = sample["sample_key"]
            trajectory_dict = None
            task_agent = None
            error_traceback = None
            started_at = perf_counter()
            try:
                if not hasattr(worker_state, "timeline"):
                    worker_state.timeline = EgoLifeTimelineReader(video_root)
                    worker_state.agent = agent_class(
                        config=config,
                        video_path=first_video,
                        subtitle_path=None,
                        output_dir=str(run_dir),
                        tools=get_agent_tools(config, agent_name),
                        verbose=args.verbose,
                        video_reader=worker_state.timeline,
                    )
                    worker_state.agent.subtitles = transcripts
                worker_state.timeline.set_query_time(sample["query_time"])
                task_agent = worker_state.agent
                task_agent.duration = round(worker_state.timeline.duration, 2)
                task_agent.task_types = [sample["category"]]
                task_agent.task_type = sample["category"]
                task_agent.memory_context = {
                    **source_context('egolife', video_root, sample['query_time']),
                    'timeline_id': timeline.time_mapping['id'], 'sample_id': key,
                    'query': sample['question'],
                }
                trajectory = task_agent.run(
                    build_question(sample, worker_state.timeline)
                )
                trajectory_dict = trajectory.to_dict()
                raw_prediction = trajectory_dict.get("final_answer", "")
                record = {
                    "sample_key": key,
                    "status": "success",
                    "question": sample["question"],
                    "category": sample["category"],
                    "query_time": sample.get("query_time"),
                    "day": sample["day"],
                    "raw_prediction": raw_prediction,
                    "prediction": normalize_answer(raw_prediction),
                    "answer": sample["answer"],
                }
            except Exception as error:
                error_traceback = traceback.format_exc()
                record = {
                    "sample_key": key,
                    "status": "error",
                    "question": sample["question"],
                    "category": sample["category"],
                    "day": sample["day"],
                    "error": f"{type(error).__name__}: {error}",
                }
            attach_task_metrics(
                record, task_agent, trajectory_dict, perf_counter() - started_at
            )
            return key, record, trajectory_dict, error_traceback

        progress = tqdm(
            total=len(samples),
            desc=f"EgoLifeQA {args.mode}",
            dynamic_ncols=True,
            file=progress_stream,
            unit="question",
        )
        pending_samples = []
        for sample in samples:
            key = sample["sample_key"]
            if is_completed_record(latest.get(key, {})):
                progress.update(1)
                progress.set_postfix_str(f"id={key} skipped")
                continue
            pending_samples.append(sample)

        with ThreadPoolExecutor(
            max_workers=getattr(args, "workers", 4)
        ) as executor:
            futures = {
                executor.submit(run_sample, sample) for sample in pending_samples
            }
            for future in as_completed(futures):
                futures.remove(future)
                key, record, trajectory_dict, error_traceback = future.result()
                if error_traceback is not None:
                    print(error_traceback, file=sys.stderr, end="")
                if record["status"] == "success" and trajectory_dict is not None:
                    record["trajectory"] = write_trajectory(
                        run_dir, key, trajectory_dict
                    )
                append_record(records_path, record)
                latest[key] = record
                progress.update(1)
                progress.set_postfix_str(
                    f"id={key} {record['status']} "
                    f"rounds={record['agent_rounds']} "
                    f"tokens={record['token_usage']['total_tokens']} "
                    f"time={record['elapsed_seconds']:.1f}s"
                )
                print(
                    f"Sample {key}: {record['status']}, "
                    f"rounds={record['agent_rounds']}, "
                    f"tokens={record['token_usage']['total_tokens']}, "
                    f"elapsed={record['elapsed_seconds']:.3f}s"
                )
                del trajectory_dict, future
        progress.close()

        predictions = {
            key: latest.get(key, {}).get("prediction", "") for key in selected_keys
        }
        write_json(run_dir / "predictions.json", predictions)
        metrics = compute_metrics(samples, latest)
        metrics["run_statistics"] = aggregate_task_metrics(selected_keys, latest)
        metrics['evolution_usage'] = aggregate_evolution_usage(selected_keys, latest)
        write_json(run_dir / "metrics.json", metrics)
        failures = sum(
            latest.get(key, {}).get("status") != "success" for key in selected_keys
        )
        write_report(
            run_dir,
            agent_name,
            args.mode,
            model_name,
            samples,
            latest,
            metrics,
        )
        print(f"Report: {run_dir / 'report.md'}")
        print(f"Run directory: {run_dir}")
        print(f"Completed: {len(samples) - failures}/{len(samples)}, failures: {failures}")
        return 1 if failures and failures == len(samples) else 0


def main(argv: Optional[list[str]] = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
