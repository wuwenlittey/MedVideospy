import argparse
import copy
import json
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from time import perf_counter
from typing import Optional

from tqdm import tqdm

from benchmark.lvbench.utils import (
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
from benchmark.streaming_cutoff import QueryCutoffVideoReader


ResolveVideo = Callable[[dict], Path]


def _choice_line(letter: str, text: str) -> str:
    value = str(text).strip()
    stripped = re.sub(r"^[A-Da-d][\.\:\)]\s*", "", value, count=1)
    prefix = f"{letter}."
    if value[:2].upper() in {f"{letter}.", f"{letter})"}:
        return value
    if stripped != value:
        return f"{prefix} {stripped}"
    return f"{prefix} {value}"


def build_question(sample: dict) -> str:
    lines = [sample["question"], ""]
    for letter in sample["choices"]:
        lines.append(_choice_line(letter, sample["choices"][letter]))
    query_time = float(sample["query_time"])
    lines.extend(
        [
            "",
            f"Current query time: {query_time:.2f} seconds.",
            "Only video up to this timestamp is available. Do not assume future events.",
            "",
            "Output requirement: Return exactly one uppercase option letter from the listed options.",
        ]
    )
    return "\n".join(lines)


def normalize_answer(value: str) -> str:
    text = str(value).strip().upper()
    direct = re.fullmatch(r"\(?\s*([A-D])\s*\)?[\s.,:;-]*", text)
    if direct:
        return direct.group(1)
    candidates = re.findall(r"\b([A-D])\b", text)
    return candidates[-1] if candidates else ""


def letter_from_index(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= 3:
        raise ValueError(f"Choice index must be 0-3, got {index!r}")
    return "ABCD"[index]


def choices_from_options(options) -> dict[str, str]:
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("StreamEQA evaluation requires four options.")
    return {letter: str(option) for letter, option in zip("ABCD", options)}


def _manifest_values(path: str | Path) -> list[str]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("sample_ids", value.get("question_ids"))
    if not isinstance(value, list):
        raise ValueError("Streaming sample manifest must be a JSON list of IDs.")
    return [str(item) for item in value]


def select_samples(
    samples: list[dict],
    mode: str,
    manifest_path: str | Path | None,
    smoke_per_category: int = 2,
    lite_per_category: int = 10,
) -> list[dict]:
    if manifest_path:
        selected = _manifest_values(manifest_path)
        samples_by_key = {item["sample_key"]: item for item in samples}
        unknown = [key for key in selected if key not in samples_by_key]
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
        return [samples_by_key[key] for key in selected]
    if mode == "full":
        return samples
    if mode == "lite":
        limit = lite_per_category
    else:
        limit = smoke_per_category
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["category"]].append(sample)
    selected = []
    for category in sorted(grouped):
        selected.extend(grouped[category][:limit])
    return selected


def _record_failed(record: dict) -> bool:
    """A failed model/tool attempt has no usable multiple-choice prediction."""
    return (
        record.get("status") != "success"
        or not str(record.get("prediction", "")).strip()
    )


def has_task_metrics(record: dict) -> bool:
    """Whether a persisted record has the metrics required for safe resume."""
    return (
        isinstance(record, dict)
        and "agent_rounds" in record
        and isinstance(record.get("token_usage"), dict)
        and "elapsed_seconds" in record
    )


def compute_metrics(samples: list[dict], records: dict[str, dict]) -> dict:
    totals = defaultdict(int)
    correct = defaultdict(int)
    valid_totals = defaultdict(int)
    valid_correct = defaultdict(int)
    overall = 0
    valid_overall = 0
    failed = 0
    for sample in samples:
        record = records.get(sample["sample_key"], {})
        is_failed = _record_failed(record)
        is_correct = record.get("prediction") == sample["answer"] and not is_failed
        overall += int(is_correct)
        totals[sample["category"]] += 1
        correct[sample["category"]] += int(is_correct)
        failed += int(is_failed)
        if not is_failed:
            valid_overall += int(is_correct)
            valid_totals[sample["category"]] += 1
            valid_correct[sample["category"]] += int(is_correct)
    by_category = {
        key: {
            "accuracy": correct[key] / total,
            "valid_accuracy": (
                valid_correct[key] / valid_totals[key] if valid_totals[key] else 0.0
            ),
            "correct": correct[key],
            "total": total,
            "valid_correct": valid_correct[key],
            "valid_total": valid_totals[key],
        }
        for key, total in sorted(totals.items())
    }
    return {
        "overall": {
            "accuracy": overall / len(samples) if samples else 0.0,
            "valid_accuracy": valid_overall / (len(samples) - failed)
            if len(samples) > failed
            else 0.0,
            "correct": overall,
            "total": len(samples),
            "valid_correct": valid_overall,
            "valid_total": len(samples) - failed,
            "failed": failed,
        },
        "by_category": by_category,
        "macro_average": (
            sum(item["accuracy"] for item in by_category.values()) / len(by_category)
            if by_category
            else 0.0
        ),
    }


def write_report(
    run_dir: Path,
    dataset_name: str,
    agent_name: str,
    mode: str,
    model_name: str,
    samples: list[dict],
    records: dict[str, dict],
    metrics: dict,
) -> None:
    def make_row(
        task: str,
        task_samples: list[dict],
        accuracy: float,
        valid_accuracy: float,
    ) -> list[object]:
        keys = [sample["sample_key"] for sample in task_samples]
        statistics = aggregate_task_metrics(keys, records)
        completed = sum(
            records.get(key, {}).get("status") == "success" for key in keys
        )
        failed = sum(_record_failed(records.get(key, {})) for key in keys)
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
            failed,
            f"{accuracy * 100:.1f}",
            f"{valid_accuracy * 100:.1f}",
            f"{statistics['agent_rounds']['average']:.1f}",
            f"{average_tokens:.1f}",
            f"{statistics['elapsed_seconds']['average']:.2f}",
        ]

    rows = [
        make_row(
            "Overall",
            samples,
            metrics["overall"]["accuracy"],
            metrics["overall"]["valid_accuracy"],
        )
    ]
    for category, category_metrics in metrics["by_category"].items():
        category_samples = [
            sample for sample in samples if sample["category"] == category
        ]
        rows.append(
            make_row(
                category,
                category_samples,
                category_metrics["accuracy"],
                category_metrics["valid_accuracy"],
            )
        )

    write_markdown_table(
        run_dir / "report.md",
        f"{dataset_name} Run Report",
        [
            "Task",
            "Agent",
            "Mode",
            "Model",
            "Completed",
            "Failed",
            "Acc. all (%)",
            "Acc. valid (%)",
            "Avg. rounds",
            "Avg. tokens",
            "Avg. time (s)",
        ],
        rows,
        notes=[
            f"- Macro Average all (%): {metrics['macro_average'] * 100:.1f}",
            "- Acc. all treats failed/unanswerable questions as incorrect.",
            "- Acc. valid excludes failed/unanswerable questions from the denominator.",
        ],
    )


def add_streaming_arguments(
    parser: argparse.ArgumentParser,
    *,
    qa_path: str,
    video_root: str,
) -> None:
    parser.set_defaults(agent="videospy")
    for action in parser._actions:
        if action.dest == "mode":
            action.choices = ("smoke", "lite", "full", "debug")
            break
    parser.add_argument("--qa-path", default=qa_path)
    parser.add_argument("--video-root", default=video_root)


def run_streaming_benchmark(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    samples: list[dict],
    resolve_video: ResolveVideo,
    extra_manifest: Optional[dict] = None,
    agent_class=None,
) -> int:
    if args.agent != "videospy":
        raise ValueError(f"{dataset_name} currently supports VideoSpy only.")
    selected_keys = [sample["sample_key"] for sample in samples]
    agent_name = args.agent
    config = copy.deepcopy(build_agent_config(args))
    config.setdefault("agent", {})["include_subtitles_in_question"] = False
    model_name = get_agent_model_name(config, agent_name)
    run_dir = create_run_dir(
        dataset_name,
        args.mode,
        args.output_dir,
        args.run_dir,
        agent_name,
        model_name,
    )
    progress_stream = sys.stderr
    with tee_output(run_dir / "run.log"):
        print(
            f"Starting {dataset_name}: agent={agent_name}, mode={args.mode}, "
            f"samples={len(samples)}, workers={getattr(args, 'workers', 4)}, "
            f"verbose={args.verbose}",
            flush=True,
        )
        print(f"Log file: {run_dir / 'run.log'}", flush=True)
        manifest = {
            "dataset": dataset_name,
            "agent": agent_name,
            "mode": args.mode,
            "qa_path": str(Path(args.qa_path).expanduser().resolve()),
            "video_root": str(Path(args.video_root).expanduser().resolve()),
            "sample_keys": selected_keys,
            "agent_config": safe_config(config),
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        ensure_run_manifest(run_dir, manifest)
        agent_class = agent_class or get_agent_class(agent_name)
        records_path = run_dir / "records.jsonl"
        latest = load_latest_records(records_path)
        worker_state = local()

        def run_sample(sample: dict) -> tuple[str, dict, dict | None, str | None]:
            key = sample["sample_key"]
            trajectory_dict = None
            task_agent = None
            error_traceback = None
            started_at = perf_counter()
            try:
                video_path = resolve_video(sample)
                video_key = str(video_path)
                if getattr(worker_state, "current_video", None) != video_key:
                    reader = QueryCutoffVideoReader(video_path)
                    worker_state.agent = agent_class(
                        config=config,
                        video_path=video_key,
                        subtitle_path=None,
                        output_dir=str(run_dir),
                        tools=get_agent_tools(config, agent_name),
                        verbose=args.verbose,
                        video_reader=reader,
                    )
                    worker_state.agent.subtitles = []
                    worker_state.current_video = video_key
                    worker_state.reader = reader
                worker_state.reader.set_query_time(sample["query_time"])
                task_agent = worker_state.agent
                task_agent.duration = round(worker_state.reader.duration, 2)
                task_agent.task_types = [sample["category"]]
                task_agent.task_type = sample["category"]
                trajectory = task_agent.run(build_question(sample))
                trajectory_dict = trajectory.to_dict()
                raw_prediction = trajectory_dict.get("final_answer", "")
                prediction = normalize_answer(raw_prediction)
                record = {
                    "sample_key": key,
                    "status": "success" if prediction else "failed",
                    "question": sample["question"],
                    "category": sample["category"],
                    "video_id": sample["video_id"],
                    "query_time": sample["query_time"],
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                    "answer": sample["answer"],
                }
                if not prediction:
                    record["failure_reason"] = "Model returned no parseable option letter."
            except Exception as error:
                error_traceback = traceback.format_exc()
                record = {
                    "sample_key": key,
                    "status": "failed",
                    "question": sample.get("question"),
                    "category": sample.get("category"),
                    "video_id": sample.get("video_id"),
                    "error": f"{type(error).__name__}: {error}",
                    "failure_reason": "Model/tool execution failed.",
                }
            attach_task_metrics(
                record, task_agent, trajectory_dict, perf_counter() - started_at
            )
            return key, record, trajectory_dict, error_traceback

        progress = tqdm(
            total=len(samples),
            desc=f"{dataset_name} {args.mode}",
            dynamic_ncols=True,
            file=progress_stream,
            unit="question",
        )
        pending_samples = []
        for sample in samples:
            key = sample["sample_key"]
            existing = latest.get(key, {})
            if (
                is_completed_record(existing)
                or existing.get("status") == "failed"
            ) and has_task_metrics(existing):
                progress.update(1)
                progress.set_postfix_str(f"id={key} skipped")
                continue
            pending_samples.append(sample)

        with ThreadPoolExecutor(max_workers=getattr(args, "workers", 4)) as executor:
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
        write_json(run_dir / "metrics.json", metrics)
        failures = sum(_record_failed(latest.get(key, {})) for key in selected_keys)
        write_report(
            run_dir,
            dataset_name,
            agent_name,
            args.mode,
            model_name,
            samples,
            latest,
            metrics,
        )
        print(f"Report: {run_dir / 'report.md'}")
        print(f"Run directory: {run_dir}")
        print(
            f"Completed: {len(samples) - failures}/{len(samples)}, "
            f"failures: {failures}"
        )
        # Failed questions are recorded in metrics/records.jsonl.
        # They must not fail the process, or the ablation suite stops later splits.
        return 0
