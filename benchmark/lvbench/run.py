import argparse
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
    lvbench_resources,
    safe_config,
    tee_output,
    write_json,
    write_markdown_table,
    write_trajectory,
)
from benchmark.sampling import get_sampling_stride
from benchmark.evolution_diagnostics import aggregate_evolution_usage, evolution_usage_notes
from videospy.evolving.common.memory_scope import source_context


DEFAULT_METADATA_PATH = "./data/LVBench/zai-org-LVBench/video_info.meta.jsonl"
DEFAULT_VIDEO_DIR = "./data/LVBench/AIWinter-LVBench/all_videos"
DEFAULT_SAMPLE_MANIFEST = str(
    Path(__file__).resolve().parents[2] / "experiments/data_splits/main/lvbench/test.json"
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a video Agent on LVBench.")
    add_common_arguments(parser, lvbench=True)
    parser.set_defaults(sample_manifest=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    args = parser.parse_args(argv)
    args.config_path = args.config_paths[-1] if args.config_paths else None
    return args


def load_dataset(metadata_path: Path) -> list[dict]:
    samples = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for video in map(json.loads, file):
            for question in video["qa"]:
                samples.append(
                    {
                        "uid": str(question["uid"]),
                        "video_id": video["key"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "question_types": question["question_type"],
                    }
                )
    return samples


def _manifest_values(path: str | Path) -> list:
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest = manifest.get("sample_ids", manifest.get("uids"))
    if not isinstance(manifest, list):
        raise ValueError("LVBench sample manifest must be a JSON list of sample IDs.")
    return manifest


def select_samples(
    samples: list[dict], mode: str, sample_manifest: str | Path | None = None
) -> list[dict]:
    if sample_manifest is not None:
        selected_uids = [str(uid) for uid in _manifest_values(sample_manifest)]
    else:
        selected_uids = [sample["uid"] for sample in samples]
    samples_by_uid = {sample["uid"]: sample for sample in samples}
    unknown = set(selected_uids) - samples_by_uid.keys()
    if unknown:
        raise ValueError(f"Unknown LVBench sample IDs: {sorted(unknown)}")
    stride = get_sampling_stride("lvbench", mode)
    selected_uids = selected_uids[::stride]
    return [samples_by_uid[uid] for uid in selected_uids]


def build_question(question: str) -> str:
    return (
        f"{question}\n\n"
        "Output requirement: Return exactly one uppercase option letter: A, B, "
        "C, or D. Do not include parentheses, explanation, or any other text."
    )


def normalize_answer(answer: str) -> str:
    text = answer.strip().upper()
    direct = re.fullmatch(r"\s*\(?([A-D])\)?[\s.:,\-]*", text)
    if direct:
        return direct.group(1)

    strong_candidates = []
    labelled_patterns = (
        r"\b(?:FINAL|BEST|CORRECT)\s+"
        r"(?:ANSWER|OPTION|CHOICE)(?:'S\s+LETTER)?\s*"
        r"(?:IS\s*:?|[:=])?\s*\**\s*\(?\s*([A-D])\s*\)?\s*\**",
        r"\b(?:ANSWER|OPTION|CHOICE)(?:'S\s+LETTER)?\s*"
        r"(?:IS\s*:?|[:=])\s*\**\s*\(?\s*([A-D])\s*\)?\s*\**",
    )
    for pattern in labelled_patterns:
        strong_candidates.extend(
            (match.start(1), match.group(1))
            for match in re.finditer(pattern, text)
        )

    last_line = next(
        (line for line in reversed(text.splitlines()) if line.strip()), ""
    )
    line_offset = text.rfind(last_line)
    formatted_line = re.fullmatch(
        r"\s*\**\s*(?:"
        r"\(\s*([A-D])\s*\)|"
        r"\[\s*([A-D])\s*\]|"
        r"([A-D])\s*[.):\-]"
        r")\s*\**(?:\s+.+)?\s*",
        last_line,
    )
    if formatted_line:
        option = next(group for group in formatted_line.groups() if group)
        strong_candidates.append((line_offset, option))
    else:
        bare_line = re.fullmatch(r"\s*\**\s*([A-D])\s*\**[.,:]?\s*", last_line)
        if bare_line:
            strong_candidates.append((line_offset, bare_line.group(1)))

    if strong_candidates:
        return max(strong_candidates, key=lambda candidate: candidate[0])[1]

    candidates = set(re.findall(r"\b([A-D])\b", text))
    return next(iter(candidates)) if len(candidates) == 1 else ""


def compute_metrics(samples: list[dict], records: dict[str, dict]) -> dict:
    totals = defaultdict(int)
    correct = defaultdict(int)
    overall_correct = 0
    for sample in samples:
        prediction = records.get(sample["uid"], {}).get("prediction", "")
        is_correct = prediction == sample["answer"]
        overall_correct += int(is_correct)
        for question_type in sample["question_types"]:
            totals[question_type] += 1
            correct[question_type] += int(is_correct)
    count = len(samples)
    return {
        "overall": {
            "accuracy": overall_correct / count if count else 0.0,
            "correct": overall_correct,
            "total": count,
        },
        "by_question_type": {
            question_type: {
                "accuracy": correct[question_type] / total,
                "correct": correct[question_type],
                "total": total,
            }
            for question_type, total in sorted(totals.items())
        },
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
    def make_row(
        task: str, task_samples: list[dict], accuracy: float
    ) -> list[object]:
        keys = [sample["uid"] for sample in task_samples]
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
    for task, task_metrics in metrics["by_question_type"].items():
        task_samples = [
            sample for sample in samples if task in sample["question_types"]
        ]
        rows.append(make_row(task, task_samples, task_metrics["accuracy"]))

    write_markdown_table(
        run_dir / "report.md",
        "LVBench Run Report",
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
        notes=evolution_usage_notes(aggregate_evolution_usage([s['uid'] for s in samples], records)),
    )


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    metadata_path = Path(args.metadata_path).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    samples = select_samples(
        load_dataset(metadata_path),
        args.mode,
        getattr(args, "sample_manifest", None),
    )
    selected_keys = [sample["uid"] for sample in samples]
    agent_name = getattr(args, "agent", "videoseek")
    config = build_agent_config(args)
    model_name = get_agent_model_name(config, agent_name)
    run_dir = create_run_dir(
        "lvbench", args.mode, args.output_dir, args.run_dir, agent_name, model_name
    )
    progress_stream = sys.stderr
    with lvbench_resources(run_dir), tee_output(run_dir / "run.log"):
        print(
            f"Starting LVBench: agent={agent_name}, mode={args.mode}, "
            f"samples={len(samples)}, "
            f"workers={getattr(args, 'workers', 1)}, verbose={args.verbose}",
            flush=True,
        )
        print(f"Log file: {run_dir / 'run.log'}", flush=True)
        try:
            return _execute_benchmark(
                args=args,
                agent_name=agent_name,
                agent_class=agent_class,
                config=config,
                metadata_path=metadata_path,
                progress_stream=progress_stream,
                run_dir=run_dir,
                samples=samples,
                selected_keys=selected_keys,
                video_dir=video_dir,
            )
        except Exception:
            traceback.print_exc()
            raise


def _execute_benchmark(
    args: argparse.Namespace,
    agent_name: str,
    agent_class,
    config: dict,
    metadata_path: Path,
    progress_stream,
    run_dir: Path,
    samples: list[dict],
    selected_keys: list[str],
    video_dir: Path,
) -> int:
    ensure_run_manifest(
        run_dir,
        {
            "dataset": "lvbench",
            "agent": agent_name,
            "mode": args.mode,
            "metadata_path": str(metadata_path),
            "video_dir": str(video_dir),
            "sample_keys": selected_keys,
            "agent_config": safe_config(config),
        },
    )

    if agent_class is None:
        agent_class = get_agent_class(agent_name)

    records_path = run_dir / "records.jsonl"
    latest = load_latest_records(records_path)
    worker_state = local()

    def run_sample(sample: dict) -> tuple[str, dict, dict | None, str | None]:
        uid = sample["uid"]
        task_agent = None
        trajectory_dict = None
        error_traceback = None
        started_at = perf_counter()
        try:
            video_path = video_dir / f"{sample['video_id']}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Video not found: {video_path}")
            if sample["video_id"] != getattr(
                worker_state, "current_video_id", None
            ):
                worker_state.agent = agent_class(
                    config=config,
                    video_path=str(video_path),
                    subtitle_path=None,
                    output_dir=str(run_dir),
                    tools=get_agent_tools(config, agent_name),
                    verbose=args.verbose,
                )
                worker_state.current_video_id = sample["video_id"]
            task_agent = worker_state.agent
            if agent_name == "videospy":
                task_agent.task_types = list(sample["question_types"])
                task_agent.task_type = "|".join(task_agent.task_types)
                task_agent.memory_context = {**source_context('lvbench', sample['video_id']),
                                             'sample_id': uid, 'query': sample['question']}
            trajectory = task_agent.run(build_question(sample["question"]))
            trajectory_dict = trajectory.to_dict()
            raw_prediction = trajectory_dict.get("final_answer", "")
            record = {
                "sample_key": uid,
                "status": "success",
                "video_id": sample["video_id"],
                "question_types": sample["question_types"],
                "raw_prediction": raw_prediction,
                "prediction": normalize_answer(raw_prediction),
                "answer": sample["answer"],
            }
        except Exception as error:
            error_traceback = traceback.format_exc()
            record = {
                "sample_key": uid,
                "status": "error",
                "video_id": sample["video_id"],
                "error": f"{type(error).__name__}: {error}",
            }
        attach_task_metrics(
            record,
            task_agent,
            trajectory_dict,
            perf_counter() - started_at,
        )
        return uid, record, trajectory_dict, error_traceback

    progress = tqdm(
        total=len(samples),
        desc=f"LVBench {args.mode}",
        dynamic_ncols=True,
        file=progress_stream,
        unit="question",
    )
    pending_samples = []
    for sample in samples:
        uid = sample["uid"]
        if is_completed_record(latest.get(uid, {})):
            progress.update(1)
            progress.set_postfix_str(f"uid={uid} skipped")
            continue
        pending_samples.append(sample)

    with ThreadPoolExecutor(
        max_workers=getattr(args, "workers", 1)
    ) as executor:
        futures = {
            executor.submit(run_sample, sample) for sample in pending_samples
        }
        for future in as_completed(futures):
            futures.remove(future)
            uid, record, trajectory_dict, error_traceback = future.result()
            if error_traceback is not None:
                print(error_traceback, file=sys.stderr, end="")
            if record["status"] == "success" and trajectory_dict is not None:
                record["trajectory"] = write_trajectory(
                    run_dir, uid, trajectory_dict
                )
            append_record(records_path, record)
            latest[uid] = record
            progress.update(1)
            progress.set_postfix_str(
                f"uid={uid} {record['status']} "
                f"rounds={record['agent_rounds']} "
                f"tokens={record['token_usage']['total_tokens']} "
                f"time={record['elapsed_seconds']:.1f}s"
            )
            print(
                f"Sample {uid}: {record['status']}, "
                f"rounds={record['agent_rounds']}, "
                f"tokens={record['token_usage']['total_tokens']}, "
                f"elapsed={record['elapsed_seconds']:.3f}s"
            )
            del trajectory_dict, future
    progress.close()

    predictions = {
        uid: latest.get(uid, {}).get("prediction", "") for uid in selected_keys
    }
    write_json(run_dir / "predictions.json", predictions)
    metrics = compute_metrics(samples, latest)
    metrics["run_statistics"] = aggregate_task_metrics(selected_keys, latest)
    metrics['evolution_usage'] = aggregate_evolution_usage(selected_keys, latest)
    write_json(run_dir / "metrics.json", metrics)
    failures = sum(latest.get(uid, {}).get("status") != "success" for uid in selected_keys)
    write_report(
        run_dir,
        agent_name,
        args.mode,
        get_agent_model_name(config, agent_name),
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
