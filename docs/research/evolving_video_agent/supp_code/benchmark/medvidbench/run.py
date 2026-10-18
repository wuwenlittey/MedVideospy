import argparse
import copy
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Optional

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.evolution_diagnostics import aggregate_evolution_usage, evolution_usage_notes
from benchmark.medvidbench.utils import (
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
    write_trajectory,
)
from benchmark.medvidbench.image_sequence import ImageSequenceReader
from benchmark.sampling import get_sampling_stride
from videospy.evolving.common.memory_scope import medvidbench_source_context


DEFAULT_DATASET = "test"
DATASET_CONFIGS = {
    "test": {
        "metadata_path": (
            "./data/MedVidBench/MedVidU_ECCV2026_TrainVal/"
            "medvidu_eccv2026_trainval.json"
        ),
        "frame_root": "./data/MedVidBench/MedVidU_ECCV2026_TrainVal/valdata",
    },
    "leaderboard_test": {
        "metadata_path": (
            "./data/MedVidBench/MedVidBench-data/cleaned_test_data_11_04.json"
        ),
        "frame_root": "./data/MedVidBench/MedVidBench-data/testdata",
    },
}
DEFAULT_METADATA_PATH = DATASET_CONFIGS[DEFAULT_DATASET]["metadata_path"]
DEFAULT_FRAME_ROOT = DATASET_CONFIGS[DEFAULT_DATASET]["frame_root"]
DEFAULT_SAMPLE_MANIFEST = str(
    Path(__file__).resolve().parents[2] / "experiments/data_splits/main/medvidbench/test.json"
)

ANSWER_FORMATS = {
    "tal": (
        'Return only the time span(s), for example: "1.1-3.0 seconds." '
        'For multiple spans, use: "1.1-3.0, 5.0-7.0 seconds."'
    ),
    "stg": (
        "Return only the requested timestamped bounding boxes in this format: "
        '"<timestamp> seconds: [x1, y1, x2, y2]". Use exactly the timestamps '
        "requested in the question. Do not add, omit, or replace timestamps."
    ),
    "next_action": "Return only the next action label.",
    "skill_assessment": (
        "Return only: "
        '"Respect for tissue: <1-5>/5, Suture/needle handling: <1-5>/5, '
        'Time and motion: <1-5>/5, Flow of operation: <1-5>/5, '
        'Overall performance: <1-5>/5, Quality of final product: <1-5>/5".'
    ),
    "cvs_assessment": (
        "Return only: "
        '"Two structures: <0-2>, Cystic plate: <0-2>, '
        'Hepatocystic triangle: <0-2>".'
    ),
}

CAPTION_FORMATS = {
    "dense_captioning": (
        "Return one event per line in this format: "
        '"1.0-29.0 seconds: event label: factual description".'
    ),
    "video_summary": "Return only one concise summary paragraph.",
    "region_caption": "Return only one concise factual description.",
}

REPORT_TASKS = {
    "tal": "TAL",
    "stg": "STG",
    "dvc": "DVC",
    "next_action": "Next Action",
    "rc": "Region Caption",
    "vs": "Video Summary",
    "skill_assessment": "Skill Assessment",
    "cvs_assessment": "CVS Assessment",
}

LEADERBOARD_METRICS = (
    ("CVS_acc", "cvs_acc"),
    ("NAP_acc", "nap_acc"),
    ("SA_acc", "sa_acc"),
    ("STG_mIoU", "stg_miou"),
    ("TAG_mIoU@0.3", "tag_miou_03"),
    ("TAG_mIoU@0.5", "tag_miou_05"),
    ("DVC_F1", "dvc_f1"),
    ("DVC_llm", "dvc_llm"),
    ("VS_llm", "vs_llm"),
    ("RC_llm", "rc_llm"),
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a video Agent on MedVidBench.")
    add_common_arguments(parser)
    parser.set_defaults(sample_manifest=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--frame-root", default=None)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a successful local test run after inference.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip caption LLM judging when --evaluate is used.",
    )
    args = parser.parse_args(argv)
    args.config_path = args.config_paths[-1] if args.config_paths else None
    return args


def load_dataset(metadata_path: Path) -> list[dict]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_manifest(path: str | Path) -> tuple[str, list[int] | None]:
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    dataset_name = DEFAULT_DATASET
    if isinstance(manifest, dict):
        dataset_name = manifest.get("dataset", DEFAULT_DATASET)
        manifest = manifest.get("sample_indices", manifest.get("indices"))
        if manifest == "all":
            manifest = None
        elif not isinstance(manifest, list):
            raise ValueError("MedVidBench sample_indices must be a list or 'all'.")
    elif not isinstance(manifest, list):
        raise ValueError("MedVidBench sample manifest must be a JSON list or object.")
    if not isinstance(dataset_name, str) or dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported MedVidBench dataset: {dataset_name}")
    if manifest is not None and any(
        not isinstance(value, int) or isinstance(value, bool) for value in manifest
    ):
        raise ValueError("MedVidBench sample indices must be integers.")
    return dataset_name, manifest


def select_indices(
    sample_count: int, mode: str, sample_manifest: str | Path | None = None
) -> list[int]:
    dataset_name, values = (
        _load_manifest(sample_manifest)
        if sample_manifest is not None
        else (DEFAULT_DATASET, None)
    )
    indices = list(range(sample_count)) if values is None else values
    invalid = [index for index in indices if index < 0 or index >= sample_count]
    if invalid:
        raise ValueError(f"Invalid MedVidBench sample indices: {invalid}")
    if dataset_name == "leaderboard_test":
        if mode != "full":
            raise ValueError("The MedVidBench leaderboard test set only supports --mode full.")
        return indices
    stride = get_sampling_stride("medvidbench", mode)
    return indices[::stride]


def resolve_dataset(args: argparse.Namespace) -> tuple[str, Path, Path]:
    sample_manifest = getattr(args, "sample_manifest", None)
    dataset_name = (
        _load_manifest(sample_manifest)[0] if sample_manifest is not None else DEFAULT_DATASET
    )
    if dataset_name == "leaderboard_test" and args.mode != "full":
        raise ValueError(
            "The MedVidBench leaderboard test set only supports --mode full. "
            "Use a public-data sample manifest for lite or debug runs."
        )
    if dataset_name == "leaderboard_test" and getattr(args, "evaluate", False):
        raise ValueError(
            "The MedVidBench leaderboard test set has no local Ground Truth. "
            "Upload submission.json to the Leaderboard instead."
        )

    defaults = DATASET_CONFIGS[dataset_name]
    metadata_path = Path(
        getattr(args, "metadata_path", None) or defaults["metadata_path"]
    ).expanduser().resolve()
    frame_root = Path(
        getattr(args, "frame_root", None) or defaults["frame_root"]
    ).expanduser().resolve()
    return dataset_name, metadata_path, frame_root


def build_question(sample: dict) -> str:
    question = next(
        message["value"]
        for message in sample["conversations"]
        if message["from"] == "human"
    )
    question = question.removeprefix("<video>\n")
    answer_format = ANSWER_FORMATS.get(sample["qa_type"])
    if answer_format is None:
        answer_format = next(
            (
                value
                for prefix, value in CAPTION_FORMATS.items()
                if sample["qa_type"].startswith(prefix)
            ),
            None,
        )
    if answer_format is None:
        return question
    return f"{question}\n\nOutput requirement: {answer_format}"


def build_submission(
    dataset: list[dict], selected_indices: list[int], records: dict[str, dict]
) -> list[dict]:
    return [
        {
            "id": dataset[index]["id"],
            "qa_type": dataset[index]["qa_type"],
            "prediction": records.get(str(index), {}).get("prediction", ""),
        }
        for index in selected_indices
    ]


def conversation_value(sample: dict, roles: set[str]) -> str:
    value = ""
    for message in sample.get("conversations", []):
        if message.get("from") in roles:
            value = message.get("value", "")
    return value.replace("<video>\n", "").replace("<video>", "")


def report_task(qa_type: str) -> str:
    if qa_type.startswith("dense_captioning"):
        return "dvc"
    if qa_type.startswith("region_caption"):
        return "rc"
    if qa_type.startswith("video_summary"):
        return "vs"
    return qa_type


def task_prompt_key(qa_type: str) -> str:
    for prefix in ("dense_captioning", "region_caption", "video_summary"):
        if qa_type.startswith(prefix):
            return prefix
    return qa_type


def build_task_agent_config(config: dict, qa_type: str, agent_name: str) -> dict:
    if agent_name != "videospy":
        return config
    task_config = copy.deepcopy(config)
    task_config.setdefault('_runtime', {})['task_type'] = task_prompt_key(qa_type)
    return task_config


def write_report(
    run_dir: Path,
    agent_name: str,
    mode: str,
    model_name: str,
    sample_keys: list[str],
    submission: list[dict],
    records: dict[str, dict],
    metrics: dict,
    dataset_name: str = DEFAULT_DATASET,
    evaluation: Optional[dict] = None,
) -> None:
    grouped_keys = {task: [] for task in REPORT_TASKS}
    for sample_key, sample in zip(sample_keys, submission):
        task = report_task(sample["qa_type"])
        if task in grouped_keys:
            grouped_keys[task].append(sample_key)

    def make_row(task: str, keys: list[str]) -> list[object]:
        statistics = aggregate_task_metrics(keys, records)
        generated = sum(
            records.get(key, {}).get("status") == "success" for key in keys
        )
        measured_tasks = statistics["measured_tasks"]
        average_tokens = (
            statistics["token_usage"]["total_tokens"] / measured_tasks
            if measured_tasks
            else 0.0
        )
        return [
            REPORT_TASKS.get(task, "Overall"),
            agent_name,
            dataset_name,
            mode,
            model_name,
            f"{generated}/{len(keys)}",
            f"{statistics['agent_rounds']['average']:.1f}",
            f"{average_tokens:.1f}",
            f"{statistics['elapsed_seconds']['average']:.2f}",
        ]

    rows = [make_row("overall", sample_keys)]
    rows.extend(
        make_row(task, grouped_keys[task])
        for task in REPORT_TASKS
        if grouped_keys[task]
    )
    evaluation = evaluation or {}
    evaluation_status = evaluation.get(
        "status", "complete" if metrics else "not_evaluated"
    )
    skip_llm_judge = evaluation.get("skip_llm_judge", False)
    missing_metrics = set(evaluation.get("missing_metrics", []))
    not_applicable_metrics = set(evaluation.get("not_applicable_metrics", []))
    requested_tasks = set(evaluation.get("tasks", []))
    llm_metric_tasks = {
        "dvc_llm": "dvc",
        "vs_llm": "vs",
        "rc_llm": "rc",
    }

    leaderboard_values = []
    for _, metric_key in LEADERBOARD_METRICS:
        if metric_key in metrics:
            leaderboard_values.append(f"{metrics[metric_key]:.4f}")
        elif (
            skip_llm_judge
            and metric_key in llm_metric_tasks
            and llm_metric_tasks[metric_key] in requested_tasks
        ):
            leaderboard_values.append("Skipped")
        elif evaluation_status == "failed" and metric_key in missing_metrics:
            leaderboard_values.append("Failed")
        elif metric_key in not_applicable_metrics:
            leaderboard_values.append("N/A")
        else:
            leaderboard_values.append("—")

    if dataset_name == "leaderboard_test":
        note = (
            "The leaderboard test set has hidden Ground Truth. Upload the complete "
            "submission.json to the MedVidBench Leaderboard for scoring."
        )
    elif evaluation_status == "failed":
        note = (
            "Local evaluation failed or produced incomplete metrics. See "
            "evaluation.json and evaluation.log."
        )
    elif metrics:
        note = (
            "Leaderboard metrics use the official evaluator. LLM configuration "
            "is recorded in evaluation.json."
        )
    else:
        note = "Metrics will be filled after local test evaluation."

    def table(columns: list[str], table_rows: list[list[object]]) -> list[str]:
        def escape(value: object) -> str:
            return str(value).replace("|", "\\|").replace("\n", "<br>")

        lines = [
            f"| {' | '.join(map(escape, columns))} |",
            f"| {' | '.join('---' for _ in columns)} |",
        ]
        lines.extend(
            f"| {' | '.join(map(escape, row))} |" for row in table_rows
        )
        return lines

    lines = ["# MedVidBench Run Report", "", "## Run Statistics", ""]
    lines.extend(
        table(
            [
                "Task",
                "Agent",
                "Split",
                "Mode",
                "Model",
                "Generated",
                "Avg. rounds",
                "Avg. tokens",
                "Avg. time (s)",
            ],
            rows,
        )
    )
    lines.extend(["", "## Leaderboard Metrics", ""])
    lines.extend(
        table(
            ["Model", *(label for label, _ in LEADERBOARD_METRICS)],
            [[model_name, *leaderboard_values]],
        )
    )
    if not_applicable_metrics:
        lines.extend(["", "N/A: no samples of this task were selected for this run."])
    if "nap_exact_acc" in metrics:
        lines.extend(["", "## Local Diagnostic Metrics", ""])
        lines.extend(
            table(
                ["Model", "NAP_exact_acc"],
                [[model_name, f"{metrics['nap_exact_acc']:.4f}"]],
            )
        )
    lines.extend(["", f"Evaluation status: `{evaluation_status}`."])
    llm = evaluation.get("llm", {})
    if llm.get("enabled"):
        lines.append(
            "Evaluation LLM: "
            f"`{llm.get('model')}` via "
            f"`{llm.get('api_base') or 'OpenAI default endpoint'}`."
        )
    elif skip_llm_judge:
        lines.append("Evaluation LLM: `skipped`.")
    for warning in evaluation.get("warnings", []):
        lines.append(f"Warning: {warning}")
    lines.extend(["", f"> {note}"])
    usage = aggregate_evolution_usage(sample_keys, records)
    lines.extend(['', *evolution_usage_notes(usage)])
    write_json(run_dir / 'metrics.json', {'task_metrics': metrics,
                                        'run_statistics': aggregate_task_metrics(sample_keys, records),
                                        'evolution_usage': usage})
    temporary_path = (run_dir / "report.md").with_suffix(".md.tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(run_dir / "report.md")


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    dataset_name, metadata_path, frame_root = resolve_dataset(args)
    dataset = load_dataset(metadata_path)
    selected_indices = select_indices(
        len(dataset),
        args.mode,
        getattr(args, "sample_manifest", None),
    )
    selected_keys = [str(index) for index in selected_indices]
    agent_name = getattr(args, "agent", "videoseek")
    config = build_agent_config(args)
    model_name = get_agent_model_name(config, agent_name)
    run_dir = create_run_dir(
        "medvidbench",
        args.mode,
        args.output_dir,
        args.run_dir,
        agent_name,
        model_name,
    )
    progress_stream = sys.stderr
    with tee_output(run_dir / "run.log"):
        print(
            f"Starting MedVidBench: dataset={dataset_name}, agent={agent_name}, "
            f"mode={args.mode}, "
            f"samples={len(selected_indices)}, "
            f"workers={getattr(args, 'workers', 4)}, verbose={args.verbose}",
            flush=True,
        )
        print(f"Log file: {run_dir / 'run.log'}", flush=True)
        try:
            status = _execute_benchmark(
                args=args,
                agent_name=agent_name,
                agent_class=agent_class,
                config=config,
                dataset_name=dataset_name,
                dataset=dataset,
                frame_root=frame_root,
                metadata_path=metadata_path,
                progress_stream=progress_stream,
                run_dir=run_dir,
                selected_indices=selected_indices,
                selected_keys=selected_keys,
            )
            if getattr(args, "evaluate", False):
                from benchmark.medvidbench.evaluate import (
                    DEFAULT_LEADERBOARD_DIR,
                    evaluate_run,
                )

                evaluation_args = argparse.Namespace(
                    run_dir=str(run_dir),
                    ground_truth_path=None,
                    leaderboard_dir=DEFAULT_LEADERBOARD_DIR,
                    tasks=None,
                    skip_llm_judge=getattr(args, "skip_llm_judge", False),
                )
                eval_status = evaluate_run(evaluation_args)
                if status == 0:
                    status = eval_status
            return status
        except Exception:
            traceback.print_exc()
            raise


def _execute_benchmark(
    args: argparse.Namespace,
    agent_name: str,
    agent_class,
    config: dict,
    dataset_name: str,
    dataset: list[dict],
    frame_root: Path,
    metadata_path: Path,
    progress_stream,
    run_dir: Path,
    selected_indices: list[int],
    selected_keys: list[str],
) -> int:
    ensure_run_manifest(
        run_dir,
        {
            "dataset": "medvidbench",
            "dataset_variant": dataset_name,
            "agent": agent_name,
            "mode": args.mode,
            "metadata_path": str(metadata_path),
            "frame_root": str(frame_root),
            "sample_keys": selected_keys,
            "agent_config": safe_config(config),
        },
    )

    if agent_class is None:
        agent_class = get_agent_class(agent_name)

    records_path = run_dir / "records.jsonl"
    latest = load_latest_records(records_path)

    def run_sample(
        source_index: int,
    ) -> tuple[str, dict, dict | None, str | None]:
        sample_key = str(source_index)
        sample = dataset[source_index]
        task_agent = None
        trajectory_dict = None
        error_traceback = None
        started_at = perf_counter()
        try:
            reader = ImageSequenceReader(
                frame_paths=sample["video"],
                fps=float(sample["metadata"]["fps"]),
                frame_root=frame_root,
                rc_info=sample.get("RC_info") if sample.get("is_RC") else None,
            )
            video_id = sample["metadata"].get(
                "video_id", f"sample_{source_index}"
            )
            task_config = build_task_agent_config(
                config, sample["qa_type"], agent_name
            )
            task_agent = agent_class(
                config=task_config,
                video_path=f"{video_id}.frames",
                subtitle_path=None,
                output_dir=str(run_dir),
                tools=get_agent_tools(task_config, agent_name),
                verbose=args.verbose,
                video_reader=reader,
            )
            task_agent.memory_context = {
                **medvidbench_source_context(sample), 'sample_id': sample_key,
                'query': conversation_value(sample, {'human', 'user'}).strip().removeprefix(
                    'You are an expert in understanding procedural healthcare videos.').strip(),
            }
            trajectory = task_agent.run(build_question(sample))
            trajectory_dict = trajectory.to_dict()
            prediction = trajectory_dict.get("final_answer", "")
            record = {
                "sample_key": sample_key,
                "status": "success",
                "id": sample["id"],
                "qa_type": sample["qa_type"],
                "video_id": video_id,
                "memory_source": medvidbench_source_context(sample),
                "prediction": prediction,
            }
        except Exception as error:
            error_traceback = traceback.format_exc()
            record = {
                "sample_key": sample_key,
                "status": "error",
                "id": sample.get("id", ""),
                "qa_type": sample.get("qa_type", ""),
                "video_id": (sample.get("metadata") or {}).get("video_id", ""),
                "error": f"{type(error).__name__}: {error}",
            }
        if dataset_name == "test":
            record.update(
                {
                    "data_source": sample.get(
                        "data_source", sample.get("dataset_name", "")
                    ),
                    "question": conversation_value(sample, {"human", "user"}),
                    "ground_truth": conversation_value(
                        sample, {"gpt", "assistant"}
                    ),
                    "struc_info": sample.get("struc_info", []),
                }
            )
        attach_task_metrics(
            record,
            task_agent,
            trajectory_dict,
            perf_counter() - started_at,
        )
        return sample_key, record, trajectory_dict, error_traceback

    progress = tqdm(
        total=len(selected_indices),
        desc=f"MedVidBench {args.mode}",
        dynamic_ncols=True,
        file=progress_stream,
        unit="question",
    )
    pending_indices = []
    for source_index in selected_indices:
        sample_key = str(source_index)
        if is_completed_record(latest.get(sample_key, {})):
            progress.update(1)
            progress.set_postfix_str(f"index={sample_key} skipped")
            continue
        pending_indices.append(source_index)

    with ThreadPoolExecutor(
        max_workers=getattr(args, "workers", 4)
    ) as executor:
        futures = {
            executor.submit(run_sample, source_index)
            for source_index in pending_indices
        }
        for future in as_completed(futures):
            futures.remove(future)
            sample_key, record, trajectory_dict, error_traceback = future.result()
            if error_traceback is not None:
                print(error_traceback, file=sys.stderr, end="")
            if record["status"] == "success" and trajectory_dict is not None:
                record["trajectory"] = write_trajectory(
                    run_dir, sample_key, trajectory_dict
                )
            append_record(records_path, record)
            latest[sample_key] = record
            progress.update(1)
            progress.set_postfix_str(
                f"index={sample_key} {record['status']} "
                f"rounds={record['agent_rounds']} "
                f"tokens={record['token_usage']['total_tokens']} "
                f"time={record['elapsed_seconds']:.1f}s"
            )
            print(
                f"Sample {sample_key}: {record['status']}, "
                f"rounds={record['agent_rounds']}, "
                f"tokens={record['token_usage']['total_tokens']}, "
                f"elapsed={record['elapsed_seconds']:.3f}s"
            )
            del trajectory_dict, future
    progress.close()

    submission = build_submission(dataset, selected_indices, latest)
    write_json(run_dir / "submission.json", submission)
    failures = sum(
        latest.get(str(index), {}).get("status") != "success"
        for index in selected_indices
    )
    evaluation_path = run_dir / "evaluation.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file()
        else {}
    )
    local_metrics = (
        evaluation.get("metrics", {}) if dataset_name == "test" else {}
    )
    write_report(
        run_dir,
        agent_name,
        args.mode,
        get_agent_model_name(config, agent_name),
        selected_keys,
        submission,
        latest,
        local_metrics,
        dataset_name,
        evaluation,
    )
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Run directory: {run_dir}")
    print(
        f"Completed: {len(selected_indices) - failures}/{len(selected_indices)}, "
        f"failures: {failures}"
    )
    return 1 if failures and failures == len(selected_indices) else 0


def main(argv: Optional[list[str]] = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
