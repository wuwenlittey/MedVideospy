import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.medvidbench.run import conversation_value, report_task, write_report
from benchmark.medvidbench.utils import (
    append_record,
    get_agent_model_name,
    load_latest_records,
    write_json,
)


DEFAULT_LEADERBOARD_DIR = "./data/MedVidBench/MedVidBench-Leaderboard"
DEFAULT_EVALUATION_MODEL = "gpt-4.1"
CONFIG_DIR = Path(__file__).with_name("config")
LOCAL_CONFIG_PATH = CONFIG_DIR / "evaluation.yml"
TASKS = (
    "dvc",
    "tal",
    "next_action",
    "stg",
    "rc",
    "vs",
    "skill_assessment",
    "cvs_assessment",
)
TASK_METRICS = {
    "dvc": ("dvc_f1", "dvc_llm"),
    "tal": ("tag_miou_03", "tag_miou_05"),
    "next_action": ("nap_acc",),
    "stg": ("stg_miou",),
    "rc": ("rc_llm",),
    "vs": ("vs_llm",),
    "skill_assessment": ("sa_acc",),
    "cvs_assessment": ("cvs_acc",),
}
LLM_METRICS = {"dvc_llm", "vs_llm", "rc_llm"}
METRIC_PATTERNS = {
    "cvs_acc": r"^\s*component_balanced_accuracy:\s*([-+]?\d*\.?\d+)",
    "nap_acc": r"^\s*accuracy:\s*([-+]?\d*\.?\d+)",
    "sa_acc": r"^\s*aspect_balanced_accuracy:\s*([-+]?\d*\.?\d+)",
    "stg_miou": r"^\s*mean_iou:\s*([-+]?\d*\.?\d+)",
    "tag_miou_03": r"^\s*mIoU@0\.3:\s*([-+]?\d*\.?\d+)",
    "tag_miou_05": r"^\s*mIoU@0\.5:\s*([-+]?\d*\.?\d+)",
    "dvc_f1": r"^\s*temporal_f1:\s*([-+]?\d*\.?\d+)",
    "dvc_llm": r"^\s*caption_score:\s*([-+]?\d*\.?\d+)",
    "vs_llm": r"VS - Overall Evaluation[\s\S]*?score:\s*([-+]?\d*\.?\d+)",
    "rc_llm": r"RC - Overall Evaluation[\s\S]*?score:\s*([-+]?\d*\.?\d+)",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed MedVidBench run with the official code."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--ground-truth-path",
        default=None,
        help="Override the local test Ground Truth path stored in run.json.",
    )
    parser.add_argument("--leaderboard-dir", default=DEFAULT_LEADERBOARD_DIR)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Compute deterministic metrics only.",
    )
    return parser.parse_args(argv)


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_llm_config() -> dict:
    config_path = LOCAL_CONFIG_PATH.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"MedVidBench config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"MedVidBench config must be a mapping: {config_path}")
    evaluation = config.get("evaluation", {})
    llm = evaluation.get("llm", {}) if isinstance(evaluation, dict) else None
    if not isinstance(llm, dict):
        raise ValueError(f"evaluation.llm must be a mapping: {config_path}")

    defaults = {"max_workers": 2, "enable_thinking": False, "max_completion_tokens": 2048, "timeout": 120}
    unknown_keys = llm.keys() - {"model", "api_base", "api_key", *defaults}
    if unknown_keys:
        raise ValueError(
            f"Unknown evaluation.llm fields: {sorted(unknown_keys)}. "
            "Use model (not model_name), api_base, api_key, max_workers, "
            "enable_thinking, max_completion_tokens, and timeout."
        )

    invalid_keys = [
        key
        for key, value in llm.items()
        if key in {"model", "api_base", "api_key"} and value is not None and not isinstance(value, str)
    ]
    if invalid_keys:
        raise ValueError(
            f"evaluation.llm fields must be strings: {sorted(invalid_keys)}"
        )

    settings = {key: llm.get(key, value) for key, value in defaults.items()}
    for key in ("max_workers", "max_completion_tokens"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"evaluation.llm.{key} must be a positive integer.")
    if type(settings["enable_thinking"]) is not bool:
        raise ValueError("evaluation.llm.enable_thinking must be a boolean.")
    if type(settings["timeout"]) not in (int, float) or not 0 < settings["timeout"] < float("inf"):
        raise ValueError("evaluation.llm.timeout must be a positive finite number.")
    return {
        **settings,
        "config_path": str(config_path),
        "model": llm.get("model") or DEFAULT_EVALUATION_MODEL,
        "api_base": llm.get("api_base"),
        "api_key": llm.get("api_key"),
    }


def select_ground_truth(ground_truth: list[dict], sample_keys: list[str]) -> list[dict]:
    if not isinstance(ground_truth, list):
        raise ValueError("Ground truth must be a JSON array.")
    indices = [int(key) for key in sample_keys]
    invalid = [index for index in indices if index < 0 or index >= len(ground_truth)]
    if invalid:
        raise ValueError(f"Ground-truth indices out of range: {invalid[:10]}")
    return [ground_truth[index] for index in indices]


def validate_alignment(submission: list[dict], ground_truth: list[dict]) -> None:
    if not isinstance(submission, list):
        raise ValueError("submission.json must be a JSON array.")
    if len(submission) != len(ground_truth):
        raise ValueError(
            f"Submission has {len(submission)} samples, but selected ground truth has "
            f"{len(ground_truth)}."
        )

    for index, (prediction, reference) in enumerate(zip(submission, ground_truth)):
        missing = {"id", "qa_type", "prediction"} - prediction.keys()
        if missing:
            raise ValueError(
                f"Submission sample {index} is missing fields: {sorted(missing)}"
            )
        reference_id = reference.get("id")
        if reference_id and prediction["id"] != reference_id:
            raise ValueError(f"Sample {index} id does not match ground truth.")
        reference_type = reference.get("qa_type")
        if reference_type and prediction["qa_type"] != reference_type:
            raise ValueError(f"Sample {index} qa_type does not match ground truth.")


def normalize_action_label(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def compute_nap_exact_accuracy(
    submission: list[dict], ground_truth: list[dict]
) -> float | None:
    comparisons = [
        (
            normalize_action_label(prediction.get("prediction", "")),
            normalize_action_label(
                conversation_value(reference, {"gpt", "assistant"})
            ),
        )
        for prediction, reference in zip(submission, ground_truth)
        if prediction.get("qa_type") == "next_action"
    ]
    if not comparisons:
        return None
    correct = sum(prediction == reference for prediction, reference in comparisons)
    return correct / len(comparisons)


def enrich_test_records(
    records_path: Path,
    latest_records: dict[str, dict],
    sample_keys: list[str],
    submission: list[dict],
    ground_truth: list[dict],
) -> dict[str, dict]:
    comparison_fields = {"question", "ground_truth", "struc_info"}
    for sample_key, prediction, reference in zip(
        sample_keys, submission, ground_truth
    ):
        current = latest_records[sample_key]
        if comparison_fields <= current.keys():
            continue
        enriched = {
            **current,
            "prediction": prediction["prediction"],
            "data_source": reference.get(
                "data_source", reference.get("dataset_name", "")
            ),
            "question": conversation_value(reference, {"human", "user"}),
            "ground_truth": conversation_value(
                reference, {"gpt", "assistant"}
            ),
            "struc_info": reference.get("struc_info", []),
        }
        append_record(records_path, enriched)
        latest_records[sample_key] = enriched
    return latest_records


def parse_official_metrics(output: str) -> dict[str, float]:
    marker = re.compile(r"^LEADERBOARD METRICS SUMMARY$", re.MULTILINE)
    end_marker = "END LEADERBOARD METRICS SUMMARY"
    starts = list(marker.finditer(output))
    if not starts:
        return {}
    start = starts[-1].end()
    end = output.find(end_marker, start)
    output = output[start : end if end >= 0 else None]

    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def judge_call_results(output):
    """Use per-pair outcomes, including exhausted retries and parsing failures."""
    task_metrics = {"dense_captioning": "dvc_llm", "region_caption": "rc_llm", "video_summary": "vs_llm"}
    calls = {}
    invalid = set()
    events = re.findall(r"^VIDEOSPY_JUDGE_RESULT (.+)$", output, re.MULTILINE)
    for line in events:
        event = json.loads(line)
        task = event["task"]
        call = calls.setdefault(task, {"task": task, "successful": 0, "total": 0, "errors": []})
        call["total"] += 1
        call["successful"] += int(event["successful"])
        if not event["successful"]:
            invalid.update({task_metrics[task]} if task in task_metrics else LLM_METRICS)
            if event.get("error") and event["error"] not in call["errors"]:
                call["errors"].append(event["error"])
    if not events:
        # Older external runners have no task-tagged records. Do not publish an
        # incomplete aggregate when its affected metric cannot be identified.
        for index, (completed, total) in enumerate(re.findall(
                r"LLM Judge completed:\s*(\d+)/(\d+) successful(?: API calls)?", output)):
            calls[index] = {"successful": int(completed), "total": int(total)}
            if int(completed) < int(total):
                invalid.update(LLM_METRICS)
    for call in calls.values():
        call["failed"] = call["total"] - call["successful"]
        call["success_rate"] = call["successful"] / call["total"] if call["total"] else 0.0
    return list(calls.values()), invalid


def run_official_evaluation(
    command: list[str], cwd: Path, log_path: Path, env: dict[str, str]
) -> str:
    output = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                output.append(line)
            return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Official evaluation failed with exit code {return_code}. See {log_path}."
        )
    return "".join(output)


def evaluate_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    leaderboard_dir = Path(args.leaderboard_dir).expanduser().resolve()
    evaluation_script = leaderboard_dir / "evaluation" / "evaluate_predictions.py"
    official_runner = Path(__file__).with_name("official_runner.py")
    preflight_errors: list[str] = []
    warnings: list[str] = []

    manifest = load_json(run_dir / "run.json")
    if manifest.get("dataset") != "medvidbench":
        raise ValueError(f"Not a MedVidBench run directory: {run_dir}")
    dataset_name = manifest.get("dataset_variant")
    if dataset_name == "leaderboard_test":
        raise ValueError(
            "The MedVidBench leaderboard test set has no local Ground Truth. "
            "Upload submission.json to the Leaderboard instead."
        )
    ground_truth_argument = getattr(args, "ground_truth_path", None)
    if ground_truth_argument:
        ground_truth_path = Path(ground_truth_argument).expanduser().resolve()
    elif dataset_name == "test":
        ground_truth_path = Path(manifest["metadata_path"]).expanduser().resolve()
    else:
        raise ValueError(
            "This legacy run does not identify its dataset. Pass "
            "--ground-truth-path explicitly."
        )

    records_path = run_dir / "records.jsonl"
    latest_records = load_latest_records(records_path)

    # Filter samples that failed to run so they do not drag evaluation scores down.
    all_sample_keys: list[str] = list(manifest["sample_keys"])
    selected_ground_truth = select_ground_truth(
        load_json(ground_truth_path), all_sample_keys
    )
    selected_tasks = {report_task(sample["qa_type"]) for sample in selected_ground_truth}
    success_keys = [
        key
        for key in all_sample_keys
        if latest_records.get(key, {}).get("status") == "success"
    ]
    if len(success_keys) != len(all_sample_keys):
        skipped = len(all_sample_keys) - len(success_keys)
        warnings.append(
            f"Skipping {skipped} failed samples in evaluation; using "
            f"{len(success_keys)}/{len(all_sample_keys)} completed samples only."
        )
        if not success_keys:
            raise ValueError(
                "No successful samples available for evaluation; at least one "
                "sample must complete successfully."
            )
        manifest = {**manifest, "sample_keys": success_keys}

    submission_path = run_dir / "submission.json"
    submission_all = load_json(submission_path)
    # Align submission order with the original (full) manifest, then filter to
    # the same successful sample keys.
    key_to_submission = {
        key: entry for key, entry in zip(all_sample_keys, submission_all)
    }
    submission = [key_to_submission[key] for key in manifest["sample_keys"]]
    key_to_ground_truth = dict(zip(all_sample_keys, selected_ground_truth))
    ground_truth = [key_to_ground_truth[key] for key in manifest["sample_keys"]]
    validate_alignment(submission, ground_truth)
    latest_records = enrich_test_records(
        records_path,
        latest_records,
        manifest["sample_keys"],
        submission,
        ground_truth,
    )
    if not evaluation_script.is_file():
        raise FileNotFoundError(f"Official evaluator not found: {evaluation_script}")

    # Infer expectations before filtering failures, so a task whose samples all
    # failed cannot be mistaken for a task absent from the selected subset.
    requested_tasks = list(args.tasks or [task for task in TASKS if task in selected_tasks])
    successful_tasks = {report_task(sample["qa_type"]) for sample in ground_truth}
    failed_tasks = sorted((set(requested_tasks) & selected_tasks) - successful_tasks)
    not_applicable_metrics = [
        metric
        for task in TASKS
        if task not in selected_tasks and task not in requested_tasks
        for metric in TASK_METRICS[task]
    ]
    expected_metrics = [
        metric
        for task in requested_tasks
        for metric in TASK_METRICS[task]
        if not (args.skip_llm_judge and metric in LLM_METRICS)
    ]
    llm_tasks = {"dvc", "rc", "vs"} & set(requested_tasks)
    llm_enabled = bool(llm_tasks) and not args.skip_llm_judge
    llm_config = load_evaluation_llm_config()
    llm_model = llm_config["model"]
    llm_api_base = llm_config["api_base"]
    if llm_enabled and not llm_config["api_key"]:
        preflight_errors.append(
            f"Evaluation API key is missing. Set evaluation.llm.api_key in "
            f"{llm_config['config_path']}."
        )

    log_path = run_dir / "evaluation.log"
    output = ""
    execution_errors = []
    if preflight_errors:
        log_path.write_text("\n".join(preflight_errors) + "\n", encoding="utf-8")
    else:
        environment = os.environ.copy()
        if llm_config["api_key"]:
            environment["OPENAI_API_KEY"] = llm_config["api_key"]
        with tempfile.TemporaryDirectory(prefix="medvidbench-eval-") as directory:
            selected_ground_truth_path = Path(directory) / "ground_truth.json"
            write_json(selected_ground_truth_path, ground_truth)
            # When some samples were skipped (failed runs), also write the
            # filtered submission so lengths stay aligned with ground truth.
            effective_submission_path = submission_path
            if len(submission_all) != len(submission):
                effective_submission_path = Path(directory) / "submission.json"
                write_json(effective_submission_path, submission)
            command = [
                sys.executable,
                str(official_runner),
                "--model",
                llm_model,
            ]
            if llm_api_base:
                command.extend(["--api-base", llm_api_base])
            for key in ("max_workers", "max_completion_tokens", "timeout"):
                command.extend(["--" + key.replace("_", "-"), str(llm_config[key])])
            if llm_config["enable_thinking"]:
                command.append("--enable-thinking")
            command.extend(
                [
                    str(evaluation_script),
                    str(effective_submission_path),
                    "--ground-truth",
                    str(selected_ground_truth_path),
                    "--grouping",
                    "overall",
                ]
            )
            if args.tasks:
                command.extend(["--tasks", *args.tasks])
            if args.skip_llm_judge:
                command.append("--skip-llm-judge")
            try:
                output = run_official_evaluation(
                    command, leaderboard_dir, log_path, environment
                )
            except RuntimeError as error:
                execution_errors.append(str(error))
                output = log_path.read_text(encoding="utf-8")

    metrics = parse_official_metrics(output)
    if "next_action" in requested_tasks:
        nap_exact_acc = compute_nap_exact_accuracy(submission, ground_truth)
        if nap_exact_acc is not None:
            metrics["nap_exact_acc"] = nap_exact_acc
            official_nap_acc = metrics.get("nap_acc")
            if (
                official_nap_acc is not None
                and abs(official_nap_acc - nap_exact_acc) > 1e-12
            ):
                warnings.append(
                    "Official NAP_acc differs from local exact label matching: "
                    f"{official_nap_acc:.4f} vs {nap_exact_acc:.4f}."
                )
    task_errors = re.findall(
        r"^Error running ([^ ]+) evaluation: (.+)$", output, re.MULTILINE
    )
    errors = (
        preflight_errors
        + execution_errors
        + [f"{task}: {message}" for task, message in task_errors]
    )
    for task in failed_tasks:
        errors.append(f"{task}: all selected samples failed to run; no successful samples to evaluate.")
        for metric in TASK_METRICS[task]:
            metrics.pop(metric, None)
    llm_calls, invalid_llm_metrics = judge_call_results(output)
    for call in llm_calls:
        if call["successful"] < call["total"]:
            message = f"LLM evaluation completed {call['successful']}/{call['total']} calls successfully."
            warnings.append(message)
            errors.append(message)
    for metric in invalid_llm_metrics:
        metrics.pop(metric, None)
    semantic_fallback = llm_enabled and "semantic_similarity" in output.lower()
    if semantic_fallback:
        for metric_key in LLM_METRICS:
            metrics.pop(metric_key, None)
        errors.append(
            "The official evaluator used semantic-similarity fallback; these "
            "scores are not comparable to Leaderboard LLM metrics."
        )
    missing_metrics = [key for key in expected_metrics if key not in metrics]
    if missing_metrics:
        errors.append("Missing required metrics: " + ", ".join(missing_metrics))
    status = "failed" if errors else "complete"
    evaluation = {
        "status": status,
        "dataset": dataset_name or "legacy",
        "ground_truth_path": str(ground_truth_path),
        "leaderboard_dir": str(leaderboard_dir),
        "tasks": requested_tasks,
        "skip_llm_judge": args.skip_llm_judge,
        "expected_metrics": expected_metrics,
        "missing_metrics": missing_metrics,
        "not_applicable_metrics": not_applicable_metrics,
        "metrics": metrics,
        "llm": {
            "enabled": llm_enabled,
            "config_path": llm_config["config_path"],
            "model": llm_model if llm_enabled else None,
            "api_base": llm_api_base if llm_enabled else None,
            "calls": llm_calls,
            "temperature": 0.0,
            **{key: llm_config[key] for key in ("max_workers", "enable_thinking", "max_completion_tokens", "timeout")},
        },
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "evaluation.json", evaluation)
    write_report(
        run_dir,
        manifest.get("agent", "videoseek"),
        manifest["mode"],
        get_agent_model_name(
            manifest.get("agent_config", {}),
            manifest.get("agent", "videoseek"),
        ),
        manifest["sample_keys"],
        submission,
        latest_records,
        metrics,
        dataset_name or "test",
        evaluation,
    )

    print(f"Evaluation log: {log_path}")
    print(f"Evaluation metrics: {run_dir / 'evaluation.json'}")
    print(f"Report: {run_dir / 'report.md'}")
    for warning in warnings:
        print(f"Warning: {warning}")
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return evaluate_run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
