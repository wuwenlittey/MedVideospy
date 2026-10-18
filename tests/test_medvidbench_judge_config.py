import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from benchmark.medvidbench import evaluate, official_runner


class JudgeConfigurationTest(unittest.TestCase):
    def test_checked_in_config_accepts_typed_settings(self):
        cfg = evaluate.load_evaluation_llm_config()
        self.assertEqual({key: cfg[key] for key in (
            "max_workers", "enable_thinking", "max_completion_tokens", "timeout")},
            {"max_workers": 2, "enable_thinking": False, "max_completion_tokens": 2048, "timeout": 120})

    def test_evaluation_does_not_publish_partial_or_failed_llm_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "leaderboard/evaluation/evaluate_predictions.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.touch()
            reference = {"id": "sample", "qa_type": "video_summary", "conversations": []}
            (root / "ground_truth.json").write_text(json.dumps([reference]))
            (root / "run.json").write_text(json.dumps({
                "dataset": "medvidbench", "dataset_variant": "test", "mode": "lite",
                "sample_keys": ["0"], "metadata_path": str(root / "ground_truth.json"),
                "agent_config": {"model_name": "test-model"}}))
            (root / "submission.json").write_text(json.dumps([{**reference, "prediction": "A scene."}]))
            (root / "records.jsonl").write_text(json.dumps({"sample_key": "0", "status": "success"}) + "\n")
            settings = {"config_path": "test", "model": "test-model", "api_base": "http://unused",
                        "api_key": "dummy", "max_workers": 2, "enable_thinking": False,
                        "max_completion_tokens": 2048, "timeout": 120}
            args = types.SimpleNamespace(run_dir=str(root), ground_truth_path=None,
                leaderboard_dir=str(root / "leaderboard"), tasks=["vs", "dvc"], skip_llm_judge=False)
            for successes in ((True, False), (False, False), (True, True)):
                events = [{"task": "video_summary", "successful": success, "error": None if success else "Incomplete parsing"}
                          for success in successes]
                output = "\n".join("VIDEOSPY_JUDGE_RESULT " + json.dumps(event) for event in events)
                output += "\nLEADERBOARD METRICS SUMMARY\nDense Video Captioning Metrics\n  temporal_f1: 0.7\n  caption_score: 0.0\nVS - Overall Evaluation\n  score: 3.4\nEND LEADERBOARD METRICS SUMMARY\n"
                with patch.object(evaluate, "load_evaluation_llm_config", return_value=settings), \
                     patch.object(evaluate, "run_official_evaluation", return_value=output), \
                     patch.object(evaluate, "write_report"), redirect_stdout(io.StringIO()):
                    status = evaluate.evaluate_run(args)
                result = json.loads((root / "evaluation.json").read_text())
                self.assertEqual(status, 0 if all(successes) else 1)
                self.assertEqual("vs_llm" in result["metrics"], all(successes))
                self.assertEqual(result["metrics"]["dvc_llm"], 0.0)  # No DVC pairs failed.
                self.assertEqual(result["metrics"]["dvc_f1"], .7)
                self.assertEqual(result["llm"]["calls"][0]["failed"], successes.count(False))
                self.assertEqual(result["llm"]["max_completion_tokens"], 2048)
                self.assertEqual(result["llm"]["temperature"], 0.0)

    def test_requests_override_official_defaults_and_limit_concurrency(self):
        active = peak = 0
        lock = threading.Lock()
        requests, clients = [], []

        def create(**kwargs):
            nonlocal active, peak
            with lock:
                requests.append(kwargs)
                active += 1
                peak = max(peak, active)
            time.sleep(.01)
            with lock:
                active -= 1
            return object()

        def make_client(**kwargs):
            clients.append(kwargs)
            return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

        module = types.SimpleNamespace(OpenAI=make_client)
        with patch.dict(sys.modules, {"openai": module}), patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
            official_runner._configure_openai("judge-model", "http://localhost/v1")
            with ThreadPoolExecutor(max_workers=10) as pool:
                list(pool.map(lambda _: module.OpenAI().chat.completions.create(
                    model="gpt-4.1", timeout=30, max_tokens=9999, temperature=1.0,
                    extra_body={"enable_thinking": True}), range(12)))
        self.assertEqual(peak, 2)
        self.assertTrue(all(c["max_retries"] == 0 for c in clients))
        for request in requests:
            self.assertEqual(request["temperature"], 0.0)
            self.assertEqual(request["timeout"], 120)
            self.assertEqual(request["max_completion_tokens"], 2048)
            self.assertFalse(request["extra_body"]["enable_thinking"])
            self.assertNotIn("max_tokens", request)

    def test_dynamic_modules_record_failure_task_and_configure_both_worker_pools(self):
        def load(name):
            return types.SimpleNamespace(call_llm_judge_api=lambda *args: {"api_success": False, "error": "Request timed out."})
        entrypoint = types.SimpleNamespace(load_eval_module=load)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"evaluate_all_pai": entrypoint}), patch.object(sys, "path", sys.path.copy()):
            script = Path(directory) / "evaluate_predictions.py"
            script.with_name("evaluate_all_pai.py").touch()
            official_runner._configure_evaluator(str(script), 2)
            output = io.StringIO()
            with redirect_stdout(output):
                dvc = entrypoint.load_eval_module("eval_dvc")
                caption = entrypoint.load_eval_module("eval_caption_llm_judge")
                self.assertEqual(dvc.DVC_MAX_WORKERS, 2)
                self.assertEqual(caption.MAX_WORKERS, 2)
                dvc.call_llm_judge_api("prediction", "reference", "dense_captioning", "key")
                caption.call_llm_judge_api("prediction", "reference", "video_summary", "key")
            calls, invalid = evaluate.judge_call_results(output.getvalue())
            self.assertEqual(invalid, {"dvc_llm", "vs_llm"})
            self.assertTrue(all(c["successful"] == 0 and c["total"] == 1 for c in calls))

    def test_partial_scores_are_invalid_but_no_matches_is_a_valid_zero(self):
        events = [{"task": "video_summary", "successful": success, "error": None if success else "Incomplete parsing"}
                  for success in (True, False)]
        output = "\n".join("VIDEOSPY_JUDGE_RESULT " + json.dumps(event) for event in events)
        calls, invalid = evaluate.judge_call_results(output)
        self.assertEqual(invalid, {"vs_llm"})
        self.assertEqual(calls[0]["success_rate"], .5)
        self.assertEqual(evaluate.judge_call_results("Matched 0 samples, 0 total pairs across all thresholds"), ([], set()))

