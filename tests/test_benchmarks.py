import argparse
import copy
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.lvbench import run as lvbench
from benchmark.lvbench import utils as lvbench_utils
from benchmark.medvidbench import evaluate as medvidbench_evaluate
from benchmark.medvidbench import official_runner as medvidbench_official_runner
from benchmark.medvidbench import run as medvidbench
from benchmark.medvidbench import utils as medvidbench_utils
from benchmark.medvidbench.image_sequence import ImageSequenceReader
from videospy import utils as videospy_utils


def setUpModule():
    # Fake-agent tests must not contend with a real inference process on this host.
    directory = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(directory.cleanup)
    location = Path(directory.name) / "benchmark/lvbench/utils.py"
    patcher = patch.object(lvbench_utils, "__file__", str(location))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class FakeTrajectory:
    def __init__(self, answer):
        self.answer = answer

    def to_dict(self):
        return {"final_answer": self.answer, "steps": []}


class RunDirectoryNameTest(unittest.TestCase):
    def test_current_commit_id_uses_first_seven_characters(self):
        result = types.SimpleNamespace(stdout="1234567890abcdef\n")

        with patch.object(
            videospy_utils.subprocess, "run", return_value=result
        ) as git:
            commit_id = videospy_utils.current_commit_id()

        self.assertEqual(commit_id, "1234567")
        self.assertEqual(git.call_args.args[0], ["git", "rev-parse", "HEAD"])
        self.assertEqual(git.call_args.kwargs["cwd"], videospy_utils.PROJECT_ROOT)

    def test_automatic_benchmark_run_dirs_append_commit_id(self):
        for utils in (lvbench_utils, medvidbench_utils):
            with self.subTest(module=utils.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    with patch.object(
                        utils,
                        "append_commit_id",
                        side_effect=lambda name: f"{name}_abc1234",
                    ):
                        path = utils.create_run_dir(
                            "dataset",
                            "smoke",
                            directory,
                            None,
                            "videospy",
                            "model",
                        )

                    self.assertTrue(path.name.endswith("_abc1234"))
                    self.assertTrue((path / "trajectories").is_dir())

    def test_automatic_videoseek_run_dirs_are_unchanged(self):
        for utils in (lvbench_utils, medvidbench_utils):
            with self.subTest(module=utils.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    with patch.object(utils, "append_commit_id") as append:
                        path = utils.create_run_dir(
                            "dataset",
                            "smoke",
                            directory,
                            None,
                            "videoseek",
                            "model",
                        )

                    self.assertFalse(path.name.endswith("_abc1234"))
                    append.assert_not_called()

    def test_explicit_run_dir_is_unchanged(self):
        for utils in (lvbench_utils, medvidbench_utils):
            with self.subTest(module=utils.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "explicit"
                    with patch.object(utils, "append_commit_id") as append:
                        result = utils.create_run_dir(
                            "dataset",
                            "smoke",
                            "unused",
                            str(path),
                        )

                    self.assertEqual(result, path.resolve())
                    append.assert_not_called()


class FakeAgent:
    init_count = 0
    run_count = 0

    def __init__(self, **kwargs):
        type(self).init_count += 1
        self.reader = kwargs.get("video_reader")
        self.last_run_rounds = 0
        self.last_run_token_usage = {}

    def run(self, question):
        type(self).run_count += 1
        self.last_run_rounds = 2
        self.last_run_token_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "llm_calls": 2,
            "calls_without_usage": 0,
        }
        if self.reader is not None:
            self.reader.get_batch([0])
        return FakeTrajectory("Option (B)")


class FailFirstAgent(FakeAgent):
    should_fail = True

    def run(self, question):
        if type(self).should_fail:
            type(self).should_fail = False
            raise RuntimeError("temporary failure")
        return FakeTrajectory("B")


class ConcurrentAgent(FakeAgent):
    barrier = None

    def run(self, question):
        type(self).barrier.wait(timeout=5)
        return super().run(question)


class ReleaseTrackedDict(dict):
    released = threading.Event()

    def __del__(self):
        type(self).released.set()


class ReleaseTrackedTrajectory:
    def to_dict(self):
        return ReleaseTrackedDict(final_answer="B", steps=[])


class MemoryReleaseAgent(FakeAgent):
    run_lock = threading.Lock()
    run_count = 0

    def run(self, question):
        with type(self).run_lock:
            type(self).run_count += 1
            run_number = type(self).run_count
        if run_number == 3 and not ReleaseTrackedDict.released.wait(timeout=2):
            raise RuntimeError("completed trajectory was retained")
        return ReleaseTrackedTrajectory()


def runner_args(**overrides):
    values = {
        "agent": "videoseek",
        "mode": "full",
        "output_dir": "unused",
        "run_dir": None,
        "verbose": False,
        "workers": 1,
        "config_path": None,
        "model_name": None,
        "api_base": None,
        "api_key": None,
        "api_version": None,
        "seed": None,
        "temperature": None,
        "max_tokens": None,
        "max_completion_tokens": None,
        "max_steps": None,
        "evaluate": False,
        "skip_llm_judge": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SampleManifestTest(unittest.TestCase):
    def test_modes_default_to_full_and_reject_removed_options(self):
        from benchmark.egolife import run as egolife
        for runner in (lvbench, medvidbench, egolife):
            self.assertEqual(runner.parse_args([]).mode, "full")
            for mode in ("full", "lite", "debug"):
                self.assertEqual(runner.parse_args(["--mode", mode]).mode, mode)
            for option in (["--mode", "smoke"], ["--debug"]):
                with self.subTest(runner=runner.__name__, option=option):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(option)

    def test_cli_defaults_to_each_test_split_and_allows_explicit_override(self):
        from benchmark.egolife import run as egolife
        split_root = Path(__file__).resolve().parents[1] / "experiments/data_splits/main"
        for name, runner in (("lvbench", lvbench), ("medvidbench", medvidbench), ("egolife", egolife)):
            with self.subTest(dataset=name):
                args = runner.parse_args([])
                self.assertEqual(Path(args.sample_manifest), split_root / name / "test.json")
                self.assertTrue(Path(args.sample_manifest).is_file())
                overridden = runner.parse_args(["--sample-manifest", "custom.json"])
                self.assertEqual(overridden.sample_manifest, "custom.json")

    def test_hidden_medvidbench_split_does_not_inherit_local_test_indices(self):
        manifest = Path(medvidbench.DEFAULT_SAMPLE_MANIFEST).with_name("leaderboard_test.json")
        args = medvidbench.parse_args(["--sample-manifest", str(manifest)])
        dataset, metadata, frames = medvidbench.resolve_dataset(args)
        self.assertEqual(dataset, "leaderboard_test")
        expected = medvidbench.DATASET_CONFIGS[dataset]
        self.assertEqual(metadata, Path(expected["metadata_path"]).resolve())
        self.assertEqual(frames, Path(expected["frame_root"]).resolve())
        with patch.object(medvidbench, "get_sampling_stride", return_value=3):
            self.assertEqual(medvidbench.select_indices(5, "full", manifest), list(range(5)))

    def test_medvidbench_manifest_selects_source_and_preserves_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "custom.json"
            for dataset in ("test", "leaderboard_test"):
                manifest.write_text(json.dumps({"dataset": dataset, "sample_indices": [2, 0]}))
                args = medvidbench.parse_args(["--sample-manifest", str(manifest)])
                self.assertEqual(medvidbench.resolve_dataset(args)[0], dataset)
                self.assertEqual(medvidbench.select_indices(3, "full", manifest), [2, 0])
            for value in ({"dataset": "unknown", "sample_indices": [0]},
                          {"dataset": "leaderboard_test"},
                          {"dataset": "leaderboard_test", "sample_indices": [True]}):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    medvidbench.resolve_dataset(args)

    def test_medvidbench_rejects_removed_dataset_flags(self):
        for flag in ("--split", "--dataset"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                medvidbench.parse_args([flag, "test"])

    def test_default_test_manifest_is_sampled_before_modes(self):
        from benchmark.egolife import run as egolife
        for name, runner, lite_stride in (("lvbench", lvbench, 3), ("medvidbench", medvidbench, 10), ("egolife", egolife, 3)):
            for mode, stride in (("full", 1), ("debug", 100), ("lite", lite_stride)):
                with self.subTest(dataset=name, mode=mode):
                    args = runner.parse_args(["--mode", mode])
                    ids = json.loads(Path(args.sample_manifest).read_text())
                    if runner is medvidbench:
                        selected = runner.select_indices(max(ids) + 1, args.mode, args.sample_manifest)
                    else:
                        key = "uid" if runner is lvbench else "sample_key"
                        samples = [{key: str(i)} for i in ids] + [{key: "not-in-test"}]
                        selected = [s[key] for s in runner.select_samples(samples, args.mode, args.sample_manifest)]
                        ids = [str(i) for i in ids]
                    self.assertEqual(selected, ids[::stride])

    def test_both_benchmarks_apply_stride_after_manifest_selection(self):
        samples = [{"uid": str(i)} for i in range(400)]
        indices = list(reversed(range(315)))
        ids = [str(i) for i in indices]
        with tempfile.TemporaryDirectory() as directory:
            lv_manifest, med_manifest = Path(directory) / "lv.json", Path(directory) / "med.json"
            for lv_value, med_value in ((ids, indices), ({"sample_ids": ids}, {"sample_indices": indices})):
                lv_manifest.write_text(json.dumps(lv_value))
                med_manifest.write_text(json.dumps(med_value))
                for mode, lv_step, med_step in (("full", 1, 1), ("debug", 100, 100), ("lite", 3, 10)):
                    with self.subTest(mode=mode):
                        selected = lvbench.select_samples(samples, mode, lv_manifest)
                        self.assertEqual([s["uid"] for s in selected], ids[::lv_step])
                        self.assertEqual(selected, lvbench.select_samples(samples, mode, lv_manifest))
                        self.assertEqual(medvidbench.select_indices(400, mode, med_manifest), indices[::med_step])

    def test_stride_without_manifest_and_boundary_counts(self):
        for count in (0, 1, 10, 49, 50, 51, 100, 101):
            with self.subTest(count=count):
                samples = [{"uid": str(i)} for i in range(count)]
                for mode, lv_step, med_step in (("full", 1, 1), ("debug", 100, 100), ("lite", 3, 10)):
                    self.assertEqual(lvbench.select_samples(samples, mode), samples[::lv_step])
                    self.assertEqual(medvidbench.select_indices(count, mode), list(range(count))[::med_step])

    def test_manifest_validation_precedes_sampling_and_rejects_mode_specific_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            for value in (["0", "missing"], {"debug": ["0"]}):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    lvbench.select_samples([{"uid": "0"}], "debug", manifest)
            for value in ([0, 5], [0, True], {"debug": [0]}):
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    medvidbench.select_indices(1, "debug", manifest)

    def test_lvbench_manifest_order_is_preserved(self):
        samples = [{"uid": "1"}, {"uid": "2"}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('["2", "1"]', encoding="utf-8")
            selected = lvbench.select_samples(samples, "full", manifest)
        self.assertEqual([sample["uid"] for sample in selected], ["2", "1"])

    def test_lvbench_modes_stride_full_dataset(self):
        metadata = Path(lvbench.DEFAULT_METADATA_PATH)
        if not metadata.exists():
            self.skipTest("LVBench data link is unavailable")
        samples = lvbench.load_dataset(metadata)
        debug = lvbench.select_samples(samples, "debug")
        lite = lvbench.select_samples(samples, "lite")
        self.assertEqual(debug, samples[::100])
        self.assertEqual(lite, samples[::3])
        self.assertTrue({x["uid"] for x in debug} <= {x["uid"] for x in samples})

    def test_medvidbench_modes_stride_full_dataset(self):
        metadata = Path(medvidbench.DEFAULT_METADATA_PATH)
        if not metadata.exists():
            self.skipTest("MedVidBench data link is unavailable")
        dataset = medvidbench.load_dataset(metadata)
        debug = medvidbench.select_indices(len(dataset), "debug")
        lite = medvidbench.select_indices(len(dataset), "lite")
        self.assertEqual(debug, list(range(len(dataset)))[::100])
        self.assertEqual(lite, list(range(len(dataset)))[::10])
        self.assertLessEqual(set(debug), set(lite))
        self.assertEqual(len(medvidbench.select_indices(len(dataset), "full")), 6270)

    def test_medvidbench_leaderboard_test_submission_order_and_schema(self):
        metadata = Path(
            medvidbench.DATASET_CONFIGS["leaderboard_test"]["metadata_path"]
        )
        if not metadata.exists():
            self.skipTest("MedVidBench leaderboard test data link is unavailable")
        dataset = medvidbench.load_dataset(metadata)
        indices = medvidbench.select_indices(len(dataset), "full")
        records = {
            str(index): {"prediction": f"prediction-{index}"}
            for index in indices
        }
        submission = medvidbench.build_submission(dataset, indices, records)

        self.assertEqual(len(submission), 6245)
        self.assertEqual(
            [item["id"] for item in submission],
            [item["id"] for item in dataset],
        )
        self.assertTrue(
            all(set(item) == {"id", "qa_type", "prediction"} for item in submission)
        )


class WorkerArgumentTest(unittest.TestCase):
    def test_worker_defaults_and_validation(self):
        for runner in (lvbench, medvidbench):
            with self.subTest(runner=runner.__name__):
                self.assertEqual(runner.parse_args([]).workers, 2 if runner is lvbench else 4)
                self.assertEqual(runner.parse_args(["--workers", "2"]).workers, 2)
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    runner.parse_args(["--workers", "0"])


class ImageSequenceReaderTest(unittest.TestCase):
    def test_fps_batch_and_region_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            for name in ("0.png", "1.png"):
                Image.new("RGB", (24, 16), (0, 0, 0)).save(frame_dir / name)
            paths = ["/root/data/frames/0.png", "/root/data/frames/1.png"]
            reader = ImageSequenceReader(
                paths,
                0.5,
                root,
                {"start_frame": paths[0], "start_frame_bbox": [2, 2, 12, 10]},
            )
            frames = reader.get_batch(np.array([0, 1])).asnumpy()
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader.get_avg_fps(), 0.5)
            self.assertEqual(frames.shape, (2, 16, 24, 3))
            self.assertTrue(np.any(frames[0, :, :, 1] == 255))
            self.assertFalse(np.any(frames[1, :, :, 1] == 255))

    def test_focus_accepts_a_short_low_fps_interval(self):
        decord_stub = types.ModuleType("decord")
        decord_stub.VideoReader = object
        litellm_stub = types.ModuleType("litellm")
        litellm_stub.completion = lambda **kwargs: None
        with patch.dict(
            sys.modules,
            {"decord": decord_stub, "litellm": litellm_stub},
        ):
            from videoseek.tools import focus

        class LowFpsReader:
            def __len__(self):
                return 50

            def get_avg_fps(self):
                return 0.1

            def get_batch(self, indices):
                self.indices = list(indices)
                return types.SimpleNamespace(
                    asnumpy=lambda: np.zeros((len(indices), 16, 16, 3), dtype=np.uint8)
                )

        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="observation")
                )
            ]
        )
        reader = LowFpsReader()
        config = {
            "model_name": "test",
            "api_base": "test",
            "api_key": "test",
            "api_version": "test",
            "max_tokens": 10,
            "reasoning_effort": "low",
            "seed": 42,
            "temperature": 0.0,
        }
        with patch.object(focus, "call_llm_api", return_value=response):
            result = focus.execute_focus(
                config,
                {
                    "query": "test",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "vr": reader,
                    "subtitles": [],
                },
            )
        self.assertEqual(result, "observation")
        self.assertEqual(reader.indices, [0])


class LVBenchRunnerTest(unittest.TestCase):
    def test_stride_debug_records_selected_ids_and_resumes_only_that_subset(self):
        FakeAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video1.mp4").touch()
            metadata = root / "metadata.jsonl"
            metadata.write_text(json.dumps({"key": "video1", "qa": [
                {"uid": str(i), "question": "Question?", "answer": "B", "question_type": ["reasoning"]}
                for i in range(100)
            ]}) + "\n")
            args = runner_args(run_dir=str(root / "run"), metadata_path=str(metadata),
                               video_dir=str(root), mode="debug")
            self.assertEqual(lvbench.run_benchmark(args, FakeAgent), 0)
            manifest = json.loads((root / "run/run.json").read_text())
            expected = lvbench.select_samples(lvbench.load_dataset(metadata), "debug")
            self.assertEqual(manifest["sample_keys"], [s["uid"] for s in expected])
            self.assertEqual(FakeAgent.run_count, 1)
            self.assertEqual(lvbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.run_count, 1)
            args.mode = "full"
            with self.assertRaises(ValueError):
                lvbench.run_benchmark(args, FakeAgent)
            self.assertEqual(FakeAgent.run_count, 1)

    def test_agent_selection_defaults_to_videoseek(self):
        self.assertEqual(lvbench.parse_args(["--mode", "debug"]).agent, "videoseek")
        self.assertEqual(
            lvbench.parse_args(["--mode", "debug", "--agent", "videospy"]).agent,
            "videospy",
        )

    def test_verbose_is_opt_in(self):
        self.assertFalse(lvbench.parse_args(["--mode", "debug"]).verbose)
        self.assertTrue(lvbench.parse_args(["--mode", "debug", "--verbose"]).verbose)

    def test_evolved_component_switches_require_database(self):
        args = lvbench.parse_args(
            ["--agent", "videospy", "--no-evolved-memory"]
        )
        self.assertFalse(args.evolved_memory)
        with self.assertRaisesRegex(ValueError, "require --evolution-db"):
            lvbench_utils.build_agent_config(args)

    def test_custom_videoseek_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "lvbench.yml"
            config_path.write_text(
                "model_name: benchmark-model\n"
                "api_base: https://example.com/v1\n"
                "api_key: test-key\n"
                "api_version: v1\n"
                "reasoning_effort: low\n"
                "seed: 7\n"
                "temperature: 0.2\n"
                "max_tokens: 100\n"
                "max_steps: 9\n"
                "tools: []\n",
                encoding="utf-8",
            )
            args = lvbench.parse_args(
                ["--config", str(config_path), "--model-name", "cli-model"]
            )

            config = lvbench_utils.build_agent_config(args)

            self.assertEqual(args.config_path, str(config_path))
            self.assertEqual(config["model_name"], "cli-model")
            self.assertEqual(config["max_steps"], 9)

    def test_custom_videospy_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "lvbench-videospy.yml"
            second_config_path = Path(directory) / "lvbench-override.yml"
            config_path.write_text(
                "agent:\n"
                "  max_steps: 6\n"
                "  model:\n"
                "    model_name: benchmark-model\n"
                "    api_base: https://example.com/v1\n"
                "    api_key: test-key\n"
                "tools:\n"
                "  enabled: []\n",
                encoding="utf-8",
            )
            second_config_path.write_text(
                "SYSTEM_PROMPT: benchmark prompt\n",
                encoding="utf-8",
            )
            args = lvbench.parse_args(
                [
                    "--agent",
                    "videospy",
                    "--config",
                    str(config_path),
                    "--config",
                    str(second_config_path),
                    "--max-steps",
                    "8",
                ]
            )

            config = lvbench_utils.build_agent_config(args)

            self.assertEqual(config["agent"]["model"]["model_name"], "benchmark-model")
            self.assertEqual(config["agent"]["max_steps"], 8)
            self.assertEqual(config["SYSTEM_PROMPT"], "benchmark prompt")
            self.assertIn("overview", config["tools"])

    def test_videospy_uses_explicit_benchmark_yaml(self):
        for runner, utils in (
            (lvbench, lvbench_utils),
            (medvidbench, medvidbench_utils),
        ):
            with self.subTest(module=utils.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    config_path = Path(directory) / "videospy.yml"
                    config_path.write_text(
                        "agent:\n  max_steps: 21\n"
                        "TASK_SKILLS:\n  retrieval:\n    - instruction: inspect evidence\n",
                        encoding="utf-8",
                    )
                    args = runner.parse_args(
                        ["--agent", "videospy", "--config", str(config_path)]
                    )
                    config = utils.build_agent_config(args)

                self.assertEqual(config["agent"]["max_steps"], 21)
                self.assertIn("retrieval", config["TASK_SKILLS"])

    def test_benchmark_yaml_is_not_loaded_without_config_argument(self):
        for utils in (lvbench_utils, medvidbench_utils):
            with self.subTest(module=utils.__name__):
                config = utils.build_agent_config(runner_args(agent="videospy"))
                self.assertNotEqual(config["agent"]["max_steps"], 999)

    def test_outputs_metrics_and_resume(self):
        FakeAgent.init_count = 0
        FakeAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "video1.mp4").touch()
            metadata.write_text(
                json.dumps(
                    {
                        "key": "video1",
                        "qa": [
                            {
                                "uid": "1",
                                "question": "Question?\n(A) x\n(B) y\n(C) z\n(D) w",
                                "answer": "B",
                                "question_type": ["reasoning"],
                            },
                            {
                                "uid": "2",
                                "question": "Question?\n(A) x\n(B) y\n(C) z\n(D) w",
                                "answer": "A",
                                "question_type": ["reasoning"],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            args = runner_args(
                run_dir=str(run_dir),
                metadata_path=str(metadata),
                video_dir=str(video_dir),
            )
            self.assertEqual(lvbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.init_count, 1)
            self.assertEqual(FakeAgent.run_count, 2)
            self.assertEqual(lvbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.run_count, 2)
            predictions = json.loads((run_dir / "predictions.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            run_log = (run_dir / "run.log").read_text(encoding="utf-8")
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertEqual(predictions, {"1": "B", "2": "B"})
            self.assertEqual(metrics["overall"]["accuracy"], 0.5)
            self.assertEqual(metrics["run_statistics"]["agent_rounds"]["total"], 4)
            self.assertEqual(
                metrics["run_statistics"]["token_usage"]["total_tokens"], 30
            )
            self.assertGreaterEqual(
                metrics["run_statistics"]["elapsed_seconds"]["total"],
                0.0,
            )
            self.assertIn("Starting LVBench", run_log)
            self.assertIn("Sample 1: success", run_log)
            self.assertIn("Completed: 2/2", run_log)
            self.assertIn("# LVBench Run Report", report)
            self.assertIn("| Overall | videoseek | full |", report)
            self.assertIn("| reasoning | videoseek | full |", report)
            self.assertIn("| Accuracy (%) | Avg. rounds | Avg. tokens |", report)
            self.assertIn("| 50.0 | 2.0 | 15.0 |", report)

    def test_samples_run_concurrently_and_records_remain_valid(self):
        ConcurrentAgent.barrier = threading.Barrier(2)
        ConcurrentAgent.init_count = 0
        ConcurrentAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "video1.mp4").touch()
            metadata.write_text(
                json.dumps(
                    {
                        "key": "video1",
                        "qa": [
                            {
                                "uid": str(index),
                                "question": "Question?",
                                "answer": "B",
                                "question_type": ["reasoning"],
                            }
                            for index in (1, 2)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            args = runner_args(
                run_dir=str(run_dir),
                metadata_path=str(metadata),
                video_dir=str(video_dir),
                workers=2,
            )

            self.assertEqual(lvbench.run_benchmark(args, ConcurrentAgent), 0)

            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text().splitlines()
            ]
            self.assertEqual({record["sample_key"] for record in records}, {"1", "2"})
            self.assertTrue(all(record["status"] == "success" for record in records))
            self.assertTrue(
                all((run_dir / record["trajectory"]).is_file() for record in records)
            )

    def test_completed_trajectories_are_released_while_run_is_active(self):
        ReleaseTrackedDict.released.clear()
        MemoryReleaseAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "video1.mp4").touch()
            metadata.write_text(
                json.dumps(
                    {
                        "key": "video1",
                        "qa": [
                            {
                                "uid": str(index),
                                "question": "Question?",
                                "answer": "B",
                                "question_type": ["reasoning"],
                            }
                            for index in range(3)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = runner_args(
                run_dir=str(root / "run"),
                metadata_path=str(metadata),
                video_dir=str(video_dir),
                workers=2,
            )

            status = lvbench.run_benchmark(args, MemoryReleaseAgent)

        self.assertEqual(status, 0)
        self.assertTrue(ReleaseTrackedDict.released.is_set())

    def test_multiple_choice_question_requires_single_letter(self):
        question = lvbench.build_question("Which option is correct?")

        self.assertIn("Return exactly one uppercase option letter", question)
        self.assertIn("Do not include parentheses", question)

    def test_answer_normalization(self):
        self.assertEqual(lvbench.normalize_answer("A"), "A")
        self.assertEqual(lvbench.normalize_answer("Option (C)"), "C")
        self.assertEqual(lvbench.normalize_answer("The answer is D."), "D")
        self.assertEqual(lvbench.normalize_answer("The best answer is:\n(B)"), "B")
        self.assertEqual(
            lvbench.normalize_answer("The best option's letter is **(B)**."), "B"
        )
        self.assertEqual(
            lvbench.normalize_answer("Therefore, the correct option is:\n(C)"), "C"
        )
        self.assertEqual(
            lvbench.normalize_answer("A detailed explanation mentioning C.\n\n(B)"),
            "B",
        )
        self.assertEqual(
            lvbench.normalize_answer(
                "The answer is A because option B is inconsistent."
            ),
            "A",
        )
        self.assertEqual(
            lvbench.normalize_answer("Option A seems likely. Final answer: C"),
            "C",
        )
        self.assertEqual(
            lvbench.normalize_answer(
                "The options mention A, B, C, and D.\n\nA"
            ),
            "A",
        )
        self.assertEqual(
            lvbench.normalize_answer("The platform is Netflix.\n\n(B) Netflix"),
            "B",
        )
        self.assertEqual(
            lvbench.normalize_answer("The liquid is milk.\n\n**(C)** Milk"),
            "C",
        )
        self.assertEqual(
            lvbench.normalize_answer("The evidence supports it.\n\nD. Option text"),
            "D",
        )
        self.assertEqual(
            lvbench.normalize_answer(
                "Final answer: A\nAfter checking again:\n(B) Revised option"
            ),
            "B",
        )
        self.assertEqual(lvbench.normalize_answer("A or B"), "")
        self.assertEqual(lvbench.normalize_answer("A person chose option B or C."), "")

    def test_failed_sample_is_retried_on_resume(self):
        FailFirstAgent.should_fail = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "video1.mp4").touch()
            metadata.write_text(
                json.dumps(
                    {
                        "key": "video1",
                        "qa": [
                            {
                                "uid": "1",
                                "question": "Question?",
                                "answer": "B",
                                "question_type": ["reasoning"],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = runner_args(
                run_dir=str(root / "run"),
                metadata_path=str(metadata),
                video_dir=str(video_dir),
            )
            self.assertEqual(lvbench.run_benchmark(args, FailFirstAgent), 1)
            run_log = (root / "run" / "run.log").read_text(encoding="utf-8")
            self.assertIn("RuntimeError: temporary failure", run_log)
            self.assertEqual(lvbench.run_benchmark(args, FailFirstAgent), 0)
            records = lvbench_utils.load_latest_records(
                root / "run" / "records.jsonl"
            )
            self.assertEqual(records["1"]["status"], "success")


class ResumeRecordTest(unittest.TestCase):
    def test_truncated_final_record_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                '{"sample_key":"1","status":"success"}\n{"sample_key":',
                encoding="utf-8",
            )
            records = lvbench_utils.load_latest_records(path)
            self.assertEqual(set(records), {"1"})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"sample_key":"1","status":"success"}\n',
            )

    def test_manifest_rejects_a_different_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            original = {"dataset": "lvbench", "agent": "videoseek"}
            lvbench_utils.ensure_run_manifest(run_dir, original)
            with self.assertRaisesRegex(ValueError, "does not match"):
                lvbench_utils.ensure_run_manifest(
                    run_dir, {"dataset": "lvbench", "agent": "videospy"}
                )

    def test_legacy_videoseek_manifest_remains_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {"dataset": "lvbench", "agent_config": {"model_name": "test"}}
                ),
                encoding="utf-8",
            )
            lvbench_utils.ensure_run_manifest(
                run_dir,
                {
                    "dataset": "lvbench",
                    "agent": "videoseek",
                    "agent_config": {"model_name": "test", "max_tokens": 100},
                },
            )

    def test_manifest_ignores_updated_evolution_database_hash(self):
        for utils in (lvbench_utils, medvidbench_utils):
            with (
                self.subTest(module=utils.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                run_dir = Path(directory)
                original = {
                    "dataset": "test",
                    "agent": "videospy",
                    "agent_config": {
                        "_evolution": {
                            "database_path": "/same/server/evolution.sqlite3",
                            "database_sha256": "before",
                            "version_id": "run:initial",
                        }
                    },
                }
                utils.ensure_run_manifest(run_dir, original)
                updated = copy.deepcopy(original)
                updated["agent_config"]["_evolution"]["database_sha256"] = "after"
                utils.ensure_run_manifest(run_dir, updated)

    def test_manifest_still_rejects_a_different_database_path(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            original = {
                "dataset": "lvbench",
                "agent": "videospy",
                "agent_config": {
                    "_evolution": {
                        "database_path": "/server/run-a/evolution.sqlite3",
                        "database_sha256": "same",
                    }
                },
            }
            lvbench_utils.ensure_run_manifest(run_dir, original)
            updated = copy.deepcopy(original)
            updated["agent_config"]["_evolution"]["database_path"] = (
                "/server/run-b/evolution.sqlite3"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                lvbench_utils.ensure_run_manifest(run_dir, updated)

    def test_nested_credentials_are_redacted(self):
        safe = medvidbench_utils.safe_config(
            {
                "agent": {
                    "model": {
                        "model_name": "test",
                        "api_key": "secret",
                        "max_tokens": 100,
                    }
                }
            }
        )
        self.assertEqual(
            safe,
            {"agent": {"model": {"model_name": "test", "max_tokens": 100}}},
        )


class TokenUsageTest(unittest.TestCase):
    def test_llm_response_usage_is_accumulated(self):
        decord_stub = types.ModuleType("decord")
        decord_stub.VideoReader = object
        litellm_stub = types.ModuleType("litellm")
        litellm_stub.completion = lambda **kwargs: None
        with patch.dict(
            sys.modules,
            {"decord": decord_stub, "litellm": litellm_stub},
        ):
            from videoseek import metrics, utils

        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
            )
        )
        with metrics.track_token_usage() as usage:
            with patch.object(utils, "completion", return_value=response):
                utils.call_llm_api(
                    model_name="test",
                    messages=[],
                    api_base="test",
                )
        self.assertEqual(
            usage.to_dict(),
            {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "llm_calls": 1,
                "calls_without_usage": 0,
            },
        )


class MedVidBenchRunnerTest(unittest.TestCase):
    def test_dataset_defaults_and_leaderboard_test_restrictions(self):
        args = medvidbench.parse_args([])
        lvbench_args = lvbench.parse_args([])
        for field in (
            "agent",
            "mode",
            "output_dir",
            "run_dir",
            "verbose",
            "config_path",
            "model_name",
            "api_base",
            "api_key",
            "api_version",
            "seed",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "max_steps",
        ):
            self.assertEqual(getattr(args, field), getattr(lvbench_args, field))
        dataset_name, metadata_path, frame_root = medvidbench.resolve_dataset(args)
        self.assertEqual(dataset_name, "test")
        self.assertEqual(args.agent, "videoseek")
        self.assertEqual(args.mode, "full")
        self.assertEqual(metadata_path, Path(medvidbench.DEFAULT_METADATA_PATH).resolve())
        self.assertEqual(frame_root, Path(medvidbench.DEFAULT_FRAME_ROOT).resolve())

        manifest = Path(medvidbench.DEFAULT_SAMPLE_MANIFEST).with_name("leaderboard_test.json")
        args = medvidbench.parse_args(
            ["--agent", "videospy", "--sample-manifest", str(manifest)]
        )
        self.assertEqual(args.agent, "videospy")
        self.assertEqual(medvidbench.resolve_dataset(args)[0], "leaderboard_test")
        self.assertEqual(args.mode, "full")

        args = medvidbench.parse_args(
            ["--sample-manifest", str(manifest), "--mode", "debug"]
        )
        with self.assertRaisesRegex(ValueError, "only supports --mode full"):
            medvidbench.resolve_dataset(args)

        args = medvidbench.parse_args(
            ["--sample-manifest", str(manifest), "--mode", "full", "--evaluate"]
        )
        with self.assertRaisesRegex(ValueError, "no local Ground Truth"):
            medvidbench.resolve_dataset(args)

    def test_official_answer_formats(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "<video>\nWhen does cutting happen?"}
            ],
            "qa_type": "tal",
        }
        question = medvidbench.build_question(sample)
        self.assertTrue(question.startswith("When does cutting happen?"))
        self.assertIn("1.1-3.0 seconds.", question)

        sample["qa_type"] = "stg"
        question = medvidbench.build_question(sample)
        self.assertIn("<timestamp> seconds", question)
        self.assertIn("Use exactly the timestamps requested", question)
        self.assertNotIn("1.0 seconds: [x1", question)
        self.assertNotIn("5.0 seconds: [x1", question)

        sample["qa_type"] = "cvs_assessment"
        question = medvidbench.build_question(sample)
        self.assertIn("Two structures: <0-2>", question)
        self.assertIn("Hepatocystic triangle: <0-2>", question)

        sample["qa_type"] = "dense_captioning_gemini"
        question = medvidbench.build_question(sample)
        self.assertIn("one event per line", question)

    def test_task_prompt_is_injected_only_for_the_current_videospy_task(self):
        from videospy.evolved_config import format_skills

        config = {
            "SYSTEM_PROMPT": "base prompt\n",
            "TASK_SKILLS": {task: {'format': 'document', 'instruction': action,
                                   'sources': [], 'source_tasks': [], 'review': []}
                            for task, action in [('tal', 'tal strategy'), ('dense_captioning', 'dvc strategy'),
                                                 ('region_caption', 'region strategy'), ('video_summary', 'summary strategy')]},
        }

        cases = {
            "tal": "tal strategy",
            "dense_captioning_gpt": "dvc strategy",
            "region_caption_gemini": "region strategy",
            "video_summary_gpt": "summary strategy",
        }
        for qa_type, expected in cases.items():
            with self.subTest(qa_type=qa_type):
                task_config = medvidbench.build_task_agent_config(
                    config, qa_type, "videospy"
                )
                self.assertEqual(task_config["SYSTEM_PROMPT"], config["SYSTEM_PROMPT"])
                prompt, _ = format_skills(task_config, [task_config['_runtime']['task_type']])
                self.assertIn(expected, prompt)
                for other in set(cases.values()) - {expected}:
                    self.assertNotIn(other, prompt)

        self.assertIs(
            medvidbench.build_task_agent_config(config, "tal", "videoseek"),
            config,
        )
        disabled = {"SYSTEM_PROMPT": "base prompt", "TASK_SKILLS": {}}
        task_config = medvidbench.build_task_agent_config(
            disabled, "tal", "videospy"
        )
        self.assertEqual(task_config["_runtime"]["task_type"], "tal")
        self.assertEqual(format_skills(task_config, ['tal'])[0], '')

    def test_custom_yml_can_override_or_disable_task_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yml"
            config_path.write_text(
                "TASK_SKILLS:\n  tal:\n    - instruction: custom tal strategy\n",
                encoding="utf-8",
            )
            args = medvidbench.parse_args(
                ["--agent", "videospy", "--config", str(config_path)]
            )
            config = medvidbench_utils.build_agent_config(args)
            self.assertEqual(
                config["TASK_SKILLS"]["tal"][0]["instruction"],
                "custom tal strategy",
            )

            config_path.write_text("TASK_SKILLS: {}\n", encoding="utf-8")
            config = medvidbench_utils.build_agent_config(args)
            self.assertEqual(config["TASK_SKILLS"], {})

    def test_submission_and_resume(self):
        FakeAgent.init_count = 0
        FakeAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_root = root / "testdata"
            frame_dir = frame_root / "frames"
            frame_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (0, 0, 0)).save(frame_dir / "0.png")
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "id": "clip&&0&&1&&1.0",
                            "video": ["/root/data/frames/0.png"],
                            "conversations": [
                                {"from": "human", "value": "<video>\nWhat happens?"},
                                {"from": "gpt", "value": "1.0-2.0 seconds."},
                            ],
                            "metadata": {"video_id": "clip", "fps": "1.0"},
                            "qa_type": "tal",
                            "is_RC": False,
                            "RC_info": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            args = runner_args(
                run_dir=str(run_dir),
                metadata_path=str(metadata),
                frame_root=str(frame_root),
            )
            self.assertEqual(medvidbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.run_count, 1)
            (run_dir / "evaluation.json").write_text(
                json.dumps({"metrics": {"tag_miou_05": 0.5}}), encoding="utf-8"
            )
            self.assertEqual(medvidbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.run_count, 1)
            submission = json.loads((run_dir / "submission.json").read_text())
            record = medvidbench_utils.load_latest_records(
                run_dir / "records.jsonl"
            )["0"]
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertEqual(record["agent_rounds"], 2)
            self.assertEqual(record["token_usage"]["total_tokens"], 15)
            self.assertGreaterEqual(record["elapsed_seconds"], 0.0)
            self.assertEqual(record["prediction"], "Option (B)")
            self.assertEqual(record["ground_truth"], "1.0-2.0 seconds.")
            self.assertEqual(record["question"], "What happens?")
            self.assertFalse((run_dir / "summary.json").exists())
            self.assertFalse((run_dir / "comparison.jsonl").exists())
            self.assertIn("# MedVidBench Run Report", report)
            self.assertIn("| TAL | videoseek | test | full |", report)
            self.assertIn("| Avg. rounds | Avg. tokens | Avg. time (s) |", report)
            self.assertIn("| 1/1 | 2.0 | 15.0 |", report)
            self.assertIn("0.5000", report)
            self.assertIn("| CVS_acc | NAP_acc | SA_acc | STG_mIoU |", report)
            self.assertEqual(
                submission,
                [
                    {
                        "id": "clip&&0&&1&&1.0",
                        "qa_type": "tal",
                        "prediction": "Option (B)",
                    }
                ],
            )

            # Changing only the manifest selects hidden-data behavior and preserves resumability.
            hidden_manifest = root / "leaderboard.json"
            hidden_manifest.write_text(json.dumps({"dataset": "leaderboard_test", "sample_indices": "all"}))
            args.sample_manifest = str(hidden_manifest)
            args.run_dir = str(root / "hidden-run")
            self.assertEqual(medvidbench.run_benchmark(args, FakeAgent), 0)
            hidden_run = Path(args.run_dir)
            self.assertEqual(json.loads((hidden_run / "run.json").read_text())["dataset_variant"], "leaderboard_test")
            self.assertEqual(json.loads((hidden_run / "submission.json").read_text()), submission)
            hidden_record = medvidbench_utils.load_latest_records(hidden_run / "records.jsonl")["0"]
            self.assertNotIn("ground_truth", hidden_record)
            completed_calls = FakeAgent.run_count
            self.assertEqual(medvidbench.run_benchmark(args, FakeAgent), 0)
            self.assertEqual(FakeAgent.run_count, completed_calls)

    def test_samples_run_concurrently_and_records_remain_valid(self):
        ConcurrentAgent.barrier = threading.Barrier(2)
        ConcurrentAgent.init_count = 0
        ConcurrentAgent.run_count = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_root = root / "testdata"
            frame_dir = frame_root / "frames"
            frame_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (0, 0, 0)).save(frame_dir / "0.png")
            metadata = root / "metadata.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "id": f"clip&&{index}&&1&&1.0",
                            "video": ["/root/data/frames/0.png"],
                            "conversations": [
                                {"from": "human", "value": "<video>\nQuestion?"},
                                {"from": "gpt", "value": "Answer."},
                            ],
                            "metadata": {"video_id": "clip", "fps": "1.0"},
                            "qa_type": "tal",
                            "is_RC": False,
                            "RC_info": None,
                        }
                        for index in range(2)
                    ]
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            args = runner_args(
                run_dir=str(run_dir),
                metadata_path=str(metadata),
                frame_root=str(frame_root),
                workers=2,
            )

            self.assertEqual(
                medvidbench.run_benchmark(args, ConcurrentAgent), 0
            )

            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text().splitlines()
            ]
            self.assertEqual({record["sample_key"] for record in records}, {"0", "1"})
            self.assertTrue(all(record["status"] == "success" for record in records))
            self.assertTrue(
                all((run_dir / record["trajectory"]).is_file() for record in records)
            )

    def test_local_test_run_can_trigger_automatic_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_root = root / "valdata"
            frame_dir = frame_root / "frames"
            frame_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (0, 0, 0)).save(frame_dir / "0.png")
            metadata = root / "test.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "id": "clip&&0&&1&&1.0",
                            "video": ["/root/data/frames/0.png"],
                            "conversations": [
                                {"from": "human", "value": "<video>\nWhat happens?"},
                                {"from": "gpt", "value": "1.0-2.0 seconds."},
                            ],
                            "metadata": {"video_id": "clip", "fps": "1.0"},
                            "qa_type": "tal",
                            "is_RC": False,
                            "RC_info": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = runner_args(
                run_dir=str(root / "run"),
                metadata_path=str(metadata),
                frame_root=str(frame_root),
                evaluate=True,
                skip_llm_judge=True,
            )
            with (
                patch(
                    "benchmark.medvidbench.evaluate.evaluate_run", return_value=0
                ) as evaluate_run,
                patch.object(
                    medvidbench,
                    "create_run_dir",
                    wraps=medvidbench.create_run_dir,
                ) as create_run_dir,
            ):
                self.assertEqual(medvidbench.run_benchmark(args, FakeAgent), 0)
            evaluation_args = evaluate_run.call_args.args[0]
            self.assertIsNone(evaluation_args.ground_truth_path)
            self.assertTrue(evaluation_args.skip_llm_judge)
            self.assertEqual(create_run_dir.call_args.args[1], "full")
            manifest = json.loads((root / "run" / "run.json").read_text())
            self.assertEqual(manifest["dataset_variant"], "test")


class MedVidBenchEvaluationTest(unittest.TestCase):
    def test_evaluation_llm_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yml"
            config_path.write_text(
                "evaluation:\n"
                "  llm:\n"
                "    model: config-model\n"
                "    api_base: https://evaluation.example/v1\n"
                "    api_key: secret\n",
                encoding="utf-8",
            )
            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                settings = medvidbench_evaluate.load_evaluation_llm_config()

            self.assertEqual(settings["model"], "config-model")
            self.assertEqual(
                settings["api_base"], "https://evaluation.example/v1"
            )
            self.assertEqual(settings["api_key"], "secret")

    def test_evaluation_llm_config_rejects_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "evaluation.yml"
            config_path.write_text(
                "evaluation:\n"
                "  llm:\n"
                "    model_name: openai/Qwen3.5-4B\n"
                "    api_base: http://127.0.0.1:23333/v1\n"
                "    api_key: dummy\n",
                encoding="utf-8",
            )
            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                with self.assertRaisesRegex(ValueError, "Use model"):
                    medvidbench_evaluate.load_evaluation_llm_config()

    def test_evaluation_llm_config_requires_evaluation_yml(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "evaluation.yml"
            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                with self.assertRaises(FileNotFoundError):
                    medvidbench_evaluate.load_evaluation_llm_config()

    def test_evaluation_llm_configuration_wraps_openai_client(self):
        client_calls = []
        completion_calls = []

        class FakeCompletions:
            def create(self, *args, **kwargs):
                completion_calls.append(kwargs)
                message = types.SimpleNamespace(
                    content="R2: 4\nR3: 4\nR8: 4\nR5: 4\nR4: 4"
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions())
        )

        def create_client(*args, **kwargs):
            client_calls.append(kwargs)
            return client

        fake_openai = types.SimpleNamespace(OpenAI=create_client)
        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}),
        ):
            medvidbench_official_runner._configure_openai(
                "evaluation-model", "https://evaluation.example/v1"
            )
            configured = fake_openai.OpenAI(api_key="secret")
            configured.chat.completions.create(model="gpt-4.1")

        self.assertEqual(
            client_calls[0]["base_url"], "https://evaluation.example/v1"
        )
        self.assertEqual(completion_calls[0]["model"], "evaluation-model")

    def test_ground_truth_selection_and_alignment(self):
        ground_truth = [
            {"id": "first", "qa_type": "tal"},
            {"id": "second", "qa_type": "stg"},
        ]
        selected = medvidbench_evaluate.select_ground_truth(ground_truth, ["1"])
        submission = [
            {"id": "second", "qa_type": "stg", "prediction": "answer"}
        ]
        self.assertEqual(selected, [ground_truth[1]])
        medvidbench_evaluate.validate_alignment(submission, selected)

        submission[0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "id does not match"):
            medvidbench_evaluate.validate_alignment(submission, selected)

    def test_nap_exact_accuracy_uses_only_normalized_full_labels(self):
        submission = [
            {
                "qa_type": "next_action",
                "prediction": "  Lift   Flap with Forceps ",
            },
            {"qa_type": "next_action", "prediction": "closure"},
            {"qa_type": "tal", "prediction": "1-2 seconds"},
        ]
        ground_truth = [
            {
                "conversations": [
                    {"from": "gpt", "value": "lift flap with forceps"}
                ]
            },
            {"conversations": [{"from": "assistant", "value": "design"}]},
            {"conversations": [{"from": "gpt", "value": "1-2 seconds"}]},
        ]

        self.assertEqual(
            medvidbench_evaluate.compute_nap_exact_accuracy(
                submission, ground_truth
            ),
            0.5,
        )

    def test_parse_official_metrics_summary(self):
        output = """
noise
LEADERBOARD METRICS SUMMARY
CVS Assessment - Overall Evaluation
  component_balanced_accuracy: 0.1000
Next Action - Overall Evaluation
  accuracy: 0.2000
Skill Assessment - Overall Evaluation
  aspect_balanced_accuracy: 0.3000
STG - Overall Evaluation
  mean_iou: 0.4000
TAL - Overall Evaluation
  mIoU@0.3: 0.5000
  mIoU@0.5: 0.6000
Dense Video Captioning Metrics
  temporal_f1: 0.7000
  caption_score: 3.8000
VS - Overall Evaluation
  score: 3.9000
RC - Overall Evaluation
  score: 4.0000
END LEADERBOARD METRICS SUMMARY
"""
        self.assertEqual(
            medvidbench_evaluate.parse_official_metrics(output),
            {
                "cvs_acc": 0.1,
                "nap_acc": 0.2,
                "sa_acc": 0.3,
                "stg_miou": 0.4,
                "tag_miou_03": 0.5,
                "tag_miou_05": 0.6,
                "dvc_f1": 0.7,
                "dvc_llm": 3.8,
                "vs_llm": 3.9,
                "rc_llm": 4.0,
            },
        )

    def test_evaluate_completed_run_with_official_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "dataset": "medvidbench",
                        "dataset_variant": "test",
                        "mode": "full",
                        "sample_keys": ["1"],
                        "metadata_path": str(root / "ground_truth.json"),
                        "agent_config": {"model_name": "test-model"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "submission.json").write_text(
                json.dumps(
                    [{"id": "second", "qa_type": "tal", "prediction": "1-2"}]
                ),
                encoding="utf-8",
            )
            (run_dir / "records.jsonl").write_text(
                json.dumps(
                    {
                        "sample_key": "1",
                        "status": "success",
                        "agent_rounds": 2,
                        "token_usage": {"total_tokens": 15},
                        "elapsed_seconds": 1.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    [
                        {"id": "first", "qa_type": "stg"},
                        {"id": "second", "qa_type": "tal"},
                    ]
                ),
                encoding="utf-8",
            )
            leaderboard_dir = root / "leaderboard"
            evaluation_dir = leaderboard_dir / "evaluation"
            evaluation_dir.mkdir(parents=True)
            config_path = root / "config.yml"
            config_path.write_text(
                "evaluation:\n  llm:\n    model: gpt-4.1\n    api_key:\n",
                encoding="utf-8",
            )
            (evaluation_dir / "evaluate_predictions.py").write_text(
                "print('LEADERBOARD METRICS SUMMARY')\n"
                "print('TAL - Overall Evaluation')\n"
                "print('  mIoU@0.3: 0.5000')\n"
                "print('  mIoU@0.5: 0.4000')\n"
                "print('END LEADERBOARD METRICS SUMMARY')\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_dir=str(run_dir),
                ground_truth_path=None,
                leaderboard_dir=str(leaderboard_dir),
                tasks=["tal"],
                skip_llm_judge=True,
            )

            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                self.assertEqual(medvidbench_evaluate.evaluate_run(args), 0)
            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            self.assertEqual(
                evaluation["metrics"],
                {"tag_miou_03": 0.5, "tag_miou_05": 0.4},
            )
            self.assertNotIn("prediction_format", evaluation)
            self.assertTrue((run_dir / "evaluation.log").is_file())
            self.assertFalse((run_dir / "summary.json").exists())
            record = medvidbench_utils.load_latest_records(
                run_dir / "records.jsonl"
            )["1"]
            self.assertIn("ground_truth", record)
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn(
                "| TAL | videoseek | test | full | test-model | 1/1 |",
                report,
            )
            self.assertIn("0.5000", report)
            self.assertIn("0.4000", report)
            self.assertIn("Generated", report)
            self.assertNotIn("Structured-output coverage", report)

            (evaluation_dir / "evaluate_predictions.py").write_text(
                "print('LEADERBOARD METRICS SUMMARY')\n"
                "print('TAL - Overall Evaluation')\n"
                "print('  mIoU@0.3: 0.5000')\n"
                "print('END LEADERBOARD METRICS SUMMARY')\n",
                encoding="utf-8",
            )
            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                self.assertEqual(medvidbench_evaluate.evaluate_run(args), 1)
            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            self.assertEqual(evaluation["status"], "failed")
            self.assertEqual(evaluation["missing_metrics"], ["tag_miou_05"])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn(
                "| test-model | N/A | N/A | N/A | N/A | 0.5000 | Failed |", report
            )

            args.tasks = ["dvc"]
            args.skip_llm_judge = False
            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                self.assertEqual(medvidbench_evaluate.evaluate_run(args), 1)
            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            self.assertIn("evaluation.llm.api_key", evaluation["errors"][0])

    def test_evaluation_distinguishes_absent_tasks_from_failed_samples(self):
        cases = [
            # VS state, explicit tasks, skip Judge, expected status, VS display
            (None, None, False, "complete", "N/A"),
            ("failed", None, False, "failed", "Failed"),
            ("success", None, False, "complete", "4.0000"),
            ("success", None, True, "complete", "Skipped"),
            ("success", ["tal"], False, "complete", "—"),
            (None, ["tal", "vs"], False, "failed", "Failed"),
        ]
        for vs_state, tasks, skip_judge, expected_status, vs_display in cases:
            with self.subTest(vs_state=vs_state, tasks=tasks, skip_judge=skip_judge), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                samples = [{"id": "tal", "qa_type": "tal"},
                           {"id": "rc", "qa_type": "region_caption_gemini"}]
                if vs_state is not None:
                    samples.append({"id": "vs", "qa_type": "video_summary_gpt"})
                keys = [str(i) for i in range(len(samples))]
                reference_path = root / "ground_truth.json"
                reference_path.write_text(json.dumps(samples))
                (root / "run.json").write_text(json.dumps({
                    "dataset": "medvidbench", "dataset_variant": "test", "mode": "debug",
                    "sample_keys": keys, "metadata_path": str(reference_path),
                    "agent_config": {"model_name": "test-model"},
                }))
                (root / "submission.json").write_text(json.dumps([
                    {**sample, "prediction": "test"} for sample in samples
                ]))
                (root / "records.jsonl").write_text("".join(
                    json.dumps({"sample_key": key,
                                "status": vs_state if sample["id"] == "vs" else "success"}) + "\n"
                    for key, sample in zip(keys, samples)
                ))
                evaluator = root / "leaderboard/evaluation/evaluate_predictions.py"
                evaluator.parent.mkdir(parents=True)
                evaluator.touch()
                output = ("LEADERBOARD METRICS SUMMARY\nTAL - Overall Evaluation\n"
                          "  mIoU@0.3: 0.5\n  mIoU@0.5: 0.4\n")
                if not skip_judge and tasks is None:
                    output += "RC - Overall Evaluation\n  score: 3.4\n"
                    if vs_state == "success":
                        output += "VS - Overall Evaluation\n  score: 4.0\n"
                output += "END LEADERBOARD METRICS SUMMARY\n"
                settings = {"config_path": "test", "model": "test-model", "api_base": None,
                            "api_key": "dummy", "max_workers": 2, "enable_thinking": False,
                            "max_completion_tokens": 2048, "timeout": 120}
                args = argparse.Namespace(run_dir=str(root), ground_truth_path=None,
                                          leaderboard_dir=str(root / "leaderboard"),
                                          tasks=tasks, skip_llm_judge=skip_judge)
                with patch.object(medvidbench_evaluate, "load_evaluation_llm_config", return_value=settings), \
                        patch.object(medvidbench_evaluate, "run_official_evaluation", return_value=output):
                    self.assertEqual(medvidbench_evaluate.evaluate_run(args),
                                     int(expected_status == "failed"))
                result = json.loads((root / "evaluation.json").read_text())
                self.assertEqual(result["status"], expected_status)
                self.assertEqual("vs_llm" in result["not_applicable_metrics"], vs_display == "N/A")
                self.assertEqual("vs_llm" in result["missing_metrics"], vs_display == "Failed")
                self.assertEqual("vs_llm" in result["expected_metrics"],
                                 vs_display in {"Failed", "4.0000"})
                self.assertEqual(result["metrics"]["tag_miou_03"], .5)
                if vs_state == "failed":
                    self.assertTrue(any("vs: all selected samples failed" in e for e in result["errors"]))
                report = (root / "report.md").read_text()
                model_row = next(line for line in report.splitlines() if line.startswith("| test-model |"))
                self.assertEqual(model_row.split("|")[-3].strip(), vs_display)
                self.assertEqual("N/A: no samples" in report, bool(result["not_applicable_metrics"]))

    def test_nap_exact_metric_preserves_official_score_and_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "sample",
                            "qa_type": "next_action",
                            "conversations": [
                                {"from": "human", "value": "What is next?"},
                                {"from": "gpt", "value": "design"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "dataset": "medvidbench",
                        "dataset_variant": "test",
                        "mode": "debug",
                        "sample_keys": ["0"],
                        "metadata_path": str(ground_truth_path),
                        "agent_config": {"model_name": "test-model"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "submission.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "sample",
                            "qa_type": "next_action",
                            "prediction": "closure",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "records.jsonl").write_text(
                json.dumps(
                    {
                        "sample_key": "0",
                        "status": "success",
                        "agent_rounds": 2,
                        "token_usage": {"total_tokens": 15},
                        "elapsed_seconds": 1.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            leaderboard_dir = root / "leaderboard"
            evaluation_dir = leaderboard_dir / "evaluation"
            evaluation_dir.mkdir(parents=True)
            (evaluation_dir / "evaluate_predictions.py").write_text(
                "print('LEADERBOARD METRICS SUMMARY')\n"
                "print('Next Action - Overall Evaluation')\n"
                "print('  accuracy: 1.0000')\n"
                "print('END LEADERBOARD METRICS SUMMARY')\n",
                encoding="utf-8",
            )
            config_path = root / "config.yml"
            config_path.write_text(
                "evaluation:\n  llm:\n    model: gpt-4.1\n    api_key:\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_dir=str(run_dir),
                ground_truth_path=None,
                leaderboard_dir=str(leaderboard_dir),
                tasks=["next_action"],
                skip_llm_judge=True,
            )

            with patch.object(
                medvidbench_evaluate, "LOCAL_CONFIG_PATH", config_path
            ):
                self.assertEqual(medvidbench_evaluate.evaluate_run(args), 0)

            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            self.assertEqual(evaluation["metrics"]["nap_acc"], 1.0)
            self.assertEqual(evaluation["metrics"]["nap_exact_acc"], 0.0)
            self.assertIn("differs from local exact", evaluation["warnings"][0])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Local Diagnostic Metrics", report)
            self.assertIn("NAP_exact_acc", report)
            self.assertIn("1.0000", report)
            self.assertIn("0.0000", report)

    def test_leaderboard_test_run_cannot_be_evaluated_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "dataset": "medvidbench",
                        "dataset_variant": "leaderboard_test",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_dir=str(run_dir),
                ground_truth_path=None,
                leaderboard_dir="unused",
                tasks=None,
                skip_llm_judge=True,
            )
            with self.assertRaisesRegex(ValueError, "no local Ground Truth"):
                medvidbench_evaluate.evaluate_run(args)


if __name__ == "__main__":
    unittest.main()
