import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import decord  # noqa: F401
except ModuleNotFoundError:
    decord_stub = types.ModuleType("decord")
    decord_stub.VideoReader = object
    sys.modules["decord"] = decord_stub

from benchmark.egolife import run as egolife
from benchmark.egolife.timeline import EgoLifeTimelineReader, parse_egolife_time


class FakeBatch:
    def __init__(self, values):
        self.values = values

    def asnumpy(self):
        return np.array(self.values, dtype=np.uint8)


class FakeReader:
    def __init__(self, path):
        self.value = 1 if "11000000" in str(path) else 2

    def __len__(self):
        return 20

    def get_avg_fps(self):
        return 2.0

    def get_batch(self, indices):
        return FakeBatch(
            [np.full((2, 2, 3), self.value, dtype=np.uint8) for _ in indices]
        )


class FakeTrajectory:
    def to_dict(self):
        return {"final_answer": "B", "steps": []}


class FakeAgent:
    durations = []

    def __init__(self, **kwargs):
        self.duration = len(kwargs["video_reader"]) - 1
        self.subtitles = []
        self.task_type = "general"
        self.last_run_rounds = 0
        self.last_run_token_usage = {}

    def run(self, question):
        type(self).durations.append(self.duration)
        return FakeTrajectory()


class ConcurrentAgent(FakeAgent):
    barrier = None

    def run(self, question):
        type(self).barrier.wait(timeout=5)
        return super().run(question)


class EgoLifeTimelineTest(unittest.TestCase):
    def test_lazy_timeline_maps_global_seconds_to_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DAY1_A1_JAKE_11000000.mp4").touch()
            (root / "DAY1_A1_JAKE_11010000.mp4").touch()
            with patch("benchmark.egolife.timeline.VideoReader", FakeReader):
                reader = EgoLifeTimelineReader(root)
                first = reader.get_batch([0]).asnumpy()[0]
                second_time = reader.to_timeline_seconds(
                    parse_egolife_time("DAY1@11010000")
                )
                second = reader.get_batch([second_time]).asnumpy()[0]

            self.assertTrue(np.all(first == 1))
            self.assertTrue(np.all(second == 2))
            self.assertEqual(reader.get_avg_fps(), 1.0)
            self.assertEqual(reader.to_wall_seconds(second_time), parse_egolife_time('DAY1@11010000'))
            self.assertEqual(reader.time_mapping['segments'][1]['start'], second_time)

    def test_query_time_limits_future_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DAY1_A1_JAKE_11000000.mp4").touch()
            (root / "DAY1_A1_JAKE_11010000.mp4").touch()
            with patch("benchmark.egolife.timeline.VideoReader", FakeReader):
                reader = EgoLifeTimelineReader(root)
                reader.set_query_time({"date": "DAY1", "time": "11004500"})

                self.assertEqual(len(reader), 30)
                with self.assertRaises(IndexError):
                    reader.get_batch([30])


class EgoLifeRunnerTest(unittest.TestCase):
    def test_worker_defaults_and_validation(self):
        self.assertEqual(egolife.parse_args([]).workers, 4)
        self.assertEqual(egolife.parse_args(["--workers", "2"]).workers, 2)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            egolife.parse_args(["--workers", "0"])

    def test_official_dataset_manifest_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_path = root / "qa.json"
            qa_path.write_text(
                json.dumps(
                    [
                        {
                            "ID": "1",
                            "question": "Who arrived?",
                            "choice_a": "A person",
                            "choice_b": "Nobody",
                            "choice_c": "A dog",
                            "choice_d": "A cat",
                            "answer": "A",
                            "type": "EntityLog",
                            "query_time": {"date": "DAY1", "time": "12000000"},
                            "target_time": {"date": "DAY1", "time": "11590000"},
                            "keywords": "arrive",
                        },
                        {
                            "ID": "2",
                            "question": "What happened?",
                            "choice_a": "One",
                            "choice_b": "Two",
                            "choice_c": "Three",
                            "choice_d": "Four",
                            "answer": "B",
                            "type": "EventRecall",
                            "query_time": {"date": "DAY2", "time": "12000000"},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text('["2", "1"]', encoding="utf-8")
            samples = egolife.load_dataset(qa_path)
            selected = egolife.select_samples(samples, "full", manifest)
            metrics = egolife.compute_metrics(
                samples,
                {
                    "1": {"prediction": "A"},
                    "2": {"prediction": "C"},
                },
            )

            self.assertEqual(
                [item["sample_key"] for item in selected], ["2", "1"]
            )
            self.assertEqual(samples[1]["answer"], "B")
            self.assertEqual(samples[1]["day"], "DAY2")
            self.assertNotIn("target_time", samples[0])
            self.assertNotIn("keywords", samples[0])
            self.assertEqual(metrics["overall"]["accuracy"], 0.5)
            self.assertEqual(metrics["macro_average"], 0.5)

    def test_official_transcript_uses_hour_file_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "DAY1_A1_JAKE_11000000.mp4").touch()
            transcript_root = root / "transcripts" / "DAY1"
            transcript_root.mkdir(parents=True)
            (transcript_root / "A1_JAKE_DAY1_11000000.srt").write_text(
                "1\n00:00:05,000 --> 00:00:07,000\nJake: Hello.\n",
                encoding="utf-8",
            )

            with patch("benchmark.egolife.timeline.VideoReader", FakeReader):
                timeline = EgoLifeTimelineReader(video_root)
                transcripts = egolife.load_transcripts(
                    transcript_root.parent, timeline
                )

            self.assertEqual(transcripts[0]["start_time"], 5.0)
            self.assertEqual(transcripts[0]["end_time"], 7.0)
            self.assertEqual(transcripts[0]["subtitle"], "Jake: Hello.")

    def test_committed_train_and_test_splits_are_disjoint(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "experiments/data_splits/main/egolife"
        )
        train = json.loads((root / "train.json").read_text())
        test = json.loads((root / "test.json").read_text())

        self.assertEqual((len(train), len(test)), (405, 95))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(train, [str(index) for index in range(1, 406)])
        self.assertEqual(test, [str(index) for index in range(406, 501)])

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "data/EgoLife/lmms-lab_EgoLife/EgoLifeQA/"
            "EgoLifeQA_A1_JAKE.json"
        ).is_file(),
        "EgoLifeQA metadata is not available",
    )
    def test_committed_split_is_date_disjoint(self):
        repository = Path(__file__).resolve().parents[1]
        qa_path = (
            repository
            / "data/EgoLife/lmms-lab_EgoLife/EgoLifeQA/"
            "EgoLifeQA_A1_JAKE.json"
        )
        samples = json.loads(qa_path.read_text(encoding="utf-8"))
        days = {str(item["ID"]): item["query_time"]["date"] for item in samples}
        split_root = repository / "experiments/data_splits/main/egolife"
        train = json.loads((split_root / "train.json").read_text())
        test = json.loads((split_root / "test.json").read_text())

        self.assertEqual(set(train) | set(test), set(days))
        self.assertEqual(
            {days[key] for key in train}, {f"DAY{day}" for day in range(1, 6)}
        )
        self.assertEqual({days[key] for key in test}, {"DAY6", "DAY7"})

    def test_runner_reuses_agent_with_query_time_limit(self):
        class ContextAgent(FakeAgent):
            contexts = []

            def run(self, question):
                self.contexts.append(self.memory_context.copy())
                return super().run(question)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_path = root / "qa.json"
            samples = []
            for identifier, clock in (("1", "11000500"), ("2", "11010500")):
                samples.append(
                    {
                        "ID": identifier,
                        "query_time": {"date": "DAY1", "time": clock},
                        "type": "EventRecall",
                        "question": "What happened?",
                        "choice_a": "One",
                        "choice_b": "Two",
                        "choice_c": "Three",
                        "choice_d": "Four",
                        "answer": "B",
                    }
                )
            qa_path.write_text(json.dumps(samples), encoding="utf-8")

            video_root = root / "A1_JAKE"
            video_root.mkdir()
            (video_root / "DAY1_A1_JAKE_11000000.mp4").touch()
            (video_root / "DAY1_A1_JAKE_11010000.mp4").touch()
            transcript_root = root / "transcripts"
            transcript_root.mkdir()
            (transcript_root / "A1_JAKE_DAY1_11000000.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nJake: Hello.\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            manifest = root / "test.json"
            manifest.write_text(json.dumps([str(s["ID"]) for s in json.loads(qa_path.read_text())]))
            args = egolife.parse_args(
                [
                    "--sample-manifest",
                    str(manifest),
                    "--mode",
                    "full",
                    "--qa-path",
                    str(qa_path),
                    "--video-root",
                    str(video_root),
                    "--transcript-root",
                    str(transcript_root),
                    "--run-dir",
                    str(run_dir),
                    "--workers",
                    "1",
                ]
            )
            config = {
                "agent": {"model": {"model_name": "fake"}},
                "tools": {"enabled": []},
            }
            FakeAgent.durations = []
            with (
                patch("benchmark.egolife.timeline.VideoReader", FakeReader),
                patch.object(egolife, "build_agent_config", return_value=config),
            ):
                status = egolife.run_benchmark(args, ContextAgent)

            metrics = json.loads((run_dir / "metrics.json").read_text())
            predictions = json.loads((run_dir / "predictions.json").read_text())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            run_log = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(status, 0)
        self.assertEqual(FakeAgent.durations, [5.0, 35.0])
        self.assertEqual([context["query_time"] for context in ContextAgent.contexts], [sample["query_time"] for sample in samples])
        self.assertTrue(all(context["source_id"] == "egolife:A1_JAKE" for context in ContextAgent.contexts))
        self.assertEqual([context['query'] for context in ContextAgent.contexts], [sample['question'] for sample in samples])
        self.assertEqual(len({context['timeline_id'] for context in ContextAgent.contexts}), 1)
        self.assertLess(ContextAgent.contexts[0]["available_at"], ContextAgent.contexts[1]["available_at"])
        self.assertEqual(predictions, {"1": "B", "2": "B"})
        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertIn("# EgoLifeQA Run Report", report)
        self.assertIn("| Overall | videospy | full | fake |", report)
        self.assertIn("| EventRecall | videospy | full | fake |", report)
        self.assertIn("| Accuracy (%) | Avg. rounds | Avg. tokens |", report)
        self.assertIn("| 100.0 | 0.0 | 0.0 |", report)
        self.assertIn("- Macro Average (%): 100.0", report)
        self.assertIn("Starting EgoLifeQA", run_log)
        self.assertIn("Sample 1: success", run_log)
        self.assertIn("Completed: 2/2", run_log)

    def test_samples_run_concurrently_with_isolated_timelines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa_path = root / "qa.json"
            qa_path.write_text(
                json.dumps(
                    [
                        {
                            "ID": identifier,
                            "query_time": {"date": "DAY1", "time": clock},
                            "type": "EventRecall",
                            "question": "What happened?",
                            "choice_a": "One",
                            "choice_b": "Two",
                            "choice_c": "Three",
                            "choice_d": "Four",
                            "answer": "B",
                        }
                        for identifier, clock in (
                            ("1", "11000500"),
                            ("2", "11010500"),
                        )
                    ]
                ),
                encoding="utf-8",
            )
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "DAY1_A1_JAKE_11000000.mp4").touch()
            (video_root / "DAY1_A1_JAKE_11010000.mp4").touch()
            transcript_root = root / "transcripts"
            transcript_root.mkdir()
            (transcript_root / "A1_JAKE_DAY1_11000000.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nJake: Hello.\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            manifest = root / "test.json"
            manifest.write_text(json.dumps([str(s["ID"]) for s in json.loads(qa_path.read_text())]))
            args = egolife.parse_args(
                [
                    "--sample-manifest",
                    str(manifest),
                    "--mode",
                    "full",
                    "--qa-path",
                    str(qa_path),
                    "--video-root",
                    str(video_root),
                    "--transcript-root",
                    str(transcript_root),
                    "--run-dir",
                    str(run_dir),
                    "--workers",
                    "2",
                ]
            )
            config = {
                "agent": {"model": {"model_name": "fake"}},
                "tools": {"enabled": []},
            }
            ConcurrentAgent.barrier = threading.Barrier(2)
            ConcurrentAgent.durations = []
            with (
                patch("benchmark.egolife.timeline.VideoReader", FakeReader),
                patch.object(egolife, "build_agent_config", return_value=config),
            ):
                status = egolife.run_benchmark(args, ConcurrentAgent)

            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text().splitlines()
            ]
            trajectory_files_exist = all(
                (run_dir / record["trajectory"]).is_file() for record in records
            )

        self.assertEqual(status, 0)
        self.assertEqual(sorted(ConcurrentAgent.durations), [5.0, 35.0])
        self.assertEqual({record["sample_key"] for record in records}, {"1", "2"})
        self.assertTrue(all(record["status"] == "success" for record in records))
        self.assertTrue(trajectory_files_exist)

if __name__ == "__main__":
    unittest.main()
