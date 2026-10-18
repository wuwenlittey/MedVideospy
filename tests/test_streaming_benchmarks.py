import json
import tempfile
import unittest
from pathlib import Path

from benchmark.streaming_cutoff import QueryCutoffVideoReader, parse_timestamp
from benchmark.streameqa.run import load_dataset as load_streameqa
from benchmark.streameqa.run import parse_args as parse_streameqa_args
from benchmark.streamingbench.run import load_dataset as load_streamingbench
from benchmark.streamingbench.run import resolve_video


class _FakeReader:
    def __init__(self, frames: int = 100, fps: float = 10.0) -> None:
        self._frames = frames
        self._fps = fps

    def get_avg_fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return self._frames

    def get_batch(self, frame_indices):
        return [self._value(index) for index in frame_indices]

    def _value(self, index: int) -> int:
        if index < 0 or index >= self._frames:
            raise IndexError(index)
        return index


class StreamingBenchmarkTests(unittest.TestCase):
    def test_streameqa_cli_accepts_smoke_mode(self):
        args = parse_streameqa_args(["--mode", "smoke", "--split", "Perception"])
        self.assertEqual(args.mode, "smoke")

    def test_parse_timestamp_accepts_seconds_and_clock(self):
        self.assertEqual(parse_timestamp(12.5), 12.5)
        self.assertEqual(parse_timestamp("01:02:03"), 3723.0)

    def test_parse_timestamp_rejects_negative_and_missing(self):
        with self.assertRaises(ValueError):
            parse_timestamp(-0.1)
        with self.assertRaises(ValueError):
            parse_timestamp(None)

    def test_query_cutoff_hides_frames_after_query_time(self):
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "clip.mp4"
            video.write_bytes(b"")
            reader = QueryCutoffVideoReader(video, video_reader=_FakeReader())
            reader.set_query_time(2.5)
            self.assertEqual(len(reader), 26)
            self.assertEqual(reader.get_batch([25]), [25])
            with self.assertRaises(IndexError):
                reader.get_batch([26])

    def test_streameqa_skips_items_without_query_time_or_four_options(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Interaction.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "ok",
                            "task": "action",
                            "video": "P01-1",
                            "question": "What happens?",
                            "options": ["a", "b", "c", "d"],
                            "gt": 1,
                            "realtime": 3.0,
                        },
                        {
                            "id": "bad-options",
                            "task": "action",
                            "video": "P01-1",
                            "question": "Missing choice",
                            "options": ["a", "b", "c"],
                            "gt": 3,
                            "realtime": 1.0,
                        },
                        {
                            "id": "negative",
                            "task": "action",
                            "video": "P01-1",
                            "question": "Before start",
                            "options": ["a", "b", "c", "d"],
                            "gt": 0,
                            "realtime": -1,
                        },
                        {
                            "id": "null-time",
                            "task": "plan",
                            "video": "P01-1",
                            "question": "No timestamp",
                            "options": ["a", "b", "c", "d"],
                            "gt": 0,
                            "realtime": None,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_streameqa(path, "Interaction")
        self.assertEqual([item["sample_key"] for item in samples], ["Interaction:ok"])
        self.assertEqual(samples[0]["answer"], "B")
        self.assertEqual(samples[0]["query_time"], 3.0)

    def test_streamingbench_loads_questions_and_resolves_flat_mp4(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video_root = root / "videos"
            video_root.mkdir()
            (video_root / "sample_316_real.mp4").write_bytes(b"")
            path = root / "questions_real.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "video_path": "somewhere/sample_316_real.mp4",
                            "questions": [
                                {
                                    "question": "What is visible?",
                                    "time_stamp": "00:01",
                                    "answer": "C",
                                    "options": ["a", "b", "c", "d"],
                                    "task_type": "Object",
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_streamingbench(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["video_id"], "sample_316")
            self.assertEqual(samples[0]["query_time"], 1.0)
            self.assertEqual(
                resolve_video(samples[0], video_root),
                video_root / "sample_316_real.mp4",
            )

    def test_streamingbench_flattens_sqa_and_skips_proactive_items(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "questions_sqa.json"
            path.write_text(
                json.dumps(
                    [
                        [
                            {
                                "video_path": "./videos/sample_17_sqa.mp4",
                                "questions": [
                                    {
                                        "question": "Who serves?",
                                        "time_stamp": "00:00:36",
                                        "answer": "A",
                                        "options": ["a", "b", "c", "d"],
                                        "task_type": "Sequential Question Answering",
                                    },
                                    {
                                        "question": "Say 3",
                                        "time_stamp": "00:00:20",
                                        "task_type": "Proactive Output",
                                        "ground_truth_output": "3",
                                    },
                                ],
                            }
                        ]
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_streamingbench(path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["category"], "Sequential Question Answering")
        self.assertEqual(samples[0]["answer"], "A")


if __name__ == "__main__":
    unittest.main()
