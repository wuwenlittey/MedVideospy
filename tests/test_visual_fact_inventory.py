import json
import tempfile
import unittest
from pathlib import Path

from videospy.visual_memory.inventory import inventory_sources


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trajectories = self.root / "trajectories"
        self.trajectories.mkdir()
        self.dataset = self.root / "dataset.json"
        self.splits = self.root / "splits.jsonl"
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"inventory-only-does-not-decode")
        self.rows = [{"key": "source-a", "video_path_original": "a.mp4", "qa": [
            {"uid": 1, "question": "Which tool?", "task_name": "instrument_onset",
             "answer": "A", "options": ["Holder", "Scissors"]},
            {"uid": 2, "question": "Which action?", "task_name": "action recognition",
             "answer": "B", "options": ["Pulling", "Cutting"]}]}]
        self.partitions = [{"dataset_video_key": "source-a", "source_video_id": "original-a",
                            "split": "train", "video_path": "video.mp4"}]
        for uid, question, correct in [(1, "Which tool?", True), (2, "Which action?", False)]:
            (self.trajectories / f"{uid}.json").write_text(json.dumps({
                "question": question, "messages": [], "final_answer": "A", "correct": correct}), encoding="utf-8")

    def run_inventory(self):
        self.dataset.write_text(json.dumps(self.rows), encoding="utf-8")
        self.splits.write_text("".join(json.dumps(row) + "\n" for row in self.partitions), encoding="utf-8")
        return inventory_sources(self.trajectories, self.dataset, self.splits, self.root / "inventory")

    def sources(self, report):
        return [json.loads(line) for line in Path(report["sources_path"]).read_text(encoding="utf-8").splitlines()]

    def test_success_and_failure_are_both_included_and_gold_not_prediction(self):
        report = self.run_inventory()
        self.assertEqual(report["sources"], 2)
        self.assertEqual(report["trajectory_outcomes"], {"correct": 1, "incorrect": 1})
        self.assertEqual(self.sources(report)[1]["gold_option"], "B")
        self.assertEqual(len(report["tasks"]), 2)
        self.assertEqual(self.sources(report)[0]["source_video_id"], "original-a")

    def test_explicit_dataset_fact_annotations_are_preserved_without_inference(self):
        annotation = {"annotation_id": "supplied-event", "fact_type": "phase",
                      "label_source": {"kind": "dataset_gold"},
                      "alignment": {"kind": "interval", "start": 10, "end": 15,
                                    "scope": "phase_interval", "source": "dataset_annotation",
                                    "reference": "dataset/events/1"}}
        self.rows[0]["qa"][0]["visual_fact_annotations"] = [annotation]
        self.rows[0]["qa"][1]["start_time"] = 10  # Ambiguous fields do not create annotations.
        rows = self.sources(self.run_inventory())
        self.assertEqual(rows[0]["supervision"], [annotation])
        self.assertEqual(rows[0]["question"], "Which tool?")
        self.assertNotIn("supervision", rows[1])

    def test_test_partition_never_becomes_training_by_qa_parity(self):
        self.partitions[0]["split"] = "test"
        report = self.run_inventory()
        self.assertEqual(report["sources"], 0)
        self.assertEqual(report["skip_reasons"], {"held_out_test_video": 2})

    def test_missing_partition_is_reported_not_guessed(self):
        self.partitions[0]["dataset_video_key"] = "other"
        report = self.run_inventory()
        self.assertEqual(report["skip_reasons"], {"missing_explicit_video_partition": 2})

    def test_cross_partition_video_and_patient_rejected(self):
        self.partitions.append({"dataset_video_key": "another", "source_video_id": "original-a", "split": "test"})
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            self.run_inventory()
        self.partitions[1]["source_video_id"] = "original-b"
        self.partitions[0]["patient_id"] = self.partitions[1]["patient_id"] = "dataset:patient-1"
        with self.assertRaisesRegex(ValueError, "Patient"):
            self.run_inventory()

    def test_wrong_question_and_nontrajectory_are_reported(self):
        path = self.trajectories / "1.json"
        value = json.loads(path.read_text())
        value["question"] = "Unrelated question"
        path.write_text(json.dumps(value))
        (self.trajectories / "metrics.json").write_text("{}")
        report = self.run_inventory()
        self.assertEqual(report["sources"], 1)
        self.assertEqual(report["skip_reasons"]["dataset_trajectory_question_mismatch"], 1)
        self.assertEqual(report["skip_reasons"]["not_a_trajectory"], 1)

    def test_frame_source_paths_are_preserved_with_absolute_root(self):
        self.partitions[0].pop("video_path")
        self.partitions[0].update(frame_paths=["one.png", "two.png"], frame_root="frames", fps=1)
        source = self.sources(self.run_inventory())[0]
        self.assertEqual(source["frame_paths"], ["one.png", "two.png"])
        self.assertEqual(source["frame_root"], str((self.root / "frames").resolve()))

    def test_existing_output_refused(self):
        self.run_inventory()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_inventory()

    def test_ambiguous_video_alias_is_not_arbitrarily_selected(self):
        self.partitions.append({**self.partitions[0], "dataset_video_key": "a.mp4"})
        report = self.run_inventory()
        self.assertEqual(report["skip_reasons"], {"ambiguous_video_partition_aliases": 2})

    def test_same_media_under_different_ids_cannot_cross_partitions(self):
        self.partitions.append({"dataset_video_key": "another", "source_video_id": "original-b",
                                "split": "test", "video_path": "video.mp4"})
        with self.assertRaisesRegex(ValueError, "Same resolved media"):
            self.run_inventory()

    def test_mismatching_recorded_video_is_rejected(self):
        path = self.trajectories / "1.json"
        value = json.loads(path.read_text())
        value["video_path"] = "/other/video.mp4"
        path.write_text(json.dumps(value))
        report = self.run_inventory()
        self.assertEqual(report["sources"], 1)
        self.assertEqual(report["skip_reasons"]["dataset_trajectory_video_identity_mismatch"], 1)

    def test_dataset_question_embedded_in_unrelated_text_not_accepted(self):
        path = self.trajectories / "1.json"
        value = json.loads(path.read_text())
        value["question"] = "Ignore the old question: Which tool? Answer a different question."
        path.write_text(json.dumps(value))
        report = self.run_inventory()
        self.assertEqual(report["sources"], 1)


if __name__ == "__main__":
    unittest.main()
