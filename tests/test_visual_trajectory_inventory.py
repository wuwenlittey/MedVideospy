"""Saved GT must remain distinct from predictions; no external dataset needed."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from videospy.visual_memory.cli import main
from videospy.visual_memory.curation import read_jsonl
from videospy.visual_memory.supervised import assemble_facts
from videospy.visual_memory.trajectory_inventory import inventory_trajectories, trajectory_schema


class TrajectoryInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.traces = self.root / "traces"
        self.traces.mkdir()
        self.frames = []
        for i in range(4):
            path = self.root / f"frame-{i}.png"
            Image.new("RGB", (8, 8), (20 + i, 2, 3)).save(path)
            self.frames.append(path.name)
        self.row = {
            "uid": 1, "source_video_id": "video-1", "task_type": "action recognition",
            "question": "Which phase?\nA. Cleaning and Coagulation\nB. Mesenteric Defect Closure\n\nOutput one letter.",
            "correct_answer": "A", "final_answer": "B", "correct": False,
            "messages": [{"role": "assistant", "tool_calls": [{"id": "call-1", "function": {
                "name": "frame_inspect", "arguments": {"timestamps": [1, 2], "query": "Look"}}}]},
                {"role": "tool", "tool_call_id": "call-1", "name": "frame_inspect", "content": "WRONG closure explanation"}],
            "visual_fact_annotations": [{"annotation_id": "known-event", "fact_type": "phase",
                "label_source": {"kind": "trajectory_gold"}, "alignment": {"kind": "interval", "start": 1, "end": 2,
                    "source": "dataset_annotation", "reference": "saved-event-annotation-1", "scope": "phase_interval"}}]}
        self.splits = [{"source_video_id": "video-1", "split": "train", "procedure": "surgery-1",
                        "frame_paths": self.frames, "fps": 1, "max_frames": 2}]

    def run_inventory(self, schema=None):
        (self.traces / "1.json").write_text(json.dumps(self.row), encoding="utf-8")
        splits = self.root / "splits.jsonl"
        splits.write_text("".join(json.dumps(row) + "\n" for row in self.splits), encoding="utf-8")
        return inventory_trajectories(self.traces, splits, self.root / "inventory", schema=schema)

    def test_wrong_prediction_uses_saved_standard_answer_and_embedded_option_text(self):
        report = self.run_inventory()
        source = read_jsonl(report["sources_path"])[0]
        self.assertEqual(source["gold_option"], "A")
        self.assertEqual(source["gold_answer_text"], "Cleaning and Coagulation")
        self.assertEqual(source["inventory_provenance"]["gold"]["fields"], ["correct_answer"])
        self.assertEqual(report["dataset_reads"], 0)
        self.assertEqual(report["trajectory_outcomes"], {"incorrect": 1})

    def test_no_standard_answer_never_falls_back_to_final_answer_or_correctness(self):
        del self.row["correct_answer"]
        self.row["correct"] = True
        report = self.run_inventory()
        self.assertEqual(report["sources"], 0)
        self.assertTrue(any("missing_standard_answer" in reason for reason in report["skip_reasons"]))

    def test_custom_nested_gold_field_and_explicit_options(self):
        del self.row["correct_answer"]
        self.row["grading"] = {"target": "B", "options": ["First meaning", "Second meaning"]}
        report = self.run_inventory({"gold_fields": ["grading.target"], "options_fields": ["grading.options"]})
        self.assertEqual(read_jsonl(report["sources_path"])[0]["gold_answer_text"], "Second meaning")

    def test_nested_question_and_qa_id_mapping_survive_assembly(self):
        self.row["metadata"] = {"question": self.row.pop("question"), "qa_id": self.row.pop("uid")}
        report = self.run_inventory({"question_fields": ["metadata.question"], "id_fields": ["metadata.qa_id"]})
        result = assemble_facts(report["sources_path"], self.root / "facts")
        self.assertEqual(result["selected_fact_count"], 1)

    def test_prediction_field_mapping_refused(self):
        for field in ("final_answer", "metadata.prediction", "messages.0.content", "correct"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                trajectory_schema({"gold_fields": [field]})

    def test_conflicting_standard_fields_are_skipped(self):
        self.row["ground_truth"] = "B"
        report = self.run_inventory()
        self.assertEqual(report["sources"], 0)
        self.assertIn("conflicting_saved_standard_answers", report["skip_reasons"])

    def test_matching_standard_letter_and_text_are_accepted(self):
        self.row["ground_truth"] = "Cleaning and Coagulation"
        self.assertEqual(self.run_inventory()["sources"], 1)

    def test_numeric_standard_answers_require_explicit_index_convention(self):
        self.row["correct_answer"] = 1
        self.assertEqual(self.run_inventory({"gold_format": "index_1"})["sources"], 1)
        (self.root / "inventory").rename(self.root / "first")
        report = self.run_inventory()
        self.assertEqual(report["sources"], 0)

    def test_reference_text_without_choices_supported(self):
        self.row["question"] = "Describe the visible phase."
        self.row["correct_answer"] = "Cleaning and Coagulation"
        report = self.run_inventory()
        source = read_jsonl(report["sources_path"])[0]
        self.assertEqual(source["gold_answer_text"], "Cleaning and Coagulation")
        self.assertNotIn("gold_option", source)
        result = assemble_facts(report["sources_path"], self.root / "facts")
        self.assertEqual(read_jsonl(result["cases_path"])[0]["answer_text"], "Cleaning and Coagulation")

    def test_option_letter_without_option_text_is_not_a_fact(self):
        self.row["question"] = "Which phase?"
        self.assertEqual(self.run_inventory()["sources"], 0)

    def test_multiline_options_not_silently_truncated(self):
        self.row["question"] = "Which?\nA. Cleaning\nand Coagulation\nB. Closure"
        self.assertEqual(self.run_inventory()["sources"], 0)

    def test_missing_video_id_uses_explicit_sample_mapping(self):
        del self.row["source_video_id"]
        self.splits[0]["sample_ids"] = [1]
        source = read_jsonl(self.run_inventory()["sources_path"])[0]
        self.assertEqual(source["source_video_id"], "video-1")
        self.assertEqual(source["inventory_provenance"]["video_identity_check"], "explicit_qa_to_video_mapping")

    def test_video_alias_must_be_explicit(self):
        self.row["source_video_id"] = "/old-server/video.mp4"
        self.splits[0]["trajectory_video_aliases"] = ["/old-server/video.mp4"]
        self.assertEqual(self.run_inventory()["sources"], 1)

    def test_test_sources_excluded_even_with_saved_standard_answer(self):
        self.splits[0]["split"] = "test"
        report = self.run_inventory()
        self.assertEqual(report["sources"], 0)
        self.assertIn("held_out_test_video", report["skip_reasons"])

    def test_patient_and_media_aliases_cannot_cross_partitions(self):
        self.splits.append({**self.splits[0], "source_video_id": "other", "split": "test"})
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            self.run_inventory()

    def test_dataset_free_inventory_to_assembly_preserves_label_provenance(self):
        report = self.run_inventory()
        result = assemble_facts(report["sources_path"], self.root / "facts")
        self.assertEqual(result["selected_fact_count"], 1)
        record = read_jsonl(result["cases_path"])[0]
        self.assertEqual(record["timestamps"], [1, 2])
        self.assertEqual(record["grounding"]["label_source"]["kind"], "trajectory_gold")
        self.assertIn("correct_answer", record["grounding"]["label_source"]["reference"])
        self.assertNotIn("WRONG closure", json.dumps(record))

    def test_exported_gold_tampering_is_detected_against_original_trajectory(self):
        report = self.run_inventory()
        source = read_jsonl(report["sources_path"])[0]
        source["gold_option"] = "B"
        source["gold_answer_text"] = "Mesenteric Defect Closure"
        Path(report["sources_path"]).write_text(json.dumps(source) + "\n")
        result = assemble_facts(report["sources_path"], self.root / "facts")
        self.assertEqual(result["selected_fact_count"], 0)
        self.assertIn("saved_standard_answer_changed_since_inventory", result["pending_reasons"])

    def test_cli_does_not_require_dataset_argument(self):
        self.run_inventory()
        with redirect_stdout(io.StringIO()):
            result = main(["inventory-trajectories", "--trajectories", str(self.traces),
                           "--splits", str(self.root / "splits.jsonl"), "--output", str(self.root / "cli")])
        self.assertEqual(result, 0)
        self.assertEqual(read_jsonl(self.root / "cli/sources.jsonl")[0]["gold_option"], "A")


if __name__ == "__main__":
    unittest.main()
