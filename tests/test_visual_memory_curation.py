import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from videospy.visual_memory.cli import _embedding_config, main
from videospy.visual_memory.curation import curate_cases, read_jsonl, validate_manifest


class VisualMemoryCurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".curation-test-", dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        self.frames = self.root / "source_frames"
        self.frames.mkdir()
        for index in range(12):
            Image.new("RGB", (10, 8), (index, index * 2, 0)).save(self.frames / f"{index}.png")

    def tearDown(self):
        self.temporary.cleanup()

    def selection(self, *, name="clip_skim", arguments=None, image_count=4, tool_content="Facts: visible objects."):
        if arguments is None:
            arguments = {"query": "Observe the action", "start_time": 0.7, "end_time": 4.7}
        call = {"id": "call_1", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}
        model_call = {"input": {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<omitted>"}}
            for _ in range(image_count)]}]}, "output": {"content": "Observation"}, "metrics": {"error": None}}
        trajectory = {"question": "Which action occurs during the specified interval?",
                      "final_answer": "B", "messages": [
                          {"role": "assistant", "tool_calls": [call]},
                          {"role": "tool", "tool_call_id": "call_1", "name": name,
                           "content": tool_content, "model_call": model_call}]}
        trajectory_path = self.root / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
        return {"trajectory_path": "trajectory.json", "source_video_id": "case_video_1",
                "task_type": "action", "split": "train", "answer_text": "Cutting tissue",
                "tool_call_id": "call_1", "frame_root": "source_frames", "fps": 2,
                "frame_paths": [f"{index}.png" for index in range(12)]}

    def run_curate(self, selection, name="export"):
        path = self.root / "selections.jsonl"
        path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        manifest = curate_cases(path, self.root / name)
        return manifest, read_jsonl(manifest)

    def test_clip_sampling_floor_linspace_and_question_label_not_broadcast(self):
        manifest, cases = self.run_curate(self.selection())
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["timestamps"], [0.5, 1.5, 3.0, 4.5])
        self.assertEqual(case["provenance"]["sampling"]["frame_indices"], [1, 3, 6, 9])
        self.assertEqual(case["answer_text"], "Cutting tissue")
        self.assertEqual(case["previous_prediction"], "B")
        self.assertFalse(case["verified"])
        self.assertEqual(case["verification_note"], "")
        self.assertNotIn("frame_labels", case)
        self.assertEqual(case["evidence_scope"], "clip")
        for image_path, index in zip(case["image_paths"], [1, 3, 6, 9]):
            with Image.open(manifest.parent / image_path) as image:
                self.assertEqual(image.getpixel((0, 0)), (index, index * 2, 0))
        self.assertEqual(len(validate_manifest(manifest)), 1)
        with self.assertRaisesRegex(ValueError, "pending"):
            validate_manifest(manifest, require_verified=True)

    def test_frame_nearest_dedup_and_last_frame_rounding_tolerance(self):
        selection = self.selection(name="frame_inspect", image_count=3,
                                   arguments={"query": "Inspect", "timestamps": [0.26, 0.3, 1.26, 6.0, 99] },
                                   tool_content="Requested 5 timestamps; only the first 4 in input order were used.")
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["timestamps"], [0.5, 1.5, 5.5])
        self.assertEqual(cases[0]["evidence_scope"], "clip")
        self.assertEqual(cases[0]["provenance"]["sampling"]["max_frames"], 4)

    def test_single_frame_scope_and_message_index(self):
        selection = self.selection(name="frame_inspect", image_count=1,
                                   arguments={"query": "Inspect", "timestamps": [1.1]})
        selection.pop("tool_call_id")
        selection["tool_message_index"] = 1
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["evidence_scope"], "frame")
        self.assertEqual(cases[0]["timestamps"], [1.0])

    def test_clip_end_is_clipped_to_video_duration(self):
        selection = self.selection(arguments={"query": "Inspect", "start_time": 5, "end_time": 20}, image_count=2)
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["timestamps"], [5.0, 5.5])
        self.assertEqual(cases[0]["provenance"]["sampling"]["effective_interval"], [5.0, 6.0])

    def test_provenance_hashes_and_source_paths_are_reproducible(self):
        manifest, cases = self.run_curate(self.selection())
        provenance = cases[0]["provenance"]
        self.assertEqual(provenance["trajectory_sha256"], hashlib.sha256((self.root / "trajectory.json").read_bytes()).hexdigest())
        self.assertEqual(len(provenance["tool_call_sha256"]), 64)
        self.assertEqual(Path(provenance["source_frames"][0]["path"]), self.frames / "1.png")
        self.assertEqual(provenance["source_frames"][0]["sha256"], hashlib.sha256((self.frames / "1.png").read_bytes()).hexdigest())
        self.assertTrue((manifest.parent / "curation.json").is_file())

    def test_legacy_source_prefix_requires_explicit_remapping(self):
        selection = self.selection()
        selection["frame_paths"] = [f"/root/data/{index}.png" for index in range(12)]
        selection["frame_path_prefix"] = "/root/data/"
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["timestamps"][0], 0.5)

    def test_failed_duplicate_and_overview_calls_rejected_before_output(self):
        for number, (name, content) in enumerate([
            ("clip_skim", "Tool `clip_skim` execution failed.\nException: bad"),
            ("clip_skim", "Duplicate `clip_skim` inspection skipped."),
            ("overview", "Some global observations")]):
            selection = self.selection(name=name, tool_content=content)
            path = self.root / "selections.jsonl"
            path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaises(ValueError):
                curate_cases(path, self.root / f"bad_{number}")
            self.assertFalse((self.root / f"bad_{number}").exists())

    def test_explicit_semantic_gt_and_allowed_split_required(self):
        for field, value in [("answer_text", "B"), ("answer_text", "Answer: D"), ("split", "test"),
                             ("source_video_id", ""), ("task_type", "")]:
            selection = self.selection()
            selection[field] = value
            with self.assertRaises(ValueError):
                self.run_curate(selection)

    def test_selection_needs_exactly_one_tool_selector(self):
        selection = self.selection()
        selection["tool_message_index"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_curate(selection)

    def test_recorded_image_count_disagreement_is_not_silently_accepted(self):
        selection = self.selection()
        selection["num_frames"] = 3
        with self.assertRaisesRegex(ValueError, "recorded tool saw"):
            self.run_curate(selection)
        self.assertTrue((self.root / "export" / "curation_failed.json").is_file())
        self.assertFalse((self.root / "export" / "cases.jsonl").exists())

    def test_existing_output_is_never_overwritten(self):
        selection = self.selection()
        manifest, _ = self.run_curate(selection)
        before = manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_curate(selection)
        self.assertEqual(manifest.read_bytes(), before)

    def test_legacy_record_without_model_inputs_requires_historical_clip_count(self):
        selection = self.selection()
        path = self.root / "trajectory.json"
        trajectory = json.loads(path.read_text())
        trajectory["messages"][1].pop("model_call")
        path.write_text(json.dumps(trajectory), encoding="utf-8")
        selection["num_frames"] = 4
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["provenance"]["sampling"]["recorded_image_count"], None)

    def test_validate_review_note_and_cli(self):
        manifest, cases = self.run_curate(self.selection())
        with patch("builtins.print"):
            self.assertEqual(main(["validate", "--cases", str(manifest)]), 0)
        cases[0]["verified"] = True
        manifest.write_text(json.dumps(cases[0]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification_note"):
            validate_manifest(manifest)
        cases[0]["verification_note"] = "Clinician checked the full ordered clip and semantic answer."
        manifest.write_text(json.dumps(cases[0]), encoding="utf-8")
        self.assertEqual(len(validate_manifest(manifest, require_verified=True)), 1)

    def test_embedding_config_accepts_project_or_standalone_yaml(self):
        for number, payload in enumerate(["backend: fake\n", "embedding:\n  backend: fake\n",
                                          "tools:\n  visual_memory:\n    embedding:\n      backend: fake\n"]):
            path = self.root / f"embedding_{number}.yml"
            path.write_text(payload, encoding="utf-8")
            self.assertEqual(_embedding_config(str(path)), {"backend": "fake"})

    def test_visual_memory_reference_images_do_not_change_source_sampling(self):
        selection = self.selection()
        path = self.root / "trajectory.json"
        trajectory = json.loads(path.read_text())
        model_call = trajectory["messages"][1]["model_call"]
        model_call["input"]["messages"][0]["content"] = [
            {"type": "text", "text": "HISTORICAL REFERENCE CASES"},
            {"type": "image_url", "image_url": {"url": "reference-example"}},
            {"type": "text", "text": "END HISTORICAL REFERENCES. Current video follows."},
        ] + model_call["input"]["messages"][0]["content"]
        trajectory["messages"][1]["visual_memory"] = [{"status": "injected", "added_reference_images": 1}]
        path.write_text(json.dumps(trajectory), encoding="utf-8")
        _, cases = self.run_curate(selection)
        self.assertEqual(cases[0]["provenance"]["sampling"]["frame_indices"], [1, 3, 6, 9])

    def test_fake_reader_preserves_frame_inspect_end_rounding_logic(self):
        selection = self.selection(name="frame_inspect", image_count=1,
                                   arguments={"query": "Inspect", "timestamps": ["0:01.2"]})
        path = self.root / "selections.jsonl"
        path.write_text(json.dumps(selection), encoding="utf-8")

        class FakeReader:
            duration = 1.16

            def __len__(self):
                return 2

            def get_avg_fps(self):
                return 2

            def get_batch(self, indices):
                class Batch:
                    def asnumpy(self):
                        return np.zeros((len(indices), 8, 10, 3), dtype=np.uint8)
                return Batch()

        manifest = curate_cases(path, self.root / "export", reader_factory=lambda *_: (FakeReader(), {"kind": "fake"}))
        self.assertEqual(read_jsonl(manifest)[0]["timestamps"], [0.5])

    def test_cli_build_uses_reviewed_manifest_and_config(self):
        manifest, cases = self.run_curate(self.selection())
        cases[0].update(verified=True, verification_note="Reviewed ordered evidence against semantic answer.")
        manifest.write_text(json.dumps(cases[0]), encoding="utf-8")
        config = self.root / "embedding.yml"
        config.write_text("backend: fake\n", encoding="utf-8")
        with patch("videospy.visual_memory.store.build_index") as build, patch("builtins.print"):
            self.assertEqual(main(["build", "--cases", str(manifest), "--output", str(self.root / "index"),
                                   "--embedding-config", str(config)]), 0)
            build.assert_called_once_with(manifest.resolve(), (self.root / "index").resolve(), {"backend": "fake"})

    def test_explicit_gold_option_resolves_mapping_or_list(self):
        for number, options in enumerate([{"A": "Grasping", "B": "Cutting tissue"}, ["Grasping", "Cutting tissue"]]):
            selection = self.selection()
            selection.pop("answer_text")
            selection.update(gold_option="B", options=options)
            _, cases = self.run_curate(selection, name=f"export_{number}")
            self.assertEqual(cases[0]["answer_text"], "Cutting tissue")
            self.assertEqual(cases[0]["provenance"]["answer_source"]["gold_option"], "B")

    def test_explicit_gt_cannot_disagree_with_option_mapping(self):
        selection = self.selection()
        selection.update(gold_option="A", options={"A": "Grasping", "B": "Cutting tissue"})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.run_curate(selection)

    def test_probe_prints_only_safe_identity_dimension_and_norm(self):
        config = self.root / "embedding.yml"
        config.write_text("api_base: http://127.0.0.1:8080/v1\nmodel: test-embedding\n", encoding="utf-8")
        with patch("videospy.visual_memory.embedding.EmbeddingClient") as client, patch("builtins.print") as output:
            client.return_value.identity = {"model": "test-embedding"}
            client.return_value.encode.return_value = np.array([[3.0, 4.0]], dtype=np.float32)
            self.assertEqual(main(["probe", "--embedding-config", str(config),
                                   "--image", str(self.frames / "0.png")]), 0)
            payload = json.loads(output.call_args.args[0])
            self.assertEqual(payload, {"identity": {"model": "test-embedding"}, "dimension": 2, "norm": 5.0})
            client.return_value.encode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
