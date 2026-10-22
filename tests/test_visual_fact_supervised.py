"""Source label fidelity, temporal alignment, isolation and single-call inference."""

import copy
import gc
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from tests import test_visual_memory_curation as curation_tests
from tests.test_visual_memory_integration import PixelEncoder, ColorReader, joined_text
from tests.test_videospy import FakeModelClient, clip_skim_config
from videospy.tools.clip_skim import execute_clip_skim
from videospy.visual_memory.cli import main
from videospy.visual_memory.curation import read_jsonl
from videospy.visual_memory.facts import validate_fact_record
from videospy.visual_memory.integration import VisualMemorySession, memory_config
from videospy.visual_memory.store import VisualMemoryStore, build_index, validate_case
from videospy.visual_memory.supervised import assemble_facts


def annotation(**changes):
    value = {"annotation_id": "event-1", "fact_type": "action",
             "label_source": {"kind": "dataset_gold", "reference": "dataset-v1/qa/1/answer"},
             "alignment": {"kind": "interval", "start": 1, "end": 3.2,
                           "source": "dataset_annotation", "reference": "dataset-v1/qa/1/event",
                           "scope": "local_clip"}}
    value.update(changes)
    return value


class SupervisedFactsTests(unittest.TestCase):
    setUp = curation_tests.VisualMemoryCurationTests.setUp
    tearDown = curation_tests.VisualMemoryCurationTests.tearDown
    selection = curation_tests.VisualMemoryCurationTests.selection

    def source(self, **changes):
        row = self.selection(tool_content="A confidently WRONG visual explanation.")
        row.pop("tool_call_id")
        row.update(source_sample_id="1", gold_option="A", options=["Cleaning and Coagulation", "WRONG phase"],
                   supervision=[annotation()], procedure="procedure-a")
        row.update(changes)
        return row

    def assemble(self, sources=None, settings=None, annotations=None):
        path = self.root / "sources.jsonl"
        if sources is not None:
            path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
        kwargs = {}
        if annotations is not None:
            apath = self.root / "annotations.jsonl"
            apath.write_text("".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8")
            kwargs["annotations_path"] = apath
        with patch("videospy.visual_memory.vlm.FactMiningVLM", side_effect=AssertionError("no VLM")), patch(
            "videospy.model.ModelClient.chat", side_effect=AssertionError("no chat")):
            return assemble_facts(path, self.root / "assembled", settings, **kwargs)

    def records(self, report):
        path = Path(report["cases_path"])
        return read_jsonl(path) if path.stat().st_size else []

    def test_gold_replaces_wrong_prediction_without_reading_pixels_with_vlm(self):
        source = self.source()
        report = self.assemble([source])
        record = self.records(report)[0]
        self.assertEqual(record["answer_text"], "Cleaning and Coagulation")
        self.assertEqual(record["timestamps"], [1.5, 3.0])
        self.assertEqual(record["verification"]["status"], "source_grounded")
        self.assertFalse(record["verified"])
        self.assertNotIn("WRONG visual", json.dumps(record))
        self.assertEqual(report["vlm_calls"], 0)
        self.assertEqual(record["fact"]["conditions"], [])
        self.assertNotIn("temporal_change", record["verification"])

    def test_correct_and_incorrect_outcomes_both_use_only_dataset_label(self):
        source = self.source()
        path = self.root / "trajectory.json"
        raw = json.loads(path.read_text())
        raw["correct"] = False
        path.write_text(json.dumps(raw))
        first = self.assemble([source])
        self.assertEqual(first["trajectory_outcomes"], {"incorrect": 1})
        raw["correct"] = True
        path.write_text(json.dumps(raw))
        sources_path = self.root / "sources.jsonl"
        second = assemble_facts(sources_path, self.root / "correct")
        self.assertEqual(second["trajectory_outcomes"], {"correct": 1})
        self.assertEqual(self.records(first)[0]["answer_text"], self.records(second)[0]["answer_text"])

    def test_no_alignment_never_broadcasts_gold_to_whole_observation(self):
        report = self.assemble([self.source(supervision=[])])
        self.assertEqual(self.records(report), [])
        pending = read_jsonl(report["pending_path"])[0]
        self.assertEqual(pending["dataset_label_candidate"], "Cleaning and Coagulation")
        self.assertEqual(pending["reason"], "missing_explicit_label_alignment")
        self.assertEqual(pending["observed_timestamps"], [0.5, 1.5, 3.0, 4.5])

    def test_missing_label_is_pending_even_if_model_final_answer_is_correct(self):
        source = self.source()
        source.pop("gold_option")
        report = self.assemble([source])
        self.assertFalse(self.records(report))
        self.assertIn("requires gold_option", next(iter(report["pending_reasons"])))

    def test_verified_description_is_copied_exactly_and_needs_verifier(self):
        item = annotation(fact_type="instrument_identity",
                          label_source={"kind": "verified_description", "text": "A needle holder.",
                                        "reference": "curated-v2/row-4", "verified_by": "annotator-1"})
        report = self.assemble([self.source(supervision=[item])])
        self.assertEqual(self.records(report)[0]["answer_text"], "A needle holder.")
        source_path = self.root / "sources.jsonl"
        source = read_jsonl(source_path)[0]
        del source["supervision"][0]["label_source"]["verified_by"]
        source_path.write_text(json.dumps(source) + "\n")
        second = assemble_facts(source_path, self.root / "missing-verifier")
        self.assertFalse(self.records(second))

    def test_annotation_type_not_guessed_from_action_recognition_task(self):
        item = annotation(fact_type="phase")
        item["alignment"]["scope"] = "phase_interval"
        report = self.assemble([self.source(task_type="action recognition", supervision=[item])])
        self.assertEqual(self.records(report)[0]["fact"]["type"], "phase")

    def test_phase_interval_cannot_be_relabelled_as_anatomy(self):
        item = annotation(fact_type="anatomy")
        item["alignment"]["scope"] = "phase_interval"
        report = self.assemble([self.source(supervision=[item])])
        self.assertFalse(self.records(report))

    def test_unsupported_global_or_diagnostic_types_never_banked(self):
        items = [annotation(annotation_id=kind, fact_type=kind) for kind in ("histology", "global_count", "onset_time")]
        report = self.assemble([self.source(supervision=items)])
        self.assertEqual(report["pending_count"], 3)
        self.assertFalse(self.records(report))

    def test_action_requires_multiple_observed_frames(self):
        item = annotation()
        item["alignment"].update(start=1, end=2)
        report = self.assemble([self.source(supervision=[item])])
        self.assertIn("insufficient_observed_frames_for_clip_label", report["pending_reasons"])

    def test_explicit_keyframes_cannot_be_replaced_with_unobserved_nearby_frames(self):
        item = annotation()
        item["alignment"] = {"kind": "timestamps", "timestamps": [1.5, 2.5], "scope": "local_clip",
                             "source": "human_verified", "verified_by": "expert", "reference": "annotation-1"}
        report = self.assemble([self.source(supervision=[item])])
        self.assertIn("annotated_keyframe_not_observed", report["pending_reasons"])

    def test_intersection_uses_only_original_samples_not_extra_frames(self):
        source = self.source()
        source["supervision"][0]["alignment"].update(start=0.6, end=3.1)
        record = self.records(self.assemble([source]))[0]
        self.assertEqual(record["timestamps"], [1.5, 3.0])
        self.assertEqual([frame["index"] for frame in record["provenance"]["selected_frames"]], [3, 6])

    def test_multiple_calls_union_keeps_all_explicit_observed_keyframes(self):
        source = self.source()
        raw = json.loads((self.root / "trajectory.json").read_text())
        extra = copy.deepcopy(raw["messages"])
        extra[0]["tool_calls"][0]["id"] = "call_2"
        extra[1]["tool_call_id"] = "call_2"
        extra[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"start_time": 4, "end_time": 5.5})
        raw["messages"].extend(extra)
        (self.root / "trajectory.json").write_text(json.dumps(raw))
        source["supervision"][0]["alignment"] = {"kind": "timestamps", "timestamps": [1.5, 5.5],
            "scope": "local_clip", "source": "dataset_annotation", "reference": "event-1"}
        record = self.records(self.assemble([source]))[0]
        self.assertEqual(record["timestamps"], [1.5, 5.5])

    def test_failed_and_duplicate_calls_cannot_supply_evidence(self):
        source = self.source()
        raw = json.loads((self.root / "trajectory.json").read_text())
        raw["messages"][1]["content"] = "Duplicate tool call skipped."
        (self.root / "trajectory.json").write_text(json.dumps(raw))
        report = self.assemble([source])
        self.assertFalse(self.records(report))

    def test_image_budget_never_silently_drops_support_frames(self):
        report = self.assemble([self.source()], {"max_images_per_fact": 1})
        self.assertFalse(self.records(report))
        self.assertTrue(any("exceeds_image_budget" in reason for reason in report["pending_reasons"]))

    def test_existing_pairs_can_be_imported_without_a_trajectory(self):
        row = {"source_video_id": "existing-video", "source_sample_id": "pair-1", "split": "train",
               "task_type": "action recognition", "question": "Which action?",
               "image_paths": ["source_frames/3.png", "source_frames/6.png"], "timestamps": [1.5, 3],
               "gold_option": "A", "options": ["Cleaning and Coagulation", "WRONG"], "supervision": [annotation()]}
        record = self.records(self.assemble([row]))[0]
        self.assertEqual(record["timestamps"], [1.5, 3])
        self.assertIsNone(record["provenance"]["trajectory_path"])

    def test_sidecar_join_requires_both_video_and_qa_and_reports_unmatched(self):
        source = self.source(supervision=[])
        item = {**annotation(), "source_video_id": source["source_video_id"], "source_sample_id": "1"}
        wrong = {**item, "source_video_id": "different-video"}
        report = self.assemble([source], annotations=[item, wrong])
        self.assertEqual(len(self.records(report)), 1)
        self.assertEqual(report["pending_reasons"]["annotation_has_no_matching_source"], 1)

    def test_test_sources_and_cross_partition_aliases_are_rejected(self):
        report = self.assemble([self.source(split="test")])
        self.assertFalse(self.records(report))
        self.assertIn("held_out_or_missing_partition", report["pending_reasons"])
        first = self.source()
        second = {**first, "source_video_id": "different-id", "source_sample_id": "2", "split": "test"}
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            self.assemble([first, second])

    def test_mismatching_trajectory_video_identity_is_rejected(self):
        source = self.source()
        raw = json.loads((self.root / "trajectory.json").read_text())
        raw["source_video_id"] = "wrong-source"
        (self.root / "trajectory.json").write_text(json.dumps(raw))
        self.assertFalse(self.records(self.assemble([source])))

    def test_resume_preserves_records_and_refuses_annotation_changes(self):
        first = self.assemble([self.source()])
        original = Path(first["cases_path"]).read_bytes()
        self.assemble(settings={"resume": True})
        self.assertEqual(Path(first["cases_path"]).read_bytes(), original)
        rows = read_jsonl(self.root / "sources.jsonl")
        rows[0]["supervision"][0]["alignment"]["end"] = 4.6
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.assemble(rows, {"resume": True})
        self.assertEqual(Path(first["cases_path"]).read_bytes(), original)

    def test_resume_detects_modified_exported_images(self):
        from PIL import Image
        first = self.assemble([self.source()])
        path = Path(first["cases_path"]).parent / self.records(first)[0]["image_paths"][0]
        Image.new("RGB", (10, 8), (255, 255, 255)).save(path)
        with self.assertRaisesRegex(ValueError, "checkpoint_evidence_changed"):
            self.assemble(settings={"resume": True})

    def test_generated_or_edited_label_cannot_pass_grounding_validation(self):
        original = self.records(self.assemble([self.source()]))[0]
        for change in ("label", "conditions", "times", "human"):
            record = copy.deepcopy(original)
            if change == "label":
                record["answer_text"] = record["fact"]["text"] = "Hallucinated phase"
            elif change == "conditions":
                record["fact"]["conditions"] = ["Invented instrument detail"]
            elif change == "times":
                record["timestamps"] = [0, 1]
            else:
                record["verified"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_fact_record(record)

    def test_build_and_search_have_separate_explicit_admission(self):
        report = self.assemble([self.source()])
        case = self.records(report)[0]
        with self.assertRaisesRegex(ValueError, "allow_source_grounded"):
            validate_case(case)
        validate_case(case, allow_source_grounded=True)
        build_index(report["cases_path"], self.root / "index", {}, embedder=PixelEncoder(), allow_source_grounded=True)
        store = VisualMemoryStore(self.root / "index")
        context = {"source_video_id": "new-video", "task_type": "instrument_onset", "procedure": "procedure-a"}
        result = store.search([1, 1, 1], context=context, match_mode="fact")
        self.assertEqual(result["matches"], [])
        self.assertIn("source_grounded_not_enabled", result["exclusions"][0]["reasons"])
        result = store.search([1, 1, 1], context=context, match_mode="fact", allow_source_grounded=True)
        self.assertEqual(len(result["matches"]), 1)
        context["source_video_id"] = case["source_video_id"]
        self.assertEqual(store.search([1, 1, 1], context=context, match_mode="fact", allow_source_grounded=True)["matches"], [])
        del store
        gc.collect()

    def test_new_online_preset_one_vlm_call_no_review_or_old_distractors(self):
        report = self.assemble([self.source()])
        build_index(report["cases_path"], self.root / "index", {}, embedder=PixelEncoder(), allow_source_grounded=True)
        path = Path(__file__).resolve().parents[1] / "config/videospy/visual_facts_supervised.yml"
        settings = yaml.safe_load(path.read_text())["tools"]["visual_memory"]
        settings["database_path"] = str(self.root / "index")
        session = VisualMemorySession(settings)
        client = FakeModelClient([{"content": "Independent observation with references."}])
        parameters = {"vr": ColorReader(), "subtitles": [], "start_time": 0, "end_time": 4,
                      "query": "Describe the action", "original_question": "What happens?",
                      "_visual_memory_session": session,
                      "_visual_memory_context": {"source_video_id": "current", "task_type": "action recognition",
                                                 "procedure": "procedure-a"}}
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.clip_skim.ModelClient", return_value=client):
            result = execute_clip_skim(clip_skim_config(num_frames=2), parameters)
        self.assertFalse(hasattr(session, "review"))
        self.assertEqual(len(client.calls), 1)
        text = joined_text(client.calls[0]["messages"][0]["content"])
        self.assertIn("Cleaning and Coagulation", text)
        self.assertNotIn("WRONG phase", text)
        self.assertNotIn("WRONG visual", text)
        self.assertNotIn("Additional visual fact review", result)
        self.assertEqual(session.audit[0]["matches"][0]["grounding"]["label_source"]["kind"], "dataset_gold")
        self.assertFalse(memory_config(settings)["allow_model_reviewed"])
        del session
        gc.collect()

    def test_cli_assemble_validate_and_legacy_mine_requires_opt_in(self):
        source = self.source()
        path = self.root / "sources.jsonl"
        path.write_text(json.dumps(source) + "\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["assemble", "--sources", str(path), "--output", str(self.root / "assembled")]), 0)
            self.assertEqual(main(["validate", "--cases", str(self.root / "assembled/cases.jsonl"), "--allow-source-grounded"]), 0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["mine", "--sources", str(path), "--config", "unused.yml", "--output", str(self.root / "legacy")])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "legacy").exists())

    def test_dedup_and_type_balancing_preserve_rare_labels(self):
        from videospy.visual_memory.supervised import _balance

        record = self.records(self.assemble([self.source()]))[0]
        rows = []
        for i in range(5):
            value = copy.deepcopy(record)
            value.update(case_id=str(i), answer_text=f"Action {i}")
            rows.append(value)
        duplicate = {**rows[0], "case_id": "duplicate"}
        rare = copy.deepcopy(record)
        rare.update(case_id="rare", source_video_id="different-video")
        rare["fact"]["type"] = "anatomy"
        selected, duplicates, _ = _balance(rows + [duplicate, rare],
                                           {"max_facts_per_video_type": 20, "max_total_facts": 2})
        self.assertEqual({r["fact"]["type"] for r in selected}, {"anatomy", "action"})
        self.assertEqual(len(duplicates), 1)

    def test_new_runtime_excludes_legacy_model_reviewed_records_in_mixed_index(self):
        from tests.test_visual_facts import fact_record

        report = self.assemble([self.source()])
        case = self.records(report)[0]
        old = fact_record("old-generated", image_paths=[case["image_paths"][0]], timestamps=[1.5])
        manifest = Path(report["cases_path"])
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(old) + "\n")
        build_index(manifest, self.root / "mixed", {}, embedder=PixelEncoder(),
                    allow_source_grounded=True, allow_model_reviewed=True)
        store = VisualMemoryStore(self.root / "mixed")
        result = store.search([1, 1, 1], context={"source_video_id": "current", "task_type": "action recognition"},
                              match_mode="fact", allow_source_grounded=True)
        self.assertEqual([m["case_id"] for m in result["matches"]], [case["case_id"]])
        self.assertTrue(any(e["case_id"] == "old-generated" and "model_reviewed_not_enabled" in e["reasons"]
                            for e in result["exclusions"]))
        del store
        gc.collect()

    def test_cli_build_source_grounded_records(self):
        report = self.assemble([self.source()])
        config = self.root / "embedding.yml"
        config.write_text("embedding:\n  model: pixel-mean\n", encoding="utf-8")
        with patch("videospy.visual_memory.store.EmbeddingClient", PixelEncoder), redirect_stdout(io.StringIO()):
            result = main(["build", "--cases", report["cases_path"], "--output", str(self.root / "cli-index"),
                           "--embedding-config", str(config), "--allow-source-grounded"])
        self.assertEqual(result, 0)
        policy = json.loads((self.root / "cli-index/manifest.json").read_text())["admission_policy"]
        self.assertEqual(policy, {"allow_model_reviewed": False, "allow_source_grounded": True})

    def test_corrupt_video_decode_is_reported_without_aborting_other_sources(self):
        source = self.source()
        path = self.root / "sources.jsonl"
        path.write_text(json.dumps(source) + "\n")
        def broken_reader(*args):
            raise RuntimeError("decoder failed")
        report = assemble_facts(path, self.root / "assembled", reader_factory=broken_reader)
        self.assertFalse(self.records(report))
        self.assertIn("cannot_reconstruct_observed_frames", report["pending_reasons"])

    def test_nonfinite_annotations_fail_before_any_output(self):
        source = self.source()
        source["supervision"][0]["alignment"]["start"] = float("nan")
        with self.assertRaises(ValueError):
            self.assemble([source])
        self.assertFalse((self.root / "assembled").exists())


if __name__ == "__main__":
    unittest.main()
