"""Text curation must not invent labels/evidence or upgrade inferred supervision."""
import copy
import gc
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests import test_visual_trajectory_inventory as inventory_tests
from tests.test_visual_memory_integration import PixelEncoder, ColorReader, joined_text
from tests.test_videospy import FakeModelClient, clip_skim_config
from videospy.tools.clip_skim import execute_clip_skim
from videospy.visual_memory.cli import main
from videospy.visual_memory.curation import read_jsonl
from videospy.visual_memory.facts import validate_fact_record
from videospy.visual_memory.integration import VisualMemorySession, memory_config
from videospy.visual_memory.store import VisualMemoryStore, build_index, validate_case
from videospy.visual_memory.supervised import assemble_facts
from videospy.visual_memory.trajectory_alignment import (PROMPT, TrajectoryTextModel, align_trajectories,
                                                       candidate_annotation, validate_candidate)
from videospy.visual_memory.vlm import VLMError


def suggestion(packet):
    return {"label_text": packet["gold_answer_text"], "fact_type": "phase", "scope": "phase_interval",
            "frame_ids": [1, 2], "tool_message_indices": [1], "question_quote": packet["question"],
            "call_citations": [{"tool_message_index": 1, "field": "arguments",
                                "quote": packet["observations"][0]["arguments"]}],
            "link_basis": "question_time", "rationale": "Question specifies 1–2 seconds; the selected call observed these times."}


class FakeText:
    identity = {"model_name": "fake-text", "temperature": 0}

    def __init__(self, payload=None):
        self.payload = payload
        self.packets = []

    def generate_json(self, prompt):
        packet = json.loads(prompt[len(PROMPT):])
        self.packets.append(packet)
        payload = self.payload(packet) if self.payload else {
            "decision": "usable", "reason": "Question time aligns with the observed call.",
            "candidates": [suggestion(packet)]}
        return {"payload": payload, "metrics": {"submitted_images": []}}


class TrajectoryAlignmentTests(unittest.TestCase):
    setUp = inventory_tests.TrajectoryInventoryTests.setUp
    run_inventory = inventory_tests.TrajectoryInventoryTests.run_inventory

    def inventory(self):
        self.row.pop("visual_fact_annotations", None)
        self.row["question"] = "Which phase is shown from 1 to 2 seconds?\nA. Cleaning and Coagulation\nB. Mesenteric Defect Closure"
        return self.run_inventory()

    def align(self, fake=None, config=None, sources=None, output="aligned"):
        if sources is None:
            sources = self.inventory()["sources_path"]
        fake = fake or FakeText()
        report = align_trajectories(sources, self.root / output, config or {}, client_factory=lambda _: fake)
        return report, fake

    def assembled(self):
        report, fake = self.align()
        assembled = assemble_facts(self.root / "inventory/sources.jsonl", self.root / "assembled",
                                   annotations_path=report["annotations_path"])
        self.assertEqual(assembled["selected_fact_count"], 1, assembled)
        return read_jsonl(assembled["cases_path"])[0], assembled, fake

    def test_wrong_answer_trace_can_supply_saved_label_with_cited_localization(self):
        record, report, fake = self.assembled()
        self.assertEqual(record["answer_text"], "Cleaning and Coagulation")
        self.assertEqual(record["timestamps"], [1, 2])
        self.assertEqual(record["verification"]["status"], "trajectory_aligned")
        self.assertFalse(record["verified"])
        self.assertEqual(report["alignment_by_status"], {"trajectory_aligned": 1})
        packet = fake.packets[0]
        self.assertNotIn("final_answer", packet)
        self.assertNotIn("correct", packet)
        self.assertNotIn("Mesenteric Defect Closure", packet["question"])
        self.assertTrue(packet["observations"][0]["observation_is_fallible_model_text"])

    def test_correct_outcome_is_not_admission_rule(self):
        self.row["correct"] = True
        fake = FakeText(lambda p: {"decision": "uncertain", "reason": "No reliable association.", "candidates": []})
        report, _ = self.align(fake)
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["decisions"], {"uncertain": 1})

    def test_generated_label_invented_frame_citation_and_scope_are_rejected(self):
        report, fake = self.align()
        packet = fake.packets[0]
        mutations = [
            {"label_text": "Confident invented medical label"}, {"frame_ids": [3]},
            {"tool_message_indices": [22]}, {"question_quote": "There is no such question"},
            {"link_basis": "final_answer_correct"}, {"frame_ids": [2, 1]},
            {"frame_ids": [1]}, {"call_citations": []}, {"fact_type": "anatomy"},
            {"call_citations": [{"tool_message_index": 1, "field": "observation", "quote": "invented observation"}]},
            {"added_attribute": "hallucination"},
        ]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_candidate({**suggestion(packet), **change}, packet)

    def test_invalid_model_output_goes_pending_and_never_becomes_annotation(self):
        fake = FakeText(lambda p: {"decision": "usable", "reason": "A guess", "candidates": [
            {**suggestion(p), "label_text": "invented"}]})
        report, _ = self.align(fake)
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["decisions"], {"invalid_candidate": 1})
        self.assertIn("generated labels forbidden", read_jsonl(report["pending_path"])[0]["reason"])

    def test_uncertain_cannot_smuggle_candidates_into_index(self):
        fake = FakeText(lambda p: {"decision": "uncertain", "reason": "Maybe", "candidates": [suggestion(p)]})
        report, _ = self.align(fake)
        self.assertEqual(report["candidate_count"], 0)

    def test_nonfinite_model_output_is_pending_without_aborting_batch(self):
        fake = FakeText(lambda p: {"decision": "usable", "reason": float("nan"), "candidates": [suggestion(p)]})
        report, _ = self.align(fake)
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["decisions"], {"invalid_candidate": 1})
        self.assertTrue(Path(report["pending_path"]).is_file())

    def test_missing_gold_is_rejected_before_model_call(self):
        self.row.pop("correct_answer")
        fake = FakeText()
        with self.assertRaisesRegex(ValueError, "no records"):
            self.align(fake)
        self.assertFalse(fake.packets)

    def test_test_sources_never_reach_model(self):
        inv = self.inventory()
        path = Path(inv["sources_path"])
        source = read_jsonl(path)[0]
        source["split"] = "test"
        path.write_text(json.dumps(source) + "\n")
        report, fake = self.align(sources=path)
        self.assertFalse(fake.packets)
        self.assertEqual(report["candidate_count"], 0)

    def test_partition_conflict_precedes_model_calls(self):
        inv = self.inventory()
        path = Path(inv["sources_path"])
        source = read_jsonl(path)[0]
        other = {**source, "source_sample_id": "2", "split": "test"}
        path.write_text(json.dumps(source) + "\n" + json.dumps(other) + "\n")
        fake = FakeText()
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            self.align(fake, sources=path)
        self.assertFalse(fake.packets)

    def test_failed_or_overview_only_calls_supply_no_candidate(self):
        self.row["messages"][1]["success"] = False
        report, fake = self.align()
        self.assertFalse(fake.packets)
        self.assertEqual(report["candidate_count"], 0)

    def test_context_limit_never_silently_truncates(self):
        report, fake = self.align(config={"max_packet_chars": 10})
        self.assertFalse(fake.packets)
        self.assertIn("no silent truncation", read_jsonl(report["pending_path"])[0]["reason"])

    def test_resume_reuses_completed_llm_decisions(self):
        first, fake = self.align()
        second, _ = self.align(fake, {"resume": True}, self.root / "inventory/sources.jsonl")
        self.assertEqual(first["candidate_count"], second["candidate_count"])
        self.assertEqual(second["new_text_model_calls"], 0)
        self.assertEqual(len(fake.packets), 1)

    def test_request_error_is_pending_and_retryable(self):
        class Broken(FakeText):
            def generate_json(self, prompt):
                raise VLMError("Sanitized failure")
        report, _ = self.align(Broken())
        self.assertEqual(report["decisions"], {"request_error": 1})
        retry, fake = self.align(config={"resume": True}, sources=self.root / "inventory/sources.jsonl")
        self.assertEqual(retry["candidate_count"], 1)
        self.assertEqual(len(fake.packets), 1)

    def test_modified_trajectory_refuses_resume_and_assembly(self):
        report, _ = self.align()
        self.row["messages"][1]["content"] = "changed observation"
        (self.traces / "1.json").write_text(json.dumps(self.row))
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.align(config={"resume": True}, sources=self.root / "inventory/sources.jsonl")
        result = assemble_facts(self.root / "inventory/sources.jsonl", self.root / "assembled",
                                annotations_path=report["annotations_path"])
        self.assertEqual(result["selected_fact_count"], 0)

    def test_forged_packet_is_checked_against_raw_trace_before_export(self):
        report, fake = self.align()
        packet = copy.deepcopy(fake.packets[0])
        packet["question"] = "A completely different question at 1 to 2 seconds"
        forged = candidate_annotation(suggestion(packet), packet, fake.identity)
        Path(report["annotations_path"]).write_text(json.dumps(forged) + "\n")
        result = assemble_facts(self.root / "inventory/sources.jsonl", self.root / "assembled",
                                annotations_path=report["annotations_path"])
        self.assertEqual(result["selected_fact_count"], 0)
        self.assertTrue(any("no longer matches" in r for r in result["pending_reasons"]))

    def test_inferred_facts_cannot_be_relabelled_as_source_or_human_verified(self):
        record, _, _ = self.assembled()
        for status in ("source_grounded", "human_verified"):
            forged = copy.deepcopy(record)
            forged["verification"]["status"] = status
            with self.assertRaises(ValueError):
                validate_fact_record(forged)
        with self.assertRaisesRegex(ValueError, "allow_trajectory_aligned"):
            validate_case(record, allow_source_grounded=True, allow_model_reviewed=True)
        validate_case(record, allow_trajectory_aligned=True)

    def test_build_load_and_retrieval_need_independent_gate(self):
        record, result, _ = self.assembled()
        encoder = PixelEncoder()
        with self.assertRaisesRegex(ValueError, "allow_trajectory_aligned"):
            build_index(result["cases_path"], self.root / "refused", {}, embedder=encoder)
        build_index(result["cases_path"], self.root / "index", {}, embedder=encoder, allow_trajectory_aligned=True)
        store = VisualMemoryStore(self.root / "index")
        context = {"source_video_id": "other", "task_type": "action recognition", "procedure": "surgery-1"}
        off = store.search(store.vectors[0], context=context, match_mode="fact")
        on = store.search(store.vectors[0], context=context, match_mode="fact", allow_trajectory_aligned=True)
        self.assertFalse(off["matches"])
        self.assertEqual(len(on["matches"]), 1)
        del store

    def test_text_transport_submits_zero_images(self):
        with patch("videospy.visual_memory.trajectory_alignment.FactMiningVLM") as adapter:
            adapter.return_value.identity = {"model_name": "text-only", "image_max_side": 2048}
            adapter.return_value.generate_json.return_value = {"payload": {}, "metrics": {}}
            client = TrajectoryTextModel({})
            client.generate_json("question and calls")
            adapter.return_value.generate_json.assert_called_once_with("question and calls", [], purpose="trajectory_alignment_text_only")

    def test_online_uses_one_observation_and_never_injects_fallible_old_text(self):
        _, result, _ = self.assembled()
        build_index(result["cases_path"], self.root / "index", {}, embedder=PixelEncoder(), allow_trajectory_aligned=True)
        session = VisualMemorySession({"enabled": True, "match_mode": "fact", "allow_trajectory_aligned": True,
                                       "database_path": str(self.root / "index")})
        client = FakeModelClient([{"content": "Current observation."}])
        parameters = {"vr": ColorReader(), "subtitles": [], "start_time": 0, "end_time": 4,
                      "query": "Which phase?", "original_question": "A different video question",
                      "_visual_memory_session": session,
                      "_visual_memory_context": {"source_video_id": "new-video", "task_type": "action recognition",
                                                 "procedure": "surgery-1"}}
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
                "videospy.tools.clip_skim.ModelClient", return_value=client):
            execute_clip_skim(clip_skim_config(num_frames=2), parameters)
        self.assertEqual(len(client.calls), 1)
        text = joined_text(client.calls[0]["messages"][0]["content"])
        self.assertIn("Cleaning and Coagulation", text)
        self.assertIn("trajectory_aligned", text)
        self.assertIn("not independently visually verified", text)
        self.assertNotIn("WRONG closure", text)
        self.assertNotIn("question_quote", text)
        self.assertNotIn("Question specifies", text)
        del session
        parameters.clear()
        gc.collect()

    def test_cli_alignment_to_assembly_and_build(self):
        inv = self.inventory()
        cfg = self.root / "align.yml"
        cfg.write_text("trajectory_alignment:\n  llm: {}\n")
        fake = FakeText()
        with patch("videospy.visual_memory.trajectory_alignment.TrajectoryTextModel", return_value=fake), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["align-trajectories", "--sources", inv["sources_path"], "--config", str(cfg),
                                   "--output", str(self.root / "aligned")]), 0)
            self.assertEqual(main(["assemble", "--sources", inv["sources_path"], "--annotations",
                                   str(self.root / "aligned/annotations.jsonl"), "--output", str(self.root / "assembled")]), 0)
            self.assertEqual(main(["validate", "--cases", str(self.root / "assembled/cases.jsonl"),
                                   "--allow-trajectory-aligned"]), 0)
        embedding = self.root / "embedding.yml"
        embedding.write_text("embedding:\n  model: pixel-mean\n")
        with patch("videospy.visual_memory.store.EmbeddingClient", PixelEncoder), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["build", "--cases", str(self.root / "assembled/cases.jsonl"),
                                   "--output", str(self.root / "index"), "--embedding-config", str(embedding),
                                   "--allow-trajectory-aligned"]), 0)
        self.assertTrue(json.loads((self.root / "index/manifest.json").read_text())["admission_policy"]["allow_trajectory_aligned"])

    def test_resume_budget_processes_next_source_without_repeating_completed_calls(self):
        self.row.pop("visual_fact_annotations", None)
        self.row["question"] = "Which phase is shown from 1 to 2 seconds?\nA. Cleaning and Coagulation\nB. Closure"
        (self.traces / "2.json").write_text(json.dumps({**self.row, "uid": 2}))
        inv = self.run_inventory()
        first, fake = self.align(config={"max_sources": 1}, sources=inv["sources_path"])
        self.assertEqual(first["unfinished_sources"], 1)
        second, _ = self.align(fake, {"max_sources": 1, "resume": True}, inv["sources_path"])
        self.assertEqual(second["candidate_count"], 2)
        self.assertEqual(second["unfinished_sources"], 0)
        self.assertEqual(len(fake.packets), 2)

    def test_new_admission_is_off_by_default_and_disable_never_loads_models(self):
        self.assertFalse(memory_config()["allow_trajectory_aligned"])
        with patch("videospy.visual_memory.integration.EmbeddingClient", side_effect=AssertionError("no embedding")):
            session = VisualMemorySession({"enabled": False, "allow_trajectory_aligned": True})
            self.assertFalse(session.settings["enabled"])


if __name__ == "__main__":
    unittest.main()
