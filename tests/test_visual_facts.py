"""Synthetic schema, admission-policy and fact-transfer tests; no model calls."""

import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from videospy.visual_memory.facts import (
    allowed_fact_types, normalize_fact_task, requested_fact_types, validate_fact_record,
)
from videospy.visual_memory.store import VisualMemoryStore, build_index, validate_case


def fact_record(case_id="fact-1", **changes):
    text = "A needle driver holds a curved needle."
    record = {
        "record_kind": "visual_fact", "case_id": case_id,
        "question": "What is being performed?", "answer_text": text,
        "source_video_id": "train-" + case_id, "source_id": "source-" + case_id,
        "source_sample_id": "sample-" + case_id, "patient_id": "patient-" + case_id,
        "task_type": "action recognition", "procedure": "repair", "split": "train",
        "image_paths": ["frame.png"], "timestamps": [10.0], "evidence_scope": "frame",
        "verified": False, "verification_note": "Model review; pending human verification.",
        "fact": {"type": "instrument_identity", "text": text, "basis": "visual",
                 "entity": "needle driver", "conditions": ["Visible jaws and needle"],
                 "evidence_frame_ids": [0]},
        "verification": {"status": "model_reviewed", "decision": "supported",
                         "reason": "The referenced frame shows the jaw and curved needle.",
                         "model": {"model_name": "test-reviewer"},
                         "human_verified": False, "evidence_frame_ids": [0],
                         "temporal_change": None},
    }
    record.update(changes)
    return record


class FactSchemaTest(unittest.TestCase):
    def test_machine_review_is_valid_provenance_but_never_promotes_verified(self):
        original = fact_record()
        result = validate_fact_record(original)
        self.assertFalse(result["verified"])
        self.assertFalse(result["verification"]["human_verified"])
        result["fact"]["conditions"].append("new")
        self.assertNotIn("new", original["fact"]["conditions"])

    def test_rejects_answer_based_or_unsupported_fact_kinds(self):
        for changes in ({"basis": "gold_answer"}, {"type": "histology"},
                        {"type": "lesion_identity"}, {"type": "global_count"},
                        {"type": "onset_time"}):
            case = fact_record()
            case["fact"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_fact_record(case)

    def test_fact_text_must_not_be_replaced_by_qa_answer(self):
        with self.assertRaisesRegex(ValueError, "answer_text"):
            validate_fact_record(fact_record(answer_text="Pedicle control"))

    def test_final_evidence_ids_and_scope_are_exact(self):
        for ids in ([], [1], [0, 1], [True], [0.0], "0"):
            case = fact_record()
            case["fact"]["evidence_frame_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_fact_record(case)
        case = fact_record()
        case["verification"]["evidence_frame_ids"] = []
        with self.assertRaises(ValueError):
            validate_fact_record(case)
        with self.assertRaises(ValueError):
            validate_fact_record(fact_record(evidence_scope="clip"))

    def test_bad_scope_times_and_conditions_are_rejected(self):
        cases = [fact_record(timestamps=[-1]), fact_record(timestamps=[True]),
                 fact_record(timestamps=[float("nan")]), fact_record(image_paths=[]),
                 fact_record(split="test")]
        invalid_conditions = fact_record()
        invalid_conditions["fact"]["conditions"] = [""]
        cases.append(invalid_conditions)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_fact_record(case)

    def test_action_requires_multiple_frames_and_described_change(self):
        case = fact_record()
        case["fact"]["type"] = "action"
        with self.assertRaisesRegex(ValueError, "two"):
            validate_fact_record(case)
        case.update(image_paths=["frame.png", "frame2.png"], timestamps=[10.0, 11.0], evidence_scope="clip")
        case["fact"]["evidence_frame_ids"] = [0, 1]
        case["verification"]["evidence_frame_ids"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "temporal_change"):
            validate_fact_record(case)
        case["verification"]["temporal_change"] = "The jaws close around the needle between the two frames."
        self.assertEqual(validate_fact_record(case)["evidence_scope"], "clip")

    def test_machine_status_cannot_claim_human_verification_or_unknown_review(self):
        for key, value in (("decision", "unknown"), ("reason", ""), ("model", {}), ("human_verified", True)):
            case = fact_record()
            case["verification"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_fact_record(case)
        with self.assertRaises(ValueError):
            validate_fact_record(fact_record(verified=True))

    def test_human_review_requires_explicit_consistent_flags(self):
        case = fact_record(verified=True)
        case["verification"].update(status="human_verified", human_verified=True)
        self.assertTrue(validate_fact_record(case)["verified"])

    def test_aliases_are_explicit_and_unknown_never_means_all(self):
        self.assertEqual(normalize_fact_task("MedVidBench_C1"), "c1")
        self.assertIn("action", allowed_fact_types("action_recognition"))
        self.assertIn("instrument_identity", allowed_fact_types("instrument_onset"))
        self.assertIn("lesion_attribute", allowed_fact_types("site-histology"))
        self.assertIn("lesion_attribute", allowed_fact_types("lesion_size"))
        self.assertIn("action", allowed_fact_types("global_statistics"))
        self.assertEqual(allowed_fact_types("unknown-task"), frozenset())
        self.assertEqual(requested_fact_types("unknown-task", ["anatomy"]), {"anatomy"})
        for values in ([], ["histology"], "action"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                requested_fact_types("unknown-task", values)


class PixelEncoder:
    identity = {"model": "test-pixels", "dimension": 2}

    def encode(self, images):
        return np.asarray([[1.0, 0.2] for image in images], dtype=np.float32)


class FactStoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="visual-facts-test-", dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.directory.name)
        Image.new("RGB", (8, 6), (100, 20, 10)).save(self.root / "frame.png")
        self.manifest = self.root / "cases.jsonl"
        self.index = self.root / "index"

    def build(self, cases, **kwargs):
        self.manifest.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
        return build_index(self.manifest, self.index, {}, embedder=PixelEncoder(), **kwargs)

    def context(self, **changes):
        context = {"source_video_id": "eval-video", "task_type": "instrument_onset", "procedure": "repair"}
        context.update(changes)
        return context

    def search(self, store, **kwargs):
        return store.search([1.0, 0.2], context=self.context(), match_mode="fact", allow_model_reviewed=True, **kwargs)

    def test_default_admission_refuses_machine_facts_and_flag_requires_boolean(self):
        case = fact_record()
        with self.assertRaisesRegex(ValueError, "allow_model_reviewed"):
            validate_case(case)
        self.assertFalse(validate_case(case, allow_model_reviewed=True)["verified"])
        with self.assertRaisesRegex(ValueError, "allow_model_reviewed"):
            self.build([case])
        with self.assertRaises(ValueError):
            self.build([case], allow_model_reviewed="true")

    def test_explicit_build_records_policy_but_runtime_still_requires_opt_in(self):
        manifest = self.build([fact_record()], allow_model_reviewed=True)
        self.assertTrue(manifest["admission_policy"]["allow_model_reviewed"])
        store = VisualMemoryStore(self.index)
        result = store.search([1.0, 0.2], context=self.context(), match_mode="fact")
        self.assertEqual(result["matches"], [])
        self.assertIn("model_reviewed_not_enabled", result["exclusions"][0]["reasons"])
        self.assertEqual(len(self.search(store)["matches"]), 1)

    def test_cross_task_fact_is_usable_but_task_mode_retains_exact_matching(self):
        self.build([fact_record()], allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        self.assertEqual(len(self.search(store)["matches"]), 1)
        self.assertEqual(store.search([1.0, 0.2], context=self.context(), allow_model_reviewed=True)["matches"], [])

    def test_legacy_qa_cannot_cross_through_fact_mode_even_if_verified(self):
        legacy = fact_record(verified=True)
        for key in ("record_kind", "fact", "verification"):
            legacy.pop(key)
        self.build([legacy])
        store = VisualMemoryStore(self.index)
        result = self.search(store)
        self.assertEqual(result["matches"], [])
        self.assertIn("legacy_qa_not_cross_task_fact", result["exclusions"][0]["reasons"])
        same_task = self.context(task_type="action recognition")
        self.assertEqual(len(store.search([1.0, 0.2], context=same_task)["matches"]), 1)

    def test_unknown_task_skips_without_explicit_capability_request(self):
        self.build([fact_record()], allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        context = self.context(task_type="new-unmapped-task")
        result = store.search([1.0, 0.2], context=context, match_mode="fact", allow_model_reviewed=True)
        self.assertEqual(result["matches"], [])
        result = store.search([1.0, 0.2], context=context, match_mode="fact", fact_types=["instrument_identity"], allow_model_reviewed=True)
        self.assertEqual(len(result["matches"]), 1)

    def test_same_video_patient_source_and_sample_remain_excluded(self):
        cases = [fact_record("video", source_video_id="eval-video"),
                 fact_record("patient", patient_id="eval-patient"),
                 fact_record("source", source_id="eval-source"),
                 fact_record("sample", source_sample_id="eval-sample"), fact_record("valid")]
        self.build(cases, allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        result = store.search([1.0, 0.2], context=self.context(patient_id="eval-patient", source_id="eval-source",
                              source_sample_id="eval-sample"), match_mode="fact", allow_model_reviewed=True, top_k=8)
        self.assertEqual([match["case_id"] for match in result["matches"]], ["valid"])
        self.assertEqual(len(result["exclusions"]), 4)

    def test_procedure_mismatch_still_blocks_generic_facts(self):
        self.build([fact_record(procedure="different-procedure")], allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        result = self.search(store)
        self.assertEqual(result["matches"], [])
        self.assertIn("procedure_mismatch", result["exclusions"][0]["reasons"])

    def test_phase_and_lesion_facts_require_known_matching_procedure(self):
        phase, lesion = fact_record("phase"), fact_record("lesion")
        phase["fact"]["type"] = "phase"
        lesion["fact"]["type"] = "lesion_attribute"
        self.build([phase, lesion], allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        for fact_type in ("phase", "lesion_attribute"):
            with self.subTest(fact_type=fact_type):
                missing = store.search([1.0, 0.2], context=self.context(procedure=None), match_mode="fact",
                                      fact_types=[fact_type], allow_model_reviewed=True)
                self.assertEqual(missing["matches"], [])
                matched = self.search(store, fact_types=[fact_type])
                self.assertEqual(len(matched["matches"]), 1)

    def test_fact_type_request_is_validated_and_does_not_relax_task_mode(self):
        self.build([fact_record()], allow_model_reviewed=True)
        store = VisualMemoryStore(self.index)
        for kwargs in ({"match_mode": "all"}, {"fact_types": ["anatomy"]},
                       {"allow_model_reviewed": 1}, {"match_mode": "fact", "fact_types": ["diagnosis"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                store.search([1.0, 0.2], context=self.context(), **kwargs)

    def test_tampering_admission_policy_cannot_make_machine_facts_human(self):
        self.build([fact_record()], allow_model_reviewed=True)
        path = self.index / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["admission_policy"]["allow_model_reviewed"] = False
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "allow_model_reviewed"):
            VisualMemoryStore(self.index)


if __name__ == "__main__":
    unittest.main()
