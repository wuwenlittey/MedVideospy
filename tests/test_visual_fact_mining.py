"""Exercise offline extraction/review with deterministic fake VLMs and pixels."""

import copy
import json
import os
import gc
import io
from contextlib import redirect_stderr
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_visual_memory_curation as curation_tests
from videospy.visual_memory.mining import _balanced, mine_facts
from videospy.visual_memory.vlm import resolve_model_config, safe_model_identity


def candidate(kind="instrument_identity", text="A needle holder is visible.", ids=None):
    return {"type": kind, "text": text, "entity": "tool", "conditions": ["Visible jaws"],
            "evidence_frame_ids": [1] if ids is None else ids}


class FakeFactClient:
    def __init__(self, responses, name="fake-vlm"):
        self.responses = list(responses)
        self.identity = {"model_name": name}
        self.calls = []

    def generate_json(self, prompt, images, *, purpose):
        self.calls.append({"prompt": prompt, "ids": [i for i, image in images], "purpose": purpose})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


class FactMiningTests(unittest.TestCase):
    setUp = curation_tests.VisualMemoryCurationTests.setUp
    tearDown = curation_tests.VisualMemoryCurationTests.tearDown
    selection = curation_tests.VisualMemoryCurationTests.selection
    def sources(self, **changes):
        selection = self.selection()
        selection.pop("tool_call_id")
        selection.update(changes)
        path = self.root / "sources.jsonl"
        path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        return path

    def run_mine(self, extractor, reviewer, settings=None, sources=None):
        clients = iter([extractor, reviewer])
        return mine_facts(sources or self.root / "sources.jsonl", self.root / "mined",
                          {"vlm": {"model_name": "fake"}, **(settings or {})},
                          client_factory=lambda config: next(clients))

    def records(self, report):
        return [json.loads(line) for line in Path(report["cases_path"]).read_text().splitlines()]

    def test_failed_qa_is_mined_from_actual_frames_and_review_subset_only(self):
        self.sources()
        extractor = FakeFactClient([{"facts": [candidate()], "unknowns": []}])
        reviewer = FakeFactClient([{"decision": "supported", "reason": "Visible characteristic jaws.",
                                    "evidence_frame_ids": [1], "temporal_change": ""}])
        report = self.run_mine(extractor, reviewer)
        records = self.records(report)
        self.assertEqual(report["selected_fact_count"], 1)
        self.assertEqual(extractor.calls[0]["ids"], [0, 1, 2, 3])
        self.assertEqual(reviewer.calls[0]["ids"], [1])
        self.assertNotIn("Cutting tissue", extractor.calls[0]["prompt"])
        self.assertFalse(records[0]["verified"])
        self.assertEqual(records[0]["fact"]["evidence_frame_ids"], [0])
        self.assertEqual(records[0]["timestamps"], [1.5])
        self.assertEqual(records[0]["verification"]["status"], "model_reviewed")

    def test_unknown_and_invalid_fact_ids_are_not_banked(self):
        self.sources()
        extractor = FakeFactClient([{"facts": [candidate(ids=[99]), candidate(),
                                                candidate(text="Histology is adenoma.")]}])
        reviewer = FakeFactClient([{"decision": "unknown", "reason": "Jaws not clear.", "evidence_frame_ids": []}])
        report = self.run_mine(extractor, reviewer)
        self.assertEqual(report["selected_fact_count"], 0)
        self.assertEqual(len(reviewer.calls), 1)
        self.assertEqual(report["pending_outcomes"]["pending"], 2)
        self.assertEqual(report["pending_outcomes"]["unknown"], 1)

    def test_invalid_review_ids_retry_on_resume_without_reextracting(self):
        self.sources()
        extractor = FakeFactClient([{"facts": [candidate()]}])
        bad_review = FakeFactClient([{"decision": "supported", "reason": "Claim.", "evidence_frame_ids": [2]}])
        first = self.run_mine(extractor, bad_review)
        self.assertEqual(first["status"], "completed_with_errors")
        self.assertEqual(first["selected_fact_count"], 0)
        extractor2 = FakeFactClient([])
        good_review = FakeFactClient([{"decision": "supported", "reason": "The cited jaws are visible.",
                                      "evidence_frame_ids": [1]}])
        second = self.run_mine(extractor2, good_review, {"resume": True})
        self.assertEqual(second["selected_fact_count"], 1)
        self.assertEqual(extractor2.calls, [])
        self.assertEqual(second["new_vlm_calls"], {"review": 1})
        third = self.run_mine(FakeFactClient([]), FakeFactClient([]), {"resume": True})
        self.assertEqual(third["new_vlm_calls"], {})

    def test_action_requires_current_change_and_multiple_referenced_images(self):
        self.sources()
        extractor = FakeFactClient([{"facts": [candidate("action", "Tool moves tissue.", [0, 2])]}])
        reviewer = FakeFactClient([{"decision": "supported", "reason": "Tissue changes position.",
                                    "evidence_frame_ids": [0, 2], "temporal_change": "Tissue moves left."}])
        report = self.run_mine(extractor, reviewer)
        record = self.records(report)[0]
        self.assertEqual(record["evidence_scope"], "clip")
        self.assertEqual(record["timestamps"], [0.5, 3.0])
        self.assertEqual(reviewer.calls[0]["ids"], [0, 2])

    def test_invalid_extraction_json_shape_is_retryable(self):
        self.sources()
        first = self.run_mine(FakeFactClient([{"other": []}]), FakeFactClient([]))
        self.assertEqual(first["status"], "completed_with_errors")
        second = self.run_mine(FakeFactClient([{"facts": []}]), FakeFactClient([]), {"resume": True})
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["new_vlm_calls"], {"extraction": 1})

    def test_resume_refuses_changed_sources_or_model(self):
        self.sources()
        self.run_mine(FakeFactClient([{"facts": []}]), FakeFactClient([]))
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_mine(FakeFactClient([], name="changed"), FakeFactClient([]), {"resume": True})

    def test_missing_model_configuration_does_not_reserve_output(self):
        self.sources()
        with self.assertRaises(ValueError):
            mine_facts(self.root / "sources.jsonl", self.root / "mined", {"vlm": {}})
        self.assertFalse((self.root / "mined").exists())

    def test_test_sources_rejected_without_model_calls(self):
        self.sources(split="test")
        extractor, reviewer = FakeFactClient([]), FakeFactClient([])
        report = self.run_mine(extractor, reviewer)
        self.assertEqual(len(report["source_errors"]), 1)
        self.assertEqual(extractor.calls, [])
        self.assertEqual(reviewer.calls, [])

    def test_output_not_overwritten_and_image_budget_not_silently_subsampled(self):
        self.sources()
        report = self.run_mine(FakeFactClient([]), FakeFactClient([]), {"max_images_per_observation": 2})
        self.assertEqual(report["observations"][0]["reason"], "observation_exceeds_image_budget")
        with self.assertRaises(FileExistsError):
            self.run_mine(FakeFactClient([]), FakeFactClient([]))

    def test_type_balancing_protects_rare_facts(self):
        def row(kind, index):
            return {"case_id": str(index), "fact": {"type": kind, "text": str(index)},
                    "source_video_id": str(index), "provenance": {"support_image_sha256": [str(index)]}}
        records = [row("action", i) for i in range(20)] + [row("anatomy", 99)]
        selected, _, _ = _balanced(records, 20, 2)
        self.assertEqual({item["fact"]["type"] for item in selected}, {"action", "anatomy"})

    def test_observation_budget_can_resume_without_repeating_completed_calls(self):
        self.sources()
        path = self.root / "trajectory.json"
        trajectory = json.loads(path.read_text())
        second = copy.deepcopy(trajectory["messages"])
        second[0]["tool_calls"][0]["id"] = "call_2"
        second[1]["tool_call_id"] = "call_2"
        trajectory["messages"].extend(second)
        path.write_text(json.dumps(trajectory))
        first = self.run_mine(FakeFactClient([{"facts": []}]), FakeFactClient([]), {"max_observations": 1})
        self.assertEqual(first["status"], "budget_limited")
        self.assertEqual(first["new_vlm_calls"], {"extraction": 1})
        second = self.run_mine(FakeFactClient([{"facts": []}]), FakeFactClient([]), {"max_observations": 1, "resume": True})
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["new_vlm_calls"], {"extraction": 1})

    def test_mined_fact_cli_build_and_cross_task_lookup_end_to_end(self):
        from tests.test_visual_memory_integration import PixelEncoder
        from videospy.visual_memory.cli import main
        from videospy.visual_memory.store import VisualMemoryStore
        import numpy as np

        self.sources()
        report = self.run_mine(FakeFactClient([{"facts": [candidate()]}]),
                               FakeFactClient([{"decision": "supported", "reason": "Visible jaws.", "evidence_frame_ids": [1]}]))
        config = self.root / "embedding.yml"
        config.write_text("model: fake\n", encoding="utf-8")
        args = ["build", "--cases", report["cases_path"], "--output", str(self.root / "index"),
                "--embedding-config", str(config)]
        with patch("videospy.visual_memory.store.EmbeddingClient", PixelEncoder), patch("builtins.print"):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                main(args)
            self.assertEqual(main(args + ["--allow-model-reviewed"]), 0)
        store = VisualMemoryStore(self.root / "index")
        result = store.search(np.ones(3), context={"source_video_id": "heldout-original", "task_type": "instrument_onset"},
                              match_mode="fact", allow_model_reviewed=True)
        self.assertEqual(len(result["matches"]), 1)
        self.assertFalse(result["matches"][0]["case"]["verified"])
        del store
        gc.collect()


class VLMConfigurationTests(unittest.TestCase):
    def test_reviewer_env_override_wins_and_public_identity_hides_secrets(self):
        with patch.dict(os.environ, {"REVIEW_MODEL": "openai/reviewer", "REVIEW_KEY": "secret-test"}):
            config = resolve_model_config({"model_name_env": "REVIEW_MODEL", "api_key_env": "REVIEW_KEY"},
                                          fallback={"model_name": "openai/extractor", "api_base": "https://user:password@example.test/v1?token=hidden"})
        self.assertEqual(config["model_name"], "openai/reviewer")
        self.assertEqual(config["api_key"], "secret-test")
        safe = json.dumps(safe_model_identity(config))
        for secret in ("secret-test", "password", "hidden"):
            self.assertNotIn(secret, safe)


if __name__ == "__main__":
    unittest.main()
