"""Two-stage reference use and its current-only control; no external models."""

import copy
import json
import unittest
import types
from unittest.mock import patch

from tests.test_visual_memory_integration import VisualMemoryFixture, PixelEncoder, ColorReader, joined_text, count_images
from tests.test_videospy import FakeModelClient, clip_skim_config, model_config
from videospy.tools.clip_skim import execute_clip_skim
from videospy.tools.frame_inspect import execute_frame_inspect
from videospy.visual_memory.integration import VisualMemorySession, memory_config
from videospy.visual_memory.review import validate_review
from videospy.model import collect_model_calls


ORIGINAL = "The instrument is scissors."


def check(*, current_only=False, **changes):
    value = {"original_claim": ORIGINAL, "fact_type": "instrument_identity", "status": "conflict",
             "current_frame_ids": ["current_0"], "reference_fact_ids": [] if current_only else ["reference_0"],
             "visible_difference": "The visible tip matches the reference jaws.",
             "revision_suggestion": "The instrument may be a needle holder.", "temporal_change": ""}
    return {"checks": [{**value, **changes}]}


class FactReviewValidationTests(unittest.TestCase):
    def validate(self, value, **kwargs):
        return validate_review(value, original_observation=ORIGINAL, timestamps=[1, 2],
                               references={"reference_0": {}}, **kwargs)

    def test_valid_narrow_current_evidence_correction(self):
        self.assertEqual(self.validate(check())["checks"][0]["status"], "conflict")

    def test_invalid_references_or_missing_current_frames_rejected(self):
        for values in ({"current_frame_ids": ["current_99"]}, {"reference_fact_ids": ["invented"]},
                       {"current_frame_ids": []}, {"reference_fact_ids": []},
                       {"original_claim": "Never present in original"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.validate(check(**values))

    def test_current_only_cannot_claim_historical_support(self):
        with self.assertRaises(ValueError):
            self.validate(check(), current_only=True)
        self.validate(check(current_only=True), current_only=True)

    def test_action_needs_ordered_frames_and_change(self):
        for values in ({"current_frame_ids": ["current_0"]},
                       {"current_frame_ids": ["current_1", "current_0"], "temporal_change": "Moves"},
                       {"current_frame_ids": ["current_0", "current_1"]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.validate(check(fact_type="action", **values))
        self.validate(check(fact_type="action", current_frame_ids=["current_0", "current_1"], temporal_change="The tip moves."))

    def test_unknown_cannot_supply_a_revision(self):
        result = self.validate(check(status="unknown", current_frame_ids=[], reference_fact_ids=[], revision_suggestion="Ignore image"))
        self.assertEqual(result["checks"][0]["revision_suggestion"], "")


class FactReviewRuntimeTests(VisualMemoryFixture):
    def tool_parameters(self, session):
        return {**self.parameters, "vr": ColorReader(), "subtitles": [], "start_time": 0, "end_time": 4,
                "timestamps": [1.0], "_visual_memory_session": session}

    def execute(self, client, session, *, frame=False):
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.clip_skim.ModelClient", return_value=client), patch(
            "videospy.tools.frame_inspect.ModelClient", return_value=client), patch(
            "videospy.visual_memory.review.ModelClient", return_value=client):
            if frame:
                return execute_frame_inspect({"max_frames": 1, "model": model_config("inspect")}, self.tool_parameters(session))
            return execute_clip_skim(clip_skim_config(num_frames=2), self.tool_parameters(session))

    def test_first_observation_never_sees_reference_then_review_uses_same_current_images(self):
        session = self.session(usage_mode="review", mode="image_text", top_k=1)
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check())}])
        result = self.execute(client, session)
        self.assertEqual(len(client.calls), 2)
        first, second = [call["messages"][0]["content"] for call in client.calls]
        self.assertNotIn("HISTORICAL", joined_text(first))
        self.assertNotIn("historical-reference", joined_text(first))
        self.assertEqual(count_images(first), 2)
        self.assertEqual(count_images(second), 3)
        self.assertIn("current_0", joined_text(second))
        self.assertIn("historical-reference", joined_text(second))
        self.assertIn(ORIGINAL, result)
        self.assertIn("needle holder", result)
        self.assertEqual(session.audit[-1]["original_observation"], ORIGINAL)

    def test_frame_tool_also_performs_two_stages(self):
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check())}])
        result = self.execute(client, self.session(usage_mode="review", top_k=1), frame=True)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("Additional visual fact review", result)

    def test_current_only_needs_neither_index_nor_embeddings(self):
        session = VisualMemorySession({"enabled": True, "usage_mode": "current_only_review"})
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check(current_only=True))}])
        with patch.object(session, "_load", side_effect=AssertionError("must not retrieve")):
            result = self.execute(client, session)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(count_images(client.calls[1]["messages"][0]["content"]), 2)
        self.assertIn("CURRENT-ONLY", joined_text(client.calls[1]["messages"][0]["content"]))
        self.assertIn(ORIGINAL, result)

    def test_malformed_review_keeps_exact_original_tool_result(self):
        session = self.session(usage_mode="review", top_k=1)
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check(current_frame_ids=["current_999"]))}])
        result = self.execute(client, session)
        self.assertNotIn("Additional visual fact review", result)
        self.assertTrue(result.endswith(ORIGINAL))
        self.assertEqual(session.audit[-1]["status"], "review_failed")

    def test_unknown_review_keeps_original(self):
        session = self.session(usage_mode="review", top_k=1)
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check(status="unknown", revision_suggestion=""))}])
        result = self.execute(client, session)
        self.assertTrue(result.endswith(ORIGINAL))
        self.assertEqual(session.audit[-1]["review_status"], "unknown_only")

    def test_no_matches_does_not_add_vlm_call(self):
        session = self.session(usage_mode="review", context={"task_type": "nonexistent"})
        self.parameters["_visual_memory_context"]["task_type"] = "nonexistent"
        client = FakeModelClient([{"content": ORIGINAL}])
        result = self.execute(client, session)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(result.endswith(ORIGINAL))

    def test_budget_and_reset_control_additional_calls(self):
        session = self.session(usage_mode="current_only_review", review={"max_calls_per_question": 1})
        client = FakeModelClient([{"content": ORIGINAL}, {"content": json.dumps(check(current_only=True))}, {"content": ORIGINAL}])
        self.execute(client, session)
        self.execute(client, session)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(session.audit[-1]["reason"], "question_review_budget")
        session.reset()
        self.assertEqual(session._review_attempts, 0)

    def test_runtime_summary_omits_model_credentials(self):
        session = self.session(review={"model": {"api_key": "SECRET_TEST", "api_base": "https://secret.example"}})
        self.assertNotIn("SECRET_TEST", json.dumps(session.summary()))
        self.assertNotIn("secret.example", json.dumps(session.summary()))

    def test_alias_mapping_validates_and_unknown_tasks_skip_before_embedding(self):
        settings = memory_config({"task_fact_types": {"Custom_Task": ["anatomy"]}})
        self.assertEqual(settings["task_fact_types"], {"custom task": ["anatomy"]})
        session = self.session(match_mode="fact")
        self.parameters["_visual_memory_context"]["task_type"] = "new unknown task"
        with patch.object(session, "_load", side_effect=AssertionError("must skip unknown")):
            self.augment(session)
        self.assertEqual(session.audit[-1]["reason"], "unknown_fact_task_or_empty_capability")

    def test_two_calls_are_recorded_by_existing_model_metrics(self):
        session = VisualMemorySession({"enabled": True, "usage_mode": "current_only_review"})
        replies = [types.SimpleNamespace(choices=[types.SimpleNamespace(message={"content": value}, finish_reason="stop")],
                                         usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
                   for value in (ORIGINAL, json.dumps(check(current_only=True)))]
        with patch("videospy.model.completion", side_effect=replies), collect_model_calls("compact") as calls:
            execute_clip_skim(clip_skim_config(num_frames=2), self.tool_parameters(session))
        self.assertEqual(len(calls), 2)
        self.assertEqual(sum(call["metrics"]["input_tokens"] for call in calls), 20)
        self.assertNotIn("HISTORICAL", json.dumps(calls[0]["input"]["messages"]))


if __name__ == "__main__":
    unittest.main()
