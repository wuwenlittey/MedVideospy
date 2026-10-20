"""Offline behavior tests for reference-assisted visual observations.

Only synthetic pixels, temporary verified cases, and fake VLM replies are used.
The real immutable store is exercised, including its source/task exclusions.
"""

import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.test_videospy import (
    FakeModelClient, FakeVideoReader, agent_config, build_agent, clip_skim_config,
)
from videospy.tools.clip_skim import execute_clip_skim
from videospy.tools.frame_inspect import execute_frame_inspect
from videospy.visual_memory.integration import VisualMemorySession
from videospy.visual_memory.store import build_index


class PixelEncoder:
    """A deterministic image encoder; it never performs network requests."""

    identity = {"adapter": "test-pixels-v1", "model": "pixel-mean"}

    def __init__(self, config=None):
        self.calls = []

    def encode(self, images):
        self.calls.append([image.copy() for image in images])
        return np.asarray([
            np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1
            for image in images
        ], dtype=np.float32)


class ColorReader(FakeVideoReader):
    def get_batch(self, indices):
        self.indices = [int(index) for index in indices]
        frames = np.empty((len(indices), 24, 32, 3), dtype=np.uint8)
        frames[:] = (220, 10, 10)
        return type("Batch", (), {"asnumpy": lambda instance: frames})()


def joined_text(content):
    return "\n".join(part.get("text", "") for part in content)


def count_images(content):
    return sum(part.get("type") == "image_url" for part in content)


class VisualMemoryFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # Failed mock calls can retain traceback cycles containing a NumPy
        # memmap; collect them before removing its file on Windows.
        self.addCleanup(gc.collect)
        self.root = Path(self.directory.name)
        image_path = self.root / "reference.png"
        Image.new("RGB", (80, 60), (220, 10, 10)).save(image_path)
        template = {
            "question": "Which instrument is visible?",
            "answer_text": "The visible instrument is a needle holder.",
            "task_type": "C1", "verified": True, "split": "train",
            "verification_note": "Synthetic fixture, independently checked.",
            "evidence_scope": "frame", "timestamps": [125.0],
            "image_paths": [str(image_path)], "procedure": "repair",
        }
        self.records = [
            {**template, "case_id": "historical-reference", "source_video_id": "train-video",
             "source_sample_id": "train-qa", "source_id": "train-source", "patient_id": "train-patient"},
            {**template, "case_id": "same-video-excluded", "source_video_id": "current-video"},
            {**template, "case_id": "wrong-task-excluded", "source_video_id": "other-video", "task_type": "M1"},
        ]
        manifest = self.root / "cases.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")
        self.index = self.root / "index"
        build_index(manifest, self.index, {}, embedder=PixelEncoder())
        self.settings = {
            "enabled": True, "mode": "text", "database_path": str(self.index),
            "top_k": 2, "min_similarity": 0.5, "query_frames": 2,
            "reference_frames": 1, "reference_short_side": 48,
            "max_calls_per_question": 2, "apply_to": ["clip_skim", "frame_inspect"],
            "fail_open": True,
            "embedding": {"url": "http://example.invalid/embeddings", "model": "pixel-mean"},
        }
        self.context = {
            "source_video_id": "current-video", "task_type": "C1",
            "source_id": "current-source", "patient_id": "current-patient",
            "source_sample_id": "current-qa", "procedure": "repair",
        }
        self.parameters = {
            "query": "Describe the visible instrument.",
            "original_question": "Which instrument is shown? A. Holder B. Scissors",
            "_visual_memory_context": self.context,
        }
        self.frames = np.empty((4, 24, 32, 3), dtype=np.uint8)
        self.frames[:] = (220, 10, 10)
        self.timestamps = [1.0, 2.0, 3.0, 4.0]
        self.content = [{"type": "text", "text": "CURRENT OBSERVATION: 1–4 seconds."}]

    def session(self, **overrides):
        return VisualMemorySession({**self.settings, **overrides})

    def augment(self, session, **overrides):
        args = dict(frames=self.frames, timestamps=self.timestamps,
                    parameters=self.parameters, tool_name="clip_skim")
        args.update(overrides)
        return session.augment(self.content, **args)


class VisualMemorySessionTest(VisualMemoryFixture):
    def test_disabled_is_lazy_and_does_not_mutate_or_audit(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient") as encoder, patch(
            "videospy.visual_memory.integration.VisualMemoryStore"
        ) as store:
            session = VisualMemorySession({"enabled": False})
            result = self.augment(session)
        self.assertEqual(result, self.content)
        self.assertEqual(session.audit, [])
        encoder.assert_not_called()
        store.assert_not_called()

    def test_enabled_constructor_does_not_open_index_or_initialize_client(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient") as encoder, patch(
            "videospy.visual_memory.integration.VisualMemoryStore"
        ) as store:
            self.session()
        encoder.assert_not_called()
        store.assert_not_called()

    def test_text_reference_retrieves_real_store_and_excludes_same_video_and_wrong_task(self):
        original = copy.deepcopy(self.content)
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session()
            result = self.augment(session)
        self.assertEqual(self.content, original)
        text = joined_text(result)
        self.assertIn("historical-reference", text)
        self.assertIn("needle holder", text)
        self.assertNotIn("same-video-excluded", text)
        self.assertNotIn("wrong-task-excluded", text)
        self.assertEqual(count_images(result), 0)
        audit = json.dumps(session.audit, ensure_ascii=False)
        self.assertIn("historical-reference", audit)
        self.assertIn("train-video", audit)
        self.assertIn("same-video-excluded", audit)
        self.assertIn("wrong-task-excluded", audit)
        self.assertIn("sha256", audit)

    def test_image_text_appends_reference_image_and_preserves_original_content(self):
        original = copy.deepcopy(self.content)
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session(mode="image_text")
            result = self.augment(session)
        self.assertEqual(self.content, original)
        self.assertTrue(all(part in result for part in original))
        self.assertEqual(count_images(result), 1)
        self.assertIn("historical-reference", joined_text(result))
        self.assertIn("needle holder", joined_text(result))

    def test_no_match_preserves_original_observation_and_records_skip(self):
        context = {**self.context, "task_type": "unknown-task"}
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session()
            result = self.augment(session, parameters={**self.parameters, "_visual_memory_context": context})
        self.assertEqual(result, self.content)
        self.assertEqual(len(session.audit), 1)
        self.assertNotIn("needle holder", json.dumps(result))

    def test_embedding_failure_fails_open_without_partial_reference(self):
        class UnavailableEncoder(PixelEncoder):
            def encode(self, images):
                raise RuntimeError("embedding service unavailable")

        original = copy.deepcopy(self.content)
        with patch("videospy.visual_memory.integration.EmbeddingClient", UnavailableEncoder):
            session = self.session(mode="image_text")
            result = self.augment(session)
        self.assertEqual(result, original)
        self.assertEqual(self.content, original)
        self.assertEqual(len(session.audit), 1)

    def test_reference_image_failure_is_atomic_after_successful_retrieval(self):
        original = copy.deepcopy(self.content)
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.visual_memory.integration.Image.open", side_effect=OSError("reference unreadable")
        ):
            session = self.session(mode="image_text")
            result = self.augment(session)
        self.assertEqual(result, original)
        self.assertEqual(self.content, original)
        self.assertEqual(count_images(result), 0)
        self.assertEqual(session.audit[0]["status"], "failed")

    def test_corrupt_store_fails_open_and_never_adds_reference(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session(database_path=str(self.root / "missing-index"))
            result = self.augment(session)
        self.assertEqual(result, self.content)
        self.assertEqual(len(session.audit), 1)

    def test_identity_mismatch_is_not_silently_used(self):
        class WrongEncoder(PixelEncoder):
            identity = {"adapter": "test-pixels-v1", "model": "different-model"}

        with patch("videospy.visual_memory.integration.EmbeddingClient", WrongEncoder):
            session = self.session()
            result = self.augment(session)
        self.assertEqual(result, self.content)
        self.assertEqual(len(session.audit), 1)

    def test_overview_never_queries_visual_memory(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient") as encoder:
            session = self.session()
            result = self.augment(session, tool_name="overview")
        self.assertEqual(result, self.content)
        encoder.assert_not_called()

    def test_missing_identity_does_not_fall_back_to_unscoped_search(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session()
            result = self.augment(session, parameters={"query": "instrument"})
        self.assertEqual(result, self.content)
        self.assertEqual(len(session.audit), 1)

    def test_cap_and_reset_control_query_budget(self):
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            session = self.session(max_calls_per_question=1)
            first = self.augment(session)
            second = self.augment(session)
            self.assertIn("historical-reference", joined_text(first))
            self.assertEqual(second, self.content)
            session.reset()
            self.assertEqual(session.audit, [])
            third = self.augment(session)
        self.assertIn("historical-reference", joined_text(third))


class VisualMemoryToolHookTest(VisualMemoryFixture):
    def tool_parameters(self, session):
        return {**self.parameters, "vr": ColorReader(), "subtitles": [],
                "_visual_memory_session": session}

    def clip_reference_index(self):
        case = {**self.records[0], "case_id": "verified-two-frame-clip",
                "evidence_scope": "clip", "timestamps": [125.0, 126.0],
                "image_paths": [self.records[0]["image_paths"][0]] * 2}
        manifest = self.root / "clip-case.jsonl"
        manifest.write_text(json.dumps(case) + "\n", encoding="utf-8")
        index = self.root / "clip-index"
        build_index(manifest, index, {}, embedder=PixelEncoder())
        return index

    def test_clip_reference_over_budget_is_skipped_without_partial_label_or_images(self):
        index = self.clip_reference_index()
        model = FakeModelClient([{"content": "Current observation."}])
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.clip_skim.ModelClient", return_value=model
        ):
            session = self.session(mode="image_text", database_path=str(index), reference_frames=1)
            execute_clip_skim(clip_skim_config(num_frames=4), {
                **self.tool_parameters(session), "start_time": 1.0, "end_time": 4.0,
            })
        self.assertEqual(len(model.calls), 1)
        content = model.calls[0]["messages"][0]["content"]
        self.assertEqual(count_images(content), 4)
        self.assertNotIn("verified-two-frame-clip", joined_text(content))
        self.assertNotIn("needle holder", joined_text(content))
        self.assertEqual(session.audit[0]["reason"], "reference_frame_budget")

    def test_clip_reference_with_sufficient_budget_keeps_all_verified_frames_in_one_vlm_call(self):
        index = self.clip_reference_index()
        model = FakeModelClient([{"content": "Current observation."}])
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.clip_skim.ModelClient", return_value=model
        ):
            session = self.session(mode="image_text", database_path=str(index), reference_frames=2)
            execute_clip_skim(clip_skim_config(num_frames=4), {
                **self.tool_parameters(session), "start_time": 1.0, "end_time": 4.0,
            })
        self.assertEqual(len(model.calls), 1)
        content = model.calls[0]["messages"][0]["content"]
        self.assertEqual(count_images(content), 6)
        self.assertIn("verified-two-frame-clip", joined_text(content))
        self.assertIn("needle holder", joined_text(content))
        references = session.audit[0]["matches"][0]["reference_images"]
        self.assertEqual([frame["timestamp"] for frame in references], [125.0, 126.0])

    def test_clip_hook_injects_reference_into_existing_single_vlm_call(self):
        model = FakeModelClient([{"content": "Current surgical observation."}])
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.clip_skim.ModelClient", return_value=model
        ):
            session = self.session()
            result = execute_clip_skim(clip_skim_config(num_frames=4), {
                **self.tool_parameters(session), "start_time": 1.0, "end_time": 4.0,
            })
        self.assertEqual(len(model.calls), 1)
        content = model.calls[0]["messages"][0]["content"]
        self.assertIn("historical-reference", joined_text(content))
        self.assertEqual(count_images(content), 4)
        self.assertIn("Current surgical observation.", result)
        self.assertEqual(len(session.audit), 1)

    def test_frame_hook_injects_reference_images_into_existing_single_vlm_call(self):
        model = FakeModelClient([{"content": "Visible detail."}])
        config = {"max_frames": 4, "model": clip_skim_config()["model"]}
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder), patch(
            "videospy.tools.frame_inspect.ModelClient", return_value=model
        ):
            session = self.session(mode="image_text")
            result = execute_frame_inspect(config, {**self.tool_parameters(session), "timestamps": [1.0, 2.0]})
        self.assertEqual(len(model.calls), 1)
        content = model.calls[0]["messages"][0]["content"]
        self.assertEqual(count_images(content), 3)
        self.assertIn("historical-reference", joined_text(content))
        self.assertIn("Visible detail.", result)

    def test_disabled_session_preserves_exact_clip_request_and_observation(self):
        parameters = {**self.tool_parameters(None), "start_time": 1.0, "end_time": 4.0}
        parameters.pop("_visual_memory_session")
        original_model = FakeModelClient([{"content": "Unchanged."}])
        disabled_model = FakeModelClient([{"content": "Unchanged."}])
        with patch("videospy.tools.clip_skim.ModelClient", return_value=original_model):
            original = execute_clip_skim(clip_skim_config(4), parameters)
        with patch("videospy.tools.clip_skim.ModelClient", return_value=disabled_model):
            disabled = execute_clip_skim(clip_skim_config(4), {
                **parameters, "_visual_memory_session": VisualMemorySession({"enabled": False}),
            })
        self.assertEqual(original, disabled)
        self.assertEqual(original_model.calls, disabled_model.calls)

    def test_agent_keeps_audit_in_trajectory_but_out_of_model_messages(self):
        config = agent_config()
        config["tools"]["enabled"] = ["clip_skim"]
        config["tools"]["visual_memory"] = self.settings
        config["_runtime"] = {"task_type": "C1"}
        with patch("videospy.visual_memory.integration.EmbeddingClient", PixelEncoder):
            agent = build_agent(config, FakeModelClient([]))
            agent.vr = ColorReader()
            agent.question = self.parameters["original_question"]
            agent.memory_context = {**self.context, "benchmark": "medvidbench",
                                    "video": "current-video", "sample_id": "current-qa"}
            agent.memory_audit = {}
            call = {"id": "observe", "function": {"name": "clip_skim", "arguments": json.dumps({
                "query": "Describe the instrument.", "start_time": 1.0, "end_time": 4.0,
            })}}
            model = FakeModelClient([{"content": "Current observation."}])
            with patch("videospy.tools.clip_skim.ModelClient", return_value=model):
                message, succeeded = agent._execute_tool_call(call)
        self.assertTrue(succeeded)
        self.assertIn("visual_memory", message)
        agent.messages.append(message)
        self.assertTrue(all("visual_memory" not in item for item in agent._messages_for_model()))
        trajectory = agent._trajectory(1, "stop").to_dict()
        self.assertIn("visual_memory", trajectory["diagnostics"])
        self.assertIn("historical-reference", json.dumps(trajectory["diagnostics"]["visual_memory"]))
        agent.reset()
        session = getattr(agent, "visual_memory_session", None)
        if session is not None:
            self.assertEqual(session.audit, [])


if __name__ == "__main__":
    unittest.main()
