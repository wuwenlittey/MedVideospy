"""Tests use synthetic pixel arrays and mock model responses; no API or data access."""

import base64
import io
import json
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.test_videospy import (
    FakeModelClient, FakeVideoReader, agent_config, build_agent, clip_skim_config,
)
from config.videospy import CONFIG_DIR, load_config, merge_configs
from videospy.tools.clip_refinement import gap_midpoints, refinement_config, refine_clip
from videospy.tools.clip_skim import execute_clip_skim


class ImageReader(FakeVideoReader):
    def __init__(self, num_frames=101, fps=1.0, width=640, height=480):
        super().__init__(num_frames, fps)
        self.width, self.height = width, height
        self.batches = []

    def get_batch(self, indices):
        self.indices = [int(i) for i in indices]
        self.batches.append(self.indices)
        frames = np.zeros((len(indices), self.height, self.width, 3), dtype=np.uint8)
        for frame, index in zip(frames, indices):
            frame[:, :, 0] = int(index) % 256
        return types.SimpleNamespace(asnumpy=lambda: frames)


def image_sizes(call):
    content = call["messages"][0]["content"]
    sizes = []
    for item in content:
        if item.get("type") == "image_url":
            encoded = item["image_url"]["url"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                sizes.append(image.size)
    return sizes


class ClipRefinementTest(unittest.TestCase):
    def inspect(self, settings=None, reader=None, count=16):
        reader = reader or ImageReader()
        config = clip_skim_config(num_frames=count)
        if settings is not None:
            config["refinement"] = settings
        audit = []
        coarse, extra = FakeModelClient([{"content": "coarse"}]), FakeModelClient([{"content": "extra"}])
        with patch("videospy.tools.clip_skim.ModelClient", return_value=coarse), patch(
            "videospy.tools.clip_refinement.ModelClient", return_value=extra
        ):
            parameters = {
                "query": "Describe the operative activity.",
                "original_question": "Which phase is shown? A. First B. Second",
                "start_time": 0, "end_time": (len(reader) - 1) / reader.fps,
                "vr": reader, "subtitles": [], "_clip_refinement_audit": audit,
            }
            result = execute_clip_skim(config, parameters) + refine_clip(config=config, parameters=parameters)
        return result, audit, reader, coarse, extra

    def test_disabled_matches_legacy_request_and_observation(self):
        legacy = self.inspect()
        disabled = self.inspect({"high_resolution": False, "temporal_fill": False})
        self.assertEqual(legacy[0], disabled[0])
        self.assertEqual(legacy[3].calls, disabled[3].calls)
        self.assertEqual(disabled[1], [])
        self.assertEqual(disabled[4].calls, [])

    def test_detail_reuses_exact_frames_and_preserves_native_resolution(self):
        _, audit, reader, coarse, extra = self.inspect({"high_resolution": True})
        event = audit[0]
        detail = event["groups"][0]
        self.assertEqual(len(detail["frames"]), 4)
        self.assertTrue({f["frame_index"] for f in detail["frames"]} <= set(event["base_frame_indices"]))
        self.assertEqual(len(reader.batches), 2)
        self.assertEqual(len(coarse.calls), 1)
        self.assertEqual(image_sizes(extra.calls[0]), [(640, 480)] * 4)
        self.assertTrue(all(f["higher_resolution_than_coarse"] for f in detail["frames"]))
        self.assertEqual(event["status"], "completed")

    def test_temporal_frames_are_new_inside_gaps_at_coarse_resolution(self):
        _, audit, _, _, extra = self.inspect({"temporal_fill": True})
        event = audit[0]
        new = [f["frame_index"] for f in event["groups"][0]["frames"]]
        base = event["base_frame_indices"]
        self.assertEqual(len(new), 4)
        self.assertTrue(set(new).isdisjoint(base))
        self.assertTrue(all(any(a < i < b for a, b in zip(base, base[1:])) for i in new))
        self.assertEqual([min(size) for size in image_sizes(extra.calls[0])], [256] * 4)

    def test_both_modes_share_one_followup_and_keep_frame_groups_separate(self):
        _, audit, _, _, extra = self.inspect({"high_resolution": True, "temporal_fill": True})
        self.assertEqual(len(extra.calls), 1)
        self.assertEqual([g["kind"] for g in audit[0]["groups"]], ["high_resolution", "temporal_fill"])
        self.assertEqual(len(image_sizes(extra.calls[0])), 8)

    def test_control_has_same_detail_frame_ids_but_coarse_size(self):
        high = self.inspect({"high_resolution": True})
        low = self.inspect({"high_resolution": True, "high_resolution_short_side": 256})
        key = lambda result: [f["frame_index"] for f in result[1][0]["groups"][0]["frames"]]
        self.assertEqual(key(high), key(low))
        self.assertEqual([min(size) for size in image_sizes(low[4].calls[0])], [256] * 4)

    def test_detail_never_upscales_low_resolution_source(self):
        _, audit, _, _, extra = self.inspect({"high_resolution": True}, ImageReader(width=64, height=48))
        self.assertEqual(image_sizes(extra.calls[0]), [(64, 48)] * 4)
        self.assertFalse(any(f["higher_resolution_than_coarse"] for f in audit[0]["groups"][0]["frames"]))

    def test_dense_source_skips_temporal_when_no_unseen_frames_exist(self):
        _, audit, _, _, extra = self.inspect({"temporal_fill": True}, ImageReader(num_frames=4), count=4)
        self.assertEqual(audit[0]["status"], "skipped")
        self.assertEqual(extra.calls, [])
        self.assertEqual(gap_midpoints([0, 1, 2, 3], 4), [])

    def test_invalid_settings_are_rejected(self):
        for value in ({"high_resolution": "false"}, {"temporal_frames": 0},
                      {"max_clips_per_question": True}, {"high_resolution_short_side": -1},
                      {"typo": True}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                refinement_config(value)

    def test_followup_failure_preserves_coarse_observation(self):
        config = clip_skim_config()
        config["refinement"] = {"high_resolution": True}
        audit = []
        with patch("videospy.tools.clip_skim.ModelClient") as coarse, patch(
            "videospy.tools.clip_refinement.ModelClient"
        ) as extra:
            coarse.return_value.chat.return_value = {"content": "valid coarse"}
            extra.return_value.chat.side_effect = RuntimeError("mock unavailable")
            parameters = {
                "query": "action", "start_time": 0, "end_time": 100,
                "vr": ImageReader(), "subtitles": [], "_clip_refinement_audit": audit,
            }
            result = execute_clip_skim(config, parameters) + refine_clip(config=config, parameters=parameters)
        self.assertIn("valid coarse", result)
        self.assertEqual(audit[0]["status"], "failed")


class AgentRefinementTest(unittest.TestCase):
    def test_overrides_can_be_composed_and_explicitly_disabled(self):
        detail = load_config(CONFIG_DIR / "refinement_detail.yml")
        temporal = load_config(CONFIG_DIR / "refinement_temporal.yml")
        combined = merge_configs(detail, temporal)
        flags = combined["tools"]["clip_skim"]["refinement"]
        self.assertTrue(flags["high_resolution"] and flags["temporal_fill"])
        disabled = merge_configs(combined, load_config(CONFIG_DIR / "refinement_off.yml"))
        flags = disabled["tools"]["clip_skim"]["refinement"]
        self.assertFalse(flags["high_resolution"] or flags["temporal_fill"])

    def test_run_automatically_supplies_followup_before_agent_can_stop(self):
        config = agent_config()
        config["tools"]["enabled"] = ["clip_skim"]
        config["tools"]["clip_skim"]["refinement"] = {"high_resolution": True}
        call = {"id": "clip", "type": "function", "function": {
            "name": "clip_skim", "arguments": json.dumps({"query": "activity", "start_time": 0, "end_time": 40})}}
        main = FakeModelClient([
            {"role": "assistant", "tool_calls": [call]},
            {"role": "assistant", "content": "Enough evidence."},
            {"role": "assistant", "content": "A"},
        ])
        agent = build_agent(config, main)
        agent.vr = ImageReader()
        coarse = FakeModelClient([{"content": "coarse observation"}])
        extra = FakeModelClient([{"content": "additional visual evidence"}])
        with patch("videospy.tools.clip_skim.ModelClient", return_value=coarse), patch(
            "videospy.tools.clip_refinement.ModelClient", return_value=extra
        ):
            trajectory = agent.run("Which phase? A. First B. Second").to_dict()
        self.assertEqual(trajectory["final_answer"], "A")
        self.assertIn("additional visual evidence", main.calls[1]["messages"][-1]["content"])
        self.assertEqual(trajectory["diagnostics"]["clip_refinement"]["inspections"][0]["status"], "completed")
        self.assertEqual(len(extra.calls), 1)

    def test_per_question_cap_reset_and_all_model_calls_are_auditable(self):
        config = agent_config()
        config["tools"]["enabled"] = ["clip_skim"]
        config["tools"]["clip_skim"]["refinement"] = {"high_resolution": True, "temporal_fill": True}
        agent = build_agent(config, FakeModelClient([]))
        agent.vr = ImageReader()
        agent.question = "Describe activity"
        agent.memory_audit = {}

        def response(**kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message={"content": "observed"}, finish_reason="stop")],
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            )

        def call(start, end):
            return {"id": str(start), "function": {"name": "clip_skim", "arguments":
                    json.dumps({"query": "activity", "start_time": start, "end_time": end})}}

        with patch("videospy.model.completion", side_effect=response):
            first, ok = agent._execute_tool_call(call(0, 40))
            second, ok2 = agent._execute_tool_call(call(50, 90))
        self.assertTrue(ok and ok2)
        self.assertEqual(len(first["model_calls"]), 2)
        self.assertIn("clip_refinement", first)
        self.assertNotIn("model_calls", second)
        self.assertEqual(len(agent.clip_refinement_audit), 1)
        agent.messages.extend([first, second])
        self.assertTrue(all("model_calls" not in m and "clip_refinement" not in m for m in agent._messages_for_model()))
        trajectory = agent._trajectory(2, "stop").to_dict()
        self.assertEqual(len(trajectory["diagnostics"]["clip_refinement"]["inspections"]), 1)
        agent.reset()
        self.assertEqual(agent.clip_refinement_audit, [])


if __name__ == "__main__":
    unittest.main()
