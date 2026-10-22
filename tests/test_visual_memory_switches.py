"""Online memory is a single-call on/off feature; the old reviewer is removed."""

import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tests import test_visual_memory_integration as fixtures
from tests.test_videospy import FakeModelClient, clip_skim_config, model_config
from videospy.tools.clip_skim import execute_clip_skim
from videospy.tools.frame_inspect import execute_frame_inspect
from videospy.visual_memory.integration import VisualMemorySession, memory_config


class SwitchTests(fixtures.VisualMemoryFixture):
    def test_removed_review_modes_and_configuration_fail_with_migration_message(self):
        for settings in ({"usage_mode": "review"}, {"usage_mode": "current_only_review"}, {"review": {}}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "removed"):
                memory_config(settings)
        self.assertFalse(hasattr(VisualMemorySession, "review"))

    def test_on_off_each_uses_one_model_call_for_both_local_tools(self):
        for enabled in (True, False):
            for frame in (True, False):
                with self.subTest(enabled=enabled, frame=frame):
                    client = FakeModelClient([{"content": "Local observation."}])
                    session = self.session(enabled=enabled, mode="image_text")
                    parameters = {**self.parameters, "vr": fixtures.ColorReader(), "subtitles": [],
                                  "start_time": 0, "end_time": 4, "timestamps": [1.0],
                                  "_visual_memory_session": session}
                    with patch("videospy.visual_memory.integration.EmbeddingClient", fixtures.PixelEncoder), patch(
                        "videospy.tools.clip_skim.ModelClient", return_value=client), patch(
                        "videospy.tools.frame_inspect.ModelClient", return_value=client):
                        if frame:
                            execute_frame_inspect({"max_frames": 1, "model": model_config("inspect")}, parameters)
                        else:
                            execute_clip_skim(clip_skim_config(num_frames=2), parameters)
                    self.assertEqual(len(client.calls), 1)
                    self.assertEqual(bool(session.audit), enabled)

    def test_off_preset_overrides_on_without_loading_index_or_embedding(self):
        root = Path(__file__).resolve().parents[1] / "config/videospy"
        on = yaml.safe_load((root / "visual_facts_supervised.yml").read_text())["tools"]["visual_memory"]
        off = yaml.safe_load((root / "visual_facts_supervised_off.yml").read_text())["tools"]["visual_memory"]
        session = VisualMemorySession({**on, **off})
        with patch.object(session, "_load", side_effect=AssertionError("disabled must be lazy")):
            self.assertEqual(self.augment(session), self.content)
        self.assertEqual(session.audit, [])


if __name__ == "__main__":
    unittest.main()
