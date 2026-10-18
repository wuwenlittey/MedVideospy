import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import yaml

from experiments.misc.package_videospy import PACKAGE_NAME, ROOT, package


# Exercise the unpacked CLI, tool execution, finalization, and output writing
# without a remote model or a platform-specific video decoder.
INFERENCE_SMOKE = r'''
import json
from pathlib import Path
import sys
import types

import numpy as np

class VideoReader:
    def __init__(self, path):
        pass

    def __len__(self):
        return 16

    def get_avg_fps(self):
        return 4

    def get_batch(self, indices):
        frames = np.zeros((len(indices), 8, 8, 3), dtype=np.uint8)
        frames[:, :, :, 0] = 255
        return types.SimpleNamespace(asnumpy=lambda: frames)

decord = types.ModuleType("decord")
decord.VideoReader = VideoReader
sys.modules["decord"] = decord
calls = []

def completion(**kwargs):
    assert kwargs["model"] == "openai/test-vision"
    assert kwargs["api_base"] == "https://example.invalid/v1"
    assert kwargs["api_key"] == "test-only"
    calls.append(kwargs)
    if len(calls) == 1:
        assert {t["function"]["name"] for t in kwargs["tools"]} == {
            "overview", "clip_skim", "frame_inspect"}
        message = {"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "overview", "arguments": '{"query":"What color is visible?"}'}}]}
    else:
        if len(calls) == 2:
            assert "data:image/jpeg;base64," in json.dumps(kwargs["messages"])
        message = {"role": "assistant", "content": "The video shows a red square."}
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

litellm = types.ModuleType("litellm")
litellm.completion = completion
sys.modules["litellm"] = litellm

from config.videospy import build_config
from videospy.cli import main
from videospy.evolving.cli import parse_args
from videospy.utils import current_commit_id

assert "videoseek" not in sys.modules
assert current_commit_id() == "nogit"
assert main(["--video_path", "synthetic.mp4", "--user_query", "What color is visible?",
             "--output_dir", "results"]) == 0
assert len(calls) == 4, len(calls)
outputs = list(Path("results/videospy/demo").glob("*/prediction.json"))
assert len(outputs) == 1
assert json.loads(outputs[0].read_text())["prediction"] == "The video shows a red square."
trajectory = json.loads(outputs[0].with_name("trajectory.json").read_text())
assert trajectory["total_steps"] == 3
assert any(m.get("role") == "tool" for m in trajectory["messages"])
parse_args(["run", "--benchmark", "lvbench", "--train-manifest", "train.json",
            "--inference-run-dir", "records"])
print("Standalone inference and evolution imports passed.")
'''


class SupplementaryPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="videospy-package-test-")
        cls.archive_path = package(Path(cls.temp.name) / "code.zip")
        with ZipFile(cls.archive_path) as archive:
            archive.extractall(cls.temp.name)
        cls.bundle = Path(cls.temp.name) / PACKAGE_NAME

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_archive_contains_only_source_and_required_support(self):
        names = {p.relative_to(self.bundle).as_posix()
                 for p in self.bundle.rglob("*") if p.is_file()}
        self.assertIn("videospy/evolving/reflector/model.py", names)
        self.assertIn("config/videospy/default.yml", names)
        self.assertIn("benchmark/sampling.py", names)
        self.assertEqual((self.bundle / "LICENSE").read_bytes(), (ROOT / "LICENSE").read_bytes())
        self.assertEqual((self.bundle / "README.md").read_bytes(),
                         (self.bundle / "videospy/README.md").read_bytes())
        self.assertFalse(any(name.startswith(("data/", "output/", "videoseek/", ".git/"))
                             or "__pycache__" in name for name in names))
        self.assertNotIn("videoseek", (self.bundle / "config/__init__.py").read_text().lower())

    def test_private_connection_values_are_absent_from_archive(self):
        original = yaml.safe_load((ROOT / "config/videospy/default.yml").read_text())
        exported = yaml.safe_load((self.bundle / "config/videospy/default.yml").read_text())
        with ZipFile(self.archive_path) as archive:
            payload = b"\n".join(archive.read(name) for name in archive.namelist())

        def check(source, clean):
            if not isinstance(source, dict):
                return
            for key, value in source.items():
                if key in {"api_key", "api_base"}:
                    self.assertTrue(clean[key].startswith("VIDEOSPY_"))
                    if value:
                        self.assertFalse(str(value).encode() in payload,
                                         "An original connection value leaked into the archive.")
                else:
                    check(value, clean[key])

        check(original, exported)
        self.assertEqual(original["SYSTEM_PROMPT"], exported["SYSTEM_PROMPT"])
        self.assertEqual(original["agent"]["max_steps"], exported["agent"]["max_steps"])
        for tool, field in (("overview", "num_frames"), ("clip_skim", "num_frames"),
                            ("frame_inspect", "max_frames")):
            self.assertEqual(original["tools"][tool][field], exported["tools"][tool][field])

    def test_unpacked_cli_runs_without_repository_or_external_service(self):
        env = dict(os.environ, VIDEOSPY_MODEL="openai/test-vision",
                   VIDEOSPY_API_BASE="https://example.invalid/v1", VIDEOSPY_API_KEY="test-only")
        result = subprocess.run([sys.executable, "-c", INFERENCE_SMOKE],
                                cwd=self.bundle, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
