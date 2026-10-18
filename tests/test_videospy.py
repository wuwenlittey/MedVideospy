import copy
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np


try:
    import decord  # noqa: F401
except ModuleNotFoundError:
    decord_stub = types.ModuleType("decord")
    decord_stub.VideoReader = object
    sys.modules["decord"] = decord_stub

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.completion = lambda **_: None
    sys.modules["litellm"] = litellm_stub


from config.videospy import default_config, load_config
from videospy import cli as videospy_cli
from videospy import metrics
from videospy.agent import VideoSpyAgent, VisualEvidenceRequiredError
from videospy.model import ModelClient, collect_model_calls
from videospy.tools import DEFAULT_TOOL_REGISTRY
from videospy.tools.clip_skim import execute_clip_skim
from videospy.tools.frame_inspect import execute_frame_inspect
from videospy.tools.overview import execute_overview
from videospy.tools.registry import ToolRegistry
from videospy.utils import (
    parse_video_timestamp,
    sampled_subtitles_text,
    select_subtitles_in_range,
)


class FakeVideoReader:
    def __init__(self, num_frames=100, fps=10.0):
        self.num_frames = num_frames
        self.fps = fps
        self.indices = []

    def __len__(self):
        return self.num_frames

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        self.indices = [int(index) for index in indices]
        frames = np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)
        return types.SimpleNamespace(asnumpy=lambda: frames)


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        return copy.deepcopy(self.responses.pop(0))


def model_config(name):
    return {
        "model_name": name,
        "api_base": "https://example.test/v1",
        "api_key": "TEST_API_KEY",
        "api_version": None,
        "seed": 42,
        "temperature": 0.0,
        "max_retries": 0,
        "max_completion_tokens": 1024,
    }


def clip_skim_config(num_frames=16, max_duration=None, subtitle_context_seconds=0):
    return {
        "num_frames": num_frames,
        "max_duration": max_duration,
        "subtitle_context_seconds": subtitle_context_seconds,
        "model": model_config("clip-skim-model"),
    }


def agent_config(max_steps=3):
    return {
        "SYSTEM_PROMPT": (
            "overview={overview_num_frames}, skim={clip_skim_num_frames}"
        ),
        "agent": {"max_steps": max_steps, "model": model_config("agent-model")},
        "tools": {
            "enabled": ["overview"],
            "overview": {"num_frames": 8, "model": model_config("overview-model")},
            "clip_skim": clip_skim_config(num_frames=4),
            "frame_inspect": {
                "max_frames": 4,
                "model": model_config("frame-inspect-model"),
            },
        },
    }


def build_agent(config, model_client, verbose=False, tools=None):
    with patch("videospy.agent.ModelClient", return_value=model_client) as client_class:
        agent = VideoSpyAgent(
            config=config,
            video_path="video.frames",
            subtitle_path=None,
            output_dir="output",
            tools=config["tools"]["enabled"] if tools is None else tools,
            verbose=verbose,
            video_reader=FakeVideoReader(),
        )
    client_class.assert_called_once_with(config["agent"]["model"])
    return agent


class DemoRunDirectoryNameTest(unittest.TestCase):
    def test_videospy_demo_run_dir_appends_commit_id(self):
        config = {"tools": {"enabled": []}}
        trajectory = types.SimpleNamespace(to_dict=lambda: {"final_answer": "A"})
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    videospy_cli, "build_agent_config", return_value=config
                ),
                patch.object(
                    videospy_cli,
                    "append_commit_id",
                    side_effect=lambda name: f"{name}_abc1234",
                ) as append,
                patch.object(videospy_cli, "VideoSpyAgent") as agent_class,
            ):
                agent_class.return_value.run.return_value = trajectory
                videospy_cli.main(
                    [
                        "--video_path",
                        "video.mp4",
                        "--user_query",
                        "Question?",
                        "--output_dir",
                        directory,
                    ]
                )

            run_dir = Path(agent_class.call_args.kwargs["output_dir"])
            self.assertEqual(run_dir.parent.name, "demo")
            self.assertTrue(run_dir.name.endswith("_abc1234"))
            self.assertTrue(append.call_args.args[0].startswith("video_"))


class VideoSpyConfigTest(unittest.TestCase):
    def test_enabled_tools_cli_overrides_yaml_in_all_inference_entrypoints(self):
        from benchmark.egolife import run as egolife
        from benchmark.lvbench import run as lvbench
        from benchmark.medvidbench import run as medvidbench

        entrypoints = [
            (videospy_cli, ["--video_path", "video.mp4", "--user_query", "Question?"]),
            (lvbench, ["--agent", "videospy"]),
            (medvidbench, ["--agent", "videospy"]),
            (egolife, ["--agent", "videospy"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.yml"
            path.write_text("tools:\n  enabled: [clip_skim]\n")
            for entrypoint, base_args in entrypoints:
                for names in (None, [], ["overview"], ["overview", "frame_inspect"]):
                    with self.subTest(entrypoint=entrypoint.__name__, names=names):
                        argv = [*base_args, "--config", str(path)]
                        if names is not None:
                            argv.extend(["--enabled-tools", *names])
                        config = entrypoint.build_agent_config(entrypoint.parse_args(argv))
                        self.assertEqual(
                            config["tools"]["enabled"],
                            ["clip_skim"] if names is None else names,
                        )
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    entrypoint.parse_args([*base_args, "--enabled-tools", "invalid_tool"])
                self.assertEqual(error.exception.code, 2)

    def test_enabled_tools_cli_rejects_videoseek(self):
        from benchmark.egolife import run as egolife
        from benchmark.lvbench import run as lvbench
        from benchmark.medvidbench import run as medvidbench

        for entrypoint in (lvbench, medvidbench, egolife):
            for names in ([], ["overview"]):
                with self.subTest(entrypoint=entrypoint.__name__, names=names):
                    args = entrypoint.parse_args(["--agent", "videoseek", "--enabled-tools", *names])
                    with self.assertRaisesRegex(ValueError, "only supported with --agent videospy"):
                        entrypoint.build_agent_config(args)

    def test_enabled_tool_override_replaces_the_list_including_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.yml"
            for enabled in (["overview"], []):
                with self.subTest(enabled=enabled):
                    path.write_text(json.dumps({"tools": {"enabled": enabled}}))
                    args = videospy_cli.parse_args([
                        "--video_path", "video.mp4", "--user_query", "Question?",
                        "--config", str(path),
                    ])
                    config = videospy_cli.build_agent_config(args)
                    self.assertEqual(config["tools"]["enabled"], enabled)
                    agent = build_agent(config, FakeModelClient([]))
                    self.assertEqual(list(agent.enabled_tools), enabled)

    def test_completion_token_cli_override(self):
        args = videospy_cli.parse_args(
            [
                "--video_path",
                "video.mp4",
                "--user_query",
                "Question?",
                "--max_completion_tokens",
                "123",
            ]
        )
        config = videospy_cli.build_agent_config(args)

        self.assertEqual(config["agent"]["model"]["max_completion_tokens"], 123)

    def test_multiple_config_overrides_are_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "general.yml"
            second_config_path = root / "override.yml"
            config_path.write_text(
                "agent:\n"
                "  max_steps: 4\n"
                "  model:\n"
                "    temperature: 0.25\n"
                "    max_completion_tokens: 777\n"
                "tools:\n"
                "  overview:\n"
                "    num_frames: 12\n",
                encoding="utf-8",
            )
            second_config_path.write_text(
                "SYSTEM_PROMPT: custom prompt\n",
                encoding="utf-8",
            )
            args = videospy_cli.parse_args(
                [
                    "--video_path",
                    "video.mp4",
                    "--user_query",
                    "Question?",
                    "--config",
                    str(config_path),
                    "--config",
                    str(second_config_path),
                    "--max_steps",
                    "7",
                ]
            )

            config = videospy_cli.build_agent_config(args)

            self.assertEqual(config["SYSTEM_PROMPT"], "custom prompt")
            self.assertEqual(config["agent"]["max_steps"], 7)
            self.assertEqual(config["agent"]["model"]["temperature"], 0.25)
            self.assertEqual(
                config["agent"]["model"]["max_completion_tokens"], 777
            )
            self.assertIn("model_name", config["agent"]["model"])
            self.assertEqual(config["tools"]["overview"]["num_frames"], 12)
            self.assertIn("clip_skim", config["tools"])
            self.assertIn("frame_inspect", config["tools"])
            self.assertEqual(config["tools"]["frame_inspect"]["max_frames"], 4)
            self.assertNotIn("inspect", config["tools"])

    def test_default_config_does_not_implicitly_load_local_yaml(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "config/videospy/default.yml"
        )

        self.assertEqual(default_config, load_config(config_path))

    def test_visual_tools_default_to_sampled_frame_subtitles(self):
        self.assertEqual(
            default_config["tools"]["model_call_input_mode"], "compact"
        )
        for tool_name, context_seconds in (
            ("overview", 60), ("clip_skim", 120), ("frame_inspect", 30)
        ):
            with self.subTest(tool=tool_name):
                tool_config = default_config["tools"][tool_name]
                self.assertTrue(tool_config["use_subtitles"])
                self.assertEqual(tool_config["subtitle_context_seconds"], context_seconds)

    def test_default_prompt_is_unchanged_without_override(self):
        args = videospy_cli.parse_args(
            ["--video_path", "video.mp4", "--user_query", "Question?"]
        )

        config = videospy_cli.build_agent_config(args)

        self.assertEqual(
            config["SYSTEM_PROMPT"],
            default_config["SYSTEM_PROMPT"],
        )


class VideoSpyAgentTest(unittest.TestCase):
    def test_visual_tools_receive_original_task_and_return_actual_coverage(self):
        question = "Which object does the woman place on the table? (A) cup (B) plate"
        observation = "Facts: a round object. Interpretation: possibly a plate. Unknowns: identity."
        cases = [
            ("overview", {}, "8", "0.000s - 9.900s", "1.500s"),
            ("clip_skim", {"start_time": 0, "end_time": 25}, "4", "0.000s - 9.900s", "3.300s"),
            ("frame_inspect", {"timestamps": [0.26, 0.29, 9.999]}, "2", "0.300s - 9.900s", "9.600s"),
        ]
        for name, arguments, count, span, gap in cases:
            with self.subTest(tool=name):
                arguments = {"query": "Describe the held object.", **arguments}
                client = FakeModelClient([
                    {"tool_calls": [{"id": "inspect-1", "function": {
                        "name": name, "arguments": json.dumps(arguments)}}]},
                    {"content": "The object is possibly a plate; identity remains uncertain."},
                    {"content": "B"},
                ])
                agent = build_agent(agent_config(), client, tools=[name])
                visual = FakeModelClient([{"content": observation}])
                with patch(f"videospy.tools.{name}.ModelClient", return_value=visual):
                    trajectory = agent.run(question)
                self.assertEqual(len(visual.calls), 1)
                self.assertEqual(len(client.calls), 3)
                texts = "\n".join(item["text"] for item in visual.calls[0]["messages"][0]["content"]
                                  if item["type"] == "text")
                self.assertIn(f"Original task (read-only context):\n{question}", texts)
                self.assertIn("Current inspection goal:\nDescribe the held object.", texts)
                self.assertIn("Facts (", texts)
                self.assertIn("Interpretation (", texts)
                self.assertIn("Unknowns (", texts)
                tool_message = next(m for m in trajectory.messages if m["role"] == "tool")
                for text in [f"Actual sampled frames: {count};", f"timestamp range: {span}",
                             f"maximum adjacent sample gap: {gap}"]:
                    self.assertIn(text, texts)
                    self.assertIn(text, tool_message["content"])
                self.assertTrue(tool_message["content"].endswith(observation))
                self.assertEqual(tool_message["tool_call_id"], "inspect-1")
                self.assertIn(tool_message["content"], [m.get("content") for m in client.calls[-1]["messages"]])
                self.assertIn("distinguishing details", client.calls[-1]["messages"][-1]["content"])
                self.assertNotIn("original_question", agent.tools[0]["function"]["parameters"]["properties"])
                if name == "frame_inspect":
                    from videospy.core.evidence import frame_timestamps
                    self.assertEqual(frame_timestamps(tool_message), [0.3, 9.9])

    def test_default_prompt_defines_output_and_inspection_rules(self):
        prompt = build_agent(
            default_config, FakeModelClient([]),
            tools=["overview", "clip_skim", "frame_inspect"],
        ).messages[0]["content"]

        self.assertIn("NEVER repeat a tool request", prompt)
        self.assertIn("unresolved difference between plausible options", prompt)
        self.assertIn("Preserve uncertainty in subsequent summaries", prompt)
        self.assertIn("Treat incompatible observations", prompt)
        self.assertIn("interpretations may be uncertain", prompt)
        self.assertNotIn("clip_skim", prompt)
        self.assertNotIn("frame_inspect", prompt)
        self.assertNotIn("# Toolkit", prompt)
        self.assertNotIn("`inspect`", prompt)
        self.assertIn("return exactly one uppercase option letter", prompt)
        self.assertIn("separate finalization request", prompt)
        self.assertIn("preserve the original temporal relation", prompt)
        self.assertIn("non-detection cannot by itself negate", prompt)

    def test_enabled_tools_control_schema_and_prompt(self):
        tool_names = ["overview", "clip_skim", "frame_inspect"]
        for enabled in (
            [], ["overview"], ["clip_skim"], ["frame_inspect"],
            ["overview", "clip_skim"], ["overview", "frame_inspect"],
            ["clip_skim", "frame_inspect"], tool_names,
        ):
            with self.subTest(enabled=enabled):
                config = copy.deepcopy(default_config)
                config["tools"]["enabled"] = enabled
                for name in tool_names:
                    if name not in enabled:
                        del config["tools"][name]
                agent = build_agent(config, FakeModelClient([]))
                self.assertEqual(
                    [tool["function"]["name"] for tool in agent.tools], enabled
                )
                for name in tool_names:
                    self.assertNotIn(name, agent.messages[0]["content"])
                    if name in enabled:
                        description = next(
                            t["function"]["description"] for t in agent.tools
                            if t["function"]["name"] == name
                        )
                        self.assertNotIn("{num_frames}", description)
                self.assertNotIn("{tool_instructions}", agent.messages[0]["content"])
                self.assertNotIn("{frame_inspect_max_frames}", agent.messages[0]["content"])
                agent.reset()
                self.assertIn("Only call tools provided in this request", agent.messages[0]["content"])

    def test_disabled_tools_are_rejected_before_execution_and_do_not_count_as_evidence(self):
        def call(name, arguments):
            return {"id": name, "function": {"name": name, "arguments": arguments}}

        client = FakeModelClient([
            {"tool_calls": [
                call("frame_inspect", '{"query":"detail","timestamps":"[1]"}'),
                call("clip_skim", '{"query":"action","start_time":0,"end_time":1}'),
            ]},
            {"tool_calls": [call("overview", '{"query":"scene"}')]},
            {"content": "The scene supports B."},
            {"content": "B"},
        ])
        agent = build_agent(agent_config(), client)
        with patch.object(agent.tool_registry, "get_function") as get_function:
            get_function.return_value.return_value = "A blue scene."
            result = agent.run("Which color?").to_dict()

        get_function.assert_called_once_with("overview")
        sent_tools = client.calls[0]["tools"]
        self.assertEqual([t["function"]["name"] for t in sent_tools], ["overview"])
        self.assertIn("8 frames", sent_tools[0]["function"]["description"])
        self.assertEqual(result["final_answer"], "B")
        observations = [m for m in result["messages"] if m["role"] == "tool"]
        for observation in observations[:2]:
            self.assertIn("not enabled", observation["content"])
            self.assertIn("Enabled tools: overview", observation["content"])
        self.assertEqual(client.calls[1]["tool_choice"], "required")
        self.assertIsNone(client.calls[2]["tool_choice"])
        self.assertEqual(result["diagnostics"]["visual_calls"]["failed"], 2)
        self.assertEqual(agent.vr.indices, [])

    def test_custom_prompt_cannot_enable_a_disabled_tool(self):
        config = agent_config()
        config["SYSTEM_PROMPT"] = "Always call frame_inspect before answering."
        agent = build_agent(config, FakeModelClient([]))
        self.assertIn("Only call tools provided in this request", agent.messages[0]["content"])
        with patch.object(agent.tool_registry, "get_function") as get_function:
            message, succeeded = agent._execute_tool_call({
                "id": "disabled", "function": {
                    "name": "frame_inspect",
                    "arguments": '{"query":"detail","timestamps":[1]}',
                },
            })
        self.assertFalse(succeeded)
        self.assertIn("not enabled", message["content"])
        get_function.assert_not_called()

    def test_empty_enabled_tools_does_not_fall_back_to_registered_tools(self):
        config = agent_config()
        config["tools"]["enabled"] = []
        client = FakeModelClient([])
        agent = build_agent(config, client)
        message, succeeded = agent._execute_tool_call({
            "id": "disabled", "function": {"name": "overview", "arguments": "{}"},
        })
        self.assertFalse(succeeded)
        self.assertIn("not enabled", message["content"])
        with self.assertRaises(VisualEvidenceRequiredError):
            agent.run("Which color?")
        self.assertEqual(client.calls, [])

    def test_disabled_frame_inspect_does_not_validate_its_limit(self):
        config = agent_config()
        config["tools"]["frame_inspect"]["max_frames"] = 0
        build_agent(config, FakeModelClient([]))
        with self.assertRaisesRegex(ValueError, "max_frames must be a positive integer"):
            build_agent(config, FakeModelClient([]), tools=["frame_inspect"])

    def test_answer_uses_complete_message_trajectory(self):
        summary = "Subtitle at 0s supports the observed color: option B, Blue."
        client = FakeModelClient(
            [
                {"role": "assistant", "content": summary},
                {"role": "assistant", "content": "B"},
            ]
        )
        agent = build_agent(agent_config(), client)
        agent.subtitles = [
            {"start_time": 0, "end_time": 1, "subtitle": "The color is blue."}
        ]

        result = agent.run("Which color? A: White; B: Blue").to_dict()

        self.assertEqual(result["final_answer"], "B")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(client.calls[0]["tool_choice"], "auto")
        self.assertEqual(client.calls[1]["tools"], [])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["messages"][-2]["content"], summary)
        self.assertEqual(result["messages"][-3]["content"], summary)
        self.assertIn("first match the factual evidence to the meaning of an option", client.calls[1]["messages"][-1]["content"])
        self.assertIn(
            "Return only the requested answer",
            client.calls[1]["messages"][-1]["content"],
        )

    def test_no_visual_evidence_requires_a_structured_tool_call(self):
        client = FakeModelClient([{"role": "assistant", "content": "B"}])
        agent = build_agent(agent_config(), client)

        with self.assertRaisesRegex(
            VisualEvidenceRequiredError,
            "no structured visual tool call",
        ):
            agent.run("Which option?")

        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_hidden_subtitles_still_require_a_visual_tool_call(self):
        config = agent_config()
        config["agent"]["include_subtitles_in_question"] = False
        client = FakeModelClient([{"role": "assistant", "content": "B"}])
        agent = build_agent(config, client)
        agent.subtitles = [
            {"start_time": 0, "end_time": 1, "subtitle": "Option B is shown."}
        ]

        with self.assertRaisesRegex(
            VisualEvidenceRequiredError,
            "no structured visual tool call",
        ):
            agent.run("Which option?")

        self.assertEqual(client.calls[0]["tool_choice"], "required")
        self.assertNotIn("Option B is shown", client.calls[0]["messages"][1]["content"])

    def test_reasoning_content_is_saved_and_printed(self):
        client = FakeModelClient(
            [
                {
                    "role": "assistant",
                    "reasoning_content": "I should inspect the evidence.",
                    "content": "B",
                },
                {"role": "assistant", "content": "B"},
            ]
        )
        agent = build_agent(agent_config(), client, verbose=True)
        agent.subtitles = [
            {"start_time": 0, "end_time": 1, "subtitle": "Evidence."}
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.run("Which option?").to_dict()

        self.assertEqual(
            result["messages"][2]["reasoning_content"],
            "I should inspect the evidence.",
        )
        self.assertIn("## == REASONING CONTENT ==", output.getvalue())
        self.assertIn("I should inspect the evidence.", output.getvalue())

    def test_tool_call_and_observation_remain_standard_messages(self):
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "overview", "arguments": "{}"},
        }
        client = FakeModelClient(
            [
                {"role": "assistant", "content": "Checking.", "tool_calls": [tool_call]},
                {"role": "assistant", "content": "The evidence supports C."},
                {"role": "assistant", "content": "C"},
            ]
        )
        agent = build_agent(agent_config(), client)
        agent.tool_registry = ToolRegistry(
            tools={"overview": agent.tools[0]},
            tool_functions={"overview": lambda **_: "Evidence at 1.0s"},
        )

        result = agent.run("What happens?").to_dict()
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(client.calls[0]["tool_choice"], "required")
        self.assertIsNone(client.calls[1]["tool_choice"])
        self.assertEqual(client.calls[2]["tools"], [])
        self.assertEqual(result["messages"][3]["role"], "tool")
        self.assertEqual(result["messages"][3]["tool_call_id"], "call-1")
        self.assertEqual(result["messages"][3]["content"], "Evidence at 1.0s")

    def test_tool_message_saves_model_input_and_output(self):
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "overview", "arguments": "{}"},
        }
        client = FakeModelClient(
            [
                {"role": "assistant", "content": "Checking.", "tool_calls": [tool_call]},
                {"role": "assistant", "content": "The evidence supports C."},
                {"role": "assistant", "content": "C"},
            ]
        )
        agent = build_agent(agent_config(), client)
        image_url = "data:image/jpeg;base64,QUJDRA=="

        def execute_tool(**_):
            response = ModelClient(model_config("tool-model")).chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        ],
                    }
                ]
            )
            return response["content"]

        agent.tool_registry = ToolRegistry(
            tools={"overview": agent.tools[0]},
            tool_functions={"overview": execute_tool},
        )
        tool_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message={
                        "role": "assistant",
                        "reasoning_content": "Visual reasoning.",
                        "content": "A person is visible.",
                    }
                )
            ],
            usage=None,
        )

        with patch(
            "videospy.model.completion", return_value=tool_response
        ) as completion:
            result = agent.run("What happens?").to_dict()

        model_call = result["messages"][3]["model_call"]
        recorded_url = model_call["input"]["messages"][0]["content"][0][
            "image_url"
        ]["url"]
        sent_url = completion.call_args.kwargs["messages"][0]["content"][0][
            "image_url"
        ]["url"]
        self.assertEqual(
            recorded_url,
            "data:image/jpeg;base64,<omitted 8 base64 characters>",
        )
        self.assertEqual(sent_url, image_url)
        self.assertEqual(model_call["input"]["model"], "tool-model")
        self.assertEqual(
            model_call["output"]["reasoning_content"], "Visual reasoning."
        )
        self.assertEqual(model_call["output"]["content"], "A person is visible.")
        self.assertNotIn("model_call", client.calls[1]["messages"][3])
        self.assertEqual(
            client.calls[1]["messages"][3]["content"], "A person is visible."
        )

    def test_model_call_input_can_be_saved_in_full(self):
        image_url = "data:image/jpeg;base64,QUJDRA=="
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message={"content": "description"})],
            usage=None,
        )

        with (
            patch("videospy.model.completion", return_value=response),
            collect_model_calls("full") as model_calls,
        ):
            ModelClient(model_config("tool-model")).chat(
                [{"role": "user", "content": image_url}]
            )

        self.assertEqual(
            model_calls[0]["input"]["messages"][0]["content"], image_url
        )

    def test_invalid_tool_model_call_input_mode_is_rejected(self):
        config = agent_config()
        config["tools"]["model_call_input_mode"] = "invalid"

        with self.assertRaisesRegex(ValueError, "model_call_input_mode"):
            build_agent(config, FakeModelClient([]))

    def test_max_steps_forces_a_tool_free_final_response(self):
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "overview", "arguments": "{}"},
        }
        client = FakeModelClient(
            [
                {"role": "assistant", "content": "Checking.", "tool_calls": [tool_call]},
                {"role": "assistant", "content": "B"},
            ]
        )
        agent = build_agent(agent_config(max_steps=1), client)
        agent.tool_registry = ToolRegistry(
            tools={"overview": agent.tools[0]},
            tool_functions={"overview": lambda **_: "Partial evidence"},
        )

        result = agent.run("Which color? A: White; B: Blue. Return only the option letter.").to_dict()
        self.assertEqual(result["finish_reason"], "reach_max_steps")
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(agent.last_run_rounds, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["final_answer"], "B")
        self.assertEqual(client.calls[-1]["tools"], [])
        self.assertEqual(client.calls[-1]["messages"][-2]["content"], "Partial evidence")
        final_request = client.calls[-1]["messages"][-1]["content"]
        self.assertIn("Return only the requested answer", final_request)
        self.assertIn("Do not copy a previously proposed letter", final_request)

    def test_budget_decreases_after_failed_and_duplicate_inspections(self):
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "overview", "arguments": "{}"},
        }
        invalid_call = copy.deepcopy(tool_call)
        invalid_call["id"] = "call-2"
        invalid_call["function"]["arguments"] = "not-json"
        duplicate_call = copy.deepcopy(tool_call)
        duplicate_call["id"] = "call-3"
        client = FakeModelClient([
            {"role": "assistant", "tool_calls": [call]}
            for call in (tool_call, invalid_call, duplicate_call)
        ] + [{"role": "assistant", "content": "B"}])
        agent = build_agent(agent_config(max_steps=3), client)
        agent.tool_registry = ToolRegistry(
            tools={"overview": agent.tools[0]},
            tool_functions={"overview": lambda **_: "Partial evidence"},
        )

        result = agent.run("Which option?").to_dict()

        for call, remaining in zip(client.calls[:-1], (3, 2, 1)):
            prompt = call["messages"][-1]["content"]
            self.assertIn(f"{remaining}/3 rounds remain, including the next round", prompt)
            self.assertEqual(str(call["messages"]).count("Evidence-gathering budget:"), 1)
        self.assertNotIn("Evidence-gathering budget:", str(client.calls[-1]["messages"]))
        self.assertNotIn("Evidence-gathering budget:", str(result["messages"]))
        observations = [m["content"] for m in result["messages"] if m["role"] == "tool"]
        self.assertIn("execution failed", observations[1])
        self.assertTrue(observations[2].startswith("Duplicate `"))
        self.assertEqual(result["finish_reason"], "reach_max_steps")
        self.assertEqual(result["final_answer"], "B")

    def test_invalid_arguments_become_a_tool_observation(self):
        tool_call = {
            "id": "bad-call",
            "type": "function",
            "function": {"name": "overview", "arguments": "not-json"},
        }
        client = FakeModelClient(
            [{"role": "assistant", "content": "Checking.", "tool_calls": [tool_call]}]
        )
        agent = build_agent(agent_config(max_steps=1), client)

        with self.assertRaisesRegex(
            VisualEvidenceRequiredError,
            "without collecting any successful visual evidence",
        ):
            agent.run("What happens?")

        self.assertIn("JSONDecodeError", agent.messages[3]["content"])

    def test_clip_duplicate_requires_the_same_normalized_query(self):
        client = FakeModelClient([])
        agent = build_agent(agent_config(), client, tools=["clip_skim"])
        executions = []

        def execute_clip(**_):
            executions.append("clip")
            return "clip evidence"

        agent.tool_registry = ToolRegistry(
            tools={
                "clip_skim": DEFAULT_TOOL_REGISTRY.get_tool("clip_skim")
            },
            tool_functions={"clip_skim": execute_clip},
        )
        first = {
            "id": "clip-1",
            "function": {
                "name": "clip_skim",
                "arguments": '{"query":"first goal","start_time":1,"end_time":4}',
            },
        }
        duplicate = {
            "id": "clip-2",
            "function": {
                "name": "clip_skim",
                "arguments": '{"query":"  FIRST   goal ","start_time":"00:01","end_time":4}',
            },
        }
        different_query = {
            "id": "clip-3",
            "function": {
                "name": "clip_skim",
                "arguments": '{"query":"different wording","start_time":1,"end_time":4}',
            },
        }

        first_message, first_succeeded = agent._execute_tool_call(first)
        duplicate_message, duplicate_succeeded = agent._execute_tool_call(duplicate)
        different_message, different_succeeded = agent._execute_tool_call(
            different_query
        )

        self.assertTrue(first_succeeded)
        self.assertFalse(duplicate_succeeded)
        self.assertTrue(different_succeeded)
        self.assertEqual(executions, ["clip", "clip"])
        self.assertEqual(first_message["content"], "clip evidence")
        self.assertEqual(different_message["content"], "clip evidence")
        self.assertIn(
            "Duplicate `clip_skim` inspection skipped",
            duplicate_message["content"],
        )

    def test_overview_duplicate_requires_the_same_normalized_query(self):
        client = FakeModelClient([])
        config = agent_config()
        config["tools"]["overview"]["once_per_task"] = False
        agent = build_agent(config, client)
        executions = []

        def execute_overview(**_):
            executions.append("overview")
            return "overview evidence"

        agent.tool_registry = ToolRegistry(
            tools={"overview": DEFAULT_TOOL_REGISTRY.get_tool("overview")},
            tool_functions={"overview": execute_overview},
        )

        def call(identifier, query):
            return agent._execute_tool_call(
                {
                    "id": identifier,
                    "function": {
                        "name": "overview",
                        "arguments": json.dumps({"query": query}),
                    },
                }
            )

        self.assertTrue(call("overview-1", "Find the main action.")[1])
        self.assertFalse(call("overview-2", " find THE main   action. ")[1])
        self.assertTrue(call("overview-3", "Identify the people.")[1])
        self.assertEqual(executions, ["overview", "overview"])

    def test_overview_is_removed_from_requests_and_restored_for_next_task(self):
        client = FakeModelClient([
            {"tool_calls": [{"id": "overview-1", "function": {
                "name": "overview", "arguments": '{"query":"scene"}',
            }}]},
            {"content": "Evidence supports B."},
            {"content": "B"},
        ] * 2)
        agent = build_agent(agent_config(), client, tools=["overview", "clip_skim"])
        with patch.object(agent.tool_registry, "get_function") as get_function:
            get_function.return_value.return_value = "Scene evidence."
            agent.run("First question?")
            agent.run("Second question?")

        self.assertEqual(get_function.call_count, 2)
        for start in (0, 3):
            self.assertEqual(
                [t["function"]["name"] for t in client.calls[start]["tools"]],
                ["overview", "clip_skim"],
            )
            self.assertEqual(
                [t["function"]["name"] for t in client.calls[start + 1]["tools"]],
                ["clip_skim"],
            )
            self.assertEqual(client.calls[start + 2]["tools"], [])

    def test_overview_limit_guards_execution_even_after_failure(self):
        for once in (True, False):
            for fails in (True, False):
                with self.subTest(once=once, fails=fails):
                    config = agent_config()
                    config["tools"]["overview"]["once_per_task"] = once
                    agent = build_agent(config, FakeModelClient([]))
                    with patch.object(agent.tool_registry, "get_function") as get_function:
                        execute = get_function.return_value
                        execute.return_value = "Scene evidence."
                        if fails:
                            execute.side_effect = RuntimeError("Unavailable")
                        for query in ("scene", "different detail"):
                            message, _ = agent._execute_tool_call({
                                "id": query, "function": {
                                    "name": "overview",
                                    "arguments": json.dumps({"query": query}),
                                },
                            })
                    self.assertEqual(execute.call_count, 1 if once else 2)
                    self.assertEqual(bool(agent._available_tools()), not once)
                    if once:
                        self.assertIn("unavailable for this task", message["content"])
                    agent.reset()
                    self.assertEqual(len(agent._available_tools()), 1)

    def test_frame_duplicate_allows_a_materially_different_query(self):
        client = FakeModelClient([])
        agent = build_agent(agent_config(), client, tools=["frame_inspect"])
        executions = []

        def execute_frame(**_):
            executions.append("frame")
            return "frame evidence"

        agent.tool_registry = ToolRegistry(
            tools={
                "frame_inspect": DEFAULT_TOOL_REGISTRY.get_tool("frame_inspect")
            },
            tool_functions={"frame_inspect": execute_frame},
        )

        def call(identifier, query, timestamps=(2, 3)):
            return agent._execute_tool_call(
                {
                    "id": identifier,
                    "function": {
                        "name": "frame_inspect",
                        "arguments": json.dumps(
                            {"query": query, "timestamps": timestamps}
                        ),
                    },
                }
            )

        self.assertTrue(call("frame-1", "Check the tool tip.")[1])
        self.assertFalse(
            call("frame-2", "  check THE tool   tip. ", timestamps=(3, 2))[1]
        )
        self.assertTrue(call("frame-3", "Read the visible label.")[1])
        self.assertEqual(executions, ["frame", "frame"])

    def test_frame_inspect_arrays_execute_and_deduplicate(self):
        for timestamps in ([2399.3], [2399.3, 2400.0], ["39:59.3", "40:00"]):
            with self.subTest(timestamps=timestamps):
                agent = build_agent(agent_config(), FakeModelClient([]), tools=["frame_inspect"])
                agent.vr = FakeVideoReader(num_frames=30000)
                client = FakeModelClient([{"content": "A bird is visible."}])
                tool_call = {
                    "id": "frame-array",
                    "function": {
                        "name": "frame_inspect",
                        "arguments": json.dumps({
                            "query": "Identify the animal.",
                            "timestamps": timestamps,
                        }),
                    },
                }

                with patch("videospy.tools.frame_inspect.ModelClient", return_value=client):
                    message, succeeded = agent._execute_tool_call(tool_call)
                    tool_call["function"]["arguments"] = json.dumps({
                        "query": "Identify the animal.",
                        "timestamps": list(reversed(timestamps)),
                    })
                    duplicate, duplicate_succeeded = agent._execute_tool_call(tool_call)

                self.assertTrue(succeeded, message["content"])
                self.assertIn("A bird is visible.", message["content"])
                self.assertEqual(agent.vr.indices, [23993] if len(timestamps) == 1 else [23993, 24000])
                self.assertFalse(duplicate_succeeded)
                self.assertIn("Duplicate `frame_inspect`", duplicate["content"])
                self.assertEqual(len(client.calls), 1)

    def test_frame_inspect_rejects_non_array_tool_arguments(self):
        for timestamps in ([], "[]", "[2399.3]", "[2399.3, 2400.0]", "2399.3", "null", "{}", "[2399.3", 2399.3, None):
            with self.subTest(timestamps=timestamps):
                agent = build_agent(agent_config(), FakeModelClient([]), tools=["frame_inspect"])
                with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
                    message, succeeded = agent._execute_tool_call({
                        "id": "invalid-frame",
                        "function": {
                            "name": "frame_inspect",
                            "arguments": json.dumps({
                                "query": "Identify the animal.",
                                "timestamps": timestamps,
                            }),
                        },
                    })
                self.assertFalse(succeeded)
                self.assertIn("execution failed", message["content"])
                self.assertIn("timestamps must be a non-empty array", message["content"])
                client_class.assert_not_called()
                self.assertEqual(agent.vr.indices, [])

    def test_failed_tool_request_can_be_retried(self):
        client = FakeModelClient([])
        agent = build_agent(agent_config(), client)
        executions = []

        def execute_overview(**_):
            executions.append("overview")
            if len(executions) == 1:
                raise RuntimeError("temporary failure")
            return "overview evidence"

        agent.tool_registry = ToolRegistry(
            tools={"overview": DEFAULT_TOOL_REGISTRY.get_tool("overview")},
            tool_functions={"overview": execute_overview},
        )
        tool_call = {
            "id": "overview-1",
            "function": {"name": "overview", "arguments": "{}"},
        }

        self.assertFalse(agent._execute_tool_call(tool_call)[1])
        self.assertTrue(agent._execute_tool_call(tool_call)[1])
        self.assertEqual(executions, ["overview", "overview"])


class VideoTimestampTest(unittest.TestCase):
    def test_supported_timestamp_formats(self):
        cases = (
            (2179, 2179),
            (36.19, 36.19),
            ("2179", 2179),
            ("36:19", 2179),
            ("37:08", 2228),
            ("01:02:03.5", 3723.5),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_video_timestamp(value), expected)

    def test_invalid_timestamp_values_are_rejected(self):
        invalid_values = (
            "",
            "1:60",
            -1,
            float("inf"),
            float("nan"),
            True,
            {},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_video_timestamp(value)


class VideoSpyToolTest(unittest.TestCase):
    def test_default_registry_exposes_new_video_tools(self):
        tools = DEFAULT_TOOL_REGISTRY.resolve_tools(
            ["overview", "clip_skim", "frame_inspect"]
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["overview", "clip_skim", "frame_inspect"],
        )
        self.assertFalse(DEFAULT_TOOL_REGISTRY.has_tool("inspect"))
        self.assertFalse(DEFAULT_TOOL_REGISTRY.has_tool("skim"))
        self.assertFalse(DEFAULT_TOOL_REGISTRY.has_tool("focus"))

    def test_new_tool_schemas_use_explicit_time_parameters(self):
        overview_parameters = DEFAULT_TOOL_REGISTRY.get_tool("overview")["function"][
            "parameters"
        ]
        clip_parameters = DEFAULT_TOOL_REGISTRY.get_tool("clip_skim")["function"][
            "parameters"
        ]
        frame_parameters = DEFAULT_TOOL_REGISTRY.get_tool("frame_inspect")[
            "function"
        ]["parameters"]

        self.assertEqual(overview_parameters["required"], ["query"])
        self.assertEqual(
            clip_parameters["required"], ["query", "start_time", "end_time"]
        )
        self.assertNotIn("duration", clip_parameters["properties"])
        self.assertEqual(frame_parameters["required"], ["query", "timestamps"])
        self.assertNotIn("timestamp", frame_parameters["properties"])
        self.assertIn(
            "in seconds",
            clip_parameters["properties"]["start_time"]["description"],
        )
        self.assertIn(
            "in seconds",
            clip_parameters["properties"]["end_time"]["description"],
        )
        self.assertIn(
            "in seconds",
            frame_parameters["properties"]["timestamps"]["description"],
        )

    def test_agent_applies_configured_frame_inspect_limit_to_tool_schema(self):
        config = agent_config()
        config["tools"]["frame_inspect"]["max_frames"] = 6
        agent = build_agent(
            config, FakeModelClient([]), tools=["frame_inspect"]
        )

        frame_tool = next(
            tool
            for tool in agent.tools
            if tool["function"]["name"] == "frame_inspect"
        )

        self.assertEqual(
            frame_tool["function"]["parameters"]["properties"]["timestamps"][
                "maxItems"
            ],
            6,
        )

    def test_tool_descriptions_and_limits_are_local_to_each_agent(self):
        names = ["overview", "clip_skim", "frame_inspect"]
        templates = copy.deepcopy(DEFAULT_TOOL_REGISTRY.resolve_tools(names))
        agents = []
        for overview_frames, clip_frames, max_frames in ((64, 16, 4), (32, 8, 2)):
            config = agent_config()
            config["tools"]["overview"]["num_frames"] = overview_frames
            config["tools"]["clip_skim"]["num_frames"] = clip_frames
            config["tools"]["frame_inspect"]["max_frames"] = max_frames
            agents.append(build_agent(config, FakeModelClient([]), tools=names))
        for agent, counts in zip(agents, ((64, 16, 4), (32, 8, 2))):
            overview, clip, frame = [tool["function"] for tool in agent.tools]
            self.assertIn(f"{counts[0]} frames", overview["description"])
            self.assertIn(f"{counts[1]} frames", clip["description"])
            self.assertIn("does not inspect the interval", frame["description"])
            self.assertEqual(frame["parameters"]["properties"]["timestamps"]["maxItems"], counts[2])
        self.assertEqual(DEFAULT_TOOL_REGISTRY.resolve_tools(names), templates)

    def test_subtitles_are_selected_by_interval_overlap(self):
        subtitles = [
            {"start_time": 0, "end_time": 1, "subtitle": "before"},
            {"start_time": 2, "end_time": 4, "subtitle": "left overlap"},
            {"start_time": 5, "end_time": 6, "subtitle": "inside"},
            {"start_time": 7, "end_time": 9, "subtitle": "right overlap"},
            {"start_time": 10, "end_time": 11, "subtitle": "after"},
        ]

        selected = select_subtitles_in_range(subtitles, 4, 7)

        self.assertEqual(
            [subtitle["subtitle"] for subtitle in selected],
            ["left overlap", "inside", "right overlap"],
        )

    def test_sampled_subtitles_are_compact_ordered_and_deduplicated(self):
        subtitles = [
            {"start_time": 0, "end_time": 3, "subtitle": "spans both"},
            {"start_time": 2, "end_time": 2, "subtitle": "second"},
        ]

        text = sampled_subtitles_text(subtitles, [1, 2])

        self.assertEqual(
            text,
            "[0.0s-3.0s] spans both\n[2.0s-2.0s] second",
        )

    def test_clip_skim_uniformly_samples_the_selected_interval(self):
        reader = FakeVideoReader(num_frames=400)
        parameters = {
            "query": "Find the action.",
            "start_time": 0,
            "end_time": 20,
            "vr": reader,
            "subtitles": [],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "observation"}
            result = execute_clip_skim(clip_skim_config(), parameters)

        self.assertEqual(
            result,
            "Inspection mode: clip_skim\n"
            "Evidence status: coarse clip inspection.\n"
            "Actual sampled frames: 16; timestamp range: 0.000s - 20.000s; "
            "maximum adjacent sample gap: 1.400s.\n"
            "Coverage: individual sampled frames only, not continuous observation. "
            "Unsampled moments were not inspected.\n"
            "observation",
        )
        self.assertEqual(len(reader.indices), 16)
        self.assertEqual(len(set(reader.indices)), 16)
        client_class.assert_called_once_with(clip_skim_config()["model"])
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 16)
        self.assertIn("give a direct answer", content[-1]["text"])
        self.assertIn("Treat the question only as an inspection goal", content[-1]["text"])
        self.assertIn("any claim or suggested interpretation", content[-1]["text"])
        self.assertNotIn("anatomy", content[-1]["text"])
        self.assertIn("one concise bullet", content[-1]["text"])
        self.assertIn("candidate timestamps", content[-1]["text"])
        self.assertNotIn("frame_inspect", content[-1]["text"])

    def test_clip_skim_uses_compact_subtitle_format(self):
        parameters = {
            "query": "Find the action.",
            "start_time": 0,
            "end_time": 2,
            "vr": FakeVideoReader(),
            "subtitles": [
                {"start_time": 0.2, "end_time": 1.4, "subtitle": "Hello."}
            ],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "observation"}
            execute_clip_skim(clip_skim_config(), parameters)

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][-1]["text"]
        self.assertIn("[0.2s-1.4s] Hello.", text)
        self.assertNotIn("**Timestamp**", text)

    def test_clip_skim_has_no_default_duration_limit(self):
        reader = FakeVideoReader(num_frames=7000, fps=10.0)
        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "observation"}
            execute_clip_skim(
                clip_skim_config(),
                {
                    "query": "Search the full recording.",
                    "start_time": 0,
                    "end_time": 601,
                    "vr": reader,
                    "subtitles": [],
                },
            )

        client_class.return_value.chat.assert_called_once()

    def test_clip_skim_enforces_configured_duration_limit(self):
        reader = FakeVideoReader(num_frames=7000, fps=10.0)
        with self.assertRaisesRegex(ValueError, "must not exceed 600 seconds"):
            execute_clip_skim(
                clip_skim_config(max_duration=600),
                {
                    "query": "Search the full recording.",
                    "start_time": 0,
                    "end_time": 601,
                    "vr": reader,
                    "subtitles": [],
                },
            )

    def test_clip_skim_selects_subtitles_near_sampled_frames(self):
        reader = FakeVideoReader(num_frames=10001, fps=10.0)
        parameters = {
            "query": "Find the action.",
            "start_time": 0,
            "end_time": 1000,
            "vr": reader,
            "subtitles": [
                {"start_time": 5, "end_time": 6, "subtitle": "near first"},
                {"start_time": 500, "end_time": 501, "subtitle": "far away"},
                {"start_time": 995, "end_time": 996, "subtitle": "near last"},
            ],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "observation"}
            execute_clip_skim(
                clip_skim_config(num_frames=2, subtitle_context_seconds=10),
                parameters,
            )

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][-1]["text"]
        self.assertIn("near first", text)
        self.assertIn("near last", text)
        self.assertNotIn("far away", text)

    def test_clip_skim_can_disable_subtitles(self):
        config = clip_skim_config()
        config["use_subtitles"] = False
        parameters = {
            "query": "Find the action.",
            "start_time": 0,
            "end_time": 2,
            "vr": FakeVideoReader(),
            "subtitles": [
                {"start_time": 0, "end_time": 2, "subtitle": "hidden evidence"}
            ],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "observation"}
            execute_clip_skim(config, parameters)

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][-1]["text"]
        self.assertIn("Subtitles disabled.", text)
        self.assertNotIn("hidden evidence", text)

    def test_clip_skim_rejects_invalid_video_ranges(self):
        reader = FakeVideoReader()
        cases = ((-1, 1), (10, 11), (3, 3))
        for start_time, end_time in cases:
            with self.subTest(start_time=start_time, end_time=end_time):
                with self.assertRaises(ValueError):
                    execute_clip_skim(
                        clip_skim_config(),
                        {
                            "query": "Inspect the video.",
                            "start_time": start_time,
                            "end_time": end_time,
                            "vr": reader,
                            "subtitles": [],
                        },
                    )

    def test_clip_skim_clamps_end_time_to_video_duration(self):
        reader = FakeVideoReader()
        parameters = {
            "query": "Inspect the end of the video.",
            "start_time": 9,
            "end_time": 29,
            "vr": reader,
            "subtitles": [],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "ending"}
            result = execute_clip_skim(clip_skim_config(), parameters)

        self.assertIn("Inspection mode: clip_skim", result)
        self.assertIn(
            "requested 9.0s - 29.0s, inspected 9.0s - 10.0s", result
        )
        client_class.assert_called_once_with(clip_skim_config()["model"])
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertTrue(content[0]["text"].startswith("Video clip (9.0s - 10.0s):\n"))
        self.assertIn("timestamp range: 9.000s - 9.900s", content[0]["text"])
        self.assertLessEqual(max(reader.indices), len(reader) - 1)

    def test_clip_skim_accepts_colon_timestamp_strings(self):
        reader = FakeVideoReader(num_frames=25000)
        parameters = {
            "query": "What happens in the requested interval?",
            "start_time": "36:19",
            "end_time": "37:08",
            "vr": reader,
            "subtitles": [],
        }

        with patch("videospy.tools.clip_skim.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "evidence"}
            result = execute_clip_skim(clip_skim_config(), parameters)

        self.assertEqual(reader.indices[0], 21790)
        self.assertEqual(reader.indices[-1], 22280)
        self.assertIn("Evidence status: coarse clip inspection", result)

    def test_frame_inspect_uses_the_nearest_frame_and_matching_subtitle(self):
        reader = FakeVideoReader()
        parameters = {
            "query": "Read the visible label.",
            "timestamps": [0.26],
            "vr": reader,
            "subtitles": [
                {"start_time": 0.2, "end_time": 0.4, "subtitle": "Label shown."}
            ],
        }
        config = {"max_frames": 4, "model": model_config("frame-inspect-model")}

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "ABC"}
            result = execute_frame_inspect(config, parameters)

        self.assertEqual(reader.indices, [3])
        self.assertTrue(result.startswith("Frame inspection at 0.300s (requested 0.260s):\n"))
        self.assertIn("Actual sampled frames: 1; timestamp range: 0.300s - 0.300s", result)
        self.assertIn("maximum adjacent sample gap: 0.000s", result)
        self.assertTrue(result.endswith("\nABC"))
        client_class.assert_called_once_with(config["model"])
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 1)
        self.assertIn("Actual frame timestamp: 0.300s", content[0]["text"])
        self.assertIn("Label shown.", content[0]["text"])
        self.assertIn("Treat the question only as an inspection goal", content[-1]["text"])
        self.assertIn("any claim or suggested interpretation", content[-1]["text"])
        self.assertNotIn("anatomy", content[-1]["text"])
        self.assertIn("plausible alternatives", content[-1]["text"])
        self.assertIn("Insufficient visual evidence.", content[-1]["text"])
        self.assertIn("[0.2s-0.4s] Label shown.", content[0]["text"])
        self.assertNotIn("**Timestamp**", content[0]["text"])

    def test_frame_inspect_uses_configured_subtitle_window(self):
        reader = FakeVideoReader()
        config = {
            "max_frames": 4,
            "use_subtitles": True,
            "subtitle_context_seconds": 1,
            "model": model_config("frame-inspect-model"),
        }
        parameters = {
            "query": "Inspect the frame.",
            "timestamps": [5],
            "vr": reader,
            "subtitles": [
                {"start_time": 4.1, "end_time": 4.2, "subtitle": "nearby"},
                {"start_time": 3.8, "end_time": 3.9, "subtitle": "outside"},
            ],
        }

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "detail"}
            execute_frame_inspect(config, parameters)

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        self.assertIn("nearby", text)
        self.assertNotIn("outside", text)

    def test_frame_inspect_can_disable_subtitles(self):
        reader = FakeVideoReader()
        config = {
            "max_frames": 4,
            "use_subtitles": False,
            "subtitle_context_seconds": 0,
            "model": model_config("frame-inspect-model"),
        }

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "detail"}
            execute_frame_inspect(
                config,
                {
                    "query": "Inspect the frame.",
                    "timestamps": [1],
                    "vr": reader,
                    "subtitles": [
                        {"start_time": 0, "end_time": 2, "subtitle": "hidden"}
                    ],
                },
            )

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        self.assertIn("Subtitles disabled.", text)
        self.assertNotIn("hidden", text)

    def test_frame_inspect_accepts_video_endpoint_as_the_last_frame(self):
        reader = FakeVideoReader()
        config = {"max_frames": 4, "model": model_config("frame-inspect-model")}

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "ending"}
            result = execute_frame_inspect(
                config,
                {
                    "query": "Inspect the ending.",
                    "timestamps": [10],
                    "vr": reader,
                    "subtitles": [],
                },
            )

        self.assertEqual(reader.indices, [99])
        self.assertIn("Frame inspection at 9.900s", result)

    def test_frame_inspect_accepts_colon_timestamp_string_in_array(self):
        reader = FakeVideoReader()
        config = {"max_frames": 4, "model": model_config("frame-inspect-model")}

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "detail"}
            result = execute_frame_inspect(
                config,
                {
                    "query": "Inspect the frame.",
                    "timestamps": ["00:00:00.26"],
                    "vr": reader,
                    "subtitles": [],
                },
            )

        self.assertEqual(reader.indices, [3])
        self.assertIn("requested 0.260s", result)

    def test_frame_inspect_batches_sorted_frames_and_deduplicates_indices(self):
        reader = FakeVideoReader()
        config = {"max_frames": 4, "model": model_config("frame-inspect-model")}

        with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "comparison"}
            result = execute_frame_inspect(
                config,
                {
                    "query": "Compare the frames.",
                    "timestamps": [0.51, 0.49, 0.1],
                    "vr": reader,
                    "subtitles": [
                        {"start_time": 0.1, "end_time": 0.1, "subtitle": "First"},
                        {"start_time": 0.5, "end_time": 0.5, "subtitle": "Second"},
                    ],
                },
            )

        self.assertEqual(reader.indices, [1, 5])
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 2)
        self.assertIn("First", content[0]["text"])
        self.assertIn("Second", content[2]["text"])
        self.assertIn("requested 0.100s, 0.490s, 0.510s", result)

    def test_frame_inspect_truncates_in_input_order_and_reports_actual_frames(self):
        for limit, timestamps, expected_indices in (
            (4, [5, 3, 1, 2, 0, "invalid"], [10, 20, 30, 50]),
            (2, [0.51, 0.49, 1], [5]),
        ):
            with self.subTest(limit=limit):
                reader = FakeVideoReader()
                config = {"max_frames": limit, "model": model_config("frame-model")}
                with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
                    client_class.return_value.chat.return_value = {"content": "detail"}
                    result = execute_frame_inspect(
                        config,
                        {"query": "Compare.", "timestamps": timestamps,
                         "vr": reader, "subtitles": []},
                    )
                self.assertEqual(reader.indices, expected_indices)
                client_class.return_value.chat.assert_called_once()
                content = client_class.return_value.chat.call_args.kwargs["messages"][0]["content"]
                self.assertEqual(
                    sum(item["type"] == "image_url" for item in content),
                    len(expected_indices),
                )
                notice = content[0]["text"]
                self.assertIn(f"only the first {limit} in input order", notice)
                self.assertIn(f"{len(expected_indices)} distinct frames", notice)
                self.assertIn("remaining timestamps were not inspected", notice)
                self.assertIn(f"at most {limit} timestamps per call", notice)
                self.assertTrue(result.startswith(notice))

    def test_frame_inspect_accepts_rounded_endpoint_without_reading_past_it(self):
        for duration, timestamp in (
            (134717.17000000004, 134717.2),
            (140321.15000000002, 140321.2),
            (9.96, 10.0),
            (9.749, 9.8),  # Agent rounds to two decimals, then displays one.
        ):
            with self.subTest(duration=duration):
                reader = FakeVideoReader(num_frames=int(duration) + 1, fps=1)
                reader.duration = duration
                config = {"max_frames": 4, "model": model_config("frame-model")}
                with patch("videospy.tools.frame_inspect.ModelClient") as client_class:
                    client_class.return_value.chat.return_value = {"content": "ending"}
                    result = execute_frame_inspect(
                        config,
                        {"query": "Ending?", "timestamps": [timestamp],
                         "vr": reader, "subtitles": []},
                    )
                self.assertEqual(reader.indices, [int(duration)])
                self.assertLessEqual(reader.indices[0], duration)
                self.assertIn(f"requested {timestamp:.3f}s", result)
                self.assertIn(f"Frame inspection at {int(duration):.3f}s", result)

    def test_frame_inspect_rejects_invalid_timestamp_arrays(self):
        config = {"max_frames": 2, "model": model_config("frame-inspect-model")}
        invalid_values = ([], [-0.1], [10.1])
        for timestamps in invalid_values:
            with self.subTest(timestamps=timestamps):
                with self.assertRaises(ValueError):
                    execute_frame_inspect(
                        config,
                        {
                            "query": "Inspect the frame.",
                            "timestamps": timestamps,
                            "vr": FakeVideoReader(),
                            "subtitles": [],
                        },
                    )

    def test_frame_inspect_rejects_invalid_max_frames(self):
        with self.assertRaisesRegex(ValueError, "max_frames must"):
            execute_frame_inspect(
                {"max_frames": 0, "model": model_config("frame-inspect-model")},
                {
                    "query": "Inspect the frame.",
                    "timestamps": [1],
                    "vr": FakeVideoReader(),
                    "subtitles": [],
                },
            )

    def test_frame_inspect_rejects_out_of_range_timestamp(self):
        config = {"max_frames": 4, "model": model_config("frame-inspect-model")}
        for timestamp in (-0.1, 10.1):
            with self.subTest(timestamp=timestamp):
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    execute_frame_inspect(
                        config,
                        {
                            "query": "Inspect the frame.",
                            "timestamps": [timestamp],
                            "vr": FakeVideoReader(),
                            "subtitles": [],
                        },
                    )

    def test_overview_uses_largest_available_multiple_of_eight(self):
        reader = FakeVideoReader(num_frames=50)
        config = {"num_frames": 64, "model": model_config("overview-model")}
        with patch("videospy.tools.overview.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {
                "content": "0.0s: first frame\n4.9s: last frame"
            }
            result = execute_overview(
                config, {"query": "What happens?", "vr": reader, "subtitles": []}
            )

        self.assertTrue(result.endswith("0.0s: first frame\n4.9s: last frame"))
        self.assertIn("Actual sampled frames: 48; timestamp range: 0.000s - 4.900s", result)
        self.assertEqual(len(reader.indices), 48)
        self.assertEqual(len(set(reader.indices)), 48)
        call = client_class.return_value.chat.call_args
        self.assertNotIn("return_json", call.kwargs)
        content = call.kwargs["messages"][0]["content"]
        self.assertEqual(
            sum(item["type"] == "image_url" for item in content),
            6,
        )

    def test_overview_rounds_configuration_up_to_multiple_of_eight(self):
        reader = FakeVideoReader(num_frames=100)
        config = {"num_frames": 60, "model": model_config("overview-model")}
        with patch("videospy.tools.overview.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "summary"}
            execute_overview(
                config,
                {"query": "What happens?", "vr": reader, "subtitles": []},
            )

        self.assertEqual(len(reader.indices), 64)
        self.assertEqual(len(set(reader.indices)), 64)
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertEqual(
            sum(item["type"] == "image_url" for item in content),
            8,
        )

    def test_overview_uses_all_frames_when_video_has_fewer_than_eight(self):
        reader = FakeVideoReader(num_frames=7)
        config = {"num_frames": 64, "model": model_config("overview-model")}
        with patch("videospy.tools.overview.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "summary"}
            execute_overview(
                config,
                {"query": "What happens?", "vr": reader, "subtitles": []},
            )

        self.assertEqual(reader.indices, list(range(7)))
        content = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ]
        self.assertEqual(
            sum(item["type"] == "image_url" for item in content),
            7,
        )

    def test_overview_uses_sampled_subtitles_for_all_video_readers(self):
        subtitles = [
            {"start_time": 0, "end_time": 0, "subtitle": "sampled"},
            {"start_time": 0.4, "end_time": 0.4, "subtitle": "not sampled"},
        ]
        prompts = []
        for sparse in (False, True):
            reader = FakeVideoReader(num_frames=8, fps=1.0)
            reader.sparse_subtitles = sparse
            config = {
                "num_frames": 8,
                "use_subtitles": True,
                "subtitle_context_seconds": 0,
                "model": model_config("overview-model"),
            }
            with patch("videospy.tools.overview.ModelClient") as client_class:
                client_class.return_value.chat.return_value = {"content": "summary"}
                execute_overview(
                    config,
                    {"query": "What happens?", "vr": reader, "subtitles": subtitles},
                )
            prompts.append(
                client_class.return_value.chat.call_args.kwargs["messages"][0][
                    "content"
                ][-1]["text"]
            )

        self.assertEqual(prompts[0], prompts[1])
        self.assertIn("[0.0s-0.0s] sampled", prompts[0])
        self.assertNotIn("not sampled", prompts[0])
        self.assertNotIn("**Timestamp**", prompts[0])

    def test_overview_can_disable_subtitles(self):
        reader = FakeVideoReader(num_frames=8, fps=1.0)
        config = {
            "num_frames": 8,
            "use_subtitles": False,
            "subtitle_context_seconds": 0,
            "model": model_config("overview-model"),
        }
        with patch("videospy.tools.overview.ModelClient") as client_class:
            client_class.return_value.chat.return_value = {"content": "summary"}
            execute_overview(
                config,
                {
                    "query": "What happens?",
                    "vr": reader,
                    "subtitles": [
                        {
                            "start_time": 0,
                            "end_time": 1,
                            "subtitle": "SECRET_SUBTITLE_123",
                        }
                    ],
                },
            )

        text = client_class.return_value.chat.call_args.kwargs["messages"][0][
            "content"
        ][-1]["text"]
        self.assertIn("Subtitles disabled.", text)
        self.assertNotIn("SECRET_SUBTITLE_123", text)


class VideoSpyUsageTest(unittest.TestCase):
    def test_model_client_uses_bounded_default_output(self):
        config = model_config("test")
        config.pop("max_completion_tokens")
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message={"role": "assistant", "content": "B"})
            ],
            usage=None,
        )

        with patch("videospy.model.completion", return_value=response) as call:
            ModelClient(config).chat([])

        self.assertEqual(call.call_args.kwargs["max_completion_tokens"], 4096)

    def test_default_retry_limit_is_three(self):
        from videospy.evolving.reflector import Reflector

        config = model_config("test")
        config.pop("max_retries")
        error = RuntimeError("internal server error")
        client = ModelClient(config)
        reflector = Reflector({"agent": {"model": config}}, client)
        for request in (lambda: client.chat([]), lambda: reflector.classify({})):
            with (
                self.subTest(request=request),
                patch("videospy.model.completion", side_effect=error) as call,
                patch("videospy.model.time.sleep"),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    request()
                self.assertIs(raised.exception, error)
                self.assertEqual(call.call_count, 4)
                self.assertTrue(all(c.kwargs['num_retries'] == 0 for c in call.call_args_list))

    def test_model_client_passes_enable_thinking_from_config(self):
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message={"role": "assistant", "content": "B"})
            ],
            usage=None,
        )

        for enable_thinking in (True, False):
            with self.subTest(enable_thinking=enable_thinking):
                config = model_config("test")
                config["enable_thinking"] = enable_thinking
                with patch("videospy.model.completion", return_value=response) as call:
                    ModelClient(config).chat([])

                self.assertEqual(
                    call.call_args.kwargs["extra_body"],
                    {"enable_thinking": enable_thinking},
                )

    def test_explicit_enable_thinking_overrides_config(self):
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message={"role": "assistant", "content": "B"})
            ],
            usage=None,
        )
        config = model_config("test")
        config["enable_thinking"] = True

        with patch("videospy.model.completion", return_value=response) as call:
            ModelClient(config).chat(
                [], extra_body={"enable_thinking": False, "custom": "value"}
            )

        self.assertEqual(
            call.call_args.kwargs["extra_body"],
            {"enable_thinking": False, "custom": "value"},
        )

    def test_model_client_records_usage(self):
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message={"role": "assistant", "content": "B"})
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )
        with metrics.track_token_usage() as usage:
            with patch("videospy.model.completion", return_value=response):
                message = ModelClient(model_config("test")).chat([])

        self.assertEqual(message["content"], "B")
        self.assertEqual(usage.total_tokens, 12)
        self.assertEqual(usage.llm_calls, 1)


if __name__ == "__main__":
    unittest.main()
