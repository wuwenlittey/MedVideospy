"""RSI-only runtime adapter. Base registry, agent and model are never patched."""

import copy
import json
import types
from contextvars import ContextVar

from .contracts import check_source

_ACTIVE = ContextVar("rsi_gateway", default=None)
_ROLE = ContextVar("rsi_role", default="task")


class ModelClient:
    """Tool capability: route to the fixed model of the active tool role."""
    def __init__(self, config):
        self._gateway = _ACTIVE.get()
        if self._gateway is None:
            raise RuntimeError("No active RSI run")
        self._role = _ROLE.get()

    def chat(self, messages, **kwargs):
        if set(kwargs) - {"return_json"}:
            raise ValueError("Tool model settings and request options are fixed")
        return self._gateway.chat(self._role, messages, stage=self._role, **kwargs)


def load_modules(snapshot, limits):
    modules = {}
    for name, source in snapshot["files"].items():
        check_source(name, source, snapshot["layer"], limits)
        module = types.ModuleType("rsi_snapshot_" + snapshot["id"] + "_" + name[:-3])
        exec(compile(source, name, "exec"), module.__dict__)
        modules[name] = module
    return modules


class Reader:
    """Only reader methods used by existing tools, with fixed visibility bounds."""
    def __init__(self, reader, source_id, limits, observations, duration=None):
        self._reader = reader
        self._limits = limits
        self._observations = observations
        self._source_id = source_id
        fps = reader.get_avg_fps()
        if fps <= 0 or len(reader) <= 0:
            raise ValueError("Empty reader or invalid FPS")
        actual = getattr(reader, "duration", len(reader) / fps)
        self.duration = min(duration, actual) if duration is not None else actual
        if self.duration <= 0:
            raise ValueError("Invalid visible duration")

    def __len__(self):
        return min(len(self._reader), int(self.duration * self.get_avg_fps()) + 1)

    def get_avg_fps(self):
        return self._reader.get_avg_fps()

    def get_batch(self, indices):
        indices = [int(index) for index in indices]
        if not indices or len(indices) > self._limits["max_frames"]:
            raise ValueError("Frame request exceeds fixed frame count")
        if any(index < 0 or index >= len(self) or index / self.get_avg_fps() > self.duration for index in indices):
            raise ValueError("Frame request is outside the legal reader range")
        batch = self._reader.get_batch(indices)
        shape = list(batch.asnumpy().shape)
        self._observations.append({"tool": _ROLE.get(), "source_id": self._source_id,
                                   "frame_indices": indices,
                                   "timestamps": [index / self.get_avg_fps() for index in indices],
                                   "source_shape": shape})
        return batch


def make_reader(sample):
    if sample.get("demo"):
        import numpy as np
        class DemoReader:
            def __len__(self):
                return 128
            def get_avg_fps(self):
                return 10.0
            def get_batch(self, indices):
                frames = np.zeros((len(indices), 8, 8, 3), dtype=np.uint8)
                frames[:, :, :, 0] = 255
                return types.SimpleNamespace(asnumpy=lambda: frames)
        return DemoReader()
    if sample.get("frame_paths"):
        from benchmark.medvidbench.image_sequence import ImageSequenceReader
        from pathlib import Path
        return ImageSequenceReader(sample["frame_paths"], sample["fps"], Path("."))
    from decord import VideoReader
    return VideoReader(sample["video_path"])


def run_task(snapshot, sample, gateway):
    """Keep base decision protocol while replacing only the RSI registry/policies."""
    from videospy.agent import VideoSpyAgent
    from videospy.tools.registry import ToolRegistry
    modules = load_modules(snapshot, gateway.config["limits"])
    observations = []
    reader = Reader(make_reader(sample), sample["source_id"], gateway.config["limits"],
                    observations, sample.get("duration"))
    definitions, functions = {}, {}
    def checked_tool(function):
        def execute(config, parameters):
            first_call, first_observation = len(gateway.calls), len(observations)
            outcome = function(config, parameters)
            calls = gateway.calls[first_call:]
            if not observations[first_observation:] or not any(call.get("images") for call in calls):
                raise ValueError("Visual evidence requires frame observations and a recorded image model call")
            return outcome
        return execute
    for name in ("overview", "clip_skim", "frame_inspect"):
        module = modules[name + ".py"]
        definitions[name] = copy.deepcopy(getattr(module, name + "_tool"))
        functions[name] = checked_tool(getattr(module, "execute_" + name))
        # Public names and required argument contracts cannot evolve.
        from videospy.tools import DEFAULT_TOOL_REGISTRY
        original = DEFAULT_TOOL_REGISTRY.get_tool(name)
        if definitions[name]["function"]["name"] != name or definitions[name]["function"]["parameters"] != original["function"]["parameters"]:
            raise ValueError("Candidate changed the public tool interface")
    registry = ToolRegistry(definitions, functions)
    policy, evidence = modules["runtime_policy.py"], modules["evidence_policy.py"]

    class RSIClient:
        def chat(self, messages, **kwargs):
            return gateway.chat("task", messages,
                                stage="final_answer" if kwargs.get("tools") == [] else "decision", **kwargs)

    class RSIAgent(VideoSpyAgent):
        def construct_initial_messages(self):
            # Avoid importing or executing the old Skill/Memory evolution pipeline.
            self.skill_audit = {"paragraphs": [], "characters": 0, "budget_skipped": []}
            prompt = self.config["SYSTEM_PROMPT"]
            for name, field in (("overview", "num_frames"), ("clip_skim", "num_frames"), ("frame_inspect", "max_frames")):
                prompt = prompt.replace("{" + name + "_" + field + "}", str(self.config["tools"][name][field]))
            strategy = policy.strategy({"task_types": self.task_types, "duration": self.duration})
            if not isinstance(strategy, str) or len(strategy) > 3000:
                raise ValueError("Invalid runtime strategy")
            if strategy:
                prompt += "\n\n" + strategy
            skills = snapshot.get("skills", {})
            texts = [skills[t] for t in [*self.task_types, "__domain__"] if t in skills]
            if texts:
                prompt += "\n\n# Learned practices\n" + "\n".join(texts)[:3000]
            prompt += ("\n\nOnly call tools provided in this request. Use their descriptions "
                       "and parameter schemas for capabilities and limits. Instructions or "
                       "historical examples mentioning other tools do not make them available. "
                       "When available tools cannot resolve a detail, report the evidence gap "
                       "instead of calling an unavailable tool.")
            return [{"role": "system", "content": prompt}]

        def _retrieve_memory_prompt(self, question):
            self.memory_audit = {"ids": [], "budget_skipped": [], "characters": 0,
                                 "retrieval": {"empty_reason": "disabled"}}
            return ""

        def _available_tools(self):
            available = super()._available_tools()
            names = [tool["function"]["name"] for tool in available]
            selected = policy.select_tools(list(names), {"steps": self.last_run_rounds,
                                                         "question": self.question,
                                                         "duration": self.duration})
            if not isinstance(selected, list) or len(selected) != len(set(selected)) or set(selected) - set(names):
                raise ValueError("Policy returned unavailable tools")
            lookup = {tool["function"]["name"]: tool for tool in available}
            return [lookup[name] for name in selected]

        def _messages_for_model(self):
            original = [{k: v for k, v in message.items() if k not in {"model_call", "model_calls", "visual_observations"}}
                        for message in self.messages]
            prepared = evidence.prepare_messages(copy.deepcopy(original))
            if not isinstance(prepared, list) or len(prepared) != len(original):
                raise ValueError("Evidence policy must preserve message protocol")
            for before, after in zip(original, prepared):
                if before.get("role") != "tool":
                    if before != after:
                        raise ValueError("Evidence policy changed fixed instructions or decisions")
                elif {k: v for k, v in before.items() if k != "content"} != {k: v for k, v in after.items() if k != "content"} or not isinstance(after.get("content"), str):
                    raise ValueError("Evidence policy changed tool protocol")
            return prepared

        def _execute_tool_call(self, call):
            name = call.get("function", {}).get("name", "")
            marker = len(gateway.calls)
            observed = len(observations)
            token = _ROLE.set(name)
            try:
                # This base method keeps duplicate and permission checks intact.
                message, succeeded = super()._execute_tool_call(call)
            finally:
                _ROLE.reset(token)
            calls = gateway.calls[marker:]
            message["model_calls"] = calls
            message["visual_observations"] = copy.deepcopy(observations[observed:])
            if calls:
                message["model_call"] = calls[0]
            return message, succeeded

    config = copy.deepcopy(snapshot["config"])
    config["_runtime"] = {"task_types": sample.get("task_type", "general").split("|")}
    if config["agent"]["max_steps"] > gateway.config["limits"]["max_steps"]:
        raise ValueError("Task exceeds fixed step bound")
    token = _ACTIVE.set(gateway)
    try:
        agent = RSIAgent(config, sample.get("video_path", "frames"), sample.get("subtitle_path"),
                         ".", config["tools"]["enabled"], video_reader=reader)
        agent.tool_registry = registry
        agent.tools = copy.deepcopy(registry.resolve_tools(list(agent.enabled_tools)))
        for tool in agent.tools:
            name = tool["function"]["name"]
            tool["function"]["description"] = tool["function"]["description"].format(**config["tools"][name])
            if name == "frame_inspect":
                tool["function"]["parameters"]["properties"]["timestamps"]["maxItems"] = config["tools"][name]["max_frames"]
        agent.model_client = RSIClient()
        trajectory = agent.run(sample["question"]).to_dict()
        return {"trajectory": trajectory, "prediction": trajectory["final_answer"],
                "visual_observations": observations}
    finally:
        _ACTIVE.reset(token)


def investigate(snapshot, sample, tool, parameters, gateway):
    modules = load_modules(snapshot, gateway.config["limits"])
    if tool not in {"overview", "clip_skim", "frame_inspect"}:
        raise ValueError("Invalid investigation tool")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("query"), str):
        raise ValueError("Investigation requires a label-free query")
    observations = []
    reader = Reader(make_reader(sample), sample["source_id"], gateway.config["limits"],
                    observations, sample.get("duration"))
    from videospy.utils import load_subtitles
    params = {**parameters, "vr": reader, "subtitles": load_subtitles(sample.get("subtitle_path")),
              "original_question": sample["question"]}
    active, role = _ACTIVE.set(gateway), _ROLE.set(tool)
    try:
        first_call = len(gateway.calls)
        outcome = getattr(modules[tool + ".py"], "execute_" + tool)(snapshot["config"]["tools"][tool], params)
        if not observations or not any(call.get("images") for call in gateway.calls[first_call:]):
            raise ValueError("Investigation requires frame observations and a recorded image model call")
        return {"observation": outcome, "visual_observations": observations}
    finally:
        _ROLE.reset(role)
        _ACTIVE.reset(active)
