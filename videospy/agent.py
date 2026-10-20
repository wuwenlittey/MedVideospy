import copy
import json
import logging
import traceback
from abc import ABC, abstractmethod

from decord import VideoReader

from .core import Trajectory
from .core.prompts import final_answer_message
from .metrics import track_agent_usage
from .model import ModelClient, collect_model_calls
from .tools import DEFAULT_TOOL_REGISTRY
from .tools.clip_refinement import refinement_config, refinement_enabled, refine_clip
from .utils import (
    convert_to_free_form_text_representation,
    load_subtitles,
    parse_video_timestamp,
)


class VisualEvidenceRequiredError(RuntimeError):
    pass


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.final_answer = None
        self.question = None
        self.last_run_rounds = 0
        self.last_run_token_usage = {}

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None
        self.last_run_rounds = 0

    @abstractmethod
    def construct_initial_messages(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSpyAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str | None,
        output_dir: str,
        tools: list[str],
        verbose: bool = False,
        video_reader=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = video_reader if video_reader is not None else VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = copy.deepcopy(self.tool_registry.resolve_tools(tools))
        self.enabled_tools = tuple(tool["function"]["name"] for tool in self.tools)
        self.model_call_input_mode = config["tools"].get(
            "model_call_input_mode", "compact"
        )
        if self.model_call_input_mode not in {"compact", "full"}:
            raise ValueError(
                "tools.model_call_input_mode must be 'compact' or 'full', got "
                f"{self.model_call_input_mode!r}."
            )
        for tool in self.tools:
            function = tool["function"]
            tool_config = config["tools"][function["name"]]
            function["description"] = function["description"].format(
                **tool_config
            )
            if function["name"] == "frame_inspect":
                max_frames = tool_config["max_frames"]
                if (
                    not isinstance(max_frames, int)
                    or isinstance(max_frames, bool)
                    or max_frames <= 0
                ):
                    raise ValueError(
                        f"tools.frame_inspect.max_frames must be a positive integer, "
                        f"got {max_frames!r}."
                    )
                function["parameters"]["properties"]["timestamps"][
                    "maxItems"
                ] = max_frames
        self.output_dir = output_dir
        self.verbose = verbose
        runtime_task_types = config.get("_runtime", {}).get("task_types")
        if isinstance(runtime_task_types, list) and runtime_task_types:
            self.task_types = [str(value) for value in runtime_task_types]
        else:
            self.task_types = str(
                config.get("_runtime", {}).get("task_type") or "general"
            ).split("|")
        self.task_type = "|".join(self.task_types)

        self.duration = round(
            getattr(self.vr, "duration", len(self.vr) / self.vr.get_avg_fps()), 2
        )
        self.subtitles = load_subtitles(subtitle_path)
        self.model_client = ModelClient(config["agent"]["model"])
        self.max_steps = config["agent"]["max_steps"]
        self.overview_once_per_task = config["tools"].get("overview", {}).get(
            "once_per_task", True
        )
        self._overview_called = False
        memory_config = config["tools"].get("memory_search", {})
        self.memory_store = None
        self.memory_context = None
        if memory_config.get("enabled") and memory_config.get("database_path"):
            from .evolving.storage.memory_store import (
                MemoryStore,
                embedding_function_from_config,
            )

            self.memory_store = MemoryStore(
                memory_config["database_path"],
                embedding_function=embedding_function_from_config(memory_config),
                read_only=True,
            )
        self._successful_tool_requests: set[tuple] = set()
        self.clip_refinement_settings = refinement_config(
            config["tools"].get("clip_skim", {}).get("refinement")
        )
        self.clip_refinement_audit = []
        self.visual_memory_session = None
        visual_memory = config["tools"].get("visual_memory", {})
        if visual_memory.get("enabled"):
            from .visual_memory.integration import VisualMemorySession

            self.visual_memory_session = VisualMemorySession(visual_memory)
        self.messages = self.construct_initial_messages()

    def reset(self) -> None:
        super().reset()
        self._successful_tool_requests.clear()
        self._overview_called = False
        self.clip_refinement_audit.clear()
        if self.visual_memory_session is not None:
            self.visual_memory_session.reset()

    def _available_tools(self) -> list[dict]:
        return [
            tool for tool in self.tools
            if not (
                self.overview_once_per_task
                and self._overview_called
                and tool["function"]["name"] == "overview"
            )
        ]

    def construct_initial_messages(self) -> list[dict]:
        tool_config = self.config["tools"]
        system_prompt = self.config["SYSTEM_PROMPT"]
        for name, field in (
            ("overview", "num_frames"),
            ("clip_skim", "num_frames"),
            ("frame_inspect", "max_frames"),
        ):
            if field in tool_config.get(name, {}):
                system_prompt = system_prompt.replace(
                    f"{{{name}_{field}}}", str(tool_config[name][field])
                )
        from .evolved_config import format_skills
        skills, self.skill_audit = format_skills(self.config, self.task_types)
        if skills:
            system_prompt += '\n\n' + skills
        system_prompt += (
            "\n\nOnly call tools provided in this request. Use their descriptions "
            "and parameter schemas for capabilities and limits. Instructions or "
            "historical examples "
            "mentioning other tools do not make them available. "
            "When available tools cannot resolve a detail, report the evidence gap "
            "instead of calling an unavailable tool."
        )
        return [{"role": "system", "content": system_prompt}]

    def _retrieve_memory_prompt(self, question: str) -> str:
        self.memory_audit = {'ids': [], 'budget_skipped': [], 'characters': 0, 'retrieval': {}}
        if self.memory_store is None:
            self.memory_audit['retrieval']['empty_reason'] = 'disabled'
            return ""
        memory_config = self.config["tools"]["memory_search"]
        try:
            memories = self.memory_store.search(
                version_id=str(memory_config.get("version_id") or ""),
                query=(self.memory_context or {}).get("query", question),
                task_types=None,
                context=self.memory_context,
                audit=self.memory_audit['retrieval'],
                top_k=int(memory_config.get("top_k", 3)),
                similarity_threshold=float(
                    memory_config.get("similarity_threshold", 0.5)
                ),
            )
        except Exception as error:
            self.memory_audit['retrieval'].update(empty_reason='retrieval_failed', error=str(error))
            logging.getLogger(__name__).warning("Memory retrieval failed; continuing without prior memory: %s", error)
            return ""
        if not memories:
            return ""

        from .evolving.common.schemas import MEMORY_PROMPT_CHARS
        header = ('# Prior Memory\nHistorical cases teach methods; their people, answers and timestamps '
                  'do not describe this video. Adapt the method only when its trigger matches. '
                  'Local observations are from this source, not necessarily the latest state. '
                  'Use relevant positions to narrow inspection; current observations take precedence. '
                  'Do not infer absence, habits or the latest event from one observation. '
                  'Video seconds are not elapsed calendar time.\n')
        text, included, skipped = header, [], []
        for memory in memories:
            metadata = memory.get('metadata') or {}
            source = metadata.get('source') or {}
            if metadata.get('kind') == 'reasoning_case':
                entry = '\nHistorical reasoning case:\n' + memory['content'] + '\n'
                if len(text) + len(entry) > MEMORY_PROMPT_CHARS:
                    skipped.append(memory.get('memory_id'))
                    continue
                text += entry
                included.append(memory.get('memory_id'))
                continue
            event_time = metadata.get('time') or {}
            position = ', '.join(f'{label}={event_time[key]}s' for key, label in
                                 (('start_seconds', 'start'), ('end_seconds', 'end'))
                                 if event_time.get(key) is not None)
            calendar_time = event_time.get('label', '')
            if calendar_time and (event_time.get('calendar') or {}).get('continuous') is False:
                calendar_time += ' (recording gaps)'
            fields = [('Source', source.get('source_id')), ('Observation', memory['content']),
                      ('Video position', position), ('Event time', calendar_time),
                      ('Available from', source.get('query_time')),
                      ('Uncertainty', metadata.get('uncertainty'))]
            if not position and metadata.get('evidence_frames'):
                frames = ', '.join(f'{timestamp}s' for timestamp in metadata['evidence_frames'])
                fields.insert(2, ('Source inspection frames (event time unknown)', frames))
            entry = '\n' + '\n'.join(f'{k}: {v}' for k, v in fields if v is not None and str(v).strip()) + '\n'
            if len(text) + len(entry) > MEMORY_PROMPT_CHARS:
                skipped.append(memory.get('memory_id'))
                continue
            text += entry
            included.append(memory.get('memory_id'))
        text = text if included else ''
        self.memory_audit.update(ids=included, budget_skipped=skipped, characters=len(text))
        return text

    def _tool_request_key(
        self, function_name: str, parameters: dict
    ) -> tuple | None:
        query = " ".join(str(parameters.get("query", "")).split()).casefold()
        if function_name == "overview":
            return (function_name, query)
        if function_name == "clip_skim":
            return (
                function_name,
                parse_video_timestamp(parameters["start_time"]),
                parse_video_timestamp(parameters["end_time"]),
                query,
            )
        if function_name == "frame_inspect":
            requested_timestamps = parameters["timestamps"]
            if not isinstance(requested_timestamps, list) or not requested_timestamps:
                raise ValueError("timestamps must be a non-empty array.")
            timestamps = {
                parse_video_timestamp(timestamp)
                for timestamp in requested_timestamps
            }
            return (function_name, tuple(sorted(timestamps)), query)
        return None

    def _execute_tool_call(self, tool_call: dict) -> tuple[dict, bool]:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        model_calls = []
        succeeded = False
        refinement_audit_start = len(self.clip_refinement_audit)
        visual_memory_start = len(self.visual_memory_session.audit) if self.visual_memory_session else 0

        try:
            parameters = json.loads(function.get("arguments", "{}"))
            if not isinstance(parameters, dict):
                raise ValueError("Tool arguments must be a JSON object")
            if not self.tool_registry.has_tool(function_name):
                raise ValueError(f"Invalid function name: {function_name}")
            if function_name not in self.enabled_tools:
                raise ValueError(
                    f"Tool `{function_name}` is not enabled. "
                    f"Enabled tools: {', '.join(self.enabled_tools) or 'none'}."
                )

            request_key = self._tool_request_key(function_name, parameters)
            if (
                function_name == "overview"
                and self.overview_once_per_task
                and self._overview_called
            ):
                outcome = (
                    "Duplicate `overview` inspection skipped: overview was already "
                    "called and is unavailable for this task. DO NOT call overview "
                    "again, even with a different query. No new inspection was "
                    "performed. Reuse its observation "
                    "or inspect a relevant local clip or frame with an available tool."
                )
            elif request_key in self._successful_tool_requests:
                outcome = (
                    f"Duplicate `{function_name}` inspection skipped because it "
                    "already succeeded. STOP repeating this request. This call "
                    "was NOT executed and supplied NO new evidence. You MUST NOT "
                    "submit the same request again; it will be blocked again. "
                    "Rephrasing the same question about the same frames does not "
                    "make it a new inspection. Before your next call, identify "
                    "the unresolved detail and choose new relevant frames, a "
                    "different interval, or a materially different detail to inspect. "
                    "If no useful new inspection is available, end evidence gathering "
                    "with a brief summary of the existing evidence and remaining "
                    "uncertainty. Do not continue this duplicate-call loop."
                )
            else:
                execution_parameters = dict(parameters)
                execution_parameters.update(
                    {
                        "vr": self.vr,
                        "subtitles": self.subtitles,
                        "original_question": self.question,
                    }
                )
                if function_name == "overview":
                    self._overview_called = True
                execution_config = self.config["tools"][function_name]
                if self.visual_memory_session is not None and function_name in {"clip_skim", "frame_inspect"}:
                    from pathlib import PurePosixPath

                    source = self.memory_context or {}
                    explicit = self.visual_memory_session.settings["context"]
                    # Runtime context is assigned by benchmarks AFTER construction.
                    # MedVidBench source_id identifies a clip, so also exclude the
                    # original video, including its other clips and questions.
                    video = source.get("source_video_id") or source.get("video") or explicit.get("source_video_id")
                    if not video:
                        video = PurePosixPath(str(self.video_path).replace("\\", "/")).stem
                    execution_parameters["_visual_memory_session"] = self.visual_memory_session
                    execution_parameters["_visual_memory_context"] = {
                        **explicit,
                        "source_video_id": str(video),
                        "source_id": source.get("source_id") or explicit.get("source_id"),
                        "source_sample_id": source.get("sample_id") or source.get("source_sample_id") or explicit.get("source_sample_id"),
                        "patient_id": source.get("patient_id") or explicit.get("patient_id"),
                        "task_type": explicit.get("task_type") or self.task_type,
                    }
                should_refine = (
                    function_name == "clip_skim"
                    and refinement_enabled(self.clip_refinement_settings)
                    and len(self.clip_refinement_audit) < self.clip_refinement_settings["max_clips_per_question"]
                )
                if should_refine:
                    execution_parameters["_clip_refinement_audit"] = self.clip_refinement_audit
                    execution_parameters["_clip_refinement_video_path"] = self.video_path
                with collect_model_calls(self.model_call_input_mode) as model_calls:
                    outcome = self.tool_registry.get_function(function_name)(
                        config=execution_config,
                        parameters=execution_parameters,
                    )
                    if outcome is not None and should_refine:
                        outcome = str(outcome) + refine_clip(
                            config=execution_config, parameters=execution_parameters,
                        )
                if outcome is None:
                    raise ValueError(f"Tool returns None results: {function_name}")
                succeeded = True
                if request_key is not None:
                    self._successful_tool_requests.add(request_key)
        except Exception as error:
            outcome = (
                f"Tool `{function_name}` execution failed.\n"
                f"Exception: {type(error).__name__}: {error}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

        if self.verbose:
            print("OBSERVATION")
            print(outcome)
            print("--------------------------------")

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": function_name,
            "content": str(outcome),
        }
        if model_calls:
            tool_message["model_call"] = model_calls[0]
            if len(model_calls) > 1:
                tool_message["model_calls"] = model_calls
        if len(self.clip_refinement_audit) > refinement_audit_start:
            tool_message["clip_refinement"] = copy.deepcopy(
                self.clip_refinement_audit[refinement_audit_start:]
            )
        if self.visual_memory_session and len(self.visual_memory_session.audit) > visual_memory_start:
            tool_message["visual_memory"] = copy.deepcopy(self.visual_memory_session.audit[visual_memory_start:])
        return tool_message, succeeded

    def _print_assistant_message(self, label: str, message: dict) -> None:
        print(label)
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            print("## == REASONING CONTENT ==")
            print(reasoning_content)
        print("## == CONTENT ==")
        print(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            print("## == TOOL CALLS ==")
            print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
        print("--------------------------------")

    def _messages_for_model(self) -> list[dict]:
        return [
            {key: value for key, value in message.items()
             if key not in {"model_call", "model_calls", "clip_refinement", "visual_memory"}}
            for message in self.messages
        ]

    def _trajectory(self, total_steps: int, finish_reason: str) -> Trajectory:
        observations = [m for m in self.messages if m.get('role') == 'tool']
        duplicates = sum(str(m.get('content', '')).startswith('Duplicate `') for m in observations)
        failures = sum(str(m.get('content', '')).startswith('Tool `') and 'execution failed.' in str(m.get('content', '')) for m in observations)
        return Trajectory(
            question=self.question or "",
            messages=self.messages,
            total_steps=total_steps,
            final_answer=self.final_answer or "",
            finish_reason=finish_reason,
            diagnostics={'sample_id': (self.memory_context or {}).get('sample_id'),
                         'skill': self.skill_audit, 'memory': self.memory_audit,
                         'visual_calls': {'executed': len(observations) - duplicates,
                                          'failed': failures, 'duplicates': duplicates},
                         **({'clip_refinement': {
                             'settings': self.clip_refinement_settings,
                             'inspections': self.clip_refinement_audit,
                         }} if refinement_enabled(self.clip_refinement_settings) else {}),
                         **({'visual_memory': {
                             'settings': self.visual_memory_session.summary(),
                             'inspections': self.visual_memory_session.audit,
                         }} if self.visual_memory_session is not None else {})},
        )

    def _answer(self, total_steps: int, finish_reason: str) -> Trajectory:
        self.messages.append(final_answer_message(finish_reason))
        assistant_message = self.model_client.chat(
            self._messages_for_model(),
            tools=[],
        )
        assistant_message.setdefault("role", "assistant")
        self.messages.append(assistant_message)
        total_steps += 1
        self.last_run_rounds = total_steps
        if self.verbose:
            self._print_assistant_message("[FINAL ANSWER] ASSISTANT", assistant_message)
        self.final_answer = (assistant_message.get("content") or "").strip()
        return self._trajectory(total_steps, finish_reason)

    @track_agent_usage
    def run(self, question: str) -> Trajectory:
        self.reset()
        self.question = question
        memory_prompt = self._retrieve_memory_prompt(question)
        logging.getLogger(__name__).info('%s', json.dumps({
            'event': 'evolution_injection', 'sample_id': (self.memory_context or {}).get('sample_id'),
            'skill': self.skill_audit, 'memory': self.memory_audit,
        }, ensure_ascii=False))
        if memory_prompt:
            self.messages[0]["content"] += f"\n\n{memory_prompt}"
        include_subtitles = self.config.get("agent", {}).get(
            "include_subtitles_in_question", True
        )
        has_initial_evidence = include_subtitles and bool(self.subtitles)
        subtitles = (
            convert_to_free_form_text_representation(
                self.subtitles, content_type="subtitle"
            )
            if include_subtitles
            else "Available through video inspection tools."
        )
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        if self.verbose:
            video_id = self.video_path.split("/")[-1].split(".")[0]
            print("--------------------------------")
            print(f"Video ID: {video_id} ({self.duration:.01f}s)")
            print("--------------------------------")
            print("Question:")
            print(question)
            print("--------------------------------")

        if not has_initial_evidence and not self.tools:
            raise VisualEvidenceRequiredError(
                "Visual evidence is required because no subtitles are available, "
                "but no visual tools are configured."
            )

        total_steps = 0
        has_visual_evidence = False
        for step in range(self.max_steps):
            visual_evidence_required = (
                not has_initial_evidence and not has_visual_evidence
            )
            available_tools = self._available_tools()
            if visual_evidence_required and not available_tools:
                raise VisualEvidenceRequiredError(
                    "No visual tools remain available without collecting any "
                    "successful visual evidence."
                )
            assistant_message = self.model_client.chat(
                self._messages_for_model(),
                tools=available_tools,
                tool_choice=(
                    "required" if visual_evidence_required
                    else "auto" if available_tools else None
                ),
            )
            assistant_message.setdefault("role", "assistant")
            self.messages.append(assistant_message)
            total_steps += 1
            self.last_run_rounds = total_steps

            tool_calls = assistant_message.get("tool_calls") or []
            if self.verbose:
                self._print_assistant_message(
                    f"[STEP {step + 1} / {self.max_steps}] ASSISTANT",
                    assistant_message,
                )

            if not tool_calls:
                if visual_evidence_required:
                    raise VisualEvidenceRequiredError(
                        "The model returned no structured visual tool call before "
                        "any visual evidence was collected."
                    )
                return self._answer(total_steps, "stop")

            for tool_call in tool_calls:
                tool_message, succeeded = self._execute_tool_call(tool_call)
                self.messages.append(tool_message)
                tool_name = tool_call.get("function", {}).get("name")
                has_visual_evidence = has_visual_evidence or (
                    succeeded
                    and tool_name in {"overview", "clip_skim", "frame_inspect"}
                )

        if not has_initial_evidence and not has_visual_evidence:
            raise VisualEvidenceRequiredError(
                "The visual tool-call limit was reached without collecting any "
                "successful visual evidence."
            )
        return self._answer(total_steps, "reach_max_steps")
