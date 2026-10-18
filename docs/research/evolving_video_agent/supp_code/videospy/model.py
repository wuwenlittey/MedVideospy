import copy
import json
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from litellm import completion

from .metrics import record_llm_response, _usage_value


_active_model_calls: ContextVar[list[dict] | None] = ContextVar(
    "videospy_model_calls",
    default=None,
)
_model_call_input_mode: ContextVar[str] = ContextVar(
    "videospy_model_call_input_mode",
    default="compact",
)
DEFAULT_MAX_COMPLETION_TOKENS = 4096


def _compact_model_input(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact_model_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_model_input(item) for item in value]
    if isinstance(value, str) and value.startswith("data:"):
        separator = value.find(",")
        if separator != -1 and ";base64" in value[:separator]:
            payload_size = len(value) - separator - 1
            return (
                f"{value[:separator + 1]}"
                f"<omitted {payload_size} base64 characters>"
            )
    return value


@contextmanager
def collect_model_calls(input_mode: str = "compact"):
    if input_mode not in {"compact", "full"}:
        raise ValueError(
            "Model call input mode must be 'compact' or 'full', got "
            f"{input_mode!r}."
        )
    model_calls = []
    calls_token = _active_model_calls.set(model_calls)
    mode_token = _model_call_input_mode.set(input_mode)
    try:
        yield model_calls
    finally:
        _model_call_input_mode.reset(mode_token)
        _active_model_calls.reset(calls_token)


class ModelClient:
    """Configured wrapper around a chat-completions model."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.max_completion_tokens = self.config.get(
            "max_completion_tokens",
            DEFAULT_MAX_COMPLETION_TOKENS,
        )
        if (
            not isinstance(self.max_completion_tokens, int)
            or isinstance(self.max_completion_tokens, bool)
            or self.max_completion_tokens <= 0
        ):
            raise ValueError(
                "Model max_completion_tokens must be a positive integer, got "
                f"{self.max_completion_tokens!r}."
            )

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        return_json: bool = False,
        extra_body: dict | None = None,
    ) -> dict:
        delay = 1.0
        max_retries = self.config.get("max_retries", 3)
        model_calls = _active_model_calls.get()
        model_input = None
        if model_calls is not None:
            recorded_messages = (
                _compact_model_input(messages)
                if _model_call_input_mode.get() == "compact"
                else copy.deepcopy(messages)
            )
            model_input = {
                "model": self.config["model_name"],
                "messages": recorded_messages,
            }
        request_extra_body = dict(extra_body or {})
        enable_thinking = self.config.get("enable_thinking")
        if enable_thinking is not None:
            request_extra_body.setdefault("enable_thinking", enable_thinking)

        input_characters = len(json.dumps(model_input['messages'], ensure_ascii=False)) if model_input else None
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            try:
                response = completion(
                    model=self.config["model_name"],
                    messages=messages,
                    api_base=self.config.get("api_base"),
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version"),
                    max_completion_tokens=self.max_completion_tokens,
                    seed=self.config.get("seed", 42),
                    temperature=self.config.get("temperature", 0.0),
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format={"type": "json_object"} if return_json else None,
                    timeout=self.config.get('request_timeout', 900),
                    num_retries=0,  # This client owns retries; avoid multiplying SDK attempts.
                    drop_params=True,
                    extra_body=request_extra_body or None,
                )
                record_llm_response(response)
                message = self._message_to_dict(response.choices[0].message)
                if model_calls is not None:
                    model_calls.append(
                        {
                            "input": model_input,
                            "output": copy.deepcopy(message),
                            "metrics": {
                                "elapsed_seconds": time.perf_counter() - started,
                                "input_characters": input_characters,
                                "input_tokens": _usage_value(getattr(response, 'usage', None), 'prompt_tokens', 'input_tokens'),
                                "output_tokens": _usage_value(getattr(response, 'usage', None), 'completion_tokens', 'output_tokens'),
                                "finish_reason": getattr(response.choices[0], 'finish_reason', None),
                                "api_attempt": attempt + 1, "error": None,
                            },
                        }
                    )
                return message
            except Exception as error:
                if model_calls is not None:
                    model_calls.append({"input": model_input, "output": {}, "metrics": {
                        "elapsed_seconds": time.perf_counter() - started, "input_characters": input_characters,
                        "input_tokens": None, "output_tokens": None, "finish_reason": None,
                        "api_attempt": attempt + 1, "error": str(error)[:1000]}})
                if not self._is_retryable(error) or attempt == max_retries:
                    raise
                delay *= 2 * (1 + random.random())
                print(f"Retrying in {delay:.1f} seconds: {error}")
                time.sleep(delay)

        raise RuntimeError("Model request failed")

    @staticmethod
    def _message_to_dict(message: Any) -> dict:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if hasattr(message, "json"):
            return json.loads(message.json())
        if isinstance(message, dict):
            return dict(message)
        raise TypeError(f"Unsupported model message type: {type(message).__name__}")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        message = str(error).lower()
        if any(marker in message for marker in (
            "context_length_exceeded", "maximum context length", "context window",
            "input is too long", "prompt is too long", "context length exceeded",
        )):
            return False
        return any(
            marker in message
            for marker in (
                "rate limit",
                "timed out",
                "too many requests",
                "forbidden for url",
                "the maximum usage",
                "server had an error",
                "has no attribute 'upper'",
                "internal",
            )
        )
