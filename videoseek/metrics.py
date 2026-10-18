from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    calls_without_usage: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


_active_token_usage: ContextVar[TokenUsage | None] = ContextVar(
    "active_token_usage",
    default=None,
)


def _usage_value(usage, *names: str) -> int | None:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def record_llm_response(response) -> None:
    tracker = _active_token_usage.get()
    if tracker is None:
        return

    tracker.llm_calls += 1
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        tracker.calls_without_usage += 1
        return

    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens") or 0
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens") or 0
    total_tokens = _usage_value(usage, "total_tokens")
    tracker.input_tokens += input_tokens
    tracker.output_tokens += output_tokens
    tracker.total_tokens += (
        total_tokens if total_tokens is not None else input_tokens + output_tokens
    )


@contextmanager
def track_token_usage():
    tracker = TokenUsage()
    token = _active_token_usage.set(tracker)
    try:
        yield tracker
    finally:
        _active_token_usage.reset(token)


def track_agent_usage(function):
    @wraps(function)
    def wrapper(agent, *args, **kwargs):
        agent.last_run_token_usage = TokenUsage().to_dict()
        with track_token_usage() as token_usage:
            try:
                return function(agent, *args, **kwargs)
            finally:
                agent.last_run_token_usage = token_usage.to_dict()

    return wrapper
