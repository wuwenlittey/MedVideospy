"""Run the external MedVidBench evaluator with local LLM configuration."""

import argparse
import functools
import json
import os
import runpy
import sys
import threading
from pathlib import Path


def _configure_openai(model: str, api_base: str | None, *, max_workers=2,
                      enable_thinking=False, max_completion_tokens=2048, timeout=120) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        return

    import openai

    original_openai = openai.OpenAI
    slots = threading.BoundedSemaphore(max_workers)

    class CompletionsProxy:
        def __init__(self, completions):
            self._completions = completions

        def create(self, *args, **kwargs):
            kwargs["model"] = model
            kwargs["temperature"] = 0.0
            kwargs["timeout"] = timeout
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_completion_tokens
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "enable_thinking": enable_thinking}
            with slots:
                return self._completions.create(*args, **kwargs)

    class ChatProxy:
        def __init__(self, chat):
            self._chat = chat
            self.completions = CompletionsProxy(chat.completions)

        def __getattr__(self, name):
            return getattr(self._chat, name)

    class ClientProxy:
        def __init__(self, client):
            self._client = client
            self.chat = ChatProxy(client.chat)

        def __getattr__(self, name):
            return getattr(self._client, name)

    def configured_openai(*args, **kwargs):
        if api_base:
            kwargs.setdefault("base_url", api_base)
        kwargs["max_retries"] = 0  # The official Judge already retries failed calls.
        return ClientProxy(original_openai(*args, **kwargs))

    openai.OpenAI = configured_openai


def _configure_evaluator(script: str, max_workers: int) -> None:
    """Configure dynamically loaded official modules without editing their files."""
    if not Path(script).with_name("evaluate_all_pai.py").is_file():
        return  # Standalone evaluators have no dynamically loaded task modules.
    sys.path.insert(0, str(Path(script).resolve().parent))
    import evaluate_all_pai

    original_loader = evaluate_all_pai.load_eval_module
    output_lock = threading.Lock()

    def configured_loader(name):
        module = original_loader(name)
        if name not in {"eval_dvc", "eval_caption_llm_judge"}:
            return module
        if name == "eval_dvc":
            module.DVC_MAX_WORKERS = max_workers
        else:
            module.MAX_WORKERS = max_workers
        original_judge = module.call_llm_judge_api

        @functools.wraps(original_judge)
        def tracked_judge(*args, **kwargs):
            task = kwargs.get("task_type", args[2] if len(args) > 2 else "unknown")
            result = None
            error = None
            try:
                result = original_judge(*args, **kwargs)
                return result
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                successful = bool(result and result.get("api_success"))
                event = {"task": task, "successful": successful,
                         "error": None if successful else error or (result or {}).get("error", "Incomplete Judge scores")}
                with output_lock:
                    print("VIDEOSPY_JUDGE_RESULT " + json.dumps(event), flush=True)

        module.call_llm_judge_api = tracked_judge
        return module

    evaluate_all_pai.load_eval_module = configured_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _configure_openai(args.model, args.api_base, max_workers=args.max_workers,
                      enable_thinking=args.enable_thinking,
                      max_completion_tokens=args.max_completion_tokens, timeout=args.timeout)
    _configure_evaluator(args.script, args.max_workers)
    sys.argv = [args.script, *args.arguments]
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
