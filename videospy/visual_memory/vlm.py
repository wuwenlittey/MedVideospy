"""Configurable remote VLM adapter for evidence extraction and independent review.

No model is downloaded. Credentials are resolved only in memory and never form
part of the public identity or call audit. An OpenAI-compatible chat endpoint is
used through the project's existing ModelClient.
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

from PIL import Image


class VLMError(RuntimeError):
    """A sanitized remote request or JSON-response error."""


def resolve_model_config(value=None, *, fallback=None) -> dict:
    """Resolve model/api_base/api_key env settings to ModelClient arguments.

    Explicit overrides win over their env aliases, then the fallback. Returned
    api_key must stay in memory: callers must log safe_model_identity instead.
    """
    value, fallback = value or {}, fallback or {}
    if not isinstance(value, dict) or not isinstance(fallback, dict):
        raise ValueError("VLM configuration and fallback must be mappings.")
    result = {key: fallback[key] for key in (
        "model_name", "api_base", "api_key", "api_version", "temperature", "seed",
        "max_completion_tokens", "max_retries", "request_timeout", "enable_thinking",
    ) if key in fallback}
    for field in ("model_name", "api_base", "api_key"):
        aliases = (field, "model") if field == "model_name" else (field,)
        env_aliases = ("model_name_env", "model_env") if field == "model_name" else (field + "_env",)
        for layer in (value, fallback):
            direct = next((layer[key] for key in aliases if key in layer), None)
            if direct is not None:
                result[field] = direct
                break
            env_name = next((layer[key] for key in env_aliases if key in layer), None)
            if env_name is not None:
                if not isinstance(env_name, str) or not env_name.strip():
                    raise ValueError(f"{field}_env must name an environment variable.")
                resolved = os.environ.get(env_name)
                if not resolved:
                    raise ValueError(f"Required environment variable is not set: {env_name}")
                result[field] = resolved
                break
    for key in ("api_version", "temperature", "seed", "max_completion_tokens", "max_retries",
                "request_timeout", "enable_thinking"):
        if key in value:
            result[key] = value[key]
    if not isinstance(result.get("model_name"), str) or not result["model_name"].strip():
        raise ValueError("Configure VLM model_name or model_name_env.")
    result.setdefault("temperature", 0.0)
    result.setdefault("max_completion_tokens", 4096)
    result.setdefault("max_retries", 2)
    result.setdefault("request_timeout", 180)
    return result


def safe_model_identity(config: dict) -> dict:
    """A credential-free, JSON-serializable audit identity."""
    base = config.get("api_base")
    endpoint = None
    if base:
        parsed = urlsplit(str(base))
        host = parsed.hostname or ""
        if parsed.port is not None:
            host += f":{parsed.port}"
        endpoint = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return {key: value for key, value in {
        "model_name": config.get("model_name"), "endpoint": endpoint,
        "temperature": config.get("temperature"), "seed": config.get("seed"),
        "max_completion_tokens": config.get("max_completion_tokens"),
        "api_version": config.get("api_version"),
    }.items() if value is not None}


class FactMiningVLM:
    def __init__(self, config):
        self.config = resolve_model_config(config)
        self.image_max_side = config.get("image_max_side", 2048)
        if (not isinstance(self.image_max_side, int) or isinstance(self.image_max_side, bool)
                or self.image_max_side < 64):
            raise ValueError("image_max_side must be an integer >=64.")
        self.identity = {**safe_model_identity(self.config), "image_max_side": self.image_max_side}

    def generate_json(self, prompt, images, *, purpose):
        """images is an ordered list of (original observation frame ID, PIL image)."""
        from videospy.model import ModelClient, collect_model_calls

        content = [{"type": "text", "text": prompt}]
        submitted = []
        for frame_id, original in images:
            image = original.convert("RGB")
            if max(image.size) > self.image_max_side:
                image.thumbnail((self.image_max_side, self.image_max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            submitted.append({"frame_id": frame_id, "size": list(image.size)})
            image.close()
            content.extend([
                {"type": "text", "text": f"OBSERVATION FRAME ID {frame_id}"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}},
            ])
        started = time.monotonic()
        try:
            with collect_model_calls("compact") as calls:
                result = ModelClient(self.config).chat(messages=[{"role": "user", "content": content}], return_json=True)
        except Exception as error:
            # Provider exceptions can echo credentials or base64. Never persist them.
            raise VLMError(f"{purpose} VLM request failed ({type(error).__name__}).") from None
        response = result.get("content")
        if not isinstance(response, str):
            raise VLMError(f"{purpose} VLM returned no text content.")
        stripped = response.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            payload = json.loads(stripped)
        except (ValueError, TypeError):
            raise VLMError(f"{purpose} VLM returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise VLMError(f"{purpose} VLM JSON must be an object.")
        metrics = [record.get("metrics", {}) for record in calls]
        return {"payload": payload, "model": self.identity,
                "metrics": {"elapsed_seconds": time.monotonic() - started,
                            "api_attempts": len(calls),
                            "input_tokens": sum(m.get("input_tokens") or 0 for m in metrics),
                            "output_tokens": sum(m.get("output_tokens") or 0 for m in metrics),
                            "submitted_images": submitted}}
