"""HTTP image embeddings, with explicit provider formats and no local fallback.

The default ``vllm_chat`` protocol uses ``/v1/embeddings`` with chat messages,
one request per image, for WeMM-Embedding-2B served with its embedding chat
template. ``openai`` uses ``{"model": ..., "input": [image_url_part, ...]}``.
Image embedding endpoints are not standardized across providers.
``request_template`` recursively replaces *whole-value* placeholders ``{model}``,
``{images}`` (the image_url parts), and ``{image_urls}`` (plain data URL strings).
``{image_url}`` is also available for a single-image request. ``per_image:true``
works with custom templates. ``image_item_template`` optionally changes each
image part, with the whole-value placeholder ``{image_url}``.
For example, a provider may require ``{"model": "{model}", "images":
"{image_urls}"}``. No template is evaluated as code.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image


class EmbeddingError(RuntimeError):
    """A provider failed or returned invalid image embeddings."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def validate_vectors(value: Any, expected_count: int, expected_dimension=None) -> np.ndarray:
    """Validate a finite, nonzero NxD numeric matrix without silent repair."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError("vectors must contain numbers")
        result = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError("embedding vectors must be a numeric matrix") from exc
    if result.ndim != 2 or result.shape[0] != expected_count or result.shape[1] < 1:
        raise EmbeddingError(f"expected {expected_count} embedding rows in an NxD matrix")
    if expected_dimension is not None and result.shape[1] != expected_dimension:
        raise EmbeddingError("embedding dimension changed or does not match the index")
    if not np.isfinite(result).all():
        raise EmbeddingError("embedding vectors contain NaN or infinity")
    norms = np.linalg.norm(result.astype(np.float64), axis=1)
    if np.any(norms <= 0):
        raise EmbeddingError("embedding vectors must be nonzero")
    return result


def _expand_template(value, replacements):
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        for placeholder in replacements:
            if placeholder in value:
                raise ValueError(f"template placeholder {placeholder} must occupy a whole value")
        return value
    if isinstance(value, list):
        return [_expand_template(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _expand_template(item, replacements) for key, item in value.items()}
    return value


def _response_value(response, path):
    if path in (None, "", []):
        return response
    parts = path.split(".") if isinstance(path, str) else path
    if not isinstance(parts, (list, tuple)):
        raise ValueError("response_path must be a dotted path or a list")
    current = response
    try:
        for part in parts:
            current = current[int(part)] if isinstance(current, list) else current[str(part)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EmbeddingError("embedding response_path was not found") from exc
    return current


class EmbeddingClient:
    """Encode PIL images through a configured, user-hosted HTTP service.

    ``url`` is used verbatim; ``api_base`` gets an ``/embeddings`` suffix unless
    it already has one. Authentication uses only ``api_key_env`` and is never
    included in ``identity``. ``response_path`` defaults to ``data``; each row
    can be a vector or a dictionary with ``vector_key`` (default ``embedding``).
    If indices are returned, all rows must have unique contiguous indices.
    """

    def __init__(self, config: Mapping):
        if not isinstance(config, Mapping):
            raise ValueError("embedding configuration must be a mapping")
        self.config = dict(config)
        self.request_count = 0
        self.protocol = self.config.get("protocol", "vllm_chat")
        if self.protocol not in {"vllm_chat", "openai", "custom"}:
            raise ValueError("embedding protocol must be vllm_chat, openai, or custom")
        url = self.config.get("url")
        if not url:
            base = self.config.get("api_base")
            if not base and self.config.get("api_base_env"):
                base = os.environ.get(self.config["api_base_env"])
            if not isinstance(base, str) or not base.strip():
                raise ValueError("configure embedding api_base, api_base_env, or url")
            base = base.rstrip("/")
            url = base if base.endswith("/embeddings") else base + "/embeddings"
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("configure an http(s) embedding url or api_base")
        if parsed.username or parsed.password:
            raise ValueError("put embedding credentials in api_key_env, not the URL")
        self.url = url
        self.model = self.config.get("model", "WeMM-Embedding-2B")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("embedding model must be a nonempty string")
        self.timeout = float(self.config.get("timeout", 60))
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("embedding timeout must be positive")
        self.max_retries = self.config.get("max_retries", 2)
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be an integer between 0 and 10")
        self.response_path = self.config.get("response_path", "data")
        self.vector_key = self.config.get("vector_key", "embedding")
        if not isinstance(self.vector_key, str) or not self.vector_key:
            raise ValueError("vector_key must be a nonempty string")
        template = self.config.get("request_template")
        if template is not None and not isinstance(template, (dict, list)):
            raise ValueError("request_template must be a JSON object or list")
        # Fail before making requests when the template is not JSON serializable.
        _canonical(template)
        self.template = template
        if self.protocol == "custom" and template is None:
            raise ValueError("custom protocol requires request_template")
        self.per_image = self.config.get("per_image", self.protocol == "vllm_chat")
        if not isinstance(self.per_image, bool):
            raise ValueError("per_image must be a boolean")
        if self.protocol == "vllm_chat" and not self.per_image:
            raise ValueError("vllm_chat requires one request per image")
        self.image_item_template = self.config.get("image_item_template")
        if self.image_item_template is not None and not isinstance(self.image_item_template, (dict, str)):
            raise ValueError("image_item_template must be a JSON object or string")
        _canonical(self.image_item_template)
        self._dimension = self.config.get("dimension")
        if self._dimension is not None and (isinstance(self._dimension, bool) or not isinstance(self._dimension, int) or self._dimension < 1):
            raise ValueError("embedding dimension must be a positive integer")
        safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        self.identity = {
            "adapter": "http-image-embedding-v1", "model": self.model,
            "protocol": self.protocol, "per_image": self.per_image,
            "endpoint": safe_url,
            "endpoint_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "request_template_sha256": hashlib.sha256(_canonical(template)).hexdigest(),
            "image_item_template_sha256": hashlib.sha256(_canonical(self.image_item_template)).hexdigest(),
            "response_path": self.response_path, "vector_key": self.vector_key,
            "image_encoding": "RGB-PNG-data-url",
            "dimension": self._dimension,
        }
        _canonical(self.identity)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not isinstance(images, list) or not images:
            raise ValueError("encode requires a nonempty list of PIL images")
        if any(not isinstance(picture, Image.Image) or min(picture.size) <= 0 for picture in images):
            raise ValueError("encode inputs must be nonempty PIL images")
        if self.per_image:
            return np.concatenate([self._encode_batch([picture]) for picture in images], axis=0)
        return self._encode_batch(images)

    def _encode_batch(self, images):
        urls = []
        for picture in images:
            if not isinstance(picture, Image.Image) or min(picture.size) <= 0:
                raise ValueError("encode inputs must be nonempty PIL images")
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, format="PNG")
            urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
        parts = [(_expand_template(self.image_item_template, {"{image_url}": url})
                  if self.image_item_template is not None
                  else {"type": "image_url", "image_url": {"url": url}}) for url in urls]
        replacements = {"{model}": self.model, "{images}": parts, "{image_urls}": urls}
        if len(urls) == 1:
            replacements["{image_url}"] = urls[0]
        if self.template is not None:
            payload = _expand_template(self.template, replacements)
        elif self.protocol == "vllm_chat":
            payload = {"model": self.model, "messages": [{"role": "user", "content": parts}],
                       "encoding_format": "float"}
        else:
            payload = {"model": self.model, "input": parts}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        env_name = self.config.get("api_key_env")
        if env_name:
            key = os.environ.get(env_name)
            if not key:
                raise EmbeddingError("embedding api_key_env is configured but not set")
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.url, data=_canonical(payload), headers=headers, method="POST")
        for attempt in range(self.max_retries + 1):
            try:
                self.request_count += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                if attempt == self.max_retries or exc.code not in {408, 429, 500, 502, 503, 504}:
                    # Server bodies may echo keys, URLs, or image content: do not log them.
                    raise EmbeddingError(f"embedding endpoint returned HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.max_retries:
                    raise EmbeddingError("embedding endpoint could not be reached") from None
            time.sleep(min(0.25 * 2 ** attempt, 3))
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise EmbeddingError("embedding endpoint returned invalid JSON") from None
        rows = _response_value(decoded, self.response_path)
        if not isinstance(rows, list) or len(rows) != len(images):
            raise EmbeddingError("embedding response count does not match image count")
        if rows and isinstance(rows[0], dict):
            if not all(isinstance(row, dict) for row in rows):
                raise EmbeddingError("embedding response mixes incompatible row formats")
            has_index = ["index" in row for row in rows]
            if any(has_index):
                if not all(has_index):
                    raise EmbeddingError("embedding response contains partial indices")
                indices = [row["index"] for row in rows]
                if (any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
                        or sorted(indices) != list(range(len(images)))):
                    raise EmbeddingError("embedding response indices must be unique and contiguous")
                rows = sorted(rows, key=lambda row: row["index"])
            try:
                rows = [row[self.vector_key] for row in rows]
            except KeyError:
                raise EmbeddingError("embedding response lacks the configured vector_key") from None
        result = validate_vectors(rows, len(images), self._dimension)
        self._dimension = result.shape[1]
        self.identity["dimension"] = self._dimension
        return result
