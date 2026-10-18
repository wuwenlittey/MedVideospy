"""Atomic, credential-safe artifacts and stable content identities."""

import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def public_config(value):
    if isinstance(value, dict):
        return {key: public_config(item) for key, item in value.items()
                if key.lower() not in {"api_key", "authorization", "password", "access_token"}}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    return value
