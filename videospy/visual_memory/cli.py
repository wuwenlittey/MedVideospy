"""Human-reviewed visual memory curation and indexing commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .curation import curate_cases, ensure_new_output, validate_manifest
from .embedding import EmbeddingError


def _embedding_config(path: str) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8-sig") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise ValueError("Embedding configuration is not valid YAML.") from None
    if not isinstance(config, dict):
        raise ValueError("Embedding configuration must be a YAML mapping.")
    if "tools" in config:
        config = config.get("tools", {}).get("visual_memory", {}).get("embedding")
    elif "embedding" in config:
        config = config["embedding"]
    if not isinstance(config, dict) or not config:
        raise ValueError("Provide an embedding mapping or tools.visual_memory.embedding configuration.")
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Curate and build a human-reviewed visual case memory.")
    commands = parser.add_subparsers(dest="command", required=True)
    curate = commands.add_parser("curate", help="Export selected successful observations; always unverified.")
    curate.add_argument("--selections", required=True, help="JSONL with explicit GT, source and selected tool call.")
    curate.add_argument("--output", required=True, help="New directory; existing paths are refused.")
    validate = commands.add_parser("validate", help="Read-only case/image validation, no embeddings.")
    validate.add_argument("--cases", required=True)
    validate.add_argument("--require-verified", action="store_true")
    build = commands.add_parser("build", help="Build a new index only from reviewed train/dev cases.")
    build.add_argument("--cases", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--embedding-config", required=True)
    probe = commands.add_parser("probe", help="Check one image embedding request without calling a VLM.")
    probe.add_argument("--embedding-config", required=True)
    probe.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "curate":
            result = {"manifest": str(curate_cases(args.selections, args.output)),
                      "verified": False, "next_step": "Review cases.jsonl; set verified=true and verification_note."}
        elif args.command == "validate":
            cases = validate_manifest(args.cases, require_verified=args.require_verified)
            result = {"cases": len(cases), "verified": sum(case["verified"] for case in cases),
                      "pending": sum(not case["verified"] for case in cases)}
        elif args.command == "build":
            output = ensure_new_output(args.output)
            validate_manifest(args.cases, require_verified=True)
            config = _embedding_config(args.embedding_config)
            from .store import build_index

            build_index(Path(args.cases).resolve(), output, config)
            result = {"index": str(output)}
        else:
            from .embedding import EmbeddingClient, validate_vectors

            client = EmbeddingClient(_embedding_config(args.embedding_config))
            with Image.open(args.image) as source:
                image = source.convert("RGB")
                try:
                    vectors = validate_vectors(client.encode([image]), 1)
                finally:
                    image.close()
            result = {"identity": client.identity, "dimension": int(vectors.shape[1]),
                      "norm": float(np.linalg.norm(vectors[0].astype(np.float64)))}
    except (ValueError, OSError, KeyError, ImportError, EmbeddingError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
