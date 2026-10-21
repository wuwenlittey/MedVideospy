"""Auditable visual memory inventory, machine curation and indexing commands."""

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


def _mining_config(path: str) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8-sig") as stream:
        try:
            value = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise ValueError("Mining configuration is not valid YAML.") from None
    if not isinstance(value, dict):
        raise ValueError("Mining configuration must be a mapping.")
    value = value.get("visual_fact_mining", value)
    if not isinstance(value, dict):
        raise ValueError("visual_fact_mining must be a mapping.")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inventory trajectories, curate visual facts and build audited memory.")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Join all trajectories to dataset QA and explicit video splits; no model calls.")
    inventory.add_argument("--trajectories", required=True)
    inventory.add_argument("--dataset", required=True, help="MedHorizon video/qa JSON or flat QA JSON.")
    inventory.add_argument("--splits", required=True, help="JSONL video partitions with dataset_video_key and source_video_id.")
    inventory.add_argument("--video-root", help="Root for dataset video paths, if split rows do not supply media.")
    inventory.add_argument("--output", required=True)
    mine = commands.add_parser("mine", help="Extract and independently review facts from all successful local observations.")
    mine.add_argument("--sources", required=True, help="JSONL with explicit video identity, split and media per trajectory.")
    mine.add_argument("--config", required=True, help="VLM mining YAML; embedding config is separate.")
    mine.add_argument("--output", required=True)
    mine.add_argument("--resume", action="store_true", help="Resume only matching inputs/configuration; reuse completed checkpoints.")
    curate = commands.add_parser("curate", help="Export selected successful observations; always unverified.")
    curate.add_argument("--selections", required=True, help="JSONL with explicit GT, source and selected tool call.")
    curate.add_argument("--output", required=True, help="New directory; existing paths are refused.")
    validate = commands.add_parser("validate", help="Read-only case/image validation, no embeddings.")
    validate.add_argument("--cases", required=True)
    validate.add_argument("--require-verified", action="store_true")
    validate.add_argument("--allow-model-reviewed", action="store_true", help="Explicitly validate machine-reviewed facts for admission.")
    build = commands.add_parser("build", help="Build a new index only from reviewed train/dev cases.")
    build.add_argument("--cases", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--embedding-config", required=True)
    build.add_argument("--allow-model-reviewed", action="store_true", help="Experimental: admit supported machine-reviewed facts, preserving their provenance.")
    probe = commands.add_parser("probe", help="Check one image embedding request without calling a VLM.")
    probe.add_argument("--embedding-config", required=True)
    probe.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from .inventory import inventory_sources

            result = inventory_sources(args.trajectories, args.dataset, args.splits, args.output,
                                       video_root=args.video_root)
        elif args.command == "mine":
            from .mining import mine_facts

            config = _mining_config(args.config)
            if args.resume:
                config["resume"] = True
            result = mine_facts(args.sources, args.output, config)
        elif args.command == "curate":
            result = {"manifest": str(curate_cases(args.selections, args.output)),
                      "verified": False, "next_step": "Review cases.jsonl; set verified=true and verification_note."}
        elif args.command == "validate":
            cases = validate_manifest(args.cases, require_verified=False)
            from .facts import validate_fact_record
            from .store import validate_case

            for case in cases:
                if case.get("record_kind") == "visual_fact":
                    validate_fact_record(case)
                if args.require_verified or args.allow_model_reviewed:
                    validate_case(case, allow_model_reviewed=args.allow_model_reviewed)
            result = {"cases": len(cases), "verified": sum(case["verified"] for case in cases),
                      "model_reviewed": sum(case.get("verification", {}).get("status") == "model_reviewed" for case in cases),
                      "pending": sum(not case["verified"] and case.get("verification", {}).get("status") != "model_reviewed" for case in cases)}
        elif args.command == "build":
            output = ensure_new_output(args.output)
            cases = validate_manifest(args.cases, require_verified=False)
            config = _embedding_config(args.embedding_config)
            from .store import build_index, validate_case

            for case in cases:
                validate_case(case, max_images=config.get("max_images_per_case", 16),
                              allow_model_reviewed=args.allow_model_reviewed)
            if args.allow_model_reviewed:
                build_index(Path(args.cases).resolve(), output, config, allow_model_reviewed=True)
            else:
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
