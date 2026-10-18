"""Finish a saved reflection report using embeddings, without a reflector."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from config.videospy import build_config
from .common.schemas import EvolutionUpdate, ReflectionReport, training_status, validate_skill_document
from .optimizer import MemoryOptimizer
from .storage import artifacts
from .storage.memory_store import (
    CachedEmbedding, EvolutionStateStore, embedding_function_from_config, embedding_metadata,
)


def export_snapshot(root, store, version, snapshot_name, status):
    skills = store.load_skills(version)
    memories = [{key: value for key, value in entry.items() if key not in {"embedding", "version_id"}}
                for entry in store.load_memories(version)]
    snapshot = root / "versions" / snapshot_name
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, value in (("skills.json", skills), ("memories.json", memories)):
        if not (snapshot / name).is_file():
            artifacts.write_json(snapshot / name, value)
        artifacts.write_json(root / name, value)
    artifacts.write_json(root / "final_version.json", {
        "database_path": str(root / "evolution.sqlite3"), "snapshot_version": version,
        "version_id": version if status != "failed" else None,
        "snapshot_name": snapshot_name, "status": status, "training_mode": "offline"})


def publish_result(root, store, run_id, report, memories, memory_summary, input_hash, sample_count, logger):
    initial = f"{run_id}:initial"
    skill_characters = sum(len(skill['instruction']) for skill in report.skills.values())
    has_task_skill = any(task != '__domain__' for task in report.skills)
    has_domain_skill = '__domain__' in report.skills
    update = EvolutionUpdate(
        skills=report.skills, memories=memories,
        hyperparameters=store.load_hyperparameters(initial),
        summary={"skill": {"count": len(report.skills), "characters": skill_characters}, "memory": memory_summary})
    status = training_status(report.stages, report.failures, has_task_skill=has_task_skill)
    evaluation_blockers = []
    if not has_task_skill and any(f['stage'] in {'task_skill', 'skill_review', 'skill_revision'}
                                  for f in report.failures):
        evaluation_blockers.append('Skill generation or review failed without a published Task Skill; '
                                   'automatic evaluation is blocked. Memory and checkpoints are preserved.')
    for counts in report.stages.values():
        total = counts["succeeded"] + counts["failed"]
        counts["coverage"] = counts["succeeded"] / total if total else None
        counts['nonempty'] = counts['succeeded'] - counts['empty']
    artifacts.write_json(root / "reflection/report.json", report.to_dict())
    # Rebuild from initial + checkpoints. Never learn the same groups a
    # second time from a previous final Skill or mutate an exported snapshot.
    content_hash = artifacts.hash_json({"input": input_hash, "skills": update.skills,
        "memories": [asdict(entry) for entry in update.memories], "hyperparameters": update.hyperparameters})
    snapshot_name = content_hash[:24]
    version = f"{run_id}:{snapshot_name}"
    try:
        store.version_metadata(version)
    except ValueError:
        step = store.next_step_index(run_id)
        store.commit_version(run_id, initial, version, step, 1, step, input_hash, report, update)
    else:
        logger.info("Reusing unchanged offline snapshot: %s", version)
    result = {"version_id": version, "snapshot_name": snapshot_name, "status": status,
        "evaluation_version": version if status != "failed" and not evaluation_blockers else None,
        "evaluation_blockers": evaluation_blockers,
        "sample_count": sample_count, "selected_sample_count": len(report.selection["selected_sample_ids"]),
        "group_count": report.selection["group_count"], "skill_count": len(update.skills),
        "skill_characters": skill_characters,
        "domain_skill_available": has_domain_skill,
        "memory_count": len(update.memories), "stages": report.stages,
        "failures": report.failures, "changes": update.summary}
    artifacts.write_json(root / "offline_result.json", result)
    if status != "failed":
        store.mark_final(version)
    export_snapshot(root, store, version, snapshot_name, status)
    store.finish_run(run_id, status)
    logger.info("Offline evolution %s: version=%s stages=%s", status, version, json.dumps(report.stages))
    return result


def finalize_memories(run_dir, config):
    root = Path(run_dir).expanduser().resolve()
    database = root / 'evolution.sqlite3'
    readonly = EvolutionStateStore(database, read_only=True)
    metadata = readonly.run_metadata()
    run_id = metadata['run_id']
    initial = f'{run_id}:initial'
    readonly.version_metadata(initial)
    report = ReflectionReport(**json.loads((root / 'reflection/report.json').read_text()))
    classification = json.loads((root / 'reflection/classification.json').read_text())
    saved_config = yaml.safe_load((root / 'run_config.yml').read_text())
    for task, document in report.skills.items():
        validate_skill_document(document)
        if not document['instruction'].strip() or not document['review'] or not document['review'][-1]['approved']:
            raise ValueError(f'Cannot publish an unapproved Skill: {task}')
    memory_config = config.get('tools', {}).get('memory_search') or {}
    embedding = embedding_metadata(artifacts.config_identity(memory_config))
    for key in ('backend', 'model_name', 'config_hash', 'dimension'):
        if embedding[key] != metadata[f'embedding_{key}']:
            raise ValueError(f'Embedding {key} differs from the recorded run; use the original embedding configuration')
    # Refresh vectorization results only. Preserve every reflection failure.
    report.failures = [item for item in report.failures if item['stage'] != 'embedding']
    report.stages.pop('embedding', None)
    sample_count = sum(report.stages['classification'][key] for key in ('succeeded', 'failed'))
    input_hash = artifacts.hash_json({'classification': classification, 'selection': report.selection})
    cached = CachedEmbedding(embedding_function_from_config(memory_config), memory_config,
                             root / 'reflection/embeddings')
    store = EvolutionStateStore(database, embedding_function=cached)
    optimizer = MemoryOptimizer(saved_config.get('evolving', {}).get('memory_dedup_threshold', .95))
    logger, handlers = artifacts.create_run_logger(root / 'evolution.log')
    optimizer.set_logger(logger)
    try:
        store.resume_run(run_id)
        logger.info('Finalizing saved reflection only: skills=%d memory_candidates=%d retained_failures=%d; no reflector calls',
                    len(report.skills), len(report.memories), len(report.failures))
        memories, summary = optimizer.optimize(store, initial, report)
        return publish_result(root, store, run_id, report, memories, summary, input_hash, sample_count, logger)
    except (Exception, KeyboardInterrupt):
        store.finish_run(run_id, 'failed')
        logger.exception('Memory finalization stopped; reflection and embedding checkpoints preserved')
        raise
    finally:
        artifacts.close_run_logger(logger, handlers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Existing run whose reflection/report.json is complete. Stop its original process first.')
    parser.add_argument('--config', action='append', default=[], help='YAML configuration for the original embedding model; repeat to compose overrides.')
    args = parser.parse_args(argv)
    config = build_config(argparse.Namespace(), config_paths=[Path(path) for path in args.config])
    result = finalize_memories(args.run_dir, config)
    print(f"Finalization finished: status={result['status']} skills={result['skill_count']} memories={result['memory_count']}")
    print(f"Remaining failures: {len(result['failures'])}; version={result['version_id']}")
    print(f"Artifacts: {Path(args.run_dir).expanduser().resolve()}")
    # A usable partial snapshot does not mean the original training fully succeeded.
    return 0 if result['status'] == 'complete' and not result['evaluation_blockers'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
