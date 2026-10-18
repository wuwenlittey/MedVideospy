import copy
import logging
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from benchmark.sampling import get_sampling_stride
from videospy.utils import append_commit_id, current_commit_id, sanitize_run_name_segment
from .storage import artifacts
from .common import training_data
from .storage.memory_store import CachedEmbedding, EvolutionStateStore, embedding_function_from_config, embedding_metadata
from .optimizer import MemoryOptimizer
from .reflector import Reflector, ReflectionRunner
from .common.schemas import EvolutionUpdate, training_status


class EvolutionPipeline:
    def __init__(
        self,
        benchmark: str,
        train_manifest: str | Path,
        output_dir: str | Path,
        config: dict,
        inference_run_dir: str | Path,
        config_paths: list[str | Path] | None = None,
        reflector=None,
        workers: int = 2,
        run_id: str | None = None,
        run_dir: str | Path | None = None,
        reuse_reflection_dir: str | Path | None = None,
    ) -> None:
        if benchmark not in {"lvbench", "egolife", "medvidbench"}:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        self.benchmark = benchmark
        self.train_manifest = Path(train_manifest).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.explicit_run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self.inference_run_dir = Path(inference_run_dir).expanduser().resolve()
        self.config = copy.deepcopy(config)
        self.config_paths = [
            Path(path).expanduser().resolve() for path in config_paths or []
        ]
        self.workers = workers
        self.reuse_reflection_dir = reuse_reflection_dir
        self.reflector = reflector or Reflector(self.config)
        self.reflector_model_source = getattr(
            self.reflector, "model_source", "agent"
        )
        self.reflector_model_config = getattr(
            self.reflector,
            "model_config",
            copy.deepcopy(self.config["agent"]["model"]),
        )
        self.memory_optimizer = MemoryOptimizer(
            dedup_threshold=self.config.get("evolving", {}).get("memory_dedup_threshold", 0.95))
        self.logger: logging.Logger | None = None
        self.inference_source: dict | None = None
        self.offline_records: list[dict] | None = None
        self.offline_config_differences: list[dict[str, object]] = []
        self.result = None
        if run_id is not None:
            self.run_id = run_id
        elif self.explicit_run_dir is not None:
            self.run_id = self.explicit_run_dir.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_run = json.loads((self.inference_run_dir / "run.json").read_text(encoding="utf-8"))
            source_model = source_run.get("agent_config", {}).get("agent", {}).get("model", {}).get("model_name")
            reflector_model = self.reflector_model_config.get("model_name")
            name_parts = [
                "source", sanitize_run_name_segment(source_model or "unknown"),
                "reflector", sanitize_run_name_segment(reflector_model or "unknown"),
                timestamp,
            ]
            self.run_id = append_commit_id("_".join(name_parts))


    @property
    def run_dir(self) -> Path:
        if self.explicit_run_dir is not None:
            return self.explicit_run_dir
        return self.output_dir / self.run_id


    @property
    def database_path(self) -> Path:
        return self.run_dir / "evolution.sqlite3"


    def _validate_inference_source(self, samples: list) -> dict:
        (
            source_metadata,
            self.offline_records,
            self.offline_config_differences,
        ) = training_data.validate_inference_source(
            self.inference_run_dir,
            self.benchmark,
            samples,
            self.config,
        )
        return source_metadata


    def _run_metadata(self) -> dict:
        identity = artifacts.config_identity
        model = self.config.get("agent", {}).get("model", {})
        reflector_model = self.reflector_model_config
        memory_config = self.config.get("tools", {}).get("memory_search") or {}
        embedding = embedding_metadata(identity(memory_config))
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "model_name": str(model.get("model_name", "")),
            "reflector_model_name": str(reflector_model.get("model_name", "")),
            "reflector_model_source": self.reflector_model_source,
            "reflector_config_hash": artifacts.hash_json(
                identity(
                    {
                        "model": reflector_model,
                        "sampling": self.config.get("evolving", {})
                        .get("reflector", {})
                        .get("sampling", {}),
                    }
                )
            ),
            "embedding_backend": embedding["backend"],
            "embedding_model_name": embedding["model_name"],
            "embedding_config_hash": embedding["config_hash"],
            "embedding_dimension": embedding["dimension"],
            "seed": model.get("seed"),
            "git_commit": current_commit_id(),
            "base_config_path": ",".join(str(path) for path in self.config_paths),
            "base_config_hash": artifacts.hash_json(
                identity(artifacts.training_config(self.config))
            ),
            "train_manifest_path": str(self.train_manifest),
            "train_manifest_hash": artifacts.hash_file(self.train_manifest),
            "epochs": 1,
            "batch_size": 0,
            "training_mode": "offline",
            "inference_source_path": (
                self.inference_source["path"] if self.inference_source else None
            ),
            "inference_source_hash": (
                self.inference_source["content_sha256"]
                if self.inference_source
                else None
            ),
        }


    def _prepare_training_inputs(self) -> list:
        if not self.train_manifest.is_file():
            raise FileNotFoundError(
                f"Evolution train manifest not found: {self.train_manifest}"
            )
        evolving = self.config.setdefault("evolving", {})
        memory_config = self.config.get("tools", {}).get("memory_search") or {}
        embedding = embedding_metadata(memory_config)
        embedding_model = memory_config.get("embedding_model") or {}
        unresolved_embedding_fields = [
            key
            for key, value in embedding_model.items()
            if isinstance(value, str)
            and value.startswith("MEMORY_EMBEDDING_")
        ]
        if unresolved_embedding_fields:
            raise ValueError(
                "Evolution embedding environment is unresolved: "
                + ", ".join(sorted(unresolved_embedding_fields))
            )
        if (
            embedding["backend"] == "hash"
            and not memory_config.get("allow_hash_embedding", False)
        ):
            raise ValueError(
                "Evolution training requires tools.memory_search.embedding_model. "
                "Set allow_hash_embedding: true only for debug runs."
            )
        samples = training_data.manifest_values(
            self.train_manifest, self.benchmark
        )
        stride = get_sampling_stride(self.benchmark, evolving.get("mode", "full"))
        evolving["sample_stride"] = stride
        samples = samples[::stride]
        self.inference_source = self._validate_inference_source(samples)
        valid_keys = set(self.inference_source["sample_keys"])
        samples = [sample for sample in samples if str(sample) in valid_keys]
        return samples


    def _prepare_run_directory(self) -> tuple[Path, bool]:
        root = self.run_dir
        if self.explicit_run_dir is None:
            root.mkdir(parents=True, exist_ok=False)
            resume = False
        else:
            root.mkdir(parents=True, exist_ok=True)
            resume = self.database_path.is_file()
            if resume:
                EvolutionStateStore(self.database_path, read_only=True).validate()
            if not resume and any(root.iterdir()):
                raise ValueError(
                    "Explicit evolution run directory is not empty and does not "
                    f"contain evolution.sqlite3: {root}"
                )
        return root, resume


    def _export(self, store, version, snapshot_name, status):
        skills = store.load_skills(version)
        memories = [{key: value for key, value in entry.items() if key not in {"embedding", "version_id"}}
                    for entry in store.load_memories(version)]
        snapshot = self.run_dir / "versions" / snapshot_name
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, value in (("skills.json", skills), ("memories.json", memories)):
            if not (snapshot / name).is_file():
                artifacts.write_json(snapshot / name, value)
            artifacts.write_json(self.run_dir / name, value)
        artifacts.write_json(self.run_dir / "final_version.json", {
            "database_path": str(self.database_path), "snapshot_version": version,
            "version_id": version if status != "failed" else None,
            "snapshot_name": snapshot_name, "status": status, "training_mode": "offline"})

    def run(self) -> Path:
        # Validate configuration before creating run artifacts or calling models.
        runner = ReflectionRunner(self.reflector, self.config, workers=self.workers,
                                         reuse_reflection_dir=self.reuse_reflection_dir)
        samples = self._prepare_training_inputs()
        root, resume = self._prepare_run_directory()
        self.logger, handlers = artifacts.create_run_logger(root / "evolution.log")
        runner.logger = self.logger
        self.reflector.logger = self.logger
        self.memory_optimizer.set_logger(self.logger)
        print(f"Evolution log: {root / 'evolution.log'}", flush=True)
        store = None
        started = False
        try:
            if self.offline_config_differences:
                self.logger.warning(training_data.format_config_difference_warning(self.offline_config_differences))
            memory_config = self.config.get("tools", {}).get("memory_search", {})
            embedding = CachedEmbedding(embedding_function_from_config(memory_config), memory_config,
                                        root / "reflection/embeddings",
                                        Path(self.reuse_reflection_dir) / "embeddings" if self.reuse_reflection_dir else None)
            store = EvolutionStateStore(self.database_path, embedding_function=embedding)
            if resume:
                stored = store.run_metadata()
                self.run_id = stored["run_id"]
                expected = self._run_metadata()
                for key in ("benchmark", "base_config_hash", "train_manifest_hash", "inference_source_path"):
                    if stored[key] != expected[key]:
                        raise ValueError(f"Resume input changed: {key}; start a new run directory")
                store.resume_run(self.run_id)
            else:
                store.create_run(self._run_metadata())
                store.create_initial_version(self.run_id, f"{self.run_id}:initial",
                                             self.config.get("TASK_SKILLS") or {},
                                             training_data.initial_hyperparameters(self.config))
            started = True
            initial = f"{self.run_id}:initial"
            artifacts.write_json(root / "inference_source.json", self.inference_source)
            artifacts.write_yaml(root / "run_config.yml", {"benchmark": self.benchmark,
                "train_manifest": str(self.train_manifest), "inference_run_dir": str(self.inference_run_dir),
                "workers": self.workers, "evolving": artifacts.redact(self.config.get("evolving", {}))})
            self.logger.info("Offline document evolution started: benchmark=%s samples=%d workers=%d", self.benchmark, len(samples), self.workers)
            records = training_data.records_for_samples(self.offline_records, samples)
            with logging_redirect_tqdm(loggers=[self.logger]):
                report, input_hash = runner.run(root / "reflection", self.inference_run_dir, records,
                                                {"skills": store.load_skills(initial)})
                report.failures.extend(self.inference_source.get("failures", []))
                memories, memory_summary = self.memory_optimizer.optimize(store, initial, report)
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
            version = f"{self.run_id}:{snapshot_name}"
            try:
                store.version_metadata(version)
            except ValueError:
                step = store.next_step_index(self.run_id)
                store.commit_version(self.run_id, initial, version, step, 1, step, input_hash, report, update)
            else:
                self.logger.info("Reusing unchanged offline snapshot: %s", version)
            self.result = {"version_id": version, "snapshot_name": snapshot_name, "status": status,
                "evaluation_version": version if status != "failed" and not evaluation_blockers else None,
                "evaluation_blockers": evaluation_blockers,
                "sample_count": len(samples), "selected_sample_count": len(report.selection["selected_sample_ids"]),
                "group_count": report.selection["group_count"], "skill_count": len(update.skills),
                "skill_characters": skill_characters,
                "domain_skill_available": has_domain_skill,
                "memory_count": len(update.memories), "stages": report.stages,
                "failures": report.failures, "changes": update.summary}
            artifacts.write_json(root / "offline_result.json", self.result)
            if status != "failed":
                store.mark_final(version)
            self._export(store, version, snapshot_name, status)
            store.finish_run(self.run_id, status)
            self.logger.info("Offline evolution %s: version=%s stages=%s", status, version, json.dumps(report.stages))
        except (Exception, KeyboardInterrupt):
            self.logger.exception("Offline evolution failed")
            if started and store is not None:
                store.finish_run(self.run_id, "failed")
            raise
        finally:
            artifacts.close_run_logger(self.logger, handlers)
        return root
