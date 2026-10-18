import hashlib
import json
import logging
from collections import Counter
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from ..common.schemas import validate_skill_document, EvolutionUpdate, ReflectionReport, MemoryEntry
from ..common.memory_scope import case_question, memory_available, memory_exclusion


SCHEMA_VERSION = 7
EmbeddingFunction = Callable[[str], list[float]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_embedding(text: str, dimension: int = 256) -> list[float]:
    """Create a deterministic dependency-free embedding for the prototype."""
    vector = [0.0] * dimension
    for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class ModelEmbedding:
    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.config.setdefault("dimensions", 1024)

    def __call__(self, text: str) -> list[float]:
        from openai import OpenAI

        model_name = str(self.config["model_name"])
        if model_name.startswith("openai/"):
            model_name = model_name.removeprefix("openai/")
        client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("api_base"),
            timeout=60,
            max_retries=3,
        )
        request = {"model": model_name, "input": [text]}
        if self.config.get("dimensions") is not None:
            request["dimensions"] = int(self.config["dimensions"])
        try:
            response = client.embeddings.create(**request)
        finally:
            client.close()
        data = response.get("data") if isinstance(response, dict) else response.data
        item = data[0]
        values = item.get("embedding") if isinstance(item, dict) else item.embedding
        return validate_embedding(values, self.config["dimensions"])


def validate_embedding(values, dimension):
    if not isinstance(values, list) or len(values) != dimension or any(
        not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError(f"Embedding API must return {dimension} finite numeric values.")
    return [float(v) for v in values]


class CachedEmbedding:
    """Reuse immutable vectors only for identical text and embedding configuration."""
    def __init__(self, function, config, directory, reuse_directory=None):
        from .artifacts import config_identity

        self.function = function
        self.metadata = embedding_metadata(config_identity(config))
        self.directory = Path(directory)
        self.reuse_directory = Path(reuse_directory) if reuse_directory else None
        self.reused = 0

    def __call__(self, text):
        from .artifacts import hash_json, write_json
        identity = {"embedding": self.metadata, "text_hash": hash_json(text)}
        key = hash_json(identity)
        path = self.directory / f"{key}.json"
        for source in (path, self.reuse_directory / path.name if self.reuse_directory else None):
            if source is not None and source.is_file():
                try:
                    stored = json.loads(source.read_text())
                    if not isinstance(stored, dict) or stored.get("identity") != identity or "vector" not in stored:
                        continue
                    vector = validate_embedding(stored["vector"], self.metadata["dimension"])
                except (OSError, ValueError) as error:
                    logging.getLogger(__name__).warning("Ignoring invalid embedding checkpoint %s: %s", source, error)
                else:
                    self.reused += 1
                    if source != path:
                        self.directory.mkdir(parents=True, exist_ok=True)
                        write_json(path, {**stored, "reused_from": str(source)})
                    return vector
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            vector = validate_embedding(self.function(text), self.metadata["dimension"])
        except Exception as exc:
            write_json(path, {"identity": identity, "error_type": type(exc).__name__})
            raise
        write_json(path, {"identity": identity, "vector": vector})
        return vector


def embedding_function_from_config(config: dict | None) -> EmbeddingFunction:
    model_config = (config or {}).get("embedding_model") or {}
    if not model_config.get("model_name"):
        return hash_embedding
    return ModelEmbedding(model_config)


def embedding_metadata(config: dict | None) -> dict:
    memory_config = config or {}
    model = memory_config.get("embedding_model") or {}
    if model.get("model_name"):
        model = {"dimensions": 1024, **model}
        safe = {
            key: value
            for key, value in model.items()
            if str(key).lower()
            not in {"api_key", "key", "token", "access_token"}
            and not str(key).lower().endswith("_key")
            and "secret" not in str(key).lower()
        }
        encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {
            "backend": "model",
            "model_name": str(model["model_name"]),
            "config_hash": hashlib.sha256(encoded).hexdigest(),
            "dimension": model.get("dimensions"),
        }
    return {
        "backend": "hash",
        "model_name": "videospy-hash-embedding",
        "config_hash": hashlib.sha256(b"videospy-hash-embedding:256").hexdigest(),
        "dimension": 256,
    }


def _flatten_mapping(value: dict, prefix: str = ""):
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            yield from _flatten_mapping(item, path)
        else:
            yield path, item


class EvolutionStateStore:
    """Versioned SQLite state shared by evolution training and inference."""

    def __init__(
        self,
        database_path: str | Path,
        embedding_function: EmbeddingFunction | None = None,
        read_only: bool = False,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.embedding_function = embedding_function or hash_embedding
        self.read_only = read_only

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro", uri=True
            )
        else:
            connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            if not self.read_only:
                connection.commit()
        except Exception:
            if not self.read_only:
                connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        if self.read_only:
            raise RuntimeError("Cannot initialize an evolution database in read-only mode.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
            if version not in {0, SCHEMA_VERSION}:
                raise ValueError(
                    f"Unsupported evolution database schema {version}; "
                    f"expected {SCHEMA_VERSION}. Regenerate from base trajectories; legacy artifacts are not migrated."
                )
            if version == 0 and tables:
                raise ValueError(
                    "Unversioned or legacy evolution database is not supported."
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evolution_runs (
                    run_id TEXT PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    reflector_model_name TEXT NOT NULL,
                    reflector_model_source TEXT NOT NULL,
                    reflector_config_hash TEXT NOT NULL,
                    embedding_backend TEXT NOT NULL,
                    embedding_model_name TEXT NOT NULL,
                    embedding_config_hash TEXT NOT NULL,
                    embedding_dimension INTEGER,
                    seed INTEGER,
                    git_commit TEXT NOT NULL,
                    base_config_path TEXT,
                    base_config_hash TEXT NOT NULL,
                    train_manifest_path TEXT NOT NULL,
                    train_manifest_hash TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    training_mode TEXT NOT NULL,
                    inference_source_path TEXT,
                    inference_source_hash TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evolution_versions (
                    version_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES evolution_runs(run_id),
                    parent_version_id TEXT REFERENCES evolution_versions(version_id),
                    step_index INTEGER NOT NULL,
                    epoch_index INTEGER NOT NULL,
                    batch_index INTEGER NOT NULL,
                    is_final INTEGER NOT NULL DEFAULT 0,
                    update_summary TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_versions_run_step
                    ON evolution_versions(run_id, step_index);

                CREATE TABLE IF NOT EXISTS reflection_reports (
                    report_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES evolution_runs(run_id),
                    version_id TEXT NOT NULL REFERENCES evolution_versions(version_id),
                    epoch_index INTEGER NOT NULL,
                    batch_index INTEGER NOT NULL,
                    input_hash TEXT NOT NULL,
                    report TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_reflection_reports_run_batch
                    ON reflection_reports(run_id, epoch_index, batch_index);

                CREATE TABLE IF NOT EXISTS skills (
                    version_id TEXT NOT NULL REFERENCES evolution_versions(version_id),
                    task_type TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    source_tasks TEXT NOT NULL,
                    review TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    PRIMARY KEY (version_id, task_type)
                );
                CREATE INDEX IF NOT EXISTS idx_skills_version_task
                    ON skills(version_id, task_type);

                CREATE TABLE IF NOT EXISTS memories (
                    version_id TEXT NOT NULL REFERENCES evolution_versions(version_id),
                    memory_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_sample_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    PRIMARY KEY (version_id, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_version_task
                    ON memories(version_id, task_type);

                CREATE TABLE IF NOT EXISTS hyperparameters (
                    version_id TEXT NOT NULL REFERENCES evolution_versions(version_id),
                    path TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (version_id, path)
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def validate(self) -> None:
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"Evolution database not found: {self.database_path}"
            )
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported evolution database schema {version}; "
                f"expected {SCHEMA_VERSION}. Regenerate from base trajectories; legacy artifacts are not migrated."
            )
        required = {
            "evolution_runs", "evolution_versions", "reflection_reports",
            "skills", "memories", "hyperparameters",
        }
        missing = required - tables
        if missing:
            raise ValueError(f"Evolution database is missing tables: {sorted(missing)}")

    def create_run(self, metadata: dict) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evolution_runs (
                    run_id, benchmark, model_name, reflector_model_name, seed,
                    reflector_model_source, reflector_config_hash,
                    embedding_backend, embedding_model_name,
                    embedding_config_hash, embedding_dimension,
                    git_commit, base_config_path, base_config_hash,
                    train_manifest_path, train_manifest_hash, epochs, batch_size,
                    training_mode, inference_source_path,
                    inference_source_hash, started_at, finished_at, status
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    metadata["run_id"], metadata["benchmark"], metadata["model_name"],
                    metadata.get("reflector_model_name", metadata["model_name"]),
                    metadata.get("seed"),
                    metadata["reflector_model_source"],
                    metadata["reflector_config_hash"],
                    metadata["embedding_backend"],
                    metadata["embedding_model_name"],
                    metadata["embedding_config_hash"],
                    metadata.get("embedding_dimension"),
                    metadata["git_commit"],
                    metadata.get("base_config_path"), metadata["base_config_hash"],
                    metadata["train_manifest_path"], metadata["train_manifest_hash"],
                    metadata["epochs"], metadata["batch_size"],
                    metadata["training_mode"],
                    metadata.get("inference_source_path"),
                    metadata.get("inference_source_hash"),
                    _now(), None, "running",
                ),
            )

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE evolution_runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, _now(), run_id),
            )

    def run_metadata(self, run_id: str | None = None) -> dict:
        self.validate()
        with self._connect() as connection:
            if run_id is None:
                rows = connection.execute("SELECT * FROM evolution_runs").fetchall()
                if len(rows) != 1:
                    raise ValueError(
                        "Evolution database must contain exactly one run to resume."
                    )
                row = rows[0]
            else:
                row = connection.execute(
                    "SELECT * FROM evolution_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
        if row is None:
            raise ValueError(f"Evolution run not found: {run_id}")
        return dict(row)

    def resume_run(self, run_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE evolution_runs SET status = 'running', finished_at = NULL "
                "WHERE run_id = ?",
                (run_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Evolution run not found: {run_id}")

    def next_step_index(self, run_id):
        with self._connect() as connection:
            return connection.execute(
                "SELECT COALESCE(MAX(step_index), 0) + 1 FROM evolution_versions WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

    def create_initial_version(
        self, run_id: str, version_id: str, task_skills: dict, hyperparameters: dict
    ) -> None:
        skill_rows = [
            (version_id, *self._skill_row(str(task_type), document))
            for task_type, document in task_skills.items()
        ]
        hyperparameter_rows = [
            (version_id, path, json.dumps(item, ensure_ascii=False))
            for path, item in _flatten_mapping(hyperparameters)
        ]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evolution_versions (
                    version_id, run_id, parent_version_id, step_index,
                    epoch_index, batch_index, is_final, created_at
                ) VALUES (?, ?, NULL, 0, 0, 0, 0, ?)
                """,
                (version_id, run_id, _now()),
            )
            connection.executemany(
                "INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?)", skill_rows
            )
            connection.executemany(
                "INSERT INTO hyperparameters VALUES (?, ?, ?)",
                hyperparameter_rows,
            )

    def mark_final(self, version_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM evolution_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Evolution version not found: {version_id}")
            connection.execute(
                "UPDATE evolution_versions SET is_final = 0 WHERE run_id = ?",
                (row["run_id"],),
            )
            connection.execute(
                "UPDATE evolution_versions SET is_final = 1 WHERE version_id = ?",
                (version_id,),
            )

    def resolve_version(self, version_id: str | None = None) -> str:
        self.validate()
        with self._connect() as connection:
            if version_id and version_id != "latest":
                row = connection.execute(
                    "SELECT version_id FROM evolution_versions WHERE version_id = ?",
                    (version_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Evolution version not found: {version_id}")
                return str(row["version_id"])
            row = connection.execute(
                """
                SELECT version_id FROM evolution_versions
                ORDER BY is_final DESC, step_index DESC, created_at DESC LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise ValueError("Evolution database has no versions.")
        return str(row["version_id"])

    def latest_version(self) -> str:
        """Latest saved snapshot, including successful work from a failed run."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version_id FROM evolution_versions ORDER BY step_index DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError("Evolution database has no versions.")
        return str(row["version_id"])

    def version_metadata(self, version_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT v.*, r.benchmark, r.model_name, r.git_commit
                FROM evolution_versions v JOIN evolution_runs r ON r.run_id = v.run_id
                WHERE v.version_id = ?
                """,
                (version_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Evolution version not found: {version_id}")
        return dict(row)

    @staticmethod
    def _skill_row(task_type, value):
        validate_skill_document(value)
        instruction = value['instruction'].strip()
        if not instruction:
            raise ValueError('Stored Skill requires a nonempty instruction')
        return (task_type,
                json.dumps(value['sources'], ensure_ascii=False),
                json.dumps(value['source_tasks'], ensure_ascii=False),
                json.dumps(value['review'], ensure_ascii=False), instruction)

    def load_skills(self, version_id: str) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM skills WHERE version_id = ? ORDER BY task_type', (version_id,)
            ).fetchall()
        return {row['task_type']: {
            'format': 'document', 'instruction': row['instruction'],
            'sources': json.loads(row['sources']), 'source_tasks': json.loads(row['source_tasks']),
            'review': json.loads(row['review']),
        } for row in rows}

    def load_hyperparameters(self, version_id: str) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT path, value FROM hyperparameters WHERE version_id = ? ORDER BY path",
                (version_id,),
            ).fetchall()
        result: dict = {}
        for row in rows:
            current = result
            keys = str(row["path"]).split(".")
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = json.loads(row["value"])
        return result

    @staticmethod
    def memory_embedding_text(entry: MemoryEntry) -> str:
        if entry.metadata.get('kind') == 'reasoning_case':
            return '\n'.join((entry.context, entry.metadata['event']))
        return "\n".join((entry.context, entry.content, entry.metadata.get("event", "")))

    def commit_version(
        self,
        run_id: str,
        parent_version_id: str,
        version_id: str,
        step_index: int,
        epoch_index: int,
        batch_index: int,
        input_hash: str,
        report: ReflectionReport,
        update: EvolutionUpdate,
    ) -> None:
        skill_rows = [
            (version_id, *self._skill_row(str(task_type), document))
            for task_type, document in update.skills.items()
        ]
        hyperparameter_rows = [
            (version_id, path, json.dumps(item, ensure_ascii=False))
            for path, item in _flatten_mapping(update.hyperparameters)
        ]
        memory_rows = [
            (
                version_id,
                entry.id,
                entry.task_type,
                entry.context,
                entry.content,
                entry.source_sample_id,
                json.dumps(entry.embedding),
                json.dumps(entry.metadata, ensure_ascii=False),
            )
            for entry in update.memories
        ]
        dimensions = {len(entry.embedding) for entry in update.memories}
        if len(dimensions) > 1:
            raise ValueError("Memory candidates use inconsistent embedding dimensions.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evolution_versions (
                    version_id, run_id, parent_version_id, step_index,
                    epoch_index, batch_index, is_final, update_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    version_id,
                    run_id,
                    parent_version_id,
                    step_index,
                    epoch_index,
                    batch_index,
                    json.dumps(update.summary, ensure_ascii=False, sort_keys=True),
                    _now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO memories
                SELECT ?, memory_id, task_type, context, content, source_sample_id, embedding, metadata
                FROM memories WHERE version_id = ?
                """,
                (version_id, parent_version_id),
            )
            connection.executemany(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                memory_rows,
            )
            connection.executemany(
                "INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?)", skill_rows
            )
            connection.executemany(
                "INSERT INTO hyperparameters VALUES (?, ?, ?)",
                hyperparameter_rows,
            )
            connection.execute(
                "INSERT INTO reflection_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    run_id,
                    version_id,
                    epoch_index,
                    batch_index,
                    input_hash,
                    json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True),
                    _now(),
                ),
            )
            if dimensions:
                dimension = next(iter(dimensions))
                row = connection.execute(
                    "SELECT embedding_dimension FROM evolution_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                existing = row["embedding_dimension"]
                if existing is not None and int(existing) != dimension:
                    raise ValueError(
                        "Memory embedding dimension changed during evolution."
                    )
                connection.execute(
                    "UPDATE evolution_runs SET embedding_dimension = ? "
                    "WHERE run_id = ?",
                    (dimension, run_id),
                )

    def search(
        self, version_id: str, query: str, task_types: str | list[str] | None, top_k: int = 3,
        similarity_threshold: float = 0.5, context: dict | None = None,
        audit: dict | None = None,
    ) -> list[dict]:
        rows = self._memory_rows(version_id, task_types)
        eligible, excluded = [], Counter()
        for row in rows:
            reason = memory_exclusion(json.loads(row['metadata']), context)
            if reason:
                excluded[reason] += 1
            else:
                eligible.append(row)
        started = time.perf_counter()
        result = []
        if eligible and top_k > 0:
            # Case retrieval matches the problem; local facts may also match option entities.
            partitions = {False: [], True: []}
            for row in eligible:
                partitions[json.loads(row['metadata']).get('kind') == 'reasoning_case'].append(row)
            for is_case, candidates in partitions.items():
                if candidates:
                    vector = self.embedding_function(case_question(query) if is_case else query)
                    # Transfer one method at a time; unrelated examples can anchor the answer.
                    limit = 1 if is_case else top_k
                    result.extend(self._rank_memories(candidates, vector, limit, similarity_threshold))
            result.sort(key=lambda item: (-item['similarity'], item['memory_id']))
            result = result[:top_k]
        detail = {
            'event': 'memory_retrieval', 'sample_id': (context or {}).get('sample_id'),
            'query_sha256': hashlib.sha256(query.encode()).hexdigest(), 'version_id': version_id,
            'candidate_count': len(rows), 'eligible_count': len(eligible), 'excluded': dict(excluded),
            'embedding_and_ranking_seconds': time.perf_counter() - started if eligible and top_k > 0 else 0,
            'hits': [{'id': r.get('memory_id'), 'similarity': r['similarity'],
                      'source_id': r['metadata']['source'].get('source_id'),
                      'relation': ('reasoning_case' if r['metadata'].get('kind') == 'reasoning_case'
                                   else 'same_source')} for r in result],
            'empty_reason': ('no_applicable_source' if not eligible else 'below_threshold_or_zero_k') if not result else None,
        }
        if audit is not None:
            audit.update(detail)
        logging.getLogger(__name__).info('%s', json.dumps(detail, ensure_ascii=False))
        return result

    def _memory_rows(self, version_id, task_types):
        if isinstance(task_types, str):
            task_types = [task_types]
        task_types = list(dict.fromkeys(task_types)) if task_types is not None else None
        placeholders = ', '.join('?' for _ in (task_types or []))
        with self._connect() as connection:
            return connection.execute(
                'SELECT * FROM memories WHERE version_id = ? '
                + (f'AND task_type IN ({placeholders})' if task_types is not None else ''),
                (version_id, *(task_types or [])),
            ).fetchall()

    def search_by_embedding(
        self, version_id: str, embedding: list[float], task_types: str | list[str] | None,
        top_k: int, similarity_threshold: float, context: dict | None = None,
    ) -> list[dict]:
        rows = [r for r in self._memory_rows(version_id, task_types)
                if memory_available(json.loads(r['metadata']), context)]
        return self._rank_memories(rows, embedding, top_k, similarity_threshold)

    @staticmethod
    def _rank_memories(rows, embedding, top_k, similarity_threshold):
        ranked = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            stored_embedding = json.loads(row["embedding"])
            if len(embedding) != len(stored_embedding):
                raise ValueError(
                    "Memory embedding dimension does not match the evolution database."
                )
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity < similarity_threshold:
                continue
            item = {
                key: row[key] for key in row.keys()
                if key not in {"embedding", "version_id"}
            }
            item["metadata"] = metadata
            item["similarity"] = similarity
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["similarity"], item["memory_id"]))
        deduplicated = []
        seen_sources = set()
        for item in ranked:
            source = (item["source_sample_id"], item['metadata'].get('kind', 'observation'))
            if source in seen_sources:
                continue
            seen_sources.add(source)
            deduplicated.append(item)
        return deduplicated[: max(0, int(top_k))]

    def load_memories(self, version_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories WHERE version_id = ? ORDER BY memory_id", (version_id,)).fetchall()
        return [{**dict(row), "embedding": json.loads(row["embedding"]),
                 "metadata": json.loads(row["metadata"])} for row in rows]

    def count_memories(self, version_id: str, task_type: str | None = None) -> int:
        with self._connect() as connection:
            if task_type is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM memories WHERE version_id = ?",
                    (version_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM memories WHERE version_id = ? AND task_type = ?",
                    (version_id, task_type),
                ).fetchone()
        return int(row["count"])


MemoryStore = EvolutionStateStore

__all__ = [
    "EmbeddingFunction", "EvolutionStateStore", "MemoryStore", "ModelEmbedding",
    "SCHEMA_VERSION", "cosine_similarity", "embedding_function_from_config",
    "embedding_metadata",
    "hash_embedding",
]
