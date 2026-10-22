"""Immutable, audited visual-case memory backed by NumPy cosine search.

Builds only explicitly verified train/dev cases. A case vector is the L2-
normalized mean of its individually L2-normalized image embeddings. This is an
unordered image representation, NOT a temporal video embedding or proof that
every frame supports the case answer. Reviewers must verify the evidence scope.
"""

from __future__ import annotations

import hashlib
import copy
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .embedding import EmbeddingClient, EmbeddingError, validate_vectors
from .facts import fact_exclusion, requested_fact_types, validate_fact_record


SCHEMA_VERSION = 1
AGGREGATION = "l2_mean_of_l2_image_vectors_without_temporal_encoding"
_IDS = ("source_video_id", "source_id", "patient_id", "source_sample_id")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _text(record, key, *, required=False):
    value = record.get(key)
    if value is None and not required:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value.strip()


def _validate_case(record, max_images, *, allow_model_reviewed=False, allow_source_grounded=False, allow_trajectory_aligned=False):
    if not isinstance(record, dict):
        raise ValueError("each case must be a JSON object")
    is_fact = record.get("record_kind") == "visual_fact"
    result = validate_fact_record(record) if is_fact else dict(record)
    for key in ("case_id", "question", "answer_text", "source_video_id", "task_type", "verification_note"):
        result[key] = _text(record, key, required=True)
    if re.fullmatch(r"\(?[A-Da-d]\)?[.)]?", result["answer_text"]):
        raise ValueError("answer_text must contain answer semantics, not only an option letter")
    machine_fact = is_fact and result["verification"]["status"] == "model_reviewed"
    source_fact = is_fact and result["verification"]["status"] == "source_grounded"
    inferred_fact = is_fact and result["verification"]["status"] == "trajectory_aligned"
    if inferred_fact and not allow_trajectory_aligned:
        raise ValueError("trajectory_aligned facts require explicit allow_trajectory_aligned")
    if machine_fact and not allow_model_reviewed:
        raise ValueError("model_reviewed visual facts require explicit allow_model_reviewed")
    if source_fact and not allow_source_grounded:
        raise ValueError("source_grounded visual facts require explicit allow_source_grounded")
    if not machine_fact and not source_fact and not inferred_fact and record.get("verified") is not True:
        raise ValueError(f"case {result['case_id']} is not explicitly verified")
    if record.get("split") not in {"train", "dev"}:
        raise ValueError("visual memory only accepts train/dev cases")
    for key in (*_IDS[1:], "procedure", "previous_prediction"):
        if record.get(key) is not None:
            result[key] = _text(record, key)
    scope = record.get("evidence_scope")
    if scope not in {"frame", "clip"}:
        raise ValueError("evidence_scope must be frame or clip")
    timestamps, paths = record.get("timestamps"), record.get("image_paths")
    if not isinstance(timestamps, list) or not isinstance(paths, list) or len(paths) != len(timestamps):
        raise ValueError("timestamps and image_paths must be aligned lists")
    if not 1 <= len(paths) <= max_images:
        raise ValueError(f"each case requires 1..{max_images} reference images")
    if scope == "frame" and len(paths) != 1:
        raise ValueError("frame evidence must have exactly one image")
    if scope == "clip" and len(paths) < 2:
        raise ValueError("clip evidence requires at least two timestamped images")
    values = []
    for value in timestamps:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("timestamps must be finite nonnegative seconds")
        values.append(float(value))
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("case timestamps must be strictly increasing")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("image_paths must contain nonempty paths")
    result["timestamps"] = values
    result["image_paths"] = list(paths)
    return result


def validate_case(record, max_images=16, *, allow_model_reviewed=False, allow_source_grounded=False, allow_trajectory_aligned=False):
    """Validate a reviewed case without opening images or contacting a model."""
    if isinstance(max_images, bool) or not isinstance(max_images, int) or not 1 <= max_images <= 64:
        raise ValueError("max_images must be an integer between 1 and 64")
    if not isinstance(allow_model_reviewed, bool):
        raise ValueError("allow_model_reviewed must be a boolean")
    if not isinstance(allow_source_grounded, bool):
        raise ValueError("allow_source_grounded must be a boolean")
    if not isinstance(allow_trajectory_aligned, bool):
        raise ValueError("allow_trajectory_aligned must be a boolean")
    return _validate_case(record, max_images, allow_model_reviewed=allow_model_reviewed,
                          allow_source_grounded=allow_source_grounded, allow_trajectory_aligned=allow_trajectory_aligned)


def _inside(root: Path, relative: str) -> Path:
    """Resolve a regular file only; reject traversal and any symlink component."""
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts):
        raise ValueError("index file paths must be relative and contained in the index")
    candidate = root / path
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("index files may not be symlinks")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("index path escapes its directory") from None
    if not candidate.is_file():
        raise ValueError(f"index file is missing: {relative}")
    return candidate


def build_index(manifest_path, output_dir, embedding_config, *, embedder=None, allow_model_reviewed=False,
                allow_source_grounded=False, allow_trajectory_aligned=False):
    """Create a new immutable index; refuse existing outputs and clean staging.

    ``max_images_per_case`` in embedding_config defaults to 16 (range 1..64).
    Source paths are relative to the input JSONL. Images are copied byte-for-byte
    into ``media``; model input is RGB. The returned dictionary is manifest.json.
    """
    if not isinstance(allow_model_reviewed, bool):
        raise ValueError("allow_model_reviewed must be a boolean")
    if not isinstance(allow_source_grounded, bool):
        raise ValueError("allow_source_grounded must be a boolean")
    if not isinstance(allow_trajectory_aligned, bool):
        raise ValueError("allow_trajectory_aligned must be a boolean")
    source, output = Path(manifest_path).resolve(), Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"index output already exists: {output}")
    settings = dict(embedding_config)
    max_images = settings.get("max_images_per_case", 16)
    if isinstance(max_images, bool) or not isinstance(max_images, int) or not 1 <= max_images <= 64:
        raise ValueError("max_images_per_case must be an integer between 1 and 64")
    source_bytes = source.read_bytes()
    records, seen = [], set()
    for line_number, line in enumerate(source_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = _validate_case(json.loads(line), max_images, allow_model_reviewed=allow_model_reviewed,
                                  allow_source_grounded=allow_source_grounded, allow_trajectory_aligned=allow_trajectory_aligned)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid visual memory case on line {line_number}: {exc}") from exc
        if case["case_id"] in seen:
            raise ValueError("duplicate case_id in visual memory manifest")
        seen.add(case["case_id"])
        records.append(case)
    if not records:
        raise ValueError("visual memory manifest contains no cases")
    client = embedder if embedder is not None else EmbeddingClient(settings)
    identity = client.identity
    _json(identity)  # Identity must be serializable; the HTTP adapter omits keys.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    created_output = False
    try:
        (staging / "media").mkdir()
        hashes, vectors, dimension = {}, [], None
        for case in records:
            images, copied = [], []
            for image_number, image_path in enumerate(case["image_paths"]):
                original = Path(image_path)
                if not original.is_absolute():
                    original = source.parent / original
                raw = original.read_bytes()
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        images.append(image.convert("RGB").copy())
                except (OSError, ValueError) as exc:
                    raise ValueError(f"invalid evidence image for case {case['case_id']}") from exc
                suffix = original.suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                    suffix = ".image"
                name = f"{_sha(case['case_id'].encode('utf-8'))}-{image_number}{suffix}"
                relative = "media/" + name
                (staging / relative).write_bytes(raw)
                hashes[relative] = _sha(raw)
                copied.append(relative)
            try:
                encoded = validate_vectors(client.encode(images), len(images), dimension)
            finally:
                for image in images:
                    image.close()
            dimension = encoded.shape[1]
            encoded = encoded.astype(np.float64)
            encoded /= np.linalg.norm(encoded, axis=1, keepdims=True)
            mean = encoded.mean(axis=0)
            norm = np.linalg.norm(mean)
            if not np.isfinite(norm) or norm <= 1e-12:
                raise EmbeddingError("case image embeddings cancel to an invalid mean")
            vectors.append((mean / norm).astype(np.float32))
            case["image_paths"] = copied
        record_bytes = b"".join(_json(case) + b"\n" for case in records)
        (staging / "records.jsonl").write_bytes(record_bytes)
        np.save(staging / "vectors.npy", np.stack(vectors), allow_pickle=False)
        hashes["records.jsonl"] = _sha(record_bytes)
        hashes["vectors.npy"] = _sha((staging / "vectors.npy").read_bytes())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(records), "dimension": dimension,
            "aggregation": AGGREGATION, "embedder": identity,
            "build_settings": {"max_images_per_case": max_images},
            "admission_policy": {"allow_model_reviewed": allow_model_reviewed,
                                 "allow_source_grounded": allow_source_grounded,
                                 "allow_trajectory_aligned": allow_trajectory_aligned},
            "source_manifest_sha256": _sha(source_bytes), "files": hashes,
        }
        (staging / "manifest.json").write_bytes(_json(manifest) + b"\n")
        # Exclusive reservation avoids replacing even an existing empty directory.
        output.mkdir(exist_ok=False)
        created_output = True
        for name in ("media", "records.jsonl", "vectors.npy", "manifest.json"):
            (staging / name).rename(output / name)
        return manifest
    except BaseException:
        if created_output and not output.is_symlink():
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(staging)


class VisualMemoryStore:
    """Read-only verified-case search, with mandatory leakage-aware context."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads(_inside(self.path, "manifest.json").read_bytes())
        manifest = self.manifest
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("aggregation") != AGGREGATION:
            raise ValueError("unsupported visual memory schema or aggregation")
        if (isinstance(manifest.get("count"), bool) or not isinstance(manifest.get("count"), int)
                or manifest["count"] < 1 or isinstance(manifest.get("dimension"), bool)
                or not isinstance(manifest.get("dimension"), int) or manifest["dimension"] < 1):
            raise ValueError("index manifest requires positive count and dimension")
        if not isinstance(manifest.get("embedder"), dict):
            raise ValueError("index manifest is missing the embedder identity")
        files = manifest.get("files")
        if not isinstance(files, dict) or not {"records.jsonl", "vectors.npy"} <= files.keys():
            raise ValueError("index manifest is missing file hashes")
        for name, expected in files.items():
            if not isinstance(expected, str) or _sha(_inside(self.path, name).read_bytes()) != expected:
                raise ValueError(f"index hash mismatch: {name}")
        self.records = []
        ids, referenced = set(), {"records.jsonl", "vectors.npy"}
        max_images = manifest.get("build_settings", {}).get("max_images_per_case", 16)
        if isinstance(max_images, bool) or not isinstance(max_images, int) or not 1 <= max_images <= 64:
            raise ValueError("index manifest has an invalid maximum image count")
        policy = manifest.get("admission_policy", {"allow_model_reviewed": False})
        if not isinstance(policy, dict) or not isinstance(policy.get("allow_model_reviewed"), bool):
            raise ValueError("index manifest has an invalid admission policy")
        self.allow_model_reviewed = policy["allow_model_reviewed"]
        self.allow_trajectory_aligned = policy.get("allow_trajectory_aligned", False)
        if not isinstance(self.allow_trajectory_aligned, bool):
            raise ValueError("invalid trajectory-aligned admission policy")
        self.allow_source_grounded = policy.get("allow_source_grounded", False)
        if not isinstance(self.allow_source_grounded, bool):
            raise ValueError("index manifest has an invalid source-grounded admission policy")
        for line in _inside(self.path, "records.jsonl").read_text(encoding="utf-8").splitlines():
            case = _validate_case(json.loads(line), max_images, allow_model_reviewed=self.allow_model_reviewed,
                                  allow_source_grounded=self.allow_source_grounded, allow_trajectory_aligned=self.allow_trajectory_aligned)
            if case["case_id"] in ids:
                raise ValueError("duplicate case_id in index")
            ids.add(case["case_id"])
            for name in case["image_paths"]:
                if not name.startswith("media/") or name not in files:
                    raise ValueError("case evidence is not a hashed media file")
                _inside(self.path, name)
                referenced.add(name)
            self.records.append(case)
        if referenced != set(files):
            raise ValueError("index manifest has unreferenced or missing files")
        if len(self.records) != manifest.get("count") or not self.records:
            raise ValueError("index case count does not match its manifest")
        self.vectors = np.load(_inside(self.path, "vectors.npy"), allow_pickle=False, mmap_mode="r")
        validate_vectors(self.vectors, len(self.records), manifest.get("dimension"))
        norms = np.linalg.norm(self.vectors.astype(np.float64), axis=1)
        if self.vectors.dtype != np.float32 or not np.allclose(norms, 1, atol=1e-5):
            raise ValueError("index vectors must be normalized float32 values")
        self.vectors.flags.writeable = False
        self.dimension = self.vectors.shape[1]
        self.identity = manifest["embedder"]

    def search(self, query_vector, *, context, top_k=2, min_similarity=0.0,
               match_mode="task", fact_types=None, allow_model_reviewed=False, allow_source_grounded=False, allow_trajectory_aligned=False):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(min_similarity, bool) or not isinstance(min_similarity, (float, int)) or not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        if not isinstance(context, dict):
            raise ValueError("search context must be a dictionary")
        if match_mode not in {"task", "fact"}:
            raise ValueError("match_mode must be task or fact")
        if not isinstance(allow_model_reviewed, bool):
            raise ValueError("allow_model_reviewed must be a boolean")
        if not isinstance(allow_source_grounded, bool):
            raise ValueError("allow_source_grounded must be a boolean")
        if not isinstance(allow_trajectory_aligned, bool):
            raise ValueError("allow_trajectory_aligned must be a boolean")
        if match_mode == "task" and fact_types is not None:
            raise ValueError("fact_types requires explicit match_mode=fact")
        current = {key: _text(context, key, required=key in {"source_video_id", "task_type"})
                   for key in (*_IDS, "task_type", "procedure")}
        allowed_types = requested_fact_types(current["task_type"], fact_types) if match_mode == "fact" else None
        raw = np.asarray(query_vector)
        if raw.ndim == 1:
            raw = raw[None, :]
        query = validate_vectors(raw, 1, self.dimension).astype(np.float64)[0]
        query /= np.linalg.norm(query)
        scores = np.clip(self.vectors.astype(np.float64) @ query, -1, 1)
        eligible, exclusions = [], []
        for case, score in zip(self.records, scores):
            reasons = []
            machine_fact = (case.get("record_kind") == "visual_fact"
                            and case.get("verification", {}).get("status") == "model_reviewed")
            source_fact = (case.get("record_kind") == "visual_fact"
                           and case.get("verification", {}).get("status") == "source_grounded")
            inferred_fact = (case.get("record_kind") == "visual_fact"
                             and case.get("verification", {}).get("status") == "trajectory_aligned")
            if inferred_fact and not allow_trajectory_aligned:
                reasons.append("trajectory_aligned_not_enabled")
            if machine_fact and not allow_model_reviewed:
                reasons.append("model_reviewed_not_enabled")
            if source_fact and not allow_source_grounded:
                reasons.append("source_grounded_not_enabled")
            if ((case.get("verified") is not True and not (machine_fact and allow_model_reviewed)
                 and not (source_fact and allow_source_grounded) and not (inferred_fact and allow_trajectory_aligned))
                    or case.get("split") not in {"train", "dev"}):
                reasons.append("unverified_or_nontraining_case")
            for field in _IDS:
                if current[field] is not None and case.get(field) is not None and current[field] == str(case[field]):
                    reasons.append("same_" + field)
            if match_mode == "fact":
                reasons.extend(fact_exclusion(case, current, allowed_types))
            elif case["task_type"] != current["task_type"]:
                reasons.append("task_type_mismatch")
            if case.get("procedure") and current["procedure"] and case["procedure"] != current["procedure"]:
                reasons.append("procedure_mismatch")
            if score < min_similarity:
                reasons.append("below_min_similarity")
            if reasons:
                exclusions.append({"case_id": case["case_id"], "reasons": reasons})
            else:
                eligible.append((float(score), case))
        eligible.sort(key=lambda item: (-item[0], item[1]["case_id"]))
        matches, videos = [], set()
        for score, case in eligible:
            if case["source_video_id"] in videos:
                exclusions.append({"case_id": case["case_id"], "reasons": ["duplicate_source_video"]})
                continue
            if len(matches) >= top_k:
                continue
            videos.add(case["source_video_id"])
            resolved = copy.deepcopy(case)
            resolved["image_paths"] = [str(_inside(self.path, name).resolve()) for name in case["image_paths"]]
            matches.append({"case_id": case["case_id"], "score": score, "case": resolved})
        exclusions.sort(key=lambda item: item["case_id"])
        return {"matches": matches, "exclusions": exclusions}
