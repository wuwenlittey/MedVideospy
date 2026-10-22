"""Schema and explicit compatibility rules for local, image-grounded facts.

Model review is provenance, not human verification or proof of visual truth.
Task compatibility only permits use as a local reference; it never transfers
onset, global counts, stage boundaries, diagnosis, or physical measurements.
"""

from __future__ import annotations

import copy
import math
import re


FACT_TYPES = (
    "instrument_identity", "anatomy", "action", "visible_state",
    "spatial_relation", "phase", "lesion_attribute", "visible_text",
)
_LOCAL_ACTION = frozenset({"instrument_identity", "anatomy", "action", "visible_state",
                           "spatial_relation", "phase", "visible_text"})
_LESION = frozenset({"anatomy", "lesion_attribute", "visible_state", "spatial_relation", "visible_text"})
_TASK_TYPES = {
    "action": _LOCAL_ACTION, "action recognition": _LOCAL_ACTION,
    "phase": _LOCAL_ACTION, "phase recognition": _LOCAL_ACTION,
    "c1": _LOCAL_ACTION,
    "instrument": _LOCAL_ACTION, "instrument identity": _LOCAL_ACTION,
    "instrument identification": _LOCAL_ACTION, "instrument onset": _LOCAL_ACTION,
    "instrument at onset": _LOCAL_ACTION, "m1": _LOCAL_ACTION,
    "site histology": _LESION, "site and histology": _LESION,
    "histology": _LESION, "anatomical site": _LESION,
    "lesion size": _LESION | {"instrument_identity"}, "polyp size": _LESION | {"instrument_identity"},
    "lesion attribute": _LESION, "size": _LESION | {"instrument_identity"},
    # A global question may borrow local cues, never the old video's aggregate.
    "global statistics": frozenset(FACT_TYPES), "full video statistics": frozenset(FACT_TYPES),
    "whole video statistics": frozenset(FACT_TYPES), "workload": _LOCAL_ACTION,
    "instrument workload": _LOCAL_ACTION, "bbps": _LESION,
    "bowel preparation": _LESION, "quality assessment": _LESION,
}


def normalize_fact_task(task_type):
    """Normalize only the new fact route; legacy QA exact matching is unchanged."""
    value = str(task_type or "").strip().casefold()
    value = re.sub(r"^(?:medvidbench|medhorizon)[_\s:/-]+", "", value)
    return " ".join(re.sub(r"[_\-/]+", " ", value).split())


def allowed_fact_types(task_type):
    """Known task aliases have explicit capabilities; unknown tasks get none."""
    return frozenset(_TASK_TYPES.get(normalize_fact_task(task_type), ()))


def requested_fact_types(task_type, fact_types=None):
    """Validate an explicit capability request, or use the known task mapping."""
    if fact_types is None:
        return allowed_fact_types(task_type)
    if (not isinstance(fact_types, (list, tuple, set, frozenset)) or not fact_types
            or any(not isinstance(value, str) or value not in FACT_TYPES for value in fact_types)):
        raise ValueError("fact_types must be a nonempty collection of known visual fact types")
    return frozenset(fact_types)


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def validate_fact_record(record):
    """Validate a final fact record, including review and exact evidence scope.

    Does not open files, call models, or require human verification. Admission of
    model-reviewed records is a separate explicit build and retrieval policy.
    Every selected image must support this fact: IDs name the final exported
    images, not positions in an earlier, larger observation.
    """
    if not isinstance(record, dict) or record.get("record_kind") != "visual_fact":
        raise ValueError("record_kind must be visual_fact")
    result = copy.deepcopy(record)
    for key in ("case_id", "question", "answer_text", "source_video_id", "task_type", "verification_note"):
        result[key] = _required_text(result.get(key), key)
    if result.get("split") not in {"train", "dev"}:
        raise ValueError("visual facts must come from explicit train/dev sources")
    fact = result.get("fact")
    if not isinstance(fact, dict) or fact.get("type") not in FACT_TYPES:
        raise ValueError("fact.type must be a known visual fact type")
    verification = result.get("verification")
    source_grounded = isinstance(verification, dict) and verification.get("status") == "source_grounded"
    trajectory_aligned = isinstance(verification, dict) and verification.get("status") == "trajectory_aligned"
    expected_basis = "trajectory_alignment" if trajectory_aligned else "source_annotation" if source_grounded else "visual"
    if fact.get("basis") != expected_basis:
        raise ValueError(f"fact.basis must be {expected_basis} for its verification provenance")
    fact["text"] = _required_text(fact.get("text"), "fact.text")
    fact["entity"] = _required_text(fact.get("entity"), "fact.entity")
    if result["answer_text"] != fact["text"]:
        raise ValueError("answer_text must equal fact.text, not the original QA answer")
    conditions = fact.get("conditions")
    if not isinstance(conditions, list) or any(not isinstance(item, str) or not item.strip() for item in conditions):
        raise ValueError("fact.conditions must be a list of nonempty strings")
    times, paths = result.get("timestamps"), result.get("image_paths")
    if (not isinstance(times, list) or not isinstance(paths, list) or not 1 <= len(paths) <= 64
            or len(times) != len(paths)):
        raise ValueError("fact timestamps and image_paths must be nonempty aligned lists, at most 64")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("fact image_paths must contain nonempty paths")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("fact timestamps must be finite nonnegative seconds")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("fact timestamps must be strictly increasing")
    expected_scope = "frame" if len(paths) == 1 else "clip"
    if result.get("evidence_scope") != expected_scope:
        raise ValueError("fact evidence_scope must match its exact frame set")
    expected_ids = list(range(len(paths)))
    evidence_ids = fact.get("evidence_frame_ids")
    if (not isinstance(evidence_ids, list) or any(type(value) is not int for value in evidence_ids)
            or evidence_ids != expected_ids):
        raise ValueError("fact.evidence_frame_ids must be exactly 0..n-1 for final images")
    review = result.get("verification")
    if not isinstance(review, dict):
        raise ValueError("visual facts require explicit verification provenance")
    _required_text(review.get("reason"), "verification.reason")
    review_ids = review.get("evidence_frame_ids")
    if (not isinstance(review_ids, list) or any(type(value) is not int for value in review_ids)
            or review_ids != expected_ids):
        raise ValueError("verification.evidence_frame_ids must match the final fact evidence")
    if review.get("status") == "model_reviewed":
        if (result.get("verified") is not False or review.get("human_verified") is not False
                or review.get("decision") != "supported"):
            raise ValueError("model_reviewed facts must remain verified=false and explicitly supported")
        if not isinstance(review.get("model"), dict) or not review["model"]:
            raise ValueError("model_reviewed facts require a nonempty review model identity")
    elif review.get("status") == "human_verified":
        if result.get("verified") is not True or review.get("human_verified") is not True:
            raise ValueError("human_verified facts require explicit human verification")
    elif source_grounded or trajectory_aligned:
        from .grounding import validate_source_grounding

        validate_source_grounding(result)
    else:
        raise ValueError("facts require model_reviewed, human_verified, source_grounded or trajectory_aligned provenance")
    if fact["type"] == "action":
        if len(paths) < 2:
            raise ValueError("action facts require at least two ordered evidence images")
        if not source_grounded and not trajectory_aligned:
            _required_text(review.get("temporal_change"), "verification.temporal_change")
    return result


def fact_exclusion(record, context, allowed_types):
    """Return compatibility reasons; this is policy, not semantic verification."""
    if record.get("record_kind") != "visual_fact":
        return ["legacy_qa_not_cross_task_fact"]
    fact = record.get("fact", {})
    reasons = []
    if not allowed_types:
        reasons.append("unknown_fact_task_or_empty_capability")
    elif fact.get("type") not in allowed_types:
        reasons.append("fact_type_mismatch")
    # Stages and lesion characteristics are especially context-dependent. Do
    # not transfer them when either side lacks a known, matching procedure.
    if fact.get("type") in {"phase", "lesion_attribute"}:
        if not record.get("procedure") or not context.get("procedure"):
            reasons.append("context_sensitive_fact_requires_procedure")
        elif record["procedure"] != context["procedure"]:
            reasons.append("context_sensitive_fact_procedure_mismatch")
    return reasons
