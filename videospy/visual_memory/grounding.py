"""Validate supplied labels and alignment annotations without a generative model.

These checks establish provenance and temporal containment, not medical truth.
The caller must supply an annotation that actually labels the referenced local
event/frame set. QA correctness and a model's observation are never annotation.
"""

from __future__ import annotations

import copy
import math

from .curation import _answer, _object_hash, _selection_answer, _text


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("annotation times must be finite nonnegative seconds (not MM.SS)")
    return float(value)


def validate_alignment(value):
    if not isinstance(value, dict):
        raise ValueError("missing explicit alignment annotation")
    result = copy.deepcopy(value)
    if result.get("source") not in {"dataset_annotation", "human_verified", "trajectory_llm"}:
        raise ValueError("alignment.source must be dataset_annotation, human_verified or trajectory_llm")
    if result["source"] == "trajectory_llm":
        from .trajectory_alignment import validate_alignment_evidence

        validate_alignment_evidence(result)
    result["reference"] = _text(result.get("reference"), "alignment.reference")
    if result["source"] == "human_verified":
        result["verified_by"] = _text(result.get("verified_by"), "alignment.verified_by")
    if result.get("scope") not in {"frame_set", "local_clip", "phase_interval"}:
        raise ValueError("alignment.scope must explicitly label a frame_set, local_clip or phase_interval")
    if result.get("kind") == "interval":
        result["start"], result["end"] = _seconds(result.get("start")), _seconds(result.get("end"))
        if result["end"] <= result["start"]:
            raise ValueError("alignment interval must have end > start")
    elif result.get("kind") == "timestamps":
        times = result.get("timestamps")
        if not isinstance(times, list) or not times:
            raise ValueError("alignment.timestamps must be a nonempty list")
        times = [_seconds(value) for value in times]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("annotation timestamps must be strictly increasing")
        result["timestamps"] = times
    else:
        raise ValueError("alignment.kind must be interval or timestamps")
    return result


def resolve_annotation(annotation, source):
    """Keep the supplied semantic label verbatim; never use a model prediction."""
    from .facts import FACT_TYPES

    if not isinstance(annotation, dict):
        raise ValueError("supervision must contain annotation objects")
    annotation_id = _text(annotation.get("annotation_id"), "annotation_id")
    kind = annotation.get("fact_type")
    if kind not in FACT_TYPES:
        raise ValueError("fact_type must be explicitly supplied; it is not inferred from task names")
    alignment = validate_alignment(annotation.get("alignment"))
    if alignment["scope"] == "phase_interval" and kind != "phase":
        raise ValueError("a phase annotation cannot become an instrument, anatomy or action label")
    if kind == "action" and alignment["scope"] != "local_clip":
        raise ValueError("action labels need an explicitly annotated local_clip")
    label = copy.deepcopy(annotation.get("label_source"))
    if not isinstance(label, dict):
        raise ValueError("missing trusted label_source; trajectory answers/observations are not labels")
    if label.get("kind") in {"dataset_gold", "trajectory_gold"}:
        provenance = source.get("inventory_provenance", {})
        from_trajectory = label["kind"] == "trajectory_gold"
        if from_trajectory:
            if (provenance.get("label_origin") != "trajectory" or not provenance.get("trajectory_sha256")
                    or not provenance.get("gold", {}).get("fields")):
                raise ValueError("trajectory_gold requires an inventory of explicitly mapped standard-answer fields")
            label["trajectory_provenance"] = copy.deepcopy(provenance)
        elif provenance.get("label_origin") == "trajectory":
            raise ValueError("Use trajectory_gold for saved trajectory standard answers")
        if source.get("gold_option"):
            text, _ = _selection_answer({"gold_option": source["gold_option"], "options": source.get("options")})
            label.update(gold_option=source["gold_option"], options=copy.deepcopy(source["options"]))
            if source.get("gold_answer_text") is not None and source["gold_answer_text"] != text:
                raise ValueError("saved standard text conflicts with the selected option")
        elif from_trajectory:
            text = _answer(source.get("gold_answer_text"))
        else:
            raise ValueError("dataset_gold requires gold_option and options from the dataset/source manifest")
        if "text" in label and label["text"] != text:
            raise ValueError("supplied text differs from the source standard answer")
        label["text"] = text
        if from_trajectory:
            label.setdefault("reference", "trajectory_sha256:" + provenance["trajectory_sha256"]
                             + "/qa:" + str(provenance.get("qa_id")) + "/fields:" + ",".join(provenance["gold"]["fields"]))
        elif not label.get("reference") and provenance.get("dataset_sha256"):
            label["reference"] = "dataset_sha256:" + provenance["dataset_sha256"] + "/qa:" + str(provenance.get("qa_id"))
    elif label.get("kind") in {"verified_description", "dataset_description"}:
        # A supplied description must already be verified/annotated externally.
        # An LLM-generated replacement cannot be substituted here implicitly.
        label["text"] = _answer(label.get("text"))
        if label["kind"] == "verified_description":
            label["verified_by"] = _text(label.get("verified_by"), "label_source.verified_by")
    else:
        raise ValueError("label_source.kind must be trajectory_gold, dataset_gold, dataset_description or verified_description")
    label["reference"] = _text(label.get("reference"), "label_source.reference")
    label["text"] = _answer(label["text"])
    if alignment["source"] == "trajectory_llm":
        packet = alignment["evidence"]["packet"]
        if (label["kind"] != "trajectory_gold" or label["text"] != packet["gold_answer_text"]
                or packet["trajectory_sha256"] != source["inventory_provenance"]["trajectory_sha256"]
                or str(packet["source_sample_id"]) != str(source["inventory_provenance"]["qa_id"])
                or kind != alignment["evidence"]["candidate"]["fact_type"]):
            raise ValueError("trajectory candidate differs from saved standard label or source")
    return {"annotation_id": annotation_id, "fact_type": kind, "label_source": label, "alignment": alignment}


def validate_source_grounding(record):
    grounding = record.get("grounding")
    if not isinstance(grounding, dict):
        raise ValueError("source_grounded facts require supplied label/alignment provenance")
    label = grounding.get("label_source", {})
    source = {"gold_option": label.get("gold_option"), "options": label.get("options")}
    if label.get("kind") == "trajectory_gold":
        source.update(gold_answer_text=label.get("text"), inventory_provenance=label.get("trajectory_provenance", {}))
    normalized = resolve_annotation(grounding, source)
    if normalized["label_source"]["text"] != record["fact"]["text"]:
        raise ValueError("fact text must equal the original source label; no generated elaboration")
    if normalized["fact_type"] != record["fact"]["type"]:
        raise ValueError("fact type differs from its source annotation")
    if record["fact"].get("conditions"):
        raise ValueError("source-grounded labels must not add unannotated visual conditions")
    if grounding.get("annotation_sha256") != _object_hash(normalized):
        raise ValueError("source annotation fingerprint differs")
    times = record["timestamps"]
    alignment = normalized["alignment"]
    # For explicit keyframes, assembly records exact matched decoded times.
    # A small round-to-video-frame tolerance is recorded, never an arbitrary window.
    if alignment["kind"] == "interval":
        if any(t < alignment["start"] or t > alignment["end"] for t in times):
            raise ValueError("fact includes frames outside the annotated interval")
    else:
        tolerance = grounding.get("timestamp_tolerance", 0.0)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 0.5:
            raise ValueError("invalid keyframe timestamp tolerance")
        expected = alignment["timestamps"]
        if len(times) != len(expected) or any(abs(a - b) > tolerance + 1e-9 for a, b in zip(times, expected)):
            raise ValueError("fact must contain all and only annotated keyframes")
    if normalized["alignment"]["scope"] in {"local_clip", "phase_interval"} and len(times) < 2:
        raise ValueError("clip/phase labels require multiple observed frames")
    if record["verified"] is not False or record["verification"].get("human_verified") is not False:
        raise ValueError("source_grounded records must not claim a new human verification")
    inferred = normalized["alignment"]["source"] == "trajectory_llm"
    expected_status = "trajectory_aligned" if inferred else "source_grounded"
    expected_decision = "candidate_from_trajectory" if inferred else "supported_by_source_annotation"
    if record["verification"].get("status") != expected_status:
        raise ValueError("inferred alignment must remain trajectory_aligned, not externally verified")
    if inferred:
        packet = normalized["alignment"]["evidence"]["packet"]
        if (record["source_video_id"] != packet["source_video_id"]
                or str(record.get("source_sample_id")) != str(packet["source_sample_id"])):
            raise ValueError("trajectory candidate belongs to a different source")
    if record["verification"].get("decision") != expected_decision:
        raise ValueError("source_grounded records need an annotation-based decision")
