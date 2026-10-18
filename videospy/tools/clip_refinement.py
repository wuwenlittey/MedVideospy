"""Optional, deterministic follow-up observations for clip_skim.

This module does not select intervals or change the agent's search policy.
"""

import hashlib

import numpy as np
from PIL import Image

from videospy.model import ModelClient
from videospy.utils import parse_video_timestamp
from .inspection import inspection_prompt, sampling_summary


DEFAULTS = {
    "high_resolution": False,
    "temporal_fill": False,
    "high_resolution_frames": 4,
    "temporal_frames": 4,
    "high_resolution_short_side": None,
    "max_clips_per_question": 1,
}


def refinement_config(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("clip_skim.refinement must be a mapping.")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown clip_skim.refinement keys: {sorted(unknown)}")
    result = {**DEFAULTS, **value}
    for key in ("high_resolution", "temporal_fill"):
        if not isinstance(result[key], bool):
            raise ValueError(f"clip_skim.refinement.{key} must be a boolean.")
    for key in ("high_resolution_frames", "temporal_frames", "max_clips_per_question"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError(f"clip_skim.refinement.{key} must be a positive integer.")
    side = result["high_resolution_short_side"]
    if side is not None and (isinstance(side, bool) or not isinstance(side, int) or side <= 0):
        raise ValueError("clip_skim.refinement.high_resolution_short_side must be null or a positive integer.")
    return result


def refinement_enabled(config):
    return config["high_resolution"] or config["temporal_fill"]


def spread_positions(length, count):
    """Select deterministic bin centers, spread across the complete sequence."""
    count = min(length, count)
    if count <= 0:
        return []
    return [int((i + 0.5) * length / count) for i in range(count)]


def gap_midpoints(frame_indices, count):
    """Return distinct, previously unseen frame IDs inside the original gaps."""
    indices = sorted(set(int(i) for i in frame_indices))
    candidates = [(a + b) // 2 for a, b in zip(indices, indices[1:]) if b - a > 1]
    return [candidates[i] for i in spread_positions(len(candidates), count)]


def _detail_frames(frames, short_side):
    result = []
    for frame in frames:
        height, width = frame.shape[:2]
        if short_side is None or min(height, width) <= short_side:
            result.append(frame)
        else:
            scale = short_side / min(height, width)
            result.append(np.asarray(Image.fromarray(frame).resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.BICUBIC,
            )))
    return result


def _frame_records(indices, raw, submitted, fps):
    return [
        {
            "frame_index": int(index),
            "timestamp": float(index / fps),
            "source_size": [int(source.shape[1]), int(source.shape[0])],
            "submitted_size": [int(image.shape[1]), int(image.shape[0])],
            "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        }
        for index, source, image in zip(indices, raw, submitted)
    ]


def refine_clip(*, config, parameters):
    """Make at most one extra VLM call; combined mode submits both frame groups.

    Subtitles remain identical to the coarse call, so temporal fill does not
    inadvertently introduce additional textual evidence.
    """
    settings = refinement_config(config.get("refinement"))
    if not refinement_enabled(settings):
        return ""
    # Import lazily: the original tool remains byte-for-byte unchanged and can
    # still be snapshotted independently by the existing RSI workflow.
    from .clip_skim import _frame_content, _resize_frames
    from videospy.utils import sampled_subtitles_text

    reader = parameters["vr"]
    fps = reader.get_avg_fps()
    duration = getattr(reader, "duration", len(reader) / fps)
    start_time = parse_video_timestamp(parameters["start_time"])
    end_time = min(parse_video_timestamp(parameters["end_time"]), duration)
    # Match clip_skim._sample_frames exactly, including integer rounding.
    first = min(max(int(start_time * fps), 0), len(reader) - 1)
    last = min(max(int(end_time * fps), first), len(reader) - 1)
    frame_indices = np.linspace(first, last, min(last - first + 1, config["num_frames"])).astype(int)
    # Preserve the coarse call's float32 timestamp representation for subtitles.
    base_times = np.asarray([i / fps for i in frame_indices], dtype=np.float32)
    subtitles_text = sampled_subtitles_text(
        parameters["subtitles"], base_times,
        enabled=config.get("use_subtitles", True),
        context_seconds=config.get("subtitle_context_seconds", 0),
        start_time=start_time, end_time=end_time,
    )
    audit = {
        "settings": settings,
        "video_path": parameters.get("_clip_refinement_video_path"),
        "query": parameters["query"],
        "interval": [float(start_time), float(end_time)],
        "base_frame_indices": [int(i) for i in frame_indices],
        "base_short_side": 256,
        "server_image_processing": "not_observable_from_client",
        "status": "pending",
        "groups": [],
    }
    sink = parameters.get("_clip_refinement_audit")
    if sink is not None:
        sink.append(audit)

    content = []
    all_timestamps = []
    try:
        if settings["high_resolution"]:
            positions = spread_positions(len(frame_indices), settings["high_resolution_frames"])
            selected = [int(frame_indices[i]) for i in positions]
            raw = reader.get_batch(selected).asnumpy()
            submitted = _detail_frames(raw, settings["high_resolution_short_side"])
            records = _frame_records(selected, raw, submitted, fps)
            # The legacy coarse path always resizes the short side to 256.
            for record in records:
                record["higher_resolution_than_coarse"] = min(record["submitted_size"]) > 256
            audit["groups"].append({"kind": "high_resolution", "frames": records})
            times = [i / fps for i in selected]
            content.append({"type": "text", "text": "Previously sampled frames, re-inspected for detail:"})
            content.extend(_frame_content(submitted, times))
            all_timestamps.extend(times)

        if settings["temporal_fill"]:
            selected = gap_midpoints(frame_indices, settings["temporal_frames"])
            if selected:
                raw = reader.get_batch(selected).asnumpy()
                submitted = _resize_frames(raw)
                records = _frame_records(selected, raw, submitted, fps)
                audit["groups"].append({"kind": "temporal_fill", "frames": records})
                times = [i / fps for i in selected]
                content.append({"type": "text", "text": "New frames from the gaps between the original samples:"})
                content.extend(_frame_content(submitted, times))
                all_timestamps.extend(times)
            else:
                audit["groups"].append({"kind": "temporal_fill", "frames": [], "skip_reason": "no_unseen_gap_frames"})

        if not all_timestamps:
            audit["status"] = "skipped"
            audit["reason"] = "no_frames_to_inspect"
            observation = "No additional frames were available; no extra model call was made."
        else:
            content.append({"type": "text", "text": (
                f"Follow-up inspection of the same clip ({start_time:.1f}s - {end_time:.1f}s).\n"
                f"{sampling_summary(all_timestamps)}"
                f"Subtitles (unchanged from coarse inspection):\n{subtitles_text}\n\n"
                f"{inspection_prompt(parameters)}"
                "Describe only the supplied visual evidence. Do not assume that an earlier "
                "interpretation or any answer option is correct. Compare temporal states only "
                "when supported by the displayed frames. Preserve uncertainty and identify "
                "the remaining evidence gap when these observations are insufficient."
            )})
            response = ModelClient(config["model"]).chat(messages=[{"role": "user", "content": content}])
            observation = response["content"]
            audit["status"] = "completed"
    except Exception as error:
        # An optional failed follow-up must not discard the successful coarse observation.
        audit["status"] = "failed"
        audit["error"] = f"{type(error).__name__}: {error}"
        observation = "Follow-up inspection failed and supplied no verified new observation."

    groups = "; ".join(
        f"{group['kind']}: " + (
            ", ".join(f"{frame['timestamp']:.3f}s" for frame in group["frames"])
            or group.get("skip_reason", "no frames")
        )
        for group in audit["groups"]
    )
    return (
        "\n\nFollow-up clip inspection:\n"
        f"Status: {audit['status']}. Frame groups: {groups}.\n"
        f"{observation}"
    )
