import base64
from io import BytesIO

from PIL import Image

from videospy.model import ModelClient
from .inspection import inspection_prompt, sampling_summary
from videospy.utils import (
    parse_video_timestamp,
    sampled_subtitles_text,
)


frame_inspect_tool = {
    "type": "function",
    "function": {
        "name": "frame_inspect",
        "description": (
            "Inspect visible details in one or more video frames at selected "
            "timestamps, selecting the nearest frames without resizing them. "
            "Request at most {max_frames} timestamps per call. "
            "Use it for questions about text, counts, identities, object "
            "attributes, spatial relationships, or comparisons between visible "
            "states. Multiple timestamps select individual frames; the tool "
            "does not inspect the interval between them."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise question describing the detail to check in the "
                        "selected frame or frames."
                    ),
                },
                "timestamps": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0},
                    "minItems": 1,
                    "description": (
                        "One or more frame times in seconds from the beginning of "
                        "the video."
                    ),
                },
            },
            "required": ["query", "timestamps"],
            "additionalProperties": False,
        },
    },
}


def execute_frame_inspect(config: dict, parameters: dict) -> str:
    max_frames = config.get("max_frames")
    if (
        not isinstance(max_frames, int)
        or isinstance(max_frames, bool)
        or max_frames <= 0
    ):
        raise ValueError(f"max_frames must be a positive integer, got {max_frames!r}.")
    requested_timestamps = parameters["timestamps"]
    if not isinstance(requested_timestamps, list) or not requested_timestamps:
        raise ValueError("timestamps must be a non-empty array.")
    requested_count = len(requested_timestamps)
    timestamps = sorted(
        {
            parse_video_timestamp(timestamp)
            for timestamp in requested_timestamps[:max_frames]
        }
    )
    video_reader = parameters["vr"]
    fps = video_reader.get_avg_fps()
    video_duration = getattr(video_reader, "duration", len(video_reader) / fps)
    # Question timestamps use one decimal; agent duration is rounded to two first.
    displayed_duration = max(
        video_duration, round(video_duration, 1), round(round(video_duration, 2), 1)
    )
    subtitle_context_seconds = config.get("subtitle_context_seconds", 0)
    frame_requests = {}
    for timestamp in timestamps:
        if timestamp > displayed_duration:
            raise ValueError(
                f"timestamps must be within the video duration "
                f"(0.0s - {video_duration:.1f}s), got {timestamp}."
            )
        bounded_timestamp = min(timestamp, video_duration)
        frame_index = min(int(bounded_timestamp * fps + 0.5), len(video_reader) - 1)
        frame_requests.setdefault(frame_index, []).append(timestamp)

    frame_indices = sorted(frame_requests)
    frames = video_reader.get_batch(frame_indices).asnumpy()
    actual_timestamps = [index / fps for index in frame_indices]
    coverage = sampling_summary(actual_timestamps)
    limit_notice = ""
    if requested_count > max_frames:
        limit_notice = (
            f"Requested {requested_count} timestamps; only the first {max_frames} "
            f"in input order were used, yielding {len(frame_indices)} distinct frames. "
            "The remaining timestamps were not inspected. "
            f"frame_inspect supports at most {max_frames} timestamps per call.\n"
        )
    content = [{"type": "text", "text": limit_notice}] if limit_notice else []
    for frame, frame_index, actual_timestamp in zip(
        frames, frame_indices, actual_timestamps
    ):
        requested = ", ".join(
            f"{timestamp:.3f}s" for timestamp in frame_requests[frame_index]
        )
        subtitles = sampled_subtitles_text(
            parameters["subtitles"],
            [actual_timestamp],
            enabled=config.get("use_subtitles", True),
            context_seconds=subtitle_context_seconds,
            start_time=0,
            end_time=video_duration,
        )
        image_buffer = BytesIO()
        Image.fromarray(frame).save(image_buffer, format="jpeg")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Requested timestamp(s): {requested}\n"
                        f"Actual frame timestamp: {actual_timestamp:.3f}s\n"
                        f"Video subtitles near this frame "
                        f"(within ±{subtitle_context_seconds:g}s):\n{subtitles}\n"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"{coverage}\n{inspection_prompt(parameters)}"
                "Treat the question only as an inspection goal, not as evidence. "
                "Do not accept any claim or suggested interpretation in the question "
                "unless the frames or provided subtitles directly support it. "
                "For dialogue claims, quote the subtitle, speaker, and timestamp; "
                "the speaker is not necessarily the person visible in the frame. "
                "For each actual "
                "timestamp, first state the directly visible details. Compare frames "
                "only when the question requires it; do not infer events unsupported "
                "by frames or subtitles. For visible text, "
                "transcribe exact characters, numbers, units, and currency symbols. "
                "For counts, enumerate distinct visible objects by position. State "
                "plausible alternatives when ambiguous and keep the response "
                "concise. If the frames or subtitles provide sufficient evidence, give a direct "
                "answer supported by the relevant timestamps. If the requested "
                "visual detail is not clearly visible, explicitly report 'Insufficient "
                "visual evidence.'"
            ),
        }
    )

    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    actual_text = ", ".join(f"{timestamp:.3f}s" for timestamp in actual_timestamps)
    requested_text = ", ".join(f"{timestamp:.3f}s" for timestamp in timestamps)
    return (
        f"{limit_notice}Frame inspection at {actual_text} "
        f"(requested {requested_text}):\n{coverage}{response['content']}"
    )
