import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videospy.model import ModelClient
from .inspection import inspection_prompt, sampling_summary
from videospy.utils import (
    parse_video_timestamp,
    sampled_subtitles_text,
    validate_time_range,
)


clip_skim_tool = {
    "type": "function",
    "function": {
        "name": "clip_skim",
        "description": (
            "Browse a selected video interval by uniformly sampling up to "
            "{num_frames} frames. Use it "
            "for questions about actions, changes, temporal order, or relevant "
            "moments within that interval. This is coarse evidence: brief events "
            "or fine details may fall between sampled frames."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise question or purpose for browsing the selected "
                        "video clip."
                    ),
                },
                "start_time": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "The clip start time in seconds from the beginning of "
                        "the video."
                    ),
                },
                "end_time": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "The clip end time in seconds from the beginning of the "
                        "video. It must be greater than start_time. A value past "
                        "the video end is clipped."
                    ),
                },
            },
            "required": ["query", "start_time", "end_time"],
            "additionalProperties": False,
        },
    },
}


def _sample_frames(video_reader, start_time, end_time, num_frames):
    fps = video_reader.get_avg_fps()
    start_frame = min(max(int(start_time * fps), 0), len(video_reader) - 1)
    end_frame = min(max(int(end_time * fps), start_frame), len(video_reader) - 1)
    sample_count = min(end_frame - start_frame + 1, num_frames)
    frame_indices = np.linspace(start_frame, end_frame, sample_count).astype(int)
    timestamps = np.array(
        [index / fps for index in frame_indices],
        dtype=np.float32,
    )
    frames = video_reader.get_batch(frame_indices).asnumpy()
    return frames, timestamps


def _resize_frames(frames):
    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    return [
        np.array(
            Image.fromarray(frame).resize(
                (target_width, target_height), Image.BICUBIC
            )
        )
        for frame in frames
    ]


def _frame_content(frames, timestamps):
    content = []
    for frame, timestamp in zip(frames, timestamps):
        image_buffer = BytesIO()
        Image.fromarray(frame).save(image_buffer, format="jpeg")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        content.extend(
            [
                {"type": "text", "text": f"{timestamp:.1f}s"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
    return content


def execute_clip_skim(config: dict, parameters: dict) -> str:
    num_frames = config["num_frames"]
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    max_duration = config.get("max_duration")
    if max_duration is not None and (
        not isinstance(max_duration, (int, float))
        or isinstance(max_duration, bool)
        or max_duration <= 0
    ):
        raise ValueError(
            f"max_duration must be null or a positive number, got {max_duration!r}."
        )
    subtitle_context_seconds = config.get("subtitle_context_seconds", 0)

    start_time = parse_video_timestamp(parameters["start_time"])
    requested_end_time = parse_video_timestamp(parameters["end_time"])
    video_reader = parameters["vr"]
    video_duration = getattr(
        video_reader,
        "duration",
        len(video_reader) / video_reader.get_avg_fps(),
    )
    end_time = min(requested_end_time, video_duration)
    validate_time_range(start_time, end_time, video_duration)
    clip_duration = end_time - start_time
    if max_duration is not None and clip_duration > max_duration:
        raise ValueError(
            f"clip_skim interval must not exceed {max_duration:g} seconds; "
            f"got {clip_duration:.1f} seconds. Split the inspection into shorter "
            "intervals."
        )

    frames, timestamps = _sample_frames(
        video_reader,
        start_time,
        end_time,
        num_frames,
    )
    subtitles = sampled_subtitles_text(
        parameters["subtitles"],
        timestamps,
        enabled=config.get("use_subtitles", True),
        context_seconds=subtitle_context_seconds,
        start_time=start_time,
        end_time=end_time,
    )
    memory_frames = frames
    frames = _resize_frames(frames)
    coverage = sampling_summary(timestamps)
    content = [
        {
            "type": "text",
            "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n{coverage}",
        }
    ]
    # Image.fromarray(frames[0]).show()
    content.extend(_frame_content(frames, timestamps))
    content.append(
        {
            "type": "text",
            "text": (
                f"Video Subtitles near the sampled frames "
                f"(within ±{subtitle_context_seconds:g}s):\n{subtitles}\n\n"
                f"{inspection_prompt(parameters)}"
                "Treat the question only as an inspection goal, not as evidence. "
                "Do not accept any claim or suggested interpretation in the question "
                "unless the frames or provided subtitles directly support it. "
                "For dialogue claims, quote the subtitle, speaker, and timestamp; "
                "the speaker is not necessarily the person visible in the frame. "
                "In Facts, give one concise bullet per relevant observation, "
                "citing the displayed timestamps. "
                "Use only the provided frame or subtitle timestamps; do not infer unsupported events. "
                "State uncertainty or plausible alternatives when evidence is "
                "ambiguous. If the sampled evidence is sufficient, give a direct "
                "answer supported by the relevant timestamps. If it is insufficient, "
                "state the evidence gap and identify the candidate timestamps where "
                "the relevant evidence is clearest."
            ),
        }
    )

    memory_session = parameters.get("_visual_memory_session")
    if memory_session is not None:
        content = memory_session.augment(
            content, frames=memory_frames, timestamps=timestamps,
            parameters={**parameters, "_visual_memory_model_name": config["model"].get("model_name")},
            tool_name="clip_skim",
        )
    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    observation = response["content"]
    if end_time < requested_end_time:
        interval_status = (
            f"Interval adjusted to video bounds: requested {start_time:.1f}s - "
            f"{requested_end_time:.1f}s, inspected {start_time:.1f}s - "
            f"{end_time:.1f}s.\n"
        )
    else:
        interval_status = ""
    return (
        "Inspection mode: clip_skim\n"
        f"{interval_status}Evidence status: coarse clip inspection.\n"
        f"{coverage}"
        f"{observation}"
    )
