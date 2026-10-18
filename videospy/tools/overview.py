import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videospy.model import ModelClient
from .inspection import inspection_prompt, sampling_summary
from videospy.utils import sampled_subtitles_text


def _encode_image(frame: np.ndarray) -> dict:
    image_buffer = BytesIO()
    Image.fromarray(frame).save(image_buffer, format="jpeg")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }


overview_tool = {
    "type": "function",
    "function": {
        "name": "overview",
        "description": (
            "Inspect the full video through uniformly sampled frames, with a "
            "configured sampling target of {num_frames} frames (rounded up to "
            "batches of eight and limited by available frames). Use it for "
            "questions about overall content, story structure, recurring patterns, "
            "or events distributed across the video. Call at most once per task. "
            "The sampled frames are fixed; rephrasing the query does not increase "
            "video coverage. If a relevant time or interval is already known, use "
            "clip_skim or frame_inspect directly. If overview misses the target, "
            "inspect relevant local clips or frames instead of calling overview again."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise question or purpose for inspecting the full video."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def execute_overview(config: dict, parameters: dict) -> str:
    video_reader = parameters["vr"]
    duration = round(
        getattr(
            video_reader,
            "duration",
            len(video_reader) / video_reader.get_avg_fps(),
        ),
        1,
    )
    total_frames = len(video_reader)
    target_num_frames = int(np.ceil(config["num_frames"] / 8.0) * 8)
    num_frames = (
        min(target_num_frames, total_frames // 8 * 8)
        if total_frames >= 8
        else total_frames
    )

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    timestamps = np.array(
        [index / video_reader.get_avg_fps() for index in frame_indices],
        dtype=np.float32,
    )
    subtitle_context_seconds = config.get("subtitle_context_seconds", 0)
    subtitles = sampled_subtitles_text(
        parameters["subtitles"],
        timestamps,
        enabled=config.get("use_subtitles", True),
        context_seconds=subtitle_context_seconds,
        start_time=0,
        end_time=duration,
    )
    timestamp_values = ", ".join(f"{timestamp:.1f}s" for timestamp in timestamps)
    coverage = sampling_summary(timestamps)
    frames = video_reader.get_batch(frame_indices).asnumpy()

    _, height, width, _ = frames.shape
    scale = 256 / min(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    frames = np.stack(
        [
            np.array(
                Image.fromarray(frame).resize(
                    (target_width, target_height), Image.BICUBIC
                )
            )
            for frame in frames
        ],
        axis=0,
    )

    content = [
        {
            "type": "text",
            "text": (
                f"The video segment is located at 0.0s - {duration:.1f}s:\n"
                "The video frames are uniformly sampled.\n"
                f"{coverage}"
            ),
        }
    ]
    # Image.fromarray(frames[0]).show()
    if num_frames >= 8:
        _, height, width, channels = frames.shape
        timestamp_grids = timestamps.reshape(-1, 2, 4)
        frame_grids = frames.reshape(-1, 2, 4, height, width, channels)
        frame_grids = frame_grids.transpose(0, 1, 3, 2, 4, 5).reshape(
            -1, 2 * height, 4 * width, channels
        )
        for frame, timestamp_grid in zip(frame_grids, timestamp_grids):
            row_1 = ", ".join(f"{value:.1f}s" for value in timestamp_grid[0])
            row_2 = ", ".join(f"{value:.1f}s" for value in timestamp_grid[1])
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"[{timestamp_grid[0, 0]:.1f}s - "
                            f"{timestamp_grid[-1, -1]:.1f}s]:\n"
                            f"Timestamp Matrix:\n[[{row_1}],\n[{row_2}]]\n"
                        ),
                    },
                    _encode_image(frame),
                ]
            )
    else:
        for frame, timestamp in zip(frames, timestamps):
            content.extend(
                [
                    {"type": "text", "text": f"{timestamp:.1f}s"},
                    _encode_image(frame),
                ]
            )

    content.append(
        {
            "type": "text",
            "text": (
                f"Video Subtitles near the sampled frames "
                f"(within ±{subtitle_context_seconds:g}s):\n{subtitles}\n\n"
                f"{inspection_prompt(parameters)}"
                "Treat the question only as an inspection goal, not as evidence. "
                "For dialogue claims, quote the provided subtitle, speaker, and timestamp; "
                "the speaker is not necessarily the person visible in the frame. "
                "Use the sampled frames and any provided subtitles to describe "
                "the video's overall content and temporal structure, emphasizing evidence "
                "relevant to the question. For visual claims, cite only these frame timestamps: "
                f"[{timestamp_values}]. Do not infer events unsupported by frames or subtitles. "
                "If the evidence is sufficient, give a direct answer supported by "
                "the relevant observations. If it is insufficient or ambiguous, "
                "state the evidence gap and plausible alternatives. Keep the "
                "response concise and evidence-focused."
            ),
        }
    )

    response = ModelClient(config["model"]).chat(
        messages=[{"role": "user", "content": content}]
    )
    return coverage + response["content"].strip()
