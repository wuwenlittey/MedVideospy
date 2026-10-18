import math
import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def current_commit_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Source archives can run without Git or repository metadata.
        return "nogit"

    commit_id = result.stdout.strip()
    if len(commit_id) < 7:
        raise RuntimeError(f"Invalid Git commit ID: {commit_id!r}.")
    return commit_id[:7]


def append_commit_id(name: str) -> str:
    return f"{name}_{current_commit_id()}"


def sanitize_run_name_segment(value: object) -> str:
    return "".join(
        character
        if character.isalnum() or character in ("-", "_", ".")
        else "_"
        for character in str(value)
    )


def parse_video_timestamp(value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"timestamp must be a number or string, got {value!r}.")

    if isinstance(value, str):
        parts = value.strip().split(":")
        if not 1 <= len(parts) <= 3 or not all(parts):
            raise ValueError(
                "timestamp must be total seconds, MM:SS, or HH:MM:SS; "
                f"got {value!r}."
            )
        try:
            seconds = float(parts[-1])
            if len(parts) >= 2:
                if not 0 <= seconds < 60:
                    raise ValueError
                minutes = int(parts[-2])
                if len(parts) == 3 and not 0 <= minutes < 60:
                    raise ValueError
                seconds += minutes * 60
            if len(parts) == 3:
                seconds += int(parts[0]) * 3600
        except ValueError as error:
            raise ValueError(
                "timestamp must be total seconds, MM:SS, or HH:MM:SS; "
                f"got {value!r}."
            ) from error
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(
            f"timestamp must be a number or string, got {type(value).__name__}."
        )

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"timestamp must be finite and non-negative, got {value!r}.")
    return seconds


def validate_time_range(
    start_time: float, end_time: float, video_duration: float
) -> None:
    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}.")
    if end_time <= start_time:
        raise ValueError(
            f"end_time must be greater than start_time, got "
            f"{start_time}-{end_time}."
        )
    if end_time > video_duration:
        raise ValueError(
            f"end_time must not exceed the video duration "
            f"({video_duration:.1f}s), got {end_time}."
        )


def select_subtitles_in_range(
    subtitles: list[dict], start_time: float, end_time: float
) -> list[dict]:
    return [
        subtitle
        for subtitle in subtitles
        if float(subtitle["start_time"]) <= end_time
        and float(subtitle["end_time"]) >= start_time
    ]


def sampled_subtitles_text(
    subtitles: list[dict],
    timestamps,
    *,
    enabled: bool = True,
    context_seconds: float = 0,
    start_time: float = 0,
    end_time: float = math.inf,
) -> str:
    if not isinstance(enabled, bool):
        raise ValueError(f"use_subtitles must be a boolean, got {enabled!r}.")
    if (
        not isinstance(context_seconds, (int, float))
        or isinstance(context_seconds, bool)
        or not math.isfinite(context_seconds)
        or context_seconds < 0
    ):
        raise ValueError(
            "subtitle_context_seconds must be a finite non-negative number, got "
            f"{context_seconds!r}."
        )
    if not enabled:
        return "Subtitles disabled."

    windows = [
        (
            max(start_time, float(timestamp) - context_seconds),
            min(end_time, float(timestamp) + context_seconds),
        )
        for timestamp in timestamps
    ]
    selected = [
        subtitle
        for subtitle in subtitles
        if any(
            float(subtitle["start_time"]) <= window_end
            and float(subtitle["end_time"]) >= window_start
            for window_start, window_end in windows  # Overlap with any sampling window.
        )
    ]
    lines = [
        f"[{float(item['start_time']):.1f}s-{float(item['end_time']):.1f}s] "
        f"{item['subtitle']}"
        for item in selected
        if item.get("subtitle") is not None
    ]
    return "\n".join(lines) if lines else "No subtitle found."


def load_subtitles(subtitle_path: str | None) -> list[dict]:
    if subtitle_path is None or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read()

    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    def to_seconds(hours, minutes, seconds, milliseconds):
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(milliseconds) / 1000.0
        )

    result = []
    for block in re.split(r"\n\n+", content.strip()):
        match = pattern.search(block)
        if match:
            result.append(
                {
                    "start_time": round(to_seconds(*match.groups()[:4]), 1),
                    "end_time": round(to_seconds(*match.groups()[4:8]), 1),
                    "subtitle": block[match.end() :].strip().replace("\n", " "),
                }
            )
    return result


def convert_to_free_form_text_representation(
    history: list[dict], content_type: str = "caption"
) -> str:
    if not history:
        return f"No {content_type} found."

    sections = []
    for item in history:
        if item[content_type] is None:
            continue
        sections.append(
            f"**Timestamp**: {item['start_time']}s - {item['end_time']}s\n"
            f"**{content_type.capitalize()}**: {item[content_type]}\n"
        )
    return "\n".join(sections)
