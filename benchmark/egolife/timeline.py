import re
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from math import nextafter
from pathlib import Path

import numpy as np
from decord import VideoReader

from videospy.evolving.common.memory_scope import timeline_wall_seconds
from videospy.evolving.storage.artifacts import hash_json


TIMESTAMP_PATTERN = re.compile(r"DAY(\d+)_A1_JAKE_(\d{8})")
MAX_CLIP_DURATION = 30.0


def absolute_seconds(day: int, clock: str) -> float:
    value = str(clock).zfill(8)
    return (
        (day - 1) * 86400
        + int(value[:2]) * 3600
        + int(value[2:4]) * 60
        + int(value[4:6])
        + int(value[6:8]) / 100.0
    )


def parse_egolife_time(value: str | Mapping) -> float:
    if isinstance(value, Mapping):
        date = str(value.get("date", "")).upper()
        clock = str(value.get("time", ""))
        match = re.fullmatch(r"DAY(\d+)", date)
        if not match or not re.fullmatch(r"\d{8}", clock):
            raise ValueError(f"Unsupported EgoLife timestamp: {value!r}")
        return absolute_seconds(int(match.group(1)), clock)

    match = re.search(r"DAY(\d+)[@_A-Z]*(\d{8})", str(value).upper())
    if not match:
        raise ValueError(f"Unsupported EgoLife timestamp: {value!r}")
    return absolute_seconds(int(match.group(1)), match.group(2))


class _FrameBatch:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames

    def asnumpy(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


class EgoLifeTimelineReader:
    """Lazy one-frame-per-second view over contiguous recorded A1 clips."""

    def __init__(self, video_root: str | Path, cache_size: int = 8) -> None:
        root = Path(video_root).expanduser().resolve()
        clips = []
        for path in sorted(root.rglob("*.mp4")):
            match = TIMESTAMP_PATTERN.search(path.stem.upper())
            if match:
                clips.append(
                    (absolute_seconds(int(match.group(1)), match.group(2)), path)
                )
        if not clips:
            raise FileNotFoundError(f"No A1_JAKE MP4 clips found under: {root}")
        clips.sort(key=lambda item: item[0])
        self.origin = clips[0][0]
        self._wall_starts = [start for start, _ in clips]
        self._paths = [path for _, path in clips]
        self.cache_size = cache_size
        self._readers: OrderedDict[Path, VideoReader] = OrderedDict()

        durations = [
            min(next_start - start, MAX_CLIP_DURATION)
            for start, next_start in zip(self._wall_starts, self._wall_starts[1:])
        ]
        last_reader = self._reader(self._paths[-1])
        durations.append(len(last_reader) / last_reader.get_avg_fps())
        self._durations = durations

        self._timeline_starts = []
        timeline_start = 0.0
        for duration in durations:
            self._timeline_starts.append(timeline_start)
            timeline_start += duration
        self.clips = list(zip(self._timeline_starts, self._paths))
        self.full_duration = timeline_start
        self.duration = self.full_duration
        segments = [{'start': start, 'wall_start': wall, 'duration': duration, 'clip': path.name}
                    for start, wall, duration, path in zip(
                        self._timeline_starts, self._wall_starts, self._durations, self._paths)]
        self.time_mapping = {'id': hash_json(segments), 'segments': segments}

    def __len__(self) -> int:
        return max(1, int(self.duration) + 1)

    @staticmethod
    def get_avg_fps() -> float:
        return 1.0

    def to_timeline_seconds(self, absolute_time: float) -> float:
        index = bisect_right(self._wall_starts, absolute_time) - 1
        if index < 0:
            return absolute_time - self.origin
        local_seconds = min(
            max(absolute_time - self._wall_starts[index], 0.0),
            self._durations[index],
        )
        return self._timeline_starts[index] + local_seconds

    def to_wall_seconds(self, timestamp: float) -> float | None:
        return timeline_wall_seconds(self.time_mapping, timestamp)

    def set_query_time(self, query_time: str | Mapping) -> None:
        absolute_time = parse_egolife_time(query_time)
        if absolute_time < self.origin:
            raise ValueError(
                f"Query time precedes the first EgoLife clip: {query_time!r}"
            )
        duration = self.to_timeline_seconds(absolute_time)
        index = bisect_right(self._wall_starts, absolute_time) - 1
        if (
            index + 1 < len(self._wall_starts)
            and absolute_time < self._wall_starts[index + 1]
            and absolute_time >= self._wall_starts[index] + self._durations[index]
        ):
            duration = nextafter(self._timeline_starts[index + 1], 0.0)
        self.duration = min(duration, self.full_duration)

    def _reader(self, path: Path) -> VideoReader:
        reader = self._readers.pop(path, None)
        if reader is None:
            reader = VideoReader(str(path))
        self._readers[path] = reader
        while len(self._readers) > self.cache_size:
            self._readers.popitem(last=False)
        return reader

    def _clip_for_time(self, timestamp: float) -> tuple[float, Path]:
        index = max(0, bisect_right(self._timeline_starts, timestamp) - 1)
        return self.clips[index]

    def get_batch(self, frame_indices) -> _FrameBatch:
        frames = []
        for value in frame_indices:
            timestamp = float(value)
            if timestamp < 0 or timestamp > self.duration:
                raise IndexError(
                    f"Timeline timestamp must be within 0.0-{self.duration:.1f}s, "
                    f"got {timestamp:.1f}s."
                )
            clip_start, path = self._clip_for_time(timestamp)
            reader = self._reader(path)
            local_seconds = max(0.0, timestamp - clip_start)
            frame_index = min(
                int(local_seconds * reader.get_avg_fps()), len(reader) - 1
            )
            frames.append(reader.get_batch([frame_index]).asnumpy()[0])
        return _FrameBatch(frames)


__all__ = ["EgoLifeTimelineReader", "absolute_seconds", "parse_egolife_time"]
