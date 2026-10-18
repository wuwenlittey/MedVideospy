from collections.abc import Iterable
from pathlib import Path


try:
    from decord import VideoReader
except ModuleNotFoundError:  # pragma: no cover - tests may stub decord
    VideoReader = None


def parse_timestamp(value) -> float:
    if value is None:
        raise ValueError("Query timestamp is missing.")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Unsupported query timestamp: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Query timestamp must be non-negative: {value!r}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("Query timestamp is empty.")
    if ":" in text:
        parts = text.split(":")
        if not 2 <= len(parts) <= 3:
            raise ValueError(f"Unsupported clock timestamp: {value!r}")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
        return seconds
    seconds = float(text)
    if seconds < 0:
        raise ValueError(f"Query timestamp must be non-negative: {value!r}")
    return seconds


class QueryCutoffVideoReader:
    """Single-file VideoReader that hides frames after the current query time.

    VideoSpy tools sample with both `duration` and `len(video_reader)`. EgoLife
    already truncates both; this adapter does the same for one MP4.
    """

    def __init__(self, video_path: str | Path, video_reader=None) -> None:
        path = Path(video_path).expanduser().resolve()
        if video_reader is None:
            if VideoReader is None:
                raise ModuleNotFoundError("decord is required to read streaming videos.")
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            video_reader = VideoReader(str(path))
        self.video_path = str(path)
        self._reader = video_reader
        fps = float(self._reader.get_avg_fps())
        if fps <= 0:
            raise ValueError(f"Invalid FPS for video: {path}")
        self._fps = fps
        self._full_frames = len(self._reader)
        if self._full_frames <= 0:
            raise ValueError(f"Empty video: {path}")
        self.full_duration = self._full_frames / self._fps
        self.duration = self.full_duration

    def get_avg_fps(self) -> float:
        return self._fps

    def set_query_time(self, query_time) -> None:
        seconds = parse_timestamp(query_time)
        self.duration = min(seconds, self.full_duration)

    def __len__(self) -> int:
        last_index = min(int(self.duration * self._fps), self._full_frames - 1)
        return max(1, last_index + 1)

    def get_batch(self, frame_indices: Iterable):
        max_index = len(self) - 1
        indices = []
        for value in frame_indices:
            index = int(value)
            if index < 0 or index > max_index:
                raise IndexError(
                    f"Frame index must be within 0-{max_index} "
                    f"(query cutoff {self.duration:.1f}s), got {index}."
                )
            indices.append(index)
        return self._reader.get_batch(indices)


__all__ = ["QueryCutoffVideoReader", "parse_timestamp"]
