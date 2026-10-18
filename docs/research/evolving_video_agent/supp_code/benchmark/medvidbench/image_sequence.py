from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class _Batch:
    def __init__(self, frames: np.ndarray):
        self.frames = frames

    def asnumpy(self) -> np.ndarray:
        return self.frames


class ImageSequenceReader:
    """Small Decord-compatible reader for MedVidBench frame sequences."""

    def __init__(
        self,
        frame_paths: list[str],
        fps: float,
        frame_root: Path,
        rc_info: dict | None = None,
    ):
        if not frame_paths:
            raise ValueError("The image sequence is empty.")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}.")
        self.original_paths = frame_paths
        self.frame_paths = [
            self._resolve_path(path, frame_root) for path in frame_paths
        ]
        self.fps = fps
        self.cache: dict[int, np.ndarray] = {}
        self.rc_frame_index = None
        self.rc_bbox = None
        if rc_info:
            start_frame = rc_info.get("start_frame")
            if start_frame in frame_paths:
                self.rc_frame_index = frame_paths.index(start_frame)
                self.rc_bbox = rc_info.get("start_frame_bbox")

        missing_path = next(
            (path for path in self.frame_paths if not path.is_file()), None
        )
        if missing_path is not None:
            raise FileNotFoundError(f"Frame not found: {missing_path}")

    @staticmethod
    def _resolve_path(path: str, frame_root: Path) -> Path:
        prefix = "/root/data/"
        relative_path = path[len(prefix) :] if path.startswith(prefix) else path
        return frame_root / relative_path

    def __len__(self) -> int:
        return len(self.frame_paths)

    def get_avg_fps(self) -> float:
        return self.fps

    def get_batch(self, indices) -> _Batch:
        return _Batch(np.stack([self._load_frame(int(index)) for index in indices]))

    def _load_frame(self, index: int) -> np.ndarray:
        if index not in self.cache:
            with Image.open(self.frame_paths[index]) as source:
                image = source.convert("RGB")
                if index == self.rc_frame_index and self.rc_bbox:
                    image = image.copy()
                    ImageDraw.Draw(image).rectangle(
                        self.rc_bbox,
                        outline=(0, 255, 0),
                        width=8,
                    )
                self.cache[index] = np.asarray(image)
        return self.cache[index]
