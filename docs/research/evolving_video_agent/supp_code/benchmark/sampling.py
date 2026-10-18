"""Shared mode sampling settings for benchmark inference and offline training."""
from pathlib import Path

import yaml


SAMPLING_CONFIG = Path(__file__).resolve().parents[1] / "config/benchmark_sampling.yml"


def get_sampling_stride(benchmark: str, mode: str) -> int:
    settings = yaml.safe_load(SAMPLING_CONFIG.read_text(encoding="utf-8"))
    stride = settings[benchmark][mode]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"{SAMPLING_CONFIG}: {benchmark}.{mode} must be a positive integer stride")
    return stride
