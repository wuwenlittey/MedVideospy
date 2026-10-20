"""Opt-in verified visual case memory; importing never downloads a model."""

from .embedding import EmbeddingClient, EmbeddingError
from .store import VisualMemoryStore, build_index, validate_case

__all__ = ["EmbeddingClient", "EmbeddingError", "VisualMemoryStore", "build_index", "validate_case"]
