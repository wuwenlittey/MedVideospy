from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .agent import VideoSpyAgent

__all__ = ["VideoSpyAgent"]


def __getattr__(name: str):
    if name == "VideoSpyAgent":
        from .agent import VideoSpyAgent

        return VideoSpyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
