from .clip_skim import clip_skim_tool, execute_clip_skim
from .frame_inspect import execute_frame_inspect, frame_inspect_tool
from .overview import execute_overview, overview_tool
from .registry import ToolRegistry


TOOLS = {
    "overview": overview_tool,
    "clip_skim": clip_skim_tool,
    "frame_inspect": frame_inspect_tool,
}

TOOL_FUNCTIONS = {
    "overview": execute_overview,
    "clip_skim": execute_clip_skim,
    "frame_inspect": execute_frame_inspect,
}

DEFAULT_TOOL_REGISTRY = ToolRegistry(TOOLS, TOOL_FUNCTIONS)

__all__ = ["DEFAULT_TOOL_REGISTRY", "ToolRegistry"]
