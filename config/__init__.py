"""Backward-compatible access to the original VideoSeek configuration."""

from .videoseek import general_config, init_config, prompts_config

__all__ = ["general_config", "init_config", "prompts_config"]
