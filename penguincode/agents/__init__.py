"""Agent implementations for PenguinCode."""

from .base import AgentConfig, AgentResult, BaseAgent, Permission, TOOL_DEFINITIONS
from .chat import ChatAgent
from .executor import ExecutorAgent
from .explorer import ExplorerAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "Permission",
    "TOOL_DEFINITIONS",
    "ChatAgent",
    "ExecutorAgent",
    "ExplorerAgent",
]
