"""Shared interfaces and types for PenguinCode client-server architecture."""

from .interfaces import (
    IChatService,
    IToolExecutor,
    IAuthService,
    ToolResult,
)
from .types import (
    SessionInfo,
    ChatMessage,
    AgentStatus,
    ServerMode,
)

__all__ = [
    # Interfaces
    "IChatService",
    "IToolExecutor",
    "IAuthService",
    "ToolResult",
    # Types
    "SessionInfo",
    "ChatMessage",
    "AgentStatus",
    "ServerMode",
]
