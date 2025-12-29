"""PenguinCode Client for gRPC communication.

Provides the client-side implementation for client-server mode.
Handles tool execution locally and communicates with remote server.
"""

from .grpc_client import GRPCClient
from .tool_executor import LocalToolExecutor
from .auth import TokenManager

__all__ = ["GRPCClient", "LocalToolExecutor", "TokenManager"]
