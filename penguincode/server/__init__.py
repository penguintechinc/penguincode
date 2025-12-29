"""PenguinCode gRPC Server.

Provides the server-side implementation for client-server mode.
Handles agent orchestration, Ollama communication, and tool callbacks.
"""

from .main import serve, PenguinCodeServer

__all__ = ["serve", "PenguinCodeServer"]
