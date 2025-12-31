"""Generated gRPC code for PenguinCode client-server communication."""

from .penguincode_pb2 import (
    # Auth messages
    AuthRequest,
    AuthResponse,
    RefreshRequest,
    ValidateRequest,
    ValidateResponse,
    # Chat messages
    CreateSessionRequest,
    CreateSessionResponse,
    ClientCapabilities,
    ServerInfo,
    ChatRequest,
    ChatResponse,
    TextChunk,
    AgentSpawn,
    AgentResult,
    StatusUpdate,
    Error,
    GetHistoryRequest,
    GetHistoryResponse,
    HistoryMessage,
    CloseSessionRequest,
    CloseSessionResponse,
    # Tool messages
    ToolRequest,
    ToolResponse,
    # Health messages
    HealthCheckRequest,
    HealthCheckResponse,
)

from .penguincode_pb2_grpc import (
    # Service stubs (client-side)
    AuthServiceStub,
    ChatServiceStub,
    ToolCallbackServiceStub,
    HealthServiceStub,
    # Service servicers (server-side)
    AuthServiceServicer,
    ChatServiceServicer,
    ToolCallbackServiceServicer,
    HealthServiceServicer,
    # Server registration functions
    add_AuthServiceServicer_to_server,
    add_ChatServiceServicer_to_server,
    add_ToolCallbackServiceServicer_to_server,
    add_HealthServiceServicer_to_server,
)

__all__ = [
    # Auth
    "AuthRequest",
    "AuthResponse",
    "RefreshRequest",
    "ValidateRequest",
    "ValidateResponse",
    # Chat
    "CreateSessionRequest",
    "CreateSessionResponse",
    "ClientCapabilities",
    "ServerInfo",
    "ChatRequest",
    "ChatResponse",
    "TextChunk",
    "AgentSpawn",
    "AgentResult",
    "StatusUpdate",
    "Error",
    "GetHistoryRequest",
    "GetHistoryResponse",
    "HistoryMessage",
    "CloseSessionRequest",
    "CloseSessionResponse",
    # Tools
    "ToolRequest",
    "ToolResponse",
    # Health
    "HealthCheckRequest",
    "HealthCheckResponse",
    # Stubs
    "AuthServiceStub",
    "ChatServiceStub",
    "ToolCallbackServiceStub",
    "HealthServiceStub",
    # Servicers
    "AuthServiceServicer",
    "ChatServiceServicer",
    "ToolCallbackServiceServicer",
    "HealthServiceServicer",
    # Registration
    "add_AuthServiceServicer_to_server",
    "add_ChatServiceServicer_to_server",
    "add_ToolCallbackServiceServicer_to_server",
    "add_HealthServiceServicer_to_server",
]
