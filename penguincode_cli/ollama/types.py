"""Type definitions for Ollama API."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Tool call from assistant message."""

    function: Dict[str, Any]


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str
    images: Optional[List[str]] = None  # For vision models
    tool_calls: Optional[List[ToolCall]] = None  # Tool calls from assistant


@dataclass
class GenerateRequest:
    """Generate API request parameters."""

    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[str] = None  # "json" for JSON mode
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None


@dataclass
class GenerateResponse:
    """Generate API response."""

    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ChatRequest:
    """Chat API request parameters."""

    model: str
    messages: List[Message]
    stream: bool = True
    format: Optional[str] = None  # "json" for JSON mode
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatResponse:
    """Chat API response."""

    model: str
    created_at: str
    message: Message
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ModelInfo:
    """Model information."""

    name: str
    modified_at: str
    size: int
    digest: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class UsageStats:
    """Token usage statistics from a response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    @classmethod
    def from_response(cls, response: GenerateResponse) -> "UsageStats":
        """Create usage stats from a generate response."""
        return cls(
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_tokens=(response.prompt_eval_count or 0) + (response.eval_count or 0),
            prompt_eval_duration_ms=(response.prompt_eval_duration or 0) / 1_000_000,
            eval_duration_ms=(response.eval_duration or 0) / 1_000_000,
            total_duration_ms=(response.total_duration or 0) / 1_000_000,
        )

    @classmethod
    def from_chat_response(cls, response: ChatResponse) -> "UsageStats":
        """Create usage stats from a chat response."""
        return cls(
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_tokens=(response.prompt_eval_count or 0) + (response.eval_count or 0),
            prompt_eval_duration_ms=(response.prompt_eval_duration or 0) / 1_000_000,
            eval_duration_ms=(response.eval_duration or 0) / 1_000_000,
            total_duration_ms=(response.total_duration or 0) / 1_000_000,
        )
