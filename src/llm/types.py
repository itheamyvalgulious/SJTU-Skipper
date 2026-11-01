"""Unified data types for the LLM layer.

All request/response objects used by providers, the service tier, and plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

ContentPart = Union["TextPart", "ImagePart"]


@dataclass(frozen=True, slots=True)
class TextPart:
    """A plain-text content segment."""

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image supplied via URL, raw bytes, or a base64 data URL.

    ``data`` is serialised to a ``data:<mime_type>;base64,...`` URL at request
    time by the OpenAI-compatible provider.
    """

    url: str | None = None
    data: bytes | None = None
    mime_type: str = "image/jpeg"
    detail: str = "auto"


@dataclass(slots=True)
class ToolCall:
    """A tool call the model wants the caller to execute."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolDef:
    """A tool definition sent to the model (OpenAI function-calling shape)."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    """A chat message.

    ``content`` can be a plain string or a list of ``TextPart`` / ``ImagePart``
    for multi-modal requests.
    """

    role: str  # system | user | assistant | tool
    content: str | list[ContentPart]
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class Usage:
    """Token usage returned by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ChatRequest:
    """A provider-agnostic chat/stream request."""

    model: str | None
    messages: list[Message]
    tools: list[ToolDef] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    stream: bool = False


@dataclass(slots=True)
class ChatResponse:
    """A completed chat response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""


@dataclass(slots=True)
class StreamEvent:
    """A single event emitted during streaming."""

    type: str  # "text" | "reasoning" | "tool_call" | "done" | "error"
    text: str = ""
    tool_call: ToolCall | None = None
    response: ChatResponse | None = None
    error: Exception | None = None


@dataclass(slots=True)
class ProviderConfig:
    """Snapshot of parameters used to construct a provider request."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    provider: str = ""  # provider name to route to, e.g. "openai"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 60.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetryPolicy:
    """Controls automatic retry behaviour for transient failures."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 20.0
    jitter: float = 0.2
    retry_statuses: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)