"""LLM layer: types, providers, service, and retry logic."""

from __future__ import annotations

from .errors import (
    LLMAuthError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMRetryExhausted,
    LLMStreamError,
    LLMTimeoutError,
)
from .openai_compat import OpenAICompatibleProvider
from .provider import Provider, ProviderRegistry
from .service import LLMService
from .types import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    ProviderConfig,
    RetryPolicy,
    StreamEvent,
    TextPart,
    ToolCall,
    ToolDef,
    Usage,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ImagePart",
    "LLMAuthError",
    "LLMError",
    "LLMHTTPError",
    "LLMRateLimitError",
    "LLMRetryExhausted",
    "LLMStreamError",
    "LLMTimeoutError",
    "LLMService",
    "Message",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderConfig",
    "ProviderRegistry",
    "RetryPolicy",
    "StreamEvent",
    "TextPart",
    "ToolCall",
    "ToolDef",
    "Usage",
]