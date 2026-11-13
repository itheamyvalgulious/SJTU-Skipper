"""LLMService — the single entry point for all LLM calls.

Profiles map logical roles (``agent``, ``vlm``, ``stt``) to provider
configurations.  Configuration is fetched lazily from a callback so the service
stays decoupled from any particular settings plugin.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

from .errors import LLMError
from .provider import Provider, ProviderRegistry
from .types import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    ProviderConfig,
    StreamEvent,
    TextPart,
    ToolDef,
    Usage,
)

logger = logging.getLogger("lecture_agent.llm.service")

ConfigCallback = Callable[[str], ProviderConfig]

# Allow callers to pass Message objects or OpenAI-style dicts.
MessageInput = Union[Message, dict[str, Any]]

_DEFAULT_CONFIG = ProviderConfig()


def _coerce_message(msg: MessageInput) -> Message:
    """Accept both ``Message`` objects and OpenAI-style dicts."""
    if isinstance(msg, Message):
        return msg
    if isinstance(msg, dict):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[TextPart | ImagePart] = []
            for part in content:
                if isinstance(part, TextPart | ImagePart):
                    parts.append(part)
                elif isinstance(part, dict):
                    kind = part.get("type")
                    if kind == "text":
                        parts.append(TextPart(text=str(part.get("text", ""))))
                    elif kind == "image_url":
                        image = part.get("image_url")
                        url = image.get("url") if isinstance(image, dict) else image
                        if url:
                            parts.append(ImagePart(url=str(url)))
                    else:
                        parts.append(TextPart(text=str(part)))
                else:
                    parts.append(TextPart(text=str(part)))
            content = parts
        name = msg.get("name")
        tool_calls_raw = msg.get("tool_calls", [])
        from .types import ToolCall as TC

        tool_calls = [
            TC(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", {}),
            )
            for tc in tool_calls_raw
        ]
        return Message(
            role=role,
            content=content,
            name=name,
            tool_calls=tool_calls,
            tool_call_id=msg.get("tool_call_id"),
        )
    raise TypeError(f"expected Message or dict, got {type(msg).__name__}")


def _coerce_tool(tool: ToolDef | dict[str, Any]) -> ToolDef:
    """Accept both ``ToolDef`` and OpenAI-style function dicts."""
    if isinstance(tool, ToolDef):
        return tool
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict):
            return ToolDef(
                name=str(fn.get("name", "")),
                description=str(fn.get("description", "")),
                parameters=dict(fn.get("parameters") or {}),
            )
        return ToolDef(
            name=str(tool.get("name", "")),
            description=str(tool.get("description", "")),
            parameters=dict(tool.get("parameters") or {}),
        )
    raise TypeError(f"expected ToolDef or dict, got {type(tool).__name__}")


@dataclass
class LLMService:
    """Multi-provider LLM facade.

    Parameters
    ----------
    config_provider : callable
        ``(profile: str) -> ProviderConfig``.  Called on every request so
        config changes take effect immediately.
    default_provider : str
        Fallback provider name when a profile config doesn't specify one.
    on_event : callable | None
        ``(event_type: str, payload: dict) -> None`` callback for observability
        (``llm.request``, ``llm.response``, ``llm.error``).
    """

    config_provider: ConfigCallback = field(default=lambda _: _DEFAULT_CONFIG)
    default_provider: str = "openai"
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    registry: ProviderRegistry = field(default_factory=ProviderRegistry)
    usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    """Per-model token totals: ``requests``, ``input_tokens``, ``output_tokens``,
    ``cache_read_tokens``.  Counted, not priced: pricing depends on the plan and
    peak hours, so it is left to a later accounting layer."""
    _usage_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def register_provider(self, name: str, provider: Provider) -> None:
        self.registry.register(name, provider)

    def _record_usage(self, model: str, usage: Usage) -> dict[str, int]:
        """Add one response to the per-model totals and log the request line."""

        key = model or "unknown"
        with self._usage_lock:
            totals = self.usage_by_model.setdefault(
                key,
                {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                },
            )
            totals["requests"] += 1
            totals["input_tokens"] += usage.prompt_tokens
            totals["output_tokens"] += usage.completion_tokens
            totals["cache_read_tokens"] += getattr(usage, "cached_tokens", 0)
            snapshot = dict(totals)

        logger.info(
            "llm usage model=%s input=%d output=%d cache_read=%d | "
            "total input=%d output=%d cache_read=%d",
            key,
            usage.prompt_tokens,
            usage.completion_tokens,
            getattr(usage, "cached_tokens", 0),
            snapshot["input_tokens"],
            snapshot["output_tokens"],
            snapshot["cache_read_tokens"],
        )
        return snapshot

    def usage_snapshot(self) -> dict[str, dict[str, int]]:
        """Copy of the per-model token totals (safe to hand to the UI)."""

        with self._usage_lock:
            return {model: dict(totals) for model, totals in self.usage_by_model.items()}

    def _provider_for(self, config: ProviderConfig) -> Provider:
        provider = self.registry.require(config.provider or self.default_provider)
        reconfigure = getattr(provider, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(config)
        return provider


    # ------------------------------------------------------------------
    # chat / stream
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        profile: str = "agent",
        messages: list[MessageInput],
        model: str | None = None,
        tools: list[ToolDef] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        config = self.config_for(profile)
        provider = self._provider_for(config)
        coerced = [_coerce_message(m) for m in messages]

        request = ChatRequest(
            model=model or config.model,
            messages=coerced,
            tools=[_coerce_tool(t) for t in (tools or [])],
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else config.temperature,
            max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
            response_format=response_format,
            extra=dict(extra or {}),
            stream=False,
            session_id=session_id,
            parent_session_id=parent_session_id,
        )

        if self.on_event:
            self.on_event("llm.request", {
                "profile": profile,
                "model": request.model,
                "message_count": len(coerced),
                "tool_count": len(request.tools),
                "stream": False,
                "session_id": session_id,
            })

        try:
            response = provider.chat(request)
        except LLMError:
            if self.on_event:
                self.on_event("llm.error", {"profile": profile, "phase": "chat"})
            raise

        totals = self._record_usage(response.model or request.model or "", response.usage)
        if self.on_event:
            self.on_event("llm.response", {
                "profile": profile,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "content_len": len(response.content),
                "tool_calls": len(response.tool_calls),
                "session_id": session_id,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cached_tokens": response.usage.cached_tokens,
                },
                "usage_totals": totals,
            })
        return response

    def stream(
        self,
        *,
        profile: str = "agent",
        messages: list[MessageInput],
        model: str | None = None,
        tools: list[ToolDef] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        config = self.config_for(profile)
        provider = self._provider_for(config)
        coerced = [_coerce_message(m) for m in messages]

        request = ChatRequest(
            model=model or config.model,
            messages=coerced,
            tools=[_coerce_tool(t) for t in (tools or [])],
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else config.temperature,
            max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
            response_format=response_format,
            extra=dict(extra or {}),
            stream=True,
            session_id=session_id,
            parent_session_id=parent_session_id,
        )

        if self.on_event:
            self.on_event("llm.request", {
                "profile": profile,
                "model": request.model,
                "message_count": len(coerced),
                "tool_count": len(request.tools),
                "stream": True,
                "session_id": session_id,
            })

        try:
            for event in provider.stream(request):
                if event.type == "done" and event.response is not None:
                    self._record_usage(
                        event.response.model or request.model or "",
                        event.response.usage,
                    )
                if event.type == "error" and self.on_event:
                    self.on_event("llm.error", {
                        "profile": profile,
                        "phase": "stream",
                        "error": str(event.error) if event.error else "",
                    })
                yield event
        except LLMError:
            if self.on_event:
                self.on_event("llm.error", {"profile": profile, "phase": "stream"})
            raise

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    def config_for(self, profile: str) -> ProviderConfig:
        """Resolve config for *profile* (e.g. ``"agent"``, ``"vlm"``, ``"stt"``).

        The *vlm* profile overrides ``model`` with ``vlm_model`` when set.
        """
        base = self.config_provider(profile)
        config = ProviderConfig(
            base_url=base.base_url or _DEFAULT_CONFIG.base_url,
            api_key=base.api_key or _DEFAULT_CONFIG.api_key,
            model=base.model or _DEFAULT_CONFIG.model,
            provider=base.provider or _DEFAULT_CONFIG.provider,
            temperature=(
                base.temperature
                if base.temperature is not None
                else _DEFAULT_CONFIG.temperature
            ),
            max_tokens=(
                base.max_tokens
                if base.max_tokens is not None
                else _DEFAULT_CONFIG.max_tokens
            ),
            timeout=base.timeout or _DEFAULT_CONFIG.timeout,
            max_retries=(
                base.max_retries
                if base.max_retries is not None
                else _DEFAULT_CONFIG.max_retries
            ),
            extra_headers=dict(base.extra_headers),
            extra_body=dict(base.extra_body),
        )
        return config

    # ------------------------------------------------------------------
    # lifecycle / info
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close every registered provider."""
        self.registry.close_all()

    def info(self) -> dict[str, Any]:
        return {
            "registered": True,
            "provider": self.default_provider,
            "providers": self.registry.names(),
            "default_provider": self.default_provider,
            "usage": self.usage_snapshot(),
        }