"""Scripted mock provider for demos and tests.

Returns pre-configured responses in order; raises an error when the script is
exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

from .errors import LLMError
from .provider import Provider
from .types import (
    ChatRequest,
    ChatResponse,
    StreamEvent,
    ToolCall,
    Usage,
)

ScriptItem = Union[
    str,
    ChatResponse,
    Callable[["ChatRequest"], Union[str, ChatResponse]],
]


@dataclass
class MockProvider:
    """A provider that returns scripted responses.

    Parameters
    ----------
    script : list
        Items returned in order by ``chat()`` and ``stream()``. Each item may be:

        * a ``str`` — treated as ``ChatResponse(content=str)``
        * a ``ChatResponse`` — returned as-is
        * a callable ``(ChatRequest) -> str | ChatResponse`` — called with the
          request to produce the response
    name : str
        Provider name exposed in ``LLMService.info()``.
    """

    script: list[ScriptItem] = field(default_factory=list)
    name: str = "mock"
    calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def _next_response(self, request: ChatRequest) -> ChatResponse:
        self.calls.append({
            "model": request.model,
            "message_count": len(request.messages),
            "tools": [t.name for t in request.tools],
            "stream": request.stream,
        })
        if not self.script:
            raise LLMError("mock script exhausted")
        item = self.script.pop(0)
        if callable(item):
            result = item(request)
        else:
            result = item
        if isinstance(result, str):
            return ChatResponse(content=result, model=self.name)
        if isinstance(result, ChatResponse):
            return result
        return ChatResponse(content=str(result), model=self.name)

    def chat(self, request: ChatRequest) -> ChatResponse:
        return self._next_response(request)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        response = self._next_response(request)
        chunks = _chunk_text(response.content)
        for text in chunks:
            yield StreamEvent(type="text", text=text)
        for tc in response.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=tc)
        yield StreamEvent(type="done", response=response)

    def close(self) -> None:
        pass


def _chunk_text(text: str, size: int = 3) -> list[str]:
    """Split text into small chunks for realistic streaming simulation."""
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]