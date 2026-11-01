"""Provider protocol and a thread-safe registry."""

from __future__ import annotations

import threading
from typing import Any, Iterator, Protocol

from .types import ChatRequest, ChatResponse, StreamEvent


class Provider(Protocol):
    """Protocol every LLM provider must satisfy."""

    name: str

    def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


class ProviderRegistry:
    """Thread-safe name -> ``Provider`` mapping."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._lock = threading.RLock()

    def register(self, name: str, provider: Provider) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name must be a non-empty string")
        with self._lock:
            existing = self._providers.get(name)
            if existing is not None:
                if existing is not provider:
                    raise ValueError(
                        f"provider {name!r} already registered with a different object"
                    )
                # Same object — idempotent re-registration, no-op.
                return
            self._providers[name] = provider

    def unregister(self, name: str) -> None:
        with self._lock:
            self._providers.pop(name, None)

    def get(self, name: str) -> Provider | None:
        with self._lock:
            return self._providers.get(name)

    def require(self, name: str) -> Provider:
        provider = self.get(name)
        if provider is None:
            raise KeyError(f"provider {name!r} is not registered; available: {self.names()}")
        return provider

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    def close_all(self) -> None:
        with self._lock:
            for provider in list(self._providers.values()):
                try:
                    provider.close()
                except Exception:  # noqa: BLE001 – best-effort
                    pass
            self._providers.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None