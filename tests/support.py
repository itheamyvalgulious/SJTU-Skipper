"""Shared test doubles.

These intentionally do not import production LLM/agent types, so tests can run
while those modules are being developed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from plugins.settings.spec import SettingsSpec


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FakeResponse:
    content: str = ""
    tool_calls: list[FakeToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: FakeUsage = field(default_factory=FakeUsage)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = "fake"


ScriptItem = FakeResponse | str | Callable[[dict[str, Any]], FakeResponse]


class FakeLLM:
    """Scripted LLM used by agent/summarize/vlm tests."""

    def __init__(self, script: Iterable[ScriptItem] | None = None) -> None:
        self.script: list[ScriptItem] = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def _next(self, kwargs: dict[str, Any]) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.script:
            return FakeResponse(content="done")
        item = self.script.pop(0)
        if callable(item):
            return item(kwargs)
        if isinstance(item, str):
            return FakeResponse(content=item)
        return item

    def chat(self, **kwargs: Any) -> FakeResponse:
        return self._next(kwargs)

    def stream(self, **kwargs: Any):
        response = self._next(kwargs)
        yield SimpleNamespace(type="text", text=response.content, tool_call=None, response=None, error=None)
        yield SimpleNamespace(type="done", text="", tool_call=None, response=response, error=None)

    def config_for(self, profile: str) -> SimpleNamespace:
        return SimpleNamespace(
            profile=profile,
            base_url="http://fake",
            api_key="fake",
            model="fake",
            timeout=10,
            max_retries=0,
        )

    def info(self) -> dict[str, Any]:
        return {"provider": "fake", "calls": len(self.calls)}


class FakeSettings:
    """Minimal settings service used by plugin tests."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})
        self.specs_map: dict[str, SettingsSpec] = {}
        self.changes: list[tuple[str, Any, Any]] = []

    def register(self, spec: SettingsSpec) -> SettingsSpec:
        self.specs_map[spec.key] = spec
        if spec.key not in self.values:
            self.values[spec.key] = spec.default
        return spec

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key]
        spec = self.specs_map.get(key)
        return spec.default if spec is not None else default

    def set(self, key: str, value: Any) -> Any:
        old = self.get(key)
        self.values[key] = value
        self.changes.append((key, old, value))
        return value

    def set_many(self, values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in values.items():
            out[key] = self.set(key, value)
        return out

    def all(self, *, public: bool = False) -> dict[str, Any]:
        out = {}
        for key, value in self.values.items():
            spec = self.specs_map.get(key)
            if public and spec is not None and spec.secret:
                out[key] = "******" if value else ""
            else:
                out[key] = value
        return out

    def specs(self) -> list[SettingsSpec]:
        return sorted(self.specs_map.values(), key=lambda s: (s.group, s.order, s.key))
