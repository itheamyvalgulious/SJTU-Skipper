"""Metadata registry used by the ``info`` agent tool."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Mapping

InfoProvider = Callable[[], Mapping[str, Any] | str]


class MetaRegistry:
    """Plugins contribute small, human-readable metadata snapshots."""

    def __init__(self) -> None:
        self._providers: dict[str, tuple[str, InfoProvider]] = {}
        self._lock = threading.RLock()

    def register(self, name: str, provider: InfoProvider, *, plugin_id: str = "") -> None:
        if not name:
            raise ValueError("meta provider name is required")
        if not callable(provider):
            raise TypeError("meta provider must be callable")
        with self._lock:
            self._providers[name] = (plugin_id, provider)

    def unregister(self, name: str, *, plugin_id: str | None = None) -> None:
        with self._lock:
            current = self._providers.get(name)
            if current is None:
                return
            if plugin_id is not None and current[0] != plugin_id:
                raise ValueError(f"meta provider {name!r} is not owned by {plugin_id!r}")
            self._providers.pop(name, None)

    def remove_plugin(self, plugin_id: str) -> None:
        with self._lock:
            for name, (owner, _) in list(self._providers.items()):
                if owner == plugin_id:
                    self._providers.pop(name, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            providers = list(self._providers.items())
        out: dict[str, Any] = {}
        for name, (_owner, provider) in providers:
            try:
                out[name] = provider()
            except Exception as exc:  # noqa: BLE001 - one bad provider must not break info
                out[name] = f"<error: {exc}>"
        return out

    def render(self) -> str:
        snapshot = self.snapshot()
        lines: list[str] = []
        for name, value in snapshot.items():
            if isinstance(value, str):
                rendered = value
            else:
                try:
                    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
                except TypeError:
                    rendered = str(value)
            lines.append(f"## {name}\n{rendered}")
        return "\n\n".join(lines)
