"""Named service registry for inter-plugin calls."""

from __future__ import annotations

import threading
from typing import Any, Iterable


class ServiceError(RuntimeError):
    """Base error for service registration/resolution."""


class ServiceNotFound(ServiceError):
    pass


class ServiceRegistry:
    """Thread-safe name -> object registry.

    Plugins must not import each other's implementation.  They publish a
    service object and call methods through this registry.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._owners: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, name: str, obj: Any, *, plugin_id: str = "") -> None:
        if not name or not isinstance(name, str):
            raise ValueError("service name must be a non-empty string")
        if obj is None:
            raise ValueError("service object must not be None")
        with self._lock:
            existing = self._services.get(name)
            if existing is not None and existing is not obj:
                raise ServiceError(
                    f"service {name!r} already registered by {self._owners.get(name)!r}"
                )
            self._services[name] = obj
            self._owners[name] = plugin_id

    def unregister(self, name: str, *, plugin_id: str | None = None) -> None:
        with self._lock:
            if name not in self._services:
                return
            if plugin_id is not None and self._owners.get(name) != plugin_id:
                raise ServiceError(f"service {name!r} is not owned by {plugin_id!r}")
            self._services.pop(name, None)
            self._owners.pop(name, None)

    def get(self, name: str) -> Any | None:
        with self._lock:
            return self._services.get(name)

    def require(self, name: str) -> Any:
        obj = self.get(name)
        if obj is None:
            raise ServiceNotFound(f"required service {name!r} is not registered")
        return obj

    def call(self, name: str, method: str, *args: Any, **kwargs: Any) -> Any:
        service = self.require(name)
        func = getattr(service, method, None)
        if func is None or not callable(func):
            raise ServiceError(f"service {name!r} has no callable method {method!r}")
        return func(*args, **kwargs)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._services)

    def owner(self, name: str) -> str | None:
        with self._lock:
            return self._owners.get(name)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {name: self._owners.get(name, "") for name in sorted(self._services)}

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterable[str]:  # pragma: no cover - convenience
        return iter(self.names())
