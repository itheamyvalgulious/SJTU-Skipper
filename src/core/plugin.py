"""Plugin protocol, context and DAG loader."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .events import EventBus
from .meta import MetaRegistry
from .services import ServiceRegistry
from .streams import StreamHub


class PluginError(RuntimeError):
    """Raised when a plugin cannot be loaded or its DAG is invalid."""


class Plugin:
    """Base class for behavior and low-level plugins.

    ``requires`` lists plugin ids, not service names.  The manager loads
    dependencies first and detects cycles.
    """

    id: str = ""
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def setup(self, ctx: "PluginContext") -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def teardown(self) -> None:  # pragma: no cover - optional
        pass


@dataclass(slots=True)
class PluginContext:
    """The only capability surface exposed to a plugin."""

    plugin_id: str
    data_dir: Path
    events: EventBus
    streams: StreamHub
    llm: Any
    ui: Any
    services: ServiceRegistry
    meta: MetaRegistry
    logger: logging.Logger

    def service(self, name: str) -> Any | None:
        return self.services.get(name)

    def require(self, name: str) -> Any:
        return self.services.require(name)

    def register_service(self, name: str, obj: Any) -> None:
        self.services.register(name, obj, plugin_id=self.plugin_id)

    def log(self, level: int, message: str, **fields: Any) -> None:
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
        self.logger.log(level, f"{message} {extra}".rstrip())

    def session_dir(self, session_id: str | None = None) -> Path:
        sid = session_id or self.streams.current.session_id
        path = self.data_dir / "sessions" / sid
        path.mkdir(parents=True, exist_ok=True)
        return path


ContextFactory = Callable[[str], PluginContext]


class PluginManager:
    """Topologically loads plugins declared with ``requires``."""

    def __init__(
        self,
        *,
        bus: EventBus,
        services: ServiceRegistry,
        meta: MetaRegistry,
        context_factory: ContextFactory,
    ) -> None:
        self._bus = bus
        self._services = services
        self._meta = meta
        self._context_factory = context_factory
        self._plugins: dict[str, Plugin] = {}
        self._order: list[str] = []
        self._loaded: list[str] = []
        self._lock = threading.RLock()

    def register(self, plugin: Plugin | type[Plugin]) -> None:
        if isinstance(plugin, type):
            plugin = plugin()
        if not isinstance(plugin, Plugin):
            raise PluginError(f"{plugin!r} is not a Plugin")
        if not plugin.id:
            raise PluginError(f"plugin {plugin!r} has no id")
        with self._lock:
            if plugin.id in self._plugins:
                raise PluginError(f"duplicate plugin id {plugin.id!r}")
            self._plugins[plugin.id] = plugin

    def register_all(self, plugins: Iterable[Plugin | type[Plugin]]) -> None:
        for plugin in plugins:
            self.register(plugin)

    def _sort(self) -> list[str]:
        plugins = self._plugins
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
        order: list[str] = []
        stack: list[str] = []

        def visit(pid: str) -> None:
            status = state.get(pid, 0)
            if status == 2:
                return
            if status == 1:
                cycle = " -> ".join(stack[stack.index(pid) :] + [pid])
                raise PluginError(f"plugin dependency cycle: {cycle}")
            state[pid] = 1
            stack.append(pid)
            plugin = plugins[pid]
            for dep in plugin.requires:
                if dep not in plugins:
                    raise PluginError(
                        f"plugin {pid!r} requires missing plugin {dep!r}"
                    )
                visit(dep)
            stack.pop()
            state[pid] = 2
            order.append(pid)

        for pid in sorted(plugins):
            visit(pid)
        return order

    @property
    def order(self) -> list[str]:
        with self._lock:
            return list(self._order)

    @property
    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._loaded)

    def get(self, plugin_id: str) -> Plugin | None:
        return self._plugins.get(plugin_id)

    def load(self) -> list[str]:
        order = self._sort()
        with self._lock:
            self._order = order
            self._loaded = []
        for pid in order:
            plugin = self._plugins[pid]
            ctx = self._context_factory(pid)
            try:
                plugin.setup(ctx)
            except Exception as exc:  # noqa: BLE001 - surface plugin id
                self.teardown()
                self._cleanup_plugin(pid)
                raise PluginError(f"plugin {pid!r} setup failed: {exc}") from exc
            with self._lock:
                self._loaded.append(pid)
        return order

    def _cleanup_plugin(self, pid: str) -> None:
        self._bus.remove_plugin(pid)
        self._meta.remove_plugin(pid)
        for service_name, owner in list(self._services.snapshot().items()):
            if owner == pid:
                self._services.unregister(service_name)

    def teardown(self) -> None:
        with self._lock:
            loaded = list(reversed(self._loaded))
            self._loaded = []
        for pid in loaded:
            plugin = self._plugins.get(pid)
            if plugin is None:
                continue
            try:
                plugin.teardown()
            except Exception:  # noqa: BLE001 - best effort shutdown
                logging.getLogger("lecture_agent.plugins").exception(
                    "plugin %s teardown failed", pid
                )
            self._cleanup_plugin(pid)
