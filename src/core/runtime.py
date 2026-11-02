"""Runtime wiring: event bus, streams, UI, services and plugin DAG."""

from __future__ import annotations

import dataclasses
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from config import AppConfig
from .events import Clock, EventBus
from .meta import MetaRegistry
from .plugin import Plugin, PluginContext, PluginManager
from .services import ServiceRegistry
from .streams import StreamHub
from .ui import UIManager

logger = logging.getLogger("lecture_agent.runtime")


class LLMProxy:
    """Late-bound facade over the ``llm`` service.

    The low-level LLM plugin is itself a plugin, so the context must exist
    before it registers the real ``LLMService``.  This proxy resolves the
    service at call time and keeps plugin contexts stable.
    """

    def __init__(self, services: ServiceRegistry) -> None:
        self._services = services

    def _service(self) -> Any:
        service = self._services.get("llm")
        if service is None:
            raise RuntimeError(
                "LLM service is not registered; load the 'llm' plugin or inject a fake"
            )
        return service

    def chat(self, **kwargs: Any) -> Any:
        return self._service().chat(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        return self._service().stream(**kwargs)

    def config_for(self, profile: str) -> Any:
        return self._service().config_for(profile)

    def info(self) -> dict[str, Any]:
        service = self._services.get("llm")
        if service is None:
            return {"registered": False}
        getter = getattr(service, "info", None)
        if callable(getter):
            try:
                return dict(getter())
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        return {"registered": True}

    def close(self) -> None:
        service = self._services.get("llm")
        if service is not None:
            closer = getattr(service, "close", None)
            if callable(closer):
                closer()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service(), name)


class Runtime:
    """Owns all long-lived components and the plugin DAG."""

    def __init__(
        self,
        *,
        config: AppConfig,
        bus: EventBus,
        clock: Clock,
        hub: StreamHub,
        ui: UIManager,
        meta: MetaRegistry,
        services: ServiceRegistry,
        llm: LLMProxy,
        plugins: PluginManager,
    ) -> None:
        self.config = config
        self.bus = bus
        self.clock = clock
        self.hub = hub
        self.ui = ui
        self.meta = meta
        self.services = services
        self.llm = llm
        self.plugins = plugins
        self._lock = threading.RLock()
        self._started = False

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def started(self) -> bool:
        return self._started

    @property
    def session_id(self) -> str:
        return self.hub.current.session_id

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.bus.start()
            self.clock.start()
            try:
                order = self.plugins.load()
            except Exception:
                self.clock.stop()
                self.bus.stop()
                raise
            self._started = True
            logger.info("runtime started; plugin order: %s", " -> ".join(order))
            self.ui.refresh("runtime-started")

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            logger.info("runtime stopping")
            try:
                self.hub.end_session("runtime stopped")
            except Exception:  # noqa: BLE001
                logger.exception("failed to end session")
            self.plugins.teardown()
            self.clock.stop()
            self.llm.close()
            self.bus.stop()
            self._started = False

    def add_plugin(self, plugin: Plugin | type[Plugin]) -> None:
        self.plugins.register(plugin)

    def add_plugins(self, plugins: Iterable[Plugin | type[Plugin]]) -> None:
        self.plugins.register_all(plugins)

    def status(self) -> dict[str, Any]:
        current = self.hub.current
        return {
            "runtime": {
                "started": self._started,
                "data_dir": str(self.data_dir),
                "plugin_order": self.plugins.order,
                "plugins_loaded": self.plugins.loaded,
                "services": self.services.snapshot(),
            },
            "session": current.status(),
            "sessions": [
                dataclasses.asdict(s) for s in self.hub.list_sessions()
            ][-10:],
            "llm": self.llm.info(),
        }


def build_runtime(
    config: AppConfig,
    *,
    plugins: Iterable[Plugin | type[Plugin]] | None = None,
    llm: Any | None = None,
    audio_emit_interval: float = 0.5,
    video_emit_interval: float = 1.0,
    clock_interval: float = 1.0,
) -> Runtime:
    """Construct a runtime without starting it.

    ``llm`` can be injected for tests/demo.  Otherwise the ``llm`` plugin must
    register the service during setup.
    """

    config.ensure_dirs()
    services = ServiceRegistry()
    if llm is not None:
        services.register("llm", llm, plugin_id="<injected>")
    llm_proxy = LLMProxy(services)
    bus = EventBus()
    hub = StreamHub(
        emit=bus.emit,
        audio_emit_interval=audio_emit_interval,
        video_emit_interval=video_emit_interval,
    )
    meta = MetaRegistry()

    def _runtime_meta() -> dict[str, Any]:
        now = datetime.now().astimezone()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": now.tzname() or "",
            "app": "sjtu-skipper",
        }

    meta.register("runtime", _runtime_meta, plugin_id="runtime")

    def on_ui_change(reason: str) -> None:
        bus.emit("ui.changed", reason=reason)

    ui = UIManager(on_change=on_ui_change)

    def context_factory(plugin_id: str) -> PluginContext:
        return PluginContext(
            plugin_id=plugin_id,
            data_dir=config.data_dir,
            events=bus,
            streams=hub,
            llm=llm_proxy,
            ui=ui,
            services=services,
            meta=meta,
            logger=logging.getLogger(f"lecture_agent.plugin.{plugin_id}"),
        )

    manager = PluginManager(
        bus=bus,
        services=services,
        meta=meta,
        context_factory=context_factory,
    )

    runtime = Runtime(
        config=config,
        bus=bus,
        clock=Clock(bus, interval=clock_interval),
        hub=hub,
        ui=ui,
        meta=meta,
        services=services,
        llm=llm_proxy,
        plugins=manager,
    )

    def status_provider() -> dict[str, Any]:
        return runtime.status()

    ui.set_status_provider(status_provider)
    if plugins:
        manager.register_all(plugins)
    return runtime
