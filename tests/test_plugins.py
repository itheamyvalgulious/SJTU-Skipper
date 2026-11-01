from __future__ import annotations

import pytest

from config import load_config
from core.plugin import Plugin, PluginContext, PluginError
from core.runtime import build_runtime


class Recorder(Plugin):
    id = "recorder"
    requires: tuple[str, ...] = ()

    def __init__(self, log: list[str], name: str | None = None) -> None:
        self.id = name or type(self).id
        self.log = log

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.log.append(f"setup:{self.id}")
        ctx.register_service(f"svc.{self.id}", self)

    def teardown(self) -> None:
        self.log.append(f"teardown:{self.id}")


class Consumer(Plugin):
    id = "consumer"
    requires = ("recorder",)

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def setup(self, ctx: PluginContext) -> None:
        assert ctx.services.get("svc.recorder") is not None
        self.log.append("setup:consumer")
        ctx.events.on("ping", lambda e: self.log.append("pong"), plugin_id=self.id)

    def teardown(self) -> None:
        self.log.append("teardown:consumer")


def test_topological_order_and_services(tmp_path) -> None:
    log: list[str] = []
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[Consumer(log), Recorder(log)])
    runtime.start()
    try:
        assert runtime.plugins.order == ["recorder", "consumer"]
        assert log == ["setup:recorder", "setup:consumer"]
        runtime.bus.emit("ping")
        assert runtime.bus.flush(2.0)
        assert "pong" in log
    finally:
        runtime.stop()
    assert log[-1] == "teardown:recorder"
    assert log[-2] == "teardown:consumer"


def test_cycle_is_rejected(tmp_path) -> None:
    log: list[str] = []

    class A(Recorder):
        id = "a"
        requires = ("b",)

    class B(Recorder):
        id = "b"
        requires = ("a",)

    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[A(log), B(log)])
    with pytest.raises(PluginError, match="cycle"):
        runtime.plugins.load()


def test_missing_dependency_is_rejected(tmp_path) -> None:
    log: list[str] = []

    class A(Recorder):
        id = "a"
        requires = ("nope",)

    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[A(log)])
    with pytest.raises(PluginError, match="missing plugin"):
        runtime.plugins.load()


def test_setup_failure_rolls_back(tmp_path) -> None:
    log: list[str] = []

    class Bad(Recorder):
        id = "bad"
        requires = ("recorder",)

        def setup(self, ctx: PluginContext) -> None:
            self.log.append("setup:bad")
            raise RuntimeError("boom")

    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[Recorder(log), Bad(log)])
    with pytest.raises(PluginError, match="bad"):
        runtime.start()
    assert runtime.started is False
    assert "teardown:recorder" in log


def test_duplicate_plugin_id_rejected(tmp_path) -> None:
    log: list[str] = []
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[Recorder(log)])
    with pytest.raises(PluginError, match="duplicate"):
        runtime.add_plugin(Recorder(log))


def test_service_registry() -> None:
    from core.services import ServiceError, ServiceNotFound, ServiceRegistry

    registry = ServiceRegistry()
    obj = object()
    registry.register("x", obj, plugin_id="p")
    assert registry.get("x") is obj
    assert registry.require("x") is obj
    assert "x" in registry
    with pytest.raises(ServiceError):
        registry.register("x", object(), plugin_id="q")
    registry.unregister("x", plugin_id="p")
    with pytest.raises(ServiceNotFound):
        registry.require("x")
