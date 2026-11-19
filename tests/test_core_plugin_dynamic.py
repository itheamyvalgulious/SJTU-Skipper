from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config import load_config
from core.events import EventBus
from core.media import MediaKind
from core.plugin import Plugin, PluginContext, PluginError
from core.runtime import build_runtime
from core.services import ServiceError, ServiceRegistry
from core.streams import StreamHub
from core.ui import Side, UIManager


class ScopedPlugin(Plugin):
    id = "scoped"
    provides = ("scoped",)

    def setup(self, ctx: PluginContext) -> None:
        # Every call below omits plugin_id on purpose.
        ctx.register_service("scoped", object())
        ctx.ui.add_tab(Side.LEFT, "scoped.tab", "范围")
        ctx.ui.add_page("scoped.page", "范围页", render=lambda: "hi")
        ctx.ui.register_action("scoped.ping", lambda payload: {"ok": True})
        ctx.ui.register_command("scoped.cmd", "命令", page_id="scoped.page")
        ctx.ui.register_task_provider(lambda: [{"id": "t"}])
        ctx.events.on("scoped.event", lambda event: None)
        ctx.meta.register("scoped", lambda: "ok")


def test_context_injects_plugin_id_and_cleanup_removes_everything(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[ScopedPlugin()])
    original_commands = runtime.ui.snapshot()["commands"]
    runtime.start()
    try:
        snapshot = runtime.ui.snapshot()
        assert snapshot["tabs"] == {"scoped.tab": "left"}
        assert "scoped.page" in snapshot["pages"]
        assert "scoped.ping" in snapshot["actions"]
        assert runtime.bus.handler_count("scoped.event") == 1
        assert runtime.services.owner("scoped") == "scoped"
        assert runtime.ui.state()["tasks"] == [{"id": "t"}]
    finally:
        runtime.stop()

    assert "scoped.tab" not in runtime.ui.snapshot()["tabs"]
    assert "scoped.ping" not in runtime.ui.snapshot()["actions"]
    assert runtime.ui.snapshot()["commands"] == original_commands
    assert runtime.ui.state()["tasks"] == []
    assert runtime.bus.handler_count("scoped.event") == 0
    assert runtime.services.get("scoped") is None
    # The built-in navigation action survives (it belongs to "core").
    assert runtime.ui.has_action("ui.open_page")


def test_foreign_service_requires_declaration(tmp_path: Path) -> None:
    class Snoop(Plugin):
        id = "snoop"

        def setup(self, ctx: PluginContext) -> None:
            ctx.require("scoped")

    runtime = build_runtime(
        load_config(data_dir=tmp_path), plugins=[ScopedPlugin(), Snoop()]
    )
    with pytest.raises(PluginError):
        runtime.start()
    assert runtime.started is False


def test_declared_soft_dependency_allows_access(tmp_path: Path) -> None:
    seen: list[object] = []

    class Snoop(Plugin):
        id = "snoop"
        optional_requires = ("scoped",)

        def setup(self, ctx: PluginContext) -> None:
            seen.append(ctx.require("scoped"))

    runtime = build_runtime(
        load_config(data_dir=tmp_path), plugins=[ScopedPlugin(), Snoop()]
    )
    runtime.start()
    try:
        assert runtime.plugins.order.index("scoped") < runtime.plugins.order.index(
            "snoop"
        )
        assert seen == [runtime.services.require("scoped")]
    finally:
        runtime.stop()


def test_register_service_must_match_declared_provides() -> None:
    class WrongName(Plugin):
        id = "wrong"
        provides = ("wrong",)

    bus = EventBus()
    ctx = PluginContext(
        plugin_id="wrong",
        data_dir=Path("."),
        events=bus,
        streams=StreamHub(),
        llm=None,
        ui=UIManager(),
        services=ServiceRegistry(),
        meta=__import__("core.meta", fromlist=["MetaRegistry"]).MetaRegistry(),
        logger=__import__("logging").getLogger("t"),
        declared_provides=("wrong",),
    )
    with pytest.raises(PluginError):
        ctx.register_service("other", object())
    with pytest.raises(PluginError):
        ctx.register_service("wrong", WrongName())


def test_shared_tab_keeps_other_owner_and_drops_when_last_leaves() -> None:
    ui = UIManager()
    ui.add_tab(Side.RIGHT, "note", "笔记", plugin_id="notes")
    ui.add_toolbox_item("note", "note.a", "A", plugin_id="notes")
    ui.add_tab(Side.RIGHT, "note", "笔记", plugin_id="other-plugin")
    ui.add_toolbox_item("note", "note.b", "B", plugin_id="other-plugin")
    assert len(ui.state()["right_tabs"][0]["toolbox"]) == 2

    ui.remove_plugin("notes")
    assert ui.has_tab("note")
    assert not ui.has_toolbox_item("note.a")

    ui.remove_plugin("other-plugin")
    assert not ui.has_tab("note")


def test_service_registry_replace_injected_and_refuse_private_calls() -> None:
    registry = ServiceRegistry()
    registry.register("llm", object(), plugin_id="<injected>")
    real = object()
    registry.register("llm", real, plugin_id="llm")
    assert registry.get("llm") is real
    with pytest.raises(ServiceError):
        registry.register("llm", object(), plugin_id="other")

    class Facade:
        def ping(self) -> str:
            return "pong"

        def _secret(self) -> str:  # pragma: no cover - must stay unreachable
            return "nope"

    registry.register("svc", Facade(), plugin_id="plug")
    assert registry.call("svc", "ping") == "pong"
    with pytest.raises(ServiceError):
        registry.call("svc", "_secret")
    assert registry.remove_plugin("plug") == ["svc"]
    assert registry.get("svc") is None


def test_event_bus_drops_subscriptions_physically() -> None:
    bus = EventBus()
    subs = [bus.on("x", lambda event: None, plugin_id="p") for _ in range(3)]
    bus.on("*", lambda event: None, plugin_id="p")
    bus.on("x", lambda event: None, plugin_id="q")
    assert bus.handler_count("x") == 5

    bus.remove_plugin("p")
    assert bus.handler_count("x") == 1
    assert all(not sub.active for sub in subs)

    for sub in list(bus._handlers.get("x", [])):
        sub.cancel()
    assert bus.handler_count() == 0
    assert not bus._handlers


def test_stream_hub_media_consumers() -> None:
    hub = StreamHub()
    hub.register_consumer(
        "stt",
        MediaKind.AUDIO,
        cursor_fn=lambda: 4.0,
        busy_fn=lambda: True,
        gate_limit_fn=lambda: 30.0,
    )
    hub.register_consumer(
        "vlm",
        MediaKind.AUDIO,
        cursor_fn=lambda: 2.0,
        busy_fn=lambda: False,
        gate_limit_fn=lambda: 60.0,
    )
    assert hub.consumer_cursor(MediaKind.AUDIO) == 2.0
    assert hub.consumer_gate_limit(MediaKind.AUDIO) == 30.0
    assert hub.consumers_busy() == ["stt"]
    assert hub.consumer_cursor(MediaKind.VIDEO) == 0.0
    hub.register_consumer("waiting", MediaKind.AUDIO, cursor_fn=lambda: 0.0)
    assert hub.consumer_cursor(MediaKind.AUDIO) == 0.0
    hub.remove_plugin("waiting")

    hub.remove_plugin("stt")
    assert hub.consumers_busy() == []
    assert hub.consumer_cursor(MediaKind.AUDIO) == 2.0


_EXTERNAL = """
from core.plugin import Plugin, PluginContext
from core.ui import Side


class External(Plugin):
    id = "external"
    provides = ("external",)

    def setup(self, ctx: PluginContext) -> None:
        ctx.register_service("external", object())
        ctx.ui.add_tab(Side.RIGHT, "external", "外部")

    def teardown(self) -> None:
        pass
"""


def _write_external(path: Path, body: str = _EXTERNAL) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_hot_plug_unload_and_reload(tmp_path: Path) -> None:
    plugin_file = _write_external(tmp_path / "ext_plugin.py")
    runtime = build_runtime(load_config(data_dir=tmp_path))
    runtime.start()
    try:
        loaded = runtime.load_plugin(str(plugin_file))
        assert loaded.id == "external"
        assert "external" in runtime.plugins.loaded
        assert runtime.ui.has_tab("external")

        class Dependent(Plugin):
            id = "dependent"
            requires = ("external",)

            def setup(self, ctx: PluginContext) -> None:
                pass

        runtime.add_plugin(Dependent())
        assert runtime.plugins.loaded[-1] == "dependent"

        with pytest.raises(PluginError):
            runtime.unload_plugin("external")

        removed = runtime.unload_plugin("external", cascade=True)
        assert removed == ["dependent", "external"]
        assert "external" not in runtime.services
        assert not runtime.ui.has_tab("external")

        runtime.load_plugin(str(plugin_file))
        assert runtime.reload_plugin("external").id == "external"
        assert runtime.ui.has_tab("external")
    finally:
        runtime.stop()


def test_dynamic_load_validates_dependencies(tmp_path: Path) -> None:
    runtime = build_runtime(load_config(data_dir=tmp_path))
    runtime.start()

    class Orphan(Plugin):
        id = "orphan"
        requires = ("missing",)

    try:
        with pytest.raises(PluginError):
            runtime.load_plugin(Orphan())
        assert "orphan" not in runtime.plugins.loaded
    finally:
        runtime.stop()


def test_missing_optional_dependency_is_not_an_error(tmp_path: Path) -> None:
    class Soft(Plugin):
        id = "soft"
        optional_requires = ("absent",)
        provides = ("soft",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.register_service("soft", "available")

    runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[Soft()])
    runtime.start()
    try:
        assert runtime.services.require("soft") == "available"
    finally:
        runtime.stop()


def test_uses_actions_must_exist(tmp_path: Path) -> None:
    class Needs(Plugin):
        id = "needs"
        uses_actions = ("absent.message",)

        def setup(self, ctx: PluginContext) -> None:
            pass

    runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[Needs()])
    with pytest.raises(PluginError, match="absent.message"):
        runtime.start()
    assert runtime.started is False


def test_uses_actions_resolves_owner(tmp_path: Path) -> None:
    class Channel(Plugin):
        id = "channel"
        provides = ("channel",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.register_service("channel", object())
            ctx.ui.register_action("channel.send", lambda payload: {"ok": True})

    class Consumer(Plugin):
        id = "consumer"
        requires = ("channel",)
        uses_actions = ("channel.send",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.require("channel")

    runtime = build_runtime(
        load_config(data_dir=tmp_path), plugins=[Channel(), Consumer()]
    )
    runtime.start()
    try:
        assert runtime.ui.action_owner("channel.send") == "channel"
        assert runtime.plugins.loaded == ["channel", "consumer"]
    finally:
        runtime.stop()


def test_ui_action_namespace_is_enforced(tmp_path: Path) -> None:
    class Wrong(Plugin):
        id = "wrong"
        provides = ("wrong",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.ui.register_action("other.ping", lambda payload: {"ok": True})

    runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[Wrong()])
    with pytest.raises(PluginError, match="must register UI actions under"):
        runtime.start()
    assert runtime.started is False


def test_ui_action_unregister_checks_owner(tmp_path: Path) -> None:
    class Owner(Plugin):
        id = "owner"
        provides = ("owner",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.register_service("owner", object())
            ctx.ui.register_action("owner.ping", lambda payload: {"ok": True})

    class Thief(Plugin):
        id = "thief"
        requires = ("owner",)
        provides = ("thief",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.register_service("thief", object())
            with pytest.raises(PluginError, match="cannot register UI actions for"):
                ctx.ui.register_action(
                    "owner.extra", lambda payload: {}, plugin_id="owner"
                )
            with pytest.raises(PluginError, match="cannot unregister UI actions for"):
                ctx.ui.unregister_action("owner.ping", plugin_id="owner")
            with pytest.raises(ValueError, match="belongs to 'owner'"):
                ctx.ui.unregister_action("owner.ping")
            ctx.ui.unregister_action("thief.missing")

    runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[Owner(), Thief()])
    runtime.start()
    try:
        assert runtime.ui.has_action("owner.ping")
        assert runtime.ui.action_owner("owner.ping") == "owner"
        assert not runtime.ui.has_action("owner.extra")
        assert runtime.ui.action_owner("owner.absent") is None
    finally:
        runtime.stop()


def test_unload_survives_failing_reader_cleanup(tmp_path: Path) -> None:
    class BadReader(Plugin):
        id = "reader"
        provides = ("reader",)

        def setup(self, ctx: PluginContext) -> None:
            class Facade:
                def unregister_selection_action(self, owner: str = "") -> None:
                    raise RuntimeError("selection cleanup failed")

                def unregister_activity_provider(self, plugin_id: str) -> None:
                    raise RuntimeError("activity cleanup failed")

            ctx.register_service("reader", Facade())

    class Target(Plugin):
        id = "target"
        provides = ("target",)

        def setup(self, ctx: PluginContext) -> None:
            ctx.register_service("target", object())
            ctx.ui.add_tab(Side.LEFT, "target.tab", "目标")

    runtime = build_runtime(
        load_config(data_dir=tmp_path), plugins=[BadReader(), Target()]
    )
    runtime.start()
    try:
        assert runtime.unload_plugin("target") == ["target"]
        assert "target" not in runtime.plugins.loaded
        assert runtime.services.get("target") is None
        assert not runtime.ui.has_tab("target.tab")
    finally:
        runtime.stop()
