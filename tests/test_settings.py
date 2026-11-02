"""Tests for the P2 settings plugin (store, service, UI)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from config import load_config
from core.runtime import build_runtime
from plugins.settings import (
    SECRET_MASK,
    SettingsPlugin,
    SettingsService,
    SettingsSpec,
    SettingsStore,
)

GROUPED = [
    SettingsSpec(
        key="llm.model",
        label="模型",
        type="string",
        default="deepseek-chat",
        group="LLM",
        order=1,
    ),
    SettingsSpec(
        key="llm.api_key",
        label="API Key",
        type="secret",
        default="",
        group="LLM",
        order=2,
        placeholder="key...",
    ),
    SettingsSpec(
        key="input.playback_speed",
        label="播放速度",
        type="float",
        default=1.0,
        group="输入",
        order=1,
    ),
]


def make_service(tmp_path: Path, specs=()) -> SettingsService:
    store = SettingsStore(tmp_path / "settings.json")
    service = SettingsService(store)
    for spec in specs:
        service.register(spec)
    return service


def make_runtime(tmp_path: Path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[SettingsPlugin()])
    runtime.start()
    return runtime


# ----------------------------------------------------------------------
# Spec / service
# ----------------------------------------------------------------------
def test_register_returns_spec_and_get_falls_back_to_default(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    spec = SettingsSpec(key="llm.model", label="模型", default="deepseek-chat")
    assert service.register(spec) is spec
    assert service.get("llm.model") == "deepseek-chat"
    assert service.get("unknown.key") is None
    assert service.get("unknown.key", "fallback") == "fallback"
    assert [s.key for s in service.specs()] == ["llm.model"]
    assert service.get_spec("llm.model") is spec
    # Defaults must not be persisted before an explicit set().
    assert not (tmp_path / "settings.json").exists()


def test_set_persists_atomically_and_reloads(tmp_path: Path) -> None:
    service = make_service(
        tmp_path, [SettingsSpec(key="a.b", label="B", type="int", default=1)]
    )
    service.set("a.b", 7)
    path = tmp_path / "settings.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a.b": 7}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # No temp files left behind (atomic tmp + os.replace).
    assert list(tmp_path.glob(".settings.json.*")) == []

    reloaded = SettingsService(SettingsStore(path))
    assert reloaded.store.get("a.b") == 7
    assert reloaded.get("a.b") == 7
    assert reloaded.get("missing") is None


def test_set_many_is_atomic_on_validation_error(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(key="x.n", label="N", type="int", default=0),
            SettingsSpec(key="x.m", label="M", type="int", default=0),
        ],
    )
    with pytest.raises(ValueError):
        service.set_many({"x.n": "3", "x.m": "not-a-number"})
    assert service.get("x.n") == 0
    assert not (tmp_path / "settings.json").exists()

    assert service.set_many({"x.n": "3", "x.m": "4"}) == {"x.n": 3, "x.m": 4}
    assert service.all() == {"x.n": 3, "x.m": 4}


@pytest.mark.parametrize(
    ("spec_type", "default", "raw", "expected"),
    [
        ("string", "", 12, "12"),
        ("text", "", None, ""),
        ("int", 0, "42", 42),
        ("int", 0, 7.0, 7),
        ("int", 0, 7.9, ValueError),
        ("int", 0, "abc", ValueError),
        ("float", 0.0, "1.5", 1.5),
        ("float", 0.0, 2, 2.0),
        ("float", 0.0, "nan", ValueError),
        ("bool", False, "true", True),
        ("bool", False, 0, False),
        ("json", None, '{"a": [1, 2]}', {"a": [1, 2]}),
        ("json", None, "", None),
        ("json", None, "{bad", ValueError),
    ],
)
def test_type_conversion(tmp_path: Path, spec_type, default, raw, expected) -> None:
    service = make_service(
        tmp_path,
        [SettingsSpec(key="k", label="K", type=spec_type, default=default)],
    )
    if expected is ValueError:
        with pytest.raises(ValueError):
            service.set("k", raw)
        assert service.get("k") == default
    else:
        assert service.set("k", raw) == expected
        assert service.get("k") == expected
        assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) == {
            "k": expected
        }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", True),
        ("ON", True),
        ("1", True),
        ("off", False),
        ("FALSE", False),
        ("0", False),
        ("", False),
        (2, True),
        (True, True),
    ],
)
def test_bool_strings(tmp_path: Path, raw, expected) -> None:
    service = make_service(
        tmp_path, [SettingsSpec(key="b", label="B", type="bool", default=False)]
    )
    assert service.set("b", raw) is expected


def test_select_options_and_invalid_value(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(
                key="s",
                label="S",
                type="select",
                default="a",
                options=("a", "b"),
            ),
            SettingsSpec(
                key="n",
                label="N",
                type="select",
                default=1,
                options=(1, 2),
            ),
        ],
    )
    assert service.set("s", "b") == "b"
    assert service.set("n", "2") == 2
    with pytest.raises(ValueError, match="a.*b|取值"):
        service.set("s", "z")
    with pytest.raises(ValueError):
        service.set_many({"s": "c"})
    assert service.get("s") == "b"


def test_secret_masking_and_internal_value(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(key="llm.api_key", label="API Key", type="secret", default=""),
            SettingsSpec(key="llm.model", label="模型", default="m"),
        ],
    )
    assert service.public_all() == {"llm.model": "m", "llm.api_key": ""}
    service.set("llm.api_key", "real-token")
    assert service.get("llm.api_key") == "real-token"
    assert service.all()["llm.api_key"] == "real-token"
    assert service.public_all()["llm.api_key"] == SECRET_MASK
    assert service.all(public=True)["llm.api_key"] == SECRET_MASK
    described = {item["key"]: item for item in service.describe()}
    assert described["llm.api_key"]["value"] == SECRET_MASK
    assert described["llm.api_key"]["is_set"] is True
    assert described["llm.model"]["value"] == "m"


def test_set_many_keeps_secret_when_empty_or_masked(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [SettingsSpec(key="llm.api_key", label="API Key", type="secret", default="")],
    )
    service.set("llm.api_key", "secret-value")
    assert service.set_many({"llm.api_key": ""}) == {}
    assert service.set_many({"llm.api_key": SECRET_MASK}) == {}
    assert service.get("llm.api_key") == "secret-value"
    # An explicit (non-empty) value replaces it, and "" via set() clears it.
    service.set_many({"llm.api_key": "new-secret"})
    assert service.get("llm.api_key") == "new-secret"
    service.set("llm.api_key", "")
    assert service.get("llm.api_key") == ""


def test_corrupt_json_falls_back_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json at all", encoding="utf-8")

    store = SettingsStore(path)
    assert store.to_dict() == {}
    backups = list(tmp_path.glob("settings.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not json at all"

    service = SettingsService(store)
    service.register(SettingsSpec(key="k", label="K", default="v"))
    assert service.get("k") == "v"
    service.set("k", "written")
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "written"}


# ----------------------------------------------------------------------
# Plugin wiring / UI
# ----------------------------------------------------------------------
def test_plugin_registers_service_and_tab(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        assert isinstance(service, SettingsService)
        assert service.store.path == tmp_path / "settings.json"
        snapshot = runtime.ui.snapshot()
        assert snapshot["tabs"]["settings"] == "right"
        assert snapshot["pages"]["settings:main"] == "设置"
        assert "settings.save" in snapshot["actions"]
        tab = next(t for t in runtime.ui.state()["right_tabs"] if t["id"] == "settings")
        assert tab["title"] == "设置"
    finally:
        runtime.stop()


def test_page_shows_hint_without_specs(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        runtime.ui.open_page("settings:main")
        blocks = runtime.ui.state()["page_content"]["settings:main"]
        assert blocks and blocks[0]["type"] == "markdown"
        assert "尚未注册任何设置项" in blocks[0]["text"]
    finally:
        runtime.stop()


def test_ui_state_has_form_per_group_and_toolbox_items(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        for spec in GROUPED:
            service.register(spec)
        service.set("llm.api_key", "secret-value")

        state = runtime.ui.state()
        tab = next(t for t in state["right_tabs"] if t["id"] == "settings")
        item_ids = [item["id"] for item in tab["toolbox"]]
        assert item_ids == ["settings.open"]
        settings_item = tab["toolbox"][0]
        assert any(block["type"] == "markdown" for block in settings_item["blocks"])
        assert not any(block["type"] == "form" for block in settings_item["blocks"])

        runtime.ui.open_page("settings:main")
        blocks = runtime.ui.state()["page_content"]["settings:main"]
        assert blocks[0]["type"] == "tabs"
        assert blocks[0]["id"] == "settings.groups"
        forms = [
            block
            for group_tab in blocks[0]["tabs"]
            for block in group_tab["blocks"]
            if block["type"] == "form"
        ]
        assert len(forms) == len({spec.group for spec in GROUPED})
        fields = {field["name"]: field for form in forms for field in form["fields"]}
        assert set(fields) == {spec.key for spec in GROUPED}
        field = fields["llm.api_key"]
        assert field["secret"] is True
        assert field["label"] == "API Key"
        assert field["type"] == "secret"
        assert field["value"] == ""
        assert field["secret_length"] == len("secret-value")
        assert field["is_set"] is True
        assert field["placeholder"] == "key..."
        assert fields["llm.model"]["value"] == "deepseek-chat"
        assert forms[0]["submit"]["action"] == "settings.save"
    finally:
        runtime.stop()


def test_action_save_converts_and_keeps_secret(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        for spec in GROUPED:
            service.register(spec)
        service.set("llm.api_key", "secret-value")

        result = runtime.ui.dispatch_action(
            "settings.save",
            {
                "values": {
                    "llm.model": "deepseek-reasoner",
                    "llm.api_key": SECRET_MASK,
                    "input.playback_speed": "1.5",
                }
            },
        )
        assert result.ok is True
        assert "已保存" in result.message
        assert service.get("llm.model") == "deepseek-reasoner"
        assert service.get("llm.api_key") == "secret-value"
        assert service.get("input.playback_speed") == 1.5

        # An empty secret means "keep existing", not "clear".
        kept = runtime.ui.dispatch_action(
            "settings.save", {"values": {"llm.api_key": ""}}
        )
        assert kept.ok is True
        assert service.get("llm.api_key") == "secret-value"

        cleared = runtime.ui.dispatch_action(
            "settings.clear_secret", {"values": {"key": "llm.api_key"}}
        )
        assert cleared.ok is True
        assert service.get("llm.api_key") == ""

        bad = runtime.ui.dispatch_action(
            "settings.save",
            {"values": {"input.playback_speed": "fast"}},
        )
        assert bad.ok is False
        assert "播放速度" in bad.message
        assert service.get("input.playback_speed") == 1.5

        empty = runtime.ui.dispatch_action("settings.save", {"values": {}})
        assert empty.ok is True
        assert "没有需要保存" in empty.message
    finally:
        runtime.stop()


def test_settings_changed_event_payload(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(SettingsSpec(key="x.n", label="N", type="int", default=1))
        events: list[dict] = []
        runtime.bus.on("settings.changed", lambda event: events.append(dict(event.payload)))

        service.set("x.n", "5")
        assert runtime.bus.flush(2.0)
        assert events == [{"key": "x.n", "old": 1, "new": 5}]

        # Unchanged values do not emit; resetting back to the default does.
        service.set("x.n", 5)
        assert runtime.bus.flush(2.0)
        assert len(events) == 1
        service.reset("x.n")
        assert runtime.bus.flush(2.0)
        assert events[-1] == {"key": "x.n", "old": 5, "new": 1}
    finally:
        runtime.stop()


def test_ui_refresh_after_settings_change(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        before = runtime.ui.state()["version"]
        service.register(SettingsSpec(key="x.n", label="N", type="int", default=0))
        after_register = runtime.ui.state()["version"]
        assert after_register > before
        service.set("x.n", 3)
        assert runtime.ui.state()["version"] > after_register
    finally:
        runtime.stop()


def test_setting_persists_across_runtime_restart(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(
            SettingsSpec(key="llm.model", label="模型", default="deepseek-chat")
        )
        service.set("llm.model", "qwen")
    finally:
        runtime.stop()

    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(
            SettingsSpec(key="llm.model", label="模型", default="deepseek-chat")
        )
        assert service.get("llm.model") == "qwen"
        assert stat.S_IMODE((tmp_path / "settings.json").stat().st_mode) == 0o600
    finally:
        runtime.stop()
