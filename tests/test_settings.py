from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from config import load_config
from core.agent_types import IpcMethod
from core.ui import FormInputType
from plugins.llm_core.plugin import CredentialType, ProviderAuth, ProviderSource
from core.runtime import build_runtime
from core.settings_spec import SECRET_MASK, SettingType, SettingsSpec
from plugins.settings import (
    SettingsPlugin,
    SettingsService,
    SettingsStore,
)

GROUPED = [
    SettingsSpec(
        key="llm.model",
        label="模型",
        type=SettingType.STRING,
        default="deepseek-flash",
        group="LLM",
        order=1,
    ),
    SettingsSpec(
        key="llm.api_key",
        label="API Key",
        type=SettingType.SECRET,
        default="",
        group="LLM",
        order=2,
        placeholder="key...",
    ),
    SettingsSpec(
        key="input.keep_recordings",
        label="本地保留录像",
        type=SettingType.FLOAT,
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


def test_setting_type_serialization() -> None:
    spec = SettingsSpec(key="k", label="K", type=SettingType.SECRET)
    assert spec.type is SettingType.SECRET
    assert spec.secret
    assert json.loads(json.dumps({"type": spec.type})) == {"type": "secret"}
    with pytest.raises(ValueError, match="unsupported setting type"):
        SettingsSpec(key="k", label="K", type="secret")


def test_register_returns_spec_and_get_falls_back_to_default(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    spec = SettingsSpec(key="llm.model", label="模型", default="deepseek-flash")
    assert service.register(spec) is spec
    assert service.get("llm.model") == "deepseek-flash"
    assert service.get("unknown.key") is None
    assert service.get("unknown.key", "fallback") == "fallback"
    assert [s.key for s in service.specs()] == ["llm.model"]
    assert service.get_spec("llm.model") is spec
    assert not (tmp_path / "settings.json").exists()


def test_set_persists_atomically_and_reloads(tmp_path: Path) -> None:
    service = make_service(
        tmp_path, [SettingsSpec(key="a.b", label="B", type=SettingType.INT, default=1)]
    )
    service.set("a.b", 7)
    path = tmp_path / "settings.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a.b": 7}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".settings.json.*")) == []

    reloaded = SettingsService(SettingsStore(path))
    assert reloaded._store.get("a.b") == 7
    assert reloaded.get("a.b") == 7
    assert reloaded.get("missing") is None


def test_set_many_is_atomic_on_validation_error(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(key="x.n", label="N", type=SettingType.INT, default=0),
            SettingsSpec(key="x.m", label="M", type=SettingType.INT, default=0),
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
        (SettingType.STRING, "", 12, "12"),
        (SettingType.TEXT, "", None, ""),
        (SettingType.INT, 0, "42", 42),
        (SettingType.INT, 0, 7.0, 7),
        (SettingType.INT, 0, 7.9, ValueError),
        (SettingType.INT, 0, "abc", ValueError),
        (SettingType.FLOAT, 0.0, "1.5", 1.5),
        (SettingType.FLOAT, 0.0, 2, 2.0),
        (SettingType.FLOAT, 0.0, "nan", ValueError),
        (SettingType.BOOL, False, "true", True),
        (SettingType.BOOL, False, 0, False),
        (SettingType.JSON, None, '{"a": [1, 2]}', {"a": [1, 2]}),
        (SettingType.JSON, None, "", None),
        (SettingType.JSON, None, "{bad", ValueError),
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
        tmp_path,
        [SettingsSpec(key="b", label="B", type=SettingType.BOOL, default=False)],
    )
    assert service.set("b", raw) is expected


def test_select_options_and_invalid_value(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(
                key="s",
                label="S",
                type=SettingType.SELECT,
                default="a",
                options=("a", "b"),
            ),
            SettingsSpec(
                key="n",
                label="N",
                type=SettingType.SELECT,
                default=1,
                options=(1, 2),
            ),
        ],
    )
    assert service.set("s", "b") == "b"
    assert service.set("n", "2") == 2
    with pytest.raises(ValueError):
        service.set("s", "z")
    with pytest.raises(ValueError):
        service.set_many({"s": "c"})
    assert service.get("s") == "b"


def test_secret_masking_and_internal_value(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(
                key="llm.api_key", label="API Key", type=SettingType.SECRET, default=""
            ),
            SettingsSpec(key="llm.model", label="模型", default="m"),
        ],
    )
    assert service.all(public=True) == {"llm.model": "m", "llm.api_key": ""}
    service.set("llm.api_key", "real-token")
    assert service.get("llm.api_key") == "real-token"
    assert service.all()["llm.api_key"] == "real-token"
    assert service.all(public=True)["llm.api_key"] == SECRET_MASK
    described = {item["key"]: item for item in service.describe()}
    assert described["llm.api_key"]["value"] == SECRET_MASK
    assert described["llm.api_key"]["is_set"] is True
    assert described["llm.model"]["value"] == "m"


def test_set_many_keeps_secret_when_empty_or_masked(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        [
            SettingsSpec(
                key="llm.api_key", label="API Key", type=SettingType.SECRET, default=""
            )
        ],
    )
    service.set("llm.api_key", "secret-value")
    assert service.set_many({"llm.api_key": ""}) == {}
    assert service.set_many({"llm.api_key": SECRET_MASK}) == {}
    assert service.get("llm.api_key") == "secret-value"
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


def test_ui_state_has_form_per_group_and_toolbox_items(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        for spec in GROUPED:
            service.register(spec)
        service.set("llm.api_key", "secret-value")

        state = runtime.ui.state()
        assert all(t["id"] != "settings" for t in state["right_tabs"])
        assert all(
            i["id"] != "settings.open"
            for t in state["right_tabs"]
            for i in t["toolbox"]
        )

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
        assert fields["llm.model"]["value"] == "deepseek-flash"
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
                    "input.keep_recordings": "3",
                }
            },
        )
        assert result.ok is True
        assert service.get("llm.model") == "deepseek-reasoner"
        assert service.get("llm.api_key") == "secret-value"
        assert service.get("input.keep_recordings") == 3

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
            {"values": {"input.keep_recordings": "fast"}},
        )
        assert bad.ok is False
        assert "本地保留录像" in bad.message
        assert service.get("input.keep_recordings") == 3

        empty = runtime.ui.dispatch_action("settings.save", {"values": {}})
        assert empty.ok is True
    finally:
        runtime.stop()


def test_settings_changed_event_payload(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(
            SettingsSpec(key="x.n", label="N", type=SettingType.INT, default=1)
        )
        events: list[dict] = []
        runtime.bus.on(
            "settings.changed", lambda event: events.append(dict(event.payload))
        )

        service.set("x.n", "5")
        assert runtime.bus.flush(2.0)
        assert events == [{"key": "x.n", "old": 1, "new": 5}]

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
        service.register(
            SettingsSpec(key="x.n", label="N", type=SettingType.INT, default=0)
        )
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
            SettingsSpec(key="llm.model", label="模型", default="deepseek-flash")
        )
        service.set("llm.model", "qwen")
    finally:
        runtime.stop()

    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(
            SettingsSpec(key="llm.model", label="模型", default="deepseek-flash")
        )
        assert service.get("llm.model") == "qwen"
        assert stat.S_IMODE((tmp_path / "settings.json").stat().st_mode) == 0o600
    finally:
        runtime.stop()


def test_action_and_readonly_settings_fields(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    try:
        service = runtime.services.require("settings")
        service.register(
            SettingsSpec(
                key="demo.status",
                label="状态",
                type=SettingType.STRING,
                default="原始",
                readonly=True,
                group="Demo",
            )
        )
        service.register(
            SettingsSpec(
                key="demo.login",
                label="登录",
                type=SettingType.ACTION,
                action="demo.login",
                button_label="去登录",
                group="Demo",
            )
        )
        runtime.ui.open_page("settings:main")
        blocks = runtime.ui.state()["page_content"]["settings:main"]
        forms = [
            block
            for tab in blocks[0]["tabs"]
            for block in tab["blocks"]
            if block["type"] == "form"
        ]
        fields = {field["key"]: field for form in forms for field in form["fields"]}
        assert fields["demo.status"]["readonly"] is True
        assert fields["demo.login"]["type"] == "action"
        assert fields["demo.login"]["action"] == "demo.login"
        assert fields["demo.login"]["button_label"] == "去登录"
        assert "name" not in fields["demo.login"]

        result = runtime.ui.dispatch_action(
            "settings.save", {"values": {"demo.status": "被篡改"}}
        )
        assert result.ok is True
        assert service.get("demo.status") == "原始"
    finally:
        runtime.stop()


class ConfigurationRuntime:
    """In-memory stand-in for the Bun runtime's configuration surface."""

    def __init__(self, directory: Path, *, settings=None) -> None:
        self.directory = directory
        self.calls: list[tuple[IpcMethod, dict]] = []
        self.fail_configure = False
        self.closed = False
        self.providers = {
            "lecture": {
                "baseUrl": "http://lecture.test/v1",
                "api": "openai-completions",
                "auth": "apiKey",
                "configured": True,
                "models": [
                    {
                        "id": "example",
                        "contextWindow": 32000,
                        "maxTokens": 4096,
                        "input": ["text", "image"],
                        "reasoning": True,
                    }
                ],
            },
            "opencode": {
                "baseUrl": "https://opencode.ai/inference/go/openai/v1",
                "api": "openai-completions",
                "auth": "oauth",
                "configured": False,
                "models": [
                    {
                        "id": "flash",
                        "contextWindow": 200000,
                        "maxTokens": 8000,
                        "input": ["text"],
                        "reasoning": False,
                    }
                ],
            },
        }
        self.selectors = {
            "default": "lecture/example:high",
            "note_writer": "lecture/example:medium",
            "board_recognition": "lecture/example:off",
        }
        self.discovered = [
            {
                "id": "extra",
                "name": "Extra Model",
                "contextWindow": 100000,
                "maxTokens": 2000,
                "input": ["text", "image"],
                "reasoning": True,
            },
            {
                "id": "plain",
                "contextWindow": 64000,
                "maxTokens": 4000,
                "input": ["text"],
                "reasoning": False,
            },
        ]
        self.configuration = {
            "models_yaml": "providers: {}\n",
            "roles_yaml": "default: lecture/example:high\n",
        }

    def _resolved(self, role: str) -> dict:
        selector = self.selectors.get(role, self.selectors["default"])
        target, _, level = selector.partition(":")
        provider, _, model = target.partition("/")
        return {"provider": provider, "model": model, "thinkingLevel": level or None}

    def _catalog(self) -> dict:
        providers = []
        for pid, provider in self.providers.items():
            providers.append(
                {
                    "id": pid,
                    "api": provider["api"],
                    "auth": provider["auth"],
                    "baseUrl": provider["baseUrl"],
                    "configured": provider["configured"],
                    "source": "project",
                    "models": [dict(model) for model in provider["models"]],
                }
            )
        return {
            **self.configuration,
            "providers": providers,
            "selectors": dict(self.selectors),
            "roles": {role: self._resolved(role) for role in self.selectors},
            "usage": {},
        }

    def call(self, method: IpcMethod, **params):
        self.calls.append((method, params))
        if method == IpcMethod.CONFIGURE:
            if self.fail_configure:
                raise ValueError("Invalid role selector")
            self.configuration.update(
                models_yaml=params["models"], roles_yaml=params["roles"]
            )
        elif method == IpcMethod.CATALOG:
            return self._catalog()
        elif method == IpcMethod.INFO:
            return self._catalog()
        elif method == IpcMethod.DISCOVER:
            return {"base_url": "", "models": [dict(m) for m in self.discovered]}
        elif method == IpcMethod.PROVIDER_SET:
            provider = self.providers.setdefault(
                params["id"],
                {
                    "models": [],
                    "api": params["api"],
                    "auth": params["auth"],
                    "configured": params["auth"] != "apiKey",
                },
            )
            provider.update(api=params["api"], auth=params["auth"])
            provider["baseUrl"] = params["base_url"]
            if not provider["models"]:
                provider["models"] = [dict(m) for m in self.discovered]
        elif method == IpcMethod.PROVIDER_REMOVE:
            self.providers.pop(params["provider"], None)
            for role, selector in list(self.selectors.items()):
                if selector.startswith(f"{params['provider']}/"):
                    self.selectors[role] = "@default"
        elif method == IpcMethod.MODELS_SET:
            if not params["models"]:
                raise ValueError("至少保留一个模型")
            provider = self.providers[params["provider"]]
            provider["models"] = [dict(m) for m in params["models"]]
        elif method == IpcMethod.ROLE_SET:
            self.selectors[params["role"]] = params["selector"]
        elif method == IpcMethod.CREDENTIALS:
            self.providers[params["provider"]]["configured"] = True
        return self._catalog()

    def close(self):
        self.closed = True


def make_llm_settings_runtime(tmp_path, monkeypatch):
    from plugins.llm_core import plugin as llm_plugin

    monkeypatch.setattr(llm_plugin, "RuntimeClient", ConfigurationRuntime)
    plugin = llm_plugin.LLMPlugin()
    runtime = build_runtime(
        load_config(data_dir=tmp_path), plugins=[SettingsPlugin(), plugin]
    )
    runtime.start()
    return runtime, plugin, runtime.services.require("llm")


def llm_tabs(runtime) -> dict[str, list[dict]]:
    runtime.ui.open_page("settings:main")
    tabs = runtime.ui.state()["page_content"]["settings:main"][0]["tabs"]
    section = next(tab for tab in tabs if tab["title"] == "LLM")
    nested = section["blocks"][0]["tabs"]
    return {tab["title"]: tab["blocks"] for tab in nested}


def _table(blocks, action=None) -> dict:
    tables = [block for block in blocks if block["type"] == "table"]
    if action is None:
        return tables[0]
    return next(
        table
        for table in tables
        if any(item.get("action") == action for item in table.get("row_actions", []))
    )


def _form(blocks, action: str) -> dict:
    return next(
        block
        for block in blocks
        if block["type"] == "form" and block.get("submit", {}).get("action") == action
    )


def test_settings_page_is_a_workflow(tmp_path, monkeypatch):
    runtime, _plugin, _backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        tabs = llm_tabs(runtime)
        assert list(tabs) == ["概览", "提供方", "模型", "角色", "原始配置"]
        overview = json.dumps(tabs["概览"], ensure_ascii=False)
        assert "默认模型" in overview
        assert "opencode" in overview
        assert "未配置凭据" in overview
        assert "lecture/example" in overview
    finally:
        runtime.stop()


def test_external_yaml_editors_save_without_settings_copies(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        tabs = llm_tabs(runtime)
        editors = _form(tabs["原始配置"], "llm.configure")
        assert {field["name"] for field in editors["fields"]} == {
            "models_yaml",
            "roles_yaml",
        }
        assert all(
            field["input"] == FormInputType.TEXTAREA for field in editors["fields"]
        )

        result = runtime.ui.dispatch_action(
            "llm.configure",
            {
                "values": {
                    "models_yaml": "providers: {lecture: {}}\n",
                    "roles_yaml": "default: lecture/example\n",
                }
            },
        )
        assert result.ok
        assert backend.calls[-1] == (
            IpcMethod.CONFIGURE,
            {
                "models": "providers: {lecture: {}}\n",
                "roles": "default: lecture/example\n",
            },
        )
        service = runtime.services.require("settings")
        assert "models_yaml" not in service.all()
        assert "roles_yaml" not in service.all()
    finally:
        runtime.stop()


def test_provider_cards_carry_credentials_and_presets(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        blocks = llm_tabs(runtime)["提供方"]
        cards = next(block for block in blocks if block["type"] == "cards")
        assert [card["id"] for card in cards["items"]] == ["lecture", "opencode"]
        lecture, opencode = cards["items"]
        assert [action["action"] for action in lecture["actions"]] == [
            "llm.logout",
            "llm.provider.remove",
        ]
        assert "llm.opencode_login" in [
            action["action"] for action in opencode["actions"]
        ]
        assert {meta[0] for meta in lecture["meta"]} == {"协议", "鉴权", "凭据", "模型"}
        assert _form(lecture["blocks"], "llm.provider.save")["fields"][-1][
            "visible_when"
        ] == {
            "key": "auth",
            "values": ["apiKey"],
        }

        add = _form(blocks, "llm.provider.save")
        fields = {field["name"]: field for field in add["fields"]}
        assert fields["preset"]["options"] == ["opencode-go", "deepseek", "custom"]
        assert fields["base_url"]["visible_when"] == {
            "key": "preset",
            "values": ["custom"],
        }
        assert fields["key"]["visible_when"] == {
            "key": "preset",
            "values": ["deepseek", "custom"],
        }

        result = runtime.ui.dispatch_action(
            "llm.provider.save",
            {"values": {"preset": "opencode-go", "provider": "", "key": ""}},
        )
        assert result.ok, result.message
        assert backend.calls[-1] == (
            IpcMethod.PROVIDER_SET,
            {
                "id": "opencode-go",
                "base_url": "https://opencode.ai/inference/go/openai/v1",
                "api": "openai-completions",
                "auth": "oauth",
            },
        )
        assert [m["id"] for m in backend.providers["opencode-go"]["models"]] == [
            "extra",
            "plain",
        ]
    finally:
        runtime.stop()


def test_endpoint_check_reports_gaps_then_adds_selected_models(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        result = runtime.ui.dispatch_action(
            "llm.check", {"values": {"provider": "lecture"}}
        )
        assert result.ok, result.message
        assert backend.calls[-1][0] == IpcMethod.DISCOVER
        assert not any(call[0] == IpcMethod.MODELS_SET for call in backend.calls)

        blocks = llm_tabs(runtime)["模型"]
        table = _table(blocks, "llm.model.add")
        assert [row["model"] for row in table["rows"]] == ["extra", "plain"]
        assert table["rows"][0]["context"] == "100K"

        added = runtime.ui.dispatch_action(
            "llm.model.add", {"values": {"provider": "lecture", "model": "extra"}}
        )
        assert added.ok, added.message
        method, params = backend.calls[-1]
        assert method == IpcMethod.MODELS_SET
        assert [model["id"] for model in params["models"]] == ["example", "extra"]
        assert params["models"][1]["contextWindow"] == 100000

        blocks = llm_tabs(runtime)["模型"]
        assert "1 个尚未添加" in json.dumps(blocks, ensure_ascii=False)
        everything = runtime.ui.dispatch_action(
            "llm.model.add_all", {"values": {"provider": "lecture"}}
        )
        assert everything.ok, everything.message
        assert [m["id"] for m in backend.providers["lecture"]["models"]] == [
            "example",
            "extra",
            "plain",
        ]
        again = runtime.ui.dispatch_action(
            "llm.model.add_all", {"values": {"provider": "lecture"}}
        )
        assert not again.ok
    finally:
        runtime.stop()


def test_endpoint_check_failure_is_shown_inline(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        original = backend.call

        def call(method, **params):
            if method == IpcMethod.DISCOVER:
                backend.calls.append((method, params))
                raise ValueError("Model discovery failed: HTTP 401")
            return original(method, **params)

        monkeypatch.setattr(backend, "call", call)
        result = runtime.ui.dispatch_action(
            "llm.check", {"values": {"provider": "lecture"}}
        )
        assert not result.ok
        assert "HTTP 401" in result.message
        blocks = llm_tabs(runtime)["模型"]
        assert any(
            "检查失败" in block.get("text", "")
            for block in blocks
            if block["type"] == "markdown"
        )
    finally:
        runtime.stop()


def test_model_page_selects_then_edits_one_model(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        blocks = llm_tabs(runtime)["模型"]
        rows = next(block for block in blocks if block["type"] == "rows")
        assert [item["title"] for item in rows["items"]] == ["lecture", "opencode"]
        assert rows["items"][0]["active"] is True
        table = _table(blocks, "llm.select.model")
        assert [row["model"] for row in table["rows"]] == ["example"]
        assert table["rows"][0]["context"] == "32K"
        assert table["rows"][0]["roles"] == "default、board_recognition、note_writer"
        editor = _form(blocks, "llm.model.save")
        values = {field["name"]: field["value"] for field in editor["fields"]}
        assert values["model"] == "example"
        assert values["context_window"] == 32000

        runtime.ui.dispatch_action(
            "llm.select.model", {"values": {"provider": "opencode", "model": "flash"}}
        )
        blocks = llm_tabs(runtime)["模型"]
        editor = _form(blocks, "llm.model.save")
        values = {field["name"]: field["value"] for field in editor["fields"]}
        assert (values["provider"], values["model"]) == ("opencode", "flash")
        assert values["context_window"] == 200000

        saved = runtime.ui.dispatch_action(
            "llm.model.save",
            {
                "values": {
                    "provider": "opencode",
                    "model": "flash",
                    "context_window": 123000,
                    "max_tokens": 9000,
                    "input_text": True,
                    "input_image": True,
                    "reasoning": False,
                }
            },
        )
        assert saved.ok, saved.message
        assert backend.providers["opencode"]["models"][0] == {
            "id": "flash",
            "contextWindow": 123000,
            "maxTokens": 9000,
            "input": ["text", "image"],
            "reasoning": False,
        }
    finally:
        runtime.stop()


def test_model_delete_is_blocked_while_roles_use_it(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        blocked = runtime.ui.dispatch_action(
            "llm.model.remove", {"values": {"provider": "lecture", "model": "example"}}
        )
        assert not blocked.ok
        assert "仍被角色" in blocked.message
        assert "note_writer" in blocked.message

        backend.selectors["note_writer"] = "opencode/flash:high"
        backend.selectors["board_recognition"] = "opencode/flash:off"
        backend.selectors["default"] = "opencode/flash:high"
        backend.providers["lecture"]["models"].append(
            {
                "id": "spare",
                "contextWindow": 8000,
                "maxTokens": 1000,
                "input": ["text"],
                "reasoning": False,
            }
        )
        allowed = runtime.ui.dispatch_action(
            "llm.model.remove", {"values": {"provider": "lecture", "model": "example"}}
        )
        assert allowed.ok, allowed.message
        assert [m["id"] for m in backend.providers["lecture"]["models"]] == ["spare"]
    finally:
        runtime.stop()


def test_role_page_selects_then_edits_one_role(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        blocks = llm_tabs(runtime)["角色"]
        table = _table(blocks, "llm.select.role")
        writer = next(row for row in table["rows"] if row["role"] == "note_writer")
        assert writer["selector"] == "lecture/example:medium"
        assert writer["active"] == "lecture/example"
        assert [column["key"] for column in table["columns"]] == [
            "title",
            "role",
            "selector",
            "active",
            "status",
        ]
        board = next(row for row in table["rows"] if row["role"] == "board_recognition")
        assert board["status"] == "图像可用"

        backend.selectors["board_recognition"] = "opencode/flash:off"
        blocks = llm_tabs(runtime)["角色"]
        board = next(
            row
            for row in _table(blocks, "llm.select.role")["rows"]
            if row["role"] == "board_recognition"
        )
        assert board["status"] == "缺少 image 输入"
        editor = _form(blocks, "llm.role.save")
        values = {field["name"]: field["value"] for field in editor["fields"]}
        assert (values["role"], values["model"], values["level"]) == (
            "default",
            "lecture/example",
            "high",
        )

        runtime.ui.dispatch_action(
            "llm.select.role", {"values": {"role": "note_writer"}}
        )
        blocks = llm_tabs(runtime)["角色"]
        editor = _form(blocks, "llm.role.save")
        values = {field["name"]: field["value"] for field in editor["fields"]}
        assert (values["role"], values["level"]) == ("note_writer", "medium")

        rejected = runtime.ui.dispatch_action(
            "llm.role.save",
            {
                "values": {
                    "role": "board_recognition",
                    "model": "opencode/flash",
                    "level": "off",
                }
            },
        )
        assert not rejected.ok
        assert "image" in rejected.message

        saved = runtime.ui.dispatch_action(
            "llm.role.save",
            {
                "values": {
                    "role": "board_recognition",
                    "model": "lecture/example",
                    "level": "off",
                }
            },
        )
        assert saved.ok, saved.message
        assert backend.selectors["board_recognition"] == "lecture/example:off"
    finally:
        runtime.stop()


def test_invalid_yaml_keeps_effective_configuration(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        previous = dict(backend.configuration)
        backend.fail_configure = True
        result = runtime.ui.dispatch_action(
            "llm.configure",
            {
                "values": {
                    "models_yaml": "broken",
                    "roles_yaml": "invalid",
                }
            },
        )
        assert not result.ok
        assert "Invalid role selector" in result.message
        assert backend.configuration == previous
        assert runtime.services.require("settings").all() == {}
    finally:
        runtime.stop()


def test_provider_credentials_are_write_only_runtime_actions(tmp_path, monkeypatch):
    runtime, _plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        result = runtime.ui.dispatch_action(
            "llm.credentials", {"values": {"provider": "lecture", "key": "secret-key"}}
        )
        assert result.ok
        assert backend.calls[-1] == (
            IpcMethod.CREDENTIALS,
            {"provider": "lecture", "key": "secret-key"},
        )
        runtime.ui.open_page("settings:main")
        assert "secret-key" not in json.dumps(runtime.ui.state(), ensure_ascii=False)
        assert runtime.services.require("settings").all() == {}
        result = runtime.ui.dispatch_action(
            "llm.logout", {"values": {"provider": "lecture"}}
        )
        assert result.ok
        assert backend.calls[-1] == (IpcMethod.LOGOUT, {"provider": "lecture"})
    finally:
        runtime.stop()


def device_flow_fixture(monkeypatch, *, tokens=True):
    from integrations.opencode_go import DeviceFlow, TokenSet
    from plugins.llm_core import plugin as module

    state = {"started": 0, "cancelled": []}

    def start():
        state["started"] += 1
        return DeviceFlow(
            device_code=f"device-{state['started']}",
            user_code=f"CODE-{state['started']}",
            verification_uri_complete=f"https://opencode.ai/verify?code=CODE-{state['started']}",
            interval=1,
            expires_in=600,
        )

    def poll(flow):
        return TokenSet("access", "refresh", 9999, "org") if tokens else None

    monkeypatch.setattr(module, "start_device_flow", start)
    monkeypatch.setattr(module, "poll_device_token", poll)
    monkeypatch.setattr(module.webbrowser, "open", lambda _url: True)
    return state


def test_device_login_imports_oauth_into_runtime(tmp_path, monkeypatch):
    state = device_flow_fixture(monkeypatch)
    runtime, plugin, backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        result = runtime.ui.dispatch_action(
            "llm.opencode_login", {"values": {"provider": "opencode"}}
        )
        assert result.ok
        # The URL is not only in the toast: it is a page block the user can copy.
        assert "https://opencode.ai/verify" not in result.message
        login = plugin._login
        assert login is not None
        assert login.thread is not None
        login.thread.join(timeout=2)
        assert not login.thread.is_alive()
        assert state["started"] == 1
        assert (
            IpcMethod.IMPORT_CREDENTIALS,
            {
                "provider": "opencode",
                "credential": {
                    "type": CredentialType.OAUTH,
                    "access": "access",
                    "refresh": "refresh",
                    "expires": 9999,
                    "org_id": "org",
                },
            },
        ) in backend.calls
        assert plugin._login is None
        blocks = llm_tabs(runtime)["提供方"]
        assert "已关联 OpenCode 套餐" in json.dumps(blocks, ensure_ascii=False)
        assert runtime.services.require("settings").all() == {}
        result = runtime.ui.dispatch_action(
            "llm.logout", {"values": {"provider": "opencode"}}
        )
        assert result.ok
        assert backend.calls[-1] == (IpcMethod.LOGOUT, {"provider": "opencode"})
    finally:
        runtime.stop()


def test_pending_login_keeps_a_copyable_link_on_the_page(tmp_path, monkeypatch):
    state = device_flow_fixture(monkeypatch, tokens=False)
    runtime, plugin, _backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        first = runtime.ui.dispatch_action(
            "llm.opencode_login", {"values": {"provider": "opencode"}}
        )
        assert first.ok
        # The browser opens this; the page keeps it as a fallback.
        assert first.open_url == "https://opencode.ai/verify?code=CODE-1"
        blocks = llm_tabs(runtime)["提供方"]
        text = json.dumps(blocks, ensure_ascii=False)
        assert "等待 OpenCode 授权" in text
        assert "https://opencode.ai/verify?code=CODE-1" in text
        assert "CODE-1" in text
        assert "10 分钟内有效" in text
        assert "重新开始授权" in text and "取消授权" in text
        buttons = next(block for block in blocks if block["type"] == "buttons")
        assert [button["action"] for button in buttons["buttons"]] == [
            "llm.opencode_login",
            "llm.opencode_cancel",
        ]

        # A second click must not silently start another flow.
        again = runtime.ui.dispatch_action(
            "llm.opencode_login", {"values": {"provider": "opencode"}}
        )
        assert again.ok
        assert "仍在进行" in again.message
        assert state["started"] == 1

        restarted = runtime.ui.dispatch_action(
            "llm.opencode_login",
            {"values": {"provider": "opencode", "force": True}},
        )
        assert restarted.ok
        assert state["started"] == 2
        assert plugin._login is not None
        assert plugin._login.flow.user_code == "CODE-2"
        assert "CODE-2" in json.dumps(llm_tabs(runtime)["提供方"], ensure_ascii=False)

        cancelled = runtime.ui.dispatch_action(
            "llm.opencode_cancel", {"values": {"provider": "opencode"}}
        )
        assert cancelled.ok
        assert plugin._login is None
        blocks = llm_tabs(runtime)["提供方"]
        assert "等待 OpenCode 授权" not in json.dumps(blocks, ensure_ascii=False)
        assert "已取消授权" in json.dumps(blocks, ensure_ascii=False)
    finally:
        runtime.stop()


def test_expired_login_allows_a_fresh_start(tmp_path, monkeypatch):
    state = device_flow_fixture(monkeypatch, tokens=False)
    runtime, plugin, _backend = make_llm_settings_runtime(tmp_path, monkeypatch)
    try:
        runtime.ui.dispatch_action(
            "llm.opencode_login", {"values": {"provider": "opencode"}}
        )
        assert plugin._login is not None
        plugin._login.expires_at = 0.0
        # The stale link is not rendered any more.
        assert "opencode.ai/verify" not in json.dumps(
            llm_tabs(runtime)["提供方"], ensure_ascii=False
        )
        result = runtime.ui.dispatch_action(
            "llm.opencode_login", {"values": {"provider": "opencode"}}
        )
        assert result.ok
        assert state["started"] == 2
        assert plugin._login is not None
    finally:
        runtime.stop()


def test_unload_preserves_settings_owned_by_pipeline(tmp_path):
    service = SettingsService(SettingsStore(tmp_path / "settings.json"))
    service.register(
        SettingsSpec(
            key="summarize.review_enabled",
            label="Review",
            type=SettingType.BOOL,
            default=True,
        ),
        plugin_id="notes-pipeline",
    )
    service.register(
        SettingsSpec(
            key="summarize.on_stop",
            label="Stop",
            type=SettingType.STRING,
            default="skip",
        ),
        plugin_id="summarize",
    )
    service.unregister(plugin_id="summarize")
    assert service.get_spec("summarize.review_enabled") is not None
    assert service.get_spec("summarize.on_stop") is None
    service.unregister(plugin_id="notes-pipeline")
    assert service.get_spec("summarize.review_enabled") is None
