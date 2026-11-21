"""Tests for the tutor plugin: modes, prompt composition, propose-only memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugins.tutor.modes import ModeStore
from plugins.tutor.plugin import TutorPlugin
from plugins.tutor.prompts import COMMON_PROMPT, MODES, WRITER_CONTRACT
from plugins.tutor.store import PreferenceStore


class FakeSettings:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> Any:
        self.values[key] = value
        return value


def make_prefs(tmp_path: Path) -> PreferenceStore:
    defaults = tmp_path / "defaults.json"
    defaults.write_text(
        json.dumps({"version": 1, "items": [{"text": "先给结论"}]}, ensure_ascii=False),
        "utf-8",
    )
    store = PreferenceStore(tmp_path / "prefs.json", defaults_path=defaults)
    store.load()
    return store


def make_modes(tmp_path: Path) -> ModeStore:
    store = ModeStore(tmp_path / "modes.json", MODES)
    store.load()
    return store


def make_plugin(tmp_path: Path) -> TutorPlugin:
    plugin = TutorPlugin()
    plugin.settings = FakeSettings()
    plugin.prefs = make_prefs(tmp_path)
    plugin.modes = make_modes(tmp_path)
    plugin.ctx = None
    return plugin


# ---------------------------------------------------------------------------
# Preference profile (propose-only contract)
# ---------------------------------------------------------------------------
def test_defaults_seed_on_first_load(tmp_path: Path) -> None:
    store = make_prefs(tmp_path)
    assert [item.text for item in store.items()] == ["先给结论"]


def test_propose_does_not_apply_until_accepted(tmp_path: Path) -> None:
    store = make_prefs(tmp_path)
    proposal = store.propose(
        [{"op": "add", "text": "不要用类比", "category": "dislike"}],
        reason="student said so",
    )
    assert proposal is not None
    assert store.items() and store.items()[0].text != "不要用类比"
    ok, _ = store.accept(proposal.id)
    assert ok
    assert "不要用类比" in [item.text for item in store.items()]


def test_reject_discards_proposal(tmp_path: Path) -> None:
    store = make_prefs(tmp_path)
    proposal = store.propose([{"op": "add", "text": "x"}])
    assert proposal is not None and store.reject(proposal.id)
    assert store.pending() == []


def test_malformed_proposal_operations_are_filtered(tmp_path: Path) -> None:
    store = make_prefs(tmp_path)
    proposal = store.propose(
        [
            {"op": "add", "text": ""},
            {"op": "explode", "text": "x"},
            {"op": "update", "id": ""},
            {"op": "add", "text": "合法的一条"},
        ]
    )
    assert proposal is not None
    assert len(proposal.operations) == 1


def test_export_and_reset_defaults(tmp_path: Path) -> None:
    store = make_prefs(tmp_path)
    store.clear()
    store.add("先给结论")
    target = store.export_defaults()
    assert json.loads(target.read_text("utf-8"))["items"][0]["text"] == "先给结论"
    store.add("第二条")
    assert store.reset_to_defaults() == 1


# ---------------------------------------------------------------------------
# Mode store
# ---------------------------------------------------------------------------
def test_modes_seeded_with_four_defaults(tmp_path: Path) -> None:
    store = make_modes(tmp_path)
    ids = [item["id"] for item in store.items()]
    assert ids == ["abstract", "interactive", "detailed", "summary"]


def test_mode_add_update_delete(tmp_path: Path) -> None:
    store = make_modes(tmp_path)
    ok, _ = store.add("复习卡片", "生成复习卡片。", "自测用")
    assert ok
    mode = next(item for item in store.items() if item["name"] == "复习卡片")
    ok, _ = store.update(mode["id"], "复习卡片", "新的提示词。")
    assert ok and store.get(mode["id"])["prompt"] == "新的提示词。"
    assert store.delete(mode["id"])
    assert store.get(mode["id"]) is None


def test_modes_persist_and_reset(tmp_path: Path) -> None:
    store = make_modes(tmp_path)
    store.update("abstract", "梗概2", "改过的梗概")
    store2 = make_modes(tmp_path)
    assert store2.get("abstract")["name"] == "梗概2"
    assert store2.reset_to_defaults() == 4
    assert store2.get("abstract")["name"] == "梗概"


# ---------------------------------------------------------------------------
# Prompt composition / plugin behaviour
# ---------------------------------------------------------------------------
def test_compose_notes_prompt_contains_common_mode_contract_and_prefs(
    tmp_path: Path,
) -> None:
    plugin = make_plugin(tmp_path)
    plugin.prefs.add("先给结论再展开")
    prompt = plugin.compose_notes_prompt("detailed", "sid")
    assert "只输出笔记或问题的回答" in prompt
    assert MODES[2]["prompt"].splitlines()[0] in prompt
    assert "输出约定" in WRITER_CONTRACT
    assert "先给结论再展开" in prompt


def test_ask_extension_exposes_propose_tool(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    extension = plugin._ask_extension()
    assert extension.roles == ("ask_instant", "ask_research")
    assert [tool.name for tool in extension.tools] == ["propose_preference"]
    assert "propose_preference" in extension.prompt


def test_propose_tool_records_pending_but_not_profile(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    result = plugin._propose_tool(
        {
            "parameters": {
                "operations": [
                    {"op": "add", "text": "不要用类比", "category": "dislike"}
                ],
                "reason": "学生原话",
            }
        }
    )
    assert "已提交" in result["content"][0]["text"]
    assert len(plugin.prefs.pending()) == 1
    assert "不要用类比" not in [item.text for item in plugin.prefs.items()]


def test_propose_tool_disabled(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    plugin.settings.values["tutor.propose_enabled"] = False
    result = plugin._propose_tool({"parameters": {"operations": [{"op": "add", "text": "x"}]}})
    assert result.get("isError")
    assert plugin.prefs.pending() == []


def test_common_prompt_has_expected_rules() -> None:
    assert "不要让学生自己去观看课程回放" in COMMON_PROMPT
    assert "只输出笔记或问题的回答" in COMMON_PROMPT


# ---------------------------------------------------------------------------
# Notes generation action / restore defaults
# ---------------------------------------------------------------------------
class FakeNotes:
    def __init__(self) -> None:
        self.started: list[str] = []

    def start(self, sid: str) -> bool:
        self.started.append(sid)
        return True


def test_generate_notes_sets_mode_prompt_and_starts(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    notes = FakeNotes()
    plugin.ctx = type(
        "Ctx",
        (),
        {
            "service": lambda self, name=None: notes,
            "data_dir": tmp_path,
            "streams": type("S", (), {"get_session": lambda self, sid: None})(),
        },
    )()
    result = plugin._generate_notes({"session_id": "sid-1", "mode": "summary"})
    assert result["ok"] and notes.started == ["sid-1"]
    prompt = plugin.settings.values["notes.mode_prompt"]
    assert "你现在需要生成一份总结" in prompt


def test_restore_defaults_restores_prompts_and_modes(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    plugin.settings.values["tutor.common_prompt"] = "被改坏的提示词"
    plugin.settings.values["tutor.ask_prompt"] = "被改坏的提问提示词"
    plugin.modes.update("abstract", "改过的梗概", "改过的提示词")
    result = plugin._prompt_restore({"scope": "all"})
    assert result["ok"]
    assert "被改坏" not in plugin.settings.values["tutor.common_prompt"]
    assert plugin.modes.get("abstract")["name"] == "梗概"


def test_pref_tool_is_in_agent_extension_snapshot(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path)
    extension = plugin._ask_extension()
    tool = extension.tools[0]
    context = {
        "parameters": {
            "operations": [{"op": "add", "text": "多给推导", "category": "depth"}],
            "reason": "学生说基础差",
        },
        "role": "ask_instant",
        "session_id": "s",
    }
    result = tool.handler(context)
    assert "已提交" in result["content"][0]["text"]
    assert len(plugin.prefs.pending()) == 1
