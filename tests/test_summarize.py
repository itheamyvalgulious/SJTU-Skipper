"""Tests for the (non-instant) whole-lecture summarize plugin.

The handout is produced by one agent run that reads the material and writes
the notes itself; these tests fake the agent service.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.reader import ReaderPlugin
from plugins.summarize import SummarizePlugin
from plugins.summarize.plugin import NOTE_FILES, NOTE_HEADERS, SOURCE_FILENAME


class FakeAgentService:
    def __init__(
        self,
        *,
        block: bool = False,
        fail: bool = False,
        review_verdicts: list[str] | None = None,
    ) -> None:
        self.block = block
        self.fail = fail
        self.review_verdicts = list(review_verdicts or [])
        self._review_index = 0
        self._writer_runs = 0
        self.sessions: dict[str, SimpleNamespace] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []
        self._gate = threading.Event()
        if not block:
            self._gate.set()

    def release(self) -> None:
        self._gate.set()

    def create_session(self, name: str, **kwargs: Any) -> SimpleNamespace:
        sid = f"agent-{len(self.sessions) + 1}"
        session = SimpleNamespace(id=sid, name=name, files=dict(kwargs.get("files") or {}))
        self.sessions[sid] = session
        self.create_calls.append({"name": name, **kwargs})
        return session

    def run(self, session_id: str, user_input: str, on_event: Any = None) -> Any:
        self._gate.wait(timeout=10.0)
        self.run_calls.append({"session_id": session_id, "user_input": user_input})
        if self.fail:
            raise RuntimeError("agent failed")
        session = self.sessions[session_id]
        if session.name.startswith("review-"):
            verdict = "PASS"
            if self.review_verdicts:
                index = min(self._review_index, len(self.review_verdicts) - 1)
                verdict = self.review_verdicts[index]
                self._review_index += 1
            if verdict == "PASS":
                return SimpleNamespace(
                    final_message="REVIEW: PASS", steps=2, error=None
                )
            return SimpleNamespace(
                final_message=(
                    "## 审阅意见\n\n"
                    "1. [note_knowledge.md] 数学表述有误 — 应改写 — 请修正。\n\n"
                    "REVIEW: FAIL"
                ),
                steps=2,
                error=None,
            )
        self._writer_runs += 1
        for filename, path in session.files.items():
            if filename == SOURCE_FILENAME:
                continue
            target = Path(path)
            text = target.read_text("utf-8") if target.exists() else ""
            target.write_text(
                text
                + f"\n## 章节（运行 {self._writer_runs}）\n\nFake summary line.\n",
                "utf-8",
            )
        return SimpleNamespace(final_message="done", steps=3, error=None)

    def wait_calls(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while len(self.run_calls) < count:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True


class _StubSettingsPlugin(Plugin):
    id = "settings"

    def setup(self, ctx: PluginContext) -> None:
        from tests.support import FakeSettings

        self._service = FakeSettings()
        ctx.register_service("settings", self._service)

    def teardown(self) -> None:
        pass


class _StubAgentPlugin(Plugin):
    id = "llm-agent"
    requires = ("settings",)

    def __init__(self, agent: FakeAgentService | None = None) -> None:
        super().__init__()
        self._agent = agent or FakeAgentService()

    def setup(self, ctx: PluginContext) -> None:
        ctx.register_service("agent", self._agent)

    def teardown(self) -> None:
        pass


def _build(tmp_path: Path, agent: FakeAgentService) -> Any:
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(
        cfg,
        plugins=[
            _StubSettingsPlugin(),
            _StubAgentPlugin(agent),
            ReaderPlugin(),
            SummarizePlugin(),
        ],
    )
    runtime.start()
    return runtime


@pytest.fixture()
def harness(tmp_path: Path):
    agent = FakeAgentService()
    runtime = _build(tmp_path, agent)
    settings = runtime.services.get("settings")
    yield runtime, settings, agent
    runtime.stop()


def _start_session(runtime: Any) -> str:
    session = runtime.hub.start_session(meta={"course_name": "数学分析"})
    runtime.bus.flush(2.0)
    return session.session_id


def _wait_for(predicate: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_session_start_creates_note_files(harness: tuple) -> None:
    runtime, _settings, _agent = harness
    sid = _start_session(runtime)
    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    for name in NOTE_FILES:
        path = notes_dir / name
        assert path.exists()
        assert NOTE_HEADERS[name].strip() in path.read_text("utf-8")


def test_manual_action_runs_one_agent(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    runtime.hub.current.transcript.append("第一段：柯西积分公式。" * 40)
    runtime.hub.current.transcript.append("最后一段：留数定理。" * 40)
    runtime.bus.flush(1.0)

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert agent.wait_calls(1)

    session_call = agent.create_calls[0]
    assert [name for name, _ in session_call["tools"]] == ["info", "read", "replace", "append"]
    assert session_call["max_steps"] == 0, "summarize must not cap the step count"
    assert session_call["max_tool_output"] == 200000
    assert session_call["max_history_chars"] == 0
    assert session_call["max_tokens"] == 32000
    assert session_call["llm_extra"] == {"reasoning_effort": "max"}

    prompt = agent.run_calls[0]["user_input"]
    assert SOURCE_FILENAME in prompt
    system_prompt = session_call["system_prompt"]
    assert "read(file" in system_prompt and "info()" in system_prompt
    assert "200,000" in system_prompt

    material = (runtime.data_dir / "sessions" / sid / SOURCE_FILENAME).read_text("utf-8")
    assert "柯西积分公式" in material and "留数定理" in material

    notes = runtime.data_dir / "sessions" / sid / "notes"
    assert "Fake summary" in (notes / "note_knowledge.md").read_text("utf-8")


def test_material_is_merged_from_the_two_plugin_logs(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    session_dir = runtime.data_dir / "sessions" / sid

    # The stt / vlm plugins own these two files; a handout must be rebuilt
    # from them even when the in-memory stream is missing or truncated.
    (session_dir / "transcript.jsonl").write_text(
        json.dumps(
            {"start": 0.0, "end": 4.0, "text": "落盘的讲授内容。"},
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    (session_dir / "visual.jsonl").write_text(
        json.dumps(
            {"start": 10.0, "end": 16.0, "text": "落盘的板书内容。"},
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    runtime.hub.current.transcript.append("内存里的旧内容。" * 5)
    runtime.bus.flush(1.0)

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert agent.wait_calls(1)

    material = (session_dir / SOURCE_FILENAME).read_text("utf-8")
    assert "落盘的讲授内容。" in material
    assert "落盘的板书内容。" in material
    assert "内存里的旧内容" not in material, "persisted logs must win over the live stream"
    assert material.index("落盘的讲授内容。") < material.index("落盘的板书内容。")


def test_empty_action(harness: tuple) -> None:
    runtime, _settings, agent = harness
    _start_session(runtime)
    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert "暂无" in result.message
    assert agent.run_calls == []


def test_auto_on_stop(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    runtime.hub.current.transcript.append("自动总结内容。" * 20)
    runtime.bus.flush(1.0)

    runtime.hub.end_session("test")
    runtime.bus.flush(2.0)
    assert agent.wait_calls(1)

    notes = runtime.data_dir / "sessions" / sid / "notes"
    assert _wait_for(
        lambda: "Fake summary" in (notes / "note_knowledge.md").read_text("utf-8")
    )


def test_busy_rejects_second_run(tmp_path: Path) -> None:
    agent = FakeAgentService(block=True)
    runtime = _build(tmp_path, agent)
    try:
        _start_session(runtime)
        runtime.hub.current.transcript.append("并发测试。" * 20)
        runtime.bus.flush(1.0)

        first = runtime.ui.dispatch_action("summarize.full", {})
        assert first.ok
        second = runtime.ui.dispatch_action("summarize.full", {})
        assert second.ok
        assert "运行中" in second.message

        agent.release()
        assert agent.wait_calls(1)
    finally:
        runtime.stop()


def test_failure_emits_error(tmp_path: Path) -> None:
    agent = FakeAgentService(fail=True)
    runtime = _build(tmp_path, agent)
    try:
        _start_session(runtime)
        runtime.hub.current.transcript.append("失败测试。" * 20)
        runtime.bus.flush(1.0)

        errors: list[dict] = []
        runtime.bus.on("summarize.error", lambda event: errors.append(dict(event.payload)))

        runtime.ui.dispatch_action("summarize.full", {})
        plugin = runtime.services.get("summarize")
        assert _wait_for(lambda: not plugin.busy)
        runtime.bus.flush(1.0)
        assert errors
    finally:
        runtime.stop()


def test_settings_and_service(harness: tuple) -> None:
    runtime, settings, _agent = harness
    expected = {
        "summarize.max_tool_output": 200000,
        "summarize.max_history_chars": 0,
        "summarize.max_tokens": 32000,
        "summarize.model": "",
        "summarize.notes_dir": "notes",
        "summarize.retry_max": 1,
        "summarize.on_stop": "direct",
        "summarize.review_enabled": True,
        "summarize.review_model": "",
        "summarize.review_max_steps": 0,
        "summarize.review_max_tokens": 32000,
        "summarize.review_max_rounds": 0,
    }
    for key, value in expected.items():
        assert settings.get(key) == value
    from plugins.summarize.prompts import REVIEWER_SYSTEM_PROMPT

    assert settings.get("summarize.reviewer_prompt") == REVIEWER_SYSTEM_PROMPT
    assert runtime.services.get("summarize") is not None


def test_ui_state(harness: tuple) -> None:
    runtime, _settings, _agent = harness
    _start_session(runtime)
    snap = runtime.ui.snapshot()
    assert snap["items"]["summarize.notes"] == "note"
    assert "summarize.full" in snap["actions"]
    sid = runtime.hub.current.session_id
    result = runtime.ui.dispatch_action("summarize.read", {"doc": "knowledge"})
    assert result.ok
    assert result.open_page == f"reader:notes:{sid}"


def test_running_task_is_exposed_and_button_hidden(tmp_path: Path) -> None:
    agent = FakeAgentService(block=True)
    runtime = _build(tmp_path, agent)
    try:
        sid = _start_session(runtime)
        runtime.hub.current.transcript.append("任务展示测试。" * 20)
        runtime.bus.flush(1.0)

        result = runtime.ui.dispatch_action("summarize.full", {})
        assert result.ok
        plugin = runtime.services.get("summarize")
        assert _wait_for(lambda: plugin.busy)

        state = runtime.ui.state()
        tasks = [task for task in state.get("tasks", []) if task.get("id") == "summarize"]
        assert tasks, "running summarize task must be listed"
        assert "生成讲义" in tasks[0]["label"]

        item = next(
            item
            for tab in state["right_tabs"]
            for item in tab["toolbox"]
            if item["id"] == "summarize.notes"
        )
        assert not any(
            button.get("action") == "summarize.full"
            for block in item["blocks"]
            if block["type"] == "buttons"
            for button in block.get("buttons", [])
        ), "no generate button may show while the handout is still being written"
        assert "正在生成讲义" in item["blocks"][0]["text"]

        agent.release()
        assert _wait_for(lambda: not runtime.services.get("summarize").busy)
    finally:
        runtime.stop()


def test_generate_button_only_after_a_handout_exists(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    notes_dir = runtime.data_dir / "sessions" / sid / "notes"

    def action_buttons() -> list[dict[str, Any]]:
        item = next(
            item
            for tab in runtime.ui.state()["right_tabs"]
            for item in tab["toolbox"]
            if item["id"] == "summarize.notes"
        )
        return [
            button
            for block in item["blocks"]
            if block["type"] == "buttons"
            for button in block.get("buttons", [])
            if button.get("action") == "summarize.full"
        ]

    # New session: only headers on disk → no manual generation button.
    assert action_buttons() == []

    knowledge = notes_dir / "note_knowledge.md"
    knowledge.write_text(
        knowledge.read_text("utf-8") + "## 第一节\n\n已生成的讲义正文。\n",
        "utf-8",
    )
    buttons = action_buttons()
    assert [button["label"] for button in buttons] == ["重新生成讲义"]
    assert buttons[0].get("disabled") is not True
    assert agent.run_calls == []


def test_review_loop_sends_feedback_back_to_writer(tmp_path: Path) -> None:
    agent = FakeAgentService(review_verdicts=["FAIL", "PASS"])
    runtime = _build(tmp_path, agent)
    try:
        sid = _start_session(runtime)
        runtime.hub.current.transcript.append("审阅循环测试。" * 20)
        runtime.bus.flush(1.0)

        result = runtime.ui.dispatch_action("summarize.full", {})
        assert result.ok
        plugin = runtime.services.get("summarize")
        assert _wait_for(lambda: not plugin.busy, timeout=10.0)

        # writer -> review(FAIL) -> writer revision -> review(PASS)
        assert len(agent.run_calls) == 4
        assert "审阅者提出以下问题" in agent.run_calls[2]["user_input"]
        assert "REVIEW: FAIL" not in agent.run_calls[2]["user_input"]

        notes = runtime.data_dir / "sessions" / sid / "notes"
        text = (notes / "note_knowledge.md").read_text("utf-8")
        assert "运行 1" in text and "运行 2" in text

        reviewer_calls = [
            call
            for call in agent.create_calls
            if str(call["name"]).startswith("review-")
        ]
        assert len(reviewer_calls) == 1, "reviewer must live in one persistent context"
        assert [name for name, _ in reviewer_calls[0]["tools"]] == ["info", "read"]
    finally:
        runtime.stop()


def test_review_disabled_runs_writer_only(tmp_path: Path) -> None:
    agent = FakeAgentService()
    runtime = _build(tmp_path, agent)
    try:
        settings = runtime.services.get("settings")
        settings.set("summarize.review_enabled", False)
        _start_session(runtime)
        runtime.hub.current.transcript.append("关闭审阅测试。" * 20)
        runtime.bus.flush(1.0)

        result = runtime.ui.dispatch_action("summarize.full", {})
        assert result.ok
        plugin = runtime.services.get("summarize")
        assert _wait_for(lambda: not plugin.busy, timeout=10.0)

        assert len(agent.run_calls) == 1
        assert all(
            not str(call["name"]).startswith("review-")
            for call in agent.create_calls
        )
    finally:
        runtime.stop()


def test_parse_review_marker() -> None:
    from plugins.summarize.plugin import _parse_review

    verdict, body = _parse_review("## 审阅意见\n\n1. 问题\n\nREVIEW: FAIL")
    assert verdict == "FAIL"
    assert "1. 问题" in body and "REVIEW" not in body

    verdict, body = _parse_review("REVIEW: PASS")
    assert (verdict, body) == ("PASS", "")

    verdict, body = _parse_review("审阅结论：通过")
    assert verdict == "PASS"

    verdict, body = _parse_review("没有给出结论")
    assert verdict == "FAIL"
    assert body == "没有给出结论"


def test_second_run_asks_writer_to_merge_existing_notes(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    runtime.hub.current.transcript.append("合并测试。" * 20)
    runtime.bus.flush(1.0)

    plugin = runtime.services.get("summarize")
    runtime.ui.dispatch_action("summarize.full", {})
    assert _wait_for(lambda: not plugin.busy, timeout=10.0)
    assert "已有笔记" not in agent.run_calls[0]["user_input"]

    runtime.ui.dispatch_action("summarize.full", {})
    assert _wait_for(lambda: not plugin.busy, timeout=10.0)

    # writer -> reviewer -> writer (second job) -> reviewer
    second_writer_prompt = agent.run_calls[2]["user_input"]
    assert "已有笔记" in second_writer_prompt
    assert "最新记录" in second_writer_prompt




# ---------------------------------------------------------------------------
# Live instant notes as a draft
# ---------------------------------------------------------------------------


def test_instant_notes_are_passed_as_a_draft(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)
    runtime.hub.current.transcript.append("课堂上讲了导数。" * 20)
    runtime.bus.flush(1.0)

    notes = runtime.data_dir / "sessions" / sid / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "instant_note_knowledge.md").write_text(
        "# 即时知识\n\n导数就是切线斜率（待核实）。\n", "utf-8"
    )

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert agent.wait_calls(1)

    files = agent.create_calls[0]["files"]
    assert "instant_draft.md" in files
    draft = Path(files["instant_draft.md"]).read_text("utf-8")
    assert "导数就是切线斜率" in draft

    prompt = agent.run_calls[0]["user_input"]
    assert "即时笔记底稿" in prompt
    assert "必须删除" in prompt


# ---------------------------------------------------------------------------
# On-stop modes
# ---------------------------------------------------------------------------


def test_on_stop_skip_leaves_the_lecture_alone(harness: tuple) -> None:
    runtime, settings, agent = harness
    settings.set("summarize.on_stop", "skip")
    sid = _start_session(runtime)
    runtime.hub.current.transcript.append("这段不该被总结。" * 20)
    runtime.bus.flush(1.0)

    runtime.hub.end_session("test")
    runtime.bus.flush(2.0)

    assert agent.run_calls == []
    notes = runtime.data_dir / "sessions" / sid / "notes" / "note_knowledge.md"
    assert "Fake summary" not in notes.read_text("utf-8")


def test_on_stop_fill_falls_back_when_no_recording(harness: tuple) -> None:
    runtime, settings, agent = harness
    settings.set("summarize.on_stop", "fill")
    _start_session(runtime)
    runtime.hub.current.transcript.append("没有录播时也要出讲义。" * 20)
    runtime.bus.flush(1.0)

    runtime.hub.end_session("test")
    runtime.bus.flush(2.0)

    # No input service in this harness → the fill pass is unavailable and the
    # plugin summarizes what it already has.
    assert agent.wait_calls(1)

    state = runtime.ui.state()
    blocks = [
        item
        for tab in state["right_tabs"]
        for item in tab["toolbox"]
        if item["id"] == "summarize.notes"
    ][0]["blocks"]
    assert any("未获取到本节课录播" in str(b.get("text", "")) for b in blocks)
    assert any(
        button.get("action") == "summarize.fill_again"
        for block in blocks
        if block["type"] == "buttons"
        for button in block.get("buttons", [])
    )


def test_set_mode_action_validates(harness: tuple) -> None:
    runtime, settings, _agent = harness

    ok = runtime.ui.dispatch_action("summarize.set_mode", {"mode": "fill"})
    assert ok.ok
    assert settings.get("summarize.on_stop") == "fill"

    bad = runtime.ui.dispatch_action("summarize.set_mode", {"mode": "nonsense"})
    assert not bad.ok
    assert settings.get("summarize.on_stop") == "fill"


def test_material_merges_gap_fill_sessions(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)

    sessions = runtime.data_dir / "sessions"
    import json as _json

    # Live pass covered the first minute; the fill pass covers minute two.
    live = sessions / sid
    (live / "transcript.jsonl").write_text(
        _json.dumps({"start": 0.0, "end": 60.0, "text": "前半段"}) + "\n", "utf-8"
    )
    (live / "meta.json").write_text(
        _json.dumps({"started_at": 1000.0, "meta": {}}), "utf-8"
    )

    fill_id = "fillsession1"
    fill = sessions / fill_id
    fill.mkdir(parents=True)
    (fill / "transcript.jsonl").write_text(
        _json.dumps({"start": 120.0, "end": 180.0, "text": "后半段"}) + "\n", "utf-8"
    )
    (fill / "meta.json").write_text(
        _json.dumps(
            {
                "started_at": 5000.0,
                "meta": {"fill_for": sid, "media_begin": 940.0},
            }
        ),
        "utf-8",
    )

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert agent.wait_calls(1)
    prompt = agent.run_calls[0]["user_input"]
    source = (live / "lecture_source.md").read_text("utf-8")
    assert "前半段" in source and "后半段" in source
    assert source.index("前半段") < source.index("后半段")
