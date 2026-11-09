"""Tests for the (non-instant) whole-lecture summarize plugin.

The handout is produced by one agent run that reads the material and writes
the notes itself; these tests fake the agent service.
"""

from __future__ import annotations

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
    def __init__(self, *, block: bool = False, fail: bool = False) -> None:
        self.block = block
        self.fail = fail
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
        for filename, path in session.files.items():
            if filename == SOURCE_FILENAME:
                continue
            target = Path(path)
            text = target.read_text("utf-8") if target.exists() else ""
            if "Fake summary" not in text:
                target.write_text(
                    text + "\n## 章节（运行）\n\nFake summary line.\n", "utf-8"
                )
        return SimpleNamespace(content="done", steps=3, error=None)

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
    assert set(session_call["tools"]) == {"info", "read", "replace", "append"}
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
        "summarize.auto_on_stop": True,
    }
    for key, value in expected.items():
        assert settings.get(key) == value
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


def test_running_task_is_exposed_and_button_disabled(tmp_path: Path) -> None:
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
        button = next(
            b
            for b in item["blocks"][-1]["buttons"]
            if b.get("action") == "summarize.full"
        )
        assert button.get("disabled") is True
        assert button["label"] == "生成中…"

        agent.release()
        assert _wait_for(lambda: not runtime.services.get("summarize").busy)
    finally:
        runtime.stop()
