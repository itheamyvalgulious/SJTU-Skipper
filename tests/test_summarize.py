"""Tests for the (non-instant) whole-lecture summarize plugin."""

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
from plugins.summarize.plugin import (
    NOTE_FILES,
    NOTE_HEADERS,
    _chunk_blocks,
    _collect_blocks,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeAgentService:
    """Scriptable fake agent service.

    ``noop_runs`` makes the first N runs succeed without writing anything,
    ``fail_runs`` makes the first N runs raise, ``block`` makes every run wait
    for :meth:`release` so concurrency guards can be tested.
    """

    def __init__(
        self,
        *,
        noop_runs: int = 0,
        fail_runs: int = 0,
        block: bool = False,
    ) -> None:
        self._noop_runs = noop_runs
        self._fail_runs = fail_runs
        self._run_count = 0
        self._next_id = 0
        self._gate = threading.Event()
        self._gate.set()
        if block:
            self._gate.clear()
        self.sessions: dict[str, SimpleNamespace] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []

    def release(self) -> None:
        self._gate.set()

    # -- agent service contract -------------------------------------------

    def create_session(
        self,
        name: str,
        system_prompt: str,
        files: dict[str, str],
        tools: list[str],
        model: str | None = None,
        max_steps: int = 8,
        max_tool_output: int | None = None,
        extra_instructions: list[str] | None = None,
        parent_id: str | None = None,
    ) -> SimpleNamespace:
        self._next_id += 1
        sid = f"agent-{self._next_id}"
        session = SimpleNamespace(id=sid, name=name, files=dict(files))
        self.sessions[sid] = session
        self.create_calls.append(
            dict(
                name=name,
                tools=list(tools),
                model=model,
                max_steps=max_steps,
                max_tool_output=max_tool_output,
            )
        )
        return session

    def run(
        self,
        session_id: str,
        user_input: str,
        on_event: Any = None,
    ) -> SimpleNamespace:
        self._gate.wait(timeout=5.0)
        self._run_count += 1
        try:
            if self._run_count <= self._fail_runs:
                raise RuntimeError(f"fake agent failure #{self._run_count}")

            if self._run_count <= self._fail_runs + self._noop_runs:
                return SimpleNamespace(content="done (noop)", finish_reason="stop")

            session = self.sessions.get(session_id)
            if session is not None:
                for filename, path in sorted(session.files.items()):
                    p = Path(path)
                    existing = p.read_text("utf-8") if p.exists() else ""
                    p.write_text(
                        existing
                        + f"\n## 章节（第 {self._run_count} 次运行）\n\n"
                        + f"Fake summary line for {filename}.\n"
                    )

            return SimpleNamespace(
                content=f"done run #{self._run_count}", finish_reason="stop"
            )
        finally:
            self.run_calls.append(
                dict(session_id=session_id, user_input=user_input)
            )

    # -- test helpers -----------------------------------------------------

    def wait_for_runs(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while len(self.run_calls) < count:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True


class _StubSettingsPlugin(Plugin):
    """In-process settings plugin stub."""

    id = "settings"

    def setup(self, ctx: PluginContext) -> None:
        from tests.support import FakeSettings

        self._service = FakeSettings()
        ctx.register_service("settings", self._service)

    def teardown(self) -> None:
        pass


class _StubAgentPlugin(Plugin):
    """In-process agent plugin stub."""

    id = "llm-agent"
    requires = ("settings",)

    def __init__(self, agent: FakeAgentService | None = None) -> None:
        super().__init__()
        self._agent = agent or FakeAgentService()

    def setup(self, ctx: PluginContext) -> None:
        ctx.register_service("agent", self._agent)

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


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


@pytest.fixture
def harness(tmp_path: Path) -> tuple[Any, Any, FakeAgentService]:
    agent = FakeAgentService()
    runtime = _build(tmp_path, agent)
    settings = runtime.services.get("settings")
    assert settings is not None
    yield runtime, settings, agent
    runtime.stop()


def _start_session(runtime: Any, meta: dict[str, Any] | None = None) -> str:
    ss = runtime.hub.start_session(meta=meta)
    runtime.bus.flush(2.0)
    return ss.session_id


def _wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# 1. Files are created on session start
# ---------------------------------------------------------------------------


def test_session_start_creates_note_files(harness: tuple) -> None:
    runtime, _settings, _agent = harness
    sid = _start_session(runtime)

    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    for fname in NOTE_FILES:
        path = notes_dir / fname
        assert path.exists(), f"{fname} should exist after session start"
        assert NOTE_HEADERS[fname].strip() in path.read_text("utf-8")
        # The instant plugin's files must not collide with these.
        assert fname.startswith("note_")


# ---------------------------------------------------------------------------
# 2. Manual action processes the *whole* lecture, not only the newest tail
# ---------------------------------------------------------------------------


def test_manual_action_uses_whole_lecture(harness: tuple) -> None:
    runtime, _settings, agent = harness
    sid = _start_session(runtime)

    streams = runtime.hub.current
    streams.transcript.append("第一段：柯西积分公式。" * 10)
    streams.transcript.append("最后一段：留数定理的应用。" * 10)
    runtime.bus.flush(1.0)

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert "讲义" in result.message
    assert agent.wait_for_runs(1)

    prompt = agent.run_calls[0]["user_input"]
    assert "柯西积分公式" in prompt, "whole lecture must be summarized, not just the tail"
    assert "留数定理" in prompt
    assert "第 1 / 1 批" in prompt

    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    assert _wait_for(
        lambda: all(
            "Fake summary" in (notes_dir / f).read_text("utf-8") for f in NOTE_FILES
        )
    )


# ---------------------------------------------------------------------------
# 3. Long lectures are processed in sequential batches against one outline
# ---------------------------------------------------------------------------


def test_long_lecture_is_batched(harness: tuple) -> None:
    runtime, settings, agent = harness
    settings.set("summarize.chunk_chars", 500)
    sid = _start_session(runtime)

    streams = runtime.hub.current
    for index in range(6):
        streams.transcript.append(f"第{index}段内容。" * 40)
    runtime.bus.flush(1.0)

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert agent.wait_for_runs(1), "the handout job never started"
    plugin = runtime.services.get("summarize")
    assert _wait_for(lambda: not plugin.busy, timeout=5.0), "job did not finish"

    parts = len(agent.run_calls)
    assert parts >= 3, f"expected >=3 batches, got {parts}"
    assert all(f"第 {i + 1} / {parts} 批" in call["user_input"] for i, call in enumerate(agent.run_calls))
    assert "最后一批的额外要求" in agent.run_calls[-1]["user_input"]
    assert "最后一批的额外要求" not in agent.run_calls[0]["user_input"]

    # Every batch writes into the same three files (one coherent handout).
    note_paths = set()
    for session in agent.sessions.values():
        note_paths.add(tuple(sorted(session.files.items())))
    assert len(note_paths) == 1, "all batches must share the same note files"


def test_chunk_blocks_offsets() -> None:
    blocks = ["a" * 600, "b" * 600, "c" * 600]
    chunks = _chunk_blocks(blocks, 1000)
    assert len(chunks) == 3
    assert [start for _blocks, start, _end in chunks] == [0, 602, 1204]
    assert [end for _blocks, _start, end in chunks][-1] == 1806


# ---------------------------------------------------------------------------
# 4. Auto run when the session stops
# ---------------------------------------------------------------------------


def test_auto_on_stop(harness: tuple) -> None:
    runtime, _settings, agent = harness
    _start_session(runtime)

    runtime.hub.current.transcript.append("课后自动整理。" * 20)
    runtime.bus.flush(1.0)

    runtime.hub.end_session("test")
    runtime.bus.flush(1.0)

    assert agent.wait_for_runs(1), "session stop should trigger the handout"


def test_auto_on_stop_disabled(harness: tuple) -> None:
    runtime, settings, agent = harness
    settings.set("summarize.auto_on_stop", False)
    _start_session(runtime)

    runtime.hub.current.transcript.append("不应自动整理。" * 20)
    runtime.bus.flush(1.0)
    runtime.hub.end_session("test")
    runtime.bus.flush(1.0)
    time.sleep(0.3)

    assert len(agent.run_calls) == 0


# ---------------------------------------------------------------------------
# 5. Empty lecture
# ---------------------------------------------------------------------------


def test_empty_action(harness: tuple) -> None:
    runtime, _settings, agent = harness
    _start_session(runtime)

    result = runtime.ui.dispatch_action("summarize.full", {})
    assert result.ok
    assert "暂无" in result.message
    assert len(agent.run_calls) == 0


# ---------------------------------------------------------------------------
# 6. Concurrency guard
# ---------------------------------------------------------------------------


def test_busy_rejects_second_run(tmp_path: Path) -> None:
    agent = FakeAgentService(block=True)
    runtime = _build(tmp_path, agent)
    try:
        _start_session(runtime)
        runtime.hub.current.transcript.append("并发测试。" * 20)
        runtime.bus.flush(1.0)

        first = runtime.ui.dispatch_action("summarize.full", {})
        assert first.ok
        assert "开始" in first.message

        second = runtime.ui.dispatch_action("summarize.full", {})
        assert second.ok
        assert "运行中" in second.message

        agent.release()
        assert agent.wait_for_runs(1)
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# 7. Failure surfaces as an error event
# ---------------------------------------------------------------------------


def test_failure_emits_error(tmp_path: Path) -> None:
    agent = FakeAgentService(fail_runs=100)
    runtime = _build(tmp_path, agent)
    errors: list[Any] = []
    try:
        runtime.bus.on("summarize.error", errors.append)
        _start_session(runtime)
        runtime.hub.current.transcript.append("失败路径。" * 20)
        runtime.bus.flush(1.0)

        assert runtime.ui.dispatch_action("summarize.full", {}).ok
        assert agent.wait_for_runs(1)
        assert _wait_for(lambda: bool(errors), timeout=3.0), "summarize.error not emitted"
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# 8. Prompts: structured outline, no timestamps, no compression
# ---------------------------------------------------------------------------


def test_prompts_require_structure_and_keep_detail(harness: tuple) -> None:
    runtime, _settings, agent = harness
    _start_session(runtime)
    runtime.hub.current.transcript.append("提示词检查。" * 50)
    runtime.bus.flush(1.0)

    runtime.ui.dispatch_action("summarize.full", {})
    assert agent.wait_for_runs(1)

    session_prompt = agent.create_calls[0]
    assert "write" in session_prompt["tools"]
    assert session_prompt["max_tool_output"] == 24000

    prompt = agent.run_calls[0]["user_input"]
    assert "禁止写时间戳" in prompt or "不写时间戳" in prompt
    assert "大纲" in prompt
    assert "压缩" in prompt
    # Input blocks must not be timestamped logs.
    assert not any(
        line.startswith("[00:") for line in prompt.splitlines()
    ), "stream timestamps leaked into the prompt"


def test_system_prompt_rewrite_rules(harness: tuple) -> None:
    _runtime, settings, _agent = harness
    system_prompt = settings.get("summarize.system_prompt")
    assert "write" in system_prompt
    assert "大纲" in system_prompt
    assert "时间戳" in system_prompt
    assert "禁止压缩" in system_prompt


# ---------------------------------------------------------------------------
# 9. UI surface
# ---------------------------------------------------------------------------


def test_ui_state(harness: tuple) -> None:
    runtime, _settings, _agent = harness
    _start_session(runtime)

    snap = runtime.ui.snapshot()
    assert snap["tabs"]["note"] == "right"
    assert snap["items"]["summarize.notes"] == "note"
    assert "summarize.full" in snap["actions"]
    assert "summarize.read" in snap["actions"]
    assert "reader:main" in snap["pages"]
    # Notes are read through the shared reader page, not a plugin-owned page.
    assert "summarize:notes" not in snap["pages"]

    item = next(
        item
        for tab in runtime.ui.state()["right_tabs"]
        for item in tab["toolbox"]
        if item["id"] == "summarize.notes"
    )
    assert item["blocks"], "note toolbox item rendered empty blocks"

    result = runtime.ui.dispatch_action("summarize.read", {"doc": "knowledge"})
    assert result.ok
    assert result.open_page == "reader:main"
    assert runtime.ui.state()["active_page"] == "reader:main"
    reader_blocks = runtime.ui.state()["page_content"]["reader:main"]
    tabs = next(block for block in reader_blocks if block["type"] == "tabs")
    assert [tab["title"] for tab in tabs["tabs"]] == ["知识", "闲话与分享", "通知"]


# ---------------------------------------------------------------------------
# 10. Settings + service registration
# ---------------------------------------------------------------------------


def test_settings_and_service(harness: tuple) -> None:
    runtime, settings, _agent = harness

    expected = {
        "summarize.max_steps": 16,
        "summarize.max_tool_output": 24000,
        "summarize.chunk_chars": 6000,
        "summarize.model": "",
        "summarize.notes_dir": "notes",
        "summarize.retry_max": 1,
        "summarize.auto_on_stop": True,
    }
    for key, value in expected.items():
        assert settings.get(key) == value, f"setting {key} wrong: {settings.get(key)!r}"

    assert "summarize" in runtime.services


# ---------------------------------------------------------------------------
# 11. Block ordering
# ---------------------------------------------------------------------------


def test_collect_blocks_merges_chronologically() -> None:
    transcript = SimpleNamespace(
        items_since=lambda _n: [
            SimpleNamespace(text="讲授一", start=0.0, source="stt"),
            SimpleNamespace(text="讲授二", start=20.0, source="stt"),
        ]
    )
    visual = SimpleNamespace(
        items_since=lambda _n: [
            SimpleNamespace(text="板书一", start=10.0, source="vlm"),
        ]
    )
    streams = SimpleNamespace(transcript=transcript, visual=visual)

    blocks = _collect_blocks(streams)
    assert blocks == ["【讲授】讲授一", "【板书】板书一", "【讲授】讲授二"]


def test_collect_blocks_without_start_times() -> None:
    transcript = SimpleNamespace(
        items_since=lambda _n: [
            SimpleNamespace(text="讲授一", start=None, source="stt"),
        ]
    )
    visual = SimpleNamespace(
        items_since=lambda _n: [
            SimpleNamespace(text="板书一", start=None, source="vlm"),
        ]
    )
    streams = SimpleNamespace(transcript=transcript, visual=visual)

    assert _collect_blocks(streams) == ["【讲授】讲授一", "【板书】板书一"]
