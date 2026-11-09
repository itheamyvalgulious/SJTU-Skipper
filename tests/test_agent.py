"""Tests for the P3 LLM Agent plugin.

Covers:
- Default four tools registration and schema
- FileWorkspace read_lines / replace_unique / append_line edge cases
- Multi-session independent history; subsession inheritance
- max_steps
- Prompt contributors and extra instructions
- UI state (agent page, actions)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.agent import (
    AgentEvent,
    AgentPlugin,
    AgentService,
    AgentSession,
    FileWorkspace,
    ToolSpec,
)
from plugins.agent.tools import _build_default_tools
from tests.support import FakeLLM, FakeResponse, FakeSettings, FakeToolCall


# ---------------------------------------------------------------------------
# Stub plugins to satisfy agent deps in tests
# ---------------------------------------------------------------------------


class StubSettingsPlugin(Plugin):
    """Provides the ``settings`` service backed by a FakeSettings."""

    id = "settings"
    requires: tuple[str, ...] = ()

    def __init__(self, settings: FakeSettings | None = None) -> None:
        self._settings = settings or FakeSettings()

    def setup(self, ctx: PluginContext) -> None:
        ctx.register_service("settings", self._settings)

    def teardown(self) -> None:
        pass


class StubLLMPlugin(Plugin):
    """Satisfies the ``llm`` plugin requirement (service is pre-injected)."""

    id = "llm"
    requires: tuple[str, ...] = ()

    def setup(self, ctx: PluginContext) -> None:
        pass  # LLM service injected via build_runtime(llm=...)

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_runtime(
    tmp_path: Path,
    *,
    fake_llm: FakeLLM | None = None,
    fake_settings: FakeSettings | None = None,
) -> tuple:
    """Build a minimal runtime with the agent plugin wired up."""
    llm = fake_llm or FakeLLM()
    settings = fake_settings or FakeSettings()
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(
        cfg,
        llm=llm,
        plugins=[StubSettingsPlugin(settings), StubLLMPlugin(), AgentPlugin()],
    )
    runtime.start()
    agent = runtime.services.require("agent")
    return runtime, agent, llm, settings


# ===================================================================
# FileWorkspace unit tests
# ===================================================================


class TestFileWorkspace:
    def test_register_and_names(self, tmp_path: Path):
        ws = FileWorkspace()
        ws.register("a.txt", tmp_path / "a.txt")
        ws.register("b.txt", tmp_path / "b.txt")
        assert ws.names == ("a.txt", "b.txt")
        assert ws.path("a.txt") == (tmp_path / "a.txt").resolve()
        assert "a.txt" in ws
        assert "c.txt" not in ws

    def test_reject_dangerous_names(self):
        ws = FileWorkspace()
        with pytest.raises(ValueError, match="empty"):
            ws.register("", "/tmp/x")
        with pytest.raises(ValueError, match="'..'"):
            ws.register("a/../b", "/tmp/x")
        with pytest.raises(ValueError, match="absolute"):
            ws.register("/etc/passwd", "/tmp/x")
        with pytest.raises(ValueError):
            ws._validate_name(None)  # noqa: SLF001

    def test_hidden_spill_files(self, tmp_path: Path):
        ws = FileWorkspace({"a.txt": tmp_path / "a.txt"})
        name = ws.spill("x" * 5000)
        assert ws.is_hidden(name)
        assert name not in ws.visible_names
        assert name in ws.names
        assert ws.text(name) == "x" * 5000

    def test_hidden_files_stay_hidden_after_copy(self, tmp_path: Path):
        ws = FileWorkspace({"a.txt": tmp_path / "a.txt"})
        name = ws.spill("payload")
        clone = ws.copy()
        assert clone.is_hidden(name)

    def test_read_lines_l_r_m1(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        ws = FileWorkspace({"test.txt": f})

        # l=-1, r=-1: all lines
        out = ws.read_lines("test.txt", -1, -1)
        assert "# test.txt (lines 1-5 of 5)" in out
        assert "line1" in out
        assert "line5" in out

        # l=2, r=4
        out = ws.read_lines("test.txt", 2, 4)
        assert "lines 2-4 of 5" in out
        assert "line1" not in out

        # l=2, r=-1
        out = ws.read_lines("test.txt", 2, -1)
        assert "lines 2-5 of 5" in out
        assert "line1" not in out
        assert "line5" in out

        # l=-1, r=2
        out = ws.read_lines("test.txt", -1, 2)
        assert "lines 1-2 of 5" in out
        assert "line1" in out
        assert "line3" not in out

    def test_read_lines_oob_truncates(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\n", encoding="utf-8")
        ws = FileWorkspace({"test.txt": f})

        out = ws.read_lines("test.txt", 1, 100)
        assert "lines 1-2 of 2" in out

        out = ws.read_lines("test.txt", 10, 20)
        # Both bounds are beyond N=2 → clipped to 2-2
        assert "lines 2-2 of 2" in out

    def test_read_lines_missing_file(self, tmp_path: Path):
        ws = FileWorkspace({"ghost.txt": tmp_path / "ghost.txt"})
        out = ws.read_lines("ghost.txt", 1, 10)
        assert "lines ?-? of 0" in out

    def test_read_lines_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        ws = FileWorkspace({"empty.txt": f})
        out = ws.read_lines("empty.txt", 1, 10)
        assert "lines ?-? of 0" in out

    def test_replace_unique_success(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("hello world\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        result = ws.replace_unique("notes.md", "hello", "hi")
        assert "已替换" in result
        assert f.read_text(encoding="utf-8") == "hi world\n"

    def test_replace_zero_occurrences(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("hello\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        with pytest.raises(ValueError, match="0 次"):
            ws.replace_unique("notes.md", "nope", "x")

    def test_replace_multiple_occurrences(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("aaa bbb\nccc aaa\naaa ddd\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        with pytest.raises(ValueError) as exc_info:
            ws.replace_unique("notes.md", "aaa", "xxx")
        msg = str(exc_info.value)
        assert "3 次" in msg
        # Line numbers: aaa appears at lines 1, 2, 3
        assert "行" in msg
        assert "1" in msg
        assert "2" in msg
        assert "3" in msg

    def test_append_line_adds_newline(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("existing", encoding="utf-8")  # no trailing newline
        ws = FileWorkspace({"notes.md": f})
        ws.append_line("notes.md", "new content")
        text = f.read_text(encoding="utf-8")
        assert text == "existing\nnew content\n"

    def test_append_line_empty_file(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        ws.append_line("notes.md", "first")
        assert f.read_text(encoding="utf-8") == "first\n"

    def test_text_method(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("hello", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        assert ws.text("notes.md") == "hello"
        assert ws.text("nonexistent") == ""

    def test_atomic_write(self, tmp_path: Path):
        f = tmp_path / "atomic.txt"
        f.write_text("original", encoding="utf-8")
        ws = FileWorkspace({"atomic.txt": f})
        ws.write("atomic.txt", "replaced")
        assert f.read_text(encoding="utf-8") == "replaced"

    def test_copy(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        ws = FileWorkspace({"a.txt": f})
        ws2 = ws.copy()
        assert ws2.names == ("a.txt",)
        assert ws2.path("a.txt") == f.resolve()
        # Independent
        ws2.register("b.txt", tmp_path / "b.txt")
        assert ws.names == ("a.txt",)


# ===================================================================
# ToolSpec and default tools
# ===================================================================


class TestToolSpec:
    def test_openai_schema(self):
        spec = ToolSpec(
            name="test",
            description="desc",
            parameters={"type": "object", "properties": {}},
            handler=lambda **kw: "ok",
        )
        schema = spec.openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"
        assert schema["function"]["description"] == "desc"

    def test_default_four_tools(self, tmp_path: Path):
        ws = FileWorkspace({"f.txt": tmp_path / "f.txt"})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        names = {t.name for t in tools}
        assert names == {"info", "read", "replace", "append"}

        # Every tool has a valid JSON Schema
        for tool in tools:
            schema = tool.openai_schema()
            assert "function" in schema
            func = schema["function"]
            assert func["name"] == tool.name
            assert "parameters" in func
            params = func["parameters"]
            assert "type" in params

    def test_info_tool(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("hello", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        sess = AgentSession(name="test-sess", workspace=ws, model="gpt-4")
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "## meta\nsome info",
        )
        info_tool = next(t for t in tools if t.name == "info")
        result = info_tool.handler()
        assert "test-sess" in result
        assert "notes.md" in result
        assert "meta" in result
        assert "gpt-4" in result

    def test_read_tool(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        read_tool = next(t for t in tools if t.name == "read")
        result = read_tool.handler(file="notes.md", l=2, r=2)
        assert "lines 2-2 of 3" in result
        assert "b" in result

    def test_read_tool_unregistered_file(self, tmp_path: Path):
        ws = FileWorkspace({"notes.md": tmp_path / "notes.md"})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        read_tool = next(t for t in tools if t.name == "read")
        result = read_tool.handler(file="secret.txt")
        assert "未注册" in result

    def test_replace_tool(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("old text\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        replace_tool = next(t for t in tools if t.name == "replace")
        result = replace_tool.handler(file="notes.md", old_str="old", new_str="new")
        assert "已替换" in result
        assert f.read_text(encoding="utf-8") == "new text\n"

    def test_replace_tool_unregistered_file(self, tmp_path: Path):
        ws = FileWorkspace({"notes.md": tmp_path / "notes.md"})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        replace_tool = next(t for t in tools if t.name == "replace")
        result = replace_tool.handler(file="secret.txt", old_str="x", new_str="y")
        assert "未注册" in result

    def test_append_tool(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("line1\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        append_tool = next(t for t in tools if t.name == "append")
        result = append_tool.handler(file="notes.md", text="line2")
        assert "已追加" in result
        assert f.read_text(encoding="utf-8") == "line1\nline2\n"

    def test_write_tool_is_opt_in(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("old\n", encoding="utf-8")
        ws = FileWorkspace({"notes.md": f})
        sess = AgentSession(workspace=ws)

        default_tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
        )
        assert "write" not in {t.name for t in default_tools}

        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
            names=("info", "read", "replace", "append", "write"),
        )
        write_tool = next(t for t in tools if t.name == "write")
        result = write_tool.handler(file="notes.md", text="# 新大纲\n\n## 第一节\n")
        assert "已整体写入" in result
        assert f.read_text(encoding="utf-8") == "# 新大纲\n\n## 第一节\n"

    def test_write_tool_unregistered_file(self, tmp_path: Path):
        ws = FileWorkspace({"notes.md": tmp_path / "notes.md"})
        sess = AgentSession(workspace=ws)
        tools = _build_default_tools(
            workspace=ws,
            session_getter=lambda: sess,
            meta_getter=lambda: "",
            names=("write",),
        )
        write_tool = next(t for t in tools if t.name == "write")
        result = write_tool.handler(file="secret.txt", text="x")
        assert "未注册" in result


# ===================================================================
# AgentSession tests
# ===================================================================


class TestAgentSession:
    def test_add_message_and_history(self):
        sess = AgentSession(name="test")
        sess.add_message("user", "hello")
        sess.add_message("assistant", "hi", tool_calls=[{"id": "c1"}])
        assert len(sess.messages) == 2
        assert sess.messages[0]["role"] == "user"
        assert sess.messages[1]["tool_calls"] == [{"id": "c1"}]

    def test_last_assistant(self):
        sess = AgentSession()
        sess.add_message("user", "q")
        assert sess.last_assistant() is None
        sess.add_message("assistant", "a")
        assert sess.last_assistant()["content"] == "a"

    def test_clear_messages(self):
        sess = AgentSession()
        sess.add_message("user", "x")
        sess.clear_messages()
        assert len(sess.messages) == 0

    def test_to_dict(self):
        sess = AgentSession(name="test", model="gpt-4")
        d = sess.to_dict()
        assert d["name"] == "test"
        assert d["model"] == "gpt-4"
        assert d["message_count"] == 0
        sess.add_message("assistant", "done")
        d2 = sess.to_dict()
        assert d2["message_count"] == 1
        assert d2["last_message"] == "done"


# ===================================================================
# AgentService integration tests
# ===================================================================


class TestAgentService:
    def test_create_session_defaults(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session("test")
            assert session.name == "test"
            assert session.max_steps == 12
            assert len(session.tools) == 4
            assert set(session.tools) == {"info", "read", "replace", "append"}
            assert session.messages == []
        finally:
            runtime.stop()

    def test_create_session_custom_tools(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            # Register a custom tool
            custom = ToolSpec(
                name="my_tool",
                description="custom",
                parameters={"type": "object", "properties": {}},
                handler=lambda **kw: "custom result",
            )
            agent.register_tool(custom)

            session = agent.create_session("test", tools=["info", "read"])
            assert set(session.tools) == {"info", "read", "my_tool"}
        finally:
            runtime.stop()

    def test_create_session_write_tool_and_output_cap(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session(
                "test",
                tools=["info", "read", "replace", "append", "write"],
                max_tool_output=24000,
            )
            assert "write" in session.tools
            assert session.max_tool_output == 24000
        finally:
            runtime.stop()

    def test_list_get_close(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            s1 = agent.create_session("a")
            s2 = agent.create_session("b")
            sessions = agent.list_sessions()
            assert len(sessions) == 2
            assert agent.get_session(s1.id) is s1
            agent.close(s1.id)
            assert agent.get_session(s1.id) is None
            assert len(agent.list_sessions()) == 1
        finally:
            runtime.stop()

    def test_reset(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session("test")
            session.add_message("user", "hi")
            assert len(session.messages) == 1
            agent.reset(session.id)
            assert len(session.messages) == 0
        finally:
            runtime.stop()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def test_run_simple_loop(self, tmp_path: Path):
        """Two-step run: tool calls → final text."""
        f = tmp_path / "notes.md"
        f.write_text("line1\nline2\n", encoding="utf-8")

        script = [
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(id="c1", name="info", arguments={}),
                    FakeToolCall(id="c2", name="read", arguments={"file": "notes.md", "l": 1, "r": 10}),
                ],
            ),
            FakeResponse(content="All done, I have read the file."),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f})
            events: list[AgentEvent] = []
            result = agent.run(session.id, "read the file", on_event=lambda e: events.append(e))

            assert result.error is None
            assert result.steps == 2
            assert result.final_message == "All done, I have read the file."
            assert len(fake_llm.calls) == 2
            # Events
            types = [e.type for e in events]
            assert "agent.run.started" in types
            assert "agent.tool_call" in types
            assert "agent.tool_result" in types
            assert "agent.message" in types
            assert "agent.run.finished" in types
        finally:
            runtime.stop()

    def test_run_with_replace_and_append(self, tmp_path: Path):
        """Four tool calls in one step, then final."""
        f = tmp_path / "notes.md"
        f.write_text("old line\nkeep\n", encoding="utf-8")

        script = [
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(id="c1", name="info", arguments={}),
                    FakeToolCall(id="c2", name="read", arguments={"file": "notes.md"}),
                    FakeToolCall(id="c3", name="append", arguments={"file": "notes.md", "text": "new line"}),
                    FakeToolCall(id="c4", name="replace", arguments={"file": "notes.md", "old_str": "old line", "new_str": "fixed line"}),
                ],
            ),
            FakeResponse(content="File updated successfully."),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f})
            result = agent.run(session.id, "fix the file")

            assert result.error is None
            assert result.steps == 2
            assert result.final_message == "File updated successfully."
            content = f.read_text(encoding="utf-8")
            assert "fixed line" in content
            assert "new line" in content
        finally:
            runtime.stop()

    def test_file_writes_refresh_ui(self, tmp_path: Path):
        """Every notes write must bump the UI version (live note refresh)."""
        f = tmp_path / "notes.md"
        f.write_text("old\n", encoding="utf-8")
        script = [
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(id="c1", name="append", arguments={"file": "notes.md", "text": "新内容"}),
                ],
            ),
            FakeResponse(content="done"),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            version_before = runtime.ui.state()["version"]
            session = agent.create_session("test", files={"notes.md": f})
            agent.run(session.id, "append")
            assert runtime.ui.state()["version"] > version_before
        finally:
            runtime.stop()

    def test_run_max_steps(self, tmp_path: Path):
        """Script returns tool_calls forever; stops at max_steps."""
        f = tmp_path / "notes.md"
        f.write_text("content", encoding="utf-8")

        # Every response has tool_calls → loop never ends without cap
        script = [
            FakeResponse(
                content="thinking...",
                tool_calls=[FakeToolCall(id="cx", name="info", arguments={})],
            )
            for _ in range(20)
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f}, max_steps=3)
            result = agent.run(session.id, "do something")

            assert result.error == "max_steps exceeded"
            assert result.steps == 3
            assert len(fake_llm.calls) == 3
        finally:
            runtime.stop()

    def test_run_tool_exception(self, tmp_path: Path):
        """Tool that raises → error string returned to LLM, not crash."""
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")

        script = [
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(id="c1", name="read", arguments={"file": "notes.md", "l": 1000, "r": 2000}),
                ],
            ),
            FakeResponse(content="Recovered from error."),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f})
            result = agent.run(session.id, "read")

            assert result.error is None
            assert result.steps == 2
            assert result.final_message == "Recovered from error."
            # The tool ran without crashing the loop
        finally:
            runtime.stop()

    def test_run_concurrent_rejected(self, tmp_path: Path):
        """Lock prevents concurrent run on same session."""
        import threading

        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [FakeResponse(content="done")]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f})
            # Acquire lock from another thread to simulate a running session
            lock_held = threading.Event()
            lock_released = threading.Event()

            def hold_lock() -> None:
                session.lock.acquire()
                lock_held.set()
                lock_released.wait()
                session.lock.release()

            holder = threading.Thread(target=hold_lock, daemon=True)
            holder.start()
            lock_held.wait(timeout=2)

            result = agent.run(session.id, "msg")
            assert result.error == "session is already running"

            lock_released.set()
            holder.join(timeout=2)
        finally:
            runtime.stop()

    # ------------------------------------------------------------------
    # Multi-session / subsession
    # ------------------------------------------------------------------
    def test_multi_session_independent_history(self, tmp_path: Path):
        script1 = [FakeResponse(content="response for s1")]
        script2 = [FakeResponse(content="response for s2")]
        f1 = tmp_path / "f1.md"
        f2 = tmp_path / "f2.md"
        f1.write_text("a", encoding="utf-8")
        f2.write_text("b", encoding="utf-8")

        # Need separate FakeLLMs per session? No, the script is consumed.
        # Let's use a single FakeLLM with enough responses
        script = [
            FakeResponse(content="response for s1"),
            FakeResponse(content="response for s2"),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            s1 = agent.create_session("s1", files={"f1.md": f1})
            s2 = agent.create_session("s2", files={"f2.md": f2})
            r1 = agent.run(s1.id, "msg1")
            r2 = agent.run(s2.id, "msg2")

            assert r1.final_message == "response for s1"
            assert r2.final_message == "response for s2"
            assert len(s1.messages) == 2  # user + assistant
            assert len(s2.messages) == 2
            # Histories are independent
            assert s1.messages[0]["content"] == "msg1"
            assert s2.messages[0]["content"] == "msg2"
        finally:
            runtime.stop()

    def test_subsession_inheritance(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [
            FakeResponse(content="parent done"),
            FakeResponse(content="child done"),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            parent = agent.create_session(
                "parent",
                files={"notes.md": f},
                model="gpt-4",
                max_steps=5,
                system_prompt="parent prompt",
                extra_instructions=["be careful"],
            )
            r1 = agent.run(parent.id, "parent msg")
            assert r1.final_message == "parent done"

            child = agent.create_subsession(parent.id, "child")
            assert child.parent_id == parent.id
            assert child.model == parent.model == "gpt-4"
            assert child.max_steps == parent.max_steps == 5
            assert child.system_prompt == parent.system_prompt
            assert child.extra_instructions == parent.extra_instructions
            assert child.workspace.names == parent.workspace.names
            # Independent history
            assert child.messages == []

            r2 = agent.run(child.id, "child msg")
            assert r2.final_message == "child done"
        finally:
            runtime.stop()

    def test_subsession_override(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        fake_llm = FakeLLM([FakeResponse(content="ok")])
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            parent = agent.create_session("parent", files={"notes.md": f})
            child = agent.create_subsession(
                parent.id,
                "child",
                model="gpt-5",
                max_steps=3,
                system_prompt="overridden",
            )
            assert child.model == "gpt-5"
            assert child.max_steps == 3
            assert child.system_prompt == "overridden"
        finally:
            runtime.stop()

    # ------------------------------------------------------------------
    # Prompt contributors / extra instructions
    # ------------------------------------------------------------------
    def test_prompt_contributor(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [FakeResponse(content="done")]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            contributor_called = []

            def my_contributor(session: AgentSession) -> str:
                contributor_called.append(session.name)
                return "EXTRA CONTEXT: time is now"

            agent.add_prompt_contributor(my_contributor)
            session = agent.create_session("test", files={"notes.md": f})
            agent.run(session.id, "msg")

            assert len(contributor_called) == 1
            assert contributor_called[0] == "test"
            # Verify the system prompt was in the LLM call
            call = fake_llm.calls[0]
            messages = call.get("messages", [])
            system_msgs = [m for m in messages if m.get("role") == "system"]
            assert len(system_msgs) > 0
            system_text = str(system_msgs[0].get("content", ""))
            assert "EXTRA CONTEXT" in system_text
        finally:
            runtime.stop()

    def test_extra_instructions(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [FakeResponse(content="done")]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session(
                "test",
                files={"notes.md": f},
                extra_instructions=["ONLY use append", "NEVER use replace"],
            )
            agent.run(session.id, "msg")

            call = fake_llm.calls[0]
            messages = call.get("messages", [])
            system_msgs = [m for m in messages if m.get("role") == "system"]
            system_text = str(system_msgs[0].get("content", ""))
            assert "ONLY use append" in system_text
            assert "NEVER use replace" in system_text
        finally:
            runtime.stop()

    def test_set_system_prompt_and_add_instruction(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [FakeResponse(content="done"), FakeResponse(content="done again")]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            session = agent.create_session("test", files={"notes.md": f})
            agent.set_system_prompt(session.id, "CUSTOM PROMPT")
            agent.add_instruction(session.id, "LATE INSTRUCTION")

            agent.run(session.id, "msg")

            call = fake_llm.calls[0]
            system_text = str(call["messages"][0]["content"])
            assert "CUSTOM PROMPT" in system_text
            assert "LATE INSTRUCTION" in system_text
        finally:
            runtime.stop()

    # ------------------------------------------------------------------
    # Custom tool registration
    # ------------------------------------------------------------------
    def test_register_unregister_tool(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            spec = ToolSpec(
                name="my_tool",
                description="desc",
                parameters={"type": "object", "properties": {}},
                handler=lambda **kw: "ok",
            )
            agent.register_tool(spec)
            session = agent.create_session("test")
            assert "my_tool" in session.tools

            agent.unregister_tool("my_tool")
            session2 = agent.create_session("test2")
            assert "my_tool" not in session2.tools
        finally:
            runtime.stop()

    # ------------------------------------------------------------------
    # Tool output truncation
    # ------------------------------------------------------------------
    def test_tool_output_truncation(self, tmp_path: Path):
        """Very large tool output is truncated to max_tool_output."""
        f = tmp_path / "big.txt"
        long_text = "x" * 500 + "\n"
        f.write_text(long_text * 20, encoding="utf-8")  # ~10k
        script = [
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(id="c1", name="read", arguments={"file": "big.txt", "l": 1, "r": -1}),
                ],
            ),
            FakeResponse(content="truncated, but done"),
        ]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            settings = runtime.services.require("settings")
            settings.set("agent.max_tool_output", 200)
            session = agent.create_session("test", files={"big.txt": f})
            result = agent.run(session.id, "read big file")
            assert result.error is None
            # The tool result should have been truncated
            calls = fake_llm.calls
            assert len(calls) >= 2
            # Second call's messages should contain the truncated tool result
            msgs2 = calls[1].get("messages", [])
            tool_msgs = [m for m in msgs2 if m.get("role") == "tool"]
            assert len(tool_msgs) == 1
            tool_content = str(tool_msgs[0].get("content", ""))
            assert "截断" in tool_content
        finally:
            runtime.stop()


# ===================================================================
# UI tests
# ===================================================================


class TestAgentUI:
    def test_ui_state_has_agent_page(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            state = runtime.ui.state()
            # The agent is a page opened from the palette / menu bar.
            right_ids = [t["id"] for t in state["right_tabs"]]
            assert "agent" not in right_ids
            page_ids = [p["id"] for p in state["pages"]]
            assert "agent:chat" in page_ids
            command = next(c for c in state["commands"] if c["id"] == "agent.open")
            assert command["page"] == "agent:chat"
        finally:
            runtime.stop()

    def test_action_new_session(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            result = runtime.ui.dispatch_action("agent.new_session", {"name": "my-session"})
            assert result.ok
            assert "my-session" in result.message
            sessions = agent.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["name"] == "my-session"
        finally:
            runtime.stop()

    def test_action_new_session_opens_page(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            result = runtime.ui.dispatch_action("agent.new_session", {})
            assert result.ok
            assert result.open_page == "agent:chat"
        finally:
            runtime.stop()

    def test_no_agent_sidebar_item(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            snapshot = runtime.ui.snapshot()
            assert "agent.sessions" not in snapshot["items"]
            assert "agent.open" in snapshot["commands"]
        finally:
            runtime.stop()

    def test_action_agent_chat_without_session(self, tmp_path: Path):
        """agent.chat should auto-create a session when none exists."""
        f = tmp_path / "notes.md"
        f.write_text("data", encoding="utf-8")
        script = [FakeResponse(content="hello from agent")]
        fake_llm = FakeLLM(script)
        runtime, agent, _, _ = _make_agent_runtime(tmp_path, fake_llm=fake_llm)
        try:
            result = runtime.ui.dispatch_action("agent.chat", {"message": "test"})
            assert result.ok
            # A session was auto-created
            sessions = agent.list_sessions()
            assert len(sessions) == 1
        finally:
            runtime.stop()

    def test_action_reset(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session("test")
            session.add_message("user", "hello")
            assert len(session.messages) == 1
            runtime.ui.dispatch_action("agent.reset", {"session_id": session.id})
            assert len(session.messages) == 0
        finally:
            runtime.stop()


# ===================================================================
# AgentPlugin settings registration
# ===================================================================


class TestAgentSettings:
    def test_settings_registered(self, tmp_path: Path):
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path)
        try:
            assert settings.get("agent.model") is not None
            assert settings.get("agent.max_steps") is not None
            assert settings.get("agent.max_tool_output") is not None
            assert settings.get("agent.system_prompt") is not None
        finally:
            runtime.stop()

    def test_settings_affect_session_defaults(self, tmp_path: Path):
        fsettings = FakeSettings()
        fsettings.values["agent.model"] = "gpt-5"
        fsettings.values["agent.max_steps"] = 8
        fsettings.values["agent.system_prompt"] = "custom default"
        runtime, agent, llm, settings = _make_agent_runtime(tmp_path, fake_settings=fsettings)
        try:
            session = agent.create_session("test")
            assert session.model == "gpt-5"
            assert session.max_steps == 8
            assert session.system_prompt == "custom default"
        finally:
            runtime.stop()

# ===================================================================
# Context trimming
# ===================================================================


class TestHistoryTrimming:
    def test_old_tool_results_are_spilled(self, tmp_path: Path) -> None:
        runtime, agent, _llm, _settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session(
                "trim",
                files={"notes.md": tmp_path / "notes.md"},
                max_history_chars=100,
            )
            session.add_message("user", "go")
            session.add_message("tool", "A" * 80, tool_call_id="t1")
            session.add_message("tool", "B" * 80, tool_call_id="t2")
            session.add_message("tool", "C" * 80, tool_call_id="t3")
            payload = agent._payload_messages(session)
            tool_messages = [m for m in payload if m["role"] == "tool"]
            assert tool_messages[-1]["content"] == "C" * 80
            assert tool_messages[0]["content"] != "A" * 80
            assert "隐藏文件" in tool_messages[0]["content"]
            # Spilled content is readable again through the workspace.
            name = session.messages[1]["spilled_as"]
            assert agent.get_session(session.id).workspace.text(name) == "A" * 80
            # Repeated calls reuse the same spill file.
            again = agent._payload_messages(session)
            assert again[1]["content"] == tool_messages[0]["content"]
        finally:
            runtime.stop()

    def test_without_budget_history_is_untouched(self, tmp_path: Path) -> None:
        runtime, agent, _llm, _settings = _make_agent_runtime(tmp_path)
        try:
            session = agent.create_session("plain", files={"a.md": tmp_path / "a.md"})
            session.add_message("tool", "x" * 5000, tool_call_id="t1")
            payload = agent._payload_messages(session)
            assert payload[0]["content"] == "x" * 5000
        finally:
            runtime.stop()
