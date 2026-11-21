from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import load_config
from core.agent_types import (
    AgentEvent,
    AgentEventType,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    ExecutionRole,
    ExecutionStatus,
    RuntimeFrameType,
)
from core.agent_types import IpcMethod
from core.navigation import Navigation, navigation_scope
from core.note_document import AskIntensity, QaStatus
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from core.ui import BlockType
from plugins.agent import AgentPlugin
from plugins.ask.plugin import AskPlugin, _Live
from plugins.reader.plugin import ReaderPlugin
from plugins.settings.plugin import SettingsPlugin

from tests.test_agent import ScriptedRuntime


class StubLLM(Plugin):
    id = "llm"
    requires = ("settings",)

    def setup(self, ctx: PluginContext) -> None:
        pass


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _build(tmp_path: Path, script) -> tuple:
    cfg = load_config(data_dir=tmp_path)
    llm = ScriptedRuntime(script)
    runtime = build_runtime(
        cfg,
        plugins=[
            SettingsPlugin(),
            StubLLM(),
            AgentPlugin(),
            ReaderPlugin(),
            AskPlugin(),
        ],
        llm=llm,
    )
    runtime.start()
    return runtime, llm


def _open_note(runtime, tmp_path: Path) -> Path:
    notes = tmp_path / "sessions" / "ask-session" / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    note = notes / "note_knowledge.md"
    note.write_text("# 笔记\n\n第一段：这是定义。\n\n第二段：这是例题。\n", "utf-8")
    runtime.services.require("reader").show(
        title="测试笔记",
        docs=[{"key": "knowledge", "label": "知识", "path": str(note)}],
        page_key="ask-session",
        owner="test",
    )
    return note


def test_instant_ask_answers_and_persists_fold(tmp_path: Path) -> None:
    snapshots = []

    def script(sdk, params):
        yield {
            "type": RuntimeFrameType.EVENT,
            "name": AgentEventType.EXECUTION,
            "data": {
                "metrics": {"model": "test-model", "output_tokens": 3},
                "block": {
                    "id": "1:0",
                    "kind": ExecutionBlockKind.TEXT,
                    "status": ExecutionBlockStatus.COMPLETED,
                    "text": "Answer",
                },
            },
        }
        assert runtime.bus.flush()
        qa_id = events[0]["id"]
        snapshots.append(runtime.services.require("ask").activity(str(note), qa_id))
        page_id = "reader:notes:ask-session"
        with navigation_scope(
            Navigation(page=page_id, pages=[page_id], doc="knowledge")
        ):
            pages = runtime.ui.state()["page_content"]

        def activities(blocks):
            for block in blocks:
                if block.get("type") == BlockType.QA:
                    yield block.get("activity")
                for tab in block.get("tabs", []):
                    yield from activities(tab.get("blocks", []))

        assert snapshots[-1] in list(activities(next(iter(pages.values()))))
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {"final_message": "Answer", "steps": 1},
        }

    runtime, llm = _build(tmp_path, script)
    events: list[dict] = []
    runtime.bus.on("ui.notify", lambda event: events.append(dict(event.get("payload"))))
    try:
        note = _open_note(runtime, tmp_path)
        result = runtime.ui.dispatch_action(
            "ask.ask",
            {
                "path": str(note),
                "selection": "第一段：这是定义。",
                "question": "这一段在说什么？",
                "intensity": AskIntensity.INSTANT,
            },
        )
        assert result.ok, result.message
        assert _wait_until(lambda: '"status": "done"' in note.read_text("utf-8")), (
            note.read_text("utf-8")
        )

        text = note.read_text("utf-8")
        assert "Answer" in text
        assert '"intensity": "instant"' in text
        assert '"collapsed": true' in text
        assert text.index("这一段在说什么？") < text.index("第二段：这是例题。")

        declaration = next(
            params for method, params in llm.calls if method == IpcMethod.CREATE_SESSION
        )
        assert declaration["role"] == "ask_instant"
        assert declaration["tools"] == []
        prompt = llm.runs[0]["instruction"]
        assert "第一段：这是定义。" in prompt
        assert "第二段：这是例题。" in prompt

        assert _wait_until(
            lambda: any(item.get("status") == QaStatus.DONE for item in events)
        )
        final = next(item for item in events if item.get("status") == QaStatus.DONE)
        assert final["collapsed"] is True
        assert result.data["qa_id"] == final["id"]
        assert "answer_html" in final
        activity = final["activity"]
        assert activity["id"] == f"qa-{final['id']}"
        assert activity["agents"][0]["status"] == ExecutionStatus.COMPLETED
        assert activity["agents"][0]["blocks"][0]["text"] == "Answer"
        assert events[0]["activity"]["agents"][0]["blocks"] == []
        assert snapshots[0]["agents"][0]["model"] == "test-model"
        assert snapshots[0]["agents"][0]["role"] == ExecutionRole.QUESTION_ANSWER
        assert _wait_until(
            lambda: (
                runtime.services.require("ask").activity(str(note), final["id"]) is None
            )
        )
    finally:
        runtime.stop()


def test_research_ask_reads_complete_snapshot(tmp_path: Path) -> None:
    snapshots = []

    def script(sdk, params):
        declaration = sdk.sessions[params["session_id"]]
        snapshots.append(
            {
                name: Path(path).read_text()
                for name, path in declaration["files"].items()
            }
        )
        assert declaration["role"] == "ask_research"
        assert declaration["tools"] == ["read", "grep", "glob"]
        assert set(declaration["readonly"]) == set(declaration["files"])
        yield {
            "type": RuntimeFrameType.EVENT,
            "name": AgentEventType.TOOL_CALL,
            "data": {"name": "grep", "arguments": {"pattern": "condition"}},
        }
        yield {
            "type": RuntimeFrameType.EVENT,
            "name": AgentEventType.TOOL_RESULT,
            "data": {"name": "grep", "result": "condition"},
        }
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {
                "final_message": "The source includes every condition.",
                "steps": 2,
            },
        }

    runtime, sdk = _build(tmp_path, script)
    try:
        note = _open_note(runtime, tmp_path)
        source = note.parent.parent / "lecture_source.md"
        source.write_text("All conditions\n" * 1000)
        result = runtime.ui.dispatch_action(
            "ask.ask",
            {
                "path": str(note),
                "selection": "第一段：这是定义。",
                "question": "What are the conditions?",
                "intensity": AskIntensity.RESEARCH,
            },
        )
        assert result.ok
        assert _wait_until(lambda: '"status": "done"' in note.read_text())
        assert "every condition" in note.read_text()
        assert snapshots[0]["lecture_source.md"] == "All conditions\n" * 1000
        assert "note_knowledge.md" in snapshots[0]
        # Conversations stay open so the next ask can reuse the prefix.
        assert sdk.sessions
    finally:
        runtime.stop()


def test_live_accumulates_execution() -> None:
    plugin = SimpleNamespace(ctx=SimpleNamespace(ui=Mock(), logger=Mock()))
    with patch("plugins.ask.plugin.time.monotonic", return_value=1.0):
        live = _Live(plugin, "/note.md", "question")
        initial = plugin.ctx.ui.notify.call_args.args[1]
        for data in (
            {"metrics": {"model": "model", "context_tokens": 123}},
            {
                "block": {
                    "id": "thought",
                    "kind": ExecutionBlockKind.THINKING,
                    "status": ExecutionBlockStatus.STREAMING,
                    "text": "Consider",
                }
            },
            {
                "block": {
                    "id": "tool",
                    "kind": ExecutionBlockKind.TOOL,
                    "status": ExecutionBlockStatus.RUNNING,
                    "name": "read",
                    "arguments": {"path": "note.md"},
                }
            },
            {
                "block": {
                    "id": "thought",
                    "kind": ExecutionBlockKind.THINKING,
                    "status": ExecutionBlockStatus.COMPLETED,
                    "text": "Consider all conditions",
                }
            },
            {
                "metrics": {"tool_seconds": 2},
                "block": {
                    "id": "tool",
                    "kind": ExecutionBlockKind.TOOL,
                    "status": ExecutionBlockStatus.COMPLETED,
                    "name": "read",
                    "result": "source",
                },
            },
            {
                "block": {
                    "id": "answer",
                    "kind": ExecutionBlockKind.TEXT,
                    "status": ExecutionBlockStatus.STREAMING,
                    "text": "Answer",
                }
            },
        ):
            live.on_event(AgentEvent(AgentEventType.EXECUTION, "session", data=data))
        live.on_event(
            AgentEvent(AgentEventType.REASONING, "session", data={"text": "duplicate"})
        )
        assert plugin.ctx.ui.notify.call_count == 1
        live.publish(force=True)
        pending = plugin.ctx.ui.notify.call_args.args[1]
        agent = pending["activity"]["agents"][0]
        assert agent["context_tokens"] == 123
        assert agent["tool_seconds"] == 2
        assert [block["id"] for block in agent["blocks"]] == [
            "thought",
            "tool",
            "answer",
        ]
        assert agent["blocks"][0]["text"] == "Consider all conditions"
        assert agent["blocks"][1]["result"] == "source"
        live.finish(status=QaStatus.ERROR, answer="Failure", elapsed=2)
        final = plugin.ctx.ui.notify.call_args.args[1]
    assert initial["activity"]["agents"][0]["blocks"] == []
    assert agent["status"] == ExecutionStatus.RUNNING
    assert agent["blocks"][-1]["status"] == ExecutionBlockStatus.STREAMING
    assert final["activity"]["agents"][0]["status"] == ExecutionStatus.FAILED
    assert (
        final["activity"]["agents"][0]["blocks"][-1]["status"]
        == ExecutionBlockStatus.FAILED
    )
    assert final["answer"] == "Failure"
    assert final["collapsed"] is False


def test_live_hides_channel_without_agent_action() -> None:
    ui = Mock()
    ui.has_action.return_value = False
    plugin = SimpleNamespace(ctx=SimpleNamespace(ui=ui, logger=Mock()))
    live = _Live(plugin, "/note.md", "question")
    activity = ui.notify.call_args.args[1]["activity"]
    assert activity["send_action"] == ""
    assert activity["send_values"] == {}


def test_live_binds_exact_session():
    plugin = SimpleNamespace(ctx=SimpleNamespace(ui=Mock(), logger=Mock()))
    live = _Live(plugin, "/note.md", "question")
    live.bind_session("native-qa-session")
    activity = plugin.ctx.ui.notify.call_args.args[1]["activity"]
    assert activity["send_action"] == "agent.message"
    assert activity["send_values"] == {}
    assert activity["agents"][0]["session_id"] == "native-qa-session"


def test_completed_qa_message_does_not_reopen_or_finish_twice(tmp_path):
    runtime, sdk = _build(tmp_path, None)
    try:
        note = _open_note(runtime, tmp_path)
        reader = runtime.services.require("reader")
        finish = type(reader).qa_finish
        with patch.object(
            type(reader), "qa_finish", autospec=True, side_effect=finish
        ) as finished:
            result = runtime.ui.dispatch_action(
                "ask.ask", {"path": str(note), "question": "Question"}
            )
            assert result.ok
            assert _wait_until(
                lambda: (
                    finished.call_count == 1
                    and '"status": "done"' in note.read_text()
                )
            )
            before = note.read_text()
            target = next(
                params["id"]
                for method, params in sdk.calls
                if method == IpcMethod.CREATE_SESSION
            )
            rejected = runtime.ui.dispatch_action(
                "agent.message", {"agent_session_id": target, "message": "Continue"}
            )
            assert not rejected.ok
            assert note.read_text() == before
            assert finished.call_count == 1
            assert len(sdk.runs) == 1
            assert not any(
                method == IpcMethod.REOPEN_SESSION for method, _ in sdk.calls
            )
    finally:
        runtime.stop()


def test_live_qa_messages_keep_one_owning_run(tmp_path):
    import threading

    started = threading.Event()
    release = threading.Event()

    def script(sdk, params):
        started.set()
        assert release.wait(5)
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {"final_message": "Updated answer"},
        }

    runtime, sdk = _build(tmp_path, script)
    original = sdk.call
    sent = []

    def call(method, **params):
        if method == IpcMethod.MESSAGE:
            sent.append(params)
            release.set()
            return {"accepted": True}
        return original(method, **params)

    sdk.call = call
    try:
        note = _open_note(runtime, tmp_path)
        assert runtime.ui.dispatch_action(
            "ask.ask", {"path": str(note), "question": "Question"}
        ).ok
        assert started.wait(3)
        session_id = next(iter(sdk.sessions))
        result = runtime.ui.dispatch_action(
            "agent.message",
            {"agent_session_id": session_id, "message": "Use this\ncondition"},
        )
        assert result.ok
        assert _wait_until(lambda: '"status": "done"' in note.read_text())
        assert sent == [{"session_id": session_id, "message": "Use this\ncondition"}]
        assert len(sdk.runs) == 1
        assert note.read_text().count("Updated answer") == 1
    finally:
        release.set()
        runtime.stop()


def test_course_context_uses_note_target(tmp_path, navigation):
    from core.streams import StreamHub

    hub = StreamHub()
    hub.start_session("processing", meta={"course_name": "Background course"})
    plugin = AskPlugin()
    plugin.ctx = SimpleNamespace(streams=hub, data_dir=tmp_path)
    path = str(tmp_path / "sessions" / "unprocessed" / "notes" / "note.md")
    assert plugin._course_name(path) == "未知课程"
    navigation.class_id = "selected"
    navigation.class_meta = {"course_name": "Selected course"}
    assert plugin._course_name(path) == "未知课程"
    target = str(tmp_path / "sessions" / "processing" / "notes" / "note.md")
    assert plugin._course_name(target) == "Background course"


def test_selection_character_limit():
    from plugins.ask.plugin import _truncate_chars

    text = "English 中文𠀀 $x^2$!"
    assert _truncate_chars(text, len(text)) == text
    assert _truncate_chars(text, 11) == text[:11] + "\n……（选中内容过长，已截断）"
    assert _truncate_chars(text, 0) == ""


def test_second_ask_reuses_conversation_and_skips_note_snapshot(tmp_path: Path) -> None:
    runs: list[str] = []

    def script(sdk, params):
        runs.append(str(params.get("instruction", "")))
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {"final_message": "Answer", "steps": 1},
        }

    runtime, llm = _build(tmp_path, script)
    try:
        note = _open_note(runtime, tmp_path)
        for index, question in enumerate(("Question one", "Question two"), start=1):
            result = runtime.ui.dispatch_action(
                "ask.ask", {"path": str(note), "question": question}
            )
            assert result.ok
            assert _wait_until(lambda index=index: len(llm.runs) == index)
        creates = [c for c in llm.calls if c[0] == IpcMethod.CREATE_SESSION]
        assert len(creates) == 1
        assert len(llm.runs) == 2
        assert "## 笔记" in runs[0]
        assert "## 笔记" not in runs[1]
        assert "Question two" in runs[1]
    finally:
        runtime.stop()


def test_preference_change_does_not_invalidate_conversation(tmp_path: Path) -> None:
    runs: list[str] = []

    def script(sdk, params):
        runs.append(str(params.get("instruction", "")))
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {"final_message": "Answer", "steps": 1},
        }

    runtime, llm = _build(tmp_path, script)
    try:
        settings = runtime.services.require("settings")
        settings.set("tutor.preferences_text", "## 偏好\n\n- 先给结论")
        note = _open_note(runtime, tmp_path)
        result = runtime.ui.dispatch_action(
            "ask.ask", {"path": str(note), "question": "Q1"}
        )
        assert result.ok
        assert _wait_until(lambda: len(llm.runs) == 1)
        settings.set("tutor.preferences_text", "## 偏好\n\n- 多给推导")
        result = runtime.ui.dispatch_action(
            "ask.ask", {"path": str(note), "question": "Q2"}
        )
        assert result.ok
        assert _wait_until(lambda: len(llm.runs) == 2)
        creates = [c for c in llm.calls if c[0] == IpcMethod.CREATE_SESSION]
        assert len(creates) == 1
        assert "偏好更新" in runs[1]
        assert "多给推导" in runs[1]
    finally:
        runtime.stop()
