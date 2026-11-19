from __future__ import annotations

import time
from pathlib import Path

from config import load_config
from core.agent_types import AgentRole, IpcMethod
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.agent import AgentPlugin
from plugins.ask.plugin import AskPlugin
from plugins.instant_summarize.plugin import InstantSummarizePlugin
from plugins.reader import ReaderPlugin
from plugins.settings.plugin import SettingsPlugin
from plugins.summarize.plugin import SummarizePlugin
from plugins.notes_pipeline import NotesPipelinePlugin

from tests.test_agent import ScriptedRuntime


def _notes_llm(contents, role):
    def script(sdk, params):
        declaration = sdk.sessions[params["session_id"]]
        assert declaration["role"] == role
        assert set(declaration["tools"]) == {"read", "grep", "glob", "edit", "write"}
        for name, content in contents.items():
            assert name in declaration["files"]
            yield {
                "type": "event",
                "name": "agent.file",
                "data": {"file": name, "content": content},
            }
        yield {"type": "result", "result": {"final_message": "Saved", "steps": 2}}

    return ScriptedRuntime(script)


class StubLLM(Plugin):
    """Provide the plugin id required by llm-agent; service is injected."""

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


def test_instant_summarize_writes_three_files(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    llm = _notes_llm(
        {
            "note_knowledge.md": "# 知识\n\n" + "- 留数定理\n" * 5,
            "note_gossip.md": "# 闲话\n\n" + "- 老师聊到 AI\n" * 3,
            "note_notice.md": "# 通知\n\n" + "- 作业 3.1\n" * 3,
        },
        AgentRole.NOTE_CHUNK,
    )
    runtime = build_runtime(
        cfg,
        plugins=[
            SettingsPlugin(),
            StubLLM(),
            AgentPlugin(),
            ReaderPlugin(),
            NotesPipelinePlugin(),
            InstantSummarizePlugin(),
        ],
        llm=llm,
    )
    runtime.start()
    runtime.services.require("settings").set("notes.chunk_seconds", 60)
    try:
        session = runtime.hub.start_session(
            "instant-notes", meta={"course": "集成测试"}
        )
        session.transcript.append(
            "今天我们讲留数定理。" * 80,
            start=0.0,
            end=60.0,
            source="stt",
        )
        session.visual.append(
            "板书：$\\int_C f(z)dz = 2\\pi i \\sum Res$。" * 20,
            start=0.0,
            end=60.0,
            source="vlm",
        )
        runtime.hub.end_session()
        assert runtime.bus.flush(2.0)

        notes = tmp_path / "sessions" / session.session_id / "notes"
        assert _wait_until(
            lambda: (
                (notes / "instant_note_knowledge.md").exists()
                and "留数定理"
                in (notes / "instant_note_knowledge.md").read_text(encoding="utf-8")
            )
        ), "instant_note_knowledge.md was not written"
        assert "AI" in (notes / "instant_note_gossip.md").read_text(encoding="utf-8")
        assert "作业" in (notes / "instant_note_notice.md").read_text(encoding="utf-8")
        assert llm.runs, "instant notes never started an SDK run"
    finally:
        runtime.stop()


def test_whole_lecture_summarize_rewrites_outline(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    outline = (
        "# 笔记：知识\n\n"
        "## 留数定理\n\n"
        "### 定义\n\n"
        "设 $f$ 在 $D$ 内除有限个孤立奇点外解析……\n"
    )
    llm = _notes_llm(
        {
            "note_knowledge.md": outline,
            "note_gossip.md": "# 闲话与分享\n\n",
            "note_notice.md": "# 通知\n\n## 作业\n\n- 习题 3.1\n",
        },
        AgentRole.NOTE_WRITER,
    )
    runtime = build_runtime(
        cfg,
        plugins=[
            SettingsPlugin(),
            StubLLM(),
            AgentPlugin(),
            ReaderPlugin(),
            NotesPipelinePlugin(),
            SummarizePlugin(),
        ],
        llm=llm,
    )
    runtime.start()
    runtime.services.require("settings").set("instant-summarize.enabled", False)
    runtime.services.require("settings").set("summarize.review_enabled", False)
    runtime.services.require("settings").set("summarize.on_stop", "skip")
    try:
        session = runtime.hub.start_session(
            "whole-lecture", meta={"course": "集成测试"}
        )
        session.transcript.append(
            "留数定理：设 f 在 D 内除有限个孤立奇点外解析……" * 30,
            start=0.0,
            end=60.0,
            source="stt",
        )
        runtime.hub.end_session()
        assert runtime.bus.flush(1.0)

        result = runtime.ui.dispatch_action("summarize.generate", {})
        assert result.ok

        notes = tmp_path / "sessions" / session.session_id / "notes"
        knowledge = notes / "note_knowledge.md"
        assert _wait_until(
            lambda: (
                runtime.services.require("notes-pipeline").status(session.session_id)[
                    "phase"
                ]
                == "published"
            )
        ), "the whole lecture was not published"

        content = knowledge.read_text(encoding="utf-8")
        assert "## 留数定理" in content
        assert "[00:" not in content, "notes must not be timestamped logs"
        assert "习题 3.1" in (notes / "note_notice.md").read_text(encoding="utf-8")
        assert llm.runs, "whole lecture never started an SDK run"
    finally:
        runtime.stop()


def test_selection_question_runs_a_read_only_agent_and_inserts_the_answer(
    tmp_path: Path,
) -> None:

    cfg = load_config(data_dir=tmp_path)

    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.delta",
            "data": {"text": "**答**：第一段是在说定义．"},
        }
        yield {
            "type": "result",
            "result": {"final_message": "**答**：第一段是在说定义．", "steps": 1},
        }

    llm = ScriptedRuntime(script)
    runtime = build_runtime(
        cfg,
        plugins=[
            SettingsPlugin(),
            StubLLM(),
            AgentPlugin(),
            ReaderPlugin(),
            NotesPipelinePlugin(),
            AskPlugin(),
            SummarizePlugin(),
        ],
        llm=llm,
    )
    runtime.start()
    try:
        session = runtime.hub.start_session(
            "instant-asking", meta={"course": "集成测试"}
        )
        assert runtime.bus.flush(1.0)
        sid = session.session_id
        knowledge = tmp_path / "sessions" / sid / "notes" / "note_knowledge.md"
        knowledge.parent.mkdir(parents=True, exist_ok=True)
        knowledge.write_text(
            "## 第一节\n\n这是第一段笔记内容。\n\n第二段笔记内容。\n", encoding="utf-8"
        )
        open_result = runtime.ui.dispatch_action("summarize.read", {"doc": "knowledge"})
        assert open_result.ok

        result = runtime.ui.dispatch_action(
            "ask.ask",
            {
                "path": str(knowledge),
                "selection": "这是第一段笔记内容。",
                "question": "这一段的定义是什么？",
                "intensity": "instant",
            },
        )
        assert result.ok, result.message
        assert _wait_until(
            lambda: "第一段是在说定义" in knowledge.read_text(encoding="utf-8")
        ), "the answer was never written into the note"

        content = knowledge.read_text(encoding="utf-8")
        assert content.index("第一段是在说定义") < content.index("第二段笔记内容。")
        assert "这一段的定义是什么？" in content

        declaration = next(
            params for method, params in llm.calls if method == IpcMethod.CREATE_SESSION
        )
        prompt = llm.runs[-1]["instruction"]
        assert "这是第一段笔记内容。" in prompt
        assert "第二段笔记内容。" in prompt
        assert "这一段的定义是什么？" in prompt
        assert declaration["role"] == AgentRole.ASK_INSTANT
        assert declaration["tools"] == [], "instant asking must not expose tools"
    finally:
        runtime.stop()
