"""End-to-end integration: settings -> agent -> summarize -> note files."""

from __future__ import annotations

import time
from pathlib import Path

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.agent import AgentPlugin
from plugins.instant_summarize.plugin import InstantSummarizePlugin
from plugins.reader import ReaderPlugin
from plugins.settings.plugin import SettingsPlugin
from plugins.summarize.plugin import SummarizePlugin

from support import FakeLLM, FakeResponse, FakeToolCall


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
    script = [
        FakeResponse(
            tool_calls=[
                FakeToolCall(
                    "c1",
                    "append",
                    {"file": "instant_note_knowledge.md", "text": "- 留数定理" * 5},
                ),
                FakeToolCall(
                    "c2",
                    "append",
                    {"file": "instant_note_gossip.md", "text": "- 老师聊到 AI" * 3},
                ),
                FakeToolCall(
                    "c3",
                    "append",
                    {"file": "instant_note_notice.md", "text": "- 作业 3.1" * 3},
                ),
            ]
        ),
        FakeResponse(content="done"),
    ]
    llm = FakeLLM(script)
    runtime = build_runtime(
        cfg,
        plugins=[SettingsPlugin(), StubLLM(), AgentPlugin(), ReaderPlugin(), InstantSummarizePlugin()],
        llm=llm,
    )
    runtime.start()
    runtime.services.require("settings").set("instant-summarize.trigger_chars", 100)
    runtime.services.require("settings").set("instant-summarize.trigger_audio_seconds", 1)
    try:
        session = runtime.hub.start_session(meta={"course": "集成测试"})
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
        assert runtime.bus.flush(2.0)

        notes = tmp_path / "sessions" / session.session_id / "notes"
        assert _wait_until(
            lambda: (notes / "instant_note_knowledge.md").exists()
            and "留数定理"
            in (notes / "instant_note_knowledge.md").read_text(encoding="utf-8")
        ), "instant_note_knowledge.md was not written"
        assert "AI" in (notes / "instant_note_gossip.md").read_text(encoding="utf-8")
        assert "作业" in (notes / "instant_note_notice.md").read_text(encoding="utf-8")
        assert llm.calls, "summarize never called the LLM"
    finally:
        runtime.stop()


def test_whole_lecture_summarize_rewrites_outline(tmp_path: Path) -> None:
    """The non-instant plugin lets the agent write the three notes."""
    cfg = load_config(data_dir=tmp_path)
    outline = (
        "# 课堂讲义：知识\n\n"
        "## 留数定理\n\n"
        "### 定义\n\n"
        "设 $f$ 在 $D$ 内除有限个孤立奇点外解析……\n"
    )
    script = [
        FakeResponse(
            tool_calls=[
                FakeToolCall(
                    "w1",
                    "append",
                    {"file": "note_knowledge.md", "text": outline},
                ),
                FakeToolCall(
                    "w2",
                    "append",
                    {"file": "note_notice.md", "text": "# 通知\n\n## 作业\n\n- 习题 3.1\n"},
                ),
            ]
        ),
        FakeResponse(content="done"),
    ]
    llm = FakeLLM(script)
    runtime = build_runtime(
        cfg,
        plugins=[SettingsPlugin(), StubLLM(), AgentPlugin(), ReaderPlugin(), SummarizePlugin()],
        llm=llm,
    )
    runtime.start()
    runtime.services.require("settings").set("instant-summarize.trigger_chars", 100)
    runtime.services.require("settings").set("instant-summarize.trigger_audio_seconds", 1)
    try:
        session = runtime.hub.start_session(meta={"course": "集成测试"})
        session.transcript.append(
            "留数定理：设 f 在 D 内除有限个孤立奇点外解析……" * 30,
            start=0.0,
            end=60.0,
            source="stt",
        )
        assert runtime.bus.flush(1.0)

        result = runtime.ui.dispatch_action("summarize.full", {})
        assert result.ok

        notes = tmp_path / "sessions" / session.session_id / "notes"
        knowledge = notes / "note_knowledge.md"
        assert _wait_until(
            lambda: knowledge.exists()
            and "### 定义" in knowledge.read_text(encoding="utf-8")
        ), "note_knowledge.md outline was not written"

        content = knowledge.read_text(encoding="utf-8")
        assert "## 留数定理" in content
        assert "[00:" not in content, "notes must not be timestamped logs"
        assert "习题 3.1" in (notes / "note_notice.md").read_text(encoding="utf-8")
        assert llm.calls, "summarize never called the LLM"
    finally:
        runtime.stop()

