"""End-to-end integration: settings -> agent -> summarize -> note files."""

from __future__ import annotations

import time
from pathlib import Path

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.agent import AgentPlugin
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


def test_summarize_writes_three_files(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    script = [
        FakeResponse(
            tool_calls=[
                FakeToolCall("c1", "append", {"file": "knowledge.md", "text": "- 留数定理" * 5}),
                FakeToolCall("c2", "append", {"file": "gossip.md", "text": "- 老师聊到 AI" * 3}),
                FakeToolCall("c3", "append", {"file": "notice.md", "text": "- 作业 3.1" * 3}),
            ]
        ),
        FakeResponse(content="done"),
    ]
    llm = FakeLLM(script)
    runtime = build_runtime(
        cfg,
        plugins=[SettingsPlugin(), StubLLM(), AgentPlugin(), SummarizePlugin()],
        llm=llm,
    )
    runtime.start()
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
            lambda: (notes / "knowledge.md").exists()
            and "留数定理" in (notes / "knowledge.md").read_text(encoding="utf-8")
        ), "knowledge.md was not written"
        assert "AI" in (notes / "gossip.md").read_text(encoding="utf-8")
        assert "作业" in (notes / "notice.md").read_text(encoding="utf-8")
        assert llm.calls, "summarize never called the LLM"
    finally:
        runtime.stop()
