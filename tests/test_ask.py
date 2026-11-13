"""Ask plugin: intensities, tool parameters, streaming and fold persistence."""

from __future__ import annotations

import time
from pathlib import Path

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.agent import AgentPlugin
from plugins.agent.tools import ToolContext, build_tool
from plugins.agent.workspace import FileWorkspace
from plugins.ask.plugin import AskPlugin
from plugins.reader.plugin import ReaderPlugin
from plugins.settings.plugin import SettingsPlugin

from support import FakeLLM, FakeResponse, FakeToolCall


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
    llm = FakeLLM(script)
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
    note.write_text(
        "# 讲义\n\n第一段：这是定义。\n\n第二段：这是例题。\n", "utf-8"
    )
    runtime.services.require("reader").show(
        title="测试讲义",
        docs=[{"key": "knowledge", "label": "知识", "path": str(note)}],
        page_key="ask-session",
        owner="test",
    )
    return note


def test_instant_ask_answers_and_persists_fold(tmp_path: Path) -> None:
    runtime, llm = _build(tmp_path, [FakeResponse(content="**答**：第一段是定义。")])
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
                "intensity": "instant",
            },
        )
        assert result.ok, result.message
        assert _wait_until(
            lambda: '"status": "done"' in note.read_text("utf-8")
        ), note.read_text("utf-8")

        text = note.read_text("utf-8")
        assert "第一段是定义" in text
        assert '"intensity": "instant"' in text
        assert '"collapsed": true' in text
        assert text.index("这一段在说什么？") < text.index("第二段：这是例题。")

        # Instant mode is exactly one call with no tools.
        assert len(llm.calls) == 1
        assert llm.calls[0]["tools"] in (None, [])
        # The prompt carries both sides of the selection.
        prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0]["messages"])
        assert "第一段：这是定义。" in prompt
        assert "第二段：这是例题。" in prompt

        assert _wait_until(
            lambda: any(item.get("status") == "done" for item in events)
        )
        final = next(item for item in events if item.get("status") == "done")
        assert final["collapsed"] is True
        assert "answer_html" in final
        # Streaming text deltas reached the UI before the final payload.
        assert any(item.get("text_delta") for item in events)
    finally:
        runtime.stop()


def test_research_ask_declares_read_and_search(tmp_path: Path) -> None:
    script = [
        FakeResponse(
            content="先找一下",
            tool_calls=[FakeToolCall("c1", "search", {"keywords": ["例题"]})],
        ),
        FakeResponse(content="**答**：第二段是例题。"),
    ]
    runtime, llm = _build(tmp_path, script)
    try:
        note = _open_note(runtime, tmp_path)
        result = runtime.ui.dispatch_action(
            "ask.ask",
            {
                "path": str(note),
                "selection": "第一段：这是定义。",
                "question": "例题在哪一段？",
                "intensity": "research",
            },
        )
        assert result.ok, result.message
        assert _wait_until(
            lambda: '"status": "done"' in note.read_text("utf-8")
        ), note.read_text("utf-8")
        assert "第二段是例题" in note.read_text("utf-8")

        first_names = [tool["function"]["name"] for tool in llm.calls[0]["tools"]]
        assert first_names == ["read", "search"]
        read_schema = next(
            tool for tool in llm.calls[0]["tools"]
            if tool["function"]["name"] == "read"
        )
        assert "50 行" in read_schema["function"]["description"]
        assert "先 `search` 再 `read`" in llm.calls[0]["messages"][0]["content"]

        # The search result (line number + line text) reached the model.
        second_prompt = "\n".join(
            str(m.get("content", "")) for m in llm.calls[1]["messages"]
        )
        assert "例题" in second_prompt
    finally:
        runtime.stop()


def test_read_tool_honours_max_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("\n".join(f"第 {i} 行" for i in range(1, 21)), "utf-8")
    context = ToolContext(
        workspace=FileWorkspace({"a.md": path}),
        session_getter=lambda: None,
        meta_getter=lambda: "",
    )
    tool = build_tool("read", context, {"max_lines": 3})
    out = tool.handler(file="a.md", l=1, r=-1)
    assert "已截断" in out
    assert "第 4 行" not in out


def test_search_tool_returns_line_content_pairs(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("alpha\nbeta\ngamma\n", "utf-8")
    context = ToolContext(
        workspace=FileWorkspace({"a.md": path}),
        session_getter=lambda: None,
        meta_getter=lambda: "",
    )
    tool = build_tool("search", context, {"max_hits": 10})
    out = tool.handler(keywords=["beta", "gamma"])
    assert "(2, beta)" in out
    assert "(3, gamma)" in out
