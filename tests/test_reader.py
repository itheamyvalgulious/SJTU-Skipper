"""Tests for the shared reader plugin."""

from __future__ import annotations

from pathlib import Path

from config import load_config
from core.runtime import build_runtime
from plugins.reader import ReaderPlugin
from plugins.reader.plugin import split_markdown


def _make_runtime(tmp_path: Path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[ReaderPlugin()])
    runtime.start()
    return runtime


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_show_opens_reader_page(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        docs = [
            {"key": "knowledge", "label": "知识", "path": _write(tmp_path, "k.md", "# 知识\n\n## 定义\n\n$a^2$\n")},
            {"key": "gossip", "label": "闲话", "path": _write(tmp_path, "g.md", "# 闲话\n\n聊天。\n")},
        ]
        reader = runtime.services.require("reader")
        reader.show(title="课堂讲义", docs=docs, active="gossip")

        state = runtime.ui.state()
        assert state["active_page"] == "reader:main"
        blocks = state["page_content"]["reader:main"]
        tabs = next(block for block in blocks if block["type"] == "tabs")
        assert [tab["title"] for tab in tabs["tabs"]] == ["知识", "闲话"]
        # The requested document is the active tab.
        assert tabs["tabs"][0]["active"] is False
        assert tabs["tabs"][1]["active"] is True
        # Markdown is rendered (and therefore KaTeX-able).
        markdown = [b for tab in tabs["tabs"] for b in tab["blocks"] if b["type"] == "markdown"]
        assert markdown and "定义" in markdown[0]["html"]
    finally:
        runtime.stop()


def test_missing_documents_show_placeholder(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        reader = runtime.services.require("reader")
        reader.show(title="空", docs=[{"key": "k", "label": "知识", "path": str(tmp_path / "nope.md")}])
        blocks = runtime.ui.state()["page_content"]["reader:main"]
        text_blocks = [b for b in blocks if b["type"] == "tabs"]
        inner = text_blocks[0]["tabs"][0]["blocks"]
        assert any(b.get("text") == "（暂无内容）" for b in inner)
    finally:
        runtime.stop()


def test_reader_action_rejects_empty_docs(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        result = runtime.ui.dispatch_action("reader.open", {"docs": []})
        assert not result.ok
        assert "文稿" in result.message
    finally:
        runtime.stop()


def test_reader_action_opens_page(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        docs = [{"key": "k", "label": "知识", "path": _write(tmp_path, "k.md", "# hi\n")}]
        result = runtime.ui.dispatch_action("reader.open", {"title": "T", "docs": docs})
        assert result.ok
        assert result.open_page == "reader:main"
        blocks = runtime.ui.state()["page_content"]["reader:main"]
        assert any(b["type"] == "markdown" and "T" in b["text"] for b in blocks)
    finally:
        runtime.stop()


def test_split_markdown_keeps_display_math_together() -> None:
    text = "para\n\n$$\na^2+b^2\n$$\n\n" + ("x" * 100 + "\n\n") * 5
    chunks = split_markdown(text, max_chars=60)
    assert all(chunk.count("$$") % 2 == 0 for chunk in chunks), chunks
