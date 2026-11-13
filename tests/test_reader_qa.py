"""Tests for the reader selection menu and inline Q&A blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import load_config
from core.runtime import build_runtime
from plugins.reader import ReaderPlugin
from plugins.reader import qa


def _make_runtime(tmp_path: Path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[ReaderPlugin()])
    runtime.start()
    return runtime


def _note(tmp_path: Path, name: str = "note_knowledge.md") -> Path:
    path = tmp_path / name
    path.write_text(
        "# 课堂讲义：知识\n\n## 第一节\n\n第一段内容，讲定义。\n\n"
        "第二段内容，讲定理 $a^2+b^2=c^2$ 的证明。\n\n第三段内容，讲例题。\n",
        encoding="utf-8",
    )
    return path


def _open(runtime: Any, path: Path, *, owner: str = "summarize", key: str = "s1") -> str:
    reader = runtime.services.require("reader")
    return reader.show(
        title="课堂讲义",
        docs=[{"key": "knowledge", "label": "知识", "path": str(path)}],
        active="knowledge",
        page_key=key,
        owner=owner,
    )


def _tabs(runtime: Any, page_id: str) -> list[dict[str, Any]]:
    blocks = runtime.ui.state()["page_content"][page_id]
    return next(block for block in blocks if block["type"] == "tabs")["tabs"]


# ---------------------------------------------------------------------------
# Selection menu registration is per reader page
# ---------------------------------------------------------------------------
def test_selection_actions_are_scoped_to_page_and_owner(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        page_id = _open(runtime, path)
        reader = runtime.services.require("reader")
        reader.register_selection_action(
            page_id,
            {
                "id": "demo.ask",
                "label": "提问",
                "kind": "prompt",
                "action": "demo.ask",
            },
            owner="summarize",
        )

        meta = _tabs(runtime, page_id)[0]["meta"]
        assert meta["doc_key"] == "knowledge"
        assert meta["path"] == str(path)
        assert [item["id"] for item in meta["selection_actions"]] == ["demo.ask"]
        assert meta["selection_actions"][0]["kind"] == "prompt"

        # The same page re-used by another plugin shows no summarize menu.
        _open(runtime, path, owner="instant-summarize")
        meta = _tabs(runtime, page_id)[0]["meta"]
        assert meta["owner"] == "instant-summarize"
        assert meta["selection_actions"] == []
    finally:
        runtime.stop()


def test_selection_actions_are_not_global(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        page_a = _open(runtime, path, key="a")
        reader = runtime.services.require("reader")
        reader.register_selection_action(
            page_a,
            {"id": "ask", "label": "提问", "action": "demo.ask"},
            owner="summarize",
        )
        page_b = _open(runtime, path, key="b")
        assert _tabs(runtime, page_b)[0]["meta"]["selection_actions"] == []
        runtime.ui.open_page(page_a)
        assert _tabs(runtime, page_a)[0]["meta"]["selection_actions"]
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# Inline Q&A blocks
# ---------------------------------------------------------------------------
def test_qa_begin_and_finish_render_blocks(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        page_id = _open(runtime, path)
        reader = runtime.services.require("reader")

        created = reader.qa_begin(
            str(path), question="为什么这一步成立？", selection="第二段内容"
        )
        assert created["anchored"] is True
        qa_id = created["id"]
        text = path.read_text("utf-8")
        assert "为什么这一步成立？" in text
        assert qa_id in text
        # The block sits below the selected paragraph, not at the end.
        assert text.index(qa_id) < text.index("第三段内容")

        blocks = _tabs(runtime, page_id)[0]["blocks"]
        qa_block = next(b for b in blocks if b["type"] == "qa")
        assert qa_block["status"] == "pending"
        assert qa_block["question"] == "为什么这一步成立？"
        assert qa_block["delete_action"] == "reader.qa_delete"
        assert "为什么这一步成立" in qa_block["question_html"]

        reader.qa_finish(str(path), qa_id, answer="因为 $a+b>c$。")
        blocks = _tabs(runtime, page_id)[0]["blocks"]
        qa_block = next(b for b in blocks if b["type"] == "qa")
        assert qa_block["status"] == "done"
        assert "$a+b" in qa_block["answer_html"]
        # Markers never leak into the markdown blocks.
        markdown = "".join(
            b.get("text", "") for b in blocks if b["type"] == "markdown"
        )
        assert "qa:begin" not in markdown
        assert "第三段内容" in markdown
    finally:
        runtime.stop()


def test_qa_delete_action_removes_the_pair(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        _open(runtime, path)
        reader = runtime.services.require("reader")
        created = reader.qa_begin(str(path), question="Q?", selection="第一段内容")
        reader.qa_finish(str(path), created["id"], answer="A!")

        result = runtime.ui.dispatch_action(
            "reader.qa_delete", {"path": str(path), "id": created["id"]}
        )
        assert result.ok
        text = path.read_text("utf-8")
        assert "Q?" not in text and "A!" not in text
        assert "第一段内容" in text and "第三段内容" in text

        # Unknown ids and unknown paths are rejected.
        assert not runtime.ui.dispatch_action(
            "reader.qa_delete", {"path": str(path), "id": "qa-deadbeef"}
        ).ok
        assert not runtime.ui.dispatch_action(
            "reader.qa_delete", {"path": "/etc/passwd", "id": created["id"]}
        ).ok
    finally:
        runtime.stop()


def test_qa_context_budgets_are_independent(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        _open(runtime, path)
        reader = runtime.services.require("reader")

        first = reader.qa_begin(str(path), question="第一个问题", selection="第一段内容")
        reader.qa_finish(str(path), first["id"], answer="第一个回答")

        second = reader.qa_begin(str(path), question="第二个问题", selection="第三段内容")
        context = reader.qa_context(
            str(path), second["id"], before_words=100, after_words=20, history_words=100
        )
        assert context["question"] == "第二个问题"
        assert "第一段内容" in context["before"]
        # Prior Q&A pairs are a separate bucket, not part of the note prose.
        assert "qa:begin" not in context["before"]
        assert context["history"] == [
            {"question": "第一个问题", "answer": "第一个回答"}
        ]

        limited = reader.qa_context(
            str(path), second["id"], before_words=3, after_words=3, history_words=1
        )
        assert qa.word_count(limited["before"]) <= 3
        assert limited["after"] == ""

        third = reader.qa_begin(str(path), question="第三个问题", selection="第三段内容")
        context = reader.qa_context(
            str(path), third["id"], history_words=1000, before_words=10, after_words=10
        )
        assert [entry["question"] for entry in context["history"]] == [
            "第一个问题",
            "第二个问题",
        ]
    finally:
        runtime.stop()


def test_qa_begin_rejects_unknown_path(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        _note(tmp_path)
        reader = runtime.services.require("reader")
        try:
            reader.qa_begin("/tmp/not-a-note.md", question="x", selection="")
        except ValueError:
            pass
        else:  # pragma: no cover - the guard must reject it
            raise AssertionError("unknown paths must be rejected")
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def test_qa_block_persists_intensity_and_fold(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    try:
        path = _note(tmp_path)
        page_id = _open(runtime, path)
        reader = runtime.services.require("reader")

        created = reader.qa_begin(
            str(path),
            question="为什么？",
            selection="第一段内容",
            intensity="research",
        )
        reader.qa_finish(str(path), created["id"], answer="因为定义。")
        block = next(b for b in _tabs(runtime, page_id)[0]["blocks"] if b["type"] == "qa")
        assert block["intensity"] == "research"
        assert block["collapsed"] is True
        assert '"intensity": "research"' in path.read_text("utf-8")
        assert '"collapsed": true' in path.read_text("utf-8")

        result = runtime.ui.dispatch_action(
            "reader.qa_collapse",
            {"path": str(path), "id": created["id"], "collapsed": False},
        )
        assert result.ok
        assert result.refresh is False
        assert '"collapsed": false' in path.read_text("utf-8")
        block = next(b for b in _tabs(runtime, page_id)[0]["blocks"] if b["type"] == "qa")
        assert block["collapsed"] is False
    finally:
        runtime.stop()


def test_insert_block_falls_back_to_the_end() -> None:
    text = "# 标题\n\n第一段。\n\n第二段。\n"
    block = qa.build_block("qa-1", "2026-11-10 10:00", question="Q", answer="A")
    updated, anchored = qa.insert_block(text, block, "完全不存在的文字")
    assert anchored is False
    assert updated.rstrip().endswith("<!-- qa:end -->")
    assert "第一段。" in updated


def test_rendered_math_does_not_break_anchoring() -> None:
    text = "# 标题\n\n第二段内容，讲定理 $a^2+b^2=c^2$ 的证明。\n\n后文段落。\n"
    block = qa.build_block("qa-2", "2026-11-10 10:00", question="Q", answer="A")
    # The browser returns rendered text (no $...$), so the exact line cannot
    # match; a plain prefix still resolves to the right paragraph.
    updated, anchored = qa.insert_block(
        text, block, "第二段内容，讲定理 a2+b2=c2 的证明。"
    )
    assert anchored is True
    assert updated.index("qa-2") < updated.index("后文段落")


def test_parse_and_remove_roundtrip() -> None:
    text = "# t\n\npara\n\n" + qa.build_block(
        "qa-3", "2026-11-10 10:00", question="问题", answer="回答"
    )
    blocks = qa.parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].question == "问题"
    assert blocks[0].answer == "回答"
    assert qa.remove_block(text, "qa-3").strip() == "# t\n\npara"
    assert qa.remove_block(text, "nope") is None