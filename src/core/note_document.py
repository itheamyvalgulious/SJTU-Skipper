"""Q&A blocks use HTML comment delimiters below the selected paragraph.

Plain Markdown viewers show the questions and answers. The reader replaces
these spans with collapsible controls.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from enum import StrEnum

from .ui import BlockType


class QaStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class AskIntensity(StrEnum):
    INSTANT = "instant"
    RESEARCH = "research"


QUESTION_PREFIX = "提问"
ANSWER_PREFIX = "回答"

_BLOCK_RE = re.compile(
    r"<!--\s*qa:begin\s*(?P<meta>\{.*?\})\s*-->(?P<body>.*?)<!--\s*qa:end\s*-->",
    re.S,
)
_QUESTION_HEAD_RE = re.compile(r"^[ \t]*\*\*[^\n]*提问[^\n]*\*\*[ \t]*(?:\n|$)", re.M)
_ANSWER_SEP_RE = re.compile(r"^[ \t]*\*\*[^\n]*回答[^\n]*\*\*[ \t]*(?:\n|$)", re.M)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_LEADING_BLOCK_RE = re.compile(
    r"[ \t]*<!--\s*qa:begin\s*\{.*?\}\s*-->.*?<!--\s*qa:end\s*-->", re.S
)


@dataclass(slots=True)
class QABlock:
    """One parsed question/answer pair and its character span in the file."""

    id: str
    ts: str
    status: QaStatus
    question: str
    answer: str
    start: int
    end: int
    intensity: AskIntensity | None = None
    collapsed: bool = False


def new_id() -> str:
    return f"qa-{uuid.uuid4().hex[:8]}"


def build_block(
    qa_id: str,
    ts: str,
    *,
    status: QaStatus = QaStatus.PENDING,
    question: str = "",
    answer: str = "",
    intensity: AskIntensity | None = None,
    collapsed: bool = False,
) -> str:
    """Return the canonical on-disk representation of one Q&A pair."""

    meta = json.dumps(
        {
            "id": qa_id,
            "ts": ts,
            "status": status,
            "intensity": intensity,
            "collapsed": bool(collapsed),
        },
        ensure_ascii=False,
    )
    answer_text = (answer or "").strip() or "_正在思考…_"
    lines = [
        f"<!-- qa:begin {meta} -->",
        f"**{QUESTION_PREFIX}（{ts}）**",
        "",
        (question or "").strip(),
        "",
        f"**{ANSWER_PREFIX}**",
        "",
        answer_text,
        "",
        "<!-- qa:end -->",
    ]
    return "\n".join(lines) + "\n"


def parse_blocks(text: str) -> list[QABlock]:
    """Return every Q&A pair in *text*, in document order."""

    blocks: list[QABlock] = []
    for match in _BLOCK_RE.finditer(text):
        try:
            meta = json.loads(match.group("meta"))
        except (json.JSONDecodeError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        question, answer = _split_body(match.group("body"))
        blocks.append(
            QABlock(
                id=str(meta.get("id") or ""),
                ts=str(meta.get("ts") or ""),
                status=QaStatus(meta.get("status") or QaStatus.DONE),
                question=question,
                answer=answer,
                start=match.start(),
                end=match.end(),
                intensity=AskIntensity(meta["intensity"])
                if meta.get("intensity")
                else None,
                collapsed=bool(meta.get("collapsed")),
            )
        )
    return blocks


def segments(text: str) -> list[tuple[BlockType, object]]:
    """Split *text* into ``("markdown", str)`` / ``("qa", QABlock)`` pieces."""

    out: list[tuple[BlockType, object]] = []
    cursor = 0
    for block in parse_blocks(text):
        if block.start > cursor:
            out.append((BlockType.MARKDOWN, text[cursor : block.start]))
        out.append((BlockType.QA, block))
        cursor = block.end
    if cursor < len(text):
        out.append((BlockType.MARKDOWN, text[cursor:]))
    return out


def strip_blocks(text: str) -> str:
    """Remove every Q&A block (used to build pure note context)."""

    cleaned = _BLOCK_RE.sub("", text)
    return _MULTI_BLANK_RE.sub("\n\n", cleaned)


def replace_block(text: str, qa_id: str, block: str) -> str | None:
    """Replace the block with id *qa_id* by *block*; ``None`` when missing."""

    for existing in parse_blocks(text):
        if existing.id != qa_id:
            continue
        return text[: existing.start] + block.strip("\n") + text[existing.end :]
    return None


def remove_block(text: str, qa_id: str) -> str | None:
    """Delete the Q&A pair *qa_id*; ``None`` when it is not in *text*."""

    for existing in parse_blocks(text):
        if existing.id != qa_id:
            continue
        before = text[: existing.start].rstrip("\n")
        after = text[existing.end :].lstrip("\n")
        if before and after:
            return f"{before}\n\n{after}"
        if before:
            return before + "\n"
        return after
    return None


def find_anchor(text: str, selection: str) -> int | None:
    """Return the offset of the paragraph the user selected, or ``None``.

    The browser hands us rendered text (math is already typeset, emphasis
    markers are gone), so an exact match is tried first, then a
    whitespace-insensitive match, then progressively shorter prefixes of each
    selected line.  When everything fails the caller appends at the end.
    """

    sel = (selection or "").strip()
    if not sel:
        return None

    index = text.find(sel)
    if index >= 0:
        return index

    compact, positions = _compact(text)
    target = _compact(sel)[0]
    if target:
        found = compact.find(target)
        if found >= 0:
            return positions[found]

    lines = [line.strip() for line in sel.splitlines() if line.strip()]
    lines.sort(key=len, reverse=True)
    for line in lines[:8]:
        for size in (len(line), 48, 32, 24, 16, 10, 6):
            if size > len(line):
                continue
            probe = line[:size].strip()
            if len(probe) < 4:
                continue
            index = text.find(probe)
            if index >= 0:
                return index
            found = compact.find(_compact(probe)[0])
            if found >= 0:
                return positions[found]
    return None


def insert_block(text: str, block: str, selection: str = "") -> tuple[str, bool]:
    """Insert *block* below the paragraph containing *selection*.

    Returns ``(new_text, anchored)``; ``anchored`` is False when the selection
    could not be located and the block was appended at the end.
    """

    body = block.strip("\n")
    anchor = find_anchor(text, selection)
    if anchor is None:
        base = text.rstrip("\n")
        return (f"{base}\n\n{body}\n" if base else f"{body}\n"), False

    end = text.find("\n\n", anchor)
    if end < 0:
        base = text.rstrip("\n")
        return f"{base}\n\n{body}\n", True
    end += 2
    # Keep chronological order: a new question goes *below* the Q&A pairs
    # already hanging under the same paragraph.
    while True:
        match = _LEADING_BLOCK_RE.match(text, end)
        if match is None:
            break
        end = match.end()
        while end < len(text) and text[end] == "\n":
            end += 1
    head = text[:end].rstrip("\n")
    tail = text[end:].lstrip("\n")
    pieces = [head, body]
    if tail:
        pieces.append(tail)
    return "\n\n".join(pieces) + "\n", True


# CJK characters count as one word each.
_WORD_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+|[0-9]+")


def word_count(text: str) -> int:
    return sum(1 for _ in _WORD_RE.finditer(text or ""))


def tail_words(text: str, limit: int) -> str:
    """Return roughly the last *limit* words of *text*."""

    if limit <= 0 or not text:
        return ""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= limit:
        return text.strip("\n")
    start = matches[len(matches) - limit].start()
    return text[start:].strip("\n")


def head_words(text: str, limit: int) -> str:
    """Return roughly the first *limit* words of *text*."""

    if limit <= 0 or not text:
        return ""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= limit:
        return text.strip("\n")
    end = matches[limit - 1].end()
    return text[:end].strip("\n")


def _split_body(body: str) -> tuple[str, str]:
    match = _ANSWER_SEP_RE.search(body)
    if match:
        question_part = body[: match.start()]
        answer_part = body[match.end() :]
    else:
        question_part, answer_part = body, ""
    question = _QUESTION_HEAD_RE.sub("", question_part, count=1).strip()
    if not question:
        question = question_part.strip()
    return question, answer_part.strip()


# Remove Markdown syntax from the search key, but keep source offset mappings.
_SYNTAX_CHARS = frozenset("$*_`#>|\\~^")


def _compact(text: str) -> tuple[str, list[int]]:
    """Drop whitespace and Markdown markers, keeping a map to source indices."""

    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace() or char in _SYNTAX_CHARS:
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions
