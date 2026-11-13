"""VLM blackboard recognition prompts.

The VLM is an *observer*: it only transcribes what is visible in the frame.
It must not infer, complete or correct anything.  Recent transcriptions are
passed back as text so the model can emit only what changed since the last
window instead of re-stating the whole board every time.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llm.types import ImagePart, TextPart

BOARD_RECOGNITION_PROMPT = """这是大学数学课堂在 {timestamp} 时的黑板板书图像。

【转录原则（绝对严格）】
1. 只转录图中真实可见的内容：不推断、不补全、不纠正、不猜测任何看不清的公式或文字。
2. 看不清的字符保留原样或写作 □，绝不用“应该是……”“大概是……”的方式补全。
3. 不要输出“黑板上的板书内容为”之类的引导语，直接给出板书内容。
4. LaTeX 只能使用 $（行内公式）与 $$（行间公式）；严禁使用 \\[、\\]、\\(、\\)。
5. 同一个公式、同一行推导或同一段文字只输出一次；不要因为反光、重影或重复笔迹重复转录。
6. 如果图中没有可辨认的板书，只回复【黑板暂无板书】；不要转录摄像头时间戳水印。

【手绘示意图】
只有能清晰辨认出粉笔或马克笔绘制的线条、曲线、箭头、几何图形时，才写成 Markdown 占位符：
![简短中文描述]()
黑板边框、白板边框、反光、阴影、摄像头时间戳、教师身体、桌椅、讲台、投影幕布一律不算。
纯公式与文字推导绝不添加占位符。

{history_block}

{output_rule}

不要输出问候寒暄。"""


BLANK_BOARD_MARKER = "【黑板暂无板书】"
NO_CHANGE_MARKER = "【无变化】"

_HISTORY_HEADER = """【历史转录（最近 {count} 次，仅供判断哪些是新增内容；不要重复这些内容）】
{items}"""

_HISTORY_ITEM = """--- 第 {index} 次（{when}）---
{text}"""

_FIRST_WINDOW_HISTORY = """【历史转录】
（无。这是本节课第一次识别。）"""

_FIRST_WINDOW_RULE = "请完整转录当前黑板。"

_INCREMENTAL_RULE = """只输出当前画面相对最近一次转录**新增、修改或擦除**的内容；仍然保留在黑板上的旧内容不要重复。
如果与最近一次转录完全一致，只回复【无变化】。"""


def build_recognition_prompt(
    template: str,
    *,
    timestamp: str = "",
    history: Sequence[str] = (),
) -> str:
    """Fill the ``{timestamp}`` / ``{history_block}`` / ``{output_rule}`` slots.

    Custom prompts saved in settings may not contain the new placeholders, so
    the history block and output rule are appended when missing.
    """

    text = str(template or "").replace("{timestamp}", timestamp)
    items = [str(item).strip() for item in history if str(item).strip()]
    if items:
        rendered = "\n\n".join(
            _HISTORY_ITEM.format(
                index=index + 1,
                when="最近" if index == len(items) - 1 else "较早",
                text=item,
            )
            for index, item in enumerate(items)
        )
        history_block = _HISTORY_HEADER.format(count=len(items), items=rendered)
        output_rule = _INCREMENTAL_RULE
    else:
        history_block = _FIRST_WINDOW_HISTORY
        output_rule = _FIRST_WINDOW_RULE

    if "{history_block}" in text:
        text = text.replace("{history_block}", history_block)
    else:
        text = f"{text}\n\n{history_block}"
    if "{output_rule}" in text:
        text = text.replace("{output_rule}", output_rule)
    else:
        text = f"{text}\n\n{output_rule}"
    return text


def build_user_message(
    *,
    timestamp: str = "",
    image_data_url: str = "",
    prompt_text: str | None = None,
    history: Sequence[str] = (),
) -> list[TextPart | ImagePart]:
    """Build an OpenAI-style user message with text + image.

    Parameters
    ----------
    timestamp:
        Human-readable timestamp string (e.g. ``"00:30-01:00"``).
    image_data_url:
        Data URL of the fused board image (``data:image/jpeg;base64,...``).
    history:
        Recent transcriptions of the same board, oldest first.  They are
        text-only: no extra images are attached.

    Returns
    -------
    list[TextPart | ImagePart]
        A typed content list suitable for :class:`llm.types.Message`.
    """
    if prompt_text is None:
        prompt_text = BOARD_RECOGNITION_PROMPT
    text = build_recognition_prompt(
        prompt_text, timestamp=timestamp, history=history
    )
    content: list[TextPart | ImagePart] = [TextPart(text=text)]
    if image_data_url:
        content.append(ImagePart(url=image_data_url))
    return content


def collapse_repetitions(text: str, *, max_period: int = 8) -> str:
    """Collapse consecutive identical blocks in a VLM transcription.

    Long math transcriptions sometimes degenerate into repeating the same
    display-math block dozens of times.  The model did not do this in the
    legacy single-frame pipeline; with fused windows it occasionally does.
    Keep one copy of every consecutive repeated block (display math blocks are
    treated as one unit).
    """

    if not text or not text.strip():
        return text

    lines = text.splitlines()
    units: list[str] = []
    buf: list[str] = []
    in_display = False
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if in_display:
            buf.append(line)
            if stripped == "$$":
                units.append("\n".join(buf).strip())
                buf = []
                in_display = False
            continue
        if stripped == "$$":
            if buf:
                units.append("\n".join(buf).strip())
                buf = []
            in_display = True
            buf = [line]
            continue
        if not stripped:
            continue
        units.append(stripped)
    if buf:
        units.append("\n".join(buf).strip())

    if len(units) < 2:
        return text

    out: list[str] = []
    changed = False
    i = 0
    while i < len(units):
        collapsed = False
        limit = min(max_period, (len(units) - i) // 2)
        for period in range(1, limit + 1):
            block = units[i : i + period]
            if units[i + period : i + 2 * period] != block:
                continue
            j = i + period
            while j + period <= len(units) and units[j : j + period] == block:
                j += period
            out.extend(block)
            i = j
            changed = True
            collapsed = True
            break
        if not collapsed:
            out.append(units[i])
            i += 1

    if not changed:
        return text
    return "\n\n".join(out)


# Matches one or more ![description]() placeholders, possibly with surrounding
# whitespace/newlines, but no other substantive text, LaTeX, or markdown.
_TRIVIAL_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:!\[[^\]]*\]\(\)\s*)+$",
    re.DOTALL,
)


def is_trivial_board_text(text: str) -> bool:
    """Return True when *text* contains only diagram placeholders with no
    real mathematical or textual content.

    This guards against VLM hallucinations that emit ``![...]()`` when the
    board is actually empty; such outputs must be skipped just like the
    ``BLANK_BOARD_MARKER``.
    """
    if not text or not text.strip():
        return True
    # Strip leading/trailing whitespace and check whether every non-empty
    # "unit" is a placeholder.
    stripped = text.strip()
    if stripped == "【黑板暂无板书】":
        # The canonical marker is filtered explicitly by the plugin; keep
        # this predicate focused on hallucinated/empty variants.
        return False
    normalized = stripped.strip("【】[]（）() \t\n")
    if "黑板暂无板书" in normalized or normalized in {
        "暂无板书",
        "黑板为空",
        "无板书",
        "empty blackboard",
    }:
        return True
    if _TRIVIAL_PLACEHOLDER_RE.match(stripped):
        return True
    # Also catch cases where the model wraps the placeholder in a paragraph
    # like "![...]()\n" or lines that are purely placeholders with no
    # $/$$/chinese/english content.
    non_placeholder = _TRIVIAL_PLACEHOLDER_RE.sub("", stripped).strip()
    if not non_placeholder:
        return True
    # If after removing all placeholders we only have whitespace, skip.
    return False