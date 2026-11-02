"""VLM blackboard recognition prompts.

Inherits the strict LaTeX and diagram-placeholder conventions from the
original ``process_single_frame`` prompt in ``legacy/blackboard_lecture_agent.py``.
"""

from __future__ import annotations

import re

from llm.types import ImagePart, TextPart

BOARD_RECOGNITION_PROMPT = """这是大学数学课堂在 {timestamp} 时的黑板板书图像。

请准确识别并完整转录图中的全部手写数学板书内容（包括引理、定理、定义、命题、推论、证明推导细节、数学公式及中文说明）。

【LaTeX 公式规范（绝对严格）】：
1. 只能使用 $（行内公式）与 $$（行间公式）。严禁使用 \\[、\\]、\\(、\\)。
2. 行内公式外侧留空格，内侧紧贴不留空格；行间公式 $$ 独占一行前后各空一行。
3. 如果黑板上尚未书写数学内容或黑板为空，请仅回复【黑板暂无板书】，不要转录监控摄像机的时间戳水印。

【手绘示意图占位符规范（绝对严格）】：
对黑板上的手绘示意图（如几何图形、闭合曲线围道、区域映射示意图、复平面图、图表等），必须使用 Markdown 图片语法占位符：
![{对图形的简短中文描述}]()
例如黑板画了同心圆，必须写入：![同心圆区域示意图]()。
严禁写成普通文字描述或括号备注（如严禁写成"（手绘示意图：…）"）。

**示意图判定红线（违反即错误）**：
- 只有能清晰辨认出粉笔或马克笔绘制的线条、曲线、箭头、几何图形时，才能输出 ![...]() 占位符。
- 黑板边框、白板边框、反光、阴影、摄像头时间戳数字字符、水印文字 **绝不能** 生成示意图占位符。
- 如果图中只有空白黑板或白板（无任何粉笔字迹或手绘线条），只回复【黑板暂无板书】，绝对不要凭空编造任何示意图占位符。
- 教师身体、桌椅、讲台、投影幕布等非板书内容绝不能触发示意图占位符。

不需要精确内容或精确画，而只需要有该 ![...]() 格式的占位符即可。
注意：必须严格复刻黑板上的示意图，不可随意增减；纯公式与文字推导绝不添加占位符。

【禁止重复】：同一个公式、同一行推导或同一段文字只能输出一次；不要因为黑板上有重复笔迹、反光或重叠而重复转录。

请直接输出板书转录结果，不得有任何省略或总结概括，不要输出问候寒暄。"""


BLANK_BOARD_MARKER = "【黑板暂无板书】"


def build_user_message(
    *,
    timestamp: str = "",
    image_data_url: str = "",
    prompt_text: str | None = None,
) -> list[TextPart | ImagePart]:
    """Build an OpenAI-style user message with text + image.

    Parameters
    ----------
    timestamp:
        Human-readable timestamp string (e.g. ``"00:30-01:00"``).
    image_data_url:
        Data URL of the fused board image (``data:image/jpeg;base64,...``).

    Returns
    -------
    list[TextPart | ImagePart]
        A typed content list suitable for :class:`llm.types.Message`.
    """
    if prompt_text is None:
        prompt_text = BOARD_RECOGNITION_PROMPT
    prompt_text = prompt_text.replace("{timestamp}", timestamp)
    content: list[TextPart | ImagePart] = [TextPart(text=prompt_text)]
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