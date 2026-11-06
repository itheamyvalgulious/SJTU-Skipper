"""Default agent tools: info, read, replace, append, write.

Each tool is a ``ToolSpec`` with a JSON Schema parameter definition and a
callable handler.  The default four tools are always available unless the
caller provides an explicit *tools* argument to ``create_session``; ``write``
(full-file overwrite) is opt-in and must be requested by name.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .workspace import FileWorkspace


@dataclass(slots=True)
class ToolSpec:
    """A named tool with JSON Schema parameters and a callable handler.

    The handler receives keyword arguments matching the declared parameters
    and returns a string (the tool result).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def openai_schema(self) -> dict[str, Any]:
        """Return an OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Default four tools
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_NAMES = ("info", "read", "replace", "append")

_INFO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "要读取的文件名（逻辑名，由会话注册的工作区文件）",
        },
        "l": {
            "type": "integer",
            "description": "起始行号（1-based，含该行）。-1 表示从第一行开始。",
            "default": 1,
        },
        "r": {
            "type": "integer",
            "description": "结束行号（1-based，含该行）。-1 表示到最后一行。",
            "default": -1,
        },
    },
    "required": ["file"],
    "additionalProperties": False,
}

_REPLACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "要修改的文件名（逻辑名）",
        },
        "old_str": {
            "type": "string",
            "description": "要被替换的原始文本（必须在文件中恰好出现一次）",
        },
        "new_str": {
            "type": "string",
            "description": "替换后的新文本",
        },
    },
    "required": ["file", "old_str", "new_str"],
    "additionalProperties": False,
}

_APPEND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "要追加的文件名（逻辑名）",
        },
        "text": {
            "type": "string",
            "description": "要追加的文本内容（会自动在末尾添加换行符）",
        },
    },
    "required": ["file", "text"],
    "additionalProperties": False,
}

_WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "要整体写入的文件名（逻辑名）",
        },
        "text": {
            "type": "string",
            "description": "文件的完整新内容，将整体替换原有内容（不是追加）",
        },
    },
    "required": ["file", "text"],
    "additionalProperties": False,
}


def _safe_int(value: Any, default: int) -> int:
    """Coerce *value* to int, falling back to *default*."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_default_tools(
    workspace: FileWorkspace,
    session_getter: Callable[[], Any],
    meta_getter: Callable[[], str],
    max_tool_output: int = 8000,
    names: tuple[str, ...] = _DEFAULT_TOOL_NAMES,
) -> list[ToolSpec]:
    """Return the built-in tools selected by *names*."""

    tools: list[ToolSpec] = []

    if "info" in names:

        def tool_info(**kwargs: Any) -> str:
            session = session_getter()
            files: list[str] = []
            for fname in workspace.names:
                try:
                    p = workspace._files.get(fname)  # noqa: SLF001
                    if p is not None and p.exists():
                        size = p.stat().st_size
                        files.append(f"  {fname} ({size} bytes)")
                    else:
                        files.append(f"  {fname} (not created)")
                except Exception:
                    files.append(f"  {fname}")
            elapsed = time.time() - session.created_at
            total_chars = sum(
                len(str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", "")))
                for m in session.messages
            )
            lines = [
                meta_getter(),
                "",
                "## 当前会话",
                f"  session_id: {session.id}",
                f"  name: {session.name}",
                f"  model: {session.model}",
                f"  max_steps: {session.max_steps}",
                f"  messages: {len(session.messages)}",
                f"  started: {session.created_at:.1f}",
                f"  elapsed: {elapsed:.1f}s",
                f"  total_chars: {total_chars}",
                "",
                "## 可操作文件",
            ]
            if files:
                lines.extend(files)
            else:
                lines.append("  (无)")
            return "\n".join(lines)

        tools.append(
            ToolSpec(
                name="info",
                description="获取当前会话状态、可操作文件列表及所有插件的元信息。无参数。",
                parameters=_INFO_SCHEMA,
                handler=tool_info,
            )
        )

    if "read" in names:

        def tool_read(file: str, l: int = 1, r: int = -1, **kwargs: Any) -> str:
            if file not in workspace.names:
                return f"错误：未注册的文件 {file!r}"
            l = _safe_int(l, 1)
            r = _safe_int(r, -1)
            result = workspace.read_lines(file, l, r)
            if len(result) > max_tool_output:
                result = result[:max_tool_output] + "\n... (输出已截断)"
            return result

        tools.append(
            ToolSpec(
                name="read",
                description=(
                    "读取指定文件的指定行范围。行号从 1 开始，l 和 r 均包含在内。"
                    "l 或 r 设为 -1 表示该侧无界。返回内容不包含行号前缀，可直接复制用于 replace。"
                ),
                parameters=_READ_SCHEMA,
                handler=tool_read,
            )
        )

    if "replace" in names:

        def tool_replace(file: str, old_str: str, new_str: str, **kwargs: Any) -> str:
            if file not in workspace.names:
                return f"错误：未注册的文件 {file!r}"
            try:
                return workspace.replace_unique(file, old_str, new_str)
            except ValueError as exc:
                return f"错误：{exc}"

        tools.append(
            ToolSpec(
                name="replace",
                description=(
                    "在文件中执行唯一文本替换。old_str 必须在文件中恰好出现一次，"
                    "否则会报错（0 次或多次均报错，多次时报告出现次数与行号）。"
                ),
                parameters=_REPLACE_SCHEMA,
                handler=tool_replace,
            )
        )

    if "append" in names:

        def tool_append(file: str, text: str, **kwargs: Any) -> str:
            if file not in workspace.names:
                return f"错误：未注册的文件 {file!r}"
            return workspace.append_line(file, text)

        tools.append(
            ToolSpec(
                name="append",
                description="在文件末尾追加一行新文本。自动确保末尾有换行。",
                parameters=_APPEND_SCHEMA,
                handler=tool_append,
            )
        )

    if "write" in names:

        def tool_write(file: str, text: str, **kwargs: Any) -> str:
            if file not in workspace.names:
                return f"错误：未注册的文件 {file!r}"
            workspace.write(file, text)
            line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
            return f"已整体写入 {file}（{len(text)} 字符，{line_count} 行）"

        tools.append(
            ToolSpec(
                name="write",
                description=(
                    "用给定文本整体替换文件内容（覆盖写入，不是追加）。"
                    "适合重排大纲、合并条目、整篇重写；需要保留的旧内容必须先 read。"
                ),
                parameters=_WRITE_SCHEMA,
                handler=tool_write,
            )
        )

    return tools