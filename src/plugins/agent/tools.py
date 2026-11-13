"""Default agent tools: info, read, search, replace, append, write.

Every tool is built by a named factory.  Behavior plugins declare the tools
they want *and their parameters* when they register an agent profile, e.g.
``read`` with ``max_lines=50``; nothing about a particular caller lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .workspace import FileWorkspace

_DEFAULT_TOOL_NAMES = ("info", "read", "replace", "append")


@dataclass(slots=True)
class ToolSpec:
    """A named tool with JSON Schema parameters and a callable handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolContext:
    """Everything a tool factory may need, independent of any plugin."""

    workspace: FileWorkspace
    session_getter: Callable[[], Any]
    meta_getter: Callable[[], str]
    max_tool_output: int = 8000


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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

_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要查找的关键字（任一命中即算）。",
        },
        "file": {
            "type": "string",
            "description": "只查这个逻辑文件名；省略则查全部可读文件。",
        },
    },
    "required": ["keywords"],
    "additionalProperties": False,
}

_REPLACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "description": "要修改的文件名（逻辑名）"},
        "old_str": {
            "type": "string",
            "description": "要被替换的原始文本（必须在文件中恰好出现一次）",
        },
        "new_str": {"type": "string", "description": "替换后的新文本"},
    },
    "required": ["file", "old_str", "new_str"],
    "additionalProperties": False,
}

_APPEND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "description": "要追加的文件名（逻辑名）"},
        "text": {"type": "string", "description": "要追加的文本内容（会自动在末尾添加换行符）"},
    },
    "required": ["file", "text"],
    "additionalProperties": False,
}

_WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {"type": "string", "description": "要整体写入的文件名（逻辑名）"},
        "text": {
            "type": "string",
            "description": "文件的完整新内容，将整体替换原有内容（不是追加）",
        },
    },
    "required": ["file", "text"],
    "additionalProperties": False,
}


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def info_tool(ctx: ToolContext) -> ToolSpec:
    def handler(**kwargs: Any) -> str:
        session = ctx.session_getter()
        files: list[str] = []
        for name in ctx.workspace.visible_names:
            path = ctx.workspace._files.get(name)  # noqa: SLF001
            if path is not None and path.exists():
                files.append(f"  {name} ({path.stat().st_size} bytes)")
            else:
                files.append(f"  {name} (not created)")
        elapsed = time.time() - session.created_at
        total_chars = sum(
            len(str(message.get("content", ""))) for message in session.messages
        )
        lines = [
            ctx.meta_getter(),
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
        lines.extend(files or ["  (无)"])
        return "\n".join(lines)

    return ToolSpec(
        name="info",
        description="获取当前会话状态、可操作文件列表及所有插件的元信息。无参数。",
        parameters=_INFO_SCHEMA,
        handler=handler,
    )


def read_tool(ctx: ToolContext, *, max_lines: int | None = None) -> ToolSpec:
    description = (
        "读取指定文件的指定行范围。行号从 1 开始，l 和 r 均包含在内。"
        "l 或 r 设为 -1 表示该侧无界。返回内容不包含行号前缀，可直接复制用于 replace。"
    )
    if max_lines:
        description += f"单次最多返回 {max_lines} 行，超出会被截断。"

    def handler(file: str, l: int = 1, r: int = -1, **kwargs: Any) -> str:
        if file not in ctx.workspace.names:
            return f"错误：未注册的文件 {file!r}"
        result = ctx.workspace.read_lines(file, _safe_int(l, 1), _safe_int(r, -1))
        if max_lines is not None:
            lines = result.splitlines(keepends=True)
            if len(lines) - 1 > max_lines:
                result = (
                    "".join(lines[: max_lines + 1])
                    + f"\n[已截断：单次 read 最多 {max_lines} 行，请用 l/r 缩小范围继续]"
                )
        if len(result) > ctx.max_tool_output:
            result = result[: ctx.max_tool_output] + "\n... (输出已截断)"
        return result

    return ToolSpec(
        name="read",
        description=description,
        parameters=_READ_SCHEMA,
        handler=handler,
    )


def search_tool(ctx: ToolContext, *, max_hits: int = 200) -> ToolSpec:
    def handler(keywords: Any, file: str | None = None, **kwargs: Any) -> str:
        if isinstance(keywords, str):
            keywords = [keywords]
        terms = [str(term).strip() for term in keywords]
        terms = [term for term in terms if term]
        if not terms:
            return "错误：keywords 不能为空"
        names = [file] if file else list(ctx.workspace.visible_names)
        for name in names:
            if name not in ctx.workspace.names:
                return f"错误：未注册的文件 {name!r}"

        sections: list[str] = []
        hits = 0
        for name in names:
            found: list[str] = []
            for number, line in enumerate(
                ctx.workspace.text(name).splitlines(), 1
            ):
                lowered = line.lower()
                if any(term.lower() in lowered for term in terms):
                    text = line.strip()
                    if len(text) > 300:
                        text = text[:300] + "…"
                    found.append(f"({number}, {text})")
                    hits += 1
                    if hits >= max_hits:
                        break
            if found:
                sections.append(f"## {name}（{len(found)} 处）\n" + "\n".join(found))
            if hits >= max_hits:
                break
        if not sections:
            return "（没有匹配的行）"
        if hits >= max_hits:
            sections.append(f"[只显示前 {max_hits} 处，请用更精确的关键字]")
        return "\n\n".join(sections)

    return ToolSpec(
        name="search",
        description=(
            "按关键字在文件中查找行，返回 (行号, 该行内容) 组成的列表；"
            "keywords 是若干关键字（任一命中即算），file 可选（省略则查全部文件）。"
            "先用 search 定位，再用 read 读取命中行附近的 50 行左右内容。"
        ),
        parameters=_SEARCH_SCHEMA,
        handler=handler,
    )


def replace_tool(ctx: ToolContext) -> ToolSpec:
    def handler(file: str, old_str: str, new_str: str, **kwargs: Any) -> str:
        if file not in ctx.workspace.names:
            return f"错误：未注册的文件 {file!r}"
        try:
            return ctx.workspace.replace_unique(file, old_str, new_str)
        except ValueError as exc:
            return f"错误：{exc}"

    return ToolSpec(
        name="replace",
        description=(
            "在文件中执行唯一文本替换。old_str 必须在文件中恰好出现一次，"
            "否则会报错（0 次或多次均报错，多次时报告出现次数与行号）。"
        ),
        parameters=_REPLACE_SCHEMA,
        handler=handler,
    )


def append_tool(ctx: ToolContext) -> ToolSpec:
    def handler(file: str, text: str, **kwargs: Any) -> str:
        if file not in ctx.workspace.names:
            return f"错误：未注册的文件 {file!r}"
        return ctx.workspace.append_line(file, text)

    return ToolSpec(
        name="append",
        description="在文件末尾追加一行新文本。自动确保末尾有换行。",
        parameters=_APPEND_SCHEMA,
        handler=handler,
    )


def write_tool(ctx: ToolContext) -> ToolSpec:
    def handler(file: str, text: str, **kwargs: Any) -> str:
        if file not in ctx.workspace.names:
            return f"错误：未注册的文件 {file!r}"
        ctx.workspace.write(file, text)
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
        return f"已整体写入 {file}（{len(text)} 字符，{lines} 行）"

    return ToolSpec(
        name="write",
        description=(
            "用给定文本整体替换文件内容（覆盖写入，不是追加）。"
            "适合重排大纲、合并条目、整篇重写；需要保留的旧内容必须先 read。"
        ),
        parameters=_WRITE_SCHEMA,
        handler=handler,
    )


TOOL_FACTORIES: dict[str, Callable[..., ToolSpec]] = {
    "info": info_tool,
    "read": read_tool,
    "search": search_tool,
    "replace": replace_tool,
    "append": append_tool,
    "write": write_tool,
}


def build_tool(
    name: str, ctx: ToolContext, params: Mapping[str, Any] | None = None
) -> ToolSpec:
    """Instantiate one tool by name; unknown names raise ``KeyError``."""

    return TOOL_FACTORIES[name](ctx, **dict(params or {}))


def build_tools(
    declarations: Iterable["str | tuple[str, Mapping[str, Any]] | ToolSpec"],
    ctx: ToolContext,
) -> dict[str, ToolSpec]:
    """Build a tool mapping from names, ``(name, params)`` pairs or specs."""

    tools: dict[str, ToolSpec] = {}
    for declaration in declarations:
        if isinstance(declaration, ToolSpec):
            tools[declaration.name] = declaration
            continue
        if isinstance(declaration, str):
            name, params = declaration, {}
        else:
            name, params = declaration
        tools[name] = build_tool(str(name), ctx, params)
    return tools


def _build_default_tools(
    workspace: FileWorkspace,
    session_getter: Callable[[], Any],
    meta_getter: Callable[[], str],
    max_tool_output: int = 8000,
    names: tuple[str, ...] = _DEFAULT_TOOL_NAMES,
) -> list[ToolSpec]:
    """Legacy helper: instantiate the named built-ins with default params."""

    ctx = ToolContext(
        workspace=workspace,
        session_getter=session_getter,
        meta_getter=meta_getter,
        max_tool_output=max_tool_output,
    )
    return [build_tool(name, ctx) for name in names]
