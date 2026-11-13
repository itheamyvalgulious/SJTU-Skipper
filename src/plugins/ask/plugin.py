"""Ask plugin: in-note question answering with streaming and two intensities.

The plugin owns *questions*, not the agent loop: the ``llm-agent`` plugin runs
the model, builds the tools from the declarations below and streams deltas
back.  Answers live in the note as ``qa`` markdown blocks (owned by ``reader``).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from core.plugin import Plugin, PluginContext
from core.ui import render_markdown
from plugins.agent import AgentProfile, AgentTool
from plugins.settings.spec import SettingsSpec

from .prompts import ASK_PROMPT_TEMPLATE, INSTANT_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT

ACTION = "ask.ask"
SELECTION_ID = "ask.ask"

INSTANT = "instant"
RESEARCH = "research"

# Tool parameters for the research intensity; the read line cap lives here,
# not inside the tool implementation.
READ_MAX_LINES_DEFAULT = 50
SEARCH_MAX_HITS_DEFAULT = 200


class AskPlugin(Plugin):
    """Reader selection menu → in-note Q&A with a live, collapsible answer."""

    id = "ask"
    requires = ("settings", "llm-agent", "reader")

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._settings = ctx.require("settings")
        self._agent = ctx.require("agent")
        self._reader = ctx.require("reader")

        for spec in _SETTING_SPECS:
            self._settings.register(spec)

        ctx.register_service("ask", self)
        ctx.ui.register_action(ACTION, self._on_ask, plugin_id=self.id)

        self._reader.register_selection_action(
            "",
            {
                "id": SELECTION_ID,
                "label": "提问",
                "icon": "💬",
                "kind": "prompt",
                "action": ACTION,
                "placeholder": "针对选中的内容提问，Enter 发送",
                "intensities": [
                    {"value": INSTANT, "label": "快速"},
                    {"value": RESEARCH, "label": "研究"},
                ],
                "order": 10,
            },
            owner=self.id,
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _setting(self, key: str, default: Any) -> Any:
        return self._settings.get(f"ask.{key}", default)

    @property
    def _model(self) -> str:
        return str(self._setting("model", "") or "")

    @property
    def _instant_max_tokens(self) -> int:
        return max(1000, int(self._setting("instant_max_tokens", 16000)))

    @property
    def _research_max_tokens(self) -> int:
        return max(1000, int(self._setting("research_max_tokens", 131072)))

    @property
    def _research_max_steps(self) -> int:
        return max(1, int(self._setting("research_max_steps", 24)))

    @property
    def _before_words(self) -> int:
        return max(0, int(self._setting("before_words", 2000)))

    @property
    def _after_words(self) -> int:
        return max(0, int(self._setting("after_words", 2000)))

    @property
    def _history_words(self) -> int:
        return max(0, int(self._setting("history_words", 1000)))

    @property
    def _selection_words(self) -> int:
        return max(50, int(self._setting("selection_words", 800)))

    @property
    def _read_max_lines(self) -> int:
        return max(1, int(self._setting("read_max_lines", READ_MAX_LINES_DEFAULT)))

    @property
    def _search_max_hits(self) -> int:
        return max(1, int(self._setting("search_max_hits", SEARCH_MAX_HITS_DEFAULT)))

    @property
    def _instant_prompt(self) -> str:
        return str(self._setting("instant_prompt", INSTANT_SYSTEM_PROMPT) or INSTANT_SYSTEM_PROMPT)

    @property
    def _research_prompt(self) -> str:
        return str(self._setting("research_prompt", RESEARCH_SYSTEM_PROMPT) or RESEARCH_SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _on_ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "") or "").strip()
        selection = str(payload.get("selection", "") or "").strip()
        intensity = str(payload.get("intensity", "") or INSTANT)
        if intensity not in (INSTANT, RESEARCH):
            intensity = INSTANT
        path = str(payload.get("path", "") or "")
        if not question:
            return {"ok": False, "message": "问题不能为空"}

        created = self._reader.qa_begin(
            path,
            question=question,
            selection=selection,
            intensity=intensity,
        )
        qa_id = str(created.get("id", "") or "")
        threading.Thread(
            target=self._run,
            args=(path, qa_id, selection, intensity),
            name=f"ask-{qa_id}",
            daemon=True,
        ).start()

        message = "已提交，Agent 正在回答…"
        if not created.get("anchored", True):
            message += "（未能定位选中片段，已附在笔记末尾）"
        return {"ok": True, "message": message, "refresh": True}

    def _run(
        self, path: str, qa_id: str, selection: str, intensity: str
    ) -> None:
        live = _Live(self, path, qa_id)
        status = "done"
        answer = ""
        started = time.time()
        try:
            answer = self._answer(path, qa_id, selection, intensity, live)
        except Exception as exc:  # noqa: BLE001 - the failure belongs in the note
            self.ctx.logger.exception("ask %s failed", qa_id)
            status = "error"
            answer = f"（回答失败：{exc}）"

        self._reader.qa_finish(
            path,
            qa_id,
            answer=answer,
            status=status,
            collapsed=(status == "done"),
        )
        live.finish(status=status, answer=answer, elapsed=time.time() - started)

    def _answer(
        self,
        path: str,
        qa_id: str,
        selection: str,
        intensity: str,
        live: "_Live",
    ) -> str:
        context = self._reader.qa_context(
            path,
            qa_id,
            before_words=self._before_words,
            after_words=self._after_words,
            history_words=self._history_words,
        )
        prompt = self._build_prompt(path, context, selection, intensity)
        files = self._workspace_files(path) if intensity == RESEARCH else {}

        profile = self._profile(intensity)
        self._agent.register_agent(profile)
        session = self._agent.create_session(
            name=f"ask-{qa_id}",
            agent=profile.name,
            files=files,
            meta={"role": "ask", "qa_id": qa_id, "intensity": intensity},
        )

        started = time.time()
        try:
            result = self._agent.run(session.id, prompt, on_event=live.on_event)
        finally:
            self._agent.close(session.id)

        self.ctx.logger.info(
            "ask %s finished in %.1fs: intensity=%s steps=%s tokens=%s error=%s",
            qa_id,
            time.time() - started,
            intensity,
            result.steps,
            result.total_tokens,
            result.error,
        )
        if result.error:
            raise RuntimeError(result.error)
        return result.final_message.strip() or "（模型没有返回内容）"

    def _profile(self, intensity: str) -> AgentProfile:
        if intensity == INSTANT:
            return AgentProfile(
                name="ask.instant",
                system_prompt=self._instant_prompt,
                tools=[],
                model=self._model or None,
                max_steps=1,
                max_tokens=self._instant_max_tokens,
                llm_extra={"reasoning_effort": "max"},
                stream=True,
            )
        research_prompt = self._research_prompt
        if "{read_max_lines}" in research_prompt:
            research_prompt = research_prompt.format(
                read_max_lines=self._read_max_lines
            )
        return AgentProfile(
            name="ask.research",
            system_prompt=research_prompt,
            tools=[
                AgentTool("read", {"max_lines": self._read_max_lines}),
                AgentTool("search", {"max_hits": self._search_max_hits}),
            ],
            model=self._model or None,
            max_steps=self._research_max_steps,
            max_tokens=self._research_max_tokens,
            llm_extra={"reasoning_effort": "max"},
            final_answer_on_limit=True,
            stream=True,
        )

    def _build_prompt(
        self,
        path: str,
        context: Mapping[str, Any],
        selection: str,
        intensity: str,
    ) -> str:
        name = Path(path).name
        if intensity == RESEARCH:
            readable = "、".join(f"`{item}`" for item in self._workspace_files(path))
            instructions = (
                f"可以先 `search` 再用 `read` 核对材料；`read` 单次最多 {self._read_max_lines} 行。\n"
                f"可读文件：{readable}。"
            )
        else:
            instructions = "材料已经足够，直接回答；本题没有工具可用。"
        return ASK_PROMPT_TEMPLATE.format(
            course_name=self._course_name(),
            file_label=_doc_label(name),
            file_name=name,
            selection=_truncate_words(selection, self._selection_words)
            or "（浏览器没有回传选中文本）",
            before_words=self._before_words,
            before=context.get("before") or "（无）",
            after_words=self._after_words,
            after=context.get("after") or "（无）",
            history_words=self._history_words,
            history=_format_history(context.get("history") or []),
            question=context.get("question") or "",
            instructions=instructions,
        )

    def _workspace_files(self, path: str) -> dict[str, str]:
        notes_dir = Path(path).parent
        files = {
            item.name: str(item)
            for item in sorted(notes_dir.glob("*.md"))
            if item.is_file()
        }
        source = notes_dir.parent / "lecture_source.md"
        if source.exists():
            files["lecture_source.md"] = str(source)
        return files

    def _course_name(self) -> str:
        meta = self.ctx.streams.notes_meta or self.ctx.streams.current.meta or {}
        name: Any = meta.get("course_name") or meta.get("course")
        if isinstance(name, dict):
            name = name.get("course_name") or name.get("name") or ""
        return str(name or "").strip() or "未知课程"


# ---------------------------------------------------------------------------
# Live streaming helper
# ---------------------------------------------------------------------------
class _Live:
    """Accumulates the agent's stream and pushes throttled UI notifications."""

    _INTERVAL = 0.1

    def __init__(self, plugin: AskPlugin, path: str, qa_id: str) -> None:
        self._plugin = plugin
        self._path = path
        self._qa_id = qa_id
        self._phase = "thinking"
        self._tools: list[dict[str, Any]] = []
        self._reasoning = ""
        self._answer = ""
        self._pending_reasoning = ""
        self._pending_text = ""
        self._started = time.time()
        self._last = 0.0
        self._lock = threading.Lock()

    def on_event(self, event: Any) -> None:
        kind = event.type
        data = event.data or {}
        step = event.step
        logger = self._plugin.ctx.logger

        if kind == "agent.reasoning":
            delta = str(data.get("text") or "")
            if not delta:
                return
            with self._lock:
                self._reasoning += delta
                self._pending_reasoning += delta
                self._phase = "thinking"
            self.publish()
        elif kind == "agent.delta":
            delta = str(data.get("text") or "")
            if not delta:
                return
            with self._lock:
                self._answer += delta
                self._pending_text += delta
                self._phase = "answering"
            self.publish()
        elif kind == "agent.tool_call":
            name = str(data.get("name") or "?")
            arguments = data.get("arguments") or {}
            logger.info(
                "ask %s step %s tool %s %s",
                self._qa_id,
                step,
                name,
                json.dumps(arguments, ensure_ascii=False),
            )
            with self._lock:
                self._tools.append(
                    {"name": name, "detail": _tool_detail(name, arguments), "done": False}
                )
                self._phase = "tool"
            self.publish(force=True)
        elif kind == "agent.tool_result":
            logger.info(
                "ask %s step %s tool %s result: %s",
                self._qa_id,
                step,
                data.get("name", "?"),
                str(data.get("result", ""))[:500],
            )
            with self._lock:
                for tool in reversed(self._tools):
                    if not tool["done"]:
                        tool["done"] = True
                        break
            self.publish(force=True)
        elif kind == "agent.message":
            content = str(data.get("content") or "")
            if content:
                logger.info("ask %s step %s reply:\n%s", self._qa_id, step, content)
        elif kind == "agent.error":
            logger.warning(
                "ask %s step %s error: %s", self._qa_id, step, data.get("error")
            )

    def publish(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if not force and now - self._last < self._INTERVAL:
                return
            self._last = now
            reasoning = self._pending_reasoning
            text = self._pending_text
            self._pending_reasoning = ""
            self._pending_text = ""
            payload = {
                "path": self._path,
                "id": self._qa_id,
                "status": "pending",
                "phase": self._phase,
                "reasoning_delta": reasoning,
                "text_delta": text,
                "tools": [dict(tool) for tool in self._tools],
                "elapsed": round(now - self._started, 1),
            }
        self._plugin.ctx.ui.notify("qa.live", payload)

    def finish(self, *, status: str, answer: str, elapsed: float) -> None:
        payload = {
            "path": self._path,
            "id": self._qa_id,
            "status": status,
            "phase": "done" if status == "done" else "error",
            "answer": answer,
            "answer_html": render_markdown(answer),
            "collapsed": status == "done",
            "elapsed": round(elapsed, 1),
        }
        logger = self._plugin.ctx.logger
        # The full thinking trace goes to the log verbatim, never truncated.
        if self._reasoning:
            logger.info(
                "ask %s reasoning (%d chars):\n%s",
                self._qa_id,
                len(self._reasoning),
                self._reasoning,
            )
        logger.info(
            "ask %s answer (%s, %.1fs):\n%s",
            self._qa_id,
            status,
            elapsed,
            answer,
        )
        self._plugin.ctx.ui.notify("qa.live", payload)


def _tool_detail(name: str, arguments: Mapping[str, Any]) -> str:
    if name == "read":
        file = arguments.get("file", "?")
        left = arguments.get("l", 1)
        right = arguments.get("r", -1)
        return f"{file} {left}-{right}"
    if name == "search":
        keywords = arguments.get("keywords") or []
        file = arguments.get("file")
        text = "、".join(str(item) for item in keywords)
        return f"{text} @ {file}" if file else text
    return json.dumps(arguments, ensure_ascii=False)[:120]


def _doc_label(name: str) -> str:
    return {
        "note_knowledge.md": "知识",
        "note_gossip.md": "闲话与分享",
        "note_notice.md": "通知",
        "instant_note_knowledge.md": "即时知识",
        "instant_note_gossip.md": "即时闲话",
        "instant_note_notice.md": "即时通知",
    }.get(name, Path(name).stem)


_WORD_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+|[0-9]+")


def _truncate_words(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or not text:
        return text
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= limit:
        return text
    return text[: matches[limit - 1].end()] + "\n……（选中内容过长，已截断）"


def _format_history(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "（无）"
    parts: list[str] = []
    for index, entry in enumerate(entries, 1):
        parts.append(
            f"### 问答 {index}\n\n问：{entry.get('question', '')}\n\n"
            f"答：{entry.get('answer', '')}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
_SETTING_SPECS: list[SettingsSpec] = [
    SettingsSpec(
        key="ask.model",
        label="提问模型",
        type="string",
        default="",
        group="Ask · 提问",
        description="回答提问使用的模型（空=agent 默认）",
    ),
    SettingsSpec(
        key="ask.instant_max_tokens",
        label="快速模式输出上限",
        type="int",
        default=16000,
        group="Ask · 提问",
        description="快速模式单次调用的输出 token 上限（含思考 token）",
    ),
    SettingsSpec(
        key="ask.research_max_tokens",
        label="研究模式输出上限",
        type="int",
        default=131072,
        group="Ask · 提问",
        description="研究模式单次调用的输出 token 上限（含思考 token）",
    ),
    SettingsSpec(
        key="ask.research_max_steps",
        label="研究模式步数上限",
        type="int",
        default=24,
        group="Ask · 提问",
        description="研究模式最多几轮工具调用；最后一步强制直接回答",
    ),
    SettingsSpec(
        key="ask.before_words",
        label="选中前文词数",
        type="int",
        default=2000,
        group="Ask · 提问",
        description="上下文里选中段落之前的笔记内容（按词计，中文按字计）",
    ),
    SettingsSpec(
        key="ask.after_words",
        label="选中后文词数",
        type="int",
        default=2000,
        group="Ask · 提问",
        description="上下文里选中段落之后的笔记内容",
    ),
    SettingsSpec(
        key="ask.history_words",
        label="历史问答词数",
        type="int",
        default=1000,
        group="Ask · 提问",
        description="同一条笔记中此前问答记录的独立限额",
    ),
    SettingsSpec(
        key="ask.selection_words",
        label="选中内容词数",
        type="int",
        default=800,
        group="Ask · 提问",
        description="选中片段送入模型的上限，超出部分截断",
    ),
    SettingsSpec(
        key="ask.read_max_lines",
        label="研究模式 read 行数",
        type="int",
        default=50,
        group="Ask · 提问",
        description="研究模式单次 read 最多返回的行数",
    ),
    SettingsSpec(
        key="ask.search_max_hits",
        label="search 命中上限",
        type="int",
        default=200,
        group="Ask · 提问",
        description="search 工具最多返回的命中行数",
    ),
    SettingsSpec(
        key="ask.instant_prompt",
        label="快速模式提示词",
        type="text",
        default=INSTANT_SYSTEM_PROMPT,
        group="Ask · 提问",
        description="快速模式（无工具、一步回答）的系统提示词",
    ),
    SettingsSpec(
        key="ask.research_prompt",
        label="研究模式提示词",
        type="text",
        default=RESEARCH_SYSTEM_PROMPT,
        group="Ask · 提问",
        description="研究模式（search + read）的系统提示词；支持 {read_max_lines}",
    ),
]


__all__ = ["AskPlugin", "ACTION"]
