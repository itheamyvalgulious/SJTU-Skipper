"""Deterministic in-process LLM used by ``sjtu-skipper demo``.

No network, no API key.  It understands the marker prefixes in the demo
script and emits real tool calls, so the actual agent / summarize / file
pipeline is exercised end to end.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace
from typing import Any

from core.plugin import Plugin

_CATEGORY = {"知识": "knowledge.md", "闲聊": "gossip.md", "通知": "notice.md"}


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", part)))
        return "\n".join(parts)
    return str(content or "")


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return _flatten_content(message.get("content", ""))
    return _flatten_content(getattr(message, "content", ""))


def _response(
    content: str = "",
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        raw={"provider": "demo"},
        model="demo",
    )


class DemoLLM:
    """Profile-aware scripted LLM."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._tool_seq = 0

    def chat(self, **kwargs: Any) -> SimpleNamespace:
        with self._lock:
            self.calls.append(kwargs)
        profile = str(kwargs.get("profile") or "agent")
        messages = list(kwargs.get("messages") or [])
        if profile == "vlm":
            return _response(
                "【板书】留数定理：设 $f$ 在简单闭曲线 $C$ 及其内部除有限个孤立奇点外解析，"
                "则 $\\int_C f(z)\\,dz=2\\pi i\\sum_k \\operatorname{Res}(f,z_k)$。"
            )
        if profile == "stt":
            return _response("演示转写文本。")
        return self._agent_response(messages)

    def _agent_response(self, messages: list[Any]) -> SimpleNamespace:
        # Second round: tool results are back; finish.
        if any(
            (isinstance(m, dict) and m.get("role") == "tool")
            or (not isinstance(m, dict) and getattr(m, "role", "") == "tool")
            for m in messages
        ):
            return _response("本轮笔记整理完成。")

        # Only parse user messages: the summarize prompt wraps transcript
        # lines with a timestamp prefix such as "[00:00:01 - 00:00:02] 【知识】…".
        user_text = "\n".join(
            _message_text(m)
            for m in messages
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", ""))
            == "user"
        )
        if "新建会话" in user_text or "演示模式" in user_text:
            return _response("演示模式已就绪。点击左侧「演示」开始。")

        buckets: dict[str, list[str]] = {"knowledge.md": [], "gossip.md": [], "notice.md": []}
        for raw in user_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            match = re.search(r"【(知识|闲聊|通知)】\s*(.*)$", line)
            if match:
                buckets[_CATEGORY[match.group(1)]].append(match.group(2).strip())
            elif "【板书】" in line:
                board = re.sub(r"^.*?【板书】\s*", "", line).strip()
                if board:
                    buckets["knowledge.md"].append(board)

        if not any(buckets.values()):
            if "新增课堂内容" in user_text:
                buckets["knowledge.md"].append(user_text[-800:])
            else:
                return _response("我是演示 Agent。请在左侧「演示」中开始一段合成课堂。")

        tool_calls = []
        for filename, lines in buckets.items():
            if not lines:
                continue
            self._tool_seq += 1
            body = "\n".join(f"- {line}" for line in lines)
            tool_calls.append(
                SimpleNamespace(
                    id=f"demo-call-{self._tool_seq}",
                    name="append",
                    arguments={
                        "file": filename,
                        "text": f"\n### 片段 {self._tool_seq}\n{body}\n",
                    },
                )
            )
        if not tool_calls:
            return _response("没有可整理的新内容。")
        return _response("", tool_calls=tool_calls, finish_reason="tool_calls")

    def stream(self, **kwargs: Any):
        response = self.chat(**kwargs)
        if response.content:
            yield SimpleNamespace(
                type="text", text=response.content, tool_call=None, response=None, error=None
            )
        yield SimpleNamespace(
            type="done", text="", tool_call=None, response=response, error=None
        )

    def config_for(self, profile: str) -> SimpleNamespace:
        return SimpleNamespace(
            profile=profile,
            base_url="demo://local",
            api_key="demo",
            model="demo",
            timeout=10.0,
            max_retries=0,
        )

    def info(self) -> dict[str, Any]:
        return {"provider": "demo", "calls": len(self.calls), "registered": True}

    def close(self) -> None:
        pass


class DemoLLMPlugin(Plugin):
    """Plugin with id ``llm`` that registers :class:`DemoLLM`."""

    id = "llm"
    requires: tuple[str, ...] = ("settings",)
    provides = ("llm-service",)

    def __init__(self) -> None:
        self.service = DemoLLM()

    def setup(self, ctx: Any) -> None:
        ctx.register_service("llm", self.service)

    def teardown(self) -> None:
        pass
