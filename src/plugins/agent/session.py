"""AgentSession: per-conversation state with thread-safe message history."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolSpec
from .workspace import FileWorkspace


@dataclass(slots=True)
class AgentSession:
    """A single agent conversation: its own history, tools, workspace and lock.

    Messages are stored as plain dicts compatible with the LLM layer's
    ``Message`` and ``ChatRequest.messages``.  The lock prevents concurrent
    ``run`` calls on the same session.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    """Unique session identifier (12 hex chars)."""

    name: str = "default"
    """Human-readable label shown in the session list."""

    system_prompt: str = ""
    """Effective system prompt used at the start of every ``run``."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Thread-safe message history (role/content/tool_calls/tool_call_id/…)."""

    workspace: FileWorkspace = field(default_factory=FileWorkspace)
    """Files the agent tools are allowed to read/write."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)
    """Tool name → ToolSpec mapping."""

    model: str = ""
    """LLM model name for this session."""

    max_steps: int = 12
    """Maximum REPL iterations before the run is cut short."""

    max_tool_output: int = 8000
    """Tool output longer than this is truncated before being sent to the LLM."""

    extra_instructions: tuple[str, ...] = ()
    """Per-session instructions appended to the system prompt on every run."""

    parent_id: str | None = None
    """Session id this session was forked from (or None)."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when the session was created."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    """Acquired during ``run``; prevents concurrent execution."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Arbitrary caller metadata (not used by the agent core)."""

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------
    def add_message(self, role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        """Append a message to the history (thread-safe).

        Returns the new message dict so callers can mutate it in-place
        (e.g. adding ``tool_calls`` after the assistant message is created).
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(kwargs)
        with self.lock:
            self.messages.append(msg)
        return msg

    def last_assistant(self) -> dict[str, Any] | None:
        """Return the most recent assistant message, or None."""
        with self.lock:
            for msg in reversed(self.messages):
                if msg.get("role") == "assistant":
                    return msg
        return None

    def clear_messages(self) -> None:
        """Drop all messages (used by ``reset``)."""
        with self.lock:
            self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        """Summary for the session list UI."""
        with self.lock:
            msg_count = len(self.messages)
            last_content = ""
            for m in reversed(self.messages):
                if m.get("role") == "assistant" and m.get("content"):
                    last_content = str(m["content"])[:120]
                    break
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "max_steps": self.max_steps,
            "message_count": msg_count,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "last_message": last_content,
        }