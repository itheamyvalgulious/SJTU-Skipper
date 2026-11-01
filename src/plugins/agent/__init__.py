"""LLM Agent plugin: tool-use REPL loop, file workspace, multi-session.

Public API (frozen contract used by summarise and other plugins):

- ``AgentPlugin``: id="llm-agent", requires=("settings","llm")
- ``AgentService``: the ``agent`` service
- ``AgentSession``: isolated conversation
- ``AgentEvent`` / ``AgentRunResult``: run observability
- ``FileWorkspace``: logical-name → path sandbox
- ``ToolSpec``: tool definition
- ``DEFAULT_SYSTEM_PROMPT``: the default Chinese system prompt
"""

from __future__ import annotations

from .plugin import AgentEvent, AgentPlugin, AgentRunResult, AgentService
from .prompts import DEFAULT_SYSTEM_PROMPT
from .session import AgentSession
from .tools import ToolSpec
from .workspace import FileWorkspace

__all__ = [
    "AgentEvent",
    "AgentPlugin",
    "AgentRunResult",
    "AgentService",
    "AgentSession",
    "DEFAULT_SYSTEM_PROMPT",
    "FileWorkspace",
    "ToolSpec",
]