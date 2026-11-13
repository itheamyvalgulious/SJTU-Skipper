"""LLM Agent plugin: tool-use REPL loop, profiles, workspaces, sessions.

Public API (the contract used by summarize / instant-summarize / ask):

- ``AgentPlugin``: id="llm-agent", requires=("settings","llm")
- ``AgentService``: the ``agent`` service — register_agent(profile),
  create_session(agent=...), run(...), context and session access
- ``AgentProfile`` / ``AgentTool``: an agent declaration with tools + params
- ``AgentSession``: isolated conversation
- ``AgentEvent`` / ``AgentRunResult``: run observability
- ``FileWorkspace``: logical-name → path sandbox
- ``ToolSpec``: tool definition
- ``DEFAULT_SYSTEM_PROMPT``: the default Chinese system prompt
"""

from __future__ import annotations

from .plugin import (
    AgentEvent,
    AgentPlugin,
    AgentProfile,
    AgentRunResult,
    AgentService,
    AgentTool,
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .session import AgentSession
from .tools import ToolContext, ToolSpec
from .workspace import FileWorkspace

__all__ = [
    "AgentEvent",
    "AgentPlugin",
    "AgentProfile",
    "AgentRunResult",
    "AgentService",
    "AgentSession",
    "AgentTool",
    "DEFAULT_SYSTEM_PROMPT",
    "FileWorkspace",
    "ToolContext",
    "ToolSpec",
]
