"""Application declarations and events for SDK agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentRole(StrEnum):
    DEFAULT = "default"
    COMPACTION = "compaction"


class ExecutionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionBlockStatus(StrEnum):
    STREAMING = "streaming"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionBlockKind(StrEnum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL = "tool"
    REVIEW = "review"
    USER = "user"


class ExecutionRole(StrEnum):
    WRITER = "writer"
    QUESTION_ANSWER = "question_answer"
    REVIEWER = "reviewer"
    CHUNK_WRITER = "chunk_writer"


class AgentEventType(StrEnum):
    RUN_STARTED = "agent.run.started"
    RUN_READY = "agent.run.ready"
    RUN_FINISHED = "agent.run.finished"
    DELTA = "agent.delta"
    REASONING = "agent.reasoning"
    MESSAGE = "agent.message"
    TOOL_CALL = "agent.tool_call"
    TOOL_RESULT = "agent.tool_result"
    FILE = "agent.file"
    EXECUTION = "agent.execution"
    REVIEWER = "agent.reviewer"
    ERROR = "agent.error"


class IpcMethod(StrEnum):
    PING = "ping"
    INFO = "info"
    CONFIGURE = "configure"
    CATALOG = "catalog"
    DISCOVER = "discover"
    MODELS_SET = "models_set"
    PROVIDER_SET = "provider_set"
    PROVIDER_REMOVE = "provider_remove"
    ROLE_SET = "role_set"
    ROLES_SYNC = "roles_sync"
    GENERATE = "generate"
    CREATE_SESSION = "create_session"
    REOPEN_SESSION = "reopen_session"
    SESSION_TRANSCRIPT = "session_transcript"
    RUN = "run"
    MESSAGE = "message"
    CANCEL = "cancel"
    CLOSE_SESSION = "close_session"
    LOGOUT = "logout"
    CREDENTIALS = "credentials"
    IMPORT_CREDENTIALS = "import_credentials"


class RuntimeFrameType(StrEnum):
    EVENT = "event"
    RESULT = "result"
    ERROR = "error"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "toolUse"
    ERROR = "error"
    ABORTED = "aborted"


class GenerationErrorCode(StrEnum):
    THINKING_LOOP = "thinking_loop"
    TOKEN_LIMIT = "token_limit"


class AgentToolName(StrEnum):
    READ = "read"
    GREP = "grep"
    GLOB = "glob"
    EDIT = "edit"
    WRITE = "write"


@dataclass(slots=True)
class AgentTool:
    name: str


@dataclass(slots=True)
class AgentProfile:
    name: str
    role: str = AgentRole.DEFAULT
    system_prompt: str = ""
    description: str = ""
    tools: list[str | AgentTool] = field(default_factory=list)


@dataclass(slots=True)
class AgentEvent:
    type: AgentEventType
    session_id: str
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentSessionInfo:
    """Immutable snapshot of one agent session, safe across plugin boundaries.

    The live session object stays inside the ``agent`` plugin. Consumers
    read this snapshot instead of the mutable session state.
    """

    id: str
    name: str = ""
    role: str = AgentRole.DEFAULT
    model: str = ""
    step: int = 0
    created_at: float = 0.0


@dataclass(slots=True)
class AgentRunResult:
    session_id: str
    final_message: str = ""
    steps: int = 0
    tool_calls_made: int = 0
    finish_reason: FinishReason = FinishReason.STOP
    error: str | None = None
    total_tokens: int = 0
    review_passed: bool = False


__all__ = [
    "AgentEvent",
    "AgentEventType",
    "AgentProfile",
    "AgentRole",
    "AgentRunResult",
    "AgentSessionInfo",
    "AgentTool",
    "AgentToolName",
    "ExecutionStatus",
    "ExecutionBlockStatus",
    "ExecutionBlockKind",
    "ExecutionRole",
    "FinishReason",
    "GenerationErrorCode",
    "IpcMethod",
    "RuntimeFrameType",
]
