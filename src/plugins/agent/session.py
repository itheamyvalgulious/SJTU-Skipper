from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.agent_extensions import ExtensionSnapshot
from core.agent_types import AgentRole, AgentSessionInfo
from core.workspace import FileWorkspace


@dataclass(slots=True)
class AgentSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "default"
    role: str = AgentRole.DEFAULT
    agent: str = ""
    system_prompt: str = ""
    workspace: FileWorkspace = field(default_factory=FileWorkspace)
    tools: tuple[str, ...] = ()
    extensions: ExtensionSnapshot | None = None
    reviewer_extensions: ExtensionSnapshot | None = None
    model: str = ""
    step: int = 0
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: threading.Event = field(default_factory=threading.Event)
    runtime_ready: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    compaction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "created_at": self.created_at,
        }

    def to_info(self) -> AgentSessionInfo:
        """Return the read-only view handed to other plugins."""

        return AgentSessionInfo(
            id=self.id,
            name=self.name,
            role=self.role,
            model=self.model,
            step=self.step,
            created_at=self.created_at,
        )
