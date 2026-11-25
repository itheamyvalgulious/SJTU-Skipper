from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Mapping
from contextlib import AbstractContextManager

from core.agent_extensions import AgentExtension, AgentExtensionRegistry
from core.file_access import publication_lock
from core.agent_types import (
    IpcMethod,
    AgentEvent,
    AgentEventType,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    ExecutionStatus,
    FinishReason,
    RuntimeFrameType,
    AgentProfile,
    AgentRunResult,
    AgentRole,
    AgentSessionInfo,
    AgentTool,
    AgentToolName,
)
from core.plugin import Plugin, PluginContext
from .session import AgentSession
from core.workspace import FileWorkspace

_FILE_TOOLS = tuple(AgentToolName)


class AgentService:
    """Keep UI metadata while the SDK owns conversations and tools."""

    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._runtime = ctx.require("llm")
        self._sessions: dict[str, AgentSession] = {}
        self._agents: dict[str, AgentProfile] = {}
        self._lock = threading.RLock()
        self._extensions = AgentExtensionRegistry(self._runtime.list_roles)
        self._prompt_providers: dict[str, tuple[Any, tuple[str, ...], int]] = {}
        self._prompt_lock = threading.RLock()
        self._callbacks_ready = False

    def list_roles(self) -> list[dict[str, Any]]:
        return self._runtime.list_roles()

    def supports_callbacks(self) -> bool:
        """Whether the runtime can execute extension tools/hooks."""

        return callable(getattr(self._runtime, "set_callback_handler", None))

    def register_extension(self, owner: str, extension: AgentExtension) -> None:
        self._extensions.register(owner, extension)

    def unregister_extensions(self, owner: str) -> None:
        self._extensions.unregister(owner)

    def register_prompt_provider(
        self,
        owner: str,
        provider: Any,
        *,
        roles: tuple[str, ...] | list[str] = ("*",),
        priority: int = 0,
    ) -> None:
        """Register dynamic context injected before every agent turn."""

        if not owner or not callable(provider):
            raise ValueError("Prompt provider requires an owner and a callable")
        with self._prompt_lock:
            self._prompt_providers[owner] = (
                provider,
                tuple(str(role) for role in roles),
                int(priority),
            )

    def unregister_prompt_provider(self, owner: str) -> None:
        with self._prompt_lock:
            self._prompt_providers.pop(owner, None)

    def _prompt_context(self, session: AgentSession) -> str:
        with self._prompt_lock:
            entries = sorted(
                self._prompt_providers.values(),
                key=lambda item: item[2],
            )
        blocks: list[str] = []
        for provider, roles, _priority in entries:
            if "*" not in roles and session.role not in roles:
                continue
            try:
                text = str(provider(session.role, session.meta) or "").strip()
            except Exception:  # noqa: BLE001 - context providers are best effort
                self._ctx.logger.warning("prompt provider failed", exc_info=True)
                continue
            if text:
                blocks.append(text)
        return "\n\n".join(blocks)

    def _prepare_extensions(self, session: AgentSession) -> None:
        review = session.meta.get("review")
        reviewer_role = review["role"] if review else AgentRole.DEFAULT
        session.extensions, session.reviewer_extensions = self._extensions.snapshots(
            session.role, reviewer_role
        )
        unknown = set(session.tools) - set(_FILE_TOOLS) - set(session.extensions.tools)
        if unknown:
            raise ValueError(f"Undeclared tools: {', '.join(sorted(unknown))}")
        if len(set(session.tools)) != len(session.tools):
            raise ValueError("Session tool names must be unique")
        callbacks = session.extensions.needs_callbacks or (
            bool(session.meta.get("review"))
            and session.reviewer_extensions.needs_callbacks
        )
        if callbacks:
            with self._lock:
                if not self._callbacks_ready:
                    install = getattr(self._runtime, "set_callback_handler", None)
                    if not callable(install):
                        raise RuntimeError(
                            "Agent extensions require runtime callback support"
                        )
                    install(self._extension_callback)
                    self._callbacks_ready = True

    @staticmethod
    def _extension_payload(session: AgentSession) -> dict[str, Any]:
        payload = session.extensions.payload()
        payload["reviewer"] = session.reviewer_extensions.payload()
        return payload

    def _extension_callback(self, request: dict[str, Any]) -> Any:
        session_id = request["session_id"]
        role = request["role"]
        with self._lock:
            session = self._sessions.get(session_id)
            reviewer = False
            if session is None and session_id.endswith("-review"):
                session = self._sessions.get(session_id[:-7])
                reviewer = True
            if session is None:
                raise KeyError(f"Extension callback session is not live: {session_id}")
            snapshot = session.reviewer_extensions if reviewer else session.extensions
            if reviewer and not session.meta.get("review"):
                raise PermissionError("Session does not have a reviewer")
            if role != snapshot.role:
                raise PermissionError(f"Extension callback role does not match: {role}")
        arguments = request["arguments"]
        workspace = dict(arguments["workspace"])
        if set(workspace) - set(session.workspace.names):
            raise PermissionError(
                "Extension callback contains unauthorized workspace files"
            )
        context = {"session_id": session_id, "role": role, "workspace": workspace}
        name = request["name"]
        if request["kind"] == "tool":
            tool = snapshot.tools.get(name)
            if tool is None:
                raise KeyError(
                    f"Extension tool is not declared for role {role}: {name}"
                )
            context["parameters"] = arguments["parameters"]
            return tool.handler(context)
        if request["kind"] == "completion_hook":
            hook = snapshot.hooks.get(name)
            if hook is None:
                raise KeyError(
                    f"Completion hook is not declared for role {role}: {name}"
                )
            context["text"] = arguments["text"]
            context["tool_calls"] = arguments["tool_calls"]
            result = hook.handler(context)
            if result is not None and (
                not isinstance(result, dict)
                or set(result) != {"continue"}
                or not isinstance(result["continue"], str)
                or not result["continue"].strip()
            ):
                raise TypeError(
                    "Completion hook must return None or {'continue': non-empty str}"
                )
            return result
        raise ValueError(f"Unknown extension callback kind: {request['kind']}")

    def register_agent(self, profile: AgentProfile) -> None:
        if not profile.name:
            raise ValueError("Agent profile needs a name")
        with self._lock:
            self._agents[profile.name] = profile

    def create_session(
        self,
        name: str,
        *,
        agent: str | AgentProfile | None = None,
        role: str | None = None,
        system_prompt: str | None = None,
        files: Mapping[str, str | Path] | FileWorkspace | None = None,
        workspace: FileWorkspace | None = None,
        tools: list[str | AgentTool] | None = None,
        compaction: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> AgentSessionInfo:
        if isinstance(agent, str):
            with self._lock:
                profile = self._agents.get(agent)
            if profile is None:
                raise KeyError(f"Agent profile {agent!r} is not registered")
        else:
            profile = agent
        if workspace is not None:
            ws = workspace.copy()
        elif isinstance(files, FileWorkspace):
            ws = files.copy()
        else:
            ws = FileWorkspace(files)
        declarations = (
            tools
            if tools is not None
            else (profile.tools if profile else list(_FILE_TOOLS))
        )
        names = tuple(
            str(item.name if isinstance(item, AgentTool) else item)
            for item in declarations
        )
        prompt = (
            system_prompt
            if system_prompt is not None
            else (profile.system_prompt if profile else "")
        )
        session = AgentSession(
            name=name,
            agent=profile.name if profile else "",
            role=(
                role
                if role is not None
                else (profile.role if profile else AgentRole.DEFAULT)
            ),
            system_prompt=prompt,
            workspace=ws,
            tools=names,
            compaction=dict(compaction or {}),
            meta=dict(meta or {}),
        )
        self._prepare_extensions(session)
        with self._lock:
            self._sessions[session.id] = session
        try:
            self._create_runtime_session(session)
        except BaseException:
            with self._lock:
                self._sessions.pop(session.id, None)
            raise
        return session.to_info()

    def reopen(
        self,
        session_id: str,
        *,
        name: str,
        workspace: FileWorkspace,
        system_prompt: str,
        role: str,
        tools: list[str],
        compaction: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any],
    ) -> AgentSessionInfo:
        copied = workspace.copy()
        session = AgentSession(
            id=session_id,
            name=name,
            workspace=copied,
            system_prompt=system_prompt,
            role=role,
            tools=tuple(str(tool) for tool in tools),
            compaction=dict(compaction or {}),
            meta=dict(meta),
        )
        self._prepare_extensions(session)
        with self._lock:
            if session_id in self._sessions:
                raise RuntimeError("Agent session is still live")
            self._sessions[session_id] = session
        try:
            self._create_runtime_session(session, method=IpcMethod.REOPEN_SESSION)
        except BaseException:
            with self._lock:
                self._sessions.pop(session_id, None)
            raise
        return session.to_info()

    def compact(self, session_id: str) -> dict[str, Any]:
        """Ask the runtime to compact the live session context now."""

        if not session_id:
            return {"ok": False, "reason": "session is required"}
        if self._session(session_id) is None:
            return {"ok": False, "reason": "agent session is not live"}
        try:
            result = self._runtime.call(IpcMethod.COMPACT, session_id=session_id)
            return result if isinstance(result, dict) else {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc)}

    def message(self, session_id: str, message: str) -> dict[str, Any]:
        if not session_id or not message.strip():
            return {
                "accepted": False,
                "reason": "A session and a non-empty message are required",
            }
        session = self._session(session_id)
        if session is not None and session.cancelled.is_set():
            return {"accepted": False, "reason": "Agent session is closing"}
        if session is not None:
            context = self._prompt_context(session)
            if context:
                message = context + "\n\n" + message
        # Bun owns reviewer sessions and checks admission for every exact ID.
        try:
            result = self._runtime.call(
                IpcMethod.MESSAGE, session_id=session_id, message=message
            )
            if not isinstance(result, dict) or not isinstance(
                result.get("accepted"), bool
            ):
                raise RuntimeError("Agent runtime returned no message admission result")
            return result
        except Exception as exc:
            return {"accepted": False, "reason": str(exc)}

    def can_resume(self, session_id: str) -> tuple[bool, str]:
        """Report whether ``session_id`` still has a reopenable transcript.

        The runtime owns the transcript layout; the check stays behind the
        ``llm`` service instead of a consumer plugin reading runtime files.
        """

        if not session_id:
            return False, "agent session is unknown"
        with self._lock:
            live = session_id in self._sessions
        if live:
            return False, "agent session is still live"
        if self._runtime.session_transcript(session_id) is None:
            return False, "saved session transcript is unavailable"
        return True, ""

    def _create_runtime_session(
        self, session: AgentSession, *, method: IpcMethod = IpcMethod.CREATE_SESSION
    ) -> None:
        result = self._runtime.call(
            method,
            id=session.id,
            role=session.role,
            prompt=session.system_prompt,
            files={
                name: str(path) for name, path in session.workspace.to_dict().items()
            },
            readonly=list(session.workspace.readonly),
            tools=list(session.tools),
            compaction=dict(session.compaction),
            extensions=self._extension_payload(session),
            meta=session.meta,
        )
        session.model = str(result["model"])
        session.runtime_ready = True

    def get_session(self, session_id: str) -> AgentSessionInfo | None:
        """Return the immutable view other plugins are allowed to read."""

        session = self._session(session_id)
        return None if session is None else session.to_info()

    def session_workspace(self, session_id: str) -> FileWorkspace | None:
        """Return a copy of the authorized file map for a live session."""

        session = self._session(session_id)
        return None if session is None else session.workspace.copy()

    def _session(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _require_session(self, session_id: str) -> AgentSession:
        session = self._session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} not found")
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [session.to_dict() for session in self._sessions.values()]

    def cancel(self, session_id: str) -> None:
        session = self._require_session(session_id)
        session.cancelled.set()
        self._runtime.cancel(session_id)

    def close(self, session_id: str) -> None:
        session = self._session(session_id)
        if session is None:
            return
        try:
            self.cancel(session_id)
        finally:
            try:
                with session.lock:
                    self._runtime.call(IpcMethod.CLOSE_SESSION, session_id=session_id)
            finally:
                with self._lock:
                    self._sessions.pop(session_id, None)

    def run(
        self,
        session_id: str,
        user_input: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        write_guard: Callable[[Path], AbstractContextManager] | None = None,
        follow_ups: tuple[str, ...] | list[str] = (),
    ) -> AgentRunResult:
        session = self._require_session(session_id)
        if not session.lock.acquire(blocking=False):
            return AgentRunResult(
                session_id=session_id, error="Session is already running"
            )
        result = AgentRunResult(session_id=session_id)
        calls = 0
        try:
            if session.cancelled.is_set():
                raise RuntimeError("Agent run cancelled")
            if not session.runtime_ready:
                self._create_runtime_session(session)
            context = self._prompt_context(session)
            if context:
                user_input = context + "\n\n" + user_input
            session.step = 0
            self._emit(AgentEventType.RUN_STARTED, session, 0, on_event)
            completed = False
            for frame in self._runtime.stream(
                IpcMethod.RUN,
                session_id=session_id,
                instruction=user_input,
                **({"follow_ups": list(follow_ups)} if follow_ups else {}),
            ):
                if frame.get("type") == RuntimeFrameType.RESULT:
                    values = frame.get("result") or {}
                    result = AgentRunResult(
                        session_id=session_id,
                        final_message=str(values.get("final_message") or ""),
                        steps=int(values.get("steps") or 0),
                        error=values.get("error"),
                        total_tokens=int(values.get("total_tokens") or 0),
                        input_tokens=int(values.get("input_tokens") or 0),
                        output_tokens=int(values.get("output_tokens") or 0),
                        cache_read_tokens=int(values.get("cache_read_tokens") or 0),
                        cache_miss_tokens=int(values.get("cache_miss_tokens") or 0),
                        tool_calls_made=calls,
                        finish_reason=FinishReason(
                            values.get("finish_reason") or FinishReason.STOP
                        ),
                        review_passed=values.get("review_passed") is True,
                    )
                    session.step = result.steps
                    completed = True
                    continue
                if session.cancelled.is_set():
                    continue
                if frame.get("type") != RuntimeFrameType.EVENT:
                    continue
                name = AgentEventType(frame["name"])
                data = dict(frame.get("data") or {})
                if name == AgentEventType.EXECUTION:
                    self._decode_execution(data)
                elif name == AgentEventType.REVIEWER:
                    if "finished" in data:
                        if not isinstance(data["finished"], bool):
                            raise ValueError("Reviewer finished flag must be a boolean")
                    else:
                        data["event"] = AgentEventType(data["event"])
                        if data["event"] == AgentEventType.EXECUTION:
                            self._decode_execution(data)
                event_step = data.pop("step", None)
                step = 0 if event_step is None else int(event_step)
                if name != AgentEventType.REVIEWER and event_step is not None:
                    session.step = step
                data.pop("session_id", None)
                if name == AgentEventType.EXECUTION:
                    model = (data.get("metrics") or {}).get("model")
                    if model is not None:
                        session.model = str(model)
                elif name == AgentEventType.TOOL_CALL:
                    calls += 1
                elif name == AgentEventType.FILE:
                    logical = str(data.get("file") or "")
                    if logical not in session.workspace:
                        raise PermissionError(
                            f"Runtime changed an unauthorized file: {logical}"
                        )
                    path = session.workspace.path(logical)
                    guard = (
                        write_guard(path)
                        if write_guard
                        else publication_lock(path.parent)
                    )
                    with guard:
                        if (
                            "previous" in data
                            and path.read_text(encoding="utf-8") != data["previous"]
                        ):
                            raise RuntimeError(
                                f"Document changed outside this session: {logical}"
                            )
                        session.workspace.write(logical, str(data["content"]))
                self._emit(name, session, step, on_event, **data)
                if name == AgentEventType.FILE:
                    self._ctx.ui.refresh(f"agent.file:{data.get('path', '')}")
            if not completed:
                raise RuntimeError("Agent runtime ended without a result")
            if session.cancelled.is_set():
                result.error = "Agent run cancelled"
        except Exception as exc:
            try:
                self._runtime.cancel(session_id)
            except Exception:
                pass  # Preserve the run failure if cancellation also fails.
            session.runtime_ready = False
            result.error = str(exc)
        finally:
            try:
                if result.error:
                    self._emit(
                        AgentEventType.ERROR,
                        session,
                        result.steps,
                        on_event,
                        error=result.error,
                    )
                self._emit(
                    AgentEventType.RUN_FINISHED,
                    session,
                    result.steps,
                    on_event,
                    content=result.final_message,
                    error=result.error,
                )
            finally:
                session.lock.release()
        return result

    @staticmethod
    def _decode_execution(data):
        metrics = data.get("metrics") or {}
        if "status" in metrics:
            metrics["status"] = ExecutionStatus(metrics["status"])
        if block := data.get("block"):
            block["kind"] = ExecutionBlockKind(block["kind"])
            block["status"] = ExecutionBlockStatus(block["status"])

    def _emit(self, event_type, session, step, on_event, **data):
        self._ctx.events.emit(event_type, session_id=session.id, step=step, **data)
        if on_event is not None:
            on_event(
                AgentEvent(type=event_type, session_id=session.id, step=step, data=data)
            )


class AgentPlugin(Plugin):
    id = "agent"
    requires = ("llm",)
    provides = ("agent",)

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.service: AgentService | None = None

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.service = AgentService(ctx)
        ctx.register_service("agent", self.service)
        ctx.ui.register_action("agent.message", self._message, plugin_id=self.id)

        ctx.meta.register(
            "agent",
            lambda: {
                "sessions": len(self.service.list_sessions()),
            },
            plugin_id=self.id,
        )

        for evt_name, evt_desc in [
            (AgentEventType.RUN_STARTED, "Agent run started"),
            (AgentEventType.RUN_READY, "Agent run accepts messages"),
            (AgentEventType.DELTA, "Streaming text delta"),
            (AgentEventType.REASONING, "Streaming reasoning delta"),
            (AgentEventType.MESSAGE, "Assistant message received"),
            (AgentEventType.TOOL_CALL, "Tool call issued"),
            (AgentEventType.TOOL_RESULT, "Tool result received"),
            (AgentEventType.FILE, "Authorized file changed"),
            (AgentEventType.RUN_FINISHED, "Agent run finished"),
            (AgentEventType.ERROR, "Agent error"),
        ]:
            ctx.events.define(evt_name, evt_desc)

        ctx.logger.info("agent plugin ready; %s sessions", 0)

    def _message(self, payload):
        result = self.service.message(
            str(payload.get("agent_session_id") or ""),
            str(payload.get("message") or ""),
        )
        return {
            "ok": result["accepted"],
            "message": result.get("reason", ""),
            "refresh": False,
        }

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is not None:
            ctx.ui.unregister_action("agent.message")
        if self.service is not None:
            for item in self.service.list_sessions():
                try:
                    self.service.close(item["id"])
                except Exception:
                    if ctx is not None:
                        ctx.logger.exception(
                            "Failed to close agent session %s", item["id"]
                        )
        self.service = None
