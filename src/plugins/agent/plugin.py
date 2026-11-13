"""AgentPlugin: ``llm-agent`` plugin that registers the ``agent`` service.

This is the largest behaviour plugin in the system.  It provides:

- ``AgentService`` – session management + synchronous / async REPL run loop.
- ``AgentSession`` – per-conversation state, lock, message history.
- ``FileWorkspace`` – sandboxed file I/O.
- Four default tools (info / read / replace / append).
- Extension points: custom tools, prompt contributors, per-session instructions.
- UI: right-side tab "Agent", chat page, actions.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from core.plugin import Plugin, PluginContext
from .prompts import DEFAULT_SYSTEM_PROMPT
from .session import AgentSession
from .tools import ToolContext, ToolSpec, build_tools
from .workspace import FileWorkspace

# ---------------------------------------------------------------------------
# Run result types
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_DECLARATIONS = ("info", "read", "replace", "append")


@dataclass(slots=True)
class AgentEvent:
    """An observable event emitted during a run loop (passed to *on_event*)."""

    type: str
    """One of ``run.started``, ``message``, ``tool_call``, ``tool_result``,
    ``run.finished``, ``error``."""

    session_id: str
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    """Returned by ``AgentService.run`` after the REPL loop finishes."""

    session_id: str
    steps: int = 0
    final_message: str = ""
    total_tokens: int = 0
    error: str | None = None


@dataclass(slots=True)
class AgentTool:
    """One tool a behavior plugin wants, with its parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentProfile:
    """A named agent configuration registered by a behavior plugin.

    The plugin declares the tools (and their parameters), prompt, model and
    limits; the agent plugin owns the loop, the tools themselves and the
    context.  ``None`` means "fall back to the service default".
    """

    name: str
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    model: str | None = None
    max_steps: int | None = None
    max_tool_output: int | None = None
    max_history_chars: int | None = None
    max_tokens: int | None = None
    llm_extra: dict[str, Any] = field(default_factory=dict)
    final_answer_on_limit: bool = False
    stream: bool = False


# ---------------------------------------------------------------------------
# Tool-call normalisation
# ---------------------------------------------------------------------------


def _normalize_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool-calls from a response that may be a dataclass,
    :class:`~types.SimpleNamespace`, or plain dict.

    Handles both flat ``{id, name, arguments}`` and the OpenAI nesting
    ``{id, function: {name, arguments}}``.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls is None:
        tool_calls = []
    if isinstance(tool_calls, dict):
        tool_calls = tool_calls.get("tool_calls", [])

    result: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_id: str = ""
        tc_name: str = ""
        tc_args: Any = {}

        if hasattr(tc, "id"):
            tc_id = str(getattr(tc, "id", ""))
            tc_name = str(getattr(tc, "name", ""))
            tc_args = getattr(tc, "arguments", {})
        elif isinstance(tc, dict):
            tc_id = str(tc.get("id", ""))
            func = tc.get("function")
            if isinstance(func, dict):
                tc_name = str(func.get("name", ""))
                tc_args = func.get("arguments", {})
            else:
                tc_name = str(tc.get("name", ""))
                tc_args = tc.get("arguments", {})
        else:
            continue

        # Arguments may be JSON-encoded, and some models double- or triple-
        # encode them ({"arguments": "\"<json>\""}).  Unwrap until a dict is
        # reached; the loop is bounded so malformed input cannot spin.
        for _ in range(6):
            if isinstance(tc_args, str):
                try:
                    decoded = json.loads(tc_args)
                except (json.JSONDecodeError, TypeError):
                    break
                tc_args = decoded
                continue
            if (
                isinstance(tc_args, dict)
                and len(tc_args) == 1
                and "arguments" in tc_args
            ):
                nested = tc_args["arguments"]
                if isinstance(nested, (str, dict)):
                    tc_args = nested
                    continue
            break

        result.append(
            {
                "id": tc_id,
                "name": tc_name,
                "arguments": dict(tc_args) if isinstance(tc_args, dict) else {},
            }
        )
    return result


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------


class AgentService:
    """The ``agent`` service registered by the llm-agent plugin.

    Public methods form the frozen contract consumed by the summarise plugin
    and other behaviour plugins:

    - ``create_session`` / ``create_subsession``
    - ``get_session`` / ``list_sessions`` / ``reset`` / ``close``
    - ``run`` / ``run_async``
    - ``register_tool`` / ``unregister_tool``
    - ``add_prompt_contributor`` / ``set_system_prompt`` / ``add_instruction``
    """

    def __init__(
        self,
        ctx: PluginContext,
        default_model: str,
        default_max_steps: int,
        default_max_tool_output: int,
        default_system_prompt: str,
        default_max_history_chars: int = 0,
        settings_getter: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self._ctx = ctx
        self._default_model = default_model
        self._default_max_steps = default_max_steps
        self._default_max_tool_output = default_max_tool_output
        self._default_max_history_chars = default_max_history_chars
        self._default_system_prompt = default_system_prompt
        self._settings_getter = settings_getter
        self._sessions: dict[str, AgentSession] = {}
        self._agents: dict[str, AgentProfile] = {}
        self._custom_tools: dict[str, ToolSpec] = {}
        self._prompt_contributors: list[Callable[[AgentSession], str]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def _setting(self, key: str, fallback: Any) -> Any:
        """Read a live setting; fall back to the setup-time value."""

        if self._settings_getter is None:
            return fallback
        try:
            return self._settings_getter(key, fallback)
        except Exception:  # noqa: BLE001 - settings must not break sessions
            return fallback

    def create_session(
        self,
        name: str,
        *,
        agent: str | AgentProfile | None = None,
        system_prompt: str | None = None,
        files: Mapping[str, str | Path] | FileWorkspace | None = None,
        workspace: FileWorkspace | None = None,
        tools: (
            list[str]
            | list[AgentTool]
            | list[tuple[str, Mapping[str, Any]]]
            | list[ToolSpec]
            | None
        ) = None,
        model: str | None = None,
        max_steps: int | None = None,
        max_tool_output: int | None = None,
        max_history_chars: int | None = None,
        max_tokens: int | None = None,
        final_answer_on_limit: bool | None = None,
        stream: bool | None = None,
        llm_extra: Mapping[str, Any] | None = None,
        extra_instructions: Iterable[str] = (),
        parent_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> AgentSession:
        """Create and register a new agent session.

        Parameters:
            name: Human-readable label.
            agent: Registered profile name (or a profile object).  The profile
                supplies prompt, tools (with their parameters), model, limits
                and streaming; explicit keyword arguments override it.
            system_prompt: Overrides the profile/default prompt.
            files: Logical-name → path mapping, or a pre-built FileWorkspace.
            workspace: Alias for *files* (takes precedence when both are given).
            tools: Explicit tool declarations (names, ``(name, params)``,
                ``AgentTool`` or ``ToolSpec``); ``None`` inherits the profile
                or the four built-in defaults.
            model: LLM model; ``None`` uses the profile/plugin default.
            max_steps: REPL iteration cap for this session.
            final_answer_on_limit: When True, the last allowed iteration runs
                without tools and asks the model for a final answer.
            stream: Use the provider's streaming API for runs.
            max_tool_output: Per-tool-call output cap in characters.
            max_history_chars: Cap on tool-result characters kept in the LLM
                payload; older results are spilled to hidden files.
            extra_instructions: Per-session lines appended after the system prompt.
            parent_id: Session to fork from (for create_subsession).
            meta: Arbitrary caller metadata.
        """
        profile = self._resolve_agent(agent)
        sid = _new_session_id()

        # --- workspace ---
        if workspace is not None:
            ws = workspace
        elif isinstance(files, FileWorkspace):
            ws = files
        elif files is not None:
            ws = FileWorkspace(files)
        else:
            ws = FileWorkspace()

        prompt = (
            system_prompt
            if system_prompt is not None
            else (profile.system_prompt if profile else "")
        )
        if not prompt:
            prompt = self._setting("agent.system_prompt", self._default_system_prompt)

        extra = dict(profile.llm_extra) if profile else {}
        extra.update(llm_extra or {})

        session = AgentSession(
            id=sid,
            name=name,
            agent=profile.name if profile else "",
            system_prompt=prompt,
            workspace=ws,
            tools={},
            model=(
                model
                if model is not None
                else (
                    profile.model
                    if profile and profile.model is not None
                    else self._setting("agent.model", self._default_model)
                )
            ),
            max_steps=(
                max_steps
                if max_steps is not None
                else (
                    profile.max_steps
                    if profile and profile.max_steps is not None
                    else self._setting("agent.max_steps", self._default_max_steps)
                )
            ),
            final_answer_on_limit=bool(
                final_answer_on_limit
                if final_answer_on_limit is not None
                else (profile.final_answer_on_limit if profile else False)
            ),
            stream=bool(
                stream
                if stream is not None
                else (profile.stream if profile else False)
            ),
            max_tool_output=(
                max_tool_output
                if max_tool_output is not None
                else (
                    profile.max_tool_output
                    if profile and profile.max_tool_output is not None
                    else self._setting(
                        "agent.max_tool_output", self._default_max_tool_output
                    )
                )
            ),
            max_history_chars=(
                max(0, int(max_history_chars))
                if max_history_chars is not None
                else max(
                    0,
                    int(
                        (
                            profile.max_history_chars
                            if profile and profile.max_history_chars is not None
                            else self._setting(
                                "agent.max_history_chars",
                                self._default_max_history_chars,
                            )
                        )
                        or 0
                    ),
                )
            ),
            max_tokens=max(
                0,
                int(
                    max_tokens
                    if max_tokens is not None
                    else ((profile.max_tokens if profile else None) or 0)
                ),
            ),
            llm_extra=extra,
            extra_instructions=tuple(extra_instructions),
            parent_id=parent_id,
            meta=dict(meta or {}),
        )

        # --- tools ---
        # ``None`` => built-in defaults; a profile with an empty tool list
        # really means "no tools" (the instant ask agent).
        declarations = (
            tools
            if tools is not None
            else (list(profile.tools) if profile else None)
        )
        session.tools = self._resolve_tools(declarations, session)

        with self._lock:
            self._sessions[sid] = session
        return session

    def _resolve_agent(
        self, agent: str | AgentProfile | None
    ) -> AgentProfile | None:
        if agent is None:
            return None
        if isinstance(agent, AgentProfile):
            return agent
        profile = self.get_agent(str(agent))
        if profile is None:
            raise KeyError(f"agent profile {agent!r} is not registered")
        return profile

    def create_subsession(
        self,
        parent_id: str,
        name: str,
        **kwargs: Any,
    ) -> AgentSession:
        """Fork *parent_id* inheriting its model/tools/files/prompt.

        Any keyword argument overrides the inherited value.
        """
        parent = self._require_session(parent_id)
        inherit = {
            "system_prompt": parent.system_prompt,
            "workspace": parent.workspace.copy(),
            "tools": list(parent.tools.values()),
            "model": parent.model,
            "max_steps": parent.max_steps,
            "max_tokens": parent.max_tokens,
            "final_answer_on_limit": parent.final_answer_on_limit,
            "llm_extra": dict(parent.llm_extra),
            "extra_instructions": parent.extra_instructions,
            "meta": dict(parent.meta),
        }
        inherit.update(kwargs)
        inherit["parent_id"] = parent_id
        return self.create_session(name, **inherit)

    def get_session(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def reset(self, session_id: str) -> None:
        """Drop all messages from a session (does not close it)."""
        session = self._require_session(session_id)
        session.clear_messages()

    def close(self, session_id: str) -> None:
        """Remove the session entirely."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def _require_session(self, session_id: str) -> AgentSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"session {session_id!r} not found")
        return session

    # ------------------------------------------------------------------
    # Agent profile registry
    # ------------------------------------------------------------------
    def register_agent(self, profile: AgentProfile) -> None:
        """Register (or replace) an agent profile by name."""

        if not profile.name:
            raise ValueError("agent profile needs a name")
        with self._lock:
            self._agents[profile.name] = profile

    def unregister_agent(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)

    def get_agent(self, name: str) -> AgentProfile | None:
        with self._lock:
            return self._agents.get(name)

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------
    def _resolve_tools(
        self,
        tools: (
            list[str]
            | list[AgentTool]
            | list[tuple[str, Mapping[str, Any]]]
            | list[ToolSpec]
            | None
        ),
        session: AgentSession,
    ) -> dict[str, ToolSpec]:
        context = ToolContext(
            workspace=session.workspace,
            session_getter=lambda: session,
            meta_getter=self._ctx.meta.render,
            max_tool_output=session.max_tool_output,
        )
        declarations: list[Any]
        if tools is None:
            declarations = [AgentTool(name) for name in _DEFAULT_TOOL_DECLARATIONS]
        else:
            declarations = list(tools)

        prepared = [
            (item.name, item.params) if isinstance(item, AgentTool) else item
            for item in declarations
        ]
        result = build_tools(prepared, context)

        # Custom tools are available to every session; explicit/registered
        # built-ins win when names collide.
        for name, spec in self._custom_tools.items():
            result.setdefault(name, spec)
        return result

    # ------------------------------------------------------------------
    # Tool extension
    # ------------------------------------------------------------------
    def register_tool(self, spec: ToolSpec) -> None:
        """Register a custom tool available to all (future) sessions."""
        with self._lock:
            self._custom_tools[spec.name] = spec

    def unregister_tool(self, name: str) -> None:
        """Remove a previously registered custom tool."""
        with self._lock:
            self._custom_tools.pop(name, None)

    # ------------------------------------------------------------------
    # Prompt extension
    # ------------------------------------------------------------------
    def add_prompt_contributor(self, fn: Callable[[AgentSession], str]) -> None:
        """Register a function that contributes text to every session's system
        prompt at the start of each ``run``.  Called with the session object."""
        with self._lock:
            self._prompt_contributors.append(fn)

    def set_system_prompt(self, session_id: str, text: str) -> None:
        """Replace the system prompt for *session_id*."""
        session = self._require_session(session_id)
        session.system_prompt = text

    def add_instruction(self, session_id: str, text: str) -> None:
        """Append an extra instruction to *session_id*."""
        session = self._require_session(session_id)
        session.extra_instructions = (*session.extra_instructions, text)

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------
    def _payload_messages(self, session: AgentSession) -> list[dict[str, Any]]:
        """History for the next LLM call, with oversized tool results trimmed.

        A whole-lecture summarizer reads the same material over many steps; if
        every ``read`` result stayed in the conversation, the context would grow
        past the model window.  Results beyond ``max_history_chars`` are spilled
        to hidden workspace files (the agent can ``read`` them again by name)
        and replaced with a short pointer.

        Cache note: prefix caching matches from the start of the prompt, so
        changing any message invalidates everything after it.  A sliding spill
        window rewrites the oldest kept result on every step, which re-bills the
        whole tail each time.  Keep ``max_history_chars`` at ``0`` (default)
        unless the context window genuinely requires trimming.
        """

        budget = int(getattr(session, "max_history_chars", 0) or 0)
        with session.lock:
            messages = [dict(message) for message in session.messages]
            originals = list(session.messages)

        if budget <= 0:
            return messages

        kept = 0
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            size = len(content)
            if kept + size <= budget:
                kept += size
                continue
            name = originals[index].get("spilled_as")
            if not name:
                try:
                    name = session.spill_tool_output(content)
                except Exception:  # noqa: BLE001 - fall back to a plain stub
                    name = ""
                else:
                    with session.lock:
                        originals[index]["spilled_as"] = name
            if name:
                message["content"] = (
                    f"[工具输出已转存为隐藏文件 {name}（{size} 字符）；"
                    f"需要原文时用 read 读取该文件]"
                )
            else:
                message["content"] = f"[已省略 {size} 字符的工具输出]"
        return messages

    def _build_system_prompt(self, session: AgentSession) -> str:
        """Assemble the full system prompt for a run."""
        parts: list[str] = [session.system_prompt]
        with self._lock:
            contributors = list(self._prompt_contributors)
        for contributor in contributors:
            try:
                part = contributor(session)
                if part:
                    parts.append(part)
            except Exception:
                self._ctx.logger.exception("prompt contributor failed")
        parts.extend(session.extra_instructions)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(
        self,
        session_id: str,
        user_input: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        stream: bool | None = None,
    ) -> AgentRunResult:
        """Execute one synchronous REPL loop.

        Acquires the session lock; returns immediately with an error if the
        session is already running.  *stream* overrides the session default;
        when enabled the run emits ``agent.delta`` / ``agent.reasoning``
        events while the provider streams.
        """
        session = self._require_session(session_id)
        acquired = session.lock.acquire(blocking=False)
        if not acquired:
            return AgentRunResult(
                session_id=session_id,
                error="session is already running",
            )
        try:
            return self._run_locked(
                session,
                user_input,
                on_event,
                session.stream if stream is None else stream,
            )
        finally:
            session.lock.release()

    def run_async(
        self,
        session_id: str,
        user_input: str,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        stream: bool | None = None,
    ) -> threading.Thread:
        """Execute one REPL loop on a background thread.

        Returns the thread (already started; daemon).  The caller may
        ``join()`` or ignore.
        """
        thread = threading.Thread(
            target=self.run,
            args=(session_id, user_input),
            kwargs={"on_event": on_event, "stream": stream},
            daemon=True,
            name=f"agent-run-{session_id[:8]}",
        )
        thread.start()
        return thread

    def _run_locked(
        self,
        session: AgentSession,
        user_input: str,
        on_event: Callable[[AgentEvent], None] | None,
        use_stream: bool,
    ) -> AgentRunResult:
        # --- prepare ---
        system = self._build_system_prompt(session)
        session.add_message("user", user_input)
        self._emit("agent.run.started", session, 0, on_event)

        step = 0
        total_tokens = 0
        final_message = ""

        tool_schemas = [t.openai_schema() for t in session.tools.values()]

        # --- REPL ---
        while session.max_steps <= 0 or step < session.max_steps:
            step += 1
            session.meta["step"] = step

            # Last allowed step: take the tools away and ask for the answer.
            # Otherwise a model that keeps reading files burns the whole budget
            # and the run dies with "max_steps exceeded" even though it already
            # has enough material to answer.  Opt-in: writing agents keep their
            # tools on the last step.
            final_step = (
                bool(session.max_steps)
                and session.final_answer_on_limit
                and step >= session.max_steps
            )

            # Build messages for the LLM (old tool results are trimmed).
            llm_messages: list[dict[str, Any]] = []
            if system:
                llm_messages.append({"role": "system", "content": system})
            llm_messages.extend(self._payload_messages(session))
            if final_step and tool_schemas:
                llm_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "工具调用次数已经用完。现在不要再调用工具，"
                            "直接根据已有信息给出最终回答。"
                        ),
                    }
                )

            # --- LLM call ---
            try:
                response, step_reasoning = self._call_llm(
                    session,
                    llm_messages,
                    None if final_step else (tool_schemas or None),
                    step,
                    on_event,
                    use_stream,
                )
            except Exception as exc:
                self._emit(
                    "agent.error",
                    session,
                    step,
                    on_event,
                    error=str(exc),
                )
                return AgentRunResult(
                    session_id=session.id,
                    steps=step,
                    final_message=f"LLM error: {exc}",
                    total_tokens=total_tokens,
                    error=str(exc),
                )

            # --- usage ---
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_tokens += getattr(usage, "total_tokens", 0)

            # --- content & tool calls ---
            content: str = str(getattr(response, "content", "") or "")
            tool_calls = _normalize_tool_calls(response)

            # Build assistant message dict
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(
                                tc["arguments"], ensure_ascii=False
                            ),
                        },
                    }
                    for tc in tool_calls
                ]
            msg = session.add_message("assistant", content)
            # Mutate in-place so tool_calls land in the stored message
            with session.lock:
                msg.update(assistant_msg)

            self._emit(
                "agent.message",
                session,
                step,
                on_event,
                content=content,
                reasoning=step_reasoning or _response_reasoning(response),
            )

            if not tool_calls:
                final_message = content
                break

            # --- execute tool calls ---
            for tc in tool_calls:
                tc_name = tc["name"]
                tc_id = tc["id"]
                tc_args = tc["arguments"]

                self._emit(
                    "agent.tool_call",
                    session,
                    step,
                    on_event,
                    name=tc_name,
                    arguments=tc_args,
                )

                tool = session.tools.get(tc_name)
                if tool is None:
                    result = f"错误：未知工具 {tc_name!r}"
                else:
                    required = list((tool.parameters or {}).get("required") or [])
                    missing = [name for name in required if name not in tc_args]
                    if missing:
                        result = (
                            f"工具参数缺失：{', '.join(missing)}。"
                            f"请重新调用 {tc_name} 并提供完整参数。"
                        )
                    else:
                        try:
                            result = tool.handler(**tc_args)
                        except Exception as exc:
                            self._ctx.logger.exception(
                                "tool %s failed in session %s", tc_name, session.id
                            )
                            result = f"工具执行错误：{exc}"

                # Truncate
                max_out = session.max_tool_output
                if len(result) > max_out:
                    result = (
                        result[:max_out]
                        + f"\n... (输出已截断，原始长度 {len(result)} 字符)"
                    )

                session.add_message(
                    "tool",
                    result,
                    tool_call_id=tc_id,
                    name=tc_name,
                )

                self._emit(
                    "agent.tool_result",
                    session,
                    step,
                    on_event,
                    name=tc_name,
                    result=result[:500],
                )

                # The notes reader renders files from disk; refresh the web UI
                # as soon as a tool actually writes so new notes appear live.
                if tc_name in {"write", "append", "replace"} and not str(
                    result
                ).startswith("错误"):
                    try:
                        self._ctx.ui.refresh(f"agent.file:{tc_name}")
                    except Exception:  # noqa: BLE001 - refresh is best effort
                        self._ctx.logger.debug("ui refresh after %s failed", tc_name)
        else:
            # max_steps exceeded
            self._emit(
                "agent.error",
                session,
                step,
                on_event,
                error="max_steps exceeded",
            )
            return AgentRunResult(
                session_id=session.id,
                steps=step,
                final_message=final_message or "(达到最大步数限制)",
                total_tokens=total_tokens,
                error="max_steps exceeded",
            )

        self._emit(
            "agent.run.finished",
            session,
            step,
            on_event,
            content=final_message,
        )
        return AgentRunResult(
            session_id=session.id,
            steps=step,
            final_message=final_message,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _call_llm(
        self,
        session: AgentSession,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        step: int,
        on_event: Callable[[AgentEvent], None] | None,
        use_stream: bool,
    ) -> tuple[Any, str]:
        """One LLM round trip; returns ``(response, reasoning_text)``.

        Streaming forwards every text/reasoning chunk as an event so callers
        can display progress; the final response still carries the tool calls.
        """

        kwargs: dict[str, Any] = {
            "profile": "agent",
            "messages": messages,
            "tools": tools,
            "model": session.model,
            "temperature": 0.2,
            "max_tokens": session.max_tokens or None,
            "extra": dict(session.llm_extra) or None,
            # Conversation affinity: one id per session, so independent runs do
            # not thrash each other's prompt cache at the gateway.
            "session_id": session.id,
            "parent_session_id": session.parent_id,
        }
        if not use_stream:
            response = self._ctx.llm.chat(**kwargs)
            return response, _response_reasoning(response)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        response = None
        for event in self._ctx.llm.stream(**kwargs):
            if event.type == "text":
                content_parts.append(event.text)
                self._emit(
                    "agent.delta",
                    session,
                    step,
                    on_event,
                    text=event.text,
                    content="".join(content_parts),
                )
            elif event.type == "reasoning":
                reasoning_parts.append(event.text)
                self._emit(
                    "agent.reasoning",
                    session,
                    step,
                    on_event,
                    text=event.text,
                    content="".join(reasoning_parts),
                )
            elif event.type == "done":
                response = event.response
            elif event.type == "error":
                raise event.error or RuntimeError("LLM stream failed")
        if response is None:
            raise RuntimeError("LLM stream ended without a response")
        return response, "".join(reasoning_parts) or _response_reasoning(response)
    def _emit(
        self,
        event_type: str,
        session: AgentSession,
        step: int,
        on_event: Callable[[AgentEvent], None] | None,
        **data: Any,
    ) -> None:
        """Send a bus event and call *on_event*."""
        try:
            self._ctx.events.emit(
                event_type,
                session_id=session.id,
                step=step,
                **data,
            )
        except Exception:
            pass
        if on_event is not None:
            try:
                on_event(
                    AgentEvent(
                        type=event_type,
                        session_id=session.id,
                        step=step,
                        data=data,
                    )
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

_AGENT_SETTINGS_GROUP = "Agent"
_AGENT_TAB_ID = "agent"

# Session-name prefixes used by plugins that drive the agent internally; these
# are implementation details and must not show up in the user's session list.
_INTERNAL_SESSION_PREFIXES = ("summarize-", "instant-summarize-")


def _is_internal_session(name: Any) -> bool:
    """Return True for agent sessions owned by background note workers."""

    text = str(name or "")
    return text.startswith(_INTERNAL_SESSION_PREFIXES)
_AGENT_PAGE_ID = "agent:chat"


def _new_session_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


class AgentPlugin(Plugin):
    """Registers the ``agent`` service, tools, settings, and UI.

    id: ``"llm-agent"``
    requires: ``("settings", "llm")``
    """

    id = "llm-agent"
    requires = ("settings", "llm")
    provides = ("agent",)

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.service: AgentService | None = None
        self._tab_added = False
        self._page_added = False
        self._actions: list[str] = []
        self._current_session_id = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        settings = ctx.require("settings")

        # --- register settings ---
        from plugins.settings.spec import SettingsSpec

        agent_specs = [
            SettingsSpec(
                key="agent.model",
                label="Agent 模型",
                type="string",
                default="deepseek-chat",
                group=_AGENT_SETTINGS_GROUP,
                description="Agent 循环使用的模型；新建会话生效。",
            ),
            SettingsSpec(
                key="agent.max_steps",
                label="最大步数",
                type="int",
                default=12,
                group=_AGENT_SETTINGS_GROUP,
                description="每次运行允许的最大工具调用轮数；新建会话生效。",
            ),
            SettingsSpec(
                key="agent.max_tool_output",
                label="工具输出上限",
                type="int",
                default=8000,
                group=_AGENT_SETTINGS_GROUP,
                description="超过此长度的工具输出会被截断；新建会话生效。",
            ),
            SettingsSpec(
                key="agent.max_history_chars",
                label="上下文工具结果上限",
                type="int",
                default=0,
                group=_AGENT_SETTINGS_GROUP,
                description=(
                    "保留在上下文里的工具结果字符数上限；更早的结果转存为隐藏文件"
                    "（用 read 可再读）；0 = 不限制。新建会话生效。"
                    "注意：非 0 时会在每步改写较早的工具结果，从而击穿提示词前缀缓存，"
                    "导致该步之后的内容全按原价重算。"
                ),
            ),
            SettingsSpec(
                key="agent.system_prompt",
                label="系统提示词",
                type="text",
                default=DEFAULT_SYSTEM_PROMPT,
                group=_AGENT_SETTINGS_GROUP,
                description="新建 Agent 会话的默认系统提示词。",
            ),
        ]
        for spec in agent_specs:
            try:
                settings.register(spec)
            except Exception:
                ctx.logger.warning("settings register failed for %s", spec.key)

        # --- build service ---
        default_model = str(settings.get("agent.model", "deepseek-chat"))
        default_max_steps = _safe_int(settings.get("agent.max_steps", 12), 12)
        default_max_tool_output = _safe_int(settings.get("agent.max_tool_output", 8000), 8000)
        default_system_prompt = str(
            settings.get("agent.system_prompt", DEFAULT_SYSTEM_PROMPT)
        )

        self.service = AgentService(
            ctx=ctx,
            default_model=default_model,
            default_max_steps=default_max_steps,
            default_max_tool_output=default_max_tool_output,
            default_system_prompt=default_system_prompt,
            settings_getter=settings.get,
        )
        ctx.register_service("agent", self.service)

        # --- meta ---
        ctx.meta.register(
            "agent",
            lambda: {
                "default_model": default_model,
                "default_max_steps": default_max_steps,
                "sessions": len(self.service._sessions),  # noqa: SLF001
            },
            plugin_id=self.id,
        )

        # --- event definitions ---
        for evt_name, evt_desc in [
            ("agent.run.started", "Agent run started"),
            ("agent.delta", "Streaming text delta"),
            ("agent.reasoning", "Streaming reasoning delta"),
            ("agent.message", "Assistant message received"),
            ("agent.tool_call", "Tool call issued"),
            ("agent.tool_result", "Tool result received"),
            ("agent.run.finished", "Agent run finished"),
            ("agent.error", "Agent error"),
        ]:
            ctx.events.define(evt_name, evt_desc)

        # --- UI ---
        # The agent is a page, not a sidebar tab: open it from the menu bar or
        # the command palette (Ctrl+P).
        ui = ctx.ui
        ui.add_page(
            _AGENT_PAGE_ID,
            "Agent",
            render=self._render_chat,
            icon="✦",
            order=600,
            plugin_id=self.id,
        )
        self._page_added = True

        # Actions
        for action_id, handler in [
            ("agent.chat", self._handle_chat),
            ("agent.new_session", self._handle_new_session),
            ("agent.select_session", self._handle_select_session),
            ("agent.reset", self._handle_reset),
        ]:
            ui.register_action(action_id, handler, plugin_id=self.id)
            self._actions.append(action_id)

        ui.register_command(
            "agent.open",
            "Agent",
            subtitle="与课堂 Agent 对话、查看会话",
            category="视图",
            icon="✦",
            order=20,
            page_id=_AGENT_PAGE_ID,
            keywords="agent chat session llm",
            plugin_id=self.id,
        )
        self._command_added = True

        ctx.logger.info("agent plugin ready; %s sessions", 0)

    def teardown(self) -> None:
        ctx = self.ctx
        if ctx is not None:
            for action_id in self._actions:
                try:
                    ctx.ui.unregister_action(action_id)
                except Exception:
                    pass
            self._actions.clear()
            if getattr(self, "_command_added", False):
                try:
                    ctx.ui.unregister_command("agent.open")
                except Exception:
                    pass
                self._command_added = False
            if self._page_added:
                try:
                    ctx.ui.remove_page(_AGENT_PAGE_ID)
                except Exception:
                    pass
                self._page_added = False
        self.service = None

    # ------------------------------------------------------------------
    # UI renderers
    # ------------------------------------------------------------------
    def _ui_sessions(self) -> list[dict[str, Any]]:
        """User-facing sessions; hide summarize's internal worker sessions."""

        service = self.service
        if service is None:
            return []
        return [
            s
            for s in service.list_sessions()
            if not _is_internal_session(s.get("name", ""))
        ]

    def _current_session(self, sessions: list[dict[str, Any]]) -> str:
        """Return the selected session id, falling back to the newest one."""

        ids = {s["id"] for s in sessions}
        if self._current_session_id in ids:
            return self._current_session_id
        preferred = next(
            (
                s
                for s in reversed(sessions)
                if not _is_internal_session(s.get("name", ""))
            ),
            sessions[-1] if sessions else None,
        )
        self._current_session_id = preferred["id"] if preferred else ""
        return self._current_session_id

    def _render_sessions(self) -> list[dict[str, Any]]:
        service = self.service
        if service is None:
            return [{"type": "text", "text": "服务未就绪"}]
        sessions = self._ui_sessions()
        if not sessions:
            return [
                {
                    "type": "markdown",
                    "text": "### 会话\n\n暂无会话。点击下方按钮创建。",
                },
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "新建会话",
                            "action": "agent.new_session",
                            "value": {},
                        }
                    ],
                },
            ]
        current_id = self._current_session(sessions)
        items = []
        for s in sessions:
            items.append(
                {
                    "title": s["name"],
                    "subtitle": f"{s['message_count']} 条消息 · {s['model']}",
                    "action": "agent.select_session",
                    "value": {"session_id": s["id"]},
                    "label": "打开",
                    "class": (
                        "btn btn-sm btn-primary"
                        if s["id"] == current_id
                        else "btn btn-sm"
                    ),
                    "active": s["id"] == current_id,
                }
            )
        return [
            {"type": "rows", "items": items},
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "新建会话",
                        "action": "agent.new_session",
                        "value": {},
                    }
                ],
            },
        ]

    def _render_chat(self) -> list[dict[str, Any]]:
        service = self.service
        if service is None:
            return [{"type": "text", "text": "服务未就绪"}]
        sessions = self._ui_sessions()
        if not sessions:
            return [
                {
                    "type": "markdown",
                    "text": "### Agent\n\n暂无会话。请先创建会话。",
                }
            ]
        current_id = self._current_session(sessions)
        current = next(
            (s for s in sessions if s["id"] == current_id), sessions[0]
        )
        session = service.get_session(current["id"])
        blocks: list[dict[str, Any]] = [
            {
                "type": "markdown",
                "text": f"### Agent — {current['name']}\n\n"
                f"model: {current['model']}  |  "
                f"messages: {current['message_count']}",
            }
        ]
        if session is not None:
            # A running session holds its lock for the whole run (minutes for a
            # handout job), so waiting for it would freeze the entire UI state
            # render.  Take a best-effort snapshot instead.
            if session.lock.acquire(timeout=0.05):
                try:
                    msgs = list(session.messages)
                finally:
                    session.lock.release()
            else:
                msgs = list(session.messages)
                blocks.append(
                    {
                        "type": "text",
                        "text": "（会话正在运行，消息列表可能不完整）",
                    }
                )
            for msg in msgs[-20:]:
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))
                if not content and role == "assistant":
                    content = "*(tool calls)*"
                if not content:
                    content = f"*(tool result for {msg.get('name', '?')})*"
                blocks.append(
                    {
                        "type": "log",
                        "lines": [
                            {
                                "time": role.upper(),
                                "text": content[:500],
                            }
                        ],
                    }
                )
        blocks.append(
            {
                "type": "form",
                "fields": [
                    {
                        "name": "message",
                        "label": "输入",
                        "type": "text",
                        "placeholder": "输入消息...",
                    }
                ],
                "submit": {"label": "发送", "action": "agent.chat"},
            }
        )
        return blocks

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _handle_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        if service is None:
            return {"ok": False, "message": "服务未就绪"}
        message = str(payload.get("message", "") or payload.get("values", {}).get("message", ""))
        if not message.strip():
            return {"ok": False, "message": "消息为空"}

        sessions = self._ui_sessions()
        if not sessions:
            session = service.create_session("default")
            self._current_session_id = session.id
            session_id = session.id
        else:
            session_id = self._current_session(sessions)

        service.run_async(session_id, message)
        return {"ok": True, "message": "Agent 正在处理…", "refresh": True}

    def _handle_new_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        if service is None:
            return {"ok": False, "message": "服务未就绪"}
        name = str(payload.get("name", "") or "default")
        session = service.create_session(name)
        self._current_session_id = session.id
        return {
            "ok": True,
            "message": f"已创建会话 {name!r}",
            "refresh": True,
            "open_page": _AGENT_PAGE_ID,
        }

    def _handle_select_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        if service is None:
            return {"ok": False, "message": "服务未就绪"}
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        session_id = str(payload.get("session_id") or values.get("session_id") or "")
        if not session_id:
            return {"ok": False, "message": "缺少会话 ID"}
        if service.get_session(session_id) is None:
            return {"ok": False, "message": "会话不存在或已关闭"}
        self._current_session_id = session_id
        return {
            "ok": True,
            "message": "已切换会话",
            "refresh": True,
            "open_page": _AGENT_PAGE_ID,
        }

    def _handle_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        if service is None:
            return {"ok": False, "message": "服务未就绪"}
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        session_id = str(payload.get("session_id") or values.get("session_id") or "")
        if not session_id:
            session_id = self._current_session(self._ui_sessions())
        if not session_id:
            return {"ok": False, "message": "没有可重置的会话"}
        service.reset(session_id)
        return {"ok": True, "message": "已重置", "refresh": True}


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _response_reasoning(response: Any) -> str:
    """Best-effort extraction of the model's hidden reasoning text (if any).

    Reasoning models expose it in the raw message body under
    ``reasoning_content`` (DeepSeek style) or ``reasoning`` (OpenRouter
    style); non-reasoning responses simply yield an empty string.
    """

    raw = getattr(response, "raw", None)
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if value:
            return str(value)
    return ""
