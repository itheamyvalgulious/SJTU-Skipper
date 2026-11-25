from __future__ import annotations

import json
import shutil
import threading

import pytest

from config import load_config
from core.agent_extensions import (
    AgentExtension,
    AgentExtensionTool,
    AgentCompletionHook,
)
from core.agent_types import (
    AgentEventType,
    AgentProfile,
    AgentRole,
    AgentTool,
    IpcMethod,
)
from core.plugin import Plugin
from core.runtime import build_runtime
from core.workspace import FileWorkspace
from plugins.agent import AgentPlugin
from tests.support import FakeLLM, FakeSettings


class ScriptedRuntime(FakeLLM):
    def __init__(self, script=None):
        super().__init__()
        self.script = script
        self.calls = []
        self.runs = []
        self.sessions = {}
        self.cancelled = {}
        self.transcripts = {}
        self.closed = False

    def call(self, method: IpcMethod, **params):
        self.calls.append((method, params))
        if method == IpcMethod.CREATE_SESSION:
            self.sessions[params["id"]] = params
            self.cancelled[params["id"]] = threading.Event()
            return {"model": "lecture/test"}
        elif method == IpcMethod.CLOSE_SESSION:
            self.sessions.pop(params["session_id"], None)
        return {}

    def cancel(self, session_id):
        self.cancelled[session_id].set()

    def session_transcript(self, session_id):
        return self.transcripts.get(session_id)

    def stream(self, method: IpcMethod, **params):
        assert method == IpcMethod.RUN
        self.runs.append(params)
        if self.script:
            yield from self.script(self, params)
        else:
            yield {"type": "event", "name": "agent.delta", "data": {"text": "Answer"}}
            yield {
                "type": "result",
                "result": {"final_message": "Answer", "steps": 1, "total_tokens": 14},
            }

    def close(self):
        self.closed = True


class StubSettingsPlugin(Plugin):
    id = "settings"
    requires = ()

    def __init__(self, settings):
        self.settings = settings

    def setup(self, ctx):
        ctx.register_service("settings", self.settings)


class StubLLMPlugin(Plugin):
    id = "llm"
    requires = ()

    def setup(self, ctx):
        pass


def _make_agent_runtime(
    tmp_path, script=None, settings=None, runtime_type=ScriptedRuntime
):
    llm = runtime_type(script)
    settings = settings or FakeSettings()
    runtime = build_runtime(
        load_config(data_dir=tmp_path),
        llm=llm,
        plugins=[StubSettingsPlugin(settings), StubLLMPlugin(), AgentPlugin()],
    )
    runtime.start()
    return runtime, runtime.services.require("agent"), llm, settings


def test_session_has_no_generic_prompt(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        session = agent.create_session("plain")
        assert sdk.sessions[session.id]["prompt"] == ""
        explicit = agent.create_session("explicit", system_prompt="Task instructions")
        assert sdk.sessions[explicit.id]["prompt"] == "Task instructions"
    finally:
        runtime.stop()


def test_ordered_followups_and_review_approval(tmp_path):
    def script(sdk, params):
        yield {
            "type": "result",
            "result": {
                "final_message": "Stopped",
                "review_passed": True,
            },
        }

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("writer")
        result = agent.run(session.id, "Stop reviewing", follow_ups=["Keep the notes"])
        assert sdk.runs[0]["instruction"] == "Stop reviewing"
        assert sdk.runs[0]["follow_ups"] == ["Keep the notes"]
        assert result.review_passed is True
    finally:
        runtime.stop()


def test_sessions_use_roles_and_authorized_files(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        source = tmp_path / "source.md"
        source.write_text("complete source")
        profile = AgentProfile(
            name="review",
            role="note_review",
            tools=[AgentTool("read"), AgentTool("grep"), AgentTool("glob")],
        )
        agent.register_extension(
            "review_context",
            AgentExtension(
                id="context",
                roles=("note_review",),
                prompt="Review the source.",
            ),
        )
        agent.register_agent(profile)
        workspace = FileWorkspace({"source.md": source}, readonly=("source.md",))
        session = agent.create_session(
            "review",
            agent="review",
            workspace=workspace,
        )
        declaration = sdk.sessions[session.id]
        assert declaration["role"] == "note_review"
        assert declaration["extensions"]["prompts"] == ["Review the source."]
        assert declaration["files"] == {"source.md": str(source)}
        assert declaration["readonly"] == ["source.md"]
        assert declaration["tools"] == ["read", "grep", "glob"]
        assert agent.list_sessions()[0]["model"] == "lecture/test"

        exported = agent.session_workspace(session.id)
        assert exported is not None
        workspace.register("source.md", tmp_path / "other.md")
        exported.register("source.md", tmp_path / "redirected.md")
        authorized = agent.session_workspace(session.id)
        assert authorized is not None
        assert authorized.path("source.md") == source
        assert authorized.readonly == ("source.md",)
        assert authorized.text("source.md") == "complete source"
    finally:
        runtime.stop()


def test_session_step_is_independent_of_runtime_options(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        session = agent.create_session("chat", meta={"step": "runtime option"})
        assert session.step == 0
        during = []

        def observe(event):
            if event.type == AgentEventType.RUN_STARTED:
                during.append(agent.get_session(session.id).step)

        for _ in range(2):
            assert agent.run(session.id, "Run", on_event=observe).steps == 1
            assert agent.get_session(session.id).step == 1
        assert during == [0, 0]
        assert session.step == 0
        assert sdk.sessions[session.id]["meta"]["step"] == "runtime option"
    finally:
        runtime.stop()


def test_runs_keep_sdk_history_and_forward_events(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        session = agent.create_session("chat")
        observed = []
        first = agent.run(session.id, "First question", on_event=observed.append)
        second = agent.run(session.id, "Second question")
        assert first.final_message == second.final_message == "Answer"
        assert first.total_tokens == 14
        assert [frame["instruction"] for frame in sdk.runs] == [
            "First question",
            "Second question",
        ]
        assert all(set(frame) == {"session_id", "instruction"} for frame in sdk.runs)
        assert [event.type for event in observed] == [
            "agent.run.started",
            "agent.delta",
            "agent.run.finished",
        ]
        agent.close(session.id)
        assert agent.get_session(session.id) is None
        assert session.id not in sdk.sessions
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"finished": False},
        {"event": AgentEventType.RUN_READY},
        {"event": AgentEventType.DELTA},
        {"finished": True},
    ],
)
def test_reviewer_events_preserve_run(tmp_path, payload):
    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.reviewer",
            "data": {
                "role": "note_review",
                **payload,
                "reviewer_id": "writer-review",
                "text": "Review text",
            },
        }
        yield {"type": "result", "result": {"final_message": "Notes complete"}}

    runtime, agent, _, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("writer")
        observed = []
        result = agent.run(session.id, "Write notes", on_event=observed.append)
        assert result.error is None
        assert result.final_message == "Notes complete"
        forwarded = next(item for item in observed if item.type == "agent.reviewer")
        for name, value in payload.items():
            assert forwarded.data[name] is value
        assert forwarded.data["reviewer_id"] == "writer-review"
    finally:
        runtime.stop()


def test_file_content_is_published_before_callbacks_and_refresh(tmp_path, monkeypatch):
    text = "# Formula\n\n$$\nf(x) = \"quoted\" + 'nested'\n$$\n"

    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.file",
            "data": {"file": "notes.md", "content": text},
        }
        yield {"type": "result", "result": {"final_message": "Saved", "steps": 2}}

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, script)
    try:
        from contextlib import contextmanager

        guarded = []

        @contextmanager
        def write_guard(target):
            assert target == path
            guarded.append(True)
            try:
                yield
            finally:
                guarded.pop()

        path = tmp_path / "notes.md"
        path.write_text("old")
        write = FileWorkspace.write

        def guarded_write(workspace, logical, content):
            assert guarded
            return write(workspace, logical, content)

        monkeypatch.setattr(FileWorkspace, "write", guarded_write)
        session = agent.create_session("writer", files={"notes.md": path})
        seen = []
        result = agent.run(
            session.id,
            "Save",
            on_event=lambda event: (
                seen.append((path.read_text(), bool(guarded)))
                if event.type == "agent.file"
                else None
            ),
            write_guard=write_guard,
        )
        assert not result.error
        assert seen == [(text, False)]
        assert path.read_text() == text
    finally:
        runtime.stop()


def test_external_change_is_preserved(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("original")

    def script(sdk, params):
        path.write_text("external change")
        yield {
            "type": "event",
            "name": "agent.file",
            "data": {
                "file": "notes.md",
                "previous": "original",
                "content": "model change",
            },
        }

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("writer", files={"notes.md": path})
        result = agent.run(session.id, "Edit")
        assert result.error
        assert path.read_text() == "external change"
    finally:
        runtime.stop()


@pytest.mark.parametrize("logical", ["source.md", "unauthorized.md"])
def test_runtime_cannot_publish_readonly_or_unknown_files(tmp_path, logical):
    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.file",
            "data": {"file": logical, "content": "overwritten"},
        }

    runtime, agent, _, _ = _make_agent_runtime(tmp_path, script)
    try:
        source = tmp_path / "source.md"
        source.write_text("original")
        session = agent.create_session(
            "reader",
            workspace=FileWorkspace({"source.md": source}, readonly=("source.md",)),
        )
        assert agent.run(session.id, "Read").error
        assert source.read_text() == "original"
        assert not (tmp_path / "unauthorized.md").exists()
    finally:
        runtime.stop()


def test_incomplete_runtime_stream_is_failure(tmp_path):
    def script(sdk, params):
        yield {"type": "event", "name": "agent.delta", "data": {"text": "unfinished"}}

    runtime, agent, _, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("chat")
        assert agent.run(session.id, "Run").error
        assert not agent._session(session.id).runtime_ready
    finally:
        runtime.stop()


def test_cancel_stops_active_runtime_run(tmp_path):
    started = threading.Event()

    def script(sdk, params):
        started.set()
        assert sdk.cancelled[params["session_id"]].wait(3)
        yield {"type": "result", "result": {"error": "cancelled"}}

    runtime, agent, _, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("chat")
        thread = threading.Thread(target=agent.run, args=(session.id, "Run"))
        thread.start()
        assert started.wait(3)
        assert agent.run(session.id, "Concurrent").error
        agent.cancel(session.id)
        thread.join(3)
        assert not thread.is_alive()
        assert agent._session(session.id).cancelled.is_set()
    finally:
        runtime.stop()


def test_run_preserves_failure_when_cancel_fails(tmp_path):
    def fail(sdk, params):
        raise RuntimeError("runtime exited")
        yield

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, fail)
    session = agent.create_session("failure")

    def cancel(session_id):
        raise RuntimeError("cancel transport failed")

    sdk.cancel = cancel
    try:
        result = agent.run(session.id, "Run")
        assert result.error == "runtime exited"
        with pytest.raises(RuntimeError, match="cancel transport failed"):
            agent.close(session.id)
        assert agent.get_session(session.id) is None
        assert session.id not in sdk.sessions
    finally:
        runtime.stop()


def test_runtime_rejects_string_method(tmp_path):
    from agent_runtime.client import RuntimeClient

    client = RuntimeClient(tmp_path)
    with pytest.raises(TypeError, match="IpcMethod"):
        client.call("info")
    assert client._process is None


def test_runtime_json_survives_restart(tmp_path):
    from agent_runtime.client import RuntimeClient

    if shutil.which("bun") is None:
        pytest.skip("Bun is required for the runtime")
    auth = {"users": {"alice": {"password": "test"}}, "session_secret": "unchanged"}
    (tmp_path / "auth.json").write_text(json.dumps(auth))
    client = RuntimeClient(tmp_path)
    client.register_role("ask", "ask_instant", "快速问答")
    try:
        initial = client.call(IpcMethod.INFO)
        directory = client.runtime_dir
        client.call(IpcMethod.CREDENTIALS, provider="deepseek", key="test-key")
        client.call(
            IpcMethod.IMPORT_CREDENTIALS,
            provider="opencode-go",
            credential={
                "type": "oauth",
                "access": "access",
                "refresh": "refresh",
                "expires": 9999999999999,
            },
        )
        client.call(IpcMethod.ROLE_SET, role="ask_instant", selector="@default")
        saved = json.loads((tmp_path / "settings.json").read_text())
        assert saved["llm.roles"]["ask_instant"] == "@default"
        credentials = json.loads((tmp_path / "auth.json").read_text())
        assert credentials["users"] == auth["users"]
        assert credentials["session_secret"] == "unchanged"
        assert credentials["llm"]["deepseek"]["key"] == "test-key"
        assert (tmp_path / "auth.json").stat().st_mode & 0o777 == 0o600
    finally:
        client.close()
    assert directory.exists()
    assert directory == tmp_path / "llm_runtime"
    assert (tmp_path / "llm_runtime" / "bootstrap.json").exists()
    replacement = RuntimeClient(tmp_path)
    replacement.register_role("ask", "ask_instant", "快速问答")
    try:
        restored = replacement.call(IpcMethod.INFO)
        assert (
            json.loads((tmp_path / "settings.json").read_text())["llm.models"]
            == saved["llm.models"]
        )
        assert restored["roles"]["ask_instant"] == restored["roles"]["default"]
        assert (
            json.loads((replacement.runtime_dir / "bootstrap.json").read_text())[
                "credentials"
            ]
            == credentials["llm"]
        )
        assert replacement.session_transcript("old-conversation") is None
        replacement.call(IpcMethod.LOGOUT, provider="opencode-go")
        assert (
            "opencode-go" not in json.loads((tmp_path / "auth.json").read_text())["llm"]
        )
    finally:
        replacement.close()


def test_cleanup_does_not_start_runtime(tmp_path, monkeypatch):
    from agent_runtime.client import RuntimeClient

    client = RuntimeClient(tmp_path)

    def start():
        raise AssertionError("Cleanup restarted the runtime")

    monkeypatch.setattr(client, "_start", start)
    client.cancel("completed")
    client.call(IpcMethod.CLOSE_SESSION, session_id="completed")
    assert client._process is None


def test_execution_updates_session_model(tmp_path):
    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.execution",
            "data": {"metrics": {"model": "resolved-model"}},
        }
        yield {"type": "result", "result": {"final_message": "Answer"}}

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, script)
    try:
        session = agent.create_session("chat")
        result = agent.run(session.id, "Question")
        assert not result.error
        assert agent.list_sessions()[0]["model"] == "resolved-model"
    finally:
        runtime.stop()


def test_message_does_not_wait_for_run_lock(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def script(sdk, params):
        yield {
            "type": "event",
            "name": "agent.run.ready",
            "data": {"session_id": params["session_id"]},
        }
        started.set()
        assert release.wait(3)
        yield {"type": "result", "result": {"final_message": "continued"}}

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, script)
    original = sdk.call
    messages = []

    def call(method, **params):
        if method == IpcMethod.MESSAGE:
            messages.append(params)
            release.set()
            return {"accepted": True}
        return original(method, **params)

    sdk.call = call
    session = agent.create_session("writer")
    observed = []
    thread = threading.Thread(
        target=lambda: agent.run(session.id, "start", on_event=observed.append)
    )
    thread.start()
    try:
        assert started.wait(3)
        assert agent.message(session.id, "first\nsecond") == {"accepted": True}
        thread.join(3)
        assert not thread.is_alive()
        assert messages == [{"session_id": session.id, "message": "first\nsecond"}]
        assert any(
            event.type == "agent.run.ready" and event.session_id == session.id
            for event in observed
        )
        assert not agent._session(session.id).cancelled.is_set()
        assert agent.message("bun-owned-reviewer", "review detail")["accepted"]
    finally:
        release.set()
        thread.join(3)
        runtime.stop()


def test_message_rejects_missing_admission_and_closed_runtime(tmp_path, monkeypatch):
    from agent_runtime.client import RuntimeClient

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        assert not agent.message("unknown", "hello")["accepted"]
        assert not agent.message("unknown", " \n ")["accepted"]
        client = RuntimeClient(tmp_path)
        monkeypatch.setattr(
            client, "_start", lambda: pytest.fail("Message started a process")
        )
        agent._runtime = client
        result = agent.message("completed", "hello")
        assert not result["accepted"]
        assert "closed" in result["reason"]
        assert client._process is None
    finally:
        runtime.stop()


def test_reopen_uses_strict_native_operation(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    session = agent.create_session("writer", role="note_writer", tools=[])
    values = dict(
        name="writer",
        workspace=agent.session_workspace(session.id),
        system_prompt="system",
        role="note_writer",
        tools=[],
        meta={},
    )
    try:
        with pytest.raises(RuntimeError, match="still live"):
            agent.reopen(session.id, **values)
        agent.close(session.id)
        original = sdk.call

        def missing(method, **params):
            if method == IpcMethod.REOPEN_SESSION:
                raise RuntimeError("Saved transcript is missing")
            return original(method, **params)

        sdk.call = missing
        with pytest.raises(RuntimeError, match="transcript"):
            agent.reopen(session.id, **values)
        assert agent.get_session(session.id) is None

        def reopen(method, **params):
            if method == IpcMethod.REOPEN_SESSION:
                sdk.sessions[params["id"]] = params
                sdk.cancelled[params["id"]] = threading.Event()
                return {"model": "restored-model"}
            return original(method, **params)

        sdk.call = reopen
        restored = agent.reopen(session.id, **values)
        assert restored.id == session.id
        assert restored.model == "restored-model"
        assert not agent._session(session.id).cancelled.is_set()
    finally:
        runtime.stop()


def test_can_resume_reads_saved_transcript_through_runtime(tmp_path):
    from agent_runtime.client import RuntimeClient

    if shutil.which("bun") is None:
        pytest.skip("Bun is required for the runtime")
    runtime, agent, _sdk, _ = _make_agent_runtime(tmp_path)
    client = RuntimeClient(tmp_path)
    try:
        live = agent.create_session("writer")
        agent._runtime = client
        assert agent.can_resume("") == (False, "agent session is unknown")
        assert agent.can_resume(live.id) == (False, "agent session is still live")
        assert agent.can_resume("writer") == (
            False,
            "saved session transcript is unavailable",
        )
        pointer = client.runtime_dir / "session-writer.path"
        transcript = client.runtime_dir / "writer.jsonl"
        pointer.write_text(str(transcript), encoding="utf-8")
        assert not agent.can_resume("writer")[0]
        transcript.write_text("{}\n", encoding="utf-8")
        assert not agent.can_resume("writer")[0]
        header = {
            "type": "session",
            "version": 3,
            "id": "writer",
            "timestamp": "2026-01-01T00:00:00Z",
            "cwd": str(tmp_path),
        }
        entry = {
            "type": "message",
            "id": "prompt",
            "parentId": None,
            "timestamp": header["timestamp"],
            "message": {"role": "user", "content": "Continue", "timestamp": 0},
        }
        transcript.write_text(json.dumps(header) + "\n", encoding="utf-8")
        assert not agent.can_resume("writer")[0]
        history = "\n".join(map(json.dumps, (header, entry))) + "\n"
        transcript.write_text(history, encoding="utf-8")
        assert agent.can_resume("writer") == (True, "")
        assert transcript.read_text(encoding="utf-8") == history
    finally:
        client.close()
        runtime.stop()


class CallbackRuntime(ScriptedRuntime):
    def set_callback_handler(self, handler):
        self.callback_handler = handler


def _custom_tool(name="lookup", handler=None):
    return AgentExtensionTool(
        name=name,
        description="Read a plugin value.",
        parameters={"type": "object", "properties": {}},
        handler=handler
        or (
            lambda context: {
                "content": [{"type": "text", "text": "value"}],
                "details": {},
            }
        ),
    )


def test_extensions_are_role_scoped(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)
    seen = []
    tool = _custom_tool(
        handler=lambda context: (
            seen.append(context)
            or {"content": [{"type": "text", "text": "found"}], "details": {"count": 1}}
        )
    )
    try:
        agent.register_extension(
            "search",
            AgentExtension(
                id="context",
                roles=("ask_research",),
                prompt="Read plugin context.",
                tools=(tool,),
            ),
        )
        plain = agent.create_session("plain")
        research = agent.create_session(
            "research", role="ask_research", tools=["lookup"]
        )
        assert plain.role == AgentRole.DEFAULT
        assert research.role == "ask_research"
        assert sdk.sessions[plain.id]["extensions"]["tools"] == []
        declaration = sdk.sessions[research.id]
        assert declaration["tools"] == ["lookup"]
        assert declaration["extensions"]["tools"] == [tool.descriptor()]
        assert declaration["extensions"]["prompts"] == ["Read plugin context."]
        request = {
            "session_id": research.id,
            "role": "ask_research",
            "kind": "tool",
            "name": "lookup",
            "arguments": {"parameters": {"query": "formula"}, "workspace": {}},
        }
        assert sdk.callback_handler(request)["details"] == {"count": 1}
        assert seen == [
            {
                "session_id": research.id,
                "role": "ask_research",
                "workspace": {},
                "parameters": {"query": "formula"},
            }
        ]
        with pytest.raises(PermissionError, match="role"):
            sdk.callback_handler({**request, "role": "default"})
        with pytest.raises(KeyError, match="not declared"):
            sdk.callback_handler({**request, "session_id": plain.id, "role": "default"})
    finally:
        runtime.stop()


def test_prompt_extensions_do_not_require_callbacks(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        agent.register_extension(
            "prompt",
            AgentExtension(id="context", roles=("default",), prompt="Use the source."),
        )
        session = agent.create_session("plain")
        assert sdk.sessions[session.id]["extensions"]["prompts"] == ["Use the source."]
        assert not hasattr(sdk, "set_callback_handler")
    finally:
        runtime.stop()


def test_extensions_reject_tool_collisions(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        first = AgentExtension(id="first", roles=("default",), tools=(_custom_tool(),))
        agent.register_extension("one", first)
        with pytest.raises(ValueError, match="already registered"):
            agent.register_extension("one", first)
        with pytest.raises(ValueError, match="Duplicate tool"):
            agent.register_extension(
                "two",
                AgentExtension(
                    id="second", roles=("default",), tools=(_custom_tool(),)
                ),
            )
        for reserved in ("read", "grep", "glob", "edit", "write", "review_notes"):
            with pytest.raises(ValueError, match="reserved tool"):
                agent.register_extension(
                    "reserved",
                    AgentExtension(
                        id=reserved,
                        roles=("default",),
                        tools=(_custom_tool(reserved),),
                    ),
                )
        agent.register_extension(
            "two",
            AgentExtension(
                id="other_role",
                roles=("ask_research",),
                tools=(_custom_tool(),),
            ),
        )
        with pytest.raises(RuntimeError, match="callback support"):
            agent.create_session("requires_callbacks")
        assert not sdk.sessions
        with pytest.raises(ValueError, match="Undeclared tools"):
            agent.create_session("undeclared", role="note_chunk", tools=["lookup"])
    finally:
        runtime.stop()


def test_hook_order_and_reviewer_snapshot(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)
    seen = []

    def hook(name, priority=0):
        return AgentCompletionHook(
            name,
            lambda context: seen.append(context) or {"continue": name},
            priority=priority,
        )

    try:
        agent.register_extension(
            "z",
            AgentExtension(
                id="last",
                roles=("note_review",),
                prompt="Last prompt.",
                completion_hooks=(hook("a"),),
                priority=1,
            ),
        )
        agent.register_extension(
            "a",
            AgentExtension(
                id="z",
                roles=("note_review",),
                prompt="Second prompt.",
                completion_hooks=(hook("z"), hook("A")),
            ),
        )
        agent.register_extension(
            "a",
            AgentExtension(
                id="A",
                roles=("note_review",),
                prompt="First prompt.",
                completion_hooks=(hook("early", -1),),
                tools=(_custom_tool(),),
            ),
        )
        extra = tmp_path / "extra.md"
        extra.write_text("content")
        writer = agent.create_session(
            "writer",
            role="note_writer",
            files={"extra.md": extra},
            meta={
                "review": {
                    "role": "note_review",
                    "targets": ["extra.md"],
                    "sources": [],
                }
            },
        )
        payload = sdk.sessions[writer.id]["extensions"]
        assert payload["prompts"] == []
        reviewer = payload["reviewer"]
        assert "reviewer" not in reviewer
        assert reviewer["prompts"] == [
            "First prompt.",
            "Second prompt.",
            "Last prompt.",
        ]
        names = [
            f"{item['owner']}/{item['id']}" for item in reviewer["completion_hooks"]
        ]
        assert names == ["a/early", "a/A", "a/z", "z/a"]
        calls = [{"name": "read", "arguments": {}, "result": "content"}]
        for name in names:
            assert sdk.callback_handler(
                {
                    "session_id": writer.id + "-review",
                    "role": "note_review",
                    "kind": "completion_hook",
                    "name": name,
                    "arguments": {
                        "text": "report",
                        "workspace": {"extra.md": str(extra)},
                        "tool_calls": calls,
                    },
                }
            ) == {"continue": name.split("/")[1]}
        assert len(seen) == 4
        assert all(
            context
            == {
                "session_id": writer.id + "-review",
                "role": "note_review",
                "workspace": {"extra.md": str(extra)},
                "text": "report",
                "tool_calls": calls,
            }
            for context in seen
        )
        assert (
            sdk.callback_handler(
                {
                    "session_id": writer.id + "-review",
                    "role": "note_review",
                    "kind": "tool",
                    "name": "lookup",
                    "arguments": {
                        "parameters": {},
                        "workspace": {"extra.md": str(extra)},
                    },
                }
            )["content"][0]["text"]
            == "value"
        )
    finally:
        runtime.stop()


def test_unregister_preserves_live_callbacks(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)
    try:
        agent.register_extension(
            "owner",
            AgentExtension(id="tool", roles=("default",), tools=(_custom_tool(),)),
        )
        session = agent.create_session("live")
        agent.unregister_extensions("owner")
        fresh = agent.create_session("fresh")
        assert sdk.sessions[fresh.id]["extensions"]["tools"] == []
        request = {
            "session_id": session.id,
            "role": "default",
            "kind": "tool",
            "name": "lookup",
            "arguments": {"parameters": {}, "workspace": {}},
        }
        assert sdk.callback_handler(request)["content"][0]["text"] == "value"
        agent.close(session.id)
        with pytest.raises(KeyError, match="not live"):
            sdk.callback_handler(request)
    finally:
        runtime.stop()


def test_plugin_unload_removes_extensions(tmp_path):
    class ExtensionPlugin(Plugin):
        id = "extension_owner"
        requires = ("agent",)

        def setup(self, ctx):
            ctx.require("agent").register_extension(
                self.id,
                AgentExtension(id="prompt", roles=("default",), prompt="Owner prompt."),
            )

    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        runtime.plugins.load_plugin(ExtensionPlugin())
        existing = agent.create_session("existing")
        runtime.plugins.unload("extension_owner")
        fresh = agent.create_session("fresh")
        assert sdk.sessions[existing.id]["extensions"]["prompts"] == ["Owner prompt."]
        assert sdk.sessions[fresh.id]["extensions"]["prompts"] == []
        assert agent.get_session(existing.id) is not None
    finally:
        runtime.stop()


def test_reopen_resolves_extensions_again(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        agent.register_extension(
            "owner",
            AgentExtension(
                id="prompt", roles=("ask_research",), prompt="Original prompt."
            ),
        )
        session = agent.create_session("original", role="ask_research", tools=[])
        agent.close(session.id)
        agent.unregister_extensions("owner")
        agent.register_extension(
            "owner",
            AgentExtension(
                id="prompt",
                roles=("ask_research",),
                prompt="Replacement prompt.",
            ),
        )
        original = sdk.call

        def reopen(method, **params):
            if method == IpcMethod.REOPEN_SESSION:
                return original(IpcMethod.CREATE_SESSION, **params)
            return original(method, **params)

        sdk.call = reopen
        restored = agent.reopen(
            session.id,
            name="restored",
            workspace=FileWorkspace(),
            system_prompt="",
            role="ask_research",
            tools=[],
            meta={},
        )
        assert restored.role == "ask_research"
        assert sdk.sessions[restored.id]["extensions"]["prompts"] == [
            "Replacement prompt."
        ]
    finally:
        runtime.stop()


@pytest.mark.parametrize("result", [{"continue": ""}, {"continue": "  "}, True])
def test_hook_rejects_invalid_results(tmp_path, result):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)
    try:
        agent.register_extension(
            "owner",
            AgentExtension(
                id="hook",
                roles=("default",),
                completion_hooks=(
                    AgentCompletionHook("check", lambda context: result),
                ),
            ),
        )
        session = agent.create_session("hook")
        with pytest.raises(TypeError, match="non-empty str"):
            sdk.callback_handler(
                {
                    "session_id": session.id,
                    "role": "default",
                    "kind": "completion_hook",
                    "name": "owner/check",
                    "arguments": {
                        "text": "malformed report",
                        "workspace": {},
                        "tool_calls": [],
                    },
                }
            )
    finally:
        runtime.stop()


def test_hook_exceptions_propagate(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)

    def fail(context):
        raise RuntimeError("hook failure")

    try:
        agent.register_extension(
            "owner",
            AgentExtension(
                id="hook",
                roles=("default",),
                completion_hooks=(AgentCompletionHook("check", fail),),
            ),
        )
        session = agent.create_session("hook")
        with pytest.raises(RuntimeError, match="hook failure"):
            sdk.callback_handler(
                {
                    "session_id": session.id,
                    "role": "default",
                    "kind": "completion_hook",
                    "name": "owner/check",
                    "arguments": {
                        "text": "malformed report",
                        "workspace": {},
                        "tool_calls": [],
                    },
                }
            )
    finally:
        runtime.stop()


def test_extensions_reject_unknown_roles(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        with pytest.raises(ValueError):
            agent.register_extension(
                "invalid",
                AgentExtension(
                    id="context", roles=("unknown_role",), prompt="Invalid role."
                ),
            )
        session = agent.create_session("plain")
        assert sdk.sessions[session.id]["extensions"]["prompts"] == []
    finally:
        runtime.stop()


def test_custom_roles_and_reviewer_callbacks(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path, runtime_type=CallbackRuntime)
    try:
        sdk.register_role("custom", "custom_writer", "Writer")
        sdk.register_role("custom", "custom_review", "Reviewer")
        for role in ("custom_writer", "custom_review"):
            agent.register_extension(
                "custom",
                AgentExtension(
                    id=role,
                    roles=(role,),
                    prompt=role,
                    tools=(_custom_tool(handler=lambda context: context["role"]),),
                ),
            )
        agent.register_agent(AgentProfile(name="custom", role="custom_writer"))
        session = agent.create_session(
            "custom",
            agent="custom",
            tools=["lookup"],
            meta={"review": {"role": "custom_review", "targets": [], "sources": []}},
        )
        assert session.role == "custom_writer"
        extensions = sdk.sessions[session.id]["extensions"]
        assert extensions["prompts"] == ["custom_writer"]
        assert extensions["reviewer"]["prompts"] == ["custom_review"]
        for suffix, role in (("", "custom_writer"), ("-review", "custom_review")):
            assert (
                sdk.callback_handler(
                    {
                        "session_id": session.id + suffix,
                        "role": role,
                        "kind": "tool",
                        "name": "lookup",
                        "arguments": {"parameters": {}, "workspace": {}},
                    }
                )
                == role
            )
        agent.close(session.id)
        sdk.unregister_roles("custom")
        with pytest.raises(ValueError, match="Unregistered agent role"):
            agent.reopen(
                session.id,
                name="custom",
                workspace=FileWorkspace(),
                system_prompt="",
                role="custom_writer",
                tools=[],
                meta={},
            )
    finally:
        runtime.stop()


def test_unregistered_roles_are_rejected(tmp_path):
    runtime, agent, sdk, _ = _make_agent_runtime(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unregistered agent role"):
            agent.register_extension(
                "custom", AgentExtension(id="missing", roles=("missing",))
            )
        with pytest.raises(ValueError, match="Unregistered agent role"):
            agent.create_session("missing", role="missing")
        with pytest.raises(ValueError, match="Unregistered agent role"):
            agent.create_session("review", meta={"review": {"role": "missing"}})
        assert not sdk.sessions
    finally:
        runtime.stop()
