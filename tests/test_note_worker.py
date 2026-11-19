import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace


from core.streams import StreamHub
from core.workspace import FileWorkspace
from plugins.agent import AgentService
from plugins.notes.pipeline import NotesPipeline
from plugins.notes.types import (
    ChunkState,
    GenerationState,
    NotePhase,
)
from plugins.notes.view import OnStopAction
from core.agent_types import (
    AgentEventType,
    AgentRole,
    ExecutionBlockKind,
    ExecutionBlockStatus,
    RuntimeFrameType,
)
from core.agent_types import IpcMethod
from plugins.notes.prompts import NOTE_HEADERS, SOURCE_FILENAME
from tests.support import FakeSettings
from tests.test_agent import ScriptedRuntime


@contextmanager
def pipeline_context(tmp_path, script):
    sdk = ScriptedRuntime(script)
    settings = FakeSettings({"notes.concurrency": 4, "notes.chunks_enabled": False})
    ctx = SimpleNamespace(
        data_dir=tmp_path,
        streams=StreamHub(),
        llm=sdk,
        events=SimpleNamespace(on=lambda *a, **k: None, emit=lambda *a, **k: None),
        ui=SimpleNamespace(
            refresh=lambda reason: None,
            register_task_provider=lambda *a, **k: None,
            unregister_action=lambda *a, **k: None,
            register_action=lambda *a, **k: None,
        ),
        register_service=lambda *a: None,
        require=lambda name: sdk if name == "llm" else settings,
    )
    agent = AgentService(ctx)
    ctx.require = lambda name: {"settings": settings, "agent": agent, "llm": sdk}[name]
    ctx.service = lambda name: None
    pipeline = NotesPipeline()
    pipeline.setup(ctx)
    try:
        yield pipeline, agent, sdk
    finally:
        pipeline.teardown()


def test_stop_cancels_all_active_sdk_sessions(tmp_path):
    started = queue.Queue()

    def script(sdk, params):
        started.put(params["session_id"])
        assert sdk.cancelled[params["session_id"]].wait(5)
        yield {"type": RuntimeFrameType.RESULT, "result": {"error": "cancelled"}}

    with pipeline_context(tmp_path, script) as (pipeline, agent, sdk):
        errors = queue.Queue()
        threads = []
        for role in (AgentRole.NOTE_WRITER, AgentRole.NOTE_REVIEW):
            session = agent.create_session(role, role=role, tools=[])

            def execute(session=session):
                try:
                    pipeline._agent_run("lecture", session.id, "Wait")
                except Exception as exc:
                    errors.put(exc)
                finally:
                    agent.close(session.id)

            thread = threading.Thread(target=execute)
            thread.start()
            threads.append(thread)
        ids = [started.get(timeout=3) for _ in threads]
        assert pipeline.stop("lecture")
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
        assert all(sdk.cancelled[sid].is_set() for sid in ids)
        assert errors.qsize() == 2
        assert not pipeline.agent_tasks
        assert (
            pipeline.status("lecture")["generation_state"] == GenerationState.FINISHED
        )


def test_writer_preview_is_published_before_terminal_result(tmp_path):
    release = threading.Event()
    changed = threading.Event()

    def script(sdk, params):
        yield {
            "type": RuntimeFrameType.EVENT,
            "name": AgentEventType.FILE,
            "data": {
                "file": "note_knowledge.md",
                "content": "# Notes\n\nFirst theorem and all conditions.\n",
            },
        }
        changed.set()
        assert release.wait(5)
        yield {
            "type": RuntimeFrameType.RESULT,
            "result": {"final_message": "Saved", "steps": 2},
        }

    with pipeline_context(tmp_path, script) as (pipeline, agent, sdk):
        directory = tmp_path / "private"
        directory.mkdir()
        files = {name: directory / name for name in (*NOTE_HEADERS, SOURCE_FILENAME)}
        for name, path in files.items():
            path.write_text(NOTE_HEADERS.get(name, "Complete original source"))
        session = agent.create_session(
            "writer",
            role=AgentRole.NOTE_WRITER,
            workspace=FileWorkspace(files, readonly=(SOURCE_FILENAME,)),
        )
        results = queue.Queue()
        thread = threading.Thread(
            target=lambda: results.put(
                pipeline._agent_run("lecture", session.id, "Write chapters")
            )
        )
        thread.start()
        try:
            assert changed.wait(3)
            assert thread.is_alive()
            notes = tmp_path / "sessions" / "lecture" / "notes"
            assert "First theorem" in (notes / "note_knowledge.md").read_text()
            assert "未审阅" in (notes / "note_knowledge.md").read_text()
            assert "First theorem" in files["note_knowledge.md"].read_text()
            assert pipeline.status("lecture")["phase"] != NotePhase.PUBLISHED
        finally:
            release.set()
            thread.join(3)
            agent.close(session.id)
        assert not results.get(timeout=1).error


def test_note_runs_send_only_session_and_instruction(tmp_path):
    with pipeline_context(tmp_path, None) as (pipeline, agent, sdk):
        session = agent.create_session("writer", role=AgentRole.NOTE_WRITER, tools=[])
        try:
            first = pipeline._agent_run("lecture", session.id, "First stage")
            second = pipeline._agent_run("lecture", session.id, "Continue")
            assert first.final_message == second.final_message == "Answer"
            assert len(sdk.runs) == 2
            assert all(set(run) == {"session_id", "instruction"} for run in sdk.runs)
            assert all(run["session_id"] == session.id for run in sdk.runs)
            assert "lecture" in sdk.runs[0]["instruction"]
        finally:
            agent.close(session.id)


def test_empty_finalization_releases_generation(tmp_path):
    with pipeline_context(tmp_path, None) as (pipeline, agent, sdk):
        session = pipeline.ctx.streams.start_session("empty-finalization")
        session.input_done = True
        with pipeline.condition:
            assert pipeline.finalize(session.session_id)
            pipeline._schedule(session.session_id)
            assert (
                pipeline.status(session.session_id)["generation_state"]
                == GenerationState.NOT_STARTED
            )
            assert not pipeline.queue
            assert not pipeline.stop(session.session_id)


def test_chunk_only_completion_releases_generation(tmp_path):
    with pipeline_context(tmp_path, None) as (pipeline, agent, sdk):
        pipeline.settings.set("notes.chunks_enabled", True)
        pipeline.settings.set("notes.on_stop", OnStopAction.SKIP)
        session = pipeline.ctx.streams.start_session("chunk-completion")
        with pipeline.condition:
            state = pipeline._state(session.session_id)
            assert (
                pipeline.status(session.session_id)["generation_state"]
                == GenerationState.GENERATING
            )
            state["chunks"] = [{"state": ChunkState.EMPTY, "index": 0, "end": 600}]
            session.input_done = True
            assert (
                pipeline.status(session.session_id)["generation_state"]
                == GenerationState.FINISHED
            )
            assert not pipeline.stop(session.session_id)
            assert pipeline.delete(session.session_id)
            assert (
                pipeline.status(session.session_id)["generation_state"]
                == GenerationState.NOT_STARTED
            )


def test_repeated_writer_runs_keep_one_activity_record(tmp_path):
    count = 0

    def script(sdk, params):
        nonlocal count
        count += 1
        yield {
            "type": RuntimeFrameType.EVENT,
            "name": AgentEventType.EXECUTION,
            "data": {
                "block": {
                    "id": f"run-{count}:text",
                    "kind": ExecutionBlockKind.TEXT,
                    "status": ExecutionBlockStatus.COMPLETED,
                    "text": f"turn {count}",
                }
            },
        }
        yield {"type": RuntimeFrameType.RESULT, "result": {"final_message": "done"}}

    with pipeline_context(tmp_path, script) as (pipeline, agent, sdk):
        session = agent.create_session("writer", role=AgentRole.NOTE_WRITER, tools=[])
        try:
            pipeline._agent_run("lecture", session.id, "First")
            pipeline._agent_run("lecture", session.id, "Second")
            tasks = pipeline.status("lecture")["tasks"]
            assert len(tasks) == 1
            main = tasks[0]["agents"]["main"]
            assert main["session_id"] == session.id
            assert [block["text"] for block in main["blocks"]] == ["turn 1", "turn 2"]
        finally:
            agent.close(session.id)


def test_completed_chunk_message_is_rejected(tmp_path):
    with pipeline_context(tmp_path, None) as (pipeline, agent, sdk):
        session = agent.create_session("chunk", role=AgentRole.NOTE_CHUNK, tools=[])
        pipeline._agent_run("lecture", session.id, "Summarize")
        agent.close(session.id)
        result = pipeline.message(
            {
                "session_id": "lecture",
                "agent_session_id": session.id,
                "message": "Change",
            }
        )
        assert not result["ok"]
        assert len(sdk.runs) == 1
        assert not any(method == IpcMethod.REOPEN_SESSION for method, _ in sdk.calls)


def test_live_message_releases_condition_before_ipc(tmp_path):
    entered = threading.Event()
    released = threading.Event()
    running = threading.Event()

    def script(sdk, params):
        running.set()
        assert sdk.cancelled[params["session_id"]].wait(5)
        yield {"type": RuntimeFrameType.RESULT, "result": {"error": "cancelled"}}

    with pipeline_context(tmp_path, script) as (pipeline, agent, sdk):
        session = agent.create_session("writer", role=AgentRole.NOTE_WRITER, tools=[])
        original = sdk.call

        def call(method, **params):
            if method == IpcMethod.MESSAGE:
                entered.set()
                assert released.wait(5)
                return {"accepted": False, "reason": "Agent session is closing"}
            return original(method, **params)

        sdk.call = call

        def run():
            try:
                pipeline._agent_run("lecture", session.id, "Wait")
            except RuntimeError:
                pass
            finally:
                agent.close(session.id)

        owner = threading.Thread(target=run)
        owner.start()
        assert running.wait(3)
        results = queue.Queue()
        sender = threading.Thread(
            target=lambda: results.put(
                pipeline.message(
                    {
                        "session_id": "lecture",
                        "agent_session_id": session.id,
                        "message": "Continue",
                    }
                )
            )
        )
        sender.start()
        try:
            assert entered.wait(3)
            with pipeline.condition:
                assert session.id in pipeline.agent_tasks
            assert pipeline.stop("lecture")
            released.set()
            sender.join(3)
            owner.join(3)
            assert not sender.is_alive() and not owner.is_alive()
            assert not results.get(timeout=1)["ok"]
            assert not pipeline.agent_tasks
        finally:
            released.set()
            if agent.get_session(session.id) is not None:
                agent.cancel(session.id)
            sender.join(3)
            owner.join(3)
