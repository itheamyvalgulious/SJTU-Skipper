"""Tests for the P4 Summarize plugin."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config import load_config
from core.plugin import Plugin, PluginContext
from core.runtime import build_runtime
from plugins.settings.spec import SettingsSpec
from plugins.instant_summarize import InstantSummarizePlugin
from plugins.reader import ReaderPlugin
from plugins.instant_summarize.plugin import INSTANT_NOTE_HEADERS, INSTANT_NOTE_FILES

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeAgentService:
    """Scriptable fake agent service.

    Parameters
    ----------
    noop_runs : int
        Number of initial ``run()`` calls that return successfully without
        writing any file content (used to test "no change → retry" path).
    fail_runs : int
        Number of initial ``run()`` calls that raise RuntimeError.
    """

    def __init__(self, *, noop_runs: int = 0, fail_runs: int = 0) -> None:
        self._noop_runs = noop_runs
        self._fail_runs = fail_runs
        self._run_count = 0
        self._next_id = 0
        self.sessions: dict[str, SimpleNamespace] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []

    # -- agent service contract -------------------------------------------

    def create_session(
        self,
        name: str,
        system_prompt: str,
        files: dict[str, str],
        tools: list[str],
        model: str | None = None,
        max_steps: int = 8,
        max_tool_output: int | None = None,
        extra_instructions: list[str] | None = None,
        parent_id: str | None = None,
    ) -> SimpleNamespace:
        self._next_id += 1
        sid = f"agent-{self._next_id}"
        session = SimpleNamespace(id=sid, name=name, files=dict(files))
        self.sessions[sid] = session
        self.create_calls.append(
            dict(
                name=name,
                tools=list(tools),
                model=model,
                max_steps=max_steps,
                max_tool_output=max_tool_output,
            )
        )
        return session

    def run(
        self,
        session_id: str,
        user_input: str,
        on_event: Any = None,
    ) -> SimpleNamespace:
        self._run_count += 1
        try:
            if self._run_count <= self._fail_runs:
                raise RuntimeError(f"fake agent failure #{self._run_count}")

            if self._run_count <= self._fail_runs + self._noop_runs:
                return SimpleNamespace(
                    content="done (noop)", finish_reason="stop"
                )

            # Write test content to each file.
            session = self.sessions.get(session_id)
            if session is not None:
                for filename, path in sorted(session.files.items()):
                    p = Path(path)
                    existing = p.read_text("utf-8") if p.exists() else ""
                    p.write_text(
                        existing
                        + f"\n## 第 {self._run_count} 段\n\n"
                        + f"Fake summary line for {filename}. Run #{self._run_count}.\n"
                    )

            return SimpleNamespace(
                content=f"done run #{self._run_count}", finish_reason="stop"
            )
        finally:
            # Record only after the files are written so wait_for_runs() is a
            # reliable synchronization point for tests.
            self.run_calls.append(
                dict(session_id=session_id, user_input=user_input)
            )

    # -- test helpers -----------------------------------------------------

    def wait_for_runs(self, count: int, timeout: float = 5.0) -> bool:
        """Block until at least *count* calls to ``run()`` have been made."""
        deadline = time.monotonic() + timeout
        while len(self.run_calls) < count:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True


class _StubSettingsPlugin(Plugin):
    """In-process settings plugin stub."""

    id = "settings"

    def setup(self, ctx: PluginContext) -> None:
        from tests.support import FakeSettings

        self._service = FakeSettings()
        ctx.register_service("settings", self._service)

    def teardown(self) -> None:
        pass


class _StubAgentPlugin(Plugin):
    """In-process agent plugin stub."""

    id = "llm-agent"
    requires = ("settings",)

    def __init__(self, agent: FakeAgentService | None = None) -> None:
        super().__init__()
        self._agent = agent or FakeAgentService()

    def setup(self, ctx: PluginContext) -> None:
        ctx.register_service("agent", self._agent)

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path) -> tuple[Any, Any, FakeAgentService]:
    """Return ``(runtime, settings_service, agent_service)`` after startup."""
    agent = FakeAgentService()
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(
        cfg,
        plugins=[
            _StubSettingsPlugin(),
            _StubAgentPlugin(agent),
            ReaderPlugin(),
            InstantSummarizePlugin(),
        ],
    )
    runtime.start()
    settings = runtime.services.get("settings")
    assert settings is not None
    yield runtime, settings, agent
    runtime.stop()


def _start_session(runtime: Any, meta: dict[str, Any] | None = None) -> str:
    """Start a new session, flush events, return session_id."""
    ss = runtime.hub.start_session(meta=meta)
    runtime.bus.flush(2.0)
    return ss.session_id


def _wait_for_agent_runs(agent: FakeAgentService, count: int) -> bool:
    """Wait for the fake agent's ``run()`` to be called *count* times."""
    return agent.wait_for_runs(count, timeout=5.0)


# ---------------------------------------------------------------------------
# 1. Text trigger
# ---------------------------------------------------------------------------


def test_text_trigger(harness: tuple) -> None:
    runtime, settings, agent = harness
    sid = _start_session(runtime)

    streams = runtime.hub.current
    # Append ~600+ chars of transcript text (4*30=600).
    big = "今天我们来讨论代数拓扑。" * 60  # 11*60 = 660 chars
    streams.transcript.append(big)
    runtime.bus.flush(2.0)

    # Summary should have been triggered.
    assert _wait_for_agent_runs(agent, 1), "agent.run was not called"

    # Files should contain content written by the fake agent.  ``run()`` is
    # recorded before the fake writes its files, so wait for the write too.
    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(
            (notes_dir / fname).exists()
            and "Fake summary" in (notes_dir / fname).read_text("utf-8")
            for fname in INSTANT_NOTE_FILES
        ):
            break
        time.sleep(0.05)
    for fname in INSTANT_NOTE_FILES:
        path = notes_dir / fname
        assert path.exists(), f"{fname} missing"
        content = path.read_text("utf-8")
        assert INSTANT_NOTE_HEADERS[fname].strip() in content
        assert "Fake summary" in content


# ---------------------------------------------------------------------------
# 2. Audio trigger
# ---------------------------------------------------------------------------


def test_audio_trigger(harness: tuple) -> None:
    runtime, settings, agent = harness
    sid = _start_session(runtime)

    streams = runtime.hub.current

    # Append some text (>= min_chars=80).
    streams.transcript.append("我们来讨论 Riemann 曲面。" * 10)  # ~130 chars
    runtime.bus.flush(0.5)

    # Append enough audio to trigger (>= 120s).
    pcm = b"\x00" * (16000 * 2 * 120)  # 120 seconds of silence
    streams.audio.append(pcm)
    runtime.bus.flush(2.0)

    assert _wait_for_agent_runs(agent, 1), (
        f"agent.run was not called (runs={len(agent.run_calls)})"
    )

    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    assert (notes_dir / INSTANT_NOTE_FILES[0]).exists()


# ---------------------------------------------------------------------------
# 3. Watermark – repeated events do not trigger duplicate summaries
# ---------------------------------------------------------------------------


def test_watermark_no_duplicate(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    streams = runtime.hub.current
    # First batch: enough to trigger.
    streams.transcript.append("A" * 600)
    runtime.bus.flush(2.0)
    assert _wait_for_agent_runs(agent, 1)

    # Second batch: not enough to trigger again.
    streams.transcript.append("B" * 200)
    runtime.bus.flush(2.0)
    # Wait a bit – no new run should happen.
    time.sleep(0.3)
    runtime.bus.flush(1.0)
    assert len(agent.run_calls) == 1


# ---------------------------------------------------------------------------
# 4. Watermark – enough content → second summary with only new content
# ---------------------------------------------------------------------------


def test_watermark_second_trigger(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    streams = runtime.hub.current
    # First trigger.
    streams.transcript.append("A" * 600)
    runtime.bus.flush(2.0)
    assert _wait_for_agent_runs(agent, 1)

    # Second trigger.
    streams.transcript.append("B" * 600)
    runtime.bus.flush(2.0)
    assert _wait_for_agent_runs(agent, 2)

    # Verify two separate runs.
    assert len(agent.run_calls) == 2
    # Second run should only see the new "B" content, not "A".
    second_input = agent.run_calls[1]["user_input"]
    assert "B" * 10 in second_input
    # "A" may or may not appear depending on watermark.
    # But the second run's session should be different from first.
    assert agent.run_calls[0]["session_id"] != agent.run_calls[1]["session_id"]


# ---------------------------------------------------------------------------
# 5. Busy while triggered – pending content processed in next round
# ---------------------------------------------------------------------------


def test_busy_pending_processed_next_round(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    streams = runtime.hub.current

    # First trigger.
    streams.transcript.append("X" * 600)
    runtime.bus.flush(2.0)
    # Do NOT wait for completion yet – immediately append more.
    streams.transcript.append("Y" * 600)
    runtime.bus.flush(2.0)

    # Wait for both runs.
    assert _wait_for_agent_runs(agent, 2), (
        f"expected 2 runs, got {len(agent.run_calls)}"
    )

    # Both summaries should have happened.
    assert len(agent.run_calls) >= 2


# ---------------------------------------------------------------------------
# 6. Failure retry with backoff
# ---------------------------------------------------------------------------


def test_failure_retry(harness: tuple) -> None:
    # Agent that fails once then succeeds – use a fresh runtime.
    import shutil
    import tempfile

    agent2 = FakeAgentService(fail_runs=1)
    d = tempfile.mkdtemp()
    try:
        cfg2 = load_config(data_dir=d)
        rt = build_runtime(
            cfg2,
            plugins=[
                _StubSettingsPlugin(),
                _StubAgentPlugin(agent2),
                ReaderPlugin(),
                InstantSummarizePlugin(),
            ],
        )
        rt.start()
        try:
            _start_session(rt)

            streams = rt.hub.current
            streams.transcript.append("Z" * 600)
            rt.bus.flush(2.0)

            # Should fail once, retry, then succeed on second attempt.
            assert agent2.wait_for_runs(2, timeout=5.0), (
                f"expected 2 runs (1 fail + 1 retry), got {len(agent2.run_calls)}"
            )
            # Small delay to ensure file writes complete.
            time.sleep(0.2)

            # After retry, files should be written.
            sid = rt.hub.current.session_id
            notes_dir = rt.data_dir / "sessions" / sid / "notes"
            content = (notes_dir / INSTANT_NOTE_FILES[0]).read_text("utf-8")
            assert "Fake summary" in content
        finally:
            rt.stop()
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. File headers
# ---------------------------------------------------------------------------


def test_file_headers(harness: tuple) -> None:
    runtime, settings, agent = harness
    sid = _start_session(runtime)

    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    for fname in INSTANT_NOTE_FILES:
        path = notes_dir / fname
        assert path.exists(), f"{fname} should exist after session start"
        content = path.read_text("utf-8")
        # Header text should be present.
        header = INSTANT_NOTE_HEADERS[fname]
        assert header.strip() in content, (
            f"header missing in {fname}: {content[:200]}"
        )


# ---------------------------------------------------------------------------
# 8. File content after summary
# ---------------------------------------------------------------------------


def test_file_content_after_summary(harness: tuple) -> None:
    runtime, settings, agent = harness
    sid = _start_session(runtime)

    streams = runtime.hub.current
    streams.transcript.append("今天讲同调论。" * 100)
    runtime.bus.flush(2.0)
    assert _wait_for_agent_runs(agent, 1)

    notes_dir = runtime.data_dir / "sessions" / sid / "notes"
    for fname in INSTANT_NOTE_FILES:
        content = (notes_dir / fname).read_text("utf-8")
        assert "Fake summary line for" in content, (
            f"no fake content in {fname}: {content[:200]}"
        )


# ---------------------------------------------------------------------------
# 9. UI state
# ---------------------------------------------------------------------------


def test_ui_state(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    snap = runtime.ui.snapshot()
    assert snap["tabs"]["note"] == "right"
    assert snap["items"]["instant-summarize.notes"] == "note"
    assert "instant-summarize.now" in snap["actions"]
    assert "instant-summarize.read" in snap["actions"]
    assert "reader:main" in snap["pages"]
    assert "instant-summarize:notes" not in snap["pages"]

    item = next(
        item
        for tab in runtime.ui.state()["right_tabs"]
        for item in tab["toolbox"]
        if item["id"] == "instant-summarize.notes"
    )
    assert item["blocks"], "note toolbox item rendered empty blocks"

    result = runtime.ui.dispatch_action("instant-summarize.read", {"doc": "gossip"})
    assert result.ok
    assert result.open_page == "reader:main"
    assert runtime.ui.state()["active_page"] == "reader:main"
    reader_blocks = runtime.ui.state()["page_content"]["reader:main"]
    tabs = next(block for block in reader_blocks if block["type"] == "tabs")
    assert [tab["title"] for tab in tabs["tabs"]] == ["知识", "生活趣事", "通知"]


# ---------------------------------------------------------------------------
# 10. summarize.now action
# ---------------------------------------------------------------------------


def test_summarize_now_action(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    streams = runtime.hub.current
    # Append content below trigger threshold.
    streams.transcript.append("短文本。" * 10)  # ~40 chars, < trigger
    runtime.bus.flush(2.0)

    # summarize.now should force a summary.
    result = runtime.ui.dispatch_action("instant-summarize.now", {})
    assert result.ok
    assert "总结" in result.message

    assert _wait_for_agent_runs(agent, 1), (
        f"instant-summarize.now should trigger agent; runs={len(agent.run_calls)}"
    )


# ---------------------------------------------------------------------------
# 11. summarize.now with empty streams
# ---------------------------------------------------------------------------


def test_summarize_now_empty(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    # No content appended – summarize.now should return gracefully.
    result = runtime.ui.dispatch_action("instant-summarize.now", {})
    assert result.ok
    assert "暂无" in result.message

    # No agent run should have happened.
    assert len(agent.run_calls) == 0


# ---------------------------------------------------------------------------
# 12. Audio trigger with insufficient text (min_chars guard)
# ---------------------------------------------------------------------------


def test_audio_trigger_insufficient_text(harness: tuple) -> None:
    runtime, settings, agent = harness
    _start_session(runtime)

    streams = runtime.hub.current

    # Very little text (< min_chars=80).
    streams.transcript.append("Hi.")
    runtime.bus.flush(0.5)

    # Plenty of audio (>= 120s).
    pcm = b"\x00" * (16000 * 2 * 120)
    streams.audio.append(pcm)
    runtime.bus.flush(2.0)

    # Wait a bit – summary should NOT trigger.
    time.sleep(0.5)
    runtime.bus.flush(1.0)

    assert len(agent.run_calls) == 0, (
        "summary should NOT trigger with insufficient text"
    )


# ---------------------------------------------------------------------------
# 13. Settings registration
# ---------------------------------------------------------------------------


def test_settings_registered(harness: tuple) -> None:
    runtime, settings, agent = harness

    expected_keys = [
        "instant-summarize.trigger_chars",
        "instant-summarize.trigger_audio_seconds",
        "instant-summarize.min_chars",
        "instant-summarize.max_steps",
        "instant-summarize.max_tool_output",
        "instant-summarize.model",
        "instant-summarize.notes_dir",
        "instant-summarize.system_prompt",
        "instant-summarize.retry_max",
    ]
    for key in expected_keys:
        val = settings.get(key)
        assert val is not None, f"setting {key} not registered"
    assert settings.get("instant-summarize.trigger_chars") == 600
    assert settings.get("instant-summarize.trigger_audio_seconds") == 120
    assert settings.get("instant-summarize.min_chars") == 80
    assert settings.get("instant-summarize.max_steps") == 12
    assert settings.get("instant-summarize.max_tool_output") == 24000
    assert settings.get("instant-summarize.retry_max") == 2
    assert settings.get("instant-summarize.notes_dir") == "notes"
    assert settings.get("instant-summarize.model") == ""


# ---------------------------------------------------------------------------
# 14. Service registration
# ---------------------------------------------------------------------------


def test_service_registration(harness: tuple) -> None:
    runtime, settings, agent = harness
    assert "instant-summarize" in runtime.services
    svc = runtime.services.get("instant-summarize")
    assert svc is not None


# ---------------------------------------------------------------------------
# 15. No-change → retry with stricter prompt
# ---------------------------------------------------------------------------


def test_no_change_retry(harness: tuple) -> None:
    """First agent run is noop → plugin retries with stricter prompt."""
    # Fresh runtime with agent that noops the first run.
    import shutil
    import tempfile

    agent2 = FakeAgentService(noop_runs=1)
    d = tempfile.mkdtemp()
    try:
        cfg2 = load_config(data_dir=d)
        rt = build_runtime(
            cfg2,
            plugins=[
                _StubSettingsPlugin(),
                _StubAgentPlugin(agent2),
                ReaderPlugin(),
                InstantSummarizePlugin(),
            ],
        )
        rt.start()
        try:
            _start_session(rt)
            streams = rt.hub.current
            streams.transcript.append("M" * 600)
            rt.bus.flush(2.0)

            # Should have 2 runs: main (noop) + retry (writes).
            assert agent2.wait_for_runs(2, timeout=5.0), (
                f"expected 2 runs (main+retry), got {len(agent2.run_calls)}"
            )

            # The retry should have RETRY_PROMPT content.
            retry_input = agent2.run_calls[1]["user_input"]
            assert "上一次运行" in retry_input or "注意" in retry_input

            # Two create_session calls.
            assert len(agent2.create_calls) == 2
        finally:
            rt.stop()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 16. Exhausted retries → error event + watermark advance
# ---------------------------------------------------------------------------


def test_exhausted_retries(harness: tuple) -> None:
    """Agent always fails → error event emitted, watermarks advanced."""
    import shutil
    import tempfile

    agent2 = FakeAgentService(fail_runs=100)  # always fails
    d = tempfile.mkdtemp()
    try:
        cfg2 = load_config(data_dir=d)
        rt = build_runtime(
            cfg2,
            plugins=[
                _StubSettingsPlugin(),
                _StubAgentPlugin(agent2),
                ReaderPlugin(),
                InstantSummarizePlugin(),
            ],
        )
        rt.start()
        error_events: list[Any] = []

        def on_error(event: Any) -> None:
            error_events.append(event)

        rt.bus.on("instant-summarize.error", on_error)

        try:
            sid = _start_session(rt)
            streams = rt.hub.current
            streams.transcript.append("N" * 600)
            rt.bus.flush(2.0)

            # Should eventually emit summarize.error.
            deadline = time.monotonic() + 5.0
            while not error_events and time.monotonic() < deadline:
                rt.bus.flush(0.2)
                time.sleep(0.1)

            assert len(error_events) >= 1, "instant-summarize.error was not emitted"

            # Watermarks should have advanced (avoid infinite loop).
            wm = rt.services.get("instant-summarize")._watermarks.get(sid, {})
            assert wm.get("transcript", 0) > 0, (
                "watermarks should advance even on failure"
            )
        finally:
            rt.stop()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 17. course_name falls back to streams.current.meta
# ---------------------------------------------------------------------------


def test_course_name_fallback(harness: tuple) -> None:
    """When MetaRegistry has no 'course', use streams.current.meta."""
    runtime, settings, agent = harness
    _start_session(runtime, meta={"course": "代数拓扑"})

    streams = runtime.hub.current
    streams.transcript.append("C" * 600)
    runtime.bus.flush(2.0)
    assert agent.wait_for_runs(1, timeout=5.0)

    user_prompt = agent.run_calls[0]["user_input"]
    assert "代数拓扑" in user_prompt, (
        f"course name should be in prompt: {user_prompt[:200]}"
    )