"""Tests for the STT plugin: WAV/windowing, backends, worker and UI.

All backends are faked or scripted; nothing here touches the network and no
worker is left running between tests.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.events import EventBus
from core.meta import MetaRegistry
from core.plugin import PluginContext
from core.services import ServiceRegistry
from core.streams import StreamHub
from core.ui import UIManager
from plugins.stt import (
    FasterWhisperBackend,
    OpenAITranscriptionBackend,
    STTBackendError,
    STTResult,
    STTSegment,
    STTUnavailableError,
    compute_windows,
    create_backend,
    dedupe_overlap,
    pcm_to_wav,
    transcription_endpoint,
    wav_duration,
)
from plugins.stt import backends as backends_module
from plugins.stt.audio import common_overlap
from plugins.stt.plugin import (
    EVENT_ERROR,
    EVENT_SEGMENT,
    ITEM_ID,
    PAGE_ID,
    TAB_ID,
    SttPlugin,
)

from support import FakeLLM, FakeSettings

SAMPLE_RATE = 16000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pcm(seconds: float, *, sample_rate: int = SAMPLE_RATE, fill: int = 0) -> bytes:
    """Deterministic mono s16le PCM of the requested duration."""

    frames = int(round(seconds * sample_rate))
    sample = int(fill).to_bytes(2, "little", signed=True)
    return sample * frames


class FakeSTTBackend:
    """Test-only scripted backend; production has no mock backend."""

    name = "fake"

    def __init__(self, *, script: list[str] | None = None) -> None:
        self.script = list(script or [])
        self.calls = 0

    def transcribe(self, wav_bytes: bytes, *, language=None, model=None) -> STTResult:
        self.calls += 1
        duration = wav_duration(wav_bytes) if wav_bytes else 0.0
        if self.script:
            text = self.script[(self.calls - 1) % len(self.script)]
        else:
            text = f"（测试转写）第 {self.calls} 段，约 {duration:.1f} 秒。"
        return STTResult(
            text=text,
            segments=[STTSegment(start=0.0, end=round(duration, 3), text=text)],
            raw={"provider": self.name, "duration": duration, "call": self.calls},
        )


class ConfigLLM:
    """LLM stand-in exposing only ``config_for`` (used for fallbacks)."""

    def __init__(self, *, base_url: str = "", api_key: str = "", model: str = "cfg-model") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.profiles: list[str] = []

    def config_for(self, profile: str) -> SimpleNamespace:
        self.profiles.append(profile)
        return SimpleNamespace(
            profile=profile,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout=10,
            max_retries=0,
        )


def make_env(
    tmp_path,
    values: dict[str, Any] | None = None,
    *,
    llm: Any | None = None,
):
    bus = EventBus()
    hub = StreamHub(emit=bus.emit)
    ui = UIManager()
    services = ServiceRegistry()
    settings = FakeSettings(dict(values or {}))
    llm = llm if llm is not None else FakeLLM()
    services.register("settings", settings, plugin_id="settings")
    services.register("llm", llm, plugin_id="llm")
    ctx = PluginContext(
        plugin_id="stt",
        data_dir=tmp_path,
        events=bus,
        streams=hub,
        llm=llm,
        ui=ui,
        services=services,
        meta=MetaRegistry(),
        logger=logging.getLogger("lecture_agent.test.stt"),
    )
    return SimpleNamespace(
        ctx=ctx,
        settings=settings,
        hub=hub,
        bus=bus,
        ui=ui,
        services=services,
        llm=llm,
    )


def capture_events(bus: EventBus) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    bus.on(EVENT_SEGMENT, lambda event: segments.append(dict(event.payload)), plugin_id="test")
    bus.on(EVENT_ERROR, lambda event: errors.append(dict(event.payload)), plugin_id="test")
    return segments, errors


BASE_SETTINGS: dict[str, Any] = {
    "stt.provider": "openai",
    "stt.base_url": "http://fake/v1",
    "stt.api_key": "fake-key",
    "stt.chunk_seconds": 1.0,
    "stt.overlap": 0.2,
    "stt.min_seconds": 0.4,
}


def make_plugin(env, **kwargs: Any) -> SttPlugin:
    settings: dict[str, Any] = dict(BASE_SETTINGS)
    settings.update(kwargs.pop("settings", {}))
    for key, value in settings.items():
        env.settings.set(key, value)
    if "backend_factory" not in kwargs:
        kwargs["backend_factory"] = lambda cfg: FakeSTTBackend()
    plugin = SttPlugin(auto_start=False, **kwargs)
    plugin.setup(env.ctx)
    return plugin


# ----------------------------------------------------------------------
# audio.py
# ----------------------------------------------------------------------
def test_pcm_to_wav_roundtrip() -> None:
    data = pcm(0.5)
    wav = pcm_to_wav(data)

    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.getnframes() == SAMPLE_RATE // 2
        assert reader.readframes(reader.getnframes()) == data

    assert wav_duration(wav) == pytest.approx(0.5)
    assert wav_duration(pcm_to_wav(b"")) == 0.0
    with pytest.raises(TypeError):
        pcm_to_wav("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid wav"):
        wav_duration(b"definitely not a wav")
    with pytest.raises(ValueError, match="empty wav"):
        wav_duration(b"")


def test_compute_windows_covers_timeline_without_slivers() -> None:
    windows = compute_windows(10.0, 4.0, 1.0)
    assert [(w.start, w.end) for w in windows] == [(0.0, 4.0), (3.0, 7.0), (6.0, 10.0)]
    assert all(w.duration == pytest.approx(4.0) for w in windows)

    windows = compute_windows(10.0, 4.0, 0.0)
    assert [(w.start, w.end) for w in windows] == [(0.0, 4.0), (4.0, 8.0), (8.0, 10.0)]

    windows = compute_windows(2.0, 4.0, 0.5)
    assert [(w.start, w.end) for w in windows] == [(0.0, 2.0)]

    assert compute_windows(0.0, 4.0) == []
    assert compute_windows(-3.0, 4.0) == []

    with pytest.raises(ValueError, match="chunk_seconds"):
        compute_windows(5.0, 0.0)
    with pytest.raises(ValueError, match="overlap"):
        compute_windows(5.0, 1.0, 1.0)


def test_dedupe_overlap_longest_prefix_suffix() -> None:
    assert common_overlap("hello world", "world peace") == 5
    assert dedupe_overlap("hello world", "world peace") == "peace"

    assert dedupe_overlap("你好世界再见", "世界再见同学们") == "同学们"
    # Fully repeated window -> nothing new.
    assert dedupe_overlap("我们今天上课", "我们今天上课") == ""
    # Too short to be trusted -> keep everything.
    assert dedupe_overlap("甲乙丙", "丙丁") == "丙丁"
    assert dedupe_overlap("", "第一句") == "第一句"
    assert dedupe_overlap("第一句", "") == ""
    # Explicit min_overlap lets short overlaps through.
    assert dedupe_overlap("上课啦", "上课啦", min_overlap=3) == ""


# ----------------------------------------------------------------------
# backends.py
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/audio/transcriptions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/audio/transcriptions"),
        (
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/audio/transcriptions",
        ),
        (
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/audio/transcriptions",
        ),
        ("https://api.openai.com", "https://api.openai.com/v1/audio/transcriptions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/audio/transcriptions"),
        (
            "https://host/v1/audio/transcriptions",
            "https://host/v1/audio/transcriptions",
        ),
    ],
)
def test_transcription_endpoint_derivation(base_url: str, expected: str) -> None:
    assert transcription_endpoint(base_url) == expected


def test_transcription_endpoint_requires_base_url() -> None:
    with pytest.raises(STTUnavailableError, match="base_url"):
        transcription_endpoint("   ")


def test_openai_backend_parses_verbose_segments() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "text": "你好世界",
                "segments": [
                    {"start": 0.0, "end": 1.2, "text": "你好"},
                    {"start": 1.2, "end": 2.0, "text": "世界"},
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    backend = OpenAITranscriptionBackend(
        base_url="http://test/v1", api_key="secret-key", model="whisper-1", client=client
    )
    result = backend.transcribe(pcm_to_wav(pcm(2.0)), language="zh", model="whisper-1")

    assert result.text == "你好世界"
    assert [segment.text for segment in result.segments] == ["你好", "世界"]
    assert result.segments[1].end == pytest.approx(2.0)

    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "http://test/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer secret-key"
    body = request.content
    assert b'name="model"' in body
    assert b"whisper-1" in body
    assert b'name="language"' in body
    assert b'name="file"' in body
    assert b"RIFF" in body
    client.close()


def test_openai_backend_falls_back_to_plain_json() -> None:
    formats: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        formats.append(request.content)
        if b"verbose_json" in request.content:
            return httpx.Response(422, text="response_format unsupported")
        return httpx.Response(200, json={"text": "只有文本"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    backend = OpenAITranscriptionBackend(base_url="http://test/v1", api_key="k", client=client)
    result = backend.transcribe(pcm_to_wav(pcm(1.0)))

    assert result.text == "只有文本"
    assert result.segments == []
    assert len(formats) == 2
    client.close()


def test_openai_backend_surfaces_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream boom")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    backend = OpenAITranscriptionBackend(base_url="http://test/v1", api_key="k", client=client)
    with pytest.raises(STTBackendError, match="500"):
        backend.transcribe(pcm_to_wav(pcm(1.0)))
    client.close()


def test_faster_whisper_missing_dependency_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    def raiser() -> Any:
        raise STTUnavailableError("未安装 faster-whisper，请安装可选依赖")

    monkeypatch.setattr(backends_module, "_import_faster_whisper", raiser)
    backend = FasterWhisperBackend(model="base")
    with pytest.raises(STTUnavailableError, match="faster-whisper"):
        backend.transcribe(pcm_to_wav(pcm(1.0)))


def test_faster_whisper_maps_segments_with_injected_model() -> None:
    class FakeSegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class FakeInfo:
        duration = 2.0

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def transcribe(self, handle, *, language=None, vad_filter=True):
            self.calls.append({"language": language, "vad_filter": vad_filter})
            assert handle.read(4) == b"RIFF"
            return iter([FakeSegment(0.0, 1.0, " 你好 "), FakeSegment(1.0, 2.0, "世界")]), FakeInfo()

    engine = FakeEngine()
    backend = FasterWhisperBackend(model="small", model_factory=lambda name: engine)
    result = backend.transcribe(pcm_to_wav(pcm(2.0)), language="zh", model="tiny")

    assert result.text == "你好世界"
    assert [segment.text for segment in result.segments] == ["你好", "世界"]
    assert result.raw["model"] == "tiny"
    assert engine.calls == [{"language": "zh", "vad_filter": True}]


def test_fake_backend_uses_duration_and_script() -> None:
    backend = FakeSTTBackend()
    first = backend.transcribe(pcm_to_wav(pcm(1.0)))
    second = backend.transcribe(pcm_to_wav(pcm(2.5)))

    assert "1.0" in first.text
    assert "2.5" in second.text
    assert first.text != second.text
    assert first.segments[0].end == pytest.approx(1.0)
    assert second.segments[0].end == pytest.approx(2.5)

    scripted = FakeSTTBackend(script=["甲", "乙"])
    wav = pcm_to_wav(pcm(0.5))
    assert [scripted.transcribe(wav).text for _ in range(3)] == ["甲", "乙", "甲"]


def test_create_backend_validates_configuration() -> None:
    with pytest.raises(STTUnavailableError, match="base_url"):
        create_backend("openai", base_url="", api_key="k")
    with pytest.raises(STTUnavailableError, match="API Key"):
        create_backend("openai", base_url="http://x/v1", api_key="")
    with pytest.raises(STTUnavailableError, match="未知 STT provider"):
        create_backend("nope")
    with pytest.raises(STTUnavailableError, match="none"):
        create_backend("none")
    assert isinstance(
        create_backend("faster-whisper", model="base"), FasterWhisperBackend
    )


# ----------------------------------------------------------------------
# plugin: registration, config, worker
# ----------------------------------------------------------------------
def test_plugin_metadata_and_registration(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(env)

    assert plugin.id == "stt"
    assert plugin.requires == ("settings", "llm")

    specs = {spec.key: spec for spec in env.settings.specs()}
    assert set(specs) == {
        "stt.enabled",
        "stt.provider",
        "stt.model",
        "stt.language",
        "stt.chunk_seconds",
        "stt.overlap",
        "stt.min_seconds",
        "stt.base_url",
        "stt.api_key",
    }
    assert specs["stt.provider"].options == ("none", "openai", "faster-whisper")
    assert specs["stt.api_key"].secret is True
    assert specs["stt.api_key"].type == "secret"

    service = env.services.require("stt")
    assert service.status()["provider"] == "openai"
    assert service.backends() == ["none", "openai", "faster-whisper"]
    assert env.ctx.meta.snapshot()["stt"]["state"] in {"idle", "running"}

    snapshot = env.ui.snapshot()
    assert snapshot["tabs"][TAB_ID] == "left"
    assert snapshot["items"][ITEM_ID] == TAB_ID
    assert snapshot["pages"][PAGE_ID] == "实时转写"
    plugin.teardown()


def test_existing_classroom_tab_is_reused(tmp_path) -> None:
    env = make_env(tmp_path)
    env.ui.add_tab("left", TAB_ID, "课堂", plugin_id="other")
    plugin = make_plugin(env)

    assert env.ui.snapshot()["tabs"][TAB_ID] == "left"
    plugin.teardown()
    snapshot = env.ui.snapshot()
    assert ITEM_ID not in snapshot["items"]
    assert PAGE_ID not in snapshot["pages"]
    # Another plugin owns the tab, so it must survive our teardown.
    assert snapshot["tabs"][TAB_ID] == "left"


def test_pump_consumes_audio_with_chunk_and_overlap(tmp_path) -> None:
    env = make_env(tmp_path)
    segments, errors = capture_events(env.bus)
    plugin = make_plugin(env)

    env.hub.current.audio.append(pcm(3.2))
    appended = plugin.pump()

    assert appended == 3
    items = list(env.hub.current.transcript)
    assert [item.source for item in items] == ["stt", "stt", "stt"]
    assert (items[0].start, items[0].end) == (0.0, 1.0)
    assert items[1].start == pytest.approx(0.8)
    assert items[2].start == pytest.approx(1.6)
    assert all(item.meta["model"] == "whisper-1" for item in items)
    assert all(item.meta["language"] == "zh" for item in items)

    status = plugin.status()
    assert status["cursor"] == pytest.approx(2.4)
    assert status["segments"] == 3
    assert status["state"] == "running"
    assert status["worker"] is False

    assert plugin.pump(flush=True) == 1
    assert len(env.hub.current.transcript) == 4
    assert plugin.pump() == 0
    assert plugin.pump(flush=True) == 0

    env.bus.flush(2.0)
    assert len(segments) == 4
    assert errors == []
    assert segments[0]["text"]
    assert segments[0]["start"] == 0.0
    plugin.teardown()


def test_overlap_dedupe_trims_repeated_text(tmp_path) -> None:
    env = make_env(tmp_path)
    script = ["你好，我们开始上课", "我们开始上课。今天讲群论"]
    settings = {"stt.chunk_seconds": 4.0, "stt.overlap": 1.0, "stt.min_seconds": 0.5}
    plugin = make_plugin(
        env,
        settings=settings,
        backend_factory=lambda cfg: FakeSTTBackend(script=script),
    )

    env.hub.current.audio.append(pcm(5.0))
    assert plugin.pump() == 1
    assert plugin.pump(flush=True) == 1

    items = list(env.hub.current.transcript)
    assert [item.text for item in items] == ["你好，我们开始上课", "。今天讲群论"]
    assert items[1].start == pytest.approx(3.0)
    plugin.teardown()


def test_pump_flushes_tail_when_session_ends(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(env)

    env.hub.current.audio.append(pcm(1.5))
    assert plugin.pump() == 1  # 0.0-1.0, cursor moves to 0.8

    env.hub.end_session("input finished")
    assert plugin.pump() == 1  # tail 0.8-1.5
    items = list(env.hub.current.transcript)
    assert len(items) == 2
    assert items[-1].end == pytest.approx(1.5)
    assert plugin.pump() == 0
    plugin.teardown()


def test_session_switch_flushes_old_and_resets_cursor(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(env)

    old = env.hub.current
    old.audio.append(pcm(1.5))
    assert plugin.pump() == 1

    new = env.hub.start_session(meta={"course": "algebra"})
    assert new.session_id != old.session_id
    new.audio.append(pcm(1.2))

    # One item flushes the old tail, one item for the new session.
    assert plugin.pump() == 2
    assert len(old.transcript) == 2
    assert len(new.transcript) == 1
    assert new.transcript[0].start == 0.0
    assert plugin.status()["session_id"] == new.session_id
    assert plugin.status()["cursor"] == pytest.approx(0.8)
    plugin.teardown()


def test_disabled_plugin_does_not_consume(tmp_path) -> None:
    env = make_env(tmp_path)
    segments, _errors = capture_events(env.bus)
    plugin = make_plugin(env, settings={"stt.enabled": False})

    env.hub.current.audio.append(pcm(2.0))
    assert plugin.pump() == 0
    assert plugin.pump(flush=True) == 0
    assert len(env.hub.current.transcript) == 0
    assert plugin.status()["state"] == "disabled"

    env.bus.flush(1.0)
    assert segments == []
    plugin.teardown()


def test_missing_api_key_reports_error_without_blocking(tmp_path) -> None:
    env = make_env(tmp_path, llm=ConfigLLM())
    segments, errors = capture_events(env.bus)
    plugin = make_plugin(
        env,
        settings={"stt.provider": "openai", "stt.base_url": "", "stt.api_key": ""},
    )

    env.hub.current.audio.append(pcm(2.0))
    assert plugin.pump() == 0
    status = plugin.status()
    assert status["state"] == "error"
    assert "base_url" in status["error"]

    # Repeated pumps stay quiet: same error is not re-emitted.
    assert plugin.pump() == 0
    env.bus.flush(2.0)
    assert len(errors) == 1
    assert errors[0]["error"] == status["error"]
    assert segments == []
    plugin.teardown()


def test_config_uses_explicit_stt_settings_only(tmp_path) -> None:
    env = make_env(tmp_path, llm=ConfigLLM(base_url="https://llm.example/v1", api_key="llm-key"))
    plugin = make_plugin(
        env,
        settings={
            "stt.provider": "openai",
            "stt.model": "whisper-1",
            "stt.base_url": "",
            "stt.api_key": "",
        },
    )
    cfg = plugin.config()
    assert cfg.base_url == ""
    assert cfg.api_key == ""
    assert cfg.model == "whisper-1"

    env.settings.set("stt.base_url", "https://own.example/v1")
    env.settings.set("stt.api_key", "own-key")
    cfg = plugin.config()
    assert cfg.base_url == "https://own.example/v1"
    assert cfg.api_key == "own-key"
    plugin.teardown()


def test_failures_back_off_and_keep_window(tmp_path) -> None:
    class FailingBackend:
        name = "failing"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, wav_bytes, *, language=None, model=None):
            self.calls += 1
            raise STTBackendError("后端炸了")

    backend = FailingBackend()
    env = make_env(tmp_path)
    segments, errors = capture_events(env.bus)
    plugin = make_plugin(
        env,
        backend_factory=lambda cfg: backend,
        base_backoff=0.2,
        max_backoff=1.0,
    )

    env.hub.current.audio.append(pcm(2.0))
    assert plugin.pump() == 0
    assert backend.calls == 1
    status = plugin.status()
    assert status["state"] == "error"
    assert "后端炸了" in status["error"]
    assert status["failures"] == 1

    # Backoff gate keeps the same window instead of hammering the backend.
    assert plugin.pump() == 0
    assert backend.calls == 1

    # Forced retries keep the window and surface the error; no silent skip.
    assert plugin.pump(force=True) == 0
    assert plugin.pump(force=True) == 0
    assert backend.calls == 3
    assert plugin.status()["cursor"] == pytest.approx(0.0)
    assert plugin.status()["state"] == "error"

    assert plugin.pump(force=True) == 0
    assert backend.calls == 4
    assert segments == []
    env.bus.flush(2.0)
    assert len(errors) >= 3
    assert all("后端炸了" in payload["error"] for payload in errors)
    plugin.teardown()


def test_transcribe_once_does_not_touch_streams(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(env)

    wav = pcm_to_wav(pcm(0.75))
    result = env.services.require("stt").transcribe_once(wav)

    assert result.text
    assert result.segments[0].end == pytest.approx(0.75)
    assert len(env.hub.current.transcript) == 0

    with pytest.raises(STTUnavailableError, match="未知 STT provider"):
        plugin.transcribe_once(wav, provider="nope")
    plugin.teardown()


def test_worker_thread_consumes_audio(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(env, idle_poll=0.05)
    plugin.start()
    try:
        env.hub.current.audio.append(pcm(1.2))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(env.hub.current.transcript) == 0:
            time.sleep(0.02)
        assert plugin.status()["worker"] is True
        assert len(env.hub.current.transcript) >= 1
        assert env.hub.current.transcript[0].source == "stt"
    finally:
        plugin.stop()
        plugin.teardown()
    assert plugin.running is False


def test_ui_page_shows_live_transcript(tmp_path) -> None:
    env = make_env(tmp_path)
    plugin = make_plugin(
        env,
        backend_factory=lambda cfg: FakeSTTBackend(script=["第一句：拓扑空间", "第二句：开集"]),
    )
    env.hub.current.audio.append(pcm(1.2))
    plugin.pump()
    text = env.hub.current.transcript[0].text

    env.ui.open_page(PAGE_ID)
    state = env.ui.state()
    blocks = state["page_content"][PAGE_ID]

    keyvalue = next(block for block in blocks if block["type"] == "keyvalue")
    rows = dict(keyvalue["items"])
    assert rows["状态"] == "转写中"
    assert rows["后端"] == "openai"

    markdown = next(block for block in blocks if block["type"] == "markdown")
    assert text in markdown["text"]
    assert "实时转写" in markdown["text"]

    progress = next(block for block in blocks if block["type"] == "progress")
    assert 0.0 <= progress["value"] <= 1.0

    item_blocks = state["left_tabs"][0]["toolbox"][0]["blocks"]
    assert any(block["type"] == "keyvalue" for block in item_blocks)
    plugin.teardown()


def test_ui_page_reports_error_state(tmp_path) -> None:
    env = make_env(tmp_path, llm=ConfigLLM())
    plugin = make_plugin(
        env,
        settings={"stt.provider": "openai", "stt.base_url": "", "stt.api_key": ""},
    )
    env.ui.open_page(PAGE_ID)
    state = env.ui.state()
    blocks = state["page_content"][PAGE_ID]
    html = [block for block in blocks if block["type"] == "html"]
    assert html and "缺少 STT base_url" in html[0]["html"]
    plugin.teardown()


def test_plugin_loads_through_plugin_manager(tmp_path) -> None:
    from config import load_config
    from core.plugin import Plugin
    from core.runtime import build_runtime

    class SettingsStub(Plugin):
        id = "settings"

        def setup(self, ctx) -> None:
            ctx.register_service(
                "settings",
                FakeSettings(
                    {
                        "stt.provider": "none",
                        "stt.chunk_seconds": 1.0,
                        "stt.overlap": 0.2,
                        "stt.min_seconds": 0.4,
                    }
                ),
            )

    class LlmStub(Plugin):
        id = "llm"

        def setup(self, ctx) -> None:
            # ``build_runtime(llm=...)`` already registered the fake service.
            assert ctx.services.get("llm") is not None

    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(
        cfg,
        plugins=[SttPlugin(auto_start=False), SettingsStub(), LlmStub()],
        llm=FakeLLM(),
    )
    runtime.start()
    try:
        order = runtime.plugins.order
        assert order.index("settings") < order.index("stt")
        assert order.index("llm") < order.index("stt")
        assert runtime.services.get("stt") is not None

        runtime.hub.current.audio.append(pcm(1.2))
        plugin = runtime.plugins.get("stt")
        assert isinstance(plugin, SttPlugin)
        # Provider "none" consumes nothing and must not emit text.
        assert plugin.pump() == 0
        assert len(runtime.hub.current.transcript) == 0
    finally:
        runtime.stop()
    assert runtime.services.get("stt") is None
