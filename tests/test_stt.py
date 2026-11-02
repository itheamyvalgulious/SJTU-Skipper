"""Tests for the intentionally empty STT placeholder.

There is no speech-recognition backend and no ``stt.*`` setting.  The plugin
must let the audio stream run without emitting text or blocking the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.events import EventBus
from core.meta import MetaRegistry
from core.plugin import PluginContext
from core.services import ServiceRegistry
from core.streams import StreamHub
from core.ui import UIManager
from plugins.stt import STTPlugin, STTService, SttPlugin
from plugins.stt.plugin import ITEM_ID, PAGE_ID, TAB_ID


def make_ctx(tmp_path: Path) -> PluginContext:
    bus = EventBus()
    hub = StreamHub(emit=bus.emit)
    return PluginContext(
        plugin_id="stt",
        data_dir=tmp_path,
        events=bus,
        streams=hub,
        llm=None,
        ui=UIManager(),
        services=ServiceRegistry(),
        meta=MetaRegistry(),
        logger=logging.getLogger("test_stt"),
    )


def test_plugin_is_empty_and_has_no_settings() -> None:
    assert SttPlugin.id == "stt"
    assert SttPlugin.requires == ()
    assert SttPlugin.provides == ("stt",)
    assert STTPlugin is SttPlugin


def test_setup_registers_service_and_ui(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    plugin = SttPlugin()
    plugin.setup(ctx)
    try:
        service = ctx.services.require("stt")
        assert isinstance(service, STTService)
        status = service.status()
        assert status["state"] == "unavailable"
        assert status["backend"] == "none"
        assert status["error"] == "未接入 STT 后端"
        assert status["segments"] == 0
        assert status["chars"] == 0

        snap = ctx.ui.snapshot()
        assert snap["tabs"][TAB_ID] == "left"
        assert ITEM_ID in snap["items"]
        assert PAGE_ID in snap["pages"]
    finally:
        plugin.teardown()


def test_audio_flows_but_no_text_is_emitted(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    plugin = SttPlugin()
    plugin.setup(ctx)
    try:
        ctx.streams.current.audio.append(b"\x00" * 32000)  # 1 s s16le mono
        status = ctx.services.require("stt").status()
        assert status["audio_seconds"] == pytest.approx(1.0)
        assert ctx.streams.current.transcript.items_since(0) == []
        assert status["segments"] == 0
        assert status["chars"] == 0
    finally:
        plugin.teardown()


def test_no_stt_settings_are_registered(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    plugin = SttPlugin()
    plugin.setup(ctx)
    try:
        # The STT plugin does not require or touch the settings service.
        assert ctx.services.get("settings") is None
    finally:
        plugin.teardown()
