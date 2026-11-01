"""Regression: settings changes must reach the live LLM provider."""

from __future__ import annotations

from pathlib import Path

from config import load_config
from core.runtime import build_runtime
from plugins.llm_core.plugin import LLMPlugin
from plugins.settings.plugin import SettingsPlugin


def test_provider_reconfigures_without_restart(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(cfg, plugins=[SettingsPlugin(), LLMPlugin()])
    runtime.start()
    try:
        service = runtime.services.require("llm")
        provider = service.registry.require("openai")
        settings = runtime.services.require("settings")

        settings.set("llm.base_url", "http://127.0.0.1:9/v1")
        settings.set("llm.model", "test-model")
        service._provider_for(service.config_for("agent"))  # noqa: SLF001
        assert str(provider._client.base_url).rstrip("/").endswith(":9/v1")  # noqa: SLF001
        assert provider.config.model == "test-model"
    finally:
        runtime.stop()
