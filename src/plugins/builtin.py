"""Built-in plugin composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import AppConfig

if TYPE_CHECKING:  # pragma: no cover
    from core.plugin import Plugin


def builtin_plugins(config: AppConfig) -> list["Plugin"]:
    """Instantiate the default plugin graph for a runtime.

    Imports are intentionally local so a partially installed development tree
    can still run isolated plugin tests.
    """

    from .agent.plugin import AgentPlugin
    from .ask.plugin import AskPlugin
    from .input.plugin import InputPlugin
    from .instant_summarize.plugin import InstantSummarizePlugin
    from .llm_core.plugin import LLMPlugin
    from .reader.plugin import ReaderPlugin
    from .settings.plugin import SettingsPlugin
    from .stt.plugin import STTPlugin
    from .summarize.plugin import SummarizePlugin
    from .vlm.plugin import VLMPlugin

    plugins: list["Plugin"] = [
        SettingsPlugin(),
        LLMPlugin(),
        AgentPlugin(),
        ReaderPlugin(),
        AskPlugin(),
    ]
    if config.enable_input:
        plugins.append(InputPlugin())
    if config.enable_stt:
        plugins.append(STTPlugin())
    if config.enable_vlm:
        plugins.append(VLMPlugin())
    if config.enable_summarize:
        plugins.append(SummarizePlugin())
    if config.enable_instant_summarize:
        plugins.append(InstantSummarizePlugin())
    return plugins
