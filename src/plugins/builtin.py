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
    from .conversation.plugin import ConversationPlugin
    from .input.plugin import InputPlugin
    from .learning.plugin import LearningPlugin
    from .llm_core.plugin import LLMPlugin
    from .notes import NotesPlugin
    from .preference.plugin import PreferencePlugin
    from .note_check_stc import NoteCheckSTCPlugin
    from .reader.plugin import ReaderPlugin
    from .settings.plugin import SettingsPlugin
    from .teaching.plugin import TeachingPlugin
    from .stt import STTPlugin
    from .vlm.plugin import VLMPlugin

    plugins: list["Plugin"] = [
        SettingsPlugin(),
        LLMPlugin(),
        AgentPlugin(),
        ReaderPlugin(),
        ConversationPlugin(),
        AskPlugin(),
    ]
    if config.enable_input:
        plugins.append(InputPlugin())
    if config.enable_stt:
        plugins.append(STTPlugin())
    if config.enable_vlm:
        plugins.append(VLMPlugin())
    if config.enable_notes:
        plugins.append(NotesPlugin())
        plugins.append(NoteCheckSTCPlugin())
    plugins.append(PreferencePlugin())
    plugins.append(LearningPlugin())
    plugins.append(TeachingPlugin())
    return plugins
