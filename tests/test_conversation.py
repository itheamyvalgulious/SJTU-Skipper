from __future__ import annotations

from config import load_config
from core.runtime import build_runtime
from plugins.conversation.plugin import ConversationPlugin
from plugins.reader.plugin import ReaderPlugin
from plugins.settings.plugin import SettingsPlugin


def _runtime(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    runtime = build_runtime(
        cfg,
        plugins=[SettingsPlugin(), ReaderPlugin(), ConversationPlugin()],
    )
    runtime.start()
    return runtime


def test_conversation_messages_and_pages_survive_restart(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        service = runtime.services.require("conversation")
        conversation_id = service.create(
            "ask",
            "恢复测试",
            {"session_id": "class-1", "course": "课程", "session_title": "第一节"},
        )
        service.append(conversation_id, "user", "问题")
        service.append(conversation_id, "assistant", "回答")
        assert runtime.ui.has_page(f"conversation:{conversation_id}")
    finally:
        runtime.stop()

    restored = _runtime(tmp_path)
    try:
        service = restored.services.require("conversation")
        conversation = service.get(conversation_id)
        assert conversation is not None
        assert [row["text"] for row in conversation.messages()] == ["问题", "回答"]
        assert service.for_class("class-1")[0]["id"] == conversation_id
        assert restored.ui.has_page(f"conversation:{conversation_id}")
        assert service.delete(conversation_id)
        assert service.get(conversation_id) is None
        assert not restored.ui.has_page(f"conversation:{conversation_id}")
    finally:
        restored.stop()
