from types import SimpleNamespace

from core.ui import UIManager
from plugins.notes.pipeline import NotesPipeline
from plugins.settings.store import SettingsService, SettingsStore


def test_pipeline_registers_generation_settings(tmp_path):
    settings = SettingsService(SettingsStore(tmp_path / "settings.json"))
    services = {"settings": settings, "agent": SimpleNamespace()}
    ctx = SimpleNamespace(
        data_dir=tmp_path,
        require=services.__getitem__,
        register_service=services.__setitem__,
        events=SimpleNamespace(on=lambda *args, **kwargs: None),
        ui=UIManager(),
    )
    pipeline = NotesPipeline()
    pipeline.setup(ctx)
    try:
        settings.set("notes.chunks_enabled", False)
        settings.set("notes.review_max_rounds", 3)
        assert pipeline.setting("notes.chunks_enabled", True) is False
        assert pipeline.setting("notes.review_max_rounds", 0) == 3
    finally:
        pipeline.teardown()


def test_notes_settings_share_one_form(tmp_path):
    from config import load_config
    from core.plugin import Plugin
    from core.runtime import build_runtime
    from plugins.notes.plugin import NotesPlugin
    from plugins.reader.plugin import ReaderPlugin
    from plugins.settings.plugin import SettingsPlugin

    class AgentPlugin(Plugin):
        id = "agent"

        def setup(self, ctx):
            ctx.register_service("agent", SimpleNamespace())

    runtime = build_runtime(
        load_config(data_dir=tmp_path),
        plugins=[SettingsPlugin(), AgentPlugin(), ReaderPlugin(), NotesPlugin()],
    )
    runtime.start()
    try:
        settings = runtime.services.require("settings")
        settings.set("notes.chunks_enabled", False)
        settings.set("notes.on_stop", "skip")
        runtime.ui.open_page("settings:main")
        blocks = runtime.ui.state()["page_content"]["settings:main"]
        groups = blocks[0]["tabs"]
        assert [group["title"] for group in groups] == ["笔记"]
        forms = groups[0]["blocks"]
        assert len(forms) == 1
        form = forms[0]
        fields = {field["name"]: field for field in form["fields"]}
        assert set(fields) == {spec.key for spec in settings.specs()}
        assert fields["notes.chunks_enabled"]["value"] is False
        assert fields["notes.on_stop"]["value"] == "skip"

        values = {name: field["value"] for name, field in fields.items()}
        values.update(
            {
                "notes.chunks_enabled": True,
                "notes.on_stop": "direct",
                "notes.chunk_system_prompt": "chunk instructions",
                "notes.system_prompt": "writer instructions",
                "notes.reviewer_prompt": "review instructions",
            }
        )
        result = runtime.ui.dispatch_action(
            form["submit"]["action"], {"values": values}
        )
        assert result.ok
        saved = SettingsService(SettingsStore(tmp_path / "settings.json"))
        for key, value in values.items():
            assert saved.get(key) == value
    finally:
        runtime.stop()
