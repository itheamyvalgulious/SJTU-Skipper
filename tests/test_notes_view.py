from types import SimpleNamespace
import pytest
from plugins.notes.view import NotesView


def test_stop_policy_is_session_bound():
    calls = []
    plugin = NotesView()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    plugin.settings = SimpleNamespace(get=lambda key, default: "direct")
    old = SimpleNamespace(meta={})
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(get_session=lambda sid: old))
    plugin._stopped({"session_id": "old-course"})
    assert calls == ["old-course"]


def test_fill_keeps_class_identity():
    calls = []
    plugin = NotesView()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    fill = SimpleNamespace(meta={"fill_for": "course"})
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(get_session=lambda sid: fill))
    plugin._stopped({"session_id": "course", "meta": fill.meta})
    assert calls == ["course"]


def test_stale_stop_does_not_finalize_replacement():
    calls = []
    plugin = NotesView()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            get_session=lambda sid: SimpleNamespace(started_at=2, meta={})
        )
    )
    plugin._stopped({"session_id": "course", "started_at": 1, "meta": {}})
    assert calls == []


def test_skip_preserves_preview():
    calls = []
    plugin = NotesView()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    plugin.settings = SimpleNamespace(get=lambda key, default: "skip")
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(get_session=lambda sid: SimpleNamespace(meta={}))
    )
    plugin._stopped({"session_id": "course"})
    assert calls == []


def test_log_pages_keep_their_session():
    from core.ui import UIManager
    from tests.support import FakeSettings

    ui = UIManager()
    streams = SimpleNamespace(selected_class_id="first")
    states = {
        sid: {
            "session_id": sid,
            "generation_state": "finished",
            "tasks": [{"agents": {"main": {"role": "main", "label": sid}}}],
            "error": "generation failed" if sid == "first" else "",
            "errors": [{"error": "source failed"}, {"error": "source failed"}]
            if sid == "first"
            else [],
        }
        for sid in ("first", "second")
    }
    services = {
        "settings": FakeSettings(),
        "notes": SimpleNamespace(status=states.__getitem__),
    }
    plugin = NotesView()
    plugin.setup(
        SimpleNamespace(
            streams=streams,
            ui=ui,
            require=services.__getitem__,
            register_service=lambda *args: None,
            events=SimpleNamespace(on=lambda *args, **kwargs: None),
        ),
        services["notes"],
    )
    try:
        result = ui.dispatch_action("notes.logs", {})
        first_page = result.open_page
        ui.open_page(first_page)
        streams.selected_class_id = "second"
        second_page = ui.dispatch_action("notes.logs", {}).open_page
        ui.open_page(second_page)
        assert second_page != first_page
        states["first"]["tasks"][0]["agents"]["main"]["label"] = "updated first"
        ui.open_page(first_page)
        first = ui.state()["page_content"][first_page]
        assert first[0]["agents"][0]["label"] == "updated first"
        assert all(block["type"] == "agent_activity" for block in first)
        ui.open_page(second_page)
        assert (
            ui.state()["page_content"][second_page][0]["agents"][0]["label"] == "second"
        )
        assert ui.dispatch_action("notes.logs", {}).open_page == second_page
    finally:
        plugin.teardown()
    assert not ui.has_page(first_page) and not ui.has_page(second_page)


@pytest.mark.parametrize("selected", ["", "boot"])
def test_empty_selection_hides_notes(selected):
    plugin = NotesView()
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            selected_class_id=selected,
            current=SimpleNamespace(session_id="processing"),
        )
    )
    assert plugin._render() == []
    for action in ("_generate", "_stop", "_delete", "_read", "_logs", "_fill_again"):
        assert not getattr(plugin, action)({})["ok"]


@pytest.mark.parametrize("target", ["", "boot"])
def test_invalid_action_target_does_not_use_selection(target):
    plugin = NotesView()
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(selected_class_id="selected"))
    for action in ("_generate", "_stop", "_delete", "_read", "_logs", "_fill_again"):
        assert not getattr(plugin, action)({"session_id": target})["ok"]


def test_generation_uses_selected_or_explicit_class():
    calls = []
    plugin = NotesView()
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            selected_class_id="selected",
            current=SimpleNamespace(session_id="processing"),
        )
    )
    plugin.pipeline = SimpleNamespace(start=lambda sid: calls.append(sid) or True)
    assert plugin._generate({})["ok"]
    plugin.ctx.streams.selected_class_id = ""
    assert plugin._generate({"session_id": "task-page"})["ok"]
    assert calls == ["selected", "task-page"]


def test_sidebar_status_uses_selected_class():
    from tests.support import FakeSettings

    calls = []
    plugin = NotesView()
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            selected_class_id="selected",
            current=SimpleNamespace(session_id="processing"),
        )
    )
    plugin.settings = FakeSettings()
    plugin.notices = {}
    plugin.pipeline = SimpleNamespace(
        status=lambda sid: calls.append(sid) or {"generation_state": "not_started"}
    )
    assert plugin._render()
    assert calls == ["selected"]
