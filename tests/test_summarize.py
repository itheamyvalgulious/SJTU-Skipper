from types import SimpleNamespace
from plugins.summarize.plugin import SummarizePlugin


def test_stop_policy_is_session_bound():
    calls = []
    plugin = SummarizePlugin()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    plugin.settings = SimpleNamespace(get=lambda key, default: "direct")
    old = SimpleNamespace(meta={})
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(get_session=lambda sid: old))
    plugin._stopped({"session_id": "old-course"})
    assert calls == ["old-course"]


def test_fill_requests_original_course():
    calls = []
    plugin = SummarizePlugin()
    plugin.pipeline = SimpleNamespace(finalize=calls.append)
    fill = SimpleNamespace(meta={"fill_for": "original-course"})
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(get_session=lambda sid: fill))
    plugin._stopped({"session_id": "fill-course"})
    assert calls == ["original-course"]


def test_skip_preserves_preview():
    calls = []
    plugin = SummarizePlugin()
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
    streams = SimpleNamespace(notes_session_id="first")
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
        "notes-pipeline": SimpleNamespace(status=states.__getitem__),
    }
    plugin = SummarizePlugin()
    plugin.setup(
        SimpleNamespace(
            streams=streams,
            ui=ui,
            require=services.__getitem__,
            register_service=lambda *args: None,
            events=SimpleNamespace(on=lambda *args, **kwargs: None),
        )
    )
    try:
        result = ui.dispatch_action("summarize.logs", {})
        first_page = result.open_page
        ui.open_page(first_page)
        streams.notes_session_id = "second"
        second_page = ui.dispatch_action("summarize.logs", {}).open_page
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
        assert ui.dispatch_action("summarize.logs", {}).open_page == second_page
    finally:
        plugin.teardown()
    assert not ui.has_page(first_page) and not ui.has_page(second_page)
