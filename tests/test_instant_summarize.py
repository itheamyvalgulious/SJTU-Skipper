from types import SimpleNamespace
import pytest
from plugins.instant_summarize.plugin import (
    InstantSummarizeService,
    InstantSummarizePlugin,
)


def test_service_uses_pipeline_files(tmp_path):
    expected = {"knowledge": tmp_path / "instant_note_knowledge.md"}
    pipeline = SimpleNamespace(
        draft_paths=lambda sid: expected,
        note_files=lambda kind: {"knowledge": (f"{kind}.md", "知识")},
        running_tasks=lambda: [],
    )
    service = InstantSummarizeService(pipeline)
    assert service.draft_paths("course") == expected
    assert service.note_files()["knowledge"][0] == "instant.md"


def test_toggle_retains_source_records():
    values = {}
    plugin = InstantSummarizePlugin()
    plugin.settings = SimpleNamespace(
        set=lambda key, value: values.update({key: value})
    )
    plugin.ctx = SimpleNamespace(ui=SimpleNamespace(refresh=lambda event: None))
    assert plugin._toggle({"enabled": False})["ok"]
    assert values["instant-summarize.enabled"] is False


@pytest.mark.parametrize("selected", ["", "boot"])
def test_empty_selection_hides_notes(selected):
    plugin = InstantSummarizePlugin()
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            selected_class_id=selected,
            current=SimpleNamespace(session_id="processing"),
        )
    )
    assert plugin._render() == []
    assert not plugin._read({})["ok"]


@pytest.mark.parametrize("target", ["", "boot"])
def test_invalid_read_target_does_not_use_selection(target):
    plugin = InstantSummarizePlugin()
    plugin.ctx = SimpleNamespace(streams=SimpleNamespace(selected_class_id="selected"))
    assert not plugin._read({"session_id": target})["ok"]


def test_notes_use_selected_or_explicit_class(tmp_path):
    reads = []
    statuses = []
    reader = SimpleNamespace(show=lambda **kwargs: reads.append(kwargs) or "note-page")
    plugin = InstantSummarizePlugin()
    plugin.ctx = SimpleNamespace(
        data_dir=tmp_path,
        streams=SimpleNamespace(
            selected_class_id="selected",
            current=SimpleNamespace(session_id="processing"),
        ),
        require=lambda name: reader,
    )
    plugin.settings = SimpleNamespace(get=lambda key, default: default)
    plugin.pipeline = SimpleNamespace(
        status=lambda sid: statuses.append(sid) or {"errors": [], "published_end": 12},
        note_files=lambda kind: {"knowledge": ("instant_note_knowledge.md", "知识")},
    )
    assert plugin._render()
    assert statuses == ["selected"]
    assert plugin._read({})["ok"]
    plugin.ctx.streams.selected_class_id = ""
    assert plugin._read({"session_id": "task-page"})["ok"]
    assert [read["page_key"] for read in reads] == [
        "selected:instant-summarize",
        "task-page:instant-summarize",
    ]
    assert all("processing" not in read["docs"][0]["path"] for read in reads)
