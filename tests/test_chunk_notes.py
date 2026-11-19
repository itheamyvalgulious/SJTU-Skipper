from types import SimpleNamespace
import pytest
from plugins.notes import NotesService
from plugins.notes.view import NotesView
from plugins.notes.types import NotePublication


def test_service_uses_pipeline_files(tmp_path):
    expected = {"knowledge": tmp_path / "chunk_note_knowledge.md"}
    pipeline = SimpleNamespace(
        draft_paths=lambda sid: expected,
        note_files=lambda kind: {"knowledge": (f"{kind}.md", "知识")},
        running_tasks=lambda: [],
    )
    service = NotesService(pipeline, SimpleNamespace())
    assert service.draft_paths("course") == expected
    assert service.note_files(NotePublication.CHUNK)["knowledge"][0] == "chunk.md"


def test_service_opens_final_chunk_and_log_views():
    calls = []
    view = SimpleNamespace(
        open_notes=lambda sid, doc: calls.append(("final", sid, doc)),
        open_chunk_notes=lambda sid, doc: calls.append(("chunk", sid, doc)),
        open_logs=lambda sid: calls.append(("logs", sid)),
    )
    service = NotesService(SimpleNamespace(), view)
    service.open_notes("course", "knowledge")
    service.open_chunk_notes("course", "notice")
    service.open_logs("course")
    assert calls == [
        ("final", "course", "knowledge"),
        ("chunk", "course", "notice"),
        ("logs", "course"),
    ]


def test_toggle_retains_source_records():
    values = {}
    plugin = NotesView()
    plugin.settings = SimpleNamespace(
        set=lambda key, value: values.update({key: value})
    )
    plugin.ctx = SimpleNamespace(ui=SimpleNamespace(refresh=lambda event: None))
    assert plugin._chunk_toggle({"enabled": False})["ok"]
    assert values["notes.chunks_enabled"] is False


@pytest.mark.parametrize("selected", ["", "boot"])
def test_empty_selection_hides_notes(selected, navigation):
    plugin = NotesView()
    navigation.class_id = selected
    plugin.ctx = SimpleNamespace(
        streams=SimpleNamespace(
            current=SimpleNamespace(session_id="processing"),
        )
    )
    assert plugin._render_chunks() == []
    assert not plugin._chunk_read({})["ok"]


@pytest.mark.parametrize("target", ["", "boot"])
def test_invalid_read_target_does_not_use_selection(target, navigation):
    plugin = NotesView()
    navigation.class_id = "selected"
    plugin.ctx = SimpleNamespace()
    assert not plugin._chunk_read({"session_id": target})["ok"]


def test_notes_use_selected_or_explicit_class(tmp_path, navigation):
    reads = []
    statuses = []
    reader = SimpleNamespace(show=lambda **kwargs: reads.append(kwargs) or "note-page")
    plugin = NotesView()
    navigation.class_id = "selected"
    plugin.ctx = SimpleNamespace(
        data_dir=tmp_path,
        streams=SimpleNamespace(
            current=SimpleNamespace(session_id="processing"),
        ),
        require=lambda name: reader,
    )
    plugin.settings = SimpleNamespace(get=lambda key, default: default)
    plugin.pipeline = SimpleNamespace(
        status=lambda sid: statuses.append(sid) or {"errors": [], "published_end": 12},
        note_files=lambda kind: {"knowledge": ("chunk_note_knowledge.md", "知识")},
    )
    assert plugin._render_chunks()
    assert statuses == ["selected"]
    assert plugin._chunk_read({})["ok"]
    navigation.class_id = ""
    assert plugin._chunk_read({"session_id": "task-page"})["ok"]
    assert [read["page_key"] for read in reads] == [
        "selected:chunk-notes",
        "task-page:chunk-notes",
    ]
    assert all("processing" not in read["docs"][0]["path"] for read in reads)
