from core.lecture import source_id
from core.media import SourceKind
from core.streams import StreamHub
from plugins.input.sources import FFmpegSource, SourceManager, SourceSpec


def completed_source(monkeypatch):
    def start(source):
        source._running = True
        source._ended_emitted = True
        source._session_id = source._hub.current.session_id

    monkeypatch.setattr(FFmpegSource, "start", start)


def test_class_identity_overrides_source_url(monkeypatch, tmp_path):
    completed_source(monkeypatch)
    hub = StreamHub()
    root = tmp_path / "sessions"
    manager = SourceManager(hub=hub, emit=lambda *args, **kwargs: None, media_root=root)
    class_id = "12345-20261110-0755"
    for kind, url in (
        (SourceKind.LIVE, "https://example.test/live"),
        (SourceKind.RECORDING, "https://example.test/recording-1"),
        (SourceKind.RECORDING, "https://example.test/recording-2"),
    ):
        manager.start(SourceSpec(kind=kind, url=url, course={"class_id": class_id}))
        assert hub.current.session_id == class_id
        assert manager._current._media_dir == root / class_id / "media"
    assert [session.session_id for session in hub.list_sessions()].count(class_id) == 1
    manager.stop()


def test_manual_source_identity_is_repeatable(monkeypatch):
    completed_source(monkeypatch)
    hub = StreamHub()
    manager = SourceManager(hub=hub, emit=lambda *args, **kwargs: None)
    url = "https://example.test/manual.wav"
    manager.start(SourceSpec(url=url))
    first = hub.current
    manager.start(SourceSpec(url=url))
    assert hub.current.session_id == first.session_id == source_id(url)
    assert hub.current is not first
    manager.start(SourceSpec(url=url + "?version=2"))
    assert hub.current.session_id != first.session_id
    manager.stop()


def test_continuation_requires_same_class(monkeypatch):
    completed_source(monkeypatch)
    hub = StreamHub()
    manager = SourceManager(hub=hub, emit=lambda *args, **kwargs: None)
    manager.start(SourceSpec(url="part-1", course={"class_id": "first"}))
    first = hub.current
    first.transcript.append("first period")
    manager.start(
        SourceSpec(url="part-2", course={"class_id": "first"}, continue_session=True)
    )
    assert hub.current is first
    assert first.transcript[0].text == "first period"
    manager.start(
        SourceSpec(url="other", course={"class_id": "second"}, continue_session=True)
    )
    assert hub.current is not first and hub.current.session_id == "second"
    assert len(hub.current.transcript) == 0
    manager.stop()


def test_fill_uses_target_id_and_preserves_artifacts(monkeypatch, tmp_path):
    completed_source(monkeypatch)
    hub = StreamHub()
    target = "12345-20261110-0755"
    directory = tmp_path / target
    directory.mkdir()
    transcript = directory / "transcript.jsonl"
    transcript.write_text('{"text":"existing"}\n', encoding="utf-8")
    manager = SourceManager(
        hub=hub, emit=lambda *args, **kwargs: None, media_root=tmp_path
    )
    manager.start(
        SourceSpec(
            url="https://example.test/recording",
            meta={"class_id": target, "fill_for": target},
        )
    )
    assert hub.current.session_id == target
    assert manager._current._media_dir == directory / "media"
    assert transcript.read_text("utf-8") == '{"text":"existing"}\n'
    manager.stop()
