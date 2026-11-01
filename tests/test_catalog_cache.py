"""TTL cache regression for the input catalog."""

from __future__ import annotations

from plugins.input.catalog import Catalog

from support import FakeSettings


class FakeUpstream:
    def __init__(self) -> None:
        self.calls = 0

    def list_courses(self):
        self.calls += 1
        return [{"id": "1", "name": "x"}]

    def list_recordings(self, course_id, limit=50):
        self.calls += 1
        return []

    def get_recording_url(self, course_id, video_id):
        return "http://example/x.mp4"

    def list_lives(self, start, end):
        self.calls += 1
        return []

    def get_live_devices(self, live_id):
        self.calls += 1
        return []

    def close(self):
        pass


def test_catalog_caches_courses(monkeypatch) -> None:
    upstream = FakeUpstream()
    catalog = Catalog(settings=FakeSettings())
    monkeypatch.setattr(Catalog, "_cat", lambda self: upstream)

    assert catalog.list_courses() == [{"id": "1", "name": "x"}]
    assert catalog.list_courses() == [{"id": "1", "name": "x"}]
    assert upstream.calls == 1

    catalog.reset()
    assert catalog.list_courses() == [{"id": "1", "name": "x"}]
    assert upstream.calls == 2
