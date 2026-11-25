from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.coverage import Coverage, CoverageStore, skip_coverage
from core.media import MediaKind


def test_add_merges_and_sorts() -> None:
    coverage = Coverage()
    coverage.add(60, 120)
    coverage.add(0, 30)
    coverage.add(20, 70)
    assert coverage.intervals == [(0.0, 120.0)]
    assert coverage.total() == 120.0


def test_gaps_and_covers() -> None:
    coverage = Coverage([(10, 20), (30, 40)])
    assert coverage.covers(10, 20)
    assert not coverage.covers(15, 35)
    assert coverage.gaps(0, 50) == [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
    assert coverage.overlaps(15, 35) == 10.0


def test_store_persists_across_instances(tmp_path: Path) -> None:
    store = CoverageStore(tmp_path, "sess1")
    store.mark(MediaKind.AUDIO, 0, 50)
    store.mark(MediaKind.VIDEO, 10, 30)

    again = CoverageStore(tmp_path, "sess1")
    assert again.covered_seconds(MediaKind.AUDIO) == 50.0
    assert again.covered_seconds(MediaKind.VIDEO) == 20.0
    assert again.snapshot()["audio"]["intervals"] == [[0.0, 50.0]]
    assert again.gaps(MediaKind.AUDIO, 0, 100) == [(50.0, 100.0)]


def test_concurrent_stores_merge_instead_of_clobbering(tmp_path: Path) -> None:
    """STT and VLM each own a store over the same file; neither may lose ranges."""

    stt = CoverageStore(tmp_path, "sess-merge")
    vlm = CoverageStore(tmp_path, "sess-merge")

    stt.mark(MediaKind.AUDIO, 0, 20)
    vlm.mark(MediaKind.VIDEO, 0, 20)
    stt.mark(MediaKind.AUDIO, 20, 40)
    vlm.mark(MediaKind.VIDEO, 20, 40)

    raw = json.loads(
        (tmp_path / "sessions" / "sess-merge" / "coverage.json").read_text("utf-8")
    )
    assert raw["audio"] == [[0.0, 40.0]]
    assert raw["video"] == [[0.0, 40.0]]
    # A fresh reader (the input plugin renders progress this way) sees both.
    fresh = CoverageStore(tmp_path, "sess-merge")
    assert fresh.covered_seconds(MediaKind.AUDIO) == 40.0
    assert fresh.covered_seconds(MediaKind.VIDEO) == 40.0


def test_skip_coverage_reads_session_meta() -> None:
    meta = {"skip_ranges": {"audio": [[0, 60], [120, 180]], "video": [[0, 60]]}}
    assert skip_coverage(meta, MediaKind.AUDIO).covers(0, 60)
    assert not skip_coverage(meta, MediaKind.AUDIO).covers(60, 120)
    assert skip_coverage(meta, MediaKind.VIDEO).covers(0, 60)
    assert skip_coverage(None, MediaKind.AUDIO).intervals == []


def test_store_requires_media_kind(tmp_path: Path) -> None:
    store = CoverageStore(tmp_path, "sess1")
    with pytest.raises(TypeError, match="MediaKind"):
        store.mark("audio", 0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        store.coverage("audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        store.covered_seconds("audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        store.gaps("audio", 0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MediaKind"):
        store.covers("audio", 0, 1)  # type: ignore[arg-type]


def test_skip_coverage_requires_media_kind() -> None:
    with pytest.raises(TypeError, match="MediaKind"):
        skip_coverage(None, "audio")  # type: ignore[arg-type]
