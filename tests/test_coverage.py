"""Tests for the STT/VLM coverage bookkeeping."""

from __future__ import annotations

from pathlib import Path

from core.coverage import Coverage, CoverageStore, skip_coverage


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
    store.mark("audio", 0, 50)
    store.mark("video", 10, 30)

    again = CoverageStore(tmp_path, "sess1")
    assert again.covered_seconds("audio") == 50.0
    assert again.covered_seconds("video") == 20.0
    assert again.snapshot()["audio"]["intervals"] == [[0.0, 50.0]]
    assert again.gaps("audio", 0, 100) == [(50.0, 100.0)]


def test_skip_coverage_reads_session_meta() -> None:
    meta = {"skip_ranges": {"audio": [[0, 60], [120, 180]], "video": [[0, 60]]}}
    assert skip_coverage(meta, "audio").covers(0, 60)
    assert not skip_coverage(meta, "audio").covers(60, 120)
    assert skip_coverage(meta, "video").covers(0, 60)
    assert skip_coverage(None, "audio").intervals == []
