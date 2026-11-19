from datetime import datetime

from core.lecture import lecture_id, source_id


def test_lecture_id_uses_scheduled_start():
    begin = "2026-11-10 07:55:00"
    assert lecture_id("12345", begin) == "12345-20261110-0755"
    assert lecture_id("12345", datetime(2026, 11, 10, 7, 55)) == "12345-20261110-0755"
    assert lecture_id("12345", "2026-11-10 08:50:00") != lecture_id("12345", begin)
    assert lecture_id("67890", begin) != lecture_id("12345", begin)


def test_source_id_hashes_exact_url():
    assert source_id("abc") == "source-ba7816bf8f01cfea4141"
    assert source_id("abc") != source_id("ABC")
    assert source_id("https://example.test/media?a=1") != source_id(
        "https://example.test/media?a=2"
    )
