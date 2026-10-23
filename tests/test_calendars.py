from __future__ import annotations

from datetime import UTC, date, datetime

from mw.core.calendars import is_session, next_session, previous_session, session_open_utc


def test_holidays():
    assert not is_session("US", date(2026, 7, 3))  # Independence Day observed
    assert not is_session("JP", date(2026, 1, 1))
    assert is_session("JP", date(2026, 1, 5))


def test_next_previous():
    assert next_session("US", date(2026, 1, 2)) == date(2026, 1, 5)
    assert previous_session("JP", date(2026, 1, 5)) == date(2025, 12, 30)


def test_open_dst():
    # NYSE opens 09:30 ET: 14:30 UTC in winter, 13:30 UTC in summer.
    assert session_open_utc("US", date(2026, 1, 5)) == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    assert session_open_utc("US", date(2026, 7, 6)) == datetime(2026, 7, 6, 13, 30, tzinfo=UTC)
    assert session_open_utc("JP", date(2026, 1, 5)) == datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
