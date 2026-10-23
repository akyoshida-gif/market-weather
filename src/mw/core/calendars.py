"""Exchange calendars (XNYS, XTKS) wrapper. Handles holidays and DST via exchange_calendars."""

from __future__ import annotations

from datetime import UTC, date, datetime
from functools import cache

import exchange_calendars as xcals
import pandas as pd

MARKET_CALENDARS = {"US": "XNYS", "JP": "XTKS"}


@cache
def get_calendar(market: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(MARKET_CALENDARS[market])


def is_session(market: str, d: date) -> bool:
    return bool(get_calendar(market).is_session(pd.Timestamp(d)))


def sessions(market: str, start: date, end: date) -> list[date]:
    idx = get_calendar(market).sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in idx]


def next_session(market: str, d: date) -> date:
    cal = get_calendar(market)
    ts = pd.Timestamp(d)
    nxt = cal.next_session(ts) if cal.is_session(ts) else cal.date_to_session(ts, "next")
    return nxt.date()  # type: ignore[no-any-return]


def previous_session(market: str, d: date) -> date:
    cal = get_calendar(market)
    ts = pd.Timestamp(d)
    prv = cal.previous_session(ts) if cal.is_session(ts) else cal.date_to_session(ts, "previous")
    return prv.date()  # type: ignore[no-any-return]


def session_open_utc(market: str, d: date) -> datetime:
    ts = get_calendar(market).session_open(pd.Timestamp(d))
    return ts.to_pydatetime().astimezone(UTC)  # type: ignore[no-any-return]


def session_close_utc(market: str, d: date) -> datetime:
    ts = get_calendar(market).session_close(pd.Timestamp(d))
    return ts.to_pydatetime().astimezone(UTC)  # type: ignore[no-any-return]
