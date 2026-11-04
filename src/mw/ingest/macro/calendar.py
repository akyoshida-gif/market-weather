"""Event calendars (VALIDATION_PLAN_V3 §5): FOMC decisions and major US data release dates.

FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (recent years) and
      https://www.federalreserve.gov/monetarypolicy/fomchistorical{YEAR}.htm (older years).
      The decision date is the last day of each meeting (statement at 14:00 New York).
Releases: FRED API /fred/release/dates (CPI, jobs report, GDP, PCE, PPI, retail sales).
These are schedules known in advance (FOMC dates are published a year ahead), so they can be used
for "days until / since the event" without look-ahead. Option expiry, month ends and holidays are
computed from the exchange calendar where needed (no download).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import polars as pl

FOMC_CURRENT = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
FRED_RELEASE_DATES = "https://api.stlouisfed.org/fred/release/dates"
RELEASES = {10: "cpi", 50: "jobs", 53: "gdp", 54: "pce", 46: "ppi", 9: "retail_sales"}
MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                          "August", "September", "October", "November", "December"])}  # fmt: skip
MONTHS |= {k[:3]: v for k, v in MONTHS.items()} | {
    "Sept": 9
}  # split meetings use abbreviations ("Oct/Nov")

_YEAR_RE = re.compile(r">(\d{4}) FOMC Meetings<")
_MONTH_RE = re.compile(r"fomc-meeting__month[^>]*>(?:<strong>)?([A-Za-z/]+)")
_DATE_RE = re.compile(r"fomc-meeting__date[^>]*>([0-9]+(?:-[0-9]+)?)(\*?)")
_HIST_RE = re.compile(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
                      r"(?:/(?:January|February|March|April|May|June|July|August|September|October|November|December))?)"
                      r" (\d{1,2})(?:-(\d{1,2}))? (?:Meeting|\(unscheduled\))")  # fmt: skip


def _last_day(year: int, month_txt: str, d1: str, d2: str | None) -> date:
    months = month_txt.split("/")
    m = MONTHS[months[-1]] if (d2 and len(months) > 1) else MONTHS[months[0]]
    return date(year, m, int(d2 or d1))


def parse_current(html: str) -> list[date]:
    """Meeting decision dates on the current calendar page (several years, one panel per year)."""
    out: list[date] = []
    parts = _YEAR_RE.split(html)
    # parts = [before, year1, body1, year2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        year, body = int(parts[i]), parts[i + 1]
        months = _MONTH_RE.findall(body)
        dates = _DATE_RE.findall(body)
        for mtxt, (dtxt, _) in zip(months, dates, strict=False):
            d = dtxt.split("-")
            try:
                out.append(_last_day(year, mtxt, d[0], d[1] if len(d) > 1 else None))
            except (KeyError, ValueError):
                continue
    return sorted(set(out))


def parse_historical(html: str, year: int) -> list[date]:
    out = []
    for mtxt, d1, d2 in _HIST_RE.findall(html):
        try:
            out.append(_last_day(year, mtxt, d1, d2 or None))
        except (KeyError, ValueError):
            continue
    return sorted(set(out))


def fomc_frame(dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {"event": "fomc", "date": sorted(set(dates))}, schema={"event": pl.String, "date": pl.Date}
    )


def parse_release_dates(js: dict[str, Any], name: str) -> pl.DataFrame:
    ds = sorted({r["date"] for r in js.get("release_dates", []) if r.get("date")})
    return pl.DataFrame({"event": name, "date": ds}).with_columns(pl.col("date").str.to_date())
