"""S&P 500 point-in-time membership reconstructed from Wikipedia (ADR-0015).

Sources (one fetch each per run):
  current : https://en.wikipedia.org/wiki/List_of_S%26P_500_companies (#constituents)
  changes : https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500 (#changes)
Method: start from today's members and walk the change list backwards in time
(undo additions after date d, restore removals after date d). Wikipedia's history may be
incomplete; the daily member count is reported as a quality check (expected ~500-505).
Wikipedia content is CC BY-SA; used for personal research only.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import polars as pl

CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def _norm(t: object) -> str | None:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip().upper()
    return s.replace(".", "-") if s else None  # BRK.B -> BRK-B (Yahoo style)


def parse_current(html: str) -> pl.DataFrame:
    t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    return pl.DataFrame({
        "symbol": [_norm(x) for x in t["Symbol"]],
        "name": t["Security"].fillna("").astype(str).tolist(),
        "sector": t["GICS Sector"].fillna("").astype(str).tolist(),
        "added": pd.to_datetime(t["Date added"], errors="coerce").dt.date.tolist(),
        "cik": t["CIK"].astype("Int64").tolist(),
    })  # fmt: skip


def parse_changes(html: str) -> pl.DataFrame:
    t = pd.read_html(io.StringIO(html), attrs={"id": "changes"})[0]
    t.columns = ["date", "add_ticker", "add_name", "rem_ticker", "rem_name", "reason", "refs"]
    d = pd.to_datetime(t["date"], errors="coerce").dt.date
    return pl.DataFrame({
        "date": d.tolist(),
        "added": [_norm(x) for x in t["add_ticker"]],
        "removed": [_norm(x) for x in t["rem_ticker"]],
        "added_name": t["add_name"].fillna("").astype(str).tolist(),
        "removed_name": t["rem_name"].fillna("").astype(str).tolist(),
        "reason": t["reason"].fillna("").astype(str).tolist(),
    }).filter(pl.col("date").is_not_null())  # fmt: skip


def intervals(current: pl.DataFrame, changes: pl.DataFrame, start: date) -> pl.DataFrame:
    """Membership intervals (symbol, start, end) with end exclusive; end=None means still a member.

    Walk backwards: for each change on date d (effective at the start of d), before d the
    added ticker was not a member and the removed ticker was a member.
    """
    members: dict[str, date | None] = dict.fromkeys(
        current["symbol"].to_list()
    )  # symbol -> end (None)
    open_since: dict[str, date | None] = {}
    rows: list[dict[str, object]] = []
    ends: dict[str, date | None] = {s: None for s in members}
    for d, add, rem in (
        changes.sort("date", descending=True).select("date", "added", "removed").iter_rows()
    ):
        if d < start:
            break
        if add and add in ends:
            rows.append({"symbol": add, "start": d, "end": ends.pop(add)})
        if rem:
            ends[rem] = d  # was a member until d
    for s, e in ends.items():
        rows.append({"symbol": s, "start": start, "end": e})
    _ = (open_since, members)
    return pl.DataFrame(rows, schema={"symbol": pl.String, "start": pl.Date, "end": pl.Date})


def members_on(iv: pl.DataFrame, d: date) -> list[str]:
    m = iv.filter((pl.col("start") <= d) & (pl.col("end").is_null() | (pl.col("end") > d)))
    return sorted(m["symbol"].unique().to_list())
