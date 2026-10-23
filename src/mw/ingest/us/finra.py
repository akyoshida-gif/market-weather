"""FINRA Reg SHO daily short volume and bi-monthly equity short interest (DESIGN §8.1).

Reg SHO daily (consolidated NMS file):
  https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt  (pipe-delimited, decimals)
  Published no later than 18:00 ET same day; unpublished -> HTTP 403. History from 2018-08-01.
  published_at rule: session date 18:00 America/New_York.
Short interest:
  https://cdn.finra.org/equity/otcmarket/biweekly/shrt{YYYYMMDD}.csv  (pipe-delimited, quoted)
  Settlement: 15th (or prior business day) and last business day of month. Published ~7th
  business day after settlement. published_at = min(capture, settlement + 7 sessions 23:59 UTC).
  FINRA may move to weekly reporting (UNVERIFIED) -> missing data for >25 days raises an alert.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from mw.core.calendars import get_calendar, is_session, previous_session, sessions
from mw.core.http import HttpClient
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, silver_key, stamp_pit, write_raw_bytes

REGSHO_ID = "finra_regsho_daily"
SI_ID = "finra_short_interest"
SOURCE_VERSION = "0.1.0"
REGSHO_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
SI_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d:%Y%m%d}.csv"
NY = ZoneInfo("America/New_York")


class NotPublished(Exception):
    pass


# ---------------------------------------------------------------- Reg SHO daily
def regsho_published_at(d: date) -> datetime:
    return datetime.combine(d, time(18, 0), tzinfo=NY).astimezone(UTC)


def parse_regsho(data: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(data), separator="|", infer_schema=False)
    expected = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]
    if df.columns != expected:
        raise ValueError(f"unexpected header {df.columns}")
    df = df.filter(pl.col("Date").str.contains(r"^\d{8}$"))  # drop any trailer line
    return df.select(
        pl.col("Date").str.to_date("%Y%m%d").alias("trade_date"),
        pl.col("Symbol").alias("symbol"),
        pl.col("ShortVolume").cast(pl.Float64).alias("short_volume"),
        pl.col("ShortExemptVolume").cast(pl.Float64).alias("short_exempt_volume"),
        pl.col("TotalVolume").cast(pl.Float64).alias("total_volume"),
        pl.col("Market").alias("markets"),
    ).with_columns(
        pl.when(pl.col("total_volume") > 0)
        .then(pl.col("short_volume") / pl.col("total_volume"))
        .otherwise(None)
        .alias("short_volume_ratio")
    )


def regsho_silver_key(d: date) -> str:
    return silver_key("finra_regsho_daily", "US", d, f"{d.isoformat()}.parquet")


def _get(http: HttpClient, url: str) -> bytes:
    resp = http.get(url)
    if resp.status_code in (403, 404):
        raise NotPublished(url)
    resp.raise_for_status()
    return resp.content


def ingest_regsho_day(storage: GuardedStorage, http: HttpClient, d: date) -> dict[str, Any] | None:
    skey = regsho_silver_key(d)
    if storage.exists(skey):
        return None
    data = _get(http, REGSHO_URL.format(d=d))
    df = parse_regsho(data)
    if df.height == 0 or df["trade_date"].unique().to_list() != [d]:
        raise ValueError(f"unexpected trade dates in file for {d}")
    ingested_at = datetime.now(UTC)
    rk = raw_key(REGSHO_ID, ingested_at, f"CNMSshvol{d:%Y%m%d}.txt")
    sha = write_raw_bytes(storage, rk, data)
    silver = stamp_pit(
        df.with_columns((pl.lit("US:") + pl.col("symbol")).alias("entity_id")),
        as_of=d,
        published_at=regsho_published_at(d),
        ingested_at=ingested_at,
        source_id=REGSHO_ID,
        source_version=SOURCE_VERSION,
        raw_ref=f"{rk}#sha256={sha}",
    )
    storage.write_parquet(skey, silver)
    return {"date": d.isoformat(), "rows": silver.height}


# ---------------------------------------------------------------- Short interest
def settlement_dates(start: date, end: date) -> list[date]:
    """15th (or prior session) and last session of each month within [start, end]."""
    out: set[date] = set()
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        mid = date(y, m, 15)
        out.add(mid if is_session("US", mid) else previous_session("US", mid))
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out.add(previous_session("US", nxt))
        y, m = nxt.year, nxt.month
    return sorted(d for d in out if start <= d <= end)


def si_rule_published_at(settle: date) -> datetime:
    cal = get_calendar("US")
    import pandas as pd

    pub = cal.session_offset(pd.Timestamp(settle), 7).date()
    return datetime.combine(pub, time(23, 59), tzinfo=UTC)


SI_COLUMNS = {
    "symbolCode": "symbol",
    "issueName": "issue_name",
    "issuerServicesGroupExchangeCode": "exchange_code",
    "marketClassCode": "market_class",
    "currentShortPositionQuantity": "short_interest",
    "previousShortPositionQuantity": "prev_short_interest",
    "stockSplitFlag": "split_flag",
    "averageDailyVolumeQuantity": "avg_daily_volume",
    "daysToCoverQuantity": "days_to_cover",
    "revisionFlag": "revision_flag",
    "changePercent": "change_pct",
    "changePreviousNumber": "change_shares",
    "settlementDate": "settlement_date",
}


def parse_short_interest(data: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(data), separator="|", infer_schema=False, quote_char='"')
    missing = [c for c in SI_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    df = df.select([pl.col(k).alias(v) for k, v in SI_COLUMNS.items()])
    nums = ["short_interest", "prev_short_interest", "avg_daily_volume", "days_to_cover",
            "change_pct", "change_shares"]  # fmt: skip
    return df.with_columns(
        pl.col("settlement_date").str.to_date("%Y-%m-%d"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in nums],
    )


def si_silver_key(settle: date) -> str:
    return silver_key("finra_short_interest", "US", settle, f"{settle.isoformat()}.parquet")


def ingest_si(storage: GuardedStorage, http: HttpClient, settle: date) -> dict[str, Any] | None:
    skey = si_silver_key(settle)
    if storage.exists(skey):
        return None
    data = _get(http, SI_URL.format(d=settle))
    df = parse_short_interest(data)
    captured = datetime.now(UTC)
    rk = raw_key(SI_ID, captured, f"shrt{settle:%Y%m%d}.csv")
    sha = write_raw_bytes(storage, rk, data)
    silver = stamp_pit(
        df.with_columns((pl.lit("US:") + pl.col("symbol")).alias("entity_id")),
        as_of=settle,
        published_at=min(captured, si_rule_published_at(settle)),
        ingested_at=captured,
        source_id=SI_ID,
        source_version=SOURCE_VERSION,
        raw_ref=f"{rk}#sha256={sha}",
    )
    storage.write_parquet(skey, silver)
    return {"settlement": settle.isoformat(), "rows": silver.height}


def recent_sessions(today: date, n: int) -> list[date]:
    return sessions("US", today - timedelta(days=n * 2 + 7), today)[-n:]
