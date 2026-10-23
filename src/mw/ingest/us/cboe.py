"""Cboe volatility indices daily history (DESIGN §8.1 `vol_indices`).

https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYM}_History.csv (307 -> cdn-api; full
history each file). Dates MM/DD/YYYY. Observed Last-Modified ~01:30 UTC next day.
published_at rule: as_of + 1 day 03:00 UTC. ToS: personal non-commercial use.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, time, timedelta
from typing import Any

import polars as pl

from mw.core.http import HttpClient
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, stamp_pit, upsert_timeseries, write_raw_bytes

SOURCE_ID = "vol_indices"
SOURCE_VERSION = "0.1.0"
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
SYMBOLS = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW"]
PUBLISH_UTC = time(3, 0)


def parse(sym: str, data: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(data), infer_schema=False)
    cols = [c.upper() for c in df.columns]
    df.columns = cols
    if "DATE" not in cols:
        raise ValueError(f"{sym}: no DATE column ({cols})")
    if "CLOSE" not in cols:  # single-value files (VVIX, SKEW)
        value_col = [c for c in cols if c != "DATE"]
        if len(value_col) != 1:
            raise ValueError(f"{sym}: unexpected columns {cols}")
        df = df.rename({value_col[0]: "CLOSE"})
    for c in ("OPEN", "HIGH", "LOW"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(c))
    return df.select(
        pl.col("DATE").str.to_date("%m/%d/%Y").alias("as_of"),
        *[pl.col(c).cast(pl.Float64, strict=False).alias(c.lower())
          for c in ("OPEN", "HIGH", "LOW", "CLOSE")],
    ).filter(pl.col("close").is_not_null())  # fmt: skip


def ingest(storage: GuardedStorage, http: HttpClient, since_days: int | None) -> dict[str, Any]:
    """since_days=None -> full history; else only rows within the last N days are upserted."""
    ingested_at = datetime.now(UTC)
    frames: list[pl.DataFrame] = []
    for sym in SYMBOLS:
        resp = http.get(URL.format(sym=sym))
        resp.raise_for_status()
        rk = raw_key(SOURCE_ID, ingested_at, f"{sym}_History_{ingested_at:%H%M%S}.csv")
        sha = write_raw_bytes(storage, rk, resp.content)
        df = parse(sym, resp.content)
        if since_days is not None:
            cutoff = ingested_at.date() - timedelta(days=since_days)
            df = df.filter(pl.col("as_of") >= cutoff)
        df = stamp_pit(
            df.drop("as_of").with_columns(pl.lit(f"IDX:US:{sym}").alias("entity_id")),
            as_of=ingested_at.date(),  # placeholder, replaced below
            published_at=ingested_at,
            ingested_at=ingested_at,
            source_id=SOURCE_ID,
            source_version=SOURCE_VERSION,
            raw_ref=f"{rk}#sha256={sha}",
        ).with_columns(df["as_of"].alias("as_of"))
        df = df.with_columns(
            (pl.col("as_of").cast(pl.Datetime("us")) + pl.duration(days=1, hours=PUBLISH_UTC.hour))
            .dt.replace_time_zone("UTC")
            .alias("published_at")
        )
        frames.append(df)
    all_df = pl.concat(frames, how="vertical_relaxed")
    years = upsert_timeseries(storage, "vol_indices", "US", all_df)
    return {"rows": all_df.height, "years_written": len(years)}
