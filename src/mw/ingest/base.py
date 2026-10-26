"""Shared helpers for source adapters (DESIGN §22).

Adapter contract: fetch -> raw (immutable, under raw/{source_id}/dt={ingest_date}/), parse -> silver
rows carrying PIT columns.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

import polars as pl

from mw.core.storage import GuardedStorage


def raw_key(source_id: str, ingest_dt: datetime, name: str) -> str:
    return f"raw/{source_id}/dt={ingest_dt:%Y-%m-%d}/{name}"


def silver_key(table: str, market: str, as_of: date, name: str) -> str:
    return f"silver/{table}/market={market}/year={as_of:%Y}/month={as_of:%m}/{name}"


def write_raw_bytes(storage: GuardedStorage, key: str, data: bytes) -> str:
    """Write immutable raw bytes; returns sha256 for raw_ref."""
    storage.put_bytes(key, data, overwrite=False)
    return hashlib.sha256(data).hexdigest()


def stamp_pit(
    df: pl.DataFrame,
    *,
    as_of: date,
    published_at: datetime,
    ingested_at: datetime,
    source_id: str,
    source_version: str,
    raw_ref: str,
) -> pl.DataFrame:
    """Add PIT columns (all timestamps must be tz-aware UTC)."""
    return df.with_columns(
        pl.lit(as_of).alias("as_of"),
        pl.lit(published_at).dt.convert_time_zone("UTC").alias("published_at"),
        pl.lit(ingested_at).dt.convert_time_zone("UTC").alias("ingested_at"),
        pl.lit(source_id).alias("source_id"),
        pl.lit(source_version).alias("source_version"),
        pl.lit(raw_ref).alias("raw_ref"),
    )


def year_key(table: str, market: str, year: int) -> str:
    return f"silver/{table}/market={market}/year={year}/data.parquet"


def upsert_timeseries(
    storage: GuardedStorage,
    table: str,
    market: str,
    df: pl.DataFrame,
    key_cols: tuple[str, ...] = ("entity_id", "as_of"),
) -> dict[int, int]:
    """Merge rows into year-partitioned silver files (one object per year; few R2 ops).

    New rows win on key collision (latest ingestion). Raw keeps every vintage, so silver may be
    rewritten. Returns {year: row_count_after}.
    """
    out: dict[int, int] = {}
    if df.height == 0:
        return out
    df = df.with_columns(pl.col("as_of").dt.year().alias("_year"))
    for (year,), part in df.group_by("_year"):
        y = int(str(year))
        key = year_key(table, market, y)
        part = part.drop("_year")
        if storage.exists(key):
            old = storage.read_parquet(key)
            part = pl.concat([old, part], how="diagonal_relaxed")
        merged = part.unique(subset=list(key_cols), keep="last", maintain_order=True).sort(
            list(key_cols)
        )
        storage.write_parquet(key, merged)
        out[y] = merged.height
    return out


def date_key(table: str, market: str, d: Any) -> str:
    return f"silver/{table}/market={market}/date={d.isoformat()}.parquet"


def upsert_by_date(
    storage: GuardedStorage, table: str, market: str, df: pl.DataFrame, key_cols: list[str]
) -> dict[str, int]:
    """Merge event rows into per-date silver files (date = column `as_of`). First seen wins:
    an event's published_at/ingested_at must not move later on re-polls."""
    out: dict[str, int] = {}
    if df.height == 0:
        return out
    for (d,), part in df.group_by("as_of"):
        key = date_key(table, market, d)
        if storage.exists(key):
            part = pl.concat([storage.read_parquet(key), part], how="diagonal_relaxed")
        merged = part.unique(subset=key_cols, keep="first", maintain_order=True)
        storage.write_parquet(key, merged)
        out[str(d)] = merged.height
    return out
