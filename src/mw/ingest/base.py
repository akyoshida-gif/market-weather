"""Shared helpers for source adapters (DESIGN §22).

Adapter contract: fetch -> raw (immutable, under raw/{source_id}/dt={ingest_date}/), parse -> silver
rows carrying PIT columns.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

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
