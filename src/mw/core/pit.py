"""Point-in-time (PIT) utilities — DESIGN principle P1.

Every silver table carries PIT_COLUMNS. Features may only use rows with published_at <= cutoff.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

PIT_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "as_of",
    "published_at",
    "ingested_at",
    "source_id",
    "source_version",
    "raw_ref",
)


class PITViolation(AssertionError):
    """Highest-severity bug: data from the future reached a feature/model."""


def missing_pit_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in PIT_COLUMNS if c not in df.columns]


def validate_pit_frame(df: pl.DataFrame) -> None:
    """Structural check for silver tables."""
    missing = missing_pit_columns(df)
    if missing:
        raise PITViolation(f"missing PIT columns: {missing}")
    for col in ("published_at", "ingested_at"):
        dtype = df.schema[col]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise PITViolation(f"{col} must be Datetime with time_zone='UTC', got {dtype}")
    if df["published_at"].null_count() > 0:
        raise PITViolation("published_at contains nulls")


def as_of_cutoff(df: pl.LazyFrame | pl.DataFrame, cutoff: datetime) -> pl.LazyFrame:
    """The ONLY sanctioned way to read silver data for features: filter by published_at."""
    _require_utc(cutoff)
    return df.lazy().filter(pl.col("published_at") <= cutoff)


def assert_no_future(df: pl.DataFrame, cutoff: datetime, col: str = "published_at") -> None:
    """PIT assertion used by feature builders and CI leakage tests."""
    _require_utc(cutoff)
    if df.height == 0:
        return
    max_pub = df[col].max()
    if max_pub is not None and max_pub > cutoff:  # type: ignore[operator]
        raise PITViolation(f"max {col}={max_pub!s} > cutoff={cutoff}")


def utc_now() -> datetime:
    return datetime.now(UTC)


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None or ts.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"cutoff must be timezone-aware UTC, got {ts!r}")
