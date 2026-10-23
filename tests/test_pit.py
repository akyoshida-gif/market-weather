"""PIT leakage tests (DESIGN §11): injected future data must be detected."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from mw.core.pit import PITViolation, as_of_cutoff, assert_no_future, validate_pit_frame

CUTOFF = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def frame(pubs):
    n = len(pubs)
    return pl.DataFrame(
        {
            "entity_id": [f"US:X{i}" for i in range(n)],
            "as_of": [date(2026, 1, 5)] * n,
            "published_at": pubs,
            "ingested_at": [CUTOFF] * n,
            "source_id": ["t"] * n,
            "source_version": ["1"] * n,
            "raw_ref": ["r"] * n,
            "value": list(range(n)),
        }
    ).with_columns(
        pl.col("published_at").dt.convert_time_zone("UTC"),
        pl.col("ingested_at").dt.convert_time_zone("UTC"),
    )


def test_valid_frame_passes():
    validate_pit_frame(frame([CUTOFF]))


def test_missing_column_detected():
    with pytest.raises(PITViolation):
        validate_pit_frame(frame([CUTOFF]).drop("published_at"))


def test_naive_timestamp_rejected():
    df = frame([CUTOFF]).with_columns(pl.col("published_at").dt.replace_time_zone(None))
    with pytest.raises(PITViolation):
        validate_pit_frame(df)


def test_injected_future_row_detected():
    df = frame([CUTOFF - timedelta(hours=1), CUTOFF + timedelta(seconds=1)])
    with pytest.raises(PITViolation):
        assert_no_future(df, CUTOFF)


def test_cutoff_filter_removes_future():
    df = frame([CUTOFF - timedelta(hours=1), CUTOFF, CUTOFF + timedelta(days=1)])
    out = as_of_cutoff(df, CUTOFF).collect()
    assert out.height == 2
    assert_no_future(out, CUTOFF)


def test_naive_cutoff_rejected():
    with pytest.raises(ValueError):
        as_of_cutoff(frame([CUTOFF]), CUTOFF.replace(tzinfo=None))
