"""Phase 0 acceptance job: write PIT-stamped parquet via the storage backend, read back, verify.

Usage: python -m jobs.dummy_pit [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, date, datetime, timedelta

import polars as pl

from mw.core.calendars import previous_session, session_close_utc
from mw.core.config import load_settings
from mw.core.logging import job_run, setup_logging
from mw.core.pit import as_of_cutoff, assert_no_future, validate_pit_frame
from mw.core.storage import open_storage

SOURCE_ID = "dummy"
SOURCE_VERSION = "0.1.0"


def build_frame(d: date, now: datetime) -> pl.DataFrame:
    close_us = session_close_utc("US", d)
    rows = [
        {"entity_id": "US:AAPL", "value": 1.0, "published_at": close_us},
        # A row "published" in the future relative to cutoff — must be filtered out by PIT reads.
        {"entity_id": "US:MSFT", "value": 2.0, "published_at": close_us + timedelta(days=3)},
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.lit(d).alias("as_of"),
        pl.col("published_at").dt.replace_time_zone("UTC"),
        pl.lit(now).dt.replace_time_zone("UTC").alias("ingested_at"),
        pl.lit(SOURCE_ID).alias("source_id"),
        pl.lit(SOURCE_VERSION).alias("source_version"),
    )
    return df.with_columns(
        pl.concat_str([pl.lit("raw/dummy/"), pl.col("entity_id")]).alias("raw_ref")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    setup_logging()
    settings = load_settings()
    storage = open_storage(settings.storage)
    d = args.date or previous_session("US", datetime.now(UTC).date())

    with job_run(storage, "dummy_pit", date=d.isoformat(), backend=settings.storage.backend) as s:
        now = datetime.now(UTC).replace(tzinfo=None)
        df = build_frame(d, now)
        validate_pit_frame(df)
        key = f"silver/dummy/market=US/year={d:%Y}/month={d:%m}/{d.isoformat()}.parquet"
        storage.write_parquet(key, df)

        back = storage.read_parquet(key)
        validate_pit_frame(back)
        cutoff = session_close_utc("US", d) + timedelta(hours=1)
        visible = as_of_cutoff(back, cutoff).collect()
        assert_no_future(visible, cutoff)
        if visible["entity_id"].to_list() != ["US:AAPL"]:
            raise AssertionError(f"PIT filter wrong: {visible['entity_id'].to_list()}")

        s["key"] = key
        s["rows_written"] = df.height
        s["rows_visible_at_cutoff"] = visible.height
        s["sha256"] = hashlib.sha256(storage.get_bytes(key)).hexdigest()
        s["usage"] = storage.usage


if __name__ == "__main__":
    main()
