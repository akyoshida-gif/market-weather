"""US catalyst events: SEC filings + trading halts. Polled every 15 min (live) with backfill modes.

Usage:
  python -m jobs.news_poll                         # live: SEC Atom + Nasdaq halts RSS
  python -m jobs.news_poll --daily       # + ticker map snapshot + NYSE halts (last 3 days)
  python -m jobs.news_poll --backfill-sec 2021-01-01 [--cik-limit N]
  python -m jobs.news_poll --backfill-halts 2021-01-01
"""

from __future__ import annotations

import argparse
import gzip
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from mw.core.config import load_settings
from mw.core.http import HttpClient
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest.base import raw_key, upsert_by_date, write_raw_bytes
from mw.ingest.us import halts, sec_filings

TICKER_KEY = "silver/sec_tickers/latest.parquet"


def _stamp(
    df: pl.DataFrame, source_id: str, version: str, raw_ref: str, now: datetime
) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(now).dt.convert_time_zone("UTC").alias("ingested_at"),
        pl.lit(source_id).alias("source_id"),
        pl.lit(version).alias("source_version"),
        pl.lit(raw_ref).alias("raw_ref"),
    )


def _raw(storage: GuardedStorage, source: str, name: str, parts: list[bytes], now: datetime) -> str:
    blob = gzip.compress(b"\n<!--MW-SPLIT-->\n".join(parts))
    rk = raw_key(source, now, f"{name}_{now:%H%M%S}.gz")
    return f"{rk}#sha256={write_raw_bytes(storage, rk, blob)}"


def tickers(storage: GuardedStorage, sec: sec_filings.SecClient, refresh: bool) -> pl.DataFrame:
    if refresh or not storage.exists(TICKER_KEY):
        now = datetime.now(UTC)
        data = sec.get(sec_filings.TICKERS)
        _raw(storage, "sec_tickers", "company_tickers_exchange", [data], now)
        df = sec_filings.ticker_map(data)
        storage.write_parquet(TICKER_KEY, df)
        storage.write_parquet(f"silver/sec_tickers/date={now.date().isoformat()}.parquet", df)
        return df
    return storage.read_parquet(TICKER_KEY)


def run_sec_live(
    storage: GuardedStorage, sec: sec_filings.SecClient, tick: pl.DataFrame
) -> dict[str, Any]:
    now = datetime.now(UTC)
    rows, raws, errors = sec_filings.poll_live(sec)
    ref = _raw(storage, sec_filings.SOURCE_ID, "atom", raws, now) if raws else ""
    df = sec_filings.to_frame(rows, tick)
    if df.height:
        df = _stamp(df, sec_filings.SOURCE_ID, sec_filings.SOURCE_VERSION, ref, now)
        upsert_by_date(storage, "sec_filings", "US", df, ["accession", "cik", "role"])
    if errors and not rows:
        raise RuntimeError(f"all SEC polls failed: {errors}")
    return {"rows": df.height, "errors": errors}


def run_halts_live(storage: GuardedStorage, http: HttpClient) -> dict[str, Any]:
    now = datetime.now(UTC)
    data = http.get(halts.NASDAQ_RSS).content
    ref = _raw(storage, halts.SOURCE_ID, "nasdaq_rss", [data], now)
    df = halts.to_frame(halts.parse_nasdaq_rss(data))
    if df.height:
        df = _stamp(df, halts.SOURCE_ID, halts.SOURCE_VERSION, ref, now)
        upsert_by_date(storage, "trading_halts", "US", df, ["symbol", "halted_at", "source"])
    return {"rows": df.height}


def run_halts_hist(
    storage: GuardedStorage, http: HttpClient, start: date, end: date
) -> dict[str, Any]:
    total = 0
    d = start
    while d <= end:
        e = min(end, d + timedelta(days=30))
        now = datetime.now(UTC)
        data = http.get(halts.nyse_url(d, e)).content
        ref = _raw(storage, halts.SOURCE_ID, f"nyse_{d}_{e}", [data], now)
        df = halts.to_frame(halts.parse_nyse_csv(data))
        if df.height:
            df = _stamp(df, halts.SOURCE_ID, halts.SOURCE_VERSION, ref, now)
            upsert_by_date(storage, "trading_halts", "US", df, ["symbol", "halted_at", "source"])
            total += df.height
        d = e + timedelta(days=1)
    return {"rows": total}


def run_sec_backfill(
    storage: GuardedStorage,
    sec: sec_filings.SecClient,
    tick: pl.DataFrame,
    since: date,
    limit: int | None,
) -> dict[str, Any]:
    ciks = tick["cik"].to_list()[:limit] if limit else tick["cik"].to_list()
    done, errors, raws = 0, 0, []
    frames: list[pl.DataFrame] = []
    for i, cik in enumerate(ciks):
        try:
            data = sec.get(sec_filings.SUBMISSIONS.format(cik=cik))
            raws.append(data)
            rows = sec_filings.parse_submissions(data, since)
            done += 1
        except Exception:
            errors += 1
            rows = []
        if rows:
            frames.append(pl.DataFrame(rows))
        if len(raws) >= 500 or i == len(ciks) - 1:  # raw archive in large batches
            now = datetime.now(UTC)
            ref = _raw(storage, sec_filings.SOURCE_ID, f"submissions_{i:05d}", raws, now)
            raws = []
            if frames:
                frames[-1] = frames[-1].with_columns(pl.lit(ref).alias("_ref"))
    # write each date file once (rewriting ~1,500 date files per batch was the bottleneck)
    if frames:
        now = datetime.now(UTC)
        allrows = pl.concat(frames, how="diagonal_relaxed")
        df = sec_filings.to_frame(allrows.drop("_ref", strict=False).to_dicts(), tick)
        df = _stamp(
            df,
            sec_filings.SOURCE_ID,
            sec_filings.SOURCE_VERSION,
            "raw/sec_filings (submissions batches)",
            now,
        )
        upsert_by_date(storage, "sec_filings", "US", df, ["accession", "cik", "role"])
    return {"ciks": len(ciks), "ok": done, "errors": errors}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--daily", action="store_true")
    p.add_argument("--backfill-sec", type=date.fromisoformat)
    p.add_argument("--backfill-halts", type=date.fromisoformat)
    p.add_argument("--cik-limit", type=int)
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    http = HttpClient(rate_per_host=0.5)
    failed = False

    def step(name: str, fn: Any) -> None:
        nonlocal failed
        try:
            with job_run(storage, f"news_{name}") as s:
                s["result"] = fn()
        except Exception:
            failed = True

    holder: dict[str, Any] = {}

    def sec() -> sec_filings.SecClient:
        # created lazily: a missing SEC_USER_AGENT fails only the SEC steps (halts still run)
        if "sec" not in holder:
            holder["sec"] = sec_filings.SecClient()
        client: sec_filings.SecClient = holder["sec"]
        return client

    tick_holder: dict[str, pl.DataFrame] = {}
    step(
        "sec_tickers",
        lambda: {"rows": (tick_holder.setdefault("t", tickers(storage, sec(), args.daily))).height},
    )
    tick = tick_holder.get(
        "t", pl.DataFrame(schema={"cik": pl.Int64, "ticker": pl.String, "exchange": pl.String})
    )
    if args.backfill_sec:
        step(
            "sec_backfill",
            lambda: run_sec_backfill(storage, sec(), tick, args.backfill_sec, args.cik_limit),
        )
    elif args.backfill_halts:
        step(
            "halts_backfill",
            lambda: run_halts_hist(storage, http, args.backfill_halts, datetime.now(UTC).date()),
        )
    else:
        step("sec_live", lambda: run_sec_live(storage, sec(), tick))
        step("halts_live", lambda: run_halts_live(storage, http))
        if args.daily:
            today = datetime.now(UTC).date()
            step(
                "halts_nyse",
                lambda: run_halts_hist(storage, http, today - timedelta(days=3), today),
            )
    http.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
