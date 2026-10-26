"""Reference data for validation: S&P 500 membership, fails-to-deliver, insider trades, shares outstanding.

Usage: python -m jobs.reference_data [--sources sp500,ftd,insider,shares] [--backfill]
Normal mode fetches only files not yet stored (plus the latest few); --backfill fetches all history.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from mw.core.config import load_settings
from mw.core.http import HttpClient
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest.base import raw_key, write_raw_bytes
from mw.ingest.us import sec_filings
from mw.ingest.us import sec_reference as R
from mw.universe import sp500

VERSION = "0.1.0"
WIKI_UA = "MarketWeather personal research (github.com/akyoshida-gif/market-weather)"


def _pit(df: pl.DataFrame, source: str, ref: str, now: datetime) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(now).dt.convert_time_zone("UTC").alias("ingested_at"),
        pl.lit(source).alias("source_id"),
        pl.lit(VERSION).alias("source_version"),
        pl.lit(ref).alias("raw_ref"),
    )


def run_sp500(storage: GuardedStorage, http: HttpClient, backfill: bool) -> dict[str, Any]:
    now = datetime.now(UTC)
    cur_html = http.get(sp500.CURRENT_URL).text
    ch_html = http.get(sp500.CHANGES_URL).text
    for name, txt in (("current", cur_html), ("changes", ch_html)):
        write_raw_bytes(
            storage, raw_key("sp500_wikipedia", now, f"{name}_{now:%H%M%S}.html"), txt.encode()
        )
    cur, ch = sp500.parse_current(cur_html), sp500.parse_changes(ch_html)
    iv = sp500.intervals(cur, ch, date(2000, 1, 1))
    counts = {str(y): len(sp500.members_on(iv, date(y, 6, 30))) for y in range(2010, now.year + 1)}
    storage.write_parquet(
        "silver/sp500_membership/latest.parquet",
        iv.with_columns(pl.lit(now).dt.convert_time_zone("UTC").alias("ingested_at")),
    )
    storage.write_parquet(f"silver/sp500_membership/date={now.date().isoformat()}.parquet", iv)
    storage.write_parquet("silver/sp500_membership/changes.parquet", ch)
    return {"current": cur.height, "changes": ch.height, "member_counts": counts}


def run_ftd(storage: GuardedStorage, sec: sec_filings.SecClient, backfill: bool) -> dict[str, Any]:
    links = R.ftd_links(sec.get(R.FTD_PAGE).decode("utf-8", "replace"))
    todo = [(u, p) for u, p in links if int(p[:4]) >= 2009]
    if not backfill:
        todo = todo[-4:]
    done, skipped, errors = [], 0, {}
    for url, period in todo:
        key = f"silver/sec_ftd/market=US/period={period}.parquet"
        if storage.exists(key) and not period >= todo[-1][1]:
            skipped += 1
            continue
        try:
            now = datetime.now(UTC)
            data = sec.get(url)
            rk = raw_key("sec_ftd", now, f"cnsfails{period}_{now:%H%M%S}.zip")
            ref = f"{rk}#sha256={write_raw_bytes(storage, rk, data)}"
            df = R.parse_ftd(data).with_columns(
                (pl.lit("US:") + pl.col("symbol")).alias("entity_id"),
                pl.col("settle_date").alias("as_of"),
                pl.lit(R.ftd_published_at(period)).alias("published_at"),
                pl.lit(period).alias("period"),
            )
            storage.write_parquet(key, _pit(df, "sec_ftd", ref, now))
            done.append(period)
        except Exception as e:
            errors[period] = f"{type(e).__name__}: {e}"[:160]
    return {
        "links": len(links),
        "ingested": len(done),
        "skipped": skipped,
        "errors": errors,
        "latest": todo[-1][1] if todo else None,
    }


def run_insider(
    storage: GuardedStorage, sec: sec_filings.SecClient, backfill: bool
) -> dict[str, Any]:
    links = R.insider_links(sec.get(R.INSIDER_PAGE).decode("utf-8", "replace"))
    todo = links if backfill else links[-2:]
    done, skipped, errors = [], 0, {}
    for url, quarter in todo:
        key = f"silver/sec_insider/market=US/quarter={quarter}.parquet"
        if storage.exists(key) and quarter != links[-1][1]:
            skipped += 1
            continue
        try:
            now = datetime.now(UTC)
            data = sec.get(url)
            rk = raw_key("sec_insider", now, f"{quarter}_form345_{now:%H%M%S}.zip")
            ref = f"{rk}#sha256={write_raw_bytes(storage, rk, data)}"
            df = R.parse_insider(data).with_columns(
                pl.when(pl.col("symbol").is_not_null() & (pl.col("symbol") != ""))
                .then(pl.lit("US:") + pl.col("symbol"))
                .otherwise(pl.lit("SEC:") + pl.col("cik").cast(pl.String))
                .alias("entity_id"),
                pl.col("filing_date").alias("as_of"),
                R.insider_published_at(pl.col("filing_date")).alias("published_at"),
            )
            storage.write_parquet(key, _pit(df, "sec_insider", ref, now))
            done.append(quarter)
        except Exception as e:
            errors[quarter] = f"{type(e).__name__}: {e}"[:160]
    return {"links": len(links), "ingested": done, "skipped": skipped, "errors": errors}


def run_shares(
    storage: GuardedStorage, sec: sec_filings.SecClient, backfill: bool
) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    qs = R.quarters(date(2009, 1, 1) if backfill else today - timedelta(days=400), today)
    tick = sec_filings.ticker_map(sec.get(sec_filings.TICKERS)).select("cik", "ticker")
    done, errors = [], {}
    for y, q in qs:
        key = f"silver/sec_shares/market=US/quarter={y}q{q}.parquet"
        try:
            now = datetime.now(UTC)
            data = sec.get(R.frames_url(y, q))
            rk = raw_key("sec_shares", now, f"CY{y}Q{q}I_{now:%H%M%S}.json")
            ref = f"{rk}#sha256={write_raw_bytes(storage, rk, data)}"
            df = (
                R.parse_frames(data)
                .join(tick, on="cik", how="left")
                .with_columns(
                    pl.when(pl.col("ticker").is_not_null())
                    .then(pl.lit("US:") + pl.col("ticker"))
                    .otherwise(pl.lit("SEC:") + pl.col("cik").cast(pl.String))
                    .alias("entity_id"),
                    pl.col("period_end").alias("as_of"),
                    (
                        (
                            pl.col("period_end").cast(pl.Datetime("us")) + pl.duration(days=90)
                        ).dt.replace_time_zone("UTC")
                    ).alias("published_at"),
                )
            )
            storage.write_parquet(key, _pit(df, "sec_shares", ref, now))
            done.append(f"{y}q{q}")
        except Exception as e:  # frames for very recent quarters may not exist yet
            errors[f"{y}q{q}"] = f"{type(e).__name__}"[:80]
    return {"quarters": len(done), "errors": errors}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="sp500,ftd,insider,shares")
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    wiki = HttpClient(rate_per_host=0.5, user_agent=WIKI_UA)
    failed = False
    holder: dict[str, Any] = {}

    def sec() -> sec_filings.SecClient:
        if "s" not in holder:
            holder["s"] = sec_filings.SecClient()
        c: sec_filings.SecClient = holder["s"]
        return c

    runners = {
        "sp500": lambda: run_sp500(storage, wiki, args.backfill),
        "ftd": lambda: run_ftd(storage, sec(), args.backfill),
        "insider": lambda: run_insider(storage, sec(), args.backfill),
        "shares": lambda: run_shares(storage, sec(), args.backfill),
    }
    for name in args.sources.split(","):
        try:
            with job_run(storage, f"reference_{name}", backfill=args.backfill) as s:
                s["result"] = runners[name]()
        except Exception:
            failed = True
    wiki.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
