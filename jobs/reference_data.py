"""Reference data for validation: S&P 500 membership, fails-to-deliver, insider trades, shares outstanding,
FINRA margin statistics, CFTC positioning, Cboe put/call ratios, SEC company profiles (SIC).

Usage: python -m jobs.reference_data [--sources sp500,ftd,insider,shares,margin,cftc,pcr,sic] [--backfill]
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
from mw.ingest.base import raw_key, remote_raw_ref, write_raw_bytes
from mw.ingest.us import market_extra as X
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
            # SEC keeps publishing these files: record URL + sha256 instead of a copy (ADR-0019)
            ref = remote_raw_ref(
                storage, "sec_insider", f"{quarter}_form345_{now:%H%M%S}.zip", url, data, now
            )
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


def _store(storage: GuardedStorage, source: str, name: str, data: bytes, now: datetime) -> str:
    rk = raw_key(source, now, f"{name}_{now:%H%M%S}")
    return f"{rk}#sha256={write_raw_bytes(storage, rk, data)}"


def run_margin(storage: GuardedStorage, http: HttpClient, backfill: bool) -> dict[str, Any]:
    now = datetime.now(UTC)
    url = X.finra_link(http.get(X.FINRA_PAGE).text)
    data = http.get(url).content
    ref = _store(storage, "finra_margin", "margin-statistics.xlsx", data, now)
    df = X.parse_margin(data).with_columns(pl.lit("MACRO:FINRA:MARGIN").alias("entity_id"))
    storage.write_parquet(
        "silver/finra_margin/market=US/latest.parquet", _pit(df, "finra_margin", ref, now)
    )
    return {"months": df.height, "first": str(df["as_of"].min()), "last": str(df["as_of"].max())}


def run_cftc(storage: GuardedStorage, http: HttpClient, backfill: bool) -> dict[str, Any]:
    """CFTC positioning via the public reporting API, one request series per contract; stored by year."""
    today = datetime.now(UTC).date()
    since = date(2006, 1, 1) if backfill else today - timedelta(days=60)
    now = datetime.now(UTC)
    frames, raws = [], []
    for kind, ds in X.COT_DATASETS.items():
        for code in X.API_MARKETS[kind]:
            offset = 0
            while True:
                r = http.get(
                    X.COT_API.format(ds=ds), params=X.cot_api_params(kind, code, since, offset)
                )
                rows = r.json()
                raws.append(r.content)
                frames.append(X.parse_cot_api(rows, kind))
                if len(rows) < 5000:
                    break
                offset += 5000
    ref = _store(storage, "cftc_cot", "api.json", b"\n".join(raws), now)
    df = _pit(pl.concat(frames, how="diagonal_relaxed"), "cftc_cot", ref, now).with_columns(
        (pl.lit("MACRO:CFTC:") + pl.col("market")).alias("entity_id")
    )
    years = []
    for y in sorted(df["as_of"].dt.year().unique().to_list()):
        part = df.filter(pl.col("as_of").dt.year() == y)
        key = f"silver/cftc_cot/market=GLOBAL/year={y}.parquet"
        if storage.exists(key):
            old = storage.read_parquet(key)
            part = pl.concat(
                [
                    old.join(part.select("market", "as_of"), on=["market", "as_of"], how="anti"),
                    part,
                ],
                how="diagonal_relaxed",
            )
        storage.write_parquet(key, part.sort(["market", "as_of"]))
        years.append(y)
    counts = df.group_by("market").agg(pl.len(), pl.col("as_of").min()).sort("market").rows()
    if df.height == 0:
        raise RuntimeError("cftc: no rows")
    return {"rows": df.height, "years": years, "markets": [[m, n, str(d)] for m, n, d in counts]}


def run_pcr(storage: GuardedStorage, http: HttpClient, backfill: bool) -> dict[str, Any]:
    """Cboe put/call ratios: archived CSVs to 2019-10, then one JSON per session (yearly files)."""
    today = datetime.now(UTC).date()
    now = datetime.now(UTC)
    if backfill:
        frames = []
        for kind, name in (("total", "pc_total"), ("equity", "pc_equity"), ("index", "pc_index")):
            data = http.get(X.CBOE_CSV.format(kind=kind)).content
            _store(storage, "cboe_pcr", f"{kind}pc.csv", data, now)
            frames.append(X.parse_cboe_csv(data, name))
        old = (
            frames[0]
            .join(frames[1], on="as_of", how="full", coalesce=True)
            .join(frames[2], on="as_of", how="full", coalesce=True)
        )
        old = old.filter(pl.col("as_of") <= X.CBOE_CSV_END)
        for y in old["as_of"].dt.year().unique().to_list():
            part = old.filter(pl.col("as_of").dt.year() == y)
            _write_pcr(storage, y, part, "archive csv", now)
        start = X.CBOE_CSV_END + timedelta(days=1)
    else:
        start = today - timedelta(days=14)
    rows, missing = [], 0
    d = start
    while d <= today:
        if d.weekday() < 5:
            r = http.get(X.CBOE_DAILY.format(d=d.isoformat()))
            if r.status_code == 200:
                rows.append(X.parse_cboe_daily(r.content, d))
            else:
                missing += 1  # holidays / not yet published
        d += timedelta(days=1)
    if rows:
        new = pl.DataFrame(rows).with_columns(pl.col("as_of").cast(pl.Date))
        for y in new["as_of"].dt.year().unique().to_list():
            _write_pcr(storage, y, new.filter(pl.col("as_of").dt.year() == y), "daily json", now)
    return {"daily_rows": len(rows), "missing_days": missing, "from": str(start)}


def _write_pcr(
    storage: GuardedStorage, y: int, part: pl.DataFrame, ref: str, now: datetime
) -> None:
    key = f"silver/cboe_pcr/market=US/year={y}.parquet"
    part = _pit(
        part.with_columns(
            X.cboe_published_at(pl.col("as_of")).alias("published_at"),
            pl.lit("MACRO:CBOE:PCR").alias("entity_id"),
        ),
        "cboe_pcr",
        ref,
        now,
    )
    if storage.exists(key):
        old = storage.read_parquet(key)
        part = pl.concat(
            [old.filter(~pl.col("as_of").is_in(part["as_of"].implode())), part],
            how="diagonal_relaxed",
        )
    storage.write_parquet(key, part.sort("as_of"))


def run_sic(storage: GuardedStorage, backfill: bool) -> dict[str, Any]:
    """Company profiles (SIC) from the stored raw submissions batches; latest profile per CIK wins."""
    now = datetime.now(UTC)
    keys = sorted(k for k in storage.list("raw/sec_filings/") if "submissions" in k)
    prof: dict[int, dict[str, Any]] = {}
    for k in keys:
        for r in X.parse_profiles(storage.get_bytes(k)):
            prof[int(r["cik"])] = r  # type: ignore[call-overload]
    df = pl.DataFrame(
        list(prof.values()),
        schema={
            "cik": pl.Int64,
            "name": pl.String,
            "sic": pl.String,
            "sic_desc": pl.String,
            "tickers": pl.List(pl.String),
        },
    )
    df = (
        df.explode("tickers")
        .with_columns(
            pl.when(pl.col("tickers").is_not_null())
            .then(pl.lit("US:") + pl.col("tickers").str.replace_all("-", "."))
            .alias("entity_id")
        )
        .rename({"tickers": "ticker"})
    )
    df = _pit(df, "sec_company", f"{len(keys)} raw submissions batches", now)
    storage.write_parquet("silver/sec_company/market=US/latest.parquet", df)
    return {
        "companies": len(prof),
        "rows": df.height,
        "with_sic": int(df["sic"].is_not_null().sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="sp500,ftd,insider,shares,margin,cftc,pcr")
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    wiki = HttpClient(rate_per_host=0.5, user_agent=WIKI_UA)
    web = HttpClient(rate_per_host=2.0, user_agent="Mozilla/5.0 (MarketWeather personal research)")
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
        "margin": lambda: run_margin(storage, web, args.backfill),
        "cftc": lambda: run_cftc(storage, web, args.backfill),
        "pcr": lambda: run_pcr(storage, web, args.backfill),
        "sic": lambda: run_sic(storage, args.backfill),
    }
    for name in args.sources.split(","):
        try:
            with job_run(storage, f"reference_{name}", backfill=args.backfill) as s:
                s["result"] = runners[name]()
        except Exception:
            failed = True
    wiki.close()
    web.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
