"""Universe snapshot + daily prices for one market.

Usage: python -m jobs.prices_daily --market US|JP [--days 7] [--start YYYY-MM-DD] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
import yaml

from mw.core.config import REPO_ROOT, load_settings
from mw.core.http import HttpClient
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest import prices
from mw.ingest.base import raw_key, write_raw_bytes, year_key
from mw.universe import jp, us


def universe_key(market: str, d: date) -> str:
    return f"silver/universe_membership/market={market}/date={d.isoformat()}.parquet"


def latest_universe(storage: GuardedStorage, market: str) -> pl.DataFrame:
    keys = storage.list(f"silver/universe_membership/market={market}/")
    if not keys:
        raise RuntimeError(f"no universe snapshot for {market}")
    return storage.read_parquet(sorted(keys)[-1])


def snapshot_us(storage: GuardedStorage, http: HttpClient) -> dict[str, Any]:
    now = datetime.now(UTC)
    nas = http.get(us.NASDAQ_URL).content
    oth = http.get(us.OTHER_URL).content
    for name, data in (("nasdaqlisted", nas), ("otherlisted", oth)):
        write_raw_bytes(storage, raw_key("us_universe", now, f"{name}_{now:%H%M%S}.txt"), data)
    df = us.build(nas, oth, us.load_etfs(), now.date())
    storage.write_parquet(universe_key("US", now.date()), df)
    return {"common": int((df["kind"] == "common").sum()), "etf": int((df["kind"] == "etf").sum())}


def snapshot_jp(storage: GuardedStorage, http: HttpClient) -> dict[str, Any]:
    now = datetime.now(UTC)
    link = jp.find_link(http.get(jp.INDEX).text)
    data = http.get(link).content
    cfg = yaml.safe_load((REPO_ROOT / "config/universe/jp_etfs.yaml").read_text(encoding="utf-8"))
    etfs = [c for group in cfg.values() for c in group]
    df = jp.build(data, etfs, now.date())
    d = jp.snapshot_date(df)
    key = universe_key("JP", d)
    if storage.exists(key):
        return {"skipped": f"snapshot {d} exists"}
    write_raw_bytes(storage, raw_key("jp_universe", now, f"data_j_{d}_{now:%H%M%S}.xlsx"), data)
    storage.write_parquet(key, df)
    return {
        "as_of": d.isoformat(),
        "common": int((df["kind"] == "common").sum()),
        "etf": int((df["kind"] == "etf").sum()),
    }


def repair_splits(
    storage: GuardedStorage, market: str, uni: pl.DataFrame, since: date, end: date
) -> dict[str, Any]:
    """Stocks with a split since `since`: download their whole history again.

    Their older rows may have been stored before the split; refetching makes every stored row
    adjusted the same way (BACKLOG D5).
    """
    ids: set[str] = set()
    for y in range(since.year, end.year + 1):
        key = year_key("prices_daily", market, y)
        if not storage.exists(key):
            continue
        df = storage.read_parquet(key).filter(pl.col("as_of") >= since)
        ids |= set(prices.split_entities(df))
    sub = uni.filter(pl.col("entity_id").is_in(sorted(ids)))
    if sub.height == 0:
        return {"split_stocks": 0}
    r = prices.ingest(storage, market, sub, prices.FULL_START, end, refetch_splits=False)
    return {"split_stocks": sub.height, "rows": r["rows"], "missing": r["missing"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["US", "JP"], required=True)
    p.add_argument("--days", type=int, default=7, help="re-download the last N calendar days")
    p.add_argument("--start", type=date.fromisoformat, default=None, help="backfill start date")
    p.add_argument("--limit", type=int, default=None, help="only first N symbols (testing)")
    p.add_argument("--skip-universe", action="store_true")
    p.add_argument(
        "--repair-splits-since",
        type=date.fromisoformat,
        default=None,
        help="one-off: download the full history again for stocks with a split since this date",
    )
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    http = HttpClient(rate_per_host=0.5)
    failed = False
    if not args.skip_universe:
        try:
            with job_run(storage, f"universe_{args.market.lower()}") as s:
                s["result"] = (snapshot_us if args.market == "US" else snapshot_jp)(storage, http)
        except Exception:
            failed = True
    try:
        with job_run(
            storage, f"prices_{args.market.lower()}", start=str(args.start), days=args.days
        ) as s:
            uni = latest_universe(storage, args.market)
            if args.limit:
                uni = uni.head(args.limit)
            end = datetime.now(UTC).date()
            if args.repair_splits_since:
                s["result"] = repair_splits(
                    storage, args.market, uni, args.repair_splits_since, end
                )
            else:
                start = args.start or end - timedelta(days=args.days)
                s["result"] = prices.ingest(storage, args.market, uni, start, end)
    except Exception:
        failed = True
    http.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
