"""Daily paper (demo) trading for one market. Forward-only; idempotent per session.

Usage: python -m jobs.paper_trade --market US|JP [--start YYYY-MM-DD]  (--start only for replays)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from typing import Any

import polars as pl
import yaml

from mw.core.config import REPO_ROOT, load_settings
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.paper import trader
from mw.report import dashboard as D
from mw.strategy import engine, library, regime

BENCH = {"US": "US:SPY", "JP": "JP:1306"}


def load_config(market: str) -> list[dict[str, Any]]:
    cfg = yaml.safe_load((REPO_ROOT / "config/strategies/paper.yaml").read_text(encoding="utf-8"))
    return list(cfg.get(market, []))


def recent_years(storage: GuardedStorage, table: str, market: str, years: int = 2) -> pl.DataFrame:
    y0 = datetime.now(UTC).year - years
    keys = [k for k in storage.list(f"silver/{table}/market={market}/") if k.endswith(".parquet")]
    keys = [k for k in keys if int(k.split("year=")[1][:4]) >= y0]
    return D._read_many(storage, keys)


def build_ind(storage: GuardedStorage, market: str) -> pl.DataFrame:
    cols = ["entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"]
    px = recent_years(storage, "prices_daily", market).select(cols)
    ind = engine.indicators(px, BENCH[market])
    vol = D._read_years(storage, "vol_indices", "US")
    fred = D._read_years(storage, "fred", "GLOBAL")
    us_px = (
        px
        if market == "US"
        else recent_years(storage, "prices_daily", "US")
        .select("entity_id", "as_of", "close")
        .filter(pl.col("entity_id").str.contains(r"^US:[A-Z]{2,4}$"))
    )
    usr = regime.us_regime(us_px, vol, fred)
    reg = usr if market == "US" else regime.jp_regime(px, usr, fred)
    reg = reg.join(regime.breadth(ind), on="as_of", how="left")
    return ind.join(reg, on="as_of", how="left")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["US", "JP"], required=True)
    p.add_argument("--start", type=date.fromisoformat, default=None)
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    with job_run(storage, f"paper_{args.market.lower()}") as s:
        cfg = load_config(args.market)
        strategies = {
            c["key"]: library.by_key(args.market, BENCH[args.market], c["key"]) for c in cfg
        }
        ind = build_ind(storage, args.market)
        s["result"] = trader.run(storage, args.market, strategies, ind, start=args.start)
        s["tiers"] = {c["key"]: c["tier"] for c in cfg}
    return 0


if __name__ == "__main__":
    sys.exit(main())
