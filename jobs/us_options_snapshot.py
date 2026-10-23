"""Capture a US option chain snapshot for the watchlist.

Usage: python -m jobs.us_options_snapshot --snapshot pre|post [--limit N] [--force]
Exit code 1 if fewer than --min-success of underlyings succeeded (data still written).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from mw.core.config import load_settings
from mw.core.http import RateLimiter
from mw.core.logging import get_logger, job_run, setup_logging
from mw.core.storage import open_storage
from mw.ingest.us.options_chain import (
    YFinanceProvider,
    capture,
    is_degraded,
    last_completed_session,
    load_watchlist,
    persist,
    quality_metrics,
    silver_chain_key,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", choices=["pre", "post"], required=True)
    p.add_argument("--limit", type=int, default=None, help="only first N symbols (testing)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--rate", type=float, default=3.0, help="requests/second")
    p.add_argument("--min-success", type=float, default=0.8)
    p.add_argument("--force", action="store_true", help="re-capture even if snapshot exists")
    args = p.parse_args()

    setup_logging()
    log = get_logger("us_options_snapshot")
    settings = load_settings()
    storage = open_storage(settings.storage)
    symbols, max_dte = load_watchlist()
    if args.limit:
        symbols = symbols[: args.limit]

    now = datetime.now(UTC)
    as_of = last_completed_session(now)
    exit_code = 0
    with job_run(storage, f"us_options_{args.snapshot}", as_of=as_of.isoformat()) as s:
        if storage.exists(silver_chain_key(as_of, args.snapshot)) and not args.force:
            s["skipped"] = "snapshot already exists"
            log.info("skip", extra={"data": {"as_of": as_of.isoformat()}})
            return 0
        provider = YFinanceProvider(RateLimiter(args.rate, burst=args.workers))
        res = capture(provider, symbols, args.snapshot, max_dte, now, workers=args.workers)
        s["symbols"] = len(symbols)
        s["ok"] = len(res.ok)
        s["failed"] = res.failed
        q = quality_metrics(res.chains)
        s["quality"] = q
        degraded = is_degraded(q)
        s["degraded"] = degraded
        if degraded:  # likely off-hours provider reset; archived under *_degraded, retry allowed
            log.warning("degraded snapshot", extra={"data": q})
        s["persisted"] = persist(storage, res, degraded=degraded)
        rate = len(res.ok) / max(1, len(symbols))
        s["success_rate"] = round(rate, 3)
        if rate < args.min_success:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
