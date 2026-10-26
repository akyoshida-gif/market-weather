"""US/global daily ingest: FINRA Reg SHO daily, FINRA short interest, Cboe vol indices, FRED.

Usage: python -m jobs.us_daily [--sources ...] [--backfill-days N]
Each source isolated (P6). Backfill: --backfill-days 750 (~2y Reg SHO), SI/vol/FRED full history.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from mw.core.config import load_settings
from mw.core.http import HttpClient
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest.macro import fred
from mw.ingest.us import cboe, finra

SI_STALE_DAYS = 35  # settlement -> publication can take ~26 days incl. month-end gap


def run_regsho(
    storage: GuardedStorage, http: HttpClient, backfill_days: int | None
) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    days = (
        finra.recent_sessions(today, 5)
        if backfill_days is None
        else [
            d
            for d in finra.recent_sessions(today, backfill_days * 5 // 7 + 5)
            if d >= today - timedelta(days=backfill_days)
        ]
    )
    days = [d for d in days if d >= date(2018, 8, 1)]  # FINRA consolidated (CNMS) files start here
    done, pending, errors = [], [], {}
    for d in days:
        try:
            r = finra.ingest_regsho_day(storage, http, d)
            if r:
                done.append(r["date"])
        except finra.NotPublished:
            pending.append(d.isoformat())
        except Exception as e:
            errors[d.isoformat()] = f"{type(e).__name__}: {e}"[:200]
    # only the most recent session may legitimately be unpublished
    late = [p for p in pending if date.fromisoformat(p) < days[-1]] if days else []
    if errors or late:
        raise RuntimeError(f"regsho errors={errors} missing_older={late}")
    return {"ingested": done, "pending": pending}


def run_short_interest(
    storage: GuardedStorage, http: HttpClient, backfill_days: int | None
) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    start = today - timedelta(days=backfill_days or 60)
    done, pending = [], []
    for settle in finra.settlement_dates(start, today):
        try:
            r = finra.ingest_si(storage, http, settle)
            if r:
                done.append(r["settlement"])
        except finra.NotPublished:
            pending.append(settle.isoformat())
    existing = [
        k for k in storage.list("silver/finra_short_interest/")
        if k.endswith(".parquet")
    ]  # fmt: skip
    latest = max((k.rsplit("/", 1)[-1][:10] for k in existing), default=None)
    stale = latest is None or (today - date.fromisoformat(latest)).days > SI_STALE_DAYS
    if stale:
        raise RuntimeError(f"short interest stale (latest={latest}); schedule may have changed")
    return {"ingested": done, "pending": pending, "latest": latest}


def run_cboe(
    storage: GuardedStorage, http: HttpClient, backfill_days: int | None
) -> dict[str, Any]:
    return cboe.ingest(storage, http, since_days=None if backfill_days else 30)


def run_fred(
    storage: GuardedStorage, http: HttpClient, backfill_days: int | None
) -> dict[str, Any]:
    return fred.ingest(storage, http, since_days=None if backfill_days else 60)


RUNNERS: dict[str, Callable[[GuardedStorage, HttpClient, int | None], dict[str, Any]]] = {
    "finra_regsho": run_regsho,
    "finra_si": run_short_interest,
    "cboe_vol": run_cboe,
    "fred": run_fred,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=",".join(RUNNERS))
    p.add_argument("--backfill-days", type=int, default=None)
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    http = HttpClient(rate_per_host=1.0)
    failed = False
    for name in args.sources.split(","):
        try:
            with job_run(storage, f"us_{name}", backfill_days=args.backfill_days) as s:
                s["result"] = RUNNERS[name](storage, http, args.backfill_days)
        except Exception:
            failed = True
    http.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
