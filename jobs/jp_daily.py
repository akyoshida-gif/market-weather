"""Japan daily ingest: JSF lending/premium, JPX short positions. Each source isolated (P6).

Usage: python -m jobs.jp_daily [--sources jsf,jpx_short_positions] [--backfill]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mw.core.config import load_settings
from mw.core.http import HttpClient
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest.jp import jpx_short_positions, jsf


def run_jsf(storage: GuardedStorage, http: HttpClient, backfill: bool) -> dict[str, Any]:
    now = datetime.now(UTC)
    return jsf.persist(storage, jsf.fetch(http), now)


def run_jpx_short_positions(
    storage: GuardedStorage, http: HttpClient, backfill: bool
) -> dict[str, Any]:
    links = jpx_short_positions.list_files(http, include_archives=backfill)
    ingested: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for link in links:
        try:
            r = jpx_short_positions.ingest_file(storage, http, link)
            if r:
                ingested.append(r)
        except Exception as e:  # one bad file must not block the rest
            errors[link.disclosure_date.isoformat()] = f"{type(e).__name__}: {e}"[:300]
    if errors and not ingested:
        raise RuntimeError(f"all new files failed: {errors}")
    return {"listed": len(links), "ingested": ingested, "errors": errors}


RUNNERS: dict[str, Callable[[GuardedStorage, HttpClient, bool], dict[str, Any]]] = {
    "jsf": run_jsf,
    "jpx_short_positions": run_jpx_short_positions,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=",".join(RUNNERS))
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    http = HttpClient(rate_per_host=0.5)
    failed = False
    for name in args.sources.split(","):
        try:
            with job_run(storage, f"jp_{name}", backfill=args.backfill) as s:
                s["result"] = RUNNERS[name](storage, http, args.backfill)
        except Exception:
            failed = True  # logged by job_run; continue with other sources
    http.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
