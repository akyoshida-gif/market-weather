"""Storage maintenance (ADR-0019). Dry run by default; --execute actually deletes.

Usage:
  python -m jobs.storage_maintenance prune-research --keep RUN[,RUN...] [--execute]
      delete research/event_study/<run>/ folders not listed in --keep (results can be regenerated)
  python -m jobs.storage_maintenance retire-raw --source sec_insider [--execute]
      for every raw/<source>/ object: download, compute sha256, find the SEC URL it came from, write the
      manifest (raw_manifest/<source>.parquet), then delete the copy. Objects whose URL cannot be
      determined are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import UTC, datetime
from typing import Any

from mw.core.config import load_settings
from mw.core.logging import job_run, setup_logging
from mw.core.storage import GuardedStorage, open_storage
from mw.ingest.base import record_manifest
from mw.ingest.us import sec_filings
from mw.ingest.us import sec_reference as R

QUARTER_RE = re.compile(r"(\d{4}q[1-4])_form345", re.I)


def prune_research(storage: GuardedStorage, keep: set[str], execute: bool) -> dict[str, Any]:
    keys = storage.list("research/event_study/")
    runs = sorted({k.split("/")[2] for k in keys})
    drop = [r for r in runs if r not in keep]
    targets = [k for k in keys if k.split("/")[2] in drop]
    if execute:
        for k in targets:
            storage.delete(k)
    return {
        "runs": runs,
        "kept": sorted(keep & set(runs)),
        "deleted_runs": drop,
        "objects": len(targets),
        "executed": execute,
    }


def retire_raw(storage: GuardedStorage, source: str, execute: bool) -> dict[str, Any]:
    if source != "sec_insider":
        raise ValueError("only sec_insider is supported (the SEC keeps these files published)")
    sec = sec_filings.SecClient()
    urls = {q: u for u, q in R.insider_links(sec.get(R.INSIDER_PAGE).decode("utf-8", "replace"))}
    keys = storage.list(f"raw/{source}/")
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    total = 0
    for k in keys:
        m = QUARTER_RE.search(k)
        url = urls.get(m.group(1).lower()) if m else None
        if url is None:
            skipped.append(k)
            continue
        data = storage.get_bytes(k)
        total += len(data)
        rows.append({"raw_key": k, "url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                     "recorded_at": datetime.now(UTC).isoformat()})  # fmt: skip
    if execute:
        record_manifest(storage, source, rows)  # the manifest is written BEFORE anything is deleted
        for r in rows:
            storage.delete(str(r["raw_key"]), retire_raw=True)
    return {
        "objects": len(keys),
        "retired": len(rows),
        "bytes": total,
        "kept_unknown_url": skipped[:20],
        "executed": execute,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prune-research")
    a.add_argument("--keep", required=True)
    a.add_argument("--execute", action="store_true")
    b = sub.add_parser("retire-raw")
    b.add_argument("--source", required=True)
    b.add_argument("--execute", action="store_true")
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    with job_run(storage, f"storage_{args.cmd}") as s:
        if args.cmd == "prune-research":
            s["result"] = prune_research(storage, set(args.keep.split(",")), args.execute)
        else:
            s["result"] = retire_raw(storage, args.source, args.execute)
        s["result"]["size_after_gb"] = round(storage.refresh_size() / 1e9, 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
