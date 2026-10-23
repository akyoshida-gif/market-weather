"""Measure total bucket size and persist usage (enables the size soft limit, ADR-0001)."""

from __future__ import annotations

from mw.core.config import load_settings
from mw.core.logging import job_run, setup_logging
from mw.core.storage import open_storage


def main() -> None:
    setup_logging()
    storage = open_storage(load_settings().storage)
    with job_run(storage, "storage_audit") as s:
        s["bytes"] = storage.refresh_size()
        s["usage"] = storage.usage


if __name__ == "__main__":
    main()
