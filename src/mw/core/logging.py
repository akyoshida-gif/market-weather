"""Structured JSON logging + per-job run summary appended to registry/runs.jsonl."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from mw.core.storage import GuardedStorage

RUNS_PREFIX = "registry/runs"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def job_run(storage: GuardedStorage, job: str, **params: Any) -> Iterator[dict[str, Any]]:
    """Record a run summary. Each run is its own object (no read-modify-write on a shared file)."""
    started = datetime.now(UTC)
    t0 = time.monotonic()
    summary: dict[str, Any] = {"job": job, "params": params, "started_at": started.isoformat()}
    log = get_logger(job)
    try:
        yield summary
        summary["status"] = "ok"
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = f"{type(e).__name__}: {e}"
        log.exception("job failed", extra={"data": {"job": job}})
        raise
    finally:
        summary["duration_s"] = round(time.monotonic() - t0, 3)
        key = f"{RUNS_PREFIX}/dt={started:%Y-%m-%d}/{job}_{started:%H%M%S%f}.json"
        storage.put_bytes(key, json.dumps(summary, default=str).encode())
        storage.flush_usage()
        log.info("job finished", extra={"data": summary})
