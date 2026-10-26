"""Storage backends (DESIGN §5, §6). Keys are POSIX-style relative paths, e.g. 'silver/x/a.parquet'.

Backends: LocalFS (dev/CI) and S3Compatible (Cloudflare R2). Both are wrapped by GuardedStorage,
which enforces the app-side cost guard (ADR-0001) and immutability of raw/ and ledger/.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from mw.core.config import GuardSettings, StorageSettings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

IMMUTABLE_PREFIXES = ("raw/", "ledger/forecasts/")
USAGE_KEY = "registry/storage_usage.json"


class StorageError(RuntimeError):
    pass


class StorageGuardError(StorageError):
    """Raised when a write is refused by the cost guard."""


class Backend(Protocol):
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def total_bytes(self) -> int: ...


def _check_key(key: str) -> str:
    if key.startswith("/") or ".." in key.split("/") or "\\" in key:
        raise StorageError(f"invalid key: {key!r}")
    return key


class LocalFS:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise FileNotFoundError(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> list[str]:
        base = self.root
        out = [
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        ]
        return sorted(k for k in out if k.startswith(prefix))

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


class S3Compatible:
    """Cloudflare R2 (S3-compatible). Credentials from env; never logged."""

    def __init__(self, bucket: str, client: S3Client | None = None) -> None:
        self.bucket = bucket
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=os.environ["R2_ENDPOINT_URL"],
                aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
        self.client = client

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=_check_key(key), Body=data)

    def get_bytes(self, key: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=_check_key(key))
        except self.client.exceptions.NoSuchKey as e:
            raise FileNotFoundError(key) from e
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=_check_key(key), MaxKeys=1)
        return any(o.get("Key") == key for o in resp.get("Contents", []))

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(o["Key"] for o in page.get("Contents", []) if "Key" in o)
        return sorted(keys)

    def total_bytes(self) -> int:
        total = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            total += sum(o.get("Size", 0) for o in page.get("Contents", []))
        return total


class GuardedStorage:
    """Wraps a backend: immutability rules, op counting, soft-limit cost guard."""

    def __init__(self, backend: Backend, guard: GuardSettings) -> None:
        self.backend = backend
        self.guard = guard
        self._usage = self._load_usage()

    # --- usage accounting -------------------------------------------------
    @staticmethod
    def _month() -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    def _load_usage(self) -> dict[str, Any]:
        try:
            usage: dict[str, Any] = json.loads(self.backend.get_bytes(USAGE_KEY))
        except FileNotFoundError:
            usage = {}
        if usage.get("month") != self._month():
            usage = {"month": self._month(), "class_a": 0, "class_b": 0, "bytes": None}
        return usage

    def flush_usage(self) -> None:
        """Persist op counters (call once at end of a job). Counts itself as one Class A op."""
        self._usage["class_a"] += 1
        self._usage["updated_at"] = datetime.now(UTC).isoformat()
        self.backend.put_bytes(USAGE_KEY, json.dumps(self._usage).encode())

    def refresh_size(self) -> int:
        size = self.backend.total_bytes()
        self._usage["bytes"] = size
        self._usage["class_a"] += 1
        return size

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self._usage)

    def _check_write(self, key: str, nbytes: int) -> None:
        if any(key.startswith(p) for p in self.guard.exempt_prefixes):
            return
        g, u = self.guard, self._usage
        if u["class_a"] + 1 > g.soft_limit_class_a_per_month:
            raise StorageGuardError(f"Class A soft limit reached ({u['class_a']}); refused {key}")
        if u.get("bytes") is not None and u["bytes"] + nbytes > g.soft_limit_bytes:
            raise StorageGuardError(f"storage soft limit reached ({u['bytes']} B); refused {key}")

    def _check_read(self) -> None:
        if self._usage["class_b"] + 1 > self.guard.soft_limit_class_b_per_month:
            raise StorageGuardError("Class B soft limit reached")

    # --- operations -------------------------------------------------------
    def put_bytes(self, key: str, data: bytes, *, overwrite: bool = True) -> None:
        immutable = key.startswith(IMMUTABLE_PREFIXES)
        if (immutable or not overwrite) and self.backend.exists(key):
            raise StorageError(f"refusing to overwrite immutable key: {key}")
        self._check_write(key, len(data))
        self.backend.put_bytes(key, data)
        self._usage["class_a"] += 1
        if self._usage.get("bytes") is not None:
            self._usage["bytes"] += len(data)

    def get_bytes(self, key: str) -> bytes:
        self._check_read()
        self._usage["class_b"] += 1
        return self.backend.get_bytes(key)

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)

    def list(self, prefix: str) -> list[str]:
        self._usage["class_a"] += 1  # LIST is a Class A op on R2
        return self.backend.list(prefix)

    def write_parquet(self, key: str, df: pl.DataFrame, *, overwrite: bool = True) -> None:
        buf = io.BytesIO()
        df.write_parquet(buf, compression="zstd")
        self.put_bytes(key, buf.getvalue(), overwrite=overwrite)

    def read_parquet(self, key: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
        return pl.read_parquet(
            io.BytesIO(self.get_bytes(key)), columns=None if columns is None else [*columns]
        )


def open_storage(settings: StorageSettings) -> GuardedStorage:
    backend: Backend
    if settings.backend == "local":
        backend = LocalFS(settings.local_root)
    else:
        backend = S3Compatible(settings.r2.bucket)
    return GuardedStorage(backend, settings.guard)
