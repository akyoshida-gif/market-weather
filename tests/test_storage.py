from __future__ import annotations

import polars as pl
import pytest

from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, LocalFS, StorageError, StorageGuardError


def make(tmp_path, **guard):
    return GuardedStorage(LocalFS(tmp_path), GuardSettings(**guard))


def test_roundtrip_parquet(tmp_path):
    s = make(tmp_path)
    df = pl.DataFrame({"a": [1, 2]})
    s.write_parquet("silver/t/x.parquet", df)
    assert s.read_parquet("silver/t/x.parquet").equals(df)
    assert s.list("silver/") == ["silver/t/x.parquet"]


def test_raw_and_ledger_are_immutable(tmp_path):
    s = make(tmp_path)
    for key in ("raw/src/dt=2026-01-01/a.bin", "ledger/forecasts/date=2026-01-01/f.parquet"):
        s.put_bytes(key, b"1")
        with pytest.raises(StorageError):
            s.put_bytes(key, b"2")


def test_invalid_keys_rejected(tmp_path):
    s = make(tmp_path)
    for key in ("/abs", "a/../b", "a" + chr(92) + "b"):
        with pytest.raises(StorageError):
            s.put_bytes(key, b"x")


def test_guard_blocks_non_exempt_writes(tmp_path):
    s = make(tmp_path, soft_limit_class_a_per_month=2)
    s.put_bytes("silver/a", b"x")
    s.put_bytes("silver/b", b"x")
    with pytest.raises(StorageGuardError):
        s.put_bytes("silver/c", b"x")
    s.put_bytes("ledger/manifest.jsonl", b"x")  # exempt


def test_guard_size_limit(tmp_path):
    s = make(tmp_path, soft_limit_bytes=10)
    s.refresh_size()
    with pytest.raises(StorageGuardError):
        s.put_bytes("silver/big", b"x" * 11)


def test_usage_persisted(tmp_path):
    s = make(tmp_path)
    s.put_bytes("silver/a", b"x")
    s.flush_usage()
    s2 = make(tmp_path)
    assert s2.usage["class_a"] >= 2
