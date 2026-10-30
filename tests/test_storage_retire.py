from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobs import storage_maintenance as M
from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, LocalFS, StorageError
from mw.ingest.base import manifest_key, remote_raw_ref


def _storage(tmp_path: Path) -> GuardedStorage:
    return GuardedStorage(LocalFS(tmp_path), GuardSettings())


def test_ledger_can_never_be_deleted(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    s.put_bytes("ledger/forecasts/x.json", b"{}")
    with pytest.raises(StorageError):
        s.delete("ledger/forecasts/x.json")
    with pytest.raises(StorageError):
        s.delete("ledger/forecasts/x.json", retire_raw=True)


def test_raw_needs_explicit_retire(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    s.put_bytes("raw/a/dt=2026-01-01/f.zip", b"abc", overwrite=False)
    with pytest.raises(StorageError):
        s.delete("raw/a/dt=2026-01-01/f.zip")
    s.delete("raw/a/dt=2026-01-01/f.zip", retire_raw=True)
    assert not s.exists("raw/a/dt=2026-01-01/f.zip")


def test_remote_raw_ref_keeps_url_and_hash_only(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    now = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    ref = remote_raw_ref(
        s, "sec_insider", "2015q1_form345_120000.zip", "https://www.sec.gov/x.zip", b"hello", now
    )
    assert ref.startswith("raw/sec_insider/dt=2026-10-30/2015q1_form345_120000.zip#sha256=")
    assert not s.exists("raw/sec_insider/dt=2026-10-30/2015q1_form345_120000.zip")  # no copy kept
    m = s.read_parquet(manifest_key("sec_insider"))
    assert m["url"].to_list() == ["https://www.sec.gov/x.zip"]
    assert m["sha256"].to_list() == [ref.split("sha256=")[1]]


def test_prune_research_keeps_listed_runs(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    for run in ("20261027T0409", "20261030T0447"):
        s.put_bytes(f"research/event_study/{run}/a.json", b"{}")
    dry = M.prune_research(s, {"20261030T0447"}, execute=False)
    assert dry["deleted_runs"] == ["20261027T0409"] and s.exists(
        "research/event_study/20261027T0409/a.json"
    )
    M.prune_research(s, {"20261030T0447"}, execute=True)
    assert not s.exists("research/event_study/20261027T0409/a.json")
    assert s.exists("research/event_study/20261030T0447/a.json")
