from __future__ import annotations

import sys

from jobs import dummy_pit


def test_dummy_job_local(tmp_path, monkeypatch):
    monkeypatch.setenv("MW_STORAGE_BACKEND", "local")
    monkeypatch.setenv("MW_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["dummy_pit", "--date", "2026-01-05"])
    dummy_pit.main()
    assert (tmp_path / "silver/dummy/market=US/year=2026/month=01/2026-01-05.parquet").exists()
    assert list((tmp_path / "registry/runs").rglob("*.json"))
