"""Contract tests for the options adapter using a synthetic provider (no network, no Yahoo data)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import polars as pl

from mw.core.config import GuardSettings
from mw.core.pit import validate_pit_frame
from mw.core.storage import GuardedStorage, LocalFS
from mw.ingest.us import options_chain as oc


class FakeProvider:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()

    def expiries(self, symbol: str) -> list[str]:
        if symbol in self.fail:
            raise RuntimeError("boom")
        return ["2026-01-09", "2026-02-20", "2026-12-18", "2027-01-15"]

    def chain(self, symbol, expiry):
        def side(kind):
            return pd.DataFrame(
                {
                    "contractSymbol": [f"{symbol}{expiry}{kind}{k}" for k in (90, 100)],
                    "lastTradeDate": pd.to_datetime(["2026-01-05 20:59", "2026-01-05 20:58"]),
                    "strike": [90.0, 100.0],
                    "lastPrice": [1.0, 2.0],
                    "bid": [0.9, 1.9],
                    "ask": [1.1, 2.1],
                    "change": [0.1, -0.1],
                    "percentChange": [1.0, -1.0],
                    "volume": [10, None],
                    "openInterest": [100, 0],
                    "impliedVolatility": [0.3, 0.00001],
                    "inTheMoney": [True, False],
                    "contractSize": ["REGULAR", "REGULAR"],
                    "currency": ["USD", "USD"],
                }
            )

        return side("C"), side("P"), {"regularMarketPrice": 99.5, "marketState": "POST"}


CAPTURED = datetime(2026, 1, 6, 0, 30, tzinfo=UTC)  # after 2026-01-05 close


def test_last_completed_session():
    assert oc.last_completed_session(CAPTURED) == date(2026, 1, 5)
    # pre-market Monday -> previous Friday
    assert oc.last_completed_session(datetime(2026, 1, 5, 12, 30, tzinfo=UTC)) == date(2026, 1, 2)
    # during session -> previous session
    assert oc.last_completed_session(datetime(2026, 1, 5, 16, 0, tzinfo=UTC)) == date(2026, 1, 2)


def test_select_expiries():
    exps = ["2026-01-02", "2026-01-09", "2026-06-19", "2026-12-18", "2027-01-15"]
    assert oc.select_expiries(exps, date(2026, 1, 5), 200) == [
        "2026-01-09",
        "2026-06-19",
        "2026-12-18",
    ]


def test_capture_and_persist(tmp_path):
    res = oc.capture(FakeProvider(fail={"BAD"}), ["AAA", "BAD"], "post", 200, CAPTURED, workers=2)
    assert res.ok == ["AAA"] and "BAD" in res.failed
    assert res.as_of == date(2026, 1, 5)
    # 2026-01-09, 2026-02-20 (<=200d), 2026-12-18 (first beyond) -> 3 x 2 sides x 2 strikes
    assert res.chains.height == 12
    q = oc.quality_metrics(res.chains)
    assert q["pct_oi_zero"] == 0.5 and q["pct_iv_placeholder"] == 0.5

    storage = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    out = oc.persist(storage, res)
    silver = storage.read_parquet(out["chains"]["silver"])
    validate_pit_frame(silver)
    assert silver["entity_id"].unique().to_list() == ["US:AAA"]
    assert silver.filter(pl.col("implied_vol").is_null()).height == 6  # placeholders nulled
    assert silver["dte"].min() == 4
    assert silver["published_at"].max() == CAPTURED
    raw = storage.read_parquet(out["chains"]["raw"])
    assert raw["implied_vol"].min() < 1e-4  # raw untouched


def test_degraded_goes_to_separate_key(tmp_path):
    res = oc.capture(FakeProvider(), ["AAA"], "pre", 200, CAPTURED)
    storage = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    out = oc.persist(storage, res, degraded=True)
    assert out["chains"]["silver"].endswith("_pre_degraded.parquet")
    assert not storage.exists(oc.silver_chain_key(res.as_of, "pre"))
    assert oc.is_degraded({"pct_oi_zero": 0.9, "pct_iv_placeholder": 0.0})
