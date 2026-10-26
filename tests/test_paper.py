"""Paper trading: order -> fill at next open -> exit; sizing; idempotency."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, LocalFS
from mw.paper import trader
from mw.strategy import engine

C = pl.col


def ind(rows):
    d0 = date(2026, 1, 5)
    n = len(rows)
    df = pl.DataFrame({
        "entity_id": ["US:X"] * n, "as_of": [d0 + timedelta(days=i) for i in range(n)],
        "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows],
        "close": [r[3] for r in rows], "volume": [1e6] * n, "bad": [False] * n,
        "sig": [i == 0 for i in range(n)],
    })  # fmt: skip
    return df


def strat():
    return engine.Strategy(
        key="t", label="t", horizon="短期", side="long", entry=lambda: C("sig"),
        stop=lambda: C("close") - 5, target=lambda: C("close") + 10, max_hold=5,
    )  # fmt: skip


def test_order_fill_target_and_idempotent(tmp_path):
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    df = ind([(100, 101, 99, 100), (100, 103, 99, 102), (102, 111, 101, 110), (110, 111, 109, 110)])
    r = trader.run(s, "US", {"t": strat()}, df, start=date(2026, 1, 5))
    b = r["books"]["t"]
    assert b["closed"] == 1 and b["open"] == 0
    # risk 1% of 100k = 1000 / 5 = 200 shares, but capped at 10% notional = 100 shares
    assert (
        abs(b["equity"] - (100_000 + 100 * (110 - 100) - 100 * 100 * 5 / 1e4 - 100 * 110 * 5 / 1e4))
        < 1e-6
    )
    assert trader.run(s, "US", {"t": strat()}, df)["sessions"] == []


def test_skip_fill_when_open_below_stop(tmp_path):
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    df = ind([(100, 101, 99, 100), (94, 95, 93, 94), (94, 95, 93, 94)])
    r = trader.run(s, "US", {"t": strat()}, df, start=date(2026, 1, 5))
    assert r["books"]["t"]["closed"] == 0 and r["books"]["t"]["open"] == 0
