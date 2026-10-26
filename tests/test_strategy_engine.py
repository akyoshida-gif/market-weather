"""Trade simulation: IN/TP/LS mechanics, gaps, costs, bad-data exclusion."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from mw.strategy import engine

C = pl.col


def bars(rows, eid="US:X", splits=None):
    d0 = date(2026, 1, 5)
    n = len(rows)
    return pl.DataFrame({
        "entity_id": [eid] * n, "as_of": [d0 + timedelta(days=i) for i in range(n)],
        "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows],
        "close": [r[3] for r in rows], "volume": [1e6] * n, "splits": splits or [0.0] * n,
    })  # fmt: skip


def strat(side="long", stop=95.0, target=110.0, hold=5):
    return engine.Strategy(
        key="t", label="t", horizon="短期", side=side,
        entry=lambda: C("as_of") == date(2026, 1, 5),
        stop=lambda: pl.lit(stop), target=lambda: pl.lit(target), max_hold=hold,
    )  # fmt: skip


def frame(df):
    return df.with_columns(pl.lit(False).alias("bad")) if "bad" not in df.columns else df


def test_long_target_hit_with_cost():
    df = bars([(100, 101, 99, 100), (100, 102, 99, 101), (101, 111, 100, 110)])
    tr = engine.simulate(frame(df), strat(), "US")
    r = tr.row(0, named=True)
    assert r["reason"] == "target" and r["exit"] == 110.0 and r["entry"] == 100.0
    assert abs(r["ret"] - (0.10 - 10 / 1e4)) < 1e-12


def test_stop_first_when_both_touched_and_gap_fill():
    df = bars([(100, 101, 99, 100), (100, 111, 94, 100)])
    assert engine.simulate(frame(df), strat(), "US").row(0, named=True)["reason"] == "stop"
    df2 = bars([(100, 101, 99, 100), (100, 101, 99, 100), (90, 91, 89, 90)])  # gap below stop
    r = engine.simulate(frame(df2), strat(), "US").row(0, named=True)
    assert r["reason"] == "stop" and r["exit"] == 90.0


def test_no_trade_if_open_gaps_through_stop():
    df = bars([(100, 101, 99, 100), (94, 95, 93, 94)])
    assert engine.simulate(frame(df), strat(), "US").height == 0


def test_short_side():
    df = bars([(100, 101, 99, 100), (100, 101, 89, 90)])
    r = engine.simulate(frame(df), strat("short", stop=105, target=90), "US").row(0, named=True)
    assert r["reason"] == "target" and abs(r["ret"] - (100 / 90 - 1 - 10 / 1e4)) < 1e-12


def test_bad_data_window_dropped():
    df = bars([(100, 101, 99, 100), (100, 102, 99, 101), (101, 105, 100, 104)])
    df = df.with_columns(pl.Series("bad", [False, False, True]))
    assert engine.simulate(df, strat(), "US").height == 0
