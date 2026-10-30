from datetime import date, timedelta

import numpy as np
import polars as pl

from mw.research import tradesim


def _panel(
    closes: list[float], lows: list[float] | None = None, highs: list[float] | None = None
) -> tradesim.Panel:
    n = len(closes)
    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(n)]
    lo = lows or [c - 0.5 for c in closes]
    hi = highs or [c + 0.5 for c in closes]
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * n,
            "as_of": days,
            "open": closes,
            "high": hi,
            "low": lo,
            "close": closes,
            "splits": [0.0] * n,
        }
    )
    return tradesim.build_panel(df)


def test_time_exit_and_costs() -> None:
    # flat at 100 for 30 days (ATR ~1), then rises 1 per day
    closes = [100.0] * 30 + [100.0 + i for i in range(1, 40)]
    p = _panel(closes)
    sig = np.array([29])
    sim = tradesim.simulate(p, sig, "atr2", "time", 5)
    # entry = open of row 30 = 101; after 5 sessions the close is 105 -> +4 / R(=2*ATR)
    R = 101 - (101 - 2 * p.atr[29])
    assert sim["valid"][0]
    assert sim["hold"][0] == 5
    assert abs(sim["r"][0] - ((105 - 101) / R - 0.001 * 101 / R)) < 1e-6


def test_stop_is_hit_and_gap_exits_at_open() -> None:
    closes = [100.0] * 30 + [100.0, 99.0, 90.0, 95.0] + [95.0] * 10
    p = _panel(closes)
    sim = tradesim.simulate(p, np.array([29]), "atr2", "chand", 3.0)
    # entry 100 (open of row 30); stop = 100 - 2*ATR ~ 98; row 32 opens at 90 (gap) -> exit at 90
    R = 100 - (100 - 2 * p.atr[29])
    assert sim["exit_row"][0] == 32
    assert abs(sim["r"][0] + 0.001 * 100 / R - (90 - 100) / R) < 1e-6


def test_target_exit() -> None:
    closes = [100.0] * 30 + [100.0 + 2 * i for i in range(0, 20)]
    p = _panel(closes)
    sim = tradesim.simulate(p, np.array([29]), "atr2", "target", 2.0)
    R = 100 - (100 - 2 * p.atr[29])
    assert abs(sim["r"][0] + 0.001 * 100 / R - 2.0) < 1e-6  # exactly +2R before costs


def test_portfolio_grows_with_winning_trades() -> None:
    closes = [100.0] * 30 + [100.0 + i for i in range(1, 60)]
    p = _panel(closes)
    trades = pl.DataFrame(
        {
            "pattern": ["X"],
            "entity": ["US:A"],
            "entry_row": [31],
            "exit_row": [50],
            "entry_date": [p.date[31].astype(object)],
            "r": [2.0],
            "rp": [0.05],  # position = 1% risk / 5% stop = 20% of equity (under the 25% cap)
            "priority": [1.0],
        }
    ).with_columns(pl.col("entry_date").cast(pl.Date))
    r = tradesim.portfolio(p, trades, date(2023, 1, 1), date(2023, 12, 31), risk=0.01, max_pos=5)
    assert r["trades"] == 1
    assert abs(r["final"] - 100_000 * (1 + 0.01 * 2.0)) < 1.0  # risked 1% and made 2R
