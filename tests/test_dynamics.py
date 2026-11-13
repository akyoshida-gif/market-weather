from datetime import date, timedelta

import numpy as np
import polars as pl

from mw.research import dynamics

C = pl.col


def _frame(closes: list[float], eid: str = "US:X") -> pl.DataFrame:
    n = len(closes)
    d0 = date(2020, 1, 1)
    c = np.array(closes, dtype=float)
    return pl.DataFrame(
        {
            "entity_id": [eid] * n,
            "as_of": [d0 + timedelta(days=i) for i in range(n)],
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": np.full(n, 1e6),
            "splits": np.zeros(n),
        }
    )


def _spy(n: int) -> pl.DataFrame:
    d0 = date(2020, 1, 1)
    spy = pl.DataFrame(
        {"as_of": [d0 + timedelta(days=i) for i in range(n)], "close": np.linspace(100, 120, n)}
    )
    return dynamics.market_runs(spy)


def test_runs_and_crosses_of_the_20_day_average():
    # 70 days up, then 10 days well below the 20-day average (crosses need 60 sessions of history)
    closes = [100 + i for i in range(70)] + [100.0] * 10
    out = dynamics.stock_dynamics(_frame(closes), _spy(80)).sort("as_of")
    run = out["t_run20"].to_list()
    assert run[69] > 0  # still above while rising
    assert run[-1] == -10  # ten sessions below
    assert out["t_prun20"][-1] == run[69]  # the previous state was the rising run
    assert out["t_x20_60"][-1] == 1
    # deepest point of the current state is on its first day; it came back since (prices flat, average falls)
    assert out["t_back20"][-1] > 0


def test_lower_low_with_higher_rsi_is_a_bullish_divergence():
    # a sharp fall to 70, a rebound, then a slow drift to a slightly lower low (69)
    closes = [100.0] * 30 + list(np.linspace(100, 70, 5)) + list(np.linspace(70, 85, 15))
    closes += list(np.linspace(85, 69, 15)) + [69.5] * 2
    out = dynamics.stock_dynamics(_frame(closes), _spy(len(closes))).sort("as_of")
    last = out.row(-1, named=True)
    assert last["t_ll"] < 0  # lower low
    assert last["t_div_lo_rsi"] > 0  # RSI was higher at the second low
    assert last["t_bull_rsi"] == last["t_div_lo_rsi"]
    assert last["t_bear_rsi"] is None  # no higher high


def test_relative_strength_and_rank():
    closes = [100 * 1.01**i for i in range(80)]
    out = dynamics.stock_dynamics(_frame(closes), _spy(80)).sort("as_of")
    assert out["t_rs_ch20"][-1] > 0  # beats the slowly rising market
    px = pl.concat([_frame(closes, "US:A"), _frame([100.0] * 80, "US:B")]).lazy()
    rk = dynamics.rank_dynamics(px).sort("as_of")
    a = rk.filter(C("entity_id") == "US:A")["t_rk60"][-1]
    b = rk.filter(C("entity_id") == "US:B")["t_rk60"][-1]
    assert a > b
