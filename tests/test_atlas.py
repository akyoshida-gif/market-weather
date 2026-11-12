from datetime import date, timedelta

import numpy as np
import polars as pl

from mw.research import atlas


def _px(close: list[float], eid: str = "US:AAA", vol: float = 1e6) -> pl.DataFrame:
    d0 = date(2020, 1, 1)
    c = np.array(close, dtype=float)
    return pl.DataFrame({
        "entity_id": [eid] * len(c), "as_of": [d0 + timedelta(days=i) for i in range(len(c))],
        "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": [vol] * len(c),
    })  # fmt: skip


def test_features_do_not_look_ahead() -> None:
    rng = np.random.default_rng(1)
    c = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    a = atlas.past_relative(atlas.stock_features(_px(c)))
    c2 = c[:300] + [x * 1.5 for x in c[300:]]  # change only the future
    b = atlas.past_relative(atlas.stock_features(_px(c2)))
    cols = [x for x in a.columns if x.startswith("s_")]
    left, right = a.slice(0, 300).select(cols), b.slice(0, 300).select(cols)
    for col in cols:
        x, y = left[col].to_numpy().astype(float), right[col].to_numpy().astype(float)
        assert np.allclose(x, y, equal_nan=True), col


def test_streak_counts_runs() -> None:
    c = [10, 11, 12, 13, 12, 11, 12.0]
    f = atlas.stock_features(_px(c))
    assert f["s_streak"].to_list()[1:] == [1, 2, 3, -1, -2, 1]
    assert abs(f["s_streak_ret"][3] - 0.3) < 1e-9  # 10 -> 13 over the run


def test_stop_then_target_is_recorded_in_order() -> None:
    # flat at 100 (ATR = 2 from the +-1% candles), then a drop to 95 and a rally to 106
    c = [100.0] * 30 + [97.0, 95.0, 99.0, 103.0, 106.0] + [106.0] * 70
    df = atlas.stock_features(_px(c))
    y = atlas.path_labels(df)
    r = y.row(29, named=True)  # signal day: last flat day, entry next open = 97
    atr = r["atr"]
    assert abs(atr - 2.0) < 0.2
    # low 95*0.99 = 94.05 is ~1.5 ATR under 97 on session 2; high 106*1.01 is ~5 ATR above on session 5
    assert r["y_dn1"] == 2 and r["y_up4"] == 5
    assert r["y_dn1"] < r["y_up4"]  # stop first, then target
    assert r["y_dn3"] == -1  # 3 ATR under the entry was never reached
    assert abs(r["y_ret5"] - (106 / 97 - 1)) < 1e-6


def test_labels_unknown_at_the_end_of_data() -> None:
    c = [100.0 + i * 0.1 for i in range(100)]
    y = atlas.path_labels(atlas.stock_features(_px(c)))
    last = y.row(95, named=True)
    assert last["y_ret60"] is None or np.isnan(last["y_ret60"])
    # not reached, but the window is incomplete: unknown, not "never"
    assert last["y_up6"] is None or np.isnan(last["y_up6"])
