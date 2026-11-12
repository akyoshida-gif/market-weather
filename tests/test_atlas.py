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


def test_price_level_uses_the_price_seen_at_the_time() -> None:
    # adjusted history of a stock that later did a 1-for-20 reverse split: old prices look 20x higher
    df = _px([10.0] * 5 + [10.0] * 5).with_columns(
        pl.Series("splits", [0.0] * 5 + [0.05] + [0.0] * 4)
    )
    f = atlas.stock_features(df)
    assert f["raw_close"].to_list()[:5] == [0.5] * 5  # what traders saw before the reverse split
    assert f["raw_close"].to_list()[5:] == [10.0] * 5


def test_split_missing_from_the_history_is_repaired() -> None:
    # a 2-for-1 split on day 5 that the stored history did not apply: the close drops from 100 to 50
    df = _px([100.0] * 5 + [50.0] * 5).with_columns(
        pl.Series("splits", [0.0] * 5 + [2.0] + [0.0] * 4)
    )
    f = atlas.stock_features(df)
    assert f["close"].to_list() == [50.0] * 10  # older rows adjusted by us
    assert f["raw_close"].to_list() == [100.0] * 5 + [50.0] * 5  # what traders saw
    assert abs(f["volume"][0] - 2e6) < 1e-6
    # already-adjusted history is left alone
    g = atlas.stock_features(
        _px([50.0] * 10).with_columns(pl.Series("splits", [0.0] * 5 + [2.0] + [0.0] * 4))
    )
    assert (
        g["close"].to_list() == [50.0] * 10 and g["raw_close"].to_list() == [100.0] * 5 + [50.0] * 5
    )


def test_entities_do_not_mix_and_entry_is_the_next_open() -> None:
    a = _px([10.0 + i for i in range(80)], "US:AAA")
    b = _px([200.0 - i for i in range(80)], "US:BBB").with_columns(
        (pl.col("close") + 3).alias("open")
    )
    df = pl.concat([a, b]).sample(fraction=1.0, shuffle=True, seed=3)  # rows in random order
    y = atlas.path_labels(atlas.stock_features(df)).sort("entity_id", "as_of")
    ra = y.filter(pl.col("entity_id") == "US:AAA").row(30, named=True)
    assert (
        abs(ra["y_ret1"] - 0.0) < 1e-6
    )  # open == close for AAA: entry and close are the same price
    rb = y.filter(pl.col("entity_id") == "US:BBB").row(30, named=True)
    # BBB: signal close 170, next session open = 169 + 3 = 172, close 169
    assert abs(rb["y_ret1"] - (169 / 172 - 1)) < 1e-6
    assert (
        y.filter(pl.col("entity_id") == "US:AAA")["s_ret1"].drop_nulls().min() > 0
    )  # never sees BBB's falls


def test_fred_is_used_from_the_day_after_publication() -> None:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    px = pl.DataFrame({"entity_id": ["US:SPY"] * 5, "as_of": days, **{c: [0.1] * 5 for c in (
        "s_dma50", "s_dma200", "s_dhi250", "s_dlo250", "s_ret1", "s_ret5", "s_ret20", "s_vr20", "s_streak", "s_vol20")}})  # fmt: skip
    vol = pl.DataFrame({"entity_id": [f"IDX:US:{s}" for s in ("VIX", "VIX3M", "VIX9D") for _ in days],
                        "as_of": days * 3, "close": [20.0] * 15})  # fmt: skip
    from datetime import datetime

    fred = pl.DataFrame({"entity_id": ["MACRO:FRED:DGS10"], "as_of": [date(2024, 1, 2)], "value": [4.0],
                         "published_at": [datetime(2024, 1, 3, 23, 0)]})  # fmt: skip
    m = atlas.market_features(px, vol, fred).sort("as_of")
    assert (
        m.filter(pl.col("as_of") == date(2024, 1, 3))["m_y10"][0] is None
    )  # published that evening
    assert m.filter(pl.col("as_of") == date(2024, 1, 4))["m_y10"][0] == 4.0


def test_search_purges_rows_whose_outcome_crosses_the_period_end() -> None:
    from mw.research import search

    days = [date(2019, 12, 1) + timedelta(days=i) for i in range(60)]
    df = pl.DataFrame({"as_of": days})
    p = {
        "discovery": (date(2011, 1, 1), date(2019, 12, 31)),
        "validation": (date(2020, 1, 1), date(2024, 6, 30)),
    }
    kept = search.purge(df, "y_ret10", p)["as_of"]
    assert kept.filter(kept <= date(2019, 12, 31)).max() <= date(2019, 12, 13)
    assert date(2020, 1, 5) in kept.to_list()


def test_repair_finds_the_step_before_the_split_day_and_reverse_splits() -> None:
    # 2-for-1 on day 8, but the refetch already rewrote days 4..7: the step sits on day 4
    c = [100.0] * 4 + [50.0] * 8
    s = [0.0] * 8 + [2.0] + [0.0] * 3
    f = atlas.stock_features(_px(c).with_columns(pl.Series("splits", s)))
    assert f["close"].to_list() == [50.0] * 12
    assert f["raw_close"].to_list() == [100.0] * 8 + [50.0] * 4
    # 1-for-20 reverse split not applied: 0.5 -> 10 on the split day
    g = atlas.stock_features(
        _px([0.5] * 5 + [10.0] * 5).with_columns(
            pl.Series("splits", [0.0] * 5 + [0.05] + [0.0] * 4)
        )
    )
    assert [round(x, 6) for x in g["close"].to_list()] == [10.0] * 10
    assert [round(x, 6) for x in g["raw_close"].to_list()] == [0.5] * 5 + [10.0] * 5


def test_small_ratios_are_never_repaired() -> None:
    # an adjusted spin-off (ratio 1.04) on a day the stock simply fell 3%
    c = [100.0] * 5 + [97.0] * 5
    f = atlas.stock_features(
        _px(c).with_columns(pl.Series("splits", [0.0] * 5 + [1.04] + [0.0] * 4))
    )
    assert f["close"].to_list() == c


def test_label_sessions() -> None:
    from mw.research import search

    assert search.label_sessions("y_ret10") == 10 and search.label_sessions("y_mfe60") == 60
    assert (
        search.label_sessions("y_up1_5") == atlas.FWD
        and search.label_sessions("y_dn0_5") == atlas.FWD
    )
