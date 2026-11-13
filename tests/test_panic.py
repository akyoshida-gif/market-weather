from datetime import date, timedelta

import polars as pl
import yaml

from mw.core.config import REPO_ROOT
from mw.strategy import library


def _frame(vix: list[float], term: list[float]) -> pl.DataFrame:
    n = len(vix)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    rows = []
    for eid, beta, pre in (("US:AAA", 0.95, 0.95), ("US:BBB", 0.5, 0.95), ("US:SPY", 0.6, 0.6)):
        for i, d in enumerate(days):
            rows.append({"entity_id": eid, "as_of": d, "close": 50.0, "atr": 1.0, "dvol20": 5e7, "beta_pct": beta,
                         "pre_pct": pre, "vix": vix[i], "vix_prev": vix[i - 1] if i else None, "vix_term": term[i],
                         "vix_term_prev": term[i - 1] if i else None, "spy_trend_pre": True})  # fmt: skip
    df = pl.DataFrame(rows).sort("entity_id", "as_of")
    return df.with_columns(
        pl.col("vix").rolling_max(30, min_samples=1).over("entity_id").alias("vix_max30")
    )


def test_panic_buys_only_on_the_first_crossing_and_only_the_selected_stocks() -> None:
    # VIX crosses 30 on day 5, falls back, crosses again on day 12 (same panic), then much later on day 50
    vix = [20.0] * 60
    for i in (5, 6, 12, 13, 50):
        vix[i] = 33.0
    term = [0.9] * 60
    df = _frame(vix, term)
    s = library.panic_rebound("US", "US:SPY", 30, "cross", "bs", 60, 3.0)
    sig = df.with_columns(s.entry().alias("sig")).filter(pl.col("sig"))
    assert sig["entity_id"].unique().to_list() == [
        "US:AAA"
    ]  # beta and strength both in the top 20%
    days = sorted(d.day for d in sig["as_of"].to_list())
    assert len(days) == 2 and sig["as_of"].min() == date(2024, 1, 6)  # day 5 and day 50, not day 12


def test_index_variant_buys_the_benchmark_only() -> None:
    vix = [20.0] * 20
    vix[5] = 31.0
    df = _frame(vix, [0.9] * 20)
    s = library.panic_rebound("US", "US:SPY", 30, "cross", "index", 60, 0.0)
    sig = df.with_columns(s.entry().alias("sig"), s.stop().alias("stop")).filter(pl.col("sig"))
    assert sig["entity_id"].to_list() == ["US:SPY"] and sig["stop"].to_list() == [
        30.0
    ]  # -40% disaster stop


def test_every_paper_strategy_key_can_be_rebuilt() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "config/strategies/paper.yaml").read_text(encoding="utf-8"))
    bench = {"US": "US:SPY", "JP": "JP:1306"}
    for market, items in cfg.items():
        for c in items:
            assert library.by_key(market, bench[market], c["key"]).key == c["key"]


def test_cap_drops_names_that_move_too_much() -> None:
    vix = [20.0] * 20
    vix[5] = 31.0
    df = _frame(vix, [0.9] * 20).with_columns(
        pl.when(pl.col("entity_id") == "US:AAA").then(5.0).otherwise(1.0).alias("atr"),
        pl.lit(0.5).alias("pre_str"),
    )  # AAA: ATR 10% of the price
    s = library.panic_rebound("US", "US:SPY", 30, "cross", "bs", 60, 3.0, cap="atr6")
    assert s.key.endswith("|atr6")
    assert df.filter(s.entry()).height == 0
    s2 = library.panic_rebound("US", "US:SPY", 30, "cross", "bs", 60, 3.0, cap="pre100")
    assert df.filter(s2.entry())["entity_id"].to_list() == ["US:AAA"]


def test_weak_dip_buys_the_first_weak_day_of_an_uptrend_only() -> None:
    n = 40
    score = [0] * n
    for i in (5, 6, 7, 30):
        score[i] = 3
    rows = []
    for eid in ("US:SPY", "US:AAA"):
        for i in range(n):
            rows.append({"entity_id": eid, "as_of": date(2024, 1, 1) + timedelta(days=i), "close": 50.0, "atr": 1.0,
                         "dvol20": 5e7, "beta_pct": 0.95, "pre_pct": 0.95, "weak_score": score[i],
                         "spy_trend_pre": i < 20})  # fmt: skip
    df = pl.DataFrame(rows).sort("entity_id", "as_of")
    s = library.weak_dip("US", "US:SPY", 3, "index", 60, 0.0)
    sig = df.filter(s.entry())
    # day 5 only: days 6-7 are the same episode, day 30 is no longer in an uptrend
    assert sig["entity_id"].to_list() == ["US:SPY"] and sig["as_of"].to_list() == [date(2024, 1, 6)]


def test_weak_score_counts_the_four_conditions() -> None:
    from mw.strategy import regime

    df = pl.DataFrame({
        "spy_dma50": [0.01, -0.03, -0.03, -0.03, None],
        "spy_ret20": [0.02, 0.00, -0.05, -0.05, -0.05],
        "skew": [130.0, 130.0, 130.0, 105.0, 105.0],
        "vix_z250": [0.0, 0.0, 1.5, 1.5, None],
    })  # fmt: skip
    assert df.select(regime.weak_score())["spy_dma50"].to_list() == [0, 1, 3, 4, 2]
