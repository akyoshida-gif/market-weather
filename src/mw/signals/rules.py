"""Rule-based buy/sell point signals on daily OHLCV (ADR-0008).

Signals are reference markers, logged and scored like forecasts. Computed per entity on bars
sorted by date; only data up to and including the signal day is used (PIT: a signal for day t
is known after t's close).

  pullback_buy   押し目買い: 50-day MA rising, price above it, the day's low touched the 20-day MA
                 (within 0.5%) after closing above it on >=3 of the prior 5 days, and the day
                 closed up and above the 20-day MA.
  breakout_buy   ブレイクアウト買い: close above the prior 60-day high, volume >= 1.5x its
                 20-day average, close in the upper 30% of the day's range.
  take_profit    利益確定: RSI(14) >= 75, the day made a new 60-day high but closed below its open.
  breakdown_sell 損切り・トレンド終了: close crosses below the 50-day MA on volume >= 1.3x, or
                 closes below the prior 60-day low.
  volume_spike   急騰（出来高急増）: volume >= 5x its 20-day average and the close is +20% or more.
"""

from __future__ import annotations

import polars as pl

SIGNALS = {
    "pullback_buy": ("押し目買い", "buy"),
    "breakout_buy": ("ブレイクアウト買い", "buy"),
    "take_profit": ("利益確定", "sell"),
    "breakdown_sell": ("損切り・トレンド終了", "sell"),
    "volume_spike": ("急騰（出来高急増）", "buy"),
}
HORIZON = 10  # sessions used to score a signal


def indicators(df: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, as_of, open, high, low, close, volume (any order). Adds indicator columns."""
    g = "entity_id"
    df = df.sort([g, "as_of"])
    chg = pl.col("close").diff().over(g)
    gain = pl.when(chg > 0).then(chg).otherwise(0.0)
    loss = pl.when(chg < 0).then(-chg).otherwise(0.0)
    df = df.with_columns(
        pl.col("close").rolling_mean(20).over(g).alias("ma20"),
        pl.col("close").rolling_mean(50).over(g).alias("ma50"),
        pl.col("close").rolling_mean(200).over(g).alias("ma200"),
        pl.col("volume").rolling_mean(20).shift(1).over(g).alias("vol20"),
        pl.col("high").rolling_max(60).shift(1).over(g).alias("hi60_prev"),
        pl.col("low").rolling_min(60).shift(1).over(g).alias("lo60_prev"),
        pl.col("high").rolling_max(60).over(g).alias("hi60"),
        gain.rolling_mean(14).over(g).alias("_g"),
        loss.rolling_mean(14).over(g).alias("_l"),
        pl.col("close").shift(1).over(g).alias("prev_close"),
        (pl.col("high") - pl.col("low")).rolling_mean(14).over(g).alias("atr14"),
    )
    df = df.with_columns(
        (
            100
            - 100
            / (1 + pl.col("_g") / pl.when(pl.col("_l") > 0).then(pl.col("_l")).otherwise(1e-9))
        ).alias("rsi14"),
        pl.col("ma50").shift(10).over(g).alias("ma50_10"),
        pl.col("ma50").shift(1).over(g).alias("ma50_prev"),
        (pl.col("close") > pl.col("ma20")).cast(pl.Int8).alias("_above20"),
    ).drop("_g", "_l")
    return df.with_columns(
        pl.col("_above20").shift(1).rolling_sum(5).over(g).alias("_above20_5"),
    )


def signals(df: pl.DataFrame) -> pl.DataFrame:
    """Return indicator frame with boolean signal columns."""
    x = indicators(df)
    rng = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
    vr = pl.col("volume") / pl.col("vol20")
    return x.with_columns(
        (
            (pl.col("ma50") > pl.col("ma50_10"))
            & (pl.col("close") > pl.col("ma50"))
            & (pl.col("low") <= pl.col("ma20") * 1.005)
            & (pl.col("close") > pl.col("ma20"))
            & (pl.col("close") > pl.col("open"))
            & (pl.col("close") > pl.col("prev_close"))
            & (pl.col("_above20_5") >= 3)
        )
        .fill_null(False)
        .alias("pullback_buy"),
        (
            (pl.col("close") > pl.col("hi60_prev"))
            & (vr >= 1.5)
            & ((pl.col("close") - pl.col("low")) / rng >= 0.7)
        )
        .fill_null(False)
        .alias("breakout_buy"),
        (
            (pl.col("rsi14") >= 75)
            & (pl.col("high") >= pl.col("hi60"))
            & (pl.col("close") < pl.col("open"))
        )
        .fill_null(False)
        .alias("take_profit"),
        (
            (
                (pl.col("close") < pl.col("ma50"))
                & (pl.col("prev_close") >= pl.col("ma50_prev"))
                & (vr >= 1.3)
            )
            | (pl.col("close") < pl.col("lo60_prev"))
        )
        .fill_null(False)
        .alias("breakdown_sell"),
        ((vr >= 5) & (pl.col("close") / pl.col("prev_close") - 1 >= 0.20))
        .fill_null(False)
        .alias("volume_spike"),
    )


def events(sig: pl.DataFrame) -> pl.DataFrame:
    """Long table: one row per (entity, date, signal) with forward outcome columns."""
    g = "entity_id"
    sig = sig.sort([g, "as_of"]).with_columns(
        (pl.col("close").shift(-HORIZON).over(g) / pl.col("close") - 1).alias("fwd_ret"),
        pl.col("close").shift(-HORIZON).over(g).is_not_null().alias("matured"),
    )
    parts = [
        sig.filter(pl.col(k)).select(
            g, "as_of", "close", "volume", "vol20", "atr14", "fwd_ret", "matured",
            pl.lit(k).alias("signal"),
        )
        for k in SIGNALS
    ]  # fmt: skip
    return pl.concat(parts, how="vertical_relaxed").sort(["as_of", g])


def score(ev: pl.DataFrame, bench_fwd: pl.DataFrame | None = None) -> pl.DataFrame:
    """Win = buy signal and price higher after HORIZON sessions (sell: lower).
    bench_fwd (as_of, bench_fwd_ret) adds the excess return vs the market."""
    ev = ev.filter(pl.col("matured"))
    if bench_fwd is not None:
        ev = ev.join(bench_fwd, on="as_of", how="left").with_columns(
            (pl.col("fwd_ret") - pl.col("bench_fwd_ret")).alias("fwd_excess")
        )
    side = pl.col("signal").replace_strict({k: v[1] for k, v in SIGNALS.items()})
    return ev.with_columns(
        pl.when(side == "buy")
        .then(pl.col("fwd_ret") > 0)
        .otherwise(pl.col("fwd_ret") < 0)
        .alias("win")
    )
