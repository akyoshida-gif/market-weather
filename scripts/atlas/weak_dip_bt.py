"""Backtest 'buy the dip of an uptrend' as the demo would trade it (research only; not in paper.yaml).

Weakness score = how many of four conditions hold: SPY under its 50-day line by 1.95%+, SPY 20-day return
-2.41% or worse, Cboe SKEW 111.8 or lower, VIX z-score over its last 250 days 0.916 or higher (edges: the weakest
20% of days of the S&P 500 index 1990-2009). Buy on the first day of an episode (score >= 3, none in the previous
20 sessions; judged once for the whole market) when 20 sessions earlier SPY was above a rising 200-day line.
Only trades whose exit is on or before 2024-06-30 count (the holdout starts 2024-07).
Usage: python scripts/atlas/weak_dip_bt.py <px_us.parquet> <vol_us.parquet>"""

import sys
from datetime import date

import numpy as np
import polars as pl

from mw.strategy import engine
from mw.strategy.engine import Strategy

C = pl.col
INF = float("inf")
END = date(2024, 6, 30)
sys.stdout.reconfigure(encoding="utf-8")
px = pl.read_parquet(
    sys.argv[1], columns=["entity_id", "as_of", "open", "high", "low", "close", "volume"]
)
px = px.filter(C("as_of") <= END)
vol = pl.read_parquet(sys.argv[2])
spy = (
    px.filter(C("entity_id") == "US:SPY")
    .sort("as_of")
    .select(
        "as_of",
        (C("close") / C("close").rolling_mean(50) - 1).alias("dma50"),
        (C("close") / C("close").shift(20) - 1).alias("ret20"),
        (
            (C("close") > C("close").rolling_mean(200))
            & (C("close").rolling_mean(200) > C("close").rolling_mean(200).shift(20))
        )
        .shift(20)
        .fill_null(False)
        .alias("up_before"),
    )
)
vix = (
    vol.filter(C("entity_id") == "IDX:US:VIX")
    .sort("as_of")
    .drop_nulls("close")
    .select(
        "as_of",
        ((C("close") - C("close").rolling_mean(250)) / C("close").rolling_std(250)).alias("vixz"),
    )
)
skew = (
    vol.filter(C("entity_id") == "IDX:US:SKEW")
    .sort("as_of")
    .drop_nulls("close")
    .select("as_of", C("close").alias("skew"))
)
m = spy.join(vix, on="as_of", how="left").join(skew, on="as_of", how="left").sort("as_of")
m = m.with_columns(
    (
        (C("dma50") <= -0.0195).fill_null(False).cast(pl.Int8)
        + (C("ret20") <= -0.0241).fill_null(False).cast(pl.Int8)
        + (C("skew") <= 111.8).fill_null(False).cast(pl.Int8)
        + (C("vixz") >= 0.916).fill_null(False).cast(pl.Int8)
    ).alias("score")
)
hot = (m["score"] >= 3).to_numpy()
first = np.zeros(len(hot), dtype=bool)
last = -999
for i, h in enumerate(hot):
    if h:
        first[i] = i - last > 20
        last = i
m = m.with_columns(pl.Series("weak_first", first))
low_skew = m.filter(C("as_of") >= date(2011, 1, 1))["skew"].drop_nulls()
print(f"2011年以降で SKEW が 111.8 以下だった日：{(low_skew <= 111.8).mean() * 100:.1f}%")

ind = engine.indicators(px.with_columns(pl.lit(0.0).alias("splits")), "US:SPY")
ind = ind.join(m.select("as_of", "weak_first", "up_before"), on="as_of", how="left")


def strat(pick: str, k: float) -> Strategy:
    def cond() -> pl.Expr:
        if pick == "index":
            return C("entity_id") == "US:SPY"
        return (
            (C("beta_pct") >= 0.8)
            & (C("pre_pct") >= 0.8)
            & (C("dvol20") >= 10e6)
            & (C("close") >= 10)
        )

    def rank() -> pl.Expr:
        return pl.lit(1.0) if pick == "index" else C("beta_pct") + C("pre_pct")

    return Strategy(
        key=f"weak_dip|{pick}|k{k:g}", label="", horizon="中期", side="long",
        entry=lambda: C("weak_first").fill_null(False) & C("up_before").fill_null(False) & cond(),
        stop=(lambda: C("close") - k * C("atr")) if k > 0 else (lambda: C("close") * 0.6),
        target=lambda: pl.lit(INF), max_hold=60, rank=rank,
    )  # fmt: skip


s_all = ind.filter(C("entity_id") == "US:SPY").sort("as_of")
c = s_all["close"].to_numpy()
f60 = np.full(len(c), np.nan)
f60[:-60] = c[60:] / c[:-60] - 1
sel = (s_all["as_of"] >= date(2011, 1, 1)).to_numpy()
print(
    f"比較：2011年〜（60日後が2024年6月まで）の全日の SPY 60日後の平均 {np.nanmean(f60[sel]) * 100:+.1f}%（上がった {np.nanmean(f60[sel] > 0) * 100:.0f}%）"
)
for s in (strat("index", 0.0), strat("bs", 3.0)):
    tr = engine.simulate(ind, s, "US").filter((C("exit_date") <= END) & (C("reason") != "open"))
    rk = ind.select("entity_id", C("as_of").alias("signal_date"), s.rank().alias("rk"))
    tr = tr.join(rk, on=["entity_id", "signal_date"], how="left").sort(
        "signal_date", "rk", descending=[False, True]
    )
    tr = tr.group_by("signal_date", maintain_order=True).head(10)
    ep = tr.group_by("signal_date").agg(C("ret").mean(), pl.len().alias("n")).sort("signal_date")
    print(f"\n### {s.key}：{ep.height} 回")
    for r in ep.iter_rows(named=True):
        print(f"  {r['signal_date']} {r['n']:2d}銘柄 {r['ret'] * 100:+6.1f}%")
    print(
        f"  平均 {ep['ret'].mean() * 100:+.1f}% 中央 {ep['ret'].median() * 100:+.1f}% 勝ち {(ep['ret'] > 0).sum()}/{ep.height} 最悪 {ep['ret'].min() * 100:+.1f}%"
    )
