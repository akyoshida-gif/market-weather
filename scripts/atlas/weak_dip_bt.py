"""Backtest the weak_dip variants as the demo trades them (top 10 per signal day), 2011 - 2024-06 signals.
Usage: python scripts/atlas/weak_dip_bt.py <px_us.parquet> <vol_us.parquet>"""

import sys
from datetime import date

import polars as pl

from mw.strategy import engine, library, regime

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
px = pl.read_parquet(
    sys.argv[1], columns=["entity_id", "as_of", "open", "high", "low", "close", "volume"]
)
px = px.filter(C("as_of") <= date(2024, 12, 31))
vol = pl.read_parquet(sys.argv[2])
ind = engine.indicators(px.with_columns(pl.lit(0.0).alias("splits")), "US:SPY")
reg = regime.us_regime(px.select("entity_id", "as_of", "close"), vol, pl.DataFrame())
ind = ind.join(reg.select("as_of", "weak_score", "spy_trend_pre"), on="as_of", how="left")
spy = ind.filter(C("entity_id") == "US:SPY").select("as_of", "open", "close").sort("as_of")
sd, so, sc = spy["as_of"].to_list(), spy["open"].to_numpy(), spy["close"].to_numpy()
pos = {d: i for i, d in enumerate(sd)}
for s in library.weak_grid("US", "US:SPY"):
    tr = engine.simulate(ind, s, "US").filter(C("signal_date") <= date(2024, 6, 30))
    rk = ind.select("entity_id", C("as_of").alias("signal_date"), s.rank().alias("rk"))
    tr = tr.join(rk, on=["entity_id", "signal_date"], how="left").sort(
        "signal_date", "rk", descending=[False, True]
    )
    tr = tr.group_by("signal_date", maintain_order=True).head(10)
    ep = (
        tr.group_by("signal_date")
        .agg(C("ret").mean(), C("exit_date").max().alias("end"), pl.len().alias("n"))
        .sort("signal_date")
    )
    ep = ep.with_columns(
        pl.Series(
            "spy",
            [
                sc[pos[e]] / so[pos[d] + 1] - 1
                for d, e in zip(ep["signal_date"], ep["end"], strict=True)
            ],
        )
    )
    print(f"\n### {s.key}：{ep.height} 回")
    for r in ep.iter_rows(named=True):
        print(
            f"  {r['signal_date']} {r['n']:2d}銘柄 {r['ret'] * 100:+6.1f}%（SPY {r['spy'] * 100:+6.1f}%）"
        )
    print(
        f"  平均 {ep['ret'].mean() * 100:+.1f}% 中央 {ep['ret'].median() * 100:+.1f}% 勝ち {(ep['ret'] > 0).sum()}/{ep.height} 最悪 {ep['ret'].min() * 100:+.1f}% / SPY 平均 {ep['spy'].mean() * 100:+.1f}%"
    )
last = reg.sort("as_of").tail(1).row(0, named=True)
