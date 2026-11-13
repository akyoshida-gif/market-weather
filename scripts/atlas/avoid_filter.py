"""Does skipping buys right after a short-term spike improve the existing long strategies?
Every long strategy (library.grid + panic_grid) is simulated once; each trade is tagged by whether its signal day
was a spike (thresholds = top 1% of liquid common stocks 2011-2019 in the atlas study). Spike trades vs the rest,
per period. Usage: python scripts/atlas/avoid_filter.py <px_us.parquet> <vol_us.parquet>"""

import sys
import time
from datetime import date

import polars as pl

from mw.strategy import engine, library, regime

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
px = pl.read_parquet(
    sys.argv[1], columns=["entity_id", "as_of", "open", "high", "low", "close", "volume"]
)
# signals up to 2024-06 only (holdout untouched); prices to 2024-12 so their exits can happen
px = px.filter(C("as_of") <= date(2024, 12, 31))
vol = pl.read_parquet(sys.argv[2])
ind = engine.indicators(px.with_columns(pl.lit(0.0).alias("splits")), "US:SPY")
reg = regime.us_regime(px.select("entity_id", "as_of", "close"), vol, pl.DataFrame())
ind = ind.join(
    reg.select(
        "as_of",
        "vix",
        "vix_term",
        "vix_prev",
        "vix_term_prev",
        "vix_max30",
        "vix_term_max30",
        "spy_trend_pre",
    ),
    on="as_of",
    how="left",
)
G = "entity_id"
ind = ind.with_columns(
    (C("close") / C("close").shift(2) - 1).over(G).alias("_r2"),
    (C("close") / C("close").shift(5) - 1).over(G).alias("_r5"),
    (C("close") / C("open") - 1).rolling_sum(5).over(G).alias("_id5"),
)
ind = ind.with_columns(
    ((C("_r2") > 0.0855) | (C("_r5") > 0.139) | (C("_id5") > 0.112)).fill_null(False).alias("spike")
)
flags = ind.select(G, C("as_of").alias("signal_date"), "spike")
print(f"指標 {time.time() - t0:.0f}秒", flush=True)

strategies = [
    s for s in library.grid("US", "US:SPY") if s.side == "long" and s.key != "index_trend"
]
strategies += library.panic_grid("US", "US:SPY")
rows = []
for s in strategies:
    tr = engine.simulate(ind, s, "US")
    if tr.height == 0:
        continue
    tr = (
        tr.filter(C("reason") != "open")
        .join(flags, on=[G, "signal_date"], how="left")
        .with_columns(
            pl.when(C("signal_date") <= date(2019, 12, 31))
            .then(pl.lit("探す"))
            .when(C("signal_date") <= date(2024, 6, 30))
            .then(pl.lit("確認"))
            .otherwise(None)
            .alias("period")
        )
        .filter(C("period").is_not_null())
    )
    for (p, sp), g in tr.group_by("period", "spike"):
        rows.append(
            {
                "key": s.key,
                "period": p,
                "spike": bool(sp),
                "n": g.height,
                "exp_R": float(g["R"].fill_null(0).mean()),
                "win": float((g["ret"] > 0).mean()),
                "avg": float(g["ret"].mean()),
            }
        )
    print(f"  {s.key} {time.time() - t0:.0f}秒", flush=True)
r = pl.DataFrame(rows)
r.write_parquet(sys.argv[1].replace(".parquet", "_avoid.parquet"))
print(
    "\n=== 急騰直後の買い（除く対象）と、それ以外の買いの成績。R＝1回の売買の損益を、損切りまでの幅で割った値"
)
for k in r["key"].unique(maintain_order=True).to_list():
    x = {(z["period"], z["spike"]): z for z in r.filter(C("key") == k).iter_rows(named=True)}
    parts = []
    for p in ("探す", "確認"):
        a, b = x.get((p, False)), x.get((p, True))
        if a is None:
            continue
        sb = f"{b['exp_R']:+.2f}R（{b['n']}回）" if b else "—"
        parts.append(f"{p}：急騰直後 {sb} / それ以外 {a['exp_R']:+.2f}R（{a['n']}回）")
    print(f"  {k:<40} " + "  ".join(parts))
agg = (
    r.group_by("period", "spike")
    .agg((C("exp_R") * C("n")).sum() / C("n").sum(), C("n").sum())
    .sort("period", "spike")
)
print("\n全部の型をまとめると：")
print(agg)
both = 0
tot = 0
for k in r["key"].unique().to_list():
    ok = True
    have = False
    for p in ("探す", "確認"):
        x = r.filter((C("key") == k) & (C("period") == p))
        a = x.filter(~C("spike"))
        b = x.filter(C("spike"))
        if a.height and b.height and b["n"][0] >= 10:
            have = True
            ok &= b["exp_R"][0] < a["exp_R"][0]
        else:
            ok = False
    if have:
        tot += 1
        both += ok
print(f"急騰直後が両方の期間で劣っていた型：{both}/{tot}（急騰直後の売買が各期間10回以上ある型）")
print(f"合計 {time.time() - t0:.0f}秒")
