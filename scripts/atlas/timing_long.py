"""Weakness score on the long history (S&P 500 index, VIX, SKEW since 1990; Cboe closes).
Usage: python scripts/atlas/timing_long.py <vol_us.parquet>"""

import sys
from datetime import date

import numpy as np
import polars as pl

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
v = pl.read_parquet(sys.argv[1]).filter(
    C("entity_id").is_in(["IDX:US:SPX", "IDX:US:VIX", "IDX:US:SKEW"])
)
w = (
    v.pivot(on="entity_id", index="as_of", values="close")
    .sort("as_of")
    .rename({"IDX:US:SPX": "spx", "IDX:US:VIX": "vix", "IDX:US:SKEW": "skew"})
)
END = date(2024, 6, 30)
# cut at the end of the validation period BEFORE computing forward returns: no price after 2024-06 is used
w = (
    w.filter((C("as_of") >= date(1990, 1, 2)) & (C("as_of") <= END))
    .drop_nulls(["spx"])
    .with_columns(
        (C("spx") / C("spx").rolling_mean(50) - 1).alias("dma50"),
        (C("spx") / C("spx").shift(20) - 1).alias("ret20"),
        ((C("vix") - C("vix").rolling_mean(250)) / C("vix").rolling_std(250)).alias("vixz"),
        *[(C("spx").shift(-h) / C("spx") - 1).alias(f"f{h}") for h in (20, 60)],
    )
)
disc = C("as_of") <= date(2009, 12, 31)  # edges from 1990-2009 only
conds = [("dma50", "low"), ("ret20", "low"), ("skew", "low"), ("vixz", "high")]
d = w.filter(disc)
score = pl.lit(0)
for f, side in conds:
    q = d[f].drop_nulls().quantile(0.2 if side == "low" else 0.8)
    score = score + (((C(f) <= q) if side == "low" else (C(f) >= q)).fill_null(False).cast(pl.Int8))
w = w.with_columns(score.alias("score"), (C("as_of").dt.year() // 10 * 10).alias("decade"))
# episodes: a run of days with score >= 3, separated by at least 20 sessions with score < 3
w = w.with_columns((C("score") >= 3).alias("hot"))
hot = w["hot"].to_numpy()
ep = np.zeros(len(hot), dtype=int)
k, last = 0, -999
for i, h in enumerate(hot):
    if h:
        if i - last > 20:
            k += 1
        ep[i] = k
        last = i
w = w.with_columns(pl.Series("episode", ep))
print("点数（4つの弱った条件の重なり、0〜4）ごとの、その後の S&P 500（指数、配当なし）")
for h in (20, 60):
    t = (
        w.drop_nulls(f"f{h}")
        .group_by("decade", "score")
        .agg(C(f"f{h}").mean().alias("m"), pl.len().alias("n"))
        .sort("decade", "score")
    )
    print(f"\n  {h}日後：年代 × 点数")
    for dec in sorted(t["decade"].unique().to_list()):
        x = t.filter(C("decade") == dec)
        print(
            f"    {dec}年代 "
            + "  ".join(
                f"点{r['score']}:{r['m'] * 100:+.1f}%({r['n']})" for r in x.iter_rows(named=True)
            )
        )
    a = (
        w.drop_nulls(f"f{h}")
        .group_by("score")
        .agg(C(f"f{h}").mean().alias("m"), (C(f"f{h}") > 0).mean().alias("p"), pl.len().alias("n"))
        .sort("score")
    )
    print(
        "    全期間 "
        + "  ".join(
            f"点{r['score']}:{r['m'] * 100:+.1f}%（上がった{r['p'] * 100:.0f}%）"
            for r in a.iter_rows(named=True)
        )
    )
# per episode (score >= 3): buy at the first day, hold 60 sessions
e = (
    w.filter(C("episode") > 0)
    .group_by("episode")
    .agg(
        C("as_of").first().alias("start"),
        C("f60").first().alias("f60"),
        C("f20").first().alias("f20"),
    )
    .sort("start")
)
e = e.drop_nulls("f60")
print(f"\n点数3以上の局面：{e.height} 回（局面の最初の日に買って60日後）")
print(
    f"  平均 {e['f60'].mean() * 100:+.1f}% / 中央 {e['f60'].median() * 100:+.1f}% / 上がった {(e['f60'] > 0).mean() * 100:.0f}% / 最悪 {e['f60'].min() * 100:+.1f}%"
)
for dec in (1990, 2000, 2010, 2020):
    x = e.filter(C("start").dt.year() // 10 * 10 == dec)
    if x.height:
        print(
            f"  {dec}年代：{x.height}回 平均 {x['f60'].mean() * 100:+.1f}% 上がった {(x['f60'] > 0).mean() * 100:.0f}%"
        )
base = w.drop_nulls("f60")["f60"]
print(f"  比較：全日の平均 {base.mean() * 100:+.1f}%（上がった {(base > 0).mean() * 100:.0f}%）")

# split by the trend BEFORE the weakness: 20 sessions earlier the index was above a rising 200-day line
w = w.with_columns(
    (
        (C("spx") > C("spx").rolling_mean(200))
        & (C("spx").rolling_mean(200) > C("spx").rolling_mean(200).shift(20))
    )
    .shift(20)
    .fill_null(False)
    .alias("up_before")
)
e = (
    w.filter(C("episode") > 0)
    .group_by("episode")
    .agg(
        C("as_of").first().alias("start"),
        C("f60").first().alias("f60"),
        C("up_before").first().alias("up"),
    )
    .drop_nulls("f60")
    .sort("start")
)
print("\n点数3以上の局面を、弱る前の相場の流れで分ける（局面の最初の日に買って60日後）")
for flag, name in ((True, "上昇相場の中で弱った"), (False, "下落相場の中で弱った")):
    x = e.filter(C("up") == flag)
    print(
        f"  {name}：{x.height}回 平均 {x['f60'].mean() * 100:+.1f}% 中央 {x['f60'].median() * 100:+.1f}% 上がった {(x['f60'] > 0).mean() * 100:.0f}% 最悪 {x['f60'].min() * 100:+.1f}%"
    )
    for dec in (1990, 2000, 2010, 2020):
        y = x.filter(C("start").dt.year() // 10 * 10 == dec)
        if y.height:
            print(
                f"    {dec}年代：{y.height}回 平均 {y['f60'].mean() * 100:+.1f}% 上がった {(y['f60'] > 0).mean() * 100:.0f}%"
            )
