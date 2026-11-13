"""Dynamics layer (t_*): one number at a time, and the 'days in the state x depth' tables (owner 2026-11-13).

Outcomes (signal day close -> entry next open):
  ex20  20-session return minus the same-day average of all stocks
  big20 the low within 20 sessions was 10% or more below the entry (rate minus the same-day rate)
Quintile edges from the discovery period only; counted by month (t-value); holdout never loaded.
Usage: python scripts/atlas/dyn_search.py <atlas_dir> <every>"""

import sys
import time

import numpy as np
import polars as pl

from mw.research import search

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A, every = sys.argv[1], int(sys.argv[2])
t0 = time.time()
P = search.PERIODS
lo, hi = P["discovery"][0], P["validation"][1]
days = (
    pl.scan_parquet(f"{A}/market.parquet").select("as_of").filter((C("as_of") >= lo) & (C("as_of") <= hi))
    .collect().sort("as_of").with_row_index("_d").filter(C("_d") % every == 0).select("as_of")
)  # fmt: skip
st = (
    pl.scan_parquet(f"{A}/stocks/*.parquet")
    .filter((C("kind") == "common") & (C("dvol20") >= 5e6) & (C("raw_close") >= 5))
    .join(days.lazy(), on="as_of", how="semi")
    .select("entity_id", "as_of", "s_atrp", "s_atr20_120", "y_ret20", "y_mae20")
    .filter(C("y_ret20").is_finite() & C("y_mae20").is_finite())
)
dyn = pl.scan_parquet(f"{A}/dyn/*.parquet").join(days.lazy(), on="as_of", how="semi")
df = st.join(dyn, on=["entity_id", "as_of"], how="left").collect()
df = df.with_columns(
    (C("y_mae20") * C("s_atrp") <= -0.10).cast(pl.Float64).alias("_big"),
    # volatility group of the day (quintile of ATR / price): big losses are compared only within the same group,
    # otherwise "volatile stocks fall more" dominates everything
    (C("s_atrp").rank() / pl.len() * 5).ceil().clip(1, 5).over("as_of").alias("_vg"),
    # is the current volatility above the stock's usual (ATR 20 vs 120)? volatility tends to fall back, so an
    # unusually wild stock looks "safe" against its current ATR
    (C("s_atr20_120").rank() / pl.len() * 3).ceil().clip(1, 3).over("as_of").alias("_vg2"),
)
df = df.with_columns(
    (C("y_ret20") - C("y_ret20").mean().over("as_of")).alias("ex20"),
    (C("_big") - C("_big").mean().over("as_of", "_vg", "_vg2")).alias("big20"),
)
df = search.purge(df, "y_ret20", P)
feats = [c for c in df.columns if c.startswith("t_")]
print(
    f"{df.height:,} 行・{len(feats)} 個 {time.time() - t0:.0f}秒（大負けの率 全体 {df['_big'].mean() * 100:.1f}%）",
    flush=True,
)

asof = df["as_of"]
disc = ((asof >= P["discovery"][0]) & (asof <= P["discovery"][1])).to_numpy()
val = ((asof >= P["validation"][0]) & (asof <= P["validation"][1])).to_numpy()
ym = (asof.dt.year().cast(pl.Int64) * 12 + asof.dt.month().cast(pl.Int64)).to_numpy()
months, mi = np.unique(ym, return_inverse=True)
yom = months // 12
Y = {k: df[k].to_numpy() for k in ("ex20", "big20")}


def st_(mask: np.ndarray, y: np.ndarray) -> dict:
    s = np.bincount(mi[mask], weights=y[mask], minlength=len(months))
    c = np.bincount(mi[mask], minlength=len(months))
    return search._monthly(s, c, yom)


rows = []
for f in feats:
    x = df[f].cast(pl.Float64).to_numpy()
    fin = np.isfinite(x)
    xd = x[disc & fin]
    if len(xd) < 2000:
        continue
    e = np.unique(np.quantile(xd, [0.2, 0.4, 0.6, 0.8]))
    q = np.full(len(x), -1)
    q[fin] = np.searchsorted(e, x[fin], side="right")
    for k in range(len(e) + 1):
        m = q == k
        r = {
            "f": f,
            "q": k,
            "lo": -np.inf if k == 0 else e[k - 1],
            "hi": np.inf if k == len(e) else e[k],
        }
        for yk, y in Y.items():
            a, b = st_(m & disc, y), st_(m & val, y)
            r |= {
                f"{yk}_d": a["mean"],
                f"{yk}_dt": a["t"],
                f"{yk}_v": b["mean"],
                f"{yk}_vt": b["t"],
                "n": a["n"],
            }
        rows.append(r)
R = pl.DataFrame(rows)
R.write_parquet(f"{A}/dyn_single.parquet")


def show(col: str, name: str, good_high: bool) -> None:
    d, dt, v = f"{col}_d", f"{col}_dt", f"{col}_v"
    big = R.filter(C(dt).is_finite() & C(f"{col}_vt").is_finite() & (C(dt).abs() >= 3))
    same = big.filter(C(d) * C(v) > 0)
    print(
        f"\n=== {name}：探す期間で t値3以上の段階 {big.height} 個のうち、確かめる期間でも同じ向き {same.height} 個"
    )
    for title, desc in (("良い側", good_high), ("悪い側", not good_high)):
        x = same.filter((C(d) > 0) if desc else (C(d) < 0)).sort(dt, descending=desc).head(20)
        print(f"  --- {title}")
        for r in x.iter_rows(named=True):
            print(
                f"  {r['f']:<16} {r['lo']:>9.3g}〜{r['hi']:<9.3g} 探す {r[d] * 100:+.2f}%（t{r[dt]:5.1f}） / 確認 {r[v] * 100:+.2f}%（t{r[f'{col}_vt']:5.1f}）"
            )


show("ex20", "20日後の市場平均との差", True)
show(
    "big20",
    "20日以内に −10% 以上下げる率（同じ日・同じ値動きの激しさの銘柄との差、低いほど良い）",
    False,
)

# ---- owner's question: how long in the state x how deep
print(
    "\n=== 200日線の上下にいる日数 × 200日線からの距離（20日後の市場平均との差 / 大負けの率の差）"
)
runb = [
    (-9999, -121),
    (-120, -61),
    (-60, -21),
    (-20, -6),
    (-5, -2),
    (-1, -1),
    (1, 1),
    (2, 5),
    (6, 20),
    (21, 60),
    (61, 120),
    (121, 9999),
]
depth = [
    (-9, -0.20),
    (-0.20, -0.10),
    (-0.10, -0.05),
    (-0.05, 0),
    (0, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 9),
]
run = df["t_run200"].cast(pl.Float64).to_numpy()
dd = df["t_d200"].cast(pl.Float64).to_numpy()
for per, pm in (("探す", disc), ("確認", val)):
    print(f"\n  [{per}] 行＝日数（−は下にいる日数）、列＝距離")
    print(
        "  "
        + " " * 12
        + "".join(
            f"{f'{a:+.0%}〜{b:+.0%}' if abs(a) < 9 and abs(b) < 9 else ('〜' + f'{b:+.0%}' if a < -8 else f'{a:+.0%}〜'):>16}"
            for a, b in depth
        )
    )
    for a, b in runb:
        cells = []
        for c0, c1 in depth:
            m = pm & (run >= a) & (run <= b) & (dd > c0) & (dd <= c1)
            if m.sum() < 300:
                cells.append(f"{'':>16}")
                continue
            r1, r2 = st_(m, Y["ex20"]), st_(m, Y["big20"])
            cells.append(
                f"{r1['mean'] * 100:+5.1f}%/{r2['mean'] * 100:+5.1f}pt{'*' if abs(r1['t']) >= 3 else ' ':1}"
            )
        lab = f"{a}〜{b}" if abs(a) < 9999 and abs(b) < 9999 else (f"〜{b}" if a < 0 else f"{a}〜")
        print(f"  {lab:>12}" + "".join(f"{c:>16}" for c in cells))
print(
    "  （各升：20日後の市場平均との差 / 大負けの率の、同じ日・同じ値動きの激しさの銘柄との差。* は月ごとの t値の絶対値が3以上。300件未満は空欄）"
)
print(f"\n合計 {time.time() - t0:.0f}秒")
