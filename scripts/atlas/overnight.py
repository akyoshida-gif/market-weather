"""Idea 49/50: hold only overnight (buy the close, sell the next open) or only intraday (buy the next open, sell
that close). Which stock states make each side strong? Per-trade returns are absolute (no costs, owner 2026-11-13),
NOT minus the same-day average, so on sampled days the market's own move dominates: ATLAS_STUDY 9 shows that the
picks of this sampled search shrink on every session. Kept as a record of that pitfall. Quintile edges from
discovery only; counted by month.
Usage: python scripts/atlas/overnight.py <atlas_dir> <px.parquet> <every>"""

import sys
import time

import numpy as np
import polars as pl

from mw.research import atlas, search

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A, PX, every = sys.argv[1], sys.argv[2], int(sys.argv[3])
t0 = time.time()
P = search.PERIODS
lo, hi = P["discovery"][0], P["validation"][1]
days = (
    pl.scan_parquet(f"{A}/market.parquet").select("as_of").filter((C("as_of") >= lo) & (C("as_of") <= hi))
    .collect().sort("as_of").with_row_index("_d").filter(C("_d") % every == 0).select("as_of")
)  # fmt: skip
sf = [
    c
    for c in pl.scan_parquet(f"{A}/stocks/part_000.parquet").collect_schema().names()
    if c.startswith("s_")
]
st = (
    pl.scan_parquet(f"{A}/stocks/*.parquet")
    .filter((C("kind") == "common") & (C("dvol20") >= 5e6) & (C("raw_close") >= 5))
    .join(days.lazy(), on="as_of", how="semi")
    .select("entity_id", "as_of", *sf, C("y_ret1").alias("id1"))
)
# prices with unadjusted split steps repaired (as in the atlas); only the stocks in the study
_ids = st.select("entity_id").unique().collect()["entity_id"].to_list()
_px = (
    pl.scan_parquet(PX)
    .filter(C("entity_id").is_in(_ids))
    .select("entity_id", "as_of", "open", "high", "low", "close", "volume", "splits")
    .collect()
)
nxt = (
    atlas.repair_splits(_px)
    .sort("entity_id", "as_of")
    .with_columns(
        (C("open").shift(-1) / C("close") - 1).over("entity_id").alias("on1"),
    )
    .select("entity_id", "as_of", "on1")
    .lazy()
)
del _px
dyn = pl.scan_parquet(f"{A}/dyn/*.parquet")
df = (
    st.join(nxt, on=["entity_id", "as_of"], how="left")
    .join(dyn, on=["entity_id", "as_of"], how="left")
    .collect()
)
# no filter on the outcomes themselves: data errors are excluded by the atlas from signal-day data (bad20)
df = df.filter(C("on1").is_finite() & C("id1").is_finite())
df = search.purge(df, "y_ret1", P)
feats = [c for c in df.columns if c.startswith(("s_", "t_"))]
asof = df["as_of"]
disc = ((asof >= P["discovery"][0]) & (asof <= P["discovery"][1])).to_numpy()
val = ((asof >= P["validation"][0]) & (asof <= P["validation"][1])).to_numpy()
ym = (asof.dt.year().cast(pl.Int64) * 12 + asof.dt.month().cast(pl.Int64) - 1).to_numpy()
months, mi = np.unique(ym, return_inverse=True)
yom = months // 12
print(f"{df.height:,} 行・{len(feats)} 個 {time.time() - t0:.0f}秒", flush=True)

print("\n=== 全体：1回あたりの平均（年ごと）")
yr = (
    df.group_by(C("as_of").dt.year().alias("y"))
    .agg(C("on1").mean(), C("id1").mean(), pl.len())
    .sort("y")
)
for r in yr.iter_rows(named=True):
    print(f"  {r['y']}  夜間 {r['on1'] * 100:+.3f}%  日中 {r['id1'] * 100:+.3f}%")


def st_(mask: np.ndarray, y: np.ndarray) -> dict:
    s = np.bincount(mi[mask], weights=y[mask], minlength=len(months))
    c = np.bincount(mi[mask], minlength=len(months))
    r = search._monthly(s, c, yom)
    r["hit"] = float((y[mask] > 0).mean()) if mask.any() else np.nan
    return r


for lab, name in (
    ("on1", "夜間だけ持つ（引けで買い、翌朝の寄りで売る）"),
    ("id1", "日中だけ持つ（翌朝の寄りで買い、その日の引けで売る）"),
):
    y = df[lab].to_numpy()
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
            a, b = st_(m & disc, y), st_(m & val, y)
            rows.append({"f": f, "lo": -np.inf if k == 0 else e[k - 1], "hi": np.inf if k == len(e) else e[k],
                         "d": a["mean"], "dt": a["t"], "dh": a["hit"], "v": b["mean"], "vt": b["t"], "vh": b["hit"]})  # fmt: skip
    R = pl.DataFrame(rows).filter(C("dt").is_finite() & C("vt").is_finite())
    R.write_parquet(f"{A}/overnight_{lab}.parquet")
    both = R.filter((C("dt") >= 3) & (C("vt") >= 2))
    print(
        f"\n=== {name}：探す期間 t値3以上 かつ 確かめる期間 t値2以上で、1回あたりの平均が高い段階（{both.height} 個）"
    )
    for r in both.sort("v", descending=True).head(25).iter_rows(named=True):
        print(
            f"  {r['f']:<18} {r['lo']:>9.3g}〜{r['hi']:<9.3g} 探す {r['d'] * 100:+.3f}%（勝率{r['dh'] * 100:.0f}% t{r['dt']:4.1f}） / 確認 {r['v'] * 100:+.3f}%（勝率{r['vh'] * 100:.0f}% t{r['vt']:4.1f}）"
        )
    worst = R.filter((C("dt") <= -3) & (C("vt") <= -2))
    print(f"  --- 逆に、平均がはっきりマイナスの段階（{worst.height} 個、上位10）")
    for r in worst.sort("v").head(10).iter_rows(named=True):
        print(
            f"  {r['f']:<18} {r['lo']:>9.3g}〜{r['hi']:<9.3g} 探す {r['d'] * 100:+.3f}%（t{r['dt']:4.1f}） / 確認 {r['v'] * 100:+.3f}%（t{r['vt']:4.1f}）"
        )
print(f"\n合計 {time.time() - t0:.0f}秒")
