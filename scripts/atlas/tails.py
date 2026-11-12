"""Extreme states only: for every stock number, the bottom/top 1% and 5% (edges from discovery), outcome over
5/10/20 sessions vs the same-day average, counted by month. Usage: python tails.py <atlas_dir>"""

import sys
import time
from datetime import date

import numpy as np
import polars as pl

from mw.research import search

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1]
t0 = time.time()
st = pl.scan_parquet(f"{A}/stocks/*.parquet")
feats = [c for c in st.collect_schema().names() if c.startswith("s_")]
labels = ["y_ret5", "y_ret10", "y_ret20"]
base = st.filter(
    (C("kind") == "common")
    & (C("as_of") <= date(2024, 6, 30))
    & (C("dvol20") >= 5e6)
    & (C("raw_close") >= 5)
)
df = base.select("as_of", *labels).collect()
df = df.with_columns([(C(lb) - C(lb).mean().over("as_of")).alias(lb) for lb in labels])
print(f"{df.height:,} 行 {time.time() - t0:.0f}秒", flush=True)
asof = df["as_of"]
disc = (asof <= date(2019, 12, 31)).to_numpy()
val = ~disc
ym = (asof.dt.year().cast(pl.Int64) * 12 + asof.dt.month().cast(pl.Int64)).to_numpy()
mi = np.unique(ym, return_inverse=True)[1]
nm = mi.max() + 1
Y = {lb: df[lb].to_numpy().astype(np.float64) for lb in labels}
# purge: an outcome must end inside its own period (no validation prices in discovery, nothing after 2024-06)
for lb in labels:
    gap = search.label_gap(lb)
    cross = ((asof > date(2019, 12, 31) - gap) & (asof <= date(2019, 12, 31))) | (
        asof > date(2024, 6, 30) - gap
    )
    Y[lb][cross.to_numpy()] = np.nan
keys = df["as_of"]


def mstat(mask: np.ndarray, y: np.ndarray) -> tuple[float, float, int, float]:
    ok = mask & np.isfinite(y)
    s = np.bincount(mi[ok], weights=y[ok], minlength=nm)
    c = np.bincount(mi[ok], minlength=nm)
    g = c >= 10
    if g.sum() < 6:
        return np.nan, np.nan, int(g.sum()), np.nan
    mm = s[g] / c[g]
    return (
        float(y[ok].mean()),
        float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))),
        int(g.sum()),
        float((mm > 0).mean()),
    )


rows = []
for f in feats:
    col = base.select("as_of", f).collect()
    assert col["as_of"].equals(keys), "row order changed between reads"
    x = col[f].cast(pl.Float64).to_numpy()
    xd = x[disc & np.isfinite(x)]
    if len(xd) < 10000:
        continue
    q = np.quantile(xd, [0.01, 0.05, 0.95, 0.99])
    for name, m in (
        ("下1%", x <= q[0]),
        ("下1〜5%", (x > q[0]) & (x <= q[1])),
        ("上5〜1%", (x >= q[2]) & (x < q[3])),
        ("上1%", x >= q[3]),
    ):
        for lb in labels:
            dm, dt, dk, dp = mstat(m & disc, Y[lb])
            vm, vt, vk, vp = mstat(m & val, Y[lb])
            rows.append(
                {
                    "feature": f,
                    "tail": name,
                    "label": lb,
                    "edge": f"{q[0] if name == '下1%' else q[1] if name == '下1〜5%' else q[2] if name == '上5〜1%' else q[3]:.3g}",
                    "disc_mean": dm,
                    "disc_t": dt,
                    "disc_months": dk,
                    "disc_mpos": dp,
                    "val_mean": vm,
                    "val_t": vt,
                    "val_months": vk,
                    "val_mpos": vp,
                }
            )
r = pl.DataFrame(rows).filter(C("disc_t").is_finite())
r.write_parquet(A + "/tails.parquet")
print(f"{r.height} 通り {time.time() - t0:.0f}秒")
for lb in labels:
    x = r.filter(C("label") == lb)
    strong = x.filter(C("disc_t").abs() >= 3)
    same = strong.filter((C("disc_mean") * C("val_mean")) > 0)
    print(
        f"\n=== {lb}：探す期間で |t|≥3 の {strong.height} 通りのうち、確かめる期間でも同じ向き {same.height}（{same.height / max(strong.height, 1) * 100:.0f}%）"
    )
    both = x.filter(
        (C("disc_t").abs() >= 3) & (C("val_t").abs() >= 2) & ((C("disc_mean") * C("val_mean")) > 0)
    ).sort(C("val_mean").abs(), descending=True)
    print(f"  両方の期間で強いもの（探す |t|≥3・確認 |t|≥2・同じ向き）：{both.height} 通り")
    for z in both.head(25).iter_rows(named=True):
        print(
            f"  {z['feature']:<22} {z['tail']:<6}（境目 {z['edge']}） 探す {z['disc_mean'] * 100:+.2f}% t{z['disc_t']:+.1f} 月{z['disc_mpos'] * 100:.0f}% / "
            f"確認 {z['val_mean'] * 100:+.2f}% t{z['val_t']:+.1f} 月{z['val_mpos'] * 100:.0f}%"
        )
