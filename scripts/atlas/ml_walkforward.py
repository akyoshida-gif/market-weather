"""Walk-forward: bundle all weak signals with LightGBM. Train on years before Y (10-session gap), predict year Y.
Usage: python scripts/atlas/ml_walkforward.py <atlas_dir> <label> <every>"""

import sys
import time
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl

from mw.research import search

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A, label, every = sys.argv[1], sys.argv[2], int(sys.argv[3])
t0 = time.time()
periods = {
    "discovery": (date(2011, 1, 1), date(2019, 12, 31)),
    "validation": (date(2020, 1, 1), date(2024, 6, 30)),
}
df, names = search.load(A, label, periods, every=every)
df = df.with_columns(C("as_of").dt.year().alias("yr"))
print(f"{df.height:,} 行・{len(names)} 個 {time.time() - t0:.0f}秒", flush=True)
X = df.select(names).to_numpy().astype(np.float32)
y = df["y"].to_numpy()
yr = df["yr"].to_numpy()
asof = df["as_of"].to_numpy()
params = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 3000,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "num_threads": 8,
}
pred = np.full(len(y), np.nan)
imp = np.zeros(len(names))
for Y in range(2014, 2025):
    # gap so the outcomes of the last training rows end before the predicted year starts
    cut = np.datetime64(date(Y, 1, 1)) - np.timedelta64(search.label_gap(label).days, "D")
    tr = asof < cut
    te = yr == Y
    if Y == 2024:
        te &= asof <= np.datetime64(date(2024, 6, 30))
    # clip the target (from the training rows only) so a few huge moves do not drive the trees
    lo_q, hi_q = np.quantile(y[tr], [0.01, 0.99])
    m = lgb.train(params, lgb.Dataset(X[tr], np.clip(y[tr], lo_q, hi_q)), num_boost_round=300)
    pred[te] = m.predict(X[te])
    imp += m.feature_importance("gain")
    print(f"  {Y} 学習 {tr.sum():,} 行 → 予測 {te.sum():,} 行 {time.time() - t0:.0f}秒", flush=True)
res = df.select("as_of", "yr", "y").with_columns(pl.Series("p", pred)).filter(C("p").is_not_nan())
res = res.with_columns(
    ((C("p").rank() / C("p").count()).over("as_of") * 10)
    .ceil()
    .clip(1, 10)
    .cast(pl.Int8)
    .alias("dec")
)
res.write_parquet(A + f"/ml_{label}.parquet")
print("\n=== 予測の点数で毎日10組に分けたとき、10日後に全銘柄の平均を何%上回ったか（年ごと）")
tab = (
    res.group_by("yr", "dec").agg(C("y").mean()).pivot(on="dec", index="yr", values="y").sort("yr")
)
cols = [str(i) for i in range(1, 11)]
print("  年    " + " ".join(f"{c:>6}" for c in cols) + "   上−下")
for r in tab.iter_rows(named=True):
    print(
        f"  {r['yr']}  "
        + " ".join(f"{(r[c] or 0) * 100:+6.2f}" for c in cols)
        + f"  {((r['10'] or 0) - (r['1'] or 0)) * 100:+6.2f}"
    )
mon = (
    res.with_columns(C("as_of").dt.truncate("1mo").alias("mo"))
    .group_by("mo")
    .agg(
        C("y").filter(C("dec") == 10).mean().alias("top"),
        C("y").filter(C("dec") == 1).mean().alias("bot"),
    )
    .sort("mo")
)
for nm, col in (("一番上の組", "top"), ("上−下", None)):
    v = (mon["top"] - mon["bot"]).to_numpy() if col is None else mon[col].to_numpy()
    for lo, hi, tag in ((2014, 2019, "2014〜2019"), (2020, 2024, "2020〜2024年6月")):
        s = v[(mon["mo"].dt.year() >= lo).to_numpy() & (mon["mo"].dt.year() <= hi).to_numpy()]
        print(
            f"  {nm} {tag}：月平均 {s.mean() * 100:+.2f}% t値 {s.mean() / (s.std(ddof=1) / np.sqrt(len(s))):.1f} 勝った月 {(s > 0).mean() * 100:.0f}%"
        )
print("\n効いた数値の上位25：")
for k in np.argsort(-imp)[:25]:
    print(f"  {names[k]:<26} {imp[k] / imp.sum() * 100:5.1f}%")
print(f"合計 {time.time() - t0:.0f}秒")
