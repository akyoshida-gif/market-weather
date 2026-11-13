"""Does the dynamics layer add anything? Same walk-forward LightGBM twice: (a) atlas numbers only, (b) atlas + t_*.
Train on years before Y (label gap), predict year Y, 2014 .. 2024-06. Outcome: label minus the same-day average.
Also reports the big-loss rate (low within 20 sessions 10%+ below entry) of the bottom decile.
Usage: python scripts/atlas/ml_compare.py <atlas_dir> <label> <every>"""

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
df, base = search.load(A, label, search.PERIODS, every=every)
dyn = pl.scan_parquet(f"{A}/dyn/*.parquet").join(
    df.select("entity_id", "as_of").lazy(), on=["entity_id", "as_of"], how="semi"
)
df = df.join(dyn.collect(), on=["entity_id", "as_of"], how="left")
extra = pl.scan_parquet(f"{A}/stocks/*.parquet").select(
    "entity_id", "as_of", "y_mae20", C("s_atrp").alias("_atrp")
)
df = df.join(
    extra.collect().join(df.select("entity_id", "as_of"), on=["entity_id", "as_of"], how="semi"),
    on=["entity_id", "as_of"],
    how="left",
)
dynf = [c for c in df.columns if c.startswith("t_")]
df = df.with_columns(
    C("as_of").dt.year().alias("yr"), ((C("y_mae20") * C("_atrp")) <= -0.10).alias("big")
)
print(
    f"{df.height:,} 行・元の数値 {len(base)} 個・動きの数値 {len(dynf)} 個 {time.time() - t0:.0f}秒",
    flush=True,
)
y = df["y"].to_numpy()
yr = df["yr"].to_numpy()
asof = df["as_of"].to_numpy()
params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 3000, "feature_fraction": 0.5,
          "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 8}  # fmt: skip
out = df.select("as_of", "yr", "y", "big")
for tag, names in (("a", base), ("b", base + dynf)):
    X = df.select(names).to_numpy().astype(np.float32)
    pred = np.full(len(y), np.nan)
    imp = np.zeros(len(names))
    for Y in range(2014, 2025):
        cut = np.datetime64(date(Y, 1, 1)) - np.timedelta64(search.label_gap(label).days, "D")
        tr = asof < cut
        te = (yr == Y) & (asof <= np.datetime64(date(2024, 6, 30)))
        lo_q, hi_q = np.quantile(y[tr], [0.01, 0.99])
        m = lgb.train(params, lgb.Dataset(X[tr], np.clip(y[tr], lo_q, hi_q)), num_boost_round=300)
        pred[te] = m.predict(X[te])
        imp += m.feature_importance("gain")
    print(f"  モデル {tag} 完了 {time.time() - t0:.0f}秒", flush=True)
    out = out.with_columns(pl.Series(f"p_{tag}", pred))
    if tag == "b":
        print("\n動きの数値を入れたモデルで効いた数値の上位30（★は動きの数値）：")
        for k in np.argsort(-imp)[:30]:
            print(
                f"  {'★' if names[k].startswith('t_') else ' '} {names[k]:<26} {imp[k] / imp.sum() * 100:5.1f}%"
            )
    del X
out = out.filter(C("p_a").is_not_nan())
out.write_parquet(f"{A}/mlcmp_{label}.parquet")
print(f"\n=== 予測の点数で毎日10組に分けたとき（{label}、全銘柄の平均との差）")
for tag, nm in (("a", "瞬間の値だけ"), ("b", "＋動き")):
    r = out.with_columns(
        ((C(f"p_{tag}").rank() / C(f"p_{tag}").count()).over("as_of") * 10)
        .ceil()
        .clip(1, 10)
        .cast(pl.Int8)
        .alias("dec")
    )
    mon = r.with_columns(C("as_of").dt.truncate("1mo").alias("mo")).group_by("mo").agg(
        C("y").filter(C("dec") == 10).mean().alias("top"), C("y").filter(C("dec") == 1).mean().alias("bot"),
        C("big").filter(C("dec") == 1).mean().alias("bigb"), C("big").filter(C("dec") == 10).mean().alias("bigt"),
        C("big").mean().alias("bigall")).sort("mo")  # fmt: skip
    print(f"\n  【{nm}】")
    for lo, hi, per in ((2014, 2019, "2014〜2019"), (2020, 2024, "2020〜2024年6月")):
        s = mon.filter((C("mo").dt.year() >= lo) & (C("mo").dt.year() <= hi))
        for col, lab in (("top", "一番上の組"), (None, "上−下")):
            v = (s["top"] - s["bot"]).to_numpy() if col is None else s[col].to_numpy()
            print(
                f"    {per} {lab}：月平均 {v.mean() * 100:+.2f}% t値 {v.mean() / (v.std(ddof=1) / np.sqrt(len(v))):.1f} 勝った月 {(v > 0).mean() * 100:.0f}%"
            )
        print(
            f"    {per} 大負けの率：一番下の組 {s['bigb'].mean() * 100:.1f}% / 一番上の組 {s['bigt'].mean() * 100:.1f}% / 全体 {s['bigall'].mean() * 100:.1f}%"
        )
print(f"合計 {time.time() - t0:.0f}秒")
