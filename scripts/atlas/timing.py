"""Timing study: when does the whole market (SPY) go up? Market features (atlas market.parquet) plus the daily
distribution of stock states. Usage: python scripts/atlas/timing.py <atlas_dir>"""

import sys
import time
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl

from mw.research import search

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1]
t0 = time.time()
END = date(2024, 6, 30)
st = pl.scan_parquet(f"{A}/stocks/*.parquet")
spy = (
    st.filter(C("entity_id") == "US:SPY")
    .select("as_of", "close", "y_ret10", "y_ret20", "y_ret60", C("y_ret1").alias("y1"))
    .collect()
    .sort("as_of")
)
# distribution of stock states each day (liquid common stocks)
dist = (
    st.filter((C("kind") == "common") & (C("dvol20") >= 5e6) & (C("raw_close") >= 5))
    .group_by("as_of")
    .agg(
        (C("s_ret5") < -0.10).mean().alias("d_crash5"),
        (C("s_dlo60") <= 0.05).mean().alias("d_at_lo60"),
        (C("s_trapped60") >= 0.99).mean().alias("d_trapped"),
        C("s_dma50").median().alias("d_med_dma50"),
        C("s_flow20").median().alias("d_med_flow20"),
        (C("s_vr20") > 3).mean().alias("d_climax"),
        C("s_atrp").median().alias("d_med_atrp"),
        (C("s_streak") <= -4).mean().alias("d_down4"),
        C("s_upvol20").median().alias("d_med_upvol"),
    )
    .collect()
)
m = pl.read_parquet(f"{A}/market.parquet")
df = spy.join(m, on="as_of", how="left").join(dist, on="as_of", how="left").sort("as_of")
feats = [c for c in df.columns if c.startswith(("m_", "d_"))]
print(f"{df.height} 日・{len(feats)} 個 {time.time() - t0:.0f}秒", flush=True)
mo = (df["as_of"].dt.year().cast(pl.Int64) * 12 + df["as_of"].dt.month().cast(pl.Int64)).to_numpy()
mi = np.unique(mo, return_inverse=True)[1]
yr = df["as_of"].dt.year().to_numpy()
asof = df["as_of"]


def periods(label: str) -> tuple[np.ndarray, np.ndarray]:
    gap = search.label_gap(label)
    disc = (asof <= date(2019, 12, 31) - gap).to_numpy()
    val = ((asof >= date(2020, 1, 1)) & (asof <= END - gap)).to_numpy()
    return disc, val


def mstat(mask: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    ok = mask & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan, np.nan, np.nan
    s = np.bincount(mi[ok], weights=y[ok], minlength=mi.max() + 1)
    c = np.bincount(mi[ok], minlength=mi.max() + 1)
    g = c > 0
    mm = s[g] / c[g]
    t = (
        mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))
        if len(mm) > 2 and mm.std(ddof=1) > 0
        else np.nan
    )
    return float(y[ok].mean()), float(t), float((y[ok] > 0).mean())


for label in ("y_ret20", "y_ret60"):
    y = df[label].to_numpy().astype(float)
    disc, val = periods(label)
    bd, bv = mstat(disc, y), mstat(val, y)
    print(
        f"\n=== {label}（S&P 500）基準：探す 平均{bd[0] * 100:+.2f}% 上がった割合{bd[2] * 100:.0f}% / 確認 平均{bv[0] * 100:+.2f}% {bv[2] * 100:.0f}%"
    )
    rows = []
    for f in feats:
        x = df[f].cast(pl.Float64).to_numpy()
        xd = x[disc & np.isfinite(x)]
        if len(xd) < 200:
            continue
        e = np.quantile(xd, [0.2, 0.4, 0.6, 0.8])
        q = np.where(np.isfinite(x), np.searchsorted(e, x, side="right"), -1)
        for k, name in ((0, "下20%"), (4, "上20%")):
            dm, dt, dp = mstat(disc & (q == k), y)
            vm, vt, vp = mstat(val & (q == k), y)
            rows.append(
                {
                    "f": f,
                    "q": name,
                    "edge": float(e[0] if k == 0 else e[3]),
                    "dm": dm - bd[0],
                    "dt": dt,
                    "vm": vm - bv[0],
                    "vt": vt,
                    "dp": dp,
                    "vp": vp,
                }
            )
    r = pl.DataFrame(rows).filter(C("dm").is_finite() & C("vm").is_finite())
    same = r.filter((C("dm").abs() > 0.01) & ((C("dm") * C("vm")) > 0))
    big = r.filter(C("dm").abs() > 0.01)
    print(
        f"  探す期間で基準と1%以上違った {big.height} 通りのうち、確かめる期間でも同じ向き {same.height}"
    )
    for z in (
        same.sort(C("vm").abs() + C("dm").abs(), descending=True).head(15).iter_rows(named=True)
    ):
        print(
            f"  {z['f']:<18} {z['q']}（境目 {z['edge']:.3g}） 基準との差：探す {z['dm'] * 100:+.2f}%（上がった割合{z['dp'] * 100:.0f}%） / 確認 {z['vm'] * 100:+.2f}%（{z['vp'] * 100:.0f}%）"
        )

# walk-forward model: hold SPY only when the predicted 20-day return is above the training median
print(
    "\n=== 束ねる：前年までで学習し、予測が学習期間の真ん中より上の日だけ SPY を持つ（持たない日は現金・利息なし）"
)
label = "y_ret20"
y = df[label].to_numpy().astype(float)
X = df.select(feats).to_numpy().astype(np.float32)
pred = np.full(len(y), np.nan)
params = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "num_threads": 4,
}
thr = np.full(len(y), np.nan)
for Y in range(2014, 2025):
    cut = np.datetime64(date(Y, 1, 1)) - np.timedelta64(search.label_gap(label).days, "D")
    tr = (asof.to_numpy() < cut) & np.isfinite(y)
    te = yr == Y
    te &= asof.to_numpy() <= np.datetime64(END)
    mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=200)
    pred[te] = mdl.predict(X[te])
    thr[te] = np.median(mdl.predict(X[tr]))
hold = np.where(np.isfinite(pred), pred > thr, False)
# daily close-to-close return of SPY; decide at the close, hold from the next session
c = df["close"].to_numpy()
r1 = np.zeros(len(c))
r1[1:] = c[1:] / c[:-1] - 1
pos = np.zeros(len(c))
pos[1:] = hold[:-1]
use = (yr >= 2014) & (asof.to_numpy() <= np.datetime64(END))
for lo, hi, tag in ((2014, 2019, "2014〜2019"), (2020, 2024, "2020〜2024年6月")):
    sel = use & (yr >= lo) & (yr <= hi)
    bh = np.cumprod(1 + r1[sel])
    st_ = np.cumprod(
        1 + r1[sel] * pos[sel] - np.abs(np.diff(np.concatenate([[0], pos[sel]]))) * 0.0005
    )
    dd = lambda e: float((e / np.maximum.accumulate(e) - 1).min())  # noqa: E731
    print(
        f"  {tag}：持ち続け {bh[-1]:.2f}倍（最大の減り{dd(bh) * 100:.0f}%） / 予測で出入り {st_[-1]:.2f}倍（最大の減り{dd(st_) * 100:.0f}%、持っていた日{pos[sel].mean() * 100:.0f}%）"
    )
for Y in range(2014, 2025):
    sel = use & (yr == Y)
    if sel.any():
        print(
            f"    {Y}：持ち続け {np.prod(1 + r1[sel]) * 100 - 100:+.1f}% / 出入り {np.prod(1 + r1[sel] * pos[sel]) * 100 - 100:+.1f}%（持っていた日{pos[sel].mean() * 100:.0f}%）"
        )
print(f"合計 {time.time() - t0:.0f}秒")

# ---- weakness score: how many "market is weak" conditions hold at once (edges from the discovery period)
print("\n=== 弱った条件がいくつ重なっているか（0〜6）と、その後の S&P 500")
conds = [
    ("m_spy_dma50", "low"),
    ("m_spy_ret20", "low"),
    ("d_med_flow20", "low"),
    ("d_med_upvol", "low"),
    ("m_vix_term", "high"),
    ("m_skew", "low"),
]
disc60, _ = periods("y_ret60")
score = np.zeros(len(df), dtype=int)
for f, side in conds:
    x = df[f].cast(pl.Float64).to_numpy()
    xd = x[disc60 & np.isfinite(x)]
    edge = np.quantile(xd, 0.2 if side == "low" else 0.8)
    score += np.where(np.isfinite(x), (x <= edge) if side == "low" else (x >= edge), False).astype(
        int
    )
for label in ("y_ret20", "y_ret60"):
    y = df[label].to_numpy().astype(float)
    disc, val = periods(label)
    print(f"  {label}")
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 6)):
        sel = (score >= lo) & (score <= hi)
        dm, dt, dp = mstat(disc & sel, y)
        vm, vt, vp = mstat(val & sel, y)
        nd, nv = int((disc & sel).sum()), int((val & sel).sum())
        f = lambda v: "—" if not np.isfinite(v) else f"{v * 100:+.2f}%"  # noqa: E731
        print(
            f"    点数{lo}{'〜6' if hi == 6 else ''}：探す {f(dm)}（{nd}日・上がった割合{dp * 100 if np.isfinite(dp) else 0:.0f}%） / 確認 {f(vm)}（{nv}日・{vp * 100 if np.isfinite(vp) else 0:.0f}%）"
        )
