"""Strategy card for idea 49: hold overnight only. Every session, buy the stocks that meet the condition at the close
(equal weight), sell all at the next open. No costs (owner 2026-11-13). Every session, not sampled.
Conditions were chosen on 2011-2019 (overnight.py); 2020-2024-06 is the check. The holdout is not loaded.
Usage: python scripts/atlas/overnight_card.py <atlas_dir> <px.parquet>"""

import sys
from datetime import date

import numpy as np
import polars as pl

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A, PX = sys.argv[1], sys.argv[2]
END = date(2024, 6, 30)
st = (
    pl.scan_parquet(f"{A}/stocks/*.parquet")
    .filter(
        (C("kind") == "common")
        & (C("dvol20") >= 5e6)
        & (C("raw_close") >= 5)
        & (C("as_of") >= date(2011, 1, 1))
        & (C("as_of") < END)
    )
    .select("entity_id", "as_of", "s_id20", "s_atrp", "s_vol20", "s_ret5")
)
nxt = (
    pl.scan_parquet(PX)
    .select("entity_id", "as_of", "open", "close")
    .sort("entity_id", "as_of")
    .with_columns(
        (C("open").shift(-1) / C("close") - 1).over("entity_id").alias("on1"),
        C("as_of").shift(-1).over("entity_id").alias("next_day"),
    )
    .select("entity_id", "as_of", "on1", "next_day")
)
df = st.join(nxt, on=["entity_id", "as_of"], how="inner").collect()
# the sale happens on the next session; keep only trades that end before the holdout
df = df.filter(C("on1").is_finite() & (C("on1").abs() < 0.5) & (C("next_day") <= END))
spy = (
    pl.scan_parquet(PX)
    .filter(C("entity_id") == "US:SPY")
    .select("as_of", "close")
    .sort("as_of")
    .with_columns((C("close") / C("close").shift(1) - 1).alias("spy"))
    .collect()
)

RULES = {
    "全銘柄（比較用）": pl.lit(True),
    "日中に売られ続けている（20日の日中の合計 −4.9% 以下）": C("s_id20") <= -0.0492,
    "値動きが激しい（ATR が株価の 3.5% 以上）": C("s_atrp") >= 0.0351,
    "両方": (C("s_id20") <= -0.0492) & (C("s_atrp") >= 0.0351),
    "両方＋直近5日で下げている": (C("s_id20") <= -0.0492)
    & (C("s_atrp") >= 0.0351)
    & (C("s_ret5") < 0),
}


def dd(eq: np.ndarray) -> float:
    return float((eq / np.maximum.accumulate(eq) - 1).min())


for name, rule in RULES.items():
    t = df.filter(rule)
    day = t.group_by("as_of").agg(C("on1").mean().alias("r"), pl.len().alias("n")).sort("as_of")
    print(f"\n### {name}")
    for lo, hi, per in (
        (date(2011, 1, 1), date(2019, 12, 31), "2011〜2019（探す）"),
        (date(2020, 1, 1), END, "2020〜2024年6月（確認）"),
    ):
        tt = t.filter((C("as_of") >= lo) & (C("as_of") <= hi))
        d = day.filter((C("as_of") >= lo) & (C("as_of") <= hi))
        r = d["r"].to_numpy()
        eq = np.cumprod(1 + r)
        yrs = len(r) / 252
        mo = d.group_by(C("as_of").dt.truncate("1mo")).agg(((C("r") + 1).product() - 1).alias("m"))
        print(
            f"  {per}：売買 {tt.height:,} 回（1日平均 {d['n'].mean():.0f} 銘柄）"
            f" 1回の平均 {tt['on1'].mean() * 100:+.3f}% 勝率 {(tt['on1'] > 0).mean() * 100:.1f}%"
            f" 最悪の1回 {tt['on1'].min() * 100:+.1f}%"
        )
        print(
            f"      毎日均等に持つと：年率 {eq[-1] ** (1 / yrs) * 100 - 100:+.0f}%"
            f" 最大の落ち込み {dd(eq) * 100:.0f}% 勝った月 {(mo['m'] > 0).mean() * 100:.0f}%"
            f" 最悪の1日 {r.min() * 100:+.1f}%"
        )
s = spy.filter((C("as_of") >= date(2011, 1, 1)) & (C("as_of") <= END)).drop_nulls("spy")
for lo, hi, per in ((2011, 2019, "2011〜2019"), (2020, 2024, "2020〜2024年6月")):
    r = s.filter((C("as_of").dt.year() >= lo) & (C("as_of").dt.year() <= hi))["spy"].to_numpy()
    eq = np.cumprod(1 + r)
    print(
        f"\n比較：SPY を持ち続ける {per} 年率 {eq[-1] ** (252 / len(r)) * 100 - 100:+.0f}%"
        f" 最大の落ち込み {dd(eq) * 100:.0f}%"
    )
