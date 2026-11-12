"""After the stop is touched, how often does the price still reach the target? (all liquid common stocks)"""

import sys

import polars as pl

C = pl.col
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1]
ups, dns = ["1", "1_5", "2", "3", "4"], ["0_5", "1", "1_5", "2", "3"]
df = (
    pl.scan_parquet(f"{A}/stocks/*.parquet")
    .filter(
        (C("kind") == "common")
        & (C("dvol20") >= 5e6)
        & (C("raw_close") >= 5)
        & (C("as_of") <= pl.date(2024, 6, 30))
    )
    .select("as_of", *[f"y_up{u}" for u in ups], *[f"y_dn{d}" for d in dns])
    .collect()
)
print(f"{df.height:,} 件（2011〜2024年6月、翌日の始値で買った場合。単位は買った日の ATR）\n")
print(
    "損切り幅 → 利確幅 | 先に利確 | 先に損切り | 損切りの後に利確まで戻った（損切りに掛かった中の割合）| 60日で両方届かず"
)
for d in dns:
    for u in ups:
        up, dn = C(f"y_up{u}"), C(f"y_dn{d}")
        r = (
            df.filter(up.is_not_null() & dn.is_not_null())
            .select(
                pl.len().alias("n"),
                ((up > 0) & ((dn < 0) | (up < dn))).mean().alias("tp_first"),
                ((dn > 0) & ((up < 0) | (dn <= up))).mean().alias("sl_first"),
                ((dn > 0) & (up > dn)).sum().alias("sl_then_tp"),
                ((dn > 0) & ((up < 0) | (dn <= up))).sum().alias("sl_n"),
                ((up < 0) & (dn < 0)).mean().alias("none"),
            )
            .row(0, named=True)
        )
        back = r["sl_then_tp"] / max(r["sl_n"], 1)
        print(
            f"  −{d.replace('_', '.')}ATR → +{u.replace('_', '.')}ATR | {r['tp_first'] * 100:5.1f}% | {r['sl_first'] * 100:5.1f}% | "
            f"{back * 100:5.1f}% | {r['none'] * 100:5.1f}%"
        )
    print()
# how long after the stop does the comeback take
d, u = "1", "2"
x = df.filter((C(f"y_dn{d}") > 0) & (C(f"y_up{u}") > C(f"y_dn{d}"))).select(
    (C(f"y_up{u}") - C(f"y_dn{d}")).alias("gap")
)
print(
    "−1ATR に掛かってから +2ATR に届くまでの日数：",
    x["gap"].quantile(0.25),
    x["gap"].median(),
    x["gap"].quantile(0.75),
)
