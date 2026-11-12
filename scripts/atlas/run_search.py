"""Run the search on the atlas. Usage: python run_search.py <atlas_dir> <label> <every> [recent]"""

import json
import sys
import time

import numpy as np

from mw.research import search

sys.stdout.reconfigure(encoding="utf-8")
A, label, every = sys.argv[1], sys.argv[2], int(sys.argv[3])
recent = len(sys.argv) > 4
periods = search.RECENT if recent else search.PERIODS
t0 = time.time()
df, names = search.load(A, label, periods, every=every)
print(f"{df.height:,} 行・{len(names)} 個の数値 読込 {time.time() - t0:.0f}秒", flush=True)
d = search.encode(df, names, periods)
del df
print(
    f"使える数値 {len(d.names)} 個 {time.time() - t0:.0f}秒（結果：{label}、同じ日の全銘柄の平均との差）",
    flush=True,
)


def fmt(s: dict) -> str:
    if not s["n"] or not np.isfinite(s.get("t", np.nan)):
        return f"{s['n']:>8,}件（少なすぎて測れない）"
    return (
        f"{s['n']:>8,}件 平均{s['mean'] * 100:+6.2f}% 勝率{s['hit'] * 100:5.1f}% t値{s['t']:5.1f} "
        f"勝った月{s['m_pos'] * 100:3.0f}% 勝った年{s['y_pos'] * s['years']:.0f}/{s['years']}"
    )


sg = search.singles(d)
print(f"\n=== 銘柄の数値だけの単独の条件・上位30（全 {len(sg)} 個）")
for _t, j, lo, hi in sg[:30]:
    m = (d.codes[:, j] >= lo) & (d.codes[:, j] <= hi)
    print(f"  {search.cond_text(d.names[j], lo, hi, d.edges)}")
    print(f"      探す {fmt(search.stats(d, m & d.disc))}")
    print(f"      確認 {fmt(search.stats(d, m & d.val))}")
okv = sum(
    1
    for t, j, lo, hi in sg[:100]
    if search.stats(d, (d.codes[:, j] >= lo) & (d.codes[:, j] <= hi) & d.val)["mean"] > 0
)
print(
    f"  上位100個のうち、確かめる期間でも他の銘柄に勝ったもの：{okv}/100（偶然なら約50）",
    flush=True,
)

print("\n=== 段階的な絞り込み（出発点を変えた20本。t値が上がらなくなったら止める）")
weapons, seen = [], set()
for _t, j, lo, hi in sg:
    if d.names[j] in seen:
        continue
    seen.add(d.names[j])
    path = search.stepwise(d, (j, lo, hi))
    weapons.append(path)
    print(f"\n--- 武器 {len(weapons)}")
    for p in path:
        print(f"  {p['step']}. {p['cond']}")
        print(f"      探す {fmt(p['disc'])}")
        print(f"      確認 {fmt(p['val'])}")
    if len(weapons) >= 20:
        break
print(f"\n段階的な絞り込み {time.time() - t0:.0f}秒", flush=True)

pr = search.pairs(d)
pos = sum(1 for p in pr if p["val"]["n"] and p["val"]["mean"] > 0)
print(
    f"\n=== 総当たりの地図（2つの数値の5×5、t値3超・8割以上の年で勝ち）{len(pr)} 升のうち、確かめる期間でも勝ち：{pos}"
)
for p in pr[:60]:
    ds = p["disc"]
    print(f"  {p['a']} × {p['b']}")
    print(
        f"      探す {ds['n']:,}件 平均{ds['mean'] * 100:+.2f}% t値{ds['t']:.1f}（{ds['months']}か月） / 確認 {fmt(p['val'])}"
    )
out = A + f"/search_{label}{'_recent' if recent else ''}.json"
singles = [
    {"cond": search.cond_text(d.names[j], lo, hi, d.edges), "t": t} for t, j, lo, hi in sg[:300]
]
with open(out, "w", encoding="utf-8") as fh:
    payload = {"label": label, "weapons": weapons, "pairs": pr, "singles": singles}
    json.dump(payload, fh, ensure_ascii=False, default=float)
print(f"合計 {time.time() - t0:.0f}秒")
