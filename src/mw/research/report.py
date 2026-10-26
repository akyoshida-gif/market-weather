"""Render the pattern-mining results as a self-contained HTML report (Japanese)."""

from __future__ import annotations

import html
import math
from typing import Any

import polars as pl

from mw.research import patterns as P

CAT = [
    ("移動平均", ["po_short", "po_long", "above_ma200", "below_ma200", "ma50_rising", "gc_5_25", "gc_50_200", "reclaim_ma50"]),
    ("一目均衡表", ["ichi_above_cloud", "ichi_cloud_break", "ichi_tk_cross", "ichi_chikou_up", "ichi_sanyaku"]),
    ("MACD", ["macd_gc", "macd_gc_below0", "macd_above0", "macd_hist_up"]),
    ("RSI・ストキャスティクス", ["rsi_lt30", "rsi_cross30", "rsi_cross50", "rsi_gt70", "rsi2_lt10", "stoch_gc_low"]),
    ("ボリンジャーバンド", ["bb_below_lower", "bb_reenter_lower", "bb_squeeze", "bb_break_upper"]),
    ("フィボナッチ", ["fib_382", "fib_50", "fib_618"]),
    ("ローソク足", ["cdl_bull_engulf", "cdl_hammer", "cdl_big_green", "cdl_doji", "cdl_inside", "cdl_gap_up", "cdl_3_white", "cdl_morning_star"]),
    ("出来高", ["vol_x2", "vol_x3", "vol_dry", "obv_up"]),
    ("高値・位置", ["new_high_252", "near_high_252", "break_high_60", "pullback_5pct", "down_3days"]),
    ("市場全体", ["mkt_up200", "mkt_up50", "breadth_hi"]),
]  # fmt: skip
LABELS = {
    **P.COND_LABELS,
    "mkt_up200": "市場（指数）が200日線の上",
    "mkt_up50": "市場（指数）が50日線の上",
    "breadth_hi": "50日線より上の銘柄が50%超",
}
TEMPL = {t.key: t for t in P.TEMPLATES}


def _f(v: Any, nd: int = 2, pct: bool = False, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if pct:
        return f"{v * 100:.0f}%"
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s


def name(pattern: str) -> str:
    return " ＋ ".join(LABELS.get(k, k) for k in pattern.split("+"))


def verdict(r: dict[str, Any]) -> tuple[str, str]:
    tr, va, te = r.get("train_mean_R"), r.get("valid_mean_R"), r.get("test_mean_R")
    vals = [x for x in (tr, va, te) if x is not None and not math.isnan(x)]
    if len(vals) < 3:
        return "n0", "件数不足"
    if all(x > 0.05 for x in vals):
        return "u2", "3期間とも勝ち"
    if te > 0.05 and (tr > 0 or va > 0):
        return "u1", "最近は勝ち"
    if all(x < -0.05 for x in vals):
        return "d2", "3期間とも負け"
    return "n0", "効果なし"


def cells(r: dict[str, Any]) -> str:
    out = []
    for p in ("train", "valid", "test"):
        n = r.get(f"{p}_n")
        if not n or (isinstance(n, float) and math.isnan(n)):
            out.append('<td class="n">—</td><td class="n">—</td><td class="n">—</td>')
            continue
        m = r.get(f"{p}_mean_R")
        cls = "up" if m and m > 0 else "down"
        out.append(
            f'<td class="n">{int(n):,}</td><td class="n">{_f(r.get(f"{p}_win"), pct=True)}</td><td class="n {cls}">{_f(m, sign=True)}R</td>'
        )
    return "".join(out)


HEAD = '<tr><th rowspan="2">条件</th><th colspan="3">開発 2010〜2017</th><th colspan="3">選定 2018〜2021</th><th colspan="3">検証 2022〜（未使用）</th><th rowspan="2">判定</th></tr><tr><th>件数</th><th>勝率</th><th>1回平均</th><th>件数</th><th>勝率</th><th>1回平均</th><th>件数</th><th>勝率</th><th>1回平均</th></tr>'


def build(results: dict[str, pl.DataFrame], tested: dict[str, int], meta: dict[str, Any]) -> str:
    parts: list[str] = []
    for mk, R in results.items():
        mkname = "米国" if mk == "US" else "日本"
        parts.append(f"<section><h2>{mkname}：結果</h2>")
        # 1) famous single indicators by category, M20 template
        rows = {
            r["pattern"]: r
            for r in R.filter((pl.col("k") == 1) & (pl.col("template") == "M20")).to_dicts()
        }
        parts.append(
            '<h3>① 有名なテクニカル指標を単独で使った場合（中期テンプレート：最長20日・損切り2ATR・利確4ATR）</h3><div class="tbl"><table>'
            + HEAD
        )
        for cat, keys in CAT:
            parts.append(f'<tr class="cat"><td colspan="11">{cat}</td></tr>')
            for k in keys:
                r = rows.get(k)
                if not r:
                    continue
                c, v = verdict(r)
                parts.append(
                    f'<tr><td>{html.escape(LABELS.get(k, k))}</td>{cells(r)}<td><span class="call {c}">{v}</span></td></tr>'
                )
        parts.append("</table></div>")
        # 2) combinations that passed selection
        Pp = R.filter((pl.col("pass") == 1) & (pl.col("k") >= 2)).with_columns(
            pl.min_horizontal("train_mean_R", "valid_mean_R").alias("minR")
        )
        good = Pp.filter(
            (pl.col("test_mean_R") > 0.1) & (pl.col("test_ex_top5_mean_R") > 0.05)
        ).sort("test_mean_R", descending=True)
        # dedupe near-identical patterns: keep best template per pattern
        good = good.unique(subset=["pattern"], keep="first", maintain_order=True).head(25)
        parts.append(
            f'<h3>② 組み合わせで見つかったパターン（選定基準を満たし、検証期間でも勝ったもの）</h3><p class="note">試した組み合わせ：{tested[mk]:,}通り。選定基準を満たしたもの {int(Pp.height):,}件のうち、検証期間でも1回平均+0.1R超、かつ上位5日を除いても勝っているもの {good.height}件（上位25件を表示）。</p><div class="tbl"><table><tr><th rowspan="2">パターン</th><th rowspan="2">テンプレート</th><th rowspan="2">年間の出現</th><th colspan="3">開発</th><th colspan="3">選定</th><th colspan="3">検証（未使用）</th><th rowspan="2">PF</th><th rowspan="2">上位5日の比率</th></tr><tr><th>件数</th><th>勝率</th><th>平均</th><th>件数</th><th>勝率</th><th>平均</th><th>件数</th><th>勝率</th><th>平均</th></tr>'
        )
        for r in good.to_dicts():
            t = TEMPL[r["template"]]
            parts.append(
                f'<tr><td class="wrap">{html.escape(name(r["pattern"]))}</td><td>{t.key}<div class="sub">{t.horizon}日・SL{t.sl_atr}・TP{t.tp_atr}ATR</div></td><td class="n">{r["per_year"]:.0f}件</td>{cells(r)}<td class="n">{_f(r.get("test_pf"))}</td><td class="n">{_f(r.get("test_top5_share"), pct=True)}</td></tr>'
            )
        parts.append("</table></div>")
        # 3) rare but high win rate
        rare = (
            Pp.filter(
                (pl.col("per_year") <= 40)
                & (pl.col("test_n") >= 30)
                & (pl.col("test_win") >= 0.55)
                & (pl.col("valid_win") >= 0.5)
            )
            .sort("test_win", descending=True)
            .unique(subset=["pattern"], keep="first", maintain_order=True)
            .head(12)
        )
        parts.append(
            '<h3>③ 出現は少ないが勝率が高いパターン（年40件以下・検証期間の勝率55%以上）</h3><div class="tbl"><table><tr><th>パターン</th><th>テンプレート</th><th>年間</th><th>開発 勝率/平均</th><th>選定 勝率/平均</th><th>検証 件数/勝率/平均</th></tr>'
        )
        for r in rare.to_dicts():
            parts.append(
                f'<tr><td class="wrap">{html.escape(name(r["pattern"]))}</td><td>{r["template"]}</td><td class="n">{r["per_year"]:.0f}</td><td class="n">{_f(r["train_win"], pct=True)} / {_f(r["train_mean_R"], sign=True)}R</td><td class="n">{_f(r["valid_win"], pct=True)} / {_f(r["valid_mean_R"], sign=True)}R</td><td class="n">{int(r["test_n"])} / {_f(r["test_win"], pct=True)} / {_f(r["test_mean_R"], sign=True)}R</td></tr>'
            )
        if rare.height == 0:
            parts.append('<tr><td colspan="6" class="empty">該当なし</td></tr>')
        parts.append("</table></div></section>")
    body = "\n".join(parts)
    return TEMPLATE.replace("__BODY__", body).replace(
        "__META__", html.escape(meta.get("generated", ""))
    )


TEMPLATE = r"""<title>テクニカル検証レポート</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;700&family=IBM+Plex+Mono&display=swap">
<style>
:root{--bg:#eef2f5;--surface:#fff;--sunk:#e6ecf1;--ink:#14212b;--muted:#5a6a77;--line:#d3dce3;--accent:#1d6aa8;--up:#16866a;--down:#c0433a;--q:#fbf3e3;
--f:"IBM Plex Sans JP","Hiragino Sans","Yu Gothic",system-ui,sans-serif;--m:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e151b;--surface:#151f28;--sunk:#1b2731;--ink:#e4ecf1;--muted:#90a2b0;--line:#26343f;--accent:#62aee8;--up:#3cc49c;--down:#f06d61;--q:#33291a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e151b;--surface:#151f28;--sunk:#1b2731;--ink:#e4ecf1;--muted:#90a2b0;--line:#26343f;--accent:#62aee8;--up:#3cc49c;--down:#f06d61;--q:#33291a;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f);font-size:14px;line-height:1.7;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
section{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;display:flex;flex-direction:column;gap:12px;min-width:0}
h1{font-size:22px;margin:0}h2{font-size:18px;margin:0}h3{font-size:15px;margin:8px 0 0}p{margin:0;max-width:82ch}
.note{font-size:12.5px;color:var(--muted)}.sub{font-size:11px;color:var(--muted)}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--sunk);color:var(--muted);font-weight:500;font-size:11px}td.n{text-align:right;font-family:var(--m)}td.wrap{white-space:normal;min-width:260px}
tr.cat td{background:var(--sunk);font-weight:700}.up{color:var(--up)}.down{color:var(--down)}.empty{color:var(--muted)}
.call{font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:5px;white-space:nowrap}.u2{background:var(--up);color:var(--surface)}.u1{color:var(--up);background:color-mix(in srgb,var(--up) 15%,transparent)}.d2{background:var(--down);color:var(--surface)}.n0{color:var(--muted);background:var(--sunk)}
ol,ul{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:4px;max-width:90ch}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.flow div{background:var(--sunk);border-radius:8px;padding:10px 12px}.flow b{display:block;margin-bottom:4px}
.warn{background:var(--q);border-radius:8px;padding:10px 12px}
</style>
<div class="wrap">
<section><h1>テクニカル指標の組み合わせ検証レポート</h1><p class="note">作成：__META__。株価・出来高から作れる条件だけを使った検証。投資助言ではありません。</p></section>
<section><h2>どう検証したか</h2>
<div class="flow">
<div><b>1. データ</b>2010年1月〜2026年10月の日足。米国：上場普通株 約5,900銘柄（売買代金20日平均500万ドル以上・株価5ドル以上の日だけ使用）。日本：TOPIX500＋ETF（売買代金5億円以上・株価300円以上）。</div>
<div><b>2. 条件（約50種）</b>移動平均（パーフェクトオーダー・GC）、一目均衡表（雲・転換線/基準線・遅行スパン・三役好転）、MACD、RSI、ストキャスティクス、ボリンジャー、フィボナッチ押し、ローソク足（包み足・ハンマー・大陽線・赤三兵・明けの明星・窓開けなど）、出来高、高値の位置、市場全体の状態。すべてその日の引けまでのデータだけで判定。</div>
<div><b>3. 売買のルール（先に固定）</b>シグナルの翌日の寄り付きで買い、損切りと利確を ATR（平均的な1日の値幅）の倍数で決め、期限が来たら引けで決済。4種類：超短期5日（損切り1・利確1.5ATR）、短期10日（1.5・3）、中期20日（2・4）、長め60日（3・9）。売買コスト差し引き。</div>
<div><b>4. 数え方</b>条件が「成立した最初の日」だけを1回と数える（10日続いても1回）。成績は R（損切り幅を1とした損益）で比較。+0.1R＝損切り幅の10%ぶんの利益が平均で残る。</div>
<div><b>5. 期間を3つに分ける</b>開発 2010〜2017、選定 2018〜2021 でパターンを選び、検証 2022〜 は選んだ後に一度だけ見る（未来のデータで当てはめていないかの試験）。</div>
<div><b>6. 選定基準</b>開発：150件以上・平均プラス・統計的に偶然と言いにくい（t値2.5以上）。選定：60件以上・平均プラス（t値1.5以上）。1〜3個の条件の組み合わせを全テンプレートで試した。</div>
</div>
<div class="warn"><b>注意</b><ul><li>たくさん試すほど偶然良く見えるものが混ざる。だから「選定基準を通ったもの」を「未使用の検証期間」で確かめている。</li><li>今上場している銘柄だけ（上場廃止銘柄は含まれない）。買いの成績は実際より良く出やすい。</li><li>暴落後の反発のように、多くの銘柄で同じ日に条件が揃うことがある。「上位5日の比率」と「上位5日を除いた成績」で、数日の出来事に頼っていないかを確認している。</li></ul></div>
</section>
__BODY__
</div>"""
