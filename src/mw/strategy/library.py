"""Strategy families (parameterised). Each returns a Strategy with IN/TP/LS fixed at signal time.

Design intent per family (why it could have an edge):
  rsi2_dip       短期: short-term overreaction inside an established uptrend mean-reverts (liquid names).
  spike_fade     短期(空売り): small/mid caps up >=20% on >=5x volume give back gains (our data: 68% fall in 10d).
  squeeze_break  中期: volatility contraction -> breakout on heavy volume with strong relative strength,
                 only in an up market (institutional accumulation).
  momentum       中期: top cross-sectional 6-month momentum (skip last week) in uptrends, exit on trend break.
  gap_go         短期〜中期: news/earnings gap that holds into the close on heavy volume tends to drift.
  index_trend    長期: hold the index only above its 200-day MA (avoid the worst drawdowns).
"""

from __future__ import annotations

import itertools
from typing import Any

import polars as pl

from mw.strategy.engine import Strategy

C = pl.col
INF = float("inf")


def liq(market: str, usd: float) -> pl.Expr:
    return C("dvol20") >= (usd if market == "US" else usd * 150)  # rough JPY conversion


def rsi2_dip(market: str, rsi: float, k: float) -> Strategy:
    return Strategy(
        key=f"rsi2_dip|rsi{rsi}|k{k}", label="押し目（短期の売られすぎ）", horizon="短期", side="long",
        entry=lambda: (C("close") > C("ma200")) & (C("ma50") > C("ma200")) & (C("rsi2") < rsi)
        & liq(market, 20e6) & (C("close") >= (10 if market == "US" else 500)),
        stop=lambda: C("close") - k * C("atr"),
        target=lambda: pl.lit(INF),
        exit_rule=lambda: C("close") > C("ma5"),
        max_hold=10, rank=lambda: -C("rsi2"), params={"rsi": rsi, "k": k},
        why="上昇トレンド（200日線の上、50日線＞200日線）の中で2日RSIが極端に低い＝短期の売られすぎ。5日線を上回って引けたら利確。",
    )  # fmt: skip


def spike_fade(market: str, k: float, t: float, confirm: bool) -> Strategy:
    spike = (
        (C("ret1") >= 0.20)
        & (C("vr") >= 5)
        & (C("close") >= 1)
        & (C("close") <= 50)
        & liq(market, 1e6)
    )
    if confirm:  # enter only after the first red day following the spike
        prev_spike = spike.shift(1).over("entity_id")
        entry = lambda: prev_spike & (C("close") < C("open")) & (C("close") < C("prev_close"))  # noqa: E731
    else:
        entry = lambda: spike  # noqa: E731
    return Strategy(
        key=f"spike_fade|k{k}|t{t}|c{int(confirm)}", label="急騰の反落（空売り）", horizon="短期", side="short",
        entry=entry, stop=lambda: C("high") + k * C("atr"), target=lambda: C("close") * (1 - t),
        max_hold=10, rank=lambda: C("ret1"), params={"k": k, "t": t, "confirm": confirm},
        why="出来高5倍以上で+20%以上の急騰は、その後10日で反落しやすい（過去5年で68%が下落）。空売りが可能な銘柄のみ。",
    )  # fmt: skip


def squeeze_break(market: str, vr: float, contr: float, rs: float, rr: float) -> Strategy:
    stop = lambda: pl.max_horizontal(C("close") - 2 * C("atr"), C("hi60p") - C("atr"))  # noqa: E731
    return Strategy(
        key=f"squeeze_break|vr{vr}|c{contr}|rs{rs}|R{rr}", label="値幅収縮後のブレイクアウト", horizon="中期", side="long",
        entry=lambda: (C("close") > C("hi60p")) & (C("vr") >= vr) & (C("clv") >= 0.7)
        & ((C("atr10") / C("atr50")).shift(1).over("entity_id") <= contr) & (C("rs_pct") >= rs)
        & C("mkt_up50") & liq(market, 5e6) & (C("close") >= (5 if market == "US" else 300)),
        stop=stop, target=lambda: C("close") + rr * (C("close") - stop()),
        exit_rule=lambda: C("close") < C("ma20"), max_hold=40, rank=lambda: C("rs_pct") * C("vr"),
        params={"vr": vr, "contr": contr, "rs": rs, "R": rr},
        why="値動きが縮んだ（エネルギーを溜めた）後に、出来高を伴って60日高値を上抜け。市場より強い銘柄だけ、市場が上昇局面のときだけ。",
    )  # fmt: skip


def momentum(market: str, pct: float) -> Strategy:
    return Strategy(
        key=f"momentum|p{pct}", label="強い銘柄に乗る（モメンタム）", horizon="中期", side="long",
        entry=lambda: (C("mom_pct") >= pct) & (C("close") > C("ma50")) & (C("ma50") > C("ma200")) & C("mkt_up200")
        & liq(market, 10e6) & (C("close") >= (10 if market == "US" else 500)) & (C("as_of").dt.weekday() == 5),
        stop=lambda: C("close") - 2.5 * C("atr"), target=lambda: pl.lit(INF),
        exit_rule=lambda: C("close") < C("ma50"), max_hold=60, rank=lambda: C("mom_pct"), params={"pct": pct},
        why="過去6か月（直近1週を除く）の上昇率が上位の銘柄は、その後も強い傾向（モメンタム）。50日線割れで手仕舞い。毎週金曜に判定。",
    )  # fmt: skip


def gap_go(market: str, gap: float, rr: float) -> Strategy:
    return Strategy(
        key=f"gap_go|g{gap}|R{rr}", label="材料ギャップの継続", horizon="短期", side="long",
        entry=lambda: (C("gap") >= gap) & (C("vr") >= 3) & (C("clv") >= 0.75) & (C("close") > C("open"))
        & liq(market, 5e6) & (C("close") >= (5 if market == "US" else 300)),
        stop=lambda: C("low"), target=lambda: C("close") + rr * (C("close") - C("low")),
        max_hold=20, rank=lambda: C("gap") * C("vr"), params={"gap": gap, "R": rr},
        why="決算やニュースで窓を開けて上昇し、高値圏で引けた（売りを吸収した）銘柄は上昇が続きやすい。当日安値を割ったら撤退。",
    )  # fmt: skip


def pullback_rs(market: str, rs: float, rr: float) -> Strategy:
    stop = lambda: pl.min_horizontal(C("lo10"), C("ma50")) - 0.2 * C("atr")  # noqa: E731
    return Strategy(
        key=f"pullback_rs|rs{rs}|R{rr}", label="強い銘柄の押し目", horizon="中期", side="long",
        entry=lambda: (C("rs_pct") >= rs) & (C("close") > C("ma50")) & (C("ma50") > C("ma200")) & (C("ma50") > C("ma50_10"))
        & (C("low") <= C("ma20") * 1.01) & (C("close") > C("ma20")) & (C("close") > C("open")) & (C("vr") < 1.2)
        & liq(market, 5e6) & (C("close") >= (5 if market == "US" else 300)),
        stop=stop, target=lambda: C("close") + rr * (C("close") - stop()),
        exit_rule=lambda: C("close") < C("ma50"), max_hold=30, rank=lambda: C("rs_pct"), params={"rs": rs, "R": rr},
        why="市場より強い上昇トレンドの銘柄が、出来高を減らしながら20日線まで押して陽線で反発。売りが細っている押し目を拾う。",
    )  # fmt: skip


PANIC_TIMING = {
    "term": ("VIX÷VIX3Mが1を下回った日（恐怖の山を越えた）", lambda: (C("vix_term") < 1) & (C("vix_term_prev") >= 1)),
    "below": ("VIXが基準を下回った日", None),
    "cross": ("VIXが基準を超えた日（早めに買う）", None),
}  # fmt: skip
PANIC_PICK = {
    "bs": ("ベータ上位20%×パニック前の半年の強さ上位20%", lambda: (C("beta_pct") >= 0.8) & (C("pre_pct") >= 0.8), lambda: C("beta_pct") + C("pre_pct")),
    "beta": ("ベータ上位20%", lambda: C("beta_pct") >= 0.8, lambda: C("beta_pct")),
    "index": ("S&P 500（SPY）そのもの", None, None),
}  # fmt: skip
# Caps on over-extended names (backlog P17, docs/PANIC_STUDY.md section 6): the plain top picks lost most when the
# names that ran hardest before the panic broke down (2021-12, 2022).
PANIC_CAP = {
    "atr6": ("1日の値幅（ATR）が株価の6%以下の銘柄だけ", lambda: C("atr") <= 0.06 * C("close")),
    "pre100": ("パニック前の半年の上昇が+100%以下の銘柄だけ", lambda: C("pre_str") <= 1.0),
}  # fmt: skip


def panic_rebound(
    market: str,
    bench: str,
    x: float,
    timing: str,
    pick: str,
    hold: int,
    k: float,
    trend: bool = False,
    cap: str = "",
) -> Strategy:
    """Buy after a VIX panic. Episode: VIX closed above `x` within the last 30 sessions. Entry day by `timing`,
    stocks by `pick`, stop k*ATR below the close (k=0: only a -40% disaster stop; with 1% risk per trade the
    position is then 2.5% of equity), no target, sell after `hold` sessions. trend: only panics that hit an uptrend
    (20 sessions earlier SPY was above a rising 200-day line)."""
    t_name, t_cond = PANIC_TIMING[timing]
    if timing == "below":
        t_cond = lambda: (C("vix") < x) & (C("vix_prev") >= x)  # noqa: E731
    elif timing == "cross":
        t_cond = lambda: (C("vix") > x) & (C("vix_prev") <= x)  # noqa: E731
    p_name, p_cond, p_rank = PANIC_PICK[pick]
    if pick == "index":
        p_cond, p_rank = (lambda: C("entity_id") == bench), (lambda: pl.lit(1.0))  # noqa: E731
        universe = lambda: pl.lit(True)  # noqa: E731
    else:
        universe = lambda: liq(market, 10e6) & (C("close") >= 10)  # noqa: E731
    assert t_cond is not None and p_cond is not None and p_rank is not None
    tc0, pc, pr = t_cond, p_cond, p_rank

    def tc() -> pl.Expr:
        """Only the FIRST such day of a panic: the timing condition did not fire in the previous 30 sessions
        (VIX swings around the level and would otherwise buy again and again in one panic)."""
        fired = tc0().fill_null(False)
        before = (
            fired.cast(pl.Int8)
            .shift(1)
            .rolling_max(30, min_samples=1)
            .over("entity_id")
            .fill_null(0)
        )
        return fired & (before == 0)

    trend_ok = (lambda: C("spy_trend_pre").fill_null(False)) if trend else (lambda: pl.lit(True))  # noqa: E731
    cap_name, cap_ok = PANIC_CAP[cap] if cap else ("", lambda: pl.lit(True))
    stop_txt = (
        f"損切りは {k:g}ATR 下"
        if k > 0
        else "損切りは −40%（大事故の備えだけ。そのぶん1銘柄は資金の2.5%と小さく買う）"
    )
    return Strategy(
        key=f"panic_rebound|x{x:g}|{timing}|{pick}|h{hold}|k{k:g}" + ("|up" if trend else "") + (f"|{cap}" if cap else ""),
        label="パニック後の買い",
        horizon="中期", side="long",
        entry=lambda: (C("vix_max30") > x) & tc() & pc() & universe() & trend_ok() & cap_ok().fill_null(False),
        stop=(lambda: C("close") - k * C("atr")) if k > 0 else (lambda: C("close") * 0.6), target=lambda: pl.lit(INF),
        max_hold=hold, rank=pr, params={"x": x, "timing": timing, "pick": pick, "hold": hold, "k": k, "trend": trend, "cap": cap},
        why=f"VIX が {x:g} を超えるパニックの後は、市場全体が戻りやすい（1990年以降、VIX 30超で60日後の S&P 500 は平均+6%）。"
        f"買う日：{t_name}。銘柄：{p_name}。{stop_txt}、{hold} 日後に売る。"
        + ("上昇相場の中の急落だけ（20日前に S&P 500 が上向きの200日線の上）。" if trend else "")
        + (f"行き過ぎた銘柄を除く：{cap_name}。" if cap else ""),
    )  # fmt: skip


def weak_dip(
    market: str, bench: str, min_score: int, pick: str, hold: int, k: float, cap: str = ""
) -> Strategy:
    """Buy the dip of an uptrend: the first day the weakness score (regime.weak_score: S&P 500 under its 50-day
    line, weak 20-day return, low SKEW, high VIX for its last year) reaches min_score, only if 20 sessions earlier
    the S&P 500 was above a rising 200-day line. 1990-2024 (S&P 500): 46 such episodes, +3.9% after 60 sessions
    on average (all days +2.2%), worst -9.4% (docs/ATLAS_STUDY.md section 7)."""
    p_name, p_cond, p_rank = PANIC_PICK[pick]
    if pick == "index":
        p_cond, p_rank = (lambda: C("entity_id") == bench), (lambda: pl.lit(1.0))  # noqa: E731
        universe = lambda: pl.lit(True)  # noqa: E731
    else:
        universe = lambda: liq(market, 10e6) & (C("close") >= 10)  # noqa: E731
    assert p_cond is not None and p_rank is not None
    pc, pr = p_cond, p_rank
    cap_name, cap_ok = PANIC_CAP[cap] if cap else ("", lambda: pl.lit(True))

    def first_day() -> pl.Expr:
        fired = (C("weak_score") >= min_score).fill_null(False)
        before = (
            fired.cast(pl.Int8)
            .shift(1)
            .rolling_max(20, min_samples=1)
            .over("entity_id")
            .fill_null(0)
        )
        return fired & (before == 0)

    stop_txt = f"損切りは {k:g}ATR 下" if k > 0 else "損切りは −40%（大事故の備えだけ）"
    return Strategy(
        key=f"weak_dip|s{min_score}|{pick}|h{hold}|k{k:g}" + (f"|{cap}" if cap else ""), label="上昇相場の押し目",
        horizon="中期", side="long",
        entry=lambda: first_day() & C("spy_trend_pre").fill_null(False) & pc() & universe() & cap_ok().fill_null(False),
        stop=(lambda: C("close") - k * C("atr")) if k > 0 else (lambda: C("close") * 0.6), target=lambda: pl.lit(INF),
        max_hold=hold, rank=pr, params={"min_score": min_score, "pick": pick, "hold": hold, "k": k, "cap": cap},
        why=f"上昇相場の中で市場が弱った日に買う（S&P 500 が50日線の下・20日で下げ・SKEW が低い・VIX が高い、の4つのうち"
        f"{min_score}つ以上が重なった最初の日。20日前は上向きの200日線の上）。1990年以降46回、60日後の平均+3.9%（全日+2.2%）。"
        f"銘柄：{p_name}。{stop_txt}、{hold} 日後に売る。" + (f"行き過ぎた銘柄を除く：{cap_name}。" if cap else ""),
    )  # fmt: skip


def weak_grid(market: str, bench: str) -> list[Strategy]:
    """Dip-in-uptrend variants. Research only, NOT in paper.yaml: 2011-2024 they did no better than holding the
    S&P 500 (docs/ATLAS_STUDY.md section 7)."""
    if market != "US":
        return []
    return [weak_dip(market, bench, 3, "index", 60, 0.0), weak_dip(market, bench, 3, "bs", 60, 3.0),
            weak_dip(market, bench, 3, "bs", 60, 3.0, "atr6")]  # fmt: skip


def panic_grid(market: str, bench: str) -> list[Strategy]:
    """Variants run side by side (owner 2026-11-06: many patterns, compare live). Backtest 2010-2026 in
    docs/PANIC_STUDY.md: results depend on a few large rebounds; none is proven yet."""
    if market != "US":
        return []
    keys = [(30, "cross", "bs", 60, 3.0, False), (30, "below", "bs", 60, 3.0, False), (30, "term", "bs", 60, 3.0, False),
            (35, "term", "bs", 60, 3.0, False), (30, "cross", "bs", 60, 0.0, False), (30, "cross", "bs", 60, 3.0, True),
            (30, "cross", "index", 60, 0.0, False), (30, "term", "index", 60, 0.0, False), (25, "term", "bs", 60, 3.0, False),
            (35, "cross", "bs", 60, 0.0, True),
            # caps on over-extended names (P17, 2026-11-09)
            (30, "cross", "bs", 60, 3.0, False, "atr6"), (30, "cross", "bs", 60, 0.0, False, "atr6"),
            (30, "cross", "bs", 60, 0.0, False, "pre100")]  # fmt: skip
    return [panic_rebound(market, bench, *k) for k in keys]


def index_trend(market: str, bench: str) -> Strategy:
    return Strategy(
        key="index_trend", label="指数の長期トレンド", horizon="長期", side="long",
        entry=lambda: (C("entity_id") == bench) & (C("close") > C("ma200")) & (C("close").shift(1).over("entity_id") <= C("ma200").shift(1).over("entity_id")),
        stop=lambda: C("close") * 0.8, target=lambda: pl.lit(INF),
        exit_rule=lambda: C("close") < C("ma200") * 0.98, max_hold=2000, params={},
        why="指数が200日線を上抜けたら保有し、200日線を2%下回ったら手放す。大きな下落局面を避ける長期の基本戦略。",
    )  # fmt: skip


def grid(market: str, bench: str) -> list[Strategy]:
    out: list[Strategy] = []
    out += [rsi2_dip(market, r, k) for r, k in itertools.product([5, 10], [2.0, 3.0])]
    if market == "US":
        out += [
            spike_fade(market, k, t, c)
            for k, t, c in itertools.product([1.0, 2.0], [0.15, 0.3], [False, True])
        ]
    out += [
        squeeze_break(market, v, c, r, rr)
        for v, c, r, rr in itertools.product([1.5, 2.5], [0.8, 1.0], [0.7, 0.9], [2.0, 3.0])
    ]
    out += [momentum(market, p) for p in [0.9, 0.97]]
    out += [gap_go(market, g, rr) for g, rr in itertools.product([0.04, 0.08], [2.0, 3.0])]
    out += [pullback_rs(market, r, rr) for r, rr in itertools.product([0.7, 0.85], [2.0, 3.0])]
    out += [index_trend(market, bench)]
    return out


def family(key: str) -> str:
    return key.split("|")[0]


def describe(s: Strategy) -> dict[str, Any]:
    return {
        "key": s.key,
        "label": s.label,
        "horizon": s.horizon,
        "side": s.side,
        "max_hold": s.max_hold,
        "params": s.params,
        "why": s.why,
    }


def with_regime(s: Strategy, name: str, cond: Any, desc: str) -> Strategy:
    """Gate a strategy's entries by a market-regime condition (columns joined from mw.strategy.regime)."""
    base = s.entry
    return Strategy(
        key=f"{s.key}|rg:{name}", label=s.label, horizon=s.horizon, side=s.side,
        entry=lambda: base() & cond(), stop=s.stop, target=s.target, max_hold=s.max_hold,
        limit=s.limit, exit_rule=s.exit_rule, rank=s.rank, params={**s.params, "regime": name},
        why=s.why + f"（市場環境の条件：{desc}）",
    )  # fmt: skip


def by_key(market: str, bench: str, key: str) -> Strategy:
    """Rebuild a strategy from its key, e.g. 'pullback_rs|rs0.7|R2.0|rg:topix_up50&us_hy_tightening'."""
    from mw.strategy import regime

    base_key, _, rg = key.partition("|rg:")
    found = {
        s.key: s for s in grid(market, bench) + panic_grid(market, bench) + weak_grid(market, bench)
    }  # panic / weak_dip need regime columns
    if base_key not in found:
        raise KeyError(f"unknown strategy {base_key}")
    s = found[base_key]
    if not rg:
        return s
    filters = regime.FILTERS_US if market == "US" else regime.FILTERS_JP
    parts = rg.split("&")
    conds = [filters[p][0] for p in parts]
    desc = "、".join(filters[p][1] for p in parts)

    def cond() -> pl.Expr:
        e = conds[0]()
        for c in conds[1:]:
            e = e & c()
        return e.fill_null(False)

    return with_regime(s, rg, cond, desc)
