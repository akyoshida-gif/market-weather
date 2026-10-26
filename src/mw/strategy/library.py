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
