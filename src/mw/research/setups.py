"""Textbook setups in market context (VALIDATION_PLAN_V3 §1-§3). Everything uses data known at the time.

Building blocks
  pivots():        swing highs / lows (zigzag). A pivot is CONFIRMED only once price has moved away from it
                   by max(2 ATR, 4%); detectors may only use pivots confirmed on or before the current day.
  market_phase():  uptrend / correction / downtrend / bottoming / panic, from the S&P 500 (SPY), its moving
                   averages and the VIX of that day.
  weinstein_stage(): stage 1-4 from the WEEKLY 30-week moving average, as of the last completed week.
  trend_template(): Minervini's trend template on daily moving averages and the 52-week range.

Setups are ARMED at the close of day t-1 with a buy-stop level (pivot), an initial stop and the pattern's
measured-move target; they trigger on day t only if the day's high reaches the level (fill = max(open,
level); skipped if the open is already > 5% above it). This mirrors placing the order in advance.

Each detector documents WHY it should work (rules fixed before testing, VALIDATION_PLAN_V3 §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

C = pl.col
G = "entity_id"

PHASES = {1: "上昇", 2: "調整", 3: "下落", 4: "底入れ", 5: "パニック"}
STAGES = {
    1: "ステージ1（底固め）",
    2: "ステージ2（上昇）",
    3: "ステージ3（天井圏）",
    4: "ステージ4（下落）",
}


# ---------------------------------------------------------------- market phase
def market_phase(spy: pl.DataFrame, vix: pl.DataFrame | None) -> pl.DataFrame:
    """spy: as_of, close. Returns as_of, phase (1..5) known at that day's close."""
    s = (
        spy.sort("as_of")
        .with_columns(
            C("close").rolling_mean(50).alias("ma50"), C("close").rolling_mean(200).alias("ma200")
        )
        .with_columns(
            (C("ma200") / C("ma200").shift(20) - 1).alias("ma200_slope"),
            (C("ma50") / C("ma50").shift(10) - 1).alias("ma50_slope"),
        )
    )
    if vix is not None:
        s = s.join(
            vix.select("as_of", C("close").alias("vix")), on="as_of", how="left"
        ).with_columns(C("vix").forward_fill())
    else:
        s = s.with_columns(pl.lit(None, pl.Float64).alias("vix"))
    c = C("close")
    phase = (
        pl.when(C("vix") >= 30).then(5)
        .when((c > C("ma200")) & (c > C("ma50")) & (C("ma50") > C("ma200"))).then(1)
        .when(c > C("ma200")).then(2)
        .when((c > C("ma50")) | (C("ma50_slope") > 0)).then(4)
        .otherwise(3)
    )  # fmt: skip
    return s.with_columns(phase.cast(pl.Int8).alias("phase")).select("as_of", "phase")


# ---------------------------------------------------------------- weekly stage (Weinstein)
def weinstein_stage(daily: pl.DataFrame) -> pl.DataFrame:
    """daily: entity_id, as_of, close (sorted). Weekly close (last session of each week), 30-week MA and its
    4-week slope. The stage of a week becomes usable from the NEXT session after the week's last session."""
    w = (
        daily.with_columns(C("as_of").dt.truncate("1w").alias("wk"))
        .group_by(G, "wk")
        .agg(C("close").last().alias("wclose"), C("as_of").max().alias("wlast"))
        .sort([G, "wk"])
        .with_columns(C("wclose").rolling_mean(30).over(G).alias("ma30w"))
        .with_columns((C("ma30w") / C("ma30w").shift(4).over(G) - 1).alias("slope"))
    )
    up = (C("slope") > 0.005) & (C("wclose") > C("ma30w"))
    dn = (C("slope") < -0.005) & (C("wclose") < C("ma30w"))
    w = w.with_columns(pl.when(up).then(2).when(dn).then(4).otherwise(None).alias("_trend"))
    # neutral weeks: stage 1 after a decline, stage 3 after an advance (last non-neutral trend)
    w = w.with_columns(C("_trend").forward_fill().over(G).alias("_last"))
    stage = (
        pl.when(C("_trend").is_not_null())
        .then(C("_trend"))
        .when(C("_last") == 4)
        .then(1)
        .when(C("_last") == 2)
        .then(3)
        .otherwise(None)
    )
    w = w.with_columns(stage.cast(pl.Int8).alias("stage")).filter(C("ma30w").is_not_null())
    return w.select(G, C("wlast").alias("as_of_week"), "stage")


def attach_stage(daily: pl.DataFrame, stages: pl.DataFrame) -> pl.DataFrame:
    """As-of join on the last COMPLETED week strictly before each day."""
    s = stages.sort("as_of_week").with_columns(
        (C("as_of_week") + pl.duration(days=1)).alias("as_of")
    )
    return (
        daily.sort("as_of")
        .join_asof(s.select(G, "as_of", "stage"), on="as_of", by=G, strategy="backward")
        .sort([G, "as_of"])
    )


# ---------------------------------------------------------------- trend template (Minervini)
def trend_template() -> pl.Expr:
    c = C("close")
    return (
        (c > C("ma150")) & (c > C("ma200")) & (C("ma150") > C("ma200")) & (C("ma200") > C("ma200").shift(20).over(G))
        & (C("ma50") > C("ma150")) & (c > C("ma50")) & (c >= 1.3 * C("ll252")) & (c >= 0.75 * C("hh252"))
    ).fill_null(False)  # fmt: skip


def daily_indicators(df: pl.DataFrame) -> pl.DataFrame:
    c, h, lo, o, v = C("close"), C("high"), C("low"), C("open"), C("volume")
    pc = c.shift(1).over(G)
    tr = pl.max_horizontal(h - lo, (h - pc).abs(), (lo - pc).abs())
    return df.sort([G, "as_of"]).with_columns(
        pc.alias("pc"),
        tr.rolling_mean(14).over(G).alias("atr"),
        *[c.rolling_mean(n).over(G).alias(f"ma{n}") for n in (10, 20, 50, 150, 200)],
        v.rolling_mean(50).shift(1).over(G).alias("vol50"),
        h.rolling_max(252).over(G).alias("hh252"),
        lo.rolling_min(252).over(G).alias("ll252"),
        (c * v).rolling_mean(20).over(G).alias("dvol20"),
        (c / c.shift(126).over(G) - 1).alias("ret126"),
        (c - o).abs().alias("body"), (h - lo).alias("rng"),
        (pl.min_horizontal(o, c) - lo).alias("lsh"), (h - pl.max_horizontal(o, c)).alias("ush"),
    )  # fmt: skip


# ---------------------------------------------------------------- pivots
@dataclass
class Pivots:
    idx: np.ndarray  # bar index of the pivot (the extreme)
    price: np.ndarray
    kind: np.ndarray  # +1 high, -1 low
    confirmed: np.ndarray  # bar index on which it became known


def pivots(
    high: np.ndarray, low: np.ndarray, atr: np.ndarray, k_atr: float = 2.0, pct: float = 0.04
) -> Pivots:
    """Zigzag on highs/lows. The current extreme becomes a pivot when price moves away from it by
    max(k_atr * ATR at the extreme, pct of its price); the confirmation bar is that moving-away bar."""
    n = len(high)
    out_i, out_p, out_k, out_c = [], [], [], []
    if n < 3:
        return Pivots(np.array([], int), np.array([]), np.array([], int), np.array([], int))
    trend = 0
    ext_i, ext_p = 0, high[0]
    lo_i, lo_p = 0, low[0]
    for t in range(1, n):
        a = atr[t] if np.isfinite(atr[t]) else 0.0
        if trend >= 0:  # looking for a high
            if high[t] >= ext_p:
                ext_i, ext_p = t, high[t]
            th = max(k_atr * (atr[ext_i] if np.isfinite(atr[ext_i]) else a), pct * ext_p)
            if trend == 1 and ext_p - low[t] >= th:
                out_i.append(ext_i), out_p.append(ext_p), out_k.append(1), out_c.append(t)
                trend, lo_i, lo_p = -1, t, low[t]
                continue
        if trend <= 0:  # looking for a low
            if low[t] <= lo_p:
                lo_i, lo_p = t, low[t]
            th = max(k_atr * (atr[lo_i] if np.isfinite(atr[lo_i]) else a), pct * lo_p)
            if trend == -1 and high[t] - lo_p >= th:
                out_i.append(lo_i), out_p.append(lo_p), out_k.append(-1), out_c.append(t)
                trend, ext_i, ext_p = 1, t, high[t]
                continue
        if trend == 0:  # start: whichever side moves first
            if high[t] - lo_p >= max(k_atr * a, pct * lo_p):
                trend, ext_i, ext_p = 1, t, high[t]
            elif ext_p - low[t] >= max(k_atr * a, pct * ext_p):
                trend, lo_i, lo_p = -1, t, low[t]
    return Pivots(
        np.array(out_i, int), np.array(out_p, float), np.array(out_k, int), np.array(out_c, int)
    )


# ---------------------------------------------------------------- setups
@dataclass
class Armed:
    """A setup known at the close of `t` (bar index): buy-stop `level` valid for the next `valid` bars."""

    setup: str
    t: int
    level: float
    stop: float
    target: float | None
    valid: int = 5
    info: dict[str, Any] = field(default_factory=dict)


SETUPS = {
    "flat_base": ("平らなベースの上抜け", "上昇局面の銘柄が値幅15%以内で5〜13週もみ合う＝売りたい人が売り切った。上限を抜けると売り物が薄く上がりやすい"),
    "cup_handle": ("カップウィズハンドル", "高値から12〜35%下げて元の高値近くまで戻り、浅い押し（ハンドル）で残りの売りを消化。ハンドルの高値抜けで上昇再開"),
    "vcp": ("値幅の縮小（VCP）", "押しの深さが回を追って小さくなる＝売りが細っている。最後の小さな押しの高値抜けは、売り物が少なく伸びやすい"),
    "inv_hs": ("逆三尊のネックライン抜け", "下落の後、安値を更新できなくなり（右肩が頭より高い）、売り方の負け。ネックライン抜けで流れが上に変わる"),
    "double_bottom": ("ダブルボトム", "同じ水準で2回下げ止まった＝そこに強い買いがある。間の戻り高値を抜けると底入れの確認"),
    "asc_triangle": ("上昇三角形", "上値は同じ水準で抑えられ、下値は切り上がる＝買いが強まっている。水平な上値を抜けると放れやすい"),
    "bull_flag": ("フラッグ", "急騰（旗竿）の後、浅く短い押しで利益確定をこなした。押しの高値を抜けると、旗竿と同じくらいの上昇が見込まれる"),
    "ma_pullback": ("上昇中の移動平均線への押し目＋反転足", "上昇トレンドの銘柄が20日・50日線まで押し、ハンマーや包み足で切り返した＝押し目買いが入っている"),
    "support_reversal": ("下落後の支持線での反転足", "下げ続けた銘柄が、200日線や前の安値の近くで明けの明星・ハンマー・包み足を出した＝売りの勢いが尽きた"),
}  # fmt: skip


def detect(df: pl.DataFrame) -> list[Armed]:
    """df: ONE stock, sorted by date, with daily_indicators() columns. Returns armed setups (bar indexes)."""
    o, h, lo, c, v = (
        df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close", "volume")
    )
    atr = df["atr"].to_numpy().astype(float)
    ma = {n: df[f"ma{n}"].to_numpy().astype(float) for n in (10, 20, 50, 150, 200)}
    vol50 = df["vol50"].to_numpy().astype(float)
    body, rng, lsh, ush = (df[k].to_numpy().astype(float) for k in ("body", "rng", "lsh", "ush"))
    n = len(c)
    pv = pivots(h, lo, atr)
    out: list[Armed] = []
    if n < 260:
        return out
    up_trend = (c > ma[50]) & (ma[50] > ma[200]) & np.isfinite(ma[200])
    # pivots known at each bar: positions in the pivot list confirmed up to t
    conf_pos = np.searchsorted(pv.confirmed, np.arange(n), side="right")
    last_flat = -999
    seen: set[tuple[Any, ...]] = set()

    def arm(a: Armed, key: tuple[Any, ...]) -> None:
        """One record per pattern instance (the day it first becomes tradable); it stays armed `valid` bars."""
        if key not in seen:
            seen.add(key)
            out.append(a)

    for t in range(252, n - 1):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        k = conf_pos[t]  # pivots[0:k] are known at t
        # ---- flat base: 25-65 bars, range <= 15% of the top, prior advance into it, price near the top
        if up_trend[t] and t - last_flat > 10:
            for L in (65, 45, 25):
                top = h[t - L + 1 : t + 1].max()
                bot = lo[t - L + 1 : t + 1].min()
                start = t - L + 1
                prior = (
                    c[start] / max(lo[max(0, start - 120) : start + 1].min(), 1e-9) - 1
                )  # advance into the base
                near_high = top >= 0.85 * np.nanmax(h[max(0, t - 251) : t + 1])
                if (top > 0 and (top - bot) / top <= 0.15 and c[t] >= top * 0.95 and c[t - L] < top * 0.95
                        and prior >= 0.30 and near_high):  # fmt: skip
                    out.append(
                        Armed(
                            "flat_base",
                            t,
                            top * 1.001,
                            max(bot, top * 0.92),
                            top + (top - bot),
                            info={"len": L},
                        )
                    )
                    last_flat = t
                    break
        if k >= 3:
            ps, pp, pk = pv.idx[:k], pv.price[:k], pv.kind[:k]
            # ---- cup with handle: high H0, low L (12-35% below), high H1 >= 0.95*H0, then handle low in upper half
            if pk[-1] == -1 and k >= 4 and pk[-2] == 1 and pk[-3] == -1 and pk[-4] == 1:
                h0, lcup, h1, hl = pp[-4], pp[-3], pp[-2], pp[-1]
                depth = 1 - lcup / h0
                handle = 1 - hl / h1
                if (0.12 <= depth <= 0.35 and h1 >= 0.95 * h0 and ps[-2] - ps[-4] >= 30 and 0.03 <= handle <= 0.15
                        and hl > (h0 + lcup) / 2 and t - ps[-1] <= 15 and c[t] < h1 and up_trend[t]):  # fmt: skip
                    arm(
                        Armed(
                            "cup_handle",
                            t,
                            h1 * 1.001,
                            hl * 0.995,
                            h1 + (h0 - lcup),
                            valid=15,
                            info={
                                "h0": int(ps[-4]),
                                "low": int(ps[-3]),
                                "h1": int(ps[-2]),
                                "hl": int(ps[-1]),
                            },
                        ),
                        ("cup", int(ps[-2])),
                    )
            # ---- VCP: last three pullbacks (high->low) each shallower; last <= 10%; volume drying up
            if k >= 6 and pk[-1] == -1:
                pulls = []
                for j in (-5, -3, -1):
                    if pk[j - 1] == 1 and pk[j] == -1:
                        pulls.append(1 - pp[j] / pp[j - 1])
                if (len(pulls) == 3 and pulls[0] > pulls[1] > pulls[2] and pulls[2] <= 0.10 and up_trend[t]
                        and np.nanmean(v[t - 9 : t + 1]) < vol50[t] and c[t] < pp[-2] and t - ps[-1] <= 15):  # fmt: skip
                    arm(
                        Armed(
                            "vcp",
                            t,
                            pp[-2] * 1.001,
                            pp[-1] * 0.995,
                            pp[-2] * 1.2,
                            valid=15,
                            info={"pulls": [round(x, 3) for x in pulls], "h": int(ps[-2])},
                        ),
                        ("vcp", int(ps[-2])),
                    )
            # ---- inverse head & shoulders: L1 H1 L2(head) H2 L3, after a decline, shoulders similar
            if (
                k >= 5
                and pk[-1] == -1
                and pk[-2] == 1
                and pk[-3] == -1
                and pk[-4] == 1
                and pk[-5] == -1
            ):
                l1, h1_, l2, h2_, l3 = pp[-5], pp[-4], pp[-3], pp[-2], pp[-1]
                if (
                    l2 < l1
                    and l2 < l3
                    and abs(l1 - l3) <= 0.5 * (min(h1_, h2_) - l2)
                    and ps[-5] >= 60
                ):
                    prior_drop = (
                        c[ps[-5] - 60] > l1 * 1.15
                    )  # a real decline before the left shoulder
                    slope = (h2_ - h1_) / max(ps[-2] - ps[-4], 1)
                    neck_t = h2_ + slope * (t + 1 - ps[-2])
                    if prior_drop and c[t] < neck_t and t - ps[-1] <= 20 and neck_t > l3:
                        arm(
                            Armed(
                                "inv_hs",
                                t,
                                neck_t * 1.001,
                                l3 * 0.995,
                                neck_t + (neck_t - l2),
                                valid=20,
                                info={
                                    "l1": int(ps[-5]),
                                    "head": int(ps[-3]),
                                    "l3": int(ps[-1]),
                                    "h1": int(ps[-4]),
                                    "h2": int(ps[-2]),
                                },
                            ),
                            ("ihs", int(ps[-3])),
                        )
            # ---- double bottom: L1 H L2 with L2 within 3% of L1, >= 15 bars apart, after a decline
            if k >= 3 and pk[-1] == -1 and pk[-2] == 1 and pk[-3] == -1:
                l1, mid, l2 = pp[-3], pp[-2], pp[-1]
                if (
                    abs(l2 / l1 - 1) <= 0.03
                    and ps[-1] - ps[-3] >= 15
                    and mid >= 1.08 * l1
                    and ps[-3] >= 60
                    and c[ps[-3] - 60] > l1 * 1.15
                    and c[t] < mid
                    and t - ps[-1] <= 20
                ):
                    arm(
                        Armed(
                            "double_bottom",
                            t,
                            mid * 1.001,
                            min(l1, l2) * 0.995,
                            mid + (mid - min(l1, l2)),
                            valid=20,
                            info={"l1": int(ps[-3]), "l2": int(ps[-1]), "mid": int(ps[-2])},
                        ),
                        ("db", int(ps[-3]), int(ps[-1])),
                    )
            # ---- ascending triangle: two highs within 2% (flat top), two rising lows after them
            if k >= 4 and pk[-1] == -1 and pk[-2] == 1 and pk[-3] == -1 and pk[-4] == 1:
                ha, la, hb, lb = pp[-4], pp[-3], pp[-2], pp[-1]
                if (
                    abs(hb / ha - 1) <= 0.02
                    and lb > la * 1.02
                    and c[t] < max(ha, hb)
                    and t - ps[-1] <= 15
                    and c[t] > ma[50][t]
                ):
                    top = max(ha, hb)
                    arm(
                        Armed(
                            "asc_triangle",
                            t,
                            top * 1.001,
                            lb * 0.995,
                            top + (top - la),
                            valid=15,
                            info={"h": [int(ps[-4]), int(ps[-2])], "l": [int(ps[-3]), int(ps[-1])]},
                        ),
                        ("asc", int(ps[-2])),
                    )
        # ---- bull flag: +20% in <= 15 bars (pole), then 3-15 bars holding >= half of the pole, depth <= 15%
        for fl in range(3, 16):
            top_i = t - fl
            if top_i - 15 < 0:
                break
            pole_lo = lo[top_i - 15 : top_i + 1].min()
            pole_top = h[top_i]
            if pole_top == h[top_i - 2 : t + 1].max() and pole_top >= 1.2 * pole_lo:
                flag_lo = lo[top_i + 1 : t + 1].min()
                if (
                    flag_lo >= pole_top - 0.5 * (pole_top - pole_lo)
                    and flag_lo >= 0.85 * pole_top
                    and c[t] < pole_top
                ):
                    arm(
                        Armed(
                            "bull_flag",
                            t,
                            pole_top * 1.001,
                            flag_lo * 0.995,
                            pole_top + (pole_top - pole_lo),
                            valid=10,
                            info={"pole": int(top_i)},
                        ),
                        ("flag", int(top_i)),
                    )
                break
        # ---- pullback to the 20/50-day line with a reversal candle, in an uptrend
        hammer = lsh[t] >= 2 * body[t] and ush[t] <= body[t] and rng[t] > 0
        engulf = c[t] > o[t] and c[t - 1] < o[t - 1] and o[t] <= c[t - 1] and c[t] >= o[t - 1]
        if up_trend[t] and (hammer or engulf):
            for m in (20, 50):
                if lo[t] <= ma[m][t] + 0.3 * atr[t] and c[t] > ma[m][t]:
                    out.append(
                        Armed(
                            "ma_pullback",
                            t,
                            h[t] * 1.001,
                            lo[t] - 0.1 * atr[t],
                            None,
                            valid=3,
                            info={"ma": m, "hammer": bool(hammer)},
                        )
                    )
                    break
        # ---- reversal candle at support after a decline (morning star / hammer / engulfing)
        star = (c[t - 2] < o[t - 2] and body[t - 1] <= 0.3 * max(rng[t - 1], 1e-9) and c[t] > o[t]
                and c[t] > (o[t - 2] + c[t - 2]) / 2)  # fmt: skip
        if (hammer or engulf or star) and c[t - 10] > c[t] * 1.08:
            near200 = abs(lo[t] / ma[200][t] - 1) <= 0.02 if np.isfinite(ma[200][t]) else False
            near_low = k >= 1 and any(
                pv.kind[:k][j] == -1
                and abs(lo[t] / pv.price[:k][j] - 1) <= 0.02
                and t - pv.idx[:k][j] > 20
                for j in range(max(0, k - 6), k)
            )
            if near200 or near_low:
                lo3 = lo[t - 2 : t + 1].min()
                out.append(
                    Armed(
                        "support_reversal",
                        t,
                        h[t] * 1.001,
                        lo3 - 0.1 * atr[t],
                        None,
                        valid=3,
                        info={
                            "kind": "star" if star else ("hammer" if hammer else "engulf"),
                            "at": "200日線" if near200 else "前の安値",
                        },
                    )
                )
    return out


def pivot_frame(df: pl.DataFrame) -> pl.DataFrame:
    """For charts: the pivots of one stock (index, price, kind, confirmed index)."""
    pv = pivots(
        df["high"].to_numpy().astype(float),
        df["low"].to_numpy().astype(float),
        df["atr"].to_numpy().astype(float),
    )
    return pl.DataFrame(
        {"idx": pv.idx, "price": pv.price, "kind": pv.kind, "confirmed": pv.confirmed}
    )
