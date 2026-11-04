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
    hh252, ll252 = df["hh252"].to_numpy().astype(float), df["ll252"].to_numpy().astype(float)
    ma200_20 = np.r_[np.full(20, np.nan), ma[200][:-20]]
    tt = ((c > ma[150]) & (c > ma[200]) & (ma[150] > ma[200]) & (ma[200] > ma200_20) & (ma[50] > ma[150])
          & (c > ma[50]) & (c >= 1.3 * ll252) & (c >= 0.75 * hh252))  # fmt: skip
    pvL = pivots(
        h, lo, atr, k_atr=3.0, pct=0.07
    )  # pattern-scale swings (chart patterns span weeks)
    confL = np.searchsorted(pvL.confirmed, np.arange(n), side="right")
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
        if up_trend[t] and t - last_flat > 20:
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
                            valid=20,
                            info={"len": L},
                        )
                    )
                    last_flat = t
                    break
        kL = confL[t]
        # ---- cup with handle (O'Neil): a prior advance of 30%+ into a high H0 near the 52-week high, a
        #      U-shaped cup 12-40% deep lasting 7+ weeks, a right side back near H0, then a 1-6 week handle
        #      in the upper half of the cup that pulls back 3-12% (and less than half the cup).
        if kL >= 2 and up_trend[t] and pvL.kind[kL - 1] == -1 and pvL.kind[kL - 2] == 1:
            i0, h0, ic, lcup = (
                pvL.idx[kL - 2],
                pvL.price[kL - 2],
                pvL.idx[kL - 1],
                pvL.price[kL - 1],
            )
            i1 = ic + int(np.argmax(h[ic : t + 1]))
            h1 = h[i1]
            ihl = i1 + int(np.argmin(lo[i1 : t + 1]))
            hl = lo[ihl]
            depth = 1 - lcup / h0
            prior = h0 / lo[max(0, i0 - 250) : i0 + 1].min() - 1
            bottom = (
                c[i0 : i1 + 1] <= lcup + (h0 - lcup) / 3
            ).mean()  # time near the bottom: U, not V
            if (0.12 <= depth <= 0.40 and prior >= 0.30 and h0 >= 0.9 * hh252[i0] and 35 <= i1 - i0 <= 250
                    and 0.92 * h0 <= h1 <= 1.05 * h0 and 5 <= t - i1 <= 30
                    and 0.03 <= 1 - hl / h1 <= min(0.12, depth / 2) and hl > (h0 + lcup) / 2
                    and bottom >= 0.2 and hl < c[t] < h1):  # fmt: skip
                arm(
                    Armed("cup_handle", t, h1 * 1.001, hl * 0.995, h1 + (h0 - lcup), valid=10,
                          info={"h0": int(i0), "low": int(ic), "h1": int(i1), "hl": int(ihl)}),
                    ("cup", int(i0), int(i1)),
                )  # fmt: skip
        # ---- VCP (Minervini): in a stage-2 trend template, a base from the highest high of up to 200
        #      bars; 2+ pullbacks inside it, each clearly shallower (<= 80%) than the one before, with
        #      higher lows, the first 8-40% and the last <= 10%, volume drying up. Pivot = last high.
        if tt[t]:
            w0 = max(0, t - 200)
            i0 = w0 + int(np.argmax(h[w0 : t + 1]))
            if t - i0 >= 20 and h[i0] >= 0.95 * hh252[t]:
                sel = [j for j in range(k) if pv.idx[j] >= i0]
                if sel and pv.kind[sel[0]] == 1 and pv.kind[sel[-1]] == -1:
                    hi_p = [pv.price[j] for j in sel if pv.kind[j] == 1]
                    lo_p = [pv.price[j] for j in sel if pv.kind[j] == -1]
                    pulls = [1 - b / a for a, b in zip(hi_p, lo_p, strict=False)]
                    ok = (len(pulls) >= 2 and 0.08 <= pulls[0] <= 0.40 and pulls[-1] <= 0.10
                          and all(b <= 0.8 * a for a, b in zip(pulls, pulls[1:], strict=False))
                          and all(b > a for a, b in zip(lo_p, lo_p[1:], strict=False)))  # fmt: skip
                    last_h, last_l = sel[-2], sel[-1]
                    if (ok and np.nanmean(v[t - 9 : t + 1]) < vol50[t] and c[t] < pv.price[last_h]
                            and t - pv.idx[last_l] <= 15):  # fmt: skip
                        arm(
                            Armed("vcp", t, pv.price[last_h] * 1.001, pv.price[last_l] * 0.995, None, valid=15,
                                  info={"pulls": [round(x, 3) for x in pulls], "h": [int(pv.idx[j]) for j in sel if pv.kind[j] == 1],
                                        "l": [int(pv.idx[j]) for j in sel if pv.kind[j] == -1]}),
                            ("vcp", int(i0), int(pv.idx[last_h])),
                        )  # fmt: skip
        if kL >= 5:
            ps, pp, pk = pvL.idx[:kL], pvL.price[:kL], pvL.kind[:kL]
            # ---- inverse head & shoulders: after a 20%+ decline, three lows on the pattern scale; the head is
            #      the 52-week low, 3%+ below both shoulders; shoulders similar, halves of similar length,
            #      4-50 weeks wide, 10%+ high; the neckline (through the two highs) not steep.
            if list(pk[-5:]) == [-1, 1, -1, 1, -1]:
                l1, h1_, l2, h2_, l3 = pp[-5:]
                i_l1, i_h1, i_l2, i_h2, i_l3 = (int(x) for x in ps[-5:])
                neck_h = (h1_ + h2_) / 2
                height = neck_h - l2
                left, right = i_l2 - i_l1, i_l3 - i_l2
                prior_hi = h[max(0, i_l1 - 150) : i_l1 + 1].max()
                slope = (h2_ - h1_) / max(i_h2 - i_h1, 1)
                neck_t = h2_ + slope * (t + 1 - i_h2)
                if (l2 < 0.97 * min(l1, l3) and abs(l1 - l3) <= 0.5 * height and height / neck_h >= 0.10
                        and 20 <= i_l3 - i_l1 <= 250 and 1 / 3 <= left / max(right, 1) <= 3 and abs(h2_ / h1_ - 1) <= 0.10
                        and prior_hi >= 1.25 * l2 and l2 <= 1.02 * ll252[i_l2] and t - i_l3 <= 25
                        and l3 < c[t] < neck_t):  # fmt: skip
                    arm(
                        Armed("inv_hs", t, neck_t * 1.001, l3 * 0.995, neck_t + (neck_t - l2), valid=20,
                              info={"l1": i_l1, "head": i_l2, "l3": i_l3, "h1": i_h1, "h2": i_h2}),
                        ("ihs", i_l2),
                    )  # fmt: skip
        if kL >= 3:
            ps, pp, pk = pvL.idx[:kL], pvL.price[:kL], pvL.kind[:kL]
            # ---- double bottom: after a 20%+ decline, two lows on the pattern scale within 4%, 3-30 weeks
            #      apart, near the 52-week low, with a 10%+ rally between them. Pivot = the middle high.
            if list(pk[-3:]) == [-1, 1, -1]:
                l1, mid, l2 = pp[-3:]
                i1, im, i2 = (int(x) for x in ps[-3:])
                base = min(l1, l2)
                if (abs(l2 / l1 - 1) <= 0.04 and 15 <= i2 - i1 <= 150 and mid >= 1.10 * base
                        and h[max(0, i1 - 150) : i1 + 1].max() >= 1.25 * l1 and base <= 1.03 * ll252[i2]
                        and t - i2 <= 25 and l2 < c[t] < mid):  # fmt: skip
                    arm(
                        Armed("double_bottom", t, mid * 1.001, base * 0.995, mid + (mid - base), valid=20,
                              info={"l1": i1, "l2": i2, "mid": im}),
                        ("db", i1, i2),
                    )  # fmt: skip
        if k >= 4:
            ps, pp, pk = pv.idx[:k], pv.price[:k], pv.kind[:k]
            # ---- ascending triangle: two highs within 2% (flat top), two rising lows after them
            if pk[-1] == -1 and pk[-2] == 1 and pk[-3] == -1 and pk[-4] == 1:
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
        if (
            (hammer or engulf or star)
            and c[t - 10] > c[t] * 1.08
            and h[max(0, t - 60) : t].max() >= 1.15 * c[t]
        ):
            # support = the 200-day line, or a major (pattern-scale) low 20-250 bars old that price tests
            # from above (low within 2% of it, close above it)
            sup, at = None, ""
            if (
                np.isfinite(ma[200][t])
                and abs(lo[t] / ma[200][t] - 1) <= 0.02
                and c[t] > ma[200][t]
            ):
                sup, at = ma[200][t], "200日線"
            else:
                for j in range(kL - 1, -1, -1):
                    if t - pvL.idx[j] > 250:
                        break
                    if (pvL.kind[j] == -1 and t - pvL.idx[j] > 20 and abs(lo[t] / pvL.price[j] - 1) <= 0.02
                            and c[t] > pvL.price[j]):  # fmt: skip
                        sup, at = pvL.price[j], "前の大きな安値"
                        break
            if sup is not None:
                lo3 = lo[t - 2 : t + 1].min()
                out.append(
                    Armed("support_reversal", t, h[t] * 1.001, lo3 - 0.1 * atr[t], None, valid=3,
                          info={"kind": "star" if star else ("hammer" if hammer else "engulf"), "at": at, "sup": round(float(sup), 4)})
                )  # fmt: skip
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
