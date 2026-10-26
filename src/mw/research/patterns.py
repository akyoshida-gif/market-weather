"""Pattern mining: technical conditions x trade templates (ADR-0014).

1. ~50 boolean technical conditions per (stock, day), computed only from data up to that close:
   trend / moving-average order, Ichimoku, MACD, RSI, stochastics, Bollinger, Fibonacci retracement,
   candlestick shapes, volume, relative strength, market regime.
2. Trade templates (fixed BEFORE looking at results): entry = next open, stop / target in ATR
   multiples, time stop. Outcome per row is computed once with first-touch logic (stop first if
   both are touched in one bar), costs deducted, result also in R (risk units).
3. A pattern = AND of 1-3 conditions; an event = first day the pattern becomes true (onset), so a
   condition that stays true for 10 days counts once.
4. Periods: train 2010-2017, valid 2018-2021, test 2022- (never used for selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

C = pl.col
TRAIN_END = date(2017, 12, 31)
VALID_END = date(2021, 12, 31)


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    horizon: int
    sl_atr: float
    tp_atr: float


TEMPLATES = [
    Template("S5", "超短期（最長5日・損切り1ATR・利確1.5ATR）", 5, 1.0, 1.5),
    Template("S10", "短期（最長10日・損切り1.5ATR・利確3ATR）", 10, 1.5, 3.0),
    Template("M20", "中期（最長20日・損切り2ATR・利確4ATR）", 20, 2.0, 4.0),
    Template("L60", "長め（最長60日・損切り3ATR・利確9ATR）", 60, 3.0, 9.0),
]

# condition key -> (label for the report)
LABELS: dict[str, str] = {}


def _ema(col: str, n: int) -> pl.Expr:
    return C(col).ewm_mean(span=n, adjust=False).over("entity_id")


def conditions(df: pl.DataFrame, market: str) -> pl.DataFrame:
    """df: entity_id, as_of, open, high, low, close, volume (+ bench rows). Adds c_* boolean columns."""
    g = "entity_id"
    df = df.sort([g, "as_of"])
    o, h, lo, c, v = C("open"), C("high"), C("low"), C("close"), C("volume")
    pc = c.shift(1).over(g)
    tr = pl.max_horizontal(h - lo, (h - pc).abs(), (lo - pc).abs())
    df = df.with_columns(
        pc.alias("pc"), o.shift(1).over(g).alias("po"), h.shift(1).over(g).alias("ph"), lo.shift(1).over(g).alias("pl_"),
        tr.alias("tr"),
        *[c.rolling_mean(n).over(g).alias(f"ma{n}") for n in (5, 20, 25, 50, 75, 200)],
        v.rolling_mean(20).shift(1).over(g).alias("vol20"),
        (c * v).rolling_mean(20).over(g).alias("dvol20"),
        h.rolling_max(9).over(g).alias("hh9"), lo.rolling_min(9).over(g).alias("ll9"),
        h.rolling_max(26).over(g).alias("hh26"), lo.rolling_min(26).over(g).alias("ll26"),
        h.rolling_max(52).over(g).alias("hh52"), lo.rolling_min(52).over(g).alias("ll52"),
        h.rolling_max(60).over(g).alias("hh60"), lo.rolling_min(60).over(g).alias("ll60"),
        h.rolling_max(252).shift(1).over(g).alias("hh252p"),
        h.rolling_max(60).shift(1).over(g).alias("hh60p"),
        c.rolling_std(20).over(g).alias("sd20"),
        h.rolling_max(14).over(g).alias("hh14"), lo.rolling_min(14).over(g).alias("ll14"),
        (c / c.shift(63).over(g) - 1).alias("ret63"),
        (c.diff().over(g)).alias("chg"),
        _ema("close", 12).alias("ema12"), _ema("close", 26).alias("ema26"),
    )  # fmt: skip
    up, dn = (
        pl.when(C("chg") > 0).then(C("chg")).otherwise(0.0),
        pl.when(C("chg") < 0).then(-C("chg")).otherwise(0.0),
    )
    df = df.with_columns(
        C("tr").rolling_mean(14).over(g).alias("atr"),
        (100 - 100 / (1 + up.ewm_mean(alpha=1 / 14, adjust=False).over(g) / dn.ewm_mean(alpha=1 / 14, adjust=False).over(g).clip(lower_bound=1e-9))).alias("rsi"),
        (100 - 100 / (1 + up.rolling_mean(2).over(g) / dn.rolling_mean(2).over(g).clip(lower_bound=1e-9))).alias("rsi2"),
        (C("ema12") - C("ema26")).alias("macd"),
        ((C("hh9") + C("ll9")) / 2).alias("tenkan"),
        ((C("hh26") + C("ll26")) / 2).alias("kijun"),
        ((c - C("ll14")) / (C("hh14") - C("ll14")).clip(lower_bound=1e-9) * 100).alias("stk"),
        (C("ma20") + 2 * C("sd20")).alias("bb_up"), (C("ma20") - 2 * C("sd20")).alias("bb_lo"),
        (4 * C("sd20") / C("ma20")).alias("bbw"),
        (v / C("vol20")).alias("vr"),
        (c - o).abs().alias("body"), (h - lo).alias("rng"),
        (pl.min_horizontal(o, c) - lo).alias("lsh"), (h - pl.max_horizontal(o, c)).alias("ush"),
    )  # fmt: skip
    df = df.with_columns(
        C("macd").ewm_mean(span=9, adjust=False).over(g).alias("macd_sig"),
        C("stk").rolling_mean(3).over(g).alias("stk_d"),
        ((C("tenkan") + C("kijun")) / 2).shift(26).over(g).alias("span_a"),
        ((C("hh52") + C("ll52")) / 2).shift(26).over(g).alias("span_b"),
        C("bbw").rolling_quantile(0.2, window_size=120).over(g).alias("bbw_q20"),
        (C("chg").sign() * v).cum_sum().over(g).alias("obv"),
    )
    df = df.with_columns(
        pl.max_horizontal("span_a", "span_b").alias("cloud_top"),
        pl.min_horizontal("span_a", "span_b").alias("cloud_bot"),
        (C("obv") - C("obv").shift(20).over(g)).alias("obv20"),
    )
    prev = lambda e: e.shift(1).over(g)  # noqa: E731
    cross_up = lambda a, b: (C(a) > C(b)) & (prev(C(a)) <= prev(C(b)))  # noqa: E731
    green, red = c > o, c < o
    fib_r = (C("hh60") - c) / (C("hh60") - C("ll60")).clip(lower_bound=1e-9)
    conds = {
        # --- trend / moving averages
        "po_short": ((C("ma5") > C("ma25")) & (C("ma25") > C("ma75"))) if market == "JP" else ((C("ma5") > C("ma20")) & (C("ma20") > C("ma50"))),
        "po_long": (C("ma20") > C("ma50")) & (C("ma50") > C("ma200")) & (c > C("ma20")),
        "above_ma200": c > C("ma200"),
        "below_ma200": c < C("ma200"),
        "ma50_rising": C("ma50") > prev(C("ma50")),
        "gc_5_25": cross_up("ma5", "ma25") if market == "JP" else cross_up("ma20", "ma50"),
        "gc_50_200": cross_up("ma50", "ma200"),
        "reclaim_ma50": cross_up("close", "ma50"),
        # --- Ichimoku
        "ichi_above_cloud": c > C("cloud_top"),
        "ichi_cloud_break": cross_up("close", "cloud_top"),
        "ichi_tk_cross": cross_up("tenkan", "kijun"),
        "ichi_chikou_up": c > c.shift(26).over(g),
        "ichi_sanyaku": (c > C("cloud_top")) & (C("tenkan") > C("kijun")) & (c > c.shift(26).over(g)),
        # --- MACD
        "macd_gc": cross_up("macd", "macd_sig"),
        "macd_gc_below0": cross_up("macd", "macd_sig") & (C("macd") < 0),
        "macd_above0": C("macd") > 0,
        "macd_hist_up": (C("macd") - C("macd_sig")) > prev(C("macd") - C("macd_sig")),
        # --- RSI / stochastics
        "rsi_lt30": C("rsi") < 30,
        "rsi_cross30": (C("rsi") > 30) & (prev(C("rsi")) <= 30),
        "rsi_cross50": (C("rsi") > 50) & (prev(C("rsi")) <= 50),
        "rsi_gt70": C("rsi") > 70,
        "rsi2_lt10": C("rsi2") < 10,
        "stoch_gc_low": cross_up("stk", "stk_d") & (C("stk") < 25),
        # --- Bollinger
        "bb_below_lower": c < C("bb_lo"),
        "bb_reenter_lower": (c > C("bb_lo")) & (prev(c) <= prev(C("bb_lo"))),
        "bb_squeeze": C("bbw") <= C("bbw_q20"),
        "bb_break_upper": cross_up("close", "bb_up"),
        # --- Fibonacci retracement of the 60-day swing (only when the 60-day trend is up)
        "fib_382": (fib_r >= 0.30) & (fib_r <= 0.45) & (C("ma50") > prev(C("ma50"))),
        "fib_50": (fib_r > 0.45) & (fib_r <= 0.55) & (C("ma50") > prev(C("ma50"))),
        "fib_618": (fib_r > 0.55) & (fib_r <= 0.70) & (C("ma50") > prev(C("ma50"))),
        # --- candlesticks
        "cdl_bull_engulf": green & prev(red) & (o <= prev(c)) & (c >= prev(o)),
        "cdl_hammer": (C("lsh") >= 2 * C("body")) & (C("ush") <= C("body")) & ((c - lo) / C("rng").clip(lower_bound=1e-9) >= 0.6),
        "cdl_big_green": green & (C("body") >= 0.7 * C("rng")) & (C("rng") >= 1.5 * C("atr")),
        "cdl_doji": C("body") <= 0.1 * C("rng"),
        "cdl_inside": (h < prev(h)) & (lo > prev(lo)),
        "cdl_gap_up": o > prev(h),
        "cdl_3_white": green & prev(green) & prev(prev(green)) & (c > prev(c)) & (prev(c) > prev(prev(c))),
        "cdl_morning_star": green & prev(C("body") <= 0.3 * C("rng")) & prev(prev(red)) & (c > (prev(prev(o)) + prev(prev(c))) / 2),
        # --- volume
        "vol_x2": C("vr") >= 2,
        "vol_x3": C("vr") >= 3,
        "vol_dry": C("vr") <= 0.6,
        "obv_up": C("obv20") > 0,
        # --- strength / location
        "new_high_252": c > C("hh252p"),
        "near_high_252": c >= 0.95 * C("hh252p"),
        "break_high_60": c > C("hh60p"),
        "pullback_5pct": (c <= 0.95 * C("hh60")) & (c >= 0.85 * C("hh60")),
        "down_3days": (c < prev(c)) & (prev(c) < prev(prev(c))) & (prev(prev(c)) < prev(prev(prev(c)))),
    }  # fmt: skip
    df = df.with_columns([e.fill_null(False).alias("c_" + k) for k, e in conds.items()])
    return df


COND_LABELS = {
    "po_short": "短期パーフェクトオーダー（米:5>20>50日線／日:5>25>75日線）",
    "po_long": "長期パーフェクトオーダー（20>50>200日線、終値が20日線の上）",
    "above_ma200": "200日線の上", "below_ma200": "200日線の下", "ma50_rising": "50日線が上向き",
    "gc_5_25": "短期ゴールデンクロス（米:20/50日線、日:5/25日線）", "gc_50_200": "50/200日線ゴールデンクロス",
    "reclaim_ma50": "50日線を上抜け",
    "ichi_above_cloud": "一目：雲の上", "ichi_cloud_break": "一目：雲を上抜け", "ichi_tk_cross": "一目：転換線が基準線を上抜け",
    "ichi_chikou_up": "一目：遅行スパンが26日前の株価の上", "ichi_sanyaku": "一目：三役好転",
    "macd_gc": "MACDゴールデンクロス", "macd_gc_below0": "MACDゴールデンクロス（0より下）", "macd_above0": "MACDが0より上",
    "macd_hist_up": "MACDヒストグラムが増加",
    "rsi_lt30": "RSI(14) 30未満", "rsi_cross30": "RSI(14) が30を上抜け", "rsi_cross50": "RSI(14) が50を上抜け",
    "rsi_gt70": "RSI(14) 70超", "rsi2_lt10": "RSI(2) 10未満",
    "stoch_gc_low": "ストキャスティクス低位のゴールデンクロス",
    "bb_below_lower": "ボリンジャー−2σ割れ", "bb_reenter_lower": "ボリンジャー−2σから内側へ戻る",
    "bb_squeeze": "ボリンジャーのスクイーズ（幅が過去120日の下位20%）", "bb_break_upper": "ボリンジャー+2σを上抜け",
    "fib_382": "フィボナッチ38.2%押し（上昇中）", "fib_50": "フィボナッチ50%押し（上昇中）", "fib_618": "フィボナッチ61.8%押し（上昇中）",
    "cdl_bull_engulf": "陽の包み足", "cdl_hammer": "下ヒゲの長い陽線（ハンマー）", "cdl_big_green": "大陽線（1.5ATR以上）",
    "cdl_doji": "十字線", "cdl_inside": "はらみ足（前日の値幅の内側）", "cdl_gap_up": "窓開け上昇（前日高値より上で寄り付き）",
    "cdl_3_white": "赤三兵（3日連続の陽線で上昇）", "cdl_morning_star": "明けの明星",
    "vol_x2": "出来高20日平均の2倍以上", "vol_x3": "出来高20日平均の3倍以上", "vol_dry": "出来高20日平均の0.6倍以下（閑散）",
    "obv_up": "OBVが20日前より増加",
    "new_high_252": "52週高値更新", "near_high_252": "52週高値から5%以内", "break_high_60": "60日高値を上抜け",
    "pullback_5pct": "60日高値から5〜15%下落した位置", "down_3days": "3日続落",
}  # fmt: skip


def outcomes(df: pl.DataFrame, t: Template, market: str) -> tuple[np.ndarray, np.ndarray]:
    """Per row: (net return, R multiple) of the template trade entered at the next open. NaN if
    not evaluable (end of data, gap through stop, missing ATR)."""
    ent = df["entity_id"].to_numpy()
    o, h, lo, c = (df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy().astype(float)
    n = len(o)
    same = np.r_[ent[1:] == ent[:-1], False]  # row i and i+1 same entity

    def sh(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[: n - k] = a[k:]
        return out

    valid_next = same.copy()
    entry = sh(o, 1)
    entry[~valid_next] = np.nan
    sl = entry - t.sl_atr * atr
    tp = entry + t.tp_atr * atr
    ok = np.isfinite(entry) & np.isfinite(atr) & (atr > 0)
    done = np.zeros(n, bool)
    exit_px = np.full(n, np.nan)
    alive = ok.copy()  # chain of same-entity rows
    for k in range(1, t.horizon + 1):
        if k > 1:
            alive &= sh(same.astype(float), k - 1) == 1  # row i+k-1 and i+k same entity
        hk, lk, ok_, ck = sh(h, k), sh(lo, k), sh(o, k), sh(c, k)
        live = alive & ~done & np.isfinite(hk)
        hit_sl = live & (lk <= sl)
        hit_tp = live & ~hit_sl & (hk >= tp)
        # gap fills at the open beyond the level (except on the entry bar)
        px_sl = np.where((k > 1) & (ok_ < sl), ok_, sl)
        px_tp = np.where((k > 1) & (ok_ > tp), ok_, tp)
        exit_px = np.where(hit_sl, px_sl, exit_px)
        exit_px = np.where(hit_tp, px_tp, exit_px)
        done |= hit_sl | hit_tp
        if k == t.horizon:
            timeout = live & ~hit_sl & ~hit_tp
            exit_px = np.where(timeout, ck, exit_px)
            done |= timeout
    exit_px[~done] = np.nan
    cost = (
        np.where(entry < 5, 0.01, np.where(entry < 20, 0.003, 0.001)) if market == "US" else 0.002
    )
    ret = exit_px / entry - 1 - cost
    risk = t.sl_atr * atr / entry
    ret[(~ok) | (entry <= sl)] = np.nan
    ratio = exit_px / entry
    ret[(ratio > 5) | (ratio < 0.2)] = (
        np.nan
    )  # impossible within the horizon for liquid names: data error
    ret[risk < 0.003] = np.nan  # stop closer than 0.3%: bad/flat price data
    return ret, ret / risk


def period_of(d: pl.Series) -> np.ndarray:
    a = d.to_numpy()
    return np.where(a <= np.datetime64(TRAIN_END), 0, np.where(a <= np.datetime64(VALID_END), 1, 2))
