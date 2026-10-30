"""Stock-level numeric features and R outcomes per stop placement (VALIDATION_PLAN §2-1, §2-6, §6).

All features use data up to the signal day's close only. Columns:
  f_*  numeric features (float32), turned into "low / high" conditions later from their distribution
  r_<stop>_<h>_fin / _mfe   R outcomes per stop candidate and horizon (float32)
  rp_<stop>                 stop distance as a fraction of the entry price (for costs in R)
"""

from __future__ import annotations

import numpy as np
import polars as pl

C = pl.col
G = "entity_id"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
INDUSTRY_ETFS = ["SMH", "SOXX", "IGV", "XBI", "IBB", "KRE", "KBE", "ITB", "XHB", "XRT", "XOP", "OIH", "GDX", "GDXJ",
                 "TAN", "URA", "ITA", "JETS", "IYT", "PAVE", "GRID", "ARKK", "BLOK", "IPO"]  # fmt: skip
FACTOR_ETFS = ["MTUM", "QUAL", "USMV", "VLUE", "IWF", "IWD", "IWO", "IWN"]
ROT_ETFS = SECTOR_ETFS + INDUSTRY_ETFS + FACTOR_ETFS

STOPS = {
    "sig_low": "シグナル日の安値の少し下（0.1ATR）",
    "low5": "直近5日の安値の少し下",
    "low10": "直近10日の安値の少し下",
    "low20": "直近20日の安値の少し下",
    "ma20": "20日線の少し下（0.2ATR）",
    "ma50": "50日線の少し下（0.2ATR）",
    "atr1_5": "買値から1.5ATR下",
    "atr2": "買値から2ATR下",
    "atr3": "買値から3ATR下",
}
R_HORIZONS = [5, 20, 60, 120]

# how far back each feature looks (sessions). A data-error day (split / |move| > 60%) inside that window
# makes the value meaningless (e.g. an unrecorded 1:10 split turns into a "+900% in a year"), so it is
# set to "no data". Features not listed look back 20 sessions or less (those rows are not eligible at all).
LOOKBACK = {
    252: ["f_dist_hi", "f_dist_lo", "f_days_since_hi", "f_ret252", "f_dev_ma200", "f_slope_ma200", "f_ma50_200",
          "f_sec_corr", "f_sec_rs63", "f_sec_rank", "f_rs_sec20", "f_rs_sec63", "f_rs_ind63", "f_ind_rs63", "f_share_growth"],
    126: ["f_ret126", "f_rs_spy126", "f_cup", "f_bbw_rel", "f_ichi_cloud"],
    63: ["f_ret63", "f_rs_spy63", "f_dev_ma50", "f_slope_ma50", "f_ma20_50", "f_pos60", "f_fib", "f_base_width",
         "f_vcp", "f_lo_retest", "f_contract", "f_pole", "f_hi_slope", "f_lo_slope", "f_ichi_tk"],
}  # fmt: skip
COND_LOOKBACK = {
    252: ["c_new_high_252", "c_near_high_252", "c_above_ma200", "c_below_ma200", "c_po_long", "c_gc_50_200"],
    63: ["c_break_high_60", "c_pullback_5pct", "c_fib_382", "c_fib_50", "c_fib_618", "c_ma50_rising", "c_reclaim_ma50",
         "c_ichi_above_cloud", "c_ichi_cloud_break", "c_ichi_tk_cross", "c_ichi_chikou_up", "c_ichi_sanyaku", "c_bb_squeeze"],
}  # fmt: skip

FEATURE_LABELS = {
    "f_dev_ma5": "5日線からの乖離率", "f_dev_ma20": "20日線からの乖離率", "f_dev_ma50": "50日線からの乖離率",
    "f_dev_ma200": "200日線からの乖離率", "f_slope_ma20": "20日線の傾き（5日）", "f_slope_ma50": "50日線の傾き（10日）",
    "f_slope_ma200": "200日線の傾き（20日）", "f_ma20_50": "20日線と50日線の距離", "f_ma50_200": "50日線と200日線の距離",
    "f_ret1": "当日の騰落率", "f_ret5": "5日の騰落率", "f_ret20": "20日の騰落率", "f_ret63": "3か月の騰落率",
    "f_ret126": "6か月の騰落率", "f_ret252": "1年の騰落率", "f_rsi": "RSI(14)", "f_rsi2": "RSI(2)", "f_stoch": "ストキャスティクス",
    "f_macd": "MACD（株価比）", "f_macd_hist": "MACDヒストグラム（株価比）", "f_dist_hi": "52週高値からの距離",
    "f_dist_lo": "52週安値からの距離", "f_pos60": "60日の値幅の中での位置", "f_days_since_hi": "52週高値からの経過日数",
    "f_atr_pct": "値動きの大きさ（ATR÷株価）", "f_bbw": "ボリンジャーバンドの幅", "f_bbw_rel": "バンド幅の過去120日比",
    "f_contract": "値幅の収縮度（5日÷50日のATR）", "f_vr": "出来高倍率（20日平均比）", "f_vr5": "5日平均出来高の倍率",
    "f_dvol20": "20日平均の売買代金", "f_updown_vol": "上昇日と下落日の出来高の比（20日）", "f_obv": "OBVの20日変化（出来高比）",
    "f_gap": "寄り付きの窓", "f_streak": "連続上昇・下落の日数", "f_clv": "引けの位置（当日の値幅の中）",
    "f_body": "ローソク足の実体（ATR比）", "f_ichi_cloud": "一目の雲からの距離", "f_ichi_tk": "転換線と基準線の差",
    "f_fib": "60日の押しの深さ（フィボナッチ）", "f_pole": "直前の急騰の大きさ（フラッグの旗竿）",
    "f_flag_depth": "直近10日の押しの深さ（フラッグ）", "f_base_width": "60日のベースの値幅", "f_hi_slope": "高値の切り上げ・切り下げ",
    "f_lo_slope": "安値の切り上げ・切り下げ", "f_vcp": "値幅の段階的な縮み（VCP）", "f_cup": "カップの戻り具合",
    "f_lo_retest": "直近安値と60日安値の差（二番底）", "f_rs_spy20": "S&P500に対する強さ（20日）",
    "f_rs_spy63": "S&P500に対する強さ（3か月）", "f_rs_spy126": "S&P500に対する強さ（6か月）",
    "f_rs_sec20": "所属セクターに対する強さ（20日）", "f_rs_sec63": "所属セクターに対する強さ（3か月）",
    "f_rs_ind63": "所属業種に対する強さ（3か月）", "f_sec_rs63": "所属セクターの強さ（3か月、対S&P500）",
    "f_sec_rank": "所属セクターの強さの順位（11の中）", "f_ind_rs63": "所属業種の強さ（3か月、対S&P500）",
    "f_sec_corr": "所属セクターとの連動の強さ", "f_rank_ret20": "20日騰落率の全銘柄内順位",
    "f_rank_ret63": "3か月騰落率の全銘柄内順位", "f_rank_ret126": "6か月騰落率の全銘柄内順位",
    "f_rank_dvol": "売買代金の全銘柄内順位", "f_mcap": "時価総額", "f_si_pct": "空売り残高（発行済株式比）",
    "f_dtc": "カバー日数", "f_svr5": "空売り比率（5日平均）", "f_svr_chg": "空売り比率の変化（5日平均−60日平均）",
    "f_ftd": "受け渡し不履行（出来高比）", "f_ins_value": "インサイダーの買い金額（30日）",
    "f_days_since_earn": "決算からの経過日数", "f_share_growth": "発行済株式数の1年の増加率",
}  # fmt: skip


def stock_features(df: pl.DataFrame, spy: pl.DataFrame) -> pl.DataFrame:
    """df: output of technical.conditions (sorted by entity, as_of). spy: as_of, spy_close."""
    c, h, lo, o, v = C("close"), C("high"), C("low"), C("open"), C("volume")
    prev = lambda e: e.shift(1).over(G)  # noqa: E731
    df = df.join(spy, on="as_of", how="left").with_columns(
        lo.rolling_min(252).over(G).alias("_ll252"),
        h.rolling_max(120).over(G).alias("_hh120"),
        h.rolling_max(10).over(G).alias("_hh10"),
        lo.rolling_min(10).over(G).alias("_ll10"),
        C("tr").rolling_mean(5).over(G).alias("_atr5"),
        C("tr").rolling_mean(50).over(G).alias("_atr50"),
        (h.rolling_max(20).over(G) - lo.rolling_min(20).over(G)).alias("_rng20"),
        (h.rolling_max(10).over(G) - lo.rolling_min(10).over(G)).alias("_rng10"),
        (C("hh60") - C("ll60")).alias("_rng60"),
        pl.when(c > prev(c)).then(v).otherwise(0).rolling_sum(20).over(G).alias("_upv"),
        pl.when(c < prev(c)).then(v).otherwise(0).rolling_sum(20).over(G).alias("_dnv"),
        v.rolling_mean(5).over(G).alias("_v5"),
        C("bbw").rolling_mean(120).over(G).alias("_bbw120"),
    )
    # streak: run length of the current direction
    df = (
        df.with_columns(
            pl.when(c > prev(c)).then(1).when(c < prev(c)).then(-1).otherwise(0).alias("_dir")
        )
        .with_columns(
            (C("_dir") != C("_dir").shift(1).over(G))
            .fill_null(True)
            .cum_sum()
            .over(G)
            .alias("_run")
        )
        .with_columns(
            (pl.int_range(1, pl.len() + 1).over([G, "_run"]) * C("_dir")).alias("_streak")
        )
    )
    # days since the 252-day high
    df = df.with_columns(
        pl.when(h >= h.rolling_max(252).over(G))
        .then(C("as_of"))
        .otherwise(None)
        .forward_fill()
        .over(G)
        .alias("_hi_day")
    )
    spy_ret = lambda n: C("spy_close") / C("spy_close").shift(n).over(G) - 1  # noqa: E731
    ret = lambda n: c / c.shift(n).over(G) - 1  # noqa: E731
    f = {
        "f_dev_ma5": c / C("ma5") - 1, "f_dev_ma20": c / C("ma20") - 1, "f_dev_ma50": c / C("ma50") - 1, "f_dev_ma200": c / C("ma200") - 1,
        "f_slope_ma20": C("ma20") / C("ma20").shift(5).over(G) - 1, "f_slope_ma50": C("ma50") / C("ma50").shift(10).over(G) - 1,
        "f_slope_ma200": C("ma200") / C("ma200").shift(20).over(G) - 1,
        "f_ma20_50": C("ma20") / C("ma50") - 1, "f_ma50_200": C("ma50") / C("ma200") - 1,
        "f_ret1": ret(1), "f_ret5": ret(5), "f_ret20": ret(20), "f_ret63": ret(63), "f_ret126": ret(126), "f_ret252": ret(252),
        "f_rsi": C("rsi"), "f_rsi2": C("rsi2"), "f_stoch": C("stk"),
        "f_macd": C("macd") / c, "f_macd_hist": (C("macd") - C("macd_sig")) / c,
        "f_dist_hi": c / C("hh252p") - 1, "f_dist_lo": c / C("_ll252") - 1,
        "f_pos60": (c - C("ll60")) / C("_rng60").clip(lower_bound=1e-9),
        "f_days_since_hi": (C("as_of") - C("_hi_day")).dt.total_days(),
        "f_atr_pct": C("atr") / c, "f_bbw": C("bbw"), "f_bbw_rel": C("bbw") / C("_bbw120"),
        "f_contract": C("_atr5") / C("_atr50"), "f_vr": C("vr"), "f_vr5": C("_v5") / C("vol20"),
        "f_dvol20": C("dvol20").log10(), "f_updown_vol": C("_upv") / C("_dnv").clip(lower_bound=1),
        "f_obv": C("obv20") / (C("vol20") * 20).clip(lower_bound=1),
        "f_gap": o / C("pc") - 1, "f_streak": C("_streak"), "f_clv": (c - lo) / (h - lo).clip(lower_bound=1e-9),
        "f_body": (c - o) / C("atr"), "f_ichi_cloud": c / C("cloud_top") - 1, "f_ichi_tk": (C("tenkan") - C("kijun")) / c,
        "f_fib": (C("hh60") - c) / C("_rng60").clip(lower_bound=1e-9),
        # chart-pattern geometry
        "f_pole": c.shift(10).over(G) / c.shift(30).over(G) - 1,
        "f_flag_depth": (C("_hh10") - C("_ll10")) / C("_hh10"),
        "f_base_width": C("_rng60") / c,
        "f_hi_slope": C("_hh10") / C("_hh10").shift(20).over(G) - 1,
        "f_lo_slope": C("_ll10") / C("_ll10").shift(20).over(G) - 1,
        "f_vcp": (C("_rng10") / C("_rng20").clip(lower_bound=1e-9)) * (C("_rng20") / C("_rng60").clip(lower_bound=1e-9)),
        "f_cup": (c - C("ll60")) / (C("_hh120") - C("ll60")).clip(lower_bound=1e-9),
        "f_lo_retest": C("_ll10") / C("ll60") - 1,
        # relative strength vs the market
        "f_rs_spy20": ret(20) - spy_ret(20), "f_rs_spy63": ret(63) - spy_ret(63), "f_rs_spy126": ret(126) - spy_ret(126),
    }  # fmt: skip
    df = df.with_columns([e.cast(pl.Float32).alias(k) for k, e in f.items()])
    df = df.with_columns(
        [pl.when(C(k).is_infinite()).then(None).otherwise(C(k)).alias(k) for k in f]
    )
    return df.drop([x for x in df.columns if x.startswith("_")] + ["spy_close"])


def etf_returns(prices: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    """Wide table: as_of, r_<SYM> (daily return), px_<SYM> (close)."""
    ids = {f"US:{s}": s for s in symbols}
    e = prices.filter(C(G).is_in(list(ids))).select(G, "as_of", "close").sort([G, "as_of"])
    e = e.with_columns(
        (C("close") / C("close").shift(1).over(G) - 1).alias("r"),
        C(G).replace_strict(ids).alias("sym"),
    )
    r = e.pivot(on="sym", index="as_of", values="r").rename(
        {s: f"r_{s}" for s in symbols if s in e["sym"].unique().to_list()}
    )
    p = e.pivot(on="sym", index="as_of", values="close").rename(
        {s: f"px_{s}" for s in symbols if s in e["sym"].unique().to_list()}
    )
    return r.join(p, on="as_of", how="full", coalesce=True).sort("as_of")


def sector_assign(df: pl.DataFrame, etf: pl.DataFrame, rs_long: pl.DataFrame) -> pl.DataFrame:
    """Assign each stock-day to the sector ETF / industry ETF it moved with most over the past 250 sessions
    (correlation of daily returns), then attach relative-strength features of that sector / industry."""
    have_sec = [s for s in SECTOR_ETFS if f"r_{s}" in etf.columns]
    have_ind = [s for s in INDUSTRY_ETFS if f"r_{s}" in etf.columns]
    x = df.select(G, "as_of", "close").join(
        etf.select("as_of", *[f"r_{s}" for s in have_sec + have_ind]), on="as_of", how="left"
    )
    x = x.with_columns((C("close") / C("close").shift(1).over(G) - 1).alias("_r"))
    x = x.with_columns(
        [
            pl.rolling_corr(C("_r"), C(f"r_{s}"), window_size=250, min_samples=120)
            .over(G)
            .alias(f"k_{s}")
            for s in have_sec + have_ind
        ]
    )
    sec_k = pl.concat_list([C(f"k_{s}").fill_null(-9) for s in have_sec])
    ind_k = pl.concat_list([C(f"k_{s}").fill_null(-9) for s in have_ind])
    x = (
        x.with_columns(
            sec_k.list.arg_max().alias("_si"),
            sec_k.list.max().alias("f_sec_corr"),
            ind_k.list.arg_max().alias("_ii"),
            ind_k.list.max().alias("_icorr"),
        )
        .with_columns(
            pl.when(C("f_sec_corr") > 0.2)
            .then(C("_si").replace_strict(dict(enumerate(have_sec)), default=None))
            .alias("sector"),
            pl.when(C("_icorr") > 0.4)
            .then(C("_ii").replace_strict(dict(enumerate(have_ind)), default=None))
            .alias("industry"),
        )
        .select(G, "as_of", "sector", "industry", C("f_sec_corr").cast(pl.Float32))
    )
    out = df.join(x, on=[G, "as_of"], how="left")
    sec = rs_long.select(C("sym").alias("sector"), "as_of", C("rs63").alias("f_sec_rs63"), C("rank").alias("f_sec_rank"),
                         C("ret20").alias("_sret20"), C("ret63").alias("_sret63"))  # fmt: skip
    ind = rs_long.select(
        C("sym").alias("industry"),
        "as_of",
        C("rs63").alias("f_ind_rs63"),
        C("ret63").alias("_iret63"),
    )
    out = out.join(sec, on=["sector", "as_of"], how="left").join(
        ind, on=["industry", "as_of"], how="left"
    )
    ret = lambda n: C("close") / C("close").shift(n).over(G) - 1  # noqa: E731
    out = out.with_columns(
        (ret(20) - C("_sret20")).cast(pl.Float32).alias("f_rs_sec20"),
        (ret(63) - C("_sret63")).cast(pl.Float32).alias("f_rs_sec63"),
        (ret(63) - C("_iret63")).cast(pl.Float32).alias("f_rs_ind63"),
        C("f_sec_rs63").cast(pl.Float32),
        C("f_sec_rank").cast(pl.Float32),
        C("f_ind_rs63").cast(pl.Float32),
    )
    return out.drop("_sret20", "_sret63", "_iret63")


def rs_table(etf: pl.DataFrame) -> pl.DataFrame:
    """Long table per (ETF, day): returns, relative strength vs SPY, rank among the 11 sectors."""
    rows = []
    for s in ROT_ETFS:
        if f"px_{s}" not in etf.columns:
            continue
        p, b = C(f"px_{s}"), C("px_SPY")
        rows.append(etf.select("as_of", pl.lit(s).alias("sym"),
                               (p / p.shift(20) - 1).alias("ret20"), (p / p.shift(63) - 1).alias("ret63"),
                               ((p / b) / (p / b).shift(20) - 1).alias("rs20"), ((p / b) / (p / b).shift(63) - 1).alias("rs63")))  # fmt: skip
    t = pl.concat(rows)
    rk = (
        t.filter(C("sym").is_in(SECTOR_ETFS))
        .with_columns(C("rs63").rank(descending=True).over("as_of").alias("rank"))
        .select("sym", "as_of", "rank")
    )
    return t.join(rk, on=["sym", "as_of"], how="left")


def r_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    """For each stop candidate: entry = next session's open, R = entry - stop.

    Walk forward day by day: a day that opens at/below the stop exits at its open (gap), a day whose low
    touches the stop exits at the stop (-1R). _fin = R at the horizon's close (or the exit), _mfe = best R
    reached on days before the stop day. A trade whose stop is not below the entry is invalid (null).
    """
    n = df.height
    ent = df[G].to_numpy()
    o, h, lo, c = (df[k].to_numpy().astype(np.float64) for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy().astype(np.float64)
    ll5 = df.select(C("low").rolling_min(5).over(G))["low"].to_numpy().astype(np.float64)
    ll10 = df.select(C("low").rolling_min(10).over(G))["low"].to_numpy().astype(np.float64)
    ll20 = df.select(C("low").rolling_min(20).over(G))["low"].to_numpy().astype(np.float64)
    ma20, ma50 = df["ma20"].to_numpy().astype(np.float64), df["ma50"].to_numpy().astype(np.float64)
    # last row index of each entity (to stop paths at the stock's end)
    idx = np.arange(n)
    change = np.r_[ent[1:] != ent[:-1], True]
    ends = np.flatnonzero(change)
    last = np.repeat(ends, np.diff(np.r_[-1, ends]))
    entry = np.full(n, np.nan)
    ok1 = idx + 1 <= last
    entry[ok1] = o[idx[ok1] + 1]
    stops = {
        "sig_low": lo - 0.1 * atr, "low5": ll5 - 0.1 * atr,
        "low10": ll10 - 0.1 * atr, "low20": ll20 - 0.1 * atr,
        "ma20": ma20 - 0.2 * atr, "ma50": ma50 - 0.2 * atr, "atr1_5": entry - 1.5 * atr,
        "atr2": entry - 2 * atr, "atr3": entry - 3 * atr,
    }  # fmt: skip
    out: dict[str, np.ndarray] = {}
    hmax = max(R_HORIZONS)
    for name, stop in stops.items():
        R = entry - stop
        # a stop closer than half a day's range or 1% of the price is not a real stop
        valid = np.isfinite(R) & (np.maximum(0.5 * atr, 0.01 * entry) <= R) & (0.5 * entry > R)
        alive = valid.copy()
        exit_r = np.full(n, np.nan)
        mfe = np.zeros(n)
        fin = {hh: np.full(n, np.nan) for hh in R_HORIZONS}
        mfe_h = {hh: np.full(n, np.nan) for hh in R_HORIZONS}
        stopped = np.zeros(n, dtype=bool)
        for k in range(1, hmax + 1):
            j = idx + k
            inside = valid & (j <= last)
            jj = np.where(inside, j, 0)
            ok = alive & inside
            gap = ok & (o[jj] <= stop) & (k > 1)
            hit = ok & ~gap & (lo[jj] <= stop)
            exit_r[gap] = (o[jj][gap] - entry[gap]) / R[gap]
            exit_r[hit] = -1.0
            stopped |= gap | hit
            alive &= ~(gap | hit)
            upd = alive & inside
            mfe[upd] = np.maximum(mfe[upd], (h[jj][upd] - entry[upd]) / R[upd])
            if k in fin:
                done = valid & inside
                fin[k][done] = np.where(
                    stopped[done], exit_r[done], (c[jj][done] - entry[done]) / R[done]
                )
                mfe_h[k][done] = mfe[done]
        for hh in R_HORIZONS:
            out[f"r_{name}_{hh}_fin"] = fin[hh].astype(np.float32)
            out[f"r_{name}_{hh}_mfe"] = mfe_h[hh].astype(np.float32)
        out[f"rp_{name}"] = np.where(valid, R / entry, np.nan).astype(np.float32)
    return df.with_columns([pl.Series(k, v, nan_to_null=True) for k, v in out.items()])
