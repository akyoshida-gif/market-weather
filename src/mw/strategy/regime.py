"""Market regime / macro / cross-asset features (one row per date), PIT-safe.

All features for date t use only data published by t's close: FRED values are aligned by their
published_at (as stored in silver), ETF ratios use closes up to t.
"""

from __future__ import annotations

import polars as pl

C = pl.col

# ETF baskets for rotation ratios (US)
DEFENSIVE = ["XLU", "XLP", "XLV"]
CYCLICAL = ["XLY", "XLK", "XLI", "XLF"]


def _etf_close(prices: pl.DataFrame, market: str, syms: list[str]) -> pl.DataFrame:
    ids = [f"{market}:{s}" for s in syms]
    w = (
        prices.filter(C("entity_id").is_in(ids))
        .pivot(on="entity_id", index="as_of", values="close")
        .sort("as_of")
    )
    return w.rename({c: c.split(":")[1] for c in w.columns if ":" in c})


def _chg(col: str, n: int) -> pl.Expr:
    return C(col) / C(col).shift(n) - 1


def us_regime(prices: pl.DataFrame, vol: pl.DataFrame, fred: pl.DataFrame) -> pl.DataFrame:
    syms = [
        "SPY",
        "IWM",
        "RSP",
        "QQQ",
        "TLT",
        "IEF",
        "HYG",
        "LQD",
        "GLD",
        "USO",
        "UUP",
        *DEFENSIVE,
        *CYCLICAL,
    ]
    e = _etf_close(prices, "US", syms)
    have = set(e.columns)
    e = e.with_columns(
        (pl.mean_horizontal([C(s) / C(s).first() for s in DEFENSIVE if s in have])
         / pl.mean_horizontal([C(s) / C(s).first() for s in CYCLICAL if s in have])).alias("def_cyc"),
        (C("IWM") / C("SPY")).alias("iwm_spy"),
        (C("RSP") / C("SPY")).alias("rsp_spy"),
        (C("HYG") / C("IEF")).alias("hyg_ief"),
        (C("TLT") / C("SPY")).alias("tlt_spy"),
        (C("GLD") / C("SPY")).alias("gld_spy"),
    )  # fmt: skip
    r = e.select(
        "as_of",
        (C("SPY") > C("SPY").rolling_mean(50)).alias("spy_up50"),
        (C("SPY") > C("SPY").rolling_mean(200)).alias("spy_up200"),
        _chg("SPY", 20).alias("spy_ret20"),
        _chg("def_cyc", 20).alias("def_cyc_chg20"),
        _chg("iwm_spy", 20).alias("iwm_spy_chg20"),
        _chg("rsp_spy", 20).alias("rsp_spy_chg20"),
        _chg("hyg_ief", 20).alias("hyg_ief_chg20"),
        _chg("tlt_spy", 20).alias("tlt_spy_chg20"),
        _chg("gld_spy", 20).alias("gld_spy_chg20"),
        _chg("USO", 20).alias("oil_chg20"),
        _chg("UUP", 20).alias("usd_chg20"),
        _chg("TLT", 20).alias("tlt_chg20"),
    )
    if vol.height:
        v = vol.pivot(
            on="entity_id", index="as_of", values="close", aggregate_function="last"
        ).sort("as_of")
        v = v.select(
            "as_of",
            C("IDX:US:VIX").alias("vix"),
            (C("IDX:US:VIX") / C("IDX:US:VIX3M")).alias("vix_term"),
            (C("IDX:US:VIX") - C("IDX:US:VIX").shift(5)).alias("vix_chg5"),
        )
        r = r.join(v, on="as_of", how="left")
    if fred.height:
        # align each series by publication time: value known at the close of date t
        f = fred.with_columns(
            C("published_at").dt.convert_time_zone("America/New_York").dt.date().alias("known")
        )
        series = {"MACRO:FRED:BAMLH0A0HYM2": "hy", "MACRO:FRED:DGS10": "y10", "MACRO:FRED:DGS2": "y2",
                  "MACRO:FRED:T10Y2Y": "curve", "MACRO:FRED:DFII10": "real10", "MACRO:FRED:NFCI": "nfci",
                  "MACRO:FRED:DTWEXBGS": "dxy", "MACRO:FRED:DCOILWTICO": "wti", "MACRO:FRED:DFF": "ff"}  # fmt: skip
        for sid, name in series.items():
            s = (
                f.filter(C("entity_id") == sid)
                .sort("known")
                .select(C("known").alias("as_of"), C("value").alias(name))
            )
            s = s.group_by("as_of").agg(C(name).last()).sort("as_of")
            r = r.sort("as_of").join_asof(s, on="as_of", strategy="backward")
        r = r.sort("as_of").with_columns(
            (C("hy") - C("hy").shift(20)).alias("hy_chg20"),
            (C("y10") - C("y10").shift(20)).alias("y10_chg20"),
            (C("y2") - C("y2").shift(20)).alias("y2_chg20"),
            (C("real10") - C("real10").shift(20)).alias("real10_chg20"),
            (C("nfci") - C("nfci").shift(20)).alias("nfci_chg20"),
        )
    return r.sort("as_of")


def jp_regime(prices: pl.DataFrame, us: pl.DataFrame, fred: pl.DataFrame) -> pl.DataFrame:
    e = _etf_close(prices, "JP", ["1306", "1321", "1631", "1625", "1627", "1617"])
    r = e.select(
        "as_of",
        (C("1306") > C("1306").rolling_mean(50)).alias("topix_up50"),
        (C("1306") > C("1306").rolling_mean(200)).alias("topix_up200"),
        _chg("1306", 20).alias("topix_ret20"),
        ((C("1627") + C("1617")) / (C("1625") + C("1631")))
        .pipe(lambda x: x / x.shift(20) - 1)
        .alias("jp_def_cyc_chg20"),
    )
    # US context known before Tokyo opens: previous US session (join_asof backward on date-1)
    us_cols = [
        c
        for c in ("vix", "vix_term", "hy_chg20", "spy_up50", "spy_ret20", "y10_chg20", "usd_chg20")
        if c in us.columns
    ]
    usp = us.select("as_of", *us_cols).with_columns(
        (C("as_of") + pl.duration(days=1)).alias("as_of")
    )
    r = r.sort("as_of").join_asof(usp.sort("as_of"), on="as_of", strategy="backward")
    if fred.height:
        f = fred.with_columns(
            C("published_at").dt.convert_time_zone("Asia/Tokyo").dt.date().alias("known")
        )
        s = (
            f.filter(C("entity_id") == "MACRO:FRED:DEXJPUS")
            .sort("known")
            .select(C("known").alias("as_of"), C("value").alias("usdjpy"))
        )
        s = s.group_by("as_of").agg(C("usdjpy").last()).sort("as_of")
        r = r.join_asof(s, on="as_of", strategy="backward").with_columns(
            (C("usdjpy") / C("usdjpy").shift(20) - 1).alias("usdjpy_chg20")
        )
    return r.sort("as_of")


def breadth(ind: pl.DataFrame) -> pl.DataFrame:
    b = (
        ind.filter(C("ma50").is_not_null())
        .group_by("as_of")
        .agg((C("close") > C("ma50")).mean().alias("breadth"))
    )
    return b.sort("as_of").with_columns(
        (C("breadth") - C("breadth").shift(10)).alias("breadth_chg10")
    )


# candidate regime filters: name -> (expr builder, human description)
FILTERS_US = {
    "spy_up50": (lambda: C("spy_up50"), "S&P500が50日線の上"),
    "spy_up200": (lambda: C("spy_up200"), "S&P500が200日線の上"),
    "breadth_up": (lambda: C("breadth_chg10") > 0, "50日線より上の銘柄の割合が10日前より増加"),
    "breadth_hi": (lambda: C("breadth") > 0.5, "50日線より上の銘柄が50%超"),
    "vix_lt20": (lambda: C("vix") < 20, "VIX 20未満"),
    "vix_term_lt1": (lambda: C("vix_term") < 1, "VIX期近/期先が1未満（平常）"),
    "vix_falling": (lambda: C("vix_chg5") < 0, "VIXが5日前より低下"),
    "hy_tightening": (lambda: C("hy_chg20") < 0, "HYスプレッドが20日前より縮小"),
    "hy_lt4": (lambda: C("hy") < 4, "HYスプレッド4%未満"),
    "curve_pos": (lambda: C("curve") > 0, "10年−2年がプラス"),
    "y10_falling": (lambda: C("y10_chg20") < 0, "10年金利が20日前より低下"),
    "real_falling": (lambda: C("real10_chg20") < 0, "実質金利が20日前より低下"),
    "usd_falling": (lambda: C("usd_chg20") < 0, "ドルが20日前より下落"),
    "oil_falling": (lambda: C("oil_chg20") < 0, "原油が20日前より下落"),
    "risk_on_rotation": (lambda: C("def_cyc_chg20") < 0, "景気敏感株がディフェンシブ株より強い"),
    "small_lead": (lambda: C("iwm_spy_chg20") > 0, "小型株が大型株より強い"),
    "broad_lead": (
        lambda: C("rsp_spy_chg20") > 0,
        "均等加重が時価総額加重より強い（上昇の広がり）",
    ),
    "credit_risk_on": (lambda: C("hyg_ief_chg20") > 0, "ハイイールド債が国債より強い"),
    "bonds_weak_vs_stocks": (lambda: C("tlt_spy_chg20") < 0, "株が国債より強い"),
    "nfci_easing": (lambda: C("nfci_chg20") < 0, "金融環境が緩和方向"),
}
FILTERS_JP = {
    "topix_up50": (lambda: C("topix_up50"), "TOPIXが50日線の上"),
    "topix_up200": (lambda: C("topix_up200"), "TOPIXが200日線の上"),
    "breadth_up": (lambda: C("breadth_chg10") > 0, "50日線より上の銘柄の割合が増加"),
    "breadth_hi": (lambda: C("breadth") > 0.5, "50日線より上の銘柄が50%超"),
    "jp_risk_on": (
        lambda: C("jp_def_cyc_chg20") < 0,
        "景気敏感（電機・銀行）がディフェンシブより強い",
    ),
    "us_vix_lt20": (lambda: C("vix") < 20, "米VIX 20未満"),
    "us_vix_term_lt1": (lambda: C("vix_term") < 1, "米VIX期近/期先が1未満"),
    "us_hy_tightening": (lambda: C("hy_chg20") < 0, "米HYスプレッド縮小"),
    "us_spy_up50": (lambda: C("spy_up50"), "米S&P500が50日線の上"),
    "yen_weakening": (lambda: C("usdjpy_chg20") > 0, "円安方向（20日）"),
    "yen_strengthening": (lambda: C("usdjpy_chg20") < 0, "円高方向（20日）"),
    "us_y10_falling": (lambda: C("y10_chg20") < 0, "米10年金利低下"),
}
