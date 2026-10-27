"""Market-wide daily features, leading-indicator scan and sector-rotation scan (VALIDATION_PLAN §2-4..2-6).

day_table(): one row per US session with m_* numeric features, each usable at that session's close:
  FRED series (as-of their publication), VIX family, Cboe put/call ratios, FINRA margin, CFTC positioning,
  ETF-based ratios and the relative strength of every sector / industry / factor ETF.
  For every series: level, position in its own 5-year range, and 20/60-session changes.
leading(): does a feature at t line up with returns of the market / sectors over the NEXT weeks-months?
rotation(): after sector X peaks out (or bottoms out) versus the S&P 500, which sectors outperform next?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from mw.report import dashboard as D
from mw.research import external, features

C = pl.col

FRED_NAMES = {
    "DGS3MO": "米国債3か月", "DGS1": "米国債1年", "DGS2": "米国債2年", "DGS5": "米国債5年", "DGS10": "米国債10年", "DGS30": "米国債30年",
    "DFF": "政策金利（FF）", "SOFR": "SOFR", "T10Y2Y": "10年−2年", "T10Y3M": "10年−3か月", "DFII10": "実質金利10年",
    "T10YIE": "期待インフレ率10年", "BAA10Y": "BAA社債スプレッド", "BAMLH0A0HYM2": "ハイイールド債スプレッド",
    "BAMLC0A0CM": "投資適格債スプレッド", "NFCI": "金融環境指数", "DTWEXBGS": "ドル指数（広義）", "DEXJPUS": "ドル円",
    "DCOILWTICO": "原油WTI", "DCOILBRENTEU": "原油ブレント", "DHHNGSP": "天然ガス", "PCOPPUSDM": "銅",
    "M2SL": "マネーサプライM2", "WALCL": "FRB総資産", "RRPONTSYD": "逆レポ残高", "WTREGEN": "財務省口座残高",
    "ICSA": "新規失業保険申請", "UNRATE": "失業率", "CPIAUCSL": "CPI", "INDPRO": "鉱工業生産", "UMCSENT": "消費者心理",
}  # fmt: skip
RATE_LIKE = {"DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "DFF", "SOFR", "T10Y2Y", "T10Y3M", "DFII10", "T10YIE",
             "BAA10Y", "BAMLH0A0HYM2", "BAMLC0A0CM", "NFCI", "UNRATE"}  # fmt: skip
ETF_NAMES = {
    "XLK": "テクノロジー", "XLF": "金融", "XLV": "ヘルスケア", "XLY": "一般消費財", "XLP": "生活必需品", "XLE": "エネルギー",
    "XLI": "資本財", "XLB": "素材", "XLU": "公益", "XLRE": "不動産", "XLC": "通信", "SMH": "半導体", "SOXX": "半導体（SOXX）",
    "IGV": "ソフトウェア", "XBI": "バイオ（小型）", "IBB": "バイオ", "KRE": "地銀", "KBE": "銀行", "ITB": "住宅建設",
    "XHB": "住宅関連", "XRT": "小売", "XOP": "石油探査", "OIH": "石油サービス", "GDX": "金鉱株", "GDXJ": "金鉱株（小型）",
    "TAN": "太陽光", "URA": "ウラン", "ITA": "防衛・航空宇宙", "JETS": "航空", "IYT": "運輸", "PAVE": "インフラ", "GRID": "送電網",
    "ARKK": "破壊的イノベーション", "BLOK": "ブロックチェーン", "IPO": "新規上場株", "MTUM": "モメンタム", "QUAL": "クオリティ",
    "USMV": "低ボラティリティ", "VLUE": "バリュー", "IWF": "大型グロース", "IWD": "大型バリュー", "IWO": "小型グロース", "IWN": "小型バリュー",
}  # fmt: skip
COT_NAMES = {"es": "S&P500先物", "nq": "ナスダック100先物", "rty": "ラッセル2000先物", "ty": "米国債10年先物", "tu": "米国債2年先物",
             "dx": "ドル指数先物", "jy": "円先物", "vx": "VIX先物", "cl": "原油先物", "gc": "金先物", "hg": "銅先物", "ng": "天然ガス先物"}  # fmt: skip
COT_GROUPS = {
    "lev_money": "ヘッジファンド等",
    "asset_mgr": "運用会社",
    "dealer": "ディーラー",
    "managed_money": "投機筋",
    "producer": "生産者",
}

POS_WINDOW = 1260  # ~5 years of sessions


def _derive(
    t: pl.DataFrame,
    col: str,
    label: str,
    rate_like: bool,
    labels: dict[str, str],
    level: bool = True,
) -> list[pl.Expr]:
    """Level, position in the 5-year range, and 20 / 60-session changes of one daily series."""
    x = C(col)
    lo, hi = x.rolling_min(POS_WINDOW, min_samples=250), x.rolling_max(POS_WINDOW, min_samples=250)
    chg = (lambda n: x - x.shift(n)) if rate_like else (lambda n: x / x.shift(n) - 1)
    out = [
        ((x - lo) / (hi - lo)).alias(f"m_{col}_pos"),
        chg(20).alias(f"m_{col}_chg20"),
        chg(60).alias(f"m_{col}_chg60"),
    ]
    labels |= {
        f"m_{col}_pos": f"{label}（過去5年の中での位置）",
        f"m_{col}_chg20": f"{label}の変化（20日）",
        f"m_{col}_chg60": f"{label}の変化（60日）",
    }
    if level:
        out.append(x.alias(f"m_{col}_lvl"))
        labels[f"m_{col}_lvl"] = f"{label}（水準）"
    return out


def _asof_daily(cal: pl.DataFrame, s: pl.DataFrame, name: str) -> pl.DataFrame:
    """s: known (date usable at its close), value -> calendar with the latest known value."""
    s = (
        s.group_by("known")
        .agg(C("value").last())
        .sort("known")
        .rename({"known": "as_of", "value": name})
    )
    return cal.join_asof(s, on="as_of", strategy="backward")


def day_table(
    storage: Any, prices: pl.DataFrame, etf: pl.DataFrame, breadth: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, str]]:
    labels: dict[str, str] = {}
    cal = etf.select("as_of").filter(C("as_of").is_not_null()).unique().sort("as_of")
    t = cal
    # ---- FRED
    fred = D._read_years(storage, "fred", "GLOBAL").with_columns(
        external.known_date(C("published_at"))
    )
    for sid in FRED_NAMES:
        s = fred.filter(C("entity_id") == f"MACRO:FRED:{sid}").select("known", "value")
        if s.height:
            t = t.join(_asof_daily(cal, s, sid), on="as_of", how="left")
    # ---- VIX family (index closes, same session)
    vol = D._read_years(storage, "vol_indices", "US")
    if vol.height:
        v = vol.pivot(
            on="entity_id", index="as_of", values="close", aggregate_function="last"
        ).sort("as_of")
        keep = {
            "IDX:US:VIX": "VIX",
            "IDX:US:VIX3M": "VIX3M",
            "IDX:US:VVIX": "VVIX",
            "IDX:US:SKEW": "SKEW",
        }
        v = v.select("as_of", *[C(k).alias(n) for k, n in keep.items() if k in v.columns])
        t = t.join(v, on="as_of", how="left")
        if "VIX3M" in t.columns:
            t = t.with_columns((C("VIX") / C("VIX3M")).alias("VIXTERM"))
    # ---- Cboe put/call (usable next session)
    pk = [k for k in storage.list("silver/cboe_pcr/") if k.endswith(".parquet")]
    if pk:
        p = D._read_many(storage, pk).with_columns(external.known_date(C("published_at")))
        for col in ("pc_total", "pc_equity", "pc_index"):
            s = p.select("known", C(col).alias("value")).drop_nulls()
            t = t.join(_asof_daily(cal, s, col.upper()), on="as_of", how="left")
            t = t.with_columns(C(col.upper()).rolling_mean(5).alias(col.upper()))
    # ---- FINRA margin (monthly)
    if storage.exists("silver/finra_margin/market=US/latest.parquet"):
        m = (
            storage.read_parquet("silver/finra_margin/market=US/latest.parquet")
            .with_columns(external.known_date(C("published_at")))
            .sort("as_of")
        )
        m = m.with_columns(
            (C("margin_debit") / C("margin_debit").shift(12) - 1).alias("MARGIN_YOY"),
            (
                C("margin_debit") / (C("free_credit_cash") + C("free_credit_margin").fill_null(0))
            ).alias("MARGIN_TO_CREDIT"),
            C("margin_debit").alias("MARGIN"),
        )
        for col in ("MARGIN", "MARGIN_YOY", "MARGIN_TO_CREDIT"):
            t = t.join(
                _asof_daily(cal, m.select("known", C(col).alias("value")).drop_nulls(), col),
                on="as_of",
                how="left",
            )
    # ---- CFTC (net position / open interest, weekly)
    ck = [k for k in storage.list("silver/cftc_cot/") if k.endswith(".parquet")]
    cot_cols: list[tuple[str, str]] = []
    if ck:
        cf = D._read_many(storage, ck).with_columns(external.known_date(C("published_at")))
        cf = cf.with_columns(
            pl.when(C("market") == "rty_cme")
            .then(pl.lit("rty"))
            .otherwise(C("market"))
            .alias("market")
        )
        for mk in COT_NAMES:
            sub = cf.filter(C("market") == mk).sort("as_of")
            for g in COT_GROUPS:
                if f"net_{g}" not in sub.columns or sub[f"net_{g}"].null_count() == sub.height:
                    continue
                s = sub.select(
                    "known", (C(f"net_{g}") / C("open_interest")).alias("value")
                ).drop_nulls()
                if s.height < 50:
                    continue
                name = f"COT_{mk}_{g}".upper()
                t = t.join(_asof_daily(cal, s, name), on="as_of", how="left")
                cot_cols.append((name, f"{COT_NAMES[mk]}の{COT_GROUPS[g]}の買い越し（建玉比）"))
    # ---- ETF-based (same session closes)
    e = etf.select("as_of", *[c for c in etf.columns if c.startswith("px_")])
    t = t.join(e, on="as_of", how="left")
    ratio = {"IWM_SPY": ("px_IWM", "px_SPY", "小型株÷大型株"), "RSP_SPY": ("px_RSP", "px_SPY", "均等加重÷時価総額加重"),
             "QQQ_SPY": ("px_QQQ", "px_SPY", "ナスダック100÷S&P500"), "HYG_IEF": ("px_HYG", "px_IEF", "ハイイールド債÷国債"),
             "XLY_XLP": ("px_XLY", "px_XLP", "一般消費財÷生活必需品"), "EEM_SPY": ("px_EEM", "px_SPY", "新興国÷米国"),
             "GLD_SPY": ("px_GLD", "px_SPY", "金÷S&P500"), "TLT_SPY": ("px_TLT", "px_SPY", "長期国債÷S&P500")}  # fmt: skip
    t = t.with_columns(
        [
            (C(a) / C(b)).alias(k)
            for k, (a, b, _) in ratio.items()
            if a in t.columns and b in t.columns
        ]
    )
    if "PCOPPUSDM" in t.columns and "px_GLD" in t.columns:
        t = t.with_columns((C("PCOPPUSDM") / C("px_GLD")).alias("CU_AU"))
    t = t.join(breadth, on="as_of", how="left")
    # ---- derive features
    exprs: list[pl.Expr] = []
    for sid, name in FRED_NAMES.items():
        if sid in t.columns:
            exprs += _derive(t, sid, name, sid in RATE_LIKE, labels)
    misc = {"VIX": ("VIX", True), "VIX3M": ("VIX（3か月）", True), "VVIX": ("VIXのボラティリティ", True), "SKEW": ("SKEW指数", True),
            "VIXTERM": ("VIX期近÷期先", True), "PC_TOTAL": ("プット・コール比率（全体、5日平均）", True),
            "PC_EQUITY": ("プット・コール比率（個別株、5日平均）", True), "PC_INDEX": ("プット・コール比率（指数、5日平均）", True),
            "MARGIN": ("信用取引の買い残高", False), "MARGIN_YOY": ("信用買い残高の前年比", True),
            "MARGIN_TO_CREDIT": ("信用買い残高÷現金残高", True), "CU_AU": ("銅÷金", False), "BREADTH": ("50日線より上の銘柄の割合", True)}  # fmt: skip
    for col, (name, rl) in misc.items():
        if col in t.columns:
            exprs += _derive(t, col, name, rl, labels)
    for col, name in cot_cols:
        exprs += _derive(t, col, name, True, labels)
    for k, (_, _, name) in ratio.items():
        if k in t.columns:
            exprs += _derive(t, k, name, False, labels, level=False)
    for s in ["SPY", "TLT", "GLD", "USO", "UUP", "IWM", "QQQ"]:
        if f"px_{s}" in t.columns:
            x = C(f"px_{s}")
            exprs += [
                (x / x.rolling_mean(200) - 1).alias(f"m_{s}_dev200"),
                (x / x.shift(20) - 1).alias(f"m_{s}_ret20"),
                (x / x.shift(63) - 1).alias(f"m_{s}_ret63"),
            ]
            labels |= {
                f"m_{s}_dev200": f"{s}の200日線からの乖離率",
                f"m_{s}_ret20": f"{s}の20日騰落率",
                f"m_{s}_ret63": f"{s}の3か月騰落率",
            }
    for s in features.ROT_ETFS:
        if f"px_{s}" in t.columns and "px_SPY" in t.columns:
            rs = C(f"px_{s}") / C("px_SPY")
            exprs += [
                (rs / rs.shift(20) - 1).alias(f"m_rs_{s}_20"),
                (rs / rs.shift(63) - 1).alias(f"m_rs_{s}_63"),
            ]
            nm = ETF_NAMES.get(s, s)
            labels |= {
                f"m_rs_{s}_20": f"{nm}（{s}）のS&P500に対する強さ（20日）",
                f"m_rs_{s}_63": f"{nm}（{s}）のS&P500に対する強さ（3か月）",
            }
    out = t.select("as_of", *exprs)
    mcols = [c for c in out.columns if c.startswith("m_")]
    out = out.with_columns(
        [
            pl.when(C(c).is_infinite()).then(None).otherwise(C(c)).cast(pl.Float32).alias(c)
            for c in mcols
        ]
    )
    return out, labels


# ---------------------------------------------------------------- leading-indicator scan
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 12:
        return float("nan")
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def leading(day: pl.DataFrame, etf: pl.DataFrame, dev_end: Any, targets: list[str], horizons: tuple[int, ...] = (20, 60, 120),
            lags: tuple[int, ...] = (0, 20, 60)) -> list[dict[str, Any]]:  # fmt: skip
    """Feature at t vs return of each target over sessions (t+lag, t+lag+h]. Non-overlapping samples (every h sessions).

    Relative targets (sectors) use the return minus SPY's. Reports the rank correlation in the development
    period (to dev_end) and in the later period, plus mean returns by feature quintile.
    """
    t = day.join(
        etf.select("as_of", *[f"px_{s}" for s in targets + ["SPY"] if f"px_{s}" in etf.columns]),
        on="as_of",
        how="left",
    ).sort("as_of")
    dates = t["as_of"].to_numpy()
    dev = dates <= np.datetime64(dev_end)
    feats = [c for c in t.columns if c.startswith("m_")]
    px = {
        s: t[f"px_{s}"].to_numpy().astype(np.float64)
        for s in targets + ["SPY"]
        if f"px_{s}" in t.columns
    }
    out = []
    n = t.height
    for s in targets:
        if s not in px:
            continue
        for h in horizons:
            for lag in lags:
                a = np.full(n, np.nan)
                i = np.arange(n - lag - h)
                a[i] = px[s][i + lag + h] / px[s][i + lag] - 1
                if s != "SPY":
                    a[i] -= px["SPY"][i + lag + h] / px["SPY"][i + lag] - 1
                for f in feats:
                    x = t[f].to_numpy().astype(np.float64)
                    res: dict[str, Any] = {"feature": f, "target": s, "h": h, "lag": lag}
                    for nm, msk in (("dev", dev), ("test", ~dev)):
                        cors = []
                        for off in range(
                            0, h, max(1, h // 4)
                        ):  # several phases of the non-overlapping grid
                            sel = np.flatnonzero(msk)[off::h]
                            cors.append(_spearman(x[sel], a[sel]))
                        cors_a = np.array(cors, dtype=float)
                        res[f"{nm}_corr"] = (
                            float(np.nanmean(cors_a)) if np.isfinite(cors_a).any() else None
                        )
                        res[f"{nm}_n"] = int(msk.sum() // h)
                    out.append(res)
    return out


def quintiles(
    day: pl.DataFrame, etf: pl.DataFrame, feature: str, target: str, h: int, lag: int, dev_end: Any
) -> dict[str, Any]:
    """Mean forward return of the target by quintile of the feature (cut points from the development period)."""
    t = (
        day.select("as_of", feature)
        .join(etf.select("as_of", f"px_{target}", "px_SPY"), on="as_of", how="left")
        .sort("as_of")
    )
    x = t[feature].to_numpy().astype(np.float64)
    p = t[f"px_{target}"].to_numpy().astype(np.float64)
    b = t["px_SPY"].to_numpy().astype(np.float64)
    n = t.height
    a = np.full(n, np.nan)
    i = np.arange(n - lag - h)
    a[i] = (
        p[i + lag + h] / p[i + lag]
        - 1
        - ((b[i + lag + h] / b[i + lag] - 1) if target != "SPY" else 0)
    )
    dev = t["as_of"].to_numpy() <= np.datetime64(dev_end)
    q = np.nanquantile(x[dev], [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(x, q)
    res: dict[str, Any] = {
        "feature": feature,
        "target": target,
        "h": h,
        "lag": lag,
        "cuts": q.tolist(),
    }
    for nm, msk in (("dev", dev), ("test", ~dev)):
        res[nm] = [{"q": k + 1, "n": int((msk & (bins == k) & np.isfinite(x) & np.isfinite(a)).sum()),
                    "mean": float(np.nanmean(a[msk & (bins == k) & np.isfinite(x)])) if (msk & (bins == k) & np.isfinite(a)).any() else None}
                   for k in range(5)]  # fmt: skip
    return res


# ---------------------------------------------------------------- sector rotation
def rotation(
    etf: pl.DataFrame, dev_end: Any, horizons: tuple[int, ...] = (20, 60, 120)
) -> dict[str, Any]:
    """Peak-out / bottom-out events of each sector-industry-factor ETF versus the S&P 500, and what came next.

    peak-out of X: X's 3-month relative strength ranked in the top quarter of all rotation ETFs at some point in the
    last 20 sessions, and its 20-session relative strength has just turned negative (first day).
    bottom-out: mirror image (bottom quarter, 20-session relative strength just turned positive).
    For every other ETF Y: mean relative return over the next h sessions after X's events.
    """
    syms = [s for s in features.ROT_ETFS if f"px_{s}" in etf.columns]
    t = etf.select("as_of", "px_SPY", *[f"px_{s}" for s in syms]).sort("as_of")
    rs = {s: (t[f"px_{s}"] / t["px_SPY"]).to_numpy().astype(np.float64) for s in syms}
    n = t.height
    dates = t["as_of"].to_numpy()
    dev = dates <= np.datetime64(dev_end)

    def chg(x: np.ndarray, k: int) -> np.ndarray:
        o = np.full(n, np.nan)
        o[k:] = x[k:] / x[:-k] - 1
        return o

    rs63 = np.vstack([chg(rs[s], 63) for s in syms])
    rs20 = np.vstack([chg(rs[s], 20) for s in syms])
    with np.errstate(invalid="ignore"):
        rank = np.argsort(np.argsort(-np.nan_to_num(rs63, nan=-9), axis=0), axis=0) / max(
            1, len(syms) - 1
        )
    top = rank <= 0.25
    bot = rank >= 0.75
    recent_top = np.array([np.convolve(row.astype(int), np.ones(20, int))[:n] > 0 for row in top])
    recent_bot = np.array([np.convolve(row.astype(int), np.ones(20, int))[:n] > 0 for row in bot])
    neg, pos = rs20 < 0, rs20 > 0
    prev = lambda a: np.c_[np.zeros((a.shape[0], 1), bool), a[:, :-1]]  # noqa: E731
    peak = recent_top & neg & ~prev(neg)
    bottom = recent_bot & pos & ~prev(pos)
    fwd = {
        h: np.vstack([np.r_[rs[s][h:] / rs[s][:-h] - 1, np.full(h, np.nan)] for s in syms])
        for h in horizons
    }
    out: dict[str, Any] = {
        "symbols": syms,
        "names": {s: ETF_NAMES.get(s, s) for s in syms},
        "events": {},
    }
    for kind, ev in (("peak", peak), ("bottom", bottom)):
        res = []
        for i, x in enumerate(syms):
            idx = np.flatnonzero(ev[i])
            # avoid counting the same episode twice: at least 20 sessions apart
            keep, last = [], -999
            for j in idx:
                if j - last >= 20:
                    keep.append(j)
                    last = j
            idx = np.array(keep, dtype=int)
            for h in horizons:
                for nm, msk in (("dev", dev), ("test", ~dev)):
                    sel = idx[msk[idx]] if len(idx) else idx
                    if len(sel) == 0:
                        continue
                    for k, y in enumerate(syms):
                        v = fwd[h][k][sel]
                        v = v[np.isfinite(v)]
                        if len(v) == 0:
                            continue
                        res.append(
                            {
                                "from": x,
                                "to": y,
                                "h": h,
                                "period": nm,
                                "n": len(v),
                                "mean": float(v.mean()),
                                "win": float((v > 0).mean()),
                            }
                        )
        out["events"][kind] = res
    return out
