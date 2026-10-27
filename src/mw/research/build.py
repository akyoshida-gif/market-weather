"""Assemble the event-study dataset for the US market in memory-bounded chunks.

Steps per chunk of stocks: technical conditions -> forward returns -> non-price conditions (as-of joins)
-> previous-row state of every condition (computed BEFORE dropping ineligible rows, so a signal's onset
is judged against the stock's real previous session) -> keep eligible rows -> write a part file.
Then: same-day means (excess returns, one per universe) and volatility tiers over all parts.

Price basis: yfinance prices are split-adjusted to today. split_fac (product of split ratios after
each row) restores the price of the day (close * split_fac) for the $1 rule, cost tiers and market cap.
"""

from __future__ import annotations

import gc
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from mw.report import dashboard as D
from mw.research import events, external, features, macro, technical
from mw.strategy import regime

C = pl.col
PRICE_COLS = ["entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"]
JUMP = 0.6  # one-day move treated as a data error (VALIDATION_PLAN §1)


def market_conditions(
    prices: pl.DataFrame, vol: pl.DataFrame, fred: pl.DataFrame, etfs: set[str]
) -> pl.DataFrame:
    # regime.us_regime applies the publication-time rule to FRED itself (ADR-0016)
    r = regime.us_regime(prices.select("entity_id", "as_of", "close"), vol, fred)
    br = (
        prices.filter(~C("entity_id").is_in(list(etfs)))
        .select("entity_id", "as_of", "close")
        .sort(["entity_id", "as_of"])
        .with_columns(C("close").rolling_mean(50).over("entity_id").alias("ma50"))
        .filter(C("ma50").is_not_null())
        .group_by("as_of")
        .agg((C("close") > C("ma50")).mean().alias("breadth"))
        .sort("as_of")
        .with_columns((C("breadth") - C("breadth").shift(10)).alias("breadth_chg10"))
    )
    r = r.join(br, on="as_of", how="left")
    # BAA corporate spread has full history (ICE HY on FRED keeps only 3 years)
    baa = fred.filter(C("entity_id") == "MACRO:FRED:BAA10Y").with_columns(
        external.known_date(C("published_at"))
    )
    baa = (
        baa.group_by("known")
        .agg(C("value").last().alias("baa"))
        .sort("known")
        .rename({"known": "as_of"})
    )
    r = (
        r.sort("as_of")
        .join_asof(baa, on="as_of", strategy="backward")
        .with_columns((C("baa") - C("baa").shift(20)).alias("baa_chg20"))
    )
    conds = {
        "mkt_up200": C("spy_up200"), "mkt_up50": C("spy_up50"), "mkt_breadth_hi": C("breadth") > 0.5,
        "mkt_breadth_up": C("breadth_chg10") > 0, "mkt_vix_lt20": C("vix") < 20, "mkt_vix_gt30": C("vix") > 30,
        "mkt_vix_term_lt1": C("vix_term") < 1, "mkt_credit_easing": C("baa_chg20") < 0, "mkt_y10_falling": C("y10_chg20") < 0,
        "mkt_usd_falling": C("usd_chg20") < 0, "mkt_oil_rising": C("oil_chg20") > 0, "mkt_risk_on_rotation": C("def_cyc_chg20") < 0,
        "mkt_small_lead": C("iwm_spy_chg20") > 0, "mkt_broad_lead": C("rsp_spy_chg20") > 0, "mkt_hyg_ief_up": C("hyg_ief_chg20") > 0,
        "mkt_curve_pos": C("curve") > 0,
    }  # fmt: skip
    return r.select("as_of", *[e.fill_null(False).alias("c_" + k) for k, e in conds.items()])


MKT_LABELS = {
    "mkt_up200": "市場：S&P500が200日線の上", "mkt_up50": "市場：S&P500が50日線の上", "mkt_breadth_hi": "市場：50日線より上の銘柄が50%超",
    "mkt_breadth_up": "市場：上昇銘柄の割合が増加中", "mkt_vix_lt20": "市場：VIX 20未満", "mkt_vix_gt30": "市場：VIX 30超（恐怖）",
    "mkt_vix_term_lt1": "市場：VIX期近/期先が1未満（平常）", "mkt_credit_easing": "市場：社債スプレッド縮小（20日）",
    "mkt_y10_falling": "市場：10年金利低下（20日）", "mkt_usd_falling": "市場：ドル安（20日）", "mkt_oil_rising": "市場：原油高（20日）",
    "mkt_risk_on_rotation": "市場：景気敏感株がディフェンシブより強い", "mkt_small_lead": "市場：小型株が大型株より強い",
    "mkt_broad_lead": "市場：均等加重が時価総額加重より強い", "mkt_hyg_ief_up": "市場：ハイイールド債が国債より強い", "mkt_curve_pos": "市場：10年−2年がプラス",
}  # fmt: skip


def sp500_flag(storage: Any) -> pl.DataFrame:
    iv = storage.read_parquet("silver/sp500_membership/latest.parquet")
    return iv.with_columns(
        (pl.lit("US:") + C("symbol").str.replace_all("-", ".")).alias("entity_id")
    ).select("entity_id", "start", "end")


def etf_ids(storage: Any) -> set[str]:
    keys = [
        k for k in storage.list("silver/universe_membership/market=US/") if k.endswith(".parquet")
    ]
    u = storage.read_parquet(sorted(keys)[-1])
    return set(u.filter(C("kind") == "etf")["entity_id"].to_list())


def split_factor() -> pl.Expr:
    """Product of split ratios strictly after each row (per stock); 1.0 when no later split."""
    lg = pl.when(C("splits").fill_null(0) > 0).then(C("splits").log()).otherwise(0.0)
    return (lg.sum().over("entity_id") - lg.cum_sum().over("entity_id")).exp().alias("split_fac")


def bad_windows(df: pl.DataFrame) -> pl.DataFrame:
    """Null the forward metrics of horizon h when a data-error day (split day or |move| > 60%)
    lies in the window used: the 5 sessions before the signal through t+h."""
    jump = (
        (((C("close") / C("pc") - 1).abs() > JUMP) | (C("splits").fill_null(0) != 0))
        .fill_null(False)
        .cast(pl.Int8)
    )
    df = df.with_columns(jump.alias("_bad"))
    df = df.with_columns(
        C("_bad").rolling_sum(6, min_samples=1).over("entity_id").alias("_bad_back")
    )
    out = []
    for h in events.HORIZONS:
        fwd = (
            C("_bad")
            .reverse()
            .rolling_max(h, min_samples=1)
            .reverse()
            .shift(-1)
            .over("entity_id")
            .fill_null(0)
        )
        bad = (fwd > 0) | (C("_bad_back") > 0)
        out += [
            pl.when(bad).then(None).otherwise(C(f"{k}_{h}")).alias(f"{k}_{h}")
            for k in ("ret", "mfe", "mae")
        ]
    return df.with_columns(out)


KEEP_BASE = ["entity_id", "as_of", "close", "raw_close", "atr", "eligible", "in_sp500", "contig"]


def external_features() -> list[pl.Expr]:
    """Numeric versions of the non-price data (after external.* joins)."""
    days = lambda col: (C("as_of") - C(col)).dt.total_days()  # noqa: E731
    return [
        C("mcap").log10().alias("f_mcap"),
        (C("short_interest") / C("shares_out")).alias("f_si_pct"),
        C("days_to_cover").alias("f_dtc"),
        C("svr5").alias("f_svr5"),
        (C("svr5") - C("svr60")).alias("f_svr_chg"),
        (C("ftd_max") / (C("vol20") / C("split_fac"))).alias("f_ftd"),
        C("ins_value30").fill_null(0).log1p().alias("f_ins_value"),
        days("last_earn").alias("f_days_since_earn"),
        (C("shares_adj") / C("shares_adj_1y") - 1).alias("f_share_growth"),
    ]


def sic_table(storage: Any) -> pl.DataFrame:
    key = "silver/sec_company/market=US/latest.parquet"
    if not storage.exists(key):
        return pl.DataFrame(schema={"entity_id": pl.String, "k_sic2": pl.Int16})
    s = storage.read_parquet(key).filter(C("entity_id").is_not_null() & C("sic").is_not_null())
    return s.select(
        "entity_id", C("sic").str.slice(0, 2).cast(pl.Int16, strict=False).alias("k_sic2")
    ).unique("entity_id")


def build_us(
    storage: Any,
    out_dir: Path,
    chunks: int = 8,
    min_dvol: float = 1e6,
    min_px: float = 1.0,
    only: int | None = None,
) -> list[Path]:
    """Writes part_<i>.parquet (stock rows), day.parquet (market-wide m_* per session), etf.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = D._read_years(storage, "prices_daily", "US").select(PRICE_COLS)
    vol = D._read_years(storage, "vol_indices", "US")
    fred = D._read_years(storage, "fred", "GLOBAL")
    etfs = etf_ids(storage)
    mkt = market_conditions(prices, vol, fred, etfs)
    del vol, fred
    etf = features.etf_returns(prices, sorted({x.split(":", 1)[1] for x in etfs} | {"SPY"}))
    etf.write_parquet(out_dir / "etf.parquet")
    rs_long = features.rs_table(etf)
    spy = etf.select("as_of", C("px_SPY").alias("spy_close"))
    br = (
        prices.filter(~C("entity_id").is_in(list(etfs))).select("entity_id", "as_of", "close").sort(["entity_id", "as_of"])
        .with_columns(C("close").rolling_mean(50).over("entity_id").alias("ma50")).filter(C("ma50").is_not_null())
        .group_by("as_of").agg((C("close") > C("ma50")).mean().alias("BREADTH")).sort("as_of")
    )  # fmt: skip
    day, day_labels = macro.day_table(storage, prices, etf, br)
    day.write_parquet(out_dir / "day.parquet")
    (out_dir / "day_labels.json").write_text(
        json.dumps(day_labels, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  市場全体の表：{day.height:,}日・{day.width - 1}項目", flush=True)
    sp = sp500_flag(storage)
    sic = sic_table(storage)
    ents = sorted(set(prices["entity_id"].unique().to_list()) - etfs)
    parts: list[Path] = []
    for i in range(chunks):
        if only is not None and i >= only:
            break
        sel = ents[i::chunks]
        df = prices.filter(C("entity_id").is_in(sel)).sort(["entity_id", "as_of"])
        df = technical.conditions(df, "US")
        df = df.with_columns(
            (C("open") / C("pc") - 1).alias("gap"),
            C("as_of").shift(1).over("entity_id").alias("prev_as_of"),
            split_factor(),
        ).with_columns((C("close") * C("split_fac")).alias("raw_close"))
        df = features.stock_features(df, spy)
        df = features.sector_assign(df, etf, rs_long)
        df = events.forward_raw(df)
        df = features.r_outcomes(df)
        df = bad_windows(df)
        # R outcomes share the data-error windows of the plain returns
        df = df.with_columns(
            [pl.when(C(f"ret_{h}").is_null()).then(None).otherwise(C(f"r_{s}_{h}_{k}")).alias(f"r_{s}_{h}_{k}")
             for s in features.STOPS for h in features.R_HORIZONS for k in ("fin", "mfe")]
        )  # fmt: skip
        # dollar volume is split-invariant; the $1 rule uses the price of the day
        df = df.with_columns(
            (
                (C("dvol20") >= min_dvol)
                & (C("raw_close") >= min_px)
                & (C("_bad_back") == 0)
                & C("atr").is_not_null()
            )
            .fill_null(False)
            .alias("eligible")
        )
        # the previous session was also eligible (onsets of numeric conditions need a real previous row)
        df = df.with_columns(
            C("eligible").shift(1).over("entity_id").fill_null(False).alias("contig")
        )
        # non-price conditions (order matters: shares before short interest)
        df = external.shares(storage, df)
        df = external.short_interest(storage, df)
        df = external.short_volume(storage, df)
        df = external.fails(storage, df)
        df = external.sec_events(storage, df)
        df = external.insiders(storage, df)
        df = df.with_columns([e.cast(pl.Float32) for e in external_features()])
        df = df.join(mkt, on="as_of", how="left").join(sic, on="entity_id", how="left")
        # S&P 500 point-in-time membership
        m = (
            df.select("entity_id", "as_of")
            .join(sp, on="entity_id", how="inner")
            .filter((C("as_of") >= C("start")) & (C("end").is_null() | (C("as_of") < C("end"))))
            .select("entity_id", "as_of")
            .unique()
            .with_columns(pl.lit(True).alias("in_sp500"))
        )
        df = (
            df.join(m, on=["entity_id", "as_of"], how="left")
            .with_columns(C("in_sp500").fill_null(False))
            .sort(["entity_id", "as_of"])
        )
        conds = [c for c in df.columns if c.startswith("c_")]
        df = df.with_columns([C(c).fill_null(False) for c in conds])
        # previous session's state of each condition, taken before ineligible rows are dropped
        df = df.with_columns(
            [C(c).shift(1).over("entity_id").fill_null(False).alias("p_" + c[2:]) for c in conds]
        )
        fcols = [c for c in df.columns if c.startswith("f_")]
        df = df.with_columns(
            [
                pl.when(C(c).is_infinite() | C(c).is_nan())
                .then(None)
                .otherwise(C(c))
                .cast(pl.Float32)
                .alias(c)
                for c in fcols
            ]
        )
        keep = KEEP_BASE + ["k_sic2"] + conds + ["p_" + c[2:] for c in conds] + fcols + [
            f"{k}_{h}" for h in events.HORIZONS for k in ("ret", "mfe", "mae")] + [
            c for c in df.columns if c.startswith(("r_", "rp_"))]  # fmt: skip
        df = df.filter(C("eligible")).select(keep)
        p = out_dir / f"part_{i}.parquet"
        df.write_parquet(p)
        parts.append(p)
        print(f"  part {i + 1}/{chunks}: rows {df.height:,}", flush=True)
        del df
        gc.collect()
    return parts


RANKED = {
    "f_ret20": "f_rank_ret20",
    "f_ret63": "f_rank_ret63",
    "f_ret126": "f_rank_ret126",
    "f_dvol20": "f_rank_dvol",
}


def finalize(parts: list[Path]) -> list[Path]:
    """Cross-sectional pieces computed over all parts without loading everything at once:
    same-day means (excess returns), cross-sectional ranks, volatility tiers. Parts are rewritten in place."""
    scan = pl.scan_parquet([str(p) for p in parts])
    rets = [f"ret_{h}" for h in events.HORIZONS]
    day_all = scan.group_by("as_of").agg([C(r).mean().alias(f"_d_{r}") for r in rets]).collect()
    day_sp = (
        scan.filter(C("in_sp500"))
        .group_by("as_of")
        .agg([C(r).mean().alias(f"_s_{r}") for r in rets])
        .collect()
    )
    ranks = (
        scan.select("entity_id", "as_of", *RANKED)
        .with_columns(
            [
                (C(a).rank() / C(a).count()).over("as_of").cast(pl.Float32).alias(b)
                for a, b in RANKED.items()
            ]
        )
        .select("entity_id", "as_of", *RANKED.values())
        .collect()
    )
    v = C("atr") / C("close")
    q = (
        scan.filter(C("as_of") <= date(2024, 12, 31))
        .select(
            v.quantile(0.25).alias("a"), v.quantile(0.5).alias("b"), v.quantile(0.75).alias("c")
        )
        .collect()
        .row(0)
    )
    for p in parts:
        df = (
            pl.read_parquet(p)
            .join(day_all, on="as_of", how="left")
            .join(day_sp, on="as_of", how="left")
            .join(ranks, on=["entity_id", "as_of"], how="left")
        )
        df = df.with_columns(
            [
                (C(f"ret_{h}") - C(f"_d_ret_{h}")).cast(pl.Float32).alias(f"xret_{h}")
                for h in events.HORIZONS
            ]
            + [
                (C(f"ret_{h}") - C(f"_s_ret_{h}")).cast(pl.Float32).alias(f"xsp_{h}")
                for h in events.HORIZONS
            ]
            + [
                pl.when(v <= q[0])
                .then(1)
                .when(v <= q[1])
                .then(2)
                .when(v <= q[2])
                .then(3)
                .otherwise(4)
                .cast(pl.Int8)
                .alias("vol_tier")
            ]
        )
        df = df.drop([c for c in df.columns if c.startswith(("_d_", "_s_"))])
        df.sort(["entity_id", "as_of"]).write_parquet(p)
        del df
        gc.collect()
    return parts
