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
from pathlib import Path
from typing import Any

import polars as pl

from mw.report import dashboard as D
from mw.research import events, external, technical
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


def build_us(
    storage: Any,
    out_dir: Path,
    chunks: int = 8,
    min_dvol: float = 1e6,
    min_px: float = 1.0,
    only: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = D._read_years(storage, "prices_daily", "US").select(PRICE_COLS)
    vol = D._read_years(storage, "vol_indices", "US")
    fred = D._read_years(storage, "fred", "GLOBAL")
    etfs = etf_ids(storage)
    mkt = market_conditions(prices, vol, fred, etfs)
    del vol
    sp = sp500_flag(storage)
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
            (C("close") / C("close").shift(5).over("entity_id") - 1)
            .cast(pl.Float32)
            .alias("f_ret5"),
            (C("close") / C("hh252p") - 1).cast(pl.Float32).alias("f_dist_hi"),
            (C("atr") / C("close")).cast(pl.Float32).alias("f_atr_pct"),
            C("dvol20").cast(pl.Float32).alias("f_dvol20"),
            C("ret63").cast(pl.Float32).alias("f_ret63"),
        ).with_columns((C("close") * C("split_fac")).alias("raw_close"))
        df = events.forward_raw(df)
        df = bad_windows(df)
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
        # non-price conditions (order matters: shares before short interest)
        df = external.shares(storage, df)
        df = external.short_interest(storage, df)
        df = external.short_volume(storage, df)
        df = external.fails(storage, df)
        df = external.sec_events(storage, df)
        df = external.insiders(storage, df)
        df = df.join(mkt, on="as_of", how="left")
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
        keep = ["entity_id", "as_of", "close", "raw_close", "atr", "eligible", "in_sp500"] + conds + ["p_" + c[2:] for c in conds] + [
            c for c in df.columns if c.startswith("f_")] + [f"{k}_{h}" for h in events.HORIZONS for k in ("ret", "mfe", "mae")]  # fmt: skip
        df = df.filter(C("eligible")).select(keep)
        p = out_dir / f"part_{i}.parquet"
        df.write_parquet(p)
        parts.append(p)
        print(f"  part {i + 1}/{chunks}: rows {df.height:,}", flush=True)
        del df
        gc.collect()
    return parts


def finalize(parts: list[Path]) -> pl.DataFrame:
    df = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed").sort(
        ["entity_id", "as_of"]
    )
    df = events.add_excess(df)
    return events.vol_tier(df)
