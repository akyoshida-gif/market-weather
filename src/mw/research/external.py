"""Non-price conditions for the event study (VALIDATION_PLAN §2), all point-in-time.

Rule: a value is usable on trading day t only if it was published before t's close
(16:00 New York). known_date = publish date in New York, +1 day if published at/after 16:00.
Values are attached with an as-of join (latest known value on or before t).
"""

from __future__ import annotations

from datetime import time
from typing import Any

import polars as pl

from mw.report import dashboard as D

C = pl.col
NY = "America/New_York"
_CACHE: dict[str, pl.DataFrame] = {}
# first date each non-price condition can be evaluated (rows before it are "no data", not "false")
STARTS: dict[str, Any] = {}


def _start(conds: list[str], first: Any) -> None:
    for c in conds:
        STARTS[c] = first


def _cached(key: str, build: Any) -> pl.DataFrame:
    """Load + preprocess each source once per process (the study runs in chunks of stocks)."""
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def known_date(pub: pl.Expr) -> pl.Expr:
    local = pub.dt.convert_time_zone(NY)
    after_close = local.dt.time() >= pl.lit(time(16, 0))
    return (
        local.dt.date()
        + pl.when(after_close).then(pl.duration(days=1)).otherwise(pl.duration(days=0))
    ).alias("known")


def _asof(base: pl.DataFrame, feat: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Attach latest known feature row per entity (feat has entity_id, known + cols)."""
    feat = feat.sort("known").select("entity_id", "known", *cols)
    out = base.sort("as_of").join_asof(
        feat.rename({"known": "as_of"}), on="as_of", by="entity_id", strategy="backward"
    )
    return out.sort(["entity_id", "as_of"])


# ---------------------------------------------------------------- short sale volume (FINRA daily)
def short_volume(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        df = D._read_many(
            storage,
            [k for k in storage.list("silver/finra_regsho_daily/") if k.endswith(".parquet")],
            ["entity_id", "as_of", "short_volume_ratio", "published_at"],
        )
        df = df.select("entity_id", "as_of", "short_volume_ratio", "published_at").sort(
            ["entity_id", "as_of"]
        )
        df = df.with_columns(
            C("short_volume_ratio").rolling_mean(5).over("entity_id").alias("svr5"),
            C("short_volume_ratio").rolling_mean(60).over("entity_id").alias("svr60"),
            known_date(C("published_at")),
        ).with_columns(C("svr5").shift(10).over("entity_id").alias("svr5_10"))
        return df.select(
            "entity_id",
            "known",
            pl.col("svr5").cast(pl.Float32),
            pl.col("svr60").cast(pl.Float32),
            pl.col("svr5_10").cast(pl.Float32),
        )

    full = _cached("svr", build)
    _start(["c_svr_low", "c_svr_high", "c_svr_drop"], full["known"].min())
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    out = _asof(base, df, ["svr5", "svr60", "svr5_10"])
    return out.with_columns(
        (C("svr5") < C("svr60") - 0.08).fill_null(False).alias("c_svr_low"),
        (C("svr5") > C("svr60") + 0.08).fill_null(False).alias("c_svr_high"),
        (C("svr5") < C("svr5_10") - 0.10).fill_null(False).alias("c_svr_drop"),
    )


# ---------------------------------------------------------------- short interest (FINRA bi-monthly)
def short_interest(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        df = D._read_many(
            storage,
            [k for k in storage.list("silver/finra_short_interest/") if k.endswith(".parquet")],
            ["entity_id", "short_interest", "days_to_cover", "change_pct", "published_at"],
        )
        return df.select(
            "entity_id", "short_interest", "days_to_cover", "change_pct", "published_at"
        ).with_columns(known_date(C("published_at")))

    full = _cached("si", build)
    _start(["c_si_high", "c_dtc_high", "c_si_down", "c_si_up"], full["known"].min())
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    out = _asof(base, df, ["short_interest", "days_to_cover", "change_pct"])
    si_pct = C("short_interest") / C("shares_out")
    return out.with_columns(
        (si_pct >= 0.10).fill_null(False).alias("c_si_high"),
        (C("days_to_cover") >= 5).fill_null(False).alias("c_dtc_high"),
        (C("change_pct") <= -20).fill_null(False).alias("c_si_down"),
        (C("change_pct") >= 20).fill_null(False).alias("c_si_up"),
    )


# ---------------------------------------------------------------- fails-to-deliver
def fails(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        df = D._read_many(
            storage,
            [k for k in storage.list("silver/sec_ftd/") if k.endswith(".parquet")],
            ["entity_id", "period", "published_at", "fails"],
        )
        return (
            df.group_by("entity_id", "period", "published_at")
            .agg(C("fails").max().alias("ftd_max"))
            .with_columns(known_date(C("published_at")))
        )

    full = _cached("ftd", build)
    _start(["c_ftd_high"], full["known"].min())
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    out = _asof(base, df, ["ftd_max"])
    # fails are raw share counts; volume is split-adjusted -> compare with raw volume (vol / factor)
    return out.with_columns(
        ((C("ftd_max") / (C("vol20") / C("split_fac"))) >= 0.5).fill_null(False).alias("c_ftd_high")
    )


# ---------------------------------------------------------------- SEC events: earnings (8-K 2.02), other items, offerings, 13D
def sec_events(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        keys = [k for k in storage.list("silver/sec_filings/") if k.endswith(".parquet")]
        d = D._read_many(storage, keys, ["entity_id", "form", "items", "published_at"]).filter(
            C("entity_id").str.starts_with("US:")
        )
        return d.with_columns(known_date(C("published_at")).alias("known")).drop("published_at")

    full = _cached("sec", build)
    _start(["c_earn_day", "c_earn_gap_up", "c_earn_strong", "c_earn_weak", "c_ev_contract", "c_ev_listing_issue",
            "c_ev_unregistered_sale", "c_ev_officer_change", "c_ev_offering", "c_ev_13d"], full["known"].min())  # fmt: skip
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    flags = {
        "earn": C("form").str.starts_with("8-K") & C("items").str.contains("2.02"),
        "item_101": C("form").str.starts_with("8-K") & C("items").str.contains("1.01"),
        "item_301": C("form").str.starts_with("8-K") & C("items").str.contains("3.01"),
        "item_302": C("form").str.starts_with("8-K") & C("items").str.contains("3.02"),
        "item_502": C("form").str.starts_with("8-K") & C("items").str.contains("5.02"),
        "offering": C("form").str.contains(r"^(S-1|S-3|424B|F-1|F-3)"),
        "stake_13d": C("form").str.contains("13D"),
    }
    ev = df.with_columns([e.fill_null(False).alias(k) for k, e in flags.items()])
    out = base
    for k in flags:
        d = (
            ev.filter(C(k))
            .select("entity_id", "known")
            .unique()
            .with_columns(C("known").alias(f"last_{k}"))
        )
        out = _asof(out, d, [f"last_{k}"])
    gap, ret1, vr, clv = (
        C("gap"),
        C("close") / C("pc") - 1,
        C("vr"),
        (C("close") - C("low")) / (C("high") - C("low")).clip(lower_bound=1e-9),
    )
    # earnings reaction day: first session on/after the known date (as-of join gives last known <= t)
    earn_day = (C("last_earn") == C("as_of")) | (
        (C("last_earn") < C("as_of")) & (C("last_earn") > C("prev_as_of"))
    )
    recent = lambda k, days: ((C("as_of") - C(f"last_{k}")).dt.total_days() <= days).fill_null(
        False
    )  # noqa: E731
    return out.with_columns(
        earn_day.fill_null(False).alias("c_earn_day"),
        (earn_day & (gap >= 0.03)).fill_null(False).alias("c_earn_gap_up"),
        (earn_day & (ret1 >= 0.05) & (vr >= 2) & (clv >= 0.6))
        .fill_null(False)
        .alias("c_earn_strong"),
        (earn_day & (ret1 <= -0.05)).fill_null(False).alias("c_earn_weak"),
        recent("item_101", 7).alias("c_ev_contract"),
        recent("item_301", 7).alias("c_ev_listing_issue"),
        recent("item_302", 7).alias("c_ev_unregistered_sale"),
        recent("item_502", 7).alias("c_ev_officer_change"),
        recent("offering", 7).alias("c_ev_offering"),
        recent("stake_13d", 7).alias("c_ev_13d"),
    )


# ---------------------------------------------------------------- insider purchases (Form 4, code P)
INS_WINDOW = 30  # calendar days


def insiders(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    """Open-market purchases known within the last 30 calendar days (t-29 .. t).

    Each (owner, known day) is spread over the 30 days it stays "recent"; per stock-day we then sum the
    value and count DISTINCT owners (the same person buying twice is one buyer).
    """

    def build() -> pl.DataFrame:
        d = D._read_many(
            storage,
            [k for k in storage.list("silver/sec_insider/") if k.endswith(".parquet")],
            ["entity_id", "code", "shares", "price", "published_at", "title", "owner"],
        )
        d = d.filter((C("code") == "P") & C("entity_id").str.starts_with("US:"))
        d = d.with_columns(
            (C("shares") * C("price")).alias("value"),
            known_date(C("published_at")),
            C("title")
            .fill_null("")
            .str.to_uppercase()
            .str.contains("CEO|CFO|CHIEF EXECUTIVE|CHIEF FINANCIAL")
            .alias("ceo_cfo"),
        )
        return d.group_by("entity_id", "known", "owner").agg(
            C("value").sum().alias("v"), C("ceo_cfo").any().alias("cc")
        )

    full = _cached("ins", build)
    _start(["c_ins_buy", "c_ins_cluster", "c_ins_ceo"], full["known"].min())
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    spread = (
        df.with_columns(pl.int_ranges(0, INS_WINDOW).alias("off"))
        .explode("off")
        .with_columns((C("known") + pl.duration(days=C("off"))).alias("as_of"))
        .group_by("entity_id", "as_of")
        .agg(
            C("v").sum().alias("ins_value30"),
            C("owner").n_unique().alias("ins_buyers30"),
            C("cc").any().alias("ins_cc30"),
        )
    )
    out = base.join(spread, on=["entity_id", "as_of"], how="left")
    return out.with_columns(
        (C("ins_value30") >= 100_000).fill_null(False).alias("c_ins_buy"),
        (C("ins_buyers30") >= 2).fill_null(False).alias("c_ins_cluster"),
        C("ins_cc30").fill_null(False).alias("c_ins_ceo"),
    ).drop("ins_value30", "ins_buyers30", "ins_cc30")


# ---------------------------------------------------------------- shares outstanding -> market cap, dilution
def shares(storage: Any, base: pl.DataFrame) -> pl.DataFrame:
    """Market cap and dilution on a consistent share basis.

    Prices are split-adjusted to today (yfinance) while SEC share counts are as reported. base carries
    split_fac = product of split ratios AFTER each row's date, so raw price = close * split_fac and
    shares on today's basis = reported shares * split_fac at the report's period end.
    """

    def build() -> pl.DataFrame:
        d = D._read_many(
            storage,
            [k for k in storage.list("silver/sec_shares/") if k.endswith(".parquet")],
            ["entity_id", "as_of", "shares_out", "published_at"],
        )
        d = d.filter(C("entity_id").str.starts_with("US:")).select(
            "entity_id", "as_of", "shares_out", "published_at"
        )
        return (
            d.sort(["entity_id", "as_of"])
            .unique(["entity_id", "as_of"], keep="last")
            .sort(["entity_id", "as_of"])
            .with_columns(
                known_date(C("published_at")),
                C("shares_out").shift(4).over("entity_id").alias("shares_out_1y"),
                C("as_of").shift(4).over("entity_id").alias("as_of_1y"),
            )
        )

    full = _cached("shares", build)
    _start(
        ["c_cap_mega", "c_cap_large", "c_cap_mid", "c_cap_small", "c_cap_micro", "c_dilution"],
        full["known"].min(),
    )
    df = full.filter(C("entity_id").is_in(base["entity_id"].unique()))
    fac = base.select("entity_id", "as_of", "split_fac").sort("as_of")

    def fac_at(col: str, name: str) -> pl.DataFrame:
        return (
            df.select("entity_id", "known", col)
            .filter(C(col).is_not_null())
            .sort(col)
            .join_asof(fac.rename({"as_of": col}), on=col, by="entity_id", strategy="nearest")
            .select("entity_id", "known", C("split_fac").alias(name))
        )

    df = (
        df.join(fac_at("as_of", "fac_p"), on=["entity_id", "known"], how="left")
        .join(fac_at("as_of_1y", "fac_p1y"), on=["entity_id", "known"], how="left")
        .unique(["entity_id", "known"], keep="last")
    )
    df = df.with_columns(
        (C("shares_out") * C("fac_p")).alias("shares_adj"),
        (C("shares_out_1y") * C("fac_p1y")).alias("shares_adj_1y"),
    )
    out = _asof(base, df, ["shares_out", "shares_adj", "shares_adj_1y"])
    mcap = C("shares_adj") * C("close")  # same basis as the adjusted close
    return out.with_columns(
        mcap.alias("mcap"),
        (mcap >= 200e9).fill_null(False).alias("c_cap_mega"),
        ((mcap >= 10e9) & (mcap < 200e9)).fill_null(False).alias("c_cap_large"),
        ((mcap >= 2e9) & (mcap < 10e9)).fill_null(False).alias("c_cap_mid"),
        ((mcap >= 0.3e9) & (mcap < 2e9)).fill_null(False).alias("c_cap_small"),
        (mcap < 0.3e9).fill_null(False).alias("c_cap_micro"),
        (C("shares_adj") >= 1.2 * C("shares_adj_1y")).fill_null(False).alias("c_dilution"),
    )


EXT_LABELS = {
    "svr_low": "空売り比率が平常より低い（5日平均が60日平均より8pt以上低い）", "svr_high": "空売り比率が平常より高い（8pt以上）",
    "svr_drop": "空売り比率が急低下（5日平均が10日前より10pt以上低下）",
    "si_high": "空売り残高が発行済株式の10%以上", "dtc_high": "カバー日数5日以上", "si_down": "空売り残高が前回比20%以上減少（買い戻し）",
    "si_up": "空売り残高が前回比20%以上増加", "ftd_high": "受け渡し不履行が出来高の半日分以上",
    "earn_day": "決算発表への反応日", "earn_gap_up": "決算で窓開け上昇（+3%以上）", "earn_strong": "決算で強い上昇（+5%以上・出来高2倍・高値引け）",
    "earn_weak": "決算で下落（−5%以下）", "ev_contract": "重要な契約の開示（7日以内）", "ev_listing_issue": "上場基準違反の通知（7日以内）",
    "ev_unregistered_sale": "未登録株式の発行＝希薄化（7日以内）", "ev_officer_change": "役員の異動（7日以内）",
    "ev_offering": "増資の届出（S-1/S-3/424B、7日以内）", "ev_13d": "5%超の大量保有（13D、7日以内）",
    "ins_buy": "インサイダーの市場での買い（30日で10万ドル以上）", "ins_cluster": "インサイダー2人以上の買い（30日）", "ins_ceo": "CEO・CFOの買い（30日）",
    "cap_mega": "時価総額2000億ドル以上", "cap_large": "時価総額100億〜2000億ドル", "cap_mid": "時価総額20億〜100億ドル",
    "cap_small": "時価総額3億〜20億ドル", "cap_micro": "時価総額3億ドル未満", "dilution": "発行済株式数が1年で20%以上増加",
}  # fmt: skip
