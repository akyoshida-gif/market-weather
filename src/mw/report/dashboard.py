"""Build the dashboard payload (JSON-serialisable dict) from silver tables (read-only).

Everything shown is real collected data. Model forecasts do not exist yet (Phase 2); the
payload says so explicitly. Signal statistics are backtests on the current universe
(survivorship-biased: delisted names are missing).
"""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import polars as pl

from mw.core.storage import GuardedStorage
from mw.signals import rules

US_SECTORS = {"XLK": "テクノロジー", "XLF": "金融", "XLV": "ヘルスケア", "XLY": "一般消費財", "XLP": "生活必需品",
              "XLE": "エネルギー", "XLI": "資本財", "XLB": "素材", "XLU": "公益", "XLRE": "不動産", "XLC": "通信",
              "SMH": "半導体", "IGV": "ソフトウェア", "XBI": "バイオ", "KRE": "地銀", "ITA": "防衛", "GDX": "金鉱",
              "URA": "ウラン", "TAN": "太陽光"}  # fmt: skip
JP_SECTORS = {"1617": "食品", "1618": "エネルギー資源", "1619": "建設・資材", "1620": "素材・化学", "1621": "医薬品",
              "1622": "自動車・輸送機", "1623": "鉄鋼・非鉄", "1624": "機械", "1625": "電機・精密", "1626": "情報通信",
              "1627": "電力・ガス", "1628": "運輸・物流", "1629": "商社・卸売", "1630": "小売", "1631": "銀行",
              "1632": "金融（除く銀行）", "1633": "不動産"}  # fmt: skip
BENCH = {"US": "US:SPY", "JP": "JP:1306"}
ITEM_NAMES = {"1.01": "重要な契約", "1.02": "契約終了", "1.03": "破産", "2.01": "買収・売却完了", "2.02": "決算発表",
              "2.03": "債務の発生", "2.05": "リストラ費用", "2.06": "減損", "3.01": "上場廃止・基準違反の通知",
              "3.02": "未登録株式の発行（希薄化）", "3.03": "株主権利の変更", "4.01": "監査人の交代",
              "4.02": "過去決算の訂正", "5.01": "支配権の変更", "5.02": "役員の異動", "5.03": "定款変更（株式併合など）",
              "5.07": "株主総会の結果", "7.01": "開示（Reg FD）", "8.01": "その他の重要事象",
              "9.01": "財務諸表・添付資料"}  # fmt: skip


def _read_many(
    storage: GuardedStorage, keys: list[str], columns: list[str] | None = None
) -> pl.DataFrame:
    if not keys:
        return pl.DataFrame()
    with ThreadPoolExecutor(max_workers=16) as ex:
        frames = list(ex.map(lambda k: storage.read_parquet(k, columns=columns), sorted(keys)))
    return pl.concat(frames, how="diagonal_relaxed")


def _read_years(storage: GuardedStorage, table: str, market: str) -> pl.DataFrame:
    keys = [k for k in storage.list(f"silver/{table}/market={market}/") if k.endswith(".parquet")]
    return _read_many(storage, keys)


def _read_dates(storage: GuardedStorage, table: str, market: str, since: date) -> pl.DataFrame:
    keys = [k for k in storage.list(f"silver/{table}/market={market}/") if k.endswith(".parquet")]
    pick = []
    for k in keys:
        stem = k.rsplit("/", 1)[-1].replace(".parquet", "").replace("date=", "")
        try:
            d = date.fromisoformat(stem[:10])
        except ValueError:
            continue
        if d >= since:
            pick.append(k)
    return (
        pl.concat([storage.read_parquet(k) for k in sorted(pick)], how="diagonal_relaxed")
        if pick
        else pl.DataFrame()
    )


def _f(v: Any, nd: int = 4) -> Any:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), nd) if isinstance(v, float) else v


def _series(df: pl.DataFrame, col: str, n: int = 260) -> list[list[Any]]:
    d = df.sort("as_of").tail(n)
    return [
        [a.isoformat(), _f(b)] for a, b in zip(d["as_of"], d[col], strict=True) if b is not None
    ]


# ---------------------------------------------------------------- market level
def market_kpis(
    prices: pl.DataFrame, market: str, vol: pl.DataFrame, fred: pl.DataFrame
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(name: str, df: pl.DataFrame, col: str, fmt: str, note: str) -> None:
        if df.height:
            out.append({"name": name, "fmt": fmt, "note": note, "series": _series(df, col)})

    b = prices.filter(pl.col("entity_id") == BENCH[market])
    add(
        "S&P 500（SPY）" if market == "US" else "TOPIX（1306）",
        b,
        "close",
        "px",
        "指数連動ETFの終値",
    )
    if market == "JP":
        add(
            "日経225（1321）",
            prices.filter(pl.col("entity_id") == "JP:1321"),
            "close",
            "px",
            "指数連動ETFの終値",
        )
    # breadth: share of common stocks above 50-day MA
    br = breadth(prices)
    add(
        "50日線より上の銘柄の割合",
        br,
        "pct_above50",
        "pct",
        "高いほど幅広く強い。指数上昇中に低下したら一部銘柄頼み",
    )
    if market == "US" and vol.height:
        v = vol.pivot(on="entity_id", index="as_of", values="close").sort("as_of")
        for sym, nm, note in (
            ("IDX:US:VIX", "VIX", "20前後が平常、30超で恐怖"),
            ("IDX:US:VVIX", "VVIX", "VIXの変動率。急騰はヘッジ需要の急増"),
        ):
            if sym in v.columns:
                add(nm, v.select("as_of", pl.col(sym).alias("x")), "x", "num", note)
        if {"IDX:US:VIX", "IDX:US:VIX3M"} <= set(v.columns):
            add(
                "VIX 期近/期先",
                v.select("as_of", (pl.col("IDX:US:VIX") / pl.col("IDX:US:VIX3M")).alias("x")),
                "x",
                "num2",
                "1未満＝平常、1超＝目先のパニック",
            )
    if fred.height:
        f = fred.pivot(
            on="entity_id", index="as_of", values="value", aggregate_function="last"
        ).sort("as_of")
        spec = [("MACRO:FRED:BAMLH0A0HYM2", "HYスプレッド（%）", "num2", "低格付け社債の上乗せ金利。拡大は信用不安"),
                ("MACRO:FRED:DGS10", "米10年債利回り（%）", "num2", "長期金利"),
                ("MACRO:FRED:T10Y2Y", "米10年−2年（%）", "num2", "マイナスは逆イールド"),
                ("MACRO:FRED:DEXJPUS", "ドル円", "num2", "週1回更新（H.10）")]  # fmt: skip
        for sym, nm, fmt, note in spec:
            if market == "JP" and sym != "MACRO:FRED:DEXJPUS":
                continue
            if market == "US" and sym == "MACRO:FRED:DEXJPUS":
                continue
            if sym in f.columns:
                add(nm, f.select("as_of", pl.col(sym).alias("x")).drop_nulls(), "x", fmt, note)
    return out


def breadth(prices: pl.DataFrame) -> pl.DataFrame:
    p = prices.filter(pl.col("kind") == "common") if "kind" in prices.columns else prices
    p = p.sort(["entity_id", "as_of"]).with_columns(
        pl.col("close").rolling_mean(50).over("entity_id").alias("ma50")
    )
    return (
        p.filter(pl.col("ma50").is_not_null())
        .group_by("as_of")
        .agg((pl.col("close") > pl.col("ma50")).mean().alias("pct_above50"), pl.len().alias("n"))
        .filter(pl.col("n") > 50)
        .sort("as_of")
    )


def rrg(prices: pl.DataFrame, market: str, units: dict[str, str]) -> list[dict[str, Any]]:
    bench = prices.filter(pl.col("entity_id") == BENCH[market]).select(
        "as_of", pl.col("close").alias("b")
    )
    out = []
    for sym, name in units.items():
        eid = f"{market}:{sym}"
        p = (
            prices.filter(pl.col("entity_id") == eid)
            .select("as_of", "close")
            .join(bench, on="as_of")
            .sort("as_of")
        )
        if p.height < 120:
            continue
        p = p.with_columns((pl.col("close") / pl.col("b")).alias("rs"))
        p = p.with_columns((100 * pl.col("rs") / pl.col("rs").rolling_mean(50)).alias("ratio"))
        p = p.with_columns(
            (100 * pl.col("ratio") / pl.col("ratio").shift(10)).alias("mom")
        ).drop_nulls()
        # normalise around 100 for display
        tail = p.tail(25).gather_every(5)
        pts = [[_f(x, 3), _f(y, 3)] for x, y in zip(tail["ratio"], tail["mom"], strict=True)]
        x, y = pts[-1]
        quad = ("主導" if y >= 100 else "失速") if x >= 100 else ("改善" if y >= 100 else "出遅れ")
        last = p["close"][-1]
        rets = {}
        for h in (5, 20, 60):
            if p.height > h:
                r = last / p["close"][-1 - h] - 1
                rb = p["b"][-1] / p["b"][-1 - h] - 1
                rets[str(h)] = _f((r - rb) * 100, 2)
        out.append({"id": sym, "name": name, "tail": pts, "quad": quad, "excess": rets})
    return out


# ---------------------------------------------------------------- signals
def signal_payload(prices: pl.DataFrame, market: str, min_dollar_vol: float) -> dict[str, Any]:
    base = prices.select("entity_id", "as_of", "open", "high", "low", "close", "volume")
    sig = rules.signals(base)
    ev = rules.events(sig)
    bench = (
        prices.filter(pl.col("entity_id") == BENCH[market])
        .sort("as_of")
        .select(
            "as_of",
            (pl.col("close").shift(-rules.HORIZON) / pl.col("close") - 1).alias("bench_fwd_ret"),
        )
    )
    sc = rules.score(ev, bench)
    bucket = (
        pl.when(pl.col("close") < 5)
        .then(pl.lit("$5未満"))
        .when(pl.col("close") < 20)
        .then(pl.lit("$5〜20"))
        .otherwise(pl.lit("$20以上"))
        if market == "US"
        else pl.lit("全銘柄")
    )
    sc = sc.with_columns(bucket.alias("bucket"))

    def agg(df: pl.DataFrame) -> pl.DataFrame:
        return df.agg(
            pl.len().alias("n"),
            pl.col("win").mean().alias("win"),
            pl.col("fwd_ret").median().alias("med"),
            pl.col("fwd_ret").mean().alias("mean"),
            pl.col("fwd_excess").median().alias("med_excess"),
            (pl.col("fwd_ret") <= -0.10).mean().alias("p_dd10"),
            (pl.col("fwd_ret") >= 0.10).mean().alias("p_up10"),
        )

    stats = []
    for frame, by in (
        (agg(sc.group_by("signal")), "全体"),
        (agg(sc.group_by("signal", "bucket")), None),
    ):
        for r in frame.iter_rows(named=True):
            stats.append(
                {
                    "signal": r["signal"],
                    "bucket": by or r["bucket"],
                    **{
                        k: _f(r[k])
                        for k in ("n", "win", "med", "mean", "med_excess", "p_dd10", "p_up10")
                    },
                }
            )
    # yearly stability
    yearly = agg(
        sc.with_columns(pl.col("as_of").dt.year().alias("y")).group_by("signal", "y")
    ).sort("signal", "y")
    stats_year = [
        {
            "signal": r["signal"],
            "year": r["y"],
            "n": r["n"],
            "win": _f(r["win"]),
            "med": _f(r["med"]),
        }
        for r in yearly.iter_rows(named=True)
    ]
    # today's signals (liquidity filter)
    last = sig["as_of"].max()
    today = sig.filter(pl.col("as_of") == last).with_columns(
        (pl.col("close") * pl.col("vol20")).alias("dvol")
    )
    today = today.filter(pl.col("dvol") >= min_dollar_vol)
    rows = []
    for k in rules.SIGNALS:
        for r in today.filter(pl.col(k)).iter_rows(named=True):
            rows.append({"id": r["entity_id"], "signal": k, "close": _f(r["close"]), "chg": _f(r["close"] / r["prev_close"] - 1 if r["prev_close"] else None),
                         "vr": _f(r["volume"] / r["vol20"] if r["vol20"] else None, 2), "rsi": _f(r["rsi14"], 1), "dvol": _f(r["dvol"], 0)})  # fmt: skip
    hist = ev.filter(pl.col("as_of") >= last - timedelta(days=400)).select(
        "entity_id", "as_of", "signal"
    )
    return {
        "asof": last.isoformat(),
        "stats": stats,
        "stats_year": stats_year,
        "today": rows,
        "events": hist,
        "sig": sig,
    }


# ---------------------------------------------------------------- stock detail
def stock_bars(prices: pl.DataFrame, ids: list[str], n: int = 260) -> dict[str, list[list[Any]]]:
    out = {}
    p = prices.filter(pl.col("entity_id").is_in(ids)).sort(["entity_id", "as_of"])
    for (eid,), g in p.group_by("entity_id"):
        g = g.tail(n)
        out[str(eid)] = [[a.isoformat(), _f(o), _f(h), _f(lo), _f(c), _f(v, 0)] for a, o, h, lo, c, v in
                         zip(g["as_of"], g["open"], g["high"], g["low"], g["close"], g["volume"], strict=True)]  # fmt: skip
    return out


def options_summary(storage: GuardedStorage) -> dict[str, dict[str, Any]]:
    keys = sorted(
        k for k in storage.list("silver/options_chain/market=US/") if k.endswith("_post.parquet")
    )
    if not keys:
        keys = sorted(
            k for k in storage.list("silver/options_chain/market=US/") if k.endswith("_pre.parquet")
        )
    if not keys:
        return {}
    ch = storage.read_parquet(keys[-1])
    und = storage.read_parquet(keys[-1].replace("options_chain", "options_underlying"))
    spot = dict(zip(und["entity_id"], und["regularMarketPrice"], strict=True))
    out: dict[str, dict[str, Any]] = {}
    for (eid,), g in ch.group_by("entity_id"):
        s = spot.get(eid)
        calls, puts = g.filter(pl.col("option_type") == "C"), g.filter(pl.col("option_type") == "P")
        rec: dict[str, Any] = {
            "asof": str(g["as_of"][0]),
            "pcr_oi": _f(puts["open_interest"].sum() / max(1.0, calls["open_interest"].sum()), 3),
            "pcr_vol": _f(
                puts["volume"].fill_null(0).sum() / max(1.0, calls["volume"].fill_null(0).sum()), 3
            ),
            "call_oi": _f(calls["open_interest"].sum(), 0),
            "put_oi": _f(puts["open_interest"].sum(), 0),
        }
        if s:
            near = g.filter((pl.col("dte") >= 20) & pl.col("implied_vol").is_not_null())
            if near.height:
                exp = near["dte"].min()
                atm = (
                    near.filter(pl.col("dte") == exp)
                    .with_columns((pl.col("strike") - s).abs().alias("d"))
                    .sort("d")
                    .head(4)
                )
                rec["atm_iv"] = _f(atm["implied_vol"].mean(), 4)
                rec["atm_dte"] = int(exp)
            otm_p = g.filter(
                (pl.col("option_type") == "P")
                & (pl.col("strike") < s * 0.95)
                & (pl.col("dte").is_between(20, 60))
            )
            otm_c = g.filter(
                (pl.col("option_type") == "C")
                & (pl.col("strike") > s * 1.05)
                & (pl.col("dte").is_between(20, 60))
            )
            if otm_p.height and otm_c.height:
                rec["skew"] = (
                    _f(
                        otm_p["implied_vol"].drop_nulls().mean()
                        - otm_c["implied_vol"].drop_nulls().mean(),
                        4,
                    )
                    if otm_p["implied_vol"].drop_nulls().len()
                    and otm_c["implied_vol"].drop_nulls().len()
                    else None
                )
        out[str(eid)] = rec
    return out


def finra_short(storage: GuardedStorage, ids: list[str], since: date) -> dict[str, list[list[Any]]]:
    df = _read_dates_named(storage, "finra_regsho_daily", "US", since)
    if df.height == 0:
        return {}
    df = df.filter(pl.col("entity_id").is_in(ids))
    out = {}
    for (eid,), g in df.sort("as_of").group_by("entity_id"):
        out[str(eid)] = [
            [a.isoformat(), _f(r, 4)]
            for a, r in zip(g["as_of"], g["short_volume_ratio"], strict=True)
        ]
    return out


def _read_dates_named(
    storage: GuardedStorage, table: str, market: str, since: date
) -> pl.DataFrame:
    keys = [k for k in storage.list(f"silver/{table}/market={market}/") if k.endswith(".parquet")]
    pick = [k for k in keys if k.rsplit("/", 1)[-1].replace("date=", "")[:10] >= since.isoformat()]
    return _read_many(storage, pick)


def short_interest(storage: GuardedStorage, ids: list[str]) -> dict[str, list[list[Any]]]:
    keys = sorted(
        k for k in storage.list("silver/finra_short_interest/market=US/") if k.endswith(".parquet")
    )[-12:]
    if not keys:
        return {}
    df = pl.concat([storage.read_parquet(k) for k in keys], how="diagonal_relaxed").filter(
        pl.col("entity_id").is_in(ids)
    )
    out = {}
    for (eid,), g in df.sort("as_of").group_by("entity_id"):
        out[str(eid)] = [
            [a.isoformat(), _f(si, 0), _f(dc, 2)]
            for a, si, dc in zip(g["as_of"], g["short_interest"], g["days_to_cover"], strict=True)
        ]
    return out


def jp_short_positions(storage: GuardedStorage, ids: list[str]) -> dict[str, dict[str, Any]]:
    keys = [
        k for k in storage.list("silver/jpx_short_positions/market=JP/") if k.endswith(".parquet")
    ]
    if not keys:
        return {}
    df = _read_many(storage, keys).select(
        "entity_id", "disclosure_date", "calc_date", "short_seller", "ratio"
    )
    df = df.filter(pl.col("entity_id").is_in(ids)).sort("calc_date")
    dates = sorted(df["disclosure_date"].unique().to_list())[-120:]
    # forward-fill each (entity, seller) position across disclosure dates, then sum per date
    grid = (
        df.select("entity_id", "short_seller")
        .unique()
        .join(pl.DataFrame({"disclosure_date": dates}), how="cross")
    )
    last = df.group_by("entity_id", "short_seller", "disclosure_date").agg(pl.col("ratio").last())
    full = (
        grid.join(last, on=["entity_id", "short_seller", "disclosure_date"], how="left")
        .sort("entity_id", "short_seller", "disclosure_date")
        .with_columns(pl.col("ratio").forward_fill().over("entity_id", "short_seller"))
        .filter(pl.col("ratio") >= 0.005)
    )
    daily = full.group_by("entity_id", "disclosure_date").agg(
        pl.len().alias("n"), pl.col("ratio").sum().alias("tot")
    )
    out: dict[str, dict[str, Any]] = {}
    for (eid,), g in daily.sort("disclosure_date").group_by("entity_id"):
        m = {d: (n, t) for d, n, t in zip(g["disclosure_date"], g["n"], g["tot"], strict=True)}
        series = [
            [d.isoformat(), m.get(d, (0, 0.0))[0], _f(m.get(d, (0, 0.0))[1], 4)] for d in dates
        ]
        out[str(eid)] = {"series": series, "holders": []}
    latest = df.group_by("entity_id", "short_seller").agg(
        pl.col("ratio").last(), pl.col("calc_date").last()
    )
    latest = latest.filter(pl.col("ratio") >= 0.005).sort("ratio", descending=True)
    for (eid,), g in latest.group_by("entity_id"):
        out.setdefault(str(eid), {"series": [], "holders": []})["holders"] = [
            [r["short_seller"], _f(r["ratio"], 4), r["calc_date"].isoformat()]
            for r in g.head(8).iter_rows(named=True)
        ]
    return out


def jsf_latest(storage: GuardedStorage, ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted(
        k for k in storage.list("silver/jsf_balance/market=JP/") if k.endswith(".parquet")
    )
    if keys:
        z = storage.read_parquet(keys[-1]).filter(
            pl.col("entity_id").is_in(ids) & pl.col("exchange").str.contains("東証")
        )
        for r in z.iter_rows(named=True):
            out[r["entity_id"]] = {
                "asof": str(r["trade_date"]),
                "loan": _f(r["loan_balance_shares"], 0),
                "lend": _f(r["lend_balance_shares"], 0),
                "ratio": _f(r["loan_lend_ratio"], 3),
            }
    keys = sorted(
        k for k in storage.list("silver/jsf_premium/market=JP/") if k.endswith(".parquet")
    )
    if keys:
        p = storage.read_parquet(keys[-1]).filter(
            pl.col("entity_id").is_in(ids) & (pl.col("exchange") == "東証")
        )
        for r in p.iter_rows(named=True):
            out.setdefault(r["entity_id"], {})["premium"] = _f(r["premium_rate_yen"], 2)
    return out


def news(storage: GuardedStorage, since: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    f = _read_dates_named(storage, "sec_filings", "US", since)
    h = _read_dates_named(storage, "trading_halts", "US", since)
    return f, h


def to_json_bytes(obj: Any) -> bytes:
    import json

    buf = io.StringIO()
    json.dump(obj, buf, ensure_ascii=False, separators=(",", ":"), default=str)
    return buf.getvalue().encode("utf-8")
