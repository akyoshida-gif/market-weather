"""日本証券金融 (JSF, taisyaku.jp) lending balances and premium charges (DESIGN §8.2 `jsf_lending`).

Files have fixed URLs and hold the latest day only -> must be captured daily; history cannot be
backfilled. ToS: private use permitted (site policy). One fetch per file per run.
- zandaka.csv: 貸借取引残高 by issue (CP932, 36 columns, header line 1)
- shina.csv:   品貸料率 (逆日歩) by issue (CP932, header on line 4, '*****' = not priced)
`as_of` = 申込日 (trade date); `published_at` = capture time (exact publication time unknown;
capture time is a conservative upper bound).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from mw.core.http import HttpClient
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, silver_key, stamp_pit, write_raw_bytes

SOURCE_ID = "jsf_lending"
SOURCE_VERSION = "0.1.0"
BASE = "https://www.taisyaku.jp/data/"

ZANDAKA_COLUMNS = [
    "trade_date", "settle_date", "code", "name", "exchange", "listing", "status",
    "loan_new_shares", "loan_repay_shares", "loan_balance_shares",
    "lend_new_shares", "lend_repay_shares", "lend_balance_shares", "net_balance_shares",
    "loan_new_yen", "loan_repay_yen", "loan_balance_yen",
    "lend_new_yen", "lend_repay_yen", "lend_balance_yen", "net_balance_yen",
    "std_margin_buy_shares", "std_margin_sell_shares",
    "loan_ex_rights_yen", "lend_ex_rights_yen",
    "renewal_loan_up_yen", "renewal_loan_down_yen", "renewal_lend_down_yen", "renewal_lend_up_yen",
    "turnover_days_total", "turnover_days_loan_new", "turnover_days_loan_repay",
    "turnover_days_loan_balance", "turnover_days_lend_new", "turnover_days_lend_repay",
    "turnover_days_lend_balance",
]  # fmt: skip

SHINA_COLUMNS = [
    "trade_date", "settle_date", "code", "name", "exchange", "settlement_reason",
    "settlement_date", "loan_price_yen", "lend_excess_shares", "max_rate_yen",
    "premium_rate_yen", "premium_days", "prev_premium_rate_yen", "remarks", "restriction",
    "bid_ratio_rank",
]  # fmt: skip

STATUS_MAP = {"確報": "final", "速報": "preliminary"}


@dataclass
class Captured:
    name: str
    raw: bytes
    frame: pl.DataFrame
    as_of: date


def _read_csv(data: bytes, skip_rows: int, columns: list[str]) -> pl.DataFrame:
    text = data.decode("cp932")
    df = pl.read_csv(
        io.StringIO(text),
        skip_rows=skip_rows,
        has_header=True,
        infer_schema=False,  # all strings; typed below
        truncate_ragged_lines=True,
    )
    if df.width != len(columns):
        raise ValueError(f"unexpected column count {df.width} != {len(columns)}")
    return df.rename(dict(zip(df.columns, columns, strict=True)))


def _num(cols: list[str]) -> list[pl.Expr]:
    return [
        pl.col(c).str.strip_chars().replace({"*****": None, "": None}).cast(pl.Float64)
        for c in cols
    ]


def parse_zandaka(data: bytes) -> pl.DataFrame:
    df = _read_csv(data, 0, ZANDAKA_COLUMNS)
    text_cols = {"code", "name", "exchange", "listing", "status", "trade_date", "settle_date"}
    num_cols = [c for c in ZANDAKA_COLUMNS if c not in text_cols]
    return df.with_columns(
        pl.col("trade_date").str.to_date("%Y/%m/%d"),
        pl.col("settle_date").str.to_date("%Y/%m/%d"),
        pl.col("status").replace(STATUS_MAP),
        *_num(num_cols),
    ).with_columns(
        # 貸借倍率 = 融資残高 / 貸株残高 (null when no lending balance)
        pl.when(pl.col("lend_balance_shares") > 0)
        .then(pl.col("loan_balance_shares") / pl.col("lend_balance_shares"))
        .otherwise(None)
        .alias("loan_lend_ratio"),
    )


def parse_shina(data: bytes) -> pl.DataFrame:
    df = _read_csv(data, 3, SHINA_COLUMNS)
    num_cols = [
        "loan_price_yen", "lend_excess_shares", "max_rate_yen", "premium_rate_yen",
        "premium_days", "prev_premium_rate_yen",
    ]  # fmt: skip
    return df.with_columns(
        pl.col("trade_date").str.to_date("%Y%m%d"),
        pl.col("settle_date").str.to_date("%Y%m%d"),
        *_num(num_cols),
    )


def _single_as_of(df: pl.DataFrame) -> date:
    dates = df["trade_date"].drop_nulls().unique().to_list()
    if len(dates) != 1:
        raise ValueError(f"expected one trade_date, got {dates[:5]}")
    d: date = dates[0]
    return d


def fetch(http: HttpClient) -> list[Captured]:
    out: list[Captured] = []
    for name, parser in (("zandaka", parse_zandaka), ("shina", parse_shina)):
        resp = http.get(BASE + f"{name}.csv")
        resp.raise_for_status()
        df = parser(resp.content)
        out.append(Captured(name, resp.content, df, _single_as_of(df)))
    return out


def table_for(name: str) -> str:
    return {"zandaka": "jsf_balance", "shina": "jsf_premium"}[name]


def silver_key_for(name: str, as_of: date, status: str) -> str:
    return silver_key(table_for(name), "JP", as_of, f"{as_of.isoformat()}_{status}.parquet")


def _status(c: Captured) -> str:
    if c.name == "zandaka":
        statuses = set(c.frame["status"].drop_nulls().unique().to_list())
        return "final" if statuses == {"final"} else "preliminary"
    return "final"


def persist(
    storage: GuardedStorage, captured: list[Captured], captured_at: datetime
) -> dict[str, Any]:
    ingested_at = datetime.now(UTC)
    result: dict[str, Any] = {}
    for c in captured:
        status = _status(c)
        skey = silver_key_for(c.name, c.as_of, status)
        if storage.exists(skey):
            result[c.name] = {"as_of": c.as_of.isoformat(), "skipped": "exists"}
            continue
        rk = raw_key(
            SOURCE_ID, ingested_at, f"{c.name}_{c.as_of.isoformat()}_{captured_at:%H%M%S}.csv"
        )
        sha = write_raw_bytes(storage, rk, c.raw)
        silver = stamp_pit(
            c.frame.with_columns((pl.lit("JP:") + pl.col("code")).alias("entity_id")),
            as_of=c.as_of,
            published_at=captured_at,
            ingested_at=ingested_at,
            source_id=SOURCE_ID,
            source_version=SOURCE_VERSION,
            raw_ref=f"{rk}#sha256={sha}",
        )
        storage.write_parquet(skey, silver)
        result[c.name] = {"as_of": c.as_of.isoformat(), "status": status, "rows": silver.height}
    return result
