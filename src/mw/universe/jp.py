"""JP universe: TOPIX500 (Core30 + Large70 + Mid400) + configured ETFs (TOPIX-17 sector ETFs etc.).

Source: JPX 東証上場銘柄一覧 (data_j.xlsx, monthly, as of previous month-end):
  index page https://www.jpx.co.jp/markets/statistics-equities/misc/01.html (link has opaque id)
Columns: 日付, コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分, 17業種コード, 17業種区分,
         規模コード, 規模区分
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any

import openpyxl
import polars as pl

SITE = "https://www.jpx.co.jp"
INDEX = SITE + "/markets/statistics-equities/misc/01.html"
LINK_RE = re.compile(r'href="(/markets/statistics-equities/misc/[^"]+/data_j\.xlsx?)"')
SIZE_IN = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}
COLUMNS = ["list_date", "code", "name", "market_segment", "sector33_code", "sector33",
           "sector17_code", "sector17", "size_code", "size"]  # fmt: skip


def find_link(html: str) -> str:
    m = LINK_RE.search(html)
    if not m:
        raise ValueError("data_j link not found on JPX page")
    return SITE + m.group(1)


def parse(data: bytes) -> pl.DataFrame:
    ws = openpyxl.load_workbook(io.BytesIO(data), read_only=True).active
    rows: list[tuple[Any, ...]] = list(ws.iter_rows(values_only=True))
    if len(rows) < 2 or len(rows[0]) < 10:
        raise ValueError("unexpected data_j layout")
    recs = [dict(zip(COLUMNS, r[:10], strict=True)) for r in rows[1:] if r[1] is not None]
    df = pl.DataFrame(recs, schema={c: pl.String for c in COLUMNS}, strict=False)
    return df.with_columns(
        pl.col("list_date").str.to_date("%Y%m%d", strict=False),
        pl.col("code").str.replace(r"\.0$", ""),
    )


def build(data: bytes, etfs: list[str], as_of: date) -> pl.DataFrame:
    df = parse(data)
    core = df.filter(pl.col("size").is_in(list(SIZE_IN))).with_columns(
        pl.lit("common").alias("kind")
    )
    etf = df.filter(pl.col("code").is_in(etfs)).with_columns(pl.lit("etf").alias("kind"))
    out = pl.concat([core, etf], how="vertical_relaxed").unique(subset=["code"], keep="last")
    return out.with_columns(
        pl.col("code").alias("symbol"),
        (pl.col("code") + ".T").alias("yahoo_symbol"),
        (pl.lit("JP:") + pl.col("code")).alias("entity_id"),
        pl.lit(as_of).alias("as_of"),
    ).sort("code")


def snapshot_date(df: pl.DataFrame) -> date:
    d = df["list_date"].drop_nulls().max()
    return d if isinstance(d, date) else datetime.now().date()
