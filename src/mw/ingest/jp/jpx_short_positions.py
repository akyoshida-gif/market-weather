"""JPX 空売り残高報告 (short positions >= 0.5% of shares outstanding), DESIGN §8.2.

Index pages list daily XLS files with opaque folder ids -> links are scraped from HTML.
Current month: index.html; history (~13 months): 00-archives-01.html .. 00-archives-12.html.
File date = disclosure date (published ~17:00 JST). Rows carry `Date of Calculation` (as_of).
`published_at` rule: disclosure date 09:00 UTC (18:00 JST, 1h after the stated target);
if HTTP Last-Modified is later than that and within 7 days (late publication), use it instead.
ToS (ADR-0005): private use, no redistribution, each file fetched once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import polars as pl
import xlrd

from mw.core.http import HttpClient
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, silver_key, stamp_pit, write_raw_bytes

SOURCE_ID = "jpx_short_positions"
SOURCE_VERSION = "0.1.0"
SITE = "https://www.jpx.co.jp"
INDEX = SITE + "/markets/public/short-selling/index.html"
ARCHIVES = [SITE + f"/markets/public/short-selling/00-archives-{i:02d}.html" for i in range(1, 13)]
LINK_RE = re.compile(r'href="(/markets/public/short-selling/[^"]+/(\d{8})_Short_Positions\.xls)"')
PUBLISH_UTC = time(9, 0)
FIRST_DATA_ROW = 8

COLUMNS = [
    "calc_date", "code", "name_ja", "name_en", "short_seller", "short_seller_address",
    "discretionary_contractor", "discretionary_contractor_address", "fund_name",
    "ratio", "shares", "units", "prev_calc_date", "prev_ratio", "notes",
]  # fmt: skip


@dataclass(frozen=True)
class FileLink:
    url: str
    disclosure_date: date


def parse_links(html: str) -> list[FileLink]:
    seen: dict[str, FileLink] = {}
    for path, ymd in LINK_RE.findall(html):
        d = datetime.strptime(ymd, "%Y%m%d").date()
        seen[path] = FileLink(SITE + path, d)
    return sorted(seen.values(), key=lambda f: f.disclosure_date)


def list_files(http: HttpClient, include_archives: bool = False) -> list[FileLink]:
    pages = [INDEX, *(ARCHIVES if include_archives else [])]
    links: dict[str, FileLink] = {}
    for url in pages:
        resp = http.get(url)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        for f in parse_links(resp.text):
            links[f.url] = f
    return sorted(links.values(), key=lambda f: f.disclosure_date)


def _excel_date(v: Any, datemode: int) -> date | None:
    if v in ("", None):
        return None
    if isinstance(v, float):
        dt: datetime = xlrd.xldate_as_datetime(v, datemode)
        return dt.date()
    return datetime.strptime(str(v).strip(), "%Y/%m/%d").date()


def _code(v: Any) -> str:
    if isinstance(v, float):
        return str(int(v))
    return str(v).strip()


def parse_xls(data: bytes) -> pl.DataFrame:
    book = xlrd.open_workbook(file_contents=data, encoding_override="cp932")
    sh = book.sheet_by_index(0)
    header = str(sh.cell_value(FIRST_DATA_ROW - 1, 1))
    if "Date of Calculation" not in header:
        raise ValueError(f"unexpected layout: header cell={header!r}")
    rows: list[dict[str, Any]] = []
    for r in range(FIRST_DATA_ROW, sh.nrows):
        vals = [sh.cell_value(r, c) if c < sh.ncols else "" for c in range(1, 16)]
        if vals[0] in ("", None):
            continue
        rec = dict(zip(COLUMNS, vals, strict=True))
        rec["calc_date"] = _excel_date(rec["calc_date"], book.datemode)
        rec["prev_calc_date"] = _excel_date(rec["prev_calc_date"], book.datemode)
        rec["code"] = _code(rec["code"])
        for k in ("ratio", "shares", "units", "prev_ratio"):
            rec[k] = float(rec[k]) if rec[k] not in ("", None) else None
        for k in COLUMNS:
            if isinstance(rec[k], str):
                rec[k] = rec[k].strip() or None
        rows.append(rec)
    schema: dict[str, Any] = {c: pl.String for c in COLUMNS}
    schema.update(
        calc_date=pl.Date,
        prev_calc_date=pl.Date,
        ratio=pl.Float64,
        shares=pl.Float64,
        units=pl.Float64,
        prev_ratio=pl.Float64,
    )
    return pl.DataFrame(rows, schema=schema)


def published_at_for(disclosure: date, last_modified: datetime | None) -> datetime:
    rule = datetime.combine(disclosure, PUBLISH_UTC, tzinfo=UTC)
    if last_modified and rule < last_modified <= rule + timedelta(days=7):
        return last_modified
    return rule


def silver_key_for(disclosure: date) -> str:
    return silver_key("jpx_short_positions", "JP", disclosure, f"{disclosure.isoformat()}.parquet")


def ingest_file(storage: GuardedStorage, http: HttpClient, link: FileLink) -> dict[str, Any] | None:
    """Fetch one disclosure file if not yet ingested. Returns summary or None if skipped."""
    skey = silver_key_for(link.disclosure_date)
    if storage.exists(skey):
        return None
    resp = http.get(link.url)
    resp.raise_for_status()
    lm_header = resp.headers.get("last-modified")
    last_modified = parsedate_to_datetime(lm_header).astimezone(UTC) if lm_header else None
    df = parse_xls(resp.content)
    ingested_at = datetime.now(UTC)
    rk = raw_key(
        SOURCE_ID,
        ingested_at,
        f"{link.disclosure_date:%Y%m%d}_Short_Positions_{ingested_at:%H%M%S}.xls",
    )
    sha = write_raw_bytes(storage, rk, resp.content)
    silver = stamp_pit(
        df.with_columns(
            (pl.lit("JP:") + pl.col("code")).alias("entity_id"),
            pl.lit(link.disclosure_date).alias("disclosure_date"),
            pl.lit(last_modified).cast(pl.Datetime("us", "UTC")).alias("http_last_modified"),
        ),
        as_of=link.disclosure_date,
        published_at=published_at_for(link.disclosure_date, last_modified),
        ingested_at=ingested_at,
        source_id=SOURCE_ID,
        source_version=SOURCE_VERSION,
        raw_ref=f"{rk}#sha256={sha}",
    )
    # as_of per row = calculation date (what the position describes)
    silver = silver.with_columns(pl.col("calc_date").alias("as_of"))
    storage.write_parquet(skey, silver)
    return {"disclosure_date": link.disclosure_date.isoformat(), "rows": silver.height}
