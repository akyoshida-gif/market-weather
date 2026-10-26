"""SEC reference datasets for validation features (VALIDATION_PLAN §2, ADR-0015).

1) Fails-to-deliver (half-monthly zip, 2009-):
   https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data  (links scraped)
   pipe-delimited: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
   published_at rule: first half (a) -> last day of the month + 3 days; second half (b) ->
   18th of the following month (SEC: "end of the month" / "about the 15th"), 23:59 UTC.
2) Insider transactions (quarterly zip, 2006-):
   https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets (links scraped)
   tab-separated tables SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS.
   published_at rule: filing date 22:00 ET (Form 4 dissemination cutoff) -> usable next session.
3) Shares outstanding (XBRL frames, quarterly instants, 2009-):
   https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{Y}Q{q}I.json
   frames carry no filing date -> published_at rule: period end + 90 days (covers 10-K deadlines).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

NY = ZoneInfo("America/New_York")
FTD_PAGE = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
INSIDER_PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
FRAMES = "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{y}Q{q}I.json"
FTD_RE = re.compile(r'href="([^"]*cnsfails(\d{6})([ab])\.zip)"', re.I)
INSIDER_RE = re.compile(r'href="([^"]*?(\d{4})q([1-4])_form345\.zip)"', re.I)


def absolute(href: str) -> str:
    return href if href.startswith("http") else "https://www.sec.gov" + href


# ---------------------------------------------------------------- fails-to-deliver
def ftd_links(html: str) -> list[tuple[str, str]]:
    """[(url, 'YYYYMMa|b'), ...] sorted."""
    out = {f"{ym}{h}".lower(): absolute(u) for u, ym, h in FTD_RE.findall(html)}
    return sorted(((u, k) for k, u in out.items()), key=lambda x: x[1])


def ftd_published_at(period: str) -> datetime:
    y, m, half = int(period[:4]), int(period[4:6]), period[6]
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    d = nxt - timedelta(days=1) + timedelta(days=3) if half == "a" else nxt.replace(day=18)
    return datetime.combine(d, time(23, 59), tzinfo=UTC)


def parse_ftd(data: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    lines = [
        ln for ln in raw.splitlines() if ln and ln[:1].isdigit() or ln.startswith("SETTLEMENT")
    ]
    df = pl.read_csv(
        io.StringIO("\n".join(lines)),
        separator="|",
        infer_schema=False,
        quote_char=None,
        truncate_ragged_lines=True,
    )
    df.columns = [c.strip().upper() for c in df.columns]
    return df.select(
        pl.col("SETTLEMENT DATE")
        .str.strip_chars()
        .str.to_date("%Y%m%d", strict=False)
        .alias("settle_date"),
        pl.col("CUSIP").str.strip_chars().alias("cusip"),
        pl.col("SYMBOL").str.strip_chars().alias("symbol"),
        pl.col("QUANTITY (FAILS)").str.strip_chars().cast(pl.Float64, strict=False).alias("fails"),
        pl.col("PRICE").str.strip_chars().cast(pl.Float64, strict=False).alias("price"),
    ).filter(pl.col("settle_date").is_not_null() & pl.col("symbol").is_not_null())


# ---------------------------------------------------------------- insider (Form 3/4/5)
def insider_links(html: str) -> list[tuple[str, str]]:
    out = {f"{y}q{q}": absolute(u) for u, y, q in INSIDER_RE.findall(html)}
    return sorted(((u, k) for k, u in out.items()), key=lambda x: x[1])


def _tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> pl.DataFrame:
    member = next((n for n in z.namelist() if n.upper().endswith(name + ".TSV")), None)
    if member is None:
        raise ValueError(f"{name}.tsv not found")
    df = pl.read_csv(
        io.BytesIO(z.read(member)),
        separator="\t",
        infer_schema=False,
        quote_char=None,
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )
    df.columns = [c.strip().upper() for c in df.columns]
    # older quarterly files lack some columns (e.g. AFF10B5ONE was added later): fill with null
    missing = [pl.lit(None, dtype=pl.String).alias(c) for c in cols if c not in df.columns]
    return df.with_columns(missing).select(cols)


def parse_insider(data: bytes) -> pl.DataFrame:
    """One row per non-derivative transaction with issuer, owner role and filing date."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sub = _tsv(
            z,
            "SUBMISSION",
            [
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "DOCUMENT_TYPE",
                "ISSUERCIK",
                "ISSUERTRADINGSYMBOL",
                "AFF10B5ONE",
            ],
        )
        own = _tsv(
            z,
            "REPORTINGOWNER",
            [
                "ACCESSION_NUMBER",
                "RPTOWNERCIK",
                "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP",
                "RPTOWNER_TITLE",
            ],
        )
        tr = _tsv(z, "NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                        "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP"])  # fmt: skip
    own1 = own.unique(subset=["ACCESSION_NUMBER"], keep="first")
    df = tr.join(sub, on="ACCESSION_NUMBER", how="left").join(
        own1, on="ACCESSION_NUMBER", how="left"
    )
    return df.select(
        pl.col("ACCESSION_NUMBER").alias("accession"),
        pl.col("FILING_DATE").str.to_date("%d-%b-%Y", strict=False).alias("filing_date"),
        pl.col("DOCUMENT_TYPE").alias("form"),
        pl.col("ISSUERCIK").cast(pl.Int64, strict=False).alias("cik"),
        pl.col("ISSUERTRADINGSYMBOL").str.strip_chars().str.to_uppercase().alias("symbol"),
        pl.col("AFF10B5ONE").alias("plan_10b5_1"),
        pl.col("RPTOWNERNAME").alias("owner"),
        pl.col("RPTOWNER_RELATIONSHIP").alias("relationship"),
        pl.col("RPTOWNER_TITLE").alias("title"),
        pl.col("TRANS_DATE").str.to_date("%d-%b-%Y", strict=False).alias("trans_date"),
        pl.col("TRANS_CODE").alias("code"),
        pl.col("TRANS_SHARES").cast(pl.Float64, strict=False).alias("shares"),
        pl.col("TRANS_PRICEPERSHARE").cast(pl.Float64, strict=False).alias("price"),
        pl.col("TRANS_ACQUIRED_DISP_CD").alias("acq_disp"),
        pl.col("SHRS_OWND_FOLWNG_TRANS").cast(pl.Float64, strict=False).alias("owned_after"),
        pl.col("DIRECT_INDIRECT_OWNERSHIP").alias("direct"),
    ).filter(pl.col("filing_date").is_not_null())


def insider_published_at(filing: pl.Expr) -> pl.Expr:
    """filing date 22:00 New York time, as UTC."""
    return (
        (filing.cast(pl.Datetime("us")) + pl.duration(hours=22))
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )


# ---------------------------------------------------------------- shares outstanding
def frames_url(y: int, q: int) -> str:
    return FRAMES.format(y=y, q=q)


def parse_frames(data: bytes) -> pl.DataFrame:
    d = json.loads(data)
    rows = d.get("data", [])
    return pl.DataFrame(
        {"cik": [r["cik"] for r in rows], "name": [r.get("entityName") for r in rows],
         "period_end": [r["end"] for r in rows], "shares_out": [float(r["val"]) for r in rows], "accn": [r.get("accn") for r in rows]},
        schema={"cik": pl.Int64, "name": pl.String, "period_end": pl.String, "shares_out": pl.Float64, "accn": pl.String},
    ).with_columns(pl.col("period_end").str.to_date("%Y-%m-%d", strict=False))  # fmt: skip


def quarters(start: date, end: date) -> list[tuple[int, int]]:
    out, y, q = [], start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def now_utc() -> datetime:
    return datetime.now(UTC)
