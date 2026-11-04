"""Institutional holdings from the SEC Form 13F data sets (VALIDATION_PLAN_V3 §5).

Source page: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets (zip per filing window,
tables SUBMISSION, COVERPAGE, INFOTABLE; tab separated).
Only original 13F-HR reports (no amendments) filed within 46 days of the quarter end count, so every
manager is counted once and the totals are known by `period end + 46 days` (published_at, 23:59 UTC).
Per (CUSIP, quarter): number of managers holding it, shares, value. Option rows (PUT/CALL) are excluded.
A filing window can hold part of a quarter's reports; totals are summed over windows when used.
"""

from __future__ import annotations

import io
import re
import zipfile

import polars as pl

PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
ZIP_RE = re.compile(r'href="([^"]*?form13f[^"]*?\.zip)"', re.I)
DEADLINE_DAYS = 46


def links(html: str) -> list[str]:
    out = []
    for u in ZIP_RE.findall(html):
        url = u if u.startswith("http") else "https://www.sec.gov" + u
        if url not in out:
            out.append(url)
    return sorted(out)


def window_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].removesuffix(".zip").lower()


def _table(z: zipfile.ZipFile, name: str, cols: list[str]) -> pl.DataFrame:
    member = next((n for n in z.namelist() if n.upper().endswith(name + ".TSV")), None)
    if member is None:
        raise ValueError(f"{name}.tsv not found in {z.namelist()[:10]}")
    df = pl.read_csv(io.BytesIO(z.read(member)), separator="\t", infer_schema=False, quote_char=None,
                     truncate_ragged_lines=True, encoding="utf8-lossy")  # fmt: skip
    df = df.rename({c: c.strip().upper() for c in df.columns})
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: columns {missing} not found (have {df.columns[:20]})")
    return df.select(cols)


def _date(col: str) -> pl.Expr:
    s = pl.col(col).str.strip_chars()
    return pl.coalesce(s.str.to_date("%d-%b-%Y", strict=False), s.str.to_date("%Y-%m-%d", strict=False),
                       s.str.to_date("%m-%d-%Y", strict=False))  # fmt: skip


def parse_window(data: bytes) -> pl.DataFrame:
    """Aggregate one filing window: cusip, period, managers, shares, value, last_filed."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sub = _table(
            z,
            "SUBMISSION",
            ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"],
        )
        info = _table(
            z,
            "INFOTABLE",
            ["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
        )
    sub = sub.with_columns(_date("FILING_DATE").alias("filed"), _date("PERIODOFREPORT").alias("period")).filter(
        (pl.col("SUBMISSIONTYPE").str.strip_chars() == "13F-HR")
        & pl.col("filed").is_not_null() & pl.col("period").is_not_null()
        & (pl.col("filed") <= pl.col("period") + pl.duration(days=DEADLINE_DAYS))
    )  # fmt: skip
    info = info.filter(
        (pl.col("PUTCALL").is_null() | (pl.col("PUTCALL").str.strip_chars() == ""))
        & (pl.col("SSHPRNAMTTYPE").str.strip_chars() == "SH")
    ).with_columns(
        pl.col("CUSIP").str.strip_chars().str.to_uppercase().alias("cusip"),
        pl.col("SSHPRNAMT").cast(pl.Float64, strict=False).alias("shares"),
        pl.col("VALUE").cast(pl.Float64, strict=False).alias("value"),
    )
    j = info.join(
        sub.select("ACCESSION_NUMBER", "CIK", "filed", "period"), on="ACCESSION_NUMBER", how="inner"
    )
    return (
        j.group_by("cusip", "period")
        .agg(
            pl.col("CIK").n_unique().alias("managers"),
            pl.col("shares").sum(),
            pl.col("value").sum(),
            pl.col("filed").max().alias("last_filed"),
        )  # fmt: skip
        .filter(pl.col("cusip").str.len_chars() >= 8)
    )


def published_at(period: pl.Expr) -> pl.Expr:
    return (
        period.cast(pl.Datetime("us")) + pl.duration(days=DEADLINE_DAYS, hours=23, minutes=59)
    ).dt.replace_time_zone("UTC")
