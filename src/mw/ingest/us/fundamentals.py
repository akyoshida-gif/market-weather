"""Company financials from SEC XBRL (company facts), point in time (VALIDATION_PLAN_V3 §5).

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (one JSON per company).
Each fact carries the period (start/end), the form (10-Q/10-K/...), and the date it was FILED.
published_at rule: filed date 23:59 New York -> usable from the next session (the exact acceptance
time is not in this API, so the conservative end of day is used).
Only the concepts below are kept; restated values appear as later filings of the same period and are
kept as separate rows (the as-of join then uses what was known at the time).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import polars as pl

COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A", "10-KT", "10-QT"}

# our name -> XBRL tags tried in order (first one a company reports wins, per period)
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "shares_out": ["CommonStockSharesOutstanding"],
}  # fmt: skip
TAG_TO_NAME = {t: (n, i) for n, tags in CONCEPTS.items() for i, t in enumerate(tags)}


def parse_companyfacts(data: bytes) -> pl.DataFrame:
    """Long table: cik, concept, tag_rank, unit, start, end, val, form, fy, fp, filed, accn."""
    d = json.loads(data)
    cik = int(d.get("cik") or 0)
    rows: list[dict[str, Any]] = []
    for ns in ("us-gaap", "ifrs-full", "dei"):
        for tag, body in (d.get("facts", {}).get(ns, {}) or {}).items():
            hit = TAG_TO_NAME.get(tag)
            if hit is None:
                continue
            name, rank = hit
            for unit, facts in (body.get("units") or {}).items():
                for f in facts:
                    if f.get("form") not in FORMS or f.get("val") is None or not f.get("filed"):
                        continue
                    rows.append({"cik": cik, "concept": name, "tag_rank": rank, "unit": unit, "start": f.get("start"),
                                 "end": f.get("end"), "val": float(f["val"]), "form": f.get("form"), "fy": f.get("fy"),
                                 "fp": f.get("fp"), "filed": f.get("filed"), "accn": f.get("accn")})  # fmt: skip
    schema = {"cik": pl.Int64, "concept": pl.String, "tag_rank": pl.Int8, "unit": pl.String, "start": pl.String,
              "end": pl.String, "val": pl.Float64, "form": pl.String, "fy": pl.Int32, "fp": pl.String,
              "filed": pl.String, "accn": pl.String}  # fmt: skip
    df = pl.DataFrame(rows, schema=schema)
    if df.height == 0:
        return df
    return df.with_columns(
        pl.col("start").str.to_date(strict=False),
        pl.col("end").str.to_date(strict=False),
        pl.col("filed").str.to_date(strict=False),
    ).filter(pl.col("end").is_not_null() & pl.col("filed").is_not_null())


def published_at(filed: pl.Expr) -> pl.Expr:
    """Filed date 23:59 New York, as UTC (usable from the next session)."""
    return (
        (filed.cast(pl.Datetime("us")) + pl.duration(hours=23, minutes=59))
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )


def quarterly(df: pl.DataFrame) -> pl.DataFrame:
    """One value per (cik, concept, period end, filed): quarterly durations (~3 months) and instants.

    Q4 is rarely filed as a quarter (only in the 10-K as a full year); it is derived later as
    full year minus the first three quarters when needed (features).
    """
    days = (pl.col("end") - pl.col("start")).dt.total_days()
    kind = (
        pl.when(pl.col("start").is_null()).then(pl.lit("instant"))
        .when(days.is_between(80, 100)).then(pl.lit("quarter"))
        .when(days.is_between(350, 380)).then(pl.lit("year"))
        .otherwise(pl.lit("other"))
    )  # fmt: skip
    out = df.with_columns(kind.alias("period")).filter(pl.col("period") != "other")
    # the most preferred tag per (company, concept, period, end, filed)
    return (
        out.sort(["cik", "concept", "period", "end", "filed", "tag_rank"])
        .unique(["cik", "concept", "period", "end", "filed"], keep="first")
        .sort(["cik", "concept", "period", "end", "filed"])
    )


def now_utc() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)
