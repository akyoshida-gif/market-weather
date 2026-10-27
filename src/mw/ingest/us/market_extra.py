"""Market-wide reference series for the validation (VALIDATION_PLAN §2-4 / §2-6).

1) FINRA margin statistics (monthly, 1997-):
   https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics (xlsx link scraped)
   published_at rule: month end + 25 days, 23:59 UTC (FINRA publishes in the third week of the next month).
2) CFTC Commitments of Traders, futures only (weekly, report date = Tuesday), via the CFTC public reporting API
   (the www.cftc.gov yearly zip files are blocked from GitHub Actions runners):
   financial futures (TFF)  https://publicreporting.cftc.gov/resource/gpe5-46if.json
   disaggregated (commod.)  https://publicreporting.cftc.gov/resource/72hh-3qpy.json
   published_at rule: report date + 3 days (Friday) 22:00 UTC -> usable from the next session
   (released Fridays 15:30 ET; holiday weeks slip, so the next-session rule keeps it conservative).
3) Cboe put/call ratios (daily):
   2006-11 .. 2019-10  https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{total,equity,index}pc.csv
   2019-10 ..          https://cdn.cboe.com/data/us/options/market_statistics/daily/{YYYY-MM-DD}_daily_options
   published_at rule: trade date 22:00 UTC (after the close; usable from the next session).
4) SEC company profile (SIC industry code) from the raw submissions archive already stored.
   Latest values only (industry codes rarely change; known limitation).
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import openpyxl
import polars as pl

FINRA_PAGE = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
FINRA_RE = re.compile(r'href="([^"]*margin-statistics[^"]*\.xlsx)"', re.I)
COT_API = "https://publicreporting.cftc.gov/resource/{ds}.json"
COT_DATASETS = {"tff": "gpe5-46if", "disagg": "72hh-3qpy"}
CFTC_TFF = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip"
CFTC_DIS = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip"
CBOE_CSV = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{kind}pc.csv"
CBOE_DAILY = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
CBOE_CSV_END = date(2019, 10, 4)

# CFTC contract codes -> short name
TFF_MARKETS = {"13874A": "es", "209742": "nq", "23977A": "rty", "239742": "rty_cme", "043602": "ty",
               "042601": "tu", "098662": "dx", "097741": "jy", "1170E1": "vx"}  # fmt: skip
DIS_MARKETS = {"067651": "cl", "088691": "gc", "085692": "hg", "023651": "ng"}
CBOE_RATIOS = {"TOTAL PUT/CALL RATIO": "pc_total", "EQUITY PUT/CALL RATIO": "pc_equity",
               "INDEX PUT/CALL RATIO": "pc_index"}  # fmt: skip


def _utc(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=UTC)


# ---------------------------------------------------------------- FINRA margin
def finra_link(html: str) -> str:
    m = FINRA_RE.search(html)
    if not m:
        raise ValueError("margin statistics xlsx link not found")
    u = m.group(1)
    return u if u.startswith("http") else "https://www.finra.org" + u


def parse_margin(data: bytes) -> pl.DataFrame:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    out = []
    for r in rows[1:]:
        if not r or not r[0] or not re.match(r"^\d{4}-\d{2}$", str(r[0]).strip()):
            continue
        y, m = map(int, str(r[0]).strip().split("-"))
        month_end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
        out.append({"as_of": month_end, "margin_debit": r[1], "free_credit_cash": r[2], "free_credit_margin": r[3],
                    "published_at": _utc(month_end + timedelta(days=25), 23, 59)})  # fmt: skip
    return pl.DataFrame(out, schema={"as_of": pl.Date, "margin_debit": pl.Float64, "free_credit_cash": pl.Float64,
                                     "free_credit_margin": pl.Float64, "published_at": pl.Datetime("us", "UTC")})  # fmt: skip


# ---------------------------------------------------------------- CFTC COT
def parse_cot(
    data: bytes, markets: dict[str, str], groups: dict[str, tuple[str, str]]
) -> pl.DataFrame:
    """One row per (market, report date): open interest and net positions of each trader group."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        df = pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), infer_schema=False)
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.filter(pl.col("CFTC_Contract_Market_Code").str.strip_chars().is_in(list(markets)))

    def num(c: str) -> pl.Expr:
        return pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)

    cols = [
        pl.col("CFTC_Contract_Market_Code")
        .str.strip_chars()
        .replace_strict(markets)
        .alias("market"),
        pl.col("Report_Date_as_YYYY-MM-DD")
        .str.strip_chars()
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d")
        .alias("as_of"),
        num("Open_Interest_All").alias("open_interest"),
    ] + [(num(lo) - num(sh)).alias(f"net_{g}") for g, (lo, sh) in groups.items()]
    out = df.select(cols).filter(pl.col("as_of").is_not_null())
    return out.with_columns(
        (
            (
                pl.col("as_of").cast(pl.Datetime("us")) + pl.duration(days=3, hours=22)
            ).dt.replace_time_zone("UTC")
        ).alias("published_at")
    )


TFF_GROUPS = {"dealer": ("Dealer_Positions_Long_All", "Dealer_Positions_Short_All"),
              "asset_mgr": ("Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
              "lev_money": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All")}  # fmt: skip
DIS_GROUPS = {"producer": ("Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All"),
              "managed_money": ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All")}  # fmt: skip
# field names in the public reporting API (lower case, some without the _all suffix)
API_GROUPS = {
    "tff": {"dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
            "asset_mgr": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
            "lev_money": ("lev_money_positions_long", "lev_money_positions_short")},
    "disagg": {"producer": ("prod_merc_positions_long", "prod_merc_positions_short"),
               "managed_money": ("m_money_positions_long_all", "m_money_positions_short_all")},
}  # fmt: skip
API_MARKETS = {"tff": TFF_MARKETS, "disagg": DIS_MARKETS}


def cot_api_params(
    kind: str, code: str, since: date, offset: int, limit: int = 5000
) -> dict[str, str]:
    return {"cftc_contract_market_code": code, "$where": f"report_date_as_yyyy_mm_dd >= '{since.isoformat()}'",
            "$order": "report_date_as_yyyy_mm_dd", "$limit": str(limit), "$offset": str(offset)}  # fmt: skip


def parse_cot_api(rows: list[dict[str, str]], kind: str) -> pl.DataFrame:
    groups = API_GROUPS[kind]
    markets = API_MARKETS[kind]

    def num(r: dict[str, str], k: str) -> float | None:
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return None

    out = []
    for r in rows:
        code = str(r.get("cftc_contract_market_code", "")).strip()
        if code not in markets:
            continue
        d = date.fromisoformat(str(r["report_date_as_yyyy_mm_dd"])[:10])
        rec: dict[str, object] = {
            "market": markets[code],
            "as_of": d,
            "open_interest": num(r, "open_interest_all"),
        }
        for g, (lo, sh) in groups.items():
            a, b = num(r, lo), num(r, sh)
            rec[f"net_{g}"] = a - b if a is not None and b is not None else None
        rec["published_at"] = _utc(d + timedelta(days=3), 22)
        out.append(rec)
    schema: dict[str, Any] = {"market": pl.String, "as_of": pl.Date, "open_interest": pl.Float64, **{f"net_{g}": pl.Float64 for g in groups},
              "published_at": pl.Datetime("us", "UTC")}  # fmt: skip
    return pl.DataFrame(out, schema=schema)


# ---------------------------------------------------------------- Cboe put/call
def parse_cboe_csv(data: bytes, name: str) -> pl.DataFrame:
    lines = data.decode("latin-1").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE"))
    df = pl.read_csv(
        io.StringIO("\n".join(lines[start:])), infer_schema=False, truncate_ragged_lines=True
    )
    df = df.rename({c: c.strip().upper() for c in df.columns})
    pc = next(c for c in df.columns if "P/C" in c)
    return df.select(
        pl.col("DATE").str.strip_chars().str.to_date("%m/%d/%Y", strict=False).alias("as_of"),
        pl.col(pc).str.strip_chars().cast(pl.Float64, strict=False).alias(name),
    ).filter(pl.col("as_of").is_not_null() & pl.col(name).is_not_null())


def parse_cboe_daily(data: bytes, d: date) -> dict[str, float | date]:
    j = json.loads(data)
    out: dict[str, float | date] = {"as_of": d}
    for r in j.get("ratios", []):
        k = CBOE_RATIOS.get(str(r.get("name", "")).strip().upper())
        if k:
            with contextlib.suppress(TypeError, ValueError):
                out[k] = float(r["value"])
    return out


def cboe_published_at(col: pl.Expr) -> pl.Expr:
    return (col.cast(pl.Datetime("us")) + pl.duration(hours=22)).dt.replace_time_zone("UTC")


# ---------------------------------------------------------------- SEC company profile (SIC)
SPLIT = b"\n<!--MW-SPLIT-->\n"


def parse_profiles(blob_gz: bytes) -> list[dict[str, object]]:
    out = []
    for part in gzip.decompress(blob_gz).split(SPLIT):
        try:
            d = json.loads(part)
        except ValueError:
            continue
        if not isinstance(d, dict) or "sic" not in d:
            continue  # "older filings" pages have no profile
        out.append({"cik": int(d.get("cik") or 0), "name": d.get("name"), "sic": str(d.get("sic") or "") or None,
                    "sic_desc": d.get("sicDescription"), "tickers": [str(t) for t in d.get("tickers") or []]})  # fmt: skip
    return out
