"""US trading halts (all exchanges) as catalyst events.

Live: Nasdaq Trader RSS https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts[&haltdate=MMDDYYYY]
      (reason codes T1 news pending, T2 news released, LUDP volatility pause; <=1 req/min)
History: NYSE https://www.nyse.com/api/trade-halts/historical/download?haltDateFrom=&haltDateTo=
      (CSV, all US exchanges, ~2021-; undocumented endpoint, ADR-0009)
published_at = halt time (halts are public immediately).
"""

from __future__ import annotations

import io
import re
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

NY = ZoneInfo("America/New_York")
SOURCE_ID = "trading_halts"
SOURCE_VERSION = "0.1.0"
NASDAQ_RSS = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
NYSE_HIST = (
    "https://www.nyse.com/api/trade-halts/historical/download?haltDateFrom={start}&haltDateTo={end}"
)

ITEM_RE = re.compile(r"<item>(.*?)</item>", re.S)


def _tag(body: str, name: str) -> str:
    m = re.search(rf"<ndaq:{name}>(.*?)</ndaq:{name}>", body, re.S)
    return m.group(1).strip() if m else ""


def _et(d: str, t: str, fmt_d: str) -> datetime | None:
    if not d or not t:
        return None
    t = t.split(".")[0]
    try:
        return datetime.strptime(f"{d} {t}", f"{fmt_d} %H:%M:%S").replace(tzinfo=NY).astimezone(UTC)
    except ValueError:
        return None


def parse_nasdaq_rss(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    out = []
    for body in ITEM_RE.findall(text):
        halt = _et(_tag(body, "HaltDate"), _tag(body, "HaltTime"), "%m/%d/%Y")
        if halt is None:
            continue
        rd = _tag(body, "ResumptionDate")
        out.append({
            "symbol": _tag(body, "IssueSymbol"),
            "name": _tag(body, "IssueName"),
            "exchange": _tag(body, "Market"),
            "reason": _tag(body, "ReasonCode"),
            "halted_at": halt,
            "resume_trade_at": _et(rd, _tag(body, "ResumptionTradeTime"), "%m/%d/%Y"),
            "source": "nasdaq_rss",
        })  # fmt: skip
    return out


def parse_nyse_csv(data: bytes) -> list[dict[str, Any]]:
    df = pl.read_csv(io.BytesIO(data), infer_schema=False)
    out = []
    for r in df.iter_rows(named=True):
        halt = _et(r.get("Halt Date") or "", r.get("Halt Time") or "", "%Y-%m-%d")
        if halt is None:
            continue
        out.append({
            "symbol": (r.get("Symbol") or "").strip(),
            "name": (r.get("Name") or "").strip().strip('"'),
            "exchange": r.get("Exchange") or "",
            "reason": r.get("Reason") or "",
            "halted_at": halt,
            "resume_trade_at": _et(
                r.get("Resume Date") or "", r.get("NYSE Resume Time") or "", "%Y-%m-%d"
            ),
            "source": "nyse_hist",
        })  # fmt: skip
    return out


def to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, schema_overrides={"resume_trade_at": pl.Datetime("us", "UTC")})
    return df.with_columns(
        (pl.lit("US:") + pl.col("symbol")).alias("entity_id"),
        pl.col("halted_at").alias("published_at"),
        pl.col("halted_at").dt.convert_time_zone("America/New_York").dt.date().alias("as_of"),
    )


def nyse_url(start: date, end: date) -> str:
    return NYSE_HIST.format(start=start.isoformat(), end=end.isoformat())
