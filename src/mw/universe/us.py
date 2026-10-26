"""US universe: all listed common stocks (incl. small caps / penny stocks) + configured ETFs.

Source: Nasdaq Trader symbol directory (updated nightly):
  https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt  (Nasdaq-listed)
  https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt   (NYSE, NYSE American, Arca, Cboe)
Pipe-delimited, last line "File Creation Time: MMDDYYYYHH:MM".
Common-stock filter: not ETF, not test issue, and name does not indicate warrant/right/unit/
preferred/notes. ADRs are kept (many small caps are ADRs). Snapshot stored daily (PIT, DESIGN §6.4).
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import polars as pl
import yaml

from mw.core.config import REPO_ROOT

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCHANGES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}

NON_COMMON = (
    r"(?i)\b(warrants?|rights?|units?|preferred|notes?|debentures?|subordinated|perpetual|"
    r"depositary shares,? each representing a 1/\d+)\b|% series"
)


def _read(data: bytes) -> pl.DataFrame:
    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return pl.read_csv(io.StringIO("\n".join(lines)), separator="|", infer_schema=False)


def parse_nasdaq(data: bytes) -> pl.DataFrame:
    df = _read(data)
    return df.select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Security Name").alias("name"),
        pl.lit("Nasdaq").alias("exchange"),
        pl.col("Market Category").alias("nasdaq_tier"),
        pl.col("Financial Status").alias("financial_status"),
        (pl.col("ETF") == "Y").alias("is_etf"),
        (pl.col("Test Issue") == "Y").alias("is_test"),
    )


def parse_other(data: bytes) -> pl.DataFrame:
    df = _read(data)
    return df.select(
        pl.col("ACT Symbol").alias("symbol"),
        pl.col("Security Name").alias("name"),
        pl.col("Exchange").replace_strict(EXCHANGES, default="Other").alias("exchange"),
        pl.lit(None, dtype=pl.String).alias("nasdaq_tier"),
        pl.lit(None, dtype=pl.String).alias("financial_status"),
        (pl.col("ETF") == "Y").alias("is_etf"),
        (pl.col("Test Issue") == "Y").alias("is_test"),
    )


def load_etfs(path: Path | None = None) -> list[str]:
    p = path or REPO_ROOT / "config" / "universe" / "us_etfs.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    out: list[str] = []
    for group in cfg.values():
        out.extend(group)
    return list(dict.fromkeys(out))


def to_yahoo(symbol: str) -> str:
    return symbol.replace(".", "-")


def build(nasdaq: bytes, other: bytes, etfs: list[str], as_of: date) -> pl.DataFrame:
    """Return universe snapshot: one row per symbol with `kind` in {common, etf}."""
    df = pl.concat([parse_nasdaq(nasdaq), parse_other(other)], how="vertical_relaxed")
    df = df.filter(~pl.col("is_test")).unique(subset=["symbol"], keep="first")
    is_common = (
        ~pl.col("is_etf")
        & ~pl.col("symbol").str.contains(r"[\$\^=]")
        & ~pl.col("name").str.contains(NON_COMMON)
    )
    common = df.filter(is_common).with_columns(pl.lit("common").alias("kind"))
    etf = df.filter(pl.col("symbol").is_in(etfs)).with_columns(pl.lit("etf").alias("kind"))
    out = pl.concat([common, etf], how="vertical_relaxed").unique(subset=["symbol"], keep="last")
    return out.with_columns(
        pl.col("symbol").map_elements(to_yahoo, return_dtype=pl.String).alias("yahoo_symbol"),
        (pl.lit("US:") + pl.col("symbol")).alias("entity_id"),
        pl.lit(as_of).alias("as_of"),
        # Nasdaq financial status: D=deficient, E=delinquent, Q=bankrupt, G/H/J/K = combos
        pl.col("financial_status")
        .is_in(["D", "E", "Q", "G", "H", "J", "K"])
        .alias("listing_issue"),
    ).sort("symbol")
