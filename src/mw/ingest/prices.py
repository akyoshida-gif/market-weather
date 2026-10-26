"""Daily OHLCV for US and JP universes via yfinance batch download (DESIGN §8 us_prices/jp_prices).

- Batch download in chunks (few requests even for ~6,000 symbols); per-chunk failure isolated.
- yfinance `Close` is split-adjusted as of download time; `adj_close` also includes dividends.
  Raw keeps every download vintage; silver keeps the latest. Splits/dividends kept as columns.
- PIT: published_at = session close (exchange calendar) + 30 minutes.
Silver: silver/prices_daily/market={US|JP}/year=YYYY/data.parquet (upsert by entity_id, as_of).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pandas as pd
import polars as pl

from mw.core.calendars import session_close_utc, sessions
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, upsert_timeseries, write_raw_bytes

SOURCE_ID = {"US": "us_prices", "JP": "jp_prices"}
SOURCE_VERSION = "0.1.0"
PUBLISH_LAG = timedelta(minutes=30)
log = logging.getLogger("prices")

FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


class Downloader(Protocol):
    def __call__(self, symbols: list[str], start: date, end: date) -> pd.DataFrame: ...


def yf_download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=True,
        group_by="column",
        threads=True,
        progress=False,
    )
    return df if df is not None else pd.DataFrame()


def to_long(df: pd.DataFrame, symbols: list[str]) -> pl.DataFrame:
    """yfinance wide frame (field x symbol MultiIndex columns) -> long polars frame."""
    if df.empty:
        return pl.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):  # single symbol
        df.columns = pd.MultiIndex.from_product([df.columns, symbols[:1]])
    frames = []
    for field, col in FIELDS.items():
        if field not in df.columns.get_level_values(0):
            continue
        part = df[field].copy()
        part.index.name = "as_of"
        part = part.reset_index().melt(id_vars="as_of", var_name="yahoo_symbol", value_name=col)
        frames.append(part.set_index(["as_of", "yahoo_symbol"]))
    wide = pd.concat(frames, axis=1).reset_index()
    wide["as_of"] = pd.to_datetime(wide["as_of"]).dt.date
    out = pl.from_pandas(wide)
    for col in FIELDS.values():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return out.with_columns(pl.col(list(FIELDS.values())).cast(pl.Float64)).filter(
        pl.col("close").is_not_null()
    )


def _wide_to_parquet(wide: pd.DataFrame) -> bytes:
    flat = wide.copy()
    flat.columns = ["|".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in flat.columns]
    buf = io.BytesIO()
    flat.reset_index().to_parquet(buf, index=False)
    return buf.getvalue()


def fetch(
    symbols: Sequence[str],
    start: date,
    end: date,
    downloader: Downloader = yf_download,
    chunk: int = 200,
    pause: float = 1.0,
) -> tuple[pl.DataFrame, list[bytes], dict[str, str]]:
    frames: list[pl.DataFrame] = []
    raws: list[bytes] = []
    errors: dict[str, str] = {}
    syms = list(symbols)
    for i in range(0, len(syms), chunk):
        part = syms[i : i + chunk]
        try:
            wide = downloader(part, start, end)
            raws.append(_wide_to_parquet(wide))  # raw vintage of what the provider returned
            frames.append(to_long(wide, part))
        except Exception as e:  # isolate chunk failures (P6)
            errors[f"chunk{i // chunk}"] = f"{type(e).__name__}: {e}"[:200]
        time.sleep(pause)
    df = (
        pl.concat([f for f in frames if f.height], how="vertical_relaxed")
        if frames
        else pl.DataFrame()
    )
    return df, raws, errors


def stamp(
    df: pl.DataFrame, market: str, mapping: dict[str, str], ingested_at: datetime, raw_ref: str
) -> pl.DataFrame:
    """Attach entity ids and PIT columns. published_at = session close + 30 min."""
    dates = sorted(set(df["as_of"].to_list()))
    close_map = {
        d: session_close_utc(market, d) + PUBLISH_LAG for d in dates if _is_session(market, d)
    }
    df = df.filter(pl.col("as_of").is_in(list(close_map)))
    pub = pl.Series([close_map[d] for d in df["as_of"].to_list()], dtype=pl.Datetime("us", "UTC"))
    return df.with_columns(
        pl.col("yahoo_symbol").replace_strict(mapping, default=None).alias("entity_id"),
        pub.alias("published_at"),
        pl.lit(ingested_at).dt.convert_time_zone("UTC").alias("ingested_at"),
        pl.lit(SOURCE_ID[market]).alias("source_id"),
        pl.lit(SOURCE_VERSION).alias("source_version"),
        pl.lit(raw_ref).alias("raw_ref"),
    ).filter(
        pl.col("entity_id").is_not_null()
        # drop bars of sessions not yet closed (+lag): yfinance returns partial intraday bars
        & (pl.col("published_at") <= ingested_at)
    )


def _is_session(market: str, d: date) -> bool:
    return bool(sessions(market, d, d))


def ingest(
    storage: GuardedStorage,
    market: str,
    universe: pl.DataFrame,
    start: date,
    end: date,
    downloader: Downloader = yf_download,
    chunk: int = 200,
) -> dict[str, Any]:
    mapping = dict(
        zip(universe["yahoo_symbol"].to_list(), universe["entity_id"].to_list(), strict=True)
    )
    ingested_at = datetime.now(UTC)
    df, raws, errors = fetch(list(mapping), start, end, downloader, chunk)
    refs = []
    for k, raw in enumerate(raws):
        rk = raw_key(
            SOURCE_ID[market],
            ingested_at,
            f"{start}_{end}_part{k:03d}_{ingested_at:%H%M%S}.parquet",
        )
        refs.append(f"{rk}#sha256={write_raw_bytes(storage, rk, raw)}")
    if df.height == 0:
        raise RuntimeError(f"no price rows downloaded ({errors})")
    silver = stamp(
        df,
        market,
        mapping,
        ingested_at,
        raw_ref=";".join(refs[:3]) + (";..." if len(refs) > 3 else ""),
    )
    years = upsert_timeseries(storage, "prices_daily", market, silver)
    got = set(silver["entity_id"].unique().to_list())
    missing = [e for e in mapping.values() if e not in got]
    return {
        "symbols": len(mapping),
        "with_data": len(got),
        "missing": len(missing),
        "missing_sample": missing[:20],
        "rows": silver.height,
        "years": sorted(years),
        "errors": errors,
    }
