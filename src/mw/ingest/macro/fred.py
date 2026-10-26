"""FRED series via the official API (DESIGN §8.3 `fred`). Key from env FRED_API_KEY (never logged).

PIT: observations are stored with a conservative publish-lag rule per series
(published_at = as_of + lag_days at 23:00 UTC). Revisions: raw keeps every vintage; silver keeps
the latest value (ALFRED vintages can be added later if revisions matter).
ICE BofA series (BAMLH0A0HYM2, BAMLC0A0CM) keep only 3 rolling years on FRED -> we archive.
ToS: personal use; no unreasonable bandwidth.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from mw.core.http import HttpClient
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, stamp_pit, upsert_timeseries, write_raw_bytes

SOURCE_ID = "fred"
SOURCE_VERSION = "0.1.0"
API = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> publish lag in calendar days after the observation date (conservative)
SERIES: dict[str, int] = {
    "DGS2": 1,
    "DGS10": 1,
    "DGS30": 1,
    "DFF": 1,
    "T10YIE": 1,
    "T10Y2Y": 1,
    "DFII10": 1,
    "BAMLH0A0HYM2": 1,
    "BAMLC0A0CM": 1,
    "DTWEXBGS": 8,  # H.10 weekly release
    "DEXJPUS": 8,
    "DCOILWTICO": 9,  # weekly EIA-based updates
    "DCOILBRENTEU": 9,
    "NFCI": 6,  # week ending Friday, released following Wednesday
    # --- rates curve / funding (ADR-0012) ---
    "DGS3MO": 1,
    "DGS1": 1,
    "DGS5": 1,
    "T10Y3M": 1,
    "SOFR": 1,
    "RRPONTSYD": 1,  # overnight reverse repo usage (liquidity)
    "BAA10Y": 1,  # corporate credit spread
    # --- liquidity / money (weekly, monthly) ---
    "WALCL": 2,  # Fed balance sheet, Wed level released Thu
    "WTREGEN": 2,  # Treasury General Account
    "M2SL": 55,  # monthly; date = 1st of month, released ~4th week of the following month
    # --- economy ---
    "ICSA": 6,  # initial claims, week ending Sat, released Thu
    "CPIAUCSL": 45,
    "UNRATE": 40,
    "INDPRO": 48,
    "UMCSENT": 35,
    "PCOPPUSDM": 45,  # copper price (monthly)
    "DHHNGSP": 8,  # natural gas (weekly EIA updates)
}


def fetch_series(http: HttpClient, series_id: str, start: date | None) -> bytes:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    if start:
        params["observation_start"] = start.isoformat()
    try:
        resp = http.get(API, params=params)
        resp.raise_for_status()
    except Exception as e:  # never surface the URL (contains api_key) in logs/registry
        status = getattr(getattr(e, "response", None), "status_code", None)
        raise RuntimeError(
            f"FRED request failed for {series_id}: {type(e).__name__} {status}"
        ) from None
    return resp.content


def parse(series_id: str, data: bytes) -> pl.DataFrame:
    obs = json.loads(data)["observations"]
    df = pl.DataFrame(
        {"as_of": [o["date"] for o in obs], "value": [o["value"] for o in obs]},
        schema={"as_of": pl.String, "value": pl.String},
    )
    return df.select(
        pl.col("as_of").str.to_date("%Y-%m-%d"),
        pl.col("value").replace({".": None}).cast(pl.Float64),
    ).filter(pl.col("value").is_not_null())


def ingest(storage: GuardedStorage, http: HttpClient, since_days: int | None) -> dict[str, Any]:
    ingested_at = datetime.now(UTC)
    start = None if since_days is None else ingested_at.date() - timedelta(days=since_days)
    frames: list[pl.DataFrame] = []
    errors: dict[str, str] = {}
    for sid, lag in SERIES.items():
        try:
            data = fetch_series(http, sid, start)
            rk = raw_key(SOURCE_ID, ingested_at, f"{sid}_{ingested_at:%H%M%S}.json")
            sha = write_raw_bytes(storage, rk, data)
            df = parse(sid, data)
            pub = pl.col("as_of").cast(pl.Datetime("us")) + pl.duration(days=lag, hours=23)
            df = stamp_pit(
                df.with_columns(pl.lit(f"MACRO:FRED:{sid}").alias("entity_id")),
                as_of=ingested_at.date(),
                published_at=ingested_at,
                ingested_at=ingested_at,
                source_id=SOURCE_ID,
                source_version=SOURCE_VERSION,
                raw_ref=f"{rk}#sha256={sha}",
            ).with_columns(df["as_of"].alias("as_of"))
            df = df.with_columns(pub.dt.replace_time_zone("UTC").alias("published_at"))
            frames.append(df)
        except Exception as e:  # isolate per series
            errors[sid] = f"{type(e).__name__}: {e}"[:200]
    if not frames:
        raise RuntimeError(f"all FRED series failed: {errors}")
    all_df = pl.concat(frames, how="vertical_relaxed")
    upsert_timeseries(storage, "fred", "GLOBAL", all_df)
    return {"rows": all_df.height, "series_ok": len(frames), "errors": errors}
