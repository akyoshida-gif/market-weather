"""US option chain snapshots via yfinance (DESIGN §8.1 `us_options_chain`).

History cannot be backfilled: each run captures the chain as of now. Two snapshots per session:
- "pre":  pre-market ET, open interest updated overnight (reflects the last completed session).
- "post": after the close, day volume of the session just completed.
`as_of` = last completed XNYS session at capture time; `published_at` = capture time.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
import polars as pl
import yaml

from mw.core.calendars import is_session, previous_session, session_close_utc
from mw.core.config import REPO_ROOT
from mw.core.http import RateLimiter, with_retry
from mw.core.storage import GuardedStorage
from mw.ingest.base import raw_key, silver_key, stamp_pit, write_raw_bytes

SOURCE_ID = "us_options_chain"
SOURCE_VERSION = "0.1.0"
Snapshot = Literal["pre", "post"]

log = logging.getLogger(SOURCE_ID)

CHAIN_RENAME = {
    "contractSymbol": "contract_symbol",
    "lastTradeDate": "last_trade_at",
    "strike": "strike",
    "lastPrice": "last_price",
    "bid": "bid",
    "ask": "ask",
    "change": "change",
    "percentChange": "pct_change",
    "volume": "volume",
    "openInterest": "open_interest",
    "impliedVolatility": "implied_vol",
    "inTheMoney": "in_the_money",
    "contractSize": "contract_size",
    "currency": "currency",
}

UNDERLYING_FIELDS = {
    "regularMarketPrice": pl.Float64,
    "regularMarketPreviousClose": pl.Float64,
    "regularMarketVolume": pl.Float64,
    "regularMarketTime": pl.Int64,
    "postMarketPrice": pl.Float64,
    "preMarketPrice": pl.Float64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "sharesOutstanding": pl.Float64,
    "marketState": pl.String,
    "quoteType": pl.String,
}


class ChainProvider(Protocol):
    def expiries(self, symbol: str) -> list[str]: ...
    def chain(self, symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Return (calls, puts, underlying_info)."""
        ...


class YFinanceProvider:
    def __init__(self, limiter: RateLimiter) -> None:
        import yfinance as yf

        self._yf = yf
        self.limiter = limiter

    def expiries(self, symbol: str) -> list[str]:
        def _do() -> list[str]:
            self.limiter.acquire()
            return list(self._yf.Ticker(symbol).options)

        return with_retry(_do, attempts=3)

    def chain(self, symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        def _do() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
            self.limiter.acquire()
            oc = self._yf.Ticker(symbol).option_chain(expiry)
            return oc.calls, oc.puts, dict(getattr(oc, "underlying", {}) or {})

        return with_retry(_do, attempts=3)


@dataclass
class SnapshotResult:
    as_of: date
    snapshot: Snapshot
    captured_at: datetime
    chains: pl.DataFrame
    underlyings: pl.DataFrame
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def load_watchlist(path: Path | None = None) -> tuple[list[str], int]:
    p = path or REPO_ROOT / "config" / "universe" / "options_watchlist.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    tickers: list[str] = list(cfg["tickers"])
    if cfg.get("include_theme_members"):
        themes = yaml.safe_load(
            (REPO_ROOT / "config" / "universe" / "themes.yaml").read_text(encoding="utf-8")
        )
        for members in themes.values():
            tickers.extend(m for m in members if "." not in m)  # US symbols only
    seen: dict[str, None] = dict.fromkeys(t.upper() for t in tickers)
    return list(seen), int(cfg.get("max_dte_days", 200))


def last_completed_session(captured_at: datetime) -> date:
    """Last XNYS session whose close is <= captured_at."""
    d = captured_at.astimezone(UTC).date()
    for candidate in (d, previous_session("US", d)):
        if is_session("US", candidate) and session_close_utc("US", candidate) <= captured_at:
            return candidate
    return previous_session("US", previous_session("US", d))


def select_expiries(expiries: Sequence[str], as_of: date, max_dte: int) -> list[str]:
    """Expiries within max_dte days, plus the first one beyond (for term-structure)."""
    out: list[str] = []
    for e in sorted(expiries):
        dte = (date.fromisoformat(e) - as_of).days
        if dte < 0:
            continue
        out.append(e)
        if dte > max_dte:
            break
    return out


def _fetch_one(
    provider: ChainProvider, symbol: str, as_of: date, max_dte: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    underlying: dict[str, Any] = {}
    for exp in select_expiries(provider.expiries(symbol), as_of, max_dte):
        calls, puts, und = provider.chain(symbol, exp)
        underlying = und or underlying
        for kind, df in (("C", calls), ("P", puts)):
            if df is None or df.empty:
                continue
            df = df.copy()
            df["expiry"] = exp
            df["option_type"] = kind
            frames.append(df)
    if not frames:
        raise ValueError("no option contracts returned")
    out = pd.concat(frames, ignore_index=True)
    out["underlying"] = symbol
    return out, underlying


def capture(
    provider: ChainProvider,
    symbols: Sequence[str],
    snapshot: Snapshot,
    max_dte: int,
    captured_at: datetime,
    workers: int = 3,
) -> SnapshotResult:
    as_of = last_completed_session(captured_at)
    chain_frames: list[pd.DataFrame] = []
    und_rows: list[dict[str, Any]] = []
    result = SnapshotResult(as_of, snapshot, captured_at, pl.DataFrame(), pl.DataFrame())

    def task(sym: str) -> tuple[str, pd.DataFrame | None, dict[str, Any], str | None]:
        try:
            df, und = _fetch_one(provider, sym, as_of, max_dte)
            return sym, df, und, None
        except Exception as e:  # failure isolation per underlying (P6)
            return sym, None, {}, f"{type(e).__name__}: {e}"[:300]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, df, und, err in ex.map(task, symbols):
            if err is not None or df is None:
                result.failed[sym] = err or "unknown"
                log.warning("underlying failed", extra={"data": {"symbol": sym, "error": err}})
                continue
            result.ok.append(sym)
            chain_frames.append(df)
            row: dict[str, Any] = {"underlying": sym}
            row.update({k: und.get(k) for k in UNDERLYING_FIELDS})
            und_rows.append(row)

    if chain_frames:
        result.chains = _normalize_raw_chains(pd.concat(chain_frames, ignore_index=True))
    schema: dict[str, Any] = {"underlying": pl.String, **UNDERLYING_FIELDS}
    result.underlyings = pl.DataFrame(und_rows, schema=schema)
    return result


def _normalize_raw_chains(df: pd.DataFrame) -> pl.DataFrame:
    """Provider frame -> stable typed polars frame (still provider semantics)."""
    df = df.rename(columns=CHAIN_RENAME)
    for col in CHAIN_RENAME.values():
        if col not in df.columns:
            df[col] = None
    if "last_trade_at" in df.columns:
        df["last_trade_at"] = pd.to_datetime(df["last_trade_at"], utc=True, errors="coerce")
    cols = ["underlying", "expiry", "option_type", *CHAIN_RENAME.values()]
    out = pl.from_pandas(df[cols])
    return out.with_columns(
        pl.col("expiry").str.to_date(),
        pl.col("strike", "last_price", "bid", "ask", "change", "pct_change", "implied_vol").cast(
            pl.Float64
        ),
        pl.col("volume", "open_interest").cast(pl.Float64),
        pl.col("in_the_money").cast(pl.Boolean),
        pl.col("contract_symbol", "contract_size", "currency").cast(pl.String),
    )


IV_PLACEHOLDER_MAX = 1e-4  # Yahoo returns ~1e-5 when IV is unavailable (observed off-hours)


def quality_metrics(chains: pl.DataFrame) -> dict[str, float]:
    """Snapshot-level quality: share of zero OI and placeholder IV (off-hours resets)."""
    if chains.height == 0:
        return {"rows": 0, "pct_oi_zero": 1.0, "pct_iv_placeholder": 1.0}
    oi_zero = chains.select((pl.col("open_interest").fill_null(0) <= 0).mean()).item()
    iv_ph = chains.select((pl.col("implied_vol").fill_null(0) <= IV_PLACEHOLDER_MAX).mean()).item()
    return {
        "rows": chains.height,
        "pct_oi_zero": round(float(oi_zero), 4),
        "pct_iv_placeholder": round(float(iv_ph), 4),
    }


def _to_parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


def _name(as_of: date, snapshot: Snapshot, degraded: bool) -> str:
    return f"{as_of.isoformat()}_{snapshot}{'_degraded' if degraded else ''}.parquet"


def silver_chain_key(as_of: date, snapshot: Snapshot, degraded: bool = False) -> str:
    return silver_key("options_chain", "US", as_of, _name(as_of, snapshot, degraded))


def silver_underlying_key(as_of: date, snapshot: Snapshot, degraded: bool = False) -> str:
    return silver_key("options_underlying", "US", as_of, _name(as_of, snapshot, degraded))


def is_degraded(q: dict[str, float]) -> bool:
    """Most OI zero or IV placeholder -> provider off-hours reset; keep but don't count as done."""
    return q["pct_oi_zero"] > 0.5 or q["pct_iv_placeholder"] > 0.5


def persist(storage: GuardedStorage, res: SnapshotResult, degraded: bool = False) -> dict[str, Any]:
    """Raw (immutable, timestamped) then silver (PIT-stamped, one file per as_of+snapshot)."""
    ingested_at = datetime.now(UTC)
    stamp = res.captured_at.strftime("%H%M%S")
    out: dict[str, Any] = {}
    for table, df, silver in (
        ("chains", res.chains, silver_chain_key(res.as_of, res.snapshot, degraded)),
        ("underlyings", res.underlyings, silver_underlying_key(res.as_of, res.snapshot, degraded)),
    ):
        if df.height == 0:
            continue
        rk = raw_key(SOURCE_ID, ingested_at, f"{res.snapshot}_{table}_{stamp}.parquet")
        # degraded silver may be overwritten by a later re-capture; good silver is final

        sha = write_raw_bytes(storage, rk, _to_parquet_bytes(df))
        silver_df = stamp_pit(
            df.with_columns(
                (pl.lit("US:") + pl.col("underlying")).alias("entity_id"),
                pl.lit(res.snapshot).alias("snapshot"),
            ),
            as_of=res.as_of,
            published_at=res.captured_at,
            ingested_at=ingested_at,
            source_id=SOURCE_ID,
            source_version=SOURCE_VERSION,
            raw_ref=f"{rk}#sha256={sha}",
        )
        if table == "chains":
            # Silver: placeholder IV -> null (raw keeps provider values).
            silver_df = silver_df.with_columns(
                (pl.col("expiry") - pl.col("as_of")).dt.total_days().alias("dte"),
                pl.when(pl.col("implied_vol") > IV_PLACEHOLDER_MAX)
                .then(pl.col("implied_vol"))
                .otherwise(None)
                .alias("implied_vol"),
            )
        storage.write_parquet(silver, silver_df)
        out[table] = {"raw": rk, "silver": silver, "rows": silver_df.height}
    return out
