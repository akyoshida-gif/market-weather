"""moomoo OpenAPI option-chain provider (DISABLED until owner's account is funded; ADR-0006).

Implements the same ChainProvider protocol as YFinanceProvider so it can be swapped in.
Requires the OpenD gateway running locally (owner PC, self-hosted runner) — never in cloud runners,
because OpenD holds the brokerage login.

Field names below follow the OpenAPI docs (get_option_expiration_date, get_option_chain,
get_market_snapshot) but are UNVERIFIED against a live OpenD; verify before enabling.
Rate limits (docs): option chain 10 req/30s (date span <= 30 days); snapshot 60 req/30s, 400 codes.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from mw.core.http import RateLimiter, with_retry

SNAPSHOT_BATCH = 400

# moomoo snapshot column -> yfinance-compatible column (provider schema used by options_chain)
SNAPSHOT_MAP = {
    "code": "contractSymbol",
    "update_time": "lastTradeDate",
    "option_strike_price": "strike",
    "last_price": "lastPrice",
    "bid_price": "bid",
    "ask_price": "ask",
    "volume": "volume",
    "option_open_interest": "openInterest",
    "option_implied_volatility": "impliedVolatility",  # percent in moomoo -> divided by 100
}


class MoomooProvider:
    def __init__(self, host: str = "127.0.0.1", port: int = 11111) -> None:
        from moomoo import OpenQuoteContext  # type: ignore[import-not-found]

        self.ctx = OpenQuoteContext(host=host, port=port)
        self.chain_limiter = RateLimiter(10 / 30, burst=1)
        self.snap_limiter = RateLimiter(60 / 30, burst=1)
        self._ok: Any = None
        import moomoo  # type: ignore[import-not-found]

        self._ok = moomoo.RET_OK

    @staticmethod
    def _code(symbol: str) -> str:
        return f"US.{symbol}"

    def _call(self, limiter: RateLimiter, fn: Any, *args: Any, **kwargs: Any) -> Any:
        def _do() -> Any:
            limiter.acquire()
            ret, data = fn(*args, **kwargs)
            if ret != self._ok:
                raise RuntimeError(str(data))
            return data

        return with_retry(_do, attempts=3, base_delay=5.0)

    def expiries(self, symbol: str) -> list[str]:
        df = self._call(self.chain_limiter, self.ctx.get_option_expiration_date, self._code(symbol))
        return sorted(str(x) for x in df["strike_time"])

    def chain(self, symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        d = date.fromisoformat(expiry)
        contracts = self._call(
            self.chain_limiter,
            self.ctx.get_option_chain,
            self._code(symbol),
            start=d.isoformat(),
            end=d.isoformat(),
        )
        codes = list(contracts["code"])
        snaps = [
            self._call(
                self.snap_limiter, self.ctx.get_market_snapshot, codes[i : i + SNAPSHOT_BATCH]
            )
            for i in range(0, len(codes), SNAPSHOT_BATCH)
        ]
        snap = pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame()
        snap = snap.merge(contracts[["code", "option_type"]], on="code", how="left")
        out = snap.rename(columns=SNAPSHOT_MAP)
        out["impliedVolatility"] = out["impliedVolatility"] / 100.0
        for col in ("change", "percentChange", "inTheMoney", "contractSize"):
            out[col] = None
        out["currency"] = "USD"
        calls = out[out["option_type"] == "CALL"].drop(columns=["option_type"])
        puts = out[out["option_type"] == "PUT"].drop(columns=["option_type"])
        und = self._call(self.snap_limiter, self.ctx.get_market_snapshot, [self._code(symbol)])
        info = {"regularMarketPrice": float(und["last_price"].iloc[0])} if len(und) else {}
        return calls, puts, info

    def close(self) -> None:
        self.ctx.close()
