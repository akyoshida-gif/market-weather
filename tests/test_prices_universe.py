"""Universe filters and price normalisation with synthetic inputs (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import polars as pl

from mw.core.config import GuardSettings
from mw.core.pit import validate_pit_frame
from mw.core.storage import GuardedStorage, LocalFS
from mw.ingest import prices
from mw.ingest.base import year_key
from mw.universe import us

NASDAQ = (
    b"Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|"
    b"ETF|NextShares\n"
    b"AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    b"ABCDW|Abcd Corp - Warrant|S|N|N|100|N|N\n"
    b"QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
    b"ZZZT|Test Issue|S|Y|N|100|N|N\n"
    b"PENY|Penny Co - American Depositary Shares|S|N|D|100|N|N\n"
    b"File Creation Time: 1023202621:31|||||||\n"
)
OTHER = (
    b"ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    b"BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B\n"
    b"ABR$D|Arbor Realty 6.375% Series D Preferred Stock|N|ABRpD|N|100|N|ABR-D\n"
    b"SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
    b"File Creation Time: 1023202621:31|||||||\n"
)


def test_us_universe_filters():
    df = us.build(NASDAQ, OTHER, ["SPY", "QQQ"], date(2026, 10, 26))
    kinds = dict(zip(df["symbol"].to_list(), df["kind"].to_list(), strict=True))
    assert kinds == {
        "AAPL": "common",
        "BRK.B": "common",
        "PENY": "common",
        "QQQ": "etf",
        "SPY": "etf",
    }
    row = df.filter(pl.col("symbol") == "BRK.B")
    assert row["yahoo_symbol"][0] == "BRK-B" and row["entity_id"][0] == "US:BRK.B"
    assert df.filter(pl.col("symbol") == "PENY")["listing_issue"][0] is True


def fake_download(symbols, start, end):
    idx = pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-26"])
    cols = pd.MultiIndex.from_product([list(prices.FIELDS), symbols])
    data = [[100.0 + i] * len(cols) for i in range(3)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_prices_ingest_drops_unclosed_session(tmp_path, monkeypatch):
    uni = pl.DataFrame({"yahoo_symbol": ["AAA", "BRK-B"], "entity_id": ["US:AAA", "US:BRK.B"]})
    real = prices.datetime

    class FakeDT(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 10, 26, 18, 0, tzinfo=UTC)  # before 10/26 close

    monkeypatch.setattr(prices, "datetime", FakeDT)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    r = prices.ingest(
        s, "US", uni, date(2026, 10, 20), date(2026, 10, 26), downloader=fake_download
    )
    assert r["with_data"] == 2 and r["missing"] == 0
    df = s.read_parquet(year_key("prices_daily", "US", 2026))
    validate_pit_frame(df)
    assert sorted(set(df["as_of"].to_list())) == [date(2026, 10, 22), date(2026, 10, 23)]
    assert df.filter(pl.col("as_of") == date(2026, 10, 23))["published_at"][0] == datetime(
        2026, 10, 23, 20, 30, tzinfo=UTC
    )
