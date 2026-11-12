"""Universe filters and price normalisation with synthetic inputs (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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


def test_a_split_in_the_daily_window_refetches_the_full_history(tmp_path, monkeypatch):
    calls: list[tuple[list[str], date]] = []

    def dl(symbols, start, end):
        calls.append((sorted(symbols), start))
        idx = pd.to_datetime(["2026-10-22", "2026-10-23"])
        cols = pd.MultiIndex.from_product([list(prices.FIELDS), symbols])
        df = pd.DataFrame([[50.0] * len(cols)] * 2, index=idx, columns=cols)
        df.loc[:, ("Stock Splits", slice(None))] = 0.0
        if "AAA" in symbols:
            df.loc[pd.Timestamp("2026-10-23"), ("Stock Splits", "AAA")] = 2.0  # 2-for-1 split
        return df

    real = prices.datetime

    tick = {"s": 0}

    class FakeDT(real):
        @classmethod
        def now(cls, tz=None):
            tick["s"] += 1  # each run gets its own time (raw files are named by it)
            return real(2026, 10, 26, 22, 0, tzinfo=UTC) + timedelta(seconds=tick["s"])

    monkeypatch.setattr(prices, "datetime", FakeDT)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    uni = pl.DataFrame({"yahoo_symbol": ["AAA", "BBB"], "entity_id": ["US:AAA", "US:BBB"]})
    r = prices.ingest(s, "US", uni, date(2026, 10, 19), date(2026, 10, 26), downloader=dl)
    assert r["refetch_for_split"]["stocks"] == ["US:AAA"]
    assert calls[-1] == (["AAA"], prices.FULL_START)  # only the split stock, whole history
    assert len(calls) == 2
    # the split stays in the 7-day window for a week: the next runs do not refetch again
    r2 = prices.ingest(s, "US", uni, date(2026, 10, 19), date(2026, 10, 26), downloader=dl)
    assert r2["refetch_for_split"] == {"already_done": 1} and len(calls) == 3
    # a manual backfill does not add a second full download
    prices.ingest(
        s, "US", uni, date(2026, 10, 19), date(2026, 10, 26), downloader=dl, refetch_splits=False
    )
    assert len(calls) == 4


def test_ratio_one_or_missing_is_not_a_split():
    df = pl.DataFrame({"entity_id": ["US:A", "US:B", "US:C", "US:D"], "as_of": [date(2026, 1, 2)] * 4,
                       "splits": [1.0, None, 0.0, 0.1]})  # fmt: skip
    assert prices.split_events(df)["entity_id"].to_list() == ["US:D"]  # 1-for-10 reverse split


def test_a_failed_refetch_keeps_the_daily_rows(tmp_path, monkeypatch):
    state = {"n": 0}

    def dl(symbols, start, end):
        state["n"] += 1
        if state["n"] > 1:
            raise RuntimeError("provider down")
        idx = pd.to_datetime(["2026-10-22"])
        cols = pd.MultiIndex.from_product([list(prices.FIELDS), symbols])
        df = pd.DataFrame([[50.0] * len(cols)], index=idx, columns=cols)
        df.loc[:, ("Stock Splits", slice(None))] = 2.0
        return df

    real = prices.datetime

    class FakeDT(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 10, 26, 22, 0, tzinfo=UTC)

    monkeypatch.setattr(prices, "datetime", FakeDT)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    uni = pl.DataFrame({"yahoo_symbol": ["AAA"], "entity_id": ["US:AAA"]})
    r = prices.ingest(s, "US", uni, date(2026, 10, 19), date(2026, 10, 26), downloader=dl)
    assert r["rows"] == 1 and "error" in r["refetch_for_split"]
    assert not s.exists(prices.refetch_key("US"))  # nothing marked done: retried next run
