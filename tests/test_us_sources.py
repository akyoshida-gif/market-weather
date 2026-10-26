"""Contract tests for US/macro adapters with synthetic payloads."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl

from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, LocalFS
from mw.ingest.base import upsert_timeseries, year_key
from mw.ingest.macro import fred
from mw.ingest.us import cboe, finra

REGSHO = (
    b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    b"20261022|A|493721.435214|7862|997855.818340|B,Q,N\n"
    b"20261022|ZZ|0|0|0|Q\n"
)

SI = (
    b'"accountingYearMonthNumber"|"symbolCode"|"issueName"|"issuerServicesGroupExchangeCode"|'
    b'"marketClassCode"|"currentShortPositionQuantity"|"previousShortPositionQuantity"|'
    b'"stockSplitFlag"|"averageDailyVolumeQuantity"|"daysToCoverQuantity"|"revisionFlag"|'
    b'"changePercent"|"changePreviousNumber"|"settlementDate"\n'
    b'"20260930"|"A"|"Agilent"|"A"|"NYSE"|"5240309"|"5078629"||"2950874"|"1.78"||"3.18"|'
    b'"161680"|"2026-09-30"\n'
)


def test_parse_regsho():
    df = finra.parse_regsho(REGSHO)
    assert df["short_volume"][0] == 493721.435214
    assert df["short_volume_ratio"].to_list()[1] is None
    assert df["trade_date"].unique().to_list() == [date(2026, 10, 22)]


def test_regsho_published_at_dst():
    assert finra.regsho_published_at(date(2026, 10, 22)) == datetime(2026, 10, 22, 22, tzinfo=UTC)
    assert finra.regsho_published_at(date(2026, 12, 1)) == datetime(2026, 12, 1, 23, tzinfo=UTC)


def test_parse_short_interest():
    df = finra.parse_short_interest(SI)
    assert df["short_interest"][0] == 5240309.0
    assert df["settlement_date"][0] == date(2026, 9, 30)


def test_settlement_dates():
    got = finra.settlement_dates(date(2026, 8, 1), date(2026, 11, 30))
    assert date(2026, 8, 14) in got  # 15th is Saturday -> Friday 14th
    assert date(2026, 8, 31) in got
    assert date(2026, 10, 15) in got
    assert date(2026, 11, 30) in got
    assert len(got) == 8


def test_si_rule():
    # 2026-10-15 settlement -> 7 sessions later = 2026-10-26
    assert finra.si_rule_published_at(date(2026, 10, 15)).date() == date(2026, 10, 26)


def test_cboe_parse_ohlc_and_single():
    ohlc = b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n"
    single = b"DATE,VVIX\n10/22/2026,107.94\n"
    a = cboe.parse("VIX", ohlc)
    b = cboe.parse("VVIX", single)
    assert a["close"][0] == 17.24 and a["as_of"][0] == date(1990, 1, 2)
    assert b["close"][0] == 107.94 and b["open"][0] is None


def test_fred_parse_missing_values():
    payload = json.dumps(
        {
            "observations": [
                {"date": "2026-10-21", "value": "4.1"},
                {"date": "2026-10-22", "value": "."},
            ]
        }
    ).encode()
    df = fred.parse("DGS10", payload)
    assert df.height == 1 and df["value"][0] == 4.1


def test_upsert_timeseries(tmp_path):
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    a = pl.DataFrame(
        {"entity_id": ["X", "X"], "as_of": [date(2025, 12, 31), date(2026, 1, 2)], "v": [1, 2]}
    )
    upsert_timeseries(s, "t", "US", a)
    b = pl.DataFrame({"entity_id": ["X"], "as_of": [date(2026, 1, 2)], "v": [3]})
    upsert_timeseries(s, "t", "US", b)
    y26 = s.read_parquet(year_key("t", "US", 2026))
    assert y26["v"].to_list() == [3]
    assert s.read_parquet(year_key("t", "US", 2025))["v"].to_list() == [1]


def test_cboe_ingest_twice_same_day(tmp_path, monkeypatch):
    """Daily schedule runs twice a day; raw keys must not collide (regression)."""
    import itertools

    from mw.ingest.us import cboe as cb

    class Resp:
        content = b"DATE,OPEN,HIGH,LOW,CLOSE\n10/22/2026,1,1,1,19.0\n"

        def raise_for_status(self):
            return None

    class Http:
        def get(self, url):
            return Resp()

    times = itertools.count()
    real = cb.datetime

    class FakeDT(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 10, 23, 2, 30, next(times), tzinfo=UTC)

    monkeypatch.setattr(cb, "datetime", FakeDT)
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    cb.ingest(s, Http(), since_days=None)
    cb.ingest(s, Http(), since_days=None)
    assert len(s.list("raw/vol_indices/")) == 12
