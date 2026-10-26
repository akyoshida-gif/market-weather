"""SEC filings / halts parsers and PIT dissemination rule (synthetic payloads)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl

from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, LocalFS
from mw.ingest.base import date_key, upsert_by_date
from mw.ingest.us import halts, sec_filings

ATOM = b"""<?xml version="1.0"?><feed>
<entry>
<title>8-K - Tiny Bio Inc. (0001234567) (Filer)</title>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-10-23
&lt;br&gt;Item 3.01: Notice of Delisting
&lt;br&gt;Item 9.01: Financial Statements and Exhibits</summary>
<updated>2026-10-23T08:15:00-04:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0001234567-26-000001</id>
</entry>
<entry>
<title>SCHEDULE 13D - Tiny Bio Inc. (0001234567) (Subject)</title>
<summary type="html">x</summary>
<updated>2026-10-23T18:01:00-04:00</updated>
<category term="SCHEDULE 13D"/>
<id>urn:tag:sec.gov,2008:accession-number=0009999999-26-000002</id>
</entry>
</feed>"""

TICKERS = json.dumps(
    {"fields": ["cik", "name", "ticker", "exchange"],
     "data": [[1234567, "TINY BIO", "TINY", "Nasdaq"], [1234567, "TINY BIO", "TINYW", "Nasdaq"],
              [7654321, "OTC CO", "OTCC", "OTC"]]}
).encode()  # fmt: skip


def test_parse_atom_and_items():
    rows = sec_filings.parse_atom(ATOM)
    assert len(rows) == 2
    assert rows[0]["items"] == "3.01,9.01" and rows[0]["form"] == "8-K"
    assert rows[0]["accepted_at"] == datetime(2026, 10, 23, 12, 15, tzinfo=UTC)
    assert rows[1]["role"] == "Subject" and rows[1]["form"] == "SCHEDULE 13D"


def test_dissemination_rule():
    ny = sec_filings.NY
    # during window -> immediate
    a = datetime(2026, 10, 23, 8, 15, tzinfo=ny)
    assert sec_filings.dissemination_time(a, "8-K") == a.astimezone(UTC)
    # Friday 18:01 ET 8-K -> Monday 06:00 ET
    b = datetime(2026, 10, 23, 18, 1, tzinfo=ny)
    assert sec_filings.dissemination_time(b, "8-K") == datetime(2026, 10, 26, 10, 0, tzinfo=UTC)
    # Form 4 at 21:00 ET -> same evening (22:00 cutoff)
    c = datetime(2026, 10, 22, 21, 0, tzinfo=ny)
    assert sec_filings.dissemination_time(c, "4") == c.astimezone(UTC)
    # 05:00 ET on a session day -> 06:00 same day
    d = datetime(2026, 10, 22, 5, 0, tzinfo=ny)
    assert sec_filings.dissemination_time(d, "8-K") == datetime(2026, 10, 22, 10, 0, tzinfo=UTC)


def test_to_frame_maps_ticker_and_dates():
    tick = sec_filings.ticker_map(TICKERS)
    assert tick.height == 1 and tick["ticker"][0] == "TINY"
    df = sec_filings.to_frame(sec_filings.parse_atom(ATOM), tick)
    assert df["entity_id"].to_list() == ["US:TINY", "US:TINY"]
    assert df["as_of"].to_list() == [date(2026, 10, 23), date(2026, 10, 26)]


def test_parse_submissions_since():
    sub = json.dumps({"cik": "1234567", "name": "Tiny", "filings": {"recent": {
        "accessionNumber": ["a1", "a2"], "filingDate": ["2026-10-22", "2019-01-02"],
        "acceptanceDateTime": ["2026-10-22T22:32:20.000Z", "2019-01-02T15:00:00.000Z"],
        "form": ["4", "8-K"], "items": ["", "2.02"]}}}).encode()  # fmt: skip
    rows = sec_filings.parse_submissions(sub, date(2021, 1, 1))
    assert [r["accession"] for r in rows] == ["a1"]


RSS = b"""<?xml version="1.0"?><rss xmlns:ndaq="http://www.nasdaqtrader.com/"><channel>
<item><ndaq:HaltDate>10/23/2026</ndaq:HaltDate><ndaq:HaltTime>09:50:00.000</ndaq:HaltTime>
<ndaq:IssueSymbol>TINY</ndaq:IssueSymbol><ndaq:IssueName>Tiny</ndaq:IssueName><ndaq:Market>NASDAQ</ndaq:Market>
<ndaq:ReasonCode>T1</ndaq:ReasonCode><ndaq:ResumptionDate>10/23/2026</ndaq:ResumptionDate>
<ndaq:ResumptionTradeTime>10:30:00.000</ndaq:ResumptionTradeTime></item>
</channel></rss>"""
NYSE = (
    b"Halt Date,Halt Time,Symbol,Name,Exchange,Reason,Resume Date,NYSE Resume Time\n"
    b'2026-10-23,19:50:00,IPW,"""iPower, Inc.""",Nasdaq,News Pending,,\n'
)


def test_halts_parsers(tmp_path):
    a = halts.parse_nasdaq_rss(RSS)
    assert a[0]["halted_at"] == datetime(2026, 10, 23, 13, 50, tzinfo=UTC)
    assert a[0]["resume_trade_at"] == datetime(2026, 10, 23, 14, 30, tzinfo=UTC)
    b = halts.parse_nyse_csv(NYSE)
    assert (
        b[0]["symbol"] == "IPW"
        and b[0]["reason"] == "News Pending"
        and b[0]["resume_trade_at"] is None
    )
    df = halts.to_frame(a + b)
    s = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    upsert_by_date(s, "trading_halts", "US", df, ["symbol", "halted_at", "source"])
    upsert_by_date(s, "trading_halts", "US", df, ["symbol", "halted_at", "source"])
    out = s.read_parquet(date_key("trading_halts", "US", date(2026, 10, 23)))
    assert out.height == 2 and set(out["entity_id"]) == {"US:TINY", "US:IPW"}
    assert out.schema["published_at"] == pl.Datetime("us", "UTC")


def test_older_submission_files():
    main = json.dumps({"cik": "1234567", "name": "Tiny", "filings": {
        "recent": {"accessionNumber": [], "filingDate": [], "acceptanceDateTime": [], "form": [], "items": []},
        "files": [{"name": "CIK0001234567-submissions-001.json", "filingFrom": "2012-01-01", "filingTo": "2019-12-31"},
                  {"name": "CIK0001234567-submissions-002.json", "filingFrom": "2001-01-01", "filingTo": "2009-12-31"}]}}).encode()  # fmt: skip
    assert sec_filings.older_files(main, date(2010, 1, 1)) == ["CIK0001234567-submissions-001.json"]
    extra = json.dumps({"accessionNumber": ["a9"], "filingDate": ["2015-04-30"],
                        "acceptanceDateTime": ["2015-04-30T20:05:00.000Z"], "form": ["8-K"], "items": ["2.02,9.01"]}).encode()  # fmt: skip
    rows = sec_filings.parse_older(extra, 1234567, "Tiny", date(2010, 1, 1))
    assert rows[0]["items"] == "2.02,9.01" and rows[0]["cik"] == 1234567
