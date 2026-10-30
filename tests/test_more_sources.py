import io
import zipfile
from datetime import date

from mw.ingest.macro import calendar as K
from mw.ingest.us import holdings_13f as H
from mw.ingest.us import wiki_views as W


def test_fomc_current_page() -> None:
    html = (
        '<h4><a id="1">2026 FOMC Meetings</a></h4>'
        '<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>'
        '<div class="fomc-meeting__date col-xs-4">27-28</div>'
        '<div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>'
        '<div class="fomc-meeting__date col-xs-4">30-1</div>'
        '<h4><a id="2">2025 FOMC Meetings</a></h4>'
        '<div class="fomc-meeting__month col-xs-5"><strong>March</strong></div>'
        '<div class="fomc-meeting__date col-xs-4">18-19*</div>'
        '<div class="fomc-meeting__month col-xs-5"><strong>Oct/Nov</strong></div>'
        '<div class="fomc-meeting__date col-xs-4">31-1</div>'
    )
    assert K.parse_current(html) == [
        date(2025, 3, 19),
        date(2025, 11, 1),
        date(2026, 1, 28),
        date(2026, 5, 1),
    ]


def test_fomc_historical_page() -> None:
    html = (
        "<p>January 27-28 Meeting</p><p>March 15 (unscheduled)</p><p>July/August 31-1 Meeting</p>"
    )
    assert K.parse_historical(html, 2020) == [
        date(2020, 1, 28),
        date(2020, 3, 15),
        date(2020, 8, 1),
    ]


def test_wiki_mapping_and_views() -> None:
    js = {"results": {"bindings": [
        {"ticker": {"value": "BRK-B"}, "article": {"value": "https://en.wikipedia.org/wiki/Berkshire_Hathaway"}},
        {"ticker": {"value": "AAPL"}, "article": {"value": "https://en.wikipedia.org/wiki/Apple_Inc."}},
        {"ticker": {"value": "AAPL"}, "article": {"value": "https://en.wikipedia.org/wiki/Apple_Inc._litigation"}},
    ]}}  # fmt: skip
    m = W.parse_mapping(js)
    assert dict(m.iter_rows()) == {"US:AAPL": "Apple_Inc.", "US:BRK.B": "Berkshire_Hathaway"}
    v = W.parse_views({"items": [{"timestamp": "2024010100", "views": 5}]}, "US:AAPL")
    assert v.row(0) == ("US:AAPL", date(2024, 1, 1), 5)


def _zip13f() -> bytes:
    sub = "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
    sub += "a1\t10-FEB-2024\t13F-HR\t1\t31-DEC-2023\n"
    sub += "a2\t12-FEB-2024\t13F-HR\t2\t31-DEC-2023\n"
    sub += "a3\t20-MAR-2024\t13F-HR/A\t1\t31-DEC-2023\n"  # amendment: not counted
    sub += "a4\t01-APR-2024\t13F-HR\t3\t31-DEC-2023\n"  # after the 46-day deadline: not counted
    info = "ACCESSION_NUMBER\tCUSIP\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
    info += "a1\t037833100\t1000\t10\tSH\t\n"
    info += "a2\t037833100\t2000\t20\tSH\t\n"
    info += "a2\t037833100\t500\t5\tSH\tCall\n"  # option: excluded
    info += "a3\t037833100\t9999\t99\tSH\t\n"
    info += "a4\t037833100\t9999\t99\tSH\t\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("INFOTABLE.tsv", info)
    return buf.getvalue()


def test_13f_counts_each_manager_once_by_the_deadline() -> None:
    df = H.parse_window(_zip13f())
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["cusip"] == "037833100" and r["period"] == date(2023, 12, 31)
    assert r["managers"] == 2 and r["shares"] == 30.0 and r["value"] == 3000.0
