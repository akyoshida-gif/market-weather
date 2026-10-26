"""SEC reference datasets and S&P 500 membership (synthetic inputs)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, date, datetime

import polars as pl

from mw.ingest.us import sec_reference as R
from mw.universe import sp500


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return buf.getvalue()


def test_ftd_parse_and_publish_rule():
    txt = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260915|B5950S113|MDXH|15436|MDXHEALTH SA|0.71\nTrailer record count 1\n"
    df = R.parse_ftd(_zip({"cnsfails202609b": txt}))
    assert df.height == 1 and df["symbol"][0] == "MDXH" and df["fails"][0] == 15436.0
    assert R.ftd_published_at("202609a") == datetime(2026, 10, 3, 23, 59, tzinfo=UTC)
    assert R.ftd_published_at("202609b") == datetime(2026, 10, 18, 23, 59, tzinfo=UTC)
    links = R.ftd_links('<a href="/files/data/fails-deliver-data/cnsfails202609b.zip">x</a>')
    assert links == [
        ("https://www.sec.gov/files/data/fails-deliver-data/cnsfails202609b.zip", "202609b")
    ]


def test_insider_parse_join():
    sub = "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\tAFF10B5ONE\nA1\t30-SEP-2026\t4\t123\tabc\t0\n"
    own = "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\nA1\t9\tDoe John\tOfficer\tCEO\n"
    tr = ("ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\tSHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
          "A1\t28-SEP-2026\tP\t1000\t12.5\tA\t5000\tD\n")  # fmt: skip
    df = R.parse_insider(
        _zip({"SUBMISSION.tsv": sub, "REPORTINGOWNER.tsv": own, "NONDERIV_TRANS.tsv": tr})
    )
    r = df.row(0, named=True)
    assert (
        r["symbol"] == "ABC" and r["code"] == "P" and r["shares"] == 1000.0 and r["title"] == "CEO"
    )
    assert r["filing_date"] == date(2026, 9, 30)
    pub = df.select(R.insider_published_at(pl.col("filing_date")))[0, 0]
    assert pub == datetime(2026, 10, 1, 2, 0, tzinfo=UTC)  # 22:00 EDT


def test_frames_and_quarters():
    data = json.dumps(
        {
            "data": [
                {
                    "cik": 1750,
                    "entityName": "AAR",
                    "end": "2026-06-30",
                    "val": 39892472,
                    "accn": "x",
                }
            ]
        }
    ).encode()
    df = R.parse_frames(data)
    assert df["shares_out"][0] == 39892472.0 and df["period_end"][0] == date(2026, 6, 30)
    assert R.quarters(date(2025, 11, 1), date(2026, 4, 1)) == [(2025, 4), (2026, 1), (2026, 2)]


def test_sp500_intervals_walk_back():
    cur = pl.DataFrame({"symbol": ["AAA", "BBB"]})
    ch = pl.DataFrame({"date": [date(2024, 1, 2)], "added": ["BBB"], "removed": ["OLD"]})
    iv = sp500.intervals(cur, ch, date(2020, 1, 1))
    assert sp500.members_on(iv, date(2023, 6, 1)) == ["AAA", "OLD"]
    assert sp500.members_on(iv, date(2024, 6, 1)) == ["AAA", "BBB"]


def test_insider_old_layout_without_10b5_column():
    sub = "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\nA1\t15-MAR-2008\t4\t123\tabc\n"
    own = "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\nA1\t9\tDoe\tDirector\t\n"
    tr = "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\tSHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\nA1\t14-MAR-2008\tP\t500\t3\tA\t900\tD\n"
    df = R.parse_insider(
        _zip({"SUBMISSION.tsv": sub, "REPORTINGOWNER.tsv": own, "NONDERIV_TRANS.tsv": tr})
    )
    assert df.height == 1 and df["plan_10b5_1"][0] is None and df["code"][0] == "P"
