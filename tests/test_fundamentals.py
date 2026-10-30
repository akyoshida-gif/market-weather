import json
from datetime import date

from mw.ingest.us import fundamentals as F


def _facts() -> bytes:
    usd = [
        # Q1 2024 revenue, first filed, then restated in a later 10-Q
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 100.0,
            "form": "10-Q",
            "fy": 2024,
            "fp": "Q1",
            "filed": "2024-05-01",
            "accn": "a",
        },
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 101.0,
            "form": "10-Q",
            "fy": 2025,
            "fp": "Q1",
            "filed": "2025-05-01",
            "accn": "b",
        },
        # full year in the 10-K
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "val": 420.0,
            "form": "10-K",
            "fy": 2024,
            "fp": "FY",
            "filed": "2025-02-15",
            "accn": "c",
        },
        # 6-month year-to-date: neither quarter nor year -> dropped
        {
            "start": "2024-01-01",
            "end": "2024-06-30",
            "val": 210.0,
            "form": "10-Q",
            "fy": 2024,
            "fp": "Q2",
            "filed": "2024-08-01",
            "accn": "d",
        },
        # an 8-K value is not a financial statement -> dropped
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "val": 99.0,
            "form": "8-K",
            "fy": 2024,
            "fp": "Q1",
            "filed": "2024-04-20",
            "accn": "e",
        },
    ]
    d = {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": usd}},
                "SalesRevenueNet": {
                    "units": {"USD": [dict(usd[0], val=55.0)]}
                },  # lower-ranked tag, same period
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-03-31",
                                "val": 900.0,
                                "form": "10-Q",
                                "fy": 2024,
                                "fp": "Q1",
                                "filed": "2024-05-01",
                            }
                        ]
                    }
                },
                "SomethingElse": {"units": {"USD": [dict(usd[0])]}},
            }
        },
    }
    return json.dumps(d).encode()


def test_parse_keeps_statement_forms_and_known_tags() -> None:
    df = F.parse_companyfacts(_facts())
    assert set(df["concept"].unique()) == {"revenue", "assets"}
    assert "8-K" not in df["form"].to_list()


def test_quarterly_keeps_each_filing_and_prefers_the_main_tag() -> None:
    q = F.quarterly(F.parse_companyfacts(_facts()))
    rev_q = q.filter((q["concept"] == "revenue") & (q["period"] == "quarter")).sort("filed")
    # the original and the restated value are both kept (as known at each filing date)
    assert rev_q["val"].to_list() == [100.0, 101.0]
    assert rev_q["filed"].to_list() == [date(2024, 5, 1), date(2025, 5, 1)]
    assert q.filter((q["concept"] == "revenue") & (q["period"] == "year"))["val"].to_list() == [
        420.0
    ]
    assert q.filter(q["concept"] == "assets")["period"].to_list() == ["instant"]
    assert 210.0 not in q["val"].to_list()  # year-to-date values are dropped
