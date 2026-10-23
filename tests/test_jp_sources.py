"""Contract tests for JP adapters with synthetic files (real data is never committed)."""

from __future__ import annotations

import io
from datetime import UTC, date, datetime

import xlwt

from mw.core.config import GuardSettings
from mw.core.pit import validate_pit_frame
from mw.core.storage import GuardedStorage, LocalFS
from mw.ingest.jp import jpx_short_positions as sp
from mw.ingest.jp import jsf

ZANDAKA_HEADER = (
    '"申込日","決済日","銘柄コード","銘柄名","取引所区分名","上場区分","速報／確報",'
    + ",".join(f'"c{i}"' for i in range(8, 37))
)


def zandaka_bytes(status: str = "確報") -> bytes:
    nums = ",".join(["100"] * 29)
    rows = [
        f'"2026/10/22","2026/10/26","1301","極洋","東証およびＰＴＳ",,"{status}",{nums}',
        f'"2026/10/22","2026/10/26","130A","ＶＩＳ","東証およびＰＴＳ",,"{status}",{nums}',
    ]
    # make lend balance (col 13) zero in second row
    r2 = rows[1].split(",")
    r2[12] = "0"
    rows[1] = ",".join(r2)
    return ("\n".join([ZANDAKA_HEADER, *rows]) + "\n").encode("cp932")


def shina_bytes() -> bytes:
    lines = [
        "品貸料率一覧,,,,,,,,,,,,,,,",
        "（注）説明,,,,,,,,,,,,,,,",
        "抽出条件,全体,,,,,,,,,,,,,,",
        "貸借申込日,決済日,コード,銘柄名,取引所区分,決算事由,決算等,貸借値段（円）,貸株超過株数,"
        "最高料率（円）,当日品貸料率（円）,当日品貸日数,前日品貸料率（円）,備考,制限,応札倍率ランク",
        "20261022,20261026,1301,極洋,東証,決算,20270331,4505.00,0,9.20,*****,*****,*****,満額,,",
        "20261022,20261026,1306,ＮＦＴＯＰＩＸ,東証,決算,20270710,420.60,222710,6.00,0.05,1,*****,,,F",
    ]
    return ("\n".join(lines) + "\n").encode("cp932")


def test_parse_zandaka():
    df = jsf.parse_zandaka(zandaka_bytes())
    assert df.height == 2 and df["status"].to_list() == ["final", "final"]
    assert df["trade_date"].to_list() == [date(2026, 10, 22)] * 2
    assert df["loan_lend_ratio"].to_list() == [1.0, None]
    assert df["code"].to_list() == ["1301", "130A"]


def test_parse_shina():
    df = jsf.parse_shina(shina_bytes())
    assert df["premium_rate_yen"].to_list() == [None, 0.05]
    assert df["lend_excess_shares"].to_list() == [0.0, 222710.0]


def test_jsf_persist_and_skip(tmp_path):
    storage = GuardedStorage(LocalFS(tmp_path), GuardSettings())
    caps = [
        jsf.Captured("zandaka", b, jsf.parse_zandaka(b), date(2026, 10, 22))
        for b in [zandaka_bytes()]
    ]
    now = datetime(2026, 10, 23, 3, 0, tzinfo=UTC)
    r = jsf.persist(storage, caps, now)
    assert r["zandaka"]["status"] == "final"
    silver = storage.read_parquet(jsf.silver_key_for("zandaka", date(2026, 10, 22), "final"))
    validate_pit_frame(silver)
    assert silver["entity_id"].to_list() == ["JP:1301", "JP:130A"]
    assert jsf.persist(storage, caps, now)["zandaka"]["skipped"] == "exists"


def xls_bytes() -> bytes:
    wb = xlwt.Workbook(encoding="cp932")
    ws = wb.add_sheet("20261022")
    ws.write(7, 1, "Date of Calculation")
    datefmt = xlwt.easyxf(num_format_str="yyyy/mm/dd")
    ws.write(8, 1, 46315.0, datefmt)
    ws.write(8, 2, 1356.0)
    ws.write(8, 3, "ＴＯＰＩＸベア")
    ws.write(8, 4, "TOPIX Bear")
    ws.write(8, 5, "個人")
    ws.write(8, 10, 0.0049)
    ws.write(8, 11, 121840.0)
    ws.write(8, 12, 12184.0)
    ws.write(8, 13, 46293.0)
    ws.write(8, 14, 0.0051)
    ws.write(9, 1, 46315.0)
    ws.write(9, 2, "646A")
    ws.write(9, 5, "Barclays Capital Securities Ltd")
    ws.write(9, 10, 0.0282)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_parse_xls():
    df = sp.parse_xls(xls_bytes())
    assert df.height == 2
    assert df["calc_date"].to_list() == [date(2026, 10, 20)] * 2
    assert df["code"].to_list() == ["1356", "646A"]
    assert df["short_seller"].to_list()[0] == "個人"
    assert df["ratio"].to_list() == [0.0049, 0.0282]
    assert df["prev_calc_date"].to_list()[0] == date(2026, 9, 28)


def test_parse_links():
    html = (
        '<a href="/markets/public/short-selling/t13vrt000002hegv-att/20261022_Short_Positions.xls">'
        '<a href="/markets/public/short-selling/abc-att/20261021_Short_Positions.xls">'
    )
    links = sp.parse_links(html)
    assert [x.disclosure_date for x in links] == [date(2026, 10, 21), date(2026, 10, 22)]
    assert links[1].url.startswith("https://www.jpx.co.jp/markets/public/short-selling/t13")


def test_published_at_rule():
    d = date(2026, 10, 22)
    rule = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
    assert sp.published_at_for(d, None) == rule
    assert sp.published_at_for(d, datetime(2026, 10, 22, 8, 0, tzinfo=UTC)) == rule
    late = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)
    assert sp.published_at_for(d, late) == late
    assert sp.published_at_for(d, datetime(2027, 1, 1, tzinfo=UTC)) == rule  # regenerated file
