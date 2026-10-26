from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from mw.research import build, events, external


def _frame() -> pl.DataFrame:
    days = [date(2019, 1, 1) + timedelta(days=i) for i in range(6)]
    rows = []
    for ent, flags in (("US:A", [0, 1, 1, 0, 1, 1]), ("US:B", [1, 1, 0, 0, 0, 1])):
        prev = (
            1 if ent == "US:B" else 0
        )  # B was already true on the (dropped) row before its first kept row
        for d, f in zip(days, flags, strict=True):
            rows.append(
                {
                    "entity_id": ent,
                    "as_of": d,
                    "c_x": bool(f),
                    "p_x": bool(prev),
                    "c_y": True,
                    "p_y": True,
                    "close": 30.0,
                    "raw_close": 30.0,
                    "in_sp500": ent == "US:A",
                    "vol_tier": 2,
                }
            )
            prev = f
    df = pl.DataFrame(rows)
    return df.with_columns(
        [
            pl.lit(0.01, pl.Float32).alias(f"{k}_{h}")
            for h in events.HORIZONS
            for k in ("ret", "mfe", "mae", "xret", "xsp")
        ]
    )


def test_onset_uses_previous_session_state() -> None:
    st = events.Study(_frame(), ["c_x", "c_y"])
    ev = st.events(("c_x",))
    # A: days 1 and 4; B: day 0 is NOT an onset (true on the previous session), day 5 is
    assert np.flatnonzero(ev).tolist() == [1, 4, 11]
    # c_y is always true, so the combination starts when c_x starts
    assert np.flatnonzero(st.events(("c_x", "c_y"))).tolist() == [1, 4, 11]
    # NOT conditions only filter the state: no positive condition -> no events
    assert st.events(("!c_x",)).sum() == 0
    # c_y with c_x false: c_y never "starts" (it was true the session before)
    assert st.events(("c_y", "!c_x")).sum() == 0


def test_data_start_excludes_earlier_rows() -> None:
    st = events.Study(_frame(), ["c_x"], starts={"c_x": date(2019, 1, 4)})
    assert np.flatnonzero(st.events(("c_x",))).tolist() == [4, 11]


def test_sp500_universe_filters_rows() -> None:
    st = events.Study(_frame(), ["c_x"], universe="sp500")
    assert st.n == 6


def test_excess_return_per_universe() -> None:
    df = pl.DataFrame(
        {
            "as_of": [date(2020, 1, 2)] * 3,
            "in_sp500": [True, True, False],
            **{f"ret_{h}": [0.01, 0.03, 0.20] for h in events.HORIZONS},
        }
    )
    out = events.add_excess(df)
    assert abs(out["xsp_20"][0] - (-0.01)) < 1e-6  # vs S&P 500 mean 0.02
    assert abs(out["xret_20"][0] - (0.01 - 0.08)) < 1e-6  # vs all-stock mean 0.08


def test_forward_returns_start_next_open() -> None:
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * 4,
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.5, 14.0, 13.5],
            "low": [9.5, 10.5, 11.0, 12.5],
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )
    out = events.forward_raw(df)
    # signal day 0: entry = open of day 1 (11); ret_1 = close day 1 / 11 - 1 = 0
    assert abs(out["ret_1"][0]) < 1e-6
    # ret_3 = close day 3 (13) / 11 - 1; mfe_3 = max high days 1-3 (14) / 11 - 1; mae_3 = min low (10.5) / 11 - 1
    assert abs(out["ret_3"][0] - (13 / 11 - 1)) < 1e-6
    assert abs(out["mfe_3"][0] - (14 / 11 - 1)) < 1e-6
    assert abs(out["mae_3"][0] - (10.5 / 11 - 1)) < 1e-6
    assert out["ret_3"][1] is None  # window runs past the data


def test_known_date_close_boundary() -> None:
    df = pl.DataFrame(
        {
            "published_at": [
                datetime(2024, 3, 5, 20, 59, tzinfo=UTC),  # 15:59 New York
                datetime(2024, 3, 5, 21, 0, tzinfo=UTC),  # 16:00 New York
            ]
        }
    )
    out = df.select(external.known_date(pl.col("published_at")))["known"].to_list()
    assert out == [date(2024, 3, 5), date(2024, 3, 6)]


def test_fred_published_after_close_is_next_day() -> None:
    df = pl.DataFrame(
        {
            "published_at": [
                datetime(2024, 3, 5, 20, 59, tzinfo=UTC),
                datetime(2024, 3, 5, 23, 0, tzinfo=UTC),  # FRED rule: 23:00 UTC = 18:00 New York
            ]
        }
    )
    out = build.fred_known(df).select(
        pl.col("published_at").dt.convert_time_zone("America/New_York").dt.date()
    )
    assert out["published_at"].to_list() == [date(2024, 3, 5), date(2024, 3, 6)]


def test_split_factor_restores_price_of_the_day() -> None:
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * 4,
            "splits": [0.0, 0.0, 4.0, 0.0],
            "close": [25.0, 25.0, 25.0, 25.0],
        }
    )
    out = df.with_columns(build.split_factor())
    # adjusted 25 before a 4:1 split was really 100
    assert out["split_fac"].to_list() == [4.0, 4.0, 1.0, 1.0]


def test_grade_rules() -> None:
    def s(net: float, n: int = 50) -> dict:
        return {"n": n, "net": net, "xmean": net}

    base = {"all": s(0.01), "dev": s(0.01), "recent3": s(0.01), "recent1": s(0.01)}
    blocks = {b[0]: s(0.01) for b in events.BLOCKS}
    assert events.grade(base | blocks) == "有力"
    blocks["2010-13"] = s(-0.01)
    blocks["2014-17"] = s(-0.01)
    assert events.grade(base | blocks) == "最近有効"
    assert events.grade(base | blocks | {"recent1": s(-0.01)}) == "候補"
    assert (
        events.grade(
            {"all": s(-0.01), "dev": s(0.01), "recent3": s(-0.01), "recent1": s(-0.01)} | blocks
        )
        == "不採用"
    )
    # too few development events -> reference only
    assert events.grade(base | blocks | {"dev": s(0.01, n=10)}) == "参考"
    # data only in the last two blocks: both positive is enough
    short = {"2010-13": {"n": 0}, "2014-17": {"n": 0}, "2018-21": s(0.01), "2022-26": s(0.01)}
    assert events.grade(base | short) == "有力"


def test_cost_by_price() -> None:
    assert (
        events.cost_of(np.array([25.0, 10.0, 2.0])).tolist()
        == np.array([0.001, 0.003, 0.01], dtype=np.float32).tolist()
    )
