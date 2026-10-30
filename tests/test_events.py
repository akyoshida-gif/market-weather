from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from mw.research import build, events, external, features
from mw.strategy import regime


def _parts(tmp_path: Path, f_values: list[float] | None = None) -> tuple[list[Path], pl.DataFrame]:
    days = [date(2019, 1, 1) + timedelta(days=i) for i in range(6)]
    rows = []
    for ent, flags in (("US:A", [0, 1, 1, 0, 1, 1]), ("US:B", [1, 1, 0, 0, 0, 1])):
        prev = (
            1 if ent == "US:B" else 0
        )  # B was already true on the (dropped) row before its first kept row
        for i, (d, f) in enumerate(zip(days, flags, strict=True)):
            rows.append(
                {
                    "entity_id": ent,
                    "as_of": d,
                    "c_x": bool(f),
                    "p_x": bool(prev),
                    "c_y": True,
                    "p_y": True,
                    "raw_close": 30.0,
                    "in_sp500": ent == "US:A",
                    "vol_tier": 2,
                    "contig": i > 0,
                    "f_z": (f_values or [float(i)] * 6)[i] if ent == "US:A" else float(i),
                }
            )
            prev = f
    df = pl.DataFrame(rows).with_columns(
        [
            pl.lit(0.01, pl.Float32).alias(f"{k}_{h}")
            for h in events.HORIZONS
            for k in ("ret", "mfe", "mae", "xret", "xsp")
        ]
        + [
            pl.lit(0.5, pl.Float32).alias(f"r_{s}_{h}_{k}")
            for s in features.STOPS
            for h in features.R_HORIZONS
            for k in ("fin", "mfe")
        ]
        + [pl.lit(0.05, pl.Float32).alias(f"rp_{s}") for s in features.STOPS]
    )
    p = tmp_path / "part_0.parquet"
    df.write_parquet(p)
    day = pl.DataFrame({"as_of": days, "m_q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    return [p], day


def _study(tmp_path: Path, **kw: object) -> events.Study:
    events.MIN_CUT_ROWS, events.MIN_CUT_DAYS, events.MAX_BIN_SHARE = 1, 1, 1.0
    parts, day = _parts(tmp_path)
    return events.Study(parts, day, log=lambda *_: None, **kw)  # type: ignore[arg-type]


def test_onset_uses_previous_session_state(tmp_path: Path) -> None:
    st = _study(tmp_path)
    ev = st.events(("c_x",))
    # A: days 1 and 4; B: day 0 is NOT an onset (true on the previous session), day 5 is
    assert np.flatnonzero(ev).tolist() == [1, 4, 11]
    assert np.flatnonzero(st.events(("c_x", "c_y"))).tolist() == [1, 4, 11]
    # NOT conditions only filter the state: no positive condition -> no events
    assert st.events(("!c_x",)).sum() == 0
    assert st.events(("c_y", "!c_x")).sum() == 0


def test_data_start_excludes_earlier_rows(tmp_path: Path) -> None:
    st = _study(tmp_path, starts={"c_x": date(2019, 1, 4)})
    assert np.flatnonzero(st.events(("c_x",))).tolist() == [4, 11]


def test_sp500_universe_filters_rows(tmp_path: Path) -> None:
    st = _study(tmp_path, universe="sp500")
    assert st.n == 6


def test_market_wide_condition_starts_on_its_day(tmp_path: Path) -> None:
    st = _study(tmp_path)
    # m_q >= its 80% cut (from the data, ~5.0) holds on the last day(s): every stock "starts" there
    ev = st.events(("m_q__hi20",))
    days = {int(st.date_rank[i]) for i in np.flatnonzero(ev)}
    assert days == {min(int(r) for r in st.date_rank[st._cond("m_q__hi20")])}


def test_numeric_bin_needs_a_real_previous_row(tmp_path: Path) -> None:
    st = _study(tmp_path)
    # first row of each stock has contig = False -> no onset there even if the bin holds
    ev = st.events(("f_z__lo20",))
    assert not ev[0] and not ev[6]


def test_excess_return_per_universe(tmp_path: Path) -> None:
    rows = {
        "entity_id": ["US:A", "US:B", "US:C"],
        "as_of": [date(2020, 1, 2)] * 3,
        "in_sp500": [True, True, False],
        "atr": [1.0, 1.0, 1.0],
        "close": [10.0, 10.0, 10.0],
        **{f"ret_{h}": [0.01, 0.03, 0.20] for h in events.HORIZONS},
        **{f: [0.1, 0.2, 0.3] for f in build.RANKED},
    }
    p = tmp_path / "part_0.parquet"
    pl.DataFrame(rows).write_parquet(p)
    build.finalize([p])
    out = pl.read_parquet(p)
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
    assert abs(out["ret_1"][0]) < 1e-6
    assert abs(out["ret_3"][0] - (13 / 11 - 1)) < 1e-6
    assert abs(out["mfe_3"][0] - (14 / 11 - 1)) < 1e-6
    assert abs(out["mae_3"][0] - (10.5 / 11 - 1)) < 1e-6
    assert out["ret_3"][1] is None


def test_r_outcomes_stop_and_gap() -> None:
    n = 130
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    c = np.full(n, 100.0)
    lo[3] = 90.0  # day 3 touches any stop between 90 and 99
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * n,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "atr": np.full(n, 2.0),
            "ma20": np.full(n, 95.0),
            "ma50": np.full(n, 90.0),
        }
    )
    out = features.r_outcomes(df)
    # signal on row 0, entry = open of row 1 = 100; atr2 stop = 96 (R = 4) -> hit on row 3 -> -1R
    assert abs(out["r_atr2_5_fin"][0] - (-1.0)) < 1e-6
    # ma20 stop = 95 - 0.4 = 94.6 -> also hit on row 3
    assert abs(out["r_ma20_5_fin"][0] - (-1.0)) < 1e-6
    # ma50 stop = 89.6 -> not hit; R = 10.4; close 100 -> 0R; best high 101 -> ~0.096R
    assert abs(out["r_ma50_5_fin"][0]) < 1e-6
    assert abs(out["r_ma50_5_mfe"][0] - 1 / 10.4) < 1e-4
    # sig_low stop = 99 - 0.2 = 98.8 -> R = 1.2 < 0.5*ATR? no (1.2 >= max(1.0, 1.0)) -> valid, hit on row 3
    assert abs(out["r_sig_low_5_fin"][0] - (-1.0)) < 1e-6


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
    out = df.select(regime.ny_known_date(pl.col("published_at")).alias("d"))
    assert out["d"].to_list() == [date(2024, 3, 5), date(2024, 3, 6)]


def test_split_factor_restores_price_of_the_day() -> None:
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * 4,
            "splits": [0.0, 0.0, 4.0, 0.0],
            "close": [25.0, 25.0, 25.0, 25.0],
        }
    )
    out = df.with_columns(build.split_factor())
    assert out["split_fac"].to_list() == [4.0, 4.0, 1.0, 1.0]


def test_grade_rules() -> None:
    def s(v: float, n: int = 50, t: float = 3.0) -> dict:
        return {"n": n, "net": v, "xmean": v, "xt": t}

    blocks_up = {b[0]: s(0.01) for b in events.BLOCKS}
    base = {"search": s(0.01), "test": s(0.01), "ex_top5": s(0.01), "top5_share": 0.2}
    assert events.grade(base | blocks_up) == "主力候補（長寿命）"
    blocks_new = blocks_up | {"2010-13": s(-0.01), "2014-17": s(-0.01), "2018-21": s(-0.01)}
    assert events.grade(base | blocks_new) == "主力候補（新しい傾向）"
    # confirmation period negative, too small or not reliable -> observe
    assert events.grade(base | blocks_up | {"test": s(-0.01)}) == "観察"
    assert events.grade(base | blocks_up | {"test": s(0.01, n=15)}) == "観察"
    assert events.grade(base | blocks_up | {"test": s(0.01, t=1.0)}) == "観察"
    # concentrated on a few days -> not a main candidate
    assert events.grade(base | blocks_up | {"top5_share": 0.8}) == "観察"
    # weak in the search window
    assert events.grade(base | blocks_up | {"search": s(-0.01)}) == "不採用"
    assert events.grade(base | blocks_up | {"search": s(0.01, n=10)}) == "参考"
    # market-wide conditions: excess return is ~0 by construction and is not required
    mk = (
        base
        | blocks_up
        | {"search": {"n": 50, "net": 0.01, "xmean": 0.0, "xt": 3.0}, "market_wide": True}
    )
    assert events.grade(mk) == "主力候補（長寿命）"
    # 2025-only search: no confirmation period -> at most "observe"
    assert events.grade(base, has_test=False) == "観察"


def test_cost_by_price() -> None:
    assert (
        events.cost_of(np.array([25.0, 10.0, 2.0])).tolist()
        == np.array([0.001, 0.003, 0.01], dtype=np.float32).tolist()
    )


def test_lookback_feature_is_dropped_after_a_data_error() -> None:
    df = pl.DataFrame(
        {
            "entity_id": ["US:A"] * 5,
            "_bad": [0, 1, 0, 0, 0],
            "f_ret252": [0.1, 9.0, 9.0, 9.0, 9.0],
            "f_ret5": [0.1, 0.2, 0.3, 0.4, 0.5],
            "c_new_high_252": [True] * 5,
        },
        schema_overrides={"_bad": pl.Int8},
    )
    out = build.lookback_clean(df)
    assert out["f_ret252"].to_list() == [0.1, None, None, None, None]
    assert out["c_new_high_252"].to_list() == [True, False, False, False, False]
    assert out["f_ret5"].to_list() == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]  # short-window features are untouched


def test_market_wide_condition_must_beat_holding_everything() -> None:
    def r(e: float, n: int = 50, t: float = 4.0) -> dict:
        return {"n": n, "exp": e, "t": t}

    s = {
        "search": {"n": 50, "net": 0.01, "xmean": 0.0},
        "market_wide": True,
        "top5_share": 0.2,
        "R": {"stop": "atr2", "search": r(0.5), "test": r(0.5), "ex_top5": r(0.5)},
    }
    assert events.grade(s).startswith("主力候補")
    # every stock did +0.6R in those periods anyway -> the condition adds nothing
    assert events.grade(s | {"R_base": {"search": 0.6, "test": 0.6}}) == "不採用"
