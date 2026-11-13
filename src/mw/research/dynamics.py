"""Dynamics layer of the state atlas: how numbers MOVE over time, not their value on one day (owner 2026-11-13).

t_* columns, all known at the signal day's close. Joined to the atlas on (entity_id, as_of).

  moving-average state over time (n = 20, 50, 200)
    t_d{n}         distance from the n-day average, as a fraction of it
    t_d{n}_ch5/20  change of that distance over 5 / 20 sessions; _acc = change of the 5-day change
    t_d{n}_conv5   |distance| now minus |distance| 5 sessions ago (< 0: converging toward the average)
    t_run{n}       sessions in the current state, signed (+ above the average, - below)
    t_prun{n}      length of the previous state, signed
    t_x{n}_60      how many times the close crossed the average in the last 60 sessions
    t_ext{n}       most extreme distance inside the current state; t_back{n} = how far it came back from it
  volume while moving
    t_v5_20, t_vtrend (10-day volume vs the 10 days before), t_updn_v10 (volume on up days / on down days),
    t_updn_v_ch (that ratio vs the 20 sessions before), t_dnleg_v (down-day volume last 10 vs 11..40),
    t_pv_corr20 (return vs log volume), t_quiet_drift (return of the low-volume days in the last 20)
  path quality
    t_eff20 / t_eff60  |net move| / sum of |daily moves| (1 = straight line, 0 = back and forth)
    t_bigday_ch        share of big days (|r| > 2 std) in the last 20 vs the 60 before; t_updn_size (mean up
                       day / mean down day size, 20)
  extremes
    t_dd250, t_age_hi250, t_age_lo250, t_reb_lo60 (rebound from the 60-day low)
  relation to the market (SPY)
    t_rs_d50 (relative line vs its 50-day mean), t_rs_ch5 / t_rs_ch20, t_rs_run (signed run above/below that
    mean), t_corr60, t_corr_ch (change of the 60-day correlation over 20 sessions), t_run200_vs_mkt
    (sessions in the current 200-day state minus the market's, both signed)
  divergences (price makes a lower low / higher high, the other number does not)
    recent extreme = within the last 10 sessions, earlier extreme = sessions 11..40 back
    t_ll / t_hh             price at the recent low vs the earlier low (< 0: lower low) / same for highs
    t_div_lo_{x} / _hi_{x}  number x at the recent extreme minus at the earlier one, x in rsi, rs, vol, clv
    t_bull_{x} / t_bear_{x} the same, only when the price made a lower low / higher high (else null)
  candle sequences
    t_clv_trend (close location: mean of last 5 minus mean of 6..15), t_lwick10 (days with a long lower wick
    in the last 10), t_rng_shrink (run of sessions whose range is smaller than the day before), t_on_id5
    (overnight sum minus intraday sum, 5 sessions: > 0 = opens high, sold during the day)
  rank in the market (needs all stocks, see rank_dynamics)
    t_rk60 (percentile of the 60-day return among all stocks that day), t_rk60_ch20 (its change over 20)
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from mw.research.atlas import EPS, G, repair_splits

C = pl.col
RECENT, EARLIER = 10, 30  # divergence windows: last 10 sessions vs the 30 before them


def _with_run(df: pl.DataFrame, sign: str, name: str) -> pl.DataFrame:
    """Signed run length of the +1/-1 column `sign` per entity: adds _{name}_id (run number) and name."""
    df = df.with_columns(
        (C(sign) != C(sign).shift(1)).fill_null(True).cum_sum().over(G).alias(f"_{name}_id")
    )
    return df.with_columns(
        (C(sign) * pl.int_range(1, pl.len() + 1).over(G, f"_{name}_id")).alias(name)
    )


def _rsi(n: int = 14) -> pl.Expr:
    d = C("close").diff()
    up = d.clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
    dn = (-d).clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
    return 100 - 100 / (1 + up / dn.clip(lower_bound=EPS))


def stock_dynamics(df: pl.DataFrame, spy: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, as_of, open, high, low, close, volume, splits. spy: as_of, spy_close, spy_run200.
    Returns entity_id, as_of and the t_* columns (see module doc)."""
    df = repair_splits(df.sort(G, "as_of"))
    df = df.join(spy, on="as_of", how="left")
    df = df.sort(G, "as_of")
    r1 = C("close") / C("close").shift(1) - 1
    rng = (C("high") - C("low")).clip(lower_bound=EPS)
    df = df.with_columns(
        r1.over(G).alias("_r1"),
        ((C("close") - C("low")) / rng).alias("_clv"),
        ((pl.min_horizontal("open", "close") - C("low")) / rng).alias("_lw"),
        (C("high") - C("low")).alias("_rng"),
        (C("open") / C("close").shift(1) - 1).over(G).alias("_on"),
        (C("close") / C("open") - 1).alias("_id"),
        (C("close") / C("spy_close")).log().alias("_rs"),
        (C("spy_close") / C("spy_close").shift(1) - 1).over(G).alias("_mr1"),
        _rsi().over(G).alias("_rsi"),
        C("volume").rolling_mean(50).over(G).alias("_v50"),
    )
    ex: list[pl.Expr] = []
    # ---- moving-average state over time
    for n in (20, 50, 200):
        d = C("close") / C("close").rolling_mean(n) - 1
        ex.append(d.over(G).alias(f"t_d{n}"))
    df = df.with_columns(ex)
    ex = []
    for n in (20, 50, 200):
        d = C(f"t_d{n}")
        ex += [
            (d - d.shift(5)).over(G).alias(f"t_d{n}_ch5"),
            (d - d.shift(20)).over(G).alias(f"t_d{n}_ch20"),
            ((d - d.shift(5)) - (d.shift(5) - d.shift(10))).over(G).alias(f"t_d{n}_acc"),
            (d.abs() - d.abs().shift(5)).over(G).alias(f"t_d{n}_conv5"),
            pl.when(d >= 0).then(1).otherwise(-1).cast(pl.Int32).alias(f"_sg{n}"),
        ]
    df = df.with_columns(ex)
    for n in (20, 50, 200):
        df = _with_run(df, f"_sg{n}", f"t_run{n}")
    ex = []
    for n in (20, 50, 200):
        rid, run, d = C(f"_t_run{n}_id"), C(f"t_run{n}"), C(f"t_d{n}")
        start = (rid != rid.shift(1)).fill_null(True)
        ex += [
            pl.when(start).then(run.shift(1)).otherwise(None).forward_fill().over(G).alias(f"t_prun{n}"),
            start.cast(pl.Int32).rolling_sum(60).over(G).alias(f"t_x{n}_60"),
            pl.when(run < 0).then(d.cum_min().over(G, rid)).otherwise(d.cum_max().over(G, rid)).alias(f"t_ext{n}"),
        ]  # fmt: skip
    df = df.with_columns(ex)
    df = df.with_columns(
        [(C(f"t_d{n}") - C(f"t_ext{n}")).alias(f"t_back{n}") for n in (20, 50, 200)]
    )

    # ---- volume while moving, path quality, extremes, market relation, candles
    up, dn = C("_r1") > 0, C("_r1") < 0
    v = C("volume")
    big = (C("_r1").abs() > 2 * C("_r1").rolling_std(60)).cast(pl.Float32)
    lowv = v < v.rolling_median(20)
    rs = C("_rs")
    rs_m50 = rs.rolling_mean(50)
    ex = [
        (v.rolling_mean(5) / v.rolling_mean(20).clip(lower_bound=EPS)).alias("t_v5_20"),
        (v.rolling_mean(10) / v.rolling_mean(10).shift(10).clip(lower_bound=EPS)).log().alias("t_vtrend"),
        (pl.when(up).then(v).otherwise(None).rolling_mean(10, min_samples=2)
         / pl.when(dn).then(v).otherwise(None).rolling_mean(10, min_samples=2).clip(lower_bound=EPS)).log().alias("_udv10"),
        (pl.when(up).then(v).otherwise(None).rolling_mean(20, min_samples=3).shift(10)
         / pl.when(dn).then(v).otherwise(None).rolling_mean(20, min_samples=3).shift(10).clip(lower_bound=EPS)).log().alias("_udv_prev"),
        (pl.when(dn).then(v).otherwise(None).rolling_mean(10, min_samples=2)
         / pl.when(dn).then(v).otherwise(None).rolling_mean(30, min_samples=3).shift(10).clip(lower_bound=EPS)).log().alias("t_dnleg_v"),
        pl.rolling_corr(C("_r1"), v.clip(lower_bound=1).log(), window_size=20).alias("t_pv_corr20"),
        pl.when(lowv).then(C("_r1")).otherwise(0.0).rolling_sum(20).alias("t_quiet_drift"),
        ((C("close") - C("close").shift(20)).abs()
         / C("close").diff().abs().rolling_sum(20).clip(lower_bound=EPS)).alias("t_eff20"),
        ((C("close") - C("close").shift(60)).abs()
         / C("close").diff().abs().rolling_sum(60).clip(lower_bound=EPS)).alias("t_eff60"),
        (big.rolling_mean(20) - big.shift(20).rolling_mean(60)).alias("t_bigday_ch"),
        (pl.when(up).then(C("_r1")).otherwise(None).rolling_mean(20, min_samples=3)
         / (-pl.when(dn).then(C("_r1")).otherwise(None).rolling_mean(20, min_samples=3)).clip(lower_bound=EPS)).log().alias("t_updn_size"),
        (C("close") / C("high").rolling_max(250) - 1).alias("t_dd250"),
        (C("close") / C("low").rolling_min(60) - 1).alias("t_reb_lo60"),
        ((rs - rs_m50) / C("_r1").rolling_std(50).clip(lower_bound=EPS)).alias("t_rs_d50"),
        (rs - rs.shift(5)).alias("t_rs_ch5"),
        (rs - rs.shift(20)).alias("t_rs_ch20"),
        pl.when(rs >= rs_m50).then(1).otherwise(-1).cast(pl.Int32).alias("_rs_sg"),
        pl.rolling_corr(C("_r1"), C("_mr1"), window_size=60).alias("t_corr60"),
        (C("_clv").rolling_mean(5) - C("_clv").shift(5).rolling_mean(10)).alias("t_clv_trend"),
        (C("_lw") > 0.4).cast(pl.Int32).rolling_sum(10).alias("t_lwick10"),
        pl.when(C("_rng") < C("_rng").shift(1)).then(1).otherwise(0).cast(pl.Int32).alias("_rsh"),
        (C("_on").rolling_sum(5) - C("_id").rolling_sum(5)).alias("t_on_id5"),
        (C("_clv").rolling_mean(5)).alias("_clv5"),
        (v.rolling_mean(5) / C("_v50").clip(lower_bound=EPS)).alias("_vn5"),
    ]  # fmt: skip
    df = df.with_columns([e.over(G) for e in ex])
    df = df.with_columns(
        (C("_udv10") - C("_udv_prev")).alias("t_updn_v_ch"),
        C("_udv10").alias("t_updn_v10"),
        (C("t_corr60") - C("t_corr60").shift(20)).over(G).alias("t_corr_ch"),
        (C("t_run200") - C("spy_run200")).alias("t_run200_vs_mkt"),
    )
    df = _with_run(df, "_rs_sg", "t_rs_run")
    df = _with_run(df, "_rsh", "t_rng_shrink")
    df = _window_dynamics(df)
    keep = [G, "as_of"] + [c for c in df.columns if c.startswith("t_")]
    out = df.select(keep)
    fl = [c for c, t in out.schema.items() if t in (pl.Float64, pl.Float32)]
    return out.with_columns([C(c).cast(pl.Float32).fill_nan(None) for c in fl])


def _window_dynamics(df: pl.DataFrame) -> pl.DataFrame:
    """Divergences and 250-day extreme ages (argmin / argmax over windows), per entity in numpy."""
    W = RECENT + EARLIER
    keys = ["t_age_hi250", "t_age_lo250", "t_ll", "t_hh"] + [
        f"t_div_{s}_{x}" for s in ("lo", "hi") for x in ("rsi", "rs", "vol", "clv")
    ]
    cols: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for _, g in df.group_by(G, maintain_order=True):
        c = g["close"].to_numpy().astype(float)
        h, lo = g["high"].to_numpy().astype(float), g["low"].to_numpy().astype(float)
        xs = {
            "rsi": g["_rsi"].to_numpy().astype(float),
            "rs": g["_rs"].to_numpy().astype(float),
            "vol": np.log(np.clip(g["_vn5"].to_numpy().astype(float), EPS, None)),
            "clv": g["_clv5"].to_numpy().astype(float),
        }
        n = len(c)
        for N, arr, key, fill in (
            (250, h, "t_age_hi250", -np.inf),
            (250, lo, "t_age_lo250", np.inf),
        ):
            res = np.full(n, np.nan)
            if n >= N:
                w = sliding_window_view(np.nan_to_num(arr, nan=fill), N)[:, ::-1]
                res[N - 1 :] = w.argmax(1) if fill < 0 else w.argmin(1)
            cols[key].append(res)
        for side, key, fill in (("lo", "t_ll", np.inf), ("hi", "t_hh", -np.inf)):
            res = np.full(n, np.nan)
            div = {x: np.full(n, np.nan) for x in xs}
            if n >= W:
                w = sliding_window_view(np.nan_to_num(c, nan=fill), W)
                pick = np.argmin if side == "lo" else np.argmax
                i1 = pick(w[:, EARLIER:], axis=1) + EARLIER
                i2 = pick(w[:, :EARLIER], axis=1)
                base = np.arange(n - W + 1)
                a1, a2 = base + i1, base + i2
                res[W - 1 :] = c[a1] / c[a2] - 1
                for x, arr in xs.items():
                    div[x][W - 1 :] = arr[a1] - arr[a2]
            cols[key].append(res)
            for x in xs:
                cols[f"t_div_{side}_{x}"].append(div[x])
    df = df.with_columns(
        [pl.Series(k, np.concatenate(v), dtype=pl.Float32).fill_nan(None) for k, v in cols.items()]
    )
    out = []
    for x in ("rsi", "rs", "vol", "clv"):
        out += [
            pl.when(C("t_ll") < 0).then(C(f"t_div_lo_{x}")).otherwise(None).alias(f"t_bull_{x}"),
            pl.when(C("t_hh") > 0).then(C(f"t_div_hi_{x}")).otherwise(None).alias(f"t_bear_{x}"),
        ]
    return df.with_columns(out)


def market_runs(spy: pl.DataFrame) -> pl.DataFrame:
    """spy: as_of, close (SPY). Returns as_of, spy_close, spy_run200 (signed run above/below its 200-day mean)."""
    s = spy.sort("as_of").with_columns(
        pl.when(C("close") >= C("close").rolling_mean(200)).then(1).otherwise(-1).alias("_sg")
    )
    run = (C("_sg") != C("_sg").shift(1)).fill_null(True).cum_sum()
    s = s.with_columns(run.alias("_r"))
    s = s.with_columns((C("_sg") * pl.int_range(1, pl.len() + 1).over("_r")).alias("spy_run200"))
    return s.select("as_of", C("close").alias("spy_close"), "spy_run200")


def rank_dynamics(px: pl.LazyFrame) -> pl.DataFrame:
    """Percentile of the 60-day return among all stocks each day and its change over 20 sessions.
    px: entity_id, as_of, close (split-adjusted is fine: ratios only)."""
    r = px.sort(G, "as_of").with_columns(
        (C("close") / C("close").shift(60) - 1).over(G).alias("_r60")
    )
    r = r.filter(C("_r60").is_finite()).with_columns(
        (C("_r60").rank() / pl.len()).over("as_of").cast(pl.Float32).alias("t_rk60")
    )
    r = r.sort(G, "as_of").with_columns(
        (C("t_rk60") - C("t_rk60").shift(20)).over(G).alias("t_rk60_ch20")
    )
    return r.select(G, "as_of", "t_rk60", "t_rk60_ch20").collect()
