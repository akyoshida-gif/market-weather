"""State atlas: fine-grained numbers for every stock-day and the full path afterwards (owner 2026-11-09 / 11-12).

Not built from named chart patterns. Each stock-day is described by many raw numbers (candle anatomy, sequences,
memory of special days, volume behaviour, cost basis of recent buyers, volatility, relation to the market) plus,
for the most important ones, how they stand against the stock's own past (z-score, change, acceleration).
Search code (stepwise / pair maps / nearest neighbours / ML) works on top of this table.

Prefixes:
  s_*  stock features, known at the signal day's close (no look-ahead)
  m_*  market features of that day (same for every stock), joined later
  y_*  outcomes after the signal day. Entry = next session's open; ATR of the signal day is the unit.
       y_ret_h       return to the close of the h-th session counting the entry session as 1 (h in HORIZONS)
       y_mfe_h/mae_h highest high / lowest low within h sessions, in ATR from entry
       y_up_k / y_dn_k  session (1..FWD) on which the high first reached entry + k*ATR / the low first reached
                     entry - k*ATR. -1 = not reached within FWD sessions; null = unknown (data ends or a data
                     error comes first). Compare only positive values: "stop first, then target" =
                     (y_dn > 0) & (y_up > y_dn). The same session (y_up == y_dn) cannot be ordered with daily
                     bars; treat it as stop first (conservative).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

C = pl.col
G = "entity_id"
FWD = 60
HORIZONS = (1, 2, 3, 5, 10, 20, 40, 60)
UP_K = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DN_K = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BAD_JUMP = 0.6
EPS = 1e-9


def _k(x: float) -> str:
    return f"{x:g}".replace(".", "_")


def _streak(sign: pl.Expr, name: str) -> list[pl.Expr]:
    """Signed run length of a +1/-1/0 series (0 breaks the run), per entity."""
    run = (sign != sign.shift(1)).fill_null(True).cum_sum()
    n = pl.int_range(1, pl.len() + 1).over(G, run)
    return [(sign * n).alias(name)]


def _ret(n: int) -> pl.Expr:
    return C("close") / C("close").shift(n) - 1


SPLIT_LOOKBACK = 10  # sessions before the split day where the unadjusted step may sit
SPLIT_MIN = np.log(
    1.25
)  # smaller ratios (spin-offs etc.) are never repaired: too close to normal daily moves


def repair_splits(df: pl.DataFrame) -> pl.DataFrame:
    """The daily price job refetches the last 7 days, so a split that happened after the backfill can be missing
    from the stored history. The refetch rewrites the last few rows already adjusted, so the unadjusted step sits
    on the split day OR up to SPLIT_LOOKBACK sessions before it (where the refetched range starts). For every
    split with |ln ratio| >= ln 1.25, find the day in that window whose close-to-close move matches 1/ratio
    (within 25% of |ln ratio| in log terms) and adjust all rows before it (prices / ratio, volume x ratio), the
    same way the data source adjusts history. No matching step = the history is already adjusted."""
    if "splits" not in df.columns:
        return df
    df = df.sort(G, "as_of")
    s_all = df["splits"].fill_null(0).to_numpy()
    if not ((s_all > 0) & (np.abs(np.log(np.where(s_all > 0, s_all, 1.0))) >= SPLIT_MIN)).any():
        return df
    fix_parts = []
    for _, g in df.group_by(G, maintain_order=True):
        c = g["close"].to_numpy().astype(float)
        s = g["splits"].fill_null(0).to_numpy().astype(float)
        n = len(c)
        fix = np.ones(n)
        lr_day = np.full(n, np.nan)
        lr_day[1:] = np.log(c[1:] / c[:-1])
        for d in np.flatnonzero(s > 0):
            lr = np.log(s[d])
            if abs(lr) < SPLIT_MIN:
                continue
            ks = np.arange(max(1, d - SPLIT_LOOKBACK), d + 1)
            err = np.abs(lr_day[ks] + lr)
            ok = np.isfinite(err) & (err < 0.25 * abs(lr))
            if not ok.any():
                continue  # already adjusted
            k = ks[ok][np.argmin(err[ok])]
            fix[:k] *= s[d]
        fix_parts.append(fix)
    fix_all = np.concatenate(fix_parts)
    return (
        df.with_columns(pl.Series("_fix", fix_all))
        .with_columns(
            *[(C(k) / C("_fix")).alias(k) for k in ("open", "high", "low", "close")],
            (C("volume") * C("_fix")).alias("volume"),
        )
        .drop("_fix")
    )


def raw_close(df: pl.DataFrame) -> pl.Expr:
    """Prices are split-adjusted afterwards (a later 1-for-20 reverse split multiplies all older prices by 20),
    so price LEVELS leak the future. The price traders actually saw = adjusted close x product of the split
    ratios after that day (needs a `splits` column: ratio on the split day, 0/null = none). Run repair_splits
    first so that every split is applied to the history exactly once."""
    if "splits" not in df.columns:
        return C("close")
    f = pl.when(C("splits").fill_null(0) > 0).then(C("splits")).otherwise(1.0)
    after = f.reverse().cum_prod().reverse().over(G) / f  # product of ratios strictly after the day
    return C("close") * after


def stock_features(df: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, as_of, open, high, low, close, volume (+ splits). Returns df + raw_close + s_* columns.
    Price-level features use raw_close (see raw_close); everything else is scale-free."""
    df = repair_splits(df.sort(G, "as_of"))
    df = df.with_columns(raw_close(df).alias("raw_close"))
    pc = C("close").shift(1)
    tr = pl.max_horizontal(C("high") - C("low"), (C("high") - pc).abs(), (C("low") - pc).abs())
    df = df.with_columns(
        pc.over(G).alias("_pc"),
        tr.over(G).alias("_tr"),
        _ret(1).over(G).alias("_r1"),
        ((C("high") + C("low") + C("close")) / 3).alias("_tp"),
        (C("close") * C("volume")).alias("_dv"),
    )
    df = df.with_columns(
        C("_tr").rolling_mean(14).over(G).alias("atr"),
        C("volume").rolling_mean(20).shift(1).over(G).alias("_v20"),
        C("volume").rolling_mean(50).shift(1).over(G).alias("_v50"),
        C("_dv").rolling_mean(20).over(G).alias("dvol20"),
        (
            (C("_r1").abs() > BAD_JUMP).cast(pl.Int8).rolling_max(20, min_samples=1).over(G) > 0
        ).alias("bad20"),
    )
    a = C("atr").clip(lower_bound=EPS)
    rng = (C("high") - C("low")).clip(lower_bound=EPS)
    top, bot = pl.max_horizontal("open", "close"), pl.min_horizontal("open", "close")
    on = C("open") / C("_pc") - 1  # overnight
    intr = C("close") / C("open") - 1  # intraday
    up_v = pl.when(C("_r1") > 0).then(C("volume")).otherwise(0.0)
    sgn_v = C("_r1").sign() * C("volume")
    big = (C("_tr") > 2 * C("atr").shift(1)) & (C("volume") > 2 * C("_v20"))
    big_up = big & (C("close") > C("open"))
    big_dn = big & (C("close") < C("open"))
    gap_up = (C("open") - C("_pc")) > 1.0 * C("atr").shift(1)
    gap_dn = (C("_pc") - C("open")) > 1.0 * C("atr").shift(1)

    exprs: list[pl.Expr] = [
        # ---- one candle
        ((C("close") - C("open")) / a).alias("s_body"),
        ((C("high") - top) / a).alias("s_wick_up"),
        ((bot - C("low")) / a).alias("s_wick_dn"),
        (rng / a).alias("s_range"),
        ((C("close") - C("low")) / rng).alias("s_clv"),
        ((C("close") - C("open")).abs() / rng).alias("s_body_share"),
        ((C("open") - C("_pc")) / a).alias("s_gap"),
        # ---- overnight vs intraday, accumulated
        on.rolling_sum(5).alias("s_on5"),
        on.rolling_sum(20).alias("s_on20"),
        intr.rolling_sum(5).alias("s_id5"),
        intr.rolling_sum(20).alias("s_id20"),
        # ---- returns over many windows, also in volatility units
        *[_ret(n).alias(f"s_ret{n}") for n in (1, 2, 3, 5, 10, 20, 60, 120, 250)],
        *[
            ((C("close") - C("close").shift(n)) / (a * np.sqrt(n))).alias(f"s_retz{n}")
            for n in (1, 5, 20, 60)
        ],
        # ---- position against highs / lows of many windows
        *[
            ((C("close") - C("high").rolling_max(n)) / a).alias(f"s_dhi{n}")
            for n in (5, 20, 60, 250)
        ],
        *[
            ((C("close") - C("low").rolling_min(n)) / a).alias(f"s_dlo{n}")
            for n in (5, 20, 60, 250)
        ],
        *[
            (
                (C("close") - C("low").rolling_min(n))
                / (C("high").rolling_max(n) - C("low").rolling_min(n)).clip(lower_bound=EPS)
            ).alias(f"s_pos{n}")
            for n in (5, 20, 60, 250)
        ],  # fmt: skip
        # ---- moving averages: distance and slope
        *[
            ((C("close") - C("close").rolling_mean(n)) / a).alias(f"s_dma{n}")
            for n in (5, 10, 20, 50, 100, 200)
        ],
        (C("close").rolling_mean(20) / C("close").rolling_mean(20).shift(5) - 1).alias(
            "s_ma20_slope"
        ),
        (C("close").rolling_mean(50) / C("close").rolling_mean(50).shift(10) - 1).alias(
            "s_ma50_slope"
        ),
        (C("close").rolling_mean(200) / C("close").rolling_mean(200).shift(20) - 1).alias(
            "s_ma200_slope"
        ),
        ((C("close").rolling_mean(20) - C("close").rolling_mean(50)) / a).alias("s_ma20_50"),
        ((C("close").rolling_mean(50) - C("close").rolling_mean(200)) / a).alias("s_ma50_200"),
        # ---- volatility
        (C("atr") / C("close")).alias("s_atrp"),
        (C("_tr").rolling_mean(5) / C("_tr").rolling_mean(50).clip(lower_bound=EPS)).alias(
            "s_atr5_50"
        ),
        (C("_tr").rolling_mean(20) / C("_tr").rolling_mean(120).clip(lower_bound=EPS)).alias(
            "s_atr20_120"
        ),
        C("_r1").rolling_std(20).alias("s_vol20"),
        (C("_r1").rolling_std(5) / C("_r1").rolling_std(60).clip(lower_bound=EPS)).alias(
            "s_vol5_60"
        ),
        (
            (C("high").rolling_max(5) - C("low").rolling_min(5))
            / (C("high").rolling_max(60) - C("low").rolling_min(60)).clip(lower_bound=EPS)
        ).alias("s_squeeze"),
        (C("_r1").rolling_max(20) / C("_r1").rolling_std(60).clip(lower_bound=EPS)).alias(
            "s_maxup20"
        ),
        (C("_r1").rolling_min(20) / C("_r1").rolling_std(60).clip(lower_bound=EPS)).alias(
            "s_maxdn20"
        ),
        C("_r1").rolling_skew(60).alias("s_skew60"),
        pl.rolling_corr(C("_r1"), C("_r1").shift(1), window_size=60).alias("s_autocorr60"),
        # ---- volume
        (C("volume") / C("_v20").clip(lower_bound=EPS)).alias("s_vr20"),
        (C("volume") / C("_v50").clip(lower_bound=EPS)).alias("s_vr50"),
        (C("volume").rolling_mean(5) / C("_v50").clip(lower_bound=EPS)).alias("s_v5_50"),
        (C("volume").rolling_mean(20) / C("volume").rolling_mean(120).clip(lower_bound=EPS)).alias(
            "s_v20_120"
        ),
        (up_v.rolling_sum(20) / C("volume").rolling_sum(20).clip(lower_bound=EPS)).alias(
            "s_upvol20"
        ),
        (sgn_v.rolling_sum(20) / C("volume").rolling_sum(20).clip(lower_bound=EPS)).alias(
            "s_flow20"
        ),
        (sgn_v.rolling_sum(5) / C("volume").rolling_sum(5).clip(lower_bound=EPS)).alias("s_flow5"),
        (C("volume") > C("_v20")).cast(pl.Int8).rolling_sum(10).alias("s_hivol_days10"),
        (
            (C("_r1").abs() / C("_dv").clip(lower_bound=1.0)).rolling_mean(20)
            / (C("_r1").abs() / C("_dv").clip(lower_bound=1.0))
            .rolling_mean(250)
            .clip(lower_bound=EPS)
        ).alias("s_illiq_chg"),  # fmt: skip
        (C("_dv").rolling_max(20) / C("_dv").rolling_sum(20).clip(lower_bound=EPS)).alias(
            "s_vol_conc20"
        ),
        # volume on down days in the last 5 vs the prior 20 ("selling dries up")
        (
            pl.when(C("_r1") < 0).then(C("volume")).otherwise(None).rolling_mean(5, min_samples=1)
            / pl.when(C("_r1") < 0)
            .then(C("volume"))
            .otherwise(None)
            .rolling_mean(25, min_samples=3)
            .clip(lower_bound=EPS)
        ).alias("s_dnvol_dry"),  # fmt: skip
        # ---- cost basis of recent buyers (volume-weighted price)
        *[
            (
                (
                    C("close")
                    - (C("_tp") * C("volume")).rolling_sum(n)
                    / C("volume").rolling_sum(n).clip(lower_bound=EPS)
                )
                / a
            ).alias(f"s_vwap{n}")
            for n in (5, 20, 60, 250)
        ],  # fmt: skip
        # ---- liquidity / price level
        C("dvol20").log10().alias("s_ldvol"),
        C("raw_close").log10().alias("s_lprice"),
        (((C("raw_close") / 10).round(0) * 10 - C("raw_close")).abs() / C("raw_close")).alias(
            "s_round10"
        ),
    ]
    df = df.with_columns([e.over(G) for e in exprs])

    # ---- sequences (streaks)
    df = df.with_columns(
        C("_r1").sign().fill_null(0).alias("_sg"),
        (C("low") - C("low").shift(1)).over(G).sign().fill_null(0).alias("_lsg"),
        (C("high") - C("high").shift(1)).over(G).sign().fill_null(0).alias("_hsg"),
        ((C("high") <= C("high").shift(1)) & (C("low") >= C("low").shift(1)))
        .over(G)
        .fill_null(False)
        .cast(pl.Int8)
        .alias("_in"),
    )
    df = df.with_columns(
        *_streak(C("_sg"), "s_streak"),
        *_streak(C("_lsg"), "s_low_streak"),
        *_streak(C("_hsg"), "s_high_streak"),
        *_streak(C("_in"), "s_inside_streak"),
    )
    run = (C("_sg") != C("_sg").shift(1)).fill_null(True).cum_sum().over(G)
    df = df.with_columns(run.alias("_run"))
    df = df.with_columns(
        (C("close") / C("_pc").first().over(G, "_run") - 1).alias("s_streak_ret"),
        (C("_sg").rolling_sum(10)).over(G).alias("s_updays10"),
    )

    # ---- memory of special days: days since, and where the price is inside that day's range
    df = df.with_columns(pl.int_range(0, pl.len()).over(G).alias("_i"))
    mem = []
    for name, cond in (("bigup", big_up), ("bigdn", big_dn), ("gapup", gap_up), ("gapdn", gap_dn)):
        mem += [
            pl.when(cond).then(C("_i")).otherwise(None).forward_fill().over(G).alias(f"_{name}_i"),
            pl.when(cond)
            .then(C("high"))
            .otherwise(None)
            .forward_fill()
            .over(G)
            .alias(f"_{name}_h"),
            pl.when(cond).then(C("low")).otherwise(None).forward_fill().over(G).alias(f"_{name}_l"),
            pl.when(cond)
            .then(C("_pc"))
            .otherwise(None)
            .forward_fill()
            .over(G)
            .alias(f"_{name}_pc"),
        ]
    df = df.with_columns(mem)
    out = []
    for name in ("bigup", "bigdn", "gapup", "gapdn"):
        out += [
            (C("_i") - C(f"_{name}_i")).clip(upper_bound=500).alias(f"s_{name}_age"),
            (
                (C("close") - C(f"_{name}_l"))
                / (C(f"_{name}_h") - C(f"_{name}_l")).clip(lower_bound=EPS)
            ).alias(f"s_{name}_pos"),
        ]
    out += [
        ((C("close") - C("_gapup_pc")) / a).alias("s_gapup_fill"),  # <0: the gap up has been filled
        ((C("close") - C("_gapdn_pc")) / a).alias(
            "s_gapdn_fill"
        ),  # >0: the gap down has been filled
    ]
    df = df.with_columns(out)
    df = _window_features(df)
    return df.drop([c for c in df.columns if c.startswith("_")])


def _window_features(df: pl.DataFrame) -> pl.DataFrame:
    """Things polars has no rolling op for: days since the N-day high / low, share of recent volume traded
    above today's close ("trapped buyers")."""
    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "s_age_hi20",
            "s_age_lo20",
            "s_age_hi60",
            "s_age_lo60",
            "s_trapped60",
            "s_trapped20",
        )
    }
    for _, g in df.group_by(G, maintain_order=True):
        h, lo = g["high"].to_numpy().astype(float), g["low"].to_numpy().astype(float)
        c, tp, v = (
            g["close"].to_numpy().astype(float),
            g["_tp"].to_numpy().astype(float),
            g["volume"].to_numpy().astype(float),
        )
        n = len(c)
        for N in (20, 60):
            for arr, key, fill in ((h, f"s_age_hi{N}", -np.inf), (lo, f"s_age_lo{N}", np.inf)):
                res = np.full(n, np.nan)
                if n >= N:
                    w = sliding_window_view(np.nan_to_num(arr, nan=fill), N)[:, ::-1]
                    res[N - 1 :] = w.argmax(1) if fill < 0 else w.argmin(1)
                cols[key].append(res)
            res = np.full(n, np.nan)
            if n >= N:
                wt, wv = sliding_window_view(tp, N), sliding_window_view(np.nan_to_num(v), N)
                tot = wv.sum(1)
                res[N - 1 :] = np.where(
                    tot > 0,
                    (wv * (wt > c[N - 1 :, None])).sum(1) / np.where(tot > 0, tot, 1),
                    np.nan,
                )
            cols[f"s_trapped{N}"].append(res)
    return df.with_columns(
        [pl.Series(k, np.concatenate(v), dtype=pl.Float32).fill_nan(None) for k, v in cols.items()]
    )


# features that also get "how does it stand against the stock's own past"
PAST = ["s_body", "s_wick_up", "s_wick_dn", "s_range", "s_clv", "s_gap", "s_ret1", "s_ret5", "s_ret20", "s_vr20",
        "s_flow20", "s_upvol20", "s_atrp", "s_vol20", "s_squeeze", "s_dma20", "s_dma50", "s_vwap20", "s_vwap60",
        "s_trapped60", "s_on20", "s_id20", "s_autocorr60", "s_ldvol"]  # fmt: skip


def past_relative(df: pl.DataFrame) -> pl.DataFrame:
    """z-score against the own last 60 / 250 sessions, 5-day change, and acceleration (change of the change)."""
    ex = []
    for f in PAST:
        x = C(f)
        ex += [
            ((x - x.rolling_mean(60)) / x.rolling_std(60).clip(lower_bound=EPS))
            .over(G)
            .alias(f"{f}__z60"),
            ((x - x.rolling_mean(250)) / x.rolling_std(250).clip(lower_bound=EPS))
            .over(G)
            .alias(f"{f}__z250"),
            (x - x.shift(5)).over(G).alias(f"{f}__d5"),
            ((x - x.shift(5)) - (x.shift(5) - x.shift(10))).over(G).alias(f"{f}__acc"),
        ]
    return df.sort(G, "as_of").with_columns(ex)


def path_labels(df: pl.DataFrame) -> pl.DataFrame:
    """y_* outcomes (see module doc). Needs entity_id, as_of, open, high, low, close, atr; sorted by entity/date."""
    df = df.sort(G, "as_of")
    keys = (
        [f"y_ret{h}" for h in HORIZONS]
        + [f"y_mfe{h}" for h in (5, 20, 60)]
        + [f"y_mae{h}" for h in (5, 20, 60)]
    )
    keys += [f"y_up{_k(k)}" for k in UP_K] + [f"y_dn{_k(k)}" for k in DN_K]
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for _, g in df.group_by(G, maintain_order=True):
        o, h, lo, c = (g[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
        atr = g["atr"].to_numpy().astype(float)
        n = len(c)
        bad = np.zeros(n, bool)
        bad[1:] = np.abs(c[1:] / c[:-1] - 1) > BAD_JUMP
        e = np.full(n, np.nan)
        e[:-1] = o[1:]  # entry at next open
        ok = np.isfinite(e) & (e > 0) & np.isfinite(atr) & (atr > 0)
        # forward windows of length FWD starting at the entry session (t+1)
        pad = lambda x: np.concatenate([x[1:], np.full(FWD, np.nan)])  # noqa: E731
        H = sliding_window_view(pad(h), FWD)[:n]
        L = sliding_window_view(pad(lo), FWD)[:n]
        Cc = sliding_window_view(pad(c), FWD)[:n]
        B = sliding_window_view(np.concatenate([bad[1:], np.zeros(FWD, bool)]), FWD)[:n]
        badfwd = np.cumsum(B, axis=1) > 0  # a data error at or before session j
        avail = np.isfinite(Cc)
        for hz in HORIZONS:
            j = hz - 1
            v = Cc[:, j] / e - 1
            out[f"y_ret{hz}"].append(np.where(ok & avail[:, j] & ~badfwd[:, j], v, np.nan))
        for hz in (5, 20, 60):
            j = hz - 1
            mf = (np.nanmax(np.where(np.isfinite(H[:, :hz]), H[:, :hz], -np.inf), 1) - e) / atr
            ma = (np.nanmin(np.where(np.isfinite(L[:, :hz]), L[:, :hz], np.inf), 1) - e) / atr
            good = ok & avail[:, j] & ~badfwd[:, j]
            out[f"y_mfe{hz}"].append(np.where(good, mf, np.nan))
            out[f"y_mae{hz}"].append(np.where(good, ma, np.nan))
        valid_upto = np.where(badfwd, 0, avail).cumprod(1).sum(1)  # sessions with clean data
        for k in UP_K:
            hit = ((e + k * atr)[:, None] <= H) & ~badfwd
            first = np.where(hit.any(1), hit.argmax(1) + 1, 0).astype(float)
            first[(first == 0) & (valid_upto < FWD)] = (
                np.nan
            )  # not hit, but the window is incomplete: unknown
            first[first == 0] = -1  # not hit within FWD
            first[~ok] = np.nan
            out[f"y_up{_k(k)}"].append(first)
        for k in DN_K:
            hit = ((e - k * atr)[:, None] >= L) & ~badfwd
            first = np.where(hit.any(1), hit.argmax(1) + 1, 0).astype(float)
            first[(first == 0) & (valid_upto < FWD)] = np.nan
            first[first == 0] = -1
            first[~ok] = np.nan
            out[f"y_dn{_k(k)}"].append(first)
    res = df.with_columns(
        [pl.Series(k, np.concatenate(v), dtype=pl.Float32).fill_nan(None) for k, v in out.items()]
    )
    # -1 = not reached within FWD sessions (kept distinct from null = unknown)
    return res


def market_features(px: pl.DataFrame, vol: pl.DataFrame, fred: pl.DataFrame) -> pl.DataFrame:
    """One row per date: breadth and dispersion from the stock table, VIX family, rates / credit (by publication
    date, so nothing is known before it was published). px: common stocks with s_* already computed."""
    br = (
        px.group_by("as_of")
        .agg(
            (C("s_dma50") > 0).mean().alias("m_above50"),
            (C("s_dma200") > 0).mean().alias("m_above200"),
            (C("s_dhi250") >= 0).mean().alias("m_newhi"),
            (C("s_dlo250") <= 0).mean().alias("m_newlo"),
            (C("s_ret1") > 0).mean().alias("m_adv"),
            C("s_ret1").std().alias("m_disp1"),
            C("s_ret20").std().alias("m_disp20"),
            C("s_ret20").median().alias("m_med_ret20"),
            (C("s_vr20") > 2).mean().alias("m_hivol"),
        )
        .sort("as_of")
    )
    br = br.with_columns(
        (C("m_newhi") - C("m_newlo")).alias("m_hilo"),
        C("m_adv").rolling_mean(10).alias("m_adv10"),
        (C("m_above50") - C("m_above50").shift(5)).alias("m_above50_d5"),
    )
    spy = px.filter(C(G) == "US:SPY").select("as_of", *[C(c).alias("m_spy" + c[1:]) for c in
                                                      ("s_ret1", "s_ret5", "s_ret20", "s_dma50", "s_dma200", "s_streak", "s_vol20", "s_dhi250")])  # fmt: skip
    v = vol.pivot(on=G, index="as_of", values="close").sort("as_of")
    ren = {c: "m_" + c.split(":")[-1].lower() for c in v.columns if c != "as_of"}
    v = v.rename(ren)
    vx = [c for c in ("m_vix", "m_vix3m", "m_vix9d", "m_vvix", "m_skew") if c in v.columns]
    v = v.select("as_of", *vx).with_columns(
        (C("m_vix") / C("m_vix3m")).alias("m_vix_term"),
        (C("m_vix9d") / C("m_vix")).alias("m_vix_9d"),
        (C("m_vix") / C("m_vix").shift(5) - 1).alias("m_vix_chg5"),
        ((C("m_vix") - C("m_vix").rolling_mean(250)) / C("m_vix").rolling_std(250)).alias(
            "m_vix_z250"
        ),
    )
    out = br.join(spy, on="as_of", how="left").join(v, on="as_of", how="left")
    out = out.with_columns(
        (C("m_vix") / 100 / (C("m_spy_vol20") * np.sqrt(252)).clip(lower_bound=EPS)).alias("m_vrp")
    )
    series = {"DGS10": "m_y10", "T10Y2Y": "m_curve", "BAMLH0A0HYM2": "m_hy", "DTWEXBGS": "m_usd", "DCOILWTICO": "m_oil",
              "NFCI": "m_nfci", "T10YIE": "m_infl_exp"}  # fmt: skip
    # FRED publishes after the US close (stored published_at ~23:00 UTC): usable from the next day. Values are the
    # latest revision, not the first print (revised series such as NFCI read slightly better than they were).
    for sid, name in series.items():
        s = (fred.filter(C(G) == f"MACRO:FRED:{sid}").select((C("published_at").dt.date() + pl.duration(days=1)).alias("known"), C("value").alias(name))
             .drop_nulls().sort("known", maintain_order=True).unique("known", keep="last", maintain_order=True))  # fmt: skip
        if s.height == 0:
            continue
        out = (
            out.sort("as_of")
            .join_asof(s, left_on="as_of", right_on="known", strategy="backward")
            .drop("known")
        )
        out = out.with_columns((C(name) - C(name).shift(20)).alias(f"{name}_d20"))
    d = out["as_of"]
    out = out.with_columns(
        d.dt.weekday().alias("m_dow"), d.dt.day().alias("m_dom"), d.dt.month().alias("m_month"),
        (C("as_of").rank("ordinal", descending=True).over(d.dt.year(), d.dt.month()) - 1).alias("m_to_month_end"),
    )  # fmt: skip
    return out


def cross_rank(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Percentile of each value among all stocks on the same day (x_ prefix)."""
    return df.with_columns(
        [
            (C(c).rank() / C(c).count()).over("as_of").cast(pl.Float32).alias("x_" + c[2:])
            for c in cols
        ]
    )
