"""Trade-level backtest engine for daily-bar strategies.

A strategy decides at the close of day t (PIT: only data <= t). Plan fixed at signal time:
  IN  = entry plan (next open, or a limit price valid for the next session only)
  LS  = stop price, TP = take-profit price, max_hold = time stop (sessions), optional exit rule.
Simulation (per entity, no overlapping trades):
  - fill at next open (or limit if next low <= limit; fill = min(open, limit) for longs)
  - skip if the open already gaps through the stop
  - each bar: stop first, then target (conservative when both touched in one bar);
    gaps through a level fill at the open
  - optional close-based exit rule (exit at that close); time stop exits at the close
  - round-trip costs by price bucket (bps) are deducted
Results are reported by period: train < 2024, valid = 2024, test >= 2025 (out-of-sample).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

TRAIN_END = date(2023, 12, 31)
VALID_END = date(2024, 12, 31)


@dataclass
class Strategy:
    key: str
    label: str
    horizon: str  # 短期 / 中期 / 長期
    side: str  # long / short
    entry: Callable[[], pl.Expr]  # boolean expr on indicator frame (signal day)
    stop: Callable[[], pl.Expr]  # price expr
    target: Callable[[], pl.Expr]  # price expr (inf/0 for none)
    max_hold: int
    limit: Callable[[], pl.Expr] | None = None  # limit entry price (next session only)
    exit_rule: Callable[[], pl.Expr] | None = None  # boolean expr evaluated on each close
    rank: Callable[[], pl.Expr] | None = None  # higher = better when picking today's signals
    params: dict[str, Any] = field(default_factory=dict)
    why: str = ""


def cost_bps(price: float, market: str) -> float:
    """Round-trip cost incl. spread/slippage (conservative)."""
    if market == "JP":
        return 20.0
    if price < 5:
        return 100.0
    if price < 20:
        return 30.0
    return 10.0


BAD_JUMP = 0.6  # one-day move treated as a data error / split artefact
BAD_LOOKBACK = 5


def indicators(df: pl.DataFrame, bench: str) -> pl.DataFrame:
    """df needs entity_id, as_of, open, high, low, close, volume (+ optional splits)."""
    g = "entity_id"
    df = df.sort([g, "as_of"])
    split_flag = (pl.col("splits").fill_null(0) != 0) if "splits" in df.columns else pl.lit(False)
    df = df.with_columns(
        (split_flag | ((pl.col("close") / pl.col("close").shift(1).over(g) - 1).abs() > BAD_JUMP))
        .fill_null(False)
        .alias("bad")
    )
    pc = pl.col("close").shift(1).over(g)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"), (pl.col("high") - pc).abs(), (pl.col("low") - pc).abs()
    )
    chg = pl.col("close").diff().over(g)
    up, dn = pl.when(chg > 0).then(chg).otherwise(0.0), pl.when(chg < 0).then(-chg).otherwise(0.0)
    df = (
        df.with_columns(
            pc.alias("prev_close"),
            tr.alias("tr"),
            pl.col("close").rolling_mean(5).over(g).alias("ma5"),
            pl.col("close").rolling_mean(20).over(g).alias("ma20"),
            pl.col("close").rolling_mean(50).over(g).alias("ma50"),
            pl.col("close").rolling_mean(200).over(g).alias("ma200"),
            pl.col("volume").rolling_mean(20).shift(1).over(g).alias("vol20"),
            (pl.col("close") * pl.col("volume")).rolling_mean(20).over(g).alias("dvol20"),
            pl.col("high").rolling_max(60).shift(1).over(g).alias("hi60p"),
            pl.col("high").rolling_max(20).shift(1).over(g).alias("hi20p"),
            pl.col("low").rolling_min(10).over(g).alias("lo10"),
            pl.col("low").rolling_min(20).shift(1).over(g).alias("lo20p"),
            up.rolling_mean(2).over(g).alias("_u2"),
            dn.rolling_mean(2).over(g).alias("_d2"),
            up.rolling_mean(14).over(g).alias("_u14"),
            dn.rolling_mean(14).over(g).alias("_d14"),
            (pl.col("close") / pl.col("close").shift(126).over(g) - 1).alias("ret126"),
            (pl.col("close").shift(5).over(g) / pl.col("close").shift(126).over(g) - 1).alias(
                "mom6_1w"
            ),
            (pl.col("close") / pl.col("close").shift(63).over(g) - 1).alias("ret63"),
            (pl.col("close") / pl.col("close").shift(1).over(g) - 1).alias("ret1"),
            (pl.col("open") / pc - 1).alias("gap"),
        )
        .with_columns(
            pl.col("tr").rolling_mean(14).over(g).alias("atr"),
            pl.col("tr").rolling_mean(10).over(g).alias("atr10"),
            pl.col("tr").rolling_mean(50).over(g).alias("atr50"),
            (
                100
                - 100
                / (
                    1
                    + pl.col("_u2") / pl.when(pl.col("_d2") > 0).then(pl.col("_d2")).otherwise(1e-9)
                )
            ).alias("rsi2"),
            (
                100
                - 100
                / (
                    1
                    + pl.col("_u14")
                    / pl.when(pl.col("_d14") > 0).then(pl.col("_d14")).otherwise(1e-9)
                )
            ).alias("rsi14"),
            (pl.col("volume") / pl.col("vol20")).alias("vr"),
            (
                (pl.col("close") - pl.col("low"))
                / (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
            ).alias("clv"),
            pl.col("ma50").shift(10).over(g).alias("ma50_10"),
        )
        .drop("_u2", "_d2", "_u14", "_d14")
    )
    # market regime + relative strength rank across the universe each day
    b = df.filter(pl.col("entity_id") == bench).select(
        "as_of",
        (pl.col("close") > pl.col("ma50")).alias("mkt_up50"),
        (pl.col("close") > pl.col("ma200")).alias("mkt_up200"),
        pl.col("ret63").alias("mkt_ret63"),
    )
    df = df.join(b, on="as_of", how="left")
    df = (
        df.with_columns(
            (pl.col("ret63") - pl.col("mkt_ret63")).alias("rs63"),
        )
        .with_columns(
            pl.col("rs63").rank(descending=False).over("as_of").alias("_r"),
            pl.col("rs63").count().over("as_of").alias("_n"),
            pl.col("mom6_1w").rank(descending=False).over("as_of").alias("_m"),
        )
        .with_columns(
            (pl.col("_r") / pl.col("_n")).alias("rs_pct"),
            (pl.col("_m") / pl.col("_n")).alias("mom_pct"),
        )
        .drop("_r", "_n", "_m")
    )
    return df


def simulate(ind: pl.DataFrame, s: Strategy, market: str) -> pl.DataFrame:
    cols = ["entity_id", "as_of", "open", "high", "low", "close"]
    frame = ind.with_columns(
        s.entry().fill_null(False).alias("_sig"),
        s.stop().alias("_stop"),
        s.target().alias("_tgt"),
        (s.limit() if s.limit else pl.lit(None, dtype=pl.Float64)).alias("_lim"),
        (s.exit_rule() if s.exit_rule else pl.lit(False)).fill_null(False).alias("_exit"),
    ).select(*cols, "_sig", "_stop", "_tgt", "_lim", "_exit", "bad")
    out: list[dict[str, Any]] = []
    long = s.side == "long"
    for (eid,), g in frame.group_by("entity_id"):
        sig = g["_sig"].to_numpy()
        if not sig.any():
            continue
        d = g["as_of"].to_list()
        o, h, lo, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
        stp, tgt, lim, ex = (
            g["_stop"].to_numpy(),
            g["_tgt"].to_numpy(),
            g["_lim"].to_numpy(),
            g["_exit"].to_numpy(),
        )
        bad = g["bad"].to_numpy()
        n, busy_until = len(d), -1
        for i in np.flatnonzero(sig):
            if i <= busy_until or i + 1 >= n or bad[max(0, i - BAD_LOOKBACK) : i + 1].any():
                continue
            e = i + 1
            stop, target = stp[i], tgt[i]
            if not np.isfinite(stop):
                continue
            if s.limit is not None:
                L = lim[i]
                if not np.isfinite(L):
                    continue
                if long and lo[e] > L or (not long) and h[e] < L:
                    continue  # limit not filled
                px = min(o[e], L) if long else max(o[e], L)
            else:
                px = o[e]
            if (long and px <= stop) or ((not long) and px >= stop):
                continue  # gapped through stop: no trade
            exit_px, reason, j = None, "time", e
            last = min(n - 1, e + s.max_hold - 1)
            for j in range(e, last + 1):
                if long:
                    if lo[j] <= stop:
                        exit_px, reason = (o[j] if o[j] < stop and j > e else stop), "stop"
                        break
                    if np.isfinite(target) and h[j] >= target:
                        exit_px, reason = (o[j] if o[j] > target and j > e else target), "target"
                        break
                else:
                    if h[j] >= stop:
                        exit_px, reason = (o[j] if o[j] > stop and j > e else stop), "stop"
                        break
                    if np.isfinite(target) and target > 0 and lo[j] <= target:
                        exit_px, reason = (o[j] if o[j] < target and j > e else target), "target"
                        break
                if ex[j] and j > e:
                    exit_px, reason = c[j], "rule"
                    break
            if exit_px is None:
                if last < e + s.max_hold - 1 and last == n - 1:
                    reason = "open"  # still running at data end
                exit_px = c[last]
                j = last
            if bad[e : j + 1].any():
                busy_until = j
                continue  # data error / split inside the trade: result unreliable, drop
            gross = (exit_px / px - 1) if long else (px / exit_px - 1)
            risk = abs(px - stop) / px
            net = gross - cost_bps(px, market) / 1e4
            out.append({
                "entity_id": eid, "signal_date": d[i], "entry_date": d[e], "exit_date": d[j], "entry": px,
                "stop": float(stop), "target": float(target) if np.isfinite(target) else None, "exit": float(exit_px),
                "reason": reason, "hold": j - e + 1, "ret": net, "R": net / risk if risk > 0 else None,
            })  # fmt: skip
            busy_until = j
    if not out:
        return pl.DataFrame()
    return pl.DataFrame(out).with_columns(pl.lit(s.key).alias("strategy"))


def period(d: pl.Expr) -> pl.Expr:
    return (
        pl.when(d <= TRAIN_END)
        .then(pl.lit("train"))
        .when(d <= VALID_END)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
    )


def metrics(tr: pl.DataFrame) -> pl.DataFrame:
    """Per strategy x period: trades, win%, avg/median return, expectancy R, profit factor, hold, max DD (R)."""
    if tr.height == 0:
        return pl.DataFrame()
    tr = tr.filter(pl.col("reason") != "open").with_columns(
        period(pl.col("signal_date")).alias("period")
    )
    rows = []
    for (k, p), g in tr.sort("exit_date").group_by("strategy", "period"):
        r = g["ret"].to_numpy()
        R = g["R"].fill_null(0).to_numpy()
        wins, losses = r[r > 0].sum(), -r[r < 0].sum()
        eq = np.cumsum(R)
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
        yrs = max(1e-9, (g["exit_date"].max() - g["entry_date"].min()).days / 365.25)
        rows.append({
            "strategy": k, "period": p, "trades": len(r), "per_year": len(r) / yrs,
            "win": float((r > 0).mean()), "avg": float(r.mean()), "median": float(np.median(r)),
            "exp_R": float(R.mean()), "pf": float(wins / losses) if losses > 0 else None,
            "hold": float(g["hold"].mean()), "max_dd_R": dd,
            "tp_rate": float((g["reason"] == "target").mean()), "sl_rate": float((g["reason"] == "stop").mean()),
        })  # fmt: skip
    return pl.DataFrame(rows).sort("strategy", "period")
