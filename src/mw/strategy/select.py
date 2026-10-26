"""Run the strategy grid, select parameters, apply the adoption gate, and build today's picks.

Protocol (DESIGN §11, §13.5):
  - parameters chosen on train (<2024) + valid (2024) only
  - adoption gate: PF > 1.05 and positive average return in train AND valid AND test (>=2025,
    never used for selection), with >= 30 trades in each period
  - today's picks only from adopted strategies, ranked, with IN/TP/LS fixed at signal time
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl

from mw.strategy import engine, library

MIN_TRADES = 30
MIN_PF = 1.05


def _m(met: pl.DataFrame, key: str, period: str) -> dict[str, Any] | None:
    r = met.filter((pl.col("strategy") == key) & (pl.col("period") == period))
    return r.to_dicts()[0] if r.height else None


def _ok(m: dict[str, Any] | None) -> bool:
    return bool(m and m["trades"] >= MIN_TRADES and (m["pf"] or 0) > MIN_PF and m["avg"] > 0)


def run(ind: pl.DataFrame, market: str, bench: str) -> dict[str, Any]:
    strategies = library.grid(market, bench)
    trades, mets = [], []
    for s in strategies:
        tr = engine.simulate(ind, s, market)
        if tr.height:
            trades.append(tr)
            mets.append(engine.metrics(tr))
    met = pl.concat(mets) if mets else pl.DataFrame()
    allt = pl.concat(trades, how="diagonal_relaxed") if trades else pl.DataFrame()
    by_fam: dict[str, list[Any]] = {}
    for s in strategies:
        by_fam.setdefault(library.family(s.key), []).append(s)
    families = []
    for fam, ss in by_fam.items():
        # pick the variant with the best valid PF among those passing train+valid
        cands = [
            (s, _m(met, s.key, "train"), _m(met, s.key, "valid"), _m(met, s.key, "test"))
            for s in ss
        ]
        passing = [c for c in cands if _ok(c[1]) and _ok(c[2])]
        pool = passing or [c for c in cands if c[2]]
        if not pool:
            continue
        best = max(pool, key=lambda c: (c[2] or {}).get("pf") or 0)
        s, tr_m, va_m, te_m = best
        adopted = bool(passing) and _ok(te_m)
        reason = (
            "3期間すべてで利益（採用）"
            if adopted
            else (
                "検証期間（2025年〜）で利益が出なかった"
                if passing
                else "開発期間・選定期間で利益が出なかった"
            )
        )
        families.append({"family": fam, "strategy": library.describe(s), "adopted": adopted, "reason": reason,
                         "train": tr_m, "valid": va_m, "test": te_m, "variants_tested": len(ss)})  # fmt: skip
    return {"families": families, "trades": allt, "strategies": {s.key: s for s in strategies}}


def detail(trades: pl.DataFrame, key: str) -> dict[str, Any]:
    t = trades.filter((pl.col("strategy") == key) & (pl.col("reason") != "open")).sort("exit_date")
    if t.height == 0:
        return {}
    yearly = (
        t.with_columns(pl.col("signal_date").dt.year().alias("y"))
        .group_by("y")
        .agg(
            pl.len().alias("n"),
            (pl.col("ret") > 0).mean().alias("win"),
            pl.col("ret").mean().alias("avg"),
            (
                pl.col("ret").filter(pl.col("ret") > 0).sum()
                / (-pl.col("ret").filter(pl.col("ret") < 0).sum())
            ).alias("pf"),
        )  # fmt: skip
        .sort("y")
    )
    reasons = (
        t.group_by("reason")
        .agg(pl.len().alias("n"), pl.col("ret").mean().alias("avg"))
        .sort("n", descending=True)
    )
    # equity: equal fraction (10% of capital per trade), compounding by exit date
    eq, curve = 1.0, []
    for d, r in zip(t["exit_date"], t["ret"], strict=True):
        eq *= 1 + 0.1 * r
        curve.append([d.isoformat(), round(eq, 4)])
    peak, mdd = 0.0, 0.0
    for _, v in curve:
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak)
    hold = t["hold"].to_numpy()
    return {
        "yearly": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in yearly.to_dicts()
        ],
        "reasons": reasons.to_dicts(),
        "curve": curve[:: max(1, len(curve) // 400)] + ([curve[-1]] if curve else []),
        "max_dd": round(mdd, 4),
        "hold_pct": {str(q): float(np.percentile(hold, q)) for q in (25, 50, 75, 90)},
        "recent": [
            {
                **r,
                "signal_date": r["signal_date"].isoformat(),
                "entry_date": r["entry_date"].isoformat(),
                "exit_date": r["exit_date"].isoformat(),
            }
            for r in t.tail(40).reverse().to_dicts()
        ],  # fmt: skip
    }


def picks(ind: pl.DataFrame, s: engine.Strategy, asof: date, top: int = 15) -> list[dict[str, Any]]:
    """Signals on the latest bar, excluding names with recent data errors."""
    f = ind.with_columns(
        s.entry().fill_null(False).alias("_sig"),
        s.stop().alias("_stop"),
        s.target().alias("_tgt"),
        (s.rank() if s.rank else pl.lit(0.0)).alias("_rank"),
        pl.col("bad")
        .cast(pl.Int8)
        .rolling_sum(engine.BAD_LOOKBACK + 1)
        .over("entity_id")
        .alias("_badn"),
    ).filter((pl.col("as_of") == asof) & pl.col("_sig") & (pl.col("_badn").fill_null(0) == 0))
    out = []
    for r in f.sort("_rank", descending=True).head(top).iter_rows(named=True):
        ref = r["close"]
        stop, tgt = r["_stop"], r["_tgt"]
        risk = abs(ref - stop) if stop is not None else None
        out.append({
            "id": r["entity_id"], "close": ref, "stop": stop, "target": None if tgt is None or not np.isfinite(tgt) or tgt <= 0 else tgt,
            "rr": (abs(tgt - ref) / risk) if risk and tgt is not None and np.isfinite(tgt) and tgt > 0 else None,
            "risk_pct": risk / ref if risk else None,
            "volume": r["volume"], "vol20": r["vol20"], "vr": r["vr"], "dvol": r["close"] * r["volume"], "dvol20": r["dvol20"],
            "ret1": r["ret1"], "gap": r["gap"], "clv": r["clv"], "rsi2": r["rsi2"], "rsi14": r["rsi14"], "atr": r["atr"],
            "ma20": r["ma20"], "ma50": r["ma50"], "ma200": r["ma200"], "rs_pct": r["rs_pct"], "hi60p": r["hi60p"],
            "open": r["open"], "high": r["high"], "low": r["low"], "prev_close": r["prev_close"],
        })  # fmt: skip
    return out
