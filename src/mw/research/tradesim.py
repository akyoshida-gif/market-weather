"""Method B / C: trades with real exit rules, then a portfolio with money management (VALIDATION_PLAN §7, §5-6).

Why: a fixed "sell after h days" (Method A) holds losers to the deadline and cuts winners short, and it says
nothing about how much capital a pattern ties up. What compounds is R earned per day of capital in use.

simulate():  for each signal, enter at the next open, walk forward day by day with one exit rule.
  Every rule keeps the initial stop (gap below it -> exit at the open). Max hold 120 sessions.
  time_h      sell at the close after h sessions
  target_m    take profit at entry + m*R (intraday high touches it)
  chand_k     trailing stop: highest close since entry - k * ATR(at entry), never below the initial stop
  ma_n        sell at the close when it closes below its n-day moving average
  revert_n    sell at the close when it closes at/above its n-day moving average (rebounds), max 40 sessions
  dead_n      after n sessions, sell if not up at least +0.5R; otherwise trail like chand_3
  half2       sell half at +2R, trail the rest like chand_3
portfolio(): daily loop with limited slots and cash; risk a fixed share of equity per trade; marked to market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from mw.research import events

C = pl.col
G = "entity_id"
MAX_HOLD = 120
EXITS: list[tuple[str, float]] = (
    [("time", h) for h in (5, 10, 20, 40, 60, 120)]
    + [("target", m) for m in (1.5, 2.0, 3.0, 5.0)]
    + [("chand", k) for k in (2.0, 3.0, 4.0)]
    + [("ma", n) for n in (10, 20, 50)]
    + [("revert", n) for n in (10, 20, 50)]
    + [("dead", n) for n in (5, 10, 20)]
    + [("half2", 0.0)]
)
EXIT_LABELS = {"time": "{:g}日後に売る", "target": "+{:g}Rで利確", "chand": "最高値から{:g}ATR下で追いかける",
               "ma": "{:g}日線を割ったら売る", "revert": "{:g}日線まで戻ったら売る（反発用）",
               "dead": "{:g}日で+0.5Rに届かなければ売る（届けば追いかける）", "half2": "+2Rで半分売り、残りは追いかける"}  # fmt: skip
STOP_KINDS = ["sig_low", "low5", "low10", "low20", "ma20", "ma50", "atr1_5", "atr2", "atr3"]


def exit_label(kind: str, p: float) -> str:
    return EXIT_LABELS[kind].format(p)


@dataclass
class Panel:
    """Daily prices of all US stocks, sorted by (entity, date), with the indicators the exits need."""

    ent: np.ndarray
    date: np.ndarray  # datetime64[D]
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    ma: dict[int, np.ndarray]
    ll: dict[int, np.ndarray]
    raw_close: np.ndarray
    last: np.ndarray  # last row index of the row's entity
    keys: pl.DataFrame  # entity_id, as_of, row


def build_panel(prices: pl.DataFrame) -> Panel:
    p = prices.sort([G, "as_of"]).with_columns(
        C("close").shift(1).over(G).alias("pc"),
    )
    tr = pl.max_horizontal(
        C("high") - C("low"), (C("high") - C("pc")).abs(), (C("low") - C("pc")).abs()
    )
    lg = pl.when(C("splits").fill_null(0) > 0).then(C("splits").log()).otherwise(0.0)
    p = p.with_columns(
        tr.rolling_mean(14).over(G).alias("atr"),
        *[C("close").rolling_mean(n).over(G).alias(f"ma{n}") for n in (10, 20, 50)],
        *[C("low").rolling_min(n).over(G).alias(f"ll{n}") for n in (5, 10, 20)],
        (lg.sum().over(G) - lg.cum_sum().over(G)).exp().alias("split_fac"),
    )
    ent = p[G].to_numpy()
    n = len(ent)
    change = np.r_[ent[1:] != ent[:-1], True]
    ends = np.flatnonzero(change)
    last = np.repeat(ends, np.diff(np.r_[-1, ends]))
    f = lambda k: p[k].to_numpy().astype(np.float64)  # noqa: E731
    return Panel(
        ent=ent, date=p["as_of"].to_numpy().astype("datetime64[D]"), o=f("open"), h=f("high"), lo=f("low"), c=f("close"),
        atr=f("atr"), ma={k: f(f"ma{k}") for k in (10, 20, 50)}, ll={k: f(f"ll{k}") for k in (5, 10, 20)},
        raw_close=f("close") * f("split_fac"), last=last,
        keys=p.select(G, "as_of").with_columns(pl.Series("row", np.arange(n))),
    )  # fmt: skip


def signal_rows(panel: Panel, ev: pl.DataFrame) -> np.ndarray:
    """ev: entity_id, as_of of signals -> panel row index of the signal day."""
    j = ev.select(G, "as_of").join(panel.keys, on=[G, "as_of"], how="inner")
    return np.sort(j["row"].to_numpy())


def initial_stop(panel: Panel, i: np.ndarray, kind: str, entry: np.ndarray) -> np.ndarray:
    a = panel.atr[i]
    return {
        "sig_low": panel.lo[i] - 0.1 * a, "low5": panel.ll[5][i] - 0.1 * a, "low10": panel.ll[10][i] - 0.1 * a,
        "low20": panel.ll[20][i] - 0.1 * a, "ma20": panel.ma[20][i] - 0.2 * a, "ma50": panel.ma[50][i] - 0.2 * a,
        "atr1_5": entry - 1.5 * a, "atr2": entry - 2.0 * a, "atr3": entry - 3.0 * a,
    }[kind]  # fmt: skip


def simulate(
    panel: Panel, sig: np.ndarray, stop_kind: str, exit_kind: str, p: float
) -> dict[str, np.ndarray]:
    """One exit rule for every signal row in `sig`. Returns per-trade R (after costs), holding sessions,
    entry/exit row, stop distance as a share of the entry price, and a validity mask."""
    n = len(sig)
    e_row = sig + 1
    ok = e_row <= panel.last[sig]
    e_row = np.where(ok, e_row, sig)
    entry = panel.o[e_row]
    stop = initial_stop(panel, sig, stop_kind, entry)
    R = entry - stop
    a0 = panel.atr[sig]
    valid = (
        ok
        & np.isfinite(R)
        & (np.maximum(0.5 * a0, 0.01 * entry) <= R)
        & (0.5 * entry > R)
        & np.isfinite(entry)
    )
    maxh = 40 if exit_kind == "revert" else MAX_HOLD
    alive = valid.copy()
    out_r = np.full(n, np.nan)
    hold = np.zeros(n, dtype=np.int32)
    x_row = e_row.copy()
    trail = stop.copy()
    best_close = entry.copy()
    half_done = np.zeros(n, dtype=bool)
    banked = np.zeros(n)  # R already banked by a partial exit (weighted)
    for k in range(0, maxh):
        j = e_row + k
        inside = alive & (j <= panel.last[sig])
        if not inside.any() and not alive.any():
            break
        ended = alive & ~inside  # ran out of data: close at the last available bar
        if ended.any():
            jl = panel.last[sig][ended]
            out_r[ended] = (
                banked[ended]
                + (1 - 0.5 * half_done[ended]) * (panel.c[jl] - entry[ended]) / R[ended]
            )
            hold[ended] = jl - e_row[ended]
            x_row[ended] = jl
            alive &= ~ended
        jj = np.where(inside, j, e_row)
        o, hi, lo, c = panel.o[jj], panel.h[jj], panel.lo[jj], panel.c[jj]
        # 1) stop / trailing stop (gap through it -> the open)
        level = trail
        gap = inside & (k > 0) & (o <= level)
        hit = inside & ~gap & (lo <= level)
        px = np.where(gap, o, level)
        done = gap | hit
        w = 1 - 0.5 * half_done
        out_r[done] = banked[done] + w[done] * (px[done] - entry[done]) / R[done]
        # 2) profit-taking rules on days the stop was not hit
        still = inside & ~done
        if exit_kind == "target":
            tgt = entry + p * R
            th = still & (hi >= tgt)
            tpx = np.where(o >= tgt, o, tgt)
            out_r[th] = (tpx[th] - entry[th]) / R[th]
            done |= th
        elif exit_kind == "half2":
            tgt = entry + 2 * R
            th = still & ~half_done & (hi >= tgt)
            banked[th] = 0.5 * 2.0
            half_done |= th
        elif exit_kind == "ma":
            mh = still & (k > 0) & (c < panel.ma[int(p)][jj])
            out_r[mh] = banked[mh] + w[mh] * (c[mh] - entry[mh]) / R[mh]
            done |= mh
        elif exit_kind == "revert":
            mh = still & (c >= panel.ma[int(p)][jj])
            out_r[mh] = (c[mh] - entry[mh]) / R[mh]
            done |= mh
        elif exit_kind == "dead" and k == int(p):
            mh = still & (c < entry + 0.5 * R)
            out_r[mh] = (c[mh] - entry[mh]) / R[mh]
            done |= mh
        elif exit_kind == "time" and k == int(p) - 1:
            mh = still.copy()
            out_r[mh] = (c[mh] - entry[mh]) / R[mh]
            done |= mh
        hold[done] = k + 1
        x_row[done] = jj[done]
        alive &= ~done
        # 3) update the trailing stop with today's close (used from tomorrow)
        best_close = np.where(alive, np.maximum(best_close, c), best_close)
        if exit_kind == "chand" or exit_kind == "half2" or (exit_kind == "dead" and k >= int(p)):
            kk = 3.0 if exit_kind != "chand" else p
            trail = np.where(alive, np.maximum(trail, best_close - kk * a0), trail)
    # still open at max hold: sell at that close
    rest = alive
    if rest.any():
        jl = np.minimum(e_row[rest] + maxh - 1, panel.last[sig][rest])
        out_r[rest] = (
            banked[rest] + (1 - 0.5 * half_done[rest]) * (panel.c[jl] - entry[rest]) / R[rest]
        )
        hold[rest] = jl - e_row[rest] + 1
        x_row[rest] = jl
    rp = R / entry
    cost = events.cost_of(panel.raw_close[sig]) / np.where(rp > 0, rp, np.nan)
    return {"r": np.where(valid, out_r - cost, np.nan), "hold": hold, "entry_row": e_row, "exit_row": x_row,
            "rp": np.where(valid, rp, np.nan), "valid": valid}  # fmt: skip


def summarize(
    sim: dict[str, np.ndarray], mask: np.ndarray, dates: np.ndarray | None = None
) -> dict[str, Any]:
    """dates: entry date of every trade (for a t-value with same-day trades averaged first)."""
    v = sim["valid"] & mask & np.isfinite(sim["r"])
    r, hd = sim["r"][v], sim["hold"][v].astype(np.float64)
    n = len(r)
    if n == 0:
        return {"n": 0}
    t = events._clustered_t(r, dates[v].astype(np.int64)) if dates is not None else 0.0
    k10 = max(1, n // 10)
    top = -np.partition(-r, k10 - 1)[:k10] if n > k10 else r
    return {"n": n, "exp": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean()), "hold": float(hd.mean()),
            "r_per_100d": float(r.sum() / max(hd.sum(), 1) * 100), "top10": float(top.mean()),
            "big": float((r >= 3).mean()), "worst": float(r.min()), "t": t}  # fmt: skip


# ---------------------------------------------------------------- portfolio
def _mark(panel: Panel, q: dict[str, Any], d: Any) -> float:
    """Value of an open position at the close of day d (last bar on or before d, within the trade)."""
    er, xr = q["entry_row"], q["exit_row"]
    k = er + int(np.searchsorted(panel.date[er : xr + 1], d, side="right")) - 1
    return float(q["value"] * panel.c[max(er, min(k, xr))] / q["entry_px"])


def portfolio(panel: Panel, trades: pl.DataFrame, start: Any, end: Any, risk: float = 0.01, max_pos: int = 10,
              max_weight: float = 0.25, start_equity: float = 100_000.0, scale: dict[Any, float] | None = None) -> dict[str, Any]:  # fmt: skip
    """trades: one row per candidate trade with entry_row, exit_row, r, rp, priority, pattern, entity.
    Each day: close the trades that exit, then open new ones by priority while slots and cash allow.
    Position value = equity * risk / stop distance, capped at max_weight of equity and the cash left.
    scale: optional {date: multiplier of risk} (market-wide timing overlay)."""
    t = trades.filter(C("entry_date").is_between(start, end)).sort(
        ["entry_date", "priority"], descending=[False, True]
    )
    days = np.unique(
        panel.date[(panel.date >= np.datetime64(start)) & (panel.date <= np.datetime64(end))]
    )
    by_day: dict[Any, list[dict[str, Any]]] = {}
    for row in t.iter_rows(named=True):
        by_day.setdefault(np.datetime64(row["entry_date"], "D"), []).append(row)
    cash = start_equity
    pos: list[dict[str, Any]] = []
    curve, taken, skipped = [], [], 0
    for d in days:
        # exits (at the exit row's price; trade R already includes the exit price and costs)
        keep = []
        for q in pos:
            if panel.date[q["exit_row"]] <= d:
                cash += q["value"] * (1 + q["r"] * q["rp"])
                taken.append({**q, "exit_date": str(panel.date[q["exit_row"]])})
            else:
                keep.append(q)
        pos = keep
        held = {q["entity"] for q in pos}
        mtm = cash + sum(_mark(panel, q, d) for q in pos)
        m = (scale or {}).get(d, 1.0)
        for c in by_day.get(d, []):
            if len(pos) >= max_pos or c["entity"] in held or m <= 0:
                skipped += 1
                continue
            value = min(mtm * risk * m / c["rp"], mtm * max_weight, cash)
            if value < mtm * 0.01:
                skipped += 1
                continue
            cash -= value
            er = int(c["entry_row"])
            pos.append({"entity": c["entity"], "pattern": c["pattern"], "value": value, "r": c["r"], "rp": c["rp"],
                        "entry_row": er, "exit_row": int(c["exit_row"]), "entry_px": panel.o[er]})  # fmt: skip
            held.add(c["entity"])
        mtm = cash + sum(_mark(panel, q, d) for q in pos)
        curve.append((str(d), mtm, len(pos)))
    eq = np.array([x[1] for x in curve])
    if len(eq) < 2:
        return {"trades": 0}
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    years = len(eq) / 252
    cagr = (eq[-1] / start_equity) ** (1 / years) - 1 if years > 0 else 0.0
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    months: dict[str, float] = {}
    for dstr, v, _ in curve:
        months[dstr[:7]] = v
    mvals = np.array(list(months.values()))
    mret = np.diff(np.r_[start_equity, mvals]) / np.r_[start_equity, mvals][:-1]
    tr = [q["r"] for q in taken]
    by_pat: dict[str, list[float]] = {}
    for q in taken:
        by_pat.setdefault(q["pattern"], []).append(q["r"])
    return {
        "start": str(days[0]), "end": str(days[-1]), "final": float(eq[-1]), "cagr": float(cagr), "max_dd": float(dd.min()),
        "sharpe": sharpe, "trades": len(taken), "skipped": skipped, "avg_r": float(np.mean(tr)) if tr else 0.0,
        "win": float(np.mean([x > 0 for x in tr])) if tr else 0.0, "avg_positions": float(np.mean([x[2] for x in curve])),
        "months_pos": float((mret > 0).mean()), "worst_month": float(mret.min()), "best_month": float(mret.max()),
        "by_pattern": {k: {"n": len(v), "avg_r": float(np.mean(v)), "sum_r": float(np.sum(v))} for k, v in by_pat.items()},
        "curve": [(x[0], round(x[1], 2)) for x in curve[::5]],
    }  # fmt: skip
