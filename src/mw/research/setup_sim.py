"""Trades from armed setups, managed like a real trader would (VALIDATION_PLAN_V3 §3).

Entry: a buy-stop at the setup's level, placed after the close of day t and kept for `valid` sessions.
  Filled on the first day whose high reaches the level, at max(open, level); not filled (skipped) when that
  day opens more than 5% above the level (too extended to chase) or when price closes below the setup's
  stop before triggering (the pattern failed first).
Initial stop: the pattern's own invalidation point, but never more than 8% below the entry (textbook cap).
  On the entry day itself a low at/below the stop counts as stopped (conservative: we cannot know whether
  the low came before or after the fill).
Exit variants (all with the initial stop; a gap through a stop exits at the open):
  target   - take profit at the pattern's measured-move target (3R if the pattern has none); max 60 sessions
  ma50     - hold while above the 50-day line: exit at the close of the first close below it; max 120
  half     - sell half at the target (or 2R), trail the rest under the 20-day line; max 120
  time20   - exit at the close after 20 sessions (reference)
Costs: round trip by price as in VALIDATION_PLAN §1. Results in R (multiples of the initial risk).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mw.research.events import cost_of
from mw.research.setups import Armed

EXITS = {
    "target": "形の目標で利確",
    "ma50": "50日線を割るまで持つ",
    "half": "目標で半分売り、残りは20日線割れまで",
    "time20": "20日後に売る（基準）",
}
MAX_STOP = 0.08
CHASE = 1.05


def entry(a: Armed, o: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[int, float] | None:
    n = len(o)
    for j in range(a.t + 1, min(a.t + 1 + a.valid, n)):
        if h[j] >= a.level:
            if o[j] > a.level * CHASE:
                return None
            return j, max(o[j], a.level)
        if c[j] < a.stop:
            return None
    return None


def manage(kind: str, j: int, px: float, stop: float, target: float | None,
           o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, ma20: np.ndarray, ma50: np.ndarray) -> tuple[float, int, str]:  # fmt: skip
    """Walk forward from the entry bar j. Returns (R before costs, sessions held, exit reason)."""
    n = len(c)
    R = px - stop
    tgt = (
        target
        if (target is not None and target > px)
        else px + (2 * R if kind == "half" else 3 * R)
    )
    maxh = {"target": 60, "ma50": 120, "half": 120, "time20": 20}[kind]
    half = False
    banked = 0.0
    k = j
    for k in range(j, min(j + maxh, n)):
        w = 0.5 if half else 1.0
        if k > j and o[k] <= stop:
            return banked + w * (o[k] - px) / R, k - j + 1, "損切り（窓）"
        if lo[k] <= stop:
            return banked + w * (stop - px) / R, k - j + 1, "損切り"
        if kind in ("target", "half") and h[k] >= tgt:
            fill = max(o[k], tgt) if k > j else tgt
            if kind == "target":
                return (fill - px) / R, k - j + 1, "利確"
            if not half:
                half, banked = True, 0.5 * (fill - px) / R
        if kind == "ma50" and k > j and np.isfinite(ma50[k]) and c[k] < ma50[k]:
            return (c[k] - px) / R, k - j + 1, "50日線割れ"
        if kind == "half" and half and k > j and np.isfinite(ma20[k]) and c[k] < ma20[k]:
            return banked + 0.5 * (c[k] - px) / R, k - j + 1, "20日線割れ"
    w = 0.5 if half else 1.0
    return banked + w * (c[k] - px) / R, k - j + 1, "期限"


def trades_for_stock(
    armed: list[Armed], cols: dict[str, np.ndarray], attrs: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """All exit variants for every armed setup that triggers. attrs: per-bar context known at the close."""
    o, h, lo, c = cols["open"], cols["high"], cols["low"], cols["close"]
    out = []
    for a in armed:
        e = entry(a, o, h, c)
        if e is None:
            continue
        j, px = e
        stop = max(a.stop, px * (1 - MAX_STOP))
        if stop >= px:
            continue
        R = px - stop
        bad = cols["bad"][max(0, a.t - 252) : min(len(c), j + 121)].any()
        cost_r = float(cost_of(np.array([cols["raw_close"][a.t]]))[0]) * px / R
        row: dict[str, Any] = {"setup": a.setup, "t": a.t, "j": j, "entry": px, "stop": stop, "risk_pct": R / px, "bad": bool(bad),
                               "capped": stop > a.stop, **{k: (v[a.t].item() if hasattr(v[a.t], "item") else v[a.t]) for k, v in attrs.items()}}  # fmt: skip
        for kind in EXITS:
            r, held, why = manage(
                kind, j, px, stop, a.target, o, h, lo, c, cols["ma20"], cols["ma50"]
            )
            row[f"r_{kind}"] = r - cost_r
            row[f"d_{kind}"] = held
            row[f"x_{kind}"] = why
        out.append(row)
    return out


def random_trades(cols: dict[str, np.ndarray], attrs: dict[str, np.ndarray], rng: np.random.Generator, per_stock: int,
                  risk_pct: float) -> list[dict[str, Any]]:  # fmt: skip
    """Benchmark: random days of the same stock, bought at the next open with a stop `risk_pct` below
    and the same exits (no pattern target: 3R / 2R)."""
    o, h, lo, c = cols["open"], cols["high"], cols["low"], cols["close"]
    n = len(c)
    if n < 300:
        return []
    out = []
    for t in rng.integers(252, n - 2, size=per_stock):
        j, px = int(t) + 1, o[int(t) + 1]
        if not np.isfinite(px) or px <= 0:
            continue
        stop = px * (1 - risk_pct)
        R = px - stop
        bad = cols["bad"][max(0, int(t) - 252) : min(n, j + 121)].any()
        cost_r = float(cost_of(np.array([cols["raw_close"][int(t)]]))[0]) * px / R
        row: dict[str, Any] = {"setup": "random", "t": int(t), "j": j, "entry": px, "stop": stop, "risk_pct": risk_pct, "bad": bool(bad),
                               "capped": False, **{k: (v[int(t)].item() if hasattr(v[int(t)], "item") else v[int(t)]) for k, v in attrs.items()}}  # fmt: skip
        for kind in EXITS:
            r, held, why = manage(kind, j, px, stop, None, o, h, lo, c, cols["ma20"], cols["ma50"])
            row[f"r_{kind}"] = r - cost_r
            row[f"d_{kind}"] = held
            row[f"x_{kind}"] = why
        out.append(row)
    return out
