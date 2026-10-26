"""Paper (demo) trading: forward-only, one virtual book per strategy (DESIGN §13 shadow evidence).

Daily, after the market's prices are stored, for each session not yet processed:
  1. fill pending orders at the session open (long: skipped if the open is at/below the stop)
  2. manage open positions with the session bar (stop first, then target; gap fills at the open;
     close-based exit rule; time stop at the close of the last allowed session)
  3. create orders from signals at the session close (IN/LS/TP fixed now, filled next open)
  4. mark to market
Sizing: risk 1% of book equity per trade (shares = risk / (ref - stop)), capped at 10% of equity
per position and 10 open positions per book. Costs as in the backtest engine.
State lives in storage: paper/{market}/state.json (mutable) + append-only daily event logs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import polars as pl

from mw.core.storage import GuardedStorage
from mw.strategy import engine

START_EQUITY = {"US": 100_000.0, "JP": 10_000_000.0}
RISK_PER_TRADE = 0.01
MAX_POS_FRAC = 0.10
MAX_OPEN = 10


def state_key(market: str) -> str:
    return f"paper/{market}/state.json"


def events_key(market: str, d: date) -> str:
    return f"paper/{market}/events/date={d.isoformat()}.parquet"


@dataclass
class Book:
    key: str
    cash: float
    positions: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    closed: int = 0

    def equity(self, px: dict[str, float]) -> float:
        val = 0.0
        for p in self.positions:
            last = px.get(p["entity_id"], p["entry"])
            val += p["shares"] * last
        return self.cash + val


def load_state(storage: GuardedStorage, market: str, keys: list[str]) -> dict[str, Any]:
    try:
        st: dict[str, Any] = json.loads(storage.get_bytes(state_key(market)))
    except FileNotFoundError:
        st = {"market": market, "last_session": None, "books": {}, "started": None}
    for k in keys:
        st["books"].setdefault(
            k, {"cash": START_EQUITY[market], "positions": [], "orders": [], "closed": 0}
        )
    return st


def _num(v: Any) -> float:
    return (
        float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")
    )


def process_session(
    st: dict[str, Any],
    strategies: dict[str, engine.Strategy],
    ind: pl.DataFrame,
    d: date,
    market: str,
) -> list[dict[str, Any]]:
    """Advance all books by one session `d`. Returns event rows."""
    day = ind.filter(pl.col("as_of") == d)
    bar = {r["entity_id"]: r for r in day.iter_rows(named=True)}
    events: list[dict[str, Any]] = []
    for key, b in st["books"].items():
        s = strategies.get(key)
        # 1) fill pending orders at the open
        keep_orders = []
        for o in b["orders"]:
            r = bar.get(o["entity_id"])
            if r is None:
                keep_orders.append(
                    o
                )  # no bar today (halt/holiday for that name): keep one more day
                continue
            px = r["open"]
            if px <= o["stop"] or (o["target"] is not None and px >= o["target"]):
                events.append(
                    {
                        **o,
                        "event": "cancel",
                        "date": d,
                        "price": px,
                        "note": "寄り付きが損切り/利確水準を越えたため見送り",
                    }
                )
                continue
            cost = engine.cost_bps(px, market) / 1e4 / 2
            shares = o["shares"]
            b["cash"] -= shares * px * (1 + cost)
            pos = {**o, "entry": px, "entry_date": d.isoformat(), "held": 0}
            b["positions"].append(pos)
            events.append({**pos, "event": "fill", "date": d, "price": px})
        b["orders"] = [o for o in keep_orders if o.get("age", 0) < 1]
        for o in b["orders"]:
            o["age"] = o.get("age", 0) + 1
        # 2) manage open positions
        still = []
        for p in b["positions"]:
            r = bar.get(p["entity_id"])
            if r is None:
                still.append(p)
                continue
            p["held"] += 1
            o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
            exit_px, reason = None, None
            first = p["entry_date"] == d.isoformat()
            if lo <= p["stop"]:
                exit_px, reason = (o if (o < p["stop"] and not first) else p["stop"]), "stop"
            elif p["target"] is not None and h >= p["target"]:
                exit_px, reason = (o if (o > p["target"] and not first) else p["target"]), "target"
            elif s is not None and s.exit_rule is not None and not first and bool(r.get("_exit")):
                exit_px, reason = c, "rule"
            elif p["held"] >= p["max_hold"]:
                exit_px, reason = c, "time"
            if exit_px is None:
                still.append(p)
                continue
            cost = engine.cost_bps(exit_px, market) / 1e4 / 2
            b["cash"] += p["shares"] * exit_px * (1 - cost)
            ret = exit_px / p["entry"] - 1 - engine.cost_bps(p["entry"], market) / 1e4
            b["closed"] += 1
            events.append(
                {**p, "event": "exit", "date": d, "price": exit_px, "reason": reason, "ret": ret}
            )
        b["positions"] = still
        # 3) new orders from today's signals
        if s is None:
            continue
        held = {p["entity_id"] for p in b["positions"]} | {o["entity_id"] for o in b["orders"]}
        slots = MAX_OPEN - len(b["positions"]) - len(b["orders"])
        if slots <= 0:
            continue
        px_map = {k: v["close"] for k, v in bar.items()}
        eq = Book(key, b["cash"], b["positions"], b["orders"]).equity(px_map)
        cands = day.filter(pl.col(f"_sig_{key}")).sort(f"_rank_{key}", descending=True)
        for r in cands.iter_rows(named=True):
            if slots <= 0:
                break
            if r["entity_id"] in held:
                continue
            ref, stop, tgt = r["close"], _num(r[f"_stop_{key}"]), _num(r[f"_tgt_{key}"])
            if not np.isfinite(stop) or stop >= ref:
                continue
            per_share = ref - stop
            shares = math.floor(min(eq * RISK_PER_TRADE / per_share, eq * MAX_POS_FRAC / ref))
            if market == "JP":
                shares = (shares // 100) * 100  # 単元株
            if shares <= 0:
                continue
            order = {
                "book": key, "entity_id": r["entity_id"], "signal_date": d.isoformat(), "ref": ref, "stop": stop,
                "target": tgt if np.isfinite(tgt) else None, "shares": shares, "max_hold": s.max_hold,
                "vr": r.get("vr"), "volume": r.get("volume"), "age": 0,
            }  # fmt: skip
            b["orders"].append(order)
            held.add(r["entity_id"])
            slots -= 1
            events.append({**order, "event": "order", "date": d, "price": ref})
        b["equity"] = Book(key, b["cash"], b["positions"], b["orders"]).equity(px_map)
    st["last_session"] = d.isoformat()
    st["started"] = st.get("started") or d.isoformat()
    return events


def annotate(ind: pl.DataFrame, strategies: dict[str, engine.Strategy]) -> pl.DataFrame:
    """Add per-strategy signal/stop/target/rank/exit columns."""
    cols = []
    for k, s in strategies.items():
        cols += [
            s.entry().fill_null(False).alias(f"_sig_{k}"),
            s.stop().alias(f"_stop_{k}"),
            s.target().alias(f"_tgt_{k}"),
            (s.rank() if s.rank else pl.lit(0.0)).alias(f"_rank_{k}"),
        ]
    out = ind.with_columns(cols)
    # one shared exit flag per row is not enough (rules differ); store per strategy and pick at use
    rules = [
        (s.exit_rule() if s.exit_rule else pl.lit(False)).fill_null(False).alias(f"_exit_{k}")
        for k, s in strategies.items()
    ]
    out = out.with_columns(rules)
    # sanitize recent bad data: no new signals within the lookback after a split / bad print
    badn = (
        pl.col("bad")
        .cast(pl.Int8)
        .rolling_sum(engine.BAD_LOOKBACK + 1)
        .over("entity_id")
        .fill_null(0)
    )
    return out.with_columns(
        [(pl.col(f"_sig_{k}") & (badn == 0)).alias(f"_sig_{k}") for k in strategies]
    )


def run(
    storage: GuardedStorage, market: str, strategies: dict[str, engine.Strategy], ind: pl.DataFrame,
    start: date | None = None,
) -> dict[str, Any]:  # fmt: skip
    st = load_state(storage, market, list(strategies))
    ann = annotate(ind, strategies)
    sessions = sorted(set(ann["as_of"].to_list()))
    last = date.fromisoformat(st["last_session"]) if st["last_session"] else None
    todo = [d for d in sessions if (last is None or d > last) and (start is None or d >= start)]
    if last is None and start is None:
        todo = todo[-1:]  # forward-only: first run starts with the latest session (signals only)
    all_events: list[dict[str, Any]] = []
    for d in todo:
        evs = _process_with_exit_cols(st, strategies, ann, d, market)
        st["last_session"] = d.isoformat()
        st["started"] = st.get("started") or d.isoformat()
        if evs:
            df = pl.DataFrame(
                [
                    {k: (str(v) if isinstance(v, (date, datetime)) else v) for k, v in e.items()}
                    for e in evs
                ],
                infer_schema_length=None,
            )
            storage.put_bytes(events_key(market, d), _parquet(df), overwrite=True)
        all_events += evs
    st["updated_at"] = datetime.now(UTC).isoformat()
    storage.put_bytes(
        state_key(market), json.dumps(st, ensure_ascii=False, default=str).encode(), overwrite=True
    )
    return {"sessions": [d.isoformat() for d in todo], "events": len(all_events),
            "books": {k: {"equity": round(v.get("equity", v["cash"]), 2), "open": len(v["positions"]), "orders": len(v["orders"]), "closed": v["closed"]} for k, v in st["books"].items()}}  # fmt: skip


def _process_with_exit_cols(
    st: dict[str, Any],
    strategies: dict[str, engine.Strategy],
    ann: pl.DataFrame,
    d: date,
    market: str,
) -> list[dict[str, Any]]:
    evs: list[dict[str, Any]] = []
    for key in strategies:
        sub = {"market": st["market"], "books": {key: st["books"][key]}}
        frame = ann.filter(pl.col("as_of") == d).with_columns(pl.col(f"_exit_{key}").alias("_exit"))
        evs += process_session(sub, {key: strategies[key]}, frame, d, market)
    return evs


def _parquet(df: pl.DataFrame) -> bytes:
    import io

    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()
