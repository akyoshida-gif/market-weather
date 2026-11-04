"""Setup validation (VALIDATION_PLAN_V3): textbook setups x market phase x stage, traded like a real trader.

Usage: python -m jobs.setup_study [--limit N] [--out-dir DIR] [--run ID] [--no-upload]
Writes research/setups/<run>/trades.parquet (every trade with its context) and summary.json:
  - by setup x market phase x stage x exit rule: trades, average R, win rate, share reaching +3R, days held
  - the same for random entries in the same phase and stage (does the pattern add anything?)
  - walk-forward: each year, use only the (setup, phase) cells that worked in all earlier years, then trade
    that year; compared with random entries in those years
The log shows progress only (public repository).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mw.core.config import load_settings
from mw.core.logging import setup_logging
from mw.core.storage import open_storage
from mw.report import dashboard as D
from mw.research import setup_sim, setups

C = pl.col
MIN_DVOL = 2e6
MIN_PRICE = 3.0
RANDOM_PER_STOCK = 40


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    return o


def summarize(tr: pl.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for kind in setup_sim.EXITS:
        r, d = C(f"r_{kind}"), C(f"d_{kind}")
        g = tr.group_by(keys).agg(
            pl.len().alias("n"), r.mean().alias("avg_r"), r.median().alias("median_r"), (r > 0).mean().alias("win"),
            (r >= 3).mean().alias("big"), d.mean().alias("days"), (r.sum() / d.sum() * 100).alias("r_per_100d"),
            C("as_of").dt.year().n_unique().alias("years"),
        )  # fmt: skip
        out += [{**row, "exit": kind} for row in g.iter_rows(named=True)]
    return out


def walk_forward(
    tr: pl.DataFrame, rnd: pl.DataFrame, kind: str, first_test: int = 2015, min_n: int = 30
) -> list[dict[str, Any]]:
    """Each year Y: cells (setup, phase) with >= min_n trades and positive average R in years < Y are used in Y."""
    r = f"r_{kind}"
    t = tr.with_columns(C("as_of").dt.year().alias("y"))
    q = rnd.with_columns(C("as_of").dt.year().alias("y"))
    last = int(t["y"].max())
    out = []
    for y in range(first_test, last + 1):
        hist = (
            t.filter(C("y") < y)
            .group_by("setup", "phase")
            .agg(pl.len().alias("n"), C(r).mean().alias("m"))
        )
        chosen = hist.filter((C("n") >= min_n) & (C("m") > 0)).select("setup", "phase")
        cur = t.filter(C("y") == y).join(chosen, on=["setup", "phase"], how="inner")
        base = q.filter(C("y") == y).join(chosen.select("phase").unique(), on="phase", how="inner")
        out.append({"year": y, "cells": chosen.height, "trades": cur.height, "avg_r": cur[r].mean() if cur.height else None,
                    "win": (cur[r] > 0).mean() if cur.height else None, "random_avg_r": base[r].mean() if base.height else None,
                    "chosen": [f"{a}/{b}" for a, b in chosen.iter_rows()]})  # fmt: skip
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only N stocks (trial)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run", default=datetime.now(UTC).strftime("%Y%m%dT%H%M"))
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    t0 = time.time()
    px = D._read_years(storage, "prices_daily", "US").select(
        "entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"
    )
    vol = D._read_years(storage, "vol_indices", "US")
    vix = (
        vol.filter(C("entity_id") == "IDX:US:VIX").select("as_of", "close") if vol.height else None
    )
    phase = setups.market_phase(px.filter(C("entity_id") == "US:SPY").select("as_of", "close"), vix)
    keys = [
        k for k in storage.list("silver/universe_membership/market=US/") if k.endswith(".parquet")
    ]
    etfs = set(
        storage.read_parquet(sorted(keys)[-1]).filter(C("kind") == "etf")["entity_id"].to_list()
    )
    px = px.filter(~C("entity_id").is_in(list(etfs)))
    ents = sorted(px["entity_id"].unique().to_list())
    if args.limit:
        rng0 = np.random.default_rng(0)
        ents = sorted(rng0.choice(ents, args.limit, replace=False).tolist())
        px = px.filter(C("entity_id").is_in(ents))
    log(f"価格 {px.height:,} 行・{len(ents):,} 銘柄")
    # cross-sectional strength rank (6-month return percentile among liquid stocks that day) and weekly stage
    base = px.sort(["entity_id", "as_of"]).with_columns(
        (C("close") / C("close").shift(126).over("entity_id") - 1).alias("ret126"),
        (C("close") * C("volume")).rolling_mean(20).over("entity_id").alias("dvol20"),
    )
    rs = (
        base.filter(C("dvol20") >= MIN_DVOL)
        .with_columns((C("ret126").rank() / C("ret126").count()).over("as_of").alias("rs_rank"))
        .select("entity_id", "as_of", "rs_rank")
    )
    stages = setups.weinstein_stage(base.select("entity_id", "as_of", "close"))
    base = setups.attach_stage(base.drop("ret126", "dvol20"), stages)
    base = base.join(rs, on=["entity_id", "as_of"], how="left").join(phase, on="as_of", how="left")
    log(f"指標の準備 {time.time() - t0:.0f}秒")
    rng = np.random.default_rng(1)
    all_tr, all_rnd = [], []
    done = 0
    for part in base.partition_by("entity_id", maintain_order=False):
        g = setups.daily_indicators(part)
        if g.height < 300:
            continue
        jump = ((C("close") / C("pc") - 1).abs() > 0.6) | (C("splits").fill_null(0) != 0)
        g = g.with_columns(jump.fill_null(False).alias("bad"), setups.trend_template().alias("tt"),
                           (C("close") / C("hh252") - 1).alias("dist_hi"))  # fmt: skip
        cols = {k: g[k].to_numpy() for k in ("open", "high", "low", "close", "ma20", "ma50", "bad")}
        cols["raw_close"] = cols[
            "close"
        ]  # costs on the adjusted price (conservative for split-heavy names)
        attrs = {"phase": g["phase"].fill_null(0).to_numpy(), "stage": g["stage"].fill_null(0).to_numpy(), "tt": g["tt"].to_numpy(),
                 "rs_rank": g["rs_rank"].to_numpy(), "dist_hi": g["dist_hi"].to_numpy(), "dvol20": g["dvol20"].to_numpy(), "close_t": g["close"].to_numpy()}  # fmt: skip
        armed = [
            a
            for a in setups.detect(g)
            if attrs["dvol20"][a.t] >= MIN_DVOL and attrs["close_t"][a.t] >= MIN_PRICE
        ]
        stock_tr = setup_sim.trades_for_stock(armed, cols, attrs)
        stock_rnd = setup_sim.random_trades(cols, attrs, rng, RANDOM_PER_STOCK, 0.06)
        dates = g["as_of"].to_list()
        ent = g["entity_id"][0]
        for row in stock_tr + stock_rnd:
            row["entity_id"], row["as_of"], row["entry_date"] = (
                ent,
                dates[row["t"]],
                dates[row["j"]],
            )
        all_tr += stock_tr
        all_rnd += [
            r
            for r in stock_rnd
            if r["dvol20"] is not None and r["dvol20"] >= MIN_DVOL and r["close_t"] >= MIN_PRICE
        ]
        done += 1
        if done % 500 == 0:
            log(f"  {done:,} 銘柄・取引 {len(all_tr):,}")
    tr = (
        pl.DataFrame(all_tr, infer_schema_length=None)
        .filter(~C("bad"))
        .with_columns(C("as_of").cast(pl.Date), C("entry_date").cast(pl.Date))
    )
    rnd = (
        pl.DataFrame(all_rnd, infer_schema_length=None)
        .filter(~C("bad"))
        .with_columns(C("as_of").cast(pl.Date), C("entry_date").cast(pl.Date))
    )
    tr = tr.with_columns(
        pl.when(C("stage") == 2).then(pl.lit("2")).otherwise(pl.lit("他")).alias("stage2")
    )
    rnd = rnd.with_columns(
        pl.when(C("stage") == 2).then(pl.lit("2")).otherwise(pl.lit("他")).alias("stage2")
    )
    log(f"取引 {tr.height:,}・ランダム {rnd.height:,} {time.time() - t0:.0f}秒")
    summary = {
        "run": args.run, "stocks": done, "trades": tr.height, "random": rnd.height,
        "setups": {k: {"name": v[0], "why": v[1]} for k, v in setups.SETUPS.items()},
        "phases": setups.PHASES, "exits": setup_sim.EXITS,
        "by_setup": summarize(tr, ["setup"]),
        "by_setup_phase": summarize(tr, ["setup", "phase"]),
        "by_setup_phase_stage": summarize(tr, ["setup", "phase", "stage2"]),
        "by_setup_tt": summarize(tr, ["setup", "tt"]),
        "random_by_phase": summarize(rnd, ["phase"]),
        "random_by_phase_stage": summarize(rnd, ["phase", "stage2"]),
        "by_setup_year": summarize(tr.with_columns(C("as_of").dt.year().alias("year")), ["setup", "year"]),
        "walk_forward": {k: walk_forward(tr, rnd, k) for k in setup_sim.EXITS},
    }  # fmt: skip
    out = Path(args.out_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_clean(summary), ensure_ascii=False, allow_nan=False, default=str).encode(
        "utf-8"
    )
    (out / "summary.json").write_bytes(body)
    tr.write_parquet(out / "trades.parquet")
    if not args.no_upload:
        storage.put_bytes(f"research/setups/{args.run}/summary.json", body)
        storage.write_parquet(f"research/setups/{args.run}/trades.parquet", tr)
    log(f"保存 {len(body) / 1e6:.1f}MB・合計 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
