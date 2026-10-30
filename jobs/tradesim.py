"""Method B / C: exit rules per pattern, then the whole library as a portfolio (docs/VALIDATION_PLAN.md §5-6).

Usage: python -m jobs.tradesim --data-dir DIR [--run ID] [--no-upload] [--only P01,P03]
DIR is the event-study working data (part_*.parquet, day.parquet, starts.json).

1) signals of every pattern in config/research/arsenal_v1.yaml (onset events, all years)
2) every stop x exit rule per pattern; the rule with the most R per 100 days of capital in 2022-2024
   (at least 30 trades, positive expectancy) is chosen, then reported for 2025- and 2010-2021 untouched
3) portfolio: all chosen trades together with limited slots and cash, several risk / slot settings,
   with and without the market-wide timing overlay; 2022-2024, 2025-, 2010-2021 separately
Results go to R2 research/tradesim/<run>/ only (the log prints progress).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from mw.core.config import REPO_ROOT, load_settings
from mw.core.logging import setup_logging
from mw.core.storage import open_storage
from mw.report import dashboard as D
from mw.research import events, tradesim

PERIODS = {"search": (date(2022, 1, 1), date(2024, 12, 31)), "test": (date(2025, 1, 1), date(2099, 1, 1)),
           "old": (date(2010, 1, 1), date(2021, 12, 31)), "dev": (date(2010, 1, 1), date(2024, 12, 31))}  # fmt: skip
# exit rules depend less on the market regime than entries, so they are chosen on the long history
# (2010-2024, trades ending inside it) to avoid picking the luckiest of ~200 rules on 3 years
MIN_T = 2.0
MIN_TRADES = 30


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


def _period_mask(panel: tradesim.Panel, rows: np.ndarray, name: str) -> np.ndarray:
    a, b = PERIODS[name]
    d = panel.date[rows]
    m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run", default=datetime.now(UTC).strftime("%Y%m%dT%H%M"))
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--only", default=None)
    args = p.parse_args()
    setup_logging()
    storage = open_storage(load_settings().storage)
    data = Path(args.data_dir)
    parts = sorted(data.glob("part_*.parquet"))
    day = pl.read_parquet(data / "day.parquet")
    starts_raw = (
        json.loads((data / "starts.json").read_text(encoding="utf-8"))
        if (data / "starts.json").exists()
        else {}
    )
    starts = {c: date.fromisoformat(v) for c, v in starts_raw.items() if v}
    cfg = yaml.safe_load(
        (REPO_ROOT / "config/research/arsenal_v1.yaml").read_text(encoding="utf-8")
    )
    pats = cfg["stock"]
    if args.only:
        keep = set(args.only.split(","))
        pats = [x for x in pats if x["key"] in keep]

    t0 = time.time()
    log("価格の表を作成中")
    prices = D._read_years(storage, "prices_daily", "US").select(
        "entity_id", "as_of", "open", "high", "low", "close", "splits"
    )
    ents = (
        pl.scan_parquet([str(x) for x in parts]).select("entity_id").unique().collect()["entity_id"]
    )
    prices = prices.filter(
        (pl.col("as_of") >= date(2009, 1, 1)) & pl.col("entity_id").is_in(ents.implode())
    )
    panel = tradesim.build_panel(prices)
    del prices
    gc.collect()
    log(f"価格の表：{len(panel.c):,} 行 {time.time() - t0:.0f}秒")

    out: dict[str, Any] = {"run": args.run, "exits": [[k, v, tradesim.exit_label(k, v)] for k, v in tradesim.EXITS],
                           "stops": tradesim.STOP_KINDS, "periods": {k: [str(a), str(b)] for k, (a, b) in PERIODS.items()},
                           "patterns": {}, "timing": {}}  # fmt: skip
    trades_all: list[pl.DataFrame] = []
    timing_states: dict[str, dict[Any, bool]] = {}
    for uni in ("all", "sp500"):
        todo = [x for x in pats if x["universe"] == uni]
        need_timing = uni == "all"
        if not todo and not need_timing:
            continue
        log(f"対象 {uni}：条件を読み込み中")
        study = events.Study(parts, day, uni, starts=starts, log=log, outcomes=False)
        if need_timing:
            for tm in cfg.get("timing", []):
                if tm["conds"][0] in study.day_cond:
                    timing_states[tm["key"]] = study.state_by_date(tm["conds"][0])
        for pat in todo:
            t1 = time.time()
            missing = [
                c
                for c in pat["conds"]
                if c.lstrip("!") not in study.packed and c.lstrip("!") not in study.day_cond
            ]
            if missing:
                log(f"  {pat['key']}：条件が見つからないため見送り {missing}")
                out["patterns"][pat["key"]] = {
                    "name": pat["name"],
                    "skipped": f"条件が見つからない: {missing}",
                }
                continue
            ev = study.events_frame(tuple(pat["conds"]))
            sig = tradesim.signal_rows(panel, ev)
            res: dict[str, Any] = {
                "name": pat["name"],
                "type": pat["type"],
                "universe": uni,
                "conds": pat["conds"],
                "signals": int(len(sig)),
                "grid": [],
            }
            best = None
            masks = {k: _period_mask(panel, sig + 1, k) for k in PERIODS}
            for stop in tradesim.STOP_KINDS:
                for kind, prm in tradesim.EXITS:
                    sim = tradesim.simulate(panel, sig, stop, kind, prm)
                    # windows used for choosing must not see later prices: their trades also END inside them
                    end24 = panel.date[sim["exit_row"]] <= np.datetime64(PERIODS["search"][1])
                    ms = {**masks, "search": masks["search"] & end24, "dev": masks["dev"] & end24}
                    ed = panel.date[sim["entry_row"]]
                    row = {
                        "stop": stop,
                        "exit": kind,
                        "p": prm,
                        **{k: tradesim.summarize(sim, m, ed) for k, m in ms.items()},
                    }
                    res["grid"].append(row)
                    srch, dv = row["search"], row["dev"]
                    ok = (dv.get("n", 0) >= 3 * MIN_TRADES and dv["exp"] > 0 and dv["t"] >= MIN_T
                          and srch.get("n", 0) >= MIN_TRADES and srch["exp"] > 0)  # fmt: skip
                    if ok and (best is None or dv["r_per_100d"] > best[0]["dev"]["r_per_100d"]):
                        best = (row, sim)
            if best is not None:
                row, sim = best
                res["best"] = {
                    k: row[k] for k in ("stop", "exit", "p", "dev", "search", "test", "old")
                }
                res["best"]["label"] = tradesim.exit_label(row["exit"], row["p"])
                v = sim["valid"] & np.isfinite(sim["r"])
                trades_all.append(pl.DataFrame({
                    "pattern": pat["key"], "entity": panel.ent[sig[v]].astype(str), "entry_row": sim["entry_row"][v], "exit_row": sim["exit_row"][v],
                    "entry_date": pl.Series(panel.date[sim["entry_row"][v]]).cast(pl.Date), "r": sim["r"][v], "rp": sim["rp"][v],
                    "priority": float(row["dev"]["r_per_100d"]),
                }))  # fmt: skip
            # keep the grid small in the output: the top 15 rules by 2010-2024 R per 100 days
            res["grid"] = sorted(
                [g for g in res["grid"] if g["dev"].get("n", 0) >= MIN_TRADES],
                key=lambda g: -g["dev"]["r_per_100d"],
            )[:15]
            out["patterns"][pat["key"]] = res
            b = res.get("best", {})
            log(
                f"  {pat['key']}：シグナル {len(sig):,} 件・最良 {b.get('stop')}/{b.get('exit')}{b.get('p', '')} {time.time() - t1:.0f}秒"
            )
        del study
        gc.collect()

    # ---------------------------------------------------------------- portfolio
    if trades_all:
        trades = pl.concat(trades_all)
        n_pat = trades["pattern"].n_unique()
        sets = {"全部": None}
        types = {x["key"]: x["type"] for x in pats}
        sets["モメンタム型だけ"] = [k for k, t in types.items() if t == "momentum"]
        sets["モメンタム以外"] = [k for k, t in types.items() if t != "momentum"]
        good = [
            k
            for k, r in out["patterns"].items()
            if r.get("best", {}).get("dev", {}).get("r_per_100d", 0) > 0
        ]
        sets["2010〜2024年の上位8つ"] = sorted(
            good, key=lambda k: -out["patterns"][k]["best"]["dev"]["r_per_100d"]
        )[:8]
        overlay_any = {}
        if timing_states:
            all_days = set().union(*[set(v) for v in timing_states.values()])
            for d in all_days:
                on = any(v.get(d, False) for v in timing_states.values())
                overlay_any[d] = 1.0 if on else 0.5
        port = []
        for set_name, keys in sets.items():
            tt = trades if keys is None else trades.filter(pl.col("pattern").is_in(keys))
            if tt.height == 0:
                continue
            for risk in (0.005, 0.01, 0.02):
                for max_pos in (10, 20):
                    for ov in (False, True):
                        if ov and not overlay_any:
                            continue
                        for per, (a, b) in PERIODS.items():
                            r = tradesim.portfolio(
                                panel,
                                tt,
                                a,
                                b,
                                risk=risk,
                                max_pos=max_pos,
                                scale=overlay_any if ov else None,
                            )
                            r.pop("curve", None) if not (
                                set_name == "全部" and risk == 0.01 and max_pos == 10
                            ) else None
                            port.append(
                                {
                                    "set": set_name,
                                    "risk": risk,
                                    "max_pos": max_pos,
                                    "overlay": ov,
                                    "period": per,
                                    **r,
                                }
                            )
            log(f"  資金管理：{set_name} 完了")
        out["portfolio"] = port
        out["n_patterns_traded"] = n_pat
    body = json.dumps(_clean(out), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
    (data / "tradesim.json").write_bytes(body)
    if not args.no_upload:
        storage.put_bytes(f"research/tradesim/{args.run}/tradesim.json", body)
    log(f"保存：tradesim.json（{len(body) / 1e6:.1f}MB）・合計 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
