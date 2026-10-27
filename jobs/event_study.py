"""Method A (event study) for US stocks, VALIDATION_PLAN §3-§6.

Usage:
  python -m jobs.event_study build   --out-dir DIR [--chunks 8] [--only N]
  python -m jobs.event_study analyze --data-dir DIR --universe all|sp500 --window main|recent --task TASK [--run ID]
  python -m jobs.event_study macro   --data-dir DIR [--run ID]
TASK: singles_stock | singles_mkt | s5 | s20 | s60 | tree | ml
window main  : search 2022-01..2024-12 (judged mainly there), confirm 2025-01..now, older years = lifespan
window recent: search 2025-01..now only (owner request 2026-10-27), no confirmation period -> at most "観察"
Results go to R2 research/event_study/<run>/; the log prints progress only (the repository is public).
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

import polars as pl

from mw.core.config import load_settings
from mw.core.logging import setup_logging
from mw.core.storage import open_storage
from mw.research import build, events, external, features, macro, rules, technical

SEARCH_H = {"s5": 5, "s20": 20, "s60": 60}
WINDOWS = {
    "main": ((date(2022, 1, 1), date(2024, 12, 31)), date(2025, 1, 1)),
    "recent": ((date(2025, 1, 1), date(2099, 12, 31)), None),
}
REGIME_KEYS = ["m_SPY_dev200__hi20", "m_SPY_dev200__lo20", "m_VIX_lvl__hi20", "m_VIX_lvl__lo20", "m_DGS10_chg60__up", "m_DGS10_chg60__dn",
               "m_DCOILWTICO_pos__hi20", "m_DCOILWTICO_pos__lo20", "m_BAA10Y_chg20__up", "m_BAA10Y_chg20__dn", "m_IWM_SPY_chg20__up",
               "m_IWM_SPY_chg20__dn", "m_BREADTH_lvl__hi20", "m_BREADTH_lvl__lo20", "m_T10Y2Y_lvl__lo20", "m_DTWEXBGS_chg20__up"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)


def labels(data_dir: Path) -> dict[str, str]:
    out = {f"c_{k}": v for k, v in technical.COND_LABELS.items()}
    out |= {f"c_{k}": v for k, v in external.EXT_LABELS.items()}
    out |= {f"c_{k}": v for k, v in build.MKT_LABELS.items()}
    out |= features.FEATURE_LABELS
    p = data_dir / "day_labels.json"
    if p.exists():
        out |= json.loads(p.read_text(encoding="utf-8"))
    return out


def _clean(o: Any) -> Any:
    """NaN / infinity -> None so the files are strict JSON (browsers reject NaN)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    return o


def _dump(obj: Any) -> bytes:
    return json.dumps(_clean(obj), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def save(storage: Any, data_dir: Path, run: str, name: str, obj: Any, upload: bool) -> None:
    body = _dump(obj)
    (data_dir / name).write_bytes(body)
    if upload:
        storage.put_bytes(f"research/event_study/{run}/{name}", body)
    log(f"保存：{name}（{len(body) / 1e6:.1f}MB）")


# ---------------------------------------------------------------- build
def cmd_build(args: argparse.Namespace) -> int:
    storage = open_storage(load_settings().storage)
    out = Path(args.out_dir)
    log("作業データを作成中")
    parts = build.build_us(storage, out, chunks=args.chunks, only=args.only)
    build.finalize(parts)
    starts = {c: (str(v) if v else None) for c, v in external.STARTS.items()}
    (out / "starts.json").write_text(json.dumps(starts, ensure_ascii=False), encoding="utf-8")
    rows = pl.scan_parquet([str(p) for p in parts]).select(pl.len()).collect().item()
    log(f"作業データ完成：{rows:,} 行・{len(parts)} ファイル")
    return 0


# ---------------------------------------------------------------- analyze
def _profile_many(
    study: events.Study, sets: dict[tuple[str, ...], set[str]], full: bool = True
) -> list[dict[str, Any]]:
    out = []
    for k, found_by in sets.items():
        p = study.profile(k, full=full)
        p["found_by"] = sorted(found_by)
        out.append(p)
    return out


def _rank_key(p: dict[str, Any], h: int) -> float:
    s = p.get(str(h), {})
    r = s.get("R", {}).get("search", {})
    return float(r.get("exp", -9)) if r.get("n", 0) >= events.MIN_DEV else -9.0


def cmd_analyze(args: argparse.Namespace) -> int:
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
    window, test_start = WINDOWS[args.window]
    name = f"{args.universe}_{args.window}_{args.task}"
    t0 = time.time()
    if args.task == "ml":
        res: dict[str, Any] = {
            "task": args.task,
            "universe": args.universe,
            "window": args.window,
            "models": [],
        }
        for h in (5, 20, 60):
            for tr in (
                (date(2022, 1, 1), date(2024, 12, 31)),
                (date(2010, 1, 1), date(2024, 12, 31)),
            ):
                log(f"  機械学習 {h}日・学習 {tr[0].year}〜{tr[1].year}")
                res["models"].append(
                    rules.ml_scan(parts, day, args.universe, h, tr, date(2025, 1, 1), log=log)
                )
                gc.collect()
        save(storage, data, args.run, f"{name}.json", res, not args.no_upload)
        return 0
    log(f"対象 {args.universe}・期間 {args.window}・作業 {args.task}：データを読み込み中")
    study = events.Study(parts, day, args.universe, window, test_start, starts, log=log)
    conds = study.all_conditions()
    stock = [c for c in study.c_cols + study.fbin_cols if not c.startswith("c_mkt_")]
    mkt = study.mbin_cols + [c for c in study.c_cols if c.startswith("c_mkt_")]
    res = {"task": args.task, "universe": args.universe, "window": args.window, "rows": study.n,
           "search_window": [str(window[0]), str(window[1])], "test_start": str(test_start) if test_start else None,
           "cuts": study.cuts, "n_conditions": len(conds)}  # fmt: skip
    full = args.window == "main"
    if args.task in ("singles_stock", "singles_mkt"):
        # market-wide conditions fire for every stock on their day: many events, so a lighter profile
        todo = stock if args.task == "singles_stock" else mkt
        light = args.task == "singles_mkt"
        res["singles"] = []
        res["tried"] = len(todo)
        for i, c in enumerate(todo):
            res["singles"].append(study.profile((c,), full=full and not light))
            if (i + 1) % 200 == 0:
                log(f"  単独条件 {i + 1}/{len(todo)} {time.time() - t0:.0f}秒")
                res["partial"] = True
                save(storage, data, args.run, f"{name}.json", res, not args.no_upload)
        res["partial"] = False
        log(f"  単独条件 {len(todo)} 種類の集計 {time.time() - t0:.0f}秒")
    elif args.task in SEARCH_H:
        h = SEARCH_H[args.task]
        sw, tried = events.stepwise(study, stock, mkt, h, log=log)
        log(
            f"  段階的探索（{h}日）：見つかった組み合わせ {len(sw)}・試した数 {tried:,} {time.time() - t0:.0f}秒"
        )
        top = {tuple(r["conds"]): {f"段階的探索・{h}日"} for r in sw[:120]}
        res["tried"] = tried
        res["found"] = [
            {"conds": list(r["conds"]), "n": r["n"], "x": r["x"], "t": r["t"]} for r in sw[:500]
        ]
        res["profiles"] = _profile_many(study, top, full=full)
        best = sorted(res["profiles"], key=lambda p: _rank_key(p, h), reverse=True)[:25]
        res["regimes"] = [
            {"conds": p["conds"], **events.regimes(study, tuple(p["conds"]), h, REGIME_KEYS)}
            for p in best
        ]
        res["strong_days"] = [
            {"conds": p["conds"], **events.strong_days(study, tuple(p["conds"]), h)} for p in best
        ]
    elif args.task == "tree":
        sets: dict[tuple[str, ...], set[str]] = {}
        tried = 0
        # conditions whose data starts after the search window opens would silently shorten the training period
        tree_cols = [c for c in conds if study.start_rank.get(c, 0) <= study.win_lo]
        res["tree_conditions"] = len(tree_cols)
        for h in (5, 20, 60):
            found, leaves = rules.tree_rules(study, tree_cols, h, max_rows=500_000)
            tried += leaves
            for r in found:
                sets.setdefault(tuple(r["conds"]), set()).add(f"決定木・{h}日")
            log(f"  決定木（{h}日）：ルール {len(found)} {time.time() - t0:.0f}秒")
        res["tried"] = tried
        res["profiles"] = _profile_many(study, sets, full=full)
    log(f"  完了 {time.time() - t0:.0f}秒")
    save(storage, data, args.run, f"{name}.json", res, not args.no_upload)
    return 0


# ---------------------------------------------------------------- macro (leading indicators, sector rotation)
def cmd_macro(args: argparse.Namespace) -> int:
    storage = open_storage(load_settings().storage)
    data = Path(args.data_dir)
    day = pl.read_parquet(data / "day.parquet")
    etf = pl.read_parquet(data / "etf.parquet")
    dev_end = date(2024, 12, 31)
    targets = [
        "SPY",
        "RSP",
        "IWM",
        "QQQ",
        *features.SECTOR_ETFS,
        "SMH",
        "XBI",
        "KRE",
        "ITB",
        "XOP",
        "GDX",
        "IWO",
        "IWN",
        "MTUM",
    ]
    t0 = time.time()
    lead = macro.leading(day, etf, dev_end, targets)
    log(f"  先行指標の総当たり {len(lead):,} 通り {time.time() - t0:.0f}秒")
    good = [
        r
        for r in lead
        if r.get("dev_corr") is not None and r.get("test_corr") is not None and r["dev_n"] >= 20
    ]
    good.sort(key=lambda r: -abs(r["dev_corr"]))
    quint = [
        macro.quintiles(day, etf, r["feature"], r["target"], r["h"], r["lag"], dev_end)
        for r in good[:150]
    ]
    rot = macro.rotation(etf, dev_end)
    log(f"  セクター循環 {time.time() - t0:.0f}秒")
    body = {
        "leading_top": good[:2000],
        "n_tested": len(lead),
        "quintiles": quint,
        "rotation": rot,
        "etf_names": macro.ETF_NAMES,
    }
    save(storage, data, args.run, "macro.json", body, not args.no_upload)
    save(storage, data, args.run, "labels.json", labels(data), not args.no_upload)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--chunks", type=int, default=8)
    b.add_argument("--only", type=int, default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--data-dir", required=True)
    a.add_argument("--universe", choices=["all", "sp500"], required=True)
    a.add_argument("--window", choices=list(WINDOWS), required=True)
    a.add_argument(
        "--task", choices=["singles_stock", "singles_mkt", "tree", "ml", *SEARCH_H], required=True
    )
    a.add_argument("--run", default=datetime.now(UTC).strftime("%Y%m%d"))
    a.add_argument("--no-upload", action="store_true")
    m = sub.add_parser("macro")
    m.add_argument("--data-dir", required=True)
    m.add_argument("--run", default=datetime.now(UTC).strftime("%Y%m%d"))
    m.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    setup_logging()
    return {"build": cmd_build, "analyze": cmd_analyze, "macro": cmd_macro}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
