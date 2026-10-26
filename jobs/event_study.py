"""Method A (event study) for US stocks, VALIDATION_PLAN §3.

Usage: python -m jobs.event_study [--out-dir DIR] [--chunks 8] [--only N] [--dev-end 2021-12-31] [--no-upload]

Steps: build the dataset (all eligible US rows) -> for each universe (all / S&P 500 point-in-time):
singles, stepwise combinations and decision-tree rules for the short / swing / long horizons ->
full profile (all horizons, period blocks, volatility tiers, ex-top-5-days) and grade of every candidate ->
strong-day analysis for the best ones. Results go to R2 (research/event_study/<run>/) only; the log prints
progress counts, never results (the repository and its logs are public).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from mw.core.config import load_settings
from mw.core.logging import setup_logging
from mw.core.storage import open_storage
from mw.research import build, events, external, rules, technical

SEARCH_H = {"short": 5, "swing": 20, "long": 60}


def labels() -> dict[str, str]:
    out = {f"c_{k}": v for k, v in technical.COND_LABELS.items()}
    out |= {f"c_{k}": v for k, v in external.EXT_LABELS.items()}
    out |= {f"c_{k}": v for k, v in build.MKT_LABELS.items()}
    return out


FULL_FROM = date(2010, 6, 30)  # conditions with data from here are "full history"
SHORT_FROM = date(
    2019, 1, 1
)  # second decision tree: conditions with data from 2018 (short sale volume)


def run_universe(
    study: events.Study, conds: list[str], starts: dict[str, Any], log: Any
) -> dict[str, Any]:
    stock = [c for c in conds if not c.startswith("c_mkt_")]
    mkt = [c for c in conds if c.startswith("c_mkt_")]
    full = [c for c in conds if starts.get(c) is None or starts[c] <= FULL_FROM]
    mid = [c for c in conds if starts.get(c) is None or starts[c] <= SHORT_FROM]
    t0 = time.time()
    singles = [study.profile((c,)) for c in conds]
    log(f"    単独条件 {len(singles)} 件の集計 {time.time() - t0:.0f}秒")
    cands: dict[tuple[str, ...], set[str]] = {}
    tried = len(conds)
    importance: dict[str, Any] = {}
    for style, h in SEARCH_H.items():
        t0 = time.time()
        sw, n_sw = events.stepwise(study, stock, mkt, h)
        tried += n_sw
        for r in sw[:60]:
            cands.setdefault(tuple(r["conds"]), set()).add(f"段階的探索・{style}")
        n_tr = 0
        for tag, cols in (("全期間の条件", full), ("2018年からの条件", mid)):
            tr, leaves = rules.tree_rules(study, cols, h)
            tried += leaves
            n_tr += len(tr)
            for r in tr:
                cands.setdefault(tuple(r["conds"]), set()).add(f"決定木（{tag}）・{style}")
        importance[style] = rules.importance(study, full, h)[:30]
        log(
            f"    探索（{style}・{h}日）：段階的 {len(sw)} 件・決定木 {n_tr} 件 {time.time() - t0:.0f}秒"
        )
    t0 = time.time()
    combos = []
    for k, found_by in cands.items():
        if len(k) < 2:
            continue
        p = study.profile(k)
        p["found_by"] = sorted(found_by)
        combos.append(p)
    log(f"    組み合わせ候補 {len(combos)} 件の集計 {time.time() - t0:.0f}秒")

    # strong-day analysis for the most promising (graded at the swing horizon)
    def score(p: dict[str, Any]) -> float:
        s = p.get("20", {})
        return s.get("all", {}).get("xmean", -1) if s.get("grade") in ("有力", "最近有効") else -1

    ranked = sorted(singles + combos, key=score, reverse=True)[:30]
    strong = [
        {"conds": p["conds"], **events.strong_days(study, tuple(p["conds"]))}
        for p in ranked
        if score(p) > 0
    ]
    return {
        "tried": tried,
        "singles": singles,
        "combos": combos,
        "importance": importance,
        "strong_days": strong,
    }


def _dump(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o)
    ).encode("utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument(
        "--only", type=int, default=None, help="build only the first N chunks (trial runs)"
    )
    p.add_argument("--dev-end", default="2021-12-31")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    setup_logging()

    def log(msg: str) -> None:
        print(f"[{datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)

    storage = open_storage(load_settings().storage)
    work = Path(args.out_dir or tempfile.mkdtemp(prefix="event_study_"))
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M")

    def save(name: str, obj: Any) -> None:
        """Save each piece as soon as it is ready, so a timeout keeps what was finished."""
        body = _dump(obj)
        (work / name).write_bytes(body)
        if not args.no_upload:
            storage.put_bytes(f"research/event_study/{run}/{name}", body)
        log(f"保存：{name}（{len(body) / 1e6:.1f}MB）")

    log("作業データを作成中")
    parts = build.build_us(storage, work / "parts", chunks=args.chunks, only=args.only)
    df = build.finalize(parts)
    conds = [c for c in df.columns if c.startswith("c_")]
    starts = {c: external.STARTS.get(c) for c in conds}
    log(f"作業データ {df.height:,} 行・条件 {len(conds)} 種類")
    cov = df.select(
        [
            pl.struct(
                pl.col("as_of").filter(pl.col(c)).min().alias("first"),
                pl.col(c).sum().alias("rows"),
            ).alias(c)
            for c in conds
        ]
    ).row(0, named=True)
    meta = {
        "run": run, "rows": df.height, "first": str(df["as_of"].min()), "last": str(df["as_of"].max()),
        "dev_end": args.dev_end, "horizons": events.HORIZONS, "search_horizons": SEARCH_H,
        "labels": labels(), "features": events.FEATURES, "min_dev": events.MIN_DEV, "min_slice": events.MIN_SLICE,
        "data_start": {c: str(v) if v else None for c, v in starts.items()},
        "coverage": {c: [str(v["first"]), int(v["rows"] or 0)] for c, v in cov.items()},
        "only": args.only, "chunks": args.chunks,
    }  # fmt: skip
    dev_end = date.fromisoformat(args.dev_end)
    studies = {"sp500": events.Study(df, conds, "sp500", dev_end, starts)}
    meta["rows_sp500"] = studies["sp500"].n
    studies["all"] = events.Study(df, conds, "all", dev_end, starts)
    del df
    gc.collect()
    save("meta.json", meta)
    for u in ("sp500", "all"):
        log(f"対象 {u}：{studies[u].n:,} 行")
        save(f"results_{u}.json", run_universe(studies[u], conds, starts, log))
        studies.pop(u)
        gc.collect()
    log("完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
