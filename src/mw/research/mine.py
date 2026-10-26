"""Search 1-3 condition patterns for each trade template; select on train+valid, report test.

Selection rules (fixed before looking at test):
  train: >= 150 events, mean R > 0, t-stat >= 2.5
  valid: >= 60 events,  mean R > 0, t-stat >= 1.5
Triples are built by beam search from the best pairs on TRAIN only.
All patterns tested are counted; the report states the number (multiple-testing context).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import polars as pl

from mw.research import patterns as P

MIN_TRAIN, MIN_VALID = 150, 60
T_TRAIN, T_VALID = 2.5, 1.5


def prepare(
    df: pl.DataFrame, market: str, bench: str, min_dvol: float, min_px: float
) -> dict[str, Any]:
    d = P.conditions(df, market)
    # market-level conditions joined by date
    b = d.filter(pl.col("entity_id") == bench).select(
        "as_of",
        (pl.col("close") > pl.col("ma200")).alias("c_mkt_up200"),
        (pl.col("close") > pl.col("ma50")).alias("c_mkt_up50"),
    )
    br = (
        d.filter(pl.col("ma50").is_not_null())
        .group_by("as_of")
        .agg((pl.col("close") > pl.col("ma50")).mean().alias("_br"))
    )
    d = (
        d.join(b, on="as_of", how="left")
        .join(br, on="as_of", how="left")
        .with_columns(
            pl.col("c_mkt_up200").fill_null(False),
            pl.col("c_mkt_up50").fill_null(False),
            (pl.col("_br") > 0.5).fill_null(False).alias("c_breadth_hi"),
        )
        .sort(["entity_id", "as_of"])
    )
    keys = [c[2:] for c in d.columns if c.startswith("c_")]
    # data errors: split events / >60% one-day moves -> no events within 5 sessions after, and
    # outcomes spanning them are dropped
    jump = (pl.col("close") / pl.col("pc") - 1).abs() > 0.6
    split = (pl.col("splits").fill_null(0) != 0) if "splits" in d.columns else pl.lit(False)
    d = d.with_columns((jump | split).fill_null(False).alias("bad"))
    d = d.with_columns(pl.col("bad").cast(pl.Int8).rolling_sum(6).over("entity_id").fill_null(0).alias("bad_back"),
                       pl.col("bad").cast(pl.Int8).reverse().rolling_sum(61).over("entity_id").reverse().fill_null(0).alias("bad_fwd"))  # fmt: skip
    ent = d["entity_id"].to_numpy()
    same_prev = np.r_[False, ent[1:] == ent[:-1]]
    conds = {k: d["c_" + k].to_numpy() for k in keys}
    prev = {k: np.where(same_prev, np.r_[False, v[:-1]], False) for k, v in conds.items()}
    liquid = (
        (d["dvol20"].fill_null(0) >= min_dvol)
        & (d["close"] >= min_px)
        & (d["bad_back"] == 0)
        & (d["bad_fwd"] == 0)
    ).to_numpy()
    outs = {}
    for t in P.TEMPLATES:
        ret, R = P.outcomes(d, t, market)
        outs[t.key] = (ret, R)
    per = P.period_of(d["as_of"])
    years = d["as_of"].dt.year().to_numpy()
    return {"conds": conds, "prev": prev, "liquid": liquid, "outs": outs, "per": per, "years": years,
            "keys": keys, "dates": d["as_of"], "entity": ent, "frame": d,
            "date_idx": d["as_of"].rank("dense").cast(pl.Int64).to_numpy()}  # fmt: skip


def _stats(R: np.ndarray, ret: np.ndarray) -> dict[str, float]:
    n = len(R)
    if n == 0:
        return {"n": 0}
    m, sd = float(R.mean()), float(R.std(ddof=1)) if n > 1 else 0.0
    pos, neg = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return {"n": n, "win": float((ret > 0).mean()), "mean_R": m, "t": m / (sd / np.sqrt(n)) if sd > 0 else 0.0,
            "avg_ret": float(ret.mean()), "pf": float(pos / neg) if neg > 0 else float("inf")}  # fmt: skip


def evaluate(prep: dict[str, Any], pattern: tuple[str, ...], tkey: str) -> dict[str, Any]:
    cur = np.logical_and.reduce([prep["conds"][k] for k in pattern])
    prv = np.logical_and.reduce([prep["prev"][k] for k in pattern])
    ev = cur & ~prv & prep["liquid"]
    ret, R = prep["outs"][tkey]
    ok = ev & np.isfinite(R)
    out: dict[str, Any] = {"pattern": pattern, "template": tkey}
    for p, name in ((0, "train"), (1, "valid"), (2, "test")):
        sel = ok & (prep["per"] == p)
        out[name] = _stats(R[sel], ret[sel])
    # robustness: are results driven by a few crash/rebound days? (events cluster by date)
    sel = ok & (prep["per"] == 2)
    if sel.sum() >= 30:
        dts = prep["date_idx"][sel]
        rr = R[sel]
        uniq, inv = np.unique(dts, return_inverse=True)
        per_day = np.bincount(inv, weights=rr) / np.bincount(inv)
        counts = np.bincount(inv)
        top = np.argsort(-counts)[:5]
        mask = ~np.isin(inv, top)
        out["test_days"] = int(len(uniq))
        out["test_top5_share"] = float(counts[top].sum() / counts.sum())
        out["test_day_mean_R"] = float(per_day.mean())
        out["test_ex_top5_mean_R"] = float(rr[mask].mean()) if mask.any() else float("nan")
    yrs = prep["years"][ok]
    span = max(1, len(np.unique(prep["years"])))
    out["per_year"] = len(yrs) / span
    return out


def passes(r: dict[str, Any]) -> bool:
    a, b = r["train"], r["valid"]
    return (a.get("n", 0) >= MIN_TRAIN and a["mean_R"] > 0 and a["t"] >= T_TRAIN
            and b.get("n", 0) >= MIN_VALID and b["mean_R"] > 0 and b["t"] >= T_VALID)  # fmt: skip


def search(prep: dict[str, Any], beam: int = 60) -> dict[str, Any]:
    keys = prep["keys"]
    results: list[dict[str, Any]] = []
    tested = 0
    for t in P.TEMPLATES:
        singles = [evaluate(prep, (k,), t.key) for k in keys]
        pairs = [evaluate(prep, pr, t.key) for pr in itertools.combinations(keys, 2)]
        tested += len(singles) + len(pairs)
        # beam on TRAIN only: best pairs by train mean R with enough events
        cand = sorted(
            [p for p in pairs if p["train"].get("n", 0) >= MIN_TRAIN],
            key=lambda r: -r["train"]["mean_R"],
        )[:beam]
        seen, triples = set(), []
        for p in cand:
            for k in keys:
                if k in p["pattern"]:
                    continue
                tri = tuple(sorted((*p["pattern"], k)))
                if tri in seen:
                    continue
                seen.add(tri)
                triples.append(evaluate(prep, tri, t.key))
        tested += len(triples)
        results += singles + pairs + triples
    return {"results": results, "tested": tested}
