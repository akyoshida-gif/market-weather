"""Rule extraction and machine learning (VALIDATION_PLAN §4).

tree_rules(): a shallow decision tree on the yes/no conditions (incl. numeric bins and market-wide bins)
  learns which combinations preceded large excess returns in the search window; each leaf becomes a rule.
ml_scan(): gradient boosting (LightGBM) on the raw numbers (stock f_* plus market-wide m_* of that day).
  Outputs: importance, how the outcome changes across each feature's deciles (training vs later period),
  stock x market interactions, and whether a higher model score really meant better results later ("確度").
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

from mw.research.events import MIN_DEV, Study


def _sample_idx(study: Study, h: int, max_rows: int, seed: int, cols: list[str]) -> np.ndarray:
    lo, hi = study.search_mask_h(h)
    y = study.m[f"xret_{h}"]
    ok = (
        (study.date_rank >= max(lo, study.start_of(tuple(cols))))
        & (study.date_rank <= hi)
        & np.isfinite(y)
    )
    idx = np.flatnonzero(ok)
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))
    return idx


def tree_rules(study: Study, cols: list[str], h: int, max_rows: int = 2_000_000, depth: int = 6,
               min_leaf: int = 1500, top: int = 30, seed: int = 0) -> tuple[list[dict[str, Any]], int]:  # fmt: skip
    idx = _sample_idx(study, h, max_rows, seed, cols)
    X = np.empty((len(idx), len(cols)), dtype=np.uint8)
    for j, c in enumerate(cols):
        X[:, j] = study._cond(c)[idx]
    y = np.clip(
        study.m[f"xret_{h}"][idx], -0.5, 1.0
    )  # winsorised so a few 10x moves do not dominate
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed).fit(
        X, y
    )
    t = tree.tree_
    leaves: list[dict[str, Any]] = []

    def walk(node: int, path: tuple[str, ...]) -> None:
        if t.children_left[node] == -1:
            leaves.append(
                {
                    "conds": path,
                    "leaf_n": int(t.n_node_samples[node]),
                    "leaf_mean": float(t.value[node][0][0]),
                }
            )
            return
        c = cols[t.feature[node]]
        walk(t.children_left[node], path + ("!" + c,))
        walk(t.children_right[node], path + (c,))

    walk(0, ())
    base = float(y.mean())
    good = sorted(
        [
            r
            for r in leaves
            if r["leaf_mean"] > base and any(not c.startswith("!") for c in r["conds"])
        ],
        key=lambda r: -r["leaf_mean"],
    )
    out: list[dict[str, Any]] = []
    for r in good[:top]:
        conds = tuple(sorted(set(r["conds"])))
        n, mu, tt = study.quick(study.events(conds), h)
        if n >= MIN_DEV:
            out.append({**r, "conds": conds, "n": n, "x": mu, "t": tt})
    return out, len(leaves)


def ml_scan(parts: list[Path], day: pl.DataFrame, universe: str, h: int, train: tuple[date, date], test_start: date,
            max_rows: int = 800_000, seed: int = 0, log: Any = print) -> dict[str, Any]:  # fmt: skip
    """Gradient boosting on raw numbers. Train on `train` (outcomes ending inside it), evaluate from test_start."""
    import lightgbm as lgb

    xk = "xsp" if universe == "sp500" else "xret"
    scan = pl.scan_parquet([str(p) for p in parts])
    if universe == "sp500":
        scan = scan.filter(pl.col("in_sp500"))
    fcols = [c for c in scan.collect_schema().names() if c.startswith("f_")] + ["k_sic2"]
    mcols = [c for c in day.columns if c.startswith("m_")]
    y = f"{xk}_{h}"
    tr_end = np.datetime64(train[1]) - np.timedelta64(
        int(h * 1.45) + 3, "D"
    )  # h sessions ~ 1.45h calendar days
    d = pl.col("as_of")
    in_tr = (d >= train[0]) & (d <= pl.lit(tr_end.astype(object)))
    in_te = d >= test_start
    # thin out rows while reading (hash of stock+date = reproducible pseudo-random sample)
    counts = (
        scan.filter(pl.col(y).is_not_null())
        .select(in_tr.sum().alias("tr"), in_te.sum().alias("te"))
        .collect()
        .row(0)
    )
    keep_tr = min(1.0, max_rows / max(counts[0], 1))
    keep_te = min(1.0, max_rows / 2 / max(counts[1], 1))
    u = (pl.struct("entity_id", "as_of").hash(seed) % 1_000_000).cast(pl.Float64) / 1_000_000
    df = (
        scan.filter(pl.col(y).is_not_null() & ((in_tr & (u < keep_tr)) | (in_te & (u < keep_te))))
        .select("as_of", y, *fcols)
        .collect()
        .join(day, on="as_of", how="left")
    )
    dn = df["as_of"].to_numpy()
    tr = (dn >= np.datetime64(train[0])) & (dn <= tr_end)
    te = dn >= np.datetime64(test_start)
    feats = fcols + mcols
    # fill column by column (no full-table copy); the joined table is the largest object in this job
    X = np.empty((df.height, len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        X[:, j] = df[f].cast(pl.Float32).to_numpy()
    raw = df[y].to_numpy().astype(np.float32)
    del df
    Y = np.clip(raw, -0.5, 1.0)
    tri = np.flatnonzero(tr)
    log(f"      機械学習：学習 {len(tri):,} 行・確認 {int(te.sum()):,} 行・項目 {len(feats)}")
    model = lgb.LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.03, min_child_samples=1000, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=seed)  # fmt: skip
    model.fit(X[tri], Y[tri], categorical_feature=[feats.index("k_sic2")])
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = sorted(
        zip(feats, (gain / max(gain.sum(), 1e-9)).tolist(), strict=True), key=lambda z: -z[1]
    )
    out: dict[str, Any] = {"h": h, "train": [str(train[0]), str(train[1])], "test_start": str(test_start), "n_train": len(tri),
                           "n_test": int(te.sum()), "importance": imp[:60]}  # fmt: skip
    if te.sum() > 1000:
        p = model.predict(X[te])
        q = np.quantile(p, np.linspace(0.1, 0.9, 9))
        b = np.digitize(p, q)
        yt = raw[te]
        dts = dn[te]
        out["score_deciles_test"] = [{"decile": k + 1, "n": int((b == k).sum()), "xmean": float(np.nanmean(yt[b == k])),
                                      "days": int(len(np.unique(dts[b == k])))} for k in range(10)]  # fmt: skip
        out["test_top_decile_by_month"] = _monthly(dts[b == 9], yt[b == 9])
    dec = []
    for f, _ in imp[:40]:
        if f == "k_sic2":
            continue
        v = X[:, feats.index(f)]
        ok = np.isfinite(v)
        ref = v[tr & ok]
        if len(ref) < 1000:
            continue
        q = np.quantile(ref, np.linspace(0.1, 0.9, 9))
        b = np.digitize(v, q)
        row: dict[str, Any] = {"feature": f, "cuts": q.tolist()}
        for nm, msk in (("train", tr), ("test", te)):
            row[nm] = [
                float(np.nanmean(raw[msk & ok & (b == k)]))
                if (msk & ok & (b == k)).sum() > 50
                else None
                for k in range(10)
            ]
        dec.append(row)
    out["deciles"] = dec
    top_f = [f for f, _ in imp if f.startswith("f_")][:6]
    top_m = [f for f, _ in imp if f.startswith("m_")][:6]
    grids = []
    for a in top_f:
        for bb in top_m:
            va, vb = X[:, feats.index(a)], X[:, feats.index(bb)]
            ok = np.isfinite(va) & np.isfinite(vb)
            if (ok & tr).sum() < 5000:
                continue
            ba = np.digitize(va, np.quantile(va[ok & tr], [0.2, 0.4, 0.6, 0.8]))
            bq = np.digitize(vb, np.quantile(vb[ok & tr], [0.2, 0.4, 0.6, 0.8]))
            g: dict[str, Any] = {"a": a, "b": bb}
            for nm, msk in (("train", tr), ("test", te)):
                g[nm] = [[float(np.nanmean(raw[msk & ok & (ba == i) & (bq == k)])) if (msk & ok & (ba == i) & (bq == k)).sum() > 100 else None
                          for k in range(5)] for i in range(5)]  # fmt: skip
            grids.append(g)
    out["interactions"] = grids
    return out


def _monthly(dates: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    if len(dates) == 0:
        return []
    m = dates.astype("datetime64[M]")
    return [
        {"month": str(u), "n": int((m == u).sum()), "xmean": float(np.nanmean(y[m == u]))}
        for u in np.unique(m)
    ]
