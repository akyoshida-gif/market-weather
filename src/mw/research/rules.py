"""Rule extraction (VALIDATION_PLAN §2): learn which condition combinations preceded large excess returns.

A shallow decision tree is fitted on development-period rows (sampled) with target xret_h; each leaf
is a conjunction of conditions (X or NOT X). Leaves with the highest mean become candidate rules,
which are then evaluated like any other condition set (onset events, all periods) by Study.profile.
LightGBM gain importance is reported alongside to show which conditions matter overall.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from mw.research.events import MIN_DEV, Study


def _sample(
    study: Study, cols: list[str], h: int, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    y_all = study.m[f"xret_{h}"]
    # development rows whose outcome ends inside the period, and where every used condition has data
    ok = study.dev_mask_h(h) & np.isfinite(y_all) & (study.date_rank >= study.start_of(tuple(cols)))
    idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    X = np.empty((len(idx), len(cols)), dtype=np.uint8)
    for j, c in enumerate(cols):
        X[:, j] = study._cond(c)[idx]
    # winsorise the target so a handful of 10x moves do not dominate the splits
    y = np.clip(y_all[idx], -0.5, 1.0)
    return X, y


def tree_rules(study: Study, cols: list[str], h: int, max_rows: int = 3_000_000, depth: int = 6,
               min_leaf: int = 2000, top: int = 25, seed: int = 0) -> tuple[list[dict[str, Any]], int]:  # fmt: skip
    X, y = _sample(study, cols, h, max_rows, seed)
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed).fit(
        X, y
    )
    t = tree.tree_
    rules: list[dict[str, Any]] = []

    def walk(node: int, path: tuple[str, ...]) -> None:
        if t.children_left[node] == -1:
            rules.append(
                {
                    "conds": path,
                    "leaf_n": int(t.n_node_samples[node]),
                    "leaf_mean": float(t.value[node][0][0]),
                }
            )
            return
        c = cols[t.feature[node]]
        walk(t.children_left[node], path + ("!" + c,))  # value <= 0.5 -> condition false
        walk(t.children_right[node], path + (c,))

    walk(0, ())
    n_leaves = len(rules)
    base = float(y.mean())
    rules = [r for r in rules if r["conds"] and r["leaf_mean"] > base]
    # keep only positive conditions plus the negations needed; drop rules made only of negations
    rules = [r for r in rules if any(not c.startswith("!") for c in r["conds"])]
    rules.sort(key=lambda r: -r["leaf_mean"])
    out: list[dict[str, Any]] = []
    for r in rules[:top]:
        conds = tuple(sorted(set(r["conds"])))
        n, mu, tt = study.quick(study.events(conds), h)
        if n >= MIN_DEV:
            out.append({**r, "conds": conds, "n_dev": n, "x_dev": mu, "t_dev": tt})
    return out, n_leaves


def importance(
    study: Study, cols: list[str], h: int, max_rows: int = 2_000_000, seed: int = 0
) -> list[tuple[str, float]]:
    import lightgbm as lgb

    X, y = _sample(study, cols, h, max_rows, seed)
    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05, min_child_samples=2000,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=seed)  # fmt: skip
    m.fit(X, y)
    g = m.booster_.feature_importance(importance_type="gain")
    tot = float(g.sum()) or 1.0
    return sorted(((c, float(v) / tot) for c, v in zip(cols, g, strict=True)), key=lambda z: -z[1])
