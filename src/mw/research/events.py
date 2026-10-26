"""Method A: event study (VALIDATION_PLAN §3).

For every (stock, day) row and horizon h (sessions), starting at the NEXT session's open:
  ret_h  = close[t+h] / open[t+1] - 1
  mfe_h  = max(high[t+1 .. t+h]) / open[t+1] - 1      (largest gain available)
  mae_h  = min(low[t+1 .. t+h]) / open[t+1] - 1       (largest drawdown suffered)
  xret_h = ret_h - mean(ret_h of the universe's eligible stocks that day)   (market/day effect removed)
           (all US stocks for universe "all"; S&P 500 members only for universe "sp500")
An event is the first day a combination of positive conditions becomes true for a stock (onset),
judged against the stock's previous session; "!c" (NOT c) conditions only filter the state that day.
Rows before a condition's data starts are "no data" and are excluded, not counted as false.
Results are reported by period block, last 3 years / last 1 year, and volatility tier.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import polars as pl

HORIZONS = [1, 3, 5, 10, 20, 40, 60, 120]
BLOCKS = [("2010-13", date(2010, 1, 1), date(2013, 12, 31)), ("2014-17", date(2014, 1, 1), date(2017, 12, 31)),
          ("2018-21", date(2018, 1, 1), date(2021, 12, 31)), ("2022-26", date(2022, 1, 1), date(2099, 1, 1))]  # fmt: skip
FEATURES = {"f_atr_pct": "値動きの大きさ（ATR÷株価）", "f_dvol20": "20日平均の売買代金", "f_ret5": "直前5日の騰落率",
            "f_ret63": "直前3か月の騰落率", "f_dist_hi": "52週高値からの距離"}  # fmt: skip
MIN_DEV = 30  # minimum events in the development period (VALIDATION_PLAN §2); fewer -> "参考"
MIN_SLICE = 10  # a period block / recent window counts in the grade only with this many events


def forward_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Add ret_h / mfe_h / mae_h columns (df sorted by entity, as_of). Data-error windows are nulled later."""
    g = "entity_id"
    entry = pl.col("open").shift(-1).over(g)
    cols = []
    for h in HORIZONS:
        fwd_close = pl.col("close").shift(-h).over(g)
        fwd_high = pl.col("high").reverse().rolling_max(h).reverse().shift(-1).over(g)
        fwd_low = pl.col("low").reverse().rolling_min(h).reverse().shift(-1).over(g)
        cols += [
            (fwd_close / entry - 1).cast(pl.Float32).alias(f"ret_{h}"),
            (fwd_high / entry - 1).cast(pl.Float32).alias(f"mfe_{h}"),
            (fwd_low / entry - 1).cast(pl.Float32).alias(f"mae_{h}"),
        ]
    return df.with_columns(cols)


def add_excess(df: pl.DataFrame) -> pl.DataFrame:
    """xret_h: minus the same-day mean of all rows; xsp_h: minus the same-day mean of S&P 500 rows."""
    day = df.group_by("as_of").agg([pl.col(f"ret_{h}").mean().alias(f"_d_{h}") for h in HORIZONS])
    sp = (
        df.filter(pl.col("in_sp500"))
        .group_by("as_of")
        .agg([pl.col(f"ret_{h}").mean().alias(f"_s_{h}") for h in HORIZONS])
    )
    df = df.join(day, on="as_of", how="left").join(sp, on="as_of", how="left")
    return df.with_columns(
        [
            (pl.col(f"ret_{h}") - pl.col(f"_d_{h}")).cast(pl.Float32).alias(f"xret_{h}")
            for h in HORIZONS
        ]
        + [
            (pl.col(f"ret_{h}") - pl.col(f"_s_{h}")).cast(pl.Float32).alias(f"xsp_{h}")
            for h in HORIZONS
        ]
    ).drop([f"_d_{h}" for h in HORIZONS] + [f"_s_{h}" for h in HORIZONS])


def vol_tier(df: pl.DataFrame, cut_end: date = date(2021, 12, 31)) -> pl.DataFrame:
    """Volatility tier 1 (calm) .. 4 (wild) by ATR/close; cut points from rows up to cut_end."""
    v = pl.col("atr") / pl.col("close")
    q = (
        df.filter(pl.col("as_of") <= cut_end)
        .select(
            v.quantile(0.25).alias("a"), v.quantile(0.5).alias("b"), v.quantile(0.75).alias("c")
        )
        .row(0)
    )
    return df.with_columns(
        pl.when(v <= q[0])
        .then(1)
        .when(v <= q[1])
        .then(2)
        .when(v <= q[2])
        .then(3)
        .otherwise(4)
        .cast(pl.Int8)
        .alias("vol_tier")
    )


def cost_of(close: np.ndarray) -> np.ndarray:
    """Round-trip cost by the day's (unadjusted) price (VALIDATION_PLAN §1)."""
    return np.where(close >= 20, 0.001, np.where(close >= 5, 0.003, 0.01)).astype(np.float32)


def _clustered_t(x: np.ndarray, ranks: np.ndarray) -> float:
    """t of the mean with events averaged per day first (same-day signals are not independent)."""
    if len(x) < 2:
        return 0.0
    _, inv = np.unique(ranks, return_inverse=True)
    cnt = np.bincount(inv)
    if len(cnt) < 2:
        return 0.0
    dm = np.bincount(inv, weights=x) / cnt
    sd = float(dm.std(ddof=1))
    return float(dm.mean() / (sd / math.sqrt(len(dm)))) if sd > 0 else 0.0


class Study:
    """Numpy view of the dataset for fast evaluation of many condition sets.

    Conditions (today and previous session) are stored bit-packed; a name prefixed with "!" means NOT.
    starts: first date each condition has data (None / missing = full history).
    """

    def __init__(self, df: pl.DataFrame, cond_cols: list[str], universe: str = "all", dev_end: date = date(2021, 12, 31),
                 starts: dict[str, Any] | None = None) -> None:  # fmt: skip
        if universe == "sp500":
            df = df.filter(pl.col("in_sp500"))
        self.universe = universe
        self.n = df.height
        self.cond_cols = list(cond_cols)
        self.packed = {c: np.packbits(df[c].fill_null(False).to_numpy()) for c in cond_cols}
        self.packed_prev = {
            c: np.packbits(df["p_" + c[2:]].fill_null(False).to_numpy()) for c in cond_cols
        }
        d = df["as_of"]
        self.date_rank = d.rank("dense").cast(pl.Int32).to_numpy()
        udates = d.unique().sort()
        last = d.max()
        self.last = last
        self.dev_end = dev_end
        self.dev_rank = int((udates <= dev_end).sum())  # last development-period date rank
        self.start_rank: dict[str, int] = {}
        for c, s in (starts or {}).items():
            if c in self.packed and s is not None:
                self.start_rank[c] = int((udates < s).sum()) + 1
        self.recent3 = (d >= date(last.year - 3, last.month, 1)).to_numpy()
        self.recent1 = (d >= date(last.year - 1, last.month, 1)).to_numpy()
        self.dev = (d <= dev_end).to_numpy()
        self.block = np.zeros(self.n, dtype=np.int8)
        for i, (_, a, b) in enumerate(BLOCKS):
            self.block[((d >= a) & (d <= b)).to_numpy()] = i
        self.tier = df["vol_tier"].to_numpy()
        self.cost = cost_of(df["raw_close"].to_numpy())
        self.feat = {c: df[c].to_numpy().astype(np.float32) for c in FEATURES if c in df.columns}
        xk = "xsp" if universe == "sp500" else "xret"
        self.m: dict[str, np.ndarray] = {}
        for h in HORIZONS:
            for k in ("ret", "mfe", "mae"):
                self.m[f"{k}_{h}"] = df[f"{k}_{h}"].to_numpy().astype(np.float32)
            self.m[f"xret_{h}"] = df[f"{xk}_{h}"].to_numpy().astype(np.float32)

    # ---- condition masks
    def _cond(self, name: str) -> np.ndarray:
        neg = name.startswith("!")
        v = np.unpackbits(self.packed[name.lstrip("!")], count=self.n).view(bool)
        return ~v if neg else v

    def _prev(self, name: str) -> np.ndarray:
        return np.unpackbits(self.packed_prev[name], count=self.n).view(bool)

    def start_of(self, conds: tuple[str, ...]) -> int:
        return max((self.start_rank.get(c.lstrip("!"), 0) for c in conds), default=0)

    def state(self, conds: tuple[str, ...]) -> np.ndarray:
        cur = self._cond(conds[0])
        for c in conds[1:]:
            cur &= self._cond(c)
        return cur

    def prev_state(self, conds: tuple[str, ...]) -> np.ndarray:
        """Previous-session state of the positive conditions only."""
        pos = [c for c in conds if not c.startswith("!")]
        prv = self._prev(pos[0])
        for c in pos[1:]:
            prv &= self._prev(c)
        return prv

    def onset(self, cur: np.ndarray, prv: np.ndarray, start: int) -> np.ndarray:
        ev = cur & ~prv
        if start > 0:
            ev &= self.date_rank >= start
        return ev

    def events(self, conds: tuple[str, ...]) -> np.ndarray:
        """Onset days of the positive conditions, with the NOT conditions holding that day; data-covered rows only."""
        if not any(not c.startswith("!") for c in conds):
            return np.zeros(self.n, dtype=bool)
        return self.onset(self.state(conds), self.prev_state(conds), self.start_of(conds))

    # ---- statistics
    def stats(self, idx: np.ndarray, h: int) -> dict[str, Any]:
        """idx: index array of event rows."""
        r, x, mf, ma = (self.m[f"{k}_{h}"][idx] for k in ("ret", "xret", "mfe", "mae"))
        ok = np.isfinite(r) & np.isfinite(x)
        r, x, mf, ma = r[ok], x[ok], mf[ok], ma[ok]
        cost, rk = self.cost[idx][ok], self.date_rank[idx][ok]
        n = len(r)
        if n == 0:
            return {"n": 0}
        net = r - cost
        k10 = max(1, n // 10)
        top = -np.partition(-net, k10 - 1)[:k10] if n > k10 else net
        return {
            "n": n, "mean": float(r.mean()), "net": float(net.mean()), "median": float(np.median(r)), "win": float((net > 0).mean()),
            "xmean": float(x.mean()), "xt": _clustered_t(x, rk),
            "p10": float(np.quantile(r, 0.1)), "p90": float(np.quantile(r, 0.9)), "top10_mean": float(top.mean()),
            "mfe_med": float(np.median(mf)), "mfe_p75": float(np.quantile(mf, 0.75)), "mae_med": float(np.median(ma)), "mae_p25": float(np.quantile(ma, 0.25)),
            "hit20": float((mf >= 0.2).mean()), "hit50": float((mf >= 0.5).mean()), "hit100": float((mf >= 1.0).mean()),
        }  # fmt: skip

    def dev_mask_h(self, h: int) -> np.ndarray:
        """Development rows whose h-session outcome ends inside the development period."""
        return self.date_rank <= self.dev_rank - h

    def quick(self, ev: np.ndarray, h: int) -> tuple[int, float, float]:
        """(n, mean excess return, day-clustered t) for the search; development period only."""
        idx = np.flatnonzero(ev)
        idx = idx[self.date_rank[idx] <= self.dev_rank - h]
        x = self.m[f"xret_{h}"][idx]
        ok = np.isfinite(x)
        x, rk = x[ok], self.date_rank[idx][ok]
        n = len(x)
        if n < 2:
            return n, 0.0, 0.0
        return n, float(x.mean()), _clustered_t(x, rk)

    def profile(self, conds: tuple[str, ...], horizons: list[int] | None = None) -> dict[str, Any]:
        """Full breakdown for one condition set: horizons x (all, dev/test, blocks, recent3, recent1, vol tiers, ex-top5-days)."""
        ev = self.events(conds)
        out: dict[str, Any] = {"conds": list(conds), "events": int(ev.sum())}
        st = self.start_of(conds)
        out["data_from_rank"] = st
        if ev.sum() == 0:
            return out
        idx = np.flatnonzero(ev)  # work on event rows only (much faster than full-length masks)
        dr = self.date_rank[idx]
        days, counts = np.unique(dr, return_counts=True)
        out["top5_share"] = float(np.sort(counts)[-5:].sum() / counts.sum())
        out["days"] = int(len(days))
        top5 = days[np.argsort(counts)[-5:]]
        blk, tier = self.block[idx], self.tier[idx]
        slices = {"all": idx, "test": idx[dr > self.dev_rank], "recent3": idx[self.recent3[idx]],
                  "recent1": idx[self.recent1[idx]], "ex_top5": idx[~np.isin(dr, top5)]}  # fmt: skip
        for i, (name, _, _) in enumerate(BLOCKS):
            slices[name] = idx[blk == i]
        for t in (1, 2, 3, 4):
            slices[f"vol{t}"] = idx[tier == t]
        for h in horizons or HORIZONS:
            s = {"dev": self.stats(idx[dr <= self.dev_rank - h], h)}
            s |= {k: self.stats(m, h) for k, m in slices.items()}
            s["grade"] = grade(s)
            out[str(h)] = s
        return out


GRADES = ["有力", "最近有効", "候補", "参考", "不採用"]


def grade(s: dict[str, Any]) -> str:
    """VALIDATION_PLAN §5 (net = after-cost mean; 'better than the universe' = mean excess > 0).

    Blocks count only with MIN_SLICE+ events. "有力" needs a positive net mean in at least 3 of 4 blocks;
    when a condition's data covers fewer blocks, in at least 3/4 of the blocks it covers (min. 2).
    Fewer than MIN_DEV development events -> "参考" (shown separately, VALIDATION_PLAN §2).
    """

    def has(k: str) -> bool:
        return s.get(k, {}).get("n", 0) >= MIN_SLICE

    def pos(k: str) -> bool:
        return has(k) and s[k]["net"] > 0

    if s.get("dev", {}).get("n", 0) < MIN_DEV:
        return "参考"
    avail = sum(has(b[0]) for b in BLOCKS)
    need = max(2, math.ceil(0.75 * avail))
    blocks = sum(pos(b[0]) for b in BLOCKS)
    if avail >= 2 and blocks >= need and pos("recent3") and s["all"].get("xmean", -1) > 0:
        return "有力"
    if pos("recent3") and pos("recent1"):
        return "最近有効"
    if pos("all"):
        return "候補"
    return "不採用"


def stepwise(study: Study, seeds: list[str], extras: list[str], h: int, beam: int = 40, max_depth: int = 8,
             min_gain: float = 0.002, min_t: float = 2.0) -> tuple[list[dict[str, Any]], int]:  # fmt: skip
    """Beam search: add one condition at a time, keep the best by development-period mean excess return.

    A child is kept only if it beats its parent by `min_gain` and stays at MIN_DEV+ events.
    Returns (every set with n >= MIN_DEV and day-clustered t >= min_t, number of sets evaluated).
    """
    pool = seeds + extras
    found: dict[tuple[str, ...], dict[str, Any]] = {}
    level: list[tuple[tuple[str, ...], float]] = []
    for c in seeds:
        n, mu, t = study.quick(study.events((c,)), h)
        if n >= MIN_DEV:
            if t >= min_t:
                found[(c,)] = {"conds": (c,), "n_dev": n, "x_dev": mu, "t_dev": t}
            level.append(((c,), mu))
    tried = len(seeds)
    level = sorted(level, key=lambda z: -z[1])[:beam]
    for _ in range(max_depth - 1):
        nxt: dict[tuple[str, ...], float] = {}
        for conds, parent_mu in level:
            cur_p, prv_p, st_p = study.state(conds), study.prev_state(conds), study.start_of(conds)
            for c in pool:
                if c in conds:
                    continue
                key = tuple(sorted(conds + (c,)))
                if key in nxt or key in found:
                    continue
                ev = study.onset(
                    cur_p & study._cond(c),
                    prv_p & study._prev(c),
                    max(st_p, study.start_rank.get(c, 0)),
                )
                n, mu, t = study.quick(ev, h)
                tried += 1
                if n < MIN_DEV or mu < parent_mu + min_gain:
                    continue
                nxt[key] = mu
                if t >= min_t:
                    found[key] = {"conds": key, "n_dev": n, "x_dev": mu, "t_dev": t}
        if not nxt:
            break
        level = sorted(nxt.items(), key=lambda z: -z[1])[:beam]
    return sorted(found.values(), key=lambda r: -r["x_dev"]), tried


def strong_days(
    study: Study, conds: tuple[str, ...], h: int = 20, n_days: int = 5
) -> dict[str, Any]:
    """VALIDATION_PLAN §3 'strong days': on the days this signal fired most, which stock traits did best.

    For each feature, events on those days are split into quartiles and the mean excess return per quartile is given.
    """
    ev = study.events(conds)
    if ev.sum() == 0:
        return {}
    dr = study.date_rank[ev]
    days, counts = np.unique(dr, return_counts=True)
    top = days[np.argsort(counts)[-n_days:]]
    sel = ev & np.isin(study.date_rank, top)
    x = study.m[f"xret_{h}"][sel]
    out: dict[str, Any] = {
        "events": int(sel.sum()),
        "xmean": float(np.nanmean(x)) if len(x) else 0.0,
        "by": {},
    }
    for f, v in study.feat.items():
        fv = v[sel]
        ok = np.isfinite(fv) & np.isfinite(x)
        if ok.sum() < 40:
            continue
        q = np.quantile(fv[ok], [0.25, 0.5, 0.75])
        b = np.digitize(fv[ok], q)
        out["by"][f] = [{"q": i + 1, "n": int((b == i).sum()), "xmean": float(x[ok][b == i].mean()),
                         "lo": float(fv[ok][b == i].min()), "hi": float(fv[ok][b == i].max())} for i in range(4) if (b == i).any()]  # fmt: skip
    return out
