"""Method A: event study (VALIDATION_PLAN §3, §5, §6).

For every (stock, day) row and horizon h (sessions), starting at the NEXT session's open:
  ret_h  = close[t+h] / open[t+1] - 1
  mfe_h  = max(high[t+1 .. t+h]) / open[t+1] - 1
  mae_h  = min(low[t+1 .. t+h]) / open[t+1] - 1
  xret_h = ret_h - same-day mean of the universe (all US stocks, or S&P 500 members for that universe)
  R outcomes per stop candidate (features.STOPS): the trade's result in units of its stop distance.
Conditions come in three kinds:
  c_*                  yes/no conditions (previous-session state stored as p_*)
  f_*__lo10/lo20/hi20/hi10   a stock's numeric feature in its bottom/top 10-20% (cut points from 2010-2024)
  m_*__lo20/hi20/up/dn       a market-wide feature (per session) low/high, or rising/falling
An event is the first day a combination of positive conditions becomes true for a stock (onset);
"!x" (NOT x) only filters the state that day. Rows before a condition's data starts are excluded.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from mw.research import features

HORIZONS = [1, 3, 5, 10, 20, 40, 60, 120]
BLOCKS = [("2010-13", date(2010, 1, 1), date(2013, 12, 31)), ("2014-17", date(2014, 1, 1), date(2017, 12, 31)),
          ("2018-21", date(2018, 1, 1), date(2021, 12, 31)), ("2022-24", date(2022, 1, 1), date(2024, 12, 31)),
          ("2025-", date(2025, 1, 1), date(2099, 1, 1))]  # fmt: skip
FEATURES = {"f_atr_pct": "値動きの大きさ（ATR÷株価）", "f_dvol20": "20日平均の売買代金", "f_ret5": "直前5日の騰落率",
            "f_ret63": "直前3か月の騰落率", "f_dist_hi": "52週高値からの距離"}  # fmt: skip
MIN_DEV = 30  # minimum events in the search window (VALIDATION_PLAN §4); fewer -> "参考"
MIN_SLICE = 10  # a period counts in the grade only with this many events
MIN_TEST = 20  # events needed in the confirmation period and after removing the 5 busiest days
MIN_T_SEARCH = 2.0  # day-clustered t of the R expectancy in the search window
MIN_T_TEST = 1.5  # same in the confirmation period
MAX_TOP5_SHARE = 0.5  # at most half of the events may sit on the 5 busiest days
MIN_DAYS = 20  # the search keeps only patterns that occurred on at least this many different days
CUT_END = date(2024, 12, 31)
MIN_CUT_ROWS = (
    1000  # a numeric stock feature needs this many rows up to CUT_END to define its cut points
)
MIN_CUT_DAYS = 250  # same for a market-wide feature (sessions)
MAX_BIN_SHARE = 0.30  # a numeric bin holding on more than this share of rows is dropped
MIN_VALID_STOP = 0.80  # a stop candidate competes only if it is usable on this share of the events  # cut points of numeric conditions come from 2010..this date
F_BINS = {"lo10": 0.10, "lo20": 0.20, "hi20": 0.80, "hi10": 0.90}
M_BINS = {"lo20": 0.20, "hi20": 0.80}
BIN_LABELS = {
    "lo10": "下位10%",
    "lo20": "下位20%",
    "hi20": "上位20%",
    "hi10": "上位10%",
    "up": "上昇中",
    "dn": "低下中",
}


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


def _first_rank(dates_sorted: np.ndarray, d: Any) -> int:
    return int(np.searchsorted(dates_sorted, np.datetime64(d))) + 1


class Study:
    """Numpy view of the dataset for fast evaluation of many condition sets (memory-bounded loading).

    window: (search_start, search_end) — patterns are searched and judged mainly on this window;
    test_start: first date of the untouched confirmation period (None for the 2025-only search).
    """

    def __init__(self, parts: list[Path], day: pl.DataFrame, universe: str = "all",
                 window: tuple[date, date] = (date(2022, 1, 1), date(2024, 12, 31)), test_start: date | None = date(2025, 1, 1),
                 starts: dict[str, Any] | None = None, horizons: list[int] | None = None, log: Any = print) -> None:  # fmt: skip
        self.universe = universe
        scan = pl.scan_parquet([str(p) for p in parts])
        if universe == "sp500":
            scan = scan.filter(pl.col("in_sp500"))
        self._scan = scan
        base = scan.select("entity_id", "as_of", "raw_close", "vol_tier", "contig").collect()
        self.n = base.height
        ent = base["entity_id"].to_numpy()
        self.same_prev = np.r_[False, ent[1:] == ent[:-1]]
        self.contig = base["contig"].to_numpy() & self.same_prev
        del ent
        d = base["as_of"]
        self.udates = np.sort(d.unique().to_numpy())
        self.date_rank = (np.searchsorted(self.udates, d.to_numpy()) + 1).astype(np.int32)
        self.tier = base["vol_tier"].to_numpy()
        self.cost = cost_of(base["raw_close"].to_numpy())
        self.ent_codes = base["entity_id"].cast(pl.Categorical).to_physical().to_numpy()
        del base
        self.win = window
        self.win_lo = _first_rank(self.udates, window[0])
        self.win_hi = int(np.searchsorted(self.udates, np.datetime64(window[1]), side="right"))
        self.test_lo = _first_rank(self.udates, test_start) if test_start else None
        self.block = np.zeros(self.n, dtype=np.int8)
        for i, (_, a, b) in enumerate(BLOCKS):
            self.block[
                (self.date_rank >= _first_rank(self.udates, a))
                & (
                    self.date_rank
                    <= int(np.searchsorted(self.udates, np.datetime64(b), side="right"))
                )
            ] = i
        self.start_rank: dict[str, int] = {}
        for c, s in (starts or {}).items():
            if s is not None:
                self.start_rank[c] = _first_rank(self.udates, s)
        # ---- yes/no conditions
        cols = scan.collect_schema().names()
        self.c_cols = [c for c in cols if c.startswith("c_")]
        self.packed: dict[str, np.ndarray] = {}
        self.packed_prev: dict[str, np.ndarray] = {}
        for c in self.c_cols:
            a = scan.select(c, "p_" + c[2:]).collect()
            self.packed[c] = np.packbits(a[c].fill_null(False).to_numpy())
            self.packed_prev[c] = np.packbits(a["p_" + c[2:]].fill_null(False).to_numpy())
        log(f"    はい・いいえの条件 {len(self.c_cols)} 種類を読み込み")
        # ---- numeric stock features -> bins (cut points from rows up to CUT_END, all eligible rows)
        self.f_cols = [c for c in cols if c.startswith("f_")]
        self.cuts: dict[str, dict[str, float]] = {}
        self.feat: dict[str, np.ndarray] = {}
        cut_rank = int(np.searchsorted(self.udates, np.datetime64(CUT_END), side="right"))
        for c in self.f_cols:
            v = scan.select(c).collect()[c].to_numpy().astype(np.float32)
            ok = np.isfinite(v)
            ref = v[ok & (self.date_rank <= cut_rank)]
            if len(ref) < MIN_CUT_ROWS:
                continue
            qs = {k: float(np.quantile(ref, q)) for k, q in F_BINS.items()}
            lo_ref, hi_ref = float(ref.min()), float(ref.max())
            kept: dict[str, float] = {}
            for k, q in qs.items():
                if k.startswith("lo"):
                    m = ok & ((v < q) if q <= lo_ref else (v <= q))
                else:
                    m = ok & ((v > q) if q >= hi_ref else (v >= q))
                share = (
                    float(m[self.date_rank <= cut_rank].mean())
                    if (self.date_rank <= cut_rank).any()
                    else 0.0
                )
                if (
                    share == 0.0 or share > MAX_BIN_SHARE
                ):  # a bin that is (almost) everything or nothing says nothing
                    continue
                self.packed[f"{c}__{k}"] = np.packbits(m)
                kept[k] = q
            if not kept:
                continue
            self.cuts[c] = kept
            first = int(self.date_rank[ok].min()) if ok.any() else 0
            for k in kept:
                self.start_rank[f"{c}__{k}"] = max(self.start_rank.get(f"{c}__{k}", 0), first)
            if c in FEATURES:
                self.feat[c] = v
        self.fbin_cols = [k for k in self.packed if "__" in k]
        log(f"    銘柄の数値 {len(self.cuts)} 項目 → 条件 {len(self.fbin_cols)} 種類")
        # ---- market-wide (per session) features -> bins, stored per date
        day = day.filter(
            pl.col("as_of").is_in(pl.Series(self.udates).cast(pl.Date).implode())
        ).sort("as_of")
        drank = np.searchsorted(self.udates, day["as_of"].to_numpy()) + 1
        self.day_cond: dict[str, np.ndarray] = {}
        dref = day["as_of"].to_numpy() <= np.datetime64(CUT_END)
        for c in [x for x in day.columns if x.startswith("m_")]:
            v = day[c].to_numpy().astype(np.float64)
            ok = np.isfinite(v)
            if (ok & dref).sum() < MIN_CUT_DAYS:
                continue
            first = int(drank[ok].min())
            bins = {
                k: (v <= np.quantile(v[ok & dref], q))
                if k.startswith("lo")
                else (v >= np.quantile(v[ok & dref], q))
                for k, q in M_BINS.items()
            }
            self.cuts[c] = {k: float(np.quantile(v[ok & dref], q)) for k, q in M_BINS.items()}
            if "_chg" in c or "_ret" in c or c.startswith("m_rs_"):
                bins |= {"up": v > 0, "dn": v < 0}
            for k, b in bins.items():
                arr = np.zeros(len(self.udates) + 2, dtype=bool)
                arr[drank] = b & ok
                self.day_cond[f"{c}__{k}"] = arr
                self.start_rank[f"{c}__{k}"] = first
        self.mbin_cols = list(self.day_cond)
        log(f"    市場全体の条件 {len(self.mbin_cols)} 種類")
        # ---- outcomes
        hz = horizons or HORIZONS
        self.horizons = hz
        xk = "xsp" if universe == "sp500" else "xret"
        self.m: dict[str, np.ndarray] = {}
        for h in hz:
            a = scan.select(f"ret_{h}", f"{xk}_{h}", f"mfe_{h}", f"mae_{h}").collect()
            self.m[f"ret_{h}"] = a[f"ret_{h}"].to_numpy().astype(np.float32)
            self.m[f"xret_{h}"] = a[f"{xk}_{h}"].to_numpy().astype(np.float32)
            self.m[f"mfe_{h}"] = a[f"mfe_{h}"].to_numpy().astype(np.float16)
            self.m[f"mae_{h}"] = a[f"mae_{h}"].to_numpy().astype(np.float16)
        self.rh = [h for h in features.R_HORIZONS if h in hz]
        self.r: dict[str, np.ndarray] = {}
        for s in features.STOPS:
            rp = scan.select(f"rp_{s}").collect()[f"rp_{s}"].to_numpy().astype(np.float32)
            self.r[f"costR_{s}"] = (self.cost / np.where(rp > 0, rp, np.nan)).astype(np.float16)
            for h in self.rh:
                a = scan.select(f"r_{s}_{h}_fin", f"r_{s}_{h}_mfe").collect()
                self.r[f"{s}_{h}_fin"] = a[f"r_{s}_{h}_fin"].to_numpy().astype(np.float16)
                self.r[f"{s}_{h}_mfe"] = a[f"r_{s}_{h}_mfe"].to_numpy().astype(np.float16)
        self._base: dict[tuple[str, int, str], float | None] = {}
        log(f"    結果の列を読み込み（{self.n:,} 行）")

    # ---- condition masks
    def all_conditions(self) -> list[str]:
        return self.c_cols + self.fbin_cols + self.mbin_cols

    def _cond(self, name: str) -> np.ndarray:
        neg = name.startswith("!")
        key = name.lstrip("!")
        if key in self.day_cond:
            v = self.day_cond[key][self.date_rank]
        else:
            v = np.unpackbits(self.packed[key], count=self.n).view(bool)
        return ~v if neg else v

    def _prev(self, name: str) -> np.ndarray:
        """Previous-session state. Unknown previous session (not contiguous) counts as 'true' -> no onset."""
        if name in self.packed_prev:
            return np.unpackbits(self.packed_prev[name], count=self.n).view(bool)
        if name in self.day_cond:
            return self.day_cond[name][self.date_rank - 1]
        cur = np.unpackbits(self.packed[name], count=self.n).view(bool)
        return np.where(self.contig, np.r_[False, cur[:-1]], True)

    def start_of(self, conds: tuple[str, ...]) -> int:
        return max((self.start_rank.get(c.lstrip("!"), 0) for c in conds), default=0)

    def state(self, conds: tuple[str, ...]) -> np.ndarray:
        cur = self._cond(conds[0]).copy()
        for c in conds[1:]:
            cur &= self._cond(c)
        return cur

    def prev_state(self, conds: tuple[str, ...]) -> np.ndarray:
        pos = [c for c in conds if not c.startswith("!")]
        prv = self._prev(pos[0]).copy()
        for c in pos[1:]:
            prv &= self._prev(c)
        return prv

    def onset(self, cur: np.ndarray, prv: np.ndarray, start: int) -> np.ndarray:
        ev = cur & ~prv
        if start > 0:
            ev &= self.date_rank >= start
        return ev

    def events(self, conds: tuple[str, ...]) -> np.ndarray:
        if not any(not c.startswith("!") for c in conds):
            return np.zeros(self.n, dtype=bool)
        return self.onset(self.state(conds), self.prev_state(conds), self.start_of(conds))

    # ---- statistics
    def stats(self, idx: np.ndarray, h: int) -> dict[str, Any]:
        r, x = self.m[f"ret_{h}"][idx], self.m[f"xret_{h}"][idx]
        ok = np.isfinite(r) & np.isfinite(x)
        r, x = r[ok], x[ok]
        if len(r) == 0:
            return {"n": 0}
        mf = self.m[f"mfe_{h}"][idx][ok].astype(np.float32)
        ma = self.m[f"mae_{h}"][idx][ok].astype(np.float32)
        cost, rk = self.cost[idx][ok], self.date_rank[idx][ok]
        net = r - cost
        n = len(r)
        k10 = max(1, n // 10)
        top = -np.partition(-net, k10 - 1)[:k10] if n > k10 else net
        return {
            "n": n, "mean": float(r.mean()), "net": float(net.mean()), "median": float(np.median(r)), "win": float((net > 0).mean()),
            "xmean": float(x.mean()), "xt": _clustered_t(x, rk), "p10": float(np.quantile(r, 0.1)), "p90": float(np.quantile(r, 0.9)),
            "top10_mean": float(top.mean()), "mfe_med": float(np.median(mf)), "mae_med": float(np.median(ma)),
            "hit20": float((mf >= 0.2).mean()), "hit50": float((mf >= 0.5).mean()),
        }  # fmt: skip

    def rstats(self, idx: np.ndarray, stop: str, h: int) -> dict[str, Any]:
        """R outcome with the given stop (after costs in R)."""
        fin = self.r[f"{stop}_{h}_fin"][idx].astype(np.float32)
        ok = np.isfinite(fin)
        if not ok.any():
            return {"n": 0}
        stopped = fin[ok] <= -0.999
        fin = fin[ok] - np.nan_to_num(self.r[f"costR_{stop}"][idx][ok].astype(np.float32), nan=0.0)
        mfe = self.r[f"{stop}_{h}_mfe"][idx][ok].astype(np.float32)
        n = len(fin)
        k10 = max(1, n // 10)
        top = -np.partition(-fin, k10 - 1)[:k10] if n > k10 else fin
        return {"n": n, "exp": float(fin.mean()), "median": float(np.median(fin)), "win": float((fin > 0).mean()),
                "stopped": float(stopped.mean()), "top10": float(top.mean()), "t": _clustered_t(fin, self.date_rank[idx][ok]),
                "valid": float(ok.mean()), **{f"hit{k}": float((mfe >= k).mean()) for k in (1, 2, 3, 5, 10)}}  # fmt: skip

    def base_exp(self, stop: str, h: int, which: str) -> float | None:
        """R expectancy of all eligible stock-days in the search window / confirmation period (cached)."""
        key = (stop, h, which)
        if key not in self._base:
            lo, hi = self.search_mask_h(h)
            if which == "search":
                m = (self.date_rank >= lo) & (self.date_rank <= hi)
            else:
                m = (self.date_rank >= self.test_lo) if self.test_lo else np.zeros(self.n, bool)
            self._base[key] = self.rstats(np.flatnonzero(m), stop, h).get("exp")
        return self._base[key]

    def search_mask_h(self, h: int) -> tuple[int, int]:
        """Rank range of the search window whose h-session outcome also ends inside the window."""
        return self.win_lo, self.win_hi - h

    def quick(self, ev: np.ndarray, h: int) -> tuple[int, float, float]:
        """(n, mean excess return, day-clustered t) in the search window."""
        idx = np.flatnonzero(ev)
        lo, hi = self.search_mask_h(h)
        rk = self.date_rank[idx]
        idx = idx[(rk >= lo) & (rk <= hi)]
        x = self.m[f"xret_{h}"][idx]
        ok = np.isfinite(x)
        x, rk = x[ok], self.date_rank[idx][ok]
        if len(x) < 2:
            return len(x), 0.0, 0.0
        return len(x), float(x.mean()), _clustered_t(x, rk)

    def profile(
        self, conds: tuple[str, ...], horizons: list[int] | None = None, full: bool = True
    ) -> dict[str, Any]:
        ev = self.events(conds)
        out: dict[str, Any] = {"conds": list(conds), "events": int(ev.sum())}
        if ev.sum() == 0:
            return out
        idx = np.flatnonzero(ev)
        dr = self.date_rank[idx]
        out["days"] = int(len(np.unique(dr)))
        test = (dr >= self.test_lo) if self.test_lo else np.zeros(len(idx), bool)
        last = int(self.date_rank.max())
        recent1 = dr > last - 252
        for h in horizons or self.horizons:
            lo, hi = self.search_mask_h(h)
            win = (dr >= lo) & (dr <= hi)
            mid = (lo + hi) // 2
            # the 5 busiest days are chosen among the evaluated events (search + test), not over all years
            rng = win | test
            days, counts = np.unique(dr[rng], return_counts=True)
            top5 = days[np.argsort(counts)[-5:]] if len(days) else days
            if len(counts):
                s_top5 = float(np.sort(counts)[-5:].sum() / counts.sum())
            ex5 = ~np.isin(dr, top5)
            sl = {
                "search": idx[win],
                "search_1st": idx[win & (dr <= mid)],
                "search_2nd": idx[win & (dr > mid)],
                "test": idx[test],
                "recent1": idx[recent1],
                "ex_top5": idx[ex5 & rng],
                "all": idx,
            }
            if full:
                sl |= {name: idx[self.block[idx] == i] for i, (name, _, _) in enumerate(BLOCKS)}
                sl |= {f"vol{t}": idx[(self.tier[idx] == t) & (win | test)] for t in (1, 2, 3, 4)}
            s: dict[str, Any] = {k: self.stats(v, h) for k, v in sl.items()}
            s["top5_share"] = s_top5 if len(counts) else None
            s["market_wide"] = all(
                c.lstrip("!") in self.day_cond or c.startswith("c_mkt_") for c in conds
            )
            if h in self.rh:
                # pick the stop with the best expectancy in the search window, then report it everywhere
                cand = {st: self.rstats(sl["search"], st, h) for st in features.STOPS}
                # compare stops only on (nearly) the same trades: a stop usable on few events is a different selection
                n_s = max(1, len(sl["search"]))
                ok = {
                    k: v
                    for k, v in cand.items()
                    if v.get("n", 0) >= MIN_DEV and v["n"] / n_s >= MIN_VALID_STOP
                }
                best = max(ok, key=lambda k: ok[k]["exp"]) if ok else None
                s["R_by_stop"] = {
                    k: {"exp": v.get("exp"), "n": v.get("n", 0), "valid": v.get("n", 0) / n_s}
                    for k, v in cand.items()
                }
                if best:
                    rs = {
                        k: self.rstats(sl[k], best, h)
                        for k in (
                            "search",
                            "search_1st",
                            "search_2nd",
                            "test",
                            "recent1",
                            "ex_top5",
                            "all",
                        )
                    }
                    if full:
                        rs |= {name: self.rstats(sl[name], best, h) for name, _, _ in BLOCKS}
                    s["R"] = {"stop": best, **rs}
                    if s["market_wide"]:
                        # a market-wide condition picks days, not stocks: compare with every stock on every day
                        s["R_base"] = {
                            "search": self.base_exp(best, h, "search"),
                            "test": self.base_exp(best, h, "test"),
                        }
            s["grade"] = grade(s, self.test_lo is not None)
            out[str(h)] = s
        return out


GRADES = ["主力候補（長寿命）", "主力候補（新しい傾向）", "観察", "参考", "不採用"]


def grade(s: dict[str, Any], has_test: bool = True) -> str:
    """VALIDATION_PLAN §5-3 (tightened 2026-10-27 after the first full run, §5-4).

    Uses the R expectancy of the best stop (chosen in the search window) when available, otherwise the
    after-cost mean return. 'Better than the universe' = mean excess return > 0 (skipped for market-wide
    conditions, which fire for every stock on the same day so their excess return is ~0 by construction).
    """
    R = s.get("R")

    base = s.get("R_base") or {}

    def val(k: str) -> tuple[int, float, float]:
        if R and R.get(k, {}).get("n", 0):
            # market-wide conditions: what counts is the edge over simply holding every stock that period
            b = (base.get("test") if k == "test" else base.get("search")) or 0.0
            e, t = R[k]["exp"], R[k].get("t", 0.0) or 0.0
            t_adj = (
                t * (e - b) / e if b and e else t
            )  # same standard error, measured from the baseline
            return R[k]["n"], e - b, t_adj
        v = s.get(k, {})
        return v.get("n", 0), v.get("net", 0.0) or 0.0, v.get("xt", 0.0) or 0.0

    n_s, e_s, t_s = val("search")
    if n_s < MIN_DEV:
        return "参考"
    x_s = s["search"].get("xmean")
    market_wide = s.get("market_wide", False)
    if e_s <= 0 or (not market_wide and (x_s is None or x_s <= 0)):
        return "不採用"
    n_x, e_x, _ = val("ex_top5")
    conc = s.get("top5_share")
    spread_ok = n_x >= MIN_TEST and e_x > 0 and (conc is None or conc <= MAX_TOP5_SHARE)
    if t_s < MIN_T_SEARCH or not spread_ok:
        return "観察" if spread_ok or n_x >= MIN_SLICE else "不採用"
    if not has_test:
        return "観察"
    n_t, e_t, t_t = val("test")
    halves = [val("search_1st"), val("search_2nd")]
    halves_ok = (
        all(n >= MIN_SLICE and e > 0 for n, e, _ in halves)
        if all(n for n, _, _ in halves)
        else True
    )
    if n_t < MIN_TEST or e_t <= 0 or t_t < MIN_T_TEST or not halves_ok:
        return "観察"
    old = [val(b[0]) for b in BLOCKS[:3]]
    old_e = [e for n, e, _ in old if n >= MIN_SLICE]
    return (
        "主力候補（長寿命）"
        if old_e and sum(e > 0 for e in old_e) * 2 > len(old_e)
        else "主力候補（新しい傾向）"
    )


class SearchView:
    """The search window only (rows whose h-session outcome ends inside it), for fast repeated evaluation.

    Each condition's (today, previous session) state is sliced once and kept bit-packed; rows before a
    condition's data starts are set to 'false' so they can never start an event.
    """

    def __init__(self, study: Study, h: int) -> None:
        lo, hi = study.search_mask_h(h)
        self.st = study
        self.idx = np.flatnonzero((study.date_rank >= lo) & (study.date_rank <= hi))
        self.n = len(self.idx)
        self.x = study.m[f"xret_{h}"][self.idx].astype(np.float64)
        self.ok = np.isfinite(self.x)
        self.rk = study.date_rank[self.idx]
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def pair(self, c: str) -> tuple[np.ndarray, np.ndarray]:
        if c not in self._cache:
            cur = self.st._cond(c)[self.idx]
            prv = (
                self.st._prev(c.lstrip("!"))[self.idx]
                if not c.startswith("!")
                else np.ones(self.n, bool)
            )
            start = self.st.start_rank.get(c.lstrip("!"), 0)
            if start:
                cur = cur & (self.rk >= start)
            self._cache[c] = (np.packbits(cur), np.packbits(prv))
        a, b = self._cache[c]
        return np.unpackbits(a, count=self.n).view(bool), np.unpackbits(b, count=self.n).view(bool)

    def quick(self, cur: np.ndarray, prv: np.ndarray) -> tuple[int, float, float]:
        sel = np.flatnonzero(cur & ~prv & self.ok)
        n = len(sel)
        if n < 2:
            return n, 0.0, 0.0
        if len(np.unique(self.rk[sel])) < MIN_DAYS:
            return 0, 0.0, 0.0  # one event (e.g. a single crash-rebound day) is not a pattern
        x = self.x[sel]
        return n, float(x.mean()), _clustered_t(x, self.rk[sel])


def stepwise(study: Study, seeds: list[str], extras: list[str], h: int, beam: int = 25, max_depth: int = 6,
             min_gain: float = 0.002, min_t: float = 2.0, log: Any = None) -> tuple[list[dict[str, Any]], int]:  # fmt: skip
    """Beam search: add one condition at a time, keep the best by mean excess return in the search window.

    A child is kept only if it beats its parent by `min_gain` and keeps MIN_DEV+ events; one bin per feature.
    """
    view = SearchView(study, h)
    pool = seeds + extras
    found: dict[tuple[str, ...], dict[str, Any]] = {}
    level: list[tuple[tuple[str, ...], float]] = []
    for c in seeds:
        cur, prv = view.pair(c)
        n, mu, t = view.quick(cur, prv)
        if n >= MIN_DEV:
            if t >= min_t:
                found[(c,)] = {"conds": (c,), "n": n, "x": mu, "t": t}
            level.append(((c,), mu))
    tried = len(seeds)
    level = sorted(level, key=lambda z: -z[1])[:beam]
    if log:
        log(f"      段階1：候補 {len(level)}・試した数 {tried:,}")
    for depth in range(max_depth - 1):
        nxt: dict[tuple[str, ...], float] = {}
        for conds, parent_mu in level:
            cur_p, prv_p = view.pair(conds[0])
            cur_p, prv_p = cur_p.copy(), prv_p.copy()
            for c in conds[1:]:
                a, b = view.pair(c)
                cur_p &= a
                prv_p &= b
            used = {c.split("__")[0] for c in conds}
            for c in pool:
                if c in conds or c.split("__")[0] in used:
                    continue
                key = tuple(sorted(conds + (c,)))
                if key in nxt or key in found:
                    continue
                a, b = view.pair(c)
                n, mu, t = view.quick(cur_p & a, prv_p & b)
                tried += 1
                if n < MIN_DEV or mu < parent_mu + min_gain:
                    continue
                nxt[key] = mu
                if t >= min_t:
                    found[key] = {"conds": key, "n": n, "x": mu, "t": t}
        if log:
            log(f"      段階{depth + 2}：候補 {len(nxt)}・試した数 {tried:,}")
        if not nxt:
            break
        level = sorted(nxt.items(), key=lambda z: -z[1])[:beam]
    return sorted(found.values(), key=lambda r: -r["x"]), tried


def regimes(study: Study, conds: tuple[str, ...], h: int, keys: list[str]) -> dict[str, Any]:
    """Same pattern split by market regimes (e.g. oil high / low): R or return in each state, search+test windows."""
    ev = study.events(conds)
    idx = np.flatnonzero(ev)
    out: dict[str, Any] = {}
    for k in keys:
        if k not in study.day_cond:
            continue
        m = study.day_cond[k][study.date_rank[idx]]
        out[k] = {"in": study.stats(idx[m], h), "out": study.stats(idx[~m], h)}
    return out


def strong_days(
    study: Study, conds: tuple[str, ...], h: int = 20, n_days: int = 5
) -> dict[str, Any]:
    """On the days this signal fired most, which stock traits did best (quartiles of a few features)."""
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
        out["by"][f] = [
            {"q": i + 1, "n": int((b == i).sum()), "xmean": float(x[ok][b == i].mean())}
            for i in range(4)
            if (b == i).any()
        ]
    return out


def bin_label(name: str, labels: dict[str, str], cuts: dict[str, dict[str, float]]) -> str:
    """Japanese label of a condition name like 'f_dev_ma20__hi10' or '!c_x'."""
    neg = name.startswith("!")
    key = name.lstrip("!")
    if "__" in key:
        base, b = key.split("__", 1)
        cut = cuts.get(base, {}).get(b)
        cut_s = f"（{'≦' if b.startswith('lo') else '≧'}{cut:.4g}）" if cut is not None else ""
        s = f"{labels.get(base, base)}が{BIN_LABELS.get(b, b)}{cut_s}"
    else:
        s = labels.get(key, key)
    return f"「{s}」ではない" if neg else s
