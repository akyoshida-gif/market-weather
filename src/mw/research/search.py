"""Search the state atlas for conditions after which a stock did better than the other stocks (owner 2026-11-12).

No named patterns: every s_/m_ number is cut into quintiles (edges from the discovery period only) and the search
combines them freely.

Outcome y = the stock's return minus the average of all stocks on the same day ("stock picking"). Market-wide
numbers (m_*) cannot pick stocks by themselves; they only say WHEN a stock feature works.

Rows of the same period are not independent, so everything is counted by month: a condition's monthly averages
give its t-value (mean / standard error over months), and it must beat the others in most months and years.
Validation numbers are reported but never used for choosing. The holdout period is not loaded at all.

  stepwise  add one condition at a time (the best t-value), many starting points -> many weapons
  pairs     every pair of numbers as a 5x5 grid
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

C = pl.col
Q = 5
MISSING = 255
MIN_ROWS_MONTH = 20  # a month counts only with at least this many rows
PERIODS = {
    "discovery": (date(2011, 1, 1), date(2019, 12, 31)),
    "validation": (date(2020, 1, 1), date(2024, 6, 30)),
}
RECENT = {
    "discovery": (date(2021, 1, 1), date(2023, 12, 31)),
    "validation": (date(2024, 1, 1), date(2024, 6, 30)),
}
CONDS = [(0, 0), (Q - 1, Q - 1), (0, 1), (Q - 2, Q - 1), (1, 3), (2, 2)]


@dataclass
class Data:
    codes: np.ndarray  # (n, f) uint8 quintile codes, MISSING for null
    names: list[str]
    edges: dict[str, np.ndarray]
    y: np.ndarray  # outcome minus the same-day average
    month: np.ndarray  # month index
    year_of_month: np.ndarray  # year for each month index
    disc: np.ndarray
    val: np.ndarray


def load(
    atlas_dir: str, label: str, periods: dict = PERIODS, min_dvol: float = 5e6, every: int = 1
) -> tuple[pl.DataFrame, list[str]]:
    """Common stocks, liquid, in the given periods; market features joined; y = label minus the same-day mean."""
    lo, hi = min(p[0] for p in periods.values()), max(p[1] for p in periods.values())
    st = pl.scan_parquet(f"{atlas_dir}/stocks/*.parquet")
    feats = [c for c in st.collect_schema().names() if c.startswith("s_")]
    df = (st.filter((C("kind") == "common") & (C("as_of") >= lo) & (C("as_of") <= hi) & (C("dvol20") >= min_dvol) & (C("raw_close") >= 5))
          .select("entity_id", "as_of", *feats, C(label).alias("_y")).filter(C("_y").is_finite())
          .with_columns((C("_y") - C("_y").mean().over("as_of")).alias("y")).drop("_y"))  # fmt: skip
    if every > 1:
        days = (
            df.select("as_of")
            .unique()
            .sort("as_of")
            .with_row_index("_d")
            .filter(C("_d") % every == 0)
            .select("as_of")
        )
        df = df.join(days, on="as_of", how="semi")
    m = pl.scan_parquet(f"{atlas_dir}/market.parquet")
    mf = [c for c in m.collect_schema().names() if c.startswith("m_")]
    df = df.join(m.select("as_of", *mf), on="as_of", how="left").collect()
    return purge(df, label, periods), feats + mf


def purge(df: pl.DataFrame, label: str, periods: dict) -> pl.DataFrame:
    """Drop rows whose outcome window crosses the end of their period (DESIGN: purge / embargo), e.g. a 10-session
    outcome from the last days of the discovery period would use validation-period prices."""
    h = int("".join(ch for ch in label if ch.isdigit()) or 0)
    if h == 0:
        return df
    gap = timedelta(days=int(h * 1.5) + 3)  # sessions -> calendar days, generous
    keep = pl.lit(False)
    for lo, hi in periods.values():
        keep = keep | ((C("as_of") >= lo) & (C("as_of") <= hi - gap))
    return df.filter(keep)


def encode(df: pl.DataFrame, names: list[str], periods: dict = PERIODS) -> Data:
    d0, d1 = periods["discovery"]
    v0, v1 = periods["validation"]
    asof = df["as_of"]
    disc = ((asof >= d0) & (asof <= d1)).to_numpy()
    val = ((asof >= v0) & (asof <= v1)).to_numpy()
    codes = np.full((df.height, len(names)), MISSING, dtype=np.uint8)
    edges: dict[str, np.ndarray] = {}
    keep: list[int] = []
    for j, n in enumerate(names):
        x = df[n].cast(pl.Float64).to_numpy()
        fin = np.isfinite(x)
        xd = x[disc & fin]
        if len(xd) < 1000:
            continue
        e = np.unique(np.quantile(xd, np.linspace(0, 1, Q + 1)[1:-1]))
        if len(e) < 2:
            continue
        codes[fin, j] = np.searchsorted(e, x[fin], side="right").astype(np.uint8)
        edges[n] = e
        keep.append(j)
    ym = (asof.dt.year().cast(pl.Int64) * 12 + asof.dt.month().cast(pl.Int64) - 1).to_numpy()
    months = np.unique(ym)
    mi = np.searchsorted(months, ym)
    return Data(
        codes[:, keep],
        [names[j] for j in keep],
        edges,
        df["y"].to_numpy().astype(np.float64),
        mi,
        months // 12,
        disc,
        val,
    )


def _monthly(s: np.ndarray, c: np.ndarray, yom: np.ndarray) -> dict:
    """s, c: sums / counts per month (1-D). t-value over months with enough rows; share of good months / years."""
    ok = c >= MIN_ROWS_MONTH
    n = int(c.sum())
    if ok.sum() < 3:
        return {
            "n": n,
            "mean": float(s.sum() / n) if n else np.nan,
            "months": int(ok.sum()),
            "t": np.nan,
            "m_pos": np.nan,
            "y_pos": np.nan,
            "years": 0,
        }
    mm = s[ok] / c[ok]
    sd = mm.std(ddof=1)
    t = mm.mean() / (sd / np.sqrt(len(mm))) if sd > 0 else np.nan
    yrs = yom[ok]
    yu = np.unique(yrs)
    ymean = np.array([mm[yrs == y].mean() for y in yu])
    return {"n": n, "mean": float(s.sum() / n), "months": int(ok.sum()), "t": float(t), "m_pos": float((mm > 0).mean()),
            "y_pos": float((ymean > 0).mean()), "years": int(len(yu))}  # fmt: skip


def stats(d: Data, mask: np.ndarray) -> dict:
    nm = len(d.year_of_month)
    s = np.bincount(d.month[mask], weights=d.y[mask], minlength=nm)
    c = np.bincount(d.month[mask], minlength=nm)
    out = _monthly(s, c, d.year_of_month)
    out["hit"] = float((d.y[mask] > 0).mean()) if mask.any() else np.nan
    return out


def cond_text(name: str, lo: int, hi: int, edges: dict[str, np.ndarray]) -> str:
    e = edges[name]
    a = "−∞" if lo == 0 else f"{e[lo - 1]:.3g}"
    b = "+∞" if hi >= len(e) else f"{e[hi]:.3g}"
    return f"{name} が {a}〜{b}（{lo + 1}〜{hi + 1}/{len(e) + 1}分位）"


def _best_conditions(d: Data, sub: np.ndarray, skip: set[int], min_months: int, min_year_share: float,
                     stock_only: bool) -> list[tuple[float, int, int, int]]:  # fmt: skip
    nm = len(d.year_of_month)
    ys, mo = d.y[sub], d.month[sub]
    out = []
    for j in range(len(d.names)):
        if j in skip or (stock_only and d.names[j].startswith("m_")):
            continue
        cj = d.codes[sub, j]
        ok = cj != MISSING
        key = cj[ok].astype(np.int64) * nm + mo[ok]
        s = np.bincount(key, weights=ys[ok], minlength=Q * nm)[: Q * nm].reshape(Q, nm)
        c = np.bincount(key, minlength=Q * nm)[: Q * nm].reshape(Q, nm)
        for lo, hi in CONDS:
            r = _monthly(s[lo : hi + 1].sum(0), c[lo : hi + 1].sum(0), d.year_of_month)
            if r["months"] < min_months or not np.isfinite(r["t"]) or r["y_pos"] < min_year_share:
                continue
            out.append((r["t"], j, lo, hi))
    return sorted(out, reverse=True)


def singles(d: Data, min_months: int = 60) -> list[tuple[float, int, int, int]]:
    """Stock features alone (market features cannot pick stocks), best t-value first."""
    return _best_conditions(d, d.disc, set(), min_months, 0.7, stock_only=True)


def stepwise(
    d: Data,
    start: tuple[int, int, int],
    steps: int = 5,
    min_months: int = 36,
    min_year_share: float = 0.7,
) -> list[dict]:
    mask = (d.codes[:, start[0]] >= start[1]) & (d.codes[:, start[0]] <= start[2])
    path = [
        {
            "step": 1,
            "cond": cond_text(d.names[start[0]], start[1], start[2], d.edges),
            "disc": stats(d, mask & d.disc),
            "val": stats(d, mask & d.val),
        }
    ]
    used = {start[0]}
    for step in range(1, steps):
        cand = _best_conditions(
            d, mask & d.disc, used, min_months, min_year_share, stock_only=False
        )
        if not cand or cand[0][0] <= path[-1]["disc"]["t"]:
            break  # nothing improves the t-value any more
        _, j, lo, hi = cand[0]
        used.add(j)
        mask = mask & (d.codes[:, j] >= lo) & (d.codes[:, j] <= hi)
        path.append(
            {
                "step": step + 1,
                "cond": cond_text(d.names[j], lo, hi, d.edges),
                "disc": stats(d, mask & d.disc),
                "val": stats(d, mask & d.val),
            }
        )
    return path


def pairs(d: Data, min_months: int = 48, min_year_share: float = 0.8, top: int = 300) -> list[dict]:
    """Every pair (at least one stock feature) as a 5x5 grid on discovery rows, ranked by the monthly t-value."""
    nm = len(d.year_of_month)
    ys, mo = d.y[d.disc], d.month[d.disc]
    cd = d.codes[d.disc]
    found: list[tuple[float, int, int, int, int, float, int, int]] = []
    F = len(d.names)
    yom = d.year_of_month
    for i in range(F):
        ci = cd[:, i].astype(np.int64)
        oki = ci != MISSING
        for j in range(i + 1, F):
            if d.names[i].startswith("m_") and d.names[j].startswith("m_"):
                continue
            cj = cd[:, j].astype(np.int64)
            ok = oki & (cj != MISSING)
            key = (ci[ok] * Q + cj[ok]) * nm + mo[ok]
            s = np.bincount(key, weights=ys[ok], minlength=Q * Q * nm).reshape(Q * Q, nm)
            c = np.bincount(key, minlength=Q * Q * nm).reshape(Q * Q, nm)
            okm = c >= MIN_ROWS_MONTH
            k = okm.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mm = np.where(okm, s / np.where(c > 0, c, 1), np.nan)
                mean = np.nanmean(mm, 1)
                sd = np.nanstd(mm, 1, ddof=1)
                t = mean / (sd / np.sqrt(k))
            for cell in np.flatnonzero((k >= min_months) & (t > 3)):
                row = mm[cell]
                f = np.isfinite(row)
                yrs = yom[f]
                ypos = np.mean([row[f][yrs == y].mean() > 0 for y in np.unique(yrs)])
                if ypos < min_year_share:
                    continue
                found.append(
                    (
                        float(t[cell]),
                        i,
                        j,
                        int(cell // Q),
                        int(cell % Q),
                        float(s[cell].sum() / c[cell].sum()),
                        int(c[cell].sum()),
                        int(k[cell]),
                    )
                )
    found.sort(reverse=True)
    out = []
    for t, i, j, a, b, mean, n, k in found[:top]:
        m = (d.codes[:, i] == a) & (d.codes[:, j] == b)
        out.append({"a": cond_text(d.names[i], a, a, d.edges), "b": cond_text(d.names[j], b, b, d.edges), "fa": d.names[i], "fb": d.names[j],
                    "disc": {"t": t, "mean": mean, "n": n, "months": k}, "val": stats(d, m & d.val)})  # fmt: skip
    return out
