"""Build the dashboard HTML (data embedded) from silver tables. Read-only.

Usage: python -m jobs.build_dashboard --out path/to/dashboard.html
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from mw.core.config import REPO_ROOT, load_settings
from mw.core.storage import open_storage
from mw.ingest.us.options_chain import load_watchlist
from mw.report import dashboard as D
from mw.signals import rules
from mw.strategy import engine, select

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mw"
    / "report"
    / "templates"
    / "dashboard.html"
)


def latest_universe(storage: Any, market: str) -> pl.DataFrame:
    keys = sorted(storage.list(f"silver/universe_membership/market={market}/"))
    return storage.read_parquet(keys[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    storage = open_storage(load_settings().storage)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {"generated_at": now.isoformat(), "markets": {}, "item_names": D.ITEM_NAMES,
                               "signal_names": {k: v[0] for k, v in rules.SIGNALS.items()},
                               "signal_side": {k: v[1] for k, v in rules.SIGNALS.items()}, "horizon": rules.HORIZON}  # fmt: skip
    vol = D._read_years(storage, "vol_indices", "US")
    fred = D._read_years(storage, "fred", "GLOBAL")
    themes = yaml.safe_load((REPO_ROOT / "config/universe/themes.yaml").read_text(encoding="utf-8"))
    since60 = (now - timedelta(days=100)).date()
    filings, halts = D.news(storage, (now - timedelta(days=120)).date())

    for market in ("US", "JP"):
        uni = latest_universe(storage, market)
        prices = D._read_years(storage, "prices_daily", market)
        prices = prices.join(uni.select("entity_id", "kind"), on="entity_id", how="left")
        names = dict(zip(uni["entity_id"], uni["name"], strict=True))
        if market == "US":
            wl, _ = load_watchlist()
            sec_of = {
                f"US:{t}": th for th, members in themes.items() for t in members if "." not in t
            }
            detail = [f"US:{t}" for t in wl] + [f"US:{s}" for s in D.US_SECTORS]
            sectors = D.US_SECTORS
            min_dvol = 1_000_000.0
        else:
            sec_of = dict(zip(uni["entity_id"], uni["sector17"], strict=True))
            core = uni.filter(pl.col("size").is_in(["TOPIX Core30", "TOPIX Large70"]))
            detail = (
                core["entity_id"].to_list()
                + [f"JP:{s}" for s in D.JP_SECTORS]
                + ["JP:1306", "JP:1321"]
            )
            sectors = D.JP_SECTORS
            min_dvol = 100_000_000.0
        sp = D.signal_payload(prices, market, min_dvol)
        # strategies: backtest, adoption gate, today's picks (adopted only)
        ind = engine.indicators(
            prices.select("entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"),
            D.BENCH[market],
        )
        res = select.run(ind, market, D.BENCH[market])
        asof_d = ind["as_of"].max()
        strat_out, picks_out, adopted_keys = [], [], []
        for fam in res["families"]:
            key = fam["strategy"]["key"]
            item = {k: v for k, v in fam.items()}
            if fam["adopted"]:
                adopted_keys.append(key)
                item["detail"] = select.detail(res["trades"], key)
                for pk in select.picks(ind, res["strategies"][key], asof_d):
                    picks_out.append(
                        {
                            **pk,
                            "strategy": key,
                            "family": fam["family"],
                            "label": fam["strategy"]["label"],
                            "horizon": fam["strategy"]["horizon"],
                            "side": fam["strategy"]["side"],
                            "max_hold": fam["strategy"]["max_hold"],
                            "name": names.get(pk["id"], ""),
                        }
                    )
            strat_out.append(item)
        detail = list(dict.fromkeys(detail + [p["id"] for p in picks_out]))
        recent_ids = (
            res["trades"]
            .filter(pl.col("strategy").is_in(adopted_keys))
            .sort("exit_date")
            .tail(40)["entity_id"]
            .to_list()
            if adopted_keys
            else []
        )
        detail = list(dict.fromkeys(detail + recent_ids))
        detail = [d for d in detail if d in set(prices["entity_id"].unique().to_list())]
        ev = sp["events"].filter(pl.col("entity_id").is_in(detail))
        marks: dict[str, list[list[str]]] = {}
        for r in ev.iter_rows(named=True):
            marks.setdefault(r["entity_id"], []).append([r["as_of"].isoformat(), r["signal"]])
        m: dict[str, Any] = {
            "asof": sp["asof"],
            "bench": D.BENCH[market],
            "kpis": D.market_kpis(prices, market, vol if market == "US" else pl.DataFrame(), fred),
            "rrg": D.rrg(prices, market, sectors),
            "signal_stats": sp["stats"],
            "signal_stats_year": sp["stats_year"],
            "signals_today": [
                {**r, "name": names.get(r["id"], ""), "sec": sec_of.get(r["id"], "")}
                for r in sp["today"]
            ],
            "bars": D.stock_bars(prices, detail),
            "marks": marks,
            "names": {d: names.get(d, "") for d in detail},
            "sectors": {d: (sectors.get(d.split(":")[1]) or sec_of.get(d, "")) for d in detail},
            "universe_size": int((uni["kind"] == "common").sum()),
            "strategies": strat_out,
            "picks": picks_out,
        }
        tt = (
            res["trades"].filter(
                pl.col("strategy").is_in(adopted_keys) & pl.col("entity_id").is_in(detail)
            )
            if adopted_keys
            else pl.DataFrame()
        )
        m["trades"] = {}
        for r in tt.iter_rows(named=True) if tt.height else []:
            m["trades"].setdefault(r["entity_id"], []).append(
                [
                    r["entry_date"].isoformat(),
                    r["exit_date"].isoformat(),
                    r["strategy"].split("|")[0],
                    r["entry"],
                    r["exit"],
                    r["stop"],
                    r["target"],
                    round(r["ret"], 4),
                    r["reason"],
                ]
            )
        m["family_labels"] = {f["family"]: f["strategy"]["label"] for f in strat_out}
        if market == "US":
            m["options"] = {k: v for k, v in D.options_summary(storage).items() if k in detail}
            m["short_vol"] = D.finra_short(storage, detail, since60)
            m["short_interest"] = D.short_interest(storage, detail)
            f = filings.filter(pl.col("entity_id").is_in(detail)) if filings.height else filings
            m["filings"] = {}
            for r in f.sort("published_at", descending=True).iter_rows(named=True):
                m["filings"].setdefault(r["entity_id"], []).append(
                    [r["published_at"].isoformat(), r["form"], r["items"] or ""]
                )
            h = halts.filter(pl.col("entity_id").is_in(detail)) if halts.height else halts
            m["halts"] = {}
            for r in h.sort("halted_at", descending=True).iter_rows(named=True):
                m["halts"].setdefault(r["entity_id"], []).append(
                    [
                        r["halted_at"].isoformat(),
                        r["reason"],
                        r["resume_trade_at"].isoformat() if r["resume_trade_at"] else None,
                    ]
                )
        else:
            m["jp_short"] = D.jp_short_positions(storage, detail)
            m["jsf"] = D.jsf_latest(storage, detail)
        payload["markets"][market] = m
        print(market, "detail", len(detail), "picks", len(picks_out), "adopted", adopted_keys)

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", data.replace("</", "<\\/")
    )
    Path(args.out).write_text(html, encoding="utf-8")
    print("wrote", args.out, round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
