"""Build the dynamics layer (t_* columns, see mw.research.dynamics) next to an existing atlas.
Research only.

Usage: python -m jobs.dyn_build --px PX --uni UNI --atlas DIR
Writes DIR/dyn/part_NNN.parquet with the same rows (entity_id, as_of) as
DIR/stocks/part_NNN.parquet.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import polars as pl

from mw.research import dynamics

C = pl.col


def main() -> None:
    p = argparse.ArgumentParser()
    for k in ("px", "uni", "atlas"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.atlas) / "dyn"
    out.mkdir(parents=True, exist_ok=True)
    uni = pl.read_parquet(a.uni).select("entity_id", "kind")
    common = uni.filter(C("kind") == "common").select("entity_id")
    ohlcv = ["entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"]
    rk = dynamics.rank_dynamics(
        pl.scan_parquet(a.px).select(ohlcv).join(common.lazy(), on="entity_id").collect()
    )
    print(f"順位 {rk.height:,} 行 {time.time() - t0:.0f}秒", flush=True)
    spy = dynamics.market_runs(
        pl.scan_parquet(a.px).filter(C("entity_id") == "US:SPY").select("as_of", "close").collect()
    )
    parts = sorted((Path(a.atlas) / "stocks").glob("part_*.parquet"))
    for f in parts:
        o = out / f.name
        if o.exists():
            continue
        keys = pl.read_parquet(f, columns=["entity_id", "as_of"])
        ents = keys["entity_id"].unique()
        px = (
            pl.scan_parquet(a.px)
            .filter(C("entity_id").is_in(ents.to_list()))
            .select(ohlcv)
            .collect()
        )
        d = dynamics.stock_dynamics(px, spy)
        d = keys.join(d, on=["entity_id", "as_of"], how="left").join(
            rk, on=["entity_id", "as_of"], how="left"
        )
        d.write_parquet(o)
        print(f"  {f.name} {d.height:,} 行・{d.width} 列 {time.time() - t0:.0f}秒", flush=True)
        del px, d
        gc.collect()


if __name__ == "__main__":
    main()
