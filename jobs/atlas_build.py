"""Build the state atlas from local parquet copies (research only; output stays outside the repo).

Usage: python -m jobs.atlas_build --px PX --uni UNI --vol VOL --fred FRED --out DIR
(local parquet copies of prices_daily, the universe, vol_indices and fred).
Writes DIR/stocks/part_NNN.parquet (one per entity chunk) and DIR/market.parquet.
Stored rows: actual (not split-adjusted) close >= 3, 20-day dollar volume >= $2M,
about 1 year of history,
no data error in the last 20 sessions.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import polars as pl

from mw.research import atlas

C = pl.col


def main() -> None:
    p = argparse.ArgumentParser()
    for k in ("px", "uni", "vol", "fred", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--chunk", type=int, default=250)
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.out)
    (out / "stocks").mkdir(parents=True, exist_ok=True)
    uni = pl.read_parquet(a.uni).select("entity_id", "kind")
    ohlcv = ["entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"]
    px = pl.scan_parquet(a.px).select(ohlcv).join(uni.lazy(), on="entity_id", how="inner").collect()
    ents = px["entity_id"].unique().sort().to_list()
    print(f"{len(ents)} 銘柄・{px.height:,} 行", flush=True)
    for i in range(0, len(ents), a.chunk):
        f = out / "stocks" / f"part_{i // a.chunk:03d}.parquet"
        if f.exists():
            continue
        part = px.filter(C("entity_id").is_in(ents[i : i + a.chunk]))
        kind = part.select("entity_id", "kind").unique()
        df = atlas.past_relative(atlas.stock_features(part.drop("kind")))
        df = df.drop("splits")
        df = atlas.path_labels(df)
        df = df.filter(
            (C("raw_close") >= 3) & (C("dvol20") >= 2e6) & C("s_ret250").is_not_null() & ~C("bad20")
        )
        df = df.join(kind, on="entity_id", how="left").drop("bad20")
        fl = [c for c, t in df.schema.items() if t == pl.Float64]
        df.with_columns([C(c).cast(pl.Float32) for c in fl]).write_parquet(f)
        done = i + len(ents[i : i + a.chunk])
        print(f"  {done}/{len(ents)} {df.height:,} 行 {time.time() - t0:.0f}秒", flush=True)
        del df, part
        gc.collect()
    cols = ["entity_id", "as_of", "kind", "s_dma50", "s_dma200", "s_dhi250", "s_dlo250",
            "s_ret1", "s_ret5", "s_ret20", "s_vr20", "s_streak", "s_vol20"]  # fmt: skip
    st = pl.scan_parquet(str(out / "stocks" / "*.parquet")).select(cols).collect()
    common = st.filter((C("kind") == "common") | (C("entity_id") == "US:SPY"))
    m = atlas.market_features(common, pl.read_parquet(a.vol), pl.read_parquet(a.fred))
    m.write_parquet(out / "market.parquet")
    print(f"市場 {m.height} 日・{len(m.columns)} 列。合計 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
