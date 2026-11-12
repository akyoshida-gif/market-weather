"""Count splits whose step is still missing from the history (before / after atlas.repair_splits).
Usage: python scripts/atlas/check_splits.py <px_us.parquet>"""

import sys

import numpy as np
import polars as pl

from mw.research import atlas

sys.stdout.reconfigure(encoding="utf-8")
px = pl.read_parquet(
    sys.argv[1], columns=["entity_id", "as_of", "open", "high", "low", "close", "volume", "splits"]
)
s = px["splits"].fill_null(0)
ents = px.filter((s > 0) & (s.log().abs() >= atlas.SPLIT_MIN))["entity_id"].unique()
px = px.filter(pl.col("entity_id").is_in(ents.implode())).sort("entity_id", "as_of")


def count(df: pl.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, g in df.group_by("entity_id", maintain_order=True):
        c = g["close"].to_numpy().astype(float)
        sp = g["splits"].fill_null(0).to_numpy().astype(float)
        for d in np.flatnonzero(sp > 0):
            r = np.log(sp[d])
            if abs(r) < atlas.SPLIT_MIN:
                continue
            key = "反映済み" if atlas.split_step(c, d, r) is None else "未反映"
            out[key] = out.get(key, 0) + 1
    return out


print("補修の前：", count(px))
print("補修の後：", count(atlas.repair_splits(px)))
