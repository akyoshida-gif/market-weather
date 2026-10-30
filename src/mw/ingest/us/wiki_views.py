"""Daily English Wikipedia page views per listed company (attention), VALIDATION_PLAN_V3 §5.

Mapping ticker -> article: Wikidata (stock exchange NYSE/Nasdaq with a ticker qualifier, English article).
Views: Wikimedia REST API per article, daily, users only (no bots), from 2015-07-01.
published_at rule: a day's count is complete after midnight UTC -> usable from the next day
(day + 1, 12:00 UTC); the views of day D are therefore used at the close of D+1 at the earliest.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import quote

import polars as pl

UA = "MarketWeatherResearch/0.1 (https://github.com/akyoshida-gif/market-weather)"
SPARQL = "https://query.wikidata.org/sparql"
QUERY = ("SELECT ?ticker ?article WHERE { ?item p:P414 ?st . ?st ps:P414 ?ex ; pq:P249 ?ticker . "
         "VALUES ?ex { wd:Q13677 wd:Q82059 } ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }")  # fmt: skip
VIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/{s}/{e}"
FIRST = date(2015, 7, 1)


def parse_mapping(js: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for b in js.get("results", {}).get("bindings", []):
        t = b["ticker"]["value"].strip().upper().replace("-", ".")
        art = b["article"]["value"].rsplit("/wiki/", 1)[-1]
        rows.append({"entity_id": f"US:{t}", "article": art})
    df = pl.DataFrame(rows, schema={"entity_id": pl.String, "article": pl.String})
    # one article per ticker (the shortest title is usually the company itself)
    return (
        df.with_columns(pl.col("article").str.len_chars().alias("_l"))
        .sort(["entity_id", "_l"])
        .unique("entity_id", keep="first")
        .drop("_l")
    )


def views_url(article: str, start: date, end: date) -> str:
    return VIEWS.format(
        a=quote(article, safe=""), s=start.strftime("%Y%m%d"), e=end.strftime("%Y%m%d")
    )


def parse_views(js: dict[str, Any], entity_id: str) -> pl.DataFrame:
    items = js.get("items", [])
    return pl.DataFrame(
        {
            "entity_id": [entity_id] * len(items),
            "as_of": [i["timestamp"][:8] for i in items],
            "views": [int(i["views"]) for i in items],
        },
        schema={"entity_id": pl.String, "as_of": pl.String, "views": pl.Int64},
    ).with_columns(pl.col("as_of").str.to_date("%Y%m%d"))


def published_at(as_of: pl.Expr) -> pl.Expr:
    return (as_of.cast(pl.Datetime("us")) + pl.duration(days=1, hours=12)).dt.replace_time_zone(
        "UTC"
    )
