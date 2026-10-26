"""SEC EDGAR filings as catalyst events (DESIGN §8.1 sec_8k etc.; ADR-0009).

Live: Atom "latest filings" per form type
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=<FORM>&owner=include&count=100&start=N&output=atom
  `updated` = acceptance time (second precision, ET offset); summary carries 8-K item numbers.
Backfill: submissions JSON per CIK (https://data.sec.gov/submissions/CIK##########.json),
  `acceptanceDateTime` in UTC, `items`, last ~1,000 filings per company.
CIK -> ticker: https://www.sec.gov/files/company_tickers_exchange.json (snapshot daily).
Fair access: <= 10 req/s with a User-Agent containing contact info (env SEC_USER_AGENT).

PIT: EDGAR disseminates filings accepted after 17:30 ET (22:00 for Forms 3/4/5) on the next
business day from 06:00 ET -> published_at = dissemination time, not acceptance time.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from mw.core.calendars import is_session, next_session
from mw.core.http import HttpClient

SOURCE_ID = "sec_filings"
SOURCE_VERSION = "0.1.0"
NY = ZoneInfo("America/New_York")
ATOM = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form}"
    "&owner=include&count=100&start={start}&output=atom"
)
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"

# Form prefixes polled live (prefix match on EDGAR side). Catalysts for small caps:
# 8-K/6-K events & press releases, S-1/S-3/424B offerings (dilution), SCHEDULE 13D/13G stakes,
# Form 4 insider trades, 25-NSE delisting, DEF 14A (reverse split votes), 10-Q/10-K.
LIVE_FORMS = ["8-K", "6-K", "S-1", "S-3", "424B", "SCHEDULE 13D", "SCHEDULE 13G", "4", "25-NSE",
              "DEF 14A", "10-Q", "10-K", "20-F", "F-1", "F-3"]  # fmt: skip

ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.S)
TAG_RE = {
    k: re.compile(rf"<{k}[^>]*>(.*?)</{k}>", re.S) for k in ("title", "summary", "updated", "id")
}
TERM_RE = re.compile(r'<category[^>]*term="([^"]+)"')
TITLE_RE = re.compile(r"^(?P<form>.+?) - (?P<name>.+) \((?P<cik>\d{10})\) \((?P<role>[^)]+)\)$")
ITEM_RE = re.compile(r"Item (\d+\.\d+)")


class SecClient:
    def __init__(self, http: HttpClient | None = None) -> None:
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RuntimeError("SEC_USER_AGENT not set (SEC requires name + contact email)")
        self.http = http or HttpClient(rate_per_host=5.0, user_agent=ua)

    def get(self, url: str) -> bytes:
        resp = self.http.get(url)
        resp.raise_for_status()
        return resp.content


def dissemination_time(accepted: datetime, form: str) -> datetime:
    """When the public could see the filing (UTC)."""
    local = accepted.astimezone(NY)
    cutoff = time(22, 0) if form.split("/")[0] in {"3", "4", "5"} else time(17, 30)
    d = local.date()
    if is_session("US", d) and time(6, 0) <= local.time() <= cutoff:
        return accepted.astimezone(UTC)
    early = is_session("US", d) and local.time() < time(6, 0)
    nxt = d if early else next_session("US", d)
    return datetime.combine(nxt, time(6, 0), tzinfo=NY).astimezone(UTC)


def parse_atom(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    out: list[dict[str, Any]] = []
    for body in ENTRY_RE.findall(text):
        get = {k: (m.group(1).strip() if (m := r.search(body)) else "") for k, r in TAG_RE.items()}
        t = TITLE_RE.match(html.unescape(get["title"]))
        if not t:
            continue
        summary = html.unescape(get["summary"])
        acc = get["id"].rsplit("=", 1)[-1]
        term = TERM_RE.search(body)
        out.append({
            "accession": acc,
            "form": term.group(1) if term else t.group("form"),
            "cik": int(t.group("cik")),
            "company": t.group("name"),
            "role": t.group("role"),
            "accepted_at": datetime.fromisoformat(get["updated"]).astimezone(UTC),
            "items": ",".join(dict.fromkeys(ITEM_RE.findall(summary))),
        })  # fmt: skip
    return out


OLDER = "https://data.sec.gov/submissions/{name}"


def older_files(data: bytes, since: date) -> list[str]:
    """Names of the extra submissions files (filings older than the ~1,000 most recent) that
    overlap [since, today]."""
    d = json.loads(data)
    files = d.get("filings", {}).get("files", []) or []
    return [f["name"] for f in files if str(f.get("filingTo", "9999")) >= since.isoformat()]


def parse_older(data: bytes, cik: int, company: str, since: date) -> list[dict[str, Any]]:
    """Extra submissions file: same column arrays as filings.recent, at the top level."""
    return _rows(json.loads(data), cik, company, since)


def parse_submissions(data: bytes, since: date) -> list[dict[str, Any]]:
    d = json.loads(data)
    return _rows(d["filings"]["recent"], int(d["cik"]), d.get("name", ""), since)


def _rows(r: dict[str, Any], cik: int, company: str, since: date) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, acc in enumerate(r["accessionNumber"]):
        if r["filingDate"][i] < since.isoformat():
            continue
        out.append({
            "accession": acc,
            "form": r["form"][i],
            "cik": cik,
            "company": company,
            "role": "Filer",
            "accepted_at": datetime.fromisoformat(
                r["acceptanceDateTime"][i].replace("Z", "+00:00")
            ),
            "items": r["items"][i] or "",
        })  # fmt: skip
    return out


def ticker_map(data: bytes) -> pl.DataFrame:
    d = json.loads(data)
    df = pl.DataFrame(d["data"], schema=d["fields"], orient="row")
    # one CIK may list several tickers; keep the first (primary) per CIK on a listed exchange
    return (
        df.filter(pl.col("exchange").is_in(["Nasdaq", "NYSE", "CBOE", "TXSE"]))
        .unique(subset=["cik"], keep="first", maintain_order=True)
        .select(pl.col("cik").cast(pl.Int64), "ticker", "exchange")
    )


def to_frame(rows: list[dict[str, Any]], tickers: pl.DataFrame) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows).with_columns(pl.col("cik").cast(pl.Int64))
    df = df.join(tickers, on="cik", how="left")
    pub = [dissemination_time(a, f) for a, f in zip(df["accepted_at"], df["form"], strict=True)]
    return df.with_columns(
        pl.Series("published_at", pub, dtype=pl.Datetime("us", "UTC")),
        pl.when(pl.col("ticker").is_not_null())
        .then(pl.lit("US:") + pl.col("ticker"))
        .otherwise(pl.lit("SEC:") + pl.col("cik").cast(pl.String))
        .alias("entity_id"),
        pl.Series("as_of", [p.astimezone(NY).date() for p in pub], dtype=pl.Date),
    )


def poll_live(
    sec: SecClient, forms: list[str] = LIVE_FORMS, pages: int = 2
) -> tuple[list[dict[str, Any]], list[bytes], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    raws: list[bytes] = []
    errors: dict[str, str] = {}
    for form in forms:
        for p in range(pages):
            try:
                data = sec.get(ATOM.format(form=form.replace(" ", "%20"), start=p * 100))
            except Exception as e:
                errors[f"{form}:{p}"] = f"{type(e).__name__}"
                break
            raws.append(data)
            got = parse_atom(data)
            rows.extend(got)
            if len(got) < 100:
                break
    return rows, raws, errors


def since_days(n: int) -> date:
    return (datetime.now(UTC) - timedelta(days=n)).date()
