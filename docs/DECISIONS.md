# DECISIONS.md — ADR log

Format: `ADR-NNNN | date | status | title`, then Context / Decision / Consequences.
Every deviation from DESIGN.md and every production change is recorded here.

---

## ADR-0001 | 2026-10-23 | accepted | Deployment Option B + Cloudflare R2

### Context
DESIGN.md §2 offers Option A (private repo, ~2,000 Actions min/month on Free plan) and Option B (public code-only repo, data in private storage).
Estimated workload (options snapshots for ~200 underlyings twice daily, S&P500+NDX+TOPIX500 prices, weekly retrain/HPO) likely exceeds the private-repo budget.
Verified 2026-10-23 (docs.github.com): standard GitHub-hosted runners are free for public repositories; private Free plan includes 2,000 min/month; job time limit 6h; 20 concurrent jobs on Free plan.

### Decision
- Option B: public repo `akyoshida-gif/market-weather` containing code + config only.
- All data layers (raw/silver/gold/ledger/scores/reports/registry) live in a private Cloudflare R2 bucket `market-weather-data`. Local dev uses `LocalFS` backend.
- Secrets only in GitHub Actions secrets. No personal identifiers (e.g. owner email) in code; SEC User-Agent contact comes from a secret.
- R2 API token scoped to the single bucket (Object Read & Write), never account-wide.

### Cost guardrails (R2 has no hard spending cap; only budget alerts — verified 2026-10-23)
- Cloudflare budget alert at USD 1 (owner configures).
- App-side guard in the storage backend: track bucket size and monthly write/read op counts in `registry/storage_usage.jsonl`; refuse non-essential writes above soft limits (8 GB storage, 800k Class A ops/month, 8M Class B ops/month) and open an incident. Ledger writes are exempt below the hard free-tier limit.
- Compaction (DESIGN §6.3) keeps object counts low.

### Consequences
- Public scheduled workflows are disabled after ~60 days of repo inactivity (unverified in official docs) → keepalive workflow required.
- Fork PRs must not have access to secrets (default GitHub behavior; do not use `pull_request_target`).

---

## ADR-0002 | 2026-10-23 | accepted | Phase 0 implementation choices

- Python pinned to 3.12 (`.python-version`); `requires-python >=3.11,<3.14` for library compatibility (local machine has 3.14; uv provisions 3.12).
- `GitRepoFS` backend not implemented: Option B (ADR-0001) never stores data in the repo. Backends: `LocalFS`, `S3Compatible` (R2).
- `raw/` and `ledger/forecasts/` are write-once at the storage layer (`GuardedStorage`), enforcing P2/P3.
- Run summaries are written as one object per run under `registry/runs/dt=.../` instead of appending to a single `runs.jsonl` (object stores have no append; avoids read-modify-write races). A compaction job may later build `runs.jsonl`.
- Keepalive implemented by re-enabling workflows via `gh workflow enable` (no dummy commits).

---

## ADR-0003 | 2026-10-23 | accepted | R2 bucket name is `market-weather-actions`

The bucket was created as `market-weather-actions` (ADR-0001 said `market-weather-data`). R2 buckets cannot be renamed; the name carries no meaning, so config follows the actual bucket.

---

## ADR-0004 | 2026-10-23 | accepted | US options snapshot design

- Provider: yfinance (unofficial). Raw = provider DataFrame serialized to Parquet (original HTTP bytes are not exposed by yfinance); one raw object per snapshot per table to minimise R2 Class A ops.
- Watchlist: initial hand-picked list (`config/universe/options_watchlist.yaml`, ~110 symbols + US theme members). DESIGN §7 "top ~200 by option volume" will be derived from collected data later.
- Expiries: all within `max_dte_days` (200) + first beyond (term structure). Full chain beyond that is skipped to bound runtime.
- `as_of` = last completed XNYS session at capture; `published_at` = capture time. A snapshot (as_of, pre|post) is captured once; re-runs skip unless `--force`.
- Observed 2026-10-23 06:42 UTC (02:42 ET): Yahoo returned OI≈0 and IV=1e-5 placeholders for nearly all contracts (off-hours reset, cause unverified). Quality metrics are computed per snapshot; if >50% OI zero or IV placeholder, silver is written as `*_degraded.parquet` and the snapshot is not considered done (a later run may re-capture). Silver nulls placeholder IV; raw keeps provider values.
- Snapshot timing (12:30 / 00:30 UTC) to be re-evaluated against observed quality after the first week.

---

## ADR-0005 | 2026-10-23 | accepted | Owner decisions on source ToS and paid plans

- JPX data (short positions, short ratio, margin, investor type): owner judges private analysis without redistribution is not 「二次利用」; use with ≤1 fetch per file per day, never redistribute, never commit data to the public repo.
- J-Quants: stay on free tier for now (no paid plan). JPX per-issue margin via PDF parse / JSF CSV.
- Stooq dropped (anti-bot + robots Disallow). US EOD prices: Tiingo free (primary, token required) + yfinance (fallback).
- Yahoo/yfinance for options: kept running (history accumulates). moomoo OpenAPI under evaluation as a licensed alternative (owner question pending).

---

## ADR-0006 | 2026-10-23 | accepted | moomoo OpenAPI prepared as options provider (disabled)

- Owner prefers a licensed source over Yahoo for options but the moomoo account is not funded yet; yfinance stays primary meanwhile (history must not gap).
- `MoomooProvider` implements the same ChainProvider protocol; field mapping is UNVERIFIED until a live OpenD test.
- OpenD holds brokerage credentials → run only on owner PC / self-hosted runner, never on GitHub-hosted runners or with credentials in GitHub Secrets.
- When enabled: run both providers in parallel for ≥2 weeks, compare quality metrics, then switch (ADR).

---

## ADR-0007 | 2026-10-23 | accepted | Phase 1 US/macro ingest details

- Package `mw.ingest.global` renamed to `mw.ingest.macro` (`global` is a Python keyword).
- Small time series (vol indices, FRED) stored as year-partitioned silver files (`silver/{table}/market=.../year=YYYY/data.parquet`), upserted by (entity_id, as_of); large cross-sections (Reg SHO, short interest, options) stored per date.
- FRED: latest values with per-series conservative publish lag; ALFRED vintages deferred (revisions are rare for these series).
- FRED request errors are re-raised without URL (the URL contains the API key).

---

## ADR-0008 | 2026-10-26 | accepted | US universe = all listed common stocks; news/catalyst and signal modules; separate UI

Owner direction (2026-10-26): focus on US incl. small caps / penny stocks; JP scope unchanged; add news collection; add buy/sell point indicators; the small-cap/news screen is a separate UI from the forecasting dashboard.
- US universe: Nasdaq Trader symbol directory (nasdaqlisted + otherlisted), common stocks only (not ETF/test issue/warrant/right/unit/preferred/notes; ADRs kept) + configured ETFs (`config/universe/us_etfs.yaml`). Daily PIT snapshot in `silver/universe_membership/market=US/date=...`. Nasdaq financial status (deficient/delinquent/bankrupt) kept as `listing_issue`.
- JP universe: JPX data_j.xlsx, TOPIX Core30+Large70+Mid400 + ETFs (`config/universe/jp_etfs.yaml`); monthly snapshot keyed by list date.
- Prices: yfinance batch download (chunks of 200). `close` is split-adjusted as of download, `adj_close` adds dividends; raw keeps every vintage as Parquet (never pickle). Bars of sessions not yet closed (+30 min) are dropped. Silver year-partitioned, upsert by (entity_id, as_of).
- Planned next: news/catalyst collection (SEC EDGAR filings, press-release wires, trading halts) and rule-based buy/sell signals logged and scored like forecasts. Real-time pre-market detection deferred until an event study shows value (DESIGN §0.3 non-goal stays for now).
