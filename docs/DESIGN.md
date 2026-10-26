# DESIGN.md — Market Weather: probabilistic forecasting system for US & JP equities

> Audience: Claude Code (AI implementer). Human readability is NOT a goal.
> Owner language: Japanese. Talk to the owner in Japanese; code, identifiers, comments in English.
> This file is the single source of truth. Keep `docs/DECISIONS.md` (ADR log) for every deviation.

---

## 0. Mission, principles, non-goals

### 0.1 Mission
Build an autonomous, self-improving system that:
1. Collects as much price-relevant, numeric (and later textual) market data as possible for US and JP equities, daily, point-in-time correct.
2. Produces **probabilistic forecasts** ("weather forecasts") at multiple horizons: e.g. "P(up tomorrow)=0.57", "P(outperform sector over 20d)=0.61", "20d vol forecast", "P(drawdown < -10% in 20d)".
3. Logs every forecast immutably, scores it when the outcome matures, and **uses the scores to improve itself** (recalibration, re-weighting, retraining, feature discovery, model promotion/rollback).
4. Tracks and visualizes **cross-sector capital rotation** (which sectors/themes money is flowing into/out of), in both markets, with forecasts of rotation.

### 0.2 Hard principles (MUST)
- **P1 Point-in-time (PIT)**: every stored datum carries `as_of` (what period it describes), `published_at` (when the public could know it), `ingested_at` (when we stored it). Features may only use data with `published_at <= forecast_cutoff`. Violations are bugs of the highest severity.
- **P2 Immutable raw**: raw downloads are never modified. All cleaning is reproducible from raw.
- **P3 Immutable forecast ledger**: forecasts are appended before outcomes are known, hashed, never edited.
- **P4 Measure against baselines**: every model is judged vs. a naive baseline (base rate / climatology) and vs. the current champion. No model ships without beating both out-of-sample.
- **P5 Live shadow > backtest**: forecasts generated live (and logged) are the only truly out-of-sample evidence. Promotion decisions weight live shadow performance highest.
- **P6 Failure isolation**: one failing source/job must never block others. Every job is idempotent and re-runnable for any date.
- **P7 Free first, paid-ready**: start with free data/compute only; every source and backend sits behind an interface so paid replacements are drop-in.
- **P8 Compliance**: obey each source's ToS/robots, rate limits, and required headers (e.g. SEC User-Agent). Personal, non-redistributed use. Data never committed to a public location.
- **P9 Everything logged**: every experiment, model version, feature version, promotion, rollback, and data-quality incident is recorded in machine-readable registries.

### 0.3 Non-goals
- No order execution / brokerage integration.
- Not investment advice; outputs are probabilities with measured skill, nothing more.
- No intraday/tick-level system (daily granularity; extensible later).

---

## 1. Forecast targets (labels)

All labels computed strictly from data after the forecast cutoff. Horizons in trading days of the relevant exchange.

| id | entity | horizon | definition | type |
|---|---|---|---|---|
| `dir_1d` | stock, ETF, sector, index | 1d | sign(close[t+1]/close[t]-1) (adjusted) | binary |
| `dir_1d_xs` | stock | 1d | sign(stock ret − sector ret) over t→t+1 | binary |
| `gap_1d` | stock, index | 1d | sign(open[t+1]/close[t]-1) | binary |
| `ret_q_{h}` | stock, sector | 1,5,20,60 | return quantiles (5,25,50,75,95) | quantile regression |
| `xs_sector_{h}` | stock | 5,20,60 | P(stock excess return vs own sector > 0) | binary |
| `xs_mkt_{h}` | stock, sector | 5,20,60 | P(excess return vs market benchmark > 0) | binary |
| `rank_{h}` | stock | 5,20,60 | cross-sectional rank of excess return within universe | ranking |
| `vol_20d` | stock, sector, index | 20 | realized vol next 20d (annualized, log) | regression |
| `dd_20d` | stock, index | 20 | P(max drawdown within next 20d <= -10%) (thresholds configurable: -5/-10/-20%) | binary |
| `regime_20d` | market (US, JP) | 20 | P(risk-off regime) where risk-off := index drawdown <= -7% OR realized vol top-quintile (configurable) | binary |
| `rrg_quad_{h}` | sector, theme | 20,60 | probability distribution over RRG quadrant next period (Leading/Weakening/Lagging/Improving) | multiclass |

Notes:
- 1-day direction IS in scope. Expect small edge; measure it honestly (Brier skill score vs rolling base rate, AUC, calibration). Do not suppress it; let the data decide.
- Benchmarks: US market = SPY (and RSP equal-weight as secondary); JP market = TOPIX (1306.T as proxy) and Nikkei 225 secondary. Sector benchmarks: Section 9.
- Corporate actions: use split/dividend-adjusted prices for returns; keep unadjusted for price-level features.

---

## 2. Constraints

- Budget: $0 at start. Paid add-ons later behind adapters (Section 18).
- Compute: GitHub Actions. Two deployment options (choose in Phase 0, record in ADR):
  - **Option A (default)**: private repo, data in private object storage (Cloudflare R2 free tier ~10GB, S3-compatible, zero egress) or in-repo Parquet while small. Private-repo Actions minutes are limited (~2,000 min/month on free plan, verify current limits) → budget runtime (Section 16).
  - **Option B**: public repo containing code only (no data, no secrets in code), data in private R2. Public-repo Actions minutes are free/unlimited (verify). Note: scheduled workflows in public repos get disabled after ~60 days without repo activity → add a keepalive mechanism.
- Scheduled Actions can be delayed by minutes–hours; jobs must tolerate delay and be date-parameterized (backfill any missed date).
- All URLs/endpoints in this doc are hints; **verify each at build time** and record the verified endpoint + ToS notes in `config/sources/*.yaml`.

---

## 3. Architecture

```
            ┌──────────── schedulers (GitHub Actions cron + workflow_dispatch) ─────────────┐
            │                                                                                │
 sources ─► ingest/ (one adapter per source) ─► raw/ (immutable, partitioned)               │
                                                 │                                          │
                                                 ▼                                          │
                                   normalize/ ─► silver/ (typed, PIT-stamped, validated)    │
                                                 │                                          │
                                                 ▼                                          │
                                   features/ ─► gold/features (versioned feature store)     │
                                                 │                                          │
                     labels/ ◄── silver prices   │                                          │
                        │                        ▼                                          │
                        └──────────► models/ (train, calibrate, ensemble, registry)         │
                                                 │                                          │
                                                 ▼                                          │
                                   forecast/ ─► ledger/ (append-only, hashed)               │
                                                 │                                          │
                                   scoring/ ◄────┘ (when outcomes mature)                   │
                                                 │                                          │
                                   improve/ (self-improvement loops) ──► registry updates ──┘
                                                 │
                                   rotation/ (sector & theme flow analytics)
                                                 │
                                   report/ (JSON for AI, HTML dashboard, Discord alerts)
```

Layer naming: raw (bronze) → silver → gold. Query engine: DuckDB over Parquet. Dataframes: Polars preferred (pandas allowed where libraries require).

---

## 4. Repository layout

```
.
├── CLAUDE.md                     # short pointer: "read docs/DESIGN.md first"
├── docs/
│   ├── DESIGN.md                 # this file
│   ├── DECISIONS.md              # ADR log
│   ├── DATA_DICTIONARY.md        # auto-generated from schemas
│   └── RUNBOOK.md                # ops: how to backfill, rerun, rotate secrets
├── config/
│   ├── settings.yaml             # global config (paths, horizons, thresholds)
│   ├── universe/                 # universe definitions & theme baskets
│   ├── sources/                  # one yaml per source: endpoint, schedule, lag, ToS notes, rate limits
│   └── models/                   # model specs, promotion gates
├── src/mw/                       # python package "mw" (market weather)
│   ├── core/                     # storage backend, PIT utils, calendars, logging, http client w/ rate limit
│   ├── ingest/{us,jp,macro}/     # source adapters ("global" is a Python keyword; ADR-0007)
│   ├── normalize/
│   ├── quality/                  # schema validation (pandera), anomaly checks, source health
│   ├── universe/
│   ├── features/                 # feature families, registry, versioning
│   ├── labels/
│   ├── models/                   # baselines, ml, calibration, conformal, ensemble, registry
│   ├── forecast/
│   ├── ledger/
│   ├── scoring/
│   ├── improve/                  # self-improvement loops
│   ├── rotation/
│   ├── report/
│   └── llm/                      # (Phase 6+) text→features, research agent, narratives
├── jobs/                         # thin CLI entrypoints called by workflows
├── tests/                        # unit + PIT-leakage tests + adapter contract tests (recorded fixtures)
└── .github/workflows/            # cron workflows
```

---

## 5. Tech stack

- Python 3.11+, dependency mgmt with `uv`.
- Data: polars, pyarrow, duckdb, pandas (as needed).
- HTTP: httpx with retry/backoff + per-host token-bucket rate limiter; cache layer for idempotent fetches.
- Validation: pandera (or equivalent) schemas per silver table.
- ML: scikit-learn, lightgbm, statsmodels; MAPIE (conformal); optuna (bounded HPO).
- Explainability: SHAP (tree explainer) for importance stability tracking.
- Calendars: exchange_calendars (XNYS, XTKS).
- Viz: plotly → static self-contained HTML.
- Storage backend interface: `LocalFS`, `GitRepoFS`, `S3Compatible(R2)`. Swappable by config.
- Secrets via GitHub Actions secrets (FRED key, EDINET key, R2 creds, Discord webhook, later Anthropic API key).

---

## 6. Data model & storage

### 6.1 Required columns on every silver table
| column | meaning |
|---|---|
| `entity_id` | canonical id: `US:AAPL`, `JP:7203`, `ETF:US:XLK`, `IDX:JP:TOPIX`, `MACRO:FRED:BAMLH0A0HYM2` |
| `as_of` | date/period the value describes |
| `published_at` | earliest UTC timestamp the public could know it (from source metadata or source-specific lag rule in config) |
| `ingested_at` | UTC timestamp of our ingestion |
| `source_id` | adapter id |
| `source_version` | adapter version / schema hash |
| `raw_ref` | pointer to raw file + row hash |

If `published_at` is unknown, derive it conservatively via the source's documented release schedule + safety margin (config `publish_lag_rule`). Backfilled history uses the same rule.

### 6.2 Partitioning
`raw/{source_id}/dt={ingest_date}/...` (original bytes, compressed)
`silver/{table}/market={US|JP}/year=YYYY/month=MM/*.parquet`
`gold/features/v={feature_set_version}/market=.../date=.../*.parquet`
`ledger/forecasts/date=.../*.parquet` (+ `ledger/manifest.jsonl` with sha256 of each file)
`registry/{models,features,experiments,incidents}.jsonl`

### 6.3 Size management
- Start in-repo only if total < ~500MB; otherwise R2 from day one (recommended).
- Options chains: store full chain only for the options watchlist; store per-underlying daily summaries for all.
- Compaction job monthly (merge small parquet files).

### 6.4 Universe & survivorship
- Snapshot universe membership daily into `silver/universe_membership` (PIT). Never rebuild history from today's constituents without flagging `survivorship_biased=true` on any backtest that does so.
- Store security master: identifiers, names, exchange, GICS-like sector/industry (US), TSE 33-industry & TOPIX-17 (JP), listing/delisting dates when known.

---

## 7. Universe

- US core: **all exchange-listed common stocks incl. small caps / penny stocks** (Nasdaq Trader symbol directory, ~5,900; ADR-0008). Model universes (e.g. S&P 500 + Nasdaq-100) are subsets selected by liquidity filters.
- US ETFs: 11 SPDR sector ETFs (XLK XLF XLV XLY XLP XLE XLI XLB XLU XLRE XLC), industry ETFs (SMH, SOXX, IGV, XBI, KRE, KBE, ITB, XHB, XRT, XOP, OIH, GDX, TAN, URA, ITA, JETS, IYT, PAVE, GRID, etc.), factor ETFs (MTUM, QUAL, USMV, VLUE, IWM, RSP), rates/credit/commodity ETFs (TLT, IEF, HYG, LQD, GLD, USO, UUP) as cross-asset context.
- JP core: TOPIX 500 (or Prime large/mid), Nikkei 225 constituents.
- JP sector instruments: TOPIX-17 ETFs (1617.T–1633.T), TSE 33-industry indices (fetch if free source exists; else compute from constituents).
- Theme baskets (config, user-editable): e.g. `AI_INFRA_NEOCLOUD` (IREN, CRWV, NBIS, APLD, CORZ, WULF, CIFR), `AI_POWER` (VRT, GEV, BE, CEG, VST, TLN, ETN), `SEMIS`, `MEMORY`, `JP_SEMI_EQUIP`, `JP_POWER_INFRA`, etc. Baskets are equal-weight and cap-weight variants.
- Options watchlist (US): top ~200 by option volume within universe + all theme basket members. Rotate if rate limits bind.

---

## 8. Data source catalog

Priority: P1 = Phase 1, P2 = Phase 2, P3 = later. "Backfill" = history obtainable now. Verify every endpoint & ToS at build time.

### 8.1 US
| id | data | freq | backfill | method / notes | prio |
|---|---|---|---|---|---|
| `us_prices` | OHLCV adj/unadj, dividends, splits | daily | yes (decades) | yfinance; fallback Stooq. Cloud IPs may be throttled → retries, jitter, fallback, cache | P1 |
| `finra_regsho_daily` | daily short sale volume & total volume by symbol (FINRA facilities) | daily | yes (years) | FINRA CDN daily files | P1 |
| `finra_short_interest` | short interest, days to cover | 2x/month | yes | FINRA (API or files). `published_at` = FINRA publication date, not settlement date | P1 |
| `finra_ats` | off-exchange (ATS/non-ATS) weekly volume by symbol | weekly | yes | 2–4 week publication lag | P2 |
| `us_options_chain` | per contract: strike, expiry, bid/ask/last, volume, OI, IV | daily | **no** | yfinance option chains; OI reflects prior day (OCC) → run pre-market ET for updated OI and post-close for volume; record both snapshots | P1 (start ASAP; history only accumulates) |
| `cboe_pcr` | market put/call ratios (total/equity/index) | daily | yes | CBOE market statistics | P2 |
| `vol_indices` | VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, MOVE (if free) | daily | yes | yfinance ^ tickers / CBOE | P1 |
| `sec_form4` | insider transactions (code P/S/A/M…) | event | yes | SEC EDGAR; User-Agent with contact required; ≤10 req/s | P2 |
| `sec_form144` | proposed insider sales | event | yes | EDGAR | P2 |
| `sec_13dg` | beneficial ownership >5% (13D activist / 13G passive) | event | yes | EDGAR | P2 |
| `sec_13f` | institutional holdings | quarterly (45d lag) | yes | EDGAR; aggregate per security: holders count, net shares change, new/closed positions; `published_at` = filing date | P2 |
| `sec_8k` | 8-K item codes (events) | event | yes | EDGAR submissions; item codes as categorical features | P2 |
| `sec_xbrl` | fundamentals (companyfacts / frames) | quarterly | yes | revenue, margins, shares outstanding (dilution!), cash, debt, buyback | P2 |
| `earnings_calendar` | next/last earnings dates, (EPS actual/estimate if available) | daily | partial | yfinance; verify reliability | P2 |
| `etf_flows` | ETF shares outstanding & NAV → creations/redemptions | daily | partial | issuer sites (SPDR, iShares…) / yfinance; compute flow = Δshares × NAV | P2 |
| `finra_margin` | aggregate margin debt | monthly | yes | market-level context | P3 |
| `ici_flows` | aggregate fund flows (equity/bond) | weekly | yes | market-level | P3 |
| `cftc_cot` | futures positioning (ES, NQ, RTY, NKD, ZN, CL, JPY…) | weekly | yes | CFTC public files | P2 |

### 8.2 Japan
| id | data | freq | backfill | method / notes | prio |
|---|---|---|---|---|---|
| `jp_prices` | OHLCV (.T) | daily | yes | yfinance; fallback Stooq; J-Quants free plan (delayed ~12 weeks: use for history/validation only — verify current terms) | P1 |
| `jsf_lending` | 日本証券金融: 貸借取引残高 (融資/貸株), 品貸料 (逆日歩), 貸借倍率 | daily | yes | JSF public data | P1 |
| `jpx_short_positions` | 空売り残高報告 (holders ≥0.5%: institution, ratio, shares, calc date) | daily | yes | JPX public files. Very high value: per-institution short positions | P1 |
| `jpx_short_ratio` | 空売り比率 (market & 33-industry) | daily | yes | JPX | P1 |
| `jpx_margin_weekly` | 銘柄別信用取引週末残高 (信用買残/売残) | weekly | yes | JPX; derive 信用倍率, Δ残 | P1 |
| `jpx_investor_type` | 投資部門別売買状況 (foreigners, individuals, trust banks, corporates…) | weekly | yes | JPX; market-level flow regime | P2 |
| `jpx_daily_notice` | 日々公表銘柄, 信用規制 | daily | partial | JPX/JSF; categorical flags | P2 |
| `mof_portfolio_flows` | 対内対外証券投資 (foreign net purchases of JP equities) | weekly | yes | Ministry of Finance | P2 |
| `edinet` | 大量保有報告書 (holder, ratio, purpose), 変更報告書, 自己株券買付状況報告書, 有報/四半期 XBRL | event | yes | EDINET API v2 (free key) | P2 |
| `jp_earnings_calendar` | 決算発表予定日 | daily | partial | JPX published schedule | P2 |
| `tdnet` | 適時開示 titles | event | partial | **check ToS before any automated access**; if disallowed, skip and rely on EDINET | P3 |
| `boj_rates` | JGB yields, policy rate | daily | yes | BOJ time-series data (API) | P2 |

### 8.3 Global macro / cross-asset / alternative (free)
| id | data | freq | notes | prio |
|---|---|---|---|---|
| `fred` | UST yields (2y,10y,30y), curve, HY OAS, IG OAS, breakevens, DXY proxy, WTI, Brent, USDJPY, NFCI, fed funds | daily/weekly | FRED API key | P1 |
| `gpr_index` | Geopolitical Risk Index (Caldara–Iacoviello), daily | daily | quantifies geopolitics | P2 |
| `epu_index` | Economic Policy Uncertainty (US, JP) | daily/monthly | | P2 |
| `gdelt` | global news event counts & tone by country/theme (e.g. Iran, Israel, China-Taiwan, tariffs, data centers) | 15-min → aggregate daily | GDELT 2.0; build country/theme intensity & tone series | P2 |
| `wiki_pageviews` | Wikipedia pageviews per company/theme | daily | attention proxy; Wikimedia API | P3 |
| `google_trends` | search interest | weekly | unofficial access, fragile → optional | P3 |
| `event_calendar` | FOMC, BOJ MPM, CPI/PPI/NFP, JP CPI, elections, option expiries (OPEX, SQ), index rebalances | static+updates | hand-maintained YAML + scraped official calendars | P1 |

---

## 9. Feature catalog

All features computed PIT at forecast cutoff. Each feature has: `feature_id`, `family`, `version`, `inputs`, `min_history`, `pit_lag_rule`, `owner` (`human` | `auto_discovery` | `llm_agent`). Default transforms per raw feature: level, Δ over {1,5,20}, z-score vs own 252d history, cross-sectional rank (within universe and within sector).

### 9.1 Price / volume / liquidity
returns {1,2,5,10,20,60,120,252}d; 12-1 momentum; short-term reversal (1–5d); realized vol {5,20,60}; Parkinson/Garman-Klass vol; vol-of-vol; ATR; gap stats; distance from MAs {20,50,200}; 52w high/low distance; drawdown from peak; relative volume (vs 20/60d); dollar volume; turnover; Amihud illiquidity; up-volume/down-volume ratio; OBV slope; accumulation/distribution line slope; **price-volume divergence** (OBV/AD slope z minus price slope z); volume dry-up near support; close location value; consecutive up/down days; beta & idiosyncratic vol vs market/sector (60d); correlation to market (60d).

### 9.2 Relative strength (core for "who leads the recovery")
excess return vs market & sector over horizons; **downside capture during market drawdowns** (stock return on market-down days, rolling); **relative strength during the most recent market drawdown window** (auto-detected); RS trend slope; RRG coordinates of the stock vs its sector.

### 9.3 Short interest & lending
US: short volume ratio (FINRA daily) level/trend; short interest % float, Δ, days to cover; SI change vs price change interaction (shorts adding into weakness vs covering).
JP: per-institution short positions (count of reporting holders, total reported ratio, Δ, new entrants/exits, notable institutions); 空売り比率 (stock if available, else industry); 信用倍率, Δ信用買残, Δ信用売残; 貸借倍率; 逆日歩 (level, days nonzero); 日々公表/規制 flags.

### 9.4 Options (US)
ATM IV (30d interpolated); IV rank/percentile (once history exists); IV − RV spread; term structure slope (IV30 vs IV90); 25-delta skew (put−call), skew Δ; put/call volume & OI ratios; OI change by moneyness bucket & expiry bucket (e.g. OTM calls 30–90d buildup); unusual volume (volume/OI); estimated dealer gamma exposure (GEX) & gamma flip level (from OI, assumption-flagged); max-pain distance near expiry; days to OPEX.

### 9.5 Ownership / insiders / flows
US: insider net buy $ (open-market P only) over {30,90}d, insider buyer count (cluster buys), Form 144 intensity; 13D/13G new/increase flags; 13F: Δholders, Δshares held, new positions count (lagged PIT); ETF-flow-implied demand (sum over ETFs holding the stock of flow × weight) — P3.
JP: 大量保有 new/increase/decrease flags with purpose category; 自己株買い progress (reported purchases vs authorized); investor-type flows (market-level).

### 9.6 Fundamentals (slow, PIT by filing date)
revenue growth, margin trend, accruals, net share issuance (dilution rate!), buyback yield, cash runway proxy, debt/EBITDA (where computable), earnings surprise & post-earnings drift (if EPS actual/estimate available).

### 9.7 Events & calendar
days to/from earnings; days to FOMC/BOJ/CPI; OPEX/SQ week; index rebalance window; month/quarter-end; holidays; election proximity; 8-K item flags (recent 5/20d).

### 9.8 Macro regime (shared features)
HY OAS level/Δ/z; IG OAS; 2s10s; 10y Δ; real yields; VIX level, VIX term-structure (VIX/VIX3M), VVIX; DXY Δ; USDJPY Δ; oil Δ; NFCI; GPR & EPU z; GDELT tone/intensity for watched geographies/themes; CFTC net positioning z; JP: JGB10y Δ, foreign net buying z. Plus a **regime label** from a regime model (HMM or rules) used both as feature and for conditional models.

### 9.9 Breadth & sector aggregates (also used in rotation module)
per sector/theme: % members above 50/200dma, new highs − new lows, advance/decline, median RS, dollar-volume share of total market (capital allocation proxy) and its Δ, aggregated short metrics, aggregated options skew, ETF flows.

### 9.10 Text-derived (Phase 6, LLM)
event type, severity, sentiment, guidance direction, risk-factor change magnitude (10-K diff), 大量保有 purpose classification, 適時開示 category — all as numeric/categorical features with `published_at` of the source document.

---

## 10. Models

### 10.1 Model ladder (each rung must beat the previous one to be used)
1. **Climatology baseline**: rolling base rate per target/entity-class (e.g. P(up 1d) = trailing 252d frequency). Quantiles from trailing empirical distribution. Vol from EWMA.
2. **Simple rules**: momentum/reversal/RS rules → probabilities via logistic mapping.
3. **Linear**: regularized logistic / ridge / quantile regression on standardized features.
4. **GBDT**: LightGBM classifiers, quantile regressors, LambdaRank for `rank_{h}`. Monotonic constraints where theory is clear (optional).
5. **Regime-conditional models**: separate or regime-feature-interacted models per macro regime.
6. **Ensemble (stacking / online weights)**: combine rungs; weights updated online (Section 12.2).

Market scope: train US and JP separately by default; also test pooled models with `market` feature. Horizon-specific models.

### 10.2 Probability calibration
Isotonic or Platt calibration fitted on a calibration window that is disjoint from training; refit weekly. Track ECE and reliability curves per target.

### 10.3 Uncertainty ("forecast cone")
- Conformal prediction (MAPIE, time-series-aware) for return intervals with target coverage (e.g. 80%/90%); monitor empirical coverage.
- Ensemble disagreement (std of member probabilities) → `confidence` field.
- Output for each forecast: point probability, interval, confidence, top contributing features (SHAP).

---

## 11. Validation protocol

- **Walk-forward** with expanding (default) and rolling windows. Retrain cadence: weekly.
- **Purging & embargo**: remove training samples whose label window overlaps the validation window; embargo = horizon length (López de Prado).
- **Metrics**: binary → Brier, Brier Skill Score vs climatology, log loss, AUC, ECE, hit rate by confidence decile; quantile → pinball loss, coverage; ranking → rank IC (Spearman), top-minus-bottom decile spread; vol → QLIKE/RMSE of log vol.
- **Significance**: Diebold–Mariano (or bootstrap) for loss differences champion vs challenger; block bootstrap for time-dependence.
- **Multiple testing**: every experiment is counted in `registry/experiments.jsonl`; apply FDR control (Benjamini–Hochberg) when screening many features/models; report deflated metrics.
- **Slices**: evaluate by market, sector, cap bucket, vol bucket, regime, event proximity (earnings week, FOMC week).
- **Leakage tests (CI)**: synthetic test where future data is injected must be detected; PIT assertion that every feature's max `published_at` <= cutoff.

---

## 12. Forecast ledger & scoring

### 12.1 Ledger schema (append-only)
`forecast_id, created_at, cutoff_at, market, entity_id, target_id, horizon, model_id, model_version, feature_set_version, data_snapshot_hash, prob | quantiles | value, interval_lo, interval_hi, confidence, top_features(json), role (champion|challenger|baseline|ensemble), code_commit`

Rules: write before market open of the target period (JP: before 09:00 JST next day; US: before 09:30 ET next day). Hash each partition; append hash to `ledger/manifest.jsonl`. Never mutate. Challengers and baselines are ALSO logged (shadow mode) — this is the live out-of-sample evidence base.

### 12.2 Scoring
Daily job joins matured forecasts with realized labels → `scores/` table. Maintains rolling metrics (20d, 60d, 252d) per (model, target, market, slice). Publishes `scores/summary.json`.

---

## 13. Self-improvement system (core requirement)

Self-improvement is a set of scheduled loops with explicit gates. The system may change itself automatically only through these loops; every change is logged, reversible, and justified by out-of-sample evidence.

### 13.1 Daily loop (cheap, safe)
1. Score matured forecasts; update rolling metrics.
2. **Online ensemble re-weighting**: update member weights with exponentially weighted log-loss (Hedge / multiplicative weights), with floor/ceiling per member to avoid collapse. Separate weights per (target, market, regime).
3. Drift monitors: PSI per feature (today vs trailing 252d), label base-rate shift, calibration drift (rolling ECE). Write alerts to `registry/incidents.jsonl`.
4. Data-quality monitors (Section 15). Quarantine features from failing sources (fall back to models without them).

### 13.2 Weekly loop (retrain & promote)
1. Retrain champion architecture on latest data (expanding window) → candidate `champion_refresh`.
2. Train challengers: (a) bounded HPO (optuna, fixed trial budget, nested CV), (b) feature-set variants from the discovery pipeline, (c) alternative model families.
3. Recalibrate probabilities on the disjoint calibration window.
4. New challengers enter **shadow mode** (logged live forecasts, not shown as primary).
5. **Promotion gate** (all must pass; thresholds in `config/models/gates.yaml`):
   - ≥ N weeks of live shadow forecasts (default 4 weeks for 1d/5d targets, 8 for 20d, 13 for 60d; matured outcomes count only).
   - Better primary loss than champion with DM test p < 0.10 on live shadow AND on walk-forward backtest.
   - Calibration not worse (ECE delta ≤ +0.005) and interval coverage within ±3pp of target.
   - Beats climatology baseline (BSS > 0).
   - No slice catastrophically worse (configurable max degradation per slice).
6. Promote → old champion archived (kept as ensemble member if it adds value).
7. **Rollback rule**: if live champion underperforms its previous version or the baseline for K consecutive weeks (default 3) beyond tolerance, auto-revert and open an incident.

### 13.3 Monthly loop (discover & prune)
1. **Automated feature discovery**:
   - Candidate generator: apply operator library (lag, Δ, z, rank, ratio, product/interaction across families, regime-conditional splits, rolling quantiles) to existing silver series and to newly added sources.
   - Screen on training folds only: rank IC / mutual information with targets, stability across time folds and regimes, low redundancy (correlation cap with existing features).
   - Control false discoveries (BH-FDR across all candidates tested this cycle; experiment counter increments).
   - Survivors → added to a challenger feature set → must pass Section 13.2 gates to reach production.
2. **Pruning**: track SHAP importance stability and rolling univariate IC per feature; features with negligible contribution for 3 consecutive months or whose removal improves validation loss are deprecated (soft-delete, versioned).
3. **Signal decay ledger**: per feature & per source, chart rolling IC; flag decaying signals; surface in report.
4. **Error analysis**: find slices with worst skill (sector, regime, event windows); write `reports/error_analysis_{YYYYMM}.json` that the research loop consumes.
5. **Source value audit**: estimate marginal contribution of each data source (drop-source ablation). Feeds decisions on paid upgrades (Section 18).

### 13.4 Research loop (Phase 6+, LLM agent; optional, costs API)
- A scheduled Claude Code / Claude API agent reads: error analysis, decay ledger, incident log, experiment registry.
- Produces hypotheses as structured JSON (`hypothesis_id, rationale, proposed_features(code), expected_effect, test_plan`), implements them on a branch, runs the standard backtest + gates in CI.
- Auto-merge allowed ONLY for feature additions that pass all gates in shadow; model-architecture or validation-protocol changes require owner notification (Discord) and a 7-day delay before merge.
- The agent must never edit: ledger, PIT rules, gates thresholds, or this file's principles.

### 13.5 Guardrails against self-deception
- Live shadow evidence is primary; backtests are secondary.
- Global experiment counter → deflated metrics; cap experiments per month (config).
- Frozen "exam" set: a rolling block of recent weeks used only for final promotion checks, never for training/selection.
- Max number of production changes per week (default 2).
- Every change: ADR line in `docs/DECISIONS.md` (auto-appended) + registry record + reproducible commit.

---

## 14. Sector & theme rotation module (`src/mw/rotation/`)

Goal: show where capital is rotating across sectors/themes (not just within), in US and JP, and forecast the next rotation step.

### 14.1 Units
- US: 11 sectors (SPDR ETFs) + industry ETFs + theme baskets.
- JP: TOPIX-17 (ETFs 1617–1633.T) + TSE 33-industry (index or constituent-built) + theme baskets.
- Cross-market map: config table mapping JP industries ↔ US sectors (approximate GICS alignment) for side-by-side comparison.

### 14.2 Analytics
1. **Relative Rotation Graph (RRG)** per unit vs benchmark (SPY / TOPIX):
   - RS = price_unit / price_benchmark; RS-Ratio = normalized, smoothed RS trend (centered at 100); RS-Momentum = normalized rate of change of RS-Ratio (centered at 100). Document exact normalization windows in config (daily: 10/50; weekly: 10/26 — tune & record).
   - Quadrants: Leading (≥100,≥100), Weakening (≥100,<100), Lagging (<100,<100), Improving (<100,≥100). Store daily & weekly coordinates + trailing tails.
2. **Flow proxies**:
   - Dollar-volume share of each unit in total market dollar volume, and its Δ (z-scored).
   - ETF net flows (creations/redemptions) per sector ETF (US), JP sector ETF flows if obtainable.
   - Breadth per unit (Section 9.9).
   - Aggregated short/lending changes per unit (shorts covering vs adding).
   - JP investor-type flows (market-level overlay) and foreign net buying.
3. **Rotation dynamics**:
   - Empirical quadrant transition matrix (Markov, by regime) → baseline P(next quadrant).
   - Model forecast `rrg_quad_{h}` (Section 1) using unit-level aggregated features + macro regime.
   - Lead–lag detection between units (e.g. cross-correlation of RS changes; which sectors historically lead after risk-off ends).
4. **Recovery-leader detector**: during/after detected market drawdowns, rank units and stocks by relative strength + accumulation signals; log as forecasts (so the detector is scored like everything else).

### 14.3 Visual outputs (static HTML, plotly)
- RRG scatter with tails (US and JP, daily & weekly toggle), animated over last N weeks.
- Heatmap: units × horizons {1d,5d,20d,60d} excess returns; second heatmap of flow proxies z-scores.
- Stacked area: dollar-volume share by sector over time (capital allocation drift).
- Rotation forecast panel: P(Leading next 20d) per unit with confidence.
- Cross-market panel: US sector vs mapped JP industry RS side by side.
Human-facing, but generated automatically; layout simplicity over polish.

---

## 15. Data quality & source health

- Schema validation per silver table (types, ranges, uniqueness on (entity_id, as_of, source_id)).
- Expected-row-count checks vs universe size & calendar; staleness checks (`as_of` lag vs expected).
- Anomaly checks: price jumps without corporate action, zero volume streaks, duplicated days, OI negative, IV outliers.
- Cross-source reconciliation where 2 sources exist (e.g. yfinance vs Stooq close within tolerance).
- Source health score (success rate, latency, schema hash changes). Schema change → adapter marked `degraded`, raw kept, incident opened, downstream features from that source quarantined automatically.
- Contract tests for adapters using recorded fixtures (so parser changes are testable offline).

---

## 16. Scheduling (UTC; verify DST handling via exchange calendars, skip holidays)

| job | when (UTC) | content |
|---|---|---|
| `jp_eod` | Mon–Fri 10:30 | JP prices, JSF lending, JPX short positions/ratio (after publication), JP macro; features; JP forecasts for next session → ledger |
| `us_options_pre` | Mon–Fri 12:30 | options OI snapshot (OI updated overnight) for watchlist |
| `us_eod` | Tue–Sat 00:30 | US prices, FINRA RegSHO daily, vol indices, FRED, options post-close volume snapshot; features; US forecasts for next session → ledger |
| `events_daily` | daily 06:00 | EDGAR (Form 4/144/13D-G/8-K), EDINET, calendars, GDELT/GPR/EPU aggregates |
| `score_daily` | daily 13:00 | scoring, online ensemble weights, drift & quality monitors, daily report + Discord |
| `weekly` | Sat 03:00 | JPX weekly margin, investor types, MOF flows, CFTC, FINRA ATS; retrain/challengers/calibration/promotion gates; rotation weekly view |
| `semimonthly` | after FINRA SI publication | short interest ingest |
| `monthly` | 1st Sun 03:00 | feature discovery, pruning, decay ledger, error analysis, source value audit, compaction |
| `quarterly` | 13F windows | 13F ingest; (Phase 6) research agent cycle |

Runtime budget (Option A private repo): target ≤ 1,500 min/month total; log per-job minutes to `registry/runtime.jsonl`; if over budget, degrade gracefully (shrink options watchlist first, then HPO trials). Heavy backfills via `workflow_dispatch` chunked by date range.

---

## 17. Outputs

### 17.1 Machine-readable (primary; for AI consumers)
`reports/daily/{date}/forecast.json`:
```
{
  "date": "...", "market": "US|JP",
  "market_weather": {"regime_prob_risk_off_20d": 0.31, "vol_forecast_20d": ..., "dd_10pct_prob_20d": ..., "confidence": ...},
  "sectors": [{"unit_id","rrg":{"ratio","momentum","quadrant"},"p_leading_20d","xs_mkt_20d","flows_z","confidence","drivers":[...]}],
  "stocks": [{"entity_id","dir_1d":{"p":0.56,"bss_live_60d":0.012},"xs_sector_20d":{"p":...},"ret_q_20d":{...},"dd_20d":{...},"confidence","drivers":[...],"flags":[...]}],
  "changes_since_yesterday": [...],
  "model_health": {"champion_versions":{...}, "live_skill":{...}, "incidents":[...]}
}
```
Every probability is accompanied by the live measured skill of the model that produced it (so consumers can discount weak forecasts).

### 17.2 Human-facing (secondary)
- `reports/daily/{date}/index.html`: market weather, rotation charts, top lists (highest/lowest probabilities with confidence filter), biggest changes, model health.
- Discord webhook: short summary + link/path; alerts for incidents and promotions.

---

## 18. Extensibility (paid upgrades, compute, LLM)

- Data adapters planned (disabled until enabled in config): Polygon/Massive (US prices, options history), ORATS or CBOE DataShop (options history/IV surfaces), Ortex/S3 (borrow fee, utilization), J-Quants paid (JP real-time-ish fundamentals/prices), Tiingo/EODHD (prices/fundamentals), news APIs.
- Decision rule for paid data: buy only if the source value audit (13.3.5) or a cheap trial ablation shows expected skill gain on live/backtest beyond a threshold; record in ADR.
- Compute: self-hosted GitHub runner on owner PC or a small cloud VM if Actions minutes bind.
- Storage: R2 → larger S3/GCS; DuckDB → MotherDuck/Postgres if concurrent access needed.

---

## 19. LLM layer (Phase 6+, uses Anthropic API; owner approves cost)

- **Text→features**: classify and score 8-K items, earnings releases, 10-K risk-factor diffs, EDINET 大量保有 purposes, (TDnet if permitted), headlines → numeric/categorical features with source `published_at`.
- **Narratives**: daily natural-language explanation of the forecast JSON (drivers, what changed), written in Japanese for the owner.
- **Research agent**: Section 13.4.
- **Look-ahead caveat (critical)**: LLMs know historical outcomes from training data. Any LLM-derived feature/judgment must NOT be evaluated on periods before the LLM feature went live. Store `llm_model_id` and prompt version with every output; evaluate LLM features only on post-launch live data.
- Cost controls: per-run token budget, caching by document hash, batch processing.

---

## 20. Roadmap with acceptance criteria

| phase | scope | done when |
|---|---|---|
| 0 | repo skeleton, storage backend (choose Option A/B, R2), config system, calendars, logging, CI (lint, tests, PIT tests), ADR log | CI green; a dummy job writes/reads PIT parquet via backend in Actions |
| 1 | P1 sources (US/JP prices, FINRA RegSHO daily + backfill 2y, FINRA SI, vol indices, FRED, JSF lending, JPX short positions/ratio/weekly margin, event calendar, **options snapshots start**) + quality checks | 10 consecutive trading days ingested for both markets with no unhandled failure; backfills complete; data dictionary generated |
| 2 | features 9.1–9.3, 9.7, 9.8; labels; climatology & linear models; ledger + scoring; daily JSON/HTML report; Discord | ledger receiving daily forecasts for all targets; scoring produces live metrics after first maturities; leakage tests pass |
| 3 | P2 sources (EDGAR, EDINET, CFTC, ATS, ETF flows, GDELT/GPR/EPU, investor types, MOF); features 9.4–9.6, 9.9; GBDT, calibration, conformal, ensemble | walk-forward report beats climatology on ≥1 target per market with DM p<0.10 (if not, document and continue — no forcing) |
| 4 | rotation module + visuals; recovery-leader detector logged as forecasts | RRG/heatmaps/flows generated daily for US & JP; rotation forecasts in ledger |
| 5 | self-improvement loops 13.1–13.3 fully automated with gates, rollback, registries | at least one full weekly + monthly cycle completed autonomously with logged decisions |
| 6 | LLM layer (owner approval for API cost) | text features live with versioning; research agent produces gated PRs |
| 7 | paid-data trials driven by source value audit | ADR decision per candidate source |

---

## 21. Known limitations & risks (keep updated)

- Free data fragility (rate limits, cloud IP throttling, format changes), especially yfinance and JPX/JSF file formats.
- Options history only accumulates from start date; options-based features cannot be validated until enough history exists (≥6–12 months for robust results).
- Survivorship bias in backfilled history built from current constituents (flag it).
- Earnings estimates/consensus are not reliably free → surprise features may be weak.
- 13F, FINRA ATS, short interest are lagged; PIT rules handle it but signal value is reduced.
- Markets are adaptive/reflexive: signals decay; expect modest skill (e.g. 1d direction AUC 0.51–0.55 would already be meaningful). Never inflate claims.
- GitHub Actions schedule delays and minute limits.

---

## 22. Engineering conventions

- Type hints everywhere; ruff + mypy (strict on core/ledger/labels/features).
- Every adapter: `fetch(date_range) -> raw files`, `parse(raw) -> silver rows`, declared `publish_lag_rule`, rate limit, retries, contract test with fixture.
- Every feature: pure function of silver tables + cutoff; registered with metadata; unit test with synthetic data incl. PIT edge cases.
- Deterministic seeds; record library versions in model registry.
- Jobs idempotent: re-running a date overwrites only derived layers for that date, never raw or ledger.
- Logs: structured JSON; per-job summary appended to `registry/runs.jsonl`.
- Secrets never printed; configuration validated at startup (pydantic).

## 23. First actions for Claude Code
1. Create `CLAUDE.md` pointing here; initialize `docs/DECISIONS.md` with ADR-0001 (storage/deployment option choice; ask owner if unclear — default Option A + R2).
2. Implement Phase 0, then Phase 1 in risk order: start **options snapshots** and **JPX/JSF adapters** first (history-limited and format-fragile), then prices/FINRA/FRED.
3. Verify every endpoint/ToS listed in Section 8; write verified details into `config/sources/*.yaml`; report unavailable sources to owner (in Japanese) with proposed substitutes.
4. Report progress to owner in Japanese, concisely, at the end of each phase with acceptance-criteria evidence.
