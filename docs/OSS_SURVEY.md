# OSS_SURVEY.md — reusable components (checked 2026-10-23)

Facts from GitHub API / PyPI on 2026-10-23 unless marked (unverified).

## Adopt (dependencies)
| lib | license | role | notes |
|---|---|---|---|
| edgartools | MIT | SEC EDGAR: Form 4, 13F, XBRL | very active; pin version; SEC UA + ≤10 req/s |
| yfinance | Apache-2.0 | US/JP prices, option chains | fragile, throttled on cloud IPs; behind adapter |
| fredapi | Apache-2.0 | FRED (+ ALFRED vintages for PIT macro) | free key |
| jquants-api-client-python | Apache-2.0 | official JPX data (prices history/validation) | Free plan: 2y, 12-week delay, 5 req/min; short/margin datasets NOT on Free |
| edinet-tools | MIT | EDINET API v2 typed parsers | small project, breaking changes in 0.8 |
| gdeltdoc | MIT | GDELT DOC API | API rate limits/timeouts |
| pandera | MIT | silver schema validation (pandas & polars) | |
| MAPIE (v1) | BSD-3 | conformal intervals (EnbPI/ACI) | v1 API |
| scoringrules | Apache-2.0 | Brier/log/CRPS/interval scores | replaces archived properscoring |
| river | BSD-3 | online ensemble weights, drift detection | |
| arch | NCSA-style | SPA / Model Confidence Set, bootstrap, GARCH baseline | DM test not included (unverified) → self-write |
| purged-cross-validation | MIT | purged/embargoed CV | young; alt: skfolio CPCV (unverified) |

PIT joins: DuckDB `ASOF JOIN` / `merge_asof` on `published_at` (no extra dependency).

## Reference only (design ideas, not dependencies)
- Microsoft Qlib (MIT, active): rolling retrain / online serving / PIT DB design.
- qanat (MIT, beta): PIT store schema in DuckDB.
- QuantileLedger, calibration-log, forecast-ledger repos: append-only hash-chained ledger + scoring.
- RRG-Dashboard (MIT) / OpenBB relative_rotation: RS-Ratio / RS-Momentum approximation. Avoid RRG-Lite (GPL) and RRGPy (no license) code.

## Build in-house (no maintained OSS)
JPX 空売り残高 / 空売り比率 / 週末信用残 parsers; JSF 日証金 貸借残・逆日歩; FINRA RegSHO daily & short interest; daily option-chain archiver; forecast ledger; RRG calc; Diebold–Mariano test.

## Caveats affecting design
- Stooq CSV downloads reportedly require an API key since ~2026-03 (source: pandas-datareader issue #1012; unverified officially) → price fallback needs re-evaluation (J-Quants for JP history; candidates for US TBD).
- cot_reports stale; CFTC Socrata API direct access preferred (verify).
