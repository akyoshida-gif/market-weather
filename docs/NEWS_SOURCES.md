# NEWS_SOURCES.md — US news / catalyst source check (2026-10-26)

Checked from owner's corporate network (not Actions). UNVERIFIED = not confirmed.

| source | status | timestamp | ticker | history | limits / ToS | plan |
|---|---|---|---|---|---|---|
| SEC EDGAR Atom `browse-edgar?action=getcurrent&type=<FORM>&count=100&output=atom` | OK | acceptance time (to the second, ET offset) | via CIK | ~100/page, pageable, days | 10 req/s, UA with contact required | **backbone**: poll 8-K, 6-K, S-1, S-3, 424B, SCHEDULE 13D/13G, 4 |
| SEC submissions JSON `data.sec.gov/submissions/CIK##########.json` | OK | `acceptanceDateTime` UTC | yes | full | same | precise timestamps, backfill |
| SEC daily index `Archives/edgar/daily-index/YYYY/QTRn/form.YYYYMMDD.idx` | OK | date only | via CIK | 1994– | nightly ~22:00 ET | reconciliation / backfill |
| SEC EFTS full-text `efts.sec.gov/LATEST/search-index` | OK, undocumented | date only | in display_names | 2001– | 10k-hit cap (UNVERIFIED) | press releases as 8-K EX-99.1, PDUFA mentions |
| SEC `company_tickers_exchange.json` | OK | — | yes | current only | — | CIK→ticker (snapshot daily for PIT) |
| SEC XBRL frames `dei/EntityCommonStockSharesOutstanding` | OK | — | via CIK | full | — | shares outstanding / float |
| Nasdaq halts RSS `nasdaqtrader.com/rss.aspx?feed=tradehalts[&haltdate=MMDDYYYY]` | OK | ET ms | yes | per day param | ≤1 req/min | live halts |
| NYSE halts `nyse.com/api/trade-halts/historical/download?haltDateFrom=&haltDateTo=` | OK, undocumented | ET s | yes | ~2021– (all US exchanges) | ToS UNVERIFIED | halt history backfill |
| PR Newswire category RSS | OK (All feed 503) | pubDate UTC | in text "(NASDAQ: XXX)" | 20 items | ToS: personal noncommercial; bans robots | owner decision |
| Business Wire RSS | OK, shallow | pubDate | cashtag/text | few items | ToS restrictive | owner decision |
| GlobeNewswire RSS | 403 (Akamai) from test IP | — | — | — | do not evade | one honest test from Actions |
| ACCESS Newswire / Newsfile / EIN | no public PR RSS / WAF / login | — | — | — | — | skip; use 8-K EX-99.1 |
| openFDA drugsfda | OK | date | no | full | key optional | approvals after the fact |
| PDUFA calendar | no free structured source | — | — | — | — | EFTS "PDUFA" in 8-Ks |
| Reddit / StockTwits | 403 / robots Disallow / approval needed | — | — | — | — | skip unless research access granted |
| Yahoo 1m bars incl. pre/post (`includePrePost=true`) | OK | ET | yes | 1m: 30 days | personal use; cloud blocks possible | capture daily for movers (history expires) |

Notes
- SEC filings after 17:30 ET (22:00 for Forms 3/4/5) are disseminated next business day → `published_at` = max(acceptance, dissemination rule).
- OTC stocks are not in the Nasdaq symbol directory (in SEC exchange file as "OTC").
