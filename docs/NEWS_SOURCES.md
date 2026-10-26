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

## Additional quantitative sources (checked 2026-10-26)

| source | status | cadence / lag | history | ToS risk | plan |
|---|---|---|---|---|---|
| SEC fails-to-deliver `sec.gov/files/data/fails-deliver-data/cnsfailsYYYYMM{a\|b}.zip` | OK | half-monthly, 2–4 wk lag | 2009– (FOIA 2004–) | low | use (lagged feature) |
| Reg SHO threshold: Nasdaq `nasdaqtrader.com/dynamic/symdir/regsho/nasdaqthYYYYMMDD.txt`; NYSE `nyse.com/api/regulatory/threshold-securities/download?selectedDate=DD-Mon-YYYY&market=NYSE` (+American/Arca) | OK | daily evening ET | Nasdaq 2008– | low–medium (ICE "no robots") | 1 fetch/day |
| SEC Form 4 quarterly `.../insider-transactions-data-sets/YYYYqN_form345.zip` + daily index XML | OK | quarterly (~1 wk) + daily | 2006– | low | use |
| CFTC COT Socrata `publicreporting.cftc.gov/resource/{gpe5-46if (TFF), 6dca-aqww (Legacy)}.json` | OK | Fri 15:30 ET (Tue data) | 1986/2006– | low | use |
| FRED release dates `fred/release/dates?release_id=` (CPI 10, NFP 50, PPI 46, GDP 53, Retail 9, FOMC 101; `include_release_dates_with_no_data=true` for future) | OK (key) | scheduled | — | low | use |
| Fed FOMC calendar `federalreserve.gov/monetarypolicy/fomccalendars.htm` | OK | — | — | low | parse; dates tentative |
| Russell recon: semiannual (Jun/Dec); Dec 2026 effective after close Fri 2026-12-11 | OK | — | — | — | event calendar |
| Cboe daily options stats `cdn.cboe.com/data/us/options/market_statistics/daily/YYYY-MM-DD_daily_options` (2019-10-07–) + `cdn.cboe.com/resources/options/volume_and_call_put_ratios/{totalpc,equitypc,indexpc}.csv` (2006–2019) | OK | ~02:30 UTC next day | 2006– | medium (personal non-commercial) | 1 fetch/day (same terms as VIX files) |
| SSGA ETF NAV + shares outstanding `ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-{ticker}.xlsx` | OK (XLK) | ~T+1 | 2006– | ToS unchecked | sector ETF flows |
| FINRA OTC/ATS weekly `api.finra.org/data/group/otcMarket/name/weeklySummary` | OK (free account advised) | weekly, 2–4 wk lag | years | low | later (needs free FINRA account) |
| Yahoo chart futures/FX/crypto (ES=F, NQ=F, RTY=F, YM=F, BTC-USD, CL=F, GC=F, DX-Y.NYB, JPY=X, NIY=F) | OK | live incl. overnight | 2000– daily | high (same as yfinance, owner-accepted ADR-0005) | use for overnight context |
| Nasdaq earnings API `api.nasdaq.com/api/calendar/earnings` | technically OK | daily | 2015– | **high**: robots Disallow:/, ToS bans automated capture & ML use | owner decision |
| IBKR short-stock FTP `ftp3.interactivebrokers.com` user shortstock | UNVERIFIED (port 21 blocked on owner network) | ~15 min | none | unknown | test from Actions; check terms |
| Free float | none free | — | — | — | use SEC XBRL shares outstanding (− insider holdings) |
