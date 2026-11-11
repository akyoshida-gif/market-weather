# SOURCES_VERIFICATION.md — Phase 1 source check (2026-10-23)

Checked from owner's Windows host (not from Actions) unless noted. UNVERIFIED = not confirmed.

## Changes vs DESIGN §8
- JPX file URLs contain opaque folder IDs → scrape index HTML daily for links.
- JPX margin by issue: weekly 週末残高 replaced by DAILY from 2026-09-28, **PDF only (~109 pages), ~5 business days retained, no archive**.
- JPX short ratio (market / 33-industry): **PDF only**.
- JPX investor-type xlsx layout changed from 2026-09-29 release (two parsers needed).
- Stooq: unusable (JS proof-of-work, robots `Disallow: /`) → dropped. US price fallback: Tiingo free (50 req/h, 1,000/day, 500 symbols/month, personal use).
- FRED ICE BofA spreads (BAMLH0A0HYM2, BAMLC0A0CM): only 3 rolling years since 2026-04 → archive our own history.
- FINRA Reg SHO volumes now decimal → parse as float. Not-yet-published file returns 403.
- FINRA short interest: proposal to move to weekly (status UNVERIFIED) → do not hard-code calendar; poll.
- J-Quants Free: 12-week delay, 5 req/min, short/margin/investor-type NOT included (Standard+ / Light+).

## Endpoints
| source | endpoint | format | publish (UTC) | history | status |
|---|---|---|---|---|---|
| jpx_short_positions | jpx.co.jp/markets/public/short-selling/index.html (+ 00-archives-01..12) → `.../{id}-att/YYYYMMDD_Short_Positions.xls` | XLS, 2 header rows | ~08:00 | 13 months | OK (scrape) |
| jpx_short_ratio | jpx.co.jp/markets/statistics-equities/short-selling/ → `{yymmdd}-m.pdf`, `{yymmdd}-g.pdf` | PDF | ~07:30 | 13 months | PDF parse |
| jpx_margin (daily by issue) | jpx.co.jp/markets/statistics-equities/margin/01.html → `YYYYMMDD_mtall.pdf` (`_01` = correction) | PDF | 07:00–09:00 | ~5 days | owner decision |
| jpx_daily_notice | margin/index.html → `YYYYMMDD_mtdaily.xlsx` | XLSX | ~09:00 | short | OK |
| jsf_lending | taisyaku.jp/data/zandaka.csv, shina.csv (header line 4), meigara.csv, seigenichiran.csv | CSV CP932, latest day only | ~02:00 next day | none (archive ourselves) | OK |
| jpx_premium_charges | jpx.co.jp/.../Premium_Charges.xlsx (fixed URL) | XLSX | ~07:00 | latest | OK |
| jpx_investor_type | jpx.co.jp/markets/statistics-equities/investor-type/ → `stock_1_w_*.xlsx` | XLSX | Thu 06:30 | years | OK |
| finra_regsho_daily | cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt | pipe TXT | ≤22:00/23:00 | CNMS from 2018-08-01 | OK |
| finra_short_interest | cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv; API consolidatedShortInterest (no key) | pipe CSV | 7th bday after settlement | 5 years | OK |
| fred | api.stlouisfed.org series/observations (key) | JSON | per series | full (ICE 3y) | OK (key) |
| vol_indices | cdn.cboe.com/api/global/us_indices/daily_prices/{VIX,VIX9D,VIX3M,VIX6M,VVIX,SKEW}_History.csv | CSV full history | ~01:30 next day | 1990– | OK |
| MOVE | Yahoo ^MOVE only (ICE paid) | — | — | — | best-effort |
| tiingo | api.tiingo.com (token) | JSON/CSV | EOD | 30y+ | OK (token) |

## ToS notes (owner decisions recorded in DECISIONS.md)
- JPX: 「有料・無料を問わずJPXからの許諾を得ている場合を除き、商用目的によるデータ収集のほか如何なる用途に関わらず二次利用及び再配信はできません。」 robots.txt: no disallow.
- JSF: 「私的利用の範囲を超えて利用することはできず…」 → private use permitted.
- Yahoo: automated access without permission prohibited (English text UNVERIFIED); yfinance README: personal use only.
- FRED: third-party series personal use OK; no unreasonable bandwidth; Crawl-delay 1.
- Cboe: personal non-commercial use.
- Tiingo free: internal/personal use only.
- J-Quants: individuals only; commercial/academic use prohibited; stay within rate limits.
