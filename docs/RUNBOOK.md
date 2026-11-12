# RUNBOOK.md — operations

## Local dev
```
uv sync
uv run pytest -q
uv run python -m jobs.dummy_pit --date 2026-01-05   # writes to ./data (local backend)
```
Env overrides: `MW_STORAGE_BACKEND=local|r2`, `MW_LOCAL_ROOT=./data`, `MW_SETTINGS=path/to/settings.yaml`.

## GitHub Actions secrets (Settings → Secrets and variables → Actions)
| name | value |
|---|---|
| `R2_ENDPOINT_URL` | `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 API token (Object Read & Write, bucket `market-weather-actions` only) |
| `R2_SECRET_ACCESS_KEY` | same token's secret |
Without R2 secrets, the `smoke` workflow falls back to the local backend.

## Re-run / backfill
All jobs are date-parameterized and idempotent. Use Actions → workflow → "Run workflow" with `date`.

## Stock splits in prices (BACKLOG D5)
- The daily job rewrites only the last 7 days, so after a split older stored rows could stay unadjusted. Since 2026-11-12 a split seen in the window triggers a full-history refetch of that stock (once per split; done pairs are kept in `registry/split_refetch/market=US|JP.parquet`). A failed refetch does not fail the job; it shows as `refetch_for_split.error` in `registry/runs` and is retried on the next run.
- One-off repair: Actions → prices → Run workflow, set `market` and `repair_splits_since` (e.g. 2026-09-01). Avoid the scheduled times (US 21:30 / 01:30 / 04:30 UTC, JP 07:30 / 10:30 UTC); runs of one market queue, but only one waiting run is kept: a scheduled run arriving while the repair waits cancels the repair. A split is marked done only when it is 3 days old (right after a split the provider may not have adjusted the history yet). Check `split_stocks` and `errors` in `registry/runs`. If it stops halfway (some years rewritten, others not), run it again.

## Cost guard (ADR-0001)
`registry/storage_usage.json` holds monthly Class A/B op counts and last known bucket size.
Above soft limits, non-exempt writes raise `StorageGuardError` (ledger/ and registry/ exempt).

## US options snapshots
Workflow `us_options` (pre 12:30 UTC Mon–Fri, post 00:30 UTC Tue–Sat). Manual: Actions → us_options → Run workflow (snapshot, limit, force).
Check `registry/runs/` summaries for `quality` and `degraded`.

## Keepalive
`keepalive.yml` re-enables all workflows monthly (public repos disable schedules after inactivity).

## Japan daily (jp_daily)
Schedules 03:00 & 09:30 UTC Mon–Fri. JSF CSVs hold the latest day only — a missed day is lost; the 09:30 run is a retry.
JPX short positions backfill (~13 months): Actions → jp_daily → Run workflow (sources=jpx_short_positions, backfill=true).

## moomoo (disabled)
See config/sources/us_options_moomoo.yaml and ADR-0006. Requires OpenD on owner PC.
