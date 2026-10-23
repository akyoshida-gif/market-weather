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

## Cost guard (ADR-0001)
`registry/storage_usage.json` holds monthly Class A/B op counts and last known bucket size.
Above soft limits, non-exempt writes raise `StorageGuardError` (ledger/ and registry/ exempt).

## Keepalive
`keepalive.yml` re-enables all workflows monthly (public repos disable schedules after inactivity).
