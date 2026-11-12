# CLAUDE.md

Read `docs/DESIGN.md` first. It is the single source of truth for this project.

- **Everything the owner sees must be in Japanese**: replies, progress messages, tool descriptions and working notes. Only code identifiers, code comments and commit-internal text may be English. Never write English first and translate afterwards.
- Record every deviation from DESIGN.md as an ADR in `docs/DECISIONS.md`.
- Never edit: forecast ledger, PIT rules, promotion gate thresholds, or DESIGN.md principles (Section 0.2) without owner approval.
- Never commit data or secrets. Work on branches and open PRs.
- Merging (owner decision 2026-11-12, ADR-0021): in THIS repository Claude may merge its own PRs when (1) CI is green on the last commit and (2) a `reviewer` subagent, run on that last commit, reports no 必須 (must-fix) item. If review findings are fixed, rerun CI and the review before merging. Report each merge with the PR URL. This overrides the global "humans merge" rule for this repository only.
  Leave for the owner to merge:
  - the forecast ledger, PIT rules (`src/mw/core/pit.py`), promotion gate thresholds, DESIGN.md principles (Section 0.2)
  - this merge rule itself, ADR-0021
  - `.github/workflows/ci.yml`, and any change that removes, skips or weakens tests or checks
  - secrets, workflow permissions, repository visibility
  - anything that adds paid services or raises API / storage costs
  - deleting data
- Model-architecture or validation-protocol changes: record an ADR in the same PR (ADR-0021).
