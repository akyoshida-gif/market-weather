# CLAUDE.md

Read `docs/DESIGN.md` first. It is the single source of truth for this project.

- **Everything the owner sees must be in Japanese**: replies, progress messages, tool descriptions and working notes. Only code identifiers, code comments and commit-internal text may be English. Never write English first and translate afterwards.
- Record every deviation from DESIGN.md as an ADR in `docs/DECISIONS.md`.
- Never edit: forecast ledger, PIT rules, promotion gate thresholds, or DESIGN.md principles (Section 0.2) without owner approval.
- Never commit data or secrets. Work on branches and open PRs.
- Merging (owner decision 2026-11-12): in THIS repository Claude may merge its own PRs after (1) CI is green and (2) a `reviewer` subagent review finds no serious issue; report each merge with the PR URL. This overrides the global "humans merge" rule for this repository only.
  Leave for the owner to merge: anything touching the forecast ledger, PIT rules, promotion gate thresholds, DESIGN.md principles (Section 0.2), secrets or workflow permissions, or deleting data.
