# CLAUDE.md

Read `docs/DESIGN.md` first. It is the single source of truth for this project.

- **Everything the owner sees must be in Japanese**: replies, progress messages, tool descriptions and working notes. Only code identifiers, code comments and commit-internal text may be English. Never write English first and translate afterwards.
- Record every deviation from DESIGN.md as an ADR in `docs/DECISIONS.md`.
- Never edit: forecast ledger, PIT rules, promotion gate thresholds, or DESIGN.md principles (Section 0.2) without owner approval.
- Never commit data or secrets. Work on branches and open PRs; the owner merges.
