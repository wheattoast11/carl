---
last_updated: 2026-04-24
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.18.1
---

# v0.18 journey coverage + UAT discipline

This doc is the provenance stamp for the 2026-04-24 journey-coverage sweep.

## What was verified

- **Baseline:** 164 v0.18 surface tests green on `fd862b0` (`test_cli_entry_router`,
  `test_cli_session`, `test_trust`, `test_project_context`, `test_init`,
  `test_cli_resonant`, `test_parity_http_fixes`). Run: `0.93s`.
- **Journey matrix:** 12 journeys × 4 transitions = 48 transitions, tabulated
  at `tests/journeys/JOURNEYS.md`.
- **Coverage gaps found:** 4 — all "behavior present but not asserted end-to-end".
- **Gap closures:** 8 new tests in `tests/journeys/test_journeys_v18.py`.
- **Post-closure:** 172 tests green; ruff + pyright clean on touched files.

## Journey surface (v0.18.1)

The 12 journeys partition the CLI manifold:

| Journey | Entry | Expected terminal |
|---|---|---|
| J-A | `carl` with no `.initialized` on TTY | init wizard + marker written |
| J-B | `carl "<prompt>"` on TTY | trust precheck → REPL with first turn |
| J-C | `carl -p "<q>"` | one-shot ask; trust precheck SKIPPED |
| J-D | `carl <verb>` | Typer dispatch (router returns False) |
| J-E | `trust status/acknowledge/disable/reset` | trust registry state |
| J-F | `session list/show/delete` | per-project session files |
| J-G | project-context walk-up | `.carl/` anchor detection with home-guard |
| J-H | `init --json` | probe-only 7-key payload, no prompts |
| J-I | `resonant whoami/list/eval/publish` | local crypto + HTTPS-only publish |
| J-J | `update --dry-run` / `--json` | FreshnessReport without mutation |
| J-K | `doctor` | typed readiness report, exit 0/1 by severity |
| J-L | `flow "/op1 /op2"` | op-chain with trace persisted |

## Isomorphism

Each journey tile has the same 4-transition shape
`(Entry, Precondition, Action, Terminal)`. The union of tiles is the
coverage lattice.

This is the DMC two-branch partition applied to CLI invocations. The
(M, I, Φ, G) tuple for the CLI layer:

- **M** = argv + filesystem/env state space
- **I** = 48 transitions (the union of all journey rows)
- **Φ** = coverage status (`unit` / `integration` / `gap`)
- **G** = test assertion as gating predicate

## What was NOT verified in this pass

- Live carl.camp endpoints (ON-A / ON-B / ON-C in
  `tests/journeys/BATCHES.md`) — gated on explicit authorization.
- Live Anthropic one-shot (ON-D) — gated on explicit authorization.
- Live HF token probe (ON-E) — gated on explicit authorization.
- Full-suite cross-batch regression — current run is targeted (~172
  tests of the v0.18 surface). Running the full 3937-test suite is a
  separate gate and was intentionally scoped out of this pass.

## Doctrine updates

**Retrofit ≠ TDD.** Journey-coverage tests for already-shipped code
are *coverage* tests, not TDD tests. They cannot "fail first" because
the behavior already exists. The honest discipline is:

1. Write the journey assertion.
2. Run it; verify it passes.
3. (Optional) Do a mutation check — temporarily break the behavior,
   confirm the test catches it, revert.

Mutation checks are expensive; reserve them for assertions where the
behavior is non-obvious or the regression risk is high.

**TDD still applies to new features.** For any new feature added after
the journey matrix lands, follow the Red-Green-Refactor cycle in
`superpowers:test-driven-development`. The journey matrix becomes a
coverage baseline new features must extend, not replace.
