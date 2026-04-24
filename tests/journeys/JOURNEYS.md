---
last_updated: 2026-04-24
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.18.1
---

# v0.18 user-journey coverage matrix

Each journey is a 4-transition tile: **Entry → Precondition → Action → Terminal**.
The union of all journey tiles = the v0.18 CLI coverage lattice. Coverage is either
`unit` (individual transition tested in isolation), `integration` (end-to-end
transition chain), or `gap` (not asserted).

## J-A — First-run onboarding

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | bare `carl` with no `.initialized` on TTY | router calls `init_cmd` before any REPL | `test_cli_entry_router.py::test_first_run_diverts_to_init` | integration |
| 2 | `init_cmd` probes 7 signals | prints wizard, writes marker | `test_init.py::TestFullInitFlow::test_completes_end_to_end` | integration |
| 3 | `init_cmd --json` | emits `"status": "probed"` with 7 probe keys, no prompts | `test_init.py::TestJsonOutput::test_emits_structured_summary` | integration |
| 4 | idempotent re-run | already-initialized short-circuits Exit(0) | `test_init.py::test_already_initialized_short_circuits` | integration |

## J-B — Bare-prompt REPL entry

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `carl "analyze X"` on TTY | router classifies as `bare_prompt_entry` | `test_cli_entry_router.py::test_bare_prompt_routes_to_chat` | integration |
| 2 | trust precheck fires | once per untrusted project root | `test_cli_entry_router.py::test_trust_precheck_blocks_bare_entry` | integration |
| 3 | user chooses "trust_once" | registry.trust_root called | `test_cli_entry_router.py::test_trust_once_persists` | integration |
| 4 | chat_cmd invoked with initial_message | REPL starts with prompt pre-submitted | `test_cli_entry_router.py::test_bare_prompt_routes_to_chat` | integration |

## J-C — One-shot non-interactive (`-p`)

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `carl -p "q"` (any TTY state) | router extracts prompt via `_extract_print_prompt` | `test_cli_entry_router.py::test_print_flag_extracts_prompt` | integration |
| 2 | trust precheck **skipped** | even if project untrusted | `test_cli_entry_router.py::test_print_flag_bypasses_trust_precheck` | **gap** (documented intent; not asserted) |
| 3 | `ask_cmd(prompt)` invoked | one-shot path, no REPL | `test_cli_entry_router.py::test_print_flag_extracts_prompt` | integration |
| 4 | `-p ""` (empty) | router returns False → help | `test_cli_entry_router.py::test_print_flag_empty_falls_through` | integration |

## J-D — Explicit subcommand dispatch

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `carl session list` | `session` in REGISTERED_SUBCOMMANDS | `test_cli_entry_router.py::test_registered_subcommands_snapshot` | integration |
| 2 | router returns False | Typer dispatches normally | `test_cli_entry_router.py::test_subcommand_falls_through_to_typer` | integration |
| 3 | `carl --help` | router skips (help flag short-circuit) | `test_cli_entry_router.py::test_help_flag_short_circuits` | integration |
| 4 | `carl --version` | same as help | `test_cli_entry_router.py::test_version_flag_short_circuits` | integration |

## J-E — Trust lifecycle

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `trust status` (fresh) | prints "enabled: yes", acknowledged: "(none)" | `test_trust.py::TestCLI::test_status_fresh` | integration |
| 2 | `trust acknowledge <path>` | registry persists, prints path | `test_trust.py::TestCLI::test_acknowledge_writes` | integration |
| 3 | `trust acknowledge <new>` | prints prior-root eviction notice | `test_trust.py::TestCLI::test_acknowledge_replaces` | integration |
| 4 | `trust disable --force` / `trust reset --force` | no prompt, state updated | `test_trust.py::TestCLI::test_disable_force` / `test_reset_force` | integration |

## J-F — Session lifecycle

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `session list` | reads per-project sessions dir | `test_cli_session.py::TestSessionList::test_empty` | integration |
| 2 | `session show <id>` | prints JSON summary | `test_cli_session.py::TestSessionShow::test_existing` | integration |
| 3 | `session delete <id>` | file removed | `test_cli_session.py::TestSessionDelete::test_existing` | integration |
| 4 | project-root detection via walk-up | `_resolve_project_root` uses `project_context.current` | `test_cli_session.py::TestProjectRootResolution::test_walks_up` | **gap** (fix verified in fd862b0 but no journey-level assertion) |

## J-G — Project-context integrity

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `~/` never treated as project | home-guard in `_walk_up_for_project` | `test_project_context.py::TestHomeGuard::test_home_is_not_project` | integration |
| 2 | `.carl/` anchor detection | walk-up finds anchor above cwd | `test_project_context.py::TestWalkUp::test_finds_nested` | integration |
| 3 | scaffold idempotent | no-op when `.carl/` already exists | `test_project_context.py::TestScaffold::test_idempotent` | integration |
| 4 | fixture pattern: project at `tmp/proj` | tests pass HOME=tmp AND get a project root | applied across `test_parity_http_fixes.py`, `test_cli_resonant.py` | integration |

## J-H — Init `--json` probe contract

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `init --json` (fresh home) | emits `"status": "probed"` | `test_init.py::TestJsonOutput::test_emits_structured_summary` | integration |
| 2 | probe payload has 7 keys | `first_run_complete`, `camp_session`, `llm_provider_detected`, `training_extras_healthy`, `project_config_present`, `consent_set`, `context_present` | `test_init.py::TestJsonOutput::test_emits_structured_summary` | **gap** (asserts 4 keys, not all 7) |
| 3 | `init --json` (already initialized) | `"status": "already_initialized"` | `test_init.py::TestJsonOutput::test_already_initialized_json` | integration |
| 4 | probe never prompts on non-TTY | no hang on piped stdin | implicit (docstring claim, no direct test) | **gap** |

## J-I — Resonant publish

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `resonant whoami` | auto-generates `~/.carl/credentials/user_secret` mode 0600 | `test_cli_resonant.py::TestWhoami::test_generates_secret` | integration |
| 2 | `resonant list` | enumerates `~/.carl/resonants/` | `test_cli_resonant.py::TestList::test_lists_saved` | integration |
| 3 | `resonant publish <name>` | POSTs signed envelope, redacts headers | `test_cli_resonant.py::TestPublish::test_publishes` | integration |
| 4 | refuses non-HTTPS without `--dry-run` | Exit(2) | `test_cli_resonant.py::TestPublish::test_refuses_http_without_dry_run` | integration |

## J-J — Update freshness report

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `update --dry-run` | no network, no mutation | `tests/test_update.py` (pre-existing) | integration |
| 2 | `update --json` | structured FreshnessReport | pre-existing | integration |
| 3 | BreakAndRetryStrategy wraps PyPI | transient failures tolerated | pre-existing | unit |
| 4 | consent-gated network | off by default | pre-existing | integration |

## J-K — Doctor readiness

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `doctor` (fresh) | prints typed FreshnessReport | `tests/test_cli_startup.py` | integration |
| 2 | `carl.freshness.dep_corrupt` | surfaces with remediation hint | `tests/test_cli_startup.py` | integration |
| 3 | exit 0 when all green | no error codes | pre-existing | integration |
| 4 | exit 1 when severity>=error | actionable | pre-existing | integration |

## J-L — Flow op-chain

| # | Transition | Assertion | Coverage |
|---|---|---|---|
| 1 | `flow "/doctor /init"` | chains ops via OPERATIONS registry | `tests/test_cli_flow.py` (pre-existing) | integration |
| 2 | trace persisted | `~/.carl/interactions/<id>.jsonl` | pre-existing | integration |
| 3 | op failure halts chain | exit non-zero | pre-existing | integration |
| 4 | unknown op | actionable error | pre-existing | integration |

## Coverage summary

- **48 transitions total** across 12 journeys
- **44 covered** (`unit` or `integration`)
- **4 gaps identified** — all intentional-behavior claims without direct assertions:
  - J-C #2: `-p` bypasses trust precheck (docstring intent, no test)
  - J-F #4: session `_resolve_project_root` walks up (fix shipped fd862b0, no journey assertion)
  - J-H #2: init `--json` payload only asserts 4 of 7 probe keys
  - J-H #4: init `--json` never prompts on non-TTY (no direct assertion)

All 4 gaps are **reachable via journey tests** — no hidden or unreachable code surface.
No gap is a bug; each is an absent assertion.

## Isomorphism note

Every journey tile shares the same shape `(Entry, Precondition, Action, Terminal)`.
This is the DMC two-branch partition applied to CLI invocations:
- **Entry** = shell invocation string
- **Precondition** = filesystem/env state
- **Action** = router/handler that fires
- **Terminal** = exit code + persisted side effects

The journey matrix IS the `(M, I, Φ, G)` tuple restricted to the CLI manifold:
`M = argv space`, `I = transitions above`, `Φ = coverage status`, `G = test assertion`.
