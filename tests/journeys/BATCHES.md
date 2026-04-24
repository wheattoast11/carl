---
last_updated: 2026-04-24
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.18.1
---

# Parallel UAT batch spec

## Execution policy

Every batch is:
1. **Idempotent** — each test manages its own tmp_path / env / HOME; no shared state.
2. **Hermetic by default** — no network unless explicitly in an `online-*` batch.
3. **Bounded** — every test `--timeout=60` (thread-based); whole batch under 120s.
4. **Structured output** — pytest `-q --tb=line --junitxml=...` per batch.

## Pass/fail gate

A batch **passes** iff:
- pytest exit code 0
- `failed == 0`, `errors == 0` in the JUnit XML
- No new ruff or pyright diagnostics on touched files

A batch **fails** iff any of:
- Any test failure/error
- Pyright reports new strict-mode violations
- Ruff reports any violation

Agents must report **one line** on failure: first failing node-id + 3-line traceback excerpt.

## Offline batches (executable without credentials)

| Batch | Journeys | Test nodes | Owner | Runtime budget |
|---|---|---|---|---|
| **OFF-1: entry router** | J-A #1, J-B, J-C, J-D | `tests/test_cli_entry_router.py`, `tests/journeys/test_journeys_v18.py::TestJournyC_*` | agent-1 | 15s |
| **OFF-2: init wizard** | J-A #2-#4, J-H | `tests/test_init.py`, `tests/journeys/test_journeys_v18.py::TestJournyH_*` | agent-2 | 10s |
| **OFF-3: trust** | J-E | `tests/test_trust.py`, `src/carl_studio/cli/trust.py` surface | agent-3 | 10s |
| **OFF-4: session** | J-F | `tests/test_cli_session.py`, `tests/journeys/test_journeys_v18.py::TestJournyF_*` | agent-4 | 15s |
| **OFF-5: project-context** | J-G | `tests/test_project_context.py` | agent-5 | 10s |
| **OFF-6: resonant local** | J-I (local paths only) | `tests/test_cli_resonant.py` | agent-6 | 15s |
| **OFF-7: update/doctor/flow** | J-J, J-K, J-L | `tests/test_update.py`, `tests/test_cli_startup.py`, `tests/test_cli_flow.py` | agent-7 | 20s |
| **OFF-8: core parity** | all non-UI fixtures | `tests/test_parity_http_fixes.py`, `packages/carl-core/tests/` | agent-8 | 25s |

Total offline: ~120s wall clock when parallel, ~15s serial per batch.

## Online batches (gated on explicit user authorization)

**DO NOT EXECUTE WITHOUT TEJ'S EXPLICIT GO.** These hit live endpoints
with real credentials and will consume metered usage.

| Batch | Journeys | Endpoint hit | Credential required |
|---|---|---|---|
| **ON-A: carl.camp whoami** | J-I #1 (live) | `GET {CARL_CAMP_BASE}/api/resonants/whoami` | `CARL_CAMP_TOKEN` |
| **ON-B: carl.camp resonant publish** | J-I #3 (live) | `POST {CARL_CAMP_BASE}/api/resonants` | `CARL_CAMP_TOKEN` + `~/.carl/credentials/user_secret` |
| **ON-C: carl.camp agent register/publish** | J-H agent flow | `POST /api/agents/register`, `POST /api/sync/agent-cards` | `CARL_CAMP_TOKEN` |
| **ON-D: Anthropic one-shot** | J-C live | `POST api.anthropic.com/v1/messages` | `ANTHROPIC_API_KEY` |
| **ON-E: HF token probe** | `carl doctor` live | `huggingface_hub.whoami` | `HF_TOKEN` |
| **ON-F: PyPI freshness** | J-J live | `GET pypi.org/pypi/<pkg>/json` | none (public, but rate-limited) |

Online batches assert: request headers sent (redacted in logs), response
envelope matches the documented contract, directionality (client→server)
is correct, no accidental secrets in request body.

## Isomorphic state-machine per batch

Every batch emits the same end-state tuple:

```
{
  "batch_id": "OFF-1",
  "journeys": ["J-A-1", "J-B-1", "J-B-2", "J-B-3", "J-B-4", "J-C-1", ...],
  "tests_run": 47,
  "tests_passed": 47,
  "tests_failed": 0,
  "tests_errored": 0,
  "duration_s": 13.2,
  "ruff_violations": 0,
  "pyright_errors": 0,
  "gate": "pass",
  "first_fail": null
}
```

## Confluence review step

After all offline batches report, the coordinator (me) runs the confluence check:

1. Every journey tile in `JOURNEYS.md` has at least one `pass` in the batch reports.
2. Coverage matrix: every transition in every journey either (a) has a passing
   test or (b) is flagged as explicit `gap` with rationale.
3. No batch is in the "failed" or "errored" state.
4. If any online batch fires, its directionality assertions match the
   documented contracts in `docs/eml_signing_protocol.md` and
   `docs/v10_agent_card_supabase_spec.md`.

Only when all four conditions hold does the coordinator emit **sign-off**.
