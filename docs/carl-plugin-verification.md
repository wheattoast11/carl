---
last_updated: 2026-10-07
author: CARL project
applies_to: carl-studio
---

# CARL plugin implementation and verification

Source baseline: `3d8deda1d52702f7364520f7f8990efed67a698d`.
Pre-existing dirty files: none.
Release snapshot: `2026-10-08T00:35:32Z` (October 7, America/Chicago).
Writer: parent agent. Parallel agents perform read-only research and review.

## Acceptance contract

- [x] Python 3.12+, one compatible root lock, reviewed dependency exceptions.
- [x] Portable plugin and canonical `carl` skill, native host projections.
- [x] Owned, repeatable install/update/doctor/uninstall with bare skill aliases.
- [x] FREE local MCP startup through the connection owner; loopback HTTP default.
- [x] Native Codex, Claude Code, and OpenCode delegation over existing task/process owners.
- [x] Scoped permission replies, bounded capacity, atomic terminal states, execution cancellation.
- [x] Metadata-only delegation traces and explicit result artifacts.
- [x] Shared skill registry and observable A2A progress/results.
- [x] Focused positive/negative tests, safe regressions, lint/types, package and codec builds.
- [x] Fresh-process native host witnesses, final independent review, reduction table.

## Dependency decisions

The all-extra environment follows SGLang 0.5.21's upstream constraints:
Torch 2.13.0, Transformers 5.12.1, tokenizers 0.22.2, NumPy 2.3.5 on Python 3.12,
Hub 1.33.0, TRL 1.14.2, PEFT 0.21.2, datasets 5.1.0.
CUDA Tile 1.6.0rc5 and FlashAttention 4 beta are required upstream exceptions.
Do not override their constraints to force a lock or install.

The wallet extra stays separate from all/x402. Preserve its API and encrypted
state. Record exact advisories and applicability for retained legacy dependencies.
An advisory match does not establish that CARL executes the affected operation.

## Verification ladder

1. Baseline and focused deterministic checks.
2. Real MCP wire, native protocol fixtures, and positive/negative task contracts.
3. Owning subsystem and safe offline regression suites.
4. Host projections and unchanged codec/signature parity fixtures.
5. Python/TypeScript production builds and installed-artifact imports.
6. Fresh native host witnesses; separately authorized paid/device/user-data effects.

Record command, exit status, count, and evidence class for each executed check.
Keep source inspection, fake-provider native execution, and live provider evidence separate.

## Evidence

Execution used Python 3.12.12 and the complete all-extra environment.
The release/advisory lookup checked 371 registry entries and is bound to
`uv.lock` SHA-256 `c105fff653049f60acac830762eab19d481ae7cfce4b6ad300cffcf3c3ff6234`.

| Command or contract | Executed result | Evidence class |
| --- | --- | --- |
| `pytest tests/ packages/carl-core/tests/ -q --tb=short` | 4,318 passed, 29 skipped, 17 warnings | Offline execution, complete installed all-extra graph |
| Focused native continuation and MCP protocol checks | 21 passed, including forged HTTP identity, validation-log canary, rejected native acknowledgement and concurrent replay | Actual SDK with native process fixtures |
| Targeted Ruff on new harness, plugin, protocol and test files | Passed | Static analysis |
| Strict Pyright on new harness, plugin and protocol modules | 0 errors, 0 warnings | Static analysis |
| `uv lock --check --offline` using writable build cache | Resolved 373 packages, exit 0 | Resolver execution |
| `python scripts/check_moat_boundary.py` | 339 files, 0 forbidden top-level private imports | Executed source gate |
| `python -m build` and `python -m build packages/carl-core` | Studio 0.20.1 and core 0.2.0 wheels and source distributions built | Artifact builds |
| `npm test` in `packages/emlt-codec-ts` | 108 passed, 0 failed | Codec and Python golden-vector parity |
| Built-wheel import smoke | Lightweight base import and harness/protocol imports passed; packaged Python source bytes match the checkout | Executed built artifacts |
| `npm run build` | ESM, CommonJS and declaration outputs built | TypeScript 7 production build |
| Native installer in disposable profiles | Install, repeat install, uninstall passed for all three hosts | Actual host CLI execution |
| `carl plugin install`, `update`, `doctor` in operator profiles | Three hosts installed, no source/projection/alias drift | Actual profile activation |
| Native Codex, Claude Code and OpenCode witnesses | Metrics and cancellation passed for each host | Actual host binaries, synthetic localhost providers |
| Independent Compound Engineering recheck | Nine findings closed, no open findings | Source inspection and offline mechanical counterexamples |

Native versions: Codex 0.160.1, Claude Code 2.1.292, OpenCode 1.18.32.
Each metrics witness required one native permission reply and actual CARL metrics
in the second provider request. Returning the fixture's final text alone fails.
Each cancellation witness required terminal cancellation without an escaped error.
Commands, source hashes and outcomes are in
`carl-native-verification-2026-10-07.json`.
Installed Codex and Claude caches match the canonical skill and final native
manifest version `0.20.1+source.cb55fa98ab56`. Fresh Codex startup reached ready
and exposed 27 normal-scope tools, including delegation and task controls.
Fresh OpenCode reported CARL connected. The installed stdio carrier executed
coherence metrics with embedding dimension 3072 without a model/provider call.
`carl-installed-verification-2026-10-07.json` records these source/cache bindings.

The independent review's twelve file hashes were rechecked against this tree;
its closure is in `carl-review-closure-2026-10-07.json`.

The expanded Ruff check on changed existing files retains pre-existing debt;
comparison with the baseline under the same current rule set found 255 current
versus 277 baseline diagnostics and no new source diagnostics after cleanup. These broad legacy checks are not reported green.

No paid provider, GPU model training, deployment, publication, or existing
operator database migration was part of these checks. Private-runtime tests
retain their skips. Legacy wallet and unpatched upstream advisory matches are
recorded in `dependency-review-2026-10-07.md`.

## Protocol and resource boundaries

The bound stdio connection supports both MCP protocol eras. Stateless modern
HTTP delegation refuses requests without trusted caller identity; a supplied
session header does not confer ownership. Legacy HTTP uses server-issued identity.
A2A exposes delegated progress/results; cancellation explicitly directs the
caller to the submitting MCP connection.

One process owns each delegate, with a two-task admission limit, a maximum depth
of one and a bounded deadline. Unstarted reservations whose owner died can be
released. Launched tasks with lost execution custody require operator reconciliation
against the native process before reclaiming their slots.

Instructions, pending approval details and reasoning are not operational trace
payloads. Final text is an explicit owner-readable result artifact. Local host
profiles, permissions and process environments are isolated for delegation.

The restricted test sandbox blocks asyncio thread wakeups through Unix sockets.
A trivial `asyncio.to_thread` probe times out there and succeeds with local IPC
permitted. Run asynchronous/native tests with that permission before diagnosing
an SDK or adapter timeout.

## Reduction table

| File | Candidate | Disposition | Correspondence | Authority | Effects | Residuals | Terminal state | Named check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| harness/runtime.py | Reuse `carl_core.hashing.content_hash_bytes` | Applied | Same SHA-256 byte digest | Same owner and request binding | No new effect | Same result hash and replay conflict | Unchanged | Result artifact and continuation replay tests |
| harness/runtime.py | Require live context and generation | Applied | Same admitted context | Removes fallback to broader context | Same native launch | No implicit fallback | Unchanged | Scope, write-grant and stale-reply tests |
| mcp/tasks.py | Query matching identity and active rows only | Applied | Same global two-slot accounting | Same owner/request key | Same transactional admission | Launched orphan custody retained | Same terminal CAS | Capacity, dedup and dead-owner tests |
| harness/runtime.py | Release terminal live-task entries | Applied | Durable task remains queryable | Same result owner | Releases instructions and adapters | Explicit result artifact remains | Same completed/failed/cancelled result | Native result and cancellation tests |
| harness/bridge.py | Close the owned event loop | Applied | Same Session bridge | Same execution owner | Executor and loop reaped | Idempotent close | Owned delegates stopped first | Chat-tool and shutdown checks |
| harness/adapters.py | Collapse hosts to one generic protocol | Rejected | Native protocols differ | Permissions differ by host | Interrupt mechanisms differ | Native acknowledgements required | Preserve each native terminal result | Six native host witnesses |

## Installed command identity

Operator `carl` and `carl-mcp` launch `/var/home/zero/carl/.venv/bin/` entrypoints.
The previous generated Python 3.14 `carl` script failed to import CARL; its exact
bytes are preserved at `~/.carl/launchers/carl.before-2026-10-07`.
Launcher hashes are at `~/.carl/launchers/receipt-2026-10-07.json`.
The portable install instructions activate the project environment before bare
CLI use. Plugin removal retains the separately installed CARL CLI.

## Rollback

Restore explicit changed paths from the recorded baseline after preserving local edits.
Uninstall removes only entries whose content matches the installer ownership record.
User-data migrations are prepared and tested on temporary stores before separate live approval.
No deployment or publication is part of this checkout change.
