---
last_updated: 2026-04-21
author: Claude Opus 4.7 (1M context) + Tej Desai
applies_to: v0.17.0-planned
classification: internal — Team D Observability Sprint readout
---

# Team D Readout — Observability Sprint

## Scope honored

Three of five workstream items shipped; two explicitly deferred.

- **D1:** `@audited` decorator + `InteractionChain` contextvar — new module `carl_core.audit`. Chain is pulled from `CURRENT_CHAIN` contextvar at call time; supports nested scopes via `chain_context(...)` context manager; resets cleanly on exceptions. Declarative replacement for the imperative `chain.record(...)` pattern that was spread across 12 files / 61 call sites per the v0.17 plan data.
- **D3:** Content-addressed steps. `Step.to_dict()` now appends a `content_hash` field (sha256 of canonical-JSON over the rest of the dict). `verify_step_content_hash(dict) → bool` helper detects any post-serialization tampering. Computed over the secret-scrubbed form so the hash never embeds un-scrubbed values. Stable across reserializations; survives JSONL round-trip.
- **D5 (this doc):** structured team readout.

## Scope cut (with rationale)

- **D2 — Bulk migration of 5 high-value sites.** Deferred. Team D ships the machinery; migration is opportunistic per §10.10 policy. Teams B / C / E will migrate as they touch the relevant files: Team B migrates training/pipeline.py during the TrainingSpec work, Team C migrates chat_agent's tool loop during the split, Team E can sweep the remaining sites during the documentation pass. Forcing a bulk refactor here would either couple Team D to downstream team scope OR leave ripped-up intermediate state waiting for weeks.
- **D4 — Unified scrubber registry.** Explicitly deferred in Tej's decision log (v0.17 plan §13 item 5). Cosmetic consolidation of `interaction._SECRET_PATTERNS` + `cu/privacy.py` is a v0.18 candidate.

## Metrics

| Metric | Value |
|---|---:|
| New modules | `carl_core.audit` (195 LOC) |
| Existing modules extended | `carl_core.interaction` (+35 LOC for `_step_content_hash` + `verify_step_content_hash`) |
| Tests added | 22 (`packages/carl-core/tests/test_audit.py`) |
| Tests regressed | **0** |
| Full-suite count | 3,729 pass, 3 skipped, 0 fail (up from 3,707 Team F baseline) |
| Full-suite timing | within +10% ceiling of plan §6 |
| Pyright strict errors on new/modified files | **0** |
| Ruff warnings on new/modified files | **0** |

## Blast radius (actual vs predicted)

Predicted per §1.4: 12 files / 61 occurrences eligible for decorator migration. Team D actual touch: 2 files (`carl_core/audit.py` new; `carl_core/interaction.py` extended). The migration blast (12 files) is deferred per §10.10.

## Decisions made

1. **Contextvar over thread-local.** `asyncio.run` + structured concurrency require the current chain to propagate into spawned tasks. Thread-locals don't deliver that; `contextvars.ContextVar` does. Matches the plan §2.5 design.
2. **No-op passthrough when no chain is bound.** A decorated function called outside a `chain_context(...)` runs exactly as if undecorated — no audit noise for library consumers or one-off scripts. Tested (`test_audited_is_noop_when_no_chain_bound`).
3. **Content-hash computed over scrubbed form.** The hash reflects what's actually on disk (the secret-scrubbed payload). Re-hashing the serialized form produces the same hash — which is the right verification semantics AND it means a compromised logger can't trivially smuggle secrets by exploiting a mismatch between hashed + on-disk forms.
4. **Decorator composes with existing `chain.record` imperative calls.** Not exclusive. Migration happens file-by-file; both forms coexist during the transition.
5. **`name_fn` receives the wrapped function's args.** Matches the common case where the step name should embed a method argument (batch id, phase id, tool name). Exception inside `name_fn` is swallowed — bad naming shouldn't tank the audit.
6. **Failure path still records a step.** When the wrapped function raises, we still emit a `success=False` step with `duration_ms` and the exception's repr as output. The exception re-raises; decorator never swallows.
7. **Chosen decorator signature over class-based.** Class-based (`@Audited(...)`) would let users subclass for custom behavior, but the migration target is mass-use on existing methods — functional decorator ergonomics win. Users who want custom behavior write a wrapper layer.

## Content-addressing notes

The step hash extends naturally into the TwinCheckpoint hash shipped in Day 3: `TwinCheckpoint.content_hash` covers the whole chain, which now includes per-step content hashes. Result: a twin checkpoint is doubly-verifiable — tampering with any step body fails step-level verification; tampering with the checkpoint metadata fails checkpoint-level verification. This is the load-bearing primitive for v0.18+ replay, cross-process chain merging, and future Terminals-OS-kernel chain-integrity audits.

## Handoff to next workstreams

- **Team B** — The `@audited` decorator is ready. Migrate training/pipeline step methods as part of the TrainingSpec work: `@audited(ActionType.TRAINING_STEP, name_fn=lambda self, batch: f"step.{batch.id}")`.
- **Team C** — The post-split `chat_agent/` package's tool-loop body should wrap tool dispatches in `@audited(ActionType.TOOL_CALL, ...)`. Combined with the new contextvar, the chain flows naturally through async tool chains.
- **Team E** — Canonical usage example belongs in `docs/v17_audit_doctrine.md` + an entry in CLAUDE.md's "Mental model" section. Prose captured below, ready to lift.

## Canonical usage

**Decorator:**

```python
from carl_core.audit import audited, chain_context
from carl_core.interaction import ActionType, InteractionChain

@audited(
    ActionType.TRAINING_STEP,
    name_fn=lambda self, batch: f"step.{batch.id}",
    input_fn=lambda self, batch: {"batch_size": len(batch.ids)},
    output_fn=lambda result: {"loss": result.loss, "lr": result.lr},
)
def train_step(self, batch: Batch) -> TrainResult:
    ...

# Bind a chain for a scope:
chain = InteractionChain()
with chain_context(chain):
    result = pipeline.train_step(batch)   # ← audit lands in `chain`
# chain.steps now contains one TRAINING_STEP step with duration_ms set.
```

**Content-hash verification:**

```python
from carl_core.interaction import verify_step_content_hash

for step_dict in persisted_chain:  # loaded from JSONL
    if not verify_step_content_hash(step_dict):
        raise AuditError(f"step {step_dict['step_id']} hash mismatch — tampered")
```

## Sign-off

Team D machinery ships clean; 22 regression tests pin behavior. Content-addressable audit chain is now a foundational invariant. Bulk migration deferred as an opportunistic sweep across Teams B/C/E.
