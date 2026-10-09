---
last_updated: 2026-10-08
author: Codex
applies_to: carl-studio 0.21.0
---

# Prepared training verification

CARL prepares a bounded experiment, consumes matching reviewed inputs through
the existing trainer or SendItPipeline, compares the baseline and candidate,
and retains the checkpoint and acceptance result at the experiment owner.

The source baseline is `5be38462a9c3dba089e24f1386a945dc2469b99c`.
The feature branch is `codex/carl-prepared-training`.
The CARL checkout was clean before this work. The parent agent owns the changes;
reviewers were read-only. The approved construction contract is in
[the plan](plans/2026-10-08-carl-prepared-training.md).

## Entry points

```bash
carl train --prepare --config carl.yaml --goal "Improve held-out task success"
carl train --prepared-plan Eprep_YOUR_PLAN_ID --config carl.yaml --dry-run
carl train --prepared-plan Eprep_YOUR_PLAN_ID --config carl.yaml
```

MCP `prepare_training` returns `plan_id`. Pass that value as `prepared_plan_id`
to `start_training` or `submit_async_training` after authorization.
Python pipelines can use `prepare_training` and `submit_training`, or attach
their task rewards and evaluator directly to `CARLTrainer` and `EvalRunner`.

The first goal integration uses local TRL SFT and GRPO. Preparation checks
source files, local data, split separation, callable declarations, dependencies,
step limits and adapter capabilities. It does not execute the grader or load
model weights. Other backend goal hooks require their own qualified integration.
Remote legacy configuration calls retain their existing route.

Primary metric thresholds use the existing evaluation gate's [0, 1] scale.
Acceptance requires the absolute goal target, strictly positive improvement
beyond the declared margin, required policy results and complete full-logit
coherence measurements. Missing required measurements yield `inconclusive`.
Execution completion and candidate acceptance remain separate.

## Executed checks

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/ packages/carl-core/tests/ -q --tb=short -p no:cacheprovider
```

Result: **4,378 passed, 29 skipped, 77 warnings**, 88.39 seconds.
Warnings include existing marker and dependency deprecations; the new
ExperimentManager cases also exercise its existing `datetime.utcnow` default.
Tests use mocked model/trainer boundaries for weight updates and generation.
Real owner persistence, evaluator logic, callbacks, executor cleanup, task tools
and MCP transport run inside those tests.

```bash
.venv/bin/ruff check src/carl_studio/types/preparation.py src/carl_studio/training/preparation.py src/carl_studio/training/acceptance.py tests/test_prepared_training.py tests/test_pipeline.py tests/test_mcp_v2_wire.py
.venv/bin/pyright src/carl_studio/types/preparation.py src/carl_studio/training/preparation.py src/carl_studio/training/acceptance.py
git diff --check HEAD
```

Ruff passed on the new modules and selected test files.
Strict Pyright returned zero errors and zero warnings.
Whitespace checks passed. Across the complete changed Python set, comparison
with the baseline found zero introduced Ruff diagnostics; existing diagnostics
numbered 164 at baseline and 160 after this change.

The installed TRL 1.14.2 `SFTConfig` and `GRPOConfig` constructors accepted the
actual trainer arguments with `max_steps=64`, publication disabled, `bf16=False`
and reporting disabled. This was a CPU configuration check without model loads.
An optional bitsandbytes CPU-kernel trust warning was emitted. No dependency or
trust setting changed.

## Falsifiers and repair checks

| Contract | Executable check | Result |
|---|---|---|
| Preparation needs real inputs | Missing data, absent verification, overlapping tasks, unsupported hooks and unbounded steps | Refused before trainer construction |
| Reviewed sources control execution | Model templates, model membership, helpers, package files and resume membership change | Stale preparation refused |
| Reward measures the declared task | Long wrong answer versus actual target; malformed count, NaN and infinity | Wrong answer scores zero; malformed rewards refused |
| Comparison requires improvement | Unchanged and regressed candidates; high coherence with wrong task; policy failure; missing coherence | Rejected or inconclusive as specified |
| Starting lineage survives training | Base, original adapter A, SFT adapter B, GRPO adapter C | Ordered loading retained |
| Selected tokenizer controls both evaluations | Real-owner loop and stage-gate configuration capture | Bound source propagated |
| Dry run preserves the execution boundary | Prepared CLI dry run with a submission spy | No submission |
| Cancellation retains custody | Registered stop/save callback, repeated cancellation, active pipeline stage and task-tool cleanup | Cleanup awaited; actual stage saved |
| Status queries remain responsive | Legacy and modern MCP requests during blocked fixture hashing | Query completes while preparation is pending |
| Errors and settings preserve privacy | Malformed YAML, module-import canary and literal credential passthrough canary | Constant errors; credentials refused before persistence |
| Replays avoid a second training effect | Repeated submission through real ExperimentManager | Prior result reused |
| Failed gates stop promotion | Single-stage and SFT/GRPO pipeline negatives | Publication not invoked |

The original pipeline gate regressions failed before repair.
Lower-direction and repeated-cancellation regressions failed before repair.
Six final review regressions failed before the corresponding repairs.
The final suite includes each positive and negative case.

## Build and native witnesses

```bash
UV_CACHE_DIR=/tmp/carl-uv-cache uv build --offline --out-dir build/prepared-training
.venv/bin/python scripts/project_carl_plugin.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/witness_native_harness.py codex prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/witness_native_harness.py claude prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/witness_native_harness.py opencode prepare
```

The source distribution and wheel built using the existing offline cache.
All 297 packaged Python files matched source bytes. An isolated wheel install
imported the public preparation APIs without loading torch, transformers or TRL.
The canonical skill generated its repository host projection.

Each real native host completed the actual `prepare_training` tool loop with
a synthetic localhost provider. The witness requires the prepared result in
the next provider request; final text alone cannot satisfy it. Preparation,
MCP and harness source hashes were rechecked against each receipt.
These witnesses execute native protocol and permission handling, not training.
Receipt paths and artifact hashes are in
[the machine-readable record](carl-prepared-training-receipt.json).

## Independent review and reductions

Compound Engineering review completed with ten local persona outcomes and
an independent finding validator. The parent applied all eight justified
findings. A separate current-source closure check returned no residual findings.
External code review was not used; the local adversarial fallback ran.
The durable record retains the review and closure results.

Three read-only simplification lenses covered reuse, quality and efficiency.

| Candidate | Disposition | Preservation and check |
|---|---|---|
| Duplicate implementation-source lists | Shared helper | Same paths, authority, effects, failure state and stale checks; preparation suite passed |
| Duplicate goal-progress predicate | Shared pure predicate | Same threshold, direction and margin; acceptance and witness tests passed |
| Duplicate executor cancellation handling | Shared context-preserving worker helper | Same cleanup boundary; callback, repeated cancellation, gate and wire tests passed |
| Typed empty-list factories | Kept | Preserve strict type inference; Pyright passed |
| Parallel reward argument lists | Kept | Preserve the current injection surface and alignment checks; trainer tests passed |
| Hash checks at multiple boundaries | Kept | Detect edits before execution and after evaluation; stale-input checks passed |
| Full local-data parsing during preparation/evaluation | Kept | Preserve validation of the bound file and complete split checks |
| Stage and final candidate evaluation | Kept | Preserve the existing stage gate and separate policy/comparison evaluation |
| Per-sample allocator cleanup | Kept | No model-memory witness authorized for an allocation change |

## Boundaries and rollback

Actual weight updates, device execution, measured Tinny improvement, remote
provider spending, migrations, service changes and public publication were not
executed. They need their selected model/data inputs and resource authorization.
Caller Python graders run as trusted project code during authorized execution.
Running work without live worker custody remains in its recorded state until
reconciliation; an unowned cancellation cannot report a stopped worker.

The exact reviewed Tin transcript was deleted after its SHA-256 was matched.
The plan retains the sanitized findings and source identity. No transcript or
credential is retained in these records. Other Tin files were preserved.

Rollback reverts the explicit feature change on its branch. Existing experiment
records and checkpoints remain owner-readable artifacts. Host installation,
public release, push and deployment were not performed.
