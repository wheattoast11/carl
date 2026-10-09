---
last_updated: 2026-10-09
author: Codex
applies_to: carl-studio 0.21.1 source checkout
---

# Encoder upgrade verification

Source baseline: `8b1f974`.
Pre-existing change preserved: `src/carl_studio/compute/runpod.py` (L40S mapping).
Main dependency lock and packaging versions are unchanged.

## Source owners

`Session.semantic` owns the shared service. Chat, MCP and CLI use that service.
A2A conversion preserves ordered structured/media parts alongside legacy text.
KnowledgeStore and MemoryStore retain their lexical interfaces and expose
separate semantic recall results. Derived carriers use DataVault file artifacts.

`ConsentManager` owns session/workspace capture grants. Capture does not submit
training. Accepted records commit through MemoryStore once per declared event,
context, goal, interpretation revision and confirmed artifact. Corrections keep
successor lineage. The data registry projects captured records locally.

`TrainingConfig.method=encoder` branches through preparation and CARLTrainer
before the causal loader. The isolated worker implements frozen ranking/relation
heads and PEFT query/value adapters. It uses the differentiable Transformer and
Pooling forward path, retaining the raw carrier before Normalize.

Encoder preparation binds sources, interpreter, dependencies, processor and
matched modules. ExperimentManager retains execution admission and recorded
results. Checkpoints contain parameters, optimizer, scheduler, RNG, step and
data position. Representation acceptance is separate from generative token Phi.
Local activation requires acceptance and a declared workspace effect.

## Executed evidence

Model: `google/embeddinggemma-2`.
Revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`.
Model directory: `/tmp/maxwell-embeddinggemma2-model-20261006`.
Interpreter: `/tmp/maxwell-embeddinggemma2-env-20261006/bin/python`.
Environment: Transformers 5.19.0, SentenceTransformers 6.1.0.
The main CARL environment was not modified to install those versions.

Real CPU encoding produced a finite 768-dimensional raw carrier.
The initial frozen-head pilot executed 128 optimizer steps and updated
`ranker.weight`, `relation.weight`, and `relation.bias`.
The initial adapter pilot executed one optimizer step, matched 48 query/value
modules, and changed 96 encoder LoRA parameters plus the three head parameters.
Both immediate prepared-submission replays returned the recorded result.

Pilot source: cached `wheattoast11/tinny-single-agent-sft-v1`.
Cache revision: `ea7e3f8f03de6b61ab1ad2d1324c19c99b98eb49`.
Population: two train, one validation, two held-out episodes.
Targets: source-declared JSON tool actions.
Negatives: explicit wrong-name action transformations made after grouping.
Provenance: Arrow byte hashes and original row IDs.
Held-out ranking accuracy: 1.0 before and after both learning modes.
No action improvement was established and no workspace candidate was activated.

Initial pilot logs and checkpoints:
`/tmp/carl-encoder-pilot/frozen_heads.log`,
`/tmp/carl-encoder-pilot/adapter.log`,
`/tmp/carl-encoder-pilot/frozen_heads/measurements.json`,
`/tmp/carl-encoder-pilot/adapter/measurements.json`.
These executions predate subsequent source repairs and are not final-source
training qualification.

## Protocol and regression checkpoints

Offline counterexamples cover raw-prefix restriction, retained-tail recovery,
scoring, collapse, unchanged candidates, missing measurements, retention loss,
zero lexical overlap, concurrent commits, source drift, feedback atomicity and
restart replay. Structural media tests use authored offline fixtures.

Historical focused checkpoints: 48 passed; later prepared/semantic/learning/
agent checkpoint: 98 passed. Strict Pyright passed for semantic modules,
training/encoder.py and cli/semantic.py at that checkpoint.

The historical full suite recorded 4,433 passed, 29 skipped and three failures:
two tool-schema expectations and predecessor reference restoration.
The source expectations and restoration were repaired. Subsequent changes need
another complete verification pass; this historical count is not a final pass.

The Codex native semantic witness passed through a real installed host and a
synthetic localhost chat provider. It executed the real local encoder.
Receipt: `build/native-probes/codex-semantic-ff72d5/receipt.json`.
That receipt predates the latest source repairs.
Canonical skill projections were generated into `build/carl-marketplace`.

## Remaining acceptance work

Re-run targeted tests, Ruff, strict Pyright, core/studio regressions and release
checks against the final source. Re-run native semantic witnesses for all three
hosts. Build and inspect fresh wheel imports without changing the dependency
lock. Model jobs require independent fresh memory admission and sequential coordination with Root.

The four-condition comparison with the same real chat model, tasks, tools and
budgets has not executed. The small pilot does not establish a correction that
improves a held-out action. Independent action evaluation, required modality
retention, accepted activation and its replay remain required for that case.


## Shipping verification (2026-10-09)

Final studio regression: 3,656 passed, 19 skipped.
Final core regression: 784 passed, 13 skipped.
Focused source, A2A, consent and agent checks: 121 passed; release checks also passed in the final targeted run.
Strict Pyright: zero errors for semantic modules, encoder training, semantic CLI
and the captured interpretation adapter.
Ruff: the new modules and tests pass; pre-existing touched-file debt is retained.

Studio and core source distributions and wheels built offline using the existing
`build/uv-cache`. No dependencies were added to the main environment or lock.
Both wheels were installed without dependencies into a temporary target;
imports were checked against that target rather than the editable checkout.
The studio source archive excludes the build cache and virtual environment.

Input-selected tower loading uses the publisher's `config_kwargs` recipe.
The text-only worker peaked at 2,436,505,600 RSS bytes under a 2.5 GiB cap.
Its 768-dimensional carrier matched the earlier full-tower carrier exactly
(maximum absolute difference 0.0 on the same authored smoke input).
No model, dataset, dtype or processor revision was substituted.

The initial 1 GiB combined regression scope stopped at its resource limit.
Studio was then independently admitted at 1.5 GiB and core at 1 GiB.
Only owned test process groups were eligible for termination; no unrelated
process or service was changed.

Remaining model acceptance and final native-host witnesses are separate from
these source, regression and packaging checks. No candidate activation occurred.


Final-source native semantic receipts:
- Codex: `build/native-probes/codex-semantic-6c4039/receipt.json` (passed).
- Claude: `build/native-probes/claude-semantic-ec8dee/receipt.json` (passed).
- OpenCode: the 3 GiB owned-tree scope exceeded its budget and stopped.
  Its two surviving owned descendants were identified by exact scope cwd and
  executable name, terminated, and their absence confirmed. The retry remains
  queued for fresh 3.25 GiB plus reserve admission and Root sequencing.

The initial Codex scope monitor omitted children launched from worker threads.
The protocol receipt is valid; its launcher-only RSS peak is not a scope peak.
The monitor was corrected to traverse every thread's child list before Claude.
Claude's total owned-process peak was 2,870,018,048 bytes.
