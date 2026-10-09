---
last_updated: 2026-10-09
author: null
applies_to: carl-studio 0.22 and carl-encoders 0.1
---

# Reusable encoder experiments

Prepare complete instructions, source-bound alternatives and an independent action
evaluator before training. `decision_example()` preserves the original prompt,
task schema and episode lineage. Its caller must verify the supplied positive and
negative actions with the recorded oracle. Oracle feedback is distinct from
speaker confirmation.

## Reuse the encoder computation

For an authorized local encoder experiment, prepare the configuration, materialize
raw carriers once, then bind the resulting manifest into each head experiment:

```python
from pathlib import Path

from carl_studio.training.encoder import cache_embeddings
from carl_studio.training.preparation import prepare_training
from carl_studio.training.pipeline import submit_training

prepared = prepare_training(config, project_root=evaluator_root, manager=manager)
assert prepared.ready
cache = cache_embeddings(config, prepared, Path("carriers.json").resolve())
settings = config.encoder.model_copy(update={
    "embedding_cache": cache["manifest"],
    "head_layout": "per_rung",
    "relation_weight": 0,
})
candidate = config.model_copy(update={"encoder": settings})
plan = prepare_training(candidate, project_root=evaluator_root, manager=manager)
assert plan.ready
result = await submit_training(candidate, prepared_plan_id=plan.plan_id, manager=manager)
```

The manifest binds model, processor, interpreter, dependency, precision and
processing-recipe identities. Preparation binds its bytes and all carrier
artifacts. Carriers use the existing semantic artifact directory and vault
references. Event occurrences remain distinct while identical content shares
computation. A successor population uses a new manifest.

The isolated worker bounds OpenMP, OpenBLAS and MKL pools to four threads before
numerical imports. Its execution identity includes this policy alongside the
Torch thread limit; controlling Torch alone does not bound native BLAS pools.

The cached path supports text and structured inputs. Same-recipe batches preserve
part order and check each actual token count before inference. Cached head fitting
does not load the encoder. Adapter fitting retains its differentiable forward path
and refuses frozen training carriers. Its separate `baseline_cache` can reuse the
unadapted reference after a fresh numerical correspondence check. Candidate
evaluation uses current adapter weights and batches text by recipe. The selected
validation measurement is retained with its checkpoint. Media uses singleton
processing. Preparation also hashes the installed PEFT implementation.
Adapters enable nonreentrant activation checkpointing by default to bound memory;
its recomputation cost remains part of the execution deadline.

## Select with validation

`balanced_sampling` defaults to true and cycles through shuffled task families.
`head_layout` retains the legacy `shared` default; `per_rung` trains independent
linear rankers for 128, 256, 512 and 768 dimensions. Serving restores the same
selected heads. Relation loss has its own weight and gradient clipping.

Validation runs every 16 steps and at the final step. Checkpoint selection prefers
required-rung retention, then 768-dimensional accuracy, then positive-negative
margin. Three unsuccessful checks stop training. The checkpoint retains both the
final optimizer iterate and the validation-selected parameters; serving loads the
selected parameters. Stopped execution resumes the matching optimizer iterate,
sampler and RNG state.

Measurements retain validation history, losses, gradient norms, task-family
counts, encoding/cache counters and phase durations. Keep development diagnostics
separate from the unopened final test and select configurations only on validation.

## Recover completed work

Optimizer completion, evaluation and activation have separate recorded states.
Repeating submission after an evaluator failure retries evaluation against the
same completed checkpoint and measurement hashes. Concurrent retries serialize
at the existing experiment owner. Changed sources or artifacts refuse recovery.
Missing measurements remain inconclusive; accepted activation retains rollback.

Resource admission applies before any model qualification or cache computation.
Use the configured compute owner to cover all child processes, fresh aggregate
headroom, deadlines and settlement. The package does not embed a workstation's
private resource policy.
