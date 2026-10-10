---
last_updated: 2026-10-09
author: null
applies_to: carl-studio prepared training
---

# Training procedure qualification

Use CARL's existing preparation, experiment, trainer, evaluation and compute
owners. This procedure specifies their required observations; it does not claim
that an installed backend implements a control merely because it is described
here. Resolve a missing control at its owner before relying on it.

## Bound inputs and finite task domain

Freeze model revision and shards, tokenizer/processor, interpreter, dependencies,
configuration, reward/evaluator source closure, data and held-out membership.
Record path, byte count and SHA-256 from the bytes actually consumed. Include
uncommitted working-tree inputs and record HEAD and dirty ownership separately.
Reopen that inventory before submission, after execution and when resuming.
Input drift creates a successor preparation; preserve the previous execution.

Register task families, structural split rules, budgets, grading commands and
acceptance thresholds before measurements. IDs or split prefixes do not prove
family separation. New IDs for repeated prompts do not expand the proof domain.
Check graders against positive cases and deliberately wrong answers, stale
source/revision bindings, absent observations and malformed batch populations.
Retain source references and explicit residuals. Keep hidden reasoning,
credentials and unapproved conversations out of training data and receipts.

## Four evidence states

- Prepared: inputs, callables, held-out membership, policies and finite limits
  pass the preparation owner's checks. This does not establish GPU execution.
- Device-qualified: a bounded run exercises the exact model/backend, precision,
  dependencies, generation, task rewards, backward/optimizer step and checkpoint
  reload on the requested devices. Distributed qualification executes every rank
  and the real collective. A smaller model or rank count has its own domain.
- Executed: the worker has a terminal result, source-bound progress, observed
  optimizer work and retrievable model/adapter tensor artifacts. Reload the
  saved artifacts. Record finite gradient norms and compare trainable tensors
  with their recorded initial state in the same representation. Report a
  zero-update execution explicitly.
- Accepted: the configured held-out evaluator measures the candidate against
  the baseline and applies the registered objective and policy checks. A
  completed optimizer run or changed weights alone do not establish improvement.

The LoRA trainer callback records trainable adapter tensor identities and byte
hashes at train begin, after resume loading, and at train end. Its current domain
is a single device or ordinary DDP with unsharded LoRA parameters. Zero changed
tensors is an observation. Keep it separate from checkpoint reload and acceptance.
Bind callback source bytes through preparation. During reload, declare dtype
conversion and verify the appropriate round trip before comparing byte hashes.

Set thread-local CUDA device state inside the thread/process performing GPU
work. A probe on the event-loop thread cannot qualify work dispatched to an
executor. Bind local rank, observed device and collective world size there.
Exercise this executor during qualification with a finite timeout and a failing
device/rank control. A package import or available device inventory is insufficient.

Provider allocation RUNNING/EXITED status is separate from worker execution.
Synthetic losses, formula-derived accuracy, JSON summaries and a directory named
checkpoint do not substitute for optimizer/tensor evidence. Keep simulations
labelled simulated. Retain qualified evidence and approved artifacts before
allocation teardown; retain failure diagnostics without fabricating results.

## Authorization, supervision and settlement

Bind the existing grant to actor, operation, targets, input digest, allowed spend
and validity window. Reuse it for actions within those bounds. An additional
spend increment, wider destination or changed effect scope requires a matching
grant. Reconcile outstanding effects before retry or successor allocation.

Before provider submission, persist a stable run/request identity, current
price, remaining exposure, stop deadline and supervision state through the
existing execution owner. Include setup, downloads, idle allocation,
qualification, training, evaluation, storage and shutdown. Deduct settled and
outstanding exposure before deriving available runtime from the current quote.
Do not erase exposure by changing campaign or run IDs.
Bind the returned allocation identity and stop target before worker dispatch.

Supervision must survive client disconnect and worker failure. Bind its stop
action to the exact owned allocation and account for polling and shutdown delay
when deriving the deadline. Verify disconnect, deadline and stale-allocation
controls with the owner's test double before a billable effect. Verify persistent
supervision before allocation, then observe its host identity before dispatch.
A detached client, prompt policy, sleep or configuration timeout does not
establish this control.

Record ambiguous submission as unresolved and reconcile the same identity.
Cancellation request, acknowledged stop and terminal settlement are separate.
Keep supervision and custody until provider state and billed usage close.

## Owned process inspection

Use the PID recorded by the launcher or service owner. Bind PID, process start
time, parent/process group and executable identity; recheck them before a signal.
Inspect content-free status and resource observations. Avoid full command-line
or environment dumps because they can contain credentials or task payloads.
Broad process-name matching can select the inspecting shell or another lane.
Signal only the revalidated owned PID or group and observe its terminal.

## Existing source owners

- `src/carl_studio/training/preparation.py`: source closure, task preparation,
  callable resolution and stale-input validation.
- `src/carl_studio/training/pipeline.py`: prepared submission, baseline/candidate
  evaluation and execution custody.
- `src/carl_studio/training/trainer.py`: model work, optimizer and checkpointing.
- `src/carl_studio/training/callbacks.py`: trainable LoRA parameter-update witnesses.
- `src/carl_studio/eval/runner.py`: held-out execution and evaluator binding.
- `src/carl_studio/experiment/manager.py`: durable preparation and run ownership.
- `src/carl_studio/compute/`: selected backend provisioning, execution and stop.

Extend these owners and their bounded adapters. Preserve one writer per path
and separately bind source, installed plugin, host projection and loaded runtime.
