---
name: carl
description: Use CARL to observe coherence, validate training configuration, train and evaluate models, run CARL skills, or delegate bounded work to Codex, Claude Code, and OpenCode.
---

# CARL

Use the installed CARL MCP tools when available. Use the CARL CLI otherwise.
CARL means Coherence-Aware Reinforcement Learning.

1. Inspect the current project and the requested outcome.
2. For a goal, use `prepare_training` before starting training. Schema validation
   alone does not establish data, graders, reward hooks or resource readiness.
3. Present the prepared inputs, task checks, policies, model, limits and effects.
   Pass the returned `plan_id` as `prepared_plan_id` to `start_training` or
   `submit_async_training` only after the user has authorized that run.
4. Keep prepared, device-qualified, executed and accepted states separate.
   Return the checkpoint, baseline comparison, acceptance reasons and reuse instructions.
5. Use `list_skills` and `run_skill` for existing CARL workflows.
6. Use `list_agent_harnesses` and `delegate_agent` when another native agent should do part of the work.

## Goal preparation

Discover the project root, selected interpreter, existing pipeline and relevant
task fixtures. Preserve the model, backend and pipeline choices. Ask only for
information the project cannot supply, one question at a time. Offer the two
requested Tinny modes; keep universal-role ablation in advanced details.

Prepare compatible local data with stable task IDs and separate held-out tasks.
Use existing data adapters, verification records and kits. Reuse executable
project checks and source references. Never substitute unrelated datasets,
invent a grader, infer task success from answer length or move legality, or
describe a simulated environment as an independent verdict.

Resolve readiness issues before submitting. Missing model/data inputs need an
authorized preparation action. Missing dependencies need an explicit install
proposal. The first qualified goal integration uses local TRL; other backends
declare their capabilities. Python pipelines can attach explicit reward and
evaluator callables to the existing trainer and eval runner.

Translate relevant user policies into explicit checks or existing execution
controls. State which rules are measured and which are enforced during tool
execution. A description in a prompt is not an executed policy check.

Repeated prepared submission reuses its recorded result. Explicit resume uses
recorded checkpoints; unresolved execution custody needs reconciliation.
Retain metadata and approved artifacts. Do not use raw conversation exports,
credentials or hidden reasoning as automatic training data.

Local plugin access is FREE. Scheduled autonomy, paid services, publication, and
private-runtime operations retain their own authorization and entitlement checks.
Do not start paid training, transfer data, publish, or broaden a workspace grant
without the user's authorization.

Reuse authorization while actor, operation, targets, inputs, spend and validity
remain inside the same bound grant. Reconcile existing effects before a retry;
request further authorization when an increment or scope exceeds that grant.

Before billable allocation, bind price, remaining exposure, stop deadline and
persistent supervision to the existing execution owner. Cover setup, downloads,
idle allocation, qualification, evaluation and shutdown. Client disconnect or
process failure must leave the exact owned allocation supervised. A prompt,
declared timeout or client-side sleep does not enforce a spend cap.

Qualification executes the selected model and backend on the requested devices,
including generation, rewards, backward/optimizer work and checkpoint reload.
For multiple GPUs, exercise every rank and the actual collective. Initialize
thread-local CUDA device state in the executor that performs GPU work. Neither
prepared inputs nor a successful probe in another thread establishes that path.

Require the worker terminal, observed progress and retrievable tensor checkpoint
before reporting training completion. Provider status, JSON-only checkpoints and
synthetic learning curves do not establish it. Reopen frozen source bytes and
effect custody on resume. Inspect and stop only recorded owned process identities;
keep command-line credentials and environments out of inspection output.

## Delegation

Pass the chosen host, instruction, workspace, and a stable request ID.
Read-only is the default. Request editing only when the user authorized it.
Poll `tasks_get`; answer the exact pending request with `tasks_reply`.
Read its transient details with `read_agent_input` before answering.
Cancel with `tasks_cancel`. Cancellation succeeds after execution stops.
Retrieve the final artifact with `read_agent_result`.

CARL owns execution, scope, limits, and task state. Child agents cannot delegate
again through CARL. Do not change a child policy or bypass permission checks.
Keep instruction and tool content out of operational traces. Return the useful
answer and user-requested artifacts.

## CLI

```bash
carl plugin doctor
carl lab agent harnesses
carl lab agent delegate codex < instruction.txt
carl lab agent status TASK_ID
carl lab agent result TASK_ID
carl lab agent cancel TASK_ID
carl lab skill list
```

Install from a CARL checkout with `./scripts/install-plugin`.
Update with `carl plugin update`; remove only CARL's owned entries with
`carl plugin uninstall`.

See [workflows](references/workflows.md) for training and verification commands.


## Local encoders and interpretation

Use `encode_data` for ordered text, structured actions, image, audio and video
references. A Session shares the encoder across chat, MCP and data operations.
Local execution uses `CARL_ENCODER_MODEL` and `CARL_ENCODER_PYTHON`; main package
dependencies stay unchanged. Media requires hash-bound file artifacts.

Use `interpret` to record a candidate with separate utterance, context, goal and
proposal references. Use `interpretation_feedback` for explicit confirmation or
a successor correction. Silence leaves the candidate unconfirmed. Similarity
and action outcomes are separate measurements.

Local capture is opt-in through `carl lab semantic capture`. Captured accepted
records and their artifact descriptors consolidate once in the memory owner.
Capture does not authorize training or network transmission.

Use `method: encoder` with nested `encoder` settings for frozen heads or PEFT
query/value adapters. Prepare grouped train, validation and test episodes before
creating alternate views. Supply explicit positives, negatives and feedback
provenance. Worker interpreter, dependencies, processor and modules are bound.
Repeated prepared submissions return the recorded execution. Resume requires a
stopped checkpoint. Ranking improvement alone does not establish action success.
Missing action or policy measurements yield inconclusive acceptance.

`carl lab semantic invoke OPERATION REQUEST.json --session-id ID` uses the same
Session implementation and retains restart references through SessionStore.
