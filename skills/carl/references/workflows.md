# CARL workflows

Use `carl --help` and the selected subcommand's help for the installed release.

```bash
carl init
carl observe --help
carl train --help
carl eval --help
carl lab skill list
carl lab agent harnesses
```

The lightweight base package can measure coherence without training dependencies.
Training and optional backends require their documented extras and supported
hardware. Check availability before launching a job.

Use the actual objective and held-out eval to assess training. Coherence alone
does not establish task success. Preserve inputs, configuration, result artifacts,
and the named evaluation commands requested by the user.

CARL's delegation task IDs remain stable across its MCP and A2A observation
surfaces. Resume input only through the original live execution owner.

Prepare local training without starting a job:

```bash
carl train --prepare --config path/to/training.yaml --json
```

Relative datasets resolve beside the configuration first, then at the discovered
project root. Inspect `ready` and the reported issues before executing the plan.
The first checkpoint binding reads every shard. Later preparation reuses cached
byte digests while file identity, size, mtime, and ctime match; execution still
rechecks source bytes. A checkpoint index alone does not bind shard contents.

## Execution qualification

Preparation, device qualification, training execution and candidate acceptance
have separate evidence. A prepared plan binds inputs and checks readiness; a
bounded qualification executes the exact model, backend, dependency lock,
precision, requested GPU count and training path. Exercise generation, aligned
task rewards, backward/optimizer work and checkpoint save/reload. Multiple GPUs
require a collective across every requested rank. Set thread-local CUDA device
state inside the worker thread or process that executes those operations.

Provider RUNNING/EXITED describes allocation state. Completion requires the
worker's terminal result, observed optimizer progress, retrievable model/adapter
tensors and a verified reload. Formula-derived loss or accuracy, a metrics JSON
file and a checkpoint directory name do not establish training. Acceptance uses
the configured held-out objective and baseline comparison in the finite task
domain. Nonzero gradients or coherence alone do not establish improvement.
Record finite gradient norms and comparable initial/final trainable-tensor
digests; report zero-update executions explicitly.

Before a billable effect, persist the grant, quote, remaining exposure, stable
run/request identity, price-derived stop deadline and supervision state
through the existing execution owner. Include setup, downloads, idle time,
qualification, training, evaluation, storage and shutdown exposure. Verify the
disconnect, stale-identity and deadline paths before relying on that control.
Bind the returned provider allocation and stop target before worker dispatch.
Keep supervision active until provider stop and terminal settlement are observed.
Reconcile an ambiguous submission before retrying it. Reuse authorization inside
the same bound grant; obtain a new grant when a spend increment or scope exceeds it.

For resume, reopen the recorded inventory of configuration, locks, imported
reward/evaluator modules, data, model shards, tokenizer and processor. Bind each
path, byte count and SHA-256 from one frozen snapshot, including working-tree
changes when consumed. Record HEAD and dirty ownership separately. Changed
inputs require a newly prepared identity; retain successful observations and
their earlier artifacts. A dirty count or commit hash cannot replace exact bytes.

Inspect task-owned processes through the launcher/service owner's recorded PID,
start time, process group and executable identity. Recheck identity before a
signal to reject PID reuse. Read content-free metadata; avoid full command-line
or environment dumps and broad process-name matching. These procedures extend
the existing preparation, trainer, evaluation, experiment and compute owners.
The CARL source checkout documents finite checks in
`docs/training-procedure-qualification.md`.

TCG reward datasets can supply `private_keys` as a list of private draft identifiers
per sample and `trajectory` as a list of events with `phase: private` or
`phase: public`. CARL checks both the recorded public events and the completion.
A leak zeroes task and coherence contributions and adds an unweighted terminal
reward of `-3.0`, outside cascade masks. Evaluation through `EvalConfig.private_keys`
checks public events in `EvalReport.detail` and refuses a leaking report.
Keep private draft identifiers and contents out of public logs and result artifacts.


## Maxwell cycle ownership

For a stateful multi-stage improvement, maxwell-trainer:max-training-cycle owns
composition and provider reconciliation through the fixed CARL adapter. CARL's
ExperimentManager and MCPTaskStore remain the execution and task authorities.
Prepared identity must survive retry, worker restart and client disconnect.
Transport completion, optimizer completion, goal acceptance, artifact custody
and provider settlement are independent fields. The user goal remains active
when a candidate needs repair; a rejected checkpoint is retained for diagnosis.


The concrete cycle factory is `maxwell_trainer.carl_cycle.runpod_cycle`.
`MCPTaskStatePort` uses CARL's existing transactional operation record. The fixed
Runpod adapter and `RemoteCARLPort` retain provider identity and public CARL reads.
For an SFT ancestor followed by GRPO, select `comparison_baseline: base` when
comparing final progress against the original base; keep the SFT ancestor bound
for candidate loading. The default `starting` comparison remains unchanged.
