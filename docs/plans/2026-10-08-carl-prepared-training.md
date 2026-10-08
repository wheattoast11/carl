---
last_updated: 2026-10-08
author: Codex
applies_to: carl-studio 0.21.0
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
---

# CARL prepared training

## Goal

Turn a goal into a prepared experiment, a bounded training run through the
existing pipeline, a baseline comparison, and a reusable accepted artifact.

The approved choices are model or adapter improvement, preservation of the
caller pipeline, held-out goal progress plus policy checks and a coherence
floor, and beginner automation in the existing agent. Tinny alone and Tinny
with TCG are the two reference cases. Actual Ansible provisioning, a new
consumer portal, and marketplace expansion are outside this change.

## Construction contract

Baseline: `5be38462a9c3dba089e24f1386a945dc2469b99c`.
Branch: `codex/carl-prepared-training`.
Pre-existing CARL dirty files: none.
Writer: parent agent; research and review agents are read-only.

Owners: project discovery and environment setup; normalized data and kits;
ExperimentManager; training adapters, CARLTrainer and SendItPipeline;
EvalRunner and EvalGate; existing CLI, MCP task and plugin projection owners.

Inputs: configuration, explicit goal and policy checks, selected backend,
source model and data identities, held-out split, reward/evaluator bindings,
resource limits, and artifact destination.
Outputs: typed preparation and readiness issues, an execution handle,
baseline/candidate measurements, acceptance reasons, and artifact references.

Preparation must not submit training, install dependencies, contact a provider,
or publish. Missing data or graders yield an actionable readiness issue.
Training consumes matching prepared inputs. Execution completion is separate
from acceptance. Required missing measurements cannot satisfy acceptance.
Failed gates prevent promotion. Local artifacts and explicitly approved
private remote custody precede separately authorized public publication.

Keep the existing experiment/run persistence and native permissions. Preserve
configuration-only calls. Cancellation acknowledges actual stopped execution.
Keep full-logit coherence and partial-logprob proxies distinct. Operational
traces contain metadata and references rather than training content.

## Implementation

1. Correct project display/routing, stage-specific evaluation, artifact binding,
   SFT-to-GRPO continuation, final gates and publication handling.
2. Add typed preparation and goal bindings to existing config/experiment/data
   owners. Validate datasets, splits, graders, reward hooks and bounded work.
3. Attach explicit task rewards and evaluators to the current training loop;
   use TRL as the first reference integration and declare adapter capabilities.
4. Share preparation and submission between MCP and CLI. Extend the canonical
   skill and generate host projections. Show beginner summaries with details
   available; recover from recorded state and reuse prior authorization.
5. Retain comparisons and artifacts, test positive and negative cases, review
   independently, build, and exercise native hosts with synthetic providers.

## Verification

Preparation must accept usable local inputs and reject absent data, overlapping
splits, absent graders, unsupported hooks, unbounded work and stale inputs
before trainer construction. A distinct task reward must reach GRPO training.
High coherence cannot rescue a wrong task result. An unchanged candidate cannot
satisfy positive improvement. Compare identical held-out cases and generation
budgets. Verify cancellation, retry deduplication, ownership, integrity, and
publication boundaries. Preserve legacy interfaces and import-light behavior.

Initial offline checks: 76 passed for pipeline/eval/comparison/delegation/plugin;
95 passed and one pre-existing entry test failure caused by pytest arguments
entering the CLI router; the isolated normalized-argument case passed. Four
release artifact hashes matched. These checks did not execute model training.

Live provider, paid training, GPU workload, dependency installation, publication, and
service changes require their concrete approved inputs. No current provider
spending authorization or training budget was supplied.

## Sanitized case review

Source: `tin/carl/codex-session-01a11d26-09d1-7762-bd63-cf788c6c4d22.md`.
Source SHA-256: `9defa22397d360565470d560f04f5bfa35e28504014d8dfc4eb507d32459d2fc`.
Observed size: 9,202 lines, 101 activity blocks, four user sections and four
assistant sections. Two requested modes produced three configurations.
The visible ending left dataset preparation and submission outstanding.

The case exposed wrong interpreter/path assumptions, project-context friction,
separate reward declarations and pipeline construction, schema-only readiness,
and simulated reward shortcuts. Current pure counterexamples gave a long wrong
answer 1.0 without assertions and 0.0 with a failing assertion; a legal wrong
TCG move received 1.5 without a grader and -1.0 with an external rejection.
No raw conversation, credentials, internal reasoning, or training content is
retained in this note. The user requested deletion after review and scheduled
credential rotation independently. The exact file was deleted after saving
this review and rechecking the source hash. Its absence was verified.

## Implementation verification

Prepared training is implemented through the named owners.
The final offline suite passed: 4,378 passed, 29 skipped.
All eight independently reviewed defects were repaired and checked.
The bounded real TRL configuration constructors, offline build, installed-wheel
imports, generated projections and three native preparation witnesses passed.
Model training and public publication were not executed.
Commands, falsifiers and evidence boundaries are recorded in
[the verification record](../carl-prepared-training-verification.md).

## Rollback

Revert the explicit feature-owned diff on this branch. Preserve peer-owned Tin
files. Plugin rollback uses its existing owned-entry update/uninstall paths;
experiment artifacts and trained candidates remain owner-readable records.
