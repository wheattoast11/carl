---
title: "Interaction Chains as Witness Logs"
subtitle: "A Cross-Cutting Primitive for Reproducibility, Eval Gating, and Cross-Channel Telemetry"
authors:
  - name: Tej Desai
    affiliation: Intuition Labs LLC
    email: tej@terminals.tech
  - name: Claude Opus 4.6
    affiliation: Anthropic
date: 2026-04-20
keywords: [interaction-chain, witness-log, reproducibility, tool-calling, provenance, MCP, eval-gating]
license: CC-BY-4.0
series: "CARL Methods Series, paper 4 of 4"
see_also:
  - "paper/01-main-carl.md — main paper"
  - "paper/02-phase-adaptive-methods.md — phase-adaptive weighting"
  - "paper/03-coherence-trap-technical-note.md — mode-collapse technical note"
related_work:
  - "Desai, T. (2026). Bounded Informational Time Crystals. DOI: 10.5281/zenodo.18906944"
  - "Desai, T. (2026). Semantic Realizability: The Convergence-Realizability Identity. DOI: 10.5281/zenodo.18992031"
---

# Interaction Chains as Witness Logs

**Tej Desai** (Intuition Labs LLC) and **Claude Opus 4.6** (Anthropic)

April 2026 · CARL Methods Series, Paper 4 of 4

---

## Abstract

Tool-calling agents, training pipelines, eval runners, payment rails,
and MCP servers all produce provenance: a sequence of atomic events
(tool invocations, LLM replies, gate checks, payments, memory reads)
that together constitute the durable record of what the system did.
In CARL Studio we unify these sequences behind a single primitive:
`InteractionChain`, a typed append-only log of `Step` records with a
stable per-action taxonomy (`ActionType`). Because the primitive is
cross-cutting — the same `InteractionChain` object threads through
training (`training/pipeline.py`), eval (`eval/runner.py`), x402
payments (`x402.py`), the chat agent (`chat_agent.py`), and MCP tool
dispatch — we obtain reproducibility, eval gating, and cross-channel
telemetry from one schema. The v0.7.1 migration to per-request MCP
`Context` (replacing the prior module-global `_session`) illustrates
the payoff: session state is now per-chain rather than per-module, and
every tool invocation emits a step with stable redaction guarantees.
This paper specifies the primitive, enumerates its consumers in
shipped code, and documents the empirical evidence from v0.7.1.

---

## 1. Motivation

A tool-calling agent operating against an LLM and a set of side-effect
tools is, from a provenance perspective, a sequence of events:

```
user_input -> llm_reply(tool_call) -> tool_call -> tool_result
           -> llm_reply -> gate_check -> payment -> llm_reply ...
```

Standard practice in most agent frameworks leaves this sequence as an
implicit byproduct of print statements, logger records, and
in-memory structures owned by whichever orchestrator component is
active at the moment. When the agent also integrates with
training (where every generation is a data point) and eval (where
every generation is a gating signal), the absence of a shared
provenance substrate produces three problems:

1. **Reproducibility** — a training datum cannot be re-executed
   without the full tool-call context that produced it. Ad-hoc logs
   lose structure on reload.
2. **Eval gating** — a gating decision (phase advance, payment
   release, training-step checkpoint) is justified by a sequence of
   observations. Without a typed chain, justification is narrative.
3. **Cross-channel telemetry** — coherence signals sampled at
   training time (`Phi`), at eval time (`R`), at tool-dispatch time
   (tool-success rate), and at payment time (facilitator-response
   latency) live in separate systems. Correlation requires a
   universal key.

`InteractionChain` is CARL's answer: one primitive, one schema, one
redaction layer, consumed by every side-effect-producing subsystem.

---

## 2. The Primitive

**File**: `packages/carl-core/src/carl_core/interaction.py`

**Core types**: `ActionType`, `Step`, `InteractionChain`.

### 2.1 `ActionType`

An enum of interaction shapes. Shipped values
(`interaction.py:43-61`):

```python
class ActionType(str, Enum):
    USER_INPUT       = "user_input"       # text typed by user
    TOOL_CALL        = "tool_call"        # agent invokes a registered tool
    LLM_REPLY        = "llm_reply"        # model streamed a reply
    CLI_CMD          = "cli_cmd"          # top-level CLI command dispatch
    GATE             = "gate"             # credential / permission prompt
    GATE_CHECK       = "gate_check"       # structured allow/deny predicate check
    EXTERNAL         = "external"         # HTTP / file / subprocess
    PAYMENT          = "payment"          # x402 / wallet payment flow
    TRAINING_STEP    = "training_step"    # periodic training progress marker
    EVAL_PHASE       = "eval_phase"       # eval phase start/end boundary
    REWARD           = "reward"           # reward-aggregation snapshot
    CHECKPOINT       = "checkpoint"       # trainer checkpoint / model save
    MEMORY_READ      = "memory_read"      # memory recall
    MEMORY_WRITE     = "memory_write"     # memory commit
    HEARTBEAT_CYCLE  = "heartbeat_cycle"  # sticky-note cycle boundary
    STICKY_NOTE      = "sticky_note"      # note append / dequeue / status transition
```

The taxonomy is closed: adding a new shape is an API event that
touches every consumer. This is a feature — shape drift would
invalidate cross-subsystem telemetry.

### 2.2 `Step`

One atomic unit (`interaction.py:72-149`):

```python
@dataclass
class Step:
    action: ActionType
    name: str                         # short label, e.g. "carl chat"
    input: Any = None                 # prompt, args, credential *name*
    output: Any = None                # response, result, gate verdict
    success: bool = True
    started_at: datetime = field(default_factory=_utcnow)
    duration_ms: float | None = None
    parent_id: str | None = None
    step_id: str = field(default_factory=_new_id)   # 12-hex chars
    session_id: str | None = None
    trace_id: str | None = None
    phi: float | None = None                         # coherence snapshot
    kuramoto_r: float | None = None                  # phase-lock snapshot
    channel_coherence: dict[str, float] | None = None
```

The `phi`, `kuramoto_r`, and `channel_coherence` fields are the
coherence-side hooks: when populated by training, eval, the chat
agent, or a `BaseConnection`, they turn the chain into a
cross-channel coherence witness. `InteractionChain.coherence_trajectory()`
reduces the chain to a `(timestamp, phi)` series usable across any
subsystem that records `phi` on a step.

### 2.3 `InteractionChain`

An ordered log with context dict (`interaction.py:152-290`):

```python
@dataclass
class InteractionChain:
    chain_id: str = field(default_factory=_new_id)
    started_at: datetime = field(default_factory=_utcnow)
    steps: list[Step] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def record(self, action, name, *, input=None, output=None, ...) -> Step: ...
    def append(self, step, result=None) -> Step: ...
    def by_action(self, action) -> list[Step]: ...
    def success_rate(self) -> float: ...
    def coherence_trajectory(self) -> list[tuple[str, float | None]]: ...

    def to_dict(self) -> dict[str, Any]: ...
    def to_jsonl(self) -> str: ...
    @classmethod
    def from_dict(cls, d) -> InteractionChain: ...
```

Append-only by contract. Once a step lands it is not mutated. The
chain is the durable artifact; `to_jsonl` produces newline-delimited
JSON suitable for persistence at
`~/.carl/interactions/<chain_id>.jsonl`.

### 2.4 Redaction

All serialization goes through `_json_safe`
(`interaction.py:360-402`). Two redaction layers run on every
`to_dict` / `to_jsonl` call:

- **Name-based**: dict keys matching `*key*`, `*token*`, `*secret*`,
  `*password*`, `*authorization*`, or `*bearer*` have their values
  replaced with `"<redacted>"`.
- **Shape-based**: string values matching JWT / `sk-...` /
  `sk-ant-...` / `hf_...` / EVM-address regexes are replaced with a
  six-character prefix plus `"<redacted>"`, preserving the shape for
  debugging without leaking the credential.

These redactions apply unconditionally at serialization time.
In-memory `Step` objects retain raw values for programmatic use
within the same process. The net effect is that durable witness logs
are safe to feed back into training, share for reproducibility, or
attach to audit trails.

### 2.5 Why carl-core owns it

`InteractionChain` lives in `carl-core` (see
`packages/carl-core/CLAUDE.md`) rather than `carl_studio` because it
is a dependency-free lingua franca across every CARL package. Its
only dependency is the standard library. Every downstream consumer
(`carl_studio`, `terminals-runtime`, and any future external
extenders) imports from `carl_core.interaction` directly.

---

## 3. Cross-Cutting Adoption

The primitive's value comes from being consumed by every
side-effect-producing subsystem with one schema. Each adoption below
is verified in the v0.7.1 codebase.

### 3.1 Chat agent — `chat_agent.py`

**File**: `src/carl_studio/chat_agent.py`

The agentic loop owns an `InteractionChain` per session. Steps are
recorded for CLI command dispatch, LLM replies, tool calls, memory
reads, memory writes, and gate checks. The `_chain` attribute is
lazy-initialized (`chat_agent.py:850-856`) and the session persists
at `~/.carl/sessions/<id>.json` with `schema_version=1` (see the
project `CLAUDE.md`).

What the chain records and why:

| Event | `ActionType` | Justification |
|-------|--------------|---------------|
| User prompt arrival | `USER_INPUT` | Replay requires the raw prompt. |
| Streamed LLM reply | `LLM_REPLY` | Tool-call attempts live in the reply content; the reply is data. |
| Tool invocation | `TOOL_CALL` | Every side-effect tool call is recorded with args and result. |
| Memory recall | `MEMORY_READ` | Eval gating depends on knowing what was retrieved. |
| Memory write | `MEMORY_WRITE` | Training signal depends on knowing what was committed. |
| Permission gate | `GATE` / `GATE_CHECK` | Gate decisions are recorded as typed events, not free-text logs. |

### 3.2 Training pipeline — `training/pipeline.py`, `training/trainer.py`

**Files**: `src/carl_studio/training/pipeline.py`,
`src/carl_studio/training/trainer.py`,
`src/carl_studio/training/callbacks.py`,
`src/carl_studio/training/lr_resonance.py`

The training pipeline records `TRAINING_STEP` events for progress
markers, `REWARD` events for reward-aggregation snapshots, and
`CHECKPOINT` events for trainer saves. The `phi` and `kuramoto_r`
fields on `Step` are populated from the reward function's cached
`CoherenceTrace` batch, so the chain becomes the canonical source of
per-step coherence dynamics. Downstream, `carl run diff <a> <b>`
(shipped in v0.7.1) consumes these phi / q_hat / crystallization
fields for trajectory delta.

### 3.3 Eval runner — `eval/runner.py`

**File**: `src/carl_studio/eval/runner.py`

The eval runner imports `ActionType` and `InteractionChain` directly
(`eval/runner.py:28`) and threads an optional chain through the
runner constructor (`eval/runner.py:806-810`). The most important
event type is `EVAL_PHASE` — phase boundaries are recorded as
first-class steps (`eval/runner.py:828-829`). A gated eval (e.g.
Phase-2' pass/fail) reduces to a predicate on the chain's
`by_action(ActionType.EVAL_PHASE)` result plus per-step success
flags.

### 3.4 x402 payment client — `x402.py`

**File**: `src/carl_studio/x402.py`

The x402 HTTP payment rail client optionally accepts a chain
(`x402.py:356`) and records `PAYMENT` events (`x402.py:382-397`).
Every `check_x402`, `negotiate`, and `execute` call — even failing
ones — lands a typed step. Redaction of facilitator authorization
headers happens at serialization time via `_json_safe`'s shape-based
pattern list, so a chain containing x402 interactions remains safe to
persist.

### 3.5 MCP server and connection — `mcp/server.py`, `mcp/connection.py`

**Files**: `src/carl_studio/mcp/server.py`,
`src/carl_studio/mcp/connection.py`,
`src/carl_studio/mcp/sampling.py`,
`src/carl_studio/mcp/elicitation.py`

FastMCP tool handlers dispatch through a per-request `Context`
(shipped in v0.7.1). Each `MCPServerConnection` owns session state,
and tool invocations record `TOOL_CALL` steps onto the chain the
connection carries. Sampling (`mcp/sampling.py`) and elicitation
(`mcp/elicitation.py`) flows — the bidirectional server-to-client
protocols — record as `EXTERNAL` steps. The cross-cutting effect is
that an MCP client invoking a CARL tool receives a provenance trail
identical in shape to what a local CLI user would see.

### 3.6 Gating primitives — `gating.py`, `consent.py`, `tier.py`

**Files**: `src/carl_studio/gating.py`, `src/carl_studio/consent.py`,
`src/carl_studio/tier.py`

Consent and tier gate checks emit `GATE_CHECK` events via the shared
`emit_gate_event` helper in `gating.py`. The `GATE_CHECK` action
type is distinct from the older `GATE` action type: `GATE_CHECK` is a
structured allow/deny predicate check (included in the enum at
`interaction.py:51`), whereas `GATE` is a credential / permission
prompt (`interaction.py:50`). The two action types preserve the
semantic distinction between *asking a user for a credential* and
*evaluating a predicate that already has its inputs*.

### 3.7 Heartbeat loop — `heartbeat/loop.py`

**File**: `src/carl_studio/heartbeat/loop.py`

The background sticky-note worker records `HEARTBEAT_CYCLE`
boundaries and `STICKY_NOTE` events for note append / dequeue /
status transitions (`interaction.py:60-61`). The chain is the
authoritative timeline of what the heartbeat did between user
sessions, used by the sticky-note eval path.

### 3.8 Feedback and push — `feedback.py`, `cli/push.py`

**Files**: `src/carl_studio/feedback.py`,
`src/carl_studio/cli/push.py`

Both subsystems record `CLI_CMD` and `EXTERNAL` steps so that the
`carl push` flow (shipped in v0.7.0) and the feedback channel
produce chain-persistent provenance on par with training and chat.

---

## 4. v0.7.1 Evidence

The v0.7.1 release migrated MCP session handling from a module-global
`_session` variable to per-request `Context` injection (see the
project-root `CLAUDE.md` changelog entry and `mcp/server.py:54-86`).
Prior to the migration, every MCP tool call mutated module state;
concurrent requests or test fixtures could leak state across
invocations. After the migration, each `MCPServerConnection` carries
its own session, each tool invocation receives a FastMCP `Context`,
and the `InteractionChain` binding follows the connection rather than
the module.

This is the primitive paying dividends. The chain was already
session-scoped — it had always threaded through tool invocations
via a per-session object. The v0.7.1 migration updated the
surrounding session state to match the chain's shape. We did not
redesign the telemetry substrate to support multi-tenant MCP; we
discovered that the substrate already supported it, and the rest of
the stack had to catch up. The diff is in `mcp/server.py` and
`mcp/connection.py`; the chain code in
`packages/carl-core/src/carl_core/interaction.py` was not modified.

Two consequences:

1. **Multi-tenant MCP readiness** (see `docs/mcp_multitenant.md`) is
   now structural. Sessions cannot leak because there is no shared
   mutable global for them to leak through.
2. **Reproducibility improved**: an MCP-issued training request
   produces exactly the same chain shape as a locally-invoked training
   request. The provenance substrate is invariant under the
   transport.

---

## 5. Future Work

### 5.1 Hash-chain witnessing

`src/carl_studio/contract.py` already implements SHA-256 hash-chained
service contract witnessing (see project `CLAUDE.md`). A natural
extension is to anchor every `Step` to a rolling hash of its
predecessor, producing a tamper-evident chain without additional
infrastructure. The serialization format (`to_jsonl`) is already
append-only and ordered; adding a per-row hash field is compatible
with the existing schema. We have not shipped this.

### 5.2 Cross-process chain composition

Today an `InteractionChain` lives in one process. Training runs that
span multiple compute backends (SSH + local eval, HF Jobs + local
observe) produce one chain per process. A cross-process composition
operation — either a merge on identical `trace_id` values or a parent
/child nesting via `parent_id` — would let a single conceptual run be
witnessed end-to-end. The `trace_id` and `parent_id` fields on
`Step` are in place precisely for this extension; the composition
tooling is not.

### 5.3 External-consumer publication

The primitive is MIT-licensed and lives in `carl-core`. External
tool-calling frameworks that want CARL's provenance guarantees can
depend on `carl-core` without pulling in training or eval surfaces.
No public API change is needed; the shape is already minimal. What is
missing is documentation aimed at that external audience —
the current README documents `carl-core` as a dependency for the rest
of the CARL stack, not as a standalone provenance primitive.

---

## References

1. Desai, T. and Claude Opus 4.6. (2026). *Coherence-Aware
   Reinforcement Learning.* CARL Methods Series, Paper 1 of 4.
   `paper/01-main-carl.md`.

2. Desai, T. and Claude Opus 4.6. (2026). *Phase-Adaptive Coherence
   Rewards.* CARL Methods Series, Paper 2 of 4.
   `paper/02-phase-adaptive-methods.md`.

3. Desai, T. and Claude Opus 4.6. (2026). *The Coherence Trap.*
   CARL Methods Series, Paper 3 of 4.
   `paper/03-coherence-trap-technical-note.md`.

4. Desai, T. (2026). *Bounded Informational Time Crystals.* Zenodo.
   DOI: 10.5281/zenodo.18906944

5. Desai, T. (2026). *Semantic Realizability: The
   Convergence-Realizability Identity.* Zenodo.
   DOI: 10.5281/zenodo.18992031

---

## Appendix. File cross-reference

| Concept | File | Symbol |
|---|---|---|
| `InteractionChain` / `Step` / `ActionType` | `packages/carl-core/src/carl_core/interaction.py` | `InteractionChain`, `Step`, `ActionType` |
| Redaction helpers | `packages/carl-core/src/carl_core/interaction.py` | `_json_safe`, `_scrub_secrets` |
| Chat agent chain | `src/carl_studio/chat_agent.py` | `self._chain`, `get_chain()` |
| Eval runner chain | `src/carl_studio/eval/runner.py` | runner `interaction_chain` parameter |
| x402 chain | `src/carl_studio/x402.py` | `X402Client(chain=...)` |
| MCP per-request context | `src/carl_studio/mcp/server.py` | `bind_connection`, `_get_session` |
| MCP connection | `src/carl_studio/mcp/connection.py` | `MCPServerConnection` |
| Hash-chain witness | `src/carl_studio/contract.py` | — |
| Durable persistence | `~/.carl/interactions/<chain_id>.jsonl` | — |

---

**Code**: github.com/terminals-tech/carl-studio (tag `v0.7.1`)

**Intuition Labs LLC** — terminals.tech
