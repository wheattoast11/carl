---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10
status: master-plan
supersedes: docs/v10_terminals_tech_deep_dive.md (partially)
peer_reviewed_by:
  - peer_review_math.json (2026-04-20)
  - peer_review_architecture.json (2026-04-20)
  - peer_review_plan.json (2026-04-20)
  - peer_review_ip.json (2026-04-20)
---

# v0.10 Master Plan — Consilience-Validated, Peer-Reviewed

<introduction>
This document is the consolidated output of a four-wave ultrathink process:
(Wave 1) grounding via plan-file reading; (Wave 2) terminals.tech package
surface review; (Wave 3) lib/terminals-tech + lib/webcontainer deep-dive;
(Wave 4) vanilla-context peer-review agents across math / architecture /
plan / IP. This doc is phase-locked to 2026-04-20 and is the SOP for
v0.10 implementation. Future sessions should treat it as authoritative
until superseded with an updated `applies_to` header.
</introduction>

<resonance_confirmation date="2026-04-20" status="resonant">
Before writing this plan, the dispatcher confirmed resonance with: the
IRE paradigm (M, I, Φ, G), the κ = 64/3 ruling, the MIT ↔ BUSL seam,
the 67 AXON SignalTypes, the webcontainer-per-subagent-in-carl-app
boundary, the fractal self-similar pattern at all scales, and the
anchor function of CLAUDE.md. This plan operates within that frame.
</resonance_confirmation>

## Part 1 — Peer-review consilience (Bend-style MECE coalesce)

Four vanilla-context peer-review agents ran in parallel with strict JSON
output schema. The DAG-level aggregate below is the MECE-deduplicated
finding set, with each item traced to its source agent(s) and challenge
ID.

<findings_dag>

### P0 — Critical, actionable this session or next

<finding id="P0-1" severity="critical" category="architecture">
<claim>chat_agent.py tool-use loop executes tools, invokes pre/post hooks, collects results — but NEVER records `ActionType.TOOL_CALL` steps to the InteractionChain.</claim>
<source>peer_review_architecture.json F4/G1, lines 1286-1346 of chat_agent.py</source>
<impact>Breaks the "complete witness log" claim in the IRE paper. CLI and memory operations ARE logged; tool calls are NOT. This invalidates session replay and training-signal extraction through the tool path.</impact>
<remedy>Add `chain.record(ActionType.TOOL_CALL, name=block.name, input=tool_input, output=result, success=(is_error==False))` inside the tool-execution loop. ~10 LOC.</remedy>
<priority>P0 — fix in v0.9.0-alpha before any new feature work</priority>
</finding>

<finding id="P0-2" severity="critical" category="plan-dependency">
<claim>The carl.camp backend contract for `POST /api/sync/agent-cards` is undefined. v0.10-A #1 (agent-cards + Supabase) cannot ship without it.</claim>
<source>peer_review_plan.json R1, C2</source>
<impact>Blocks the highest-leverage v0.10 user-visible feature.</impact>
<remedy>Surface to Tej: publish the backend spec (endpoint URL, auth header shape, request/response JSON, error codes) before carl-studio-side work begins. OR: stub carl-studio against a local mock and land backend in parallel.</remedy>
<priority>P0 — resolve before v0.10-A #1 implementation</priority>
</finding>

<finding id="P0-3" severity="high" category="plan-estimate">
<claim>py2bend rollout-loop compilation estimated at 350 LOC is optimistic. Realistic range: 600-700 LOC when rollout.bend template, BUSL admin gate, Pyodide fallback path, and tests are counted.</claim>
<source>peer_review_plan.json R2, C1</source>
<impact>Schedule slip risk for v0.10-A #2.</impact>
<remedy>Before committing to the LOC estimate: write a minimal `rollout.bend` template (≤100 lines) that covers (a) K-sample parallel reduction, (b) reward evaluation, (c) amb-choice coupling to `τ = 1 − crystallization`. Measure the actual translation cost + testing surface. Revise estimate.</remedy>
<priority>P0 — de-risk estimate before code</priority>
</finding>

### P1 — High, land in v0.10

<finding id="P1-1" severity="high" category="architecture">
<claim>BaseGate gates on predicates (consent, tier) but NOT on coherence R. The IRE claims "coherence-gated routing" but `kuramoto_r` is ignored by the gating mechanism entirely.</claim>
<source>peer_review_architecture.json G2</source>
<impact>The G in `(M, I, Φ, G)` is incomplete. Coherence-gated routing is aspirational, not realized.</impact>
<remedy>Add optional `CoherenceGate` predicate in `src/carl_studio/gating.py` that consults `kuramoto_r` from the current chain's recent window. Emit `carl.gate.coherence_insufficient` when R is below threshold. Opt-in per feature via `@coherence_gate(min_R=0.5)`.</remedy>
<priority>P1 — v0.10 scope, ~60 LOC + tests</priority>
</finding>

<finding id="P1-2" severity="high" category="architecture">
<claim>Step coherence fields (phi, kuramoto_r, channel_coherence) are optional and rarely populated. Auto-attachment at LLM_REPLY / TOOL_CALL boundaries doesn't exist.</claim>
<source>peer_review_architecture.json G4</source>
<impact>Coherence signals are present in the schema but unused in practice. The phi field is a first-class concept in BITC/IRE papers — the gap is observability debt.</impact>
<remedy>Add a `CoherenceAttachment` callback registered on InteractionChain that, on LLM_REPLY / TOOL_CALL step creation, runs `CoherenceProbe.snapshot()` and populates phi + kuramoto_r fields. Opt-in via `CARL_COHERENCE_AUTO_ATTACH=1`. Performance budget: <10ms per step.</remedy>
<priority>P1 — v0.10 scope, ~80 LOC + tests</priority>
</finding>

<finding id="P1-3" severity="medium" category="ip-verification">
<claim>The agent-card spec assumes @terminals-tech/agent is MIT-licensed. This needs source-level verification before TerminalAgent is mirrored into carl-studio's a2a/ module.</claim>
<source>peer_review_ip.json (IP Gap 1, medium)</source>
<impact>If the package's per-file headers or LICENSE override package.json's MIT declaration, mirroring creates a license violation.</impact>
<remedy>Read `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/packages/agent-sdk/LICENSE` + check per-file headers in `src/terminal.ts`. Confirm MIT at both layers. Document the verification in the AgentCardRow module docstring.</remedy>
<priority>P1 — 10-minute verification, do before any code</priority>
</finding>

### P2 — Medium, v0.10.1 or later

<finding id="P2-1" severity="medium" category="architecture-design">
<claim>The proposed "carl-sense" primitive is 80% thin composition of existing observables (chain + phi + gate predicates). Not a novel load-bearing module.</claim>
<source>peer_review_architecture.json R4</source>
<impact>Building it as a new module creates premature abstraction.</impact>
<remedy>Implement `compose_presence_report()` as a QUERY HELPER in `packages/carl-core/src/carl_core/presence.py` (~50 LOC). Returns `PresenceReport` dataclass with (R, psi, crystallization, constructive, impedance, readiness, open_gates). Register as MCP tool `carl.presence.self` via server.py so agents can call it. Do NOT create a `carl_sense/` package.</remedy>
<priority>P2 — v0.10.1, ~70 LOC total incl. MCP tool wire</priority>
</finding>

<finding id="P2-2" severity="medium" category="ip-hygiene">
<claim>`packages/carl-core/` is missing a LICENSE file. Root carl-studio LICENSE declares MIT, but the subpackage should carry its own since it's published as a separate wheel.</claim>
<source>peer_review_ip.json (IP Gap 2, minor)</source>
<remedy>Add `packages/carl-core/LICENSE` (MIT text, same as root). One-line fix.</remedy>
<priority>P2 — 1-minute fix, do in the same v0.10 ship</priority>
</finding>

<finding id="P2-3" severity="medium" category="documentation">
<claim>The `load_private()` fallback pattern in `src/carl_studio/admin.py` is undocumented for external contributors.</claim>
<source>peer_review_ip.json (IP Gap 3, documentation)</source>
<remedy>Add a section to `docs/private_integration.md` (landed in v0.8) explaining the fallback contract: when admin-gate fails or terminals-runtime absent, the lazy-import path returns a minimal MIT-safe stub. Show the pattern via the coherence_observer.py example.</remedy>
<priority>P2 — doc only, 15-minute fix</priority>
</finding>

<finding id="P2-4" severity="medium" category="plan-dependency">
<claim>prime-rl availability on PyPI is unconfirmed. `carl-env` reserves a verifier hook for v0.10+ prime-rl integration, but if the package doesn't exist, the hook is dead code.</claim>
<source>peer_review_plan.json R4, C4</source>
<remedy>Check PyPI. If prime-rl is available, note the pin in pyproject.toml optional-dependencies (extras only, not required). If not, document as "future integration point" and remove concrete prime-rl references from design docs until it lands.</remedy>
<priority>P2 — 5-minute verification before shipping carl-env</priority>
</finding>

### P3 — Tracked, not scheduled

<finding id="P3-1" severity="medium" category="architecture-deferred">
<claim>InteractionChain's context dict is flat; IRE's "context manifold M" implies rich, queryable, layered state.</claim>
<source>peer_review_architecture.json G3</source>
<remedy>v0.11+ candidate: add typed context layers (user_intent / agent_state / tool_response / coherence_snapshot) accessible by XPath-style queries. Low value until concrete use case exists.</remedy>
<priority>P3 — defer until a consumer needs it</priority>
</finding>

<finding id="P3-2" severity="low" category="plan-housekeeping">
<claim>chat_agent.py at 2,443 LOC is still deferred. v0.8 CHANGELOG flagged for v0.9; v0.9 re-defers to v0.9.1+.</claim>
<source>peer_review_plan.json C6 + v0.8 CHANGELOG</source>
<remedy>Anti-Deferral Protocol from Tej's global CLAUDE.md: three re-deferrals = failure mode. This is deferral #2 (v0.8 → v0.9, v0.9 → v0.10). **If it gets deferred again in v0.10, it becomes P1 in v0.10.1.** Tracking explicitly now.</remedy>
<priority>P3 now, P1 if re-deferred one more time</priority>
</finding>

</findings_dag>

## Part 2 — Anti-pattern catalog (vanilla-context mirror findings)

The vanilla peer-review agents — same model as the dispatcher, bare harness — surfaced the following anti-patterns that the *dispatcher's full-context self* either caught-after-the-fact or missed entirely. These are pre-registered filters for future sessions.

<antipatterns>

<antipattern id="AP-1" caught_by="peer_review_architecture.json" dispatcher_pre_registered="no" severity="high">
<description>Assumption that `chat_agent.py` logs tool calls because it logs everything else. Reality: CLI + memory ARE logged, tool calls are NOT. The vanilla agent READ THE CODE and caught it; the full-context dispatcher ASSUMED.</description>
<lesson>When reviewing an agentic loop, don't assume coverage. Read the execution path end-to-end and verify every action-type emits a step.</lesson>
</antipattern>

<antipattern id="AP-2" caught_by="peer_review_architecture.json" dispatcher_pre_registered="no" severity="high">
<description>Assumption that BaseGate realizes coherence-gated routing. Reality: BaseGate gates on predicates; coherence R is schema-present but unconsulted. The vanilla agent traced the gating path and found the gap.</description>
<lesson>When code claims to realize a theoretical construct, trace the data flow and verify the CONNECTIVITY between schema field and consumption site. Schema-present ≠ semantically-used.</lesson>
</antipattern>

<antipattern id="AP-3" caught_by="peer_review_plan.json" dispatcher_pre_registered="partial" severity="medium">
<description>LOC estimate optimism. Dispatcher claimed 350 LOC for py2bend rollout compilation; vanilla agent estimated 600-700. Pattern: not counting test surface, BUSL-gate boilerplate, Pyodide fallback path.</description>
<lesson>Multiply first-pass LOC estimates by 1.8×. Count: implementation + tests + error path + fallback + documentation.</lesson>
</antipattern>

<antipattern id="AP-4" caught_by="peer_review_ip.json" dispatcher_pre_registered="no" severity="medium">
<description>Taking MIT license declaration on faith. Dispatcher asserted @terminals-tech/agent is MIT based on package.json. Vanilla agent flagged that per-file headers or LICENSE file could override.</description>
<lesson>Verify licenses at BOTH package.json AND LICENSE file AND per-file headers before any copy-or-mirror action. 10-minute cost, unbounded legal benefit.</lesson>
</antipattern>

<antipattern id="AP-5" caught_by="peer_review_architecture.json" dispatcher_pre_registered="yes" severity="medium">
<description>Proposing a new primitive when composition of existing suffices. Dispatcher proposed "carl-sense" as a new primitive. Vanilla agent verdict: 80% thin composition, implement as query helper.</description>
<lesson>Pre-registered anti-pattern confirmed. When proposing a new primitive, run the 80/20 test: "if this is 80% composition of X, Y, Z, just compose them." Don't create packages for query helpers.</lesson>
</antipattern>

<antipattern id="AP-6" caught_by="dispatcher_pre_registered" severity="high">
<description>Treating κ discrepancy as a bug. Pre-registered. Vanilla math agent correctly resolved — both values are correct.</description>
<lesson>When two values differ at ~0.17% across a theoretical/empirical boundary, the default assumption is "intentional calibration." Verify with the author before proposing changes.</lesson>
</antipattern>

<antipattern id="AP-7" caught_by="dispatcher_pre_registered" severity="high">
<description>Framing HVM integration as speed optimization. Pre-registered. No vanilla agent repeated it in this batch — the reframe landed.</description>
<lesson>The CLAUDE.md anchor works. Mental-model seeding in persistent memory files prevents future-session backsliding.</lesson>
</antipattern>

</antipatterns>

## Part 3 — Consolidated v0.10 scope (DAG, JSON-typed)

<scope_dag format="json">

```json
{
  "version": "v0.10",
  "resonance_confirmed": "2026-04-20",
  "tracks": [
    {
      "id": "v0.9.0-alpha",
      "name": "Close witness-log + ship carl-update + carl-env",
      "blocks": ["v0.10-A"],
      "workstreams": [
        {"id": "W1", "title": "Tool-call witness fix", "finding": "P0-1", "loc": 25, "priority": "P0", "files": ["src/carl_studio/chat_agent.py", "tests/test_chat_agent_witness.py"]},
        {"id": "W2", "title": "Implement carl-update", "design": "docs/v09_carl_update_design.md", "loc": 900, "priority": "P1"},
        {"id": "W3", "title": "Implement carl-env", "design": "docs/v09_carl_env_design.md", "loc": 1100, "priority": "P1"},
        {"id": "W4", "title": "Add carl-core LICENSE", "finding": "P2-2", "loc": 1, "priority": "P2"},
        {"id": "W5", "title": "Document load_private() pattern", "finding": "P2-3", "loc": 0, "priority": "P2"}
      ]
    },
    {
      "id": "v0.10-A",
      "name": "Agent-cards + primitive-integration (MIT-clean picks first)",
      "blocks": ["v0.10-B"],
      "blocked_by_external": ["carl.camp backend contract for POST /api/sync/agent-cards (P0-2)"],
      "workstreams": [
        {"id": "W6", "title": "Verify @terminals-tech/agent MIT at file level", "finding": "P1-3", "loc": 0, "priority": "P1", "action": "read LICENSE + headers"},
        {"id": "W7", "title": "Mirror TerminalAgent + AgentCardRow + Supabase writer", "spec": "docs/v10_agent_card_supabase_spec.md", "loc": 340, "priority": "P1"},
        {"id": "W8", "title": "py2bend rollout-loop compilation", "design": "docs/v10_terminals_tech_deep_dive.md#v0.10-A-2", "loc": 700, "priority": "P1", "risk": "LOC estimate de-risk required (P0-3)"},
        {"id": "W9", "title": "CoherenceGate predicate", "finding": "P1-1", "loc": 80, "priority": "P1"},
        {"id": "W10", "title": "Coherence auto-attach on LLM_REPLY / TOOL_CALL", "finding": "P1-2", "loc": 100, "priority": "P1"},
        {"id": "W11", "title": "Substrate presence probe (admin-gated)", "design": "docs/v10_terminals_tech_deep_dive.md#v0.10-A-3", "loc": 80, "priority": "P1"}
      ]
    },
    {
      "id": "v0.10-B",
      "name": "Presence report as MCP tool + housekeeping",
      "workstreams": [
        {"id": "W12", "title": "compose_presence_report() + carl.presence.self MCP tool", "finding": "P2-1", "loc": 70, "priority": "P2"},
        {"id": "W13", "title": "Confirm prime-rl on PyPI + pin or defer", "finding": "P2-4", "loc": 5, "priority": "P2"},
        {"id": "W14", "title": "Emit AXON-isomorphic events via HTTP to carl.camp", "context": "top-5 signals list in docs/v10_terminals_tech_deep_dive.md", "loc": 120, "priority": "P2"}
      ]
    },
    {
      "id": "v0.11-plus",
      "name": "Deferred with explicit tracking",
      "workstreams": [
        {"id": "W15", "title": "chat_agent.py decomposition", "finding": "P3-2", "loc": 0, "priority": "P3-becoming-P1-if-deferred-again"},
        {"id": "W16", "title": "Typed context manifold for InteractionChain", "finding": "P3-1", "loc": 0, "priority": "P3"},
        {"id": "W17", "title": "AT Protocol federation for agent discovery", "context": "Bluesky DIDs per terminals-tech pattern", "priority": "P3"},
        {"id": "W18", "title": "Heawood graph (14-vertex) coherence topology research", "context": "BITC paper §6.1 next step", "priority": "P3-research"}
      ]
    }
  ],
  "total_loc_v0_9_alpha_and_v0_10_A": 3426,
  "total_loc_v0_10_A_admin_gated_subset": 960,
  "external_blockers": ["carl.camp backend endpoint spec"]
}
```

</scope_dag>

## Part 4 — ER diagram (agent-cards + adjacent tables)

<er_diagram format="mermaid_ish">

```
┌──────────────────────┐         ┌────────────────────────────┐
│   carl.camp USER     │   1:N   │  agent_cards (PGLite v1.19)│
│──────────────────────│◀────────│────────────────────────────│
│ user_id (PK)         │         │ id          (PK)           │
│ tier (FREE|PAID)     │         │ user_id     (FK→USER)      │
│ jwt_expires_at       │         │ slug                       │
│ x402_wallet_addr     │         │ name                       │
└──────────────────────┘         │ description                │
                                 │ manifest    (JSONB)        │
                                 │ metadata    (JSONB)        │
                                 │ visibility                 │
                                 │ created_at                 │
                                 │ updated_at                 │
                                 │ UNIQUE (user_id, slug)     │
                                 │ INDEX (user_id)            │
                                 │ INDEX (visibility)         │
                                 └─────────────┬──────────────┘
                                               │ 1:N (sync push via electric-bridge)
                                               ▼
┌──────────────────────────────────────────────────────────────┐
│          SUPABASE (remote, carl.camp-mediated)               │
│──────────────────────────────────────────────────────────────│
│  agent_cards                                                 │
│  ├── mirrors local schema exactly                            │
│  ├── RLS policy: user_id = auth.uid()                        │
│  ├── Public-read RLS for visibility='public'                 │
│  └── conflict strategy: last-write-wins via upsert(onConflict│
│                         ='id') — mirrors deployment-sync.ts  │
│                                                              │
│  auth.users (Supabase built-in)                              │
│  ├── id (uuid) — linked to carl.camp user_id                 │
│  └── JWT signing via Supabase Auth                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────┐        ┌──────────────────────┐
│   carl-studio local  │        │ Step (InteractionChain)
│   SQLite (LocalDB)   │        │──────────────────────│
│──────────────────────│        │ id                   │
│ agent_cards (v0.10)  │ N:1 ──▶│ chain_id             │
│ runs                 │        │ action (ActionType)  │
│ metrics              │        │ input JSONB          │
│ gates                │        │ output JSONB         │
│ sync_queue           │        │ success              │
│ contracts            │        │ phi FLOAT (optional) │
│ sticky_notes         │        │ kuramoto_r FLOAT     │
│ skill_runs           │        │ channel_coherence    │
│ [metadata...]        │        │ started_at           │
└──────────────────────┘        │ duration_ms          │
                                │ parent_id            │
                                │ trace_id             │
                                │ session_id           │
                                └──────────────────────┘
```

**Parity principle (per Tej's spec):** Free tier = full local SQLite
functionality including ALL features. Paid tier = same + Supabase sync
+ agent-card marketplace discovery. No FREE-tier feature is crippled.
The carl.camp mediation layer adds durability, discovery, and x402
receipts — not core capability.
</er_diagram>

## Part 5 — Fractal policy (self-similar across layers)

<fractal_policy>

<policy_layer level="L0" scope="single-step">
<rule>Every `ActionType` emitted MUST populate phi + kuramoto_r if `CARL_COHERENCE_AUTO_ATTACH=1`. Budget: <10ms/step.</rule>
<rule>Every tool call MUST record `ActionType.TOOL_CALL` with name/input/output/success. No exceptions.</rule>
<invariant>Witness completeness: if an action happened, it appears in the chain.</invariant>
</policy_layer>

<policy_layer level="L1" scope="single-training-run">
<rule>Every rollout batch emits `skill_training_started` / `skill_crystallized` / `coherence_update` / `interaction_created` / `action_dispatched` — via carl.camp HTTP (AXON-isomorphic shape).</rule>
<rule>Every run records a terminal `Step` with final R, crystallization state, total coherence mass.</rule>
<invariant>Run reproducibility: given the same RNG seed + py2bend-compiled rollout + same reward fn, the chain trace is bit-exact across machines (v0.10-A W8).</invariant>
</policy_layer>

<policy_layer level="L2" scope="research-session">
<rule>`carl env` composes a TrainingConfig via functor-composed questions; state is JSON-serializable to `~/.carl/last_env_state.json` for resume.</rule>
<rule>Session-level coherence is the SUM of run-level coherence weighted by session continuity (not currently computed — P3 candidate).</rule>
<invariant>Session amnesia recovery: if the session crashes, `carl env --resume` restores state without data loss.</invariant>
</policy_layer>

<policy_layer level="L3" scope="user-account">
<rule>FREE tier: local SQLite, no paywall on any training/eval feature. One-line FYI nudges for publish/marketplace/sync actions.</rule>
<rule>PAID tier: same + Supabase sync + agent-card marketplace publish + x402 receipts.</rule>
<invariant>Parity guarantee: no FREE-tier feature is strictly weaker than the PAID-tier equivalent; PAID only adds durability and discovery.</invariant>
</policy_layer>

<policy_layer level="L4" scope="marketplace">
<rule>Agent cards sync via `last-write-wins` (monotonic edit semantics).</rule>
<rule>Marketplace discovery is visibility-gated: 'private' (user only), 'unlisted' (link only), 'public' (RLS-open).</rule>
<invariant>Privacy default: visibility='private' unless user explicitly chooses otherwise.</invariant>
</policy_layer>

<policy_layer level="L5" scope="platform-terminals-OS">
<rule>carl-studio emits AXON-isomorphic events; terminals.tech web app observes via mesh.</rule>
<rule>Webcontainer-per-subagent lives in carl-app (TS frontend), not carl-studio.</rule>
<invariant>Layer boundary: Python CLI layer does not import TS webcontainer machinery.</invariant>
</policy_layer>

<meta_pattern>
At every layer, the invariants have the same shape: witness completeness
(everything observable), reproducibility (same input → same trace),
privacy default (least-permissive visibility). The policy is fractal:
self-similar at each scale, enforced by the same gate machinery
(BaseGate + CoherenceGate when coherence-gated routing lands in W9).
</meta_pattern>

</fractal_policy>

## Part 6 — Validation SOP (checklist, self-contained)

<sop_checklist>

### Before starting v0.9.0-alpha implementation

- [ ] Verify `/tmp/peer_review_*.json` artifacts (math · architecture · plan · ip) are all present and readable.
- [ ] Read this doc end-to-end. Confirm resonance with all findings.
- [ ] Resolve P0-2 external blocker: get carl.camp backend contract for `POST /api/sync/agent-cards` from Tej OR agree to stub-and-defer.
- [ ] Read `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/packages/agent-sdk/LICENSE` and per-file headers to verify MIT (P1-3 / W6).
- [ ] Check `pip show prime-rl` (or web-search for "prime-rl PyPI 2026-04") to resolve W13.

### During v0.9.0-alpha implementation

- [ ] W1 lands first: tool-call witness fix in `chat_agent.py`. Must be provable via `tests/test_chat_agent_witness.py`.
- [ ] W2 + W3 parallel: carl-update + carl-env per their design docs.
- [ ] W4: `cp LICENSE packages/carl-core/LICENSE` (1-line shipping fix).
- [ ] W5: document `load_private()` pattern in `docs/private_integration.md`.
- [ ] Three-gate sign-off before commit: pytest green · ruff clean · pyright ≤ baseline.

### During v0.10-A implementation

- [ ] W6 FIRST (license verification) — it's 10 minutes and blocks W7.
- [ ] W8 de-risk: write the minimal `rollout.bend` template. Measure actual translation cost. Revise LOC estimate. If >800 LOC, split into W8a (scaffold) + W8b (full rollout).
- [ ] W9 + W10 together: CoherenceGate + auto-attach. They share the coherence read path.
- [ ] No commit until all tests green + AXON-isomorphic event emission verified against a carl.camp staging endpoint.

### Post-implementation continuous

- [ ] Every session: read CLAUDE.md. Update if new findings land.
- [ ] Every PR: run the peer-review-agent dispatch with 4 vanilla contexts on the diff. Collect anti-patterns.
- [ ] Every release: update `applies_to` field in docs frontmatter. If a doc goes stale, either update or delete.
- [ ] Track deferrals: if an item gets deferred a 3rd time, it auto-promotes to P0 (Anti-Deferral Protocol).

</sop_checklist>

## Part 7 — Confidence statement

<confidence overall="high" with_caveats="yes">

<high_confidence items="verified">
- IRE tuple (M, I, Φ, G) maps to carl-core primitives. Verified by peer review.
- κ = 64/3 is the canonical exact value; 21.37 is calibrated. Ruling stands.
- MIT ↔ BUSL seam via admin-gate + lazy-import is load-bearing and compliant.
- AXON signal shapes are the canonical vocabulary; HTTP-mediated emission is the right path for carl-studio.
- Agent-card Supabase flow via carl.camp is MIT-clean and implementable.
- Webcontainer is carl-app territory, not carl-studio.
- Fractal policy at L0-L5 holds self-consistently.
</high_confidence>

<medium_confidence items="peer-review-surfaced">
- py2bend rollout compilation LOC estimate (600-700, not 350). Will re-measure with a minimal template.
- CoherenceGate + auto-attach design feasibility — seems sound but hasn't been prototyped.
- carl-sense as composition helper (not primitive) — agent verdict lowered it, will implement as P2.
</medium_confidence>

<explicit_gaps items="acknowledged">
- HVM interaction-net reduction at trace level — conceptual not mechanical. Will ground during W8 rollout.bend write.
- Time-under-tension ↔ training-volume-vs-intensity mapping. Not derived; useful intuition for reward schedule tuning later.
- carl.camp backend contract (P0-2) — external blocker, needs Tej input.
</explicit_gaps>

</confidence>

## Part 8 — What this doc is NOT

- It is **not** a marketing document. "Maximum profitability/virality" are outcomes of many product decisions; this doc surfaces enablers, not guarantees.
- It is **not** a harmonization proposal for κ values. The 64/3 vs 21.37 delta stays.
- It is **not** a license change. MIT carl-studio stays MIT; BUSL terminals-runtime stays BUSL; 2030 Apache-2.0 horizon noted.
- It is **not** a rewrite proposal. v0.8.0 surface is stable; v0.10 is additive + fixes witness debt.

<closing_note>
This plan is phase-locked to 2026-04-20. If the date in your context
is later than 2026-06-01 and this doc's `applies_to: v0.10` has shipped,
treat this as archival. Future sessions should check `git log
docs/v10_master_plan.md` for any supersession commits before acting
on findings.
</closing_note>
