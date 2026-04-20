---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10-roadmap
status: roadmap
supersedes_partially: docs/v09_terminals_runtime_integration_matrix.md
---

# terminals.tech deep-dive — v0.10 roadmap (grounded, 2026-04-20)

This document consolidates evidence from a four-agent parallel review of
five verified paths Tej provided:

- `/Users/terminals/Documents/terminals-tech-landing/observable-computation/`
- `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/packages/`
- `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/public/wasm/`
- `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/public/compiler/`
- `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/public/workers/`

Plus the four published research papers in `observable-computation/papers/`:
BITC, DMC, IRE-paradigm, Heawood-hypothesis.

**What changed vs the earlier v0.9 integration matrix:** two real
corrections, five confirmations, three new v0.10-A picks. The earlier
matrix (`v09_terminals_runtime_integration_matrix.md`) remains accurate
for its narrower scope but is partially superseded here.

## Corrections to the earlier v0.9 matrix

### Correction 1 — `agent-sdk` is MIT, not BUSL-1.1

`packages/agent-sdk/package.json` declares MIT license. Published as
`@terminals-tech/agent` on npm. Exports `Terminal`, `TerminalAgent`,
`TerminalResult`, `TerminalConfig`, plus composition utilities
(pipes/parallel/race) and Team archetypes.

**This unblocks Track 2 (agent-cards + Supabase) from v0.10-B → v0.10-A.**
carl-studio can **mirror the `TerminalAgent` type directly** into
`src/carl_studio/a2a/types.py` without touching the admin gate. MIT ↔ MIT
is clean.

`TerminalAgent` shape:
```typescript
interface TerminalAgent {
  role: string;                 // "researcher", "writer", "reviewer"
  capabilities?: string[];      // opaque strings, no validation at SDK level
}
```

### Correction 2 — `semantic-mesh/convergence/` is NOT Kuramoto

Named "convergence" but the code is task/agent health monitoring
(healthScore, activeAgents, throughput, errorRate). **CARL is ahead
here.** `packages/carl-core/src/carl_core/coherence_trace.py::kuramoto_R`
is a full Kuramoto order-parameter implementation with phase coupling.

**Action:** remove "semantic-mesh convergence" from the v0.9-B
integration matrix. No integration opportunity; carl-core's primitive is
the canonical one.

## Confirmations (findings that reinforced the matrix)

| Claim | Evidence | License |
|-------|----------|---------|
| pglite-worker.js wraps ElectricSQL PGLite (Postgres-in-WASM, IndexedDB-backed) | 55-line worker imports `@electric-sql/pglite/worker`; uses vector/uuid/trgm/ltree/seg/tcn/tablefunc/live extensions | Apache-2.0 (ElectricSQL) |
| wasm-core exposes Substrate (Kuramoto physics at 384-dim) + WorldGrid (41-zone) + analyze_bands + centroid + compose_ops | `public/wasm/terminals-core/terminals_wasm.d.ts` fully typed | **BUSL-1.1** |
| py2bend compiles a restricted Python subset to HVM2 | `observable-computation/packages/py2bend/src/` 2,308 LOC; HVM2 syntax confirmed in `public/wasm/hvm/agent-logic.hvm` (`@name`, `?`, `&!`, `~`, `$`) | **BUSL-1.1**, patent-pending (Intuition Labs LLC) |
| void-lock implements two-branch runtime partition (DMC paper §3.1) | `compiler.ts::classifyPythonExecutionStrategy()` routes compiled vs fallback with execution witness | **BUSL-1.1** |
| Ghost is a localhost:3002 daemon (headless voice + model serving + mesh relay) | `packages/ghost/src/client.ts` | **BUSL-1.1** |
| Mesh package (distinct from semantic-mesh) provides signal-bus: `emit(topic, data)` + `subscribe(pattern, handler)` with wildcard matching | `packages/mesh/src/signals.ts` | **BUSL-1.1** |
| No `AgentCard` type, no Supabase integration, no x402 markers in the terminals-tech SDK | grep for "Supabase" / "x402" / "PaidCapability" returns zero hits | — |

## Research papers ↔ shipped CARL code

| Paper | Core object | CARL's implementation witness |
|-------|-------------|-------------------------------|
| Bounded Informational Time Crystals (DOI 10.5281/zenodo.18906944) | 5-axiom BITC: boundedness + recurrence + endogenous measurability + contrastive coherence + witnessability | `InteractionChain` (bounded window) + `CoherenceTrace` (recurrence) + `compute_phi` (endogenous) + `T_STAR(d) = int(KAPPA * d)` (contrastive) + `CoherenceProbe` (external witness) |
| Deterministic Mesh Compilation (DMC) | Two-branch runtime partition (compiled Bend / fallback Python) with trace observability | py2bend + fallback pyodide in terminals-tech; **NOT YET in carl-studio** — v0.10 integration candidate |
| IRE Paradigm | `(M, I, Φ, G)` tuple: manifold + interactions + correspondences + coherence gate | `carl_core.interaction` (M) + `ActionType`+`Step` (I) + `compute_phi`/`kuramoto_R` (Φ) + `BaseGate`+`GatingPredicate` (G) — **all four present in v0.8.0** |
| Heawood-hypothesis | Proposes Heawood graph (14-vertex, 3-regular, bipartite Levi of PG(2,2)) as genuinely sparse projective witness at N=14 | **Research frontier** — v0.11+ candidate topology for the CARL presence substrate |

**Observation:** CARL's v0.8.0 already realizes the IRE `(M, I, Φ, G)`
tuple. The missing piece is the DMC compilation branch (py2bend ↔
reward function compilation). That's the single largest v0.10 unlock.

**⚠ Fano plane / K_7 identity correction (from BITC paper §6.1).** The
Fano collinearity graph at N=7 is **isomorphic to K_7** (every two
points in PG(2,2) share exactly one line → full connectivity). The
"degree-3" refers to line incidence, not point adjacency. If CARL
implements a Fano-based witness for coherence, it should either:
(a) use K_7 all-to-all (fine at small N, O(N²) cost) or
(b) use Ring(7) / Sparse3(7) / Heawood(14) for sparse dynamics.

## v0.10 integration matrix (updated, ranked)

### v0.10-A — highest leverage, MIT-clean or well-scoped BUSL gate

**1. Mirror `TerminalAgent` type + implement Supabase agent-card writer**

- Source: `packages/agent-sdk/src/terminal.ts` lines 38-72 (**MIT**).
- CARL seam: new `src/carl_studio/a2a/agent_card.py` Pydantic mirror of `TerminalAgent`. Extends with CARL-specific fields (`tier`, `x402_receipt_hash`, `created_at`, `last_seen_at`).
- Supabase write path: existing `src/carl_studio/camp.py` HTTP contract → new endpoint `POST /api/agent_cards`.
- FREE-tier gate: `tier_gate(Tier.PAID, feature="agent_marketplace_publish")` wrapping the Supabase write call. On DENY, `CampConsole.notice()` prints one-line FYI per Tej's "no popups" rule: `ℹ  Agent registered locally. Upgrade to publish: carl camp upgrade`.
- LOC: ~180 (type + writer + CLI wire + tests).
- **Unblocked from v0.10-B → v0.10-A** by the MIT-license correction.

**2. py2bend rollout-loop compilation — CARL becomes a native IRE on HVM (admin-gated)**

- Source: `observable-computation/packages/py2bend/` (**BUSL-1.1**).
- **Framing correction (2026-04-20):** an earlier draft framed this as
  "compile `compute_reward()` for 5–10× throughput." That's shallow
  optimization thinking. The real isomorphism — stated explicitly in
  the BITC, DMC, and IRE papers — is that **CARL's GRPO rollout loop
  IS already an Interactive Research Environment**. The `(M, I, Φ, G)`
  tuple maps directly onto carl-core: `InteractionChain` (M), `Step` +
  `ActionType` (I), `compute_phi` / `kuramoto_R` (Φ), `BaseGate` (G).
  What's missing is exposing that shape as the *native substrate*.
- **The integration target is the rollout loop**, not a single reward
  function. GRPO's K-sample completion generation → per-completion
  reward evaluation → phi-field witness → amb-choice argmax/softmax is
  isomorphic to HVM interaction-net reduction: parallel-by-construction
  redexes, Church-Rosser confluent, amb-choice coupled to
  `τ = 1 − crystallization` per the IRE paper.
- CARL seam: a new `src/carl_studio/training/rollout_bend.py` writes the
  rollout as a Bend program (via py2bend admission path), compiles once
  at trainer startup, executes on HVM. Sequential pyodide fallback for
  the rejected subset (DMC's two-branch partition).
- Admin gate: new `carl_studio.compilers.bend_bridge` module lazy-imports
  py2bend under `CARL_BEND_ENABLED=1` + admin unlock. When disabled,
  the existing torch-based rollout runs unchanged.
- What this unlocks, in order of importance:
  1. **Deterministic reproducibility.** Same RNG + same net = identical
     reduction trace. GRPO runs become bit-exact reproducible across
     machines, which the torch path is not.
  2. **Anticipatory variance minimization.** Branch points ("void
     points" / amb choice points / cognitive-dissonance vectors)
     parallelize by construction; the phi-field witness crystallizes
     them without premature cutoff.
  3. **Mesh-visible execution trace** (DMC paper §3.3). Every reduction
     step is introspectable — training trace is a first-class object.
  4. **Speed as a side effect** (possibly 5–10× on multi-completion
     batches, unverified, not the point).
- LOC: ~350 (bridge + rollout-Bend writer + witness wiring + tests).
  Not "first v0.10 beta" — **this is the load-bearing v0.10 pick**
  because it realizes the IRE substrate CARL already gestures at.

**3. WASM Substrate as a presence probe (admin-gated)**

- Source: `public/wasm/terminals-core/` (**BUSL-1.1**, exposes `Substrate.tick(bass, mid, high, entropy, dt) → [R, entropy, converged, step]`).
- CARL seam: `packages/carl-core/src/carl_core/coherence_observer.py` already lazy-imports from `terminals_runtime.observe`. Extend with a `Substrate` probe that callers can optionally consult at eval time.
- Fallback: carl-core already has a numpy `kuramoto_R`. MIT stub preserves contract when Substrate absent.
- LOC: ~80 (lazy import + tests + doc).

### v0.10-B — after v0.10-A lands

**4. Mesh signal bus (BUSL-1.1).** Bridge carl-studio events into the
terminals.tech mesh for cross-process coordination. Useful when the CARL
heartbeat runs alongside the terminals.tech web app (Tauri/Electron host
scenario). Defer until a concrete cross-process use case arrives.

**5. void-lock route-signal as reward component.** Add a
`route_signal` field to `RewardComponents` that rewards programs that
py2bend can compile (compiled branch) vs those that fall back.
Rationale: compiled-path programs are more structurally legible →
higher "observability" reward. Depends on integration 2.

**6. Ghost as optional inference backend.** `@terminals-tech/ghost`
daemon at `localhost:3002` for free local inference. Could slot in via
CARL's existing provider-fallback cascade. Unclear whether it's
Python-accessible yet.

### v0.10-C — deferred with rationale

| Item | Why deferred |
|------|--------------|
| pglite migration | Keep SQLite. Bridge cost (~500 LOC Python↔JS) > benefit for CLI-Python context. Revisit if web dashboard sync becomes a concrete requirement. |
| transformers.js embeddings-worker | carl-studio already uses HuggingFace; no compelling reason to duplicate via JS worker. |
| semantic-mesh/convergence | Misnamed; not Kuramoto. carl-core's implementation is authoritative. |
| semantic-mesh/matching (BM25+RRF) | Standard retrieval math. Can port directly when CARL needs retrieval — no IP concern since it's textbook algorithms. |
| WorldGrid (41-zone orbital world) | terminals.tech product layer, not CARL training/eval. |
| `analyze_bands` (audio FFT) | Audio is out of scope for CARL's model-layer concerns. |
| `compose_ops` (emergence algebra) | Needs terminals.tech context to be meaningful; tangential to training loops. |

## License summary (verified across this review)

| Package | License |
|---------|---------|
| **observable-computation/py2bend** | **BUSL-1.1** (4-year exclusion → Apache-2.0 in 2030; patent-pending) |
| **agent-sdk** | **MIT** ← unblocks the agent-card work |
| sdk (`@terminals-tech/sdk`) | BUSL-1.1 |
| void-lock | BUSL-1.1 |
| semantic-mesh | BUSL-1.1 |
| mesh | BUSL-1.1 |
| ghost | BUSL-1.1 |
| cli (`@terminals-tech/cli`) | MIT |
| terminals-core WASM | BUSL-1.1 |
| ElectricSQL pglite (transitive) | Apache-2.0 |
| transformers.js (transitive) | Apache-2.0 |

The admin-gate + lazy-import pattern (`src/carl_studio/admin.py`,
`packages/carl-core/src/carl_core/coherence_observer.py`) is the
canonical seam for BUSL-1.1 integrations. MIT-licensed pieces can be
mirrored directly (agent-sdk, cli patterns) without the gate.

## Still-deferred until separate grounding sessions

- **HVM3** (if different from HVM2). agent-logic.hvm appears to be
  HVM2 syntax; need Tej's confirmation on whether HVM3 upgrade is
  planned or already landed elsewhere.
- **Webcontainer runtime** for sandboxed subagent-graph execution. No
  webcontainer code was found in any of the 5 paths reviewed. May live
  in a different repo.
- **BITC / IRE as a training-reward term.** Worth a dedicated methods
  paper (see v0.8 `paper/` series) before implementation — the theory
  is published but the reward-shaping implications need design work.

## Reservations (still standing)

1. **The κ discrepancy remains unresolved.** Same flag as in
   `docs/v09_terminals_runtime_integration_matrix.md` — carl-core has
   `KAPPA = 64/3 ≈ 21.333`, terminals.tech handoff says `κ = 21.37`.
   **Do not touch either side** before Tej rules on the canonical value.

2. **"Maximum profitability / virality / stickiness"** — I can ship
   enablers (agent-cards, Supabase writer, tier-gated marketplace
   publish, Ghost local inference) but the outcomes are lagging
   metrics. No feature ships with a guaranteed revenue multiplier.

3. **The semantic-mesh naming mismatch is a real finding.** If Tej
   intends to add Kuramoto semantics to semantic-mesh/convergence later,
   that's his call — but the current code doesn't do that and CARL
   shouldn't integrate against the name.

4. **BUSL license terms matter.** Every BUSL-1.1 primitive ends
   exclusivity in 2030. Integration plans that assume "always
   BSL-gated" should note that 4-year horizon — the integration seam
   may become MIT-safe without Tej needing to re-license.

## Shipping sequence proposal

- **v0.9.0-alpha** (next session after Tej's approval): implement
  `carl-update` + `carl-env` per their design docs.
- **v0.9.0** (session after): κ-discrepancy resolved; ship v0.9-A
  integrations from the earlier matrix.
- **v0.10.0-alpha**: agent-cards + Supabase writer (MIT-clean,
  un-gated from admin).
- **v0.10.0**: py2bend reward-function compilation (admin-gated).
- **v0.10.1**: WASM Substrate presence probe (admin-gated).
- **v0.11+**: Heawood graph witness topology research + methods paper.
