---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9-preview
status: roadmap
---

# terminals-runtime ↔ carl-studio integration matrix (v0.9 roadmap)

This document describes the **integration surface** between MIT-licensed
`carl-studio` and the BSL-licensed `terminals-runtime` / terminals.tech
primitives. It is a roadmap for v0.9 — no code lands here. Source of
truth for the primitive inventory is the handoff plan at
`/Users/terminals/.claude/plans/create-single-context-concise-wobbly-mist.md`
(hereafter "the plan file"), read 2026-04-20.

## IP boundary restatement

**What stays MIT in carl-studio:**
- Coherence math that is derived from published work with DOIs
  (`compute_phi`, `KAPPA`, `SIGMA`, `DEFECT_THRESHOLD`, `T_STAR` —
  all cite Zenodo 10.5281/zenodo.18906944 + 18992031 in
  `packages/carl-core/src/carl_core/constants.py`).
- The `InteractionChain` witness-log primitive.
- Gating, consent, config-registry, resilience, MCP, x402 surfaces.
- The training pipeline orchestration (cascade, rewards composition).
- Admin gate mechanism (`src/carl_studio/admin.py`) and lazy-loader
  (`load_private(module_name)`).

**What stays BSL in terminals-runtime (proprietary):**
- MicroLM implementation (value/layers/model/trainer source).
- AXON binary ring-buffer protocol + message bus internals.
- SOM 7-agent Kuramoto consensus strategy implementation.
- Sematon witness internals, impedance router, confusion-risk sensor.
- Any analytical system prompts (already precedent: `OBSERVER_SYSTEM_PROMPT`).

**The bridge:** carl-studio **lazy-imports** primitives from
`terminals-runtime`, gated by the hardware-HMAC admin check in
`admin.py`. When terminals-runtime is absent, carl-studio falls back to
a minimal MIT-safe stub that preserves the public contract without the
proprietary methodology. This is already the pattern used by:
- `packages/carl-core/src/carl_core/coherence_observer.py:55-65` —
  loads `terminals_runtime.observe.OBSERVER_SYSTEM_PROMPT` with a
  minimal fallback.
- `src/carl_studio/ttt/slot.py:47-61` — loads `SLOTOptimizerImpl`.
- `src/carl_studio/training/lr_resonance.py:82` — loads
  `ResonanceLRCallbackImpl`.

v0.9 extends this seam for ten new primitives. Adding them does not
re-license any carl-studio code; users without terminals-runtime still
get a full MIT product, just without the advanced behaviors.

## ⚠ κ-constant discrepancy (open question for Tej)

| Source | Value | Derivation |
|--------|-------|------------|
| `packages/carl-core/src/carl_core/constants.py:14` | `KAPPA = 64 / 3 ≈ 21.333` | Exact ratio from Bounded Informational Time Crystals (DOI 10.5281/zenodo.18906944) |
| plan file line 103 (terminals.tech `lib/terminals-tech/core/L0/conservation.ts`) | `κ = 21.37` | "scale-invariant ratio, bits/embedding-dim" |

Delta is ~0.17% relative. Possible explanations:
1. terminals.tech uses an **empirical calibration** against a
   benchmark suite, while carl-studio uses the **exact theoretical
   ratio** from the paper.
2. Different η normalization convention between the two systems.
3. Drift — one system was updated and the other wasn't.

**Recommendation:** do NOT harmonize without Tej's ruling. If (1), the
delta is meaningful and should be documented. If (2), an adapter
layer in the integration seam converts between the two. If (3), pick
the authoritative source and update the other with a CHANGELOG note.

## Integration matrix — primitives ranked by v0.9 leverage

Entries marked **v0.9-A** are the three highest-leverage wins
(design sketches in the next section). **v0.9-B** are integrations
with clear value that lag v0.9-A. **v0.10+** defer until the seam
above lands.

| # | Primitive | Source (per plan file) | CARL gain | carl-studio seam | Gate | Priority | Status |
|---|-----------|------------------------|-----------|------------------|------|----------|--------|
| 1 | Kuramoto-R order parameter | `lib/terminals-tech/core/L0/phase.ts` + `crates/terminals-core/src/substrate/kuramoto.rs` | Sub-ms R computation at eval time; convergence trending drives reward gating | `src/carl_studio/eval/runner.py` (coherence measurement), `training/rewards/composite.py` (phase-adaptive schedule) | `consent_gate("telemetry")` + admin | **v0.9-A** | seam exists, primitive imports proposed |
| 2 | Conservation law checker | `lib/terminals-tech/core/L0/conservation.ts` (`T* = κ·d/η`) | Per-model dynamic token budgets; x402 pricing derived from thermodynamic limit | `src/carl_studio/x402.py` (SpendTracker pre-check), `training/pipeline.py` (context-length validation) | `consent_gate("contract_witnessing")` + admin | **v0.9-A** | κ-discrepancy blocks until resolved |
| 3 | `OBSERVER_SYSTEM_PROMPT` full watchpoint set | `terminals_runtime.observe` | Richer eval observability; proprietary watchpoint catalog | `packages/carl-core/src/carl_core/coherence_observer.py:55` (already partial) | admin only | **v0.9-A** | **already wired** — upgrade is zero-code-change in carl-studio, just requires terminals-runtime to expose a richer export |
| 4 | Impedance router `Z = τ/R` | `crates/terminals-provider/src/router/impedance.rs` | Thermodynamic provider selection; local vs cloud based on coherence state | `src/carl_studio/adapters/_common.py` (provider fallback cascade) | admin | v0.9-B | seam needs adapter refactor |
| 5 | Circuit breaker with failure categories | `lib/terminals-tech/brains/providers/circuit-breaker.ts` | Richer failure taxonomy than carl-core/retry's simple retryable tuple | `packages/carl-core/src/carl_core/retry.py` (extend, don't replace) | none (already public-safe pattern) | v0.9-B | possibly MIT-safe primitive — Tej to decide |
| 6 | AXON signal bus subscriber | `lib/terminals-tech/axon/*.ts` + bridges | Event-driven coupling between training runs, eval, chat agent, heartbeat | `src/carl_studio/heartbeat/loop.py` (publish cycle events), `chat_agent.py` (subscribe to skill signals) | `consent_gate("telemetry")` + admin | v0.9-B | requires AXON bus running in-process (WASM or native) |
| 7 | MicroLM intent classifier | `lib/terminals-tech/core/micro-lm/*.ts` (~2 KLOC pure TS) | <5ms intent labeling; reward terms can weight by intent class | `src/carl_studio/training/rewards/composite.py` (per-intent reward weights), `chat_agent.py` (pre-tool-dispatch classification) | admin | v0.9-B | requires TS→Python bridge or MicroLM ported to Python; non-trivial |
| 8 | SOM 7-agent Kuramoto consensus | `scripts/hle/strategies/som.ts` | Multi-model eval consensus with phase-coupled agreement; replaces simple-ensemble eval | `src/carl_studio/eval/runner.py` (new eval strategy), `a2a/` (for cross-agent coupling) | admin | v0.9-B | needs provider surface + LLM router bridge |
| 9 | Sematon witness + `constructive` flag | `lib/terminals-tech/core/L0/sematon.ts` | Per-step "did this transform again?" invariant in the interaction chain | `packages/carl-core/src/carl_core/interaction.py` (augment `Step`) | admin | v0.10+ | requires InteractionChain schema extension |
| 10 | Confusion-risk sensor | `lib/terminals-tech/sdk/observer.ts` | Proactive pause/suggest when `surprise · (1−attribution) + 0.2·coherence_drop` > threshold | `chat_agent.py` (streaming loop), `eval/runner.py` (early-stop heuristic) | admin | v0.10+ | needs attribution-confidence primitive first |
| 11 | Presence `explain_self` mirror | `lib/resonant/brain/tools.ts` | Agent can narrate its own coherence state as a tool-call | `chat_agent.py` ToolDispatcher (register as MCP tool), `mcp/server.py` | admin | v0.10+ | best as an MCP tool registered by admin users |
| 12 | WhiteHoleNeuron four-view atom | `lib/terminals-tech/core/white-hole/neuron.ts` | Unified atom: Kuramoto phase + HVM combinator + MicroLM expert + HNSW vector | new: `packages/carl-core/src/carl_core/atom.py` | admin | v0.10+ | requires HVM grounding session first |

## Three v0.9-A design sketches

### Sketch 1 — Kuramoto-R at eval time

**MIT side (carl-core extension):**

```python
# packages/carl-core/src/carl_core/presence.py  (new)
"""Phase-oscillator order parameter as a first-class presence metric.

Uses terminals-runtime's native Kuramoto implementation when available,
falls back to a pure-numpy implementation derived from the published
formula R = |(1/N) · Σ exp(iθ)|.
"""
from __future__ import annotations
import numpy as np

def order_parameter(phases: np.ndarray) -> tuple[float, float]:
    """Compute (R, psi) for a phase array. Pure numpy fallback."""
    z = np.exp(1j * phases).mean()
    return float(abs(z)), float(np.angle(z))

try:
    from terminals_runtime.substrate.kuramoto import order_parameter as _rt_order_parameter  # type: ignore
except ImportError:
    _rt_order_parameter = None

def compute_R(phases: np.ndarray) -> tuple[float, float]:
    """Return (R, psi). Uses native runtime when present for speed."""
    if _rt_order_parameter is not None:
        return _rt_order_parameter(phases)
    return order_parameter(phases)
```

**Tests:**
- MIT fallback: known-phase input (all zeros → R=1; uniform → R≈0).
- With-runtime path: same inputs produce outputs within 1e-6.
- Performance: runtime path ≥10× faster than numpy path on N=1024
  phases (assert only when runtime detected).

### Sketch 2 — Conservation-law token budget in x402

**MIT side (x402 upgrade):**

```python
# src/carl_studio/x402.py  (extension)
def _estimate_budget_ceiling(self, model_id: str, embedding_dim: int, eta: float) -> float:
    """T* = κ·d/η — max coherent output in tokens. Uses runtime if present."""
    try:
        from terminals_runtime.conservation import T_star  # type: ignore
        return float(T_star(model_id, embedding_dim, eta))
    except ImportError:
        # Public fallback: use carl-core's KAPPA directly. If runtime
        # uses a different κ (see κ-discrepancy in docs/v09_terminals_
        # runtime_integration_matrix.md), that delta is surfaced here.
        from carl_core.constants import KAPPA
        return float(KAPPA * embedding_dim / max(eta, 1e-6))
```

**Tests:**
- Fallback path uses `KAPPA = 64/3` exactly.
- With-runtime path logs κ used; test asserts value matches whichever
  authoritative source Tej rules on.
- Per-payment: breaches of `T*` raise a new
  `BudgetError(code="carl.budget.coherence_ceiling_exceeded")`.

### Sketch 3 — OBSERVER prompt upgrade (zero-carl-studio-code)

This is **already wired**. `coherence_observer.py:55-65` lazy-loads
`terminals_runtime.observe.OBSERVER_SYSTEM_PROMPT`. Upgrading the
proprietary prompt in terminals-runtime requires **no carl-studio
change** — all that changes is the quality of observer output when
admin is unlocked. Called out here to document the seam exists and to
flag that terminals-runtime should publish a versioned prompt API
(`get_observer_prompt(version: str)`) so carl-studio can request a
specific revision if prompt engineering regresses quality.

## Agent-card + Supabase surface (v0.9-B sketch)

**Already-in-tree seams:**
- `src/carl_studio/a2a/_cli.py` — A2A agent sub-app (`carl agent`).
- `src/carl_studio/camp.py` — `CampProfile` HTTP contract to carl.camp.
- `src/carl_studio/camp.py:38-44` — Supabase edge-function endpoints
  (`check-tier`, `auth/refresh`).

**v0.9-B flow (written, not yet built):**

1. Authenticated user runs `carl agent register` (via `a2a/_cli.py`).
2. carl-studio serializes the agent card (A2A-compliant JSON).
3. For PAID tier users: card + x402 receipt hash → `camp.py` →
   `https://carl.camp/api/agent_cards` → Supabase `agent_cards` table
   (columns: `user_id`, `card_json`, `x402_receipt_hash`, `tier`,
   `created_at`, `last_seen_at`).
4. For FREE tier users: `carl agent register` succeeds locally but
   `camp.py` returns a non-blocking FYI:
   `TierGateError(code="carl.gate.tier_insufficient", tier="FREE") → CampConsole.notice(...)`
   rendered as `ℹ  Agent registered locally. Upgrade to PAID to
   publish to carl.camp marketplace: carl camp upgrade`.
5. NO popups — the nudge is a single-line notice per Tej's rule.

**What's ready:**
- Consent + tier gates (v0.7→v0.8).
- CampProfile HTTP auth + JWT refresh.
- x402 receipt hashing (`contract.py` + `x402.py`).

**What needs private-runtime work:**
- Supabase edge function to accept agent cards.
- Marketplace discovery endpoint (`GET /agent_cards?search=...`).
- MCP tool so peer agents can call into published cards.

## Deferred to v0.10+ (noted, not planned)

| Item | Why deferred |
|------|--------------|
| HVM/HVM3/Bend/py2bend | No verified code path on disk for Claude to ground in; requires separate session with Context7 + repo paths. |
| webcontainer runtime for sandboxed subagent-graph execution | Needs terminals-runtime's in-process WASM host primitive first; v0.9 scope doesn't include browser targets. |
| Audio entrainment O-RES / MachineIntuition | Tangential to training/eval — fits terminals.tech app layer better than carl-studio's model layer. |
| WhiteHoleNeuron four-view atom | Blocked on HVM grounding (needs Bend combinator semantics). |
| Fano plane witness machinery | Research-adjacent; lands as methods paper first, then code. |
| Impedance router full integration | Requires provider abstraction refactor; separate design session. |

## Ship sequencing (v0.9 as a meta-release)

- **v0.9.0-preview** (this session's delivery): three design docs
  (`carl_update`, `carl_env`, this matrix) landed.
- **v0.9.0-alpha** (next session): implement `carl-update` + `carl-env`
  per their design docs. Ships as v0.9.0 if Tej signs off on the κ
  question and the v0.9-A primitive picks.
- **v0.9.1** (session after): land the three v0.9-A integrations behind
  admin gate. No MIT-side surface regression.
- **v0.9.2-B** (later): v0.9-B primitives as time permits.
- **v0.10+**: deferred items above.

## Author note

This matrix is a reading of the plan file against carl-studio v0.8.0
HEAD. I have not read any terminals-runtime source in producing this
document — paths cited in the "Source" column are from the plan file,
not verified against the private repo. Tej should cross-check module
locations before implementation begins.
