# carl-core — scoped project memory

## Purpose

The foundational layer of the CARL stack. Pure coherence math + interaction trace primitive. Every CARL metric is a reduction of the per-token coherence field defined here. Zero network calls, zero training deps.

## Public API

- `CoherenceTrace`, `select_traces` (from `coherence_trace`)
- `CoherenceProbe`, `CoherenceSnapshot` (from `coherence_probe`)
- `CoherenceObserver` (from `coherence_observer`, anthropic is an optional, lazy import)
- `FrameBuffer`, `FrameRecord` (from `frame_buffer`)
- `compute_phi` (from `math`)
- `KAPPA`, `SIGMA`, `DEFECT_THRESHOLD`, `T_STAR` (from `constants`)
- `InteractionChain`, `Step`, `ActionType` (from `interaction`)
  - `Step.probe_call: dict | None` (v0.11) — fingerprint of the coherence
    probe invocation that populated phi/kuramoto_r/channel_coherence.
    12-hex sha256 digests, not full payloads.
  - `InteractionChain.register_coherence_probe(fn)` / `clear_coherence_probe()` —
    opt-in hook invoked at `record()` for `LLM_REPLY` / `TOOL_CALL` /
    `TRAINING_STEP` / `EVAL_PHASE` / `REWARD` when no explicit coherence
    kwargs are passed.
- `PresenceReport`, `compose_presence_report` (from `presence`, v0.11)
- `success_rate_probe` (from `presence`, v0.11) — default endogenous probe;
  estimates `kuramoto_r` from chain's tail success rate per action type.
- `BreakAndRetryStrategy`, `CircuitOpenError` (from `resilience`, v0.8) —
  composes `RetryPolicy` + `CircuitBreaker`. Used by x402 facilitator calls
  and `carl update`'s PyPI fetch.
- `CARLError` + subclasses (from `errors`): `ConfigError`, `ValidationError`,
  `CredentialError`, `NetworkError`, `BudgetError`, `PermissionError`,
  `CARLTimeoutError`. All carry stable `code` under `carl.<namespace>`.
- `canonical_json`, `content_hash`, `content_hash_bytes` (from `hashing`)
- `CircuitBreaker`, `RetryPolicy`, `retry`, `async_retry`, `poll` (from `retry`)
- `PathEscape`, `SandboxedPath`, `safe_resolve`, `within` (from `safepath`)
- `Tier`, `FEATURE_TIERS`, `TierGateError`, `feature_tier`, `tier_allows` (from `tier`)
- `MemoryItem`, `MemoryLayer`, `MemoryStore` (from `memory`)
- `InteractionStore` (from `interaction_store`)

## Dependencies

- **Required:** `numpy`, `pydantic>=2.9`
- **Optional (lazy-imported):** `anthropic` (for `CoherenceObserver`), `terminals_runtime` (for private math helpers — graceful fallback)

## Depends on us

- `carl_studio.training.*` — rewards, probes, trace callback
- `carl_studio.eval.runner` — evaluation gating
- `carl_studio.observe.*` — training dashboard

## Conventions

- `from __future__ import annotations` at top of every file.
- Modern type syntax (`list[X]`, `dict[K, V]`, `X | None`).
- Pydantic v2 models for configs and serialized data.
- No heavy imports at module level — keep import time fast.
- Constants are module-level, derived from conservation law (NOT tuning parameters).

## Do NOT

- Do NOT add torch, transformers, trl, or any training deps here.
- Do NOT add HTTP clients or API wrappers other than the existing lazy `anthropic` import in `coherence_observer`.
- Do NOT import from `carl_studio.*` — carl-core is the foundation; it depends on nothing upstream.
- Do NOT rename constants (`KAPPA`, `SIGMA`) — these are mathematical constants from the conservation law, with published DOIs.
