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
