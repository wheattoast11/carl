---
last_updated: 2026-05-08
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.10.0
---

# Resonance boundary — v0.10 reaffirmed

**Status:** boundary holds. No `resonance` code changes were required by the v0.10 carl.camp parity completion.

## Verification (2026-05-08)

`/Users/terminals/Documents/agents/models/resonance/src/resonance/__init__.py` is two lines: a docstring identifying the package as "Private training extension for terminals OS coupling, Proprietary to Intuition Labs LLC" and `__version__ = "0.1.0"`. Nothing else.

A grep across the entire `src/resonance/` tree for the patterns most likely to indicate HTTP / auth / SaaS contamination — `carl.tier`, `carl.gating`, `carl_studio.entitlements`, `requests`, `httpx`, `supabase` — returns zero hits. The package remains a pure torch autograd wrapper around `carl_core.eml.EMLTree` plus its sibling math kernels (`signals`, `geometry`, `rewards`, `ttt`, `deployment`, `eml`). It contains no HTTP client, no JWT verification path, no Supabase touch, no carl.camp coupling of any kind.

## Why this matters

The four pillars added in v0.10 (signed remote entitlements, AXON event ingestion, managed slime training submit, constitutional ledger forward) all live ABOVE the resonance layer. They consume EMLTree-shaped artifacts where relevant — the slime path's `SlimeRolloutBridge.finalize_resonant` snapshots a trained tree into a `Resonant`, the resonant publish protocol carries the bytes, the constitutional ledger blocks reference policy hashes — but none of those paths inject HTTP / JWT / supabase plumbing into the resonance math kernel. The mental model in CLAUDE.md ("Carl moves refs, not values" + "the four pillars consume EMLTree-shaped artifacts but never inject HTTP into the math kernel") survives intact.

## License posture (unchanged)

- `resonance/`: BUSL-1.1, change-date 2030-04-09 — same boundary as `terminals-runtime`.
- `carl-studio/`: MIT, public — does not import `resonance` directly; private features go through `admin.py` + lazy import + paid-tier gate.
- The v0.10 work added zero new pathways from MIT carl-studio into BUSL resonance.

## Re-review trigger

Re-check this boundary on each minor release of carl-studio (v0.11, v0.12, …). If the grep above ever returns a hit, the boundary has been violated and the violation needs explicit review before merge. The carl-studio CLAUDE.md "Keep an eye on" section already names this as a watch item.

## References

- `/Users/terminals/.claude/plans/put-together-a-plan-sleepy-crystal.md` — v0.10 implementation plan, Phase J-S6a
- `docs/v10_remote_entitlements_spec.md` — the entitlements path that gates `resonance` consumption (paid tier required for the slime + finalize_resonant flow)
- `docs/eml_signing_protocol.md` — the resonant publish wire shape (carries trained-tree bytes; never carries network state)
