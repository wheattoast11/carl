# CARL Methods Series — Index

The CARL paper series is a flat directory of markdown drafts under
`paper/` at the repository root. The main paper (01) is the canonical
publication and aligns with the shipped `v0.7.1` codebase. Papers
02-04 are follow-ups that formalize shipped subsystems. Additional
upstream work — the four Zenodo papers cited throughout — provides
the mathematical foundation for the CARL conservation law and
coherence / constructiveness identity.

## In-repo papers

| # | File | Title | Status | Primary shipped-code cross-reference |
|---|------|-------|--------|--------------------------------------|
| 01 | `paper/01-main-carl.md` | *Coherence-Aware Reinforcement Learning* | draft | `src/carl_studio/training/rewards/`, `packages/carl-core/src/carl_core/constants.py` |
| 02 | `paper/02-phase-adaptive-methods.md` | *Phase-Adaptive Coherence Rewards* | draft | `src/carl_studio/training/rewards/composite.py:158-330`, `docs/phase_thresholds.md` |
| 03 | `paper/03-coherence-trap-technical-note.md` | *The Coherence Trap* (technical note) | draft | `src/carl_studio/training/rewards/cloud.py`, `paper/01-main-carl.md` Section 5.2.3 |
| 04 | `paper/04-interaction-chains-witness-logs.md` | *Interaction Chains as Witness Logs* | draft | `packages/carl-core/src/carl_core/interaction.py`, consumers in `chat_agent.py`, `eval/runner.py`, `training/`, `x402.py`, `mcp/` |

## Cited external work (Zenodo)

| Title | Role in series | DOI |
|-------|---------------|-----|
| *Bounded Informational Time Crystals* (Desai, 2026) | Source of the conservation law `T* = kappa * d`, `kappa = 64/3`, `sigma = 3/16`. Cited by all four in-repo papers. | [10.5281/zenodo.18906944](https://doi.org/10.5281/zenodo.18906944) |
| *Semantic Realizability: The Convergence-Realizability Identity* (Desai, 2026) | Formal proof of the coherence / constructiveness identity. Ground for CARL's use of phi as a training signal. | [10.5281/zenodo.18992031](https://doi.org/10.5281/zenodo.18992031) |
| *Material Reality: Empirical Validation of Coherence Gating* (Desai, 2026) | 6,244-trial empirical validation of coherence gating across four topologies and two population sizes. | [10.5281/zenodo.18992029](https://doi.org/10.5281/zenodo.18992029) |
| *The IRE Paradigm: Interactive Research Environments* (Desai, 2026) | Methodological framework for coherence-gated research. | Zenodo (see main paper references) |

The Zenodo papers are the source of the mathematical constants that
appear verbatim in `packages/carl-core/src/carl_core/constants.py`.
We do not mirror them in-repo; the DOIs are stable and the papers
stand alone.

## Author note

The in-repo papers are drafts, not peer-reviewed publications. They
are written to ship with the code they describe: every technical
claim in papers 02, 03, and 04 resolves to an import in the v0.7.1
codebase. This is deliberate. A paper that cites code that does not
exist, or code that no longer matches the paper, is a bug. The series
directory is the versioned provenance of CARL's methodological
surface.

For anyone reading this file at a commit other than `v0.7.1`:
cross-references by file path remain meaningful even if line numbers
drift. The named symbols (`PhaseAdaptiveCARLReward`,
`cloud_quality_reward`, `InteractionChain`, `ActionType`,
`PHASE_WEIGHTS_GASEOUS`, `KURAMOTO_R_GASEOUS_MAX`, etc.) are stable
exports of the public API and are the authoritative cross-references.

— Tej Desai, April 2026
