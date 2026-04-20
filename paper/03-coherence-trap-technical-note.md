---
title: "The Coherence Trap"
subtitle: "Mode Collapse Under Individual-Completion Coherence Optimization — A Technical Note"
authors:
  - name: Tej Desai
    affiliation: Intuition Labs LLC
    email: tej@terminals.tech
  - name: Claude Opus 4.6
    affiliation: Anthropic
date: 2026-04-20
keywords: [reinforcement-learning, coherence, mode-collapse, GRPO, reward-hacking, diversity]
license: CC-BY-4.0
series: "CARL Methods Series, paper 3 of 4"
see_also:
  - "paper/01-main-carl.md — main paper (coherence-trap source observation, Section 5.2.3)"
  - "paper/02-phase-adaptive-methods.md — phase-adaptive weighting"
  - "paper/04-interaction-chains-witness-logs.md — witness-log primitive"
related_work:
  - "Desai, T. (2026). Bounded Informational Time Crystals. DOI: 10.5281/zenodo.18906944"
  - "Desai, T. (2026). Semantic Realizability: The Convergence-Realizability Identity. DOI: 10.5281/zenodo.18992031"
---

# The Coherence Trap

**Tej Desai** (Intuition Labs LLC) and **Claude Opus 4.6** (Anthropic)

April 2026 · CARL Methods Series, Paper 3 of 4 · Technical Note

---

## Abstract

Individual-completion coherence — the per-sample CARL composite
evaluated in isolation — is a necessary but not sufficient signal for
RL training of language models. We formalize the **coherence trap**:
a structural limitation where a model exhibiting mode collapse receives
near-optimal CARL composite scores because the reward landscape has a
degenerate global optimum at the delta distribution. The effect was
first observed in Phase-2 VLM grounding experiments (main paper,
Section 5.2.3): text-only GRPO produced identical 492,544 coordinate
outputs across 68% of prompts, with `Phi ~ 0.9998` and CARL composite
`~ 0.55` — higher than many partially-correct runs. We restate the
pathology in standalone form, explain the mechanism (maximum coherence
IS a delta function), and document the diversity-bearing
`CloudQualityReward` and cross-completion diversity terms that a CARL
deployment must compose with individual coherence to avoid the trap.
This note is diagnostic, not prescriptive: it catalogs the failure
signature and identifies the load-bearing role of diversity in any
CARL-based training configuration.

---

## 1. The Pathology

Consider a GRPO training run with `num_generations = 8`. At GRPO step
`N` the model produces 8 completions per prompt. Ground-truth task
reward (e.g. bounding-box hit test for click accuracy) is `0.0` or
`1.0` per completion. The CARL composite reward is evaluated per
completion against the completion's own logit distribution.

A well-functioning run shows:

- Completion-level diversity — 8 distinct outputs exploring different
  hypotheses for the prompt.
- Mixed task rewards — some hits, some misses, non-zero variance.
- Mixed CARL composites — some completions exhibit cleaner phase
  structure than others; the composite reflects this.

The **coherence trap** run shows a degenerate variant:

- 8 completions that are token-for-token identical (or near-identical
  with trivial whitespace perturbation).
- Task reward variance == 0. GRPO's group-relative advantage term
  collapses.
- CARL composite ~ 0.55, **uniform across all 8 completions**.

The run exhibits no training signal of any kind — but the CARL metric
panel reports a moderately high number. A metrics-only observer would
conclude the run is healthy.

This was the dominant failure mode of the Phase-2 VLM-grounding
text-only-GRPO iterations v1 through v3 (main paper, Section 5.2.1):

| Run | Click accuracy | Zero-gradient steps | CARL composite |
|-----|----------------|---------------------|----------------|
| v1  | 8%             | --                  | ~ 0.55         |
| v2  | 8%             | 41%                 | ~ 0.55         |
| v3  | 8%             | 41%                 | ~ 0.55         |

68% of prompts produced identical `(492, 544)` coordinate outputs
across all 8 GRPO generations. This is the canonical coherence-trap
signature.

---

## 2. Mechanism

The CARL composite is

```
R_CARL = w_mc · R_multiscale + w_cq · R_cloud + w_disc · R_discontinuity
```

We analyze each term under the mode-collapse boundary condition —
the model emits the same tokens, with `P(selected_token) ~ 1.0`, at
every position of every completion.

### 2.1 Multiscale coherence

The multiscale coherence reward measures consistency of the `Phi`
trajectory across dyadic scales (main paper, Section 3.2). With
`P(selected) ~ 1.0` the per-token `Phi_k = 1 - H(P_k)/log|V| ~ 1.0`
everywhere. A constant `Phi` trajectory has zero within-block
variance at every scale, so every `C_j` saturates at 1.0 and
`R_multiscale ~ 1.0`.

> Maximum multiscale coherence IS a constant-one trajectory, which
> IS a mode-collapsed generator.

### 2.2 Cloud quality

The cloud-quality reward is

```
R_cloud = mean_k( P_k(t_k) · Phi_k )
```

(see `src/carl_studio/training/rewards/cloud.py:13-38`).
Under mode collapse, `P_k(t_k) ~ 1.0` and `Phi_k ~ 1.0`, so the
per-token product `~ 1.0` and `R_cloud ~ 1.0`.

> Cloud quality *individually measured* does not resist mode collapse —
> it rewards it.

### 2.3 Discontinuity

Discontinuities are transitions where `|delta_phi_k| > 0.03`
(main paper, Section 3.4; `DEFECT_THRESHOLD` in
`packages/carl-core/src/carl_core/constants.py:16`). A mode-collapsed
trajectory has `Phi ~ 1.0` everywhere with zero variance — no
discontinuities detected. The reward falls back to the neutral default
`0.5` (main paper, Section 3.4 last paragraph).

### 2.4 Composite

Under the default static weighting `(0.5, 0.3, 0.2)`:

```
R_CARL = 0.5 · 1.0 + 0.3 · 1.0 + 0.2 · 0.5 = 0.9
```

The observed value of ~0.55 in v1–v3 is lower because mode collapse
in practice is not literally `P = 1.0` — it is `P ~ 0.9998` with
trivial probability mass distributed over adjacent tokens, and the
GRPO-group-mean normalization shifts the observed composite downward.
The analytical ceiling at 0.9 is nonetheless the attractor: the
CARL reward landscape has a degenerate global optimum at mode
collapse.

### 2.5 Structural, not implementation, limitation

This is not a bug. The definition of `Phi_k = 1 - H(P_k)/log|V|` is
principled; it is the per-token analog of the Kuramoto order
parameter. The compromise-weighted multiscale aggregation is
principled; it is the geometric mean of fine- and coarse-scale
coherence. What is missing from individual-completion scoring is any
mechanism that **penalizes the model for producing the same output
twice**. Maximum coherence across the vocabulary distribution is
indistinguishable from zero exploration across the completion
population.

---

## 3. The Diversity Fix

The structural fix is a **cross-completion diversity term** evaluated
over a batch of GRPO generations, not over individual completions.

### 3.1 The diversity term

A candidate formulation (main paper, Section 5.2.3):

```
R_diversity = 1 - mean_{i<j}( cosine_sim( completion_i, completion_j ) )
```

Under identical completions: all pairwise similarities `= 1.0`,
`R_diversity = 0`. Under maximally-diverse completions: pairwise
similarities `~ 0`, `R_diversity ~ 1`. The term is a direct inverse
of the mode-collapse signature.

A production implementation must specify the embedding space,
similarity measure, and weight schedule. These choices are not
load-bearing for the analysis in this note; the load-bearing claim is
that **some** term whose gradient flows across the completion
population is required.

### 3.2 The role of `CloudQualityReward` as partial mitigation

We observe that `cloud_quality` — while collapse-vulnerable in its
individual form (Section 2.2) — **does** carry partial diversity
signal when compared across completions. Two distinct completions
with `P(selected) = 0.9` and `P(selected) = 0.7` produce different
cloud scores, while two identical completions produce identical cloud
scores. This is weaker than the cross-completion diversity term
above — cloud quality measures distribution concentration within a
completion, not separation between completions — but it is not
nothing. Retaining `w_cloud > 0` in every phase of the
`PhaseAdaptiveCARLReward` (Paper 2, Section 4) is a deliberate
anti-collapse residual.

The reference implementation:
`src/carl_studio/training/rewards/cloud.py:13-38`
(`cloud_quality_reward`).

### 3.3 Other candidates

Any reward term that:

1. Takes as input the full batch of completions, not a single
   completion in isolation;
2. Is monotone-decreasing in cross-completion similarity;
3. Has non-trivial gradient at the mode-collapse boundary;

is a valid structural response. Examples include:

- Pairwise edit-distance over token sequences.
- KL-divergence between empirical completion distributions and a
  uniform-over-completions prior.
- Group entropy of the completion-length distribution.
- Vocabulary coverage across the completion batch.

The specific choice is empirical. The point of this note is that
**no** CARL deployment should ship without some such term.

---

## 4. Empirical Signatures

A CARL training operator should monitor for the following signatures
in the interaction log (`~/.carl/interactions/<id>.jsonl`, see Paper 4
for the witness-log primitive):

| Signature | Indicates |
|-----------|-----------|
| Task-reward std across a GRPO group == 0 for many consecutive steps | GRPO advantage collapse; likely mode collapse. |
| CARL composite std across a GRPO group near zero while composite mean is non-trivial | Uniform CARL score across the batch. If completions are identical, this is the trap. |
| `multiscale_coherence` component saturating at 1.0 simultaneously across all completions | Individual-completion coherence is peaked. |
| Mean `Phi` across the batch trending toward 1.0 without a corresponding task-reward rise | Crystallization-into-collapse, not crystallization-onto-task. |
| Kuramoto `R` pinned at 1.0 for many steps with no task improvement | Phase-locked to a trivial output. |

A single signature in isolation is not diagnostic — a crystalline
run (Paper 2, Section 7) legitimately exhibits elevated `R` and
elevated `Phi`. The trap is identified by the **joint** signature: high
coherence **plus** zero task-reward variance **plus** near-identical
completions.

---

## 5. Recommendation

We restate the recommendation from the main paper (Section 9.2,
"The coherence trap") as a standalone deployment rule:

> **Any CARL-based training configuration must compose the
> individual-completion CARL composite with at least one
> cross-completion diversity term. Individual coherence maximizes at
> the delta distribution, which is the mode-collapse failure mode; a
> reward landscape with no opposing force will find this optimum.**

The phase-adaptive weighting of Paper 2 does not in itself avoid the
trap — it adjusts how individual-completion observables are weighted
against each other, but all three CARL observables peak under mode
collapse. Retaining `w_cloud > 0` at every phase provides partial
protection via within-completion-distribution concentration, but the
structural fix is cross-completion diversity.

This recommendation applies to `reward_class="static"` and
`reward_class="phase_adaptive"` alike.

---

## References

1. Desai, T. and Claude Opus 4.6. (2026). *Coherence-Aware
   Reinforcement Learning.* CARL Methods Series, Paper 1 of 4.
   `paper/01-main-carl.md`. (Source of the Section 5.2.3 observation
   this note extracts.)

2. Desai, T. and Claude Opus 4.6. (2026). *Phase-Adaptive Coherence
   Rewards.* CARL Methods Series, Paper 2 of 4.
   `paper/02-phase-adaptive-methods.md`.

3. Desai, T. (2026). *Bounded Informational Time Crystals.* Zenodo.
   DOI: 10.5281/zenodo.18906944

4. Desai, T. (2026). *Semantic Realizability: The
   Convergence-Realizability Identity.* Zenodo.
   DOI: 10.5281/zenodo.18992031

---

## Appendix. File cross-reference

| Concept | File | Symbol |
|---|---|---|
| Cloud-quality reward (per-completion) | `src/carl_studio/training/rewards/cloud.py` | `cloud_quality_reward` |
| Composite and phase-adaptive reward | `src/carl_studio/training/rewards/composite.py` | `CARLReward`, `PhaseAdaptiveCARLReward` |
| `Phi` math | `packages/carl-core/src/carl_core/math.py` | `compute_phi` |
| `DEFECT_THRESHOLD` = 0.03 | `packages/carl-core/src/carl_core/constants.py` | `DEFECT_THRESHOLD` |
| Main-paper observation | `paper/01-main-carl.md` | Section 5.2.3 |

---

**Code**: github.com/terminals-tech/carl-studio (tag `v0.7.1`)

**Intuition Labs LLC** — terminals.tech
