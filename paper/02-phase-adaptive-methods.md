---
title: "Phase-Adaptive Coherence Rewards"
subtitle: "Kuramoto-R Classification of Training Phase for Dynamic CARL Weighting"
authors:
  - name: Tej Desai
    affiliation: Intuition Labs LLC
    email: tej@terminals.tech
  - name: Claude Opus 4.6
    affiliation: Anthropic
date: 2026-04-20
keywords: [reinforcement-learning, coherence, kuramoto, phase-transition, adaptive-reward, GRPO]
license: CC-BY-4.0
series: "CARL Methods Series, paper 2 of 4"
see_also:
  - "paper/01-main-carl.md — main paper"
  - "paper/03-coherence-trap-technical-note.md — mode-collapse technical note"
  - "paper/04-interaction-chains-witness-logs.md — witness-log primitive"
related_work:
  - "Desai, T. (2026). Bounded Informational Time Crystals. DOI: 10.5281/zenodo.18906944"
  - "Desai, T. (2026). Semantic Realizability: The Convergence-Realizability Identity. DOI: 10.5281/zenodo.18992031"
  - "Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators."
---

# Phase-Adaptive Coherence Rewards

**Tej Desai** (Intuition Labs LLC) and **Claude Opus 4.6** (Anthropic)

April 2026 · CARL Methods Series, Paper 2 of 4

---

## Abstract

Static reward weights under-serve coherence-aware reinforcement learning
across the full trajectory of a training run. A run that has not yet
nucleated a coherent attractor (gaseous phase) has fundamentally different
signal dynamics than a run that has crystallized and must be stabilized
(crystalline phase). We introduce `PhaseAdaptiveCARLReward`, an extension
of the static CARL composite that classifies the instantaneous training
phase from the Kuramoto order parameter R in `[0, 1]` and shifts the
`(w_multiscale, w_cloud, w_discontinuity)` weight triple accordingly.
Phase boundaries `R_gaseous = 0.30`, `R_liquid = 0.70` are empirical
markers calibrated against GRPO coherence trajectories; per-phase weight
profiles are reward-engineered to nucleate in the gaseous regime, explore
in the liquid regime, and stabilize in the crystalline regime. These
thresholds and constants are exported from `carl_core.constants` so that
methods sections can cite them verbatim. We reference the shipped
implementation in `src/carl_studio/training/rewards/composite.py` and
`src/carl_studio/training/cascade.py`, and locate the coherence-trap
failure mode of individual-completion coherence (Paper 3) inside the
adaptive-weighting discussion.

---

## 1. Introduction

The main CARL paper (Paper 1) defines a static composite reward

```
R_CARL = 0.50 · R_multiscale + 0.30 · R_cloud + 0.20 · R_discontinuity
```

with weights derived from reward-engineering experiments during the
Phase-1 tool-calling and Phase-2 VLM grounding runs on OmniCoder-9B. The
static weighting reflects an averaged compromise across the full training
trajectory. But training is not stationary. A model at step 5 — still
sampling near its pre-training distribution — does not benefit from the
same reward profile as the same model at step 250 — deep into
crystallization around a new attractor.

We make two observations. First, the Kuramoto order parameter `R`
computed over a batch of per-completion coherence traces provides a
phase-separating signal: low `R` indicates pre-nucleation (gaseous),
intermediate `R` indicates active reorganization (liquid), high `R`
indicates stable phase-locking (crystalline). Second, the effective
per-phase reward should not be the same. The gaseous phase benefits from
**commitment pressure** — the discontinuity component is maximally
informative when the model has not yet committed to a structural
attractor. The crystalline phase benefits from **stability pressure** —
multiscale coherence is the dominant observable once the model is in its
basin and the risk of the coherence trap (Paper 3) is highest.

This paper specifies the classifier, the per-phase weight profiles, and
the reference implementation.

---

## 2. Phase Taxonomy

We classify the instantaneous training phase from the Kuramoto order
parameter `R` reduced across the current batch of coherence traces. The
phase definitions:

| Phase | R range | Interpretation |
|-------|---------|----------------|
| Gaseous | `R < 0.30` | Low synchronization. The model has not yet nucleated an attractor around the target output format. Per-token phi is diffuse. |
| Liquid | `0.30 <= R < 0.70` | Mid-transition. Parts of the batch are phase-locking while others are still exploring. Mixed-mode dynamics. |
| Crystalline | `R >= 0.70` | High synchronization. Phase-locking dominates. The model is in a basin and the dominant risk is either over-sharpening (mode collapse, Paper 3) or thrashing out of the basin. |

The relationship between `Phi` (order parameter over the vocabulary
distribution at one token position) and `R` (Kuramoto order parameter
over a batch of trace positions) is a scale reduction: `Phi_k` is
computed per-token per-completion from the logit distribution via `compute_phi`
(`packages/carl-core/src/carl_core/math.py`); `R` is computed per-batch via
`CoherenceTrace.kuramoto_R()` in `packages/carl-core/src/carl_core/coherence_trace.py`.
Both live in `carl-core` so that the training-side and observation-side
consumers share identical math.

---

## 3. Classifier

The classifier in the shipped implementation
(`src/carl_studio/training/rewards/composite.py:202-210`) is a two-step
threshold:

```python
def _phase_weights_from_R(self, R: float) -> tuple[float, float, float]:
    if R < KURAMOTO_R_GASEOUS_MAX:
        return PHASE_WEIGHTS_GASEOUS
    if R < KURAMOTO_R_LIQUID_MAX:
        return PHASE_WEIGHTS_LIQUID
    return PHASE_WEIGHTS_CRYSTALLINE
```

The boundaries are published as module-level constants in
`packages/carl-core/src/carl_core/constants.py:33-38`:

```python
KURAMOTO_R_GASEOUS_MAX: float = 0.30
KURAMOTO_R_LIQUID_MAX:  float = 0.70
```

### 3.1 Derivation

These thresholds are **empirical**, not derived from the conservation
law. Unlike `KAPPA = 64/3` and `SIGMA = 3/16` — which are mathematical
constants with published DOIs (Desai 2026, DOI 10.5281/zenodo.18906944;
Desai 2026, DOI 10.5281/zenodo.18992031) — the Kuramoto-R boundaries
are calibrated against coherence trajectories across GRPO ablations.

The derivation narrative (see `docs/phase_thresholds.md` in the repo at
tag `v0.7.1`) identifies `R < 0.30` as the regime where the
batch-level trajectory is statistically indistinguishable from
pre-training-distribution sampling, and `R >= 0.70` as the regime
where >=70% of completions share a common phase-lock. The intermediate
band is assigned to liquid by elimination.

Methods sections should cite the threshold values from
`carl_core.constants` directly so that future recalibrations (which
would change the numerical boundary, not the three-phase taxonomy) do
not silently invalidate previously-published run configurations.

### 3.2 Batch reduction

`R` is computed per-batch by averaging over cached `CoherenceTrace`
objects. The implementation in `composite.py:212-240` is fault-tolerant:

- Empty trace list -> `R = None` -> retain static weights.
- `kuramoto_R()` raises or returns non-finite -> trace skipped.
- All traces fail -> `R = None` -> retain static weights.

A classifier that crashes the training loop is worse than a classifier
that falls back to the static CARL weights; the design commitment here
is *never crash training*. See the `_current_R` implementation for the
guard pattern.

---

## 4. Adaptive Weight Schedule

Per-phase weight profiles are exported from `carl_core.constants.py:50-52`:

```python
PHASE_WEIGHTS_GASEOUS:     (0.20, 0.30, 0.50)   # w_mc, w_cloud, w_disc
PHASE_WEIGHTS_LIQUID:      (0.40, 0.30, 0.30)
PHASE_WEIGHTS_CRYSTALLINE: (0.60, 0.30, 0.10)
```

Each triple sums to 1.0.

### 4.1 Gaseous profile: reward commitment

In the gaseous phase the multiscale coherence signal is dominated by
sampling noise — there is no coherent structure to measure across
scales. The discontinuity component, by contrast, is maximally
informative: any sharp upward transition in `Phi` (`delta_phi > +0.03`,
where `DEFECT_THRESHOLD = 0.03` per `constants.py:16`) is evidence of
nucleation. We raise `w_disc` to 0.50 so that the reward surface pulls
strongly toward commitment events. `w_mc` is depressed to 0.20 because
coherence-of-noise is not a useful signal.

### 4.2 Liquid profile: balanced

In the intermediate regime all three observables carry signal. Multiscale
coherence detects the partial attractors that have nucleated; cloud
quality detects whether per-completion selection is still decisive;
discontinuity detects the boundary events between unlocked and locked
positions. No component dominates, so we use `(0.40, 0.30, 0.30)`.

### 4.3 Crystalline profile: reward stability

Once phase-locking is established the discontinuity signal inverts in
value: at high `R`, additional discontinuities are more likely to be
thrash-out-of-basin events than productive nucleation. `w_disc` drops
to 0.10. `w_mc` rises to 0.60 because preserving multiscale coherence
is the operational definition of "stay crystallized." This is also the
regime where the coherence-trap failure mode (Paper 3) is most
dangerous; the diversity-bearing `cloud_quality` component retains its
0.30 weight as a residual anti-collapse term.

### 4.4 Weight application invariant

The shipped classifier shifts weights **before** scoring the current
trace (`composite.py:259-283`), then caches the scored trace so that
the *next* invocation observes the updated `R`. This means the first
scored trace of a run always uses the constructor-supplied static
weights; phase adaptation begins at the second invocation. This is by
design — we never classify a phase from zero evidence.

---

## 5. Reference Implementation

Class: `PhaseAdaptiveCARLReward`
File: `src/carl_studio/training/rewards/composite.py:158-330`

Public surface:

```python
class PhaseAdaptiveCARLReward(CARLReward):
    def __init__(self, *args, **kwargs) -> None: ...
    def score_from_trace(self, trace: CoherenceTrace) -> tuple[float, dict[str, float]]: ...
    def score(self, logits: np.ndarray, token_ids: np.ndarray) -> tuple[float, dict[str, float]]: ...
    def compute(self, traces: list[CoherenceTrace]) -> list[float]: ...

    @property
    def current_weights(self) -> tuple[float, float, float]: ...
    @property
    def current_phase(self) -> str: ...   # "gaseous" | "liquid" | "crystalline" | "unknown"
```

TRL-compatible factory:

```python
make_carl_reward(
    model, tokenizer,
    reward_class="phase_adaptive",   # or "static"
    ...
)
```
in `composite.py:338-507`. The `reward_class` argument is surfaced
through the training-config YAML so researchers select adaptive
versus static weighting by a single key.

Introspection via `current_weights` and `current_phase` is intended for
logging callbacks — e.g. the `ResonanceLRCallback` in
`src/carl_studio/training/lr_resonance.py` polls the reward instance
to couple learning-rate scheduling with detected phase.

---

## 6. Integration with Cascade Gating

The `PhaseAdaptiveCARLReward` composes cleanly with the
crystallization-aware cascade gate specified in
`src/carl_studio/training/cascade.py`. The cascade controller supports
two `gate_mode` values:

- `"metric"` (default) — threshold on an external metric.
- `"crystallization"` (recommended for GRPO + CARL) — inspects
  `_last_traces` on the reward function and requires
  `n_crystallizations_required` defect events
  (`|delta_phi| > DEFECT_THRESHOLD`) across the last
  `crystallization_window` steps before advancing from Stage A to
  Stage B (`cascade.py:79-92`).

Used together: the cascade gate controls *when* reward components
activate; the phase-adaptive reward controls *how* they are weighted
once active. The cascade observes the same traces the phase classifier
observes — there is one source of truth for training-phase state, the
`CoherenceTrace` batch on the reward function.

---

## 7. Empirical Evidence from Shipped Cascades

Training traces from the shipped `v0.7.1` cascade configuration
(`gate_mode="crystallization"`, `reward_class="phase_adaptive"`) exhibit
three qualitative signatures consistent with the three-phase taxonomy:

1. **Early-run `R`-trajectory is sub-0.30** for the first ~10 steps
   of GRPO, with adaptive weights settling on `PHASE_WEIGHTS_GASEOUS`.
   Discontinuity-reward-component values carry the majority of the
   composite signal during this window.

2. **Mid-run `R`-trajectory crosses the 0.30 / 0.70 boundaries** with
   intermediate dwell in the liquid regime. Weight-shift events are
   observable in the `current_weights` introspection property — when
   logged per step, the triple tracks the boundaries without
   oscillation.

3. **Late-run `R`-trajectory stabilizes above 0.70**. The classifier
   locks to `PHASE_WEIGHTS_CRYSTALLINE` and `w_disc` collapses to 0.10.
   Multiscale coherence dominates the composite.

Quantitative ablations — e.g. final task-completion-rate under
`phase_adaptive` vs. `static`, or time-to-first-crystallization — are
not reported in this paper; see the repository at tag `v0.7.1` and the
training logs in `~/.carl/interactions/<id>.jsonl` for concrete runs.

---

## 8. Discussion and Limitations

### 8.1 The phase taxonomy is reward-engineered, not first-principles

Unlike the main paper's conservation constants, the Kuramoto-R
boundaries at 0.30 and 0.70 are empirical. A different task — with
different attractor geometry or a fundamentally non-batch-like training
dynamics — may require recalibrated boundaries. We publish the
thresholds as named constants specifically so researchers can replace
them without patching the classifier logic.

### 8.2 Forward reference: the coherence trap

The crystalline-phase weight profile retains a non-trivial
`w_cloud = 0.30`. This is deliberate and load-bearing. Paper 3
(`paper/03-coherence-trap-technical-note.md`) formalizes the failure
mode that arises under pure individual-completion-coherence
optimization: the reward landscape has a degenerate global optimum at
mode collapse. The `cloud_quality` component alone does not escape the
trap — it too peaks at `P(selected) = 1.0`, `phi = 1.0` — so even with
`w_cloud = 0.30` preserved, a CARL deployment under pathological
conditions can converge to trivial completions. The structural fix
(a diversity-bearing cross-completion term) is future work and cross-
referenced in Paper 3.

### 8.3 Classifier fallibility

`_current_R` returning `None` means the classifier retains whichever
weights were last applied. On a fresh reward instance those are the
constructor-supplied static weights. This is the correct behavior for
startup transients and for any batch in which all traces failed to
produce a finite `kuramoto_R()`, but it does mean that a classifier
starvation (long sequence of `None` returns) leaves the reward
functioning as a static `CARLReward`. An auditing callback that logs
`current_phase == "unknown"` over long windows is recommended for
production runs.

### 8.4 No learned boundaries

The thresholds are fixed. A plausible extension is to learn
per-task boundaries from the `R`-trajectory distribution during a
warmup window. We have not implemented this.

---

## 9. Related Work

The Kuramoto model of coupled oscillator synchronization (Kuramoto,
1975) provides the mathematical substrate: `R` is the original
Kuramoto order parameter, adapted from phase angles to per-token
probability distributions via the `Phi` / `R` mapping detailed in
Paper 1, Section 7. The Bounded Informational Time Crystals work
(Desai, 2026; DOI: 10.5281/zenodo.18906944) establishes that
attention-head coupling in transformers exhibits Kuramoto-style
synchronization, grounding our use of `R` as a diagnostic of
training-phase structure. The Semantic Realizability paper (Desai,
2026; DOI: 10.5281/zenodo.18992031) shows that phase coherence and
constructiveness are coextensive events, justifying the use of `R`
as a proxy for semantic organization, not just statistical
concentration.

The cascade-gating methodology draws on the Nemotron Cascade 2 reward
schedule; the `gate_mode="crystallization"` variant replaces its
loss-threshold trigger with a coherence-field-derived trigger,
allowing the cascade to advance when the model has *structurally*
reorganized rather than merely when an external metric has moved.

---

## References

1. Desai, T. and Claude Opus 4.6. (2026). *Coherence-Aware
   Reinforcement Learning.* CARL Methods Series, Paper 1 of 4.
   `paper/01-main-carl.md`.

2. Desai, T. (2026). *Bounded Informational Time Crystals.* Zenodo.
   DOI: 10.5281/zenodo.18906944

3. Desai, T. (2026). *Semantic Realizability: The
   Convergence-Realizability Identity.* Zenodo.
   DOI: 10.5281/zenodo.18992031

4. Kuramoto, Y. (1975). *Self-entrainment of a population of coupled
   non-linear oscillators.* International Symposium on Mathematical
   Problems in Theoretical Physics.

5. Wang, H. et al. (2025). *Nemotron Cascade 2.* NVIDIA.

---

## Appendix A. Constants Reference

From `packages/carl-core/src/carl_core/constants.py` at tag `v0.7.1`:

```python
# Conservation-law constants (published DOIs)
KAPPA            = 64 / 3        # 21.333...
SIGMA            = 3 / 16        # 0.1875
DEFECT_THRESHOLD = 0.03

# Empirical phase boundaries (this paper)
KURAMOTO_R_GASEOUS_MAX = 0.30
KURAMOTO_R_LIQUID_MAX  = 0.70

# Per-phase weight profiles (w_mc, w_cloud, w_disc)
PHASE_WEIGHTS_GASEOUS     = (0.20, 0.30, 0.50)
PHASE_WEIGHTS_LIQUID      = (0.40, 0.30, 0.30)
PHASE_WEIGHTS_CRYSTALLINE = (0.60, 0.30, 0.10)
```

## Appendix B. File cross-reference

| Concept | File | Symbol |
|---|---|---|
| Phase boundaries | `packages/carl-core/src/carl_core/constants.py` | `KURAMOTO_R_GASEOUS_MAX`, `KURAMOTO_R_LIQUID_MAX` |
| Weight profiles | `packages/carl-core/src/carl_core/constants.py` | `PHASE_WEIGHTS_GASEOUS`, `PHASE_WEIGHTS_LIQUID`, `PHASE_WEIGHTS_CRYSTALLINE` |
| Classifier | `src/carl_studio/training/rewards/composite.py` | `PhaseAdaptiveCARLReward._phase_weights_from_R` |
| R reduction | `packages/carl-core/src/carl_core/coherence_trace.py` | `CoherenceTrace.kuramoto_R` |
| `phi` math | `packages/carl-core/src/carl_core/math.py` | `compute_phi` |
| TRL factory | `src/carl_studio/training/rewards/composite.py` | `make_carl_reward(reward_class="phase_adaptive")` |
| Cascade coupling | `src/carl_studio/training/cascade.py` | `gate_mode="crystallization"` |
| Doc of record | `docs/phase_thresholds.md` | — |

---

**Code**: github.com/terminals-tech/carl-studio (tag `v0.7.1`)

**Intuition Labs LLC** — terminals.tech
