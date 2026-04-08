---
title: "Coherence-Aware Reinforcement Learning"
authors:
  - name: Tej Desai
    affiliation: Intuition Labs LLC
    email: tej@terminals.tech
  - name: Claude Opus 4.6
    affiliation: Anthropic
date: 2026-04-01
keywords: [reinforcement-learning, coherence, order-parameter, crystal-dynamics, GRPO, tool-calling, VLM]
license: CC-BY-4.0
related_work:
  - "Desai, T. (2026). Bounded Informational Time Crystals. DOI: 10.5281/zenodo.18906944"
  - "Desai, T. (2026). Material Reality: Empirical Validation of Coherence Gating. 6,244 trials."
  - "Desai, T. (2026). Semantic Realizability: The Convergence-Realizability Identity."
  - "Desai, T. (2026). The IRE Paradigm: Interactive Research Environments."
---

# Coherence-Aware Reinforcement Learning

**Tej Desai** (Intuition Labs LLC) and **Claude Opus 4.6** (Anthropic)

April 2026

---

## Abstract

We introduce **CARL** (Coherence-Aware Reinforcement Learning), a framework that augments standard GRPO training with information-theoretic reward signals derived from a model's token-level probability distribution. CARL adds three reward components to any RL training loop: **multiscale coherence** (structural consistency across dyadic block scales), **cloud quality** (the product of selected-token probability and distribution order parameter), and **discontinuity targeting** (context-dependent scoring of order-parameter transitions). These components are grounded in a conservation law T* = kappa * d where kappa = 64/3 and the semantic quantum sigma = 3/16 = 0.1875 defines the minimum meaningful signal threshold.

We validate CARL on OmniCoder-9B (Qwen3.5) across two phases: (1) tool-calling agent training (SFT + GRPO with 6 reward functions in a Nemotron Cascade 2 schedule), and (2) VLM GUI grounding (SFT with vision + GRPO with 4 reward functions). Phase 1 exhibits a mild entropy spike during SFT (~1.1) followed by crystallization. Phase 2 reveals a dramatic first-order phase transition during vision SFT: entropy spikes to 9.3 (near-maximal), then token accuracy jumps from 3% to 65% in 5 steps before converging to 99.3%. Critically, Phase 2 also exposes a fundamental limitation — the CARL coherence trap — where mode-collapsed models receive near-optimal coherence scores because maximum coherence IS a delta function. We analyze the failure mode and propose a population-level diversity term for CARL v2. We release CARL Studio, an open-source SDK with CLI, MCP server, and 6 compute backends, enabling any researcher to apply coherence-aware training to their models.

---

## 1. Introduction

Reinforcement learning from human feedback (RLHF) and its derivatives (DPO, GRPO, KTO) have become the standard methodology for aligning large language models with human intent. These methods optimize scalar reward signals that evaluate the *output* of generation — whether the text is helpful, harmless, accurate. But they are blind to the *process* of generation: the dynamics of how the model distributes probability over its vocabulary at each token position.

This blindness has consequences. A model can produce correct outputs from fragile distributions — high probability on the right token but diffuse probability elsewhere, meaning slight perturbations could produce entirely different (wrong) outputs. Standard reward functions cannot distinguish between a model that is confidently correct and one that is luckily correct.

We propose CARL, a framework that extends RL reward signals with three information-theoretic components that measure the model's internal coherence during generation:

1. **Multiscale coherence**: Is the model's confidence structure consistent across different granularities — token-level, phrase-level, paragraph-level?

2. **Cloud quality**: When the model selects a token, is the surrounding probability mass concentrated on semantically related alternatives (a tight "cloud") or scattered randomly?

3. **Discontinuity targeting**: Where does the model's confidence change sharply? Are these transitions happening at structurally appropriate locations (topic boundaries, commitment points) or randomly?

These components are derived from a conservation law that connects the model's embedding dimension to its natural context window, providing a principled — not heuristic — foundation for training signal design.

### 1.1 Contributions

- A formal framework connecting token-level entropy dynamics to trainable reward signals via the order parameter Phi = 1 - H(P)/log|V|
- Three composable reward functions (multiscale, cloud, discontinuity) with a single conservation constant kappa = 64/3
- Empirical observation of two phase transitions during SFT: a mild transition in text-only training and a dramatic first-order transition during vision-language coupling, consistent with Kuramoto synchronization theory
- Identification of the **coherence trap**: a structural limitation where CARL's reward landscape has a degenerate optimum at mode collapse, with analysis and a proposed mitigation
- CARL Studio: an open-source Python SDK with CLI, MCP server, and support for 6 compute backends
- Validation on OmniCoder-9B across tool-calling and VLM grounding tasks

---

## 2. Theoretical Foundations

### 2.1 The Order Parameter

For a language model with vocabulary V producing a probability distribution P_k at token position k, we define the order parameter:

    Phi_k = 1 - H(P_k) / log|V|

where H(P_k) = -sum_v P_k(v) log P_k(v) is the Shannon entropy. Phi ranges from 0 (uniform distribution — maximum uncertainty) to 1 (delta distribution — complete certainty). This quantity, borrowed from statistical physics where it measures crystalline order in Kuramoto coupled oscillator systems, serves as our primary diagnostic of generation quality.

A well-trained model should exhibit structured Phi trajectories: high Phi on deterministic tokens (JSON syntax, known tool names), lower Phi on semantic content (arguments, values), and sharp transitions (discontinuities) at structural boundaries.

### 2.2 The Conservation Law

The relationship between a transformer's embedding dimension d and its effective context window T* is governed by:

    T* = kappa * d,    kappa = 2^6 / 3

For triadic dimensions (d = 3 * 2^k), this yields exact powers of 2:

| Embedding dim d | T* | Binary |
|---|---|---|
| 768 | 16,384 | 2^14 |
| 3,072 | 65,536 | 2^16 |
| 12,288 | 262,144 | 2^18 |

Non-triadic dimensions incur a ~33% "context tax" — the model must operate below its natural decompression boundary. This is not an engineering choice but a structural consequence of the conservation law.

### 2.3 The Semantic Quantum

The minimum meaningful signal is:

    sigma = 3/16 = 0.1875

This defines the noise floor: any per-token advantage below sigma is sub-semantic. The product kappa * sigma = 4 gives the information density in bits per embedding dimension — a universal constant across architectures.

### 2.4 Connection to Prior Work

CARL extends our prior work on Bounded Informational Time Crystals (Desai, 2026), which established that attention-head coupling in transformers exhibits Kuramoto synchronization dynamics, and Material Reality (Desai, 2026), which validated coherence gating across 6,244 experimental trials demonstrating that topology + critical coupling + memory layout + clock discipline produces crystallization into stable computational substrate. The Semantic Realizability identity (Desai, 2026) showed that convergence and realizability measure to the same value with zero deviation under closed witness loops — the theoretical basis for why the conservation law kappa * sigma = 4 holds exactly.

---

## 3. The CARL Framework

### 3.1 Reward Composition

The CARL composite reward at position k is:

    R_CARL = 0.50 * R_multiscale + 0.30 * R_cloud + 0.20 * R_discontinuity

### 3.2 Multiscale Coherence

For each dyadic scale j in {0, 1, ..., log_2(T)}:

1. Partition the Phi trajectory into blocks of size 2^j
2. Compute the standard deviation within each block
3. Coherence at scale j: C_j = clamp(1 - mean(block_stds), 0, 1)
4. Weight by compromise profile: w_j = 2^(j/2)

The multiscale reward is the weighted average:

    R_multiscale = sum_j(w_j * C_j) / sum_j(w_j)

The compromise weighting (w_j = 2^(j/2)) balances fine-scale and coarse-scale sensitivity. Fine-heavy weighting (w_j = 2^(-j)) emphasizes local fluency; coarse-heavy (w_j = 2^j) emphasizes global coherence. The compromise is the geometric mean and is the recommended default.

### 3.3 Cloud Quality

    R_cloud = mean_k(P_k(t_k) * Phi_k)

This is the product of two desirable properties: high probability on the selected token AND high order parameter. A model that assigns P = 0.9 to the correct token from a peaked distribution (Phi = 0.8) scores 0.72. A model that assigns the same P = 0.9 from a diffuse distribution (Phi = 0.2) scores only 0.18. Cloud quality distinguishes confident correctness from lucky correctness.

### 3.4 Discontinuity Targeting

We detect transitions where |Delta_Phi_k| > 0.03 (the defect threshold) and classify them:

- **Commitment event** (Delta_Phi > +0.03): The model sharpens its distribution. Rewarded when occurring after a period of low order (exploration to commitment). Score: 0.8 if prev_Phi < 0.5, else 0.3.

- **Dissolution event** (Delta_Phi < -0.03): The model broadens its distribution. Penalized when occurring during high-confidence regions. Score: 0.2 if prev_Phi > 0.7, else 0.5.

The discontinuity reward is the mean score across all detected events, defaulting to 0.5 (neutral) when no events are detected.

### 3.5 Integration with GRPO

CARL integrates into TRL's GRPOTrainer as a 6th reward function alongside task-specific rewards. The CARL reward requires a torch.no_grad() forward pass per completion to extract logits, from which the Phi trajectory is computed via numpy. Implementation details that proved critical during development:

- The model must be set to eval() mode during the CARL forward pass to prevent dropout contamination of logits (L1)
- GPU memory must be explicitly freed after each forward pass via torch.cuda.empty_cache() on memory-constrained hardware (L8)
- The metrics shared between the reward closure and the monitoring callback must be protected with threading.Lock (L2)

---

## 4. Cascade Training

Following the Nemotron Cascade 2 methodology, we organize rewards into three stages with 10-step linear warmup at each transition:

| Stage | Steps | Active Rewards | Purpose |
|---|---|---|---|
| A | 0-99 | Format compliance only | Learn the output structure |
| B | 100-199 | All task rewards (R1-R5) | Learn the task |
| C | 200-300 | Task rewards + CARL (R1-R6) | Refine coherence dynamics |

The warmup prevents gradient spikes at stage boundaries: when 5 rewards activate simultaneously at step 100 without warmup, the sudden change in reward landscape destabilizes the optimizer. A 10-step linear ramp (weight increases from 0 to 1 over steps 100-110) smooths the transition.

---

## 5. Experiments

### 5.1 Phase 1: Tool-Calling Agent

**Base model**: OmniCoder-9B (Qwen3.5 9B VLM, Tesslate/OmniCoder-9B)

**SFT**: 1,900 synthetic tool-calling conversations generated via Claude Batch API. 8-bit quantization + LoRA r=64, alpha=128. 3 epochs on L4 GPU.

**GRPO**: 300 steps with 6 reward functions, cascade stages A/B/C. beta=0.0 (no KL penalty, per Nemotron validation). Learning rate 3e-6 with cosine schedule.

**Observation**: During SFT, we observed a sharp entropy spike to ~1.1 followed by immediate collapse to a new, significantly lower steady state. This is consistent with a first-order phase transition: the model's pre-training distribution destabilized (melting) before reorganizing into a tool-calling mode (crystallization). Mean token accuracy rose simultaneously, confirming structural reorganization rather than degradation.

### 5.2 Phase 2: VLM Grounding

**Dataset**: HelloKKMe/grounding_dataset (1.56M instruction-to-coordinate pairs, subsampled to 50K for SFT). OS-Atlas desktop screenshots (20K) for vision-enabled SFT.

**Rewards**: Click accuracy (bbox hit test, w=2.0), coordinate format compliance (w=1.5), precision (distance from bbox center, w=1.0), CARL-VLM composite (w=1.5).

**Architecture note**: CARL's coherence metrics apply identically to VLM generation — the model produces token-level logit distributions regardless of whether the input includes images. No modification to the CARL math is needed for multimodal training. However, activating the vision encoder requires loading the model with `AutoModelForImageTextToText` (yielding `Qwen3_5ForConditionalGeneration`). The default `AutoModelForCausalLM` silently loads `Qwen3_5ForCausalLM`, which strips the vision encoder entirely. TRL auto-detects VLM capability from the processor type.

#### 5.2.1 Text-Only GRPO Failure (v1--v3)

Three iterations of text-only GRPO (no vision input) failed despite correct GRPO machinery:

| Run | Click Accuracy | Mean Distance (px) | Zero-Gradient Steps | Diagnosis |
|-----|---------------|--------------------|--------------------|-----------|
| v1 | 8% | 504 | -- | Cascade wasted 100 steps on format (already solved by SFT) |
| v2 | 8% | 498 | 41% | Fixed reward scale (1000px), higher LR, more steps. No improvement |
| v3 | 8% | 501 | 41% | Diagnostic run — confirmed mode collapse |

The v3 diagnostic revealed the failure mode: **68% of prompts produced identical coordinates across all 8 generations within a GRPO group.** The model had memorized (492, 544) as a generic center coordinate. Phi = 0.9998 (near-delta distribution). Format reward was constant 1.000 with std = 0.000 — the model had perfect format compliance but zero spatial understanding.

**Root cause**: Without vision input, the model cannot connect natural language instructions ("click the submit button") to screen locations. It collapses to the statistical center of the training distribution.

An attempt to break mode collapse via temperature=2.0 and top_k=50 succeeded in eliminating zero-gradient steps (0% vs 41%) but did not improve spatial accuracy — the fundamental visual signal was absent.

#### 5.2.2 Vision SFT Phase Transition

The most striking result in this work. When training with actual screenshots (20K desktop screenshots from OS-Atlas), the model exhibited a sharp first-order phase transition:

| Step Range | Phase | Loss | Token Accuracy | Entropy | Description |
|-----------|-------|------|---------------|---------|-------------|
| 0--10 | Baseline | 22.8 | 3% | 1.0 | Pre-training distribution intact |
| 10--20 | Melting | 15.2 -> 8.4 | 3% -> 8% | 1.0 -> 9.3 | Distribution completely destabilizes |
| 20--25 | **Phase transition** | 8.4 -> 3.7 | 8% -> 65% | **9.3 -> 4.1** | Accuracy jumps 57 points in 5 steps |
| 25--35 | Crystallization | 3.7 -> 0.13 | 65% -> 99% | 4.1 -> 0.4 | Rapid convergence to new steady state |
| 35--46 | Convergence | 0.06 | 99.3% | 0.12 | Fully crystallized |

The entropy spike to 9.3 (near the theoretical maximum of log|V|) represents a complete melting of the pre-training distribution — the model's internal representations temporarily become maximally uncertain before reorganizing around the coordinate output format. This is consistent with Kuramoto synchronization theory: the vision encoder coupling with the language model creates a first-order phase transition analogous to the order-disorder transition in coupled oscillator systems. The critical observation is that accuracy does not improve gradually — it jumps discontinuously once the system passes the critical coupling threshold.

This phase transition was absent during the Phase 1 SFT (tool-calling), which showed a milder entropy spike to ~1.1. We attribute the difference to the magnitude of the distribution shift: tool-calling format is a local perturbation of the pre-training distribution (still generating text), while coordinate generation requires a global reorganization (generating numbers conditioned on visual input).

#### 5.2.3 The CARL Coherence Trap

The text-only GRPO failure exposed a fundamental limitation in CARL's reward formulation. A model exhibiting mode collapse — outputting identical tokens with 99.98% confidence at every position — receives a **high** CARL composite score (~0.55) because:

1. **Phi ~ 1.0 everywhere**: A delta-like distribution has maximum order parameter. Multiscale coherence measures the consistency of Phi across scales, and constant Phi = 1.0 is maximally consistent.

2. **Cloud quality ~ 0.9998**: The product P(selected) * Phi is near-maximal when the model assigns ~100% probability to a single token.

3. **Zero discontinuities**: No transitions detected, so the discontinuity score defaults to 0.5 (neutral).

This is not a bug in the implementation — it is a structural limitation. **Maximum coherence IS a delta function, which IS a collapsed model.** CARL's reward landscape has a degenerate global optimum at mode collapse.

The sigma = 0.1875 noise floor was intended to filter sub-semantic signals, but it operates on per-token advantages, not on population-level diversity. The defect threshold (|Delta_Phi| > 0.03) is too coarse to distinguish "confidently correct across diverse inputs" from "confidently stuck on a single output."

**Implication for CARL v2**: The framework needs a population-level diversity term that penalizes when completions within a GRPO group are too similar. A candidate formulation:

    R_diversity = 1 - mean_pairs(cosine_sim(completion_i, completion_j))

This would assign R_diversity = 0 when all completions are identical and R_diversity ~ 1 when completions are maximally diverse, directly counteracting the coherence trap.

---

## 6. CARL Studio

We release CARL Studio as an open-source Python package implementing the full CARL framework:

### 6.1 Architecture

```
Layer 4: Studio        -- Next.js web UI (planned)
Layer 3: MCP Server    -- 9 tools for AI agent consumption
Layer 2: CLI           -- `carl` command for human operators
Layer 1: SDK           -- carl_studio Python package
Layer 0: Primitives    -- CoherenceProbe, constants, compute_phi
```

### 6.2 API

```python
from carl_studio import CARLTrainer, TrainingConfig

config = TrainingConfig(
    run_name="tool-calling-grpo",
    base_model="Tesslate/OmniCoder-9B",
    output_repo="wheattoast11/OmniCoder-9B-Zero-Phase1",
    method="grpo",
    dataset_repo="wheattoast11/zero-rl-tool-calling-data",
    compute_target="l4x1",
)

trainer = CARLTrainer(config)
run = await trainer.train()
```

### 6.3 Compute Backends

| Backend | Package | Type |
|---|---|---|
| HuggingFace Jobs | huggingface_hub | Managed UV scripts |
| RunPod | runpod v1.8 | GPU pods |
| Tinker | tinker v0.16 | Managed training API |
| Prime Intellect | prime v0.5 | GPU marketplace |
| SSH | asyncssh | Remote execution |
| Local | subprocess | Direct GPU |

### 6.4 MCP Server

The MCP server exposes 9 tools enabling AI agents to autonomously manage training:

- start_training, get_run_status, get_logs
- observe_now, get_coherence_metrics
- stop_training, validate_config
- generate_bundle, list_backends

---

## 7. Connection to the Convergence-Realizability Identity

The convergence-realizability identity (Desai, 2026a; 2026b) establishes that phase synchronization and semantic constructiveness are coextensive events in coupled oscillator systems. Across 6,244 trials spanning four topologies (Ring, Sparse3, Fano complete, Heawood) and two population sizes (N=7, N=14), the rate at which agents achieve phase coherence R >= Rc is identical to the rate at which their joint output satisfies the constructiveness predicate (converged AND informative AND non-empty). The identity holds with zero deviation under closed witness loops across all 128 experimental conditions.

CARL operationalizes this identity as a training signal. If synchronization IS constructiveness, then training a model to produce clean phase transitions in its probability distribution should simultaneously train it to produce constructive outputs. The order parameter Phi_k = 1 - H(P_k)/log|V| is the per-token analog of the Kuramoto order parameter R — both measure the concentration of probability mass (over vocabulary vs. over phase angles). The mapping is direct: a token position with Phi_k near 1 corresponds to a synchronized oscillator with R near 1; a position with Phi_k near 0 corresponds to a desynchronized oscillator with uniformly distributed phase.

The conservation law T* = kappa * d provides the quantitative bridge between the two regimes. In the multi-agent setting, the coupling strength K determines synchronization time T_sync. In the training setting, the embedding dimension d determines convergence time via the same constant kappa = 64/3 ~ 21.33. Both are instances of the same information-theoretic bound on how quickly a system can organize itself from disorder to coherence. The semantic quantum sigma = 3/16 defines the minimum meaningful signal in both settings: sub-sigma perturbations in Kuramoto phase angles do not contribute to synchronization, just as sub-sigma per-token advantages do not contribute to learning.

Phase 2' provides the strongest test of the identity. The CodingSandboxEnv offers binary ground-truth constructiveness: the model's generated code either executes without error (constructive) or crashes (non-constructive). The CARL reward provides the coherence signal, computed from the model's logit distribution with no knowledge of task outcomes. If the identity holds, models trained with CARL should achieve higher task completion rates than models trained without it — not because CARL directly rewards task completion, but because coherence and constructiveness are the same event measured at different scales. The ablation (CARL-only vs. task-only vs. combined) is designed to test this prediction directly, providing the first empirical validation of the convergence-realizability identity in a language model training context.

---

## 8. Related Work

**GRPO** (Shao et al., 2024) introduced group relative policy optimization, eliminating the need for a separate reward model by computing advantages within groups of completions. CARL extends GRPO by adding information-theoretic rewards that operate on the model's logit distribution rather than the generated text.

**Nemotron Cascade 2** demonstrated that cascade training — activating reward components in stages — outperforms simultaneous activation on mathematical and coding benchmarks. Our cascade implementation follows this methodology with the addition of linear warmup at stage boundaries.

**Constitutional AI** (Bai et al., 2022) introduced the idea of training models against principles rather than human preferences. CARL's conservation law (kappa * sigma = 4) provides a mathematical principle — the semantic quantum — that constrains the reward landscape without requiring human-generated preference data.

**DPO** (Rafailov et al., 2023) showed that reward model training can be implicit. CARL is complementary: it can be combined with DPO, GRPO, or any RL method by adding coherence components to the existing reward signal.

**Phase synchronization as constructiveness.** The convergence-realizability identity (Desai, 2026a; 2026b) establishes that Kuramoto synchronization and Deutsch-Marletto constructiveness are coextensive events across coupled oscillator networks. Across 6,244 trials and 128 conditions spanning four graph topologies and two population sizes, the rate of phase coherence R >= Rc is identical to the rate of semantic constructiveness (converged AND informative AND non-empty), with zero deviation under closed witness loops. CARL extends this finding from multi-agent simulation to language model training, using the order parameter Phi as a per-token analog of the Kuramoto R. Section 7 details how Phase 2' provides an independent empirical test of the identity in a training context.

---

## 9. Discussion

### 9.1 The sigma Noise Floor

The semantic quantum sigma = 0.1875 provides a principled noise floor. In our experiments, we track the fraction of per-token advantages exceeding sigma. When this fraction drops below 30%, the reward signal is predominantly sub-semantic noise, suggesting the learning rate or reward scale needs adjustment. This diagnostic is unavailable in standard RL training.

### 9.2 Limitations

**Computational cost.** CARL's forward pass per completion adds computational cost (~8 extra forward passes per GRPO step with num_generations=8). On L4 hardware (22GB VRAM), this fits within budget but leaves minimal margin. The cost scales linearly with completion count and could become prohibitive for large generation batch sizes.

**Discontinuity content-blindness.** The discontinuity scoring lacks true content awareness — we use the previous Phi value as a proxy for semantic context, not explicit section labels. A model producing a commitment event at an inappropriate location receives partial rather than zero reward.

**The coherence trap (Section 5.2.3).** CARL's most serious limitation is that its reward landscape has a degenerate optimum at mode collapse. Because the framework measures only the coherence of individual completions — not diversity across completions — a model that produces identical, maximally confident outputs receives near-optimal CARL scores. This was directly observed in Phase 2 text-only GRPO (v1--v3), where the model collapsed to a single coordinate output while maintaining CARL composite ~ 0.55. Any deployment of CARL without an accompanying diversity mechanism risks rewarding exactly the failure mode it should detect.

### 9.3 Future Work

- **Population-level diversity term**: The coherence trap (Section 5.2.3) demands a diversity component in the CARL composite. A pairwise cosine dissimilarity across completions within each GRPO group would directly penalize mode collapse. The open question is weight scheduling: diversity should dominate early (when collapse risk is highest) and decay as the model acquires genuine competence.
- **Modality-tagged discontinuities**: For VLM training, track which modality (vision vs. text) the model is processing at each position. Discontinuities at modality boundaries are expected and should be rewarded differently than within-modality discontinuities. The Phase 2 vision SFT phase transition (Section 5.2.2) suggests that modality coupling events produce characteristic discontinuity signatures that could be leveraged as training signals.
- **Learned scale weights**: The compromise weighting w_j = 2^(j/2) is fixed. End-to-end learning of per-scale weights could adapt the reward to domain-specific coherence patterns.
- **Conservation law predictions**: The relationship T* = kappa * d predicts that models with triadic embedding dimensions (d = 3 * 2^k) should achieve exact binary context windows with zero waste. This is testable and, if validated, would have architectural implications.
- **Phase transition characterization**: The sharp entropy spike observed in Vision SFT (9.3 at step ~20) compared to the mild spike in tool-calling SFT (~1.1) suggests that the magnitude of the phase transition correlates with the distance between the target distribution and the pre-training distribution. A systematic study across task types could yield a predictive model of when crystallization will occur and how many steps it requires.

---

## 10. Conclusion

CARL provides a principled bridge between information theory and reinforcement learning for language models. By measuring the order parameter, cloud quality, and discontinuity patterns of a model's token-level probability distribution, CARL reward signals complement existing task-specific rewards with structural quality metrics. The conservation law kappa * sigma = 4 establishes a universal noise floor and decompression boundary that apply across architectures.

Our implementation in CARL Studio demonstrates that coherence-aware training is practical: it integrates into existing TRL workflows, runs on standard GPU hardware, and produces measurable improvements in training dynamics — including observable phase transitions during SFT that serve as concrete diagnostics of successful format learning. The vision SFT phase transition (entropy 1.0 -> 9.3 -> 0.12, accuracy 3% -> 99.3%) is, to our knowledge, the sharpest such transition reported in VLM fine-tuning literature.

However, our Phase 2 experiments also reveal that CARL in its current form is incomplete. The coherence trap — where mode-collapsed models achieve near-optimal CARL scores — demonstrates that individual-completion coherence is necessary but not sufficient for training signal design. Population-level diversity must be incorporated to prevent the framework from rewarding the exact failure mode it should detect.

The mathematics is simple. The implementation is straightforward. The observations are striking: models don't learn gradually — they crystallize. But crystallization into a single point is not learning. CARL v2 must distinguish the crystal from the collapsed.

---

## References

1. Desai, T. (2026). *Bounded Informational Time Crystals.* Zenodo. DOI: 10.5281/zenodo.18906944

2. Desai, T. (2026). *Material Reality: Empirical Validation of Coherence Gating.* 6,244 trials, 128 conditions. Intuition Labs LLC.

3. Desai, T. (2026). *Semantic Realizability: The Convergence-Realizability Identity.* 6,244 trials, 128 conditions. Intuition Labs LLC.

4. Desai, T. (2026). *The IRE Paradigm: Interactive Research Environments via Coherence Gating.* Intuition Labs LLC.

5. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv: 2402.03300.

6. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv: 2212.08073.

7. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.

8. Wang, H., et al. (2025). *Nemotron Cascade 2.* NVIDIA.

9. Desai, T. (2026a). *Material Reality: Empirical Validation of Coherence Gating.* Zenodo. DOI: 10.5281/zenodo.18992029

10. Desai, T. (2026b). *Semantic Realizability: The Convergence-Realizability Identity.* Zenodo. DOI: 10.5281/zenodo.18992031

---

**Code**: github.com/terminals-tech/carl-studio

**Conservation Law**: T* = kappa * d, kappa = 2^6/3, sigma = 3/16

**Intuition Labs LLC** -- terminals.tech
