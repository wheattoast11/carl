# CARL Studio

**Coherence-Aware Reinforcement Learning**

> Models don't learn gradually -- they crystallize.

![CARL Phase Transition](assets/carl-paradigm.gif)

CARL adds information-theoretic reward signals to RL training that measure *how* a model generates, not just *what* it generates. One conservation law. Three reward components. Model-agnostic.

## Install

```bash
pip install carl-studio
```

## Quick Start

### Observe (zero friction -- point at any existing run)

```bash
carl observe --trackio https://your-space.hf.space
```

No training, no config, no GPU. See your model's learning geometry instantly.

### Train

```python
from carl_studio import CARLTrainer, TrainingConfig

trainer = CARLTrainer(TrainingConfig(
    run_name="my-first-carl",
    base_model="Qwen/Qwen3.5-9B",
    output_repo="your-username/my-model",
    method="grpo",
    dataset_repo="trl-lib/Capybara",
    compute_target="l4x1",
))
run = await trainer.train()
```

Or from the CLI:

```bash
carl train --model Qwen/Qwen3.5-9B --method grpo --compute l4x1
carl train --send-it                  # full pipeline: SFT -> gate -> GRPO -> eval -> push
```

## Architecture

```
Layer 4  MCP Server       9 tools for AI agent consumption
         ──────────────────────────────────────────────────
Layer 3  CLI              carl train | eval | observe | align | learn | bench | mcp
         ──────────────────────────────────────────────────
Layer 2  Training         CARLTrainer, CascadeRewardManager, environments
         ──────────────────────────────────────────────────
Layer 1  SDK              ModelSpec, TrainSpec, VRAMBudget, CoherenceProbe
         ──────────────────────────────────────────────────
Layer 0  Primitives       compute_phi, kappa, sigma, PhaseTransitionGate
```

### The Reward

```
R_CARL = 0.50 * R_coherence + 0.30 * R_cloud + 0.20 * R_discontinuity
```

| Component | Measures |
|-----------|----------|
| Multiscale coherence | Phi consistency across dyadic block scales |
| Cloud quality | P(selected) * Phi -- confident AND correct |
| Discontinuity targeting | Sharp Phi transitions at structurally appropriate locations |

### Cascade Gating

CARL is length-biased -- a verbose, confident model scores high. Without gating, it dominates sparse task signal and causes mode collapse. The cascade solves this:

```
Stage A (early):   task rewards only         -- "learn to use tools"
Stage B (gated):   task + CARL rewards       -- "now do it coherently"
```

The gate self-calibrates from the task metric's running distribution. No hardcoded threshold.

### Order Parameter

```
Phi = 1 - H(P) / log|V|
```

0 = uniform (maximum uncertainty). 1 = delta (complete certainty).

## The Conservation Law

| Constant | Value | Meaning |
|----------|-------|---------|
| kappa | 64/3 | Conservation constant |
| sigma | 3/16 | Semantic quantum |
| kappa * sigma | **4** | Bits per embedding dimension |
| T* | kappa * d | Decompression boundary |

Derived in [Bounded Informational Time Crystals](https://doi.org/10.5281/zenodo.18906944). Validated across 6,244 trials in [Material Reality](https://doi.org/10.5281/zenodo.18992029). Formally proved in [Semantic Realizability](https://doi.org/10.5281/zenodo.18992031).

## Key Finding: Phase Transitions

During VLM SFT, the model exhibits a first-order phase transition:

| Steps | Phase | Accuracy | Entropy | What happens |
|-------|-------|----------|---------|--------------|
| 0-10 | Baseline | 3% | 1.0 | Pre-training distribution intact |
| 10-20 | Melting | 8% | **9.3** | Distribution destabilizes completely |
| 20-25 | **Transition** | **65%** | 4.1 | Accuracy jumps 57 points in 5 steps |
| 25-35 | Crystallization | 99% | 0.4 | Rapid convergence |
| 35-46 | Converged | **99.3%** | 0.12 | Fully crystallized |

Entropy spikes to near-maximum, then accuracy discontinuously jumps once the system passes the critical coupling threshold. Consistent with Kuramoto synchronization in coupled oscillator systems.

## CLI

**Core triad:**
```
carl observe       See learning geometry on any run (no GPU required)
carl eval          Pass/fail gate on a checkpoint
carl train         Train with CARL rewards (SFT, GRPO, DPO, KTO, ORPO)
carl train --send-it   Full autonomous pipeline: SFT -> gate -> GRPO -> eval -> push
```

**Operations:**
```
carl status <id>   Job status
carl logs <id>     Job logs
carl stop <id>     Cancel a job
carl push          Push checkpoint to Hub
carl bundle        Generate self-contained training script
carl compute       List GPU flavors and pricing
carl setup         First-time setup
```

**Advanced** (experimental):
```
carl align         Realign a drifted model
carl learn         Ingest knowledge, generate data, train
carl bench         Coherence meta-benchmarks
carl mcp           Start MCP server (9 tools for AI agents)
carl dev           Development utilities
```

## Model-Agnostic

`ModelSpec.from_pretrained()` auto-detects architecture, modality, thinking mode, quantization constraints, and LoRA targets from any HuggingFace config.json. No per-model branches.

| Model | Status |
|-------|--------|
| Qwen 3.5 9B VLM | Primary -- 94.6% click accuracy |
| Gemma 4 E4B | Planned |
| Gemma 4 31B | Planned (multi-GPU) |

## Compute Backends

| Backend | Flag |
|---------|------|
| HuggingFace Jobs | `--compute l4x1` / `a100-large` / `h200` |
| RunPod | `--compute runpod` |
| Tinker | `--compute tinker` |
| Prime Intellect | `--compute prime` |
| SSH | `--compute ssh` |
| Local | `--compute local` |

## Test-Time Training

CARL includes TTT mechanisms for post-deployment adaptation:

- **SLOT** -- hidden delta injection (8 Adam steps, architecture-agnostic)
- **LoRA micro-update** -- rank-1 online adaptation

## IP Boundaries

CARL Studio is MIT-licensed. The mathematics -- conservation law, order parameter, reward components -- are independently derivable from the three published papers (CC-BY-4.0).

This package is the *open training framework*. It does **not** include the runtime dynamics or autonomous orchestration:

| What | Where | License |
|------|-------|---------|
| Conservation law, Phi, rewards | **CARL Studio** (this repo) | MIT |
| Observe, eval, train CLI | **CARL Studio** (this repo) | MIT |
| Resonance LR modulation | **terminals-runtime** | BUSL-1.1 |
| SLOT / LoRA micro-update (TTT) | **terminals-runtime** | BUSL-1.1 |
| Kuramoto oscillator dynamics | **terminals-runtime** | BUSL-1.1 |
| Coherence diagnosis methodology | **terminals-runtime** | BUSL-1.1 |
| Audio coherence (CHORD) | Terminals Platform | BUSL-1.1 |
| Cross-substrate isomorphisms | Terminals Platform | BUSL-1.1 |
| Interactive Research Environment | Terminals Platform | BUSL-1.1 |
| Material Reality datasets | Zenodo | CC-BY-4.0 |

The bifurcation is deliberate: CARL Studio provides the full training loop (observe, eval, train) using published mathematics. The `terminals-runtime` package adds autonomous features (resonance-aware LR, test-time training, Claude-powered diagnosis) behind BUSL-1.1. Same conservation law, different sides of the boundary.

## Citation

```bibtex
@article{desai2026carl,
  title   = {Coherence-Aware Reinforcement Learning},
  author  = {Desai, Tej},
  year    = {2026},
  url     = {https://github.com/wheattoast11/carl},
  note    = {Intuition Labs LLC}
}
```

## License

MIT -- Intuition Labs LLC
