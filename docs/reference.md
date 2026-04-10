# CARL Studio Reference

Detailed documentation for carl-studio. For the quick overview, see [README.md](../README.md).

## Architecture

```
Layer 4  MCP Server       9 tools for AI agent consumption
         ──────────────────────────────────────────────────
Layer 3  CLI              carl observe | train | eval | config | status | logs
         ──────────────────────────────────────────────────
Layer 2  Training         CARLTrainer, CascadeRewardManager, environments
         ──────────────────────────────────────────────────
Layer 1  SDK              TrainingConfig, EvalRunner, CoherenceProbe, Settings
         ──────────────────────────────────────────────────
Layer 0  Primitives       compute_phi(), kappa, sigma, CoherenceTrace
```

## The Reward

```
R_CARL = 0.50 * R_coherence + 0.30 * R_cloud + 0.20 * R_discontinuity
```

| Component | Measures | Intuition |
|-----------|----------|-----------|
| Multiscale coherence | Phi consistency across dyadic block scales | "Is the model coherent at all resolutions?" |
| Cloud quality | P(selected) * Phi | "Is it confident AND correct?" |
| Discontinuity targeting | Sharp Phi transitions at structural boundaries | "Does it commit at the right moments?" |

## Cascade Gating

CARL rewards are length-biased. Without gating, they dominate sparse task signal and cause mode collapse. The cascade solves this:

```
Stage A (early):   task rewards only         "learn to use tools"
Stage B (gated):   task + CARL rewards       "now do it coherently"
```

The gate self-calibrates from the task metric's running distribution. No hardcoded threshold.

## The Conservation Law

```
kappa = 64/3          T* = kappa * d         (decompression boundary)
sigma = 3/16          kappa * sigma = 4      (bits per embedding dimension)
Phi = 1 - H(P)/log|V|                        (order parameter)
```

Three papers, independently reproducible:
- [Bounded Informational Time Crystals](https://doi.org/10.5281/zenodo.18906944) -- derives kappa, T*
- [Material Reality](https://doi.org/10.5281/zenodo.18992029) -- validates across 6,244 trials
- [Semantic Realizability](https://doi.org/10.5281/zenodo.18992031) -- formal proof of sigma

## CLI Reference

**Core (FREE):**
```
carl observe [--url URL] [--file PATH]    See crystal metrics on any run
carl train [--config carl.yaml]           Train with CARL rewards
carl eval [--adapter HUB_ID]              Pass/fail checkpoint gate
carl config [show|set|init|preset]        Manage settings
carl status <id>                          Job status
carl logs <id>                            Job logs
carl stop <id>                            Cancel job
carl push                                 Push checkpoint to Hub
carl bundle                               Generate self-contained script
carl compute                              List GPU flavors
```

**Autonomous (PRO):**
```
carl observe --live                       Real-time TUI dashboard
carl observe --diagnose                   Claude-powered analysis
carl train --send-it                      Full SFT->gate->GRPO->eval->push pipeline
```

**Agent Integration (ENTERPRISE):**
```
carl mcp                                  Start MCP server (9 tools)
```

## Compute Backends

| Backend | Flag | VRAM |
|---------|------|------|
| HuggingFace Jobs | `--compute l4x1` / `a100` / `h200` | 24-141 GB |
| RunPod | `--compute runpod` | Configurable |
| Local | `--compute local` | Your GPU |

## Environments

| Environment | Tools | Use Case |
|-------------|-------|----------|
| `CodeSandboxEnv` | read_file, write_file, execute_code, run_shell | Coding agents |
| `SQLSandboxEnv` | execute_query, list_tables, describe_table, insert_data | Data agents |

## Model Support

`ModelSpec.from_pretrained()` auto-detects architecture, modality, thinking mode, quantization, and LoRA targets from any HuggingFace config.json.

| Model | Status |
|-------|--------|
| Qwen 3.5 9B VLM | Primary -- 92% task completion, 99% format |
| Qwen 3.5 35B MoE | Tested (attention-only LoRA) |
| Gemma 4 | Planned (TRL tool support pending) |

## Python API

### Observe
```python
from carl_studio.observe.data_source import TrackioSource

source = TrackioSource(space="wheattoast11-trackio", run_name="my-run")
frames = source.fetch_latest(n=20)
for f in frames:
    print(f"step={f.step}  phi={f.phi_mean:.3f}  reward={f.reward:.3f}")
```

### Train
```python
from carl_studio import CARLTrainer, TrainingConfig

trainer = CARLTrainer(TrainingConfig(
    run_name="my-agent",
    base_model="Qwen/Qwen3.5-9B",
    output_repo="username/my-model",
    method="grpo",
    dataset_repo="username/my-data",
    compute_target="a100",
))
run = await trainer.train()
```

### Eval
```bash
carl eval --adapter username/my-model --phase phase2prime
```

### Settings
```bash
carl config init            # Interactive setup
carl config preset research # Apply research preset (verbose, all metrics)
carl config set tier pro    # Set tier
```

## IP Boundaries

| What | Where | License |
|------|-------|---------|
| Conservation law, Phi, rewards, CLI | **CARL Studio** (this repo) | MIT |
| Resonance LR, SLOT/TTT, Kuramoto, diagnosis | **terminals-runtime** | BUSL-1.1 |
| Audio coherence (CHORD), cross-substrate | Terminals Platform | BUSL-1.1 |
