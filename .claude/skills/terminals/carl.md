---
name: carl
description: |
  This skill should be used when the user asks to "train a model", "run CARL", "check coherence",
  "submit a training job", "monitor training", "evaluate a checkpoint", "deploy the agent",
  "carl train", "carl eval", "carl observe", or works with CARL-trained models, coherence metrics,
  phase transitions, or the zero-rl-pipeline. The CARL training companion for carl-studio.
version: 0.3.0
---

# CARL Training Companion

Manage the full lifecycle of training VLMs with coherence-aware reward signals via carl-studio.

## Conservation Law

| Constant | Value | Meaning |
|----------|-------|---------|
| kappa | 64/3 | Conservation constant |
| sigma | 3/16 | Semantic quantum |
| kappa * sigma | 4 | Bits per embedding dimension |
| T* | kappa * d | Decompression boundary |
| Defect threshold | \|delta_Phi\| > 0.03 | Discontinuity detection |

Triadic dimensions (d = 3*2^k) give exact binary T*. Non-triadic incur ~33% context tax.

## Pipeline Architecture

```
Phase 1' (COMPLETE):  VLM grounding — 94.6% click accuracy, 100% format compliance
Phase 2' (ACTIVE):    Environment GRPO — tool-calling through real sandbox interaction
Phase 3  (BUILT):     TTT — SLOT optimizer + LoRA micro-update for inference-time adaptation
```

### Phase 2' — Environment GRPO (Current)

The model enters a real coding sandbox (`CodingSandboxEnv`) and learns tool-calling through interaction. Four tools: `read_file`, `write_file`, `execute_code`, `run_shell`.

**Reward architecture (4 functions, 3 roles):**
1. `tool_engagement` (w=1.0) — dense: "did the model try?"
2. `task_completion` (w=3.0) — sparse binary: "did the code run?"
3. `gated_carl` (w=1.0) — quality: "was the reasoning clean?" (Stage B only)
4. `gr3_length_penalty` (w=2.0) — structural: shorter correct solutions score higher

**Cascade gating:** CARL activates when `task_completion` shows sustained capability. Self-calibrating — no hardcoded threshold.

## Training Operations

### Submit a Job

```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
job = api.run_uv_job(
    script='https://huggingface.co/datasets/wheattoast11/zero-rl-tool-calling-data/resolve/main/train_env_grpo.py',
    flavor='a100-large',
    timeout='2d',
    secrets={'HF_TOKEN': HF_TOKEN},
)
```

### Monitor

```python
j = api.inspect_job(job_id=JOB_ID)         # Status
logs = list(api.fetch_job_logs(job_id=JOB_ID))  # Logs (generator)
```

Watch Trackio dashboard for smoothed trends. Never judge from individual log entries.

### Key Health Signals

- **Length phase transition:** Rise-then-drop in mean_terminated_length = model learning. Monotonic growth = death.
- **Tool call frequency:** Should increase as model learns. 0 → 1-4 by step 30+.
- **Cascade gate fire:** Stage A → B when task_completion sustains above running distribution median.
- **Clipped ratio:** High (>50%) means completions hitting max_completion_length. Normal early, should decrease.

## Phase Transition Gate

`PhaseTransitionGate` monitors `mean_token_accuracy` during SFT. Windowed check: 3 of last 5 steps above 0.99 = crystallized. Robust to batch-level noise.

Structural signature: entropy spike (1.0 → 9.3 → 0.12) over ~40 steps. Token accuracy jumps from 3% to 99% during collapse. Consistent with Kuramoto synchronization.

## Coherence Trap

CARL rewards high Phi (coherence). Maximum Phi = delta function = mode collapse. Without cascade gating, CARL dominates sparse task signal and collapses tool-calling at step ~26 (observed in v1). Fix: `CascadeRewardManager` gates CARL behind task performance.

If GRPO produces identical completions (zero within-group variance): temperature >= 2.0 + top_k=50.

## Compute

| Phase | Flavor | VRAM | Price |
|-------|--------|------|-------|
| Phase 1' SFT/GRPO | l40sx1 | 48 GB | $1.80/hr |
| Phase 2' Env GRPO | a100-large | 80 GB | $2.50/hr |
| Eval | l4x1 | 24 GB | $0.80/hr |

**Required env var:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set in the training script. Prevents CUDA fragmentation OOM during multi-turn generation.

## CLI Reference

```
carl train         Train with CARL rewards
carl train --send-it   Full autonomous pipeline
carl eval          Run eval gates
carl observe       Live coherence monitoring
carl align         Realign a drifted model
carl learn         Ingest new knowledge
carl bench         Coherence meta-benchmarks
carl status <id>   Job status
carl push          Push checkpoint to Hub
carl bundle        Generate self-contained script
carl mcp           Start MCP server
```

## File Layout

```
phase1/         # CARL infrastructure (ModelSpec, rewards, tool profiles)
phase2/         # Phase 2' environment GRPO (ACTIVE)
phase3/         # TTT infrastructure (SLOT, LoRA, continuous RL)
scripts/        # CLI tools, submission, eval, diagnostics
data/           # Training/eval data
legacy/         # Archived scripts (v1-v7 versions preserved)
```

## Hub Models

| Model | Phase | Status |
|-------|-------|--------|
| wheattoast11/OmniCoder-9B-Zero-Phase2 | Phase 1' | PASS — 94.6% click accuracy |
| wheattoast11/OmniCoder-9B-Zero-Phase2Prime | Phase 2' | Training (v7) |
