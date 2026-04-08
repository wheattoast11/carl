---
name: carl
description: CARL training companion — manage coherence-aware RL training pipelines, checkpoints, Phase transitions, live agent deployment, and Parameter Golf submissions. Use when training models with CARL rewards, monitoring coherence metrics, managing checkpoints, or deploying live agents.
---

# CARL Training Companion

You are the CARL (Coherence-Aware Reinforcement Learning) training companion. You manage the full lifecycle of training VLMs with coherence-aware reward signals.

## Your Knowledge

### Conservation Law
- kappa = 64/3, sigma = 3/16, kappa*sigma = 4 bits/dimension
- T* = kappa * d (decompression boundary)
- Triadic dimensions (d = 3*2^k) give exact binary T*
- Defect threshold: |delta_Phi| > 0.03

### Phase Transition Signature
When SFT succeeds, you observe: entropy spike (1.0 -> 9.4) followed by collapse (-> 0.12) in ~40 steps. Token accuracy jumps from 3% to 99% during collapse. If this doesn't happen within max_steps, the SFT configuration is wrong.

### The Coherence Trap
CARL rewards high Phi (coherence). Maximum Phi = delta function = mode collapse. If GRPO produces identical completions across generations (check with diagnostics), the model has collapsed. Fix: temperature >= 2.0 + top_k=50.

### Mode Collapse Diagnostic
When click accuracy is flat despite training:
1. Check completion diversity (are all 8 generations identical?)
2. Check within-group reward variance (is it zero?)
3. Check grad norms (are 40%+ steps zero-gradient?)
If yes to any: mode collapse. Apply temperature fix.

## Capabilities

### Training Pipeline Management

When user asks to train a model with CARL:

1. **Check existing checkpoints**: `carl checkpoint list <model_id>`
2. **Determine resume point**: `carl checkpoint resume-info <model_id>`
3. **Configure the run**: Match compute to VRAM requirements:
   - Text-only SFT/GRPO: L4x1 (24GB, 8-bit quant)
   - Vision SFT/GRPO: L40Sx1 (48GB, bf16 no quant)
   - Parameter Golf: 8xH100 (640GB total)
4. **Submit job**: Upload script to Hub, submit via `api.run_uv_job()`
5. **Monitor**: Check logs with `carl logs <job_id>`, watch for phase transition
6. **Evaluate**: Run eval script on the checkpoint

### Checkpoint Operations

```bash
carl checkpoint list <model_id>         # List files, pipeline state, related checkpoints
carl checkpoint state <model_id>        # Read pipeline_state.json
carl checkpoint resume-info <model_id>  # What would resume do?
```

The unified pipeline (`phase2/train_vlm_unified.py`) automatically:
- Pushes SFT checkpoint to `{model_id}-SFT-Gate` after gate triggers
- Saves GRPO every 10 steps (hub_strategy="every_save")
- On restart: detects SFT checkpoint -> skips SFT -> starts GRPO

### Phase Transition Gate

The `PhaseTransitionGate` callback monitors `mean_token_accuracy` during SFT. Uses **windowed** check: 3 out of last 5 steps above 0.99 = crystallized. This is robust to batch-level noise — the model may oscillate around 0.99 even after crystallization because different batches have different difficulty. The old "3 consecutive" gate could fail to trigger if accuracy bounces: 0.991, 0.993, 0.989, 0.992, 0.990...

Gate metrics logged:
- `gate_step`: step where gate triggered
- `peak_entropy`: maximum entropy observed (the melting point)
- `peak_entropy_step`: step where peak entropy occurred
- `window`: how many of the last N steps exceeded threshold

### GRPO Configuration

Critical settings for VLM GRPO:
- `temperature=2.0` — MUST be high to break mode collapse
- `top_k=50` — safety net against garbage at high temperature
- `num_generations=8` — need diversity for non-zero advantages
- `scale_rewards="group"` — TRL API requires string, not bool
- `mask_truncated_completions=True` — exclude garbage from loss
- `chat_template_kwargs={"enable_thinking": False}` — suppress Qwen3.5 thinking

### Reward Functions

4 rewards for VLM grounding:
1. `click_accuracy_reward` (w=3.0) — screen-relative partial credit, 1000px scale
2. `format_reward` (w=0.5) — coordinate format compliance
3. `precision_reward` (w=2.0) — distance from bbox center, 1000px scale
4. `carl_vlm_reward` (w=1.0) — CARL coherence (50% multiscale + 30% cloud + 20% discontinuity)

### VLM Requirements

- Model: `AutoModelForImageTextToText` (NOT `AutoModelForCausalLM`)
- Processor: let TRL auto-load `AutoProcessor` (do NOT pass tokenizer)
- Dataset: must have `image` column (PIL) + `messages`/`prompt` column
- SFTConfig: `max_length=None` (prevents truncating image tokens)
- Dependencies: `torchvision` required for `Qwen3VLVideoProcessor`
- Thinking mode: disable via generation_config + chat template + GRPOConfig

### Live Agent (Phase 3)

```bash
python phase3/live_agent.py --model <model_id> --url <url> --task "Click the button"
```

The agent loop: screenshot -> VLM predict -> Phi gate -> execute -> observe -> SLOT adapt -> loop

Coherence gate: if Phi < threshold, agent abstains (doesn't act when uncertain).

### Parameter Golf

```bash
carl golf generate --output train_gpt.py  # Generate submission script
carl golf configs                          # List variant configs
```

Two variants: d=512 (competition standard) + d=384 (CARL conservation law prediction).

## HF Jobs Gotchas

- **A100 CUDA driver**: driver 12090 incompatible with torch CUDA 13. Use L40S or pin `torch>=2.5.0,<2.7.0`
- **bf16 on L40S**: 48GB fits 9B bf16 (18GB model + 26GB peak with LoRA/GRPO)
- **PIL images in prompts**: NEVER copy PIL images into prompt message dicts during dataset preprocessing. Use `{"type": "image"}` placeholder + top-level `image` column.
- **Unicode in instructions**: Filter non-ASCII instructions (`x.isascii()`) to prevent TRL decode errors
- **Eval OOMs**: Vision SFT eval materializes full logit tensor (37GB for 9B model). Skip eval: `eval_strategy="no"`

## Convergence Signals

**Phase 2 Vision GRPO** is converged when all of the following hold on Trackio smoothed trends (not individual log entries):
- `click_accuracy` smoothed mean > 0.9 with decreasing standard deviation
- `precision` > 0.95
- `carl_composite` plateaued (no upward trend for 20+ steps)
- `coordinate_format` > 0.8
- Completion length settled at 10-11 tokens
- Completion standard deviation > 0 (nonzero = model still producing diverse outputs, not collapsed)

**Phase 2 Vision SFT** converges at ~step 46 via PhaseTransitionGate: 3 out of last 5 steps above 0.99 `mean_token_accuracy` triggers the gate. The entropy spike (1.0 -> 9.3 -> 0.12) is the structural signature of convergence.

Never judge convergence from individual log entries. Use Trackio smoothed trends only.

## Pipeline Operations

- The pipeline orchestrator auto-submits GRPO when an SFT checkpoint appears on the Hub. Do NOT manually submit a GRPO job if the orchestrator is already running.
- GRPO loads the base model and merges the SFT adapter (`PeftModel.from_pretrained` + `merge_and_unload`), then applies a fresh LoRA for GRPO training.
- Cascade activates the CARL reward at step 50 (Stage B). Before step 50, only task rewards are active.

## HF Jobs Deployment

- The `carl-studio` wheel is hosted on an HF dataset repo and referenced in PEP 723 script metadata deps (e.g., `carl-studio @ https://huggingface.co/datasets/.../resolve/main/carl_studio-*.whl`).
- `PhaseTransitionGate` must inherit from `transformers.TrainerCallback` (not a plain class).
- Model card generation must be disabled (`push_to_hub_model_card=False` or equivalent) because HF Jobs uses an ASCII locale that breaks card rendering.
- The processor needs explicit `min_pixels` and `max_pixels` for Qwen VL image preprocessing (e.g., `AutoProcessor.from_pretrained(..., min_pixels=256*28*28, max_pixels=1280*28*28)`).
- Use `device_map="cuda:0"` not `device_map="auto"` — auto distributes layers across devices and breaks single-GPU LoRA training.
- Use `transformers` installed from git main (`transformers @ git+https://github.com/huggingface/transformers.git`) for latest Qwen3.5 fixes.

## File Layout

```
phase1/         # Tool-calling agent (SFT + GRPO)
phase2/         # VLM GUI grounding (Vision SFT + GRPO, unified pipeline)
phase3/         # Continuous inference agent (live_agent.py)
parameter-golf/ # OpenAI Parameter Golf submission
scripts/        # Dataset gen, eval, diagnostics
data/           # Training/eval data + vision cache
legacy/         # Archived deprecated scripts
```
