---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.16.0
---

# Slime adapter

Backend adapter that routes `carl train --backend slime` to
[THUDM/slime](https://github.com/THUDM/slime) — an Apache-2.0 RL
post-training framework wiring Megatron-LM + SGLang. Slime is the
RL stack behind Z.ai's GLM-5 / 4.7 / 4.6 / 4.5 and is currently the
only verified OSS framework used for RL training on 100B+ parameter
MoE models.

## Why this adapter exists

Every other CARL adapter (TRL, Unsloth, Axolotl, Tinker, Atropos)
targets single-GPU / small-cluster workloads. Slime closes the only
remaining capability gap: production-grade GRPO on MoE at 100B+
scale. The shape of slime's rollout → data-buffer → training loop is
already isomorphic to CARL's `InteractionChain` — making this a
composition, not a compromise.

## Tier split

| Feature key | Tier | What it enables |
|---|---|---|
| `train.slime` | **FREE** | BYOK adapter. User brings Megatron/SGLang/GPUs. `backend: "slime"` in `carl.yaml` translates CARL config → slime args; submission spawns a local process. |
| `train.slime.rollout_bridge` | **FREE** | `SlimeRolloutBridge` wires slime's rollout + training callbacks into `InteractionChain`. Custom rewards (`EMLCompositeReward` / `CARLReward` / `PhaseAdaptiveCARLReward`) plug into slime's reward hook. |
| `train.slime.managed` | **PAID** | carl.camp orchestrates slime on managed multi-node compute. |
| `train.slime.moe_presets` | **PAID** | Presets for GLM-5 / DeepSeek-V3 / Qwen3-MoE at 100B+ scale (TP + PP + EP configured). |
| `train.slime.async_disaggregated` | **PAID** | Async mode with PD disaggregation + engine-group YAML templating. |

## Installation

Slime + Megatron-LM are **user-installed**. CARL does not silently
`pip install` a multi-gigabyte GPU-specific training stack.

1. Install the CARL rollout-side extra:

   ```bash
   pip install 'carl-studio[slime]'   # pulls sglang only
   ```

2. Follow [slime's official quick_start](https://thudm.github.io/slime/)
   to build slime and Megatron-LM against your CUDA / ROCm environment.

3. Verify availability:

   ```bash
   python -c "from carl_studio.adapters import SlimeAdapter; \
       a = SlimeAdapter(); print(a.available(), a.missing_dependencies())"
   ```

The adapter's `available()` checks for `slime`, `sglang`, and
`megatron` / `megatron.core` independently. If any are missing the
`missing_dependencies()` helper returns them by name so the CLI can
render a precise install hint.

## Minimal `carl.yaml`

```yaml
backend: slime
run_name: glm-pilot
base_model: Qwen/Qwen3-7B
dataset_repo: yourorg/grpo-prompts
dataset_split: train
method: grpo
mode: sync                    # or "async" (PAID: async_disaggregated)
max_length: 4096
max_completion_length: 1024
num_generations: 8
grpo_temperature: 1.0
learning_rate: 1.0e-6
max_steps: 1000
bf16: true
seed: 42

slime:
  tensor_parallel: 4
  pipeline_parallel: 1
  expert_parallel: 1
  rollout_engine_tp: 2
  advantage_estimator: grpo
  rollout_batch_size: 64
  num_rollout: 500
  # Optional — dotted path to a reward callable. The SlimeRolloutBridge
  # is the canonical way to wire CARL rewards here (see below).
  reward_fn: my_project.rewards:bridge_reward
```

## Coherence bridge

The adapter gets you a runnable slime submission. The
`SlimeRolloutBridge` is what makes the run **coherence-aware** — every
rollout completion becomes an `InteractionChain.Step` with phi/kuramoto_r
populated when a probe is registered, and CARL's reward heads
(`EMLCompositeReward`, `PhaseAdaptiveCARLReward`, `CARLReward`) plug
directly into slime's reward hook.

```python
from carl_core.interaction import InteractionChain
from carl_studio.training.rewards.eml import EMLCompositeReward
from carl_studio.training.slime_bridge import SlimeRolloutBridge

chain = InteractionChain()
bridge = SlimeRolloutBridge(
    chain,
    reward=EMLCompositeReward(),
    run_name="glm-pilot",
)

# Wire the bridge into slime's reward surface. The returned callable
# matches slime's --custom-reward-fn shape: (prompt, completion, **kwargs).
slime_reward_fn = bridge.as_slime_reward()

# Register slime_reward_fn via slime's reward-hook mechanism (the exact
# plumbing depends on your slime recipe / entry module; see slime docs).
```

## Launcher resolution

The adapter spawns slime through:

1. `slime.launcher_cmd` (explicit list of argv items) — **wins**.
2. `slime.launcher` == `"torchrun"` (default) → `torchrun --nproc_per_node=N train.py`
   with `N = slime.nproc_per_node` (default 1). Falls back to the
   current Python interpreter if `torchrun` is not on `PATH`.
3. `slime.launcher` == `"python"` → `sys.executable train.py`.
4. Any other `slime.launcher` value is treated as a whitespace-split
   prefix (e.g. `"accelerate launch"`).

The entry point is `slime.cli:main` by default. Override via
`slime.entry` in `carl.yaml` or the `SLIME_ENTRY` env var.

## Troubleshooting

| Symptom | Check |
|---|---|
| `AdapterError carl.adapter.unavailable` lists `megatron-lm` | Megatron-LM isn't on `PYTHONPATH`. Slime's docs require building Megatron from source — the CARL adapter intentionally does not pip-install it. |
| `AdapterError carl.adapter.translation` rejects `method: dpo` | Slime does not expose DPO. Supported methods: `sft`, `grpo`, `distill`. |
| Launcher runs `python` instead of `torchrun` | `torchrun` not on `PATH` — install PyTorch distributed bits or set `slime.launcher: python` explicitly. |
| Reward hook never fires | Check that slime's `--custom-reward-fn` resolves to `bridge.as_slime_reward()` via the dotted path you specified. The bridge records a Step per rollout, so an empty `InteractionChain` is a strong signal the hook isn't wired. |

## License and attribution

- Slime is Apache-2.0. The adapter is MIT (part of carl-studio).
- The adapter does not copy slime source; it only orchestrates subprocess
  launch. No license cross-contamination.
- Upstream: https://github.com/THUDM/slime (THUDM / Z.ai).
