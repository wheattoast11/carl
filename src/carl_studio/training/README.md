# Training

Training pipeline, reward functions, and cascade staging for coherence-aware RL.

## Key Abstractions

- `CARLTrainer` -- Orchestrates SFT and GRPO training runs. Remote targets generate bundled scripts submitted to HF Jobs; local targets run TRL in-process. All heavy imports (torch, transformers, trl) are lazy.
- `SendItPipeline` -- Full autonomous lifecycle: validate -> SFT -> eval gate -> GRPO -> eval gate -> push to Hub. Emits `PipelineEvent` objects for progress tracking.
- `CascadeRewardManager` -- Adaptive stage gating. Fixed mode transitions at a set step; adaptive mode self-calibrates from the metric's running distribution.
- `CARLReward` -- Composite reward: 50% multiscale coherence + 30% cloud quality + 20% discontinuity. Built on `CoherenceTrace` (single forward pass).
- `CascadeCallback` / `ResonanceLRCallback` / `CoherenceMonitorCallback` -- Trainer callbacks for cascade transitions, LR scheduling, and coherence logging.

## Rewards (`rewards/`)

Task rewards (R1-R5) for tool-calling RL training:

- `tool_call_format_reward` -- structural correctness (5-level)
- `tool_selection_reward` -- correct tool with partial credit
- `chain_completion_reward` -- multi-step chain progress
- `neuralese_v2_reward` -- action density ratio
- `conciseness_reward` -- linear length penalty

All follow the TRL GRPO signature: `def reward_fn(completions, **kwargs) -> list[float]`.

## Architecture

Depends on `primitives` for coherence math and `types` for config/run models. Compute backends are injected via `compute/`. The pipeline is invoked from the CLI (`carl train --send-it`) or programmatically.

## Usage

```python
from carl_studio import CARLTrainer, TrainingConfig
trainer = CARLTrainer(config)
run = await trainer.train()
```
