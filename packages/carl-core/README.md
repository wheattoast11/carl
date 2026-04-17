# carl-core

Coherence math primitives for CARL — the foundational per-token field that every CARL metric derives from.

## Install

```
pip install carl-core
```

## Use

```python
from carl_core import CoherenceTrace, compute_phi

# From logits:
trace = CoherenceTrace.from_logits(logits, token_ids=tokens, step=0)
print(trace.phi_mean, trace.cloud_quality, trace.carl_reward())

# Raw phi:
phi, probs, entropy = compute_phi(logits)
```

## What's inside

- `CoherenceTrace` — per-token coherence field (phi, entropy, selected_prob, delta_phi)
- `CoherenceProbe` / `CoherenceSnapshot` — probe + snapshot for training-step metrics
- `CoherenceObserver` — periodic Claude-backed training health assessment
- `FrameBuffer` / `FrameRecord` — rolling temporal Phi buffer
- `compute_phi()` — order parameter from logits: `phi = 1 - H/log|V|`
- Constants: `KAPPA = 64/3`, `SIGMA = 3/16`, `DEFECT_THRESHOLD`, `T_STAR(d)`
- `InteractionChain` / `Step` / `ActionType` — typed interaction trace

## Depends on

numpy, pydantic.

## Part of

Carl Studio — Coherence-Aware Reinforcement Learning.
Full docs: https://github.com/wheattoast11/carl
