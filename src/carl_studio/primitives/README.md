# Primitives

Coherence math primitives that form the foundation of CARL. Every CARL metric is a reduction of the per-token coherence field defined here.

## Key Abstractions

- `CoherenceTrace` -- Per-token coherence field (phi, entropy, selected_prob, delta_phi). Constructed via `from_logits()` or `from_entropy()`. All CARL metrics derive lazily from this field.
- `CoherenceProbe` -- Observability probe that produces `CoherenceSnapshot` measurements at each training step: phi stats, defect counts, cloud quality, multi-scale coherence, discontinuity score.
- `CoherenceObserver` -- Higher-level observer that tracks coherence over time.
- `FrameBuffer` / `FrameRecord` -- Buffered frame storage for trace sequences.
- `compute_phi()` -- Computes the order parameter Phi from raw logits: `phi = 1 - H/log|V|`.

## Constants

Derived from the conservation law (not tuning parameters):

- `KAPPA = 64/3` -- conservation constant
- `SIGMA = 3/16` -- semantic quantum
- `DEFECT_THRESHOLD = 0.03` -- minimum |delta_phi| for discontinuity detection
- `T_STAR(d)` -- decompression boundary: `kappa * d`

## Architecture

This package is the lowest layer in the CARL stack. The training module consumes `CoherenceTrace` to compute rewards. The eval module consumes `CoherenceProbe` snapshots for quality gating. No dependencies on training, CLI, or network code.

## Usage

```python
from carl_studio.primitives import CoherenceTrace, compute_phi

phi, probs, entropy = compute_phi(logits)  # logits: [T, V] numpy array
trace = CoherenceTrace.from_logits(logits, vocab_size=32000)
print(trace.cloud_quality, trace.discontinuity_score)
```
