# carl-studio

## Setup
```bash
uv pip install -e ".[dev]"
```

## Test
```bash
pytest tests/ -q --tb=short
```

## Type Check
```bash
pyright --strict src/carl_studio/types/
```

## Code Style
- Pydantic BaseModel for all configs and data structures
- Type hints on every function signature
- Constants are module-level in `primitives/constants.py`, never parameters
- `from __future__ import annotations` in every file
- Lazy imports for optional dependencies (torch, anthropic, runpod, etc.)

## Key Files
- `src/carl_studio/primitives/constants.py` — κ, σ, threshold (NEVER modify)
- `src/carl_studio/primitives/coherence_probe.py` — THE source of truth for coherence math
- `src/carl_studio/primitives/math.py` — shared compute_phi() (single source, used by all reward components)
- `src/carl_studio/primitives/coherence_observer.py` — Claude-in-the-loop diagnostics
- `src/carl_studio/types/config.py` — all config types (TrainingConfig is the root)
- `src/carl_studio/training/trainer.py` — CARLTrainer: async dispatch to remote or local
- `src/carl_studio/training/rewards/composite.py` — CARLReward + make_carl_reward factory
- `src/carl_studio/training/rewards/vlm.py` — VLM rewards (click_accuracy, coordinate_format, precision)
- `src/carl_studio/compute/protocol.py` — ComputeBackend interface
- `src/carl_studio/bundler.py` — self-contained HF Jobs script generator
- `src/carl_studio/mcp/server.py` — 9 MCP tools for AI agent consumption
- `paper/carl-paper.md` — formal research paper

## Boundaries
- Never modify constants (KAPPA, SIGMA, DEFECT_THRESHOLD)
- Bundled scripts (from bundler.py) must be self-contained — no carl_studio imports
- Backend implementations use lazy imports — don't require all SDKs installed
- `import carl_studio` must work with only pydantic + numpy + typer + anyio + pyyaml
- Crystal/coherence math uses numpy, never torch (convert at boundary)
- Qwen3.5 thinking mode must be disabled for short-output tasks (coordinates, etc.)
- GRPO num_generations=8 minimum — 4 causes zero-advantage zero-gradient

## Naming
- Public API: "coherence" (CoherenceProbe, CoherenceSnapshot, discontinuity)
- Internal math: physics names preserved (Φ, κ, σ, entropy, defect)
- The mapping: Crystal→Coherence, Defect→Discontinuity, Crystallization→Commitment, Melting→Dissolution
