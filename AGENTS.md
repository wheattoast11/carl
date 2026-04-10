# carl-studio

## Setup
```bash
pip install -e ".[all]"
```

## Test
```bash
pytest tests/ -q --tb=short
```

## Type Check
```bash
pyright src/carl_studio/types/ src/carl_studio/tier.py src/carl_studio/settings.py
```

## Code Style
- Pydantic v2 BaseModel for all configs and data structures
- Type hints on every function signature
- Constants are module-level in `primitives/constants.py`, never parameters
- `from __future__ import annotations` in every file
- Lazy imports for optional dependencies (torch, anthropic, runpod, etc.)
- CARL = **Coherence-Aware** Reinforcement Learning (NOT Crystal-Aligned)

## Key Files
- `src/carl_studio/primitives/constants.py` — kappa, sigma, threshold (NEVER modify)
- `src/carl_studio/primitives/coherence_trace.py` — per-token phi/entropy/delta_phi arrays
- `src/carl_studio/primitives/coherence_probe.py` — logits -> CoherenceTrace measurement
- `src/carl_studio/types/config.py` — TrainingConfig (root config type)
- `src/carl_studio/settings.py` — CARLSettings (layered YAML, env vars, presets)
- `src/carl_studio/tier.py` — FREE/PRO/ENTERPRISE gating
- `src/carl_studio/training/trainer.py` — CARLTrainer: async dispatch
- `src/carl_studio/training/rewards/composite.py` — CARLReward factory
- `src/carl_studio/training/cascade.py` — adaptive cascade gating
- `src/carl_studio/compute/hf_jobs.py` — HF Jobs backend (production-ready)
- `src/carl_studio/eval/runner.py` — Phase 1 + Phase 2' eval
- `src/carl_studio/cli.py` — 15+ Typer commands
- `src/carl_studio/mcp/server.py` — 9 MCP tools for AI agents
- `src/carl_studio/observe/app.py` — Textual TUI dashboard

## IP Boundaries — CRITICAL
- **MIT (this repo):** Conservation law, Phi, rewards, CLI, observe, eval, train
- **BUSL (terminals-runtime):** SLOT, resonance LR, Kuramoto, observer methodology
- **NEVER put proprietary algorithms in this repo.** Use stubs: `try: from terminals_runtime...`
- Stub files: `ttt/slot.py`, `training/lr_resonance.py`, `primitives/frame_buffer.py` (`_synchronization_index`), `primitives/coherence_observer.py` (`_load_observer_prompt`)

## Tier Model
- FREE: full CARL loop (observe, train, eval, BYOK compute)
- PRO: autonomous features (--live, --diagnose, --send-it, auto-gating)
- ENTERPRISE: MCP server, custom environments, orchestration
- Gate on **autonomy**, not capability

## Naming
- Public API: "coherence" (CoherenceProbe, CoherenceSnapshot, discontinuity)
- Internal math: physics names preserved (Phi, kappa, sigma, entropy, defect)
- Mapping: Defect->Discontinuity, Crystallization->Commitment, Melting->Dissolution

## Publishing
- PyPI: Trusted publishing via `.github/workflows/publish.yml`. Create GitHub release to trigger.
- HF wheel: Upload to `wheattoast11/zero-rl-tool-calling-data`
- Version in `pyproject.toml`. Bump before release.

## Gotchas
- `_safe_path`: Use `startswith(workdir + os.sep)` not bare `startswith(workdir)`
- `pyproject.toml`: `dependencies` must be under `[project]`, NOT under `[project.urls]`
- HF token: `huggingface_hub.get_token()` first, `os.environ` fallback
- `import carl_studio` must work with only pydantic + numpy + typer + anyio + pyyaml
- Crystal/coherence math uses numpy, never torch (convert at boundary)
- `torch.cuda.empty_cache()` in eval: once per sample, not per turn
