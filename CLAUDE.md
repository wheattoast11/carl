# CARL Studio — carl-studio

**CARL** (Coherence-Aware Reinforcement Learning) — MIT-licensed SDK and CLI.
**PyPI:** `pip install carl-studio` | **Repo:** [wheattoast11/carl](https://github.com/wheattoast11/carl) | **Author:** Tej Desai / Intuition Labs LLC

## Architecture

4 layers: primitives → types → training → CLI/MCP

| Layer | Location | Purpose |
|-------|----------|---------|
| Primitives | `primitives/` | Phi, CoherenceProbe, CoherenceTrace, FrameBuffer |
| Types | `types/` | TrainingConfig, EvalConfig, TrainingRun (Pydantic v2) |
| Training | `training/` | CARLTrainer, CascadeRewardManager, rewards, callbacks |
| CLI | `cli.py` | `carl observe/train/eval/status/logs/config/...` (Typer + Rich) |

## IP Boundary — CRITICAL

| What | Where | License |
|------|-------|---------|
| Conservation law, Phi, rewards, CLI | **This repo** | MIT |
| SLOT, resonance LR, Kuramoto, observer methodology | `terminals-runtime` | BUSL-1.1 |

**NEVER put proprietary algorithms in this repo.** Use stubs that `try: from terminals_runtime...` with graceful fallback. The following files are stubs:
- `ttt/slot.py` — delegates to `terminals_runtime.ttt`
- `training/lr_resonance.py` — delegates to `terminals_runtime.training`
- `primitives/frame_buffer.py` `_synchronization_index()` — delegates to `terminals_runtime.primitives`
- `primitives/coherence_observer.py` `_load_observer_prompt()` — delegates to `terminals_runtime.observe`

## Tier Model

- **FREE:** observe, train, eval, bench, align, learn, push, bundle, BYOK compute
- **PRO:** `--live` TUI, `--diagnose`, `--send-it`, auto-gating, scheduled runs
- **ENTERPRISE:** MCP server, custom environments, orchestration, RBAC
- Gate on **autonomy**, not capability. `tier.py` + `settings.py` enforce this.

## Publishing

- PyPI: Trusted publishing via `.github/workflows/publish.yml`. Create GitHub release → auto-publishes.
- HF wheel: Upload to `wheattoast11/zero-rl-tool-calling-data` for training script consumption.
- Version in `pyproject.toml`. Bump before release.

## Commands

```bash
pip install -e ".[all]"       # Dev install with all extras
python -m pytest tests/       # Run tests (71+ tests, no GPU needed)
carl observe --url <trackio>  # One-shot crystal metrics
carl train --dry-run           # Preview training config
carl eval --adapter <hub-id>   # Run eval
carl config show               # Display settings
```

## Gotchas

- **CARL = Coherence-Aware, NOT Crystal-Aligned.** If you see "Crystal-Aligned" anywhere, fix it.
- **pyproject.toml:** `dependencies` must be under `[project]`, NOT under `[project.urls]`. Hatchling silently misparses otherwise.
- **`_safe_path`:** Always use `startswith(workdir + os.sep)`, never bare `startswith(workdir)`.
- **Pyright strict mode:** Enabled. TRL/transformers dynamic imports show false positives — these are runtime-correct.
- **`empty_cache()` in eval loops:** Once per sample, not per turn.
- **Settings:** `CARLSettings.load()` reads YAML on every call. Cache if called repeatedly.
- **HF token:** Use `huggingface_hub.get_token()` first, `os.environ` fallback. Token lives in HF credentials, not shell env.

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | 15+ Typer commands |
| `settings.py` | CARLSettings (Pydantic BaseSettings, layered YAML) |
| `tier.py` | Freemium tier system (FREE/PRO/ENTERPRISE) |
| `primitives/coherence_trace.py` | Per-token Phi/entropy/delta_phi arrays |
| `training/cascade.py` | Adaptive cascade gating (self-calibrating) |
| `training/rewards/composite.py` | CARLReward factory (0.5 coherence + 0.3 cloud + 0.2 discontinuity) |
| `compute/hf_jobs.py` | HF Jobs backend (provision, execute, status, logs, stop) |
| `eval/runner.py` | Phase 1 + Phase 2' eval with 3-format tool call parser |
| `observe/app.py` | Textual TUI dashboard |
| `environments/builtins/` | CodeSandboxEnv, SQLSandboxEnv |
