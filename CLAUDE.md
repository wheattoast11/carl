# CARL Studio project memory

Project-specific memory for agents working in this repository.
Keep this file short, current, and grounded in code truth.

## What this repo is

- `carl-studio` is an MIT-licensed Python package and CLI for
  **Coherence-Aware Reinforcement Learning**.
- Python requirement: `>=3.11`.
- `pyproject.toml` is the source of truth for packaging, dependencies, Ruff, and Pyright.
- `README.md` is for users. `AGENTS.md` is the agent execution guide.

## Core commands

```bash
pip install -e ".[dev]"
pip install -e ".[all]"
pytest tests/ -q --tb=short
pytest tests/test_release_version.py::test_manual_release_tag_wins_when_higher -q --tb=short
ruff check path/to/changed_file.py
pyright src/carl_studio/<changed_module>.py
python -m build
```

Run pytest from the repo root. `tests/conftest.py` depends on repo-relative paths.

## Architecture snapshot

- `src/carl_studio/__init__.py` keeps top-level imports light and lazy-loads heavy modules.
- `src/carl_studio/primitives/` — coherence math, traces, probes, observers.
- `src/carl_studio/types/config.py` — Pydantic training config surface.
- `src/carl_studio/training/` — trainer, pipeline, rewards, cascade.
- `src/carl_studio/eval/runner.py` — eval runner and sandbox.
- `src/carl_studio/compute/` — backend registry and compute backends.
- `src/carl_studio/cli/` — modular Typer CLI package entrypoint.
- `src/carl_studio/mcp/server.py` — FastMCP server.
- `src/carl_studio/db.py` — local SQLite state under `~/.carl`.
- `src/carl_studio/settings.py` — layered config from env, `~/.carl/config.yaml`, and `carl.yaml`.
- `src/carl_studio/admin.py` — hardware-gated access to the private runtime.
- `src/carl_studio/chat_agent.py` — CARLAgent agentic loop (streaming, tools, sessions, cost, permissions).
- `src/carl_studio/frame.py` — WorkFrame analytical lens (domain/function/role/objectives).
- `src/carl_studio/contract.py` — service contract witnessing (SHA-256 hash chain).
- `src/carl_studio/consent.py` — privacy-first consent state machine (all flags off by default).
- `src/carl_studio/x402.py` — HTTP 402 payment rail client (facilitator-based, no web3.py).
- `src/carl_studio/carlito.py` — small specialized agents spawned from graduated curricula.
- `src/carl_studio/camp.py` — CampProfile managed-account contract for billing/credits.
- `skills/`, `a2a/`, `credits/`, `marketplace.py`, and `curriculum.py` are live code paths.

## Product and licensing boundaries

- CARL means **Coherence-Aware Reinforcement Learning**. Do not rename it.
- Public API language should prefer **coherence** terms.
- Internal math can still use Phi, kappa, sigma, entropy, and discontinuity.
- Active tier model in code is **FREE / PAID**.
- `PRO` and `ENTERPRISE` are compatibility aliases, not separate active tiers.
- Gate **autonomy**, not the core training/eval capability.
- Proprietary algorithms belong in `terminals-runtime` or the private admin runtime,
  not in this repo.

## Dependency policy

- Keep `import carl_studio` light. Do not make torch/transformers/anthropic/textual/mcp
  mandatory at import time for lightweight modules.
- Use lazy imports for optional dependencies.
- Emit clear install hints when an optional dependency is missing.
- Prefer `huggingface_hub.get_token()` before falling back to `HF_TOKEN` when touching
  backend auth flows.
- Do not add implicit `.env` loading.

## Code conventions

- `from __future__ import annotations` at the top of each Python file.
- Import order: stdlib, third-party, local.
- Modern type syntax: `list[str]`, `dict[str, Any]`, `str | None`.
- Pydantic v2 models for configs and structured data.
- `default_factory` for mutable defaults.
- Constants stay module-level; physics constants live in `primitives/constants.py`.
- Prefer `Path` over string path manipulation.
- For sandboxed paths, require:
  `resolved == workdir or resolved.startswith(workdir + os.sep)`.

## Error handling and CLI behavior

- Library code raises explicit exceptions.
- Preserve `raise ... from exc` when wrapping failures.
- CLI code should use `CampConsole` and `typer.Exit`, not ad-hoc `print` + return codes.
- Never log or persist secrets.
- Keep network clients dependency-light unless there is a compelling reason otherwise.

## Validation policy

- Docs-only change -> no tests unless commands/examples changed.
- Settings/tier/config change -> targeted tests + targeted Pyright.
- CLI change -> targeted tests + targeted Ruff.
- Packaging/release change -> `tests/test_release_version.py` + `python -m build`.
- Avoid turning unrelated lint/type debt into drive-by cleanup unless asked.

## Current repo truths

- There are no Cursor rules or Copilot instruction files in this repo.
- There is no `Makefile`, `tox.ini`, `pytest.ini`, `ruff.toml`, or `.editorconfig`.
- Publish workflow lives in `.github/workflows/publish.yml` and uses `python -m build`.
- `python -m build` works.
- Single pytest node IDs work from the repo root.
- Repo-wide Ruff and Pyright currently have pre-existing noise; validate touched files first.
- Test baseline: 1103 tests, ~4s. `tests/test_uat_e2e.py` is the E2E UAT suite.

## Claude API integration (chat_agent.py)

- The agent module is `chat_agent.py`, NOT `agent.py` — `agent/` is a package (FSM agent).
- `anthropic.Anthropic(api_key="")` blocks env var fallback. Always pass `api_key or None`.
- Adaptive thinking (`thinking: {"type": "adaptive"}`) is Opus 4.6 / Sonnet 4.6 only.
  Haiku 4.5 and Claude 3 will 400. Check model before adding.
- Model IDs use short form: `claude-opus-4-6`, `claude-sonnet-4-6`. No date suffixes.
- CLI commands lazy-import from source modules inside function bodies. Patch at
  `carl_studio.<module>.<Class>`, not `carl_studio.cli.<module>.<Class>`.
- `SourceIngester.ingest()` on an empty directory raises `ValueError`, not empty list.
- Sessions persist at `~/.carl/sessions/`. Knowledge `words` are sets — serialize as sorted lists.
- Anthropic SDK: >=0.95.0 required for `cache_control` top-level param and streaming.

## Keep an eye on

- Preserve optional dependency boundaries.
- Preserve import-time lightness.
- Keep docs current and minimal.
- If docs disagree with code, fix the docs or explicitly note the mismatch.
