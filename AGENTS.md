# carl-studio agent guide

Execution guide for autonomous coding agents in this repo.
Prefer code truth over docs; keep patches small and validation scoped.

## Truth sources
- `pyproject.toml`: packaging, deps, Ruff, Pyright.
- `README.md`: user-facing install and usage.
- `CLAUDE.md`: concise project memory for agents.
- Read the touched module and adjacent tests before changing behavior.
- There are no Cursor rules in `.cursor/rules/` or `.cursorrules`.
- There is no `.github/copilot-instructions.md`.
- There is no `Makefile`, `tox.ini`, `pytest.ini`, `ruff.toml`, or `.editorconfig`.

## Working norms
- Run commands from repo root: `/Users/terminals/Documents/agents/models/carl-studio`.
- Prefer surgical edits over broad cleanup.
- Do not fix unrelated lint/type debt unless asked.
- Update docs only when code truth changed or docs are misleading.
- Validate only the touched surface unless the change crosses modules.

## Setup
```bash
pip install -e ".[dev]"
pip install -e ".[all]"
```
- Use `.[all]` only when touching optional backends, training extras, MCP, or observe UI.

## Build
```bash
pip install build
python -m build
```
- Build backend is `hatchling`.
- `.github/workflows/publish.yml` also uses `python -m build`.

## Test
```bash
pytest tests/ -q --tb=short
pytest tests/test_release_version.py -q --tb=short
pytest tests/test_release_version.py::test_manual_release_tag_wins_when_higher -q --tb=short
pytest tests/ -k "marketplace and not network" -q --tb=short
pytest --lf -q --tb=short
```
- Run pytest from repo root. `tests/conftest.py` reads `src/carl_studio/__init__.py` via repo-relative path.
- For fast feedback, run a single file or node ID first, then broaden if needed.

## Lint and format
```bash
ruff check src/carl_studio tests
ruff check path/to/changed_file.py
ruff format src/carl_studio tests
ruff format path/to/changed_file.py
```
- Ruff target version is Python 3.11; line length is 100.
- Repo-wide Ruff currently reports pre-existing issues. Prefer targeted runs on changed files.

## Type check
```bash
pyright src/carl_studio/types/ src/carl_studio/tier.py src/carl_studio/settings.py
pyright src/carl_studio/<changed_module>.py
```
- Pyright runs in `strict` mode.
- The baseline targeted command currently reports existing issues in `settings.py`, `tier.py`, and some type modules.
- Treat those as pre-existing debt unless your change is in that area.

## High-value repo map
- `src/carl_studio/cli/` — modular Typer CLI package entrypoint.
- `src/carl_studio/settings.py` — layered settings from env, `~/.carl/config.yaml`, and `carl.yaml`.
- `src/carl_studio/tier.py` — FREE/PAID feature gating.
- `src/carl_studio/types/config.py` — Pydantic training config.
- `src/carl_studio/training/` — trainer, pipeline, rewards, cascade.
- `src/carl_studio/eval/runner.py` — eval runner and eval sandbox.
- `src/carl_studio/compute/` — backend registry and compute backends.
- `src/carl_studio/mcp/server.py` — FastMCP server.
- `src/carl_studio/db.py` — SQLite persistence under `~/.carl/carl.db`.
- `src/carl_studio/admin.py` — hardware-gated private runtime access.
- `src/carl_studio/skills/`, `a2a/`, `credits/`, `marketplace.py`, and `curriculum.py` are live modules.

## Product and architecture constraints
- CARL means **Coherence-Aware Reinforcement Learning**; never rewrite it as “Crystal-Aligned.”
- Prefer **coherence** language in public APIs; internal math may still use Phi, kappa, sigma, entropy, and discontinuity.
- The active tier model is **FREE / PAID**. `PRO` and `ENTERPRISE` remain compatibility aliases only.
- Gate **autonomy**, not the core observe/train/eval loop.
- This repo is MIT; proprietary algorithms belong in `terminals-runtime` or the private admin runtime.
- Public code may call `load_private()` or lazy-import private code, but it must degrade gracefully when unavailable.

## Dependency and import policy
- Keep `import carl_studio` lightweight.
- Do not introduce eager imports of torch, transformers, anthropic, textual, mcp, or other heavy optional packages into lightweight paths.
- Use lazy imports inside functions or guarded branches for optional dependencies.
- If an extra is required, fail close to the use site with a clear install hint.
- When touching HF auth paths, prefer `huggingface_hub.get_token()` first and use `HF_TOKEN` as fallback.
- Do not add implicit `.env` loading.

## Code style
- Add `from __future__ import annotations` at the top of every Python file.
- Import order: standard library, third-party, local project imports.
- Use modern built-in generics: `list[str]`, `dict[str, Any]`, `str | None`.
- Use `TYPE_CHECKING` for typing-only imports that would otherwise create cycles or pull in heavy deps.
- Use Pydantic v2 `BaseModel` / `BaseSettings`, `Field`, `field_validator`, and `model_validator`.
- Use `default_factory` for mutable defaults.
- Use enums for constrained string values.
- Keep constants at module scope; physics constants belong in `src/carl_studio/primitives/constants.py`.
- Preserve public docstrings for modules, classes, and user-facing functions.
- Prefer `model_dump`, `model_copy`, and `model_validate_json` over ad-hoc dict/JSON plumbing.
- Prefer `pathlib.Path` for filesystem work.
- Use `yaml.safe_load` for YAML reads.
- Never persist secrets to YAML.

## Naming and filesystem rules
- Classes: `PascalCase`; functions, methods, modules: `snake_case`; constants: `UPPER_SNAKE_CASE`.
- User state belongs under `~/.carl`.
- For sandboxed file access, require `resolved == workdir or resolved.startswith(workdir + os.sep)`.
- Never use bare `resolved.startswith(workdir)`.

## Error handling and CLI conventions
- Library code should raise specific exceptions such as `ValueError`, `RuntimeError`, `PermissionError`, or domain errors like `MarketplaceError`.
- Preserve exception chaining with `raise ... from exc`.
- CLI code should use `CampConsole` for user-facing output and terminate with `typer.Exit` for exit codes.
- Keep optional-dependency handling local: catch `ImportError` near the call site and print the relevant extra/install guidance.
- Avoid swallowing broad `Exception` unless the code is intentionally best-effort.
- Never log, print, or persist secrets.
- Existing network clients often use stdlib `urllib`; preserve that bias unless there is a strong reason not to.

## Testing guidance
- Start with targeted tests, then expand only if the change crosses modules.
- Prefer mocks and monkeypatching over live HF, Anthropic, Supabase, or browser calls.
- Keep tests CPU-only and offline-safe when possible.
- `tests/conftest.py` stubs heavy imports so lightweight modules can run without torch/transformers; do not break that bootstrap path.
- When changing packaging or release logic, always run `pytest tests/test_release_version.py -q --tb=short` and `python -m build`.

## Deterministic agent procedure
1. Classify the request: docs, config, CLI, core math, platform I/O, packaging, or optional-dependency boundary.
2. Read the smallest complete truth set: touched file, adjacent tests, `pyproject.toml`, then relevant docs.
3. Preserve public behavior unless the user explicitly asked for a behavior change.
4. Validate at the smallest scope that proves the edit:
   - docs only -> no tests unless commands/examples changed
   - settings/tier/config -> targeted tests + targeted Pyright
   - CLI -> targeted tests + targeted Ruff
   - packaging/release -> release-version tests + build
   - optional dependency boundary -> focused import or CLI smoke test
5. If unrelated baseline failures appear, report them separately and keep the patch narrow.
6. Update docs only when code truth changed or docs are misleading.

## Documentation policy
- `AGENTS.md` is the execution playbook for coding agents.
- `CLAUDE.md` should stay concise, current, and code-truthful.
- `README.md` is user-facing; change it only for user-visible behavior, setup, or workflow changes.
