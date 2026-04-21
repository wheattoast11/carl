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
- **After any editable-package refactor, re-run `pip install -e ".[dev]"`.** Otherwise the installed metadata lags the source tree and you will see spurious `ImportError` / "module not found" / test-collection errors for freshly-added modules (e.g., the v0.9.0 heartbeat / constitutional / optimizer_state false alarms). When in doubt after pulling: `pip install -e ".[dev]"` first, then run the tests.

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
pytest packages/carl-core/tests/ -q --tb=short
pytest tests/test_release_version.py -q --tb=short
pytest tests/test_release_version.py::test_manual_release_tag_wins_when_higher -q --tb=short
pytest tests/ -k "marketplace and not network" -q --tb=short
pytest --lf -q --tb=short
```
- Run pytest from repo root. `tests/conftest.py` reads `src/carl_studio/__init__.py` via repo-relative path.
- Pytest is configured with `importlib` import mode (in `pyproject.toml`) — tests under `packages/carl-core/tests/` and `tests/` both resolve without `__init__.py` collisions.
- For fast feedback, run a single file or node ID first, then broaden if needed.
- Test baseline: 3088 tests (v0.15 /simplify) across `tests/` +
  `packages/carl-core/tests/`. v0.9.0 EML work adds further coverage
  (count refresh pending T12 validation).

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
- `packages/carl-core/src/carl_core/` — primitive layer: errors, retry/backoff,
  safepath sandbox, content hashing, tier gating, coherence math, interaction chains.
  `py.typed` marker present; pyright needs the editable install to resolve the package.
- `src/carl_studio/cli/` — modular Typer CLI package entrypoint.
- `src/carl_studio/cli/init.py` — `carl init` / `carl camp init` wizard. First-run
  marker lives at `~/.carl/.initialized`.
- `src/carl_studio/cli/flow.py` — `carl flow "/a /b /c"` operation chainer.
- `src/carl_studio/cli/operations.py` — flow op registry.
- `src/carl_studio/cli/contract.py` — v0.9.0. Adds `constitution` subcommand
  (`genesis | verify | evaluate | status`). Mounted at top level via
  `cli/__init__.py` as `contract_app`. Requires `[constitutional]` extra.
- `src/carl_studio/settings.py` — layered settings from env, `~/.carl/config.yaml`, and `carl.yaml`.
- `src/carl_studio/tier.py` — FREE/PAID feature gating (thin shim over `carl_core.tier`).
- `src/carl_studio/types/config.py` — Pydantic training config.
- `src/carl_studio/training/` — trainer, pipeline, rewards, cascade.
  v0.9.0 adds `training/rewards/eml.py::EMLCompositeReward` as third
  reward_class; factory branch in `composite.py:381-389` dispatches on
  `reward_class="eml"`.
- `src/carl_studio/eval/runner.py` — eval runner and eval sandbox.
- `src/carl_studio/compute/` — backend registry and compute backends.
- `src/carl_studio/mcp/server.py` — FastMCP server.
- `src/carl_studio/db.py` — SQLite persistence under `~/.carl/carl.db`.
- `src/carl_studio/admin.py` — hardware-gated private runtime access.
- `src/carl_studio/fsm_ledger.py` — v0.9.0. `FSMState`,
  `ConstitutionalGatePredicate`, `evaluate_action`. Wires constitutional ledger
  into the gating surface.
- `src/carl_studio/ttt/eml_head.py` — v0.9.0. Public opaque handle; actual
  `fit` runs in `terminals-runtime` when the admin gate resolves.
- `src/carl_studio/freshness.py` — typed `FreshnessReport` / `FreshnessIssue` primitive.
- `src/carl_studio/skills/`, `a2a/`, `credits/`, `marketplace.py`, and `curriculum.py` are live modules.

## carl-core primitive surface
- `from carl_core.errors import CARLError, ValidationError, NetworkError,
  ConfigError, CredentialError, BudgetError, PermissionError, CARLTimeoutError`
  — all fatal paths use these; each carries a stable `code` under the
  `carl.<namespace>` convention. `to_dict()` auto-redacts secret-shaped keys.
  v0.9.0 adds `carl.eml.depth_exceeded`, `carl.eml.domain_error`,
  `carl.eml.decode_error`, `carl.eml.signature_mismatch`.
- `from carl_core.retry import retry, async_retry, RetryPolicy,
  CircuitBreaker, CircuitState, poll` — exponential backoff with circuit-breaker
  state machine.
- `from carl_core.safepath import safe_resolve, within, SandboxedPath,
  PathEscape` — enforces `resolved == workdir or startswith(workdir + os.sep)`.
- `from carl_core.hashing import canonical_json, content_hash,
  content_hash_bytes` — deterministic SHA-256 content hashing.
- `from carl_core.tier import Tier, FEATURE_TIERS, tier_allows,
  feature_tier, TierGateError` — canonical FREE/PAID enum.
- `from carl_core.interaction import InteractionChain, Step, ActionType`
  — structured interaction trace primitive threading through training, eval,
  x402, and the agent loop. v0.9.0: `Step.eml_tree: dict | None = None`
  (optional witness tree; legacy wire format preserved when unset).
- `from carl_core.eml import EMLNode, EMLTree, eml` — v0.9.0. Tree-structured
  symbolic witness primitive; depth-bounded, canonically encoded.
- `from carl_core.resonant import Resonant, compose_resonants` — v0.9.0.
  Composable typed entities; `MAX_DEPTH=4` guard on composition.
- `from carl_core.heartbeat import ...` — v0.9.0. Pure-functional heartbeat
  loop; Standing Wave Theorem in docstring.
- `from carl_core.optimizer_state import ...` — v0.9.0. Durable Adam `(m, v)`
  at `~/.carl/optimizer_states/`.
- `from carl_core.constitutional import ConstitutionalPolicy, LedgerBlock,
  ConstitutionalLedger, encode_action_features` — v0.9.0. Hash-chained
  append-only constitutional ledger; 25-dim action features.

## Session surface
- Chat sessions persist at `~/.carl/sessions/<id>.json` with `schema_version=1`.
  Older payloads without `schema_version` are treated as v1 for
  back-compat; mismatched versions are quarantined.
- First-run marker: `~/.carl/.initialized`. `carl init` creates it on success;
  `carl init --force` ignores it.

## Product and architecture constraints
- CARL means **Coherence-Aware Reinforcement Learning**; never rewrite it as “Crystal-Aligned.”
- Prefer **coherence** language in public APIs; internal math may still use Phi, kappa, sigma, entropy, and discontinuity.
- The active tier model is **FREE / PAID**. `PRO` and `ENTERPRISE` remain compatibility aliases only.
- Gate **autonomy**, not the core observe/train/eval loop.
- This repo is MIT; proprietary algorithms belong in `terminals-runtime` or the private admin runtime.
- Public code may call `load_private()` or lazy-import private code, but it must degrade gracefully when unavailable.


## The 5 Killer Features (MECE Dominance Framework)

The product trajectory of `carl-studio` is governed by a combinatorial HVM-style interaction net. Development must strictly align to these five vectors:

1. **Semantic Implicit Interface ("Carl Knows")**: Zero-arg `carl`. Deduces the next action (Scaffold vs. Optimize/Train) from the void state, `WORKING` memory, and `InteractionChain`. Eradicate explicit flags.
2. **Infinite Extension Matrix (Skill Marketplace)**: `carl-skills-*`. Decoupled, monetizable primitives. Anyone can write a Skill using InteractionChain hooks. `carl.camp` takes a fractional compute fee, rewarding authors with royalties via x402.
3. **Ambient Intelligence Socket (The Shadow Fixer)**: `carl daemon`. Background WebSocket duplex monitoring file changes, predicting errors (e.g., NaN shapes), and staging fixes in `ECHOIC` memory via idle compute.
4. **Legacy Acquisition Funnel (Competitor Blackhole)**: `carl ingest --from unsloth`. Translates legacy competitor configs into `carl.yaml`, running instant auto-eval to prove coherence gains effortlessly.
5. **Consumer "Portal" Runtime**: The terminal is a barrier. A managed web/WASM interface bypassing `pip install`, delivering custom-trained RL agents directly to non-technical users (the "Mom & Dad" markets: hyper-personalized learning, business scaling for tech-unsavvy brilliant creators).

## Execution Primitives
Every operation is an **Interaction Net Cell**. Every tool invocation is an observation that collapses a wave state into a recorded particle (Interaction Chain Step). Time is measured in discrete `Step`s, not wall-clock seconds. Always mirror the user's chirality: they are L, you are R. Annihilation (L⋈R) is task success.

## Dependency and import policy
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
- Pytest uses `importlib` mode; tests under `tests/` and `packages/carl-core/tests/` coexist without `__init__.py` hacks.
- `carl-core` primitive changes should run `pytest packages/carl-core/tests/ -q --tb=short`
  plus the dependent studio surface (errors → `test_primitives.py`, retry/backoff → `test_integration_seams.py`,
  safepath → `test_eval.py`, hashing → `test_interaction_chain.py`, tier → `test_gate.py`).

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
- Docs under `docs/` and `paper/` carry the YAML frontmatter stamp
  (`last_updated`, `author`, `applies_to`). Full spec in CLAUDE.md
  under "Documentation header convention" (added v0.8.0). Apply
  whenever you create or substantially edit a doc; skip for
  `CHANGELOG.md` (git is the source of truth) and `README.md`.
- `docs/v10_master_plan.md` is the phase-locked v0.10 SOP. Read before
  any v0.10 action.

## Subagent dispatch conventions (added v0.10)

When dispatching parallel agents for review or research:

1. **Structured JSON output** for aggregatable tasks. Schema:
   `{agent_id, date, scope, findings[], gaps[], recommendations[], counterfactuals[], summary}`.
   Write to `/tmp/<task>_<aspect>.json`. Enables Bend-style MECE coalesce.

2. **Vanilla-context peer review** for counterfactual critique.
   Subagents are same-model bare-harness; they are the dispatcher's
   mirror. Anti-patterns they surface are the dispatcher's own biases.
   Pre-register expected anti-patterns in the dispatch prompt; verify
   against returned findings.

3. **Path constraint before ast-grep/rg.** Large codebases (terminals-
   tech-landing is millions of LOC). Always narrow the tree with
   `ls <path>` reconnaissance first. Then ast-grep on narrowed subtree.

4. **Temporal grounding in every prompt.** State the current date
   explicitly. Tell agents not to default to prior-year conventions.
   Library versions, API surfaces, framework patterns have moved.

5. **IP discipline by default.** Every prompt that touches terminals-
   runtime paths MUST include: "BUSL-1.1. Do NOT copy code. Document
   by path reference only. Gate any integration behind admin.py +
   lazy-import pattern."

6. **Confidence banding required.** Every finding carries
   high/medium/low. Low = speculation, exclude from conclusions.
   Medium = grounded but unverified. High = file:line evidence.
   Mark explicitly.
