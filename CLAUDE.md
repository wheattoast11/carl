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

- `packages/carl-core/` — primitive layer (own pyproject, own tests). Owns
  `errors`, `retry`, `safepath`, `hashing`, `tier`, coherence math, interaction
  chain. `py.typed` marker published; pyright needs the editable install.
- `src/carl_studio/__init__.py` keeps top-level imports light and lazy-loads heavy modules.
- `src/carl_studio/primitives/` — removed. All in-tree consumers migrated to `carl_core.*` imports. Downstream callers must import from `carl_core` directly.
- `src/carl_studio/freshness.py` — typed `FreshnessReport`/`FreshnessIssue` with stable issue codes under `carl.freshness.*`.
- `src/carl_studio/types/config.py` — Pydantic training config surface.
- `src/carl_studio/training/` — trainer, pipeline, rewards, cascade.
- `src/carl_studio/eval/runner.py` — eval runner and sandbox.
- `src/carl_studio/compute/` — backend registry and compute backends.
- `src/carl_studio/cli/` — modular Typer CLI package entrypoint.
- `src/carl_studio/cli/init.py` — `carl init` / `carl camp init` one-shot wizard. First-run marker `~/.carl/.initialized`.
- `src/carl_studio/cli/flow.py` — `carl flow "/a /b /c"` operation chainer (trace → `~/.carl/interactions/<id>.jsonl`).
- `src/carl_studio/cli/operations.py` — flow op registry (doctor, start, init, ask, chat, flow, ship, review, simplify, train, eval, infer, publish, push, diagnose).
- `src/carl_studio/mcp/server.py` — FastMCP server.
- `src/carl_studio/db.py` — local SQLite state under `~/.carl`.
- `src/carl_studio/settings.py` — layered config from env, `~/.carl/config.yaml`, and `carl.yaml`.
- `src/carl_studio/admin.py` — hardware-gated access to the private runtime.
- `src/carl_studio/chat_agent.py` — CARLAgent agentic loop (streaming, tools, sessions, cost, permissions). Sessions persist with `schema_version=1`.
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
- Test baseline: 1556 tests in `tests/` (`~40s` full run) plus the `carl-core` suite under `packages/carl-core/tests/`. `tests/test_uat_e2e.py` is the E2E UAT suite.
- Pytest uses `importlib` import mode; `tests/` and `packages/carl-core/tests/` coexist without `__init__.py` collisions.

## Production hardening patterns (post-v0.3.0)

- All fatal paths route to `carl_core.errors.CARLError` subclasses with stable
  `code` values under the `carl.<namespace>` convention (`carl.config`,
  `carl.validation`, `carl.credential`, `carl.network`, `carl.budget`,
  `carl.permission`, `carl.timeout`, `carl.freshness.*`). `to_dict()` auto-redacts
  secret-shaped keys.
- `InteractionChain` threads through training (`training/pipeline.py`), eval
  (`eval/runner.py`), x402 (`x402.py`), and the chat agent
  (`chat_agent.py`) — every durable operation records a step with action type,
  input snapshot, output, success flag, and duration.
- `carl_core` is the primitive boundary: import from `carl_core.errors`,
  `carl_core.retry`, `carl_core.safepath`, `carl_core.hashing`,
  `carl_core.tier`, and `carl_core.interaction` directly. The legacy
  `carl_studio.primitives` shim has been removed (was marked for v0.5.0).
- `py.typed` marker ships on `carl-core`. Pyright will not see the package
  unless it has the editable install (`pip install -e packages/carl-core`),
  which `pip install -e ".[dev]"` handles transitively via the workspace.
- Freshness issues use a structured primitive (`FreshnessReport` /
  `FreshnessIssue`) with stable `code` and `severity`/`category`/`remediation`
  fields. Legacy string-list views (`stale_packages`, `config_warnings`,
  `credential_warnings`) remain for back-compat.

## Claude API integration (chat_agent.py)

- The agent module is `chat_agent.py`, NOT `agent.py` — `agent/` is a package (FSM agent).
- `anthropic.Anthropic(api_key="")` blocks env var fallback. Always pass `api_key or None`.
- Adaptive thinking (`thinking: {"type": "adaptive"}`) is Opus 4.6 / Sonnet 4.6 only.
  Haiku 4.5 and Claude 3 will 400. Check model before adding.
- Model IDs use short form: `claude-opus-4-6`, `claude-sonnet-4-6`. No date suffixes.
- CLI commands lazy-import from source modules inside function bodies. Patch at
  `carl_studio.<module>.<Class>`, not `carl_studio.cli.<module>.<Class>`.
- `SourceIngester.ingest()` on an empty directory raises `ValueError`, not empty list.
- Sessions persist at `~/.carl/sessions/` with `schema_version=1`. Knowledge `words` are sets — serialize as sorted lists.
- Anthropic SDK: >=0.95.0 required for `cache_control` top-level param and streaming.

## CLI routing (as of 2026-04-17)

| User invocation | Routes to | Behavior |
|---|---|---|
| `carl` (bare) | `cli/chat.py:chat_cmd` via `_default_to_chat` callback | Full CARLAgent loop. |
| `carl chat` | `cli/chat.py:chat_cmd` | Full CARLAgent loop (same as bare). |
| `carl "<prompt>"` | `cli/chat.py:ask_cmd` → `run_one_shot_agent` | One-shot agent — single Anthropic call with tools. |
| `carl ask "<prompt>"` | same as `carl "<prompt>"` | Alias. |
| `carl flow "/a /b /c"` | `cli/flow.py:flow_cmd` → `cli/operations.OPERATIONS` | Chains ops; trace persisted to `~/.carl/interactions/<id>.jsonl`. |
| `carl init` | `cli/init.py:init_cmd` | One-shot wizard. First-run marker `~/.carl/.initialized`. |
| `carl camp init` | `cli/init.py:init_cmd` (same callable) | Wired under `camp_app` as alias. |
| `carl doctor` | `cli/startup.py:doctor` | Readiness + typed freshness report. |
| `carl lab repl` | `cli/lab.py:chat_repl` | Simple REPL, no tool use (legacy). |
| `carl lab curriculum` / `carl lab carlito` | `cli/lab.py` | Canonical paths (not top-level). |

- `carl lab chat` no longer exists — it was renamed to `carl lab repl`.
- `settings.py` defaults: `default_model=""`, `naming_prefix=""` — user must configure.
- CLI `wiring.py` stubs print install hints when extras are missing (not silent `pass`).

## Documentation header convention (as of 2026-04-20 · v0.8.0)

All new or meaningfully-updated docs under `docs/` and `paper/` carry a
YAML frontmatter stamp so readers and auditors can tell the provenance
at a glance. Keep it minimal; do not add fields that would decay.

```markdown
---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.8.0
---
```

- `last_updated` — ISO date of the most recent meaningful edit (not
  typo fixes). Update whenever you touch a section that changes meaning.
- `author` — who did the edit. If co-authored with an agent, list both.
- `applies_to` — the release the doc reflects. Bump on each tagged
  release when the doc's claims are still accurate; if the doc goes
  stale, either fix it or add a `stale_since` field noting the delta.

**Scope.** Apply to `docs/*.md`. Do NOT apply to:
- `CHANGELOG.md` — git history is the source of truth.
- `README.md` — lives at the repo root with its own metadata block.
- `paper/*.md` — papers carry full scholarly frontmatter (`authors`,
  `date`, `keywords`, `license`) which covers the same provenance
  surface. Don't double-stamp.
- In-tree code comments — docstrings and module comments are not docs.

The goal is provenance for operational docs that describe shipped
behavior, not narration on every file.

When a doc falls more than one minor version behind current, treat it
as a review candidate: either update and re-stamp, or delete.

## v0.9 + v0.10 roadmap (docs landed 2026-04-20, implementation pending Tej sign-off)

Four design docs under `docs/v09_*.md` + `docs/v10_*.md`:

- `v09_carl_update_design.md` — self-updating meta-pipeline (`carl update`)
  with 3-day staleness nudge, PyPI/CVE/git-delta scan, positive-framed
  blast-radius reporting. Zero new deps.
- `v09_carl_env_design.md` — 7-question progressive-disclosure env
  wizard (`carl env`) with functor composition + resume-capable state.
  Verifier hook reserved for prime-rl integration.
- `v09_terminals_runtime_integration_matrix.md` — IP-respecting lazy-import
  matrix for 10 terminals-runtime primitives. Three v0.9-A picks:
  Kuramoto-R order parameter, conservation-law token budget, upgraded
  OBSERVER_SYSTEM_PROMPT.
- `v10_terminals_tech_deep_dive.md` — grounded review of 5 terminals-tech
  paths + 4 research papers. **Two corrections to the v0.9 matrix:**
  (1) `agent-sdk` is MIT (not BUSL), unblocking agent-card Supabase work
  to v0.10-A; (2) `semantic-mesh/convergence` is NOT Kuramoto — carl-core
  is already ahead. Three v0.10-A picks: TerminalAgent mirror + Supabase,
  py2bend reward compilation (admin-gated BUSL), Substrate presence probe.

## κ-constant ruling (resolved 2026-04-20)

Tej ruled: `KAPPA = 64/3 ≈ 21.333` is the canonical exact value,
derived from the early Desai papers (Zenodo 10.5281/zenodo.18906944,
18992031). terminals.tech's `κ = 21.37` is a downstream calibration
approximation — CARL keeps the exact ratio. **Do not change
`packages/carl-core/src/carl_core/constants.py:14`.**

## Mental model: AXON signal vocabulary (emit isomorphic events)

terminals-tech defines 67 AXON SignalTypes in `lib/terminals-tech/core/base/events.ts`.
AXON is TypeScript-only (no Python bindings), so carl-studio **cannot directly
subscribe** — but it SHOULD emit events with **isomorphic shapes** so that when
terminals-tech web app observes carl-studio runs (via carl.camp HTTP forwarding),
the shapes align natively.

Top 5 signals carl-studio should emit during training:
- `skill_training_started` — phase transition into learning
- `skill_crystallized` — reward crystallized, artifact learnable
- `coherence_update` — internal consistency observability
- `interaction_created` — per-episode lifecycle marker
- `action_dispatched` — fine-grained trajectory reconstruction

See `docs/v10_terminals_tech_deep_dive.md` for the full 67-signal taxonomy.
`packages/carl-core/src/carl_core/interaction.py::Step` is already shape-compatible
with the AXON signal payload format — the mapping is direct.

## Agent-card ↔ Supabase flow (v0.10-A #1, spec landed)

`docs/v10_agent_card_supabase_spec.md` is implementation-ready. Flow:

```
FREE tier: carl agent register → LocalDB → FYI side-notice
PAID tier: carl agent register → LocalDB → HTTP POST carl.camp → Supabase upsert
```

carl-studio (Python CLI) **never calls Supabase directly**. The carl.camp
backend mediates via the `electric-bridge.ts::pushToSupabase` pattern (verified
at `/Users/terminals/Documents/terminals-tech-landing/terminals-landing-new/lib/sync/`).

Conflict strategy: `last-write-wins` (monotonic agent-card evolution).
Embedding dimension if/when added: 384 (pgvector standard across terminals ecosystem).

## Mental model: HVM / Bend / py2bend as CARL's native substrate

Do NOT frame HVM integration as "a faster runtime for reward functions."
That's optimization thinking and misses the point. The real isomorphism,
stated explicitly in Tej's BITC / DMC / IRE papers:

**CARL's GRPO rollout loop is already an Interactive Research
Environment (IRE).** The tuple `(M, I, Φ, G)` exists in v0.8.0:
- `M` — `carl_core.interaction.InteractionChain` (typed event manifold)
- `I` — `Step` + `ActionType` (lifecycle-tracked interactions)
- `Φ` — `compute_phi` / `kuramoto_R` (structural correspondences)
- `G` — `BaseGate[P: GatingPredicate]` (coherence-gated routing)

HVM (Higher-order Virtual Machine) evaluates interaction combinators
with Church-Rosser confluence + Lévy-optimal sharing. Bend is the
high-level language compiling to HVM, **parallel-by-default with
automatic sequential fallback**. py2bend admits a restricted Python
subset into that path; rejected programs fall through to sequential
Python (DMC paper's two-branch partition).

The isomorphism that matters:

| CARL concept | HVM concept |
|---|---|
| K-sample completion rollout | Parallel reduction of independent redexes |
| Per-completion reward eval | Local graph rewrite (pure, confluent) |
| argmax/softmax over rewards | amb-choice coupled to `τ = 1 − crystallization` |
| phi field over K completions | Kuramoto witness `R` over parallel branches |
| Branch-point uncertainty | **Void point** = cognitive-dissonance vector |
| `InteractionChain` | DMC "mesh-visible execution trace" |

What this means concretely: the v0.10-A integration compiles the
**rollout loop itself** to Bend/HVM, not just `compute_reward()`. The
wins are (in order of importance): deterministic reproducibility,
anticipatory variance minimization via parallel void-point exploration,
mesh-visible training traces. Speed is a side effect.

## Keep an eye on

- Preserve optional dependency boundaries.
- Preserve import-time lightness.
- Keep docs current and minimal.
- If docs disagree with code, fix the docs or explicitly note the mismatch.
- Apply the YAML frontmatter stamp to every doc you create or substantially edit.
- Do NOT copy BSL-licensed terminals-runtime methodology into MIT carl-studio.
  The admin-gate + lazy-import pattern (`admin.py`, `coherence_observer.py`)
  is the canonical integration seam — extend it, don't bypass it.
