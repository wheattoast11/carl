# Changelog

## [0.4.1] — 2026-04-18

Security + correctness hotfix on top of 0.4.0. Driven by an ultrareview pass
across the 7-commit 0.4.0 window. Closes 2 P0 security holes, 9 P1 correctness
issues, and 3 P2 tech-debt items. Adds 53 new tests (1864 → 1917).

### 🔒 Security (P0)

- **`list_files` sandbox bypass** (`chat_agent.py`): the `_tool_list` handler
  called `Path(path).glob(pattern)` directly, letting a prompt-injected model
  enumerate `/`, `/etc`, etc. Now routes through `_resolve_safe_path` and
  rejects absolute/traversal globs; every match is re-verified inside workdir.

- **`run_shell` in eval sandbox** (`eval/runner.py`): `subprocess.run(cmd,
  shell=True, ...)` on model-generated strings allowed shell metacharacters
  (`;`, `|`, `$(`, backticks, `>`, `<`) to escape the tempdir. Now hard-rejects
  metacharacters, tokenizes via `shlex.split`, runs with `shell=False`.
  Eval datasets that relied on pipelines must migrate to `execute_code`.

### 🛡 Correctness (P1)

- **Infinite denial loop**: when the permission hook denied every tool in a
  turn, the agent retried forever. `_MAX_CONSECUTIVE_ALL_DENIED=5` terminates
  with `carl.all_tools_denied` error event; counter resets on any allowed tool.

- **`asyncio.get_event_loop()` inside async fn** (`trainer.py:_watch_loop`):
  would raise `RuntimeError` on Python 3.12+. Switched to
  `asyncio.get_running_loop()`.

- **`run_analysis` environment leak** (`chat_agent.py`): child subprocess
  inherited ANTHROPIC_API_KEY, HF_TOKEN, CARL_WALLET_PASSPHRASE, etc. Now
  scrubs sensitive env vars (substring match on KEY/TOKEN/SECRET/PASSWORD/
  PASSPHRASE/AUTHORIZATION/BEARER/API_KEY) and uses `sys.executable` instead
  of bare `"python"`.

- **`_resolve_safe_path` TOCTOU** (`chat_agent.py`): was passing
  `follow_symlinks=True`, defeating the protection carl_core.safepath's
  default provides. Now `follow_symlinks=False`; legitimate symlink use
  requires explicit opt-in.

- **Session quarantine silent data loss** (`chat_agent.py` + `cli/chat.py`):
  corrupted sessions were moved to `.quarantine/` with no user-visible warning.
  Now logs a warning with the destination path, exposes
  `_last_load_quarantined` on CARLAgent, and surfaces a visible CLI message
  on resume.

- **`_knowledge` list unbounded** (`chat_agent.py`): every `ingest_source`
  appended without cap. Added `_KNOWLEDGE_MAX_CHUNKS=2000` with LRU eviction
  and configurable `max_knowledge_chunks` kwarg. One-warning-per-session policy.

- **`MemoryStore.decay_pass` races with `write`** (`carl_core/memory.py`):
  tmp-replace pattern dropped concurrent appends. Added per-instance
  `threading.RLock` guarding both `write` and `decay_pass`.

- **TinkerAdapter zombie state** (`adapters/tinker_adapter.py`): `submit()`
  persisted PENDING state then unconditionally raised "not yet implemented",
  leaking state files nobody could observe. Now raises immediately with
  `carl.adapter.tinker_not_implemented` after translation validation, no
  state written.

- **UnslothAdapter silent sys.exit(3)** (`adapters/unsloth_adapter.py`):
  entrypoint template handled only `sft`/`grpo` but allowlist accepted
  `dpo`/`kto`/`orpo`. Now validates method at translation time and raises
  `carl.adapter.method_unsupported` before subprocess spawn.

### 🧹 P2 tech debt

- **`CircuitBreaker` counted programming errors**: `AttributeError`/`TypeError`
  tripped the breaker. Added `tracked_exceptions` tuple
  (default `(Exception,)` for back-compat); callers can scope to
  `(NetworkError, TimeoutError, ...)`. x402 facilitator breaker now scoped
  to infrastructure failures only.

- **Unsloth quantization double-flag**: `load_in_4bit=True` + `load_in_8bit=True`
  passed to FastLanguageModel. Rewrote as mutually exclusive precedence chain.

- **`api_key=""` pattern** (`cli/hypothesize.py`, `cli/commit.py`): passing
  empty string blocked Anthropic SDK's env-var fallback. Now `api_key=None`
  per CLAUDE.md convention.

### 🧼 Simplifications

- **Adapter shared boilerplate** (`adapters/_common.py`): extracted
  `status_common`, `logs_common`, `cancel_common`, `require_str`. Each
  adapter's status/logs/cancel is now a one-line delegate (~80 LOC removed
  across 5 adapters).

- **`trainer._watch_loop` duplicate branches**: collapsed retryable vs
  non-retryable except blocks into a single `isinstance` dispatch (~35 → 18 LOC).

- **`trainer._save_carl_checkpoint` nested try/except**: extracted
  `_safe_capture(label, fn)` helper; five nested blocks → one-liners.

- **`constitution.py` overlay error wrap**: narrowed catch-all so precise
  inner `ConfigError` / `ValidationError` codes (`bad_yaml`, `bad_rule`)
  propagate instead of being overwritten by coarser `bad_user_overlay`.

### 📋 Docs

- `AGENTS.md` test baseline updated to current reality (1864 → 1917 post-fix).

### Test counts

- v0.4.0: 1864 passing
- v0.4.1: **1917 passing** (+53 new covering every fix above)

### Migration notes

- Eval datasets using `run_shell` with pipes/redirection/substitution must
  migrate to `execute_code` (Python). Plain commands (`ls`, `cat`, `python
  script.py`, `echo hello`) continue to work.
- Symlinks inside a chat agent workdir are now rejected by file tools by
  default.

[0.4.1]: https://github.com/wheattoast11/carl/releases/tag/carl-studio@0.4.1

## [0.4.0] — 2026-04-18

The "intelligence loop" release. carl-studio is now a proper research hub with
typed error codes, retry/backoff/circuit-breaker primitives, layered memory,
constitutional rules, a hypothesize→eval→infer→commit fractal, and adapters
that let you drive Unsloth, Axolotl, Tinker, and Atropos from the same
`carl train --backend X` surface.

### The 6-verb fractal command surface

Every workflow composes from:

- `carl chat` — the meta-loop (bare `carl` opens chat)
- `carl hypothesize "<statement>"` — translate a hypothesis to carl.yaml
- `carl train --backend <trl|unsloth|axolotl|tinker|atropos>` — run the experiment
- `carl infer --propose-hypothesis` — observe and propose the next step
- `carl commit "<learning>"` — promote working memory to constitutional memory
- `carl flow "/a /b /c"` — compose chains

### Added — `carl_core` primitive layer

- `carl_core.errors` — typed hierarchy with stable codes, auto-redacted secrets
- `carl_core.retry` — retry + exponential backoff + circuit breaker
- `carl_core.safepath` — symlink-escape-proof path sandboxing
- `carl_core.hashing` — canonical-JSON `content_hash`
- `carl_core.tier` — `Tier` enum + `FEATURE_TIERS` registry
- `carl_core.interaction` — typed `InteractionChain` with 11 `ActionType`s including `MEMORY_READ`/`MEMORY_WRITE`
- `carl_core.interaction_store` — JSONL trace persistence with flock
- `carl_core.memory` — 6-layer memory store (ICONIC/ECHOIC/SHORT/WORKING/LONG/CRYSTAL) with resonance-driven recall, decay, promotion

### Added — intelligence loop

- `carl_studio.constitution` — `Constitution.load()` merges CLAUDE.md + AGENTS.md + `~/.carl/constitution.yaml`; `compile_system_prompt(topics=...)` filters by resonance tags; `append()` persists new rules
- `carl hypothesize` — CARLAgent translates NL → carl.yaml
- `carl infer --propose-hypothesis` — reads eval report + coherence trace, proposes next experiment
- `carl commit` — writes to `~/.carl/constitution.yaml`; `--from-session <id>` extracts durable learnings from a saved session
- CARLAgent injects constitution into system prompt; recalls from WORKING/LONG memory on every turn; emits MEMORY_READ/MEMORY_WRITE steps
- Session-end auto-commit: agent proposes rules to promote when a session has ≥3 turns

### Added — backend adapter spokes

`carl_studio.adapters.{trl,unsloth,axolotl,tinker,atropos}` — each implements
the `UnifiedBackend` protocol. Honest `available()` checks. CARL gate +
coherence rewards layer on top regardless of backend.

### Added — production hardening (from earlier in the 0.4.0 cycle)

- **Runtime** (`chat_agent.py`): streaming try/finally with partial-cost persistence, session corruption quarantine + schema_version, budget pre-check with BudgetError code, per-tool timeouts (30/60/120s), tool arg JSON-schema validation, `dispatch_cli` tool whitelist-gated via OPERATIONS registry + tier
- **Training** (`trainer.py`, `rewards/*`, `callbacks.py`): `.carl_checkpoint.pt` on crash with full RNG state, `.watch()` retry via `carl_core.retry` + exp backoff + 5-consecutive-failure abort, `_clamp_reward` floor (NaN/inf→0, |x|>100 clipped), logits-shape guards with `ValidationError`, callback body exception isolation
- **Eval** (`eval/runner.py`): sandbox via `carl_core.safepath.safe_resolve` (blocks symlink escape + traversal), Phase 2' per-turn GPU tensor cleanup in try/finally, empty-results and zero-tool-call branches emit typed metrics instead of NaN
- **x402** (`x402.py`): retry with jittered exp backoff, module-level `CircuitBreaker`, wallet balance pre-check, strict header parsing, InteractionChain PAYMENT steps
- **Contract/consent** (`contract.py`, `consent.py`): verify-on-load raises `ContractBroken`, `sign()` consent gate without swallow, `ConsentManager.sync_with_profile` uses timestamp precedence, flock-serialized updates
- **camp** (`camp.py`): 24h TTL cache with 7× stale-serve window, JWT 401 → `refresh_token` exchange, tier-change signal to LocalDB
- **Credits** (`trainer.py`, `credits/*`): synchronous deduction (no try-and-ignore), `--skip-credits` escape hatch, refund on post-submission failure
- **Marketplace** (`cli/marketplace.py`): idempotency keys, local publish cache, backend 409 recovery, `--force` bypass
- **CLI**: 11→23 flow ops, `carl flow --json`, `--no-continue-on-failure`, unified `error_with_hint` formatter, `carl camp init`/`camp flow` registered, `carl hypothesize`/`carl commit` top-level
- **Wallet** (`wallet_store.py`): Fernet + PBKDF2-HMAC-SHA256 (600k iters) at `~/.carl/wallet.enc` mode 0o600, OS keyring fallback, `WalletLocked`/`WalletCorrupted` typed
- **Freshness** (`freshness.py`): `FreshnessReport` + `FreshnessIssue` with stable `carl.freshness.*` codes, 24h TTL
- **E2E UAT**: 14 new failure-path scenarios exercising every typed code
- **Property tests**: 25 Hypothesis properties (hashing, safepath, retry, coherence, frame, x402)

### Changed

- Pytest `importlib` mode + explicit `pythonpath` for `tests/` + `packages/carl-core/tests/` coexistence
- `py.typed` marker on carl-core
- Test baseline: 1103 (v0.3.0) → 1864 (v0.4.0)

### Migration notes

- `from carl_studio.primitives import X` still works via shim, but prefer `from carl_core.X import ...`.
- `carl lab chat` removed in 0.3.0; use `carl lab repl` or `carl chat`.
- First-run wizard: `carl init` walks through signup + extras + consent + project in under a minute.

### Philosophy

Inspired by the old Oppenheimer/Einstein-watching-Demis-and-Karpathy analogy:
a small crystallized constitution (the polymath elders) oversees a swarm of
session-scoped agents (the frontier researchers) whose working memory decays
but whose durable learnings promote upward. Every experiment is a hypothesis;
every hypothesis carries a predicted metric; every result can propose the
next hypothesis. The loop closes itself.

[0.4.0]: https://github.com/wheattoast11/carl/releases/tag/carl-studio@0.4.0
