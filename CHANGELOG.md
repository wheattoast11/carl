# Changelog

## [Unreleased]

## [0.7.1] — 2026-04-19

Phase-2b close-out. Hardens v0.7.0's surfaces with multi-tenant MCP
session isolation, an on-platform Prometheus scrape endpoint, structured
budget caps for the x402 rail, and a trajectory-delta CLI for two runs.

### Added

- **x402 spend caps.** `SpendTracker` enforces a daily (`CARL_X402_DAILY_CAP_USD`)
  and session (`CARL_X402_SESSION_CAP_USD`) cap synchronously before any
  network call. Breaches raise `BudgetError` with stable codes
  `carl.budget.daily_cap_exceeded` / `carl.budget.session_cap_exceeded`.
  Daily rolling window persists through `LocalDB.config`.
- **`confirm_payment` hook.** `X402Client.execute` accepts an optional
  `confirm_payment_cb` — interactive or policy-based approval that fires
  after the budget check but before the consent gate. Denials raise
  `BudgetError(code="carl.budget.confirm_denied")` without recording a
  contract witness.
- **MCP per-request session state.** `MCPServerConnection.session`
  replaces the module-level `_session: dict`. Multi-tenant deployments
  now isolate auth per connection. FastMCP `Context` DI is supported on
  authenticated tools (`authenticate`, `get_tier_status`, `run_skill`,
  `dispatch_a2a_task`, `sync_data`); when bound it is preferred over the
  module-bound connection. See `docs/mcp_multitenant.md`.
- **`carl metrics serve`.** Thin Typer wrapper around
  `prometheus_client.start_http_server`; binds `127.0.0.1:9464/metrics`
  by default. Heartbeat daemon auto-hosts when `CARL_METRICS_PORT` is
  set. Private `CollectorRegistry` avoids polluting the global default.
- **`carl run diff <a> <b> [--steps]`.** Trajectory delta between two
  training runs — phi_mean / q_hat / crystallization_count /
  contraction_holds / first-divergence-step. Renders via `CampConsole`.
- **Shared gating primitives.** `carl_studio.gating.GatingPredicate`
  Protocol + `carl.gate.*` error-code namespace. `ConsentPredicate` and
  `TierPredicate` now emit identical `GATE_CHECK` steps on the active
  `InteractionChain`.

### Changed

- Heartbeat maintenance is wrapped in `RetryPolicy(max_attempts=3)` for
  transient sqlite / IO blips — failures still emit a structured log
  step but no longer tear down the daemon loop.
- `CARL_HOME` is now honoured uniformly across `db.py`, `settings.py`,
  `wallet_store.py`, and `llm.py` via the shared
  `carl_studio.settings.carl_home()` helper. The previous "partial
  override" caveat in `docs/operations.md` has been lifted.

### Removed (dead code)

- Unused `_SPEND_SESSION_KEY` reservation in `x402.py`.
- Unused private aliases `_marshal_sdk_result`, `_arg_is_missing`,
  `_estimate_cost`, `_marshal_sdk_response` in the MCP elicitation /
  sampling modules (the public helpers remain canonical).

## [0.6.0] — 2026-04-19

The resonant-heartbeat release. Four-wave execution against a 48-ticket MECE
backlog synthesized from five parallel review teams. 2637 tests pass (+161 new).
Ships the target UX vision: bare `carl` greets with a pre-coded intro, extracts
JIT context from the first input, forks a resonant heartbeat loop over an
async sticky-note queue, and closes the loop with training-feedback proposals.
Fixes the framework's central reward/eval isomorphism: `EvalGate` now requires
coherence floor (φ ≥ SIGMA) in addition to primary metric, cascade gates on
crystallization events (not reward volume), and `PhaseAdaptiveCARLReward`
shifts weights by detected Kuramoto phase.

### WAVE-0 — Security + correctness + tech-debt (v0.5.1 hotfix bundle)

#### 🔒 Security
- `_tool_dispatch_cli` now passes `_scrubbed_subprocess_env()` to child process
  (was leaking ANTHROPIC_API_KEY / HF_TOKEN / OPENAI_API_KEY / OPENROUTER_API_KEY
  into model-invoked subprocess). Tracks REV-001.
- `CARLSettings.save()` excludes `openrouter_api_key` + `openai_api_key` from
  the YAML dump (previously persisted plaintext to `~/.carl/config.yaml`).
  Tracks REV-003.

#### 🛡 Correctness
- `EvalSandbox.execute_code` uses `sys.executable` instead of bare `"python"`,
  preventing phantom tool failures in docker / CI / multi-venv environments
  (REV-005).
- `CARLAgent._one_shot_text` accumulates `_total_cost_usd` / token counters
  on every API call. `max_budget_usd` was previously bypassable via
  `suggest_learnings()` auxiliary calls (REV-006).
- `TrainingPipeline._check_gate` returns `False` on exception (was silently
  PASSing OOM / network / missing-dataset failures, allowing potentially
  broken models to reach the Hub) (REV-007).
- `CascadeRewardManager.__init__` guards `warmup_steps = max(1, warmup_steps)`
  against `ZeroDivisionError` at `get_stage_weight` (REV-009).

#### 🧹 Tech debt
- Deleted `carl_studio.primitives` compatibility shim (slated for v0.5.0
  removal per CLAUDE.md, still present at v0.5.0 ship). Zero in-tree consumers
  verified. `import carl_studio.primitives` now raises `ModuleNotFoundError`
  (SIMP-001).
- Renamed `carl_studio.agent.CARLAgent` (FSM autonomy agent) →
  `AutonomyAgent` to resolve collision with the canonical
  `carl_studio.chat_agent.CARLAgent` used by all CLI paths. Module-level
  `__getattr__` emits `DeprecationWarning` on legacy import, removal
  scheduled v0.7 (SIMP-002).
- `.gitignore` hardened against `/fix_*.py` and `/patch_*.py` scratch scripts
  (SIMP-008).

### WAVE-1 — Target UX + coherence alignment (the vision)

#### Foundations
- `ActionType.HEARTBEAT_CYCLE` + `ActionType.STICKY_NOTE` on `InteractionChain`
  (ARC-008).
- `sticky_notes` SQLite table + `src/carl_studio/sticky.py` module with
  `StickyNote` (Pydantic v2) and `StickyQueue` supporting priority-ordered
  append/dequeue/complete/archive/get/status (ARC-004).
- `src/carl_studio/jit_context.py` — `JITContext` model, `TaskIntent` enum
  (EXPLORE/TRAIN/EVAL/STICKY/FREE), `extract()` with move-key shortcircuit
  + regex classification, `WorkFrame`-aware `frame_patch` builder (ARC-002).

#### Target UX
- `src/carl_studio/cli/intro.py` — env-baked pre-coded intro with 4 keyed
  moves `[e]xplore` `[t]rain` e`[v]aluate` `[s]ticky` + free-form. Rendered
  before first `input()` in `chat_cmd`. Zero-latency, no I/O.
  `parse_intro_selection()` accepts single-letter or full-word form. Rich
  markup escape pins UAT-052 regression (ARC-001 + SIMP-009).
- `carl queue` CLI sub-app: `add` / `list` / `status` / `clear` backed by
  `StickyQueue`. Doctor gains a "Queue" stanza reporting pending count
  (ARC-007).
- `src/carl_studio/heartbeat/` — package with `HeartbeatPhase` enum
  (INTERVIEW → EXPLORE → RESEARCH → PLAN → EXECUTE → EVALUATE → RECOMMEND
  → AWAIT), `HeartbeatLoop` daemon-thread async loop draining the sticky
  queue, `HeartbeatConnection(AsyncBaseConnection)` participating in the
  Connection registry with full FSM lifecycle. Thread-safe sqlite via
  fresh `LocalDB` per worker thread (ARC-003 + ARC-006).
- `src/carl_studio/feedback.py` — `FeedbackEngine` + `EvalBaseline` +
  `TrainingProposal` Pydantic models. `cli/training.py` gains `--from-queue`
  flag that loads a pending proposal. Proposals are persisted to
  `LocalDB.config` under stable keys (ARC-005).
- Bootstrap phase: after turn-1, `ctx.frame_patch` is applied to
  `agent._frame` via `model_copy(update=...)` (only when frame is inactive,
  preserving `--frame` overrides) (JRN-002).
- Greeting gate uses `self._turn_count <= 1` (was `len(self._messages) <= 1`),
  so resumed sessions don't re-greet (REV-004).
- `_tool_frame` invalidates `_constitution_prompt` cache so frame-adapted
  rules re-compile for the new domain (REV-010).

#### Coherence alignment (framework integrity)
- `EvalGate.check` requires both primary metric AND coherence floor
  (`phi_mean ≥ SIGMA`, `0.3 ≤ discontinuity ≤ 0.7`). Restores the reward/eval
  isomorphism the package has always claimed. Legacy construction preserved
  for backward compat; new gate active when `EvalConfig.require_coherence_gate=True`
  (default) (SEM-001).
- `CascadeRewardManager(gate_mode="crystallization")` fires on
  `sum(trace.n_crystallizations) >= N` over configurable window — a
  phase-transition signature, not a reward-volume percentile. Metric mode
  preserved (SEM-006).
- `PhaseAdaptiveCARLReward(CARLReward)` reads Kuramoto-R from `_last_traces`
  and shifts `(w_mc, w_cq, w_disc)` by detected phase: gaseous rewards
  commitment, liquid balances, crystalline rewards stability (SEM-010).

### WAVE-2 — First-run polish + CLI dedup + MCP/A2A validation

#### First-run UX
- `carl init` persists `default_chat_model` so bare `carl` just works
  post-init (JRN-004).
- `carl doctor` prints a "Next steps" guide block (gated by first-run
  marker age) (JRN-005).
- Optional sample-project scaffold in `carl init` (JRN-006).
- Post-init celebration + next-step pathways (JRN-008).
- `session_theme` persisted with agent state (move:explore/train/evaluate/
  sticky or free-form) (JRN-009).
- Optional GitHub repo / HF model context-gathering (JRN-010).

#### CLI dedup + discoverability
- `_pump_events` helper shared between `chat_cmd` and `run_one_shot_agent`
  (SIMP-003).
- `parse_flags()` replaces 12 handrolled arg loops in `operations.py`
  (-69 LOC) (SIMP-006).
- `_PROMPT_OPS` dict consolidates 6 prompt-template macro ops (SIMP-010).
- Operation descriptions appear in `carl flow --list` (JRN-007).

#### Tech-debt consolidation
- 7 error classes (`ContractError` / `ConsentError` / `MarketplaceError` /
  `SyncError` / `X402Error` / `BillingError` / `CreditError`) migrated onto
  `CARLError` hierarchy with stable codes (`carl.contract` etc.). Network
  failures now use multi-inheritance with `NetworkError`
  (`MarketplaceNetworkError`, `CreditNetworkError`). Secret-redaction via
  `to_dict()` is now active for all of them (SIMP-004).
- `x402_sdk.py` deleted; `X402SDKClient` folded into `x402_connection.py`
  (SIMP-005).
- `CARLSettings.SETTABLE_FIELDS` derived dynamically from
  `cls.model_fields`; `load()` uses `local_data.keys() & model_fields.keys()`
  instead of a hardcoded allow-list (SIMP-007).

#### DB lifecycle + MCP/A2A validation
- `LocalDB._connect` commits on clean exit, rolls back on exception,
  serializes via `threading.Lock`. `sqlite3.connect(check_same_thread=False)`
  so the lock can actually do its job (REV-002).
- `LocalDB.__init__` calls `self.close()` on `_init_schema` failure so
  corrupt-DB doesn't leak a half-open connection (REV-008).
- MCP `_session` documented + single-tenant banner on server startup;
  per-request migration path documented for v0.7 (UAT-049).
- A2A `send` CLI validates `--inputs` is a JSON object and `skill` is in
  `BUILTIN_SKILLS` registry (UAT-050).
- MCP `sync_data` logs JWT cache-write failures at DEBUG instead of silent
  swallow (UAT-051).

### WAVE-3 — Framework depth (isomorphism, contraction, multi-layer, TTT)

- `carl_core.connection.coherence.ChannelCoherence` — per-transaction
  observable (phi_mean / cloud_quality / success_rate / latency_ms) that
  any channel can publish. `channel_coherence_diff()` +
  `channel_coherence_distance()` make the 1P/3P isomorphism claim
  measurable. `BaseConnection` gains `channel_coherence()` reader +
  `publish_channel_coherence()` setter; `to_dict()` surfaces it (SEM-002).
- `Step` gains optional `phi` / `kuramoto_r` / `channel_coherence` fields.
  `InteractionChain.coherence_trajectory()` returns the phi-vs-step series
  across all channels — chain is now a witness, not just a log (SEM-007).
- `carl_core.dynamics.ContractionProbe` — records trajectory, fits
  contraction constant q_hat via log-ratio OLS, fires
  `contraction_violation` on divergence from Banach-style contraction.
  Opt-in via `CARL_CONTRACTION_PROBE=1` through `ResonanceLRCallback`;
  logs `dynamics/q_hat` (SEM-003).
- `test_conservation_law.py` — smoke test for KAPPA · SIGMA = 4 and
  ∫(1-phi_t) ≈ SIGMA · T_STAR within 10× tolerance on synthetic cooling
  trajectory (`@pytest.mark.slow`) (SEM-008).
- `CoherenceProbe.measure_multi_layer(hidden_states, logits, token_ids)`
  returns `LayeredTrace` with per-layer residual cosine + optional
  attention entropy. Gated by `CARL_LAYER_PROBE=1`; fast logits-only path
  preserved (SEM-004).
- `GRPOReflectionCallback.on_log` computes `tau` from CARL reward's
  `_last_traces` via Kuramoto-R in the public training path. Publishes
  `witness/tau` + `witness/kuramoto_R` to the logs stream. Previously the
  crystalline gate was permanently closed for public users; TTT
  micro-update now fires on real phase transitions (SEM-009).

### Removed

- `carl_studio.primitives` compatibility shim (see WAVE-0 above).
- `carl_studio.x402_sdk` module — consolidated into `x402_connection`
  (see WAVE-2 above).

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

- `from carl_studio.primitives import X` still works via shim in 0.4.x, but prefer `from carl_core.X import ...`. (The shim is removed in the next release — see the Unreleased section.)
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
